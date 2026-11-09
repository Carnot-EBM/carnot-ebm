"""REQ-REPORT-8952; SCENARIO-REPORT-8952-CONTRACT: sealed unseen decisions."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import matched_calibrated_decisions_8952 as d
from carnot.reporting import matched_decisions_runner_8952 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def corpus():
    """REQ-REPORT-8952: preserve the admitted population and original role assignments."""
    assert not d.authenticate(d.ROOT)["failures"]
    return json.loads((d.ROOT / d.INPUTS[1][0]).read_text())["rows"]


@pytest.fixture(scope="module")
def measured_result(corpus, tmp_path_factory):
    """REQ-REPORT-8952: run the frozen numerical budget once for these assertions."""
    raw = tmp_path_factory.mktemp("sealed-decisions")
    return d.evaluate(corpus, raw), raw


def test_frozen_fit_and_three_bounds(measured_result):
    """REQ-REPORT-8952: all four arms retain80 independent test graphs and40 selections."""
    result, raw = measured_result
    assert len(result["rows"]) == 320
    assert result["completed_count"] == result["independent_count"] == 80
    assert result["numerical_training_calls"] == 600
    assert result["paired_intervals"]["draws"] == 10000
    assert result["paired_intervals"]["alpha_per_contrast"] == 0.05 / 3
    assert set(result["model_state_hashes"]) == {
        "energy",
        "logistic",
        "nuisance",
        "prevalence",
        "complete",
    }
    state = json.loads((raw / "model_state.json").read_text())
    assert canonical_hash(state) == result["model_state_hashes"]["complete"]
    assert state["numerical_training_calls"] == 600
    for arm in d.b.ARMS:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        assert len(rows) == 80 and sum(r["selected_50"] for r in rows) == 40
        assert all(r["value"] == (r["p_error"] - r["error"]) ** 2 for r in rows)
        assert all(r["correct"] == 1 - r["error"] and not r["fixture_only"] for r in rows)
        assert all(r["decision"] in {"accept", "reject", "escalate"} for r in rows)
    assert all(r["passed"] for r in result["oracle_access_controls"]["control_rows"])
    assert {r["metric"] for r in result["descriptive_rows"]} >= {"auroc", "auprc", "brier"}


def test_test_truth_opens_after_state_and_predictions(corpus, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8952-CONTRACT: test labels cannot be consulted before sealing."""

    class SealedTruth(dict):
        def __getitem__(self, key):
            if key == "value":
                assert (tmp_path / "model_state.json").is_file()
                assert (tmp_path / "test_probabilities.json").is_file()
            return super().__getitem__(key)

    guarded = [SealedTruth(r) if r["role"] == "test" else r for r in corpus]
    monkeypatch.setattr(d.b, "summarize", lambda _: dict(benefit_passed=False))
    d.evaluate(guarded, tmp_path)


def test_missing_inputs_and_empty_measurement(tmp_path):
    """REQ-REPORT-8952: unavailable evidence retains all scheduled comparisons."""
    evidence = d.authenticate(tmp_path)
    assert evidence["failures"][0]["check"] == "missing_file"
    empty = d.empty_result()
    assert empty["censored_count"] == 80 and len(empty["rows"]) == 320
    assert all(r["status"] == "censored" and r["value"] is None for r in empty["rows"])


@pytest.mark.parametrize("fault", ["role", "duplicate", "status", "feature"])
def test_reject_changed_roster_before_training(corpus, tmp_path, fault):
    """REQ-REPORT-8952: malformed admitted operands cannot become model inputs."""
    changed = deepcopy(corpus)
    if fault == "role":
        changed[0]["role"] = "test"
    elif fault == "duplicate":
        changed[1]["cluster_id"] = changed[0]["cluster_id"]
    elif fault == "status":
        changed[0]["status"] = "censored"
    else:
        changed[0]["features"]["label"] = 1
    with pytest.raises(ValueError):
        d.evaluate(changed, tmp_path)
    assert not (tmp_path / "model_state.json").exists()


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8952-CONTRACT: fresh processes reject rehashed aggregate drift."""
    output = tmp_path / "results" / (d.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def child(*args):
        return subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert child("--date", "bad").returncode == 2
    result = child("--date", "20261109", "--fixture-e2e", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["detector_benefit_score"] == 0
    assert child("--cold-replay", str(output)).returncode == 0
    value["detector_rows_ready_score"] = 1
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert child("--cold-replay", str(output)).returncode == 1


@pytest.mark.parametrize("fault", ["malformed", "missing", "zero", "boolean", "raw"])
def test_authentication_rejects_changed_producers(tmp_path, monkeypatch, fault):
    """REQ-REPORT-8952: explicit field and hash failures retain exact upstream evidence."""
    path = tmp_path / "producer.json"
    operand = tmp_path / "operand.json"
    operand.write_text("original")
    value = dict(
        task_id="upstream",
        required_checks_passed=True,
        score=1,
        calibration_protocol_hash=canonical_hash(d.calibration.manifest()),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    if fault == "missing":
        del value["score"]
    elif fault == "zero":
        value["score"] = 0
    elif fault == "boolean":
        value["score"] = True
    elif fault == "raw":
        value["raw_shard_hashes"] = [d.calibration.reference(operand)]
        operand.write_text("changed")
    path.write_text("bad json" if fault == "malformed" else json.dumps(value))
    monkeypatch.setattr(
        d, "INPUTS", [(path.name, d.calibration.reference(path)["sha256"], "score", "upstream")]
    )
    monkeypatch.setattr(
        d.admission,
        "authenticate",
        lambda _: dict(failures=[], references=[], authority_mode="private"),
    )
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["check"] == {"missing": "missing_field", "zero": "observed_zero"}.get(
        fault, "rejected_producer"
    )
    assert failure["upstream"] == "upstream"


@pytest.fixture(scope="module")
def frozen_work(tmp_path_factory):
    """REQ-REPORT-8952: exercise the real nonfixture producer entirely in private scratch."""
    raw = tmp_path_factory.mktemp("8952-real-private")
    work = runner.freeze(d.ROOT, raw, False)
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", dict(receipts=[], coverage={}))
    return work


def test_complete_replay_and_verdicts(frozen_work, monkeypatch, tmp_path):
    """REQ-REPORT-8952: measured rows can qualify separately from scientific benefit."""
    work = deepcopy(frozen_work)
    validation = dict(receipts=[], coverage={})
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d.shard, "qualified", lambda *_: True)
    value = d.build(work, validation, False)
    assert value["detector_rows_ready_score"] == 1
    assert value["detector_benefit_score"] == int(
        work["result"]["paired_intervals"]["benefit_passed"]
    )
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, value)
    assert runner.replay(output)["passed"]
    original = deepcopy(work["result"]["paired_intervals"]["contrasts"])
    for row in work["result"]["paired_intervals"]["contrasts"].values():
        row["passed"] = True
    assert d.build(work, validation, False)["verdict_class"] == "positive"
    work["result"]["paired_intervals"]["contrasts"] = original
    assert d.build(work, validation, False)["verdict_class"] == "null"
    work["development"]["control_rows"][0]["passed"] = False
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"


@pytest.mark.parametrize("fault", ["hash", "code", "authentication", "result", "development"])
def test_replay_refusals(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8952-CONTRACT: retained bytes and reconstructed summaries must agree."""
    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    raw = Path(work["raw"])
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = d.build(work, validation, True)
    if fault == "hash":
        (raw / "development.json").write_text("changed")
    elif fault == "code":
        value["code_config_hashes"][0]["source_path"] = str(tmp_path / "missing")
    elif fault == "authentication":
        monkeypatch.setattr(d, "authenticate", lambda _: {})
    elif fault == "result":
        monkeypatch.setattr(d, "empty_result", lambda: {})
    else:
        atomic_json(raw / "development.json", dict(control_rows=[]))
        value["raw_shard_hashes"] = [
            d.calibration.reference(Path(r["path"])) for r in value["raw_shard_hashes"]
        ]
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    assert not runner.replay(tmp_path / "missing")["passed"]


def test_sealed_state_refuses_rehashed_tamper(measured_result, tmp_path):
    """REQ-REPORT-8952: replay never replaces a changed frozen state."""
    result, raw = measured_result
    state = json.loads((raw / "model_state.json").read_text())
    with pytest.raises(ValueError, match="sealed_operand_drift"):
        d.seal(raw / "model_state.json", dict(state, prevalence=0.123), True)
    assert canonical_hash(state) == result["model_state_hashes"]["complete"]


def test_nonfixture_validation_pipeline(tmp_path, monkeypatch):
    """REQ-REPORT-8952: qualification precedes publication; global debt stays separate."""
    work = runner.freeze(d.ROOT, tmp_path / "private-work", True)
    private = tmp_path / "validation"
    private.mkdir()
    (private / ".coverage").touch()
    monkeypatch.setattr(runner, "freeze", lambda *_: deepcopy(work))
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(runner, "measured", lambda *_: {})
    monkeypatch.setattr(runner.shared, "validation_plan", lambda *_args, **_kwargs: [])
    calls = []

    def commands(_root, commands, **_kwargs):
        calls.append([c.name for c in commands])
        return [dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in commands]

    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    monkeypatch.setattr(d, "ROOT", tmp_path)
    assert runner.main(["--date", "20261109"]) == 0
    assert calls == [[], ["full_python_suite"], ["cold_replay_cli", "adversarial", "strict_rows"]]
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["verdict_class"] == "disqualified"
    assert not value["global_test_debt"]["affects_owned_readiness"]
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261109", "--fixture-e2e", str(tmp_path / "unsafe.json")])
