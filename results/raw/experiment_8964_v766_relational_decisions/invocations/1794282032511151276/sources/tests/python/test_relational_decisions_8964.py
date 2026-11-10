"""REQ-REPORT-8964 / SCENARIO-REPORT-8964-CONTRACT: fresh decisions stay scoped."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import relational_decisions_8964 as d
from carnot.reporting import relational_decisions_runner_8964 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8964: authenticate original producers before scoring cached answers."""
    value = d.authenticate(d.ROOT)
    assert not value["failures"]
    return value


@pytest.fixture(scope="module")
def evaluated(evidence, tmp_path_factory):
    """REQ-REPORT-8964: retain one real frozen evaluation for assertion reuse."""
    raw = tmp_path_factory.mktemp("8964-evaluation")
    return raw, d.evaluate(evidence, raw)


def test_seal_before_labels_and_no_fitting(evidence, tmp_path, monkeypatch):
    """REQ-REPORT-8964: the reader sees every arm, seed, decision and state first."""
    release = d.release_final_test

    def reader(labels, seal):
        saved = json.loads(seal.read_text())
        assert len(saved["rows"]) == 128
        assert set(saved["rows"][0]["probabilities"]) == set(d.methods.ARMS)
        assert set(saved["rows"][0]["decisions"]) == set(d.methods.ARMS)
        service = json.loads((tmp_path / "calibrated_states.json").read_text())
        assert saved["service_hash"] == canonical_hash(service)
        assert json.loads((tmp_path / "fixed_thresholds.json").read_text()) == d.THRESHOLDS
        for arm, _, _ in d.methods.TRAINED_ARMS:
            assert [s["seed"] for s in service["states"][arm]] == [71, 72, 73]
        return release(labels, seal)

    def forbidden(*args, **kwargs):
        raise AssertionError("training/traversal is forbidden")

    monkeypatch.setattr(d, "release_final_test", reader)
    monkeypatch.setattr(d.g, "fit", forbidden)
    monkeypatch.setattr(d.methods.source, "distance_nodes", forbidden)
    result = d.evaluate(evidence, tmp_path)
    assert result["completed_count"] == result["independent_count"] == 128
    assert result["oracle_access_controls"]["state_and_predictions_sealed_before_truth"]
    assert result["numerical_training_calls"] == 0
    assert len(result["test_prediction_rows"]) == 128
    assert len(result["rows"]) == len(result["typed_decision_rows"]) == 640


def test_metrics_and_frozen_service(evaluated):
    """REQ-REPORT-8964: headline values reduce from independent graph rows."""
    raw, result = evaluated
    service = json.loads((raw / "calibrated_states.json").read_text())
    assert set(service["temperatures"]) == set(d.methods.ARMS)
    assert set(service["temperatures"].values()) <= set(d.b.GRID)
    assert len(result["paired_intervals"]["contrasts"]) == 5
    assert result["paired_intervals"]["draws"] == 10000
    assert result["paired_intervals"]["family_alpha"] == 0.05
    for arm in d.methods.ARMS:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        assert len({r["cluster_id"] for r in rows}) == 128
        assert sum(r["selected_50"] for r in rows) == 64
        assert all(r["value"] == (r["p_error"] - r["error"]) ** 2 for r in rows)
        assert all(r["log_loss"] >= 0 for r in rows)
    view = json.loads(Path(result["inference_test_reference"]["path"]).read_text())["rows"]
    scored = d.score(service, view)
    np.testing.assert_allclose(
        d.b._sigmoid(scored["relational_logits"]), scored["probabilities"]["relational_energy"]
    )
    with pytest.raises(ValueError, match="detector_schema"):
        d.score(service, [dict(view[0], label=1)])


def test_all_five_bounds_required():
    """REQ-REPORT-8964: a matched-logistic tie defeats otherwise favorable evidence."""
    ids = [f"graph-{i:03d}" for i in range(128)]
    truth = [i % 2 for i in range(128)]
    prob = {arm: [0.5] * 128 for arm in d.methods.ARMS}
    prob["relational_energy"] = [0.99 if y else 0.01 for y in truth]
    result = d.reduce(prob, truth, ids)
    assert result["paired_intervals"]["benefit_passed"]
    prob["relational_logistic"] = prob["relational_energy"]
    result = d.reduce(prob, truth, ids)
    assert not result["paired_intervals"]["benefit_passed"]
    contrasts = result["paired_intervals"]["contrasts"]
    assert sum(r["passed"] for r in contrasts.values()) == 4
    assert contrasts["brier_vs_relational_logistic"]["one_sided_bound"] == 0


@pytest.mark.parametrize(
    "fault", ["missing", "hash", "json", "object", "field", "zero", "boolean", "raw"]
)
def test_authentication_failures(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8964-CONTRACT: external failures name exact fields and values."""
    path = tmp_path / "producer.json"
    value = dict(
        task_id="upstream",
        score=1,
        required_checks_passed=True,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    if fault == "field":
        del value["score"]
    elif fault in ("zero", "boolean"):
        value["score"] = 0 if fault == "zero" else True
    elif fault == "raw":
        value["raw_shard_hashes"] = [dict(path=str(tmp_path / "missing"), sha256="pin")]
    if fault != "missing":
        path.write_text(
            "bad" if fault == "json" else "[]" if fault == "object" else json.dumps(value)
        )
    pin = "wrong" if fault in ("hash", "missing") else d.calibration.reference(path)["sha256"]
    monkeypatch.setattr(d, "INPUTS", [(path.name, pin, "score", "upstream")])
    failure = d.authenticate(tmp_path)["failures"][0]
    assert {
        "upstream",
        "path",
        "hash",
        "check",
        "artifact_field",
        "expected",
        "observed",
    } <= failure.keys()
    assert failure["failure_kind"] == {
        "missing": "missing_file",
        "raw": "missing_file",
        "field": "missing_field",
        "zero": "observed_zero",
    }.get(fault, "rejected_producer")


def test_private_cli_publication_replay_tamper(tmp_path):
    """SCENARIO-REPORT-8964-CONTRACT: actual child CLI works outside the checkout."""
    output = tmp_path / "results" / (d.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def child(*args):
        result = subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(result.stdout + result.stderr, flush=True)
        return result.returncode

    assert child("--date", "bad") == 2
    assert child("--date", "20261110", "--fixture-e2e", str(d.ROOT / "invalid.json")) == 2
    assert child("--date", "20261110", "--fixture-e2e", str(output)) == 0
    assert child("--cold-replay", str(output)) == 0
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["detector_rows_ready_score"] == 0
    value["completed_count"] = 128
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert child("--cold-replay", str(output)) == 1
    assert not runner.replay(tmp_path / "missing")["passed"]


@pytest.mark.parametrize("fault", ["protocol", "seed", "view", "overlap", "calibration"])
def test_frozen_operands_refuse_drift(evidence, tmp_path, fault):
    """SCENARIO-REPORT-8964-CONTRACT: altered frozen roles or states cannot be reused."""
    value = deepcopy(evidence)
    method, corpus = value["producers"]
    if fault == "protocol":
        method["frozen_protocol"]["steps"] = 1
    elif fault == "seed":
        method["trained_state_hashes"][0]["sha256"] = "changed"
    else:
        view = d.read_reference(corpus["inference_views"]["calibration"])
        if fault == "view":
            view["unit_ids"][0] = "wrong"
            target = corpus["inference_views"]
            role = "calibration"
        elif fault == "overlap":
            units = [r for r in corpus["corpus_manifest"] if r["role"] == "test"]
            for row, unit in zip(units, view["unit_ids"], strict=True):
                row["unit_id"] = unit
            target = corpus["inference_views"]
            role = "test"
        else:
            view = dict(rows=[dict(unit_id="wrong", label=0)])
            target = corpus["broker_manifest"]["calibration"]
            role = "label_reference"
        path = tmp_path / "changed.json"
        atomic_json(path, view)
        target[role] = d.calibration.reference(path)
    with pytest.raises(ValueError):
        d.evaluate(value, tmp_path / "evaluation")


def test_invalid_rows_and_seals(evaluated, evidence, tmp_path):
    """REQ-REPORT-8964: invalid probabilities and post-seal changes are refused."""
    raw, result = evaluated
    with pytest.raises(ValueError, match="test_roster"):
        d.reduce({}, [0, 1], ["a", "b"])
    with pytest.raises(ValueError):
        d.reduce(
            {a: [float("nan")] * 128 for a in d.methods.ARMS},
            [i % 2 for i in range(128)],
            [str(i) for i in range(128)],
        )
    changed = json.loads((raw / "test_predictions.json").read_text())
    changed["rows"][0]["probabilities"]["prevalence"] = 0.123
    atomic_json(raw / "test_predictions.json", changed)
    with pytest.raises(ValueError, match="sealed_operand_drift"):
        d.evaluate(evidence, raw, replay=True)
    # Restore the original seal for the other assertions sharing this private run.
    predictions = result["test_prediction_rows"]
    service = json.loads((raw / "calibrated_states.json").read_text())
    atomic_json(
        raw / "test_predictions.json",
        dict(
            rows=predictions,
            prediction_hash=canonical_hash(predictions),
            service_hash=canonical_hash(service),
            thresholds=d.THRESHOLDS,
        ),
    )
    with pytest.raises(ValueError):
        d.read_reference(dict(path=str(tmp_path / "absent"), sha256="expected"))


def test_readiness_benefit_and_terminal_class(evaluated, tmp_path, monkeypatch):
    """REQ-REPORT-8964: a valid null still exports the qualified service for calibration."""
    _, result = evaluated
    work = dict(
        raw=str(tmp_path),
        result=deepcopy(result),
        evidence=dict(failures=[], references=[]),
        sources=[],
        raw_references=[],
        global_test_debt={},
        started=1.0,
        ended=2.0,
        phase_spans=[],
        prior_terminal={},
        owned_failure=False,
    )
    validation = dict(receipts=[], coverage={})
    monkeypatch.setattr(d.admission.shard, "qualified", lambda *args: True)
    work["result"]["paired_intervals"]["benefit_passed"] = False
    value = d.build(work, validation, False)
    assert value["detector_rows_ready_score"] == 1 and value["detector_benefit_score"] == 0
    assert value["verdict_class"] == "null" and value["frozen_logit_service"]["qualified"]
    assert not value["verifier_is_oracle"] and value["independent_generalization_score"] == 0
    assert value["exposure_scope"] == "fresh_matched_graph_final_answers"
    intervals = work["result"]["paired_intervals"]
    intervals["benefit_passed"] = True
    for row in intervals["contrasts"].values():
        row["passed"] = True
    assert d.build(work, validation, False)["verdict_class"] == "positive"
    work["result"]["oracle_access_controls"]["no_inference_label_access"] = False
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    work["result"] = d.empty_result()
    work["evidence"]["failures"] = [dict(check="external")]
    work["prior_terminal"] = dict(
        verdict_class="blocked",
        honest_verdict="complete_blocked_prior",
        gate_check_summary=work["evidence"]["failures"],
    )
    assert d.build(work, validation, False)["honest_verdict"] == "complete_blocked_prior"
    validation["terminal_rejected"] = True
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"


@pytest.mark.parametrize("kind", ["external", "missing", "owned"])
def test_evaluation_failure_disposition(tmp_path, monkeypatch, kind):
    """REQ-REPORT-8964: external absence blocks; failed owned controls disqualify."""
    evidence = dict(failures=[])

    def fail(*args, **kwargs):
        if kind == "external":
            raise d.Rejected(dict(check="retained_bytes", artifact_field="sha256"))
        if kind == "missing":
            raise OSError("absent")
        raise ValueError("owned control failed")

    monkeypatch.setattr(d, "evaluate", fail)
    result, owned = runner.evaluation(evidence, tmp_path)
    assert result["censored_count"] == 128 and len(result["rows"]) == 640
    assert owned == (kind == "owned") and evidence["failures"]


def test_successful_freeze_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8964-CONTRACT: private replay recomputes real frozen predictions."""
    raw = tmp_path / "raw"
    work = runner.freeze(d.ROOT, raw, False)
    assert not work["evidence"]["failures"] and not work["owned_failure"]
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.calibration.reference(raw / name)
        for name in (
            "validation.json",
            "calibrated_states.json",
            "fixed_thresholds.json",
            "test_predictions.json",
        )
    ]
    atomic_json(raw / "work.json", work)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, d.build(work, validation, True))
    assert runner.replay(output)["passed"]
    original = deepcopy(work["result"])
    work["result"]["rows"][0]["value"] += 0.1
    atomic_json(raw / "work.json", work)
    atomic_json(output, d.build(work, validation, True))
    assert not runner.replay(output)["passed"]
    work["result"] = original
    atomic_json(raw / "work.json", work)
    value = d.build(work, validation, True)
    value["raw_shard_hashes"][0]["sha256"] = "changed"
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    value = deepcopy(d.build(work, validation, True))
    value["code_config_hashes"][0]["source_path"] = str(tmp_path / "unexecuted")
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    seal = raw / "test_predictions.json"
    changed = json.loads(seal.read_text())
    changed["rows"][0]["probabilities"]["prevalence"] = 0.123
    changed["prediction_hash"] = canonical_hash(changed["rows"])
    atomic_json(seal, changed)
    work["raw_references"][-1] = d.calibration.reference(seal)
    atomic_json(raw / "work.json", work)
    atomic_json(output, d.build(work, validation, True))
    assert not runner.replay(output)["passed"]


def test_main_owned_checks_and_terminal_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8964: orchestration failure clears readiness without rerunning science."""
    original = runner.freeze(d.ROOT, tmp_path / "initial", True)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", lambda *args: deepcopy(original))
    monkeypatch.setattr(
        runner,
        "validation_plan",
        lambda *args, **kwargs: [
            runner.CommandSpec("owned_unit_child", ("private",), "test"),
            runner.CommandSpec("other", ("private",), "test"),
        ],
    )
    private = tmp_path / "private"
    private.mkdir()
    (private / ".coverage").write_text("private orchestration receipt")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(private))
    monkeypatch.setattr(runner, "measured", lambda *args: dict(complete=True))
    monkeypatch.setattr(runner, "run_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "replay", lambda *args: dict(passed=True))
    monkeypatch.setattr(
        d.calibration,
        "terminal_commands",
        lambda path: [runner.CommandSpec("terminal", (d.calibration.CLI,), "test")],
    )
    calls = []

    def publish(output, value, validator):
        calls.append(value)
        assert validator(output)["passed"]
        if len(calls) == 1:
            raise ValueError("candidate_rejected")
        return dict(primary_path=str(output))

    monkeypatch.setattr(runner, "publish_primary", publish)
    assert runner.main(["--date", "20261110"]) == 0
    assert calls[-1]["verdict_class"] == "disqualified"
    assert calls[-1]["detector_rows_ready_score"] == calls[-1]["detector_benefit_score"] == 0
    output = tmp_path / "results" / (d.NAME + ".json")
    atomic_json(output, calls[-1])
    assert runner.main(["--date", "20261110"]) == 0
    monkeypatch.setattr(
        runner,
        "publish_primary",
        lambda *args: (_ for _ in ()).throw(ValueError("conflicting_primary")),
    )
    with pytest.raises(ValueError, match="conflicting_primary"):
        runner.main(["--date", "20261110"])
