"""REQ-REPORT-8951; SCENARIO-REPORT-8951-CONTRACT: certify custody before training."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from carnot.reporting import matched_corpus_admission_8951 as d
from carnot.reporting import matched_admission_runner_8951 as runner
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def corpus():
    """REQ-REPORT-8951: original first responses supply an independent graph corpus."""
    evidence = d.authenticate(d.ROOT)
    assert evidence["failures"] == []
    units, rows, old = d.join(d.ROOT)
    return units, rows, old


def test_exact_corpus_and_support(corpus):
    """REQ-REPORT-8951: preserve 320 identities and the original 200-error floor."""
    result = d.reduce(*corpus)
    assert result["role_counts"] == dict(train=160, calibration=80, test=80)
    assert result["independent_error_count"] == 233
    assert result["minimum_class_support_achieved"] == 22
    assert result["class_support_passed"]
    assert result["error_type_counts"] == dict(
        correct=87, valid_wrong_distance=227, malformed=0, out_of_graph=6
    )
    assert result["semantic_error_count"] == 227
    assert result["format_error_count"] == 0
    assert len(result["corpus_manifest"]) == 320
    assert all(r["denominator"] == 1 and r["condition"] == "k2" for r in result["rows"])


@pytest.mark.parametrize(
    "fault", ["duplicate", "role", "pilot", "expected", "opportunity", "label"]
)
def test_reject_corpus_mutations(corpus, fault):
    """SCENARIO-REPORT-8951-CONTRACT: rehashed identities and truth cannot pass admission."""
    units, rows, old = deepcopy(corpus)
    if fault == "duplicate":
        units[1] = deepcopy(units[0])
    elif fault == "role":
        units[0]["role"] = "test" if units[0]["role"] == "train" else "train"
    elif fault == "pilot":
        old.append(units[0])
    elif fault == "expected":
        units[0]["expected"] = []
    elif fault == "opportunity":
        units[0]["valid_sets"]["3"] = []
    else:
        rows[0]["value"] = 1 - rows[0]["value"]
    with pytest.raises(ValueError):
        d.reduce(units, rows, old)


def test_missing_class_is_null_without_rebalancing(corpus):
    """REQ-REPORT-8951: a complete all-correct corpus fails support without changing roles."""
    units, rows, old = deepcopy(corpus)
    for unit, row in zip(units, rows, strict=True):
        row.update(answer=unit["expected"][0], parser_category="correct", value=0, numerator=0)
    result = d.reduce(units, rows, old)
    assert not result["class_support_passed"] and result["independent_error_count"] == 0
    assert result["minimum_class_support_achieved"] == 0
    assert result["role_counts"] == dict(train=160, calibration=80, test=80)
    with pytest.raises(ValueError):
        d.reduce(units, rows[:-1], old)


def test_missing_inputs_and_empty_schedule(tmp_path):
    """REQ-REPORT-8951: external absence blocks with exact fields and keeps every slot."""
    evidence = d.authenticate(tmp_path)
    assert evidence["failures"][0]["failure_kind"] == "missing_file"
    assert evidence["failures"][0]["artifact_field"] == "sha256"
    result = d.reduce([], [], [])
    assert result["completed_count"] == 0 and result["censored_count"] == 320
    assert len(result["rows"]) == 320


@pytest.fixture(scope="module")
def development():
    """REQ-REPORT-8951: one measured development run supplies all isolation controls."""
    return d.development()


def test_development_and_mutation_isolation(development):
    """REQ-REPORT-8951: only development fixtures fit; oracle mutations leave scores fixed."""
    result = development
    assert result["numerical_training_calls"] == 1200
    assert all(r["passed"] for r in result["control_rows"])
    controls = result["feature_access_controls"]
    assert controls["import_isolation_passed"] and controls["mutation_isolation_passed"]
    assert {r["field"] for r in controls["mutation_rows"]} == {
        "label",
        "true_distances",
        "answer_set",
        "graph_id",
    }
    assert len(result["rows"]) == 160


@pytest.mark.parametrize("statement", ["import carnot.verify", "from carnot.verify import oracle"])
def test_detector_import_refusal(development, monkeypatch, statement):
    """REQ-REPORT-8951: either Python import form must refuse graph-oracle access."""
    monkeypatch.setattr(d.calibration, "development", lambda: deepcopy(development))
    monkeypatch.setattr(Path, "read_text", lambda *_: statement)
    result = d.development()
    assert not result["feature_access_controls"]["import_isolation_passed"]
    assert not all(r["passed"] for r in result["control_rows"])


def test_private_cli_cold_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8951-CONTRACT: cold children can only publish private fixtures."""
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
    assert value["fixture_only"] and value["corpus_ready_score"] == 0
    assert child("--cold-replay", str(output)).returncode == 0
    value["independent_error_count"] = 320
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert child("--cold-replay", str(output)).returncode == 1
    assert not runner.replay(tmp_path / "missing")["passed"]


@pytest.mark.parametrize("fault", ["malformed", "missing", "zero", "boolean", "raw", "code"])
def test_authentication_refuses_exact_failure(tmp_path, monkeypatch, fault):
    """REQ-REPORT-8951: distinguish missing fields, observed zeros and rejected bytes."""
    path = tmp_path / "producer.json"
    operand = tmp_path / "operand"
    operand.write_text("original")
    ref = d.calibration.reference(operand)
    value = dict(
        task_id="upstream",
        score=1,
        required_checks_passed=True,
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
        value["raw_shard_hashes"] = [ref]
        operand.write_text("changed")
    elif fault == "code":
        value["code_config_hashes"] = [dict(ref, source_path=str(d.calibration.b.__file__))]
    path.write_text("bad JSON" if fault == "malformed" else json.dumps(value))
    monkeypatch.setattr(
        d, "INPUTS", [(path.name, d.calibration.reference(path)["sha256"], "score", "upstream")]
    )
    monkeypatch.setattr(
        d.matched.branch, "authority", lambda _: dict(failures=[], comparison_mode="private")
    )
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["failure_kind"] == {"missing": "missing_field", "zero": "observed_zero"}.get(
        fault, "rejected_producer"
    )
    assert failure["artifact_field"] == {
        "raw": "sha256",
        "code": "executed_helper_bytes",
        "malformed": "producer_json",
    }.get(fault, "score")


def test_frozen_rejection_and_raw_custody(corpus, monkeypatch):
    """REQ-REPORT-8951: do not accept changed split rules or unauthenticated raw answers."""
    with monkeypatch.context() as patch:
        patch.setattr(d.matched, "validate_protocol", lambda *_: False)
        with pytest.raises(ValueError, match="frozen_split_or_rejection_drift"):
            d.join(d.ROOT)
    monkeypatch.setattr(d.shard_a, "raw_complete", lambda _: False)
    with pytest.raises(d.matched.audit.Rejected) as error:
        d.join(d.ROOT)
    assert error.value.receipt["artifact_field"] == "raw_first_responses"


def test_build_support_and_owned_verdicts(corpus, tmp_path, monkeypatch):
    """REQ-REPORT-8951: owned failures disqualify; complete unsupported corpora are null."""
    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    work["result"] = d.reduce(*corpus)
    work["development"] = dict(control_rows=[dict(passed=True)], rows=[])
    validation = dict(receipts=[], coverage={})
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d.shard_a, "qualified", lambda *_: True)
    value = d.build(work, validation, False)
    assert value["corpus_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    work["result"]["class_support_passed"] = False
    assert d.build(work, validation, False)["verdict_class"] == "null"
    work["evidence"]["failures"] = [dict(check="external")]
    assert d.build(work, validation, False)["verdict_class"] == "blocked"
    work["development"]["control_rows"][0]["passed"] = False
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"


@pytest.mark.parametrize("fault", [None, "rejected", "malformed"])
def test_freeze_current_and_rejected_corpus(corpus, tmp_path, monkeypatch, fault):
    """REQ-REPORT-8951: private tests preserve complete work and exact external failures."""
    dev = dict(control_rows=[dict(passed=True)], rows=[])
    monkeypatch.setattr(d, "development", lambda: dev)
    if fault:
        error = (
            d.matched.audit.Rejected(dict(check="private_rejection"))
            if fault == "rejected"
            else ValueError("private_malformed")
        )
        monkeypatch.setattr(d, "join", Mock(side_effect=error))
    else:
        monkeypatch.setattr(d, "join", lambda _: corpus)
    work = runner.freeze(d.ROOT, tmp_path / "raw", False)
    assert work["result"]["completed_count"] == (0 if fault else 320)
    assert bool(work["evidence"]["failures"]) == bool(fault)
    if not fault:
        assert len(work["historical_scheduler_skips"]) == 4
        assert all(
            r["primary_reference"]["original_path"] for r in work["historical_scheduler_skips"]
        )
        validation = dict(receipts=[], coverage={})
        atomic_json(Path(work["raw"]) / "work.json", work)
        atomic_json(Path(work["raw"]) / "validation.json", validation)
        output = tmp_path / (d.NAME + ".json")
        atomic_json(output, d.build(work, validation, False))
        assert runner.replay(output)["passed"]


@pytest.mark.parametrize(
    "fault", ["raw_hash", "executed_code", "authentication", "reduction", "development", "sidecar"]
)
def test_replay_semantic_and_byte_refusal(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8951-CONTRACT: rehashed sidecars and summaries still require reconstruction."""
    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    raw = Path(work["raw"])
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = d.build(work, validation, True)
    output = tmp_path / (d.NAME + ".json")
    if fault == "raw_hash":
        (raw / "detector_view.json").write_text("changed bytes")
    elif fault == "executed_code":
        value["code_config_hashes"][0]["source_path"] = str(tmp_path / "missing_code")
    elif fault == "authentication":
        monkeypatch.setattr(d, "authenticate", lambda _: dict(failures=[dict(check="changed")]))
    elif fault in ("reduction", "development"):
        if fault == "reduction":
            work["result"]["independent_error_count"] = 1
        else:
            work["development"] = dict(control_rows=[dict(passed=True)])
            monkeypatch.setattr(d, "development", lambda: {})
        atomic_json(raw / "work.json", work)
        value["raw_shard_hashes"][0] = d.calibration.reference(raw / "work.json")
    else:
        atomic_json(raw / "detector_view.json", dict(rows=[dict(label=1)]))
        value["raw_shard_hashes"][3] = d.calibration.reference(raw / "detector_view.json")
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]


@pytest.mark.parametrize("rejection", [None, "candidate_rejected", "conflicting_primary"])
def test_main_owned_checks_and_terminal_failure(tmp_path, monkeypatch, rejection):
    """REQ-REPORT-8951: orchestration cannot publish readiness after a failed owned check."""
    root, original_freeze = d.ROOT, runner.freeze

    def freeze(_root, raw, _fixture):
        with monkeypatch.context() as patch:
            patch.setattr(d, "ROOT", root)
            return original_freeze(root, raw, True)

    def plan(private, **_):
        (private / ".coverage").write_bytes(b"private measured callback fixture")
        return []

    def commands(_root, planned, **_):
        return [dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in planned]

    def publish(output, value, validator):
        report = validator(output)
        assert report["passed"]
        if rejection and not value["flagged_adversarial"]:
            raise ValueError(rejection)
        atomic_json(output, value)
        return dict(primary_path=str(output))

    monkeypatch.setattr(runner, "freeze", freeze)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(d.shards, "validation_plan", plan)
    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "measured", lambda *_: {})
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    monkeypatch.setattr(runner, "publish_primary", publish)
    if rejection == "conflicting_primary":
        with pytest.raises(ValueError, match="conflicting_primary"):
            runner.main(["--date", "20261109"])
    else:
        assert runner.main(["--date", "20261109"]) == 0
        value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
        assert value["corpus_ready_score"] == 0
        assert value["verdict_class"] == "disqualified"
        assert value["flagged_adversarial"] == bool(rejection)
