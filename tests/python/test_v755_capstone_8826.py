"""REQ-REPORT-8826, REQ-VERIFY-8826: accounting cannot upgrade constructed science."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import v755_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


def producer(tmp_path, **changes):
    source = tmp_path / "operand.json"
    c.atomic_json(source, dict(real="private fixture"))
    value = dict(
        experiment_id=8819,
        task_id="exp8819-consensus-selective-energy",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        raw_shard_hashes=[c.reference(source)],
        code_config_hashes=[],
        source_artifact_hashes=[],
        validation_receipts=[dict(passed=True)],
        owned_statement_coverage=dict(complete=True, percent=100),
        rows=[],
    )
    value.update(changes)
    path = tmp_path / "experiment_8819_fixture.json"
    c.atomic_json(path, value)
    return path


def test_admission_and_diagnostics(tmp_path):
    """SCENARIO-VERIFY-8826-TAMPER: custody failures remain explicit."""
    path = producer(tmp_path)
    assert c.admit(path, "exp8819-consensus-selective-energy")["admitted"]
    for changes in [
        dict(required_checks_passed=False),
        dict(verdict_class="disqualified"),
        dict(flagged_adversarial=True),
        dict(task_id="exp8819-foreign"),
        dict(owned_statement_coverage=dict(complete=False)),
        dict(validation_receipts=[dict(passed=False)]),
        dict(raw_shard_hashes=[]),
    ]:
        path = producer(tmp_path, **changes)
        assert not c.admit(path, "exp8819-consensus-selective-energy")["admitted"]
    path = producer(tmp_path, raw_shard_hashes={"operand": c.reference(tmp_path / "operand.json")})
    assert c.admit(path, "exp8819-consensus-selective-energy")["admitted"]
    (tmp_path / "operand.json").write_text("changed")
    assert not c.admit(path, "exp8819-consensus-selective-energy")["admitted"]
    path.write_text("{")
    assert not c.admit(path, "exp8819-consensus-selective-energy")["admitted"]
    assert c.gate(tmp_path / "missing", "x", "score", 1)["missing_reason"] == "missing_file"
    c.atomic_json(path, {})
    assert c.gate(path, "x", "score", 1)["missing_reason"] == "missing_field"
    c.atomic_json(path, dict(score=0))
    assert c.gate(path, "x", "score", 1)["observed"] == 0
    c.atomic_json(path, dict(score=1))
    assert c.gate(path, "x", "score", 1)["passed"]


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    return c.freeze(c.ROOT, tmp_path_factory.mktemp("capstone"), "20261105")


def test_complete_accounting_and_reductions(frozen):
    """SCENARIO-REPORT-8826-ACCOUNTING: CPU nulls survive external absence."""
    value = c.build(frozen, validation())
    assert len(value["rows"]) == len(value["task_outcome_rows"]) == 13
    assert value["verdict_class"] == "blocked" and value["reconciliation_ready_score"] == 1
    assert value["constructed_measurement_count"] > 0
    assert value["independent_constructed_unit_count"] == 240
    assert value["natural_generalization_count"] == 0
    assert not value["publication_authorized"] and not value["activation_authorized"]
    by_id = {r["task_id"]: r for r in value["rows"]}
    assert by_id["exp8820-causal-objective-streams"]["primary_honest_verdict"] is None
    assert by_id["exp8820-causal-objective-streams"]["disposition"] == "conductor_cascade_skip"
    assert by_id["exp8819-consensus-selective-energy"]["verdict_class"] == "null"
    assert value["rejected_evidence"]
    assert len(value["retirement_recommendations"]) >= 13
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"


def test_replay_rejects_rehashed_semantics(frozen, tmp_path):
    """SCENARIO-REPORT-8826-REPLAY: checksum edits cannot repair a false reduction."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    for field in [
        "constructed_measurement_count",
        "natural_generalization_count",
        "reconciliation_ready_score",
    ]:
        changed = deepcopy(value)
        changed[field] += 1
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(path, changed)
        assert not c.replay(path)["passed"]
    path.write_text("{")
    assert not c.replay(path)["passed"]


def test_missing_and_skip_slots(tmp_path):
    task = dict(
        id="exp8819-consensus-selective-energy",
        title="A private task",
        deliverable="results/experiment_8819_fixture.json",
        gated_on=[],
    )
    row = c.outcome(task, tmp_path, tmp_path / "raw", [])
    assert not row["accounted"] and row["primary_honest_verdict"] is None
    path = tmp_path / task["deliverable"]
    path.parent.mkdir()
    c.atomic_json(
        path,
        dict(
            experiment=8819,
            title=task["title"],
            blocked_at_layer="conductor_pre_gate",
            honest_verdict="blocked_gate_check_failed",
            gates_evaluated=[],
        ),
    )
    row = c.outcome(task, tmp_path, tmp_path / "raw", [])
    assert row["disposition"] == "conductor_pre_gate" and row["accounted"]


def test_cli_paths(frozen, tmp_path, monkeypatch):
    from carnot.reporting import v755_capstone_runner as runner

    assert runner.main(["--date", "bad"]) == 2
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(frozen, validation()))
    assert runner.main(["--replay", str(path)]) == 0
    assert runner.main(["--replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(sys, "argv", [c.CLI, "--replay", str(path)])
    with pytest.raises(SystemExit) as end:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert end.value.code == 0


def test_replay_custody_controls(frozen, tmp_path):
    """SCENARIO-VERIFY-8826-TAMPER: changed raw work cannot repair its own authority."""
    value = c.build(frozen, validation())
    candidate = tmp_path / "candidate.json"
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    c.atomic_json(candidate, changed)
    assert not c.replay(candidate)["passed"]
    for mode in ["authority", "reduction"]:
        work = deepcopy(frozen)
        if mode == "authority":
            work["contract"]["tasks"][0]["title"] += "forged"
        else:
            slot = next(s for s in work["slots"] if s["admitted"])
            slot["reduction"]["constructed_measurement_count"] += 1
        raw_work = tmp_path / "work.json"
        c.atomic_json(raw_work, work)
        changed = deepcopy(value)
        changed["work_reference"] = c.reference(raw_work)
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(candidate, changed)
        assert not c.replay(candidate)["passed"]


def test_rejected_rows_and_preconditions(frozen, tmp_path, monkeypatch):
    """REQ-REPORT-8826: rejected measurement rows remain countable without admission."""
    slot = next(s for s in frozen["slots"] if s["experiment_id"] == 8818)
    value = json.loads(Path(slot["primary_reference"]["path"]).read_bytes())
    reduction = c.reduce(value)
    assert reduction["constructed_measurement_count"] == 4320
    assert reduction["independent_constructed_unit_count"] == 240
    assert not reduction["registered_estimands"]["acceptance_gates"]["coverage_gain"]
    with pytest.raises(ValueError, match="full_object_authority"):
        c.freeze(tmp_path, tmp_path / "missing", "20261105")
    monkeypatch.setattr(c.authority, "HISTORY_PINS", {c.authority.HISTORY: "0" * 64})
    with pytest.raises(ValueError, match="historical_authority_drift"):
        c.freeze(c.ROOT, tmp_path / "history", "20261105")


def test_private_publication(frozen, tmp_path, monkeypatch):
    """REQ-VERIFY-8826: failed checks publish disqualification; terminal rejection preserves bytes."""
    from carnot.reporting import v755_capstone_runner as runner

    monkeypatch.setattr(
        runner,
        "Coverage",
        lambda **kwargs: SimpleNamespace(
            set_option=lambda *args: None, start=lambda: None, stop=lambda: None, save=lambda: None
        ),
    )
    monkeypatch.setattr(c, "freeze", lambda *args: deepcopy(frozen))

    def qualified(raw, tracer):
        tracer.stop()
        tracer.save()
        return validation()

    monkeypatch.setattr(runner.checks, "validate", qualified)
    monkeypatch.setattr(runner.checks, "terminal", lambda p, r: dict(passed=c.replay(p)["passed"]))
    output = tmp_path / "results" / (c.NAME + ".json")
    assert runner.main(["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]) == 0
    assert json.loads(output.read_bytes())["reconciliation_ready_score"] == 1
    monkeypatch.setattr(runner.checks, "terminal", lambda p, r: dict(passed=False))
    assert runner.run(output, tmp_path / "raw", c.ROOT, "20261105") == 1
    monkeypatch.setattr(runner.checks, "terminal", lambda p, r: dict(passed=True))
    monkeypatch.setattr(
        runner.checks,
        "validate",
        lambda raw, tracer: (tracer.stop(), tracer.save(), validation(False))[-1],
    )
    assert runner.run(output, tmp_path / "raw", c.ROOT, "20261105") == 1
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
