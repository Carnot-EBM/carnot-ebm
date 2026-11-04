"""REQ-REPORT-8787 / REQ-VERIFY-8787: private evidence and exact operation limits."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_operation_boundary_8787 as a
from carnot.reporting import hardware_operation_inputs_8787 as s
from carnot.reporting import durable_component_runner_8784 as producer
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def fixture_branch(tmp_path, **changes):
    """A null producer can qualify costs without qualifying scientific improvement."""
    task = dict(id="exp8784-durable-cost-components", deliverable="private.json")
    value = dict(
        experiment_id=8784,
        task_id=task["id"],
        milestone="2026.11.752",
        invocation_id="private",
        run_date="20261104",
        durable_cost_rows_ready_score=1,
        component_cost_ready_score=0,
        required_checks_passed=True,
        flagged_adversarial=False,
        verdict_class="null",
        honest_verdict="complete_null_private",
        per_window_rows=[],
        component_cost_rows=[],
    )
    value.update(changes)
    atomic_json(tmp_path / task["deliverable"], value)
    return task


@pytest.mark.parametrize(
    "change,state",
    [
        ({}, "qualified_valid_null"),
        ({"durable_cost_rows_ready_score": 0}, "observed_zero"),
        ({"durable_cost_rows_ready_score": None}, "observed"),
        ({"required_checks_passed": False}, "rejected_producer"),
        ({"task_id": "exp8772-aliased"}, "rejected_producer"),
        ({"verdict_class": "blocked"}, "rejected_producer"),
    ],
)
def test_independent_admission(tmp_path, monkeypatch, change, state):
    """REQ-REPORT-8787: valid null and zero readiness have different meanings."""
    task = fixture_branch(tmp_path, **change)
    monkeypatch.setattr(s, "producer_operands", lambda *args: [])
    monkeypatch.setattr(s.i, "terminal", lambda *args: [])
    b = s.admit(tmp_path, task, "durable_cost_rows_ready_score")
    assert b["checks"][-1]["operand_state"] == state
    assert b["qualified"] == (state == "qualified_valid_null")
    missing = s.admit(tmp_path, dict(task, deliverable="absent.json"), "ready")
    assert missing["checks"][0]["operand_state"] == "missing_file"
    assert s.admit(tmp_path, task, "absent")["checks"][0]["operand_state"] == "missing_field"


def test_replay_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8787: producer construction is required even for a readiness one."""
    task = fixture_branch(tmp_path)
    monkeypatch.setattr(
        s, "producer_operands", lambda *args: (_ for _ in ()).throw(ValueError("bad seal"))
    )
    b = s.admit(tmp_path, task, "durable_cost_rows_ready_score")
    assert not b["qualified"] and b["checks"][-1]["operand_state"] == "rejected_producer"
    monkeypatch.undo()
    with pytest.raises(ValueError, match="unregistered"):
        s.producer_operands(dict(experiment_id=8780), tmp_path / "unknown")


def test_durable_boundary():
    """REQ-REPORT-8787: complete windows survive rejected component attribution."""
    value = dict(
        experiment_id=8784,
        component_cost_ready_score=0,
        per_window_rows=[dict(unit_id="w", complete_ns=100, updates=32)],
        component_cost_rows=[dict(component="native_arithmetic", cost_ns=50)],
    )
    rows = s.cost_operands(value)
    assert len(rows) == 1 and rows[0]["operation"] == "durable_window_unpartitioned"
    assert rows[0]["duration_s"] == rows[0]["parent_duration_s"] == 1e-7
    assert rows[0]["denominator_updates"] == 32
    assert s.cost_operands(dict(experiment_id=8780, complete_cost_rows=[])) == []


def test_amdahl_denominators():
    """SCENARIO-REPORT-8787-OPERATIONS: unknown work stays in complete parent costs."""
    spans = [
        dict(
            operation=s.PAIRWISE,
            duration_s=0.99,
            parent_duration_s=1,
            parent="p",
            process_id="u",
            clock_domain="same",
        )
    ]
    branch = dict(
        task_id="private", qualified=True, checks=[], refs=[], value=dict(complete_cost_rows=spans)
    )
    support = a.support_rows(True)
    r = a.bounds([branch], support)[0]
    assert r["compatible_fraction"] == 0.99 and not r["feasible_100x"]
    assert r["unassigned_duration_s"] == pytest.approx(0.01)
    spans[0]["duration_s"] = 0.999
    assert a.bounds([branch], support)[0]["feasible_100x"]
    spans.append(
        dict(spans[0], operation="constraint_lookup", duration_s=0.001, clock_domain="other")
    )
    assert a.bounds([branch], support)[0]["compatible_fraction"] is None
    spans.pop()
    for change in [
        dict(duration_s=-1),
        dict(duration_s=float("nan")),
        dict(duration_s=2),
        dict(nested_parent_span=True),
    ]:
        bad = deepcopy(branch)
        bad["value"]["complete_cost_rows"][0].update(change)
        with pytest.raises(ValueError):
            a.bounds([bad], support)
    assert all(r["support"] != "supported" for r in support if r["board"] == "CUDA")


def test_private_build(tmp_path, monkeypatch):
    """REQ-VERIFY-8787: absent scientific branches do not prevent a complete mapping."""
    reference = s.i.reference
    production_results = a.e.ROOT / "results"

    def private_reference(path):
        assert not path.absolute().is_relative_to(production_results), "production_result_read"
        return reference(path)

    monkeypatch.setattr(s.i, "reference", private_reference)
    work = a.r.measure(tmp_path / "raw/invocations/private", "20261104", tmp_path, True)
    work["private_control"] = False
    work["evidence"]["board_qualified"] = True
    work["evidence"]["authority"]["ready"] = True
    value = a.build(work, [dict(passed=True)], tmp_path / (a.e.NAME + ".json"))
    assert value["hardware_mapping_ready_score"] == 1
    assert (
        value["physical_hardware_ready_score"] == value["complete_100x_learning_ready_score"] == 0
    )
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 5
    assert value["numerical_training_calls"] == 0 and value["MODEL_SPECS"] == []
    assert len({r["board"] for r in value["operation_support_rows"]}) == 5
    assert len({r["boundary"] for r in value["rows"]}) == 5
    assert (
        a.build(work, [dict(passed=False)], tmp_path / (a.e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_cli_and_rehashed_primitive(tmp_path):
    """SCENARIO-VERIFY-8787-REPLAY: real private children cover publication and rejection."""
    argv = [sys.executable, "-u", str(a.e.ROOT / a.e.CLI)]
    output, logs = tmp_path / (a.e.NAME + ".json"), tmp_path / "logs"
    assert child(
        "private",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        logs,
    )["passed"]
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    work["run_date"] = "forged"
    atomic_json(Path(value["primitive_reference"]["path"]), work)
    atomic_json(output, a.build(work, value["validation_receipts"], output))
    assert child("tamper", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]
    assert child("date", argv + ["--date", "20261399"], logs, expected=1)["passed"]


def test_producer_replay_and_terminal(tmp_path, monkeypatch):
    """REQ-VERIFY-8787: the registered original replay runs before terminal admission."""
    task = fixture_branch(tmp_path)
    ref = s.i.reference(tmp_path / "private.json")
    value = s.i.load(ref)
    value.update(
        primitive_reference=ref,
        validation_reference=ref,
        terminal_validation_sidecar_path=ref["path"],
    )
    atomic_json(tmp_path / "private.json", value)
    calls = []
    monkeypatch.setattr(producer, "replay", lambda path: calls.append(path))
    monkeypatch.setattr(s.i, "terminal", lambda *args: [])
    assert s.admit(tmp_path, task, "durable_cost_rows_ready_score")["qualified"]
    assert len(calls) == 1


def test_board_authority_and_mixed_branches(tmp_path, monkeypatch):
    """REQ-REPORT-8787: one rejected sibling cannot erase independently admitted data."""
    good = dict(required_checks_passed=True, hardware_mapping_ready_score=1)
    atomic_json(tmp_path / s.BOARD, good)
    monkeypatch.setattr(s, "BOARD_PIN", s.i.reference(tmp_path / s.BOARD)["sha256"])
    monkeypatch.setattr(s.i, "terminal", lambda *args: [])
    monkeypatch.setattr(
        s.contract, "authority", lambda root: dict(ready=True, rows=[], references=[])
    )
    assert s.authenticate(tmp_path)["board_qualified"]
    atomic_json(tmp_path / s.BOARD, dict(good, required_checks_passed=False))
    monkeypatch.setattr(s, "BOARD_PIN", s.i.reference(tmp_path / s.BOARD)["sha256"])
    evidence = s.authenticate(tmp_path)
    assert not evidence["board_qualified"]
    assert evidence["failures"][0]["reason"] == "unqualified_historical_mapping"


def test_qualified_cost_build(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8787-OPERATIONS: complete null costs survive missing learning evidence."""
    work = a.r.measure(tmp_path / "raw/invocations/private", "20261104", tmp_path, True)
    branch = work["evidence"]["branches"][2]
    task = fixture_branch(
        tmp_path,
        per_window_rows=[dict(unit_id="w", complete_ns=100, updates=32)],
        component_cost_rows=[dict(component="unqualified", cost_ns=10)],
    )
    monkeypatch.setattr(s, "producer_operands", lambda *args: [])
    monkeypatch.setattr(s.i, "terminal", lambda *args: [])
    branch.update(s.admit(tmp_path, task, "durable_cost_rows_ready_score"))
    result = a.build(work, [dict(passed=True)], tmp_path / (a.e.NAME + ".json"))
    assert result["completed_count"] == 1 and result["censored_count"] == 4
    assert result["diagnostic_cost_rows"][0]["eligible_for_bound"] is False
    assert result["compatible_fraction_rows"][0]["compatible_fraction"] == 0
    assert result["component_admission_rows"][0]["operand_state"] == "observed_zero"
    spans = [
        dict(
            operation=s.PAIRWISE,
            duration_s=0.5,
            parent_duration_s=1,
            process_id="u",
            parent="p",
            clock_domain="one",
            start_ns=0,
            end_ns=10,
        ),
        dict(
            operation="host",
            duration_s=0.5,
            parent_duration_s=1,
            process_id="u",
            parent="p",
            clock_domain="one",
            start_ns=5,
            end_ns=15,
        ),
    ]
    branch["value"]["complete_cost_rows"] = spans
    with pytest.raises(ValueError, match="overlapping_costs"):
        a.bounds([branch], a.support_rows(True))


def test_runner_validation_and_freeze_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8787: failed collection and changed node unions fail owned acceptance."""
    runner = a.r
    raw = tmp_path / "raw/invocations/test"
    (raw / "validation").mkdir(parents=True)
    nodes = ["tests/private.py::test_requirement"]
    atomic_json(raw / "frozen_nodes.json", dict(nodes=nodes))
    atomic_json(raw / "validation/collected_nodes.json", nodes)
    monkeypatch.setattr(runner, "BASE_VALIDATE", lambda raw: [])
    assert runner.validate(raw)[-1]["passed"]
    atomic_json(raw / "validation/collected_nodes.json", [])
    assert not runner.validate(raw)[-1]["passed"]
    (raw / "empty").write_text("")
    monkeypatch.setattr(
        runner, "child", lambda *args, **kwargs: dict(passed=False, stdout_path=str(raw / "empty"))
    )
    monkeypatch.setattr(
        runner,
        "BASE_MEASURE",
        lambda *args: dict(source_refs=[], resource_failures=[], code_refs=[]),
    )
    assert (
        runner.measure(raw, "20261104", tmp_path, True)["resource_failures"][-1]["operand_state"]
        == "failed_owned_validation"
    )
    assert runner.main(["--date", "20261399"]) == 1
