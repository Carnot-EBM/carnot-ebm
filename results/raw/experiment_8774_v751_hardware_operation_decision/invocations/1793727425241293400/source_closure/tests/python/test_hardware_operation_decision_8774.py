"""REQ-REPORT-8774 / REQ-VERIFY-8774: private operation and byte custody controls."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_operation_decision_8774 as a
from carnot.reporting import hardware_operation_inputs_8774 as source
from carnot.reporting import ordinary_durable_runner_8772 as durable
from carnot.reporting import optimized_cost_attribution_runner_8770 as residual
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def branches():
    """Absent siblings must stay absent when one independent cost qualifies."""
    return [
        dict(task_id=str(i), field="ready", qualified=False, checks=[], refs=[], value={})
        for i in range(3)
    ]


@pytest.mark.parametrize(
    "change,state",
    [
        ({}, "qualified"),
        ({"ready": 0}, "observed_zero"),
        ({"drop": True}, "missing_field"),
        ({"ready": None}, "observed"),
        ({"required_checks_passed": False}, "rejected_producer"),
        ({"experiment_id": 1}, "rejected_producer"),
    ],
)
def test_admission(tmp_path, monkeypatch, change, state):
    """REQ-VERIFY-8774: readiness zero is different from a rejected producer."""
    task = dict(id="exp8772-private", deliverable="private.json")
    value = dict(
        experiment_id=8772,
        task_id=task["id"],
        ready=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        verdict_class="null",
        acceptance_gates=dict(owned_validation=True, bounded_durable_benefit=False),
        per_window_rows=[],
    )
    value.update(change)
    if value.pop("drop", False):
        value.pop("ready")
    atomic_json(tmp_path / "private.json", value)
    monkeypatch.setattr(source.i, "terminal", lambda *args: [])
    monkeypatch.setattr(source, "producer_operands", lambda *args: [])
    row = source.admit(tmp_path, task, "ready")
    assert row["qualified"] is (state == "qualified")
    assert row["checks"][-1]["operand_state"] == state
    assert not source.admit(tmp_path, dict(task, deliverable="absent.json"), "ready")["qualified"]


def test_costs_and_clocks():
    """SCENARIO-REPORT-8774-OPERATIONS: unsupported work cannot enter a fabric fraction."""
    b = branches()
    spans = [
        dict(
            operation="quadratic_Ising_energy_k_max_5",
            duration_s=0.9,
            parent_duration_s=1,
            parent="complete",
            clock_domain="same",
            process_id="p",
        ),
        dict(
            operation="feature_extraction",
            duration_s=0.1,
            parent_duration_s=1,
            parent="complete",
            clock_domain="same",
            process_id="p",
        ),
    ]
    b[1].update(qualified=True, value=dict(complete_cost_rows=spans))
    support = [dict(board="KV260", operation=r["operation"], support="supported") for r in spans]
    result = a.cost_rows(b, support)
    bound = result["conditional_acceleration_bounds"][0]
    assert bound["conditional_ceiling"] == pytest.approx(10)
    assert bound["physical_transfer_duration_s"] is None
    assert bound["complete_service_bound"] is None
    bad = deepcopy(b)
    bad[1]["value"]["complete_cost_rows"][1]["clock_domain"] = "different"
    assert all(
        r["conditional_ceiling"] is None
        for r in a.cost_rows(bad, support)["conditional_acceleration_bounds"]
    )
    for change in [
        dict(duration_s=float("nan")),
        dict(duration_s=0.2),
        dict(nested_parent_span=True),
    ]:
        bad = deepcopy(b)
        bad[1]["value"]["complete_cost_rows"][1].update(change)
        with pytest.raises(ValueError):
            a.cost_rows(bad, support)
    assert a.cost_rows(branches(), support)["missing_cost_rows"]


def test_durable_and_residual_adapter():
    """REQ-REPORT-8774: a storage window is separate from installed inference."""
    value = dict(
        experiment_id=8772,
        per_window_rows=[
            dict(
                unit_id="w",
                complete_ns=100,
                update_rows=[dict(arithmetic_ns=10), dict(arithmetic_ns=20)],
            )
        ],
    )
    rows = source.cost_operands(value)
    assert sum(r["duration_s"] for r in rows) == pytest.approx(1e-7)
    assert rows[1]["operation"] == "durable_window_remainder"
    value = dict(experiment_id=8770, component_cost_rows=[dict(clock_domain="child_serve")])
    assert source.cost_operands(value)[0]["parent"] == "child_serve"
    assert source.cost_operands(dict(experiment_id=8766, complete_cost_rows=[])) == []


def test_mapping_build(tmp_path):
    """SCENARIO-REPORT-8774-OPERATIONS: mapping completion never invents execution."""
    work = a.r.measure(tmp_path / "raw/invocations/private", "20261103", tmp_path, True)
    work["private_control"] = False
    work["evidence"]["board_qualified"] = True
    output = tmp_path / (a.e.NAME + ".json")
    value = a.build(work, [dict(passed=True)], output)
    assert value["hardware_mapping_ready_score"] == 1
    assert (
        value["physical_hardware_ready_score"] == value["complete_100x_learning_ready_score"] == 0
    )
    assert value["verdict_class"] == "blocked"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert {r["board"] for r in value["board_status_rows"]} == {"KV260", "PolarFire", "GateMate"}
    assert all(r["support"] != "supported" for r in value["higher_order_interface_rows"])
    assert value["cost_scope_rows"] and value["measurement_to_change_conclusion"]
    assert a.build(work, [dict(passed=False)], output)["hardware_mapping_ready_score"] == 0
    assert a.build(work, [dict(passed=False)], output)["verdict_class"] == "disqualified"


def test_actual_private_cli(tmp_path):
    """SCENARIO-VERIFY-8774-CUSTODY: real CLI children reject rehashed primitive edits."""
    argv = [sys.executable, "-u", str(a.e.ROOT / a.e.CLI)]
    output, logs = tmp_path / (a.e.NAME + ".json"), tmp_path / "logs"
    assert child(
        "private",
        argv
        + [
            "--date",
            "20261103",
            "--private-control",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        logs,
    )["passed"]
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    value = json.loads(output.read_bytes())
    path = Path(value["primitive_reference"]["path"])
    work = json.loads(path.read_bytes())
    work["run_date"] = "forged"
    atomic_json(path, work)
    atomic_json(output, a.build(work, value["validation_receipts"], output))
    assert child("tamper", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("date", argv + ["--date", "20261399"], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]


def test_producer_replay_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8774: registered original replays authenticate; unknown producers reject."""
    path = tmp_path / "producer.json"
    atomic_json(path, {})
    ref = source.i.reference(path)
    value = dict(
        experiment_id=8772,
        primitive_reference=ref,
        validation_reference=ref,
        terminal_validation_sidecar_path=str(path),
    )
    monkeypatch.setattr(durable, "replay", lambda *args: None)
    assert len(source.producer_operands(value, path)) == 3
    monkeypatch.setattr(residual, "replay", lambda *args: True)
    assert source.producer_operands(dict(value, experiment_id=8770), path)
    monkeypatch.setattr(residual, "replay", lambda *args: False)
    with pytest.raises(ValueError, match="original_residual_replay"):
        source.producer_operands(dict(value, experiment_id=8770), path)
    with pytest.raises(ValueError, match="unregistered_learning_producer"):
        source.producer_operands(dict(value, experiment_id=8766), path)


def test_authentication_board_controls(tmp_path, monkeypatch):
    """REQ-VERIFY-8774: pinned board bytes still require original mapping acceptance."""
    path = tmp_path / source.BOARD
    good = dict(
        required_checks_passed=True, flagged_adversarial=False, hardware_mapping_ready_score=1
    )
    atomic_json(path, good)
    monkeypatch.setattr(source, "BOARD_PIN", source.i.reference(path)["sha256"])
    monkeypatch.setattr(source.i, "terminal", lambda *args: [])
    monkeypatch.setattr(
        source.contract, "authority", lambda root: dict(ready=True, rows=[], references=[])
    )
    assert source.authenticate(tmp_path)["board_qualified"]
    atomic_json(path, dict(good, required_checks_passed=False))
    monkeypatch.setattr(source, "BOARD_PIN", source.i.reference(path)["sha256"])
    result = source.authenticate(tmp_path)
    assert not result["board_qualified"]
    assert result["failures"][0]["reason"] == "unqualified_original_mapping"


def test_scoped_node_and_freeze_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8774: changed node unions and failed collection suppress owned readiness."""
    runner = a.r
    raw = tmp_path / "raw/invocations/test"
    (raw / "validation").mkdir(parents=True)
    nodes = ["tests/private.py::test_req"]
    atomic_json(raw / "frozen_nodes.json", dict(nodes=nodes))
    atomic_json(raw / "measurement.json", dict(frozen_test_nodes=nodes))
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
    work = runner.measure(raw, "20261103", tmp_path, True)
    assert work["resource_failures"][-1]["operand_state"] == "failed_owned_validation"
    assert runner.main(["--date", "20261399"]) == 1


def test_diagnostic_zero_and_exact_identity(tmp_path, monkeypatch):
    """REQ-REPORT-8774: diagnostic leaves cannot supply qualified bounds or hide task mismatch."""
    task = dict(id="exp8770-private", deliverable="private.json")
    atomic_json(
        tmp_path / "private.json",
        dict(residual=0, component_cost_rows=[dict(operation="features", duration_s=1)]),
    )
    row = source.admit(tmp_path, task, "residual")
    assert row["observed_cost_rows"][0]["duration_s"] == 1
    assert row["qualified"] is False
    value = dict(
        experiment_id=8770,
        task_id="exp8770-wrong",
        residual=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        verdict_class="null",
    )
    atomic_json(tmp_path / "private.json", value)
    row = source.admit(tmp_path, task, "residual")
    assert row["checks"][-1]["field"] == "task_id"
    assert row["checks"][-1]["observed"] == "exp8770-wrong"


def test_current_operands_and_inference_fallback(tmp_path):
    """REQ-REPORT-8774: immutable published costs do not repair missing sibling operands."""
    evidence = source.authenticate(a.e.ROOT)
    assert evidence["board_qualified"], evidence["failures"]
    assert evidence["inference_fallback"]["qualified"]
    assert evidence["branches"][0]["checks"][0]["operand_state"] == "missing_file"
    assert evidence["branches"][1]["checks"][0]["operand_state"] == "observed_zero"
    assert evidence["branches"][2]["checks"][-1]["operand_state"] == "rejected_producer"
    assert evidence["branches"][2]["checks"][-1]["field"] == "task_id"
    work = a.r.measure(tmp_path / "raw/invocations/private", "20261103", tmp_path, True)
    work["evidence"] = evidence
    value = a.build(work, [dict(passed=True)], tmp_path / (a.e.NAME + ".json"))
    assert len(value["diagnostic_cost_rows"]) == 174
    assert value["cost_scope_rows"]
    assert any(r.get("producer") == "exp8757-feature-reuse-cost" for r in value["cost_scope_rows"])
    assert all(
        r["complete_service_bound"] is None for r in value["conditional_acceleration_bounds"]
    )


def test_terminal_producer_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8774: terminal rejection cannot qualify a positive readiness scalar."""
    task = dict(id="exp8772-private", deliverable="private.json")
    atomic_json(
        tmp_path / "private.json",
        dict(
            experiment_id=8772,
            task_id=task["id"],
            ready=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            verdict_class="blocked",
        ),
    )
    row = source.admit(tmp_path, task, "ready")
    assert row["checks"][-1]["reason"] == "producer_terminal_verdict"
