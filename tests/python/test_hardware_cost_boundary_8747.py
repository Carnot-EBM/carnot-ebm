"""REQ-REPORT-8747 and REQ-VERIFY-8747: independent costs preserve physical limits."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_cost_boundary_8747 as a
from carnot.reporting import bounded_durable_runner_8745 as producer
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_current_independent_sources():
    """SCENARIO-REPORT-8747-INDEPENDENT: all branches retain their own blocker."""
    evidence = a.authenticate(a.e.ROOT)
    assert evidence["authority"]["ready"]
    assert len(evidence["authority"]["rows"]) == 12
    assert evidence["board_qualified"], evidence["failures"]
    assert [b["qualified"] for b in evidence["branches"]] == [False, False, False]
    assert [b["checks"][0]["operand_state"] for b in evidence["branches"]] == [
        "missing_file",
        "missing_file",
        "observed_zero",
    ]
    assert all(len(b["checks"]) == 1 for b in evidence["branches"])
    assert evidence["branches"][1]["scheduling_disposition"]["science_evidence"] is False


def test_independent_admission(tmp_path, monkeypatch):
    """REQ-VERIFY-8747: one valid branch survives missing siblings; changed bytes fail."""
    path = tmp_path / "inference.json"
    task = dict(id="exp8744-single-extraction-cost", deliverable="inference.json")
    atomic_json(
        path,
        dict(
            experiment_id=8744,
            task_id=task["id"],
            installed_cost_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            acceptance_gates=dict(owned=True),
            verdict_class="null",
            adversarial_findings=[],
        ),
    )
    monkeypatch.setattr(a.i, "terminal", lambda *args: [])
    monkeypatch.setattr(a.i, "producer_operands", lambda *args: [])
    assert a.admit(tmp_path, task, "installed_cost_ready_score")["qualified"]
    value = json.loads(path.read_bytes())
    value.pop("installed_cost_ready_score")
    atomic_json(path, value)
    assert (
        a.admit(tmp_path, task, "installed_cost_ready_score")["checks"][0]["operand_state"]
        == "missing_field"
    )
    value.update(installed_cost_ready_score=1, required_checks_passed=False)
    atomic_json(path, value)
    assert not a.admit(tmp_path, task, "installed_cost_ready_score")["qualified"]


def test_rows_and_ceilings():
    """SCENARIO-REPORT-8747-CLOCKS: parents, child clocks and windows remain separate."""
    branches = [dict(task_id=str(n), qualified=False, checks=[]) for n in range(3)]
    support = [dict(board="KV260", operation="quadratic", support="supported")]
    for b in branches:
        b.update(
            qualified=True,
            value=dict(
                complete_cost_rows=[
                    dict(
                        operation="quadratic",
                        duration_s=0.9,
                        parent_duration_s=1,
                        process_id="p",
                        parent="test_parent",
                    ),
                    dict(
                        operation="host_transfer",
                        duration_s=0.1,
                        parent_duration_s=1,
                        process_id="p",
                        parent="test_parent",
                    ),
                ]
            ),
        )
    result = a.cost_rows(branches, support)
    assert {r["boundary"] for r in result["boundary_cost_rows"]} == set(a.BOUNDARIES)
    assert len(result["operation_support_rows"]) == 19
    ceiling = result["conditional_acceleration_bounds"][0]
    assert ceiling["conditional_ceiling"] == pytest.approx(10)
    assert ceiling["minimum_additional_compatible_fraction_100x"] == pytest.approx(0.09)
    assert all(r["achieved_speedup"] is None for r in result["conditional_acceleration_bounds"])
    rows = branches[0]["value"]["complete_cost_rows"]
    rows[1]["duration_s"] = 0.2
    with pytest.raises(ValueError, match="non_disjoint_parent_costs"):
        a.cost_rows(branches, support)
    rows[1]["duration_s"] = None
    assert (
        a.cost_rows(branches, support)["conditional_acceleration_bounds"][0]["conditional_ceiling"]
        is None
    )
    rows[1]["parent_duration_s"] = 2
    assert (
        a.cost_rows(branches, support)["conditional_acceleration_bounds"][0]["parent_duration_s"]
        is None
    )
    rows[1]["parent_duration_s"] = 1
    rows[1].update(duration_s=0.1, nested_parent_span=True)
    with pytest.raises(ValueError, match="nested_parent_child_spans"):
        a.cost_rows(branches, support)
    rows[1].pop("nested_parent_span")
    rows[1].update(start_ns=1, end_ns=3)
    rows[0].update(start_ns=2, end_ns=4)
    with pytest.raises(ValueError, match="overlapping_operation_spans"):
        a.cost_rows(branches, support)
    branches[0].update(qualified=False, checks=[dict(passed=False)])
    assert a.cost_rows(branches, support)["missing_cost_rows"][0]["duration_s"] is None


def test_mapping_retains_scope(tmp_path):
    """REQ-REPORT-8747: readiness and old evidence are not hardware acceleration."""
    work = a.r.measure(tmp_path / "raw" / "invocations" / "unit", "20261102", tmp_path, True)
    work["private_control"] = False
    work["evidence"]["board_qualified"] = True
    value = a.build(work, [dict(passed=True)], tmp_path / (a.e.NAME + ".json"))
    assert value["hardware_mapping_ready_score"] == 1
    assert (
        value["physical_hardware_ready_score"] == value["complete_100x_learning_ready_score"] == 0
    )
    assert value["boundary_cost_rows"] == []
    assert value["device_commands_executed"] == []
    assert value["MODEL_SPECS"] == []
    assert value["inference_substrate_class"] == "no_model_load"
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["verdict_class"] == "blocked"
    assert (
        a.build(work, [dict(passed=False)], tmp_path / (a.e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )
    assert value["acquisition_prerequisites"]["required"][-1] == "actual device execution"
    assert all(r["boundary"] == b for r, b in zip(value["rows"], a.BOUNDARIES, strict=True))


def test_producer_identity_and_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-8747: readiness cannot replace producer replay and byte custody."""

    path, final = tmp_path / "primary.json", tmp_path / "terminal.json"
    path.write_text("{}")
    atomic_json(
        final, dict(passed=True, publication=dict(primary_sha256=a.i.reference(path)["sha256"]))
    )
    value = dict(
        experiment_id=8745,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        primitive_reference=a.i.reference(path),
        terminal_validation_sidecar_path=str(final),
    )
    monkeypatch.setattr(producer, "replay", lambda *args: None)
    assert a.producer_operands(value, path)
    with pytest.raises(ValueError, match="unregistered_model_free_producer_replay"):
        a.producer_operands(dict(value, experiment_id=8744), path)
    atomic_json(final, dict(passed=False))
    with pytest.raises(ValueError, match="producer_final_identity"):
        a.producer_operands(value, path)
    value["primitive_reference"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="changed_bytes"):
        a.producer_operands(value, path)


def test_actual_private_cli_and_rehashed_replay(tmp_path):
    """SCENARIO-VERIFY-8747-SEAL: actual CLI rejects consistently rehashed edits."""
    argv = [sys.executable, "-u", str(a.e.ROOT / a.e.CLI)]
    output, logs = tmp_path / (a.e.NAME + ".json"), tmp_path / "logs"
    receipt = child(
        "private",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        logs,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["hardware_mapping_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    primitive = Path(value["primitive_reference"]["path"])
    work = json.loads(primitive.read_bytes())
    work["run_date"] = "forged"
    atomic_json(primitive, work)
    atomic_json(output, a.build(work, value["validation_receipts"], output))
    assert child("tampered", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("bad_date", argv + ["--date", "20261399"], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]


def test_missing_original_method(tmp_path, monkeypatch):
    """REQ-VERIFY-8747: absent original evidence is terminal blocked, with its hash."""
    monkeypatch.setattr(a.i, "PINS", {"missing.json": "0" * 64})
    evidence = a.authenticate(tmp_path)
    assert not evidence["board_qualified"]
    assert evidence["failures"][-1]["observed"] is None


def test_unnamed_clock_and_cpu_reuse(tmp_path):
    """REQ-REPORT-8747: absent clock names remain unknown; feature reuse stays on CPU."""
    branches = [
        dict(task_id=str(n), field="ready", qualified=False, checks=[], refs=[], value={})
        for n in range(3)
    ]
    branches[1].update(
        qualified=True,
        value=dict(
            complete_cost_rows=[
                dict(
                    operation="feature_extraction",
                    duration_s=0.2,
                    parent_duration_s=1,
                    process_id="p",
                )
            ]
        ),
    )
    result = a.cost_rows(branches, [])
    assert result["conditional_acceleration_bounds"][0]["conditional_ceiling"] is None
    work = a.r.measure(tmp_path / "raw" / "invocations" / "cpu", "20261102", tmp_path, True)
    work["evidence"]["branches"] = branches
    value = a.build(work, [dict(passed=True)], tmp_path / (a.e.NAME + ".json"))
    assert value["cpu_feature_reuse_rows"][0]["device_attribution"] is None
    assert len(value["cpu_feature_reuse_rows"][0]["cost_rows"]) == 1


def test_full_suite_receipt_retained(tmp_path, monkeypatch):
    """REQ-VERIFY-8747: retain actual child bytes without manufacturing coverage."""
    logs = tmp_path / "original"
    actual = child("full_python_suite", [sys.executable, "-c", "print(8747)"], logs)
    monkeypatch.setattr(a, "FULL_SUITE_LOGS", logs)
    monkeypatch.setattr(a, "BASE_VALIDATE", lambda raw: [])
    raw = tmp_path / "new"
    raw.mkdir()
    receipt = a.validate(raw)[0]
    assert receipt["passed"]
    assert receipt["stdout_sha256"] == actual["stdout_sha256"]
    assert len(receipt["references"]) == 3


def test_prior_flag_rejected(tmp_path, monkeypatch):
    """REQ-VERIFY-8747: exact hashes do not authorize a flagged historical producer."""
    path = tmp_path / "flagged.json"
    atomic_json(path, dict(required_checks_passed=False, flagged_adversarial=True))
    monkeypatch.setattr(a.i, "PINS", {path.name: a.i.reference(path)["sha256"][7:]})
    monkeypatch.setattr(a.i, "terminal", lambda *args: [])
    evidence = a.authenticate(tmp_path)
    assert not evidence["board_qualified"]
    assert evidence["failures"][-1]["reason"] == "flagged_original_board_evidence"


def test_mixed_clocks_are_not_added():
    """SCENARIO-REPORT-8747-CLOCKS: numerically equal totals do not unite clock domains."""
    branches = [dict(task_id=str(n), qualified=False, checks=[]) for n in range(3)]
    branches[0].update(
        qualified=True,
        value=dict(
            complete_cost_rows=[
                dict(
                    operation="quadratic",
                    duration_s=0.9,
                    parent_duration_s=1,
                    parent="service",
                    process_id="p",
                    clock_domain="child",
                ),
                dict(
                    operation="transfer",
                    duration_s=0.1,
                    parent_duration_s=1,
                    parent="service",
                    process_id="p",
                    clock_domain="parent",
                ),
            ]
        ),
    )
    result = a.cost_rows(
        branches, [dict(board="KV260", operation="quadratic", support="supported")]
    )
    assert all(r["conditional_ceiling"] is None for r in result["conditional_acceleration_bounds"])
    assert result["boundary_cost_rows"][0]["reported_parent_duration_s"] == 1
