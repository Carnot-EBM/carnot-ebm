"""REQ-REPORT-8735 and REQ-VERIFY-8735: measured CPU costs retain their scope."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_cost_boundary_8735 as a
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_independent_current_inputs():
    """SCENARIO-REPORT-8735-INDEPENDENT: a missing sibling cannot hide inference spans."""
    evidence = a.i.authenticate(a.e.ROOT)
    assert evidence["authority"]["ready"]
    assert len(evidence["authority"]["rows"]) == 14
    assert evidence["board_qualified"], evidence["failures"]
    assert [b["qualified"] for b in evidence["branches"]] == [False, False, True]
    assert evidence["branches"][0]["checks"][0]["operand_state"] == "missing_file"
    assert evidence["branches"][1]["checks"][0]["operand_state"] == "observed_zero"
    assert len(evidence["branches"][2]["value"]["component_span_rows"]) == 380


def test_mapping_and_new_costs(tmp_path):
    """REQ-REPORT-8735: each measured operation retains its own parent and producer."""
    work = a.r.measure(tmp_path / "raw" / "invocations" / "unit", "20261102", tmp_path, True)
    work["private_control"] = False
    work["evidence"]["board_qualified"] = True
    branches = work["evidence"]["branches"]
    for b, operation, total in zip(
        branches, ["learning", "sparse_update", "serialization"], [10, 2, 1]
    ):
        b.update(
            qualified=True,
            value={
                "complete_cost_rows": [
                    dict(
                        operation=operation,
                        duration_s=0.1,
                        parent_duration_s=total,
                        process_id=operation,
                    )
                ]
            },
        )
    branches[2]["value"]["component_span_rows"] = branches[2]["value"].pop("complete_cost_rows")
    value = a.build(work, [dict(passed=True)], tmp_path / (a.e.NAME + ".json"))
    assert value["hardware_mapping_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert len(value["complete_cost_rows"]) == 3
    assert {r["boundary"] for r in value["complete_cost_rows"]} == {
        "causal_learning",
        "constructed_resident_update",
        "installed_inference_only",
    }
    assert not value["missing_cost_rows"]
    mapped = [r for r in value["operation_support_rows"] if "cost_id" in r]
    assert len(mapped) == 9
    assert all(r["support"] != "supported" for r in mapped)
    assert all(r["conditional_ceiling"] is None for r in value["conditional_acceleration_bounds"])
    assert value["ideal_required_compatible_fraction"] == {"10x": 0.9, "100x": 0.99}
    assert (
        a.build(work, [dict(passed=False)], tmp_path / (a.e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_producer_replay_rejects_substitution(tmp_path, monkeypatch):
    """REQ-VERIFY-8735: original producer replay and source identity remain mandatory."""
    from carnot.reporting import installed_cost_attribution_8733 as producer
    from carnot.reporting import resident_update_runner_8732 as resident

    path = tmp_path / "primary.json"
    path.write_text("{}")
    source = tmp_path / "source.py"
    source.write_text("original")
    primitive = tmp_path / "work.json"
    atomic_json(primitive, {})
    final = tmp_path / "terminal.json"
    atomic_json(
        final, dict(passed=True, publication=dict(primary_sha256=a.i.reference(path)["sha256"]))
    )
    value = dict(
        experiment_id=8733,
        source_artifact_hashes=[],
        code_config_hashes=[a.i.reference(source)],
        raw_shard_hashes=[],
        primitive_reference=a.i.reference(primitive),
        terminal_validation_sidecar_path=str(final),
    )
    monkeypatch.setattr(producer, "replay", lambda *args: True)
    assert a.producer_operands(value, path)
    monkeypatch.setattr(resident, "replay", lambda *args: None)
    assert a.producer_operands(dict(value, experiment_id=8732), path)
    monkeypatch.setattr(producer, "replay", lambda *args: False)
    with pytest.raises(ValueError, match="original_producer_replay"):
        a.producer_operands(value, path)
    with pytest.raises(ValueError, match="unregistered_learning_producer"):
        a.producer_operands(dict(value, experiment_id=8727), path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed_bytes"):
        a.producer_operands(value, path)
    source.write_text("original")
    atomic_json(final, dict(passed=False))
    with pytest.raises(ValueError, match="producer_final_identity"):
        a.producer_operands(value, path)


def test_actual_private_cli_and_rehashed_replay(tmp_path):
    """REQ-VERIFY-8735: actual subprocess paths reject altered work and missing sources."""
    argv = [sys.executable, "-u", str(a.e.ROOT / a.e.CLI)]
    output, logs = tmp_path / (a.e.NAME + ".json"), tmp_path / "logs"
    receipt = child(
        "private",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        logs,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["model_invocation_counts"]["language_model_calls"] == 0
    assert value["hardware_mapping_ready_score"] == 0
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    path = Path(value["primitive_reference"]["path"])
    original = path.read_bytes()
    work = json.loads(original)
    work["run_date"] = "forged"
    atomic_json(path, work)
    atomic_json(output, a.build(work, value["validation_receipts"], output))
    assert child("tamper", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    path.write_bytes(original)
    atomic_json(output, value)
    path.unlink()
    assert child("missing", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("date", argv + ["--date", "bad"], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]


def test_only_authenticated_compatible_fraction_permits_ceiling():
    """REQ-REPORT-8735: the ideal formula uses a complete parent including host work."""
    from carnot.reporting.hardware_cost_rows_8735 import cost_rows

    branches = [dict(task_id=str(n), qualified=False, checks=[]) for n in range(3)]
    branches[0].update(
        qualified=True,
        value=dict(
            complete_cost_rows=[
                dict(operation="quadratic", duration_s=0.9, parent_duration_s=1, process_id="p"),
                dict(
                    operation="host_transfer", duration_s=0.1, parent_duration_s=1, process_id="p"
                ),
            ]
        ),
    )
    support = [dict(board="KV260", operation="quadratic", support="supported")]
    result = cost_rows(branches, support)
    bound = result["conditional_acceleration_bounds"][0]
    assert bound["conditional_ceiling"] == pytest.approx(10)
    assert bound["minimum_additional_compatible_fraction_10x"] == 0
    assert bound["minimum_additional_compatible_fraction_100x"] == pytest.approx(0.09)
    rows = branches[0]["value"]["complete_cost_rows"]
    rows[1]["duration_s"] = 0.2
    with pytest.raises(ValueError, match="non_disjoint_parent_costs"):
        cost_rows(branches, support)
    rows[1]["duration_s"] = None
    assert (
        cost_rows(branches, support)["conditional_acceleration_bounds"][0]["conditional_ceiling"]
        is None
    )
    rows[1]["parent_duration_s"] = 2
    assert (
        cost_rows(branches, support)["conditional_acceleration_bounds"][0]["parent_duration_s"]
        is None
    )


def test_missing_original_method_is_terminal_blocked(tmp_path, monkeypatch):
    """REQ-VERIFY-8735: missing method bytes produce diagnostics rather than an exception."""
    monkeypatch.setattr(
        a.contract, "PRESERVED", dict(path=str(tmp_path / "absent.md"), sha256="sha256:" + "0" * 64)
    )
    evidence = a.authenticate(tmp_path)
    assert not evidence["board_qualified"]
    gate = evidence["failures"][-1]
    assert gate["field"] == "original_sha256"
    assert gate["observed"] is None
