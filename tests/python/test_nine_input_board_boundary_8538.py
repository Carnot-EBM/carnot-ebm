"""REQ-REPORT-8538, REQ-VERIFY-8538: inventory never implies installed arithmetic."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import nine_input_board_boundary_8538 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_absent_and_rehashed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8538-BOUNDARY: missing operands and repaired summaries stay distinct."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["nine_input_cost_reference"]["qualified"] is False
    assert value["conditional_acceleration_ceiling"] is None
    assert value["intended_count"] == 21 and value["completed_count"] == 21
    assert value["independent_count"] == 0 and not value["deployment_ready"]
    assert not value["device_commands_executed"] and not value["MODEL_SPECS"]
    assert set(value) <= set(value["field_principles"])
    assert all(
        {"check", "upstream", "path", "hash", "field", "operator", "expected", "observed"} <= set(g)
        for g in value["gate_check_summary"]
    )
    atomic_json(output, value)
    assert e.replay(output) and not e.replay(tmp_path / "missing")
    value["board_boundary_recorded_score"] = 999
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["history"]["qualified"] = True
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_original_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8538-BOUNDARY: independent original seals retain physical obligations."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["board_boundary_recorded_score"] == 1
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert len(value["missing_source_hashes"]) == 2
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert value["polarfire_workload_validated"]
    assert not value["polarfire_graduation"]["fpga_fabric_acceleration"]
    assert all(not r["device_supported"] for r in value["board_operation_rows"])
    assert {r["operation"] for r in value["board_operation_rows"]} == set(e.OPERATIONS)
    assert value["conditional_acceleration_ceiling"] is None
    assert not value["NPU_qualified"] and not value["TSU_qualified"]


def test_cost_spans_and_rejections(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8538-BOUNDARY: complete disjoint clocks retain startup and transfer."""
    spans = [
        dict(operation=op, start_ns=i * 10, end_ns=(i + 1) * 10)
        for i, op in enumerate(e.COST_OPERATIONS)
    ]
    result = e.ceiling(spans)
    assert result["complete_clock_ns"] == 90
    assert result["eligible_clock_ns"] == 30
    assert result["ceiling"] == 1.5
    assert result["installed_operations_ceiling"] == 1.0
    for bad in [
        [],
        spans[:-1],
        [dict(r, start_ns=0) for r in spans],
        [dict(r, end_ns=True) for r in spans],
    ]:
        with pytest.raises(ValueError, match="disjoint_complete_spans"):
            e.ceiling(bad)
    atomic_json(tmp_path / e.COST, dict(nine_input_inference_cost_ready_score=0))
    costs = e.costs(tmp_path)
    assert costs["checks"][0]["observed"] == 0
    atomic_json(tmp_path / e.COST, dict(nine_input_inference_cost_ready_score=1))
    assert e.costs(tmp_path)["checks"][0]["field"] == "required_checks_passed"


def test_private_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8538-CLI: actual children publish privately and reject invalid invocations."""
    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--private-e2e", "--root", str(tmp_path / "absent"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--private-e2e"], 2),
        (["--date", "20261014"], 2),
    ]:
        child = subprocess.run(cli + args, capture_output=True, timeout=180, check=False)
        assert child.returncode == expected, child.stdout + child.stderr


def fixture_cost(tmp_path: Path) -> Path:
    """SCENARIO-REPORT-8538-BOUNDARY: private qualified spans exercise the optional path."""
    from carnot.reporting.primary_publication import publish_primary

    spans = [
        dict(operation=op, start_ns=i * 10, end_ns=(i + 1) * 10)
        for i, op in enumerate(e.COST_OPERATIONS)
    ]
    primitive = tmp_path / "spans.json"
    atomic_json(primitive, dict(spans=spans))
    terminal = tmp_path / "cost-terminal.json"
    path = tmp_path / e.COST
    value = dict(
        experiment_id=8534,
        task_id="exp8534-nine-input-inference-cost",
        run_date="20261015",
        honest_verdict="complete_null_private_cost",
        verdict_class="null",
        nine_input_inference_cost_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(terminal),
        disjoint_complete_spans_reference=e.reference(primitive),
    )
    publication = publish_primary(path, value, lambda candidate: dict(passed=True))
    atomic_json(terminal, dict(publication=publication))
    return path


def test_qualified_cost_and_terminal_rejection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8538-BOUNDARY: terminal binding and complete-span rejection are real."""
    from carnot.reporting import nine_input_board_evidence_8538 as source

    path = fixture_cost(tmp_path)
    current = e.costs(tmp_path)
    assert current["qualified"] and current["conditional_ceiling"]["ceiling"] == 1.5
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["board_operation_rows"][0]["cost_ns"] == 10
    assert value["acquisition_decision"]["measured_dominant_span"] == "startup"
    with patch.object(source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert e.costs(tmp_path)["checks"][0]["field"] == "bound_terminal_bytes"
    path.write_text(path.read_text() + " ")
    assert e.costs(tmp_path)["checks"][0]["field"] == "bound_terminal_bytes"
    fixture_cost(tmp_path)
    with patch.object(source, "ceiling", side_effect=ValueError("bad spans")):
        assert e.costs(tmp_path)["checks"][0]["field"] == "disjoint_complete_spans_reference"
    zero = [dict(operation=op, start_ns=0, end_ns=0) for op in e.COST_OPERATIONS]
    with pytest.raises(ValueError, match="disjoint_complete_spans"):
        e.ceiling(zero)


def test_original_rejection_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8538-BOUNDARY: independent original seals reject changed support rows."""
    from carnot.reporting import nine_input_board_evidence_8538 as source
    from carnot.reporting.current_work_receipt import sha256_file

    original = json.loads((e.ROOT / source.SOURCES[0][0]).read_bytes())
    path = tmp_path / "original.json"
    atomic_json(path, original)
    with patch.object(source, "SOURCES", [(path.name, "sha256:wrong", "20261014")]):
        assert source.history(tmp_path)["checks"][0]["field"] == "original_sha256"
    for updates, field in [
        (dict(run_date="20261015"), "run_date"),
        (
            dict(kv260_supported_operations=[dict(operation="spline", k_max=9)]),
            "supported_operation_rows",
        ),
    ]:
        atomic_json(path, dict(original, **updates))
        with patch.object(source, "SOURCES", [(path.name, sha256_file(path), "20261014")]):
            with patch.object(
                source, "read_bound_sidecar", return_value=dict(report=dict(passed=True))
            ):
                assert source.history(tmp_path)["checks"][0]["field"] == field
    atomic_json(path, original)
    with patch.object(source, "SOURCES", [(path.name, sha256_file(path), "20261014")]):
        with patch.object(
            source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))
        ):
            assert not source.history(tmp_path)["qualified"]


def test_manifest_union_and_errors(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8538-CLI: collected original node union is exact and disjoint."""
    from carnot.reporting import nine_input_board_runner_8538 as runner

    plan = runner.manifest(tmp_path / "plan")
    for collected in [r for r in plan if "original_node_ids" in r]:
        executed = [
            n
            for row in plan
            if row.get("suite") == collected["suite"]
            for n in row.get("collected_node_ids", [])
        ]
        assert executed == collected["original_node_ids"]
        assert len(executed) == len(set(executed))
    with patch.object(runner, "child", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="test_collection_failed"):
            runner.manifest(tmp_path / "failed")
    log = tmp_path / "empty.log"
    log.write_text("")
    with patch.object(runner, "child", return_value=dict(passed=True, stdout_path=str(log))):
        with pytest.raises(ValueError, match="empty_or_duplicate_collection"):
            runner.manifest(tmp_path / "empty")
