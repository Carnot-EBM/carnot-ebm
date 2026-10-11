"""REQ-REPORT-8399 / REQ-VERIFY-8399: costs cannot confer an absent kernel."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_operation_boundary_8399 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_complete_stages() -> None:
    """SCENARIO-REPORT-8399-BOUNDARY: absence and a measured zero stay distinct."""
    mapped, fraction, missing = e.operation_map([])
    assert missing == e.OPERATIONS and fraction is None
    assert all(r["cost_ns"] is None for r in mapped)
    rows = [dict(operation=op, cost_ns=0) for op in e.OPERATIONS]
    assert e.operation_map(rows)[1] is None
    rows[0]["cost_ns"] = 3
    assert e.operation_map(rows)[1] == 0
    assert e.operation_map(rows[:-1])[1] is None
    assert all(not r["kv260_supported"] for r in e.operation_map(rows)[0])
    for invalid in [[dict(operation="unknown", cost_ns=1)], rows + rows[:1]]:
        with pytest.raises(ValueError, match="operation_trace"):
            e.operation_map(invalid)


@pytest.mark.parametrize("cost", [-1, True, None, float("inf"), float("nan")])
def test_invalid_cost(cost: object) -> None:
    """REQ-REPORT-8399: malformed clocks cannot enter a transaction denominator."""
    with pytest.raises(ValueError, match="operation_trace"):
        e.operation_map([dict(operation=e.OPERATIONS[0], cost_ns=cost)])


def test_measure_replay_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8399-REPLAY: recompute the exact original board dispatch."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["experiment_id"] == 8399 and value["run_date"] == "20261011"
    assert value["verdict_class"] == "blocked" and value["board_reader_ready_score"] == 1
    assert value["current_contract_ready_score"] == 1
    assert value["polarfire_workload_validated"] and value["compatible_cost_fraction"] is None
    assert value["polarfire_graduation"]["dispatch_sha256"] == e.DISPATCH_PIN
    assert value["kv260_execution_ready_score"] == value["current_device_execution_count"] == 0
    assert value["intended_count"] == 4 and value["independent_count"] == 0
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output)
    for mutate in [
        lambda w: w["contract"].update(task_sha256="wrong"),
        lambda w: w["history"]["graduation"].update(dispatch_sha256="wrong"),
        lambda w: w.update(reader_ready=False),
    ]:
        changed = deepcopy(work)
        mutate(changed)
        atomic_json(raw / "measurement.json", changed)
        atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
        assert not e.replay(output)
    atomic_json(raw / "measurement.json", work)
    changed = deepcopy(value)
    changed["board_reader_ready_score"] = 0
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_missing_and_changed_sources(tmp_path: Path) -> None:
    """REQ-REPORT-8399: external absence blocks; changed historical bytes never qualify."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert not value["board_reader_ready_score"] and not value["polarfire_workload_validated"]
    atomic_json(output, value)
    assert e.replay(output)
    for constant in ["READER_PIN", "TERMINAL_PIN", "PROTOCOL_PIN", "DISPATCH_PIN"]:
        with patch.object(e, constant, "wrong"):
            checked = e.operands(e.ROOT, tmp_path / constant)
            assert any(not g["passed"] for g in checked["checks"])


def test_cost_qualification(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8399-BOUNDARY: optional costs need a qualified producer field."""
    path = tmp_path / (e.SOURCES[0] + ".json")
    for value in [None, dict(verdict_class="blocked"), dict(verdict_class="null")]:
        work = dict(checks=[])
        with patch.object(e, "_probe", return_value=value) as mocked:
            result = e.probe(path, tmp_path, work, None)
            assert mocked.call_args.args[-1] == "python_cost_ready_score"
        assert result == (value if value and value["verdict_class"] == "null" else None)
    with patch.object(e, "_probe", return_value={}):
        assert e.probe(tmp_path / "historical.json", tmp_path, dict(checks=[]), "ready") == {}


def test_real_cli_and_consumers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8399-CONTROLS: actual children check publication and rejection."""
    from carnot.reporting.primary_publication import reader_receipt

    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    for args, code in [
        (["--private-e2e", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--private-e2e"], 2),
        (["--date", "20261009"], 2),
    ]:
        ran = subprocess.run([*cli, *args], capture_output=True, text=True, timeout=120)
        assert ran.returncode == code, ran.stdout + ran.stderr
    value = json.loads(output.read_bytes())
    assert (
        value["required_checks_passed"]
        and Path(value["terminal_validation_sidecar_path"]).is_file()
    )
    assert reader_receipt(e.TASK, tmp_path, field="board_reader_ready_score")["passed"]
    value["current_contract_ready_score"] = 0
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    ran = subprocess.run([*cli, "--cold-replay", str(output)], timeout=120)
    assert ran.returncode == 1


def test_runner_failure_and_scoped_manifest(tmp_path: Path) -> None:
    """REQ-VERIFY-8399: owned failure disqualifies; global health remains separate."""
    from carnot.reporting import board_operation_runner_8399 as runner

    plan = runner.manifest(tmp_path)
    assert any("E2E018" in row["name"] for row in plan)
    assert all(row["scope"] == "owned" for row in plan)
    failed = [
        dict(
            name="deliberate_error",
            argv=[sys.executable, "-u", "-c", "raise SystemExit(3)"],
            deadline=10,
            expected=0,
            scope="owned",
        )
    ]
    output = tmp_path / (e.NAME + ".json")
    with patch.object(runner, "manifest", return_value=failed):
        assert runner.run(tmp_path / "missing", output, tmp_path) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and not value["board_reader_ready_score"]
    assert value["validation_receipts"][0]["exit_code"] == 3
    work = json.loads(e.base.checked(value["work_reference"]).read_bytes())
    result = e.build(
        work,
        [dict(passed=True, scope="owned"), dict(passed=False, scope="global")],
        Path(value["work_reference"]["path"]).parent,
        output,
    )
    assert result["required_checks_passed"] and result["verdict_class"] == "blocked"
    assert len(result["repository_health"]) == 1


def test_source_drift_and_dispatch_rejection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8399-REPLAY: edited aliases and dispatch are rejected independently."""
    with patch.object(
        e.prior.old, "board_history", return_value=dict(graduation=dict(dispatch_sha256="bad"))
    ):
        work = e.operands(e.ROOT, tmp_path / "dispatch")
        assert not work["reader_ready"]
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    atomic_json(output, e.build(work, [dict(passed=True)], raw, output))
    source = Path(work["refs"][-1]["path"])
    source.write_bytes(b"changed")
    assert not e.replay(output)
