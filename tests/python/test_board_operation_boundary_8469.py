"""REQ-REPORT-8469 and REQ-VERIFY-8469: evidence accounting grants no deployment."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_operation_boundary_8469 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_mapping() -> None:
    """SCENARIO-REPORT-8469-OPERANDS: zero clocks are observed; missing clocks stay null."""
    rows = e.operation_mapping([dict(operation="fsync", cost_ns=0)], None)
    assert {r["board"] for r in rows} == {"KV260", "PolarFire", "GateMate"}
    assert all(r["cost_ns"] == 0 and not r["device_supported"] for r in rows)
    assert all(r["missing_reason"] for r in e.operation_mapping([], "service_cost_ready_score"))
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            e.operation_mapping([dict(operation="fsync", cost_ns=bad)], None)


def test_missing_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8469-REPLAY: repaired hashes cannot manufacture source authority."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 4
    assert value["censored_count"] == 4 and value["run_date"] == "20261013"
    assert not any(value["model_invocation_counts"].values())
    assert value["device_commands_executed"] == [] and not value["gatemate_bitstream_flashed"]
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output) and not e.replay(tmp_path / "absent.json")
    value["board_boundary_recorded_score"] = 999
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    changed = deepcopy(work)
    changed["reader"]["gatemate"]["qualified"] = True
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_original_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8469-OPERANDS: the original authority seals remain independently valid."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    h = work["reader"]
    assert h["qualified"] and h["kv260"]["k_max"] == 5
    assert h["polarfire"]["polarfire_workload_validated"]
    assert h["gatemate"]["qualified"]
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", output)
    assert value["board_boundary_recorded_score"] == 1
    assert value["gatemate_obligation_recorded_score"] == 1
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert len(value["missing_source_hashes"]) == 2
    assert value["polarfire_graduation"] == h["polarfire"]
    atomic_json(output, value)
    assert e.replay(output)
    with patch.object(e.gm, "replay", return_value=False):
        assert not e.gatemate(e.ROOT, tmp_path / "failed")["qualified"]


def test_optional_cost(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8469-OPERANDS: optional complete clocks require genuine cold replay."""
    assert e.costs(tmp_path, tmp_path / "raw")["observed"] is None
    path = tmp_path / e.COST
    atomic_json(path, dict(service_cost_ready_score=0))
    assert e.costs(tmp_path, tmp_path / "raw")["observed"] == 0
    atomic_json(path, dict(service_cost_ready_score=1, work_reference={}))
    with patch.object(e, "child", return_value=dict(passed=False)):
        assert not e.costs(tmp_path, tmp_path / "raw")["qualified"]
    with patch.object(e, "child", return_value=dict(passed=True)):
        assert not e.costs(tmp_path, tmp_path / "raw")["qualified"]


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8469-CLI: actual private children test consumers and error exits."""
    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--private-e2e", "--root", str(tmp_path / "missing"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 7),
        (["--private-e2e"], 2),
        (["--date", "20261012"], 2),
    ]:
        result = subprocess.run(cli + args, capture_output=True, timeout=180, check=False)
        assert result.returncode == expected, result.stdout + result.stderr
    from carnot.reporting.primary_publication import reader_receipt

    assert reader_receipt(e.TASK, tmp_path, field="board_boundary_recorded_score", expected=0)[
        "passed"
    ]
    from carnot.reporting import board_operation_runner_8469 as runner

    assert len(runner.manifest(tmp_path / "manifest")) == 10


def test_empty_qualified_cost(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8469-OPERANDS: a readiness scalar without primitives cannot supply timing."""
    work = tmp_path / "cost-work.json"
    atomic_json(work, dict(primitives=[], rows=[]))
    primary = tmp_path / e.COST
    atomic_json(
        primary,
        dict(
            experiment_id=8466,
            task_id="exp8466-journal-service-cost",
            honest_verdict="complete_null_service_cost",
            verdict_class="null",
            service_cost_ready_score=1,
            work_reference=e.reference(work),
        ),
    )
    with patch.object(e, "child", return_value=dict(passed=True)):
        result = e.costs(tmp_path, tmp_path / "cost-raw")
    assert not result["qualified"] and result["missing_field"] == "work_reference.primitives[]"
    assert not result["checks"][0]["passed"]
