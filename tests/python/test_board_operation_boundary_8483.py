"""REQ-REPORT-8483 and REQ-VERIFY-8483: preserve board evidence without device work."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_operation_boundary_8483 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_mapping() -> None:
    """SCENARIO-REPORT-8483-OPERANDS: absent clocks cannot replace measured zero."""
    rows = e.operation_mapping([dict(operation="file_sync", cost_ns=0)], None)
    assert {r["board"] for r in rows} == {"KV260", "PolarFire", "GateMate"}
    assert all(r["cost_ns"] == 0 and not r["device_supported"] for r in rows)
    assert all(r["missing_reason"] for r in e.operation_mapping([], "missing_stage"))
    mixed = e.operation_mapping(
        [
            dict(operation="append_write", cost_ns=None, missing_reason="stage_ns.append_write"),
            dict(operation="file_sync", cost_ns=0),
        ],
        None,
    )
    assert len(mixed) == 6 and sum(r["cost_ns"] is None for r in mixed) == 3
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            e.operation_mapping([dict(operation="file_sync", cost_ns=bad)], None)


def test_missing_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8483-REPLAY: missing authority blocks while owned failure disqualifies."""
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
    changed["prior"]["qualified"] = True
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_original_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8483-OPERANDS: pinned board originals survive changed current authority."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    assert work["prior"]["qualified"] and work["reader"]["kv260"]["k_max"] == 5
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", output)
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert len(value["missing_source_hashes"]) == 2
    assert value["polarfire_workload_validated"]
    assert value["append_cost_boundary"]["fallback_run_date"] == "20261013"
    assert value["append_cost_boundary"]["missing_field"] == "service_cost_ready_score"
    assert value["anechoic_lead"]["carnot_speed_claim"] is False
    atomic_json(output, value)
    assert e.replay(output)
    with patch.object(e.old, "replay", return_value=False):
        assert not e.prior(e.ROOT, tmp_path / "rejected")["qualified"]


def test_optional_cost(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8483-OPERANDS: current append clocks need independent bound evidence."""
    assert e.costs(tmp_path, tmp_path / "raw")["current"]["observed"] is None
    path = tmp_path / e.COST
    atomic_json(path, dict(service_cost_ready_score=0))
    assert e.costs(tmp_path, tmp_path / "raw")["current"]["observed"] == 0
    atomic_json(path, dict(service_cost_ready_score=1))
    assert not e.costs(tmp_path, tmp_path / "raw")["current"]["qualified"]


def test_append_primitive_qualification(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8483-OPERANDS: bound sidecars and nonnegative primitive clocks govern import."""
    from carnot.reporting.primary_publication import publish_primary

    primary = tmp_path / e.COST
    primitive, work, terminal = (
        tmp_path / "primitive.json",
        tmp_path / "work.json",
        tmp_path / "terminal.json",
    )

    def candidate(stages: list[dict[str, object]], passed: bool = True) -> None:
        atomic_json(primitive, dict(arm="append", requests=stages))
        atomic_json(work, dict(primitives=[e.reference(primitive)]))
        value = dict(
            experiment_id=8480,
            task_id="exp8480-append-service-cost",
            honest_verdict="complete_null_append",
            verdict_class="null",
            service_cost_ready_score=1,
            required_checks_passed=passed,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal),
            work_reference=e.reference(work),
        )
        publication = publish_primary(primary, value, lambda path: dict(passed=True))
        atomic_json(terminal, dict(publication=publication))

    candidate([dict(stage_ns=dict(append_write=0, file_sync=7)), dict(stage_ns=dict(file_sync=2))])
    current = e.costs(tmp_path, tmp_path / "qualified")["current"]
    assert current["qualified"] and current["missing_field"] is None
    assert {r["operation"]: r["cost_ns"] for r in current["rows"]} == dict(
        append_write=0, file_sync=9
    )
    candidate([], passed=False)
    assert not e.costs(tmp_path, tmp_path / "failed")["current"]["qualified"]
    candidate([])
    assert not e.costs(tmp_path, tmp_path / "empty")["current"]["qualified"]
    candidate([dict(stage_ns=dict(file_sync=-1))])
    assert not e.costs(tmp_path, tmp_path / "negative")["current"]["qualified"]


def test_optional_operand_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8483-REPLAY: repaired work hashes cannot replace current absence with zero."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw, tmp_path)
    work["cost"]["current"]["observed"] = 0
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, [dict(passed=True)], raw, output))
    assert not e.replay(output)


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8483-CLI: private children retain actual consumer and rejection exits."""
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
    from carnot.reporting import board_operation_runner_8483 as runner

    plan = runner.manifest(tmp_path / "manifest")
    assert any(r["name"].startswith("collect_") for r in plan)
    assert len(plan) > 10
    for row in plan:
        if "original_node_ids" in row:
            shards = [
                r["collected_node_ids"]
                for r in plan
                if r.get("suite") == row["suite"] and not r["name"].startswith("collect_")
            ]
            assert [n for group in shards for n in group] == row["original_node_ids"]


def test_collection_rejections(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8483-CLI: failed or ambiguous collection cannot qualify omitted tests."""
    from carnot.reporting import board_operation_runner_8483 as runner

    with patch.object(runner, "child", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="test_collection_failed"):
            runner.manifest(tmp_path / "failed")
    log = tmp_path / "empty.log"
    log.write_text("")
    with patch.object(runner, "child", return_value=dict(passed=True, stdout_path=str(log))):
        with pytest.raises(ValueError, match="empty_or_duplicate_collection"):
            runner.manifest(tmp_path / "empty")


def test_historical_source_pin(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8483-CLI: original tests require their independent producer seal."""
    from carnot.reporting import board_operation_runner_8483 as runner

    request = tmp_path / "request.json"
    atomic_json(request, [])
    with patch.object(e, "PRIOR_PIN", "sha256:" + "0" * 64):
        with pytest.raises(ValueError):
            runner.historical(request)
    atomic_json(request, ["tests/python/test_board_operation_boundary_8483.py::test_mapping"])
    with pytest.raises(ValueError, match="historical_node_scope"):
        runner.historical(request)
