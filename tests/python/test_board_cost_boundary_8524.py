"""REQ-REPORT-8524, REQ-VERIFY-8524: current clocks retain historical board limits."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_cost_boundary_8524 as e
from carnot.reporting import board_cost_runner_8524 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary


def test_cost_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8524-COST: each rejection names its exact observed field."""
    path = tmp_path / e.COST
    for fields, field, observed in [
        ({}, "service_cost_ready_score", None),
        ({"service_cost_ready_score": 0}, "service_cost_ready_score", 0),
        ({"service_cost_ready_score": 1}, "required_checks_passed", None),
        (
            {"service_cost_ready_score": 1, "required_checks_passed": False},
            "required_checks_passed",
            False,
        ),
    ]:
        atomic_json(path, fields)
        current = e.costs(tmp_path, tmp_path / "raw")["current"]
        assert not current["qualified"] and current["missing_field"] == field
        assert current["observed"] == observed
        assert current["checks"][0]["observed"] == observed
    assert e.costs(tmp_path / "absent", tmp_path / "missing")["current"]["observed"] is None


def fixture_cost(tmp_path: Path, **updates: object) -> Path:
    """SCENARIO-REPORT-8524-COST: private original operands allow precise negative controls."""
    primitive = tmp_path / "primitive.json"
    atomic_json(
        primitive,
        dict(
            arm="append",
            start_ns=1,
            cold_end_ns=3,
            initialized_ns=6,
            end_ns=21,
            requests=[dict(end_ns=19)],
        ),
    )
    work = tmp_path / "cost-work.json"
    atomic_json(
        work,
        dict(
            rows=[
                dict(
                    warmup=False,
                    status="completed",
                    arm="append",
                    primitive_reference=e.reference(primitive),
                    unit_id="a",
                )
            ]
        ),
    )
    primary, terminal = tmp_path / e.COST, tmp_path / "terminal.json"
    value = dict(
        experiment_id=8520,
        task_id="exp8520-append-service-cost",
        run_date="20261014",
        service_cost_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        honest_verdict="complete_null_cost",
        verdict_class="null",
        work_reference=e.reference(work),
        terminal_validation_sidecar_path=str(terminal),
    )
    value.update(updates)
    publication = publish_primary(primary, value, lambda path: dict(passed=True))
    atomic_json(terminal, dict(publication=publication))
    return primary


def test_qualified_cost_and_rejections(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8524-COST: disjoint full spans cannot invent detailed operation timing."""
    primary = fixture_cost(tmp_path)
    with patch.object(e, "child", return_value=dict(passed=True)):
        current = e.costs(tmp_path, tmp_path / "raw")["current"]
        assert current["qualified"] and current["missing_field"] is None
        assert sum(r["cost_ns"] for r in current["rows"]) == 20
        assert {r["operation"]: r["cost_ns"] for r in current["rows"]} == dict(
            cold_reconstruction=2, startup=3, transactions_and_response=13, tail_checkpoint=2
        )
    with patch.object(e, "child", return_value=dict(passed=False, exit_code=9)):
        current = e.costs(tmp_path, tmp_path / "rejected")["current"]
        assert (
            current["missing_field"] == "service_cost_primitive_replay" and not current["qualified"]
        )
    fixture_cost(tmp_path, flagged_adversarial=True)
    assert (
        e.costs(tmp_path, tmp_path / "flagged")["current"]["missing_field"] == "flagged_adversarial"
    )
    fixture_cost(tmp_path, run_date="20261013")
    assert e.costs(tmp_path, tmp_path / "date")["current"]["missing_field"] == "run_date"
    primary = fixture_cost(tmp_path)
    value = json.loads(primary.read_bytes())
    atomic_json(primary, dict(value, marker="changed bytes"))
    assert (
        e.costs(tmp_path, tmp_path / "stale")["current"]["missing_field"] == "bound_terminal_bytes"
    )
    fixture_cost(tmp_path)
    with patch.object(e, "child", return_value=dict(passed=True)):
        with patch.object(e, "clock_rows", side_effect=ValueError("clock")):
            assert (
                e.costs(tmp_path, tmp_path / "badclock")["current"]["missing_field"]
                == "complete_clock_spans"
            )


def test_clock_rows() -> None:
    """SCENARIO-REPORT-8524-COST: retain warmup exclusion and clock rejection paths."""
    assert (
        e.clock_rows(dict(rows=[dict(warmup=True), dict(warmup=False, status="censored")]), [])
        == []
    )
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError, match="complete_clock"):
            e.clock_rows(
                dict(rows=[dict(warmup=False, status="completed", primitive_reference={})]),
                [],
                read=lambda ref: dict(
                    start_ns=1,
                    cold_end_ns=bad,
                    initialized_ns=5,
                    end_ns=8,
                    requests=[dict(end_ns=7)],
                    arm="append",
                ),
            )


def test_missing_rehashed_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8524-BOUNDARY: missing upstream blocks and repaired summaries reject."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 4
    assert value["run_date"] == "20261014" and not value["deployment_ready"]
    assert not any(value["model_invocation_counts"].values())
    assert not value["gatemate_bitstream_flashed"] and not value["device_commands_executed"]
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output) and not e.replay(tmp_path / "missing")
    value["board_boundary_recorded_score"] = 999
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    changed = deepcopy(work)
    changed["cost"]["current"]["observed"] = 0
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_original_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8524-BOUNDARY: current authority cannot erase historical physical limits."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, tmp_path)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["board_boundary_recorded_score"] == 1
    assert work["contract"]["activated"] and work["cost"]["current"]["qualified"]
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert len(value["missing_source_hashes"]) == 2 and value["polarfire_workload_validated"]
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert value["conditional_acceleration_ceiling"]["current_append"]["ceiling"] == 1
    assert value["conditional_acceleration_ceiling"]["input_count"] == 5
    assert not value["NPU_qualified"] and not value["TSU_qualified"]
    assert all(not r["device_supported"] for r in value["board_operation_rows"])
    assert value["append_cost_boundary"]["fallback_producer"] is None
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    atomic_json(output, value)
    assert e.replay(output)


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8524-CLI: real private children retain original rejection exit codes."""
    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--private-e2e", "--root", str(tmp_path / "missing"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 7),
        (["--private-e2e"], 2),
        (["--date", "20261013"], 2),
    ]:
        result = subprocess.run(cli + args, capture_output=True, timeout=180, check=False)
        assert result.returncode == expected, result.stdout + result.stderr
    plan = runner.manifest(tmp_path / "manifest")
    for row in plan:
        if "original_node_ids" in row:
            shards = [
                r["collected_node_ids"]
                for r in plan
                if r.get("suite") == row["suite"] and not r["name"].startswith("collect_")
            ]
            assert [n for group in shards for n in group] == row["original_node_ids"]


def test_historical_scope(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8524-CLI: historical shard scope keeps the original producer seal."""
    request = tmp_path / "request.json"
    atomic_json(request, [])
    with pytest.raises(ValueError, match="historical_node_scope"):
        runner.historical(request)


def test_empty_and_unbound_cost(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8524-COST: an empty span reduction or failed terminal report rejects."""
    fixture_cost(tmp_path)
    with patch.object(e, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert e.costs(tmp_path, tmp_path / "sidecar")["current"]["observed"] is False
    with patch.object(e, "child", return_value=dict(passed=True)):
        with patch.object(e, "clock_rows", return_value=[]):
            assert not e.costs(tmp_path, tmp_path / "empty")["current"]["qualified"]
    with pytest.raises(ValueError, match="missing_complete_clocks"):
        e.clock_rows(dict(rows=[dict(warmup=True, status="completed")]), [])


def test_rejection_expected_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8524-COST: rejected identity fields retain their exact expected value."""
    fixture_cost(tmp_path, flagged_adversarial=True)
    gate = e.costs(tmp_path, tmp_path / "flag")["current"]["checks"][0]
    assert gate["expected"] is False and gate["observed"] is True
    fixture_cost(tmp_path, run_date="20261013")
    gate = e.costs(tmp_path, tmp_path / "date")["current"]["checks"][0]
    assert gate["expected"] == "20261014" and gate["observed"] == "20261013"
