"""REQ-REPORT-8454 / REQ-VERIFY-8454: board scope survives missing journal clocks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_operation_boundary_8454 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_mapping() -> None:
    """SCENARIO-REPORT-8454-BOUNDARY: host clocks never become installed fabric support."""
    rows = [dict(operation=op, cost_ns=0, arm="journal") for op in e.OPERATIONS]
    mapped = e.operation_mapping(rows, "service_cost_ready_score")
    assert len(mapped) == 12 and all(not r["device_supported"] for r in mapped)
    assert all(r["cost_ns"] == 0 for r in mapped)
    missing = e.operation_mapping([], "service_cost_ready_score")
    assert all(r["cost_ns"] is None and r["missing_reason"] for r in missing)
    assert (
        e.operation_mapping([dict(operation="other", cost_ns=2)], "missing")[0]["operation"]
        == "other"
    )
    for bad in [-1, True, None, 1.5]:
        with pytest.raises(ValueError):
            e.operation_mapping([dict(operation="fsync", cost_ns=bad)], "missing")


def test_missing_and_rehashed_work(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8454-OPERANDS: fresh hashes cannot grant altered board support."""
    raw, root = tmp_path / "raw", tmp_path / "missing"
    work = e.measure(root, raw, tmp_path)
    out = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)], raw, out)
    assert value["verdict_class"] == "blocked"
    assert value["intended_count"] == 3 and value["censored_count"] == 3
    assert not any(value["model_invocation_counts"].values())
    assert value["device_commands_executed"] == []
    assert set(value) <= set(value["field_principles"])
    atomic_json(out, value)
    assert e.replay(out) and not e.replay(tmp_path / "absent")
    value["board_boundary_recorded_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(out, value)
    assert not e.replay(out)
    changed = deepcopy(work)
    changed["reader"].update(qualified=True, kv260=dict(k_max=99), polarfire={})
    atomic_json(raw / "measurement.json", changed)
    atomic_json(out, e.build(changed, [dict(passed=True)], raw, out))
    assert not e.replay(out)
    failed = e.build(work, [dict(passed=False)], raw, out)
    assert failed["verdict_class"] == "disqualified" and not failed["required_checks_passed"]


def test_original_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8454-OPERANDS: original authority accepts historical reader bytes."""
    work = e.measure(e.ROOT, tmp_path / "natural", tmp_path)
    result = work["reader"]
    assert result["qualified"] and result["kv260"]["k_max"] == 5
    assert result["polarfire"]["scope"] == "board_local_Linux_CPU_only"
    assert result["polarfire"]["polarfire_workload_validated"]
    assert result["host_bottleneck"]["qualified"]
    out = tmp_path / (e.NAME + ".json")
    atomic_json(out, e.build(work, [dict(passed=True)], tmp_path / "natural", out))
    assert e.replay(out)
    with patch.object(e.historical, "replay", return_value=False):
        assert not e.reader(e.ROOT, tmp_path / "failed")["qualified"]


def test_optional_absent_invalid_and_qualified(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8454-OPERANDS: optional cost qualification requires actual primitive clocks."""
    assert e.costs(tmp_path, tmp_path / "raw")["missing_field"] == "service_cost_ready_score"
    primary = tmp_path / e.COST
    atomic_json(primary, {})
    assert not e.costs(tmp_path, tmp_path / "raw")["qualified"]
    payload = dict(
        experiment_id=8451,
        task_id="exp8451-journal-service-cost",
        honest_verdict="complete_blocked_cost",
        verdict_class="blocked",
        service_cost_ready_score=0,
    )
    atomic_json(primary, payload)
    blocked = e.costs(tmp_path, tmp_path / "raw")
    assert blocked["observed"] == 0
    payload.update(
        required_checks_passed=True,
        flagged_adversarial=False,
        service_cost_ready_score=1,
        verdict_class="null",
        raw_shard_hashes=[],
        paired_repeat_rows=[dict(arm="journal", stage_ns=dict(journal_append=42))],
    )
    atomic_json(primary, payload)
    with (
        patch.object(e.base, "probe", return_value=payload),
        patch.object(e, "child", return_value=dict(passed=True)),
    ):
        qualified = e.costs(tmp_path, tmp_path / "raw")
        assert qualified["qualified"] and qualified["rows"][0]["cost_ns"] == 42
        payload["paired_repeat_rows"] = []
        atomic_json(primary, payload)
        assert (
            e.costs(tmp_path, tmp_path / "raw")["missing_field"] == "paired_repeat_rows[].stage_ns"
        )
    with (
        patch.object(e.base, "probe", return_value=payload),
        patch.object(e, "child", return_value=dict(passed=False)),
    ):
        assert not e.costs(tmp_path, tmp_path / "raw")["qualified"]


def test_resources_and_build(tmp_path: Path) -> None:
    """REQ-VERIFY-8454: failed resources and owned checks forbid readiness."""
    work = e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path)
    with patch.object(e.shutil, "which", return_value=None):
        assert not e.resources(tmp_path)["tools_available"]
    work["reader"] = dict(
        qualified=True,
        kv260=dict(k_max=5),
        polarfire=dict(scope="board_local_Linux_CPU_only"),
        models=[],
        refs=[],
        checks=[],
    )
    work["contract"]["activated"] = True
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["completed_count"] == 2 and value["board_reader_ready_score"] == 1
    assert value["board_boundary_recorded_score"] == 1 and value["verdict_class"] == "blocked"
    assert value["polarfire_graduation"]["scope"] == "board_local_Linux_CPU_only"
    work["adversarial_findings"] = [dict(kind="unknown", severity="warning")]
    assert not e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))[
        "required_checks_passed"
    ]


def test_cli_and_retained_coverage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8454-CLI: real child exits and Coverage.py bytes survive publication."""
    from carnot.reporting import board_operation_runner_8454 as r
    from carnot.reporting.primary_publication import reader_receipt

    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--private-e2e", "--root", str(tmp_path / "missing"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 7),
        (["--private-e2e"], 2),
        (["--date", "20261011"], 2),
    ]:
        result = subprocess.run([*cli, *args], capture_output=True, text=True, timeout=120)
        assert result.returncode == expected, result.stdout + result.stderr
    assert reader_receipt(e.TASK, tmp_path, field="board_boundary_recorded_score", expected=0)[
        "passed"
    ]
    private = tmp_path / "coverage"
    plan = r.manifest(private)
    assert any(p["name"] == "private_E2E021" for p in plan)
    config = private / "coverage.ini"
    cov = str(e.ROOT / ".venv/bin/coverage")
    rc = "--rcfile=" + str(config)
    commands = [
        dict(
            name="actual_child",
            argv=[cov, "run", rc, str(e.ROOT / e.CLI), "--deliberate-error"],
            expected=7,
            deadline=60,
            scope="owned",
        ),
        dict(
            name="combine",
            argv=[cov, "combine", rc, "--keep"],
            expected=0,
            deadline=30,
            scope="owned",
        ),
        dict(
            name="json",
            argv=[cov, "json", rc, "-o", str(private / "coverage.json")],
            expected=0,
            deadline=30,
            scope="owned",
        ),
    ]
    with patch.object(r, "manifest", return_value=commands):
        assert r.run(tmp_path / "missing", output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["newly_owned_coverage"] and value["raw_shard_hashes"]
    assert list(output.parent.glob("raw/*/invocations/*/coverage_shards/.coverage*"))
    commands[0].update(expected=0)
    with patch.object(r, "manifest", return_value=commands):
        assert r.run(tmp_path / "missing", output, private) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
