"""REQ-REPORT-8385 / REQ-VERIFY-8385: preserve scope through real private replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_operation_evidence_8385 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_map() -> None:
    """SCENARIO-REPORT-8385-CLOSURE: a missing denominator cannot imply measured zero."""
    mapped, fraction, missing = e.operation_map([])
    assert missing == e.OPERATIONS and fraction is None
    assert all(r["cost_ns"] is None for r in mapped)
    rows = [dict(operation=op, cost_ns=0) for op in e.OPERATIONS]
    assert e.operation_map(rows)[1] is None
    rows[0]["cost_ns"] = 3
    assert e.operation_map(rows)[1] == 0
    assert all(not r["kv260_supported"] for r in e.operation_map(rows)[0])


@pytest.mark.parametrize("cost", [-1, True, None, float("inf"), float("nan")])
def test_bad_cost(cost: object) -> None:
    """SCENARIO-REPORT-8385-CLOSURE: invalid costs never enter the denominator."""
    with pytest.raises(ValueError, match="operation_trace"):
        e.operation_map([dict(operation=e.OPERATIONS[0], cost_ns=cost)])


def test_duplicate_unknown() -> None:
    """REQ-REPORT-8385: duplicate and unknown operations are not transaction evidence."""
    for rows in [
        [dict(operation="unknown", cost_ns=1)],
        [dict(operation="updates", cost_ns=1)] * 2,
    ]:
        with pytest.raises(ValueError):
            e.operation_map(rows)


def test_original_failure_and_closure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8385-CLOSURE: reach the exact unchanged original rejection line."""
    original = e.ROOT / "results/experiment_8371_v721_hardware_operation_boundary.json"
    original_bytes = original.read_bytes()
    value = json.loads(original_bytes)
    work = json.loads(e.base.checked(value["work_reference"]).read_bytes())
    work["code_config_hashes"] = [
        e.base.reference(Path(r["path"])) for r in work["code_config_hashes"]
    ]
    raw = tmp_path / "historical"
    atomic_json(raw / "measurement.json", work)
    candidate = tmp_path / (e.old.NAME + ".json")
    atomic_json(candidate, e.old.build(work, value["validation_receipts"], raw, candidate))
    seen = []
    old_hash = e.old.base.sha256_file

    def observe(path: Path) -> str:
        seen.append(str(path))
        return str(old_hash(path))

    with patch.object(e.old.base, "sha256_file", observe):
        assert not e.old.replay(candidate)
    assert value["verdict_class"] == "disqualified"
    assert value["source_artifact_hashes"][0]["original_path"] in seen
    refs = value["source_artifact_hashes"]
    with e.source_aliases(refs):
        assert e.old.replay(candidate)
    assert original.read_bytes() == original_bytes


def test_measure_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8385-CLOSURE: immutable operands defeat rehashed primitive fraud."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["polarfire_workload_validated"]
    assert value["compatible_cost_fraction"] is None
    assert value["kv260_execution_ready_score"] == value["current_device_execution_count"] == 0
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(value)
    changed["board_reader_ready_score"] = 1 - changed["board_reader_ready_score"]
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert not e.replay(output)
    changed_work = deepcopy(work)
    changed_work["history"]["graduation"]["output_sha256"] = "wrong"
    atomic_json(raw / "measurement.json", changed_work)
    atomic_json(output, e.build(changed_work, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_absent_operands(tmp_path: Path) -> None:
    """REQ-REPORT-8385: missing authority and board receipts remain external blocks."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    assert not work["history"]
    assert any(not g["passed"] for g in work["checks"])
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [dict(passed=True)], tmp_path / "raw", output))
    assert e.replay(output)


def test_real_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8385-CLI: valid, absent, deliberate error and tamper use real children."""
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
    assert value["required_checks_passed"]
    assert Path(value["terminal_validation_sidecar_path"]).is_file()


def test_qualified_and_partial_branches(tmp_path: Path) -> None:
    """REQ-REPORT-8385: valid rows map independently; partial inputs retain their disposition."""
    root, raw = tmp_path / "root", tmp_path / "raw"
    trace = tmp_path / "trace.json"
    rows = [dict(operation=op, cost_ns=1) for op in e.OPERATIONS]
    atomic_json(trace, dict(operation_rows=rows))
    primary = root / "results" / (e.SOURCES[0] + ".json")
    atomic_json(primary, dict(verdict_class="partial"))
    value = dict(
        experiment_id=8378,
        milestone=e.MILESTONE,
        verdict_class="null",
        operation_level_workload_reference=e.base.reference(trace),
    )
    with patch.object(e.base, "probe", return_value=value):
        work = e.operands(root, raw)
        assert work["branches"][0]["rows"] == rows
        for changed in [
            dict(value, experiment_id=0),
            dict(value, milestone="old"),
            dict(value, operation_level_workload_reference=None),
        ]:
            with patch.object(e.base, "probe", return_value=changed):
                assert not e.operands(root, raw)["branches"][0]["rows"]
    with patch.object(e.base, "probe", return_value=None):
        assert e.operands(root, raw)["branches"][0]["disposition"] == "partial"
        primary.write_text("[]")
        assert e.operands(root, raw)["branches"][0]["disposition"].startswith("unreadable:")


def test_task_and_protocol_rejection(tmp_path: Path) -> None:
    """REQ-REPORT-8385: a changed active task or protocol never grants authority."""
    with patch.object(e.authority, "authority", return_value=dict(tasks=[dict(id=e.TASK)])):
        assert "active_task_digest" in str(e.operands(tmp_path, tmp_path / "bad-task")["checks"])
    with patch.object(e.authority.previous.legacy.base, "PIN", "wrong"):
        assert "preserved_protocol_hash" in str(
            e.operands(e.ROOT, tmp_path / "bad-protocol")["checks"]
        )


def test_replay_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8385-CLOSURE: every reduced primitive remains bound to independent bytes."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    changes = [
        lambda w: w["contract"].update(task_sha256="wrong"),
        lambda w: w["branches"][0].update(rows=[dict(operation="updates", cost_ns=10)]),
        lambda w: w["absent"].append("foreign"),
        lambda w: w["checks"][0].update(observed="wrong"),
    ]
    for change in changes:
        changed = deepcopy(work)
        change(changed)
        atomic_json(raw / "measurement.json", changed)
        atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
        assert not e.replay(output)
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=True)], raw, output)
    value["source_artifact_hashes"] = []
    atomic_json(output, value)
    assert not e.replay(output)


def test_runner_failure_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8385-CLI: actual owned child failure is separate from global health."""
    from carnot.reporting import board_operation_runner_8385 as runner

    plan = runner.manifest(tmp_path)
    assert any("E2E018_E2E021" in row["name"] for row in plan)
    assert sum(row["scope"] == "global" for row in plan) == 1
    plan = [
        dict(
            name="deliberate_error",
            argv=[sys.executable, "-u", "-c", "raise SystemExit(3)"],
            deadline=10,
            expected=0,
            scope="owned",
        )
    ]
    output = tmp_path / (e.NAME + ".json")
    with patch.object(runner, "manifest", return_value=plan):
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
