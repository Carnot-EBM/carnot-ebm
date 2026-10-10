"""REQ-REPORT-8371 / REQ-VERIFY-8371: private controls preserve hardware limits."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import hardware_operation_boundary_8371 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_typed_map() -> None:
    """SCENARIO-REPORT-8371-BOUNDARY: complete costs permit only measured zero fabric share."""
    rows = [dict(operation=op, cost_ns=0) for op in e.OPERATIONS]
    mapped, fraction, missing = e.operation_map(rows)
    assert fraction is None and not missing
    rows[0]["cost_ns"] = 12
    mapped, fraction, missing = e.operation_map(rows)
    assert fraction == 0 and not missing
    assert all(r["assigned_substrate"] == "host_CPU" and not r["kv260_supported"] for r in mapped)
    mapped, fraction, missing = e.operation_map([])
    assert fraction is None and missing == e.OPERATIONS
    assert all(r["cost_ns"] is None for r in mapped)


@pytest.mark.parametrize(
    "rows",
    [
        [dict(operation="unknown", cost_ns=1)],
        [dict(operation="coefficient_reads", cost_ns=-1)],
        [dict(operation="coefficient_reads", cost_ns=True)],
        [dict(operation="coefficient_reads", cost_ns=float("inf"))],
        [dict(operation="coefficient_reads", cost_ns=float("nan"))],
        [dict(operation="coefficient_reads")],
        [dict(operation="coefficient_reads", cost_ns=1)] * 2,
    ],
)
def test_typed_rejections(rows: list[dict]) -> None:
    """SCENARIO-REPORT-8371-BOUNDARY: untyped or repeated costs cannot create a fraction."""
    with pytest.raises(ValueError, match="operation_trace"):
        e.operation_map(rows)


def test_natural_absence_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8371-REPLAY: copied current evidence remains a blocked honest result."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["polarfire_workload_validated"]
    assert value["actual_substrate"] == "host_CPU_aggregation"
    assert value["verdict_class"] == "blocked"
    assert value["current_device_execution_count"] == 0
    assert value["full_service_speedup"] is value["compatible_fraction"] is None
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    changed = deepcopy(value)
    changed["polarfire_workload_validated"] = False
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing")
    assert e.build(work, [dict(passed=False)], raw, path)["verdict_class"] == "disqualified"


def test_real_cli_and_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8371-CLI: real children publish private bytes and reject changed replay."""
    import subprocess
    import sys

    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    ran = subprocess.run(
        [*cli, "--private-e2e", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert e.replay(output)
    assert (
        subprocess.run(
            [*cli, "--cold-replay", str(output)], capture_output=True, timeout=60
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            [*cli, "--cold-replay", str(tmp_path / "missing")], capture_output=True, timeout=60
        ).returncode
        == 1
    )
    assert subprocess.run([*cli, "--private-e2e"], capture_output=True, timeout=30).returncode == 2
    assert (
        subprocess.run([*cli, "--date", "20261009"], capture_output=True, timeout=30).returncode
        == 2
    )
    assert (
        subprocess.run(
            [
                *cli,
                "--private-e2e",
                "--root",
                str(tmp_path / "missing-root"),
                "--output",
                str(tmp_path / "missing" / (e.NAME + ".json")),
            ],
            capture_output=True,
            timeout=120,
        ).returncode
        == 0
    )


def test_historical_rejections(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8371-BOUNDARY: hash, terminal, repeat and dispatch failures stay separate."""
    from unittest.mock import patch

    value = json.loads((e.ROOT / e.HISTORY).read_bytes())
    with patch.object(e.base, "sha256_file", return_value="wrong"):
        with pytest.raises(ValueError, match="historical_primary_hash"):
            e.board_history(e.ROOT, tmp_path, dict(checks=[], refs=[]))
    with patch.object(e.base, "probe", return_value=None):
        with pytest.raises(ValueError, match="historical_terminal"):
            e.board_history(e.ROOT, tmp_path, dict(checks=[], refs=[]))
    with (
        patch.object(e.base, "probe", return_value=value),
        patch.object(e.tables, "complete", return_value=False),
    ):
        with pytest.raises(ValueError, match="five_distinct"):
            e.board_history(e.ROOT, tmp_path, dict(checks=[], refs=[]))
    with patch.object(e.boards, "polarfire", return_value={}):
        with pytest.raises(ValueError, match="graduation_changed"):
            e.board_history(e.ROOT, tmp_path, dict(checks=[], refs=[]))


def test_declared_operation_refs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8371-BOUNDARY: private schema controls never replace natural traces."""
    from unittest.mock import patch

    raw = tmp_path / "raw"
    path = tmp_path / "trace.json"
    atomic_json(path, dict(operation_rows=[dict(operation=op, cost_ns=1) for op in e.OPERATIONS]))
    ref = e.base.reference(path)
    value = dict(milestone=e.MILESTONE, experiment_id=8364, operation_level_workload_reference=ref)
    with (
        patch.object(e, "SOURCES", e.SOURCES[:1]),
        patch.object(e.base, "probe", return_value=value),
    ):
        work = dict(checks=[], refs=[])
        branches = e.traces(tmp_path, raw, work)
        assert len(branches[0]["rows"]) == 7 and not work["checks"]
        for changed in [
            dict(value, experiment_id=0),
            dict(value, milestone="old"),
            dict(value, operation_level_workload_reference=None),
            dict(value, operation_level_workload_reference=dict(path=str(path), sha256="wrong")),
        ]:
            with patch.object(e.base, "probe", return_value=changed):
                work = dict(checks=[], refs=[])
                assert not e.traces(tmp_path, raw, work)[0]["rows"]
                assert not work["checks"][0]["passed"]


def test_manifest_and_owned_failure(tmp_path: Path) -> None:
    """REQ-VERIFY-8371: real failed child receipts disqualify and private manifests stay scoped."""
    from unittest.mock import patch
    from carnot.reporting import hardware_operation_runner_8371 as r

    plan = r.manifest(tmp_path)
    assert any("E2E018_E2E020" in row["name"] for row in plan)
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    with patch.object(
        r.qualified, "preflight", return_value=[e.gate(tmp_path, "tool", True, None)]
    ):
        assert (
            r.run(
                tmp_path / "absent",
                tmp_path / "results" / (e.NAME + ".json"),
                tmp_path,
                control=True,
            )
            == 0
        )
    value = e.build(work, [dict(passed=False)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"


def test_five_distinct_historical_repeats() -> None:
    """SCENARIO-REPORT-8371-BOUNDARY: duplicate rows cannot replace a missing repeat."""
    value = json.loads((e.ROOT / e.HISTORY).read_bytes())
    clocks = value["table_timing_rows"]
    assert e.tables.complete(clocks)
    changed = deepcopy(clocks)
    changed[-1] = deepcopy(changed[-2])
    assert not e.tables.complete(changed)


def test_owned_child_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8371-CLI: a real nonzero owned child clears qualification."""
    import sys
    from unittest.mock import patch
    from carnot.reporting import hardware_operation_runner_8371 as r

    atomic_json(tmp_path / "coverage.json", dict(totals=dict(percent_covered=100), files={}))
    plan = [
        dict(
            name="deliberate_child_failure",
            argv=[sys.executable, "-c", "raise SystemExit(3)"],
            deadline=10,
            expected=0,
            scope="owned",
        )
    ]
    output = tmp_path / (e.NAME + ".json")
    with patch.object(r, "manifest", return_value=plan):
        assert r.run(tmp_path / "missing-root", output, tmp_path) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["required_checks_passed"]
    assert any(row["exit_code"] == 3 for row in value["validation_receipts"])
    assert Path(value["terminal_validation_sidecar_path"]).is_file()


def test_rehashed_primitive_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8371-REPLAY: each primitive mutation must still meet source authority."""
    from unittest.mock import patch

    raw, path = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    original = deepcopy(work)

    def candidate(changed: dict) -> None:
        atomic_json(raw / "measurement.json", changed)
        atomic_json(path, e.build(changed, [dict(passed=True)], raw, path))

    changes = [
        lambda w: w["contract"].update(canonical_tasks_sha256="wrong"),
        lambda w: w["history"]["graduation"].update(output_sha256="wrong"),
        lambda w: w["branches"][0].update(reference=dict(path="unknown", sha256="wrong")),
        lambda w: w["refs"][0].update(original_path=str(tmp_path / "foreign")),
        lambda w: w["refs"][0].update(original_path=str(e.ROOT / e.TEST)),
    ]
    for change in changes:
        changed = deepcopy(original)
        change(changed)
        candidate(changed)
        assert not e.replay(path)
    candidate(original)
    with patch.object(e.authority.legacy.base, "PIN", "wrong"):
        assert not e.replay(path)
    value = json.loads(path.read_bytes())
    value["source_artifact_hashes"] = value["source_artifact_hashes"][1:]
    atomic_json(path, value)
    assert not e.replay(path)


def test_authority_and_protocol_rejections(tmp_path: Path) -> None:
    """REQ-REPORT-8371: exact task identity and unchanged V717 bytes gate this invocation."""
    from unittest.mock import patch

    with patch.object(
        e.authority,
        "authority",
        return_value=dict(activated=False, tasks=[dict(id=e.TASK)], canonical_tasks_sha256="wrong"),
    ):
        work = e.measure(tmp_path, tmp_path / "raw-mismatch")
        assert not work["checks"][0]["passed"]
    with patch.object(e.authority.legacy.base, "PIN", "wrong"):
        work = e.measure(e.ROOT, tmp_path / "raw-protocol")
        assert any("V717_protocol_changed" in str(row) for row in work["checks"])
