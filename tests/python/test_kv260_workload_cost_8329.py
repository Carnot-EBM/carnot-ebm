"""REQ-REPORT-8329 / REQ-VERIFY-8329: private controls never replace sources."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import kv260_workload_cost_8329 as e
from carnot.reporting import kv260_workload_runner_8329 as r
from carnot.reporting import kv260_workload_primitives_8329 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import local_update_isolation_8306 as k


def private_root(root: Path) -> Path:
    """Copy authority bytes so private measurements still check real task contracts."""
    for name in [e.authority_module.DESIGN, e.authority_module.ACTIVE, e.authority_module.PROTOCOL]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    return root


def source(root: Path, branch: str, **fields: object) -> Path:
    """Use the unchanged publisher for private authenticated producer controls."""
    name, field = e.SOURCES[branch]
    path = root / "results" / (name + ".json")
    terminal = root / "results/raw" / name / "terminal.json"
    trajectory = k.manifest()["trajectories"][0]
    primitive = root / "results/raw" / name / "workloads.json"
    atomic_json(primitive, dict(trajectories=[trajectory]))
    value = dict(
        experiment_id=int(name.split("_")[1]),
        task_id={
            "constructed": "exp8319-local-evidence-qualification",
            "capacity": "exp8323-bounded-feedback-capacity",
            "natural": "exp8322-continuous-local-learning",
        }[branch],
        milestone="2026.10.718",
        honest_verdict="complete_circular_positive_private",
        verdict_class="circular_positive",
        required_checks_passed=True,
        flagged_adversarial=False,
        raw_shard_hashes=[],
        terminal_validation_sidecar_path=str(terminal),
        workload_reference=dict(path=str(primitive), sha256=sha256_file(primitive)),
        **{field: 1},
    )
    value.update(fields)
    publication = publish_primary(path, value, lambda _: dict(passed=True))
    atomic_json(terminal, dict(publication=publication, normal_process_exit=True))
    return path


def test_independent_branch_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8329-INDEPENDENT: missing natural input cannot suppress costs."""
    root = private_root(tmp_path / "root")
    source(root, "constructed")
    work = e.measure(root, tmp_path / "raw")
    assert len(work["timing_rows"]) == 10
    assert {x["repetition"] for x in work["timing_rows"]} == set(range(5))
    assert all(x["complete_ns"] > 0 for x in work["timing_rows"])
    assert all(x["recovered_semantics"] == x["expected_semantics"] for x in work["timing_rows"])
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["cpu_cost_ready_score"] == 1 and value["natural_cost_ready_score"] == 0
    assert value["completed_count"] == 2 and value["censored_count"] == 4
    assert value["independent_count"] == 0
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    value["timing_rows"][0]["complete_ns"] += 1
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["cpu_cost_ready_score"] == 0


def test_missing_ineligible_and_authority(tmp_path: Path) -> None:
    """REQ-REPORT-8329: absent, malformed and failed sources retain exact gates."""
    work = e.measure(tmp_path, tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "blocked" and value["cpu_cost_ready_score"] == 0
    assert value["timing_rows"] == [] and value["compatible_fraction"] is None
    assert value["censored_count"] == value["intended_count"] == 6
    assert not e.replay(tmp_path / "missing")
    root = private_root(tmp_path / "root")
    path = source(root, "constructed", local_kernel_ready_score=0)
    assert not e.measure(root, tmp_path / "ineligible")["timing_rows"]
    path = source(root, "constructed")
    payload = json.loads(path.read_text())
    Path(payload["workload_reference"]["path"]).write_text("{}")
    assert not e.measure(root, tmp_path / "corrupt")["timing_rows"]


def test_operation_boundary_and_semantics(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8329-OPERATIONS: an Ising overlay supports none of these writes."""
    trajectory = k.manifest()["trajectories"][0]
    for arm in p.ARMS:
        row = p.transaction(trajectory, arm, tmp_path / (arm + ".json"))
        p.verify_row(row)
        assert set(row["operation_ns"]) == set(p.OPERATIONS)
        changed = deepcopy(row)
        changed["recovered_semantics"]["pending"] = [999]
        with pytest.raises(ValueError, match="semantics"):
            p.verify_row(changed)
        changed = deepcopy(row)
        changed["operation_ns"]["feature_access"] = -1
        with pytest.raises(ValueError, match="clock"):
            p.verify_row(changed)
        changed = deepcopy(row)
        changed["dispatch_actions"][0] = "forged"
        with pytest.raises(ValueError, match="dispatch"):
            p.verify_row(changed)
    boundary = p.boundary([])
    assert boundary["compatible_fraction"] is None and boundary["amdahl_upper_bound"] is None
    assert all(not x["kv260_supported"] for x in boundary["operation_rows"])


def test_actual_cli_and_worker(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8329-CLI: exercise real entrypoint and fresh negative replay."""
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--root", str(tmp_path / "absent"), "--private-output", str(output)]) == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert r.main(["--root", str(tmp_path), "--worker-output", str(tmp_path / "worker.json")]) == 0
    with pytest.raises(SystemExit):
        r.main(["--date", "20000101"])
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    child = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
        env=dict(os.environ),
    )
    assert child.returncode == 0 and b"replay_passed" in child.stdout


def test_natural_capacity_and_primitive_failures(tmp_path: Path) -> None:
    """REQ-REPORT-8329: source counts survive missing peers and malformed primitives."""
    root = private_root(tmp_path / "root")
    source(root, "natural")
    source(root, "capacity")
    work = e.measure(root, tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["natural_cost_ready_score"] == 1 and value["independent_count"] == 1
    for branch in ["natural", "capacity"]:
        path = root / "results" / (e.SOURCES[branch][0] + ".json")
        data = json.loads(path.read_bytes())
        primitive = Path(data["workload_reference"]["path"])
        atomic_json(primitive, dict(trajectories=[]))
        data["workload_reference"]["sha256"] = sha256_file(primitive)
        publication = publish_primary(path, data, lambda _: dict(passed=True))
        atomic_json(Path(data["terminal_validation_sidecar_path"]), dict(publication=publication))
    assert not e.measure(root, tmp_path / "empty")["timing_rows"]
    value = e.build(work, tmp_path / "raw", [])
    assert value["verdict_class"] == "disqualified"
    candidate = tmp_path / (e.NAME + ".json")
    log = tmp_path / "stdout"
    log.write_text("real log")
    value = e.build(
        work,
        tmp_path / "raw",
        [dict(passed=True, stdout_path=str(log), stdout_sha256=sha256_file(log))],
    )
    atomic_json(candidate, value)
    assert e.replay(candidate)
    log.write_text("tampered")
    assert not e.replay(candidate)


def test_polarfire_terminal_scope_and_negative_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8329-OPERATIONS: terminal dispatch authenticates without probing."""
    work = dict(checks=[], refs=[])
    qualified = e.polarfire(e.ROOT, tmp_path / "raw", work)
    assert qualified["polarfire_workload_validated"]
    assert qualified["scope"] == "board_local_Linux_CPU_only"
    assert not qualified["fpga_fabric_acceleration"]
    with patch.object(e.packet_evaluator, "evaluate", return_value={}):
        assert not e.polarfire(e.ROOT, tmp_path / "badparity", dict(checks=[], refs=[]))[
            "polarfire_workload_validated"
        ]
    original = e.read_bound_sidecar

    def failed_receipt(primary: Path, sidecar: Path) -> dict:
        value = deepcopy(original(primary, sidecar))
        value["report"]["receipts"][0]["passed"] = False
        return value

    with patch.object(e, "read_bound_sidecar", failed_receipt):
        assert not e.polarfire(e.ROOT, tmp_path / "failed", dict(checks=[], refs=[]))[
            "polarfire_workload_validated"
        ]
    original_pin = e.pin

    def changed_transcript(path: Path, raw: Path, refs: list) -> Path:
        target = original_pin(path, raw, refs)
        if path.name == "board_evaluate.stdout":
            atomic_json(target, dict(changed=True))
        return target

    with patch.object(e, "pin", changed_transcript):
        assert not e.polarfire(e.ROOT, tmp_path / "changed", dict(checks=[], refs=[]))[
            "polarfire_workload_validated"
        ]
    original_json = e.json.loads

    def unadjudicated(data: object) -> dict:
        value = original_json(data)
        if isinstance(value, dict) and "reports" in value:
            value["reports"][0]["flags"] = [dict(kind="UNKNOWN", severity="warn")]
        return value

    with patch.object(e.json, "loads", unadjudicated):
        assert not e.polarfire(e.ROOT, tmp_path / "unknown", dict(checks=[], refs=[]))[
            "polarfire_workload_validated"
        ]


def test_cli_owned_failure_and_recovery(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8329-CLI: failed owned validation publishes disqualification."""
    output = tmp_path / "production" / (e.NAME + ".json")
    failure = dict(
        name="owned_failure",
        argv=[sys.executable, "-c", "raise SystemExit(3)"],
        expected=0,
        deadline=20,
        scope="owned",
    )
    with patch.object(r, "manifest", return_value=[failure]):
        assert r.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    with patch.object(r, "manifest", return_value=[]):
        assert r.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert e.replay(output)

    def coverage_manifest(private: Path) -> list:
        atomic_json(
            private / "coverage.json",
            dict(totals=dict(num_statements=364, covered_lines=364, missing_lines=0)),
        )
        return []

    with (
        patch.object(r, "manifest", coverage_manifest),
        patch.object(r, "controls", return_value=[]),
        patch.object(r, "publish"),
    ):
        assert (
            r.main(
                [
                    "--root",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "coverage" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
    coverage = next((tmp_path / "coverage").rglob("owned_coverage.json"))
    assert json.loads(coverage.read_bytes())["totals"]["missing_lines"] == 0
    with (
        patch.object(r, "manifest", return_value=[failure]),
        patch.object(
            r,
            "child",
            return_value=dict(
                passed=False, exit_code=73, stdout_path=str(tmp_path / "missing.log")
            ),
        ),
        patch.object(r, "controls", return_value=[]),
        patch.object(r, "publish"),
    ):
        assert (
            r.main(
                ["--root", str(tmp_path), "--output", str(tmp_path / "failed" / (e.NAME + ".json"))]
            )
            == 0
        )


def test_terminal_rejection_and_timeout(tmp_path: Path) -> None:
    """REQ-VERIFY-8329: terminal failure is retained and only owned groups time out."""
    work = e.measure(tmp_path / "missing", tmp_path / "raw")
    actual = r.publish_primary
    count = 0

    def reject_once(output: Path, value: dict, validator: object) -> dict:
        nonlocal count
        count += 1
        if count == 1:
            validator(tmp_path / "raw/audit_candidate.json")
            raise ValueError("candidate_rejected")
        return actual(output, value, validator)

    with patch.object(r, "publish_primary", reject_once):
        r.publish(work, tmp_path / (e.NAME + ".json"), tmp_path / "raw", [dict(passed=True)])
    with patch.object(r, "publish_primary", side_effect=ValueError("conflicting_identity")):
        with pytest.raises(ValueError, match="conflicting_identity"):
            r.publish(work, tmp_path / (e.NAME + ".json"), tmp_path / "raw", [dict(passed=True)])
    receipt = r.child(
        "timeout",
        [sys.executable, "-c", "import time;time.sleep(5)"],
        tmp_path / "timeout",
        deadline=0.1,
        heartbeat=0.03,
    )
    assert receipt["timed_out"] and not receipt["passed"]


def test_owned_benchmark_failure_and_private_root_guard(tmp_path: Path) -> None:
    """REQ-VERIFY-8329: owned numerical failures cannot be external missingness."""
    root = private_root(tmp_path / "root")
    source(root, "constructed")
    with patch.object(e.p, "transaction", side_effect=ValueError("owned_benchmark_failure")):
        work = e.measure(root, tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "disqualified" and value["failed_count"] == 2
    assert not value["required_checks_passed"] and value["cpu_cost_ready_score"] == 0
    with pytest.raises(SystemExit):
        r.main(["--root", str(root), "--output", str(e.ROOT / "results" / (e.NAME + ".json"))])


def test_exact_source_authority(tmp_path: Path) -> None:
    """REQ-REPORT-8329: a producer with another milestone cannot supply current costs."""
    root = private_root(tmp_path / "root")
    source(root, "constructed", milestone="2026.10.717")
    assert not e.measure(root, tmp_path / "raw")["timing_rows"]
