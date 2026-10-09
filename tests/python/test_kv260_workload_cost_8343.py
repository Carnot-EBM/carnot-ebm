"""REQ-REPORT-8343 / REQ-VERIFY-8343: private controls cannot supply science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import kv260_workload_cost_8343 as e
from carnot.reporting import kv260_arithmetic_8343 as p
from carnot.reporting import kv260_workload_runner_8343 as r
from carnot.reporting import v718_replay_history as h
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import local_update_isolation_8306 as kernel


def authority(root: Path) -> Path:
    """Private copies preserve actual task authority without writing current results."""
    for name in [e.authority_module.DESIGN, e.authority_module.ACTIVE, e.authority_module.PROTOCOL]:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((e.ROOT / name).read_bytes())
    return root


def test_arithmetic_and_negative_controls() -> None:
    """SCENARIO-REPORT-8343-INDEPENDENT: SciPy and differences precede clocks."""
    audit = p.audit()
    assert audit["passed"] and audit["finite_difference_max_abs_delta"] < 1e-7
    rows = p.measure()
    assert len(rows) == 10
    assert [row["arm"] for row in rows[:4]] == ["dense", "active", "active", "dense"]
    for row in rows:
        p.verify(row)
        assert row["wall_ns"] >= sum(row["operation_ns"].values()) > 0
    assert rows[0]["coefficient_touches"] > rows[1]["coefficient_touches"]
    for field in ["probabilities", "actions", "coefficients", "wall_ns", "coefficient_touches"]:
        bad = deepcopy(rows[0])
        if isinstance(bad[field], list):
            bad[field][0] = "wrong" if field == "actions" else 99
        else:
            bad[field] = -1
        with pytest.raises(ValueError):
            p.verify(bad)
    with patch.object(p, "basis", return_value=[0.0] * 32):
        assert not p.audit()["passed"]
    with patch.object(p.math, "sqrt", return_value=100.0):
        with pytest.raises(ValueError, match="arithmetic_semantics"):
            p.verify(p.run("dense", 0))


def test_independent_measurement_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8343-REPLAY: missing optional science retains arithmetic."""
    work = e.measure(authority(tmp_path / "root"), tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["arithmetic_cost_ready_score"] == value["cpu_cost_ready_score"] == 1
    assert value["natural_cost_ready_score"] == value["durable_cost_ready_score"] == 0
    assert value["compatible_fraction"] is value["amdahl_upper_bound"] is None
    assert value["independent_count"] == 0
    assert value["verdict_class"] == "blocked"
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    value["arithmetic_cost_ready_score"] = 0
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    ref = work["arithmetic_reference"]
    primitive = json.loads(Path(ref["path"]).read_bytes())
    primitive[0]["probabilities"][0] += 0.1
    atomic_json(Path(ref["path"]), primitive)
    work["arithmetic_reference"] = e.reference(Path(ref["path"]))
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(candidate, e.build(work, tmp_path / "raw", [dict(passed=True)]))
    assert not e.replay(candidate)


def test_failed_preconditions_and_numerics(tmp_path: Path) -> None:
    """REQ-VERIFY-8343: missing authority and owned numerical failure stay distinct."""
    work = e.measure(tmp_path / "absent", tmp_path / "missing")
    assert not work["timing_rows"]
    with patch.object(p, "audit", return_value=dict(passed=False)):
        work = e.measure(authority(tmp_path / "root"), tmp_path / "bad_numeric")
    assert work["owned_failure"]
    assert (
        e.build(work, tmp_path / "bad_numeric", [dict(passed=True)])["verdict_class"]
        == "disqualified"
    )


def test_finding_consumer_qualification(tmp_path: Path) -> None:
    """REQ-VERIFY-8343: a raw exit never establishes an accepted clean report."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, {})
    report = dict(
        candidate_sha256=sha256_file(candidate),
        verifier_sha256=h.verifier_hash(),
        reports=[dict(artifact=str(candidate), loaded=True, flags=[], flag_count=0)],
    )
    assert h.consume(report, candidate, 0, {})["passed"]
    assert not h.consume(report, candidate, 1, {})["passed"]
    assert not h.consume(report, candidate, 2, {})["passed"]
    for changes in [
        dict(reports=None),
        dict(reports=[1]),
        dict(reports=[dict(flags=None)]),
        dict(candidate_sha256="wrong"),
        dict(reports=[]),
    ]:
        assert not h.consume(dict(report, **changes), candidate, 0, {})["passed"]
    for severity, kind in [
        ("warn", "IMPLAUSIBLE_PERFECT"),
        ("info", "UNKNOWN"),
        ("critical", "TAUTOLOGY"),
        ("info", "IMPLAUSIBLE_PERFECT"),
    ]:
        bad = deepcopy(report)
        bad["reports"][0].update(
            flags=[dict(severity=severity, kind=kind, detail="unproved")], flag_count=1
        )
        assert not h.consume(bad, candidate, 1, {})["passed"]


def test_real_cli_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8343-EXECUTION: actual CLI, child and recovery bytes execute."""
    plan = r.manifest(tmp_path / "validation")
    assert all(row["deadline"] <= 600 for row in plan)
    output = tmp_path / (e.NAME + ".json")
    argv = [
        sys.executable,
        str(e.ROOT / e.CLI),
        "--root",
        str(authority(tmp_path / "root")),
        "--private-output",
        str(output),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert value["arithmetic_cost_ready_score"] == 1
    with patch.object(r.qualified, "main", return_value=0):
        assert r.main([]) == 0


def producer(root: Path, branch: str, **changes: object) -> Path:
    """Publish a private producer to exercise availability and authentication paths."""
    name, field = e.SOURCES[branch]
    path = root / "results" / (name + ".json")
    raw = root / "results/raw" / name
    primitive = raw / "workloads.json"
    atomic_json(primitive, dict(trajectories=[kernel.manifest()["trajectories"][0]]))
    terminal = raw / "terminal.json"
    value = dict(
        experiment_id=int(name.split("_")[1]),
        task_id=e.SOURCE_TASKS[branch],
        milestone="2026.10.719",
        honest_verdict="complete_circular_positive_private",
        verdict_class="circular_positive",
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(terminal),
        raw_shard_hashes=[],
        workload_reference=e.reference(primitive),
        **{field: 1},
    )
    value.update(changes)
    publication = publish_primary(path, value, lambda _: dict(passed=True))
    atomic_json(terminal, dict(publication=publication))
    return path


def test_qualified_optional_and_failure_paths(tmp_path: Path) -> None:
    """REQ-REPORT-8343: qualified producers alone admit complete transaction timing."""
    root = authority(tmp_path / "root")
    producer(root, "natural")
    work = e.measure(root, tmp_path / "raw")
    assert len(work["durable_rows"]) == 10
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["durable_cost_ready_score"] == value["natural_cost_ready_score"] == 1
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for changes in [dict(milestone="wrong"), dict(task_id="exp8336-wrong")]:
        changed_root = authority(tmp_path / ("root_" + next(iter(changes))))
        producer(changed_root, "natural", **changes)
        work = e.measure(changed_root, tmp_path / ("invalid_" + next(iter(changes))))
        assert not work["durable_rows"]
    path = producer(root, "natural")
    data = json.loads(path.read_bytes())
    atomic_json(Path(data["workload_reference"]["path"]), dict(trajectories=[]))
    data["workload_reference"] = e.reference(Path(data["workload_reference"]["path"]))
    publication = publish_primary(path, data, lambda _: dict(passed=True))
    atomic_json(Path(data["terminal_validation_sidecar_path"]), dict(publication=publication))
    work = e.measure(root, tmp_path / "empty")
    assert not work["durable_rows"] and not work["owned_failure"]
    producer(root, "natural")
    with patch.object(e.durable, "transaction", side_effect=ValueError("owned_transaction")):
        work = e.measure(root, tmp_path / "failed")
    assert work["owned_failure"] and not work["durable_rows"]


def test_primitive_mismatch_and_receipt_binding(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8343-REPLAY: independent primitives and receipt bytes bind."""
    work = e.measure(authority(tmp_path / "root"), tmp_path / "raw")
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(tmp_path / "raw/arithmetic.json", [])
    work["arithmetic_reference"] = e.reference(tmp_path / "raw/arithmetic.json")
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(candidate, e.build(work, tmp_path / "raw", [dict(passed=True)]))
    assert not e.replay(candidate)
    assert p.action(0.1) == "accept" and p.action(0.9) == "reject"
    assert p.action(0.25) == p.action(0.75) == "escalate"


def test_runner_failed_checks_and_publication_recovery(tmp_path: Path) -> None:
    """REQ-VERIFY-8343: existing actual-child failure and recovery stay enforced."""
    root = authority(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    fail = dict(
        name="intentional_failure",
        argv=[sys.executable, "-c", "raise SystemExit(4)"],
        expected=0,
        deadline=10,
        scope="owned",
    )
    with patch.object(r, "manifest", return_value=[fail]):
        assert r.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["arithmetic_cost_ready_score"] == 0
    work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
    actual = r.qualified.publish_primary
    attempts = []

    def reject_once(output: Path, value: dict, validator: object) -> dict:
        def validate(candidate: Path) -> dict:
            report = validator(candidate)
            attempts.append(report)
            if len(attempts) == 1:
                report["passed"] = False
                report["checks"][0]["passed"] = False
            return report

        return actual(output, value, validate)

    with (
        patch.object(r.qualified, "e", e),
        patch.object(r.qualified, "publish_primary", reject_once),
    ):
        r.qualified.publish(
            work, output, Path(value["measurement_reference"]["path"]).parent, [dict(passed=True)]
        )
    assert len(attempts) == 2
