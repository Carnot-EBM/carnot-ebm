"""REQ-KAN-8306, REQ-VERIFY-8306, REQ-REPORT-8306: exact constructed mechanics."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import local_update_isolation_8306 as k


def test_numeric_and_frozen_protocol():
    """SCENARIO-KAN-8306-NUMERIC: independent recursion and derivatives agree."""
    plan = k.manifest()
    assert len(plan["trajectories"]) == 48
    assert all(len(t["events"]) == 64 for t in plan["trajectories"])
    for v in [0, 1e-12, 0.2, 0.4, 0.6, 0.8, 1 - 1e-12, 1]:
        x = [0.3, v, v, v, v]
        d = k.design(x)
        assert max(abs(a - b) for a, b in zip(d, k.scalar_design(x), strict=True)) < 1e-14
        assert len(d) == 34
        assert sum(a != 0 for a in d[2:]) <= 16
    assert k.action(0.25) == k.action(0.75) == "escalate"
    assert k.action(0.2) == "accept" and k.action(0.8) == "reject"
    assert k.numeric_audit()["finite_difference_error_max"] < 1e-8
    with pytest.raises(ValueError, match="features"):
        k.design([0, 0, 0, 0, 2])
    with pytest.raises(ValueError, match="features"):
        k.design([float("nan")] * 5)
    state = k.initial(plan["trajectories"][0])
    before = list(state["coefficients"])
    row = k.update(state, dict(id="zero", x=[1, 0.3, 0.3, 0.3, 0.3], y=1, rate=0), "indexed")
    assert row["changed"] == [] and state["coefficients"] == before
    with pytest.raises(ValueError, match="arm"):
        k.update(state, dict(id="bad", x=[1, 0.3, 0.3, 0.3, 0.3], y=1), "unknown")


def test_all_trajectories_and_cache_controls(tmp_path):
    """SCENARIO-VERIFY-8306-STATE: faults never authorize stale cache reads."""
    for trajectory in k.manifest()["trajectories"]:
        full = k.execute(trajectory, "full", tmp_path / (trajectory["id"] + "f"))
        indexed = k.execute(trajectory, "indexed", tmp_path / (trajectory["id"] + "i"))
        assert k.semantic(full) == k.semantic(indexed)
        assert indexed["stale_cache_count"] == 0
        assert any(r["reason"] == "duplicate" for r in indexed["releases"])
        assert any(r["reason"] == "stale" for r in indexed["releases"])
        assert any(r["fallback"] for r in indexed["releases"])
        assert any(r["global_change"] for r in indexed["releases"])
        assert k.execute(trajectory, "indexed", tmp_path / (trajectory["id"] + "i")) == indexed
    assert k.negative_control()["detected"]
    assert k.negative_control()["action_changed"]


def child_args():
    """Collect child coverage before real hard exits so durability is measured."""
    prefix = [sys.executable]
    if os.environ.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    return [*prefix, str(k.ROOT / k.CLI)]


def test_real_child_recovery_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8306-RECOVERY: killed owned children resume pending issues."""
    plan = tmp_path / "manifest.json"
    atomic_json(plan, k.manifest())
    checkpoint = tmp_path / "state.json"
    base = child_args() + [
        "--worker",
        str(plan),
        "--trajectory",
        "0",
        "--arm",
        "indexed",
        "--checkpoint",
        str(checkpoint),
    ]
    for event in [24, 48]:
        result = subprocess.run(base + ["--crash", str(event)], capture_output=True, timeout=30)
        assert result.returncode == -9
        pending = k.load(k.manifest()["trajectories"][0], checkpoint)
        assert len(pending["issues"]) == event + 1
        assert pending["pending"]
    assert subprocess.run(base, capture_output=True, timeout=30).returncode == 0
    reference = k.execute(k.manifest()["trajectories"][0], "indexed", tmp_path / "baseline")
    assert k.semantic(k.load(k.manifest()["trajectories"][0], checkpoint)) == k.semantic(reference)
    value = json.loads(checkpoint.read_text())
    state = json.loads(value["encoded_state"])
    state["coefficients"][2] += 0.1
    value["encoded_state"] = json.dumps(state)
    value["sha256"] = canonical_hash(state)
    atomic_json(checkpoint, value)
    with pytest.raises(ValueError, match="checkpoint"):
        k.load(k.manifest()["trajectories"][0], checkpoint)
    checkpoint.write_text("{}")
    with pytest.raises(ValueError):
        k.load(k.manifest()["trajectories"][0], checkpoint)


def test_reporting_blocks_reduction_and_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8306-CLI: replay binds actual observations and byte receipts."""
    from carnot.reporting import local_update_isolation_8306 as e

    work = e.measure(e.ROOT, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert value["local_kernel_ready_score"] == 1
    value["stale_cache_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked", fixture=True)
    assert e.build(blocked, tmp_path / "blocked", [dict(passed=True)])["verdict_class"] == "blocked"
    assert not e.replay(tmp_path / "missing")
    for suffix, args in [("fixture", ["--fixture-output"]), ("worker", ["--worker-output"])]:
        out = tmp_path / suffix / (e.NAME + ".json")
        assert e.main(["--root", str(tmp_path / "absent"), *args, str(out)]) == 0
    out = tmp_path / "fixture" / (e.NAME + ".json")
    assert e.main(["--cold-replay", str(out)]) == 0
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    with pytest.raises(SystemExit, match="2"):
        e.main(["--date", "20000101"])
    with pytest.raises(SystemExit, match="2"):
        e.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    result = subprocess.run(
        child_args() + ["--cold-replay", str(out)], cwd=tmp_path, capture_output=True, timeout=30
    )
    assert result.returncode == 0
    result = subprocess.run(
        child_args() + ["--cold-replay", str(tmp_path / "missing")],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    from carnot.reporting import local_update_execution_8306 as runner

    old = runner.manifest

    def bounded(private, candidate):
        specs = old(private, candidate)
        specs["commands"] = [
            dict(name="real_child", argv=["/bin/true"], deadline_s=10, expected_exit=0)
        ]
        atomic_json(private / "coverage.json", dict(private_control=True))
        return specs

    monkeypatch.setattr(runner, "manifest", bounded)
    assert (
        runner.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "normal" / (e.NAME + ".json")),
            ]
        )
        == 0
    )
    private = tmp_path / "private"
    private.mkdir()
    spec = dict(name="failed_validator", argv=["/bin/false"], deadline_s=10, expected_exit=0)
    published = json.loads(out.read_text())
    raw = Path(published["measurement_reference"]["path"]).parent
    runner.publish(published, out, private, raw, [spec], False)
    assert json.loads(out.read_text())["local_kernel_ready_score"] == 0


def test_cohort_hard_exit_recovery(tmp_path):
    """SCENARIO-VERIFY-8306-RECOVERY: every intended trajectory survives both deaths."""
    plan = k.manifest()
    path = tmp_path / "protocol.json"
    atomic_json(path, plan)
    directory = tmp_path / "cohort"
    base = child_args() + [
        "--worker",
        str(path),
        "--trajectory",
        "-1",
        "--checkpoint",
        str(directory),
    ]
    for event in [24, 48]:
        result = subprocess.run(base + ["--crash", str(event)], capture_output=True, timeout=120)
        assert result.returncode == -9, result.stderr
        for trajectory in plan["trajectories"]:
            state = k.load(trajectory, directory / (trajectory["id"] + ".json"))
            assert len(state["issues"]) == event + 1
            assert len(state["releases"]) == event - 8
    assert subprocess.run(base, capture_output=True, timeout=120).returncode == 0
    for trajectory in plan["trajectories"]:
        expected = k.initial(trajectory)
        for slot in range(72):
            k.issue(expected, slot)
            k.release(trajectory, expected, slot, "indexed")
        assert k.semantic(
            k.load(trajectory, directory / (trajectory["id"] + ".json"))
        ) == k.semantic(expected)


@pytest.fixture(scope="module")
def primitive_work(tmp_path_factory):
    """Keep one real private measurement for adversarial replay controls."""
    from carnot.reporting import local_update_isolation_8306 as e

    directory = tmp_path_factory.mktemp("primitive8306")
    return e.measure(e.ROOT, directory / "raw", fixture=True), directory


def test_replay_faults_and_paid_clocks(primitive_work):
    """REQ-REPORT-8306: reject rehashed causal, cost, identity and receipt drift."""
    from carnot.reporting import local_update_isolation_8306 as e

    work, directory = primitive_work
    raw = directory / "raw"
    candidate = directory / "candidate.json"
    for mutation in [
        "audit",
        "stderr",
        "anchors",
        "log",
        "recovered",
        "state",
        "rows",
        "run_clock",
        "event_clock",
        "checksum",
        "hash",
    ]:
        changed = deepcopy(work)
        receipts = [dict(passed=True)]
        if mutation == "audit":
            changed["audit"]["static_slope"] += 0.1
        elif mutation == "stderr":
            changed["crashes"][0]["stderr_sha256"] = "wrong"
        elif mutation == "anchors":
            changed["refs"] = []
        elif mutation == "log":
            log = directory / "actual.log"
            log.write_text("actual output")
            receipts[0].update(log_path=str(log), log_sha256="wrong")
        elif mutation == "recovered":
            changed["recovered_states"][0]["coefficients"][2] += 0.1
        elif mutation == "state":
            changed["states"][0]["arms"]["full"]["coefficients"][2] += 0.1
        elif mutation == "rows":
            changed["rows"][0]["numerator"] = 0
        elif mutation == "run_clock":
            changed["costs"][0]["complete_run_ns"] += 1
        elif mutation == "event_clock":
            changed["costs"][0]["transactions"][0]["total_ns"] = 0
        atomic_json(raw / "measurement.json", changed)
        value = e.build(changed, raw, receipts)
        if mutation == "hash":
            value["raw_shard_hashes"][0]["sha256"] = "wrong"
            value.pop("reproducibility_checksum")
            value["reproducibility_checksum"] = canonical_hash(value)
        if mutation == "checksum":
            value["local_kernel_ready_score"] = 99
        atomic_json(candidate, value)
        assert not e.replay(candidate), mutation
    atomic_json(raw / "measurement.json", work)
    for row in work["costs"]:
        assert row["complete_run_ns"] >= sum(e["total_ns"] for e in row["transactions"])
        for event in row["transactions"]:
            assert (
                event["total_ns"]
                == event["issue_ns"] + event["release_ns"] + event["checkpoint_ns"]
            )


def test_failure_boundaries_and_kill_dispatch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8306-RECOVERY: exercise errors without killing pytest."""
    from carnot.reporting import local_update_isolation_8306 as e
    from carnot.reporting import local_update_execution_8306 as runner

    trajectory = k.manifest()["trajectories"][0]
    assert k.load(trajectory, tmp_path / "absent") == k.initial(trajectory)
    path = tmp_path / "protocol.json"
    atomic_json(path, k.manifest())
    with pytest.raises(SystemExit, match="2"):
        runner.main(["--worker", str(path)])
    kills = []
    monkeypatch.setattr(k.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    k.execute(trajectory, "indexed", tmp_path / "single", crash=24)
    runner.main(
        [
            "--worker",
            str(path),
            "--trajectory",
            "-1",
            "--cohort-count",
            "1",
            "--checkpoint",
            str(tmp_path / "cohort"),
            "--crash",
            "24",
        ]
    )
    assert len(kills) == 2 and all(pid == os.getpid() for pid, _ in kills)
    external = tmp_path / "external" / "results" / (next(iter(e.PINS)) + ".json")
    atomic_json(external, dict(required_checks_passed=True, flagged_adversarial=False))
    monkeypatch.setattr(e, "PINS", {external.stem: e.reference(external)["sha256"][7:]})
    plan = e.authenticate(external.parent.parent, tmp_path / "structural")
    assert plan["checks"][-1]["artifact_field"] == "structure"


def test_measurement_deadline_and_plain_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8306: measurement deadlines stop owned work instead of padding."""
    from types import SimpleNamespace
    import time
    from carnot.reporting import local_update_isolation_8306 as e

    calls = []

    def clock():
        calls.append(1)
        return 0.0 if len(calls) == 1 else 601.0

    monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
    monkeypatch.setattr(e, "time", SimpleNamespace(monotonic=clock, monotonic_ns=time.monotonic_ns))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        e.measure(e.ROOT, tmp_path / "deadline", fixture=True)
