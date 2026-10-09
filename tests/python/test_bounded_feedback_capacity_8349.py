"""REQ-VERIFY-8349 / REQ-REPORT-8349: constructed systems claims stay causal."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import bounded_feedback_capacity_8349 as k
from carnot.reporting import bounded_feedback_capacity_8349 as e
from carnot.reporting import bounded_feedback_execution_8349 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_frozen_traces_and_roster():
    """SCENARIO-VERIFY-8349-CAPACITY: seeds add admissions, never natural sources."""
    plan = k.manifest()
    assert len(plan["traces"]) == 3 and len(plan["units"]) == 63
    for trace in plan["traces"]:
        assert len(trace["events"]) == 512
        for row in trace["events"]:
            t = row["slot"]
            assert row["x"][:12] == [0.0] * 12
            assert row["x"][12:] == [((17 * t + 13 * j) % 101) / 100 for j in range(4)]
            x = row["x"]
            s = 1 if t <= 256 else -1
            assert row["y"] == int(s * (2 * x[12] - x[13] + x[14] - 2 * x[15] + 0.1) >= 0)
    assert [r["delay"] for r in plan["traces"][1]["events"][:4]] == [1, 8, 32, 64]
    assert plan["traces"][2]["events"][128]["delay"] == 64


def test_capacity_order_sparse_dense_and_permanent_loss():
    """SCENARIO-VERIFY-8349-CAPACITY: prediction before release loses a boundary item."""
    trace = k.manifest()["traces"][2]
    result = k.simulate(trace, dict(policy="first", capacity=64, seed=11))
    assert result["maximum_pending_count"] == 64 and result["feedback_lost"] > 0
    assert result["feedback_retained"] + result["feedback_lost"] == 512
    assert result["heads"]["sparse"] == result["heads"]["dense"]
    assert result["heads"]["sparse"][:2] == [1.0, 0.0]
    assert result["pending"] == []
    assert all("y" not in v for v in result["events"] if v["kind"] == "issue")
    assert k.audit(trace, result)
    for policy in ("unlimited", "first", "random"):
        unit = dict(policy=policy, capacity=None if policy == "unlimited" else 4, seed=22)
        state = k.simulate(trace, unit)
        assert k.audit(trace, state)
        assert (
            state["metrics"]["sparse"]["coefficient_touches"]
            < state["metrics"]["dense"]["coefficient_touches"]
        )
    assert k.controls()["overflow_rejected"]
    assert all(k.controls().values())


def test_future_and_lost_label_barriers():
    """SCENARIO-VERIFY-8349-RECOVERY: the learner cannot retrieve unreleased labels."""
    trace = k.manifest()["traces"][0]
    unit = dict(policy="first", capacity=4, seed=11)
    state = k.initial(unit)
    vault = k.Vault(trace)
    for slot, clock in [(1, 1), (1, 8)]:
        with pytest.raises(ValueError):
            vault.release(slot, clock, state)
    state["lost"] = [1]
    with pytest.raises(ValueError):
        vault.release(1, 9, state)
    state["lost"] = []
    state["pending"] = [dict(slot=1, priority=0.2)]
    assert vault.release(1, 9, state)["y"] == trace["events"][0]["y"]
    state["pending"] *= 5
    with pytest.raises(ValueError):
        k.invariant(state)
    state["pending"] = []
    with pytest.raises(ValueError):
        k.simulate(trace, dict(policy="bogus", capacity=4, seed=11))


def test_real_child_recovery_and_rehashed_checkpoint(tmp_path):
    """SCENARIO-VERIFY-8349-RECOVERY: real exits preserve RNG and causal state."""
    trace = k.manifest()["traces"][1]
    unit = dict(policy="random", capacity=16, seed=33)
    bundle = tmp_path / "bundle.json"
    atomic_json(bundle, dict(trace=trace, unit=unit))
    expected = k.simulate(trace, unit)
    dest = tmp_path / "child"
    args = r.cli() + ["--worker", str(bundle), "--worker-dir", str(dest)]
    for slot, exit_code in [(128, 73), (256, 73), (0, 0)]:
        row = r.check(
            dict(
                name=str(slot),
                argv=args + ["--crash", str(slot)],
                expected_exit=exit_code,
                deadline_s=60,
            ),
            tmp_path / "logs",
        )
        assert row["passed"]
    actual = json.loads((dest / "final.json").read_bytes())
    assert k.semantic(actual) == k.semantic(expected)
    assert k.worker(json.loads(bundle.read_bytes()), dest) == actual
    bad = json.loads((dest / "state.json").read_bytes())
    bad["scheduler_rng_state"][1][1] += 1
    bad["checksum"] = canonical_hash(k.semantic(bad))
    atomic_json(dest / "state.json", bad)
    with pytest.raises(ValueError):
        k.worker(json.loads(bundle.read_bytes()), dest)


def test_authentication_and_external_blocks(tmp_path, monkeypatch):
    """REQ-REPORT-8349: natural upstream bytes authorize constructed measurements."""
    work = e.preconditions(e.ROOT, tmp_path / "auth")
    assert not work["failures"] and work["authority"]["activated"]
    assert work["historical"]
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    assert missing["failures"] and not missing["states"]
    value = e.build(missing, tmp_path / "missing", [dict(passed=True)])
    assert value["verdict_class"] == "blocked" and value["capacity_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] is None
    assert (
        e.build(missing, tmp_path / "missing", [dict(passed=False)])["verdict_class"]
        == "disqualified"
    )
    monkeypatch.setattr(e.os, "access", lambda *args: False)
    with pytest.raises(OSError):
        e.preconditions(e.ROOT, tmp_path / "tool")


def test_measure_reduce_replay_and_tamper(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8349-RECOVERY: primitive replay checks every summary claim."""
    plan = k.manifest()
    plan["units"] = [plan["units"][1], plan["units"][4]]
    monkeypatch.setattr(k, "manifest", lambda: plan)
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert work["states"] and all(work["checks"].values())
    rows = e.summaries(work)
    assert len(rows) == 4 and all(v["status"] == "completed" for v in rows)
    proof = k.numeric_proof(work["states"])
    assert proof["recomputed"] and proof["deliberate_error_rejected"]
    altered = deepcopy(work["states"])
    altered[0]["heads"]["dense"][2] += 0.1
    assert not k.numeric_proof(altered)["recomputed"]
    assert not k.numeric_proof([])["recomputed"]
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    from scripts.adversarial_verify import check_methodology_present

    flags = []
    check_methodology_present(value, flags)
    assert flags == []
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    with monkeypatch.context() as m:
        m.setattr(e, "sha256_file", lambda path: "sha256:changed")
        assert not e.replay(path)
    with monkeypatch.context() as m:
        m.setattr(k, "manifest", lambda: dict(plan, changed=True))
        assert not e.replay(path)
    with monkeypatch.context() as m:
        m.setattr(k, "audit", lambda *args: False)
        assert not e.replay(path)
    changed_work = deepcopy(work)
    changed_work["states"][0]["metrics"]["sparse"]["bytes_written"] += 1
    atomic_json(tmp_path / "raw/measurement.json", changed_work)
    atomic_json(path, e.build(changed_work, tmp_path / "raw", [dict(passed=True)]))
    assert not e.replay(path)
    atomic_json(tmp_path / "raw/measurement.json", work)
    value["capacity_ready_score"] = 99
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent.json")
    state = deepcopy(work["states"][0])
    state["events"][0]["p"]["dense"] += 0.01
    assert not k.audit(plan["traces"][0], state)


def test_real_private_cli_and_failure_paths(tmp_path):
    """SCENARIO-REPORT-8349-CLI: real publication and negative replay stay private."""
    output = tmp_path / (e.NAME + ".json")
    assert r.check(
        dict(
            name="private",
            argv=r.cli()
            + ["--private", "--root", str(tmp_path / "absent"), "--output", str(output)],
            deadline_s=120,
        ),
        tmp_path / "logs",
    )["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert r.check(
        dict(name="cold", argv=r.cli() + ["--cold-replay", str(output)], deadline_s=60),
        tmp_path / "logs",
    )["passed"]
    for name, args in [
        ("date", ["--date", "20261010"]),
        ("unsafe", ["--private"]),
        ("worker", ["--worker", str(output)]),
    ]:
        assert r.check(
            dict(name=name, argv=r.cli() + args, expected_exit=2, deadline_s=30), tmp_path / "logs"
        )["passed"]
    plan = r.manifest(tmp_path, output)
    assert plan["no_full_repository_suite"]
    assert "--files" in plan["commands"][-1]["argv"]


def test_owned_failure_and_terminal_rejection(tmp_path, monkeypatch):
    """REQ-REPORT-8349: owned failures stay disqualified; validators remain unchanged."""
    monkeypatch.setattr(
        e,
        "preconditions",
        lambda root, raw: dict(gates=[], failures=[], refs=[], authority={}, historical=[]),
    )
    monkeypatch.setattr(
        k, "simulate", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("owned failure"))
    )
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert not all(work["checks"].values())
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "disqualified"
    monkeypatch.setattr(r, "check", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError):
        r.publish(value, tmp_path / (e.NAME + ".json"), tmp_path / "raw", {})


def test_resource_corruption_worker_and_control_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8349: actual guard and child failures must clear readiness."""
    from types import SimpleNamespace

    with monkeypatch.context() as m:
        m.setattr(e.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
        with pytest.raises(OSError, match="storage"):
            e.preconditions(e.ROOT, tmp_path / "disk")
    with monkeypatch.context() as m:

        def broken(*args):
            raise ValueError("stale terminal bytes")

        m.setattr(e, "read_bound_sidecar", broken)
        work = e.preconditions(e.ROOT, tmp_path / "stale")
        assert work["failures"][0]["observed"] == "stale terminal bytes"
    with monkeypatch.context() as m:
        m.setattr(r, "check", lambda *args: dict(passed=False))
        work = e.measure(e.ROOT, tmp_path / "failed_child")
        assert work["owned_failure"] == "owned_worker_failure"
        assert (
            e.build(work, tmp_path / "failed_child", [dict(passed=True)])["capacity_ready_score"]
            == 0
        )
    with monkeypatch.context() as m:
        m.setattr(k, "invariant", lambda state: None)
        m.setattr(k.Vault, "release", lambda *args: {})
        assert not any(k.controls().values())


def test_normal_orchestration_retains_actual_coverage(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8349-CLI: normal mode runs frozen checks and keeps coverage."""
    original = r.manifest

    def simple(private, candidate):
        plan = original(private, candidate)
        cov = str(e.ROOT / ".venv/bin/coverage")
        config = "--rcfile=" + str(private / "coverage.ini")
        plan["commands"] = [
            dict(
                name="actual_child",
                argv=[
                    cov,
                    "run",
                    config,
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(tmp_path / "absent.json"),
                ],
                expected_exit=1,
                deadline_s=30,
            ),
            dict(name="actual_combine", argv=[cov, "combine", config], deadline_s=30),
            dict(
                name="actual_coverage",
                argv=[cov, "json", config, "-o", str(private / "coverage.json")],
                deadline_s=30,
            ),
        ]
        return plan

    # A real report from a measured child is retained; only command
    # selection is narrowed to avoid recursively launching this test itself.
    monkeypatch.setattr(r, "manifest", simple)
    assert (
        r.main(["--root", str(tmp_path / "absent"), "--output", str(tmp_path / (e.NAME + ".json"))])
        == 0
    )
