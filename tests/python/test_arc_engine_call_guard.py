"""Regression tests for the generated-engine call guard.

Spec: REQ-ARC-WMTE-6400 (openspec/capabilities/arc-world-model-trust-energy/spec.md).

THE INCIDENT UNDER TEST. A generated sb26 engine contained a non-terminating,
unboundedly-allocating flood fill (`flood(r, c, val)` called without `visited`).
One click candidate from `plan_in_model` entered the loop; the process reached
~78 GB RSS and earlyoom killed it. Twice, same seed. The fixture at
tests/python/fixtures/sb26_generated_world_model_hang.py preserves that engine
verbatim; these tests prove the guard converts the hang into a handled outcome.

WHY EVERY HANG TEST HAS A BACKSTOP. Each deliberately-runaway callable stops on
its own after a bounded number of steps or seconds and returns a sentinel. A
guard regression then FAILS the assertion quickly instead of hanging the suite
or ballooning the process (earlyoom is armed on this development box).

SAFETY. CPU-only, no GPU, no LLM. The allocation test is hard-capped at 400 MB
in-process; the fixture repro runs in a subprocess under RLIMIT_AS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as wm
from carnot.agentic.arc_engine_call_guard import (
    EngineCallGuardError,
    EngineCallMemoryExceeded,
    EngineCallTimeout,
    guarded_call,
)

_TESTS_DIR = Path(__file__).resolve().parent
_REPO = _TESTS_DIR.parents[1]
_FIXTURE = _TESTS_DIR / "fixtures" / "sb26_generated_world_model_hang.py"


def _spin(backstop_s: float = 30.0) -> str:
    """Pure spin, ZERO allocation: only the timeout channel can catch this class.
    Reaching the backstop means the guard did not fire -- a test failure, not a hang."""
    end = time.monotonic() + backstop_s
    while time.monotonic() < end:
        pass
    return "backstop_reached"


def _always_hanging_engine(grid, action, data):
    """An engine that hangs on EVERY input, like a systemically-broken candidate."""
    return _spin()


def _tiny_engine(grid, action, data):
    g = grid.copy()
    g[0, 0] = action
    return g


# REQ-ARC-WMTE-6400 SCENARIO 1: the guard fires OFF the main thread. This is the
# deployment reality (one thread per game) and the reason signal.alarm is unusable:
# CPython delivers signals only on the main thread, so an alarm-based watchdog
# would be armed-looking and protect nothing in exactly the process that matters.
def test_guard_fires_off_main_thread_on_pure_spin():
    result: dict = {}

    def worker():
        try:
            result["value"] = guarded_call(_spin, timeout_s=1.0, rss_delta_bytes=None)
        except EngineCallGuardError as e:
            result["exc"] = e

    th = threading.Thread(target=worker)
    th.start()
    th.join(45.0)
    assert not th.is_alive(), "worker still spinning: guard did not fire off the main thread"
    assert "value" not in result, f"spin ran to its backstop unguarded: {result}"
    assert isinstance(result.get("exc"), EngineCallTimeout)


# REQ-ARC-WMTE-6400 SCENARIO 2: the memory channel fires on unbounded allocation
# long before earlyoom territory, independently of the timeout channel.
def test_memory_guard_fires_before_runaway_allocation():
    chunks: list = []

    def _alloc() -> str:
        # Gentle by design: 4 MB per step, hard-capped at 100 steps (400 MB total),
        # so a guard regression cannot balloon the process on this earlyoom-armed box.
        for _ in range(100):
            chunks.append(bytearray(4 * 1024 * 1024))
            time.sleep(0.01)
        return "backstop_reached"

    try:
        with pytest.raises(EngineCallMemoryExceeded):
            guarded_call(_alloc, timeout_s=None, rss_delta_bytes=96 * 1024 * 1024)
    finally:
        chunks.clear()


# REQ-ARC-WMTE-6400 SCENARIO 3: no false positive and no meaningful slowdown on a
# well-behaved engine. The bound is a generous absolute ceiling rather than a tight
# relative one so a loaded box cannot flake it; measured overhead is microseconds.
def test_happy_path_result_correct_no_trip_and_cheap():
    grid = np.zeros((14, 14), dtype=int)
    out = guarded_call(_tiny_engine, grid, 3, None)
    assert int(out[0, 0]) == 3

    n = 500
    t0 = time.monotonic()
    for i in range(n):
        guarded_call(_tiny_engine, grid, i % 7, None)
    guarded = time.monotonic() - t0
    t0 = time.monotonic()
    for i in range(n):
        _tiny_engine(grid, i % 7, None)
    bare = time.monotonic() - t0
    print(f"engine-call guard overhead: guarded={guarded:.4f}s bare={bare:.4f}s for {n} calls")
    assert guarded < 2.0, (guarded, bare)


# REQ-ARC-WMTE-6400 SCENARIO 4: the kill switch really disables the guard.
def test_kill_switch_disables_guard(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_ENGINE_CALL_GUARD", "0")
    grid = np.zeros((4, 4), dtype=int)
    out = guarded_call(_tiny_engine, grid, 5, None)
    assert int(out[0, 0]) == 5


# REQ-ARC-WMTE-6400 SCENARIO 5: the wired live path (plan_in_model) still plans
# normally with the guard active -- plan found, zero trips recorded.
def test_plan_in_model_normal_engine_no_false_positive():
    def eng(grid, action, data):
        g = grid.copy()
        if action == 1 and g[0, 0] < 3:
            g[0, 0] += 1
        return g

    def done(grid):
        return int(grid[0, 0]) == 3

    start = np.zeros((6, 6), dtype=int)
    start[3, 3] = 7  # a real component, so click candidates run through the guard too
    diag: dict = {}
    plan = wm.plan_in_model(eng, done, start, diagnostics=diag)
    assert plan is not None
    assert [s["action"] for s in plan] == [1, 1, 1]
    assert diag["termination_reason"] == "plan_found"
    assert diag["engine_guard_trips"] == 0


# REQ-ARC-WMTE-6400 SCENARIO 6: a hanging engine on the wired live path, executed
# OFF the main thread (the scored eval's one-thread-per-game shape), is converted
# into a prompt None + "engine_guard_tripped" instead of a hung per-game thread.
def test_plan_in_model_hanging_engine_aborts_promptly_off_main_thread(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "0.5")
    monkeypatch.setenv("CARNOT_ARC_ENGINE_GUARD_MAX_TRIPS", "1")

    def eng(grid, action, data):
        if action == 2:  # action 2 is always among plan candidates: deterministic hang
            return _spin()
        return grid.copy()

    start = np.zeros((6, 6), dtype=int)
    result: dict = {}

    def worker():
        diag: dict = {}
        t0 = time.monotonic()
        plan = wm.plan_in_model(eng, lambda g: False, start, diagnostics=diag)
        result["plan"] = plan
        result["diag"] = diag
        result["dt"] = time.monotonic() - t0

    th = threading.Thread(target=worker)
    th.start()
    th.join(45.0)
    assert not th.is_alive(), "plan_in_model still hung: guard did not fire off the main thread"
    assert result["plan"] is None
    assert result["diag"]["termination_reason"] == "engine_guard_tripped"
    assert result["diag"]["engine_guard_trips"] == 1
    assert result["dt"] < 10.0, result["dt"]


# REQ-ARC-WMTE-6400 SCENARIO 7: WorldModelVerifier.score runs the generated engine
# BEFORE plan_in_model on the live path. A hanging engine pays ONE timeout, then the
# remaining rows are charged as raises without running -- bounded seconds, not
# timeout * len(transitions).
def test_world_model_verifier_score_hanging_engine_short_circuits(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "0.5")
    monkeypatch.setenv("CARNOT_ARC_ENGINE_GUARD_MAX_TRIPS", "1")
    g = np.zeros((4, 4), dtype=int)
    ts = [
        wm.Transition(
            grid=g.copy(), action=1, data=None, next_grid=g.copy(), level_before=0, level_after=0
        )
        for _ in range(3)
    ]
    t0 = time.monotonic()
    vr = wm.WorldModelVerifier(ts).score(_always_hanging_engine)
    dt = time.monotonic() - t0
    assert dt < 10.0, dt
    assert vr.n_engine_raised == 3
    assert vr.engine_raise_kinds.get("EngineCallTimeout") == 3
    assert vr.accuracy == 0.0


# REQ-ARC-WMTE-6400 SCENARIO 8: offpath_structural_energy has the same exposure and
# the same conversion -- inf rows, bounded wall clock.
def test_offpath_structural_energy_hanging_engine_short_circuits(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "0.5")
    monkeypatch.setenv("CARNOT_ARC_ENGINE_GUARD_MAX_TRIPS", "1")
    g = np.zeros((4, 4), dtype=int)
    ts = [
        wm.Transition(
            grid=g.copy(), action=1, data=None, next_grid=g.copy(), level_before=0, level_after=0
        )
        for _ in range(3)
    ]
    t0 = time.monotonic()
    val = wm.WorldModelVerifier(ts).offpath_structural_energy(
        _always_hanging_engine, energy_scorer=lambda *a: 0.0
    )
    dt = time.monotonic() - t0
    assert dt < 10.0, dt
    assert val == float("inf")


# REQ-ARC-WMTE-6400 SCENARIO 9 (the incident itself): the preserved sb26 generated
# engine, driven exactly as plan_in_model drove it (a click on a >=2-cell
# same-valued component), is converted into a guard trip. Runs in a subprocess
# under RLIMIT_AS so even a total guard regression cannot balloon this process.
_CHILD_SOURCE = r"""
import os, resource, sys

os.environ["CARNOT_ARC_ENGINE_CALL_TIMEOUT_S"] = "1.5"
os.environ["CARNOT_ARC_ENGINE_CALL_RSS_DELTA_MB"] = "256"
os.environ["CARNOT_ARC_ENGINE_GUARD_MAX_TRIPS"] = "1"
# Hard address-space ceiling: if the guard regresses, the flood fill hits a
# MemoryError at 8 GB instead of re-running the 78 GB earlyoom incident.
resource.setrlimit(resource.RLIMIT_AS, (8 * 2**30, 8 * 2**30))

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("sb26_hang_fixture", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Minimal trigger, per the incident diagnosis: background 4 everywhere plus a
# 2-cell same-valued block below the progress row. Clicking it enters
# flood(row, col, v) with no `visited`, which never terminates.
H = W = 14
grid = np.full((H, W), 4, dtype=int)
grid[10, 5] = 1
grid[10, 6] = 1

from carnot.agentic.arc_engine_call_guard import EngineCallGuardError, guarded_call

# Phase 1: the raw incident call, guarded directly.
try:
    guarded_call(mod.engine, grid, 6, {"x": 5, "y": 10})
    print("PHASE1_FAIL_no_exception")
    sys.exit(2)
except EngineCallGuardError as e:
    print("PHASE1_OK", type(e).__name__)

# Phase 2: the live call path -- plan_in_model over the same generated module,
# which emits the click candidate itself and must survive it.
from carnot.agentic.arc_executable_world_model import plan_in_model

diag = {}
res = plan_in_model(mod.engine, mod.is_level_complete, grid, diagnostics=diag)
assert res is None, res
assert diag.get("termination_reason") == "engine_guard_tripped", diag
assert diag.get("engine_guard_trips", 0) >= 1, diag
print("PHASE2_OK", diag["engine_guard_trips"])
"""


def test_sb26_incident_fixture_hang_converted_subprocess(tmp_path):
    assert _FIXTURE.is_file(), _FIXTURE
    script = tmp_path / "sb26_guard_repro.py"
    script.write_text(_CHILD_SOURCE)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "python") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(script), str(_FIXTURE)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr[-2000:])
    assert "PHASE1_OK" in proc.stdout, proc.stdout
    assert "PHASE2_OK" in proc.stdout, proc.stdout
