"""Regression test for a third submission-prep pre-flight incident (2026-07-15): the local
submission gate itself was self-oversubscribing CPU. `_measure_game()` is called 8x concurrently
via `measure()`'s `ThreadPoolExecutor(max_workers=8)`, and each call spawns a fresh
`arc_leaderboard_eval.py` subprocess. A single such subprocess was found (via `/proc/PID/status`)
to spawn 24 threads on this 24-core box -- numpy/scipy/torch default to one OpenMP/OpenBLAS
thread per core when no thread-count env var is set. 8 unconstrained subprocesses in parallel
means up to 192 threads contending for 24 cores: severe, fully self-inflicted oversubscription
that made the gate fail 3 consecutive times with IDENTICAL results (1/8 solved, 7777 actions,
7 timed_out each time) even AFTER two real underlying performance bugs were fixed and
individually verified fast (REQ-ARC-FCP-5591-3, REQ-CAPSTONE-4556-2). Pinning every math
library to 1 thread per subprocess resolved it: a subsequent gate run passed with 7/8 solved
(better than the 4/8 baseline) and 0 timed out.

Spec refs: REQ-ARC-FCP-5591-3, REQ-CAPSTONE-4556-2 (this fix completes the same incident's
resolution; the gate script itself has no OpenSpec capability of its own -- it is operational
tooling, not product code).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "kaggle"))

import arc_local_submission_gate as gate  # noqa: E402


_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def test_measure_game_env_pins_every_thread_limit_var_to_one(monkeypatch) -> None:
    """SCENARIO: each of the 8 parallel game-eval subprocesses must be launched with every
    math-library thread count pinned to 1, so 8 concurrent subprocesses use <=8 threads total
    instead of self-oversubscribing a 24-core box."""

    captured_env = {}

    class _FakeCompleted:
        stdout = ""
        stderr = ""

    def _fake_run(cmd, *, capture_output, text, timeout, cwd, env):
        captured_env.update(env)
        return _FakeCompleted()

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    gate._measure_game("lp85", "e3", budget=100, cap=10)

    for var in _THREAD_LIMIT_ENV_VARS:
        assert captured_env.get(var) == "1", f"{var} was not pinned to 1"


def test_measure_game_still_sets_induction_disable_env(monkeypatch) -> None:
    """The thread-limiting fix must not disturb the pre-existing induction-disable behavior."""

    captured_env = {}

    class _FakeCompleted:
        stdout = ""
        stderr = ""

    def _fake_run(cmd, *, capture_output, text, timeout, cwd, env):
        captured_env.update(env)
        return _FakeCompleted()

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    gate._measure_game("lp85", "e3", budget=100, cap=10, disable_induction=True)
    assert captured_env.get(gate.INDUCTION_DISABLE_ENV) == "1"

    captured_env.clear()
    gate._measure_game("lp85", "e3", budget=100, cap=10, disable_induction=False)
    assert gate.INDUCTION_DISABLE_ENV not in captured_env
