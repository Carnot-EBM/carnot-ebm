"""Tests for carnot.conductor.single_run_guard.

The 2026-04-26 swap-saturation incident and the 2026-04-27 runaway-
Sonnet incidents both came from concurrent invocations of the same
experiment script. The flock-based guard is the durable fix from
openspec/change-proposals/conductor-process-isolation.md Exp B.

These tests pin the guard's contract: non-blocking semantics, proper
release on context-exit (including exception paths), and
multiprocessing isolation (subprocess can re-acquire after parent
releases).

Spec: REQ-INFRA-072
"""

from __future__ import annotations

import multiprocessing
import os
import time
import uuid

import pytest

from carnot.conductor import SingleRunHeld, acquire


@pytest.fixture
def unique_name() -> str:
    """Per-test unique lock name so tests don't interfere."""
    return f"test_{uuid.uuid4().hex[:12]}"


def test_acquire_succeeds_when_lock_is_free(unique_name: str) -> None:
    """A first acquire on an unheld lock yields and releases cleanly."""
    with acquire(unique_name):
        pass  # contract satisfied if no exception


def test_acquire_raises_when_lock_is_held_in_same_process(unique_name: str) -> None:
    """Same-process re-entry is blocked. flock semantics are advisory but
    LOCK_EX|LOCK_NB on an already-held fd raises BlockingIOError, which
    we convert to SingleRunHeld."""
    with acquire(unique_name):
        with pytest.raises(SingleRunHeld):
            with acquire(unique_name):
                pytest.fail("inner acquire should have raised")


def test_acquire_releases_after_context_exit(unique_name: str) -> None:
    """After the outer `with` block exits, a fresh acquire must succeed."""
    with acquire(unique_name):
        pass
    with acquire(unique_name):
        pass  # second acquire after release should succeed


def test_acquire_releases_on_exception(unique_name: str) -> None:
    """flock must be released even when the protected block raises."""
    with pytest.raises(RuntimeError, match="boom"):
        with acquire(unique_name):
            raise RuntimeError("boom")
    # Lock should now be free
    with acquire(unique_name):
        pass


def _hold_lock_in_subprocess(name: str, ready_event, exit_event) -> None:
    """Helper: subprocess holds the lock until told to exit."""
    try:
        with acquire(name):
            ready_event.set()
            exit_event.wait(timeout=10)
    except SingleRunHeld:
        # Subprocess could not acquire — signal failure
        ready_event.set()


def test_acquire_blocks_concurrent_subprocess(unique_name: str) -> None:
    """A subprocess holding the lock must block the parent's acquire.
    This is the canonical case the guard is designed for: two concurrent
    invocations of the same experiment script."""
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Event()
    exit_signal = ctx.Event()

    proc = ctx.Process(
        target=_hold_lock_in_subprocess,
        args=(unique_name, ready, exit_signal),
    )
    proc.start()
    try:
        # Wait for subprocess to grab the lock
        assert ready.wait(timeout=5), "Subprocess did not acquire lock in time"

        # Parent's acquire should fail immediately
        with pytest.raises(SingleRunHeld):
            with acquire(unique_name):
                pytest.fail("parent acquire should have raised")
    finally:
        exit_signal.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)


def test_acquire_is_released_on_subprocess_death(unique_name: str) -> None:
    """When the holder process dies (SIGKILL, abnormal exit, etc.) the
    kernel releases the flock automatically. This is a key property —
    no stale-lockfile recovery problem."""
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Event()
    exit_signal = ctx.Event()  # will not set; we'll terminate instead

    proc = ctx.Process(
        target=_hold_lock_in_subprocess,
        args=(unique_name, ready, exit_signal),
    )
    proc.start()
    try:
        assert ready.wait(timeout=5)
        # Forcibly kill the holder
        proc.terminate()
        proc.join(timeout=5)
        # Now the lock should be free again — parent can acquire
        with acquire(unique_name):
            pass
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)


def test_acquire_lockfile_records_holder_pid(unique_name: str) -> None:
    """The lockfile contents include the holder's PID for debugging.
    Not required for correctness (flock cares only about the fd) but
    useful when an operator runs `cat /tmp/carnot-locks/X.lock` to see
    who's holding it."""
    from carnot.conductor.single_run_guard import _LOCK_DIR

    with acquire(unique_name):
        lockfile = _LOCK_DIR / f"{unique_name}.lock"
        contents = lockfile.read_text().strip()
        assert contents == str(os.getpid())


def test_different_names_dont_block_each_other() -> None:
    """Distinct lock names are independent — running experiment_953 does
    not block experiment_967 (which is exactly what we want — different
    experiments may legitimately run concurrently)."""
    name_a = f"test_{uuid.uuid4().hex[:8]}"
    name_b = f"test_{uuid.uuid4().hex[:8]}"
    with acquire(name_a):
        with acquire(name_b):
            pass  # both held simultaneously without conflict
