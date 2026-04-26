"""Tests for the async doc-reconciliation infrastructure in research_conductor.

The module-level executor + future state in research_conductor.py is what
lets a long-running Haiku doc-reconciliation overlap with the conductor's
inter-iteration sleep. These tests exercise the helpers directly and
mock out the actual reconcile work so the suite stays fast and offline.

What the helpers must guarantee:

  1. _submit_async_recon kicks a callable to a single-worker background
     executor and stores the future.
  2. _await_pending_recon blocks on the in-flight future, then clears
     the slot. Calling it twice in a row is safe (no-op the second time).
  3. The next _submit_async_recon implicitly awaits any prior pending
     future, so doc-recons across iterations are serialised. (No two
     git operations from doc-recon ever overlap.)
  4. Exceptions in the background callable do not crash the main thread.
  5. _shutdown_recon_executor drains pending work and tears the executor
     down cleanly — needed for atexit safety.
Spec: REQ-INFRA-067
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def conductor_async_module(monkeypatch):
    """Import the conductor module fresh, with side-effect modules stubbed.

    The conductor imports yaml at module level and reads files from disk on
    import, but the async-recon helpers we want to test are pure Python and
    don't need the full conductor environment. This fixture imports the
    relevant symbols by name and clears the module-level state between
    tests so they don't bleed.
    """
    import research_conductor as rc

    # Clear state between tests: shut any leftover executor down, null out
    # the future slot. Avoid surprise behaviour if a test failed mid-flight.
    rc._shutdown_recon_executor(wait=False, timeout=1.0)
    yield rc
    rc._shutdown_recon_executor(wait=False, timeout=1.0)


# ---------------------------------------------------------------------------
# _submit_async_recon + _await_pending_recon
# ---------------------------------------------------------------------------


def test_submit_runs_callable_in_background(conductor_async_module):
    """A submitted callable runs to completion off the main thread."""
    rc = conductor_async_module
    main_thread_id = threading.get_ident()
    captured: dict = {}

    def task():
        captured["thread_id"] = threading.get_ident()
        captured["ran"] = True

    rc._submit_async_recon(task)
    rc._await_pending_recon(timeout=5.0)
    assert captured.get("ran") is True
    # Confirm the task ran on a different thread than the test
    assert captured["thread_id"] != main_thread_id


def test_await_returns_immediately_when_no_pending(conductor_async_module):
    """_await_pending_recon is a no-op when nothing is in flight."""
    rc = conductor_async_module
    # No submission — should return promptly without raising
    t0 = time.perf_counter()
    rc._await_pending_recon(timeout=2.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5  # well under the timeout


def test_await_clears_pending_slot(conductor_async_module):
    """After _await_pending_recon, the future slot is None and a second
    await is a no-op (no double-wait).
    """
    rc = conductor_async_module
    rc._submit_async_recon(lambda: None)
    rc._await_pending_recon(timeout=5.0)
    assert rc._pending_recon_future is None
    # Second await — should not block, should not raise
    rc._await_pending_recon(timeout=1.0)
    assert rc._pending_recon_future is None


def test_submit_serialises_overlapping_recons(conductor_async_module):
    """If a second _submit_async_recon arrives while the first is still
    running, the first must complete before the second starts. This is
    the property that prevents two doc-recons from issuing concurrent
    git commits.
    """
    rc = conductor_async_module
    order: list[str] = []
    first_done = threading.Event()

    def first():
        time.sleep(0.05)
        order.append("first")
        first_done.set()

    def second():
        # If serialisation is broken, this could race with first()
        order.append("second")

    rc._submit_async_recon(first)
    # Submit second immediately — should block until first completes
    rc._submit_async_recon(second)
    # Drain
    rc._await_pending_recon(timeout=5.0)
    assert first_done.is_set()
    assert order == ["first", "second"]


def test_exception_in_background_does_not_crash_main(conductor_async_module, caplog):
    """A raise inside the background callable is logged, not propagated."""
    import logging

    rc = conductor_async_module
    rc._submit_async_recon(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # Awaiting must NOT raise — the helper logs and continues so the
    # conductor's main loop stays up.
    with caplog.at_level(logging.ERROR, logger="conductor"):
        rc._await_pending_recon(timeout=5.0)
    # The exception was logged at ERROR level. logger.exception() puts the
    # traceback in record.exc_info, not in the message text — check both.
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "expected at least one ERROR log record"
    found_boom = any(
        "raised" in r.message
        or "RuntimeError" in str(r.exc_info or "")
        or "boom" in str(r.exc_info or "")
        for r in error_records
    )
    assert found_boom, (
        f"none of the {len(error_records)} ERROR records mentioned the exception; "
        f"messages={[r.message for r in error_records]}"
    )


def test_timeout_does_not_raise(conductor_async_module, caplog):
    """A background callable that exceeds the await timeout is logged but
    does not raise — the next iteration still proceeds. The main thread
    must remain responsive even if a recon is genuinely stuck.
    """
    rc = conductor_async_module
    started = threading.Event()
    can_finish = threading.Event()

    def slow():
        started.set()
        can_finish.wait(timeout=10.0)

    rc._submit_async_recon(slow)
    started.wait(timeout=2.0)  # ensure background thread is actually running
    rc._await_pending_recon(timeout=0.1)
    # Future is cleared even though the background work is still running,
    # so the next iteration can proceed without blocking again.
    assert rc._pending_recon_future is None
    # Allow the slow task to finish so the executor doesn't leak the worker
    can_finish.set()


# ---------------------------------------------------------------------------
# _ensure_recon_executor + _shutdown_recon_executor
# ---------------------------------------------------------------------------


def test_executor_is_lazily_created(conductor_async_module):
    """The executor is None until first use; one call brings it up."""
    rc = conductor_async_module
    rc._shutdown_recon_executor(wait=False, timeout=1.0)
    assert rc._recon_executor is None
    executor = rc._ensure_recon_executor()
    assert executor is not None
    assert rc._recon_executor is executor


def test_executor_is_reused_across_calls(conductor_async_module):
    """Multiple _ensure_recon_executor calls return the same instance."""
    rc = conductor_async_module
    a = rc._ensure_recon_executor()
    b = rc._ensure_recon_executor()
    assert a is b


def test_shutdown_drains_pending_before_teardown(conductor_async_module):
    """_shutdown_recon_executor finishes the in-flight task, then closes
    the executor. Without this, an atexit-triggered shutdown could lose
    a recon that was about to commit + push.
    """
    rc = conductor_async_module
    finished = threading.Event()

    def task():
        time.sleep(0.05)
        finished.set()

    rc._submit_async_recon(task)
    rc._shutdown_recon_executor(wait=True, timeout=5.0)
    assert finished.is_set()
    assert rc._recon_executor is None
    assert rc._pending_recon_future is None


def test_shutdown_is_idempotent(conductor_async_module):
    """Calling _shutdown_recon_executor twice is safe (no double-close)."""
    rc = conductor_async_module
    rc._shutdown_recon_executor(wait=False, timeout=1.0)
    rc._shutdown_recon_executor(wait=False, timeout=1.0)
    assert rc._recon_executor is None
