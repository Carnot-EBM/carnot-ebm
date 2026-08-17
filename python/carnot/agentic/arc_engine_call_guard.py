"""Bound one generated-engine call's wall time and memory growth (REQ-ARC-WMTE-6400).

WHY THIS EXISTS. The agent executes LLM-written engine code in-process. A generated
sb26 engine contained a flood fill that never terminated and never stopped
allocating: the arm process reached ~78 GB RSS, exhausted swap, and earlyoom killed
it. Twice, same seed. Search budgets (``max_nodes`` / ``max_depth``) bound HOW MANY
engine calls a search makes; nothing bounded what ONE call may cost. This module is
that bound, applied at the call boundary around every generated-engine invocation.

WHY NOT ``signal.alarm``. The scored eval runs one thread per game. CPython delivers
signals only on the main thread, so an alarm armed in a worker thread never fires.
That would be a guard that looks armed and protects nothing -- the "trusted and
silent" failure mode this project's QA-layer discipline names as the worst state a
guard can be in. The goal-probe watchdog in ``arc_executable_world_model`` already
documents this exact limitation of its own SIGALRM path.

MECHANISM. One daemon watchdog thread polls every registered call. On a violation it
raises a guard exception INSIDE the offending thread via the C API
``PyThreadState_SetAsyncExc``. Delivery happens at the target thread's next bytecode
boundary, so it works in any thread, and it STOPS the runaway loop instead of
abandoning it to keep spinning (a per-call worker subprocess would be interruptible
too, but at tens of thousands of calls per search the fork/pickle cost is far above
the engine calls themselves). If generated code swallows the exception with its own
broad ``except``, the watchdog re-fires on every poll until the call really exits.

WHAT THIS CATCHES, stated plainly:
  * a non-terminating pure-Python loop (the sb26 flood fill), in any thread;
  * a pure spin loop that allocates nothing (timeout channel);
  * unbounded allocation from pure-Python loops (RSS-delta channel, fires within
    one poll interval of crossing the bound -- far below the earlyoom threshold);
  * generated code that swallows the first raise (persistent re-fire).

WHAT THIS DOES NOT CATCH, stated plainly:
  * one C-level call that never returns to bytecode (a pathological single numpy
    op, or C code that never releases the GIL) -- async exceptions cannot
    interrupt C code, and a GIL-holding loop also starves the watchdog;
  * one single giant C-level allocation -- malloc may fail on its own, but this
    guard cannot pre-empt it mid-call (``RLIMIT_AS`` could, but it is process-wide
    and would break legitimate large users such as the llama.cpp server client);
  * RSS attribution is process-wide: another thread's allocation during a guarded
    call counts toward this call's delta. The error direction is over-report --
    a false trip skips one candidate; a miss kills the process.

Guard exceptions derive from ``Exception`` ON PURPOSE: every call site that runs
generated engines already wraps them in ``except Exception``, so a trip degrades
into the existing skip-this-candidate path rather than a new crash class.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Any, Callable, Optional


class EngineCallGuardError(Exception):
    """Base class for guard trips. See the module docstring for why Exception."""


class EngineCallTimeout(EngineCallGuardError):
    """One generated-engine call exceeded its wall-clock budget."""


class EngineCallMemoryExceeded(EngineCallGuardError):
    """Process RSS grew past the allowed delta while this call ran."""


# Two prototypes for the same C function. SETTING needs a real exception type
# (py_object). CLEARING needs a NULL argument, which ctypes only produces through
# c_void_p(None). PYFUNCTYPE keeps the GIL held, which this C API requires.
_SET_ASYNC_EXC = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.py_object)(
    ("PyThreadState_SetAsyncExc", ctypes.pythonapi)
)
_CLEAR_ASYNC_EXC = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)(
    ("PyThreadState_SetAsyncExc", ctypes.pythonapi)
)


class _Entry:
    """One in-flight guarded call, as the watchdog sees it."""

    __slots__ = ("tid", "deadline", "rss_limit", "rss_baseline", "fired", "done")

    def __init__(self, tid: int, deadline: Optional[float], rss_limit: Optional[int]) -> None:
        self.tid = tid
        self.deadline = deadline
        self.rss_limit = rss_limit
        # The watchdog fills the baseline on FIRST SIGHT, not at register time:
        # most calls finish between polls and then never pay for an RSS read.
        self.rss_baseline: Optional[int] = None
        self.fired: Optional[type] = None
        self.done = False


_LOCK = threading.Lock()
_ENTRIES: set[_Entry] = set()
_WATCHDOG_STARTED = False

try:
    _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):  # pragma: no cover - Linux always has it
    _PAGE_SIZE = 4096

_STATM_FD: Optional[int] = None


def _rss_bytes() -> Optional[int]:
    """Current process RSS. None where /proc is unavailable; the memory bound then
    simply does not apply (the timeout bound still does). An open fd + pread keeps
    the per-poll cost at a microsecond or two."""
    global _STATM_FD
    try:
        if _STATM_FD is None:
            _STATM_FD = os.open("/proc/self/statm", os.O_RDONLY)
        return int(os.pread(_STATM_FD, 256, 0).split()[1]) * _PAGE_SIZE
    except Exception:
        return None


def _enabled() -> bool:
    """Kill switch. CARNOT_ARC_ENGINE_CALL_GUARD=0 disables the guard entirely."""
    return os.environ.get("CARNOT_ARC_ENGINE_CALL_GUARD", "1") != "0"


# Parse memos keyed by the raw env string: the hot loop calls these per engine
# call, and a dict hit is ~10x cheaper than float() parsing every time. A test
# that monkeypatches the env gets a fresh parse because the raw string changes.
_TIMEOUT_MEMO: dict[str, Optional[float]] = {}
_RSS_MEMO: dict[str, Optional[int]] = {}


def default_timeout_s() -> Optional[float]:
    """Per-call wall-clock budget in seconds. <=0 disables the timeout channel.
    The 5.0 default is >100x a normal engine call (a whole 48-candidate
    plan_in_model sweep measures ~0.35s) so it cannot clip honest engines."""
    raw = os.environ.get("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "5.0")
    if raw not in _TIMEOUT_MEMO:
        try:
            v = float(raw)
        except ValueError:
            v = 5.0
        _TIMEOUT_MEMO[raw] = v if v > 0 else None
    return _TIMEOUT_MEMO[raw]


def default_rss_delta_bytes() -> Optional[int]:
    """Per-call allowed RSS growth. <=0 disables the memory channel. The 1024 MB
    default is enormous headroom for grid work (a 64x64 int grid is ~16 KB) while
    stopping the incident class ~77 GB before earlyoom would have."""
    raw = os.environ.get("CARNOT_ARC_ENGINE_CALL_RSS_DELTA_MB", "1024")
    if raw not in _RSS_MEMO:
        try:
            mb = float(raw)
        except ValueError:
            mb = 1024.0
        _RSS_MEMO[raw] = int(mb * 1024 * 1024) if mb > 0 else None
    return _RSS_MEMO[raw]


def guard_max_trips() -> int:
    """How many trips one search tolerates before abandoning the engine outright.
    A hanging engine usually hangs on many inputs; paying timeout_s for each of
    thousands of candidates would turn one bad engine into a stalled agent."""
    try:
        return max(1, int(os.environ.get("CARNOT_ARC_ENGINE_GUARD_MAX_TRIPS", "3")))
    except ValueError:
        return 3


def _poll_interval_s() -> float:
    try:
        v = float(os.environ.get("CARNOT_ARC_ENGINE_GUARD_POLL_S", "0.05"))
    except ValueError:
        return 0.05
    return v if v > 0 else 0.05


def _watchdog_loop() -> None:  # pragma: no cover - exercised via guarded_call tests
    while True:
        with _LOCK:
            has_active = any(not e.done for e in _ENTRIES)
        # Idle backoff: with nothing registered there is nothing to time. A call
        # registered mid-sleep is still seen well inside any sane budget
        # (budgets are seconds; this sleep is a quarter second).
        time.sleep(_poll_interval_s() if has_active else 0.25)
        with _LOCK:
            active = [e for e in _ENTRIES if not e.done]
        if not active:
            continue
        rss = _rss_bytes() if any(e.rss_limit is not None for e in active) else None
        now = time.monotonic()
        with _LOCK:
            for e in _ENTRIES:
                if e.done:
                    continue
                exc = e.fired
                if exc is None:
                    if e.deadline is not None and now >= e.deadline:
                        exc = EngineCallTimeout
                    elif e.rss_limit is not None and rss is not None:
                        if e.rss_baseline is None:
                            e.rss_baseline = rss
                        elif rss - e.rss_baseline > e.rss_limit:
                            exc = EngineCallMemoryExceeded
                if exc is not None:
                    # Sticky and RE-FIRED every poll: generated code that swallows
                    # the first raise inside its own broad `except` gets hit again
                    # until the call really exits (`done` flips under this lock).
                    e.fired = exc
                    _SET_ASYNC_EXC(e.tid, exc)


def _ensure_watchdog() -> None:
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:
        return
    with _LOCK:
        if _WATCHDOG_STARTED:
            return
        threading.Thread(target=_watchdog_loop, name="arc-engine-call-guard", daemon=True).start()
        _WATCHDOG_STARTED = True


_UNSET: Any = object()


def guarded_call(
    fn: Callable[..., Any],
    *args: Any,
    timeout_s: Any = _UNSET,
    rss_delta_bytes: Any = _UNSET,
    **kwargs: Any,
) -> Any:
    """Run one generated-engine call under the watchdog.

    Raises EngineCallTimeout / EngineCallMemoryExceeded inside the CALLING thread
    when the call exceeds its budget. Pass ``timeout_s=None`` or
    ``rss_delta_bytes=None`` to disable a channel; omitted arguments use the env
    defaults above. Happy-path cost is a few microseconds (two locked set ops, no
    RSS read); the watchdog only ever sees calls that outlive a poll interval.
    """
    if not _enabled():
        return fn(*args, **kwargs)
    t = default_timeout_s() if timeout_s is _UNSET else timeout_s
    m = default_rss_delta_bytes() if rss_delta_bytes is _UNSET else rss_delta_bytes
    if t is None and m is None:
        return fn(*args, **kwargs)
    entry = _Entry(
        threading.get_ident(),
        None if t is None else time.monotonic() + float(t),
        None if m is None else int(m),
    )
    with _LOCK:
        _ENTRIES.add(entry)
    _ensure_watchdog()
    try:
        return fn(*args, **kwargs)
    finally:
        with _LOCK:
            entry.done = True
            fired = entry.fired
            _ENTRIES.discard(entry)
        if fired is not None:
            # The watchdog may have fired in the instant between the call
            # returning and this cleanup. Clearing the PENDING slot (NULL arg) is
            # a no-op when the exception was already delivered, and it stops an
            # undelivered one surfacing later at a random line in the caller.
            # `done` was set under the lock, so no NEW fire can race this clear.
            # Nested guards on one thread: this clear could eat an OUTER guard's
            # pending fire, but the outer entry is still registered and violated,
            # so the watchdog re-fires it within one poll interval.
            _CLEAR_ASYNC_EXC(entry.tid, None)
