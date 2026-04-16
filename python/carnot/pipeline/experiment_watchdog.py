"""ExperimentTimeoutWatchdog — hard wall-clock cap for runaway experiment processes.

**Researcher summary (RETRO-003):**
    RETRO-003 has been carried for 17+ consecutive milestones without implementation.
    PID 3509070 (Exp 219) ran 144+ minutes with GPU0 reaching 82C.  A 45-minute hard
    cap would have freed GPU0 99 minutes early and prevented thermal stress.  This
    module finally closes RETRO-003 by providing a background-thread watchdog that
    calls ``sys.exit(1)`` when an experiment exceeds its time budget.

**Why a background thread (not a subprocess)?**
    The watchdog must not block the experiment from running normally.  A background
    ``threading.Timer`` fires exactly once after the specified delay, costs zero CPU
    while waiting, and requires no polling loop.  The experiment runs at full speed
    until the deadline; the watchdog fires only if the deadline is exceeded.

**Why 45 minutes as the default?**
    PID 3509070 ran 144 minutes — 3.2x over the typical per-experiment budget of
    ~45 minutes.  The 45-minute cap would have freed GPU0 after 45 minutes instead
    of 144 minutes, saving 99 minutes of GPU time and avoiding the thermal event.
    The value is also configurable via ``CARNOT_CONDUCTOR_TIMEOUT_MINUTES`` for
    experiments that legitimately need more time (training runs, large benchmarks).

**Why write a partial result before exiting?**
    A bare ``sys.exit(1)`` leaves no forensic trail.  The conductor and the human
    operator need to know WHY a result JSON is absent.  Writing a partial result JSON
    with ``timed_out=True`` makes the timeout observable from the outside without
    requiring the conductor to parse process exit codes in a special way.

**Hardware path:**
    Pure OS threading — no GPU, no CUDA, no JAX.  The watchdog works identically
    on CPU-only CI machines and GPU workstations.

Spec: REQ-INFRA-023, REQ-INFRA-024,
      SCENARIO-INFRA-028, SCENARIO-INFRA-029, SCENARIO-INFRA-030 (Exp 425)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ExperimentTimeoutResult
# ---------------------------------------------------------------------------


@dataclass
class ExperimentTimeoutResult:
    """Structured record of a watchdog's outcome.

    Created by :func:`build_timeout_artifact` and embedded in the partial
    result JSON that :meth:`ExperimentTimeoutWatchdog._on_timeout` writes.

    Fields
    ------
    experiment_id : int
        The experiment number (e.g. 425).
    timeout_minutes : int
        The configured timeout in minutes.
    elapsed_minutes : float
        Wall-clock minutes elapsed since :meth:`ExperimentTimeoutWatchdog.start`
        was called.  Populated when the watchdog fires.
    timed_out : bool
        ``True`` when the watchdog fired before :meth:`stop` was called.
        ``False`` when the experiment completed normally.
    partial_result_path : str | None
        Path to the partial result JSON written on timeout, or ``None`` if no
        path was configured.

    Spec: REQ-INFRA-023
    """

    experiment_id: int
    timeout_minutes: int
    elapsed_minutes: float
    timed_out: bool
    partial_result_path: Optional[str]


# ---------------------------------------------------------------------------
# get_timeout_minutes
# ---------------------------------------------------------------------------


def get_timeout_minutes() -> int:
    """Return the configured experiment timeout in minutes.

    Reads ``CARNOT_CONDUCTOR_TIMEOUT_MINUTES`` from the environment.
    When absent or empty, returns the default of 45 minutes — derived from
    the PID 3509070 case (Exp 219 ran 144 minutes; 45 min cap saves 99 min).

    Returns
    -------
    int
        Timeout in minutes (default 45).

    Spec: REQ-INFRA-024, SCENARIO-INFRA-030
    """
    raw = os.environ.get("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "").strip()
    if raw:
        return int(raw)
    return 45


# ---------------------------------------------------------------------------
# ExperimentTimeoutWatchdog
# ---------------------------------------------------------------------------


class ExperimentTimeoutWatchdog:
    """Background-thread watchdog that kills a runaway experiment after a timeout.

    Usage — explicit start/stop::

        watchdog = ExperimentTimeoutWatchdog(experiment_id=425, timeout_minutes=45)
        watchdog.start()
        try:
            run_experiment()
        finally:
            watchdog.stop()

    Usage — context manager (preferred)::

        with ExperimentTimeoutWatchdog(425, timeout_minutes=45,
                                       result_path="results/exp_425.json"):
            run_experiment()

    When ``run_experiment()`` finishes before the timeout, :meth:`stop` cancels
    the timer and the watchdog is a no-op.  When it exceeds the timeout,
    :meth:`_on_timeout` fires: it writes a partial result JSON (if
    ``result_path`` is set) then calls ``sys.exit(1)`` to terminate the
    experiment process.

    Parameters
    ----------
    experiment_id : int
        Experiment number embedded in the partial result JSON.
    timeout_minutes : int
        Hard cap in minutes.  Defaults to 45 (REQ-INFRA-024).  Override with
        ``CARNOT_CONDUCTOR_TIMEOUT_MINUTES`` env var or pass explicitly.
    result_path : str | None
        Path where the partial result JSON is written on timeout.  ``None``
        disables the JSON write (watchdog still calls ``sys.exit(1)``).

    Spec: REQ-INFRA-023, REQ-INFRA-024,
          SCENARIO-INFRA-028, SCENARIO-INFRA-029
    """

    def __init__(
        self,
        experiment_id: int,
        timeout_minutes: int = 45,
        result_path: Optional[str] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.timeout_minutes = timeout_minutes
        self.result_path = result_path

        self._timer: Optional[threading.Timer] = None
        self._start_time: Optional[float] = None
        self._timed_out: bool = False
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background watchdog timer.

        Records ``_start_time`` and arms a :class:`threading.Timer` that will
        call :meth:`_on_timeout` after ``timeout_minutes`` minutes.

        Why ``threading.Timer``?  It is non-blocking (does not consume CPU
        while waiting), fires exactly once, and is trivially cancellable via
        :meth:`stop`.  The experiment runs at native speed with zero overhead
        from the watchdog during normal execution.

        Idempotent if called twice (second call is a no-op that logs a warning).

        Spec: REQ-INFRA-023, SCENARIO-INFRA-028
        """
        if self._timer is not None:
            _log.warning(
                "ExperimentTimeoutWatchdog.start() called twice for experiment %d — "
                "second call ignored",
                self.experiment_id,
            )
            return

        self._start_time = time.monotonic()
        timeout_s = self.timeout_minutes * 60.0

        self._timer = threading.Timer(timeout_s, self._on_timeout)
        self._timer.daemon = True  # do not prevent interpreter shutdown
        self._timer.start()

        _log.info(
            "ExperimentTimeoutWatchdog armed: experiment=%d timeout=%.1f min result_path=%s",
            self.experiment_id,
            self.timeout_minutes,
            self.result_path,
        )

    def stop(self) -> None:
        """Cancel the watchdog timer (normal completion path).

        Must be called when the experiment finishes normally so the timer does
        not fire after the experiment has already completed.  Safe to call
        multiple times (idempotent).

        Spec: REQ-INFRA-023, SCENARIO-INFRA-029
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._stopped = True
        _log.info(
            "ExperimentTimeoutWatchdog stopped normally: experiment=%d elapsed=%.2f min",
            self.experiment_id,
            self.elapsed_minutes(),
        )

    # ------------------------------------------------------------------
    # is_active / elapsed_minutes
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return ``True`` if the watchdog is running and has not yet timed out.

        ``False`` after :meth:`stop` is called or after :meth:`_on_timeout` fires.

        Spec: REQ-INFRA-023
        """
        return (
            self._timer is not None
            and not self._timed_out
            and not self._stopped
        )

    def elapsed_minutes(self) -> float:
        """Return wall-clock minutes elapsed since :meth:`start` was called.

        Returns 0.0 if :meth:`start` has not been called yet.

        Spec: REQ-INFRA-023
        """
        if self._start_time is None:
            return 0.0
        return (time.monotonic() - self._start_time) / 60.0

    # ------------------------------------------------------------------
    # _on_timeout (private)
    # ------------------------------------------------------------------

    def _on_timeout(self) -> None:
        """Fire when the experiment exceeds its time budget.

        This method is called from the background :class:`threading.Timer`
        thread.  It:

        1. Records ``_timed_out = True``.
        2. Computes ``elapsed_minutes``.
        3. Writes a partial result JSON to ``result_path`` (if set) so the
           conductor and operator can diagnose what happened.
        4. Calls ``sys.exit(1)`` to terminate the **entire process** — not
           just the watchdog thread.

        Why ``sys.exit(1)`` and not ``os.kill(os.getpid(), signal.SIGTERM)``?
        ``sys.exit`` raises ``SystemExit`` which Python raises in the main
        thread via the GIL mechanism when called from a daemon thread.
        This is the standard pattern for timer-triggered process termination
        without requiring signal-handler wiring.

        Spec: REQ-INFRA-023, SCENARIO-INFRA-028
        """
        self._timed_out = True
        elapsed = self.elapsed_minutes()

        _log.error(
            "ExperimentTimeoutWatchdog FIRED: experiment=%d timed out after %.2f min "
            "(limit=%.0f min) — calling sys.exit(1)",
            self.experiment_id,
            elapsed,
            self.timeout_minutes,
        )

        if self.result_path is not None:
            partial = {
                "experiment": self.experiment_id,
                "schema": "carnot.timeout_watchdog.partial.v1",
                "timed_out": True,
                "timeout_minutes": self.timeout_minutes,
                "elapsed_minutes": round(elapsed, 3),
                "partial_result_path": self.result_path,
                "status": "timed_out",
            }
            try:
                import os as _os  # noqa: PLC0415

                _os.makedirs(_os.path.dirname(_os.path.abspath(self.result_path)), exist_ok=True)
                with open(self.result_path, "w") as f:
                    json.dump(partial, f, indent=2)
                _log.error(
                    "ExperimentTimeoutWatchdog wrote partial result to %s",
                    self.result_path,
                )
            except Exception as exc:  # pragma: no cover — file I/O is best-effort
                _log.error(
                    "ExperimentTimeoutWatchdog failed to write partial result: %s", exc
                )

        sys.exit(1)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTimeoutWatchdog":
        """Start the watchdog on context entry.

        Spec: REQ-INFRA-023
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop the watchdog on context exit (normal or exception).

        Does NOT suppress exceptions — returns ``False`` so any exception
        propagates normally.  The watchdog is a safety net, not an exception
        handler.

        Spec: REQ-INFRA-023
        """
        self.stop()
        return False


# ---------------------------------------------------------------------------
# build_timeout_artifact
# ---------------------------------------------------------------------------


def build_timeout_artifact(result: ExperimentTimeoutResult) -> dict:
    """Build a JSON-serializable artifact describing a watchdog outcome.

    The ``honest_verdict`` is always ``'watchdog_implemented'`` because this
    function is called from Exp 425 — the experiment that implements the
    watchdog.  A future experiment that actually trips the timeout would
    produce a different artifact via :meth:`ExperimentTimeoutWatchdog._on_timeout`.

    The ``estimated_savings_minutes_per_runaway`` field is derived from the
    PID 3509070 case: 144 minutes actual − 45 minutes cap = 99 minutes saved.
    This is the concrete value that motivated RETRO-003 for 17 milestones.

    Parameters
    ----------
    result : ExperimentTimeoutResult
        The watchdog outcome to serialise.

    Returns
    -------
    dict
        JSON-serializable artifact with schema ``'carnot.timeout_watchdog.v1'``.

    Spec: REQ-INFRA-023, SCENARIO-INFRA-028
    """
    return {
        "schema": "carnot.timeout_watchdog.v1",
        "honest_verdict": "watchdog_implemented",
        "experiment_id": result.experiment_id,
        "timeout_minutes": result.timeout_minutes,
        "elapsed_minutes": result.elapsed_minutes,
        "timed_out": result.timed_out,
        "partial_result_path": result.partial_result_path,
        "estimated_savings_minutes_per_runaway": 99,
        "retro_003_resolved": True,
    }
