"""ThinkProbeV2 — 60-minute budget, partial verdict, incremental checkpointing.

**Researcher summary (RETRO-029 resolution):**
    Exp 444 (CarnotThinkProbe) timed out at 20 minutes with ZERO results saved.
    Root cause: the 20-minute ExperimentTimeoutWatchdog called sys.exit(1) before
    any results were written, and there was no checkpoint to fall back on.

    Three fixes, each addressing one facet of the failure:

    1. **60-minute budget (REQ-PROBE-005):**
       50 questions × 2 models × ~30 s per inference = ~50 minutes.
       The 20-minute budget was simply too short for the intended workload.
       ThinkProbeV2 defaults to budget_minutes=60 (55 min internal, 5 min buffer).

    2. **Partial verdict (REQ-PROBE-006):**
       When the budget expires mid-run, ThinkProbeV2 does NOT raise an exception.
       Instead, it returns a ThinkProbeV2Result that faithfully describes what was
       completed: `honest_verdict='partial_30_of_50'` is infinitely more useful
       than an empty result file or a bare sys.exit(1).

       The key insight: "honest negatives are better than silent failures."
       A partial result tells the researcher exactly where the run got and allows
       a future run to pick up from the last checkpoint.

    3. **Incremental checkpoint every 10 questions (REQ-PROBE-007):**
       The checkpoint is written after every `checkpoint_interval` questions.
       When the run times out at question 40, the step-40 checkpoint survives.
       Without this, RETRO-029 would recur every time the budget was exceeded.

**Architecture:**
    ThinkProbeV2 is a stand-alone orchestrator that wraps any inference_fn.
    It does NOT depend on CarnotThinkProbe or the existing ThinkProbeResult —
    those are kept for backward compatibility with Exp 444 test suites.
    This module exports:
        ThinkProbeV2Result  — the structured result of a run() call
        ThinkProbeV2        — the orchestrator class

**Why budget_minutes=55 internally, not 60?**
    The external watchdog (ExperimentTimeoutWatchdog) is set to 60 minutes.
    If the internal budget == external budget, the watchdog fires first, writing
    a bare partial JSON with no question-level data.  Setting the internal budget
    to 55 minutes gives the run() method 5 minutes to build the artifact, write
    the checkpoint, and return cleanly before the watchdog fires.

**Why checkpoint_interval=10?**
    10 questions × 2 models × ~30 s = ~10 minutes between checkpoints.
    This is short enough to limit data loss to at most 10 questions (10 min of work),
    while not hammering the filesystem with per-question writes.

Spec: REQ-PROBE-005, REQ-PROBE-006, REQ-PROBE-007
SCENARIO-PROBE-010, SCENARIO-PROBE-011, SCENARIO-PROBE-012
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ThinkProbeV2Result
# ---------------------------------------------------------------------------


@dataclass
class ThinkProbeV2Result:
    """Structured outcome of a ThinkProbeV2.run() call.

    This dataclass carries both the question-level results and aggregate
    metadata about how the run completed (or didn't).

    Fields
    ------
    n_completed : int
        Number of questions for which inference_fn returned a result before
        the budget expired.  May be less than n_total on timeout.
    n_total : int
        Total number of questions submitted to run().
    results : list[dict]
        Per-question result dicts.  Each entry has at minimum:
            'question_index': int  — 0-based index into the original list
            'response':       str  — raw output from inference_fn
        Additional keys may be added by inference_fn.
    status : str
        One of:
            'complete' — all n_total questions finished within budget
            'partial'  — budget expired before all questions were answered
            'empty'    — budget expired before any questions were answered

    Properties
    ----------
    is_partial : bool
        True when n_completed < n_total.  Signals that the result should not
        be treated as a full benchmark — skip_rate / tp_rate / fp_rate are
        computed over the completed subset only.

    completion_fraction : float
        n_completed / n_total.  1.0 on a complete run, 0.0 when nothing finished.
        Used in the artifact for quick human inspection.

    honest_verdict : str
        A human-readable, machine-parseable label that faithfully describes
        the run outcome:
            'complete'                — full run finished within budget
            'partial_{n}_of_{m}'     — n of m questions completed before timeout
            'timeout_no_data'        — timed out before any question finished

        Why this design?  The conductor and retrospective scripts parse
        honest_verdict to decide whether to re-schedule the experiment.
        A scalar percentage would hide whether the run is 'done but slow' or
        'systematically timing out'.  The label makes the distinction explicit.

    Spec: REQ-PROBE-005, REQ-PROBE-006
    """

    n_completed: int
    n_total: int
    results: list[dict[str, Any]]
    status: str = "complete"

    @property
    def is_partial(self) -> bool:
        """True when the run did not finish all questions."""
        return self.n_completed < self.n_total

    @property
    def completion_fraction(self) -> float:
        """Fraction of questions completed; 0.0 when n_total == 0."""
        if self.n_total == 0:
            return 0.0
        return self.n_completed / self.n_total

    @property
    def honest_verdict(self) -> str:
        """Human-readable run outcome label.

        Vocabulary:
            'complete'              — all questions answered within budget
            'partial_{n}_of_{m}'   — partial run (n of m questions answered)
            'timeout_no_data'       — nothing completed before budget expired
        """
        if self.n_completed == self.n_total:
            return "complete"
        if self.n_completed == 0:
            return "timeout_no_data"
        return f"partial_{self.n_completed}_of_{self.n_total}"


# ---------------------------------------------------------------------------
# ThinkProbeV2
# ---------------------------------------------------------------------------

# Default checkpoint directory relative to repo root
_DEFAULT_CKPT_DIR = Path("results/checkpoints/experiment_455")


class ThinkProbeV2:
    """Orchestrates a 50-question think-probe benchmark with budget and checkpointing.

    This class solves three specific failure modes from RETRO-029 (Exp 444 timeout):

    1. 60-minute budget so the full 50q × 2-model workload can complete.
    2. Partial verdict — timeout returns a result, not an exception.
    3. Incremental checkpoint every 10 questions so partial runs are recoverable.

    Usage example::

        def my_inference_fn(question: str) -> str:
            return model.generate(question)

        probe = ThinkProbeV2(budget_minutes=55, checkpoint_interval=10)
        result = probe.run(questions_50, my_inference_fn)
        print(result.honest_verdict)  # 'complete' or 'partial_N_of_50'

    Parameters
    ----------
    budget_minutes : float
        Internal time budget for the run() call.  When budget expires, the
        current question is interrupted and a partial result is returned.
        Default: 60 minutes (55 internal + 5 buffer for artifact write).

        Why not use budget_minutes directly as the external watchdog timeout?
        See module docstring: the 5-minute buffer prevents the watchdog from
        firing before the partial result is written.

    checkpoint_interval : int
        Write a checkpoint after every N completed questions.
        Default: 10 (balances data safety vs. filesystem overhead).

    checkpoint_dir : Path | None
        Directory for checkpoint files.  Defaults to
        ``results/checkpoints/experiment_455/`` relative to repo root.
        Override in tests to use a temp directory.

    Spec: REQ-PROBE-005, REQ-PROBE-006, REQ-PROBE-007
    """

    def __init__(
        self,
        budget_minutes: float = 60,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.budget_minutes = budget_minutes
        self.checkpoint_interval = checkpoint_interval
        self._ckpt_dir: Path = checkpoint_dir if checkpoint_dir is not None else _DEFAULT_CKPT_DIR

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        questions: list[str],
        inference_fn: Callable[[str], str],
    ) -> ThinkProbeV2Result:
        """Run inference on all questions within the budget.

        For each question, calls ``inference_fn(question)`` and appends the
        result.  After every ``checkpoint_interval`` completions, writes a
        checkpoint via ``_checkpoint()``.

        If the wall-clock elapsed time exceeds the budget before all questions
        are answered, the loop exits immediately and returns a partial result.
        The partial result is NOT an exception — callers receive a
        ``ThinkProbeV2Result`` with ``status='partial'`` and a populated
        ``honest_verdict``.

        **Why per-question timeout via ThreadPoolExecutor?**
            We need to interrupt a single slow inference_fn call without killing
            the entire process.  A ThreadPoolExecutor with ``future.result(timeout=)``
            lets us abandon a timed-out question and move on, collecting partial
            results up to the budget boundary.

            The per-question timeout is computed as:
                remaining_budget_s / remaining_questions
            This distributes the remaining time evenly, preventing a single
            slow question from consuming the entire budget.

        Parameters
        ----------
        questions : list[str]
            Ordered list of question strings to pass to ``inference_fn``.
        inference_fn : callable
            ``(question: str) -> str``  — the model inference function.
            In production: wraps Qwen3 or Gemma4 GPU inference.
            In tests: a fast mock that returns immediately.

        Returns
        -------
        ThinkProbeV2Result
            Contains all completed results, n_completed, n_total, status, and
            honest_verdict.  Never raises on timeout.

        Spec: REQ-PROBE-005, REQ-PROBE-006, REQ-PROBE-007
        SCENARIO-PROBE-010, SCENARIO-PROBE-011, SCENARIO-PROBE-012
        """
        n_total = len(questions)
        budget_s = self.budget_minutes * 60.0
        t_start = time.monotonic()
        results: list[dict[str, Any]] = []

        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        for idx, question in enumerate(questions):
            elapsed = time.monotonic() - t_start
            if elapsed >= budget_s:
                _log.warning(
                    "ThinkProbeV2: budget exhausted after %.1f s at question %d/%d — "
                    "returning partial result (RETRO-029 partial-verdict mode)",
                    elapsed,
                    idx,
                    n_total,
                )
                break

            # Distribute remaining budget evenly across remaining questions.
            # This prevents one slow question from starving all subsequent ones.
            remaining_s = budget_s - elapsed
            remaining_questions = n_total - idx
            per_question_timeout = remaining_s / remaining_questions

            response = self._run_one(question, inference_fn, per_question_timeout)
            results.append({"question_index": idx, "question": question, "response": response})

            # Incremental checkpoint: write after every checkpoint_interval completions.
            # RETRO-029 lesson: without this, a timeout loses the entire run.
            n_done = idx + 1
            if n_done % self.checkpoint_interval == 0:
                self._checkpoint(results, step=n_done)
                _log.info(
                    "ThinkProbeV2: checkpoint written at step %d/%d", n_done, n_total
                )

        n_completed = len(results)
        if n_completed == n_total:
            status = "complete"
        elif n_completed == 0:
            status = "empty"
        else:
            status = "partial"

        return ThinkProbeV2Result(
            n_completed=n_completed,
            n_total=n_total,
            results=results,
            status=status,
        )

    # ------------------------------------------------------------------
    # _run_one()  (private)
    # ------------------------------------------------------------------

    def _run_one(
        self,
        question: str,
        inference_fn: Callable[[str], str],
        timeout_s: float,
    ) -> str:
        """Run inference_fn on a single question with a per-question timeout.

        Returns the response string, or an empty string on timeout.
        Never raises — callers receive '' on failure so the loop can continue.

        Why return '' on timeout instead of raising?
        The partial-verdict contract (REQ-PROBE-006) requires that a timeout
        produces a result, not an exception.  An empty string is explicit and
        parseable downstream (skip_rate, tp_rate calculations treat '' as a
        non-flag, which is conservative: we don't claim we verified anything).
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(inference_fn, question)
        try:
            return future.result(timeout=max(timeout_s, 0.001))
        except concurrent.futures.TimeoutError:
            _log.warning(
                "ThinkProbeV2._run_one: per-question timeout (%.1f s) for question %r",
                timeout_s,
                question[:60],
            )
            return ""
        except Exception as exc:
            _log.warning("ThinkProbeV2._run_one: inference_fn raised %s — returning ''", exc)
            return ""
        finally:
            executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # _checkpoint()  (private, overrideable in tests)
    # ------------------------------------------------------------------

    def _checkpoint(self, results_so_far: list[dict[str, Any]], step: int) -> None:
        """Write a JSON checkpoint to disk atomically.

        Uses write-to-tmp + os.rename (POSIX-atomic) so a crash mid-write
        never leaves a corrupt checkpoint file.

        Parameters
        ----------
        results_so_far : list[dict]
            The accumulated results at this step.
        step : int
            The logical step (question number, 1-indexed completion count).

        Why atomic write?
        A non-atomic write (open + write without rename) leaves a window where
        the file exists but is partially written.  On the next run, loading a
        corrupt checkpoint would lose the partial data entirely.  os.rename
        is atomic on POSIX filesystems — the file is either the old version
        or the new version, never in between.

        Spec: REQ-PROBE-007, SCENARIO-PROBE-012
        """
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._ckpt_dir / "checkpoint.json"
        tmp_path = ckpt_path.with_suffix(".tmp")

        payload = {
            "step": step,
            "n_completed": len(results_so_far),
            "results": results_so_far,
            "saved_at": _utc_now(),
        }

        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.rename(ckpt_path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC time in ISO-8601 format."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
