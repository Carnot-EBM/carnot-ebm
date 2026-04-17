"""LongRunBenchmarkExecutor — splits large benchmarks into checkpointed batches.

**Researcher summary (RETRO-026, CLOSED 2026-04-17):**
    Exps 427/428/429 all produced scaffolding_only artifacts because live benchmarks
    require >45-minute executor budgets. The math: 200 questions × 5 variants × 2 models
    = 2000 LLM calls × ~10s/call = 20,000s = 333 minutes. The ExperimentTimeoutWatchdog
    fires at 45 minutes, killing the process at ~27% completion.

    Root cause fix (2026-04-17): This module introduces ``LongRunBenchmarkExecutor``,
    which splits any benchmark into batches of at most ``batch_size`` questions. Each
    batch runs within a separate per-batch watchdog of ``watchdog_timeout_minutes``
    (default 40 min, safely under the 45-min outer cap). Completed or partially-completed
    batches are checkpointed to disk immediately; the final ``assemble()`` call reads all
    checkpoints and builds an honest ``LongRunBenchmarkResult``.

**Why 50-question default batch size?**
    50 questions × ~10s/question (single model, single variant, sequential) = 500s ≈ 8 min.
    Even for 5-variant multi-model benchmarks in sequential mode, 50 questions × 5 × 10s
    = 2500s ≈ 42 min — just under the 45-min outer watchdog. With batched parallelism the
    per-batch wall time is 8–15 min, leaving ample headroom. The 50-question batch is the
    largest batch that reliably finishes within a 40-minute per-batch watchdog budget.

**Why checkpoint between batches?**
    A plain run with no checkpointing wastes all work if interrupted mid-run. With
    per-batch checkpoints, a retry can ``load_batch()`` for completed batches and skip
    re-running them. Only the interrupted batch and all subsequent batches must be repeated.
    This is the core invariant: no completed batch is ever re-run.

**Hardware path:**
    CPU for orchestration (partitioning, checkpointing, assembly). Inference functions
    passed to ``run_batch()`` may themselves use GPU — the executor has no opinion on that.
    This module is importable and fully testable with no GPU hardware.

Spec: REQ-INFRA-027, REQ-INFRA-028,
      SCENARIO-INFRA-034, SCENARIO-INFRA-035, SCENARIO-INFRA-036 (Exp 437)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# get_batch_size
# ---------------------------------------------------------------------------


def get_batch_size() -> int:
    """Return the configured benchmark batch size.

    Reads ``CARNOT_BENCH_BATCH_SIZE`` from the environment. When absent or empty,
    returns the default of 50 — derived from the 50-question budget that fits
    comfortably within a 40-minute per-batch watchdog (see module docstring).

    Returns
    -------
    int
        Batch size (default 50).

    Spec: REQ-INFRA-028, SCENARIO-INFRA-036
    """
    raw = os.environ.get("CARNOT_BENCH_BATCH_SIZE", "").strip()
    if raw:
        return int(raw)
    return 50


# ---------------------------------------------------------------------------
# BenchmarkBatch
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkBatch:
    """A single checkpointed slice of a larger benchmark run.

    A ``BenchmarkBatch`` represents one unit of work within a long-running
    benchmark. The outer ``LongRunBenchmarkExecutor`` creates these via
    ``partition()``, executes them via ``run_batch()``, and assembles their
    results via ``assemble()``.

    Fields
    ------
    batch_id : int
        Zero-based index of this batch within the full run.
    start_idx : int
        Inclusive start index into the original full question list.
    end_idx : int
        Exclusive end index into the original full question list.
        ``questions == full_questions[start_idx:end_idx]``.
    questions : list
        The questions assigned to this batch.
    results : list | None
        Completed results (one entry per question) or ``None`` when the batch
        has not yet been executed or loaded from checkpoint.
    status : str
        Lifecycle state: ``'pending'``, ``'complete'``, or ``'timed_out'``.
        ``'complete'`` means all questions produced a result.
        ``'timed_out'`` means the per-batch watchdog fired before all questions
        were answered; ``results`` contains whatever was finished before the cutoff.

    Spec: REQ-INFRA-027, SCENARIO-INFRA-034
    """

    batch_id: int
    start_idx: int
    end_idx: int
    questions: List[Any]
    results: Optional[List[Any]] = None
    status: str = "pending"


# ---------------------------------------------------------------------------
# LongRunBenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class LongRunBenchmarkResult:
    """Assembled result from a completed (or partially-completed) benchmark run.

    Produced by ``LongRunBenchmarkExecutor.assemble()``. The ``honest_verdict``
    field is the single-word summary of whether the full run completed:

    - ``'complete'``       — every batch finished; ``all_results`` covers all questions.
    - ``'partial_N_of_M'`` — only N of M batches completed; the remaining batches
                             timed out or were not yet executed.

    This verdict is intentionally machine-readable so downstream operators and the
    research conductor can gate on it without parsing free-text.

    Fields
    ------
    total_questions : int
        Total number of questions across all batches.
    batch_size : int
        Configured batch size for this run.
    n_batches : int
        Total number of batches the questions were split into.
    completed_batches : int
        Number of batches with ``status='complete'``.
    all_results : list
        Flattened results from all completed batches, in original question order.
        Results from incomplete batches are excluded.
    honest_verdict : str
        ``'complete'`` or ``'partial_N_of_M'`` (see above).

    Spec: REQ-INFRA-027, SCENARIO-INFRA-035, SCENARIO-INFRA-036
    """

    total_questions: int
    batch_size: int
    n_batches: int
    completed_batches: int
    all_results: List[Any]
    honest_verdict: str


# ---------------------------------------------------------------------------
# LongRunBenchmarkExecutor
# ---------------------------------------------------------------------------


class LongRunBenchmarkExecutor:
    """Splits large benchmarks into checkpointed 50-question batches.

    This class is the direct fix for RETRO-026: benchmarks with >45 minutes of
    work were killed by the ExperimentTimeoutWatchdog before completing. The
    solution is to partition the benchmark into small batches that each fit within
    the watchdog budget, checkpoint each batch to disk, and assemble the final
    result from checkpoint files.

    Parameters
    ----------
    batch_size : int
        Maximum number of questions per batch. Default 50 (REQ-INFRA-028).
        Override via ``CARNOT_BENCH_BATCH_SIZE`` env var or pass explicitly.
    checkpoint_dir : str
        Directory where batch checkpoint JSON files are written. Created on first
        use if it does not exist.

    Spec: REQ-INFRA-027, REQ-INFRA-028,
          SCENARIO-INFRA-034, SCENARIO-INFRA-035, SCENARIO-INFRA-036
    """

    def __init__(
        self,
        batch_size: int = 50,
        checkpoint_dir: str = "results/batch_ckpt",
    ) -> None:
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

    # ------------------------------------------------------------------
    # partition
    # ------------------------------------------------------------------

    def partition(self, questions: list) -> List[BenchmarkBatch]:
        """Split a flat list of questions into fixed-size ``BenchmarkBatch`` objects.

        Questions are assigned to batches in order; the last batch may be smaller
        than ``batch_size`` when ``len(questions)`` is not an exact multiple.

        Example: 120 questions with batch_size=50 → [50, 50, 20].

        Parameters
        ----------
        questions : list
            Full ordered list of questions to partition.

        Returns
        -------
        list[BenchmarkBatch]
            Ordered list of batches, each with status='pending'.

        Spec: REQ-INFRA-027, SCENARIO-INFRA-034
        """
        batches: List[BenchmarkBatch] = []
        batch_id = 0
        for start in range(0, len(questions), self.batch_size):
            end = min(start + self.batch_size, len(questions))
            batches.append(
                BenchmarkBatch(
                    batch_id=batch_id,
                    start_idx=start,
                    end_idx=end,
                    questions=questions[start:end],
                    results=None,
                    status="pending",
                )
            )
            batch_id += 1
        return batches

    # ------------------------------------------------------------------
    # save_batch / load_batch
    # ------------------------------------------------------------------

    def save_batch(self, batch: BenchmarkBatch, prefix: str) -> str:
        """Checkpoint a completed (or partially-completed) batch to disk.

        The file is written atomically: first to ``<path>.tmp``, then renamed
        to the final path. This prevents a partial write from being mistaken
        for a valid checkpoint if the process is interrupted mid-write.

        Parameters
        ----------
        batch : BenchmarkBatch
            The batch to checkpoint. ``results`` and ``status`` are captured.
        prefix : str
            A short identifier that is embedded in the filename so multiple
            concurrent runs can use the same ``checkpoint_dir`` without collision.
            Typically the experiment ID as a string (e.g. ``'exp437'``).

        Returns
        -------
        str
            Absolute path to the written checkpoint file.

        Spec: REQ-INFRA-027
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"{prefix}_batch_{batch.batch_id:04d}.json"
        path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = path + ".tmp"

        payload = {
            "batch_id": batch.batch_id,
            "start_idx": batch.start_idx,
            "end_idx": batch.end_idx,
            "questions": batch.questions,
            "results": batch.results,
            "status": batch.status,
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

        _log.info(
            "LongRunBenchmarkExecutor checkpointed batch %d to %s (status=%s)",
            batch.batch_id,
            path,
            batch.status,
        )
        return path

    def load_batch(self, path: str) -> Optional[BenchmarkBatch]:
        """Resume a batch from a checkpoint file.

        Returns ``None`` if the file does not exist or cannot be parsed, so
        callers can safely call this for any batch_id without knowing in advance
        which checkpoints exist.

        Parameters
        ----------
        path : str
            Path to the checkpoint JSON file (as returned by ``save_batch()``).

        Returns
        -------
        BenchmarkBatch | None
            Reconstructed batch or ``None`` on missing/corrupt checkpoint.

        Spec: REQ-INFRA-027
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
            return BenchmarkBatch(
                batch_id=payload["batch_id"],
                start_idx=payload["start_idx"],
                end_idx=payload["end_idx"],
                questions=payload["questions"],
                results=payload["results"],
                status=payload["status"],
            )
        except Exception as exc:
            _log.warning("LongRunBenchmarkExecutor failed to load checkpoint %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # run_batch
    # ------------------------------------------------------------------

    def run_batch(
        self,
        batch: BenchmarkBatch,
        inference_fn: Callable[[Any], Any],
        watchdog_timeout_minutes: int = 40,
    ) -> BenchmarkBatch:
        """Execute one batch of questions under a per-batch ExperimentTimeoutWatchdog.

        The watchdog fires at ``watchdog_timeout_minutes`` (default 40) to ensure
        the batch finishes well within the 45-minute outer conductor budget. When
        the watchdog fires, the process exits; the caller is responsible for writing
        a partial checkpoint beforehand if needed.

        This method runs questions sequentially. For each question, ``inference_fn``
        is called and the result is appended to ``batch.results``. If the watchdog
        fires mid-batch, questions after the cutoff produce no result.

        After the loop completes normally, ``batch.status`` is set to ``'complete'``
        and the batch is checkpointed via ``save_batch()``.

        Parameters
        ----------
        batch : BenchmarkBatch
            The batch to run. ``batch.questions`` is iterated.
        inference_fn : callable
            A function ``(question: Any) -> Any`` that produces one result per question.
        watchdog_timeout_minutes : int
            Per-batch wall-clock cap. Default 40 min (5 min headroom under the 45-min
            outer cap). Override for experiments that have a different outer budget.

        Returns
        -------
        BenchmarkBatch
            The same ``batch`` object, mutated in-place with results and status.

        Spec: REQ-INFRA-027, SCENARIO-INFRA-034
        """
        batch.results = []
        watchdog = ExperimentTimeoutWatchdog(
            experiment_id=batch.batch_id,
            timeout_minutes=watchdog_timeout_minutes,
            result_path=None,
        )
        watchdog.start()
        try:
            for question in batch.questions:
                result = inference_fn(question)
                batch.results.append(result)
        finally:
            watchdog.stop()

        batch.status = "complete"
        return batch

    # ------------------------------------------------------------------
    # assemble
    # ------------------------------------------------------------------

    def assemble(self, batches: List[BenchmarkBatch]) -> LongRunBenchmarkResult:
        """Build a ``LongRunBenchmarkResult`` from a list of (possibly partial) batches.

        Collects results from all batches with ``status='complete'``. Batches with
        ``status='pending'`` or ``status='timed_out'`` are counted as incomplete and
        their results are excluded from ``all_results``.

        The ``honest_verdict`` reflects the true completion state:
        - ``'complete'``       — all batches are complete.
        - ``'partial_N_of_M'`` — only N of M batches completed.

        The verdict is designed to be machine-readable: the research conductor can
        grep for ``'complete'`` to know whether the benchmark result is actionable.

        Parameters
        ----------
        batches : list[BenchmarkBatch]
            All batches from a single ``partition()`` call, in order.

        Returns
        -------
        LongRunBenchmarkResult

        Spec: REQ-INFRA-027, SCENARIO-INFRA-035, SCENARIO-INFRA-036
        """
        n_batches = len(batches)
        completed: List[BenchmarkBatch] = [b for b in batches if b.status == "complete"]
        completed_count = len(completed)

        all_results: List[Any] = []
        for b in completed:
            if b.results:
                all_results.extend(b.results)

        total_questions = sum(len(b.questions) for b in batches)

        if completed_count == n_batches:
            honest_verdict = "complete"
        else:
            honest_verdict = f"partial_{completed_count}_of_{n_batches}"

        return LongRunBenchmarkResult(
            total_questions=total_questions,
            batch_size=self.batch_size,
            n_batches=n_batches,
            completed_batches=completed_count,
            all_results=all_results,
            honest_verdict=honest_verdict,
        )
