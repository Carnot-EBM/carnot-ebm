#!/usr/bin/env python3
"""Experiment 258: Dual-GPU benchmark harness.

Addresses the two top bottlenecks identified in the 2026-04-18 operational retrospective:
  - #1: Sequential dual-model GPU loading  (+15 % estimated wall time)
  - #2: Missing inference batching          (+10 % estimated wall time)

This script wires the DualGPURunner (Exp 224b) and warm ModelServer (Exp 224a) to the
shared Exp 218 benchmark interface so existing runners opt in without full rewrites.
The harness keeps the same function signatures and checkpoint schema as Exp 218 so
benchmark cells (gsm8k_semantic, humaneval_property, constraint_ir) can be dropped in.

Target: ≤ 3 s per case per model (down from 21 s / case observed in Exp 247 on CPU).

Usage (mock/unit mode — no GPU required):
    CARNOT_FORCE_LIVE=0 .venv/bin/pytest \\
        tests/python/test_experiment_258_dual_gpu_harness.py -q --no-cov -n0

Usage (live GPU mode):
    cd /path/to/carnot
    .venv/bin/python scripts/experiment_258_dual_gpu_harness.py \\
        --benchmark constraint_ir

Spec: REQ-VERIFY-041, REQ-VERIFY-036, REQ-VERIFY-037, REQ-VERIFY-038,
      SCENARIO-VERIFY-042, SCENARIO-VERIFY-036, SCENARIO-VERIFY-037
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_DATE = "20260413"
"""Wall-clock date of this experiment run, used in artifact metadata."""

EXPERIMENT = 258
"""Experiment number — matches the filename and artifact JSON ``experiment`` field."""

TARGET_SECONDS_PER_CASE = 3.0
"""Per-case wall-time budget.  Any model that exceeds this is flagged as failing the target."""

MIN_FREE_VRAM_GB = 20.0
"""Minimum free VRAM (in GiB) required on each GPU before the harness starts loading models."""

DEFAULT_BATCH_SIZE = 8
"""Inference batch size used when CARNOT_DUAL_GPU_BATCH_SIZE is not set in the environment."""

# Default model pair — Qwen on GPU 0, Gemma on GPU 1 (matching Exp 218 MODEL_SPECS order).
MODEL_SPECS: list[dict[str, str]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]

# ---------------------------------------------------------------------------
# Utilities (kept identical to Exp 218 so callers can swap scripts transparently)
# ---------------------------------------------------------------------------

_SLUG_KEEP = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


def safe_slug(text: str) -> str:
    """Convert a label into a filesystem-safe slug (identical to Exp 218 helper)."""
    cleaned = text.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(char if char in _SLUG_KEEP else "_" for char in cleaned)


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_repo_root() -> Path:
    """Resolve the repository root, honoring the usual test override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# ThroughputMeasurement
# ---------------------------------------------------------------------------


@dataclass
class _ModelThroughputAccumulator:
    """Running total of cases and elapsed seconds for a single model."""

    total_cases: int = 0
    total_seconds: float = 0.0


class ThroughputMeasurement:
    """Accumulate per-model timing data and report cases/sec against the ≤3 s/case target.

    Usage::

        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=10, elapsed_seconds=18.5)
        report = tm.report()
        # report["per_model"]["Qwen3.5-0.8B"]["target_met"] -> True
    """

    def __init__(self) -> None:
        self._accumulators: dict[str, _ModelThroughputAccumulator] = {}

    def record_batch(self, model_name: str, *, n_cases: int, elapsed_seconds: float) -> None:
        """Add a completed batch result for *model_name*.

        Args:
            model_name: The human-readable model label (e.g. "Qwen3.5-0.8B").
            n_cases: Number of benchmark cases completed in this batch.
            elapsed_seconds: Wall-clock seconds taken to process the batch.
        """
        if model_name not in self._accumulators:
            self._accumulators[model_name] = _ModelThroughputAccumulator()
        acc = self._accumulators[model_name]
        acc.total_cases += n_cases
        acc.total_seconds += elapsed_seconds

    def report(self) -> dict[str, Any]:
        """Return a JSON-serialisable throughput summary.

        Structure::

            {
              "target_seconds_per_case": 3.0,
              "per_model": {
                "<model>": {
                  "total_cases": int,
                  "total_seconds": float,
                  "cases_per_sec": float,
                  "mean_seconds_per_case": float,
                  "target_met": bool
                }
              }
            }
        """
        per_model: dict[str, Any] = {}
        for model_name, acc in self._accumulators.items():
            if acc.total_seconds > 0 and acc.total_cases > 0:
                cases_per_sec = acc.total_cases / acc.total_seconds
                mean_spc = acc.total_seconds / acc.total_cases
            else:
                cases_per_sec = 0.0
                mean_spc = float("inf")
            per_model[model_name] = {
                "total_cases": acc.total_cases,
                "total_seconds": round(acc.total_seconds, 3),
                "cases_per_sec": round(cases_per_sec, 4),
                "mean_seconds_per_case": round(mean_spc, 4),
                "target_met": mean_spc <= TARGET_SECONDS_PER_CASE,
            }
        return {
            "target_seconds_per_case": TARGET_SECONDS_PER_CASE,
            "per_model": per_model,
        }


# ---------------------------------------------------------------------------
# GPUAssignmentVerifier
# ---------------------------------------------------------------------------


class GPUAssignmentVerifier:
    """Verify that both target GPUs exist and have sufficient free VRAM.

    Raises :class:`RuntimeError` at startup if either GPU does not meet the
    MIN_FREE_VRAM_GB threshold.  This surfaces configuration problems early
    instead of mid-run after hours of computation.
    """

    def __init__(self, *, min_free_vram_gb: float = MIN_FREE_VRAM_GB) -> None:
        self._min_free_vram_gb = min_free_vram_gb

    def verify(self, torch_module: Any) -> None:
        """Check both GPUs and raise :class:`RuntimeError` on any violation.

        Args:
            torch_module: The ``torch`` module (or a compatible stub for tests).

        Raises:
            RuntimeError: If CUDA is unavailable, fewer than two GPUs are visible,
                          or either GPU has less than ``min_free_vram_gb`` free.
        """
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not cuda.is_available():
            raise RuntimeError(
                "DualGPUBenchmarkHarness requires CUDA but it is not available on this host. "
                "Run with CARNOT_FORCE_LIVE=0 for unit tests."
            )

        device_count = cuda.device_count()
        if device_count < 2:
            raise RuntimeError(
                f"DualGPUBenchmarkHarness requires at least two CUDA devices but found "
                f"{device_count}.  Assign two GPUs via CUDA_VISIBLE_DEVICES."
            )

        min_bytes = self._min_free_vram_gb * (1024**3)
        for device_index in range(2):
            free_bytes, _total = cuda.mem_get_info(device_index)
            free_gb = free_bytes / (1024**3)
            if free_bytes < min_bytes:
                raise RuntimeError(
                    f"GPU {device_index} has only {free_gb:.1f} GiB free VRAM "
                    f"(required ≥ {self._min_free_vram_gb:.0f} GiB).  "
                    f"Free up VRAM before starting the harness."
                )


# ---------------------------------------------------------------------------
# Checkpoint helpers (identical signatures to Exp 218)
# ---------------------------------------------------------------------------


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    benchmark: str,
    model_name: str,
    mode: str,
) -> Path:
    """Return the per-benchmark/model/mode checkpoint path (Exp 218-compatible)."""
    return checkpoint_dir / (
        f"{safe_slug(benchmark)}__{safe_slug(model_name)}__{safe_slug(mode)}.json"
    )


def _load_checkpoint(path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint when the cohort metadata still matches (Exp 218-compatible)."""
    fresh: dict[str, Any] = {
        "case_ids": list(expected_case_ids),
        "results_by_case": {},
    }
    if not path.exists():
        return fresh

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_ids") != expected_case_ids:
        return fresh
    results_by_case = payload.get("results_by_case", {})
    if not isinstance(results_by_case, dict):
        return fresh
    return {
        **payload,
        "case_ids": list(expected_case_ids),
        "results_by_case": dict(results_by_case),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically (Exp 218-compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# DualGPUBenchmarkHarness
# ---------------------------------------------------------------------------


class DualGPUBenchmarkHarness:
    """Wire DualGPURunner + ModelServer to the shared Exp 218 benchmark interface.

    Design goals
    ------------
    1. **Parallel GPU execution** — assigns Qwen/Qwen3.5-0.8B to GPU 0 and
       google/gemma-4-E4B-it to GPU 1 via DualGPURunner so both models run
       simultaneously instead of sequentially (−15 % wall time).
    2. **Inference batching** — routes generation through ModelServer with a
       configurable batch_size (default 8, env ``CARNOT_DUAL_GPU_BATCH_SIZE``)
       instead of one-at-a-time calls (−10 % wall time).
    3. **VRAM cleanup** — calls ``torch.cuda.empty_cache()`` between benchmark
       runs to prevent CUDA OOM errors on back-to-back benchmark suites.
    4. **Exp 218 compatibility** — exposes ``checkpoint_path``, ``load_checkpoint``,
       ``save_checkpoint``, and ``run_mode`` with identical signatures so callers
       using the Exp 218 helpers do not need to change their code.

    Typical usage (live mode)::

        harness = DualGPUBenchmarkHarness()
        harness.verify_gpu_assignments()   # raises RuntimeError on misconfiguration
        results = harness.run_suite(
            benchmark="constraint_ir",
            cohort=cohort,
            checkpoint_dir=Path("results/checkpoints/experiment_258"),
            policy=policy,
            max_repairs=3,
        )
        harness.empty_cache_between_runs()

    Test / mock mode (CARNOT_FORCE_LIVE=0)::

        harness = DualGPUBenchmarkHarness(torch_module=FakeTorch())
        # verify_gpu_assignments() and run_suite() work fully with injected fakes.
    """

    def __init__(
        self,
        model_specs: list[dict[str, str]] | None = None,
        *,
        batch_size: int | None = None,
        load_model_fn: Callable[..., tuple[Any, Any]] | None = None,
        unload_fn: Callable[[Any, Any], None] | None = None,
        torch_module: Any | None = None,
        clock: Callable[[], float] = perf_counter,
        min_free_vram_gb: float = MIN_FREE_VRAM_GB,
    ) -> None:
        """Initialise the harness.

        Args:
            model_specs: Ordered pair of ``{"name": ..., "hf_id": ...}`` dicts.
                         Defaults to ``MODEL_SPECS`` (Qwen on GPU 0, Gemma on GPU 1).
            batch_size: Inference batch size.  If *None*, read from
                        ``CARNOT_DUAL_GPU_BATCH_SIZE`` env var or fall back to 8.
            load_model_fn: Optional model-loader override (injected in tests).
            unload_fn: Optional model-unloader override (injected in tests).
            torch_module: The ``torch`` module to use.  Injected in tests so no
                          real GPU is required.
            clock: Monotonic clock function for timing (``time.perf_counter`` by default).
            min_free_vram_gb: Minimum free VRAM per device enforced by
                              ``verify_gpu_assignments()``.
        """
        self.model_specs: list[dict[str, str]] = list(model_specs or MODEL_SPECS)

        # Resolve batch_size: explicit arg > env var > default.
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            env_val = os.environ.get("CARNOT_DUAL_GPU_BATCH_SIZE")
            self.batch_size = int(env_val) if env_val else DEFAULT_BATCH_SIZE

        self._load_model_fn = load_model_fn
        self._unload_fn = unload_fn
        self._torch = torch_module
        self._clock = clock
        self._min_free_vram_gb = min_free_vram_gb

        # Accumulates timing data across all run_mode() calls on this harness instance.
        self.throughput = ThroughputMeasurement()

    # ------------------------------------------------------------------
    # GPU verification
    # ------------------------------------------------------------------

    def verify_gpu_assignments(self) -> None:
        """Confirm both GPUs have ≥ MIN_FREE_VRAM_GB free and are correctly assigned.

        Raises:
            RuntimeError: If CUDA is unavailable, fewer than two GPUs are visible,
                          or either GPU has insufficient free VRAM.
        """
        verifier = GPUAssignmentVerifier(min_free_vram_gb=self._min_free_vram_gb)
        verifier.verify(self._torch)

    # ------------------------------------------------------------------
    # Memory cleanup
    # ------------------------------------------------------------------

    def empty_cache_between_runs(self) -> None:
        """Call ``torch.cuda.empty_cache()`` to release cached but unused GPU memory.

        This is a no-op when CUDA is unavailable (e.g. in unit tests with a stub
        torch module that reports ``cuda.is_available() == False``).
        """
        cuda = getattr(self._torch, "cuda", None) if self._torch is not None else None
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()

    # ------------------------------------------------------------------
    # Exp 218-compatible checkpoint interface
    # ------------------------------------------------------------------

    def checkpoint_path(
        self,
        checkpoint_dir: Path,
        *,
        benchmark: str,
        model_name: str,
        mode: str,
    ) -> Path:
        """Return the per-benchmark/model/mode checkpoint path.

        Identical signature to Exp 218 ``checkpoint_path()`` for drop-in compatibility.
        """
        return _checkpoint_path(
            checkpoint_dir,
            benchmark=benchmark,
            model_name=model_name,
            mode=mode,
        )

    def load_checkpoint(self, path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
        """Load a checkpoint when the cohort metadata still matches.

        Identical signature to Exp 218 ``load_checkpoint()`` for drop-in compatibility.
        """
        return _load_checkpoint(path, expected_case_ids)

    def save_checkpoint(self, path: Path, payload: dict[str, Any]) -> None:
        """Write a checkpoint atomically.

        Identical signature to Exp 218 ``save_checkpoint()`` for drop-in compatibility.
        """
        _save_checkpoint(path, payload)

    # ------------------------------------------------------------------
    # run_mode — resume-aware case executor with throughput tracking
    # ------------------------------------------------------------------

    def run_mode(
        self,
        *,
        benchmark: str,
        model_name: str,
        mode: str,
        cases: list[dict[str, Any]],
        checkpoint_dir: Path,
        execute_case: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute one benchmark/model/mode cell with resume support and throughput tracking.

        This is a drop-in replacement for the Exp 218 ``run_mode()`` function.  It adds
        per-case timing so the harness can report whether the ≤ 3 s/case target is met.

        Args:
            benchmark: Benchmark name (e.g. ``"constraint_ir"``).
            model_name: Human-readable model label (e.g. ``"Qwen3.5-0.8B"``).
            mode: Run mode — one of ``"baseline"``, ``"verify_only"``, ``"verify_repair"``.
            cases: Ordered list of case dicts; each must have a ``"case_id"`` key.
            checkpoint_dir: Directory where per-cell checkpoint files are written.
            execute_case: Callable ``(case) -> result_dict`` to run a single case.

        Returns:
            Ordered list of result dicts, one per case.
        """
        case_ids = [str(case["case_id"]) for case in cases]
        ckpt_path = self.checkpoint_path(
            checkpoint_dir,
            benchmark=benchmark,
            model_name=model_name,
            mode=mode,
        )
        checkpoint = self.load_checkpoint(ckpt_path, case_ids)
        results_by_case: dict[str, Any] = dict(checkpoint["results_by_case"])

        for case in cases:
            case_id = str(case["case_id"])
            if case_id in results_by_case:
                continue

            t0 = self._clock()
            result = dict(execute_case(case))
            elapsed = self._clock() - t0

            result.setdefault("case_id", case_id)
            result.setdefault("mode", mode)
            results_by_case[case_id] = result

            # Record timing for throughput report.
            self.throughput.record_batch(model_name, n_cases=1, elapsed_seconds=elapsed)

            self.save_checkpoint(
                ckpt_path,
                {
                    "benchmark": benchmark,
                    "model_name": model_name,
                    "mode": mode,
                    "case_ids": case_ids,
                    "results_by_case": results_by_case,
                },
            )

        return [dict(results_by_case[cid]) for cid in case_ids]

    # ------------------------------------------------------------------
    # run_suite — parallel dual-GPU orchestration
    # ------------------------------------------------------------------

    def run_suite(
        self,
        *,
        benchmark: str,
        cohort: list[dict[str, Any]],
        checkpoint_dir: Path,
        policy: dict[str, Any],
        max_repairs: int,
        runner: Any | None = None,
        suite_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run all configured models against *benchmark* using DualGPURunner.

        Both models are dispatched simultaneously (parallel mode) when two GPUs are
        available, or sequentially with warm ModelServer caching in single-GPU fallback.
        ``empty_cache_between_runs()`` is called once after all tasks complete.

        Args:
            benchmark: The benchmark name (e.g. ``"constraint_ir"``).
            cohort: Ordered list of benchmark cases built by ``build_cohort_manifest()``.
            checkpoint_dir: Directory for per-cell checkpoint files.
            policy: Monitorability policy dict loaded from JSON.
            max_repairs: Maximum verify-repair iterations per case.
            runner: Optional pre-built ``DualGPURunner`` (injected in tests).  When
                    *None* the harness constructs one from ``self.model_specs``.
            suite_fn: Optional ``_run_model_suite``-compatible callable (injected in
                      tests).  When *None* the harness imports ``_run_model_suite``
                      from Exp 218.

        Returns:
            List of per-model result dicts (one per configured model spec).  Each dict
            has at minimum a ``"model_name"`` key and a ``"paired_runs"`` list.
        """
        if runner is None:
            runner = self._build_runner()

        if suite_fn is None:
            suite_fn = self._default_suite_fn()

        harness_self = self  # capture for closure below

        def make_task(model_spec: dict[str, str]) -> Callable[[Any], dict[str, Any]]:
            """Build the per-model task closure that DualGPURunner will invoke."""

            def _task(context: Any) -> dict[str, Any]:
                return suite_fn(
                    benchmark=benchmark,
                    model_spec=model_spec,
                    model=context.model,
                    tokenizer=context.tokenizer,
                    policy=policy,
                    cohort=cohort,
                    checkpoint_dir=checkpoint_dir,
                    max_repairs=max_repairs,
                )

            return _task

        tasks = {spec["name"]: make_task(spec) for spec in self.model_specs}
        runner_results = runner.run_model_tasks(tasks)

        # Cleanup GPU memory after all tasks complete.
        self.empty_cache_between_runs()

        return [
            {
                "model_name": res.model_name,
                **res.payload,
            }
            for res in runner_results
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_runner(self) -> Any:
        """Construct a DualGPURunner from the harness configuration."""
        from carnot.inference.dual_gpu import DualGPURunner
        from carnot.inference.model_server import ModelServer

        kwargs: dict[str, Any] = {
            "torch_module": self._torch,
        }
        if self._load_model_fn is not None:
            kwargs["load_model_fn"] = self._load_model_fn
        if self._unload_fn is not None:
            kwargs["unload_fn"] = self._unload_fn

        # Attempt to wire a warm ModelServer for batched inference.
        hf_ids = [spec["hf_id"] for spec in self.model_specs]
        try:
            server = ModelServer(
                hf_ids,
                batch_size=self.batch_size,
                torch_module=self._torch,
            )
            server.start()
            kwargs["model_server"] = server
        except Exception as exc:  # noqa: BLE001 — optional warm server
            print(f"  ModelServer unavailable ({exc}), proceeding without warm cache")

        return DualGPURunner(self.model_specs, **kwargs)

    @staticmethod
    def _default_suite_fn() -> Callable[..., dict[str, Any]]:
        """Return the ``_run_model_suite`` callable from Exp 218."""
        import importlib.util as ilu

        script_path = Path(__file__).parent / "experiment_218_live_dual_model_suite.py"
        spec = ilu.spec_from_file_location("experiment_218", script_path)
        assert spec is not None and spec.loader is not None
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod._run_model_suite  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Harness report artifact
# ---------------------------------------------------------------------------


def write_harness_report(
    path: Path,
    *,
    throughput: ThroughputMeasurement,
    run_date: str = RUN_DATE,
) -> None:
    """Write the Exp 258 harness validation report to *path* as JSON.

    The artifact documents throughput achieved and whether the ≤ 3 s/case target
    was met.  Parent directories are created automatically.

    Args:
        path: Destination path for the JSON report.
        throughput: Populated ``ThroughputMeasurement`` from the harness run.
        run_date: Run date string in ``YYYYMMDD`` format (defaults to ``RUN_DATE``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    t_report = throughput.report()
    per_model = t_report["per_model"]

    # Overall target is met only when every tracked model meets the per-case budget
    # and at least one model has been measured.
    if per_model:
        overall_target_met: bool = all(
            entry["target_met"] for entry in per_model.values()
        )
    else:
        overall_target_met = False

    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "run_date": run_date,
        "schema": "carnot.dual_gpu_harness_report.v1",
        "target_seconds_per_case": TARGET_SECONDS_PER_CASE,
        "target_met": overall_target_met,
        "throughput": t_report,
    }

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point.

    Run the dual-GPU harness against one of the Exp 218 benchmarks and write a
    harness report artifact.

    Environment variables
    ---------------------
    CARNOT_FORCE_LIVE : "1" enables live GPU execution (default "0" = mock / test).
    CARNOT_DUAL_GPU_BATCH_SIZE : Override the inference batch size (default 8).
    JAX_PLATFORMS : Set to "cpu" to keep JAX from touching the GPU.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 258: Dual-GPU benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=["gsm8k_semantic", "humaneval_property", "constraint_ir"],
        default="constraint_ir",
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of benchmark cases to sample.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=258,
        help="Random seed for cohort sampling.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=3,
        help="Maximum verify-repair iterations per case.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=get_repo_root() / "results" / "checkpoints" / "experiment_258",
        help="Directory for incremental checkpoint files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_repo_root() / "results" / "experiment_258_harness_report.json",
        help="Path for the harness report artifact.",
    )
    args = parser.parse_args(argv)

    harness = DualGPUBenchmarkHarness()

    print(f"[Exp 258] Verifying GPU assignments (≥ {MIN_FREE_VRAM_GB:.0f} GiB free each)…")
    harness.verify_gpu_assignments()
    print(f"  GPU 0: {MODEL_SPECS[0]['hf_id']}")
    print(f"  GPU 1: {MODEL_SPECS[1]['hf_id']}")
    print(f"  Batch size: {harness.batch_size}")

    started_at = utc_now()
    t_start = perf_counter()

    # Load cohort using Exp 218 helpers.
    suite_mod = DualGPUBenchmarkHarness._default_suite_fn.__func__(None)  # type: ignore[attr-defined]
    # Re-import the module properly.
    import importlib.util as ilu

    exp218_path = Path(__file__).parent / "experiment_218_live_dual_model_suite.py"
    spec218 = ilu.spec_from_file_location("experiment_218", exp218_path)
    assert spec218 is not None and spec218.loader is not None
    exp218 = ilu.module_from_spec(spec218)
    spec218.loader.exec_module(exp218)  # type: ignore[union-attr]

    records = exp218._load_benchmark_records(args.benchmark)
    cohort = exp218.build_cohort_manifest(
        records,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    policy_path = get_repo_root() / "results" / "output_policy_233.json"
    if not policy_path.exists():
        policy_path = get_repo_root() / "results" / "monitorability_policy_213.json"
    policy = exp218.load_monitorability_policy(policy_path)

    print(f"[Exp 258] Running {args.benchmark} with {len(cohort)} cases…")
    results = harness.run_suite(
        benchmark=args.benchmark,
        cohort=cohort,
        checkpoint_dir=args.checkpoint_dir,
        policy=policy,
        max_repairs=args.max_repairs,
    )

    finished_at = utc_now()
    runtime = perf_counter() - t_start

    t_report = harness.throughput.report()
    print(f"\n[Exp 258] Throughput report:")
    for model_name, entry in t_report["per_model"].items():
        target_flag = "✓" if entry["target_met"] else "✗"
        print(
            f"  {target_flag} {model_name}: "
            f"{entry['mean_seconds_per_case']:.2f} s/case  "
            f"({entry['cases_per_sec']:.3f} cases/s)"
        )

    write_harness_report(args.output, throughput=harness.throughput)
    print(f"\n[Exp 258] Report written to {args.output}")
    print(f"[Exp 258] Runtime: {runtime:.1f} s  |  started: {started_at}  finished: {finished_at}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DualGPUBenchmarkHarness",
    "GPUAssignmentVerifier",
    "ThroughputMeasurement",
    "write_harness_report",
    "MODEL_SPECS",
    "TARGET_SECONDS_PER_CASE",
    "MIN_FREE_VRAM_GB",
    "DEFAULT_BATCH_SIZE",
    "EXPERIMENT",
    "RUN_DATE",
]


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
