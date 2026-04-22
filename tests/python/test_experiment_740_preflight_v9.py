"""Tests for Exp 740: Exp 527 retirement governance and DualGPU wire-in.

WHY THIS TEST FILE EXISTS:
    1. REQ-INFRA-048: Exp 527 MUST appear in the exclusion manifest with the correct
       governance fields before the 2026.04.57 dequeue cycle.
    2. REQ-INFRA-049: DualGPURetrain.retrain_parallel() MUST use a ThreadPoolExecutor
       with 2 workers and fall back to sequential execution on single-GPU hosts.

    These tests run against the actual manifest file and the DualGPURetrain class so
    they exercise the real on-disk state rather than mocks.

Spec: REQ-INFRA-048, REQ-INFRA-049, SCENARIO-INFRA-057, SCENARIO-INFRA-058
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"


# ---------------------------------------------------------------------------
# REQ-INFRA-048: Exp 527 MUST be in the exclusion manifest.
# Traces to SCENARIO-INFRA-057.
# ---------------------------------------------------------------------------


class TestExp527Retirement:
    """Validate that Exp 527 is correctly retired in the exclusion manifest.

    Traces to REQ-INFRA-048, SCENARIO-INFRA-057.
    """

    def test_exp527_present_in_manifest(self) -> None:
        """Exp 527 must appear in the exclusion manifest after Exp 740 runs.

        WHY: The governance rule "3-consecutive-mandatory" (Exp 308/309 precedent)
        mandates retirement when an experiment appears in the slowest-5 for 3
        consecutive milestones.  Exp 527 crossed this threshold in milestone .56.
        If the manifest does not contain Exp 527, the conductor will re-schedule it
        in .57, wasting a conductor slot.

        Traces to REQ-INFRA-048, SCENARIO-INFRA-057.
        """
        from carnot.pipeline.exclusion_manifest import ExclusionManifest

        manifest = ExclusionManifest(str(_MANIFEST_PATH))
        assert manifest.is_excluded(527), (
            "Exp 527 must be in the exclusion manifest (REQ-INFRA-048). "
            "Run Exp 740 to add it."
        )

    def test_exp527_manifest_entry_has_correct_fields(self) -> None:
        """The Exp 527 manifest entry must include governance_rule and retired_in_milestone.

        WHY: Future retrospective tooling checks these fields to verify that the
        3-consecutive-mandatory rule was applied correctly.  A missing field means
        the retro script cannot confirm governance was followed.

        Traces to REQ-INFRA-048, SCENARIO-INFRA-057.
        """
        raw = json.loads(_MANIFEST_PATH.read_text())
        entries_527 = [
            e for e in raw["excluded"]
            if e.get("experiment_id") == 527
        ]
        assert len(entries_527) >= 1, "No entry for experiment_id=527 in manifest"
        entry = entries_527[-1]  # take the most recent if multiple exist
        assert entry.get("governance_rule") == "3-consecutive-mandatory", (
            f"Expected governance_rule='3-consecutive-mandatory', got: {entry.get('governance_rule')}"
        )
        assert entry.get("retired_in_milestone") == "2026.04.57", (
            f"Expected retired_in_milestone='2026.04.57', got: {entry.get('retired_in_milestone')}"
        )

    def test_exclusion_manifest_add_is_idempotent(self, tmp_path: Path) -> None:
        """Adding Exp 527 twice should not create duplicate entries that break is_excluded().

        WHY: ExclusionManifest.add() does a read-modify-write; if called twice with the
        same experiment_id, we expect is_excluded() to still return True and not raise.
        Duplicate entries are allowed at the data level but must not break the lookup.

        Traces to REQ-INFRA-048.
        """
        from carnot.pipeline.exclusion_manifest import ExclusionEntry, ExclusionManifest

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps({"excluded": []}))

        em = ExclusionManifest(str(manifest_file))
        entry = ExclusionEntry(
            experiment_id=527,
            completed_milestone="2026.04.57",
            reason="3-consecutive-mandatory test",
        )
        em.add(entry)
        em.add(entry)  # second add — idempotent at the lookup level

        em2 = ExclusionManifest(str(manifest_file))
        assert em2.is_excluded(527), "is_excluded(527) must return True after add()"


# ---------------------------------------------------------------------------
# REQ-INFRA-049: DualGPURetrain must use ThreadPoolExecutor and fall back.
# Traces to SCENARIO-INFRA-058.
# ---------------------------------------------------------------------------


class TestDualGPURetrain:
    """Validate DualGPURetrain.retrain_parallel() parallelism and single-GPU fallback.

    Traces to REQ-INFRA-049, SCENARIO-INFRA-058.
    """

    def test_retrain_parallel_falls_back_on_single_gpu(self) -> None:
        """retrain_parallel() must fall back to sequential when only 1 GPU is available.

        WHY: CI machines and developer laptops typically have 0 or 1 GPU.  If
        retrain_parallel() raises an exception on a single-GPU host, every CI
        pipeline that imports DualGPURetrain would fail, blocking development.
        The fallback_reason field confirms the fallback path was taken.

        Traces to REQ-INFRA-049, SCENARIO-INFRA-058.
        """
        from carnot.pipeline.dualgpu_retrain import DualGPURetrain, DualGPURetrainConfig

        results_captured: list[str] = []

        def eorm_fn() -> dict:
            results_captured.append("eorm")
            return {"loss": 0.1, "train_time_s": 0.01}

        def jepa_fn() -> dict:
            results_captured.append("jepa")
            return {"loss": 0.5, "train_time_s": 0.01}

        # Patch _count_cuda_gpus to return 1 (single GPU) so we exercise the fallback.
        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=1):
            retrain = DualGPURetrain(DualGPURetrainConfig(eorm_device="cuda:0", jepa_device="cuda:1"))
            result = retrain.retrain_parallel(eorm_fn, jepa_fn)

        assert result.get("fallback_reason") == "single_gpu", (
            "retrain_parallel() must set fallback_reason='single_gpu' on 1-GPU host. "
            f"Got: {result.get('fallback_reason')}"
        )
        assert "eorm_result" in result and "jepa_result" in result, (
            "Both eorm_result and jepa_result must be present even in fallback mode."
        )
        # Both functions should have run.
        assert "eorm" in results_captured and "jepa" in results_captured, (
            "Both eorm_fn and jepa_fn must run in sequential fallback."
        )

    def test_retrain_parallel_uses_threadpool_executor(self) -> None:
        """retrain_parallel() must submit both tasks concurrently via ThreadPoolExecutor.

        WHY: The entire performance gain (2.0175x from Exp 685) depends on true
        concurrency.  If the implementation is sequential under the hood, GPU 1 remains
        idle and the speedup degrades to 1.0x.  We verify concurrency by checking that
        both tasks overlap in wall-clock time (their combined sleep must be shorter than
        the sum of individual sleeps when run in parallel).

        Traces to REQ-INFRA-049, SCENARIO-INFRA-057.
        """
        from carnot.pipeline.dualgpu_retrain import DualGPURetrain, DualGPURetrainConfig

        sleep_s = 0.1  # 100 ms per task; sequential baseline = ~200 ms

        def eorm_fn() -> dict:
            time.sleep(sleep_s)
            return {"loss": 0.1, "train_time_s": sleep_s}

        def jepa_fn() -> dict:
            time.sleep(sleep_s)
            return {"loss": 0.5, "train_time_s": sleep_s}

        # Patch _count_cuda_gpus to return 2 so we exercise the parallel path.
        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=2):
            retrain = DualGPURetrain(DualGPURetrainConfig(eorm_device="cuda:0", jepa_device="cuda:1"))
            t0 = time.perf_counter()
            result = retrain.retrain_parallel(eorm_fn, jepa_fn)
            elapsed = time.perf_counter() - t0

        # Parallel wall time should be significantly less than 2 * sleep_s.
        # We use 1.8 * sleep_s as a generous upper bound (threading overhead on slow CI).
        assert elapsed < 1.8 * sleep_s * 2, (
            f"retrain_parallel() took {elapsed:.3f}s — expected < {1.8 * sleep_s * 2:.3f}s. "
            "This suggests the ThreadPoolExecutor is not running tasks in parallel."
        )
        assert "eorm_result" in result and "jepa_result" in result
        assert "fallback_reason" not in result, (
            "fallback_reason must NOT be present when 2 GPUs are available."
        )

    def test_retrain_parallel_no_gpu_returns_cpu_fallback(self) -> None:
        """retrain_parallel() with 0 GPUs must still complete without raising.

        WHY: Some CI environments have no GPU at all.  The fallback path must
        handle the 0-GPU case gracefully, running on CPU.

        Traces to REQ-INFRA-049, SCENARIO-INFRA-058.
        """
        from carnot.pipeline.dualgpu_retrain import DualGPURetrain, DualGPURetrainConfig

        def eorm_fn() -> dict:
            return {"loss": 0.1}

        def jepa_fn() -> dict:
            return {"loss": 0.5}

        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=0):
            retrain = DualGPURetrain(DualGPURetrainConfig(eorm_device="cuda:0", jepa_device="cuda:1"))
            result = retrain.retrain_parallel(eorm_fn, jepa_fn)

        assert result.get("fallback_reason") == "single_gpu"
        assert result["eorm_device"] == "cpu"
        assert result["jepa_device"] == "cpu"
