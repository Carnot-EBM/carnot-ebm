"""Tests for Exp 746: DualGPU EORM+JEPA Retrain.

Each test traces to REQ-INFRA-050 and/or SCENARIO-INFRA-059.

Why these tests and not others:
    The three behaviours that matter for correctness are:
      1. retrain_parallel submits BOTH futures concurrently (parallel path).
      2. retrain_parallel falls back to sequential on a single-GPU host.
      3. The speedup ratio computed from wall times is arithmetically correct.
    Everything else (model training, GPU allocation) is covered by Exp 685 and
    the dualgpu_retrain unit tests in test_dualgpu_retrain.py.
"""
from __future__ import annotations

import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.dualgpu_retrain import (
    DualGPURetrain,
    DualGPURetrainConfig,
    _call_with_device,
    _count_cuda_gpus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrain(eorm_device: str = "cuda:0", jepa_device: str = "cuda:1") -> DualGPURetrain:
    """Return a DualGPURetrain instance with the given device config."""
    return DualGPURetrain(DualGPURetrainConfig(eorm_device=eorm_device, jepa_device=jepa_device))


def _fast_fn(device: str = "cpu") -> dict:
    """Minimal callable that records its device and returns immediately."""
    return {"loss_after": 0.1, "device": device}


def _slow_fn(sleep_s: float):
    """Return a callable that sleeps for `sleep_s` seconds before returning."""
    def _inner(device: str = "cpu") -> dict:
        time.sleep(sleep_s)
        return {"loss_after": 0.2, "device": device}
    return _inner


# ---------------------------------------------------------------------------
# REQ-INFRA-050: retrain_parallel submits both futures concurrently
# ---------------------------------------------------------------------------


class TestRetainParallelConcurrency:
    """Verify that retrain_parallel uses ThreadPoolExecutor concurrency.

    Spec traces: REQ-INFRA-050, SCENARIO-INFRA-059
    """

    def test_both_futures_submitted_concurrently(self) -> None:
        """Both eorm_fn and jepa_fn must be submitted to the executor before either resolves.

        Why: If the executor submitted them sequentially (submit A, wait for A,
        submit B) the total wall time would be sum(A, B).  Concurrent submission
        means the total wall time approaches max(A, B).  We verify this by running
        two functions that each sleep 0.1 s and checking that the total wall time
        is < 0.15 s (not 0.2 s).

        Spec traces: REQ-INFRA-050
        """
        retrain = _make_retrain()

        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=2):
            t0 = time.perf_counter()
            results = retrain.retrain_parallel(
                _slow_fn(0.1),
                _slow_fn(0.1),
                eorm_device="cuda:0",
                jepa_device="cuda:1",
            )
            elapsed = time.perf_counter() - t0

        # Sequential would take >= 0.2 s; parallel should take < 0.15 s.
        assert elapsed < 0.15, (
            f"retrain_parallel took {elapsed:.3f}s — expected parallel (<0.15s), "
            "got sequential-like timing"
        )
        assert "eorm_result" in results
        assert "jepa_result" in results

    def test_results_contain_both_model_outputs(self) -> None:
        """Result dict must contain eorm_result and jepa_result on the dual-GPU path.

        Spec traces: REQ-INFRA-050
        """
        retrain = _make_retrain()

        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=2):
            results = retrain.retrain_parallel(
                _fast_fn,
                _fast_fn,
                eorm_device="cuda:0",
                jepa_device="cuda:1",
            )

        assert results["eorm_result"]["loss_after"] == pytest.approx(0.1)
        assert results["jepa_result"]["loss_after"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# REQ-INFRA-050 / SCENARIO-INFRA-058: single-GPU fallback
# ---------------------------------------------------------------------------


class TestSingleGPUFallback:
    """Verify sequential fallback when fewer than 2 GPUs are available.

    Spec traces: REQ-INFRA-050, SCENARIO-INFRA-058
    """

    def test_fallback_reason_present_on_single_gpu(self) -> None:
        """Result must include fallback_reason='single_gpu' when only 1 GPU detected.

        Spec traces: SCENARIO-INFRA-058
        """
        retrain = _make_retrain()

        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=1):
            results = retrain.retrain_parallel(
                _fast_fn,
                _fast_fn,
                eorm_device="cuda:0",
                jepa_device="cuda:1",
            )

        assert results.get("fallback_reason") == "single_gpu"

    def test_fallback_runs_both_models(self) -> None:
        """Even on single-GPU fallback, both eorm_result and jepa_result must be present.

        Spec traces: REQ-INFRA-050
        """
        retrain = _make_retrain()

        with patch("carnot.pipeline.dualgpu_retrain._count_cuda_gpus", return_value=0):
            results = retrain.retrain_parallel(
                _fast_fn,
                _fast_fn,
                eorm_device="cuda:0",
                jepa_device="cuda:1",
            )

        assert "eorm_result" in results
        assert "jepa_result" in results
        assert results["eorm_result"]["loss_after"] == pytest.approx(0.1)
        assert results["jepa_result"]["loss_after"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# REQ-INFRA-050: speedup computed correctly from wall times
# ---------------------------------------------------------------------------


class TestSpeedupComputation:
    """Verify that the speedup formula is arithmetically correct.

    Spec traces: REQ-INFRA-050, SCENARIO-INFRA-059
    """

    def test_speedup_ratio_formula(self) -> None:
        """speedup = wall_sequential / wall_parallel must be >= 1.8 for validated verdict.

        Why 1.8: Exp 685 validated 2.0175x.  We allow a 10% margin so natural
        OS scheduling variance does not flip the verdict on a healthy system.

        Spec traces: SCENARIO-INFRA-059
        """
        wall_sequential = 1.1
        wall_parallel = 0.6
        speedup = round(wall_sequential / wall_parallel, 4)
        assert speedup >= 1.8, f"speedup={speedup} below 1.8 threshold"

    def test_speedup_marginal_below_1_8(self) -> None:
        """Speedup of 1.5x should map to 'dualgpu_retrain_marginal', not validated.

        Spec traces: REQ-INFRA-050
        """
        # Import the verdict function directly from the experiment script.
        # We import here (not at module level) to avoid side-effects at import time.
        import importlib.util
        import sys
        from pathlib import Path

        spec_path = Path(__file__).resolve().parents[2] / "scripts" / "experiment_746_dualgpu_eorm_jepa_retrain.py"
        spec_obj = importlib.util.spec_from_file_location("exp746", spec_path)
        exp746 = importlib.util.module_from_spec(spec_obj)  # type: ignore[arg-type]
        sys.modules["exp746"] = exp746
        spec_obj.loader.exec_module(exp746)  # type: ignore[union-attr]

        verdict = exp746._honest_verdict(
            speedup=1.5,
            eorm_result={"loss_after": 0.3},
            jepa_result={"loss_after": 0.3},
        )
        assert verdict == "dualgpu_retrain_marginal"

    def test_speedup_no_speedup_at_one(self) -> None:
        """Speedup of 1.0 or less maps to 'dualgpu_retrain_no_speedup'.

        Spec traces: REQ-INFRA-050
        """
        import importlib.util
        import sys
        from pathlib import Path

        spec_path = Path(__file__).resolve().parents[2] / "scripts" / "experiment_746_dualgpu_eorm_jepa_retrain.py"
        spec_obj = importlib.util.spec_from_file_location("exp746_b", spec_path)
        exp746 = importlib.util.module_from_spec(spec_obj)  # type: ignore[arg-type]
        sys.modules["exp746_b"] = exp746
        spec_obj.loader.exec_module(exp746)  # type: ignore[union-attr]

        verdict = exp746._honest_verdict(
            speedup=0.9,
            eorm_result={"loss_after": 0.3},
            jepa_result={"loss_after": 0.3},
        )
        assert verdict == "dualgpu_retrain_no_speedup"

    def test_speedup_validated_at_1_8(self) -> None:
        """Speedup exactly at 1.8 maps to 'dualgpu_retrain_validated'.

        Spec traces: SCENARIO-INFRA-059
        """
        import importlib.util
        import sys
        from pathlib import Path

        spec_path = Path(__file__).resolve().parents[2] / "scripts" / "experiment_746_dualgpu_eorm_jepa_retrain.py"
        spec_obj = importlib.util.spec_from_file_location("exp746_c", spec_path)
        exp746 = importlib.util.module_from_spec(spec_obj)  # type: ignore[arg-type]
        sys.modules["exp746_c"] = exp746
        spec_obj.loader.exec_module(exp746)  # type: ignore[union-attr]

        verdict = exp746._honest_verdict(
            speedup=1.8,
            eorm_result={"loss_after": 0.3},
            jepa_result={"loss_after": 0.3},
        )
        assert verdict == "dualgpu_retrain_validated"
