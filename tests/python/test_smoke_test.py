"""Tests for python/carnot/pipeline/smoke_test.py.

Spec coverage: REQ-BENCH-005, SCENARIO-BENCH-012, SCENARIO-BENCH-013

Design rationale:
    These tests exercise the smoke test gate that must run before any benchmark
    experiment.  The key invariants are:

    1. CI-safe path: when CARNOT_FORCE_LIVE is not set, run_smoke_test returns a
       SmokeTestResult with is_live=False and inference_mode="ci_skip" without
       raising.  This guarantees CI stays green without GPUs.

    2. Live-required path: when CARNOT_FORCE_LIVE=1 but the GPU/model is
       unavailable, run_smoke_test raises RuntimeError rather than silently
       falling back to simulated mode.  Silent fallback produced artifacts labelled
       "live_gpu" that actually contained synthetic answers (Exps 340-347 bug).

    3. build_smoke_test_artifact produces honest_verdict: "live_confirmed" only
       when is_live=True.  Any other state maps to "blocked_simulated" or
       "blocked_error".

All heavy imports (transformers, torch) are patched out so these tests run
under JAX_PLATFORMS=cpu with no GPU hardware.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.smoke_test import (
    SmokeTestResult,
    build_smoke_test_artifact,
    run_smoke_test,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_force_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CARNOT_FORCE_LIVE is unset for CI-safe tests."""
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)


@pytest.fixture()
def force_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set CARNOT_FORCE_LIVE=1 for live-mode tests."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")


# ---------------------------------------------------------------------------
# SmokeTestResult dataclass tests
# ---------------------------------------------------------------------------


class TestSmokeTestResult:
    """REQ-BENCH-005: SmokeTestResult dataclass — SCENARIO-BENCH-012."""

    def test_fields_accessible(self) -> None:
        """SCENARIO-BENCH-012: dataclass has all required fields."""
        result = SmokeTestResult(
            inference_mode="ci_skip",
            n_questions=5,
            n_answered=0,
            elapsed_s=0.0,
            model_id="google/gemma-4-E4B-it",
            is_live=False,
            blocked_reason="CARNOT_FORCE_LIVE not set",
        )
        assert result.inference_mode == "ci_skip"
        assert result.n_questions == 5
        assert result.n_answered == 0
        assert result.elapsed_s == 0.0
        assert result.model_id == "google/gemma-4-E4B-it"
        assert result.is_live is False
        assert result.blocked_reason == "CARNOT_FORCE_LIVE not set"

    def test_serializable_to_dict(self) -> None:
        """SCENARIO-BENCH-012: SmokeTestResult is serializable via dataclasses.asdict."""
        result = SmokeTestResult(
            inference_mode="live_gpu",
            n_questions=5,
            n_answered=5,
            elapsed_s=12.3,
            model_id="google/gemma-4-E4B-it",
            is_live=True,
            blocked_reason="",
        )
        d = asdict(result)
        assert d["is_live"] is True
        assert d["inference_mode"] == "live_gpu"

    def test_live_false_defaults(self) -> None:
        """SCENARIO-BENCH-012: blocked_reason defaults to empty string."""
        result = SmokeTestResult(
            inference_mode="ci_skip",
            n_questions=5,
            n_answered=0,
            elapsed_s=0.0,
            model_id="m",
            is_live=False,
            blocked_reason="",
        )
        assert result.blocked_reason == ""


# ---------------------------------------------------------------------------
# run_smoke_test CI-skip path (CARNOT_FORCE_LIVE not set)
# ---------------------------------------------------------------------------


class TestRunSmokeTestCISkip:
    """REQ-BENCH-005: CI-skip path — SCENARIO-BENCH-012."""

    def test_returns_ci_skip_when_no_force_live(self, no_force_live: None) -> None:
        """SCENARIO-BENCH-012: returns ci_skip result without raising."""
        result = run_smoke_test("google/gemma-4-E4B-it", n_questions=5)
        assert isinstance(result, SmokeTestResult)
        assert result.is_live is False
        assert result.inference_mode == "ci_skip"
        assert result.blocked_reason == "CARNOT_FORCE_LIVE not set"

    def test_ci_skip_model_id_preserved(self, no_force_live: None) -> None:
        """SCENARIO-BENCH-012: model_id is preserved in ci_skip result."""
        result = run_smoke_test("google/gemma-4-E4B-it", n_questions=3)
        assert result.model_id == "google/gemma-4-E4B-it"

    def test_ci_skip_n_questions_preserved(self, no_force_live: None) -> None:
        """SCENARIO-BENCH-012: n_questions is preserved in ci_skip result."""
        result = run_smoke_test("google/gemma-4-E4B-it", n_questions=3)
        assert result.n_questions == 3

    def test_ci_skip_n_answered_is_zero(self, no_force_live: None) -> None:
        """SCENARIO-BENCH-012: n_answered=0 in ci_skip (no inference ran)."""
        result = run_smoke_test("google/gemma-4-E4B-it")
        assert result.n_answered == 0

    def test_ci_skip_elapsed_s_is_zero(self, no_force_live: None) -> None:
        """SCENARIO-BENCH-012: elapsed_s=0.0 in ci_skip (no inference ran)."""
        result = run_smoke_test("google/gemma-4-E4B-it")
        assert result.elapsed_s == 0.0

    def test_ci_skip_does_not_raise(self, no_force_live: None) -> None:
        """SCENARIO-BENCH-012: ci_skip path never raises RuntimeError."""
        # This must not raise — CI must stay green.
        try:
            run_smoke_test("google/gemma-4-E4B-it")
        except RuntimeError:
            pytest.fail("run_smoke_test raised RuntimeError on CI-skip path")


# ---------------------------------------------------------------------------
# run_smoke_test live-mode path (CARNOT_FORCE_LIVE=1)
# ---------------------------------------------------------------------------


class TestRunSmokeTestLiveMode:
    """REQ-BENCH-005: live-mode path — SCENARIO-BENCH-013."""

    def _make_healthy_prewarm(self) -> Any:
        """Return a mock prewarm_fn that reports a healthy GPU."""
        prewarm_result = MagicMock()
        prewarm_result.health_ok = True
        prewarm_result.load_time_s = 1.0
        prewarm_result.stall_root_cause = None
        prewarm_fn = MagicMock(return_value=prewarm_result)
        return prewarm_fn

    def _make_unhealthy_prewarm(self) -> Any:
        """Return a mock prewarm_fn that reports an unhealthy GPU."""
        prewarm_result = MagicMock()
        prewarm_result.health_ok = False
        prewarm_result.load_time_s = 0.0
        prewarm_result.stall_root_cause = "CUDA not available"
        prewarm_fn = MagicMock(return_value=prewarm_result)
        return prewarm_fn

    def test_raises_runtime_error_when_prewarm_fails(self, force_live: None) -> None:
        """SCENARIO-BENCH-013: RuntimeError raised when GPU pre-warm fails."""
        unhealthy_prewarm = self._make_unhealthy_prewarm()

        # Patch diagnose_live_gpu at its source module (lazy import in smoke_test).
        mock_diag = MagicMock()
        mock_diag.failure_reason = "CUDA not available"

        with patch(
            "carnot.pipeline.smoke_test._prewarm_model",
            unhealthy_prewarm,
        ), patch(
            "carnot.pipeline.live_gpu_diagnostic.diagnose_live_gpu",
            return_value=mock_diag,
        ):
            with pytest.raises(RuntimeError, match="Live GPU required"):
                run_smoke_test("google/gemma-4-E4B-it", n_questions=5)

    def test_returns_live_result_when_prewarm_succeeds(self, force_live: None) -> None:
        """SCENARIO-BENCH-013: returns live_gpu result when pre-warm succeeds."""
        healthy_prewarm = self._make_healthy_prewarm()
        mock_model = MagicMock(return_value=[{"generated_text": "Step 1: 3 + 2 = 5.\n#### 5"}])

        mock_monitor_instance = MagicMock()
        mock_monitor_instance.check_dual_gpu_health.return_value = {
            "all_healthy": True,
            "n_gpus_detected": 1,
            "n_zombies": 0,
            "idle_gpus": [],
        }
        mock_monitor_cls = MagicMock(return_value=mock_monitor_instance)

        with patch("carnot.pipeline.smoke_test._prewarm_model", healthy_prewarm), patch(
            "carnot.pipeline.smoke_test._load_model_for_smoke_test",
            return_value=mock_model,
        ), patch(
            "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor",
            mock_monitor_cls,
        ):
            result = run_smoke_test("google/gemma-4-E4B-it", n_questions=2)

        assert isinstance(result, SmokeTestResult)
        assert result.is_live is True
        assert result.inference_mode == "live_gpu"
        assert result.model_id == "google/gemma-4-E4B-it"
        assert result.n_questions == 2
        assert result.elapsed_s >= 0.0

    def test_live_result_n_answered_counts_responses(self, force_live: None) -> None:
        """SCENARIO-BENCH-013: n_answered counts non-empty responses."""
        healthy_prewarm = self._make_healthy_prewarm()
        mock_model = MagicMock(return_value=[{"generated_text": "#### 10"}])

        mock_monitor_instance = MagicMock()
        mock_monitor_instance.check_dual_gpu_health.return_value = {
            "all_healthy": True,
            "n_gpus_detected": 1,
            "n_zombies": 0,
            "idle_gpus": [],
        }

        with patch("carnot.pipeline.smoke_test._prewarm_model", healthy_prewarm), patch(
            "carnot.pipeline.smoke_test._load_model_for_smoke_test",
            return_value=mock_model,
        ), patch(
            "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor",
            return_value=mock_monitor_instance,
        ):
            result = run_smoke_test("google/gemma-4-E4B-it", n_questions=3)

        assert result.n_answered == 3

    def test_live_model_load_failure_raises(self, force_live: None) -> None:
        """SCENARIO-BENCH-013: RuntimeError raised if model load fails post-prewarm."""
        healthy_prewarm = self._make_healthy_prewarm()

        mock_monitor_instance = MagicMock()
        mock_monitor_instance.check_dual_gpu_health.return_value = {
            "all_healthy": True,
            "n_gpus_detected": 1,
            "n_zombies": 0,
            "idle_gpus": [],
        }

        with patch("carnot.pipeline.smoke_test._prewarm_model", healthy_prewarm), patch(
            "carnot.pipeline.smoke_test._load_model_for_smoke_test",
            side_effect=RuntimeError("model load failed"),
        ), patch(
            "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor",
            return_value=mock_monitor_instance,
        ):
            with pytest.raises(RuntimeError):
                run_smoke_test("google/gemma-4-E4B-it", n_questions=2)


# ---------------------------------------------------------------------------
# build_smoke_test_artifact
# ---------------------------------------------------------------------------


class TestBuildSmokeTestArtifact:
    """REQ-BENCH-005: artifact builder — SCENARIO-BENCH-012, SCENARIO-BENCH-013."""

    def test_schema_field(self) -> None:
        """SCENARIO-BENCH-012: schema is 'carnot.smoke_test.v1'."""
        result = SmokeTestResult(
            inference_mode="ci_skip",
            n_questions=5,
            n_answered=0,
            elapsed_s=0.0,
            model_id="google/gemma-4-E4B-it",
            is_live=False,
            blocked_reason="CARNOT_FORCE_LIVE not set",
        )
        artifact = build_smoke_test_artifact(result)
        assert artifact["schema"] == "carnot.smoke_test.v1"

    def test_honest_verdict_blocked_simulated_ci_skip(self) -> None:
        """SCENARIO-BENCH-012: ci_skip → honest_verdict='blocked_simulated'."""
        result = SmokeTestResult(
            inference_mode="ci_skip",
            n_questions=5,
            n_answered=0,
            elapsed_s=0.0,
            model_id="google/gemma-4-E4B-it",
            is_live=False,
            blocked_reason="CARNOT_FORCE_LIVE not set",
        )
        artifact = build_smoke_test_artifact(result)
        assert artifact["honest_verdict"] == "blocked_simulated"

    def test_honest_verdict_live_confirmed(self) -> None:
        """SCENARIO-BENCH-013: is_live=True → honest_verdict='live_confirmed'."""
        result = SmokeTestResult(
            inference_mode="live_gpu",
            n_questions=5,
            n_answered=5,
            elapsed_s=8.2,
            model_id="google/gemma-4-E4B-it",
            is_live=True,
            blocked_reason="",
        )
        artifact = build_smoke_test_artifact(result)
        assert artifact["honest_verdict"] == "live_confirmed"

    def test_honest_verdict_blocked_error(self) -> None:
        """SCENARIO-BENCH-013: is_live=False, non-ci_skip mode → 'blocked_error'."""
        result = SmokeTestResult(
            inference_mode="blocked",
            n_questions=5,
            n_answered=0,
            elapsed_s=0.5,
            model_id="google/gemma-4-E4B-it",
            is_live=False,
            blocked_reason="CUDA not available",
        )
        artifact = build_smoke_test_artifact(result)
        assert artifact["honest_verdict"] == "blocked_error"

    def test_artifact_contains_all_result_fields(self) -> None:
        """SCENARIO-BENCH-012: artifact embeds all SmokeTestResult fields."""
        result = SmokeTestResult(
            inference_mode="ci_skip",
            n_questions=5,
            n_answered=0,
            elapsed_s=0.0,
            model_id="google/gemma-4-E4B-it",
            is_live=False,
            blocked_reason="CARNOT_FORCE_LIVE not set",
        )
        artifact = build_smoke_test_artifact(result)
        assert artifact["inference_mode"] == "ci_skip"
        assert artifact["n_questions"] == 5
        assert artifact["n_answered"] == 0
        assert artifact["elapsed_s"] == 0.0
        assert artifact["model_id"] == "google/gemma-4-E4B-it"
        assert artifact["is_live"] is False
        assert artifact["blocked_reason"] == "CARNOT_FORCE_LIVE not set"

    def test_artifact_is_json_serializable(self) -> None:
        """SCENARIO-BENCH-013: artifact is JSON-serializable (no non-serializable types)."""
        import json

        result = SmokeTestResult(
            inference_mode="live_gpu",
            n_questions=5,
            n_answered=4,
            elapsed_s=9.1,
            model_id="google/gemma-4-E4B-it",
            is_live=True,
            blocked_reason="",
        )
        artifact = build_smoke_test_artifact(result)
        # Should not raise
        json.dumps(artifact)
