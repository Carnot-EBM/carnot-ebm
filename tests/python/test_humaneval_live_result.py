"""Tests for CodeVerificationResult and HumanEvalLiveResult data types (Exp 469).

These tests cover the improvement/regression flag logic for per-problem results and
the aggregate HumanEvalLiveResult signed_improvement and is_positive properties.

Spec: REQ-BENCH-023, REQ-BENCH-024,
      SCENARIO-BENCH-042, SCENARIO-BENCH-043
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from scripts/ without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from experiment_469_humaneval_live_vericot import (
    CodeVerificationResult,
    HumanEvalLiveResult,
)


# ---------------------------------------------------------------------------
# CodeVerificationResult tests (SCENARIO-BENCH-042)
# ---------------------------------------------------------------------------


class TestCodeVerificationResultImprovement:
    """Test improvement=True when pipeline fixes a failing baseline."""

    def test_improvement_when_pipeline_fixes_failure(self) -> None:
        # Pipeline fixed a problem that baseline failed — this is an improvement.
        r = CodeVerificationResult(
            problem_id="HumanEval/0",
            baseline_passed=False,
            pipeline_passed=True,
            violations_detected=2,
            repairs_applied=1,
            inference_mode="live_gpu",
        )
        assert r.improvement is True
        assert r.regression is False

    def test_no_improvement_when_both_pass(self) -> None:
        # Both baseline and pipeline pass — no change, not an improvement.
        r = CodeVerificationResult(
            problem_id="HumanEval/1",
            baseline_passed=True,
            pipeline_passed=True,
            violations_detected=0,
            repairs_applied=0,
            inference_mode="live_gpu",
        )
        assert r.improvement is False
        assert r.regression is False

    def test_no_improvement_when_both_fail(self) -> None:
        # Both baseline and pipeline fail — no change in either direction.
        r = CodeVerificationResult(
            problem_id="HumanEval/2",
            baseline_passed=False,
            pipeline_passed=False,
            violations_detected=1,
            repairs_applied=1,
            inference_mode="live_gpu",
        )
        assert r.improvement is False
        assert r.regression is False


class TestCodeVerificationResultRegression:
    """Test regression=True when pipeline breaks a passing baseline."""

    def test_regression_when_pipeline_breaks_passing(self) -> None:
        # Baseline passed but pipeline failed — this is a regression.
        r = CodeVerificationResult(
            problem_id="HumanEval/3",
            baseline_passed=True,
            pipeline_passed=False,
            violations_detected=0,
            repairs_applied=1,
            inference_mode="live_gpu",
        )
        assert r.regression is True
        assert r.improvement is False

    def test_no_regression_when_both_pass(self) -> None:
        # Both pass — neither regression nor improvement.
        r = CodeVerificationResult(
            problem_id="HumanEval/4",
            baseline_passed=True,
            pipeline_passed=True,
            violations_detected=0,
            repairs_applied=0,
            inference_mode="live_gpu",
        )
        assert r.regression is False

    def test_no_regression_when_both_fail(self) -> None:
        # Both fail — the pipeline didn't make things worse (they were already wrong).
        r = CodeVerificationResult(
            problem_id="HumanEval/5",
            baseline_passed=False,
            pipeline_passed=False,
            violations_detected=3,
            repairs_applied=0,
            inference_mode="live_gpu",
        )
        assert r.regression is False


# ---------------------------------------------------------------------------
# HumanEvalLiveResult tests
# ---------------------------------------------------------------------------


class TestHumanEvalLiveResultSignedImprovement:
    """Test signed_improvement and is_positive on aggregate results."""

    def test_positive_improvement(self) -> None:
        # pipeline_pass_at_1 > baseline_pass_at_1 → is_positive True
        result = HumanEvalLiveResult(
            n_problems=50,
            baseline_pass_at_1=0.60,
            pipeline_pass_at_1=0.66,
            inference_mode="live_gpu",
        )
        assert result.signed_improvement > 0.0
        assert result.is_positive is True

    def test_zero_improvement(self) -> None:
        # Equal rates → is_positive False
        result = HumanEvalLiveResult(
            n_problems=50,
            baseline_pass_at_1=0.60,
            pipeline_pass_at_1=0.60,
            inference_mode="live_gpu",
        )
        assert result.signed_improvement == 0.0
        assert result.is_positive is False

    def test_negative_improvement(self) -> None:
        # pipeline worse than baseline → signed negative, is_positive False
        result = HumanEvalLiveResult(
            n_problems=50,
            baseline_pass_at_1=0.60,
            pipeline_pass_at_1=0.54,
            inference_mode="live_gpu",
        )
        assert result.signed_improvement < 0.0
        assert result.is_positive is False

    def test_signed_improvement_computation(self) -> None:
        # Verify arithmetic: 0.70 − 0.60 = 0.10
        result = HumanEvalLiveResult(
            n_problems=10,
            baseline_pass_at_1=0.60,
            pipeline_pass_at_1=0.70,
            inference_mode="live_gpu",
        )
        assert abs(result.signed_improvement - 0.10) < 1e-9

    def test_blocked_mode_not_positive(self) -> None:
        # Even if numbers look good, blocked inference_mode must NOT be positive.
        result = HumanEvalLiveResult(
            n_problems=0,
            baseline_pass_at_1=0.0,
            pipeline_pass_at_1=0.0,
            inference_mode="blocked",
        )
        assert result.is_positive is False
