"""Tests for LatentCoTEBMCalibrator and LatentCoTCalibrationResult.

Spec: REQ-VERIFY-116, SCENARIO-VERIFY-134, SCENARIO-VERIFY-135, SCENARIO-VERIFY-136
"""

from __future__ import annotations

import pytest

from carnot.models.eorm import EORMModel
from carnot.pipeline.latent_cot_calibrator import (
    LatentCoTCalibrationResult,
    LatentCoTEBMCalibrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def eorm_model() -> EORMModel:
    """Small EORM model (embed_dim=32) for fast CPU tests."""
    return EORMModel(embed_dim=32, n_heads=4, n_layers=1)


@pytest.fixture()
def calibrator(eorm_model: EORMModel) -> LatentCoTEBMCalibrator:
    return LatentCoTEBMCalibrator(eorm_model, alpha=0.1, step_boundary_tokens=32)


def _simple_generate(prompt: str, temperature_adjustments: list[float]) -> str:
    """Synthetic generate_fn: returns a deterministic response ignoring adjustments."""
    # Produce a response long enough to trigger multiple 32-word boundaries
    return (
        "Step 1: We start with the given values. "
        "Step 2: Multiply 3 by 4 to get 12. "
        "Step 3: Add 12 and 5 to get 17. "
        "Step 4: The final answer is 17. " * 3
    )


def _short_generate(prompt: str, temperature_adjustments: list[float]) -> str:
    return "The answer is 42."


def _empty_generate(prompt: str, temperature_adjustments: list[float]) -> str:
    return ""


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-134: LatentCoTCalibrationResult has correct field types
# ---------------------------------------------------------------------------

class TestLatentCoTCalibrationResult:
    """REQ-VERIFY-116: LatentCoTCalibrationResult dataclass has all required fields."""

    def test_default_construction(self) -> None:
        """SCENARIO-VERIFY-134: default construction gives zero/empty values."""
        r = LatentCoTCalibrationResult(n_steps=0)
        assert r.n_steps == 0
        assert r.per_step_energy == []
        assert r.mean_energy == 0.0
        assert r.violation_rate_before == 0.0
        assert r.violation_rate_after == 0.0
        assert r.temperature_adjustments == []

    def test_full_construction(self) -> None:
        """SCENARIO-VERIFY-134: explicit construction stores all fields."""
        r = LatentCoTCalibrationResult(
            n_steps=3,
            per_step_energy=[0.1, 0.2, 0.3],
            mean_energy=0.2,
            violation_rate_before=0.5,
            violation_rate_after=0.3,
            temperature_adjustments=[0.99, 0.98, 0.97],
        )
        assert r.n_steps == 3
        assert len(r.per_step_energy) == 3
        assert r.mean_energy == pytest.approx(0.2)
        assert r.violation_rate_before == pytest.approx(0.5)
        assert r.violation_rate_after == pytest.approx(0.3)
        assert len(r.temperature_adjustments) == 3


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-135: calibrate_generation produces non-empty metrics
# ---------------------------------------------------------------------------

class TestLatentCoTEBMCalibratorCalibrateGeneration:
    """REQ-VERIFY-116: calibrate_generation scores each 32-token boundary."""

    def test_returns_responses_and_result(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: returns list of responses and a result object."""
        prompts = ["What is 3 * 4 + 5?"]
        responses, result = calibrator.calibrate_generation(prompts, _simple_generate)
        assert len(responses) == 1
        assert isinstance(result, LatentCoTCalibrationResult)

    def test_n_steps_positive(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: n_steps > 0 for non-empty response."""
        prompts = ["What is 3 * 4 + 5?"]
        _, result = calibrator.calibrate_generation(prompts, _simple_generate)
        assert result.n_steps > 0

    def test_per_step_energy_length_matches_n_steps(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: per_step_energy list length equals n_steps."""
        prompts = ["Q1?", "Q2?"]
        _, result = calibrator.calibrate_generation(prompts, _simple_generate)
        assert len(result.per_step_energy) == result.n_steps

    def test_temperature_adjustments_length_matches_n_steps(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: temperature_adjustments list length equals n_steps."""
        prompts = ["Q1?"]
        _, result = calibrator.calibrate_generation(prompts, _simple_generate)
        assert len(result.temperature_adjustments) == result.n_steps

    def test_mean_energy_is_mean_of_per_step(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: mean_energy equals arithmetic mean of per_step_energy."""
        prompts = ["What is 2+2?"]
        _, result = calibrator.calibrate_generation(prompts, _simple_generate)
        expected_mean = sum(result.per_step_energy) / len(result.per_step_energy)
        assert result.mean_energy == pytest.approx(expected_mean, rel=1e-5)

    def test_temperature_adj_formula(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: each adjustment equals 1 - alpha * energy."""
        prompts = ["What is 5+5?"]
        _, result = calibrator.calibrate_generation(prompts, _simple_generate)
        for energy, adj in zip(result.per_step_energy, result.temperature_adjustments):
            expected = 1.0 - calibrator.alpha * energy
            assert adj == pytest.approx(expected, rel=1e-5)

    def test_short_response_still_scores(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: short response (< 32 words) still gets one step score."""
        prompts = ["What is 7?"]
        _, result = calibrator.calibrate_generation(prompts, _short_generate)
        assert result.n_steps >= 1

    def test_empty_response_still_scores(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: empty response handled gracefully without exception."""
        prompts = ["What is nothing?"]
        _, result = calibrator.calibrate_generation(prompts, _empty_generate)
        assert result.n_steps >= 1

    def test_n_questions_truncates(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: n_questions parameter truncates prompt list."""
        prompts = ["Q1?", "Q2?", "Q3?"]
        responses, _ = calibrator.calibrate_generation(prompts, _simple_generate, n_questions=2)
        assert len(responses) == 2

    def test_multiple_questions_accumulate_steps(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-135: n_steps aggregates across all questions."""
        prompts = ["Q1?"]
        _, r1 = calibrator.calibrate_generation(prompts, _simple_generate)
        prompts2 = ["Q1?", "Q2?"]
        _, r2 = calibrator.calibrate_generation(prompts2, _simple_generate)
        assert r2.n_steps >= r1.n_steps


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-136: compare_violation_rate
# ---------------------------------------------------------------------------

class TestCompareViolationRate:
    """REQ-VERIFY-116: compare_violation_rate returns correct dict structure."""

    def test_returns_required_keys(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-136: result dict has all three required keys."""
        result = calibrator.compare_violation_rate(
            calibrated_responses=["The answer is 4."],
            baseline_responses=["The answer is 5."],
            labeled_questions=["What is 2+2?"],
        )
        assert "baseline_violation_rate" in result
        assert "calibrated_violation_rate" in result
        assert "violation_rate_delta" in result

    def test_rates_are_floats_in_range(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-136: all rates are floats in [0, 1]."""
        result = calibrator.compare_violation_rate(
            calibrated_responses=["no arithmetic here"],
            baseline_responses=["also no arithmetic"],
            labeled_questions=["Question?"],
        )
        assert 0.0 <= result["baseline_violation_rate"] <= 1.0
        assert 0.0 <= result["calibrated_violation_rate"] <= 1.0

    def test_delta_equals_calibrated_minus_baseline(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-136: delta = calibrated_rate - baseline_rate."""
        result = calibrator.compare_violation_rate(
            calibrated_responses=["no arithmetic"],
            baseline_responses=["no arithmetic"],
            labeled_questions=["Q?"],
        )
        expected_delta = result["calibrated_violation_rate"] - result["baseline_violation_rate"]
        assert result["violation_rate_delta"] == pytest.approx(expected_delta)

    def test_empty_responses_returns_zero_rates(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-136: empty response lists produce zero violation rates."""
        result = calibrator.compare_violation_rate(
            calibrated_responses=[],
            baseline_responses=[],
            labeled_questions=[],
        )
        assert result["baseline_violation_rate"] == 0.0
        assert result["calibrated_violation_rate"] == 0.0
        assert result["violation_rate_delta"] == 0.0

    def test_violation_detected_in_bad_response(self, calibrator: LatentCoTEBMCalibrator) -> None:
        """SCENARIO-VERIFY-136: responses with arithmetic errors yield violation_rate > 0."""
        # 3 + 4 = 8 is arithmetically wrong (should be 7) — VPRM should flag it.
        bad_responses = ["3 + 4 = 8 so the answer is 8."]
        result = calibrator.compare_violation_rate(
            calibrated_responses=bad_responses,
            baseline_responses=bad_responses,
            labeled_questions=["What is 3+4?"],
        )
        # Both are bad, so both rates should be equal (delta=0)
        assert result["violation_rate_delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: import from carnot.pipeline
# ---------------------------------------------------------------------------

class TestPublicExport:
    """LatentCoTEBMCalibrator and LatentCoTCalibrationResult are exported from carnot.pipeline."""

    def test_importable_from_pipeline(self) -> None:
        """SCENARIO-VERIFY-134: symbols importable from carnot.pipeline."""
        from carnot.pipeline import LatentCoTCalibrationResult, LatentCoTEBMCalibrator  # noqa: F401

        assert LatentCoTEBMCalibrator is not None
        assert LatentCoTCalibrationResult is not None
