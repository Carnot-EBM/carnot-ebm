"""Tests for carnot.pipeline.confidence_weighted_repair.

Confidence-weighted repair using two independent signals:
  1. Expression specificity: how precisely the violation text identifies an arithmetic error
  2. Energy variance: how consistently Ising samples agree the violation is real

Together these form a dual-signal gate that blocks false-positive repairs (the primary
failure mode from Exp 331 autopsy — VALID_INTERMEDIATE category) while preserving
true-positive repairs for clear arithmetic mistakes.

Spec: REQ-VERIFY-083, REQ-VERIFY-084, REQ-VERIFY-085,
      SCENARIO-VERIFY-109, SCENARIO-VERIFY-110, SCENARIO-VERIFY-111, SCENARIO-VERIFY-112
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.confidence_weighted_repair import (
    ConfidenceRepairResult,
    ConfidenceWeightedRepair,
    ViolationConfidence,
    compute_energy_variance_confidence,
    compute_expression_confidence,
)


# ---------------------------------------------------------------------------
# compute_expression_confidence — REQ-VERIFY-083
# ---------------------------------------------------------------------------


class TestComputeExpressionConfidence:
    """REQ-VERIFY-083: expression specificity → float in [0, 1]."""

    # SCENARIO-VERIFY-109
    def test_exact_arithmetic_no_spaces_high_confidence(self) -> None:
        """SCENARIO-VERIFY-109: '47+28=76' → score >= 0.90."""
        score = compute_expression_confidence("47+28=76")
        assert score >= 0.90, f"Expected >= 0.90, got {score}"

    def test_exact_arithmetic_with_spaces_high_confidence(self) -> None:
        """Explicit arithmetic with spaces: '47 + 28 = 76' → high confidence."""
        score = compute_expression_confidence("47 + 28 = 76")
        assert score >= 0.90

    def test_subtraction_exact_high_confidence(self) -> None:
        """Exact subtraction expression → high confidence."""
        score = compute_expression_confidence("100 - 37 = 64")
        assert score >= 0.90

    def test_multiplication_exact_high_confidence(self) -> None:
        """Exact multiplication expression → high confidence."""
        score = compute_expression_confidence("6 * 7 = 43")
        assert score >= 0.90

    def test_integer_mismatch_specific_values_medium_confidence(self) -> None:
        """Integer mismatch with specific values → score >= 0.75."""
        score = compute_expression_confidence("5 + 3 = 9 (correct: 8)")
        assert score >= 0.75

    # SCENARIO-VERIFY-110
    def test_approximately_language_low_confidence(self) -> None:
        """SCENARIO-VERIFY-110: 'approximately 150' → score <= 0.40."""
        score = compute_expression_confidence(
            "the intermediate result is approximately 150"
        )
        assert score <= 0.40, f"Expected <= 0.40, got {score}"

    def test_about_language_low_confidence(self) -> None:
        """'about 150' → low confidence."""
        score = compute_expression_confidence("the answer is about 150")
        assert score <= 0.40

    def test_roughly_language_low_confidence(self) -> None:
        """'roughly' language → low confidence."""
        score = compute_expression_confidence("roughly 3.33, rounded to 3")
        assert score <= 0.40

    def test_tilde_approximation_low_confidence(self) -> None:
        """'~' approximation → low confidence."""
        score = compute_expression_confidence("the value is ~150")
        assert score <= 0.40

    def test_intermediate_step_language_reduces_confidence(self) -> None:
        """Intermediate-step marker 'then' → reduces confidence vs bare arithmetic."""
        score_intermediate = compute_expression_confidence(
            "step result: 10 - 3 = 7 (intermediate — then add 4)"
        )
        score_exact = compute_expression_confidence("10 - 3 = 7")
        assert score_intermediate < score_exact

    def test_so_language_reduces_confidence(self) -> None:
        """'so' connector in violation → reduces confidence."""
        score = compute_expression_confidence("20 - 8 = 12, so the answer is 12")
        score_bare = compute_expression_confidence("20 - 8 = 12")
        assert score < score_bare

    def test_no_numeric_content_very_low_confidence(self) -> None:
        """No numeric content at all → score <= 0.15."""
        score = compute_expression_confidence("the answer is wrong")
        assert score <= 0.15

    def test_empty_string_very_low_confidence(self) -> None:
        """Empty string → score <= 0.15 (no violation specificity)."""
        score = compute_expression_confidence("")
        assert score <= 0.15

    def test_score_always_in_unit_interval(self) -> None:
        """Output is always in [0, 1] regardless of input."""
        test_inputs = [
            "",
            "47+28=76",
            "approximately 150",
            "step result: 10 - 3 = 7 (then add 4)",
            "some random text without math",
            "!@#$%^&*()",
            "1" * 1000,
        ]
        for text in test_inputs:
            score = compute_expression_confidence(text)
            assert 0.0 <= score <= 1.0, f"Out of range for input '{text[:30]}': {score}"

    def test_never_raises_on_any_input(self) -> None:
        """REQ-VERIFY-083: function never raises on any string input."""
        edge_cases = [
            "",
            " ",
            "\n\t",
            "NaN",
            "inf",
            "1/0",
            "a" * 10000,
        ]
        for text in edge_cases:
            compute_expression_confidence(text)  # must not raise


# ---------------------------------------------------------------------------
# compute_energy_variance_confidence — REQ-VERIFY-084
# ---------------------------------------------------------------------------


class TestComputeEnergyVarianceConfidence:
    """REQ-VERIFY-084: partition function variance → float in [0, 1]."""

    # SCENARIO-VERIFY-111
    def test_low_variance_gives_high_confidence(self) -> None:
        """SCENARIO-VERIFY-111: low-variance energies → confidence > 0.8."""
        energies = [2.0, 2.1, 1.9, 2.05, 1.95]
        score = compute_energy_variance_confidence(energies)
        assert score > 0.8, f"Expected > 0.8, got {score}"

    # SCENARIO-VERIFY-112
    def test_high_variance_gives_low_confidence(self) -> None:
        """SCENARIO-VERIFY-112: high-variance energies → confidence < 0.5."""
        energies = [0.1, 5.0, 0.2, 8.0, 0.05]
        score = compute_energy_variance_confidence(energies)
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_empty_list_returns_uninformative_prior(self) -> None:
        """Empty list → 0.5 (uninformative prior, no samples to measure)."""
        score = compute_energy_variance_confidence([])
        assert score == pytest.approx(0.5)

    def test_single_element_returns_uninformative_prior(self) -> None:
        """Single sample → 0.5 (cannot compute variance from one point)."""
        score = compute_energy_variance_confidence([3.0])
        assert score == pytest.approx(0.5)

    def test_all_zero_energies_handled_safely(self) -> None:
        """All-zero energies → safe result in [0, 1] (no division-by-zero)."""
        score = compute_energy_variance_confidence([0.0, 0.0, 0.0])
        assert 0.0 <= score <= 1.0

    def test_identical_nonzero_energies_give_perfect_confidence(self) -> None:
        """Identical energies → zero CV → confidence = 1.0 (perfect agreement)."""
        score = compute_energy_variance_confidence([5.0, 5.0, 5.0, 5.0])
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_score_always_in_unit_interval(self) -> None:
        """Output is always in [0, 1] for any valid energy list."""
        test_cases: list[list[float]] = [
            [],
            [0.0],
            [1.0, 2.0],
            [100.0, 100.0, 100.0],
            [0.001, 0.002, 0.001],
            [1e10, 1e-10, 1e10],
        ]
        for energies in test_cases:
            score = compute_energy_variance_confidence(energies)
            assert 0.0 <= score <= 1.0, f"Out of range for {energies}: {score}"

    def test_monotone_in_variance(self) -> None:
        """Higher variance → lower confidence (monotone relationship)."""
        low_var = [2.0, 2.01, 1.99, 2.005]  # near-zero variance
        high_var = [1.0, 4.0, 0.5, 5.0]     # high variance
        assert compute_energy_variance_confidence(low_var) > \
               compute_energy_variance_confidence(high_var)


# ---------------------------------------------------------------------------
# ViolationConfidence dataclass — REQ-VERIFY-085
# ---------------------------------------------------------------------------


class TestViolationConfidence:
    """REQ-VERIFY-085: ViolationConfidence dataclass contract."""

    def test_combined_confidence_is_geometric_mean(self) -> None:
        """combined_confidence = geometric mean of expression and energy signals."""
        vc = ViolationConfidence(
            expression_confidence=0.9,
            energy_variance_confidence=0.4,
            min_confidence=0.8,
        )
        expected = math.sqrt(0.9 * 0.4)
        assert vc.combined_confidence == pytest.approx(expected, abs=1e-9)

    def test_is_high_confidence_true_when_above_threshold(self) -> None:
        """is_high_confidence == True when combined >= min_confidence."""
        vc = ViolationConfidence(
            expression_confidence=1.0,
            energy_variance_confidence=1.0,
            min_confidence=0.8,
        )
        assert vc.is_high_confidence is True

    def test_is_high_confidence_false_when_below_threshold(self) -> None:
        """is_high_confidence == False when combined < min_confidence."""
        vc = ViolationConfidence(
            expression_confidence=0.3,
            energy_variance_confidence=0.3,
            min_confidence=0.8,
        )
        assert vc.is_high_confidence is False

    def test_is_high_confidence_at_exact_threshold(self) -> None:
        """is_high_confidence == True when combined == min_confidence exactly."""
        # We need combined = 0.8 exactly → both signals = 0.8 → geometric mean = 0.8
        vc = ViolationConfidence(
            expression_confidence=0.8,
            energy_variance_confidence=0.8,
            min_confidence=0.8,
        )
        assert vc.is_high_confidence is True

    def test_zero_expression_confidence_gives_zero_combined(self) -> None:
        """Geometric mean with zero operand → 0.0 combined_confidence."""
        vc = ViolationConfidence(
            expression_confidence=0.0,
            energy_variance_confidence=1.0,
            min_confidence=0.5,
        )
        assert vc.combined_confidence == pytest.approx(0.0)
        assert vc.is_high_confidence is False


# ---------------------------------------------------------------------------
# ConfidenceRepairResult dataclass — REQ-VERIFY-085
# ---------------------------------------------------------------------------


class TestConfidenceRepairResult:
    """REQ-VERIFY-085: ConfidenceRepairResult dataclass contract."""

    def test_all_fields_accessible(self) -> None:
        """All required fields exist and are accessible."""
        result = ConfidenceRepairResult(
            violations_found=3,
            violations_above_threshold=1,
            repair_triggered=True,
            improvement=1,
        )
        assert result.violations_found == 3
        assert result.violations_above_threshold == 1
        assert result.repair_triggered is True
        assert result.improvement == 1

    def test_no_violations_above_threshold(self) -> None:
        """violations_above_threshold=0 → repair_triggered=False."""
        result = ConfidenceRepairResult(
            violations_found=2,
            violations_above_threshold=0,
            repair_triggered=False,
            improvement=0,
        )
        assert result.repair_triggered is False
        assert result.improvement == 0


# ---------------------------------------------------------------------------
# ConfidenceWeightedRepair — REQ-VERIFY-085
# ---------------------------------------------------------------------------


class TestConfidenceWeightedRepair:
    """REQ-VERIFY-085: ConfidenceWeightedRepair orchestrator."""

    def _make_mock_pipeline(self, *, violations: list[str] | None = None) -> MagicMock:
        """Build a mock pipeline whose verify() returns canned violations."""
        pipeline = MagicMock()
        pipeline._extractor = MagicMock()

        if violations is None:
            violations = []

        # Each violation is a ConstraintResult-like object
        fake_constraints = []
        for v_text in violations:
            cr = MagicMock()
            cr.description = v_text
            cr.metadata = {"satisfied": False}
            fake_constraints.append(cr)

        pipeline._extractor.extract.return_value = fake_constraints
        return pipeline

    def test_no_violations_repair_not_triggered(self) -> None:
        """Zero violations → repair_triggered=False, improvement=0."""
        pipeline = self._make_mock_pipeline(violations=[])
        cwr = ConfidenceWeightedRepair(pipeline=pipeline, n_samples=3, min_confidence=0.8)
        result = cwr.repair("What is 2+2?", "4", domain=None)
        assert result.repair_triggered is False
        assert result.violations_found == 0
        assert result.improvement == 0

    def test_low_confidence_violation_not_triggered(self) -> None:
        """Approximate language violation → combined_confidence too low → not triggered."""
        pipeline = self._make_mock_pipeline(
            violations=["the intermediate result is approximately 150"]
        )
        cwr = ConfidenceWeightedRepair(pipeline=pipeline, n_samples=3, min_confidence=0.8)

        # Patch energy variance to return high-variance (low confidence) energies
        with patch.object(
            cwr, "_sample_energies", return_value=[0.1, 8.0, 0.05, 7.0, 0.2]
        ):
            result = cwr.repair("What is 150?", "approximately 150", domain=None)

        assert result.violations_found == 1
        assert result.violations_above_threshold == 0
        assert result.repair_triggered is False

    def test_high_confidence_violation_triggers_repair(self) -> None:
        """Exact arithmetic violation with low variance → combined_confidence high → triggered."""
        pipeline = self._make_mock_pipeline(violations=["47+28=76"])

        # Mock the underlying repair call
        repair_result = MagicMock()
        repair_result.repaired = True
        pipeline.verify_and_repair_confident.return_value = repair_result

        cwr = ConfidenceWeightedRepair(pipeline=pipeline, n_samples=3, min_confidence=0.8)

        # Patch energy variance to return low-variance (high confidence) energies
        with patch.object(
            cwr, "_sample_energies", return_value=[2.0, 2.1, 1.9]
        ):
            result = cwr.repair("What is 47+28?", "76", domain=None)

        assert result.violations_found == 1
        assert result.violations_above_threshold == 1
        assert result.repair_triggered is True

    def test_improvement_0_when_repair_fails(self) -> None:
        """repair_triggered but underlying repair fails → improvement=0."""
        pipeline = self._make_mock_pipeline(violations=["5+3=9"])
        repair_result = MagicMock()
        repair_result.repaired = False
        pipeline.verify_and_repair_confident.return_value = repair_result

        cwr = ConfidenceWeightedRepair(pipeline=pipeline, n_samples=3, min_confidence=0.8)

        with patch.object(cwr, "_sample_energies", return_value=[2.0, 2.0, 2.0]):
            result = cwr.repair("What is 5+3?", "9", domain=None)

        assert result.repair_triggered is True
        assert result.improvement == 0

    def test_multiple_violations_counts_above_threshold_correctly(self) -> None:
        """Multiple violations: only the exact arithmetic one exceeds threshold."""
        pipeline = self._make_mock_pipeline(
            violations=["47+28=76", "approximately 150"]
        )
        repair_result = MagicMock()
        repair_result.repaired = True
        pipeline.verify_and_repair_confident.return_value = repair_result

        cwr = ConfidenceWeightedRepair(pipeline=pipeline, n_samples=3, min_confidence=0.8)

        # Return low-variance for both calls (expression_confidence is what differentiates)
        with patch.object(cwr, "_sample_energies", return_value=[2.0, 2.0, 2.0]):
            result = cwr.repair("question", "response", domain=None)

        assert result.violations_found == 2
        # Only the exact arithmetic violation should exceed threshold
        assert result.violations_above_threshold >= 1

    def test_n_samples_default_is_five(self) -> None:
        """Default n_samples=5 and min_confidence=0.8."""
        pipeline = MagicMock()
        pipeline._extractor = MagicMock()
        pipeline._extractor.extract.return_value = []
        cwr = ConfidenceWeightedRepair(pipeline=pipeline)
        assert cwr.n_samples == 5
        assert cwr.min_confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Benchmark: correct responses avoid unnecessary repair — REQ-VERIFY-085
# ---------------------------------------------------------------------------


class TestBenchmarkFalsePositiveAvoidance:
    """Spot benchmark: confidence gate prevents false-positive repairs on correct responses."""

    def test_correct_intermediate_step_avoids_repair(self) -> None:
        """Intermediate-step expression (e.g. Exp 331 case) → not repaired."""
        # Simulate a correct response that was flagged as a FP by the binary extractor
        violation_text = "step result: 10 - 3 = 7 (intermediate — then add 4)"
        score = compute_expression_confidence(violation_text)
        # Combined with even perfect energy agreement, the expression score should
        # be low enough to block repair (< 0.8)
        # Geometric mean of score and 1.0 is just score (best case energy agreement)
        combined = math.sqrt(score * 1.0)
        assert combined < 0.8, (
            f"FP from Exp 331 VALID_INTERMEDIATE should be blocked "
            f"(combined={combined:.3f} must be < 0.8)"
        )

    def test_exact_wrong_answer_passes_gate(self) -> None:
        """Clear arithmetic error (47+28=76) → passes confidence gate."""
        violation_text = "47 + 28 = 76"
        score = compute_expression_confidence(violation_text)
        # Even with moderate energy variance confidence (0.85), should exceed 0.8
        combined = math.sqrt(score * 0.85)
        assert combined >= 0.8, (
            f"True positive (47+28=76) should pass gate (combined={combined:.3f})"
        )
