"""Tests for carnot.extraction.confidence_filter.

Coverage target: 100% of confidence_filter.py.

Spec: REQ-EXTRACT-031, REQ-EXTRACT-032,
      SCENARIO-EXTRACT-058, SCENARIO-EXTRACT-059, SCENARIO-EXTRACT-060
"""

from __future__ import annotations

import pytest

from carnot.extraction.confidence_filter import (
    ConfidenceWeightedExtractor,
    ViolationConfidence,
    _score_equation_error,
    score_violation,
)


# ---------------------------------------------------------------------------
# ViolationConfidence
# ---------------------------------------------------------------------------


class TestViolationConfidence:
    """SCENARIO-EXTRACT-058: ViolationConfidence stores text, score, type, is_definitive."""

    def test_is_definitive_true_at_80(self):
        """Scores >= 0.80 set is_definitive=True."""
        vc = ViolationConfidence("47 + 28 = 76", 0.95, "equation_error")
        assert vc.is_definitive is True

    def test_is_definitive_true_at_exactly_80(self):
        vc = ViolationConfidence("step text", 0.80, "default")
        assert vc.is_definitive is True

    def test_is_definitive_false_below_80(self):
        """Scores < 0.80 set is_definitive=False."""
        vc = ViolationConfidence("approximately 75", 0.20, "approximate")
        assert vc.is_definitive is False

    def test_fields_stored(self):
        vc = ViolationConfidence("some text", 0.60, "default")
        assert vc.violation_text == "some text"
        assert vc.confidence_score == 0.60
        assert vc.violation_type == "default"


# ---------------------------------------------------------------------------
# _score_equation_error
# ---------------------------------------------------------------------------


class TestScoreEquationError:
    """Unit tests for the equation-error sub-scorer."""

    def test_clear_addition_error(self):
        # 47 + 28 = 76 is wrong (should be 75), error > 5%
        score = _score_equation_error("we compute 47 + 28 = 76")
        assert score == 0.95

    def test_correct_equation_returns_none(self):
        # 47 + 28 = 75 is correct; should not flag
        score = _score_equation_error("we compute 47 + 28 = 75")
        assert score is None

    def test_no_equation_returns_none(self):
        score = _score_equation_error("the answer is approximately 75")
        assert score is None

    def test_subtraction_error(self):
        # 100 - 30 = 65, should be 70 — >5% error
        score = _score_equation_error("100 - 30 = 65")
        assert score == 0.95

    def test_multiplication_error(self):
        # 7 * 8 = 55, should be 56
        score = _score_equation_error("7 * 8 = 55")
        assert score == 0.95

    def test_division_error(self):
        # 100 / 4 = 24, should be 25 — any integer mismatch is flagged
        score = _score_equation_error("100 / 4 = 24")
        assert score == 0.95

    def test_division_large_error(self):
        # 100 / 4 = 20, should be 25 (20% off) — clearly flagged
        score = _score_equation_error("100 / 4 = 20")
        assert score == 0.95

    def test_division_by_zero_returns_none(self):
        score = _score_equation_error("5 / 0 = 0")
        assert score is None

    def test_unknown_operator_returns_none(self):
        # Operator not in +/-/*/
        score = _score_equation_error("5 % 3 = 2")
        assert score is None

    def test_negative_numbers(self):
        # -10 + 5 = -6, should be -5 (20% off)
        score = _score_equation_error("-10 + 5 = -6")
        assert score == 0.95


# ---------------------------------------------------------------------------
# score_violation
# ---------------------------------------------------------------------------


class TestScoreViolation:
    """SCENARIO-EXTRACT-058: score_violation applies heuristics in precedence order."""

    def test_equation_error_highest_priority(self):
        # Contains both an equation error AND 'approximately'
        # equation_error should win (highest priority)
        text = "approximately, 47 + 28 = 76"
        score, vtype = score_violation(text)
        assert vtype == "equation_error"
        assert score == 0.95

    def test_final_answer_second_priority(self):
        # No equation, but has 'the answer is'
        text = "therefore the answer is 75"
        score, vtype = score_violation(text)
        assert vtype == "final_answer_error"
        assert score == 0.90

    def test_approximate_lower_than_final_answer(self):
        # 'approximately' with no equation and no final-answer marker
        text = "the value is approximately 75"
        score, vtype = score_violation(text)
        assert vtype == "approximate"
        assert score == 0.20

    def test_intermediate_step(self):
        text = "step 1 we compute the subtotal"
        score, vtype = score_violation(text)
        assert vtype == "intermediate"
        assert score == 0.40

    def test_default_no_heuristic_matches(self):
        text = "this is a generic violation with no matching patterns"
        score, vtype = score_violation(text)
        assert vtype == "default"
        assert score == 0.60

    def test_roughly_triggers_approximate(self):
        text = "the value is roughly 50"
        score, vtype = score_violation(text)
        assert vtype == "approximate"

    def test_about_triggers_approximate(self):
        text = "that gives about 30 units"
        score, vtype = score_violation(text)
        assert vtype == "approximate"

    def test_first_we_triggers_intermediate(self):
        text = "first we add the two values"
        score, vtype = score_violation(text)
        assert vtype == "intermediate"

    def test_thus_triggers_final_answer(self):
        text = "thus, the total is 75"
        score, vtype = score_violation(text)
        assert vtype == "final_answer_error"


# ---------------------------------------------------------------------------
# ConfidenceWeightedExtractor.extract
# ---------------------------------------------------------------------------


class _AlwaysFlagsExtractor:
    """Test double: always returns a fixed list of violation objects."""

    def __init__(self, violations: list[str]) -> None:
        self._violations = violations

    def detect_violations(self, text: str) -> list[str]:
        return list(self._violations)


class _NeverFlagsExtractor:
    """Test double: always returns empty list (no violations)."""

    def detect_violations(self, text: str) -> list[str]:
        return []


class TestConfidenceWeightedExtractorExtract:
    """SCENARIO-EXTRACT-058: extract() scores each violation from base extractor."""

    def test_empty_when_base_finds_nothing(self):
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor(), 0.7)
        result = ext.extract("some text")
        assert result == []

    def test_returns_one_violation_confidence_per_raw(self):
        base = _AlwaysFlagsExtractor(["47 + 28 = 76", "roughly 50"])
        ext = ConfidenceWeightedExtractor(base, 0.7)
        result = ext.extract("unused input")
        assert len(result) == 2
        assert all(isinstance(v, ViolationConfidence) for v in result)

    def test_scores_violation_text_correctly(self):
        base = _AlwaysFlagsExtractor(["47 + 28 = 76"])
        ext = ConfidenceWeightedExtractor(base, 0.7)
        result = ext.extract("text")
        assert result[0].violation_type == "equation_error"
        assert result[0].confidence_score == 0.95

    def test_violation_text_is_str_of_raw(self):
        # The raw violation is an integer; str(42) == '42'
        base = _AlwaysFlagsExtractor([42])  # type: ignore[arg-type]
        ext = ConfidenceWeightedExtractor(base, 0.7)
        result = ext.extract("text")
        assert result[0].violation_text == "42"

    def test_default_threshold_is_07(self):
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor())
        assert ext.confidence_threshold == 0.7


# ---------------------------------------------------------------------------
# ConfidenceWeightedExtractor.above_threshold
# ---------------------------------------------------------------------------


class TestAboveThreshold:
    """SCENARIO-EXTRACT-059: above_threshold filters by confidence_threshold."""

    def _make_violations(self) -> list[ViolationConfidence]:
        return [
            ViolationConfidence("47 + 28 = 76", 0.95, "equation_error"),
            ViolationConfidence("approximately 50", 0.20, "approximate"),
            ViolationConfidence("generic", 0.60, "default"),
        ]

    def test_threshold_07_keeps_only_high_confidence(self):
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor(), 0.7)
        violations = self._make_violations()
        result = ext.above_threshold(violations)
        assert len(result) == 1
        assert result[0].violation_type == "equation_error"

    def test_threshold_05_keeps_high_and_default(self):
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor(), 0.5)
        violations = self._make_violations()
        result = ext.above_threshold(violations)
        assert len(result) == 2

    def test_threshold_10_keeps_nothing(self):
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor(), 1.0)
        violations = self._make_violations()
        result = ext.above_threshold(violations)
        assert result == []

    def test_empty_input_returns_empty(self):
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor(), 0.7)
        assert ext.above_threshold([]) == []

    def test_threshold_exactly_at_score_passes(self):
        # confidence_score == threshold should be included (>=)
        vc = ViolationConfidence("text", 0.70, "default")
        ext = ConfidenceWeightedExtractor(_NeverFlagsExtractor(), 0.70)
        result = ext.above_threshold([vc])
        assert result == [vc]


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-060: Full extract + above_threshold pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """SCENARIO-EXTRACT-060: End-to-end: extract then threshold on real extractor output."""

    def test_high_confidence_violation_passes_gate(self):
        base = _AlwaysFlagsExtractor(["47 + 28 = 76"])
        ext = ConfidenceWeightedExtractor(base, 0.7)
        all_violations = ext.extract("47 plus 28 gives 76")
        above = ext.above_threshold(all_violations)
        assert len(above) == 1
        assert above[0].is_definitive is True

    def test_approximate_violation_blocked_at_07_threshold(self):
        base = _AlwaysFlagsExtractor(["approximately 50"])
        ext = ConfidenceWeightedExtractor(base, 0.7)
        all_violations = ext.extract("the value is approximately 50")
        above = ext.above_threshold(all_violations)
        assert above == []

    def test_intermediate_violation_blocked_at_07_threshold(self):
        base = _AlwaysFlagsExtractor(["step 1 we compute something"])
        ext = ConfidenceWeightedExtractor(base, 0.7)
        all_violations = ext.extract("text")
        above = ext.above_threshold(all_violations)
        assert above == []

    def test_vprm_integration_no_violations(self):
        """VPRMArithmeticVerifier returns no violations on plain prose; wrap should also return empty."""
        from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier

        base = VPRMArithmeticVerifier()
        ext = ConfidenceWeightedExtractor(base, 0.7)
        all_violations = ext.extract("The train travels 60 miles per hour.")
        assert all_violations == []
        assert ext.above_threshold(all_violations) == []
