"""Tests for Tier0uVerifier — self-consistency logical-inconsistency detector.

Spec: REQ-TIER0-008, SCENARIO-TIER0-008
"""

import pytest
from carnot.verify.tier0u_logical_consistency import (
    Tier0uVerifier,
    _count_self_corrections,
    _count_contradictions_near_numbers,
    _final_answer_mismatch,
    _count_numerical_claims,
)


class TestTier0uVerifierBasic:
    """REQ-TIER0-008: score() returns float in [0, 1] for any string input."""

    def setup_method(self):
        self.v = Tier0uVerifier()

    def test_returns_float_in_range_consistent(self):
        # A clean arithmetic response with no self-corrections.
        resp = "We have 5 apples. We add 3 more. 5 + 3 = 8. The answer is 8."
        s = self.v.score(resp)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_returns_float_in_range_inconsistent(self):
        # A response with explicit self-correction.
        resp = "5 + 3 = 9. Wait, I made an error. Actually 5 + 3 = 8. The answer is 8."
        s = self.v.score(resp)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_empty_string_returns_zero(self):
        # SCENARIO-TIER0-008: empty/whitespace input is consistent by definition.
        assert self.v.score("") == 0.0
        assert self.v.score("   ") == 0.0

    def test_inconsistent_scores_higher_than_consistent(self):
        # SCENARIO-TIER0-008: inconsistency signal raises score above baseline.
        clean = "We have 5 apples and add 3 to get 8. The answer is 8."
        messy = "We have 5 apples. Wait, actually I was wrong. The answer is 8."
        assert self.v.score(messy) > self.v.score(clean)

    def test_final_answer_mismatch_raises_score(self):
        # A response where the final answer contradicts the last computed step.
        # Last intermediate number before final sentence: 15; stated answer: 10.
        resp = "3 * 5 = 15. The answer is 10."
        s = self.v.score(resp)
        # Should be penalised relative to a matching response.
        matching = "3 * 5 = 15. The answer is 15."
        assert s > self.v.score(matching)

    def test_no_numbers_returns_low_score(self):
        # Text with no numeric content has no numerical claims; denominator = 1.
        resp = "Paris is the capital of France and it is a beautiful city."
        s = self.v.score(resp)
        # No numbers, no contradiction signals → score should be very low
        assert s < 0.3


class TestSelfCorrectionDetector:
    """Unit tests for _count_self_corrections helper."""

    def test_actual(self):
        assert _count_self_corrections("5 + 3 = 9. Actually it is 8.") >= 1

    def test_wait(self):
        assert _count_self_corrections("Wait, let me recalculate.") >= 1

    def test_i_made_an_error(self):
        assert _count_self_corrections("I made an error earlier.") >= 1

    def test_no_correction(self):
        assert _count_self_corrections("5 + 3 = 8. The answer is 8.") == 0

    def test_multiple_corrections(self):
        # Two correction phrases in one response.
        text = "Actually wait, I was wrong about this."
        assert _count_self_corrections(text) >= 2

    def test_case_insensitive(self):
        assert _count_self_corrections("ACTUALLY the answer is different.") >= 1


class TestContradictionNearNumbers:
    """Unit tests for _count_contradictions_near_numbers helper."""

    def test_but_near_number(self):
        assert _count_contradictions_near_numbers("It is 10, but 5 is correct.") >= 1

    def test_however_near_number(self):
        assert _count_contradictions_near_numbers("The rate is 60 however 70 was used.") >= 1

    def test_no_number_nearby(self):
        # 'but' far from any number should not trigger
        text = "The sky is blue, but the grass is green."
        assert _count_contradictions_near_numbers(text) == 0

    def test_no_contradiction_markers(self):
        assert _count_contradictions_near_numbers("5 + 3 = 8.") == 0


class TestFinalAnswerMismatch:
    """Unit tests for _final_answer_mismatch helper."""

    def test_mismatch_detected(self):
        # Last body number (15) ≠ stated answer (10)
        assert _final_answer_mismatch("3 * 5 = 15. The answer is 10.") is True

    def test_no_mismatch(self):
        assert _final_answer_mismatch("3 * 5 = 15. The answer is 15.") is False

    def test_no_final_answer_sentence(self):
        # No 'the answer is' pattern → no mismatch by this heuristic
        assert _final_answer_mismatch("3 * 5 = 15.") is False

    def test_therefore_pattern(self):
        # 'therefore' is also a final-answer signal
        assert _final_answer_mismatch("2 + 2 = 4. Therefore, 5.") is True
        assert _final_answer_mismatch("2 + 2 = 4. Therefore, 4.") is False


class TestNumericalClaimsCount:
    """Unit tests for _count_numerical_claims helper."""

    def test_counts_sentences_with_numbers(self):
        text = "We have 5 apples. The weather is nice. We add 3 to get 8."
        # Two sentences contain numbers (first and third)
        assert _count_numerical_claims(text) == 2

    def test_no_numbers(self):
        assert _count_numerical_claims("Paris is a city in France.") == 0

    def test_empty(self):
        assert _count_numerical_claims("") == 0
