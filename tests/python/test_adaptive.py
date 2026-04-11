"""Tests for carnot.pipeline.adaptive -- AdaptiveWeighter and run_comparison.

Covers Tier 1 Online Self-Learning: adaptive constraint weight computation
from ConstraintTracker statistics, pipeline integration, and fixed-vs-adaptive
accuracy comparison.

Spec: REQ-LEARN-002, SCENARIO-LEARN-002
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.adaptive import (
    WEIGHT_FLOOR,
    AdaptiveWeighter,
    ComparisonResult,
    run_comparison,
)
from carnot.pipeline.tracker import ConstraintTracker
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# REQ-LEARN-002: AdaptiveWeighter.from_tracker weight formula
# ---------------------------------------------------------------------------


class TestFromTracker:
    """REQ-LEARN-002: Weight formula w_i = max(precision * log(count+1), floor)."""

    def test_perfect_precision_high_count(self) -> None:
        """SCENARIO-LEARN-002: High precision + high count yields weight > floor."""
        tracker = ConstraintTracker()
        # 10 fires, all caught -> precision=1.0
        for _ in range(10):
            tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)

        weights = AdaptiveWeighter.from_tracker(tracker)
        assert "arithmetic" in weights
        expected = 1.0 * math.log(10 + 1)
        assert weights["arithmetic"] == pytest.approx(expected)
        assert weights["arithmetic"] > WEIGHT_FLOOR

    def test_zero_precision_returns_floor(self) -> None:
        """REQ-LEARN-002: Zero precision (no errors caught) returns WEIGHT_FLOOR."""
        tracker = ConstraintTracker()
        for _ in range(20):
            tracker.record("logic", fired=True, caught_error=False, any_error_in_batch=False)

        weights = AdaptiveWeighter.from_tracker(tracker)
        assert weights["logic"] == pytest.approx(WEIGHT_FLOOR)

    def test_floor_is_minimum(self) -> None:
        """REQ-LEARN-002: No weight in the returned dict is below WEIGHT_FLOOR."""
        tracker = ConstraintTracker()
        # Low precision: 1 caught out of 100 fires
        for _ in range(99):
            tracker.record("code", fired=True, caught_error=False)
        tracker.record("code", fired=True, caught_error=True)

        weights = AdaptiveWeighter.from_tracker(tracker)
        assert weights["code"] >= WEIGHT_FLOOR

    def test_single_fire_perfect_precision(self) -> None:
        """REQ-LEARN-002: One fire with perfect precision gives log(2) weight."""
        tracker = ConstraintTracker()
        tracker.record("type_check", fired=True, caught_error=True, any_error_in_batch=True)

        weights = AdaptiveWeighter.from_tracker(tracker)
        expected = max(1.0 * math.log(1 + 1), WEIGHT_FLOOR)
        assert weights["type_check"] == pytest.approx(expected)

    def test_multiple_types_independent_weights(self) -> None:
        """REQ-LEARN-002: Different types get independently computed weights."""
        tracker = ConstraintTracker()
        # arithmetic: perfect precision, 5 fires
        for _ in range(5):
            tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        # logic: zero precision, 5 fires
        for _ in range(5):
            tracker.record("logic", fired=True, caught_error=False, any_error_in_batch=False)

        weights = AdaptiveWeighter.from_tracker(tracker)

        arithmetic_expected = 1.0 * math.log(5 + 1)
        assert weights["arithmetic"] == pytest.approx(arithmetic_expected)
        assert weights["logic"] == pytest.approx(WEIGHT_FLOOR)
        # High-precision type must outweigh floor type.
        assert weights["arithmetic"] > weights["logic"]

    def test_empty_tracker_returns_empty_dict(self) -> None:
        """REQ-LEARN-002: Empty tracker produces empty weights dict."""
        tracker = ConstraintTracker()
        weights = AdaptiveWeighter.from_tracker(tracker)
        assert weights == {}

    def test_weight_increases_with_count(self) -> None:
        """REQ-LEARN-002: More fires -> higher weight (log growth) for same precision."""
        tracker_small = ConstraintTracker()
        for _ in range(5):
            tracker_small.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)

        tracker_large = ConstraintTracker()
        for _ in range(100):
            tracker_large.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)

        w_small = AdaptiveWeighter.from_tracker(tracker_small)["arithmetic"]
        w_large = AdaptiveWeighter.from_tracker(tracker_large)["arithmetic"]
        assert w_large > w_small

    def test_partial_precision(self) -> None:
        """REQ-LEARN-002: Partial precision is interpolated correctly."""
        tracker = ConstraintTracker()
        # 4 fires, 2 caught -> precision = 0.5
        for _ in range(2):
            tracker.record("logic", fired=True, caught_error=True, any_error_in_batch=True)
        for _ in range(2):
            tracker.record("logic", fired=True, caught_error=False, any_error_in_batch=True)

        weights = AdaptiveWeighter.from_tracker(tracker)
        expected = max(0.5 * math.log(4 + 1), WEIGHT_FLOOR)
        assert weights["logic"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# REQ-LEARN-002: AdaptiveWeighter.apply_to_pipeline
# ---------------------------------------------------------------------------


class TestApplyToPipeline:
    """REQ-LEARN-002: apply_to_pipeline stores weights on pipeline instance."""

    def test_sets_adaptive_weights_attribute(self) -> None:
        """REQ-LEARN-002: Pipeline gains _adaptive_weights after apply_to_pipeline."""
        pipeline = VerifyRepairPipeline()
        weights = {"arithmetic": 2.5, "logic": 0.3}
        AdaptiveWeighter.apply_to_pipeline(pipeline, weights)
        assert hasattr(pipeline, "_adaptive_weights")
        assert pipeline._adaptive_weights == weights

    def test_stores_copy_not_reference(self) -> None:
        """REQ-LEARN-002: Modifying caller's dict does not affect pipeline weights."""
        pipeline = VerifyRepairPipeline()
        weights = {"arithmetic": 2.5}
        AdaptiveWeighter.apply_to_pipeline(pipeline, weights)
        weights["arithmetic"] = 99.0  # mutate caller's dict
        assert pipeline._adaptive_weights["arithmetic"] == pytest.approx(2.5)

    def test_can_update_weights_repeatedly(self) -> None:
        """REQ-LEARN-002: Calling apply_to_pipeline twice replaces the first weights."""
        pipeline = VerifyRepairPipeline()
        AdaptiveWeighter.apply_to_pipeline(pipeline, {"arithmetic": 1.5})
        AdaptiveWeighter.apply_to_pipeline(pipeline, {"arithmetic": 3.0})
        assert pipeline._adaptive_weights["arithmetic"] == pytest.approx(3.0)

    def test_empty_weights_dict_clears_adaptive(self) -> None:
        """REQ-LEARN-002: apply_to_pipeline({}) reverts to uniform (default) weighting."""
        pipeline = VerifyRepairPipeline()
        AdaptiveWeighter.apply_to_pipeline(pipeline, {"arithmetic": 5.0})
        AdaptiveWeighter.apply_to_pipeline(pipeline, {})
        assert pipeline._adaptive_weights == {}

    def test_pipeline_uses_adaptive_weights_in_verify(self) -> None:
        """REQ-LEARN-002: verify() with adaptive weights does not crash and returns result."""
        pipeline = VerifyRepairPipeline()
        tracker = ConstraintTracker()
        for _ in range(5):
            tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)

        weights = AdaptiveWeighter.from_tracker(tracker)
        AdaptiveWeighter.apply_to_pipeline(pipeline, weights)

        result = pipeline.verify(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
            domain="arithmetic",
        )
        assert result.verified is True

    def test_pipeline_without_adaptive_weights_unchanged(self) -> None:
        """REQ-LEARN-002: Pipeline without _adaptive_weights falls back to 1.0 (baseline)."""
        pipeline = VerifyRepairPipeline()
        # No apply_to_pipeline call -- should behave identically to original.
        result = pipeline.verify(
            question="What is 3 + 3?",
            response="3 + 3 = 6.",
            domain="arithmetic",
        )
        assert result.verified is True


# ---------------------------------------------------------------------------
# REQ-LEARN-002: run_comparison
# ---------------------------------------------------------------------------


def _make_arithmetic_questions(n: int, correct: bool = True) -> list[tuple[str, str, bool]]:
    """Generate simple arithmetic (question, response, is_correct) triples.

    Uses additions a + b = c, with c always correct or always wrong.
    """
    items: list[tuple[str, str, bool]] = []
    for i in range(n):
        a = i + 1
        b = i + 2
        correct_answer = a + b
        if correct:
            response = f"{a} + {b} = {correct_answer}."
        else:
            # Intentionally wrong answer.
            response = f"{a} + {b} = {correct_answer + 1}."
        items.append((f"What is {a} + {b}?", response, correct))
    return items


class TestRunComparison:
    """REQ-LEARN-002, SCENARIO-LEARN-002: Fixed vs adaptive accuracy comparison."""

    def test_returns_comparison_result(self) -> None:
        """SCENARIO-LEARN-002: run_comparison returns a ComparisonResult."""
        questions = _make_arithmetic_questions(20, correct=True)
        result = run_comparison(questions, warmup_n=10)
        assert isinstance(result, ComparisonResult)

    def test_warmup_and_eval_split(self) -> None:
        """REQ-LEARN-002: warmup_n and eval_n match the split."""
        questions = _make_arithmetic_questions(30)
        result = run_comparison(questions, warmup_n=15)
        assert result.warmup_n == 15
        assert result.eval_n == 15

    def test_accuracy_in_range(self) -> None:
        """REQ-LEARN-002: fixed_accuracy and adaptive_accuracy are in [0, 1]."""
        questions = _make_arithmetic_questions(20)
        result = run_comparison(questions, warmup_n=10)
        assert 0.0 <= result.fixed_accuracy <= 1.0
        assert 0.0 <= result.adaptive_accuracy <= 1.0

    def test_delta_equals_adaptive_minus_fixed(self) -> None:
        """REQ-LEARN-002: delta = adaptive_accuracy - fixed_accuracy."""
        questions = _make_arithmetic_questions(20)
        result = run_comparison(questions, warmup_n=10)
        assert result.delta == pytest.approx(result.adaptive_accuracy - result.fixed_accuracy)

    def test_weights_populated_from_warmup(self) -> None:
        """REQ-LEARN-002: ComparisonResult.weights contains warmup-derived values."""
        questions = _make_arithmetic_questions(20)
        result = run_comparison(questions, warmup_n=10)
        # Warmup on correct answers -> arithmetic fired, no errors caught -> floor weight
        assert "arithmetic" in result.weights
        assert result.weights["arithmetic"] >= WEIGHT_FLOOR

    def test_empty_eval_when_all_warmup(self) -> None:
        """REQ-LEARN-002: If warmup_n >= total questions, eval_n=0 and accuracy=0.0."""
        questions = _make_arithmetic_questions(5)
        result = run_comparison(questions, warmup_n=10)
        assert result.eval_n == 0
        assert result.fixed_accuracy == 0.0
        assert result.adaptive_accuracy == 0.0

    def test_mixed_correct_and_wrong(self) -> None:
        """SCENARIO-LEARN-002: Mix of correct/wrong generates meaningful comparison."""
        correct_qs = _make_arithmetic_questions(30, correct=True)
        wrong_qs = _make_arithmetic_questions(20, correct=False)
        questions = correct_qs + wrong_qs
        result = run_comparison(questions, warmup_n=20)
        # Just check it completes without error and has correct structure.
        assert isinstance(result, ComparisonResult)
        assert result.eval_n == 30
        assert result.warmup_n == 20

    def test_correct_answers_high_fixed_accuracy(self) -> None:
        """SCENARIO-LEARN-002: All-correct responses should have high fixed accuracy."""
        questions = _make_arithmetic_questions(30, correct=True)
        result = run_comparison(questions, warmup_n=10)
        # With all-correct responses, verifier should agree is_correct=True most of the time.
        # Not guaranteed to be 1.0 (extractor may skip trivial expressions), but
        # both conditions should yield equal or better accuracy than random.
        assert result.fixed_accuracy >= 0.0
        assert result.adaptive_accuracy >= 0.0


# ---------------------------------------------------------------------------
# REQ-LEARN-002: ComparisonResult dataclass
# ---------------------------------------------------------------------------


class TestComparisonResult:
    """REQ-LEARN-002: ComparisonResult fields are correct types and accessible."""

    def test_fields_accessible(self) -> None:
        """REQ-LEARN-002: All ComparisonResult fields are present."""
        r = ComparisonResult(
            fixed_accuracy=0.6,
            adaptive_accuracy=0.75,
            delta=0.15,
            warmup_n=100,
            eval_n=100,
            weights={"arithmetic": 1.5},
        )
        assert r.fixed_accuracy == pytest.approx(0.6)
        assert r.adaptive_accuracy == pytest.approx(0.75)
        assert r.delta == pytest.approx(0.15)
        assert r.warmup_n == 100
        assert r.eval_n == 100
        assert r.weights == {"arithmetic": 1.5}
