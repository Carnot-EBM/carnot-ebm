"""Tests for carnot.pipeline.hisr_weights — 100% coverage target.

Spec: REQ-LEARN-072, SCENARIO-LEARN-113, SCENARIO-LEARN-114
"""

from __future__ import annotations

import pytest

from carnot.pipeline.constraint_addition import ViolationPattern
from carnot.pipeline.hisr_weights import HISRViolationWeight, HISRWeighter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_violations(n: int, vtype: str = "carry") -> list[ViolationPattern]:
    """Build a list of n identical ViolationPattern objects for testing."""
    return [ViolationPattern(type=vtype, count=1, example_steps=[]) for _ in range(n)]


# ---------------------------------------------------------------------------
# HISRViolationWeight dataclass
# ---------------------------------------------------------------------------


def test_hisr_violation_weight_fields() -> None:
    """HISRViolationWeight stores all four fields correctly.

    Spec: REQ-LEARN-072
    """
    w = HISRViolationWeight(
        violation_type="carry",
        question_id="q001",
        final_incorrect=True,
        hindsight_score=0.75,
    )
    assert w.violation_type == "carry"
    assert w.question_id == "q001"
    assert w.final_incorrect is True
    assert w.hindsight_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# HISRWeighter.compute_hindsight_score
# ---------------------------------------------------------------------------


def test_empty_violations_returns_empty() -> None:
    """Empty violation list produces empty weight list.

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    result = weighter.compute_hindsight_score([], final_correct=False)
    assert result == []


def test_empty_violations_correct_chain_returns_empty() -> None:
    """Empty violations in a correct chain also returns empty list.

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    result = weighter.compute_hindsight_score([], final_correct=True)
    assert result == []


def test_correct_chain_all_scores_zero() -> None:
    """Violations in a correct chain (final_correct=True) all receive score 0.0.

    SCENARIO-LEARN-113: violations in correct chains are false positives.
    Spec: REQ-LEARN-072, SCENARIO-LEARN-113
    """
    weighter = HISRWeighter()
    violations = _make_violations(5)
    weights = weighter.compute_hindsight_score(violations, final_correct=True)

    assert len(weights) == 5
    for w in weights:
        assert w.hindsight_score == pytest.approx(0.0)
        assert w.final_incorrect is False


def test_single_violation_incorrect_chain_scores_one() -> None:
    """A single violation in an incorrect chain scores exactly 1.0.

    WHY: distance_from_last=0, so 1/(1+0) = 1.0.
    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = _make_violations(1)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)

    assert len(weights) == 1
    assert weights[0].hindsight_score == pytest.approx(1.0)
    assert weights[0].final_incorrect is True


def test_incorrect_chain_last_violation_scores_highest() -> None:
    """In an incorrect chain, the last violation has the highest hindsight score.

    SCENARIO-LEARN-114: violations near the final error score higher.
    Spec: REQ-LEARN-072, SCENARIO-LEARN-114
    """
    weighter = HISRWeighter()
    violations = _make_violations(5)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)

    assert len(weights) == 5
    # Scores must be strictly increasing (last is highest).
    scores = [w.hindsight_score for w in weights]
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1], f"score[{i}]={scores[i]} >= score[{i+1}]={scores[i+1]}"


def test_incorrect_chain_score_formula() -> None:
    """Verify exact score values for a 3-violation incorrect chain.

    Score formula: 1.0 / (1 + distance_from_last).
    For N=3: scores should be [1/3, 1/2, 1.0].
    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = _make_violations(3)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)

    assert weights[0].hindsight_score == pytest.approx(1.0 / 3)
    assert weights[1].hindsight_score == pytest.approx(1.0 / 2)
    assert weights[2].hindsight_score == pytest.approx(1.0)


def test_weights_preserve_violation_type() -> None:
    """Each weight preserves the violation_type from its corresponding ViolationPattern.

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = [
        ViolationPattern(type="carry", count=1, example_steps=[]),
        ViolationPattern(type="sign", count=2, example_steps=["x"]),
        ViolationPattern(type="unit", count=3, example_steps=["y", "z"]),
    ]
    weights = weighter.compute_hindsight_score(violations, final_correct=False)

    assert weights[0].violation_type == "carry"
    assert weights[1].violation_type == "sign"
    assert weights[2].violation_type == "unit"


def test_weights_question_id_is_empty_string() -> None:
    """compute_hindsight_score sets question_id to empty string by default.

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = _make_violations(2)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)
    for w in weights:
        assert w.question_id == ""


# ---------------------------------------------------------------------------
# HISRWeighter.weighted_violations
# ---------------------------------------------------------------------------


def test_weighted_violations_threshold_default() -> None:
    """weighted_violations with default threshold=0.5 filters low scores.

    For a 3-violation incorrect chain, score[0]=1/3 < 0.5 so it is filtered out.
    score[1]=0.5 >= 0.5 and score[2]=1.0 >= 0.5, so two violations are retained.
    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = _make_violations(3)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)
    promoted = weighter.weighted_violations(weights)

    assert len(promoted) == 2  # scores 0.5 and 1.0 pass threshold


def test_weighted_violations_all_zero_scores_filtered() -> None:
    """weighted_violations returns empty list when all scores are 0.0 (correct chain).

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = _make_violations(5)
    weights = weighter.compute_hindsight_score(violations, final_correct=True)
    promoted = weighter.weighted_violations(weights)

    assert promoted == []


def test_weighted_violations_custom_threshold() -> None:
    """weighted_violations respects a custom threshold value.

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = _make_violations(4)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)
    # Scores: [0.25, 0.333, 0.5, 1.0]. Threshold 0.4 keeps the last two.
    promoted = weighter.weighted_violations(weights, threshold=0.4)
    assert len(promoted) == 2


def test_weighted_violations_returns_violation_patterns() -> None:
    """weighted_violations returns ViolationPattern objects with correct type labels.

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    violations = [
        ViolationPattern(type="carry", count=1, example_steps=[]),
        ViolationPattern(type="sign", count=1, example_steps=[]),
    ]
    weights = weighter.compute_hindsight_score(violations, final_correct=False)
    promoted = weighter.weighted_violations(weights, threshold=0.0)

    assert len(promoted) == 2
    assert all(isinstance(p, ViolationPattern) for p in promoted)
    assert promoted[0].type == "carry"
    assert promoted[1].type == "sign"


def test_weighted_violations_threshold_exactly_at_score() -> None:
    """Violation with score exactly equal to threshold is included (>=).

    Spec: REQ-LEARN-072
    """
    weighter = HISRWeighter()
    # 2-violation incorrect chain: scores [0.5, 1.0]. Threshold exactly 0.5.
    violations = _make_violations(2)
    weights = weighter.compute_hindsight_score(violations, final_correct=False)
    promoted = weighter.weighted_violations(weights, threshold=0.5)
    assert len(promoted) == 2  # both meet the threshold


def test_hisr_weighter_correct_and_incorrect_combined() -> None:
    """Incorrect chain scores strictly exceed correct chain scores for same violations.

    Confirms that HISR distinguishes learning signal from noise.
    Spec: REQ-LEARN-072, SCENARIO-LEARN-113, SCENARIO-LEARN-114
    """
    weighter = HISRWeighter()
    violations = _make_violations(10, vtype="carry")

    correct_weights = weighter.compute_hindsight_score(violations, final_correct=True)
    incorrect_weights = weighter.compute_hindsight_score(violations, final_correct=False)

    correct_scores = [w.hindsight_score for w in correct_weights]
    incorrect_scores = [w.hindsight_score for w in incorrect_weights]

    assert all(s == 0.0 for s in correct_scores)
    assert all(s > 0.0 for s in incorrect_scores)
    # Last violation in incorrect chain should score highest.
    assert incorrect_scores[-1] == pytest.approx(1.0)
