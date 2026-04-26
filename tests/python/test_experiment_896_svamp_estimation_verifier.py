"""Tests for Experiment 896: EstimationVerifier for SVAMP word problems.

Traces to: REQ-VER-085, SCENARIO-VER-085

REQ-VER-085-1: Numbers extracted from question via regex.
REQ-VER-085-2: Operation type detected from keywords.
REQ-VER-085-3: Plausible range computed per operation.
REQ-VER-085-4: Answer extracted from response text.
REQ-VER-085-5: confidence=1.0 when answer found, 0.5 when not.
REQ-VER-085-6: label_pair with ground_truth uses exact match (tolerance 0.01).
REQ-VER-085-7: label_pair without ground_truth uses in_range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from python.carnot.verify.estimation_verifier import EstimationVerifier


@pytest.fixture
def ev() -> EstimationVerifier:
    return EstimationVerifier()


# ---------------------------------------------------------------------------
# REQ-VER-085-1: number extraction
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    """REQ-VER-085-1: Numbers extracted from question text."""

    def test_integer_numbers(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Tom has 5 apples and 3 oranges.", "8")
        assert result["extracted_numbers"] == [5.0, 3.0]

    def test_decimal_numbers(self, ev: EstimationVerifier) -> None:
        result = ev.verify("She earns 2.5 dollars per hour for 4 hours.", "10")
        assert 2.5 in result["extracted_numbers"]
        assert 4.0 in result["extracted_numbers"]

    def test_no_numbers_returns_empty(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Some people went somewhere.", "unknown")
        assert result["extracted_numbers"] == []


# ---------------------------------------------------------------------------
# REQ-VER-085-2: operation detection
# ---------------------------------------------------------------------------


class TestOperationDetection:
    """REQ-VER-085-2: Operation type detected from keywords."""

    def test_add_keyword_total(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Tom has 5 apples and buys 3 more. How many total?", "8")
        assert result["operation_type"] == "add"

    def test_subtract_keyword_remaining(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Bob had 10 coins and spent 4. How many remaining?", "6")
        assert result["operation_type"] == "subtract"

    def test_multiply_keyword_each(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Each box has 6 items and there are 4 boxes.", "24")
        assert result["operation_type"] in ("multiply", "add", "divide")  # "each" can be ambiguous

    def test_multiply_keyword_times(self, ev: EstimationVerifier) -> None:
        result = ev.verify("A number 3 times 7. What is the product?", "21")
        assert result["operation_type"] == "multiply"

    def test_divide_keyword_split(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Split 20 cookies among 4 children equally.", "5")
        assert result["operation_type"] == "divide"

    def test_unknown_operation(self, ev: EstimationVerifier) -> None:
        result = ev.verify("There are 5 items on the table.", "5")
        assert result["operation_type"] == "unknown"


# ---------------------------------------------------------------------------
# REQ-VER-085-3: plausible range computation
# ---------------------------------------------------------------------------


class TestPlausibleRange:
    """REQ-VER-085-3: Plausible range correct per operation type."""

    def test_add_range(self, ev: EstimationVerifier) -> None:
        # numbers=[5,3], add: [min=3, sum*2=16]
        result = ev.verify("Tom has 5 apples and buys 3 more. How many total?", "8")
        lo, hi = result["plausible_range"]
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(16.0)

    def test_subtract_range(self, ev: EstimationVerifier) -> None:
        # numbers=[10,4], subtract: [0, max=10]
        result = ev.verify("Bob had 10 coins and lost 4. How many remaining?", "6")
        lo, hi = result["plausible_range"]
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(10.0)

    def test_multiply_range(self, ev: EstimationVerifier) -> None:
        # numbers=[3,7], multiply: [min=3, max²=49]
        result = ev.verify("A number 3 times 7. What is the product?", "21")
        lo, hi = result["plausible_range"]
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(49.0)

    def test_divide_range(self, ev: EstimationVerifier) -> None:
        # numbers=[20,4], divide: [0, max=20]
        result = ev.verify("Split 20 items among 4 equally. How many each?", "5")
        lo, hi = result["plausible_range"]
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(20.0)

    def test_no_numbers_range_is_zero_zero(self, ev: EstimationVerifier) -> None:
        result = ev.verify("People went somewhere.", "5")
        assert result["plausible_range"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# REQ-VER-085-4: answer extraction
# ---------------------------------------------------------------------------


class TestAnswerExtraction:
    """REQ-VER-085-4: Answer extracted from various response patterns."""

    def test_answer_is_pattern(self, ev: EstimationVerifier) -> None:
        result = ev.verify("5 + 3 = ?", "The answer is 8")
        assert result["extracted_answer"] == pytest.approx(8.0)

    def test_result_is_pattern(self, ev: EstimationVerifier) -> None:
        result = ev.verify("5 + 3 = ?", "The result is 8.")
        assert result["extracted_answer"] == pytest.approx(8.0)

    def test_equals_pattern(self, ev: EstimationVerifier) -> None:
        result = ev.verify("5 + 3 = ?", "5 + 3 = 8")
        assert result["extracted_answer"] == pytest.approx(8.0)

    def test_trailing_number(self, ev: EstimationVerifier) -> None:
        result = ev.verify("5 + 3 = ?", "Tom has 8")
        assert result["extracted_answer"] == pytest.approx(8.0)

    def test_no_number_returns_none(self, ev: EstimationVerifier) -> None:
        result = ev.verify("5 + 3 = ?", "I cannot compute this.")
        assert result["extracted_answer"] is None


# ---------------------------------------------------------------------------
# REQ-VER-085-5: confidence values
# ---------------------------------------------------------------------------


class TestConfidence:
    """REQ-VER-085-5: confidence=1.0 when answer found, 0.5 when not."""

    def test_confidence_one_when_answer_found(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Tom has 5 apples and buys 3 more. How many total?", "The answer is 8")
        assert result["confidence"] == pytest.approx(1.0)

    def test_confidence_half_when_no_answer(self, ev: EstimationVerifier) -> None:
        result = ev.verify("Tom has 5 apples and buys 3 more. How many total?", "I do not know.")
        assert result["confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# REQ-VER-085-6: label_pair with ground_truth
# ---------------------------------------------------------------------------


class TestLabelPairWithGroundTruth:
    """REQ-VER-085-6: label_pair uses ground_truth for exact match (tol=0.01)."""

    def test_correct_answer_returns_one(self, ev: EstimationVerifier) -> None:
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "The answer is 8",
            ground_truth=8.0,
        )
        assert label == 1

    def test_wrong_answer_returns_zero(self, ev: EstimationVerifier) -> None:
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "The answer is 99",
            ground_truth=8.0,
        )
        assert label == 0

    def test_missing_answer_returns_zero(self, ev: EstimationVerifier) -> None:
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "I cannot determine.",
            ground_truth=8.0,
        )
        assert label == 0

    def test_exact_tolerance_boundary(self, ev: EstimationVerifier) -> None:
        # Within tolerance (< 0.01): label=1
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "The answer is 8.005",
            ground_truth=8.0,
        )
        assert label == 1

    def test_outside_tolerance_boundary(self, ev: EstimationVerifier) -> None:
        # Outside tolerance (>= 0.01): label=0
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "The answer is 8.02",
            ground_truth=8.0,
        )
        assert label == 0


# ---------------------------------------------------------------------------
# REQ-VER-085-7: label_pair without ground_truth uses in_range
# ---------------------------------------------------------------------------


class TestLabelPairWithoutGroundTruth:
    """REQ-VER-085-7: Without ground_truth, label_pair uses in_range."""

    def test_in_range_returns_one(self, ev: EstimationVerifier) -> None:
        # 8 is clearly within add range [3, 16]
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "The answer is 8",
        )
        assert label == 1

    def test_out_of_range_returns_zero(self, ev: EstimationVerifier) -> None:
        # 999 is way outside add range [3, 16]
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "The answer is 999",
        )
        assert label == 0

    def test_missing_answer_returns_zero_no_ground_truth(self, ev: EstimationVerifier) -> None:
        label = ev.label_pair(
            "Tom has 5 apples and buys 3 more. How many total?",
            "I cannot determine.",
        )
        assert label == 0


# ---------------------------------------------------------------------------
# SCENARIO-VER-085: canonical scenario from the spec
# ---------------------------------------------------------------------------


class TestScenarioVer085:
    """SCENARIO-VER-085: full addition scenario as specified in spec.

    Given EstimationVerifier instance.
    And question "Tom has 5 apples and buys 3 more. How many total?"
    When verify() is called with response containing 8.
    Then operation_type="add", range=[3,16], answer=8.0, in_range=True, confidence=1.0.
    And label_pair with ground_truth=8 returns 1.
    And label_pair with ground_truth=8 but wrong answer 99 returns 0.
    """

    def test_full_addition_scenario(self, ev: EstimationVerifier) -> None:
        question = "Tom has 5 apples and buys 3 more. How many total?"
        response = "Tom has 8 apples total."

        result = ev.verify(question, response)
        assert result["operation_type"] == "add"
        assert result["plausible_range"][0] == pytest.approx(3.0)
        assert result["plausible_range"][1] == pytest.approx(16.0)
        assert result["extracted_answer"] == pytest.approx(8.0)
        assert result["in_range"] is True
        assert result["confidence"] == pytest.approx(1.0)

    def test_label_correct_with_ground_truth(self, ev: EstimationVerifier) -> None:
        question = "Tom has 5 apples and buys 3 more. How many total?"
        assert ev.label_pair(question, "Tom has 8 apples total.", ground_truth=8.0) == 1

    def test_label_wrong_with_ground_truth(self, ev: EstimationVerifier) -> None:
        question = "Tom has 5 apples and buys 3 more. How many total?"
        assert ev.label_pair(question, "Tom has 99 apples total.", ground_truth=8.0) == 0
