"""Tests for rprm_step_reward.py — RPRMStepReward Tier 2.9.

Validates that the reasoning-driven step reward model:
  - Returns StepReasoningResult with correct fields in heuristic mode.
  - Flags suspicious patterns with step_score=0.7 and clean steps with 0.1.
  - Aggregates verify_response correctly (max score, n_flagged, repair_hints).
  - Splits steps robustly on edge cases (empty, single step, multi-sentence).
  - LLM mode maps VERDICT tokens to scores 0.9/0.5/0.1.

Spec: REQ-VERIFY-148, SCENARIO-VERIFY-148
"""

from __future__ import annotations

import pytest

from python.carnot.verify.rprm_step_reward import (
    RPRMResult,
    RPRMStepReward,
    StepReasoningResult,
)


# ---------------------------------------------------------------------------
# StepReasoningResult dataclass
# ---------------------------------------------------------------------------


class TestStepReasoningResult:
    """REQ-VERIFY-148-2: StepReasoningResult stores all four required fields."""

    def test_fields_accessible(self):
        r = StepReasoningResult(
            step_text="x = 5",
            reasoning="ok",
            step_score=0.1,
            reasoning_mode="heuristic",
        )
        assert r.step_text == "x = 5"
        assert r.reasoning == "ok"
        assert r.step_score == 0.1
        assert r.reasoning_mode == "heuristic"

    def test_llm_mode_string(self):
        r = StepReasoningResult("s", "text", 0.9, "llm")
        assert r.reasoning_mode == "llm"


# ---------------------------------------------------------------------------
# Heuristic scoring — REQ-VERIFY-148-3
# ---------------------------------------------------------------------------


class TestHeuristicScoring:
    """REQ-VERIFY-148-1, REQ-VERIFY-148-3: heuristic mode is CI-safe (no LLM)."""

    def setup_method(self):
        self.rprm = RPRMStepReward(llm_runner=None)

    def test_clean_step_scores_low(self):
        # A normal arithmetic step should not be flagged.
        result = self.rprm.score_step_with_reasoning(
            "Multiply 3 by 4 to get 12", context="What is 3 times 4?"
        )
        assert result.step_score == pytest.approx(0.1)
        assert result.reasoning == "ok"
        assert result.reasoning_mode == "heuristic"

    def test_zero_result_flagged(self):
        # "= 0" in a long step is suspicious (REQ-VERIFY-148-3, pattern 1).
        result = self.rprm.score_step_with_reasoning(
            "Total revenue = 0 because all items were returned by the buyer",
            context="How much was earned?",
        )
        assert result.step_score == pytest.approx(0.7)
        assert "suspicious" in result.reasoning

    def test_multiple_equals_flagged(self):
        # More than two equals signs signals a contradictory rewrite (pattern 2).
        result = self.rprm.score_step_with_reasoning(
            "x = 5 = 10 = 20",
            context="Solve for x.",
        )
        assert result.step_score == pytest.approx(0.7)

    def test_division_by_zero_pattern_flagged(self):
        # Literal "0" followed by "=" is the div-by-zero hint (pattern 3).
        result = self.rprm.score_step_with_reasoning(
            "0 = the base value so answer follows",
            context="What is the base?",
        )
        assert result.step_score == pytest.approx(0.7)

    def test_short_zero_result_not_flagged(self):
        # "= 0" only triggers when step length > 20 (short steps are exempt).
        result = self.rprm.score_step_with_reasoning(
            "x = 0",  # length < 20
            context="Solve.",
        )
        assert result.step_score == pytest.approx(0.1)

    def test_reasoning_mode_is_heuristic(self):
        result = self.rprm.score_step_with_reasoning("Add 3 and 4 to get 7", "q")
        assert result.reasoning_mode == "heuristic"


# ---------------------------------------------------------------------------
# LLM mode — REQ-VERIFY-148-4
# ---------------------------------------------------------------------------


class TestLLMScoring:
    """REQ-VERIFY-148-4: LLM mode maps VERDICT tokens to scores."""

    def _make_rprm(self, verdict_word: str) -> RPRMStepReward:
        def mock_llm(prompt: str, max_tokens: int) -> str:
            return f"This step looks off. VERDICT: {verdict_word}"

        return RPRMStepReward(llm_runner=mock_llm, n_reasoning_tokens=50)

    def test_wrong_verdict_scores_09(self):
        rprm = self._make_rprm("wrong")
        result = rprm.score_step_with_reasoning("Some step text here", "context")
        assert result.step_score == pytest.approx(0.9)
        assert result.reasoning_mode == "llm"

    def test_suspicious_verdict_scores_05(self):
        rprm = self._make_rprm("suspicious")
        result = rprm.score_step_with_reasoning("Some step text here", "context")
        assert result.step_score == pytest.approx(0.5)

    def test_correct_verdict_scores_01(self):
        rprm = self._make_rprm("correct")
        result = rprm.score_step_with_reasoning("Some step text here", "context")
        assert result.step_score == pytest.approx(0.1)

    def test_no_verdict_token_defaults_to_01(self):
        def mock_llm(prompt: str, max_tokens: int) -> str:
            return "Looks fine to me, no verdict tag here."

        rprm = RPRMStepReward(llm_runner=mock_llm)
        result = rprm.score_step_with_reasoning("x + y = 5", "q")
        assert result.step_score == pytest.approx(0.1)

    def test_reasoning_text_stored(self):
        def mock_llm(prompt: str, max_tokens: int) -> str:
            return "The step divides by zero. VERDICT: wrong"

        rprm = RPRMStepReward(llm_runner=mock_llm)
        result = rprm.score_step_with_reasoning("n / 0 = answer", "q")
        assert "divides by zero" in result.reasoning


# ---------------------------------------------------------------------------
# _split_steps edge cases — REQ-VERIFY-148-6
# ---------------------------------------------------------------------------


class TestSplitSteps:
    """REQ-VERIFY-148-6: _split_steps handles edge cases correctly."""

    def setup_method(self):
        self.rprm = RPRMStepReward()

    def test_empty_string_returns_empty(self):
        assert self.rprm._split_steps("") == []

    def test_short_fragment_discarded(self):
        # Fragments <= 10 chars are discarded.
        assert self.rprm._split_steps("Hi. Ok.") == []

    def test_single_long_step(self):
        steps = self.rprm._split_steps("Multiply 3 by 4 to get the total of 12")
        assert len(steps) == 1
        assert "Multiply" in steps[0]

    def test_multi_sentence_split(self):
        text = "Step 1: Add the numbers. Step 2: Multiply by two. Step 3: Final answer."
        steps = self.rprm._split_steps(text)
        # Each sentence becomes a separate step (period delimiter).
        assert len(steps) >= 2

    def test_newline_split(self):
        text = "First compute the sum\nThen divide by the count\nReport the result"
        steps = self.rprm._split_steps(text)
        assert len(steps) == 3

    def test_strips_whitespace(self):
        steps = self.rprm._split_steps("  Leading and trailing whitespace here  ")
        assert all(s == s.strip() for s in steps)


# ---------------------------------------------------------------------------
# verify_response aggregation — REQ-VERIFY-148-5, SCENARIO-VERIFY-148
# ---------------------------------------------------------------------------


class TestVerifyResponse:
    """REQ-VERIFY-148-5, SCENARIO-VERIFY-148: verify_response aggregates correctly."""

    def setup_method(self):
        self.rprm = RPRMStepReward(llm_runner=None)

    def test_empty_response_returns_zero_prob(self):
        result = self.rprm.verify_response("question", "")
        assert isinstance(result, RPRMResult)
        assert result.overall_violation_prob == pytest.approx(0.0)
        assert result.n_flagged == 0
        assert result.repair_hints == []

    def test_clean_response_low_violation(self):
        resp = "Step 1: Read the problem carefully.\nStep 2: Add 5 and 3 to get 8.\nStep 3: The answer is 8."
        result = self.rprm.verify_response("What is 5 + 3?", resp)
        assert result.overall_violation_prob == pytest.approx(0.1)
        assert result.n_flagged == 0
        assert result.repair_hints == []

    def test_suspicious_response_flagged(self):
        # Inject a step that triggers the heuristic.
        resp = (
            "Step 1: Read the problem.\n"
            "Step 2: Total = 0 because items were cancelled and returned fully.\n"
            "Step 3: The answer is 0."
        )
        result = self.rprm.verify_response("How much was earned?", resp)
        assert result.overall_violation_prob == pytest.approx(0.7)
        assert result.n_flagged >= 1
        assert len(result.repair_hints) >= 1
        assert all("suspicious" in h for h in result.repair_hints)

    def test_overall_prob_is_max_step_score(self):
        # With one flagged step (0.7) and clean steps (0.1), max should be 0.7.
        resp = (
            "Step 1: Compute the total items in the store.\n"
            "Step 2: Revenue = 0 because of total cancellation of all orders received.\n"
            "Step 3: Answer is correct."
        )
        result = self.rprm.verify_response("Revenue?", resp)
        assert result.overall_violation_prob == pytest.approx(
            max(r.step_score for r in result.steps)
        )

    def test_n_flagged_matches_threshold(self):
        resp = (
            "Step 1: Total = 0 because all items were returned to the original supplier.\n"
            "Step 2: Subtotal = 0 because costs were also reversed and cancelled fully.\n"
            "Step 3: Final sum equals twenty dollars."
        )
        result = self.rprm.verify_response("q", resp)
        expected_flagged = sum(1 for r in result.steps if r.step_score > 0.5)
        assert result.n_flagged == expected_flagged

    def test_steps_field_populated(self):
        resp = "Calculate the first part.\nThen calculate the second part."
        result = self.rprm.verify_response("q", resp)
        assert isinstance(result.steps, list)
        assert all(isinstance(s, StepReasoningResult) for s in result.steps)
