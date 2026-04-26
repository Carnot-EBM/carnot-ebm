"""Tests for DraftConditionedVerifier (Tier 2.8) and ThreeTierPipeline Tier 2.8 wiring.

Spec: REQ-TIER2-010
SCENARIO-TIER2-010
Spec: REQ-AUTO-011
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from carnot.verify.draft_conditioned_verifier import (
    DraftConditionedVerifier,
    draft_differs_from_response,
)

# ---------------------------------------------------------------------------
# extract_structural_constraints tests — REQ-TIER2-010-3
# ---------------------------------------------------------------------------


class TestExtractStructuralConstraints:
    """Verify structural constraint extraction from known arithmetic sentences."""

    def test_equals_pattern_produces_range_constraint(self) -> None:
        """'= 42' in draft → answer_in_range_0_to_85 constraint.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "Step 1: We add 40 and 2. The result = 42."
        constraints = verifier.extract_structural_constraints(draft)
        range_constraints = [c for c in constraints if c.startswith("answer_in_range")]
        assert len(range_constraints) == 1
        # 42 * 2 + 1 = 85
        assert range_constraints[0] == "answer_in_range_0_to_85"

    def test_subtraction_detected(self) -> None:
        """'100 - 37' → arithmetic_op_subtract.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "We have 100 - 37 apples remaining."
        constraints = verifier.extract_structural_constraints(draft)
        assert "arithmetic_op_subtract" in constraints

    def test_addition_detected(self) -> None:
        """'3 + 4' → arithmetic_op_add.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "3 + 4 = 7"
        constraints = verifier.extract_structural_constraints(draft)
        assert "arithmetic_op_add" in constraints

    def test_multiplication_detected(self) -> None:
        """'3 * 4' → arithmetic_op_multiply.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "3 * 4 = 12"
        constraints = verifier.extract_structural_constraints(draft)
        assert "arithmetic_op_multiply" in constraints

    def test_division_detected(self) -> None:
        """'20 / 4' → arithmetic_op_divide.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "20 / 4 = 5"
        constraints = verifier.extract_structural_constraints(draft)
        assert "arithmetic_op_divide" in constraints

    def test_step_count_single_sentence(self) -> None:
        """Single sentence → n_steps_1.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "The answer is 7"
        constraints = verifier.extract_structural_constraints(draft)
        step_constraints = [c for c in constraints if c.startswith("n_steps_")]
        assert len(step_constraints) == 1
        assert step_constraints[0] == "n_steps_1"

    def test_multi_step_draft(self) -> None:
        """3-sentence draft → n_steps_3.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "First, we add 5 + 3 = 8. Then, we subtract 2. So the answer = 6."
        constraints = verifier.extract_structural_constraints(draft)
        step_constraints = [c for c in constraints if c.startswith("n_steps_")]
        assert len(step_constraints) == 1
        assert step_constraints[0] == "n_steps_3"

    def test_empty_draft_returns_empty_list(self) -> None:
        """Empty draft string → empty constraint list.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        constraints = verifier.extract_structural_constraints("")
        assert constraints == []

    def test_multiple_constraints_combined(self) -> None:
        """Complex draft produces multiple constraints.

        Spec: REQ-TIER2-010-3
        """
        verifier = DraftConditionedVerifier()
        draft = "We subtract 50 - 30 = 20. So the answer = 20."
        constraints = verifier.extract_structural_constraints(draft)
        assert "arithmetic_op_subtract" in constraints
        assert any(c.startswith("answer_in_range") for c in constraints)
        assert any(c.startswith("n_steps_") for c in constraints)


# ---------------------------------------------------------------------------
# draft_differs_from_response tests — utility function
# ---------------------------------------------------------------------------


class TestDraftDiffersFromResponse:
    """Verify structural mismatch detection between draft and response."""

    def test_matching_answers_no_mismatch(self) -> None:
        """Same final number → no mismatch."""
        assert not draft_differs_from_response("The answer is 42.", "So we get 42.")

    def test_very_different_answers_mismatch(self) -> None:
        """Draft says 10, response says 100 → mismatch (900% difference)."""
        assert draft_differs_from_response("= 10", "= 100")

    def test_small_difference_no_mismatch(self) -> None:
        """Draft 42, response 45 — within 20% tolerance → no mismatch."""
        assert not draft_differs_from_response("= 42", "= 45")

    def test_no_numbers_no_mismatch(self) -> None:
        """No numbers in either → treat as no mismatch."""
        assert not draft_differs_from_response("The cat sat.", "A cat was there.")


# ---------------------------------------------------------------------------
# condition_and_verify tests — REQ-TIER2-010-4
# ---------------------------------------------------------------------------


class TestConditionAndVerify:
    """Verify condition_and_verify returns correct keys and types.

    Spec: REQ-TIER2-010-4
    """

    def test_returns_required_keys(self) -> None:
        """condition_and_verify dict must contain all four required keys.

        Spec: REQ-TIER2-010-4
        """
        verifier = DraftConditionedVerifier()
        # Patch generate_draft to avoid loading a real model in CI
        with patch.object(verifier, "generate_draft", return_value="3 + 4 = 7. Done."):
            result = verifier.condition_and_verify("What is 3 + 4?", "The answer is 7.")

        assert "draft" in result
        assert "structural_constraints" in result
        assert "draft_mismatch" in result
        assert "tier28_advisory" in result

    def test_tier28_advisory_value(self) -> None:
        """tier28_advisory must be the string 'draft_conditioned'.

        Spec: REQ-TIER2-010-4
        """
        verifier = DraftConditionedVerifier()
        with patch.object(verifier, "generate_draft", return_value="= 7"):
            result = verifier.condition_and_verify("q", "r")
        assert result["tier28_advisory"] == "draft_conditioned"

    def test_structural_constraints_is_list(self) -> None:
        """structural_constraints must be a list.

        Spec: REQ-TIER2-010-4
        """
        verifier = DraftConditionedVerifier()
        with patch.object(verifier, "generate_draft", return_value="5 + 3 = 8"):
            result = verifier.condition_and_verify("5 + 3?", "8")
        assert isinstance(result["structural_constraints"], list)

    def test_draft_mismatch_is_bool(self) -> None:
        """draft_mismatch must be a bool.

        Spec: REQ-TIER2-010-4
        """
        verifier = DraftConditionedVerifier()
        with patch.object(verifier, "generate_draft", return_value="= 8"):
            result = verifier.condition_and_verify("5 + 3?", "The answer is 8.")
        assert isinstance(result["draft_mismatch"], bool)


# ---------------------------------------------------------------------------
# ThreeTierPipeline Tier 2.8 wiring tests — REQ-TIER2-010-5, REQ-TIER2-010-6
# ---------------------------------------------------------------------------


class TestThreeTierPipelineTier28Wiring:
    """Verify Tier 2.8 injection path in ThreeTierPipeline.

    Spec: REQ-TIER2-010-5, REQ-TIER2-010-6
    """

    def _make_pipeline(self) -> Any:
        """Build a minimal ThreeTierPipeline with stub dependencies."""
        from carnot.pipeline.sink_probe import SinkProbe
        from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

        sink_probe = SinkProbe()

        # Stub EORM that always returns high energy (forces Tier 3)
        eorm_model = MagicMock()
        eorm_model.energy.return_value = 1.0

        # Stub Ising pipeline: always returns (True, 0.0)
        def ising_stub(response: str, question: str) -> tuple[bool, float]:
            return True, 0.0

        pipeline = ThreeTierPipeline(
            sink_probe=sink_probe,
            eorm_model=eorm_model,
            ising_pipeline=ising_stub,
            eorm_threshold=0.5,
        )
        return pipeline

    def test_no_tier28_advisory_without_wiring(self) -> None:
        """Without wiring, _last_tier28_advisory is None after verify().

        Spec: REQ-TIER2-010-6 (ADDITIVE — no-op when not wired)
        """
        pipeline = self._make_pipeline()
        pipeline.verify("The answer is 7.", question="What is 3+4?")
        assert pipeline._last_tier28_advisory is None

    def test_tier28_advisory_set_after_wiring(self) -> None:
        """After wiring, verify() populates _last_tier28_advisory.

        Spec: REQ-TIER2-010-5
        """
        pipeline = self._make_pipeline()
        verifier = DraftConditionedVerifier()
        with patch.object(verifier, "generate_draft", return_value="3 + 4 = 7"):
            pipeline.wire_tier_28(verifier)
            pipeline.verify("The answer is 7.", question="What is 3+4?")

        assert pipeline._last_tier28_advisory is not None
        assert pipeline._last_tier28_advisory["tier28_advisory"] == "draft_conditioned"

    def test_structural_constraints_injected_into_question(self) -> None:
        """When Tier 2.8 produces constraints, the Ising question is augmented.

        Verifies the [SC: ...] prefix is prepended to the question string
        passed to ising_pipeline when structural constraints are non-empty.

        Spec: REQ-TIER2-010-5
        """
        pipeline = self._make_pipeline()
        received_questions: list[str] = []

        def capturing_ising(response: str, question: str) -> tuple[bool, float]:
            received_questions.append(question)
            return True, 0.0

        pipeline.ising_pipeline = capturing_ising

        verifier = DraftConditionedVerifier()
        with patch.object(verifier, "generate_draft", return_value="5 - 3 = 2"):
            pipeline.wire_tier_28(verifier)
            pipeline.verify("The answer is 2.", question="What is 5-3?")

        assert len(received_questions) == 1
        assert received_questions[0].startswith("[SC:")

    def test_no_prefix_when_constraints_empty(self) -> None:
        """When structural_constraints is empty, Ising question is unchanged.

        Spec: REQ-TIER2-010-5
        """
        pipeline = self._make_pipeline()
        received_questions: list[str] = []

        def capturing_ising(response: str, question: str) -> tuple[bool, float]:
            received_questions.append(question)
            return True, 0.0

        pipeline.ising_pipeline = capturing_ising

        verifier = DraftConditionedVerifier()
        # Empty draft → no constraints → question unchanged
        with patch.object(verifier, "generate_draft", return_value=""):
            pipeline.wire_tier_28(verifier)
            pipeline.verify("Some response.", question="Some question?")

        assert len(received_questions) == 1
        assert not received_questions[0].startswith("[SC:")
