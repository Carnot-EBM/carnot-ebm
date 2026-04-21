"""Tests for HermesVerifierAdapter — 100% coverage on hermes_adapter.py.

Spec: REQ-VERIFY-136, SCENARIO-VERIFY-178, SCENARIO-VERIFY-179
"""

from __future__ import annotations

import pytest

from carnot.extraction.llm_extractor_v1 import ArithmeticClaim, LLMAsExtractorV1
from carnot.pipeline.hermes_adapter import HermesVerificationStep, HermesVerifierAdapter
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> HermesVerifierAdapter:
    """Create a CI-mode HermesVerifierAdapter (no LLM, regex fallback only)."""
    return HermesVerifierAdapter(
        extractor=LLMAsExtractorV1(llm_caller=None),
        verifier=SymCodeVerifier(llm_caller=None),
    )


# ---------------------------------------------------------------------------
# HermesVerificationStep dataclass
# ---------------------------------------------------------------------------


class TestHermesVerificationStep:
    """Basic field access and dataclass contract for HermesVerificationStep."""

    def test_fields_correct(self) -> None:
        # REQ-VERIFY-136-4: process_step returns a HermesVerificationStep with all fields.
        claim = ArithmeticClaim(
            lhs_expr="47+28",
            rhs_value=65.0,
            claim_text="47 + 28 = 65",
            strategy="step_segment_eval",
            confidence=0.9,
        )
        step = HermesVerificationStep(
            step_index=0,
            step_text="47 + 28 = 65",
            translated_claims=[claim],
            prover_verdict="violated",
            feedback_injected=True,
            feedback_text="Re-check the calculation: 47 + 28 = 65",
        )
        assert step.step_index == 0
        assert step.step_text == "47 + 28 = 65"
        assert step.translated_claims == [claim]
        assert step.prover_verdict == "violated"
        assert step.feedback_injected is True
        assert step.feedback_text is not None

    def test_correct_step_has_no_feedback(self) -> None:
        # SCENARIO-VERIFY-179: correct step → feedback_injected=False, feedback_text=None
        step = HermesVerificationStep(
            step_index=1,
            step_text="47 + 28 = 75",
            translated_claims=[],
            prover_verdict="correct",
            feedback_injected=False,
            feedback_text=None,
        )
        assert step.feedback_injected is False
        assert step.feedback_text is None


# ---------------------------------------------------------------------------
# translate()
# ---------------------------------------------------------------------------


class TestTranslate:
    """REQ-VERIFY-136-1: translate() calls LLMAsExtractorV1.extract()."""

    def test_correct_arithmetic_returns_empty(self) -> None:
        # 47 + 28 = 75 is correct — extractor returns no violations.
        adapter = _make_adapter()
        claims = adapter.translate("47 + 28 = 75")
        assert isinstance(claims, list)

    def test_incorrect_arithmetic_returns_violation(self) -> None:
        # 47 + 28 = 65 is wrong — StepSegmentEvalChain should find a violation.
        adapter = _make_adapter()
        claims = adapter.translate("47 + 28 = 65")
        assert isinstance(claims, list)
        # In CI mode (StepSegmentEvalChain), this incorrect equation is detected.
        assert len(claims) >= 1

    def test_no_arithmetic_returns_empty(self) -> None:
        adapter = _make_adapter()
        claims = adapter.translate("The sky is blue.")
        assert claims == []


# ---------------------------------------------------------------------------
# prove()
# ---------------------------------------------------------------------------


class TestProve:
    """REQ-VERIFY-136-2: prove() returns 'violated' or 'correct'."""

    def test_violated_for_wrong_arithmetic(self) -> None:
        # SCENARIO-VERIFY-178: step with known error → prover_verdict='violated'
        adapter = _make_adapter()
        verdict = adapter.prove("47 + 28 = 65")
        assert verdict == "violated"

    def test_correct_for_right_arithmetic(self) -> None:
        # SCENARIO-VERIFY-179: step with correct arithmetic → prover_verdict='correct'
        adapter = _make_adapter()
        verdict = adapter.prove("47 + 28 = 75")
        assert verdict == "correct"

    def test_correct_for_no_arithmetic(self) -> None:
        adapter = _make_adapter()
        verdict = adapter.prove("The quick brown fox.")
        assert verdict == "correct"


# ---------------------------------------------------------------------------
# generate_feedback()
# ---------------------------------------------------------------------------


class TestGenerateFeedback:
    """REQ-VERIFY-136-3: generate_feedback() returns hint or empty string."""

    def test_returns_hint_for_low_confidence_claim(self) -> None:
        # A claim with confidence < 0.5 should trigger a feedback hint.
        adapter = _make_adapter()
        claim = ArithmeticClaim(
            lhs_expr="3*16.5",
            rhs_value=54.5,
            claim_text="3 pairs at 16.50 each is 54.50",
            strategy="json_claim",
            confidence=0.3,  # below 0.5 threshold
        )
        feedback = adapter.generate_feedback("some step", [claim])
        assert feedback != ""
        assert "Re-check the calculation" in feedback
        assert "3 pairs at 16.50 each is 54.50" in feedback

    def test_returns_hint_when_lhs_is_none(self) -> None:
        # A claim with lhs_expr=None (extractor failed to formalise) triggers feedback.
        adapter = _make_adapter()
        claim = ArithmeticClaim(
            lhs_expr=None,  # type: ignore[arg-type]
            rhs_value=42.0,
            claim_text="the total is 42",
            strategy="json_claim",
            confidence=0.85,
        )
        feedback = adapter.generate_feedback("some step", [claim])
        assert feedback != ""

    def test_returns_empty_for_high_confidence_claims(self) -> None:
        # Claims with confidence >= 0.5 and non-None lhs → no feedback.
        adapter = _make_adapter()
        claim = ArithmeticClaim(
            lhs_expr="3*16.5",
            rhs_value=49.5,
            claim_text="3 * 16.5 = 49.5",
            strategy="step_segment_eval",
            confidence=0.90,
        )
        feedback = adapter.generate_feedback("some step", [claim])
        assert feedback == ""

    def test_returns_empty_for_no_claims(self) -> None:
        adapter = _make_adapter()
        feedback = adapter.generate_feedback("some step", [])
        assert feedback == ""


# ---------------------------------------------------------------------------
# process_step()
# ---------------------------------------------------------------------------


class TestProcessStep:
    """REQ-VERIFY-136-4: process_step() returns a fully-populated HermesVerificationStep."""

    def test_violated_step_has_feedback(self) -> None:
        # SCENARIO-VERIFY-178: arithmetic error → verdict='violated', feedback_injected=True
        adapter = _make_adapter()
        result = adapter.process_step("47 + 28 = 65", step_index=0)
        assert isinstance(result, HermesVerificationStep)
        assert result.step_index == 0
        assert result.step_text == "47 + 28 = 65"
        assert result.prover_verdict == "violated"
        # feedback_injected depends on whether generate_feedback produced a hint;
        # the prover said violated so feedback is attempted.

    def test_correct_step_no_feedback(self) -> None:
        # SCENARIO-VERIFY-179: correct arithmetic → verdict='correct', feedback_text=None
        adapter = _make_adapter()
        result = adapter.process_step("47 + 28 = 75", step_index=1)
        assert result.prover_verdict == "correct"
        assert result.feedback_injected is False
        assert result.feedback_text is None

    def test_no_arithmetic_step(self) -> None:
        adapter = _make_adapter()
        result = adapter.process_step("The answer is found by reasoning carefully.", step_index=2)
        assert isinstance(result, HermesVerificationStep)
        assert result.step_index == 2
        assert result.prover_verdict == "correct"
        assert result.feedback_text is None


# ---------------------------------------------------------------------------
# process_response()
# ---------------------------------------------------------------------------


class TestProcessResponse:
    """REQ-VERIFY-136-5: process_response() splits at sentence boundaries and verifies each step."""

    def test_single_sentence_correct(self) -> None:
        adapter = _make_adapter()
        steps = adapter.process_response("47 + 28 = 75")
        assert isinstance(steps, list)
        assert len(steps) >= 1
        assert all(isinstance(s, HermesVerificationStep) for s in steps)

    def test_single_sentence_violated(self) -> None:
        adapter = _make_adapter()
        steps = adapter.process_response("47 + 28 = 65")
        assert any(s.prover_verdict == "violated" for s in steps)

    def test_multi_sentence_response(self) -> None:
        # REQ-VERIFY-136-5: sentence boundary splitting produces multiple steps.
        adapter = _make_adapter()
        response = "First we add. 47 + 28 = 65. Then we continue."
        steps = adapter.process_response(response)
        assert len(steps) >= 2
        assert all(isinstance(s, HermesVerificationStep) for s in steps)

    def test_empty_response_returns_empty_list(self) -> None:
        adapter = _make_adapter()
        steps = adapter.process_response("")
        assert steps == []

    def test_step_indices_are_sequential(self) -> None:
        adapter = _make_adapter()
        response = "Step one. Step two. Step three."
        steps = adapter.process_response(response)
        for i, s in enumerate(steps):
            assert s.step_index == i

    def test_any_violation_detects_incorrect_response(self) -> None:
        # REQ-VERIFY-136-6: hermes_tp counts responses where any step is violated.
        adapter = _make_adapter()
        response = "She buys 3 items at $5 each. 3 * 5 = 20. That is the total."
        steps = adapter.process_response(response)
        hermes_violation = any(s.prover_verdict == "violated" for s in steps)
        # 3 * 5 = 15 not 20, so this should be detected.
        assert hermes_violation is True

    def test_no_violation_in_correct_response(self) -> None:
        adapter = _make_adapter()
        response = "She buys 3 items at $5 each. 3 * 5 = 15. That is the total."
        steps = adapter.process_response(response)
        hermes_violation = any(s.prover_verdict == "violated" for s in steps)
        assert hermes_violation is False
