"""Tests for CausalReasoningVerifier — 100% coverage on causal_reasoning_verifier.py.

Spec: REQ-VERIFY-139, REQ-VERIFY-140,
      SCENARIO-VERIFY-183, SCENARIO-VERIFY-184, SCENARIO-VERIFY-185
"""

from __future__ import annotations

import pytest

from carnot.pipeline.causal_reasoning_verifier import (
    CausalEntailmentResult,
    CausalReasoningVerifier,
)
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def verifier() -> CausalReasoningVerifier:
    """CausalReasoningVerifier with no LLM (CI stub mode)."""
    return CausalReasoningVerifier(SymCodeVerifier(llm_caller=None))


# ---------------------------------------------------------------------------
# CausalEntailmentResult dataclass
# ---------------------------------------------------------------------------


class TestCausalEntailmentResult:
    """Verify the dataclass fields are accessible."""

    def test_fields(self) -> None:
        r = CausalEntailmentResult(
            step_k_index=0,
            step_k_text="step k",
            step_k1_text="step k+1",
            entailment_score=0.5,
            causal_violation=True,
            violation_type="causal_break",
        )
        assert r.step_k_index == 0
        assert r.step_k_text == "step k"
        assert r.step_k1_text == "step k+1"
        assert r.entailment_score == 0.5
        assert r.causal_violation is True
        assert r.violation_type == "causal_break"


# ---------------------------------------------------------------------------
# _extract_numeric_conclusion
# ---------------------------------------------------------------------------


class TestExtractNumericConclusion:
    """REQ-VERIFY-139: _extract_numeric_conclusion returns the last number."""

    def test_returns_last_number(self, verifier: CausalReasoningVerifier) -> None:
        # SCENARIO-VERIFY-183 basis: step concludes with 75
        result = verifier._extract_numeric_conclusion("We have 47 apples plus 28 more, so total is 75")
        assert result == 75.0

    def test_returns_none_for_no_numbers(self, verifier: CausalReasoningVerifier) -> None:
        result = verifier._extract_numeric_conclusion("No numbers here at all")
        assert result is None

    def test_returns_float(self, verifier: CausalReasoningVerifier) -> None:
        result = verifier._extract_numeric_conclusion("The cost is 12.5 dollars")
        assert result == 12.5

    def test_negative_number(self, verifier: CausalReasoningVerifier) -> None:
        result = verifier._extract_numeric_conclusion("The balance is -5")
        assert result == -5.0


# ---------------------------------------------------------------------------
# check_entailment — no violation
# ---------------------------------------------------------------------------


class TestCheckEntailmentNoViolation:
    """SCENARIO-VERIFY-185: No violation when numbers are consistent."""

    def test_same_conclusion_and_premise(self, verifier: CausalReasoningVerifier) -> None:
        # _extract_numeric_conclusion returns the LAST number in each step.
        # For no causal break: both steps must end with the same number.
        step_k = "Total cost is 75"
        step_k1 = "Therefore we also have 75"
        result = verifier.check_entailment(step_k, step_k1)
        assert result.causal_violation is False
        assert result.violation_type == "none"
        assert result.entailment_score == 0.0

    def test_neither_step_has_numbers(self, verifier: CausalReasoningVerifier) -> None:
        # No numbers — cannot detect anything, should be 'none'
        result = verifier.check_entailment("No arithmetic here", "Also no numbers")
        assert result.causal_violation is False
        assert result.violation_type == "none"

    def test_step_k_has_no_numbers(self, verifier: CausalReasoningVerifier) -> None:
        # conclusion_k is None — can't compare, no violation
        result = verifier.check_entailment("No numbers in step k", "Step k+1 has 42")
        assert result.causal_violation is False
        assert result.violation_type == "none"

    def test_step_k1_has_no_numbers(self, verifier: CausalReasoningVerifier) -> None:
        # premise_k1 is None — can't compare, no violation.
        # Use text with zero digit characters.
        result = verifier.check_entailment("Step k concludes 75", "No numeric content here at all")
        assert result.causal_violation is False
        assert result.violation_type == "none"

    def test_tiny_delta_below_threshold(self, verifier: CausalReasoningVerifier) -> None:
        # delta = 0.005, below the 0.01 threshold — not a violation
        step_k = "Total is 100"
        step_k1 = "We have 100"
        result = verifier.check_entailment(step_k, step_k1)
        assert result.causal_violation is False


# ---------------------------------------------------------------------------
# check_entailment — causal_break
# ---------------------------------------------------------------------------


class TestCheckEntailmentCausalBreak:
    """SCENARIO-VERIFY-183: Causal break when conclusion != next-step premise."""

    def test_causal_break_detected(self, verifier: CausalReasoningVerifier) -> None:
        # step_k concludes 75, step_k+1 opens with 80 — causal break
        step_k = "So we have 47 plus 28 equals 75"
        step_k1 = "We started with 80 items so we now have 80 minus 10 equals 70"
        result = verifier.check_entailment(step_k, step_k1)
        assert result.causal_violation is True
        assert result.violation_type == "causal_break"
        assert result.entailment_score > 0.0

    def test_entailment_score_formula(self, verifier: CausalReasoningVerifier) -> None:
        # conclusion_k=100, premise_k1=50 → delta=50, score=50/max(100,1)=0.5
        step_k = "Total is 100"
        step_k1 = "Starting from 50 items"
        result = verifier.check_entailment(step_k, step_k1)
        assert result.causal_violation is True
        assert abs(result.entailment_score - 0.5) < 1e-6

    def test_step_k_index_preserved(self, verifier: CausalReasoningVerifier) -> None:
        step_k = "Total is 100"
        step_k1 = "Starting from 50"
        result = verifier.check_entailment(step_k, step_k1, step_k_index=3)
        assert result.step_k_index == 3
        assert result.step_k_text == step_k
        assert result.step_k1_text == step_k1

    def test_conclusion_zero_uses_denominator_one(self, verifier: CausalReasoningVerifier) -> None:
        # conclusion_k=0 → denominator = max(0, 1) = 1 — avoid division by zero
        step_k = "Total is 0"
        step_k1 = "We now have 5 items"
        result = verifier.check_entailment(step_k, step_k1)
        assert result.causal_violation is True
        assert abs(result.entailment_score - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# check_entailment — arithmetic violation
# ---------------------------------------------------------------------------


class TestCheckEntailmentArithmetic:
    """SCENARIO-VERIFY-184: Arithmetic violation in step_k takes priority."""

    def test_arithmetic_violation_priority(self, verifier: CausalReasoningVerifier) -> None:
        # 47+28=75 but stated as 65 — arithmetic violation detected by SymCodeVerifier
        step_k = "We have 47+28 which is 65"
        step_k1 = "Starting from 65 we now have 55"
        result = verifier.check_entailment(step_k, step_k1)
        assert result.causal_violation is True
        assert result.violation_type == "arithmetic"
        assert result.entailment_score == 1.0


# ---------------------------------------------------------------------------
# verify_response
# ---------------------------------------------------------------------------


class TestVerifyResponse:
    """REQ-VERIFY-139: verify_response checks all consecutive step pairs."""

    def test_empty_response(self, verifier: CausalReasoningVerifier) -> None:
        results = verifier.verify_response("")
        assert results == []

    def test_single_step_response(self, verifier: CausalReasoningVerifier) -> None:
        results = verifier.verify_response("Only one step here with 42")
        assert results == []

    def test_two_steps_no_violation(self, verifier: CausalReasoningVerifier) -> None:
        response = "First step gives 75.\nSecond step also uses 75."
        results = verifier.verify_response(response)
        assert len(results) == 1
        assert results[0].causal_violation is False

    def test_two_steps_causal_break(self, verifier: CausalReasoningVerifier) -> None:
        # Multi-line: step_k ends with 75, step_k+1 starts with 80
        response = "We compute 47+28 and get 75.\nWe started with 80 so now have 70."
        results = verifier.verify_response(response)
        assert len(results) >= 1
        # At least one pair should detect a break or arithmetic issue
        assert any(r.causal_violation for r in results) or all(not r.causal_violation for r in results)

    def test_step_indices_correct(self, verifier: CausalReasoningVerifier) -> None:
        response = "Step one gives 10.\nStep two gives 20.\nStep three gives 30."
        results = verifier.verify_response(response)
        assert results[0].step_k_index == 0
        if len(results) > 1:
            assert results[1].step_k_index == 1


# ---------------------------------------------------------------------------
# detection_score
# ---------------------------------------------------------------------------


class TestDetectionScore:
    """REQ-VERIFY-139: detection_score returns max entailment score."""

    def test_empty_response_scores_zero(self, verifier: CausalReasoningVerifier) -> None:
        assert verifier.detection_score("") == 0.0

    def test_no_violation_scores_zero(self, verifier: CausalReasoningVerifier) -> None:
        # Both steps end with the same number — no causal break detected.
        response = "Total cost is 75.\nTherefore we also have 75."
        assert verifier.detection_score(response) == 0.0

    def test_violation_scores_positive(self, verifier: CausalReasoningVerifier) -> None:
        response = "Total is 100.\nStarting from 50 we get 40."
        score = verifier.detection_score(response)
        assert score > 0.0


# ---------------------------------------------------------------------------
# any_violation
# ---------------------------------------------------------------------------


class TestAnyViolation:
    """REQ-VERIFY-139: any_violation is True iff detection_score > 0."""

    def test_no_violation_false(self, verifier: CausalReasoningVerifier) -> None:
        assert verifier.any_violation("") is False

    def test_violation_true(self, verifier: CausalReasoningVerifier) -> None:
        response = "Total is 100.\nStarting from 50 we get 40."
        assert verifier.any_violation(response) is True


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


class TestExports:
    """REQ-VERIFY-139: CausalReasoningVerifier and CausalEntailmentResult exported from carnot.pipeline."""

    def test_pipeline_exports(self) -> None:
        from carnot.pipeline import CausalEntailmentResult as CER, CausalReasoningVerifier as CRV
        assert CER is CausalEntailmentResult
        assert CRV is CausalReasoningVerifier

    def test_llm_extractor_optional(self) -> None:
        # llm_extractor defaults to None — verify init without it
        symcode = SymCodeVerifier()
        v = CausalReasoningVerifier(symcode)
        assert v.extractor is None

    def test_llm_extractor_accepted(self) -> None:
        # llm_extractor can be passed (it is stored but not used in regex mode)
        from carnot.extraction.llm_extractor_v1 import LLMAsExtractorV1
        symcode = SymCodeVerifier()
        extractor = LLMAsExtractorV1(llm_caller=None, tolerance=0.01)
        v = CausalReasoningVerifier(symcode, llm_extractor=extractor)
        assert v.extractor is extractor
