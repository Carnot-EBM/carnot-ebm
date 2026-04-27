"""Tests for python/carnot/pipeline/thinkprm_verifier.py.

All tests trace to REQ-VERIFY-098 or SCENARIO-VERIFY-130 as required by
the spec-anchored development mandate (CLAUDE.md).

Coverage: 100% of thinkprm_verifier.py public interface and internal helpers.
"""

from __future__ import annotations

import time

import pytest

from python.carnot.pipeline.thinkprm_verifier import (
    ThinkPRMResult,
    ThinkPRMVerifier,
    _build_step_prompt,
    _parse_step_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_caller(response: str):
    """Return a stub llm_caller that always returns the given response string."""

    def caller(prompt: str) -> str:  # noqa: ARG001
        return response

    return caller


def _arith_caller(prompt: str) -> str:
    """Minimal arithmetic-checking stub used in integration-level tests.

    Parses 'X + Y = Z' from the prompt and emits VERDICT: CORRECT/INCORRECT.
    Mirrors the CI stub in experiment_945 so tests verify the same logic path.
    """
    import re

    pattern = re.compile(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)")
    m = pattern.search(prompt)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a + b == c:
            return "Step 1: claim.\nStep 2: check.\nVERDICT: CORRECT"
        return "Step 1: claim.\nStep 2: check.\nVERDICT: INCORRECT"
    return "Step 1: ambiguous.\nStep 2: cannot verify."


# ---------------------------------------------------------------------------
# ThinkPRMResult dataclass
# ---------------------------------------------------------------------------


class TestThinkPRMResult:
    """REQ-VERIFY-098: ThinkPRMResult stores all required fields."""

    def test_fields_accessible(self):
        """REQ-VERIFY-098: all fields of ThinkPRMResult are accessible."""
        r = ThinkPRMResult(
            step_text="3 + 4 = 7",
            verdict="correct",
            confidence=0.95,
            reasoning_steps="Some reasoning",
            latency_ms=1.5,
        )
        assert r.step_text == "3 + 4 = 7"
        assert r.verdict == "correct"
        assert r.confidence == 0.95
        assert r.reasoning_steps == "Some reasoning"
        assert r.latency_ms == 1.5

    def test_verdict_values(self):
        """REQ-VERIFY-098: verdict accepts 'correct', 'incorrect', 'uncertain'."""
        for v in ("correct", "incorrect", "uncertain"):
            r = ThinkPRMResult(
                step_text="x",
                verdict=v,
                confidence=0.5,  # type: ignore[arg-type]
                reasoning_steps="",
                latency_ms=0.0,
            )
            assert r.verdict == v


# ---------------------------------------------------------------------------
# _build_step_prompt
# ---------------------------------------------------------------------------


class TestBuildStepPrompt:
    """REQ-VERIFY-098: prompt builder produces correct template."""

    def test_step_in_prompt(self):
        """REQ-VERIFY-098: step text appears in the generated prompt."""
        prompt = _build_step_prompt("47 + 28 = 75", "")
        assert "47 + 28 = 75" in prompt

    def test_verdict_instruction_in_prompt(self):
        """REQ-VERIFY-098: VERDICT: CORRECT/INCORRECT instructions are present."""
        prompt = _build_step_prompt("1 + 1 = 2", "")
        assert "VERDICT: CORRECT" in prompt
        assert "VERDICT: INCORRECT" in prompt

    def test_three_step_structure(self):
        """REQ-VERIFY-098: prompt contains Step 1, Step 2, Step 3 headers."""
        prompt = _build_step_prompt("2 + 2 = 4", "")
        assert "Step 1" in prompt
        assert "Step 2" in prompt
        assert "Step 3" in prompt

    def test_context_included_when_provided(self):
        """REQ-VERIFY-098: context string appears in prompt when non-empty."""
        prompt = _build_step_prompt("5 + 3 = 8", "Prior step: x = 5.")
        assert "Prior step: x = 5." in prompt

    def test_no_context_block_when_empty(self):
        """REQ-VERIFY-098: context block absent when context=''."""
        prompt = _build_step_prompt("5 + 3 = 8", "")
        assert "Context (preceding steps)" not in prompt


# ---------------------------------------------------------------------------
# _parse_step_output
# ---------------------------------------------------------------------------


class TestParseStepOutput:
    """REQ-VERIFY-098, SCENARIO-VERIFY-130: output parser handles all cases."""

    def test_correct_verdict(self):
        """SCENARIO-VERIFY-130: VERDICT: CORRECT parsed as 'correct'."""
        output = "Step 1: claim.\nStep 2: verified.\nVERDICT: CORRECT"
        verdict, confidence, reasoning = _parse_step_output(output)
        assert verdict == "correct"
        assert confidence == 0.95

    def test_incorrect_verdict(self):
        """SCENARIO-VERIFY-130: VERDICT: INCORRECT parsed as 'incorrect'."""
        output = "Step 1: claim.\nStep 2: wrong.\nVERDICT: INCORRECT"
        verdict, confidence, reasoning = _parse_step_output(output)
        assert verdict == "incorrect"
        assert confidence == 0.95

    def test_case_insensitive_correct(self):
        """REQ-VERIFY-098: parser is case-insensitive for VERDICT keyword."""
        verdict, confidence, _ = _parse_step_output("verdict: correct")
        assert verdict == "correct"

    def test_case_insensitive_incorrect(self):
        """REQ-VERIFY-098: parser is case-insensitive for VERDICT keyword."""
        verdict, confidence, _ = _parse_step_output("verdict: incorrect")
        assert verdict == "incorrect"

    def test_no_verdict_returns_uncertain(self):
        """REQ-VERIFY-098: no VERDICT line → 'uncertain', confidence=0.5."""
        output = "This step seems fine but I cannot tell."
        verdict, confidence, reasoning = _parse_step_output(output)
        assert verdict == "uncertain"
        assert confidence == 0.5

    def test_multiple_verdicts_uses_last(self):
        """REQ-VERIFY-098: when multiple VERDICT lines present, last one wins."""
        output = "VERDICT: CORRECT\nActually...\nVERDICT: INCORRECT"
        verdict, confidence, _ = _parse_step_output(output)
        assert verdict == "incorrect"

    def test_reasoning_returned(self):
        """REQ-VERIFY-098: reasoning text (raw output) is returned."""
        output = "Some reasoning.\nVERDICT: CORRECT"
        _, _, reasoning = _parse_step_output(output)
        assert "Some reasoning" in reasoning

    def test_uncertain_reasoning_is_full_output(self):
        """REQ-VERIFY-098: when uncertain, reasoning is the full output string."""
        output = "No verdict here."
        _, _, reasoning = _parse_step_output(output)
        assert reasoning == output


# ---------------------------------------------------------------------------
# ThinkPRMVerifier.verify_step — CI stub mode
# ---------------------------------------------------------------------------


class TestThinkPRMVerifierCIStub:
    """REQ-VERIFY-098: CI stub (llm_caller=None) returns deterministic uncertain."""

    def test_stub_returns_uncertain(self):
        """REQ-VERIFY-098: stub mode returns verdict='uncertain'."""
        v = ThinkPRMVerifier()
        result = v.verify_step("47 + 28 = 75")
        assert result.verdict == "uncertain"

    def test_stub_confidence_is_half(self):
        """REQ-VERIFY-098: stub confidence is 0.5 (maximum entropy)."""
        v = ThinkPRMVerifier()
        result = v.verify_step("47 + 28 = 75")
        assert result.confidence == 0.5

    def test_stub_reasoning_empty(self):
        """REQ-VERIFY-098: stub reasoning_steps is empty string."""
        v = ThinkPRMVerifier()
        result = v.verify_step("1 + 1 = 2")
        assert result.reasoning_steps == ""

    def test_stub_step_text_preserved(self):
        """REQ-VERIFY-098: step_text is preserved verbatim in result."""
        v = ThinkPRMVerifier()
        step = "99 + 1 = 100"
        result = v.verify_step(step)
        assert result.step_text == step

    def test_stub_latency_non_negative(self):
        """REQ-VERIFY-098: latency_ms is non-negative float."""
        v = ThinkPRMVerifier()
        result = v.verify_step("3 + 3 = 6")
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0

    def test_stub_is_thinkprm_result(self):
        """REQ-VERIFY-098: return type is ThinkPRMResult."""
        v = ThinkPRMVerifier()
        result = v.verify_step("2 + 2 = 4")
        assert isinstance(result, ThinkPRMResult)

    def test_stub_default_confidence_threshold(self):
        """REQ-VERIFY-098: default confidence_threshold is 0.8."""
        v = ThinkPRMVerifier()
        assert v.confidence_threshold == 0.8

    def test_stub_custom_confidence_threshold(self):
        """REQ-VERIFY-098: custom confidence_threshold is stored."""
        v = ThinkPRMVerifier(confidence_threshold=0.6)
        assert v.confidence_threshold == 0.6


# ---------------------------------------------------------------------------
# ThinkPRMVerifier.verify_step — live LLM path
# ---------------------------------------------------------------------------


class TestThinkPRMVerifierLivePath:
    """REQ-VERIFY-098, SCENARIO-VERIFY-130: live LLM caller path."""

    def test_correct_step_classified(self):
        """SCENARIO-VERIFY-130: correct arithmetic step → verdict='correct'."""
        v = ThinkPRMVerifier(llm_caller=_make_caller("VERDICT: CORRECT"))
        result = v.verify_step("3 + 4 = 7")
        assert result.verdict == "correct"
        assert result.confidence == 0.95

    def test_incorrect_step_classified(self):
        """SCENARIO-VERIFY-130: incorrect arithmetic step → verdict='incorrect'."""
        v = ThinkPRMVerifier(llm_caller=_make_caller("VERDICT: INCORRECT"))
        result = v.verify_step("3 + 4 = 8")
        assert result.verdict == "incorrect"
        assert result.confidence == 0.95

    def test_ambiguous_step_returns_uncertain(self):
        """SCENARIO-VERIFY-130: LLM output with no VERDICT → 'uncertain'."""
        v = ThinkPRMVerifier(llm_caller=_make_caller("Cannot determine correctness."))
        result = v.verify_step("The result is approximately 100.")
        assert result.verdict == "uncertain"
        assert result.confidence == 0.5

    def test_reasoning_captured(self):
        """REQ-VERIFY-098: reasoning_steps captures the LLM's raw output."""
        output = "Step 1: Extract.\nStep 2: Check.\nVERDICT: CORRECT"
        v = ThinkPRMVerifier(llm_caller=_make_caller(output))
        result = v.verify_step("5 + 5 = 10")
        assert "Step 1" in result.reasoning_steps

    def test_context_passed_to_prompt(self):
        """REQ-VERIFY-098: context string is included in the prompt sent to LLM."""
        received: list[str] = []

        def capturing_caller(prompt: str) -> str:
            received.append(prompt)
            return "VERDICT: CORRECT"

        v = ThinkPRMVerifier(llm_caller=capturing_caller)
        v.verify_step("5 + 3 = 8", context="x was defined as 5.")
        assert received, "LLM caller was never called"
        assert "x was defined as 5." in received[0]

    def test_latency_measured(self):
        """REQ-VERIFY-098: latency_ms reflects real elapsed time (>= 0)."""

        def slow_caller(prompt: str) -> str:  # noqa: ARG001
            time.sleep(0.01)
            return "VERDICT: CORRECT"

        v = ThinkPRMVerifier(llm_caller=slow_caller)
        result = v.verify_step("1 + 1 = 2")
        assert result.latency_ms >= 10.0  # at least 10 ms for 10 ms sleep

    def test_arithmetic_stub_correct(self):
        """SCENARIO-VERIFY-130: arithmetic stub returns correct for valid addition."""
        v = ThinkPRMVerifier(llm_caller=_arith_caller)
        result = v.verify_step("10 + 5 = 15")
        assert result.verdict == "correct"

    def test_arithmetic_stub_incorrect(self):
        """SCENARIO-VERIFY-130: arithmetic stub returns incorrect for wrong addition."""
        v = ThinkPRMVerifier(llm_caller=_arith_caller)
        result = v.verify_step("10 + 5 = 16")
        assert result.verdict == "incorrect"

    def test_arithmetic_stub_uncertain_for_approximation(self):
        """SCENARIO-VERIFY-130: arithmetic stub returns uncertain for approximation."""
        v = ThinkPRMVerifier(llm_caller=_arith_caller)
        result = v.verify_step("The result is approximately 100.")
        assert result.verdict == "uncertain"


# ---------------------------------------------------------------------------
# ThinkPRMVerifier.batch_verify
# ---------------------------------------------------------------------------


class TestBatchVerify:
    """REQ-VERIFY-098: batch_verify returns results in input order."""

    def test_batch_length_matches_input(self):
        """REQ-VERIFY-098: batch_verify returns one result per input step."""
        v = ThinkPRMVerifier()
        steps = ["1 + 1 = 2", "2 + 2 = 4", "3 + 3 = 6"]
        results = v.batch_verify(steps)
        assert len(results) == 3

    def test_batch_order_preserved(self):
        """REQ-VERIFY-098: results are in the same order as input steps."""
        v = ThinkPRMVerifier()
        steps = ["step_a", "step_b", "step_c"]
        results = v.batch_verify(steps)
        for step, result in zip(steps, results):
            assert result.step_text == step

    def test_batch_empty_list(self):
        """REQ-VERIFY-098: batch_verify with empty list returns empty list."""
        v = ThinkPRMVerifier()
        results = v.batch_verify([])
        assert results == []

    def test_batch_with_contexts(self):
        """REQ-VERIFY-098: contexts list is passed per-step."""
        received_prompts: list[str] = []

        def capturing_caller(prompt: str) -> str:
            received_prompts.append(prompt)
            return "VERDICT: CORRECT"

        v = ThinkPRMVerifier(llm_caller=capturing_caller)
        v.batch_verify(["3 + 4 = 7"], contexts=["ctx_for_step"])
        assert "ctx_for_step" in received_prompts[0]

    def test_batch_missing_contexts_default_empty(self):
        """REQ-VERIFY-098: if contexts shorter than steps, missing ones default to ''."""
        received_prompts: list[str] = []

        def capturing_caller(prompt: str) -> str:
            received_prompts.append(prompt)
            return "VERDICT: CORRECT"

        v = ThinkPRMVerifier(llm_caller=capturing_caller)
        v.batch_verify(["a", "b"], contexts=["ctx_a"])
        # Second prompt should not contain context block header
        assert "Context (preceding steps)" not in received_prompts[1]

    def test_batch_returns_thinkprm_results(self):
        """REQ-VERIFY-098: each element in batch result is a ThinkPRMResult."""
        v = ThinkPRMVerifier()
        results = v.batch_verify(["x", "y"])
        for r in results:
            assert isinstance(r, ThinkPRMResult)

    def test_batch_none_contexts_ok(self):
        """REQ-VERIFY-098: passing contexts=None is equivalent to no contexts."""
        v = ThinkPRMVerifier()
        results = v.batch_verify(["1 + 1 = 2"], contexts=None)
        assert len(results) == 1
