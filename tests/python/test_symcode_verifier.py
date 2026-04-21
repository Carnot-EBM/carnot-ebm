"""Tests for SymCodeVerifier — 100% coverage on symcode_verifier.py.

Spec: REQ-VERIFY-122, REQ-VERIFY-123,
      SCENARIO-VERIFY-160, SCENARIO-VERIFY-161, SCENARIO-VERIFY-162
"""

from __future__ import annotations

import pytest

from carnot.pipeline.symcode_verifier import CoTStep, SymCodeVerifier


# ---------------------------------------------------------------------------
# segment_steps
# ---------------------------------------------------------------------------


class TestSegmentSteps:
    """REQ-VERIFY-122-1: segment_steps splits response by sentence boundaries."""

    def test_splits_on_period_space(self) -> None:
        # SCENARIO-VERIFY-160 setup: a two-sentence response
        v = SymCodeVerifier()
        steps = v.segment_steps("First step. Second step.")
        assert len(steps) == 2
        assert steps[0] == "First step"
        assert steps[1] == "Second step."

    def test_splits_on_newline(self) -> None:
        v = SymCodeVerifier()
        steps = v.segment_steps("Step one\nStep two\nStep three")
        assert len(steps) == 3

    def test_empty_response(self) -> None:
        v = SymCodeVerifier()
        assert v.segment_steps("") == []

    def test_whitespace_only_lines_excluded(self) -> None:
        v = SymCodeVerifier()
        steps = v.segment_steps("  \n  \nReal step\n  ")
        assert steps == ["Real step"]

    def test_mixed_separators(self) -> None:
        v = SymCodeVerifier()
        steps = v.segment_steps("A. B\nC. D")
        # "A. B" → ["A", "B"], then "C. D" → ["C", "D"]
        assert len(steps) == 4


# ---------------------------------------------------------------------------
# extract_code_for_step — CI regex mode
# ---------------------------------------------------------------------------


class TestExtractCodeForStepCI:
    """REQ-VERIFY-123-3: CI mode (llm_caller=None) uses regex fallback."""

    def test_finds_multiplication(self) -> None:
        # SCENARIO-VERIFY-160
        v = SymCodeVerifier()
        code = v.extract_code_for_step("3 * 4 = 12")
        assert code == "3*4"

    def test_finds_addition(self) -> None:
        v = SymCodeVerifier()
        code = v.extract_code_for_step("47 + 28 = 75")
        assert code == "47+28"

    def test_finds_subtraction(self) -> None:
        v = SymCodeVerifier()
        code = v.extract_code_for_step("100 - 35 = 65")
        assert code == "100-35"

    def test_finds_division(self) -> None:
        v = SymCodeVerifier()
        code = v.extract_code_for_step("100 / 4 = 25")
        assert code == "100/4"

    def test_no_arithmetic_returns_none(self) -> None:
        # SCENARIO-VERIFY-162
        v = SymCodeVerifier()
        assert v.extract_code_for_step("The answer is therefore obvious.") is None

    def test_pure_text_returns_none(self) -> None:
        v = SymCodeVerifier()
        assert v.extract_code_for_step("We need to find the total cost.") is None


# ---------------------------------------------------------------------------
# extract_code_for_step — LLM mode
# ---------------------------------------------------------------------------


class TestExtractCodeForStepLLM:
    """REQ-VERIFY-122-2: LLM mode passes step to llm_caller and returns expression."""

    def test_llm_returns_expression(self) -> None:
        v = SymCodeVerifier(llm_caller=lambda _: "47+28")
        code = v.extract_code_for_step("47 plus 28 gives 75")
        assert code == "47+28"

    def test_llm_returns_none_literal(self) -> None:
        v = SymCodeVerifier(llm_caller=lambda _: "None")
        assert v.extract_code_for_step("No arithmetic here.") is None

    def test_llm_returns_empty_string(self) -> None:
        v = SymCodeVerifier(llm_caller=lambda _: "")
        assert v.extract_code_for_step("No arithmetic here.") is None

    def test_llm_returns_null(self) -> None:
        v = SymCodeVerifier(llm_caller=lambda _: "null")
        assert v.extract_code_for_step("step") is None

    def test_llm_strips_markdown_fences(self) -> None:
        v = SymCodeVerifier(llm_caller=lambda _: "```python\n3*4\n```")
        code = v.extract_code_for_step("3 times 4 gives 12")
        assert code == "3*4"

    def test_llm_exception_returns_none(self) -> None:
        def bad_caller(_: str) -> str:
            raise RuntimeError("LLM offline")

        v = SymCodeVerifier(llm_caller=bad_caller)
        # Should not raise; should return None gracefully.
        assert v.extract_code_for_step("3 * 4 = 12") is None


# ---------------------------------------------------------------------------
# _extract_stated_result (via verify_step)
# ---------------------------------------------------------------------------


class TestExtractStatedResult:
    """REQ-VERIFY-122-3: stated_result is extracted from the step text."""

    def test_equals_pattern(self) -> None:
        v = SymCodeVerifier()
        step = v.verify_step("3 * 4 = 12")
        assert step.stated_result == 12.0

    def test_is_pattern(self) -> None:
        v = SymCodeVerifier()
        step = v.verify_step("3 * 4 is 12")
        assert step.stated_result == 12.0

    def test_gives_pattern(self) -> None:
        v = SymCodeVerifier()
        step = v.verify_step("3 * 4 gives 12")
        assert step.stated_result == 12.0

    def test_fallback_last_number(self) -> None:
        # When step has no N op M pattern the CI regex returns None, so the
        # early-return path fires and stated_result is None per spec.
        v = SymCodeVerifier()
        step = v.verify_step("3 times 4, so 12")
        # CI regex can't extract code → early return → stated_result=None
        assert step.stated_result is None

    def test_no_numbers(self) -> None:
        v = SymCodeVerifier()
        step = v.verify_step("The answer is therefore obvious.")
        assert step.stated_result is None


# ---------------------------------------------------------------------------
# verify_step — SCENARIO-VERIFY-160, 161, 162
# ---------------------------------------------------------------------------


class TestVerifyStep:
    """REQ-VERIFY-122-3: verify_step returns CoTStep with correct fields."""

    def test_correct_arithmetic_no_violation(self) -> None:
        # SCENARIO-VERIFY-160: 3*4=12 is correct
        v = SymCodeVerifier()
        result = v.verify_step("3 * 4 = 12")
        assert isinstance(result, CoTStep)
        assert result.violation_detected is False
        assert result.generated_code == "3*4"
        assert result.executed_result == pytest.approx(12.0)
        assert result.stated_result == pytest.approx(12.0)

    def test_incorrect_arithmetic_violation(self) -> None:
        # SCENARIO-VERIFY-161: 3*4=13 is wrong
        v = SymCodeVerifier()
        result = v.verify_step("3 * 4 = 13")
        assert result.violation_detected is True
        assert result.executed_result == pytest.approx(12.0)
        assert result.stated_result == pytest.approx(13.0)

    def test_no_arithmetic_no_violation(self) -> None:
        # SCENARIO-VERIFY-162
        v = SymCodeVerifier()
        result = v.verify_step("The answer is therefore obvious.")
        assert result.violation_detected is False
        assert result.generated_code is None
        assert result.executed_result is None
        assert result.stated_result is None

    def test_step_index_propagated(self) -> None:
        v = SymCodeVerifier()
        result = v.verify_step("3 * 4 = 12", step_index=5)
        assert result.step_index == 5

    def test_step_text_preserved(self) -> None:
        v = SymCodeVerifier()
        text = "3 * 4 = 12"
        result = v.verify_step(text)
        assert result.text == text

    def test_unevaluable_code_no_violation(self) -> None:
        # If the extracted code can't be evaled, no violation should be reported.
        v = SymCodeVerifier(llm_caller=lambda _: "x + y")
        result = v.verify_step("some step with x + y = 5")
        # safe_eval("x + y") returns None → no violation possible
        assert result.violation_detected is False
        assert result.executed_result is None

    def test_none_code_no_violation(self) -> None:
        v = SymCodeVerifier(llm_caller=lambda _: "None")
        result = v.verify_step("some step")
        assert result.violation_detected is False
        assert result.generated_code is None


# ---------------------------------------------------------------------------
# verify_response
# ---------------------------------------------------------------------------


class TestVerifyResponse:
    """REQ-VERIFY-122-4: verify_response returns one CoTStep per segmented step."""

    def test_multiple_steps(self) -> None:
        v = SymCodeVerifier()
        response = "3 * 4 = 12. 5 + 6 = 11."
        results = v.verify_response(response)
        assert len(results) >= 2
        # All CoTStep instances
        for r in results:
            assert isinstance(r, CoTStep)

    def test_step_indices_sequential(self) -> None:
        v = SymCodeVerifier()
        response = "Step one.\nStep two.\nStep three."
        results = v.verify_response(response)
        for idx, r in enumerate(results):
            assert r.step_index == idx

    def test_empty_response(self) -> None:
        v = SymCodeVerifier()
        assert v.verify_response("") == []


# ---------------------------------------------------------------------------
# detection_score
# ---------------------------------------------------------------------------


class TestDetectionScore:
    """REQ-VERIFY-122-5: detection_score returns float in [0.0, 1.0]."""

    def test_no_violations_score_zero(self) -> None:
        v = SymCodeVerifier()
        score = v.detection_score("3 * 4 = 12.")
        assert score == pytest.approx(0.0)

    def test_violation_score_nonzero(self) -> None:
        # SCENARIO-VERIFY-161: at least one step has a violation
        v = SymCodeVerifier()
        score = v.detection_score("3 * 4 = 13.")
        assert score > 0.0

    def test_score_in_range(self) -> None:
        v = SymCodeVerifier()
        score = v.detection_score("3 * 4 = 12. 2 + 2 = 5.")
        assert 0.0 <= score <= 1.0

    def test_empty_response_score_zero(self) -> None:
        v = SymCodeVerifier()
        assert v.detection_score("") == pytest.approx(0.0)

    def test_all_violations_score_one(self) -> None:
        # Single step with violation
        v = SymCodeVerifier()
        # "10 * 2 = 30" is wrong (correct=20); only step → score=1.0
        score = v.detection_score("10 * 2 = 30.")
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CoTStep dataclass
# ---------------------------------------------------------------------------


class TestCoTStepDataclass:
    """Verify CoTStep fields are accessible and typed correctly."""

    def test_fields(self) -> None:
        step = CoTStep(
            text="foo",
            step_index=0,
            generated_code=None,
            executed_result=None,
            stated_result=None,
            violation_detected=False,
        )
        assert step.text == "foo"
        assert step.step_index == 0
        assert step.generated_code is None
        assert step.executed_result is None
        assert step.stated_result is None
        assert step.violation_detected is False


# ---------------------------------------------------------------------------
# Export from carnot.pipeline
# ---------------------------------------------------------------------------


class TestExports:
    """REQ-VERIFY-122: SymCodeVerifier exported from carnot.pipeline."""

    def test_symcode_verifier_importable(self) -> None:
        from carnot.pipeline import SymCodeVerifier as SV  # noqa: PLC0415

        assert SV is SymCodeVerifier

    def test_cot_step_importable_from_module(self) -> None:
        from carnot.pipeline.symcode_verifier import CoTStep as CS  # noqa: PLC0415

        assert CS is CoTStep
