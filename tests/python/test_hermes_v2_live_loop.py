"""Tests for HermesV2LiveLoop — 100% coverage of hermes_v2_live_loop.py.

Spec: REQ-VERIFY-137, REQ-VERIFY-138,
      SCENARIO-VERIFY-180, SCENARIO-VERIFY-181, SCENARIO-VERIFY-182
"""

import pytest

from carnot.pipeline.symcode_verifier import SymCodeVerifier
from carnot.pipeline.hermes_v2_live_loop import (
    HermesV2GenerationResult,
    HermesV2LiveLoop,
    HermesV2StepResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_loop(llm_caller=None, max_sentences=10):
    """Helper: build a HermesV2LiveLoop with a CI-mode SymCodeVerifier."""
    verifier = SymCodeVerifier(llm_caller=None)
    return HermesV2LiveLoop(
        llm_caller=llm_caller,
        verifier=verifier,
        max_sentences=max_sentences,
    )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-180: CI stub returns empty response
# ---------------------------------------------------------------------------


class TestCIStub:
    """REQ-VERIFY-137-2, SCENARIO-VERIFY-180: llm_caller=None exits on first step."""

    def test_ci_stub_empty_response(self):
        """generate_with_verification returns empty full_response in CI stub mode."""
        loop = _make_loop()
        result = loop.generate_with_verification("What is 2+2?")

        assert isinstance(result, HermesV2GenerationResult)
        assert result.full_response == ""
        assert result.step_results == []
        assert result.any_violation is False
        assert result.n_violations == 0
        assert result.n_hints == 0
        assert result.question == "What is 2+2?"

    def test_ci_stub_generate_step_returns_empty(self):
        """_generate_step returns empty string when llm_caller is None."""
        loop = _make_loop()
        result = loop._generate_step("some context")
        assert result == ""


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-181: hint injected on violation
# ---------------------------------------------------------------------------


class TestViolationHintInjection:
    """REQ-VERIFY-137-3, SCENARIO-VERIFY-181: hint injected when violation detected."""

    def test_hint_injected_on_violation(self):
        """When LLM generates a sentence with arithmetic violation, hint is injected."""
        # 47+28 = 75 in reality; response claims "= 76" — CI regex detects violation.
        # Using explicit N op M format so the CI-mode regex extractor can find it.
        violation_sentence = "We compute 47+28 to get = 76."

        call_count = {"n": 0}

        def _caller(prompt: str) -> str:
            # Return the violation sentence on first call, empty on subsequent calls.
            if call_count["n"] == 0:
                call_count["n"] += 1
                return violation_sentence
            return ""

        loop = _make_loop(llm_caller=_caller, max_sentences=5)
        result = loop.generate_with_verification("How many fruits?")

        assert len(result.step_results) >= 1
        step = result.step_results[0]
        assert step.step_text == violation_sentence
        assert step.violation_detected is True
        assert step.hint_injected is True
        assert step.hint_text == HermesV2LiveLoop.CORRECTION_HINT
        assert result.any_violation is True
        assert result.n_violations >= 1
        assert result.n_hints >= 1

    def test_correction_hint_constant_value(self):
        """CORRECTION_HINT is the expected sentinel string."""
        assert "[Note:" in HermesV2LiveLoop.CORRECTION_HINT


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-182: no hint on correct arithmetic
# ---------------------------------------------------------------------------


class TestNoHintOnCorrectArithmetic:
    """REQ-VERIFY-137-3, SCENARIO-VERIFY-182: no hint when arithmetic is correct."""

    def test_no_hint_on_correct_arithmetic(self):
        """When LLM generates a correct arithmetic sentence, no hint is injected."""
        # 47 + 28 = 75 — correct
        correct_sentence = "There are 47 apples and 28 oranges, total is 75."

        call_count = {"n": 0}

        def _caller(prompt: str) -> str:
            if call_count["n"] == 0:
                call_count["n"] += 1
                return correct_sentence
            return ""

        loop = _make_loop(llm_caller=_caller, max_sentences=5)
        result = loop.generate_with_verification("How many fruits?")

        assert len(result.step_results) >= 1
        step = result.step_results[0]
        assert step.step_text == correct_sentence
        assert step.violation_detected is False
        assert step.hint_injected is False
        assert step.hint_text is None
        assert result.any_violation is False
        assert result.n_violations == 0
        assert result.n_hints == 0


# ---------------------------------------------------------------------------
# Multi-step generation
# ---------------------------------------------------------------------------


class TestMultiStepGeneration:
    """REQ-VERIFY-137-4: loop runs up to max_sentences."""

    def test_loop_stops_at_max_sentences(self):
        """generate_with_verification stops at max_sentences even if LLM keeps generating."""
        responses = [
            "First sentence with no arithmetic here.",
            "Second sentence also has no arithmetic.",
            "Third sentence is still no arithmetic.",
        ]
        call_count = {"n": 0}

        def _caller(prompt: str) -> str:
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(responses):
                return responses[idx]
            return ""

        loop = _make_loop(llm_caller=_caller, max_sentences=2)
        result = loop.generate_with_verification("Tell me something.")

        # max_sentences=2 so we get at most 2 steps.
        assert len(result.step_results) <= 2

    def test_full_response_joins_steps(self):
        """full_response is all step_text strings joined with spaces."""
        sentences = ["First sentence.", "Second sentence."]
        call_count = {"n": 0}

        def _caller(prompt: str) -> str:
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(sentences):
                return sentences[idx]
            return ""

        loop = _make_loop(llm_caller=_caller, max_sentences=5)
        result = loop.generate_with_verification("Tell me.")

        assert "First sentence." in result.full_response
        assert "Second sentence." in result.full_response

    def test_word_limit_cap_stops_loop(self):
        """Loop stops when context exceeds 300 words."""
        # Return a very long sentence that pushes context over the 300-word cap.
        long_sentence = " ".join([f"word{i}" for i in range(250)])

        call_count = {"n": 0}

        def _caller(prompt: str) -> str:
            call_count["n"] += 1
            return long_sentence

        loop = _make_loop(llm_caller=_caller, max_sentences=20)
        result = loop.generate_with_verification("What?")

        # The 300-word cap should have stopped the loop before 20 sentences.
        assert len(result.step_results) < 20


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------


class TestLLMErrorHandling:
    """REQ-VERIFY-137-2: LLM errors return empty string, loop exits gracefully."""

    def test_llm_exception_returns_empty(self):
        """If llm_caller raises, _generate_step returns empty string."""
        def _failing_caller(prompt: str) -> str:
            raise RuntimeError("GPU OOM")

        loop = _make_loop(llm_caller=_failing_caller, max_sentences=5)
        # _generate_step should swallow the exception.
        result = loop._generate_step("some context")
        assert result == ""

    def test_llm_empty_response_terminates_loop(self):
        """If llm_caller returns empty string, loop terminates immediately."""
        def _empty_caller(prompt: str) -> str:
            return ""

        loop = _make_loop(llm_caller=_empty_caller, max_sentences=5)
        result = loop.generate_with_verification("What is 1+1?")
        assert result.step_results == []
        assert result.full_response == ""

    def test_llm_whitespace_only_terminates_loop(self):
        """If llm_caller returns whitespace-only string, loop terminates."""
        def _whitespace_caller(prompt: str) -> str:
            return "   \n  "

        loop = _make_loop(llm_caller=_whitespace_caller, max_sentences=5)
        result = loop.generate_with_verification("What is 1+1?")
        assert result.step_results == []


# ---------------------------------------------------------------------------
# Sentence extraction in _generate_step
# ---------------------------------------------------------------------------


class TestSentenceExtraction:
    """REQ-VERIFY-137-2: first sentence extracted from multi-sentence response."""

    def test_first_sentence_extracted(self):
        """_generate_step extracts the first sentence from a multi-sentence response."""
        def _multi_sentence(prompt: str) -> str:
            return "First sentence. Second sentence. Third."

        loop = _make_loop(llm_caller=_multi_sentence, max_sentences=1)
        sentence = loop._generate_step("Question: x\nAnswer:")
        # Should return just the first sentence.
        assert "First sentence" in sentence
        assert "Second" not in sentence

    def test_newline_split(self):
        """_generate_step handles newline-separated lines."""
        def _newline_response(prompt: str) -> str:
            return "Line one\nLine two"

        loop = _make_loop(llm_caller=_newline_response, max_sentences=1)
        sentence = loop._generate_step("context")
        assert "Line one" in sentence


# ---------------------------------------------------------------------------
# Dataclass field coverage
# ---------------------------------------------------------------------------


class TestDataclassFields:
    """REQ-VERIFY-137-5, REQ-VERIFY-137-6: dataclass fields exist and are correct type."""

    def test_hermes_v2_step_result_fields(self):
        """HermesV2StepResult has all required fields."""
        step = HermesV2StepResult(
            step_index=0,
            step_text="test",
            violation_detected=True,
            detection_score=0.5,
            hint_injected=True,
            hint_text="hint",
        )
        assert step.step_index == 0
        assert step.step_text == "test"
        assert step.violation_detected is True
        assert step.detection_score == 0.5
        assert step.hint_injected is True
        assert step.hint_text == "hint"

    def test_hermes_v2_generation_result_fields(self):
        """HermesV2GenerationResult has all required fields."""
        result = HermesV2GenerationResult(
            question="q",
            full_response="r",
            step_results=[],
            any_violation=False,
            n_violations=0,
            n_hints=0,
        )
        assert result.question == "q"
        assert result.full_response == "r"
        assert result.step_results == []
        assert result.any_violation is False
        assert result.n_violations == 0
        assert result.n_hints == 0

    def test_hermes_v2_step_result_no_hint(self):
        """HermesV2StepResult supports hint_text=None (no hint injected)."""
        step = HermesV2StepResult(
            step_index=1,
            step_text="no arithmetic here",
            violation_detected=False,
            detection_score=0.0,
            hint_injected=False,
            hint_text=None,
        )
        assert step.hint_text is None
        assert step.hint_injected is False


# ---------------------------------------------------------------------------
# Pipeline export
# ---------------------------------------------------------------------------


class TestPipelineExport:
    """REQ-VERIFY-137-7: classes exported from carnot.pipeline."""

    def test_exported_from_pipeline(self):
        """HermesV2LiveLoop, HermesV2GenerationResult, HermesV2StepResult exported."""
        from carnot.pipeline import (
            HermesV2GenerationResult,
            HermesV2LiveLoop,
            HermesV2StepResult,
        )
        assert HermesV2LiveLoop is not None
        assert HermesV2GenerationResult is not None
        assert HermesV2StepResult is not None
