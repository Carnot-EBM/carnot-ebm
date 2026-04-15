"""Tests for carnot.pipeline.llm_z3_formalizer.

Covers Z3FormalizationResult, LLMz3Formalizer, build_z3_formalization_prompt,
parse_z3_snippet, and _exec_z3_snippet at 100% branch coverage.

Spec: REQ-EXTRACT-019, REQ-EXTRACT-020,
      SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-040, SCENARIO-EXTRACT-041
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from carnot.pipeline.llm_z3_formalizer import (
    LLMz3Formalizer,
    Z3FormalizationResult,
    _CI_STUB_Z3_CODE,
    _exec_z3_snippet,
    build_z3_formalization_prompt,
    parse_z3_snippet,
)


# ---------------------------------------------------------------------------
# Z3FormalizationResult dataclass
# ---------------------------------------------------------------------------


class TestZ3FormalizationResult:
    """REQ-EXTRACT-019: Z3FormalizationResult dataclass contracts.

    Spec: SCENARIO-EXTRACT-039
    """

    def test_is_sat_true_when_z3_result_sat(self) -> None:
        """SCENARIO-EXTRACT-039: z3_result='sat' → is_sat True."""
        r = Z3FormalizationResult(
            z3_code="import z3",
            z3_result="sat",
            n_assertions=1,
            formalization_mode="llm",
            source_response_length=10,
        )
        assert r.is_sat is True

    def test_is_sat_false_when_z3_result_unsat(self) -> None:
        """z3_result='unsat' → is_sat False."""
        r = Z3FormalizationResult(
            z3_code="import z3",
            z3_result="unsat",
            n_assertions=2,
            formalization_mode="llm",
            source_response_length=20,
        )
        assert r.is_sat is False

    def test_is_sat_false_when_unknown(self) -> None:
        """z3_result='unknown' → is_sat False."""
        r = Z3FormalizationResult(
            z3_code="",
            z3_result="unknown",
            n_assertions=0,
            formalization_mode="ci_stub",
            source_response_length=0,
        )
        assert r.is_sat is False

    def test_is_sat_false_when_error(self) -> None:
        """z3_result='error' → is_sat False."""
        r = Z3FormalizationResult(
            z3_code="bad code",
            z3_result="error",
            n_assertions=0,
            formalization_mode="llm",
            source_response_length=8,
            error_message="SyntaxError",
        )
        assert r.is_sat is False
        assert r.error_message == "SyntaxError"

    def test_error_message_defaults_none(self) -> None:
        """error_message defaults to None when not provided."""
        r = Z3FormalizationResult(
            z3_code="x",
            z3_result="sat",
            n_assertions=0,
            formalization_mode="ci_stub",
            source_response_length=5,
        )
        assert r.error_message is None

    def test_all_fields_accessible(self) -> None:
        """SCENARIO-EXTRACT-039: all declared fields are present and correct."""
        r = Z3FormalizationResult(
            z3_code="code",
            z3_result="sat",
            n_assertions=3,
            formalization_mode="llm",
            source_response_length=42,
        )
        assert r.z3_code == "code"
        assert r.z3_result == "sat"
        assert r.n_assertions == 3
        assert r.formalization_mode == "llm"
        assert r.source_response_length == 42
        assert r.is_sat is True


# ---------------------------------------------------------------------------
# build_z3_formalization_prompt
# ---------------------------------------------------------------------------


class TestBuildZ3FormalizationPrompt:
    """REQ-EXTRACT-019: build_z3_formalization_prompt constructs the correct prompt."""

    def test_returns_string(self) -> None:
        """build_z3_formalization_prompt returns a string."""
        result = build_z3_formalization_prompt("What is 2+2?", "2 + 2 = 4")
        assert isinstance(result, str)

    def test_contains_question(self) -> None:
        """The prompt embeds the original question text."""
        q = "unique-question-marker-abc"
        prompt = build_z3_formalization_prompt(q, "some response")
        assert q in prompt

    def test_contains_response(self) -> None:
        """The prompt embeds the response text."""
        r = "unique-response-marker-xyz"
        prompt = build_z3_formalization_prompt("question", r)
        assert r in prompt

    def test_instructs_z3_code_output(self) -> None:
        """The prompt instructs the LLM to output only z3 Python code."""
        prompt = build_z3_formalization_prompt("q", "r")
        lower = prompt.lower()
        assert "z3" in lower
        assert "python" in lower

    def test_empty_inputs_do_not_crash(self) -> None:
        """Empty question and response produce a valid prompt string."""
        prompt = build_z3_formalization_prompt("", "")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_mentions_code_block_format(self) -> None:
        """The prompt mentions the expected ```python ... ``` output format."""
        prompt = build_z3_formalization_prompt("q", "r")
        assert "```python" in prompt or "code block" in prompt.lower()


# ---------------------------------------------------------------------------
# parse_z3_snippet
# ---------------------------------------------------------------------------


class TestParseZ3Snippet:
    """REQ-EXTRACT-019: parse_z3_snippet extracts Python code blocks correctly."""

    def test_extracts_python_block(self) -> None:
        """Standard ```python ... ``` fences are parsed correctly."""
        output = "Here is the code:\n```python\nimport z3\nprint('sat')\n```"
        snippet = parse_z3_snippet(output)
        assert "import z3" in snippet
        assert "print" in snippet

    def test_returns_empty_when_no_block(self) -> None:
        """Returns empty string when no code block is present."""
        result = parse_z3_snippet("I cannot extract any arithmetic from this.")
        assert result == ""

    def test_strips_surrounding_whitespace(self) -> None:
        """Leading/trailing whitespace inside the block is stripped."""
        output = "```python\n  import z3  \n```"
        snippet = parse_z3_snippet(output)
        assert snippet == "import z3"

    def test_returns_first_block_only(self) -> None:
        """When multiple code blocks exist, only the first is returned."""
        output = "```python\nfirst\n```\n```python\nsecond\n```"
        snippet = parse_z3_snippet(output)
        assert snippet == "first"

    def test_empty_input_returns_empty(self) -> None:
        """Empty string input returns empty string."""
        assert parse_z3_snippet("") == ""

    def test_preamble_prose_is_discarded(self) -> None:
        """Preamble prose before the code block is not included in the result."""
        output = "Let me write the Z3 code:\n```python\nx = 1\n```"
        snippet = parse_z3_snippet(output)
        assert "Let me" not in snippet
        assert "x = 1" in snippet


# ---------------------------------------------------------------------------
# _exec_z3_snippet (sandbox internals)
# ---------------------------------------------------------------------------


class TestExecZ3Snippet:
    """REQ-EXTRACT-019: _exec_z3_snippet sandbox execution contracts.

    Spec: SCENARIO-EXTRACT-040
    """

    def test_sat_code_returns_sat(self) -> None:
        """SCENARIO-EXTRACT-041: consistent constraints → 'sat'."""
        code = (
            "import z3\n"
            "s = z3.Solver()\n"
            "x = z3.Int('x')\n"
            "s.add(x == 5)\n"
            "print(s.check())\n"
        )
        result, err = _exec_z3_snippet(code)
        assert result == "sat"
        assert err is None

    def test_unsat_code_returns_unsat(self) -> None:
        """SCENARIO-EXTRACT-041: contradictory constraints → 'unsat'."""
        code = (
            "import z3\n"
            "s = z3.Solver()\n"
            "x = z3.Int('x')\n"
            "s.add(x == 5)\n"
            "s.add(x == 6)\n"
            "print(s.check())\n"
        )
        result, err = _exec_z3_snippet(code)
        assert result == "unsat"
        assert err is None

    def test_forbidden_import_os_raises_name_error(self) -> None:
        """SCENARIO-EXTRACT-040: importing os in sandbox → error with NameError."""
        code = "import os\nprint(os.getcwd())"
        result, err = _exec_z3_snippet(code)
        assert result == "error"
        assert err is not None
        assert "NameError" in err or "not allowed" in err

    def test_forbidden_import_sys_raises_name_error(self) -> None:
        """SCENARIO-EXTRACT-040: importing sys in sandbox → error."""
        code = "import sys\nprint(sys.version)"
        result, err = _exec_z3_snippet(code)
        assert result == "error"
        assert err is not None

    def test_forbidden_import_subprocess_raises_name_error(self) -> None:
        """SCENARIO-EXTRACT-040: importing subprocess in sandbox → error."""
        code = "import subprocess\nsubprocess.run(['ls'])"
        result, err = _exec_z3_snippet(code)
        assert result == "error"
        assert err is not None

    def test_syntax_error_returns_error(self) -> None:
        """Code with SyntaxError → ('error', message)."""
        result, err = _exec_z3_snippet("def (")
        assert result == "error"
        assert err is not None
        assert "SyntaxError" in err

    def test_empty_code_returns_unknown(self) -> None:
        """Empty code returns ('unknown', None)."""
        result, err = _exec_z3_snippet("")
        assert result == "unknown"
        assert err is None

    def test_whitespace_only_code_returns_unknown(self) -> None:
        """Whitespace-only code returns ('unknown', None)."""
        result, err = _exec_z3_snippet("   \n  ")
        assert result == "unknown"
        assert err is None

    def test_no_print_returns_unknown(self) -> None:
        """Code with no print output → ('unknown', None)."""
        code = "import z3\ns = z3.Solver()\ns.add(z3.Int('x') == 1)"
        result, err = _exec_z3_snippet(code)
        assert result == "unknown"
        assert err is None

    def test_unsat_takes_priority_over_sat_in_output(self) -> None:
        """'unsat' in output is detected before 'sat' substring check."""
        code = "print('unsat')"
        result, err = _exec_z3_snippet(code)
        assert result == "unsat"

    def test_runtime_exception_returns_error(self) -> None:
        """Runtime exceptions (e.g. ZeroDivisionError) → ('error', message)."""
        code = "x = 1 / 0"
        result, err = _exec_z3_snippet(code)
        assert result == "error"
        assert err is not None

    def test_ci_stub_code_returns_sat(self) -> None:
        """The CI stub code produces 'sat' when exec'd."""
        result, err = _exec_z3_snippet(_CI_STUB_Z3_CODE)
        assert result == "sat"
        assert err is None

    def test_z3_not_installed_returns_error(self) -> None:
        """When z3 is not importable, _exec_z3_snippet returns ('error', message)."""
        import builtins
        original_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "z3":
                raise ImportError("z3 not installed")
            return original_import(name, *args, **kwargs)

        import unittest.mock as mock
        with mock.patch("builtins.__import__", side_effect=_mock_import):
            result, err = _exec_z3_snippet("import z3\nprint('sat')")
        assert result == "error"
        assert err is not None
        assert "z3" in err.lower() or "not installed" in err.lower()


# ---------------------------------------------------------------------------
# LLMz3Formalizer — CI stub mode (llm_caller=None)
# ---------------------------------------------------------------------------


class TestLLMz3FormalizerCIStub:
    """REQ-EXTRACT-019: LLMz3Formalizer CI stub mode contracts.

    Spec: SCENARIO-EXTRACT-039
    """

    def test_ci_stub_returns_z3formalization_result(self) -> None:
        """SCENARIO-EXTRACT-039: CI stub mode returns Z3FormalizationResult."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("What is 2+2?", "2 + 2 = 4, so the answer is 4.")
        assert isinstance(result, Z3FormalizationResult)

    def test_ci_stub_formalization_mode(self) -> None:
        """SCENARIO-EXTRACT-039: formalization_mode is 'ci_stub' when llm_caller=None."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("q", "r")
        assert result.formalization_mode == "ci_stub"

    def test_ci_stub_z3_result_sat(self) -> None:
        """SCENARIO-EXTRACT-039: CI stub returns z3_result='sat'."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("q", "r")
        assert result.z3_result == "sat"
        assert result.is_sat is True

    def test_ci_stub_n_assertions_positive(self) -> None:
        """SCENARIO-EXTRACT-039: CI stub produces n_assertions >= 1."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("q", "r")
        assert result.n_assertions >= 1

    def test_ci_stub_z3_code_is_stub(self) -> None:
        """SCENARIO-EXTRACT-039: CI stub z3_code equals the hardcoded stub."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("q", "r")
        assert result.z3_code == _CI_STUB_Z3_CODE

    def test_ci_stub_source_response_length(self) -> None:
        """SCENARIO-EXTRACT-039: source_response_length equals len(response)."""
        response = "The answer is 42."
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("q", response)
        assert result.source_response_length == len(response)

    def test_ci_stub_last_result_set(self) -> None:
        """last_result is updated after each formalize() call."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        assert formalizer.last_result is None
        formalizer.formalize("q", "r")
        assert formalizer.last_result is not None

    def test_ci_stub_empty_response(self) -> None:
        """Empty response string does not crash in CI stub mode."""
        formalizer = LLMz3Formalizer(llm_caller=None)
        result = formalizer.formalize("q", "")
        assert isinstance(result, Z3FormalizationResult)
        assert result.source_response_length == 0

    def test_default_model_id(self) -> None:
        """Default model_id is 'ci_stub' when not specified."""
        formalizer = LLMz3Formalizer()
        assert formalizer._model_id == "ci_stub"

    def test_custom_model_id(self) -> None:
        """Custom model_id is stored correctly."""
        formalizer = LLMz3Formalizer(model_id="Qwen/Qwen3.5-0.8B")
        assert formalizer._model_id == "Qwen/Qwen3.5-0.8B"

    def test_max_iterations_clamped_to_one(self) -> None:
        """max_iterations=0 is clamped to 1 (must attempt at least once)."""
        formalizer = LLMz3Formalizer(max_iterations=0)
        assert formalizer._max_iterations == 1


# ---------------------------------------------------------------------------
# LLMz3Formalizer — live LLM mock
# ---------------------------------------------------------------------------


class TestLLMz3FormalizerLiveMode:
    """REQ-EXTRACT-019/020: LLMz3Formalizer with mocked LLM caller.

    Spec: SCENARIO-EXTRACT-041
    """

    def _unsat_llm_caller(self) -> MagicMock:
        """Return a mock that produces Z3 code asserting x==5 AND x==6 (unsat)."""
        unsat_code = (
            "```python\n"
            "import z3\n"
            "s = z3.Solver()\n"
            "x = z3.Int('x')\n"
            "s.add(x == 5)\n"
            "s.add(x == 6)\n"
            "print(s.check())\n"
            "```"
        )
        return MagicMock(return_value=unsat_code)

    def _sat_llm_caller(self) -> MagicMock:
        """Return a mock that produces consistent Z3 code (sat)."""
        sat_code = (
            "```python\n"
            "import z3\n"
            "s = z3.Solver()\n"
            "x = z3.Int('x')\n"
            "s.add(x == 5)\n"
            "print(s.check())\n"
            "```"
        )
        return MagicMock(return_value=sat_code)

    def test_unsat_returns_unsat_z3_result(self) -> None:
        """SCENARIO-EXTRACT-041: contradictory Z3 code → z3_result='unsat'."""
        formalizer = LLMz3Formalizer(llm_caller=self._unsat_llm_caller())
        result = formalizer.formalize(
            "What is the value of x?",
            "The answer is 5. Also the answer is 6.",
        )
        assert result.z3_result == "unsat"
        assert result.is_sat is False

    def test_unsat_n_assertions_at_least_2(self) -> None:
        """SCENARIO-EXTRACT-041: unsat code with 2 assertions → n_assertions >= 2."""
        formalizer = LLMz3Formalizer(llm_caller=self._unsat_llm_caller())
        result = formalizer.formalize("q", "The answer is 5. Also the answer is 6.")
        assert result.n_assertions >= 2

    def test_unsat_formalization_mode_llm(self) -> None:
        """SCENARIO-EXTRACT-041: formalization_mode is 'llm' for LLM path."""
        formalizer = LLMz3Formalizer(llm_caller=self._unsat_llm_caller())
        result = formalizer.formalize("q", "r")
        assert result.formalization_mode == "llm"

    def test_sat_returns_sat_z3_result(self) -> None:
        """Consistent Z3 code → z3_result='sat', is_sat=True."""
        formalizer = LLMz3Formalizer(llm_caller=self._sat_llm_caller())
        result = formalizer.formalize("What is x?", "The answer is 5.")
        assert result.z3_result == "sat"
        assert result.is_sat is True

    def test_llm_caller_called_once_on_success(self) -> None:
        """LLM caller is called exactly once when the first attempt succeeds."""
        mock = self._sat_llm_caller()
        formalizer = LLMz3Formalizer(llm_caller=mock, max_iterations=2)
        formalizer.formalize("q", "r")
        assert mock.call_count == 1

    def test_llm_caller_retries_on_error(self) -> None:
        """LLM caller is called again if exec returns error on first attempt.

        SCENARIO-EXTRACT-040 (retry path): the first call returns broken code,
        the second call returns valid sat code.
        """
        broken_code = "```python\nimport os\nprint(os.getcwd())\n```"
        good_code = (
            "```python\n"
            "import z3\n"
            "s = z3.Solver()\n"
            "print(s.check())\n"
            "```"
        )
        mock = MagicMock(side_effect=[broken_code, good_code])
        formalizer = LLMz3Formalizer(llm_caller=mock, max_iterations=2)
        result = formalizer.formalize("q", "r")
        assert mock.call_count == 2
        assert result.z3_result == "sat"

    def test_llm_error_code_gives_error_result(self) -> None:
        """SCENARIO-EXTRACT-040: code with forbidden import → z3_result='error'."""
        forbidden_code = "```python\nimport os\nprint(os.getcwd())\n```"
        mock = MagicMock(return_value=forbidden_code)
        formalizer = LLMz3Formalizer(llm_caller=mock, max_iterations=1)
        result = formalizer.formalize("q", "r")
        assert result.z3_result == "error"
        assert result.error_message is not None

    def test_llm_no_code_block_returns_unknown(self) -> None:
        """LLM output without a code block → z3_result='unknown'."""
        mock = MagicMock(return_value="I cannot formalize this.")
        formalizer = LLMz3Formalizer(llm_caller=mock, max_iterations=1)
        result = formalizer.formalize("q", "r")
        assert result.z3_result == "unknown"
        assert result.z3_code == ""

    def test_llm_raises_exception_returns_unknown(self) -> None:
        """LLM caller that raises → z3_result='unknown', error_message set."""
        mock = MagicMock(side_effect=RuntimeError("model unavailable"))
        formalizer = LLMz3Formalizer(llm_caller=mock)
        result = formalizer.formalize("q", "r")
        assert result.z3_result == "unknown"
        assert result.error_message is not None
        assert "LLM call failed" in result.error_message

    def test_source_response_length_in_live_mode(self) -> None:
        """source_response_length is set correctly in live LLM mode."""
        response = "The answer is 42."
        formalizer = LLMz3Formalizer(llm_caller=self._sat_llm_caller())
        result = formalizer.formalize("q", response)
        assert result.source_response_length == len(response)

    def test_last_result_updated_in_live_mode(self) -> None:
        """last_result is updated after a live formalize() call."""
        formalizer = LLMz3Formalizer(llm_caller=self._sat_llm_caller())
        formalizer.formalize("q", "r")
        assert formalizer.last_result is not None
        assert isinstance(formalizer.last_result, Z3FormalizationResult)

    def test_max_iterations_2_exhausted_gives_error(self) -> None:
        """If all iterations produce error, final result is error."""
        broken = "```python\nimport os\nprint(os.getcwd())\n```"
        mock = MagicMock(return_value=broken)
        formalizer = LLMz3Formalizer(llm_caller=mock, max_iterations=2)
        result = formalizer.formalize("q", "r")
        assert mock.call_count == 2
        assert result.z3_result == "error"

    def test_no_code_block_continues_to_next_iteration(self) -> None:
        """If LLM returns no code block, loop continues to next iteration."""
        no_block = "I cannot formalize this."
        good_code = (
            "```python\n"
            "import z3\n"
            "s = z3.Solver()\n"
            "print(s.check())\n"
            "```"
        )
        mock = MagicMock(side_effect=[no_block, good_code])
        formalizer = LLMz3Formalizer(llm_caller=mock, max_iterations=2)
        result = formalizer.formalize("q", "r")
        assert mock.call_count == 2
        assert result.z3_result == "sat"

    def test_zero_false_positives_sat_is_not_violation(self) -> None:
        """REQ-EXTRACT-020: sat result must NOT be treated as a violation.

        is_sat=True means the arithmetic is consistent — no violation.
        """
        formalizer = LLMz3Formalizer(llm_caller=self._sat_llm_caller())
        result = formalizer.formalize("q", "2 + 2 = 4")
        assert result.is_sat is True
        # No violation: callers should only flag when z3_result == "unsat"
        assert result.z3_result != "unsat"


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


def test_pipeline_exports_llm_z3_formalizer() -> None:
    """LLMz3Formalizer and Z3FormalizationResult are exported from carnot.pipeline."""
    from carnot.pipeline import LLMz3Formalizer as _LLMz3Formalizer  # noqa: F401
    from carnot.pipeline import Z3FormalizationResult as _Z3FormalizationResult  # noqa: F401

    assert _LLMz3Formalizer is LLMz3Formalizer
    assert _Z3FormalizationResult is Z3FormalizationResult


def test_pipeline_exports_helper_functions() -> None:
    """build_z3_formalization_prompt and parse_z3_snippet are exported."""
    from carnot.pipeline import build_z3_formalization_prompt as _bfp  # noqa: F401
    from carnot.pipeline import parse_z3_snippet as _pzs  # noqa: F401

    assert _bfp is build_z3_formalization_prompt
    assert _pzs is parse_z3_snippet
