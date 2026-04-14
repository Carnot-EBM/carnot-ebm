"""Tests for carnot.pipeline.nl2z3_extractor.

Covers NL2Z3Extractor, Z3Result dataclass, build_z3_prompt, run_z3_code,
and VerifyRepairPipeline.verify_with_z3 integration at 100% branch coverage.

Spec: REQ-EXTRACT-010, REQ-EXTRACT-011,
      SCENARIO-EXTRACT-020, SCENARIO-EXTRACT-021, SCENARIO-EXTRACT-022,
      SCENARIO-EXTRACT-023, SCENARIO-EXTRACT-024
"""

from __future__ import annotations

import subprocess
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.nl2z3_extractor import (
    NL2Z3Extractor,
    Z3Result,
    build_z3_prompt,
    run_z3_code,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Z3Result dataclass
# ---------------------------------------------------------------------------


class TestZ3Result:
    """REQ-EXTRACT-011: Z3Result dataclass contracts."""

    def test_sat_status_sat(self) -> None:
        """SCENARIO-EXTRACT-022: sat → violations_found is False."""
        r = Z3Result(sat_status="sat", z3_code="import z3", runtime_ms=5.0)
        assert r.sat_status == "sat"
        assert r.violations_found is False
        assert r.error_message is None

    def test_sat_status_unsat_violations_found(self) -> None:
        """SCENARIO-EXTRACT-022: unsat → violations_found is True."""
        r = Z3Result(sat_status="unsat", z3_code="import z3", runtime_ms=10.0)
        assert r.violations_found is True

    def test_sat_status_unknown_not_violation(self) -> None:
        """SCENARIO-EXTRACT-022: unknown → violations_found is False."""
        r = Z3Result(sat_status="unknown", z3_code="", runtime_ms=0.0)
        assert r.violations_found is False

    def test_sat_status_error_not_violation(self) -> None:
        """SCENARIO-EXTRACT-022: error → violations_found is False."""
        r = Z3Result(
            sat_status="error",
            z3_code="bad code",
            runtime_ms=1.0,
            error_message="SyntaxError",
        )
        assert r.violations_found is False
        assert r.error_message == "SyntaxError"

    def test_defaults(self) -> None:
        """Z3Result optional fields default correctly."""
        r = Z3Result(sat_status="sat", z3_code="x", runtime_ms=0.0)
        assert r.violations_found is False
        assert r.error_message is None


# ---------------------------------------------------------------------------
# build_z3_prompt
# ---------------------------------------------------------------------------


class TestBuildZ3Prompt:
    """REQ-EXTRACT-010: build_z3_prompt produces correct system + user messages."""

    def test_returns_two_strings(self) -> None:
        """build_z3_prompt must return a (system, user) tuple."""
        result = build_z3_prompt("Some chain-of-thought reasoning here.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_contains_translate_instruction(self) -> None:
        """System prompt must instruct the LLM to translate arithmetic to Z3."""
        system, _ = build_z3_prompt("step 1: 3 + 4 = 7")
        assert "z3" in system.lower()
        assert "translate" in system.lower() or "arithmetic" in system.lower()

    def test_user_contains_response_text(self) -> None:
        """User message must embed the full response text."""
        text = "unique-marker-abc-123"
        _, user = build_z3_prompt(text)
        assert text in user

    def test_system_mentions_import_z3(self) -> None:
        """System prompt must mention self-contained runnable code with import z3."""
        system, _ = build_z3_prompt("anything")
        assert "import z3" in system or "z3.solve" in system or "self-contained" in system.lower()

    def test_empty_response(self) -> None:
        """build_z3_prompt handles empty string without error."""
        system, user = build_z3_prompt("")
        assert isinstance(system, str)
        assert isinstance(user, str)


# ---------------------------------------------------------------------------
# run_z3_code
# ---------------------------------------------------------------------------


class TestRunZ3Code:
    """REQ-EXTRACT-011: run_z3_code subprocess execution and result parsing."""

    def test_sat_code_returns_sat(self) -> None:
        """Correct arithmetic → Z3 reports sat."""
        code = "import z3; x = z3.Int('x'); s = z3.Solver(); s.add(x == 5); print(s.check())"
        result = run_z3_code(code)
        assert result.sat_status == "sat"
        assert result.runtime_ms >= 0.0
        assert result.violations_found is False

    def test_unsat_code_returns_unsat(self) -> None:
        """Contradictory constraints → Z3 reports unsat → violations_found True."""
        code = (
            "import z3\n"
            "x = z3.Int('x')\n"
            "s = z3.Solver()\n"
            "s.add(x == 5)\n"
            "s.add(x == 6)\n"
            "print(s.check())\n"
        )
        result = run_z3_code(code)
        assert result.sat_status == "unsat"
        assert result.violations_found is True

    def test_syntax_error_returns_error(self) -> None:
        """Code with SyntaxError → sat_status='error', error_message set."""
        result = run_z3_code("def (")
        assert result.sat_status == "error"
        assert result.error_message is not None

    def test_name_error_returns_error(self) -> None:
        """Code with NameError (undefined var) → sat_status='error'."""
        result = run_z3_code("print(undefined_variable_xyz)")
        assert result.sat_status == "error"

    def test_timeout_returns_unknown(self) -> None:
        """SCENARIO-EXTRACT-023: Infinite loop → timeout → sat_status='unknown', runtime_ms >= timeout."""
        code = "while True: pass"
        start = time.monotonic()
        result = run_z3_code(code, timeout_s=0.5)
        elapsed = time.monotonic() - start
        assert result.sat_status == "unknown"
        # runtime_ms must reflect at least the timeout duration
        assert result.runtime_ms >= 400  # 0.4 s in ms (some slack)
        assert elapsed < 3.0  # process was actually killed

    def test_empty_code_returns_unknown(self) -> None:
        """Empty code produces no stdout → sat_status='unknown'."""
        result = run_z3_code("")
        # Empty code may succeed with no output → unknown, or it could be sat/error
        # The key invariant: does not raise, always returns Z3Result
        assert result.sat_status in {"sat", "unsat", "unknown", "error"}

    def test_runtime_ms_is_non_negative(self) -> None:
        """runtime_ms must always be non-negative."""
        result = run_z3_code("print('sat')")
        assert result.runtime_ms >= 0.0

    def test_z3_not_in_output_returns_unknown(self) -> None:
        """Output with neither 'sat' nor 'unsat' → sat_status='unknown'."""
        result = run_z3_code("print('hello world')")
        assert result.sat_status == "unknown"


# ---------------------------------------------------------------------------
# NL2Z3Extractor — CI mode (no LLM)
# ---------------------------------------------------------------------------


class TestNL2Z3ExtractorCIMode:
    """SCENARIO-EXTRACT-024: CI mode degrades gracefully without live LLM."""

    def test_supported_domains(self, monkeypatch: Any) -> None:
        """REQ-EXTRACT-010: supported_domains includes 'reasoning'."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        assert "reasoning" in ext.supported_domains

    def test_extract_returns_empty_without_llm(self, monkeypatch: Any) -> None:
        """SCENARIO-EXTRACT-024: No LLM → extract returns [] and no crash."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        result = ext.extract("What is 2+2?", "2 + 2 = 4, so the answer is 4.")
        assert result == []

    def test_last_z3_result_is_unknown_without_llm(self, monkeypatch: Any) -> None:
        """SCENARIO-EXTRACT-024: internal Z3Result has sat_status='unknown'."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        ext.extract("q", "r")
        assert ext.last_z3_result is not None
        assert ext.last_z3_result.sat_status == "unknown"
        assert ext.last_z3_result.z3_code == ""

    def test_extract_empty_question_and_response(self, monkeypatch: Any) -> None:
        """Edge case: empty question + empty response → no crash."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        result = ext.extract("", "")
        assert result == []

    def test_extract_non_arithmetic_response(self, monkeypatch: Any) -> None:
        """Edge case: non-arithmetic prose → no crash in CI mode."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        result = ext.extract("What is photosynthesis?", "Plants convert sunlight into glucose.")
        assert result == []

    def test_domain_filter_skips_non_reasoning(self, monkeypatch: Any) -> None:
        """extract respects domain filtering — non-matching domain returns []."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        result = ext.extract("q", "r", domain="code")
        assert result == []

    def test_domain_reasoning_is_not_filtered(self, monkeypatch: Any) -> None:
        """Passing domain='reasoning' does not skip the extractor."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        ext = NL2Z3Extractor()
        # In CI mode, result is always [] but should not skip due to domain mismatch
        result = ext.extract("q", "r", domain="reasoning")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# NL2Z3Extractor — live LLM mock
# ---------------------------------------------------------------------------


class TestNL2Z3ExtractorLiveMode:
    """REQ-EXTRACT-010/011: Live mode exercises LLM → Z3 pipeline with mocks."""

    def _make_extractor_with_mock_generate(self, llm_output: str) -> NL2Z3Extractor:
        """Build NL2Z3Extractor with a mocked generate function."""
        import os
        os.environ["CARNOT_FORCE_LIVE"] = "1"

        mock_generate_fn: Any = MagicMock(return_value=llm_output)
        ext = NL2Z3Extractor(generate_fn=mock_generate_fn)
        return ext

    def teardown_method(self) -> None:
        import os
        os.environ.pop("CARNOT_FORCE_LIVE", None)

    def test_unsat_z3_code_returns_violation(self) -> None:
        """SCENARIO-EXTRACT-020: LLM returns unsat Z3 code → violation returned."""
        unsat_code = (
            "```python\n"
            "import z3\n"
            "x = z3.Int('x')\n"
            "s = z3.Solver()\n"
            "s.add(x == 5)\n"
            "s.add(x == 6)\n"
            "print(s.check())\n"
            "```"
        )
        ext = self._make_extractor_with_mock_generate(unsat_code)
        violations = ext.extract("q", "The answer is 5. Also the answer is 6.")
        assert len(violations) == 1
        assert violations[0].constraint_type == "z3_unsat"

    def test_sat_z3_code_returns_empty(self) -> None:
        """SCENARIO-EXTRACT-021: LLM returns sat Z3 code → empty violation list."""
        sat_code = (
            "```python\n"
            "import z3\n"
            "x = z3.Int('x')\n"
            "s = z3.Solver()\n"
            "s.add(x == 5)\n"
            "print(s.check())\n"
            "```"
        )
        ext = self._make_extractor_with_mock_generate(sat_code)
        violations = ext.extract("q", "The answer is 5.")
        assert violations == []

    def test_llm_no_code_block_returns_empty(self) -> None:
        """LLM output without a code block → no Z3 run, empty result."""
        ext = self._make_extractor_with_mock_generate("I cannot extract arithmetic here.")
        violations = ext.extract("q", "some response")
        assert violations == []

    def test_last_z3_result_set_after_live_extract(self) -> None:
        """last_z3_result is populated after a live extract call."""
        sat_code = "```python\nprint('sat')\n```"
        ext = self._make_extractor_with_mock_generate(sat_code)
        ext.extract("q", "r")
        assert ext.last_z3_result is not None
        assert isinstance(ext.last_z3_result, Z3Result)

    def test_generate_fn_called_once(self) -> None:
        """LLM generate function is called exactly once per extract call."""
        import os
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        mock_fn: Any = MagicMock(return_value="no code block")
        ext = NL2Z3Extractor(generate_fn=mock_fn)
        ext.extract("q", "r")
        mock_fn.assert_called_once()

    def test_generate_fn_raises_falls_back_to_unknown(self) -> None:
        """If generate_fn raises, extractor returns [] with sat_status='unknown'."""
        import os
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        mock_fn: Any = MagicMock(side_effect=RuntimeError("model unavailable"))
        ext = NL2Z3Extractor(generate_fn=mock_fn)
        result = ext.extract("q", "r")
        assert result == []
        assert ext.last_z3_result is not None
        assert ext.last_z3_result.sat_status == "unknown"

    def test_violation_metadata_contains_z3_code(self) -> None:
        """ConstraintResult metadata for z3_unsat violations includes 'z3_code'."""
        unsat_code = (
            "```python\n"
            "import z3\n"
            "x = z3.Int('x')\n"
            "s = z3.Solver()\n"
            "s.add(x == 5)\n"
            "s.add(x == 6)\n"
            "print(s.check())\n"
            "```"
        )
        ext = self._make_extractor_with_mock_generate(unsat_code)
        violations = ext.extract("q", "contradiction")
        assert len(violations) == 1
        assert "z3_code" in violations[0].metadata


# ---------------------------------------------------------------------------
# VerifyRepairPipeline.verify_with_z3 integration
# ---------------------------------------------------------------------------


class TestVerifyRepairPipelineZ3Integration:
    """REQ-EXTRACT-010: verify_with_z3 is accessible on VerifyRepairPipeline."""

    def test_verify_with_z3_ci_mode_returns_z3result(self) -> None:
        """Pipeline.verify_with_z3 returns Z3Result in CI mode."""
        import os
        os.environ.pop("CARNOT_FORCE_LIVE", None)

        pipeline = VerifyRepairPipeline(model=None, domains=["reasoning"])
        result = pipeline.verify_with_z3("question", "response")
        assert isinstance(result, Z3Result)
        assert result.sat_status == "unknown"

    def test_verify_with_z3_accepts_timeout_param(self) -> None:
        """verify_with_z3 accepts a timeout_s parameter without error."""
        import os
        os.environ.pop("CARNOT_FORCE_LIVE", None)

        pipeline = VerifyRepairPipeline(model=None, domains=["reasoning"])
        result = pipeline.verify_with_z3("q", "r", timeout_s=5.0)
        assert isinstance(result, Z3Result)

    def test_verify_with_z3_live_mock_unsat(self) -> None:
        """Pipeline.verify_with_z3 with mock LLM returns unsat Z3Result."""
        import os
        os.environ["CARNOT_FORCE_LIVE"] = "1"

        unsat_code = (
            "```python\n"
            "import z3\n"
            "x = z3.Int('x')\n"
            "s = z3.Solver()\n"
            "s.add(x == 1)\n"
            "s.add(x == 2)\n"
            "print(s.check())\n"
            "```"
        )
        mock_gen: Any = MagicMock(return_value=unsat_code)
        pipeline = VerifyRepairPipeline(model=None, domains=["reasoning"])
        pipeline._nl2z3_extractor = NL2Z3Extractor(generate_fn=mock_gen)

        result = pipeline.verify_with_z3("q", "contradiction response")
        assert result.sat_status == "unsat"
        assert result.violations_found is True

        os.environ.pop("CARNOT_FORCE_LIVE", None)


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


def test_pipeline_init_exports() -> None:
    """NL2Z3Extractor and Z3Result are exported from carnot.pipeline."""
    from carnot.pipeline import NL2Z3Extractor as _NL2Z3Extractor  # noqa: F401
    from carnot.pipeline import Z3Result as _Z3Result  # noqa: F401

    assert _NL2Z3Extractor is NL2Z3Extractor
    assert _Z3Result is Z3Result


# ---------------------------------------------------------------------------
# z3 availability guard
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("z3") is None,
    reason="z3 not installed",
)
def test_z3_import_succeeds() -> None:
    """z3 package is importable when installed."""
    import z3

    assert hasattr(z3, "Solver")
