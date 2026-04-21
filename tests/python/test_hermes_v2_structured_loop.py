"""Tests for HermesV2StructuredLoop and HermesV2StructuredResult.

Spec traces: REQ-VERIFY-147, REQ-VERIFY-148,
             SCENARIO-VERIFY-197, SCENARIO-VERIFY-198, SCENARIO-VERIFY-199
"""

from __future__ import annotations

import pytest

from carnot.pipeline.hermes_v2_structured_loop import (
    HermesV2StructuredLoop,
    HermesV2StructuredResult,
)
from carnot.pipeline.structured_equation_forcer import StructuredEquationForcer
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def verifier() -> SymCodeVerifier:
    return SymCodeVerifier(llm_caller=None)


@pytest.fixture()
def forcer(verifier: SymCodeVerifier) -> StructuredEquationForcer:
    return StructuredEquationForcer(llm_caller=None, verifier=verifier)


@pytest.fixture()
def loop_ci(verifier: SymCodeVerifier, forcer: StructuredEquationForcer) -> HermesV2StructuredLoop:
    """CI-mode loop with no live LLM."""
    return HermesV2StructuredLoop(llm_caller=None, verifier=verifier, forcer=forcer)


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-197: HermesV2StructuredResult dataclass fields
# REQ-VERIFY-147
# ---------------------------------------------------------------------------


class TestHermesV2StructuredResult:
    """REQ-VERIFY-147: HermesV2StructuredResult captures all required fields."""

    def test_default_fields(self):
        """SCENARIO-VERIFY-197: dataclass initialises with expected defaults."""
        r = HermesV2StructuredResult(question="q", full_response="resp")
        assert r.question == "q"
        assert r.full_response == "resp"
        assert r.compute_lines == []
        assert r.n_compute_lines == 0
        assert r.n_violations == 0
        assert r.n_hints == 0
        assert r.recall_contribution is False

    def test_field_assignment(self):
        """Fields can be set to non-default values."""
        r = HermesV2StructuredResult(
            question="q",
            full_response="r",
            compute_lines=["47 + 28 = 76"],
            n_compute_lines=1,
            n_violations=1,
            n_hints=1,
            recall_contribution=True,
        )
        assert r.n_compute_lines == 1
        assert r.recall_contribution is True


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-198: CI stub produces a recall_contribution=True result
# REQ-VERIFY-147
# ---------------------------------------------------------------------------


class TestHermesV2StructuredLoopCI:
    """REQ-VERIFY-147: CI stub path exercises the full pipeline without a GPU."""

    def test_ci_generates_compute_lines(self, loop_ci: HermesV2StructuredLoop):
        """SCENARIO-VERIFY-198: CI stub response contains at least one COMPUTE: line."""
        result = loop_ci.generate_structured("What is 47 + 28?")
        assert isinstance(result, HermesV2StructuredResult)
        assert result.n_compute_lines >= 1
        assert len(result.compute_lines) == result.n_compute_lines

    def test_ci_detects_violation(self, loop_ci: HermesV2StructuredLoop):
        """SCENARIO-VERIFY-198: CI stub response has wrong arithmetic → n_violations > 0."""
        result = loop_ci.generate_structured("What is 47 + 28?")
        # The CI stub uses '47 + 28 = 76' which is wrong (correct: 75).
        # SymCodeVerifier(None) uses regex extraction → executed=75, stated=76 → violation.
        assert result.n_violations >= 1
        assert result.recall_contribution is True

    def test_ci_hints_match_violations(self, loop_ci: HermesV2StructuredLoop):
        """n_hints equals n_violations in current implementation."""
        result = loop_ci.generate_structured("math question")
        assert result.n_hints == result.n_violations

    def test_ci_full_response_non_empty(self, loop_ci: HermesV2StructuredLoop):
        """CI stub always returns a non-empty full_response."""
        result = loop_ci.generate_structured("q")
        assert result.full_response.strip() != ""

    def test_ci_question_preserved(self, loop_ci: HermesV2StructuredLoop):
        """question field in result matches the input question."""
        q = "What is 10 + 5?"
        result = loop_ci.generate_structured(q)
        assert result.question == q


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-199: Live llm_caller path dispatches correctly
# REQ-VERIFY-147, REQ-VERIFY-148
# ---------------------------------------------------------------------------


class TestHermesV2StructuredLoopLive:
    """REQ-VERIFY-147/148: Live llm_caller path uses forcer and verifier correctly."""

    def test_live_caller_receives_compute_prompt(
        self,
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
    ):
        """SCENARIO-VERIFY-199: live caller receives a prompt containing the COMPUTE: instruction."""
        captured: list[str] = []

        def mock_caller(prompt: str, system: str) -> str:
            captured.append(prompt)
            # Return a correct COMPUTE: response so we know the live path is taken.
            return "COMPUTE: 3 + 4 = 7 So the answer is 7."

        loop = HermesV2StructuredLoop(
            llm_caller=mock_caller,
            verifier=verifier,
            forcer=forcer,
        )
        result = loop.generate_structured("What is 3 + 4?")
        assert len(captured) == 1
        # The prompt must contain the COMPUTE: instruction from the forcer.
        assert "COMPUTE:" in captured[0]

    def test_live_correct_response_no_violation(
        self,
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
    ):
        """A correct COMPUTE: line produces n_violations=0 and recall_contribution=False."""

        def mock_caller(prompt: str, system: str) -> str:
            return "COMPUTE: 3 + 4 = 7 So the answer is 7."

        loop = HermesV2StructuredLoop(
            llm_caller=mock_caller,
            verifier=verifier,
            forcer=forcer,
        )
        result = loop.generate_structured("What is 3 + 4?")
        assert result.n_compute_lines >= 1
        # 3+4=7 is correct, so no violation should be detected.
        assert result.n_violations == 0
        assert result.recall_contribution is False

    def test_live_incorrect_response_violation(
        self,
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
    ):
        """An incorrect COMPUTE: line (wrong answer) produces recall_contribution=True."""

        def mock_caller(prompt: str, system: str) -> str:
            # 47+28=76 is wrong (correct: 75).
            return "COMPUTE: 47 + 28 = 76 So total is 76."

        loop = HermesV2StructuredLoop(
            llm_caller=mock_caller,
            verifier=verifier,
            forcer=forcer,
        )
        result = loop.generate_structured("What is 47 + 28?")
        assert result.n_violations >= 1
        assert result.recall_contribution is True

    def test_live_empty_response_no_crash(
        self,
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
    ):
        """An LLM returning no COMPUTE: lines produces a valid result with 0 violations."""

        def mock_caller(prompt: str, system: str) -> str:
            return "The answer is seven."

        loop = HermesV2StructuredLoop(
            llm_caller=mock_caller,
            verifier=verifier,
            forcer=forcer,
        )
        result = loop.generate_structured("q")
        assert result.n_compute_lines == 0
        assert result.n_violations == 0
        assert result.recall_contribution is False

    def test_max_sentences_stored(
        self,
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
    ):
        """max_sentences is stored on the instance for future use."""
        loop = HermesV2StructuredLoop(
            llm_caller=None,
            verifier=verifier,
            forcer=forcer,
            max_sentences=20,
        )
        assert loop.max_sentences == 20

    def test_components_stored(
        self,
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
    ):
        """verifier and forcer are accessible as instance attributes."""
        loop = HermesV2StructuredLoop(
            llm_caller=None,
            verifier=verifier,
            forcer=forcer,
        )
        assert loop.verifier is verifier
        assert loop.forcer is forcer
        assert loop.llm_caller is None
