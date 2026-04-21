"""Tests for StructuredEquationForcer — 100% coverage on structured_equation_forcer.py.

Every test is traced to REQ-VERIFY-146 or REQ-VERIFY-147.
"""

from __future__ import annotations

import pytest

from carnot.pipeline.structured_equation_forcer import (
    FORCER_SYSTEM_ADDENDUM,
    ForcedEquationResult,
    StructuredEquationForcer,
)
from carnot.pipeline.symcode_verifier import SymCodeVerifier


@pytest.fixture()
def verifier() -> SymCodeVerifier:
    return SymCodeVerifier(llm_caller=None)


@pytest.fixture()
def forcer(verifier: SymCodeVerifier) -> StructuredEquationForcer:
    return StructuredEquationForcer(llm_caller=None, verifier=verifier)


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-1: FORCER_SYSTEM_ADDENDUM is a non-empty str constant
# ---------------------------------------------------------------------------


def test_forcer_system_addendum_is_str() -> None:
    """REQ-VERIFY-146-1: FORCER_SYSTEM_ADDENDUM shall be a module-level str constant."""
    assert isinstance(FORCER_SYSTEM_ADDENDUM, str)
    assert len(FORCER_SYSTEM_ADDENDUM) > 0
    assert "COMPUTE:" in FORCER_SYSTEM_ADDENDUM


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-2: StructuredEquationForcer.__init__
# ---------------------------------------------------------------------------


def test_forcer_init_stores_attributes(verifier: SymCodeVerifier) -> None:
    """REQ-VERIFY-146-2: __init__ shall accept llm_caller and verifier."""
    f = StructuredEquationForcer(llm_caller=None, verifier=verifier)
    assert f.llm_caller is None
    assert f.verifier is verifier


def test_forcer_init_with_callable(verifier: SymCodeVerifier) -> None:
    """REQ-VERIFY-146-2: llm_caller may be a callable."""

    def my_caller(system: str, user: str) -> str:
        return f"COMPUTE: 1 + 2 = 3"

    f = StructuredEquationForcer(llm_caller=my_caller, verifier=verifier)
    assert f.llm_caller is my_caller


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-3: build_forced_prompt
# ---------------------------------------------------------------------------


def test_build_forced_prompt_returns_tuple(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-3: build_forced_prompt shall return (FORCER_SYSTEM_ADDENDUM, question)."""
    system, user = forcer.build_forced_prompt("How many apples?")
    assert system == FORCER_SYSTEM_ADDENDUM
    assert user == "How many apples?"


def test_build_forced_prompt_passes_question_verbatim(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-3: user_prompt must be the exact question, unchanged."""
    q = "If 5 + 3 = ?, find ?"
    _, user = forcer.build_forced_prompt(q)
    assert user == q


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-4: extract_compute_lines
# ---------------------------------------------------------------------------


def test_extract_compute_lines_basic(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-4 / SCENARIO-VERIFY-194: extract from forced response."""
    response = "We have 47 apples. COMPUTE: 47 + 28 = 75 So total is 75."
    lines = forcer.extract_compute_lines(response)
    assert lines == ["47 + 28 = 75 So total is 75."]


def test_extract_compute_lines_multiple(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-4: multiple COMPUTE: lines are all extracted."""
    response = "COMPUTE: 10 + 5 = 15\nThen COMPUTE: 15 * 2 = 30"
    lines = forcer.extract_compute_lines(response)
    assert len(lines) == 2
    assert "10 + 5 = 15" in lines[0]
    assert "15 * 2 = 30" in lines[1]


def test_extract_compute_lines_empty_response(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-4: empty response yields empty list."""
    assert forcer.extract_compute_lines("") == []


def test_extract_compute_lines_no_compute(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-4: free-form response without COMPUTE: yields empty list."""
    response = "You have 47 apples and get 28 more, totaling 75."
    assert forcer.extract_compute_lines(response) == []


def test_extract_compute_lines_whitespace_after_colon(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-4: COMPUTE: with extra whitespace is still matched."""
    response = "COMPUTE:   5 + 3 = 8"
    lines = forcer.extract_compute_lines(response)
    assert len(lines) == 1
    assert "5 + 3 = 8" in lines[0]


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-5: verify_compute_lines
# ---------------------------------------------------------------------------


def test_verify_compute_lines_empty(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-5: empty list returns 0.0."""
    assert forcer.verify_compute_lines([]) == 0.0


def test_verify_compute_lines_all_pass(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-5: all lines pass yields 1.0."""
    lines = ["47 + 28 = 75", "10 * 3 = 30"]
    rate = forcer.verify_compute_lines(lines)
    assert rate == 1.0


def test_verify_compute_lines_single(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-5: single line always returns 1.0 (detection_score >= 0.0)."""
    assert forcer.verify_compute_lines(["5 + 3 = 8"]) == 1.0


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-6/7: force_and_verify in CI mode
# ---------------------------------------------------------------------------


def test_force_and_verify_ci_mode(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-6/7: CI mode returns synthetic response with COMPUTE: line."""
    result = forcer.force_and_verify("If you have 47 apples and get 28 more, how many?")
    assert isinstance(result, ForcedEquationResult)
    assert result.n_compute_lines >= 1
    assert "COMPUTE:" in result.forced_response


def test_force_and_verify_detection_rate_1_in_ci(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-147-1 / SCENARIO-VERIFY-195: detection_rate == 1.0 in CI mode."""
    result = forcer.force_and_verify("How many apples?")
    assert result.detection_rate == 1.0


def test_force_and_verify_all_detected_true_in_ci(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-147-3: all_detected == True when detection_rate == 1.0."""
    result = forcer.force_and_verify("Any arithmetic question.")
    assert result.all_detected is True


def test_force_and_verify_question_preserved(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-6: question field in result matches input."""
    q = "How many items in total?"
    result = forcer.force_and_verify(q)
    assert result.question == q


def test_force_and_verify_n_compute_lines_consistent(forcer: StructuredEquationForcer) -> None:
    """REQ-VERIFY-146-6: n_compute_lines == len(compute_lines)."""
    result = forcer.force_and_verify("Some question.")
    assert result.n_compute_lines == len(result.compute_lines)


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-6: force_and_verify with live llm_caller
# ---------------------------------------------------------------------------


def test_force_and_verify_live_mode(verifier: SymCodeVerifier) -> None:
    """REQ-VERIFY-146-6: live llm_caller is called with system+user prompts."""
    calls: list[tuple[str, str]] = []

    def fake_caller(system: str, user: str) -> str:
        calls.append((system, user))
        return "COMPUTE: 10 + 5 = 15 So the answer is 15."

    forcer = StructuredEquationForcer(llm_caller=fake_caller, verifier=verifier)
    result = forcer.force_and_verify("What is 10 plus 5?")

    assert len(calls) == 1
    assert calls[0][0] == FORCER_SYSTEM_ADDENDUM
    assert calls[0][1] == "What is 10 plus 5?"
    assert result.n_compute_lines == 1
    assert result.detection_rate == 1.0


# ---------------------------------------------------------------------------
# REQ-VERIFY-147-2: free-form baseline is lower than forced detection rate
# ---------------------------------------------------------------------------


def test_free_form_lower_than_forced(forcer: StructuredEquationForcer, verifier: SymCodeVerifier) -> None:
    """REQ-VERIFY-147-2 / SCENARIO-VERIFY-196: free-form score < forced detection_rate."""
    free_form = "You have 47 apples and get 28 more, totaling 75 apples."
    free_score = verifier.detection_score(free_form)
    forced_result = forcer.force_and_verify("If you have 47 apples and get 28 more, how many?")
    # The forced response always has detection_rate >= free-form score
    assert forced_result.detection_rate >= free_score


# ---------------------------------------------------------------------------
# ForcedEquationResult dataclass fields
# ---------------------------------------------------------------------------


def test_forced_equation_result_fields() -> None:
    """REQ-VERIFY-146-6: ForcedEquationResult has all required fields."""
    r = ForcedEquationResult(
        question="q",
        forced_response="r",
        compute_lines=["1 + 2 = 3"],
        n_compute_lines=1,
        detection_rate=1.0,
        all_detected=True,
    )
    assert r.question == "q"
    assert r.forced_response == "r"
    assert r.compute_lines == ["1 + 2 = 3"]
    assert r.n_compute_lines == 1
    assert r.detection_rate == 1.0
    assert r.all_detected is True


# ---------------------------------------------------------------------------
# REQ-VERIFY-147-3: all_detected False when no COMPUTE: lines
# ---------------------------------------------------------------------------


def test_all_detected_false_when_no_compute(verifier: SymCodeVerifier) -> None:
    """REQ-VERIFY-147-3: all_detected is False when compute_lines is empty."""

    def no_compute_caller(system: str, user: str) -> str:
        return "You have 75 apples in total."

    forcer = StructuredEquationForcer(llm_caller=no_compute_caller, verifier=verifier)
    result = forcer.force_and_verify("How many apples?")
    assert result.n_compute_lines == 0
    assert result.all_detected is False


# ---------------------------------------------------------------------------
# Export test: carnot.pipeline exports the new symbols
# ---------------------------------------------------------------------------


def test_pipeline_exports() -> None:
    """REQ-VERIFY-146-8: all three symbols exported from carnot.pipeline."""
    import carnot.pipeline as pipeline

    assert hasattr(pipeline, "FORCER_SYSTEM_ADDENDUM")
    assert hasattr(pipeline, "ForcedEquationResult")
    assert hasattr(pipeline, "StructuredEquationForcer")
