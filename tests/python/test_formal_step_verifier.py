"""Tests for FormalStepVerifier (arXiv 2603.29500 approach).

Covers all public methods of FormalStepVerifier with 100% module coverage.

Spec: REQ-VERIFY-165, SCENARIO-VERIFY-217, SCENARIO-VERIFY-218
"""

from __future__ import annotations

import pytest

from carnot.pipeline.formal_step_verifier import FormalStepVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def verifier() -> FormalStepVerifier:
    """Fresh FormalStepVerifier instance for each test."""
    return FormalStepVerifier()


# ---------------------------------------------------------------------------
# verify_chain() tests
# ---------------------------------------------------------------------------

# SCENARIO-VERIFY-217: verify_chain on an empty list returns empty list.
def test_verify_chain_empty(verifier: FormalStepVerifier) -> None:
    """verify_chain([]) must return [] with no errors.

    Spec: SCENARIO-VERIFY-217
    """
    result = verifier.verify_chain([])
    assert result == []


# SCENARIO-VERIFY-217: first step always gets entailment=True (no prior context).
def test_verify_chain_single_step(verifier: FormalStepVerifier) -> None:
    """A single-step chain: step 0 has entailment=True by definition.

    Spec: SCENARIO-VERIFY-217
    """
    steps = ["47 + 28 = 75"]
    result = verifier.verify_chain(steps)
    assert len(result) == 1
    assert result[0]["step_idx"] == 0
    assert result[0]["entailment"] is True
    # Verdict for step 0 is always "correct" (no prior context).
    assert result[0]["verdict"] == "correct"


def test_verify_chain_consistent_steps(verifier: FormalStepVerifier) -> None:
    """Arithmetic-consistent steps should produce entailment=True for all.

    Spec: REQ-VERIFY-165, SCENARIO-VERIFY-217
    """
    steps = [
        "There are 5 apples and 3 oranges.",
        "The answer is 42.",
    ]
    result = verifier.verify_chain(steps)
    assert len(result) == 2
    # Step 0: always entailed.
    assert result[0]["entailment"] is True
    # Step 1: no parseable arithmetic in prior steps means Z3 cannot add premises,
    # so the step is labelled "unparseable" or "correct" — either way, entailment=True.
    assert result[1]["entailment"] is True


def test_verify_chain_returns_all_fields(verifier: FormalStepVerifier) -> None:
    """Each result dict must contain step_idx, verdict, and entailment.

    Spec: REQ-VERIFY-165
    """
    steps = ["2 + 2 = 4", "4 + 1 = 5"]
    result = verifier.verify_chain(steps)
    for r in result:
        assert "step_idx" in r
        assert "verdict" in r
        assert "entailment" in r
        assert isinstance(r["step_idx"], int)
        assert isinstance(r["verdict"], str)
        assert isinstance(r["entailment"], bool)


def test_verify_chain_violation_detected(verifier: FormalStepVerifier) -> None:
    """A step that contradicts prior steps should produce entailment=False.

    47 + 28 = 65 is arithmetically wrong (47 + 28 = 75), so Z3 should detect
    the violation when "47 + 28 = 65" follows a step asserting "47 + 28 = 75".

    Spec: REQ-VERIFY-165, SCENARIO-VERIFY-218
    """
    steps = [
        "47 + 28 = 75",   # step 0: correct, no prior
        "47 + 28 = 65",   # step 1: contradicts step 0 — Z3 should say "violation"
    ]
    result = verifier.verify_chain(steps)
    assert len(result) == 2
    # Step 0 is always entailed.
    assert result[0]["entailment"] is True
    # Step 1 should be a violation (Z3 cannot satisfy both claims simultaneously).
    assert result[1]["verdict"] == "violation"
    assert result[1]["entailment"] is False


def test_verify_chain_multiple_steps(verifier: FormalStepVerifier) -> None:
    """verify_chain with 3 steps produces 3 results in order.

    Spec: REQ-VERIFY-165, SCENARIO-VERIFY-217
    """
    steps = ["a = 1", "b = 2", "c = 3"]
    result = verifier.verify_chain(steps)
    assert len(result) == 3
    assert [r["step_idx"] for r in result] == [0, 1, 2]


# ---------------------------------------------------------------------------
# chain_correct() tests
# ---------------------------------------------------------------------------

# SCENARIO-VERIFY-217: empty chain is trivially correct.
def test_chain_correct_empty(verifier: FormalStepVerifier) -> None:
    """chain_correct([]) must return True — empty chain has no violations.

    Spec: SCENARIO-VERIFY-217
    """
    assert verifier.chain_correct([]) is True


def test_chain_correct_single_step(verifier: FormalStepVerifier) -> None:
    """A single-step chain with no prior contradictions is correct.

    Spec: REQ-VERIFY-165
    """
    assert verifier.chain_correct(["2 + 2 = 4"]) is True


def test_chain_correct_all_entailed(verifier: FormalStepVerifier) -> None:
    """A chain where all steps are unparseable returns True (conservative labelling).

    Steps with no extractable arithmetic cannot contradict anything.

    Spec: REQ-VERIFY-165
    """
    steps = [
        "We have some apples.",
        "The answer is somewhere in the range.",
        "Therefore we conclude.",
    ]
    assert verifier.chain_correct(steps) is True


def test_chain_correct_detects_violation(verifier: FormalStepVerifier) -> None:
    """chain_correct returns False when any step violates prior context.

    Spec: REQ-VERIFY-165, SCENARIO-VERIFY-218
    """
    steps = [
        "47 + 28 = 75",
        "47 + 28 = 65",  # contradiction
    ]
    assert verifier.chain_correct(steps) is False
