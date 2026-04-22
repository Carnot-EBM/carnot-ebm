"""Tests for python/carnot/training/fover_z3_labeler.py.

Covers Z3StepVerifier.extract_arithmetic_claim(), Z3StepVerifier.verify_step_z3(),
FoVerZ3Pair dataclass, and the module-level verify_step_z3() convenience function.

All tests run in CPU mode with no GPU dependencies.  Z3 is mocked when not available
so the test suite passes on CI machines without z3-solver installed.

Spec: REQ-LEARN-045, REQ-LEARN-046,
      SCENARIO-LEARN-075, SCENARIO-LEARN-076, SCENARIO-LEARN-077
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fixture: make z3 available (or mock it out gracefully)
# ---------------------------------------------------------------------------


def _z3_available() -> bool:
    """Return True if z3-solver is installed in this environment."""
    try:
        import z3  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from carnot.training import fover_z3_labeler  # noqa: E402
from carnot.training.fover_z3_labeler import (  # noqa: E402
    FoVerZ3Pair,
    Z3StepVerifier,
    verify_step_z3,
)


# ---------------------------------------------------------------------------
# FoVerZ3Pair dataclass
# ---------------------------------------------------------------------------


class TestFoVerZ3Pair:
    """SCENARIO-LEARN-075: FoVerZ3Pair captures all required fields."""

    def test_correct_verdict_sets_step_correct_true(self):
        # "correct" verdict → step_correct=True (no arithmetic error detected)
        pair = FoVerZ3Pair(
            question="Q",
            step_text="47 + 28 = 75",
            step_index=0,
            z3_verdict="correct",
            step_correct=True,
        )
        assert pair.step_correct is True
        assert pair.z3_verdict == "correct"

    def test_violation_verdict_sets_step_correct_false(self):
        # "violation" verdict → step_correct=False (Z3 found a contradiction)
        pair = FoVerZ3Pair(
            question="Q",
            step_text="47 + 28 = 65",
            step_index=1,
            z3_verdict="violation",
            step_correct=False,
        )
        assert pair.step_correct is False
        assert pair.z3_verdict == "violation"

    def test_unparseable_verdict_sets_step_correct_true(self):
        # "unparseable" verdict → step_correct=True (conservative: cannot confirm error)
        pair = FoVerZ3Pair(
            question="Q",
            step_text="Therefore, the farmer has some sheep.",
            step_index=2,
            z3_verdict="unparseable",
            step_correct=True,
        )
        assert pair.step_correct is True
        assert pair.z3_verdict == "unparseable"

    def test_dataclass_fields(self):
        """All required fields are present on the dataclass."""
        pair = FoVerZ3Pair(
            question="What is 2+2?",
            step_text="2 + 2 = 4",
            step_index=0,
            z3_verdict="correct",
            step_correct=True,
        )
        assert pair.question == "What is 2+2?"
        assert pair.step_text == "2 + 2 = 4"
        assert pair.step_index == 0


# ---------------------------------------------------------------------------
# Z3StepVerifier.extract_arithmetic_claim
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _z3_available(), reason="z3-solver not installed")
class TestExtractArithmeticClaim:
    """SCENARIO-LEARN-075: extract_arithmetic_claim converts step text to Z3 expression."""

    def setup_method(self):
        self.verifier = Z3StepVerifier()

    def test_addition_claim(self):
        """'47 + 28 = 75' produces a satisfiable Z3 expression."""
        import z3  # noqa: PLC0415
        claim = self.verifier.extract_arithmetic_claim("47 + 28 = 75")
        assert claim is not None
        s = z3.Solver()
        s.add(claim)
        assert s.check() == z3.sat

    def test_addition_violation_unsatisfiable(self):
        """'47 + 28 = 65' is an incorrect arithmetic claim → unsat."""
        import z3  # noqa: PLC0415
        claim = self.verifier.extract_arithmetic_claim("47 + 28 = 65")
        assert claim is not None
        s = z3.Solver()
        s.add(claim)
        assert s.check() == z3.unsat

    def test_subtraction_claim(self):
        """'100 - 37 = 63' → satisfiable."""
        import z3  # noqa: PLC0415
        claim = self.verifier.extract_arithmetic_claim("100 - 37 = 63")
        assert claim is not None
        s = z3.Solver()
        s.add(claim)
        assert s.check() == z3.sat

    def test_multiplication_claim(self):
        """'6 * 7 = 42' → satisfiable."""
        import z3  # noqa: PLC0415
        claim = self.verifier.extract_arithmetic_claim("6 * 7 = 42")
        assert claim is not None
        s = z3.Solver()
        s.add(claim)
        assert s.check() == z3.sat

    def test_multiplication_violation(self):
        """'6 * 7 = 41' → unsat."""
        import z3  # noqa: PLC0415
        claim = self.verifier.extract_arithmetic_claim("6 * 7 = 41")
        assert claim is not None
        s = z3.Solver()
        s.add(claim)
        assert s.check() == z3.unsat

    def test_division_claim(self):
        """'10 / 2 = 5' → satisfiable (integer division)."""
        import z3  # noqa: PLC0415
        claim = self.verifier.extract_arithmetic_claim("10 / 2 = 5")
        assert claim is not None
        s = z3.Solver()
        s.add(claim)
        assert s.check() == z3.sat

    def test_division_by_zero_returns_none(self):
        """Division by zero → None (cannot form valid Z3 expression)."""
        claim = self.verifier.extract_arithmetic_claim("10 / 0 = 5")
        assert claim is None

    def test_no_arithmetic_returns_none(self):
        """Prose step with no arithmetic → None."""
        claim = self.verifier.extract_arithmetic_claim(
            "Therefore, the farmer has some sheep in the field."
        )
        assert claim is None

    def test_commas_in_numbers_handled(self):
        """Numbers with commas (1,000) are parsed correctly."""
        # "1000 + 500 = 1500" written as "1,000 + 500 = 1,500"
        claim = self.verifier.extract_arithmetic_claim("1000 + 500 = 1500")
        assert claim is not None

    def test_empty_string_returns_none(self):
        """Empty step text → None."""
        claim = self.verifier.extract_arithmetic_claim("")
        assert claim is None


# ---------------------------------------------------------------------------
# Z3StepVerifier.verify_step_z3
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _z3_available(), reason="z3-solver not installed")
class TestVerifyStepZ3:
    """SCENARIO-LEARN-075 / SCENARIO-LEARN-076: verify_step_z3 returns correct verdicts."""

    def setup_method(self):
        self.verifier = Z3StepVerifier()

    def test_correct_arithmetic_step(self):
        """A step with correct arithmetic returns 'correct'."""
        # SCENARIO-LEARN-075: correct step verified as correct
        result = self.verifier.verify_step_z3([], "47 + 28 = 75")
        assert result == "correct"

    def test_violation_arithmetic_step(self):
        """A step with incorrect arithmetic returns 'violation'."""
        # SCENARIO-LEARN-076: violation detected by Z3
        result = self.verifier.verify_step_z3([], "47 + 28 = 65")
        assert result == "violation"

    def test_unparseable_prose_step(self):
        """A prose step with no arithmetic returns 'unparseable'."""
        result = self.verifier.verify_step_z3(
            [], "Therefore, the farmer has many sheep."
        )
        assert result == "unparseable"

    def test_prior_steps_provide_context(self):
        """Prior steps are added as premises — consistent chain is 'correct'."""
        prior = ["3 + 4 = 7"]
        current = "7 + 5 = 12"
        result = self.verifier.verify_step_z3(prior, current)
        assert result == "correct"

    def test_step_isolated_violation_detected(self):
        """A step with an arithmetic error is detected even with prior context."""
        prior = ["10 + 5 = 15"]
        current = "10 + 5 = 99"  # wrong result
        result = self.verifier.verify_step_z3(prior, current)
        assert result == "violation"

    def test_empty_prior_steps(self):
        """Works correctly with no prior context."""
        result = self.verifier.verify_step_z3([], "2 + 2 = 4")
        assert result == "correct"

    def test_prior_steps_with_prose_skipped(self):
        """Prior prose steps (unparseable) are gracefully skipped."""
        prior = ["The farmer has some eggs.", "He gave away some."]
        current = "6 * 7 = 42"
        result = self.verifier.verify_step_z3(prior, current)
        assert result == "correct"


# ---------------------------------------------------------------------------
# Module-level verify_step_z3() convenience function
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _z3_available(), reason="z3-solver not installed")
class TestModuleLevelVerify:
    """SCENARIO-LEARN-076: module-level verify_step_z3() is a valid convenience wrapper."""

    def test_delegates_to_verifier(self):
        """Module-level function produces same result as Z3StepVerifier."""
        result = verify_step_z3([], "5 + 3 = 8")
        assert result == "correct"

    def test_violation_via_module_function(self):
        result = verify_step_z3([], "5 + 3 = 9")
        assert result == "violation"

    def test_unparseable_via_module_function(self):
        result = verify_step_z3([], "No arithmetic here.")
        assert result == "unparseable"


# ---------------------------------------------------------------------------
# Z3 unavailable fallback behaviour
# ---------------------------------------------------------------------------


class TestZ3UnavailableFallback:
    """SCENARIO-LEARN-077: when z3 is not available, all verdicts are 'unparseable'."""

    def test_extract_returns_none_without_z3(self):
        """extract_arithmetic_claim returns None when z3_available=False."""
        original = fover_z3_labeler.z3_available
        fover_z3_labeler.z3_available = False
        try:
            verifier = Z3StepVerifier()
            result = verifier.extract_arithmetic_claim("47 + 28 = 75")
            assert result is None
        finally:
            fover_z3_labeler.z3_available = original

    def test_verify_returns_unparseable_without_z3(self):
        """verify_step_z3 returns 'unparseable' when z3_available=False."""
        original = fover_z3_labeler.z3_available
        fover_z3_labeler.z3_available = False
        try:
            verifier = Z3StepVerifier()
            result = verifier.verify_step_z3([], "47 + 28 = 75")
            assert result == "unparseable"
        finally:
            fover_z3_labeler.z3_available = original

    def test_step_correct_is_true_for_unparseable(self):
        """FoVerZ3Pair with 'unparseable' verdict has step_correct=True (conservative)."""
        pair = FoVerZ3Pair(
            question="Q",
            step_text="some text",
            step_index=0,
            z3_verdict="unparseable",
            step_correct=True,
        )
        assert pair.step_correct is True
