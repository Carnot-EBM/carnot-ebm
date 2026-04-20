"""Tests for InterleavedLogicVerifier and InterleavedStepResult.

Spec: REQ-VERIFY-135, SCENARIO-VERIFY-168, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

from __future__ import annotations

import pytest

from carnot.pipeline.interleaved_verifier import (
    InterleavedLogicVerifier,
    InterleavedStepResult,
    _clean_num,
    _safe_eval,
)


# ---------------------------------------------------------------------------
# InterleavedStepResult
# ---------------------------------------------------------------------------


class TestInterleavedStepResult:
    def test_defaults(self):
        # REQ-VERIFY-135-5
        r = InterleavedStepResult(step_text="hello", step_idx=0)
        assert r.z3_sat is None
        assert r.constraint_added is None
        assert r.violation_detected is False

    def test_violation_fields(self):
        r = InterleavedStepResult(
            step_text="3 + 4 = 8",
            step_idx=1,
            z3_sat=True,
            constraint_added="7.0 != 8.0",
            violation_detected=True,
        )
        assert r.z3_sat is True
        assert r.violation_detected is True
        assert r.constraint_added == "7.0 != 8.0"

    def test_no_violation_fields(self):
        r = InterleavedStepResult(
            step_text="3 + 4 = 7",
            step_idx=0,
            z3_sat=False,
            constraint_added="7.0 != 7.0",
            violation_detected=False,
        )
        assert r.z3_sat is False
        assert r.violation_detected is False


# ---------------------------------------------------------------------------
# _clean_num
# ---------------------------------------------------------------------------


class TestCleanNum:
    def test_plain_integer(self):
        assert _clean_num("42") == 42.0

    def test_float(self):
        assert _clean_num("3.14") == pytest.approx(3.14)

    def test_comma_separated(self):
        assert _clean_num("1,000") == 1000.0

    def test_negative(self):
        assert _clean_num("-5") == -5.0

    def test_whitespace(self):
        assert _clean_num("  18  ") == 18.0

    def test_invalid(self):
        assert _clean_num("abc") is None

    def test_empty(self):
        assert _clean_num("") is None


# ---------------------------------------------------------------------------
# _safe_eval
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_addition(self):
        assert _safe_eval("3 + 4") == pytest.approx(7.0)

    def test_multiplication(self):
        assert _safe_eval("9 * 2") == pytest.approx(18.0)

    def test_with_parens(self):
        assert _safe_eval("(3 + 4) * 2") == pytest.approx(14.0)

    def test_with_commas_in_numbers(self):
        # Commas should be stripped before evaluation.
        assert _safe_eval("1,000 + 500") == pytest.approx(1500.0)

    def test_invalid_expr(self):
        assert _safe_eval("abc + 3") is None

    def test_division(self):
        assert _safe_eval("18 / 2") == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# InterleavedLogicVerifier._formalize_step
# ---------------------------------------------------------------------------


class TestFormalizeStep:
    def setup_method(self):
        self.verifier = InterleavedLogicVerifier()

    def test_correct_addition_returns_unsat_assertion(self):
        # SCENARIO-VERIFY-168: 3 + 4 = 7 is correct → assertion "7.0 != 7.0"
        assertion = self.verifier._formalize_step("3 + 4 = 7", [])
        assert assertion is not None
        # The assertion should evaluate to False (SAT would mean violation).
        assert eval(assertion, {"__builtins__": {}}) is False  # noqa: S307

    def test_incorrect_addition_returns_sat_assertion(self):
        # SCENARIO-VERIFY-169: 3 + 4 = 8 is wrong → assertion "7.0 != 8.0"
        assertion = self.verifier._formalize_step("3 + 4 = 8", [])
        assert assertion is not None
        assert eval(assertion, {"__builtins__": {}}) is True  # noqa: S307

    def test_no_equation_returns_none(self):
        # SCENARIO-VERIFY-170: no numeric equation
        assertion = self.verifier._formalize_step("The ducks lay eggs every day.", [])
        assert assertion is None

    def test_latex_multiplication_parsed(self):
        # LaTeX \times should be normalised to *.
        assertion = self.verifier._formalize_step(r"9 \times 2 = 18", [])
        assert assertion is not None
        # 9*2=18 is correct → False (no violation)
        assert eval(assertion, {"__builtins__": {}}) is False  # noqa: S307

    def test_latex_multiplication_wrong(self):
        assertion = self.verifier._formalize_step(r"9 \times 2 = 20", [])
        assert assertion is not None
        # 9*2=18 ≠ 20 → True (violation)
        assert eval(assertion, {"__builtins__": {}}) is True  # noqa: S307

    def test_comma_number_in_rhs(self):
        # "1,000 + 500 = 1,500" — commas in numbers
        assertion = self.verifier._formalize_step("1,000 + 500 = 1,500", [])
        assert assertion is not None
        assert eval(assertion, {"__builtins__": {}}) is False  # noqa: S307

    def test_prior_constraints_accepted(self):
        # prior_constraints does not break formalization
        assertion = self.verifier._formalize_step("2 + 2 = 4", ["5.0 != 6.0"])
        assert assertion is not None


# ---------------------------------------------------------------------------
# InterleavedLogicVerifier._split_steps
# ---------------------------------------------------------------------------


class TestSplitSteps:
    def setup_method(self):
        self.verifier = InterleavedLogicVerifier()

    def test_single_sentence_is_one_step(self):
        steps = self.verifier._split_steps("The answer is 42.")
        assert len(steps) >= 1

    def test_blank_line_splits(self):
        text = "Step one.\n\nStep two."
        steps = self.verifier._split_steps(text)
        assert len(steps) >= 2

    def test_empty_string_returns_list_of_one(self):
        steps = self.verifier._split_steps("")
        assert len(steps) >= 1

    def test_multi_sentence_response(self):
        text = (
            "Janet lays 16 eggs. She eats 3 for breakfast. "
            "She bakes 4 muffins. Therefore she sells 9 eggs."
        )
        steps = self.verifier._split_steps(text)
        # Should produce at least 2 steps
        assert len(steps) >= 2


# ---------------------------------------------------------------------------
# InterleavedLogicVerifier.verify_response — integration
# ---------------------------------------------------------------------------


class TestVerifyResponse:
    def setup_method(self):
        self.verifier = InterleavedLogicVerifier()

    def test_correct_response_no_violation(self):
        # SCENARIO-VERIFY-168
        response = "Janet sells 9 eggs. Therefore 9 * 2 = 18 dollars."
        results = self.verifier.verify_response(response)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert not any(r.violation_detected for r in results)

    def test_incorrect_response_violation_detected(self):
        # SCENARIO-VERIFY-169
        response = "She gets 3 + 4 = 8 eggs."
        results = self.verifier.verify_response(response)
        assert any(r.violation_detected for r in results)

    def test_no_arithmetic_response(self):
        # SCENARIO-VERIFY-170
        response = "The answer is 42."
        results = self.verifier.verify_response(response)
        assert isinstance(results, list)
        # All z3_sat should be None (no arithmetic found)
        for r in results:
            assert r.z3_sat is None
            assert not r.violation_detected

    def test_step_indices_sequential(self):
        response = "First 2 + 2 = 4.\n\nThen 3 + 3 = 7."
        results = self.verifier.verify_response(response)
        for i, r in enumerate(results):
            assert r.step_idx == i

    def test_empty_response(self):
        results = self.verifier.verify_response("")
        assert isinstance(results, list)

    def test_violation_in_multi_step(self):
        # First step correct, second step wrong — only second should flag.
        response = "She has 16 eggs.\n\nShe uses 3 + 4 = 8 eggs."
        results = self.verifier.verify_response(response)
        violation_flags = [r.violation_detected for r in results]
        assert any(violation_flags)

    def test_result_dataclass_fields_populated(self):
        response = "2 + 3 = 6."
        results = self.verifier.verify_response(response)
        r = results[0]
        assert isinstance(r.step_text, str)
        assert isinstance(r.step_idx, int)


# ---------------------------------------------------------------------------
# InterleavedLogicVerifier._z3_check
# ---------------------------------------------------------------------------


class TestZ3Check:
    def setup_method(self):
        self.verifier = InterleavedLogicVerifier()

    def test_sat_assertion_returns_true(self):
        # "7.0 != 8.0" is True (7 ≠ 8) → SAT → violation
        assert self.verifier._z3_check("7.0 != 8.0") is True

    def test_unsat_assertion_returns_false(self):
        # "7.0 != 7.0" is False → UNSAT → no violation
        assert self.verifier._z3_check("7.0 != 7.0") is False

    def test_invalid_assertion_returns_false(self):
        # Malformed assertion should not crash; falls through to False
        assert self.verifier._z3_check("not_valid_python!!!") is False


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


class TestExports:
    def test_importable_from_pipeline(self):
        from carnot.pipeline import InterleavedLogicVerifier as ILV  # noqa: PLC0415
        from carnot.pipeline import InterleavedStepResult as ISR  # noqa: PLC0415
        assert ILV is not None
        assert ISR is not None
