"""Tests for carnot.extraction.vericot_validator.

All tests use use_mock=True to avoid GPU/LLM dependency in CI.

Every test references a REQ-EXTRACT-* or SCENARIO-EXTRACT-* per
spec-anchored development requirements.

Spec: REQ-EXTRACT-024, REQ-EXTRACT-025, REQ-EXTRACT-026,
      SCENARIO-EXTRACT-049, SCENARIO-EXTRACT-050, SCENARIO-EXTRACT-051
"""

from __future__ import annotations

import pytest
import z3

from carnot.extraction import FOLPremise, StepVerdict, VeriCoTStepValidator
from carnot.extraction.vericot_validator import (
    _mock_extract_expression,
    _split_steps,
)
from carnot.pipeline.extract import ArithmeticExtractor


# ---------------------------------------------------------------------------
# FOLPremise.to_z3_assertion tests
# ---------------------------------------------------------------------------


class TestFOLPremiseToZ3Assertion:
    """REQ-EXTRACT-024: FOLPremise.to_z3_assertion() returns correct Z3 BoolRef."""

    def test_addition_correct_is_sat(self) -> None:
        """SCENARIO-EXTRACT-050: 47 + 28 == 75 is SAT."""
        p = FOLPremise(expression="47 + 28 == 75", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.sat

    def test_addition_wrong_is_unsat(self) -> None:
        """SCENARIO-EXTRACT-049: 47 + 28 == 76 is UNSAT (arithmetic contradiction)."""
        p = FOLPremise(expression="47 + 28 == 76", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.unsat

    def test_subtraction_correct_is_sat(self) -> None:
        """REQ-EXTRACT-024: Subtraction assertion is SAT when correct."""
        p = FOLPremise(expression="100 - 15 == 85", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.sat

    def test_subtraction_wrong_is_unsat(self) -> None:
        """REQ-EXTRACT-024: Subtraction assertion is UNSAT when wrong."""
        p = FOLPremise(expression="100 - 15 == 90", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.unsat

    def test_multiplication_correct_is_sat(self) -> None:
        """REQ-EXTRACT-024: Multiplication assertion is SAT when correct."""
        p = FOLPremise(expression="5 * 6 == 30", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.sat

    def test_multiplication_wrong_is_unsat(self) -> None:
        """REQ-EXTRACT-024: Multiplication assertion is UNSAT when wrong."""
        p = FOLPremise(expression="5 * 6 == 31", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.unsat

    def test_division_correct_is_sat(self) -> None:
        """REQ-EXTRACT-024: Division assertion is SAT when correct."""
        p = FOLPremise(expression="20 / 4 == 5", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.sat

    def test_division_by_zero_returns_none(self) -> None:
        """REQ-EXTRACT-024: Division by zero returns None (undefinable in integer Z3)."""
        p = FOLPremise(expression="10 / 0 == 5", source_step="test")
        assert p.to_z3_assertion() is None

    def test_malformed_expression_returns_none(self) -> None:
        """REQ-EXTRACT-024: Unparseable expression returns None without crashing."""
        p = FOLPremise(expression="the answer is 42", source_step="test")
        assert p.to_z3_assertion() is None

    def test_empty_expression_returns_none(self) -> None:
        """REQ-EXTRACT-024: Empty expression returns None."""
        p = FOLPremise(expression="", source_step="test")
        assert p.to_z3_assertion() is None

    def test_negative_operand_is_handled(self) -> None:
        """REQ-EXTRACT-024: Negative integer operands produce correct SAT result."""
        p = FOLPremise(expression="-3 + 10 == 7", source_step="test")
        assertion = p.to_z3_assertion()
        assert assertion is not None
        s = z3.Solver()
        s.add(assertion)
        assert s.check() == z3.sat


# ---------------------------------------------------------------------------
# FOLPremise dataclass tests
# ---------------------------------------------------------------------------


class TestFOLPremiseDataclass:
    """REQ-EXTRACT-024: FOLPremise stores expression and source_step."""

    def test_fields_stored(self) -> None:
        p = FOLPremise(expression="1 + 1 == 2", source_step="one plus one gives two")
        assert p.expression == "1 + 1 == 2"
        assert p.source_step == "one plus one gives two"


# ---------------------------------------------------------------------------
# StepVerdict dataclass tests
# ---------------------------------------------------------------------------


class TestStepVerdictDataclass:
    """REQ-EXTRACT-025: StepVerdict stores all required fields."""

    def test_fields_stored(self) -> None:
        """SCENARIO-EXTRACT-051: StepVerdict fields accessible."""
        p = FOLPremise(expression="2 + 2 == 4", source_step="two plus two gives four")
        v = StepVerdict(
            step_idx=0,
            step_text="two plus two gives four",
            status="sat",
            fol_premises=[p],
        )
        assert v.step_idx == 0
        assert v.step_text == "two plus two gives four"
        assert v.status == "sat"
        assert len(v.fol_premises) == 1

    def test_to_dict_serializable(self) -> None:
        """REQ-EXTRACT-025: StepVerdict.to_dict() returns JSON-serializable dict."""
        import json

        p = FOLPremise(expression="3 + 3 == 6", source_step="step")
        v = StepVerdict(step_idx=1, step_text="step", status="sat", fol_premises=[p])
        d = v.to_dict()
        json.dumps(d)  # must not raise
        assert d["status"] == "sat"
        assert d["step_idx"] == 1
        assert len(d["fol_premises"]) == 1

    def test_default_fol_premises_is_empty_list(self) -> None:
        """REQ-EXTRACT-025: Default fol_premises is empty list."""
        v = StepVerdict(step_idx=0, step_text="x", status="unknown")
        assert v.fol_premises == []


# ---------------------------------------------------------------------------
# _mock_extract_expression helper tests
# ---------------------------------------------------------------------------


class TestMockExtractExpression:
    """REQ-EXTRACT-024: _mock_extract_expression handles common IT arithmetic prose."""

    def test_plus_gives_75(self) -> None:
        expr = _mock_extract_expression("the total is 47 plus 28, which gives 75")
        assert expr == "47 + 28 == 75"

    def test_plus_gives_76(self) -> None:
        expr = _mock_extract_expression("the total is 47 plus 28, which gives 76")
        assert expr == "47 + 28 == 76"

    def test_times_gives_30(self) -> None:
        expr = _mock_extract_expression("5 times 6 gives us 30")
        assert expr == "5 * 6 == 30"

    def test_subtracted_from_gives(self) -> None:
        expr = _mock_extract_expression("subtracting 15 from 100 gives 85")
        assert expr == "100 - 15 == 85"

    def test_minus_gives(self) -> None:
        expr = _mock_extract_expression("200 minus 50 gives 150")
        assert expr == "200 - 50 == 150"

    def test_multiplied_by_gives(self) -> None:
        expr = _mock_extract_expression("7 multiplied by 8 gives 56")
        assert expr == "7 * 8 == 56"

    def test_divided_by_gives(self) -> None:
        expr = _mock_extract_expression("100 divided by 4 gives 25")
        assert expr == "100 / 4 == 25"

    def test_no_match_returns_none(self) -> None:
        expr = _mock_extract_expression("the sky is blue today")
        assert expr is None

    def test_no_result_clause_returns_none(self) -> None:
        """If we find the op but not the result, return None."""
        expr = _mock_extract_expression("47 plus 28 is the total")
        # "is the total" — "total" is not a number, so no match
        assert expr is None


# ---------------------------------------------------------------------------
# _split_steps helper tests
# ---------------------------------------------------------------------------


class TestSplitSteps:
    """REQ-EXTRACT-025: CoT text is split into individual steps."""

    def test_newline_split(self) -> None:
        steps = _split_steps("step one\nstep two\nstep three")
        assert len(steps) == 3

    def test_sentence_split(self) -> None:
        steps = _split_steps("First we add. Then we multiply. Finally we divide.")
        assert len(steps) == 3

    def test_comma_not_split(self) -> None:
        """Commas within a step are NOT split boundaries (preserves mid-claim commas)."""
        steps = _split_steps("47 plus 28, which gives 75")
        assert len(steps) == 1

    def test_empty_string_returns_empty(self) -> None:
        assert _split_steps("") == []

    def test_single_step(self) -> None:
        steps = _split_steps("only one step here")
        assert steps == ["only one step here"]

    def test_strips_whitespace(self) -> None:
        steps = _split_steps("  step one  \n  step two  ")
        assert steps[0] == "step one"
        assert steps[1] == "step two"


# ---------------------------------------------------------------------------
# VeriCoTStepValidator.verify_step tests (mock mode)
# ---------------------------------------------------------------------------


class TestVerifyStepMock:
    """REQ-EXTRACT-025: verify_step() returns correct status in mock mode."""

    def setup_method(self) -> None:
        self.validator = VeriCoTStepValidator(use_mock=True)

    def test_correct_addition_is_sat(self) -> None:
        """SCENARIO-EXTRACT-050: 'gives 75' (correct) → status='sat'."""
        verdict = self.validator.verify_step(
            "the total is 47 plus 28, which gives 75"
        )
        assert verdict.status == "sat"

    def test_wrong_addition_is_unsat(self) -> None:
        """SCENARIO-EXTRACT-049: 'gives 76' (wrong) → status='unsat'."""
        verdict = self.validator.verify_step(
            "the total is 47 plus 28, which gives 76"
        )
        assert verdict.status == "unsat"

    def test_unsat_verdict_has_premises(self) -> None:
        """REQ-EXTRACT-024: UNSAT verdict preserves the extracted premises."""
        verdict = self.validator.verify_step(
            "the total is 47 plus 28, which gives 76"
        )
        assert len(verdict.fol_premises) >= 1

    def test_no_arithmetic_step_is_unknown(self) -> None:
        """REQ-EXTRACT-025: Step with no arithmetic claim returns status='unknown'."""
        verdict = self.validator.verify_step("The sky is blue and the sun is bright.")
        assert verdict.status == "unknown"
        assert verdict.fol_premises == []

    def test_correct_multiplication_is_sat(self) -> None:
        """SCENARIO-EXTRACT-050: Correct multiplication returns SAT."""
        verdict = self.validator.verify_step("5 times 6 gives us 30")
        assert verdict.status == "sat"

    def test_wrong_multiplication_is_unsat(self) -> None:
        """SCENARIO-EXTRACT-049: Wrong multiplication returns UNSAT."""
        verdict = self.validator.verify_step("5 times 6 gives us 31")
        assert verdict.status == "unsat"

    def test_correct_subtraction_is_sat(self) -> None:
        """SCENARIO-EXTRACT-050: Correct subtraction returns SAT."""
        verdict = self.validator.verify_step("subtracting 15 from 100 gives 85")
        assert verdict.status == "sat"

    def test_wrong_subtraction_is_unsat(self) -> None:
        """SCENARIO-EXTRACT-049: Wrong subtraction returns UNSAT."""
        verdict = self.validator.verify_step("subtracting 15 from 100 gives 90")
        assert verdict.status == "unsat"

    def test_step_idx_is_zero(self) -> None:
        """REQ-EXTRACT-025: verify_step() uses step_idx=0."""
        verdict = self.validator.verify_step("47 plus 3 gives 50")
        assert verdict.step_idx == 0

    def test_step_text_preserved(self) -> None:
        """REQ-EXTRACT-025: verify_step() preserves step_text in verdict."""
        text = "47 plus 3 gives 50"
        verdict = self.validator.verify_step(text)
        assert verdict.step_text == text


# ---------------------------------------------------------------------------
# VeriCoTStepValidator.detect_violations tests (mock mode)
# ---------------------------------------------------------------------------


class TestDetectViolationsMock:
    """REQ-EXTRACT-025: detect_violations() returns only UNSAT verdicts."""

    def setup_method(self) -> None:
        self.validator = VeriCoTStepValidator(use_mock=True)

    def test_correct_cot_has_no_violations(self) -> None:
        """SCENARIO-EXTRACT-050: Correct CoT returns empty violation list."""
        cot = "47 plus 28 gives 75\n5 times 6 gives us 30"
        violations = self.validator.detect_violations(cot)
        assert violations == []

    def test_wrong_step_detected(self) -> None:
        """SCENARIO-EXTRACT-049: Wrong arithmetic step is detected as violation."""
        cot = "47 plus 28 gives 76"
        violations = self.validator.detect_violations(cot)
        assert len(violations) == 1
        assert violations[0].status == "unsat"

    def test_mixed_cot_only_wrong_steps_returned(self) -> None:
        """REQ-EXTRACT-026: Only UNSAT steps returned from mixed CoT."""
        cot = "47 plus 28 gives 75\n5 times 6 gives us 31"
        violations = self.validator.detect_violations(cot)
        assert len(violations) == 1
        assert "5" in violations[0].step_text

    def test_pure_prose_no_violations(self) -> None:
        """REQ-EXTRACT-025: Pure prose CoT (no arithmetic) returns no violations."""
        cot = "The user asked about colors.\nBlue is calming.\nRed is warm."
        violations = self.validator.detect_violations(cot)
        assert violations == []

    def test_step_indices_are_correct(self) -> None:
        """REQ-EXTRACT-025: step_idx reflects position in the split CoT."""
        cot = "first step\n47 plus 28 gives 76"
        violations = self.validator.detect_violations(cot)
        assert len(violations) == 1
        assert violations[0].step_idx == 1

    def test_multiple_violations_detected(self) -> None:
        """REQ-EXTRACT-026: Multiple wrong steps all detected."""
        cot = "47 plus 28 gives 76\n5 times 6 gives us 31"
        violations = self.validator.detect_violations(cot)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# Comparison: ArithmeticExtractor vs VeriCoTStepValidator on IT natural language
# ---------------------------------------------------------------------------


class TestArithmeticVsVeriCoT:
    """REQ-EXTRACT-026: VeriCoTStepValidator detects what ArithmeticExtractor misses."""

    IT_WRONG_SAMPLE = "the total is 47 plus 28, which gives 76"

    def test_arithmetic_extractor_finds_zero_on_it_text(self) -> None:
        """SCENARIO-EXTRACT-051: ArithmeticExtractor gets 0 violations on IT prose."""
        extractor = ArithmeticExtractor()
        results = extractor.extract(self.IT_WRONG_SAMPLE)
        assert results == [], (
            f"ArithmeticExtractor should find 0 violations on IT natural language, "
            f"got: {results}"
        )

    def test_vericot_finds_violation_on_same_it_text(self) -> None:
        """SCENARIO-EXTRACT-049: VeriCoTStepValidator detects violation ArithmeticExtractor misses."""
        validator = VeriCoTStepValidator(use_mock=True)
        violations = validator.detect_violations(self.IT_WRONG_SAMPLE)
        assert len(violations) >= 1, (
            "VeriCoTStepValidator should detect at least one violation in IT prose "
            "with arithmetic error, got none."
        )

    def test_vericot_no_false_positive_on_correct_it_text(self) -> None:
        """SCENARIO-EXTRACT-050: VeriCoTStepValidator does not flag correct IT prose."""
        validator = VeriCoTStepValidator(use_mock=True)
        correct = "the total is 47 plus 28, which gives 75"
        violations = validator.detect_violations(correct)
        assert violations == [], (
            f"VeriCoTStepValidator should not flag correct arithmetic, got: {violations}"
        )


# ---------------------------------------------------------------------------
# VeriCoTStepValidator constructor tests
# ---------------------------------------------------------------------------


class TestValidatorConstructor:
    """REQ-EXTRACT-024: VeriCoTStepValidator stores configuration."""

    def test_default_extractor_llm(self) -> None:
        v = VeriCoTStepValidator(use_mock=True)
        assert v.extractor_llm == "Qwen/Qwen3.5-0.8B"

    def test_custom_extractor_llm(self) -> None:
        v = VeriCoTStepValidator(extractor_llm="custom/model", use_mock=True)
        assert v.extractor_llm == "custom/model"

    def test_use_mock_false_stored(self) -> None:
        v = VeriCoTStepValidator(use_mock=False)
        assert v.use_mock is False

    def test_use_mock_true_stored(self) -> None:
        v = VeriCoTStepValidator(use_mock=True)
        assert v.use_mock is True

    def test_model_not_loaded_at_init(self) -> None:
        """REQ-EXTRACT-024: Model is not loaded at constructor time (lazy load)."""
        v = VeriCoTStepValidator(use_mock=False)
        assert v._model is None
        assert v._tokenizer is None

    def test_extract_fol_mock_mode(self) -> None:
        """REQ-EXTRACT-024: extract_fol() in mock mode uses rule-based extractor."""
        v = VeriCoTStepValidator(use_mock=True)
        premises = v.extract_fol("47 plus 28 gives 75")
        assert len(premises) == 1
        assert premises[0].expression == "47 + 28 == 75"

    def test_extract_fol_no_match(self) -> None:
        """REQ-EXTRACT-024: extract_fol() returns empty list when no arithmetic found."""
        v = VeriCoTStepValidator(use_mock=True)
        premises = v.extract_fol("the color of the sky is blue")
        assert premises == []
