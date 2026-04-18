"""Tests for carnot.extraction.vprm_verifier — VPRM rule-based arithmetic verifier.

100% branch coverage required (REQ-EXTRACT-027/028/029).
Every test references a REQ-EXTRACT-* or SCENARIO-EXTRACT-* identifier.

Spec: REQ-EXTRACT-027, REQ-EXTRACT-028, REQ-EXTRACT-029,
      SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053, SCENARIO-EXTRACT-054
"""

from __future__ import annotations

import pytest

from carnot.extraction import VPRMArithmeticVerifier, ArithmeticRule, RuleVerdict
from carnot.extraction.vprm_verifier import (
    AdditionRule,
    SubtractionRule,
    MultiplicationRule,
    DivisionRule,
    PercentageRule,
    UnitConsistencyRule,
    _split_steps,
    _is_close,
)
from carnot.pipeline.extract import ArithmeticExtractor


# ---------------------------------------------------------------------------
# RuleVerdict dataclass
# ---------------------------------------------------------------------------


class TestRuleVerdict:
    """REQ-EXTRACT-027: RuleVerdict stores all required fields."""

    def test_fields_stored(self) -> None:
        """SCENARIO-EXTRACT-052: RuleVerdict fields are accessible."""
        v = RuleVerdict(
            rule_name="addition",
            passed=True,
            computed_value=75.0,
            stated_value=75.0,
            error_magnitude=0.0,
        )
        assert v.rule_name == "addition"
        assert v.passed is True
        assert v.computed_value == 75.0
        assert v.stated_value == 75.0
        assert v.error_magnitude == 0.0

    def test_none_fields_allowed(self) -> None:
        """REQ-EXTRACT-027: Optional fields may be None (e.g. unit_consistency rule)."""
        v = RuleVerdict(
            rule_name="unit_consistency",
            passed=False,
            computed_value=None,
            stated_value=None,
            error_magnitude=None,
        )
        assert v.computed_value is None
        assert v.stated_value is None
        assert v.error_magnitude is None


# ---------------------------------------------------------------------------
# _is_close helper
# ---------------------------------------------------------------------------


class TestIsClose:
    """Internal: _is_close handles integer equality and float near-equality."""

    def test_exact_equal(self) -> None:
        assert _is_close(75.0, 75.0)

    def test_near_equal(self) -> None:
        assert _is_close(10.0, 10.0 + 1e-10)

    def test_not_equal(self) -> None:
        assert not _is_close(75.0, 76.0)

    def test_near_zero(self) -> None:
        assert _is_close(0.0, 1e-12)

    def test_large_values(self) -> None:
        assert _is_close(1_000_000.0, 1_000_000.0 + 0.5)

    def test_large_values_not_close(self) -> None:
        assert not _is_close(1_000_000.0, 1_000_001.5)


# ---------------------------------------------------------------------------
# AdditionRule
# ---------------------------------------------------------------------------


class TestAdditionRule:
    """REQ-EXTRACT-027: AdditionRule detects addition errors in IT prose."""

    def setup_method(self) -> None:
        self.rule = AdditionRule()

    def test_correct_addition_passes(self) -> None:
        """SCENARIO-EXTRACT-052: '47 plus 28 equals 75' → passed=True."""
        v = self.rule.check("47 plus 28 equals 75")
        assert v is not None
        assert v.passed is True
        assert v.computed_value == 75.0
        assert v.stated_value == 75.0
        assert v.rule_name == "addition"

    def test_wrong_addition_fails(self) -> None:
        """SCENARIO-EXTRACT-052: '47 plus 28 equals 76' (76 ≠ 75) → passed=False."""
        v = self.rule.check("47 plus 28 equals 76")
        assert v is not None
        assert v.passed is False
        assert v.computed_value == 75.0
        assert v.stated_value == 76.0
        assert v.error_magnitude == 1.0

    def test_added_to_variant(self) -> None:
        """REQ-EXTRACT-027: 'added to' phrasing is also matched."""
        v = self.rule.check("100 added to 50 gives 150")
        assert v is not None
        assert v.passed is True

    def test_gives_us_variant(self) -> None:
        """REQ-EXTRACT-027: 'gives us' result clause is matched."""
        v = self.rule.check("10 plus 5 gives us 15")
        assert v is not None
        assert v.passed is True

    def test_no_match_returns_none(self) -> None:
        """REQ-EXTRACT-027: Step without addition returns None."""
        v = self.rule.check("the sky is blue today")
        assert v is None

    def test_match_without_result_returns_none(self) -> None:
        """REQ-EXTRACT-027: 'plus' found but no numeric result clause → None."""
        v = self.rule.check("47 plus 28 is the total amount here")
        assert v is None

    def test_negative_operand(self) -> None:
        """REQ-EXTRACT-027: Negative operands are handled."""
        v = self.rule.check("-3 plus 10 equals 7")
        assert v is not None
        assert v.passed is True

    def test_is_result_clause(self) -> None:
        """REQ-EXTRACT-027: 'is N' result clause is recognised."""
        v = self.rule.check("5 plus 3 is 8")
        assert v is not None
        assert v.passed is True


# ---------------------------------------------------------------------------
# SubtractionRule
# ---------------------------------------------------------------------------


class TestSubtractionRule:
    """REQ-EXTRACT-027: SubtractionRule detects subtraction errors in IT prose."""

    def setup_method(self) -> None:
        self.rule = SubtractionRule()

    def test_correct_minus_passes(self) -> None:
        """SCENARIO-EXTRACT-052: '100 minus 15 gives 85' → passed=True."""
        v = self.rule.check("100 minus 15 gives 85")
        assert v is not None
        assert v.passed is True
        assert v.computed_value == 85.0

    def test_wrong_minus_fails(self) -> None:
        """SCENARIO-EXTRACT-052: '100 minus 15 gives 90' → passed=False."""
        v = self.rule.check("100 minus 15 gives 90")
        assert v is not None
        assert v.passed is False
        assert v.error_magnitude == 5.0

    def test_subtracting_from_correct(self) -> None:
        """REQ-EXTRACT-027: 'subtracting B from A' phrasing — operand order is swapped."""
        v = self.rule.check("subtracting 15 from 100 gives 85")
        assert v is not None
        assert v.passed is True

    def test_subtracting_from_wrong(self) -> None:
        """REQ-EXTRACT-027: 'subtracting B from A' with wrong result → passed=False."""
        v = self.rule.check("subtracting 15 from 100 gives 90")
        assert v is not None
        assert v.passed is False

    def test_subtracted_by_variant(self) -> None:
        """REQ-EXTRACT-027: 'subtracted by' phrasing is matched."""
        v = self.rule.check("50 subtracted by 20 equals 30")
        assert v is not None
        assert v.passed is True

    def test_no_match_returns_none(self) -> None:
        """REQ-EXTRACT-027: Step without subtraction returns None."""
        v = self.rule.check("the total is 47 plus 28 gives 75")
        assert v is None

    def test_minus_without_result_returns_none(self) -> None:
        """REQ-EXTRACT-027: 'minus' found but no numeric result clause → None."""
        v = self.rule.check("100 minus 15 is the remainder")
        assert v is None

    def test_subtracting_without_result_returns_none(self) -> None:
        """REQ-EXTRACT-027: 'subtracting from' found but no numeric result → None."""
        v = self.rule.check("subtracting 15 from 100 is the remainder")
        assert v is None


# ---------------------------------------------------------------------------
# MultiplicationRule
# ---------------------------------------------------------------------------


class TestMultiplicationRule:
    """REQ-EXTRACT-027: MultiplicationRule detects multiplication errors in IT prose."""

    def setup_method(self) -> None:
        self.rule = MultiplicationRule()

    def test_correct_times_passes(self) -> None:
        """SCENARIO-EXTRACT-052: '5 times 6 gives us 30' → passed=True."""
        v = self.rule.check("5 times 6 gives us 30")
        assert v is not None
        assert v.passed is True
        assert v.computed_value == 30.0

    def test_wrong_times_fails(self) -> None:
        """SCENARIO-EXTRACT-052: '5 times 6 gives us 31' → passed=False (30 ≠ 31)."""
        v = self.rule.check("5 times 6 gives us 31")
        assert v is not None
        assert v.passed is False
        assert v.error_magnitude == 1.0

    def test_multiplied_by_variant(self) -> None:
        """REQ-EXTRACT-027: 'multiplied by' phrasing is matched."""
        v = self.rule.check("7 multiplied by 8 equals 56")
        assert v is not None
        assert v.passed is True

    def test_wrong_multiplied_by_fails(self) -> None:
        """REQ-EXTRACT-027: 'multiplied by' with wrong result → passed=False."""
        v = self.rule.check("7 multiplied by 8 equals 57")
        assert v is not None
        assert v.passed is False

    def test_no_match_returns_none(self) -> None:
        """REQ-EXTRACT-027: Step without multiplication returns None."""
        v = self.rule.check("the sky is very blue today")
        assert v is None

    def test_match_without_result_returns_none(self) -> None:
        """REQ-EXTRACT-027: 'times' found but no numeric result clause → None."""
        v = self.rule.check("5 times 6 is the product")
        assert v is None


# ---------------------------------------------------------------------------
# DivisionRule
# ---------------------------------------------------------------------------


class TestDivisionRule:
    """REQ-EXTRACT-027: DivisionRule detects division errors in IT prose."""

    def setup_method(self) -> None:
        self.rule = DivisionRule()

    def test_correct_division_passes(self) -> None:
        """SCENARIO-EXTRACT-052: '100 divided by 4 gives 25' → passed=True."""
        v = self.rule.check("100 divided by 4 gives 25")
        assert v is not None
        assert v.passed is True
        assert v.computed_value == 25.0

    def test_wrong_division_fails(self) -> None:
        """SCENARIO-EXTRACT-052: '100 divided by 4 gives 26' → passed=False."""
        v = self.rule.check("100 divided by 4 gives 26")
        assert v is not None
        assert v.passed is False
        assert v.error_magnitude == 1.0

    def test_division_by_zero_returns_none(self) -> None:
        """REQ-EXTRACT-027: Division by zero is undefined — returns None."""
        v = self.rule.check("10 divided by 0 gives 5")
        assert v is None

    def test_no_match_returns_none(self) -> None:
        """REQ-EXTRACT-027: Step without division returns None."""
        v = self.rule.check("100 plus 4 gives 104")
        assert v is None

    def test_match_without_result_returns_none(self) -> None:
        """REQ-EXTRACT-027: 'divided by' found but no numeric result → None."""
        v = self.rule.check("100 divided by 4 is the quotient")
        assert v is None

    def test_non_integer_division(self) -> None:
        """REQ-EXTRACT-027: Non-integer division result is verified correctly."""
        v = self.rule.check("10 divided by 4 gives 2.5")
        assert v is not None
        assert v.passed is True


# ---------------------------------------------------------------------------
# PercentageRule
# ---------------------------------------------------------------------------


class TestPercentageRule:
    """REQ-EXTRACT-027: PercentageRule detects percentage errors in IT prose."""

    def setup_method(self) -> None:
        self.rule = PercentageRule()

    def test_correct_percentage_passes(self) -> None:
        """SCENARIO-EXTRACT-052: '20% of 50 is 10' → passed=True."""
        v = self.rule.check("20% of 50 is 10")
        assert v is not None
        assert v.passed is True
        assert abs(v.computed_value - 10.0) < 1e-9

    def test_wrong_percentage_fails(self) -> None:
        """SCENARIO-EXTRACT-052: '20% of 50 is 11' (should be 10) → passed=False."""
        v = self.rule.check("20% of 50 is 11")
        assert v is not None
        assert v.passed is False
        assert v.computed_value == 10.0
        assert v.stated_value == 11.0
        assert v.error_magnitude == 1.0

    def test_equals_variant(self) -> None:
        """REQ-EXTRACT-027: 'equals' result clause is matched."""
        v = self.rule.check("15% of 200 equals 30")
        assert v is not None
        assert v.passed is True

    def test_gives_variant(self) -> None:
        """REQ-EXTRACT-027: 'gives' result clause is matched."""
        v = self.rule.check("10% of 100 gives 10")
        assert v is not None
        assert v.passed is True

    def test_gives_us_variant(self) -> None:
        """REQ-EXTRACT-027: 'gives us' result clause is matched."""
        v = self.rule.check("25% of 80 gives us 20")
        assert v is not None
        assert v.passed is True

    def test_no_match_returns_none(self) -> None:
        """REQ-EXTRACT-027: Step without percentage pattern returns None."""
        v = self.rule.check("47 plus 28 gives 75")
        assert v is None

    def test_wrong_large_percentage_fails(self) -> None:
        """REQ-EXTRACT-027: 50% of 200 is 101 (should be 100) → passed=False."""
        v = self.rule.check("50% of 200 is 101")
        assert v is not None
        assert v.passed is False


# ---------------------------------------------------------------------------
# UnitConsistencyRule
# ---------------------------------------------------------------------------


class TestUnitConsistencyRule:
    """REQ-EXTRACT-027: UnitConsistencyRule detects mixed-unit arithmetic."""

    def setup_method(self) -> None:
        self.rule = UnitConsistencyRule()

    def test_consistent_units_pass(self) -> None:
        """REQ-EXTRACT-027: Same units on both operands → passed=True."""
        v = self.rule.check("5 km plus 3 km gives 8 km")
        assert v is not None
        assert v.passed is True

    def test_inconsistent_units_fail(self) -> None:
        """REQ-EXTRACT-027: Different units in same family → passed=False."""
        v = self.rule.check("5 km plus 3 miles gives 8 km")
        assert v is not None
        assert v.passed is False

    def test_no_units_returns_none(self) -> None:
        """REQ-EXTRACT-027: No unit labels → None."""
        v = self.rule.check("5 plus 3 gives 8")
        assert v is None

    def test_unknown_units_return_none(self) -> None:
        """REQ-EXTRACT-027: Unrecognised unit tokens → None (not flagged as error)."""
        v = self.rule.check("5 widgets plus 3 gadgets gives 8 things")
        assert v is None

    def test_different_unit_families_return_none(self) -> None:
        """REQ-EXTRACT-027: kg and km are different families → None (not an error)."""
        v = self.rule.check("5 kg plus 3 km gives 8")
        assert v is None

    def test_mass_units_inconsistent(self) -> None:
        """REQ-EXTRACT-027: kg and lb are in the same mass family → passed=False."""
        v = self.rule.check("5 kg plus 3 lb gives 8 kg")
        assert v is not None
        assert v.passed is False

    def test_computed_and_stated_none_for_unit_rule(self) -> None:
        """REQ-EXTRACT-027: UnitConsistencyRule yields None for computed/stated fields."""
        v = self.rule.check("5 km plus 3 miles gives 8 km")
        assert v is not None
        assert v.computed_value is None
        assert v.stated_value is None
        assert v.error_magnitude is None


# ---------------------------------------------------------------------------
# _split_steps helper
# ---------------------------------------------------------------------------


class TestSplitSteps:
    """Internal: _split_steps splits CoT text into steps."""

    def test_newline_split(self) -> None:
        steps = _split_steps("step one\nstep two")
        assert len(steps) == 2

    def test_sentence_split(self) -> None:
        steps = _split_steps("First we add. Then we multiply.")
        assert len(steps) == 2

    def test_empty_returns_empty(self) -> None:
        assert _split_steps("") == []

    def test_single_step(self) -> None:
        assert _split_steps("only one step") == ["only one step"]


# ---------------------------------------------------------------------------
# VPRMArithmeticVerifier.verify_step
# ---------------------------------------------------------------------------


class TestVerifyStep:
    """REQ-EXTRACT-028: verify_step() applies all rules and returns matched verdicts."""

    def setup_method(self) -> None:
        self.verifier = VPRMArithmeticVerifier()

    def test_correct_addition_returns_passed_verdict(self) -> None:
        """SCENARIO-EXTRACT-052: Correct addition → verdict with passed=True."""
        verdicts = self.verifier.verify_step("47 plus 28 equals 75")
        assert len(verdicts) == 1
        assert verdicts[0].passed is True
        assert verdicts[0].rule_name == "addition"

    def test_wrong_addition_returns_failed_verdict(self) -> None:
        """SCENARIO-EXTRACT-052: Wrong addition → verdict with passed=False."""
        verdicts = self.verifier.verify_step("47 plus 28 equals 76")
        assert len(verdicts) == 1
        assert verdicts[0].passed is False

    def test_no_arithmetic_returns_empty(self) -> None:
        """REQ-EXTRACT-028: Step with no arithmetic claim → empty list."""
        verdicts = self.verifier.verify_step("the sky is blue today")
        assert verdicts == []

    def test_percentage_step(self) -> None:
        """SCENARIO-EXTRACT-052: Percentage step is matched by PercentageRule."""
        verdicts = self.verifier.verify_step("20% of 50 is 11")
        assert len(verdicts) == 1
        assert verdicts[0].rule_name == "percentage"
        assert verdicts[0].passed is False

    def test_multiplication_step(self) -> None:
        """SCENARIO-EXTRACT-052: Multiplication step is matched by MultiplicationRule."""
        verdicts = self.verifier.verify_step("5 times 6 gives us 30")
        assert len(verdicts) == 1
        assert verdicts[0].rule_name == "multiplication"
        assert verdicts[0].passed is True

    def test_custom_rules_only_apply_custom(self) -> None:
        """REQ-EXTRACT-028: Custom rules list overrides defaults."""
        verifier = VPRMArithmeticVerifier(rules=[MultiplicationRule()])
        verdicts = verifier.verify_step("47 plus 28 equals 76")
        # Addition rule not in custom list — should not match
        assert verdicts == []


# ---------------------------------------------------------------------------
# VPRMArithmeticVerifier.detect_violations
# ---------------------------------------------------------------------------


class TestDetectViolations:
    """REQ-EXTRACT-028: detect_violations() returns only failed verdicts."""

    def setup_method(self) -> None:
        self.verifier = VPRMArithmeticVerifier()

    def test_correct_cot_has_no_violations(self) -> None:
        """SCENARIO-EXTRACT-053: All-correct CoT → empty violation list."""
        cot = "47 plus 28 equals 75\n5 times 6 gives us 30"
        assert self.verifier.detect_violations(cot) == []

    def test_wrong_step_detected(self) -> None:
        """SCENARIO-EXTRACT-053: Wrong addition step → one violation."""
        violations = self.verifier.detect_violations("47 plus 28 equals 76")
        assert len(violations) == 1
        assert violations[0].passed is False

    def test_mixed_cot_returns_only_violations(self) -> None:
        """REQ-EXTRACT-028: Correct steps not returned in violation list."""
        cot = "47 plus 28 equals 75\n5 times 6 gives us 31"
        violations = self.verifier.detect_violations(cot)
        assert len(violations) == 1
        assert violations[0].rule_name == "multiplication"

    def test_pure_prose_no_violations(self) -> None:
        """REQ-EXTRACT-028: CoT with no arithmetic patterns → no violations."""
        cot = "The user asked about colors.\nBlue is calming."
        assert self.verifier.detect_violations(cot) == []

    def test_percentage_violation_detected(self) -> None:
        """SCENARIO-EXTRACT-053: Wrong percentage → violation detected."""
        violations = self.verifier.detect_violations("20% of 50 is 11")
        assert len(violations) == 1
        assert violations[0].rule_name == "percentage"

    def test_multiple_violations(self) -> None:
        """REQ-EXTRACT-028: Multiple wrong steps all produce violations."""
        cot = "47 plus 28 equals 76\n5 times 6 gives us 31"
        violations = self.verifier.detect_violations(cot)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# VPRMArithmeticVerifier.f1_score
# ---------------------------------------------------------------------------


class TestF1Score:
    """REQ-EXTRACT-029: f1_score() computes binary F1 correctly."""

    def test_perfect_f1(self) -> None:
        """SCENARIO-EXTRACT-054: Perfect predictions → F1 = 1.0."""
        gt = [True, True, False, False]
        pred = [True, True, False, False]
        assert VPRMArithmeticVerifier.f1_score(gt, pred) == 1.0

    def test_zero_f1_all_wrong(self) -> None:
        """SCENARIO-EXTRACT-054: All predictions wrong → F1 = 0.0."""
        gt = [True, True, False, False]
        pred = [False, False, True, True]
        assert VPRMArithmeticVerifier.f1_score(gt, pred) == 0.0

    def test_zero_f1_no_positives_predicted(self) -> None:
        """SCENARIO-EXTRACT-054: No positives predicted → F1 = 0.0."""
        gt = [True, True, True]
        pred = [False, False, False]
        assert VPRMArithmeticVerifier.f1_score(gt, pred) == 0.0

    def test_zero_f1_no_positives_in_ground_truth(self) -> None:
        """SCENARIO-EXTRACT-054: No positives in ground truth and predicted → F1 = 0.0."""
        gt = [False, False, False]
        pred = [False, False, False]
        assert VPRMArithmeticVerifier.f1_score(gt, pred) == 0.0

    def test_partial_f1(self) -> None:
        """SCENARIO-EXTRACT-054: Partial match gives expected F1."""
        # TP=1, FP=0, FN=1: precision=1.0, recall=0.5, f1=0.667
        gt = [True, True, False]
        pred = [True, False, False]
        score = VPRMArithmeticVerifier.f1_score(gt, pred)
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_f1_mismatched_lengths_raises(self) -> None:
        """REQ-EXTRACT-029: Mismatched list lengths raise ValueError."""
        with pytest.raises(ValueError):
            VPRMArithmeticVerifier.f1_score([True, False], [True])

    def test_empty_lists_returns_zero(self) -> None:
        """REQ-EXTRACT-029: Empty lists return F1 = 0.0."""
        assert VPRMArithmeticVerifier.f1_score([], []) == 0.0


# ---------------------------------------------------------------------------
# VPRMArithmeticVerifier constructor
# ---------------------------------------------------------------------------


class TestVPRMConstructor:
    """REQ-EXTRACT-028: VPRMArithmeticVerifier stores rules correctly."""

    def test_default_rules_are_six(self) -> None:
        """REQ-EXTRACT-027: Default rule set has all six rule families."""
        v = VPRMArithmeticVerifier()
        assert len(v.rules) == 6

    def test_custom_rules_stored(self) -> None:
        """REQ-EXTRACT-028: Custom rules list is stored as provided."""
        custom = [AdditionRule(), MultiplicationRule()]
        v = VPRMArithmeticVerifier(rules=custom)
        assert len(v.rules) == 2
        assert isinstance(v.rules[0], AdditionRule)

    def test_empty_rules_gives_no_verdicts(self) -> None:
        """REQ-EXTRACT-028: Empty rules list → no verdicts for any step."""
        v = VPRMArithmeticVerifier(rules=[])
        assert v.verify_step("47 plus 28 equals 76") == []


# ---------------------------------------------------------------------------
# Comparison: ArithmeticExtractor vs VPRMArithmeticVerifier on IT prose
# ---------------------------------------------------------------------------


class TestArithmeticVsVPRM:
    """REQ-EXTRACT-029: VPRMArithmeticVerifier outperforms ArithmeticExtractor on IT prose."""

    IT_WRONG = "47 plus 28 equals 76"

    def test_arithmetic_extractor_misses_it_prose(self) -> None:
        """SCENARIO-EXTRACT-053: ArithmeticExtractor finds 0 violations on IT-prose format."""
        extractor = ArithmeticExtractor()
        results = extractor.extract(self.IT_WRONG)
        assert results == []

    def test_vprm_finds_it_prose_violation(self) -> None:
        """SCENARIO-EXTRACT-053: VPRMArithmeticVerifier detects error ArithmeticExtractor misses."""
        verifier = VPRMArithmeticVerifier()
        violations = verifier.detect_violations(self.IT_WRONG)
        assert len(violations) >= 1
        assert not violations[0].passed

    def test_vprm_no_false_positive_on_correct_prose(self) -> None:
        """SCENARIO-EXTRACT-053: VPRMArithmeticVerifier does not flag correct IT prose."""
        verifier = VPRMArithmeticVerifier()
        correct = "47 plus 28 equals 75"
        violations = verifier.detect_violations(correct)
        assert violations == []

    def test_vprm_f1_improvement_over_baseline(self) -> None:
        """SCENARIO-EXTRACT-054: VPRM F1 > ArithmeticExtractor F1 on IT-prose samples."""
        # 10 wrong samples (ground truth = True), 10 correct (ground truth = False)
        wrong_samples = [
            "47 plus 28 equals 76",
            "5 times 6 gives us 31",
            "20% of 50 is 11",
            "100 minus 15 gives 90",
            "100 divided by 4 gives 26",
            "3 times 9 equals 28",
            "15% of 200 equals 31",
            "50 plus 25 gives 74",
            "subtracting 10 from 100 gives 89",
            "7 multiplied by 8 equals 57",
        ]
        correct_samples = [
            "47 plus 28 equals 75",
            "5 times 6 gives us 30",
            "20% of 50 is 10",
            "100 minus 15 gives 85",
            "100 divided by 4 gives 25",
            "3 times 9 equals 27",
            "15% of 200 equals 30",
            "50 plus 25 gives 75",
            "subtracting 10 from 100 gives 90",
            "7 multiplied by 8 equals 56",
        ]
        all_samples = wrong_samples + correct_samples
        ground_truth = [True] * 10 + [False] * 10

        extractor = ArithmeticExtractor()
        baseline_pred = [len(extractor.extract(s)) > 0 for s in all_samples]
        baseline_f1 = VPRMArithmeticVerifier.f1_score(ground_truth, baseline_pred)

        verifier = VPRMArithmeticVerifier()
        vprm_pred = [len(verifier.detect_violations(s)) > 0 for s in all_samples]
        vprm_f1 = VPRMArithmeticVerifier.f1_score(ground_truth, vprm_pred)

        assert vprm_f1 > baseline_f1, (
            f"VPRM F1 ({vprm_f1:.3f}) should exceed baseline ({baseline_f1:.3f})"
        )
