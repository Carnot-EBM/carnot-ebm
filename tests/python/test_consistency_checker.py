"""Tests for GlobalConsistencyChecker and GlobalConsistencyReport (Exp 172).

**Detailed explanation for engineers:**
    Tests the GlobalConsistencyChecker, GlobalConsistencyReport, and the
    ConstraintStateMachine.check_global_consistency() integration method.

    The core insight under test: ConstraintStateMachine's per-step verification
    can pass (verified=True) for every individual step, yet the chain can be
    globally inconsistent — step 3 says something that contradicts step 1's
    verified output. GlobalConsistencyChecker catches these cross-step
    contradictions by comparing all (i, j) step pairs' output texts.

    All pipeline calls are mocked so tests run without JAX/model dependencies.
    The pipeline always returns verified=True (local pass) for all steps,
    isolating the test of the text-level global consistency logic.

    Test coverage:
    - Empty chain (0 steps) → consistent by default              (REQ-VERIFY-001)
    - Single step (1 step) → consistent by default               (REQ-VERIFY-001)
    - 3-step locally-consistent but globally-inconsistent chain  (SCENARIO-VERIFY-005)
    - Globally consistent chain (all steps agree)                (REQ-VERIFY-001)
    - Numeric contradiction: same entity, different values       (REQ-VERIFY-001)
    - Arithmetic contradiction: same equation, different result  (REQ-VERIFY-001)
    - Factual contradiction: same subject+predicate, diff object (REQ-VERIFY-001)
    - Severity "warning" for 1 non-factual pair                  (SCENARIO-VERIFY-005)
    - Severity "critical" for 2+ pairs                           (SCENARIO-VERIFY-005)
    - Severity "critical" for any factual contradiction          (SCENARIO-VERIFY-005)
    - recommended_rollback_step is earliest step i               (SCENARIO-VERIFY-005)
    - Same-value entries (entity, arithmetic, factual) NOT flagged (100% branch cov)
    - check_global_consistency() integration on ConstraintStateMachine

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.consistency_checker import (
    GlobalConsistencyChecker,
    GlobalConsistencyReport,
    _extract_arithmetic_claims,
    _extract_factual_triples,
    _extract_numeric_claims,
    _normalize_entity,
)
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.state_machine import ConstraintStateMachine, StepResult
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_always_passes() -> VerifyRepairPipeline:
    """Return a mocked pipeline that always returns verified=True.

    **Detailed explanation for engineers:**
        All steps in these tests are intended to pass locally so we can
        isolate and test only the global consistency logic. The pipeline
        mock always returns an empty VerificationResult with verified=True.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    pipeline = MagicMock(spec=VerifyRepairPipeline)
    pipeline.extract_constraints.return_value = []
    pipeline.verify.return_value = VerificationResult(
        verified=True,
        constraints=[],
        energy=0.0,
        violations=[],
    )
    return pipeline


def _run_steps(
    machine: ConstraintStateMachine,
    outputs: list[str],
) -> list[StepResult]:
    """Run a sequence of steps through the machine, one per output_text.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    results = []
    for idx, output in enumerate(outputs):
        results.append(machine.step(f"Question {idx}", output))
    return results


# ---------------------------------------------------------------------------
# Unit tests: extraction helpers
# ---------------------------------------------------------------------------


class TestNormalizeEntity:
    """Verify entity normalisation strips articles and stop words.

    Spec: REQ-VERIFY-001
    """

    def test_strips_leading_the(self) -> None:
        """'The Widget' → 'widget'.

        Spec: REQ-VERIFY-001
        """
        assert _normalize_entity("The Widget") == "widget"

    def test_strips_leading_a(self) -> None:
        """'A price' → 'price'.

        Spec: REQ-VERIFY-001
        """
        assert _normalize_entity("A price") == "price"

    def test_plain_entity_unchanged(self) -> None:
        """'widget' (lowercase, no article) stays 'widget'.

        Spec: REQ-VERIFY-001
        """
        assert _normalize_entity("widget") == "widget"

    def test_empty_after_strip(self) -> None:
        """All-stop-word entity normalises to empty string.

        Spec: REQ-VERIFY-001
        """
        assert _normalize_entity("the") == ""

    def test_multi_word_entity(self) -> None:
        """'The item price' → 'item price'.

        Spec: REQ-VERIFY-001
        """
        assert _normalize_entity("The item price") == "item price"


class TestExtractNumericClaims:
    """Verify numeric claim extraction from text.

    Spec: REQ-VERIFY-001
    """

    def test_costs_dollar(self) -> None:
        """Extracts entity and value from 'X costs $N' pattern.

        Spec: REQ-VERIFY-001
        """
        claims = _extract_numeric_claims("The widget costs $50.")
        assert "widget" in claims
        assert claims["widget"] == 50.0

    def test_is_value(self) -> None:
        """Extracts entity and value from 'X is N' pattern.

        Spec: REQ-VERIFY-001
        """
        claims = _extract_numeric_claims("The score is 100.")
        assert "score" in claims
        assert claims["score"] == 100.0

    def test_no_match(self) -> None:
        """Returns empty dict when no numeric patterns found.

        Spec: REQ-VERIFY-001
        """
        claims = _extract_numeric_claims("Hello world, no numbers here.")
        assert claims == {}

    def test_decimal_value(self) -> None:
        """Extracts decimal numeric values correctly.

        Spec: REQ-VERIFY-001
        """
        claims = _extract_numeric_claims("The rate was 3.14.")
        assert "rate" in claims
        assert abs(claims["rate"] - 3.14) < 1e-9


class TestExtractArithmeticClaims:
    """Verify arithmetic equation extraction from text.

    Spec: REQ-VERIFY-001
    """

    def test_addition(self) -> None:
        """Extracts addition equation with correct key normalisation.

        Spec: REQ-VERIFY-001
        """
        eqs = _extract_arithmetic_claims("We computed 3 + 5 = 8 earlier.")
        # Addition: key is (min, "+", max)
        assert (3, "+", 5) in eqs
        assert eqs[(3, "+", 5)] == 8

    def test_commutative_normalisation(self) -> None:
        """5 + 3 and 3 + 5 map to the same key (3, "+", 5).

        Spec: REQ-VERIFY-001
        """
        eqs1 = _extract_arithmetic_claims("3 + 5 = 8")
        eqs2 = _extract_arithmetic_claims("5 + 3 = 8")
        assert (3, "+", 5) in eqs1
        assert (3, "+", 5) in eqs2

    def test_subtraction_not_normalised(self) -> None:
        """Subtraction is not commutative: (3, '-', 5) ≠ (5, '-', 3).

        Spec: REQ-VERIFY-001
        """
        eqs = _extract_arithmetic_claims("10 - 3 = 7")
        assert (10, "-", 3) in eqs
        assert (3, "-", 10) not in eqs

    def test_no_equations(self) -> None:
        """Returns empty dict when no arithmetic equations found.

        Spec: REQ-VERIFY-001
        """
        eqs = _extract_arithmetic_claims("No equations in this text.")
        assert eqs == {}


class TestExtractFactualTriples:
    """Verify factual triple extraction via factual_extractor patterns.

    Spec: REQ-VERIFY-001
    """

    def test_capital_of_france(self) -> None:
        """'Paris is the capital of France' extracts (france, capital) → paris.

        Spec: REQ-VERIFY-001
        """
        triples = _extract_factual_triples("Paris is the capital of France.")
        assert ("france", "capital") in triples
        assert triples[("france", "capital")] == "paris"

    def test_no_claims(self) -> None:
        """Returns empty dict when no factual patterns match.

        Spec: REQ-VERIFY-001
        """
        triples = _extract_factual_triples("The sky is blue today.")
        assert triples == {}


# ---------------------------------------------------------------------------
# GlobalConsistencyChecker — core check() tests
# ---------------------------------------------------------------------------


class TestEmptyAndSingleStep:
    """GlobalConsistencyChecker returns consistent for 0 or 1 steps.

    Spec: REQ-VERIFY-001
    """

    def test_empty_chain_is_consistent(self) -> None:
        """An empty ConstraintStateMachine reports consistent=True.

        No steps have been run, so there are no pairs to compare.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is True
        assert report.inconsistent_pairs == []
        assert report.severity == "none"
        assert report.recommended_rollback_step is None

    def test_single_step_is_consistent(self) -> None:
        """A machine with 1 step reports consistent=True (no pairs to compare).

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        machine.step("What is the price?", "The widget costs $50.")
        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is True
        assert report.severity == "none"
        assert report.recommended_rollback_step is None


class TestGloballyConsistentChain:
    """GlobalConsistencyChecker finds no issues in a fully consistent chain.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_three_steps_all_consistent(self) -> None:
        """3 steps that all agree produce consistent=True.

        All steps mention the same entity with the same numeric value and
        the same arithmetic claim — no contradictions.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The widget costs $50. We know 3 + 5 = 8.",
            "Confirming: widget costs $50.",
            "Final answer: widget costs $50, so 3 + 5 = 8 holds.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is True
        assert report.inconsistent_pairs == []
        assert report.severity == "none"
        assert report.recommended_rollback_step is None


class TestNumericContradiction:
    """GlobalConsistencyChecker detects numeric contradictions.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_same_entity_different_values(self) -> None:
        """Step 1 says 'widget costs $50'; step 3 says 'widget costs $75'.

        Each step passes locally (pipeline always returns verified=True).
        The global checker detects the contradiction in the text.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The widget costs $50.",
            "Moving on to the next topic.",
            "The widget costs $75 based on our calculation.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is False
        # Should have a pair involving steps 0 and 2
        pair_types = {ctype for _, _, ctype, _ in report.inconsistent_pairs}
        assert "numeric" in pair_types
        # Step indices: 0 (first mention) and 2 (contradicting mention)
        pair_indices = [(i, j) for i, j, _, _ in report.inconsistent_pairs]
        assert (0, 2) in pair_indices

    def test_same_entity_same_value_not_flagged(self) -> None:
        """Two steps with the same entity and same value are NOT a contradiction.

        Tests the branch where entity is in claims_j but values match.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The score is 100.",
            "The score is 100 points.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is True

    def test_numeric_recommended_rollback_step(self) -> None:
        """recommended_rollback_step is the earliest step i in any contradiction.

        Spec: SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The item costs $50.",
            "No numeric claims here.",
            "The item costs $75.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        # Earliest contradicting step i = 0
        assert report.recommended_rollback_step == 0


class TestArithmeticContradiction:
    """GlobalConsistencyChecker detects arithmetic contradictions.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_same_equation_different_claimed_result(self) -> None:
        """Step 1 says '3 + 5 = 8'; step 3 says '3 + 5 = 10'.

        Each step passes locally. The global checker finds the contradiction.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "We calculated 3 + 5 = 8 previously.",
            "Intermediate reasoning step.",
            "We know that 3 + 5 = 10 from earlier work.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is False
        pair_types = {ctype for _, _, ctype, _ in report.inconsistent_pairs}
        assert "arithmetic" in pair_types

    def test_same_equation_same_result_not_flagged(self) -> None:
        """Two steps with the same equation and same result are NOT a contradiction.

        Tests the branch where arithmetic key matches but claimed values agree.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "As noted: 3 + 5 = 8.",
            "Confirmed: 3 + 5 = 8.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is True


class TestFactualContradiction:
    """GlobalConsistencyChecker detects factual contradictions.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_same_subject_predicate_different_object(self) -> None:
        """Step 1 says 'Paris is the capital of France'; step 3 says 'Berlin is'.

        Each step passes locally. The global checker finds the factual contradiction.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "Paris is the capital of France.",
            "General discussion about Europe.",
            "Berlin is the capital of France, as we noted.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is False
        pair_types = {ctype for _, _, ctype, _ in report.inconsistent_pairs}
        assert "factual" in pair_types

    def test_same_subject_predicate_same_object_not_flagged(self) -> None:
        """Two steps with same (subject, predicate) and same object: not a contradiction.

        Tests the branch where factual key matches but objects agree.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "Paris is the capital of France.",
            "Yes, Paris is the capital of France.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is True

    def test_factual_contradiction_severity_critical(self) -> None:
        """A single factual contradiction raises severity to 'critical'.

        Per the spec: factual contradictions are always severity='critical'
        regardless of how many pairs are found.

        Spec: SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "Paris is the capital of France.",
            "Berlin is the capital of France.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.severity == "critical"


class TestLocallyConsistentGloballyInconsistent:
    """Key regression: local checks pass but global check finds contradiction.

    This is the core value-add of GlobalConsistencyChecker: ConstraintStateMachine
    passes all steps (verified=True for each) while GlobalConsistencyChecker
    detects a cross-step inconsistency in the text.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_three_step_chain_passes_locally_fails_globally(self) -> None:
        """All 3 steps are locally verified; steps 1 and 3 numerically contradict.

        Step 1: 'widget costs $50' — passes locally.
        Step 2: 'intermediate step' — passes locally.
        Step 3: 'widget costs $75' — passes locally.

        Local-only checker: sees all verified=True → no issue detected.
        Global checker: finds numeric contradiction between steps 0 and 2.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        results = _run_steps(machine, [
            "The widget costs $50 in our catalogue.",
            "Next we verify the shipping address is correct.",
            "The widget costs $75 based on the updated price list.",
        ])

        # Verify all steps pass locally (the pipeline always says verified=True)
        for result in results:
            assert result.verification.verified is True, (
                f"Step {result.step_index} should pass locally"
            )

        # Global check detects the contradiction
        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is False
        assert len(report.inconsistent_pairs) >= 1
        # The contradiction is between step 0 and step 2
        assert any(
            i == 0 and j == 2
            for i, j, _, _ in report.inconsistent_pairs
        )


class TestSeverityLevels:
    """Verify severity classification rules.

    Spec: SCENARIO-VERIFY-005
    """

    def test_severity_none_for_consistent_chain(self) -> None:
        """Consistent chain → severity='none'.

        Spec: SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The value is 42.",
            "The value is 42.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.severity == "none"

    def test_severity_warning_for_single_non_factual_pair(self) -> None:
        """One numeric contradiction pair → severity='warning'.

        Spec: SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The item costs $10.",
            "The item costs $20.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.consistent is False
        assert report.severity == "warning"

    def test_severity_critical_for_two_non_factual_pairs(self) -> None:
        """Two contradiction pairs (both numeric) → severity='critical'.

        Spec: SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        _run_steps(machine, [
            "The item costs $10. Also 3 + 5 = 8.",
            "The item costs $20. Also 3 + 5 = 10.",
        ])

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)

        assert report.severity == "critical"
        assert len(report.inconsistent_pairs) >= 2


class TestCheckGlobalConsistencyIntegration:
    """ConstraintStateMachine.check_global_consistency() integration.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_method_available_on_machine(self) -> None:
        """check_global_consistency() is callable on ConstraintStateMachine.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        report = machine.check_global_consistency()
        assert isinstance(report, GlobalConsistencyReport)

    def test_method_returns_consistent_for_empty_machine(self) -> None:
        """check_global_consistency() on fresh machine → consistent=True.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        report = machine.check_global_consistency()
        assert report.consistent is True
        assert report.severity == "none"

    def test_method_detects_contradiction(self) -> None:
        """check_global_consistency() detects cross-step numeric contradiction.

        Spec: SCENARIO-VERIFY-005
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        machine.step("Q1", "The price is $50.")
        machine.step("Q2", "The price is $80.")

        report = machine.check_global_consistency()

        assert report.consistent is False
        assert report.severity in {"warning", "critical"}
        assert report.recommended_rollback_step == 0

    def test_output_text_stored_in_step_result(self) -> None:
        """StepResult.output_text stores the step's output for consistency checking.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine(pipeline=_make_pipeline_always_passes())
        result = machine.step("What is the price?", "The widget costs $50.")

        assert result.output_text == "The widget costs $50."
