"""Tests for ConstraintStateMachine.

**Detailed explanation for engineers:**
    Tests the ConstraintStateMachine, StepResult, and all public API methods:
    step(), rollback(), history(), verified_facts(), pending_facts().

    The pipeline is mocked so tests run without JAX/model dependencies and
    control exactly which facts are extracted and which violations are raised.

    Test coverage:
    - StepResult fields are populated correctly (REQ-VERIFY-001)
    - New facts detected from step output (SCENARIO-VERIFY-005)
    - Contradictions flagged when a violation targets a previously VERIFIED fact
    - Rollback restores state and trims history
    - Rollback out of range raises IndexError
    - history() returns results in step order
    - verified_facts() and pending_facts() reflect current state
    - Multiple sequential steps accumulate facts across steps
    - step_index after rollback continues from the restored point

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.agentic import ConstraintState, FactStatus, _normalize_fact_key
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.state_machine import ConstraintStateMachine, StepResult
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cr(desc: str, satisfied: bool = True) -> ConstraintResult:
    """Build a minimal ConstraintResult for testing.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    return ConstraintResult(
        constraint_type="factual",
        description=desc,
        metadata={"satisfied": satisfied},
    )


def _make_vr(
    constraints: list[ConstraintResult],
    violations: list[ConstraintResult] | None = None,
) -> VerificationResult:
    """Build a VerificationResult matching the given constraints.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    if violations is None:
        violations = [c for c in constraints if not c.metadata.get("satisfied", True)]
    return VerificationResult(
        verified=len(violations) == 0,
        constraints=constraints,
        energy=0.0 if len(violations) == 0 else 1.0,
        violations=violations,
    )


def _mock_pipeline(
    extract_returns: list[ConstraintResult],
    verify_returns: VerificationResult,
) -> VerifyRepairPipeline:
    """Create a mocked VerifyRepairPipeline for controlled testing.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    pipeline = MagicMock(spec=VerifyRepairPipeline)
    pipeline.extract_constraints.return_value = extract_returns
    pipeline.verify.return_value = verify_returns
    return pipeline


# ---------------------------------------------------------------------------
# StepResult structure
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestStepResultStructure:
    """Verify StepResult is populated with correct types and content.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_step_result_fields_exist(self) -> None:
        """StepResult has all required fields.

        Spec: REQ-VERIFY-001
        """
        cr = _make_cr("the sky is blue")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        result = machine.step("What color is the sky?", "The sky is blue.")

        assert isinstance(result, StepResult)
        assert result.step_index == 0
        assert isinstance(result.verification, VerificationResult)
        assert isinstance(result.new_facts, list)
        assert isinstance(result.contradictions, list)
        assert isinstance(result.state_snapshot, dict)

    def test_step_index_increments(self) -> None:
        """step_index increments by 1 for each call to step().

        Spec: REQ-VERIFY-001
        """
        cr = _make_cr("sky is blue")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        r0 = machine.step("Q1", "sky is blue")
        r1 = machine.step("Q2", "sky is blue")
        r2 = machine.step("Q3", "sky is blue")

        assert r0.step_index == 0
        assert r1.step_index == 1
        assert r2.step_index == 2

    def test_state_snapshot_is_serializable(self) -> None:
        """state_snapshot is a plain dict suitable for JSON serialization.

        Spec: SCENARIO-VERIFY-005
        """
        import json

        cr = _make_cr("water is wet")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        result = machine.step("Is water wet?", "Water is wet.")

        # Should not raise
        serialized = json.dumps(result.state_snapshot)
        assert serialized  # non-empty


# ---------------------------------------------------------------------------
# New facts detection
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestNewFactsDetection:
    """Verify new_facts only contains facts first introduced in this step.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_first_step_all_facts_are_new(self) -> None:
        """All facts extracted in the first step are new facts.

        Spec: SCENARIO-VERIFY-005
        """
        cr1 = _make_cr("dogs are mammals")
        cr2 = _make_cr("cats are mammals")
        vr = _make_vr([cr1, cr2])
        pipeline = _mock_pipeline([cr1, cr2], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        result = machine.step("Tell me about mammals.", "Dogs are mammals. Cats are mammals.")

        assert len(result.new_facts) == 2
        new_descs = {f.fact for f in result.new_facts}
        assert "dogs are mammals" in new_descs
        assert "cats are mammals" in new_descs

    def test_second_step_repeated_fact_not_in_new_facts(self) -> None:
        """A fact already in state from step 0 is NOT reported as new in step 1.

        Spec: SCENARIO-VERIFY-005
        """
        cr_old = _make_cr("dogs are mammals")
        cr_new = _make_cr("fish breathe water")

        vr_step0 = _make_vr([cr_old])
        vr_step1 = _make_vr([cr_old, cr_new])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.side_effect = [[cr_old], [cr_old, cr_new]]
        pipeline.verify.side_effect = [vr_step0, vr_step1]

        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Step 0", "Dogs are mammals.")
        result = machine.step("Step 1", "Dogs are mammals. Fish breathe water.")

        new_descs = {f.fact for f in result.new_facts}
        assert "fish breathe water" in new_descs
        assert "dogs are mammals" not in new_descs

    def test_no_constraints_extracted_yields_empty_new_facts(self) -> None:
        """If no constraints are extracted, new_facts is empty.

        Spec: REQ-VERIFY-001
        """
        vr = _make_vr([])
        pipeline = _mock_pipeline([], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        result = machine.step("Hello", "Hi there.")

        assert result.new_facts == []


# ---------------------------------------------------------------------------
# Contradiction detection
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    """Verify contradictions are flagged when a step violates a previously verified fact.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_no_contradiction_when_no_violations(self) -> None:
        """No contradictions when verification passes.

        Spec: SCENARIO-VERIFY-005
        """
        cr = _make_cr("sky is blue")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        r0 = machine.step("Q", "The sky is blue.")
        r1 = machine.step("Q", "The sky is blue.")

        assert r0.contradictions == []
        assert r1.contradictions == []

    def test_contradiction_when_new_step_violates_verified_fact(self) -> None:
        """A violation of a previously VERIFIED fact is detected as a contradiction.

        Scenario: Step 0 verifies 'sky is blue'. Step 1 violates 'sky is blue'.
        Step 1 should report 'sky is blue' in contradictions.

        Spec: SCENARIO-VERIFY-005
        """
        cr_blue = _make_cr("sky is blue", satisfied=True)
        cr_not_blue = _make_cr("sky is blue", satisfied=False)  # same key, now violated

        # Step 0: sky is blue passes -> VERIFIED
        vr_step0 = _make_vr([cr_blue], violations=[])
        # Step 1: sky is blue fails -> VIOLATED (contradiction of step 0)
        vr_step1 = _make_vr([cr_not_blue], violations=[cr_not_blue])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.side_effect = [[cr_blue], [cr_not_blue]]
        pipeline.verify.side_effect = [vr_step0, vr_step1]

        machine = ConstraintStateMachine(pipeline=pipeline)
        r0 = machine.step("Q0", "The sky is blue.")
        r1 = machine.step("Q1", "The sky is not blue.")

        assert r0.contradictions == []
        assert "sky is blue" in r1.contradictions

    def test_violation_of_unverified_fact_is_not_contradiction(self) -> None:
        """Violations of ASSUMED (unverified) facts are NOT contradictions.

        Scenario: A fact is introduced and immediately violated in the same
        step. Since it was never VERIFIED, it cannot be a contradiction.

        Spec: SCENARIO-VERIFY-005
        """
        cr = _make_cr("moon is cheese", satisfied=False)
        vr = _make_vr([cr], violations=[cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        result = machine.step("Q", "The moon is cheese.")

        # cr was never VERIFIED before this step, so no contradiction
        assert result.contradictions == []

    def test_multiple_contradictions_reported(self) -> None:
        """Multiple previously verified facts violated at once are all listed.

        Spec: SCENARIO-VERIFY-005
        """
        cr_a = _make_cr("A is true", satisfied=True)
        cr_b = _make_cr("B is true", satisfied=True)
        vr0 = _make_vr([cr_a, cr_b], violations=[])

        cr_a_bad = _make_cr("A is true", satisfied=False)
        cr_b_bad = _make_cr("B is true", satisfied=False)
        vr1 = _make_vr([cr_a_bad, cr_b_bad], violations=[cr_a_bad, cr_b_bad])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.side_effect = [[cr_a, cr_b], [cr_a_bad, cr_b_bad]]
        pipeline.verify.side_effect = [vr0, vr1]

        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "A is true. B is true.")
        r1 = machine.step("Q1", "A is false. B is false.")

        assert len(r1.contradictions) == 2
        assert "A is true" in r1.contradictions
        assert "B is true" in r1.contradictions


# ---------------------------------------------------------------------------
# Rollback
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestRollback:
    """Verify rollback() restores state correctly.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_rollback_trims_history(self) -> None:
        """After rollback(0), history contains only step 0.

        Spec: REQ-VERIFY-001
        """
        cr = _make_cr("fact A")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        machine.step("Q0", "Fact A.")
        machine.step("Q1", "Fact A.")
        machine.step("Q2", "Fact A.")

        assert len(machine.history()) == 3
        machine.rollback(0)
        assert len(machine.history()) == 1
        assert machine.history()[0].step_index == 0

    def test_rollback_restores_facts(self) -> None:
        """After rollback, facts added in rolled-back steps are gone.

        Spec: SCENARIO-VERIFY-005
        """
        cr_a = _make_cr("fact a")
        cr_b = _make_cr("fact b")
        vr_a = _make_vr([cr_a])
        vr_b = _make_vr([cr_b])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.side_effect = [[cr_a], [cr_b]]
        pipeline.verify.side_effect = [vr_a, vr_b]

        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "Fact A.")
        machine.step("Q1", "Fact B.")

        # Both facts are in state
        all_facts = {f.fact for f in machine._state.facts.values()}
        assert "fact a" in all_facts
        assert "fact b" in all_facts

        # Rollback to step 0: fact b should disappear
        machine.rollback(0)
        all_facts_after = {f.fact for f in machine._state.facts.values()}
        assert "fact a" in all_facts_after
        assert "fact b" not in all_facts_after

    def test_rollback_step_index_continues_from_restored_point(self) -> None:
        """After rollback(0), the next step gets index 1 (not 3).

        Spec: REQ-VERIFY-001
        """
        cr = _make_cr("fact")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        machine.step("Q0", "Fact.")
        machine.step("Q1", "Fact.")
        machine.step("Q2", "Fact.")
        machine.rollback(0)

        result = machine.step("Q_new", "Fact.")
        assert result.step_index == 1

    def test_rollback_out_of_range_raises_index_error(self) -> None:
        """rollback() raises IndexError when to_step is out of range.

        Spec: REQ-VERIFY-001
        """
        vr = _make_vr([])
        pipeline = _mock_pipeline([], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        machine.step("Q0", "Out.")

        with pytest.raises(IndexError):
            machine.rollback(5)

        with pytest.raises(IndexError):
            machine.rollback(-1)

    def test_rollback_to_last_step_is_noop(self) -> None:
        """rollback(n-1) when n steps done is a no-op on history.

        Spec: REQ-VERIFY-001
        """
        cr = _make_cr("fact")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        machine.step("Q0", "Fact.")
        machine.step("Q1", "Fact.")
        machine.rollback(1)  # Roll back to the LAST step (no-op on content)

        assert len(machine.history()) == 2

    def test_rollback_on_empty_history_raises_index_error(self) -> None:
        """rollback() on an empty machine raises IndexError.

        Spec: REQ-VERIFY-001
        """
        pipeline = _mock_pipeline([], _make_vr([]))
        machine = ConstraintStateMachine(pipeline=pipeline)

        with pytest.raises(IndexError):
            machine.rollback(0)


# ---------------------------------------------------------------------------
# history(), verified_facts(), pending_facts()
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestAccessors:
    """Verify history(), verified_facts(), and pending_facts() are correct.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_history_returns_all_step_results_in_order(self) -> None:
        """history() returns StepResult list in step order.

        Spec: REQ-VERIFY-001
        """
        cr = _make_cr("fact")
        vr = _make_vr([cr])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        machine.step("Q0", "Fact.")
        machine.step("Q1", "Fact.")
        machine.step("Q2", "Fact.")

        h = machine.history()
        assert len(h) == 3
        assert h[0].step_index == 0
        assert h[1].step_index == 1
        assert h[2].step_index == 2

    def test_history_returns_copy_not_internal_reference(self) -> None:
        """Mutating the returned history list does not affect the machine.

        Spec: REQ-VERIFY-001
        """
        pipeline = _mock_pipeline([], _make_vr([]))
        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q", "Empty.")

        h = machine.history()
        h.clear()

        # Machine's own history is unaffected
        assert len(machine.history()) == 1

    def test_verified_facts_empty_before_any_steps(self) -> None:
        """verified_facts() is empty on a fresh machine.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine()
        assert machine.verified_facts() == []

    def test_pending_facts_empty_before_any_steps(self) -> None:
        """pending_facts() is empty on a fresh machine.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine()
        assert machine.pending_facts() == []

    def test_verified_facts_populated_after_successful_step(self) -> None:
        """verified_facts() returns VERIFIED facts after a passing step.

        Spec: SCENARIO-VERIFY-005
        """
        cr = _make_cr("paris is in france", satisfied=True)
        vr = _make_vr([cr], violations=[])
        pipeline = _mock_pipeline([cr], vr)
        machine = ConstraintStateMachine(pipeline=pipeline)

        machine.step("Where is Paris?", "Paris is in France.")

        verified = machine.verified_facts()
        assert len(verified) == 1
        assert verified[0].fact == "paris is in france"
        assert verified[0].status == FactStatus.VERIFIED

    def test_pending_facts_from_step_with_no_verification_result(self) -> None:
        """Facts not yet verified/violated remain ASSUMED (pending).

        Spec: SCENARIO-VERIFY-005
        """
        # Fact is extracted but NOT in the violations list -> stays ASSUMED
        # Note: propagate() marks facts as VERIFIED when they are extracted
        # and NOT in violations. To get an ASSUMED fact, we'd need a fact that
        # is in state but wasn't re-verified in the current step.
        # Simplest: extract a fact in step 0, then in step 1 extract a
        # different fact (the step-0 fact stays ASSUMED if it was never verified).

        # Step 0: extract fact_x, no violations -> fact_x gets VERIFIED
        cr_x = _make_cr("x exists", satisfied=True)
        vr0 = _make_vr([cr_x], violations=[])

        # Step 1: extract fact_y (new), violate nothing
        # fact_y is added as ASSUMED and immediately VERIFIED since no violations
        cr_y = _make_cr("y exists", satisfied=True)
        vr1 = _make_vr([cr_y], violations=[])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.side_effect = [[cr_x], [cr_y]]
        pipeline.verify.side_effect = [vr0, vr1]

        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "X exists.")
        machine.step("Q1", "Y exists.")

        # Both facts should be VERIFIED (not pending) since no violations
        verified = {f.fact for f in machine.verified_facts()}
        assert "x exists" in verified
        assert "y exists" in verified
        # And nothing pending
        assert machine.pending_facts() == []

    def test_pending_facts_after_step_with_unresolved_assumptions(self) -> None:
        """Facts added with ASSUMED status remain in pending_facts() until verified.

        Spec: SCENARIO-VERIFY-005
        """
        # Manually add an ASSUMED fact to state to test pending_facts().
        pipeline = _mock_pipeline([], _make_vr([]))
        machine = ConstraintStateMachine(pipeline=pipeline)

        # Directly insert an ASSUMED fact into internal state
        machine._state.add_fact("unresolved assumption", step_index=0)

        pending = machine.pending_facts()
        assert len(pending) == 1
        assert pending[0].fact == "unresolved assumption"
        assert pending[0].status == FactStatus.ASSUMED


# ---------------------------------------------------------------------------
# Multi-step integration
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestMultiStepIntegration:
    """Integration tests spanning multiple steps.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_three_step_accumulation_and_contradiction(self) -> None:
        """Three steps: two build facts, third contradicts one.

        Spec: SCENARIO-VERIFY-005
        """
        cr_a = _make_cr("alice is tall", satisfied=True)
        cr_b = _make_cr("bob is short", satisfied=True)
        cr_a_bad = _make_cr("alice is tall", satisfied=False)

        vr0 = _make_vr([cr_a], violations=[])
        vr1 = _make_vr([cr_b], violations=[])
        vr2 = _make_vr([cr_a_bad], violations=[cr_a_bad])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.side_effect = [[cr_a], [cr_b], [cr_a_bad]]
        pipeline.verify.side_effect = [vr0, vr1, vr2]

        machine = ConstraintStateMachine(pipeline=pipeline)
        r0 = machine.step("Step 0", "Alice is tall.")
        r1 = machine.step("Step 1", "Bob is short.")
        r2 = machine.step("Step 2", "Alice is not tall.")

        assert r0.contradictions == []
        assert r1.contradictions == []
        assert "alice is tall" in r2.contradictions

        # Verify history
        h = machine.history()
        assert len(h) == 3
        assert h[0].verification.verified is True
        assert h[1].verification.verified is True
        assert h[2].verification.verified is False

    def test_rollback_and_continue(self) -> None:
        """Rollback to step 0, then add two more steps -- history grows from there.

        Spec: SCENARIO-VERIFY-005
        """
        cr = _make_cr("fact")
        vr = _make_vr([cr])

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        # step 0, step 1, step 2 (before rollback), step A, step B (after rollback)
        pipeline.extract_constraints.side_effect = [
            [cr], [cr], [cr], [cr], [cr]
        ]
        pipeline.verify.side_effect = [vr, vr, vr, vr, vr]

        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "Fact.")
        machine.step("Q1", "Fact.")
        machine.step("Q2", "Fact.")

        assert len(machine.history()) == 3

        machine.rollback(0)
        assert len(machine.history()) == 1

        rA = machine.step("QA", "Fact.")
        rB = machine.step("QB", "Fact.")

        assert rA.step_index == 1
        assert rB.step_index == 2
        assert len(machine.history()) == 3

    def test_default_pipeline_construction(self) -> None:
        """ConstraintStateMachine can be constructed without explicit pipeline.

        Spec: REQ-VERIFY-001
        """
        machine = ConstraintStateMachine()
        assert machine is not None
        assert machine.history() == []
        assert machine.verified_facts() == []
        assert machine.pending_facts() == []
