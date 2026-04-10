"""Tests for multi-turn agentic verification with constraint propagation.

**Detailed explanation for engineers:**
    Tests the ConstraintState, TrackedFact, AgentStep, and propagate()
    function from carnot.pipeline.agentic. Verifies that:
    - Facts accumulate correctly across multiple steps
    - Fact status transitions work (ASSUMED -> VERIFIED, ASSUMED -> VIOLATED)
    - propagate() correctly extracts constraints and updates state
    - Step N+1 inherits step N's verified facts
    - snapshot() produces serializable dicts

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.agentic import (
    AgentStep,
    ConstraintState,
    FactStatus,
    TrackedFact,
    _normalize_fact_key,
    propagate,
)
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constraint(desc: str, satisfied: bool = True) -> ConstraintResult:
    """Create a ConstraintResult with given description and satisfaction.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    return ConstraintResult(
        constraint_type="arithmetic",
        description=desc,
        metadata={"satisfied": satisfied},
    )


def _make_verification(
    constraints: list[ConstraintResult],
    violations: list[ConstraintResult] | None = None,
) -> VerificationResult:
    """Create a VerificationResult from constraints.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    if violations is None:
        violations = [c for c in constraints if not c.metadata.get("satisfied", True)]
    verified = len(violations) == 0
    return VerificationResult(
        verified=verified,
        constraints=constraints,
        energy=0.0 if verified else 1.0,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Test ConstraintState accumulation across 3 steps
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestConstraintStateAccumulation:
    """Test that ConstraintState correctly accumulates facts across steps.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_add_facts_across_three_steps(self) -> None:
        """Facts from steps 0, 1, 2 all appear in the state.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()

        # Step 0: introduce fact A
        key_a = state.add_fact("3 * 4 = 12", step_index=0)
        assert len(state.facts) == 1
        assert state.facts[key_a].introduced_at_step == 0
        assert state.facts[key_a].status == FactStatus.ASSUMED

        # Step 1: introduce fact B
        key_b = state.add_fact("12 + 5 = 17", step_index=1)
        assert len(state.facts) == 2
        assert state.facts[key_b].introduced_at_step == 1

        # Step 2: introduce fact C
        key_c = state.add_fact("The answer is 17", step_index=2)
        assert len(state.facts) == 3
        assert state.facts[key_c].introduced_at_step == 2

        # All three are ASSUMED
        assert len(state.get_assumed()) == 3
        assert len(state.get_verified()) == 0
        assert len(state.get_violated()) == 0

    def test_duplicate_fact_is_noop(self) -> None:
        """Adding the same fact twice keeps the original entry.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        key1 = state.add_fact("3 * 4 = 12", step_index=0)
        key2 = state.add_fact("3 * 4 = 12", step_index=1)
        assert key1 == key2
        assert len(state.facts) == 1
        # Original provenance preserved
        assert state.facts[key1].introduced_at_step == 0

    def test_normalization_deduplicates(self) -> None:
        """Facts with different whitespace/case map to same key.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        key1 = state.add_fact("3 * 4 = 12", step_index=0)
        key2 = state.add_fact("3  *  4  =  12", step_index=1)
        key3 = state.add_fact("3 * 4 = 12", step_index=2)
        assert key1 == key2 == key3
        assert len(state.facts) == 1


# ---------------------------------------------------------------------------
# Test fact status transitions
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestFactStatusTransitions:
    """Test ASSUMED -> VERIFIED and ASSUMED -> VIOLATED transitions.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_assumed_to_verified(self) -> None:
        """A fact transitions from ASSUMED to VERIFIED when verify_fact is called.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        key = state.add_fact("3 * 4 = 12", step_index=0)
        assert state.facts[key].status == FactStatus.ASSUMED

        state.verify_fact(key, step_index=1, energy=0.0)
        assert state.facts[key].status == FactStatus.VERIFIED
        assert state.facts[key].verified_at_step == 1
        assert state.facts[key].energy == 0.0
        assert len(state.get_verified()) == 1
        assert len(state.get_assumed()) == 0

    def test_assumed_to_violated(self) -> None:
        """A fact transitions from ASSUMED to VIOLATED when violate_fact is called.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        key = state.add_fact("3 * 4 = 13", step_index=0)
        assert state.facts[key].status == FactStatus.ASSUMED

        state.violate_fact(key, step_index=1, energy=1.5)
        assert state.facts[key].status == FactStatus.VIOLATED
        assert state.facts[key].verified_at_step == 1
        assert state.facts[key].energy == 1.5
        assert len(state.get_violated()) == 1
        assert len(state.get_assumed()) == 0

    def test_verify_nonexistent_key_raises(self) -> None:
        """verify_fact raises KeyError for unknown keys.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        with pytest.raises(KeyError):
            state.verify_fact("no such key", step_index=0, energy=0.0)

    def test_violate_nonexistent_key_raises(self) -> None:
        """violate_fact raises KeyError for unknown keys.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        with pytest.raises(KeyError):
            state.violate_fact("no such key", step_index=0, energy=1.0)


# ---------------------------------------------------------------------------
# Test propagate() with mock pipeline
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestPropagate:
    """Test propagate() with mock pipeline results.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_propagate_with_satisfied_constraints(self) -> None:
        """propagate() marks facts as VERIFIED when all constraints pass.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        pipeline = MagicMock(spec=VerifyRepairPipeline)

        cr = _make_constraint("3 + 4 = 7", satisfied=True)
        pipeline.extract_constraints.return_value = [cr]
        pipeline.verify.return_value = _make_verification([cr])

        step = AgentStep(
            step_index=0,
            input_text="What is 3 + 4?",
            output_text="3 + 4 = 7",
        )

        result = propagate(state, step, pipeline)

        assert result is state  # mutates in-place
        assert len(state.get_verified()) == 1
        assert len(state.get_violated()) == 0
        assert step.verification is not None
        assert step.verification.verified is True

    def test_propagate_with_violated_constraint(self) -> None:
        """propagate() marks facts as VIOLATED when constraints fail.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        pipeline = MagicMock(spec=VerifyRepairPipeline)

        cr = _make_constraint("3 + 4 = 8", satisfied=False)
        pipeline.extract_constraints.return_value = [cr]
        pipeline.verify.return_value = _make_verification([cr], violations=[cr])

        step = AgentStep(
            step_index=0,
            input_text="What is 3 + 4?",
            output_text="3 + 4 = 8",
        )

        propagate(state, step, pipeline)

        assert len(state.get_violated()) == 1
        assert len(state.get_verified()) == 0
        assert step.verification is not None
        assert step.verification.verified is False

    def test_propagate_updates_step_snapshots(self) -> None:
        """propagate() fills in state_before and state_after on the step.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.return_value = []
        pipeline.verify.return_value = _make_verification([])

        step = AgentStep(step_index=0, input_text="Q", output_text="A")
        propagate(state, step, pipeline)

        assert "facts" in step.state_before
        assert "facts" in step.state_after


# ---------------------------------------------------------------------------
# Test step N+1 inherits step N's verified facts
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestStepInheritance:
    """Test that verified facts from step N persist into step N+1.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_step_n_plus_1_sees_step_n_verified_facts(self) -> None:
        """Facts verified at step 0 remain VERIFIED at step 1.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        pipeline = MagicMock(spec=VerifyRepairPipeline)

        # Step 0: one satisfied constraint
        cr0 = _make_constraint("3 * 4 = 12", satisfied=True)
        pipeline.extract_constraints.return_value = [cr0]
        pipeline.verify.return_value = _make_verification([cr0])

        step0 = AgentStep(
            step_index=0,
            input_text="What is 3*4?",
            output_text="3 * 4 = 12",
        )
        propagate(state, step0, pipeline)
        assert len(state.get_verified()) == 1

        # Step 1: a different constraint
        cr1 = _make_constraint("12 + 5 = 17", satisfied=True)
        pipeline.extract_constraints.return_value = [cr1]
        pipeline.verify.return_value = _make_verification([cr1])

        step1 = AgentStep(
            step_index=1,
            input_text="Now add 5",
            output_text="12 + 5 = 17",
        )
        propagate(state, step1, pipeline)

        # Both facts are verified
        assert len(state.get_verified()) == 2
        # Step 0's fact is still there
        key0 = _normalize_fact_key("3 * 4 = 12")
        assert state.facts[key0].status == FactStatus.VERIFIED
        assert state.facts[key0].introduced_at_step == 0

    def test_violation_at_step_1_does_not_affect_step_0_verified(self) -> None:
        """A violation at step 1 doesn't retroactively un-verify step 0's facts.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        pipeline = MagicMock(spec=VerifyRepairPipeline)

        # Step 0: verified
        cr0 = _make_constraint("3 * 4 = 12", satisfied=True)
        pipeline.extract_constraints.return_value = [cr0]
        pipeline.verify.return_value = _make_verification([cr0])
        step0 = AgentStep(step_index=0, input_text="Q0", output_text="3 * 4 = 12")
        propagate(state, step0, pipeline)

        # Step 1: violated
        cr1 = _make_constraint("12 + 5 = 18", satisfied=False)
        pipeline.extract_constraints.return_value = [cr1]
        pipeline.verify.return_value = _make_verification([cr1], violations=[cr1])
        step1 = AgentStep(step_index=1, input_text="Q1", output_text="12 + 5 = 18")
        propagate(state, step1, pipeline)

        # Step 0 fact still verified, step 1 fact violated
        key0 = _normalize_fact_key("3 * 4 = 12")
        key1 = _normalize_fact_key("12 + 5 = 18")
        assert state.facts[key0].status == FactStatus.VERIFIED
        assert state.facts[key1].status == FactStatus.VIOLATED

    def test_violation_propagates_to_later_assumed_facts(self) -> None:
        """A violation at step 0 marks ASSUMED facts from later steps as VIOLATED.

        **Detailed explanation for engineers:**
            If step 0 is violated, any ASSUMED facts from step 1+ that were
            pre-added to the state should be invalidated, since they were
            built on top of the now-violated step.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        pipeline = MagicMock(spec=VerifyRepairPipeline)

        # Pre-add an ASSUMED fact from a future step (step 2).
        state.add_fact(
            fact="result is 42",
            step_index=2,
            status=FactStatus.ASSUMED,
        )

        # Step 0: violated constraint
        cr0 = _make_constraint("1 + 1 = 3", satisfied=False)
        pipeline.extract_constraints.return_value = [cr0]
        pipeline.verify.return_value = _make_verification([cr0], violations=[cr0])
        step0 = AgentStep(step_index=0, input_text="Q0", output_text="1 + 1 = 3")
        propagate(state, step0, pipeline)

        # The step 0 fact is VIOLATED
        key0 = _normalize_fact_key("1 + 1 = 3")
        assert state.facts[key0].status == FactStatus.VIOLATED

        # The pre-added ASSUMED fact from step 2 should also be VIOLATED
        key_future = _normalize_fact_key("result is 42")
        assert state.facts[key_future].status == FactStatus.VIOLATED
        assert state.facts[key_future].verified_at_step == 0
        assert state.facts[key_future].energy == 1.0


# ---------------------------------------------------------------------------
# Test snapshot serialization
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestSnapshot:
    """Test that snapshot() produces JSON-serializable dicts.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_empty_state_snapshot(self) -> None:
        """Empty state produces a valid snapshot.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        snap = state.snapshot()
        assert snap == {
            "facts": {},
            "n_verified": 0,
            "n_assumed": 0,
            "n_violated": 0,
        }

    def test_populated_state_snapshot(self) -> None:
        """State with facts produces correct snapshot structure.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        import json

        state = ConstraintState()
        state.add_fact("A = 1", step_index=0)
        state.add_fact("B = 2", step_index=1)
        key_a = _normalize_fact_key("A = 1")
        state.verify_fact(key_a, step_index=1, energy=0.0)

        snap = state.snapshot()

        # Must be JSON-serializable
        json_str = json.dumps(snap)
        assert isinstance(json_str, str)

        assert snap["n_verified"] == 1
        assert snap["n_assumed"] == 1
        assert snap["n_violated"] == 0
        assert key_a in snap["facts"]
        assert snap["facts"][key_a]["status"] == "verified"

    def test_snapshot_with_constraint(self) -> None:
        """Snapshot correctly indicates has_constraint flag.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        cr = _make_constraint("X = 1", satisfied=True)
        state.add_fact("X = 1", step_index=0, constraint=cr)

        snap = state.snapshot()
        key = _normalize_fact_key("X = 1")
        assert snap["facts"][key]["has_constraint"] is True


# ---------------------------------------------------------------------------
# Test get_all_constraints
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestGetAllConstraints:
    """Test get_all_constraints returns ConstraintResult objects.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_returns_constraints_from_facts(self) -> None:
        """get_all_constraints returns constraints from facts that have them.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        state = ConstraintState()
        cr1 = _make_constraint("A = 1")
        cr2 = _make_constraint("B = 2")
        state.add_fact("A = 1", step_index=0, constraint=cr1)
        state.add_fact("B = 2", step_index=1, constraint=cr2)
        state.add_fact("C = 3", step_index=2)  # no constraint

        constraints = state.get_all_constraints()
        assert len(constraints) == 2
        assert cr1 in constraints
        assert cr2 in constraints


# ---------------------------------------------------------------------------
# Test normalize_fact_key
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestNormalizeFact:
    """Test the fact key normalization helper.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_lowercases(self) -> None:
        """Normalization lowercases text.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        assert _normalize_fact_key("Hello World") == "hello world"

    def test_collapses_whitespace(self) -> None:
        """Normalization collapses multiple spaces.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        assert _normalize_fact_key("a  +  b  =  c") == "a + b = c"

    def test_strips(self) -> None:
        """Normalization strips leading/trailing whitespace.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        assert _normalize_fact_key("  hello  ") == "hello"


class TestRetroactiveViolation:
    """Cover the retroactive violation branch (lines 469-471) in propagate().

    When a new fact is VIOLATED and an older ASSUMED fact was introduced at a
    future step index, that assumed fact should also be marked VIOLATED.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_assumed_fact_from_future_step_gets_violated(self) -> None:
        """REQ-VERIFY-001: Future assumed facts are retroactively violated."""
        state = ConstraintState()
        # Pre-populate a fact as if it came from a future step (step_index=5).
        state.facts["future_claim"] = TrackedFact(
            fact="some claim",
            status=FactStatus.ASSUMED,
            introduced_at_step=5,
            verified_at_step=None,
            energy=0.0,
        )

        # Create a step at index 2 (before the future fact).
        step = AgentStep(
            step_index=2,
            input_text="What is 2 + 2?",
            output_text="2 + 2 = 5",
        )

        # Mock pipeline to return a violation.
        mock_pipeline = MagicMock(spec=VerifyRepairPipeline)
        violation = ConstraintResult(
            constraint_type="arithmetic",
            description="2 + 2 = 5 (correct: 4)",
            metadata={"satisfied": False, "correct_result": 4},
        )
        mock_pipeline.verify.return_value = VerificationResult(
            verified=False,
            constraints=[violation],
            energy=1.0,
            violations=[violation],
            certificate={"total_energy": 1.0, "per_constraint": []},
        )
        mock_pipeline.extract_constraints.return_value = [violation]

        propagate(step, state, mock_pipeline)

        # The future assumed fact should now be VIOLATED.
        assert state.facts["future_claim"].status == FactStatus.VIOLATED
        assert state.facts["future_claim"].verified_at_step == 2
        assert state.facts["future_claim"].energy == 1.0
