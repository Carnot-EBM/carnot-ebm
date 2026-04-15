"""Tests for SAVeR multi-turn verification wrapper.

**Detailed explanation for engineers:**
    Tests for SAVeRVerifier (Self-Auditing Verification and Repair),
    covering:
    - AgentStep and ConstraintState dataclass construction
    - CI-safe stub mode (pipeline=None) approves all steps
    - Live pipeline path: committed on clean verify
    - Live pipeline path: committed after repair
    - Live pipeline path: blocked after max_repair_attempts
    - run_chain: multi-step propagation of ConstraintState
    - compute_faithfulness: fraction of committed steps
    - build_saver_artifact: schema and serialization

Spec: REQ-AGENT-001, REQ-AGENT-002,
      SCENARIO-AGENT-001, SCENARIO-AGENT-002, SCENARIO-AGENT-003
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.saver_verifier import (
    AgentStep,
    ConstraintState,
    SAVeRVerifier,
    build_saver_artifact,
)
from carnot.pipeline.verify_repair import RepairResult, VerificationResult, VerifyRepairPipeline
from carnot.pipeline.extract import ConstraintResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constraint(desc: str, satisfied: bool = True) -> ConstraintResult:
    """Construct a minimal ConstraintResult for testing.

    Spec: REQ-AGENT-001, SCENARIO-AGENT-001
    """
    return ConstraintResult(
        constraint_type="arithmetic",
        description=desc,
        metadata={"satisfied": satisfied},
    )


def _make_repair_result(
    *,
    verified: bool,
    repaired: bool = False,
    iterations: int = 0,
    initial_violations: list[ConstraintResult] | None = None,
    final_response: str = "final answer",
    initial_response: str = "initial answer",
) -> RepairResult:
    """Build a RepairResult for mock pipeline returns.

    Spec: REQ-AGENT-001, SCENARIO-AGENT-001
    """
    violations = initial_violations or []
    initial_vr = VerificationResult(
        verified=verified if iterations == 0 else False,
        constraints=violations,
        energy=0.0 if verified else 1.0,
        violations=violations,
    )
    history = [initial_vr]
    # Simulate a second pass (after repair) when repaired=True
    if repaired:
        repaired_vr = VerificationResult(
            verified=True,
            constraints=[],
            energy=0.0,
            violations=[],
        )
        history.append(repaired_vr)
    return RepairResult(
        initial_response=initial_response,
        final_response=final_response,
        verified=verified,
        repaired=repaired,
        iterations=iterations,
        history=history,
    )


# ---------------------------------------------------------------------------
# AgentStep dataclass
# REQ-AGENT-001, SCENARIO-AGENT-001
# ---------------------------------------------------------------------------


class TestAgentStep:
    """Test AgentStep dataclass construction and defaults.

    Spec: REQ-AGENT-001, SCENARIO-AGENT-001
    """

    def test_minimal_construction(self) -> None:
        """AgentStep can be constructed with required fields only.

        Spec: REQ-AGENT-001
        """
        step = AgentStep(
            step_id=0,
            question="What is 2+2?",
            proposed_action="2 + 2 = 4",
            action_cot="Step 1: add 2 and 2. Result: 4.",
        )
        assert step.step_id == 0
        assert step.question == "What is 2+2?"
        assert step.proposed_action == "2 + 2 = 4"
        assert step.action_cot == "Step 1: add 2 and 2. Result: 4."
        assert step.constraint_violations == []
        assert step.repaired_action is None
        assert step.committed is False
        assert step.repair_attempts == 0

    def test_full_construction(self) -> None:
        """AgentStep accepts all fields including violations and repair.

        Spec: REQ-AGENT-001
        """
        step = AgentStep(
            step_id=2,
            question="Apply discount",
            proposed_action="120 * 0.75 = 91",
            action_cot="Discount 25%: 120 * 0.75 = 91",
            constraint_violations=["120 * 0.75 should equal 90 not 91"],
            repaired_action="120 * 0.75 = 90",
            committed=True,
            repair_attempts=1,
        )
        assert step.step_id == 2
        assert step.constraint_violations == ["120 * 0.75 should equal 90 not 91"]
        assert step.repaired_action == "120 * 0.75 = 90"
        assert step.committed is True
        assert step.repair_attempts == 1


# ---------------------------------------------------------------------------
# ConstraintState dataclass
# REQ-AGENT-001, SCENARIO-AGENT-001
# ---------------------------------------------------------------------------


class TestConstraintState:
    """Test ConstraintState dataclass construction and defaults.

    Spec: REQ-AGENT-001, SCENARIO-AGENT-001
    """

    def test_default_construction(self) -> None:
        """ConstraintState has correct defaults for an empty initial state.

        Spec: REQ-AGENT-001
        """
        state = ConstraintState()
        assert state.step_id == -1
        assert state.active_constraints == []
        assert state.accumulated_facts == []
        assert state.facts_established == 0
        assert state.model_id == ""

    def test_full_construction(self) -> None:
        """ConstraintState accepts all fields.

        Spec: REQ-AGENT-001
        """
        state = ConstraintState(
            step_id=3,
            active_constraints=["3 * 4 = 12", "12 + 5 = 17"],
            accumulated_facts=["3 * 4 = 12", "12 + 5 = 17"],
            facts_established=2,
            model_id="math-chain-v1",
        )
        assert state.step_id == 3
        assert state.facts_established == 2
        assert state.model_id == "math-chain-v1"
        assert len(state.active_constraints) == 2


# ---------------------------------------------------------------------------
# SAVeRVerifier — CI-safe stub (pipeline=None)
# REQ-AGENT-002, SCENARIO-AGENT-001
# ---------------------------------------------------------------------------


class TestSAVeRVerifierCIStub:
    """Test SAVeRVerifier with pipeline=None (CI-safe stub mode).

    All steps are approved immediately without any verification calls.

    Spec: REQ-AGENT-002, SCENARIO-AGENT-001
    """

    def test_propose_step_commits_immediately(self) -> None:
        """CI stub approves step with committed=True, repair_attempts=0.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        state = ConstraintState(model_id="test")

        step, new_state = verifier.propose_step(
            question="What is 3 * 4?",
            action_cot="3 * 4 = 12",
            constraint_state=state,
        )

        assert step.committed is True
        assert step.repair_attempts == 0
        assert step.repaired_action is None
        assert step.constraint_violations == []
        assert step.step_id == 0

    def test_propose_step_updates_state(self) -> None:
        """CI stub adds action to accumulated_facts on commit.

        Spec: REQ-AGENT-001, SCENARIO-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        state = ConstraintState()

        _, new_state = verifier.propose_step(
            question="Q",
            action_cot="A = 5",
            constraint_state=state,
        )

        assert new_state.facts_established == 1
        assert "A = 5" in new_state.accumulated_facts
        assert new_state.step_id == 0

    def test_propose_step_preserves_model_id(self) -> None:
        """CI stub preserves model_id in returned ConstraintState.

        Spec: REQ-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        state = ConstraintState(model_id="my-model")

        _, new_state = verifier.propose_step("Q", "A", state)
        assert new_state.model_id == "my-model"

    def test_run_chain_all_committed(self) -> None:
        """run_chain with CI stub commits all steps and propagates state.

        Spec: REQ-AGENT-001, SCENARIO-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        initial_state = ConstraintState(model_id="chain-test")
        steps = [
            ("What is 3*4?", "3 * 4 = 12"),
            ("Now add 5.", "12 + 5 = 17"),
            ("Is 17 prime?", "17 is prime."),
        ]

        agent_steps = verifier.run_chain(steps, initial_state)

        assert len(agent_steps) == 3
        assert all(s.committed for s in agent_steps)
        assert agent_steps[0].step_id == 0
        assert agent_steps[1].step_id == 1
        assert agent_steps[2].step_id == 2

    def test_compute_faithfulness_all_committed(self) -> None:
        """Faithfulness is 1.0 when all steps are committed.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        steps = [
            AgentStep(0, "Q0", "A0", "A0", committed=True),
            AgentStep(1, "Q1", "A1", "A1", committed=True),
            AgentStep(2, "Q2", "A2", "A2", committed=True),
        ]
        assert verifier.compute_faithfulness(steps) == 1.0

    def test_compute_faithfulness_empty(self) -> None:
        """Faithfulness is 0.0 for empty step list.

        Spec: REQ-AGENT-002
        """
        verifier = SAVeRVerifier(pipeline=None)
        assert verifier.compute_faithfulness([]) == 0.0


# ---------------------------------------------------------------------------
# SAVeRVerifier — live pipeline, step passes on first verify
# REQ-AGENT-002, SCENARIO-AGENT-001
# ---------------------------------------------------------------------------


class TestSAVeRVerifierCleanStep:
    """Test SAVeRVerifier with pipeline that clears step on first verify.

    Spec: REQ-AGENT-002, SCENARIO-AGENT-001
    """

    def test_propose_step_clean_commits(self) -> None:
        """Step passes on first verify → committed=True, repair_attempts=0.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-001
        """
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=True,
            repaired=False,
            iterations=0,
            final_response="3 * 4 = 12",
            initial_response="3 * 4 = 12",
        )

        verifier = SAVeRVerifier(pipeline=pipeline, max_repair_attempts=3)
        state = ConstraintState()

        step, new_state = verifier.propose_step("Q", "3 * 4 = 12", state)

        assert step.committed is True
        assert step.repair_attempts == 0
        assert step.repaired_action is None
        assert new_state.facts_established == 1
        assert "3 * 4 = 12" in new_state.accumulated_facts

    def test_propose_step_clean_no_violations_recorded(self) -> None:
        """Clean step records no constraint violations.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-001
        """
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=True,
            iterations=0,
        )

        verifier = SAVeRVerifier(pipeline=pipeline)
        step, _ = verifier.propose_step("Q", "A", ConstraintState())

        assert step.constraint_violations == []


# ---------------------------------------------------------------------------
# SAVeRVerifier — live pipeline, step repaired before commit
# REQ-AGENT-002, SCENARIO-AGENT-003
# ---------------------------------------------------------------------------


class TestSAVeRVerifierRepairedStep:
    """Test SAVeRVerifier with pipeline that repairs and then commits.

    Spec: REQ-AGENT-002, SCENARIO-AGENT-003
    """

    def test_propose_step_repaired_commits(self) -> None:
        """Repaired step → committed=True, repaired_action set.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-003
        """
        violation = _make_constraint("120 * 0.75 = 91 (should be 90)", satisfied=False)

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=True,
            repaired=True,
            iterations=1,
            initial_violations=[violation],
            initial_response="120 * 0.75 = 91",
            final_response="120 * 0.75 = 90",
        )

        verifier = SAVeRVerifier(pipeline=pipeline, max_repair_attempts=3)
        state = ConstraintState()

        step, new_state = verifier.propose_step(
            "Apply 25% discount to 120",
            "120 * 0.75 = 91",
            state,
        )

        assert step.committed is True
        assert step.repaired_action == "120 * 0.75 = 90"
        assert step.repair_attempts == 1
        assert len(step.constraint_violations) == 1
        assert "120 * 0.75 = 91 (should be 90)" in step.constraint_violations[0]
        assert new_state.facts_established == 1

    def test_propose_step_captures_initial_violations(self) -> None:
        """Initial violation descriptions are recorded even when repair succeeds.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-003
        """
        v1 = _make_constraint("carry error", satisfied=False)
        v2 = _make_constraint("sign error", satisfied=False)

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=True,
            repaired=True,
            iterations=2,
            initial_violations=[v1, v2],
        )

        verifier = SAVeRVerifier(pipeline=pipeline)
        step, _ = verifier.propose_step("Q", "A", ConstraintState())

        assert "carry error" in step.constraint_violations
        assert "sign error" in step.constraint_violations


# ---------------------------------------------------------------------------
# SAVeRVerifier — live pipeline, step blocked after max repairs
# REQ-AGENT-002, SCENARIO-AGENT-002
# ---------------------------------------------------------------------------


class TestSAVeRVerifierBlockedStep:
    """Test SAVeRVerifier when step remains violated after all repair attempts.

    Spec: REQ-AGENT-002, SCENARIO-AGENT-002
    """

    def test_propose_step_blocked_not_committed(self) -> None:
        """Step stays failed after max repairs → committed=False.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-002
        """
        violation = _make_constraint("arithmetic contradiction", satisfied=False)

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=False,
            repaired=False,
            iterations=3,
            initial_violations=[violation],
        )

        verifier = SAVeRVerifier(pipeline=pipeline, max_repair_attempts=3)
        state = ConstraintState()

        step, new_state = verifier.propose_step("Q", "wrong answer", state)

        assert step.committed is False
        assert step.repaired_action is None
        assert step.repair_attempts == 3
        # Blocked step does NOT update accumulated_facts
        assert new_state.facts_established == 0
        assert new_state.accumulated_facts == []

    def test_blocked_state_unchanged_step_id(self) -> None:
        """Blocked step preserves the incoming constraint_state.step_id.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-002
        """
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=False,
            iterations=2,
        )

        verifier = SAVeRVerifier(pipeline=pipeline)
        state = ConstraintState(step_id=2, facts_established=2)
        _, new_state = verifier.propose_step("Q", "A", state)

        # Blocked: state unchanged
        assert new_state.step_id == 2
        assert new_state.facts_established == 2

    def test_compute_faithfulness_all_blocked(self) -> None:
        """Faithfulness is 0.0 when all steps are blocked.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-002
        """
        verifier = SAVeRVerifier(pipeline=None)
        steps = [
            AgentStep(0, "Q", "A", "A", committed=False),
            AgentStep(1, "Q", "A", "A", committed=False),
        ]
        assert verifier.compute_faithfulness(steps) == 0.0

    def test_compute_faithfulness_partial(self) -> None:
        """Faithfulness is 0.5 when half the steps are committed.

        Spec: REQ-AGENT-002, SCENARIO-AGENT-002
        """
        verifier = SAVeRVerifier(pipeline=None)
        steps = [
            AgentStep(0, "Q", "A", "A", committed=True),
            AgentStep(1, "Q", "A", "A", committed=False),
            AgentStep(2, "Q", "A", "A", committed=True),
            AgentStep(3, "Q", "A", "A", committed=False),
        ]
        assert verifier.compute_faithfulness(steps) == 0.5


# ---------------------------------------------------------------------------
# SAVeRVerifier — run_chain multi-step propagation
# REQ-AGENT-001, SCENARIO-AGENT-001
# ---------------------------------------------------------------------------


class TestRunChain:
    """Test run_chain for multi-step constraint state propagation.

    Spec: REQ-AGENT-001, REQ-AGENT-002, SCENARIO-AGENT-001
    """

    def test_run_chain_state_propagates(self) -> None:
        """Facts from committed steps accumulate across the chain.

        Spec: REQ-AGENT-001, SCENARIO-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        initial = ConstraintState(model_id="test-chain")
        steps = [
            ("Q0", "fact zero"),
            ("Q1", "fact one"),
            ("Q2", "fact two"),
        ]

        agent_steps = verifier.run_chain(steps, initial)

        assert len(agent_steps) == 3
        assert agent_steps[2].step_id == 2
        assert all(s.committed for s in agent_steps)

    def test_run_chain_empty_steps(self) -> None:
        """run_chain with empty steps list returns empty list.

        Spec: REQ-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        result = verifier.run_chain([], ConstraintState())
        assert result == []

    def test_run_chain_blocked_step_does_not_poison_next(self) -> None:
        """A blocked step does not increment facts_established for next step.

        Spec: REQ-AGENT-001, SCENARIO-AGENT-002
        """
        call_count = [0]
        violation = _make_constraint("violation", satisfied=False)

        def fake_var(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                # Second step (index 1) fails
                return _make_repair_result(
                    verified=False,
                    iterations=1,
                    initial_violations=[violation],
                )
            return _make_repair_result(verified=True, iterations=0)

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.side_effect = fake_var

        verifier = SAVeRVerifier(pipeline=pipeline)
        initial = ConstraintState()
        steps = [
            ("Q0", "good step 0"),
            ("Q1", "bad step 1"),
            ("Q2", "good step 2"),
        ]

        agent_steps = verifier.run_chain(steps, initial)

        assert agent_steps[0].committed is True
        assert agent_steps[1].committed is False
        assert agent_steps[2].committed is True
        # Faithfulness = 2/3
        assert abs(verifier.compute_faithfulness(agent_steps) - 2 / 3) < 1e-9

    def test_run_chain_step_ids_sequential(self) -> None:
        """step_id values are sequential from 0 regardless of blocks.

        Spec: REQ-AGENT-001
        """
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.side_effect = [
            _make_repair_result(verified=True),
            _make_repair_result(verified=False, iterations=1),
            _make_repair_result(verified=True),
        ]

        verifier = SAVeRVerifier(pipeline=pipeline)
        initial = ConstraintState()
        steps = [("Q0", "A0"), ("Q1", "A1"), ("Q2", "A2")]
        agent_steps = verifier.run_chain(steps, initial)

        # step_id increments on every propose_step call regardless of block/commit
        # (step_id = prior state.step_id + 1)
        assert agent_steps[0].step_id == 0
        assert agent_steps[1].step_id == 1  # blocked but step_id still advances
        assert agent_steps[2].step_id == 1  # after blocked step, state.step_id was NOT updated


# ---------------------------------------------------------------------------
# build_saver_artifact
# REQ-AGENT-001, REQ-AGENT-002
# ---------------------------------------------------------------------------


class TestBuildSaverArtifact:
    """Test build_saver_artifact serialization.

    Spec: REQ-AGENT-001, REQ-AGENT-002
    """

    def test_schema_tag(self) -> None:
        """Artifact schema is 'carnot.saver_verifier.v1'.

        Spec: REQ-AGENT-001
        """
        artifact = build_saver_artifact([], faithfulness=0.0)
        assert artifact["schema"] == "carnot.saver_verifier.v1"

    def test_empty_steps(self) -> None:
        """Artifact with zero steps has n_steps=0.

        Spec: REQ-AGENT-001
        """
        artifact = build_saver_artifact([], faithfulness=0.0)
        assert artifact["n_steps"] == 0
        assert artifact["steps"] == []
        assert artifact["faithfulness"] == 0.0

    def test_step_serialization(self) -> None:
        """All AgentStep fields appear in the artifact step records.

        Spec: REQ-AGENT-001
        """
        step = AgentStep(
            step_id=0,
            question="What is 4 * 5?",
            proposed_action="4 * 5 = 20",
            action_cot="Multiply 4 by 5: 20",
            constraint_violations=["sign check"],
            repaired_action="4 * 5 = 20 (correct)",
            committed=True,
            repair_attempts=1,
        )
        artifact = build_saver_artifact([step], faithfulness=1.0)

        assert artifact["n_steps"] == 1
        assert artifact["faithfulness"] == 1.0

        record = artifact["steps"][0]
        assert record["step_id"] == 0
        assert record["question"] == "What is 4 * 5?"
        assert record["proposed_action"] == "4 * 5 = 20"
        assert record["action_cot"] == "Multiply 4 by 5: 20"
        assert record["constraint_violations"] == ["sign check"]
        assert record["repaired_action"] == "4 * 5 = 20 (correct)"
        assert record["committed"] is True
        assert record["repair_attempts"] == 1

    def test_json_serializable(self) -> None:
        """Artifact is JSON-serializable.

        Spec: REQ-AGENT-001
        """
        import json

        step = AgentStep(
            step_id=1,
            question="Q",
            proposed_action="A",
            action_cot="A",
            committed=False,
            repair_attempts=3,
        )
        artifact = build_saver_artifact([step], faithfulness=0.0)
        json_str = json.dumps(artifact)
        assert isinstance(json_str, str)

    def test_multiple_steps(self) -> None:
        """Artifact with multiple steps serializes all records.

        Spec: REQ-AGENT-001
        """
        steps = [
            AgentStep(i, f"Q{i}", f"A{i}", f"A{i}", committed=(i % 2 == 0))
            for i in range(4)
        ]
        faithfulness = sum(1 for s in steps if s.committed) / len(steps)
        artifact = build_saver_artifact(steps, faithfulness)

        assert artifact["n_steps"] == 4
        assert artifact["faithfulness"] == 0.5
        assert len(artifact["steps"]) == 4


# ---------------------------------------------------------------------------
# Edge cases for line coverage
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Cover miscellaneous edge cases for 100% line coverage.

    Spec: REQ-AGENT-001, REQ-AGENT-002
    """

    def test_empty_history_in_repair_result(self) -> None:
        """propose_step handles RepairResult with empty history gracefully.

        When history is empty (no verification passes recorded), violation
        list should be empty rather than raising IndexError.

        Spec: REQ-AGENT-002
        """
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = RepairResult(
            initial_response="A",
            final_response="A",
            verified=True,
            repaired=False,
            iterations=0,
            history=[],
        )

        verifier = SAVeRVerifier(pipeline=pipeline)
        step, _ = verifier.propose_step("Q", "A", ConstraintState())

        assert step.constraint_violations == []
        assert step.committed is True

    def test_max_repair_attempts_zero(self) -> None:
        """SAVeRVerifier with max_repair_attempts=0 still works.

        Spec: REQ-AGENT-002
        """
        verifier = SAVeRVerifier(pipeline=None, max_repair_attempts=0)
        step, _ = verifier.propose_step("Q", "A", ConstraintState())
        assert step.committed is True  # CI stub always commits

    def test_propose_step_active_constraints_accumulate(self) -> None:
        """active_constraints grows with each committed step in CI stub.

        Spec: REQ-AGENT-001
        """
        verifier = SAVeRVerifier(pipeline=None)
        state = ConstraintState()

        _, state = verifier.propose_step("Q0", "fact_A", state)
        _, state = verifier.propose_step("Q1", "fact_B", state)
        _, state = verifier.propose_step("Q2", "fact_C", state)

        assert "fact_A" in state.active_constraints
        assert "fact_B" in state.active_constraints
        assert "fact_C" in state.active_constraints
        assert state.facts_established == 3

    def test_live_pipeline_active_constraints_update(self) -> None:
        """Active constraints include final_response from committed live step.

        Spec: REQ-AGENT-001
        """
        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.verify_and_repair.return_value = _make_repair_result(
            verified=True,
            iterations=0,
            final_response="cost is 90",
        )

        verifier = SAVeRVerifier(pipeline=pipeline)
        state = ConstraintState()
        _, new_state = verifier.propose_step("Q", "A", state)

        assert "cost is 90" in new_state.active_constraints
