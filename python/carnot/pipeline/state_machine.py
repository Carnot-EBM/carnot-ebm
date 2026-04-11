"""Production-grade ConstraintStateMachine for agent framework integration.

**Detailed explanation for engineers:**
    This module wraps the lower-level ConstraintState + propagate() machinery
    (from carnot.pipeline.agentic) into a stateful machine that agent
    frameworks can integrate step by step. Each call to step() advances the
    machine: it extracts constraints from the output, runs verification, flags
    contradictions against previously verified facts, updates accumulated
    state, and records a StepResult for audit.

    Key features:
    - Full step history with per-step verification results and state snapshots.
    - rollback(to_step) restores the machine to an earlier state using stored
      deep copies of the ConstraintState (not just the snapshot dict), so the
      machine can continue from that point.
    - Contradiction detection: a contradiction is raised when a violation in
      the current step targets a constraint that was already VERIFIED in a
      prior step -- meaning the new output contradicts a previously confirmed
      fact.
    - verified_facts() and pending_facts() provide quick access to the current
      set of VERIFIED and ASSUMED facts respectively.

    Architecture:
    - StepResult: immutable record of one step's outcome (new facts, violations,
      contradictions, state snapshot).
    - _StepRecord: internal record pairing a StepResult with a deep copy of
      the ConstraintState for rollback support.
    - ConstraintStateMachine: the main class. Wraps a VerifyRepairPipeline and
      manages state transitions.

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from carnot.pipeline.agentic import (
    AgentStep,
    ConstraintState,
    FactStatus,
    TrackedFact,
    _normalize_fact_key,
    propagate,
)
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline

if TYPE_CHECKING:
    from carnot.pipeline.consistency_checker import GlobalConsistencyReport


# ---------------------------------------------------------------------------
# StepResult — immutable record of one machine step
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Record of one step through the ConstraintStateMachine.

    **Detailed explanation for engineers:**
        Captures everything that happened when step() was called: which step
        index this was, the full verification result (verified flag, energy,
        violations), the list of new facts introduced in this step, any
        contradictions detected (facts that violate previously verified facts),
        and a JSON-serializable snapshot of the full constraint state after
        this step completes.

    Attributes:
        step_index: Zero-based index of this step in the machine's history.
        verification: Full VerificationResult from the pipeline for this step.
        new_facts: TrackedFact objects added to state for the first time in
            this step (facts that already existed are not repeated).
        contradictions: Description strings of facts that violated constraints
            that were already VERIFIED in a prior step -- meaning the new
            output directly contradicts a previously confirmed fact.
        state_snapshot: JSON-serializable snapshot (dict) of the full
            ConstraintState after this step. Suitable for logging and
            debugging. Does not replace rollback -- use rollback() for that.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    step_index: int
    verification: VerificationResult
    new_facts: list[TrackedFact]
    contradictions: list[str]
    state_snapshot: dict
    output_text: str = ""
    """Raw output text for this step, stored for global consistency checking (Exp 172)."""


# ---------------------------------------------------------------------------
# Internal record pairing StepResult with a ConstraintState copy
# ---------------------------------------------------------------------------


@dataclass
class _StepRecord:
    """Internal record linking a StepResult to a restorable state copy.

    **Detailed explanation for engineers:**
        The state_copy is a deep copy of the ConstraintState taken immediately
        after the step completes. rollback(n) restores the current state to
        this copy, then trims history to only entries through step n.
        Stored internally; not exposed in the public API.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    result: StepResult
    state_copy: ConstraintState


# ---------------------------------------------------------------------------
# ConstraintStateMachine
# ---------------------------------------------------------------------------


class ConstraintStateMachine:
    """Production-grade stateful constraint tracker for agent frameworks.

    **Detailed explanation for engineers:**
        Wraps a VerifyRepairPipeline and a ConstraintState to provide a clean,
        rollback-capable step-by-step verification interface. Each call to
        step() advances the machine by one reasoning turn:

        1. The current (before-step) set of VERIFIED fact keys is recorded.
        2. An AgentStep is created and propagate() is called, which:
           a. Extracts constraints from output_text via the pipeline.
           b. Verifies against the full output via the pipeline.
           c. Updates the ConstraintState (add facts, mark verified/violated).
        3. New facts (introduced for the first time in this step) are identified
           by comparing the fact set before and after.
        4. Contradictions are identified: any violation in this step whose
           normalized key was already VERIFIED before this step means the new
           output contradicts a previously confirmed fact.
        5. A deep copy of the ConstraintState is stored for rollback.
        6. A StepResult is returned and appended to history.

        rollback(to_step) restores the machine to the state it had immediately
        after step `to_step` completed. Subsequent calls to step() continue
        from that restored state.

    Args:
        pipeline: A VerifyRepairPipeline instance used for constraint
            extraction and verification. If None, a default pipeline is
            created with no model (verify-only mode).

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def __init__(self, pipeline: VerifyRepairPipeline | None = None) -> None:
        self._pipeline: VerifyRepairPipeline = (
            pipeline if pipeline is not None else VerifyRepairPipeline()
        )
        self._state: ConstraintState = ConstraintState()
        self._records: list[_StepRecord] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def step(self, input_text: str, output_text: str) -> StepResult:
        """Advance the machine by one reasoning step.

        **Detailed explanation for engineers:**
            Extracts constraints from output_text, verifies the output against
            accumulated state, detects contradictions (violations of previously
            VERIFIED facts), updates the internal ConstraintState, and records
            the StepResult in history.

            The step_index is automatically assigned as the next sequential
            index (0-based). After rollback(), the next step_index continues
            from the restored point, not from the end of the original history.

        Args:
            input_text: The prompt/question that generated this output. Used
                for context in the AgentStep record.
            output_text: The LLM or agent's response for this turn. Constraints
                are extracted from this text.

        Returns:
            StepResult describing what happened: new facts, verification
            outcome, contradictions, and a state snapshot.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        step_index = len(self._records)

        # Record which keys were already VERIFIED before this step.
        # Contradictions are violations of these keys.
        pre_verified_keys: set[str] = {
            _normalize_fact_key(f.fact)
            for f in self._state.get_verified()
        }

        # Record which fact keys existed before this step to find new additions.
        pre_existing_keys: set[str] = set(self._state.facts.keys())

        # Create the AgentStep and run propagate().
        agent_step = AgentStep(
            step_index=step_index,
            input_text=input_text,
            output_text=output_text,
        )
        propagate(self._state, agent_step, self._pipeline)

        # Collect new facts introduced in this step.
        new_fact_keys = set(self._state.facts.keys()) - pre_existing_keys
        new_facts: list[TrackedFact] = [
            self._state.facts[k] for k in new_fact_keys
        ]

        # Detect contradictions: violations in this step whose keys were
        # already VERIFIED before the step (the new output contradicts a
        # previously confirmed fact).
        contradictions: list[str] = []
        if agent_step.verification and agent_step.verification.violations:
            for v in agent_step.verification.violations:
                vkey = _normalize_fact_key(v.description)
                if vkey in pre_verified_keys:
                    contradictions.append(v.description)

        # Build the StepResult (use state_after snapshot captured by propagate).
        state_snapshot = agent_step.state_after

        result = StepResult(
            step_index=step_index,
            verification=agent_step.verification
            or VerificationResult(
                verified=True, constraints=[], energy=0.0, violations=[]
            ),
            new_facts=new_facts,
            contradictions=contradictions,
            state_snapshot=state_snapshot,
            output_text=output_text,
        )

        # Store a deep copy of the state for rollback support.
        record = _StepRecord(
            result=result,
            state_copy=copy.deepcopy(self._state),
        )
        self._records.append(record)

        return result

    def rollback(self, to_step: int) -> None:
        """Restore the machine to its state immediately after step `to_step`.

        **Detailed explanation for engineers:**
            Discards all records for steps after `to_step` and restores the
            internal ConstraintState to the deep copy stored when step
            `to_step` completed. Subsequent calls to step() continue from
            this restored state, with step_index starting from to_step + 1.

            This enables agent frameworks to retry reasoning from an earlier
            known-good state without replaying the full history.

        Args:
            to_step: The step index to roll back to (inclusive). Must be a
                valid index in the current history (0 <= to_step < len(history)).

        Raises:
            IndexError: If to_step is out of range.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        if to_step < 0 or to_step >= len(self._records):
            raise IndexError(
                f"to_step={to_step} is out of range "
                f"(history has {len(self._records)} steps, indices 0..{len(self._records) - 1})"
            )

        # Restore state from the deep copy stored after step `to_step`.
        self._state = copy.deepcopy(self._records[to_step].state_copy)
        # Trim history to only include steps 0..to_step.
        self._records = self._records[: to_step + 1]

    def history(self) -> list[StepResult]:
        """Return all StepResults in order from step 0 to the latest.

        **Detailed explanation for engineers:**
            Returns a new list (not a reference to the internal list) so
            callers cannot mutate the machine's history. Each StepResult is
            the original object (not a copy) -- callers should treat them
            as read-only.

        Returns:
            List of StepResult, one per completed step, in step order.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return [record.result for record in self._records]

    def verified_facts(self) -> list[TrackedFact]:
        """Return all facts currently marked VERIFIED in the constraint state.

        **Detailed explanation for engineers:**
            Delegates to ConstraintState.get_verified(). The returned list
            reflects the state at the time of the call, including any rollback
            effects. After a rollback, facts verified in the rolled-back steps
            are no longer present.

        Returns:
            List of TrackedFact objects with status VERIFIED.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return self._state.get_verified()

    def pending_facts(self) -> list[TrackedFact]:
        """Return all facts currently marked ASSUMED (pending verification).

        **Detailed explanation for engineers:**
            Delegates to ConstraintState.get_assumed(). Facts start as ASSUMED
            when first introduced and transition to VERIFIED or VIOLATED as
            steps are processed. ASSUMED facts are those that have been
            extracted but not yet fully verified or violated.

        Returns:
            List of TrackedFact objects with status ASSUMED.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return self._state.get_assumed()

    def check_global_consistency(self) -> "GlobalConsistencyReport":
        """Check all completed steps for global cross-step consistency (Exp 172).

        **Detailed explanation for engineers:**
            Delegates to GlobalConsistencyChecker().check(self). Each step's
            output_text is inspected for three types of cross-step contradiction:
            numeric (same entity, different value), arithmetic (same equation,
            different claimed result), and factual (same subject+predicate,
            different object). Local per-step verification cannot catch these
            because it checks each step in isolation.

            Meaningful only after at least 2 steps have been run via step().
            With 0 or 1 steps, returns a trivially consistent report.

            The import of GlobalConsistencyChecker is deferred to the method
            body to avoid a circular import at module load time (consistency_checker
            TYPE_CHECKING-imports ConstraintStateMachine for type hints only).

        Returns:
            GlobalConsistencyReport with consistent flag, list of contradicting
            (i, j, type, description) pairs, severity level, and
            recommended_rollback_step.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        # Deferred import prevents circular dependency at load time.
        from carnot.pipeline.consistency_checker import GlobalConsistencyChecker

        return GlobalConsistencyChecker().check(self)


__all__ = [
    "ConstraintStateMachine",
    "StepResult",
]
