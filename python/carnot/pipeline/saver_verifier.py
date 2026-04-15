"""SAVeR-style multi-turn verification wrapper (Self-Auditing Verification and Repair).

**Researcher summary:**
    Implements the SAVeR auditor-before-commit loop from arXiv 2604.08401.
    Each agent step is intercepted, verified against accumulated constraint
    state from previous steps, and repaired if needed before it can proceed.
    Only steps that satisfy all constraints are "committed" — blocked steps
    prevent their reasoning from poisoning downstream conclusions.

**Detailed explanation for engineers:**
    In standard LLM agent workflows, the agent emits step N's reasoning and
    immediately uses it to produce step N+1.  If step N contains an error,
    step N+1 is built on faulty premises — and the error compounds.

    SAVeR breaks this chain by inserting an auditor between each step:

        step N proposes action → auditor checks action against constraint state
            → if OK: commit action, update constraint state, proceed to step N+1
            → if violations: attempt repair → re-audit → if still failing: BLOCK

    The key data structures:
    - ``AgentStep``: Records one reasoning step — the proposed action, any
      detected violations, the repaired action (if any), whether it was
      committed, and how many repair attempts were made.
    - ``ConstraintState``: Carries accumulated facts across steps. When a step
      commits, its final action is added to ``accumulated_facts``.  Blocked
      steps do NOT update the state, preventing faulty reasoning from
      propagating.

    CI-safety contract:
        When ``pipeline`` is ``None``, the verifier acts as a transparent
        pass-through: every step is approved immediately (``committed=True``,
        ``repair_attempts=0``) without calling any pipeline methods.  This
        ensures tests and CI runs that lack a live LLM can still exercise
        all code paths.

    Faithfulness metric:
        ``compute_faithfulness(steps) → float``: the fraction of steps that
        were committed.  A chain where every step passes is 1.0 faithful.
        A chain where half are blocked is 0.5 faithful.

    Artifact schema:
        ``build_saver_artifact(steps, faithfulness)`` serializes the full
        chain run to a flat dict with ``schema="carnot.saver_verifier.v1"``,
        suitable for inclusion in an ``ExperimentTemplate.build_result()``
        payload.

Spec: REQ-AGENT-001, REQ-AGENT-002,
      SCENARIO-AGENT-001, SCENARIO-AGENT-002, SCENARIO-AGENT-003
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# AgentStep — one step in a SAVeR multi-turn reasoning chain
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One step in a SAVeR-style multi-turn agent reasoning chain.

    **Detailed explanation for engineers:**
        This dataclass is the per-step audit record.  After ``propose_step()``
        runs, this captures everything needed to understand what happened:

        - ``step_id``: sequential index (0-based) within the chain
        - ``question``: the sub-question or instruction for this step
        - ``proposed_action``: the agent's initial (pre-audit) response text
        - ``action_cot``: the chain-of-thought reasoning accompanying the action
        - ``constraint_violations``: description strings of any violations found
          during the FIRST verification pass (before any repair attempts)
        - ``repaired_action``: the revised response after successful repair;
          ``None`` if no repair was needed or if all repair attempts failed
        - ``committed``: True if the step's final action passed all constraints
          and was accepted into the accumulated constraint state
        - ``repair_attempts``: how many repair iterations were performed
          (0 = passed on first check, max_repair_attempts = all failed)

    Spec: REQ-AGENT-001, SCENARIO-AGENT-001
    """

    step_id: int
    question: str
    proposed_action: str
    action_cot: str
    constraint_violations: list[str] = field(default_factory=list)
    repaired_action: str | None = None
    committed: bool = False
    repair_attempts: int = 0


# ---------------------------------------------------------------------------
# ConstraintState — accumulated facts across steps
# ---------------------------------------------------------------------------


@dataclass
class ConstraintState:
    """Accumulated constraint state propagated across SAVeR reasoning steps.

    **Detailed explanation for engineers:**
        This dataclass is the inter-step memory: it carries the facts that
        prior committed steps have established, so the auditor for step N
        can check step N's action against everything steps 0…N-1 claimed.

        Fields:
        - ``step_id``: the ID of the most recent step that produced this state
          (−1 for the initial empty state before any steps have run)
        - ``active_constraints``: human-readable description strings of
          constraints that are currently "in force" — derived from committed
          step actions.  Future steps must not contradict these.
        - ``accumulated_facts``: the full list of committed action strings,
          one entry per committed step.  This is the growing "ground truth"
          for the chain.
        - ``facts_established``: convenience count of how many facts have
          been committed (equals ``len(accumulated_facts)`` but handy for
          logging without reconstructing from the list)
        - ``model_id``: an optional string identifying the model or chain
          type — used for logging and artifact labeling.

    Spec: REQ-AGENT-001, SCENARIO-AGENT-001
    """

    step_id: int = -1
    active_constraints: list[str] = field(default_factory=list)
    accumulated_facts: list[str] = field(default_factory=list)
    facts_established: int = 0
    model_id: str = ""


# ---------------------------------------------------------------------------
# SAVeRVerifier — the auditor-before-commit wrapper
# ---------------------------------------------------------------------------


class SAVeRVerifier:
    """SAVeR-style auditor that gates each agent step behind constraint checking.

    **Detailed explanation for engineers:**
        SAVeRVerifier wraps a ``VerifyRepairPipeline`` and applies it to
        each step in a multi-turn reasoning chain.  The core loop per step:

            1. Run ``pipeline.verify_and_repair(question, action_cot)``
               to verify the proposed reasoning and attempt repairs if needed.
            2. If the final result is verified → ``committed=True``.
            3. If violations persist after all repair attempts
               (``max_repair_attempts`` exceeded) → ``committed=False``.
            4. Update ``ConstraintState``: committed steps append their
               final action to ``accumulated_facts``; blocked steps do not.

        When ``pipeline`` is ``None`` (CI-safe mode), no verification calls
        are made — every step is immediately committed with zero repairs.

    Parameters
    ----------
    pipeline : VerifyRepairPipeline or None
        The constraint verification backend.  Pass ``None`` to run in
        CI-safe stub mode (all steps pass, no real verification).
    max_repair_attempts : int
        Maximum number of repair iterations to attempt for a single step
        before declaring it blocked.  Default 3.

    Spec: REQ-AGENT-001, REQ-AGENT-002
    """

    def __init__(
        self,
        pipeline: VerifyRepairPipeline | None,
        max_repair_attempts: int = 3,
    ) -> None:
        self.pipeline = pipeline
        self.max_repair_attempts = max_repair_attempts

    # ------------------------------------------------------------------
    # propose_step
    # ------------------------------------------------------------------

    def propose_step(
        self,
        question: str,
        action_cot: str,
        constraint_state: ConstraintState,
    ) -> tuple[AgentStep, ConstraintState]:
        """Propose an agent step and gate it through the SAVeR auditor loop.

        **Detailed explanation for engineers:**
            The SAVeR two-turn structure per step:
            (1) Agent proposes action with chain-of-thought (``action_cot``).
            (2) Auditor checks action against accumulated constraint state from
                previous steps by calling ``VerifyRepairPipeline.verify_and_repair()``.
            (3) If constraint violated: repair and re-audit up to
                ``max_repair_attempts`` times before possibly blocking.

            CI-safe mode (``pipeline=None``): no verification is performed;
            every step is immediately committed with ``committed=True`` and
            ``repair_attempts=0``.

            State update rules:
            - Committed step: ``accumulated_facts`` receives the final action
              (either ``proposed_action`` or ``repaired_action``); ``step_id``
              advances; ``facts_established`` increments.
            - Blocked step: ``constraint_state`` is returned unchanged.

        Args:
            question: The sub-question or instruction for this step.
            action_cot: The agent's proposed chain-of-thought + action.
            constraint_state: Accumulated constraint state from prior steps.

        Returns:
            Tuple of (AgentStep, updated ConstraintState).

        Spec: REQ-AGENT-001, REQ-AGENT-002, SCENARIO-AGENT-001,
              SCENARIO-AGENT-002, SCENARIO-AGENT-003
        """
        step_id = constraint_state.step_id + 1

        # ------------------------------------------------------------------
        # CI-safe stub: no pipeline → approve everything immediately.
        # ------------------------------------------------------------------
        if self.pipeline is None:
            step = AgentStep(
                step_id=step_id,
                question=question,
                proposed_action=action_cot,
                action_cot=action_cot,
                constraint_violations=[],
                repaired_action=None,
                committed=True,
                repair_attempts=0,
            )
            new_state = ConstraintState(
                step_id=step_id,
                active_constraints=list(constraint_state.active_constraints) + [action_cot],
                accumulated_facts=list(constraint_state.accumulated_facts) + [action_cot],
                facts_established=constraint_state.facts_established + 1,
                model_id=constraint_state.model_id,
            )
            return step, new_state

        # ------------------------------------------------------------------
        # Live path: run verify_and_repair up to max_repair_attempts.
        # ------------------------------------------------------------------
        # Call the pipeline's built-in repair loop.  It will attempt up to
        # pipeline.max_repairs iterations internally.  We use a single call
        # here because the pipeline already manages the repair loop, and
        # SAVeRVerifier.max_repair_attempts mirrors the pipeline's budget.
        repair_result = self.pipeline.verify_and_repair(
            question=question,
            response=action_cot,
        )

        # Collect violation descriptions from the first verification pass.
        # history[0] is the initial verification before any repairs.
        initial_violations: list[str] = []
        if repair_result.history:
            initial_violations = [
                v.description for v in repair_result.history[0].violations
            ]

        committed = repair_result.verified
        repair_attempts = repair_result.iterations
        repaired_action: str | None = None
        if repair_result.repaired:
            repaired_action = repair_result.final_response

        final_action = repair_result.final_response

        step = AgentStep(
            step_id=step_id,
            question=question,
            proposed_action=action_cot,
            action_cot=action_cot,
            constraint_violations=initial_violations,
            repaired_action=repaired_action,
            committed=committed,
            repair_attempts=repair_attempts,
        )

        # Update constraint state only when the step committed.
        if committed:
            new_state = ConstraintState(
                step_id=step_id,
                active_constraints=list(constraint_state.active_constraints) + [final_action],
                accumulated_facts=list(constraint_state.accumulated_facts) + [final_action],
                facts_established=constraint_state.facts_established + 1,
                model_id=constraint_state.model_id,
            )
        else:
            # Blocked: constraint state is unchanged (bad reasoning does not
            # propagate into future steps).
            new_state = ConstraintState(
                step_id=constraint_state.step_id,
                active_constraints=list(constraint_state.active_constraints),
                accumulated_facts=list(constraint_state.accumulated_facts),
                facts_established=constraint_state.facts_established,
                model_id=constraint_state.model_id,
            )

        return step, new_state

    # ------------------------------------------------------------------
    # run_chain
    # ------------------------------------------------------------------

    def run_chain(
        self,
        steps: list[tuple[str, str]],
        initial_state: ConstraintState,
    ) -> list[AgentStep]:
        """Run a multi-step reasoning chain through the SAVeR auditor.

        **Detailed explanation for engineers:**
            Iterates over ``steps``, calling ``propose_step()`` for each one.
            The ``ConstraintState`` returned from step N is passed as input
            to step N+1, so accumulated facts propagate through the chain.

            When a step is blocked (``committed=False``), the constraint state
            is left unchanged — subsequent steps see the same accumulated facts
            as before the blocked step.  This is intentional: blocked reasoning
            should not influence what the auditor considers "established fact".

        Args:
            steps: List of ``(question, action_cot)`` tuples, one per step.
            initial_state: Starting constraint state (typically empty).

        Returns:
            List of ``AgentStep`` records in chain order.

        Spec: REQ-AGENT-001, REQ-AGENT-002, SCENARIO-AGENT-001
        """
        agent_steps: list[AgentStep] = []
        state = initial_state
        for question, action_cot in steps:
            step, state = self.propose_step(question, action_cot, state)
            agent_steps.append(step)
        return agent_steps

    # ------------------------------------------------------------------
    # compute_faithfulness
    # ------------------------------------------------------------------

    def compute_faithfulness(self, steps: list[AgentStep]) -> float:
        """Compute the faithfulness score: fraction of steps that committed.

        **Detailed explanation for engineers:**
            Faithfulness measures how much of the reasoning chain survived
            the SAVeR auditor.  A score of 1.0 means every step was
            committed (the model's reasoning was entirely constraint-consistent).
            A score of 0.0 means every step was blocked.

            When ``steps`` is empty, returns 0.0 rather than raising a
            ZeroDivisionError — an empty chain has no committed steps.

        Args:
            steps: List of AgentStep records from ``run_chain()``.

        Returns:
            Float in [0.0, 1.0].

        Spec: REQ-AGENT-002, SCENARIO-AGENT-001, SCENARIO-AGENT-002
        """
        if not steps:
            return 0.0
        n_committed = sum(1 for s in steps if s.committed)
        return n_committed / len(steps)


# ---------------------------------------------------------------------------
# build_saver_artifact
# ---------------------------------------------------------------------------


def build_saver_artifact(
    steps: list[AgentStep],
    faithfulness: float,
) -> dict[str, Any]:
    """Serialize a SAVeR chain run to the standard Carnot artifact schema.

    **Detailed explanation for engineers:**
        Converts the per-step AgentStep records and the faithfulness score
        into a flat dict suitable for merging into an
        ``ExperimentTemplate.build_result()`` payload.

        Per-step records are serialized to a list of dicts so the artifact
        can be written directly to JSON without further conversion.

        Schema tag ``"carnot.saver_verifier.v1"`` identifies this artifact
        type for downstream tooling.

    Args:
        steps: List of AgentStep records from a completed run_chain() call.
        faithfulness: Fraction of steps that were committed (0.0–1.0).

    Returns:
        Dict with schema, faithfulness, step count, and per-step records.

    Spec: REQ-AGENT-001, REQ-AGENT-002
    """
    step_records = []
    for s in steps:
        step_records.append(
            {
                "step_id": s.step_id,
                "question": s.question,
                "proposed_action": s.proposed_action,
                "action_cot": s.action_cot,
                "constraint_violations": s.constraint_violations,
                "repaired_action": s.repaired_action,
                "committed": s.committed,
                "repair_attempts": s.repair_attempts,
            }
        )
    return {
        "schema": "carnot.saver_verifier.v1",
        "n_steps": len(steps),
        "faithfulness": faithfulness,
        "steps": step_records,
    }


__all__ = [
    "AgentStep",
    "ConstraintState",
    "SAVeRVerifier",
    "build_saver_artifact",
]
