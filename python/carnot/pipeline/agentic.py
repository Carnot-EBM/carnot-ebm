"""Multi-turn agentic verification with constraint propagation.

**Detailed explanation for engineers:**
    This module provides stateful constraint tracking across multiple
    reasoning steps in an agentic workflow. As an LLM produces multi-step
    reasoning, each step's output is verified and facts are accumulated
    in a ConstraintState. Facts start as ASSUMED, then transition to
    VERIFIED (if the constraint passes) or VIOLATED (if it fails).

    When a violation is detected, any ASSUMED facts from later steps
    are retroactively marked VIOLATED, since they were built on top
    of the now-invalid reasoning.

    Key components:
    - FactStatus: Enum for fact lifecycle (ASSUMED -> VERIFIED | VIOLATED)
    - TrackedFact: A single fact with its status and provenance
    - ConstraintState: Accumulator for facts across steps
    - AgentStep: One step in a multi-turn reasoning chain
    - propagate(): Runs extraction + verification on a step and updates state
    - _normalize_fact_key(): Deduplication via lowercase + whitespace collapse

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


class FactStatus(enum.Enum):
    """Lifecycle status of a tracked fact.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    ASSUMED = "assumed"
    VERIFIED = "verified"
    VIOLATED = "violated"


@dataclass
class TrackedFact:
    """A single fact tracked across reasoning steps.

    **Detailed explanation for engineers:**
        Each fact has a text description, a status (ASSUMED/VERIFIED/VIOLATED),
        the step index where it was first introduced, and optionally the step
        where it was verified/violated and the energy score at that point.
        An optional constraint field links back to the ConstraintResult that
        generated this fact.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    fact: str
    status: FactStatus
    introduced_at_step: int
    verified_at_step: int | None = None
    energy: float = 0.0
    constraint: ConstraintResult | None = None


def _normalize_fact_key(text: str) -> str:
    """Normalize a fact string for deduplication.

    Lowercases, collapses multiple whitespace to single spaces, and strips.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    import re

    return re.sub(r"\s+", " ", text.strip().lower())


class ConstraintState:
    """Accumulates tracked facts across multiple reasoning steps.

    **Detailed explanation for engineers:**
        This is the central state object for multi-turn verification.
        Facts are added via add_fact(), then marked verified or violated
        as verification results come in. The snapshot() method produces
        a JSON-serializable summary for logging and debugging.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def __init__(self) -> None:
        self.facts: dict[str, TrackedFact] = {}

    def add_fact(
        self,
        fact: str,
        step_index: int,
        status: FactStatus = FactStatus.ASSUMED,
        constraint: ConstraintResult | None = None,
    ) -> str:
        """Add a fact to tracking. Returns the normalized key.

        If the fact already exists (by normalized key), the original entry
        is kept and the same key is returned (idempotent).

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        key = _normalize_fact_key(fact)
        if key not in self.facts:
            self.facts[key] = TrackedFact(
                fact=fact,
                status=status,
                introduced_at_step=step_index,
                constraint=constraint,
            )
        return key

    def verify_fact(self, key: str, step_index: int, energy: float) -> None:
        """Mark a fact as VERIFIED.

        Raises KeyError if the key doesn't exist.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        if key not in self.facts:
            raise KeyError(key)
        self.facts[key].status = FactStatus.VERIFIED
        self.facts[key].verified_at_step = step_index
        self.facts[key].energy = energy

    def violate_fact(self, key: str, step_index: int, energy: float) -> None:
        """Mark a fact as VIOLATED.

        Raises KeyError if the key doesn't exist.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        if key not in self.facts:
            raise KeyError(key)
        self.facts[key].status = FactStatus.VIOLATED
        self.facts[key].verified_at_step = step_index
        self.facts[key].energy = energy

    def get_assumed(self) -> list[TrackedFact]:
        """Return all facts with ASSUMED status.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return [f for f in self.facts.values() if f.status == FactStatus.ASSUMED]

    def get_verified(self) -> list[TrackedFact]:
        """Return all facts with VERIFIED status.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return [f for f in self.facts.values() if f.status == FactStatus.VERIFIED]

    def get_violated(self) -> list[TrackedFact]:
        """Return all facts with VIOLATED status.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return [f for f in self.facts.values() if f.status == FactStatus.VIOLATED]

    def get_all_constraints(self) -> list[ConstraintResult]:
        """Return ConstraintResult objects from all facts that have them.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        return [
            f.constraint for f in self.facts.values() if f.constraint is not None
        ]

    def snapshot(self) -> dict:
        """Produce a JSON-serializable snapshot of current state.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        facts_snap = {}
        for key, fact in self.facts.items():
            facts_snap[key] = {
                "fact": fact.fact,
                "status": fact.status.value,
                "introduced_at_step": fact.introduced_at_step,
                "verified_at_step": fact.verified_at_step,
                "energy": fact.energy,
                "has_constraint": fact.constraint is not None,
            }
        return {
            "facts": facts_snap,
            "n_verified": len(self.get_verified()),
            "n_assumed": len(self.get_assumed()),
            "n_violated": len(self.get_violated()),
        }


@dataclass
class AgentStep:
    """One step in a multi-turn agentic reasoning chain.

    **Detailed explanation for engineers:**
        Each step captures the input question and LLM output for that turn.
        After propagate() runs, the step is annotated with the verification
        result and before/after state snapshots for auditability.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    step_index: int
    input_text: str
    output_text: str
    verification: VerificationResult | None = None
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)


def propagate(
    first: ConstraintState | AgentStep,
    second: AgentStep | ConstraintState,
    pipeline: VerifyRepairPipeline,
) -> ConstraintState:
    """Run constraint extraction and verification on a step, updating state.

    **Detailed explanation for engineers:**
        Accepts arguments in either order (state, step) or (step, state)
        for caller convenience. Extracts constraints from the step's output,
        verifies them via the pipeline, and updates the constraint state:
        - Satisfied constraints -> facts marked VERIFIED
        - Violated constraints -> facts marked VIOLATED
        - If any violation occurs, ASSUMED facts from later steps are
          retroactively marked VIOLATED (cascade invalidation)

        The step is annotated with state_before, state_after, and the
        verification result for audit trails.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    if isinstance(first, ConstraintState):
        state = first
        step = second  # type: ignore[assignment]
    else:
        step = first  # type: ignore[assignment]
        state = second  # type: ignore[assignment]

    step.state_before = state.snapshot()

    constraints = pipeline.extract_constraints(step.output_text)

    for cr in constraints:
        desc = cr.description
        state.add_fact(desc, step_index=step.step_index, constraint=cr)

    vr = pipeline.verify(step.output_text)
    step.verification = vr

    violated_keys = set()
    if vr.violations:
        for v in vr.violations:
            key = _normalize_fact_key(v.description)
            if key in state.facts:
                state.violate_fact(key, step_index=step.step_index, energy=vr.energy)
                violated_keys.add(key)

    satisfied_keys = set()
    for cr in constraints:
        key = _normalize_fact_key(cr.description)
        if key not in violated_keys and key in state.facts:
            if state.facts[key].status == FactStatus.ASSUMED:
                state.verify_fact(key, step_index=step.step_index, energy=0.0)
                satisfied_keys.add(key)

    if violated_keys:
        for key, fact in state.facts.items():
            if (
                key not in violated_keys
                and fact.status == FactStatus.ASSUMED
                and fact.introduced_at_step > step.step_index
            ):
                state.violate_fact(key, step_index=step.step_index, energy=vr.energy)

    step.state_after = state.snapshot()

    return state


__all__ = [
    "AgentStep",
    "ConstraintState",
    "FactStatus",
    "TrackedFact",
    "_normalize_fact_key",
    "propagate",
]
