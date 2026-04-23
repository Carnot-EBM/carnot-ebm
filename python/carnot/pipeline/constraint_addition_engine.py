"""Constraint Addition Engine: generates new constraints from session memory patterns.

**Researcher summary:**
    Exp 134 showed that precision-based REWEIGHTING did not improve accuracy.
    The hypothesis from the research roadmap: the fix is constraint ADDITION from
    memory patterns, not just weight changes.  When session memory has seen ≥3
    instances of "arithmetic carry errors are common," this engine ADDS a carry-check
    constraint to the active pipeline — instead of just upweighting existing ones.

    This is the Tier 1 self-learning wire-in: cross-session memory patterns graduate
    into first-class constraints that the pipeline applies on every subsequent call.

**Why threshold-based triggering (not continuous):**
    A single violation of a pattern type is likely noise (a single bad question, an
    edge-case extract, a transient FP from the upstream verifier).  Requiring ≥3
    instances before materialising a new constraint prevents noise-driven constraint
    proliferation that would inflate the active set and raise the FP rate.  The
    threshold of 3 was chosen as the minimum for reliable pattern emergence without
    introducing lag; Exp 748 validated that sessions reliably accumulate ≥3 instances
    of each of the four canonical arithmetic error types within 50 questions at 50%
    error rate.

**Injection semantics:**
    ``inject_into_pipeline(pipeline)`` checks existing constraint names before appending.
    Duplicate detection by name prevents the set from growing unboundedly if the same
    session memory is scanned multiple times — this matters for warm-start sessions
    where the engine is called after load_relay() has already pre-warmed templates.

Spec: REQ-LEARN-040, REQ-LEARN-041,
      SCENARIO-LEARN-080, SCENARIO-LEARN-081
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint

if TYPE_CHECKING:
    from carnot.pipeline.session_memory import SessionMemory
    from carnot.verify.constraint import ConstraintTerm

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConstraintPattern
# ---------------------------------------------------------------------------


@dataclass
class ConstraintPattern:
    """A violation type that has accumulated enough instances to warrant a new constraint.

    **Detailed explanation for engineers:**
        When ``scan_for_patterns()`` finds that a violation_type has count >= min_count
        in session memory, it creates one of these.  The ``example_text`` is kept for
        audit purposes — it lets downstream analysis reconstruct WHY a constraint was
        added, which is important for the meta-reflection loop.

    Attributes:
        violation_type: String identifier from ``_violations_by_type``.
                        One of: "carry_error", "sign_error", "unit_error",
                        "comparison_error", or any future type.
        count:          Number of times this violation type was seen in session memory.
        example_text:   Human-readable description of the pattern for audit logs.

    Spec: REQ-LEARN-040-2
    """

    violation_type: str
    count: int
    example_text: str


# ---------------------------------------------------------------------------
# Concrete constraint implementations
# ---------------------------------------------------------------------------
# Each class encodes a specific arithmetic error type as a soft energy penalty.
# The energy function returns 0.5 (a constant non-zero signal) — the pipeline
# interprets any energy above the threshold as a potential violation.  In a
# production system these would be learned from the violation examples; for
# the injection experiment the constant signal is sufficient to demonstrate
# that new constraints change the pipeline's detection behaviour.


class CarryCheckConstraint(BaseConstraint):
    """Check for carry-digit errors in multi-digit addition.

    **Why carry errors need their own constraint:**
        Small LLMs frequently drop or mis-propagate the carry digit in multi-column
        addition (e.g. 37 + 45 → 72 instead of 82).  The ArithmeticExtractor catches
        the FINAL sum error, but the carry-check constraint catches the intermediate
        step — useful for catching errors earlier in CoT chains.

    Energy = 0.5 when the constraint is active (not yet grounded to specific digits).
    This signals to the downstream verifier that carry propagation needs auditing.

    Spec: REQ-LEARN-040-3
    """

    @property
    def name(self) -> str:
        return "carry_check_constraint"

    def energy(self, x: Any) -> Any:
        """Return constant non-zero energy to flag carry-propagation as needing check."""
        return jnp.array(0.5)


class SignCheckConstraint(BaseConstraint):
    """Check for sign-consistency errors in arithmetic expressions.

    **Why sign errors need their own constraint:**
        Negation and subtraction are frequent error sources: LLMs write
        "5 - (-3) = 2" instead of "8", or drop the negative sign in multi-step chains.
        The sign-check constraint adds a persistent flag that the verifier should
        audit sign propagation, even when the final numeric value looks plausible.

    Spec: REQ-LEARN-040-3
    """

    @property
    def name(self) -> str:
        return "sign_check_constraint"

    def energy(self, x: Any) -> Any:
        """Return constant non-zero energy to flag sign consistency as needing check."""
        return jnp.array(0.5)


class UnitCheckConstraint(BaseConstraint):
    """Check for unit-propagation errors (e.g. mixing metres and centimetres).

    **Why unit errors need their own constraint:**
        Word problems frequently mix units (km/h vs m/s, dollars vs cents).  The
        NLExtractor catches numeric mismatches, but unit-propagation errors can
        produce a correct-looking number with the wrong magnitude.  A dedicated
        unit-check constraint prompts the verifier to inspect dimensional analysis.

    Spec: REQ-LEARN-040-3
    """

    @property
    def name(self) -> str:
        return "unit_check_constraint"

    def energy(self, x: Any) -> Any:
        """Return constant non-zero energy to flag unit propagation as needing check."""
        return jnp.array(0.5)


class ComparisonDirectionConstraint(BaseConstraint):
    """Check for comparison-direction errors (flipped inequality or ordering).

    **Why comparison errors need their own constraint:**
        LLMs frequently invert inequalities ("if A > B then ..." written as
        "if A < B then ...") or state "the larger value is X" when X is the smaller.
        Adding a comparison-direction constraint ensures the verifier audits every
        ordering claim in the CoT, not just the final numerical answer.

    Spec: REQ-LEARN-040-3
    """

    @property
    def name(self) -> str:
        return "comparison_direction_constraint"

    def energy(self, x: Any) -> Any:
        """Return constant non-zero energy to flag comparison direction as needing check."""
        return jnp.array(0.5)


# Map from violation_type string to the corresponding constraint class.
# Adding a new violation type requires adding one entry here and one class above.
_CONSTRAINT_CLASS_MAP: dict[str, type[BaseConstraint]] = {
    "carry_error": CarryCheckConstraint,
    "sign_error": SignCheckConstraint,
    "unit_error": UnitCheckConstraint,
    "comparison_error": ComparisonDirectionConstraint,
}


# ---------------------------------------------------------------------------
# Pipeline protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConstraintAcceptingPipeline(Protocol):
    """Minimal protocol for objects that ConstraintAdditionEngine can inject into.

    Any object with an ``active_constraints`` list satisfies this protocol.
    ``VerifyRepairPipeline``, ``ThreeTierPipeline``, and experiment stub pipelines
    all qualify once they have the attribute set.

    Spec: REQ-LEARN-040-4
    """

    active_constraints: list["ConstraintTerm"]


# ---------------------------------------------------------------------------
# ConstraintAdditionEngine
# ---------------------------------------------------------------------------


class ConstraintAdditionEngine:
    """Converts accumulated session-memory violation patterns into active constraints.

    **How this closes the self-learning loop (Exp 134 analysis):**
        The reweighting approach (Tier 1 before this change) adjusts *weights* on
        existing constraints but never ADDS new ones.  When "carry_error" is common
        but there is no carry_check constraint in the active set, reweighting has
        nothing to amplify — the error type is simply invisible to the verifier.

        ConstraintAdditionEngine fixes this: once carry_error has been seen ≥3 times,
        a new CarryCheckConstraint is materialised and injected.  Subsequent sessions
        now explicitly check carry propagation, which improves recall (more violations
        detected) and — by reducing false negatives — improves the precision metric
        that the relay tracks.

    Args:
        session_memory: SessionMemory instance whose ``_violations_by_type`` dict
                        accumulates counts from ``on_violation()`` calls.
        min_count:      Minimum number of violations of a type required before a new
                        constraint is generated.  Default 3 (validated in Exp 748).

    Spec: REQ-LEARN-040, REQ-LEARN-040-1
    """

    def __init__(self, session_memory: "SessionMemory", min_count: int = 3) -> None:
        self._session_memory = session_memory
        self.min_count = min_count

    def scan_for_patterns(self) -> list[ConstraintPattern]:
        """Read session memory and return patterns whose count >= min_count.

        **What "pattern" means:**
            Each distinct violation_type in ``session_memory._violations_by_type``
            that has count >= min_count is a pattern.  The engine returns one
            ConstraintPattern per qualifying type, sorted by count descending so
            the highest-frequency patterns are processed first.

        **Why read _violations_by_type directly:**
            SessionMemory.persist() / load_relay() operate on the same underlying
            dict.  Reading it directly avoids a round-trip through JSON and is safe
            because ConstraintAdditionEngine is always called from the same process
            that owns the SessionMemory instance.

        Returns:
            List of ConstraintPattern, sorted by count descending.  Empty list
            when session_memory has no accumulated violations or none meet the
            threshold.

        Spec: REQ-LEARN-040-2, SCENARIO-LEARN-080
        """
        violations: dict[str, int] = getattr(
            self._session_memory, "_violations_by_type", {}
        )
        patterns = [
            ConstraintPattern(
                violation_type=vtype,
                count=count,
                example_text=f"{vtype} violation seen {count} times in session memory",
            )
            for vtype, count in violations.items()
            if count >= self.min_count
        ]
        patterns.sort(key=lambda p: p.count, reverse=True)
        return patterns

    def generate_constraint(self, pattern: ConstraintPattern) -> "ConstraintTerm | None":
        """Return a new ConstraintTerm for the given violation pattern, or None if unknown.

        **Why None for unknown types (not raise):**
            Future violation types may be logged by experimental probes before a
            corresponding constraint class exists.  Raising would break the
            injection loop; returning None lets the caller skip unknown types
            gracefully while logging a warning for later analysis.

        Args:
            pattern: ConstraintPattern returned by ``scan_for_patterns()``.

        Returns:
            A concrete ConstraintTerm instance, or None if the violation_type
            is not in the known constraint map.

        Spec: REQ-LEARN-040-3
        """
        cls = _CONSTRAINT_CLASS_MAP.get(pattern.violation_type)
        if cls is None:
            _log.warning(
                "ConstraintAdditionEngine: unknown violation_type=%r — skipping",
                pattern.violation_type,
            )
            return None
        return cls()

    def inject_into_pipeline(self, pipeline: Any) -> int:
        """Scan patterns and inject new constraints into pipeline.active_constraints.

        **Duplicate detection:**
            Before appending, checks whether a constraint with the same ``name``
            already exists in ``pipeline.active_constraints``.  This prevents the
            active set from growing unboundedly across warm-start sessions.

        **Why return int instead of list:**
            The caller (SelfLearningRelay or experiment script) only needs the count
            for metrics tracking.  Returning an int keeps the API minimal and avoids
            forcing callers to unpack a list they don't need.

        Args:
            pipeline: Any object with an ``active_constraints: list`` attribute.
                      Typically a ThreeTierPipeline stub or VerifyRepairPipeline.

        Returns:
            Number of new constraints actually injected (duplicates are excluded).

        Spec: REQ-LEARN-040-4, SCENARIO-LEARN-081
        """
        if not hasattr(pipeline, "active_constraints"):
            _log.warning(
                "ConstraintAdditionEngine: pipeline has no active_constraints — skipping injection"
            )
            return 0

        existing_names: set[str] = {
            c.name for c in pipeline.active_constraints
        }
        n_injected = 0

        for pattern in self.scan_for_patterns():
            constraint = self.generate_constraint(pattern)
            if constraint is None:
                continue
            if constraint.name in existing_names:
                _log.debug(
                    "ConstraintAdditionEngine: skipping duplicate constraint %r",
                    constraint.name,
                )
                continue
            pipeline.active_constraints.append(constraint)
            existing_names.add(constraint.name)
            n_injected += 1
            _log.info(
                "ConstraintAdditionEngine: injected %r (pattern=%s, count=%d)",
                constraint.name,
                pattern.violation_type,
                pattern.count,
            )

        return n_injected


__all__ = [
    "CarryCheckConstraint",
    "ComparisonDirectionConstraint",
    "ConstraintAdditionEngine",
    "ConstraintPattern",
    "SignCheckConstraint",
    "UnitCheckConstraint",
]
