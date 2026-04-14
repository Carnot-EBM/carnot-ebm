"""Constraint generation from CaseMemory patterns with soundness bounds.

**Researcher summary:**
    Turns high-precision failure patterns accumulated in CaseMemory into new
    named constraint types.  Unlike Exp 134 weight reweighting (0% improvement)
    and generation.py (Tier 2 ConstraintMemory-driven), this module reads from
    Tier 3 CaseMemory (live-trace case-based memory) and applies a soundness
    bound from arXiv 2603.03538 (CoT Verifier Online Learnability) to gate
    which patterns are safe to promote to constraints.

    The 85% precision threshold means: if fewer than 85% of flagged cases were
    genuine errors, we do NOT add the constraint — it would introduce too many
    false positives and degrade soundness.

**Key insight (from the research context):**
    Exp 134 showed that reweighting existing constraints gives 0% improvement
    because you cannot fix MISSING constraint types by adjusting weights.
    This module fixes that gap at the CaseMemory (Tier 3) level: when memory
    sees 15/30 repair failures on "carry propagation in multi-digit addition,"
    it generates a new carry-check constraint and adds it to the extractor set.
    Future verify-repair calls will explicitly check carry propagation.

**Soundness guarantee (arXiv 2603.03538):**
    A constraint is safe to add when its observed precision > 0.85 on the
    historical trace.  High precision means: when this constraint fires, it
    is almost always catching a real error.  Low-precision patterns are
    explicitly logged as "rejected_soundness" so the caller knows what was
    filtered, not silently discarded.

**Architecture:**
    - ``ConstraintPattern``: dataclass capturing a violation family, its
      precision, support count, example violations, and source memory keys.
    - ``extract_patterns(case_memory, min_support=3)``: groups CaseMemory entries
      by violation_family, computes precision, returns patterns meeting support.
    - ``soundness_filter(patterns, min_precision=0.85)``: keeps high-precision
      patterns only.
    - ``LearnedConstraint``: the generated constraint object with a stable id,
      family tag, description, and back-reference to the source pattern.
    - ``generate_arithmetic_constraint(pattern)``: maps a family name to a
      targeted LearnedConstraint.
    - ``constraint_already_exists(extractor, constraint_id)``: deduplication
      guard — checks extractor._dynamic_constraints (duck-typed, no subclassing).
    - ``add_to_extractor(extractor, constraint)``: purely additive insertion into
      extractor._dynamic_constraints; never removes existing constraints.
    - ``ConstraintGenerator``: orchestrates extract → filter → generate → add,
      emitting a ``generation_log`` dict mapping each pattern key to its outcome
      ("added", "rejected_soundness", "already_exists").

Spec: REQ-LEARN-010, REQ-LEARN-011,
SCENARIO-LEARN-015, SCENARIO-LEARN-016, SCENARIO-LEARN-017, SCENARIO-LEARN-018.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.pipeline.case_memory import CaseMemory


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ConstraintPattern:
    """A pattern derived from CaseMemory ready for soundness filtering.

    **Researcher summary:**
        Summarises one violation family as observed across CaseMemory entries.
        ``observed_precision`` is the fraction of flagged cases that were
        genuine errors (repair_outcome == "improved").

    **Detailed explanation for engineers:**
        Created by ``extract_patterns()`` for each ``violation_family`` found
        in CaseMemory.  The ``source_memory_keys`` list lets you trace back to
        the exact CaseEntry fingerprints that contributed to this pattern, which
        is useful for audit trails and debugging.

    Attributes:
        pattern_type: Derived constraint type name (e.g. "carry_check",
            "sign_consistency", "magnitude_check").
        violation_family: The violation family string from CaseMemory
            (e.g. "carry_error", "sign_error", "magnitude_error").
        observed_precision: Fraction of total_support that was "improved":
            improved_support / total_support.  Range [0.0, 1.0].
        support_count: Total number of case records (across all entries)
            that share this violation_family.
        example_violations: Sampled violation_type strings from the entries,
            for human-readable debugging.
        constraint_template: Human-readable description of the intended
            constraint, set by ``extract_patterns`` based on the family.
        source_memory_keys: CaseKey.fingerprint strings for each CaseEntry
            that contributed to this pattern.

    Spec: REQ-LEARN-010
    """

    pattern_type: str
    violation_family: str
    observed_precision: float
    support_count: int
    example_violations: list[str]
    constraint_template: str
    source_memory_keys: list[str]


@dataclass
class LearnedConstraint:
    """A generated constraint derived from a high-precision CaseMemory pattern.

    **Researcher summary:**
        Named constraint with a stable id, family, description, and back-reference
        to the source pattern.  Stored in extractor._dynamic_constraints for
        downstream deduplication and audit.

    **Detailed explanation for engineers:**
        The ``constraint_id`` is the stable deduplication key — it is always
        ``"learned:{violation_family}"``.  If the same family appears again in
        a later call to ``generate_from_memory()``, ``constraint_already_exists``
        will find this id and return True, preventing duplicate insertion.

        This dataclass does NOT subclass BaseConstraint / ConstraintTerm because
        the constraints it encodes are text-pattern-level checks (upstream of
        the JAX energy layer), not differentiable energy functions.  If future
        work requires energy-based learned constraints, subclass BaseConstraint
        and add an ``energy()`` method.

    Attributes:
        constraint_id: Stable unique id: ``"learned:{violation_family}"``.
        family: The violation family that produced this constraint.
        description: Human-readable description of what this constraint checks.
        pattern: Back-reference to the ConstraintPattern that generated this.

    Spec: REQ-LEARN-010
    """

    constraint_id: str
    family: str
    description: str
    pattern: ConstraintPattern


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

# Map known violation families to a concise constraint_template string.
_FAMILY_TEMPLATES: dict[str, tuple[str, str]] = {
    # family → (pattern_type, template description)
    "carry_error": (
        "carry_check",
        "verify carry propagation: for a+b, check carries cascade correctly through each digit",
    ),
    "sign_error": (
        "sign_consistency",
        "verify sign consistency: positive/negative signs preserved through all arithmetic operations",
    ),
    "magnitude_error": (
        "magnitude_check",
        "verify order of magnitude: result magnitude within expected range of operands",
    ),
}

_FALLBACK_PATTERN_TYPE = "learned_check"
_FALLBACK_TEMPLATE = "verify constraint derived from memory pattern: {family}"


def extract_patterns(
    case_memory: CaseMemory,
    min_support: int = 3,
) -> list[ConstraintPattern]:
    """Extract ConstraintPatterns from CaseMemory by grouping on violation_family.

    **Researcher summary:**
        Groups all CaseEntry objects by violation_family, computes
        observed_precision = improved_count / total_count for each family,
        and returns ConstraintPattern objects for families with total_support
        >= min_support.

    **Detailed explanation for engineers:**
        Each CaseEntry in CaseMemory has a ``key.violation_families`` tuple.
        One entry can contribute to multiple families (rare in practice, but
        handled correctly here).

        ``observed_precision`` is computed as:
            improved_support / total_support

        where:
        - ``total_support`` = sum of entry.support for all entries with this family
        - ``improved_support`` = sum of entry.support for entries where
          key.repair_outcome == "improved" (meaning: the repair worked, so the
          flagged case was a real error)

        Families below ``min_support`` are silently excluded from the result.
        This is NOT a soundness rejection — it is merely a minimum evidence
        threshold (you need at least 3 cases to draw any conclusion).  The
        soundness bound (0.85 precision) is applied by ``soundness_filter``.

    Args:
        case_memory: The CaseMemory instance to read.
        min_support: Minimum total support required for a family to be included.
            Default 3 (matches Tier 2 frequency gate from generation.py).

    Returns:
        List of ConstraintPattern, one per qualifying violation_family, in
        deterministic order (sorted by family name).

    Spec: REQ-LEARN-010, SCENARIO-LEARN-015
    """
    # Accumulate per-family tallies: {family: (total_support, improved_support, entries)}
    family_total: dict[str, int] = {}
    family_improved: dict[str, int] = {}
    family_violation_types: dict[str, list[str]] = {}
    family_source_keys: dict[str, list[str]] = {}

    for entry in case_memory.entries():
        for family in entry.key.violation_families:
            if family not in family_total:
                family_total[family] = 0
                family_improved[family] = 0
                family_violation_types[family] = []
                family_source_keys[family] = []

            family_total[family] += entry.support
            if entry.key.repair_outcome == "improved":
                family_improved[family] += entry.support

            # Collect example violation types for human readability
            for vt in entry.violation_types:
                if vt not in family_violation_types[family]:
                    family_violation_types[family].append(vt)

            # Record source key fingerprint for audit trail
            fp = entry.key.fingerprint
            if fp not in family_source_keys[family]:
                family_source_keys[family].append(fp)

    patterns: list[ConstraintPattern] = []
    for family in sorted(family_total):
        total = family_total[family]
        if total < min_support:
            continue

        improved = family_improved[family]
        precision = improved / total if total > 0 else 0.0

        if family in _FAMILY_TEMPLATES:
            pattern_type, template = _FAMILY_TEMPLATES[family]
        else:
            pattern_type = _FALLBACK_PATTERN_TYPE
            template = _FALLBACK_TEMPLATE.format(family=family)

        patterns.append(
            ConstraintPattern(
                pattern_type=pattern_type,
                violation_family=family,
                observed_precision=precision,
                support_count=total,
                example_violations=list(family_violation_types[family]),
                constraint_template=template,
                source_memory_keys=list(family_source_keys[family]),
            )
        )

    return patterns


# ---------------------------------------------------------------------------
# Soundness filter (arXiv 2603.03538)
# ---------------------------------------------------------------------------


def soundness_filter(
    patterns: list[ConstraintPattern],
    min_precision: float = 0.85,
) -> list[ConstraintPattern]:
    """Return only patterns whose observed_precision meets the soundness bound.

    **Researcher summary:**
        Implements the precision gate from arXiv 2603.03538 (CoT Verifier Online
        Learnability).  A constraint is safe to add only when it rarely fires on
        correct answers — concretely, when observed_precision >= 0.85.

    **Detailed explanation for engineers:**
        Patterns below ``min_precision`` are NOT silently dropped.  The caller
        (``ConstraintGenerator.generate_from_memory``) inspects both the return
        value and the full input list to identify which patterns were rejected,
        then logs them as "rejected_soundness".  This function itself just does
        the filtering; the rejection logging is the generator's responsibility.

    Args:
        patterns: List of ConstraintPattern to filter.
        min_precision: Minimum required observed_precision.  Default 0.85
            per arXiv 2603.03538 soundness bound.

    Returns:
        Subset of ``patterns`` where observed_precision >= min_precision.

    Spec: REQ-LEARN-011, SCENARIO-LEARN-016
    """
    return [p for p in patterns if p.observed_precision >= min_precision]


# ---------------------------------------------------------------------------
# Constraint generation
# ---------------------------------------------------------------------------

# Map family → description suffix for LearnedConstraint.description
_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "carry_error": (
        "carry propagation check: verifies that arithmetic carry bits cascade correctly "
        "for each digit position in multi-digit addition"
    ),
    "sign_error": (
        "sign consistency check: verifies that positive/negative signs are preserved "
        "through all arithmetic operations without spurious sign flips"
    ),
    "magnitude_error": (
        "order-of-magnitude check: verifies that the result magnitude stays within "
        "the expected range given the operand magnitudes"
    ),
}


def generate_arithmetic_constraint(pattern: ConstraintPattern) -> LearnedConstraint:
    """Create a LearnedConstraint from a ConstraintPattern.

    **Researcher summary:**
        Maps violation_family to a targeted constraint type and description.
        Three families are first-class: carry_error, sign_error, magnitude_error.
        Unknown families receive a generic fallback description.

    **Detailed explanation for engineers:**
        The ``constraint_id`` is always ``"learned:{violation_family}"`` — this
        provides a stable, collision-free deduplication key across calls.  The
        same family will always produce the same id, so ``constraint_already_exists``
        can reliably prevent duplicate insertion.

        The returned LearnedConstraint is not yet added to any extractor.
        Call ``add_to_extractor()`` after verifying it does not already exist.

    Args:
        pattern: The ConstraintPattern to convert to a LearnedConstraint.

    Returns:
        A LearnedConstraint with stable constraint_id and human-readable description.

    Spec: REQ-LEARN-010, SCENARIO-LEARN-017
    """
    family = pattern.violation_family
    constraint_id = f"learned:{family}"

    if family in _FAMILY_DESCRIPTIONS:
        description = _FAMILY_DESCRIPTIONS[family]
    else:
        description = (
            f"learned constraint from memory pattern for violation family '{family}': "
            f"precision={pattern.observed_precision:.2f}, support={pattern.support_count}"
        )

    return LearnedConstraint(
        constraint_id=constraint_id,
        family=family,
        description=description,
        pattern=pattern,
    )


# ---------------------------------------------------------------------------
# Extractor integration (duck-typed, no subclassing required)
# ---------------------------------------------------------------------------


def constraint_already_exists(extractor: Any, constraint_id: str) -> bool:
    """Check whether a constraint with this id is already in the extractor.

    **Detailed explanation for engineers:**
        Reads ``extractor._dynamic_constraints`` (a list of objects with a
        ``constraint_id`` attribute) using ``getattr`` with a default of ``[]``.
        This is intentionally duck-typed: any extractor class works as long as
        it has (or will have) a ``_dynamic_constraints`` list — no subclassing
        or Protocol conformance required.

    Args:
        extractor: Any object that may have a ``_dynamic_constraints`` attribute.
        constraint_id: The id to search for.

    Returns:
        True if any item in ``_dynamic_constraints`` has a matching ``constraint_id``.

    Spec: REQ-LEARN-010
    """
    existing: list[Any] = getattr(extractor, "_dynamic_constraints", [])
    return any(c.constraint_id == constraint_id for c in existing)


def add_to_extractor(extractor: Any, constraint: LearnedConstraint) -> None:
    """Append a LearnedConstraint to the extractor's dynamic constraint list.

    **Detailed explanation for engineers:**
        Creates ``extractor._dynamic_constraints`` as an empty list if it does
        not yet exist, then appends ``constraint``.  This is a purely additive
        operation: no existing constraints are removed or modified.

        Callers should call ``constraint_already_exists()`` first to avoid
        duplicates; this function does NOT check for duplicates itself.

    Args:
        extractor: The extractor to update (duck-typed).
        constraint: The LearnedConstraint to add.

    Spec: REQ-LEARN-010
    """
    if not hasattr(extractor, "_dynamic_constraints"):
        extractor._dynamic_constraints = []
    extractor._dynamic_constraints.append(constraint)


# ---------------------------------------------------------------------------
# ConstraintGenerator — orchestrator
# ---------------------------------------------------------------------------


class ConstraintGenerator:
    """Orchestrates extract → soundness_filter → generate → add.

    **Researcher summary:**
        Reads CaseMemory, identifies high-precision violation families,
        generates targeted constraint types, and adds them to the extractor.
        Every pattern's disposition is recorded in ``generation_log``.

    **Detailed explanation for engineers:**
        ``generation_log`` is a dict mapping a pattern key string to one of
        three outcomes:
        - ``"added"``: constraint was generated and added to the extractor.
        - ``"rejected_soundness"``: pattern precision < min_precision (0.85
          by default) — constraint was NOT added to preserve soundness.
        - ``"already_exists"``: a constraint with this id already exists in
          the extractor — no duplicate was inserted.

        The log key format is ``"{pattern_type}:{violation_family}"``.
        It is reset at the start of each ``generate_from_memory()`` call, so
        the log always reflects the most recent run.

    Attributes:
        generation_log: Dict of {pattern_key: outcome_string} from last call.

    Spec: REQ-LEARN-010, REQ-LEARN-011, SCENARIO-LEARN-018
    """

    def __init__(self) -> None:
        self.generation_log: dict[str, str] = {}

    def generate_from_memory(
        self,
        case_memory: CaseMemory,
        extractor: Any,
        *,
        min_support: int = 3,
        min_precision: float = 0.85,
    ) -> list[LearnedConstraint]:
        """Run the full extract → filter → generate → add pipeline.

        **Detailed explanation for engineers:**
            1. ``extract_patterns`` groups CaseMemory by violation_family and
               computes observed_precision per family.
            2. Patterns below ``min_support`` were already filtered by step 1.
            3. ``soundness_filter`` further filters by ``min_precision``; rejected
               patterns are logged as "rejected_soundness".
            4. For each sound pattern, ``generate_arithmetic_constraint`` produces
               a LearnedConstraint.
            5. If a constraint with the same id already exists in the extractor,
               it is logged as "already_exists" and skipped.
            6. Otherwise the constraint is added and logged as "added".

        Args:
            case_memory: The CaseMemory to read patterns from.
            extractor: The extractor to add new constraints to (duck-typed).
            min_support: Minimum total support for a pattern to qualify.
            min_precision: Soundness bound (default 0.85 per arXiv 2603.03538).

        Returns:
            List of LearnedConstraint objects that were successfully added.

        Spec: REQ-LEARN-010, REQ-LEARN-011, SCENARIO-LEARN-018
        """
        # Reset log for this run
        self.generation_log = {}

        all_patterns = extract_patterns(case_memory, min_support=min_support)
        if not all_patterns:
            return []

        sound_patterns = soundness_filter(all_patterns, min_precision=min_precision)
        sound_families = {p.violation_family for p in sound_patterns}

        # Log rejected-for-soundness patterns explicitly (REQ-LEARN-011)
        for pattern in all_patterns:
            if pattern.violation_family not in sound_families:
                log_key = f"{pattern.pattern_type}:{pattern.violation_family}"
                self.generation_log[log_key] = "rejected_soundness"

        added: list[LearnedConstraint] = []
        for pattern in sound_patterns:
            constraint = generate_arithmetic_constraint(pattern)
            log_key = f"{pattern.pattern_type}:{pattern.violation_family}"

            if constraint_already_exists(extractor, constraint.constraint_id):
                self.generation_log[log_key] = "already_exists"
                continue

            add_to_extractor(extractor, constraint)
            self.generation_log[log_key] = "added"
            added.append(constraint)

        return added


__all__ = [
    "ConstraintGenerator",
    "ConstraintPattern",
    "LearnedConstraint",
    "add_to_extractor",
    "constraint_already_exists",
    "extract_patterns",
    "generate_arithmetic_constraint",
    "soundness_filter",
]
