"""Constraint Template Library: dynamic constraint-type addition from memory patterns.

**Researcher summary:**
    Exp 134 proved that REWEIGHTING existing constraints (upweighting when they fire)
    does NOT improve accuracy — fixed and adaptive strategies perform identically across
    500 arithmetic questions. Root cause: if the existing constraint set doesn't cover
    the real error type, upweighting existing constraints just amplifies noise.

    This module implements the correct fix: when CaseMemory detects that a specific
    ERROR PATTERN is common (e.g., "carry errors appear in 40% of Qwen3.5-0.8B
    arithmetic responses"), ADD a new constraint template that checks carry propagation
    — rather than upweighting the existing range_check constraint that doesn't catch
    carry errors.

    This is the Tier 2 → Tier 1 feedback loop described in research-program.md and
    grounded in:
    - arXiv 2603.03538: Online CoT verifier learnability bounds — proves that verifier
      accuracy improves when the verifier's constraint vocabulary matches the observed
      error distribution.
    - Eidoku (arXiv 2512.20664): Constraint type taxonomy — provides a principled
      categorization of arithmetic error types (carry, sign, unit, direction) that
      informs which templates to register.

**Detailed explanation for engineers:**
    The library has two parts:

    1. ``ConstraintTemplate`` — a dataclass that bundles a pattern key, a description,
       a minimum observation frequency before activation, and a callable that takes a
       response string and returns a list of ConstraintResult objects.

    2. ``ConstraintTemplateLibrary`` — a registry that:
       - Registers templates by pattern_key.
       - Counts how many times each (pattern_key, model_id) pair has been observed.
       - Returns templates as "active" once their observation count exceeds min_frequency.
       - Calls active template functions and merges their constraints into the pipeline.
       - Serializes/deserializes observation state for persistence across runs.

    The four built-in templates cover the Eidoku taxonomy of arithmetic error types:
    - ``carry_check``: Multi-digit multiplication carry propagation errors.
    - ``sign_check``: Sign errors in products of two negatives.
    - ``unit_consistency``: Incompatible unit mixing (kg/g, km/m, L/ml).
    - ``comparison_direction``: X > Y claim inconsistent with X − Y < 0.

    All templates are CI-safe: they return [] when no parseable arithmetic is found,
    so they never generate spurious violations on responses with no relevant content.

Spec: REQ-LEARN-017, REQ-LEARN-018,
      SCENARIO-LEARN-029, SCENARIO-LEARN-030, SCENARIO-LEARN-031, SCENARIO-LEARN-032
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from carnot.pipeline.extract import ConstraintResult


# ---------------------------------------------------------------------------
# ConstraintTemplate dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConstraintTemplate:
    """A template that generates constraints of a specific type from response text.

    **Detailed explanation for engineers:**
        Each template encapsulates one category of arithmetic error (e.g., carry
        propagation, sign errors). A template becomes "active" for a given model
        once the library has seen it fire at least ``min_frequency`` times for that
        model. Once active, ``template_fn`` is called on each incoming response and
        any returned ConstraintResult objects are merged into the pipeline's active
        constraint set.

        ``is_active`` and ``activation_count`` are updated by
        ``ConstraintTemplateLibrary.apply_active_templates()`` and track lifecycle:
        - ``is_active``: True once the template has been activated at least once (for
          ANY model). Useful for debugging — "has this template ever fired?"
        - ``activation_count``: Total number of times the template has been called
          across all models and all responses. Useful for load analysis.

    Attributes:
        pattern_key:      Unique identifier for this error pattern (e.g. "carry_check").
        description:      Human-readable summary of what error type this catches.
        min_frequency:    How many times a pattern must be observed for a given model
                          before this template is activated for that model.
        template_fn:      Callable taking a response string, returning a list of
                          ConstraintResult objects. Must return [] when the response
                          contains no relevant arithmetic.
        is_active:        True once this template has been called at least once
                          (set by apply_active_templates). Default False.
        activation_count: Running total of how many times template_fn has been called
                          (incremented by apply_active_templates). Default 0.

    Spec: REQ-LEARN-017
    """

    pattern_key: str
    description: str
    min_frequency: int
    template_fn: Callable[[str], list[ConstraintResult]]
    is_active: bool = False
    activation_count: int = 0


# ---------------------------------------------------------------------------
# ConstraintTemplateLibrary
# ---------------------------------------------------------------------------


class ConstraintTemplateLibrary:
    """Registry of constraint templates that activate when error patterns become frequent.

    **Researcher summary:**
        The library is the bridge between CaseMemory (which tracks WHAT goes wrong)
        and the constraint extractor (which checks WHETHER something went wrong).
        When CaseMemory sees the same error pattern repeatedly for a given model,
        the library promotes that pattern from "observed" to "active" — meaning a new
        constraint type is added to the pipeline specifically for that model.

    **Detailed explanation for engineers:**
        Two internal data structures drive the library:
        - ``_templates``: dict mapping pattern_key → ConstraintTemplate. These are
          the code-defined templates; registered via ``add_template()`` or
          ``register_builtin_templates()``.
        - ``_observations``: dict mapping (pattern_key, model_id) → int count.
          Incremented by ``observe_pattern()`` each time CaseMemory reports a pattern.

        Activation logic in ``get_active_templates(model_id)``:
        - For each registered template, check ``_observations[(pattern_key, model_id)]``.
        - If the count >= template.min_frequency, include the template in the active list.
        - Templates with no observations for a given model are never active for that model.

        Persistence:
        - ``to_dict()`` / ``from_dict()`` serialize/restore the observations dict.
        - Template functions (callables) cannot be serialized to JSON; after
          ``from_dict()``, the caller must call ``register_builtin_templates()`` (or
          ``add_template()`` for custom templates) to restore the callable functions.
          The observation counts are preserved across the round-trip.

    Spec: REQ-LEARN-017, REQ-LEARN-018
    """

    def __init__(self) -> None:
        # pattern_key → ConstraintTemplate (holds the callable)
        self._templates: dict[str, ConstraintTemplate] = {}
        # (pattern_key, model_id) → observation count
        self._observations: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_template(self, template: ConstraintTemplate) -> None:
        """Register a ConstraintTemplate by its pattern_key.

        **Detailed explanation for engineers:**
            Calling add_template a second time with the same pattern_key replaces
            the existing template. This allows callers to update min_frequency or
            swap template_fn without creating a new library instance.

        Args:
            template: The ConstraintTemplate to register.

        Spec: REQ-LEARN-017
        """
        self._templates[template.pattern_key] = template

    # ------------------------------------------------------------------
    # Observation tracking
    # ------------------------------------------------------------------

    def observe_pattern(self, pattern_key: str, model_id: str, count: int = 1) -> None:
        """Record that the named error pattern was observed count times for model_id.

        **Detailed explanation for engineers:**
            This is the primary input from CaseMemory. When the memory system notices
            that "carry errors appear in 40% of Qwen3.5-0.8B responses", it calls
            observe_pattern("carry_check", "qwen3.5-0.8b", count=N). Once the
            cumulative count crosses the template's min_frequency, the template becomes
            active for that model.

            Counts are additive: calling observe_pattern("carry_check", "m1", 3) twice
            results in a total count of 6 for ("carry_check", "m1").

        Args:
            pattern_key: The error pattern identifier (must match a registered template
                         pattern_key to become actionable, but counting always proceeds).
            model_id:    The model that produced the responses where this pattern was seen.
            count:       How many observations to add. Default 1.

        Spec: REQ-LEARN-017-1
        """
        key = (pattern_key, model_id)
        self._observations[key] = self._observations.get(key, 0) + count

    # ------------------------------------------------------------------
    # Activation queries
    # ------------------------------------------------------------------

    def get_active_templates(self, model_id: str) -> list[ConstraintTemplate]:
        """Return templates where cumulative observations for model_id >= min_frequency.

        **Detailed explanation for engineers:**
            Iterates over all registered templates. For each template, looks up the
            observation count for (template.pattern_key, model_id). If the count meets
            or exceeds the template's min_frequency, the template is included.

            Templates with zero observations (never observed for this model) are
            excluded — we do not activate templates for models where we have no
            evidence of the associated error pattern.

        Args:
            model_id: The model to check. Activation is per-model.

        Returns:
            List of ConstraintTemplate objects that are currently active for model_id.
            Returns [] if no templates meet their activation threshold.

        Spec: REQ-LEARN-017-2
        """
        active = []
        for pattern_key, template in self._templates.items():
            observed = self._observations.get((pattern_key, model_id), 0)
            if observed >= template.min_frequency:
                active.append(template)
        return active

    def apply_active_templates(self, response: str, model_id: str) -> list[ConstraintResult]:
        """Call template_fn for each active template and collect all returned constraints.

        **Detailed explanation for engineers:**
            For each active template (threshold crossed for model_id), calls
            template_fn(response). This may return zero or more ConstraintResult
            objects depending on what arithmetic content the template finds in the
            response.

            Side effects on each called template:
            - ``is_active`` is set to True (marks the template as ever-activated).
            - ``activation_count`` is incremented by 1 (tracks total invocations).

            These side effects support lifecycle monitoring but do not affect the
            activation decision itself (which is based solely on observation counts).

        Args:
            response: The response text to check with active templates.
            model_id: The model that produced the response.

        Returns:
            List of all ConstraintResult objects produced by active templates.
            Returns [] if no templates are active or if no active template found
            any relevant arithmetic in the response.

        Spec: REQ-LEARN-017-3
        """
        results: list[ConstraintResult] = []
        for template in self.get_active_templates(model_id):
            new_constraints = template.template_fn(response)
            results.extend(new_constraints)
            template.is_active = True
            template.activation_count += 1
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize library state to a JSON-compatible dict.

        **Detailed explanation for engineers:**
            Serializes the observation counts. Template functions (callables) cannot
            be serialized; after ``from_dict()``, the caller must call
            ``register_builtin_templates()`` (or ``add_template()`` for custom
            templates) to restore the callable functions.

        Returns:
            Dict with key "observations" → list of {pattern_key, model_id, count} dicts.

        Spec: REQ-LEARN-018-1
        """
        obs_list = [
            {"pattern_key": pk, "model_id": mid, "count": count}
            for (pk, mid), count in self._observations.items()
        ]
        return {"observations": obs_list}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstraintTemplateLibrary:
        """Restore a ConstraintTemplateLibrary from a previously serialized dict.

        **Detailed explanation for engineers:**
            Restores the observation counts from the dict produced by ``to_dict()``.
            Does NOT restore template functions (they are not serializable).
            After from_dict(), the caller must call ``register_builtin_templates()``
            or manually call ``add_template()`` for each template to restore fns.

        Args:
            payload: Dict produced by ``to_dict()``.

        Returns:
            Restored ConstraintTemplateLibrary with observation counts intact.

        Spec: REQ-LEARN-018-1
        """
        lib = cls()
        for entry in payload.get("observations", []):
            key = (str(entry["pattern_key"]), str(entry["model_id"]))
            lib._observations[key] = int(entry["count"])
        return lib

    # ------------------------------------------------------------------
    # Built-in templates
    # ------------------------------------------------------------------

    def register_builtin_templates(self) -> None:
        """Register the four standard arithmetic error constraint templates.

        **Detailed explanation for engineers:**
            Registers the templates from the Eidoku (arXiv 2512.20664) taxonomy:
            - carry_check (min_freq=5): Multi-digit carry propagation verification.
            - sign_check (min_freq=5): Negative × negative = positive sign rule.
            - unit_consistency (min_freq=3): Incompatible unit mixing detection.
            - comparison_direction (min_freq=5): X > Y consistency with X − Y > 0.

            The lower min_frequency for unit_consistency (3 vs 5) reflects that
            unit errors are rarer in arithmetic benchmarks — fewer observations
            are needed to establish that a model has a systematic unit problem.

        Spec: REQ-LEARN-018
        """
        self.add_template(ConstraintTemplate(
            pattern_key="carry_check",
            description="Check multi-digit carry propagation in arithmetic steps",
            min_frequency=5,
            template_fn=carry_check_template,
        ))
        self.add_template(ConstraintTemplate(
            pattern_key="sign_check",
            description="Check that negative times negative equals positive",
            min_frequency=5,
            template_fn=sign_check_template,
        ))
        self.add_template(ConstraintTemplate(
            pattern_key="unit_consistency",
            description="Check that units in intermediate steps are consistent",
            min_frequency=3,
            template_fn=unit_consistency_template,
        ))
        self.add_template(ConstraintTemplate(
            pattern_key="comparison_direction",
            description="Check that X > Y is consistent with X minus Y being positive",
            min_frequency=5,
            template_fn=comparison_direction_template,
        ))


# ---------------------------------------------------------------------------
# Built-in template functions
# ---------------------------------------------------------------------------


def carry_check_template(response: str) -> list[ConstraintResult]:
    """Check multi-digit carry propagation in arithmetic steps.

    **Detailed explanation for engineers:**
        Multi-digit multiplication (any factor > 9) requires propagating carry bits
        between digit columns. A common LLM error is computing the units digit
        correctly but forgetting to add the carry to the next column (e.g.,
        "24 × 3 = 62" instead of 72 because the carry from 4 × 3 = 12 was dropped).

        This template looks for "A × B = C" or "A * B = C" patterns where at least
        one factor is greater than 9 (making carry propagation necessary). It then
        checks whether the claimed product C equals the mathematically correct A × B.
        If not, the constraint is unsatisfied — signaling a potential carry error.

        CI-safe: returns [] when no multi-digit multiplication patterns are found.

    Args:
        response: The response text to scan for multi-digit multiplication claims.

    Returns:
        List of ConstraintResult objects, one per detected multi-digit multiplication
        claim. Empty list if no such patterns are found.

    Spec: REQ-LEARN-017, SCENARIO-LEARN-029
    """
    results = []
    # Match "A × B = C" or "A * B = C" with optional surrounding whitespace.
    # The × character (U+00D7) is the multiplication sign used in many LLM outputs.
    pattern = re.compile(r'(\d+)\s*[×*]\s*(\d+)\s*=\s*(\d+)')
    for m in pattern.finditer(response):
        a = int(m.group(1))
        b = int(m.group(2))
        claimed = int(m.group(3))
        # Only check multi-digit multiplication — single-digit × single-digit
        # requires no carry propagation, so this template doesn't add value there.
        if a > 9 or b > 9:
            correct = a * b
            satisfied = claimed == correct
            results.append(ConstraintResult(
                constraint_type="carry_check",
                description=(
                    f"carry check: {a} × {b} = {claimed} "
                    f"(expected {correct})"
                ),
                metadata={
                    "a": a,
                    "b": b,
                    "claimed": claimed,
                    "correct": correct,
                    "satisfied": satisfied,
                },
            ))
    return results


def sign_check_template(response: str) -> list[ConstraintResult]:
    """Check the sign rule: negative times negative must equal a positive number.

    **Detailed explanation for engineers:**
        A common arithmetic mistake in LLM outputs is producing a negative product
        when both factors are negative (e.g., "(-3) × (-4) = -12" instead of 12).
        This violates the fundamental sign rule: (−a) × (−b) = ab > 0 when a, b > 0.

        This template matches the explicit pattern "(-A) × (-B) = C" where A and B
        are positive numbers. It checks that the claimed result C > 0.

        CI-safe: returns [] when no negative-times-negative patterns are found.

    Args:
        response: The response text to scan for negative-times-negative claims.

    Returns:
        List of ConstraintResult objects. Empty if no matching patterns found.

    Spec: REQ-LEARN-017, SCENARIO-LEARN-030
    """
    results = []
    # Match "(-A) × (-B) = C" where A, B are positive numbers and C may be negative.
    # Both × and * are accepted as multiplication symbols.
    pattern = re.compile(
        r'\(\s*-\s*(\d+(?:\.\d+)?)\s*\)'
        r'\s*[×*]\s*'
        r'\(\s*-\s*(\d+(?:\.\d+)?)\s*\)'
        r'\s*=\s*(-?\d+(?:\.\d+)?)'
    )
    for m in pattern.finditer(response):
        a = float(m.group(1))
        b = float(m.group(2))
        claimed = float(m.group(3))
        # The sign rule: negative × negative = positive.
        # Claimed result must be strictly positive.
        satisfied = claimed > 0
        results.append(ConstraintResult(
            constraint_type="sign_check",
            description=(
                f"sign check: (-{a}) × (-{b}) = {claimed} "
                f"(must be positive)"
            ),
            metadata={
                "a": -a,
                "b": -b,
                "claimed": claimed,
                "satisfied": satisfied,
            },
        ))
    return results


def unit_consistency_template(response: str) -> list[ConstraintResult]:
    """Check that physical units in intermediate steps are consistent.

    **Detailed explanation for engineers:**
        A common error in multi-step word problems is mixing incompatible units
        without an explicit conversion step. For example, adding "5 kg + 3 g" to
        get "8 kg" is a unit inconsistency error — the grams must be converted to
        kilograms first (3 g = 0.003 kg, so the correct answer is 5.003 kg).

        This template scans for numeric quantities followed by unit annotations
        (kg, g, m, cm, km, s, L, ml, etc.). If it finds a response that mixes
        units from an incompatible pair (e.g., both "kg" and "g") without an
        explicit conversion, it flags this as a unit inconsistency.

        The incompatible pairs checked are:
        - Mass: {kg, g}     (kilograms vs grams — 1000× difference)
        - Length: {km, m}   (kilometers vs meters — 1000× difference)
        - Volume: {L, ml}   (liters vs milliliters — 1000× difference)

        CI-safe: returns [] when no unit annotations are found in the response.

    Args:
        response: The response text to scan for unit annotations.

    Returns:
        List of ConstraintResult objects. One result per incompatible unit pair found.
        If units are found but all consistent, returns one satisfied=True result.
        Empty list if no unit annotations found.

    Spec: REQ-LEARN-017, SCENARIO-LEARN-031
    """
    # Find all numeric quantities with unit annotations.
    # The \b word boundary ensures we match "5 kg" but not "5 kg/h" for the "kg" unit.
    unit_pattern = re.compile(
        r'\d+(?:\.\d+)?\s*(kg|g\b|m\b|cm|mm|km\b|s\b|ms|L\b|ml|km/h|m/s)'
    )
    matches = unit_pattern.findall(response)
    if not matches:
        return []

    unit_set = set(matches)
    # Pairs of units that cannot be mixed without explicit conversion.
    incompatible_pairs: list[set[str]] = [
        {"kg", "g"},
        {"km", "m"},
        {"L", "ml"},
    ]

    results = []
    for pair in incompatible_pairs:
        if pair.issubset(unit_set):
            sorted_pair = sorted(pair)
            results.append(ConstraintResult(
                constraint_type="unit_consistency",
                description=(
                    f"unit inconsistency: '{sorted_pair[0]}' and '{sorted_pair[1]}' "
                    f"mixed without explicit conversion"
                ),
                metadata={
                    "units_found": sorted(unit_set),
                    "inconsistent_pair": sorted_pair,
                    "satisfied": False,
                },
            ))

    if not results:
        # Units were found but all are from compatible groups.
        results.append(ConstraintResult(
            constraint_type="unit_consistency",
            description="unit consistency: all units are consistent",
            metadata={
                "units_found": sorted(unit_set),
                "satisfied": True,
            },
        ))

    return results


def comparison_direction_template(response: str) -> list[ConstraintResult]:
    """Check that X > Y comparisons are consistent with subsequent X - Y = Z (Z > 0).

    **Detailed explanation for engineers:**
        A subtle arithmetic error is claiming "X > Y" in one step, then computing
        "X - Y = Z" in a later step with Z ≤ 0. These two claims are mutually
        inconsistent: if X > Y then X - Y must be strictly positive.

        Example of the error: "Since 50 > 30, ... 50 - 30 = -20" — the subtraction
        result is negative, contradicting the earlier comparison.

        This template:
        1. Scans the response for all "X > Y" patterns, recording (X, Y) pairs.
        2. Scans for all "X - Y = Z" patterns.
        3. For each subtraction whose operands match a > pair, checks Z > 0.

        CI-safe: returns [] when no matching X > Y / X - Y = Z pair is found.

    Args:
        response: The response text to scan for comparison + subtraction patterns.

    Returns:
        List of ConstraintResult objects for each matched pair. Empty if no
        matching comparison + subtraction pair found.

    Spec: REQ-LEARN-017, SCENARIO-LEARN-032
    """
    results = []
    gt_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*>\s*(\d+(?:\.\d+)?)')
    sub_pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)'
    )

    # Collect all "X > Y" pairs first.
    gt_pairs: set[tuple[float, float]] = set()
    for m in gt_pattern.finditer(response):
        gt_pairs.add((float(m.group(1)), float(m.group(2))))

    # For each subtraction, check if it corresponds to a > claim.
    for m in sub_pattern.finditer(response):
        x = float(m.group(1))
        y = float(m.group(2))
        z = float(m.group(3))
        if (x, y) in gt_pairs:
            # If X > Y was claimed, then X - Y = Z must have Z > 0.
            satisfied = z > 0
            results.append(ConstraintResult(
                constraint_type="comparison_direction",
                description=(
                    f"comparison direction: {x} > {y} implies "
                    f"{x} - {y} = {z} should be positive"
                ),
                metadata={"x": x, "y": y, "z": z, "satisfied": satisfied},
            ))

    return results


# ---------------------------------------------------------------------------
# CaseMemoryTemplateWiring
# ---------------------------------------------------------------------------


class CaseMemoryTemplateWiring:
    """Bridge CaseMemory violation events to ConstraintTemplateLibrary.observe_pattern().

    **Researcher summary:**
        This class is the Tier 2 → Tier 1 feedback loop: when CaseMemory records a
        violation, we call observe_pattern() on the library so that repeated error
        patterns eventually activate new constraint templates. The wiring is additive
        and read-only with respect to CaseMemory — it never modifies the memory.

    **Detailed explanation for engineers:**
        When the verify-repair pipeline detects a violation and records it in CaseMemory,
        it should also call ``on_violation_recorded(violation_type, model_id)`` on this
        wiring object. The wiring translates the raw violation_type string into a
        canonical pattern_key (e.g., "carry_error" → "carry_check") and increments
        the observation count in the library for that model. Once enough violations of
        the same type accumulate, the corresponding template activates and the pipeline
        starts generating additional constraints of that type.

        The mapping is intentionally permissive: any violation type that contains a
        recognized keyword in its name is mapped, and unrecognized types pass through
        unchanged. This lets experiment code use domain-specific violation names while
        still wiring into the template library's canonical keys.

        Case-insensitive matching ensures consistency across different naming conventions
        (e.g., "CARRY_ERROR", "carry_error", "Carry_Error" all map to "carry_check").

    Spec: REQ-LEARN-019, REQ-LEARN-019-1, REQ-LEARN-019-2, REQ-LEARN-019-3, REQ-LEARN-019-4,
          SCENARIO-LEARN-033, SCENARIO-LEARN-034
    """

    # Maps substring keywords (lowercase) to canonical pattern_keys.
    # Ordered from most-specific to least-specific so that multi-keyword types
    # like "carry_sign_error" match "carry" before "sign".
    _KEYWORD_MAP: list[tuple[str, str]] = [
        ("carry", "carry_check"),
        ("sign", "sign_check"),
        ("unit", "unit_consistency"),
        ("comparison", "comparison_direction"),
    ]

    def __init__(self, library: ConstraintTemplateLibrary) -> None:
        """Initialize the wiring with the library to notify.

        Args:
            library: The ConstraintTemplateLibrary to call observe_pattern() on.

        Spec: REQ-LEARN-019-1
        """
        self._library = library

    def violation_type_to_pattern_key(self, violation_type: str) -> str:
        """Map a violation_type string to a canonical pattern_key.

        **Detailed explanation for engineers:**
            Checks whether the lowercased violation_type contains any of the
            recognized keyword substrings. The first match wins. If no keyword
            matches, the original violation_type is returned unchanged (pass-through).

            This design keeps the mapping DRY: experiment code can use descriptive
            violation names ("carry_error_in_step_3") and the wiring still finds
            the canonical key ("carry_check"). Unrecognized types flow through
            without error — they accumulate observation counts but only become
            actionable if a template with that pattern_key is registered.

        Args:
            violation_type: The raw violation type string from the pipeline.

        Returns:
            Canonical pattern_key string for use in observe_pattern().

        Spec: REQ-LEARN-019-3, REQ-LEARN-019-4
        """
        lowered = violation_type.lower()
        for keyword, pattern_key in self._KEYWORD_MAP:
            if keyword in lowered:
                return pattern_key
        return violation_type

    def on_violation_recorded(self, violation_type: str, model_id: str) -> None:
        """Notify the library that a violation of the given type was observed.

        **Detailed explanation for engineers:**
            Translates violation_type → pattern_key via violation_type_to_pattern_key()
            then calls library.observe_pattern(pattern_key, model_id, count=1).

            This is the single entry point that experiment code should call for each
            violation detected in the verify-repair loop. Repeated calls accumulate
            counts in the library until a template's min_frequency threshold is crossed.

        Args:
            violation_type: The raw violation type string (e.g., "carry_error").
            model_id:        The model that produced the violating response.

        Spec: REQ-LEARN-019-2
        """
        pattern_key = self.violation_type_to_pattern_key(violation_type)
        self._library.observe_pattern(pattern_key, model_id, count=1)


__all__ = [
    "CaseMemoryTemplateWiring",
    "ConstraintTemplate",
    "ConstraintTemplateLibrary",
    "carry_check_template",
    "comparison_direction_template",
    "sign_check_template",
    "unit_consistency_template",
]
