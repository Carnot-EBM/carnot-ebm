"""Additive constraint generation from mature recurring failure families in case memory.

**Researcher summary:**
    Turns persistent, high-confidence failure patterns that case memory has
    accumulated into lightweight constraint templates.  These templates are
    then loaded into a ``ConstraintAdditionRegistry`` that the
    ``VerifyRepairPipeline`` can query at inference time — entirely on CPU,
    with no matrix operations.

**Key design principle — additive, not replacement:**
    This module produces ``ConstraintTemplate`` objects.  It does NOT emit
    ``ThresholdOverride``, ``RoutingHint``, or any other ``SelfLearningPolicy``
    artifact.  Both compilation paths can run independently and their outputs
    can be used together.

**Template kinds:**

``text_pattern_guard``
    A set of short substring patterns (e.g. ``"answer_target_mismatch"``)
    compiled from the violation-type labels of the recurring failure family.
    The registry's ``apply()`` method checks whether any pattern appears in
    the response text and, if so, returns the template as a guard signal.
    No regex — just ``pattern in response_text.lower()`` comparisons.

``budget_addition``
    Tells the verifier to run ``budget_delta`` additional check passes for
    this failure family.  Always returned by ``apply()`` when the violation
    type matches.  No text scanning needed.

``verifier_guard_clause``
    A named gate that the pipeline can recognise by ``guard_name`` before
    invoking the more expensive verifier path.  Always returned by ``apply()``
    when violation type matches.

**CPU cost:**
    Compilation is O(N) where N = number of case memory entries.
    Registry lookup is O(T) where T = number of compiled templates.
    ``apply()`` adds an O(T * P) text scan where P = max patterns per template.
    All operations are pure Python string comparisons — safe at inference speed.

Spec: REQ-VERIFY-060,
SCENARIO-VERIFY-070, SCENARIO-VERIFY-071, SCENARIO-VERIFY-072,
SCENARIO-VERIFY-073, SCENARIO-VERIFY-074
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from carnot.pipeline.case_memory import CaseEntry, CaseMemory

# Fixed compile date for all artifacts produced by this module.
RUN_DATE: str = "20260413"

# Format version for the serialized ConstraintAdditionResult.
VERSION: int = 1

# Defaults for the compiler qualification thresholds.
_DEFAULT_MIN_SUPPORT: int = 3
_DEFAULT_MIN_CONFIDENCE: float = 0.85

# Maximum number of patterns to embed in a text_pattern_guard template.
# Keeps the guard set small for fast substring scanning.
_MAX_GUARD_PATTERNS: int = 6


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintProvenance:
    """Explains why a ConstraintTemplate was created.

    Every template must carry at least one of these so that offline replay
    and live pipeline diagnostics can explain where the constraint came from.

    Fields
    ------
    source_type
        Always ``"case_memory"`` for templates compiled by this module.
    source_case_ids
        The ``case_id`` strings of the ``CaseProvenance`` records inside the
        originating ``CaseEntry``.  Preserved verbatim from case memory so
        users can cross-reference the original benchmark traces.
    source_experiment
        The experiment number from the first case provenance record, or
        ``None`` if the entry had no experiment tag.
    failure_family
        The primary violation family that qualified this entry (e.g.
        ``"semantic"``, ``"question_grounding_failures"``).
    support
        The ``CaseEntry.support`` count at compile time.
    confidence
        The ``CaseEntry.confidence`` score at compile time (0.0 – 1.0).
    compiled_date
        Fixed string ``"20260413"`` for traceability across serialized runs.
    """

    source_type: str
    source_case_ids: tuple[str, ...]
    source_experiment: int | None
    failure_family: str
    support: int
    confidence: float
    compiled_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_case_ids": list(self.source_case_ids),
            "source_experiment": self.source_experiment,
            "failure_family": self.failure_family,
            "support": self.support,
            "confidence": self.confidence,
            "compiled_date": self.compiled_date,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstraintProvenance:
        source_experiment = payload.get("source_experiment")
        return cls(
            source_type=str(payload.get("source_type") or "case_memory"),
            source_case_ids=tuple(str(s) for s in payload.get("source_case_ids", [])),
            source_experiment=int(source_experiment) if source_experiment is not None else None,
            failure_family=str(payload.get("failure_family") or ""),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            compiled_date=str(payload.get("compiled_date") or RUN_DATE),
        )


# ---------------------------------------------------------------------------
# Constraint template
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintTemplate:
    """One lightweight constraint guard compiled from a recurring failure family.

    A ``ConstraintTemplate`` is intentionally cheap to store and evaluate.
    No matrix operations, no numerical models — just string checks and counters.

    Fields
    ------
    template_id
        Deterministic identifier of the form
        ``"<kind>:<model_name>:<benchmark_slice>:<failure_family>"``.
    kind
        One of ``"text_pattern_guard"``, ``"budget_addition"``, or
        ``"verifier_guard_clause"``.
    model_name
        The model whose failure patterns drove this template.
    benchmark_slice
        The ``benchmark/domain`` slice this template applies to.
    failure_family
        The primary violation family (first token of the violation type label).
    guard_patterns
        Substrings to search for in response text.  Non-empty only when
        ``kind == "text_pattern_guard"``.  Patterns are lowercased so the
        ``apply()`` comparison can use ``pattern in response_text.lower()``.
    budget_delta
        Number of additional verifier passes to add.  Positive only when
        ``kind == "budget_addition"``.
    guard_name
        Short identifier for the guard clause hook.  Non-empty only when
        ``kind == "verifier_guard_clause"``.
    guard_reason
        Human-readable explanation of why this guard clause was added.
    support
        Aggregate support count from the qualifying case entry.
    confidence
        Aggregate confidence score from the qualifying case entry.
    provenance
        One or more ``ConstraintProvenance`` records tracing this template
        back to the originating case memory entries.
    """

    template_id: str
    kind: str
    model_name: str
    benchmark_slice: str
    failure_family: str
    guard_patterns: tuple[str, ...]
    budget_delta: int
    guard_name: str
    guard_reason: str
    support: int
    confidence: float
    provenance: tuple[ConstraintProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "kind": self.kind,
            "model_name": self.model_name,
            "benchmark_slice": self.benchmark_slice,
            "failure_family": self.failure_family,
            "guard_patterns": list(self.guard_patterns),
            "budget_delta": self.budget_delta,
            "guard_name": self.guard_name,
            "guard_reason": self.guard_reason,
            "support": self.support,
            "confidence": self.confidence,
            "provenance": [p.to_dict() for p in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstraintTemplate:
        return cls(
            template_id=str(payload.get("template_id") or ""),
            kind=str(payload.get("kind") or ""),
            model_name=str(payload.get("model_name") or ""),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            failure_family=str(payload.get("failure_family") or ""),
            guard_patterns=tuple(str(p) for p in payload.get("guard_patterns", [])),
            budget_delta=int(payload.get("budget_delta") or 0),
            guard_name=str(payload.get("guard_name") or ""),
            guard_reason=str(payload.get("guard_reason") or ""),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            provenance=tuple(
                ConstraintProvenance.from_dict(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
        )


# ---------------------------------------------------------------------------
# Compilation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintAdditionResult:
    """Compiled set of constraint templates derived from a CaseMemory snapshot.

    This is the output artifact of ``ConstraintAdditionCompiler.compile()``.
    It serializes deterministically: ``to_dict()`` uses only list/dict/str/int/
    float/None — stable types for ``json.dumps(..., sort_keys=True)``.

    Fields
    ------
    run_date
        Fixed compile date ``"20260413"`` embedded for traceability.
    templates
        Sorted tuple of ``ConstraintTemplate`` objects ordered by
        ``template_id`` for deterministic serialization.
    """

    run_date: str = RUN_DATE
    templates: tuple[ConstraintTemplate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "run_date": self.run_date,
            "summary": {"n_templates": len(self.templates)},
            "templates": [t.to_dict() for t in self.templates],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstraintAdditionResult:
        version = payload.get("version")
        if version is not None and int(version) != VERSION:
            raise ValueError(
                f"Unsupported ConstraintAdditionResult format (expected version={VERSION})"
            )
        return cls(
            run_date=str(payload.get("run_date") or RUN_DATE),
            templates=tuple(
                ConstraintTemplate.from_dict(item)
                for item in payload.get("templates", [])
                if isinstance(item, dict)
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> ConstraintAdditionResult:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("ConstraintAdditionResult payload must be a JSON object")
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class ConstraintAdditionCompiler:
    """Compile lightweight constraint templates from mature case memory entries.

    Usage
    -----
    ::

        memory = CaseMemory.load("results/case_memory.json")
        result = ConstraintAdditionCompiler().compile(memory)
        # result is a ConstraintAdditionResult — serializable, additive

    Qualification thresholds
    ------------------------
    Only ``CaseEntry`` objects satisfying both ``entry.support >= min_support``
    and ``entry.confidence >= min_confidence`` produce templates.  The defaults
    (3 / 0.85) are tuned so that a family must appear in at least three
    distinct traces with high average confidence before it influences the
    constraint set.

    Compile cost
    ------------
    O(N) where N = number of entries in the case memory.  No numerical ops.
    """

    def __init__(
        self,
        *,
        min_support: int = _DEFAULT_MIN_SUPPORT,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._min_support = min_support
        self._min_confidence = min_confidence

    def compile(self, case_memory: CaseMemory) -> ConstraintAdditionResult:
        """Produce a ``ConstraintAdditionResult`` from *case_memory*.

        The result is additive — it does not mutate ``case_memory`` and it
        does not contain ``ThresholdOverride`` or ``RoutingHint`` objects.
        """
        entries = case_memory.entries()
        qualifying = [
            entry
            for entry in entries
            if entry.support >= self._min_support
            and entry.confidence >= self._min_confidence
        ]

        templates: list[ConstraintTemplate] = []
        for entry in qualifying:
            templates.extend(self._templates_for_entry(entry))

        # Sort by template_id for deterministic output
        templates.sort(key=lambda t: t.template_id)

        return ConstraintAdditionResult(
            run_date=RUN_DATE,
            templates=tuple(templates),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _templates_for_entry(self, entry: CaseEntry) -> list[ConstraintTemplate]:
        """Return up to three templates (one per kind) for *entry*."""
        failure_family = self._primary_family(entry)
        if not failure_family:
            return []

        provenance = (self._provenance_from_entry(entry, failure_family),)

        return [
            self._make_text_pattern_guard(entry, failure_family, provenance),
            self._make_budget_addition(entry, failure_family, provenance),
            self._make_verifier_guard_clause(entry, failure_family, provenance),
        ]

    @staticmethod
    def _primary_family(entry: CaseEntry) -> str:
        """Return the first non-empty violation family for *entry*, or ''."""
        if entry.key.violation_families:
            return entry.key.violation_families[0]
        if entry.violation_types:
            return entry.violation_types[0].split(":", 1)[0]
        return ""

    def _make_text_pattern_guard(
        self,
        entry: CaseEntry,
        family: str,
        provenance: tuple[ConstraintProvenance, ...],
    ) -> ConstraintTemplate:
        """Build a text_pattern_guard from violation type labels.

        Patterns are derived from the human-readable parts of violation type
        strings.  Each token after a colon separator becomes a searchable
        pattern (lowercased).  Limited to ``_MAX_GUARD_PATTERNS`` items so
        the substring scan stays cheap.

        Example: violation_type ``"semantic:answer_target_mismatch"`` →
        patterns ``["semantic", "answer_target_mismatch"]``.
        """
        patterns: list[str] = []
        seen: set[str] = set()
        for vtype in entry.violation_types:
            for part in vtype.lower().split(":"):
                token = part.strip()
                if token and token not in seen:
                    seen.add(token)
                    patterns.append(token)
                    if len(patterns) >= _MAX_GUARD_PATTERNS:
                        break
            if len(patterns) >= _MAX_GUARD_PATTERNS:
                break

        template_id = (
            f"text_pattern_guard:{entry.key.model_name}:"
            f"{entry.key.benchmark_slice}:{family}"
        )
        return ConstraintTemplate(
            template_id=template_id,
            kind="text_pattern_guard",
            model_name=entry.key.model_name,
            benchmark_slice=entry.key.benchmark_slice,
            failure_family=family,
            guard_patterns=tuple(patterns),
            budget_delta=0,
            guard_name="",
            guard_reason="",
            support=entry.support,
            confidence=entry.confidence,
            provenance=provenance,
        )

    def _make_budget_addition(
        self,
        entry: CaseEntry,
        family: str,
        provenance: tuple[ConstraintProvenance, ...],
    ) -> ConstraintTemplate:
        """Build a budget_addition template.

        ``budget_delta`` scales with support up to a maximum of 3 so the
        verifier never runs an unbounded number of extra passes from a single
        template.
        """
        budget_delta = min(3, max(1, entry.support // self._min_support))
        template_id = (
            f"budget_addition:{entry.key.model_name}:"
            f"{entry.key.benchmark_slice}:{family}"
        )
        return ConstraintTemplate(
            template_id=template_id,
            kind="budget_addition",
            model_name=entry.key.model_name,
            benchmark_slice=entry.key.benchmark_slice,
            failure_family=family,
            guard_patterns=(),
            budget_delta=budget_delta,
            guard_name="",
            guard_reason="",
            support=entry.support,
            confidence=entry.confidence,
            provenance=provenance,
        )

    def _make_verifier_guard_clause(
        self,
        entry: CaseEntry,
        family: str,
        provenance: tuple[ConstraintProvenance, ...],
    ) -> ConstraintTemplate:
        """Build a verifier_guard_clause that names a pre-check gate.

        The ``guard_name`` follows the pattern
        ``"guard:<failure_family>:<benchmark_slice>"`` so pipeline code can
        register handlers by name without needing to parse violation types.
        """
        guard_name = f"guard:{family}:{entry.key.benchmark_slice}"
        guard_reason = (
            f"Recurring {family} failures (support={entry.support}, "
            f"confidence={entry.confidence:.2f}) compiled from case memory "
            f"on {RUN_DATE}"
        )
        template_id = (
            f"verifier_guard_clause:{entry.key.model_name}:"
            f"{entry.key.benchmark_slice}:{family}"
        )
        return ConstraintTemplate(
            template_id=template_id,
            kind="verifier_guard_clause",
            model_name=entry.key.model_name,
            benchmark_slice=entry.key.benchmark_slice,
            failure_family=family,
            guard_patterns=(),
            budget_delta=0,
            guard_name=guard_name,
            guard_reason=guard_reason,
            support=entry.support,
            confidence=entry.confidence,
            provenance=provenance,
        )

    @staticmethod
    def _provenance_from_entry(
        entry: CaseEntry,
        failure_family: str,
    ) -> ConstraintProvenance:
        """Build a ``ConstraintProvenance`` from a ``CaseEntry``.

        Collects all ``case_id`` values from the entry's provenance records
        into ``source_case_ids`` and takes the experiment number from the
        first record that has one.
        """
        source_case_ids: list[str] = []
        source_experiment: int | None = None
        for cp in entry.provenance:
            if cp.case_id and cp.case_id not in source_case_ids:
                source_case_ids.append(cp.case_id)
            if source_experiment is None and cp.source_experiment is not None:
                source_experiment = cp.source_experiment

        return ConstraintProvenance(
            source_type="case_memory",
            source_case_ids=tuple(source_case_ids),
            source_experiment=source_experiment,
            failure_family=failure_family,
            support=entry.support,
            confidence=entry.confidence,
            compiled_date=RUN_DATE,
        )


# ---------------------------------------------------------------------------
# Runtime registry
# ---------------------------------------------------------------------------


class ConstraintAdditionRegistry:
    """Runtime lookup facade over a compiled ``ConstraintAdditionResult``.

    Usage
    -----
    ::

        result = ConstraintAdditionCompiler().compile(memory)
        registry = ConstraintAdditionRegistry(result)

        # Retrieve templates for a specific failure family:
        templates = registry.lookup(model_name, benchmark_slice, "semantic")

        # During verification — filter to templates that fire for this response:
        active = registry.apply(model_name, benchmark_slice, violation_types, response_text)

    Lookup cost
    -----------
    O(T) where T = total number of compiled templates.  All comparisons are
    string equality or ``in`` substring checks — safe at inference speed.
    """

    def __init__(self, result: ConstraintAdditionResult) -> None:
        self._templates: tuple[ConstraintTemplate, ...] = result.templates

    def lookup(
        self,
        model_name: str,
        benchmark_slice: str,
        failure_family: str,
    ) -> tuple[ConstraintTemplate, ...]:
        """Return templates matching *model_name*, *benchmark_slice*, and *failure_family*.

        Results are sorted by ``template_id`` for deterministic ordering.
        Returns an empty tuple when no templates match.
        """
        matches = [
            t
            for t in self._templates
            if t.failure_family == failure_family
            and t.model_name == model_name
            and t.benchmark_slice == benchmark_slice
        ]
        return tuple(sorted(matches, key=lambda t: t.template_id))

    def apply(
        self,
        model_name: str,
        benchmark_slice: str,
        violation_types: tuple[str, ...],
        response_text: str,
    ) -> tuple[ConstraintTemplate, ...]:
        """Return templates that are active for this verification context.

        Activation rules by kind:

        ``text_pattern_guard``
            Active when at least one of the template's ``guard_patterns`` is
            found as a substring of ``response_text.lower()``.  A template
            with no guard patterns is never activated.

        ``budget_addition`` / ``verifier_guard_clause``
            Active when at least one of the template's ``failure_family``
            values appears as a prefix of any ``violation_types`` element
            (i.e. ``violation_type.startswith(failure_family)``).

        All matching templates are returned sorted by ``template_id``.
        """
        lowered_response = response_text.lower()
        violation_prefixes = tuple(vt.split(":", 1)[0] for vt in violation_types)

        active: list[ConstraintTemplate] = []
        for template in self._templates:
            if template.model_name != model_name:
                continue
            if template.benchmark_slice != benchmark_slice:
                continue

            if template.kind == "text_pattern_guard":
                if not template.guard_patterns:
                    continue
                if any(pattern in lowered_response for pattern in template.guard_patterns):
                    active.append(template)

            elif template.kind in {"budget_addition", "verifier_guard_clause"}:
                if template.failure_family in violation_prefixes:
                    active.append(template)

        return tuple(sorted(active, key=lambda t: t.template_id))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "RUN_DATE",
    "VERSION",
    "ConstraintProvenance",
    "ConstraintTemplate",
    "ConstraintAdditionResult",
    "ConstraintAdditionCompiler",
    "ConstraintAdditionRegistry",
]
