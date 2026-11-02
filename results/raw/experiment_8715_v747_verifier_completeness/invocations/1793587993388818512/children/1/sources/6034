"""Execute the bounded relation fragment used by Exp7195.

The executor treats extraction as untrusted input. It checks byte evidence and
entity bindings before it applies relation rules. Missing or conflicting input
stays unknown, because absent evidence does not prove a claim false.

Spec refs: REQ-VERIFY-7195 and SCENARIO-VERIFY-7195-EXECUTION/UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Decision = Literal["supported", "contradicted", "unknown"]
Polarity = Literal["positive", "negative"]
CanonicalRelation = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class EntityBinding:
    """Bind one explicit entity ID to exact bytes in the source."""

    entity_id: str
    surface: str
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class TypedRelation:
    """Hold one typed relation and its claimed source byte span."""

    subject_id: str
    operator: str
    object_id: str
    polarity: str
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Keep the decision and every reason that prevented a Boolean result."""

    decision: Decision
    abstention: bool
    uncertainty_reasons: tuple[str, ...]
    matched_relation_indexes: tuple[int, ...]
    normalized_claim: CanonicalRelation | None


# Each directional inverse maps to one canonical forward operator. This makes
# argument direction part of execution instead of a token-overlap feature.
_OPERATORS: dict[str, tuple[str, bool, bool]] = {
    "precedes": ("precedes", False, False),
    "follows": ("precedes", True, False),
    "starts before": ("starts_before", False, False),
    "starts after": ("starts_before", True, False),
    "ends before": ("ends_before", False, False),
    "ends after": ("ends_before", True, False),
    "occurs before": ("occurs_before", False, False),
    "occurs after": ("occurs_before", True, False),
    "is separated from": ("separated", False, True),
    "equals": ("equals", False, True),
}
_ORDERING_FAMILIES = frozenset({"precedes", "starts_before", "ends_before", "occurs_before"})
_COMPLEMENTS = {"separated": "equals", "equals": "separated"}


def _valid_offset(start: object, end: object, size: int) -> bool:
    """Accept a non-empty half-open byte range inside the source."""

    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= size
    )


def _binding_errors(source: bytes, bindings: tuple[EntityBinding, ...]) -> tuple[str, ...]:
    """Find invalid or ambiguous mappings without guessing an entity."""

    reasons: list[str] = []
    by_id: dict[str, list[EntityBinding]] = {}
    by_surface: dict[str, set[str]] = {}
    for binding in bindings:
        by_id.setdefault(binding.entity_id, []).append(binding)
        by_surface.setdefault(binding.surface, set()).add(binding.entity_id)
        if not _valid_offset(binding.source_start, binding.source_end, len(source)):
            reasons.append("invalid_source_offsets")
        elif source[binding.source_start : binding.source_end] != binding.surface.encode("utf-8"):
            reasons.append("entity_surface_mismatch")
    if any(len(values) != 1 for values in by_id.values()) or any(
        len(entity_ids) != 1 for entity_ids in by_surface.values()
    ):
        reasons.append("ambiguous_entity_mapping")
    return tuple(dict.fromkeys(reasons))


def _normalize(relation: TypedRelation) -> CanonicalRelation | None:
    """Convert inverse spellings into one directional proposition."""

    spec = _OPERATORS.get(relation.operator)
    if spec is None:
        return None
    family, reverse, symmetric = spec
    subject, obj = relation.subject_id, relation.object_id
    if reverse:
        subject, obj = obj, subject
    if symmetric and obj < subject:
        subject, obj = obj, subject
    return family, subject, obj


def _relation_errors(
    source: bytes,
    bindings: tuple[EntityBinding, ...],
    relations: tuple[TypedRelation, ...],
    claim: TypedRelation,
) -> tuple[str, ...]:
    """Validate typed values and exact evidence spans before execution."""

    reasons = list(_binding_errors(source, bindings))
    known_ids = {binding.entity_id for binding in bindings}
    all_relations = (*relations, claim)
    if any(relation.operator not in _OPERATORS for relation in all_relations):
        reasons.append("unsupported_relation_operator")
    if any(relation.polarity not in {"positive", "negative"} for relation in all_relations):
        reasons.append("unsupported_polarity")
    if any(
        relation.subject_id not in known_ids or relation.object_id not in known_ids
        for relation in all_relations
    ):
        reasons.append("missing_entity_mapping")

    binding_by_id = {binding.entity_id: binding for binding in bindings}
    for relation in relations:
        if not _valid_offset(relation.source_start, relation.source_end, len(source)):
            reasons.append("invalid_source_offsets")
            continue
        span = source[relation.source_start : relation.source_end]
        if relation.operator.encode("utf-8") not in span:
            reasons.append("relation_span_mismatch")
        if relation.polarity == "negative" and b"not" not in span:
            reasons.append("polarity_span_mismatch")
        for entity_id in (relation.subject_id, relation.object_id):
            binding = binding_by_id.get(entity_id)
            if binding is not None and not (
                relation.source_start <= binding.source_start
                and binding.source_end <= relation.source_end
            ):
                reasons.append("entity_outside_relation_span")
    return tuple(dict.fromkeys(reasons))


def _assertions(relation: TypedRelation) -> tuple[tuple[CanonicalRelation, bool], ...]:
    """Expand one literal into the facts that the bounded fragment entails."""

    normalized = _normalize(relation)
    if normalized is None:
        return ()
    truth = relation.polarity == "positive"
    assertions = [(normalized, truth)]
    family, subject, obj = normalized
    if family in _ORDERING_FAMILIES and truth:
        assertions.append(((family, obj, subject), False))
    complement = _COMPLEMENTS.get(family)
    if complement is not None:
        assertions.append(((complement, subject, obj), not truth))
    return tuple(assertions)


def execute_relation(
    source_bytes: bytes,
    entity_bindings: tuple[EntityBinding, ...],
    source_relations: tuple[TypedRelation, ...],
    claim_relation: TypedRelation,
) -> ExecutionResult:
    """Execute one claim while preserving any reason that requires abstention."""

    reasons = list(
        _relation_errors(source_bytes, entity_bindings, source_relations, claim_relation)
    )
    if not source_relations:
        reasons.append("missing_source_relation")
    normalized_claim = _normalize(claim_relation)
    if reasons or normalized_claim is None:
        return ExecutionResult("unknown", True, tuple(dict.fromkeys(reasons)), (), normalized_claim)

    assignments: dict[CanonicalRelation, set[bool]] = {}
    contributors: dict[CanonicalRelation, set[int]] = {}
    for index, relation in enumerate(source_relations):
        for proposition, truth in _assertions(relation):
            assignments.setdefault(proposition, set()).add(truth)
            contributors.setdefault(proposition, set()).add(index)

    values = assignments.get(normalized_claim, set())
    matched = tuple(sorted(contributors.get(normalized_claim, set())))
    if len(values) > 1:
        return ExecutionResult(
            "unknown", True, ("contradictory_evidence",), matched, normalized_claim
        )
    if not values:
        return ExecutionResult("unknown", True, ("unresolved_evidence",), (), normalized_claim)
    expected = claim_relation.polarity == "positive"
    decision: Decision = "supported" if expected in values else "contradicted"
    return ExecutionResult(decision, False, (), matched, normalized_claim)
