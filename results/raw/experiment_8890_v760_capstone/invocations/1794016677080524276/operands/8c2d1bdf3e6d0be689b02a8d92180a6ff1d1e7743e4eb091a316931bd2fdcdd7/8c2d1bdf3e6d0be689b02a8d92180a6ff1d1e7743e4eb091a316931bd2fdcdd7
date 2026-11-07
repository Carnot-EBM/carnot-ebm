"""REQ-REPORT-8858: installed dependency closure of the exact Exp8845 evaluator.

The algorithm and accounting remain identical to the preserved research source.
Only its imports and parser binding change; fixtures and publication stay outside
this module so requests depend on explicit records alone.
"""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import product
import json
from types import MappingProxyType
from typing import Any

from . import _indexed_witness_meter as meter

Json = dict[str, Any]
Fact = tuple[str, str, str, bool]
Key = tuple[str, str | None, str | None]
REQUEST: ContextVar[object | None] = ContextVar("installed_indexed_witness_request", default=None)


EXTRA = (
    "index_allocation",
    "index_build",
    "index_lookup",
    "conflict_checks",
    "result_serialization",
    "disposal",
    "snapshot",
    "witness_copy",
    "binding_copy",
)


def allocate(kind: str) -> None:
    """Pay disposal before allocation, so budget exhaustion cannot leave unpaid cleanup."""
    meter.charge(kind)
    meter.charge("disposal")


def copy_binding(binding: dict[str, str]) -> dict[str, str]:
    """Copying a growing assignment costs each entry, even when a later join rejects it."""
    allocate("binding_copy")
    for _ in binding:
        allocate("binding_copy")
    return dict(binding)


@dataclass(frozen=True)
class RequestIndex:
    """Immutable tuples preserve duplicates while a context token prevents stale reuse."""

    owner: object
    table: Mapping[Key, tuple[Fact, ...]]

    def candidates(self, predicate: list[str], binding: dict[str, str]) -> tuple[Fact, ...]:
        """Only already-bound terms filter facts; repeated variables are checked in the join."""
        meter.charge("index_lookup")
        if self.owner is not REQUEST.get():
            raise ValueError("stale_request_index")
        terms = []
        for term in predicate:
            meter.charge("lookup_attempts")
            terms.append(binding.get(term) if term.startswith("?") else term)
        return self.table.get((predicate[1], terms[0], terms[2]), ())


def build_index(payload: Json) -> tuple[RequestIndex, dict[tuple[str, ...], set[bool]], set[str]]:
    """Copy actual fact tuples once so later source mutation cannot change candidate lists."""
    table: dict[Key, list[Fact]] = {}
    signs: dict[tuple[str, ...], set[bool]] = {}
    domain: set[str] = set()
    for row in payload["facts"]:
        allocate("snapshot")
        s, r, o, sign = row
        fact = (s, r, o, sign)
        meter.charge("conflict_checks")
        signs.setdefault((s, r, o), set()).add(sign)
        allocate("snapshot")
        allocate("snapshot")
        domain.update((s, o))
        if sign:
            for subject, obj in product((None, s), (None, o)):
                allocate("index_build")
                table.setdefault((r, subject, obj), []).append(fact)
    frozen: dict[Key, tuple[Fact, ...]] = {}
    for key, facts in table.items():
        for _ in facts:
            allocate("index_allocation")
        frozen[key] = tuple(facts)
    owner = REQUEST.get()
    assert owner is not None
    index = RequestIndex(owner, MappingProxyType(frozen))
    return index, signs, domain


def solve(payload: Json) -> Json:
    """Join indexed positive facts; enumerate every remaining assignment for refutation.

    An absent match is unknown unless each possible shared assignment has an
    explicit negative predicate. Conflicts anywhere in the request prevent support.
    """
    meter.parse(payload)
    allocate("index_allocation")
    index, signs, domain = build_index(payload)
    variables: set[str] = set()
    conflict = False
    for values in signs.values():
        meter.charge("conflict_checks")
        conflict |= len(values) > 1
    for q in payload["query"]:
        for position, term in enumerate(q):
            allocate("query_visits")
            if term.startswith("?"):
                variables.add(term)
            elif position != 1:
                domain.add(term)
    allocate("binding_copy")
    bindings: list[dict[str, str]] = [{}]
    if not conflict:
        for predicate in payload["query"]:
            joined = []
            for binding in bindings:
                meter.charge("join_attempts")
                for fact in index.candidates(predicate, binding):
                    meter.charge("candidate_bindings")
                    candidate = copy_binding(binding)
                    matched = True
                    for term, actual in zip(predicate, fact[:3], strict=True):
                        meter.charge("fact_comparisons")
                        if term.startswith("?"):
                            matched &= candidate.setdefault(term, actual) == actual
                        else:
                            matched &= term == actual
                    if matched:
                        joined.append(candidate)
            bindings = joined
    decision = "insufficient"
    if not conflict and bindings:
        decision = "support"
    elif not conflict:
        decision = "refute"
        for assignment in product(sorted(domain), repeat=len(variables)):
            meter.charge("candidate_bindings")
            for _ in variables:
                allocate("binding_copy")
            binding = dict(zip(sorted(variables), assignment, strict=True))
            negative = False
            for q in payload["query"]:
                atom = []
                for term in q:
                    meter.charge("lookup_attempts")
                    atom.append(binding.get(term, term))
                meter.charge("conflict_checks")
                negative |= signs.get(tuple(atom), set()) == {False}
            if not negative:
                decision = "insufficient"
                break
    meter.charge("witness_copy")
    witness = copy_binding(bindings[0]) if decision == "support" else None
    order_difference = False
    if witness is not None:
        keys = sorted(variables)
        ordered = []
        for binding in bindings:
            meter.charge("witness_copy")
            witness_values = []
            for key in keys:
                allocate("witness_copy")
                witness_values.append(binding[key])
            ordered.append(tuple(witness_values))
        order_difference = tuple(witness[key] for key in keys) != min(ordered)
    result = dict(
        witness_order_difference=order_difference,
        decision=decision,
        witness=witness,
        supporting_bindings=len(bindings) if decision == "support" else 0,
    )
    meter.charge("result_serialization")
    json.dumps(result, sort_keys=True)
    del index, signs
    return result
