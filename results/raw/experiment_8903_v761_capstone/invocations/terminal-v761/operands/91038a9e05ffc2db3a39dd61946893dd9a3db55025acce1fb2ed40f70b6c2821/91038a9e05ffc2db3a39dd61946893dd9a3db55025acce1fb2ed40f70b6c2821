"""REQ-REPORT-8845: pay for request-local indexing before accepting exact witnesses."""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

from carnot.reporting import ordered_evidence_cost_8831 as reference

__all__ = ["reference", "meter"]
from carnot.reporting import evidence_operation_meter_8818 as meter
from carnot.reporting import evidence_set_methods_8768 as methods

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8845_v757_indexed_witness", "exp8845-indexed-witness"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/indexed_witness_8845.py",
    "python/carnot/reporting/indexed_witness_runner_8845.py",
    CLI,
]
TESTS = ["tests/python/test_indexed_witness_8845.py"]
CAP, SEED = 200000, 7578845
REQUEST: ContextVar[object | None] = ContextVar("indexed_witness_request", default=None)
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
Fact = tuple[str, str, str, bool]
Key = tuple[str, str | None, str | None]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so a long child cannot hide a stalled phase."""
    print(f"[exp8845] phase={phase} completed={completed} pending={pending}", flush=True)


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
    parse()(payload)
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


@lru_cache(maxsize=1)
def parse() -> Any:
    """Compile the same strict parser once; every invocation still pays its input visits."""
    return meter.instrument(methods.parse)


def attempt(payload: Json, *, indexed: bool = False, cap: int | None = CAP) -> Json:
    """The default remains the unchanged always-joint path; exhaustion erases partial proof."""
    if not indexed:
        return dict(reference.attempt("always_joint", payload, cap))
    counter = meter.Counter(cap)
    counter.counts.update(dict.fromkeys(EXTRA, 0))
    token, request = meter.CURRENT.set(counter), REQUEST.set(object())
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    exhausted = False
    try:
        result = solve(payload)
    except meter.Exhausted:
        result = dict(decision="insufficient", witness=None, supporting_bindings=0)
        exhausted = True
    finally:
        REQUEST.reset(request)
        meter.CURRENT.reset(token)
    return dict(
        result,
        budget_exhausted=exhausted,
        primitive_limit=cap,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        wall_time_s=(time.monotonic_ns() - start) / 1e9,
        cpu_time_s=(time.process_time_ns() - cpu) / 1e9,
    )


def stable(value: Any) -> Any:
    """Timing is retained as measurement but cannot decide deterministic replay parity."""
    return reference.stable(value)


def valid_witness(payload: Json, row: Json) -> bool:
    """Reconstruct every supported predicate independently of the index candidate table."""
    if row["decision"] != "support":
        return row.get("witness") is None
    binding = row["witness"]
    return not reference.conflicts(payload) and all(
        [*(binding.get(term, term) for term in predicate), True] in payload["facts"]
        for predicate in payload["query"]
    )


def fixtures() -> list[Json]:
    """Retain original V756 interventions beside small hand-written adversaries."""
    edges = [[i, (i + 1) % 8] for i in range(8)]
    rows = reference.base.original.variants(edges, 2, False)
    rows += [
        dict(
            condition="valid_alternative",
            input=reference.base.original.variants(edges, 2, True)[0]["input"],
        )
    ]
    for name, facts, query in [
        ("duplicates", [["a", "r", "b", True]] * 2, [["?x", "r", "b"]]),
        ("repeated_variable", [["a", "r", "b", True]], [["?x", "r", "?x"]]),
        (
            "absent_join",
            [["a", "r", "b", True], ["c", "s", "d", True]],
            [["?x", "r", "b"], ["?x", "s", "d"]],
        ),
        (
            "outside_conflict",
            [["a", "r", "b", True], ["c", "s", "d", True], ["c", "s", "d", False]],
            [["a", "r", "b"]],
        ),
        (
            "swapped_bindings",
            [["a", "r", "b", True], ["a", "s", "b", True]],
            [["?x", "r", "?y"], ["?y", "s", "?x"]],
        ),
        ("explicit_refutation", [["a", "r", "b", False]], [["a", "r", "b"]]),
        ("witness_order", [["z", "r", "b", True], ["a", "r", "b", True]], [["?x", "r", "b"]]),
    ]:
        rows.append(dict(condition=name, input=dict(facts=facts, query=query)))
    return list(rows)


def lifetime_controls() -> list[Json]:
    """A released request and a different active request must reject the same saved index."""
    counter = meter.Counter(None)
    counter.counts.update(dict.fromkeys(EXTRA, 0))
    paid = meter.CURRENT.set(counter)
    owner = object()
    request = REQUEST.set(owner)
    facts = [["a", "r", "b", True]]
    index, _, _ = build_index(dict(facts=facts))
    snapshot: tuple[Fact, ...] = (("a", "r", "b", True),)
    facts[0][0] = "mutated"
    rows = [
        dict(
            check="source_mutation_isolated",
            passed=index.candidates(["?x", "r", "b"], {}) == snapshot,
        )
    ]
    try:
        index.table[("r", None, "b")] = ()  # type: ignore[index]
    except TypeError:
        rows.append(dict(check="immutable_table", passed=True))
    REQUEST.reset(request)
    for label, current in [("stale_reuse_rejected", None), ("cross_query_rejected", object())]:
        request = REQUEST.set(current)
        try:
            index.candidates(["?x", "r", "b"], {})
        except ValueError:
            rows.append(dict(check=label, passed=True))
        finally:
            REQUEST.reset(request)
    meter.CURRENT.reset(paid)
    return rows
