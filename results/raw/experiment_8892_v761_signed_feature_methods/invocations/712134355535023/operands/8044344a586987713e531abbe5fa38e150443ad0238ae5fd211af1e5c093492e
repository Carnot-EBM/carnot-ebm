"""REQ-VERIFY-8768: opt-in relational sufficiency without truth-bearing features."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from itertools import product
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v749_verifier_protocol as historical
from carnot.reporting.verifier_family_surface_8755 import verifier as historical_verifier

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8768_v751_evidence_set_methods", "exp8768-evidence-set-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/evidence_set_{s}_8768.py"
    for s in ["methods", "corpus", "analysis", "report", "runner"]
] + [CLI]
TESTS = ["tests/python/test_evidence_set_methods_8768.py"]
CONSUMERS = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
ARMS = ("local_record", "joint_symbolic", "evidence_set")
SEED = 7518768
MODEL_SPECS: list[Json] = []


class Decision(StrEnum):
    """Keep absence distinct from refutation so callers can escalate unknowns."""

    SUPPORT = "support"
    REFUTE = "refute"
    INSUFFICIENT = "insufficient"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts because a quiet child must remain distinguishable from a stall."""
    print(f"[exp8768] phase={phase} completed={completed} pending={pending}", flush=True)


def parse(value: Any) -> Json:
    """Allow only records and predicates so labels and construction metadata cannot leak."""
    if not isinstance(value, dict) or set(value) != {"facts", "query"}:
        raise ValueError("input_fields")
    if (
        not isinstance(value["facts"], list)
        or not isinstance(value["query"], list)
        or not value["query"]
    ):
        raise ValueError("input_lists")
    for kind, width in [("facts", 4), ("query", 3)]:
        for row in value[kind]:
            if (
                not isinstance(row, list)
                or len(row) != width
                or any(not isinstance(x, str) or not x or x.strip() != x for x in row[:3])
            ):
                raise ValueError("record_shape")
            if row[1].startswith("?") or (
                kind == "facts"
                and (type(row[3]) is not bool or any(x.startswith("?") for x in row[:3]))
            ):
                raise ValueError("record_types")
    return value


def unique_object(pairs: list[tuple[str, Any]]) -> Json:
    """Duplicate keys could silently replace evidence during transport."""
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("duplicate_key")
    return dict(pairs)


def reject_constant(value: str) -> None:
    """Nonfinite JSON is outside the signed-record protocol."""
    raise ValueError("nonfinite:" + value)


def decode(text: str) -> Json:
    """Use the same strict boundary for real serialized children and direct callers."""
    return parse(json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant))


def decide(payload: Json) -> Json:
    """Join positive records; explicit universal counterevidence can refute an existential.

    This is an opt-in symbolic rule, not a learned semantic extractor. Partial
    matches explain missing premises; incompatible shared bindings explain joins.
    Any contradictory records escalate before a positive witness can be accepted.
    """
    p = parse(payload)
    facts, query = p["facts"], p["query"]
    signs: dict[tuple[str, ...], set[bool]] = {}
    for s, r, o, sign in facts:
        signs.setdefault((s, r, o), set()).add(sign)
    conflicts = sorted(k for k, values in signs.items() if len(values) > 1)
    variables = sorted({x for q in query for x in q if x.startswith("?")})
    counts = Counter(x for q in query for x in q if x.startswith("?"))
    bindings: list[dict[str, str]] = [{}]
    missing = []
    for index, predicate in enumerate(query):
        joined, local = [], False
        for fact in facts:
            if not fact[3]:
                continue
            for binding in bindings + ([{}] if {} not in bindings else []):
                candidate = dict(binding)
                matched = True
                for term, actual in zip(predicate, fact[:3], strict=True):
                    if term.startswith("?"):
                        matched &= candidate.setdefault(term, actual) == actual
                    else:
                        matched &= term == actual
                if matched:
                    local = True
                    if binding in bindings:
                        joined.append(candidate)
        if not local:
            missing.append(index)
        bindings = joined
    negative = sorted(k for k, values in signs.items() if values == {False})
    entities = sorted(
        {x for s, _, o, _ in facts for x in (s, o)}
        | {x for q in query for x in (q[0], q[2]) if not x.startswith("?")}
    )
    refuted = all(
        any(tuple(dict(zip(variables, values)).get(x, x) for x in q) in negative for q in query)
        for values in product(entities, repeat=len(variables))
    )
    decision = (
        Decision.INSUFFICIENT
        if conflicts
        else Decision.SUPPORT
        if bindings
        else (Decision.REFUTE if refuted else Decision.INSUFFICIENT)
    )
    return dict(
        decision=decision,
        required_joins=sorted(x for x in variables if counts[x] > 1),
        missing_premises=missing,
        missing_join=not bindings and not missing,
        explicit_refutations=[list(k) for k in negative],
        conflicts=[list(k) for k in conflicts],
        supporting_bindings=len(bindings),
    )


def compare(arm: str, payload: Json) -> tuple[str, int]:
    """Historical evaluators remain unchanged; local premises deliberately lack shared joins."""
    p = parse(payload)
    if arm == "evidence_set":
        return str(decide(p)["decision"]), 1
    if arm == "joint_symbolic":
        result = historical.truth(p["facts"], p["query"])
        calls = 1
    elif arm == "local_record":
        result = historical_verifier("omitted_shared_variable_join", p["facts"], p["query"])
        calls = len(p["query"])
    else:
        raise ValueError("arm")
    return {"true": "support", "false": "refute", "unknown": "insufficient"}[result], calls


def method() -> Json:
    """Freeze decisions and promotion before any evaluation result can select rules."""
    return dict(
        schema="carnot.v751.evidence-set.v1",
        seed=SEED,
        arms=list(ARMS),
        semantics="Open world; absence unknown; explicit opposite can refute; any conflict insufficient.",
        opt_in=True,
        maximum_evaluator_invocations=5,
        early_exit_between_units=False,
        feature_allowlist=["facts", "query"],
        split_design="60 distinct four-edge development graphs; 120 distinct five-edge evaluation graphs; disjoint entities, templates and graph isomorphism classes",
        statistical_protocol=dict(
            primary="macro three-class error averaged per evaluation world",
            draws=10000,
            cluster="world_id",
            paired=True,
            familywise_confidence=0.975,
            comparators=2,
            individual_two_sided_confidence=0.9875,
            percentile_indices=[62, 9937],
            multiplicity="Bonferroni two comparisons",
            promotion="both lower improvements > 0; no added false acceptance or valid rejection; exact correct invariance",
            seed=SEED,
        ),
    )
