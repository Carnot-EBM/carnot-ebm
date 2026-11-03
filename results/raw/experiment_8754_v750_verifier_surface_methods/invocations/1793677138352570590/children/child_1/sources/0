"""REQ-VERIFY-8754: preserve signed truth before any verifier outcome is observed."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import random
from typing import Any

from carnot.reporting import v749_verifier_protocol as historical
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.current_work_receipt import canonical_hash

__all__ = ["historical", "reference", "load"]

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8754_v750_verifier_surface_methods", "exp8754-verifier-surface-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/verifier_surface_{s}_8754.py" for s in ["methods", "runner", "report"]
] + [CLI]
TESTS = ["tests/python/test_verifier_surface_methods_8754.py"]
CONSUMERS = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
PROTOCOL = (
    ROOT
    / "results/raw/experiment_8737_v749_contract_methods/invocations/1793646952232607865/verifier_transfer_protocol.json"
)
PIN = "sha256:9517880b83a79f5e36ff72e69d3b222539e6b68e07ba7ba3b28bdb2e54280f8b"
SEED = 7508754
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the supervisor can distinguish long checks from a stall."""
    print(f"[exp8754] phase={phase} completed={completed} pending={pending}", flush=True)


def retained(path: Path = PROTOCOL) -> Json:
    """A rejected producer can supply pinned inputs without supplying qualification."""
    value = load(dict(path=str(path), sha256=PIN))
    if value != historical.freeze():
        raise ValueError("retained_protocol_drift")
    return dict(value)


def truth(facts: list[list[Any]], query: list[list[str]] | None) -> str:
    """Check each possible shared binding using explicit positive and negative sets.

    Missing facts carry no negative evidence. This implementation never imports
    a verifier and rejects inconsistent inputs rather than choosing one sign.
    """
    positive = {tuple(f[:3]) for f in facts if f[3] is True}
    negative = {tuple(f[:3]) for f in facts if f[3] is False}
    if positive & negative:
        raise ValueError("inconsistent_signed_facts")
    if not query:
        return "unknown"
    domain = sorted(
        {str(x) for f in facts for x in (f[0], f[2])}
        | {x for q in query for x in (q[0], q[2]) if not x.startswith("?")}
    )
    variables = sorted({x for q in query for x in (q[0], q[2]) if x.startswith("?")})
    possible = set()
    for values in product(domain, repeat=len(variables)):
        binding = dict(zip(variables, values))
        atoms = [tuple(binding.get(x, x) for x in q) for q in query]
        possible.add(
            "false"
            if any(a in negative for a in atoms)
            else "true"
            if all(a in positive for a in atoms)
            else "unknown"
        )
    return "true" if "true" in possible else "unknown" if "unknown" in possible else "false"


def equivalence(canonical: Json, transformed: Json) -> list[Json]:
    """Validate labels independently on both paired representations before scoring."""
    rows = []
    for original, changed in zip(canonical["worlds"], transformed["worlds"], strict=True):
        for left, right in zip(original["probes"], changed["probes"], strict=True):
            expected, observed = (
                truth(left["facts"], left["query"]),
                truth(right["facts"], right["query"]),
            )
            rows.append(
                dict(
                    world_id=original["world_id"],
                    transform=left["transform"],
                    canonical_truth=expected,
                    transformed_truth=observed,
                    passed=expected == observed == left["expected"] == right["expected"],
                    status="completed",
                    absolute_metric=int(
                        expected == observed == left["expected"] == right["expected"]
                    ),
                )
            )
    return rows


def surface(panel: Json) -> tuple[Json, list[Json], list[Json]]:
    """One published seed fixes all representations without consulting defect results."""
    result, controls = deepcopy(panel), []
    rng = random.Random(SEED)
    for world in result["worlds"]:
        entities = sorted(
            {str(x) for p in world["probes"] for f in p["facts"] for x in (f[0], f[2])}
        )
        names = [f"renamed_{i:02d}" for i in range(len(entities))]
        rng.shuffle(names)
        rename = dict(zip(entities, names))
        order = list(range(5))
        rng.shuffle(order)

        def records(facts: list[list[Any]]) -> list[list[Any]]:
            return [
                [rename[s], r, rename[o], t]
                for i in order
                if i < len(facts)
                for s, r, o, t in [facts[i]]
            ]

        def predicates(query: list[list[str]] | None) -> list[list[str]] | None:
            return [[rename.get(s, s), r, rename.get(o, o)] for s, r, o in query] if query else None

        for probe in world["probes"]:
            probe["facts"] = records(probe["facts"])
            probe["supporting_premises"] = records(probe["supporting_premises"])
            probe["query"] = predicates(probe["query"])
        world["facts"] = records(world["facts"])
        world["supporting_premises"] = records(world["supporting_premises"])
        world["query"] = predicates(world["query"])
        answer = world["proposed_answer"]
        for old in sorted(rename, key=len, reverse=True):
            answer = answer.replace(old, rename[old])
        world["proposed_answer"] = answer
        world["answer_spans"] = [dict(start=0, end=len(answer), text=answer)]
        controls.append(
            dict(
                world_id=world["world_id"],
                entity_bijection=rename,
                premise_order=order,
                random_seed=SEED,
                independent_cluster=world["world_id"],
            )
        )
        progress("surface_world", len(controls), 60 - len(controls))
    result["roster_sha256"] = canonical_hash(result["worlds"])
    result["surface"] = "fixed_bijection_and_premise_permutation"
    return result, controls, equivalence(panel, result)


def mutations(canonical: Json, transformed: Json) -> list[Json]:
    """Truth, identity and budgets have explicit rejection witnesses before scoring."""
    rows = []
    for mode in ["label", "truth", "identity", "calls", "schedule", "variant", "evaluation"]:
        changed = deepcopy(transformed)
        if mode == "label":
            changed["worlds"][0]["probes"][0]["expected"] = "false"
        elif mode == "truth":
            changed["worlds"][0]["probes"][0]["query"] = None
        elif mode == "identity":
            changed["worlds"][0]["world_id"] = "forged"
        elif mode == "calls":
            changed["rows"].pop()
        elif mode == "schedule":
            changed["schedules"]["baseline"].pop()
        elif mode == "variant":
            changed["variants"].pop()
        else:
            changed["worlds"][10]["partition"] = "development"
        rows.append(dict(mutation=mode, passed=not valid(canonical, changed)))
    return rows


def valid(canonical: Json, transformed: Json) -> bool:
    """A surface control cannot silently change the sampling or call contract."""
    return (
        len(transformed["rows"]) == 720
        and transformed["rows"] == canonical["rows"]
        and transformed["variants"] == canonical["variants"]
        and transformed["schedules"] == canonical["schedules"]
        and [(w["world_id"], w["family"], w["partition"]) for w in canonical["worlds"]]
        == [(w["world_id"], w["family"], w["partition"]) for w in transformed["worlds"]]
        and all(r["passed"] for r in equivalence(canonical, transformed))
    )
