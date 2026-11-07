"""REQ-REPORT-8791: keep construction metadata outside signed-record solver inputs."""

from copy import deepcopy
import json
import re
from typing import Any

from carnot.reporting import evidence_set_corpus_8768 as oracle
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CONDITIONS = (
    "clean_support",
    "omitted_required_premise",
    "shared_variable_mismatch",
    "signed_contradiction",
    "redundant_irrelevant_facts",
    "valid_alternative_proof",
)
BINS = (1024, 1536)


def tokens(payload: Json) -> int:
    """Protocol tokens count words and punctuation in canonical JSON without a model tokenizer."""
    return len(
        re.findall(r"\w+|[^\w\s]", json.dumps(payload, sort_keys=True, separators=(",", ":")))
    )


def pad(payload: Json, target: int) -> Json:
    """Only a relation absent from the query can supply the exact nuisance length."""
    result = deepcopy(payload)
    result["facts"].append(["padding", "unused", "z", True])
    remaining = target - tokens(result)
    if remaining < 0:
        raise ValueError("padding_target_too_small")
    result["facts"][-1][2] = " ".join(["z"] * (remaining + 1))
    assert tokens(result) == target
    return result


def rename(payload: Json) -> Json:
    """An opaque bijection changes every entity and relation while preserving variable identity."""
    names = sorted(
        {
            x
            for kind in ("facts", "query")
            for row in payload[kind]
            for x in row[:3]
            if not x.startswith("?")
        }
    )
    mapping = {x: f"opaque{i}" for i, x in enumerate(names)}
    return {
        kind: [
            [mapping.get(x, x) if isinstance(x, str) else x for x in row] for row in payload[kind]
        ]
        for kind in ("facts", "query")
    }


def fixtures() -> list[Json]:
    """All variants share one tiny world; the future roster is never measured here."""
    return units(((0, 1), (0, 2)), "tiny/shared-binding")


def units(edges: tuple[Any, ...], world: str) -> list[Json]:
    """Build paired interventions using neutral literals with evaluator-only world identifiers."""
    nodes = ["?x", "a", "b", "c", "d"]
    query = [[nodes[a], f"r{k}", nodes[b]] for k, (a, b) in enumerate(edges)]
    base: list[list[Any]] = [
        ["w" if x == "?x" else x, r, "w" if y == "?x" else y, True] for x, r, y in query
    ]
    hop = next(i for i, edge in enumerate(edges) if 0 in edge)
    position = 0 if edges[hop][0] == 0 else 2
    rows = []
    for condition in CONDITIONS:
        facts = deepcopy(base)
        if condition == "omitted_required_premise":
            facts.pop(hop)
        elif condition == "shared_variable_mismatch":
            facts[hop][position] = "z"
        elif condition == "signed_contradiction":
            facts.append([*base[hop][:3], False])
        elif condition == "redundant_irrelevant_facts":
            facts += [["c", "noise", "d", True]] * 2
        elif condition == "valid_alternative_proof":
            facts[hop][3] = False
            facts += [
                ["z" if x == "w" else x, r, "z" if y == "w" else y, sign] for x, r, y, sign in base
            ]
        for ordering in ("forward", "reverse"):
            for token_bin in BINS:
                payload = pad(dict(facts=facts, query=query), token_bin)
                if ordering == "reverse":
                    payload = {k: list(reversed(v)) for k, v in payload.items()}
                truth = oracle.enumerate_truth(payload)
                rows.append(
                    dict(
                        unit_id=f"{world}/{condition}/{ordering}/{token_bin}",
                        world_id=world,
                        condition=condition,
                        ordering=ordering,
                        token_bin=token_bin,
                        expected=truth["decision"],
                        oracle=truth,
                        input=payload,
                    )
                )
    return rows


def counterexamples() -> list[tuple[Json, str]]:
    """Explicit truth tables make absence, signed negation and repeated variables inspectable."""
    return [
        (dict(facts=[], query=[["a", "r", "b"]]), "insufficient"),
        (dict(facts=[["a", "r", "b", False]], query=[["a", "r", "b"]]), "refute"),
        (
            dict(facts=[["a", "r", "b", True], ["a", "r", "b", False]], query=[["a", "r", "b"]]),
            "insufficient",
        ),
        (dict(facts=[["a", "r", "b", True]], query=[["?x", "r", "?x"]]), "insufficient"),
        (dict(facts=[["a", "r", "a", True]], query=[["?x", "r", "?x"]]), "support"),
    ]


def audit(rows: list[Json]) -> bool:
    """Reconstruction rejects forged labels, condition metadata and repaired row hashes."""
    return (
        rows == fixtures()
        and all(
            row["expected"]
            == (
                "support"
                if row["condition"]
                in {"clean_support", "redundant_irrelevant_facts", "valid_alternative_proof"}
                else "insufficient"
            )
            for row in rows
        )
        and all(oracle.enumerate_truth(p)["decision"] == label for p, label in counterexamples())
    )


def roster() -> Json:
    """Different edge counts exclude every earlier graph and the reserved learning structures."""
    worlds: list[Json] = [
        dict(
            world_id=f"v753/intervention/{edge_count}/{i:03d}",
            edges=edges,
            edge_count=edge_count,
            structure=canonical_hash(edges),
        )
        for edge_count in (11, 12, 13)
        for i, edges in enumerate(oracle.graph_roster(edge_count, 40))
    ]
    historical = {
        canonical_hash(edges)
        for count, target in [(4, 60), (5, 120), (6, 132), (7, 120)]
        for edges in oracle.graph_roster(count, target)
    }
    return dict(
        worlds=worlds,
        independent_count=120,
        measured_count=0,
        reserved_learning_edge_counts=[8, 9, 10],
        historical_edge_counts=[4, 5, 6, 7],
        disjoint=len({r["structure"] for r in worlds}) == 120
        and not historical.intersection(r["structure"] for r in worlds)
        and all(len(r["edges"]) == r["edge_count"] for r in worlds),
        historical_structure_hashes=sorted(historical),
        clustering="world_id; all conditions, orders and bins share one cluster",
        canonicalization="directed graphs with distinguished shared-variable vertex; constant vertices permuted",
        comparison_status="frozen_only; execute in a later experiment",
    )


def freeze() -> Json:
    """Seal future evidence and independent labels without invoking any compared solver."""
    manifest = roster()
    inputs, labels, balance = [], [], []
    for i, world in enumerate(manifest["worlds"]):
        rows = units(world["edges"], world["world_id"])
        for row in rows:
            inputs.append(
                dict(unit_id=row["unit_id"], world_id=row["world_id"], input=row["input"])
            )
            labels.append({k: v for k, v in row.items() if k != "input"})
        for target in BINS:
            group = [r for r in rows if r["token_bin"] == target]
            supports = sum(r["expected"] == "support" for r in group)
            balance.append(
                dict(
                    world_id=world["world_id"],
                    token_bin=target,
                    supported=supports,
                    insufficient=len(group) - supports,
                    passed=supports == 6 and all(tokens(r["input"]) == target for r in group),
                )
            )
        if (i + 1) % 20 == 0:
            print(
                f"[exp8791] phase=seal_future_interventions completed={i + 1} pending={120 - i - 1}",
                flush=True,
            )
    return dict(split_manifest=manifest, inputs=inputs, labels=labels, length_balance_rows=balance)
