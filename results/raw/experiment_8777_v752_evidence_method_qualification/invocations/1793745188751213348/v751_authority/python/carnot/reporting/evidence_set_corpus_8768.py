"""REQ-REPORT-8768: construct records separately from an independent Boolean oracle."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CONDITIONS = (
    "complete_support",
    "explicit_refutation",
    "missing_required_hop",
    "disconnected_join",
    "irrelevant_distractor",
    "explicit_conflict",
)


def enumerate_truth(payload: Json) -> Json:
    """Enumerate compatible complete Boolean worlds, never calling an evaluated arm.

    Missing atoms may be either true or false. An existential conjunction is
    supported only if every completion satisfies it, refuted only if none do.
    Two witnesses suffice for unknown; inconsistent records have no models.
    """
    records = deepcopy(payload["facts"])
    predicates = deepcopy(payload["query"])
    literals: dict[tuple[str, ...], bool] = {}
    for *atom, sign in records:
        key = tuple(atom)
        if key in literals and literals[key] != sign:
            return dict(
                decision="insufficient",
                satisfiable_outcomes=[],
                models_examined=0,
                contradictory=True,
            )
        literals[key] = sign
    domain = sorted(
        {str(f[i]) for f in records for i in (0, 2)}
        | {x for q in predicates for x in (q[0], q[2]) if x[:1] != "?"}
    )
    names = sorted({x for q in predicates for x in q if x[:1] == "?"})
    clauses = []
    for assignment in product(domain, repeat=len(names)):
        substitution = dict(zip(names, assignment, strict=True))
        clauses.append([tuple(substitution.get(t, t) for t in q) for q in predicates])
    outcomes: set[bool] = set()
    examined = 0

    def visit(complete: dict[tuple[str, ...], bool]) -> None:
        """Prune an entire completion subcube only when every world has the same outcome."""
        nonlocal examined
        examined += 1
        states = [[complete.get(atom) for atom in clause] for clause in clauses]
        if any(all(x is True for x in state) for state in states):
            outcomes.add(True)
        elif all(any(x is False for x in state) for state in states):
            outcomes.add(False)
        else:
            atom = next(atom for clause in clauses for atom in clause if atom not in complete)
            for bit in (False, True):
                visit(dict([*complete.items(), (atom, bit)]))
                if len(outcomes) == 2:
                    break

    visit(literals)
    return dict(
        decision="support"
        if outcomes == {True}
        else "refute"
        if outcomes == {False}
        else "insufficient",
        satisfiable_outcomes=sorted(outcomes),
        models_examined=examined,
        contradictory=False,
    )


@lru_cache(maxsize=2)
def graph_roster(edge_count: int, target: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Canonical graph classes prevent entity renamings from multiplying the case count."""
    seen: set[tuple[tuple[int, int], ...]] = set()
    for n in (4, 5):
        for edges in combinations(
            [(a, b) for a in range(n) for b in range(n) if a != b], edge_count
        ):
            if sum(0 in edge for edge in edges) < 2 or not any(0 not in edge for edge in edges):
                continue
            reached = {0}
            for _ in range(n):
                reached |= {a for a, b in edges if b in reached} | {
                    b for a, b in edges if a in reached
                }
            if len(reached) != n:
                continue
            key = min(
                tuple(sorted(((0, *p)[a], (0, *p)[b]) for a, b in edges))
                for p in permutations(range(1, n))
            )
            seen.add(key)
            if len(seen) == target:
                return tuple(sorted(seen))
    raise ValueError("insufficient_distinct_graphs")


def construct() -> tuple[list[Json], list[Json]]:
    """Disjoint edge counts and canonical graphs prevent renamed cases crossing splits."""
    inputs, sources = [], []
    for split, count in [("development", 60), ("evaluation", 120)]:
        roster = graph_roster(4 if split == "development" else 5, count)
        for index, edges in enumerate(roster):
            world = f"v751/{split}/{index:03d}"
            nodes = ["?join"] + [world + "/constant/" + str(i) for i in range(1, 5)]
            witness, distractor = world + "/witness", world + "/distractor"
            query = [
                [nodes[a], world + "/rule/" + str(k), nodes[b]] for k, (a, b) in enumerate(edges)
            ]
            facts: list[list[Any]] = [
                [witness if x == "?join" else x, r, witness if y == "?join" else y, True]
                for x, r, y in query
            ]
            ground = next(k for k, (a, b) in enumerate(edges) if a != 0 and b != 0)
            hop = next(k for k, edge in enumerate(edges) if 0 in edge)
            variable_position = 0 if edges[hop][0] == 0 else 2
            structure = canonical_hash(edges)
            for condition in CONDITIONS:
                records = deepcopy(facts)
                if condition == "explicit_refutation":
                    records[ground][3] = False
                elif condition == "missing_required_hop":
                    records.pop(hop)
                elif condition == "disconnected_join":
                    records[hop][variable_position] = distractor
                elif condition == "irrelevant_distractor":
                    records.append([distractor, world + "/irrelevant", nodes[1], True])
                elif condition == "explicit_conflict":
                    records.append([*facts[ground][:3], False])
                for view in ["original", "renamed_ordered"]:
                    predicates = deepcopy(query)
                    viewed = deepcopy(records)
                    if view == "renamed_ordered":
                        rename = {
                            x: world + "/renamed/" + str(i)
                            for i, x in enumerate([distractor, witness, *reversed(nodes[1:])])
                        }
                        viewed = [
                            [rename[x], rel, rename[y], z] for x, rel, y, z in reversed(records)
                        ]
                        predicates = [
                            [rename.get(x, x), rel, rename.get(y, y)]
                            for x, rel, y in reversed(query)
                        ]
                    uid = world + "/" + condition + "/" + view
                    inputs.append(
                        dict(
                            unit_id=uid,
                            world_id=world,
                            split=split,
                            condition=condition,
                            view=view,
                            structure=structure,
                            template=split + "/template/" + structure,
                            input=dict(facts=viewed, query=predicates),
                        )
                    )
                    document, spans = "", []
                    for x, rel, y, sign in viewed:
                        line = f"{x}\t{rel}\t{y}\t{int(sign)}\n"
                        spans.append([len(document), len(document) + len(line), line])
                        document += line
                    sources.append(
                        dict(
                            unit_id=uid,
                            document=document,
                            spans=spans,
                            document_sha256=canonical_hash(document),
                            scope="Constructed signed records; no natural-language extraction.",
                        )
                    )
    return inputs, sources


def freeze() -> tuple[list[Json], list[Json], list[Json]]:
    """Oracle records and labels live separately from the verifier input projection."""
    inputs, sources = construct()
    labels = []
    for index, unit in enumerate(inputs):
        truth_records = deepcopy(unit["input"])
        labels.append(
            dict(
                unit_id=unit["unit_id"],
                truth_records=truth_records,
                **enumerate_truth(truth_records),
            )
        )
        if (index + 1) % 120 == 0:
            print(
                f"[exp8768] phase=oracle_seal completed={index + 1} pending={len(inputs) - index - 1}",
                flush=True,
            )
    return inputs, labels, sources


def disjoint(inputs: list[Json]) -> bool:
    """Audit actual entities, relation vocabularies and registered structures across splits."""
    inventories = {}
    for split in ["development", "evaluation"]:
        units = [p for p in inputs if p["split"] == split]
        inventories[split] = [
            {str(f[i]) for p in units for f in p["input"]["facts"] for i in (0, 2)},
            {p["template"] for p in units},
            {p["structure"] for p in units},
            {str(q[1]) for p in units for q in p["input"]["query"]},
        ]
    return all(
        a and b and not a.intersection(b)
        for a, b in zip(inventories["development"], inventories["evaluation"], strict=True)
    )
