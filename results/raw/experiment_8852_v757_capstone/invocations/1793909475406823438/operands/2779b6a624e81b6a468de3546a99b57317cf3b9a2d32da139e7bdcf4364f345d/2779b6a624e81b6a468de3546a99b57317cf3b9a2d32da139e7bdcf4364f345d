"""REQ-REPORT-8816: private worlds separate protocol readiness from scientific benefit."""

from copy import deepcopy
from functools import lru_cache
from itertools import permutations
from pathlib import Path
import random
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_intervention_methods_8791 import strong
from carnot.reporting.evidence_set_corpus_8768 import enumerate_truth

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8816_v755_selective_evidence_methods"
TASK = "exp8816-selective-evidence-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/selective_{s}_8816.py" for s in ["evidence", "protocol", "runner"]
] + [CLI]
TESTS = ["tests/python/test_selective_evidence_8816.py"]
SEED = 7558816
LABELS = ("support", "refute", "insufficient")
COUNTS = dict(fit=60, calibration=60, evaluation=240, canary=12)
HISTORY = {
    "results/experiment_8805_v754_evidence_intervention_study.json": "sha256:988fff3e89321f67fac853b7a544460a99c23abd9f7c274c37d59411804fe50f",
    "results/experiment_8806_v754_consensus_challenge_energy.json": "sha256:2e7328b13d9be694e25d6f5c7e0ede74ef7412809956bb7a0b9787b34816eacc",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts let supervisors detect work without extending a measured deadline."""
    print(f"[exp8816] phase={phase} completed={completed} pending={pending}", flush=True)


@lru_cache(maxsize=800)
def _canonical(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Erase all six vertex names, including variable roles, before assigning a split."""
    return min(tuple(sorted((p[a], p[b]) for a, b in edges)) for p in permutations(range(6)))


def topology(edges: list[list[int]]) -> str:
    """Unlabeled topology is stricter than preserving variable colors or relation names."""
    return canonical_hash(_canonical(tuple(sorted(tuple(e) for e in edges))))


def truth(payload: Json) -> str:
    """Independent completion and relational evaluators must agree before labels seal."""
    if set(payload) != {"facts", "query"}:
        raise ValueError("input_fields")
    result = str(enumerate_truth(payload)["decision"])
    if result != strong(payload):
        raise ValueError("independent_evaluator_disagreement")
    return result


def disjoint(worlds: list[Json]) -> bool:
    """Topology reuse remains contamination even if another query uses different variables."""
    return len(worlds) == len({topology(w["edges"]) for w in worlds}) and all(
        w["topology_hash"] == topology(w["edges"]) for w in worlds
    )


def variants(edges: list[list[int]], variables: int) -> list[Json]:
    """Ground witnesses balance labels without using missing facts as signed negatives."""
    names = ["?x", "?y" if variables == 2 else "b", "c", "d", "e", "f"]
    query = [[names[a], f"r{i}", names[b]] for i, (a, b) in enumerate(edges)]
    base = [["u" if t == "?x" else "v" if t == "?y" else t for t in q] + [True] for q in query]
    ground = next(i for i, q in enumerate(query) if all(not t.startswith("?") for t in q))
    result = []
    for condition in ["support", "padding", "false", "false_reordered", "missing", "conflict"]:
        facts = deepcopy(base)
        if condition == "padding":
            facts.append(["noise_a", "unused", "noise_b", True])
        if condition.startswith("false"):
            facts[ground][3] = False
        if condition == "false_reordered":
            facts.reverse()
        if condition == "missing":
            facts.pop(ground)
        if condition == "conflict":
            facts.append([*base[ground][:3], False])
        result.append(dict(condition=condition, input=dict(facts=facts, query=query)))
    return result


def freeze(limit: int = 10000) -> Json:
    """Generate 372 non-isomorphic worlds; exhaustion reports actual support without aliases."""
    rng = random.Random(SEED)
    seen: set[str] = set()
    worlds: list[Json] = []
    inputs: list[Json] = []
    labels: list[Json] = []
    roster = [(split, i) for split, count in COUNTS.items() for i in range(count)]
    universe = [(a, b) for a in range(6) for b in range(6) if a != b]
    ring = [(i, (i + 1) % 6) for i in range(6)]
    attempts = 0
    while len(worlds) < len(roster) and attempts < limit:
        attempts += 1
        extras = rng.sample([e for e in universe if e not in ring], rng.randint(2, 6))
        edges = [list(e) for e in sorted(ring + extras)]
        structure = topology(edges)
        if structure in seen:
            continue
        seen.add(structure)
        split, index = roster[len(worlds)]
        wid = f"v755/{split}/{index:03d}"
        variables = 1 + index % 2
        worlds.append(
            dict(
                world_id=wid,
                split=split,
                edges=edges,
                variables=variables,
                topology_hash=structure,
                query_hash=canonical_hash(variants(edges, variables)[0]["input"]["query"]),
            )
        )
        for j, row in enumerate(variants(edges, variables)):
            uid = f"{wid}/{j}"
            inputs.append(
                dict(
                    unit_id=uid,
                    cluster_id=wid,
                    split=split,
                    input=row["input"],
                    candidates=[[label] * 3 for label in LABELS],
                )
            )
            labels.append(
                dict(
                    unit_id=uid,
                    cluster_id=wid,
                    split=split,
                    condition=row["condition"],
                    truth=truth(row["input"]),
                )
            )
        if len(worlds) % 30 == 0:
            progress("seal_worlds", len(worlds), len(roster) - len(worlds))
    return dict(
        counts=COUNTS,
        worlds=worlds,
        inputs=inputs,
        labels=labels,
        status="sealed" if len(worlds) == 372 else "blocked",
        intended_count=372,
        completed_count=len(worlds),
        attempts=attempts,
        structure_disjointness_rows=[
            dict(
                check="unlabeled_topology_unique", passed=disjoint(worlds), denominator=len(worlds)
            ),
            dict(
                check="V751_V754_vertex_count_disjoint",
                passed=all(len({v for e in w["edges"] for v in e}) == 6 for w in worlds),
                denominator=len(worlds),
                historical_max_vertices=5,
            ),
        ],
    )


def release(cohort: Json, predictions: list[Json], seal: str) -> list[Json]:
    """Only a complete input-only prediction commitment authorizes label release."""
    expected = {x["unit_id"] for x in cohort["inputs"]}
    if (
        canonical_hash(predictions) != seal
        or len(predictions) != len(expected)
        or {x.get("unit_id") for x in predictions} != expected
        or any(
            set(x) != {"unit_id", "decision"}
            or x["decision"] not in {"accept", "reject", "escalate"}
            for x in predictions
        )
    ):
        raise ValueError("prediction_seal")
    return list(cohort["labels"])


def controls(cohort: Json) -> list[Json]:
    """Mutations exercise disclosure, signed absence, nuisance stability and split reuse."""
    payloads = variants(cohort["worlds"][0]["edges"], 2)
    outcomes = [truth(x["input"]) for x in payloads]
    reordered = deepcopy(payloads[0]["input"])
    reordered["facts"].reverse()
    reordered["query"].reverse()
    bad = deepcopy(cohort["worlds"])
    bad.append(bad[0])
    exposed_rejected = False
    try:
        truth(dict(payloads[0]["input"], truth="support"))
    except ValueError:
        exposed_rejected = True
    return [
        dict(control="label_exposure", passed=exposed_rejected),
        dict(
            control="missing_versus_false",
            passed=outcomes[4] == "insufficient" and outcomes[2] == "refute",
        ),
        dict(
            control="padding_order",
            passed=outcomes[0] == outcomes[1] == truth(reordered) == "support"
            and outcomes[2] == outcomes[3],
        ),
        dict(control="split_contamination", passed=not disjoint(bad)),
        dict(control="conflict_requires_abstention", passed=outcomes[5] == "insufficient"),
    ]
