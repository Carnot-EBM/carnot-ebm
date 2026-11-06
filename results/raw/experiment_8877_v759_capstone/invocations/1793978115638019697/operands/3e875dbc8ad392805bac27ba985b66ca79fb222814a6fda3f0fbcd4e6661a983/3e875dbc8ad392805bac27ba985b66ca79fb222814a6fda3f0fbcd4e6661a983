"""REQ-REPORT-8847: compare complete paid requests without changing either exact solver."""

from collections import defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from carnot.reporting import indexed_witness_8845 as witness
from carnot.reporting.current_work_receipt import canonical_hash

__all__ = ["witness"]

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8847_v757_indexed_cost", "exp8847-indexed-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/indexed_cost_8847.py",
    "python/carnot/reporting/indexed_cost_runner_8847.py",
    CLI,
]
TESTS = ["tests/python/test_indexed_cost_8847.py"]
ARMS = ("always_joint", "indexed_witness")
SEED, CAP, DRAWS = 7578847, 200000, 10000
SCHEDULES = (11, 13, 17, 19, 23)
WIRE_LENGTHS = {
    d: len(json.dumps(dict(decision=d), sort_keys=True).encode())
    for d in ("support", "refute", "insufficient")
}
EXTRA = ("request_parse_bytes", "request_dispatch", "request_serialization", "request_disposal")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so expensive work remains observable without extending deadlines."""
    print(f"[exp8847] phase={phase} completed={completed} pending={pending}", flush=True)


def stress() -> list[Json]:
    """Construct balanced counterexamples with exactly the registered fact counts.

    The alternate binding supplies a complete witness after the original binding
    breaks. These public synthetic cases establish no language generalization.
    """
    units = []
    for size in (16, 32, 64):
        for world in range(40):
            cluster = f"stress/{size}/{world:03}"
            for condition in ("missing_hop", "contradiction", "binding_swap", "valid_alternative"):
                facts = [["a", "r", "b", True], ["b", "s", "c", True]]
                if condition == "missing_hop":
                    facts.pop()
                elif condition == "contradiction":
                    facts.append(["a", "r", "b", False])
                elif condition == "binding_swap":
                    facts[1][0] = "d"
                else:
                    facts[1][0] = "d"
                    facts.extend([["a", "r", "e", True], ["e", "s", "c", True]])
                while len(facts) < size:
                    facts.append([f"p{world}_{len(facts)}", "padding", "z", True])
                payload = dict(facts=facts, query=[["a", "r", "?x"], ["?x", "s", "c"]])
                units.append(
                    dict(
                        unit_id=cluster + "/" + condition,
                        cluster_id=cluster,
                        split="stress",
                        size=size,
                        condition=condition,
                        input=payload,
                        input_sha256=canonical_hash(payload),
                    )
                )
    return units


def schedule(units: list[Json], seed: int) -> list[tuple[Json, tuple[str, ...]]]:
    """Fixed paired order measures scheduling noise without creating new independent worlds."""
    rng = random.Random(seed)
    ordered = list(units)
    rng.shuffle(ordered)
    return [(u, ARMS if rng.randrange(2) else tuple(reversed(ARMS))) for u in ordered]


def attempt(encoded: str, arm: str, cap: int = CAP) -> Json:
    """Pay transport, dispatch and cleanup around the unchanged admitted request paths.

    Cleanup is reserved before decoding so an exhausted request cannot leave
    uncharged input disposal. Algorithm allocations retain their original meter.
    """
    if arm not in ARMS:
        raise ValueError("arm")
    counter = witness.meter.Counter(cap)
    counter.counts.update(dict.fromkeys((*witness.EXTRA, *EXTRA), 0))
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    result: Json = dict(decision="insufficient", budget_exhausted=False)
    status, reason = "completed", None
    payload, serialized = None, None
    try:
        for _ in encoded.encode():
            counter.charge("request_parse_bytes")
            counter.charge("request_disposal")
        payload = json.loads(encoded)
        counter.charge("request_dispatch")
        result = witness.attempt(payload, indexed=arm == ARMS[1], cap=cap - counter.total)
        counter.total += result["primitive_operations"]
        for kind, count in result["operation_counts"].items():
            counter.counts[kind] += count
        if result["budget_exhausted"]:
            raise witness.meter.Exhausted
        for _ in range(WIRE_LENGTHS[result["decision"]]):
            counter.charge("request_serialization")
            counter.charge("request_disposal")
        serialized = json.dumps(dict(decision=result["decision"]), sort_keys=True)
    except witness.meter.Exhausted:
        result["decision"], result["budget_exhausted"] = "insufficient", True
        status, reason = "censored", "operation_budget_exhausted"
    except (ValueError, KeyError, TypeError) as error:
        result["decision"] = "insufficient"
        status, reason = "failed", "invalid_request:" + str(error)
    del payload, serialized
    return dict(
        decision=result["decision"],
        budget_exhausted=result["budget_exhausted"],
        primitive_limit=cap,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        status=status,
        missing_reason=reason,
        wall_time_s=(time.monotonic_ns() - start) / 1e9,
        cpu_time_s=(time.process_time_ns() - cpu) / 1e9,
    )


def measure(
    units: list[Json], seed_only: int | None = None, arm_only: str | None = None
) -> list[Json]:
    """Seal all predictions before opening labels; every scheduling attempt is retained."""
    rows = []
    start = last = time.monotonic()
    witness.reference.evaluator()
    witness.parse()
    total = (
        len(units)
        * (1 if seed_only is not None else len(SCHEDULES))
        * (1 if arm_only is not None else 2)
    )
    progress("before_benchmark", 0, total)
    for seed in (seed_only,) if seed_only is not None else SCHEDULES:
        for unit, arms in schedule(units, seed):
            if time.monotonic() - start > 300:
                raise ValueError("measurement_deadline")
            encoded = json.dumps(unit["input"], sort_keys=True)
            for arm in arms:
                if arm_only is not None and arm != arm_only:
                    continue
                row = attempt(encoded, arm)
                rows.append(
                    dict(
                        row,
                        arm=arm,
                        seed=seed,
                        **{
                            k: unit[k]
                            for k in ("unit_id", "cluster_id", "split", "condition", "input_sha256")
                        },
                        size=unit.get("size"),
                        metric=int(row["decision"] != "insufficient"),
                        denominator=1,
                    )
                )
            if time.monotonic() - last >= 30:
                progress("benchmark", len(rows), total - len(rows))
                last = time.monotonic()
    progress("after_benchmark", len(rows), 0)
    return rows


def reduce(rows: list[Json]) -> Json:
    """Resample paired whole worlds after averaging scheduling repeats, including all censors."""
    if not rows:
        return dict(
            per_world_results=[],
            paired_intervals={},
            wall_speed_ratio={},
            completed_decision_parity=True,
            indexed_cost_benefit_score=0,
            acceptance_gates={},
        )
    pairs: dict[tuple[str, int], Json] = defaultdict(dict)
    worlds: dict[str, dict[str, list[Json]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair_key = (row["unit_id"], row["seed"])
        if row["arm"] in pairs[pair_key]:
            raise ValueError("roster_duplicate")
        pairs[pair_key][row["arm"]] = row
        worlds[row["cluster_id"]][row["arm"]].append(row)
    if any(set(p) != set(ARMS) for p in pairs.values()):
        raise ValueError("roster_pair")
    parity = all(
        p[ARMS[0]]["decision"] == p[ARMS[1]]["decision"]
        for p in pairs.values()
        if all(r["status"] == "completed" for r in p.values())
    )
    summaries: list[Json] = []
    for cluster, arms in sorted(worlds.items()):
        aggregates: Json = {}
        for arm, sample in arms.items():
            complete = [
                r
                for r in sample
                if all(
                    v["status"] == "completed" for v in pairs[(r["unit_id"], r["seed"])].values()
                )
            ]
            aggregates[arm] = dict(
                operations=sum(r["primitive_operations"] for r in sample) / len(sample),
                coverage=sum(r["decision"] != "insufficient" for r in sample) / len(sample),
                false_accept=sum(
                    r["decision"] == "support" and r["truth"] != "support" for r in sample
                )
                / len(sample),
                complete_wall=sum(r["wall_time_s"] for r in complete),
                complete_count=len(complete),
                denominator=len(sample),
            )
        summaries.append(dict(cluster_id=cluster, arms=aggregates))
    indices = np.random.default_rng(SEED).integers(0, len(summaries), (DRAWS, len(summaries)))
    intervals: Json = {}
    for name, key in [
        ("paid_operation_reduction", "operations"),
        ("coverage_loss", "coverage"),
        ("semantic_false_accept_increase", "false_accept"),
    ]:
        a = np.array([w["arms"][ARMS[0]][key] for w in summaries])
        b = np.array([w["arms"][ARMS[1]][key] for w in summaries])
        if key == "operations":
            draws, estimate = (
                1 - b[indices].mean(axis=1) / a[indices].mean(axis=1),
                1 - b.mean() / a.mean(),
            )
        else:
            delta = b - a if key == "false_accept" else a - b
            draws, estimate = delta[indices].mean(axis=1), delta.mean()
        low, high = np.quantile(draws, [0.01 / 6, 1 - 0.01 / 6])
        intervals[name] = dict(
            estimate=float(estimate),
            lower=float(low),
            upper=float(high),
            familywise_confidence=0.99,
            draws=DRAWS,
            cluster="whole_world",
            iid_certificate=False,
        )
    a = np.array([w["arms"][ARMS[0]]["complete_wall"] for w in summaries])
    b = np.array([w["arms"][ARMS[1]]["complete_wall"] for w in summaries])
    valid = b[indices].sum(axis=1) > 0
    speed: Json = dict(
        estimate=None,
        lower=None,
        upper=None,
        confidence=0.99,
        paired_complete_count=sum(w["arms"][ARMS[0]]["complete_count"] for w in summaries),
        claim=False,
    )
    if b.sum() > 0 and bool(valid.all()):
        low, high = np.quantile(a[indices].sum(axis=1) / b[indices].sum(axis=1), [0.005, 0.995])
        speed.update(
            estimate=float(a.sum() / b.sum()),
            lower=float(low),
            upper=float(high),
            claim=bool(low > 1),
        )
    gates: Json = dict(
        paid_operation_reduction=dict(
            passed=intervals["paid_operation_reduction"]["lower"] >= 0.20,
            threshold=0.20,
            bound="lower",
        ),
        coverage_loss=dict(
            passed=intervals["coverage_loss"]["upper"] <= 0.01, threshold=0.01, bound="upper"
        ),
        semantic_false_accept_increase=dict(
            passed=intervals["semantic_false_accept_increase"]["upper"] <= 0.01,
            threshold=0.01,
            bound="upper",
        ),
        completed_decision_parity=dict(passed=parity),
    )
    return dict(
        per_world_results=summaries,
        paired_intervals=intervals,
        wall_speed_ratio=speed,
        completed_decision_parity=parity,
        acceptance_gates=gates,
        indexed_cost_benefit_score=int(all(g["passed"] for g in gates.values())),
    )


def score(rows: list[Json], units: list[Json]) -> list[Json]:
    """Open encoded oracle labels only after predictions are durably sealed by the caller."""
    labels: dict[str, str] = {}
    last = time.monotonic()
    for unit in units:
        labels[unit["unit_id"]] = witness.reference.base.truth(unit["input"])
        now = time.monotonic()
        if now - last >= 30:
            progress("independent_labels", len(labels), len(units) - len(labels))
            last = now
    return [dict(r, truth=labels[r["unit_id"]]) for r in rows]
