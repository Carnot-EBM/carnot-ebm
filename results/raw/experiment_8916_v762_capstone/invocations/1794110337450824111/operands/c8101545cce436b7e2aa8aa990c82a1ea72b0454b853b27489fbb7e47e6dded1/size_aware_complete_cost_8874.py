"""REQ-REPORT-8874: keep repeated clocks inside independent paired worlds."""

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting import size_aware_verifier_8871 as design
from carnot.reporting.installed_cost_8861 import baseline_source as baseline_source
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8874_v759_size_aware_complete_cost", "exp8874-size-aware-complete-cost"
CLI = f"scripts/experiments/{NAME}.py"
WORKER = "scripts/experiments/size_aware_cost_worker_8874.py"
OWNED = [
    "python/carnot/reporting/size_aware_complete_cost_8874.py",
    "python/carnot/reporting/size_aware_complete_runner_8874.py",
    WORKER,
    CLI,
]
TESTS = ["tests/python/test_size_aware_complete_cost_8874.py"]
SIZES, SEED, CAP, DRAWS = (8, 16, 17, 64, 256, 1024), 7598874, 200000, 10000
ARMS, BOUNDARIES = ("scan", "indexed", "size_aware"), ("cold", "resident")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so a supervisor can enforce its fixed deadline."""
    print(f"[exp8874] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(*, pilot: bool = False) -> list[Json]:
    """Reuse the qualified construction with fresh seeds and a sealed three-arm order."""
    producer = namespace(design, "cost8874_worlds")
    producer.SEED = SEED
    worlds = list(producer.freeze(pilot=pilot))
    rng = random.Random(SEED + int(pilot))
    for world in worlds:
        world["arm_order"] = list(ARMS)
        rng.shuffle(world["arm_order"])
    return worlds


def reduce(rows: list[Json], installation_s: float) -> Json:
    """Average repeats first, then resample paired worlds within each fixed stratum."""
    if not rows:
        return dict(
            per_world_results=[],
            paired_intervals={},
            quality_gates={},
            benefit_gates={},
            size_cost_benefit_score=0,
            cold_benefit_score=0,
        )
    observations: dict[tuple[str, str, str], list[Json]] = defaultdict(list)
    pairs: dict[tuple[str, str, int], Json] = defaultdict(dict)
    seen = set()
    for row in rows:
        key = (row["boundary"], row["unit_id"], row["repeat"], row["arm"])
        if key in seen or row["arm"] not in ARMS or row["repeat"] not in range(3):
            raise ValueError("roster_duplicate_or_foreign")
        seen.add(key)
        observations[(row["boundary"], row["cluster_id"], row["arm"])].append(row)
        pairs[key[:3]][row["arm"]] = row
    if any(set(p) != set(ARMS) for p in pairs.values()):
        raise ValueError("roster_pair")
    quality = dict(
        exact_completed_parity=True, no_coverage_loss=True, no_false_accept_increase=True
    )
    for pair in pairs.values():
        a, b = pair["scan"], pair["size_aware"]
        if a["status"] == b["status"] == "completed":
            quality["exact_completed_parity"] &= (a["decision"], a["witness"]) == (
                b["decision"],
                b["witness"],
            )
        quality["no_coverage_loss"] &= int(b["decision"] != "insufficient") >= int(
            a["decision"] != "insufficient"
        )
        quality["no_false_accept_increase"] &= int(
            b["decision"] == "support" and b["truth"] != "support"
        ) <= int(a["decision"] == "support" and a["truth"] != "support")
    summaries = []
    groups = sorted({(r["boundary"], r["cluster_id"]) for r in rows})
    for boundary, cluster in groups:
        arms = {}
        for arm in ARMS:
            sample = observations[(boundary, cluster, arm)]
            if len(sample) != 12 or len({r["unit_id"] for r in sample}) != 4:
                raise ValueError("roster_world")
            arms[arm] = dict(
                wall=sum(r["complete_wall_time_s"] for r in sample) / 3 + installation_s / 180,
                operations=sum(r["primitive_operations"] for r in sample) / 3,
                coverage=sum(
                    r["status"] == "completed" and r["decision"] != "insufficient" for r in sample
                )
                / 12,
                false_accept=sum(
                    r["decision"] == "support" and r["truth"] != "support" for r in sample
                )
                / 12,
            )
        first = observations[(boundary, cluster, "scan")][0]
        summaries.append(
            dict(
                boundary=boundary,
                cluster_id=cluster,
                size=first["size"],
                population=first["population"],
                arms=arms,
            )
        )
    intervals: Json = {}
    alpha = 0.01 / (2 * 160)
    for boundary in BOUNDARIES:
        strata: Json = {}
        for size, population in [(s, "ordinary") for s in SIZES] + [(64, "stress")]:
            worlds = [
                s
                for s in summaries
                if s["boundary"] == boundary and s["size"] == size and s["population"] == population
            ]
            if not worlds:
                continue
            indices = np.random.default_rng(SEED + size + int(population == "stress")).integers(
                0, len(worlds), (DRAWS, len(worlds))
            )
            values = np.array(
                [
                    [
                        [
                            w["arms"][a][k]
                            for k in ("wall", "operations", "coverage", "false_accept")
                        ]
                        for a in ARMS
                    ]
                    for w in worlds
                ]
            )
            strata[str(size) if population == "ordinary" else "stress64"] = (
                values.mean(axis=0),
                values[indices].mean(axis=1),
            )
        regular = [str(s) for s in SIZES if str(s) in strata]
        if regular:
            strata["mixture"] = tuple(
                np.mean([strata[s][i] for s in regular], axis=0) for i in (0, 1)
            )
        intervals[boundary] = {a: {} for a in ARMS[1:]}
        for name, (means, draws) in strata.items():
            for i, arm in enumerate(ARMS[1:], 1):
                endpoints: Json = {}
                for endpoint, column, kind in [
                    ("speed_ratio", 0, "ratio"),
                    ("latency_regression", 0, "regression"),
                    ("operation_ratio", 1, "ratio"),
                    ("coverage_loss", 2, "loss"),
                    ("false_accept_increase", 3, "increase"),
                ]:
                    a, b = means[0, column], means[i, column]
                    av, bv = draws[:, 0, column], draws[:, i, column]
                    if (kind in ("ratio", "regression")) and (a <= 0 or b <= 0):
                        endpoints[endpoint] = dict(
                            estimate=None, lower=None, upper=None, missing_reason="no_paid_work"
                        )
                        continue
                    estimate, values = (
                        (a / b, av / bv)
                        if kind == "ratio"
                        else (b / a - 1, bv / av - 1)
                        if kind == "regression"
                        else (a - b, av - bv)
                        if kind == "loss"
                        else (b - a, bv - av)
                    )
                    lo, hi = np.quantile(values, [alpha, 1 - alpha])
                    endpoints[endpoint] = dict(
                        estimate=float(estimate),
                        lower=float(lo),
                        upper=float(hi),
                        familywise_confidence=0.99,
                        simultaneous_endpoint_count=160,
                        draws=DRAWS,
                        cluster="whole_world",
                        independent_count=len(
                            [
                                s
                                for s in summaries
                                if s["boundary"] == boundary
                                and (
                                    s["population"] == "ordinary"
                                    if name == "mixture"
                                    else s["population"] == "stress"
                                    if name == "stress64"
                                    else s["population"] == "ordinary" and s["size"] == int(name)
                                )
                            ]
                        ),
                    )
                intervals[boundary][arm][name] = endpoints
    gates = {}
    for boundary in BOUNDARIES:
        endpoints = intervals[boundary]["size_aware"]
        gates[boundary] = dict(
            quality,
            complete_size_mixture=all(str(s) in endpoints for s in SIZES),
            mixture_speed_lower_above_one=(
                endpoints.get("mixture", {}).get("speed_ratio", {}).get("lower") or 0
            )
            > 1,
            **{
                f"size{s}_latency_regression_upper_at_most_five_percent": (
                    upper := endpoints.get(str(s), {}).get("latency_regression", {}).get("upper")
                )
                is not None
                and upper <= 0.05
                for s in (8, 16)
            },
        )
    return dict(
        per_world_results=summaries,
        paired_intervals=intervals,
        quality_gates=quality,
        benefit_gates=gates,
        size_cost_benefit_score=int(all(gates["resident"].values())),
        cold_benefit_score=int(all(gates["cold"].values())),
    )
