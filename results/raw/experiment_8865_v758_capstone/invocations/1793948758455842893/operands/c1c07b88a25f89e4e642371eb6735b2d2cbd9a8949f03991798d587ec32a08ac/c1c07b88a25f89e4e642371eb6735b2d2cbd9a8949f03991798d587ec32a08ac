"""REQ-REPORT-8861: freeze construction labels and reduce costs by independent world."""

import ast
from collections import defaultdict
from copy import deepcopy
import inspect
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting import evidence_operation_meter_8818 as instrumentation
from carnot.reporting.evidence_intervention_methods_8791 import strong

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8861_v758_installed_index_cost", "exp8861-installed-index-cost"
CLI = f"scripts/experiments/{NAME}.py"
WORKER = "scripts/experiments/installed_cost_worker_8861.py"
OWNED = [
    "python/carnot/reporting/installed_cost_8861.py",
    "python/carnot/reporting/installed_cost_runner_8861.py",
    WORKER,
    CLI,
]
TESTS = ["tests/python/test_installed_index_cost_8861.py"]
ARMS, SIZES = ("always_joint", "indexed_witness"), (16, 64, 256, 1024)
SEED, CAP, DRAWS = 7588861, 200000, 10000


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work boundaries so a supervisor can distinguish progress from a stall."""
    print(f"[exp8861] phase={phase} completed={completed} pending={pending}", flush=True)


def baseline_source() -> str:
    """Keep the historical enumerator and expose its found binding without another search."""
    tree = instrumentation.Instrument().visit(ast.parse(inspect.getsource(strong)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            binding = (
                ast.Name(id="binding", ctx=ast.Load())
                if isinstance(node.value, ast.Constant) and node.value.value == "support"
                else ast.Constant(None)
            )
            node.value = ast.Tuple(elts=[node.value, binding], ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def freeze() -> list[Json]:
    """Construct independent signed worlds; labels follow explicit interventions, never an arm."""
    rng, worlds = random.Random(SEED), []
    for size, count, population in [(s, 30, "deployment") for s in SIZES] + [
        (1024, 40, "adversarial")
    ]:
        for index in range(count):
            cluster = f"{population}/{size}/{index:03}"
            prefix = f"{rng.getrandbits(24):06x}"
            x = prefix + "m"
            query = [["?x", "r0", prefix + "a"], ["?x", "r1", prefix + "b"]]
            query += [
                [("?x" if rng.randrange(2) else prefix + str(i)), f"r{i}", prefix + chr(99 + i)]
                for i in range(2, 7)
            ]
            query += [[prefix + "g", "r7", prefix + "h"]]
            if population == "adversarial":
                query = [[f"?v{i}", f"r{i}", prefix + str(i)] for i in range(7)] + [query[7]]
                facts = [
                    [prefix + f"{j:03}", f"r{i}", prefix + str(i), True]
                    for i in range(7)
                    for j in range(127)
                ]
                facts.append([*query[7], True])
            else:
                facts = [[*(x if t == "?x" else t for t in q), True] for q in query]
            requests = []
            for condition in [
                "support",
                "refute",
                "missing",
                "conflict" if index % 2 else "binding",
            ]:
                records = deepcopy(facts)
                target = len(records) - 1
                if condition == "refute":
                    records[target][3] = False
                elif condition == "missing":
                    records.pop(target)
                elif condition == "conflict":
                    records.append([*records[0][:3], False])
                elif condition == "binding":
                    if population == "adversarial":
                        records.pop(target)
                    else:
                        records[1][0] = prefix + "other"
                while len(records) < size:
                    n = len(records)
                    records.append(
                        [f"{rng.getrandbits(32):08x}", "noise", f"{n:x}", bool(rng.randrange(2))]
                    )
                payload = dict(facts=records, query=query)
                requests.append(
                    dict(
                        unit_id=cluster + "/" + condition,
                        condition=condition,
                        input=payload,
                        input_sha256=canonical_hash(payload),
                        truth="support"
                        if condition == "support"
                        else "refute"
                        if condition == "refute"
                        else "insufficient",
                    )
                )
            order = list(ARMS)
            rng.shuffle(order)
            worlds.append(
                dict(
                    cluster_id=cluster,
                    world_identity=canonical_hash(requests),
                    size=size,
                    population=population,
                    arm_order=order,
                    requests=requests,
                )
            )
    return worlds


def reduce(rows: list[Json], clocks: list[Json], installation_s: float) -> Json:
    """Resample paired worlds; all requests including censors remain in the denominators."""
    if not rows:
        return dict(
            per_world_results=[],
            paired_intervals={},
            quality_gates={},
            benefit_gates={},
            installed_cost_benefit_score=0,
        )
    pairs: dict[tuple[str, str], Json] = defaultdict(dict)
    groups: dict[tuple[str, str], dict[str, list[Json]]] = defaultdict(lambda: defaultdict(list))
    clock_map = {(c["boundary"], c["cluster_id"], c["arm"]): c for c in clocks}
    for row in rows:
        pair = pairs[(row["boundary"], row["unit_id"])]
        if row["arm"] in pair:
            raise ValueError("roster_duplicate")
        pair[row["arm"]] = row
        groups[(row["boundary"], row["cluster_id"])][row["arm"]].append(row)
    if any(set(p) != set(ARMS) for p in pairs.values()) or len(clock_map) != len(clocks):
        raise ValueError("roster_pair")
    quality = dict(
        exact_completed_parity=True, no_coverage_loss=True, no_false_accept_increase=True
    )
    summaries = []
    for (boundary, cluster), arms in sorted(groups.items()):
        if any(len(arms[a]) != 4 for a in ARMS):
            raise ValueError("roster_world")
        data: Json = {}
        for arm in ARMS:
            sample = arms[arm]
            clock = clock_map[(boundary, cluster, arm)]
            if any(r["world_wall_time_s"] != clock["wall_time_s"] for r in sample):
                raise ValueError("parent_clock_drift")
            data[arm] = dict(
                wall=sample[0]["world_wall_time_s"] + installation_s / 120,
                operations=sum(r["primitive_operations"] for r in sample),
                coverage=sum(
                    r["status"] == "completed" and r["decision"] != "insufficient" for r in sample
                )
                / 4,
                false_accept=sum(
                    r["decision"] == "support" and r["truth"] != "support" for r in sample
                )
                / 4,
                denominator=4,
            )
        quality["no_coverage_loss"] &= data[ARMS[1]]["coverage"] >= data[ARMS[0]]["coverage"]
        quality["no_false_accept_increase"] &= (
            data[ARMS[1]]["false_accept"] <= data[ARMS[0]]["false_accept"]
        )
        first = arms[ARMS[0]][0]
        summaries.append(
            dict(
                cluster_id=cluster,
                boundary=boundary,
                size=first["size"],
                population=first["population"],
                arms=data,
            )
        )
    for pair in pairs.values():
        a, b = (pair[arm] for arm in ARMS)
        if a["status"] == b["status"] == "completed":
            quality["exact_completed_parity"] &= (a["decision"], a["witness"]) == (
                b["decision"],
                b["witness"],
            )
    intervals: Json = {}
    alpha = 0.01 / (2 * 2 * 5 * 4)
    for boundary in ("cold", "resident_inner"):
        regular = [
            s for s in summaries if s["boundary"] == boundary and s["population"] == "deployment"
        ]
        intervals[boundary] = {}
        sampled: dict[int, Json] = {}
        for size in SIZES:
            worlds = [s for s in regular if s["size"] == size]
            if not worlds:
                continue
            indices = np.random.default_rng(SEED + size).integers(
                0, len(worlds), (DRAWS, len(worlds))
            )
            sampled[size] = {
                key: np.array([[w["arms"][arm][key] for arm in ARMS] for w in worlds])
                for key in ("wall", "operations", "coverage", "false_accept")
            }
            sampled[size]["draws"] = {
                key: values[indices].mean(axis=1)
                for key, values in sampled[size].items()
                if key != "draws"
            }
        for name, sizes in [(str(s), [s]) for s in sampled] + [("mixture", list(sampled))]:
            endpoints: Json = {}
            for key, endpoint in [
                ("wall", "speed_ratio"),
                ("operations", "operation_ratio"),
                ("coverage", "coverage_loss"),
                ("false_accept", "false_accept_increase"),
            ]:
                means = np.mean([sampled[s][key].mean(axis=0) for s in sizes], axis=0)
                draws = np.mean([sampled[s]["draws"][key] for s in sizes], axis=0)
                if key in ("wall", "operations") and (
                    means[1] <= 0 or bool((draws[:, 1] <= 0).any())
                ):
                    endpoints[endpoint] = dict(
                        estimate=None,
                        lower=None,
                        upper=None,
                        missing_reason="no_paid_completed_work",
                    )
                    continue
                estimate = (
                    means[0] / means[1]
                    if key in ("wall", "operations")
                    else means[0] - means[1]
                    if key == "coverage"
                    else means[1] - means[0]
                )
                values = (
                    draws[:, 0] / draws[:, 1]
                    if key in ("wall", "operations")
                    else draws[:, 0] - draws[:, 1]
                    if key == "coverage"
                    else draws[:, 1] - draws[:, 0]
                )
                lower, upper = np.quantile(values, [alpha, 1 - alpha])
                endpoints[endpoint] = dict(
                    estimate=float(estimate),
                    lower=float(lower),
                    upper=float(upper),
                    familywise_confidence=0.99,
                    draws=DRAWS,
                    cluster="whole_world",
                    mixture_weights="equal_per_size",
                    simultaneous_endpoint_count=40,
                )
            intervals[boundary][name] = endpoints
    cold = intervals["cold"]
    regression = 1 / cold["16"]["speed_ratio"]["lower"] - 1 if "16" in cold else None
    gates: Json = dict(
        quality,
        complete_size_mixture=all(str(s) in cold for s in SIZES),
        mixture_speed_lower_above_one=cold["mixture"]["speed_ratio"]["lower"] > 1,
        size16_latency_regression_upper_at_most_five_percent=regression is not None
        and regression <= 0.05,
    )
    gates["size16_latency_regression"] = dict(upper=regression, threshold=0.05)
    benefit = (
        all(quality.values())
        and gates["complete_size_mixture"]
        and gates["mixture_speed_lower_above_one"]
        and gates["size16_latency_regression_upper_at_most_five_percent"]
    )
    return dict(
        per_world_results=summaries,
        paired_intervals=intervals,
        quality_gates=quality,
        benefit_gates=gates,
        installed_cost_benefit_score=int(benefit),
    )
