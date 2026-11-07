"""REQ-REPORT-8882: diagnose exposed request costs without claiming new superiority."""

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting import size_aware_complete_cost_8874 as previous
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT = previous.ROOT
NAME, TASK = "experiment_8882_v760_request_cost_attribution", "exp8882-request-cost-attribution"
CLI = f"scripts/experiments/{NAME}.py"
WORKER = "scripts/experiments/request_cost_worker_8882.py"
OWNED = [
    "python/carnot/reporting/request_cost_attribution_8882.py",
    "python/carnot/reporting/request_cost_runner_8882.py",
    WORKER,
    CLI,
]
TESTS = ["tests/python/test_request_cost_attribution_8882.py"]
ARMS, SIZES, CAP, BOUNDARIES = previous.ARMS, previous.SIZES, previous.CAP, previous.BOUNDARIES
SEED, DRAWS = 7608882, 10000
SEMANTICS = (
    "decision",
    "witness",
    "status",
    "missing_reason",
    "primitive_operations",
    "operation_counts",
)
CATEGORIES = ("parse_validate", "preparation", "exact_joins", "response_encode")
baseline_source = previous.baseline_source


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured boundaries so heartbeats never replace the fixed deadline."""
    print(f"[exp8882] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze() -> list[Json]:
    """Select retained exposed worlds before consulting component or observer clocks."""
    try:
        primary = json.loads(
            (ROOT / "results/experiment_8874_v759_size_aware_complete_cost.json").read_bytes()
        )
        work = json.loads(Path(primary["work_reference"]["path"]).read_bytes())
    except (OSError, ValueError, KeyError, TypeError):
        work = dict(worlds=previous.freeze())
    rng, selected = random.Random(SEED), []
    for size in SIZES:
        worlds = [w for w in work["worlds"] if w["size"] == size and w["population"] == "ordinary"][
            :10
        ]
        for world in worlds:
            world["observer_order"] = [False, True]
            rng.shuffle(world["observer_order"])
            rng.shuffle(world["arm_order"])
            selected.append(world)
    return selected


def reduce(rows: list[Json]) -> Json:
    """Average repeats within worlds; retain observer failures and all residuals."""
    pairs: dict[tuple[Any, ...], dict[bool, Json]] = defaultdict(dict)
    components, plain, residuals = [], [], []
    component_worlds: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    groups: dict[tuple[Any, ...], Json] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[k] for k in ("boundary", "unit_id", "arm", "repeat"))
        observer = row["instrumented"]
        if observer in pairs[key] or row["arm"] not in ARMS or row["repeat"] not in range(3):
            raise ValueError("roster_duplicate_or_foreign")
        pairs[key][observer] = row
        groups[(row["size"], row["boundary"], row["arm"])][row["cluster_id"]].append(row)
        if observer:
            paid = (
                sum(row.get("components_ns", {}).values()) / 1e9
                + row.get("startup_import_ns", 0) / 1e9
            )
            elapsed = row["complete_wall_time_s"]
            residual = elapsed - paid
            clocks = all(k in row for k in ("components_ns", "startup_import_ns")) and elapsed > 0
            residuals.append(
                dict(
                    unit_id=row["unit_id"],
                    arm=row["arm"],
                    boundary=row["boundary"],
                    repeat=row["repeat"],
                    residual_s=residual,
                    fraction=residual / elapsed if elapsed else None,
                    clocks_present=clocks,
                    valid=clocks and residual >= -1e-12 and residual <= elapsed * 0.05,
                )
            )
            components.append(
                dict(
                    row,
                    component_seconds={
                        k: row.get("components_ns", {}).get(k, 0) / 1e9 for k in CATEGORIES
                    },
                    process_import_startup_s=row.get("startup_import_ns", 0) / 1e9,
                    unaccounted_residual_s=residual,
                )
            )
            component_worlds[(row["cluster_id"], row["boundary"], row["arm"])].append(
                components[-1]
            )
        else:
            plain.append(row)
    if any(set(p) != {False, True} for p in pairs.values()):
        raise ValueError("roster_pair")
    quality = dict(
        paired_semantics=bool(rows)
        and all(all(p[False][k] == p[True][k] for k in SEMANTICS) for p in pairs.values()),
        no_owned_failures=bool(rows)
        and all(r["status"] in ("completed", "censored") for r in rows),
    )
    intervals, overheads = [], []
    for (size, boundary, arm), worlds in sorted(groups.items()):
        values = []
        for cluster, sample in sorted(worlds.items()):
            if len(sample) != 24 or len({r["unit_id"] for r in sample}) != 4:
                raise ValueError("roster_world")
            values.append(
                [
                    sum(r["complete_wall_time_s"] for r in sample if r["instrumented"] == observer)
                    / 3
                    for observer in (False, True)
                ]
            )
        data = np.array(values)
        if np.any(data <= 0):
            overheads.append(
                dict(
                    size=size,
                    boundary=boundary,
                    arm=arm,
                    accepted=False,
                    upper=None,
                    missing_reason="missing_complete_clock",
                )
            )
            continue
        idx = np.random.default_rng(SEED + size).integers(0, len(data), (DRAWS, len(data)))
        draws = data[idx].mean(axis=1)
        ratios = draws[:, 1] / draws[:, 0] - 1
        lo, hi = np.quantile(ratios, [0.005, 0.995])
        label = dict(
            size=size,
            boundary=boundary,
            arm=arm,
            independent_count=len(data),
            cluster="whole_world",
            confidence=0.99,
            draws=DRAWS,
        )
        overheads.append(
            dict(
                label,
                estimate=float(data[:, 1].mean() / data[:, 0].mean() - 1),
                lower=float(lo),
                upper=float(hi),
                accepted=bool(hi <= 0.05),
            )
        )
        interval: Json = dict(
            label,
            uninstrumented_world_s=float(data[:, 0].mean()),
            instrumented_world_s=float(data[:, 1].mean()),
            components={},
        )
        for category in CATEGORIES + ("process_import_startup", "unaccounted_residual"):
            totals = []
            for cluster in sorted(worlds):
                sample = component_worlds[(cluster, boundary, arm)]
                totals.append(
                    sum(
                        r["component_seconds"].get(category, r.get(category + "_s", 0))
                        for r in sample
                    )
                    / 3
                )
            mean_draws = np.array(totals)[idx].mean(axis=1)
            lower, upper = np.quantile(mean_draws, [0.005, 0.995])
            interval["components"][category] = dict(
                estimate=float(np.mean(totals)), lower=float(lower), upper=float(upper)
            )
        intervals.append(interval)
    gates = dict(
        quality,
        nonnegative_closed_clocks=bool(residuals) and all(r["valid"] for r in residuals),
        observer_upper_at_most_five_percent=bool(overheads)
        and all(r["accepted"] for r in overheads),
    )
    ready = int(all(gates.values()))
    return dict(
        component_rows=components,
        uninstrumented_rows=plain,
        closure_residuals=residuals,
        whole_world_intervals=intervals,
        observer_overhead_interval=overheads,
        quality_gates=quality,
        attribution_gates=gates,
        cost_attribution_ready_score=ready,
        dominant_cost=dict(
            component=None,
            measured_removable=False,
            reason="No qualified removable cost identified."
            if ready
            else "Observer overhead or closure prevents qualified component attribution; raw intervals are diagnostic only.",
        ),
        next_mechanism="No optimization selected; qualify lower-overhead interval observation before selecting a mechanism.",
        frozen_subset_checksum=canonical_hash(freeze()),
    )
