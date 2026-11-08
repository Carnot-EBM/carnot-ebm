"""REQ-REPORT-8894: partition buffered endpoints and freeze future clock tests."""

from collections import defaultdict
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting import size_aware_complete_cost_8874 as previous
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8894_v761_buffered_clock_methods", "exp8894-buffered-clock-methods"
CLI, WORKER = f"scripts/experiments/{NAME}.py", "scripts/experiments/buffered_clock_worker_8894.py"
OWNED = [
    "python/carnot/reporting/buffered_clock_methods_8894.py",
    "python/carnot/reporting/buffered_clock_runner_8894.py",
    WORKER,
    CLI,
]
TESTS = ["tests/python/test_buffered_clock_methods_8894.py"]
ARMS, SIZES, BOUNDARIES, CAP = previous.ARMS, previous.SIZES, previous.BOUNDARIES, previous.CAP
SEED, DRAWS = 7618894, 10000
PHASES = ("unclassified", "decode_validation", "construction", "lookup", "serialization")
SEMANTICS = (
    "decision",
    "witness",
    "status",
    "missing_reason",
    "primitive_operations",
    "operation_counts",
)
baseline_source = previous.baseline_source


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries between timed blocks; never print inside a timed request."""
    print(f"[exp8894] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(*, pilot: bool = False) -> list[Json]:
    """Use distinct seeds for development and the unexecuted future world roster."""
    producer = namespace(previous, "buffered8894_worlds")
    producer.SEED = SEED + int(pilot)
    worlds = producer.freeze()
    rng, selected = random.Random(SEED + int(pilot)), []
    for size in SIZES:
        for world in [w for w in worlds if w["size"] == size and w["population"] == "ordinary"][
            : 4 if pilot else 24
        ]:
            prefix = "development8894/" if pilot else "future8899/"
            world["cluster_id"] = prefix + world["cluster_id"]
            for unit in world["requests"]:
                unit["unit_id"] = prefix + unit["unit_id"]
            world["block_order"] = (
                [False, True, True, False] if rng.randrange(2) else [True, False, False, True]
            )
            selected.append(world)
    return selected


def repetitions(resolution_ns: int, sample_ns: list[int]) -> int:
    """Resolve at least1000 clock ticks with one fixed count, without using overhead."""
    if resolution_ns <= 0 or not sample_ns or min(sample_ns) <= 0:
        raise ValueError("resolution_or_sample")
    return min(1024, max(1, math.ceil(1000 * resolution_ns / min(sample_ns))))


def partition(events: list[list[int]], start: int, end: int, *, endpoint_count: int) -> Json:
    """Reduce one flat timeline and retain signed residuals when endpoints exceed it."""
    totals = dict.fromkeys(PHASES, 0)
    errors, opened, closed = [], None, []
    if any(len(e) != 3 or any(type(x) is not int for x in e) for e in events):
        raise ValueError("endpoint_type")
    if type(endpoint_count) is not int or len(events) != endpoint_count:
        errors.append("dropped_endpoint")
    if not events or events[0][0] != start or events[-1] != [end, 5, 0]:
        errors.append("dropped_endpoint")
    for i, (tick, phase, site) in enumerate(events):
        if tick < 0 or phase not in range(6) or (phase == 5 and i != len(events) - 1):
            errors.append("endpoint_schema")
        if site > 0:
            if opened is not None or phase != 4 or site not in (1, 2, 3, 4):
                errors.append("overlap_or_hidden_serialization")
            opened = site
        elif site < 0:
            if opened != -site or phase != 0:
                errors.append("dropped_serialization_endpoint")
            closed.append(-site)
            opened = None
        elif opened is not None:
            errors.append("interrupted_serialization")
        if i + 1 < len(events):
            delta = events[i + 1][0] - tick
            if delta < 0:
                errors.append("nonmonotone_clock")
            if phase in range(5):
                totals[PHASES[phase]] += delta
    if (
        opened is not None
        or closed.count(1) != 1
        or closed.count(4) != 1
        or len(closed) != len(set(closed))
    ):
        errors.append("serialization_coverage")
    residual = end - start - sum(totals.values())
    if residual != 0 or end <= start or any(x < 0 for x in totals.values()):
        errors.append("nonnegative_closure")
    return dict(
        valid=not errors,
        errors=errors,
        components_ns=totals,
        residual_ns=residual,
        elapsed_ns=end - start,
        tolerance_ns=0,
        endpoint_count=endpoint_count,
        retained_endpoint_count=len(events),
    )


def reduce(rows: list[Json]) -> Json:
    """Average ABBA pairs within clusters; report simultaneous diagnostic intervals."""
    cells: dict[tuple[Any, ...], dict[str, list[Json]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("cluster_id", "boundary", "arm", "block"))
        if (
            key in seen
            or row["arm"] not in ARMS
            or row["boundary"] not in BOUNDARIES
            or row["block"] not in range(4)
        ):
            raise ValueError("roster_duplicate_or_foreign")
        seen.add(key)
        cells[(row["size"], row["boundary"], row["arm"])][row["cluster_id"]].append(row)
    semantics, closure, overheads = bool(rows), bool(rows), []
    for (size, boundary, arm), clusters in sorted(cells.items()):
        pairs = []
        for sample in clusters.values():
            sample = sorted(sample, key=lambda r: r["block"])
            if len(sample) != 4 or sorted(r["instrumented"] for r in sample) != [
                False,
                False,
                True,
                True,
            ]:
                raise ValueError("roster_pair")
            semantics &= all(r["semantics"] == sample[0]["semantics"] for r in sample)
            closure &= all(
                r["closure_valid"]
                and r["status"] != "failed"
                and math.isfinite(r["elapsed_ns"])
                and r["elapsed_ns"] > 0
                for r in sample
            )
            pairs.append(
                [
                    sum(r["elapsed_ns"] for r in sample if r["instrumented"] == flag) / 2
                    for flag in (False, True)
                ]
            )
        data = np.array(pairs)
        label = dict(
            size=size,
            boundary=boundary,
            arm=arm,
            independent_count=len(data),
            cluster="whole_world",
            familywise_confidence=0.99,
            simultaneous_cell_count=36,
            scope="development_diagnostic_only",
            draws=DRAWS,
        )
        complete_clocks = all(
            math.isfinite(r["elapsed_ns"]) and r["elapsed_ns"] > 0
            for sample in clusters.values()
            for r in sample
        )
        if not complete_clocks or np.any(data <= 0) or not np.all(np.isfinite(data)):
            overheads.append(
                dict(
                    label,
                    estimate=None,
                    upper=None,
                    accepted=False,
                    missing_reason="missing_real_clock",
                )
            )
            continue
        idx = np.random.default_rng(SEED + size).integers(0, len(data), (DRAWS, len(data)))
        draws = data[idx].mean(axis=1)
        upper = float(np.quantile(draws[:, 1] / draws[:, 0] - 1, 1 - 0.01 / 36))
        overheads.append(
            dict(
                label,
                estimate=float(data[:, 1].mean() / data[:, 0].mean() - 1),
                upper=upper,
                accepted=upper <= 0.05,
                missing_reason=None,
            )
        )
    return dict(
        semantic_identity=semantics,
        finite_nonnegative_closure=closure,
        pilot_overhead_rows=overheads,
    )


def controls() -> Json:
    """Attack reduction with injected clocks; these operands cannot grant elapsed acceptance."""
    good = [
        [0, 4, 1],
        [10, 0, -1],
        [20, 1, 0],
        [30, 2, 0],
        [40, 3, 0],
        [50, 4, 4],
        [60, 0, -4],
        [70, 5, 0],
    ]
    cases = dict(
        overlap=good[:1] + [[5, 4, 4]] + good[1:],
        dropped_endpoint=good[:-1],
        dropped_phase_endpoint=good[:2] + good[3:],
        nonmonotone=good[:2] + [[5, 1, 0]] + good[3:],
        hidden_serialization=good[:5] + [[50, 3, 4]] + good[6:],
        negative_residual=good[:-1] + [[90, 5, 0]],
    )
    return dict(
        scope="injected_reduction_only",
        valid_control=partition(good, 0, 70, endpoint_count=8),
        negative_controls={k: partition(v, 0, 70, endpoint_count=8) for k, v in cases.items()},
        passed=partition(good, 0, 70, endpoint_count=8)["valid"]
        and all(not partition(v, 0, 70, endpoint_count=8)["valid"] for v in cases.values()),
    )

# unrelated later append
