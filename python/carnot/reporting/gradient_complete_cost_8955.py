"""REQ-REPORT-8955: charge gradient scans and ordinary deployment end to end.

Existing service boundaries remain intact. Counters are read after the timed
window; no component clock can turn unmeasured hardware work into a saving.
"""

from copy import deepcopy
import json
from pathlib import Path
import resource
import time
from typing import Any, cast

import numpy as np
from scipy.stats import t

from carnot.memory import gradient_replay_8947 as replay
from carnot.memory import compatible_replay_8933 as ordinary
from carnot.reporting import gradient_replay_methods_8947 as methods
from carnot.reporting import replay_complete_cost_8942 as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.selective_runner_8816 import stream as stream

Json = dict[str, Any]
__all__ = ["methods", "previous", "causal", "require"]
ROOT, SEED = methods.ROOT, 7658955
NAME, TASK = "experiment_8955_v765_gradient_complete_cost", "exp8955-gradient-complete-cost"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/gradient_complete_cost_8955.py",
    "python/carnot/reporting/gradient_complete_cost_runner_8955.py",
    CLI,
]
TESTS = ["tests/python/test_gradient_complete_cost_8955.py"]
ARMS = replay.ARMS[1:] + ("ordinary_guarded_current",)
SOURCE = methods.NAME
PINS = {
    SOURCE: (
        "sha256:65fffc8b8675bc066f25584d1fccf93257f9497b8d4e3ff2cad6a17a8f49a2ff",
        "gradient_methods_ready_score",
    )
}
QUALITY = "experiment_8954_v765_gradient_benefit_audit"
QUALITY_PIN = "sha256:90a5716c9c3f640d3060c5981fbb73509b9d0929a03b4cc26fa4cb9fe2c74740"
METRICS = previous.METRICS
require, authenticate = previous.require, previous.authenticate
causal = previous.causal
runtime = namespace(previous, "gradient8955_service")
STATS = dict(durable_bytes=0, writes=0)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so slow child work remains visible to supervision."""
    print(f"[exp8955] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Freeze fresh worlds and a seeded randomized balanced order before measurement."""
    planner = namespace(methods.old, "gradient8955_roster")
    planner.SEED = SEED
    order = np.random.default_rng(SEED).permutation(ARMS).tolist()
    rows = []
    for index in range(24):
        plan = planner.plan(600 + index)
        plan.update(seed=SEED + index, stream_id=f"cost8955-{index}", delays=[8] * 48)
        for role in ("online", "retention"):
            for i, unit in enumerate(plan[role]):
                unit["unit_id"] = f"{plan['stream_id']}/{role}/{i}"
        arms = order[index % 6 :] + order[: index % 6]
        for position, arm in enumerate(arms):
            rows.append(
                dict(
                    unit_id=plan["stream_id"] + "/" + arm,
                    cluster_id=plan["stream_id"],
                    seed=plan["seed"],
                    arm=arm,
                    order=position,
                    plan=plan,
                )
            )
    return rows


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate current methods; retained historical operands are byte inputs only."""
    raw.mkdir(parents=True, exist_ok=True)
    checks = [authenticate(sources / (n + ".json"), pin, field) for n, (pin, field) in PINS.items()]
    quality = authenticate(sources / (QUALITY + ".json"), QUALITY_PIN, None)
    refs = [r for c in checks for r in c["references"]]
    heads = next((Path(r["path"]) for r in refs if Path(r["path"]).name == "heads.json"), None)
    pre = dict(
        external_ready=all(c["passed"] for c in checks),
        references=refs + quality["references"],
        diagnostics=[d for c in checks for d in c["diagnostics"]],
        heads=json.loads(heads.read_bytes()) if heads else {},
        resources=dict(
            passed=all(
                (ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")
            )
        ),
        quality_reference=dict(
            authenticated=quality["passed"],
            gradient_benefit_score=quality["value"].get("gradient_benefit_score"),
            reference=quality["references"][:1],
            missing_quality_gate=quality["diagnostics"],
        ),
    )
    require(not pre["external_ready"] or bool(pre["heads"]), "retained_heads")
    atomic_json(raw / "preconditions.json", pre)
    return pre


class Store(replay.Store):
    """Keep both existing services unchanged and charge every durable state write."""

    def __init__(self, path: Path, head: Json, arm: str, anchors: list[Json], *, seed: int) -> None:
        self.baseline = arm == "ordinary_guarded_current"
        initializer = ordinary.Store.__init__ if self.baseline else replay.Store.__init__
        initializer(
            self, path, head, "guarded_current" if self.baseline else arm, anchors, seed=seed
        )

    def commit(self, state: Json) -> None:
        """Acknowledge a write only after both file and directory barriers complete."""
        if self.baseline:
            ordinary.Store.commit(self, state)
        else:
            replay.Store.commit(self, state)
        previous.Store.sync_directory(cast(Any, self))
        STATS["durable_bytes"] += self.path.stat().st_size
        STATS["writes"] += 1

    def train_release(self, state: Json, row: Json) -> None:
        """The deployment control applies ordinary updates without candidate scans."""
        if self.baseline:
            ordinary.Store.train_release(self, state, row)
        else:
            replay.Store.train_release(self, state, row)


def worker(request: Path, stop: int) -> Json:
    """Read RSS and storage counts after the original complete service window closes."""
    STATS.update(durable_bytes=0, writes=0)
    result = dict(runtime.worker(request, stop))
    result.update(
        storage=dict(STATS),
        peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    atomic_json(request.parent / f"segment-{stop}.json", result)
    return result


def operations(segments: list[Json]) -> Json:
    """Separate evaluated gradients from applied ones without observing timed loops."""
    op = dict(runtime.operations(segments))
    state = segments[-1]["state"]
    op.update(
        evaluated_gradients=state.get("evaluated_gradients", state["training_calls"]),
        applied_gradients=state["training_calls"],
        gradient_scan_candidates=len(state.get("candidate_gradient_rows", [])),
        gradient_dot_products=len(state.get("candidate_gradient_rows", [])),
        distance_candidates=(2 if state["arm"] == "signature_replay" else 1)
        * sum(len(r["buffer_ids"]) for r in state["selection_rows"])
        if "candidate_gradient_rows" in state
        else 0,
        durable_bytes=sum(s["storage"]["durable_bytes"] for s in segments),
    )
    return op


def verify(cell: Json) -> None:
    """Reuse all restart assertions and additionally bind scan and durable counters."""
    normalized = deepcopy(cell)
    if normalized["spec"]["arm"] == "ordinary_guarded_current":
        normalized["spec"]["arm"] = "guarded_current"
    verifier.verify(normalized)
    state = cell["segments"][-1]["state"]
    baseline = cell["spec"]["arm"] == "ordinary_guarded_current"
    require(
        (baseline and not state.get("candidate_gradient_rows"))
        or (
            not baseline
            and state["evaluated_gradients"]
            == len(state["candidate_gradient_rows"]) + state["training_calls"] // 2
        ),
        "scan_accounting",
    )
    require(
        all(
            s["peak_memory_bytes"] > 0 and s["storage"]["durable_bytes"] > 0
            for s in cell["segments"]
        ),
        "resource_accounting",
    )
    require(
        sum(s["storage"]["writes"] for s in cell["segments"]) == cell["operations"]["file_fsync"],
        "durable_writes",
    )


runtime.Store, runtime.methods = Store, methods
executor = namespace(previous, "gradient8955_execute")
executor.ROOT, executor.CLI = ROOT, CLI
executor.operations, executor.verify = operations, verify


def execute(raw: Path, spec: Json, heads: Json) -> Json:
    """Bind the shared process runner to the current child transport and controls."""
    executor.stream = stream
    return dict(executor.execute(raw, spec, heads))


verifier = namespace(previous, "gradient8955_verify")
verifier.operations = operations


def hardware() -> list[Json]:
    """Retain board limits; dot products have no qualified offload implementation."""
    rows = previous.hardware()
    for row in rows:
        if row["target"] in ("GPU", "NPU"):
            row["work"] = "batch gradient dot-products only; transfers/runtime unmeasured"
        if row["target"] == "KV260":
            row["work"] = "qualified quadratic sampling k_max<=5; no gradient-scan implementation"
    return rows


def reduce(raw: Path, reason: str) -> Json:
    """Reduce each scheduled process cell, keeping missing pairs out of interval n."""
    rows, processes, durability = [], [], []
    totals: Json = {}
    calls = 0
    for spec in roster():
        path = raw / spec["cluster_id"] / spec["arm"] / "unit.json"
        cell = json.loads(path.read_bytes()) if path.exists() else None
        failure = path.with_name("failure.json")
        status = "complete" if cell else "failed" if failure.exists() else "censored"
        missing = json.loads(failure.read_bytes())["reason"] if failure.exists() else reason
        if cell:
            require(cell["spec"] == spec, "cell_roster_drift")
            verify(cell)
            op = cell["operations"]
            control_calls = cell["control"]["state"]["training_calls"]
            calls += op["evaluated_gradients"] + cell["control"]["state"].get(
                "evaluated_gradients", control_calls
            )
            processes.append(
                dict(
                    **{k: spec[k] for k in ("unit_id", "cluster_id", "seed", "arm", "order")},
                    parent_bounds=cell["parent_bounds"],
                    child_bounds=[
                        dict(cold=s["cold_bounds"], warm=s["warm_bounds"], pid=s["pid"])
                        for s in cell["segments"]
                    ],
                    operations=op,
                    training_calls=op["evaluated_gradients"]
                    + cell["control"]["state"].get("evaluated_gradients", control_calls),
                    peak_memory_bytes=max(s["peak_memory_bytes"] for s in cell["segments"]),
                    request_denominator=48,
                    update_denominator=48,
                )
            )
            durability.append(
                dict(
                    unit_id=spec["unit_id"],
                    cluster_id=spec["cluster_id"],
                    seed=spec["seed"],
                    arm=spec["arm"],
                    **cell["restart"],
                    durable_bytes=op["durable_bytes"],
                    file_fsync=op["file_fsync"],
                    directory_fsync=op["directory_fsync"],
                    state_hash=canonical_hash(cell["segments"][-1]["state"]),
                )
            )
            for key, count in op.items():
                totals[key] = totals.get(key, 0) + count
        for metric, bounds in zip(
            METRICS, ("parent_bounds", "cold_bounds", "warm_bounds"), strict=True
        ):
            duration = (cell[bounds][1] - cell[bounds][0]) / 1e9 if cell else None
            rows.append(
                dict(
                    unit_id=spec["unit_id"],
                    cluster_id=spec["cluster_id"],
                    seed=spec["seed"],
                    arm=spec["arm"],
                    metric=metric,
                    value=duration,
                    numerator=duration,
                    denominator=1,
                    request_denominator=48,
                    update_denominator=48,
                    status=status,
                    missing_reason=None if cell else missing,
                    censor_reason=None if cell else missing,
                    condition="ordinary_deployment" if spec["arm"] == ARMS[-1] else "matched_scan",
                )
            )
    summary = []
    intervals = []
    for metric in METRICS:
        for arm in ARMS:
            samples = [
                r["value"]
                for r in rows
                if r["metric"] == metric and r["arm"] == arm and r["status"] == "complete"
            ]
            summary.append(
                dict(
                    arm=arm,
                    metric=metric,
                    independent_count=len(samples),
                    median=float(np.median(samples)) if samples else None,
                    p50=float(np.median(samples)) if samples else None,
                    p95=float(np.quantile(samples, 0.95)) if samples else None,
                    p95_scope="descriptive across independent process/world seeds",
                )
            )
        lookup = {
            (r["cluster_id"], r["arm"]): r["value"]
            for r in rows
            if r["metric"] == metric and r["status"] == "complete"
        }
        for arm in (
            "guarded_current",
            "uniform_replay",
            "signature_replay",
            "ordinary_guarded_current",
        ):
            deltas = [
                lookup[c, "gradient_replay"] - lookup[c, arm]
                for c in sorted({r["cluster_id"] for r in rows})
                if (c, arm) in lookup and (c, "gradient_replay") in lookup
            ]
            n = len(deltas)
            mean = float(np.mean(deltas)) if n else None
            margin = (
                float(t.ppf(1 - 0.01 / 24, n - 1) * np.std(deltas, ddof=1) / n**0.5)
                if n > 1
                else None
            )
            intervals.append(
                dict(
                    metric=metric,
                    arm="gradient_replay",
                    against=arm,
                    independent_count=n,
                    paired_differences=deltas,
                    mean=mean,
                    lower=mean - margin if margin is not None and mean is not None else None,
                    upper=mean + margin if margin is not None and mean is not None else None,
                    family_alpha=0.01,
                    family_comparisons=12,
                    method="paired process Student t; simultaneous99-percent Bonferroni; descriptive finite workloads",
                )
            )
    complete = sum(r["status"] == "complete" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    independent = sum(
        sum(p["cluster_id"] == c for p in processes) == 6
        for c in {s["cluster_id"] for s in roster()}
    )
    return dict(
        rows=rows,
        service_cost_rows=summary,
        deployment_baseline_rows=[r for r in rows if r["arm"] == "ordinary_guarded_current"],
        peak_memory_bytes=max((p["peak_memory_bytes"] for p in processes), default=0),
        process_seed_rows=processes,
        durability_rows=durability,
        paired_cost_intervals=intervals,
        operation_counts=totals,
        numerical_training_calls=calls,
        intended_count=432,
        completed_count=complete,
        censored_count=432 - complete - failed,
        failed_count=failed,
        excluded_count=0,
        independent_count=independent,
        sampling_call_count=0,
        removable_fraction=None,
        service_100x_projection=None,
        hardware_compatibility_rows=hardware(),
    )


def measure(raw: Path, pre: Json) -> None:
    """Execute every scheduled cell, retaining failures without shrinking denominators."""
    atomic_json(raw / "roster.json", dict(rows=roster()))
    began = time.monotonic()
    failures = []
    for i, spec in enumerate(roster()):
        require(time.monotonic() - began < 2400, "measurement_budget")
        progress("cell_before", i, 144 - i)
        directory = raw / spec["cluster_id"] / spec["arm"]
        try:
            execute(directory, spec, pre["heads"])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            atomic_json(directory / "failure.json", dict(reason=str(error), spec=spec))
            failures.append(str(error))
        progress("cell_after", i + 1, 143 - i)
    require(not failures, "service_cell_failure:" + str(failures))
