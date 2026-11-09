"""REQ-REPORT-8942: charge complete replay service without an inline observer.

Only outer lifecycle boundaries use clocks. Retained state supplies operation
counts after timing, so feature, search, gradient and storage costs stay joined.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import t

from carnot.memory.compatible_replay_8933 import Store as ReplayStore
from carnot.reporting import compatible_replay_streams_8940 as causal
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.selective_runner_8816 import stream
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["require", "methods", "causal"]
methods = causal.methods
ROOT, SEED = causal.ROOT, 7648942
NAME, TASK = "experiment_8942_v764_replay_complete_cost", "exp8942-replay-complete-cost"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/replay_complete_cost_8942.py",
    "python/carnot/reporting/replay_complete_cost_runner_8942.py",
    CLI,
]
TESTS = ["tests/python/test_replay_complete_cost_8942.py"]
ARMS = methods.ARMS[1:]
SOURCE = causal.NAME
PINS = {
    SOURCE: (
        "sha256:2a4a7abddbc91cc4cca180cc7e65799639adee6ef881b5966123e812d2287a8f",
        "replay_rows_ready_score",
    ),
    "experiment_8926_v763_durable_cost_qualification": (
        "sha256:1c5f1969924341ce5416db9c649265dac03985946d722e111792b03372cb5a24",
        "adaptation_cost_ready_score",
    ),
    "experiment_8928_v763_hardware_operation_budget": (
        "sha256:efc2dabc7531fce61a6150657593d53f325656355669df28593be0af7d788230",
        None,
    ),
}
QUALITY = "experiment_8941_v764_compatible_replay_benefit"
QUALITY_PIN = "sha256:99a9be53396e5fd3867ab0981296d6be37e8863686d1b278188c11ceb0b9b489"
METRICS = ("complete_process_s", "cold_initialization_s", "warm_lifecycle_s")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so supervisors can detect slow work without guessing."""
    print(f"[exp8942] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Rotate arm order on24 fresh independent worlds, pairing every arm per world."""
    planner = namespace(methods.old, "replay8942_roster")
    planner.SEED = SEED
    rows = []
    for index in range(400, 424):
        plan = planner.plan(index)
        plan.update(seed=SEED + index, stream_id=f"cost8942-{index}", delays=[8] * 48)
        for role in ("online", "retention"):
            for i, unit in enumerate(plan[role]):
                unit["unit_id"] = f"{plan['stream_id']}/{role}/{i}"
        order = list(ARMS[index % 4 :] + ARMS[: index % 4])
        for position, arm in enumerate(order):
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


def authenticate(path: Path, pin: str, field: str | None) -> Json:
    """Retain precise gate diagnostics rather than turning missing evidence into zero."""
    d: Json = dict(
        check="missing_file",
        upstream=path.stem,
        artifact_field=field or "producer_bytes",
        expected=1 if field else pin,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs = []
    try:
        d["hash"] = sha256_file(path)
        d["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        if field:
            d["check"] = "missing_field"
            gate = value[field]
            d.update(
                check="observed_zero" if type(gate) is int and gate == 0 else "rejected_producer",
                observed=gate,
            )
            require(type(gate) is int and gate == 1, "upstream_gate")
        d.update(
            check="rejected_producer",
            artifact_field="producer_bytes",
            expected=pin,
            observed=d["hash"],
        )
        require(
            d["hash"] == pin
            and value["experiment_id"] == int(path.stem.split("_")[1])
            and value["required_checks_passed"] is True,
            "producer_identity",
        )
        refs.append(reference(path))
        operands = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for i, ref in enumerate(operands):
            d.update(artifact_field="original_operand", path=ref["path"], expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "operand_drift")
            refs.append(ref)
            if i % 200 == 0:
                progress("authenticate_operands", i, len(operands) - i)
        return dict(passed=True, references=refs, value=value, diagnostics=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.update(
            error=str(error),
            check="missing_file"
            if isinstance(error, OSError)
            else "missing_field"
            if isinstance(error, KeyError)
            else d["check"],
        )
        return dict(passed=False, references=refs, value={}, diagnostics=[d])


def preconditions(raw: Path, sources: Path) -> Json:
    """Admit cost inputs independently of the optional scientific quality result."""
    raw.mkdir(parents=True, exist_ok=True)
    checks = [
        authenticate(sources / (name + ".json"), pin, field) for name, (pin, field) in PINS.items()
    ]
    pre = causal.preconditions(raw / "methods", sources)
    pre["diagnostics"] = [d for c in checks for d in c["diagnostics"]] + pre["diagnostics"]
    pre["references"] += [r for c in checks for r in c["references"]]
    pre["external_ready"] = not pre["diagnostics"]
    quality = authenticate(sources / (QUALITY + ".json"), QUALITY_PIN, None)
    pre["quality_reference"] = dict(
        authenticated=quality["passed"],
        compatible_benefit_score=quality["value"].get("compatible_benefit_score"),
        reference=quality["references"][:1],
        missing_quality_gate=quality["diagnostics"],
    )
    pre["references"] += quality["references"]
    atomic_json(raw / "preconditions.json", pre)
    return pre


class Store(ReplayStore):
    """Keep replay semantics and acknowledge file replacement only after directory fsync."""

    def __init__(self, path: Path, head: Json, arm: str, anchors: list[Json], *, seed: int) -> None:
        super().__init__(path, head, arm, anchors, seed=seed)
        self.sync_directory()

    def commit(self, state: Json) -> None:
        super().commit(state)
        self.sync_directory()

    def sync_directory(self) -> None:
        """Keep the directory entry durable after the existing file barrier."""
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def worker(request: Path, stop: int) -> Json:
    """Time one complete service segment; collect ledgers after closing its window."""
    progress("worker_before", 0, stop)
    data = json.loads(request.read_bytes())
    spec, plan = data["spec"], data["spec"]["plan"]
    cold_start = time.monotonic_ns()
    store = Store(
        request.parent / "state.json",
        data["head"],
        spec["arm"],
        methods.anchors(),
        seed=spec["seed"],
    )
    cold_end = time.monotonic_ns()
    initial = deepcopy(store.state)
    predictions = []
    warm_start = time.monotonic_ns()
    values = [
        methods.old.world(
            methods.old.graphs()["development"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]
    tick = store.state["prediction_cursor"]
    released = store.state["cursor"]
    while tick < stop:
        tick += 1
        predictions.append(
            store.predict(plan["online"][tick - 1]["unit_id"], values[tick - 1], tick, tick + 8)
        )
        if tick >= released + 16:
            store.release(
                [(plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)],
                tick,
            )
            released += 8
    if stop == 48:
        while released < 48:
            tick = max(tick, released + 16)
            store.release(
                [(plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)],
                tick,
            )
            released += 8
    warm_end = time.monotonic_ns()
    result = dict(
        pid=os.getpid(),
        cold_bounds=[cold_start, cold_end],
        warm_bounds=[warm_start, warm_end],
        initial_state=initial,
        state=store.state,
        predictions=predictions,
    )
    atomic_json(request.parent / f"segment-{stop}.json", result)
    progress("worker_after", stop, 48 - stop)
    return result


def operations(segments: list[Json]) -> Json:
    """Read counters from completed state; no callbacks observe the timed service."""
    state = segments[-1]["state"]
    requests, updates = state["prediction_cursor"], state["cursor"]
    training = sum(r["split"] == "train" for r in state["released"])
    searches = (
        sum(len(r["buffer_ids"]) for r in state["selection_rows"])
        if state["arm"] in ARMS[2:]
        else 0
    )
    return dict(
        requests=requests,
        updates=updates,
        train_releases=training,
        gradients=state["training_calls"],
        guard_scores=state["guard_scores"],
        distance_candidates=searches,
        prediction_feature_vectors=requests,
        release_blocks=len(state["blocks"]),
        file_fsync=len(segments) + requests + len(state["blocks"]),
        directory_fsync=len(segments) + requests + len(state["blocks"]),
        sampling_call_count=0,
    )


def execute(raw: Path, spec: Json, heads: Json) -> Json:
    """Charge child startup through final exit and keep control costs outside comparison."""
    request = raw / "request.json"
    head = list(heads.values())[int(spec["cluster_id"].split("-")[-1]) % len(heads)]
    atomic_json(request, dict(spec=spec, head=head))
    commands = []
    began = time.monotonic_ns()
    for stop in (24, 48):
        progress("subprocess_before", 0, 1)
        commands.append(
            stream(
                "segment" + str(stop),
                [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--segment",
                    str(request),
                    "--stop",
                    str(stop),
                ],
                raw / "logs",
                deadline=120,
            )
        )
        require(commands[-1]["passed"], "service_child_failure")
        progress("subprocess_after", 1, 0)
    ended = time.monotonic_ns()
    segments = [json.loads((raw / f"segment-{stop}.json").read_bytes()) for stop in (24, 48)]
    control_request = raw / "control" / "request.json"
    atomic_json(control_request, dict(spec=spec, head=head))
    progress("control_before", 0, 1)
    control_receipt = stream(
        "control",
        [sys.executable, "-u", str(ROOT / CLI), "--segment", str(control_request), "--stop", "48"],
        raw / "logs",
        deadline=120,
    )
    require(control_receipt["passed"], "control_child_failure")
    control = json.loads((control_request.parent / "segment-48.json").read_bytes())
    progress("control_after", 1, 0)
    result = dict(
        spec=spec,
        segments=segments,
        control=control,
        commands=commands,
        control_receipt=control_receipt,
        parent_bounds=[began, ended],
        cold_bounds=[began, segments[0]["cold_bounds"][1]],
        warm_bounds=[segments[0]["warm_bounds"][0], segments[1]["warm_bounds"][1]],
        operations=operations(segments),
        restart=dict(
            passed=segments[0]["state"] == segments[1]["initial_state"]
            and segments[0]["pid"] != segments[1]["pid"],
            prediction_cursor=24,
        ),
    )
    verify(result)
    atomic_json(raw / "unit.json", result)
    return result


def verify(cell: Json) -> None:
    """Recompute service controls from retained states so repaired hashes cannot bless drift."""
    first, last = cell["segments"]
    control = cell["control"]
    require(
        last["state"]["arm"] == cell["spec"]["arm"]
        and [r["group_id"] for r in control["predictions"]]
        == [u["unit_id"] for u in cell["spec"]["plan"]["online"]],
        "cell_identity",
    )
    require(
        cell["restart"] == dict(passed=True, prediction_cursor=24)
        and first["pid"] != last["pid"]
        and first["state"] == last["initial_state"]
        and first["state"]["prediction_cursor"] == 24
        and last["state"] == control["state"]
        and first["predictions"] + last["predictions"] == control["predictions"],
        "restart_parity",
    )
    require(
        cell["operations"] == operations(cell["segments"])
        and cell["operations"]["requests"] == cell["operations"]["updates"] == 48
        and cell["operations"]["gradients"] == 2 * cell["operations"]["train_releases"],
        "operation_denominators",
    )
    require(
        all(c["passed"] for c in cell["commands"]) and cell["control_receipt"]["passed"],
        "child_controls",
    )
    parent, warm, cold = (cell[k] for k in ("parent_bounds", "warm_bounds", "cold_bounds"))
    for segment, command in zip(cell["segments"], cell["commands"], strict=True):
        require(
            parent[0]
            <= command["started_monotonic_ns"]
            <= segment["cold_bounds"][0]
            < segment["cold_bounds"][1]
            <= segment["warm_bounds"][0]
            < segment["warm_bounds"][1]
            <= command["ended_monotonic_ns"]
            <= parent[1],
            "parent_child_clock_bounds",
        )
    require(
        parent[0] == cold[0] < cold[1] <= warm[0] < warm[1] <= parent[1]
        and warm == [first["warm_bounds"][0], last["warm_bounds"][1]]
        and cold[1] == first["cold_bounds"][1],
        "outer_clock_bounds",
    )
    state = last["state"]
    require(
        not state["pending"]
        and len(state["gradient_ledger"]) == state["training_calls"]
        and all(r["split"] == "train" for r in state["buffer"])
        and all(r["label"] == r["true_label"] for r in state["released"] if r["split"] == "guard"),
        "causal_controls",
    )


def hardware() -> list[Json]:
    """Map existing limits without new probes or an unmeasured acceleration fraction."""
    return [
        dict(
            target=target,
            work=work,
            status=status,
            measured_acceleration=False,
            removable_fraction=None,
            service_100x_projection=None,
        )
        for target, work, status in [
            (
                "CPU",
                "feature extraction, label release, guard, serialization, fsync, restart",
                "measured_complete_service",
            ),
            (
                "GPU",
                "batchable signed distance and two small-head gradients; transfer and batching unmeasured",
                "algorithmically_batchable_only",
            ),
            (
                "KV260",
                "existing compatible sampler only; k_max<=5; no sampler in replay",
                "qualified_sampler_only_not_replay",
            ),
            ("PolarFire", "Linux CPU dispatch only", "cpu_dispatch_only"),
            ("GateMate", "unchanged physical/JTAG 0xffffffff", "physically_blocked"),
            ("NPU", "no qualified runtime", "unqualified"),
            ("TSU", "no qualified runtime", "unqualified"),
        ]
    ]


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
            calls += op["gradients"] + control_calls
            processes.append(
                dict(
                    **{k: spec[k] for k in ("unit_id", "cluster_id", "seed", "arm", "order")},
                    parent_bounds=cell["parent_bounds"],
                    child_bounds=[
                        dict(cold=s["cold_bounds"], warm=s["warm_bounds"], pid=s["pid"])
                        for s in cell["segments"]
                    ],
                    operations=op,
                    training_calls=op["gradients"] + control_calls,
                    request_denominator=48,
                    update_denominator=48,
                )
            )
            durability.append(
                dict(
                    unit_id=spec["unit_id"],
                    **cell["restart"],
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
                    p95=float(np.quantile(samples, 0.95)) if samples else None,
                    p95_scope="descriptive across independent process/world seeds",
                )
            )
        lookup = {
            (r["cluster_id"], r["arm"]): r["value"]
            for r in rows
            if r["metric"] == metric and r["status"] == "complete"
        }
        for arm in ARMS[1:]:
            deltas = [
                lookup[c, arm] - lookup[c, ARMS[0]]
                for c in sorted({r["cluster_id"] for r in rows})
                if (c, arm) in lookup and (c, ARMS[0]) in lookup
            ]
            n = len(deltas)
            mean = float(np.mean(deltas)) if n else None
            margin = (
                float(t.ppf(1 - 0.01 / 18, n - 1) * np.std(deltas, ddof=1) / n**0.5)
                if n > 1
                else None
            )
            intervals.append(
                dict(
                    metric=metric,
                    arm=arm,
                    against=ARMS[0],
                    independent_count=n,
                    paired_differences=deltas,
                    mean=mean,
                    lower=mean - margin if margin is not None and mean is not None else None,
                    upper=mean + margin if margin is not None and mean is not None else None,
                    family_alpha=0.01,
                    family_comparisons=9,
                    method="paired process Student t; simultaneous99-percent Bonferroni; descriptive finite workloads",
                )
            )
    complete = sum(r["status"] == "complete" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    independent = sum(
        sum(p["cluster_id"] == c for p in processes) == 4
        for c in {s["cluster_id"] for s in roster()}
    )
    return dict(
        rows=rows,
        service_cost_rows=summary,
        process_seed_rows=processes,
        durability_rows=durability,
        paired_cost_intervals=intervals,
        operation_counts=totals,
        numerical_training_calls=calls,
        intended_count=288,
        completed_count=complete,
        censored_count=288 - complete - failed,
        failed_count=failed,
        excluded_count=0,
        independent_count=independent,
        sampling_call_count=0,
        removable_fraction=None,
        service_100x_projection=None,
        hardware_compatibility_rows=hardware(),
    )


def measure(raw: Path, pre: Json) -> None:
    """Execute the frozen census within a budget that reserves validation time."""
    atomic_json(raw / "roster.json", dict(rows=roster()))
    began = time.monotonic()
    for i, spec in enumerate(roster()):
        require(time.monotonic() - began < 2400, "measurement_budget")
        progress("cell_before", i, 96 - i)
        directory = raw / spec["cluster_id"] / spec["arm"]
        try:
            execute(directory, spec, pre["heads"])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            atomic_json(directory / "failure.json", dict(reason=str(error), spec=spec))
            raise
        progress("cell_after", i + 1, 95 - i)
