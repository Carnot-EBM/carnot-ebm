"""REQ-REPORT-8888: measure full durable adaptation on separate paired streams.

The four admitted Exp8881 mechanisms run unchanged. Whole-query oracle labels
are paid only by verified arms. These CPU costs establish no learning benefit.
"""

from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
import random
import resource
import stat
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from carnot.reporting import topology_memory_methods_8881 as methods
from carnot.memory import categorical_memory_8830 as categorical
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import feedback_information_8867 as diagnosis
from carnot.reporting import memory_size_cost_8848 as process_base
from carnot.memory.transactional_constraint_memory import _atomic_write, canonical_json_bytes
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8888_v760_matched_adaptation_cost"
TASK = "exp8888-matched-adaptation-cost"
CLI = "scripts/experiments/" + NAME + ".py"
SEED = 7608888
OWNED = [
    "python/carnot/reporting/matched_adaptation_cost_8888.py",
    "python/carnot/reporting/matched_adaptation_runner_8888.py",
    CLI,
]
TESTS = ["tests/python/test_matched_adaptation_cost_8888.py"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts while the fixed execution deadline continues to run."""
    print(f"[exp8888] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Freeze twelve new RNG trials and arm order without reading science outcomes."""
    rng = random.Random(SEED)
    rows = []
    for index in range(12):
        plan = methods.plan(48 + index if index < 6 else 60 + index - 6)
        plan["stream_id"] = f"cost8888-{index:02d}"
        for r in plan["online"]:
            r["unit_id"] = "cost8888/" + r["unit_id"]
        arms = list(methods.ARMS)
        rng.shuffle(arms)
        for order, arm in enumerate(arms):
            rows.append(
                dict(
                    unit_id=plan["stream_id"] + "/" + arm,
                    cluster_id=plan["stream_id"],
                    seed=plan["seed"],
                    arm=arm,
                    plan=plan,
                    order=order,
                    denominator=48,
                )
            )
    return rows


def segment(request: Path, start: int, stop: int | None, *, emit: bool = False) -> Json:
    """Run real atomic stores, recording inclusive phases and actual storage barriers."""
    spec = json.loads(request.read_bytes())
    plan, arm = spec["plan"], spec["arm"]
    phases: Json = {}
    labels = dict(acquisition=0, source_check=0, primitive_checks=0)
    audit: Json = dict(file_fsync=0, directory_fsync=0, serialized_bytes=0, commits=[])
    original_fsync, original_decide = os.fsync, symbolic.decide
    current = "initialization"

    def fsync(fd: int) -> None:
        audit["directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync"] += 1
        original_fsync(fd)

    def write(path: Path, data: bytes) -> None:
        began = time.monotonic_ns()
        _atomic_write(path, data)
        audit["serialized_bytes"] += len(data)
        audit["commits"].append(
            dict(
                phase=current,
                bytes=len(data),
                hash=canonical_hash(json.loads(data)),
                duration_ns=time.monotonic_ns() - began,
            )
        )

    def decide(value: Json) -> Json:
        kind = "primitive_checks" if len(value["query"]) == 1 else current
        labels[kind] += 1
        return dict(original_decide(value))

    def timed(name: str, fn: Any, *args: Any) -> Any:
        nonlocal current
        current = name
        began = time.monotonic_ns()
        value = fn(*args)
        phases[name] = phases.get(name, 0) + time.monotonic_ns() - began
        return value

    predictions, releases, occupancy = [], [], []
    schedule = {
        max(i + plan["delays"][i - 1] for i in range(end - 5, end + 1)): list(
            range(end - 5, end + 1)
        )
        for end in range(6, 49, 6)
    }
    with (
        patch.object(os, "fsync", fsync),
        patch.object(categorical, "_atomic_write", write),
        patch.object(primitive, "_atomic_write", write),
        patch.object(symbolic, "decide", decide),
    ):
        store = timed("initialization", methods.Store, request.parent / "state.json", spec["head"])
        initial = deepcopy(store.state)
        graphs = methods.graphs()["online"]
        worlds = timed(
            "world_construction",
            lambda: [
                methods.world(
                    graphs[r["template"]], r["unit_id"] + "/", r["error_type"], r["world_seed"]
                )
                for r in plan["online"]
            ],
        )
        tail_started, tail_events, maximum_pending = None, 0, len(store.state["pending"])
        checkpoint24 = None
        for ordinal in range(start + 1, (stop or max(schedule)) + 1):
            if ordinal <= 48:
                value = worlds[ordinal - 1]
                predictions.append(
                    timed("prediction_features_pending", store.predict, f"e{ordinal}", value, 0.5)
                )
                store.close()
                maximum_pending = max(maximum_pending, len(store.state["pending"]))
                if arm == "frozen_query_only":
                    timed("syntax_admission", store.admit, value["query"])
            if ordinal == 49:
                tail_started = time.monotonic_ns()
            if ordinal in schedule:
                events = schedule[ordinal]
                if ordinal > 48:
                    tail_events += len(events)
                if arm == "frozen_query_only":
                    timed("acknowledgement", store.acknowledge, events)
                else:
                    values = [worlds[i - 1] for i in events]
                    targets = timed(
                        "acquisition", lambda: [str(decide(v)["decision"]) for v in values]
                    )
                    if arm == "matched_permuted":
                        targets = timed(
                            "permutation",
                            diagnosis.permute,
                            targets,
                            methods.SEED + plan["rng_seed"] + ordinal,
                        )
                    releases += timed(
                        "source_check",
                        methods.release,
                        store,
                        values,
                        events,
                        ordinal,
                        arm,
                        targets,
                    )
            occupancy.append(
                dict(
                    ordinal=ordinal,
                    topology_keys=list(store.state["templates"]),
                    pending=len(store.state["pending"]),
                )
            )
            if ordinal == 24:
                checkpoint24 = deepcopy(store.state)
            if ordinal % 8 == 0:
                progress(arm, ordinal, max(schedule) - ordinal)
        result = dict(
            initial_state=initial,
            state=deepcopy(store.state),
            predictions=predictions,
            releases=releases,
            occupancy=occupancy,
            label_counts=labels,
            maximum_pending=maximum_pending,
            phase_costs=phases,
            audit=audit,
            checkpoint24_state=checkpoint24,
            tail_drain=dict(
                duration_ns=time.monotonic_ns() - tail_started if tail_started else 0,
                release_events=tail_events,
                pending_after=len(store.state["pending"]),
            ),
        )
        if emit:
            current = "response"
            began = time.monotonic_ns()
            write(request.parent / f"response-{start}.json", canonical_json_bytes(result))
            phases["response"] = time.monotonic_ns() - began
            print("AUDIT " + json.dumps(dict(audit=audit, phase_costs=phases)), flush=True)
    return result


def execute(raw: Path, spec: Json, head: Json) -> Json:
    """Pay for two real launches and compare their result with an uninterrupted store."""
    began = time.monotonic_ns()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    raw.mkdir(parents=True, mode=0o700)
    request = raw / "request.json"
    _atomic_write(request, canonical_json_bytes(dict(spec, head=head)))
    supervisor = namespace(process_base, "cost8888_process")
    supervisor.CLI, supervisor.progress = CLI, progress
    commands: list[Json] = []
    segments: list[Json] = []
    for start in (0, 24):
        progress("subprocess_before", len(segments), 2 - len(segments))
        command = supervisor.process(request, start, 60)
        commands.append(command)
        progress("subprocess_after", len(commands), 2 - len(commands))
        if command["status"] != "completed":
            break
        value = command["result"]
        audit = [
            line[6:]
            for line in Path(command["log"]["path"]).read_text().splitlines()
            if line.startswith("AUDIT ")
        ]
        value.update(json.loads(audit[-1]))
        segments.append(value)
    ended = time.monotonic_ns()
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = dict(
        spec,
        status="complete" if len(segments) == 2 else "failed",
        missing_reason=None if len(segments) == 2 else "child_failure_or_deadline",
        metric=(ended - began) / 48,
        complete_ns=ended - began,
        parent_started_ns=began,
        parent_ended_ns=ended,
        commands=commands,
        segments=segments,
        restart=dict(passed=False),
        reference_duration_ns=0,
        parent_io=dict(
            request_bytes=request.stat().st_size,
            file_fsync=1 + len(commands),
            directory_fsync=1,
            request_commits=1,
            log_file_fsync=len(commands),
        ),
        cpu_counters=dict(
            user_s=cpu_after.ru_utime - cpu_before.ru_utime,
            system_s=cpu_after.ru_stime - cpu_before.ru_stime,
            child_max_rss_kib=cpu_after.ru_maxrss,
            memory_scope="cumulative child peak RSS; no per-phase allocation claim",
        ),
    )
    row["operation_counts"] = dict(
        queries=sum(len(s["predictions"]) for s in segments),
        feedback_labels=sum(s["label_counts"]["acquisition"] for s in segments),
        source_label_checks=sum(s["label_counts"]["source_check"] for s in segments),
        applied_updates=sum(r["optimizer_steps"] for s in segments for r in s["releases"]),
        rejected_updates=0,
        store_and_response_commits=sum(len(s["audit"]["commits"]) for s in segments),
        file_fsync=sum(s["audit"]["file_fsync"] for s in segments) + row["parent_io"]["file_fsync"],
        directory_fsync=sum(s["audit"]["directory_fsync"] for s in segments) + 1,
        state_response_request_bytes=sum(s["audit"]["serialized_bytes"] for s in segments)
        + request.stat().st_size,
        process_log_bytes=sum(
            Path(c["log"]["path"]).stat().st_size for c in commands if "log" in c
        ),
        complete=len(segments) == 2,
    )
    if len(segments) == 2:
        progress("uninterrupted_control_before", 0, 1)
        baseline = raw / "uninterrupted" / "request.json"
        atomic_json(baseline, dict(spec, head=head))
        control_start = time.monotonic_ns()
        control = segment(baseline, 0, None)
        row["reference_duration_ns"] = time.monotonic_ns() - control_start
        atomic_json(raw / "uninterrupted.json", control)
        first, second = segments
        row["restart"] = dict(
            passed=first["state"] == second["initial_state"]
            and first["predictions"] + second["predictions"] == control["predictions"]
            and second["state"] == control["state"],
            checkpoint=24,
            exact_state=first["state"] == second["initial_state"],
            uninterrupted_reference=reference(raw / "uninterrupted.json"),
            process_ids_paid=True,
        )
        progress("uninterrupted_control_after", 1, 0)
    atomic_json(raw / "unit.json", row)
    return row


def parity(row: Json) -> bool:
    """Reject incomplete labels, barriers, occupancy or tail work before cost readiness."""
    if row["status"] != "complete":
        return False
    first, last = row["segments"]
    segments, query_only = row["segments"], row["arm"] == "frozen_query_only"
    releases = [r for s in segments for r in s["releases"]]
    commits = [r for s in segments for r in s["audit"]["commits"]]
    return bool(
        row["restart"]["passed"]
        and row["restart"]["exact_state"]
        and first["state"] == last["initial_state"]
        and not first["initial_state"]["templates"]
        and len(first["predictions"] + last["predictions"]) == 48
        and len(last["state"]["templates"]) == len(set(last["state"]["templates"])) == 8
        and last["state"]["pending"] == {}
        and len(set(last["state"]["acks"])) == 48
        and len(last["state"]["released"]) == (0 if query_only else 48)
        and len(releases) == (0 if query_only else 48)
        and sum(r["optimizer_steps"] for r in releases)
        == (48 if row["arm"].startswith("matched_") else 0)
        and all(
            sum(s["label_counts"][k] for s in segments) == (0 if query_only else 48)
            for k in ("acquisition", "source_check")
        )
        and all(s["maximum_pending"] <= 32 for s in segments)
        and sum(s["audit"]["file_fsync"] for s in segments)
        == sum(s["audit"]["directory_fsync"] for s in segments)
        == len(commits)
        == (108 if query_only else 60)
        and sum(s["audit"]["serialized_bytes"] for s in segments)
        == sum(c["bytes"] for c in commits)
        and last["tail_drain"]["pending_after"] == 0
        and last["tail_drain"]["release_events"] > 0
        and row["complete_ns"] == row["parent_ended_ns"] - row["parent_started_ns"] > 0
        and all(
            row["parent_started_ns"] <= c["started_ns"] < c["ended_ns"] <= row["parent_ended_ns"]
            and c["exit_code"] == 0
            for c in row["commands"]
        )
    )


def reduce_costs(rows: list[Json], setup_ns: int) -> Json:
    """Use six simultaneous99-percent paired t intervals on whole-stream costs."""
    lookup = {(r["cluster_id"], r["arm"]): r for r in rows if r["status"] == "complete"}
    intervals, amortization = [], []
    for control, treatment in combinations(methods.ARMS, 2):
        differences = [
            lookup[c, treatment]["complete_ns"] - lookup[c, control]["complete_ns"]
            for c in sorted({r["cluster_id"] for r in roster()})
            if (c, control) in lookup and (c, treatment) in lookup
        ]
        n = len(differences)
        mean = float(np.mean(differences)) if n else None
        margin = (
            float(t.ppf(1 - 0.01 / 12, n - 1) * np.std(differences, ddof=1) / n**0.5)
            if n > 1
            else None
        )
        intervals.append(
            dict(
                control_arm=control,
                treatment_arm=treatment,
                differences_ns=differences,
                paired_streams=n,
                missing_pairs=12 - n,
                mean_difference_ns=mean,
                lower_ns=mean - margin if mean is not None and margin is not None else None,
                upper_ns=mean + margin if mean is not None and margin is not None else None,
                familywise_confidence=0.99,
                family_size=6,
                method="Bonferroni paired Student-t; independent Gaussian stream differences assumed",
            )
        )
    for arm in methods.ARMS:
        costs = [r["complete_ns"] for r in rows if r["status"] == "complete" and r["arm"] == arm]
        for windows in (1, 10, 100):
            amortization.append(
                dict(
                    arm=arm,
                    windows=windows,
                    setup_ns=setup_ns,
                    setup_per_stream_ns=setup_ns / windows,
                    mean_query_ns=(float(np.mean(costs)) + setup_ns / windows) / 48
                    if costs
                    else None,
                )
            )
    return dict(
        intervals=intervals,
        amortization=amortization,
        speed_claim=False,
        quality_evidence_refs=[],
        scope="complete CPU cost only; no admitted scientific quality evidence",
    )
