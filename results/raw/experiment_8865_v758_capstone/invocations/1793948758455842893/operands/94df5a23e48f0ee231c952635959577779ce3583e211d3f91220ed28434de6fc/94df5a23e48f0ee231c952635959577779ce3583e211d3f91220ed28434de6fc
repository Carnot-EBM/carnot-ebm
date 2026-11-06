"""REQ-REPORT-8862: observe whole durable linear windows without timing attribution."""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from carnot.reporting import memory_size_cost_8848 as old
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8862_v758_linear_durable_cost"
TASK = "exp8862-linear-durable-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/linear_durable_cost_8862.py",
    "python/carnot/reporting/linear_durable_runner_8862.py",
    CLI,
]
TESTS = ["tests/python/test_linear_durable_cost_8862.py"]
ARMS = ("linear_frozen", "linear_calibration_only", "linear_combined")
SEED = 7588862
SOURCES = [
    (
        "results/experiment_8848_v757_memory_size_cost.json",
        "sha256:62ddef9b5966312650bf33fecb9b02308beffab6f58929e041b31a977c2994d3",
        "whole_cost_ready_score",
    ),
    (
        "results/experiment_8854_v758_linear_ablation_methods.json",
        "sha256:830795e2e0c73a0c819f0bff8ac9685a2d8dd142664d149d6e043fb7a295afe0",
        "linear_methods_ready_score",
    ),
]
SPEC_PIN = "sha256:1c4765e4f4be50bcd1dd765daab044cff107478cb024e0955c78a26b463f713e"
SPEC_COMMIT = "85aed359ed4134fd4aa717643c9ae1bcce9ba85f"
reference = old.reference
durable = old.durable


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so slow durable work remains distinguishable from a stall."""
    print(f"[exp8862] phase={phase} completed={completed} pending={pending}", flush=True)


def workload(block: int) -> list[Json]:
    """Pair24 repeats of eight prepared shapes with24 novel shapes across all arms."""
    return old.workload(block)[:48]


def roster() -> list[Json]:
    """Freeze independent seeds and randomize all six paired cells before timing."""
    rng = random.Random(SEED)
    rows = []
    for block in range(10):
        cells = list(itertools.product((0, 32), ARMS))
        rng.shuffle(cells)
        for position, (occupancy, arm) in enumerate(cells):
            rows.append(
                dict(
                    unit_id=f"b{block}:n{occupancy}:{arm}",
                    cluster_id=f"b{block}",
                    block=block,
                    seed=(11, 23, 37, 47, 59)[block % 5],
                    occupancy=occupancy,
                    arm=arm,
                    order_in_block=position,
                    denominator=48,
                )
            )
    return rows


def preflight(raw: Path) -> Json:
    """Authenticate exact producers and operands, resolving only the known historical spec."""
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    durable(raw / "writable.json", dict(private=True))
    failures, refs, heads = [], [], {}
    initialization_reference = None
    for name, pin, field in SOURCES:
        path = ROOT / name
        value, failure = old.old.authenticate(path, pin, field)
        if failure:
            failures.append(dict(failure, upstream=path.stem))
            continue
        refs += [reference(path)] + sum(
            (
                value[k]
                for k in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes")
            ),
            [],
        )
        if field == "linear_methods_ready_score":
            initialization_reference = value["raw_shard_hashes"]
    unique = {(r["path"], r["sha256"]): r for r in refs}
    bound = []
    for i, ref in enumerate(unique.values()):
        path = Path(ref["path"])
        if ref["sha256"] == SPEC_PIN and path.name == "spec.md":
            path = raw / "historical-reporting-spec.md"
            progress("historical_authority_before")
            data = subprocess.check_output(
                ["git", "show", SPEC_COMMIT + ":openspec/capabilities/research-reporting/spec.md"],
                cwd=ROOT,
                timeout=30,
            )
            path.write_bytes(data)
            progress("historical_authority_after", 1)
        digest = sha256_file(path) if path.is_file() else None
        if digest != ref["sha256"]:
            failures.append(
                dict(
                    check="immutable_operand",
                    upstream="historical_cost_or_methods",
                    artifact_field="raw_or_original_authority",
                    expected=ref["sha256"],
                    observed="missing_file" if digest is None else "rejected_producer",
                    path=str(path),
                    sha256=digest,
                )
            )
        else:
            bound.append(dict(path=str(path), sha256=digest))
        if i % 5000 == 0:
            progress("authenticate_operands", i, len(unique) - i)
    if not failures and initialization_reference:
        evidence = json.loads(Path(initialization_reference[0]["path"]).read_bytes())
        heads = {str(r["seed"]): r["parameters"] for r in evidence["initialization_rows"]}
    progress("numerical_dependency_before")
    import jax

    checks = dict(
        repository_virtualenv=(ROOT / ".venv/bin/python").is_file(),
        private_output=json.loads((raw / "writable.json").read_text())["private"],
        free_disk_bytes=shutil.disk_usage(raw).free,
        minimum_free_bytes=128 * 1024**2,
        numpy_version=np.__version__,
        jax_version=jax.__version__,
        cpu_devices=[str(x) for x in jax.devices("cpu")],
        total_budget_s=4200,
    )
    progress("numerical_dependency_after", 1)
    return dict(
        failures=failures, sources=bound, heads=heads, checks=checks, internal_failure=False
    )


def worker(request: Path, start: int) -> None:
    """Use the admitted linear release while counting actual store writes and barriers."""
    progress("linear_import_before")
    from carnot.reporting import linear_ablation_methods_8854 as linear

    progress("linear_import_after")
    spec = json.loads(request.read_bytes())
    state_path = request.parent / "state.json"
    before = json.loads(state_path.read_bytes()) if state_path.exists() else None
    counts = dict(file=0, directory=0)
    serialized = 0
    original_fsync, original_write = os.fsync, linear.c._atomic_write

    def fsync(fd: int) -> None:
        counts["directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"] += 1
        original_fsync(fd)

    def write(path: Path, data: bytes) -> None:
        nonlocal serialized
        serialized += len(data)
        original_write(path, data)

    predictions, releases = [], []
    hits, maximum_pending = 0, 0
    with (
        patch.object(os, "fsync", fsync),
        patch.object(linear.c, "_atomic_write", write),
        patch.object(linear.c.m, "_atomic_write", write),
    ):
        store = linear.c.Store(state_path, spec["head"])
        initial = deepcopy(store.state)
        preparation = spec["preparation"]
        total = spec["occupancy"] if preparation else 48
        stop = total if preparation else start + 24
        offset = 0 if preparation else 1000
        arm = "linear_additions_only" if preparation else spec["arm"]
        inputs = (
            [dict(payload=old.payload(spec["block"], i)) for i in range(total)]
            if preparation
            else workload(spec["block"])
        )

        def release(i: int) -> None:
            releases.append(
                linear.release(
                    store,
                    f"e{offset + i}",
                    inputs[i]["payload"],
                    offset + i + 8,
                    arm,
                    8,
                    label="support",
                )
            )

        for i in range(start, stop):
            body = inputs[i]["payload"]
            hits += int(linear.c.m.normalize(body) in store.state["templates"])
            predictions.append(store.predict(f"e{offset + i}", body, 0.5))
            maximum_pending = max(maximum_pending, len(store.state["pending"]))
            store.close()
            if i >= 8:
                release(i - 8)
            if (i + 1) % 8 == 0:
                progress("preparation" if preparation else "queries", i + 1, total - i - 1)
        if preparation or start == 24:
            for i in range(max(0, total - 8), total):
                release(i)
    snapshot = request.parent / f"state-{start}.json"
    durable(snapshot, store.state)
    durable(
        request.parent / f"response-{start}.json",
        dict(
            predictions=predictions,
            releases=releases,
            initial_state=initial,
            state=store.state,
            state_hash=canonical_hash(store.state),
            restart_input_hash=canonical_hash(before),
            state_file=reference(snapshot),
            fsync_counts=counts,
            template_hits=hits,
            maximum_pending=maximum_pending,
            serialized_bytes=serialized,
            state_bytes=state_path.stat().st_size,
        ),
    )
    progress("worker_exit", len(predictions))


def execute(
    raw: Path,
    spec: Json,
    head: Json,
    *,
    preparation: bool = False,
    prepared: Json | None = None,
    deadline: float = 60,
) -> Json:
    """Measure parent startup through final exit with fixture installation outside timing."""
    from carnot.memory.transactional_constraint_memory import _atomic_write

    raw.mkdir(parents=True, exist_ok=True)
    if prepared is not None:
        source = Path(prepared["segments"][0]["state_file"]["path"])
        _atomic_write(raw / "state.json", source.read_bytes())
    request = raw / "request.json"
    durable(request, dict(spec, head=head, preparation=preparation))
    commands, segments = [], []
    began = time.monotonic_ns()
    for start in (0,) if preparation else (0, 24):
        progress("child_before", start, (spec["occupancy"] if preparation else 48) - start)
        with patch.multiple(old, CLI=CLI, progress=lambda phase, *_counts: progress(phase)):
            command = old.process(request, start, deadline)
        commands.append(command)
        progress("child_after", start, 0)
        if command["status"] != "completed":
            break
        segments.append(command["result"])
    ended = time.monotonic_ns()
    complete = len(segments) == (1 if preparation else 2)
    initial = segments[0]["initial_state"] if segments else {}
    final = segments[-1]["state"] if segments else {}
    releases = [r for s in segments for r in s["releases"]]
    row = dict(
        spec,
        preparation=preparation,
        status="completed" if complete else "failed",
        missing_reason=None
        if complete
        else "child_deadline"
        if any(c["timed_out"] for c in commands)
        else "child_failed",
        metric=(ended - began) / (spec["occupancy"] or 1) if preparation else (ended - began) / 48,
        complete_ns=ended - began,
        parent_started_ns=began,
        parent_ended_ns=ended,
        commands=commands,
        segments=segments,
        query_count=sum(len(s["predictions"]) for s in segments),
        initial_occupancy=len(initial.get("templates", [])),
        final_occupancy=len(final.get("templates", [])),
        attempted_updates=len(releases),
        applied_updates=sum(r["optimizer_steps"] for r in releases),
        parameter_movement=sum(r["parameter_delta"] for r in releases),
        gradient_norm_sum=sum(r["gradient_norm"] for r in releases),
        template_hits=sum(s["template_hits"] for s in segments),
        template_additions=sum(r["template_added"] for r in releases),
        fsync_counts={
            k: sum(s["fsync_counts"][k] for s in segments) for k in ("file", "directory")
        },
        serialized_bytes=sum(s["serialized_bytes"] for s in segments),
        serialized_state_bytes=[s["state_bytes"] for s in segments],
        response_bytes=[
            (raw / f"response-{c['start']}.json").stat().st_size
            for c in commands
            if c["status"] == "completed"
        ],
        request=reference(request),
    )
    durable(raw / "window.json", row)
    return row


def parity(row: Json) -> Json:
    """Check exact restart, causal labels, state acknowledgement and counted durability."""
    passed = row["status"] == "completed"
    if passed:
        first, last = row["segments"][0], row["segments"][-1]
        initial, final = first["initial_state"], last["state"]
        prep, n = row["preparation"], row["occupancy"] if row["preparation"] else 48
        releases = [r for s in row["segments"] for r in s["releases"]]
        calibration = not prep and row["arm"] != "linear_frozen"
        additions = prep or row["arm"] == "linear_combined"
        passed = (
            len(row["segments"]) == (1 if prep else 2)
            and row["query_count"] == row["attempted_updates"] == len(releases) == n
            and row["initial_occupancy"]
            == len(initial["templates"])
            == (0 if prep else row["occupancy"])
            and row["final_occupancy"]
            == len(final["templates"])
            == (min(32, n + row["initial_occupancy"]) if additions else row["initial_occupancy"])
            and len(final["acks"]) == (n if prep else row["occupancy"] + n)
            and not final["pending"]
            and all(s["maximum_pending"] <= 32 for s in row["segments"])
            and (prep or last["restart_input_hash"] == first["state_hash"])
            and row["applied_updates"] == n * calibration
            and all(
                r["status"] == "complete"
                and r["feedback_ordinal"] == r["prediction_ordinal"] + 8
                and r["feedback_label"] == r["verified_label"] == "support"
                and r["optimizer_steps"] == int(calibration)
                for r in releases
            )
            and (calibration or initial["parameters"] == final["parameters"])
            and (not prep or all(r["template_added"] for r in releases))
            and row["fsync_counts"]
            == dict(file=2 * n + (1 if prep else 2), directory=2 * n + (1 if prep else 2))
            and all(s["state_hash"] == canonical_hash(s["state"]) for s in row["segments"])
            and row["complete_ns"] == row["parent_ended_ns"] - row["parent_started_ns"] > 0
            and all(
                row["parent_started_ns"]
                <= c["started_ns"]
                < c["ended_ns"]
                <= row["parent_ended_ns"]
                for c in row["commands"]
            )
        )
    return dict(unit_id=row["unit_id"], passed=bool(passed))


def reduce_costs(rows: list[Json], preparations: list[Json]) -> Json:
    """Use six simultaneous99-percent paired intervals and retain missing pairs."""
    lookup = {(r["block"], r["occupancy"], r["arm"]): r for r in rows if r["status"] == "completed"}
    intervals, absolute, amortization = [], [], []
    for occupancy in (0, 32):
        for arm in ARMS[1:]:
            differences = [
                lookup[b, occupancy, arm]["complete_ns"]
                - lookup[b, occupancy, ARMS[0]]["complete_ns"]
                for b in range(10)
                if (b, occupancy, arm) in lookup and (b, occupancy, ARMS[0]) in lookup
            ]
            intervals.append(
                dict(
                    occupancy=occupancy,
                    control_arm=ARMS[0],
                    treatment_arm=arm,
                    differences_ns=differences,
                )
            )
    for arm in ARMS[1:]:
        differences = [
            lookup[b, 32, arm]["complete_ns"]
            - lookup[b, 32, ARMS[0]]["complete_ns"]
            - lookup[b, 0, arm]["complete_ns"]
            + lookup[b, 0, ARMS[0]]["complete_ns"]
            for b in range(10)
            if all((b, n, a) in lookup for n in (0, 32) for a in (arm, ARMS[0]))
        ]
        intervals.append(
            dict(
                occupancy="32_minus_0",
                control_arm=ARMS[0],
                treatment_arm=arm,
                differences_ns=differences,
            )
        )
    for interval in intervals:
        differences = interval["differences_ns"]
        mean = float(np.mean(differences)) if differences else None
        margin = (
            float(
                t.ppf(1 - 0.01 / 12, len(differences) - 1)
                * np.std(differences, ddof=1)
                / len(differences) ** 0.5
            )
            if len(differences) > 1
            else None
        )
        interval.update(
            paired_blocks=len(differences),
            missing_pairs=10 - len(differences),
            familywise_confidence_percent=99,
            family_size=6,
            mean_increment_ns=mean,
            lower_ns=mean - margin if mean is not None and margin is not None else None,
            upper_ns=mean + margin if mean is not None and margin is not None else None,
            method="Bonferroni paired Student-t; Gaussian independent block differences assumed",
        )
    for occupancy, arm in itertools.product((0, 32), ARMS):
        costs = [
            r["complete_ns"]
            for r in rows
            if r["status"] == "completed" and r["occupancy"] == occupancy and r["arm"] == arm
        ]
        preps = [
            r["complete_ns"]
            for r in preparations
            if r["status"] == "completed" and r["occupancy"] == occupancy
        ]
        mean, prep = (
            (float(np.mean(costs)) if costs else None),
            (float(np.mean(preps)) if preps else None),
        )
        absolute.append(
            dict(
                occupancy=occupancy,
                arm=arm,
                count=len(costs),
                mean_window_ns=mean,
                p50_window_ns=float(np.quantile(costs, 0.5)) if costs else None,
                p95_window_ns=float(np.quantile(costs, 0.95)) if costs else None,
            )
        )
        for windows in (1, 10, 100):
            amortization.append(
                dict(
                    occupancy=occupancy,
                    arm=arm,
                    windows=windows,
                    preparation_mean_ns=prep,
                    preparation_per_window_ns=prep / windows if prep is not None else None,
                    amortized_query_ns=(mean + prep / windows) / 48
                    if mean is not None and prep is not None
                    else None,
                )
            )
    return dict(
        paired_cost_intervals=intervals,
        occupancy_rows=absolute,
        preparation_amortization_rows=amortization,
    )


def microbenchmark(head: Json) -> Json:
    """Time a pure numerical48-update batch separately; it cannot partition service costs."""
    progress("microbenchmark_before")
    from carnot.memory.categorical_memory_8830 import update

    weights = deepcopy(head)
    began = time.monotonic_ns()
    movements = []
    for _ in range(48):
        weights, movement = update(weights, [5, 5, 0, 2, 0, 1], "support")
        movements.append(movement)
    ended = time.monotonic_ns()
    progress("microbenchmark_after", 48)
    return dict(
        duration_ns=ended - began,
        numerical_training_calls=48,
        movement_rows=movements,
        scope="pure numerical CPU batch including first compilation; no service subtraction",
    )


def replay(path: Path) -> Json:
    """Rebuild all summaries from authenticated operands and reject lifecycle mutations."""
    try:
        from carnot.reporting.linear_durable_runner_8862 import build

        value = json.loads(path.read_bytes())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
        validation = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
        if build(work, validation) != value or work["schedule"] != roster():
            raise ValueError("reduction_drift")
        for row in work["window_rows"] + work["preparation_cost_rows"]:
            if row["status"] == "completed" and not parity(row)["passed"]:
                raise ValueError("lifecycle_drift")
            for segment in row.get("segments", []):
                if (
                    sha256_file(Path(segment["state_file"]["path"]))
                    != segment["state_file"]["sha256"]
                ):
                    raise ValueError("state_drift")
        return dict(passed=True, replay_passed=True, windows=len(work["window_rows"]))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="replay_operand_or_reduction_failure")
