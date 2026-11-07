"""REQ-REPORT-8848: measure complete durable costs at fixed template occupancies."""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import random
import selectors
import signal
import stat
import subprocess
import time
from typing import Any

import numpy as np
from scipy.stats import t

from carnot.reporting import complete_adaptation_cost_8835 as old
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8848_v757_memory_size_cost"
TASK = "exp8848-memory-size-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/memory_size_cost_8848.py",
    "python/carnot/reporting/memory_size_runner_8848.py",
    CLI,
]
TESTS = ["tests/python/test_memory_size_cost_8848.py"]
SEED = 7578848
COST = "results/experiment_8835_v756_complete_adaptation_cost.json"
COST_PIN = "sha256:87688903046083ca1ddad418c9f72d1c054712f4f8b496ff2b8241fb2f6fff49"
reference = old.reference
durable = old.durable


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so slow numerical work stays distinguishable from a stalled child."""
    print(f"[exp8848] phase={phase} completed={completed} pending={pending}", flush=True)


def payload(block: int, template: int) -> Json:
    """Six variable-equality bits give64 distinct structures without encoding answers."""
    terms = (
        ["?v0"]
        + ["?v0" if template & (1 << i) else f"?v{i + 1}" for i in range(6)]
        + ["?v7", "?v8", "?v9"]
    )
    query = [[terms[2 * i], f"r{i}", terms[2 * i + 1]] for i in range(5)]
    return dict(query=query, facts=[[f"b{block}{s}", r, f"b{block}{o}", True] for s, r, o in query])


def workload(block: int) -> list[Json]:
    """Half the requests repeat eight prepared structures; half use32 novel structures."""
    return [
        dict(
            template_index=(i // 2) % 8 if i % 2 == 0 else 32 + i // 2,
            payload=payload(block, (i // 2) % 8 if i % 2 == 0 else 32 + i // 2),
        )
        for i in range(64)
    ]


def roster() -> list[Json]:
    """Randomize six paired cells within each independent block before timing."""
    rng = random.Random(SEED)
    rows = []
    for block in range(10):
        cells = list(itertools.product((0, 8, 32), ("read_only", "combined")))
        rng.shuffle(cells)
        for position, (occupancy, arm) in enumerate(cells):
            rows.append(
                dict(
                    unit_id=f"b{block}:n{occupancy}:{arm}",
                    cluster_id=f"b{block}",
                    block=block,
                    occupancy=occupancy,
                    arm=arm,
                    order_in_block=position,
                    denominator=64,
                )
            )
    return rows


def preflight(raw: Path) -> Json:
    """Authenticate complete historical clocks and original authorities before new timing."""
    pre = old.preflight(raw)
    value, failure = old.authenticate(ROOT / COST, COST_PIN, "whole_cost_ready_score")
    if failure:
        pre["failures"].append(dict(failure, upstream="exp8835-complete-adaptation-cost"))
    else:
        refs = [
            reference(ROOT / COST),
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]
        for ref in refs:
            path = Path(ref["path"])
            if not path.is_file() or sha256_file(path) != ref["sha256"]:
                pre["failures"].append(
                    dict(
                        check="historical_operand",
                        upstream="exp8835",
                        artifact_field="raw_shard_hashes",
                        expected=ref["sha256"],
                        observed="missing_file" if not path.is_file() else "rejected_producer",
                        path=str(path),
                        sha256=sha256_file(path) if path.is_file() else None,
                    )
                )
        pre["sources"] += refs
    pre["checks"]["scipy_version"] = __import__("scipy").__version__
    pre["internal_failure"] = pre["internal_failure"] or any(
        "/python/" in f["path"] and f["observed"] == "rejected_producer" for f in pre["failures"]
    )
    return pre


def worker(request: Path, start: int) -> None:
    """Call the unchanged store and count actual barriers while preserving their behavior."""
    progress("numerical_import_before")
    from carnot.memory import causal_memory_8802 as store_module

    progress("numerical_import_after")
    spec = json.loads(request.read_text())
    state_path = request.parent / "state.json"
    before = json.loads(state_path.read_text()) if state_path.exists() else None
    counts = dict(file=0, directory=0)
    original_fsync = os.fsync

    def fsync(fd: int) -> None:
        counts["directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"] += 1
        original_fsync(fd)

    os.fsync = fsync
    predictions, releases, hits, rejections = [], [], 0, 0
    try:
        store = store_module.Store(state_path, spec["head"]["parameters"])
        initial = deepcopy(store.state)
        preparation = spec.get("preparation", False)
        total = spec["occupancy"] if preparation else 64
        offset = 0 if preparation else 1000
        stop = total if preparation else start + 32
        arm = "additions_only" if preparation else spec["arm"]
        inputs = (
            [dict(payload=payload(spec["block"], i)) for i in range(total)]
            if preparation
            else workload(spec["block"])
        )

        def release(i: int) -> None:
            nonlocal rejections
            body = inputs[i]["payload"]
            template = store_module.normalize(body)
            rejections += int(
                arm == "combined"
                and template not in store.state["templates"]
                and len(store.state["templates"]) == 32
            )
            releases.append(store.release(f"e{offset + i}", body, offset + i + 8, arm))

        for i in range(start, stop):
            body = inputs[i]["payload"]
            hits += int(store_module.normalize(body) in store.state["templates"])
            predictions.append(store.predict(f"e{offset + i}", body, spec["head"]["threshold"]))
            store.close()
            if i >= 8:
                release(i - 8)
            if (i + 1) % 8 == 0:
                progress("preparation" if preparation else "queries", i + 1, total - i - 1)
        if preparation or start == 32:
            for i in range(max(0, total - 8), total):
                release(i)
    finally:
        os.fsync = original_fsync
    snapshot = request.parent / f"state-{start}.json"
    durable(snapshot, store.state)
    durable(
        request.parent / f"response-{start}.json",
        dict(
            predictions=predictions,
            releases=releases,
            state=store.state,
            initial_state=initial,
            state_hash=canonical_hash(store.state),
            restart_input_hash=canonical_hash(before),
            state_file=reference(snapshot),
            fsync_counts=counts,
            template_hits=hits,
            capacity_rejections=rejections,
            state_bytes=len(state_path.read_bytes()),
        ),
    )
    progress("worker_exit", len(predictions), 0)


def process(request: Path, start: int, deadline: float) -> Json:
    """Forward output while enforcing a fixed deadline independent of heartbeats."""
    began = time.monotonic_ns()
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / CLI),
        "--worker",
        str(request),
        "--start",
        str(start),
    ]
    log = request.parent / f"process-{start}.log"
    timed_out = False
    progress("child_before", start, 64 - start)
    with (
        log.open("wb") as output,
        subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1"),
        ) as child,
    ):
        assert child.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - (time.monotonic_ns() - began) / 1e9
                if remaining <= 0:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                events = selector.select(min(60, remaining))
                if not events:
                    progress(f"child_elapsed_{(time.monotonic_ns() - began) / 1e9:.1f}")
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(child.stdout)
                    else:
                        output.write(chunk)
                        output.flush()
                        print(chunk.decode(errors="replace"), end="", flush=True)
        child.wait(timeout=5)
        output.flush()
        os.fsync(output.fileno())
    ended = time.monotonic_ns()
    response = request.parent / f"response-{start}.json"
    result = (
        json.loads(response.read_text()) if child.returncode == 0 and response.is_file() else {}
    )
    progress("child_after", start + 32, max(0, 32 - start))
    return dict(
        start=start,
        started_ns=began,
        ended_ns=ended,
        exit_code=child.returncode,
        timed_out=timed_out,
        status="completed" if result and not timed_out else "failed",
        result=result,
        log=reference(log),
        argv=argv,
    )


def execute(
    raw: Path,
    spec: Json,
    head: Json,
    preparation: bool,
    prepared: Json | None = None,
    deadline: float = 60,
) -> Json:
    """Time child startup through final exit; install prepared fixture bytes beforehand."""
    from carnot.memory.transactional_constraint_memory import _atomic_write

    raw.mkdir(parents=True, exist_ok=True)
    if prepared is not None:
        source = Path(prepared["segments"][0]["state_file"]["path"])
        _atomic_write(raw / "state.json", source.read_bytes())
    request = raw / "request.json"
    durable(request, dict(spec, head=head, preparation=preparation))
    began = time.monotonic_ns()
    commands, segments = [], []
    for start in (0,) if preparation else (0, 32):
        command = process(request, start, deadline)
        commands.append(command)
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
        status="completed" if complete else "failed",
        missing_reason=None if complete else "child_failed_or_deadline",
        metric=(ended - began) / (spec["occupancy"] or 1) if preparation else (ended - began) / 64,
        complete_ns=ended - began,
        parent_started_ns=began,
        parent_ended_ns=ended,
        commands=commands,
        segments=segments,
        preparation=preparation,
        query_count=sum(len(s["predictions"]) for s in segments),
        initial_occupancy=len(initial.get("templates", [])),
        final_occupancy=len(final.get("templates", [])),
        attempted_updates=len(releases),
        applied_updates=sum(r["coefficient_committed"] for r in releases),
        template_hits=sum(s["template_hits"] for s in segments),
        template_additions=sum(r["template_added"] for r in releases),
        capacity_rejections=sum(s["capacity_rejections"] for s in segments),
        fsync_counts={
            k: sum(s["fsync_counts"][k] for s in segments) for k in ("file", "directory")
        },
        serialized_state_bytes=[s["state_bytes"] for s in segments],
        response_bytes=[
            (raw / f"response-{c['start']}.json").stat().st_size
            for c in commands
            if c["status"] == "completed"
        ],
        request=reference(request),
        prepared_reference=reference(Path(prepared["segments"][0]["state_file"]["path"]))
        if prepared
        else None,
    )
    durable(raw / "window.json", row)
    return row


def prepare(raw: Path, spec: Json, head: Json) -> Json:
    """Warm only through the same verified delayed release calls used by service."""
    return execute(raw, spec, head, True)


def window(raw: Path, spec: Json, head: Json, prepared: Json, *, deadline: float = 60) -> Json:
    """Two fresh children include the midpoint restart and full delayed tail drain."""
    return execute(raw, spec, head, False, prepared, deadline)


def parity(row: Json) -> Json:
    """Check exact recovery, admission, delay, counts and complete observed clocks."""
    passed = row["status"] == "completed" and len(row["segments"]) == 2
    if passed:
        first, last = row["segments"]
        initial, final = first["initial_state"], last["state"]
        predictions = first["predictions"] + last["predictions"]
        releases = first["releases"] + last["releases"]
        combined = row["arm"] == "combined"
        passed = (
            row["query_count"] == len(predictions) == len(releases) == 64
            and len(initial["templates"]) == row["occupancy"] == row["initial_occupancy"]
            and row["final_occupancy"] == len(final["templates"]) <= 32
            and last["restart_input_hash"] == first["state_hash"]
            and not final["pending"]
            and len(final["acks"]) == row["occupancy"] + 64
            and final["acks"][-64:] == [f"e{1000 + i}" for i in range(64)]
            and row["attempted_updates"] == 64
            and row["applied_updates"] == 64 * combined
            and all(
                r["feedback_ordinal"] == r["prediction_ordinal"] + 8
                and r["coefficient_committed"] == combined
                and r["feedback_label"] == r["independently_verified_label"]
                for r in releases
            )
            and all(s["state_hash"] == canonical_hash(s["state"]) for s in row["segments"])
            and (combined or initial["parameters"] == final["parameters"])
            and row["fsync_counts"] == dict(file=130, directory=130)
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


def preparation_parity(row: Json) -> bool:
    """Require every warm template to have its own verified delayed admission."""
    if row["status"] != "completed":
        return False
    from carnot.memory.causal_memory_8802 import normalize

    segment = row["segments"][0]
    n = row["occupancy"]
    return bool(
        row["final_occupancy"] == n
        and segment["initial_state"]["templates"] == []
        and segment["state"]["templates"] == [normalize(payload(row["block"], i)) for i in range(n)]
        and segment["initial_state"]["parameters"] == segment["state"]["parameters"]
        and not segment["state"]["pending"]
        and [r["event_id"] for r in segment["releases"]] == [f"e{i}" for i in range(n)]
        and all(
            r["template_added"]
            and not r["coefficient_committed"]
            and r["feedback_ordinal"] == r["prediction_ordinal"] + 8
            and r["feedback_label"] == r["independently_verified_label"]
            for r in segment["releases"]
        )
        and segment["fsync_counts"] == dict(file=1 + 2 * n, directory=1 + 2 * n)
    )


def reduce_costs(rows: list[Json], preparations: list[Json]) -> Json:
    """Use paired block differences with six Bonferroni-corrected Student-t intervals."""
    lookup = {(r["block"], r["occupancy"], r["arm"]): r for r in rows if r["status"] == "completed"}
    intervals, occupancy_rows, absolute, amortization = [], [], [], []
    for arm in ("read_only", "combined"):
        for control, treatment in ((0, 8), (0, 32), (8, 32)):
            differences = [
                lookup[b, treatment, arm]["complete_ns"] - lookup[b, control, arm]["complete_ns"]
                for b in range(10)
                if (b, treatment, arm) in lookup and (b, control, arm) in lookup
            ]
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
            intervals.append(
                dict(
                    arm=arm,
                    control_occupancy=control,
                    treatment_occupancy=treatment,
                    paired_blocks=len(differences),
                    missing_pairs=10 - len(differences),
                    familywise_confidence_percent=99,
                    family_size=6,
                    method="Bonferroni paired Student-t; independent block differences assumed Gaussian",
                    mean_increment_ns=mean,
                    mean_increment_query_ns=mean / 64 if mean is not None else None,
                    lower_ns=mean - margin if margin is not None and mean is not None else None,
                    upper_ns=mean + margin if margin is not None and mean is not None else None,
                    raw_paired_differences_ns=differences,
                )
            )
        for occupancy in (0, 8, 32):
            selected = [
                r
                for r in rows
                if r["arm"] == arm and r["occupancy"] == occupancy and r["status"] == "completed"
            ]
            costs = [r["complete_ns"] for r in selected]
            mean_cost = float(np.mean(costs)) if costs else None
            occupancy_rows.append(
                dict(
                    arm=arm,
                    initial_occupancy=occupancy,
                    count=len(costs),
                    mean_window_ns=mean_cost,
                    mean_query_ns=mean_cost / 64 if mean_cost is not None else None,
                    p50_window_ns=float(np.quantile(costs, 0.5)) if costs else None,
                    p95_window_ns=float(np.quantile(costs, 0.95)) if costs else None,
                )
            )
            prep_costs = [
                r["complete_ns"]
                for r in preparations
                if r["occupancy"] == occupancy and r["status"] == "completed"
            ]
            for windows in (1, 10, 100):
                prep_mean = float(np.mean(prep_costs)) if prep_costs else None
                amortization.append(
                    dict(
                        arm=arm,
                        occupancy=occupancy,
                        windows=windows,
                        preparation_mean_ns=prep_mean,
                        preparation_per_window_ns=prep_mean / windows
                        if prep_mean is not None
                        else None,
                        amortized_query_ns=(mean_cost + prep_mean / windows) / 64
                        if mean_cost is not None and prep_mean is not None
                        else None,
                    )
                )
            absolute += [
                dict(
                    unit_id=r["unit_id"],
                    window_ns=r["complete_ns"],
                    query_ns=r["complete_ns"] / 64,
                    denominator=64,
                )
                for r in selected
            ]
    return dict(
        paired_cost_intervals=intervals,
        occupancy_rows=occupancy_rows,
        absolute_cost_rows=absolute,
        preparation_amortization_rows=amortization,
    )


def replay(path: Path) -> Json:
    """Rebuild the artifact from hash-bound work and reject rehashed aggregate changes."""
    try:
        from carnot.reporting.memory_size_runner_8848 import build

        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_hash_drift")
        work = json.loads(Path(value["primitive_reference"]["path"]).read_text())
        validation = json.loads(Path(value["validation_reference"]["path"]).read_text())
        if build(work, validation) != value:
            raise ValueError("reduction_drift")
        for row in work["window_rows"] + work["pilot_window_rows"]:
            if row["status"] == "completed" and not parity(row)["passed"]:
                raise ValueError("lifecycle_drift")
            for segment in row["segments"]:
                if (
                    sha256_file(Path(segment["state_file"]["path"]))
                    != segment["state_file"]["sha256"]
                ):
                    raise ValueError("state_drift")
        if any(
            not preparation_parity(r)
            for r in work["preparation_cost_rows"]
            if r["status"] == "completed"
        ):
            raise ValueError("preparation_drift")
        return dict(passed=True, replay_passed=True, windows=len(work["window_rows"]))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="replay_operand_or_reduction_failure")
