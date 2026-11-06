"""REQ-REPORT-8835: measure whole store operations without internal attribution."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import selectors
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8835_v756_complete_adaptation_cost"
TASK = "exp8835-complete-adaptation-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/complete_adaptation_cost_8835.py",
    "python/carnot/reporting/complete_adaptation_runner_8835.py",
    CLI,
]
TESTS = ["tests/python/test_complete_adaptation_cost_8835.py"]
ARMS = ("read_only", "calibration_only", "combined")
SEED = 7568835
PRIMARY = "results/experiment_8802_v754_causal_memory_prototype.json"
PIN = "sha256:ddbdbb66aa7118e6a03bb702d9f51e91a346337d1d89fa7befc9469ef1b8d497"
STORE = "python/carnot/memory/causal_memory_8802.py"
STORE_PIN = "sha256:8a9d7c5242199fd4f8cb82b892fb4aef3e80b1d68fa909dc0ca96d6c5063a3c8"
DESIGN = "openspec/change-proposals/research-roadmap-v755-preserved-20261104.md"
DESIGN_PIN = "sha256:90a2b04b863a04a452035e88efb9c5634297f8777ed8469f8852a3649895a3d8"
BOUNDARY = "Parent before process startup/imports through 64 queries, eight-query delayed feedback, restart at32, full tail drain, typed response serialization, file and directory fsync and final child exit. Only coarse parent service halves; no component clocks."


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Counts keep a slow child distinguishable from a stopped producer."""
    print(f"[exp8835] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Exact bytes bind a replay to the operand actually read."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def authenticate(path: Path, pin: str, field: str) -> tuple[Json, Json | None]:
    """Missing fields and observed zero require different upstream repairs."""
    value: Json = {}
    observed = "missing_file"
    digest = None
    if path.is_file():
        digest = sha256_file(path)
        try:
            value = json.loads(path.read_text())
            observed = (
                "rejected_producer"
                if digest != pin
                else "missing_field"
                if field not in value
                else "observed_zero"
                if value[field] == 0
                else "rejected_producer"
                if value[field] != 1
                or value.get("required_checks_passed") is not True
                or value.get("verdict_class") in {"blocked", "disqualified"}
                else "passed"
            )
        except (ValueError, TypeError):
            observed = "rejected_producer"
    failure = dict(
        check="immutable_producer_admission",
        upstream="exp8802-causal-memory-prototype",
        artifact_field=field,
        expected=1,
        observed=observed,
        path=str(path),
        sha256=digest,
        expected_sha256=pin,
    )
    return value, None if observed == "passed" else failure


def preflight(raw: Path) -> Json:
    """Authenticate historical controls before spending time on numerical work."""
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "writable.json", {"private": True})
    value, rejected = authenticate(ROOT / PRIMARY, PIN, "memory_execution_ready_score")
    failures = [rejected] if rejected else []
    refs = []
    if not rejected:
        refs = [
            reference(ROOT / PRIMARY),
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
            dict(path=str(ROOT / DESIGN), sha256=DESIGN_PIN),
        ]
        for ref in refs:
            path = Path(ref["path"])
            if not path.is_file() or sha256_file(path) != ref["sha256"]:
                failures.append(
                    dict(
                        check="immutable_operand",
                        upstream=TASK,
                        artifact_field="source_artifact_hashes",
                        expected=ref["sha256"],
                        observed="missing_file" if not path.is_file() else "rejected_producer",
                        path=str(path),
                        sha256=sha256_file(path) if path.is_file() else None,
                    )
                )
        primitive = json.loads(Path(refs[1]["path"]).read_text()) if not failures else {}
        if primitive and not all(
            value[k] == primitive[k]
            for k in ("rows", "restart_rows", "leakage_controls", "calibrated_head_parameters")
        ):
            failures.append(
                dict(
                    check="original_controls",
                    upstream=TASK,
                    artifact_field="rows",
                    expected="original qualified control bytes",
                    observed="rejected_producer",
                    path=str(ROOT / PRIMARY),
                    sha256=PIN,
                )
            )
    optional = ROOT / "results/experiment_8830_v756_categorical_memory.json"
    candidate = json.loads(optional.read_text()) if optional.is_file() else {}
    # This scheduled execution freezes the available, rejected producer explicitly.
    optional_ready = (
        candidate.get("memory_execution_ready_score") == 1
        and candidate.get("required_checks_passed") is True
        and candidate.get("verdict_class") not in {"blocked", "disqualified"}
    )
    progress("cpu_dependency_check_before")
    import jax

    checks = dict(
        repository_virtualenv=(ROOT / ".venv/bin/python").is_file(),
        private_output=json.loads((raw / "writable.json").read_text())["private"],
        free_disk_bytes=shutil.disk_usage(raw).free,
        minimum_free_bytes=128 * 1024**2,
        numpy_version=np.__version__,
        cpu_platform=os.environ.get("JAX_PLATFORMS", "cpu"),
        jax_version=jax.__version__,
        cpu_devices=[str(device) for device in jax.devices("cpu")],
    )
    progress("cpu_dependency_check_after", 1, 0)
    return dict(
        failures=failures,
        sources=refs,
        head=value.get("calibrated_head_parameters", [{}])[0],
        internal_failure=any(
            f["check"] == "original_controls"
            or (f["observed"] == "rejected_producer" and "/python/" in f["path"])
            for f in failures
        ),
        checks=checks,
        arm_admission_rows=[
            dict(arm="historical_v754", admitted=not failures, sha256=STORE_PIN),
            dict(
                arm="categorical_optional",
                admitted=False,
                qualified_at_seal=optional_ready,
                path=str(optional),
                sha256=sha256_file(optional) if optional.is_file() else None,
                missing_reason="rejected_producer" if candidate else "missing_file",
            ),
        ],
    )


def roster() -> list[Json]:
    """Randomize whole matched blocks before any timing can select their order."""
    rng = random.Random(SEED)
    blocks = list(range(30))
    rng.shuffle(blocks)
    rows = []
    for block in blocks:
        arms = list(ARMS)
        rng.shuffle(arms)
        for position, arm in enumerate(arms):
            rows.append(
                dict(
                    unit_id=f"block-{block:02}:{arm}",
                    cluster_id=f"stream-{block:02}",
                    block=block,
                    arm=arm,
                    seed=11,
                    process_order=len(rows),
                    order_in_block=position,
                    denominator=64,
                )
            )
    return rows


def durable(path: Path, value: Json) -> None:
    """A final directory barrier includes rename durability in the operation."""
    atomic_json(path, value)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def worker(request: Path, start: int) -> None:
    """Run unchanged store methods; the parent owns every clock observation."""
    progress("numerical_import_before")
    from carnot.memory import causal_memory_8802 as store_module

    progress("numerical_import_after")
    spec = json.loads(request.read_text())
    head = spec["head"]
    state_path = request.parent / "state.json"
    before = json.loads(state_path.read_text()) if state_path.exists() else None
    store = store_module.Store(state_path, head["parameters"])
    print(json.dumps(dict(protocol="ready", start=start)), flush=True)
    if sys.stdin.readline().strip() != "run":
        raise ValueError("parent_start_required")
    predictions, releases = [], []

    def payload(i: int) -> Json:
        return store_module.world(f"stream-{spec['block']}-world-{i}", store_module.LABELS[i % 3])

    for i in range(start, start + 32):
        predictions.append(store.predict(f"e{i}", payload(i), head["threshold"]))
        store.close()
        if i >= 8:
            releases.append(store.release(f"e{i - 8}", payload(i - 8), i, spec["arm"]))
        if (i + 1) % 16 == 0:
            progress("queries", i + 1, 64 - i - 1)
    if start == 32:
        for i in range(56, 64):
            releases.append(store.release(f"e{i}", payload(i), i + 8, spec["arm"]))
    snapshot = request.parent / f"state-{start}.json"
    durable(snapshot, store.state)
    result = dict(
        start=start,
        predictions=predictions,
        releases=releases,
        state=store.state,
        restart_input_hash=canonical_hash(before) if before is not None else None,
        state_hash=canonical_hash(store.state),
        state_file=reference(snapshot),
    )
    durable(request.parent / f"response-{start}.json", result)
    print(
        json.dumps(
            dict(protocol="result", response=reference(request.parent / f"response-{start}.json"))
        ),
        flush=True,
    )


def process(request: Path, start: int, deadline: float) -> Json:
    """Forward child output while enforcing a fixed deadline, independent of heartbeats."""
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
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
    progress("process_before", start, 64 - start)
    service_begin = service_end = None
    result: Json = {}
    timed_out = False
    log = request.parent / f"process-{start}.log"
    with (
        log.open("wb") as output,
        subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        ) as child,
    ):
        assert child.stdout is not None and child.stdin is not None
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            buffer = b""
            while selector.get_map():
                remaining = deadline - (time.monotonic_ns() - began) / 1e9
                if remaining <= 0:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                events = selector.select(timeout=min(60, remaining))
                if not events:
                    progress(
                        f"process_heartbeat_elapsed_{(time.monotonic_ns() - began) / 1e9:.1f}",
                        start,
                        64 - start,
                    )
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(child.stdout)
                        continue
                    output.write(chunk)
                    output.flush()
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        print(line.decode(errors="replace"), flush=True)
                        if line.startswith(b'{"protocol":'):
                            message = json.loads(line)
                            if message["protocol"] == "ready":
                                service_begin = time.monotonic_ns()
                                child.stdin.write(b"run\n")
                                child.stdin.flush()
                            if message["protocol"] == "result":
                                service_end = time.monotonic_ns()
                                result = json.loads(Path(message["response"]["path"]).read_text())
        child.wait(timeout=5)
        output.flush()
        os.fsync(output.fileno())
    ended = time.monotonic_ns()
    progress("process_after", start + 32, 32 - start)
    return dict(
        start=start,
        started_monotonic_ns=began,
        ended_monotonic_ns=ended,
        service_started_ns=service_begin,
        service_ended_ns=service_end,
        status="completed" if child.returncode == 0 and result and not timed_out else "failed",
        exit_code=child.returncode,
        timed_out=timed_out,
        result=result,
        log=reference(log),
        argv=argv,
    )


def window(raw: Path, spec: Json, head: Json, *, deadline: float = 60) -> Json:
    """Surround both fresh processes, restart, responses and durable state with one clock."""
    raw.mkdir(parents=True, exist_ok=True)
    request = raw / "request.json"
    durable(request, dict(spec, head=head))
    began = time.monotonic_ns()
    commands, segments = [], []
    for start in (0, 32):
        command = process(request, start, deadline)
        commands.append(command)
        if command["status"] != "completed":
            break
        segments.append(command["result"])
    ended = time.monotonic_ns()
    complete = len(segments) == 2
    row = dict(
        spec,
        status="completed" if complete else "failed",
        missing_reason=None if complete else "child_failed_or_deadline",
        metric=(ended - began) / 64,
        complete_ns=ended - began,
        service_ns=sum(
            c["service_ended_ns"] - c["service_started_ns"]
            for c in commands
            if c["service_ended_ns"] is not None
        ),
        parent_started_ns=began,
        parent_ended_ns=ended,
        commands=commands,
        segments=segments,
        request=reference(request),
        initial_parameter_hash=canonical_hash(head["parameters"]),
    )
    durable(raw / "window.json", row)
    return row


def parity(row: Json) -> Json:
    """Check the entire lifecycle, rather than treating an incremented counter as proof."""
    segments = row["segments"]
    passed = row["status"] == "completed" and len(segments) == 2
    if passed:
        first, last = segments
        state = last["state"]
        predictions = first["predictions"] + last["predictions"]
        releases = first["releases"] + last["releases"]
        events = [f"e{i}" for i in range(64)]
        passed = (
            last["restart_input_hash"] == first["state_hash"]
            and [p["event_id"] for p in predictions] == events
            and [r["event_id"] for r in releases] == events == state["acks"]
            and not state["pending"]
            and len(state["released"]) == 64
            and all(
                r["feedback_ordinal"] >= r["prediction_ordinal"] + 8
                and r["optimizer_steps"] == 1
                and r["feedback_label"] == r["independently_verified_label"]
                and r["coefficient_committed"] == (row["arm"] != "read_only")
                for r in releases
            )
            and (canonical_hash(state["parameters"]) == row["initial_parameter_hash"])
            == (row["arm"] == "read_only")
            and bool(state["templates"]) == (row["arm"] == "combined")
            and all(s["state_hash"] == canonical_hash(s["state"]) for s in segments)
            and row["complete_ns"] == row["parent_ended_ns"] - row["parent_started_ns"]
            and row["complete_ns"] >= row["service_ns"] >= 0
            and all(
                row["parent_started_ns"]
                <= c["started_monotonic_ns"]
                <= c["service_started_ns"]
                <= c["service_ended_ns"]
                <= c["ended_monotonic_ns"]
                <= row["parent_ended_ns"]
                for c in row["commands"]
            )
        )
    return dict(unit_id=row["unit_id"], passed=bool(passed))


def reduce_costs(rows: list[Json]) -> Json:
    """Bootstrap matched stream blocks; signed differences are interventions, not fractions."""
    lookup = {(r["block"], r["arm"]): r for r in rows if r["status"] == "completed"}
    intervals = []
    for treatment, control in [("calibration_only", "read_only"), ("combined", "calibration_only")]:
        differences = [
            lookup[b, treatment]["complete_ns"] - lookup[b, control]["complete_ns"]
            for b in range(30)
            if (b, treatment) in lookup and (b, control) in lookup
        ]
        estimates = (
            np.random.default_rng(SEED).choice(differences, (10000, len(differences))).mean(axis=1)
            if differences
            else None
        )
        intervals.append(
            dict(
                treatment=treatment,
                control=control,
                paired_blocks=len(differences),
                missing_pairs=30 - len(differences),
                confidence_percent=99,
                mean_increment_ns=float(np.mean(differences)) if differences else None,
                mean_increment_query_ns=float(np.mean(differences)) / 64 if differences else None,
                lower99_ns=float(np.quantile(estimates, 0.005)) if estimates is not None else None,
                upper99_ns=float(np.quantile(estimates, 0.995)) if estimates is not None else None,
                raw_paired_differences_ns=differences,
                scope="constructed_stream_block_resampling_only",
            )
        )
    absolute, order = [], []
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm and r["status"] == "completed"]
        for position in range(3):
            group = [r["complete_ns"] for r in selected if r["order_in_block"] == position]
            order.append(
                dict(
                    arm=arm,
                    order_in_block=position,
                    count=len(group),
                    mean_window_ns=float(np.mean(group)) if group else None,
                )
            )
        for r in selected:
            absolute.append(
                dict(
                    unit_id=r["unit_id"],
                    arm=arm,
                    window_ns=r["complete_ns"],
                    complete_query_ns=r["complete_ns"] / 64,
                    service_query_ns=r["service_ns"] / 64,
                    future_latency_requirement_ns=r["complete_ns"] / 64 / 100,
                    measured_speedup=None,
                )
            )
    return dict(
        paired_cost_intervals=intervals, absolute_cost_rows=absolute, process_order_effects=order
    )


def replay(path: Path) -> Json:
    """Rebuild reductions and readiness only from bound operands and real validation logs."""
    try:
        from carnot.reporting.complete_adaptation_runner_8835 import build

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
        expected = build(work, validation)
        if value != expected:
            raise ValueError("reduction_drift")
        for row in work["window_rows"] + work["pilot_window_rows"]:
            if row["status"] == "completed":
                if not parity(row)["passed"]:
                    raise ValueError("lifecycle_drift")
                for segment in row["segments"]:
                    if (
                        sha256_file(Path(segment["state_file"]["path"]))
                        != segment["state_file"]["sha256"]
                    ):
                        raise ValueError("state_drift")
        return dict(passed=True, replay_passed=True, windows=len(work["window_rows"]))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="replay_operand_or_reduction_failure")
