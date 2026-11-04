"""REQ-VERIFY-8784: sealed durable children and exact-byte terminal publication."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import durable_cost_components_8784 as e
from carnot.reporting import durable_component_clocks_8784 as clocks
from carnot.reporting import ordinary_durable_runner_8772 as historical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
runtime = namespace(historical, "components8784_runner")
runtime.e = e.measurement
runtime.E2E = ["tests/python/test_primary_publication_7928.py", *historical.PRIVATE_E2E]
base_measure, base_replay = runtime.measure, runtime.replay


def worker(request: Json, folder: Path) -> Json:
    """Run the unchanged durable driver with optional direct coarse block clocks."""
    numeric = clocks.numeric
    arm, start, stop = request["arm"], request["start"], request["stop"]
    extension = None
    if arm != "python":
        e.progress("native_load_before")
        extension = numeric.resident.load(Path(request["extension"]["path"]))
        if e.reference(Path(extension.__file__))["sha256"] != request["extension"]["sha256"]:
            raise ValueError("loaded_binary_identity")
        e.progress("native_load_after")
    configuration = dict(capacity=8, job_hash=canonical_hash(request["job"]))
    store = numeric.store_module.Store(folder / "store", configuration)
    if start and not request.get("resume", False):
        saved = request["snapshot"]
        store.commit(
            dict(
                schema=numeric.store_module.SCHEMA,
                configuration_hash=canonical_hash(configuration),
                pending=[],
                cursor=start,
                head=saved["state"],
                counts=saved["counts"],
                operations=saved["operations"],
                maximum=saved["maximum"],
                phase="acknowledged",
                version=2 * start - 1,
            )
        )
    imported = store.read() if start else None
    spans: list[Any] = []
    sync_count = 0
    real_fsync = os.fsync

    def sync(fd: int) -> None:
        """Count successful barriers equally in both arms without timing each wrapper."""
        nonlocal sync_count
        real_fsync(fd)
        sync_count += 1

    begin = time.perf_counter_ns()
    with ExitStack() as stack:
        stack.enter_context(patch.object(os, "fsync", sync))
        if request.get("condition") == "instrumented":
            clocks.install(stack, arm, spans)
        result: Json = numeric.drive(request["job"], folder, arm, start, stop, extension)
    verified = store.read()
    if verified != result["checkpoint"]:
        raise ValueError("post_restart_verification")
    end = time.perf_counter_ns()
    result.update(
        fsync_calls=sync_count,
        start=start,
        stop=stop,
        begin_ns=begin,
        end_ns=end,
        imported_checkpoint_hash=canonical_hash(imported),
        final_checkpoint_hash=canonical_hash(verified),
        spans=spans,
        components=clocks.partition(spans, begin, end),
        rows=e.read(e.reference(Path(result["rows_path"])))["rows"],
    )
    for row in result["rows"]:
        row.update(
            world_count=1,
            workload_count=1,
            denominator_updates=1,
            arm=arm,
            condition=request.get("condition", "plain"),
        )
    atomic_json(folder / "result.json", result)
    return result


def cell(spec: Json, admitted: Json, raw: Path) -> Json:
    """Every window spans two fresh processes, including post-restart verification."""
    begin = time.perf_counter_ns()
    folder = raw / "cells" / spec["unit_id"].replace(":", "-")
    folder.mkdir(parents=True)
    request = e.historical.request(admitted, spec["offset"], spec["arm"])
    segments, commands, updates = [], [], []
    for part in range(2):
        start = spec["offset"] + 16 * part
        path = folder / f"request-{part}.json"
        atomic_json(
            path,
            dict(
                request, start=start, stop=start + 16, resume=part > 0, condition=spec["condition"]
            ),
        )
        receipt = child(
            f"segment-{part}",
            [
                sys.executable,
                "-u",
                e.CLI,
                "--worker",
                str(path),
                "--work-dir",
                str(folder / "work"),
            ],
            folder / "logs",
            deadline=120,
        )
        commands.append(receipt)
        if not receipt["passed"]:
            break
        summary = e.read(e.reference(folder / "work/result.json"))
        e.historical.check_window(summary, 16)
        summary["components"]["costs_ns"].update(
            process_startup=summary["begin_ns"] - receipt["started_monotonic_ns"],
            process_shutdown=receipt["ended_monotonic_ns"] - summary["end_ns"],
        )
        updates += summary["rows"]
        segments.append(summary)
        atomic_json(folder / f"segment-{part}.json", summary)
    restart = (
        len(segments) == 2
        and segments[1]["imported_checkpoint_hash"] == segments[0]["final_checkpoint_hash"]
    )
    expected = {r["update_index"]: r for r in admitted["trace"]}
    for row in updates:
        old = expected[row["update_index"]]
        row["passed"] = (
            row["state_hash"] == old["state_hash"]
            and row["probability_after"] == old["probability"]
            and row["renormalization_writes"] == old["dense_writes"]
        )
        row.update(unit_id=spec["unit_id"], repeat=spec["repeat"], offset=spec["offset"])
    complete = restart and len(updates) == 32 and all(u["passed"] for u in updates)
    complete_ns = time.perf_counter_ns() - begin
    for row in updates:
        row["complete_window_amortized_ns"] = complete_ns / 32 if complete else None
    for index in range(spec["offset"] + len(updates) + 1, spec["offset"] + 33):
        updates.append(
            dict(
                update_index=index,
                unit_id=spec["unit_id"],
                status="censored",
                passed=False,
                censored=True,
                excluded=False,
                latency_ns=None,
                arm=spec["arm"],
                condition=spec["condition"],
                world_count=1,
                workload_count=1,
            )
        )
    return dict(
        spec,
        status="completed" if complete else "failed",
        passed=complete,
        censored=False,
        excluded=False,
        complete_ns=complete_ns if complete else None,
        observed_updates=sum(u["status"] == "completed" for u in updates),
        update_rows=updates,
        segments=segments,
        commands=commands,
        restart_verified=restart,
        world_count=1,
        workload_count=1,
        denominator_updates=32,
    )


def seal(work: Json) -> None:
    """An independent seal binds primitive bytes before the report is built."""
    raw = Path(work["raw"])
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )


def measure(raw: Path, date: str, private_input: Path | None = None) -> Json:
    """Freeze private fixtures or authenticated real operands before the bounded probe."""
    admitted = e.read(e.reference(private_input)) if private_input else e.inputs()
    if not admitted["failures"]:
        frozen = raw.with_name(raw.name + "-task-contract.md")
        frozen.write_bytes((e.ROOT / e.identity.contract.DESIGN).read_bytes())
        admitted["refs"] = [
            dict(ref, path=str(frozen))
            if ref["path"] == str(e.ROOT / e.identity.contract.DESIGN)
            else ref
            for ref in admitted["refs"]
        ]
    control_roster = admitted.get("private_roster", e.roster()) if private_input else e.roster()
    with (
        patch.object(runtime.e, "inputs", lambda: admitted),
        patch.object(runtime.e, "roster", lambda: control_roster),
    ):
        work: Json = base_measure(raw, date)
    work["private_control"] = private_input is not None
    seal(work)
    return work


def replay(path: Path) -> None:
    """Original independent seals and source checks precede deterministic reduction."""
    value = json.loads(path.read_bytes())
    if any(value.get(k) != v for k, v in e.descriptor(value["invocation_id"]).items()):
        raise ValueError("producer_identity")
    work = e.read(value["primitive_reference"])
    with patch.object(
        runtime.e, "roster", lambda: work["roster"] if work["private_control"] else e.roster()
    ):
        base_replay(path)


def run(
    output: Path, date: str, *, private: bool = False, private_input: Path | None = None
) -> int:
    """Keep private controls outside the repository and global health outside acceptance."""
    if private and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private_output")
    if private != (private_input is not None):
        raise ValueError("private_input_required")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.parent.mkdir(parents=True, exist_ok=True)
    work = measure(raw, date, private_input)
    receipts = (
        [dict(name="private_control_only", passed=True)] if private else runtime.validate(raw)
    )
    if not private:
        health = child(
            "full_python_suite",
            [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            raw / "repository_health",
            deadline=180,
            scope="repository_health",
        )
        work["repository_health_reference"] = dict(
            current_receipt=health, full_suite_pass_claimed=False
        )
        seal(work)
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json")))
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: runtime.terminal(candidate, raw / "terminal")
    )
    atomic_json(raw / "terminal_validation.json", dict(publication=publication, passed=True))
    e.progress(value["honest_verdict"], value["completed_count"], value["censored_count"])
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """One thin CLI exposes real workers, private fixture controls and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261104")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-input", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if not args.work_dir:
                raise ValueError("worker_work_dir")
            worker(e.read(e.reference(args.worker)), args.work_dir)
            return 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if args.date != "20261104":
            raise ValueError("calendar_date")
        return run(
            args.output,
            args.date,
            private=args.private_input is not None,
            private_input=args.private_input,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8784] error={error}", file=sys.stderr, flush=True)
        return 1


runtime.cell = cell
