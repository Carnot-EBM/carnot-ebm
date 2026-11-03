"""REQ-VERIFY-8772: qualify storage before the fixed ordinary durable workload."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import ordinary_durable_cost_8772 as e
from carnot.reporting import bounded_durable_runner_8745 as previous
from carnot.reporting import v743_capstone as checks
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.lean_runner_8608 import COMMON, E2E as PRIVATE_E2E

Json = dict[str, Any]
E2E = [
    "tests/python/test_resident_sparse_8731.py",
    "tests/python/test_source_boundary_7852.py",
    *PRIVATE_E2E,
]


def worker(request: Json, folder: Path) -> Json:
    """Count real file and directory fsync calls around the unchanged shipped store."""
    numeric = e.previous.previous.numerical
    arm, start, stop = request["arm"], request["start"], request["stop"]
    extension = None
    if arm != "python":
        e.progress("native_load_before")
        extension = numeric.resident.load(Path(request["extension"]["path"]))
        if e.reference(Path(extension.__file__))["sha256"] != request["extension"]["sha256"]:
            raise ValueError("loaded_binary_identity")
        e.progress("native_load_after")
    calls, arithmetic = [], []
    real_fsync = os.fsync
    owner = (
        numeric.numeric
        if arm == "python"
        else numeric.resident.original
        if arm == "json_native"
        else numeric.resident.engine
    )
    original_step = owner.step

    def sync(fd: int) -> None:
        """Only successfully returned system calls count as durable barriers."""
        real_fsync(fd)
        calls.append(fd)

    def step(*args: Any) -> Any:
        """Retain arithmetic timing separately without removing persistence work."""
        begin = time.perf_counter_ns()
        result = original_step(*args)
        arithmetic.append(time.perf_counter_ns() - begin)
        e.progress(
            "durable_update_arithmetic", start + len(arithmetic), stop - start - len(arithmetic)
        )
        return result

    folder.mkdir(parents=True, exist_ok=True)
    with patch.object(os, "fsync", sync), patch.object(owner, "step", step):
        if start and not request.get("resume", False):
            saved, job = request["snapshot"], request["job"]
            configuration = dict(capacity=8, job_hash=canonical_hash(job))
            store = numeric.store_module.Store(folder / "store", configuration)
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
        result: Json = numeric.drive(request["job"], folder, arm, start, stop, extension)
        result["fsync_calls"] = len(calls)
    e.check_window(result, stop - start)
    result["rows"] = e.read(e.reference(Path(result["rows_path"])))["rows"]
    for row, cost in zip(result["rows"], arithmetic, strict=True):
        row["arithmetic_ns"] = cost
    atomic_json(folder / "result.json", result)
    return result


def cell(spec: Json, admitted: Json, raw: Path) -> Json:
    """A middle window spans two child lifetimes and one durable store."""
    begin = time.perf_counter_ns()
    folder = raw / "cells" / spec["unit_id"].replace(":", "-")
    folder.mkdir(parents=True)
    request = e.request(admitted, spec["offset"], spec["arm"])
    boundaries = [
        request["start"],
        *([2048] if request["start"] < 2048 < request["stop"] else []),
        request["stop"],
    ]
    segments, commands, updates = [], [], []
    for part, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        path = folder / f"request-{part}.json"
        atomic_json(path, dict(request, start=start, stop=stop, resume=part > 0))
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
        e.check_window(summary, stop - start)
        segments.append(summary)
        updates += summary["rows"]
        atomic_json(folder / f"segment-{part}.json", summary)
    complete = len(updates) == 32 and all(c["passed"] for c in commands)
    latencies = [u["latency_ns"] for u in updates]
    rows = [
        dict(
            u,
            unit_id=spec["unit_id"],
            arm=spec["arm"],
            repeat=spec["repeat"],
            offset=spec["offset"],
            censored=False,
            excluded=False,
        )
        for u in updates
    ]
    rows += [
        dict(
            update_index=i,
            unit_id=spec["unit_id"],
            arm=spec["arm"],
            repeat=spec["repeat"],
            offset=spec["offset"],
            status="censored",
            passed=False,
            censored=True,
            excluded=False,
            latency_ns=None,
            arithmetic_ns=None,
        )
        for i in range(spec["offset"] + len(updates) + 1, spec["offset"] + 33)
    ]
    return dict(
        spec,
        status="completed" if complete else "failed",
        passed=complete,
        censored=False,
        excluded=False,
        complete_ns=time.perf_counter_ns() - begin if complete else None,
        p95_ns=float(np.quantile(latencies, 0.95)) if complete else None,
        observed_updates=len(updates),
        update_rows=rows,
        segments=segments,
        commands=commands,
    )


def measure(raw: Path, date: str) -> Json:
    """Freeze all cells and exact test nodes before the real durable admission probe."""
    begin, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True)
    pre = resources(raw)
    e.progress("authenticate_before")
    admitted = e.inputs()
    failures = list(admitted["failures"])
    bootstrap = dict(draws=np.random.default_rng(e.SEED).integers(0, 10, size=(10000, 10)).tolist())
    atomic_json(raw / "bootstrap.json", bootstrap)
    collection = child(
        "frozen_nodes",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *e.TESTS, *E2E],
        raw / "plan",
        deadline=90,
    )
    nodes = [
        s.strip()
        for s in Path(collection["stdout_path"]).read_text().splitlines()
        if s.startswith("tests/") and "::" in s
    ]
    if not collection["passed"] or not nodes:
        raise ValueError("premeasurement_node_collection")
    code = [
        e.reference(e.ROOT / p)
        for p in [
            *e.OWNED,
            *e.TESTS,
            *(suite.split("::")[0] for suite in E2E),
            "python/carnot/reporting/bounded_durable_cost_8745.py",
            "python/carnot/reporting/v743_capstone.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/verify/resident_update_cost_8732.py",
            "python/carnot/verify/block_feedback_state_8433.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
            "ops/e2e-test-plan.md",
            str(Path(sys.executable).resolve()),
        ]
    ]
    atomic_json(
        raw / "premeasurement_seal.json",
        dict(
            method=e.METHOD,
            roster=e.roster(),
            inputs=admitted["refs"],
            code=code,
            nodes=nodes,
            validation_suites=e.TESTS + E2E,
            bootstrap=e.reference(raw / "bootstrap.json"),
        ),
    )
    pre.update(
        file_fsync=False,
        atomic_replace=False,
        directory_fsync=False,
        no_model_load=True,
        MODEL_SPECS=[],
        python_version=sys.version,
        numpy_version=np.__version__,
    )
    temporary = raw / "storage.tmp"
    with temporary.open("wb") as stream:
        stream.write(b"ordinary durable storage probe")
        stream.flush()
        os.fsync(stream.fileno())
    pre["file_fsync"] = True
    temporary.replace(raw / "storage_probe")
    pre["atomic_replace"] = True
    fd = os.open(raw, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    pre["directory_fsync"] = True
    if not pre["disk_backed"] or not pre["tools_available"]:
        failures.append(
            e.failure(
                e.reference(raw / "storage_probe"),
                "persistent_storage_and_dependencies",
                True,
                False,
            )
        )
    starts: list[Json] = []
    costs: list[Json] = []
    probe: Json = {}
    projected: Json = {}
    e.progress("sources_plan_frozen", 0, 90)
    if not failures:
        starts, costs, _ = e.starting_states(admitted, raw)
        path = raw / "probe-request.json"
        atomic_json(path, e.request(admitted, 0, "python", 4))
        receipt = child(
            "durable_probe",
            [sys.executable, "-u", e.CLI, "--worker", str(path), "--work-dir", str(raw / "probe")],
            raw / "probe_logs",
            deadline=120,
        )
        if not receipt["passed"]:
            raise ValueError("durable_probe_failed")
        probe = e.read(e.reference(raw / "probe/result.json"))
        e.check_window(probe, 4)
        projected = e.projection(probe["latencies_ns"], round(receipt["duration_s"] * 1e9))
        projected["projected_with_reserve_s"] += (time.monotonic_ns() - begin) / 1e9 * 1.25
        projected["admitted"] = projected["projected_with_reserve_s"] <= 3000
        probe["process_receipt"] = receipt
        atomic_json(raw / "resource_projection.json", projected)
        if not projected["admitted"]:
            failures.append(
                dict(
                    e.failure(
                        e.reference(raw / "resource_projection.json"),
                        "projected_with_reserve_s",
                        3000,
                        projected["projected_with_reserve_s"],
                        "durability_resource_budget_exceeded",
                        "<=",
                    ),
                    upstream="host_disk_durability",
                )
            )
    rows = []
    for index, spec in enumerate(e.roster()):
        e.progress("window_before", index, 90 - index)
        if failures:
            updates = [
                dict(
                    update_index=i,
                    unit_id=spec["unit_id"],
                    arm=spec["arm"],
                    repeat=spec["repeat"],
                    offset=spec["offset"],
                    status="censored",
                    passed=False,
                    censored=True,
                    excluded=False,
                    latency_ns=None,
                    arithmetic_ns=None,
                )
                for i in range(spec["offset"] + 1, spec["offset"] + 33)
            ]
            row = dict(
                spec,
                status="censored",
                passed=False,
                censored=True,
                excluded=False,
                complete_ns=None,
                p95_ns=None,
                observed_updates=0,
                update_rows=updates,
                segments=[],
                commands=[],
            )
        else:
            row = cell(spec, admitted, raw)
        rows.append(row)
        atomic_json(raw / "window_rows.json", dict(rows=rows))
        e.progress("window_after", index + 1, 89 - index)
    end = time.monotonic_ns()
    work = dict(
        raw=str(raw),
        run_date=date,
        roster=e.roster(),
        failures=failures,
        window_rows=rows,
        starting_state_rows=starts,
        excluded_prefix_cost_rows=costs,
        durability_probe=probe,
        resource_projection=projected,
        historical_probe=admitted.get("historical_probe", {}),
        loaded_extension_reference=admitted.get("extension", {}),
        preconditions=pre,
        source_refs=admitted["refs"],
        code_refs=code,
        raw_refs=[e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
        start_ns=begin,
        end_ns=end,
        duration_s=(end - begin) / 1e9,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_prepare_probe_and_windows",
                start_ns=begin,
                end_ns=end,
                duration_s=(end - begin) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )
    return work


def replay(path: Path) -> None:
    """Independent seals reject changed primitives even if the candidate is rehashed."""
    value = json.loads(path.read_bytes())
    raw = (
        Path(value["publication_output"]).parent
        / "raw"
        / e.NAME
        / "invocations"
        / value["invocation_id"]
    )
    if (
        e.read(e.reference(raw / "measurement_seal.json"))["measurement"]
        != value["primitive_reference"]
    ):
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    for ref in work["source_refs"] + work["code_refs"] + work["raw_refs"]:
        e.verify(ref)
    frozen = e.read(e.reference(raw / "premeasurement_seal.json"))
    if (
        work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
        or frozen["method"] != e.METHOD
        or frozen["roster"] != e.roster()
        or frozen["code"] != work["code_refs"]
    ):
        raise ValueError("clock_or_method_drift")
    if e.read(e.reference(raw / "window_rows.json"))["rows"] != work["window_rows"]:
        raise ValueError("raw_window_drift")
    if (
        e.read(e.reference(raw / "validation_seal.json"))["validation"]
        != value["validation_reference"]
    ):
        raise ValueError("independent_validation_seal")
    receipts = e.read(value["validation_reference"])["receipts"]
    for receipt in receipts:
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                e.verify(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed")


def terminal(path: Path, raw: Path) -> Json:
    """Existing terminal scanners inspect exactly the bytes that will be published."""
    with patch("carnot.reporting.resident_update_runner_8732.e", e):
        return dict(previous.old_terminal(path, raw))


def validate(raw: Path) -> list[Json]:
    """Keep real disjoint node, JUnit and subprocess SQLite custody unchanged."""
    with patch.multiple(checks, e=e, E2E=E2E):
        return list(checks.validate(raw))


def run(output: Path, date: str, *, private: bool = False) -> int:
    """Public acceptance excludes the separate bounded repository health diagnostic."""
    if private and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private_output")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date)
    receipts = [dict(name="private_control_only", passed=True)] if private else validate(raw)
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
        atomic_json(raw / "measurement.json", work)
        atomic_json(
            raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
        )
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json")))
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    atomic_json(raw / "terminal_validation.json", dict(publication=publication, passed=True))
    e.progress(value["honest_verdict"], value["completed_count"], value["censored_count"])
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """One thin CLI supports real workers, private controls and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261103")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-control", action="store_true")
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
        if args.date != "20261103":
            raise ValueError("calendar_date")
        return run(args.output, args.date, private=args.private_control)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8772] error={error}", file=sys.stderr, flush=True)
        return 1
