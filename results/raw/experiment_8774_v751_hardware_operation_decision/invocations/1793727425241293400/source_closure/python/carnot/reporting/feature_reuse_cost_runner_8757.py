"""REQ-VERIFY-8757: freeze, measure, validate and atomically publish one invocation."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import feature_reuse_cost_8757 as e
from carnot.reporting import feature_reuse_cost_measure_8757 as m
from carnot.reporting import installed_batch_runner_8702 as publication
from carnot.reporting import relational_feature_runner_8701 as shared
from carnot.reporting.installed_batch_cost_runner_8717 import heartbeat
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, progress
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
BASE_CHILD = shared.child
SUITES = e.TESTS + [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def scoped_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Measure only owned files, including actual standalone CLI subprocesses."""
    for arg in argv:
        if arg.startswith("--rcfile="):
            path = Path(arg.removeprefix("--rcfile="))
            path.write_text(
                path.read_text().replace(
                    "[run]\n", "[run]\ninclude=" + ",".join(str(e.ROOT / p) for p in e.OWNED) + "\n"
                )
            )
    return dict(BASE_CHILD(name, argv, logs, **kwargs))


def validate(raw: Path, scratch: Path, sources: list[Json]) -> Json:
    """Shared custody retains original node IDs, JUnit, coverage and full log hashes."""
    with (
        patch.multiple(shared, e=e, child=scoped_child, CONSUMERS=[]),
        patch.object(e, "TESTS", SUITES),
    ):
        return dict(shared.validate(raw, scratch, sources))


def replay(path: Path, authority: Json | None = None) -> bool:
    """Independently reduce every retained session before comparing the entire report."""
    try:
        value = json.loads(path.read_bytes())
        if authority is None or value["primitive_reference"] != authority:
            return False
        work = e.load(authority)
        if e.load(work["seal_reference"]) != {
            k: v for k, v in work.items() if k != "seal_reference"
        }:
            return False
        if e.inputs(Path(work["input_root"])) != work["inputs"]:
            return False
        for ref in (
            work["sources"] + work["raw_references"] + work["validation"]["raw_shard_hashes"]
        ):
            if e.reference(Path(ref["path"])) != ref:
                return False
        measured = work["measurement"]
        rebuilt: Json = dict(
            process_rows=e.roster(), request_rows=[], parity_rows=[], complete_cost_rows=[]
        )
        for index, row in enumerate(measured["process_rows"]):
            if row["status"] == "completed":
                primitive = e.load(row["session_reference"])
                reduced = m.original.reduce(primitive, work["inputs"])
                rebuilt["process_rows"][index] = dict(
                    reduced["process_row"],
                    scientific_request_sha256=e.canonical_hash(work["inputs"]["requests"]),
                    output_sha256=e.canonical_hash(
                        [
                            dict(
                                responses=o["response"]["responses"],
                                feature_hashes=o["response"]["feature_hashes"],
                            )
                            for o in primitive["observations"]
                        ]
                    ),
                    session_reference=row["session_reference"],
                )
                config = e.load(primitive["config_reference"])
                if config["reuse_request_features"] is not (row["mode"] == "reuse"):
                    return False
                for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
                    rebuilt[key] += reduced[key]
            else:
                rebuilt["process_rows"][index] = row
        if any(rebuilt[k] != measured[k] for k in rebuilt):
            return False
        custody = work["validation"]["custody"]
        if (
            custody.get("retained_child_rows")
            and shared.retained.aggregate(custody["retained_child_rows"], custody["node_ids"])
            != custody
        ):
            return False
        return e.build(work, Path(work["output"])) == value
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return False


def run(output: Path, *, root: Path = e.ROOT, private: bool = False) -> int:
    """Freeze first and reserve validation time outside the2700-second timing budget."""
    started, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="feature8757-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        pre = shared.shared.preconditions(scratch)
        health = (
            e.ROOT / "results/raw" / e.NAME / "development/health/repository_health.receipt.json"
        )
        pre["repository_health_reference"] = e.reference(health) if health.is_file() else None
        inputs = e.inputs(root)
        sources = [
            e.reference(e.ROOT / p)
            for p in e.OWNED
            + SUITES
            + [
                "python/carnot/reporting/installed_batch_cost_measure_8717.py",
                "python/carnot/reporting/relational_feature_runner_8701.py",
                "python/carnot/reporting/v746_retained_evidence.py",
                "python/carnot/reporting/v709_execution.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/installed_batch_8702.py",
                "python/carnot/reporting/installed_batch_cost_8717.py",
                "python/carnot/reporting/installed_batch_runner_8702.py",
                "python/carnot/inference/resident_protocol.py",
                "python/carnot/inference/resident_energy.py",
                "python/carnot/inference/resident_batch.py",
                "python/carnot/inference/resident_features_8743.py",
                "scripts/experiment_template.py",
                "pyproject.toml",
                "tests/python/conftest.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ]
        collection = child(
            "freeze_nodes",
            [sys.executable, "-m", "pytest", *shared.OPTIONS, "--collect-only", *SUITES],
            raw / "checks",
            deadline=120,
        )
        nodes = [
            n
            for n in Path(collection["stdout_path"]).read_text().splitlines()
            if n.startswith("tests/") and "::" in n
        ]
        e.atomic_json(
            raw / "premeasurement.json",
            dict(
                inputs=inputs,
                sources=sources,
                method=e.method(),
                roster=e.roster(),
                node_ids=nodes,
                node_ids_sha256=e.canonical_hash(nodes),
                preconditions=pre,
                collection=collection,
            ),
        )
        measured = m.measure(
            raw / "measurement", scratch, dict(inputs, passed=False), started + 2700 * 10**9
        )
        error = None
        try:
            if not collection["passed"] or not nodes or len(nodes) != len(set(nodes)):
                raise ValueError("frozen_test_collection")
            if not (
                pre["writable"]
                and pre["disk_backed"]
                and pre["free_memory_kib"] >= 1048576
                and pre["free_disk_bytes"] >= 104857600
            ):
                raise ValueError("private_resources")
            progress("8757_before_measurement", 0, 40)
            measured = m.measure(raw / "measurement", scratch, inputs, started + 2700 * 10**9)
            error = measured["owned_error"]
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as failure:
            error = str(failure)
        boundary = time.monotonic_ns()
        if boundary - started > 2700 * 10**9:
            error = "measurement_cap_exceeded"
        progress("8757_after_measurement_before_validation")
        validation = (
            dict(
                receipts=[],
                coverage=dict(complete=False),
                custody=dict(passed=False),
                raw_shard_hashes=[],
            )
            if private
            else validate(raw, scratch, sources)
        )
        if not all(e.reference(Path(ref["path"])) == ref for ref in sources):
            error = "source_seal_changed"
        ended = time.monotonic_ns()
        work = dict(
            raw=str(raw),
            output=str(output.absolute()),
            input_root=str(root),
            invocation_id=raw.name,
            private_control=private,
            inputs=inputs,
            preconditions=pre,
            sources=sources,
            measurement=measured,
            validation=validation,
            owned_error=error,
            raw_references=[e.reference(raw / "premeasurement.json")] + measured["references"],
            duration_s=(ended - started) / 1e9,
            started_at_utc=wall,
            ended_at_utc=datetime.now(UTC).isoformat(),
            phase_spans=[
                dict(phase=n, start_monotonic_ns=a, end_monotonic_ns=b, duration_s=(b - a) / 1e9)
                for n, a, b in [
                    ("preconditions_and_measurement", started, boundary),
                    ("validation", boundary, ended),
                ]
            ],
        )
        e.atomic_json(raw / "work.json", work)
        candidate = raw / "preterminal.json"
        e.atomic_json(candidate, e.build(work, output))
        work["adversarial_findings"] = verify_artifact(candidate)["flags"]
        e.atomic_json(raw / "invocation_seal.json", work)
        (raw / "invocation_seal.json").chmod(0o444)
        work["seal_reference"] = e.reference(raw / "invocation_seal.json")
        e.atomic_json(raw / "work.json", work)
        value = e.build(work, output)
        with patch.object(publication, "e", e):
            published = publish_primary(output, value, lambda path: publication.terminal(path, raw))
        e.atomic_json(Path(value["terminal_validation_sidecar_path"]), dict(publication=published))
    progress("8757_published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject aliases before writes and keep parent progress visible during children."""
    progress("8757_start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261103")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    if args.date != "20261103":
        return 2
    if args.private_output and (
        args.private_output.absolute().is_relative_to(e.ROOT)
        or args.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if args.cold_replay:
        authority = dict(zip(["path", "sha256"], args.authority)) if args.authority else None
        return int(not replay(args.cold_replay, authority))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    stop = Event()
    worker = Thread(target=heartbeat, args=(stop,), daemon=True)
    worker.start()
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            root=args.root,
            private=bool(args.private_output),
        )
    finally:
        stop.set()
        worker.join(timeout=1)
        signal.setitimer(signal.ITIMER_REAL, *previous)
