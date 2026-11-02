"""REQ-VERIFY-8717: seal bounded installed measurements before atomic publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import installed_batch_cost_8717 as e
from carnot.reporting import installed_batch_cost_measure_8717 as m
from carnot.reporting import installed_batch_runner_8702 as publication
from carnot.reporting import relational_feature_runner_8701 as shared
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import progress
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
BASE_CHILD = shared.child
COVERAGE_INCLUDE = "include=" + ",".join(str(e.ROOT / p) for p in e.OWNED) + "\n"


def heartbeat(stop: Event) -> None:
    """Keep the invocation visible even while large evidence files are validated."""
    while not stop.wait(30):
        progress("8717_parent_heartbeat")


def scoped_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Instrument owned files only, leaving upstream execution and assertions intact."""
    for arg in argv:
        if arg.startswith("--rcfile="):
            config = Path(arg.removeprefix("--rcfile="))
            config.write_text(config.read_text().replace("[run]\n", "[run]\n" + COVERAGE_INCLUDE))
    return dict(BASE_CHILD(name, argv, logs, **kwargs))


def validate(raw: Path, scratch: Path, sources: list[Json]) -> Json:
    """Reuse genuine disjoint test children; consumer tests cannot cover this producer."""
    with patch.multiple(shared, e=e, child=scoped_child):
        return dict(shared.validate(raw, scratch, sources))


def replay(path: Path, authority: Json | None = None) -> bool:
    """Rebuild from raw frames under an independently supplied work identity."""
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
        refs = work["sources"] + work["raw_references"]
        if not all(e.reference(Path(ref["path"])) == ref for ref in refs):
            return False
        measured = work["measurement"]
        rebuilt: Json = dict(
            process_rows=e.roster(e.method()),
            request_rows=[],
            parity_rows=[],
            complete_cost_rows=[],
        )
        for index, row in enumerate(measured["process_rows"]):
            if row["status"] == "completed":
                primitive = e.load(
                    e.reference(Path(work["raw"]) / "measurement" / row["unit_id"] / "session.json")
                )
                reduced = m.reduce(primitive, work["inputs"])
                rebuilt["process_rows"][index] = reduced["process_row"]
                for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
                    rebuilt[key] += reduced[key]
            else:
                rebuilt["process_rows"][index] = row
        if any(rebuilt[k] != measured[k] for k in rebuilt):
            return False
        validation = work["validation"]
        if validation["raw_shard_hashes"]:
            custody = validation["custody"]
            if (
                shared.retained.aggregate(custody["retained_child_rows"], custody["node_ids"])
                != custody
            ):
                return False
        return e.build(work, Path(work["output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(output: Path, *, root: Path = e.ROOT, private: bool = False) -> int:
    """Freeze first, stop measurement within3600 seconds and retain900 seconds for checks."""
    started, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="batch8717-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        pre = shared.shared.preconditions(scratch)
        inputs = e.inputs(root)
        if not (
            pre["disk_backed"]
            and pre["writable"]
            and pre["free_memory_kib"] >= 1048576
            and pre["free_disk_bytes"] >= 104857600
        ):
            inputs["passed"] = False
            inputs["gate_check_summary"].append(
                e.contract.gate(scratch, "private_resources", True, pre, e.TASK)
            )
        for i in [0, 1]:
            (raw / f"{i}.ini").write_text(
                f"[run]\n{COVERAGE_INCLUDE}parallel=true\npatch=subprocess\ndata_file={scratch / str(i) / '.coverage'}\n[report]\nexclude_lines=\n"
            )
        sources = [
            e.reference(e.ROOT / p)
            for p in e.OWNED
            + e.TESTS
            + [
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/relational_feature_runner_8701.py",
                "python/carnot/reporting/v746_retained_evidence.py",
                "python/carnot/reporting/v709_execution.py",
                "python/carnot/reporting/installed_batch_8702.py",
                "python/carnot/reporting/installed_batch_runner_8702.py",
                "python/carnot/reporting/v747_contract_methods.py",
                "python/carnot/inference/resident_energy.py",
                "python/carnot/inference/resident_protocol.py",
                "python/carnot/inference/resident_batch.py",
                "python/carnot/verify/source_alignment.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ]
        sources += [e.reference(raw / f"{i}.ini") for i in [0, 1]]
        e.atomic_json(
            raw / "premeasurement.json",
            dict(inputs=inputs, sources=sources, roster=e.roster(e.method()), preconditions=pre),
        )
        progress("8717_before_measurement", 0, 60)
        measured = m.measure(
            raw / "measurement", scratch, dict(inputs, passed=False), started + 3600 * 10**9
        )
        error = None
        try:
            measured = m.measure(raw / "measurement", scratch, inputs, started + 3600 * 10**9)
            error = measured["owned_error"]
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as failure:
            error = str(failure)
        boundary = time.monotonic_ns()
        if boundary - started > 3600 * 10**9:
            error = "measurement_cap_exceeded"
        progress(
            "8717_after_measurement_before_validation",
            sum(r["status"] == "completed" for r in measured["process_rows"]),
            0,
        )
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
    progress("8717_published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject private-output aliases before any write and keep one bounded invocation."""
    progress("8717_start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    if args.date != "20261102":
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
