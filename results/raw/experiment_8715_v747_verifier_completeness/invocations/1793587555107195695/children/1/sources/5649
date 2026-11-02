"""REQ-VERIFY-8715: reuse bounded child custody and reject altered sealed primitives."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import relational_feature_runner_8701 as shared
from carnot.reporting import verifier_completeness_8715 as e
from carnot.reporting.primary_publication import publish_primary
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def validate(raw: Path, scratch: Path, sources: list[Json]) -> Json:
    """Reuse the shipped disjoint child, JUnit and subprocess coverage mechanism."""
    with patch.object(shared, "e", e):
        return dict(shared.validate(raw, scratch, sources))


def replay(path: Path) -> bool:
    """Recreate labels from original authority so consistently rehashed edits still fail."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != e.checksum(value):
            return False
        work = e.load(value["primitive_reference"])
        seal = e.load(work["seal_reference"])
        if seal != {k: v for k, v in work.items() if k != "seal_reference"}:
            return False
        refs = work["sources"] + work["raw_references"] + work["validation"]["raw_shard_hashes"]
        if not all(e.upstream.authenticate(ref)["passed"] for ref in refs):
            return False
        if e.inputs(e.ROOT) != work["inputs"] or e.authority.authority(e.ROOT) != work["contract"]:
            return False
        frozen = e.freeze(
            work["inputs"]["worlds"]
            if work["inputs"]["passed"] and work["resources"]["ready"]
            else []
        )
        if frozen != work["frozen"]:
            return False
        with TemporaryDirectory(prefix="cold8715-", dir="/var/tmp") as directory:
            if e.measure(frozen, Path(directory)) != work["measurement"]:
                return False
            validation = work["validation"]
            if not work["private_control"] and validation["custody"].get("retained_child_rows"):
                custody = validation["custody"]
                children = custody["retained_child_rows"]
                if shared.retained.aggregate(children, custody["node_ids"]) != custody:
                    return False
                with patch.object(shared, "e", e):
                    summary = shared.coverage_data(
                        [r for r in children if e.load(r)["role"] == "owned"],
                        Path(directory) / "combined",
                    )
                if {k: v for k, v in summary.items() if k != "combined"} != {
                    k: v for k, v in validation["coverage"].items() if k != "combined"
                }:
                    return False
        return e.build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(output: Path, *, private: bool = False) -> int:
    """Freeze all operands before measurement and reserve validation within the hard cap."""
    started, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="verifier8715-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        resources = shared.shared.preconditions(scratch)
        resources["ready"] = bool(
            resources["disk_backed"]
            and resources["writable"]
            and resources["free_memory_kib"] >= 1048576
            and resources["free_disk_bytes"] >= 104857600
        )
        for i in [0, 1]:
            (raw / f"{i}.ini").write_text(
                "[run]\nparallel=true\npatch=subprocess\ndata_file="
                + str(scratch / str(i) / ".coverage")
                + "\n[report]\nexclude_lines=\n"
            )
        sources = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
        sources += [e.reference(raw / f"{i}.ini") for i in [0, 1]]
        sources += shared.shared.source_hashes()
        e.progress("authenticate_and_freeze")
        branch, contract = e.inputs(e.ROOT), e.authority.authority(e.ROOT)
        frozen = e.freeze(branch["worlds"] if branch["passed"] and resources["ready"] else [])
        e.atomic_json(
            raw / "premeasurement.json",
            dict(
                sources=sources,
                inputs=branch,
                contract=contract,
                resources=resources,
                frozen=frozen,
            ),
        )
        (raw / "premeasurement.json").chmod(0o444)
        e.progress("before_measurement", 0, 720)
        measurement = e.measure(frozen, raw / "measurement")
        measured = time.monotonic_ns()
        if measured - started > 3600 * 10**9:
            raise ValueError("measurement_cap_exceeded")
        e.progress("after_measurement_validation_reserved", len(measurement["rows"]), 0)
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
        ended = time.monotonic_ns()
        work = dict(
            raw=str(raw),
            invocation_id=raw.name,
            private_control=private,
            resources=resources,
            sources=sources,
            inputs=branch,
            contract=contract,
            frozen=frozen,
            measurement=measurement,
            validation=validation,
            raw_references=[
                e.reference(raw / "premeasurement.json"),
                e.reference(raw / "measurement/probes.json"),
            ],
            adversarial_findings=[],
            started_at_utc=wall,
            ended_at_utc=datetime.now(UTC).isoformat(),
            duration_s=(ended - started) / 1e9,
            phase_spans=[
                dict(phase=n, start_monotonic_ns=a, end_monotonic_ns=b, duration_s=(b - a) / 1e9)
                for n, a, b in [
                    ("preconditions_and_measurement", started, measured),
                    ("validation", measured, ended),
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
        with patch.object(shared, "e", e):
            publication = publish_primary(output, value, lambda path: shared.terminal(path, raw))
        e.atomic_json(
            Path(value["terminal_validation_sidecar_path"]), dict(publication=publication)
        )
    e.progress("published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep private controls outside the checkout and bound the whole invocation."""
    e.progress("start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261102":
        return 2
    if args.private_output and (
        args.private_output.absolute().is_relative_to(e.ROOT)
        or args.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            private=bool(args.private_output),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
