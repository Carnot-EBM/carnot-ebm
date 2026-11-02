"""Reuse bounded validation custody while freezing model-free protocol operands."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import relational_feature_runner_8701 as historical
from carnot.reporting import verification_methods_8726 as e
from carnot.reporting.primary_publication import publish_primary
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
shared = e.binding.namespace(historical, "verification_methods_validation")
shared.e = e
shared.CONSUMERS = [
    "tests/python/test_primary_publication_7928.py::test_nested_reports_and_terminal_revalidation"
]


def replay(path: Path) -> bool:
    """Rebuild from original labels and sources so rehashing a forged summary is insufficient."""
    try:
        value = json.loads(path.read_bytes())
        work = e.load(value["primitive_reference"])
        seal = e.load(work["seal_reference"])
        if value["reproducibility_checksum"] != e.checksum(value) or seal != {
            k: v for k, v in work.items() if k != "seal_reference"
        }:
            return False
        refs = work["sources"] + work["raw_references"] + work["validation"]["raw_shard_hashes"]
        if not all(e.upstream.authenticate(ref)["passed"] for ref in refs):
            return False
        before = e.load(work["raw_references"][0])
        if any(
            before[k] != work[k]
            for k in ["sources", "inputs", "contract", "resources", "frozen", "runtime"]
        ):
            return False
        if e.load(work["raw_references"][1]) != work["measurement"]:
            return False
        spans = work["phase_spans"]
        if (
            work["duration_s"]
            != (spans[-1]["end_monotonic_ns"] - spans[0]["start_monotonic_ns"]) / 1e9
        ):
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
        with TemporaryDirectory(prefix="cold8726-", dir="/var/tmp") as directory:
            measured = e.measure(frozen, Path(directory))
            if shared.semantic(measured) != shared.semantic(work["measurement"]):
                return False
            custody = work["validation"]["custody"]
            if not work["private_control"] and custody.get("retained_child_rows"):
                children = custody["retained_child_rows"]
                if shared.retained.aggregate(children, custody["node_ids"]) != custody:
                    return False
                coverage = shared.coverage_data(
                    [r for r in children if e.load(r)["role"] == "owned"],
                    Path(directory) / "combined",
                )
                if {k: v for k, v in coverage.items() if k != "combined"} != {
                    k: v for k, v in work["validation"]["coverage"].items() if k != "combined"
                }:
                    return False
        return e.build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(output: Path, *, private: bool = False) -> int:
    """Freeze before controls and reserve the final 900 seconds for bounded qualification."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="methods8726-", dir="/var/tmp") as directory:
        resources = shared.shared.preconditions(Path(directory))
        resources["ready"] = bool(
            resources["disk_backed"]
            and resources["writable"]
            and resources["free_memory_kib"] >= 1048576
            and resources["free_disk_bytes"] >= 104857600
        )
        sources = [
            e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS
        ] + shared.shared.source_hashes()
        e.progress("authenticate_and_freeze")
        branch, contract = e.inputs(e.ROOT), e.authority.authority(e.ROOT)
        runtime = e.upstream.runtime(e.ROOT, raw / "runtime")
        frozen = e.freeze(branch["worlds"] if branch["passed"] and resources["ready"] else [])
        e.atomic_json(
            raw / "premeasurement.json",
            dict(
                sources=sources,
                inputs=branch,
                contract=contract,
                resources=resources,
                frozen=frozen,
                runtime=runtime,
                started_monotonic_ns=start,
                started_at_utc=wall,
            ),
        )
        (raw / "premeasurement.json").chmod(0o444)
        e.progress("before_constructed_controls", 0, 1440)
        measurement = e.measure(frozen, raw / "measurement")
        measured = time.monotonic_ns()
        if measured - start > 3600 * 10**9:
            raise ValueError("measurement_cap_exceeded")
        e.progress("after_constructed_controls_before_validation", len(measurement["rows"]), 0)
        evidence = [
            e.reference(raw / "premeasurement.json"),
            e.reference(raw / "measurement/probes.json"),
            runtime["diagnostic_reference"],
        ]
        validation = (
            dict(
                receipts=[],
                coverage=dict(complete=False),
                custody=dict(passed=False),
                raw_shard_hashes=[],
            )
            if private
            else shared.validate(raw, Path(directory), sources + evidence)
        )
        end = time.monotonic_ns()
        work = dict(
            raw=str(raw),
            invocation_id=raw.name,
            private_control=private,
            resources=resources,
            sources=sources,
            inputs=branch,
            contract=contract,
            frozen=frozen,
            runtime=runtime,
            measurement=measurement,
            validation=validation,
            raw_references=[
                e.reference(raw / "premeasurement.json"),
                e.reference(raw / "measurement/probes.json"),
                runtime["diagnostic_reference"],
            ],
            adversarial_findings=[],
            started_at_utc=wall,
            ended_at_utc=datetime.now(UTC).isoformat(),
            duration_s=(end - start) / 1e9,
            phase_spans=[
                dict(phase=n, start_monotonic_ns=a, end_monotonic_ns=b, duration_s=(b - a) / 1e9)
                for n, a, b in [
                    ("preconditions_and_controls", start, measured),
                    ("validation", measured, end),
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
        publication = publish_primary(output, value, lambda path: shared.terminal(path, raw))
        e.atomic_json(
            Path(value["terminal_validation_sidecar_path"]), dict(publication=publication)
        )
    e.progress("published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private checks use the real CLI with a deadline and cannot grant public readiness."""
    e.progress("start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if (
        args.date != "20261102"
        or args.private_output
        and (
            args.private_output.absolute().is_relative_to(e.ROOT)
            or args.private_output.resolve().is_relative_to(e.ROOT)
        )
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
