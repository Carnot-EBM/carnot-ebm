"""REQ-REPORT-8712: retain bounded children and publish only cold-checked evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import relational_capacity_8712 as e
from carnot.reporting import v746_contract_runner as shared
from carnot.reporting import v746_retained_evidence as retained
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v725_contract_methods import runtime as supplied_runtime
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]


def coverage_data(rows: list[Json], directory: Path) -> Json:
    """Combine copied owned SQLite only; consumer execution cannot qualify this producer."""
    combined = retained.combine(rows, directory)
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files = []
    for name in e.OWNED:
        _, statements, _, missing, _ = cov.analysis2(str(e.ROOT / name))
        files.append(dict(path=name, statements=len(statements), missing=missing))
    return dict(
        complete=all(not r["missing"] for r in files),
        files=files,
        statements=sum(r["statements"] for r in files),
        missing=sum(len(r["missing"]) for r in files),
        combined=combined,
    )


def validate(raw: Path, scratch: Path, sources: list[Json]) -> Json:
    """Collect once, execute disjoint nodes, and retain every child's originals immediately."""
    collection = child(
        "collect_once",
        [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *e.TESTS, *CONSUMERS],
        raw / "checks",
    )
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    e.atomic_json(raw / "inventory.json", dict(nodes=nodes, receipt=collection))
    receipts, children, owned = [collection], [], []
    if (
        not collection["passed"]
        or not nodes
        or len(nodes) != len(set(nodes))
        or not all(n in nodes for n in CONSUMERS)
    ):
        return dict(
            receipts=receipts,
            coverage=dict(complete=False),
            custody=dict(passed=False),
            raw_shard_hashes=[],
        )
    for index, selected in enumerate(
        [
            [n for n in nodes if n.startswith(e.TESTS[0] + "::")],
            [n for n in nodes if not n.startswith(e.TESTS[0] + "::")],
        ]
    ):
        private = scratch / str(index)
        private.mkdir()
        config = raw / (str(index) + ".ini")
        config.write_text(
            "[run]\nparallel=true\npatch=subprocess\ndata_file="
            + str(private / ".coverage")
            + "\n[report]\nexclude_lines=\n"
        )
        args = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "-m",
            "pytest",
            *OPTIONS,
            *selected,
            "--basetemp=" + str(private / "tmp"),
            "--junitxml=" + str(private / "junit.xml"),
        ]
        receipt = child("owned" if index == 0 else "consumer", args, raw / "checks", deadline=900)
        receipts.append(receipt)
        ref = retained.retain(
            receipt,
            selected,
            private,
            raw / "children" / str(index),
            sources + [e.reference(config)],
            "owned" if index == 0 else "consumer",
        )
        children.append(ref)
        if index == 0:
            owned.append(ref)
    custody = retained.aggregate(children, nodes)
    coverage = coverage_data(owned, raw / "combined") if custody["passed"] else dict(complete=False)
    for name, args in [
        ("ruff_check", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipts.append(child(name, [sys.executable, *args], raw / "checks"))
    return dict(
        receipts=receipts,
        coverage=coverage,
        custody=custody,
        raw_shard_hashes=children + [e.reference(raw / "inventory.json")],
    )


def semantic(value: Any) -> Any:
    """Replay compares exact scientific values while remeasuring clocks in private scratch."""
    if isinstance(value, dict):
        return {
            k: semantic(v)
            for k, v in value.items()
            if k
            not in {"latency_s", "inspection_duration_s", "window_reference", "database_reference"}
        }
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def replay(path: Path) -> bool:
    """Rebuild original inputs and deterministic outcomes; repaired report hashes cannot help."""
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
        diagnostic = e.load(work["runtime"]["diagnostic_reference"])
        supplied = supplied_runtime(e.ROOT)
        validator = e.upstream.custody(
            dict(
                path=str(
                    e.ROOT / "results/experiment_8580_v737_current_validator_qualification.json"
                ),
                sha256="sha256:7656f6f49b69237528263cd6dc38ac9503e853b1f53d1b5b1037f6510d2892f4",
            ),
            "current_validator_ready_score",
            8580,
        )
        changed = any(
            r["changed"] and r["field"] in {"libraries", "environment"}
            for r in supplied.get("changes", [])
        )
        expected_runtime = int(
            supplied["ready"]
            and changed
            and validator["passed"]
            and "diagnostic" not in diagnostic["cached"]
        )
        if (
            semantic(diagnostic["supplied"]) != semantic(supplied)
            or diagnostic["validator"] != validator
            or work["runtime"]["ready"] != expected_runtime
        ):
            return False
        with TemporaryDirectory(prefix="cold8712-", dir="/var/tmp") as directory:
            recomputed = e.measure(
                work["inputs"]["worlds"]
                if work["inputs"]["passed"] and work["resources"]["ready"]
                else [],
                Path(directory),
            )
            if semantic(recomputed) != semantic(work["measurement"]):
                return False
            validation = work["validation"]
            if not work["private_control"] and validation["custody"].get("retained_child_rows"):
                custody = validation["custody"]
                rows = custody["retained_child_rows"]
                if retained.aggregate(rows, custody["node_ids"]) != custody:
                    return False
                owned = [r for r in rows if e.load(r)["role"] == "owned"]
                summary = coverage_data(owned, Path(directory) / "combined")
                if {k: v for k, v in summary.items() if k != "combined"} != {
                    k: v for k, v in validation["coverage"].items() if k != "combined"
                }:
                    return False
        return e.build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Preserve scanner severities and validate exactly the bytes publication will expose."""
    findings = json.loads(path.read_bytes())["adversarial_findings"]
    receipts = []
    for name, args in [
        ("cold", [e.CLI, "--cold-replay", str(path)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
    ]:
        receipts.append(
            child(
                name,
                [sys.executable, "-u", *args],
                raw / "terminal",
                deadline=180,
                expected=int(bool(findings)) if name == "adversarial" else 0,
            )
        )
    scan = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts) and scan["reports"][0]["flags"] == findings,
        receipts=receipts,
        adversarial=scan,
    )


def run(output: Path, *, private: bool = False) -> int:
    """Freeze before measurement, stop by 3600 seconds, and reserve validation time."""
    started, started_at = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="relational8712-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        resources = shared.preconditions(scratch)
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
        sources = (
            shared.source_hashes()
            + [e.reference(e.ROOT / p) for p in e.TESTS]
            + [e.reference(raw / f"{i}.ini") for i in [0, 1]]
        )
        e.progress("before_runtime_reader")
        runtime = e.upstream.runtime(e.ROOT, raw / "runtime")
        e.progress("after_runtime_reader")
        e.progress("authenticate_and_seal")
        branch, contract = e.inputs(e.ROOT), e.authority.authority(e.ROOT)
        e.atomic_json(
            raw / "premeasurement.json",
            dict(sources=sources, inputs=branch, contract=contract, resources=resources),
        )
        (raw / "premeasurement.json").chmod(0o444)
        e.progress("before_measurement", 0, 480)
        measurement = (
            e.measure(branch["worlds"], raw / "measurement")
            if branch["passed"] and resources["ready"]
            else e.measure([], raw / "measurement")
        )
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
            else validate(raw, scratch, [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS])
        )
        ended = time.monotonic_ns()
        raw_refs = [
            e.reference(raw / "premeasurement.json"),
            runtime["diagnostic_reference"],
        ] + runtime["references"]
        raw_refs += [
            arm["window_reference"]
            for arm in measurement["lexical_rows"]
            if "window_reference" in arm
        ]
        raw_refs += [
            row["database_reference"]
            for row in measurement["symbolic_rows"]
            if "database_reference" in row
        ]
        work = dict(
            raw=str(raw),
            invocation_id=raw.name,
            private_control=private,
            resources=resources,
            sources=sources,
            inputs=branch,
            contract=contract,
            measurement=measurement,
            validation=validation,
            raw_references=raw_refs,
            runtime=runtime,
            adversarial_findings=[],
            started_at_utc=started_at,
            ended_at_utc=datetime.now(UTC).isoformat(),
            duration_s=(ended - started) / 1e9,
            phase_spans=[
                dict(phase=name, start_monotonic_ns=a, end_monotonic_ns=b, duration_s=(b - a) / 1e9)
                for name, a, b in [
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
        publication = publish_primary(output, value, lambda path: terminal(path, raw))
        e.atomic_json(
            Path(value["terminal_validation_sidecar_path"]), dict(publication=publication)
        )
    e.progress("published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject public private-output aliases before any files can be created."""
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
