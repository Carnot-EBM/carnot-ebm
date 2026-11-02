"""REQ-REPORT-8698: retain owned child execution and publish only cold-checked bytes."""

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

from carnot.reporting import relational_evidence_8698 as e
from carnot.reporting import v746_contract_runner as shared
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]


def replay(path: Path) -> bool:
    """Rebuild from original authority; repaired report hashes cannot hide changed operands."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != e.checksum(value):
            return False
        work = e.load(value["primitive_reference"])
        for ref in (
            work["sources"]
            + work["runtime"]["references"]
            + [work["runtime"]["diagnostic_reference"]]
            + work["validation"]["raw_shard_hashes"]
        ):
            if not e.authenticate(ref)["passed"]:
                return False
        seal = e.load(work["seal_reference"])
        if seal != {k: v for k, v in work.items() if k != "seal_reference"}:
            return False
        original = e.inputs(e.ROOT)
        if not work["resources"]["ready"]:
            original["passed"] = False
            original["failures"].append(
                e.authority.gate(
                    Path(work["resources"]["scratch"]),
                    "disk_memory_scratch",
                    True,
                    work["resources"],
                )
            )
        if original != work["inputs"] or e.qualify(original["worlds"]) != work["corpus"]:
            return False
        if e.authority.authority(e.ROOT) != work["contract"]:
            return False
        if not work["private_control"]:
            custody = work["validation"]["custody"]
            rows = custody.get("retained_child_rows", [])
            if not rows:
                return (
                    work["validation"]["coverage"] == dict(complete=False)
                    and custody.get("passed") is False
                    and e.build(work, Path(value["publication_output"])) == value
                )
            if e.retained.aggregate(rows, custody["node_ids"]) != custody:
                return False
            owned = [ref for ref in rows if e.load(ref)["role"] == "owned"]
            with TemporaryDirectory(prefix="cold8698-", dir="/var/tmp") as directory:
                measured, _ = coverage_data(owned, Path(directory))
            if measured != work["validation"]["coverage"]:
                return False
        return e.build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def coverage_data(rows: list[Json], directory: Path) -> tuple[Json, Json]:
    """Recompute statements from copied owned SQLite; a summary cannot certify itself."""
    combined = e.retained.combine(rows, directory)
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files = []
    for name in e.OWNED:
        _, statements, _, missing, _ = cov.analysis2(str(e.ROOT / name))
        files.append(dict(path=name, statements=len(statements), missing=missing))
    return (
        dict(
            complete=all(not row["missing"] for row in files),
            files=files,
            statements=sum(row["statements"] for row in files),
            missing=sum(len(row["missing"]) for row in files),
        ),
        combined["combined_reference"],
    )


def validate(raw: Path, scratch: Path, sources: list[Json]) -> Json:
    """Collect once and retain disjoint owned and consumer children before combining copies."""
    logs = raw / "checks"
    collection = child(
        "collect_once",
        [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *e.TESTS, *CONSUMERS],
        logs,
    )
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    e.atomic_json(raw / "inventory.json", dict(nodes=nodes, receipt=collection))
    if (
        not collection["passed"]
        or not nodes
        or len(nodes) != len(set(nodes))
        or not all(n in nodes for n in CONSUMERS)
    ):
        return dict(
            receipts=[collection],
            coverage=dict(complete=False),
            custody=dict(passed=False),
            raw_shard_hashes=[],
        )
    receipts, retained = [collection], []
    owned_refs: list[Json] = []
    for name, selected in [
        ("owned", [n for n in nodes if n.startswith(e.TESTS[0] + "::")]),
        ("consumer", [n for n in nodes if not n.startswith(e.TESTS[0] + "::")]),
    ]:
        private = scratch / name
        private.mkdir()
        config = raw / (name + ".ini")
        argv = [
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
        receipt = child(name, argv, logs, deadline=600)
        receipts.append(receipt)
        ref = e.retained.retain(receipt, selected, private, raw / "children" / name, sources, name)
        retained.append(ref)
        if name == "owned":
            owned_refs.append(ref)
    custody = e.retained.aggregate(retained, nodes)
    if not custody["passed"]:
        return dict(
            receipts=receipts,
            coverage=dict(complete=False),
            custody=custody,
            raw_shard_hashes=retained + [e.reference(raw / "inventory.json")],
        )
    coverage, combined = coverage_data(owned_refs, raw / "combined")
    for name, args in [
        ("ruff_check", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipts.append(child(name, [sys.executable, *args], logs))
    return dict(
        receipts=receipts,
        coverage=coverage,
        custody=custody,
        raw_shard_hashes=[
            *retained,
            combined,
            e.reference(raw / "inventory.json"),
        ],
    )


def terminal(path: Path, raw: Path) -> Json:
    """Preserve actual scanner findings and require fresh replay before atomic publication."""
    receipts = []
    findings = json.loads(path.read_bytes())["adversarial_findings"]
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
    """Seal inputs before CPU measurement and reserve bounded children for validation."""
    started, started_at = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="relational8698-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        resources = shared.preconditions(scratch)
        resources["ready"] = bool(
            resources["disk_backed"]
            and resources["writable"]
            and resources["free_memory_kib"] >= 1048576
            and resources["free_disk_bytes"] >= 104857600
        )
        for name in ["owned", "consumer"]:
            config = raw / (name + ".ini")
            config.write_text(
                "[run]\nparallel=true\npatch=subprocess\ndata_file="
                + str(scratch / name / ".coverage")
                + "\n[report]\nexclude_lines=\n"
            )
        sources = shared.source_hashes() + [e.reference(e.ROOT / p) for p in e.TESTS]
        sources += [e.reference(raw / (n + ".ini")) for n in ["owned", "consumer"]]
        e.progress("preconditions_and_source_seal")
        branch, contract = e.inputs(e.ROOT), e.authority.authority(e.ROOT)
        if not resources["ready"]:
            branch["passed"] = False
            branch["failures"].append(
                e.authority.gate(scratch, "disk_memory_scratch", True, resources)
            )
        corpus = e.qualify(branch["worlds"])
        optional = e.runtime(e.ROOT, raw / "runtime")
        measured = time.monotonic_ns()
        if measured - started > 3600 * 10**9:
            raise ValueError("measurement_cap_exceeded")
        e.progress("measurement_complete_validation_reserved", len(corpus["rows"]), 0)
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
        work = dict(
            raw=str(raw),
            run_date="20261101",
            invocation_id=raw.name,
            private_control=private,
            resources=resources,
            sources=sources,
            inputs=branch,
            contract=contract,
            corpus=corpus,
            runtime=optional,
            validation=validation,
            adversarial_findings=[],
            started_at_utc=started_at,
            ended_at_utc=datetime.now(UTC).isoformat(),
            duration_s=(ended - started) / 1e9,
            phase_spans=[
                dict(
                    phase="preconditions_and_measurement",
                    start_monotonic_ns=started,
                    end_monotonic_ns=measured,
                    duration_s=(measured - started) / 1e9,
                ),
                dict(
                    phase="validation",
                    start_monotonic_ns=measured,
                    end_monotonic_ns=ended,
                    duration_s=(ended - measured) / 1e9,
                ),
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
    """Keep private evidence outside the checkout and accept only the registered date."""
    e.progress("start_no_model_load_MODEL_SPECS=[]_llm_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261101":
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
