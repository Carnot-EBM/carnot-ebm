"""REQ-VERIFY-8566: bound CPU work, real child validation and atomic publication.

The thin CLI can publish a private disqualified fixture without recursive tests.
Production validates the scoped suite once and stores immutable child receipts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET
from typing import Any

from carnot.reporting import count_semantics_8566 as e
from carnot.reporting import count_semantics_report_8566 as report
from carnot.reporting.count_semantics_evidence_8566 import reference
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
CLI = str(e.ROOT / "scripts/experiments" / (e.NAME + ".py"))
TESTS = [
    "tests/python/test_count_semantics_8566.py",
    "tests/python/test_count_semantics_runner_8566.py",
]
OWNED = [
    f"python/carnot/reporting/count_semantics{s}_8566.py"
    for s in ("", "_evidence", "_report", "_runner")
] + [str(Path(CLI).relative_to(e.ROOT))]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]


def progress(phase: str) -> None:
    """Flush phase boundaries because data-only CPU work has no model heartbeat."""
    print("[exp8566] " + phase, flush=True)


def coverage_complete(value: Json, required: list[str] | None = None) -> bool:
    """Empty reports and any missing owned statement prevent qualification."""
    return (
        bool(value["files"])
        and set(required or []) <= set(value["files"])
        and all(r["summary"]["missing_lines"] == 0 for r in value["files"].values())
    )


def exact_union(original: list[str], shards: list[list[str]]) -> bool:
    """Keep original test identities, rejecting both lost and duplicate nodes."""
    flattened = [node for shard in shards for node in shard]
    return len(flattened) == len(set(flattened)) and set(flattened) == set(original)


def terminal_nodes(path: Path) -> list[Json]:
    """Count actual pytest terminal cases so skipped work cannot look complete."""
    return [
        dict(
            node_id=str(row.get("classname")).replace(".", "/") + ".py::" + str(row.get("name")),
            outcome="passed" if len(row) == 0 else "failed",
        )
        for row in ET.parse(path).iter("testcase")
    ]


def prepare(root: Path, raw: Path) -> Json:
    """Authenticate tools and disk-backed scratch before freezing primitive controls."""
    start = time.monotonic()
    progress("start date=20261016 no_model_load MODEL_SPECS=[]")
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8566-probe-", dir="/var/tmp") as scratch:
        probe = Path(scratch) / "write"
        with probe.open("wb") as stream:
            stream.write(b"disk-backed")
            stream.flush()
            os.fsync(stream.fileno())
        preconditions = dict(
            disk_backed_scratch=str(Path(scratch).parent),
            writable=probe.read_bytes() == b"disk-backed",
            free_disk_bytes=shutil.disk_usage(scratch).free,
            memory_available=Path("/proc/meminfo").read_text(),
            tools={
                name: (e.ROOT / ".venv/bin" / name).is_file()
                for name in ("python", "pytest", "ruff", "mypy")
            },
            measurement_deadline_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
            cuda="optional_not_required",
            board="optional_not_required",
            current_llm_calls=0,
        )
    progress("freeze_corpus")
    corpus = e.corpus()
    atomic_json(raw / "corpus.json", corpus)
    progress("authenticate_and_reduce_original")
    evidence = e.measure(root, raw)
    work = dict(
        raw=str(raw.absolute()),
        corpus=corpus,
        evidence=evidence,
        receipts=[],
        code=[
            reference(e.ROOT / p)
            for p in [
                *OWNED,
                *TESTS,
                "scripts/adversarial_verify.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
        preconditions=preconditions,
        duration_s=time.monotonic() - start,
        phases=[dict(phase="freeze_and_reduce", duration_s=time.monotonic() - start)],
    )
    atomic_json(raw / "work.json", work)
    return work


def run(output: Path, *, fixture: bool = False) -> int:
    """Complete bounded qualification once, preserving rejected children as failures."""
    start = time.monotonic()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = prepare(e.ROOT, raw)
    with TemporaryDirectory(prefix="carnot-8566-", dir="/var/tmp") as temporary:
        scratch = Path(temporary)
        if not fixture:
            validate(work, scratch)
        work["duration_s"] = time.monotonic() - start
        work["phases"].append(
            dict(
                phase="validation", duration_s=work["duration_s"] - work["phases"][0]["duration_s"]
            )
        )
        atomic_json(raw / "work.json", work)
        value = report.build(work, output)
        publication = publish_primary(output, value, lambda p: terminal(p, raw))
        consumer = reader_receipt(
            e.TASK,
            output.parent,
            field="count_controls_ready_score",
            expected=int(value["required_checks_passed"]),
        )
        atomic_json(
            raw / "terminal_validation.json", dict(publication=publication, consumer=consumer)
        )
    progress("complete verdict=" + value["verdict_class"])
    return 0


def terminal(path: Path, raw: Path) -> Json:
    """Validate exact final bytes with independent readers before atomic replacement."""
    progress("final_bytes_validation")
    scan = verify_artifact(path)
    receipt = child(
        "final_row_lint",
        [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        raw / "terminal_logs",
        deadline=60,
    )
    atomic_json(raw / "final_adversarial.json", scan)
    return dict(
        passed=report.replay(path)
        and not any(f["severity"] == "critical" for f in scan["flags"])
        and receipt["passed"],
        adversarial=scan,
        row_lint=receipt,
    )


def main(argv: list[str] | None = None) -> int:
    """Reject date drift and keep private fixtures away from production authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261016" or (
        args.fixture_output and args.fixture_output.resolve().is_relative_to(e.ROOT / "results")
    ):
        return 2
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        progress("cold_replay=" + str(passed))
        return 0 if passed else 1
    return run(
        args.fixture_output or e.ROOT / "results" / (e.NAME + ".json"),
        fixture=args.fixture_output is not None,
    )


def validate(work: Json, scratch: Path) -> None:
    """Freeze node IDs before bounded children and retain their exact disjoint union."""
    raw = Path(work["raw"])
    logs = raw / "logs"
    py = sys.executable
    receipts = work["receipts"]

    def command(name: str, argv: list[str], expected: int = 0, deadline: int = 180) -> Json:
        receipt = child(name, argv, logs, expected=expected, deadline=deadline)
        receipts.append(receipt)
        return receipt

    command(
        "original_adversarial",
        [
            py,
            "scripts/adversarial_verify.py",
            "--json",
            work["evidence"]["private_original"]["path"],
        ],
        expected=1,
        deadline=60,
    )

    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    pytest = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    collection = command(
        "collection", [py, "-m", "pytest", *pytest, "--collect-only", *TESTS, *E2E]
    )
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if "::" in line and line.startswith("tests/")
    ]
    groups = [
        [node for node in nodes if node.split("::")[0] in TESTS],
        [node for node in nodes if node == E2E[0]],
        [node for node in nodes if node == E2E[1]],
    ]
    work["node_union"] = dict(original_node_ids=nodes, child_node_ids=groups)
    work["union_passed"] = exact_union(nodes, groups) and bool(nodes)
    outcomes = []
    for name, group in zip(("owned", "e2e018", "e2e021"), groups, strict=True):
        junit = raw / (name + ".xml")
        command(
            name,
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *pytest,
                "--junitxml=" + str(junit),
                *group,
            ],
            deadline=300,
        )
        if junit.is_file():
            outcomes.extend(terminal_nodes(junit))
    work["node_union"]["terminal_node_outcomes"] = outcomes
    work["union_passed"] = (
        work["union_passed"]
        and exact_union(nodes, [[r["node_id"] for r in outcomes]])
        and all(r["outcome"] == "passed" for r in outcomes)
    )
    command("ruff_check", [py, "-m", "ruff", "check", *OWNED, *TESTS])
    command("ruff_format", [py, "-m", "ruff", "format", "--check", *OWNED, *TESTS])
    command("mypy", [py, "-m", "mypy", "--strict", "--follow-imports=silent", *OWNED])
    command("spec", [py, "scripts/check_spec_coverage.py", "--files", *TESTS])
    red = raw / "red_old_behavior.py"
    red.write_text(
        "from carnot.reporting.count_semantics_8566 import corpus,classify\n"
        "rows=classify(corpus())\n"
        "assert all(not r['old_critical'] for r in rows if r['expected']=='valid_discrete_identity')\n"
    )
    command("red_harness", [py, str(red)], expected=1)
    work["red_reference"] = reference(red)
    for name, mutation in [
        ("cold", "none"),
        ("missing", "absent"),
        ("owned_failure", "owned"),
        ("rehashed", "corpus"),
    ]:
        candidate = scratch / (name + ".json")
        atomic_json(raw / "work.json", work)
        value = report.build(work, candidate)
        if mutation == "absent":
            value["primitive_reference"]["path"] = str(scratch / "absent")
        elif mutation == "owned":
            value["required_checks_passed"] = True
        elif mutation == "corpus":
            value["expected_classification_rows"] = []
        value["reproducibility_checksum"] = report.checksum(value)
        atomic_json(candidate, value)
        command(
            name,
            [py, "-u", CLI, "--cold-replay", str(candidate)],
            expected=0 if name == "cold" else 1,
        )
    candidate = scratch / "scan.json"
    atomic_json(raw / "work.json", work)
    atomic_json(candidate, report.build(work, candidate))
    command(
        "adversarial", [py, "scripts/adversarial_verify.py", "--json", str(candidate)], deadline=60
    )
    command(
        "row_lint",
        [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        deadline=60,
    )
    command("combine", [py, "-m", "coverage", "combine", "--rcfile=" + str(config), str(raw)])
    command(
        "coverage",
        [py, "-m", "coverage", "json", "--rcfile=" + str(config), "-o", str(raw / "coverage.json")],
    )
    work["coverage_complete"] = coverage_complete(
        json.loads((raw / "coverage.json").read_bytes()), OWNED
    )
