"""REQ-REPORT-8608: bounded validation precedes atomic prototype publication."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import lean_resident_8608 as e
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]


def validate(raw: Path) -> list[Json]:
    """Collect original nodes once and preserve the exact disjoint execution union."""
    scratch = Path(tempfile.mkdtemp(prefix="carnot8608-validation-", dir="/var/tmp"))
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    collection = child(
        "collection",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *e.TESTS, *E2E],
        raw / "validation",
        deadline=90,
    )
    nodes = [
        line.strip()
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    atomic_json(
        raw / "original_nodes.json",
        dict(
            nodes=nodes,
            suites=e.TESTS + E2E,
            authority="unchanged original fixtures and historical CLI dates",
        ),
    )
    receipts, outcomes = [collection], []
    for i, suite in enumerate([*e.TESTS, *E2E]):
        selected = [n for n in nodes if n == suite or n.startswith(suite + "::")]
        if not selected:
            receipts.append(dict(name=f"empty_shard_{i}", passed=False))
            continue
        xml = scratch / f"nodes-{i}.xml"
        receipt = child(
            f"tests_{i}",
            [
                "env",
                "COVERAGE_RCFILE=" + str(config),
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
                "--basetemp=" + str(scratch / f"suite-{i}"),
            ],
            raw / "validation",
            deadline=240,
        )
        receipts.append(receipt)
        if receipt["passed"]:
            outcomes.extend(xml_nodes(xml))
    union = dict(
        name="exact_disjoint_node_union",
        passed=bool(nodes) and len(nodes) == len(set(nodes)) and sorted(nodes) == sorted(outcomes),
        collected=nodes,
        passed_node_ids=outcomes,
        original_authority="sealed original fixtures and original historical dates",
    )
    receipts.append(union)
    atomic_json(raw / "node_union.json", union)
    report = scratch / "coverage.json"
    plan = [
        (
            "coverage_combine",
            [sys.executable, "-m", "coverage", "combine", "--rcfile=" + str(config), "--keep"],
        ),
        (
            "coverage_report",
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + str(config),
                "--fail-under=100",
                "--show-missing",
            ],
        ),
        (
            "coverage_json",
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(report),
            ],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS]),
        (
            "strict_mypy",
            [
                str(e.ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *e.OWNED,
            ],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]
    receipts.extend(child(name, argv, raw / "validation", deadline=90) for name, argv in plan)
    coverage = json.loads(report.read_bytes())
    atomic_json(raw / "coverage.json", coverage)
    receipts.append(
        dict(
            name="new_owned_statement_coverage_100",
            passed=coverage["totals"]["percent_covered"] == 100,
            report=e.reference(raw / "coverage.json"),
        )
    )
    return receipts


def terminal(path: Path, raw: Path) -> Json:
    """Preserve unchanged validators' actual exits and findings before publication."""
    receipts = [
        child(name, [sys.executable, "-u", *argv], raw, deadline=60)
        for name, argv in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def run(output: Path, run_date: str) -> int:
    """A private publication qualifies the exact bytes later made reader-visible."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(raw, run_date)
    receipts = validate(raw)
    value = e.artifact(work, receipts, output)
    private_output = raw / "private" / output.name
    private = publish_primary(private_output, value, lambda p: terminal(p, raw / "private_checks"))
    reader = reader_receipt(
        value["task_id"],
        private_output.parent,
        field="lean_prototype_ready_score",
        expected=value["lean_prototype_ready_score"],
    )
    receipts.append(
        dict(
            name="private_publication_readers",
            passed=reader["passed"],
            publication=private,
            reader=reader,
        )
    )
    value = e.artifact(work, receipts, output)
    published = publish_primary(output, value, lambda p: terminal(p, raw / "terminal_checks"))
    report = read_bound_sidecar(output, Path(published["sidecar_path"]))
    atomic_json(raw / "terminal_validation.json", report)
    progress("8608_complete", 1, 0)
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Dates are validated at runtime, while replay uses the immutable invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            e.replay(args.cold_replay)
            return 0
        datetime.strptime(args.date, "%Y%m%d")
        return run(args.output, args.date)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8608] error={error}", file=sys.stderr, flush=True)
        return 1
