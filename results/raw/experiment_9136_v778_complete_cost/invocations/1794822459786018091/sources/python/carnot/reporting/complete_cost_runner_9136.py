"""REQ-REPORT-9136-V778: validate a pre-load inventory before atomic publication."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import complete_cost_9136 as d
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.matched_corpus_runner_8949 import validation_plan
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]


def plan(private: Path) -> list[CommandSpec]:
    """Freeze coverage, lint, strict types, spec references and affected readers first."""
    return [
        *(
            replace(c, argv=(*c.argv, "--noconftest"))
            if c.name in {"owned_unit_child", "e2e018", "e2e021"}
            else c
            for c in validation_plan(private, producer=d)
        ),
        CommandSpec(
            "direct_consumers",
            (
                str(d.ROOT / ".venv/bin/pytest"),
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "--noconftest",
                "-q",
                "tests/python/test_current_work_receipt.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_service_cost_9093.py",
                "tests/python/test_hardware_boundary_9079.py",
            ),
            "direct_consumers",
            600,
        ),
    ]


def validate(commands: list[CommandSpec], private: Path, raw: Path) -> Json:
    """Use real child exits, immutable logs and SQLite statement coverage as evidence."""
    receipts = run_commands(d.ROOT, commands, log_dir=raw / "validation", heartbeat_s=15)
    coverage = measured([private / ".coverage"], d.OWNED, raw / "coverage")
    return dict(
        passed=all(r["passed"] for r in receipts) and verify_measured(coverage, d.OWNED),
        receipts=receipts,
        coverage=coverage,
    )


def main(argv: list[str] | None = None) -> int:
    """Keep checks private until terminal validation accepts the exact candidate bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--sources", type=Path, default=d.ROOT)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = d.replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261116":
        parser.error("requires --date 20261116")
    output = args.private_output or d.ROOT / "results" / (d.NAME + ".json")
    if args.private_output and output.resolve().is_relative_to(d.ROOT):
        parser.error("private output must be outside checkout")
    private = Path(tempfile.mkdtemp(prefix="carnot-9136-", dir="/var/tmp"))
    commands = plan(private / "checks")
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    candidate = raw / "candidate.json"
    py = str(d.ROOT / ".venv/bin/python")
    terminal = [
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(candidate), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            120,
        ),
    ]
    health = [
        CommandSpec(
            "global_python_suite",
            (str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            600,
        )
    ]
    scope = [dict(asdict(c), argv=list(c.argv)) for c in [*commands, *health, *terminal]]
    atomic_json(raw / "validation_scope.json", dict(commands=scope))
    work = d.collect(args.sources, raw)
    work["fixture_only"] = args.private_output is not None
    work["validation_scope_manifest"] = scope
    work["sources"] = []
    for label in [*d.OWNED, *d.TESTS, "ops/exclusion_manifest.yaml"]:
        original = d.ROOT / label
        target = raw / "sources" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        work["sources"].append(dict(d.ref(target), source_path=str(original)))
    atomic_json(raw / "work.json", {k: v for k, v in work.items() if k != "work_reference"})
    work["work_reference"] = d.ref(raw / "work.json")
    validation: Json = dict(passed=True, receipts=[], coverage={})
    if not args.private_output:
        d.progress("before_owned_validation")
        validation = validate(commands, private / "checks", raw)
        d.progress("after_owned_validation")
        if validation["passed"]:
            validation["repository_health"] = run_commands(
                d.ROOT, health, log_dir=raw / "repository_health", heartbeat_s=30
            )
    atomic_json(raw / "validation.json", validation)
    value = d.build(work, validation, args.date)
    atomic_json(candidate, value)
    if not validation["passed"]:
        d.progress("owned_checks_failed_candidate_retained")
        return 1

    def validator(path: Path) -> Json:
        report = d.replay(path)
        if not args.private_output:
            receipts = run_commands(d.ROOT, terminal, log_dir=raw / "terminal", heartbeat_s=15)
            report.update(
                passed=report["passed"] and all(r["passed"] for r in receipts),
                validation_receipts=receipts,
            )
        return report

    d.progress("before_atomic_publication")
    publish_primary(output, value, validator)
    d.progress("after_atomic_publication")
    print(json.dumps(dict(output=str(output), honest_verdict=value["honest_verdict"])), flush=True)
    return 0
