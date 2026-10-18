"""REQ-REPORT-8145: freeze owned checks and publish only checked natural bytes.

Private CLI panels qualify the transport only. They never replace a natural
measurement or grant readiness to a missing update category.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8145_v704_natural_service_cost as e
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]


def validation_plan(private: Path) -> list[CommandSpec]:
    """Scope coverage to added statements and keep all check output private."""
    commands = build_scoped_commands(
        e.ROOT,
        [e.TEST, "tests/python/test_primary_publication_7928.py"],
        e.OWNED[:2],
        static_paths=e.OWNED[2:],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    return [
        CommandSpec(
            c.name,
            (*c.argv, "--strict", "--follow-imports=silent")
            if c.name == "changed_module_mypy"
            else c.argv,
            c.scope,
            300,
        )
        for c in commands
    ]


def validators(path: Path) -> list[CommandSpec]:
    """Unchanged validators and cold CLI replay must finish normally."""
    py = str(e.ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)),
            "terminal",
            300,
        ),
        CommandSpec(
            "adversarial",
            (py, str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            300,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(e.ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            300,
        ),
    ]


def run_owned(commands: list[CommandSpec], private: Path) -> list[Json]:
    """Reuse the qualified supervisor with30s waits and real flushed exit counts."""
    return e.prior.execute(commands, private)


def main(argv: list[str] | None = None) -> int:
    """Validate first, measure once, authenticate reductions and atomically publish."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-small", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_failed")
        return 0 if passed else 1
    output = args.output.absolute()
    if args.private_small and output.is_relative_to(e.ROOT / "results"):
        parser.error("private panels require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8145-validation-"))
    os.environ["COVERAGE_FILE"] = str(private / ".coverage.repository")
    config = (
        dict(e.CONFIG, batches=[1, 4], warmups=1, repetitions=2) if args.private_small else e.CONFIG
    )
    plan = validation_plan(private)
    e.atomic_json(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            measurement_config=config,
            frozen_before_measurement=True,
        ),
    )
    receipts = [] if args.private_small else run_owned(plan, private)
    e.progress("owned_validation_complete", len(receipts), 0)
    data = e.inputs(args.root, raw)
    work: Json = dict(pairs=[], updates=[], warmups=[], recovery=[], config=config)
    if data["ready"] and all(r["passed"] for r in receipts):
        native, binding = e.prior.old.host.load_binding(data)
        e.atomic_json(raw / "loaded_binding_receipt.json", binding)
        e.progress("measurement_before")
        work = e.measure(data, native, raw, config)
        e.progress("measurement_after", len(work["pairs"]), 0)
    e.atomic_json(raw / "primitive_rows.json", work)
    e.atomic_json(
        raw / "independent_reduction.json",
        e.reduce_rows(work) if data["ready"] and work["pairs"] else dict(passed=not data["ready"]),
    )
    value = e.build(
        data, work, raw, receipts, args.date, time.monotonic() - began, args.private_small
    )
    if not args.private_small:
        health = run_owned(
            [
                CommandSpec(
                    "repository_health_once",
                    (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                    "repository_health_not_science_gate",
                    120,
                )
            ],
            private / "health",
        )
        value["repository_health"] = health
    value["reproducibility_checksum"] = e.checksum(value)
    candidate = private / (e.NAME + ".json")
    e.atomic_json(candidate, value)
    terminal_receipts = run_owned(validators(candidate), private / "terminal")
    if not all(r["passed"] and r["normal_exit"] for r in terminal_receipts):
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        e.atomic_json(
            raw / "terminal_validation.json", dict(passed=False, receipts=terminal_receipts)
        )
        e.progress("terminal_validation_failed")
        return 1
    # The exact final receipt bytes are themselves checked before publication.
    value["validation_receipts"] += terminal_receipts
    value["duration_s"] = time.monotonic() - began
    value["phase_spans"].extend(
        dict(phase=r["name"], duration_s=r.get("duration_s", 0))
        for r in value["validation_receipts"]
    )
    value["reproducibility_checksum"] = e.checksum(value)

    def terminal(path: Path) -> Json:
        checks = run_owned(validators(path), private / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    if output.exists():
        e.atomic_json(raw / "preserved_primary.json", json.loads(output.read_text()))
    publication = publish_primary(output, value, terminal)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            owned_checks_passed=value["required_checks_passed"],
            normal_process_exit=True,
        ),
    )
    e.progress("complete", value["completed_count"], value["censored_count"])
    return 0
