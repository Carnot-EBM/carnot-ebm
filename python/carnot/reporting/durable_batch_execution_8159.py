"""REQ-REPORT-8159: run owned checks before measuring or publishing batch claims.

Private panels exercise the same CLI with temporary outputs. They earn fixture
credit only; repository health observations remain separate from owned checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

from carnot.verify import durable_batch_8159 as e
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

execute = e.old.prior.execute


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze exact test, coverage and lint scopes so unrelated code cannot change the gate."""
    plan = build_scoped_commands(
        e.ROOT,
        [e.TEST],
        e.OWNED[:2],
        static_paths=[e.CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    commands = []
    for c in plan:
        argv = tuple(a + ",*/" + e.CLI if a.startswith("--include=") else a for a in c.argv)
        if c.name == "changed_module_mypy":
            argv += ("--strict", "--follow-imports=silent")
        commands.append(CommandSpec(c.name, argv, c.scope, 600))
    return commands


def validators(path: Path) -> list[CommandSpec]:
    """Check exact candidate bytes with the unchanged tools and an independent process."""
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


def main(argv: list[str] | None = None) -> int:
    """Validate, measure once, reconstruct the result and publish only checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-small", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--crash", choices=["before_commit", "after_commit"])
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    if args.worker:
        payload = json.loads(args.worker.read_text())
        native, _ = e.old.prior.old.host.load_binding(payload["data"])
        e.progress("crash_benchmark_before")
        e.transaction(
            payload["data"],
            native,
            payload["slot"],
            e.ARMS[-1],
            Path(payload["raw"]),
            crash=args.crash,
        )
        e.progress("crash_benchmark_after")
        return 0
    output = args.output.absolute()
    if args.private_small and output.is_relative_to(e.ROOT / "results"):
        parser.error("private panels require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8159-validation-"))
    os.environ["COVERAGE_FILE"] = str(private / ".coverage.repository")
    config = dict(e.CONFIG, batches=[1, 8], repetitions=1) if args.private_small else e.CONFIG
    plan = validation_plan(private)
    candidate = private / (e.NAME + ".json")
    health = CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_health_not_science_gate",
        1800,
    )
    e.atomic_json(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            terminal=[asdict(c) for c in validators(candidate)],
            repository_health=asdict(health),
            measurement_config=config,
            frozen_before_measurement=True,
        ),
    )
    e.progress("preconditions_before")
    data = e.inputs(args.root, raw)
    e.progress("preconditions_after", int(data["ready"]), 0)
    receipts = [] if args.private_small else execute(plan, raw)
    e.progress("owned_validation_after", len(receipts), 0)
    work = dict(config=config, pairs=[], durability=[])
    if data["ready"] and all(r["passed"] and r["normal_exit"] for r in receipts):
        e.progress("binding_load_before")
        native, binding = e.old.prior.old.host.load_binding(data)
        e.atomic_json(raw / "loaded_binding_receipt.json", binding)
        e.progress("binding_load_after")
        e.progress("measurement_before")
        work = e.measure(data, native, raw, config)
        e.progress("measurement_after", len(work["pairs"]), 0)
    e.atomic_json(raw / "primitive_rows.json", work)
    value = e.build(
        data, work, raw, receipts, args.date, time.monotonic() - began, args.private_small
    )
    if not args.private_small:
        value["repository_health"] = execute([health], raw / "health")
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(candidate, value)
    e.progress("independent_reduction_before")
    independent = e.replay(candidate)
    e.atomic_json(
        raw / "independent_reduction.json",
        dict(
            passed=independent,
            rows=value["natural_batch_rows"],
            paired_intervals=value["paired_speed_intervals"],
            independently_scored=True,
        ),
    )
    e.progress("independent_reduction_after", int(independent), 0)
    terminal = execute(validators(candidate), raw / "terminal")
    if not independent or not all(r["passed"] and r["normal_exit"] for r in terminal):
        value.update(
            host_batch_ready_score=0,
            required_checks_passed=False,
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
        )
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        e.atomic_json(raw / "terminal_validation.json", dict(passed=False, receipts=terminal))
        return 1
    value["validation_receipts"] += terminal
    value["duration_s"] = time.monotonic() - began
    value["phase_spans"].extend(
        dict(phase=r["name"], duration_s=r.get("duration_s", 0)) for r in receipts + terminal
    )
    value["raw_shard_hashes"].append(e.reference(raw / "independent_reduction.json"))
    value["reproducibility_checksum"] = e.checksum(value)

    def checked(path: Path) -> e.Json:
        checks = execute(validators(path), raw / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    if output.exists():
        e.atomic_json(raw / "preserved_primary.json", json.loads(output.read_text()))
    e.progress("publication_before")
    publication = publish_primary(output, value, checked)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            required_checks_passed=value["required_checks_passed"],
            normal_process_exit=True,
        ),
    )
    e.progress("complete", value["completed_count"], 0)
    return 0
