"""REQ-REPORT-8160: validate and publish a bounded shared acquisition comparison.

Private traffic checks transport only. Current model work uses the qualified
owned CUDA worker; repository health remains a separate observation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

from carnot.verify import shared_acquisition_8160 as e
from carnot.reporting import durable_batch_execution_8159 as qualified
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

execute = qualified.execute


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze explicit owned paths and actual crash transport tests before measurement."""
    commands = build_scoped_commands(
        e.ROOT,
        [e.TEST, "tests/python/test_durable_batch_8159.py::test_crashes_deduplication"],
        e.OWNED[:2],
        static_paths=[e.CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    result = []
    for command in commands:
        argv = tuple(a + ",*/" + e.CLI if a.startswith("--include=") else a for a in command.argv)
        if command.name == "changed_module_mypy":
            argv += ("--strict", "--follow-imports=silent")
        result.append(CommandSpec(command.name, argv, command.scope, 600))
    return result


def validators(path: Path) -> list[CommandSpec]:
    """Bind candidate bytes to unchanged auditors and a separate cold replay process."""
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
    """Seal public inputs, qualify owned checks, measure once and publish checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("8160_start", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if e.replay(args.cold_replay) else 1
    fixture = args.fixture_e2e is not None
    output = (args.fixture_e2e or args.output).absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("private fixtures require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8160-validation-"))
    os.environ["COVERAGE_FILE"] = str(private / ".coverage.repository")
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
            measurement_config=e.CONFIG,
            frozen_before_measurement=True,
        ),
    )
    e.progress("8160_preconditions_before", 0, 1)
    data = e.inputs(args.root, raw)
    if fixture:
        data["slots"] = data["slots"][:2]
    e.seal(data, raw, fixture=fixture)
    e.progress("8160_preconditions_after", int(data["ready"]), 0)
    receipts = execute(plan[:1] if fixture else plan, raw)
    repository_health = [] if fixture else execute([health], raw / "health")
    result: e.Json = dict(work={}, ledger=[], checks=[])
    if data["ready"] and all(r["passed"] and r["normal_exit"] for r in receipts):
        if fixture:
            e.progress("8160_binding_load_before", 0, 1)
            native, binding = e.host.old.prior.old.host.load_binding(data)
            e.atomic_json(raw / "loaded_binding_receipt.json", binding)
            e.progress("8160_binding_load_after", 1, 0)
            ledger = e.Ledger(raw / "ledger.json")
            work = e.measure(data, e.prior.FixtureRuntime(), native, raw, ledger)
            result.update(work=work, ledger=ledger.rows)
        else:
            result = e.live(data, raw, private)
    e.atomic_json(raw / "input_data.json", data)
    value = e.build(data, result, raw, receipts, args.date, time.monotonic() - began, fixture)
    value["repository_health"] = repository_health
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(candidate, value)
    e.progress("8160_independent_reduction_before", 0, 1)
    independent = e.replay(candidate)
    e.atomic_json(
        raw / "independent_reduction.json",
        dict(
            passed=independent,
            composed_cost_rows=value["composed_cost_rows"],
            paired_speed_intervals=value["paired_speed_intervals"],
            zero_arithmetic_ceiling=value["zero_arithmetic_ceiling"],
        ),
    )
    e.progress("8160_independent_reduction_after", int(independent), 0)
    terminal = execute(validators(candidate), raw / "terminal")
    if not independent or not all(r["passed"] and r["normal_exit"] for r in terminal):
        value.update(
            acquisition_composition_ready_score=0,
            required_checks_passed=False,
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
        )
        value["reproducibility_checksum"] = e.checksum(value)
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        e.atomic_json(raw / "terminal_validation.json", dict(passed=False, receipts=terminal))
        return 1
    value["validation_receipts"] += terminal
    value["raw_shard_hashes"].extend(
        [
            e.reference(raw / "validation_commands.json"),
            e.reference(raw / "independent_reduction.json"),
        ]
    )
    value["duration_s"] = time.monotonic() - began
    value["phase_spans"].extend(
        dict(phase=r["name"], duration_s=r.get("duration_s", 0)) for r in receipts + terminal
    )
    value["reproducibility_checksum"] = e.checksum(value)

    def checked(path: Path) -> e.Json:
        checks = execute(validators(path), raw / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    if output.exists():
        e.atomic_json(raw / "preserved_primary.json", json.loads(output.read_text()))
    e.progress("8160_publication_before", 0, 1)
    publication = publish_primary(output, value, checked)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            required_checks_passed=value["required_checks_passed"],
            normal_process_exit=True,
        ),
    )
    e.progress("8160_complete", value["completed_count"], 0)
    return 0
