"""REQ-REPORT-8174: freeze validation before independent full-request measurement.

Private scripted traffic exercises publication and replay but receives no live
inference credit. Repository health stays separate from the owned science gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

from carnot.verify import complete_request_8174 as e
from carnot.reporting import durable_batch_execution_8159 as qualified
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

execute = qualified.execute


def validation_plan(private: Path) -> list[CommandSpec]:
    """Only pytest accepts selectors; static tools must receive actual file paths."""
    tests = [
        e.TEST,
        "tests/python/test_durable_batch_8159.py::test_complete_transactions_and_queue",
        "tests/python/test_durable_batch_8159.py::test_crashes_deduplication",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    commands = build_scoped_commands(
        e.ROOT,
        tests,
        e.OWNED[:2],
        static_paths=[e.CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    result = []
    for command in commands:
        args = command.argv
        if command.name in {"ruff_check", "ruff_format", "scoped_spec_coverage"}:
            args = tuple(a.split("::", 1)[0] for a in args)
        args = tuple(a + ",*/" + e.CLI if a.startswith("--include=") else a for a in args)
        if command.name == "changed_module_mypy":
            args += ("--strict", "--follow-imports=silent")
        result.append(CommandSpec(command.name, args, command.scope, 600))
    return result


def validators(path: Path) -> list[CommandSpec]:
    """The independent process and unchanged auditors check exact candidate bytes."""
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
    """Measure once after owned checks, then publish only independently checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("8174_start", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--resume-evidence", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("8174_cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    fixture = args.fixture_e2e is not None
    output = (args.fixture_e2e or args.output).absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("private fixtures require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8174-validation-"))
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
    e.progress("8174_preconditions_before", 0, 1)
    resumed = e.resume_evidence(args.resume_evidence) if args.resume_evidence else None
    data = resumed[0] if resumed else e.inputs(args.root, raw, fixture=fixture)
    e.progress("8174_preconditions_after", int(data["ready"]), 0)
    receipts = execute(plan[:1] if fixture else plan, raw / "validation")
    result: e.Json = resumed[1] if resumed else dict(work={}, ledger=[], checks=[])
    if not resumed and data["ready"] and all(r["passed"] and r["normal_exit"] for r in receipts):
        if fixture:
            e.progress("8174_binding_load_before", 0, 1)
            native, binding = e.shared.host.old.prior.old.host.load_binding(data)
            e.atomic_json(raw / "loaded_binding_receipt.json", binding)
            e.progress("8174_binding_load_after", 1, 0)
            ledger = e.shared.Ledger(raw / "ledger.json")
            result.update(
                work=e.measure(data, e.shared.prior.FixtureRuntime(), native, raw, ledger),
                ledger=ledger.rows,
            )
        else:
            result = e.live(data, raw, private)
    value = e.build(data, result, raw, receipts, args.date, time.monotonic() - began, fixture)
    value["source_artifact_hashes"].extend(result.get("recovery_references", []))
    value["repository_health"] = [] if fixture else execute([health], raw / "health")
    value["raw_shard_hashes"].append(e.reference(raw / "validation_commands.json"))
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(candidate, value)
    e.progress("8174_independent_reduction_before", 0, 1)
    independent = e.replay(candidate)
    e.atomic_json(
        raw / "independent_reduction.json",
        dict(passed=independent, clocks_recomputed=True, decisions_recomputed=True),
    )
    e.progress("8174_independent_reduction_after", int(independent), 0)
    terminal = execute(validators(candidate), raw / "terminal")
    if not independent or not all(r["passed"] and r["normal_exit"] for r in terminal):
        value.update(
            complete_service_ready_score=0,
            equivalent_behavior_score=0,
            nfr01_met=False,
            required_checks_passed=False,
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
        )
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        return 1
    value["validation_receipts"] += terminal
    value["duration_s"] = time.monotonic() - began
    value["raw_shard_hashes"].append(e.reference(raw / "independent_reduction.json"))
    value["reproducibility_checksum"] = e.checksum(value)

    def checked(path: Path) -> e.Json:
        """Publication validates the exact locked candidate and binds the sidecar hash."""
        checks = execute(validators(path), raw / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    if output.exists():
        e.atomic_json(raw / "preserved_primary.json", json.loads(output.read_text()))
    e.progress("8174_publication_before", 0, 1)
    publication = publish_primary(output, value, checked)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            required_checks_passed=value["required_checks_passed"],
            normal_process_exit=True,
        ),
    )
    e.progress("8174_complete", value["completed_count"], 0)
    return 0
