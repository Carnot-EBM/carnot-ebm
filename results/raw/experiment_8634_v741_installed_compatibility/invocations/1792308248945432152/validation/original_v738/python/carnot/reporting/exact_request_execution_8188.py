"""REQ-REPORT-8188: freeze owned validation before complete service measurement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.verify import exact_request_8188 as e
from carnot.reporting import complete_request_execution_8174 as qualified
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary


def validation_plan(private: Path) -> list[CommandSpec]:
    """Selectors belong to pytest; static tools receive real files only."""
    tests = [
        e.TEST,
        *[
            "tests/python/test_durable_batch_8159.py::test_complete_transactions_and_queue[all_at_once-"
            + c
            + "]"
            for c in ("cold", "warm", "restart")
        ],
        "tests/python/test_durable_batch_8159.py::test_crashes_deduplication",
    ]
    commands = build_scoped_commands(
        e.ROOT,
        tests,
        e.OWNED[:2],
        static_paths=[e.CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    plan = []
    for command in commands:
        args = command.argv
        if command.name in {"ruff_check", "ruff_format", "scoped_spec_coverage"}:
            args = tuple(a.split("::", 1)[0] for a in args)
        args = tuple(a + ",*/" + e.CLI if a.startswith("--include=") else a for a in args)
        if command.name == "changed_module_mypy":
            args += ("--strict", "--follow-imports=silent")
        plan.append(CommandSpec(command.name, args, command.scope, 600))
    return plan


def validators(path: Path) -> list[CommandSpec]:
    """Unmodified auditors inspect the exact candidate selected for publication."""
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


def execute(plan: list[CommandSpec], raw: Path) -> list[dict[str, Any]]:
    """Copy completed logs into hash-specific custody without reusing sealed paths."""
    receipts = qualified.execute(plan, raw / ("attempt-" + str(time.time_ns())))
    for row in receipts:
        if "log_path" in row:
            original = Path(row["log_path"])
            sealed = raw / "custody" / (e.sha256_file(original).split(":")[1] + ".log")
            sealed.parent.mkdir(parents=True, exist_ok=True)
            if not sealed.exists():
                shutil.copyfile(original, sealed)
            row.update(log_path=str(sealed), log_sha256=e.sha256_file(sealed))
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Close one measured attempt; external blocks retain honest terminal artifacts."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("8188_start", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--resume-evidence", type=Path)
    parser.add_argument("--repository-health-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("8188_cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    fixture = args.fixture_e2e is not None
    output = (args.fixture_e2e or args.output).absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("private fixtures require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8188-validation-"))
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
            repository_health_env=dict(PYTEST_ADDOPTS="-o addopts= --no-cov -n 4"),
            protocol=e.PROTOCOL,
            config=e.CONFIG,
            frozen_before_measurement=True,
        ),
    )
    e.progress("8188_preconditions_before", 0, 1)
    data = e.inputs(args.root, raw, fixture=fixture)
    e.progress("8188_preconditions_after", int(data["ready"]), 0)
    receipts = execute(plan[:1] if fixture else plan, raw / "validation")
    result: e.Json = dict(work={}, ledger=[], checks=[])
    if args.resume_evidence:
        if (args.resume_evidence / "live_result.json").is_file():
            prior = json.loads((args.resume_evidence / "live_result.json").read_text())
            frozen = json.loads((args.resume_evidence / "input_data.json").read_text())
        else:
            frozen, prior = e.resume_evidence(args.resume_evidence)
        if not e.validate_work(frozen, prior["work"]):
            raise ValueError("resume_evidence_custody")
        data, result = frozen, prior
    elif data["ready"] and all(r["passed"] and r["normal_exit"] for r in receipts):
        if fixture:
            data["slots"] = data["slots"][:2]
            e.progress("8188_binding_load_before", 0, 1)
            native, binding = e.prior.shared.host.old.prior.old.host.load_binding(data)
            e.atomic_json(raw / "loaded_binding_receipt.json", binding)
            e.progress("8188_binding_load_after", 1, 0)
            ledger = e.prior.shared.Ledger(raw / "ledger.json")
            result.update(
                work=e.measure(data, e.prior.shared.prior.FixtureRuntime(), native, raw, ledger),
                ledger=ledger.rows,
            )
        else:
            result = e.live(data, raw, private)
    value = e.build(data, result, raw, receipts, args.date, time.monotonic() - began, fixture)
    value["source_artifact_hashes"].extend(result.get("recovery_references", []))
    if not fixture:
        if args.repository_health_receipt:
            value["repository_health"] = json.loads(args.repository_health_receipt.read_text())[
                "receipts"
            ]
            value["source_artifact_hashes"].append(e.reference(args.repository_health_receipt))
        else:
            with patch.dict(os.environ, PYTEST_ADDOPTS="-o addopts= --no-cov -n 4"):
                value["repository_health"] = execute([health], raw / "health")
        e.atomic_json(
            raw / "repository_health_receipts.json", dict(receipts=value["repository_health"])
        )
    e.atomic_json(
        raw / "independent_reduction.json",
        dict(
            passed=e.validate_work(data, result.get("work", {})),
            reduction=e.reduce(result.get("work", {})),
        ),
    )
    value["raw_shard_hashes"] += [
        e.reference(raw / "validation_commands.json"),
        e.reference(raw / "independent_reduction.json"),
    ]
    value["reproducibility_checksum"] = e.prior.checksum(value)
    e.atomic_json(candidate, value)
    terminal = execute(validators(candidate), raw / "terminal")
    if not all(r["passed"] and r["normal_exit"] for r in terminal):
        value.update(
            cached_service_ready_score=0,
            equivalent_behavior_score=0,
            required_checks_passed=False,
            nfr01_met=False,
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
        )
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        return 1
    value["validation_receipts"] += terminal
    value["duration_s"] = time.monotonic() - began
    value["reproducibility_checksum"] = e.prior.checksum(value)

    def checked(path: Path) -> e.Json:
        """The publisher's lock keeps candidate bytes fixed during final auditing."""
        checks = execute(validators(path), raw / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    if output.exists():
        shutil.copyfile(
            output, raw / ("preserved-primary-" + e.sha256_file(output).split(":")[1] + ".json")
        )
    e.progress("8188_publication_before", 0, 1)
    publication = publish_primary(output, value, checked)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            required_checks_passed=value["required_checks_passed"],
            normal_process_exit=True,
        ),
    )
    e.progress("8188_complete", value["completed_count"], 0)
    return 0
