"""REQ-REPORT-9093-V775: publish a validated inventory without requiring positive science."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import service_cost_9093 as d
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.matched_corpus_runner_8949 import validation_plan
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
]


def plan(private: Path) -> list[CommandSpec]:
    """Freeze owned tests, actual CLI coverage and existing consumer checks together."""
    commands = validation_plan(private, producer=d)
    return [
        *commands,
        CommandSpec(
            "direct_consumers",
            (str(d.ROOT / ".venv/bin/pytest"), "-n0", "-o", "addopts=", "--no-cov", "-q", *DIRECT),
            "direct_consumers",
            180,
        ),
    ]


def qualify(work: Json, validation: Json) -> bool:
    """A boolean cannot replace frozen commands, exact logs or measured SQLite coverage."""
    receipts = validation.get("receipts", [])
    expected = work["validation_scope_manifest"]
    if len(receipts) != len(expected) or not expected:
        return False
    for command, receipt in zip(expected, receipts, strict=True):
        log = Path(receipt.get("log_path", ""))
        log = log if log.is_absolute() else d.ROOT / log
        if (
            receipt.get("name") != command["name"]
            or receipt.get("command_argv") != command["argv"]
            or receipt.get("exit_code") != 0
            or receipt.get("passed") is not True
            or receipt.get("timed_out") is not False
            or d.ref(log)["sha256"] != receipt.get("log_sha256")
        ):
            return False
    return bool(verify_measured(validation.get("coverage", {}), d.OWNED))


def measure(
    root: Path, raw: Path, scope: list[Json], terminal_scope: list[Json] | None = None
) -> Json:
    """Freeze commands before a bounded fixture and immutable source reduction."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700)
    atomic_json(
        raw / "validation_scope.json", dict(commands=scope, terminal_commands=terminal_scope)
    )
    manifest = d.freeze(root, raw)
    result = d.inventory(manifest)
    fixture = d.fixture_replay(raw / "fixture")
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            work="immutable inventory and bounded CPU mechanics fixture", llm_calls=0
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    atomic_json(raw / "current_work.json", receipt)
    sources = [
        d.ref(d.ROOT / p)
        for p in [
            *d.OWNED,
            *d.TESTS,
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    work = dict(
        manifest=manifest,
        result=result,
        fixture_rows=fixture,
        raw=str(raw),
        current_work_reference=d.ref(raw / "current_work.json"),
        sources=sources,
        validation_scope_manifest=scope,
        terminal_scope_manifest=terminal_scope or [],
        duration_s=(end - start) / 1e9,
    )
    atomic_json(raw / "work.json", work)
    for name in ("current_work", "work", "validation_scope"):
        (raw / (name + ".json")).chmod(0o400)
    d.progress("measurement_complete_validation_begins")
    return work


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Missing science does not disqualify a truthful inventory; owned failures do."""
    passed = qualify(work, validation)
    result = d.inventory(work["manifest"])
    verdict = "blocked" if fixture else "null" if passed else "disqualified"
    value = dict(
        result,
        schema="carnot.v775_service_cost.v1",
        experiment_id=9093,
        task_id=d.TASK,
        milestone="2026.11.775",
        run_date="20261115",
        random_seed=775093,
        honest_verdict="complete_blocked_private_unqualified_inventory"
        if fixture
        else "complete_null_inventory_unavailable_service"
        if passed
        else "complete_disqualified_owned_validation",
        verdict_class=verdict,
        status="blocked" if fixture else "success" if passed else "error",
        MODEL_SPECS=[],
        model_specs=[],
        current_llm_invocations=0,
        model_invoked=False,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Authenticated accounting inventory only; no scientific service ratio or hardware speedup.",
        exposure_scope="Graph labels are executable-oracle and any graph benefit is circular_positive; human LPHB is exposed development, never independent generalization.",
        duration_s=work["duration_s"],
        required_checks_passed=passed,
        cost_inventory_ready_score=int(passed and not fixture),
        complete_service_cost_ready_score=0,
        fixture_only=fixture,
        fixture_operation_rows=work["fixture_rows"],
        code_config_hashes=work["sources"],
        source_references=work["manifest"]["entries"],
        current_work_receipts=[work["current_work_reference"]],
        raw_shard_hashes=[d.ref(Path(work["raw"]) / "work.json")],
        owned_statement_coverage=validation.get("coverage", {}),
        validation_receipts=validation.get("receipts", []),
        validation_scope_manifest=work["validation_scope_manifest"],
        repository_health=validation.get("repository_health", {}),
        flagged_adversarial=not passed and not fixture,
        methodology_note="Freeze declared paths, original producer bindings and all available invocation work receipts. Task counts are administrative; requests, repeated seeds and fixture rows never multiply independent units. Charge only within an authenticated interval domain using union. Parent intervals include children; shared load has no per-request allocation. Missing scientific timings, memory and power integrals remain unavailable. No model is loaded, no external failure is retried, and no deployment is recommended.",
        acceptance_gates=dict(
            owned_checks=dict(passed=passed),
            inventory_complete=dict(passed=passed and not fixture),
            scientific_service_complete=dict(
                passed=False,
                reason="scientific request clocks, model/device attribution and denominators unavailable",
            ),
        ),
    )
    if fixture:
        value["gate_check_summary"].append(
            d.gate(work["raw"], "owned_validation", True, False, "private_unqualified", d.TASK)
        )
    value["field_principles"] = {
        k: "Authenticate raw operands and distinguish measured, censored and unavailable work; no benefit or zero cost is inferred."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute every headline from primitive receipts, including rehashed tampering."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_bytes())
        unsigned = dict(value)
        if unsigned.pop("reproducibility_checksum") != canonical_hash(unsigned):
            raise ValueError("checksum_drift")
        references = (
            value["raw_shard_hashes"] + value["code_config_hashes"] + value["current_work_receipts"]
        )
        for reference in references:
            if d.ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
                raise ValueError("receipt_hash_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        if d.inventory(work["manifest"]) != work["result"]:
            raise ValueError("primitive_reduction_drift")
        for row in work["fixture_rows"]:
            reference = row["source_reference"]
            if d.ref(Path(reference["path"])) != reference:
                raise ValueError("fixture_hash_drift")
        validation = json.loads((Path(work["raw"]) / "validation.json").read_bytes())
        passed = build(work, validation, value["fixture_only"]) == value
        d.progress("after_cold_replay")
        return dict(passed=passed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("cold_replay_refused")
        return dict(passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Keep measurements private until frozen checks and independent replay succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--sources", type=Path, default=d.ROOT)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261115":
        parser.error("requires --date 20261115")
    fixture = args.private_output is not None
    output = args.private_output or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("private output must be outside the checkout")
    private = Path(tempfile.mkdtemp(prefix="carnot-9093-", dir="/var/tmp"))
    commands = plan(private / "checks")
    scope = [dict(asdict(c), argv=list(c.argv)) for c in commands]
    raw = private / "invocation"
    py = str(d.ROOT / ".venv/bin/python")
    candidate = raw / "candidate.json"
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
    work = measure(args.sources, raw, scope, [dict(asdict(c), argv=list(c.argv)) for c in terminal])
    atomic_json(raw / "terminal_scope.json", dict(commands=[asdict(c) for c in terminal]))
    validation: Json = dict(receipts=[], coverage={}, repository_health={})
    if not fixture:
        d.progress("before_owned_validation")
        validation["receipts"] = run_commands(
            d.ROOT, commands, log_dir=raw / "validation", heartbeat_s=15
        )
        data = private / "checks" / ".coverage"
        if data.is_file():
            validation["coverage"] = measured([data], d.OWNED, raw / "coverage")
        d.progress("before_full_python_suite_once")
        validation["repository_health"] = dict(
            receipts=run_commands(
                d.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                        "broader_repository_health",
                        900,
                    )
                ],
                log_dir=raw / "repository_health",
                heartbeat_s=15,
            ),
            affects_owned_readiness=False,
            historical_failures_preserved=True,
        )
        d.progress("after_full_python_suite_once")
    atomic_json(raw / "validation.json", validation)
    value = build(work, validation, fixture)
    atomic_json(candidate, value)
    if not fixture and not value["required_checks_passed"]:
        d.progress("owned_validation_failed_candidate_retained_privately")
        return 1

    def validator(path: Path) -> Json:
        report = replay(path)
        if not fixture:
            receipts = run_commands(d.ROOT, terminal, log_dir=raw / "terminal", heartbeat_s=15)
            report.update(
                passed=report["passed"] and all(r["passed"] for r in receipts),
                validation_receipts=receipts,
            )
        return report

    d.progress("before_primary_publication")
    publish_primary(output, value, validator)
    d.progress("after_primary_publication")
    print(json.dumps(dict(output=str(output), verdict=value["honest_verdict"])), flush=True)
    return 0 if fixture or value["required_checks_passed"] else 1
