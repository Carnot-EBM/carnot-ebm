"""REQ-REPORT-9094-V775-PROVENANCE: frozen checks qualify CPU numerical evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import susceptibility_9094 as d
from carnot.reporting import susceptibility_evidence_9094 as e
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


def plan(private: Path) -> list[CommandSpec]:
    """Freeze owned statement coverage, direct readers and all applicable private E2E checks."""
    commands = validation_plan(private, producer=d)
    return [
        *commands,
        CommandSpec(
            "direct_consumers",
            (
                str(d.ROOT / ".venv/bin/pytest"),
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
            ),
            "direct",
            180,
        ),
        CommandSpec(
            "e2e020",
            (
                str(d.ROOT / ".venv/bin/pytest"),
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "--basetemp=" + str(private / "e2e020"),
                "tests/python/test_hard_exit_learning_qualification_8206.py",
            ),
            "private_e2e",
            240,
        ),
    ]


def qualify(work: Json, validation: Json) -> bool:
    """Re-read exact command logs and actual coverage SQLite rather than trusting a percentage."""
    receipts, scope = validation.get("receipts", []), work["validation_scope_manifest"]
    if not scope or len(receipts) != len(scope):
        return False
    for command, receipt in zip(scope, receipts, strict=True):
        if (
            receipt.get("name") != command["name"]
            or receipt.get("command_argv") != command["argv"]
            or receipt.get("exit_code") != 0
            or receipt.get("passed") is not True
            or receipt.get("timed_out") is not False
            or e.ref(Path(receipt.get("log_path", "")))["sha256"] != receipt.get("log_sha256")
        ):
            return False
    return bool(verify_measured(validation.get("coverage", {}), d.OWNED))


def measure(
    root: Path, raw: Path, source_dir: Path, scope: list[Json], terminal: list[Json]
) -> Json:
    """Authenticate and freeze commands before any coupling measurement starts."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700)
    atomic_json(
        raw / "validation_scope.json",
        dict(commands=scope, terminal_commands=terminal, parameter_grid=d.GRID),
    )
    evidence = e.authenticate(root, raw, source_dir)
    d.progress("authentication_complete")
    sources = [
        e.retain(d.ROOT / name, raw)
        for name in [
            *d.OWNED,
            *d.TESTS,
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            "openspec/capabilities/research-reporting/spec.md",
        ]
    ]
    references = []
    if all(c["passed"] for c in evidence["gate_check_summary"] if c["required"]):
        for seed in d.SEEDS:
            case = d.measure_case(seed)
            path = raw / "cases" / f"{seed}.json"
            atomic_json(path, case)
            path.chmod(0o400)
            references.append(e.ref(path))
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.parent.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_numerical_diagnostic",
        inference_substrate_details=dict(
            work="phase energy derivatives and linear solves; zero model loads"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
    )
    atomic_json(raw / "current_work.json", receipt)
    work = dict(
        evidence=evidence,
        case_references=references,
        code_config_hashes=sources,
        current_work_receipts=[e.ref(raw / "current_work.json")],
        raw=str(raw),
        duration_s=(ended - started) / 1e9,
        validation_scope_manifest=scope,
        terminal_scope_manifest=terminal,
    )
    atomic_json(raw / "work.json", work)
    for name in ("validation_scope", "current_work", "work"):
        (raw / (name + ".json")).chmod(0o400)
    d.progress("measurement_complete_validation_begins")
    return work


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """A failed mechanism is a retained null; missing sources block dependent measurements."""
    cases = [json.loads(Path(r["path"]).read_bytes()) for r in work["case_references"]]
    summary = d.summarize(cases)
    checked = qualify(work, validation)
    evidence = work["evidence"]
    failed = [c for c in evidence["gate_check_summary"] if c["required"] and not c["passed"]]
    verdict = (
        "blocked"
        if failed or fixture
        else "disqualified"
        if not checked
        else "circular_positive"
        if summary["mechanism_reproduced"]
        else "null"
    )
    suffix = (
        "blocked_" + failed[0]["check"]
        if failed
        else "blocked_private_unqualified"
        if fixture
        else "disqualified_owned_validation"
        if not checked
        else "circular_positive_susceptibility_mechanism"
        if summary["mechanism_reproduced"]
        else "null_susceptibility_mechanism"
    )
    ready = checked and not fixture and not failed and len(cases) == 32
    checks = list(evidence["gate_check_summary"])
    if fixture:
        checks.append(
            e.gate(Path(work["raw"]), "owned_validation", True, False, "private_unqualified", True)
        )
    value = dict(
        schema="carnot.v775_susceptibility.v1",
        experiment_id=9094,
        task_id=d.TASK,
        milestone="2026.11.775",
        run_date="20261115",
        random_seed=775094,
        honest_verdict="complete_" + suffix,
        verdict_class=verdict,
        status="blocked"
        if verdict == "blocked"
        else "error"
        if verdict == "disqualified"
        else "success",
        MODEL_SPECS=[],
        model_specs=[],
        current_llm_invocations=0,
        model_invoked=False,
        inference_substrate="cpu_numerical_diagnostic",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="CPU numerical equation diagnostic only; no generator, inference-speed, joule or hardware benefit.",
        exposure_scope="Synthetic coupling cases. Graph truth is exact BFS; any graph success is circular_positive. Human LPHB remains exposed development, never independent generalization.",
        hardware_measurement_scope="CPU numerical only",
        parameter_grid=d.GRID,
        summary=summary,
        rows=summary["rows"],
        intended_attempted_completed_failed_censored_counts=summary["counts"],
        effective_independent_groups=summary["effective_independent_groups"],
        paired_rows=summary["paired_rows"],
        numerical_diagnostic_ready_score=int(ready),
        susceptibility_mechanism_reproduced_score=int(ready and summary["mechanism_reproduced"]),
        hardware_map_ready_score=int(checked and not fixture),
        operation_mapping_rows=e.operation_map(evidence),
        unavailable_cost_inputs=evidence["unavailable_cost_inputs"],
        historical_model_provenance=evidence["historical_model_provenance"],
        code_config_hashes=work["code_config_hashes"],
        source_references=evidence["source_references"],
        current_work_receipts=work["current_work_receipts"],
        duration_s=work["duration_s"],
        gate_check_summary=checks,
        required_checks_passed=checked,
        flagged_adversarial=not checked and not fixture,
        validation_receipts=validation.get("receipts", []),
        owned_statement_coverage=validation.get("coverage", {}),
        repository_health=validation.get("repository_health", {}),
        fixture_only=fixture,
        validation_scope_manifest=work["validation_scope_manifest"],
        terminal_scope_manifest=work["terminal_scope_manifest"],
        raw_shard_hashes=[e.ref(Path(work["raw"]) / "work.json"), *work["case_references"]],
        acceptance_gates=dict(
            owned_checks=checked,
            source_authentication=not failed,
            complete_registered_grid=len(cases) == 32,
            equal_binary_energies=summary["equal_energy_agreement"],
            analytic_derivatives=summary["derivatives_agree"],
            susceptibility_mechanism=summary["mechanism_reproduced"],
            physical_device_measurement=False,
            independent_generalization=False,
        ),
        methodology_note="Primary-equation CPU implementation, all64 binary configurations,32 independently seeded coupling cases, fixed paired parameter grid. Test equations before measuring; independent unit-norm output forcing separates susceptibility from the binary cosine-loss zero. No ridge, fitting, grid expansion, significance claim or hardware dispatch. Finite displacement error slopes are descriptive and retain floating-point floors. Operation maps cannot establish speed or energy savings.",
    )
    for key in (
        "binary_energy_rows",
        "derivative_rows",
        "susceptibility_rows",
        "finite_perturbation_rows",
        "stability_rows",
    ):
        value[key] = [r for c in cases for r in c[key]]
    value["field_principles"] = {
        key: "Preserve authenticated primitive numerical evidence, exact scope and failed checks; repeated conditions do not create independent units or device benefit."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild public headlines and rows from immutable case receipts and checked log bytes."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_bytes())
        unsigned = dict(value)
        if unsigned.pop("reproducibility_checksum") != canonical_hash(unsigned):
            raise ValueError("checksum_drift")
        references = (
            value["raw_shard_hashes"]
            + value["code_config_hashes"]
            + value["current_work_receipts"]
            + value["source_references"]
        )
        for reference in references:
            if (
                reference["sha256"] is not None
                and e.ref(Path(reference["path"]))["sha256"] != reference["sha256"]
            ):
                raise ValueError("receipt_hash_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        validation = json.loads((Path(work["raw"]) / "validation.json").read_bytes())
        passed = build(work, validation, value["fixture_only"]) == value
        d.progress("after_cold_replay")
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("cold_replay_refused")
        return dict(passed=False, replay_passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Private fixtures cannot grant readiness or recursively validate themselves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--sources", type=Path, default=d.ROOT)
    parser.add_argument("--source-dir", type=Path, default=e.SOURCE_DIR)
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
    private = Path(tempfile.mkdtemp(prefix="carnot-9094-", dir="/var/tmp"))
    commands = plan(private / "checks")
    scope = [dict(asdict(c), argv=list(c.argv)) for c in commands]
    raw, py = private / "invocation", str(d.ROOT / ".venv/bin/python")
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
    work = measure(
        args.sources,
        raw,
        args.source_dir,
        scope,
        [dict(asdict(c), argv=list(c.argv)) for c in terminal],
    )
    validation: Json = dict(receipts=[], coverage={}, repository_health={})
    if not fixture:
        d.progress("before_owned_validation")
        validation["receipts"] = run_commands(
            d.ROOT, commands, log_dir=raw / "validation", heartbeat_s=15
        )
        data = private / "checks/.coverage"
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
            repeated=False,
        )
        d.progress("after_full_python_suite_once")
    atomic_json(raw / "validation.json", validation)
    (raw / "validation.json").chmod(0o400)
    value = build(work, validation, fixture)
    atomic_json(candidate, value)
    if not fixture and not value["required_checks_passed"]:
        d.progress("owned_validation_failed_zero_readiness_candidate_retained_privately")
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
    return 0
