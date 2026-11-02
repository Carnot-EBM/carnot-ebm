#!/usr/bin/env python3
"""Reuse qualified live supervisor evidence without running games. REQ-REPORT-7911."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.arc_supervisor_v685_delta import reduce_receipts, replay_delta
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.experiments.experiment_7899_v685_arc_supervisor_delta import _run

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "results/experiment_7899_v685_arc_supervisor_delta.json"
EXPECTED_PRIOR = "sha256:0a1119998829458eabf061e213d70cd9fc3b352f90cb287cc6dca0bda4822d0c"
CLI = "scripts/experiments/experiment_7911_v686_arc_supervisor_delta.py"
INCLUDE = "*/" + CLI
TESTS = (
    "tests/python/test_arc_supervisor_delta_7911.py",
    "tests/python/test_arc_supervisor_delta_7899.py",
    "tests/python/test_arc_supervisor_delta_7887.py",
    "tests/python/test_arc_supervisor_delta_7874.py",
)
DURABLE = ROOT / "results/raw/experiment_7911_v686_arc_supervisor_delta"
OUTPUT = ROOT / "results/experiment_7911_v686_arc_supervisor_delta.json"


def progress(started: float, phase: str, units: int) -> None:
    """Expose completed work so the operator can distinguish progress from a stall."""
    print(
        f"[exp7911] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def precheck() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate the successful producer and its exact frozen dependencies."""
    checks: list[dict[str, Any]] = []

    def check(path: Path, field: str, expected: Any, observed: Any, role: str) -> None:
        checks.append(
            {
                "upstream_id": "exp7899",
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "artifact_field": field,
                "op": "==",
                "expected": expected,
                "observed": observed,
                "role": role,
                "exposure_status": "exposed_development",
            }
        )

    check(
        PRIOR,
        "sha256",
        EXPECTED_PRIOR,
        sha256_file(PRIOR) if PRIOR.is_file() else "missing",
        "qualified_cutoff",
    )
    prior: dict[str, Any] = {}
    if checks[0]["expected"] == checks[0]["observed"]:
        prior = json.loads(PRIOR.read_text())
        for field, expected in (
            ("verdict_class", "null"),
            ("flagged_adversarial", False),
            ("arc_delta_ready_score", 1),
        ):
            check(PRIOR, field, expected, prior.get(field), "qualified_cutoff")
        for label, expected in prior["source_artifact_hashes"].items():
            path = Path(label)
            check(
                path,
                "sha256",
                expected,
                sha256_file(path) if path.is_file() else "missing",
                "frozen_upstream_dependency",
            )
    return prior, checks, [row for row in checks if row["expected"] != row["observed"]]


def audit(prior: dict[str, Any]) -> dict[str, Any]:
    """Use content identity and the qualified reducer; file dates cannot create science."""
    baseline = dict(prior.get("cutoff_receipt_hashes", {}))
    for row in prior.get("outcome_rows", []):
        if row.get("event_id") and row.get("content_sha256"):
            baseline[row["event_id"]] = row["content_sha256"]
        if row.get("source_sha256"):
            baseline["raw:" + row["source_sha256"]] = row["source_sha256"]
    inventory = json.loads(Path(prior["receipt_inventory_path"]).read_text())
    producers = [Path(label) for label in inventory["producer_paths"]]
    for path in sorted((ROOT / "results").glob("experiment_*arc*.json")):
        number = path.name.split("_")[1]
        if number.isdecimal() and int(number) > 7899 and path != OUTPUT:
            producers.append(path)
    delta = reduce_receipts(ROOT, sorted(set(producers)), baseline, prior["registry_precheck"])
    delta.update(
        cutoff_receipt_hashes=baseline,
        null_fast_path=not producers,
        producer_paths=[str(path) for path in producers],
    )
    return delta


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze explicit private routes before work; negative exits have a stated reason."""
    py, cov, pytest = (str(ROOT / ".venv/bin" / name) for name in ("python", "coverage", "pytest"))
    producer, output = private / "producer.json", private / "delta.json"
    atomic_json(producer, {"source_artifact_hashes": {}, "verdict_class": "null"})
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    base = [CLI, "--date", "20260930", "--reduce-ledger", str(private), "--producer", str(producer)]
    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        argv: list[str],
        expected_exit: int = 0,
        reason: str | None = None,
        scope: str = "required",
        deadline: int = 240,
    ) -> None:
        rows.append(
            {
                "name": name,
                "argv": argv,
                "deadline_s": deadline,
                "classification": scope,
                "expected_exit": expected_exit,
                "expected_text": reason,
            }
        )

    add("affected_pytest", [pytest, *TESTS, *common, f"--basetemp={private / 'unit-temp'}"])
    add(
        "unit_coverage",
        [
            cov,
            "run",
            f"--data-file={private / 'coverage.unit'}",
            f"--include={INCLUDE}",
            "-m",
            "pytest",
            *TESTS,
            *common,
            f"--basetemp={private / 'coverage-temp'}",
        ],
    )
    for name, route, exit_code, reason in (
        ("success", [*base, "--output", str(output)], 0, None),
        ("failure", base, 2, "--output"),
        ("replay", [CLI, "--date", "20260930", "--cold-replay", str(output)], 0, None),
    ):
        add(
            "cli_" + name + "_coverage",
            [
                cov,
                "run",
                f"--data-file={private / ('coverage.' + name)}",
                f"--include={INCLUDE}",
                *route,
            ],
            exit_code,
            reason,
        )
    add(
        "coverage_combine",
        [
            cov,
            "combine",
            f"--data-file={private / 'coverage.combined'}",
            *(
                str(private / ("coverage." + name))
                for name in ("unit", "success", "failure", "replay")
            ),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            f"--data-file={private / 'coverage.combined'}",
            f"--include={INCLUDE}",
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", CLI, TESTS[0]])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", CLI, TESTS[0]])
    add("mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", CLI])
    add("scoped_spec", [py, "scripts/check_spec_coverage.py", *TESTS])
    add("e2e_017", [pytest, TESTS[3], *common, f"--basetemp={private / 'e2e017-temp'}"])
    for mode in ("fixture-e2e", "cold-replay"):
        add(
            "e2e_016_" + mode,
            [
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260930",
                "--" + mode,
                str(private / "e2e016.json"),
            ],
        )
    add(
        "full_python_suite",
        [pytest, "tests/python", "-q", f"--basetemp={private / 'full-temp'}"],
        scope="repository_health",
        deadline=600,
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Publish a bounded audit whose evidence belongs to this producer alone."""
    started = time.monotonic()
    progress(started, "start", 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930")
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--producer", type=Path, action="append")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        errors = replay_delta(json.loads(args.cold_replay.read_text()))
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_ledger:
        if not args.producer or args.output is None:
            parser.error("--reduce-ledger requires --producer and --output")
        progress(started, "before_private_reduce", 0)
        delta = reduce_receipts(args.reduce_ledger, args.producer, {}, {})
        atomic_json(args.output, delta)
        progress(started, "after_private_reduce", delta["new_outcome_count"])
        return 0
    progress(started, "before_precheck", 0)
    prior, checks, failures = precheck()
    progress(started, "after_precheck", len(checks))
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7911-", dir="/tmp"))
    rows = commands(private)
    DURABLE.mkdir(parents=True, exist_ok=True)
    manifest = DURABLE / "validation_command_manifest.json"
    imports = {
        name: str(Path(module.__file__).resolve())
        for name, module in tuple(sys.modules.items())
        if getattr(module, "__file__", None)
        and Path(module.__file__).resolve().is_relative_to(ROOT)
    }
    closure = sorted(
        set(imports.values())
        | {
            str(ROOT / name)
            for name in (
                *TESTS,
                CLI,
                "pyproject.toml",
                "scripts/check_spec_coverage.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "python/carnot/reporting/arc_supervisor_receipt_delta.py",
                "python/carnot/agentic/arc_solver_kit.py",
                "ops/exclusion_manifest.yaml",
            )
        }
    )
    sources = {label: sha256_file(Path(label)) for label in closure if Path(label).is_file()}
    sources.update({row["path"]: row["sha256"] for row in checks if row["sha256"]})
    atomic_json(
        manifest,
        {
            "commands": rows,
            "affected_closure": closure,
            "dependency_hashes": sources,
            "measured_files": [CLI],
            "coverage_include": INCLUDE,
            "applicable_e2e": ["E2E-016", "E2E-017"],
            "inapplicable_e2e": [f"E2E-{i:03d}" for i in range(1, 16)] + ["E2E-018"],
            "exposure_role": "runtime_mechanics_only",
        },
    )
    sources[str(manifest)] = sha256_file(manifest)
    progress(started, "scope_frozen_before_compute", len(sources))
    delta = audit(prior) if not failures else reduce_receipts(ROOT, [], {}, {})
    inventory = DURABLE / "receipt_inventory.json"
    atomic_json(inventory, delta)
    sources[str(inventory)] = sha256_file(inventory)
    progress(
        started,
        "null_fast_path" if delta.get("null_fast_path") else "delta_reduced",
        delta["new_outcome_count"],
    )
    receipts = [_run(row, DURABLE, started) for row in rows]
    owned_failures = [
        row for row in receipts if row["classification"] == "required" and not row["passed"]
    ]
    failures.extend(
        {
            "upstream_id": "exp7911",
            "path": row.get("log_path"),
            "sha256": row.get("log_sha256"),
            "artifact_field": "exit_code",
            "op": "==",
            "expected": row["expected_exit"],
            "observed": row["exit_code"],
            "check": row["name"],
        }
        for row in owned_failures
    )
    artifact = dict(prior)
    artifact.update(
        experiment_id=7911,
        task_id="exp7911-arc-supervisor-delta",
        milestone="2026.09.686",
        run_date=args.date,
        honest_verdict="complete_disqualified_required_validation"
        if owned_failures
        else "complete_blocked_source_precondition"
        if failures
        else "complete_null_no_new_supervisor_outcomes"
        if not delta["new_outcome_count"]
        else "complete_positive_observational_supervisor_delta",
        verdict_class="disqualified"
        if owned_failures
        else "blocked"
        if failures
        else "null"
        if not delta["new_outcome_count"]
        else "positive",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=delta["rows"],
        outcome_rows=delta["rows"],
        sample_size_budget={
            **delta["sample_size_budget"],
            "unit": "supervisor_firing",
            "independent": 0,
            "independence_scope": "exposed_development",
        },
        acceptance_gate_results={
            "validity": not bool(failures),
            "readiness": int(not failures),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        source_artifact_hashes=sources,
        preconditions_checked=checks,
        resolved_imports=imports,
        validation_receipts=receipts,
        observed_child_commands=[row["command_argv"] for row in receipts],
        validation_command_manifest_path=str(manifest),
        historical_required_failures=prior.get("historical_required_failures", []),
        repository_health={
            "status": "historical_diagnostic_open",
            "repository_wide_check_repeated": True,
            "current_repository_checks": [
                row for row in receipts if row["classification"] == "repository_health"
            ],
            "affects_required_checks": False,
        },
        verifier_is_oracle=False,
        claim_scope="exposed_development; observational content-hash delta; no causal improvement or new game solve",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        planned_inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model="none",
        model_invocation_counts={"loads": 0, "calls": 0, "tokens": 0},
        trained_head_specs=[],
        arc_delta_ready_score=int(not failures),
        new_live_outcome_count=delta["new_outcome_count"],
        new_outcome_count=delta["new_outcome_count"],
        cutoff_receipt_hashes=delta.get("cutoff_receipt_hashes", {}),
        per_game_results=delta["per_game_results"],
        recommendation_rows=delta["recommendation_rows"],
        firings=delta["firings"],
        new_level_solves=0,
        registry_precheck=delta["registry_precheck"],
        solve_provenance=[row.get("solve_provenance") for row in delta["rows"]],
        receipt_inventory_path=str(inventory),
        null_fast_path=delta.get("null_fast_path", False),
        random_seed=0,
        reproducibility_checksum=canonical_hash(
            {"sources": sources, "cutoff": delta.get("cutoff_receipt_hashes"), "seed": 0}
        ),
        validation_errors=[row["name"] for row in owned_failures],
        terminal_validation_sidecar_path=str(DURABLE / "terminal_reports.json"),
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact["phase_spans"] = [
        {
            "phase": "authenticate_freeze_reduce_validate",
            "start_s": 0,
            "end_s": artifact["duration_s"],
            "duration_s": artifact["duration_s"],
        }
    ]
    artifact["field_principles"] = {
        key: "Keep current producer evidence distinct from historical, fixture and independent scientific claims."
        for key in artifact
    }
    artifact["field_principles"].update(
        cutoff_receipt_hashes="Content identity prevents repeated receipts from becoming fresh evidence.",
        new_level_solves="Reading receipts does not solve a game.",
        arc_delta_ready_score="An authenticated zero delta can satisfy the standing ARC floor.",
        acceptance_gate_results="Validity and readiness do not establish benefit, probability quality, retention or efficiency.",
        sample_size_budget="Count firing dispositions separately; development observations are not independent validation.",
    )
    candidate = DURABLE / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    for attempt in range(3):
        reports = []
        for name, script, option in (
            ("terminal_adversarial", "scripts/adversarial_verify.py", "--json"),
            ("terminal_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ):
            reports.append(
                _run(
                    {
                        "name": name,
                        "argv": [str(ROOT / ".venv/bin/python"), script, option, str(candidate)],
                        "deadline_s": 90,
                        "classification": "required",
                        "expected_exit": 0,
                        "expected_text": None,
                    },
                    DURABLE,
                    started,
                )
            )
        atomic_json(
            DURABLE / f"terminal_reports_{attempt}.json",
            {"candidate_sha256": sha256_file(candidate), "reports": reports},
        )
        actual_flag = not reports[0]["passed"]
        if all(row["passed"] for row in reports) and artifact["flagged_adversarial"] == actual_flag:
            break
        artifact.update(
            flagged_adversarial=actual_flag,
            honest_verdict="complete_disqualified_terminal_verification",
            verdict_class="disqualified",
            arc_delta_ready_score=0,
        )
        artifact["acceptance_gate_results"].update(validity=False, readiness=0)
        artifact["validation_errors"] = sorted(
            set(
                artifact["validation_errors"]
                + [row["name"] for row in reports if not row["passed"]]
            )
        )
        atomic_json(candidate, artifact)
    atomic_json(
        DURABLE / "terminal_reports.json",
        {"candidate_sha256": sha256_file(candidate), "reports": reports},
    )
    atomic_json(args.output or OUTPUT, json.loads(candidate.read_text()))
    progress(started, "deliverable_written", delta["new_outcome_count"])
    return int(not artifact["arc_delta_ready_score"])


if __name__ == "__main__":
    sys.exit(main())
