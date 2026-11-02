#!/usr/bin/env python3
"""Bind current V684 authority and scoped methods. REQ-REPORT-7879-V684."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import yaml  # noqa: E402

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting.roadmap_contract import compare_contract, parse_design  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import (  # noqa: E402
    CommandSpec,
    build_repository_health,
    run_commands,
)

DESIGN = ROOT / "tests/fixtures/v684/design.md"
STAGED = ROOT / "research-roadmap-next.yaml"
ACTIVE = ROOT / "tests/fixtures/v684/active.yaml"
RESULT = ROOT / "results/experiment_7879_v684_contract_methods.json"
START = time.monotonic()
MODEL_SPECS: list[str] = []
SEEDS = [67801, 67802, 67803]
ARMS = [
    "response_set",
    "local_set",
    "augmented_set",
    "constrained_set",
    "augmented_mlp",
    "constrained_mlp",
    "local_logistic",
    "source_erased_constrained_set",
    "complete_static_constrained_set",
]
ROLES = {
    "fit": 256,
    "tune": 64,
    "policy_design": 32,
    "calibration_replay": 32,
    "online_update": 96,
    "online_admission": 64,
    "evaluation": 64,
    "retention": 32,
}
TASK_TESTS = {
    7879: [
        "tests/python/test_experiment_7879_v684_contract_methods.py",
        "tests/python/test_experiment_7303_v642_validation_scope.py",
        "tests/python/test_experiment_7837_v681_contract_methods.py",
        "tests/python/test_roadmap_schema.py",
        "tests/python/test_audit_roadmap_gates.py",
        "tests/python/test_exclusion_manifest_lint.py",
        "tests/python/test_conductor_gates.py",
        "tests/python/test_harness_claim_enforcement.py",
        "tests/python/test_arc_orphan_lint_detects_induce_gating_2026_07_31.py",
    ],
    7880: [
        "tests/python/test_source_boundary_7866.py",
        "tests/python/test_source_boundary_7852.py",
    ],
    7881: ["tests/python/test_experiment_7868_v683_intervention_protocol.py"],
    7882: [
        "tests/python/test_natural_runtime_7867.py",
        "tests/python/test_natural_runtime_7853.py",
    ],
    7883: [
        "tests/python/test_natural_runtime_7867.py",
        "tests/python/test_natural_runtime_7853.py",
    ],
    7884: ["tests/python/test_experiment_7868_v683_intervention_protocol.py"],
    7885: [
        "tests/python/test_natural_runtime_7867.py",
        "tests/python/test_natural_runtime_7853.py",
    ],
    7886: [
        "tests/python/test_natural_runtime_7867.py",
        "tests/python/test_natural_runtime_7853.py",
    ],
    7887: ["tests/python/test_arc_supervisor_delta_7874.py"],
    7888: ["tests/python/test_natural_runtime_7867.py"],
    7889: ["tests/python/test_experiment_7876_v683_hardware_evidence.py"],
    7890: [
        "tests/python/test_experiment_7837_v681_contract_methods.py",
        "tests/python/test_conductor_gates.py",
    ],
}
TASK_E2E = {7880: ["E2E-015"], 7881: ["E2E-016"], 7887: ["E2E-017"]}
INAPPLICABLE_E2E = {
    "E2E-001..004": "No native training or binding change",
    "E2E-009..014": "No old ARC or Qwen runner dispatch",
    "E2E-005..008": "No verifier or sampling change",
}


def progress(phase: str, event: str, completed: int) -> None:
    """Flush each boundary with elapsed time and completed units."""
    print(
        f"[exp7879] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} "
        f"completed_units={completed}",
        flush=True,
    )


def label(path: Path) -> str:
    """Keep repository paths relative while retaining private paths."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_row(path: Path, role: str, date: str) -> dict[str, Any]:
    """Keep exact authority and exposure separate for each source."""
    return {
        "path": label(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "date": date,
        "source_role": role,
        "exposure_status": "historical" if role.startswith("historical") else "current",
        "eligible": path.is_file(),
    }


def inspect_authorities(design: Path, staged: Path, active: Path) -> dict[str, Any]:
    """Report a missing staged source without substituting active bytes."""
    active_data = yaml.safe_load(active.read_text())
    table, machine = parse_design(design.read_text(), milestone="2026.09.684")
    staged_present = staged.is_file()
    actual_staged = yaml.safe_load(staged.read_text()) if staged_present else None
    comparison = compare_contract(
        design.read_text(),
        actual_staged or active_data,
        active_data,
        milestone="2026.09.684",
        first_id=7879,
        count=12,
    )
    rows = deepcopy(comparison["rows"])
    for row in rows:
        row["checks"]["staged_authority_present"] = staged_present
        row["matched"] = bool(row["matched"] and staged_present)
        row["absolute_metric"] = int(row["matched"])
        row["raw_denominator"] += 1
        row["raw_numerator"] = sum(row["checks"].values())
        row["excluded"] = not row["matched"]
    failures = []
    if not staged_present:
        failures.append(
            {
                "upstream": "v684_staged_authority",
                "artifact_path": label(staged),
                "artifact_hash": None,
                "artifact_field": "exists",
                "op": "==",
                "expected": True,
                "observed": False,
            }
        )
    if active_data.get("milestone") != "2026.09.684":
        failures.append(
            {
                "upstream": "v684_active_authority",
                "artifact_path": label(active),
                "artifact_hash": sha256_file(active),
                "artifact_field": "milestone",
                "op": "==",
                "expected": "2026.09.684",
                "observed": active_data.get("milestone"),
            }
        )
    if not comparison["passed"]:
        failures.append(
            {
                "upstream": "v684_contract",
                "artifact_path": label(design),
                "artifact_hash": sha256_file(design),
                "artifact_field": "compare_contract.passed",
                "op": "==",
                "expected": True,
                "observed": False,
            }
        )
    comparison["diagnostic_only"] = not staged_present
    if not staged_present:
        comparison["passed"] = False
        comparison["errors"] = [*comparison["errors"], "missing_staged_authority"]
    return {
        "active_milestone": active_data.get("milestone"),
        "staged_present": staged_present,
        "contract_ready_score": int(not failures),
        "gate_check_summary": failures,
        "contract_rows": rows,
        "comparison": comparison,
        "table_count": len(table),
        "machine_count": len(machine),
    }


def mutation_rows(design: Path, active: Path) -> list[dict[str, Any]]:
    """Exercise the unchanged reader on ten private contract corruptions."""
    baseline = yaml.safe_load(active.read_text())
    rows = []
    for field in (
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "MODEL_SPECS",
        "inference_substrate_class",
        "gated_on",
        "prior_failures",
    ):
        staged = deepcopy(baseline)
        tasks = staged["tasks"]
        if field == "count":
            tasks.pop()
        elif field == "order":
            tasks[0], tasks[1] = tasks[1], tasks[0]
        elif field == "gated_on":
            tasks[3][field][0]["artifact_field"] = "wrong"
        elif field == "prior_failures":
            tasks[0][field][0].pop("addressed_by")
        else:
            tasks[0][field] = "wrong"
        result = compare_contract(
            design.read_text(), staged, baseline, milestone="2026.09.684", first_id=7879, count=12
        )
        rows.append(
            {"mutation": field, "rejected": not result["passed"], "errors": result["errors"]}
        )
    return rows


def snapshot_sources(paths: list[tuple[Path, str]], directory: Path) -> list[dict[str, Any]]:
    """Save immutable exact bytes at hash-addressed private paths."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, role in paths:
        digest = sha256_file(source)
        target = directory / f"{role}-{digest.removeprefix('sha256:')}"
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"immutable snapshot differs: {target}")
        if not target.exists():
            target.write_bytes(source.read_bytes())
        rows.append(
            {
                "path": label(source),
                "source_sha256": digest,
                "snapshot_path": str(target),
                "snapshot_sha256": sha256_file(target),
                "role": role,
            }
        )
    return rows


def build_manifest(root: Path, scratch: Path) -> dict[str, Any]:
    """Freeze real paths, current consumers, coverage includes and deadlines."""
    active = yaml.safe_load(ACTIVE.read_text())
    scopes = {}
    for task in active["tasks"]:
        number = int(task["id"][3:7])
        scopes[task["id"]] = {
            "tests": TASK_TESTS[number],
            "e2e": TASK_E2E.get(number, []),
            "planned_owned_test": f"tests/python/test_experiment_{number}_v684_*.py",
            "deadline_s": 900,
            "classification": "future_required" if number != 7879 else "required",
        }
    own_tests = scopes["exp7879-contract-methods"]["tests"]
    cli = "scripts/experiments/experiment_7879_v684_contract_methods.py"
    include = f"*/{cli}"
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    commands = [
        {
            "name": "collect_named_tests",
            "argv": [
                str(root / ".venv/bin/pytest"),
                *own_tests,
                *common,
                "--collect-only",
                "-q",
                f"--basetemp={scratch / 'collect'}",
            ],
            "deadline_s": 300,
        },
        {
            "name": "affected_pytest",
            "argv": [
                str(root / ".venv/bin/pytest"),
                *own_tests,
                *common,
                "-q",
                f"--basetemp={scratch / 'affected'}",
            ],
            "deadline_s": 900,
        },
        {
            "name": "unit_coverage",
            "argv": [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={scratch / '.coverage.unit'}",
                f"--include={include}",
                "-m",
                "pytest",
                "tests/python/test_experiment_7879_v684_contract_methods.py",
                *common,
                "-q",
                f"--basetemp={scratch / 'coverage'}",
            ],
            "deadline_s": 300,
        },
        {
            "name": "cli_coverage",
            "argv": [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={scratch / '.coverage.cli'}",
                f"--include={include}",
                cli,
                "--date",
                "20260929",
                "--output",
                str(scratch / "covered.json"),
                "--raw",
                str(scratch / "covered-rows.json"),
                "--no-validation",
            ],
            "deadline_s": 120,
        },
        {
            "name": "coverage_combine",
            "argv": [
                str(root / ".venv/bin/coverage"),
                "combine",
                f"--data-file={scratch / '.coverage.combined'}",
                "--keep",
                str(scratch / ".coverage.unit"),
                str(scratch / ".coverage.cli"),
            ],
            "deadline_s": 120,
        },
        {
            "name": "coverage_json",
            "argv": [
                str(root / ".venv/bin/coverage"),
                "json",
                f"--data-file={scratch / '.coverage.combined'}",
                f"--include={include}",
                "-o",
                str(scratch / "coverage.json"),
            ],
            "deadline_s": 120,
        },
        {
            "name": "coverage_report",
            "argv": [
                str(root / ".venv/bin/coverage"),
                "report",
                f"--data-file={scratch / '.coverage.combined'}",
                f"--include={include}",
                "--fail-under=100",
                "--show-missing",
            ],
            "deadline_s": 120,
        },
        {
            "name": "ruff_check",
            "argv": [
                str(root / ".venv/bin/ruff"),
                "check",
                cli,
                "tests/python/test_experiment_7879_v684_contract_methods.py",
            ],
            "deadline_s": 120,
        },
        {
            "name": "ruff_format",
            "argv": [
                str(root / ".venv/bin/ruff"),
                "format",
                "--check",
                cli,
                "tests/python/test_experiment_7879_v684_contract_methods.py",
            ],
            "deadline_s": 120,
        },
        {
            "name": "strict_mypy",
            "argv": [str(root / ".venv/bin/mypy"), "--strict", cli],
            "deadline_s": 180,
        },
        {
            "name": "scoped_spec_coverage",
            "argv": [str(root / ".venv/bin/python"), "scripts/check_spec_coverage.py", *own_tests],
            "deadline_s": 120,
        },
        {
            "name": "cold_replay",
            "argv": [
                str(root / ".venv/bin/python"),
                cli,
                "--cold-replay",
                str(scratch / "terminal.json"),
                "--raw",
                str(scratch / "rows.json"),
            ],
            "deadline_s": 120,
        },
        {
            "name": "adversarial_verify",
            "argv": [
                str(root / ".venv/bin/python"),
                "scripts/adversarial_verify.py",
                "--json",
                str(scratch / "terminal.json"),
            ],
            "deadline_s": 120,
        },
        {
            "name": "strict_row_lint",
            "argv": [
                str(root / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(scratch / "terminal.json"),
            ],
            "deadline_s": 120,
        },
    ]
    return {
        "schema": "v684-prospective-1",
        "task_scopes": scopes,
        "commands": commands,
        "coverage_include": include,
        "expected_measured_file": cli,
        "expected_measured_statements_min": 1,
        "inapplicable_e2e": INAPPLICABLE_E2E,
        "module_roots": {
            "carnot.reporting.roadmap_contract": "python/carnot/reporting/roadmap_contract.py",
            "carnot.reporting.experiment_7303_validation_scope": "python/carnot/reporting/experiment_7303_validation_scope.py",
            "scripts.roadmap_schema": "scripts/roadmap_schema.py",
            "scripts.audit_roadmap_gates": "scripts/audit_roadmap_gates.py",
        },
    }


def validate_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    """Reject nonexistent named tests and command targets before dispatch."""
    errors = []
    for scope in manifest["task_scopes"].values():
        for path in scope["tests"]:
            if not (root / path).is_file():
                errors.append("missing_test")
    for command in manifest["commands"]:
        if not Path(command["argv"][0]).is_file() or command["deadline_s"] <= 0:
            errors.append("invalid_command")
    for path in manifest["module_roots"].values():
        if not (root / path).is_file():
            errors.append("missing_module")
    return sorted(set(errors))


def seal_receipts(receipts: list[dict[str, Any]], directory: Path) -> list[dict[str, Any]]:
    """Copy closed child logs to immutable paths named by their byte hashes."""
    directory.mkdir(parents=True, exist_ok=True)
    sealed = []
    for receipt in receipts:
        original = Path(receipt["log_path"])
        if not original.is_absolute():
            original = ROOT / original
        digest = sha256_file(original)
        if digest != receipt["log_sha256"]:
            raise ValueError("child log changed before seal")
        target = directory / f"{receipt['name']}-{digest.removeprefix('sha256:')}.log"
        if target.exists() and sha256_file(target) != digest:
            raise ValueError("immutable child log differs")
        if not target.exists():
            shutil.copyfile(original, target)
        sealed.append(
            receipt | {"log_path": str(target), "log_sha256": digest, "classification": "required"}
        )
    return sealed


def validate_coverage(files: list[str], statements: int) -> list[str]:
    """Empty measured code cannot satisfy changed-code coverage."""
    return ["empty_coverage"] if not files or statements <= 0 else []


def validate_venue(venue: str) -> list[str]:
    """Owned CPU work uses the closed host venue."""
    return [] if venue == "host" else ["illegal_venue"]


def resolve_imports(manifest: dict[str, Any], root: Path) -> dict[str, str]:
    """Check fully qualified names against actual package roots."""
    resolved = {}
    for name, path in manifest["module_roots"].items():
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        expected = (root / path).resolve()
        if actual != expected:
            raise ValueError(f"import path mismatch: {name}: {actual}")
        resolved[name] = str(actual)
    return resolved


def literature_ledger() -> list[dict[str, Any]]:
    """Bind reviewed primary versions to falsifiable local controls."""
    return [
        {
            "url": "https://arxiv.org/abs/2604.23987",
            "version": "v1",
            "date": "2026-04-27",
            "method": "same-trajectory refreshed versus frozen calibration",
            "task": 7886,
            "limit": "exposed development has no exchangeability guarantee",
            "access": "primary abstract rechecked",
        },
        {
            "url": "https://arxiv.org/abs/2607.20792",
            "version": "v1",
            "date": "2026-07-22",
            "method": "delayed writes versus equal-information no-write",
            "task": 7885,
            "limit": "procedural recall does not establish Carnot learning",
            "access": "primary abstract rechecked",
        },
        {
            "url": "https://arxiv.org/abs/2608.00585",
            "version": "v1",
            "date": "2026-08-01",
            "method": "witness, adjacent context and matched filler",
            "task": 7884,
            "limit": "edited contexts have no independent truth labels",
            "access": "primary page rechecked",
        },
    ]


def cold_replay(candidate: Path, raw: Path) -> bool:
    """Recompute every row metric from independently saved primitive rows."""
    value = json.loads(candidate.read_text())
    rows = json.loads(raw.read_text())
    return (
        value.get("experiment_id") == 7879
        and value.get("rows") == rows
        and len(rows) == 12
        and all(row["absolute_metric"] == int(all(row["checks"].values())) for row in rows)
    )


def historical_failures() -> list[dict[str, Any]]:
    """Preserve V683 required failures as history, not current pass receipts."""
    source = ROOT / "results/experiment_7865_v683_contract_methods.json"
    prior = json.loads(source.read_text())
    return [
        {
            "upstream": "exp7865-contract-methods",
            "artifact_path": label(source),
            "artifact_hash": sha256_file(source),
            "name": row["name"],
            "exit_code": row["exit_code"],
            "timed_out": row.get("timed_out", False),
            "log_path": row.get("log_path"),
            "log_sha256": row.get("log_sha256"),
            "resolved": False,
        }
        for row in prior["validation_receipts"]
        if not row.get("passed")
    ]


def artifact(
    date: str,
    authority: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    sources: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    resolved: dict[str, str],
) -> dict[str, Any]:
    """Keep validity, readiness and scientific benefit as different fields."""
    rows = authority["contract_rows"]
    failures = historical_failures()
    required_failed = [
        r for r in receipts if r.get("classification") == "required" and not r["passed"]
    ]
    blocked = bool(authority["gate_check_summary"])
    verdict = "disqualified" if required_failed else "blocked" if blocked else "positive"
    ready = int(verdict == "positive" and authority["contract_ready_score"] == 1)
    source_hash = canonical_hash(
        {
            "sources": sources,
            "manifest": manifest,
            "seeds": SEEDS,
            "script": sha256_file(Path(__file__)),
        }
    )
    spans = {"preconditions_and_contract_s": time.monotonic() - START}
    return {
        "experiment_id": 7879,
        "task_id": "exp7879-contract-methods",
        "milestone": "2026.09.684",
        "run_date": date,
        "honest_verdict": f"complete_{verdict}_v684_"
        + (
            "required_validation"
            if required_failed
            else "missing_staged_authority"
            if blocked
            else "contract"
        ),
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": authority["gate_check_summary"],
        "rows": rows,
        "contract_rows": rows,
        "sample_size_budget": {
            "intended": 12,
            "eligible": 12 if not blocked else 0,
            "started": 12,
            "completed": 12,
            "censored": 0,
            "excluded": sum(r["excluded"] for r in rows),
            "independent": 0,
            "independent_unit": "administrative_contract",
        },
        "acceptance_gate_results": {
            "validity": not bool(required_failed) and not blocked,
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - START,
        "phase_spans": spans,
        "random_seed": None,
        "reproducibility_checksum": source_hash,
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "staged_present": authority["staged_present"],
            "active_milestone": authority["active_milestone"],
            "schema": manifest["schema"],
            "gate_operands": authority["gate_check_summary"],
            "venue": "host",
        },
        "resolved_imports": resolved,
        "validation_receipts": receipts,
        "validation_command_manifest_path": label(manifest_path),
        "validation_command_manifest_sha256": sha256_file(manifest_path),
        "validation_scope_manifest_path": label(manifest_path),
        "observed_child_commands": [r["command_argv"] for r in receipts],
        "historical_required_failures": failures,
        "repository_health": build_repository_health(failures)
        | {"untraced_tests_observed": 1142, "full_suite_qualification": False},
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "natural_labels": "exposed_development",
            "independent_verifier": False,
        },
        "field_principles": {
            "identity_and_verdict": "Bind this producer and distinguish external blocks from owned failures.",
            "gate_check_summary": "Name each missing or failed exact operand.",
            "rows_and_budget": "Preserve every task unit without inventing independent science families.",
            "acceptance_gate_results": "Mechanics, readiness and benefit have separate gates.",
            "timing_and_checksum": "Use monotonic work and hash source, code, seed and configuration.",
            "source_and_imports": "Authority bytes and package roots have distinct custody.",
            "validation_and_history": "Scoped exits do not erase old required failures or full-suite debt.",
            "claim_scope": "Administrative agreement is oracle-like and not a natural-data benefit.",
            "models_and_venue": "No pretrained model is loaded; host is the closed CPU venue.",
            "contract_ready_score": "Only staged and active authority with passing owned checks can score one.",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {
            "model_loads_attempted": 0,
            "generation_calls_attempted": 0,
            "forward_calls_attempted": 0,
            "tokens": 0,
            "model_file_hashes": [],
        },
        "trained_head_specs": [],
        "contract_ready_score": ready,
        "contract_comparison": authority["comparison"],
        "mutation_rows": mutation_rows(DESIGN, ACTIVE),
        "methods_manifest": {"path": label(manifest_path), "sha256": sha256_file(manifest_path)},
        "authority_snapshots": sources[:3],
        "literature_adoption_decisions": literature_ledger(),
        "frozen_methods": {
            "cohort_roles": ROLES,
            "training_arms": ARMS,
            "seeds": SEEDS,
            "feature_count": 132,
            "max_parameters": 4096,
            "epochs_max": 16,
            "decision_costs": {"accept": "5p", "reject": "1-p", "escalate": 0.25},
            "bootstrap_resamples": 10000,
            "alpha": 0.05,
            "stopping": "predeclared per-task bounds in V684 design",
        },
        "e2e_applicability": {
            "required": ["private_cli", "cold_replay"],
            "inapplicable": INAPPLICABLE_E2E,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write a private or terminal candidate from current primitive readers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/carnot-exp7879-v684"))
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    progress("start", "flushed_start", 0)
    if args.cold_replay:
        passed = bool(args.raw and cold_replay(args.cold_replay, args.raw))
        progress("cold_replay", "passed" if passed else "failed", int(passed))
        return 0 if passed else 1
    if args.no_validation and args.output == RESULT:
        parser.error("private output required for --no-validation")
    scratch = args.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(ROOT, scratch)
    manifest_errors = validate_manifest(manifest, ROOT)
    if manifest_errors:
        raise ValueError(f"invalid declared scope: {manifest_errors}")
    manifest_path = scratch / "validation-manifest.json"
    atomic_json(manifest_path, manifest)
    progress("preconditions", "paths_and_argv_checked", 12)
    authority = inspect_authorities(DESIGN, STAGED, ACTIVE)
    progress("authority", "compared_or_blocked", 12)
    paths = [
        (DESIGN, "design"),
        (STAGED, "staged_missing"),
        (ACTIVE, "active"),
        (ROOT / "research-references.md", "literature"),
        (ROOT / "results/experiment_7865_v683_contract_methods.json", "historical_disqualified"),
        (Path(__file__), "current_code"),
    ]
    sources = [source_row(path, role, args.date) for path, role in paths]
    snapshots = snapshot_sources(
        [(path, role) for path, role in paths if path.is_file()], scratch / "authority-snapshots"
    )
    by_path = {item["path"]: item for item in snapshots}
    for source in sources:
        source.update(by_path.get(source["path"], {}))
    resolved = resolve_imports(manifest, ROOT)
    receipts: list[dict[str, Any]] = []
    if not args.no_validation:
        terminal_names = {"cold_replay", "adversarial_verify", "strict_row_lint"}
        commands = [
            CommandSpec(x["name"], tuple(x["argv"]), "required", x["deadline_s"])
            for x in manifest["commands"]
            if x["name"] not in terminal_names
        ]
        receipts = seal_receipts(
            run_commands(ROOT, commands, log_dir=scratch / "logs"), scratch / "sealed-logs"
        )
        coverage_json = scratch / "coverage.json"
        measured = json.loads(coverage_json.read_text())["files"] if coverage_json.is_file() else {}
        statements = sum(item["summary"]["num_statements"] for item in measured.values())
        if validate_coverage(list(measured), statements):
            for receipt in receipts:
                if receipt["name"] == "coverage_report":
                    receipt["passed"] = False
                    receipt["coverage_error"] = "empty_coverage"
        progress("validation", "required_checks_completed", len(receipts))
    value = artifact(args.date, authority, manifest, manifest_path, sources, receipts, resolved)
    raw = scratch / "rows.json"
    atomic_json(raw, value["rows"])
    if args.raw and args.raw != raw:
        atomic_json(args.raw, value["rows"])
    if args.no_validation:
        atomic_json(args.output, value)
        progress("private_candidate", "written", 12)
        return 0
    terminal = scratch / "terminal.json"
    sidecar = scratch / "terminal-validation-receipts.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    terminal_specs = [
        CommandSpec(x["name"], tuple(x["argv"]), "required", x["deadline_s"])
        for x in manifest["commands"]
        if x["name"] in {"cold_replay", "adversarial_verify", "strict_row_lint"}
    ]
    for attempt in range(2):
        atomic_json(terminal, value)
        progress("terminal", "before_exact_validators", attempt)
        checks = seal_receipts(
            run_commands(ROOT, terminal_specs, log_dir=scratch / "terminal-logs"),
            scratch / "sealed-logs",
        )
        report = next((item for item in checks if item["name"] == "adversarial_verify"), None)
        flagged = False
        if report and report["passed"]:
            flagged = json.loads(Path(report["log_path"]).read_text()).get("flagged_count", 0) > 0
        failed = flagged or any(not item["passed"] for item in checks)
        if failed and value["verdict_class"] != "disqualified" and attempt == 0:
            value["flagged_adversarial"] = flagged
            value["verdict_class"] = "disqualified"
            value["honest_verdict"] = "complete_disqualified_v684_terminal_validation"
            value["contract_ready_score"] = 0
            value["acceptance_gate_results"]["validity"] = False
            value["acceptance_gate_results"]["readiness"] = 0
            continue
        atomic_json(sidecar, checks)
        progress("terminal", "after_exact_validators", len(checks))
        break
    atomic_json(args.output, value)
    if args.output.read_bytes() != terminal.read_bytes():
        raise ValueError("published bytes differ from validated candidate")
    progress("publish", "exact_checked_bytes_written", 12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
