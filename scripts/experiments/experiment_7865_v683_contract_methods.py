#!/usr/bin/env python3
"""Bind fourteen V683 task contracts; REQ-REPORT-7865-V683."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import yaml  # noqa: E402

from carnot.reporting.current_work_receipt import (  # noqa: E402
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import (  # noqa: E402
    cold_replay,
    compare_contract,
    snapshot_authorities,
    verify_snapshots,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = ROOT / "research-roadmap.yaml"
STAGED = ROOT / "research-roadmap-next.yaml"
RESULT = ROOT / "results/experiment_7865_v683_contract_methods.json"
SNAPSHOTS = ROOT / "docs/research-notes/v683-authority-snapshots"
START = time.monotonic()
START_NS = time.monotonic_ns()
MILESTONE = "2026.09.683"
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


def progress(phase: str, event: str, units: int) -> None:
    """Expose live work so a quiet child is not mistaken for a stalled run."""
    print(
        f"[exp7865] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def label(path: Path) -> str:
    """Keep repository paths stable while naming private paths exactly."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def source_rows(paths: list[Path], date: str) -> list[dict[str, Any]]:
    """Record exact bytes and authority; absence must have a null hash."""
    rows = []
    for path in paths:
        historical = "v682" in str(path) or "785" in path.name
        rows.append(
            {
                "path": label(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "date": date,
                "source_role": "historical_development" if historical else "current_contract",
                "exposure_status": "historical" if historical else "current",
                "eligible": path.is_file(),
            }
        )
    return rows


def gate_failure(path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Name the failing operand so absence and a wrong value stay distinct."""
    return {
        "upstream": "preconditions",
        "path": label(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def methods() -> dict[str, Any]:
    """Freeze the earlier exposed cohort and costs without asserting benefit."""
    return {
        "cohort": "Exp7810 exposed development families; no fresh holdout",
        "roles": {
            "fit": 256,
            "tune": 64,
            "policy": 64,
            "online_update": 96,
            "online_admission": 64,
            "evaluation": 64,
            "retention": 32,
        },
        "training_arms": ARMS,
        "seeds": SEEDS,
        "epochs_max": 16,
        "learning_rate": 0.01,
        "mlp_width": 16,
        "fitted_scalars_max": 4096,
        "decision_costs": "accept/reject/escalate cost on calibrated unsupported risk",
        "uncertainty": "10000 paired source-family bootstrap draws and paired randomization; Holm across six tests",
        "primary_contrasts": [
            "constrained_set vs augmented_set",
            "constrained_set vs constrained_mlp",
            "constrained_set vs local_logistic",
        ],
        "source_sufficiency_estimand": "Qwen full source vs witness only, immediate neighbors, and matched filler; separate from correctness",
        "source_sufficiency_budget": {
            "calls_max": 192,
            "new_tokens_max": 24576,
            "per_call_s": 120,
            "launch_stop_s": 2400,
            "service_cap_s": 3000,
        },
        "future_e2e_scope": ["private CLI", "cold replay", "source hash mutation", "row mutation"],
        "inapplicable_e2e": {
            "E2E-001_to_004": "no training or native binding change",
            "E2E-007": "no bank update",
            "E2E-009_to_013": "no ARC runtime change",
        },
    }


def mutation_rows(
    design: str, staged: dict[str, Any], active: dict[str, Any]
) -> list[dict[str, Any]]:
    """Exercise the existing contract reader on each current contract dimension."""
    rows = []
    for name in ("count", "order", "id", "title", "phase", "path", "model", "class", "gate"):
        changed = deepcopy(staged)
        tasks = changed["tasks"]
        if name == "count":
            tasks.pop()
        elif name == "order":
            tasks[0], tasks[1] = tasks[1], tasks[0]
        elif name == "gate":
            next(task for task in tasks if task["gated_on"])["gated_on"][0]["artifact_field"] = (
                "wrong_field"
            )
        else:
            key = {
                "id": "id",
                "title": "title",
                "phase": "phase",
                "path": "deliverable",
                "model": "MODEL_SPECS",
                "class": "inference_substrate_class",
            }[name]
            tasks[0][key] = ["wrong/model"] if name == "model" else "wrong"
        rejected = not compare_contract(
            design, changed, active, milestone=MILESTONE, first_id=7865, count=14
        )["passed"]
        rows.append({"mutation": name, "rejected": rejected, "status": "completed"})
        progress("mutations", name, len(rows))
    return rows


def build_artifact(
    date: str,
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    snapshots: list[dict[str, str]],
    method_path: Path,
    validation_path: Path,
    blocked: bool,
) -> dict[str, Any]:
    """Keep validity, administrative readiness and scientific value separate."""
    current = build_current_work_receipt(
        run_id=f"exp7865-{date}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="blocked_no_run"
        if blocked
        else "verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"current_model_invocations": 0},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=START_NS,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    completed = sum(row.get("status") == "completed" for row in rows)
    return {
        "experiment_id": 7865,
        "task_id": "exp7865-contract-methods",
        "milestone": MILESTONE,
        "run_date": date,
        "honest_verdict": "complete_blocked_v683_authority"
        if blocked
        else "partial_v683_validation_pending",
        "verdict_class": "blocked" if blocked else "partial",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "contract_rows": rows,
        "mutation_rows": [],
        "sample_size_budget": {
            "intended": 14,
            "eligible": 0 if blocked else 14,
            "started": 0 if blocked else 14,
            "completed": completed,
            "censored": 0,
            "excluded": sum(bool(r.get("excluded")) for r in rows),
            "independent": 0,
            "independent_unit": "none_administrative",
        },
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - START,
        "phase_spans": {"preconditions_s": time.monotonic() - START},
        "random_seed": None,
        "reproducibility_checksum": canonical_hash(
            {
                "sources": sources,
                "method_hash": sha256_file(method_path),
                "validation_hash": sha256_file(validation_path),
                "seeds": SEEDS,
            }
        ),
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "source_paths": sources,
            "gate_operands": failures,
            "resource_ownership": "host_cpu_only_no_gpu_lease",
            "staged_origin": "restored_copy_of_already_active_v683_bytes",
            "active_roadmap_modified": False,
        },
        "validation_receipts": [],
        "validation_command_manifest_path": label(validation_path),
        "validation_command_manifest_sha256": sha256_file(validation_path),
        "observed_child_commands": [],
        "repository_health": None,
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "natural_human_labels": "exposed_development_only",
            "fresh_generalization_eligible": False,
        },
        "field_principles": {
            "experiment_id/task_id/milestone/run_date": "Current work must have its own identity.",
            "honest_verdict/verdict_class": "Terminal state must match the actual work.",
            "flagged_adversarial": "Flagged evidence cannot open a gate.",
            "gate_check_summary": "Missing and wrong-valued inputs have distinct causes.",
            "rows/sample_size_budget": "Primitive units make aggregates independently recomputable.",
            "acceptance_gate_results.validity": "Checks must pass before use.",
            "acceptance_gate_results.readiness": "Administrative readiness is separate from benefit.",
            "acceptance_gate_results.probability_quality": "Calibration needs independent labels.",
            "acceptance_gate_results.decision_benefit": "Benefit needs a matched control.",
            "acceptance_gate_results.retention": "Learning needs delayed retained decisions.",
            "acceptance_gate_results.efficiency": "Service cost needs full boundary timing.",
            "duration_s/phase_spans": "Monotonic spans describe work that ran.",
            "random_seed/reproducibility_checksum": "Source, code, method and seeds are bound.",
            "source_artifact_hashes/preconditions_checked": "Historical and current sources have different authority.",
            "validation_receipts/validation_command_manifest_path/observed_child_commands/repository_health": "Required failures remain visible.",
            "verifier_is_oracle/claim_scope": "Fixture agreement is circular evidence only.",
            "inference_substrate/inference_substrate_class/execution_venue": "Actual compute differs from host location.",
            "MODEL_SPECS/model_specs/target_model/model_invocation_counts": "Cited models are not current calls.",
            "contract_ready_score": "Agreement is an administrative predicate.",
            "contract_rows/mutation_rows/authority_snapshots/methods_manifest": "The plan must be falsifiable and frozen.",
        },
        "inference_substrate": "blocked_no_run"
        if blocked
        else "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "trained_head_specs": [],
        "model_invocation_counts": {
            "model_loads_attempted": 0,
            "generation_calls_attempted": 0,
            "forward_calls_attempted": 0,
            "tokens": 0,
            "model_file_hashes": [],
        },
        "current_work_receipt": current,
        "contract_ready_score": 0,
        "authority_snapshots": snapshots,
        "methods_manifest": {"path": label(method_path), "sha256": sha256_file(method_path)},
        "historical_required_failures": [
            "Exp7852 complete_disqualified_required_checks",
            "Exp7854 complete_disqualified_required_checks",
            "six V682 science producers were not emitted",
        ],
        "e2e_applicability": methods()["inapplicable_e2e"],
    }


def immutable_json(path: Path, value: dict[str, Any]) -> None:
    """Refuse a changed freeze instead of silently replacing an earlier plan."""
    if path.is_file():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable manifest differs: {path}")
    else:
        atomic_json(path, value)


def validation_commands(private: Path, raw: Path, staged: Path) -> list[CommandSpec]:
    """Name exact required argv before comparison or validation starts."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    test = "tests/python/test_experiment_7865_v683_contract_methods.py"
    script = "scripts/experiments/experiment_7865_v683_contract_methods.py"
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    coverage_file = raw.parent / ".coverage.v683"
    return [
        CommandSpec(
            "worktree_imports",
            (
                py,
                "-c",
                "import importlib,json; names=['carnot.reporting.roadmap_contract','carnot.reporting.current_work_receipt','carnot.reporting.experiment_7303_validation_scope']; print(json.dumps({'resolved_imports': {n:importlib.import_module(n).__file__ for n in names}}))",
            ),
            "required",
            120,
        ),
        CommandSpec(
            "schema",
            (
                py,
                "-c",
                "import yaml; from scripts.roadmap_schema import Roadmap; Roadmap.model_validate(yaml.safe_load(open('research-roadmap-next.yaml'))); print('schema_ok')",
            ),
            "required",
            120,
        ),
        CommandSpec(
            "prior_failures",
            (py, "scripts/validate_prior_failures.py", str(staged)),
            "required",
            120,
        ),
        CommandSpec(
            "exclusion", (py, "scripts/exclusion_manifest_lint.py", str(staged)), "required", 120
        ),
        CommandSpec("gates", (py, "scripts/audit_roadmap_gates.py", str(staged)), "required", 120),
        CommandSpec(
            "affected_pytest",
            (pytest, test, *common, f"--basetemp={raw.parent / 'pytest-affected'}", "-q"),
            "required",
            600,
        ),
        CommandSpec(
            "full_pytest",
            (pytest, "tests/python", *common, f"--basetemp={raw.parent / 'pytest-full'}", "-q"),
            "required",
            3600,
        ),
        CommandSpec(
            "changed_coverage",
            (
                coverage,
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{script}",
                "-m",
                "pytest",
                test,
                *common,
                f"--basetemp={raw.parent / 'pytest-coverage'}",
                "-q",
            ),
            "required",
            600,
        ),
        CommandSpec(
            "coverage_cli",
            (
                coverage,
                "run",
                "--append",
                f"--data-file={coverage_file}",
                f"--include=*/{script}",
                script,
                "--date",
                "20260929",
                "--output",
                str(raw.with_name(raw.stem + "-coverage-cli.json")),
                "--raw",
                str(raw.with_name(raw.stem + "-coverage-rows.json")),
                "--no-validation",
            ),
            "required",
            120,
        ),
        CommandSpec(
            "coverage_report",
            (
                coverage,
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{script}",
                "--fail-under=100",
                "--show-missing",
            ),
            "required",
            120,
        ),
        CommandSpec("ruff_check", (ruff, "check", script, test), "required", 120),
        CommandSpec("ruff_format", (ruff, "format", "--check", script, test), "required", 120),
        CommandSpec("strict_mypy", (mypy, "--strict", script), "required", 300),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py"), "required", 300),
        CommandSpec(
            "cold_replay",
            (py, script, "--cold-replay", str(private), "--raw", str(raw)),
            "required",
            120,
        ),
    ]


def terminal_commands(candidate: Path) -> list[CommandSpec]:
    """Verify terminal candidate bytes with both required artifact readers."""
    py = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "adversarial_verify",
            (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "required",
            120,
        ),
        CommandSpec(
            "verdict_row_consistency",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "required",
            120,
        ),
    ]


def report_flagged(receipts: list[dict[str, Any]]) -> bool:
    """Use the verifier's actual JSON rather than guessing from its exit."""
    report = next(row for row in receipts if row["name"] == "adversarial_verify")
    path = Path(report["log_path"])
    if not path.is_absolute():
        path = ROOT / path
    try:
        return bool(json.loads(path.read_text())["flagged_count"])
    except (OSError, ValueError, KeyError):
        return True


def run_and_seal(commands: list[CommandSpec], scratch: Path) -> list[dict[str, Any]]:
    """Keep actual exits and copy closed logs to immutable hash-named files."""
    receipts = run_commands(ROOT, commands, log_dir=scratch / "logs", heartbeat_s=30)
    sealed = ROOT / "docs/research-notes/v683-validation-logs"
    sealed.mkdir(parents=True, exist_ok=True)
    for receipt in receipts:
        source = Path(receipt["log_path"])
        if not source.is_absolute():
            source = ROOT / source
        target = sealed / f"{receipt['name']}-{receipt['log_sha256'].removeprefix('sha256:')}.log"
        if target.exists() and sha256_file(target) != receipt["log_sha256"]:
            raise ValueError("sealed log hash collision")
        if not target.exists():
            shutil.copyfile(source, target)
        receipt["log_path"] = label(target)
        receipt["classification"] = "required"
        if receipt["name"] == "worktree_imports":
            imports = receipt.get("resolved_imports", {})
            receipt["passed"] = (
                bool(imports)
                and all(
                    Path(path).resolve().is_relative_to(ROOT / "python")
                    for path in imports.values()
                )
                and receipt["passed"]
            )
    return receipts


def gate_operand_failures(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check gate fields against the actual upstream artifact contract text."""
    failures = []
    seen: dict[str, set[str]] = {}
    for task in tasks:
        prompt = str(task.get("prompt", ""))
        required = prompt.split("REQUIRED ARTIFACT FIELDS:", 1)
        fields: set[str] = set()
        if len(required) == 2:
            for line in required[1].splitlines():
                if line.startswith("- ") and ":" in line:
                    fields.update(field.strip() for field in line[2:].split(":", 1)[0].split(","))
        for gate in task.get("gated_on", []):
            upstream = gate.get("upstream", "")
            field = gate.get("artifact_field", "")
            if upstream not in seen or field not in seen[upstream]:
                failures.append(
                    gate_failure(ACTIVE, f"{upstream}.{field}", "declared earlier", "absent")
                )
        for prior in task.get("prior_failures", []):
            for field in ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"):
                if prior.get(field) in (None, ""):
                    failures.append(
                        gate_failure(
                            ACTIVE, f"{task['id']}.prior_failures.{field}", "present", "absent"
                        )
                    )
        seen[str(task["id"])] = fields
    return failures


def main(argv: list[str] | None = None) -> int:
    """Produce one terminal administrative receipt from frozen authorities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--staged", type=Path, default=STAGED)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    progress("start", "flushed_start", 0)
    scratch = Path(os.environ.get("CARNOT_7865_TMP", "/tmp/carnot-exp7865-owned"))
    scratch.mkdir(parents=True, exist_ok=True)
    if args.cold_replay is not None:
        if args.raw is None:
            parser.error("--cold-replay requires --raw")
        passed = cold_replay(args.cold_replay, args.raw, experiment_id=7865)
        progress("cold_replay", "passed" if passed else "failed", int(passed))
        return 0 if passed else 1
    if args.no_validation and args.output == RESULT:
        parser.error("--no-validation requires private --output")
    private = args.output != RESULT
    archive = args.output.parent if private else ROOT / "docs/research-notes"
    method_path = archive / "v683-methods-manifest.json"
    validation_path = archive / "v683-validation-manifest-v2.json"
    checkpoint_key = canonical_hash(
        {
            "design": sha256_file(DESIGN) if DESIGN.is_file() else None,
            "staged": sha256_file(args.staged) if args.staged.is_file() else None,
            "active": sha256_file(ACTIVE) if ACTIVE.is_file() else None,
            "driver": sha256_file(Path(__file__)),
        }
    ).removeprefix("sha256:")
    raw = args.raw or scratch / f"rows-{checkpoint_key}.json"
    candidate = scratch / f"pending-{checkpoint_key}.json"
    terminal_candidate = scratch / f"terminal-{checkpoint_key}.json"
    method = methods()
    immutable_json(method_path, method)
    commands = validation_commands(candidate, raw, args.staged)
    terminal_specs = terminal_commands(terminal_candidate)
    closure = [
        Path(__file__),
        ROOT / "tests/python/test_experiment_7865_v683_contract_methods.py",
        ROOT / "python/carnot/reporting/roadmap_contract.py",
        ROOT / "python/carnot/reporting/current_work_receipt.py",
        ROOT / "python/carnot/reporting/experiment_7303_validation_scope.py",
    ]
    manifest = {
        "schema": "v683_validation_v1",
        "commands": [
            {
                "name": c.name,
                "argv": list(c.argv),
                "classification": "required",
                "deadline_s": c.timeout_s,
            }
            for c in (*commands, *terminal_specs)
        ],
        "frozen_hashes": {label(path): sha256_file(path) for path in closure},
        "historical_required_failures": ["Exp7852 disqualified", "Exp7854 disqualified"],
    }
    immutable_json(validation_path, manifest)
    paths = [
        DESIGN,
        args.staged,
        ACTIVE,
        *closure,
        method_path,
        validation_path,
        ROOT / "openspec/change-proposals/research-roadmap-v682-preserved-20260929.md",
        ROOT / "ops/exclusion_manifest.yaml",
    ]
    sources = source_rows(paths, args.date)
    failures = [gate_failure(path, "exists", True, False) for path in paths if not path.is_file()]
    progress("preconditions", "paths_and_hashes_checked", len(paths))
    active = yaml.safe_load(ACTIVE.read_text()) if ACTIVE.is_file() else {}
    staged = yaml.safe_load(args.staged.read_text()) if args.staged.is_file() else {}
    if active.get("milestone") != MILESTONE:
        failures.append(gate_failure(ACTIVE, "milestone", MILESTONE, active.get("milestone")))
    if staged and staged.get("milestone") != MILESTONE:
        failures.append(gate_failure(args.staged, "milestone", MILESTONE, staged.get("milestone")))
    if active.get("tasks"):
        failures.extend(gate_operand_failures(active["tasks"]))
    blocked = bool(failures)
    snapshot_dir = (archive / "v683-authority-snapshots") if private else SNAPSHOTS
    snapshots = [] if blocked else snapshot_authorities((DESIGN, args.staged, ACTIVE), snapshot_dir)
    if snapshots and not verify_snapshots(snapshots):
        raise ValueError("authority snapshot cold verification failed")
    comparison = (
        None
        if blocked
        else compare_contract(
            DESIGN.read_text(), staged, active, milestone=MILESTONE, first_id=7865, count=14
        )
    )
    rows = (
        comparison["rows"]
        if comparison
        else [
            {
                "unit_id": task.get("id", f"exp{7865 + i}-missing"),
                "arm": "four_source_contract",
                "family": task.get("id", f"exp{7865 + i}"),
                "seed": None,
                "status": "unstarted",
                "checks": {},
                "absolute_metric": 0,
                "raw_numerator": 0,
                "raw_denominator": 0,
                "censored": False,
                "excluded": False,
                "effective_independent_groups": 0,
            }
            for i, task in enumerate(active.get("tasks", [])[:14])
        ]
    )
    progress("authority", "compared_or_blocked", sum(r["status"] == "completed" for r in rows))
    value = build_artifact(
        args.date, rows, sources, failures, snapshots, method_path, validation_path, blocked
    )
    value["contract_comparison"] = comparison
    value["mutation_rows"] = [] if blocked else mutation_rows(DESIGN.read_text(), staged, active)
    value["preconditions_checked"]["gate_operands"] = failures
    value["raw_rows_path"] = str(raw)
    if raw.is_file() and json.loads(raw.read_text()) != {"rows": rows}:
        raise ValueError("checkpoint content differs for same input hash")
    atomic_json(raw, {"rows": rows})
    value["raw_rows_sha256"] = sha256_file(raw)
    if blocked:
        value["duration_s"] = time.monotonic() - START
        atomic_json(args.output, value)
        progress("terminal", "blocked_written", 0)
        return 0
    if not comparison["passed"] or not all(row["rejected"] for row in value["mutation_rows"]):
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_v683_contract_mismatch"
        value["duration_s"] = time.monotonic() - START
        atomic_json(args.output, value)
        progress("terminal", "mismatch_written", 14)
        return 1
    if args.no_validation:
        value["duration_s"] = time.monotonic() - START
        atomic_json(args.output, value)
        progress("private_cli", "candidate_written", 14)
        return 0
    for name, digest in manifest["frozen_hashes"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    atomic_json(candidate, value)
    progress("validation", "start", 0)
    receipts = run_and_seal(commands, scratch)
    passed = all(r["passed"] for r in receipts) and len(receipts) == len(commands)
    passed = (
        passed and verify_snapshots(snapshots) and cold_replay(candidate, raw, experiment_id=7865)
    )
    value["verdict_class"] = "circular_positive" if passed else "disqualified"
    value["honest_verdict"] = (
        "complete_circular_positive_v683_contract_methods"
        if passed
        else "complete_disqualified_v683_required_validation"
    )
    atomic_json(terminal_candidate, value)
    terminal_receipts = run_and_seal(terminal_specs, scratch)
    flagged = report_flagged(terminal_receipts)
    receipts.extend(terminal_receipts)
    passed = passed and not flagged and all(row["passed"] for row in terminal_receipts)
    value["flagged_adversarial"] = flagged
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [
        {"name": r["name"], "argv": r["command_argv"], "classification": "required"}
        for r in receipts
    ]
    value["repository_health"] = {
        "historical_required_failures": value["historical_required_failures"],
        "status": "unresolved",
    }
    value["acceptance_gate_results"]["validity"] = passed
    value["acceptance_gate_results"]["readiness"] = int(passed)
    value["contract_ready_score"] = int(passed)
    value["verdict_class"] = "circular_positive" if passed else "disqualified"
    value["honest_verdict"] = (
        "complete_circular_positive_v683_contract_methods"
        if passed
        else "complete_disqualified_v683_required_validation"
    )
    value["duration_s"] = time.monotonic() - START
    value["phase_spans"]["validation_s"] = (
        value["duration_s"] - value["phase_spans"]["preconditions_s"]
    )
    atomic_json(terminal_candidate, value)
    exact_receipts = run_and_seal(terminal_specs, scratch)
    if report_flagged(exact_receipts) != flagged or any(
        row["passed"] != prior["passed"] for row, prior in zip(exact_receipts, terminal_receipts)
    ):
        raise ValueError("terminal candidate verification changed after receipt insertion")
    atomic_json(args.output, value)
    progress("terminal", "artifact_written", len(receipts))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
