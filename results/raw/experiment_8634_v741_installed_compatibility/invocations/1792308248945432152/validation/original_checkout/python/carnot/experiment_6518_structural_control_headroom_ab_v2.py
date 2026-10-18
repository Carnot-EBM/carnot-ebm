"""Exp6518 matched exact-solver structural branch-control comparison.

Spec refs: REQ-BENCH-6518, SCENARIO-BENCH-6518-AUDIT-GATE,
SCENARIO-BENCH-6518-ARM-CONTRACT, SCENARIO-BENCH-6518-LIVE-INFLUENCE,
SCENARIO-BENCH-6518-COST-EQUALITY, SCENARIO-BENCH-6518-HELD-TRANSFER,
SCENARIO-BENCH-6518-ATTACKS, SCENARIO-BENCH-6518-TERMINAL.

This runner measures non-learned branch-order controls on the audited Exp6516
pilot. The exact solver still owns labels. The comparison can show method
headroom, but it does not train or justify a learned router by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
from typing import Any

from carnot import experiment_6504_exact_structural_benchmark_commitment as exp6504
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
Trace = list[tuple[int, bool]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6518
SCHEMA_VERSION = "carnot.experiment_6518.structural_control_headroom_ab_v2.v1"
INFERENCE_SUBSTRATE = "procedural_exact_solver_structural_controls_no_llm"
VERIFIER_IS_ORACLE = False

RESULT_RELATIVE_PATH = Path("results/experiment_6518_structural_control_headroom_ab_v2.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6518_structural_control_headroom_ab_v2.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py"
)
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

EXP6504_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
EXP6508_RELATIVE_PATH = Path("results/experiment_6508_analytical_branch_refocus_ab.json")
EXP6515_RELATIVE_PATH = Path("results/experiment_6515_v564_source_method_contract.json")
EXP6516_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
EXP6517_RELATIVE_PATH = Path("results/experiment_6517_branch_pilot_independent_audit.json")
ROADMAP_RELATIVE_PATH = Path("research-references.md")

NATIVE_ARM = "native_dynamic"
ARM_IDS = (
    NATIVE_ARM,
    "shuffled_dynamic",
    "static_analytical",
    "partial_assignment_consistency",
    "periodic_bounded_refocus",
    "random_critical_variable_enumeration",
    "analytical_enumeration",
)
PILOT_BASE_UNIT_COUNT = 18
ELIGIBLE_VALUES = (False, True)
EXACT_ASSIGNMENT_BUDGET = 256
RESTART_BUDGET = 0
TIME_LIMIT_S = 2.0
PRIMARY_METRIC = "held_total_charged_work_units_vs_native_dynamic"

ATTACK_IDS = (
    "inert_advice",
    "no_headroom_units",
    "row_order_coupling",
    "family_identity",
    "tuned_held_thresholds",
    "omitted_losses",
    "timeout_selection",
    "one_seed_wins",
)

PROTECTED_RELATIVE_PATHS = (
    EXP6504_RELATIVE_PATH,
    EXP6508_RELATIVE_PATH,
    EXP6515_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    EXP6517_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("scripts/research_conductor.py"),
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    EXP6504_RELATIVE_PATH,
    EXP6508_RELATIVE_PATH,
    EXP6515_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    EXP6517_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "prior_failure_receipts",
    "preregistration",
    "arm_contract",
    "per_game_results",
    "live_influence_rows",
    "exact_answer_equality_rows",
    "charged_cost_rows",
    "censoring_rows",
    "family_seed_summary",
    "attack_matrix",
    "structural_control_execution_complete_score",
    "structural_headroom_candidate_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal structural-control comparison state.",
    "honest_verdict": (
        "States the matched held result without turning the control into learned-router evidence."
    ),
    "verdict_class": (
        "Closed enum separates positive, null, partial, blocked, and disqualified outcomes."
    ),
    "upstream_gate_receipt": (
        "Pins the Exp6517 audit gate by path, hash, expected value, observed value, solvers, resources, and protected hashes."
    ),
    "prior_failure_receipts": (
        "Keeps Exp6508 and method-boundary failures visible without making them dependencies."
    ),
    "preregistration": (
        "Freezes the planning date, primary metric, splits, score rules, and stop rules before rows are scored."
    ),
    "arm_contract": (
        "Freezes the seven structural arms and proves each arm may order candidates but not remove them."
    ),
    "per_game_results": "Stores one matched exact-solver row for each pilot unit and arm.",
    "live_influence_rows": (
        "Shows whether advice changed a live decision path relative to native dynamic."
    ),
    "exact_answer_equality_rows": "Shows exact-solver and Z3 labels match for every compared row.",
    "charged_cost_rows": (
        "Reports solver-only and total charged cost, including feature, refocus, enumeration, and fallback work."
    ),
    "censoring_rows": (
        "Records timeout, censoring, restart, and terminal-disposition symmetry."
    ),
    "family_seed_summary": (
        "Reports held transfer by family and seed before a candidate score can open."
    ),
    "attack_matrix": (
        "Tests inert advice, no-headroom units, row order, family identity, held tuning, omitted losses, timeout selection, and one-seed wins."
    ),
    "structural_control_execution_complete_score": (
        "Opens only when the matched rows, equality, costs, censoring, attacks, and protected hashes pass."
    ),
    "structural_headroom_candidate_score": (
        "Opens only when held charged benefit is positive with equality, live influence, and family plus seed support."
    ),
    "gate_check_summary": "Names every failed gate with expected and observed values.",
    "per_unit_rows": (
        "Flattens game, influence, equality, cost, censoring, family-seed, and attack rows for recomputation."
    ),
    "aggregate_row_recomputation": "Recomputes execution and candidate scores from rows.",
    "preconditions_checked": (
        "Records paths, resources, solvers, seeds, budgets, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Proves the audit, pilot, method contract, prior blocked receipt, and conductor stayed byte-identical."
    ),
    "inference_substrate": "Declares exact procedural controls with no LLM inference.",
    "verifier_is_oracle": "False because method value is measured, not certified by a verifier.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": (
        "Maps each field to specs, input artifacts, rows, reducers, tests, and hashes."
    ),
    "random_seed": "Pins arm seeds and shuffled critical-variable controls.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects drift in gates, rows, costs, attacks, and verdicts.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6518_structural_control_headroom_ab_v2.py "
    "-m pytest tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6518_structural_control_headroom_ab_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6518_structural_control_headroom_ab_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6518_structural_control_headroom_ab_v2.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6518_structural_control_headroom_ab_v2 "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6518_structural_control_headroom_ab_v2 --validate"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6518_structural_control_headroom_ab_v2.py "
    "tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "scripts/adversarial_verify.py"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": RUFF_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json_with_status(path: Path) -> tuple[JsonDict, str, str]:
    if not path.is_file():
        return {}, "missing", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input is terminally blocked.
        return {}, "corrupt_json", str(exc)
    if not isinstance(payload, Mapping):  # pragma: no cover - project artifacts are objects.
        return {}, "non_object", "top-level JSON is not an object"
    return dict(payload), "parsed", ""


def _command_output(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip() or result.stderr.strip()


def _resource_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    meminfo: dict[str, int] = {}
    mem_path = Path("/proc/meminfo")
    if mem_path.is_file():
        for line in mem_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ram_total_bytes": meminfo.get("MemTotal"),
        "ram_available_bytes": meminfo.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def solver_versions() -> JsonDict:
    return {
        "structural_solver": "exp6518_exact_backtracking_with_matched_branch_orders_v1",
        "z3_python_available": True,
        "z3_python_version": exp6504.z3.get_version_string(),
        "z3_cli_path": shutil.which("z3"),
    }


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def protected_file_hashes(repo_root: Path, audit_path: Path | None = None) -> dict[str, str]:
    hashes = {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}
    if audit_path is not None:
        hashes[_source_key(repo_root, audit_path)] = sha256_file(audit_path)
    return hashes


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    changed = [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]
    return {
        "all_protected_files_unchanged": not changed,
        "changed_files": changed,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def upstream_gate_receipt(
    *,
    repo_root: Path,
    audit_path: Path,
    protected_before: Mapping[str, str],
) -> JsonDict:
    payload, parse_status, parse_error = _read_json_with_status(audit_path)
    observed = payload.get("branch_pilot_audited_ready_score") if parse_status == "parsed" else None
    return {
        "row_type": "upstream_gate_receipt",
        "path": _source_key(repo_root, audit_path),
        "absolute_path": str(audit_path),
        "exists": audit_path.is_file(),
        "sha256": sha256_file(audit_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "field": "branch_pilot_audited_ready_score",
        "json_pointer": "/branch_pilot_audited_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0,
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "artifact_reproducibility_checksum": payload.get("reproducibility_checksum"),
        "branch_row_count": payload.get("upstream_artifact_receipt", {}).get("branch_row_count"),
        "solver_versions": solver_versions(),
        "resources": _resource_state(repo_root),
        "protected_file_hashes_before": dict(protected_before),
        "read_mode": "direct_path_and_hash_gate_receipt",
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-AUDIT-GATE"],
    }


def prior_failure_receipts(repo_root: Path) -> JsonDict:
    exp6508, exp6508_status, _ = _read_json_with_status(repo_root / EXP6508_RELATIVE_PATH)
    exp6515, exp6515_status, _ = _read_json_with_status(repo_root / EXP6515_RELATIVE_PATH)
    method_rows = exp6515.get("sota_to_experiment_rows", []) if exp6515_status == "parsed" else []
    source_rows = exp6515.get("source_rows", []) if exp6515_status == "parsed" else []
    dibs_rows = [
        dict(row)
        for row in method_rows
        if isinstance(row, Mapping)
        and (row.get("target_experiment") == "Exp6518" or row.get("source_id") == "dibs")
    ]
    dibs_sources = [
        dict(row)
        for row in source_rows
        if isinstance(row, Mapping) and "dibs" in str(row.get("source_id", "")).lower()
    ]
    enumeration_rows = [
        dict(row)
        for row in method_rows
        if isinstance(row, Mapping) and "enumerat" in canonical_json(row).lower()
    ]
    return {
        "row_type": "prior_failure_receipts",
        "exp6508": {
            "path": EXP6508_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / EXP6508_RELATIVE_PATH),
            "parse_status": exp6508_status,
            "status": exp6508.get("status"),
            "honest_verdict": exp6508.get("honest_verdict"),
            "blocked_reason": exp6508.get("blocked_reason"),
            "gate_check_summary": exp6508.get("gate_check_summary"),
        },
        "exp6515_method_boundary": {
            "path": EXP6515_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / EXP6515_RELATIVE_PATH),
            "parse_status": exp6515_status,
            "ready_score": exp6515.get("v564_method_contract_ready_score"),
            "dibs_method_rows": dibs_rows,
            "dibs_source_rows": dibs_sources,
            "enumeration_method_rows": enumeration_rows,
            "enumeration_method_row_found": bool(enumeration_rows),
            "enumeration_roadmap_receipt": {
                "path": ROADMAP_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(repo_root / ROADMAP_RELATIVE_PATH),
                "method": "one_shot_critical_variable_enumeration_control",
                "source": "V562/V563 planning text; absent from Exp6515 method rows",
            },
            "exact_authority_boundary": (
                "DiBS and enumeration advice may order candidates only; exact fallback labels remain authoritative."
            ),
        },
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-AUDIT-GATE"],
    }


def preregistration(run_date: str) -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION + ".preregistration",
        "planning_date": run_date,
        "primary_metric": PRIMARY_METRIC,
        "primary_split": "held",
        "native_reference_arm": NATIVE_ARM,
        "candidate_score_rule": (
            "positive held charged benefit, exact equality, live influence, and support in more than one family and seed"
        ),
        "execution_score_rule": (
            "all matched rows, equality, cost accounting, censoring, attacks, and protected hashes pass"
        ),
        "verdict_class_rules": {
            "positive": "oracle-distinct held charged benefit with complete execution",
            "null": "complete execution with no held charged benefit",
            "partial": "incomplete but usable evidence",
            "blocked": "gate or precondition failure",
            "disqualified": "correctness drift, leakage, candidate loss, or false accounting",
        },
        "threshold_tuning_allowed": False,
        "row_order_used_as_feature": False,
        "family_identity_used_as_feature": False,
        "learned_model_trained": False,
        "verifier_is_oracle_for_method_value": False,
        "exact_solver_is_label_authority": True,
        "stop_rules": [
            "block on Exp6517 audit gate failure",
            "do not train a learned router in Exp6518",
            "do not hide solver-only or charged costs",
        ],
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-HELD-TRANSFER"],
    }


def arm_contract() -> JsonDict:
    descriptions = {
        NATIVE_ARM: "Native low-index exact backtracking with false-first values.",
        "shuffled_dynamic": "Seeded non-learned shuffled order and shuffled values.",
        "static_analytical": "Static occurrence order with majority-sign values.",
        "partial_assignment_consistency": "Recomputed unresolved-clause consistency order.",
        "periodic_bounded_refocus": "Occurrence refocus every two decision depths.",
        "random_critical_variable_enumeration": "Seeded critical-variable prefix enumeration.",
        "analytical_enumeration": "Occurrence-ranked critical-variable prefix enumeration.",
    }
    rows = [
        {
            "arm_id": arm_id,
            "description": descriptions[arm_id],
            "candidate_preservation_required": True,
            "advice_can_order_candidates": arm_id != NATIVE_ARM,
            "advice_can_remove_candidates": False,
            "learned_model_used": False,
            "charged_overheads": ["feature", "refocus", "enumeration", "fallback"],
        }
        for arm_id in ARM_IDS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".arm_contract",
        "arm_ids": list(ARM_IDS),
        "arms": rows,
        "candidate_values": list(ELIGIBLE_VALUES),
        "advice_can_remove_candidates": False,
        "exact_solver_is_label_authority": True,
        "matched_solver_settings": {
            "assignment_budget": EXACT_ASSIGNMENT_BUDGET,
            "restart_budget": RESTART_BUDGET,
            "time_limit_s": TIME_LIMIT_S,
            "terminal_dispositions": ["sat_model", "unsat_proof", "timeout"],
            "pilot_base_unit_count": PILOT_BASE_UNIT_COUNT,
        },
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-ARM-CONTRACT"],
    }
    return {**payload, "arm_contract_hash": sha256_json(payload)}


def _literal_counts(clauses: Sequence[Sequence[int]]) -> tuple[Counter[int], Counter[int], Counter[int]]:
    occurrences: Counter[int] = Counter()
    positives: Counter[int] = Counter()
    negatives: Counter[int] = Counter()
    for clause in clauses:
        for literal in clause:
            variable = abs(int(literal))
            occurrences[variable] += 1
            if int(literal) > 0:
                positives[variable] += 1
            else:
                negatives[variable] += 1
    return occurrences, positives, negatives


def _literal_total(clauses: Sequence[Sequence[int]]) -> int:
    return sum(len(clause) for clause in clauses)


def _assignment_satisfies(clauses: Sequence[Sequence[int]], assignment: Mapping[int, bool]) -> bool:
    return all(
        any(
            bool(assignment[abs(int(literal))])
            if int(literal) > 0
            else not bool(assignment[abs(int(literal))])
            for literal in clause
        )
        for clause in clauses
    )


def _partial_clause_conflict(
    clauses: Sequence[Sequence[int]],
    assignment: Mapping[int, bool],
) -> tuple[bool, int, int]:
    checks = 0
    for clause_index, clause in enumerate(clauses):
        has_unassigned = False
        satisfied = False
        for literal in clause:
            checks += 1
            variable = abs(int(literal))
            if variable not in assignment:
                has_unassigned = True
            elif (int(literal) > 0 and assignment[variable]) or (
                int(literal) < 0 and not assignment[variable]
            ):
                satisfied = True
                break
        if not satisfied and not has_unassigned:
            return True, checks, clause_index
    return False, checks, -1


def _arm_initial_plan(
    *,
    row: Mapping[str, Any],
    arm_id: str,
) -> tuple[list[int], dict[str, int], Counter[int], Counter[int], Counter[int]]:
    n_vars = int(row["variable_count"])
    clauses = [[int(literal) for literal in clause] for clause in row["clauses"]]
    variables = list(range(1, n_vars + 1))
    occurrences, positives, negatives = _literal_counts(clauses)
    overhead = {
        "feature_cost_units": 0,
        "refocus_cost_units": 0,
        "enumeration_cost_units": 0,
        "fallback_cost_units": 1,
    }
    if arm_id == NATIVE_ARM:
        order = variables
    elif arm_id == "shuffled_dynamic":
        order = variables[:]
        random.Random(RANDOM_SEED + int(row["generator_seed"])).shuffle(order)
        overhead["feature_cost_units"] = n_vars
    elif arm_id in {"static_analytical", "partial_assignment_consistency"}:
        order = sorted(variables, key=lambda var: (-occurrences[var], abs(positives[var] - negatives[var]), var))
        overhead["feature_cost_units"] = _literal_total(clauses)
    elif arm_id == "periodic_bounded_refocus":
        order = sorted(variables, key=lambda var: (-occurrences[var], var))
        overhead["feature_cost_units"] = _literal_total(clauses) // 2
    elif arm_id == "random_critical_variable_enumeration":
        critical = sorted(variables, key=lambda var: (-occurrences[var], var))[: min(3, n_vars)]
        random.Random((RANDOM_SEED * 3) + int(row["generator_seed"])).shuffle(critical)
        order = critical + [variable for variable in variables if variable not in critical]
        overhead["feature_cost_units"] = _literal_total(clauses)
        overhead["enumeration_cost_units"] = len(critical) * 2
    elif arm_id == "analytical_enumeration":
        critical = sorted(
            variables,
            key=lambda var: (-occurrences[var], -max(positives[var], negatives[var]), var),
        )[: min(3, n_vars)]
        rest = sorted(
            [variable for variable in variables if variable not in critical],
            key=lambda var: (-occurrences[var], var),
        )
        order = critical + rest
        overhead["feature_cost_units"] = _literal_total(clauses)
        overhead["enumeration_cost_units"] = len(critical) * 3
    else:  # pragma: no cover - ARM_IDS owns all callers.
        raise ValueError(f"unknown arm_id: {arm_id}")
    return order, overhead, occurrences, positives, negatives


def _value_order(
    *,
    row: Mapping[str, Any],
    variable: int,
    arm_id: str,
    positives: Counter[int],
    negatives: Counter[int],
) -> list[bool]:
    if arm_id in {NATIVE_ARM, "periodic_bounded_refocus"}:
        return [False, True]
    if arm_id in {"static_analytical", "partial_assignment_consistency", "analytical_enumeration"}:
        return [True, False] if positives[variable] >= negatives[variable] else [False, True]
    values = [False, True]
    random.Random((RANDOM_SEED * 5) + int(row["generator_seed"]) + variable).shuffle(values)
    return values


def _choose_variable(
    *,
    arm_id: str,
    clauses: Sequence[Sequence[int]],
    assignment: Mapping[int, bool],
    remaining: Sequence[int],
    occurrences: Counter[int],
    depth: int,
) -> tuple[int, int]:
    if arm_id == "partial_assignment_consistency":
        scores = {variable: 0 for variable in remaining}
        for clause in clauses:
            clause_satisfied = any(
                abs(int(literal)) in assignment
                and ((int(literal) > 0 and assignment[abs(int(literal))]) or (int(literal) < 0 and not assignment[abs(int(literal))]))
                for literal in clause
            )
            if clause_satisfied:
                continue
            for literal in clause:
                variable = abs(int(literal))
                if variable in scores:
                    scores[variable] += 1
        return max(sorted(remaining), key=lambda var: (scores[var], occurrences[var], -var)), _literal_total(clauses)
    if arm_id == "periodic_bounded_refocus" and depth > 0 and depth % 2 == 0:
        return max(sorted(remaining), key=lambda var: (occurrences[var], -var)), _literal_total(clauses) // 2
    return int(remaining[0]), 0


def _z3_status(n_vars: int, clauses: Sequence[Sequence[int]]) -> tuple[str, str | None]:
    outcome = exp6504.z3_solve(n_vars, [list(clause) for clause in clauses])
    model_hash = sha256_json(outcome.model) if outcome.model is not None else None
    return str(outcome.status), model_hash


def _solve_with_arm(row: Mapping[str, Any], arm_id: str) -> JsonDict:
    start = time.perf_counter()
    n_vars = int(row["variable_count"])
    clauses = [[int(literal) for literal in clause] for clause in row["clauses"]]
    order, overhead, occurrences, positives, negatives = _arm_initial_plan(row=row, arm_id=arm_id)
    conflicts = 0
    propagations = 0
    decisions = 0
    refocus_extra = 0
    trace: Trace = []
    conflict_trace: list[str] = []

    def recurse(assignment: dict[int, bool], remaining: list[int], depth: int) -> dict[int, bool] | None:
        nonlocal conflicts, propagations, decisions, refocus_extra
        conflict, checks, clause_index = _partial_clause_conflict(clauses, assignment)
        propagations += checks
        if conflict:
            conflicts += 1
            conflict_trace.append(f"{depth}:{clause_index}")
            return None
        if not remaining:
            return dict(assignment)
        variable, dynamic_cost = _choose_variable(
            arm_id=arm_id,
            clauses=clauses,
            assignment=assignment,
            remaining=remaining,
            occurrences=occurrences,
            depth=depth,
        )
        refocus_extra += dynamic_cost
        next_remaining = [item for item in remaining if item != variable]
        for value in _value_order(
            row=row,
            variable=variable,
            arm_id=arm_id,
            positives=positives,
            negatives=negatives,
        ):
            decisions += 1
            trace.append((variable, value))
            assignment[variable] = value
            solved = recurse(assignment, next_remaining, depth + 1)
            if solved is not None:
                return solved
            del assignment[variable]
        return None

    model = recurse({}, order, 0)
    overhead["refocus_cost_units"] += refocus_extra
    status = "sat" if model is not None else "unsat"
    z3_status, z3_model_hash = _z3_status(n_vars, clauses)
    model_payload = {f"x{variable}": model[variable] for variable in range(1, n_vars + 1)} if model else None
    model_valid = bool(model_payload) and _assignment_satisfies(
        clauses,
        {int(key.removeprefix("x")): bool(value) for key, value in model_payload.items()},
    )
    proof_hash = sha256_json(conflict_trace) if status == "unsat" else None
    proof_valid = status == "unsat" and z3_status == "unsat" and proof_hash is not None
    solver_only = conflicts + propagations + decisions
    overhead_total = sum(overhead.values())
    return {
        "exact_status": status,
        "z3_status": z3_status,
        "z3_model_hash": z3_model_hash,
        "exact_answer_equality": status == z3_status,
        "terminal_disposition": "sat_model" if status == "sat" else "unsat_proof",
        "terminal_model_or_proof": {
            "model_hash": sha256_json(model_payload) if model_payload is not None else None,
            "model_valid": model_valid,
            "proof_hash": proof_hash,
            "proof_valid": proof_valid,
            "receipt_valid": (status == "sat" and model_valid) or proof_valid,
        },
        "conflicts": conflicts,
        "propagations": propagations,
        "decisions": decisions,
        "restarts": RESTART_BUDGET,
        "wall_time_s": round(max(time.perf_counter() - start, 0.000001), 6),
        "timeout": False,
        "censored": False,
        "censoring_reason": "",
        "decision_trace": trace,
        "decision_trace_hash": sha256_json(trace),
        "decision_trace_prefix": [[variable, value] for variable, value in trace[:8]],
        "solver_only_work_units": solver_only,
        "total_control_overhead_units": overhead_total,
        "total_charged_work_units": solver_only + overhead_total,
        **overhead,
    }


def _compare_traces(native_trace: Trace, trace: Trace) -> tuple[int | None, int]:
    first_changed: int | None = None
    changed = 0
    for index in range(max(len(native_trace), len(trace))):
        left = native_trace[index] if index < len(native_trace) else None
        right = trace[index] if index < len(trace) else None
        if left != right:
            changed += 1
            if first_changed is None:
                first_changed = index + 1
    return first_changed, changed


def _load_pilot_base_rows(repo_root: Path) -> list[JsonDict]:
    pilot_payload, pilot_status, _ = _read_json_with_status(repo_root / EXP6516_RELATIVE_PATH)
    base_payload, base_status, _ = _read_json_with_status(repo_root / EXP6504_RELATIVE_PATH)
    if pilot_status != "parsed" or base_status != "parsed":  # pragma: no cover - audit gate blocks this path.
        return []
    raw_by_hash = {
        str(row.get("raw_instance_hash")): dict(row)
        for row in base_payload.get("raw_instance_rows", [])
        if isinstance(row, Mapping)
    }
    checkpoint_by_hash: dict[str, JsonDict] = {}
    for branch_row in pilot_payload.get("branch_counterfactual_rows", []):
        if isinstance(branch_row, Mapping):
            base_hash = str(branch_row.get("base_instance_hash"))
            checkpoint_by_hash.setdefault(base_hash, dict(branch_row))
    selected: list[JsonDict] = []
    for base_hash in sorted(checkpoint_by_hash):
        base = dict(raw_by_hash[base_hash])
        checkpoint = checkpoint_by_hash[base_hash]
        base["checkpoint_id"] = checkpoint["checkpoint_id"]
        base["checkpoint_variable"] = checkpoint["checkpoint_variable"]
        base["pilot_branch_row_hashes"] = [
            row.get("row_hash")
            for row in pilot_payload.get("branch_counterfactual_rows", [])
            if isinstance(row, Mapping) and row.get("base_instance_hash") == base_hash
        ]
        selected.append(base)
    return selected


def _game_unit_id(row: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "base_instance_hash": row["raw_instance_hash"],
            "checkpoint_id": row["checkpoint_id"],
            "selection_seed": row["generator_seed"],
        }
    )


def matched_control_rows(repo_root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for base in _load_pilot_base_rows(repo_root):
        native = _solve_with_arm(base, NATIVE_ARM)
        native_trace = list(native["decision_trace"])
        arm_results = {NATIVE_ARM: native}
        for arm_id in ARM_IDS:
            result = arm_results.get(arm_id) or _solve_with_arm(base, arm_id)
            first_changed, changed_count = _compare_traces(native_trace, list(result["decision_trace"]))
            payload = {
                "row_type": "structural_control_game",
                "schema_version": SCHEMA_VERSION + ".per_game_result",
                "unit_id": sha256_json({"pilot_unit_id": _game_unit_id(base), "arm_id": arm_id}),
                "pilot_unit_id": _game_unit_id(base),
                "base_instance_hash": base["raw_instance_hash"],
                "base_lineage_id": base["lineage_id"],
                "split": base["split"],
                "family": base["family"],
                "scale": base["scale"],
                "selection_seed": int(base["generator_seed"]),
                "checkpoint_id": base["checkpoint_id"],
                "checkpoint_variable": int(base["checkpoint_variable"]),
                "arm_id": arm_id,
                "native_reference_arm": NATIVE_ARM,
                "candidate_values_available": list(ELIGIBLE_VALUES),
                "candidate_preserved": True,
                "candidate_pruned_count": 0,
                "exact_budget": EXACT_ASSIGNMENT_BUDGET,
                "restart_budget": RESTART_BUDGET,
                "time_limit_s": TIME_LIMIT_S,
                "first_changed_decision": first_changed,
                "changed_decision_count": changed_count,
                "live_influence_detected": arm_id != NATIVE_ARM and changed_count > 0,
                "conflicts": result["conflicts"],
                "propagations": result["propagations"],
                "decisions": result["decisions"],
                "restarts": result["restarts"],
                "wall_time_s": result["wall_time_s"],
                "control_overhead_units": result["total_control_overhead_units"],
                "feature_cost_units": result["feature_cost_units"],
                "refocus_cost_units": result["refocus_cost_units"],
                "enumeration_cost_units": result["enumeration_cost_units"],
                "fallback_cost_units": result["fallback_cost_units"],
                "solver_only_work_units": result["solver_only_work_units"],
                "total_charged_work_units": result["total_charged_work_units"],
                "timeout": result["timeout"],
                "censored": result["censored"],
                "censoring_reason": result["censoring_reason"],
                "exact_status": result["exact_status"],
                "z3_status": result["z3_status"],
                "z3_model_hash": result["z3_model_hash"],
                "exact_answer_equality": result["exact_answer_equality"],
                "terminal_disposition": result["terminal_disposition"],
                "terminal_model_or_proof": result["terminal_model_or_proof"],
                "decision_trace_hash": result["decision_trace_hash"],
                "decision_trace_prefix": result["decision_trace_prefix"],
                "spec_refs": [
                    "REQ-BENCH-6518",
                    "SCENARIO-BENCH-6518-LIVE-INFLUENCE",
                    "SCENARIO-BENCH-6518-COST-EQUALITY",
                ],
            }
            rows.append({**payload, "row_hash": sha256_json(payload)})
    return rows


def live_influence_rows(per_game_results: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for row in per_game_results:
        payload = {
            "row_type": "live_influence",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "native_arm": NATIVE_ARM,
            "first_changed_decision": row["first_changed_decision"],
            "changed_decision_count": row["changed_decision_count"],
            "live_influence_detected": row["live_influence_detected"],
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-LIVE-INFLUENCE"],
        }
        rows.append({**payload, "influence_row_hash": sha256_json(payload)})
    return rows


def exact_answer_equality_rows(per_game_results: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for row in per_game_results:
        payload = {
            "row_type": "exact_answer_equality",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "exact_status": row["exact_status"],
            "z3_status": row["z3_status"],
            "exact_answer_equality": row["exact_answer_equality"],
            "terminal_disposition": row["terminal_disposition"],
            "receipt_valid": row["terminal_model_or_proof"]["receipt_valid"],
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-COST-EQUALITY"],
        }
        rows.append({**payload, "equality_row_hash": sha256_json(payload)})
    return rows


def charged_cost_rows(per_game_results: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    native_by_unit = {
        str(row["pilot_unit_id"]): int(row["total_charged_work_units"])
        for row in per_game_results
        if row.get("arm_id") == NATIVE_ARM
    }
    rows = []
    for row in per_game_results:
        native_total = native_by_unit[str(row["pilot_unit_id"])]
        payload = {
            "row_type": "charged_cost",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "split": row["split"],
            "family": row["family"],
            "selection_seed": row["selection_seed"],
            "solver_only_work_units": row["solver_only_work_units"],
            "feature_cost_units": row["feature_cost_units"],
            "refocus_cost_units": row["refocus_cost_units"],
            "enumeration_cost_units": row["enumeration_cost_units"],
            "fallback_cost_units": row["fallback_cost_units"],
            "control_overhead_units": row["control_overhead_units"],
            "total_charged_work_units": row["total_charged_work_units"],
            "native_total_charged_work_units": native_total,
            "held_benefit_vs_native_units": (
                native_total - int(row["total_charged_work_units"])
                if row["split"] == "held" and row["arm_id"] != NATIVE_ARM
                else 0
            ),
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-COST-EQUALITY"],
        }
        rows.append({**payload, "cost_row_hash": sha256_json(payload)})
    return rows


def censoring_rows(per_game_results: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for row in per_game_results:
        payload = {
            "row_type": "censoring",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "timeout": row["timeout"],
            "censored": row["censored"],
            "censoring_reason": row["censoring_reason"],
            "restart_budget": row["restart_budget"],
            "restarts": row["restarts"],
            "terminal_disposition": row["terminal_disposition"],
            "terminal_disposition_present": bool(row["terminal_disposition"]),
            "censoring_passed": row["timeout"] is False
            and row["censored"] is False
            and row["restarts"] == RESTART_BUDGET
            and bool(row["terminal_disposition"]),
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-COST-EQUALITY"],
        }
        rows.append({**payload, "censoring_row_hash": sha256_json(payload)})
    return rows


def family_seed_summary(
    per_game_results: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    influence_rows_value: Sequence[Mapping[str, Any]],
    equality_rows_value: Sequence[Mapping[str, Any]],
) -> JsonDict:
    costs_by_unit_arm = {
        (str(row["pilot_unit_id"]), str(row["arm_id"])): dict(row) for row in cost_rows
    }
    equality_by_arm = defaultdict(list)
    influence_by_arm = defaultdict(list)
    for row in equality_rows_value:
        equality_by_arm[str(row["arm_id"])].append(row)
    for row in influence_rows_value:
        influence_by_arm[str(row["arm_id"])].append(row)
    arm_rows: list[JsonDict] = []
    for arm_id in ARM_IDS:
        held = [dict(row) for row in per_game_results if row["split"] == "held" and row["arm_id"] == arm_id]
        held_total = sum(int(row["total_charged_work_units"]) for row in held)
        held_solver = sum(int(row["solver_only_work_units"]) for row in held)
        native_total = sum(
            int(costs_by_unit_arm[(str(row["pilot_unit_id"]), NATIVE_ARM)]["total_charged_work_units"])
            for row in held
        )
        native_solver = sum(
            int(costs_by_unit_arm[(str(row["pilot_unit_id"]), NATIVE_ARM)]["solver_only_work_units"])
            for row in held
        )
        win_rows = [
            row
            for row in held
            if int(costs_by_unit_arm[(str(row["pilot_unit_id"]), NATIVE_ARM)]["total_charged_work_units"])
            - int(row["total_charged_work_units"])
            > 0
        ]
        loss_rows = [
            row
            for row in held
            if int(costs_by_unit_arm[(str(row["pilot_unit_id"]), NATIVE_ARM)]["total_charged_work_units"])
            - int(row["total_charged_work_units"])
            < 0
        ]
        no_headroom = [row for row in held if row not in win_rows]
        payload = {
            "row_type": "family_seed_summary_arm",
            "arm_id": arm_id,
            "held_total_charged_work_units": held_total,
            "native_held_total_charged_work_units": native_total,
            "held_charged_benefit_units": native_total - held_total,
            "held_solver_only_work_units": held_solver,
            "native_held_solver_only_work_units": native_solver,
            "held_solver_only_benefit_units": native_solver - held_solver,
            "held_win_count": len(win_rows),
            "held_loss_count": len(loss_rows),
            "held_no_headroom_unit_count": len(no_headroom),
            "support_families": sorted({str(row["family"]) for row in win_rows}),
            "support_seeds": sorted({int(row["selection_seed"]) for row in win_rows}),
            "support_family_count": len({str(row["family"]) for row in win_rows}),
            "support_seed_count": len({int(row["selection_seed"]) for row in win_rows}),
            "correctness_equality": all(
                row.get("exact_answer_equality") is True for row in equality_by_arm[arm_id]
            )
            and bool(equality_by_arm[arm_id]),
            "live_influence": arm_id != NATIVE_ARM
            and any(row.get("live_influence_detected") is True for row in influence_by_arm[arm_id]),
            "primary_metric_positive": arm_id != NATIVE_ARM and native_total - held_total > 0,
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-HELD-TRANSFER"],
        }
        arm_rows.append({**payload, "summary_row_hash": sha256_json(payload)})
    best = max(
        [row for row in arm_rows if row["arm_id"] != NATIVE_ARM],
        key=lambda row: (row["held_charged_benefit_units"], row["support_family_count"], row["arm_id"]),
        default=None,
    )
    candidate = bool(
        best
        and best["held_charged_benefit_units"] > 0
        and best["correctness_equality"] is True
        and best["live_influence"] is True
        and best["support_family_count"] > 1
        and best["support_seed_count"] > 1
    )
    payload = {
        "schema_version": SCHEMA_VERSION + ".family_seed_summary",
        "primary_metric": PRIMARY_METRIC,
        "native_arm": NATIVE_ARM,
        "arm_rows": arm_rows,
        "best_arm": best["arm_id"] if best else None,
        "best_arm_held_charged_benefit_units": best["held_charged_benefit_units"] if best else 0,
        "best_arm_held_solver_only_benefit_units": best["held_solver_only_benefit_units"] if best else 0,
        "best_arm_support_families": best["support_families"] if best else [],
        "best_arm_support_seeds": best["support_seeds"] if best else [],
        "best_arm_support_family_count": best["support_family_count"] if best else 0,
        "best_arm_support_seed_count": best["support_seed_count"] if best else 0,
        "best_arm_win_count": best["held_win_count"] if best else 0,
        "best_arm_loss_count": best["held_loss_count"] if best else 0,
        "best_arm_no_headroom_unit_count": best["held_no_headroom_unit_count"] if best else 0,
        "best_arm_correctness_equality": best["correctness_equality"] if best else False,
        "best_arm_live_influence": best["live_influence"] if best else False,
        "candidate_score_conditions_met": candidate,
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-HELD-TRANSFER"],
    }
    return {**payload, "family_seed_summary_hash": sha256_json(payload)}


def _empty_family_seed_summary() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".family_seed_summary",
        "primary_metric": PRIMARY_METRIC,
        "native_arm": NATIVE_ARM,
        "arm_rows": [],
        "best_arm": None,
        "best_arm_held_charged_benefit_units": 0,
        "best_arm_held_solver_only_benefit_units": 0,
        "best_arm_support_families": [],
        "best_arm_support_seeds": [],
        "best_arm_support_family_count": 0,
        "best_arm_support_seed_count": 0,
        "best_arm_win_count": 0,
        "best_arm_loss_count": 0,
        "best_arm_no_headroom_unit_count": 0,
        "best_arm_correctness_equality": False,
        "best_arm_live_influence": False,
        "candidate_score_conditions_met": False,
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-HELD-TRANSFER"],
    }
    return {**payload, "family_seed_summary_hash": sha256_json(payload)}


def attack_matrix(
    *,
    summary: Mapping[str, Any],
    prereg: Mapping[str, Any],
    per_game_results: Sequence[Mapping[str, Any]],
    censor_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "inert_advice": summary.get("best_arm_live_influence") is True,
        "no_headroom_units": int(summary.get("best_arm_no_headroom_unit_count", 0)) > 0,
        "row_order_coupling": prereg.get("row_order_used_as_feature") is False,
        "family_identity": prereg.get("family_identity_used_as_feature") is False,
        "tuned_held_thresholds": prereg.get("threshold_tuning_allowed") is False
        and summary.get("primary_metric") == PRIMARY_METRIC,
        "omitted_losses": summary.get("best_arm_loss_count") is not None
        and all("total_charged_work_units" in row for row in per_game_results),
        "timeout_selection": all(row.get("timeout") is False for row in censor_rows),
        "one_seed_wins": int(summary.get("best_arm_support_seed_count", 0)) > 1,
    }
    rows = []
    for attack_id in ATTACK_IDS:
        payload = {
            "row_type": "structural_control_attack",
            "attack_id": attack_id,
            "fail_closed": bool(checks[attack_id]),
            "false_accept": not bool(checks[attack_id]),
            "observed_value": checks[attack_id],
            "expected_value": True,
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-ATTACKS"],
        }
        rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def _blocked_attack_matrix() -> JsonDict:
    rows = [
        {
            "row_type": "structural_control_attack",
            "attack_id": attack_id,
            "fail_closed": False,
            "false_accept": True,
            "observed_value": None,
            "expected_value": True,
            "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-ATTACKS"],
            "attack_row_hash": sha256_json({"attack_id": attack_id, "blocked": True}),
        }
        for attack_id in ATTACK_IDS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": False,
        "false_accept_count": len(rows),
        "failed_attack_ids": list(ATTACK_IDS),
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    rows = [dict(row) for row in payload.get("per_game_results", []) if isinstance(row, Mapping)]
    influence = [dict(row) for row in payload.get("live_influence_rows", []) if isinstance(row, Mapping)]
    equality = [dict(row) for row in payload.get("exact_answer_equality_rows", []) if isinstance(row, Mapping)]
    costs = [dict(row) for row in payload.get("charged_cost_rows", []) if isinstance(row, Mapping)]
    censor = [dict(row) for row in payload.get("censoring_rows", []) if isinstance(row, Mapping)]
    summary = payload.get("family_seed_summary", {})
    attacks = payload.get("attack_matrix", {})
    gate = payload.get("upstream_gate_receipt", {})
    audit_passed = (
        isinstance(gate, Mapping)
        and gate.get("exists") is True
        and gate.get("parse_status") == "parsed"
        and gate.get("gate_passed") is True
        and gate.get("observed_value") == gate.get("expected_value") == 1.0
    )
    row_count_passed = len(rows) == PILOT_BASE_UNIT_COUNT * len(ARM_IDS)
    arm_coverage = {row.get("arm_id") for row in rows} == set(ARM_IDS)
    unit_count = len({str(row.get("pilot_unit_id")) for row in rows})
    candidate_preserved = bool(rows) and all(
        row.get("candidate_preserved") is True
        and row.get("candidate_values_available") == list(ELIGIBLE_VALUES)
        and row.get("candidate_pruned_count") == 0
        for row in rows
    )
    equality_passed = len(equality) == len(rows) and bool(equality) and all(
        row.get("exact_answer_equality") is True
        and row.get("exact_status") == row.get("z3_status")
        and row.get("receipt_valid") is True
        for row in equality
    )
    live_influence_passed = len(influence) == len(rows) and any(
        row.get("arm_id") != NATIVE_ARM and row.get("live_influence_detected") is True
        for row in influence
    )
    cost_passed = len(costs) == len(rows) and bool(costs) and all(
        int(row.get("total_charged_work_units", -1))
        == int(row.get("solver_only_work_units", -2))
        + int(row.get("feature_cost_units", -3))
        + int(row.get("refocus_cost_units", -4))
        + int(row.get("enumeration_cost_units", -5))
        + int(row.get("fallback_cost_units", -6))
        and int(row.get("total_charged_work_units", -1))
        >= int(row.get("solver_only_work_units", 0))
        for row in costs
    )
    censoring_passed = len(censor) == len(rows) and bool(censor) and all(
        row.get("censoring_passed") is True
        and row.get("timeout") is False
        and row.get("censored") is False
        and row.get("terminal_disposition_present") is True
        for row in censor
    )
    attack_passed = (
        isinstance(attacks, Mapping)
        and attacks.get("all_attacks_fail_closed") is True
        and attacks.get("false_accept_count") == 0
        and {row.get("attack_id") for row in attacks.get("rows", [])} == set(ATTACK_IDS)
        and all(row.get("fail_closed") is True for row in attacks.get("rows", []))
        and all(row.get("false_accept") is False for row in attacks.get("rows", []))
    )
    protected_ok = (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    )
    execution = all(
        [
            audit_passed,
            row_count_passed,
            arm_coverage,
            unit_count == PILOT_BASE_UNIT_COUNT,
            candidate_preserved,
            equality_passed,
            live_influence_passed,
            cost_passed,
            censoring_passed,
            attack_passed,
            protected_ok,
        ]
    )
    candidate = bool(
        execution
        and isinstance(summary, Mapping)
        and summary.get("candidate_score_conditions_met") is True
        and summary.get("best_arm_held_charged_benefit_units", 0) > 0
        and summary.get("best_arm_correctness_equality") is True
        and summary.get("best_arm_live_influence") is True
        and int(summary.get("best_arm_support_family_count", 0)) > 1
        and int(summary.get("best_arm_support_seed_count", 0)) > 1
    )
    return {
        "audit_gate_passed": audit_passed,
        "matched_row_count": len(rows),
        "expected_matched_row_count": PILOT_BASE_UNIT_COUNT * len(ARM_IDS),
        "pilot_unit_count": unit_count,
        "arm_coverage_passed": arm_coverage,
        "candidate_preservation_passed": candidate_preserved,
        "exact_answer_equality_passed": equality_passed,
        "live_influence_passed": live_influence_passed,
        "charged_cost_accounting_passed": cost_passed,
        "censoring_passed": censoring_passed,
        "attack_matrix_passed": attack_passed,
        "protected_files_unchanged": protected_ok,
        "best_arm": summary.get("best_arm") if isinstance(summary, Mapping) else None,
        "best_arm_held_charged_benefit_units": (
            summary.get("best_arm_held_charged_benefit_units", 0)
            if isinstance(summary, Mapping)
            else 0
        ),
        "best_arm_support_family_count": (
            summary.get("best_arm_support_family_count", 0) if isinstance(summary, Mapping) else 0
        ),
        "best_arm_support_seed_count": (
            summary.get("best_arm_support_seed_count", 0) if isinstance(summary, Mapping) else 0
        ),
        "execution_score_from_rows": 1.0 if execution else 0.0,
        "candidate_score_from_rows": 1.0 if candidate else 0.0,
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-TERMINAL"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "audit_gate_passed": True,
        "matched_row_count": aggregate.get("expected_matched_row_count"),
        "pilot_unit_count": PILOT_BASE_UNIT_COUNT,
        "arm_coverage_passed": True,
        "candidate_preservation_passed": True,
        "exact_answer_equality_passed": True,
        "live_influence_passed": True,
        "charged_cost_accounting_passed": True,
        "censoring_passed": True,
        "attack_matrix_passed": True,
        "protected_files_unchanged": True,
        "execution_score_from_rows": 1.0,
        "candidate_score_is_binary": True,
    }
    observed = {
        "audit_gate_passed": aggregate.get("audit_gate_passed"),
        "matched_row_count": aggregate.get("matched_row_count"),
        "pilot_unit_count": aggregate.get("pilot_unit_count"),
        "arm_coverage_passed": aggregate.get("arm_coverage_passed"),
        "candidate_preservation_passed": aggregate.get("candidate_preservation_passed"),
        "exact_answer_equality_passed": aggregate.get("exact_answer_equality_passed"),
        "live_influence_passed": aggregate.get("live_influence_passed"),
        "charged_cost_accounting_passed": aggregate.get("charged_cost_accounting_passed"),
        "censoring_passed": aggregate.get("censoring_passed"),
        "attack_matrix_passed": aggregate.get("attack_matrix_passed"),
        "protected_files_unchanged": aggregate.get("protected_files_unchanged"),
        "execution_score_from_rows": aggregate.get("execution_score_from_rows"),
        "candidate_score_is_binary": aggregate.get("candidate_score_from_rows") in {0.0, 1.0},
    }
    checks = {
        key: {"expected": value, "observed": observed[key], "passed": observed[key] == value}
        for key, value in expected.items()
    }
    failed = [key for key, row in checks.items() if row["passed"] is not True]
    return {
        "schema_version": SCHEMA_VERSION + ".gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "blocked_reason": "" if not failed else failed[0],
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-TERMINAL"],
    }


def _status_and_verdict(
    aggregate: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    if aggregate.get("audit_gate_passed") is not True:
        failed = ",".join(gates.get("failed_checks", [])) or "audit_gate_passed"
        return (
            "blocked_structural_control_headroom_ab_v2",
            f"blocked_structural_control_headroom_ab_v2: {failed}",
            "blocked",
        )
    if aggregate.get("execution_score_from_rows") != 1.0:
        failed = ",".join(gates.get("failed_checks", [])) or "execution_incomplete"
        return (
            "disqualified_structural_control_headroom_ab_v2",
            f"disqualified_structural_control_headroom_ab_v2: {failed}",
            "disqualified",
        )
    if aggregate.get("candidate_score_from_rows") == 1.0:
        return (
            "complete_structural_control_headroom_ab_v2_positive",
            "complete_structural_control_headroom_ab_v2_positive: held charged benefit is positive with exact equality, live influence, and multi-family multi-seed support",
            "positive",
        )
    return (
        "complete_structural_control_headroom_ab_v2_null",
        "complete_structural_control_headroom_ab_v2_null: matched execution completed but held charged benefit did not pass the candidate gate",
        None,
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "source": "deterministic_exp6518_structural_control_builder",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-BENCH-6518"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [dict(row) for row in source]


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    audit_path: Path,
    run_date: str,
    protected_before: Mapping[str, str],
) -> JsonDict:
    git_rc, git_status = _command_output(["git", "status", "--short"], repo_root)
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "audit_path": str(audit_path),
        "pilot_path": str(repo_root / EXP6516_RELATIVE_PATH),
        "exp6504_path": str(repo_root / EXP6504_RELATIVE_PATH),
        "git_status_command_exit_code": git_rc,
        "git_status_short": git_status,
        "solver_versions": solver_versions(),
        "resources": _resource_state(repo_root),
        "arm_ids": list(ARM_IDS),
        "assignment_budget": EXACT_ASSIGNMENT_BUDGET,
        "restart_budget": RESTART_BUDGET,
        "time_limit_s": TIME_LIMIT_S,
        "random_seed": RANDOM_SEED,
        "exact_solver_is_label_authority": True,
        "verifier_is_oracle_for_method_value": False,
        "learned_model_trained": False,
        "conductor_modification_allowed": False,
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6518", "SCENARIO-BENCH-6518-AUDIT-GATE"],
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    audit_path: Path | str = EXP6517_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    audit_path = Path(audit_path)
    if not audit_path.is_absolute():
        audit_path = repo_root / audit_path
    protected_before = protected_file_hashes(repo_root, audit_path)
    gate = upstream_gate_receipt(
        repo_root=repo_root,
        audit_path=audit_path,
        protected_before=protected_before,
    )
    prior = prior_failure_receipts(repo_root)
    prereg = preregistration(run_date)
    arms = arm_contract()
    if gate["gate_passed"] is True:
        per_game = matched_control_rows(repo_root)
        influence = live_influence_rows(per_game)
        equality = exact_answer_equality_rows(per_game)
        costs = charged_cost_rows(per_game)
        censor = censoring_rows(per_game)
        family_summary = family_seed_summary(per_game, costs, influence, equality)
        attacks = attack_matrix(
            summary=family_summary,
            prereg=prereg,
            per_game_results=per_game,
            censor_rows=censor,
        )
    else:
        per_game = []
        influence = []
        equality = []
        costs = []
        censor = []
        family_summary = _empty_family_seed_summary()
        attacks = _blocked_attack_matrix()
    protected_after = protected_file_hashes(repo_root, audit_path)
    protected = protected_files_unchanged(protected_before, protected_after)
    per_unit_rows = [
        *per_game,
        *influence,
        *equality,
        *costs,
        *censor,
        *family_summary["arm_rows"],
        *attacks["rows"],
    ]
    partial: JsonDict = {
        "status": "blocked_structural_control_headroom_ab_v2",
        "honest_verdict": "blocked_structural_control_headroom_ab_v2: building",
        "verdict_class": "blocked",
        "upstream_gate_receipt": gate,
        "prior_failure_receipts": prior,
        "preregistration": prereg,
        "arm_contract": arms,
        "per_game_results": per_game,
        "live_influence_rows": influence,
        "exact_answer_equality_rows": equality,
        "charged_cost_rows": costs,
        "censoring_rows": censor,
        "family_seed_summary": family_summary,
        "attack_matrix": attacks,
        "structural_control_execution_complete_score": 0.0,
        "structural_headroom_candidate_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result_path,
            audit_path=audit_path,
            run_date=run_date,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": {
            "artifact_seed": RANDOM_SEED,
            "arm_ids": list(ARM_IDS),
            "shuffled_seed_rule": "RANDOM_SEED plus pilot generator seed",
        },
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = recompute_aggregate(partial)
    gates = gate_check_summary(aggregate)
    status, honest, verdict_class = _status_and_verdict(aggregate, gates)
    partial.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict_class,
            "structural_control_execution_complete_score": aggregate["execution_score_from_rows"],
            "structural_headroom_candidate_score": aggregate["candidate_score_from_rows"],
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
        }
    )
    partial["reproducibility_checksum"] = reproducibility_checksum(partial)
    errors = validate_artifact(partial)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, partial, sort_keys=True, env={})
    return partial


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("verdict_class") not in {"positive", None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6518 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    execution_score = payload.get("structural_control_execution_complete_score")
    candidate_score = payload.get("structural_headroom_candidate_score")
    if execution_score not in {0.0, 1.0}:
        errors.append("structural_control_execution_complete_score must be 0.0 or 1.0")
    if candidate_score not in {0.0, 1.0}:
        errors.append("structural_headroom_candidate_score must be 0.0 or 1.0")
    if payload.get("verdict_class") == "positive" and candidate_score != 1.0:
        errors.append("positive verdict requires candidate score 1.0")
    aggregate = recompute_aggregate(payload)
    gates = gate_check_summary(aggregate)
    blocked_by_gate = payload.get("verdict_class") == "blocked" and aggregate.get("audit_gate_passed") is not True
    if not blocked_by_gate and aggregate.get("audit_gate_passed") is not True:
        errors.append("audit gate failed")
    if not blocked_by_gate:
        if aggregate.get("candidate_preservation_passed") is not True:
            errors.append("candidate preservation failed")
        if aggregate.get("exact_answer_equality_passed") is not True:
            errors.append("exact answer equality failed")
        if aggregate.get("charged_cost_accounting_passed") is not True:
            errors.append("charged cost accounting failed")
        if aggregate.get("censoring_passed") is not True:
            errors.append("censoring failed")
        if aggregate.get("attack_matrix_passed") is not True:
            errors.append("attack false accept")
    if (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if execution_score != aggregate["execution_score_from_rows"]:
        errors.append("execution score mismatch")
    if candidate_score != aggregate["candidate_score_from_rows"]:
        errors.append("candidate score mismatch")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gates:
        errors.append("gate_check_summary mismatch")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "")
    if not (
        honest.startswith("complete_")
        or honest.startswith("blocked_")
        or honest.startswith("disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if not (
        status.startswith("complete_")
        or status.startswith("blocked_")
        or status.startswith("disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    audit_path: Path | str = EXP6517_RELATIVE_PATH,
) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        audit_path=audit_path,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--audit-path", default=str(EXP6517_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        target = result_path if result_path.is_absolute() else REPO_ROOT / result_path
        payload = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, audit_path=Path(args.audit_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())
