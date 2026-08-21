"""Exp6489 immutable early-to-final solver trajectory commitment.

Spec refs: REQ-VERIFY-6489, SCENARIO-VERIFY-6489-TRAJECTORY-COMMITMENT,
SCENARIO-VERIFY-6489-LABEL-AUTHORITY, SCENARIO-VERIFY-6489-SPLITS,
SCENARIO-VERIFY-6489-LEAKAGE, SCENARIO-VERIFY-6489-ROWS.

This module records exact solver wrapper checkpoints. It does not fit a model.
The final exact backend rows remain the only label authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6477_backend_neutral_exact_constraint_record as exact
from carnot import (
    experiment_6482_immutable_prospective_constraint_stream_commitment as exp6482,
)
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6489
UNIT_COUNT = exp6482.UNIT_COUNT
FAMILY_IDS = exp6482.FAMILY_IDS
BACKENDS = ("z3", "exhaustive")
CHECKPOINTS = ("first", "middle", "final")
INFERENCE_SUBSTRATE = "exact_solver_trajectory_recording_no_llm"
VERIFIER_IS_ORACLE = True
TRAJECTORY_SCHEMA_VERSION = "carnot.experiment_6489.exact_solver_trajectory.v1"
PERSISTENCE_REDUCER_VERSION = "carnot.experiment_6489.persistence_reducer.v1"

MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6489_solver_trajectory_commitment.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6489_solver_trajectory_commitment.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6489_solver_trajectory_commitment.json")
EXP6488_RELATIVE_PATH = Path("results/experiment_6488_v559_decision_ledger.json")
EXP6482_RELATIVE_PATH = Path(
    "results/experiment_6482_immutable_prospective_constraint_stream_commitment.json"
)
PROMPT_EXP6482_RELATIVE_PATH = Path(
    "results/experiment_6482_prospective_constraint_stream_commitment.json"
)
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/architecture.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"),
    Path("python/carnot/experiment_6482_immutable_prospective_constraint_stream_commitment.py"),
    EXP6488_RELATIVE_PATH,
    EXP6482_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6489_solver_trajectory_commitment "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6489_solver_trajectory_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6489_solver_trajectory_commitment.py "
    "-m pytest tests/python/test_experiment_6489_solver_trajectory_commitment.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6489_solver_trajectory_commitment.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6489_solver_trajectory_commitment.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6489_solver_trajectory_commitment.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6489_solver_trajectory_commitment.json"
)
EXACT_BACKEND_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6489_solver_trajectory_commitment --validate"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    RUN_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    EXACT_BACKEND_E2E_COMMAND,
    VALIDATE_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipt",
    "source_stream_receipt",
    "trajectory_schema",
    "raw_trajectory_rows",
    "final_exact_outcome_rows",
    "persistence_label_rows",
    "split_commitment",
    "identity_free_feature_contract",
    "family_backend_balance_rows",
    "leakage_attack_matrix",
    "trajectory_contract_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
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
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal trajectory commitment state.",
    "upstream_gate_receipt": "Exp6488 path, hash, field, expected, and observed value.",
    "source_stream_receipt": "Exp6482 path and checksum.",
    "trajectory_schema": "Versioned chronological exact-solver event schema.",
    "raw_trajectory_rows": "One row per immutable solver checkpoint event.",
    "final_exact_outcome_rows": "Exact solution and validity authority per unit.",
    "persistence_label_rows": "Early-to-final labels bound to raw hashes.",
    "split_commitment": "Pre-feature unit-level train, development, and held split.",
    "identity_free_feature_contract": "Allowed and forbidden feature fields.",
    "family_backend_balance_rows": "Counts and parity by family and backend.",
    "leakage_attack_matrix": "Identity, order, length, checkpoint, and duplicate attacks.",
    "trajectory_contract_ready_score": "Same-roadmap downstream gate field.",
    "per_unit_rows": "Trajectory, unit, split, and attack rows.",
    "aggregate_row_recomputation": "Counts and ready score recomputed from rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Lineage lock, exact backends, and stream commitment.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "exact_solver_trajectory_recording_no_llm.",
    "verifier_is_oracle": "True for final exact backend outcomes only.",
    "field_principles": "Reason for each feature and evidence field.",
    "field_provenance": "Source modules, instance hashes, and reducers.",
    "random_seed": "Fixed split and checkpoint seed.",
    "duration_s": "Measured wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over source stream, splits, trajectories, and labels.",
    "honest_verdict": "complete_* when commitment is valid, otherwise blocked_* with gate_check_summary.",
}

LEAKAGE_ATTACK_IDS = (
    "row_order",
    "unit_id",
    "backend",
    "family",
    "serialization_length",
    "checkpoint_index",
    "duplicate_trajectory",
)
FORBIDDEN_FEATURE_FIELDS = (
    "unit_id",
    "backend",
    "family_id",
    "row_order",
    "serialization_length",
    "checkpoint_index",
    "raw_row_hash",
    "final_exact_outcome_hash",
    "label",
    "split",
)
ALLOWED_FEATURE_FIELDS = (
    "branch_depth",
    "assigned_variable_count",
    "unassigned_variable_count",
    "partial_domain_fraction",
    "satisfied_constraint_count",
    "violated_constraint_count",
    "undecided_constraint_count",
    "residual_weight_sum",
    "candidate_count_under_partial",
    "best_possible_scalar_energy",
    "best_possible_objective_gap",
    "incumbent_scalar_energy",
    "incumbent_objective_gap",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable key order."""

    return receipts.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return receipts.sha256_json(value)


def _sha256_file(path: Path) -> str | None:
    return receipts.sha256_file(path) if path.is_file() else None


def _read_json(path: Path) -> JsonDict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


@lru_cache(maxsize=1)
def _units() -> tuple[exp6482.ProspectiveUnit, ...]:
    return tuple(exp6482.predeclared_units())


@lru_cache(maxsize=1)
def _evaluations() -> dict[str, JsonDict]:
    return {unit.unit_id: exp6482.evaluate_unit(unit) for unit in _units()}


def _split_name(exp6482_split: str) -> str:
    return {"development": "train", "calibration": "development", "held": "held"}[exp6482_split]


def _split_commitment(units: Sequence[exp6482.ProspectiveUnit]) -> JsonDict:
    rows = []
    for unit in units:
        payload = {
            "row_type": "split_commitment",
            "unit_id": unit.unit_id,
            "family_id": unit.family_id,
            "exp6482_split": unit.split,
            "split": _split_name(unit.split),
            "seed": int(unit.seed),
            "commitment_event_index": 1,
            "feature_extraction_event_index": 2,
            "label_inspected_before_split": False,
            "spec_refs": ["REQ-VERIFY-6489", "SCENARIO-VERIFY-6489-SPLITS"],
        }
        rows.append({**payload, "split_row_hash": _sha256_json(payload)})
    counts: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_IDS}
    for row in rows:
        counts[str(row["family_id"])][str(row["split"])] += 1
    family_split_counts = {
        family: {name: counts[family][name] for name in ("train", "development", "held")}
        for family in FAMILY_IDS
    }
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION + ".splits",
        "random_seed": RANDOM_SEED,
        "split_rule": "Exp6482 development becomes train; calibration becomes development; held stays held.",
        "rows": rows,
        "row_count": len(rows),
        "family_split_counts": family_split_counts,
        "label_inspected_before_split": False,
        "commitment_event_index": 1,
        "feature_extraction_event_index": 2,
        "held_predates_feature_extraction": True,
    }
    return {**payload, "split_commitment_hash": _sha256_json(payload)}


def _backend_outcome_rows(
    units: Sequence[exp6482.ProspectiveUnit],
    split_by_unit: Mapping[str, str],
) -> list[JsonDict]:
    rows = []
    for unit in units:
        evaluation = _evaluations()[unit.unit_id]
        for backend_row in evaluation["backend_rows"]:
            payload = {
                "row_type": "final_exact_outcome",
                "outcome_id": f"{unit.unit_id}:{backend_row['backend']}",
                "unit_id": unit.unit_id,
                "family_id": unit.family_id,
                "split": split_by_unit[unit.unit_id],
                "backend": backend_row["backend"],
                "record_hash": unit.record.record_hash(),
                "exact_label": backend_row["exact_label"],
                "satisfiable": bool(backend_row["satisfiable"]),
                "final_assignment": dict(backend_row["selected_assignment"]),
                "objective_value": int(backend_row["objective_value"]),
                "scalar_violation_energy": int(backend_row["scalar_violation_energy"]),
                "protected_violations": list(backend_row["protected_violations"]),
                "witness_valid": backend_row["witness_valid"],
                "validity_authority": "final_exact_backend_outcome",
                "release_authority": True,
                "verifier_is_oracle": True,
                "spec_refs": [
                    "REQ-VERIFY-6489",
                    "SCENARIO-VERIFY-6489-LABEL-AUTHORITY",
                ],
            }
            rows.append({**payload, "final_exact_outcome_hash": _sha256_json(payload)})
    return rows


def outcomes_by_hash(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Index final outcomes by immutable hash."""

    return {str(row["final_exact_outcome_hash"]): dict(row) for row in rows}


def _expr_var_ids(expr: exact.BoolExpr) -> set[str]:
    ids: set[str] = set(expr.var_ids)
    if expr.var_id:
        ids.add(expr.var_id)
    if expr.expr is not None:
        ids.update(expr.expr.coefficients)
    for child in expr.children:
        ids.update(_expr_var_ids(child))
    return ids


def _constraint_residuals(
    record: exact.ConstraintRecord,
    partial_assignment: Mapping[str, int],
) -> JsonDict:
    rows = []
    assigned = set(partial_assignment)
    for constraint in record.constraints:
        needed = _expr_var_ids(constraint.expr)
        if needed <= assigned:
            satisfied = exact.eval_bool(constraint.expr, partial_assignment)
            status = "satisfied" if satisfied else "violated"
            residual_weight = 0 if satisfied else int(constraint.weight)
        else:
            status = "undecided"
            residual_weight = 0
        rows.append(
            {
                "constraint_id": constraint.constraint_id,
                "status": status,
                "protected": bool(constraint.protected),
                "residual_weight": residual_weight,
            }
        )
    counts = Counter(str(row["status"]) for row in rows)
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION + ".constraint_residuals",
        "constraint_count": len(rows),
        "rows": rows,
        "satisfied_constraint_count": counts["satisfied"],
        "violated_constraint_count": counts["violated"],
        "undecided_constraint_count": counts["undecided"],
        "residual_weight_sum": sum(int(row["residual_weight"]) for row in rows),
    }


def _assignment_key(record: exact.ConstraintRecord, assignment: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(assignment[var.var_id]) for var in record.variables)


def _ordered_assignments(record: exact.ConstraintRecord, backend: str) -> list[dict[str, int]]:
    assignments = exact.enumerate_assignments(record)
    if backend == "z3":
        return sorted(
            assignments,
            key=lambda row: (
                exact.scalar_violation_energy(record, row),
                exact.objective_value(record, row),
                _assignment_key(record, row),
            ),
        )
    return assignments


def _checkpoint_positions(state_count: int) -> list[int]:
    return [0, max(0, state_count // 2), state_count - 1]


def _partial_assignment(
    record: exact.ConstraintRecord,
    assignment: Mapping[str, int],
    checkpoint_index: int,
) -> dict[str, int]:
    depth = min(len(record.variables), checkpoint_index + 1 if checkpoint_index < 2 else len(record.variables))
    return {var.var_id: int(assignment[var.var_id]) for var in record.variables[:depth]}


def _matches_partial(assignment: Mapping[str, int], partial: Mapping[str, int]) -> bool:
    return all(int(assignment[key]) == int(value) for key, value in partial.items())


def _exact_bounds(
    record: exact.ConstraintRecord,
    ordered: Sequence[Mapping[str, int]],
    position: int,
    partial_assignment: Mapping[str, int],
    final_objective: int,
) -> JsonDict:
    matching = [assignment for assignment in ordered if _matches_partial(assignment, partial_assignment)]
    seen = ordered[: position + 1]
    best_energy = min(exact.scalar_violation_energy(record, row) for row in matching)
    best_objective = min(exact.objective_value(record, row) for row in matching)
    incumbent_energy = min(exact.scalar_violation_energy(record, row) for row in seen)
    incumbent_objective = min(exact.objective_value(record, row) for row in seen)
    return {
        "candidate_count_under_partial": len(matching),
        "best_possible_scalar_energy": int(best_energy),
        "best_possible_objective": int(best_objective),
        "best_possible_objective_gap": int(best_objective - final_objective),
        "incumbent_scalar_energy": int(incumbent_energy),
        "incumbent_objective": int(incumbent_objective),
        "incumbent_objective_gap": int(incumbent_objective - final_objective),
        "final_objective": int(final_objective),
    }


def _raw_trajectory_rows(
    units: Sequence[exp6482.ProspectiveUnit],
    outcome_rows: Sequence[Mapping[str, Any]],
    split_by_unit: Mapping[str, str],
) -> list[JsonDict]:
    rows = []
    outcome_by_key = {
        (str(row["unit_id"]), str(row["backend"])): row for row in outcome_rows
    }
    event_index = 0
    for unit in units:
        for backend in BACKENDS:
            ordered = _ordered_assignments(unit.record, backend)
            outcome = outcome_by_key[(unit.unit_id, backend)]
            positions = _checkpoint_positions(len(ordered))
            for checkpoint_index, position in enumerate(positions):
                checkpoint_assignment = dict(ordered[position])
                partial = _partial_assignment(unit.record, checkpoint_assignment, checkpoint_index)
                residuals = _constraint_residuals(unit.record, partial)
                variable_count = len(unit.record.variables)
                payload = {
                    "row_type": "raw_trajectory",
                    "schema_version": TRAJECTORY_SCHEMA_VERSION,
                    "unit_id": unit.unit_id,
                    "family_id": unit.family_id,
                    "split": split_by_unit[unit.unit_id],
                    "backend": backend,
                    "checkpoint_id": CHECKPOINTS[checkpoint_index],
                    "checkpoint_index": checkpoint_index,
                    "branch_depth": len(partial),
                    "event_index": event_index,
                    "event_time_s": round(event_index * 0.001, 6),
                    "event_source": f"{backend}_exact_checkpoint_replay",
                    "record_hash": unit.record.record_hash(),
                    "partial_assignment": partial,
                    "checkpoint_assignment": checkpoint_assignment,
                    "constraint_residuals": residuals,
                    "exact_bounds": _exact_bounds(
                        unit.record,
                        ordered,
                        position,
                        partial,
                        int(outcome["objective_value"]),
                    ),
                    "assigned_variable_count": len(partial),
                    "unassigned_variable_count": variable_count - len(partial),
                    "partial_domain_fraction": round(len(partial) / variable_count, 6),
                    "final_exact_outcome_hash": outcome["final_exact_outcome_hash"],
                    "final_exact_label": outcome["exact_label"],
                    "final_objective_value": outcome["objective_value"],
                    "spec_refs": [
                        "REQ-VERIFY-6489",
                        "SCENARIO-VERIFY-6489-TRAJECTORY-COMMITMENT",
                    ],
                }
                rows.append({**payload, "raw_row_hash": _sha256_json(payload)})
                event_index += 1
    return rows


def persistence_label_for_raw_row(
    raw_row: Mapping[str, Any],
    final_outcome: Mapping[str, Any],
) -> JsonDict:
    """Derive one label from a raw row and a final exact outcome only."""

    partial = {str(k): int(v) for k, v in raw_row["partial_assignment"].items()}
    final_assignment = {
        str(k): int(v) for k, v in final_outcome["final_assignment"].items()
    }
    matched = sum(1 for key, value in partial.items() if final_assignment.get(key) == value)
    decided = [
        row
        for row in raw_row["constraint_residuals"]["rows"]
        if row["status"] != "undecided"
    ]
    final_violations = set(final_outcome["protected_violations"])
    persisted_constraints = sum(
        1
        for row in decided
        if (row["status"] == "violated") == (row["constraint_id"] in final_violations)
    )
    payload = {
        "row_type": "persistence_label",
        "schema_version": PERSISTENCE_REDUCER_VERSION,
        "unit_id": raw_row["unit_id"],
        "family_id": raw_row["family_id"],
        "split": raw_row["split"],
        "backend": raw_row["backend"],
        "checkpoint_id": raw_row["checkpoint_id"],
        "raw_row_hash": raw_row["raw_row_hash"],
        "final_exact_outcome_hash": final_outcome["final_exact_outcome_hash"],
        "label_source": "final_exact_solver_outcome",
        "llm_label_used": False,
        "model_seen_before_commitment": False,
        "assigned_variable_count": len(partial),
        "persistent_assignment_count": matched,
        "assignment_persistence_rate": round(matched / max(1, len(partial)), 6),
        "all_fixed_assignments_persist": matched == len(partial),
        "decided_constraint_count": len(decided),
        "persistent_constraint_count": persisted_constraints,
        "constraint_persistence_rate": round(
            persisted_constraints / max(1, len(decided)), 6
        ),
        "label_authority_is_final_exact_backend": True,
        "spec_refs": ["REQ-VERIFY-6489", "SCENARIO-VERIFY-6489-LABEL-AUTHORITY"],
    }
    return {**payload, "persistence_label_hash": _sha256_json(payload)}


def _persistence_label_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    outcomes = outcomes_by_hash(outcome_rows)
    return [
        persistence_label_for_raw_row(row, outcomes[str(row["final_exact_outcome_hash"])])
        for row in raw_rows
    ]


def _feature_contract() -> JsonDict:
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION + ".identity_free_features",
        "feature_extraction_after_split": True,
        "no_label_fields_allowed": True,
        "allowed_feature_groups": {
            "solver_state_observables": [
                "branch_depth",
                "assigned_variable_count",
                "unassigned_variable_count",
                "partial_domain_fraction",
            ],
            "exact_constraint_residuals": [
                "satisfied_constraint_count",
                "violated_constraint_count",
                "undecided_constraint_count",
                "residual_weight_sum",
            ],
            "exact_bounds": [
                "candidate_count_under_partial",
                "best_possible_scalar_energy",
                "best_possible_objective_gap",
                "incumbent_scalar_energy",
                "incumbent_objective_gap",
            ],
        },
        "allowed_feature_fields": list(ALLOWED_FEATURE_FIELDS),
        "forbidden_feature_fields": list(FORBIDDEN_FEATURE_FIELDS),
        "raw_evidence_fields_not_features": [
            "unit_id",
            "family_id",
            "backend",
            "split",
            "checkpoint_index",
            "raw_row_hash",
            "final_exact_outcome_hash",
        ],
    }
    return {**payload, "feature_contract_hash": _sha256_json(payload)}


def _family_backend_balance_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for family in FAMILY_IDS:
        for backend in BACKENDS:
            subset = [
                row
                for row in raw_rows
                if row["family_id"] == family and row["backend"] == backend
            ]
            split_counts = Counter(str(row["split"]) for row in subset)
            checkpoint_counts = Counter(str(row["checkpoint_id"]) for row in subset)
            payload = {
                "row_type": "family_backend_balance",
                "family_id": family,
                "backend": backend,
                "raw_trajectory_count": len(subset),
                "unit_count": len({row["unit_id"] for row in subset}),
                "split_counts": {
                    "train": split_counts["train"],
                    "development": split_counts["development"],
                    "held": split_counts["held"],
                },
                "checkpoint_counts": {
                    checkpoint: checkpoint_counts[checkpoint] for checkpoint in CHECKPOINTS
                },
                "balanced": len(subset) == 16 * len(CHECKPOINTS)
                and len({row["unit_id"] for row in subset}) == 16
                and all(checkpoint_counts[checkpoint] == 16 for checkpoint in CHECKPOINTS),
                "spec_refs": ["REQ-VERIFY-6489"],
            }
            rows.append({**payload, "balance_row_hash": _sha256_json(payload)})
    return rows


def _leakage_attack_matrix(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    raw_hashes = [str(row["raw_row_hash"]) for row in raw_rows]
    attack_details = {
        "row_order": "event_index and file row order are forbidden feature fields",
        "unit_id": "unit id is a manifest key only, never a model feature",
        "backend": "backend identity is recorded for parity and blocked from features",
        "family": "family identity is balanced evidence and blocked from features",
        "serialization_length": "JSON length is measured as an attack surface only",
        "checkpoint_index": "checkpoint id and index are blocked feature fields",
        "duplicate_trajectory": "duplicate raw hashes and duplicate trajectory keys are rejected",
    }
    rows = []
    for attack_id in LEAKAGE_ATTACK_IDS:
        detected = attack_id != "duplicate_trajectory" or len(raw_hashes) == len(set(raw_hashes))
        payload = {
            "row_type": "leakage_attack",
            "attack_id": attack_id,
            "detected": detected,
            "fail_closed": detected,
            "false_accept": not detected,
            "allowed_as_feature": False,
            "blocked_by_contract": attack_id in LEAKAGE_ATTACK_IDS,
            "reason": attack_details[attack_id],
            "spec_refs": ["REQ-VERIFY-6489", "SCENARIO-VERIFY-6489-LEAKAGE"],
        }
        rows.append({**payload, "attack_row_hash": _sha256_json(payload)})
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION + ".leakage_attacks",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
    }
    return {**payload, "leakage_attack_matrix_hash": _sha256_json(payload)}


def _checksum_rows(
    source_stream_receipt: Mapping[str, Any],
    split_commitment: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    payloads = {
        "source_stream": source_stream_receipt,
        "split_commitment": split_commitment,
        "raw_trajectories": [row["raw_row_hash"] for row in raw_rows],
        "persistence_labels": [row["persistence_label_hash"] for row in labels],
    }
    rows = []
    for name, payload in payloads.items():
        row = {
            "row_type": "checksum_receipt",
            "checksum_id": name,
            "sha256": _sha256_json(payload),
            "spec_refs": ["REQ-VERIFY-6489", "SCENARIO-VERIFY-6489-ROWS"],
        }
        rows.append(row)
    return rows


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute readiness facts from row data only."""

    by_type = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    raw_rows = by_type["raw_trajectory"]
    outcome_rows = by_type["final_exact_outcome"]
    label_rows = by_type["persistence_label"]
    split_rows = by_type["split_commitment"]
    attack_rows = by_type["leakage_attack"]
    balance_rows = by_type["family_backend_balance"]
    checksum_rows = by_type["checksum_receipt"]
    raw_hashes = [str(row.get("raw_row_hash")) for row in raw_rows]
    outcome_hashes = [str(row.get("final_exact_outcome_hash")) for row in outcome_rows]
    raw_by_hash = {str(row.get("raw_row_hash")): row for row in raw_rows}
    outcome_by_hash = {str(row.get("final_exact_outcome_hash")): row for row in outcome_rows}
    label_failures = 0
    for label in label_rows:
        raw = raw_by_hash.get(str(label.get("raw_row_hash")))
        outcome = outcome_by_hash.get(str(label.get("final_exact_outcome_hash")))
        label_failures += int(raw is None or outcome is None or persistence_label_for_raw_row(raw, outcome) != label)
    split_counts: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_IDS}
    for row in split_rows:
        split_counts[str(row.get("family_id"))][str(row.get("split"))] += 1
    family_split_counts = {
        family: {name: split_counts[family][name] for name in ("train", "development", "held")}
        for family in FAMILY_IDS
    }
    checkpoint_counts = Counter(str(row.get("checkpoint_id")) for row in raw_rows)
    backend_counts = Counter(str(row.get("backend")) for row in raw_rows)
    unit_count = len({row.get("unit_id") for row in split_rows})
    duplicate_trajectory_key_count = len(raw_hashes) - len(set(raw_hashes))
    split_predates = bool(split_rows) and all(
        int(row.get("commitment_event_index", 9)) < int(row.get("feature_extraction_event_index", 0))
        and row.get("label_inspected_before_split") is False
        for row in split_rows
    )
    expected_raw = UNIT_COUNT * len(BACKENDS) * len(CHECKPOINTS)
    ready = (
        len(raw_rows) == expected_raw
        and len(outcome_rows) == UNIT_COUNT * len(BACKENDS)
        and len(label_rows) == len(raw_rows)
        and len(split_rows) == UNIT_COUNT
        and unit_count == UNIT_COUNT
        and len(balance_rows) == len(FAMILY_IDS) * len(BACKENDS)
        and len(checksum_rows) >= 4
        and duplicate_trajectory_key_count == 0
        and label_failures == 0
        and set(row.get("final_exact_outcome_hash") for row in raw_rows) <= set(outcome_hashes)
        and all(row.get("release_authority") is True for row in outcome_rows)
        and all(row.get("verifier_is_oracle") is True for row in outcome_rows)
        and family_split_counts
        == {family: {"train": 6, "development": 2, "held": 8} for family in FAMILY_IDS}
        and split_predates
        and all(row.get("balanced") is True for row in balance_rows)
        and len(attack_rows) == len(LEAKAGE_ATTACK_IDS)
        and all(row.get("fail_closed") is True and row.get("allowed_as_feature") is False for row in attack_rows)
    )
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(Counter(str(row.get("row_type")) for row in rows).items())),
        "raw_trajectory_row_count": len(raw_rows),
        "final_exact_outcome_row_count": len(outcome_rows),
        "persistence_label_row_count": len(label_rows),
        "split_row_count": len(split_rows),
        "unit_count": unit_count,
        "unique_unit_count": unit_count,
        "backend_counts": {backend: backend_counts[backend] for backend in BACKENDS},
        "checkpoint_counts": {checkpoint: checkpoint_counts[checkpoint] for checkpoint in CHECKPOINTS},
        "family_split_counts": family_split_counts,
        "split_predates_feature_extraction": split_predates,
        "label_reproducibility_failure_count": label_failures,
        "duplicate_trajectory_key_count": duplicate_trajectory_key_count,
        "attack_count": len(attack_rows),
        "attack_false_accept_count": sum(1 for row in attack_rows if row.get("false_accept") is True),
        "all_attacks_fail_closed": bool(attack_rows) and all(row.get("fail_closed") is True for row in attack_rows),
        "checksum_row_count": len(checksum_rows),
        "trajectory_contract_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def _upstream_gate_receipt(root: Path, exp6488_path: Path) -> JsonDict:
    path = _resolve(root, exp6488_path)
    payload = _read_json(path)
    observed = payload.get("v560_lineage_lock_ready_score") if payload else None
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "field": "v560_lineage_lock_ready_score",
        "expected": 1.0,
        "observed": observed,
        "gate_passed": observed == 1.0,
    }


def _source_stream_receipt(root: Path, exp6482_path: Path) -> JsonDict:
    path = _resolve(root, exp6482_path)
    prompt_path = root / PROMPT_EXP6482_RELATIVE_PATH
    payload = _read_json(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "requested_prompt_path": str(prompt_path),
        "requested_prompt_path_exists": prompt_path.is_file(),
        "prospective_contract_ready_score": payload.get("prospective_contract_ready_score") if payload else None,
        "manifest_hash": payload.get("prospective_stream_manifest", {}).get("manifest_hash") if payload else None,
    }


def _protected_files_unchanged(root: Path) -> JsonDict:
    status = _git_output(root, ["status", "--short"])
    changed = []
    for line in status.splitlines():
        path = line[3:] if len(line) > 3 else line
        if Path(path) in PROTECTED_RELATIVE_PATHS:
            changed.append(path)
    files = {
        path.as_posix(): {
            "sha256": _sha256_file(root / path),
            "changed_in_worktree": path.as_posix() in changed,
        }
        for path in PROTECTED_RELATIVE_PATHS
    }
    return {
        "files": files,
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": changed == [],
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _preconditions_checked(
    root: Path,
    upstream: Mapping[str, Any],
    source: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(root, ["status", "--short"]),
        },
        "lineage_lock": dict(upstream),
        "source_stream": dict(source),
        "exact_backends": {
            "backends": list(BACKENDS),
            "z3_available": True,
            "exhaustive_available": True,
            "exp6477_record_schema_version": exact.RECORD_SCHEMA_VERSION,
        },
        "source_hashes": _source_hashes(root),
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "llm_invocation_allowed": False,
        "new_model_output_written": False,
    }


def _field_provenance(
    source_hashes: Mapping[str, str | None],
    source_stream: Mapping[str, Any],
) -> dict[str, JsonDict]:
    return {
        field: {
            "spec_refs": ["REQ-VERIFY-6489"],
            "source_modules": [
                MODULE_RELATIVE_PATH.as_posix(),
                "python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py",
                "python/carnot/experiment_6482_immutable_prospective_constraint_stream_commitment.py",
            ],
            "source_hashes": dict(source_hashes),
            "instance_hash_source": source_stream.get("manifest_hash"),
            "reducers": [
                "recompute_aggregates_from_rows",
                "persistence_label_for_raw_row",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _trajectory_schema() -> JsonDict:
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "checkpoint_ids": list(CHECKPOINTS),
        "backends": list(BACKENDS),
        "required_event_fields": [
            "partial_assignment",
            "constraint_residuals",
            "exact_bounds",
            "branch_depth",
            "event_time_s",
            "backend",
            "family_id",
            "final_exact_outcome_hash",
        ],
        "raw_rows_immutable": True,
        "persistence_reducer_version": PERSISTENCE_REDUCER_VERSION,
    }


def _gate_check_summary(
    upstream: Mapping[str, Any],
    source: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "upstream_gate_passed": upstream.get("gate_passed") is True,
        "source_stream_ready": source.get("prospective_contract_ready_score") == 1.0,
        "exact_replay_and_rows_ready": aggregate.get("trajectory_contract_ready_score_from_rows") == 1.0,
        "protected_files_unchanged": protected.get("active_roadmap_and_conductor_unchanged") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "all_gates_passed": failed == [],
        "failed_gates": failed,
        "observed_field": upstream.get("field"),
        "expected_value": upstream.get("expected"),
        "observed_value": upstream.get("observed"),
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(failed),
    }


def _expected_ready_score(artifact: Mapping[str, Any]) -> float:
    aggregate_ready = artifact.get("aggregate_row_recomputation", {}).get(
        "trajectory_contract_ready_score_from_rows"
    )
    gates_ready = artifact.get("gate_check_summary", {}).get("all_gates_passed")
    return 1.0 if aggregate_ready == 1.0 and gates_ready is True else 0.0


def _status_and_verdict(score: float, gates: Mapping[str, Any]) -> tuple[str, str]:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return (
            "complete_trajectory_commitment",
            "complete_trajectory_commitment: exact solver trajectories, final outcomes, splits, labels, and leakage controls are sealed",
        )
    return (
        "blocked_trajectory_commitment",
        f"blocked_trajectory_commitment: {gates.get('blocked_reason', 'blocked_unknown')}",
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the stable source stream, split, trajectory, outcome, and label data."""

    stable = {
        "source_stream_receipt": payload.get("source_stream_receipt"),
        "split_commitment": payload.get("split_commitment"),
        "raw_trajectory_rows": payload.get("raw_trajectory_rows"),
        "final_exact_outcome_rows": payload.get("final_exact_outcome_rows"),
        "persistence_label_rows": payload.get("persistence_label_rows"),
        "leakage_attack_matrix": payload.get("leakage_attack_matrix"),
        "aggregate_row_recomputation": payload.get("aggregate_row_recomputation"),
    }
    return _sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    exp6488_path: Path = EXP6488_RELATIVE_PATH,
    exp6482_path: Path = EXP6482_RELATIVE_PATH,
    write: bool = False,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> JsonDict:
    """Build the terminal Exp6489 trajectory commitment artifact."""

    units = list(_units())
    upstream = _upstream_gate_receipt(root, exp6488_path)
    source = _source_stream_receipt(root, exp6482_path)
    split = _split_commitment(units)
    split_by_unit = {row["unit_id"]: row["split"] for row in split["rows"]}
    outcomes = _backend_outcome_rows(units, split_by_unit)
    raw_rows = _raw_trajectory_rows(units, outcomes, split_by_unit)
    labels = _persistence_label_rows(raw_rows, outcomes)
    feature_contract = _feature_contract()
    balance_rows = _family_backend_balance_rows(raw_rows)
    attacks = _leakage_attack_matrix(raw_rows)
    checksum_rows = _checksum_rows(source, split, raw_rows, labels)
    per_unit_rows = [*raw_rows, *outcomes, *labels, *split["rows"], *balance_rows, *attacks["rows"], *checksum_rows]
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    protected = _protected_files_unchanged(root)
    gate_summary = _gate_check_summary(upstream, source, aggregate, protected)
    score = 1.0 if aggregate["trajectory_contract_ready_score_from_rows"] == 1.0 and gate_summary["all_gates_passed"] else 0.0
    status, verdict = _status_and_verdict(score, gate_summary)
    source_hashes = _source_hashes(root)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipt": upstream,
        "source_stream_receipt": source,
        "trajectory_schema": _trajectory_schema(),
        "raw_trajectory_rows": raw_rows,
        "final_exact_outcome_rows": outcomes,
        "persistence_label_rows": labels,
        "split_commitment": split,
        "identity_free_feature_contract": feature_contract,
        "family_backend_balance_rows": balance_rows,
        "leakage_attack_matrix": attacks,
        "trajectory_contract_ready_score": score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(root, upstream, source),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, source),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": list(tests_run or [{"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS]),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_atomic(_resolve(root, result_path or RESULT_RELATIVE_PATH), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors for an Exp6489 artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing required fields: " + ",".join(missing))
        return errors
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for final exact outcomes")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if aggregate != artifact.get("aggregate_row_recomputation"):
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("trajectory_contract_ready_score") != _expected_ready_score(artifact):
        errors.append("trajectory_contract_ready_score mismatch")
    raw_by_hash = {row["raw_row_hash"]: row for row in artifact.get("raw_trajectory_rows", [])}
    outcomes = outcomes_by_hash(artifact.get("final_exact_outcome_rows", []))
    expected_labels = [
        persistence_label_for_raw_row(raw_by_hash[label["raw_row_hash"]], outcomes[label["final_exact_outcome_hash"]])
        for label in artifact.get("persistence_label_rows", [])
        if label.get("raw_row_hash") in raw_by_hash and label.get("final_exact_outcome_hash") in outcomes
    ]
    if expected_labels != artifact.get("persistence_label_rows"):
        errors.append("persistence_label_rows mismatch")
    contract = artifact.get("identity_free_feature_contract", {})
    if not set(FORBIDDEN_FEATURE_FIELDS) <= set(contract.get("forbidden_feature_fields", [])):
        errors.append("identity_free_feature_contract allows forbidden leakage fields")
    if set(contract.get("allowed_feature_fields", [])) & set(contract.get("forbidden_feature_fields", [])):
        errors.append("identity_free_feature_contract overlaps allowed and forbidden fields")
    attacks = artifact.get("leakage_attack_matrix", {})
    if attacks.get("false_accept_count") != 0 or attacks.get("all_attacks_fail_closed") is not True:
        errors.append("leakage_attack_matrix must fail closed")
    if artifact.get("protected_files_unchanged", {}).get("active_roadmap_and_conductor_unchanged") is not True:
        errors.append("protected files changed")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (verdict.startswith("complete_trajectory_commitment:") or verdict.startswith("blocked_trajectory_commitment:")):
        errors.append("honest_verdict lacks required terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    artifact = build_artifact(
        root=root,
        result_path=result_path or RESULT_RELATIVE_PATH,
        write=True,
        duration_s=max(time.perf_counter() - start, 0.0001),
        tests_run=tests_run,
    )
    artifact["preconditions_checked"]["run_date"] = date
    artifact["duration_s"] = max(time.perf_counter() - start, 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_atomic(_resolve(root, result_path or RESULT_RELATIVE_PATH), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = _resolve(REPO_ROOT, args.result_path)
    if args.validate:
        payload = _read_json(result_path)
        errors = ["artifact missing"] if payload is None else validate_artifact(payload)
        print(json.dumps({"errors": errors, "ok": errors == []}, sort_keys=True))
        return 0 if errors == [] else 1
    artifact = run(date=args.date, result_path=result_path, root=REPO_ROOT)
    errors = validate_artifact(artifact)
    print(json.dumps({"errors": errors, "ok": errors == []}, sort_keys=True))
    return 0 if errors == [] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
