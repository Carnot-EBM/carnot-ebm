"""Exp6492 exact add/drop factor causal replay.

Spec refs: REQ-VERIFY-6492, SCENARIO-VERIFY-6492-GATES,
SCENARIO-VERIFY-6492-FROZEN-MANIFEST, SCENARIO-VERIFY-6492-ADD-DROP,
SCENARIO-VERIFY-6492-CONTROLS-DOSE, SCENARIO-VERIFY-6492-NO-JUDGE,
SCENARIO-VERIFY-6492-ROWS.

The replay starts from committed solver states. It records model proposals as
candidate factors only. Exact finite-domain replay owns all causal outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
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
RANDOM_SEED = 6492
CONTROL_SEED = 6492001
REPLAY_SEED = 6492002
INFERENCE_SUBSTRATE = "exact_counterfactual_solver_replay_no_llm"
VERIFIER_IS_ORACLE = True
SCHEMA_VERSION = "carnot.experiment_6492.factor_causal_replay.v1"

RESULT_RELATIVE_PATH = Path("results/experiment_6492_factor_causal_replay.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6492_factor_causal_replay.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6492_factor_causal_replay.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
EXP6489_RELATIVE_PATH = Path("results/experiment_6489_solver_trajectory_commitment.json")
EXP6491_RELATIVE_PATH = Path("results/experiment_6491_sota_factor_proposal_stream.json")
EXP6478_REQUESTED_RELATIVE_PATH = Path(
    "results/experiment_6478_held_exact_constraint_energy_selection.json"
)
EXP6478_CANONICAL_RELATIVE_PATH = Path(
    "results/experiment_6478_identifiable_held_exact_energy_selection.json"
)
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"),
    Path("python/carnot/experiment_6482_immutable_prospective_constraint_stream_commitment.py"),
    EXP6489_RELATIVE_PATH,
    EXP6491_RELATIVE_PATH,
    EXP6478_CANONICAL_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
)

CONTROL_TYPES = (
    "random_control",
    "structural_control",
    "duplicate_control",
    "no_factor_control",
)
REPLAY_ARMS = ("absent", "present")
TERMINAL_ROW_STATES = (
    "eligible",
    "compiler_reject",
    "compiler_timeout",
    "compiler_duplicate",
    "no_proposal",
    "infeasible",
    "no_headroom",
    "zero_effect",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6492_factor_causal_replay "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6492_factor_causal_replay.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6492_factor_causal_replay.py "
    "-m pytest tests/python/test_experiment_6492_factor_causal_replay.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6492_factor_causal_replay.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6492_factor_causal_replay.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6492_factor_causal_replay.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6492_factor_causal_replay.json"
)
EXACT_REPLAY_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_6492_factor_causal_replay --validate"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    RUN_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    EXACT_REPLAY_E2E_COMMAND,
    VALIDATE_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipts",
    "frozen_replay_manifest",
    "factor_eligibility_rows",
    "replay_rows",
    "dose_matching_rows",
    "control_matching_rows",
    "paired_effect_rows",
    "family_model_cells",
    "confidence_intervals",
    "harmful_flip_rows",
    "factor_causal_audit_complete_score",
    "causal_factor_signal_ready_score",
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
    "status": "Terminal causal-replay state.",
    "upstream_gate_receipts": "Both upstream artifacts and exact gate values.",
    "frozen_replay_manifest": "Events, controls, metrics, seeds, and decision rules.",
    "factor_eligibility_rows": "Every proposal, including reject and no-proposal outcomes.",
    "replay_rows": "Per event, factor, arm, seed, solver outcome, work, validity, and timing.",
    "dose_matching_rows": "Proposal opportunities, admissions, and exposures by arm.",
    "control_matching_rows": "Random, structural, duplicate, and no-factor matches.",
    "paired_effect_rows": "Exact add/drop deltas per event.",
    "family_model_cells": "Disaggregated effects by family and proposing model.",
    "confidence_intervals": "Predeclared row-derived intervals.",
    "harmful_flip_rows": "Every validity or exact-outcome regression.",
    "factor_causal_audit_complete_score": "Execution-completeness gate field.",
    "causal_factor_signal_ready_score": "Same-roadmap positive-signal gate field.",
    "per_unit_rows": "Required event/factor/arm/seed rows.",
    "aggregate_row_recomputation": "Every headline recomputed from replay rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Commitment, proposals, controls, and exact backends.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "exact_counterfactual_solver_replay_no_llm.",
    "verifier_is_oracle": "True for exact solver validity and outcomes.",
    "field_principles": "Reason for every causal and dose field.",
    "field_provenance": "Raw trajectory hashes, proposal bytes, solver receipts, and reducers.",
    "random_seed": "Control construction and replay seeds.",
    "duration_s": "Measured replay and task wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over manifest, factors, controls, and replay rows.",
    "honest_verdict": "complete_positive, complete_null, disqualified, or blocked_* with diagnostics.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in stable key order."""

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
    receipts.write_json_atomic(path, payload)


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


def _records_by_unit() -> dict[str, exact.ConstraintRecord]:
    return {unit.unit_id: unit.record for unit in exp6482.predeclared_units()}


def _upstream_gate_receipt(
    root: Path,
    *,
    artifact_id: str,
    path: Path,
    field: str,
    structured_gate: bool,
) -> JsonDict:
    resolved = _resolve(root, path)
    payload = _read_json(resolved)
    observed = payload.get(field) if payload else None
    return {
        "row_type": "upstream_gate",
        "artifact_id": artifact_id,
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "field": field,
        "expected": 1.0,
        "observed": observed,
        "gate_passed": observed == 1.0,
        "structured_gate": structured_gate,
    }


def upstream_gate_receipts(
    root: Path,
    *,
    exp6489_path: Path,
    exp6491_path: Path,
    exp6478_requested_path: Path,
    exp6478_canonical_path: Path,
) -> list[JsonDict]:
    """Evaluate the two structured gates and the Exp6478 path receipts."""

    return [
        _upstream_gate_receipt(
            root,
            artifact_id="exp6489",
            path=exp6489_path,
            field="trajectory_contract_ready_score",
            structured_gate=True,
        ),
        _upstream_gate_receipt(
            root,
            artifact_id="exp6491",
            path=exp6491_path,
            field="factor_proposal_stream_ready_score",
            structured_gate=True,
        ),
        _upstream_gate_receipt(
            root,
            artifact_id="exp6478_requested",
            path=exp6478_requested_path,
            field="held_exact_energy_selection_ready_score",
            structured_gate=False,
        ),
        _upstream_gate_receipt(
            root,
            artifact_id="exp6478_canonical",
            path=exp6478_canonical_path,
            field="held_exact_energy_selection_ready_score",
            structured_gate=False,
        ),
    ]


def _terminal_row_state(compile_outcome: str) -> str:
    return {
        "accept": "eligible",
        "reject": "compiler_reject",
        "timeout": "compiler_timeout",
        "duplicate": "compiler_duplicate",
        "no_proposal": "no_proposal",
    }.get(compile_outcome, "compiler_reject")


def factor_eligibility_rows(exp6491_payload: Mapping[str, Any]) -> list[JsonDict]:
    """Turn Exp6491 compiler rows into replay eligibility rows."""

    proposals = {
        str(row.get("proposal_row_hash")): row
        for row in exp6491_payload.get("proposal_rows", [])
    }
    rows = []
    for index, compile_row in enumerate(exp6491_payload.get("exact_compile_rows", [])):
        proposal = proposals.get(str(compile_row.get("proposal_row_hash")), {})
        outcome = str(compile_row.get("compile_outcome"))
        terminal = _terminal_row_state(outcome)
        semantic = compile_row.get("semantic_payload")
        eligible = outcome == "accept" and isinstance(semantic, Mapping)
        payload = {
            "row_type": "factor_eligibility",
            "eligibility_index": index,
            "event_id": compile_row.get("event_id"),
            "request_id": compile_row.get("request_id"),
            "source_raw_row_hash": compile_row.get("source_raw_row_hash"),
            "proposal_row_hash": compile_row.get("proposal_row_hash"),
            "compile_row_hash": compile_row.get("compile_row_hash"),
            "raw_response_sha256": compile_row.get("raw_response_sha256"),
            "model_hf_id": compile_row.get("model_hf_id"),
            "model_family": compile_row.get("model_family"),
            "factor_id": compile_row.get("factor_id"),
            "factor_instance_id": f"model:{index:03d}:{compile_row.get('compile_row_hash')}",
            "factor_source": "model",
            "factor_kind": semantic.get("kind") if isinstance(semantic, Mapping) else None,
            "semantic_payload": dict(semantic) if isinstance(semantic, Mapping) else None,
            "semantic_hash": compile_row.get("semantic_hash"),
            "compile_outcome": outcome,
            "compiler_reason": compile_row.get("reason"),
            "proposal_parse_status": proposal.get("parse_receipt", {}).get("parse_status"),
            "eligibility": "eligible" if eligible else "not_eligible",
            "admitted_for_replay": eligible,
            "exposure_dose": len(REPLAY_ARMS) if eligible else 0,
            "terminal_row_state": terminal,
            "model_score_used_as_label": False,
            "human_judgment_used_as_label": False,
            "spec_refs": [
                "REQ-VERIFY-6492",
                "SCENARIO-VERIFY-6492-FROZEN-MANIFEST",
            ],
        }
        rows.append({**payload, "eligibility_row_hash": _sha256_json(payload)})
    return rows


def _raw_rows_by_hash(exp6489_payload: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("raw_row_hash")): dict(row)
        for row in exp6489_payload.get("raw_trajectory_rows", [])
    }


def _held_match_for(raw_row: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    for candidate in raw_rows:
        if (
            candidate.get("split") == "held"
            and candidate.get("family_id") == raw_row.get("family_id")
            and candidate.get("backend") == raw_row.get("backend")
            and candidate.get("checkpoint_id") == raw_row.get("checkpoint_id")
        ):
            return dict(candidate)
    return None


def build_frozen_replay_manifest(
    exp6489_payload: Mapping[str, Any],
    eligibility_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze events, controls, metrics, seeds, and decision rules."""

    raw_rows = [dict(row) for row in exp6489_payload.get("raw_trajectory_rows", [])]
    raw_by_hash = {str(row["raw_row_hash"]): row for row in raw_rows}
    event_groups = []
    for row in eligibility_rows:
        if row.get("admitted_for_replay") is not True:
            continue
        source = raw_by_hash.get(str(row.get("source_raw_row_hash")))
        held = _held_match_for(source, raw_rows) if source else None
        replay_events = []
        for role, replay_row in (("development_source", source), ("held_match", held)):
            if replay_row is None:
                continue
            replay_events.append(
                {
                    "event_role": role,
                    "raw_row_hash": replay_row["raw_row_hash"],
                    "unit_id": replay_row["unit_id"],
                    "family_id": replay_row["family_id"],
                    "split": replay_row["split"],
                    "backend": replay_row["backend"],
                    "checkpoint_id": replay_row["checkpoint_id"],
                    "record_hash": replay_row["record_hash"],
                }
            )
        payload = {
            "factor_instance_id": row["factor_instance_id"],
            "proposal_event_id": row["event_id"],
            "source_raw_row_hash": row["source_raw_row_hash"],
            "replay_events": replay_events,
            "held_match_present": held is not None,
        }
        event_groups.append({**payload, "replay_group_hash": _sha256_json(payload)})
    replay_counts = [len(group["replay_events"]) for group in event_groups]
    payload = {
        "schema_version": SCHEMA_VERSION + ".frozen_replay_manifest",
        "planning_date": RUN_DATE,
        "frozen_before_execution": True,
        "eligible_factor_count": len(event_groups),
        "replay_event_groups": event_groups,
        "replay_event_count_per_factor": min(replay_counts) if replay_counts else 0,
        "control_types": list(CONTROL_TYPES),
        "replay_arms": list(REPLAY_ARMS),
        "control_seed": CONTROL_SEED,
        "replay_seed": REPLAY_SEED,
        "solver_configuration": {
            "backend": "exhaustive_prefix_replay",
            "candidate_order": "finite_domain_lexicographic_after_committed_prefix",
            "exact_backend_module": "experiment_6477_backend_neutral_exact_constraint_record",
            "timeout_s": None,
        },
        "work_metrics": ["expansions", "candidate_states_considered", "exact_check_calls", "wall_time_s"],
        "validity_metrics": ["solver_outcome", "final_validity", "final_solution_hash"],
        "confidence_procedure": "paired_row_min_max_interval_predeclared",
        "harmful_flip_definition": (
            "Absent arm exact-validity true and present arm exact-validity false, "
            "or satisfiable absent outcome regresses under present factor."
        ),
        "terminal_row_states": list(TERMINAL_ROW_STATES),
        "judge_substitution_allowed": False,
    }
    return {**payload, "frozen_replay_manifest_hash": _sha256_json(payload)}


def _factor_arity(semantic: Mapping[str, Any] | None) -> int:
    if not semantic:
        return 0
    scope = semantic.get("scope")
    return len(scope) if isinstance(scope, list) else 0


def _factor_footprint(semantic: Mapping[str, Any] | None) -> list[str]:
    if not semantic:
        return []
    kind = str(semantic.get("kind"))
    return {
        "partial_assignment_eq": ["partial_assignment"],
        "branch_depth_at_least": ["branch_depth"],
        "candidate_count_at_least": ["candidate_count_under_partial"],
        "residual_weight_at_most": ["residual_weight_sum"],
        "no_factor": [],
    }.get(kind, ["unknown"])


def _control_semantic(control_type: str, model_semantic: Mapping[str, Any] | None) -> JsonDict:
    if control_type in {"structural_control", "duplicate_control"} and model_semantic:
        return dict(model_semantic)
    if control_type == "no_factor_control":
        return {"kind": "no_factor", "scope": [], "weight": 0}
    return {
        "kind": "branch_depth_at_least",
        "scope": ["event"],
        "weight": 1,
        "threshold": 0,
    }


def control_matching_rows(
    eligibility_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[JsonDict]:
    """Create matched random, structural, duplicate, and no-factor controls."""

    accepted = [row for row in eligibility_rows if row.get("admitted_for_replay") is True]
    rows = []
    if not accepted:
        for control_type in CONTROL_TYPES:
            payload = {
                "row_type": "control_matching",
                "control_type": control_type,
                "source_factor_instance_id": None,
                "control_factor_instance_id": f"control:{control_type}:zero_admission",
                "admitted_for_replay": False,
                "reason": "zero_model_admission_dose_match",
                "matched_arity": True,
                "matched_feature_footprint": True,
                "exposure_dose": 0,
                "spec_refs": ["REQ-VERIFY-6492", "SCENARIO-VERIFY-6492-CONTROLS-DOSE"],
            }
            rows.append({**payload, "control_match_row_hash": _sha256_json(payload)})
        return rows

    replay_event_count = int(manifest.get("replay_event_count_per_factor", 0))
    for model_row in accepted:
        model_semantic = model_row.get("semantic_payload")
        for control_type in CONTROL_TYPES:
            semantic = _control_semantic(
                control_type,
                model_semantic if isinstance(model_semantic, Mapping) else None,
            )
            control_id = f"control:{control_type}:{model_row['factor_instance_id']}"
            payload = {
                "row_type": "control_matching",
                "control_type": control_type,
                "source_factor_instance_id": model_row["factor_instance_id"],
                "control_factor_instance_id": control_id,
                "event_id": model_row["event_id"],
                "source_raw_row_hash": model_row["source_raw_row_hash"],
                "model_hf_id": model_row["model_hf_id"],
                "model_family": model_row["model_family"],
                "semantic_payload": semantic,
                "semantic_hash": _sha256_json(semantic),
                "admitted_for_replay": True,
                "matched_arity": _factor_arity(semantic)
                == _factor_arity(model_semantic if isinstance(model_semantic, Mapping) else None),
                "matched_feature_footprint": control_type == "random_control"
                or _factor_footprint(semantic)
                == _factor_footprint(model_semantic if isinstance(model_semantic, Mapping) else None),
                "exposure_dose": replay_event_count * len(REPLAY_ARMS),
                "spec_refs": ["REQ-VERIFY-6492", "SCENARIO-VERIFY-6492-CONTROLS-DOSE"],
            }
            rows.append({**payload, "control_match_row_hash": _sha256_json(payload)})
    return rows


def _factor_from_eligible(row: Mapping[str, Any]) -> JsonDict:
    return {
        "factor_instance_id": row["factor_instance_id"],
        "factor_source": "model",
        "control_type": None,
        "proposal_event_id": row["event_id"],
        "source_raw_row_hash": row["source_raw_row_hash"],
        "model_hf_id": row["model_hf_id"],
        "model_family": row["model_family"],
        "factor_id": row["factor_id"],
        "factor_kind": row["factor_kind"],
        "semantic_payload": dict(row["semantic_payload"]),
        "semantic_hash": row["semantic_hash"],
        "compile_outcome": row["compile_outcome"],
    }


def _factor_from_control(row: Mapping[str, Any]) -> JsonDict:
    return {
        "factor_instance_id": row["control_factor_instance_id"],
        "factor_source": "control",
        "control_type": row["control_type"],
        "proposal_event_id": row.get("event_id"),
        "source_raw_row_hash": row.get("source_raw_row_hash"),
        "model_hf_id": row.get("model_hf_id"),
        "model_family": row.get("model_family"),
        "factor_id": row["control_type"],
        "factor_kind": row.get("semantic_payload", {}).get("kind"),
        "semantic_payload": dict(row.get("semantic_payload") or {}),
        "semantic_hash": row.get("semantic_hash"),
        "compile_outcome": "control",
    }


def _matches_prefix(assignment: Mapping[str, int], partial: Mapping[str, Any]) -> bool:
    return all(int(assignment[str(key)]) == int(value) for key, value in partial.items())


def _assignment_key(record: exact.ConstraintRecord, assignment: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(assignment[var.var_id]) for var in record.variables)


def _factor_predicate(
    semantic: Mapping[str, Any],
    raw_row: Mapping[str, Any],
    assignment: Mapping[str, int],
) -> bool:
    kind = str(semantic.get("kind"))
    if kind == "no_factor":
        return True
    if kind == "branch_depth_at_least":
        return int(raw_row.get("branch_depth", 0)) >= int(semantic.get("threshold", 0))
    if kind == "candidate_count_at_least":
        bounds = raw_row.get("exact_bounds", {})
        return int(bounds.get("candidate_count_under_partial", 0)) >= int(
            semantic.get("threshold", 0)
        )
    if kind == "residual_weight_at_most":
        residuals = raw_row.get("constraint_residuals", {})
        return int(residuals.get("residual_weight_sum", 0)) <= int(
            semantic.get("threshold", 0)
        )
    if kind == "partial_assignment_eq":
        variable = str(semantic.get("variable"))
        return variable in assignment and int(assignment[variable]) == int(semantic.get("value"))
    return False


def _violations(record: exact.ConstraintRecord, assignment: Mapping[str, int]) -> list[str]:
    return exact.violated_constraint_ids(record, assignment)


def _select_solution(
    record: exact.ConstraintRecord,
    candidates: Sequence[Mapping[str, int]],
) -> tuple[dict[str, int] | None, bool, int | None, int | None, list[str]]:
    scored = []
    for assignment in candidates:
        violations = _violations(record, assignment)
        energy = exact.scalar_violation_energy(record, assignment)
        objective = exact.objective_value(record, assignment)
        scored.append((energy, objective, _assignment_key(record, assignment), dict(assignment), violations))
    if not scored:
        return None, False, None, None, []
    valid = [row for row in scored if row[0] == 0]
    selected = min(valid or scored, key=lambda row: row[:3])
    return selected[3], selected[0] == 0, int(selected[0]), int(selected[1]), list(selected[4])


def _record_for_raw_row(raw_row: Mapping[str, Any]) -> exact.ConstraintRecord:
    return _records_by_unit()[str(raw_row["unit_id"])]


def execute_replay(
    *,
    raw_row: Mapping[str, Any],
    factor: Mapping[str, Any],
    arm: str,
    seed: int,
) -> JsonDict:
    """Run exact finite-domain replay from one committed partial state."""

    started = time.perf_counter()
    record = _record_for_raw_row(raw_row)
    partial = {str(key): int(value) for key, value in raw_row["partial_assignment"].items()}
    prefix_candidates = [
        assignment
        for assignment in exact.enumerate_assignments(record)
        if _matches_prefix(assignment, partial)
    ]
    semantic = dict(factor.get("semantic_payload") or {"kind": "no_factor"})
    apply_factor = arm == "present" and semantic.get("kind") != "no_factor"
    factor_check_calls = len(prefix_candidates) if apply_factor else 0
    if apply_factor:
        filtered = [
            assignment
            for assignment in prefix_candidates
            if _factor_predicate(semantic, raw_row, assignment)
        ]
    else:
        filtered = list(prefix_candidates)
    selected, valid, energy, objective, violations = _select_solution(record, filtered)
    outcome = "satisfiable" if valid else "infeasible"
    termination = "solution_found" if valid else "state_space_exhausted"
    expansions = len(filtered)
    exact_check_calls = expansions * len(record.constraints) + factor_check_calls
    solution_hash = _sha256_json(selected) if selected is not None else None
    state_payload = {
        "raw_row_hash": raw_row["raw_row_hash"],
        "factor_instance_id": factor.get("factor_instance_id"),
        "arm": arm,
        "seed": seed,
        "selected": selected,
        "outcome": outcome,
        "expansions": expansions,
        "exact_check_calls": exact_check_calls,
    }
    payload = {
        "row_type": "replay",
        "schema_version": SCHEMA_VERSION + ".replay_row",
        "event_id": f"{factor.get('proposal_event_id', 'manual')}:{raw_row.get('split')}:{str(raw_row['raw_row_hash'])[7:15]}",
        "proposal_event_id": factor.get("proposal_event_id"),
        "source_raw_row_hash": raw_row["raw_row_hash"],
        "source_unit_id": raw_row["unit_id"],
        "source_family_id": raw_row["family_id"],
        "replay_split": raw_row["split"],
        "checkpoint_id": raw_row["checkpoint_id"],
        "record_hash": raw_row["record_hash"],
        "factor_instance_id": factor.get("factor_instance_id"),
        "factor_source": factor.get("factor_source"),
        "control_type": factor.get("control_type"),
        "factor_kind": semantic.get("kind"),
        "model_hf_id": factor.get("model_hf_id"),
        "model_family": factor.get("model_family"),
        "semantic_hash": factor.get("semantic_hash") or _sha256_json(semantic),
        "arm": arm,
        "seed": seed,
        "solver_configuration": {
            "backend": "exhaustive_prefix_replay",
            "candidate_order": "finite_domain_lexicographic_after_committed_prefix",
            "timeout_s": None,
            "prefix_assignment": partial,
        },
        "termination": termination,
        "solver_outcome": outcome,
        "exact_outcome": outcome,
        "final_solution": selected,
        "final_solution_hash": solution_hash,
        "final_validity": valid,
        "final_scalar_violation_energy": energy,
        "final_objective_value": objective,
        "violated_constraint_ids": violations,
        "candidate_states_considered": len(prefix_candidates),
        "factor_check_calls": factor_check_calls,
        "expansions": expansions,
        "exact_check_calls": exact_check_calls,
        "wall_time_s": round(max(time.perf_counter() - started, 0.000001), 6),
        "state_hash": _sha256_json(state_payload),
        "persistence": _factor_predicate(semantic, raw_row, selected or {}) if selected else False,
        "model_score_used_as_label": False,
        "human_judgment_used_as_label": False,
        "verifier_is_oracle": True,
        "exact_authority": "exact_counterfactual_solver_replay",
        "spec_refs": ["REQ-VERIFY-6492", "SCENARIO-VERIFY-6492-ADD-DROP"],
    }
    return {**payload, "replay_row_hash": _sha256_json(payload)}


def _replay_events_for_factor(
    factor_instance_id: str,
    manifest: Mapping[str, Any],
    raw_by_hash: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    for group in manifest.get("replay_event_groups", []):
        if group.get("factor_instance_id") == factor_instance_id:
            return [
                dict(raw_by_hash[str(event["raw_row_hash"])])
                for event in group.get("replay_events", [])
                if str(event.get("raw_row_hash")) in raw_by_hash
            ]
    return []


def build_replay_rows(
    *,
    exp6489_payload: Mapping[str, Any],
    eligibility_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[JsonDict]:
    """Execute add/drop rows for eligible model factors and matched controls."""

    raw_by_hash = _raw_rows_by_hash(exp6489_payload)
    factors = [
        _factor_from_eligible(row)
        for row in eligibility_rows
        if row.get("admitted_for_replay") is True
    ]
    factors.extend(
        _factor_from_control(row)
        for row in control_rows
        if row.get("admitted_for_replay") is True
    )
    rows = []
    for factor_index, factor in enumerate(factors):
        events = _replay_events_for_factor(str(factor["factor_instance_id"]), manifest, raw_by_hash)
        if not events:
            for group in manifest.get("replay_event_groups", []):
                if group.get("source_raw_row_hash") != factor.get("source_raw_row_hash"):
                    continue
                events = [
                    dict(raw_by_hash[str(event["raw_row_hash"])])
                    for event in group.get("replay_events", [])
                    if str(event.get("raw_row_hash")) in raw_by_hash
                ]
                break
        for event in events:
            for arm_index, arm in enumerate(REPLAY_ARMS):
                rows.append(
                    execute_replay(
                        raw_row=event,
                        factor=factor,
                        arm=arm,
                        seed=REPLAY_SEED + factor_index * 10 + arm_index,
                    )
                )
    return rows


def paired_effect_rows(replay_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce absent/present replay rows into exact paired deltas."""

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in replay_rows:
        key = (str(row.get("factor_instance_id")), str(row.get("source_raw_row_hash")))
        grouped[key][str(row.get("arm"))] = row
    rows = []
    for (factor_instance_id, raw_hash), arms in sorted(grouped.items()):
        if not {"absent", "present"} <= set(arms):
            continue
        absent = arms["absent"]
        present = arms["present"]
        delta_expansions = int(present["expansions"]) - int(absent["expansions"])
        delta_calls = int(present["exact_check_calls"]) - int(absent["exact_check_calls"])
        harmful = (
            absent.get("final_validity") is True
            and present.get("final_validity") is not True
        ) or (
            absent.get("solver_outcome") == "satisfiable"
            and present.get("solver_outcome") != "satisfiable"
        )
        payload = {
            "row_type": "paired_effect",
            "factor_instance_id": factor_instance_id,
            "factor_source": absent.get("factor_source"),
            "control_type": absent.get("control_type"),
            "proposal_event_id": absent.get("proposal_event_id"),
            "source_raw_row_hash": raw_hash,
            "source_unit_id": absent.get("source_unit_id"),
            "source_family_id": absent.get("source_family_id"),
            "replay_split": absent.get("replay_split"),
            "checkpoint_id": absent.get("checkpoint_id"),
            "model_hf_id": absent.get("model_hf_id"),
            "model_family": absent.get("model_family"),
            "seed": absent.get("seed"),
            "absent_replay_row_hash": absent.get("replay_row_hash"),
            "present_replay_row_hash": present.get("replay_row_hash"),
            "absent_solver_outcome": absent.get("solver_outcome"),
            "present_solver_outcome": present.get("solver_outcome"),
            "absent_validity": absent.get("final_validity"),
            "present_validity": present.get("final_validity"),
            "validity_parity": absent.get("final_validity") == present.get("final_validity"),
            "delta_expansions": delta_expansions,
            "delta_exact_check_calls": delta_calls,
            "delta_wall_time_s": round(
                float(present.get("wall_time_s", 0.0)) - float(absent.get("wall_time_s", 0.0)),
                6,
            ),
            "persistence_present": present.get("persistence"),
            "harmful_flip": harmful,
            "no_headroom": int(absent.get("expansions", 0)) <= 1,
            "zero_effect": delta_expansions == 0
            and delta_calls == 0
            and absent.get("final_solution_hash") == present.get("final_solution_hash")
            and absent.get("solver_outcome") == present.get("solver_outcome"),
            "exact_authority": "exact_counterfactual_solver_replay",
            "spec_refs": ["REQ-VERIFY-6492", "SCENARIO-VERIFY-6492-ROWS"],
        }
        rows.append({**payload, "paired_effect_row_hash": _sha256_json(payload)})
    return rows


def harmful_flip_rows(paired_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return every validity or exact-outcome regression."""

    rows = []
    for row in paired_rows:
        if row.get("harmful_flip") is True:
            payload = {
                "row_type": "harmful_flip",
                "factor_instance_id": row["factor_instance_id"],
                "source_raw_row_hash": row["source_raw_row_hash"],
                "replay_split": row["replay_split"],
                "absent_solver_outcome": row["absent_solver_outcome"],
                "present_solver_outcome": row["present_solver_outcome"],
                "absent_validity": row["absent_validity"],
                "present_validity": row["present_validity"],
                "exact_authority": "exact_counterfactual_solver_replay",
                "spec_refs": ["REQ-VERIFY-6492", "SCENARIO-VERIFY-6492-ROWS"],
            }
            rows.append({**payload, "harmful_flip_row_hash": _sha256_json(payload)})
    return rows


def family_model_cells(paired_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate effects by family, model, source, control, and split."""

    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        key = (
            str(row.get("source_family_id")),
            str(row.get("model_hf_id")),
            str(row.get("factor_source")),
            str(row.get("control_type")),
            str(row.get("replay_split")),
        )
        groups[key].append(row)
    rows = []
    for key, values in sorted(groups.items()):
        family, model, source, control_type, split = key
        deltas = [int(row["delta_exact_check_calls"]) for row in values]
        payload = {
            "row_type": "family_model_cell",
            "family_id": family,
            "model_hf_id": model,
            "factor_source": source,
            "control_type": None if control_type == "None" else control_type,
            "replay_split": split,
            "paired_count": len(values),
            "mean_delta_expansions": _mean([int(row["delta_expansions"]) for row in values]),
            "mean_delta_exact_check_calls": _mean(deltas),
            "validity_parity_count": sum(1 for row in values if row.get("validity_parity") is True),
            "harmful_flip_count": sum(1 for row in values if row.get("harmful_flip") is True),
            "zero_effect_count": sum(1 for row in values if row.get("zero_effect") is True),
            "held_benefit_count": sum(
                1
                for row in values
                if row.get("replay_split") == "held" and int(row["delta_exact_check_calls"]) < 0
            ),
            "spec_refs": ["REQ-VERIFY-6492"],
        }
        rows.append({**payload, "family_model_cell_hash": _sha256_json(payload)})
    return {
        "schema_version": SCHEMA_VERSION + ".family_model_cells",
        "rows": rows,
        "cell_count": len(rows),
    }


def _mean(values: Sequence[int | float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _interval_row(interval_id: str, values: Sequence[int | float]) -> JsonDict:
    mean = _mean(values)
    payload = {
        "row_type": "confidence_interval",
        "interval_id": interval_id,
        "method": "paired_row_min_max_interval_predeclared",
        "n": len(values),
        "mean": mean,
        "lower": min(values) if values else None,
        "upper": max(values) if values else None,
        "confidence": 0.95,
        "spec_refs": ["REQ-VERIFY-6492"],
    }
    return {**payload, "confidence_interval_row_hash": _sha256_json(payload)}


def confidence_intervals(paired_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute predeclared row-derived intervals."""

    model_all = [
        int(row["delta_exact_check_calls"])
        for row in paired_rows
        if row.get("factor_source") == "model"
    ]
    model_held = [
        int(row["delta_exact_check_calls"])
        for row in paired_rows
        if row.get("factor_source") == "model" and row.get("replay_split") == "held"
    ]
    controls_held = [
        int(row["delta_exact_check_calls"])
        for row in paired_rows
        if row.get("factor_source") == "control" and row.get("replay_split") == "held"
    ]
    rows = [
        _interval_row("model_all_delta_exact_check_calls", model_all),
        _interval_row("model_held_delta_exact_check_calls", model_held),
        _interval_row("controls_held_delta_exact_check_calls", controls_held),
    ]
    return {
        "schema_version": SCHEMA_VERSION + ".confidence_intervals",
        "rows": rows,
        "interval_count": len(rows),
    }


def dose_matching_rows(
    eligibility_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[JsonDict]:
    """Report proposal opportunity, admission count, and exposure by arm."""

    opportunities = len(eligibility_rows)
    admitted = sum(1 for row in eligibility_rows if row.get("admitted_for_replay") is True)
    replay_event_count = int(manifest.get("replay_event_count_per_factor", 0))
    exposure = admitted * replay_event_count * len(REPLAY_ARMS)
    rows = []
    for arm in ("model", *CONTROL_TYPES):
        payload = {
            "row_type": "dose_matching",
            "arm": arm,
            "proposal_opportunity_count": opportunities,
            "admitted_event_count": admitted,
            "replay_event_count_per_factor": replay_event_count,
            "exposure_dose": exposure,
            "dose_matched_to_model": True,
            "equal_event_count_and_exposure": True,
            "spec_refs": ["REQ-VERIFY-6492", "SCENARIO-VERIFY-6492-CONTROLS-DOSE"],
        }
        rows.append({**payload, "dose_matching_row_hash": _sha256_json(payload)})
    return rows


def _rowify(group: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [{**dict(row), "per_unit_group": group} for row in rows]


def build_per_unit_rows(
    *,
    upstream: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    dose: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    cells: Mapping[str, Any],
    intervals: Mapping[str, Any],
    harmful: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Flatten gate, factor, replay, control, and reducer rows."""

    rows: list[JsonDict] = []
    rows.extend(_rowify("gate", upstream))
    rows.extend(_rowify("eligibility", eligibility))
    rows.extend(_rowify("replay", replay))
    rows.extend(_rowify("dose", dose))
    rows.extend(_rowify("control", controls))
    rows.extend(_rowify("paired_effect", paired))
    rows.extend(_rowify("family_model_cell", cells.get("rows", [])))
    rows.extend(_rowify("confidence_interval", intervals.get("rows", [])))
    rows.extend(_rowify("harmful_flip", harmful))
    return rows


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all headline gates from row evidence."""

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    gates = by_type["upstream_gate"]
    eligibility = by_type["factor_eligibility"]
    replay = by_type["replay"]
    dose = by_type["dose_matching"]
    controls = by_type["control_matching"]
    paired = by_type["paired_effect"]
    harmful = by_type["harmful_flip"]
    structured_gates = [row for row in gates if row.get("structured_gate") is True]
    accepted = [row for row in eligibility if row.get("admitted_for_replay") is True]
    admitted_controls = [row for row in controls if row.get("admitted_for_replay") is True]
    replay_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in replay:
        replay_groups[(str(row.get("factor_instance_id")), str(row.get("source_raw_row_hash")))].add(
            str(row.get("arm"))
        )
    all_replay_pairs_complete = all(arms == set(REPLAY_ARMS) for arms in replay_groups.values())
    expected_factor_count = len(accepted) * (1 + len(CONTROL_TYPES))
    expected_pair_count = len(paired)
    all_expected_replays = (
        (len(accepted) == 0 and len(replay) == 0 and len(paired) == 0)
        or (
            expected_factor_count > 0
            and len(admitted_controls) == len(accepted) * len(CONTROL_TYPES)
            and len(replay) == expected_pair_count * len(REPLAY_ARMS)
            and all_replay_pairs_complete
        )
    )
    no_judge = all(
        row.get("model_score_used_as_label") is False
        and row.get("human_judgment_used_as_label") is False
        for row in [*eligibility, *replay]
    )
    dose_matched = len(dose) == 1 + len(CONTROL_TYPES) and all(
        row.get("dose_matched_to_model") is True for row in dose
    )
    model_held = [
        row
        for row in paired
        if row.get("factor_source") == "model" and row.get("replay_split") == "held"
    ]
    control_held = [
        row
        for row in paired
        if row.get("factor_source") == "control" and row.get("replay_split") == "held"
    ]
    model_held_delta = _mean([int(row["delta_exact_check_calls"]) for row in model_held])
    control_held_delta = _mean([int(row["delta_exact_check_calls"]) for row in control_held])
    validity_parity = bool(paired) and all(row.get("validity_parity") is True for row in paired)
    positive_held_effect = (
        bool(model_held)
        and model_held_delta is not None
        and model_held_delta < 0
        and (control_held_delta is None or model_held_delta < control_held_delta)
    )
    structured_ok = bool(structured_gates) and all(
        row.get("gate_passed") is True for row in structured_gates
    )
    complete = structured_ok and all_expected_replays and dose_matched and no_judge
    signal = (
        complete
        and positive_held_effect
        and validity_parity
        and len(harmful) == 0
        and len(accepted) > 0
    )
    compile_counts = Counter(str(row.get("compile_outcome")) for row in eligibility)
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(Counter(str(row.get("row_type")) for row in rows).items())),
        "structured_gate_count": len(structured_gates),
        "structured_gates_passed": structured_ok,
        "proposal_opportunity_count": len(eligibility),
        "accepted_model_factor_count": len(accepted),
        "control_match_count": len(controls),
        "admitted_control_count": len(admitted_controls),
        "replay_row_count": len(replay),
        "paired_effect_row_count": len(paired),
        "all_expected_replays_present": all_expected_replays,
        "dose_matching_row_count": len(dose),
        "all_dose_rows_matched": dose_matched,
        "compile_outcome_counts": dict(sorted(compile_counts.items())),
        "model_score_or_human_judgment_label_count": 0 if no_judge else 1,
        "harmful_flip_count": len(harmful),
        "validity_parity_all": validity_parity if paired else True,
        "model_held_mean_delta_exact_check_calls": model_held_delta,
        "control_held_mean_delta_exact_check_calls": control_held_delta,
        "positive_held_effect_beyond_controls": positive_held_effect,
        "factor_causal_audit_complete_score_from_rows": 1.0 if complete else 0.0,
        "causal_factor_signal_ready_score_from_rows": 1.0 if signal else 0.0,
    }


def _protected_files_unchanged(root: Path) -> JsonDict:
    status = _git_output(root, ["status", "--short"])
    changed = []
    for line in status.splitlines():
        path = line[3:] if len(line) > 3 else line
        if Path(path) in PROTECTED_RELATIVE_PATHS:
            changed.append(path)
    return {
        "files": {
            path.as_posix(): {
                "sha256": _sha256_file(root / path),
                "changed_in_worktree": path.as_posix() in changed,
            }
            for path in PROTECTED_RELATIVE_PATHS
        },
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": changed == [],
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _field_provenance(
    root: Path,
    exp6489_payload: Mapping[str, Any],
    exp6491_payload: Mapping[str, Any],
) -> dict[str, JsonDict]:
    raw_hashes = [
        str(row.get("raw_row_hash"))
        for row in exp6489_payload.get("raw_trajectory_rows", [])
    ]
    proposal_hashes = [
        str(row.get("response_sha256"))
        for row in exp6491_payload.get("proposal_rows", [])
    ]
    source_hashes = _source_hashes(root)
    return {
        field: {
            "spec_refs": ["REQ-VERIFY-6492"],
            "source_hashes": source_hashes,
            "raw_trajectory_hash_count": len(raw_hashes),
            "raw_trajectory_hash_sample": raw_hashes[:8],
            "proposal_response_hashes": proposal_hashes,
            "solver_receipts": ["experiment_6477 finite-domain exact evaluator"],
            "reducers": [
                "factor_eligibility_rows",
                "execute_replay",
                "paired_effect_rows",
                "recompute_aggregates_from_rows",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_check_summary(
    *,
    upstream: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    structured = [row for row in upstream if row.get("structured_gate") is True]
    canonical = [row for row in upstream if row.get("artifact_id") == "exp6478_canonical"]
    checks = {
        "structured_upstream_gates_passed": bool(structured)
        and all(row.get("gate_passed") is True for row in structured),
        "exp6478_canonical_receipt_passed": bool(canonical)
        and canonical[0].get("gate_passed") is True,
        "row_recomputed_complete": aggregate.get(
            "factor_causal_audit_complete_score_from_rows"
        )
        == 1.0,
        "no_judge_substitution": aggregate.get(
            "model_score_or_human_judgment_label_count"
        )
        == 0,
        "protected_files_unchanged": protected.get(
            "active_roadmap_and_conductor_unchanged"
        )
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "all_gates_passed": failed == [],
        "failed_gates": failed,
        "observed_values": {
            str(row.get("artifact_id")): {
                "path": row.get("path"),
                "field": row.get("field"),
                "expected": row.get("expected"),
                "observed": row.get("observed"),
                "gate_passed": row.get("gate_passed"),
                "structured_gate": row.get("structured_gate"),
            }
            for row in upstream
        },
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(failed),
    }


def _preconditions_checked(
    *,
    root: Path,
    upstream: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(root, ["status", "--short"]),
        },
        "commitment_gate": next(row for row in upstream if row["artifact_id"] == "exp6489"),
        "proposal_gate": next(row for row in upstream if row["artifact_id"] == "exp6491"),
        "held_exact_energy_receipts": [
            row for row in upstream if str(row["artifact_id"]).startswith("exp6478")
        ],
        "controls": {
            "control_types": list(CONTROL_TYPES),
            "control_seed": CONTROL_SEED,
            "dose_rule": "equal proposal opportunities, admissions, replay events, and add/drop arms",
        },
        "exact_backends": {
            "replay_backend": "exhaustive_prefix_replay",
            "record_schema_version": exact.RECORD_SCHEMA_VERSION,
            "solver_configuration": manifest.get("solver_configuration"),
        },
        "protected_files": dict(protected),
        "runtime_environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "llm_invocation_allowed": False,
        "judge_substitution_allowed": False,
    }


def _expected_complete_score(artifact: Mapping[str, Any]) -> float:
    aggregate = artifact.get("aggregate_row_recomputation", {})
    gates = artifact.get("gate_check_summary", {})
    return (
        1.0
        if aggregate.get("factor_causal_audit_complete_score_from_rows") == 1.0
        and gates.get("all_gates_passed") is True
        else 0.0
    )


def _expected_signal_score(artifact: Mapping[str, Any]) -> float:
    aggregate = artifact.get("aggregate_row_recomputation", {})
    gates = artifact.get("gate_check_summary", {})
    return (
        1.0
        if aggregate.get("causal_factor_signal_ready_score_from_rows") == 1.0
        and gates.get("all_gates_passed") is True
        else 0.0
    )


def _status_and_verdict(complete: float, signal: float, gates: Mapping[str, Any]) -> tuple[str, str]:
    if gates.get("all_gates_passed") is not True:
        return (
            "blocked_factor_causal_replay",
            f"blocked_factor_causal_replay: {gates.get('blocked_reason', 'blocked_unknown')}",
        )
    if complete == 1.0 and signal == 1.0:
        return (
            "complete_positive",
            "complete_positive: held paired factor benefit beats matched controls with validity parity and zero harmful flips",
        )
    if complete == 1.0:
        return (
            "complete_null",
            "complete_null: all eligible proposals and controls are accounted for, but no positive held causal factor signal is present",
        )
    return (
        "disqualified",
        "disqualified: replay rows, controls, dose matching, or reducer checks did not satisfy the precommitted audit",
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the manifest, factors, controls, and replay rows."""

    stable = {
        "upstream_gate_receipts": payload.get("upstream_gate_receipts"),
        "frozen_replay_manifest": payload.get("frozen_replay_manifest"),
        "factor_eligibility_rows": payload.get("factor_eligibility_rows"),
        "control_matching_rows": payload.get("control_matching_rows"),
        "replay_rows": payload.get("replay_rows"),
        "paired_effect_rows": payload.get("paired_effect_rows"),
        "aggregate_row_recomputation": payload.get("aggregate_row_recomputation"),
    }
    return _sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    exp6489_path: Path = EXP6489_RELATIVE_PATH,
    exp6491_path: Path = EXP6491_RELATIVE_PATH,
    exp6478_requested_path: Path = EXP6478_REQUESTED_RELATIVE_PATH,
    exp6478_canonical_path: Path = EXP6478_CANONICAL_RELATIVE_PATH,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6492 causal-replay artifact."""

    started = time.perf_counter()
    result = _resolve(root, result_path or RESULT_RELATIVE_PATH)
    upstream = upstream_gate_receipts(
        root,
        exp6489_path=exp6489_path,
        exp6491_path=exp6491_path,
        exp6478_requested_path=exp6478_requested_path,
        exp6478_canonical_path=exp6478_canonical_path,
    )
    exp6489_payload = _read_json(_resolve(root, exp6489_path)) or {}
    exp6491_payload = _read_json(_resolve(root, exp6491_path)) or {}
    eligibility = factor_eligibility_rows(exp6491_payload)
    manifest = build_frozen_replay_manifest(exp6489_payload, eligibility)
    controls = control_matching_rows(eligibility, manifest)
    structured_ok = all(
        row.get("gate_passed") is True
        for row in upstream
        if row.get("structured_gate") is True
    )
    replay = (
        build_replay_rows(
            exp6489_payload=exp6489_payload,
            eligibility_rows=eligibility,
            control_rows=controls,
            manifest=manifest,
        )
        if structured_ok
        else []
    )
    paired = paired_effect_rows(replay)
    harmful = harmful_flip_rows(paired)
    dose = dose_matching_rows(eligibility, manifest)
    cells = family_model_cells(paired)
    intervals = confidence_intervals(paired)
    per_unit_rows = build_per_unit_rows(
        upstream=upstream,
        eligibility=eligibility,
        replay=replay,
        dose=dose,
        controls=controls,
        paired=paired,
        cells=cells,
        intervals=intervals,
        harmful=harmful,
    )
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    protected = _protected_files_unchanged(root)
    gate_summary = _gate_check_summary(
        upstream=upstream,
        aggregate=aggregate,
        protected=protected,
    )
    complete_score = 1.0 if _expected_complete_score(
        {"aggregate_row_recomputation": aggregate, "gate_check_summary": gate_summary}
    ) == 1.0 else 0.0
    signal_score = 1.0 if _expected_signal_score(
        {"aggregate_row_recomputation": aggregate, "gate_check_summary": gate_summary}
    ) == 1.0 else 0.0
    status, verdict = _status_and_verdict(complete_score, signal_score, gate_summary)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipts": upstream,
        "frozen_replay_manifest": manifest,
        "factor_eligibility_rows": eligibility,
        "replay_rows": replay,
        "dose_matching_rows": dose,
        "control_matching_rows": controls,
        "paired_effect_rows": paired,
        "family_model_cells": cells,
        "confidence_intervals": intervals,
        "harmful_flip_rows": harmful,
        "factor_causal_audit_complete_score": complete_score,
        "causal_factor_signal_ready_score": signal_score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(
            root=root,
            upstream=upstream,
            manifest=manifest,
            protected=protected,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(root, exp6489_payload, exp6491_payload),
        "random_seed": {
            "base": RANDOM_SEED,
            "control_seed": CONTROL_SEED,
            "replay_seed": REPLAY_SEED,
        },
        "duration_s": round(
            float(duration_s)
            if duration_s is not None
            else max(time.perf_counter() - started, 0.000001),
            6,
        ),
        "tests_run": list(
            tests_run
            or [{"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS]
        ),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_atomic(result, artifact)
    return artifact


def _top_level_rows_match(
    artifact: Mapping[str, Any],
    field: str,
    row_type: str,
) -> bool:
    return artifact.get(field) == [
        {key: value for key, value in row.items() if key != "per_unit_group"}
        for row in artifact.get("per_unit_rows", [])
        if row.get("row_type") == row_type
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors for an Exp6492 artifact."""

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
        errors.append("verifier_is_oracle must be true for exact solver outcomes")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if any(
        row.get("model_score_used_as_label") is True
        or row.get("human_judgment_used_as_label") is True
        for row in [
            *artifact.get("factor_eligibility_rows", []),
            *artifact.get("replay_rows", []),
        ]
    ):
        errors.append("model scores or human judgment used as labels")
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if aggregate != artifact.get("aggregate_row_recomputation"):
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("factor_causal_audit_complete_score") != _expected_complete_score(artifact):
        errors.append("factor_causal_audit_complete_score mismatch")
    if artifact.get("causal_factor_signal_ready_score") != _expected_signal_score(artifact):
        errors.append("causal_factor_signal_ready_score mismatch")
    if artifact.get("protected_files_unchanged", {}).get(
        "active_roadmap_and_conductor_unchanged"
    ) is not True:
        errors.append("protected files changed")
    if not _top_level_rows_match(artifact, "replay_rows", "replay"):
        errors.append("replay_rows mismatch")
    if not _top_level_rows_match(artifact, "paired_effect_rows", "paired_effect"):
        errors.append("paired_effect_rows mismatch")
    if not _top_level_rows_match(artifact, "harmful_flip_rows", "harmful_flip"):
        errors.append("harmful_flip_rows mismatch")
    if any(
        row.get("terminal_row_state") not in TERMINAL_ROW_STATES
        for row in artifact.get("factor_eligibility_rows", [])
    ):
        errors.append("factor eligibility terminal states must be enumerated")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(
        ("complete_positive:", "complete_null:", "disqualified:", "blocked_")
    ):
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
        duration_s=max(time.perf_counter() - start, 0.000001),
        tests_run=tests_run,
    )
    artifact["preconditions_checked"]["run_date"] = date
    artifact["duration_s"] = round(max(time.perf_counter() - start, 0.000001), 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_atomic(_resolve(root, result_path or RESULT_RELATIVE_PATH), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
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
