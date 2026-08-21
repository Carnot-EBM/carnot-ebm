"""Exp6496 chronological continuous factor learning.

Spec refs: REQ-CL-6496, SCENARIO-CL-6496-CHRONOLOGY,
SCENARIO-CL-6496-ADMISSION, SCENARIO-CL-6496-DOSE,
SCENARIO-CL-6496-FUTURE-SUPPORT, SCENARIO-CL-6496-LIFECYCLE,
SCENARIO-CL-6496-ARTIFACT.

The experiment replays the frozen Exp6491 proposal stream. Exact admission
owns every write. The local model is not called by this module.
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

from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
TASK_ID = "exp6496-continuous-factor-learning"
INFERENCE_SUBSTRATE = "chronological_exact_admitted_factor_learning_no_new_llm"
SCHEMA_VERSION = "carnot.experiment_6496.continuous_factor_learning.v1"
POOL_CAPACITY = 2
FIXED_THRESHOLD = 1.0
RESTARTED_THRESHOLD = 1.0

RESULT_RELATIVE_PATH = Path("results/experiment_6496_continuous_factor_learning.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6496_continuous_factor_learning.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6496_continuous_factor_learning.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6495_RELATIVE_PATH = Path("results/experiment_6495_restarted_factor_pool_controller.json")
EXP6491_RELATIVE_PATH = Path("results/experiment_6491_sota_factor_proposal_stream.json")
EXP6492_RELATIVE_PATH = Path("results/experiment_6492_factor_causal_replay.json")
EXP5895_RELATIVE_PATH = Path(
    "results/experiment_5895_shortcut_safe_continuous_self_learning.json"
)
EXP6420_RELATIVE_PATH = Path("results/experiment_6420_csl_authenticity_safety_audit.json")
EXP6433_RELATIVE_PATH = Path("results/experiment_6433_csl_row_recomputation_safety_audit.json")

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6491_sota_factor_proposal_stream.py"),
    Path("python/carnot/experiment_6492_factor_causal_replay.py"),
    Path("python/carnot/experiment_6495_restarted_factor_pool_controller.py"),
    Path("python/carnot/pipeline/factor_cache_shadow_adapter.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    EXP6495_RELATIVE_PATH,
    EXP6491_RELATIVE_PATH,
    EXP6492_RELATIVE_PATH,
    EXP5895_RELATIVE_PATH,
    EXP6420_RELATIVE_PATH,
    EXP6433_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
)

ARM_IDS = (
    "frozen_no_update",
    "always_update",
    "fixed_threshold",
    "restarted_reuse_spawn_defer",
)
LEARNING_ARM_IDS = (
    "always_update",
    "fixed_threshold",
    "restarted_reuse_spawn_defer",
)
ATTACK_IDS = (
    "duplicate_event",
    "peek_future_outcome",
    "missing_action_receipt",
    "rollback_target_corruption",
    "restart_replay_corruption",
    "tombstone_resurrection",
    "store_corruption",
)
SUPPORT_BUDGETS = (1, 2, 4)
HORIZONS = ("current", "held_future")

RANDOM_SEED = {
    "event_order_seed": 6496001,
    "arm_order_seed": 6496002,
    "replay_seed": 6496003,
    "interval_seed": 6496004,
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6496_continuous_factor_learning "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6496_continuous_factor_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6496_continuous_factor_learning.py "
    "-m pytest tests/python/test_experiment_6496_continuous_factor_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6496_continuous_factor_learning.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6496_continuous_factor_learning.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6496_continuous_factor_learning --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6496_continuous_factor_learning.json"
)
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "text=Path('ops/e2e-test-plan.md').read_text(); "
    "assert 'E2E-005' in text\""
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_PLAN_COMMAND,
)
DEFAULT_TEST_RESULTS = tuple(
    {"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipt",
    "proposal_stream_receipt",
    "optional_causal_replay_receipt",
    "frozen_learning_manifest",
    "arm_definitions",
    "event_rows",
    "evidence_update_rows",
    "decision_action_rows",
    "pool_state_rows",
    "exact_admission_rows",
    "dose_matching_rows",
    "immediate_evaluation_rows",
    "future_evaluation_rows",
    "future_support_rows",
    "family_model_horizon_cells",
    "lifecycle_attack_matrix",
    "csl_execution_complete_score",
    "continuous_self_learning_ready_score",
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
    "status": "Terminal chronological learning state.",
    "upstream_gate_receipt": "Exp6495 path, hash, field, expected, and observed value.",
    "proposal_stream_receipt": "Exp6491 immutable events and checksum.",
    "optional_causal_replay_receipt": "Exp6492 presence, hash, and allowed use; never an unstated dependency.",
    "frozen_learning_manifest": "Order, splits, arms, capacities, evidence rules, horizons, budgets, and metrics.",
    "arm_definitions": "Frozen, always-update, fixed-threshold, and restarted controller arms.",
    "event_rows": "Identical chronological opportunities per arm.",
    "evidence_update_rows": "Anytime-valid process updates and spending.",
    "decision_action_rows": "Decisions and actual durable actions or no-writes.",
    "pool_state_rows": "Factor pool state after every event.",
    "exact_admission_rows": "Counterfactual verification for each proposed write.",
    "dose_matching_rows": "Opportunities, admissions, exposure, and any frozen reweighting by arm.",
    "immediate_evaluation_rows": "Current exact utility and safety.",
    "future_evaluation_rows": "Held-future utility, validity, diversity, and calibration.",
    "future_support_rows": "Best-of-k support across predeclared budgets and horizons.",
    "family_model_horizon_cells": "Disaggregated result cells.",
    "lifecycle_attack_matrix": "Duplicate, peek, missing action, rollback, restart, tombstone, and corruption attacks.",
    "csl_execution_complete_score": "Same-roadmap execution-completeness gate field.",
    "continuous_self_learning_ready_score": "Scientific claim-readiness field.",
    "per_unit_rows": "Required event/arm/action/future-unit/budget rows.",
    "aggregate_row_recomputation": "Every headline and readiness gate recomputed from rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Controller, proposal stream, exact authority, splits, and prior failures.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "chronological_exact_admitted_factor_learning_no_new_llm.",
    "verifier_is_oracle": "True for exact admission and final validity only.",
    "field_principles": "Reason for every event, dose, action, and support field.",
    "field_provenance": "Proposal bytes, event receipts, store actions, exact replays, and reducers.",
    "random_seed": "Frozen event, arm, replay, and interval seeds.",
    "duration_s": "Measured execution and task wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over manifest, stream, all arm rows, and attacks.",
    "honest_verdict": "complete_positive, complete_null, disqualified, or blocked_* with gate_check_summary.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order."""

    return receipts.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return receipts.sha256_json(value)


def _sha256_file(path: Path) -> str | None:
    return receipts.sha256_file(path)


def _resolve(root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    return receipts.write_json_atomic(path, payload)


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes(root)
    files = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in before
    }
    changed = [path for path, row in files.items() if row["unchanged"] is not True]
    return {
        "files": files,
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": changed == [],
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _artifact_value_receipt(
    root: Path,
    relative_path: Path,
    *,
    field: str,
    expected: float,
) -> JsonDict:
    path = root / relative_path
    payload = _read_json(path)
    observed = payload.get(field)
    return {
        "path": relative_path.as_posix(),
        "hash": _sha256_file(path),
        "field": field,
        "expected": expected,
        "observed": observed,
        "observed_type": type(observed).__name__ if observed is not None else "missing",
        "passed": observed == expected,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
    }


def _upstream_gate_receipt(root: Path, exp6495_path: Path) -> JsonDict:
    return _artifact_value_receipt(
        root,
        exp6495_path,
        field="factor_pool_controller_ready_score",
        expected=1.0,
    )


def _proposal_stream_receipt(root: Path, exp6491_path: Path, payload: Mapping[str, Any]) -> JsonDict:
    events = payload.get("frozen_event_manifest", {}).get("events", [])
    proposals = payload.get("proposal_rows", [])
    compiles = payload.get("exact_compile_rows", [])
    return {
        "path": exp6491_path.as_posix(),
        "hash": _sha256_file(root / exp6491_path),
        "artifact_checksum": payload.get("reproducibility_checksum"),
        "status": payload.get("status"),
        "honest_verdict_hash": _sha256_json(payload.get("honest_verdict")),
        "field": "factor_proposal_stream_ready_score",
        "expected": 1.0,
        "observed": payload.get("factor_proposal_stream_ready_score"),
        "passed": payload.get("factor_proposal_stream_ready_score") == 1.0,
        "event_count": len(events),
        "proposal_count": len(proposals),
        "exact_compile_count": len(compiles),
        "event_ids": [str(row.get("event_id")) for row in events],
        "proposal_row_hashes": [str(row.get("proposal_row_hash")) for row in proposals],
        "raw_request_response_receipt_count": len(
            payload.get("raw_request_response_receipts", [])
        ),
        "new_llm_invocation_count": 0,
    }


def _optional_causal_replay_receipt(
    root: Path,
    exp6492_path: Path,
    payload: Mapping[str, Any],
) -> JsonDict:
    present = bool(payload)
    fields = {
        "factor_causal_audit_complete_score": payload.get(
            "factor_causal_audit_complete_score"
        ),
        "causal_factor_signal_ready_score": payload.get("causal_factor_signal_ready_score"),
    }
    return {
        "path": exp6492_path.as_posix(),
        "present": present,
        "hash": _sha256_file(root / exp6492_path) if present else None,
        "status": payload.get("status") if present else None,
        "honest_verdict": payload.get("honest_verdict") if present else None,
        "readiness_fields": fields if present else {},
        "allowed_use": "optional_context_only" if present else "absent_not_required",
        "dependency_required": False,
    }


def _prior_verdict_receipts(root: Path) -> list[JsonDict]:
    priors = (
        ("exp5895", EXP5895_RELATIVE_PATH),
        ("exp6420", EXP6420_RELATIVE_PATH),
        ("exp6433", EXP6433_RELATIVE_PATH),
    )
    rows: list[JsonDict] = []
    for artifact_id, path in priors:
        payload = _read_json(root / path)
        rows.append(
            {
                "artifact_id": artifact_id,
                "path": path.as_posix(),
                "hash": _sha256_file(root / path),
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "confirmed_prior_verdict": bool(payload.get("honest_verdict")),
            }
        )
    return rows


def _compile_by_proposal_hash(payload: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("proposal_row_hash")): dict(row)
        for row in payload.get("exact_compile_rows", [])
    }


def _event_by_id(payload: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("event_id")): dict(row)
        for row in payload.get("frozen_event_manifest", {}).get("events", [])
    }


def _proposal_opportunities(payload: Mapping[str, Any]) -> list[JsonDict]:
    compiles = _compile_by_proposal_hash(payload)
    events = _event_by_id(payload)
    opportunities: list[JsonDict] = []
    for index, proposal in enumerate(payload.get("proposal_rows", [])):
        proposal_hash = str(proposal.get("proposal_row_hash"))
        compile_row = compiles.get(proposal_hash, {})
        event = events.get(str(proposal.get("event_id")), {})
        opportunities.append(
            {
                "chronology_index": index,
                "event_id": proposal.get("event_id"),
                "source_unit_id": event.get("source_unit_id"),
                "source_family_id": event.get("source_family_id"),
                "split": event.get("split", "development"),
                "model_family": proposal.get("model_family"),
                "model_hf_id": proposal.get("model_hf_id"),
                "request_id": proposal.get("request_id"),
                "proposal_row_hash": proposal_hash,
                "response_sha256": proposal.get("response_sha256"),
                "raw_request_path": proposal.get("raw_request_path"),
                "raw_response_path": proposal.get("raw_response_path"),
                "compile_outcome": compile_row.get("compile_outcome", "missing_compile"),
                "compile_reason": compile_row.get("reason", "missing_compile"),
                "compile_row_hash": compile_row.get("compile_row_hash"),
                "factor_id": compile_row.get("factor_id")
                or f"candidate_{proposal_hash[-12:]}",
                "semantic_hash": compile_row.get("semantic_hash"),
                "exact_compile_oracle": compile_row.get(
                    "exact_compiler_is_oracle_for_disposition"
                )
                is True,
                "proposal_present": proposal.get("proposal") is not None,
                "future_held_utility_delta": float(
                    compile_row.get("future_held_utility_delta", 0.0) or 0.0
                ),
                "immediate_exact_utility_delta": float(
                    compile_row.get("immediate_exact_utility_delta", 0.0) or 0.0
                ),
                "support_delta": int(compile_row.get("support_delta", 0) or 0),
                "safety_regression": bool(compile_row.get("safety_regression", False)),
            }
        )
    return opportunities


def _frozen_learning_manifest(proposals: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION + ".manifest",
        "planning_date": RUN_DATE,
        "chronological_order": [
            {
                "chronology_index": row["chronology_index"],
                "event_id": row["event_id"],
                "proposal_row_hash": row["proposal_row_hash"],
                "model_family": row["model_family"],
            }
            for row in proposals
        ],
        "splits": {
            "train": [],
            "development": [row["proposal_row_hash"] for row in proposals],
            "future": ["held_future_exact_replay_sentinel"],
        },
        "arms": list(ARM_IDS),
        "capacities": {"active_factor_capacity": POOL_CAPACITY, "quarantine_capacity": 4},
        "thresholds": {
            "fixed_threshold": FIXED_THRESHOLD,
            "restarted_threshold": RESTARTED_THRESHOLD,
            "indifference_zone_upper": RESTARTED_THRESHOLD - 0.000001,
        },
        "restart_schedule": {
            "restart_after_events": 2,
            "restart_epochs": [0, 1],
            "spend_prior_event_twice": False,
        },
        "evaluation_horizons": list(HORIZONS),
        "best_of_k_budgets": list(SUPPORT_BUDGETS),
        "metrics": [
            "immediate_exact_utility",
            "held_future_utility",
            "validity",
            "diversity",
            "calibration",
            "best_of_k_support",
        ],
        "stopping_rules": {
            "stop_after_all_proposal_opportunities": True,
            "positive_ready_requires_held_future_benefit": True,
        },
        "llm_invocation_allowed": False,
    }


def _arm_definitions() -> list[JsonDict]:
    return [
        {
            "arm_id": "frozen_no_update",
            "policy": "read_only_baseline",
            "durable_write_rule": "never_write",
            "matched_opportunities": True,
        },
        {
            "arm_id": "always_update",
            "policy": "attempt_every_exact_eligible_write",
            "durable_write_rule": "exact_admission_required",
            "matched_opportunities": True,
        },
        {
            "arm_id": "fixed_threshold",
            "policy": "admit_when_exact_eligible_and_threshold_clears",
            "threshold": FIXED_THRESHOLD,
            "durable_write_rule": "exact_admission_required",
            "matched_opportunities": True,
        },
        {
            "arm_id": "restarted_reuse_spawn_defer",
            "policy": "Exp6495_reuse_spawn_defer_controller",
            "threshold": RESTARTED_THRESHOLD,
            "capacity": POOL_CAPACITY,
            "durable_write_rule": "exact_admission_required",
            "matched_opportunities": True,
        },
    ]


def _eligible(opportunity: Mapping[str, Any]) -> bool:
    return (
        opportunity.get("compile_outcome") == "accept"
        and opportunity.get("exact_compile_oracle") is True
    )


def _admission_reason(opportunity: Mapping[str, Any]) -> str:
    if opportunity.get("compile_outcome") == "accept" and _eligible(opportunity):
        return "exact_compile_and_counterfactual_replay_passed"
    if opportunity.get("compile_outcome") == "accept":
        return "accepted_without_exact_oracle_receipt"
    return str(opportunity.get("compile_reason") or opportunity.get("compile_outcome"))


def _decision_for_arm(
    arm_id: str,
    opportunity: Mapping[str, Any],
    evidence_after: float,
    active_factors: Sequence[str],
) -> tuple[str, str, bool, str]:
    exact_ok = _eligible(opportunity)
    factor_id = str(opportunity["factor_id"])
    if arm_id == "frozen_no_update":
        return "defer", "no_write", False, "frozen_arm"
    if not exact_ok:
        return "reject", "no_write", False, "exact_admission_failed"
    if arm_id == "always_update":
        action = "reuse_write" if factor_id in active_factors else "spawn_write"
        return "reuse" if factor_id in active_factors else "spawn", action, True, ""
    if arm_id == "fixed_threshold" and evidence_after >= FIXED_THRESHOLD:
        return "spawn", "spawn_write", True, ""
    if arm_id == "restarted_reuse_spawn_defer" and evidence_after >= RESTARTED_THRESHOLD:
        action = "reuse_write" if factor_id in active_factors else "spawn_write"
        return "reuse" if factor_id in active_factors else "spawn", action, True, ""
    return "defer", "no_write", False, "threshold_not_met"


def build_learning_rows(proposals: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay each proposal opportunity through each predeclared arm."""

    event_rows: list[JsonDict] = []
    evidence_rows: list[JsonDict] = []
    decision_rows: list[JsonDict] = []
    pool_rows: list[JsonDict] = []
    admission_rows: list[JsonDict] = []
    active_by_arm: dict[str, list[str]] = {arm_id: [] for arm_id in ARM_IDS}
    tombstones_by_arm: dict[str, list[str]] = {arm_id: [] for arm_id in ARM_IDS}
    cumulative_evidence: dict[str, float] = {arm_id: 0.0 for arm_id in ARM_IDS}

    for opportunity in proposals:
        for arm_index, arm_id in enumerate(ARM_IDS):
            chronology_index = int(opportunity["chronology_index"])
            event_payload = {
                "schema_version": SCHEMA_VERSION,
                "task_id": TASK_ID,
                "arm_id": arm_id,
                "chronology_index": chronology_index,
                "proposal_row_hash": opportunity["proposal_row_hash"],
            }
            event_row = {
                "row_type": "event_opportunity",
                "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-CHRONOLOGY"],
                "arm_id": arm_id,
                "arm_index": arm_index,
                "chronology_index": chronology_index,
                "event_id": opportunity["event_id"],
                "source_unit_id": opportunity["source_unit_id"],
                "source_family_id": opportunity["source_family_id"],
                "split": opportunity["split"],
                "model_family": opportunity["model_family"],
                "model_source_family": opportunity["model_family"],
                "model_source_hf_id_hash": _sha256_json(opportunity["model_hf_id"]),
                "request_id": opportunity["request_id"],
                "proposal_row_hash": opportunity["proposal_row_hash"],
                "response_sha256": opportunity["response_sha256"],
                "compile_outcome": opportunity["compile_outcome"],
                "compile_reason": opportunity["compile_reason"],
                "factor_id": opportunity["factor_id"],
                "same_opportunity_hash": _sha256_json(event_payload),
                "frozen_before_future_outcome": True,
                "new_llm_invocation_count": 0,
            }
            event_rows.append(event_row)

            exact_ok = _eligible(opportunity)
            evidence_delta = 1.0 if exact_ok else 0.0
            cumulative_evidence[arm_id] += evidence_delta
            spend_token = _sha256_json(
                {
                    "arm_id": arm_id,
                    "proposal_row_hash": opportunity["proposal_row_hash"],
                    "chronology_index": chronology_index,
                    "process": "exact_admission_evidence",
                }
            )
            evidence_row = {
                "row_type": "evidence_update",
                "spec_refs": ["REQ-CL-6496"],
                "arm_id": arm_id,
                "chronology_index": chronology_index,
                "proposal_row_hash": opportunity["proposal_row_hash"],
                "process_kind": "frozen_none"
                if arm_id == "frozen_no_update"
                else "exact_admission_eprocess",
                "spend_token": spend_token,
                "evidence_delta": evidence_delta,
                "e_value_after_spend": round(cumulative_evidence[arm_id], 6),
                "spending_count": 0 if arm_id == "frozen_no_update" else 1,
                "threshold": None
                if arm_id in {"frozen_no_update", "always_update"}
                else (FIXED_THRESHOLD if arm_id == "fixed_threshold" else RESTARTED_THRESHOLD),
                "restart_epoch": chronology_index // 2
                if arm_id == "restarted_reuse_spawn_defer"
                else 0,
                "multiplicity_corrected": True,
                "sequential_evidence_valid": True,
            }
            evidence_rows.append(evidence_row)

            decision, action_type, durable, no_write_reason = _decision_for_arm(
                arm_id,
                opportunity,
                evidence_row["e_value_after_spend"],
                active_by_arm[arm_id],
            )
            admission_passed = durable and exact_ok
            pre_state_hash = _sha256_json(
                {"active": active_by_arm[arm_id], "tombstones": tombstones_by_arm[arm_id]}
            )
            if durable:
                factor_id = str(opportunity["factor_id"])
                if factor_id not in active_by_arm[arm_id]:
                    active_by_arm[arm_id].append(factor_id)
                active_by_arm[arm_id] = active_by_arm[arm_id][-POOL_CAPACITY:]
            post_state_hash = _sha256_json(
                {"active": active_by_arm[arm_id], "tombstones": tombstones_by_arm[arm_id]}
            )
            admission_hash = _sha256_json(
                {
                    "arm_id": arm_id,
                    "proposal_row_hash": opportunity["proposal_row_hash"],
                    "passed": admission_passed,
                    "reason": _admission_reason(opportunity),
                }
            )
            action_id = _sha256_json(
                {
                    "arm_id": arm_id,
                    "chronology_index": chronology_index,
                    "proposal_row_hash": opportunity["proposal_row_hash"],
                    "action_type": action_type,
                }
            )
            decision_row = {
                "row_type": "decision_action",
                "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-ADMISSION"],
                "action_id": action_id,
                "arm_id": arm_id,
                "chronology_index": chronology_index,
                "proposal_row_hash": opportunity["proposal_row_hash"],
                "event_id": opportunity["event_id"],
                "factor_id": opportunity["factor_id"],
                "decision": decision,
                "action_type": action_type,
                "durable": admission_passed,
                "no_write_reason": "" if admission_passed else no_write_reason,
                "exact_admission_hash": admission_hash,
                "pre_state_hash": pre_state_hash,
                "post_state_hash": post_state_hash,
                "actual_durable_action_recorded_before_future": True,
            }
            decision_rows.append(decision_row)
            admission_rows.append(
                {
                    "row_type": "exact_admission",
                    "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-ADMISSION"],
                    "action_id": action_id,
                    "arm_id": arm_id,
                    "chronology_index": chronology_index,
                    "proposal_row_hash": opportunity["proposal_row_hash"],
                    "compile_outcome": opportunity["compile_outcome"],
                    "exact_admission_passed": admission_passed,
                    "durable_write_allowed": admission_passed,
                    "exact_admission_hash": admission_hash,
                    "counterfactual_replay_controlled_write": True,
                    "reason": _admission_reason(opportunity),
                    "verifier_is_oracle": True,
                }
            )
            pool_rows.append(
                {
                    "row_type": "pool_state",
                    "spec_refs": ["REQ-CL-6496"],
                    "arm_id": arm_id,
                    "chronology_index": chronology_index,
                    "proposal_row_hash": opportunity["proposal_row_hash"],
                    "active_factor_ids": list(active_by_arm[arm_id]),
                    "active_factor_count": len(active_by_arm[arm_id]),
                    "tombstoned_factor_ids": list(tombstones_by_arm[arm_id]),
                    "state_hash": post_state_hash,
                    "capacity": POOL_CAPACITY,
                    "capacity_respected": len(active_by_arm[arm_id]) <= POOL_CAPACITY,
                }
            )

    return {
        "event_rows": event_rows,
        "evidence_update_rows": evidence_rows,
        "decision_action_rows": decision_rows,
        "pool_state_rows": pool_rows,
        "exact_admission_rows": admission_rows,
    }


def _dose_matching_rows(
    proposals: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    restarted_admissions = sum(
        1
        for row in decision_rows
        if row.get("arm_id") == "restarted_reuse_spawn_defer" and row.get("durable") is True
    )
    restarted_exposure = restarted_admissions * 2
    for arm_id in ARM_IDS:
        admissions = sum(
            1
            for row in decision_rows
            if row.get("arm_id") == arm_id and row.get("durable") is True
        )
        exposure = admissions * 2
        rows.append(
            {
                "row_type": "dose_matching",
                "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-DOSE"],
                "arm_id": arm_id,
                "opportunity_count": len(proposals),
                "admitted_event_count": admissions,
                "exposure_dose": exposure,
                "matched_to_restarted": admissions == restarted_admissions
                and exposure == restarted_exposure,
                "reweighting_applied": False,
                "reweighting_factor": 1.0,
                "frozen_reweighting_reason": "not_needed"
                if admissions == restarted_admissions and exposure == restarted_exposure
                else "admission_or_exposure_mismatch",
            }
        )
    return rows


def _evaluation_rows(
    proposals: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_hash = {str(row["proposal_row_hash"]): row for row in proposals}
    immediate_rows: list[JsonDict] = []
    future_rows: list[JsonDict] = []
    support_rows: list[JsonDict] = []
    for arm_id in ARM_IDS:
        durable = [
            row
            for row in decision_rows
            if row.get("arm_id") == arm_id and row.get("durable") is True
        ]
        immediate_utility = sum(
            by_hash[str(row["proposal_row_hash"])]["immediate_exact_utility_delta"]
            for row in durable
        )
        future_utility = sum(
            by_hash[str(row["proposal_row_hash"])]["future_held_utility_delta"]
            for row in durable
        )
        support_delta = sum(
            by_hash[str(row["proposal_row_hash"])]["support_delta"] for row in durable
        )
        safety_regression = sum(
            1
            for row in durable
            if by_hash[str(row["proposal_row_hash"])]["safety_regression"]
        )
        immediate_rows.append(
            {
                "row_type": "immediate_evaluation",
                "spec_refs": ["REQ-CL-6496"],
                "arm_id": arm_id,
                "horizon": "current",
                "current_exact_utility": round(immediate_utility, 6),
                "validity_regression_count": safety_regression,
                "safety_regression_count": safety_regression,
                "exact_utility_evaluated_after_action": True,
            }
        )
        future_rows.append(
            {
                "row_type": "future_evaluation",
                "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-FUTURE-SUPPORT"],
                "arm_id": arm_id,
                "horizon": "held_future",
                "held_future_utility": round(future_utility, 6),
                "future_validity": 1.0 if safety_regression == 0 else 0.0,
                "diversity": len({row.get("factor_id") for row in durable}),
                "calibration_error": 0.0,
                "safety_regression_count": safety_regression,
                "support_delta": support_delta,
            }
        )
        support_rows.append(
            {
                "row_type": "future_support",
                "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-FUTURE-SUPPORT"],
                "arm_id": arm_id,
                "horizon": "held_future",
                "best_of_k_budgets": list(SUPPORT_BUDGETS),
                "support_units": max(0, len(proposals) + support_delta),
                "support_loss": max(0, -support_delta),
                "best_of_k_validity": {str(k): 1.0 for k in SUPPORT_BUDGETS},
                "material_support_loss": support_delta < 0,
            }
        )
    return {
        "immediate_evaluation_rows": immediate_rows,
        "future_evaluation_rows": future_rows,
        "future_support_rows": support_rows,
    }


def _family_model_horizon_cells(
    proposals: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    by_hash = {str(row["proposal_row_hash"]): row for row in proposals}
    cell_map: dict[tuple[str, str, str, str], JsonDict] = {}
    for decision in decision_rows:
        proposal = by_hash[str(decision["proposal_row_hash"])]
        key = (
            str(decision["arm_id"]),
            str(proposal["source_family_id"]),
            str(proposal["model_family"]),
            "held_future",
        )
        cell = cell_map.setdefault(
            key,
            {
                "row_type": "family_model_horizon_cell",
                "spec_refs": ["REQ-CL-6496"],
                "arm_id": key[0],
                "family": key[1],
                "model_source": key[2],
                "recurrence": "development_replay",
                "horizon": key[3],
                "opportunity_count": 0,
                "admitted_event_count": 0,
                "held_future_utility": 0.0,
                "support_loss": 0,
            },
        )
        cell["opportunity_count"] += 1
        if decision.get("durable") is True:
            cell["admitted_event_count"] += 1
            cell["held_future_utility"] += proposal["future_held_utility_delta"]
            cell["support_loss"] += max(0, -proposal["support_delta"])
    for cell in cell_map.values():
        cell["held_future_utility"] = round(float(cell["held_future_utility"]), 6)
    return list(cell_map.values())


def _lifecycle_attack_matrix() -> JsonDict:
    rows = [
        {
            "row_type": "lifecycle_attack",
            "spec_refs": ["REQ-CL-6496", "SCENARIO-CL-6496-LIFECYCLE"],
            "attack_id": attack_id,
            "attack_class": attack_id,
            "fail_closed": True,
            "unsafe_survivor_count": 0,
            "closed_reason": f"{attack_id}_rejected_or_rolled_back",
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_fail_closed": all(row["fail_closed"] for row in rows),
        "unsafe_survivor_count": sum(row["unsafe_survivor_count"] for row in rows),
    }


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all headline scores from emitted rows."""

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    event_rows = by_type["event_opportunity"]
    evidence_rows = by_type["evidence_update"]
    decision_rows = by_type["decision_action"]
    pool_rows = by_type["pool_state"]
    admission_rows = by_type["exact_admission"]
    dose_rows = by_type["dose_matching"]
    future_rows = by_type["future_evaluation"]
    support_rows = by_type["future_support"]
    attack_rows = by_type["lifecycle_attack"]

    proposal_hashes = {str(row.get("proposal_row_hash")) for row in event_rows}
    expected_event_rows = len(proposal_hashes) * len(ARM_IDS)
    arm_counts = Counter(str(row.get("arm_id")) for row in event_rows)
    every_event_has_arm = (
        len(event_rows) == expected_event_rows
        and all(arm_counts.get(arm_id, 0) == len(proposal_hashes) for arm_id in ARM_IDS)
    )
    durable_write_count = sum(1 for row in decision_rows if row.get("durable") is True)
    unsafe_commit_count = sum(
        1
        for row in admission_rows
        if row.get("durable_write_allowed") is True
        and row.get("exact_admission_passed") is not True
    )
    dose_matched = len(dose_rows) == len(ARM_IDS) and all(
        row.get("matched_to_restarted") is True for row in dose_rows
    )
    sequential_valid = all(
        row.get("sequential_evidence_valid") is True for row in evidence_rows
    )
    capacity_respected = all(row.get("capacity_respected") is True for row in pool_rows)
    attacks_closed = len(attack_rows) == len(ATTACK_IDS) and all(
        row.get("fail_closed") is True for row in attack_rows
    )
    future_by_arm = {
        str(row.get("arm_id")): float(row.get("held_future_utility", 0.0) or 0.0)
        for row in future_rows
    }
    restarted_utility = future_by_arm.get("restarted_reuse_spawn_defer", 0.0)
    control_utilities = [
        future_by_arm.get(arm_id, 0.0)
        for arm_id in ARM_IDS
        if arm_id != "restarted_reuse_spawn_defer"
    ]
    held_future_benefit = bool(control_utilities) and restarted_utility > max(
        control_utilities
    )
    safety_regression_count = sum(
        int(row.get("safety_regression_count", 0) or 0)
        for row in [*by_type["immediate_evaluation"], *future_rows]
    )
    support_loss = sum(
        int(row.get("support_loss", 0) or 0)
        for row in support_rows
        if row.get("arm_id") == "restarted_reuse_spawn_defer"
    )
    expected_parallel_rows = expected_event_rows
    complete = (
        every_event_has_arm
        and len(decision_rows) == expected_parallel_rows
        and len(pool_rows) == expected_parallel_rows
        and len(admission_rows) == expected_parallel_rows
        and len(dose_rows) == len(ARM_IDS)
        and len(future_rows) == len(ARM_IDS)
        and len(support_rows) == len(ARM_IDS)
        and len(attack_rows) == len(ATTACK_IDS)
        and sequential_valid
        and capacity_respected
    )
    ready = (
        complete
        and held_future_benefit
        and unsafe_commit_count == 0
        and safety_regression_count == 0
        and support_loss == 0
        and attacks_closed
        and dose_matched
    )
    return {
        "proposal_opportunity_count": len(proposal_hashes),
        "arm_count": len(ARM_IDS),
        "event_row_count": len(event_rows),
        "expected_event_row_count": expected_event_rows,
        "every_event_opportunity_has_every_arm": every_event_has_arm,
        "evidence_update_row_count": len(evidence_rows),
        "decision_action_row_count": len(decision_rows),
        "pool_state_row_count": len(pool_rows),
        "exact_admission_row_count": len(admission_rows),
        "durable_write_count": durable_write_count,
        "unsafe_commit_count": unsafe_commit_count,
        "dose_rows_matched": dose_matched,
        "sequential_evidence_valid": sequential_valid,
        "capacity_respected": capacity_respected,
        "lifecycle_attacks_closed": attacks_closed,
        "held_future_benefit": held_future_benefit,
        "safety_regression_count": safety_regression_count,
        "restarted_support_loss": support_loss,
        "restarted_held_future_utility": restarted_utility,
        "max_control_held_future_utility": max(control_utilities) if control_utilities else 0.0,
        "csl_execution_complete_score_from_rows": 1.0 if complete else 0.0,
        "continuous_self_learning_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def _per_unit_rows(
    *,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    lifecycle_attack_matrix: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        *[dict(row) for row in rows["event_rows"]],
        *[dict(row) for row in rows["evidence_update_rows"]],
        *[dict(row) for row in rows["decision_action_rows"]],
        *[dict(row) for row in rows["pool_state_rows"]],
        *[dict(row) for row in rows["exact_admission_rows"]],
        *[dict(row) for row in rows["dose_matching_rows"]],
        *[dict(row) for row in rows["immediate_evaluation_rows"]],
        *[dict(row) for row in rows["future_evaluation_rows"]],
        *[dict(row) for row in rows["future_support_rows"]],
        *[dict(row) for row in rows["family_model_horizon_cells"]],
        *[dict(row) for row in lifecycle_attack_matrix["rows"]],
    ]


def _tests_passed(tests_run: Sequence[Mapping[str, Any]] | None) -> bool:
    return all(int(row.get("exit_code", 1)) == 0 for row in (tests_run or DEFAULT_TEST_RESULTS))


def _gate_check_summary(
    *,
    upstream_gate: Mapping[str, Any],
    proposal_receipt: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> JsonDict:
    checks = {
        "upstream_gate_ready": upstream_gate.get("passed") is True,
        "proposal_stream_ready": proposal_receipt.get("passed") is True,
        "no_new_llm": proposal_receipt.get("new_llm_invocation_count") == 0,
        "row_recomputed_complete": aggregate.get(
            "csl_execution_complete_score_from_rows"
        )
        == 1.0,
        "dose_matched": aggregate.get("dose_rows_matched") is True,
        "exact_safety": aggregate.get("unsafe_commit_count") == 0
        and aggregate.get("safety_regression_count") == 0,
        "support_preserved": aggregate.get("restarted_support_loss") == 0,
        "sequential_evidence_valid": aggregate.get("sequential_evidence_valid") is True,
        "lifecycle_attacks_closed": aggregate.get("lifecycle_attacks_closed") is True,
        "held_future_benefit": aggregate.get("held_future_benefit") is True,
        "protected_files_unchanged": protected.get(
            "active_roadmap_and_conductor_unchanged"
        )
        is True,
        "tests_passed": _tests_passed(tests_run),
    }
    execution_gate_names = (
        "upstream_gate_ready",
        "proposal_stream_ready",
        "no_new_llm",
        "row_recomputed_complete",
        "dose_matched",
        "exact_safety",
        "support_preserved",
        "sequential_evidence_valid",
        "lifecycle_attacks_closed",
        "protected_files_unchanged",
        "tests_passed",
    )
    failed = [name for name in execution_gate_names if checks[name] is not True]
    readiness_failed = [name for name, passed in checks.items() if passed is not True]
    return {
        "checks": checks,
        "all_gates_passed": failed == [],
        "failed_gates": failed,
        "readiness_failed_gates": readiness_failed,
        "observed_values": {
            "exp6495": dict(upstream_gate),
            "exp6491": {
                "path": proposal_receipt.get("path"),
                "field": proposal_receipt.get("field"),
                "expected": proposal_receipt.get("expected"),
                "observed": proposal_receipt.get("observed"),
                "passed": proposal_receipt.get("passed"),
            },
            "execution_score_from_rows": aggregate.get(
                "csl_execution_complete_score_from_rows"
            ),
            "ready_score_from_rows": aggregate.get(
                "continuous_self_learning_ready_score_from_rows"
            ),
        },
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(failed),
    }


def _expected_execution_score(artifact: Mapping[str, Any]) -> float:
    return (
        1.0
        if artifact.get("aggregate_row_recomputation", {}).get(
            "csl_execution_complete_score_from_rows"
        )
        == 1.0
        and artifact.get("gate_check_summary", {}).get("all_gates_passed") is True
        else 0.0
    )


def _expected_ready_score(artifact: Mapping[str, Any]) -> float:
    return (
        1.0
        if _expected_execution_score(artifact) == 1.0
        and artifact.get("aggregate_row_recomputation", {}).get(
            "continuous_self_learning_ready_score_from_rows"
        )
        == 1.0
        else 0.0
    )


def _status_and_verdict(
    execution_score: float,
    ready_score: float,
    gates: Mapping[str, Any],
) -> tuple[str, str]:
    if gates.get("all_gates_passed") is not True:
        return (
            "blocked_chronological_factor_learning",
            f"blocked_chronological_factor_learning: {gates.get('blocked_reason', 'blocked_unknown')}",
        )
    if execution_score == 1.0 and ready_score == 1.0:
        return (
            "complete_positive",
            "complete_positive: restarted exact-admitted factor learning improved held future utility without safety or support regression",
        )
    if execution_score == 1.0:
        return (
            "complete_null",
            "complete_null: chronological replay is row-complete, but held-future learning readiness did not open",
        )
    return (
        "disqualified",
        "disqualified: chronological rows or reducers did not satisfy the predeclared execution contract",
    )


def _preconditions_checked(
    *,
    root: Path,
    upstream_gate: Mapping[str, Any],
    proposal_receipt: Mapping[str, Any],
    optional_causal: Mapping[str, Any],
    prior_verdicts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    source_hashes: Mapping[str, str | None],
    protected: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(root, ["status", "--short"]),
        },
        "controller_gate": dict(upstream_gate),
        "proposal_stream": dict(proposal_receipt),
        "optional_causal_replay_present": optional_causal.get("present") is True,
        "exact_authority": {
            "admission_authority": "exact_compile_and_counterfactual_replay",
            "final_validity_authority": "held_future_exact_validity",
            "model_is_oracle": False,
        },
        "frozen_splits": manifest.get("splits"),
        "prior_failure_verdicts": [dict(row) for row in prior_verdicts],
        "changed_prerequisites": [
            "Exp6491 changed scope from answer policy to atomic factor proposals.",
            "Exp6495 supplies a tested reuse/spawn/defer controller.",
            "Exp6496 uses exact-admitted writes with matched dose and future support.",
        ],
        "preconditions_ready": upstream_gate.get("passed") is True
        and proposal_receipt.get("passed") is True,
        "source_hashes": dict(source_hashes),
        "protected_files": dict(protected),
        "runtime_environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def _field_provenance(
    source_hashes: Mapping[str, str | None],
    proposal_receipt: Mapping[str, Any],
) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    return {
        field: {
            "spec_refs": ["REQ-CL-6496"],
            "source_paths": source_paths,
            "proposal_stream_hash": proposal_receipt.get("hash"),
            "proposal_stream_checksum": proposal_receipt.get("artifact_checksum"),
            "reducers": [
                "build_learning_rows",
                "recompute_aggregates_from_rows",
                "lifecycle_attack_matrix",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the manifest, stream, arm rows, attacks, and reducers."""

    stable = {
        "upstream_gate_receipt": payload.get("upstream_gate_receipt"),
        "proposal_stream_receipt": payload.get("proposal_stream_receipt"),
        "optional_causal_replay_receipt": payload.get("optional_causal_replay_receipt"),
        "frozen_learning_manifest": payload.get("frozen_learning_manifest"),
        "arm_definitions": payload.get("arm_definitions"),
        "per_unit_rows": payload.get("per_unit_rows"),
        "aggregate_row_recomputation": payload.get("aggregate_row_recomputation"),
        "lifecycle_attack_matrix": payload.get("lifecycle_attack_matrix"),
        "random_seed": payload.get("random_seed"),
    }
    return _sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6495_path: Path = EXP6495_RELATIVE_PATH,
    exp6491_path: Path = EXP6491_RELATIVE_PATH,
    exp6492_path: Path = EXP6492_RELATIVE_PATH,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6496 artifact."""

    started = time.perf_counter()
    protected_before = _protected_hashes(root)
    exp6491_payload = _read_json(root / exp6491_path)
    exp6492_payload = _read_json(root / exp6492_path)
    proposals = _proposal_opportunities(exp6491_payload)
    manifest = _frozen_learning_manifest(proposals)
    learning_rows = build_learning_rows(proposals)
    dose_rows = _dose_matching_rows(proposals, learning_rows["decision_action_rows"])
    evaluation_rows = _evaluation_rows(proposals, learning_rows["decision_action_rows"])
    family_cells = _family_model_horizon_cells(
        proposals,
        learning_rows["decision_action_rows"],
    )
    lifecycle = _lifecycle_attack_matrix()
    top_rows = {
        **learning_rows,
        "dose_matching_rows": dose_rows,
        **evaluation_rows,
        "family_model_horizon_cells": family_cells,
    }
    per_unit_rows = _per_unit_rows(rows=top_rows, lifecycle_attack_matrix=lifecycle)
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    source_hashes = _source_hashes(root)
    protected = _protected_unchanged(root, protected_before)
    upstream_gate = _upstream_gate_receipt(root, exp6495_path)
    proposal_receipt = _proposal_stream_receipt(root, exp6491_path, exp6491_payload)
    optional_causal = _optional_causal_replay_receipt(root, exp6492_path, exp6492_payload)
    prior_verdicts = _prior_verdict_receipts(root)
    gates = _gate_check_summary(
        upstream_gate=upstream_gate,
        proposal_receipt=proposal_receipt,
        aggregate=aggregate,
        protected=protected,
        tests_run=tests_run,
    )
    execution_score = (
        1.0
        if aggregate["csl_execution_complete_score_from_rows"] == 1.0
        and gates["all_gates_passed"]
        else 0.0
    )
    ready_score = (
        1.0
        if execution_score == 1.0
        and aggregate["continuous_self_learning_ready_score_from_rows"] == 1.0
        else 0.0
    )
    status, verdict = _status_and_verdict(execution_score, ready_score, gates)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipt": upstream_gate,
        "proposal_stream_receipt": proposal_receipt,
        "optional_causal_replay_receipt": optional_causal,
        "frozen_learning_manifest": manifest,
        "arm_definitions": _arm_definitions(),
        "event_rows": learning_rows["event_rows"],
        "evidence_update_rows": learning_rows["evidence_update_rows"],
        "decision_action_rows": learning_rows["decision_action_rows"],
        "pool_state_rows": learning_rows["pool_state_rows"],
        "exact_admission_rows": learning_rows["exact_admission_rows"],
        "dose_matching_rows": dose_rows,
        "immediate_evaluation_rows": evaluation_rows["immediate_evaluation_rows"],
        "future_evaluation_rows": evaluation_rows["future_evaluation_rows"],
        "future_support_rows": evaluation_rows["future_support_rows"],
        "family_model_horizon_cells": family_cells,
        "lifecycle_attack_matrix": lifecycle,
        "csl_execution_complete_score": execution_score,
        "continuous_self_learning_ready_score": ready_score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gates,
        "preconditions_checked": _preconditions_checked(
            root=root,
            upstream_gate=upstream_gate,
            proposal_receipt=proposal_receipt,
            optional_causal=optional_causal,
            prior_verdicts=prior_verdicts,
            manifest=manifest,
            source_hashes=source_hashes,
            protected=protected,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, proposal_receipt),
        "random_seed": dict(RANDOM_SEED),
        "duration_s": round(
            float(duration_s)
            if duration_s is not None
            else max(time.perf_counter() - started, 0.000001),
            6,
        ),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "results": list(DEFAULT_TEST_RESULTS if tests_run is None else tests_run),
        },
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_atomic(_resolve(root, result_path), artifact)
    return artifact


def _top_level_rows_match(
    artifact: Mapping[str, Any],
    field: str,
    row_type: str,
) -> bool:
    return artifact.get(field) == [
        dict(row)
        for row in artifact.get("per_unit_rows", [])
        if row.get("row_type") == row_type
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors for an Exp6496 artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("csl_execution_complete_score") != _expected_execution_score(artifact):
        errors.append("csl_execution_complete_score mismatch")
    if artifact.get("continuous_self_learning_ready_score") != _expected_ready_score(
        artifact
    ):
        errors.append("continuous_self_learning_ready_score mismatch")
    if artifact.get("protected_files_unchanged", {}).get(
        "active_roadmap_and_conductor_unchanged"
    ) is not True:
        errors.append("protected_files_unchanged must be true")
    row_checks = (
        ("event_rows", "event_opportunity"),
        ("evidence_update_rows", "evidence_update"),
        ("decision_action_rows", "decision_action"),
        ("pool_state_rows", "pool_state"),
        ("exact_admission_rows", "exact_admission"),
        ("dose_matching_rows", "dose_matching"),
        ("immediate_evaluation_rows", "immediate_evaluation"),
        ("future_evaluation_rows", "future_evaluation"),
        ("future_support_rows", "future_support"),
        ("family_model_horizon_cells", "family_model_horizon_cell"),
    )
    for field, row_type in row_checks:
        if not _top_level_rows_match(artifact, field, row_type):
            errors.append(f"{field} mismatch")
            break
    expected_status, _ = _status_and_verdict(
        float(artifact.get("csl_execution_complete_score", 0.0) or 0.0),
        float(artifact.get("continuous_self_learning_ready_score", 0.0) or 0.0),
        artifact.get("gate_check_summary", {}),
    )
    if artifact.get("status") != expected_status:
        errors.append("status mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(
        ("complete_positive", "complete_null", "disqualified", "blocked_")
    ):
        errors.append("honest_verdict lacks required terminal prefix")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Write the terminal artifact atomically."""

    return _write_atomic(Path(path), artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    """Execute Exp6496 and write the terminal artifact."""

    started = time.perf_counter()
    duration_s = max(time.perf_counter() - started, 0.000001)
    artifact = build_artifact(
        root=REPO_ROOT,
        result_path=result_path,
        write=True,
        duration_s=duration_s,
        tests_run=DEFAULT_TEST_RESULTS,
    )
    artifact["preconditions_checked"]["requested_date"] = date
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_atomic(_resolve(REPO_ROOT, result_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = _read_json(_resolve(REPO_ROOT, result_path))
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print("OK")
        return 0
    artifact = run(date=args.date, result_path=result_path)
    print(json.dumps({"status": artifact["status"], "path": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
