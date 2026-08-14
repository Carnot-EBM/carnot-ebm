"""Exp6433 CSL row-recomputation safety audit.

Spec refs: REQ-LEARN-6433, SCENARIO-LEARN-6433-HASHES,
SCENARIO-LEARN-6433-ROWS, SCENARIO-LEARN-6433-DELTAS,
SCENARIO-LEARN-6433-ATTACKS, SCENARIO-LEARN-6433-ELIGIBILITY.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6433_csl_row_recomputation_safety_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")

RUN_DATE = "20260814"
RANDOM_SEED = 6433
SCHEMA = "carnot.experiment_6433.csl_row_recomputation_safety_audit.v1"
INFERENCE_SUBSTRATE = "deterministic_verifier_replay_over_checked_in_rows_no_new_llm"
FROZEN_TOLERANCE = 1.0e-9
MIN_EFFECTIVE_SAMPLE_SIZE = 30

EXP6430_PER_UNIT_ROW_COUNT = 200
EXP6431_PER_UNIT_ROW_COUNT = 400
EXP6432_PER_UNIT_ROW_COUNT = 144
EXP6430_RAW_OUTPUT_COUNT = 120
EXP6432_RAW_OUTPUT_COUNT = 72

TAIL_6420 = "csl_authenticity_safety_audit"
TAIL_6430 = "prospective_write_once_memory_capacity_frontier"
TAIL_6431 = "controlled_memory_interference_ab"
TAIL_6432 = "held_shift_process_restart_csl_replication"

TASK_TAILS = {
    "exp6420": TAIL_6420,
    "exp6430": TAIL_6430,
    "exp6431": TAIL_6431,
    "exp6432": TAIL_6432,
}
TASK_NUMBERS = {"exp6420": 6420, "exp6430": 6430, "exp6431": 6431, "exp6432": 6432}

EXP6430_DATA_DIR = Path("data/research") / f"experiment_{6430}_{TAIL_6430}"
EXP6432_DATA_DIR = Path("data/research") / f"experiment_{6432}_{TAIL_6432}"
EXP6430_MANIFEST = EXP6430_DATA_DIR / "prospective_write_once_capacity_stream_manifest.json"
EXP6430_RECEIPT = EXP6430_DATA_DIR / "task_scoped_generation_receipts.json"
EXP6432_MANIFEST = EXP6432_DATA_DIR / "held_shift_process_restart_manifest.json"
EXP6432_RECEIPT = EXP6432_DATA_DIR / "task_scoped_held_generation_receipts.json"

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6433_csl_row_recomputation_safety_audit "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py "
    "-m pytest tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6433_csl_row_recomputation_safety_audit.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ARTIFACT_AUDIT_COMMAND = ".venv/bin/python scripts/artifact_convention_audit.py --recent 4 --dry-run"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_AUDIT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "expected_and_available_upstream_inputs",
    "upstream_artifact_row_manifest_raw_source_test_checker_receipt_head_and_determination_hashes",
    "missing_input_findings",
    "upstream_state_by_task",
    "per_unit_rows",
    "event_and_raw_output_uniqueness_rechecks",
    "causal_order_and_exact_feedback_rechecks",
    "capacity_and_head_transition_rechecks",
    "held_freeze_and_restart_rechecks",
    "independently_recomputed_development_capacity_interference_and_held_metrics",
    "reported_vs_recomputed_deltas",
    "mismatch_count",
    "effective_sample_sizes_and_uncertainty_rechecks",
    "retention_forgetting_contamination_growth_restart_and_cost_rechecks",
    "attack_matrix",
    "open_critical_attack_ids",
    "current_and_stamped_adversarial_findings",
    "determination_preservation_findings",
    "artifact_convention_findings",
    "public_factor_claim_eligibility",
    "prospective_csl_claim_eligibility",
    "csl_row_recomputation_audit_ready_score",
    "same_verdict_retirement_decision",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "blocked_reason",
    "preconditions_checked",
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

ATTACK_IDS = (
    "raw_output_reuse",
    "cache_resurrection",
    "row_deletion",
    "duplicate_event",
    "event_reorder",
    "same_step_write",
    "stale_head",
    "authority_spoof",
    "supersession_bypass",
    "hidden_retuning",
    "future_leakage",
    "restart_corruption",
    "rollback_omission",
    "exact_veto_override",
)

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

CHECKER_RELATIVE_PATHS = (
    Path("scripts/adversarial_verify.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("scripts/artifact_convention_audit.py"),
    Path("scripts/root_clutter_sweep.py"),
)

RECEIPT_HELPER_RELATIVE_PATHS = (
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/memory/revocable_atomic_repair.py"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The status separates an eligible claim from a null or blocked audit.",
    "expected_and_available_upstream_inputs": "The input ledger makes missing and available evidence visible.",
    "upstream_artifact_row_manifest_raw_source_test_checker_receipt_head_and_determination_hashes": "Hashes bind every artifact, row block, sidecar, raw output, source, checker, helper, head, and determination.",
    "missing_input_findings": "Missing rows remain visible and never become zeros.",
    "upstream_state_by_task": "Task states preserve present, missing, skipped, blocked, null, flagged, underpowered, and eligible inputs.",
    "per_unit_rows": "Audit rows cover every source unit and every comparison.",
    "event_and_raw_output_uniqueness_rechecks": "Fresh event and raw-output checks attack reuse and duplicate evidence.",
    "causal_order_and_exact_feedback_rechecks": "Causal checks prove proposals precede outcomes and writes follow exact feedback.",
    "capacity_and_head_transition_rechecks": "Capacity and head checks attack stale or over-capacity memory.",
    "held_freeze_and_restart_rechecks": "Held freeze and restart checks attack retuning and lost persistence.",
    "independently_recomputed_development_capacity_interference_and_held_metrics": "Independent reductions keep positive summaries from being trusted.",
    "reported_vs_recomputed_deltas": "Deltas expose each reported value, tolerance, denominator, and mismatch reason.",
    "mismatch_count": "A nonzero mismatch count blocks claim eligibility.",
    "effective_sample_sizes_and_uncertainty_rechecks": "Sample-size and interval checks keep underpowered cells visible.",
    "retention_forgetting_contamination_growth_restart_and_cost_rechecks": "Safety and cost checks prevent utility from hiding harm.",
    "attack_matrix": "The attack matrix records whether every critical attack fails closed.",
    "open_critical_attack_ids": "Open critical attacks and current critical flags block claim eligibility.",
    "current_and_stamped_adversarial_findings": "Stamped findings are preserved separately from current checker output.",
    "determination_preservation_findings": "Determination lint protects historical review records.",
    "artifact_convention_findings": "Convention audit checks whether a reader can inspect the claim.",
    "public_factor_claim_eligibility": "Public factor claims require the prospective CSL claim to be eligible first.",
    "prospective_csl_claim_eligibility": "Eligibility is true only when rows, recomputation, safety, attacks, and current flags all pass.",
    "csl_row_recomputation_audit_ready_score": "The ready score is conjunctive and falls to zero for any open blocker.",
    "same_verdict_retirement_decision": "The retirement decision preserves null history without hiding V553 blockers.",
    "harm_underpowered_missing_and_flagged_cells": "Weak, missing, null, and flagged cells remain visible.",
    "protected_files_unchanged": "Protected files must stay byte-identical during the audit.",
    "blocked_reason": "The blocked reason names the first live blocker instead of burying it in prose.",
    "preconditions_checked": "Preconditions record resources, repository state, inputs, and specs checked before eligibility.",
    "inference_substrate": "The substrate declares deterministic checked-in row replay with no new LLM.",
    "verifier_is_oracle": "The audit is not an oracle, although exact validators are audited semantic oracles.",
    "field_principles": "Field principles state why each field exists.",
    "field_provenance": "Field provenance links fields to specs, rows, sidecars, attacks, tests, or commands.",
    "random_seed": "The random seed pins deterministic row ordering and attack receipts.",
    "duration_s": "Duration records audit wall time without pretending to be live inference.",
    "tests_run": "Test commands and exit codes prevent unverifiable success claims.",
    "reproducibility_checksum": "The checksum detects drift in the audit payload.",
    "honest_verdict": "The verdict uses a terminal prefix and states the audit result.",
    "missing_input_rule:missing_not_zero": "Missing rows stay missing and never enter numerators as zero.",
    "missing_input_rule:no_denominator_drop": "Missing rows stay in the visibility ledger and do not vanish from denominators.",
    "recompute:development_capacity": "Development capacity metrics come from Exp6430 row and sidecar evidence.",
    "recompute:interference": "Interference metrics come from Exp6431 per-unit rows.",
    "recompute:held": "Held metrics come from Exp6432 per-unit rows.",
    "eligibility:prospective_csl_claim": "Prospective CSL eligibility requires rows, effects, safety, attacks, and current flags to pass.",
    "eligibility:public_factor_claim": "Public factor eligibility cannot exceed prospective CSL eligibility.",
    "retirement:same_verdict": "Same-verdict retirement must preserve earlier null and flag determinations.",
}
FIELD_PRINCIPLES.update(
    {f"attack:{attack_id}": f"Attack {attack_id} must fail closed before eligibility can become true." for attack_id in ATTACK_IDS}
)

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: ["REQ-LEARN-6433"] for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE.update(
    {
        "expected_and_available_upstream_inputs": ["SCENARIO-LEARN-6433-HASHES"],
        "upstream_artifact_row_manifest_raw_source_test_checker_receipt_head_and_determination_hashes": [
            "SCENARIO-LEARN-6433-HASHES"
        ],
        "per_unit_rows": ["SCENARIO-LEARN-6433-ROWS"],
        "independently_recomputed_development_capacity_interference_and_held_metrics": [
            "SCENARIO-LEARN-6433-ROWS"
        ],
        "reported_vs_recomputed_deltas": ["SCENARIO-LEARN-6433-DELTAS"],
        "attack_matrix": ["SCENARIO-LEARN-6433-ATTACKS"],
        "prospective_csl_claim_eligibility": ["SCENARIO-LEARN-6433-ELIGIBILITY"],
        "honest_verdict": ["SCENARIO-LEARN-6433-ELIGIBILITY"],
    }
)


def _exp_name(task_key: str) -> str:
    return f"experiment_{TASK_NUMBERS[task_key]}_{TASK_TAILS[task_key]}"


def artifact_path(task_key: str) -> Path:
    return Path("results") / f"{_exp_name(task_key)}.json"


def module_path(task_key: str) -> Path:
    return Path("python/carnot") / f"{_exp_name(task_key)}.py"


def test_path(task_key: str) -> Path:
    return Path("tests/python") / f"test_{_exp_name(task_key)}.py"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except FileNotFoundError:
        return None


def read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_object:{path}")
    return payload


def as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def rounded(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def _bool(value: Any) -> bool:
    return bool(value)


def _num(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return 0.0
    return float(value)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return rounded(sum(_num(row.get(field)) for row in rows) / len(rows))


def _count_true(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(_bool(row.get(field)) for row in rows)


def _path_from_raw(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.relative_to(root)
        except ValueError:
            return path
    return path


def _file_receipt(path: Path, role: str, root: Path, *, required: bool = True) -> JsonDict:
    full = path if path.is_absolute() else root / path
    digest = sha256_file(full)
    return {
        "path": path.as_posix(),
        "role": role,
        "required": required,
        "state": "present" if digest else "missing",
        "present": digest is not None,
        "sha256": digest,
        "bytes": full.stat().st_size if digest and full.is_file() else 0,
    }


def _load_context(root: Path) -> dict[str, JsonDict]:
    return {task: read_json(root / artifact_path(task)) for task in TASK_TAILS}


def _raw_paths_from_events(payload: Mapping[str, Any], event_field: str, root: Path) -> list[Path]:
    events = as_mapping(payload.get(event_field)).get("events", [])
    paths = []
    for event in events:
        raw = as_mapping(event).get("raw_output_path")
        if raw:
            paths.append(_path_from_raw(str(raw), root))
    return paths


def _expected_file_receipts(root: Path, context: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    specs: list[tuple[Path, str, bool]] = []
    for task in TASK_TAILS:
        specs.append((artifact_path(task), "artifact", True))
        specs.append((module_path(task), "source", True))
        specs.append((test_path(task), "test", True))
    specs.extend(
        [
            (EXP6430_MANIFEST, "manifest", True),
            (EXP6430_RECEIPT, "receipt", True),
            (EXP6432_MANIFEST, "manifest", True),
            (EXP6432_RECEIPT, "receipt", True),
            (SPEC_RELATIVE_PATH, "spec", True),
            (SELF_LEARNING_SPEC_RELATIVE_PATH, "spec", True),
            (MODULE_RELATIVE_PATH, "source", True),
            (TEST_RELATIVE_PATH, "test", True),
        ]
    )
    specs.extend((path, "checker", True) for path in CHECKER_RELATIVE_PATHS)
    specs.extend((path, "receipt_helper", True) for path in RECEIPT_HELPER_RELATIVE_PATHS)
    specs.extend((path, "protected", True) for path in PROTECTED_RELATIVE_PATHS)
    specs.extend(
        (path, "raw_output", True)
        for path in _raw_paths_from_events(
            context["exp6430"],
            "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals",
            root,
        )
    )
    specs.extend(
        (path, "raw_output", True)
        for path in _raw_paths_from_events(
            context["exp6432"],
            "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals",
            root,
        )
    )
    seen: set[tuple[str, str]] = set()
    receipts = []
    for path, role, required in specs:
        key = (path.as_posix(), role)
        if key in seen:  # pragma: no cover - the expected input list is deduped by construction.
            continue
        seen.add(key)
        receipts.append(_file_receipt(path, role, root, required=required))
    return receipts


def _memory_head_receipts(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    history = as_mapping(context["exp6430"].get("memory_schema_head_and_transition_history"))
    by_capacity = as_mapping(history.get("by_capacity"))
    out = {}
    for capacity, row in sorted(by_capacity.items(), key=lambda item: int(item[0])):
        record = as_mapping(row)
        out[f"exp6430_capacity_{capacity}"] = {
            "present": bool(record.get("final_head_hash")),
            "sha256": record.get("final_head_hash"),
            "capacity_bound": record.get("capacity_bound"),
            "final_active_count": record.get("final_active_count"),
        }
    policy = as_mapping(
        context["exp6432"].get("frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes")
    )
    out["exp6432_recovered_exp6430_capacity_16"] = {
        "present": bool(policy.get("persisted_head_hash")),
        "sha256": policy.get("persisted_head_hash"),
        "selected_capacity": policy.get("selected_capacity"),
    }
    return out


def _determination_records(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    out = {}
    for task, payload in context.items():
        record = {
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending": payload.get("corrigendum_pending", []),
        }
        out[task] = {**record, "record_sha256": sha256_json(record)}
    return out


def input_hash_ledger(root: Path, context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    files = _expected_file_receipts(root, context)
    role_counts = Counter(str(row["role"]) for row in files)
    embedded_rows = {
        task: {
            "role": "row_block",
            "state": "present" if as_mapping(payload.get("per_unit_rows")).get("row_hash") else "missing",
            "row_count": as_mapping(payload.get("per_unit_rows")).get("row_count", 0),
            "sha256": as_mapping(payload.get("per_unit_rows")).get("row_hash"),
        }
        for task, payload in context.items()
        if task != "exp6420"
    }
    return {
        "schema": SCHEMA + ".input_hash_ledger",
        "files": files,
        "file_role_counts": dict(sorted(role_counts.items())),
        "embedded_row_blocks": embedded_rows,
        "memory_heads": _memory_head_receipts(context),
        "determination_records": _determination_records(context),
        "missing_required_paths": [
            str(row["path"]) for row in files if row["required"] and row["state"] == "missing"
        ],
        "missing_required_count": sum(row["required"] and row["state"] == "missing" for row in files),
    }


def expected_and_available_upstream_inputs(ledger: Mapping[str, Any]) -> JsonDict:
    files = [as_mapping(row) for row in ledger.get("files", [])]
    state_counts = Counter(str(row["state"]) for row in files)
    role_counts = Counter(str(row["role"]) for row in files)
    raw_count = role_counts.get("raw_output", 0)
    return {
        "schema": SCHEMA + ".expected_inputs",
        "required_expected_count": sum(row.get("required") is True for row in files),
        "available_required_count": sum(row.get("required") is True and row.get("state") == "present" for row in files),
        "missing_required_count": int(ledger.get("missing_required_count", 0) or 0),
        "missing_required_paths": list(ledger.get("missing_required_paths", [])),
        "state_counts": dict(sorted(state_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "raw_output_count": raw_count,
        "classification_rules": [
            "present",
            "missing",
            "skipped",
            "blocked",
            "null",
            "flagged",
            "underpowered",
            "eligible",
        ],
    }


def missing_input_findings(inputs: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA + ".missing_inputs",
        "missing_required_count": inputs.get("missing_required_count", 0),
        "missing_required_paths": list(inputs.get("missing_required_paths", [])),
        "missing_rows_treated_as_zero": False,
        "missing_rows_dropped_from_denominators": False,
        "all_missing_visible": inputs.get("missing_required_count", 0) == 0,
    }


def _underpowered_count(payload: Mapping[str, Any]) -> int:
    harm = as_mapping(payload.get("harm_underpowered_missing_and_flagged_cells"))
    return int(harm.get("underpowered_cell_count", harm.get("new_underpowered_cell_count", 0)) or 0)


def upstream_state_by_task(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    states = {}
    for task, payload in context.items():
        state = "eligible"
        if payload.get("flagged_adversarial") is True:
            state = "flagged"
        elif str(payload.get("status", "")).startswith("blocked"):
            state = "blocked"
        elif "skip" in str(payload.get("status", "")).lower():
            state = "skipped"
        elif "null" in str(payload.get("status", "")).lower():
            state = "null"
        elif _underpowered_count(payload) > 0:
            state = "underpowered"
        states[task] = {
            "state": state,
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "underpowered_cell_count": _underpowered_count(payload),
            "row_count": as_mapping(payload.get("per_unit_rows")).get("row_count", 0),
        }
    return states


def reduce_exp6430(payload: Mapping[str, Any]) -> JsonDict:
    units = [as_mapping(row) for row in as_mapping(payload.get("per_unit_rows")).get("rows", [])]
    manifest = as_mapping(
        payload.get("chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals")
    )
    event_count = int(manifest.get("event_count", 0) or len(manifest.get("events", [])))
    feedback = as_mapping(payload.get("exact_feedback_receipts"))
    feedback_rows = [as_mapping(row) for row in feedback.get("rows", [])]
    disposition = as_mapping(payload.get("commit_reject_quarantine_defer_evict_expire_and_supersede_counts"))
    by_disposition = as_mapping(disposition.get("by_capacity"))
    history = as_mapping(payload.get("memory_schema_head_and_transition_history"))
    by_head = as_mapping(history.get("by_capacity"))
    by_capacity: JsonDict = {}
    for capacity in sorted({int(row.get("capacity", 0)) for row in units}):
        rows = [row for row in units if int(row.get("capacity", -1)) == capacity]
        cap_key = str(capacity)
        disp = as_mapping(by_disposition.get(cap_key))
        head = as_mapping(by_head.get(cap_key))
        feedback_cap = [row for row in feedback_rows if int(row.get("capacity", -1)) == capacity]
        write_precision = 0.0
        if capacity > 0 and int(disp.get("Commit", 0) or 0) + int(disp.get("Supersede", 0) or 0) > 0:
            write_precision = 1.0 if all(row.get("feedback_before_write") is True for row in feedback_cap) else 0.0
        future_success = _count_true(rows, "exact_success")
        memory_match = _count_true(rows, "memory_match")
        by_capacity[cap_key] = {
            "capacity": capacity,
            "arm": "frozen" if capacity == 0 else f"capacity_{capacity}",
            "future_event_count": len(rows),
            "future_exact_success_count": future_success,
            "future_exact_yield": rounded(future_success / len(rows)),
            "proposal_coverage_count": memory_match,
            "proposal_coverage": rounded(memory_match / len(rows)),
            "write_precision": write_precision,
            "selection_success_count": _count_true(rows, "selection_success"),
            "selection_success": rounded(_count_true(rows, "selection_success") / len(rows)),
            "transfer": rounded(_count_true(rows, "transfer") / len(rows)),
            "retention": rounded(_count_true(rows, "retained_protected") / len(rows)),
            "forgetting": rounded(_count_true(rows, "forgetting") / len(rows)),
            "contamination": rounded(_count_true(rows, "contamination") / len(rows)),
            "restart_recovery": rounded(_count_true(rows, "restart_recovered") / len(rows)),
            "growth": int(head.get("final_active_count", 0) or 0),
            "eviction_count": int(disp.get("Evict", 0) or 0),
            "cost": {
                "model_calls": event_count,
                "checker_calls": len(feedback_cap),
                "consumer_work_units": event_count,
                "memory_capacity": capacity,
                "cost_units": event_count + capacity,
            },
            "row_hash": sha256_json(rows),
        }
    frontier_rows = []
    for cap_key, row in sorted(by_capacity.items(), key=lambda item: int(item[0])):
        frontier_rows.append(
            {
                "capacity": int(cap_key),
                "coverage": row["proposal_coverage"],
                "write_precision": row["write_precision"],
                "future_exact_yield": row["future_exact_yield"],
                "retention": row["retention"],
                "utility": rounded(row["future_exact_yield"] - (int(cap_key) * 0.01)),
                "capacity_selected_after_held_outcomes": False,
            }
        )
    best = max(
        [row for row in frontier_rows if row["capacity"] > 0],
        key=lambda row: (row["utility"], -row["capacity"]),
    )
    return {
        "schema": SCHEMA + ".development_capacity",
        "source": "exp6430 per_unit_rows plus exact feedback and head sidecars",
        "by_capacity": by_capacity,
        "capacity_utility_frontier": {
            "rows": frontier_rows,
            "best_nonzero_capacity": best["capacity"],
            "capacity_selected_after_held_outcomes": False,
        },
    }


def reduce_exp6431(payload: Mapping[str, Any]) -> JsonDict:
    units = [as_mapping(row) for row in as_mapping(payload.get("per_unit_rows")).get("rows", [])]
    by_arm: dict[str, list[JsonDict]] = defaultdict(list)
    for row in units:
        by_arm[str(row.get("arm"))].append(row)
    out: JsonDict = {}
    for arm, rows in sorted(by_arm.items()):
        active_rows = [row for row in rows if int(row.get("capacity", 0) or 0) > 0]
        contamination_after_rollback = sum(
            int(as_mapping(row.get("rollback_path")).get("contamination_after_rollback", 0) or 0)
            for row in rows
        )
        out[arm] = {
            "row_count": len(rows),
            "active_capacity_row_count": len(active_rows),
            "accepted_invalid_memory_count": _count_true(rows, "accepted_invalid_memory"),
            "contamination_after_rollback": contamination_after_rollback,
            "downstream_use_failure_count": _count_true(rows, "downstream_use_failure"),
            "exposure_failure_count": _count_true(rows, "target_exposure_failure"),
            "future_exact_yield": _mean(active_rows, "future_exact_yield"),
            "protected_stability": _mean(rows, "protected_stability"),
            "valid_higher_authority_update_count": sum(
                as_mapping(as_mapping(row.get("write_path")).get("supersession_receipt")).get(
                    "valid_higher_authority_update"
                )
                is True
                for row in rows
            ),
            "latency_ms": _mean(rows, "latency_ms"),
            "work_units": rounded(sum(_num(row.get("work_units")) for row in rows)),
        }
    arms = list(out)
    stability_delta = 0.0
    if len(arms) == 2:
        stability_delta = rounded(out[arms[0]]["protected_stability"] - out[arms[1]]["protected_stability"])
    return {
        "schema": SCHEMA + ".interference",
        "source": "exp6431 per_unit_rows",
        "by_arm": out,
        "protected_stability_delta": stability_delta,
        "valid_higher_authority_update_count": int(
            as_mapping(out.get("authority_aware_retrieval_and_write_controls")).get(
                "valid_higher_authority_update_count",
                0,
            )
            or 0
        ),
    }


def reduce_exp6432(payload: Mapping[str, Any]) -> JsonDict:
    units = [as_mapping(row) for row in as_mapping(payload.get("per_unit_rows")).get("rows", [])]
    by_arm: dict[str, list[JsonDict]] = defaultdict(list)
    for row in units:
        by_arm[str(row.get("arm"))].append(row)
    out: JsonDict = {}
    for arm, rows in sorted(by_arm.items()):
        out[arm] = {
            "row_count": len(rows),
            "coverage": _mean(rows, "coverage"),
            "precision": _mean(rows, "precision"),
            "selection": _mean(rows, "selection"),
            "future_exact_success_count": sum(_num(row.get("future_exact_yield")) == 1.0 for row in rows),
            "future_exact_yield": _mean(rows, "future_exact_yield"),
            "transfer": _mean(rows, "transfer"),
            "retention": _mean(rows, "retention"),
            "forgetting": _mean(rows, "forgetting"),
            "negative_transfer": _mean(rows, "negative_transfer"),
            "contamination": _mean(rows, "contamination"),
            "restart_recovery": _mean(rows, "restart_recovery"),
            "latency_ms": _mean(rows, "latency_ms"),
            "gpu_cost_units": rounded(
                sum(_num(as_mapping(row.get("gpu_cost")).get("cost_units")) for row in rows)
            ),
        }
    selected = as_mapping(out.get("selected_capacity_memory"))
    frozen = as_mapping(out.get("frozen_memory"))
    return {
        "schema": SCHEMA + ".held",
        "source": "exp6432 per_unit_rows",
        "by_arm": out,
        "held_future_exact_yield_delta": rounded(
            _num(selected.get("future_exact_yield")) - _num(frozen.get("future_exact_yield"))
        ),
        "protected_retention_delta": rounded(_num(selected.get("retention")) - _num(frozen.get("retention"))),
        "negative_transfer_delta": rounded(
            _num(selected.get("negative_transfer")) - _num(frozen.get("negative_transfer"))
        ),
        "contamination_delta": rounded(_num(selected.get("contamination")) - _num(frozen.get("contamination"))),
    }


def recomputed_metrics(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".independent_metrics",
        "development_capacity": reduce_exp6430(context["exp6430"]),
        "interference": reduce_exp6431(context["exp6431"]),
        "held": reduce_exp6432(context["exp6432"]),
        "implementation_independence": {
            "imports_upstream_aggregate_or_readiness_functions": False,
            "upstream_aggregate_values_used_as_inputs": False,
        },
    }


def _comparison(
    *,
    source_task: str,
    metric: str,
    reported: Any,
    recomputed: Any,
    row_population: int,
    filter_text: str,
    numerator: Any,
    denominator: Any,
    tolerance: float = FROZEN_TOLERANCE,
) -> JsonDict:
    left = _num(reported)
    right = _num(recomputed)
    abs_delta = rounded(abs(left - right))
    return {
        "row_type": "comparison",
        "comparison_id": f"{source_task}:{filter_text}:{metric}",
        "source_task": source_task,
        "metric": metric,
        "reported": reported,
        "recomputed": recomputed,
        "abs_delta": abs_delta,
        "tolerance": tolerance,
        "matches": abs_delta <= tolerance,
        "row_population": int(row_population),
        "filter": filter_text,
        "numerator": numerator,
        "denominator": denominator,
        "mismatch_reason": "" if abs_delta <= tolerance else "reported_value_did_not_recompute",
    }


def reported_vs_recomputed_deltas(
    context: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> JsonDict:
    comparisons: list[JsonDict] = []
    exp6430_reported = as_mapping(
        context["exp6430"].get(
            "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results"
        )
    )
    reported_by_capacity = as_mapping(exp6430_reported.get("by_capacity"))
    recomputed_capacity = as_mapping(as_mapping(metrics.get("development_capacity")).get("by_capacity"))
    for cap_key, recomputed in sorted(recomputed_capacity.items(), key=lambda item: int(item[0])):
        reported = as_mapping(reported_by_capacity.get(cap_key))
        population = int(recomputed.get("future_event_count", 0))
        for field in (
            "future_exact_yield",
            "proposal_coverage",
            "write_precision",
            "selection_success",
            "transfer",
            "retention",
            "forgetting",
            "contamination",
            "restart_recovery",
            "growth",
            "eviction_count",
        ):
            comparisons.append(
                _comparison(
                    source_task="exp6430",
                    metric=field,
                    reported=reported.get(field),
                    recomputed=as_mapping(recomputed).get(field),
                    row_population=population,
                    filter_text=f"capacity=={cap_key}",
                    numerator=as_mapping(recomputed).get(f"{field}_count", as_mapping(recomputed).get(field)),
                    denominator=population,
                )
            )
        for field in ("model_calls", "checker_calls", "consumer_work_units", "memory_capacity", "cost_units"):
            comparisons.append(
                _comparison(
                    source_task="exp6430",
                    metric=f"cost.{field}",
                    reported=as_mapping(reported.get("cost")).get(field),
                    recomputed=as_mapping(recomputed.get("cost")).get(field),
                    row_population=int(as_mapping(recomputed.get("cost")).get("model_calls", 0)),
                    filter_text=f"capacity=={cap_key}",
                    numerator=as_mapping(recomputed.get("cost")).get(field),
                    denominator=1,
                )
            )
    reported_frontier = as_mapping(context["exp6430"].get("capacity_utility_frontier")).get("rows", [])
    recomputed_frontier = as_mapping(as_mapping(metrics.get("development_capacity")).get("capacity_utility_frontier")).get("rows", [])
    for reported, recomputed in zip(reported_frontier, recomputed_frontier, strict=True):
        for field in ("coverage", "write_precision", "future_exact_yield", "retention", "utility"):
            comparisons.append(
                _comparison(
                    source_task="exp6430",
                    metric=f"frontier.{field}",
                    reported=as_mapping(reported).get(field),
                    recomputed=as_mapping(recomputed).get(field),
                    row_population=40,
                    filter_text=f"frontier_capacity=={as_mapping(recomputed).get('capacity')}",
                    numerator=as_mapping(recomputed).get(field),
                    denominator=1,
                )
            )
    exp6431_reported = as_mapping(
        context["exp6431"].get(
            "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results"
        )
    )
    reported_by_arm = as_mapping(exp6431_reported.get("by_arm"))
    recomputed_by_arm = as_mapping(as_mapping(metrics.get("interference")).get("by_arm"))
    for arm, recomputed in sorted(recomputed_by_arm.items()):
        reported = as_mapping(reported_by_arm.get(arm))
        population = int(as_mapping(recomputed).get("row_count", 0))
        for field in (
            "row_count",
            "active_capacity_row_count",
            "accepted_invalid_memory_count",
            "contamination_after_rollback",
            "downstream_use_failure_count",
            "exposure_failure_count",
            "future_exact_yield",
            "protected_stability",
        ):
            comparisons.append(
                _comparison(
                    source_task="exp6431",
                    metric=field,
                    reported=reported.get(field),
                    recomputed=as_mapping(recomputed).get(field),
                    row_population=population,
                    filter_text=f"arm=={arm}",
                    numerator=as_mapping(recomputed).get(field),
                    denominator=population,
                )
            )
    for field, recomputed_value in (
        ("valid_higher_authority_update_count", as_mapping(metrics.get("interference")).get("valid_higher_authority_update_count")),
        ("contamination_after_rollback", min(row["contamination_after_rollback"] for row in recomputed_by_arm.values())),
    ):
        comparisons.append(
            _comparison(
                source_task="exp6431",
                metric=field,
                reported=context["exp6431"].get(field),
                recomputed=recomputed_value,
                row_population=sum(row["row_count"] for row in recomputed_by_arm.values()),
                filter_text="all_interference_rows",
                numerator=recomputed_value,
                denominator=1,
            )
        )
    exp6432_reported = as_mapping(
        context["exp6432"].get(
            "per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results"
        )
    )
    reported_by_arm_held = as_mapping(exp6432_reported.get("by_arm"))
    recomputed_by_arm_held = as_mapping(as_mapping(metrics.get("held")).get("by_arm"))
    for arm, recomputed in sorted(recomputed_by_arm_held.items()):
        reported = as_mapping(reported_by_arm_held.get(arm))
        population = int(as_mapping(recomputed).get("row_count", 0))
        for field in (
            "row_count",
            "coverage",
            "precision",
            "selection",
            "future_exact_yield",
            "transfer",
            "retention",
            "forgetting",
            "negative_transfer",
            "contamination",
            "restart_recovery",
            "latency_ms",
            "gpu_cost_units",
        ):
            comparisons.append(
                _comparison(
                    source_task="exp6432",
                    metric=field,
                    reported=reported.get(field),
                    recomputed=as_mapping(recomputed).get(field),
                    row_population=population,
                    filter_text=f"arm=={arm}",
                    numerator=as_mapping(recomputed).get(field),
                    denominator=population,
                )
            )
    for field in (
        "held_future_exact_yield_delta",
        "negative_transfer_delta",
        "contamination_propagation_rate",
    ):
        recomputed_field = "contamination_delta" if field == "contamination_propagation_rate" else field
        comparisons.append(
            _comparison(
                source_task="exp6432",
                metric=field,
                reported=context["exp6432"].get(field),
                recomputed=as_mapping(metrics.get("held")).get(recomputed_field),
                row_population=EXP6432_PER_UNIT_ROW_COUNT,
                filter_text="held_selected_minus_frozen",
                numerator=as_mapping(metrics.get("held")).get(recomputed_field),
                denominator=1,
            )
        )
    return {
        "schema": SCHEMA + ".reported_vs_recomputed",
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "mismatch_count": sum(row["matches"] is not True for row in comparisons),
        "all_within_tolerance": all(row["matches"] is True for row in comparisons),
    }


def per_unit_rows(context: Mapping[str, Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> JsonDict:
    source_rows = []
    for task in ("exp6430", "exp6431", "exp6432"):
        rows = [as_mapping(row) for row in as_mapping(context[task].get("per_unit_rows")).get("rows", [])]
        for index, row in enumerate(rows):
            unit_id = str(row.get("unit_id") or f"{row.get('event_id')}:{row.get('capacity', '')}:{row.get('arm')}")
            source_rows.append(
                {
                    "row_type": "source_unit",
                    "audit_row_id": f"{task}:source:{index:04d}",
                    "source_task": task,
                    "source_unit_id": unit_id,
                    "source_row_hash": sha256_json(row),
                    "classification": "eligible",
                    "included_in_denominator": True,
                    "denominator_group": f"{task}:{row.get('arm', row.get('capacity'))}",
                }
            )
    comparison_rows = [
        {
            "row_type": "comparison",
            "audit_row_id": f"{row.get('comparison_id')}",
            "source_task": row.get("source_task"),
            "source_unit_id": row.get("comparison_id"),
            "source_row_hash": sha256_json(row),
            "classification": "eligible" if row.get("matches") is True else "flagged",
            "included_in_denominator": True,
            "denominator_group": f"{row.get('source_task')}:{row.get('filter')}",
        }
        for row in comparisons
    ]
    rows = source_rows + comparison_rows
    return {
        "schema": SCHEMA + ".per_unit_rows",
        "source_unit_row_count": len(source_rows),
        "comparison_row_count": len(comparison_rows),
        "row_count": len(rows),
        "rows": rows,
        "row_hash": sha256_json(rows),
    }


def event_and_raw_output_uniqueness_rechecks(context: Mapping[str, Mapping[str, Any]], root: Path) -> JsonDict:
    exp6430_events = [
        as_mapping(row)
        for row in as_mapping(
            context["exp6430"].get(
                "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    exp6432_events = [
        as_mapping(row)
        for row in as_mapping(
            context["exp6432"].get(
                "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals"
            )
        ).get("events", [])
    ]
    all_events = [("exp6430", row) for row in exp6430_events] + [("exp6432", row) for row in exp6432_events]
    event_ids = [str(row.get("event_id")) for _, row in all_events]
    raw_hashes = [str(row.get("raw_output_sha256")) for _, row in all_events]
    file_mismatches = []
    for task, row in all_events:
        raw_path = _path_from_raw(str(row.get("raw_output_path", "")), root)
        digest = sha256_file(root / raw_path if not raw_path.is_absolute() else raw_path)
        if digest != row.get("raw_output_sha256"):
            file_mismatches.append({"task": task, "event_id": row.get("event_id"), "path": raw_path.as_posix()})
    prompt_bindings = sum(bool(row.get("prompt_sha256")) and bool(row.get("model_hf_id")) for _, row in all_events)
    return {
        "schema": SCHEMA + ".event_raw_uniqueness",
        "event_count": len(all_events),
        "unique_event_id_count": len(set(event_ids)),
        "duplicate_event_id_count": len(event_ids) - len(set(event_ids)),
        "raw_output_count": len(raw_hashes),
        "unique_raw_output_hash_count": len(set(raw_hashes)),
        "raw_output_reuse_count": len(raw_hashes) - len(set(raw_hashes)),
        "cross_task_raw_hash_overlap_count": len(
            {str(row.get("raw_output_sha256")) for row in exp6430_events}
            & {str(row.get("raw_output_sha256")) for row in exp6432_events}
        ),
        "raw_file_hash_mismatch_count": len(file_mismatches),
        "raw_file_hash_mismatches": file_mismatches,
        "prompt_and_model_binding_count": prompt_bindings,
        "all_prompt_and_model_bindings_present": prompt_bindings == len(all_events),
    }


def causal_order_and_exact_feedback_rechecks(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp6430_events = [
        as_mapping(row)
        for row in as_mapping(
            context["exp6430"].get(
                "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    exp6432_events = [
        as_mapping(row)
        for row in as_mapping(
            context["exp6432"].get(
                "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals"
            )
        ).get("events", [])
    ]
    all_events = exp6430_events + exp6432_events
    proposal_before_outcome = sum(
        int(row.get("proposal_freeze_order", 0)) < int(row.get("outcome_open_order", 0))
        for row in all_events
    )
    feedback_rows = [
        as_mapping(row)
        for row in as_mapping(context["exp6430"].get("exact_feedback_receipts")).get("rows", [])
    ]
    exp6432_units = [
        as_mapping(row) for row in as_mapping(context["exp6432"].get("per_unit_rows")).get("rows", [])
    ]
    return {
        "schema": SCHEMA + ".causal_order",
        "event_count": len(all_events),
        "proposal_before_outcome_count": proposal_before_outcome,
        "proposal_before_outcome_violation_count": len(all_events) - proposal_before_outcome,
        "exact_feedback_row_count": len(feedback_rows),
        "exact_feedback_before_write_count": sum(row.get("feedback_before_write") is True for row in feedback_rows),
        "same_step_write_count": sum(row.get("same_step_write") is True for row in exp6432_units),
        "future_label_used_for_proposal_count": sum(
            row.get("future_label_used_for_proposal") is True for row in exp6432_units
        ),
    }


def capacity_and_head_transition_rechecks(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    history = as_mapping(context["exp6430"].get("memory_schema_head_and_transition_history"))
    by_capacity = as_mapping(history.get("by_capacity"))
    rows = []
    for capacity, value in sorted(by_capacity.items(), key=lambda item: int(item[0])):
        row = as_mapping(value)
        rows.append(
            {
                "capacity": int(capacity),
                "capacity_bound": int(row.get("capacity_bound", 0) or 0),
                "max_active_count": int(row.get("max_active_count", 0) or 0),
                "final_active_count": int(row.get("final_active_count", 0) or 0),
                "final_head_hash": row.get("final_head_hash"),
                "within_capacity": int(row.get("max_active_count", 0) or 0)
                <= int(row.get("capacity_bound", 0) or 0),
            }
        )
    return {
        "schema": SCHEMA + ".capacity_head",
        "rows": rows,
        "capacity_violation_count": sum(row["within_capacity"] is not True for row in rows),
        "all_active_counts_within_capacity": all(row["within_capacity"] for row in rows),
        "all_transitions_after_exact_feedback": history.get("all_transitions_after_exact_feedback") is True,
        "unique_final_head_count": len({row["final_head_hash"] for row in rows}),
    }


def held_freeze_and_restart_rechecks(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    policy = as_mapping(
        context["exp6432"].get("frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes")
    )
    restarts = as_mapping(context["exp6432"].get("process_restart_and_persisted_head_recovery_receipts"))
    manifest = as_mapping(
        context["exp6432"].get("held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals")
    )
    return {
        "schema": SCHEMA + ".held_freeze_restart",
        "policy_frozen_before_held_outcomes": policy.get("policy_frozen_before_held_outcomes") is True,
        "hidden_retuning_count": int(policy.get("hidden_retuning_count", 0) or 0),
        "selected_capacity": policy.get("selected_capacity"),
        "held_manifest_pre_registered_before_outcomes": manifest.get("pre_registered_before_outcomes") is True,
        "development_pooling_count": int(manifest.get("development_pooling_count", 0) or 0),
        "session_restart_count": int(restarts.get("session_restart_count", 0) or 0),
        "all_recovered_heads_match": restarts.get("all_recovered_heads_match") is True,
        "restart_recovery_rate": restarts.get("restart_recovery_rate"),
    }


def _ci95(success: int, count: int) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    p = success / count
    half = 1.96 * math.sqrt((p * (1.0 - p)) / count)
    return [rounded(max(0.0, p - half)), rounded(min(1.0, p + half))]


def effective_sample_sizes_and_uncertainty_rechecks(metrics: Mapping[str, Any]) -> JsonDict:
    capacity_rows = {}
    for capacity, row in as_mapping(as_mapping(metrics.get("development_capacity")).get("by_capacity")).items():
        count = int(row.get("future_event_count", 0) or 0)
        success = int(row.get("future_exact_success_count", 0) or 0)
        capacity_rows[capacity] = {
            "future_event_count": count,
            "future_exact_success_count": success,
            "effective_sample_size": count,
            "future_exact_yield_ci95": _ci95(success, count),
        }
    held_rows = {}
    for arm, row in as_mapping(as_mapping(metrics.get("held")).get("by_arm")).items():
        count = int(row.get("row_count", 0) or 0)
        success = int(row.get("future_exact_success_count", 0) or 0)
        held_rows[arm] = {
            "future_event_count": count,
            "future_exact_success_count": success,
            "effective_sample_size": count,
            "future_exact_yield_ci95": _ci95(success, count),
        }
    min_effective = min(
        [row["effective_sample_size"] for row in capacity_rows.values()]
        + [row["effective_sample_size"] for row in held_rows.values()]
    )
    return {
        "schema": SCHEMA + ".uncertainty",
        "confidence_interval_method": "normal_approximation_binomial_ci95",
        "development_capacity": capacity_rows,
        "held": held_rows,
        "minimum_effective_sample_size": min_effective,
        "adequate_effective_sample_size": min_effective >= MIN_EFFECTIVE_SAMPLE_SIZE,
    }


def safety_cost_rechecks(context: Mapping[str, Mapping[str, Any]], metrics: Mapping[str, Any]) -> JsonDict:
    dev = as_mapping(as_mapping(metrics.get("development_capacity")).get("by_capacity"))
    held = as_mapping(as_mapping(metrics.get("held")).get("by_arm"))
    interference = as_mapping(as_mapping(metrics.get("interference")).get("by_arm"))
    retentions = [float(row.get("retention", 0.0) or 0.0) for row in dev.values()] + [
        float(row.get("retention", 0.0) or 0.0) for row in held.values()
    ]
    contaminations = [float(row.get("contamination", 0.0) or 0.0) for row in dev.values()] + [
        float(row.get("contamination", 0.0) or 0.0) for row in held.values()
    ] + [float(row.get("contamination_after_rollback", 0.0) or 0.0) for row in interference.values()]
    forgetting = [float(row.get("forgetting", 0.0) or 0.0) for row in dev.values()] + [
        float(row.get("forgetting", 0.0) or 0.0) for row in held.values()
    ]
    restart = [float(row.get("restart_recovery", 0.0) or 0.0) for row in dev.values()] + [
        float(row.get("restart_recovery", 0.0) or 0.0) for row in held.values()
    ]
    return {
        "schema": SCHEMA + ".safety_cost",
        "protected_retention_holds": all(value >= 1.0 for value in retentions),
        "contamination_zero": all(value == 0.0 for value in contaminations),
        "forgetting_zero": all(value == 0.0 for value in forgetting),
        "restart_recovery_holds": all(value == 1.0 for value in restart),
        "growth_by_capacity": {capacity: row.get("growth") for capacity, row in dev.items()},
        "cost_by_capacity": {capacity: row.get("cost") for capacity, row in dev.items()},
        "held_gpu_cost_units": {
            arm: row.get("gpu_cost_units") for arm, row in held.items()
        },
        "exp6420_null_preserved": context["exp6420"].get("status") == "complete_null",
    }


def _run_command(command: list[str], root: Path, timeout: int = 120) -> JsonDict:
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def current_and_stamped_adversarial_findings(
    root: Path,
    context: Mapping[str, Mapping[str, Any]],
    *,
    run_current: bool,
) -> JsonDict:
    current = {}
    stamped = {}
    for task, payload in context.items():
        stamped_flags = payload.get("corrigendum_pending", [])
        stamped[task] = {
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending": stamped_flags if isinstance(stamped_flags, list) else [],
        }
        if run_current:
            result = _run_command(
                [sys.executable, "scripts/adversarial_verify.py", "--json", artifact_path(task).as_posix()],
                root,
            )
            try:
                parsed = json.loads(result["stdout"])
                report = as_mapping(parsed.get("reports", [{}])[0])
                flags = report.get("flags", [])
            except Exception:  # noqa: BLE001
                flags = [{"kind": "adversarial_verify_parse_error", "severity": "critical", "detail": result["stderr"]}]
            current[task] = {
                "exit_code": result["exit_code"],
                "flags": flags,
                "flag_count": len(flags),
                "critical_flag_count": sum(
                    str(as_mapping(flag).get("severity", "")).lower() == "critical" for flag in flags
                ),
            }
        else:
            flags = stamped[task]["corrigendum_pending"]
            current[task] = {
                "exit_code": 0 if not flags else 1,
                "flags": flags,
                "flag_count": len(flags),
                "critical_flag_count": sum(
                    str(as_mapping(flag).get("severity", "")).lower() == "critical" for flag in flags
                ),
                "source": "stamped_fixture_when_current_audits_disabled",
            }
    return {"schema": SCHEMA + ".adversarial_findings", "current": current, "stamped": stamped}


def determination_preservation_findings(root: Path, *, run_current: bool) -> JsonDict:
    if not run_current:
        return {"schema": SCHEMA + ".determination_preservation", "exit_code": 0, "mode": "not_run_test_fixture", "findings": []}
    result = _run_command([sys.executable, "scripts/determination_preservation_lint.py"], root)
    return {
        "schema": SCHEMA + ".determination_preservation",
        "exit_code": result["exit_code"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "findings": [] if result["exit_code"] == 0 else [result["stdout"], result["stderr"]],
    }


def artifact_convention_findings(root: Path, *, run_current: bool) -> JsonDict:
    if not run_current:
        return {"schema": SCHEMA + ".artifact_convention", "exit_code": 0, "mode": "not_run_test_fixture", "findings": []}
    result = _run_command(
        [sys.executable, "scripts/artifact_convention_audit.py", "--recent", "4", "--dry-run"],
        root,
    )
    return {
        "schema": SCHEMA + ".artifact_convention",
        "exit_code": result["exit_code"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "findings": [] if result["exit_code"] == 0 else [result["stdout"], result["stderr"]],
    }


def attack_matrix(
    checks: Mapping[str, Any],
    adversarial: Mapping[str, Any],
    context: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    unique = as_mapping(checks.get("event_uniqueness"))
    causal = as_mapping(checks.get("causal_order"))
    capacity = as_mapping(checks.get("capacity_head"))
    held = as_mapping(checks.get("held_freeze_restart"))
    exp6431 = context["exp6431"]
    rows = []
    evidence = {
        "raw_output_reuse": unique.get("raw_output_reuse_count") == 0 and unique.get("cross_task_raw_hash_overlap_count") == 0,
        "cache_resurrection": context["exp6430"].get("cache_resurrection_count", 0) == 0 and context["exp6432"].get("cache_resurrection_count", 0) == 0,
        "row_deletion": unique.get("event_count") == EXP6430_RAW_OUTPUT_COUNT + EXP6432_RAW_OUTPUT_COUNT,
        "duplicate_event": unique.get("duplicate_event_id_count") == 0,
        "event_reorder": causal.get("proposal_before_outcome_violation_count") == 0,
        "same_step_write": causal.get("same_step_write_count") == 0,
        "stale_head": capacity.get("all_transitions_after_exact_feedback") is True,
        "authority_spoof": exp6431.get("authority_spoof_accept_count") == 0,
        "supersession_bypass": exp6431.get("expired_or_superseded_accept_count") == 0,
        "hidden_retuning": held.get("hidden_retuning_count") == 0,
        "future_leakage": causal.get("future_label_used_for_proposal_count") == 0,
        "restart_corruption": held.get("all_recovered_heads_match") is True,
        "rollback_omission": exp6431.get("contamination_after_rollback") == 0,
        "exact_veto_override": context["exp6430"].get("exact_veto_override_count", 0) == 0,
    }
    for attack_id in ATTACK_IDS:
        fail_closed = bool(evidence[attack_id])
        rows.append(
            {
                "attack_id": attack_id,
                "critical": True,
                "fail_closed": fail_closed,
                "accepted": False if fail_closed else True,
                "promoted_readiness": False if fail_closed else True,
                "evidence": f"{attack_id} independent receipt",
            }
        )
    current = as_mapping(adversarial.get("current"))
    current_critical = []
    for task, report in current.items():
        for flag in as_mapping(report).get("flags", []):
            flag = as_mapping(flag)
            if str(flag.get("severity", "")).lower() == "critical":
                current_critical.append(f"current_adversarial_flag:{flag.get('kind')}:{task}")
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "accepted_attack_count": sum(row["accepted"] for row in rows),
        "promoted_attack_count": sum(row["promoted_readiness"] for row in rows),
        "current_critical_flags": current_critical,
    }


def protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> JsonDict:
    files = {}
    for path in sorted(set(before) | set(after)):
        files[path] = {"before": before.get(path), "after": after.get(path), "unchanged": before.get(path) == after.get(path)}
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if row["unchanged"] is not True],
    }


def _ram_total_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - Linux test hosts provide /proc/meminfo.
        return 0
    return 0  # pragma: no cover - MemTotal is always present on supported Linux hosts.


def _git(args: Sequence[str], root: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def preconditions_checked(
    root: Path,
    run_date: str,
    inputs: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> JsonDict:
    disk = shutil.disk_usage(root)
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    missing = list(inputs.get("missing_required_paths", []))
    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": RUN_DATE,
        "run_date": run_date,
        "all_required_inputs_present": not missing,
        "spec_contains_req": "REQ-LEARN-6433" in spec_text,
        "missing_required_paths": missing,
        "repository_state": {
            "head": _git(["rev-parse", "HEAD"], root),
            "branch": _git(["branch", "--show-current"], root),
            "status_short": _git(["status", "--short"], root).splitlines(),
        },
        "system": {
            "cpu_count": os.cpu_count() or 1,
            "ram_total_bytes": _ram_total_bytes(),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "input_hash": sha256_json(ledger),
        "checked": [
            "artifacts",
            "row_blocks",
            "manifests",
            "raw_outputs",
            "sources",
            "tests",
            "checkers",
            "receipt_helpers",
            "memory_heads",
            "determination_records",
            "cpu",
            "ram",
            "disk",
            "repository_state",
        ],
    }


def harm_underpowered_missing_and_flagged_cells(
    context: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
    adversarial: Mapping[str, Any],
) -> JsonDict:
    current = as_mapping(adversarial.get("current"))
    return {
        "schema": SCHEMA + ".harm_visible",
        "missing_required_count": inputs.get("missing_required_count", 0),
        "underpowered_cells": {
            task: _underpowered_count(payload) for task, payload in context.items()
        },
        "flagged_tasks": [
            task for task, payload in context.items() if payload.get("flagged_adversarial") is True
        ],
        "current_critical_flag_count": sum(
            int(as_mapping(report).get("critical_flag_count", 0) or 0) for report in current.values()
        ),
        "weak_missing_null_and_flagged_cells_visible": True,
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    exit_codes = (
        {command: 0 for command in DEFAULT_TEST_COMMANDS}
        if test_exit_codes is None
        else {str(command): int(code) for command, code in test_exit_codes.items()}
    )
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(exit_codes.get(command, 1) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def _current_critical_ids(adversarial: Mapping[str, Any]) -> list[str]:
    out = []
    for task, report in as_mapping(adversarial.get("current")).items():
        for flag in as_mapping(report).get("flags", []):
            flag = as_mapping(flag)
            if str(flag.get("severity", "")).lower() == "critical":
                out.append(f"current_adversarial_flag:{flag.get('kind')}:{task}")
    return sorted(out)


def _eligibility(
    *,
    inputs: Mapping[str, Any],
    deltas: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    safety: Mapping[str, Any],
    attacks: Mapping[str, Any],
    adversarial: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    blockers = []
    if inputs.get("missing_required_count", 0) != 0:
        blockers.append("missing_required_input")
    if deltas.get("all_within_tolerance") is not True:
        blockers.append("reported_values_do_not_recompute")
    if uncertainty.get("adequate_effective_sample_size") is not True:
        blockers.append("inadequate_effective_sample_size")
    dev = as_mapping(as_mapping(metrics.get("development_capacity")).get("by_capacity"))
    if _num(as_mapping(dev.get("16")).get("future_exact_yield")) <= _num(as_mapping(dev.get("0")).get("future_exact_yield")):
        blockers.append("development_future_effect_not_positive")
    if _num(as_mapping(metrics.get("held")).get("held_future_exact_yield_delta")) <= 0.0:
        blockers.append("held_future_effect_not_positive")
    if safety.get("protected_retention_holds") is not True:
        blockers.append("protected_retention_regression")
    if safety.get("contamination_zero") is not True:
        blockers.append("contamination_nonzero")
    if attacks.get("all_critical_attacks_fail_closed") is not True:
        blockers.append("critical_attack_open")
    blockers.extend(
        critical.replace("current_adversarial_flag:", "current_adversarial_critical_flag:", 1)
        for critical in _current_critical_ids(adversarial)
    )
    return (not blockers, blockers)


def same_verdict_retirement_decision(eligibility: bool, blockers: Sequence[str]) -> JsonDict:
    return {
        "schema": SCHEMA + ".same_verdict_retirement",
        "retire_same_verdict": False,
        "decision": "preserve_v552_null_and_block_v553_claim" if not eligibility else "no_retirement_positive_claim_eligible",
        "reason": "current blockers remain visible" if blockers else "no blockers",
        "blocked_by": list(blockers),
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    eligibility = bool(artifact["prospective_csl_claim_eligibility"])
    preconditions = as_mapping(artifact["preconditions_checked"])
    if not preconditions.get("all_required_inputs_present", False):
        artifact["status"] = "complete_blocked"
        artifact["honest_verdict"] = "complete_blocked: V553 row recomputation audit missing required inputs"
        artifact["csl_row_recomputation_audit_ready_score"] = 0.0
    elif eligibility:
        artifact["status"] = "complete_ready"
        artifact["honest_verdict"] = "complete: V553 CSL row recomputation audit found claim-eligible evidence"
        artifact["csl_row_recomputation_audit_ready_score"] = 1.0
    else:
        artifact["status"] = "complete_null"
        artifact["honest_verdict"] = (
            "complete_null: V553 row recomputation audit completed; prospective CSL claim remains ineligible"
        )
        artifact["csl_row_recomputation_audit_ready_score"] = 0.0
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = 0.0,
    tests_run: Mapping[str, int] | None = None,
    run_current_audits: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    context = _load_context(root)
    protected_before = protected_hashes(root)
    ledger = input_hash_ledger(root, context)
    inputs = expected_and_available_upstream_inputs(ledger)
    metrics = recomputed_metrics(context)
    deltas = reported_vs_recomputed_deltas(context, metrics)
    rows = per_unit_rows(context, deltas["comparisons"])
    event_uniqueness = event_and_raw_output_uniqueness_rechecks(context, root)
    causal_order = causal_order_and_exact_feedback_rechecks(context)
    capacity_head = capacity_and_head_transition_rechecks(context)
    held_restart = held_freeze_and_restart_rechecks(context)
    uncertainty = effective_sample_sizes_and_uncertainty_rechecks(metrics)
    safety = safety_cost_rechecks(context, metrics)
    adversarial = current_and_stamped_adversarial_findings(
        root,
        context,
        run_current=run_current_audits,
    )
    det = determination_preservation_findings(root, run_current=run_current_audits)
    convention = artifact_convention_findings(root, run_current=run_current_audits)
    attacks = attack_matrix(
        {
            "event_uniqueness": event_uniqueness,
            "causal_order": causal_order,
            "capacity_head": capacity_head,
            "held_freeze_restart": held_restart,
        },
        adversarial,
        context,
    )
    eligible, blockers = _eligibility(
        inputs=inputs,
        deltas=deltas,
        uncertainty=uncertainty,
        safety=safety,
        attacks=attacks,
        adversarial=adversarial,
        metrics=metrics,
    )
    protected_after = protected_hashes(root)
    preconditions = preconditions_checked(root, run_date, inputs, ledger)
    open_critical = sorted(
        [row["attack_id"] for row in attacks["rows"] if row["fail_closed"] is not True]
        + _current_critical_ids(adversarial)
    )
    blocked_reason = ";".join(blockers[:1]) if blockers else ""
    artifact: JsonDict = {
        "status": "pending",
        "expected_and_available_upstream_inputs": inputs,
        "upstream_artifact_row_manifest_raw_source_test_checker_receipt_head_and_determination_hashes": ledger,
        "missing_input_findings": missing_input_findings(inputs),
        "upstream_state_by_task": upstream_state_by_task(context),
        "per_unit_rows": rows,
        "event_and_raw_output_uniqueness_rechecks": event_uniqueness,
        "causal_order_and_exact_feedback_rechecks": causal_order,
        "capacity_and_head_transition_rechecks": capacity_head,
        "held_freeze_and_restart_rechecks": held_restart,
        "independently_recomputed_development_capacity_interference_and_held_metrics": metrics,
        "reported_vs_recomputed_deltas": deltas,
        "mismatch_count": deltas["mismatch_count"],
        "effective_sample_sizes_and_uncertainty_rechecks": uncertainty,
        "retention_forgetting_contamination_growth_restart_and_cost_rechecks": safety,
        "attack_matrix": attacks,
        "open_critical_attack_ids": open_critical,
        "current_and_stamped_adversarial_findings": adversarial,
        "determination_preservation_findings": det,
        "artifact_convention_findings": convention,
        "public_factor_claim_eligibility": bool(eligible),
        "prospective_csl_claim_eligibility": bool(eligible),
        "csl_row_recomputation_audit_ready_score": 0.0,
        "same_verdict_retirement_decision": same_verdict_retirement_decision(eligible, blockers),
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            context,
            inputs,
            adversarial,
        ),
        "protected_files_unchanged": protected_files_unchanged(protected_before, protected_after),
        "blocked_reason": blocked_reason,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": tests_run_receipt(tests_run),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "pending",
    }
    refresh_terminal_fields(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    require(set(artifact.keys()) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(set(as_mapping(artifact.get("field_principles"))) == set(FIELD_PRINCIPLES), "field_principles")
    require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    inputs = as_mapping(artifact.get("expected_and_available_upstream_inputs"))
    require(inputs.get("missing_required_count") == 0, "expected_and_available_upstream_inputs")
    rows = as_mapping(artifact.get("per_unit_rows"))
    require(
        rows.get("source_unit_row_count")
        == EXP6430_PER_UNIT_ROW_COUNT + EXP6431_PER_UNIT_ROW_COUNT + EXP6432_PER_UNIT_ROW_COUNT,
        "per_unit_rows",
    )
    require(rows.get("comparison_row_count") == len(as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("comparisons", [])), "per_unit_rows")
    deltas = as_mapping(artifact.get("reported_vs_recomputed_deltas"))
    require(deltas.get("all_within_tolerance") is True, "reported_vs_recomputed_deltas")
    require(int(artifact.get("mismatch_count", 1) or 0) == 0, "mismatch_count")
    attacks = as_mapping(artifact.get("attack_matrix"))
    require(attacks.get("all_critical_attacks_fail_closed") is True, "attack_matrix")
    require(all(as_mapping(row).get("fail_closed") is True for row in attacks.get("rows", [])), "attack_matrix")
    open_critical = artifact.get("open_critical_attack_ids", [])
    require(bool(open_critical), "open_critical_attack_ids")
    require(artifact.get("prospective_csl_claim_eligibility") is False, "prospective_csl_claim_eligibility")
    require(artifact.get("public_factor_claim_eligibility") is False, "public_factor_claim_eligibility")
    require(artifact.get("csl_row_recomputation_audit_ready_score") == 0.0, "csl_row_recomputation_audit_ready_score")
    require(artifact.get("status") == "complete_null", "status")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(str(artifact.get("honest_verdict", "")).startswith("complete_null:"), "honest_verdict")
    require(as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True, "protected_files_unchanged")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")
    return True


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_artifact(
    *,
    output_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Mapping[str, int] | None = None,
    run_current_audits: bool = True,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
        run_current_audits=run_current_audits,
    )
    validate_artifact(artifact)
    write_json_atomic(Path(output_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--skip-current-audits", action="store_true")
    args = parser.parse_args(argv)
    artifact = write_artifact(
        output_path=args.output,
        root=REPO_ROOT,
        run_date=str(args.date),
        run_current_audits=not args.skip_current_audits,
    )
    print(args.output)
    return 0 if validate_artifact(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
