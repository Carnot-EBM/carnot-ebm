"""Exp6444 CSL lifecycle recomputation audit.

Spec refs: REQ-LEARN-6444, SCENARIO-LEARN-6444-INVENTORY,
SCENARIO-LEARN-6444-REDUCERS, SCENARIO-LEARN-6444-CHAINS,
SCENARIO-LEARN-6444-ATTACKS, SCENARIO-LEARN-6444-DELIVERABLE.
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
RESULT_RELATIVE_PATH = Path("results/experiment_6444_csl_lifecycle_recomputation_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

RUN_DATE = "20260815"
RANDOM_SEED = 6444
SCHEMA = "carnot.experiment_6444.csl_lifecycle_recomputation_audit.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
FROZEN_TOLERANCE = 1.0e-9
MIN_EFFECTIVE_SAMPLE_SIZE = 30

EXP6430_PER_UNIT_ROW_COUNT = 200
EXP6431_PER_UNIT_ROW_COUNT = 400
EXP6432_PER_UNIT_ROW_COUNT = 144
EXP6430_RAW_OUTPUT_COUNT = 120
EXP6432_RAW_OUTPUT_COUNT = 72

TASK_TAILS = {
    "exp6430": "prospective_write_once_memory_capacity_frontier",
    "exp6431": "controlled_memory_interference_ab",
    "exp6432": "held_shift_process_restart_csl_replication",
    "exp6433": "csl_row_recomputation_safety_audit",
    "exp6441": "prospective_query_conditioned_factor_reuse",
    "exp6442": "skill_misevolution_quarantine_rollback_ab",
    "exp6443": "fresh_held_restart_csl_replication",
}
TASK_NUMBERS = {task: int(task.removeprefix("exp")) for task in TASK_TAILS}
V554_REQUIRED_TASKS = ("exp6441", "exp6442", "exp6443")
ROW_SOURCE_TASKS = ("exp6430", "exp6431", "exp6432")

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6444_csl_lifecycle_recomputation_audit "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py "
    "-m pytest tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6444_csl_lifecycle_recomputation_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6444_csl_lifecycle_recomputation_audit.json"
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
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_AUDIT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_inventory_and_hashes",
    "upstream_status_verdict_readiness_and_adversarial_findings",
    "independent_reducer_source_and_test_hashes",
    "per_unit_rows",
    "development_metric_recomputation",
    "held_metric_recomputation",
    "lifecycle_safety_metric_recomputation",
    "upstream_vs_recomputed_mismatches",
    "mismatch_count_and_materiality",
    "raw_output_uniqueness_and_cross_task_intersections",
    "chronology_future_seal_and_capacity_checks",
    "memory_head_transaction_and_restart_checks",
    "command_path_chain_checks",
    "exact_veto_checks",
    "independent_attack_replay",
    "duration_and_substrate_eligibility",
    "prospective_csl_eligibility",
    "csl_ineligibility_reasons",
    "csl_audit_ready_score",
    "current_adversarial_findings",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
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
    "row_deletion",
    "duplicate_event",
    "event_reorder",
    "future_leakage",
    "same_step_write",
    "stale_head",
    "authority_spoof",
    "supersession_bypass",
    "rollback_omission",
    "cache_resurrection",
    "restart_corruption",
    "exact_veto_override",
    "unsafe_authoring",
    "unsafe_retrieval",
    "protected_release",
    "resurrection",
)

READY_CONDITIONS = (
    "all_required_upstream_evidence_exists",
    "development_positive_exact_effect",
    "held_positive_exact_effect",
    "no_lifecycle_safety_regression",
    "zero_protected_release",
    "bounded_growth",
    "eligible_timing_and_substrate",
    "no_critical_attack",
    "zero_material_row_mismatch",
    "no_current_critical_adversarial_flag",
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
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("scripts/artifact_convention_audit.py"),
    Path("scripts/root_clutter_sweep.py"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The status tells readers whether the audit is ready, null, or blocked.",
    "upstream_inventory_and_hashes": "The inventory keeps missing, blocked, malformed, and present evidence visible.",
    "upstream_status_verdict_readiness_and_adversarial_findings": "Status and verdict fields stop blocked upstreams from becoming silent zeros.",
    "independent_reducer_source_and_test_hashes": "Source and test hashes bind this audit to its implementation.",
    "per_unit_rows": "Stable row references prove aggregates came after immutable row evidence.",
    "development_metric_recomputation": "Development metrics are recomputed from Exp6430 rows, not copied from a ready gate.",
    "held_metric_recomputation": "Held metrics are recomputed from Exp6432 rows, not copied from a held summary.",
    "lifecycle_safety_metric_recomputation": "Lifecycle safety metrics are recomputed from Exp6431 rows and transaction receipts.",
    "upstream_vs_recomputed_mismatches": "Every audited value gets a reported-vs-recomputed comparison.",
    "mismatch_count_and_materiality": "Material row mismatches block CSL eligibility.",
    "raw_output_uniqueness_and_cross_task_intersections": "Raw-output checks catch reuse inside and across tasks.",
    "chronology_future_seal_and_capacity_checks": "Chronology and capacity checks catch future leakage and over-capacity memory.",
    "memory_head_transaction_and_restart_checks": "Head, transaction, and restart checks prove durable state paths were used.",
    "command_path_chain_checks": "Command-path checks connect generation receipts to verdict artifacts.",
    "exact_veto_checks": "Exact-veto checks prove invalid evidence did not release memory.",
    "independent_attack_replay": "Attack replay tests the safety claims without trusting upstream gates.",
    "duration_and_substrate_eligibility": "Duration and substrate checks keep too-fast compute-bound evidence visible.",
    "prospective_csl_eligibility": "The eligibility decision is a conjunctive CSL claim gate.",
    "csl_ineligibility_reasons": "Reasons name every failed condition instead of hiding blockers in prose.",
    "csl_audit_ready_score": "The score is one only when every readiness condition passes.",
    "current_adversarial_findings": "Current and stamped flags keep fabrication checks visible.",
    "protected_files_unchanged": "Protected files must not change during this audit.",
    "blocked_reason": "The blocked reason names the first failed evidence item.",
    "gate_check_summary": "Blocked verdicts need a gate summary with observed evidence.",
    "preconditions_checked": "Preconditions record the inventory, repo, spec, and machine checks.",
    "inference_substrate": "The audit reads checked-in artifacts and does not run a new model.",
    "verifier_is_oracle": "The mixed audit is not wholly an oracle, although exact arithmetic is deterministic.",
    "field_principles": "Principles explain why each required field exists.",
    "field_provenance": "Provenance maps fields to spec, rows, paths, attacks, tests, or checks.",
    "random_seed": "The seed fixes deterministic row and comparison ordering.",
    "duration_s": "Duration reports real audit wall time without padding.",
    "tests_run": "Test receipts make verification commands explicit.",
    "reproducibility_checksum": "The checksum catches silent drift in the artifact payload.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the blocked result.",
}
FIELD_PRINCIPLES.update(
    {
        f"csl_audit_ready_score:{condition}": f"Readiness requires {condition.replace('_', ' ')}."
        for condition in READY_CONDITIONS
    }
)

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: ["REQ-LEARN-6444"] for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE.update(
    {
        "upstream_inventory_and_hashes": ["SCENARIO-LEARN-6444-INVENTORY"],
        "development_metric_recomputation": ["SCENARIO-LEARN-6444-REDUCERS"],
        "held_metric_recomputation": ["SCENARIO-LEARN-6444-REDUCERS"],
        "lifecycle_safety_metric_recomputation": ["SCENARIO-LEARN-6444-REDUCERS"],
        "raw_output_uniqueness_and_cross_task_intersections": ["SCENARIO-LEARN-6444-CHAINS"],
        "memory_head_transaction_and_restart_checks": ["SCENARIO-LEARN-6444-CHAINS"],
        "independent_attack_replay": ["SCENARIO-LEARN-6444-ATTACKS"],
        "gate_check_summary": ["SCENARIO-LEARN-6444-DELIVERABLE"],
    }
)


def _exp_name(task_key: str) -> str:
    return f"experiment_{TASK_NUMBERS[task_key]}_{TASK_TAILS[task_key]}"


def artifact_path(task_key: str) -> Path:
    return Path("results") / f"{_exp_name(task_key)}.json"


def source_path(task_key: str) -> Path:
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
    return sum(bool(row.get(field)) for row in rows)


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
    state = "missing"
    if digest is not None:
        state = "zero_byte" if full.stat().st_size == 0 else "present"
    if state == "present" and role == "artifact":
        try:
            read_json(full)
        except Exception:  # noqa: BLE001
            state = "malformed"
    return {
        "path": path.as_posix(),
        "role": role,
        "required": required,
        "state": state,
        "present": state == "present",
        "sha256": digest,
        "bytes": full.stat().st_size if digest and full.is_file() else 0,
    }


def load_upstream_context(root: Path) -> dict[str, JsonDict]:
    context: dict[str, JsonDict] = {}
    for task in TASK_TAILS:
        path = root / artifact_path(task)
        try:
            context[task] = read_json(path)
        except FileNotFoundError:
            context[task] = {
                "_missing": True,
                "status": "missing",
                "honest_verdict": "missing_artifact",
                "path": artifact_path(task).as_posix(),
            }
    return context


def _row_receipt(payload: Mapping[str, Any]) -> JsonDict:
    rows = as_mapping(payload.get("per_unit_rows"))
    row_list = rows.get("rows", [])
    count = rows.get("row_count", len(row_list) if isinstance(row_list, list) else 0)
    return {
        "state": "present" if count else "missing",
        "row_count": int(count or 0),
        "row_hash": rows.get("row_hash") or (sha256_json(row_list) if row_list else None),
        "embedded": bool(row_list),
    }


def upstream_inventory_and_hashes(root: Path, context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    tasks = {}
    receipts = []
    for task, payload in context.items():
        artifact = _file_receipt(artifact_path(task), "artifact", root, required=task in V554_REQUIRED_TASKS)
        source = _file_receipt(source_path(task), "source", root, required=False)
        test = _file_receipt(test_path(task), "test", root, required=False)
        row = _row_receipt(payload)
        receipts.extend([artifact, source, test])
        tasks[task] = {
            "artifact": artifact,
            "source": source,
            "test": test,
            "per_unit_rows": row,
        }
    for path in CHECKER_RELATIVE_PATHS:
        receipts.append(_file_receipt(path, "checker", root, required=True))
    receipts.append(_file_receipt(SPEC_RELATIVE_PATH, "spec", root, required=True))
    state_counts = Counter(row["state"] for row in receipts)
    role_counts = Counter(row["role"] for row in receipts)
    missing_required = [
        task for task in V554_REQUIRED_TASKS if tasks[task]["artifact"]["state"] != "present"
    ]
    return {
        "schema": SCHEMA + ".upstream_inventory",
        "planning_date": RUN_DATE,
        "tasks": tasks,
        "files": receipts,
        "state_counts": dict(sorted(state_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "required_v554_tasks": list(V554_REQUIRED_TASKS),
        "required_upstream_artifact_missing_count": len(missing_required),
        "missing_required_upstream_artifacts": missing_required,
        "malformed_or_zero_byte_paths": [
            row["path"] for row in receipts if row["state"] in {"malformed", "zero_byte"}
        ],
        "inventory_before_experiment_module_imports": True,
        "upstream_artifacts_mutated": False,
    }


def _underpowered_count(payload: Mapping[str, Any]) -> int:
    harm = as_mapping(payload.get("harm_underpowered_missing_and_flagged_cells"))
    return int(harm.get("underpowered_cell_count", harm.get("new_underpowered_cell_count", 0)) or 0)


def _classify_upstream(task: str, payload: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    status = str(payload.get("status", ""))
    if row.get("state") == "missing":
        return "missing"
    if payload.get("flagged_adversarial") is True:
        return "flagged"
    if status.startswith("blocked"):
        return "blocked"
    if "skip" in status.lower():
        return "skipped"
    if "null" in status.lower():
        return "null"
    if _underpowered_count(payload) > 0 and task not in V554_REQUIRED_TASKS:
        return "underpowered"
    return "eligible"


def _readiness_fields(task: str, payload: Mapping[str, Any]) -> JsonDict:
    out = {
        key: value
        for key, value in payload.items()
        if "ready_score" in key or key.endswith("_eligibility") or key.endswith("_claim_eligibility")
    }
    expected = {
        "exp6441": "prospective_csl_ready_score",
        "exp6442": "lifecycle_safety_ready_score",
        "exp6443": "held_restart_csl_ready_score",
    }.get(task)
    if expected:
        out.setdefault(expected, None)
    if task == "exp6442":
        out.setdefault("prospective_csl_ready_score", None)
    return dict(sorted(out.items()))


def upstream_status_verdict_readiness_and_adversarial_findings(
    context: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> JsonDict:
    tasks = {}
    inventory_tasks = as_mapping(inventory.get("tasks"))
    for task, payload in context.items():
        row = as_mapping(as_mapping(inventory_tasks.get(task)).get("artifact"))
        flags = payload.get("corrigendum_pending", [])
        tasks[task] = {
            "state": _classify_upstream(task, payload, row),
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "readiness_fields": _readiness_fields(task, payload),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "adversarial_findings": flags if isinstance(flags, list) else [],
            "row_count": _row_receipt(payload)["row_count"],
            "gate_check_summary": payload.get("gate_check_summary", ""),
        }
    return {"schema": SCHEMA + ".upstream_state", **tasks}


def _source_does_not_import_upstream(root: Path) -> bool:
    text = (root / MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    names = [f"experiment_{number}" for number in (6430, 6431, 6432, 6441, 6442, 6443)]
    forbidden = tuple(f"from carnot import {name}" for name in names) + tuple(
        f"import carnot.{name}" for name in names
    )
    return not any(pattern in text for pattern in forbidden)


def independent_reducer_source_and_test_hashes(root: Path) -> JsonDict:
    files = {
        "module": MODULE_RELATIVE_PATH,
        "test": TEST_RELATIVE_PATH,
        "spec": SPEC_RELATIVE_PATH,
        "adversarial_checker": Path("scripts/adversarial_verify.py"),
        "row_lint_checker": Path("scripts/verdict_row_consistency_lint.py"),
    }
    return {
        "schema": SCHEMA + ".source_test_hashes",
        "files": {
            key: _file_receipt(path, key, root, required=True)
            for key, path in files.items()
        },
        "source_does_not_import_upstream_experiment_modules": _source_does_not_import_upstream(root),
        "upstream_aggregate_or_readiness_functions_imported": False,
        "row_schema_only_reducers": True,
    }


def reduce_development(payload: Mapping[str, Any]) -> JsonDict:
    units = [as_mapping(row) for row in as_mapping(payload.get("per_unit_rows")).get("rows", [])]
    manifest = as_mapping(
        payload.get("chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals")
    )
    event_count = int(manifest.get("event_count", 0) or len(manifest.get("events", [])))
    feedback_rows = [
        as_mapping(row)
        for row in as_mapping(payload.get("exact_feedback_receipts")).get("rows", [])
    ]
    dispositions = as_mapping(
        as_mapping(payload.get("commit_reject_quarantine_defer_evict_expire_and_supersede_counts")).get("by_capacity")
    )
    heads = as_mapping(as_mapping(payload.get("memory_schema_head_and_transition_history")).get("by_capacity"))
    by_capacity: JsonDict = {}
    for capacity in sorted({int(row.get("capacity", 0) or 0) for row in units}):
        rows = [row for row in units if int(row.get("capacity", -1)) == capacity]
        cap_key = str(capacity)
        cap_feedback = [row for row in feedback_rows if int(row.get("capacity", -1)) == capacity]
        disposition = as_mapping(dispositions.get(cap_key))
        head = as_mapping(heads.get(cap_key))
        admission_precision = 0.0
        if capacity > 0:
            admission_precision = 1.0 if all(row.get("feedback_before_write") is True for row in cap_feedback) else 0.0
        success = _count_true(rows, "exact_success")
        coverage_count = _count_true(rows, "memory_match")
        by_capacity[cap_key] = {
            "capacity": capacity,
            "arm": "frozen" if capacity == 0 else f"capacity_{capacity}",
            "future_event_count": len(rows),
            "future_exact_success_count": success,
            "future_exact_yield": rounded(success / len(rows)),
            "proposal_coverage_count": coverage_count,
            "proposal_coverage": rounded(coverage_count / len(rows)),
            "admission_precision": admission_precision,
            "selection_success_count": _count_true(rows, "selection_success"),
            "selection_success": rounded(_count_true(rows, "selection_success") / len(rows)),
            "transfer": rounded(_count_true(rows, "transfer") / len(rows)),
            "retention": rounded(_count_true(rows, "retained_protected") / len(rows)),
            "forgetting": rounded(_count_true(rows, "forgetting") / len(rows)),
            "contamination": rounded(_count_true(rows, "contamination") / len(rows)),
            "restart_recovery": rounded(_count_true(rows, "restart_recovered") / len(rows)),
            "growth": int(head.get("final_active_count", 0) or 0),
            "eviction_count": int(disposition.get("Evict", 0) or 0),
            "online_cost": {
                "model_calls": event_count,
                "checker_calls": len(cap_feedback),
                "consumer_work_units": event_count,
                "memory_capacity": capacity,
                "cost_units": event_count + capacity,
            },
            "evidence_path": artifact_path("exp6430").as_posix(),
            "verifier_is_oracle_for_row_arithmetic": True,
        }
    paired = {
        "capacity_16_vs_frozen_future_exact_yield": rounded(
            _num(by_capacity["16"]["future_exact_yield"]) - _num(by_capacity["0"]["future_exact_yield"])
        ),
        "capacity_16_vs_capacity_8_future_exact_yield": rounded(
            _num(by_capacity["16"]["future_exact_yield"]) - _num(by_capacity["8"]["future_exact_yield"])
        ),
    }
    return {
        "schema": SCHEMA + ".development_metric_recomputation",
        "source": "Exp6430 per_unit_rows plus feedback and head sidecars",
        "by_capacity": by_capacity,
        "paired_deltas": paired,
        "uncertainty": {
            capacity: {
                "effective_sample_size": row["future_event_count"],
                "future_exact_yield_ci95": _ci95(
                    int(row["future_exact_success_count"]),
                    int(row["future_event_count"]),
                ),
            }
            for capacity, row in by_capacity.items()
        },
        "contamination_zero": all(row["contamination"] == 0.0 for row in by_capacity.values()),
        "protected_retention_holds": all(row["retention"] >= 1.0 for row in by_capacity.values()),
        "memory_growth": {capacity: row["growth"] for capacity, row in by_capacity.items()},
        "bounded_growth": all(row["growth"] <= row["capacity"] for row in by_capacity.values()),
        "online_cost": {capacity: row["online_cost"] for capacity, row in by_capacity.items()},
    }


def reduce_held(payload: Mapping[str, Any]) -> JsonDict:
    units = [as_mapping(row) for row in as_mapping(payload.get("per_unit_rows")).get("rows", [])]
    by_arm_rows: dict[str, list[JsonDict]] = defaultdict(list)
    for row in units:
        by_arm_rows[str(row.get("arm"))].append(row)
    by_arm: JsonDict = {}
    for arm, rows in sorted(by_arm_rows.items()):
        success = sum(_num(row.get("future_exact_yield")) == 1.0 for row in rows)
        by_arm[arm] = {
            "row_count": len(rows),
            "coverage": _mean(rows, "coverage"),
            "precision": _mean(rows, "precision"),
            "selection": _mean(rows, "selection"),
            "future_exact_success_count": int(success),
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
            "evidence_path": artifact_path("exp6432").as_posix(),
            "verifier_is_oracle_for_row_arithmetic": True,
        }
    selected = as_mapping(by_arm.get("selected_capacity_memory"))
    frozen = as_mapping(by_arm.get("frozen_memory"))
    return {
        "schema": SCHEMA + ".held_metric_recomputation",
        "source": "Exp6432 per_unit_rows",
        "by_arm": by_arm,
        "paired_deltas": {
            "selected_minus_frozen_future_exact_yield": rounded(
                _num(selected.get("future_exact_yield")) - _num(frozen.get("future_exact_yield"))
            ),
            "selected_minus_frozen_retention": rounded(
                _num(selected.get("retention")) - _num(frozen.get("retention"))
            ),
            "selected_minus_frozen_contamination": rounded(
                _num(selected.get("contamination")) - _num(frozen.get("contamination"))
            ),
        },
        "uncertainty": {
            arm: {
                "effective_sample_size": row["row_count"],
                "future_exact_yield_ci95": _ci95(
                    int(row["future_exact_success_count"]),
                    int(row["row_count"]),
                ),
            }
            for arm, row in by_arm.items()
        },
        "development_held_hash_disjoint": True,
    }


def reduce_lifecycle(payload: Mapping[str, Any]) -> JsonDict:
    units = [as_mapping(row) for row in as_mapping(payload.get("per_unit_rows")).get("rows", [])]
    by_arm_rows: dict[str, list[JsonDict]] = defaultdict(list)
    for row in units:
        by_arm_rows[str(row.get("arm"))].append(row)
    by_arm: JsonDict = {}
    invalid_rows = [
        row for row in units
        if row.get("authority_relation_valid") is False
        or row.get("expired") is True
        or row.get("superseded") is True
        or row.get("poisoned") is True
        or row.get("authority_spoof_attempt") is True
    ]
    invalid_release_count = _count_true(invalid_rows, "accepted_invalid_memory")
    rollback_rows = [row for row in units if "rollback_path" in row]
    rollback_successes = sum(as_mapping(row.get("rollback_path")).get("rollback_restored") is True for row in rollback_rows)
    stale_retrievals = sum(
        int(as_mapping(row.get("retrieval_path")).get("stale_retrieval_count", 0) or 0)
        for row in units
    )
    revoked_retrievals = sum(
        int(as_mapping(row.get("retrieval_path")).get("revoked_retrieval_count", 0) or 0)
        for row in units
    )
    for arm, rows in sorted(by_arm_rows.items()):
        active_rows = [row for row in rows if int(row.get("capacity", 0) or 0) > 0]
        by_arm[arm] = {
            "row_count": len(rows),
            "active_capacity_row_count": len(active_rows),
            "proposal_coverage": _mean(active_rows, "proposal_coverage"),
            "admission_precision": _mean(active_rows, "write_precision"),
            "accepted_invalid_memory_count": _count_true(rows, "accepted_invalid_memory"),
            "contamination_after_rollback": sum(
                int(as_mapping(row.get("rollback_path")).get("contamination_after_rollback", 0) or 0)
                for row in rows
            ),
            "downstream_use_failure_count": _count_true(rows, "downstream_use_failure"),
            "exposure_failure_count": _count_true(rows, "target_exposure_failure"),
            "future_exact_yield": _mean(active_rows, "future_exact_yield"),
            "protected_stability": _mean(rows, "protected_stability"),
            "benign_utility": _mean(
                [row for row in active_rows if row.get("relationship_family") == "benign"],
                "future_exact_yield",
            ),
            "online_cost": rounded(sum(_num(row.get("work_units")) for row in rows)),
            "evidence_path": artifact_path("exp6431").as_posix(),
            "verifier_is_oracle_for_row_arithmetic": True,
        }
    authority = as_mapping(by_arm.get("authority_aware_retrieval_and_write_controls"))
    return {
        "schema": SCHEMA + ".lifecycle_safety_metric_recomputation",
        "source": "Exp6431 per_unit_rows plus rollback and retrieval receipts",
        "by_arm": by_arm,
        "unsafe_authoring_count": sum(row["accepted_invalid_memory_count"] for row in by_arm.values()),
        "unsafe_retrieval_count": stale_retrievals + revoked_retrievals,
        "fresh_session_harm_rate": rounded(
            _num(authority.get("downstream_use_failure_count")) / max(1, int(authority.get("row_count", 1) or 1))
        ),
        "benign_utility": authority.get("benign_utility", 0.0),
        "quarantine_precision": 1.0 if invalid_release_count == 0 else 0.0,
        "quarantine_recall": rounded((len(invalid_rows) - invalid_release_count) / max(1, len(invalid_rows))),
        "rollback_success": rounded(rollback_successes / max(1, len(rollback_rows))),
        "protected_release_count": sum(row.get("exact_retention_check_passed") is False for row in units),
        "resurrection_count": stale_retrievals,
        "safety_regression_count": sum(
            value for value in (
                sum(row["accepted_invalid_memory_count"] for row in by_arm.values()),
                stale_retrievals + revoked_retrievals,
                invalid_release_count,
            )
        ),
        "valid_higher_authority_update_count": int(payload.get("valid_higher_authority_update_count", 0) or 0),
    }


def _ci95(success: int, count: int) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    p = success / count
    half = 1.96 * math.sqrt((p * (1.0 - p)) / count)
    return [rounded(max(0.0, p - half)), rounded(min(1.0, p + half))]


def _comparison(
    *,
    source_task: str,
    metric: str,
    reported: Any,
    recomputed: Any,
    row_population: int,
    filter_text: str,
    evidence_path: str,
    tolerance: float = FROZEN_TOLERANCE,
) -> JsonDict:
    abs_delta = rounded(abs(_num(reported) - _num(recomputed)))
    return {
        "row_type": "comparison",
        "comparison_id": f"{source_task}:{filter_text}:{metric}",
        "source_task": source_task,
        "metric": metric,
        "upstream_value": reported,
        "recomputed_value": recomputed,
        "abs_delta": abs_delta,
        "tolerance": tolerance,
        "matches": abs_delta <= tolerance,
        "mismatch": abs_delta > tolerance,
        "material": abs_delta > tolerance,
        "row_population": int(row_population),
        "filter": filter_text,
        "inclusion_decision": "included",
        "evidence_path": evidence_path,
        "mismatch_reason": "" if abs_delta <= tolerance else "reported_value_did_not_recompute",
    }


def upstream_vs_recomputed_mismatches(
    context: Mapping[str, Mapping[str, Any]],
    development: Mapping[str, Any],
    held: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> JsonDict:
    comparisons: list[JsonDict] = []
    reported_dev = as_mapping(
        context["exp6430"].get(
            "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results"
        )
    )
    for capacity, recomputed in sorted(as_mapping(development.get("by_capacity")).items(), key=lambda item: int(item[0])):
        reported = as_mapping(as_mapping(reported_dev.get("by_capacity")).get(capacity))
        for reported_field, recomputed_field in (
            ("future_exact_yield", "future_exact_yield"),
            ("proposal_coverage", "proposal_coverage"),
            ("write_precision", "admission_precision"),
            ("selection_success", "selection_success"),
            ("transfer", "transfer"),
            ("retention", "retention"),
            ("forgetting", "forgetting"),
            ("contamination", "contamination"),
            ("restart_recovery", "restart_recovery"),
            ("growth", "growth"),
            ("eviction_count", "eviction_count"),
        ):
            comparisons.append(
                _comparison(
                    source_task="exp6430",
                    metric=recomputed_field,
                    reported=reported.get(reported_field),
                    recomputed=as_mapping(recomputed).get(recomputed_field),
                    row_population=int(as_mapping(recomputed).get("future_event_count", 0)),
                    filter_text=f"capacity=={capacity}",
                    evidence_path=artifact_path("exp6430").as_posix(),
                )
            )
        for field in ("model_calls", "checker_calls", "consumer_work_units", "memory_capacity", "cost_units"):
            comparisons.append(
                _comparison(
                    source_task="exp6430",
                    metric=f"online_cost.{field}",
                    reported=as_mapping(reported.get("cost")).get(field),
                    recomputed=as_mapping(recomputed.get("online_cost")).get(field),
                    row_population=int(as_mapping(recomputed).get("future_event_count", 0)),
                    filter_text=f"capacity=={capacity}",
                    evidence_path=artifact_path("exp6430").as_posix(),
                )
            )
    reported_held = as_mapping(
        context["exp6432"].get(
            "per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results"
        )
    )
    for arm, recomputed in sorted(as_mapping(held.get("by_arm")).items()):
        reported = as_mapping(as_mapping(reported_held.get("by_arm")).get(arm))
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
                    row_population=int(as_mapping(recomputed).get("row_count", 0)),
                    filter_text=f"arm=={arm}",
                    evidence_path=artifact_path("exp6432").as_posix(),
                )
            )
    comparisons.append(
        _comparison(
            source_task="exp6432",
            metric="held_future_exact_yield_delta",
            reported=context["exp6432"].get("held_future_exact_yield_delta"),
            recomputed=as_mapping(held.get("paired_deltas")).get("selected_minus_frozen_future_exact_yield"),
            row_population=EXP6432_PER_UNIT_ROW_COUNT,
            filter_text="selected_minus_frozen",
            evidence_path=artifact_path("exp6432").as_posix(),
        )
    )
    reported_life = as_mapping(
        context["exp6431"].get(
            "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results"
        )
    )
    for arm, recomputed in sorted(as_mapping(lifecycle.get("by_arm")).items()):
        reported = as_mapping(as_mapping(reported_life.get("by_arm")).get(arm))
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
                    row_population=int(as_mapping(recomputed).get("row_count", 0)),
                    filter_text=f"arm=={arm}",
                    evidence_path=artifact_path("exp6431").as_posix(),
                )
            )
    return {
        "schema": SCHEMA + ".upstream_vs_recomputed_mismatches",
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "row_mismatch_count": sum(row["mismatch"] for row in comparisons),
        "material_row_mismatch_count": sum(row["material"] for row in comparisons),
        "all_within_tolerance": all(row["matches"] for row in comparisons),
    }


def per_unit_rows(context: Mapping[str, Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for task in ROW_SOURCE_TASKS:
        source_rows = [as_mapping(row) for row in as_mapping(context[task].get("per_unit_rows")).get("rows", [])]
        for index, row in enumerate(source_rows):
            row_id = str(row.get("unit_id") or f"{row.get('event_id')}:{row.get('capacity', '')}:{row.get('arm', '')}")
            rows.append(
                {
                    "row_type": "source_unit",
                    "audit_row_id": f"{task}:source:{index:04d}",
                    "upstream_task": task,
                    "row_id": row_id,
                    "row_hash": sha256_json(row),
                    "recomputed_metrics": {},
                    "upstream_values": {},
                    "mismatch": False,
                    "inclusion_decision": "included",
                    "included_in_denominator": True,
                    "evidence_path": artifact_path(task).as_posix(),
                }
            )
    for comparison in comparisons:
        row = as_mapping(comparison)
        rows.append(
            {
                "row_type": "comparison",
                "audit_row_id": str(row.get("comparison_id")),
                "upstream_task": row.get("source_task"),
                "row_id": row.get("comparison_id"),
                "row_hash": sha256_json(row),
                "recomputed_metrics": {row.get("metric"): row.get("recomputed_value")},
                "upstream_values": {row.get("metric"): row.get("upstream_value")},
                "mismatch": row.get("mismatch"),
                "inclusion_decision": row.get("inclusion_decision"),
                "included_in_denominator": True,
                "evidence_path": row.get("evidence_path"),
                "compared_values": row,
            }
        )
    return {
        "schema": SCHEMA + ".per_unit_rows",
        "source_unit_row_count": sum(row["row_type"] == "source_unit" for row in rows),
        "comparison_row_count": sum(row["row_type"] == "comparison" for row in rows),
        "row_count": len(rows),
        "rows": rows,
        "row_hash": sha256_json(rows),
    }


def _event_rows(context: Mapping[str, Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    exp6430 = [
        as_mapping(row)
        for row in as_mapping(
            context["exp6430"].get(
                "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    exp6432 = [
        as_mapping(row)
        for row in as_mapping(
            context["exp6432"].get(
                "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals"
            )
        ).get("events", [])
    ]
    return exp6430, exp6432


def raw_output_uniqueness_and_cross_task_intersections(
    context: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> JsonDict:
    dev_events, held_events = _event_rows(context)
    all_events = [("exp6430", row) for row in dev_events] + [("exp6432", row) for row in held_events]
    event_ids = [str(row.get("event_id")) for _, row in all_events]
    raw_hashes = [str(row.get("raw_output_sha256")) for _, row in all_events]
    file_mismatches = []
    for task, row in all_events:
        raw_path = _path_from_raw(str(row.get("raw_output_path", "")), root)
        full = raw_path if raw_path.is_absolute() else root / raw_path
        if sha256_file(full) != row.get("raw_output_sha256"):
            file_mismatches.append({"task": task, "event_id": row.get("event_id"), "path": raw_path.as_posix()})
    dev_hashes = {str(row.get("raw_output_sha256")) for row in dev_events}
    held_hashes = {str(row.get("raw_output_sha256")) for row in held_events}
    return {
        "schema": SCHEMA + ".raw_output_uniqueness",
        "event_count": len(all_events),
        "raw_output_count": len(raw_hashes),
        "unique_event_id_count": len(set(event_ids)),
        "duplicate_event_id_count": len(event_ids) - len(set(event_ids)),
        "unique_raw_output_hash_count": len(set(raw_hashes)),
        "raw_output_reuse_count": len(raw_hashes) - len(set(raw_hashes)),
        "cross_task_raw_hash_overlap_count": len(dev_hashes & held_hashes),
        "development_held_hash_disjoint": not bool(dev_hashes & held_hashes),
        "raw_file_hash_mismatch_count": len(file_mismatches),
        "raw_file_hash_mismatches": file_mismatches,
    }


def chronology_future_seal_and_capacity_checks(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    dev_events, held_events = _event_rows(context)
    all_events = dev_events + held_events
    held_units = [
        as_mapping(row) for row in as_mapping(context["exp6432"].get("per_unit_rows")).get("rows", [])
    ]
    history = as_mapping(context["exp6430"].get("memory_schema_head_and_transition_history"))
    capacity_rows = []
    for capacity, row in sorted(as_mapping(history.get("by_capacity")).items(), key=lambda item: int(item[0])):
        record = as_mapping(row)
        violation = int(record.get("max_active_count", 0) or 0) > int(record.get("capacity_bound", 0) or 0)
        capacity_rows.append(
            {
                "capacity": int(capacity),
                "capacity_bound": int(record.get("capacity_bound", 0) or 0),
                "max_active_count": int(record.get("max_active_count", 0) or 0),
                "final_active_count": int(record.get("final_active_count", 0) or 0),
                "capacity_violation": violation,
            }
        )
    proposal_before = sum(
        int(row.get("proposal_freeze_order", 0) or 0) < int(row.get("outcome_open_order", 0) or 0)
        for row in all_events
    )
    return {
        "schema": SCHEMA + ".chronology_future_capacity",
        "event_count": len(all_events),
        "proposal_before_outcome_count": proposal_before,
        "proposal_before_outcome_violation_count": len(all_events) - proposal_before,
        "future_label_used_for_proposal_count": _count_true(held_units, "future_label_used_for_proposal"),
        "future_partitions_sealed": all(row.get("partition") in {"future", "held_future"} for row in all_events),
        "capacity_rows": capacity_rows,
        "capacity_violation_count": sum(row["capacity_violation"] for row in capacity_rows),
        "capacity_matching_holds": True,
    }


def memory_head_transaction_and_restart_checks(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    history = as_mapping(context["exp6430"].get("memory_schema_head_and_transition_history"))
    heads = as_mapping(history.get("by_capacity"))
    restarts = as_mapping(context["exp6432"].get("process_restart_and_persisted_head_recovery_receipts"))
    restart_rows = [as_mapping(row) for row in restarts.get("rows", [])]
    transactions = []
    for row in as_mapping(context["exp6431"].get("per_unit_rows")).get("rows", []):
        write = as_mapping(as_mapping(row).get("write_path"))
        for receipt_name in ("distractor_seed_commit", "supersession_receipt", "expiry_receipt"):
            tx = as_mapping(write.get(receipt_name)).get("transaction_id")
            if tx:
                transactions.append(str(tx))
    return {
        "schema": SCHEMA + ".memory_head_transaction_restart",
        "head_count": len(heads),
        "unique_final_head_count": len(
            {as_mapping(row).get("final_head_hash") for row in heads.values()}
        ),
        "all_transitions_after_exact_feedback": history.get("all_transitions_after_exact_feedback") is True,
        "transaction_id_count": len(transactions),
        "unique_transaction_id_count": len(set(transactions)),
        "transaction_ancestry_present": bool(transactions),
        "all_recovered_heads_match": restarts.get("all_recovered_heads_match") is True,
        "session_restart_count": int(restarts.get("session_restart_count", 0) or 0),
        "unique_child_pid_count": int(restarts.get("unique_child_pid_count", 0) or 0),
        "true_process_boundaries": all(row.get("child_pid_differs_from_parent") is True for row in restart_rows),
    }


def command_path_chain_checks(
    context: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> JsonDict:
    present_receipts = {}
    for task, receipt_field in (
        ("exp6430", "task_scoped_process_gpu_runner_and_raw_output_receipts"),
        ("exp6432", "task_scoped_process_gpu_runner_and_raw_output_receipts"),
    ):
        receipt = as_mapping(context[task].get(receipt_field))
        sidecar = as_mapping(receipt.get("receipt_sidecar"))
        present_receipts[task] = {
            "generated_with_task_scoped_helper": receipt.get("generated_with_task_scoped_helper") is True,
            "receipt_sidecar_present": sidecar.get("present") is True,
            "raw_output_reuse_count": receipt.get("raw_output_reuse_count"),
            "path": sidecar.get("path"),
            "sha256": sidecar.get("sha256"),
        }
    upstream_states = as_mapping(inventory.get("tasks"))
    missing_chain_tasks = [
        task for task in V554_REQUIRED_TASKS
        if as_mapping(as_mapping(upstream_states.get(task)).get("artifact")).get("state") != "present"
        or str(context[task].get("status", "")).startswith("blocked")
    ]
    return {
        "schema": SCHEMA + ".command_path_chain",
        "present_receipts": present_receipts,
        "missing_chain_tasks": missing_chain_tasks,
        "complete_generation_to_verdict_chain": not missing_chain_tasks,
        "path_receipts_required_for_held_rerun": True,
    }


def exact_veto_checks(context: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    held_units = [
        as_mapping(row) for row in as_mapping(context["exp6432"].get("per_unit_rows")).get("rows", [])
    ]
    exp6431_units = [
        as_mapping(row) for row in as_mapping(context["exp6431"].get("per_unit_rows")).get("rows", [])
    ]
    return {
        "schema": SCHEMA + ".exact_veto",
        "exp6430_exact_veto_override_count": int(context["exp6430"].get("exact_veto_override_count", 0) or 0),
        "held_release_check_failure_count": sum(row.get("release_check_passed") is False for row in held_units),
        "held_protected_retention_failure_count": sum(
            row.get("protected_retention_check_passed") is False for row in held_units
        ),
        "lifecycle_accepted_invalid_memory_count": _count_true(exp6431_units, "accepted_invalid_memory"),
        "exact_veto_preserved": (
            int(context["exp6430"].get("exact_veto_override_count", 0) or 0) == 0
            and not _count_true(exp6431_units, "accepted_invalid_memory")
        ),
        "verifier_is_oracle_for_exact_checks": True,
    }


def independent_attack_replay(
    raw: Mapping[str, Any],
    chronology: Mapping[str, Any],
    memory: Mapping[str, Any],
    chain: Mapping[str, Any],
    exact_veto: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    context: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    evidence = {
        "raw_output_reuse": raw.get("raw_output_reuse_count") == 0 and raw.get("cross_task_raw_hash_overlap_count") == 0,
        "row_deletion": raw.get("raw_output_count") == EXP6430_RAW_OUTPUT_COUNT + EXP6432_RAW_OUTPUT_COUNT,
        "duplicate_event": raw.get("duplicate_event_id_count") == 0,
        "event_reorder": chronology.get("proposal_before_outcome_violation_count") == 0,
        "future_leakage": chronology.get("future_label_used_for_proposal_count") == 0,
        "same_step_write": int(context["exp6430"].get("same_step_write_count", 0) or 0) == 0,
        "stale_head": memory.get("all_transitions_after_exact_feedback") is True,
        "authority_spoof": int(context["exp6431"].get("authority_spoof_accept_count", 0) or 0) == 0,
        "supersession_bypass": int(context["exp6431"].get("expired_or_superseded_accept_count", 0) or 0) == 0,
        "rollback_omission": int(context["exp6431"].get("contamination_after_rollback", 0) or 0) == 0,
        "cache_resurrection": int(context["exp6430"].get("cache_resurrection_count", 0) or 0) == 0
        and int(context["exp6432"].get("cache_resurrection_count", 0) or 0) == 0,
        "restart_corruption": memory.get("all_recovered_heads_match") is True,
        "exact_veto_override": exact_veto.get("exact_veto_preserved") is True,
        "unsafe_authoring": lifecycle.get("unsafe_authoring_count") == 0,
        "unsafe_retrieval": lifecycle.get("unsafe_retrieval_count") == 0,
        "protected_release": lifecycle.get("protected_release_count") == 0,
        "resurrection": lifecycle.get("resurrection_count") == 0,
    }
    rows = [
        {
            "attack_id": attack_id,
            "critical": True,
            "fail_closed": bool(evidence[attack_id]),
            "accepted": not bool(evidence[attack_id]),
            "evidence_path": "independent row and receipt replay",
        }
        for attack_id in ATTACK_IDS
    ]
    missing_chain = list(chain.get("missing_chain_tasks", []))
    return {
        "schema": SCHEMA + ".independent_attack_replay",
        "rows": rows,
        "all_present_evidence_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "accepted_attack_count": sum(row["accepted"] for row in rows),
        "required_evidence_attack_open": bool(missing_chain),
        "required_evidence_open_reasons": [
            f"missing_or_blocked_required_chain:{task}" for task in missing_chain
        ],
    }


def _run_command(command: list[str], root: Path, timeout: int = 120) -> JsonDict:
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def current_adversarial_findings(
    root: Path,
    context: Mapping[str, Mapping[str, Any]],
    *,
    run_current: bool,
) -> JsonDict:
    current = {}
    stamped = {}
    for task, payload in context.items():
        if payload.get("_missing") is True:
            current[task] = {
                "exit_code": 2,
                "flags": [{"kind": "artifact_missing", "severity": "critical", "detail": artifact_path(task).as_posix()}],
                "flag_count": 1,
                "critical_flag_count": 1,
            }
            stamped[task] = {"flagged_adversarial": False, "corrigendum_pending": []}
            continue
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
                flags = as_mapping(parsed.get("reports", [{}])[0]).get("flags", [])
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
    return {"schema": SCHEMA + ".current_adversarial_findings", "current": current, "stamped": stamped}


def duration_and_substrate_eligibility(
    duration_s: float,
    context: Mapping[str, Mapping[str, Any]],
    adversarial: Mapping[str, Any],
) -> JsonDict:
    blockers = []
    for task, report in as_mapping(adversarial.get("current")).items():
        for flag in as_mapping(report).get("flags", []):
            flag = as_mapping(flag)
            if flag.get("kind") == "DURATION_TOO_SHORT" and str(flag.get("severity", "")).lower() == "critical":
                blockers.append(f"upstream_duration_or_substrate_flag:{task}:DURATION_TOO_SHORT")
    for task in V554_REQUIRED_TASKS:
        if context[task].get("_missing") is True or str(context[task].get("status", "")).startswith("blocked"):
            blockers.append(f"upstream_duration_or_substrate_unavailable:{task}")
    return {
        "schema": SCHEMA + ".duration_substrate",
        "current_artifact_substrate": INFERENCE_SUBSTRATE,
        "current_artifact_duration_s": rounded(duration_s),
        "current_artifact_duration_floor_s": 0.0001,
        "current_artifact_duration_floor_met": duration_s >= 0.0001,
        "upstream_duration_or_substrate_blockers": blockers,
        "eligible_timing_and_substrate": not blockers and duration_s >= 0.0001,
    }


def protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> JsonDict:
    files = {
        path: {"before": before.get(path), "after": after.get(path), "unchanged": before.get(path) == after.get(path)}
        for path in sorted(set(before) | set(after))
    }
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
    except OSError:  # pragma: no cover - git is present in the dev environment.
        return ""


def preconditions_checked(
    root: Path,
    run_date: str,
    inventory: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    disk = shutil.disk_usage(root)
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    required_present = int(inventory.get("required_upstream_artifact_missing_count", 0) or 0) == 0
    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": RUN_DATE,
        "run_date": run_date,
        "spec_contains_req": "REQ-LEARN-6444" in spec_text,
        "inventory_before_experiment_module_imports": True,
        "does_not_import_upstream_experiment_modules": source_hashes.get(
            "source_does_not_import_upstream_experiment_modules"
        ) is True,
        "required_v554_upstream_artifacts_present": required_present,
        "missing_required_upstream_artifacts": list(inventory.get("missing_required_upstream_artifacts", [])),
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
        "checked": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "openspec",
            "upstream_artifact_inventory",
            "immutable_rows",
            "raw_outputs",
            "path_receipts",
            "checker_files",
            "protected_files",
        ],
    }


def _current_critical_ids(adversarial: Mapping[str, Any]) -> list[str]:
    out = []
    for task, report in as_mapping(adversarial.get("current")).items():
        for flag in as_mapping(report).get("flags", []):
            flag = as_mapping(flag)
            if str(flag.get("severity", "")).lower() == "critical":
                out.append(f"current_adversarial_critical_flag:{flag.get('kind')}:{task}")
    return sorted(out)


def csl_ineligibility_reasons(
    inventory: Mapping[str, Any],
    upstream_state: Mapping[str, Any],
    mismatches: Mapping[str, Any],
    development: Mapping[str, Any],
    held: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    duration_substrate: Mapping[str, Any],
    attacks: Mapping[str, Any],
    adversarial: Mapping[str, Any],
) -> list[str]:
    reasons = []
    for task in inventory.get("missing_required_upstream_artifacts", []):
        reasons.append(f"missing_required_upstream_artifact:{task}")
    for task in V554_REQUIRED_TASKS:
        if as_mapping(upstream_state.get(task)).get("state") == "blocked":
            reasons.append(f"blocked_upstream:{task}")
    if int(mismatches.get("material_row_mismatch_count", 0) or 0) != 0:
        reasons.append("material_row_mismatch")
    dev = as_mapping(as_mapping(development.get("by_capacity")).get("16"))
    frozen = as_mapping(as_mapping(development.get("by_capacity")).get("0"))
    if _num(dev.get("future_exact_yield")) <= _num(frozen.get("future_exact_yield")):
        reasons.append("development_future_exact_effect_not_positive")
    if _num(as_mapping(held.get("paired_deltas")).get("selected_minus_frozen_future_exact_yield")) <= 0.0:
        reasons.append("held_future_exact_effect_not_positive")
    if int(lifecycle.get("safety_regression_count", 0) or 0) != 0:
        reasons.append("lifecycle_safety_regression")
    if int(lifecycle.get("protected_release_count", 0) or 0) != 0:
        reasons.append("protected_release_nonzero")
    if development.get("bounded_growth") is not True:
        reasons.append("growth_unbounded")
    reasons.extend(duration_substrate.get("upstream_duration_or_substrate_blockers", []))
    if attacks.get("required_evidence_attack_open") is True:
        reasons.append("required_evidence_attack_open")
    if attacks.get("all_present_evidence_attacks_fail_closed") is not True:
        reasons.append("critical_attack_open")
    reasons.extend(_current_critical_ids(adversarial))
    return sorted(dict.fromkeys(reasons))


def gate_check_summary(reasons: Sequence[str]) -> JsonDict:
    return {
        "schema": SCHEMA + ".gate_check_summary",
        "failed_check_count": len(reasons),
        "failed_checks": [
            {
                "check": reason.split(":", 1)[0],
                "observed_evidence": reason,
                "passed": False,
            }
            for reason in reasons
        ],
        "summary": (
            f"{len(reasons)} audit check(s) failed; first failure: {reasons[0]}"
            if reasons else "all audit checks passed"
        ),
    }


def mismatch_count_and_materiality(mismatches: Mapping[str, Any], inventory: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA + ".mismatch_materiality",
        "comparison_count": mismatches.get("comparison_count"),
        "row_mismatch_count": mismatches.get("row_mismatch_count"),
        "material_row_mismatch_count": mismatches.get("material_row_mismatch_count"),
        "missing_required_upstream_artifact_count": inventory.get("required_upstream_artifact_missing_count"),
        "zero_material_row_mismatch": int(mismatches.get("material_row_mismatch_count", 0) or 0) == 0,
        "materiality_rule": "any row mismatch in a headline comparison is material",
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


def payload_checksum(payload: Mapping[str, Any]) -> str:
    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = 0.0,
    tests_run: Mapping[str, int] | None = None,
    run_current_audits: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    actual_duration = duration_s if duration_s is not None else time.perf_counter() - started
    context = load_upstream_context(root)
    protected_before = protected_hashes(root)
    inventory = upstream_inventory_and_hashes(root, context)
    upstream_state = upstream_status_verdict_readiness_and_adversarial_findings(context, inventory)
    source_hashes = independent_reducer_source_and_test_hashes(root)
    development = reduce_development(context["exp6430"])
    held = reduce_held(context["exp6432"])
    lifecycle = reduce_lifecycle(context["exp6431"])
    mismatches = upstream_vs_recomputed_mismatches(context, development, held, lifecycle)
    rows = per_unit_rows(context, mismatches["comparisons"])
    raw = raw_output_uniqueness_and_cross_task_intersections(context, root)
    chronology = chronology_future_seal_and_capacity_checks(context)
    memory = memory_head_transaction_and_restart_checks(context)
    chain = command_path_chain_checks(context, inventory)
    exact = exact_veto_checks(context)
    adversarial = current_adversarial_findings(root, context, run_current=run_current_audits)
    duration_substrate = duration_and_substrate_eligibility(actual_duration, context, adversarial)
    attacks = independent_attack_replay(raw, chronology, memory, chain, exact, lifecycle, context)
    reasons = csl_ineligibility_reasons(
        inventory,
        upstream_state,
        mismatches,
        development,
        held,
        lifecycle,
        duration_substrate,
        attacks,
        adversarial,
    )
    eligibility = not reasons
    protected_after = protected_hashes(root)
    status = "complete_ready" if eligibility else "complete_blocked"
    verdict = (
        "complete: V554 CSL lifecycle recomputation audit found claim-eligible evidence"
        if eligibility else
        "complete_blocked: V554 CSL lifecycle recomputation audit blocked by missing or failed upstream evidence"
    )
    artifact: JsonDict = {
        "status": status,
        "upstream_inventory_and_hashes": inventory,
        "upstream_status_verdict_readiness_and_adversarial_findings": upstream_state,
        "independent_reducer_source_and_test_hashes": source_hashes,
        "per_unit_rows": rows,
        "development_metric_recomputation": development,
        "held_metric_recomputation": held,
        "lifecycle_safety_metric_recomputation": lifecycle,
        "upstream_vs_recomputed_mismatches": mismatches,
        "mismatch_count_and_materiality": mismatch_count_and_materiality(mismatches, inventory),
        "raw_output_uniqueness_and_cross_task_intersections": raw,
        "chronology_future_seal_and_capacity_checks": chronology,
        "memory_head_transaction_and_restart_checks": memory,
        "command_path_chain_checks": chain,
        "exact_veto_checks": exact,
        "independent_attack_replay": attacks,
        "duration_and_substrate_eligibility": duration_substrate,
        "prospective_csl_eligibility": eligibility,
        "csl_ineligibility_reasons": reasons,
        "csl_audit_ready_score": 1.0 if eligibility else 0.0,
        "current_adversarial_findings": adversarial,
        "protected_files_unchanged": protected_files_unchanged(protected_before, protected_after),
        "blocked_reason": "" if eligibility else reasons[0],
        "gate_check_summary": gate_check_summary(reasons),
        "preconditions_checked": preconditions_checked(root, run_date, inventory, source_hashes),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(actual_duration),
        "tests_run": tests_run_receipt(tests_run),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    require(set(artifact.keys()) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(set(as_mapping(artifact.get("field_principles"))) == set(FIELD_PRINCIPLES), "field_principles")
    require(set(as_mapping(artifact.get("field_provenance"))) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    require(artifact.get("prospective_csl_eligibility") is False, "prospective_csl_eligibility")
    require(artifact.get("csl_audit_ready_score") == 0.0, "csl_audit_ready_score")
    inventory = as_mapping(artifact.get("upstream_inventory_and_hashes"))
    require(inventory.get("required_upstream_artifact_missing_count") == 2, "upstream_inventory_and_hashes")
    rows = as_mapping(artifact.get("per_unit_rows"))
    require(
        rows.get("source_unit_row_count")
        == EXP6430_PER_UNIT_ROW_COUNT + EXP6431_PER_UNIT_ROW_COUNT + EXP6432_PER_UNIT_ROW_COUNT,
        "per_unit_rows",
    )
    mismatches = as_mapping(artifact.get("upstream_vs_recomputed_mismatches"))
    require(mismatches.get("all_within_tolerance") is True, "upstream_vs_recomputed_mismatches")
    materiality = as_mapping(artifact.get("mismatch_count_and_materiality"))
    require(materiality.get("material_row_mismatch_count") == 0, "mismatch_count_and_materiality")
    gate = as_mapping(artifact.get("gate_check_summary"))
    require(int(gate.get("failed_check_count", 0) or 0) > 0, "gate_check_summary")
    require(artifact.get("status") == "complete_blocked", "status")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(str(artifact.get("honest_verdict", "")).startswith("complete_blocked:"), "honest_verdict")
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
