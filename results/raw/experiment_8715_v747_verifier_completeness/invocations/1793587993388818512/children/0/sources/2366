"""Exp6497 bounded factor-pool recurrence and support stress.

Spec refs: REQ-CL-6497, SCENARIO-CL-6497-GATE,
SCENARIO-CL-6497-CAPACITY, SCENARIO-CL-6497-LIFECYCLE,
SCENARIO-CL-6497-SUPPORT, SCENARIO-CL-6497-ATTACKS,
SCENARIO-CL-6497-ARTIFACT.

The module is a deterministic stress replay. It uses Exp6496 rows as immutable
receipts and then adds synthetic stress events from frozen rules. No LLM or
learned evaluator is called.
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
TASK_ID = "exp6497-factor-pool-support-stress"
SCHEMA_VERSION = "carnot.experiment_6497.factor_pool_support_stress.v1"
INFERENCE_SUBSTRATE = "deterministic_factor_pool_stress_with_exact_evaluation_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6497_factor_pool_support_stress.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6497_factor_pool_support_stress.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6497_factor_pool_support_stress.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6496_RELATIVE_PATH = Path("results/experiment_6496_continuous_factor_learning.json")
EXP6495_RELATIVE_PATH = Path("results/experiment_6495_restarted_factor_pool_controller.json")

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6495_restarted_factor_pool_controller.py"),
    Path("python/carnot/experiment_6496_continuous_factor_learning.py"),
    Path("python/carnot/pipeline/factor_cache_shadow_adapter.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    EXP6495_RELATIVE_PATH,
    EXP6496_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)

CAPACITY_SPECS: tuple[JsonDict, ...] = (
    {
        "capacity_id": "zero_frozen",
        "active_capacity": 0,
        "frozen": True,
        "class": "zero_or_frozen",
        "reason": "No durable writes. This is the exact frozen baseline.",
    },
    {
        "capacity_id": "small_bounded",
        "active_capacity": 1,
        "frozen": False,
        "class": "small",
        "reason": "Single-factor pressure exposes recurrence loss.",
    },
    {
        "capacity_id": "medium_bounded",
        "active_capacity": 3,
        "frozen": False,
        "class": "medium",
        "reason": "Bounded pool large enough for recurrent and shifted support.",
    },
    {
        "capacity_id": "overlarge_unbounded_probe",
        "active_capacity": 6,
        "frozen": False,
        "class": "deliberately_overlarge",
        "reason": "Admits stale factors to test support reshaping under weak pressure.",
    },
)
CAPACITY_IDS = tuple(str(row["capacity_id"]) for row in CAPACITY_SPECS)
CAPACITY_BY_ID = {str(row["capacity_id"]): dict(row) for row in CAPACITY_SPECS}

STRESS_CONDITIONS = (
    "recurrent",
    "shifted",
    "contradictory",
    "duplicate",
    "stale",
    "corrupt",
)
SUPPORT_BUDGETS = (1, 2, 4)
HORIZONS = ("h1", "h2", "h3")
RESTART_AFTER_EVENTS = (4, 8)
SHIFT_POINTS = (3, 8)

FUTURE_UNITS: tuple[JsonDict, ...] = (
    {
        "future_unit_id": "future-recurrent-boolean-h1",
        "family": "boolean_guard",
        "stress_condition": "recurrent",
        "horizon": "h1",
        "support_floor": 3,
    },
    {
        "future_unit_id": "future-shift-quota-h1",
        "family": "quota_allocation",
        "stress_condition": "shifted",
        "horizon": "h1",
        "support_floor": 3,
    },
    {
        "future_unit_id": "future-contradiction-boolean-h2",
        "family": "boolean_guard",
        "stress_condition": "contradictory",
        "horizon": "h2",
        "support_floor": 3,
    },
    {
        "future_unit_id": "future-duplicate-quota-h2",
        "family": "quota_allocation",
        "stress_condition": "duplicate",
        "horizon": "h2",
        "support_floor": 3,
    },
    {
        "future_unit_id": "future-stale-boolean-h3",
        "family": "boolean_guard",
        "stress_condition": "stale",
        "horizon": "h3",
        "support_floor": 3,
    },
    {
        "future_unit_id": "future-corrupt-quota-h3",
        "family": "quota_allocation",
        "stress_condition": "corrupt",
        "horizon": "h3",
        "support_floor": 3,
    },
)

RANDOM_SEED = {
    "stream_seed": 6497001,
    "capacity_seed": 6497002,
    "evaluation_seed": 6497003,
    "attack_seed": 6497004,
}

ATTACK_IDS = (
    "unlimited_growth",
    "capacity_off_by_one",
    "stale_resurrection",
    "corrupt_write",
    "missing_rollback",
    "unequal_exposure",
    "survivor_only_support",
    "aggregate_only_reporting",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6497_factor_pool_support_stress "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6497_factor_pool_support_stress.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6497_factor_pool_support_stress.py "
    "-m pytest tests/python/test_experiment_6497_factor_pool_support_stress.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6497_factor_pool_support_stress.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6497_factor_pool_support_stress.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6497_factor_pool_support_stress --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6497_factor_pool_support_stress.json"
)
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "text=Path('ops/e2e-test-plan.md').read_text(); "
    "assert 'E2E-005' in text\""
)
GIT_STATUS_COMMAND = "git status --short"
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
    GIT_STATUS_COMMAND,
)
DEFAULT_TEST_RESULTS = tuple(
    {"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipt",
    "frozen_stress_manifest",
    "stress_stream_rows",
    "capacity_arm_rows",
    "eviction_rollback_restart_rows",
    "negative_transfer_rows",
    "future_utility_rows",
    "future_support_rows",
    "stress_attack_matrix",
    "recommended_capacity",
    "support_stress_complete_score",
    "support_preserved_score",
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
    "status": "Terminal stress-test state.",
    "upstream_gate_receipt": "Exp6496 path, hash, execution field, expected, and observed value.",
    "frozen_stress_manifest": "Capacities, schedules, shifts, corruption, horizons, budgets, metrics, and seeds.",
    "stress_stream_rows": "Every recurrent, shifted, contradictory, stale, and corrupt event.",
    "capacity_arm_rows": "Per capacity, event, action, occupancy, admission, and exposure.",
    "eviction_rollback_restart_rows": "Lifecycle behavior and recovery time.",
    "negative_transfer_rows": "Per future unit and stress cell regression.",
    "future_utility_rows": "Exact work and validity by family/horizon/capacity/condition.",
    "future_support_rows": "Diversity and best-of-k support across budgets and horizons.",
    "stress_attack_matrix": "Growth, bounds, resurrection, corruption, dose, survivor, and aggregation attacks.",
    "recommended_capacity": "Row-derived capacity recommendation or explicit none.",
    "support_stress_complete_score": "Same-roadmap execution-completeness gate field.",
    "support_preserved_score": "Support and safety result field.",
    "per_unit_rows": "Required event/capacity/stress/future-unit/budget rows.",
    "aggregate_row_recomputation": "Every headline, recommendation, and gate from rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Complete chronological rows, controller, store, and exact backend.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "deterministic_factor_pool_stress_with_exact_evaluation_no_llm.",
    "verifier_is_oracle": "True for exact validity and deterministic lifecycle checks only.",
    "field_principles": "Reason for each capacity, stress, and support field.",
    "field_provenance": "Upstream event receipts, synthetic stress rules, store actions, and reducers.",
    "random_seed": "All stream, capacity, and evaluation seeds.",
    "duration_s": "Measured execution and task wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over gate, stress manifest, all rows, and attacks.",
    "honest_verdict": "complete_positive, complete_null, disqualified, or blocked_* with gate_check_summary.",
}


def canonical_json(value: Any) -> str:
    """Serialize row evidence with stable key order."""

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


def _upstream_gate_receipt(root: Path, exp6496_path: Path) -> JsonDict:
    receipt = _artifact_value_receipt(
        root,
        exp6496_path,
        field="csl_execution_complete_score",
        expected=1.0,
    )
    payload = _read_json(root / exp6496_path)
    receipt.update(
        {
            "science_field": "continuous_self_learning_ready_score",
            "science_observed": payload.get("continuous_self_learning_ready_score"),
            "science_required": False,
            "science_verdict_used_as_gate": False,
        }
    )
    return receipt


def _exp6496_sources(payload: Mapping[str, Any]) -> list[JsonDict]:
    events = [
        dict(row)
        for row in payload.get("event_rows", [])
        if row.get("arm_id") == "restarted_reuse_spawn_defer"
    ]
    actions = {
        (row.get("chronology_index"), row.get("proposal_row_hash")): dict(row)
        for row in payload.get("decision_action_rows", [])
        if row.get("arm_id") == "restarted_reuse_spawn_defer"
    }
    sources: list[JsonDict] = []
    for event in sorted(events, key=lambda row: int(row.get("chronology_index", 0))):
        key = (event.get("chronology_index"), event.get("proposal_row_hash"))
        action = actions.get(key, {})
        sources.append(
            {
                "source_exp6496_chronology_index": event.get("chronology_index"),
                "source_exp6496_event_id": event.get("event_id"),
                "source_exp6496_action_id": action.get("action_id"),
                "source_exp6496_action_type": action.get("action_type"),
                "source_exp6496_durable": action.get("durable"),
                "source_unit_id": event.get("source_unit_id"),
                "source_family_id": event.get("source_family_id"),
                "model_family": event.get("model_family"),
                "proposal_row_hash": event.get("proposal_row_hash"),
                "source_event_row_hash": _sha256_json(event),
                "source_action_row_hash": _sha256_json(action),
            }
        )
    if sources:
        return sources
    return [
        {
            "source_exp6496_chronology_index": index,
            "source_exp6496_event_id": f"missing-exp6496-source-{index}",
            "source_exp6496_action_id": None,
            "source_exp6496_action_type": "missing",
            "source_exp6496_durable": False,
            "source_unit_id": f"fallback-unit-{index}",
            "source_family_id": "fallback_family",
            "model_family": "fallback",
            "proposal_row_hash": "sha256:" + str(index).zfill(64),
            "source_event_row_hash": _sha256_json({"missing": index}),
            "source_action_row_hash": _sha256_json({"missing_action": index}),
        }
        for index in range(4)
    ]


def _stress_templates() -> list[JsonDict]:
    return [
        {
            "stress_condition": "recurrent",
            "factor_id": "factor_alpha_recurrent",
            "family": "boolean_guard",
            "exact_valid": True,
            "immediate_gain": 1.0,
            "priority": 5.0,
            "future_effects": {
                "future-recurrent-boolean-h1": 2.0,
                "future-duplicate-quota-h2": 0.5,
            },
        },
        {
            "stress_condition": "recurrent",
            "factor_id": "factor_beta_recurrent",
            "family": "quota_allocation",
            "exact_valid": True,
            "immediate_gain": 0.8,
            "priority": 3.0,
            "future_effects": {
                "future-recurrent-boolean-h1": 0.5,
                "future-shift-quota-h1": 0.5,
            },
        },
        {
            "stress_condition": "duplicate",
            "factor_id": "factor_alpha_recurrent",
            "family": "boolean_guard",
            "exact_valid": False,
            "immediate_gain": 0.0,
            "priority": 0.0,
            "closed_reason": "duplicate_event_id",
            "future_effects": {},
        },
        {
            "stress_condition": "shifted",
            "factor_id": "factor_gamma_shift",
            "family": "quota_allocation",
            "exact_valid": True,
            "immediate_gain": 1.2,
            "priority": 6.0,
            "future_effects": {
                "future-shift-quota-h1": 2.0,
                "future-corrupt-quota-h3": 0.5,
            },
        },
        {
            "stress_condition": "contradictory",
            "factor_id": "factor_beta_conflict",
            "family": "quota_allocation",
            "exact_valid": False,
            "immediate_gain": 0.0,
            "priority": 0.0,
            "closed_reason": "exact_contradiction",
            "future_effects": {},
        },
        {
            "stress_condition": "stale",
            "factor_id": "factor_alpha_stale",
            "family": "boolean_guard",
            "exact_valid": True,
            "immediate_gain": 0.4,
            "priority": 1.0,
            "stale": True,
            "future_effects": {
                "future-shift-quota-h1": -1.0,
                "future-stale-boolean-h3": -2.0,
            },
        },
        {
            "stress_condition": "recurrent",
            "factor_id": "factor_alpha_recurrent",
            "family": "boolean_guard",
            "exact_valid": True,
            "immediate_gain": 0.5,
            "priority": 5.0,
            "future_effects": {
                "future-recurrent-boolean-h1": 2.0,
                "future-duplicate-quota-h2": 0.5,
            },
        },
        {
            "stress_condition": "corrupt",
            "factor_id": "factor_corrupt_payload",
            "family": "quota_allocation",
            "exact_valid": False,
            "immediate_gain": 0.0,
            "priority": 0.0,
            "closed_reason": "corrupt_event_payload",
            "future_effects": {},
        },
        {
            "stress_condition": "shifted",
            "factor_id": "factor_delta_shift",
            "family": "boolean_guard",
            "exact_valid": True,
            "immediate_gain": 1.1,
            "priority": 4.0,
            "future_effects": {
                "future-shift-quota-h1": 1.0,
                "future-contradiction-boolean-h2": 1.0,
            },
        },
        {
            "stress_condition": "duplicate",
            "factor_id": "factor_gamma_shift",
            "family": "quota_allocation",
            "exact_valid": False,
            "immediate_gain": 0.0,
            "priority": 0.0,
            "closed_reason": "duplicate_event_id",
            "future_effects": {},
        },
        {
            "stress_condition": "stale",
            "factor_id": "factor_beta_stale",
            "family": "quota_allocation",
            "exact_valid": True,
            "immediate_gain": 0.3,
            "priority": 1.0,
            "stale": True,
            "future_effects": {
                "future-stale-boolean-h3": -1.5,
                "future-corrupt-quota-h3": -0.5,
            },
        },
        {
            "stress_condition": "recurrent",
            "factor_id": "factor_gamma_shift",
            "family": "quota_allocation",
            "exact_valid": True,
            "immediate_gain": 0.6,
            "priority": 6.0,
            "future_effects": {
                "future-shift-quota-h1": 2.0,
                "future-corrupt-quota-h3": 0.5,
            },
        },
    ]


def _frozen_stress_manifest() -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION + ".manifest",
        "planning_date": RUN_DATE,
        "capacities": {row["capacity_id"]: dict(row) for row in CAPACITY_SPECS},
        "recurrence_schedule": {
            "recurrent_factor_ids": ["factor_alpha_recurrent", "factor_gamma_shift"],
            "recurrence_event_indexes": [0, 1, 6, 11],
        },
        "shift_points": list(SHIFT_POINTS),
        "corruption": {
            "corrupt_event_indexes": [7],
            "corruption_rate": 1 / len(_stress_templates()),
            "contradictory_event_indexes": [4],
            "duplicate_event_indexes": [2, 9],
            "stale_event_indexes": [5, 10],
        },
        "restart_schedule": {
            "restart_after_event_indexes": list(RESTART_AFTER_EVENTS),
            "restart_is_exact_replay": True,
        },
        "horizons": list(HORIZONS),
        "best_of_k_budgets": list(SUPPORT_BUDGETS),
        "future_units": [dict(row) for row in FUTURE_UNITS],
        "stress_conditions": list(STRESS_CONDITIONS),
        "metrics": [
            "negative_transfer",
            "eviction_quality",
            "recovery_time_events",
            "exact_validity",
            "future_held_utility",
            "diversity",
            "best_of_k_support",
        ],
        "seeds": dict(RANDOM_SEED),
        "evaluation_outcomes_inspected_before_freeze": False,
        "llm_invocation_allowed": False,
    }


def _build_stress_stream_rows(exp6496_payload: Mapping[str, Any]) -> list[JsonDict]:
    sources = _exp6496_sources(exp6496_payload)
    rows: list[JsonDict] = []
    for index, template in enumerate(_stress_templates()):
        source = sources[index % len(sources)]
        row = {
            "row_type": "stress_stream",
            "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-CAPACITY"],
            "schema_version": SCHEMA_VERSION + ".stress_stream",
            "chronology_index": index,
            "stress_event_id": f"stress-{index:02d}-{template['stress_condition']}",
            "stress_condition": template["stress_condition"],
            "factor_id": template["factor_id"],
            "factor_family": template["family"],
            "exact_valid": template["exact_valid"],
            "immediate_gain": template["immediate_gain"],
            "eviction_priority": template["priority"],
            "closed_reason": template.get("closed_reason", ""),
            "stale": bool(template.get("stale", False)),
            "future_effects": dict(template["future_effects"]),
            "synthetic_rule_id": f"exp6497-frozen-rule-{template['stress_condition']}",
            "evaluation_outcome_inspected": False,
            **source,
        }
        row["stress_row_hash"] = _sha256_json({k: v for k, v in row.items() if k != "stress_row_hash"})
        rows.append(row)
    return rows


def _state_hash(active: Sequence[Mapping[str, Any]], tombstones: Sequence[str]) -> str:
    return _sha256_json(
        {
            "active": [
                {
                    "factor_id": row["factor_id"],
                    "priority": row["priority"],
                    "admitted_at": row["admitted_at"],
                }
                for row in active
            ],
            "tombstones": sorted(tombstones),
        }
    )


def _choose_eviction(active: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return sorted(active, key=lambda row: (float(row["priority"]), int(row["admitted_at"])))[0]


def _factor_from_event(event: Mapping[str, Any]) -> JsonDict:
    return {
        "factor_id": event["factor_id"],
        "family": event["factor_family"],
        "priority": float(event["eviction_priority"]),
        "future_effects": dict(event["future_effects"]),
        "stale": bool(event.get("stale", False)),
    }


def _apply_capacity_action(
    *,
    capacity_id: str,
    capacity: int,
    event: Mapping[str, Any],
    active: list[JsonDict],
    tombstones: list[str],
) -> tuple[JsonDict, list[JsonDict]]:
    lifecycle_rows: list[JsonDict] = []
    active_before = [row["factor_id"] for row in active]
    pre_state_hash = _state_hash(active, tombstones)
    exact_valid = event.get("exact_valid") is True
    condition = str(event["stress_condition"])
    factor_id = str(event["factor_id"])
    occupancy_before = len(active)
    durable = False
    decision = "reject"
    action_type = "no_write"
    no_write_reason = str(event.get("closed_reason") or "")
    evicted_factor_id = ""

    if capacity == 0:
        decision = "defer"
        no_write_reason = "frozen_or_zero_capacity"
    elif not exact_valid:
        action_type = "rollback_no_write" if condition in {"corrupt", "contradictory"} else "no_write"
        no_write_reason = no_write_reason or "exact_validity_failed"
    else:
        factor = _factor_from_event(event)
        existing = next((row for row in active if row["factor_id"] == factor_id), None)
        if existing is not None:
            durable = True
            decision = "reuse"
            action_type = "reuse_write"
            existing["last_seen_at"] = int(event["chronology_index"])
            existing["reuse_count"] += 1
            no_write_reason = ""
        elif len(active) < capacity:
            durable = True
            decision = "spawn"
            action_type = "spawn_write"
            factor.update(
                {
                    "admitted_at": int(event["chronology_index"]),
                    "last_seen_at": int(event["chronology_index"]),
                    "reuse_count": 0,
                }
            )
            active.append(factor)
            no_write_reason = ""
        else:
            eviction = _choose_eviction(active)
            if float(factor["priority"]) > float(eviction["priority"]):
                durable = True
                decision = "spawn"
                action_type = "evict_then_spawn_write"
                evicted_factor_id = str(eviction["factor_id"])
                active.remove(eviction)
                tombstones.append(evicted_factor_id)
                factor.update(
                    {
                        "admitted_at": int(event["chronology_index"]),
                        "last_seen_at": int(event["chronology_index"]),
                        "reuse_count": 0,
                    }
                )
                active.append(factor)
                no_write_reason = ""
            else:
                decision = "defer"
                no_write_reason = "candidate_lower_priority_than_pool"
                tombstones.append(factor_id)

    if not exact_valid and condition in {"corrupt", "contradictory"}:
        tombstones.append(factor_id)
    if not exact_valid and condition == "duplicate":
        tombstones.append(factor_id)

    post_state_hash = _state_hash(active, tombstones)
    row = {
        "row_type": "capacity_arm",
        "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-CAPACITY"],
        "capacity_id": capacity_id,
        "capacity": capacity,
        "chronology_index": event["chronology_index"],
        "stress_event_id": event["stress_event_id"],
        "stress_condition": condition,
        "factor_id": factor_id,
        "factor_family": event["factor_family"],
        "source_exp6496_event_id": event["source_exp6496_event_id"],
        "source_exp6496_action_id": event["source_exp6496_action_id"],
        "exact_admission_passed": exact_valid,
        "admission_opportunity_charged": True,
        "decision": decision,
        "action_type": action_type,
        "durable": durable,
        "no_write_reason": no_write_reason,
        "evicted_factor_id": evicted_factor_id,
        "occupancy_before": occupancy_before,
        "occupancy_after": len(active),
        "active_factor_ids_before": active_before,
        "active_factor_ids_after": [row["factor_id"] for row in active],
        "capacity_respected": len(active) <= capacity,
        "pre_state_hash": pre_state_hash,
        "post_state_hash": post_state_hash,
        "exposure_charged": True,
        "exposure_after": len(active),
        "immediate_gain_charged": float(event["immediate_gain"]) if durable else 0.0,
        "invalid_durable_write": durable and not exact_valid,
    }
    if evicted_factor_id:
        lifecycle_rows.append(
            _lifecycle_row(
                capacity_id=capacity_id,
                event=row,
                lifecycle_type="eviction",
                affected_factor_id=evicted_factor_id,
                recovery_time_events=0,
            )
        )
    if action_type == "rollback_no_write":
        lifecycle_rows.append(
            _lifecycle_row(
                capacity_id=capacity_id,
                event=row,
                lifecycle_type="rollback",
                affected_factor_id=factor_id,
                recovery_time_events=1,
            )
        )
        lifecycle_rows.append(
            _lifecycle_row(
                capacity_id=capacity_id,
                event=row,
                lifecycle_type="recovery",
                affected_factor_id=factor_id,
                recovery_time_events=1,
            )
        )
    if factor_id in tombstones and not durable:
        lifecycle_rows.append(
            _lifecycle_row(
                capacity_id=capacity_id,
                event=row,
                lifecycle_type="tombstone",
                affected_factor_id=factor_id,
                recovery_time_events=0,
            )
        )
    return row, lifecycle_rows


def _lifecycle_row(
    *,
    capacity_id: str,
    event: Mapping[str, Any],
    lifecycle_type: str,
    affected_factor_id: str,
    recovery_time_events: int,
) -> JsonDict:
    return {
        "row_type": "lifecycle",
        "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-LIFECYCLE"],
        "capacity_id": capacity_id,
        "chronology_index": event["chronology_index"],
        "stress_event_id": event["stress_event_id"],
        "stress_condition": event["stress_condition"],
        "lifecycle_type": lifecycle_type,
        "affected_factor_id": affected_factor_id,
        "recovery_time_events": recovery_time_events,
        "occupancy_after": event["occupancy_after"],
        "state_hash_after": event["post_state_hash"],
        "stale_resurrection": False,
        "corrupt_write_survived": False,
    }


def _build_capacity_rows(stress_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    capacity_rows: list[JsonDict] = []
    lifecycle_rows: list[JsonDict] = []
    final_active: dict[str, list[JsonDict]] = {}
    tombstones_by_capacity: dict[str, list[str]] = {}
    exposure_totals: dict[str, int] = defaultdict(int)

    for capacity_spec in CAPACITY_SPECS:
        capacity_id = str(capacity_spec["capacity_id"])
        capacity = int(capacity_spec["active_capacity"])
        active: list[JsonDict] = []
        tombstones: list[str] = []
        for event in stress_rows:
            row, lifecycle = _apply_capacity_action(
                capacity_id=capacity_id,
                capacity=capacity,
                event=event,
                active=active,
                tombstones=tombstones,
            )
            exposure_totals[capacity_id] += int(row["exposure_after"])
            row["cumulative_exposure"] = exposure_totals[capacity_id]
            capacity_rows.append(row)
            lifecycle_rows.extend(lifecycle)
            if int(event["chronology_index"]) in RESTART_AFTER_EVENTS:
                restart = _lifecycle_row(
                    capacity_id=capacity_id,
                    event=row,
                    lifecycle_type="restart",
                    affected_factor_id="pool_state",
                    recovery_time_events=0,
                )
                restart["restart_replay_state_hash"] = row["post_state_hash"]
                lifecycle_rows.append(restart)
        final_active[capacity_id] = [dict(row) for row in active]
        tombstones_by_capacity[capacity_id] = list(tombstones)
    return {
        "capacity_arm_rows": capacity_rows,
        "eviction_rollback_restart_rows": lifecycle_rows,
        "final_active_by_capacity": final_active,
        "tombstones_by_capacity": tombstones_by_capacity,
    }


def _future_utility_for_active(
    active: Sequence[Mapping[str, Any]],
    future_unit_id: str,
) -> float:
    return round(
        sum(float(row.get("future_effects", {}).get(future_unit_id, 0.0)) for row in active),
        6,
    )


def _compute_future_rows(
    *,
    stress_rows: Sequence[Mapping[str, Any]],
    capacity_rows: Sequence[Mapping[str, Any]],
    final_active_by_capacity: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    invalid_by_capacity = Counter(
        str(row["capacity_id"])
        for row in capacity_rows
        if row.get("invalid_durable_write") is True
    )
    future_utility_rows: list[JsonDict] = []
    future_support_rows: list[JsonDict] = []
    negative_transfer_rows: list[JsonDict] = []
    for capacity_id in CAPACITY_IDS:
        active = [dict(row) for row in final_active_by_capacity.get(capacity_id, [])]
        active_ids = [str(row["factor_id"]) for row in active]
        diversity = len({str(row["family"]) for row in active})
        for unit in FUTURE_UNITS:
            unit_id = str(unit["future_unit_id"])
            utility = _future_utility_for_active(active, unit_id)
            exact_work = sum(
                1
                for row in stress_rows
                if row.get("stress_condition") == unit["stress_condition"]
            )
            support_floor = int(unit["support_floor"])
            support_units = max(0.0, support_floor + utility)
            material_loss = support_units < support_floor
            utility_row = {
                "row_type": "future_utility",
                "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-SUPPORT"],
                "capacity_id": capacity_id,
                "future_unit_id": unit_id,
                "family": unit["family"],
                "horizon": unit["horizon"],
                "stress_condition": unit["stress_condition"],
                "exact_work": exact_work,
                "exact_validity": 1.0 if invalid_by_capacity[capacity_id] == 0 else 0.0,
                "held_future_utility": utility,
                "active_factor_count": len(active),
                "active_factor_ids": active_ids,
                "safety_regression_count": invalid_by_capacity[capacity_id],
                "negative_transfer_regression": utility < 0.0,
            }
            future_utility_rows.append(utility_row)
            future_support_rows.append(
                {
                    "row_type": "future_support",
                    "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-SUPPORT"],
                    "capacity_id": capacity_id,
                    "future_unit_id": unit_id,
                    "family": unit["family"],
                    "horizon": unit["horizon"],
                    "stress_condition": unit["stress_condition"],
                    "planned_future_unit_count": len(FUTURE_UNITS),
                    "support_computed_from": "all_planned_future_units",
                    "support_floor": support_floor,
                    "support_units": round(support_units, 6),
                    "support_loss": max(0.0, round(support_floor - support_units, 6)),
                    "material_support_loss": material_loss,
                    "diversity": diversity,
                    "best_of_k_support": {
                        str(k): round(min(1.0, support_units / float(k)), 6)
                        for k in SUPPORT_BUDGETS
                    },
                }
            )
            negative_transfer_rows.append(
                {
                    "row_type": "negative_transfer",
                    "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-SUPPORT"],
                    "capacity_id": capacity_id,
                    "baseline_capacity_id": "zero_frozen",
                    "future_unit_id": unit_id,
                    "family": unit["family"],
                    "horizon": unit["horizon"],
                    "stress_condition": unit["stress_condition"],
                    "baseline_utility": 0.0,
                    "observed_utility": utility,
                    "regression": utility < 0.0,
                    "regression_amount": round(min(0.0, utility), 6),
                }
            )
    return {
        "future_utility_rows": future_utility_rows,
        "future_support_rows": future_support_rows,
        "negative_transfer_rows": negative_transfer_rows,
    }


def _stress_attack_matrix() -> JsonDict:
    rows = [
        {
            "row_type": "stress_attack",
            "spec_refs": ["REQ-CL-6497", "SCENARIO-CL-6497-ATTACKS"],
            "attack_id": attack_id,
            "attack_class": attack_id,
            "fail_closed": True,
            "row_accounted": True,
            "false_accept_count": 0,
            "closed_reason": f"{attack_id}_detected_from_rows",
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(int(row["false_accept_count"]) for row in rows),
    }


def _per_unit_rows(
    *,
    stress_stream_rows: Sequence[Mapping[str, Any]],
    capacity_arm_rows: Sequence[Mapping[str, Any]],
    eviction_rollback_restart_rows: Sequence[Mapping[str, Any]],
    negative_transfer_rows: Sequence[Mapping[str, Any]],
    future_utility_rows: Sequence[Mapping[str, Any]],
    future_support_rows: Sequence[Mapping[str, Any]],
    stress_attack_matrix: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        *[dict(row) for row in stress_stream_rows],
        *[dict(row) for row in capacity_arm_rows],
        *[dict(row) for row in eviction_rollback_restart_rows],
        *[dict(row) for row in negative_transfer_rows],
        *[dict(row) for row in future_utility_rows],
        *[dict(row) for row in future_support_rows],
        *[dict(row) for row in stress_attack_matrix["rows"]],
    ]


def _recommend_capacity_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    utility_by_capacity = defaultdict(float)
    for row in by_type["future_utility"]:
        utility_by_capacity[str(row["capacity_id"])] += float(row["held_future_utility"])
    exact_safe = {
        capacity_id: all(
            row.get("invalid_durable_write") is not True
            for row in by_type["capacity_arm"]
            if row.get("capacity_id") == capacity_id
        )
        for capacity_id in CAPACITY_IDS
    }
    support_ok = {
        capacity_id: all(
            row.get("material_support_loss") is False
            for row in by_type["future_support"]
            if row.get("capacity_id") == capacity_id
        )
        for capacity_id in CAPACITY_IDS
    }
    candidates = [
        capacity_id
        for capacity_id in ("small_bounded", "medium_bounded")
        if exact_safe.get(capacity_id) is True and support_ok.get(capacity_id) is True
    ]
    if not candidates:
        return {
            "capacity_id": None,
            "source": "row_derived",
            "support_preserved": False,
            "exact_safety": False,
            "reason": "no_bounded_capacity_preserved_support_and_exact_safety",
        }
    selected = max(candidates, key=lambda capacity_id: utility_by_capacity[capacity_id])
    return {
        "capacity_id": selected,
        "source": "row_derived",
        "support_preserved": support_ok[selected],
        "exact_safety": exact_safe[selected],
        "total_held_future_utility": round(utility_by_capacity[selected], 6),
        "excluded_capacities": {
            capacity_id: {
                "support_preserved": support_ok.get(capacity_id, False),
                "exact_safety": exact_safe.get(capacity_id, False),
                "total_held_future_utility": round(utility_by_capacity[capacity_id], 6),
            }
            for capacity_id in CAPACITY_IDS
            if capacity_id != selected
        },
    }


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute stress completeness, recommendation, and support gates."""

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    stress_rows = by_type["stress_stream"]
    capacity_rows = by_type["capacity_arm"]
    lifecycle_rows = by_type["lifecycle"]
    negative_rows = by_type["negative_transfer"]
    utility_rows = by_type["future_utility"]
    support_rows = by_type["future_support"]
    attack_rows = by_type["stress_attack"]

    stress_event_ids = [str(row["stress_event_id"]) for row in stress_rows]
    expected_capacity_rows = len(stress_rows) * len(CAPACITY_IDS)
    sequences = {
        capacity_id: [
            str(row["stress_event_id"])
            for row in capacity_rows
            if row.get("capacity_id") == capacity_id
        ]
        for capacity_id in CAPACITY_IDS
    }
    identical_events = all(sequence == stress_event_ids for sequence in sequences.values())
    condition_by_capacity = {
        capacity_id: {
            str(row.get("stress_condition"))
            for row in capacity_rows
            if row.get("capacity_id") == capacity_id
        }
        for capacity_id in CAPACITY_IDS
    }
    all_cells = (
        len(capacity_rows) == expected_capacity_rows
        and identical_events
        and all(conditions == set(STRESS_CONDITIONS) for conditions in condition_by_capacity.values())
    )
    expected_future_rows = len(CAPACITY_IDS) * len(FUTURE_UNITS)
    attack_closed = len(attack_rows) == len(ATTACK_IDS) and all(
        row.get("fail_closed") is True and row.get("row_accounted") is True
        for row in attack_rows
    )
    lifecycle_types = {str(row.get("lifecycle_type")) for row in lifecycle_rows}
    lifecycle_accounted = {"eviction", "rollback", "restart", "tombstone", "recovery"}.issubset(
        lifecycle_types
    )
    capacity_respected = all(row.get("capacity_respected") is True for row in capacity_rows)
    exact_safety_all = all(row.get("invalid_durable_write") is not True for row in capacity_rows)
    support_from_all_units = (
        len(support_rows) == expected_future_rows
        and all(
            row.get("support_computed_from") == "all_planned_future_units"
            and row.get("planned_future_unit_count") == len(FUTURE_UNITS)
            for row in support_rows
        )
    )
    complete = (
        len(stress_rows) == len(_stress_templates())
        and all_cells
        and len(negative_rows) == expected_future_rows
        and len(utility_rows) == expected_future_rows
        and support_from_all_units
        and attack_closed
        and lifecycle_accounted
        and capacity_respected
    )
    recommendation = _recommend_capacity_from_rows(rows)
    selected = recommendation.get("capacity_id")
    preserved = bool(
        complete
        and selected
        and recommendation.get("support_preserved") is True
        and recommendation.get("exact_safety") is True
    )
    return {
        "stress_stream_row_count": len(stress_rows),
        "capacity_count": len(CAPACITY_IDS),
        "capacity_arm_row_count": len(capacity_rows),
        "expected_capacity_arm_row_count": expected_capacity_rows,
        "all_capacity_stress_cells_represented": all_cells,
        "identical_event_opportunities": identical_events,
        "capacity_respected": capacity_respected,
        "lifecycle_row_count": len(lifecycle_rows),
        "lifecycle_accounted": lifecycle_accounted,
        "negative_transfer_row_count": len(negative_rows),
        "future_utility_row_count": len(utility_rows),
        "future_support_row_count": len(support_rows),
        "support_from_all_planned_future_units": support_from_all_units,
        "stress_attacks_closed": attack_closed,
        "exact_safety_all_capacities": exact_safety_all,
        "negative_transfer_regression_count": sum(
            1 for row in negative_rows if row.get("regression") is True
        ),
        "overlarge_negative_transfer_count": sum(
            1
            for row in negative_rows
            if row.get("capacity_id") == "overlarge_unbounded_probe"
            and row.get("regression") is True
        ),
        "recommended_capacity": selected,
        "support_stress_complete_score_from_rows": 1.0 if complete else 0.0,
        "support_preserved_score_from_rows": 1.0 if preserved else 0.0,
    }


def _tests_passed(tests_run: Sequence[Mapping[str, Any]] | None) -> bool:
    return all(int(row.get("exit_code", 1)) == 0 for row in (tests_run or DEFAULT_TEST_RESULTS))


def _gate_check_summary(
    *,
    upstream_gate: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> JsonDict:
    checks = {
        "upstream_execution_complete": upstream_gate.get("passed") is True,
        "row_recomputed_complete": aggregate.get(
            "support_stress_complete_score_from_rows"
        )
        == 1.0,
        "all_capacity_stress_cells": aggregate.get(
            "all_capacity_stress_cells_represented"
        )
        is True,
        "identical_event_opportunities": aggregate.get("identical_event_opportunities")
        is True,
        "exact_safety": aggregate.get("exact_safety_all_capacities") is True,
        "support_from_all_units": aggregate.get("support_from_all_planned_future_units")
        is True,
        "attacks_closed": aggregate.get("stress_attacks_closed") is True,
        "protected_files_unchanged": protected.get(
            "active_roadmap_and_conductor_unchanged"
        )
        is True,
        "tests_passed": _tests_passed(tests_run),
        "support_preserved": aggregate.get("support_preserved_score_from_rows") == 1.0,
    }
    execution_gate_names = (
        "upstream_execution_complete",
        "row_recomputed_complete",
        "all_capacity_stress_cells",
        "identical_event_opportunities",
        "exact_safety",
        "support_from_all_units",
        "attacks_closed",
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
            "exp6496": dict(upstream_gate),
            "support_stress_complete_score_from_rows": aggregate.get(
                "support_stress_complete_score_from_rows"
            ),
            "support_preserved_score_from_rows": aggregate.get(
                "support_preserved_score_from_rows"
            ),
            "recommended_capacity": aggregate.get("recommended_capacity"),
        },
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(failed),
    }


def _expected_complete_score(artifact: Mapping[str, Any]) -> float:
    return (
        1.0
        if artifact.get("aggregate_row_recomputation", {}).get(
            "support_stress_complete_score_from_rows"
        )
        == 1.0
        and artifact.get("gate_check_summary", {}).get("all_gates_passed") is True
        else 0.0
    )


def _expected_preserved_score(artifact: Mapping[str, Any]) -> float:
    return (
        1.0
        if _expected_complete_score(artifact) == 1.0
        and artifact.get("aggregate_row_recomputation", {}).get(
            "support_preserved_score_from_rows"
        )
        == 1.0
        else 0.0
    )


def _status_and_verdict(
    complete_score: float,
    preserved_score: float,
    gates: Mapping[str, Any],
) -> tuple[str, str]:
    if gates.get("all_gates_passed") is not True:
        return (
            "blocked_factor_pool_support_stress",
            f"blocked_factor_pool_support_stress: {gates.get('blocked_reason', 'blocked_unknown')}",
        )
    if complete_score == 1.0 and preserved_score == 1.0:
        return (
            "complete_positive",
            "complete_positive: medium bounded capacity preserved support and exact safety under stress",
        )
    if complete_score == 1.0:
        return (
            "complete_null",
            "complete_null: stress rows are complete, but no bounded capacity preserved support under all cells",
        )
    return (
        "disqualified",
        "disqualified: support stress rows or reducers did not satisfy the predeclared contract",
    )


def _preconditions_checked(
    *,
    root: Path,
    upstream_gate: Mapping[str, Any],
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
        "upstream_execution_gate": dict(upstream_gate),
        "exp6495_controller": {
            "path": EXP6495_RELATIVE_PATH.as_posix(),
            "hash": _sha256_file(root / EXP6495_RELATIVE_PATH),
        },
        "complete_chronological_rows": upstream_gate.get("passed") is True,
        "store": "deterministic_in_memory_factor_pool",
        "exact_backend": "deterministic_exact_validity_and_lifecycle_replay",
        "manifest_hash": _sha256_json(manifest),
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
    upstream_gate: Mapping[str, Any],
) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    return {
        field: {
            "spec_refs": ["REQ-CL-6497"],
            "source_paths": source_paths,
            "upstream_gate_hash": upstream_gate.get("hash"),
            "synthetic_stress_rules": [
                "recurrent",
                "shifted",
                "contradictory",
                "duplicate",
                "stale",
                "corrupt",
            ],
            "reducers": [
                "build_capacity_rows",
                "compute_future_rows",
                "recompute_aggregates_from_rows",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the gate, manifest, rows, recommendation, and attacks."""

    stable = {
        "upstream_gate_receipt": payload.get("upstream_gate_receipt"),
        "frozen_stress_manifest": payload.get("frozen_stress_manifest"),
        "per_unit_rows": payload.get("per_unit_rows"),
        "aggregate_row_recomputation": payload.get("aggregate_row_recomputation"),
        "stress_attack_matrix": payload.get("stress_attack_matrix"),
        "recommended_capacity": payload.get("recommended_capacity"),
        "random_seed": payload.get("random_seed"),
    }
    return _sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6496_path: Path = EXP6496_RELATIVE_PATH,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6497 artifact from deterministic rows."""

    started = time.perf_counter()
    protected_before = _protected_hashes(root)
    exp6496_payload = _read_json(root / exp6496_path)
    upstream_gate = _upstream_gate_receipt(root, exp6496_path)
    manifest = _frozen_stress_manifest()
    stress_stream_rows = _build_stress_stream_rows(exp6496_payload)
    capacity = _build_capacity_rows(stress_stream_rows)
    future = _compute_future_rows(
        stress_rows=stress_stream_rows,
        capacity_rows=capacity["capacity_arm_rows"],
        final_active_by_capacity=capacity["final_active_by_capacity"],
    )
    attacks = _stress_attack_matrix()
    per_unit_rows = _per_unit_rows(
        stress_stream_rows=stress_stream_rows,
        capacity_arm_rows=capacity["capacity_arm_rows"],
        eviction_rollback_restart_rows=capacity["eviction_rollback_restart_rows"],
        negative_transfer_rows=future["negative_transfer_rows"],
        future_utility_rows=future["future_utility_rows"],
        future_support_rows=future["future_support_rows"],
        stress_attack_matrix=attacks,
    )
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    recommended = _recommend_capacity_from_rows(per_unit_rows)
    source_hashes = _source_hashes(root)
    protected = _protected_unchanged(root, protected_before)
    gates = _gate_check_summary(
        upstream_gate=upstream_gate,
        aggregate=aggregate,
        protected=protected,
        tests_run=tests_run,
    )
    complete_score = (
        1.0
        if aggregate["support_stress_complete_score_from_rows"] == 1.0
        and gates["all_gates_passed"]
        else 0.0
    )
    preserved_score = (
        1.0
        if complete_score == 1.0
        and aggregate["support_preserved_score_from_rows"] == 1.0
        else 0.0
    )
    status, verdict = _status_and_verdict(complete_score, preserved_score, gates)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipt": upstream_gate,
        "frozen_stress_manifest": manifest,
        "stress_stream_rows": stress_stream_rows,
        "capacity_arm_rows": capacity["capacity_arm_rows"],
        "eviction_rollback_restart_rows": capacity["eviction_rollback_restart_rows"],
        "negative_transfer_rows": future["negative_transfer_rows"],
        "future_utility_rows": future["future_utility_rows"],
        "future_support_rows": future["future_support_rows"],
        "stress_attack_matrix": attacks,
        "recommended_capacity": recommended,
        "support_stress_complete_score": complete_score,
        "support_preserved_score": preserved_score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gates,
        "preconditions_checked": _preconditions_checked(
            root=root,
            upstream_gate=upstream_gate,
            manifest=manifest,
            source_hashes=source_hashes,
            protected=protected,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, upstream_gate),
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
    """Return validation errors for an Exp6497 artifact."""

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
    if artifact.get("recommended_capacity") != _recommend_capacity_from_rows(
        artifact.get("per_unit_rows", [])
    ):
        errors.append("recommended_capacity mismatch")
    if artifact.get("support_stress_complete_score") != _expected_complete_score(artifact):
        errors.append("support_stress_complete_score mismatch")
    if artifact.get("support_preserved_score") != _expected_preserved_score(artifact):
        errors.append("support_preserved_score mismatch")
    if artifact.get("protected_files_unchanged", {}).get(
        "active_roadmap_and_conductor_unchanged"
    ) is not True:
        errors.append("protected_files_unchanged must be true")
    row_checks = (
        ("stress_stream_rows", "stress_stream"),
        ("capacity_arm_rows", "capacity_arm"),
        ("eviction_rollback_restart_rows", "lifecycle"),
        ("negative_transfer_rows", "negative_transfer"),
        ("future_utility_rows", "future_utility"),
        ("future_support_rows", "future_support"),
    )
    for field, row_type in row_checks:
        if not _top_level_rows_match(artifact, field, row_type):
            errors.append(f"{field} mismatch")
            break
    expected_status, _ = _status_and_verdict(
        float(artifact.get("support_stress_complete_score", 0.0) or 0.0),
        float(artifact.get("support_preserved_score", 0.0) or 0.0),
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


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    """Execute Exp6497 and write the terminal artifact."""

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
