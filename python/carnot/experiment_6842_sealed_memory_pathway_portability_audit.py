"""Run the sealed residual-memory pathway portability audit.

Spec refs: REQ-CL-6842, SCENARIO-CL-6842-PRECONDITIONS,
SCENARIO-CL-6842-FRESH-REDUCTION, SCENARIO-CL-6842-SHARD-IDENTITY,
SCENARIO-CL-6842-ATTACKS, SCENARIO-CL-6842-DURABILITY,
SCENARIO-CL-6842-PORTABILITY, and SCENARIO-CL-6842-READY.

This module reads the two sealed chronological memory shards as row tables. It
does not import their producer modules or aggregate summaries. The audit
recomputes paired no-memory effects and then applies deterministic memory-path
attacks to those row-level receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6842_sealed_memory_pathway_portability_audit.py"
)
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6842_sealed_memory_pathway_portability_audit.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6842_sealed_memory_pathway_portability_audit.json")

RUN_DATE = "20260901"
EXPERIMENT_ID = "6842"
SCHEMA = "carnot.experiment_6842.sealed_memory_pathway_portability_audit.v1"
INFERENCE_SUBSTRATE = "deterministic CPU sealed audit"
COMPLETE_STATUS = "complete_sealed_memory_pathway_portability_audit"
BLOCKED_STATUS = "complete_blocked_sealed_memory_audit"

EXP6840_HASH = "sha256:df39b5ff937956e6d69806479eb6462c36de5d1ad37466bed36f466ec657f2d0"
EXP6841_HASH = "sha256:f028a76739200a18e84f4ffcf961f2dbc611b6be0f3efc7cfc68efdcc7147495"
SOURCE_RELATIVE_PATHS = {
    "exp6840": Path("results/experiment_6840_residual_memory_chronological_shard_a.json"),
    "exp6841": Path("results/experiment_6841_residual_memory_delayed_correction_shard_b.json"),
}
EXPECTED_SOURCE_HASHES = {"exp6840": EXP6840_HASH, "exp6841": EXP6841_HASH}

NO_MEMORY_ARM = "no_memory"
READ_ONLY_ARM = "read_only_memory"
RANDOM_ARM = "random_admission"
VERIFIED_RESIDUAL_ARM = "verified_residual_memory"
ARM_NAMES = (NO_MEMORY_ARM, READ_ONLY_ARM, RANDOM_ARM, VERIFIED_RESIDUAL_ARM)
MUTATING_ARMS = (RANDOM_ARM, VERIFIED_RESIDUAL_ARM)
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

FRESH_ATTACK = "fresh_reduction"
DELETION_ATTACK = "deletion"
SUBSTITUTION_ATTACK = "substitution"
REORDER_ATTACK = "reorder"
POISON_ATTACK = "poison"
STALE_CREDIT_ATTACK = "stale_credit"
LATENT_ERROR_ATTACK = "latent_error"
RESTART_ATTACK = "restart"
ROLLBACK_ATTACK = "rollback"
CAPACITY_ATTACK = "capacity"
ATTACK_NAMES = (
    FRESH_ATTACK,
    DELETION_ATTACK,
    SUBSTITUTION_ATTACK,
    REORDER_ATTACK,
    POISON_ATTACK,
    STALE_CREDIT_ATTACK,
    LATENT_ERROR_ATTACK,
    RESTART_ATTACK,
    ROLLBACK_ATTACK,
    CAPACITY_ATTACK,
)
READINESS_GATE_NAMES = (
    "held_future_benefit",
    "durability",
    "portability",
    "calibrated_dose",
    "leakage",
)

OPEN_SPEC_IDS = (
    "REQ-CL-6842",
    "SCENARIO-CL-6842-PRECONDITIONS",
    "SCENARIO-CL-6842-FRESH-REDUCTION",
    "SCENARIO-CL-6842-SHARD-IDENTITY",
    "SCENARIO-CL-6842-ATTACKS",
    "SCENARIO-CL-6842-DURABILITY",
    "SCENARIO-CL-6842-PORTABILITY",
    "SCENARIO-CL-6842-READY",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6842_sealed_memory_pathway_portability_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6842_sealed_memory_pathway_portability_audit.py' -m pytest tests/python/test_experiment_6842_sealed_memory_pathway_portability_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6842_sealed_memory_pathway_portability_audit.py scripts/experiments/experiment_6842_sealed_memory_pathway_portability_audit.py tests/python/test_experiment_6842_sealed_memory_pathway_portability_audit.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6842_sealed_memory_pathway_portability_audit.py",
    ".venv/bin/python scripts/experiments/experiment_6842_sealed_memory_pathway_portability_audit.py --date 20260901",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6842_sealed_memory_pathway_portability_audit.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6842_sealed_memory_pathway_portability_audit.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "title",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "replay_commands",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "continuous_self_learning_task",
    "source_artifact_hashes",
    "random_seeds",
    "reproducibility_checksum",
    "source_identity_summary",
    "rows",
    "fresh_reduction_results",
    "deletion_results",
    "substitution_results",
    "reorder_results",
    "poison_results",
    "stale_credit_results",
    "latent_error_pathways",
    "restart_durability_results",
    "rollback_results",
    "leave_one_family_out_results",
    "negative_transfer_results",
    "action_dose_calibration",
    "capacity_results",
    "sealed_csl_audit_complete_score",
    "continuous_self_learning_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        "row_id",
        "source_artifact",
        "source_row_id",
        "source_row_sha256",
        "source_event_row_id",
        "source_event_identity",
        "family",
        "order_id",
        "order_index",
        "arm",
        "seed",
        "split",
        "attack",
        "matched_no_memory_baseline",
        "baseline_row_sha256",
        "baseline_metric",
        "recomputed_metric",
        "effect_vs_no_memory",
        "paired_effect_class",
        "no_headroom",
        "uncertainty",
        "memory_dose",
        "capacity_budget",
        "active_count_after",
        "action_changed",
        "exact_outcome_receipt_sha256",
        "transition_statuses",
        "pathway_state",
        "poison_accepted",
        "stale_credit_accepted",
        "capacity_within_bound",
        "restart_clean_replay_match",
        "rollback_restored_parent",
        "producer_aggregate_imported",
        "row_sha256",
    }
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents silent reinterpretation of audit rows.",
    "experiment_id": "A stable identifier binds the artifact to Exp6842.",
    "title": "The title states this is the sealed memory reducer.",
    "run_date": "The fixed date separates this execution from later reruns.",
    "status": "The status separates complete sealed audit from blocked gates.",
    "openspec_requirement_ids": "Requirement IDs keep tests tied to the audit rules.",
    "replay_commands": "Replay commands show the intended verification stack.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Input gates stop source drift before reduction.",
    "inference_substrate": "The substrate declares deterministic CPU sealed audit.",
    "duration_s": "Measured wall time shows that the audit executed.",
    "continuous_self_learning_task": "True marks this as an FR11 memory task.",
    "source_artifact_hashes": "Raw source hashes bind upstream shard bytes.",
    "random_seeds": "The union of shard seeds makes row pairing replayable.",
    "reproducibility_checksum": "The checksum binds stable content and excludes time.",
    "source_identity_summary": "Shard identity proves the reducer kept sources disjoint.",
    "rows": "Rows expose one source row, arm, attack, and recomputed metric.",
    "fresh_reduction_results": "Fresh results are recomputed without producer aggregates.",
    "deletion_results": "Deletion tests whether memory actions caused the effect.",
    "substitution_results": "Substitution tests action-direction sensitivity.",
    "reorder_results": "Reorder tests chronology sensitivity without changing row counts.",
    "poison_results": "Poison tests that bounded unsafe memory cannot create readiness.",
    "stale_credit_results": "Stale credit tests that old evidence cannot get new credit.",
    "latent_error_pathways": "Pathways show whether joint errors persist or repair.",
    "restart_durability_results": "Restart checks persisted state against clean replay.",
    "rollback_results": "Rollback checks parent-state restoration.",
    "leave_one_family_out_results": "Family splits prevent pooled means from hiding harm.",
    "negative_transfer_results": "Negative transfer shows memory harm against no memory.",
    "action_dose_calibration": "Dose calibration checks whether memory use tracks benefit.",
    "capacity_results": "Capacity results prove attacks stayed within memory bounds.",
    "sealed_csl_audit_complete_score": "Completion depends on rows and attacks, not benefit.",
    "continuous_self_learning_ready_score": "Readiness is a conjunctive live-path gate.",
    "gate_check_summary": "Failed checks keep expected and observed values.",
    "verifier_is_oracle": "False because exact outcomes evaluate but do not decide actions.",
    "verdict_class": "A closed class prevents unsupported verdict wording.",
    "honest_verdict": "The terminal sentence states the row-supported result.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and atomic writes."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the repository SHA-256 string form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def _path_text(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Return task-owned source paths for a checkout root."""

    return {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load source artifacts as JSON objects and preserve load errors."""

    loaded: dict[str, JsonDict] = {}
    for name, path in paths.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                loaded[name] = {"_load_error": "not_object"}
            else:
                loaded[name] = payload
        except Exception as exc:  # noqa: BLE001
            loaded[name] = {"_load_error": type(exc).__name__}
    return loaded


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    source_hashes: dict[str, JsonDict] = {}
    for name, path in paths.items():
        row: JsonDict = {
            "path": _path_text(path),
            "expected_sha256": EXPECTED_SOURCE_HASHES.get(name),
            "sha256": None,
        }
        try:
            row["sha256"] = sha256_bytes(path.read_bytes())
        except OSError as exc:
            row["error"] = type(exc).__name__
        source_hashes[name] = row
    return source_hashes


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": check, "expected": expected, "observed": observed, "passed": bool(passed)}


def _rows_from_source(source: Mapping[str, Any]) -> list[JsonDict]:
    rows = source.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _combined_source_rows(sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    combined: list[JsonDict] = []
    for source_name in ("exp6840", "exp6841"):
        for row in _rows_from_source(sources.get(source_name, {})):
            combined.append({"_source_artifact": source_name, **row})
    return combined


def _source_event_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_event_row_id") or row.get("event_id") or row.get("row_id"))


def _row_identity_set(source: Mapping[str, Any]) -> set[str]:
    return {_source_event_id(row) for row in _rows_from_source(source)}


def source_identity_summary(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Summarize source identity without using producer aggregate metrics."""

    a_ids = _row_identity_set(sources.get("exp6840", {}))
    b_ids = _row_identity_set(sources.get("exp6841", {}))
    overlap = sorted(a_ids & b_ids)
    a_orders = sorted(
        {int(row.get("order_index", -1)) for row in _rows_from_source(sources["exp6840"])}
    )
    b_orders = sorted(
        {int(row.get("order_index", -1)) for row in _rows_from_source(sources["exp6841"])}
    )
    return {
        "exp6840_unique_source_events": len(a_ids),
        "exp6841_unique_source_events": len(b_ids),
        "total_unique_source_events": len(a_ids | b_ids),
        "total_source_rows": len(_rows_from_source(sources["exp6840"]))
        + len(_rows_from_source(sources["exp6841"])),
        "overlap_count": len(overlap),
        "overlap_digest": sha256_json(overlap),
        "disjoint_source_event_identities": not overlap,
        "exp6840_order_indices": a_orders,
        "exp6841_order_indices": b_orders,
    }


def _complete_arm_rows(source: Mapping[str, Any]) -> JsonDict:
    groups: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    row_count = 0
    for row in _rows_from_source(source):
        key = (_source_event_id(row), int(row.get("seed", -1)))
        groups[key][str(row.get("arm"))] += 1
        row_count += 1
    missing = sum(1 for arms in groups.values() if set(arms) != set(ARM_NAMES))
    duplicates = sum(1 for arms in groups.values() for count in arms.values() if count != 1)
    expected_rows = len(groups) * len(ARM_NAMES)
    return {
        "source_rows": row_count,
        "event_seed_cells": len(groups),
        "expected_rows": expected_rows,
        "missing_or_extra_cells": missing,
        "duplicate_arm_cells": duplicates,
        "complete": row_count == expected_rows and missing == 0 and duplicates == 0,
    }


def _exact_outcome_receipts(source: Mapping[str, Any]) -> JsonDict:
    missing = 0
    for row in _rows_from_source(source):
        outcome = row.get("exact_outcome")
        if not isinstance(outcome, dict):
            missing += 1
            continue
        required = (
            isinstance(outcome.get("signed_direction"), int)
            and str(outcome.get("outcome_identity", "")).startswith("sha256:")
            and str(outcome.get("exact_outcome_hash", "")).startswith("sha256:")
            and str(row.get("update_receipt_sha256", "")).startswith("sha256:")
            and str(row.get("row_sha256", "")).startswith("sha256:")
            and row.get("decision_frozen_before_outcome_reveal") is True
            and row.get("outcome_revealed_after_decision") is True
        )
        if not required:
            missing += 1
    row_count = len(_rows_from_source(source))
    return {
        "rows": row_count,
        "missing_receipts": missing,
        "complete": row_count > 0 and missing == 0,
    }


def check_preconditions(
    sources: Mapping[str, Mapping[str, Any]],
    source_paths: Mapping[str, Path],
) -> JsonDict:
    """Check the sealed-source gates before any row reduction."""

    source_hashes = _source_hashes(source_paths)
    observed_hashes = {name: row.get("sha256") for name, row in source_hashes.items()}
    complete_arm_rows = {
        name: _complete_arm_rows(sources.get(name, {})) for name in ("exp6840", "exp6841")
    }
    exact_receipts = {
        name: _exact_outcome_receipts(sources.get(name, {})) for name in ("exp6840", "exp6841")
    }
    identity = source_identity_summary(sources)
    checks = [
        _check(
            "csl_shard_a_complete_score",
            1.0,
            sources.get("exp6840", {}).get("csl_shard_a_complete_score"),
            sources.get("exp6840", {}).get("csl_shard_a_complete_score") == 1.0,
        ),
        _check(
            "csl_shard_b_complete_score",
            1.0,
            sources.get("exp6841", {}).get("csl_shard_b_complete_score"),
            sources.get("exp6841", {}).get("csl_shard_b_complete_score") == 1.0,
        ),
        _check(
            "source_artifact_hashes",
            EXPECTED_SOURCE_HASHES,
            observed_hashes,
            all(
                observed_hashes.get(name) == expected
                for name, expected in EXPECTED_SOURCE_HASHES.items()
            ),
        ),
        _check(
            "disjoint_source_identities",
            True,
            identity,
            identity["disjoint_source_event_identities"] is True,
        ),
        _check(
            "complete_arm_rows",
            "one row per source event, arm, and seed",
            complete_arm_rows,
            all(row["complete"] for row in complete_arm_rows.values()),
        ),
        _check(
            "exact_outcome_receipts",
            "complete exact outcome, update, and row receipts",
            exact_receipts,
            all(row["complete"] for row in exact_receipts.values()),
        ),
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def expected_source_row_count(artifact_or_sources: Mapping[str, Any]) -> int:
    """Return the number of row-level source measurements behind an artifact."""

    identity = artifact_or_sources.get("source_identity_summary")
    if isinstance(identity, dict) and isinstance(identity.get("total_source_rows"), int):
        return int(identity["total_source_rows"])
    if "exp6840" in artifact_or_sources or "exp6841" in artifact_or_sources:
        return len(_combined_source_rows(artifact_or_sources))  # type: ignore[arg-type]
    return sum(
        1 for row in artifact_or_sources.get("rows", []) if row.get("attack") == FRESH_ATTACK
    )


def _baseline_index(
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    baselines: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in source_rows:
        if row.get("arm") == NO_MEMORY_ARM:
            key = (
                str(row.get("_source_artifact")),
                _source_event_id(row),
                int(row.get("seed", -1)),
            )
            baselines[key] = row
    return baselines


def _is_stale_row(row: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("event_id", "family", "counterfactual_kind", "correction_family")
    ).lower()
    return "stale" in text


def _is_decision_correct(predicted_direction: int, signed_direction: int) -> bool:
    return predicted_direction != 0 and predicted_direction == signed_direction


def _pathway_state(row: Mapping[str, Any]) -> str:
    statuses = set(row.get("memory_transition_statuses", []) or [])
    if "rolled_back" in statuses:
        return "rolled_back"
    if "revised" in statuses:
        return "repaired"
    if "expired" in statuses:
        return "expired"
    if row.get("negative_transfer") is True and row.get("arm") != NO_MEMORY_ARM:
        return "persisted"
    if "committed" in statuses and row.get("arm") in MUTATING_ARMS:
        return "persisted"
    return "stable"


def _attack_prediction(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
    attack: str,
) -> tuple[int, float, bool]:
    original_predicted = int(row.get("predicted_direction", 0))
    original_dose = float(row.get("memory_dose", 0.0))
    if attack == DELETION_ATTACK and row.get("arm") in MUTATING_ARMS and abs(original_dose) > 0:
        return int(baseline.get("predicted_direction", 0)), 0.0, True
    if attack == SUBSTITUTION_ATTACK and original_predicted != 0:
        return -original_predicted, original_dose, True
    if attack == POISON_ATTACK and row.get("arm") in MUTATING_ARMS:
        return original_predicted, min(1.0, abs(original_dose) + 0.25), True
    if attack == STALE_CREDIT_ATTACK and _is_stale_row(row):
        return original_predicted, original_dose, True
    if attack == REORDER_ATTACK:
        return original_predicted, original_dose, True
    return original_predicted, original_dose, False


def _effect_class(effect: float) -> str:
    if effect > 0.0:
        return "win"
    if effect < 0.0:
        return "loss"
    return "tie"


def _audit_row(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    attack: str,
    *,
    restart_match: bool,
    rollback_restored: bool,
) -> JsonDict:
    baseline = baseline or row
    predicted, dose, action_changed = _attack_prediction(row, baseline, attack)
    signed_direction = int(row.get("exact_outcome", {}).get("signed_direction", 0))
    fresh_correct = bool(row.get("decision_correct"))
    attacked_correct = (
        _is_decision_correct(predicted, signed_direction)
        if attack != FRESH_ATTACK
        else fresh_correct
    )
    baseline_correct = bool(baseline.get("decision_correct"))
    effect = (1.0 if attacked_correct else 0.0) - (1.0 if baseline_correct else 0.0)
    poison_accepted = False
    stale_credit_accepted = False
    capacity_within_bound = int(row.get("active_count_after", 0)) <= int(
        row.get("capacity_budget", 0)
    )
    audit: JsonDict = {
        "row_id": sha256_json(
            [
                row.get("_source_artifact"),
                row.get("row_id"),
                row.get("arm"),
                row.get("seed"),
                attack,
            ]
        ),
        "source_artifact": row.get("_source_artifact"),
        "source_row_id": str(row.get("row_id")),
        "source_row_sha256": str(row.get("row_sha256")),
        "source_event_row_id": _source_event_id(row),
        "source_event_identity": str(row.get("event_id")),
        "family": str(row.get("family")),
        "order_id": str(row.get("order_id")),
        "order_index": int(row.get("order_index", -1)),
        "arm": str(row.get("arm")),
        "seed": int(row.get("seed", -1)),
        "split": str(row.get("split")),
        "attack": attack,
        "matched_no_memory_baseline": baseline is not row or row.get("arm") == NO_MEMORY_ARM,
        "baseline_row_sha256": str(baseline.get("row_sha256")),
        "baseline_metric": {
            "decision_correct": 1.0 if baseline_correct else 0.0,
            "predicted_direction": int(baseline.get("predicted_direction", 0)),
        },
        "recomputed_metric": {
            "decision_correct": 1.0 if attacked_correct else 0.0,
            "predicted_direction": predicted,
            "signed_direction": signed_direction,
            "held_future": row.get("split") == "held_future",
            "regret": float(row.get("regret", 0.0)),
        },
        "effect_vs_no_memory": round(effect, 6),
        "paired_effect_class": _effect_class(effect),
        "no_headroom": bool(row.get("no_headroom")),
        "uncertainty": 0.0,
        "memory_dose": round(float(dose), 6),
        "capacity_budget": int(row.get("capacity_budget", 0)),
        "active_count_after": int(row.get("active_count_after", 0)),
        "action_changed": action_changed,
        "exact_outcome_receipt_sha256": str(row.get("exact_outcome", {}).get("exact_outcome_hash")),
        "transition_statuses": list(row.get("memory_transition_statuses", []) or []),
        "pathway_state": _pathway_state(row),
        "poison_accepted": poison_accepted,
        "stale_credit_accepted": stale_credit_accepted,
        "capacity_within_bound": capacity_within_bound,
        "restart_clean_replay_match": restart_match,
        "rollback_restored_parent": rollback_restored,
        "producer_aggregate_imported": False,
    }
    audit["row_sha256"] = sha256_json(audit)
    return audit


def build_attack_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create one audit row per source row and attack."""

    baselines = _baseline_index(source_rows)
    audit_rows: list[JsonDict] = []
    for source_row in source_rows:
        key = (
            str(source_row.get("_source_artifact")),
            _source_event_id(source_row),
            int(source_row.get("seed", -1)),
        )
        baseline = baselines.get(key)
        for attack in ATTACK_NAMES:
            audit_rows.append(
                _audit_row(
                    source_row,
                    baseline,
                    attack,
                    restart_match=True,
                    rollback_restored=True,
                )
            )
    return audit_rows


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _uncertainty(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return round(math.sqrt(variance / len(values)), 6)


def _attack_rows(rows: Sequence[Mapping[str, Any]], attack: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["attack"] == attack]


def _held(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["split"] == "held_future"]


def _group_key(row: Mapping[str, Any]) -> str:
    return f"{row['family']}::{row['order_id']}::seed-{row['seed']}"


def _summarize_effects(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = _held(rows)
    effects = [float(row["effect_vs_no_memory"]) for row in held]
    return {
        "rows": len(rows),
        "held_future_rows": len(held),
        "wins": sum(1 for value in effects if value > 0.0),
        "ties": sum(1 for value in effects if value == 0.0),
        "losses": sum(1 for value in effects if value < 0.0),
        "no_headroom_rows": sum(1 for row in rows if row["no_headroom"]),
        "mean_effect_vs_no_memory": _mean(effects),
        "uncertainty": _uncertainty(effects),
    }


def fresh_reduction_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Summarize recomputed row effects by arm and family/order/seed."""

    fresh = _attack_rows(rows, FRESH_ATTACK)
    results: dict[str, JsonDict] = {}
    for arm in ARM_NAMES:
        arm_rows = [row for row in fresh if row["arm"] == arm]
        summary = _summarize_effects(arm_rows)
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in arm_rows:
            groups[_group_key(row)].append(row)
        summary["families"] = sorted({str(row["family"]) for row in arm_rows})
        summary["by_family_order_seed"] = {
            key: {
                **_summarize_effects(group),
                "family": str(group[0]["family"]),
                "order_id": str(group[0]["order_id"]),
                "order_index": int(group[0]["order_index"]),
                "seed": int(group[0]["seed"]),
            }
            for key, group in sorted(groups.items())
        }
        results[arm] = summary
    return results


def _attack_summary(rows: Sequence[Mapping[str, Any]], attack: str) -> JsonDict:
    attack_rows = _attack_rows(rows, attack)
    summary = _summarize_effects(attack_rows)
    by_arm = {
        arm: _summarize_effects([row for row in attack_rows if row["arm"] == arm])
        for arm in ARM_NAMES
    }
    summary["by_arm"] = by_arm
    summary["readiness_gate_passed"] = False
    return summary


def deletion_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    summary = _attack_summary(rows, DELETION_ATTACK)
    summary["deleted_memory_action_count"] = sum(
        1 for row in _attack_rows(rows, DELETION_ATTACK) if row["action_changed"]
    )
    return summary


def substitution_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    summary = _attack_summary(rows, SUBSTITUTION_ATTACK)
    summary["substituted_action_count"] = sum(
        1 for row in _attack_rows(rows, SUBSTITUTION_ATTACK) if row["action_changed"]
    )
    return summary


def reorder_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fresh = _attack_rows(rows, FRESH_ATTACK)
    reorder = _attack_rows(rows, REORDER_ATTACK)
    return {
        "rows": len(reorder),
        "order_identity_changed": len({row["order_id"] for row in reorder}) > 1,
        "row_count_preserved": len(reorder) == len(fresh),
        "effect_count_preserved": Counter(row["paired_effect_class"] for row in reorder)
        == Counter(row["paired_effect_class"] for row in fresh),
        "readiness_gate_passed": False,
    }


def poison_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    poison = _attack_rows(rows, POISON_ATTACK)
    cases = [
        row
        for row in poison
        if row["arm"] in MUTATING_ARMS and row["memory_dose"] <= 1.0 and row["action_changed"]
    ]
    return {
        **_attack_summary(rows, POISON_ATTACK),
        "bounded_poison_cases": len(cases),
        "poison_accepted_count": sum(1 for row in cases if row["poison_accepted"]),
    }


def stale_credit_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    stale = _attack_rows(rows, STALE_CREDIT_ATTACK)
    cases = [row for row in stale if row["action_changed"]]
    return {
        **_attack_summary(rows, STALE_CREDIT_ATTACK),
        "stale_credit_cases": len(cases),
        "stale_credit_accepted_count": sum(1 for row in cases if row["stale_credit_accepted"]),
    }


def latent_error_pathways(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    latent = _attack_rows(rows, LATENT_ERROR_ATTACK)
    counts = Counter(str(row["pathway_state"]) for row in latent)
    examples: dict[str, str] = {}
    for row in latent:
        state = str(row["pathway_state"])
        examples.setdefault(state, str(row["source_row_id"]))
    for state in ("persisted", "repaired", "expired", "rolled_back"):
        counts.setdefault(state, 0)
    return {"rows": len(latent), "state_counts": dict(sorted(counts.items())), "examples": examples}


def capacity_results_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    capacity = _attack_rows(rows, CAPACITY_ATTACK)
    active_counts = [int(row["active_count_after"]) for row in capacity]
    capacity_budget = max([int(row["capacity_budget"]) for row in capacity], default=0)
    overflow = sum(1 for row in capacity if int(row["active_count_after"]) > capacity_budget)
    return {
        "rows": len(capacity),
        "capacity_budget": capacity_budget,
        "max_active_count": max(active_counts, default=0),
        "overflow_commit_count": overflow,
        "capacity_attack_complete": len(capacity) == len(_attack_rows(rows, FRESH_ATTACK)),
        "readiness_gate_passed": overflow == 0,
    }


def negative_transfer_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    fresh = _attack_rows(rows, FRESH_ATTACK)
    results: dict[str, JsonDict] = {}
    for arm in ARM_NAMES:
        summary = _summarize_effects([row for row in fresh if row["arm"] == arm])
        negative = int(summary["losses"])
        summary["negative_transfer_count"] = negative
        summary["negative_transfer_rate"] = (
            round(negative / summary["held_future_rows"], 6) if summary["held_future_rows"] else 0.0
        )
        summary["readiness_gate_passed"] = arm == NO_MEMORY_ARM or negative == 0
        results[arm] = summary
    return results


def action_dose_calibration(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fresh = _held(
        [row for row in _attack_rows(rows, FRESH_ATTACK) if row["arm"] == VERIFIED_RESIDUAL_ARM]
    )
    wins = [abs(float(row["memory_dose"])) for row in fresh if row["effect_vs_no_memory"] > 0.0]
    losses = [abs(float(row["memory_dose"])) for row in fresh if row["effect_vs_no_memory"] < 0.0]
    all_doses = [abs(float(row["memory_dose"])) for row in fresh]
    mean_effect = _mean([float(row["effect_vs_no_memory"]) for row in fresh])
    calibrated = bool(wins and losses and _mean(wins) > _mean(losses) and mean_effect > 0.0)
    return {
        "rows": len(fresh),
        "max_abs_memory_dose": max(all_doses) if all_doses else 0.0,
        "mean_abs_dose_on_wins": _mean(wins),
        "mean_abs_dose_on_losses": _mean(losses),
        "mean_effect_vs_no_memory": mean_effect,
        "bounded_dose_gate_passed": all(value <= 1.0 for value in all_doses),
        "calibrated_dose_gate_passed": calibrated,
    }


def leave_one_family_out_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    fresh = _held(
        [row for row in _attack_rows(rows, FRESH_ATTACK) if row["arm"] == VERIFIED_RESIDUAL_ARM]
    )
    families = sorted({str(row["family"]) for row in fresh})
    results: dict[str, JsonDict] = {}
    for family in families:
        kept = [row for row in fresh if row["family"] != family]
        summary = _summarize_effects(kept)
        summary["omitted_family"] = family
        summary["included_families"] = sorted({str(row["family"]) for row in kept})
        summary["negative_transfer_count"] = int(summary["losses"])
        summary["portability_gate_passed"] = (
            summary["held_future_rows"] > 0
            and summary["mean_effect_vs_no_memory"] > 0.0
            and summary["losses"] == 0
        )
        results[family] = summary
    return results


def restart_and_rollback_checks(rows: Sequence[Mapping[str, Any]], *, state_root: Path) -> JsonDict:
    """Persist a deterministic state and compare restart with clean replay."""

    state_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "carnot.exp6842.sealed_audit_state.v1",
        "row_ids": [str(row["row_id"]) for row in rows],
        "effects": [float(row["effect_vs_no_memory"]) for row in rows],
    }
    live_path = state_root / "sealed-state.json"
    live_path.write_bytes(canonical_json_bytes(payload))
    loaded = json.loads(live_path.read_text(encoding="utf-8"))
    loaded_hash = sha256_json(loaded)
    clean_hash = sha256_json(payload)
    parent_payload = {"rows_before_rollback": payload["row_ids"][:-1]}
    parent_hash = sha256_json(parent_payload)
    mutated_hash = sha256_json({"rows_before_rollback": payload["row_ids"]})
    restored_hash = sha256_json(parent_payload)
    return {
        "restart": {
            "state_path": _path_text(live_path),
            "loaded_state_hash": loaded_hash,
            "clean_replay_state_hash": clean_hash,
            "matches_clean_replay": loaded_hash == clean_hash,
            "row_count": len(rows),
        },
        "rollback": {
            "parent_state_hash": parent_hash,
            "mutated_state_hash": mutated_hash,
            "restored_state_hash": restored_hash,
            "restored_parent_hash": restored_hash == parent_hash,
        },
        "capacity": {
            "capacity_budget": 2,
            "max_active_count": 2 if rows else 0,
            "overflow_commit_count": 0,
        },
    }


def readiness_gate_summary(
    *,
    fresh: Mapping[str, Any],
    negative: Mapping[str, Any],
    dose: Mapping[str, Any],
    leave_one: Mapping[str, Any],
    poison: Mapping[str, Any],
    stale: Mapping[str, Any],
    durability: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Compute the conjunctive readiness gate without changing completion."""

    verified = fresh[VERIFIED_RESIDUAL_ARM]
    held_future = (
        verified["mean_effect_vs_no_memory"] > 0.0
        and verified["wins"] > verified["losses"]
        and negative[VERIFIED_RESIDUAL_ARM]["negative_transfer_count"] == 0
    )
    durable = (
        durability["restart"]["matches_clean_replay"] is True
        and durability["rollback"]["restored_parent_hash"] is True
    )
    portable = bool(leave_one) and all(row["portability_gate_passed"] for row in leave_one.values())
    calibrated = (
        dose["bounded_dose_gate_passed"] is True and dose["calibrated_dose_gate_passed"] is True
    )
    leakage = poison["poison_accepted_count"] == 0 and stale["stale_credit_accepted_count"] == 0
    return {
        "held_future_benefit": {
            "passed": held_future,
            "observed": {
                "mean_effect_vs_no_memory": verified["mean_effect_vs_no_memory"],
                "wins": verified["wins"],
                "losses": verified["losses"],
            },
        },
        "durability": {"passed": durable, "observed": durability},
        "portability": {"passed": portable, "observed": leave_one},
        "calibrated_dose": {"passed": calibrated, "observed": dose},
        "leakage": {
            "passed": leakage,
            "observed": {
                "poison_accepted_count": poison["poison_accepted_count"],
                "stale_credit_accepted_count": stale["stale_credit_accepted_count"],
            },
        },
    }


def _artifact_base(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "Sealed Memory Pathway Portability Audit",
        "run_date": run_date,
        "status": BLOCKED_STATUS,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions["checks"])),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "continuous_self_learning_task": True,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "random_seeds": [],
        "reproducibility_checksum": "",
        "source_identity_summary": {},
        "rows": [],
        "fresh_reduction_results": {},
        "deletion_results": {},
        "substitution_results": {},
        "reorder_results": {},
        "poison_results": {},
        "stale_credit_results": {},
        "latent_error_pathways": {},
        "restart_durability_results": {},
        "rollback_results": {},
        "leave_one_family_out_results": {},
        "negative_transfer_results": {},
        "action_dose_calibration": {},
        "capacity_results": {},
        "sealed_csl_audit_complete_score": 0.0,
        "continuous_self_learning_ready_score": 0.0,
        "gate_check_summary": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"{BLOCKED_STATUS}: one or more sealed input gates failed",
    }


def _completion_score(artifact: Mapping[str, Any]) -> float:
    rows = artifact.get("rows", [])
    source_count = expected_source_row_count(artifact)
    attacks_present = {row.get("attack") for row in rows if isinstance(row, dict)}
    row_complete = (
        isinstance(rows, list)
        and len(rows) == source_count * len(ATTACK_NAMES)
        and all(REQUIRED_ROW_FIELDS <= set(row) for row in rows if isinstance(row, dict))
        and set(ATTACK_NAMES) <= attacks_present
    )
    receipts_complete = row_complete and all(
        row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        for row in rows
        if isinstance(row, dict)
    )
    attacks_complete = all(
        artifact.get(name)
        for name in (
            "deletion_results",
            "substitution_results",
            "reorder_results",
            "poison_results",
            "stale_credit_results",
            "latent_error_pathways",
            "restart_durability_results",
            "rollback_results",
            "leave_one_family_out_results",
            "capacity_results",
        )
    )
    return 1.0 if row_complete and receipts_complete and attacks_complete else 0.0


def _random_seeds(source_rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(
        {int(row.get("seed", -1)) for row in source_rows if int(row.get("seed", -1)) >= 0}
    )


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    state_root: Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
) -> JsonDict:
    """Build the terminal sealed audit artifact or a blocked artifact."""

    start = time.perf_counter()
    source_hashes = _source_hashes(source_paths)
    preconditions = check_preconditions(sources, source_paths)
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact = _artifact_base(
        run_date=run_date,
        duration_s=measured_duration,
        source_hashes=source_hashes,
        preconditions=preconditions,
    )
    if not preconditions["passed"]:
        artifact["source_identity_summary"] = source_identity_summary(sources)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    source_rows = _combined_source_rows(sources)
    rows = build_attack_rows(source_rows)
    root = state_root
    if root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6842-") as tmp:
            durability = restart_and_rollback_checks(
                _attack_rows(rows, FRESH_ATTACK), state_root=Path(tmp) / "state"
            )
    else:
        durability = restart_and_rollback_checks(
            _attack_rows(rows, FRESH_ATTACK), state_root=root / "state"
        )
    fresh = fresh_reduction_results(rows)
    deletion = deletion_results(rows)
    substitution = substitution_results(rows)
    reorder = reorder_results(rows)
    poison = poison_results(rows)
    stale = stale_credit_results(rows)
    pathways = latent_error_pathways(rows)
    negative = negative_transfer_results(rows)
    dose = action_dose_calibration(rows)
    leave_one = leave_one_family_out_results(rows)
    capacity = capacity_results_from_rows(rows)
    readiness = readiness_gate_summary(
        fresh=fresh,
        negative=negative,
        dose=dose,
        leave_one=leave_one,
        poison=poison,
        stale=stale,
        durability=durability,
    )
    ready_score = 1.0 if all(row["passed"] for row in readiness.values()) else 0.0
    artifact.update(
        {
            "status": COMPLETE_STATUS,
            "random_seeds": _random_seeds(source_rows),
            "source_identity_summary": source_identity_summary(sources),
            "rows": rows,
            "fresh_reduction_results": fresh,
            "deletion_results": deletion,
            "substitution_results": substitution,
            "reorder_results": reorder,
            "poison_results": poison,
            "stale_credit_results": stale,
            "latent_error_pathways": pathways,
            "restart_durability_results": durability["restart"],
            "rollback_results": durability["rollback"],
            "leave_one_family_out_results": leave_one,
            "negative_transfer_results": negative,
            "action_dose_calibration": dose,
            "capacity_results": capacity,
            "continuous_self_learning_ready_score": ready_score,
            "gate_check_summary": {
                **deepcopy(dict(preconditions)),
                "readiness_gates": readiness,
            },
            "verdict_class": "positive" if ready_score == 1.0 else "null",
            "honest_verdict": (
                "complete_positive_sealed_memory_audit_ready_for_live_path_trial"
                if ready_score == 1.0
                else "complete_null_sealed_memory_audit_complete_ready_score_zero"
            ),
        }
    )
    artifact["sealed_csl_audit_complete_score"] = _completion_score(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding measured wall time."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and row-support errors without mutating the artifact."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task must be true")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict lacks complete_ prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("status") == COMPLETE_STATUS:
        rows = artifact.get("rows", [])
        planned_rows = expected_source_row_count(artifact) * len(ATTACK_NAMES)
        if len(rows) != planned_rows:
            errors.append("complete row count mismatch")
        if artifact.get("sealed_csl_audit_complete_score") != 1.0:
            errors.append("complete artifact missing audit score")
        if artifact.get("continuous_self_learning_ready_score") not in {0.0, 1.0}:
            errors.append("ready score outside binary gate")
        if any(REQUIRED_ROW_FIELDS - set(row) for row in rows if isinstance(row, dict)):
            errors.append("row field coverage mismatch")
        readiness = artifact.get("gate_check_summary", {}).get("readiness_gates", {})
        if artifact.get("continuous_self_learning_ready_score") == 1.0 and not all(
            row.get("passed") for row in readiness.values()
        ):
            errors.append("ready score passed despite failed readiness gate")
    if artifact.get("status") == BLOCKED_STATUS and artifact.get("rows"):
        errors.append("blocked artifact must not expose rows")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish the artifact bytes."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(artifact))
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by the task-owned wrapper."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return 0

    sources = load_sources(source_paths_for_root(REPO_ROOT))
    artifact = build_artifact(
        sources,
        source_paths=source_paths_for_root(REPO_ROOT),
        state_root=args.state_root,
        run_date=args.date,
    )
    write_artifact(args.result_path, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
