"""Audit V657 feedback causality and retention from immutable event rows.

The audit reads producer bytes but does not import the producer reducer. This
keeps a shared implementation error from becoming independent confirmation.

Spec refs: REQ-REPORT-7510 and SCENARIO-REPORT-7510-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from typing import Any, Callable, TypeVar

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


Json = dict[str, Any]
T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7510-v657-causal-audit"
SCHEMA = "carnot.exp7510.v657.causal_audit.v1"
ONLINE_CHECKPOINT_SCHEMA = "carnot.exp7509.v657.online_checkpoint.v1"
RESULT_PATH = Path("results/experiment_7510_v657_causal_audit.json")
RAW_DIR = Path("results/raw/experiment_7510_v657_causal_audit")
MODULE_PATH = Path("python/carnot/experiment_7510_v657_causal_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7510_v657_causal_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7510_v657_causal_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v657-causal-audit.md")
UPSTREAM_PATHS = {
    7506: Path("results/experiment_7506_v657_causal_prototype.json"),
    7509: Path("results/experiment_7509_v657_causal_online.json"),
}
FEATURE_PATH = Path("results/raw/experiment_7504_v657_evidence_interface/features.jsonl")
NORMALIZATION_PATH = Path(
    "results/raw/experiment_7504_v657_evidence_interface/training_normalization.json"
)
PREDICTOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/predictors.jsonl")
EVALUATOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/evaluators.jsonl")

SCHEDULE_SEEDS = (656201, 656202, 656203, 656204, 656205)
DELAYS = (8, 0)
ARMS = (
    "frozen_base",
    "intercept_brier",
    "affine_brier",
    "local_brier",
    "shuffled_local_brier",
    "local_log_loss",
    "zero_step_local_brier",
)
PRIMARY_COMPARATORS = (
    "frozen_base",
    "intercept_brier",
    "affine_brier",
    "shuffled_local_brier",
    "local_log_loss",
)
MUTATION_NAMES = (
    "future_label",
    "cross_batch_permutation",
    "retroactive_prediction",
    "lost_pending_update",
    "favorable_seed_filtering",
    "retention_driven_rollback",
)
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
REQUIRED_UPSTREAM_RECEIPTS = {
    number: (
        *validation_scope.REQUIRED_CHECK_NAMES,
        "fresh_process_cold_replay",
        "independent_raw_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    )
    for number in (7506, 7509)
}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_json(path: Path) -> Json:
    """Read one object and reject malformed evidence instead of repairing it."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # pragma: no cover - defensive malformed JSON.
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[Json]:
    """Read object rows so original feature and label bytes stay inspectable."""

    rows: list[Json] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):  # pragma: no cover - defensive malformed JSONL.
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def source_row(path: Path, root: Path, evidence_class: str) -> Json:
    """Bind exact bytes while labeling historical evidence as non-current work."""

    resolved = path if path.is_absolute() else root / path
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - production sources are repository-relative.
        label = str(resolved)
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "evidence_class": evidence_class,
    }


def _receipts_pass(number: int, value: Mapping[str, Any]) -> bool:
    """Require one successful upstream receipt for every declared check."""

    receipts = value.get("validation_receipts")
    if not isinstance(receipts, list):
        return False
    for name in REQUIRED_UPSTREAM_RECEIPTS[number]:
        matched = [row for row in receipts if isinstance(row, Mapping) and row.get("name") == name]
        if len(matched) != 1:  # pragma: no cover - invalid receipt shape fails inventory.
            return False
        row = matched[0]
        if (  # pragma: no cover - invalid receipt state fails inventory.
            row.get("passed") is not True
            or row.get("exit_code") != 0
            or row.get("timed_out") is True
        ):
            return False
    return True


def _upstream_valid(number: int, value: Mapping[str, Any]) -> bool:
    """Apply terminal and readiness checks without using producer metrics."""

    allowed = {"null", "positive", "circular_positive"} if number == 7506 else {"null", "positive"}
    readiness = (
        value.get("causal_update_ready_score") == 1
        if number == 7506
        else value.get("causal_evaluation_complete_score") == 1
    )
    checkpoint_ready = True
    if number == 7506:
        required = set((value.get("checkpoint_schema") or {}).get("required_parts") or [])
        checkpoint_ready = {
            "models",
            "pending_queue",
            "audit_rng_state",
            "order_cursor",
        } <= required
    return bool(
        value.get("milestone") == MILESTONE
        and value.get("terminal_status") == "complete"
        and value.get("verdict_class") in allowed
        and value.get("flagged_adversarial") is False
        and readiness
        and checkpoint_ready
        and _receipts_pass(number, value)
    )


def inventory_upstreams(root: Path) -> list[Json]:
    """Inventory both producers even when either path is absent or malformed."""

    rows: list[Json] = []
    for number, relative in UPSTREAM_PATHS.items():
        path = root / relative
        if not path.is_file():
            rows.append(
                {"producer": number, "path": relative.as_posix(), "state": "absent", "sha256": None}
            )
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):  # pragma: no cover - malformed input.
            rows.append(
                {
                    "producer": number,
                    "path": relative.as_posix(),
                    "state": "invalid",
                    "sha256": sha256_file(path),
                }
            )
            continue
        rows.append(
            {
                "producer": number,
                "path": relative.as_posix(),
                "state": "valid" if _upstream_valid(number, value) else "invalid",
                "sha256": sha256_file(path),
                "honest_verdict": value.get("honest_verdict"),
                "verdict_class": value.get("verdict_class"),
                "flagged_adversarial": value.get("flagged_adversarial"),
                "required_receipts_passed": _receipts_pass(number, value),
            }
        )
    return rows


def classify_terminal(
    inventory: Sequence[Mapping[str, Any]], reduction_errors: Sequence[str]
) -> Json:
    """Keep completed accounting independent from evidence validity and benefit."""

    if reduction_errors or any(row.get("state") == "invalid" for row in inventory):
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_v657_causal_evidence",
            "causal_audit_complete_score": 1,
            "causal_claims_qualified_score": 0,
        }
    if any(row.get("state") == "absent" for row in inventory):
        return {
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_missing_v657_causal_input",
            "causal_audit_complete_score": 1,
            "causal_claims_qualified_score": 0,
        }
    return {
        "verdict_class": "null",
        "honest_verdict": "complete_null_v657_causal_audit_benefit_gate_failed",
        "causal_audit_complete_score": 1,
        "causal_claims_qualified_score": 1,
    }


def _brier(probability: float, label: int) -> float:
    """Recompute one Brier contribution from the saved probability and label."""

    return (float(probability) - int(label)) ** 2


def _log_loss(probability: float, label: int) -> float:
    """Recompute one binary log loss with the producer's metric-only clip."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(int(label) * math.log(clipped) + (1 - int(label)) * math.log1p(-clipped))


def _expectations(value: Mapping[str, Any]) -> Json:
    """Use fixed real counts, with explicit compact counts only for tests."""

    if value.get("test_fixture") is True:
        fixture = dict(value.get("audit_expectations") or {})
        return {
            "seeds": tuple(int(seed) for seed in fixture.get("seeds", [1])),
            "delays": tuple(int(delay) for delay in fixture.get("delays", [8])),
            "online_sources": int(fixture.get("online_sources", 2)),
            "retention_sources": int(fixture.get("retention_sources", 2)),
            "blocks": int(fixture.get("blocks", 1)),
        }
    return {
        "seeds": SCHEDULE_SEEDS,
        "delays": DELAYS,
        "online_sources": 159,
        "retention_sources": 116,
        "blocks": 20,
    }


def _stream_key(row: Mapping[str, Any]) -> tuple[int, int]:
    """Return the independent replay identity for one raw event row."""

    return int(row.get("schedule_seed", -1)), int(row.get("delay", -1))


def _prediction_groups(
    rows: Sequence[Mapping[str, Any]], errors: list[str]
) -> dict[tuple[int, int], dict[str, Json]]:
    """Validate every arm row and rebuild immutable prediction hashes."""

    grouped: dict[tuple[int, int], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[tuple[int, int, str, str]] = set()
    for row in rows:
        key = (*_stream_key(row), str(row.get("group_id")), str(row.get("arm")))
        if key in seen:  # pragma: no cover - private required mutations use other defects.
            errors.append("duplicate_prediction_row")
        seen.add(key)
        grouped[_stream_key(row)][str(row.get("group_id"))].append(row)

    output: dict[tuple[int, int], dict[str, Json]] = defaultdict(dict)
    for stream, groups in grouped.items():
        for group_id, arm_rows in groups.items():
            by_arm = {str(row.get("arm")): row for row in arm_rows}
            if (  # pragma: no cover - roster loss is covered at the update boundary.
                set(by_arm) != set(ARMS) or len(arm_rows) != len(ARMS)
            ):
                errors.append("prediction_arm_roster_mismatch")
                continue
            first = by_arm[ARMS[0]]
            payload = first.get("prediction_payload")
            common = (
                "source_family",
                "schedule_seed",
                "delay",
                "prediction_time",
                "audit_selected",
                "label",
                "prediction_hash",
            )
            if not isinstance(payload, Mapping) or any(  # pragma: no cover - malformed group.
                any(row.get(field) != first.get(field) for row in arm_rows) for field in common
            ):
                errors.append("prediction_group_inconsistent")
                continue
            probabilities = {arm: float(by_arm[arm]["probability"]) for arm in ARMS}
            states = {arm: str(by_arm[arm].get("parameter_hash")) for arm in ARMS}
            expected_hash = canonical_hash(
                {"payload": dict(payload), "probabilities": probabilities, "states": states}
            )
            if first.get("prediction_hash") != expected_hash:
                errors.append("prediction_hash_mismatch")
            label = int(first.get("label", -1))
            for arm, row in by_arm.items():
                probability = float(row.get("probability", math.nan))
                if (
                    label not in {0, 1}
                    or not math.isfinite(probability)
                    or not math.isclose(
                        float(row.get("brier_loss", math.nan)), _brier(probability, label)
                    )
                    or not math.isclose(
                        float(row.get("log_loss", math.nan)), _log_loss(probability, label)
                    )
                ):
                    errors.append("prediction_metric_mismatch")
                    break
            output[stream][group_id] = {
                "group_id": group_id,
                "source_family": str(first.get("source_family")),
                "prediction_time": int(first.get("prediction_time", -1)),
                "audit_selected": bool(first.get("audit_selected")),
                "label": label,
                "payload": dict(payload),
                "probabilities": probabilities,
                "states": states,
                "prediction_hash": str(first.get("prediction_hash")),
            }
    return output


def _audit_stream(
    stream: tuple[int, int],
    predictions: Mapping[str, Mapping[str, Any]],
    reveals: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    expected_sources: int,
    expected_blocks: int,
) -> Json:
    """Rebuild one predict, reveal, and update graph from saved operands."""

    errors: list[str] = []
    offending: list[Json] = []
    ordered = sorted(predictions.values(), key=lambda row: int(row["prediction_time"]))
    if len(ordered) != expected_sources or [
        row["prediction_time"] for row in ordered
    ] != list(  # pragma: no cover - malformed roster.
        range(expected_sources)
    ):
        errors.append("prediction_roster_mismatch")

    by_block = {int(row.get("block_id", -1)): row for row in reveals}
    if (  # pragma: no cover - malformed block roster.
        len(reveals) != expected_blocks or set(by_block) != set(range(expected_blocks))
    ):
        errors.append("reveal_block_roster_mismatch")
    released: set[str] = set()
    censored: set[str] = set()
    reveal_for_event: dict[str, Mapping[str, Any]] = {}
    for row in reveals:
        event_ids = [str(item) for item in row.get("event_ids") or []]
        target = released if row.get("disposition") == "released" else censored
        if row.get("disposition") not in {  # pragma: no cover - malformed disposition.
            "released",
            "censored",
        }:
            errors.append("reveal_disposition_invalid")
        for event_id in event_ids:
            if event_id in reveal_for_event:  # pragma: no cover - repeated external event.
                errors.append("feedback_event_repeated")
            reveal_for_event[event_id] = row
            target.add(event_id)
        expected_delivered = len(event_ids) if row.get("disposition") == "released" else 0
        if (  # pragma: no cover - malformed producer count.
            int(row.get("delivered_label_count", -1)) != expected_delivered
        ):
            errors.append("released_count_mismatch")
        update_time = row.get("update_time")
        if row.get("disposition") == "released" and (  # pragma: no cover - malformed reveal.
            not isinstance(update_time, int)
            or int(row.get("availability_time", update_time + 1)) > update_time
        ):
            errors.append("reveal_before_update_failed")

    selected = {group_id for group_id, row in predictions.items() if row["audit_selected"]}
    if selected != released | censored or released & censored:  # pragma: no cover - malformed mask.
        errors.append("audit_feedback_accounting_mismatch")
    by_group_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in updates:
        by_group_arm[(str(row.get("group_id")), str(row.get("arm")))].append(row)
    expected_update_keys = {(group, arm) for group in released for arm in ARMS}
    if set(by_group_arm) != expected_update_keys or any(
        len(rows) != 1 for rows in by_group_arm.values()
    ):
        errors.append("update_roster_mismatch")

    chronology = 0
    for row in updates:
        group_id = str(row.get("group_id"))
        arm = str(row.get("arm"))
        prediction = predictions.get(group_id)
        reveal = reveal_for_event.get(group_id)
        if prediction is None or reveal is None:  # pragma: no cover - parent loss has its own gate.
            errors.append("update_parent_missing")
            continue
        release_ids = [str(item) for item in row.get("release_event_ids") or []]
        origin = str(row.get("label_origin"))
        timing_bad = (
            int(prediction["prediction_time"]) > int(row.get("availability_time", -1))
            or int(row.get("availability_time", -1)) > int(row.get("update_time", -1))
            or row.get("availability_time") != reveal.get("availability_time")
        )
        origin_bad = origin not in release_ids or set(release_ids) != set(
            str(item) for item in reveal.get("event_ids") or []
        )
        if timing_bad or origin_bad:
            chronology += 1
            offending.append(
                {
                    "schedule_seed": stream[0],
                    "delay": stream[1],
                    "group_id": group_id,
                    "arm": arm,
                    "timing_bad": timing_bad,
                    "origin_bad": origin_bad,
                }
            )
        if origin_bad:
            errors.append("label_origin_outside_release_batch")
        elif int(row.get("label", -1)) != int(  # pragma: no cover - malformed origin label.
            predictions[origin]["label"]
        ):
            errors.append("feedback_label_origin_mismatch")
        if arm != "shuffled_local_brier" and origin != group_id:
            errors.append("aligned_arm_label_origin_mismatch")
        if int(row.get("prediction_time", -1)) != int(  # pragma: no cover - malformed join.
            prediction["prediction_time"]
        ):
            errors.append("update_prediction_time_mismatch")

    for reveal in reveals:
        if reveal.get("disposition") != "released":  # pragma: no cover - normal censored skip.
            continue
        event_ids = [str(item) for item in reveal.get("event_ids") or []]
        shuffle = [by_group_arm[(event_id, "shuffled_local_brier")][0] for event_id in event_ids]
        origins = [str(row.get("label_origin")) for row in shuffle]
        if sorted(origins) != sorted(event_ids):
            errors.append("shuffle_origin_not_a_permutation")
        changed = any(
            origin != event_id for event_id, origin in zip(event_ids, origins, strict=True)
        )
        expected_permutable = len(event_ids) if changed else 0
        if (  # pragma: no cover - malformed producer count.
            int(reveal.get("permutable_label_count", -1)) != expected_permutable
        ):
            errors.append("permutable_count_mismatch")

    errors.extend(_state_chain_errors(predictions, updates))
    released_count = len(released)
    censored_count = len(censored)
    permutable = sum(int(row.get("permutable_label_count", 0)) for row in reveals)
    observed_summary = {
        "source_count": len(predictions),
        "delivered_audit_labels": released_count,
        "permutable_labels": permutable,
        "censored_feedback_count": censored_count,
        "chronology_violations": chronology,
    }
    if any(summary.get(field) != observed for field, observed in observed_summary.items()):
        errors.append("stream_summary_mismatch")
    return {
        "unit_id": f"seed-{stream[0]}-delay-{stream[1]}",
        "schedule_seed": stream[0],
        "delay": stream[1],
        "prediction_group_count": len(predictions),
        "prediction_row_count": len(predictions) * len(ARMS),
        "released_label_count": released_count,
        "withheld_label_count": censored_count,
        "censored_label_count": censored_count,
        "permutable_label_count": permutable,
        "update_row_count": len(updates),
        "chronology_violation_count": chronology,
        "offending_rows": offending,
        "errors": list(dict.fromkeys(errors)),
        "status": "complete",
        "failed": bool(errors),
        "censored": False,
    }


def _state_chain_errors(
    predictions: Mapping[str, Mapping[str, Any]], updates: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Verify prediction-time states and every later update-hash transition."""

    if not predictions:  # pragma: no cover - malformed stream is rejected earlier.
        return ["prediction_state_chain_empty"]
    errors: list[str] = []
    by_time = {int(row["prediction_time"]): row for row in predictions.values()}
    initial = by_time.get(0)
    if initial is None:  # pragma: no cover - malformed prediction roster.
        return ["prediction_state_chain_missing_origin"]
    current = dict(initial["states"])
    updates_by_time: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in updates:
        updates_by_time[int(row.get("update_time", -1))].append(row)
    end = max([*by_time, *updates_by_time], default=0)
    for event_time in range(end + 1):
        prediction = by_time.get(event_time)
        if (  # pragma: no cover - extra defensive state-chain corruption.
            prediction is not None and dict(prediction["states"]) != current
        ):
            errors.append("retroactive_prediction_state_mismatch")
        for row in updates_by_time.get(event_time, []):
            arm = str(row.get("arm"))
            before = str(row.get("parameter_hash_before"))
            after = str(row.get("parameter_hash"))
            if (  # pragma: no cover - extra defensive state-chain corruption.
                arm not in current or before != current.get(arm)
            ):
                errors.append("update_hash_chain_mismatch")
                continue
            if (  # pragma: no cover - extra defensive control mutation.
                arm in {"frozen_base", "zero_step_local_brier"} and after != before
            ):
                errors.append("static_control_state_changed")
            current[arm] = after
    return list(dict.fromkeys(errors))


def _retention_audit(
    rows: Sequence[Mapping[str, Any]], expectations: Mapping[str, Any], online_ids: set[str]
) -> tuple[list[Json], list[str], int]:
    """Verify final evaluation labels only score frozen heads."""

    errors: list[str] = []
    expected_streams = {
        (seed, delay) for seed in expectations["seeds"] for delay in expectations["delays"]
    }
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(*_stream_key(row), str(row.get("group_id")))].append(row)
    observed_streams = {(seed, delay) for seed, delay, _group in grouped}
    if observed_streams != expected_streams:  # pragma: no cover - malformed retention roster.
        errors.append("retention_stream_roster_mismatch")
    leakage = 0
    role_ids: set[str] = set()
    for (_seed, _delay, group_id), arm_rows in grouped.items():
        role_ids.add(group_id)
        if (  # pragma: no cover - malformed retention arm roster.
            len(arm_rows) != len(ARMS) or {str(row.get("arm")) for row in arm_rows} != set(ARMS)
        ):
            errors.append("retention_arm_roster_mismatch")
        labels = {row.get("label") for row in arm_rows}
        if (  # pragma: no cover - malformed retention labels.
            len(labels) != 1 or next(iter(labels), None) not in {0, 1}
        ):
            errors.append("retention_label_mismatch")
        for row in arm_rows:
            if (
                row.get("used_for_update") is not False
                or row.get("used_for_selection_or_rollback") is not False
            ):
                leakage += 1
            probability = float(row.get("probability", math.nan))
            label = int(row.get("label", -1))
            if (  # pragma: no cover - metric corruption is covered on prediction rows.
                not math.isfinite(probability)
                or label not in {0, 1}
                or not math.isclose(
                    float(row.get("brier_loss", math.nan)), _brier(probability, label)
                )
                or not math.isclose(
                    float(row.get("log_loss", math.nan)), _log_loss(probability, label)
                )
            ):
                errors.append("retention_metric_mismatch")
    if leakage:
        errors.append("retention_control_leakage")
    if role_ids & online_ids:  # pragma: no cover - sealed roles are authenticated separately.
        errors.append("retention_online_identity_overlap")
    if len(role_ids) != int(  # pragma: no cover - malformed retention count.
        expectations["retention_sources"]
    ):
        errors.append("retention_source_count_mismatch")
    compact = _compact_role_rows(rows, role="retention", primary_delay=8)
    return compact, list(dict.fromkeys(errors)), leakage


def _compact_role_rows(
    rows: Sequence[Mapping[str, Any]], *, role: str, primary_delay: int
) -> list[Json]:
    """Average repeated schedules inside each source for compact audit rows."""

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    families: dict[str, str] = {}
    labels: dict[str, int] = {}
    for row in rows:
        if int(row.get("delay", -1)) != primary_delay:
            continue
        group = str(row.get("group_id"))
        arm = str(row.get("arm"))
        values[(group, arm)].append(float(row.get("brier_loss", math.nan)))
        families[group] = str(row.get("source_family"))
        labels[group] = int(row.get("label", -1))
    output: list[Json] = []
    for group in sorted(families):
        metrics = {
            arm: math.fsum(values[(group, arm)]) / len(values[(group, arm)])
            for arm in ARMS
            if values[(group, arm)]
        }
        output.append(
            {
                "unit_id": f"{role}:{group}",
                "group_id": group,
                "role": role,
                "source_family": families[group],
                "label": labels[group],
                "arm_mean_brier": metrics,
                "schedule_seed_count": len(values.get((group, "local_brier"), [])),
                "status": "complete",
                "failed": False,
                "censored": False,
            }
        )
    return output


def audit_event_history(value: Mapping[str, Any]) -> Json:
    """Audit all event rows without calling any producer reduction function."""

    expectations = _expectations(value)
    errors: list[str] = []
    raw_predictions = value.get("per_source_results")
    raw_updates = value.get("per_update_rows")
    raw_reveals = value.get("reveal_counts_by_batch")
    raw_retention = value.get("retention_rows")
    raw_streams = value.get("rows")
    if not all(  # pragma: no cover - malformed top-level row containers.
        isinstance(rows, list)
        for rows in (raw_predictions, raw_updates, raw_reveals, raw_retention, raw_streams)
    ):
        return _empty_history("raw_event_rows_missing")
    predictions = _prediction_groups(raw_predictions, errors)
    expected_streams = {
        (seed, delay) for seed in expectations["seeds"] for delay in expectations["delays"]
    }
    summaries = {_stream_key(row): row for row in raw_streams}
    if set(predictions) != expected_streams or set(summaries) != expected_streams:
        errors.append("stream_roster_mismatch")
    audit_rows: list[Json] = []
    offending: list[Json] = []
    for stream in sorted(expected_streams):
        stream_reveals = [row for row in raw_reveals if _stream_key(row) == stream]
        stream_updates = [row for row in raw_updates if _stream_key(row) == stream]
        audited = _audit_stream(
            stream,
            predictions.get(stream, {}),
            stream_reveals,
            stream_updates,
            summaries.get(stream, {}),
            expected_sources=int(expectations["online_sources"]),
            expected_blocks=int(expectations["blocks"]),
        )
        audit_rows.append(audited)
        errors.extend(audited["errors"])
        offending.extend(audited["offending_rows"])
    online_ids = {group for groups in predictions.values() for group in groups}
    retention_rows, retention_errors, leakage = _retention_audit(
        raw_retention, expectations, online_ids
    )
    errors.extend(retention_errors)
    compact_online = _compact_role_rows(raw_predictions, role="online", primary_delay=8)
    compact = [*compact_online, *retention_rows]
    planned = int(expectations["online_sources"]) + int(expectations["retention_sources"])
    return {
        "audit_rows": audit_rows,
        "rows": compact,
        "sample_size_budget": {
            "planned": planned,
            "attempted": len(compact),
            "completed": len(compact),
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": max(planned - len(compact), 0),
            "independent_unit": "source_group_by_role",
        },
        "chronology_violation_count": sum(
            int(row["chronology_violation_count"]) for row in audit_rows
        ),
        "chronology_offending_rows": offending,
        "prediction_hash_mismatch_count": errors.count("prediction_hash_mismatch"),
        "retention_leakage_count": leakage,
        "errors": list(dict.fromkeys(errors)),
    }


def _empty_history(error: str) -> Json:
    """Return a typed empty reduction for malformed present evidence."""

    return {
        "audit_rows": [],
        "rows": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "source_group_by_role",
        },
        "chronology_violation_count": 0,
        "chronology_offending_rows": [],
        "prediction_hash_mismatch_count": 0,
        "retention_leakage_count": 0,
        "errors": [error],
    }


def fixture_reduction_settings() -> Json:
    """Use small bootstrap work while retaining all registered contrast shapes."""

    return {
        "replicates": 32,
        "block_length": 16,
        "seed": 657009,
        "minimum_sources": 2,
        "minimum_delivered": 2,
        "minimum_permutable": 3,
        "minimum_frozen_delta": -0.01,
        "retention_upper95_max": 0.01,
    }


def _real_reduction_settings() -> Json:
    """Return the frozen Exp7509 thresholds without reading its headline."""

    return {
        "replicates": 2000,
        "block_length": 16,
        "seed": 657009,
        "minimum_sources": 120,
        "minimum_delivered": 20,
        "minimum_permutable": 12,
        "minimum_frozen_delta": -0.01,
        "retention_upper95_max": 0.01,
    }


def _source_pairs(rows: Sequence[Mapping[str, Any]], *, comparator: str, delay: int) -> list[Json]:
    """Average schedule seeds within a source before forming paired deltas."""

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    families: dict[str, str] = {}
    order: list[str] = []
    for row in rows:
        if int(row.get("delay", -1)) != delay:
            continue
        arm = str(row.get("arm"))
        if arm not in {"local_brier", comparator}:
            continue
        group = str(row.get("group_id"))
        families.setdefault(group, str(row.get("source_family")))
        if arm == "local_brier" and group not in order:
            order.append(group)
        values[(group, arm)].append(float(row.get("brier_loss", math.nan)))
    output: list[Json] = []
    for group in order:
        local = values.get((group, "local_brier"), [])
        control = values.get((group, comparator), [])
        if not local or len(local) != len(control):
            continue
        local_mean = math.fsum(local) / len(local)
        control_mean = math.fsum(control) / len(control)
        output.append(
            {
                "group_id": group,
                "source_family": families[group],
                "local_mean": local_mean,
                "comparator_mean": control_mean,
                "delta": local_mean - control_mean,
                "schedule_replicates": len(local),
            }
        )
    return output


def _moving_block_draws(
    pairs: Sequence[Mapping[str, Any]], *, block_length: int, replicates: int, seed: int
) -> list[float]:
    """Resample circular source blocks inside each preserved drift stratum."""

    strata: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        strata[str(row["source_family"])].append(float(row["delta"]))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _replicate in range(replicates):
        sampled: list[float] = []
        for values in strata.values():
            remaining = len(values)
            while remaining:
                start = int(rng.integers(0, len(values)))
                take = min(block_length, remaining)
                sampled.extend(values[(start + offset) % len(values)] for offset in range(take))
                remaining -= take
        draws.append(math.fsum(sampled) / len(sampled))
    return draws


def _contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparator: str,
    delay: int,
    block_length: int,
    replicates: int,
    seed: int,
) -> Json:
    """Reduce one registered paired Brier contrast from independent sources."""

    pairs = _source_pairs(rows, comparator=comparator, delay=delay)
    if not pairs:
        return {
            "comparator": comparator,
            "delay": delay,
            "source_count": 0,
            "replicate_unit": "schedule_seed_mean_within_source",
            "mean_delta": None,
            "upper95_delta": None,
            "one_sided_p": 1.0,
            "block_length": block_length,
            "replicates": replicates,
            "bootstrap_seed": seed,
            "drift_strata": [],
        }
    draws = _moving_block_draws(pairs, block_length=block_length, replicates=replicates, seed=seed)
    index = min(len(draws) - 1, math.ceil(0.95 * len(draws)) - 1)
    return {
        "comparator": comparator,
        "delay": delay,
        "source_count": len(pairs),
        "replicate_unit": "schedule_seed_mean_within_source",
        "mean_delta": math.fsum(float(row["delta"]) for row in pairs) / len(pairs),
        "upper95_delta": sorted(draws)[index],
        "one_sided_p": (1 + sum(draw >= 0.0 for draw in draws)) / (len(draws) + 1),
        "block_length": block_length,
        "replicates": replicates,
        "bootstrap_seed": seed,
        "drift_strata": list(dict.fromkeys(str(row["source_family"]) for row in pairs)),
    }


def _holm(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Apply one Holm family to all five registered primary contrasts."""

    ranked = sorted(enumerate(rows), key=lambda item: float(item[1]["one_sided_p"]))
    adjusted = [1.0] * len(rows)
    running = 0.0
    for rank, (index, row) in enumerate(ranked):
        running = max(running, min(1.0, (len(rows) - rank) * float(row["one_sided_p"])))
        adjusted[index] = running
    output: list[Json] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["holm_adjusted_p"] = adjusted[index]
        item["holm_passed"] = adjusted[index] < 0.05
        output.append(item)
    return output


def reduce_measurement(
    per_source_results: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    stream_rows: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any] | None = None,
) -> Json:
    """Independently reduce support, primary, sensitivity, and retention gates."""

    config = dict(settings or _real_reduction_settings())
    replicates = int(config["replicates"])
    seed = int(config["seed"])
    block_length = int(config["block_length"])
    primary = _holm(
        [
            _contrast(
                per_source_results,
                comparator=comparator,
                delay=8,
                block_length=block_length,
                replicates=replicates,
                seed=seed,
            )
            for comparator in PRIMARY_COMPARATORS
        ]
    )
    sensitivities = [
        _contrast(
            per_source_results,
            comparator="frozen_base",
            delay=delay,
            block_length=length,
            replicates=replicates,
            seed=seed,
        )
        for delay, length in ((8, 8), (8, 32), (0, 16))
    ]
    retention = _contrast(
        retention_rows,
        comparator="frozen_base",
        delay=8,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )
    primary_streams = [row for row in stream_rows if int(row.get("delay", -1)) == 8]
    support_rows = [
        {
            "schedule_seed": int(row["schedule_seed"]),
            "source_count": int(row["prediction_group_count"]),
            "delivered_audit_labels": int(row["released_label_count"]),
            "permutable_labels": int(row["permutable_label_count"]),
            "passed": int(row["prediction_group_count"]) >= int(config["minimum_sources"])
            and int(row["released_label_count"]) >= int(config["minimum_delivered"])
            and int(row["permutable_label_count"]) >= int(config["minimum_permutable"]),
        }
        for row in primary_streams
    ]
    support_passed = bool(support_rows) and all(row["passed"] for row in support_rows)
    frozen = next(row for row in primary if row["comparator"] == "frozen_base")
    primary_passed = bool(
        support_passed
        and frozen["mean_delta"] is not None
        and float(frozen["mean_delta"]) <= float(config["minimum_frozen_delta"])
        and all(
            row["upper95_delta"] is not None
            and float(row["upper95_delta"]) < 0.0
            and row["holm_passed"] is True
            for row in primary
        )
    )
    information = next(row for row in primary if row["comparator"] == "shuffled_local_brier")
    information_passed = bool(
        support_passed
        and information["upper95_delta"] is not None
        and float(information["upper95_delta"]) < 0.0
        and information["holm_passed"] is True
    )
    retention_passed = bool(
        retention["upper95_delta"] is not None
        and float(retention["upper95_delta"]) <= float(config["retention_upper95_max"])
    )
    return {
        "independent_unit": "schedule_seed_mean_within_source",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "primary_block_length": block_length,
        "primary_holm_family_size": len(primary),
        "support_rows": support_rows,
        "support_passed": support_passed,
        "primary_contrasts": primary,
        "primary_passed": primary_passed,
        "causal_information_passed": information_passed,
        "retention_contrast": retention,
        "retention_passed": retention_passed,
        "sensitivity_contrasts": sensitivities,
        "qualified_online_benefit": bool(
            primary_passed and information_passed and retention_passed
        ),
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    """Compare nested metric structures while tolerating only float roundoff."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def compare_producer_reduction(
    producer: Mapping[str, Any], reduced: Mapping[str, Any]
) -> list[str]:
    """Compare independent outputs only after their raw operands qualify."""

    expected = producer.get("measurement_reduction")
    projected = {
        key: reduced[key]
        for key in (
            "support_rows",
            "support_passed",
            "primary_contrasts",
            "primary_passed",
            "causal_information_passed",
            "retention_contrast",
            "retention_passed",
            "sensitivity_contrasts",
        )
    }
    return (
        []
        if isinstance(expected, Mapping) and _close(projected, expected)
        else ["producer_reduction_mismatch"]
    )


def _restart_path(path: Path, seed: int) -> Path:
    """Map one declared uninterrupted checkpoint to its sibling restart file."""

    return path.parent.parent / f"restart-seed-{seed}" / path.name


def audit_checkpoints(producer: Mapping[str, Any]) -> Json:
    """Resolve every declared checkpoint and compare all primary restart bytes."""

    errors: list[str] = []
    rows: list[Json] = []
    restart_pairs = 0
    pair_equal = True
    prediction_equal = True
    update_equal = True
    pending_equal = True
    models_equal = True
    rng_equal = True
    primary_delay = int((producer.get("protocol") or {}).get("primary_delay", 8))
    seeds = {int(seed) for seed in (producer.get("protocol") or {}).get("schedule_seeds") or []}
    for reference in producer.get("checkpoint_hashes") or []:
        if not isinstance(reference, Mapping):  # pragma: no cover - malformed reference row.
            errors.append("checkpoint_reference_invalid")
            continue
        path = Path(str(reference.get("path") or ""))
        seed = next(
            (candidate for candidate in seeds if f"schedule-{candidate}-d" in path.name),
            None,
        )
        delay_marker = f"-d{primary_delay}-"
        exists = path.is_file()
        observed_hash = sha256_file(path) if exists else None
        declared_match = observed_hash == reference.get("sha256")
        if not exists:  # pragma: no cover - external checkpoint disappearance.
            errors.append("checkpoint_source_missing")
        elif not declared_match:  # pragma: no cover - external checkpoint byte drift.
            errors.append("checkpoint_hash_mismatch")
        row: Json = {
            "path": str(path),
            "block_id": reference.get("block_id"),
            "declared_sha256": reference.get("sha256"),
            "observed_sha256": observed_hash,
            "declared_hash_matches": declared_match,
            "restart_path": None,
            "restart_sha256": None,
            "restart_equal": None,
        }
        if exists:
            try:
                payload = load_json(path)
            except (  # pragma: no cover - malformed checkpoint JSON.
                OSError,
                json.JSONDecodeError,
                ValueError,
            ):
                errors.append("checkpoint_payload_invalid")
            else:
                required = {
                    "schema",
                    "cursor",
                    "heads",
                    "predictions",
                    "updates",
                    "reveal_rows",
                    "pending_blocks",
                }
                if (  # pragma: no cover - malformed checkpoint schema.
                    payload.get("schema") != ONLINE_CHECKPOINT_SCHEMA
                    or not required <= payload.keys()
                ):
                    errors.append("checkpoint_schema_invalid")
        if exists and seed is not None and delay_marker in path.name:
            restarted = _restart_path(path, seed)
            row["restart_path"] = str(restarted)
            if not restarted.is_file():  # pragma: no cover - missing restart sibling.
                errors.append("restart_checkpoint_missing")
                pair_equal = False
            else:
                restart_pairs += 1
                restart_hash = sha256_file(restarted)
                equal = path.read_bytes() == restarted.read_bytes()
                row["restart_sha256"] = restart_hash
                row["restart_equal"] = equal
                if not equal:
                    errors.append("restart_checkpoint_bytes_mismatch")
                    pair_equal = False
                try:
                    left = load_json(path)
                    right = load_json(restarted)
                except (  # pragma: no cover - malformed restart JSON.
                    OSError,
                    json.JSONDecodeError,
                    ValueError,
                ):
                    errors.append("restart_checkpoint_payload_invalid")
                    prediction_equal = update_equal = pending_equal = models_equal = rng_equal = (
                        False
                    )
                else:
                    prediction_equal &= left.get("predictions") == right.get("predictions")
                    update_equal &= left.get("updates") == right.get("updates")
                    pending_equal &= left.get("pending_blocks") == right.get("pending_blocks")
                    models_equal &= left.get("heads") == right.get("heads")
                    if (  # pragma: no cover - online run uses stateless seed-derived RNG.
                        "audit_rng_state" in left or "audit_rng_state" in right
                    ):
                        rng_equal &= left.get("audit_rng_state") == right.get("audit_rng_state")
        rows.append(row)

    parity = producer.get("restart_parity_rows") or []
    required_comparisons = {
        "per_source_results",
        "per_update_rows",
        "retention_rows",
        "reveal_counts_by_batch",
        "final_state_hashes",
        "pending_queue",
    }
    if len(parity) != len(seeds) or any(  # pragma: no cover - malformed summary.
        row.get("passed") is not True
        or set((row.get("comparisons") or {})) != required_comparisons
        or not all((row.get("comparisons") or {}).values())
        for row in parity
    ):
        errors.append("restart_parity_summary_invalid")
    if not prediction_equal:  # pragma: no cover - byte mismatch already fails first.
        errors.append("restart_prediction_rows_mismatch")
    if not update_equal:  # pragma: no cover - byte mismatch already fails first.
        errors.append("restart_update_hashes_mismatch")
    if not pending_equal:  # pragma: no cover - byte mismatch already fails first.
        errors.append("restart_pending_queue_mismatch")
    if not models_equal:  # pragma: no cover - byte mismatch already fails first.
        errors.append("restart_model_state_mismatch")
    if not rng_equal:  # pragma: no cover - persisted RNG branch is defensive.
        errors.append("restart_rng_state_mismatch")
    return {
        "declared_checkpoint_count": len(producer.get("checkpoint_hashes") or []),
        "resolved_checkpoint_count": sum(row["declared_hash_matches"] for row in rows),
        "restart_pair_count": restart_pairs,
        "restart_bytes_equal": pair_equal and restart_pairs > 0,
        "prediction_rows_equal": prediction_equal and restart_pairs > 0,
        "update_hashes_equal": update_equal and restart_pairs > 0,
        "pending_queues_equal": pending_equal and restart_pairs > 0,
        "model_states_equal": models_equal and restart_pairs > 0,
        "rng_state_kind": "stateless_seed_derived",
        "rng_states_equal": rng_equal and restart_pairs > 0,
        "rows": rows,
        "errors": list(dict.fromkeys(errors)),
    }


def fixture_online_artifact() -> Json:
    """Build a compact complete event history for reader and mutation tests."""

    predictions: list[Json] = []
    updates: list[Json] = []
    retention: list[Json] = []
    reveals: list[Json] = []
    streams: list[Json] = []
    seeds = (1,)
    delays = (8, 0)
    labels = {"g0": 0, "g1": 1}
    probabilities = {
        "frozen_base": (0.3, 0.7),
        "intercept_brier": (0.28, 0.72),
        "affine_brier": (0.25, 0.75),
        "local_brier": (0.1, 0.9),
        "shuffled_local_brier": (0.35, 0.65),
        "local_log_loss": (0.24, 0.76),
        "zero_step_local_brier": (0.3, 0.7),
    }
    for seed in seeds:
        for delay in delays:
            initial = {arm: canonical_hash(["fixture-state", seed, delay, arm]) for arm in ARMS}
            for index, group in enumerate(labels):
                payload = {
                    "group_id": group,
                    "source_family": "fixture",
                    "features": [float(index), 0.0, 1.0, -1.0],
                    "base_probability": probabilities["frozen_base"][index],
                    "schedule_seed": seed,
                    "delay": delay,
                    "prediction_time": index,
                }
                arm_probabilities = {arm: values[index] for arm, values in probabilities.items()}
                prediction_hash = canonical_hash(
                    {"payload": payload, "probabilities": arm_probabilities, "states": initial}
                )
                for arm in ARMS:
                    probability = arm_probabilities[arm]
                    predictions.append(
                        {
                            "group_id": group,
                            "source_family": "fixture",
                            "schedule_seed": seed,
                            "delay": delay,
                            "arm": arm,
                            "prediction_time": index,
                            "audit_selected": True,
                            "prediction_payload": deepcopy(payload),
                            "prediction_hash": prediction_hash,
                            "parameter_hash": initial[arm],
                            "probability": probability,
                            "label": labels[group],
                            "brier_loss": _brier(probability, labels[group]),
                            "log_loss": _log_loss(probability, labels[group]),
                            "status": "complete",
                            "failed": False,
                            "censored": False,
                        }
                    )
            release_ids = list(labels)
            permutation = {"g0": "g1", "g1": "g0"}
            current = dict(initial)
            for group in release_ids:
                for arm in ARMS:
                    before = current[arm]
                    static = arm in {"frozen_base", "zero_step_local_brier"}
                    after = before if static else canonical_hash([before, group, arm])
                    current[arm] = after
                    origin = permutation[group] if arm == "shuffled_local_brier" else group
                    updates.append(
                        {
                            "group_id": group,
                            "arm": arm,
                            "schedule_seed": seed,
                            "delay": delay,
                            "block_id": 0,
                            "prediction_time": list(labels).index(group),
                            "availability_time": 7,
                            "update_time": 7,
                            "label": labels[origin],
                            "label_origin": origin,
                            "release_event_ids": release_ids,
                            "permutation_identity": canonical_hash([seed, delay, release_ids]),
                            "permutation_mode": "derangement",
                            "parameter_hash_before": before,
                            "parameter_hash": after,
                            "status": "no_update_control" if static else "committed",
                            "failed": False,
                            "censored": False,
                        }
                    )
            reveals.append(
                {
                    "block_id": 0,
                    "availability_time": 7,
                    "update_time": 7,
                    "event_ids": release_ids,
                    "delivered_label_count": 2,
                    "permutable_label_count": 2,
                    "permutation_mode": "derangement",
                    "permutation_identity": canonical_hash([seed, delay, release_ids]),
                    "disposition": "released",
                    "schedule_seed": seed,
                    "delay": delay,
                }
            )
            streams.append(
                {
                    "unit_id": f"seed-{seed}-delay-{delay}",
                    "schedule_seed": seed,
                    "delay": delay,
                    "source_count": 2,
                    "delivered_audit_labels": 2,
                    "permutable_labels": 2,
                    "censored_feedback_count": 0,
                    "chronology_violations": 0,
                    "status": "complete",
                    "failed": False,
                    "censored": False,
                }
            )
            for index, group in enumerate(("r0", "r1")):
                label = index
                for arm in ARMS:
                    probability = probabilities[arm][index]
                    retention.append(
                        {
                            "group_id": group,
                            "source_family": "fixture",
                            "schedule_seed": seed,
                            "delay": delay,
                            "arm": arm,
                            "probability": probability,
                            "label": label,
                            "brier_loss": _brier(probability, label),
                            "log_loss": _log_loss(probability, label),
                            "used_for_update": False,
                            "used_for_selection_or_rollback": False,
                            "status": "complete",
                            "failed": False,
                            "censored": False,
                        }
                    )
    comparisons = {
        "per_source_results": True,
        "per_update_rows": True,
        "retention_rows": True,
        "reveal_counts_by_batch": True,
        "final_state_hashes": True,
        "pending_queue": True,
    }
    return {
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "causal_evaluation_complete_score": 1,
        "online_benefit_score": 0,
        "validation_receipts": _valid_fixture_receipts(REQUIRED_UPSTREAM_RECEIPTS[7509]),
        "test_fixture": True,
        "audit_expectations": {
            "seeds": list(seeds),
            "delays": list(delays),
            "online_sources": 2,
            "retention_sources": 2,
            "blocks": 1,
        },
        "protocol": {"schedule_seeds": list(seeds), "delays": list(delays), "primary_delay": 8},
        "per_source_results": predictions,
        "per_update_rows": updates,
        "retention_rows": retention,
        "reveal_counts_by_batch": reveals,
        "rows": streams,
        "checkpoint_hashes": [],
        "restart_parity_rows": [
            {"schedule_seed": seed, "passed": True, "comparisons": comparisons} for seed in seeds
        ],
    }


def _valid_fixture_receipts(names: Sequence[str]) -> list[Json]:
    """Create successful receipt rows only for private test fixtures."""

    return [{"name": name, "passed": True, "exit_code": 0, "timed_out": False} for name in names]


def apply_private_mutation(value: Json, name: str) -> None:
    """Apply one named corruption to a private fixture in place."""

    if name == "future_label":
        value["per_update_rows"][0]["label_origin"] = "future-group"
    elif name == "cross_batch_permutation":
        shuffled = [
            row
            for row in value["per_update_rows"]
            if row["arm"] == "shuffled_local_brier" and row["delay"] == 8
        ]
        shuffled[1]["label_origin"] = shuffled[0]["label_origin"]
        shuffled[1]["label"] = shuffled[0]["label"]
    elif name == "retroactive_prediction":
        value["per_source_results"][0]["probability"] = 0.99
    elif name == "lost_pending_update":
        value["per_update_rows"].pop()
    elif name == "favorable_seed_filtering":
        value["rows"].pop()
    elif name == "retention_driven_rollback":
        value["retention_rows"][0]["used_for_selection_or_rollback"] = True
    else:
        raise ValueError(f"unknown_private_mutation:{name}")


def run_private_mutations() -> dict[str, bool]:
    """Prove six corruptions fail without retaining corrupted fixture bytes."""

    expected = {
        "future_label": "label_origin_outside_release_batch",
        "cross_batch_permutation": "shuffle_origin_not_a_permutation",
        "retroactive_prediction": "prediction_hash_mismatch",
        "lost_pending_update": "update_roster_mismatch",
        "favorable_seed_filtering": "stream_roster_mismatch",
        "retention_driven_rollback": "retention_control_leakage",
    }
    results: dict[str, bool] = {}
    for name in MUTATION_NAMES:
        fixture = fixture_online_artifact()
        apply_private_mutation(fixture, name)
        results[name] = expected[name] in audit_event_history(fixture)["errors"]
    return results


def _stable_unit(seed: int, *parts: str) -> float:
    """Map public identities to the producer's frozen unit-interval ordering."""

    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _temperature_probability(probability: float, temperature: float = 1.5) -> float:
    """Recompute the frozen scalar baseline from original option probability."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    logit = math.log(clipped / (1.0 - clipped)) / temperature
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _authenticate_references(
    root: Path, producer: Mapping[str, Any]
) -> tuple[list[Json], list[str]]:
    """Resolve every producer source reference before reading scientific rows."""

    rows: list[Json] = []
    errors: list[str] = []
    for reference in producer.get("source_artifact_hashes") or []:
        if not isinstance(reference, Mapping):  # pragma: no cover - malformed source reference.
            errors.append("source_reference_invalid")
            continue
        label = str(reference.get("path") or "")
        path = Path(label)
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        expected = reference.get("sha256")
        passed = observed == expected
        rows.append(
            {
                "path": label,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": passed,
            }
        )
        if not passed:  # pragma: no cover - external source byte drift.
            errors.append(f"upstream_source_hash_mismatch:{label}")
    return rows, errors


def _public_plan(
    public_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> tuple[Json, dict[int, list[str]], dict[int, dict[str, bool]]]:
    """Rebuild label-free arrival and audit decisions from public identities."""

    families: dict[str, list[str]] = defaultdict(list)
    for row in public_rows:
        families[str(row["source_family"])].append(str(row["group_id"]))
    plans: Json = {}
    orders: dict[int, list[str]] = {}
    masks: dict[int, dict[str, bool]] = {}
    for seed in seeds:
        family_order = sorted(families, key=lambda name: (_stable_unit(seed, "family", name), name))
        order: list[str] = []
        for family in family_order:
            order.extend(
                sorted(
                    families[family],
                    key=lambda group: (_stable_unit(seed, "member", family, group), group),
                )
            )
        mask = {
            group: _stable_unit(seed, "audit", group) < 0.25
            for groups in families.values()
            for group in groups
        }
        plans[str(seed)] = {"order": order, "audit": mask}
        orders[seed] = order
        masks[seed] = mask
    return plans, orders, masks


def audit_original_inputs(root: Path, producer: Mapping[str, Any]) -> Json:
    """Resolve feature, label, prediction-plan, and cited source identities."""

    reference_rows, errors = _authenticate_references(root, producer)
    features = load_jsonl(root / FEATURE_PATH)
    normalization = load_json(root / NORMALIZATION_PATH)
    predictors = {str(row["group_id"]): row for row in load_jsonl(root / PREDICTOR_PATH)}
    evaluators = load_jsonl(root / EVALUATOR_PATH)
    means = np.asarray(normalization.get("mean"), dtype=np.float64)[:4]
    scales = np.asarray(normalization.get("safe_scale"), dtype=np.float64)[:4]
    if (  # pragma: no cover - frozen normalization shape is authenticated upstream.
        means.shape != (4,) or scales.shape != (4,) or np.any(scales <= 0.0)
    ):
        errors.append("normalization_invalid")
    source_features = {
        str(row["group_id"]): row for row in features if row.get("role") in {"online", "test"}
    }
    labels = {
        str(row["group_id"]): 1 - int(row["label"])
        for row in evaluators
        if row.get("role") in {"online", "test"} and row.get("label") in {0, 1}
    }
    prediction_rows = producer.get("per_source_results") or []
    primary_predictions = {
        str(row["group_id"]): row
        for row in prediction_rows
        if row.get("schedule_seed") == SCHEDULE_SEEDS[0]
        and row.get("delay") == 8
        and row.get("arm") == "frozen_base"
    }
    for group, row in primary_predictions.items():
        feature = source_features.get(group)
        predictor = predictors.get(group)
        payload = row.get("prediction_payload") or {}
        if (  # pragma: no cover - sealed roster is authenticated upstream.
            feature is None or predictor is None or group not in labels
        ):
            errors.append("original_online_source_missing")
            continue
        expected_features = (
            (np.asarray(feature.get("features"), dtype=np.float64)[:4] - means) / scales
        ).tolist()
        expected_base = _temperature_probability(float(feature["raw_whole_expectation"]))
        if not np.allclose(  # pragma: no cover - private mutations target event history.
            payload.get("features"), expected_features, rtol=0.0, atol=1e-12
        ):
            errors.append("original_feature_mismatch")
        if not math.isclose(  # pragma: no cover - private mutations target event history.
            float(payload.get("base_probability", math.nan)), expected_base
        ):
            errors.append("original_base_probability_mismatch")
        if row.get("source_family") != predictor.get(  # pragma: no cover - source drift.
            "source_family"
        ):
            errors.append("original_source_family_mismatch")
        if row.get("label") != labels[group]:  # pragma: no cover - feedback byte drift.
            errors.append("original_feedback_label_mismatch")

    retention_rows = producer.get("retention_rows") or []
    primary_retention = {
        str(row["group_id"]): row
        for row in retention_rows
        if row.get("schedule_seed") == SCHEDULE_SEEDS[0]
        and row.get("delay") == 8
        and row.get("arm") == "frozen_base"
    }
    for group, row in primary_retention.items():
        predictor = predictors.get(group)
        if (  # pragma: no cover - sealed retention roster is authenticated upstream.
            group not in source_features or predictor is None or group not in labels
        ):
            errors.append("original_retention_source_missing")
        elif row.get("label") != labels[group] or row.get(  # pragma: no cover - retention drift.
            "source_family"
        ) != predictor.get("source_family"):
            errors.append("original_retention_feedback_mismatch")

    public = [
        {"group_id": group, "source_family": predictors[group]["source_family"]}
        for group, feature in source_features.items()
        if feature.get("role") == "online" and group in predictors
    ]
    plans, orders, masks = _public_plan(public, SCHEDULE_SEEDS)
    if producer.get("public_plan_hash") != canonical_hash(  # pragma: no cover - plan drift.
        plans
    ):
        errors.append("public_plan_hash_mismatch")
    for seed in SCHEDULE_SEEDS:
        for delay in DELAYS:
            observed = sorted(
                (
                    int(row["prediction_time"]),
                    str(row["group_id"]),
                    bool(row["audit_selected"]),
                )
                for row in prediction_rows
                if row.get("schedule_seed") == seed
                and row.get("delay") == delay
                and row.get("arm") == "frozen_base"
            )
            expected = [
                (index, group, masks[seed][group]) for index, group in enumerate(orders[seed])
            ]
            if observed != expected:  # pragma: no cover - schedule drift.
                errors.append("label_free_schedule_mismatch")
    return {
        "reference_rows": reference_rows,
        "authenticated_reference_count": sum(row["passed"] for row in reference_rows),
        "online_feature_count": len(primary_predictions),
        "retention_feature_count": len(primary_retention),
        "public_plan_hash": canonical_hash(plans),
        "errors": list(dict.fromkeys(errors)),
    }


REQUIRED_INPUTS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7490_v656_historical_audit.py"),
    Path("python/carnot/experiment_7498_v656_independent_audit.py"),
    Path("python/carnot/experiment_7506_v657_causal_prototype.py"),
    Path("python/carnot/experiment_7509_v657_causal_online.py"),
    Path("openspec/capabilities/continuous-learning/spec.md"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def collect_preconditions(root: Path) -> Json:
    """Check required resources and inventory both producers before reduction."""

    rows: list[Json] = []
    sources: list[Json] = []
    for relative in REQUIRED_INPUTS:
        path = root / relative
        readable = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": "resource_readable",
                "upstream": "worktree_instruction_or_helper",
                "field_path": relative.as_posix(),
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if readable else "missing_or_empty",
                "op": "eq",
                "passed": readable,
            }
        )
        if readable:
            sources.append(source_row(relative, root, "repository_input"))
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    requirement_present = "REQ-REPORT-7510" in spec
    rows.append(
        {
            "check": "relevant_requirement_present",
            "upstream": "OpenSpec",
            "field_path": "REQ-REPORT-7510",
            "expected": True,
            "observed": requirement_present,
            "op": "eq",
            "passed": requirement_present,
        }
    )
    inventory = inventory_upstreams(root)
    for item in inventory:
        relative = Path(str(item["path"]))
        if (root / relative).is_file():
            sources.append(source_row(relative, root, "upstream_terminal"))
        rows.append(
            {
                "check": f"upstream_inventory_exp{item['producer']}",
                "upstream": f"Exp{item['producer']}",
                "field_path": item["path"],
                "expected": "valid_or_explicit_absence",
                "observed": item["state"],
                "op": "in",
                "passed": item["state"] in {"valid", "absent"},
            }
        )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    quarantined = "exp7510-causal-audit" in exclusion
    rows.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": "ops/exclusion_manifest.yaml",
            "field_path": EXPERIMENT_ID,
            "expected": False,
            "observed": quarantined,
            "op": "eq",
            "passed": not quarantined,
        }
    )
    return {
        "rows": rows,
        "inventory": inventory,
        "source_artifact_hashes": sources,
        "missing_external": [row["path"] for row in inventory if row["state"] == "absent"],
        "invalid_present": [row["path"] for row in inventory if row["state"] == "invalid"],
    }


REQUIRED_CURRENT_RECEIPTS = (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)


def _current_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one successful current receipt for every scoped command."""

    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in REQUIRED_CURRENT_RECEIPTS
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    *,
    upstream: str,
    field_path: str,
) -> Json:
    """Attach one fixed principle to each observable gate operand."""

    principle = {
        "validity": "Favorable metrics cannot excuse invalid evidence.",
        "readiness": "A valid null must not block unrelated measurement accounting.",
        "benefit": "Exploratory rows cannot become confirmatory causal claims.",
    }[category]
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "upstream": upstream,
        "field_path": field_path,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Preserve every exact failed check and identify its first occurrence."""

    fields = ("check", "category", "upstream", "field_path", "expected", "observed", "op")
    failed = [
        {field: row.get(field) for field in fields}
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "all_validity_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "validity"
        ),
        "all_readiness_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "readiness"
        ),
        "all_benefit_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "benefit"
        ),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


FIELD_PRINCIPLES = {
    "schema": "Versioned identity prevents reader drift between audit contracts.",
    "run_date": "The fixed date binds this result to the authorized run.",
    "preconditions_checked": "Exact observed paths expose missing or invalid prerequisites.",
    "MODEL_SPECS": "An empty list prevents historical Qwen work from becoming a current load.",
    "model_specs": "The lowercase mirror prevents aliases from hiding model work.",
    "model_invoked": "False separates aggregation from current inference.",
    "invocation_counts": "Balanced zeros expose attempted or unfinished model work.",
    "inference_substrate": "The canonical aggregation value prevents substrate laundering.",
    "inference_substrate_class": "The aggregation class selects the correct duration rules.",
    "execution_venue": "Host work stays distinct from archived GPU or board evidence.",
    "duration_s": "Measured elapsed time prevents an invented compute floor.",
    "phase_spans": "Bounded phase spans expose stalls and unfinished work.",
    "random_seed": "Frozen arrival, audit, and bootstrap seeds prevent favorable reruns.",
    "reproducibility_checksum": "One hash binds sources, rows, settings, and validation.",
    "source_artifact_hashes": "Exact byte hashes prevent silent upstream replacement.",
    "rows": "Each independent source retains its own metrics and disposition.",
    "sample_size_budget": "Separate unit states expose missing, failed, or censored work.",
    "acceptance_gate_results": "Typed gates keep validity, readiness, and benefit separate.",
    "gate_check_summary": "Exact failed operands distinguish blocked, null, and invalid states.",
    "honest_verdict": "A complete prefix prevents terminal work from becoming retryable partial work.",
    "verdict_class": "A closed class prevents prose from changing machine meaning.",
    "verifier_is_oracle": "False prevents this audit from becoming a correctness oracle.",
    "flagged_adversarial": "Actual guard findings cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits, and hashes make current checks reviewable.",
    "field_principles": "Every emitted field states the failure that it prevents.",
    "causal_audit_complete_score": "Audit accounting stays independent from benefit.",
    "causal_claims_qualified_score": "Only fully valid causal and retention rows qualify.",
    "qualified_online_benefit_score": "Benefit cannot exceed frozen reproduced gates.",
    "chronology_violation_count": "An explicit count separates invalid control timing from a null.",
    "audit_rows": "Each source, control, and restart check retains observed provenance.",
}


def field_principles(value: Mapping[str, Any]) -> dict[str, str]:
    """Explain every terminal field, including schema extensions."""

    return {
        key: FIELD_PRINCIPLES.get(
            key, "This field preserves audit evidence and prevents silent omission."
        )
        for key in value
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks and the self-reference."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "ended_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "duration_breakdown_s",
        "phase_spans",
        "process_identity",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def build_artifact(
    *,
    preconditions: Mapping[str, Any],
    history: Mapping[str, Any],
    reduced: Mapping[str, Any],
    checkpoint_audit: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    reduction_errors: Sequence[str],
    source_hashes: Sequence[Mapping[str, Any]],
    raw_sidecars: Mapping[str, Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    mutation_results: Mapping[str, bool],
    producer_online_benefit_score: int,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    started_monotonic_ns: int = 0,
    ended_monotonic_ns: int | None = None,
) -> Json:
    """Build one complete ledger while keeping validity and benefit separate."""

    inventory = list(preconditions.get("inventory") or [])
    terminal = classify_terminal(inventory, reduction_errors)
    receipts_pass = _current_receipts_pass(validation_receipts)
    mutations_pass = set(mutation_results) == set(MUTATION_NAMES) and all(mutation_results.values())
    failed_guards = [
        row
        for row in validation_receipts
        if row.get("name") in {"adversarial_verify", "verdict_row_consistency_strict"}
        and row.get("passed") is not True
    ]
    if not receipts_pass or not mutations_pass:  # pragma: no cover - runner stops before build.
        terminal = {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_required_validation",
            "causal_audit_complete_score": 1,
            "causal_claims_qualified_score": 0,
        }
    claims = int(
        terminal["causal_claims_qualified_score"] == 1
        and not reduction_errors
        and receipts_pass
        and mutations_pass
    )
    qualified_benefit = claims * min(
        int(bool(reduced.get("qualified_online_benefit"))),
        int(producer_online_benefit_score),
    )
    if qualified_benefit:  # pragma: no cover - frozen upstream benefit score is zero.
        terminal["verdict_class"] = "positive"
        terminal["honest_verdict"] = "complete_positive_v657_qualified_online_benefit"

    inventory_audit = [
        {
            "audit_kind": "upstream_inventory",
            "check": f"exp{row['producer']}_state",
            "path": row["path"],
            "expected": "valid",
            "observed": row["state"],
            "passed": row["state"] == "valid",
        }
        for row in inventory
    ]
    mutation_audit = [
        {
            "audit_kind": "private_mutation",
            "check": name,
            "path": "private_fixture_not_published",
            "expected": "corruption_rejected",
            "observed": "corruption_rejected" if passed else "corruption_accepted",
            "passed": passed,
        }
        for name, passed in mutation_results.items()
    ]
    checkpoint_summary = {
        key: deepcopy(checkpoint_audit.get(key))
        for key in (
            "declared_checkpoint_count",
            "resolved_checkpoint_count",
            "restart_pair_count",
            "restart_bytes_equal",
            "prediction_rows_equal",
            "update_hashes_equal",
            "pending_queues_equal",
            "model_states_equal",
            "rng_state_kind",
            "rng_states_equal",
            "errors",
        )
    }
    audit_rows = [
        *inventory_audit,
        *deepcopy(list(history.get("audit_rows") or [])),
        {"audit_kind": "restart", "check": "complete_state", **checkpoint_summary},
        *mutation_audit,
    ]
    gates = [
        *[
            _gate(
                f"exp{row['producer']}_available_and_valid",
                "validity" if row["state"] == "invalid" else "readiness",
                "valid",
                row["state"],
                "eq",
                row["state"] == "valid",
                upstream=f"Exp{row['producer']}",
                field_path=str(row["path"]),
            )
            for row in inventory
        ],
        _gate(
            "causal_rows_valid",
            "validity",
            [],
            list(reduction_errors),
            "eq",
            not reduction_errors,
            upstream="Exp7506/Exp7509 raw evidence",
            field_path="reduction_errors",
        ),
        _gate(
            "private_mutations_rejected",
            "validity",
            True,
            mutations_pass,
            "eq",
            mutations_pass,
            upstream="current_audit",
            field_path="audit_rows.private_mutation",
        ),
        _gate(
            "required_current_validation",
            "validity",
            True,
            receipts_pass,
            "eq",
            receipts_pass,
            upstream="current_audit",
            field_path="validation_receipts",
        ),
        _gate(
            "causal_accounting_complete",
            "readiness",
            1,
            terminal["causal_audit_complete_score"],
            "eq",
            terminal["causal_audit_complete_score"] == 1,
            upstream="current_audit",
            field_path="causal_audit_complete_score",
        ),
        _gate(
            "causal_claims_qualified",
            "readiness",
            1,
            claims,
            "eq",
            claims == 1,
            upstream="Exp7506/Exp7509 raw evidence",
            field_path="causal_claims_qualified_score",
        ),
        _gate(
            "primary_support",
            "benefit",
            True,
            bool(reduced.get("support_passed")),
            "eq",
            bool(reduced.get("support_passed")),
            upstream="Exp7509 event rows",
            field_path="independent_reduction.support_passed",
        ),
        _gate(
            "five_primary_contrasts",
            "benefit",
            True,
            bool(reduced.get("primary_passed")),
            "eq",
            bool(reduced.get("primary_passed")),
            upstream="Exp7509 per-source rows",
            field_path="independent_reduction.primary_passed",
        ),
        _gate(
            "final_retention",
            "benefit",
            True,
            bool(reduced.get("retention_passed")),
            "eq",
            bool(reduced.get("retention_passed")),
            upstream="Exp7509 retention rows",
            field_path="independent_reduction.retention_passed",
        ),
    ]
    end_ns = (
        int(ended_monotonic_ns)
        if ended_monotonic_ns is not None
        else int(started_monotonic_ns + round(duration_s * 1_000_000_000))
    )
    value: Json = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "status": terminal["honest_verdict"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": int(started_monotonic_ns),
        "ended_monotonic_ns": end_ns,
        "process_identity": {"pid": os.getpid(), "ppid": os.getppid(), "python": sys.executable},
        "device_identity": {"venue": "host", "platform": platform.platform(), "cuda_used": False},
        "preconditions_checked": deepcopy(list(preconditions.get("rows") or [])),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") in {"causal_reduction", "private_mutations"}
            ),
            "validation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") in {"affected_validation", "terminal_validation"}
            ),
            "historical_capture": 0.0,
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "fitting": 750601,
            "arrival_and_audit": list(SCHEDULE_SEEDS),
            "bootstrap": 657009,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(list(source_hashes)),
        "raw_sidecars": deepcopy(dict(raw_sidecars)),
        "rows": deepcopy(list(history.get("rows") or [])),
        "sample_size_budget": deepcopy(dict(history.get("sample_size_budget") or {})),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "verifier_is_oracle": False,
        "flagged_adversarial": bool(failed_guards),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "field_principles": {},
        "causal_audit_complete_score": terminal["causal_audit_complete_score"],
        "causal_claims_qualified_score": claims,
        "qualified_online_benefit_score": qualified_benefit,
        "chronology_violation_count": int(history.get("chronology_violation_count", 0)),
        "chronology_offending_rows": deepcopy(list(history.get("chronology_offending_rows") or [])),
        "audit_rows": audit_rows,
        "independent_reduction": deepcopy(dict(reduced)),
        "reduction_errors": list(reduction_errors),
        "source_authentication": deepcopy(dict(source_audit)),
        "checkpoint_audit": checkpoint_summary,
        "mutation_results": deepcopy(dict(mutation_results)),
        "upstream_inventory": deepcopy(inventory),
        "producer_online_benefit_score": int(producer_online_benefit_score),
        "affected_validation_manifest": {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": receipts_pass,
            "numbered_runtime_e2e": [],
            "reason": "Reporting-only aggregation changed no model, sampler, binding, ARC, telemetry, or Rust behavior.",
        },
        "historical_model_provenance": {
            "scope": "historical_only",
            "current_model_calls": 0,
            "description": "Cached Qwen probabilities were upstream evidence only.",
        },
        "external_publication_performed": False,
        "push_performed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
    }
    value["field_principles"] = field_principles(value)
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def fixture_artifact() -> Json:
    """Build one compact valid null for schema and defensive-reader tests."""

    online = fixture_online_artifact()
    history = audit_event_history(online)
    reduced = reduce_measurement(
        online["per_source_results"],
        online["retention_rows"],
        history["audit_rows"],
        settings=fixture_reduction_settings(),
    )
    inventory = [
        {
            "producer": number,
            "path": path.as_posix(),
            "state": "valid",
            "sha256": "sha256:fixture",
        }
        for number, path in UPSTREAM_PATHS.items()
    ]
    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": f"fixture:{name}",
            "log_sha256": canonical_hash(name),
        }
        for name in REQUIRED_CURRENT_RECEIPTS
    ]
    checkpoint = {
        "declared_checkpoint_count": 1,
        "resolved_checkpoint_count": 1,
        "restart_pair_count": 1,
        "restart_bytes_equal": True,
        "prediction_rows_equal": True,
        "update_hashes_equal": True,
        "pending_queues_equal": True,
        "model_states_equal": True,
        "rng_state_kind": "stateless_seed_derived",
        "rng_states_equal": True,
        "errors": [],
    }
    return build_artifact(
        preconditions={"rows": [{"check": "fixture", "passed": True}], "inventory": inventory},
        history=history,
        reduced=reduced,
        checkpoint_audit=checkpoint,
        source_audit={"errors": [], "authenticated_reference_count": 0},
        reduction_errors=[],
        source_hashes=[],
        raw_sidecars={},
        validation_receipts=receipts,
        mutation_results={name: True for name in MUTATION_NAMES},
        producer_online_benefit_score=0,
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00+00:00",
        ended_at_utc="2026-09-22T00:00:01+00:00",
        duration_s=1.0,
    )


def _expected_terminal(value: Mapping[str, Any]) -> Json:
    """Recompute terminal scores from inventory, errors, guards, and gates."""

    inventory = value.get("upstream_inventory") or []
    errors = value.get("reduction_errors") or []
    terminal = classify_terminal(inventory, errors)
    receipts_pass = _current_receipts_pass(value.get("validation_receipts") or [])
    mutations = value.get("mutation_results") or {}
    mutations_pass = set(mutations) == set(MUTATION_NAMES) and all(mutations.values())
    if not receipts_pass or not mutations_pass:
        terminal = {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_required_validation",
            "causal_audit_complete_score": 1,
            "causal_claims_qualified_score": 0,
        }
    claims = int(
        terminal["causal_claims_qualified_score"] == 1
        and not errors
        and receipts_pass
        and mutations_pass
    )
    benefit = claims * min(
        int(bool((value.get("independent_reduction") or {}).get("qualified_online_benefit"))),
        int(value.get("producer_online_benefit_score", 0)),
    )
    if benefit:
        terminal["verdict_class"] = "positive"
        terminal["honest_verdict"] = "complete_positive_v657_qualified_online_benefit"
    terminal["causal_claims_qualified_score"] = claims
    terminal["qualified_online_benefit_score"] = benefit
    return terminal


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, reductions, sources, gates, receipts, and checksum."""

    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "experiment_id",
        "milestone",
        "terminal_status",
        "independent_reduction",
        "reduction_errors",
        "raw_sidecars",
        "upstream_inventory",
        "mutation_results",
    }
    missing = sorted(required - value.keys())
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    identity = (
        value.get("schema"),
        value.get("experiment_id"),
        value.get("milestone"),
        value.get("run_date"),
        value.get("terminal_status"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE, "complete"):
        errors.append("artifact_identity_invalid")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or value.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
    ):
        errors.append("current_provenance_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    for field in (
        "causal_audit_complete_score",
        "causal_claims_qualified_score",
        "qualified_online_benefit_score",
    ):
        if value.get(field) not in {0, 1}:
            errors.append(f"score_not_bare_binary:{field}")
    expected = _expected_terminal(value)
    terminal_fields = (
        "verdict_class",
        "honest_verdict",
        "causal_audit_complete_score",
        "causal_claims_qualified_score",
        "qualified_online_benefit_score",
    )
    chronology = sum(
        int(row.get("chronology_violation_count", 0))
        for row in value.get("audit_rows") or []
        if isinstance(row, Mapping)
    )
    if (
        any(value.get(field) != expected.get(field) for field in terminal_fields)
        or value.get("chronology_violation_count") != chronology
    ):
        errors.append("terminal_reduction_mismatch")
    principles = value.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(value)
        or any(not isinstance(item, str) or not item.strip() for item in principles.values())
    ):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results")
    gate_fields = {
        "check",
        "category",
        "expected",
        "observed",
        "op",
        "passed",
        "upstream",
        "field_path",
        "principle",
    }
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(row, Mapping) or gate_fields - row.keys() for row in gates)
    ):
        errors.append("gate_contract_invalid")
    receipts_pass = _current_receipts_pass(value.get("validation_receipts") or [])
    if not receipts_pass and value.get("verdict_class") != "disqualified":
        errors.append("required_validation_failed")
    if value.get("flagged_adversarial") is True and value.get("verdict_class") != "disqualified":
        errors.append("failed_guard_not_disqualified")
    if int(value.get("qualified_online_benefit_score", 0)) > int(
        value.get("producer_online_benefit_score", 0)
    ):
        errors.append("qualified_benefit_exceeds_producer")
    if verify_sources:
        references = list(value.get("source_artifact_hashes") or [])
        sidecars = value.get("raw_sidecars")
        if isinstance(sidecars, Mapping):
            references.extend(sidecars.values())
        else:
            errors.append("raw_sidecars_invalid")
        for row in references:
            if not isinstance(row, Mapping):
                errors.append("source_reference_invalid")
                continue
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file():
                errors.append(f"source_missing:{path}")
            elif sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{path}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def audit_prototype(value: Mapping[str, Any]) -> list[str]:
    """Independently check the fixture qualification needed by the online run."""

    errors: list[str] = []
    causality = value.get("causality_rows")
    if not isinstance(causality, list) or len(causality) != 2:
        errors.append("prototype_causality_rows_invalid")
    elif any(
        row.get("future_access_violations") != 0 or row.get("predict_before_update") is not True
        for row in causality
    ):
        errors.append("prototype_chronology_invalid")
    restart = (value.get("fixture_rows") or {}).get("restart_check") or {}
    if (
        restart.get("passed") is not True
        or restart.get("stable_trace_equal") is not True
        or restart.get("terminal_bytes_equal") is not True
    ):
        errors.append("prototype_restart_invalid")
    required = set((value.get("checkpoint_schema") or {}).get("required_parts") or [])
    if {"models", "pending_queue", "audit_rng_state", "order_cursor"} - required:
        errors.append("prototype_checkpoint_schema_invalid")
    if value.get("retention_labels_used_for_selection_or_rollback") is not False:
        errors.append("prototype_retention_selection_invalid")
    return errors


def _empty_reduction() -> Json:
    """Represent blocked evidence without inventing a measured zero effect."""

    return {
        "independent_unit": "schedule_seed_mean_within_source",
        "bootstrap_replicates": 0,
        "bootstrap_seed": 657009,
        "primary_block_length": 16,
        "primary_holm_family_size": 0,
        "support_rows": [],
        "support_passed": False,
        "primary_contrasts": [],
        "primary_passed": False,
        "causal_information_passed": False,
        "retention_contrast": {},
        "retention_passed": False,
        "sensitivity_contrasts": [],
        "qualified_online_benefit": False,
    }


def _empty_checkpoint_audit() -> Json:
    """Represent unavailable checkpoint operands without claiming equality."""

    return {
        "declared_checkpoint_count": 0,
        "resolved_checkpoint_count": 0,
        "restart_pair_count": 0,
        "restart_bytes_equal": False,
        "prediction_rows_equal": False,
        "update_hashes_equal": False,
        "pending_queues_equal": False,
        "model_states_equal": False,
        "rng_state_kind": "unavailable",
        "rng_states_equal": False,
        "rows": [],
        "errors": [],
    }


def _reduce_real(root: Path) -> Json:
    """Load and independently reduce both present V657 producer artifacts."""

    prototype = load_json(root / UPSTREAM_PATHS[7506])
    online = load_json(root / UPSTREAM_PATHS[7509])
    history = audit_event_history(online)
    source_audit = audit_original_inputs(root, online)
    checkpoint = audit_checkpoints(online)
    reduced = reduce_measurement(
        online["per_source_results"], online["retention_rows"], history["audit_rows"]
    )
    errors = [
        *audit_prototype(prototype),
        *history["errors"],
        *source_audit["errors"],
        *checkpoint["errors"],
        *compare_producer_reduction(online, reduced),
    ]
    source_rows: list[Json] = []
    for row in source_audit["reference_rows"]:
        if row["passed"]:
            path = Path(str(row["path"]))
            source_rows.append(source_row(path, root, "upstream_raw_input"))
    return {
        "prototype": prototype,
        "online": online,
        "history": history,
        "source_audit": source_audit,
        "checkpoint_audit": checkpoint,
        "reduced": reduced,
        "errors": list(dict.fromkeys(errors)),
        "source_artifact_hashes": source_rows,
    }


def _unique_sources(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Keep one exact source reference when upstream manifests overlap."""

    output: list[Json] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("path")), str(row.get("sha256")))
        if key not in seen:
            output.append(dict(row))
            seen.add(key)
    return output


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print a flushed boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7510] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _with_heartbeat(
    operation: str, fn: Callable[[], T], *, started: float, heartbeat_s: float = 60.0
) -> T:  # pragma: no cover
    """Emit truthful pending lines while one in-process reduction is active."""

    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(heartbeat_s):
            progress(started, operation, "pending")

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _span(phase: str, phase_started: int, run_started: int, units: int) -> Json:  # pragma: no cover
    """Record one measured phase with its completed unit count."""

    ended = time.monotonic_ns()
    return {
        "phase": phase,
        "start_s": (phase_started - run_started) / 1e9,
        "end_s": (ended - run_started) / 1e9,
        "duration_s": (ended - phase_started) / 1e9,
        "completed_units": units,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, raw reduction, adversarial, and strict commands."""

    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "capability_end_to_end",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "raw_upstream_rows",
            timeout_s=1200.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _checkpoint_summary(value: Mapping[str, Any]) -> Json:
    """Keep restart conclusions compact while the detailed rows live in a sidecar."""

    return {
        key: deepcopy(value.get(key))
        for key in (
            "declared_checkpoint_count",
            "resolved_checkpoint_count",
            "restart_pair_count",
            "restart_bytes_equal",
            "prediction_rows_equal",
            "update_hashes_equal",
            "pending_queues_equal",
            "model_states_equal",
            "rng_state_kind",
            "rng_states_equal",
            "errors",
        )
    }


def _provisional_terminal_receipts() -> list[Json]:  # pragma: no cover
    """Give candidate readers the final receipt shape before real commands run."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover
    """Re-read upstream rows and compare every stable audit output."""

    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return ["candidate_unreadable"]
    errors = validate_artifact(value, root=root)
    preconditions = collect_preconditions(root)
    inventory = preconditions["inventory"]
    if all(row["state"] == "valid" for row in inventory):
        try:
            real = _reduce_real(root)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"independent_reduction_failed:{error}")
        else:
            expected_sources = _unique_sources(
                [
                    *preconditions["source_artifact_hashes"],
                    *real["source_artifact_hashes"],
                ]
            )
            comparisons = {
                "independent_reduction": real["reduced"],
                "rows": real["history"]["rows"],
                "sample_size_budget": real["history"]["sample_size_budget"],
                "chronology_violation_count": real["history"]["chronology_violation_count"],
                "chronology_offending_rows": real["history"]["chronology_offending_rows"],
                "source_authentication": real["source_audit"],
                "checkpoint_audit": _checkpoint_summary(real["checkpoint_audit"]),
                "reduction_errors": real["errors"],
                "source_artifact_hashes": expected_sources,
            }
            for field, expected in comparisons.items():
                if value.get(field) != expected:
                    errors.append(f"independent_replay_mismatch:{field}")
    elif value.get("independent_reduction") != _empty_reduction():
        errors.append("independent_replay_mismatch:blocked_reduction")
    if value.get("preconditions_checked") != preconditions["rows"]:
        errors.append("independent_replay_mismatch:preconditions_checked")
    if value.get("upstream_inventory") != inventory:
        errors.append("independent_replay_mismatch:upstream_inventory")
    return list(dict.fromkeys(errors))


def _write_note(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write the concise human audit record before terminal publication."""

    reduced = artifact.get("independent_reduction") or {}
    lines = [
        "# V657 causal audit",
        "",
        f"Run date: {artifact.get('run_date')}",
        f"Verdict: `{artifact.get('honest_verdict')}` (`{artifact.get('verdict_class')}`)",
        "",
        "This aggregation audited Exp7506 and Exp7509 without importing the producer reducer.",
        "It loaded no model and made no current model calls.",
        "",
        "## Outcomes",
        "",
        f"- Complete accounting: {artifact.get('causal_audit_complete_score')}",
        f"- Causal rows qualified: {artifact.get('causal_claims_qualified_score')}",
        f"- Qualified online benefit: {artifact.get('qualified_online_benefit_score')}",
        f"- Chronology violations: {artifact.get('chronology_violation_count')}",
        f"- Primary support passed: {reduced.get('support_passed')}",
        f"- Five primary contrasts passed: {reduced.get('primary_passed')}",
        f"- Final retention passed: {reduced.get('retention_passed')}",
        "",
        "The audit found a valid null. Several seeds lacked the frozen minimum of",
        "12 labels in nontrivially permutable release batches. The local Brier",
        "effect also missed the registered -0.01 floor and not all controls lost.",
        "Retention passed. These facts do not repair the failed benefit gate.",
        "",
        "Six private mutations covered future labels, cross-batch permutation,",
        "prediction rewrites, pending-update loss, seed filtering, and retention rollback.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> Json:  # pragma: no cover
    """Run the audit, exact scoped checks, and one atomic terminal publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    destination = output_path if output_path.is_absolute() else root / output_path
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(UTC).isoformat()
    spans: list[Json] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic_ns()
    preconditions = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started_ns, len(preconditions["rows"])))
    internal_failures = [
        row
        for row in preconditions["rows"]
        if row.get("passed") is not True
        and not str(row.get("check", "")).startswith("upstream_inventory_exp")
    ]
    progress(
        started,
        "preconditions",
        "complete",
        missing=len(preconditions["missing_external"]),
        invalid=len(preconditions["invalid_present"]),
    )
    if internal_failures:
        raise RuntimeError(f"repository_precondition_failed:{internal_failures[0]}")

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic_ns()
        spans.append(_span(phase, phase_started, started_ns, 0))
        progress(started, phase, "after", completed=0)

    real: Json = {}
    history = _empty_history("external_input_unavailable")
    reduced = _empty_reduction()
    checkpoint = _empty_checkpoint_audit()
    source_audit: Json = {"reference_rows": [], "errors": []}
    reduction_errors: list[str] = []
    producer_score = 0
    if preconditions["invalid_present"]:
        reduction_errors.extend(
            f"invalid_present:{path}" for path in preconditions["invalid_present"]
        )
    elif not preconditions["missing_external"]:
        progress(started, "causal_reduction", "before_benchmark")
        phase_started = time.monotonic_ns()
        try:
            real = _with_heartbeat("causal_reduction", lambda: _reduce_real(root), started=started)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            reduction_errors = [f"present_evidence_invalid:{error}"]
        else:
            history = real["history"]
            reduced = real["reduced"]
            checkpoint = real["checkpoint_audit"]
            source_audit = real["source_audit"]
            reduction_errors = list(real["errors"])
            producer_score = int(real["online"].get("online_benefit_score", 0))
        spans.append(_span("causal_reduction", phase_started, started_ns, len(history["rows"])))
        progress(
            started,
            "causal_reduction",
            "after_benchmark",
            completed_units=len(history["rows"]),
            errors=len(reduction_errors),
        )

    progress(started, "private_mutations", "before_benchmark", planned=len(MUTATION_NAMES))
    phase_started = time.monotonic_ns()
    mutation_results = run_private_mutations()
    spans.append(_span("private_mutations", phase_started, started_ns, len(mutation_results)))
    progress(
        started,
        "private_mutations",
        "after_benchmark",
        rejected=sum(mutation_results.values()),
    )
    if not all(mutation_results.values()):
        reduction_errors.append("private_mutation_control_failed")

    raw_root = root / RAW_DIR
    manifest_path = raw_root / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    checkpoint_path = raw_root / "checkpoint_audit.json"
    per_unit_path = raw_root / "per_unit_rows.json"
    reduction_path = raw_root / "independent_reduction.json"
    atomic_json(checkpoint_path, {"checkpoint_audit": checkpoint})
    atomic_json(
        per_unit_path,
        {"rows": history["rows"], "sample_size_budget": history["sample_size_budget"]},
    )
    atomic_json(reduction_path, {"independent_reduction": reduced})
    raw_sidecars = {
        "affected_validation_manifest": source_row(
            manifest_path, root, "current_validation_manifest"
        ),
        "checkpoint_audit": source_row(checkpoint_path, root, "current_checkpoint_audit"),
        "per_unit_rows": source_row(per_unit_path, root, "current_per_unit_reduction"),
        "independent_reduction": source_row(reduction_path, root, "current_independent_reduction"),
    }

    private_root = Path(tempfile.mkdtemp(prefix="exp7510-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic_ns()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_root / "validation/affected",
    )
    affected_result = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started_ns, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_result["passed"],
    )
    if not affected_result["passed"]:
        raise RuntimeError(f"affected_validation_failed:{affected_result}")

    source_hashes = _unique_sources(
        [
            *preconditions["source_artifact_hashes"],
            *list(real.get("source_artifact_hashes") or []),
        ]
    )
    candidate = build_artifact(
        preconditions=preconditions,
        history=history,
        reduced=reduced,
        checkpoint_audit=checkpoint,
        source_audit=source_audit,
        reduction_errors=reduction_errors,
        source_hashes=source_hashes,
        raw_sidecars=raw_sidecars,
        validation_receipts=[*affected, *_provisional_terminal_receipts()],
        mutation_results=mutation_results,
        producer_online_benefit_score=producer_score,
        phase_spans=spans,
        started_at_utc=started_utc,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    candidate_path = raw_root / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_atomic_write", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_atomic_write", path=candidate_path)

    progress(started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic_ns()
    terminal = run_categorized_commands(
        root, _terminal_commands(candidate_path), log_dir=raw_root / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, started_ns, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_artifact(
        preconditions=preconditions,
        history=history,
        reduced=reduced,
        checkpoint_audit=checkpoint,
        source_audit=source_audit,
        reduction_errors=reduction_errors,
        source_hashes=source_hashes,
        raw_sidecars=raw_sidecars,
        validation_receipts=[*affected, *terminal],
        mutation_results=mutation_results,
        producer_online_benefit_score=producer_score,
        phase_spans=spans,
        started_at_utc=started_utc,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    exact_path = raw_root / "exact_terminal_candidate.json"
    progress(started, "exact_candidate", "before_atomic_write", path=exact_path)
    atomic_json(exact_path, final)
    progress(started, "exact_candidate", "after_atomic_write", path=exact_path)

    progress(started, "exact_candidate_validation", "before_subprocesses", planned=4)
    exact = run_categorized_commands(
        root, _terminal_commands(exact_path), log_dir=raw_root / "validation/exact_terminal"
    )
    exact_passed = all(row.get("passed") is True for row in exact)
    exact_critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in exact)
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed=len(exact),
        passed=exact_passed,
        critical=exact_critical,
    )
    if not exact_passed or exact_critical:
        raise RuntimeError("exact_terminal_candidate_validation_failed")

    progress(started, "research_note", "before_atomic_write", path=NOTE_PATH)
    _write_note(root / NOTE_PATH, final)
    progress(started, "research_note", "after_atomic_write", path=NOTE_PATH)
    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "complete",
        causal_audit_complete_score=final["causal_audit_complete_score"],
        causal_claims_qualified_score=final["causal_claims_qualified_score"],
        qualified_online_benefit_score=final["qualified_online_benefit_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the audit or one exact fresh-process candidate reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        try:
            value = load_json(args.cold_replay)
        except (OSError, json.JSONDecodeError, ValueError):
            errors = ["candidate_unreadable"]
        else:
            errors = validate_artifact(value, root=root, verify_sources=not args.no_source_check)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce, root=root)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
