"""Independently audit V658 static decisions and delayed feedback evidence.

The module reads producer evidence but never calls a producer headline reducer.
Missing science gets a complete blocked disposition. A measured null can still
qualify its branch because evidence validity and benefit are different claims.

Spec refs: REQ-REPORT-7525 and SCENARIO-REPORT-7525-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    atomic_json,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS


JsonDict = dict[str, Any]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7525-decision-audit"
SCHEMA = "carnot.exp7525.v658.decision_audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7525_v658_decision_audit.json")
RAW_DIR = Path("results/raw/experiment_7525_v658_decision_audit")
MODULE_PATH = Path("python/carnot/experiment_7525_v658_decision_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7525_v658_decision_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7525_v658_decision_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

CONTRACT_PATH = Path("results/experiment_7516_v658_contract_methods.json")
PROTOCOL_PATH = Path("results/experiment_7517_v658_source_protocol.json")
CAPTURE_PATH = Path("results/experiment_7520_v658_source_eval_capture.json")
STATIC_CHECKPOINT_PATH = Path("results/experiment_7521_v658_consistency_energy.json")
STATIC_PATH = Path("results/experiment_7522_v658_source_evaluation.json")
ONLINE_CHECKPOINT_PATH = Path("results/experiment_7523_v658_count_memory.json")
ONLINE_PATH = Path("results/experiment_7524_v658_count_online.json")

INPUT_CONTRACT = (
    ("contract", "shared", CONTRACT_PATH, "contract_ready_score"),
    ("source_protocol", "shared", PROTOCOL_PATH, "source_protocol_ready_score"),
    ("raw_capture", "shared", CAPTURE_PATH, "eval_capture_ready_score"),
    ("static_checkpoint", "static", STATIC_CHECKPOINT_PATH, "energy_fit_ready_score"),
    ("static_producer", "static", STATIC_PATH, "static_evaluation_complete_score"),
    ("online_checkpoint", "online", ONLINE_CHECKPOINT_PATH, "count_memory_ready_score"),
    ("online_producer", "online", ONLINE_PATH, "online_evaluation_complete_score"),
)

STATIC_GATES = {
    "registered_comparator": "same_information",
    "minimum_complete_groups": 4,
    "minimum_each_class": 2,
    "minimum_brier_reduction": 0.01,
    "maximum_log_loss_deterioration": 0.01,
    "minimum_decision_cost_reduction": 0.02,
    "minimum_accepted_coverage": 0.2,
}
ONLINE_GATES = {
    "minimum_complete_sources": 4,
    "minimum_delivered_labels": 4,
    "minimum_each_class": 2,
    "minimum_mixed_batch_labels": 4,
    "minimum_brier_reduction": 0.01,
    "maximum_retention_deterioration": 0.01,
    "block_length": 2,
}
EXPECTED_OPTION_MAPPING = {"option_0": "supported", "option_1": "unsupported"}
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
CLOSED_VERDICTS = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

V658_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so one edited row changes the audit identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes so parsed values cannot hide source replacement."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object without repairing malformed external evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def inventory_inputs(root: Path) -> list[JsonDict]:
    """Inventory every declared input, including absent same-milestone producers."""

    rows: list[JsonDict] = []
    for role, branch, relative, required_field in INPUT_CONTRACT:
        path = root / relative
        exists = path.is_file()
        value = _load_object(path) if exists else {}
        observed: Any = value.get(required_field) if value else None
        rows.append(
            {
                "role": role,
                "branch": branch,
                "path": relative.as_posix(),
                "required_field": required_field,
                "expected": 1,
                "observed": observed,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "passed": exists and bool(value) and observed == 1,
            }
        )
    return rows


def branch_gate_summary(inventory: Sequence[Mapping[str, Any]], branch: str) -> JsonDict:
    """Name every shared or branch-local prerequisite that did not pass."""

    relevant = [row for row in inventory if row.get("branch") in {"shared", branch}]
    failures = [
        {
            "check": f"{branch}_input:{row.get('role')}",
            "branch": branch,
            "path": row.get("path"),
            "required_field": row.get("required_field"),
            "expected": row.get("expected"),
            "observed": row.get("observed"),
            "exists": row.get("exists"),
            "passed": False,
        }
        for row in relevant
        if row.get("passed") is not True
    ]
    return {
        "branch": branch,
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def blocked_reduction(branch: str, inventory: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Represent external absence without inventing zero-valued measurements."""

    summary = branch_gate_summary(inventory, branch)
    return {
        "branch": branch,
        "errors": [],
        "claims_qualified_score": 0,
        "value_score": 0,
        "verdict_class": "blocked",
        "honest_verdict": f"complete_blocked_{branch}_required_input",
        "gate_check_summary": summary,
        "rows": [],
        "support": None,
    }


def metric_losses(probability: float, label: int) -> JsonDict:
    """Compute Brier and log loss while clipping only the logarithm operand."""

    clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
    return {
        "brier": (probability - label) ** 2,
        "log_loss": -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped)),
    }


def typed_decision(probability: float, label: int) -> tuple[str, float]:
    """Choose the frozen 5:1:0.2 action and return its realized absolute cost."""

    expected = {
        "accept": 5.0 * probability,
        "reject": 1.0 * (1.0 - probability),
        "escalate": 0.2,
    }
    action = min(expected, key=expected.__getitem__)
    realized = {
        "accept": 5.0 if label == 1 else 0.0,
        "reject": 1.0 if label == 0 else 0.0,
        "escalate": 0.2,
    }
    return action, realized[action]


def _probability(features: Mapping[str, Any], coefficients: Mapping[str, Any]) -> float:
    """Recompute a probability from the frozen ordered logistic coefficients."""

    order = coefficients.get("feature_order") or []
    values = coefficients.get("values") or []
    if len(order) != len(values):
        raise ValueError("coefficient_shape_invalid")
    logit = sum(
        float(features[name]) * float(weight) for name, weight in zip(order, values, strict=True)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def static_fixture() -> JsonDict:
    """Build a complete, supported static fixture with an honest zero delta."""

    coefficients = {"feature_order": ["bias", "signal"], "values": [0.0, 0.0]}
    raw_rows = [
        {
            "group_id": f"g{index}",
            "disposition": "complete",
            "condition": "original",
            "label": index % 2,
            "features": {"bias": 1.0, "signal": float(index - 2)},
            "option_mapping": deepcopy(EXPECTED_OPTION_MAPPING),
        }
        for index in range(4)
    ]
    raw_rows.append(
        {"group_id": "g_failed", "disposition": "failed", "failure": "owned_call_failed"}
    )
    result_rows: list[JsonDict] = []
    for row in raw_rows[:4]:
        label = int(row["label"])
        probability = _probability(row["features"], coefficients)
        action, cost = typed_decision(probability, label)
        losses = metric_losses(probability, label)
        result_rows.append(
            {
                "group_id": row["group_id"],
                "label": label,
                "candidate_probability": probability,
                "comparator_probability": probability,
                "candidate_brier": losses["brier"],
                "comparator_brier": losses["brier"],
                "candidate_log_loss": losses["log_loss"],
                "comparator_log_loss": losses["log_loss"],
                "candidate_action": action,
                "comparator_action": action,
                "arm_a_cost": cost,
                "arm_b_cost": cost,
                "no_headroom": True,
                "positive_claim": False,
                "headroom_explanation": "Both frozen arms choose the same action and absolute cost.",
            }
        )
    boundary_hash = "sha256:" + "a" * 64
    return {
        "registered_comparator": "same_information",
        "registered_gates": deepcopy(STATIC_GATES),
        "planned_group_ids": ["g0", "g1", "g2", "g3", "g_failed"],
        "frozen_coefficients": coefficients,
        "raw_rows": raw_rows,
        "changed_source_rows": [
            {"group_id": "g0", "condition": "absent", "label": None, "label_origin": None},
            {"group_id": "g0", "condition": "mismatched", "label": None, "label_origin": None},
        ],
        "per_source_results": result_rows,
        "hash_boundaries": {
            "raw": boundary_hash,
            "model": boundary_hash,
            "evaluator": boundary_hash,
        },
        "producer_summary": {
            "static_evaluation_complete_score": 1,
            "static_probability_value_score": 0,
            "typed_decision_value_score": 0,
            "verdict_class": "null",
            "positive_claim": False,
        },
    }


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    """Compare numeric operands without treating booleans as measurements."""

    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return (
            isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
        )
    return left == right


def reduce_static_evidence(value: Mapping[str, Any]) -> JsonDict:
    """Recompute every static fixture operand without a producer reducer."""

    errors: list[str] = []
    if value.get("registered_comparator") != STATIC_GATES["registered_comparator"]:
        errors.append("registered_comparator_changed")
    if value.get("registered_gates") != STATIC_GATES:
        errors.append("registered_gates_changed")

    planned = [str(item) for item in value.get("planned_group_ids") or []]
    raw_rows = [row for row in value.get("raw_rows") or [] if isinstance(row, Mapping)]
    raw_ids = [str(row.get("group_id")) for row in raw_rows]
    for group_id in planned:
        if group_id not in raw_ids:
            errors.append(f"planned_group_missing:{group_id}")
    for group_id, count in Counter(raw_ids).items():
        if count > 1:
            errors.append(f"duplicate_group:{group_id}")

    for row in raw_rows:
        if (
            row.get("disposition") == "complete"
            and row.get("option_mapping") != EXPECTED_OPTION_MAPPING
        ):
            errors.append(f"option_mapping_mismatch:{row.get('group_id')}")
    for row in value.get("changed_source_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("changed_source_row_invalid")
            continue
        if row.get("label") is not None or row.get("label_origin") not in {None, "changed_source"}:
            errors.append(f"changed_source_original_label_exposed:{row.get('group_id')}")

    boundaries = value.get("hash_boundaries") or {}
    if not isinstance(boundaries, Mapping) or len(set(boundaries.values())) != 1:
        errors.append("static_hash_boundary_mismatch")
    coefficients = value.get("frozen_coefficients") or {}
    stored = {
        str(row.get("group_id")): row
        for row in value.get("per_source_results") or []
        if isinstance(row, Mapping)
    }
    computed: list[JsonDict] = []
    for row in raw_rows:
        if row.get("disposition") != "complete":
            continue
        group_id = str(row.get("group_id"))
        observed = stored.get(group_id)
        if observed is None:
            errors.append(f"result_row_missing:{group_id}")
            continue
        try:
            label = int(row["label"])
            probability = _probability(row["features"], coefficients)
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append(f"prediction_operand_invalid:{group_id}")
            continue
        losses = metric_losses(probability, label)
        action, cost = typed_decision(probability, label)
        expected_fields = {
            "label": label,
            "candidate_probability": probability,
            "comparator_probability": probability,
            "candidate_brier": losses["brier"],
            "comparator_brier": losses["brier"],
            "candidate_log_loss": losses["log_loss"],
            "comparator_log_loss": losses["log_loss"],
            "candidate_action": action,
            "comparator_action": action,
            "arm_a_cost": cost,
            "arm_b_cost": cost,
        }
        for field, expected in expected_fields.items():
            if not _close(observed.get(field), expected):
                errors.append(f"static_operand_mismatch:{group_id}:{field}")
        no_headroom = _close(observed.get("arm_a_cost"), observed.get("arm_b_cost"))
        if no_headroom and (
            observed.get("positive_claim") is not False
            or observed.get("no_headroom") is not True
            or not str(observed.get("headroom_explanation") or "").strip()
        ):
            errors.append(f"positive_claim_without_headroom:{group_id}")
        computed.append(
            {
                "group_id": group_id,
                **expected_fields,
                "no_headroom": no_headroom,
                "positive_claim": False,
                "headroom_explanation": "Both frozen arms choose the same action and absolute cost.",
            }
        )

    labels = [int(row["label"]) for row in raw_rows if row.get("disposition") == "complete"]
    support = {
        "complete_groups": len(computed),
        "class_0": labels.count(0),
        "class_1": labels.count(1),
        "failed_groups": sum(row.get("disposition") == "failed" for row in raw_rows),
    }
    support_passed = (
        support["complete_groups"] >= STATIC_GATES["minimum_complete_groups"]
        and min(support["class_0"], support["class_1"]) >= STATIC_GATES["minimum_each_class"]
    )
    brier_deltas = [row["candidate_brier"] - row["comparator_brier"] for row in computed]
    cost_deltas = [row["arm_a_cost"] - row["arm_b_cost"] for row in computed]
    mean_brier_delta = sum(brier_deltas) / len(brier_deltas) if brier_deltas else None
    mean_cost_delta = sum(cost_deltas) / len(cost_deltas) if cost_deltas else None
    probability_benefit = bool(
        support_passed
        and mean_brier_delta is not None
        and mean_brier_delta <= -STATIC_GATES["minimum_brier_reduction"]
    )
    decision_benefit = bool(
        support_passed
        and mean_cost_delta is not None
        and mean_cost_delta <= -STATIC_GATES["minimum_decision_cost_reduction"]
    )
    producer = value.get("producer_summary") or {}
    computed_value = int(probability_benefit and decision_benefit)
    if (
        producer.get("verdict_class") in {"positive", "circular_positive"}
        or producer.get("positive_claim") is True
        or producer.get("static_probability_value_score") == 1
        or producer.get("typed_decision_value_score") == 1
    ) and not computed_value:
        errors.append("static_false_positive_claim")
    qualified = int(not errors and producer.get("static_evaluation_complete_score") == 1)
    return {
        "branch": "static",
        "errors": list(dict.fromkeys(errors)),
        "claims_qualified_score": qualified,
        "value_score": int(qualified and computed_value),
        "verdict_class": "disqualified" if errors else ("positive" if computed_value else "null"),
        "honest_verdict": "complete_static_invalid"
        if errors
        else "complete_null_static_no_registered_benefit",
        "support": support,
        "support_passed": support_passed,
        "mean_brier_delta": mean_brier_delta,
        "brier_upper95": max(brier_deltas) if brier_deltas else None,
        "mean_decision_cost_delta": mean_cost_delta,
        "decision_cost_upper95": max(cost_deltas) if cost_deltas else None,
        "rows": computed,
    }


def strict_null_fixture() -> JsonDict:
    """Expose honest absolute-cost null rows to the unchanged strict guard."""

    reduction = reduce_static_evidence(static_fixture())
    return {
        "honest_verdict": "complete_null_no_headroom",
        "verdict_class": "null",
        "gate_met": False,
        "rows": deepcopy(reduction["rows"]),
    }


def online_fixture() -> JsonDict:
    """Build a compact delayed-release ledger with independent learner states."""

    probabilities = {arm: 0.5 for arm in ("frozen", "global", "local", "shuffled")}
    events: list[JsonDict] = [
        {
            "event_type": "predict",
            "event_index": 0,
            "source_id": "s0",
            "bin": "b0",
            "probabilities": deepcopy(probabilities),
            "private_label": None,
        },
        {
            "event_type": "predict",
            "event_index": 1,
            "source_id": "s1",
            "bin": "b0",
            "probabilities": deepcopy(probabilities),
            "private_label": None,
        },
        {
            "event_type": "release",
            "event_index": 2,
            "release_id": "r0",
            "labels": {"s0": 0, "s1": 1},
            "donors": {"s0": "s1", "s1": "s0"},
        },
        {
            "event_type": "update",
            "event_index": 3,
            "update_id": "u0",
            "source_id": "s0",
            "release_id": "r0",
            "release_event_index": 2,
            "bin": "b0",
            "labels": {"global": 0, "local": 0, "shuffled": 1},
        },
        {
            "event_type": "update",
            "event_index": 4,
            "update_id": "u1",
            "source_id": "s1",
            "release_id": "r0",
            "release_event_index": 2,
            "bin": "b0",
            "labels": {"global": 1, "local": 1, "shuffled": 0},
        },
        {
            "event_type": "predict",
            "event_index": 5,
            "source_id": "s2",
            "bin": "b0",
            "probabilities": deepcopy(probabilities),
            "private_label": None,
        },
        {
            "event_type": "predict",
            "event_index": 6,
            "source_id": "s3",
            "bin": "b0",
            "probabilities": deepcopy(probabilities),
            "private_label": None,
        },
        {
            "event_type": "release",
            "event_index": 7,
            "release_id": "r1",
            "labels": {"s2": 0, "s3": 1},
            "donors": {"s2": "s3", "s3": "s2"},
        },
        {
            "event_type": "update",
            "event_index": 8,
            "update_id": "u2",
            "source_id": "s2",
            "release_id": "r1",
            "release_event_index": 7,
            "bin": "b0",
            "labels": {"global": 0, "local": 0, "shuffled": 1},
        },
        {
            "event_type": "update",
            "event_index": 9,
            "update_id": "u3",
            "source_id": "s3",
            "release_id": "r1",
            "release_event_index": 7,
            "bin": "b0",
            "labels": {"global": 1, "local": 1, "shuffled": 0},
        },
    ]
    per_source = []
    for index in range(4):
        label = index % 2
        loss = metric_losses(0.5, label)
        per_source.append(
            {
                "source_id": f"s{index}",
                "label": label,
                "probabilities": deepcopy(probabilities),
                "brier": {arm: loss["brier"] for arm in probabilities},
                "log_loss": {arm: loss["log_loss"] for arm in probabilities},
            }
        )
    boundary_hash = "sha256:" + "b" * 64
    return {
        "registered_gates": deepcopy(ONLINE_GATES),
        "events": events,
        "per_source_results": per_source,
        "retention_rows": [
            {
                "source_id": f"s{index}",
                "label": index % 2,
                "before_probability": 0.5,
                "after_probability": 0.5,
                "read_only": True,
            }
            for index in range(4)
        ],
        "checkpoint_manifest": {
            "uninterrupted_hash": boundary_hash,
            "resumed_hash": boundary_hash,
            "zero_step_hash": boundary_hash,
        },
        "hash_boundaries": {
            "raw": boundary_hash,
            "model": boundary_hash,
            "evaluator": boundary_hash,
        },
        "producer_summary": {
            "online_evaluation_complete_score": 1,
            "online_information_value_score": 0,
            "restart_parity_score": 1,
            "verdict_class": "null",
            "positive_claim": False,
        },
    }


def _count_probability(state: Mapping[str, int]) -> float:
    """Return the frozen Beta(1,1) posterior mean from released counts."""

    return (state["positive"] + 1.0) / (state["positive"] + state["negative"] + 2.0)


def reduce_online_evidence(value: Mapping[str, Any]) -> JsonDict:
    """Replay prediction, reveal, and update events in exact event order."""

    errors: list[str] = []
    if value.get("registered_gates") != ONLINE_GATES:
        errors.append("online_registered_gates_changed")
    boundaries = value.get("hash_boundaries") or {}
    if not isinstance(boundaries, Mapping) or len(set(boundaries.values())) != 1:
        errors.append("online_hash_boundary_mismatch")
    events = [row for row in value.get("events") or [] if isinstance(row, Mapping)]
    event_indices = [row.get("event_index") for row in events]
    if len(event_indices) != len(set(event_indices)):
        errors.append("duplicate_event_index")
    states = {arm: {"positive": 0, "negative": 0} for arm in ("global", "local", "shuffled")}
    predictions: dict[str, JsonDict] = {}
    releases: dict[str, JsonDict] = {}
    source_releases: dict[str, JsonDict] = {}
    update_ids: set[str] = set()
    chronology_violations = 0
    for event in sorted(events, key=lambda row: float(row.get("event_index", -1))):
        kind = event.get("event_type")
        source_id = str(event.get("source_id") or "")
        if kind == "predict":
            if event.get("private_label") is not None:
                errors.append(f"private_label_visible_before_reveal:{source_id}")
            observed = event.get("probabilities") or {}
            expected = {
                "frozen": 0.5,
                **{arm: _count_probability(state) for arm, state in states.items()},
            }
            for arm, probability in expected.items():
                if not _close(observed.get(arm), probability):
                    errors.append(f"online_probability_mismatch:{source_id}:{arm}")
            predictions[source_id] = {
                "event_index": event.get("event_index"),
                "probabilities": expected,
            }
        elif kind == "release":
            release_id = str(event.get("release_id") or "")
            releases[release_id] = dict(event)
            labels = event.get("labels") or {}
            donors = event.get("donors") or {}
            for released_source, label in labels.items():
                donor = donors.get(released_source)
                if donor == released_source or donor not in labels:
                    errors.append(f"donor_role_invalid:{released_source}")
                source_releases[str(released_source)] = {
                    "event_index": event.get("event_index"),
                    "release_id": release_id,
                    "label": label,
                    "donor_id": donor,
                    "donor_label": labels.get(donor),
                    "mixed": len(set(labels.values())) > 1,
                }
        elif kind == "update":
            update_id = str(event.get("update_id") or "")
            if update_id in update_ids:
                errors.append(f"duplicate_update:{update_id}")
            update_ids.add(update_id)
            released = source_releases.get(source_id)
            if released is None or float(event.get("event_index", -1)) <= float(
                (released or {}).get("event_index", math.inf)
            ):
                errors.append(f"update_before_release:{update_id}")
                chronology_violations += 1
                continue
            if (
                event.get("release_id") != released["release_id"]
                or event.get("release_event_index") != released["event_index"]
            ):
                errors.append(f"release_reference_mismatch:{update_id}")
            labels = event.get("labels") or {}
            expected_labels = {
                "global": released["label"],
                "local": released["label"],
                "shuffled": released["donor_label"],
            }
            for arm, label in expected_labels.items():
                if labels.get(arm) != label:
                    errors.append(f"update_label_mismatch:{update_id}:{arm}")
                else:
                    key = "positive" if int(label) == 1 else "negative"
                    states[arm][key] += 1
        else:
            errors.append(f"event_type_invalid:{kind}")

    stored = {
        str(row.get("source_id")): row
        for row in value.get("per_source_results") or []
        if isinstance(row, Mapping)
    }
    computed_rows: list[JsonDict] = []
    for source_id, prediction in predictions.items():
        released = source_releases.get(source_id)
        observed = stored.get(source_id)
        if released is None:
            errors.append(f"source_not_released:{source_id}")
            continue
        if float(released["event_index"]) <= float(prediction["event_index"]):
            errors.append(f"reveal_not_after_prediction:{source_id}")
            chronology_violations += 1
        if observed is None:
            errors.append(f"online_result_missing:{source_id}")
            continue
        label = int(released["label"])
        probabilities = prediction["probabilities"]
        brier = {
            arm: metric_losses(probability, label)["brier"]
            for arm, probability in probabilities.items()
        }
        log_loss = {
            arm: metric_losses(probability, label)["log_loss"]
            for arm, probability in probabilities.items()
        }
        if observed.get("label") != label:
            errors.append(f"online_label_mismatch:{source_id}")
        for arm in probabilities:
            if not _close((observed.get("probabilities") or {}).get(arm), probabilities[arm]):
                errors.append(f"online_stored_probability_mismatch:{source_id}:{arm}")
            if not _close((observed.get("brier") or {}).get(arm), brier[arm]):
                errors.append(f"online_brier_mismatch:{source_id}:{arm}")
            if not _close((observed.get("log_loss") or {}).get(arm), log_loss[arm]):
                errors.append(f"online_log_loss_mismatch:{source_id}:{arm}")
        computed_rows.append(
            {
                "source_id": source_id,
                "label": label,
                "probabilities": probabilities,
                "brier": brier,
                "log_loss": log_loss,
                "release_event_index": released["event_index"],
                "donor_id": released["donor_id"],
            }
        )

    labels = [int(row["label"]) for row in source_releases.values()]
    support = {
        "complete_sources": len(computed_rows),
        "delivered_labels": len(source_releases),
        "class_0": labels.count(0),
        "class_1": labels.count(1),
        "mixed_batch_labels": sum(bool(row["mixed"]) for row in source_releases.values()),
    }
    support_passed = (
        support["complete_sources"] >= ONLINE_GATES["minimum_complete_sources"]
        and support["delivered_labels"] >= ONLINE_GATES["minimum_delivered_labels"]
        and min(support["class_0"], support["class_1"]) >= ONLINE_GATES["minimum_each_class"]
        and support["mixed_batch_labels"] >= ONLINE_GATES["minimum_mixed_batch_labels"]
    )
    contrast_deltas = {
        arm: [row["brier"]["local"] - row["brier"][arm] for row in computed_rows]
        for arm in ("frozen", "global", "shuffled")
    }
    contrast_means = {
        arm: (sum(values) / len(values) if values else None)
        for arm, values in contrast_deltas.items()
    }
    benefit = bool(
        support_passed
        and all(
            mean is not None and mean <= -ONLINE_GATES["minimum_brier_reduction"]
            for mean in contrast_means.values()
        )
    )
    retention_deltas: list[float] = []
    for row in value.get("retention_rows") or []:
        if not isinstance(row, Mapping) or row.get("read_only") is not True:
            errors.append("retention_row_not_read_only")
            continue
        label = int(row.get("label", -1))
        before = metric_losses(float(row.get("before_probability", -1)), label)["brier"]
        after = metric_losses(float(row.get("after_probability", -1)), label)["brier"]
        retention_deltas.append(after - before)
    retention_upper95 = max(retention_deltas) if retention_deltas else None
    if (
        retention_upper95 is None
        or retention_upper95 > ONLINE_GATES["maximum_retention_deterioration"]
    ):
        benefit = False

    checkpoint = value.get("checkpoint_manifest") or {}
    restart_parity = int(
        isinstance(checkpoint, Mapping)
        and len(
            {
                checkpoint.get("uninterrupted_hash"),
                checkpoint.get("resumed_hash"),
                checkpoint.get("zero_step_hash"),
            }
        )
        == 1
    )
    if not restart_parity:
        errors.append("restart_checkpoint_mismatch")
    producer = value.get("producer_summary") or {}
    if (
        producer.get("verdict_class") in {"positive", "circular_positive"}
        or producer.get("positive_claim") is True
        or producer.get("online_information_value_score") == 1
    ) and not benefit:
        errors.append("online_false_positive_claim")
    qualified = int(
        not errors
        and producer.get("online_evaluation_complete_score") == 1
        and producer.get("restart_parity_score") == restart_parity == 1
    )
    return {
        "branch": "online",
        "errors": list(dict.fromkeys(errors)),
        "claims_qualified_score": qualified,
        "value_score": int(qualified and benefit),
        "verdict_class": "disqualified" if errors else ("positive" if benefit else "null"),
        "honest_verdict": "complete_online_invalid"
        if errors
        else "complete_null_online_no_registered_benefit",
        "support": support,
        "support_passed": support_passed,
        "chronology_violations": chronology_violations,
        "contrast_mean_brier_deltas": contrast_means,
        "contrast_upper95": {
            arm: (max(values) if values else None) for arm, values in contrast_deltas.items()
        },
        "retention_upper95_deterioration": retention_upper95,
        "restart_parity_score": restart_parity,
        "rows": computed_rows,
    }


def apply_private_mutation(static: JsonDict, online: JsonDict, name: str) -> None:
    """Apply one named corruption only to private in-memory fixtures."""

    if name == "label_leakage":
        online["events"][0]["private_label"] = 0
    elif name == "swapped_option_mapping":
        static["raw_rows"][0]["option_mapping"] = {
            "option_0": "unsupported",
            "option_1": "supported",
        }
    elif name == "omitted_failed_group":
        static["raw_rows"] = [row for row in static["raw_rows"] if row["group_id"] != "g_failed"]
    elif name == "future_release":
        update = next(row for row in online["events"] if row.get("update_id") == "u0")
        update["event_index"] = 1.5
    elif name == "duplicated_update":
        update = next(row for row in online["events"] if row.get("update_id") == "u0")
        duplicate = deepcopy(update)
        duplicate["event_index"] = 3.5
        online["events"].append(duplicate)
    elif name == "changed_comparator":
        static["registered_comparator"] = "post_selected_comparator"
    elif name == "edited_gate":
        static["registered_gates"]["minimum_complete_groups"] = 3
    elif name == "false_positive_class":
        static["producer_summary"].update(
            {
                "verdict_class": "positive",
                "static_probability_value_score": 1,
                "positive_claim": True,
            }
        )
    elif name == "majority_identical_cost":
        static["per_source_results"][0].update(
            {"positive_claim": True, "no_headroom": False, "headroom_explanation": ""}
        )
    else:
        raise ValueError(f"unknown_mutation:{name}")


def run_private_mutations() -> list[JsonDict]:
    """Prove each required corruption fails without returning corrupted bytes."""

    names = (
        "label_leakage",
        "swapped_option_mapping",
        "omitted_failed_group",
        "future_release",
        "duplicated_update",
        "changed_comparator",
        "edited_gate",
        "false_positive_class",
        "majority_identical_cost",
    )
    rows: list[JsonDict] = []
    for name in names:
        static = static_fixture()
        online = online_fixture()
        apply_private_mutation(static, online, name)
        errors = [
            *reduce_static_evidence(static)["errors"],
            *reduce_online_evidence(online)["errors"],
        ]
        rows.append(
            {
                "mutation": name,
                "rejected": bool(errors),
                "errors": errors,
                "corrupted_payload_retained": False,
            }
        )
    return rows


def classify_overall(*, static: str, online: str, validation_passed: bool) -> tuple[str, str]:
    """Give audit validity precedence while preserving external blocked states."""

    if not validation_passed:
        return "disqualified", "complete_disqualified_required_audit_validation"
    if "blocked" in {static, online}:
        return "blocked", "complete_blocked_required_science_incomplete"
    if "disqualified" in {static, online}:
        return "disqualified", "complete_disqualified_required_science_invalid"
    if "positive" in {static, online}:
        return "positive", "complete_positive_independently_recomputed_decision_benefit"
    return "null", "complete_null_both_decision_branches_qualified"


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Reuse the fixed scoped command plan with private temp and coverage paths."""

    return build_command_plan(root, V658_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject expanded tests, command drift, or missing private parents."""

    return validate_command_plan(root, V658_MANIFEST, commands)


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]], publication_state: str) -> bool:
    """Require one successful receipt for every check allowed at this boundary."""

    required = list(validation_scope.REQUIRED_CHECK_NAMES)
    if publication_state == "terminal":
        required.extend(TERMINAL_CHECK_NAMES)
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in required
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "==",
    principle: str,
    **context: Any,
) -> JsonDict:
    """Attach a fixed failure-prevention principle to one observable operand."""

    passed = observed == expected if op == "==" else bool(observed)
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
        **context,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed check and its exact expected and observed values."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _branch_row(reduction: Mapping[str, Any]) -> JsonDict:
    """Keep each branch disposition visible without converting missing data to zero."""

    summary = reduction.get("gate_check_summary") or {}
    return {
        "branch": reduction.get("branch"),
        "disposition": reduction.get("verdict_class"),
        "honest_verdict": reduction.get("honest_verdict"),
        "claims_qualified_score": reduction.get("claims_qualified_score"),
        "value_score": reduction.get("value_score"),
        "support": deepcopy(reduction.get("support")),
        "failed_input_paths": [row.get("path") for row in summary.get("failures") or []],
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each field while preserving ordinary scalar and mapping types."""

    specific = {
        "schema": "Version, experiment identity, and milestone bind the terminal reader contract.",
        "run_date": "Use the fixed run date with actual UTC and monotonic process boundaries.",
        "preconditions_checked": "Exact resource observations and hashes prevent invented readiness.",
        "MODEL_SPECS": "Aggregation performs no current model load, so the list stays empty.",
        "model_specs": "The lower-case model list mirrors the empty current model declaration.",
        "model_invoked": "False distinguishes current aggregation from historical model work.",
        "invocation_counts": "Balanced zero counters prove no current loads or generations occurred.",
        "inference_substrate_class": "The aggregation class prevents an incorrect model duration floor.",
        "inference_substrate": "Aggregation is exactly aggregation_from_upstream_artifacts.",
        "duration_s": "Measured elapsed work must never be padded to pass a floor.",
        "phase_spans": "Real phase timing exposes waiting, validation, and publication boundaries.",
        "random_seed": "Frozen selection, fitting, arrival, and bootstrap seeds prevent outcome tuning.",
        "reproducibility_checksum": "The checksum binds code settings, roles, source hashes, and rows.",
        "source_artifact_hashes": "Exact byte hashes detect drift across raw, model, and evaluator inputs.",
        "rows": "Both branch dispositions remain visible; missing evidence is not a numeric zero.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted units stay distinct.",
        "acceptance_gate_results": "Validity, support, and benefit checks retain expected and observed operands.",
        "gate_check_summary": "Every block names the exact path, field, expected value, and observation.",
        "honest_verdict": "A complete prefix preserves literal null, blocked, and invalid outcomes.",
        "verdict_class": "The closed enum prevents external absence from becoming partial owned work.",
        "verifier_is_oracle": "Human labels are evidence but are not formal proof.",
        "flagged_adversarial": "Actual findings remain visible and cannot be cleared to open a gate.",
        "validation_receipts": "Exact scoped commands and exits control terminal publication.",
        "field_principles": "Every emitted field states the failure that it prevents.",
        "audit_complete_score": "One means both branches have explicit dispositions under valid audit guards.",
        "decision_claims_qualified_score": "One includes independently reducible static null evidence.",
        "online_claims_qualified_score": "One includes valid event custody for an online null.",
        "qualified_static_value_score": "Static value cannot exceed independently recomputed benefit.",
        "qualified_online_value_score": "Online value cannot exceed independently recomputed causal benefit.",
        "mutation_results": "Accepted nulls and rejected false positives prevent audit self-certification.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in keys
    }


def _source_revision(root: Path) -> str | None:
    """Read the current Git revision without starting an unbounded child process."""

    head = root / ".git/HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            value = (root / ".git" / value[5:]).read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def _expected_terminal(value: Mapping[str, Any]) -> JsonDict:
    """Recompute final scores from branches, mutations, receipts, and flags."""

    publication_state = str(value.get("publication_state") or "")
    receipts = value.get("validation_receipts") or []
    mutations = value.get("mutation_results") or []
    mutations_passed = bool(mutations) and all(row.get("rejected") is True for row in mutations)
    validation_passed = (
        publication_state in {"candidate", "terminal"}
        and _required_receipts_pass(receipts, publication_state)
        and mutations_passed
        and value.get("flagged_adversarial") is False
    )
    branches = value.get("branch_reductions") or {}
    static = branches.get("static") or {}
    online = branches.get("online") or {}
    verdict, honest = classify_overall(
        static=str(static.get("verdict_class") or "disqualified"),
        online=str(online.get("verdict_class") or "disqualified"),
        validation_passed=validation_passed,
    )
    explicit = all(
        branch.get("verdict_class") in CLOSED_VERDICTS - {"partial"} for branch in (static, online)
    )
    audit_complete = int(validation_passed and explicit)
    decision_qualified = int(audit_complete and static.get("claims_qualified_score") == 1)
    online_qualified = int(audit_complete and online.get("claims_qualified_score") == 1)
    return {
        "validation_passed": validation_passed,
        "mutations_passed": mutations_passed,
        "verdict_class": verdict,
        "honest_verdict": honest,
        "audit_complete_score": audit_complete,
        "decision_claims_qualified_score": decision_qualified,
        "online_claims_qualified_score": online_qualified,
        "qualified_static_value_score": int(decision_qualified and static.get("value_score") == 1),
        "qualified_online_value_score": int(online_qualified and online.get("value_score") == 1),
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable source, role, row, gate, mutation, and validation evidence."""

    bound = {
        key: value.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "preconditions_checked",
            "source_artifact_hashes",
            "random_seed",
            "rows",
            "branch_reductions",
            "mutation_results",
            "acceptance_gate_results",
            "validation_manifest",
            "validation_receipts",
            "audit_complete_score",
            "decision_claims_qualified_score",
            "online_claims_qualified_score",
            "qualified_static_value_score",
            "qualified_online_value_score",
            "verdict_class",
        )
    }
    return canonical_hash(bound)


def build_artifact(
    *,
    inventory: Sequence[Mapping[str, Any]],
    static_reduction: Mapping[str, Any],
    online_reduction: Mapping[str, Any],
    mutation_results: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    publication_state: str,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Build one terminal ledger while keeping availability and benefit separate."""

    inventory_rows = [deepcopy(dict(row)) for row in inventory]
    branch_reductions = {
        "static": deepcopy(dict(static_reduction)),
        "online": deepcopy(dict(online_reduction)),
    }
    mutation_rows = [deepcopy(dict(row)) for row in mutation_results]
    receipt_rows = [deepcopy(dict(row)) for row in validation_receipts]
    flagged = any(
        row.get("name") == "adversarial_verify"
        and (row.get("passed") is not True or "CRITICAL" in str(row.get("output_tail") or ""))
        for row in receipt_rows
    )
    source_hashes = {
        str(row["path"]): row["sha256"]
        for row in inventory_rows
        if isinstance(row.get("sha256"), str)
    }
    gates = [
        _gate(
            "required_audit_validation",
            "validity",
            True,
            _required_receipts_pass(receipt_rows, publication_state),
            principle="Favorable science cannot excuse a failed scoped or cold validation check.",
        ),
        _gate(
            "private_mutation_guards",
            "validity",
            True,
            bool(mutation_rows) and all(row.get("rejected") is True for row in mutation_rows),
            principle="Each named misleading mutation must fail before the audit can qualify claims.",
        ),
    ]
    for row in inventory_rows:
        gates.append(
            _gate(
                f"input:{row['role']}",
                "availability",
                row["expected"],
                row["observed"],
                principle="Required upstream science must exist and declare numeric readiness before reduction.",
                branch=row["branch"],
                path=row["path"],
                required_field=row["required_field"],
                exists=row["exists"],
            )
        )
    gates.extend(
        [
            _gate(
                "static_claim_qualification",
                "readiness",
                1,
                static_reduction.get("claims_qualified_score"),
                principle="A valid static null remains qualified even when registered benefit is absent.",
            ),
            _gate(
                "online_claim_qualification",
                "readiness",
                1,
                online_reduction.get("claims_qualified_score"),
                principle="Valid event custody qualifies an online null without promoting its value.",
            ),
            _gate(
                "static_registered_benefit",
                "benefit",
                1,
                static_reduction.get("value_score"),
                principle="Static value requires the registered support, effect, and uncertainty gates.",
            ),
            _gate(
                "online_registered_benefit",
                "benefit",
                1,
                online_reduction.get("value_score"),
                principle="Online value requires causal contrasts, chronology, restart, and retention gates.",
            ),
        ]
    )
    static_support = static_reduction.get("support") or {}
    online_support = online_reduction.get("support") or {}
    duration_s = (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "title": "Independent V658 source-decision and feedback-information audit",
        "status": "measured_candidate",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "process_identity": {
            "pid": os.getpid(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "source_revision": _source_revision(root),
        },
        "publication_state": publication_state,
        "preconditions_checked": inventory_rows,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "historical_model_provenance": {
            "classification": "historical_only",
            "absorbed_into_current_invocation_counts": False,
        },
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "selection": 658021,
            "fitting": 658021,
            "arrival": 658024,
            "static_bootstrap": 658022,
            "online_bootstrap": 658025,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": [_branch_row(static_reduction), _branch_row(online_reduction)],
        "sample_size_budget": {
            "static": {
                "planned": 120,
                "attempted": (
                    static_support.get("complete_groups", 0)
                    + static_support.get("failed_groups", 0)
                    if static_support
                    else None
                ),
                "completed": static_support.get("complete_groups") if static_support else None,
                "excluded": 0 if static_support else None,
                "failed": static_support.get("failed_groups") if static_support else None,
                "censored": 0 if static_support else None,
                "unstarted": (
                    max(
                        0,
                        120
                        - static_support.get("complete_groups", 0)
                        - static_support.get("failed_groups", 0),
                    )
                    if static_support
                    else None
                ),
            },
            "online": {
                "planned": 120,
                "attempted": online_support.get("complete_sources") if online_support else None,
                "completed": online_support.get("complete_sources") if online_support else None,
                "excluded": 0 if online_support else None,
                "failed": 0 if online_support else None,
                "censored": 0 if online_support else None,
                "unstarted": (
                    max(0, 120 - online_support.get("complete_sources", 0))
                    if online_support
                    else None
                ),
            },
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": "",
        "verdict_class": "disqualified",
        "flagged_adversarial": flagged,
        "validation_receipts": receipt_rows,
        "validation_manifest": {
            "experiment_id": V658_MANIFEST.experiment_id,
            "test_paths": list(V658_MANIFEST.test_paths),
            "changed_modules": list(V658_MANIFEST.changed_modules),
            "static_paths": list(V658_MANIFEST.static_paths),
        },
        "field_principles": {},
        "audit_complete_score": 0,
        "decision_claims_qualified_score": 0,
        "online_claims_qualified_score": 0,
        "qualified_static_value_score": 0,
        "qualified_online_value_score": 0,
        "mutation_results": mutation_rows,
        "branch_reductions": branch_reductions,
        "capability_e2e": {
            "numbered_runtime_e2e": "not_applicable_reporting_only",
            "declared_entrypoint_executed": publication_state == "terminal",
            "cold_replay_passed": any(
                row.get("name") == "declared_entrypoint_cold_replay" and row.get("passed") is True
                for row in receipt_rows
            ),
        },
        "repository_health": {
            "status": "not_gated_by_unscoped_suite",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "push_performed": False,
        "publication_performed": False,
        "submission_performed": False,
    }
    expected = _expected_terminal(artifact)
    for field in (
        "honest_verdict",
        "verdict_class",
        "audit_complete_score",
        "decision_claims_qualified_score",
        "online_claims_qualified_score",
        "qualified_static_value_score",
        "qualified_online_value_score",
    ):
        artifact[field] = expected[field]
    artifact["status"] = artifact["honest_verdict"]
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, branch scores, receipts, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_specs") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_declaration_mismatch")
    if (
        artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
    ):
        errors.append("inference_substrate_mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    if str(artifact.get("honest_verdict") or "").startswith("complete_") is False:
        errors.append("honest_verdict_prefix_invalid")
    if artifact.get("publication_state") not in {"candidate", "terminal"}:
        errors.append("publication_state_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if (
        not _required_receipts_pass(
            artifact.get("validation_receipts") or [], str(artifact.get("publication_state"))
        )
        and artifact.get("verdict_class") != "disqualified"
    ):
        errors.append("required_validation_incomplete")
    rows = artifact.get("rows") or []
    if not isinstance(rows, list) or {row.get("branch") for row in rows} != {"static", "online"}:
        errors.append("branch_rows_incomplete")
    expected = _expected_terminal(artifact)
    score_fields = (
        "verdict_class",
        "honest_verdict",
        "audit_complete_score",
        "decision_claims_qualified_score",
        "online_claims_qualified_score",
        "qualified_static_value_score",
        "qualified_online_value_score",
    )
    if any(artifact.get(field) != expected[field] for field in score_fields):
        errors.append("terminal_scores_mismatch")
    for field in (
        "audit_complete_score",
        "decision_claims_qualified_score",
        "online_claims_qualified_score",
        "qualified_static_value_score",
        "qualified_online_value_score",
    ):
        if artifact.get(field) not in {0, 1} or isinstance(artifact.get(field), bool):
            errors.append(f"score_not_bare_binary:{field}")
    if artifact.get("qualified_static_value_score", 0) > artifact.get(
        "decision_claims_qualified_score", 0
    ):
        errors.append("static_value_exceeds_qualification")
    if artifact.get("qualified_online_value_score", 0) > artifact.get(
        "online_claims_qualified_score", 0
    ):
        errors.append("online_value_exceeds_qualification")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary") or {}
        if summary.get("first_failure") is None:
            errors.append("blocked_gate_summary_missing")
        if artifact.get("decision_claims_qualified_score") or artifact.get(
            "online_claims_qualified_score"
        ):
            errors.append("blocked_claim_qualification_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_replay(path: Path) -> list[str]:
    """Read one exact candidate and independently apply the cold terminal reader."""

    return validate_artifact(_load_object(path))


def terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build fresh entrypoint, reducer, adversarial, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    reducer = (
        "import pathlib,sys;"
        "from carnot.experiment_7525_v658_decision_audit import independent_replay;"
        "e=independent_replay(pathlib.Path(sys.argv[1]));"
        "print(e,flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--date", RUN_DATE, "--validate-candidate", str(candidate)),
            "capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    )
    return [
        PlannedCommand(spec, "completion" if index != 2 else "safety", True)
        for index, spec in enumerate(specs)
    ]


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print a flushed monotonic boundary before and after each long operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7525] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    """Record one real monotonic phase span and its completed unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _load_branch(
    root: Path, inventory: Sequence[Mapping[str, Any]], branch: str
) -> JsonDict:  # pragma: no cover
    """Read a producer only after every declared dependency says it ran and is ready."""

    summary = branch_gate_summary(inventory, branch)
    if not summary["all_passed"]:
        return blocked_reduction(branch, inventory)
    path = STATIC_PATH if branch == "static" else ONLINE_PATH
    value = _load_object(root / path)
    reduction = (
        reduce_static_evidence(value) if branch == "static" else reduce_online_evidence(value)
    )
    reduction["gate_check_summary"] = summary
    return reduction


def run_experiment(  # pragma: no cover - exercised through the declared capability entrypoint.
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Run both branch audits, scoped validation, cold readers, and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    repo = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    inventory = inventory_inputs(repo)
    progress(
        started,
        "preconditions",
        "end",
        completed=len(inventory),
        passed=sum(row["passed"] is True for row in inventory),
    )
    spans.append(_span("preconditions", phase_started, started, len(inventory)))

    phase_started = time.monotonic()
    progress(started, "scientific_reduction", "before_static")
    static = _load_branch(repo, inventory, "static")
    progress(started, "scientific_reduction", "after_static", disposition=static["verdict_class"])
    progress(started, "scientific_reduction", "before_online")
    online = _load_branch(repo, inventory, "online")
    progress(started, "scientific_reduction", "after_online", disposition=online["verdict_class"])
    mutations = run_private_mutations()
    progress(started, "scientific_reduction", "mutations_complete", completed=len(mutations))
    spans.append(_span("scientific_reduction", phase_started, started, 2 + len(mutations)))

    private_root = Path(tempfile.mkdtemp(prefix="exp7525-validation-", dir="/tmp"))
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    commands = build_validation_plan(repo, private_root)
    plan_errors = validate_validation_plan(repo, commands)
    atomic_json(
        raw_dir / "affected_validation_manifest.json",
        {
            "manifest": {
                "experiment_id": V658_MANIFEST.experiment_id,
                "test_paths": list(V658_MANIFEST.test_paths),
                "changed_modules": list(V658_MANIFEST.changed_modules),
                "static_paths": list(V658_MANIFEST.static_paths),
            },
            "plan_errors": plan_errors,
        },
    )
    progress(started, "validation", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            repo,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    progress(started, "validation", "after_affected_subprocesses", completed=len(affected))
    spans.append(_span("affected_validation", phase_started, started, len(affected)))

    candidate = build_artifact(
        inventory=inventory,
        static_reduction=static,
        online_reduction=online,
        mutation_results=mutations,
        validation_receipts=affected,
        publication_state="candidate",
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        root=repo,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_checkpoint_write", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_checkpoint_write", path=candidate_path)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_terminal_subprocesses")
    terminal = run_categorized_commands(
        repo,
        terminal_commands(repo, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    progress(started, "terminal_validation", "after_terminal_subprocesses", completed=len(terminal))
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))

    final = build_artifact(
        inventory=inventory,
        static_reduction=static,
        online_reduction=online,
        mutation_results=mutations,
        validation_receipts=[*affected, *terminal],
        publication_state="terminal",
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        root=repo,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publication", "before_atomic_terminal", path=output_path)
    atomic_json(repo / output_path, final)
    progress(started, "publication", "after_atomic_terminal", verdict=final["verdict_class"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and the fresh-process candidate reader mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate-candidate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or cold-read one exact pre-publication candidate."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate_candidate is not None:
        errors = independent_replay(args.validate_candidate)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
