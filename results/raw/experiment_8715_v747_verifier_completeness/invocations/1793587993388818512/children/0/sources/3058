"""Independently audit V662 frozen decisions and delayed learning.

This module reads producer bytes but recomputes their scientific operands. It
loads no model and makes no fresh claim from data that earlier tasks exposed.

Spec refs: REQ-REPORT-7579 and SCENARIO-REPORT-7579-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7579-v662-decision-learning-audit"
MILESTONE = "2026.09.662"
SCHEMA = "carnot.exp7579.v662.decision_learning_audit.v1"
RESULT_PATH = Path("results/experiment_7579_v662_decision_learning_audit.json")
RAW_DIR = Path("results/raw/experiment_7579_v662_decision_learning_audit")
MODULE_PATH = Path("python/carnot/experiment_7579_v662_decision_learning_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7579_v662_decision_learning_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7579_v662_decision_learning_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PRODUCERS = {
    "exp7575": Path("results/experiment_7575_v662_cached_learning_protocol.json"),
    "exp7576": Path("results/experiment_7576_v662_proper_loss_energy.json"),
    "exp7577": Path("results/experiment_7577_v662_proper_loss_evaluation.json"),
    "exp7578": Path("results/experiment_7578_v662_continuous_proper_loss.json"),
}
STATIC_ARMS = (
    "proper_loss_monotone",
    "raw_original",
    "temperature_original",
    "unconstrained_nine_knot",
)
LEARNING_ARMS = ("bounded", "raw", "global_count", "local_count", "shuffled_feedback")
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "source_artifact_hashes",
    "validation_receipts",
    "field_principles",
    "verifier_is_oracle",
    "static_claims_qualified_score",
    "learning_claims_qualified_score",
    "branch_conclusions",
    "fresh_confirmatory_claim_allowed",
)
ZERO_INVOCATION_COUNTS = {
    operation: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for operation in ("model_loads", "forward_calls", "generation_calls", "tokens")
}


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so a changed operand changes identity."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes instead of trusting a path or producer headline."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_receipt(path: Path, root: Path, upstream: str) -> JsonDict:
    """Bind an input to its exact bytes and distinguish its producer."""

    return {
        "upstream": upstream,
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def authenticate_source_receipt(receipt: Mapping[str, Any], root: Path) -> Path:
    path = Path(str(receipt.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != receipt.get("sha256"):
        raise ValueError(f"source_hash_mismatch:{path}")
    if resolved.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"source_size_mismatch:{path}")
    return resolved


def binary_energies(probability: float) -> tuple[float, float]:
    """Represent one finite Bernoulli probability as two energies."""

    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability_invalid")
    clipped = min(1.0 - 1e-12, max(1e-12, value))
    return -math.log1p(-clipped), -math.log(clipped)


def probability_from_binary_energies(energies: Sequence[float]) -> float:
    """Normalize both states so no saved probability is trusted."""

    if len(energies) != 2 or any(not math.isfinite(float(value)) for value in energies):
        raise ValueError("binary_energies_invalid")
    shifted = [
        math.exp(-float(value) + min(float(item) for item in energies)) for value in energies
    ]
    return shifted[1] / sum(shifted)


def _piecewise_probability(probability: float, theta: Sequence[float]) -> float:
    values = [float(value) for value in theta]
    if len(values) != 9 or any(not math.isfinite(value) for value in values):
        raise ValueError("head_theta_invalid")
    scaled = min(1.0, max(0.0, probability)) * 8.0
    if scaled >= 8.0:
        return values[-1]
    left = math.floor(scaled)
    fraction = scaled - left
    return values[left] * (1.0 - fraction) + values[left + 1] * fraction


def _head_probability(arm: str, probability: float, heads: Mapping[str, Any]) -> float:
    head = heads.get(arm)
    if not isinstance(head, Mapping):
        raise ValueError(f"static_head_missing:{arm}")
    if arm == "raw_original":
        mapped = probability
    elif arm == "temperature_original":
        temperature = float(head.get("temperature"))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_invalid")
        clipped = min(1.0 - 1e-12, max(1e-12, probability))
        logit = math.log(clipped / (1.0 - clipped)) / temperature
        mapped = 1.0 / (1.0 + math.exp(-logit))
    else:
        mapped = _piecewise_probability(probability, head.get("theta") or [])
    return probability_from_binary_energies(binary_energies(mapped))


def _typed_action(probability: float, label: int) -> tuple[str, float, bool]:
    costs = {"accept": 5.0 * probability, "reject": 1.0 - probability, "escalate": 0.2}
    minimum = min(costs.values())
    action = next(
        name
        for name in ("escalate", "accept", "reject")
        if math.isclose(costs[name], minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    realized = {
        "accept": 5.0 if label else 0.0,
        "reject": 0.0 if label else 1.0,
        "escalate": 0.2,
    }[action]
    return action, realized, action != "escalate"


def _static_metric_row(
    source: Mapping[str, Any], arm: str, probability: float, *, seed: int
) -> JsonDict:
    label = int(source.get("label"))
    if label not in (0, 1):
        raise ValueError("static_label_invalid")
    loss = (probability - label) ** 2
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    log_loss = -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))
    action, cost, covered = _typed_action(probability, label)
    return {
        "unit_id": str(source["source_id"]),
        "group_id": str(source.get("group_id") or source["source_id"]),
        "arm": arm,
        "q": probability,
        "label": label,
        "brier": loss,
        "log_loss": log_loss,
        "action": action,
        "realized_cost": cost,
        "non_escalated": covered,
        "raw_brier_numerator": loss,
        "raw_brier_denominator": 1,
        "raw_cost_numerator": cost,
        "raw_cost_denominator": 1,
        "metric_direction": "lower_loss_and_cost_are_better",
        "seed": seed,
        "censored": False,
        "provenance": "independent_exp7575_probability_exp7576_frozen_head",
    }


def reconstruct_static_rows(
    features: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], *, seed: int
) -> list[JsonDict]:
    """Rebuild all registered forecasts from raw probabilities and head values."""

    heads = manifest.get("heads")
    if not isinstance(heads, Mapping) or set(STATIC_ARMS) - set(heads):
        raise ValueError("static_head_roster_invalid")
    seen: set[str] = set()
    output: list[JsonDict] = []
    for feature in features:
        if any(key in feature for key in ("future_label", "online_label", "test_label")):
            raise ValueError("future_label_in_static_feature")
        source_id = str(feature.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError("static_source_roster_invalid")
        seen.add(source_id)
        original = float(feature.get("probability"))
        if not math.isfinite(original) or not 0.0 <= original <= 1.0:
            raise ValueError("static_probability_invalid")
        for arm in STATIC_ARMS:
            output.append(
                _static_metric_row(feature, arm, _head_probability(arm, original, heads), seed=seed)
            )
        output.append(_static_metric_row(feature, "escalate_all", 0.5, seed=seed))
        output[-1].update({"action": "escalate", "realized_cost": 0.2, "non_escalated": False})
    return output


def compare_static_rows(
    rebuilt: Sequence[Mapping[str, Any]], published: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Compare stable operands only after independent reconstruction."""

    errors: list[str] = []
    left = {(str(row.get("unit_id")), str(row.get("arm"))): row for row in rebuilt}
    right = {(str(row.get("unit_id")), str(row.get("arm"))): row for row in published}
    if len(left) != len(rebuilt) or len(right) != len(published) or set(left) != set(right):
        return ["static_row_roster_mismatch"]
    numeric = ("q", "brier", "realized_cost", "raw_brier_numerator", "raw_cost_numerator")
    for key in sorted(left):
        if any(
            not math.isclose(float(left[key][field]), float(right[key][field]), abs_tol=1e-10)
            for field in numeric
        ):
            errors.append("static_probability_mismatch")
            break
        if any(left[key].get(field) != right[key].get(field) for field in ("label", "action")):
            errors.append("static_decision_mismatch")
            break
    return errors


def _paired_interval(values: Sequence[float], indices: np.ndarray) -> JsonDict:
    array = np.asarray(values, dtype=float)
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "direction": "control_minus_candidate_positive_is_better",
        "positive_is_better_improvement": float(array.mean()),
        "lower95": float(lower),
        "upper95": float(upper),
        "raw_numerator": float(array.sum()),
        "raw_denominator": len(array),
    }


def reduce_static_rows(rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int) -> JsonDict:
    """Reduce source components once so arms never become independent units."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        unit = str(row.get("unit_id") or "")
        arm = str(row.get("arm") or "")
        if arm in grouped[unit]:
            raise ValueError("static_duplicate_row")
        grouped[unit][arm] = row
    expected = {*STATIC_ARMS, "escalate_all"}
    if not grouped or any(set(arms) != expected for arms in grouped.values()):
        raise ValueError("static_arm_roster_invalid")
    units = sorted(grouped)
    metrics: dict[str, JsonDict] = {}
    for arm in sorted(expected):
        arm_rows = [grouped[unit][arm] for unit in units]
        brier = sum(float(row["raw_brier_numerator"]) for row in arm_rows)
        cost = sum(float(row["raw_cost_numerator"]) for row in arm_rows)
        metrics[arm] = {
            "raw_brier_numerator": brier,
            "raw_brier_denominator": len(arm_rows),
            "mean_brier": brier / len(arm_rows),
            "raw_cost_numerator": cost,
            "raw_cost_denominator": len(arm_rows),
            "mean_cost": cost / len(arm_rows),
            "coverage": sum(bool(row["non_escalated"]) for row in arm_rows) / len(arm_rows),
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(units), size=(draws, len(units)))
    candidate = "proper_loss_monotone"
    contrasts = {
        comparator: _paired_interval(
            [
                float(grouped[unit][comparator]["brier"]) - float(grouped[unit][candidate]["brier"])
                for unit in units
            ],
            indices,
        )
        for comparator in ("raw_original", "temperature_original")
    }
    covered = sum(bool(grouped[unit][candidate]["non_escalated"]) for unit in units)
    return {
        "unit_count": len(units),
        "metrics": metrics,
        "contrasts": contrasts,
        "coverage": {
            "numerator": covered,
            "denominator": len(units),
            "fraction": covered / len(units),
        },
        "seed": seed,
        "draws": draws,
    }


def _release_blocks(
    rows: Sequence[Mapping[str, Any]], seed_names: Sequence[str]
) -> dict[str, list[Mapping[str, Any]]]:
    if not seed_names or len(rows) % len(seed_names):
        raise ValueError("release_ack_partition_invalid")
    width = len(rows) // len(seed_names)
    return {
        seed: list(rows[index * width : (index + 1) * width])
        for index, seed in enumerate(seed_names)
    }


def _design(probability: float) -> np.ndarray:
    row = np.zeros(9)
    scaled = min(1.0, max(0.0, probability)) * 8.0
    if scaled >= 8.0:
        row[-1] = 1.0
    else:
        left = math.floor(scaled)
        row[left] = 1.0 - (scaled - left)
        row[left + 1] = scaled - left
    return row


def _audit_final_parameters(
    state: Mapping[str, Any],
    events: Mapping[str, Mapping[str, Any]],
    releases: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Recompute final sufficient statistics directly from released labels."""

    errors: list[str] = []
    arms = state.get("arms")
    config = state.get("count_config")
    if not isinstance(arms, Mapping) or not isinstance(config, Mapping):
        return errors  # Compact test fixtures exercise chronology only.
    gram = np.zeros((9, 9))
    target = np.zeros(9)
    shuffled_target = np.zeros(9)
    global_mean = float(config["global_mean"])
    bin_means = [float(value) for value in config["bin_means"]]
    kappa = float(config["kappa"])
    global_counts = [kappa * global_mean, kappa * (1.0 - global_mean)]
    local_counts = [[kappa * mean, kappa * (1.0 - mean)] for mean in bin_means]
    released: list[str] = []
    for release in releases:
        feedback = list(release.get("feedback") or [])
        shuffled = list(release.get("shuffled_labels") or [])
        if len(shuffled) != len(feedback):
            errors.append("shuffled_label_roster_mismatch")
            continue
        for (event_id, label), shuffled_label in zip(feedback, shuffled):
            prediction = events[str(event_id)]
            probability = float(prediction["original_source_q"])
            design = _design(probability)
            gram += np.outer(design, design)
            target += int(label) * design
            shuffled_target += int(shuffled_label) * design
            global_counts[0] += int(label)
            global_counts[1] += 1 - int(label)
            index = min(7, math.floor(min(0.9999, max(0.0001, probability)) * 8.0))
            local_counts[index][0] += int(label)
            local_counts[index][1] += 1 - int(label)
            released.append(str(event_id))
    constrained = arms.get("constrained") or {}
    shuffled_arm = arms.get("shuffled_constrained") or {}
    if not np.allclose(np.asarray(constrained.get("gram")), gram, atol=1e-10):
        errors.append("constrained_gram_mismatch")
    if not np.allclose(np.asarray(constrained.get("target")), target, atol=1e-10):
        errors.append("constrained_target_mismatch")
    if not np.allclose(np.asarray(shuffled_arm.get("gram")), gram, atol=1e-10):
        errors.append("shuffled_gram_mismatch")
    if not np.allclose(np.asarray(shuffled_arm.get("target")), shuffled_target, atol=1e-10):
        errors.append("shuffled_target_mismatch")
    if constrained.get("sample_count") != len(released) or shuffled_arm.get("sample_count") != len(
        released
    ):
        errors.append("parameter_sample_count_mismatch")
    saved_global = (arms.get("global_count") or {}).get("counts")
    saved_local = (arms.get("local_count") or {}).get("counts")
    if not np.allclose(np.asarray(saved_global), np.asarray([global_counts]), atol=1e-10):
        errors.append("global_count_update_mismatch")
    if not np.allclose(np.asarray(saved_local), np.asarray(local_counts), atol=1e-10):
        errors.append("local_count_update_mismatch")
    for name in ("constrained", "shuffled_constrained"):
        theta = np.asarray((arms.get(name) or {}).get("theta"), dtype=float)
        if (
            theta.shape != (9,)
            or np.any(np.diff(theta) < -1e-7)
            or np.max(np.abs(theta - np.linspace(0.0, 1.0, 9))) > 0.1000001
        ):
            errors.append(f"{name}_constraint_mismatch")
    journal = state.get("journal") or []
    previous = "GENESIS"
    for index, row in enumerate(journal):
        body = {
            "sequence": row.get("sequence"),
            "operation": row.get("operation"),
            "payload": row.get("payload"),
            "previous_hash": row.get("previous_hash"),
        }
        if (
            body["sequence"] != index
            or body["previous_hash"] != previous
            or row.get("entry_hash") != canonical_hash(body)
        ):
            errors.append("persisted_journal_hash_mismatch")
            break
        previous = str(row["entry_hash"])
    for prediction in (state.get("predictions") or {}).values():
        for arm in (prediction.get("arms") or {}).values():
            if not math.isclose(
                probability_from_binary_energies(arm.get("energies") or []),
                float(arm.get("probability")),
                abs_tol=1e-12,
            ):
                errors.append("persisted_energy_probability_mismatch")
                return errors
    return errors


def audit_causal_rows(
    predictions: Sequence[Mapping[str, Any]],
    release_ack: Sequence[Mapping[str, Any]],
    states: Mapping[str, Any],
    orders: Mapping[str, Sequence[str]],
    *,
    block_size: int = 8,
) -> JsonDict:
    """Rebuild chronology without accepting producer lifecycle headlines."""

    errors: list[str] = []
    seed_names = sorted(str(seed) for seed in orders)
    prediction_by_seed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_by_seed[str(row.get("seed"))].append(row)
    try:
        blocks = _release_blocks(release_ack, seed_names)
    except ValueError as exc:
        return {"qualified": False, "errors": [str(exc)]}
    total_releases = total_updates = total_acks = 0
    persist_reload = False
    for seed in seed_names:
        expected_order = [str(value) for value in orders[seed]]
        observed_rows = prediction_by_seed.get(seed, [])
        observed_rows = sorted(observed_rows, key=lambda row: int(row.get("event_count", -1)))
        observed_order = [str(row.get("event_id") or "") for row in observed_rows]
        if observed_order != expected_order or [
            row.get("event_count") for row in observed_rows
        ] != list(range(1, len(expected_order) + 1)):
            errors.append("prediction_order_mismatch")
        if any(row.get("label_available_at_prediction") is not False for row in observed_rows):
            errors.append("future_label_visible")
        if any(set(row.get("arms") or {}) != set(LEARNING_ARMS) for row in observed_rows):
            errors.append("prediction_arm_roster_mismatch")
        events = {str(row.get("event_id")): row for row in observed_rows}
        seed_block = blocks[seed]
        releases = [row for row in seed_block if row.get("operation") == "release_update_persist"]
        acknowledgments = [
            row for row in seed_block if row.get("operation") == "durable_acknowledgment"
        ]
        if [row.get("release_index") for row in releases] != list(range(len(releases))):
            errors.append("release_order_mismatch")
        if [row.get("release_index") for row in acknowledgments] != list(range(len(releases))):
            errors.append("acknowledgment_order_mismatch")
        updated: set[str] = set()
        prior_hash = (
            str(observed_rows[0].get("state_hash_before_prediction")) if observed_rows else ""
        )
        for release in releases:
            index = int(release.get("release_index", -1))
            feedback = list(release.get("feedback") or [])
            event_ids = [
                str(pair[0]) for pair in feedback if isinstance(pair, list) and len(pair) == 2
            ]
            labels = [pair[1] for pair in feedback if isinstance(pair, list) and len(pair) == 2]
            expected_ids = expected_order[index * block_size : index * block_size + block_size]
            if event_ids != expected_ids or event_ids != [
                str(value) for value in release.get("event_ids") or []
            ]:
                errors.append("release_event_roster_mismatch")
            if len(event_ids) != len(set(event_ids)) or updated.intersection(event_ids):
                errors.append("duplicate_update")
            updated.update(event_ids)
            if any(label not in (0, 1) for label in labels):
                errors.append("release_label_invalid")
            if any(
                event_id not in events
                or int(events[event_id].get("event_count", 10**9))
                > int(release.get("event_count", -1))
                for event_id in event_ids
            ):
                errors.append("feedback_before_prediction")
            if release.get("update_count") != len(event_ids):
                errors.append("update_count_mismatch")
            if release.get("state_hash_before") != prior_hash:
                errors.append("state_hash_chain_mismatch")
            prior_hash = str(release.get("state_hash_after") or "")
            persist_reload = persist_reload or bool(release.get("persisted_state_hash_before_ack"))
        if any(row.get("ack_count") != 1 for row in acknowledgments):
            errors.append("acknowledgment_count_mismatch")
        state = states.get(seed)
        if not isinstance(state, Mapping):
            errors.append("persisted_state_missing")
            continue
        receipts = state.get("release_receipts")
        saved_predictions = state.get("predictions")
        if not isinstance(receipts, Mapping) or not isinstance(saved_predictions, Mapping):
            errors.append("persisted_state_shape_invalid")
            continue
        if set(saved_predictions) != set(expected_order):
            errors.append("persisted_prediction_roster_mismatch")
        if state.get("acknowledgments") != list(range(len(releases))):
            errors.append("persisted_acknowledgment_mismatch")
        if state.get("next_release_index") != len(releases):
            errors.append("persisted_release_index_mismatch")
        for release in releases:
            index = str(release["release_index"])
            receipt = receipts.get(index)
            if not isinstance(receipt, Mapping) or receipt.get("feedback") != release.get(
                "feedback"
            ):
                errors.append("persisted_update_mismatch")
        errors.extend(_audit_final_parameters(state, events, releases))
        total_releases += len(releases)
        total_updates += len(updated)
        total_acks += len(acknowledgments)
    return {
        "qualified": not errors,
        "errors": sorted(set(errors)),
        "prediction_count": len(predictions),
        "release_count": total_releases,
        "update_count": total_updates,
        "acknowledgment_count": total_acks,
        "persist_reload_exercised": persist_reload,
    }


def mutate_causal_fixture(value: JsonDict, mutation: str) -> None:
    """Apply one named private corruption in place for fail-closed tests."""

    if mutation == "future_label":
        value["predictions"][0]["label_available_at_prediction"] = True
    elif mutation == "order":
        value["orders"]["11"] = list(reversed(value["orders"]["11"]))
    elif mutation == "missing_row":
        value["predictions"].pop()
    elif mutation == "hash":
        value["release_ack"][0]["state_hash_before"] = "sha256:corrupted"
    elif mutation == "duplicate_update":
        duplicate = deepcopy(value["release_ack"][0]["feedback"][0])
        value["release_ack"][0]["feedback"].append(duplicate)
        value["release_ack"][0]["event_ids"].append(duplicate[0])
        value["release_ack"][0]["update_count"] += 1
    else:
        raise ValueError(f"unknown_mutation:{mutation}")


def _causal_fixture() -> JsonDict:
    predictions = [
        {
            "operation": "sealed_prediction",
            "event_id": event_id,
            "event_count": index + 1,
            "release_index": 0,
            "label_available_at_prediction": False,
            "state_hash_before_prediction": "sha256:s0",
            "original_source_q": probability,
            "arms": {arm: {"q": probability, "typed_action": "escalate"} for arm in LEARNING_ARMS},
            "seed": 11,
        }
        for index, (event_id, probability) in enumerate((("e0", 0.2), ("e1", 0.8)))
    ]
    feedback = [["e0", 0], ["e1", 1]]
    release = {
        "operation": "release_update_persist",
        "release_index": 0,
        "event_count": 2,
        "event_ids": ["e0", "e1"],
        "feedback": feedback,
        "update_count": 2,
        "state_hash_before": "sha256:s0",
        "state_hash_after": "sha256:s1",
        "persisted_state_hash_before_ack": "sha256:persisted",
        "original_labels": [0, 1],
        "shuffled_labels": [1, 0],
    }
    acknowledgment = {
        "operation": "durable_acknowledgment",
        "release_index": 0,
        "event_count": 2,
        "ack_count": 1,
        "state_hash": "sha256:final",
    }
    state = {
        "acknowledgments": [0],
        "next_release_index": 1,
        "predictions": {row["event_id"]: row for row in predictions},
        "release_receipts": {"0": {"release_index": 0, "feedback": feedback, "update_count": 2}},
    }
    return {
        "predictions": predictions,
        "release_ack": [release, acknowledgment],
        "states": {"11": state},
        "orders": {"11": ["e0", "e1"]},
    }


def run_private_mutations() -> list[JsonDict]:
    """Prove six private corruptions close their applicable qualification."""

    output: list[JsonDict] = []
    for mutation in ("future_label", "order", "missing_row", "hash", "duplicate_update"):
        fixture = _causal_fixture()
        mutate_causal_fixture(fixture, mutation)
        reduction = audit_causal_rows(**fixture, block_size=2)
        output.append(
            {
                "mutation": mutation,
                "passed": reduction.get("qualified") is False,
                "observed_errors": reduction.get("errors"),
                "corrupted_fixture_published": False,
            }
        )
    features = [
        {"source_id": "a", "probability": 0.25, "label": 0},
        {"source_id": "b", "probability": 0.75, "label": 1},
    ]
    manifest = {
        "heads": {
            "proper_loss_monotone": {"theta": np.linspace(0.0, 1.0, 9).tolist()},
            "raw_original": {},
            "temperature_original": {"temperature": 0.5},
            "unconstrained_nine_knot": {"theta": np.linspace(0.0, 1.0, 9).tolist()},
        }
    }
    rows = reconstruct_static_rows(features, manifest, seed=1)
    expected = reduce_static_rows(rows, draws=16, seed=1)
    corrupted = deepcopy(expected)
    corrupted["contrasts"]["raw_original"]["positive_is_better_improvement"] += 1.0
    sign_closed = canonical_hash(expected) != canonical_hash(corrupted)
    output.insert(
        3,
        {
            "mutation": "sign",
            "passed": sign_closed,
            "observed_errors": ["static_contrast_sign_mismatch"] if sign_closed else [],
            "corrupted_fixture_published": False,
        },
    )
    return output


def reduce_learning_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Reduce every raw online or retention row from its stored operands."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str, str, int]] = set()
    for row in rows:
        key = (
            str(row.get("unit_id")),
            int(row.get("seed", -1)),
            str(row.get("phase")),
            str(row.get("arm")),
            int(row.get("event_count", -1)),
        )
        if key in seen:
            raise ValueError("learning_metric_duplicate")
        seen.add(key)
        probability = float(row.get("q"))
        label = int(row.get("label"))
        expected = (probability - label) ** 2
        if not math.isclose(expected, float(row.get("raw_squared_error_numerator")), abs_tol=1e-12):
            raise ValueError("learning_metric_sign_or_value_mismatch")
        if row.get("raw_squared_error_denominator") != 1:
            raise ValueError("learning_metric_denominator_invalid")
        action, cost, covered = _typed_action(probability, label)
        if (
            row.get("typed_action") != action
            or not math.isclose(float(row.get("realized_action_cost")), cost, abs_tol=1e-12)
            or row.get("non_escalated") is not covered
        ):
            raise ValueError("learning_decision_mismatch")
        grouped[str(row.get("arm"))].append(row)
    if set(grouped) != set(LEARNING_ARMS):
        raise ValueError("learning_metric_arm_roster_invalid")
    output: dict[str, JsonDict] = {}
    for arm, arm_rows in grouped.items():
        numerator = sum(float(row["raw_squared_error_numerator"]) for row in arm_rows)
        cost = sum(float(row["realized_action_cost"]) for row in arm_rows)
        output[arm] = {
            "raw_squared_error_numerator": numerator,
            "raw_squared_error_denominator": len(arm_rows),
            "mean_brier": numerator / len(arm_rows),
            "raw_action_cost_numerator": cost,
            "raw_action_cost_denominator": len(arm_rows),
            "mean_action_cost": cost / len(arm_rows),
            "coverage_numerator": sum(bool(row["non_escalated"]) for row in arm_rows),
            "coverage_denominator": len(arm_rows),
            "coverage": sum(bool(row["non_escalated"]) for row in arm_rows) / len(arm_rows),
        }
    return output


def _interval(values: Sequence[float]) -> JsonDict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "lower95": None, "upper95": None}
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "lower95": float(np.quantile(array, 0.025)),
        "upper95": float(np.quantile(array, 0.975)),
    }


def _holm_one_sided(contrasts: Mapping[str, Sequence[float]]) -> JsonDict:
    raw = {
        name: (1.0 + sum(value <= 0.0 for value in values)) / (len(values) + 1.0)
        for name, values in contrasts.items()
    }
    ordered = sorted(raw, key=lambda name: (raw[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, raw[name] * (len(ordered) - index)))
        adjusted[name] = running
    return {
        name: {
            "one_sided_p": raw[name],
            "holm_adjusted_p": adjusted[name],
            "reject_at_0_05": adjusted[name] <= 0.05,
        }
        for name in raw
    }


def reduce_bootstrap_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute registered group-level learning and retention intervals."""

    by_unit: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    seeds: set[int] = set()
    for row in rows:
        denominator = int(row.get("raw_squared_error_denominator", 0))
        retention_denominator = int(row.get("retention_denominator", 0))
        cost_denominator = int(row.get("action_cost_denominator", 0))
        coverage_denominator = int(row.get("coverage_denominator", 0))
        if min(denominator, retention_denominator, cost_denominator, coverage_denominator) <= 0:
            raise ValueError("bootstrap_denominator_invalid")
        checks = (
            (float(row["raw_squared_error_numerator"]) / denominator, float(row["mean_brier"])),
            (
                float(row["retention_brier_numerator"]) / retention_denominator,
                float(row["retention_mean_brier"]),
            ),
            (
                float(row["action_cost_numerator"]) / cost_denominator,
                float(row["mean_action_cost"]),
            ),
            (float(row["coverage_numerator"]) / coverage_denominator, float(row["coverage"])),
        )
        if any(not math.isclose(left, right, abs_tol=1e-12) for left, right in checks):
            raise ValueError("bootstrap_raw_operand_mismatch")
        key = (str(row.get("unit_id")), int(row.get("seed")))
        arm = str(row.get("arm"))
        if arm in by_unit[key]:
            raise ValueError("bootstrap_duplicate_arm")
        by_unit[key][arm] = row
        seeds.add(int(row.get("seed")))
    if not by_unit or any(set(arms) != set(LEARNING_ARMS) for arms in by_unit.values()):
        raise ValueError("bootstrap_arm_roster_invalid")
    values: dict[str, list[float]] = defaultdict(list)
    retention: list[float] = []
    coverage: list[float] = []
    for arms in by_unit.values():
        candidate = arms["bounded"]
        for comparator in ("raw", "global_count", "local_count", "shuffled_feedback"):
            values[f"brier_vs_{comparator}"].append(
                float(arms[comparator]["mean_brier"]) - float(candidate["mean_brier"])
            )
        values["action_cost_vs_raw"].append(
            float(arms["raw"]["mean_action_cost"]) - float(candidate["mean_action_cost"])
        )
        retention.append(
            float(candidate["retention_mean_brier"]) - float(arms["raw"]["retention_mean_brier"])
        )
        coverage.append(float(candidate["coverage"]))
    contrasts = {name: _interval(items) for name, items in values.items()}
    holm_inputs = {name: items for name, items in values.items() if name.startswith("brier_vs_")}
    holm = _holm_one_sided(holm_inputs)
    complete = len(by_unit) == 5000 and len(seeds) == 5
    benefit = bool(
        complete
        and all(
            float(contrasts[name]["lower95"]) > 0.0 and holm[name]["reject_at_0_05"]
            for name in holm_inputs
        )
        and float(contrasts["action_cost_vs_raw"]["lower95"]) > 0.0
        and float(np.quantile(coverage, 0.025)) >= 0.10
    )
    return {
        "completed_replays": len(by_unit),
        "order_count": len(seeds),
        "contrasts": contrasts,
        "holm_one_sided": holm,
        "candidate_coverage": _interval(coverage),
        "retention_degradation": _interval(retention),
        "measurement_complete": complete,
        "benefit_passed": benefit,
        "retention_passed": bool(
            len(by_unit) == 5000 and float(np.quantile(retention, 0.975)) <= 0.005
        ),
    }


def _selected_metric_errors(
    observed: Mapping[str, Any], expected: Mapping[str, Any], fields: Sequence[str]
) -> list[str]:
    errors: list[str] = []
    for arm, metrics in observed.items():
        published = expected.get(arm)
        if not isinstance(published, Mapping):
            errors.append(f"producer_metric_arm_missing:{arm}")
            continue
        for field in fields:
            left = metrics.get(field)
            right = published.get(field)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if not math.isclose(float(left), float(right), abs_tol=1e-10):
                    errors.append(f"producer_metric_mismatch:{arm}:{field}")
            elif left != right:
                errors.append(f"producer_metric_mismatch:{arm}:{field}")
    return errors


def check_row(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    op: str,
) -> JsonDict:
    """Retain both operands so a failed precondition stays reproducible."""

    if op == "eq":
        passed = observed == expected
    elif op == "in":
        passed = observed in expected
    elif op == "starts_with":
        passed = str(observed).startswith(str(expected))
    else:
        raise ValueError(f"unknown_check_op:{op}")
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "required": True,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Resolve owned instructions and producer terminal fields before measurement."""

    checks: list[JsonDict] = []
    for label in ("AGENTS.md", "CODEX.md", "CLAUDE.md", SPEC_PATH.as_posix()):
        path = root / label
        checks.append(
            check_row("required_path", "worktree", label, "exists", True, path.is_file(), "eq")
        )
    spec = root / SPEC_PATH
    checks.append(
        check_row(
            "matching_requirement",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7579",
            True,
            spec.is_file() and "REQ-REPORT-7579" in spec.read_text(encoding="utf-8"),
            "eq",
        )
    )
    producers: dict[str, JsonDict] = {}
    ready_fields = {
        "exp7575": "online_protocol_ready_score",
        "exp7576": "proper_loss_fit_ready_score",
        "exp7577": "static_measurement_complete_score",
        "exp7578": "learning_measurement_complete_score",
    }
    for upstream, relative in PRODUCERS.items():
        path = root / relative
        exists = path.is_file()
        checks.append(
            check_row(
                "producer_exists", upstream, relative.as_posix(), "exists", True, exists, "eq"
            )
        )
        producer = load_json(path) if exists else {}
        producers[upstream] = producer
        if exists:
            checks.extend(
                [
                    check_row(
                        "producer_terminal",
                        upstream,
                        relative.as_posix(),
                        "honest_verdict",
                        "complete_",
                        producer.get("honest_verdict"),
                        "starts_with",
                    ),
                    check_row(
                        "producer_unflagged",
                        upstream,
                        relative.as_posix(),
                        "flagged_adversarial",
                        False,
                        producer.get("flagged_adversarial"),
                        "eq",
                    ),
                    check_row(
                        "producer_branch_ready",
                        upstream,
                        relative.as_posix(),
                        ready_fields[upstream],
                        1,
                        producer.get(ready_fields[upstream]),
                        "eq",
                    ),
                ]
            )
    return checks, producers


def _resolved(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def read_sidecar(
    root: Path, receipt: Mapping[str, Any], name: str, source_hashes: list[JsonDict]
) -> Path:
    """Authenticate a producer-declared sidecar before opening its evidence."""

    path = _resolved(root, str(receipt.get("path") or ""))
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"sidecar_hash_mismatch:{name}")
    if path.stat().st_size != int(receipt.get("bytes", -1)):
        raise ValueError(f"sidecar_size_mismatch:{name}")
    source_hashes.append(source_receipt(path, root, name))
    return path


def load_authenticated_inputs(
    root: Path, producers: Mapping[str, Mapping[str, Any]]
) -> tuple[JsonDict, list[JsonDict]]:
    """Load only raw operands whose exact bytes a producer already bound."""

    source_hashes = [
        source_receipt(root / PRODUCERS[name], root, name) for name in sorted(PRODUCERS)
    ]
    p75 = producers["exp7575"]["raw_sidecars"]
    p76 = producers["exp7576"]["raw_sidecars"]
    p77 = producers["exp7577"]["raw_sidecars"]
    p78 = producers["exp7578"]["raw_sidecars"]
    roles = load_jsonl(
        read_sidecar(root, p75["cached_roles"], "exp7575.cached_roles", source_hashes)
    )
    protocol = load_json(
        read_sidecar(root, p75["frozen_protocol"], "exp7575.frozen_protocol", source_hashes)
    )
    heads = load_json(
        read_sidecar(root, p76["head_manifest"], "exp7576.head_manifest", source_hashes)
    )
    published_static = load_jsonl(
        read_sidecar(root, p77["evaluation_rows"], "exp7577.evaluation_rows", source_hashes)
    )
    causal = load_jsonl(
        read_sidecar(root, p78["causal_rows"], "exp7578.causal_rows", source_hashes)
    )
    comparison = load_jsonl(
        read_sidecar(root, p78["comparison_rows"], "exp7578.comparison_rows", source_hashes)
    )
    release_ack = load_jsonl(
        read_sidecar(root, p78["release_ack_rows"], "exp7578.release_ack_rows", source_hashes)
    )
    retention = load_jsonl(
        read_sidecar(root, p78["retention_rows"], "exp7578.retention_rows", source_hashes)
    )
    bootstrap = load_jsonl(
        read_sidecar(root, p78["bootstrap_rows"], "exp7578.bootstrap_rows", source_hashes)
    )
    learned = load_json(
        read_sidecar(root, p78["learned_state"], "exp7578.learned_state", source_hashes)
    )
    return {
        "roles": roles,
        "protocol": protocol,
        "heads": heads,
        "published_static": published_static,
        "causal": causal,
        "comparison": comparison,
        "release_ack": release_ack,
        "retention": retention,
        "bootstrap": bootstrap,
        "learned": learned,
    }, source_hashes


def field_principles() -> dict[str, str]:
    """Explain the omission or claim drift prevented by each required field."""

    return {
        "honest_verdict": "A complete prefix reports terminal work; completion does not establish benefit.",
        "verdict_class": "One closed class distinguishes a valid null from blocked or disqualified evidence.",
        "flagged_adversarial": "The exact terminal verifier outcome cannot open readiness when flagged.",
        "gate_check_summary": "A block names the exact upstream operand instead of becoming ambiguous absence.",
        "acceptance_gate_results": "Separate validity, readiness, benefit, retention, and freshness keep a null usable.",
        "rows": "Each comparison unit and arm retains raw operands, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Aggregation stays distinct from planned live inference.",
        "MODEL_SPECS": "No current LLM task means the current model roster is empty.",
        "invocation_counts": "Loads, forwards, generations, and tokens remain independently zero.",
        "duration_s": "Monotonic current work excludes inherited timing and artificial sleeps.",
        "source_artifact_hashes": "Every conclusion binds to exact producer and sidecar bytes.",
        "validation_receipts": "Each bounded check binds its command, worktree, exit, and log hash.",
        "field_principles": "Each required field states the failure its presence prevents.",
        "verifier_is_oracle": "Label access makes controls circular, never oracle-distinct positive evidence.",
        "static_claims_qualified_score": "A raw reconstruction qualifies a valid static finding, including a null.",
        "learning_claims_qualified_score": "Only complete causal reconstruction qualifies learning claims.",
        "branch_conclusions": "Static, causal, retention, and freshness cannot overwrite each other.",
        "fresh_confirmatory_claim_allowed": "Prior exposure makes every V662 source-data conclusion descriptive.",
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, *, passed: bool | None = None
) -> JsonDict:
    result = observed == expected if passed is None else passed
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "eq",
        "passed": bool(result),
        "principle": "Validity, readiness, benefit, retention, and freshness remain separate.",
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [row for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": deepcopy(dict(failed[0])) if failed else None,
    }


def _blocked_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        raise ValueError("blocked_artifact_requires_failed_check")
    keys = ("check", "upstream", "path", "field", "op", "expected", "observed")
    return {
        "passed": False,
        "failed_count": sum(row.get("passed") is not True for row in checks),
        "failed_checks": [row.get("check") for row in checks if row.get("passed") is not True],
        "first_failure": {key: deepcopy(failed.get(key)) for key in keys},
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding clocks and this self-reference."""

    excluded = {"reproducibility_checksum", "duration_s", "phase_spans"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _base_artifact(duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate": "authenticated_cached_rows_independent_numerical_replay",
        "duration_s": float(duration_s),
        "verifier_is_oracle": False,
        "fresh_confirmatory_claim_allowed": False,
        "field_principles": field_principles(),
        "submitted_externally": False,
        "applicable_numbered_e2e": [],
    }


def build_blocked_artifact(
    root: Path, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Represent external absence once without inventing dependent rows."""

    artifact = {
        **_base_artifact(duration_s),
        "worktree_root": str(root.resolve()),
        "honest_verdict": "complete_blocked_missing_or_invalid_external_evidence",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "preconditions_checked": [dict(row) for row in checks],
        "gate_check_summary": _blocked_summary(checks),
        "acceptance_gate_results": [_gate("external_prerequisites", "validity", True, False)],
        "rows": [],
        "static_claims_qualified_score": 0,
        "learning_claims_qualified_score": 0,
        "branch_conclusions": {
            branch: {
                "readiness": "blocked",
                "benefit": "not_measured",
                "freshness": "not_fresh",
                "failure_source": "missing_work",
            }
            for branch in ("static", "causal", "retention", "freshness")
        },
        "source_artifact_hashes": [],
        "validation_receipts": [],
        "mutation_rows": [],
        "prior_verdict_disposition": "external_absence_is_blocked_not_retryable_partial",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    expected = {*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        by_name[str(row.get("name"))].append(row)
    return all(
        len(by_name[name]) == 1
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is not True
        and by_name[name][0].get("passed") is True
        for name in expected
    )


def _published_rows(
    static_rows: Sequence[Mapping[str, Any]], bootstrap_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    output = [{"row_kind": "static_source_arm", **dict(row)} for row in static_rows]
    for row in bootstrap_rows:
        copied = dict(row)
        censoring = copied.get("censoring")
        copied.update(
            {
                "row_kind": "learning_bootstrap_arm",
                "raw_numerator": copied.get("raw_squared_error_numerator"),
                "raw_denominator": copied.get("raw_squared_error_denominator"),
                "metric_direction": copied.get(
                    "metric_direction", "lower_brier_and_cost_are_better"
                ),
                "censored": bool(
                    isinstance(censoring, Mapping)
                    and any(bool(value) for value in censoring.values())
                ),
            }
        )
        output.append(copied)
    return output


def build_artifact(
    *,
    root: Path,
    run_date: str,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    static_rows: Sequence[Mapping[str, Any]],
    static_reduction: Mapping[str, Any],
    static_errors: Sequence[str],
    causal_reduction: Mapping[str, Any],
    online_reduction: Mapping[str, Any],
    retention_reduction: Mapping[str, Any],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    bootstrap_reduction: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble independent static and learning conclusions without conflation."""

    preconditions_pass = all(row.get("passed") is True for row in preconditions)
    static_qualified = preconditions_pass and not static_errors and len(static_rows) == 400
    learning_qualified = bool(
        preconditions_pass
        and causal_reduction.get("qualified") is True
        and bootstrap_reduction.get("measurement_complete") is True
    )
    static_benefit = bool(
        static_qualified
        and all(
            float(row.get("lower95")) > 0.0
            for row in static_reduction.get("contrasts", {}).values()
        )
    )
    learning_benefit = bool(learning_qualified and bootstrap_reduction.get("benefit_passed"))
    retention_pass = bool(learning_qualified and bootstrap_reduction.get("retention_passed"))
    mutation_pass = len(mutations) == 6 and all(row.get("passed") is True for row in mutations)
    validation_pass = _receipts_pass(validation_receipts)
    gates = [
        _gate("preconditions", "validity", True, preconditions_pass),
        _gate("static_raw_reconstruction", "validity", True, static_qualified),
        _gate("causal_raw_reconstruction", "validity", True, learning_qualified),
        _gate("private_mutations", "validity", True, mutation_pass),
        _gate("required_validation", "validity", True, validation_pass),
        _gate("static_branch_ready", "readiness", True, static_qualified),
        _gate("learning_branch_ready", "readiness", True, learning_qualified),
        _gate("static_probability_benefit", "benefit", True, static_benefit),
        _gate("learning_feedback_benefit", "benefit", True, learning_benefit),
        _gate("retention_non_regression", "retention", True, retention_pass),
        _gate("fresh_confirmatory_claim", "freshness", False, False),
    ]
    validity = all(row["passed"] for row in gates if row["category"] == "validity")
    verdict_class = "null" if validity else "disqualified"
    honest_verdict = (
        "complete_null_independent_static_and_learning_audit"
        if validity
        else "complete_disqualified_independent_audit_validation_failure"
    )
    artifact = {
        **_base_artifact(duration_s),
        "worktree_root": str(root.resolve()),
        "run_date": run_date,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": any(
            row.get("name") == "adversarial_verify" and row.get("exit_code") != 0
            for row in validation_receipts
        ),
        "preconditions_checked": [dict(row) for row in preconditions],
        "source_artifact_hashes": [dict(row) for row in source_hashes],
        "validation_receipts": [dict(row) for row in validation_receipts],
        "rows": _published_rows(static_rows, bootstrap_rows),
        "static_reconstruction": {
            "errors": list(static_errors),
            "reduction": deepcopy(dict(static_reduction)),
            "descriptive_exposed_data_only": True,
        },
        "causal_reconstruction": deepcopy(dict(causal_reduction)),
        "online_reduction": deepcopy(dict(online_reduction)),
        "retention_reduction": deepcopy(dict(retention_reduction)),
        "learning_interval_reduction": deepcopy(dict(bootstrap_reduction)),
        "mutation_rows": [dict(row) for row in mutations],
        "static_claims_qualified_score": int(static_qualified),
        "learning_claims_qualified_score": int(learning_qualified),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "branch_conclusions": {
            "static": {
                "readiness": static_qualified,
                "benefit": static_benefit,
                "freshness": False,
                "failure_source": "harmful_recalibration_and_uncertainty"
                if not static_benefit
                else None,
            },
            "causal": {
                "readiness": learning_qualified,
                "benefit": learning_benefit,
                "freshness": False,
                "failure_source": "no_signal_against_counts_and_uncertainty"
                if not learning_benefit
                else None,
            },
            "retention": {
                "readiness": learning_qualified,
                "benefit": retention_pass,
                "freshness": False,
                "failure_source": "harmful_recalibration" if not retention_pass else None,
            },
            "freshness": {
                "readiness": True,
                "benefit": False,
                "freshness": False,
                "failure_source": "source_exposure",
            },
        },
        "oracle_controls": {
            "verifier_is_oracle": False,
            "label_accessing_controls": "circular_positive_only_if_own_contrasts_support_claim",
            "oracle_distinct_positive_allowed": False,
        },
        "absence_is_measured_zero": False,
        "prior_verdict_disposition": "v661_external_absence_not_repeated_narrow_verdict_not_retired",
        "phase_spans": [dict(row) for row in phase_spans],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(root: Path) -> JsonDict:
    """Build compact valid bytes for schema and fresh-process reader tests."""

    receipts = [
        {
            "name": name,
            "command": f"test {name}",
            "command_argv": ["test", name],
            "cwd": str(root.resolve()),
            "exit_code": 0,
            "timed_out": False,
            "passed": True,
            "log_sha256": "sha256:" + hashlib.sha256(name.encode()).hexdigest(),
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    gates = [
        _gate("fixture_validity", "validity", True, True),
        _gate("fixture_benefit", "benefit", True, False),
        _gate("fresh_confirmatory_claim", "freshness", False, False),
    ]
    artifact = {
        **_base_artifact(0.1),
        "worktree_root": str(root.resolve()),
        "run_date": "20260924",
        "honest_verdict": "complete_null_fixture_audit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "preconditions_checked": [],
        "source_artifact_hashes": [],
        "validation_receipts": receipts,
        "rows": [],
        "static_reconstruction": {"errors": [], "reduction": {}},
        "causal_reconstruction": {"qualified": True, "errors": []},
        "online_reduction": {},
        "retention_reduction": {},
        "learning_interval_reduction": {},
        "mutation_rows": run_private_mutations(),
        "static_claims_qualified_score": 1,
        "learning_claims_qualified_score": 1,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "branch_conclusions": {
            branch: {"readiness": True, "benefit": False, "freshness": False}
            for branch in ("static", "causal", "retention", "freshness")
        },
        "oracle_controls": {"oracle_distinct_positive_allowed": False},
        "absence_is_measured_zero": False,
        "prior_verdict_disposition": "fixture",
        "phase_spans": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _verify_sources(value: Mapping[str, Any], root: Path) -> None:
    for receipt in value.get("source_artifact_hashes") or []:
        authenticate_source_receipt(receipt, root)


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Reject identity, custody, claim, row, receipt, or checksum drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE:
        errors.append("milestone_mismatch")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_mismatch")
    if value.get("fresh_confirmatory_claim_allowed") is not False:
        errors.append("freshness_claim_not_closed")
    if set(REQUIRED_PRINCIPLE_FIELDS) - set(value.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if set(value.get("branch_conclusions") or {}) != {
        "static",
        "causal",
        "retention",
        "freshness",
    }:
        errors.append("branch_conclusions_incomplete")
    if value.get("static_claims_qualified_score") not in (0, 1):
        errors.append("static_qualification_invalid")
    if value.get("learning_claims_qualified_score") not in (0, 1):
        errors.append("learning_qualification_invalid")
    if value.get("verdict_class") == "blocked":
        first = (value.get("gate_check_summary") or {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not isinstance(first, Mapping) or set(first) != required:
            errors.append("blocked_gate_summary_invalid")
    for row in value.get("rows") or []:
        raw_ok = ("raw_numerator" in row and "raw_denominator" in row) or (
            "raw_brier_numerator" in row and "raw_brier_denominator" in row
        )
        if not raw_ok or any(
            field not in row
            for field in ("arm", "metric_direction", "seed", "censored", "provenance")
        ):
            errors.append("comparison_row_schema_invalid")
            break
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts_failed")
    mutations = value.get("mutation_rows") or []
    if mutations and (
        {row.get("mutation") for row in mutations}
        != {"future_label", "order", "missing_row", "sign", "hash", "duplicate_update"}
        or not all(row.get("passed") is True for row in mutations)
    ):
        errors.append("mutation_panel_failed")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("checksum_mismatch")
    try:
        _verify_sources(value, root)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError(";".join(dict.fromkeys(errors)))
    return {"valid": True, "errors": []}


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    value = load_json(path)
    if not value:
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root)


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze serial tests, changed-module coverage, lint, type, and spec checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    (private_root / "focused").parent.mkdir(parents=True, exist_ok=True)
    (private_root / "coverage").parent.mkdir(parents=True, exist_ok=True)
    coverage_file = private_root / ".coverage.exp7579"
    return validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(),),
        (MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=private_root,
        coverage_file=coverage_file,
    )


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    common = ("--root", str(root.resolve()), "--date", "20260924")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
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
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _provisional_terminal_receipts() -> list[JsonDict]:
    return [
        {
            "name": name,
            "command": f"pending exact candidate {name}",
            "command_argv": ["pending", name],
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "pending",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Re-reduce every embedded comparative row in a fresh process."""

    value = load_json(path)
    validate_artifact(value, root=root)
    static = [
        {key: deepcopy(item) for key, item in row.items() if key != "row_kind"}
        for row in value.get("rows") or []
        if row.get("row_kind") == "static_source_arm"
    ]
    learning = [
        {key: deepcopy(item) for key, item in row.items() if key != "row_kind"}
        for row in value.get("rows") or []
        if row.get("row_kind") == "learning_bootstrap_arm"
    ]
    if static:
        expected_static = value["static_reconstruction"]["reduction"]
        observed_static = reduce_static_rows(
            static, draws=int(expected_static["draws"]), seed=int(expected_static["seed"])
        )
        if canonical_hash(observed_static) != canonical_hash(expected_static):
            raise ValueError("independent_static_reduction_mismatch")
    if learning:
        observed_learning = reduce_bootstrap_rows(learning)
        if canonical_hash(observed_learning) != canonical_hash(
            value["learning_interval_reduction"]
        ):
            raise ValueError("independent_learning_reduction_mismatch")
    return {"valid": True, "static_rows": len(static), "learning_rows": len(learning)}


def _write_manifest(root: Path) -> tuple[Path, JsonDict]:
    path = root / RAW_DIR / "affected_validation_manifest.json"
    value = {
        "experiment_id": EXPERIMENT_ID,
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
    }
    atomic_json(path, value)
    return path, value


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each boundary so bounded work never appears stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7579] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260924")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    arguments = parser.parse_args(argv)
    arguments.root = arguments.root.resolve()
    if arguments.date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    return arguments


def run_experiment(  # pragma: no cover - exercised by the declared capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Authenticate, reconstruct, validate, and atomically publish the audit."""

    root = root.resolve()
    if run_date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    started = time.monotonic()
    spans: list[JsonDict] = []
    destination = root / RESULT_PATH
    raw_root = root / RAW_DIR
    candidate = raw_root / "terminal_candidate.json"

    progress(started, "preconditions", "start", root=root)
    phase_started = time.monotonic()
    checks, producers = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = [row for row in checks if row.get("passed") is not True]
    progress(started, "preconditions", "complete", completed_units=len(checks), failed=len(failed))
    if failed:
        blocked = build_blocked_artifact(root, checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked")
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked", bytes=destination.stat().st_size)
        return blocked

    progress(started, "manifest", "start")
    phase_started = time.monotonic()
    manifest_path, _manifest = _write_manifest(root)
    spans.append(_span("manifest", phase_started, started, 1))
    progress(started, "manifest", "complete", completed_units=1)

    progress(started, "evidence_load", "before_sidecar_reads")
    phase_started = time.monotonic()
    inputs, source_hashes = load_authenticated_inputs(root, producers)
    for relative, upstream in (
        (MODULE_PATH, "exp7579.implementation"),
        (WRAPPER_PATH, "exp7579.entrypoint"),
        (TEST_PATH, "exp7579.tests"),
        (SPEC_PATH, "exp7579.spec"),
    ):
        source_hashes.append(source_receipt(root / relative, root, upstream))
    source_hashes.append(source_receipt(manifest_path, root, "exp7579.validation_manifest"))
    spans.append(_span("evidence_load", phase_started, started, len(source_hashes)))
    progress(started, "evidence_load", "after_sidecar_reads", completed_units=len(source_hashes))

    progress(started, "static_reconstruction", "before_benchmark", planned_units=80)
    phase_started = time.monotonic()
    static_features = [row for row in inputs["roles"] if row.get("role") == "test"]
    static_rows = reconstruct_static_rows(static_features, inputs["heads"], seed=7575101)
    static_errors = compare_static_rows(static_rows, inputs["published_static"])
    static_reduction = reduce_static_rows(static_rows, draws=1000, seed=7575101)
    static_errors.extend(
        _selected_metric_errors(
            static_reduction["metrics"],
            producers["exp7577"]["probability_metrics"],
            ("raw_brier_numerator", "raw_brier_denominator", "mean_brier"),
        )
    )
    spans.append(_span("static_reconstruction", phase_started, started, len(static_features)))
    progress(
        started,
        "static_reconstruction",
        "after_benchmark",
        completed_units=len(static_features),
        errors=len(static_errors),
    )

    progress(started, "causal_reconstruction", "before_benchmark", planned_units=800)
    phase_started = time.monotonic()
    causal = audit_causal_rows(
        inputs["causal"],
        inputs["release_ack"],
        inputs["learned"],
        inputs["protocol"]["orders"],
        block_size=int(inputs["protocol"]["release_block_size"]),
    )
    online = reduce_learning_metric_rows(inputs["comparison"])
    retention = reduce_learning_metric_rows(inputs["retention"])
    causal["errors"].extend(
        _selected_metric_errors(
            online,
            producers["exp7578"]["measurement_metrics"]["prequential"],
            (
                "raw_squared_error_numerator",
                "raw_squared_error_denominator",
                "mean_brier",
                "mean_action_cost",
                "coverage",
            ),
        )
    )
    causal["errors"].extend(
        _selected_metric_errors(
            retention,
            producers["exp7578"]["measurement_metrics"]["retention"],
            (
                "raw_squared_error_numerator",
                "raw_squared_error_denominator",
                "mean_brier",
                "mean_action_cost",
                "coverage",
            ),
        )
    )
    causal["errors"] = sorted(set(causal["errors"]))
    causal["qualified"] = not causal["errors"]
    bootstrap = reduce_bootstrap_rows(inputs["bootstrap"])
    spans.append(_span("causal_reconstruction", phase_started, started, len(inputs["causal"])))
    progress(
        started,
        "causal_reconstruction",
        "after_benchmark",
        completed_units=len(inputs["causal"]),
        errors=len(causal["errors"]),
    )

    progress(started, "private_mutations", "start", planned_units=6)
    phase_started = time.monotonic()
    mutations = run_private_mutations()
    spans.append(_span("private_mutations", phase_started, started, len(mutations)))
    progress(
        started,
        "private_mutations",
        "complete",
        completed_units=len(mutations),
        passed=sum(row["passed"] is True for row in mutations),
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7579-", dir="/tmp"))
    commands = build_validation_commands(root, private_root)
    progress(started, "scoped_validation", "before_subprocesses", planned_units=len(commands))
    phase_started = time.monotonic()
    affected_receipts = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation_logs",
        heartbeat_s=60.0,
    )
    for receipt in affected_receipts:
        receipt["worktree"] = str(root)
    spans.append(_span("scoped_validation", phase_started, started, len(affected_receipts)))
    affected_pass = validation_scope.reduce_required_checks(affected_receipts)[
        "required_checks_passed"
    ]
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected_pass,
    )
    if not affected_pass:
        raise RuntimeError("required_scoped_validation_failed")

    provisional = build_artifact(
        root=root,
        run_date=run_date,
        preconditions=checks,
        source_hashes=source_hashes,
        static_rows=static_rows,
        static_reduction=static_reduction,
        static_errors=static_errors,
        causal_reduction=causal,
        online_reduction=online,
        retention_reduction=retention,
        bootstrap_rows=inputs["bootstrap"],
        bootstrap_reduction=bootstrap,
        mutations=mutations,
        validation_receipts=[*affected_receipts, *_provisional_terminal_receipts()],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    atomic_json(candidate, provisional)

    first_plan = terminal_commands(candidate, root)
    progress(started, "terminal_validation", "before_subprocesses", planned_units=len(first_plan))
    phase_started = time.monotonic()
    terminal_receipts = validation_scope.run_commands(
        root,
        first_plan,
        log_dir=raw_root / "terminal_logs_provisional",
        heartbeat_s=60.0,
    )
    for receipt in terminal_receipts:
        receipt["worktree"] = str(root)
    spans.append(_span("terminal_validation", phase_started, started, len(terminal_receipts)))
    terminal_pass = all(row.get("passed") is True for row in terminal_receipts)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=terminal_pass,
    )
    if not terminal_pass:
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_artifact(
        root=root,
        run_date=run_date,
        preconditions=checks,
        source_hashes=source_hashes,
        static_rows=static_rows,
        static_reduction=static_reduction,
        static_errors=static_errors,
        causal_reduction=causal,
        online_reduction=online,
        retention_reduction=retention,
        bootstrap_rows=inputs["bootstrap"],
        bootstrap_reduction=bootstrap,
        mutations=mutations,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    validate_artifact(final, root=root)
    atomic_json(candidate, final)

    exact_plan = terminal_commands(candidate, root)
    progress(started, "exact_candidate_validation", "before_subprocesses", planned_units=4)
    exact_receipts = validation_scope.run_commands(
        root,
        exact_plan,
        log_dir=raw_root / "terminal_logs_exact",
        heartbeat_s=60.0,
    )
    exact_pass = all(row.get("passed") is True for row in exact_receipts)
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed_units=len(exact_receipts),
        passed=exact_pass,
    )
    if not exact_pass:
        raise RuntimeError("exact_terminal_candidate_validation_failed")

    progress(started, "publish", "before_atomic_terminal")
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(candidate):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    progress(
        started,
        "publish",
        "complete_terminal",
        bytes=destination.stat().st_size,
        verdict=final["verdict_class"],
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    arguments = parse_args(argv)
    if arguments.cold_replay is not None:
        result = cold_replay(arguments.cold_replay, root=arguments.root)
        print(json.dumps({"mode": "cold_replay", **result}, sort_keys=True), flush=True)
        return 0
    if arguments.independent_reduce is not None:
        result = independent_replay(arguments.independent_reduce, root=arguments.root)
        print(json.dumps({"mode": "independent_reduction", **result}, sort_keys=True), flush=True)
        return 0
    run_experiment(arguments.root, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
