"""Independently audit V664 evidence decisions and guarded learning.

The audit runs even when upstream work stopped. It preserves valid pilot rows,
but missing static or learning producers remain an external block.

Spec refs: REQ-REPORT-7610 and SCENARIO-REPORT-7610-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7596_v663_evidence_audit import (
    ZERO_INVOCATION_COUNTS,
    authenticate_source_receipt,
    blocked_summary,
    canonical_hash,
    check_row,
    load_json,
    sha256_file,
    source_receipt,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7610-v664-evidence-audit"
MILESTONE = "2026.09.664"
SCHEMA = "carnot.exp7610.v664.evidence_audit.v1"
RESULT_PATH = Path("results/experiment_7610_v664_evidence_audit.json")
RAW_DIR = Path("results/raw/experiment_7610_v664_evidence_audit")
MODULE_PATH = Path("python/carnot/experiment_7610_v664_evidence_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7610_v664_evidence_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7610_v664_evidence_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v664-evidence-audit.md")
GAPS_PATH = Path("ops/verifier_gaps.md")
MODEL_SPECS: list[JsonDict] = []
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
GATE_OPERAND_FIELDS = {"check", "upstream", "path", "field", "op", "expected", "observed"}
EVIDENCE_FEATURE_NAMES = (
    "supported_sentence_fraction",
    "contradicted_fraction",
    "unknown_fraction",
    "valid_link_fraction",
    "exact_named_entity_overlap",
    "numeric_value_agreement",
    "negation_mismatch_indicator",
    "extraction_censor_indicator",
)
EVIDENCE_ARMS = ("factual", "erased", "deranged")
ONLINE_ARMS = ("frozen", "unguarded_factual", "guarded_factual", "guarded_deranged")


@dataclass(frozen=True)
class SourceSpec:
    """Name one planned producer and its optional conductor diagnostic."""

    upstream: str
    producer_path: Path
    conductor_path: Path | None


SOURCE_SPECS = (
    SourceSpec(
        "exp7602-evidence-requalification",
        Path("results/experiment_7602_v664_evidence_requalification.json"),
        None,
    ),
    SourceSpec(
        "exp7603-guarded-update-fixture",
        Path("results/experiment_7603_v664_guarded_update_fixture.json"),
        None,
    ),
    SourceSpec(
        "exp7604-evidence-pilot", Path("results/experiment_7604_v664_evidence_pilot.json"), None
    ),
    SourceSpec(
        "exp7605-fit-evidence",
        Path("results/experiment_7605_v664_fit_evidence.json"),
        Path("results/experiment_7605_fit_evidence.json"),
    ),
    SourceSpec(
        "exp7606-test-online-evidence",
        Path("results/experiment_7606_v664_test_online_evidence.json"),
        Path("results/experiment_7606_test_online_evidence.json"),
    ),
    SourceSpec(
        "exp7607-evidence-energy",
        Path("results/experiment_7607_v664_evidence_energy.json"),
        Path("results/experiment_7607_evidence_energy.json"),
    ),
    SourceSpec(
        "exp7608-decision-evaluation",
        Path("results/experiment_7608_v664_decision_evaluation.json"),
        Path("results/experiment_7608_decision_evaluation.json"),
    ),
    SourceSpec(
        "exp7609-guarded-learning",
        Path("results/experiment_7609_v664_guarded_learning.json"),
        Path("results/experiment_7609_guarded_learning.json"),
    ),
)


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_receipt(path: Path, root: Path) -> JsonDict:
    """Bind a raw sidecar to exact bytes with a stable path label."""

    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def classify_source(root: Path, spec: SourceSpec) -> JsonDict:
    """Classify producer, pre-gate, invalid, flagged, and absent evidence."""

    producer_path = root / spec.producer_path
    conductor_path = root / spec.conductor_path if spec.conductor_path else None
    if producer_path.is_file():
        producer = load_json(producer_path)
        receipt = source_receipt(producer_path, root, spec.upstream)
        terminal = str(producer.get("honest_verdict") or "").startswith("complete_")
        flagged = producer.get("flagged_adversarial") is True
        disposition = "authenticated_producer"
        if not producer or not terminal:
            disposition = "invalid_producer"
        elif flagged:
            disposition = "flagged_producer"
        receipt.update(
            producer_path=spec.producer_path.as_posix(),
            conductor_path=spec.conductor_path.as_posix() if spec.conductor_path else None,
            honest_verdict=producer.get("honest_verdict"),
            verdict_class=producer.get("verdict_class"),
            flagged_adversarial=producer.get("flagged_adversarial"),
            disposition=disposition,
            eligible_for_science=disposition == "authenticated_producer",
        )
        return receipt
    if conductor_path is not None and conductor_path.is_file():
        conductor = load_json(conductor_path)
        receipt = source_receipt(conductor_path, root, spec.upstream)
        receipt.update(
            producer_path=spec.producer_path.as_posix(),
            conductor_path=spec.conductor_path.as_posix(),
            honest_verdict=conductor.get("honest_verdict"),
            verdict_class="blocked",
            flagged_adversarial=False,
            disposition="conductor_pre_gate",
            eligible_for_science=False,
        )
        return receipt
    return {
        "upstream": spec.upstream,
        "path": spec.producer_path.as_posix(),
        "producer_path": spec.producer_path.as_posix(),
        "conductor_path": spec.conductor_path.as_posix() if spec.conductor_path else None,
        "sha256": None,
        "bytes": None,
        "honest_verdict": None,
        "verdict_class": None,
        "flagged_adversarial": None,
        "disposition": "missing_producer",
        "eligible_for_science": False,
    }


def text_sha256(value: str) -> str:
    """Hash exact UTF-8 text because sentence pointers address those bytes."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read an object-only JSONL sidecar without accepting malformed rows."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _resolve_receipt(root: Path, receipt: Mapping[str, Any]) -> Path:
    path = Path(str(receipt.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise ValueError(f"sidecar_missing:{path}")
    if resolved.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"sidecar_size_mismatch:{path}")
    if sha256_file(resolved) != receipt.get("sha256"):
        raise ValueError(f"sidecar_hash_mismatch:{path}")
    return resolved


def _roundtrip_segments(text: str, segments: Sequence[Mapping[str, Any]]) -> None:
    original = text.encode("utf-8")
    cursor = 0
    rebuilt = bytearray()
    for row in segments:
        start = row.get("byte_start")
        end = row.get("byte_end")
        value = row.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(value, str):
            raise ValueError("segment_roundtrip_invalid")
        encoded = value.encode("utf-8")
        if (
            start != cursor
            or end != start + len(encoded)
            or original[start:end] != encoded
            or row.get("text_sha256") != text_sha256(value)
        ):
            raise ValueError("segment_roundtrip_invalid")
        rebuilt.extend(encoded)
        cursor = end
    if cursor != len(original) or bytes(rebuilt) != original:
        raise ValueError("segment_roundtrip_invalid")


def validate_model_row(record: Mapping[str, Any]) -> bool:
    """Rebuild one predictor input while denying outcome-bearing fields."""

    if {"label", "human_label", "probability", "raw_probability"} & set(record):
        raise ValueError("predictor_label_access")
    if record.get("labels_accessible") is not False:
        raise ValueError("predictor_label_access")
    if record.get("raw_probability_accessible") is not False:
        raise ValueError("predictor_probability_access")
    if tuple(record.get("evidence_feature_names") or ()) != EVIDENCE_FEATURE_NAMES:
        raise ValueError("evidence_feature_roster_invalid")
    for name in ("source", "question", "answer"):
        text = record.get(f"complete_{name}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"complete_{name}_absent")
        if record.get(f"{name}_sha256") != text_sha256(text):
            raise ValueError(f"{name}_hash_invalid")
        _roundtrip_segments(text, record.get(f"{name}_sentences") or [])
    if not str(record.get("component_hash") or ""):
        raise ValueError("component_hash_absent")
    return True


def _audit_file_pointer(root: Path, path_value: object, expected_hash: object) -> JsonDict:
    path = Path(str(path_value or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != expected_hash:
        raise ValueError(f"raw_pointer_hash_mismatch:{path}")
    return file_receipt(resolved, root)


def audit_pilot_rows(
    root: Path, protocol: Mapping[str, Any], pilot: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Rebuild available pilot identity and labels from authenticated sidecars."""

    raw = protocol.get("raw_sidecars") or {}
    model_receipt = (raw.get("model_inputs") or {}).get("pilot") or {}
    label_receipt = (raw.get("evaluator_stores") or {}).get("pilot") or {}
    model_path = _resolve_receipt(root, model_receipt)
    label_path = _resolve_receipt(root, label_receipt)
    models = load_jsonl(model_path)
    labels = load_jsonl(label_path)
    by_model = {str(row.get("component_hash")): row for row in models}
    by_label = {str(row.get("component_hash")): row for row in labels}
    if len(by_model) != len(models) or set(by_model) != set(by_label):
        raise ValueError("label_join_identity_mismatch")
    output: list[JsonDict] = []
    pointer_receipts: list[JsonDict] = []
    for published in pilot.get("pilot_rows") or []:
        component = str(published.get("component_hash") or "")
        model = by_model.get(component)
        label = by_label.get(component)
        if not isinstance(model, Mapping) or not isinstance(label, Mapping):
            raise ValueError("pilot_component_missing")
        validate_model_row(model)
        if published.get("input_record") != model:
            raise ValueError("pilot_input_record_mismatch")
        if (
            label.get("role") != "pilot"
            or label.get("evaluator_only") is not True
            or label.get("training_allowed") is not False
            or label.get("label") not in (0, 1)
        ):
            raise ValueError("pilot_label_role_invalid")
        pointer_receipts.extend(
            [
                _audit_file_pointer(
                    root, published.get("request_path"), published.get("request_sha256")
                ),
                _audit_file_pointer(
                    root, published.get("response_path"), published.get("response_sha256")
                ),
            ]
        )
        censored = published.get("censoring") != "none"
        output.append(
            {
                "unit_id": component,
                "arm": "pilot_evidence_transport",
                "absolute_metrics": {
                    "raw_probability": float(label["raw_probability"]),
                    "label": int(label["label"]),
                    "usable_schema": int(published.get("usable_schema") is True),
                    "extraction_censor_indicator": float(censored),
                },
                "raw_numerator": int(published.get("usable_schema") is True),
                "raw_denominator": 1,
                "seed": int(published.get("seed", 7604001)),
                "direction": "higher_usable_schema_rate_is_better",
                "censored": censored,
                "censoring": str(published.get("censoring") or "unknown"),
                "provenance": "independent_exp7602_sidecar_exp7604_pointer_join",
            }
        )
    if len(output) != len(models):
        raise ValueError("pilot_row_count_mismatch")
    receipts = [dict(model_receipt), dict(label_receipt), *pointer_receipts]
    return output, receipts


def _probability_from_logit(logit: float) -> float:
    if not math.isfinite(logit):
        raise ValueError("static_logit_invalid")
    if logit >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exponential = math.exp(logit)
        probability = exponential / (1.0 + exponential)
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    energies = (-math.log1p(-clipped), -math.log(clipped))
    weights = [math.exp(-energy + min(energies)) for energy in energies]
    return weights[1] / sum(weights)


def _typed_action(probability: float, label: int) -> tuple[str, float, bool, bool]:
    expected = {"accept": 5.0 * probability, "reject": 1.0 - probability, "escalate": 0.2}
    minimum = min(expected.values())
    action = next(
        name
        for name in ("escalate", "accept", "reject")
        if math.isclose(expected[name], minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    realized = {
        "accept": 5.0 if label else 0.0,
        "reject": 0.0 if label else 1.0,
        "escalate": 0.2,
    }[action]
    return action, realized, action != "escalate", action == "accept" and label == 1


def reconstruct_static_rows(
    records: Sequence[Mapping[str, Any]], head: Mapping[str, Any], *, seed: int
) -> list[JsonDict]:
    """Recompute frozen predictions and outcomes from eight raw features."""

    weights = [float(value) for value in head.get("weights") or []]
    if len(weights) != len(EVIDENCE_FEATURE_NAMES) or any(
        not math.isfinite(value) for value in weights
    ):
        raise ValueError("frozen_head_weights_invalid")
    bias = float(head.get("bias"))
    raw_offset = float(head.get("raw_offset"))
    grouped: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
    output: list[JsonDict] = []
    for record in records:
        unit = str(record.get("unit_id") or "")
        arm = str(record.get("arm") or "")
        features = tuple(float(value) for value in record.get("features") or [])
        label = int(record.get("label", -1))
        raw_probability = float(record.get("raw_probability"))
        if (
            not unit
            or arm not in EVIDENCE_ARMS
            or len(features) != len(EVIDENCE_FEATURE_NAMES)
            or label not in (0, 1)
            or not 0.0 <= raw_probability <= 1.0
        ):
            raise ValueError("static_input_invalid")
        if record.get("included") is not True:
            raise ValueError("failed_row_excluded")
        if arm in grouped[unit]:
            raise ValueError("static_duplicate_arm")
        grouped[unit][arm] = features
        raw_clipped = min(1.0 - 1e-12, max(1e-12, raw_probability))
        raw_logit = math.log(raw_clipped / (1.0 - raw_clipped))
        logit = (
            bias
            + raw_offset * raw_logit
            + sum(weight * feature for weight, feature in zip(weights, features))
        )
        probability = _probability_from_logit(logit)
        brier = (probability - label) ** 2
        action, cost, covered, false_accept = _typed_action(probability, label)
        output.append(
            {
                "unit_id": unit,
                "arm": arm,
                "probability": probability,
                "label": label,
                "brier": brier,
                "typed_action": action,
                "realized_action_cost": cost,
                "non_escalated": covered,
                "false_accept": false_accept,
                "raw_brier_numerator": brier,
                "raw_brier_denominator": 1,
                "raw_cost_numerator": cost,
                "raw_cost_denominator": 1,
                "seed": seed,
                "direction": "lower_brier_and_cost_are_better",
                "censored": bool(record.get("censored")),
                "provenance": "independent_raw_feature_frozen_head_reduction",
            }
        )
    expected = set(EVIDENCE_ARMS)
    if not grouped or any(set(arms) != expected for arms in grouped.values()):
        raise ValueError("static_arm_roster_invalid")
    for arms in grouped.values():
        if arms["factual"] == arms["erased"] or arms["factual"] == arms["deranged"]:
            raise ValueError("controls_identical")
    return output


def _interval(values: Sequence[float], *, draws: int, seed: int) -> JsonDict:
    if not values or draws <= 0:
        raise ValueError("interval_inputs_invalid")
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws)]
    ordered = sorted(means)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {
        "mean": sum(values) / len(values),
        "lower95": lower,
        "upper95": upper,
        "raw_numerator": sum(values),
        "raw_denominator": len(values),
        "draws": draws,
        "seed": seed,
    }


def reduce_static_rows(rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int) -> JsonDict:
    """Reduce each source once so arms never multiply independent units."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        unit = str(row.get("unit_id") or "")
        arm = str(row.get("arm") or "")
        if arm in grouped[unit]:
            raise ValueError("static_duplicate_arm")
        if row.get("raw_brier_denominator") != 1 or row.get("raw_cost_denominator") != 1:
            raise ValueError("static_denominator_invalid")
        expected_brier = (float(row["probability"]) - int(row["label"])) ** 2
        if not math.isclose(expected_brier, float(row["brier"]), abs_tol=1e-12):
            raise ValueError("brier_sign_or_value_mismatch")
        grouped[unit][arm] = row
    if not grouped or any(set(arms) != set(EVIDENCE_ARMS) for arms in grouped.values()):
        raise ValueError("static_arm_roster_invalid")
    metrics: dict[str, JsonDict] = {}
    for arm in EVIDENCE_ARMS:
        selected = [arms[arm] for arms in grouped.values()]
        brier = sum(float(row["raw_brier_numerator"]) for row in selected)
        cost = sum(float(row["raw_cost_numerator"]) for row in selected)
        metrics[arm] = {
            "brier_numerator": brier,
            "brier_denominator": len(selected),
            "mean_brier": brier / len(selected),
            "cost_numerator": cost,
            "cost_denominator": len(selected),
            "mean_cost": cost / len(selected),
            "coverage": sum(bool(row["non_escalated"]) for row in selected) / len(selected),
        }
    contrasts = {}
    for offset, comparator in enumerate(("erased", "deranged")):
        values = [
            float(arms[comparator]["brier"]) - float(arms["factual"]["brier"])
            for arms in grouped.values()
        ]
        contrasts[f"factual_vs_{comparator}"] = _interval(values, draws=draws, seed=seed + offset)
    return {
        "independent_unit_count": len(grouped),
        "metrics": metrics,
        "contrasts": contrasts,
        "controls_distinct": True,
    }


def _weights_hash(weights: Sequence[float]) -> str:
    return canonical_hash([round(float(value), 15) for value in weights])


def audit_online_rows(
    rows: Sequence[Mapping[str, Any]], *, initial_weights: Sequence[float], learning_rate: float
) -> JsonDict:
    """Rebuild allowed updates and reject chronology, role, or state drift."""

    errors: list[str] = []
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row.get("order_seed", -1)), str(row.get("arm") or ""))].append(row)
    orders = sorted({key[0] for key in grouped})
    arms = sorted({key[1] for key in grouped})
    if len(orders) != 5 or set(arms) != set(ONLINE_ARMS):
        errors.append("online_order_or_arm_roster_invalid")
    update_ids: set[str] = set()
    anchor_hashes: set[str] = set()
    final_hashes: dict[str, str] = {}
    restart_hashes_match = True
    evaluator_denied = True
    for key, arm_rows in sorted(grouped.items()):
        arm = key[1]
        weights = [float(value) for value in initial_weights]
        for row in sorted(arm_rows, key=lambda value: int(value.get("prediction_index", -1))):
            features = [float(value) for value in row.get("features") or []]
            if len(features) != len(weights):
                errors.append("online_feature_width_invalid")
                continue
            predicted = _probability_from_logit(sum(a * b for a, b in zip(weights, features)))
            if not math.isclose(predicted, float(row.get("prediction_probability")), abs_tol=1e-12):
                errors.append("prediction_probability_mismatch")
            prediction_index = int(row.get("prediction_index", -1))
            release_index = int(row.get("release_index", -1))
            update_index = int(row.get("update_index", -1))
            if not prediction_index < release_index <= update_index:
                errors.append("future_origin_feedback")
            if row.get("label_available_at_prediction") is not False:
                errors.append("future_origin_feedback")
            role = str(row.get("label_role") or "")
            accepted = row.get("accepted_update") is True
            if role not in {"update", "admission"} or (role != "update" and accepted):
                errors.append("update_role_invalid")
            if row.get("evaluator_access") is not False:
                errors.append("evaluator_role_leak")
                evaluator_denied = False
            anchor_hashes.add(str(row.get("anchor_hash") or ""))
            component = str(row.get("component_hash") or "")
            origin = str(row.get("feedback_origin") or "")
            if arm in {"unguarded_factual", "guarded_factual"} and origin != component:
                errors.append("factual_feedback_origin_invalid")
            if arm == "guarded_deranged":
                if origin == component or origin not in set(
                    row.get("released_component_ids") or []
                ):
                    errors.append("deranged_feedback_origin_invalid")
                if int(row.get("origin_release_index", 10**9)) > release_index:
                    errors.append("future_origin_feedback")
            if arm == "frozen" and accepted:
                errors.append("frozen_arm_updated")
            before = _weights_hash(weights)
            if row.get("state_hash_before") != before:
                errors.append("state_hash_before_mismatch")
            if accepted:
                update_id = str(row.get("update_id") or "")
                if not update_id or update_id in update_ids:
                    errors.append("duplicate_update")
                update_ids.add(update_id)
                label = int(row.get("feedback_label", -1))
                if label not in (0, 1):
                    errors.append("feedback_label_invalid")
                else:
                    weights = [
                        weight - learning_rate * (predicted - label) * feature
                        for weight, feature in zip(weights, features)
                    ]
            after = _weights_hash(weights)
            if row.get("state_hash_after") != after:
                errors.append("state_hash_after_mismatch")
            restart_match = row.get("restart_hash_before") == row.get("restart_hash_after")
            restart_hashes_match = restart_hashes_match and restart_match
            if not restart_match:
                errors.append("restart_hash_mismatch")
        final_hashes[f"{key[0]}:{key[1]}"] = _weights_hash(weights)
    if len(anchor_hashes) != 1 or "" in anchor_hashes:
        errors.append("anchor_provenance_changed")
    return {
        "qualified": not errors,
        "errors": sorted(set(errors)),
        "order_count": len(orders),
        "arms": arms,
        "exactly_once_updates": "duplicate_update" not in errors,
        "restart_hashes_match": restart_hashes_match,
        "evaluator_denied": evaluator_denied,
        "final_state_hashes": final_hashes,
    }


def _online_fixture_rows(initial: Sequence[float], learning_rate: float) -> list[JsonDict]:
    rows: list[JsonDict] = []
    anchor = canonical_hash({"fit_anchor_ids": ["a0", "a1"]})
    for order_seed in range(7610001, 7610006):
        for arm in ONLINE_ARMS:
            weights = [float(value) for value in initial]
            for position, role in enumerate(("update", "admission")):
                component = f"{order_seed}-{role}"
                features = [1.0, float(position)]
                predicted = _probability_from_logit(sum(a * b for a, b in zip(weights, features)))
                accepted = role == "update" and arm != "frozen"
                label = 0 if arm == "guarded_deranged" else 1
                before = _weights_hash(weights)
                if accepted:
                    weights = [
                        weight - learning_rate * (predicted - label) * feature
                        for weight, feature in zip(weights, features)
                    ]
                after = _weights_hash(weights)
                donor = f"{order_seed}-donor"
                rows.append(
                    {
                        "order_seed": order_seed,
                        "arm": arm,
                        "component_hash": component,
                        "features": features,
                        "prediction_probability": predicted,
                        "prediction_index": position * 2,
                        "release_index": position * 2 + 1,
                        "update_index": position * 2 + 1,
                        "label_available_at_prediction": False,
                        "label_role": role,
                        "feedback_label": label,
                        "feedback_origin": donor if arm == "guarded_deranged" else component,
                        "released_component_ids": [component, donor],
                        "origin_release_index": position * 2 + 1,
                        "accepted_update": accepted,
                        "update_id": f"{order_seed}:{arm}:{component}" if accepted else None,
                        "evaluator_access": False,
                        "anchor_hash": anchor,
                        "state_hash_before": before,
                        "state_hash_after": after,
                        "restart_hash_before": after,
                        "restart_hash_after": after,
                    }
                )
    return rows


def private_fixture() -> JsonDict:
    """Build compact static and online evidence for fail-closed mutation tests."""

    head = {"bias": -0.1, "raw_offset": 0.5, "weights": [0.2] * 8}
    inputs: list[JsonDict] = []
    for index, (raw, label) in enumerate(((1.0, 1), (0.0, 0))):
        factual = [0.8, 0.0, 0.1, 0.9, 0.5, 0.5, 0.0, 0.0]
        erased = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        deranged = [0.1, 0.7, 0.2, 0.2, 0.0, 0.0, 1.0, 0.0]
        for arm, features in zip(EVIDENCE_ARMS, (factual, erased, deranged)):
            inputs.append(
                {
                    "unit_id": f"u{index}",
                    "arm": arm,
                    "features": features,
                    "raw_probability": raw,
                    "label": label,
                    "included": True,
                    "censored": index == 1,
                }
            )
    static_rows = reconstruct_static_rows(inputs, head, seed=7610001)
    initial = [0.0, 0.0]
    learning_rate = 0.1
    return {
        "static_inputs": inputs,
        "static_rows": static_rows,
        "frozen_head": head,
        "online_rows": _online_fixture_rows(initial, learning_rate),
        "initial_weights": initial,
        "learning_rate": learning_rate,
        "verdict_class": "blocked",
        "benefit_passed": False,
    }


MUTATIONS = (
    "false_sign",
    "identical_controls",
    "excluded_failed_rows",
    "wrong_role_labels",
    "future_origin_feedback",
    "false_positive_verdict_class",
)
MUTATION_FAILURES = {
    "false_sign": "brier_sign_or_value_mismatch",
    "identical_controls": "controls_identical",
    "excluded_failed_rows": "failed_row_excluded",
    "wrong_role_labels": "update_role_invalid",
    "future_origin_feedback": "future_origin_feedback",
    "false_positive_verdict_class": "false_positive_verdict_class",
}


def mutate_private_fixture(value: JsonDict, mutation: str) -> str:
    """Apply one named corruption and return its changed private path."""

    if mutation == "false_sign":
        target = max(value["static_rows"], key=lambda row: float(row["brier"]))
        target["brier"] = -float(target["brier"])
        return "static_rows[0].brier"
    if mutation == "identical_controls":
        factual = next(
            row
            for row in value["static_inputs"]
            if row["unit_id"] == "u0" and row["arm"] == "factual"
        )
        erased = next(
            row
            for row in value["static_inputs"]
            if row["unit_id"] == "u0" and row["arm"] == "erased"
        )
        erased["features"] = deepcopy(factual["features"])
        return "static_inputs[u0,erased].features"
    if mutation == "excluded_failed_rows":
        target = next(row for row in value["static_inputs"] if row["censored"] is True)
        target["included"] = False
        return "static_inputs[censored].included"
    if mutation == "wrong_role_labels":
        target = next(row for row in value["online_rows"] if row["accepted_update"] is True)
        target["label_role"] = "evaluator"
        return "online_rows[accepted].label_role"
    if mutation == "future_origin_feedback":
        target = next(row for row in value["online_rows"] if row["arm"] == "guarded_deranged")
        target["origin_release_index"] = target["release_index"] + 1
        return "online_rows[deranged].origin_release_index"
    if mutation == "false_positive_verdict_class":
        value["verdict_class"] = "positive"
        return "verdict_class"
    raise ValueError(f"unknown_mutation:{mutation}")


def validate_private_fixture(value: Mapping[str, Any]) -> list[str]:
    """Run independent readers over private bytes and return all failures."""

    errors: list[str] = []
    try:
        reconstruct_static_rows(
            value.get("static_inputs") or [], value.get("frozen_head") or {}, seed=7610001
        )
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
    try:
        reduce_static_rows(value.get("static_rows") or [], draws=32, seed=7610002)
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
    online = audit_online_rows(
        value.get("online_rows") or [],
        initial_weights=value.get("initial_weights") or [],
        learning_rate=float(value.get("learning_rate", 0.0)),
    )
    errors.extend(online["errors"])
    if value.get("verdict_class") == "positive" and value.get("benefit_passed") is not True:
        errors.append("false_positive_verdict_class")
    return list(dict.fromkeys(errors))


def run_private_mutations() -> list[JsonDict]:
    """Bind each changed fixture hash to the exact independent rejection."""

    receipts: list[JsonDict] = []
    for mutation in MUTATIONS:
        fixture = private_fixture()
        before = canonical_hash(fixture)
        changed_path = mutate_private_fixture(fixture, mutation)
        after = canonical_hash(fixture)
        failures = validate_private_fixture(fixture)
        expected = MUTATION_FAILURES[mutation]
        receipts.append(
            {
                "mutation": mutation,
                "changed_private_path": changed_path,
                "before_sha256": before,
                "after_sha256": after,
                "expected_failure": expected,
                "observed_failures": failures,
                "passed": before != after and expected in failures,
                "corrupted_fixture_published": False,
            }
        )
    return receipts


REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "field_principles",
    "static_audit_ready_score",
    "online_audit_ready_score",
    "branch_conclusions",
    "retirement_rows",
)


def field_principles() -> dict[str, str]:
    """Carry the prompt's omission guards beside each governed field."""

    return {
        "honest_verdict": "A complete prefix reports terminal work; completion does not prove benefit.",
        "verdict_class": "Only the closed verdict classes are legal; external absence is blocked, not partial.",
        "flagged_adversarial": "Flagged evidence cannot open readiness.",
        "gate_check_summary": "Every block retains the failed check and both exact operands.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
        "rows": "Each independent unit and arm retains absolute operands and provenance.",
        "sample_size_budget": "Sources own sample size; orders and windows do not multiply units.",
        "inference_substrate": "The current audit cannot inherit historical model execution.",
        "inference_substrate_class": "Planned and actual execution classes remain explicit.",
        "MODEL_SPECS": "No current LLM call means the current model roster is empty.",
        "invocation_counts": "Current loads, forwards, generations, and tokens stay separate from history.",
        "duration_s": "Monotonic current time excludes inherited work and artificial padding.",
        "random_seed": "Every stochastic reduction has a recorded replay seed.",
        "reproducibility_checksum": "One digest binds immutable inputs and reduction choices.",
        "source_artifact_hashes": "Producer, pre-gate, invalid, flagged, and missing custody stay distinct.",
        "validation_receipts": "Commands, exits, worktree paths, and log hashes bind terminal checks.",
        "verifier_is_oracle": "Exact fixtures cannot establish learned semantic correctness.",
        "field_principles": "Each required field carries its own omission guard.",
        "static_audit_ready_score": "Static readiness reports validity regardless of effect sign.",
        "online_audit_ready_score": "Online readiness reports chronology validity regardless of benefit.",
        "branch_conclusions": "Evidence, decisions, learning, retention, exposure, and cost stay distinct.",
        "retirement_rows": "Only a fully measured repeated construction can retire.",
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks and this self-reference."""

    excluded = {"reproducibility_checksum", "duration_s", "phase_spans"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _gate(check: str, category: str, observed: str, passed: bool) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "operator": "eq",
        "expected": "qualified",
        "observed": observed,
        "passed": passed,
        "principle": "Validity, readiness, benefit, retention, and freshness remain separate.",
    }


def _branch_conclusions(pilot_count: int) -> list[JsonDict]:
    return [
        {
            "branch": "information_value",
            "validity": "pilot_transport_valid_but_all_outputs_schema_invalid",
            "readiness": "blocked_before_static_measurement",
            "benefit": "not_measured",
            "conclusion": "transport_failure_is_not_an_incremental_information_null",
        },
        {
            "branch": "calibration",
            "validity": "blocked_missing_frozen_head_and_evaluation_rows",
            "readiness": "not_established",
            "benefit": "not_measured",
            "conclusion": "no_probability_claim",
        },
        {
            "branch": "decision_value",
            "validity": "blocked_missing_decision_evaluation",
            "readiness": "not_established",
            "benefit": "not_measured",
            "conclusion": "no_cost_or_coverage_claim",
        },
        {
            "branch": "learning",
            "validity": "blocked_missing_guarded_learning_producer",
            "readiness": "not_established",
            "benefit": "not_measured",
            "conclusion": "no_causal_learning_claim",
        },
        {
            "branch": "retention",
            "validity": "blocked_missing_evaluator_only_retention_rows",
            "readiness": "not_established",
            "benefit": "not_measured",
            "conclusion": "no_retention_claim",
        },
        {
            "branch": "exposure",
            "validity": "authenticated_historical_source_exposure",
            "readiness": "descriptive_only",
            "benefit": "not_applicable",
            "conclusion": "fresh_confirmatory_claim_not_allowed",
        },
        {
            "branch": "complete_costs",
            "validity": "pilot_rows_include_all_failed_calls"
            if pilot_count
            else "pilot_unavailable",
            "readiness": "descriptive_only",
            "benefit": "not_established",
            "conclusion": "current_audit_cost_is_aggregation_only",
        },
    ]


def _retirement_rows() -> list[JsonDict]:
    return [
        {
            "construction": "v664_exact_eight_feature_evidence_head",
            "status": "not_retired_pre_inference_block",
            "scientific_hypothesis_retired": False,
            "reopening_condition": "At least six of eight pilot outputs pass the frozen schema, followed by complete static evaluation rows.",
        },
        {
            "construction": "v664_guarded_delayed_residual_head",
            "status": "not_retired_missing_learning_producer",
            "scientific_hypothesis_retired": False,
            "reopening_condition": "Authenticated guarded-learning rows must include five orders, legal derangement, restart hashes, and evaluator-only retention.",
        },
    ]


def provisional_validation_receipts(root: Path) -> list[JsonDict]:
    """Supply schema-valid placeholders only while constructing a private candidate."""

    return [
        {
            "name": name,
            "command": f"pending exact candidate {name}",
            "command_argv": ["pending", name],
            "scope": "exact_candidate" if name in TERMINAL_CHECK_NAMES else "changed_files",
            "worktree": str(root.resolve()),
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "pending",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_blocked_artifact(
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    run_date: str,
) -> JsonDict:
    """Publish a complete external block while retaining valid pilot rows."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "worktree_root": str(root.resolve()),
        "honest_verdict": "complete_blocked_v664_evidence_chain_unavailable",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": blocked_summary(checks),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "acceptance_gate_results": [
            _gate(
                "available_pilot_validity",
                "validity",
                "qualified" if pilot_rows else "unavailable",
                bool(pilot_rows),
            ),
            _gate("static_readiness", "readiness", "blocked_upstream", False),
            _gate("online_readiness", "readiness", "blocked_upstream", False),
            _gate("incremental_information", "benefit", "not_measured", False),
            _gate("decision_value", "benefit", "not_measured", False),
            _gate("causal_learning", "benefit", "not_measured", False),
            _gate("retained_usefulness", "retention", "not_measured", False),
            _gate("fresh_confirmation", "freshness", "historically_exposed", False),
        ],
        "rows": [deepcopy(dict(row)) for row in pilot_rows],
        "sample_size_budget": [
            {
                "branch": "pilot",
                "independent_unit": "source_group",
                "intended": 8,
                "observed": len(pilot_rows),
                "excluded": 0,
                "censored": sum(bool(row.get("censored")) for row in pilot_rows),
            },
            {
                "branch": "static",
                "independent_unit": "source_group",
                "intended": 40,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": 40,
            },
            {
                "branch": "online",
                "independent_unit": "source_group",
                "intended": 80,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": 80,
                "registered_orders": 5,
                "orders_multiply_independent_units": False,
            },
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF (Exp7604 source only)",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "static_bootstrap": 7610002,
            "online_cluster": 7610003,
            "private_mutations": 7610004,
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": {
            str(row.get("name")): bool(row.get("passed"))
            for row in validation_receipts
            if row.get("name") in TERMINAL_CHECK_NAMES
        },
        "verifier_is_oracle": False,
        "oracle_defined_fixture_verdict_class": "circular_positive",
        "oracle_distinct_claim_allowed": False,
        "field_principles": field_principles(),
        "static_audit_ready_score": 0,
        "online_audit_ready_score": 0,
        "branch_conclusions": _branch_conclusions(len(pilot_rows)),
        "retirement_rows": _retirement_rows(),
        "mutation_receipts": run_private_mutations(),
        "fresh_confirmatory_claim_allowed": False,
        "production_activation_authorized": False,
        "default_promotion_authorized": False,
        "generator_weight_change_authorized": False,
        "applicable_numbered_e2e": [],
        "capability_e2e": "raw_to_report_cold_replay",
        "submitted_externally": False,
        "prior_failure_disposition": {
            "prior_experiment": "exp7596-evidence-audit",
            "prior_verdict": "complete_blocked_missing_v663_evidence_producers",
            "repeated_exactly": False,
            "retire_if_same_verdict": True,
            "action": "no_retirement_for_changed_external_block",
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    expected = {*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == expected and all(
        row.get("exit_code") == 0
        and row.get("passed") is True
        and row.get("timed_out") is not True
        and str(row.get("log_sha256") or "").startswith("sha256:")
        for row in by_name.values()
    )


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Reject custody, claim, row, receipt, or checksum drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != "20260924":
        errors.append("run_identity_mismatch")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if value.get("verdict_class") != "blocked":
        errors.append("blocked_class_required")
    if value.get("flagged_adversarial") is not False:
        errors.append("terminal_adversarial_outcome_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("substrate_mismatch")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_mismatch")
    if set(REQUIRED_PRINCIPLE_FIELDS) - set(value.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if value.get("static_audit_ready_score") != 0 or value.get("online_audit_ready_score") != 0:
        errors.append("blocked_readiness_nonzero")
    first = (value.get("gate_check_summary") or {}).get("first_failure")
    if not isinstance(first, Mapping) or set(first) != GATE_OPERAND_FIELDS:
        errors.append("blocked_gate_summary_invalid")
    branch_names = {str(row.get("branch")) for row in value.get("branch_conclusions") or []}
    if branch_names != {
        "information_value",
        "calibration",
        "decision_value",
        "learning",
        "retention",
        "exposure",
        "complete_costs",
    }:
        errors.append("branch_conclusions_incomplete")
    retirement = value.get("retirement_rows") or []
    if len(retirement) != 2 or any(
        row.get("scientific_hypothesis_retired") is not False for row in retirement
    ):
        errors.append("blocked_retirement_invalid")
    mutations = value.get("mutation_receipts") or []
    if (
        {row.get("mutation") for row in mutations} != set(MUTATIONS)
        or not all(row.get("passed") is True for row in mutations)
        or not all(row.get("before_sha256") != row.get("after_sha256") for row in mutations)
    ):
        errors.append("mutation_receipts_invalid")
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts_failed")
    required_row_fields = {
        "unit_id",
        "arm",
        "absolute_metrics",
        "raw_numerator",
        "raw_denominator",
        "seed",
        "direction",
        "censored",
        "censoring",
        "provenance",
    }
    rows = value.get("rows") or []
    if any(set(row) != required_row_fields or row.get("raw_denominator") != 1 for row in rows):
        errors.append("pilot_row_schema_invalid")
    sources = value.get("source_artifact_hashes") or []
    source_names = {
        row.get("upstream")
        for row in sources
        if row.get("upstream") in {spec.upstream for spec in SOURCE_SPECS}
    }
    if source_names != {spec.upstream for spec in SOURCE_SPECS}:
        errors.append("producer_custody_incomplete")
    pilot_source = next(
        (row for row in sources if row.get("upstream") == "exp7604-evidence-pilot"), {}
    )
    if pilot_source.get("disposition") == "authenticated_producer" and len(rows) != 8:
        errors.append("available_pilot_rows_missing")
    for receipt in sources:
        if receipt.get("sha256") is not None:
            try:
                authenticate_source_receipt(receipt, root)
            except ValueError as exc:
                errors.append(str(exc))
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("checksum_mismatch")
    if errors:
        raise ValueError(";".join(dict.fromkeys(errors)))
    return {"valid": True}


NAMED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7596_v663_evidence_audit.py"),
    Path("python/carnot/experiment_7579_v662_decision_learning_audit.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    GAPS_PATH,
    SPEC_PATH,
)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate named inputs and every current V664 producer path."""

    checks: list[JsonDict] = []
    receipts: list[JsonDict] = []
    for relative in NAMED_INPUTS:
        exists = (root / relative).is_file()
        checks.append(
            check_row(
                "required_path", "worktree", relative.as_posix(), "exists", True, exists, "eq"
            )
        )
        if exists:
            receipt = source_receipt(root / relative, root, f"named-input:{relative.as_posix()}")
            receipt.update(disposition="authenticated_instruction", eligible_for_science=False)
            receipts.append(receipt)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        check_row(
            "matching_requirement",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7610",
            True,
            "REQ-REPORT-7610" in spec_text,
            "eq",
        )
    )
    producer_receipts = [classify_source(root, spec) for spec in SOURCE_SPECS]
    receipts.extend(producer_receipts)
    by_upstream = {row["upstream"]: row for row in producer_receipts}
    for upstream in (
        "exp7602-evidence-requalification",
        "exp7603-guarded-update-fixture",
        "exp7604-evidence-pilot",
        "exp7607-evidence-energy",
        "exp7608-decision-evaluation",
        "exp7609-guarded-learning",
    ):
        row = by_upstream[upstream]
        checks.append(
            check_row(
                "required_scientific_producer",
                upstream,
                str(row["producer_path"]),
                "exists_and_eligible",
                True,
                row["eligible_for_science"],
                "eq",
            )
        )
    pilot_path = root / SOURCE_SPECS[2].producer_path
    pilot = load_json(pilot_path) if pilot_path.is_file() else {}
    checks.append(
        check_row(
            "pilot_transport_ready",
            "exp7604-evidence-pilot",
            SOURCE_SPECS[2].producer_path.as_posix(),
            "evidence_transport_ready_score",
            1,
            pilot.get("evidence_transport_ready_score"),
            "eq",
        )
    )
    return checks, receipts


def build_test_artifact(root: Path) -> JsonDict:
    """Build a compact blocked artifact for schema and CLI tests."""

    sources = [classify_source(root, spec) for spec in SOURCE_SPECS]
    checks = [
        check_row(
            "required_scientific_producer",
            row["upstream"],
            row["path"],
            "exists_and_eligible",
            True,
            row["eligible_for_science"],
            "eq",
        )
        for row in sources[-3:]
    ]
    return build_blocked_artifact(
        root,
        checks,
        sources,
        pilot_rows=[],
        validation_receipts=provisional_validation_receipts(root),
        duration_s=0.1,
        phase_spans=[],
        run_date="20260924",
    )


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload exact artifact bytes and run all schema and custody guards."""

    value = load_json(path)
    if not value:
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root)


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reopen authenticated raw inputs and independently rebuild available rows."""

    value = load_json(path)
    validate_artifact(value, root=root)
    sources = {str(row.get("upstream")): row for row in value["source_artifact_hashes"]}
    protocol_source = sources["exp7602-evidence-requalification"]
    pilot_source = sources["exp7604-evidence-pilot"]
    rebuilt: list[JsonDict] = []
    if (
        protocol_source.get("disposition") == "authenticated_producer"
        and pilot_source.get("disposition") == "authenticated_producer"
    ):
        protocol = load_json(authenticate_source_receipt(protocol_source, root))
        pilot = load_json(authenticate_source_receipt(pilot_source, root))
        rebuilt, _receipts = audit_pilot_rows(root, protocol, pilot)
    if canonical_hash(rebuilt) != canonical_hash(value.get("rows") or []):
        raise ValueError("independent_pilot_reduction_mismatch")
    return {"valid": True, "row_count": len(rebuilt), "source_count": len(sources)}


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze serial tests, changed-module coverage, and scoped static checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(),),
        (MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=private_root,
        coverage_file=private_root / ".coverage.exp7610",
    )


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    """Build bounded fresh-process readers for one exact candidate."""

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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


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


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every boundary so long validation work remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7610] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _write_manifest(root: Path) -> tuple[Path, JsonDict]:  # pragma: no cover
    path = root / RAW_DIR / "affected_validation_manifest.json"
    value = {
        "experiment_id": EXPERIMENT_ID,
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
        "spec_paths": [SPEC_PATH.as_posix()],
        "documentation_paths": [NOTE_PATH.as_posix(), GAPS_PATH.as_posix()],
    }
    atomic_json(path, value)
    return path, value


def _task_source_receipt(root: Path, relative: Path, upstream: str) -> JsonDict:  # pragma: no cover
    receipt = source_receipt(root / relative, root, upstream)
    receipt.update(disposition="task_owned_source", eligible_for_science=False)
    return receipt


def run_experiment(  # pragma: no cover - exercised by the declared raw-to-report E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Authenticate, reduce available rows, validate, and publish atomically."""

    root = root.resolve()
    if run_date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    started = time.monotonic()
    spans: list[JsonDict] = []
    destination = root / RESULT_PATH

    progress(started, "inputs", "start", root=root)
    phase_started = time.monotonic()
    checks, sources = collect_preconditions(root)
    failures = [row for row in checks if row.get("passed") is not True]
    if not failures:
        raise RuntimeError("blocked_audit_expected_currently_complete_science_requires_extension")
    spans.append(_span("inputs", phase_started, started, len(checks)))
    progress(started, "inputs", "complete", completed_units=len(checks), failed=len(failures))

    progress(started, "raw_reduction", "start")
    phase_started = time.monotonic()
    by_source = {str(row.get("upstream")): row for row in sources}
    protocol = load_json(
        authenticate_source_receipt(by_source["exp7602-evidence-requalification"], root)
    )
    pilot = load_json(authenticate_source_receipt(by_source["exp7604-evidence-pilot"], root))
    pilot_rows, raw_receipts = audit_pilot_rows(root, protocol, pilot)
    for index, receipt in enumerate(raw_receipts):
        receipt.update(
            upstream=f"exp7610-authenticated-raw-{index:02d}",
            disposition="authenticated_raw_sidecar",
            eligible_for_science=True,
        )
    sources.extend(raw_receipts)
    spans.append(_span("raw_reduction", phase_started, started, len(pilot_rows)))
    progress(started, "raw_reduction", "complete", completed_units=len(pilot_rows))

    progress(started, "manifest_and_mutations", "start", planned_units=len(MUTATIONS))
    phase_started = time.monotonic()
    manifest_path, _manifest = _write_manifest(root)
    for relative, upstream in (
        (MODULE_PATH, "exp7610.implementation"),
        (WRAPPER_PATH, "exp7610.entrypoint"),
        (TEST_PATH, "exp7610.tests"),
        (SPEC_PATH, "exp7610.spec"),
        (NOTE_PATH, "exp7610.note"),
        (GAPS_PATH, "exp7610.gaps"),
    ):
        sources.append(_task_source_receipt(root, relative, upstream))
    sources.append(_task_source_receipt(root, manifest_path.relative_to(root), "exp7610.manifest"))
    mutations = run_private_mutations()
    if not all(row["passed"] is True for row in mutations):
        raise RuntimeError("private_mutation_panel_failed")
    spans.append(_span("manifest_and_mutations", phase_started, started, len(mutations)))
    progress(started, "manifest_and_mutations", "complete", completed_units=len(mutations))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7610-", dir="/tmp")).resolve()
    commands = build_validation_commands(root, private_root / "pytest")
    progress(started, "scoped_validation", "before_subprocesses", planned_units=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root, commands, log_dir=private_root / "validation_logs", heartbeat_s=60.0
    )
    for receipt in affected:
        receipt["worktree"] = str(root)
    spans.append(_span("scoped_validation", phase_started, started, len(affected)))
    affected_passed = validation_scope.reduce_required_checks(affected)["required_checks_passed"]
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_passed,
    )
    if not affected_passed:
        raise RuntimeError("required_scoped_validation_failed")

    candidate = private_root / "terminal_candidate.json"
    pending_terminal = [
        row for row in provisional_validation_receipts(root) if row["name"] in TERMINAL_CHECK_NAMES
    ]
    provisional = build_blocked_artifact(
        root,
        failures,
        sources,
        pilot_rows=pilot_rows,
        validation_receipts=[*affected, *pending_terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        run_date=run_date,
    )
    atomic_json(candidate, provisional)
    plan = terminal_commands(candidate, root)
    progress(started, "terminal_validation", "before_subprocesses", planned_units=len(plan))
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root, plan, log_dir=private_root / "terminal_logs_provisional", heartbeat_s=60.0
    )
    for receipt in terminal:
        receipt["worktree"] = str(root)
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    if not all(row.get("passed") is True for row in terminal):
        raise RuntimeError("terminal_candidate_validation_failed")
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=True,
    )

    final = build_blocked_artifact(
        root,
        failures,
        sources,
        pilot_rows=pilot_rows,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        run_date=run_date,
    )
    validate_artifact(final, root=root)
    atomic_json(candidate, final)
    exact_plan = terminal_commands(candidate, root)
    progress(started, "exact_candidate", "before_subprocesses", planned_units=len(exact_plan))
    exact = validation_scope.run_commands(
        root, exact_plan, log_dir=private_root / "terminal_logs_exact", heartbeat_s=60.0
    )
    if not all(row.get("passed") is True for row in exact):
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(root / RAW_DIR / "exact_terminal_receipts.json", {"receipts": exact})
    progress(
        started, "exact_candidate", "after_subprocesses", completed_units=len(exact), passed=True
    )

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
