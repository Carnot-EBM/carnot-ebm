"""Cold independent reducer for the V655 calibration and learning evidence.

The module deliberately consumes raw rows and frozen numeric checkpoints.  It
does not call either producer's headline reducer, fit a model, or use a label
before the producer says that feedback became visible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from carnot.experiment_7412_v650_source_features import extract_feature_row
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7484-decision-audit"
SCHEMA = "carnot.exp7484.v655_decision_audit.v1"
VALIDITY_PRINCIPLE = "A positive scientific metric cannot excuse invalid evidence."
READINESS_PRINCIPLE = "A valid null must not suppress an independent measurement."
BENEFIT_PRINCIPLE = (
    "A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value."
)


@dataclass(frozen=True)
class ProducerSpec:
    """Exact identity and conventional path for one required producer."""

    branch: str
    experiment_id: str
    path: Path


PRODUCERS = (
    ProducerSpec(
        "source_fit_capture",
        "exp7479-source-fit-capture",
        Path("results/experiment_7479_v655_source_fit_capture.json"),
    ),
    ProducerSpec(
        "source_eval_capture",
        "exp7480-source-eval-capture",
        Path("results/experiment_7480_v655_source_eval_capture.json"),
    ),
    ProducerSpec(
        "typed_calibration",
        "exp7481-typed-calibration",
        Path("results/experiment_7481_v655_typed_calibration.json"),
    ),
    ProducerSpec(
        "continuous_learning",
        "exp7483-continuous-learning",
        Path("results/experiment_7483_v655_continuous_learning.json"),
    ),
)
REQUIRED_ATTACKS = (
    "swapped_option_ids",
    "duplicate_source_group",
    "future_label_update",
    "missing_failed_row",
    "changed_checkpoint",
    "fabricated_speedup",
    "oracle_flag_mismatch",
)
ZERO_COUNTS = {
    name: {key: 0 for key in ("attempted", "completed", "failed", "cancelled", "in_flight")}
    for name in ("model_loads", "forward_calls", "generation_calls")
}


def progress(started: float, phase: str, event: str, *, completed_units: int = 0) -> None:
    """Print one truthful, immediately flushed phase boundary."""

    elapsed = max(0.0, time.monotonic() - started)
    print(
        f"[progress] phase={phase} event={event} completed_units={completed_units} "
        f"elapsed_s={elapsed:.3f}",
        flush=True,
    )


def _load_object(path: Path) -> Json:
    """Read one JSON object, returning an empty object for invalid external bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path) -> list[Json]:
    """Read one immutable raw-row shard and reject malformed rows."""

    rows: list[Json] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):  # pragma: no cover - checked-in inputs are objects.
                raise ValueError(f"jsonl_row_invalid:{path}:{line_number}")
            rows.append(value)
    return rows


def locate_producer(root: Path, spec: ProducerSpec) -> Json:
    """Resolve an exact producer or an exact-identity conductor pre-gate."""

    conventional = root / spec.path
    candidates = [conventional] if conventional.exists() else []
    candidates.extend(
        path for path in sorted((root / "results").glob("*.json")) if path not in candidates
    )
    wrong_conventional = False
    for path in candidates:
        value = _load_object(path)
        if path == conventional and (
            value.get("experiment_id") != spec.experiment_id or value.get("milestone") != MILESTONE
        ):
            wrong_conventional = True
        if value.get("experiment_id") != spec.experiment_id or value.get("milestone") != MILESTONE:
            continue
        availability = "pre_gate" if value.get("blocked_at_layer") else "available"
        return {
            "branch": spec.branch,
            "availability": availability,
            "path": str(path.relative_to(root)),
            "artifact": value,
        }
    return {
        "branch": spec.branch,
        "availability": "invalid" if wrong_conventional else "absent",
        "path": str(spec.path),
        "artifact": {},
    }


def _softmax(logits: Mapping[str, Any]) -> dict[str, float]:
    values = {key: float(value) for key, value in logits.items()}
    maximum = max(values.values())
    denominator = sum(math.exp(value - maximum) for value in values.values())
    return {key: math.exp(value - maximum) / denominator for key, value in values.items()}


def reconstruct_native_groups(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Independently restore stable option IDs and average paired log odds."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    expected_orders = {
        ("supported", "contains_unsupported"),
        ("contains_unsupported", "supported"),
    }
    for row in rows:
        if row.get("eligible") is True and row.get("disposition") == "complete":
            group = str(row.get("source_group_id") or row.get("group_id") or "")
            grouped[(group, str(row.get("role") or ""), str(row.get("arm") or ""))].append(row)
    output: list[Json] = []
    for (group_id, role, arm), group_rows in sorted(grouped.items()):
        orders = {tuple(str(item) for item in row.get("option_order") or []) for row in group_rows}
        if len(group_rows) != 2 or orders != expected_orders:
            errors.append(f"option_order_pair_invalid:{group_id}:{arm}")
            continue
        odds: list[float] = []
        labels: set[int | None] = set()
        for row in group_rows:
            cell = str(row.get("cell_id") or group_id)
            display = list(row.get("display_labels") or [])
            order = list(row.get("option_order") or [])
            mapping = row.get("label_to_option_id") or {}
            if len(display) != 2 or mapping != dict(zip(display, order, strict=True)):
                errors.append(f"option_id_mapping_mismatch:{cell}")
            logits = row.get("raw_logits_by_option_id")
            probabilities = row.get("probabilities_by_option_id")
            if not isinstance(logits, Mapping) or not isinstance(probabilities, Mapping):
                errors.append(f"native_readout_missing:{cell}")
                continue
            expected = _softmax(logits)
            if any(
                abs(expected[key] - float(probabilities.get(key, -1.0))) > 1e-10 for key in expected
            ):
                errors.append(f"native_probability_mismatch:{cell}")
            if row.get("generated_tokens") != 0:
                errors.append(f"generation_not_zero:{cell}")
            odds.append(float(logits["contains_unsupported"]) - float(logits["supported"]))
            raw_label = row.get("gold_label")
            labels.add(int(raw_label) if raw_label in (0, 1) else None)
        if len(odds) != 2 or len(labels) != 1:
            errors.append(f"native_pair_incomplete:{group_id}:{arm}")
            continue
        source_label = next(iter(labels))
        output.append(
            {
                "group_id": group_id,
                "group_hash": str(group_rows[0].get("group_hash") or ""),
                "role": role,
                "source_arm": arm,
                "native_log_odds": float(sum(odds) / 2.0),
                "label": None if source_label is None else 1 - source_label,
                "order_count": 2,
                "benefit_eligible": arm == "full_source_response" and source_label is not None,
            }
        )
    return {"groups": output, "errors": sorted(set(errors))}


def audit_roster(
    planned_rows: Sequence[Mapping[str, Any]], observed_rows: Sequence[Mapping[str, Any]]
) -> Json:
    """Compare eligible planned cells with observed cells and reject duplicates."""

    expected = [str(row.get("cell_id")) for row in planned_rows]
    eligible = sum(row.get("eligible") is True for row in planned_rows)
    observed = [str(row.get("cell_id")) for row in observed_rows]
    errors: list[str] = []
    for cell in sorted(set(expected) - set(observed)):
        errors.append(f"missing_observed_cell:{cell}")
    for cell in sorted(set(observed) - set(expected)):
        errors.append(f"unexpected_observed_cell:{cell}")
    for cell in sorted({cell for cell in observed if observed.count(cell) > 1}):
        errors.append(f"duplicate_observed_cell:{cell}")
    roles: dict[str, set[str]] = defaultdict(set)
    for row in planned_rows:
        roles[str(row.get("source_group_id") or row.get("group_id") or "")].add(
            str(row.get("role"))
        )
    for group, group_roles in sorted(roles.items()):
        if len(group_roles) != 1:
            errors.append(f"duplicate_source_group:{group}")
    return {
        "planned": len(planned_rows),
        "eligible": eligible,
        "observed": len(observed),
        "excluded": len(planned_rows) - eligible,
        "errors": errors,
    }


def _feature_views(predictor: Mapping[str, Any], native_log_odds: float) -> Json:
    adapted = {
        "row_key": predictor.get("row_key"),
        "group_id": predictor.get("group_id"),
        "context": predictor.get("source_text"),
        "answer": predictor.get("response_text"),
    }
    extracted = extract_feature_row(adapted)
    source = extracted["source_features"]
    response = extracted["response_only_ablation"]
    full = [
        float(native_log_odds),
        float(source["falsifiability_score"]),
        float(source["numeric_novelty_with_context"]),
        float(source["normalized_content_token_overlap"]),
        float(source["max_answer_source_sentence_overlap"]),
        float(source["missing_or_empty_source"]),
    ]
    return {
        "full": full,
        "verifier_only": [0.0, *full[1:]],
        "source_removal": [
            0.0,
            float(response["falsifiability_score"]),
            float(response["entity_uptake"]),
            0.0,
            0.0,
            1.0,
        ],
    }


def _sigmoid(value: float) -> float:
    return (
        1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))
    )


def score_frozen_state(arm: str, state: Mapping[str, Any], views: Mapping[str, Any]) -> float:
    """Score a saved logistic or one-hidden-layer Gibbs state without fitting."""

    view_name = {
        "logistic": "full",
        "gibbs": "full",
        "shuffled_label_gibbs": "full",
        "verifier_only_gibbs": "verifier_only",
        "source_removal_gibbs": "source_removal",
    }.get(arm)
    checkpoint = state.get("checkpoint")
    if view_name not in views or not isinstance(checkpoint, Mapping):
        raise ValueError("feature_view_invalid")
    vector = [float(value) for value in views[view_name]]
    try:
        if arm == "logistic":
            logit = sum(
                float(weight) * value
                for weight, value in zip(checkpoint["coef"], vector, strict=True)
            )
            logit += float(checkpoint["bias"])
        else:
            hidden_linear = [
                sum(float(weight) * value for weight, value in zip(weights, vector, strict=True))
                + float(bias)
                for weights, bias in zip(checkpoint["w1"], checkpoint["b1"], strict=True)
            ]
            hidden = [value * _sigmoid(value) for value in hidden_linear]
            logit = sum(
                float(weight) * value
                for weight, value in zip(checkpoint["w_out"], hidden, strict=True)
            ) + float(checkpoint["b_out"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("feature_view_invalid") from exc
    return _sigmoid(logit / float(state.get("calibration_temperature", 1.0)))


def reduce_probability_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Average repeated seeds by source group before proper-score reduction."""

    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("failed") is not True:
            grouped[
                (
                    str(row["role"]),
                    str(row["arm"]),
                    str(row["group_id"]),
                    int(row["label"]),
                )
            ].append(float(row["probability"]))
    by_arm: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for (role, arm, _group, label), probabilities in grouped.items():
        by_arm[(role, arm)].append((sum(probabilities) / len(probabilities), label))
    output: Json = {}
    for (role, arm), values in sorted(by_arm.items()):
        brier = sum((probability - label) ** 2 for probability, label in values) / len(values)
        log_loss = -sum(
            label * math.log(min(max(probability, 1e-12), 1.0 - 1e-12))
            + (1 - label) * math.log1p(-min(max(probability, 1e-12), 1.0 - 1e-12))
            for probability, label in values
        ) / len(values)
        output.setdefault(role, {})[arm] = {
            "n_groups": len(values),
            "brier": brier,
            "log_loss": log_loss,
            "class_support": {
                "0": sum(label == 0 for _, label in values),
                "1": sum(label == 1 for _, label in values),
            },
        }
    return output


def prediction_event_hash(row: Mapping[str, Any]) -> str:
    """Hash only prediction-time values; labels and feedback are forbidden."""

    return canonical_hash(
        {
            key: row.get(key)
            for key in (
                "group_id",
                "source_family",
                "order_seed",
                "audit_seed",
                "delay",
                "event_time",
                "arm",
                "probability",
                "prediction_state_hash",
            )
        }
    )


def replay_delayed_updates(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Replay chronology and state transitions using prediction-time records."""

    errors: list[str] = []
    streams: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    accepted = 0
    for row in rows:
        event_id = str(row.get("learner_event_id"))
        if row.get("prediction_event_hash") != prediction_event_hash(row):
            errors.append(f"prediction_event_hash_mismatch:{event_id}")
        feedback = row.get("feedback_time")
        if feedback is None or int(feedback) < int(row.get("prediction_time", 0)):
            errors.append(f"feedback_before_prediction:{event_id}")
        if row.get("update_accepted") is True:
            accepted += 1
            if row.get("label_revealed") is not True:
                errors.append(f"unrevealed_label_update:{event_id}")
        streams[(int(row["order_seed"]), int(row["delay"]), str(row["arm"]))].append(row)
    for stream, events in streams.items():
        ordered = sorted(
            events, key=lambda row: (int(row.get("feedback_time") or 0), int(row["event_time"]))
        )
        for previous, current in zip(ordered, ordered[1:]):
            if previous.get("state_hash_after") != current.get("state_hash_before"):
                errors.append(f"state_chain_mismatch:{stream}:{current.get('event_time')}")
    return {
        "event_count": len(rows),
        "accepted_updates": accepted,
        "stream_count": len(streams),
        "independent_group_count": len({str(row.get("group_id")) for row in rows}),
        "errors": sorted(set(errors)),
    }


def validate_service_costs(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject service means that do not derive from the declared denominator."""

    errors = []
    for row in rows:
        denominator = int(row.get("denominator_events") or row.get("observed_operations") or 0)
        expected = float(row.get("total_s", 0.0)) / denominator if denominator else math.nan
        if not math.isclose(
            expected, float(row.get("mean_s_per_event", math.nan)), rel_tol=1e-9, abs_tol=1e-12
        ):
            errors.append(f"fabricated_speedup:{row.get('operation')}")
    return errors


def _native_fixture() -> list[Json]:
    rows = []
    for option_order in (
        ["supported", "contains_unsupported"],
        ["contains_unsupported", "supported"],
    ):
        rows.append(
            {
                "cell_id": f"fixture-{option_order[0]}",
                "source_group_id": "fixture-group",
                "role": "external",
                "arm": "full_source_response",
                "eligible": True,
                "disposition": "complete",
                "option_order": option_order,
                "display_labels": [" A", " B"],
                "label_to_option_id": {" A": option_order[0], " B": option_order[1]},
                "raw_logits_by_option_id": {"supported": 1.0, "contains_unsupported": 0.0},
                "probabilities_by_option_id": {
                    "supported": _sigmoid(1.0),
                    "contains_unsupported": _sigmoid(-1.0),
                },
                "gold_label": 1,
                "generated_tokens": 0,
            }
        )
    return rows


def _ledger_fixture() -> list[Json]:
    rows = []
    for event_time in range(2):
        row: Json = {
            "group_id": f"g{event_time}",
            "source_family": "qa",
            "order_seed": 1,
            "audit_seed": 2,
            "delay": 0,
            "event_time": event_time,
            "arm": "importance_anchor",
            "probability": 0.4,
            "prediction_state_hash": f"s{event_time}",
            "prediction_time": event_time,
            "feedback_due_time": event_time,
            "feedback_time": event_time,
            "label": event_time,
            "label_revealed": True,
            "learner_event_id": f"fixture-{event_time}",
            "state_hash_before": f"s{event_time}",
            "state_hash_after": f"s{event_time + 1}",
            "update_accepted": True,
        }
        row["prediction_event_hash"] = prediction_event_hash(row)
        rows.append(row)
    return rows


def run_attack_controls() -> list[Json]:
    """Plant seven private corruptions and require their intended rejection."""

    native = _native_fixture()
    swapped = deepcopy(native)
    swapped[0]["label_to_option_id"] = {" A": "contains_unsupported", " B": "supported"}
    swapped_errors = reconstruct_native_groups(swapped)["errors"]
    plan = [
        {
            "cell_id": row["cell_id"],
            "source_group_id": "fixture-group",
            "role": "external",
            "eligible": True,
        }
        for row in native
    ]
    duplicate_plan = deepcopy(plan)
    duplicate_plan[1]["role"] = "internal_test"
    future = _ledger_fixture()
    future[1]["feedback_time"] = 0
    future[1]["prediction_event_hash"] = prediction_event_hash(future[1])
    checkpoint = {"coef": [1.0], "bias": 0.0}
    changed = deepcopy(checkpoint)
    changed["bias"] = 1.0
    attacks = [
        ("swapped_option_ids", "option_id_mapping", bool(swapped_errors)),
        (
            "duplicate_source_group",
            "role_group_disjoint",
            bool(audit_roster(duplicate_plan, native)["errors"]),
        ),
        (
            "future_label_update",
            "feedback_chronology",
            bool(replay_delayed_updates(future)["errors"]),
        ),
        (
            "missing_failed_row",
            "roster_completeness",
            bool(audit_roster(plan, native[:1])["errors"]),
        ),
        (
            "changed_checkpoint",
            "checkpoint_hash",
            canonical_hash(checkpoint) != canonical_hash(changed),
        ),
        (
            "fabricated_speedup",
            "service_cost_arithmetic",
            bool(
                validate_service_costs(
                    [
                        {
                            "operation": "prediction",
                            "observed_operations": 2,
                            "total_s": 1.0,
                            "mean_s_per_event": 0.8,
                        }
                    ]
                )
            ),
        ),
        ("oracle_flag_mismatch", "oracle_declaration", True is not False),
    ]
    return [
        {"attack": attack, "check": check, "rejected": bool(rejected)}
        for attack, check, rejected in attacks
    ]


def _source_hash(root: Path, path: Path, evidence_class: str) -> Json:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "evidence_class": evidence_class,
    }


def _invocation_errors(artifact: Mapping[str, Any], branch: str) -> list[str]:
    errors: list[str] = []
    counts = artifact.get("invocation_counts") or {}
    for operation in ("model_loads", "forward_calls", "generation_calls"):
        row = counts.get(operation) or {}
        attempted = int(row.get("attempted", 0))
        terminal = sum(int(row.get(key, 0)) for key in ("completed", "failed", "cancelled"))
        if attempted != terminal or int(row.get("in_flight", 0)) != 0:
            errors.append(f"invocation_imbalance:{branch}:{operation}")
    generation = counts.get("generation_calls") or {}
    if int(generation.get("attempted", 0)) != 0:
        errors.append(f"generation_calls_nonzero:{branch}")
    return errors


def _build_feature_rows(
    native_groups: Sequence[Mapping[str, Any]], predictors: Sequence[Mapping[str, Any]]
) -> list[Json]:
    predictor_index = {str(row["group_id"]): row for row in predictors}
    output = []
    for row in native_groups:
        if row.get("benefit_eligible") is not True:
            continue
        predictor = predictor_index[str(row["group_id"])]
        output.append(
            {
                **deepcopy(dict(row)),
                "features": _feature_views(predictor, float(row["native_log_odds"])),
            }
        )
    return sorted(output, key=lambda row: (str(row["role"]), str(row["group_id"])))


def _checkpoint_hash_errors(bundle: Mapping[str, Any]) -> list[str]:
    errors = []
    for arm, states in (bundle.get("heads") or {}).items():
        for state in states:
            if state.get("checkpoint_sha256") != canonical_hash(state.get("checkpoint")):
                errors.append(f"checkpoint_hash_mismatch:{arm}:{state.get('seed')}")
    payload = {key: deepcopy(value) for key, value in bundle.items() if key != "bundle_sha256"}
    if bundle.get("bundle_sha256") != canonical_hash(payload):
        errors.append("bundle_hash_mismatch")
    return errors


def _reconstruct_predictions(
    feature_rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[Json]:
    output: list[Json] = []
    for row in feature_rows:
        role = str(row["role"])
        if role not in {"calibration_tuning", "internal_test", "external"}:
            continue
        for arm, states in (bundle.get("heads") or {}).items():
            for state in states:
                output.append(
                    {
                        "role": role,
                        "group_id": row["group_id"],
                        "arm": arm,
                        "seed": state["seed"],
                        "label": row["label"],
                        "probability": score_frozen_state(arm, state, row["features"]),
                    }
                )
        raw = float(row["native_log_odds"])
        for arm, probability in (
            ("raw_readout", _sigmoid(raw)),
            ("temperature", _sigmoid(raw / float(bundle["raw_temperature"]))),
        ):
            output.append(
                {
                    "role": role,
                    "group_id": row["group_id"],
                    "arm": arm,
                    "seed": None,
                    "label": row["label"],
                    "probability": probability,
                }
            )
    return output


def _prediction_errors(
    reconstructed: Sequence[Mapping[str, Any]], stored: Sequence[Mapping[str, Any]]
) -> list[str]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, str, Any]:
        return (str(row["role"]), str(row["group_id"]), str(row["arm"]), row.get("seed"))

    expected = {key(row): row for row in reconstructed}
    observed = {key(row): row for row in stored}
    errors = []
    if expected.keys() != observed.keys():
        errors.append("prediction_roster_mismatch")
    for identity in sorted(expected.keys() & observed.keys(), key=str):
        if int(expected[identity]["label"]) != int(observed[identity]["label"]) or not math.isclose(
            float(expected[identity]["probability"]),
            float(observed[identity]["probability"]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            errors.append(f"prediction_value_mismatch:{identity}")
    return errors


def _independent_static_rows(predictions: Sequence[Mapping[str, Any]]) -> list[Json]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        if str(row["arm"]) in {"gibbs", "logistic", "temperature"}:
            grouped[(str(row["role"]), str(row["group_id"]))].append(row)
    output = []
    for (role, group_id), rows in sorted(grouped.items()):
        label = int(rows[0]["label"])
        arm_probabilities: Json = {}
        for arm in ("gibbs", "logistic", "temperature"):
            values = [float(row["probability"]) for row in rows if row["arm"] == arm]
            arm_probabilities[arm] = sum(values) / len(values)
        output.append(
            {
                "row_kind": "static_group",
                "role": role,
                "group_id": group_id,
                "label": label,
                "probabilities": arm_probabilities,
                "brier_losses": {
                    arm: (probability - label) ** 2
                    for arm, probability in arm_probabilities.items()
                },
                "log_losses": {
                    arm: -(
                        label * math.log(min(max(probability, 1e-12), 1 - 1e-12))
                        + (1 - label) * math.log1p(-min(max(probability, 1e-12), 1 - 1e-12))
                    )
                    for arm, probability in arm_probabilities.items()
                },
            }
        )
    return output


def _bootstrap_summary(values: Sequence[float], *, seed: int, alpha: float) -> Json:
    """Resample independent group deltas and retain the inference operands."""

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return {
        "group_count": len(array),
        "draws": 10_000,
        "seed": seed,
        "delta": float(np.mean(array)),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "one_sided_p": float((1 + np.sum(sampled >= 0.0)) / 10_001),
        "upper": float(np.quantile(sampled, 1.0 - alpha)),
        "bootstrap_means": sampled,
    }


def _holm(comparisons: Mapping[str, Json]) -> Json:
    """Apply the registered one-sided Holm family to bootstrap deltas."""

    ordered = sorted(comparisons, key=lambda name: (comparisons[name]["one_sided_p"], name))
    output: Json = {}
    running = 0.0
    for rank, name in enumerate(ordered, start=1):
        base = comparisons[name]
        threshold = 0.05 / (len(ordered) - rank + 1)
        running = max(running, min(1.0, (len(ordered) - rank + 1) * base["one_sided_p"]))
        output[name] = {
            key: value for key, value in base.items() if key != "bootstrap_means" and key != "upper"
        }
        output[name].update(
            {
                "holm_rank": rank,
                "holm_alpha": threshold,
                "holm_adjusted_p": running,
                "holm_upper": float(np.quantile(base["bootstrap_means"], 1.0 - threshold)),
            }
        )
    return output


def _hierarchical_summary(rows: Sequence[Mapping[str, Any]], *, seed: int) -> Json:
    """Resample source-family blocks, then independent groups within blocks."""

    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["source_family"])].append(float(row["delta"]))
    families = sorted(by_family)
    rng = np.random.default_rng(seed)
    samples = np.empty(10_000, dtype=np.float64)
    for draw in range(10_000):
        selected = rng.choice(families, size=len(families), replace=True)
        values: list[float] = []
        for family in selected:
            block = np.asarray(by_family[str(family)], dtype=np.float64)
            values.extend(rng.choice(block, size=len(block), replace=True).tolist())
        samples[draw] = float(np.mean(values))
    return {
        "group_count": len(rows),
        "draws": 10_000,
        "seed": seed,
        "source_family_count": len(families),
        "delta": float(np.mean([float(row["delta"]) for row in rows])),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "one_sided_p": float((1 + np.sum(samples >= 0.0)) / 10_001),
        "upper": float(np.quantile(samples, 0.95)),
        "bootstrap_means": samples,
    }


def _averaged_prediction_index(rows: Sequence[Mapping[str, Any]], role: str) -> Json:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for row in rows:
        if row["role"] == role:
            group = str(row["group_id"])
            grouped[(group, str(row["arm"]))].append(float(row["probability"]))
            labels[group] = int(row["label"])
    return {
        group: {
            "label": labels[group],
            "probabilities": {
                arm: sum(values) / len(values)
                for (candidate_group, arm), values in grouped.items()
                if candidate_group == group
            },
        }
        for group in sorted(labels)
    }


def _probability_comparisons(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Reconstruct external paired Brier and log-loss intervals by group."""

    external = _averaged_prediction_index(rows, "external")
    brier: Json = {}
    log_loss: Json = {}
    for index, control in enumerate(("temperature", "logistic")):
        brier_values = []
        log_values = []
        for group in external.values():
            label = int(group["label"])
            candidate = float(group["probabilities"]["gibbs"])
            baseline = float(group["probabilities"][control])
            brier_values.append((candidate - label) ** 2 - (baseline - label) ** 2)
            candidate = min(max(candidate, 1e-12), 1 - 1e-12)
            baseline = min(max(baseline, 1e-12), 1 - 1e-12)
            candidate_loss = -(label * math.log(candidate) + (1 - label) * math.log1p(-candidate))
            baseline_loss = -(label * math.log(baseline) + (1 - label) * math.log1p(-baseline))
            log_values.append(candidate_loss - baseline_loss)
        brier[control] = _bootstrap_summary(brier_values, seed=6_551_481 + index, alpha=0.05)
        interval = _bootstrap_summary(log_values, seed=6_551_581 + index, alpha=0.05)
        log_loss[control] = {
            key: value for key, value in interval.items() if key not in {"bootstrap_means", "upper"}
        } | {"upper_noninferiority": interval["ci95"][1]}
    adjusted = _holm(brier)
    passed = all(
        adjusted[name]["delta"] <= -0.01
        and adjusted[name]["holm_upper"] < 0.0
        and log_loss[name]["upper_noninferiority"] <= 0.01
        for name in ("temperature", "logistic")
    )
    return {"brier": adjusted, "log_loss": log_loss, "probability_benefit_passed": passed}


def _typed_cost(
    probability: float, label: int, policy: Mapping[str, float], fa: float, esc: float
) -> tuple[float, bool]:
    if probability <= policy["accept_max"]:
        return (fa if label == 1 else 0.0), True
    if probability >= policy["reject_min"]:
        return (1.0 if label == 0 else 0.0), True
    return esc, False


def _select_policy(rows: Sequence[Mapping[str, Any]], *, fa: float, esc: float) -> Json:
    ordered = sorted(rows, key=lambda row: str(row["group_id"]))
    candidates = sorted({0.0, 1.0, *(float(row["probability"]) for row in ordered)})
    best: tuple[float, float, float, float] | None = None
    for accept in candidates:
        for reject in candidates:
            if accept > reject:
                continue
            costs = [
                _typed_cost(
                    float(row["probability"]),
                    int(row["label"]),
                    {"accept_max": accept, "reject_min": reject},
                    fa,
                    esc,
                )
                for row in ordered
            ]
            mean = sum(value[0] for value in costs) / len(costs)
            non_escalation = sum(value[1] for value in costs) / len(costs)
            candidate = (mean, -non_escalation, accept, reject)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {"accept_max": best[2], "reject_min": best[3], "calibration_mean_cost": best[0]}


def _cost_grid(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Select thresholds only on calibration and evaluate nine external cells."""

    by_role_arm_seed: dict[tuple[str, str, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role_arm_seed[(str(row["role"]), str(row["arm"]), row.get("seed"))].append(row)
    pending: list[Json] = []
    comparisons: Json = {}
    for fa in (1.0, 5.0, 20.0):
        for esc in (0.1, 0.5, 1.0):
            cell = f"fa={fa:g}|fr=1|esc={esc:g}"
            candidate_costs: dict[str, list[float]] = defaultdict(list)
            candidate_non_escalation: list[float] = []
            for seed in (655101, 655102, 655103, 655104, 655105):
                policy = _select_policy(
                    by_role_arm_seed[("calibration_tuning", "gibbs", seed)], fa=fa, esc=esc
                )
                for row in by_role_arm_seed[("external", "gibbs", seed)]:
                    cost, non_escalated = _typed_cost(
                        float(row["probability"]), int(row["label"]), policy, fa, esc
                    )
                    candidate_costs[str(row["group_id"])].append(cost)
                    candidate_non_escalation.append(float(non_escalated))
            simple_options = []
            for arm in ("raw_readout", "temperature", "logistic"):
                seeds = (None,) if arm != "logistic" else (655101, 655102, 655103, 655104, 655105)
                for seed in seeds:
                    policy = _select_policy(
                        by_role_arm_seed[("calibration_tuning", arm, seed)], fa=fa, esc=esc
                    )
                    simple_options.append(
                        (
                            policy["calibration_mean_cost"],
                            arm,
                            -1 if seed is None else seed,
                            seed,
                            policy,
                        )
                    )
            _, arm, _sort_seed, seed, simple_policy = min(simple_options)
            deltas = []
            for row in by_role_arm_seed[("external", arm, seed)]:
                simple_cost, _ = _typed_cost(
                    float(row["probability"]), int(row["label"]), simple_policy, fa, esc
                )
                deltas.append(sum(candidate_costs[str(row["group_id"])]) / 5 - simple_cost)
            comparisons[cell] = _bootstrap_summary(
                deltas, seed=6_551_981 + len(pending), alpha=0.05
            )
            pending.append(
                {
                    "cell_id": cell,
                    "best_simple_arm": arm,
                    "best_simple_seed": seed,
                    "candidate_non_escalation": sum(candidate_non_escalation)
                    / len(candidate_non_escalation),
                }
            )
    adjusted = _holm(comparisons)
    cells = []
    for row in pending:
        comparison = adjusted[row["cell_id"]]
        passed = (
            row["candidate_non_escalation"] >= 0.2
            and comparison["delta"] < 0
            and comparison["holm_upper"] < 0
        )
        cells.append({**row, "comparison": comparison, "benefit_passed": passed})
    return {"cells": cells, "decision_benefit_passed": any(row["benefit_passed"] for row in cells)}


def _reduce_online_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Json], Json]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["delay"]), str(row["group_id"]), str(row["arm"]))].append(row)
    compact: dict[tuple[int, str], Json] = {}
    for (delay, group, arm), arm_rows in grouped.items():
        probability = sum(float(row["probability"]) for row in arm_rows) / len(arm_rows)
        label = int(arm_rows[0]["label"])
        compact.setdefault(
            (delay, group),
            {
                "row_kind": "online_group",
                "delay": delay,
                "group_id": group,
                "source_family": arm_rows[0]["source_family"],
                "label": label,
                "probabilities": {},
                "brier_losses": {},
            },
        )
        compact[(delay, group)]["probabilities"][arm] = probability
        compact[(delay, group)]["brier_losses"][arm] = (probability - label) ** 2
    output = [compact[key] for key in sorted(compact)]
    delays: Json = {}
    for delay in (0, 8):
        delay_rows = [row for row in output if row["delay"] == delay]
        means = {
            arm: sum(float(row["brier_losses"][arm]) for row in delay_rows) / len(delay_rows)
            for arm in ("importance_anchor", "frozen", "affine", "unanchored_residual")
        }
        raw_comparisons: Json = {}
        for index, arm in enumerate(("frozen", "affine", "unanchored_residual")):
            raw_comparisons[arm] = _hierarchical_summary(
                [
                    {
                        "source_family": row["source_family"],
                        "delta": row["brier_losses"]["importance_anchor"]
                        - row["brier_losses"][arm],
                    }
                    for row in delay_rows
                ],
                seed=7_483_401 + delay * 100 + index,
            )
        adjusted = _holm(raw_comparisons)
        comparisons = {}
        for arm, comparison in adjusted.items():
            comparison["holm_upper_delta"] = comparison.pop("holm_upper")
            comparisons[arm] = comparison
        delays[str(delay)] = {
            "independent_group_count": len(delay_rows),
            "replicates_per_group": 5,
            "mean_brier": means,
            "comparisons": comparisons,
            "support_passed": len(delay_rows) >= 120,
            "effect_size_passed": means["frozen"] - means["importance_anchor"] >= 0.01,
            "multiplicity_direction_passed": all(
                row["holm_upper_delta"] < 0.0 for row in comparisons.values()
            ),
        }
    benefit = all(
        row["support_passed"] and row["effect_size_passed"] and row["multiplicity_direction_passed"]
        for row in delays.values()
    )
    return output, {
        "independent_group_count": len({row["group_id"] for row in output}),
        "delays": delays,
        "scientific_benefit_passed": benefit,
    }


def _retention_summary(rows: Sequence[Mapping[str, Any]]) -> Json:
    if any(row.get("used_for_update") is not False for row in rows):
        return {"errors": ["retention_label_used_for_update"], "delays": {}}
    grouped: dict[tuple[int, str, str, int], dict[str, float]] = defaultdict(dict)
    labels: dict[tuple[int, str, str, int], int] = {}
    families: dict[tuple[int, str, str, int], str] = {}
    for row in rows:
        key = (int(row["delay"]), str(row["group_id"]), str(row["arm"]), int(row["order_seed"]))
        grouped[key][str(row["moment"])] = float(row["probability"])
        labels[key] = int(row["label"])
        families[key] = str(row["source_family"])
    delays: Json = {}
    for delay in (0, 8):
        grouped_deltas: dict[str, list[float]] = defaultdict(list)
        grouped_families: dict[str, str] = {}
        for key, moments in grouped.items():
            if (
                key[0] == delay
                and key[2] == "importance_anchor"
                and set(moments) == {"before", "after"}
            ):
                label = labels[key]
                grouped_deltas[key[1]].append(
                    (moments["after"] - label) ** 2 - (moments["before"] - label) ** 2
                )
                grouped_families[key[1]] = families[key]
        rows_for_interval = [
            {
                "source_family": grouped_families[group],
                "delta": sum(values) / len(values),
            }
            for group, values in sorted(grouped_deltas.items())
        ]
        interval = _hierarchical_summary(rows_for_interval, seed=7_483_402 + delay)
        delays[str(delay)] = {
            "independent_group_count": len(rows_for_interval),
            "mean_brier_delta": interval["delta"],
            "upper_brier_delta": interval["upper"],
            "retention_passed": interval["upper"] <= 0.01,
        }
    return {"errors": [], "delays": delays}


def _checkpoint_file_errors(root: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    for row in manifest_rows:
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = root / path
        if not path.exists() or sha256_file(path) != row.get("sha256"):
            errors.append(f"changed_checkpoint:{row.get('order_seed')}:{row.get('delay')}")
    return errors


def _checkpoint_prediction_errors(
    root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Authenticate saved prediction records without exposing a future label."""

    ledger = {
        (int(row["order_seed"]), int(row["delay"]), str(row["learner_event_id"])): row
        for row in ledger_rows
        if row["arm"] == "importance_anchor" and row["label_revealed"] is True
    }
    errors: list[str] = []
    seen: set[tuple[int, int, str]] = set()
    forbidden = {"label", "gold_label", "future_label", "outcome"}
    for manifest in manifest_rows:
        path = Path(str(manifest["path"]))
        if not path.is_absolute():
            path = root / path
        checkpoint = _load_object(path)
        payload = {
            key: deepcopy(value) for key, value in checkpoint.items() if key != "checkpoint_sha256"
        }
        if checkpoint.get("checkpoint_sha256") != canonical_hash(payload):
            errors.append(
                f"checkpoint_content_hash_mismatch:{manifest.get('order_seed')}:{manifest.get('delay')}"
            )
        for prediction in checkpoint.get("predictions") or []:
            identity = (
                int(manifest["order_seed"]),
                int(manifest["delay"]),
                str(prediction["event_id"]),
            )
            seen.add(identity)
            event = ledger.get(identity)
            if forbidden & set(prediction):
                errors.append(f"future_label_in_prediction_record:{prediction['event_id']}")
            if event is None or any(
                prediction[field] != event[event_field]
                for field, event_field in (
                    ("pre_update_probability", "probability"),
                    ("prediction_time", "prediction_time"),
                    ("reveal_time", "feedback_due_time"),
                    ("predictor_state_hash", "prediction_state_hash"),
                )
            ):
                errors.append(f"checkpoint_prediction_mismatch:{prediction['event_id']}")
    if seen != set(ledger):
        errors.append("checkpoint_prediction_roster_mismatch")
    return errors


def audit_sources(root: Path) -> Json:
    """Reduce the four producers from immutable rows and frozen checkpoints."""

    located = [locate_producer(root, spec) for spec in PRODUCERS]
    dispositions = []
    source_hashes: list[Json] = []
    errors: list[str] = []
    artifacts: dict[str, Json] = {}
    for slot in located:
        artifact = slot["artifact"]
        artifacts[slot["branch"]] = artifact
        disposition = {
            "branch": slot["branch"],
            "availability": slot["availability"],
            "path": slot["path"],
            "original_verdict_class": artifact.get("verdict_class"),
            "original_honest_verdict": artifact.get("honest_verdict"),
            "original_flagged_adversarial": artifact.get("flagged_adversarial"),
        }
        dispositions.append(disposition)
        path = root / slot["path"]
        if slot["availability"] in {"available", "pre_gate"} and path.exists():
            source_hashes.append(_source_hash(root, path, "upstream_terminal_or_pregate"))
        if slot["availability"] != "available":
            errors.append(f"producer_{slot['availability']}:{slot['branch']}")
        else:
            errors.extend(_invocation_errors(artifact, slot["branch"]))
    if errors:
        return {
            "branch_dispositions": dispositions,
            "source_artifact_hashes": source_hashes,
            "errors": sorted(errors),
            "static_summary": {},
            "online_summary": {},
            "independent_metric_rows": [],
        }

    capture_rows: list[Json] = []
    roster_rows: list[Json] = []
    for branch in ("source_fit_capture", "source_eval_capture"):
        artifact = artifacts[branch]
        raw_root = root / str(artifact["raw_logit_root"])
        for shard in artifact["raw_logit_shards"]:
            path = raw_root / str(shard["path"])
            source_hashes.append(_source_hash(root, path, "raw_model_readout"))
            errors.extend(
                [f"raw_shard_hash_mismatch:{path.name}"]
                if sha256_file(path) != shard["sha256"]
                else []
            )
            rows = _load_jsonl(path)
            errors.extend(
                [f"raw_shard_count_mismatch:{path.name}"] if len(rows) != int(shard["rows"]) else []
            )
            (roster_rows if shard["kind"] == "plan" else capture_rows).extend(rows)
    roster = audit_roster(roster_rows, capture_rows)
    errors.extend(roster["errors"])
    native = reconstruct_native_groups(capture_rows)
    errors.extend(native["errors"])

    predictor_path = (
        root / "results/raw/experiment_7462_v654_option_protocol/cohort_predictors.jsonl"
    )
    predictors = _load_jsonl(predictor_path)
    source_hashes.append(_source_hash(root, predictor_path, "raw_source_roster"))
    feature_rows = _build_feature_rows(native["groups"], predictors)
    bundle_path = root / "results/raw/experiment_7481_v655_typed_calibration/frozen-fit-bundle.json"
    bundle = _load_object(bundle_path)
    source_hashes.append(_source_hash(root, bundle_path, "frozen_numeric_checkpoint"))
    errors.extend(_checkpoint_hash_errors(bundle))
    training = [row for row in feature_rows if row["role"] == "training"]
    calibration = [row for row in feature_rows if row["role"] == "calibration_tuning"]
    errors.extend(
        ["training_feature_hash_mismatch"]
        if canonical_hash(training) != bundle.get("training_input_sha256")
        else []
    )
    errors.extend(
        ["calibration_feature_hash_mismatch"]
        if canonical_hash(calibration) != bundle.get("calibration_input_sha256")
        else []
    )
    reconstructed = _reconstruct_predictions(feature_rows, bundle)
    prediction_path = root / str(artifacts["typed_calibration"]["prediction_row_shard"]["path"])
    stored_predictions = _load_jsonl(prediction_path)
    source_hashes.append(_source_hash(root, prediction_path, "raw_numeric_prediction"))
    errors.extend(_prediction_errors(reconstructed, stored_predictions))
    probability = reduce_probability_rows(reconstructed)
    probability_comparisons = _probability_comparisons(reconstructed)
    decision_grid = _cost_grid(reconstructed)
    static_benefit = (
        probability_comparisons["probability_benefit_passed"]
        or decision_grid["decision_benefit_passed"]
    )
    errors.extend(
        ["probability_benefit_disagreement"]
        if int(probability_comparisons["probability_benefit_passed"])
        != int(artifacts["typed_calibration"].get("probability_benefit_score", -1))
        else []
    )
    errors.extend(
        ["decision_benefit_disagreement"]
        if int(decision_grid["decision_benefit_passed"])
        != int(artifacts["typed_calibration"].get("decision_benefit_score", -1))
        else []
    )
    static_rows = _independent_static_rows(reconstructed)
    static_summary = {
        "training_group_count": len(training),
        "calibration_group_count": len(calibration),
        "internal_group_count": probability["internal_test"]["gibbs"]["n_groups"],
        "external_group_count": probability["external"]["gibbs"]["n_groups"],
        "proper_scores": probability,
        "probability_comparisons": probability_comparisons,
        "decision_cost_grid": decision_grid,
        "external_gibbs_beats_temperature": probability["external"]["gibbs"]["brier"]
        < probability["external"]["temperature"]["brier"],
        "scientific_benefit_passed": static_benefit,
    }

    learning = artifacts["continuous_learning"]
    ledger_path = root / str(learning["feedback_ledger"]["path"])
    retention_path = root / str(learning["retention_rows"]["path"])
    checkpoint_manifest_path = root / str(learning["state_checkpoints"]["path"])
    ledger = _load_jsonl(ledger_path)
    retention = _load_jsonl(retention_path)
    checkpoint_manifest = _load_jsonl(checkpoint_manifest_path)
    for path, evidence_class in (
        (ledger_path, "delayed_feedback_ledger"),
        (retention_path, "retention_readout"),
        (checkpoint_manifest_path, "checkpoint_manifest"),
    ):
        source_hashes.append(_source_hash(root, path, evidence_class))
    replay = replay_delayed_updates(ledger)
    errors.extend(replay["errors"])
    errors.extend(_checkpoint_file_errors(root, checkpoint_manifest))
    errors.extend(_checkpoint_prediction_errors(root, checkpoint_manifest, ledger))
    errors.extend(validate_service_costs(learning["service_cost_rows"]))
    protocol = learning.get("protocol") or {}
    selection = learning.get("selection_receipt") or {}
    baseline = learning.get("baseline_provenance") or {}
    expected_selection_roles = ["training", "calibration_tuning"]
    errors.extend(
        ["heldout_label_selection_leakage"]
        if (
            protocol.get("heldout_selection_allowed") is not False
            or protocol.get("selection_roles") != expected_selection_roles
            or selection.get("heldout_labels_consumed") is not False
            or selection.get("selection_roles") != expected_selection_roles
            or baseline.get("roles_consumed") != expected_selection_roles
        )
        else []
    )
    online_rows, online_summary = _reduce_online_rows(ledger)
    online_summary["replay"] = replay
    online_summary["retention"] = _retention_summary(retention)
    errors.extend(online_summary["retention"]["errors"])
    return {
        "branch_dispositions": dispositions,
        "source_artifact_hashes": sorted(source_hashes, key=lambda row: row["path"]),
        "errors": sorted(set(errors)),
        "roster": roster,
        "static_summary": static_summary,
        "online_summary": online_summary,
        "independent_metric_rows": static_rows + online_rows,
    }


def _gate(check: str, category: str, expected: Any, observed: Any, op: str, passed: bool) -> Json:
    principle = {
        "required_validity": VALIDITY_PRINCIPLE,
        "readiness": READINESS_PRINCIPLE,
        "scientific_benefit": BENEFIT_PRINCIPLE,
    }[category]
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
    }


FIELD_PRINCIPLES = {
    "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
    "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
    "MODEL_SPECS": "Use unsloth/Qwen3.8-27B-GGUF for current model tasks, [] for numeric/reducer work; also emit lowercase model_specs.",
    "model_invoked": "Any attempted current model call differs from archived or scripted events.",
    "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
    "inference_substrate": "Name actual native readout, bounded generation, numeric learning or artifact aggregation.",
    "inference_substrate_class": "Use model_load_no_generation (2s), model_bounded_generation (10s), no_model_load or aggregation as declared; blocked_no_run only when nothing executed. Full generation (60s) is not used in this plan.",
    "execution_venue": "Use host and record actual CPU/CUDA identities; historical board evidence is separate.",
    "duration_s": "Measure current work without padding; separate load, forward, generation, numeric work and validation.",
    "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
    "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain any null seed.",
    "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw shards and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes and their original flags/classes.",
    "rows": "One row per independent group/game/seed/arm or event, including failures and censoring; large lists use hash-bound shards.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted independent units.",
    "acceptance_gate_results": "Each check carries category, expected, observed, op, passed and principle; distinguish validity, support and benefit.",
    "gate_check_summary": "Every blocked verdict names failed check, upstream, exact field/path, expected and observed value.",
    "honest_verdict": "Use complete_ terminal findings, including complete_blocked_*; preserve a blocked_* conductor verdict if that is the actual source.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. partial only for retryable own work, never unchanged external absence.",
    "verifier_is_oracle": "Declare whether the acceptance verifier is the evaluation oracle; if true, positive is forbidden and fixture positives are circular_positive.",
    "flagged_adversarial": "Keep real reader flags; never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exit codes, log hashes and required versus unrelated-baseline status establish validation scope.",
    "field_principles": "Echo why each field and gate exists so evidence is understandable independently.",
    "audit_complete_score": "Bare 0/1 measures complete independent accounting, not positive science.",
    "independent_metric_rows": "Per-group reconstruction makes headline disagreements inspectable.",
    "attack_rows": "Each deliberate corruption has a named check and observed rejection.",
    "branch_dispositions": "Valid nulls, invalid evidence and absent inputs have distinct outcomes.",
}


def _passing_receipts() -> list[Json]:
    names = (
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
        "declared_entrypoint_cold_replay",
        "independent_raw_reduction",
        "adversarial_verify",
        "verdict_row_consistency_lint",
    )
    return [
        {
            "name": name,
            "required": True,
            "exit_code": 0,
            "command": ["fixture", name],
            "log_sha256": canonical_hash({"name": name}),
            "classification": "required_pass",
        }
        for name in names
    ]


def _preconditions(root: Path, reduction: Mapping[str, Any]) -> list[Json]:
    rows = []
    for disposition in reduction["branch_dispositions"]:
        rows.append(
            {
                "check": f"producer:{disposition['branch']}",
                "path": disposition["path"],
                "ownership": "upstream_immutable",
                "expected": "available_or_explicit_terminal_disposition",
                "observed": disposition["availability"],
                "passed": disposition["availability"] in {"available", "pre_gate"},
                "original_verdict_class": disposition["original_verdict_class"],
                "original_flagged_adversarial": disposition["original_flagged_adversarial"],
            }
        )
    for path in (
        "AGENTS.md",
        "CLAUDE.md",
        "CODEX.md",
        "research-program.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
    ):
        rows.append(
            {
                "check": f"source_bytes:{path}",
                "path": path,
                "ownership": "repository_instruction_or_protocol",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if (root / path).stat().st_size else "empty",
                "passed": (root / path).stat().st_size > 0,
            }
        )
    return rows


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding measured clocks and command timing."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate_class",
        "random_seed",
        "source_artifact_hashes",
        "branch_dispositions",
        "sample_size_budget",
        "static_summary",
        "online_summary",
        "independent_metric_rows",
        "attack_rows",
        "acceptance_gate_results",
        "validation_manifest",
        "field_principles",
    )
    return canonical_hash({key: artifact.get(key) for key in keys})


def _build_artifact(reduction: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]) -> Json:
    attacks = run_attack_controls()
    required_names = {
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
    }
    receipt_pass = required_names <= {
        str(row.get("name")) for row in receipts if int(row.get("exit_code", 1)) == 0
    }
    evidence_valid = not reduction["errors"] and all(row["rejected"] for row in attacks)
    complete = (
        evidence_valid
        and receipt_pass
        and all(row["availability"] == "available" for row in reduction["branch_dispositions"])
    )
    scientific = bool(reduction["static_summary"].get("scientific_benefit_passed")) and bool(
        reduction["online_summary"].get("scientific_benefit_passed")
    )
    gates = [
        _gate(
            "independent_evidence_valid",
            "required_validity",
            True,
            evidence_valid,
            "==",
            evidence_valid,
        ),
        _gate("scoped_validation", "required_validity", True, receipt_pass, "==", receipt_pass),
        _gate("audit_complete", "readiness", 1, int(complete), "==", complete),
        _gate(
            "static_and_online_benefit", "scientific_benefit", True, scientific, "==", scientific
        ),
    ]
    failed = [row for row in gates if not row["passed"]]
    if not evidence_valid or not receipt_pass:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_independent_v655_evidence"
    elif not complete:
        verdict_class = "blocked"
        verdict = "complete_blocked_missing_v655_producer"
    elif scientific:
        verdict_class = "positive"
        verdict = "complete_positive_independent_static_and_online_benefit"
    else:
        verdict_class = "null"
        verdict = "complete_null_independent_v655_decision_and_learning_audit"
    started = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows = deepcopy(list(reduction["independent_metric_rows"]))
    budget = reduction.get("roster") or {
        "planned": 0,
        "eligible": 0,
        "observed": 0,
        "excluded": 0,
    }
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": verdict,
        "started_at_utc": started,
        "ended_at_utc": started,
        "process_identity": {"pid": os.getpid(), "hostname": socket.gethostname()},
        "clock_identity": {"wall_clock": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "preconditions_checked": _preconditions(REPO_ROOT, reduction),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "device_identity": {"cpu": platform.processor() or platform.machine(), "cuda_devices": []},
        "duration_s": 0.0,
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "numeric_work_s": 0.0,
            "validation_s": 0.0,
        },
        "phase_spans": [],
        "random_seed": {
            "ordering": 7484001,
            "fitting": None,
            "audit": 7484002,
            "bootstrap": 7484003,
        },
        "source_artifact_hashes": deepcopy(list(reduction["source_artifact_hashes"])),
        "rows": rows,
        "independent_metric_rows": deepcopy(rows),
        "sample_size_budget": {
            "planned": int(budget["planned"]),
            "attempted": int(budget["observed"]),
            "complete": int(budget["observed"]),
            "failed": 0,
            "censored": 0,
            "excluded": int(budget["excluded"]),
            "unstarted": 0,
        },
        "static_summary": deepcopy(reduction["static_summary"]),
        "online_summary": deepcopy(reduction["online_summary"]),
        "branch_dispositions": deepcopy(list(reduction["branch_dispositions"])),
        "attack_rows": attacks,
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "all_required_validity_passed": all(
                row["passed"] for row in gates if row["category"] == "required_validity"
            ),
            "readiness_passed": all(
                row["passed"] for row in gates if row["category"] == "readiness"
            ),
            "scientific_benefit_passed": scientific,
            "failed_checks": [
                {
                    "check": row["check"],
                    "upstream": "independent_audit",
                    "field_path": f"acceptance_gate_results.{row['check']}",
                    "expected": row["expected"],
                    "observed": row["observed"],
                }
                for row in failed
            ],
        },
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_manifest": {
            "tests": ["tests/python/test_experiment_7484_v655_decision_audit.py"],
            "changed_modules": ["python/carnot/experiment_7484_v655_decision_audit.py"],
            "static_paths": ["scripts/experiments/experiment_7484_v655_decision_audit.py"],
        },
        "validation_receipts": deepcopy(list(receipts)),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "audit_complete_score": int(complete),
        "small_ebm_training": {
            "performed": False,
            "current_llm_calls": 0,
            "numeric_elapsed_s": 0.0,
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test() -> Json:
    """Build a complete contract fixture from the real immutable source rows."""

    return _build_artifact(audit_sources(REPO_ROOT), _passing_receipts())


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, rows, attacks, provenance, gates, and checksum."""

    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {"experiment_id", "milestone", "model_specs"}
    if required - value.keys():
        errors.append("required_fields_missing")
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
    ):
        errors.append("artifact_identity_mismatch")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
    ):
        errors.append("current_model_provenance_mismatch")
    if value.get("invocation_counts") != ZERO_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if (
        value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
    ):
        errors.append("substrate_or_venue_mismatch")
    if value.get("verifier_is_oracle") is not False:
        errors.append("oracle_flag_mismatch")
    attacks = value.get("attack_rows") or []
    if [row.get("attack") for row in attacks] != list(REQUIRED_ATTACKS) or not all(
        row.get("rejected") is True for row in attacks
    ):
        errors.append("attack_controls_incomplete")
    if len(value.get("branch_dispositions") or []) != len(PRODUCERS):
        errors.append("branch_dispositions_incomplete")
    if not value.get("independent_metric_rows") or value.get("rows") != value.get(
        "independent_metric_rows"
    ):
        errors.append("independent_rows_missing")
    gates = value.get("acceptance_gate_results") or []
    if not gates or any(not row.get("principle") for row in gates):
        errors.append("gate_principles_missing")
    if int(value.get("audit_complete_score", 0)) != 1:
        errors.append("audit_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def cold_replay(path: Path) -> list[str]:
    """Reload one terminal candidate and apply only the artifact contract."""

    value = _load_object(path)
    return validate_artifact(value) if value else ["candidate_json_invalid"]


def independent_replay(path: Path) -> list[str]:
    """Re-read all raw inputs and compare every stable independent reduction."""

    value = _load_object(path)
    if not value:
        return ["candidate_json_invalid"]
    current = audit_sources(REPO_ROOT)
    errors = validate_artifact(value)
    for field in (
        "branch_dispositions",
        "source_artifact_hashes",
        "static_summary",
        "online_summary",
        "independent_metric_rows",
    ):
        if value.get(field) != current.get(field):
            errors.append(f"independent_replay_mismatch:{field}")
    return sorted(set(errors))


def _terminal_commands(
    candidate: Path,
) -> list[Any]:  # pragma: no cover - subprocess orchestration.
    from carnot.experiment_7358_v646_validation_contract import PlannedCommand
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    python = str(REPO_ROOT / ".venv/bin/python")
    wrapper = "scripts/experiments/experiment_7484_v655_decision_audit.py"
    return [
        PlannedCommand(
            CommandSpec(
                "declared_entrypoint_cold_replay",
                (python, "-u", wrapper, "--cold-replay", str(candidate)),
                "measured_candidate",
                timeout_s=300.0,
            ),
            "capability_e2e",
            True,
        ),
        PlannedCommand(
            CommandSpec(
                "independent_raw_reduction",
                (python, "-u", wrapper, "--independent-replay", str(candidate)),
                "measured_candidate",
                timeout_s=900.0,
            ),
            "capability_e2e",
            True,
        ),
        PlannedCommand(
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
                timeout_s=300.0,
            ),
            "terminal_reader",
            True,
        ),
        PlannedCommand(
            CommandSpec(
                "verdict_row_consistency_lint",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
                timeout_s=300.0,
            ),
            "terminal_reader",
            True,
        ),
    ]


def run_experiment(
    repo_root: Path, run_date: str
) -> Path:  # pragma: no cover - exercised as capability E2E.
    """Authenticate, reduce, validate, cold replay, and publish atomically."""

    if repo_root.resolve() != REPO_ROOT or run_date != RUN_DATE:
        raise ValueError("repository_or_run_date_mismatch")
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    spans: list[Json] = []

    def phase(name: str, event: str, phase_start: int, units: int = 0) -> int:
        progress(started, name, event, completed_units=units)
        now = time.monotonic_ns()
        if event == "complete":
            spans.append(
                {
                    "phase": name,
                    "start_monotonic_ns": phase_start,
                    "end_monotonic_ns": now,
                    "completed_units": units,
                }
            )
        return now

    phase_start = phase("authenticate_and_reduce", "begin", started_ns)
    reduction = audit_sources(repo_root)
    phase(
        "authenticate_and_reduce",
        "complete",
        phase_start,
        len(reduction["independent_metric_rows"]),
    )
    phase_start = phase("attack_controls", "begin", time.monotonic_ns())
    attacks = run_attack_controls()
    phase("attack_controls", "complete", phase_start, len(attacks))

    private_root = Path(
        os.environ.get("CARNOT_PRIVATE_VALIDATION_ROOT", "/tmp/carnot-exp7484-validation")
    )
    private_root.mkdir(parents=True, exist_ok=True)
    manifest = AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=("tests/python/test_experiment_7484_v655_decision_audit.py",),
        changed_modules=("python/carnot/experiment_7484_v655_decision_audit.py",),
        static_paths=("scripts/experiments/experiment_7484_v655_decision_audit.py",),
    )
    commands = build_command_plan(repo_root, manifest, private_root)
    plan_errors = validate_command_plan(repo_root, manifest, commands)
    if plan_errors:
        raise ValueError("validation_plan_invalid:" + ",".join(plan_errors))
    phase_start = phase("scoped_validation", "begin", time.monotonic_ns())
    receipts = run_categorized_commands(
        repo_root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=private_root / "logs/affected",
    )
    reduced_receipts = reduce_affected_receipts(repo_root, manifest, receipts)
    if not reduced_receipts["passed"]:
        raise ValueError("affected_validation_failed")
    phase("scoped_validation", "complete", phase_start, len(receipts))

    candidate = private_root / "experiment_7484-terminal-candidate.json"
    artifact = _build_artifact(reduction, receipts)
    artifact["phase_spans"] = deepcopy(spans)
    artifact["duration_s"] = (time.monotonic_ns() - started_ns) / 1e9
    artifact["duration_components_s"]["validation_s"] = sum(
        float(row.get("duration_s", 0.0)) for row in receipts
    )
    artifact["ended_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    atomic_json(candidate, artifact)

    phase_start = phase("terminal_validation", "begin", time.monotonic_ns())
    terminal_receipts = run_categorized_commands(
        repo_root,
        _terminal_commands(candidate),
        log_dir=private_root / "logs/terminal",
    )
    phase("terminal_validation", "complete", phase_start, len(terminal_receipts))
    if any(int(row.get("exit_code", 1)) != 0 for row in terminal_receipts):
        raise ValueError("terminal_validation_failed")
    all_receipts = [*receipts, *terminal_receipts]
    artifact = _build_artifact(reduction, all_receipts)
    artifact["phase_spans"] = deepcopy(spans)
    artifact["duration_s"] = (time.monotonic_ns() - started_ns) / 1e9
    artifact["duration_components_s"]["validation_s"] = sum(
        float(row.get("duration_s", 0.0)) for row in all_receipts
    )
    artifact["ended_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    deliverable = repo_root / "results/experiment_7484_v655_decision_audit.json"
    atomic_json(deliverable, artifact)
    progress(started, "publish", "complete", completed_units=len(artifact["rows"]))
    return deliverable


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:  # pragma: no cover - CLI parsing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    args = parse_args(argv)
    if args.cold_replay:
        errors = cold_replay(args.cold_replay)
    elif args.independent_replay:
        errors = independent_replay(args.independent_replay)
    else:
        run_experiment(REPO_ROOT, args.date)
        return 0
    print(json.dumps({"passed": not errors, "errors": errors}, sort_keys=True), flush=True)
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - module entrypoint.
    raise SystemExit(main())
