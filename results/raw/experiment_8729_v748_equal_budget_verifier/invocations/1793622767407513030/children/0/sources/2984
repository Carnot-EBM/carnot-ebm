"""Fit typed source decisions from authenticated V655 native readouts.

This module trains only compact numeric heads. It never loads or updates Qwen.
Held-out labels open only after a child process has written and hashed the
complete fit bundle.

Spec: REQ-AUTO-7481 and SCENARIO-AUTO-7481-01 through -07.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7412_v650_source_features import (
    extract_feature_row,
    fit_gibbs_head,
    fit_logistic_control,
    gibbs_energy,
    initial_gibbs_checkpoint,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7481-typed-calibration"
SCHEMA = "carnot.exp7481.v655.typed_calibration.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7481_v655_typed_calibration.json")
RAW_DIR = Path("results/raw/experiment_7481_v655_typed_calibration")
MODULE_PATH = Path("python/carnot/experiment_7481_v655_typed_calibration.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7481_v655_typed_calibration.py")
TEST_PATH = Path("tests/python/test_experiment_7481_v655_typed_calibration.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/autoresearch/spec.md"
FIT_ARTIFACT = Path("results/experiment_7479_v655_source_fit_capture.json")
EVAL_ARTIFACT = Path("results/experiment_7480_v655_source_eval_capture.json")
HISTORICAL_CAPSTONE = Path("results/experiment_7474_v654_capstone.json")
HISTORICAL_NULL = Path("results/experiment_7439_v652_certified_decisions.json")
COHORT_PREDICTORS = Path("results/raw/experiment_7462_v654_option_protocol/cohort_predictors.jsonl")
COHORT_EVALUATORS = Path("results/raw/experiment_7462_v654_option_protocol/cohort_evaluators.jsonl")
FIT_RAW_DIR = Path("results/raw/experiment_7479_v655_source_fit_capture")
EVAL_RAW_DIR = Path("results/raw/experiment_7480_v655_source_eval_capture")
PREDICTION_SHARD = RAW_DIR / "prediction-rows.jsonl"
FIT_BUNDLE_PATH = RAW_DIR / "frozen-fit-bundle.json"
TERMINAL_CANDIDATE = RAW_DIR / "measured_terminal_candidate.json"

TRAINING_SEEDS = (655_101, 655_102, 655_103, 655_104, 655_105)
TEMPERATURE_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
FALSE_ACCEPT_COSTS = (1.0, 5.0, 20.0)
ESCALATION_COSTS = (0.1, 0.5, 1.0)
FALSE_REJECT_COST = 1.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_551_481
MAX_NUMERIC_SECONDS = 1_200.0
FULL_FEATURE_NAMES = (
    "native_unsupported_log_odds",
    "falsifiability_score",
    "numeric_novelty_with_context",
    "normalized_content_token_overlap",
    "max_answer_source_sentence_overlap",
    "missing_or_empty_source",
)
GIBBS_ARMS = (
    "gibbs",
    "verifier_only_gibbs",
    "shuffled_label_gibbs",
    "source_removal_gibbs",
)
SIMPLE_ARMS = ("raw_readout", "temperature", "logistic")
ALL_ARMS = (*SIMPLE_ARMS, *GIBBS_ARMS)
ROLE_MINIMUMS = {"training": 150, "calibration_tuning": 40, "internal_test": 40, "external": 60}

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "calibration_complete_score",
    "probability_benefit_score",
    "decision_benefit_score",
    "small_ebm_training",
    "frozen_policy_manifest",
)


def sigmoid(value: Any) -> np.ndarray:
    """Return a stable logistic transform for scalar or array input."""

    array = np.asarray(value, dtype=np.float64)
    positive = array >= 0
    out = np.empty_like(array)
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def aggregate_native_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Remap two option orders, then average unsupported-content log odds.

    Source-shuffled rows are returned as controls with no factual label. They
    never enter the benefit-eligible output.
    """

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("eligible") is True and row.get("disposition") == "complete":
            group = str(row.get("source_group_id") or row.get("group_id") or "")
            grouped[(group, str(row.get("role") or ""), str(row.get("arm") or ""))].append(row)
    main: list[JsonDict] = []
    controls: list[JsonDict] = []
    expected_orders = {
        ("supported", "contains_unsupported"),
        ("contains_unsupported", "supported"),
    }
    for (group_id, role, arm), group_rows in sorted(grouped.items()):
        orders = {tuple(str(item) for item in row.get("option_order") or []) for row in group_rows}
        if len(group_rows) != 2 or orders != expected_orders:
            raise ValueError(f"option_order_pair_invalid:{group_id}:{arm}")
        odds: list[float] = []
        labels: set[int | None] = set()
        for row in group_rows:
            logits = row.get("raw_logits_by_option_id")
            if not isinstance(logits, Mapping):
                raise ValueError(f"native_logits_missing:{group_id}:{arm}")
            supported = float(logits.get("supported"))
            unsupported = float(logits.get("contains_unsupported"))
            if not math.isfinite(supported) or not math.isfinite(unsupported):
                raise ValueError(f"native_logits_nonfinite:{group_id}:{arm}")
            odds.append(unsupported - supported)
            raw_label = row.get("gold_label")
            labels.add(int(raw_label) if raw_label in (0, 1) else None)
        if len(labels) != 1:
            raise ValueError(f"gold_label_disagreement:{group_id}:{arm}")
        source_label = next(iter(labels))
        reduced = {
            "group_id": group_id,
            "group_hash": str(group_rows[0].get("group_hash") or ""),
            "role": role,
            "source_arm": arm,
            "native_log_odds": float(np.mean(odds)),
            "label": None if source_label is None else 1 - source_label,
            "order_count": 2,
            "benefit_eligible": arm == "full_source_response" and source_label is not None,
        }
        (main if reduced["benefit_eligible"] else controls).append(reduced)
    return main, controls


def project_predictor_features(
    predictor: Mapping[str, Any], *, native_log_odds: float
) -> tuple[dict[str, list[float]], JsonDict]:
    """Project one predictor row to fixed full and falsifying feature views."""

    if not math.isfinite(native_log_odds):
        raise ValueError("native_log_odds_nonfinite")
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
    views = {
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
    denied = sorted(
        set(predictor)
        & {
            "gold_span",
            "source_id",
            "response_generator",
            "response_generator_identity",
            "future_label",
            "outcome_note",
            "annotation_notes",
        }
    )
    return views, {
        "input_fields": list(FULL_FEATURE_NAMES),
        "labels_consumed": False,
        "identity_consumed": False,
        "identity_used_for_join_only": True,
        "denied_fields_present": denied,
    }


def _matrix(rows: Sequence[Mapping[str, Any]], view: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate one fixed feature view and its binary labels."""

    vectors: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        features = row.get("features")
        if not isinstance(features, Mapping) or view not in features:
            raise ValueError("feature_view_missing")
        vector = np.asarray(features[view], dtype=np.float64)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("feature_shape_invalid")
        label = row.get("label")
        if label not in (0, 1):
            raise ValueError("binary_label_invalid")
        vectors.append(vector.tolist())
        labels.append(int(label))
    matrix = np.asarray(vectors, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if matrix.shape != (len(rows), 6) or set(targets.tolist()) != {0, 1}:
        raise ValueError("training_support_invalid")
    return matrix, targets


def select_temperature(logits: Any, labels: Any) -> float:
    """Choose one scalar temperature by calibration log loss only."""

    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.shape != targets.shape or values.ndim != 1 or len(values) == 0:
        raise ValueError("temperature_input_invalid")
    losses = []
    for temperature in TEMPERATURE_GRID:
        probabilities = np.clip(sigmoid(values / temperature), 1e-9, 1.0 - 1e-9)
        loss = float(
            np.mean(-(targets * np.log(probabilities) + (1.0 - targets) * np.log1p(-probabilities)))
        )
        losses.append((loss, temperature))
    return float(min(losses)[1])


def _gibbs_logits(checkpoint: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Evaluate one frozen Gibbs checkpoint as unsupported-content logits."""

    return np.asarray([gibbs_energy(checkpoint, row) for row in matrix], dtype=np.float64)


def _logistic_logits(checkpoint: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Evaluate one frozen regularized logistic checkpoint."""

    coefficient = np.asarray(checkpoint["coef"], dtype=np.float64)
    return matrix @ coefficient + float(checkpoint["bias"])


def numeric_bundle_hash(bundle: Mapping[str, Any]) -> str:
    """Hash a frozen fit bundle without its self-referential hash field."""

    payload = {key: deepcopy(value) for key, value in bundle.items() if key != "bundle_sha256"}
    return canonical_hash(payload)


def fit_numeric_bundle(
    training: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    steps: int = 500,
) -> JsonDict:
    """Fit five seeded heads per learned arm without any held-out input."""

    if any(row.get("role") != "training" for row in training):
        raise ValueError("training_role_invalid")
    if any(row.get("role") != "calibration_tuning" for row in calibration):
        raise ValueError("calibration_role_invalid")
    full_train, train_labels = _matrix(training, "full")
    full_calibration, calibration_labels = _matrix(calibration, "full")
    verifier_train, _ = _matrix(training, "verifier_only")
    verifier_calibration, _ = _matrix(calibration, "verifier_only")
    removal_train, _ = _matrix(training, "source_removal")
    removal_calibration, _ = _matrix(calibration, "source_removal")
    raw_calibration = np.asarray(
        [float(row["native_log_odds"]) for row in calibration], dtype=np.float64
    )
    started = time.monotonic()
    heads: dict[str, list[JsonDict]] = {
        "gibbs": [],
        "logistic": [],
        "verifier_only_gibbs": [],
        "shuffled_label_gibbs": [],
        "source_removal_gibbs": [],
    }
    checkpoints: list[JsonDict] = []
    for seed in TRAINING_SEEDS:
        if time.monotonic() - started > MAX_NUMERIC_SECONDS:  # pragma: no cover - wall guard.
            raise TimeoutError("numeric_fit_budget_exceeded")
        shuffled = np.random.default_rng(seed).permutation(train_labels)
        definitions = (
            ("gibbs", full_train, train_labels, full_calibration),
            ("verifier_only_gibbs", verifier_train, train_labels, verifier_calibration),
            ("shuffled_label_gibbs", full_train, shuffled, full_calibration),
            ("source_removal_gibbs", removal_train, train_labels, removal_calibration),
        )
        for arm, fit_matrix, fit_labels, tuning_matrix in definitions:
            fitted = fit_gibbs_head(fit_matrix, fit_labels, seed=seed, steps=steps)
            logits = _gibbs_logits(fitted["checkpoint"], tuning_matrix)
            temperature = select_temperature(logits, calibration_labels)
            state = {
                "seed": seed,
                "checkpoint": fitted["checkpoint"],
                "checkpoint_sha256": fitted["checkpoint_sha256"],
                "initial_checkpoint_sha256": canonical_hash(
                    initial_gibbs_checkpoint(seed=seed, input_dim=6)
                ),
                "calibration_temperature": temperature,
                "architecture": {"input_dim": 6, "hidden_dims": [4], "output_dim": 1},
                "optimizer": {
                    "name": "adam",
                    "objective": fitted["objective"],
                    "learning_rate": fitted["learning_rate"],
                    "l2": fitted["l2"],
                    "steps": fitted["update_count"],
                },
                "loss_start": fitted["loss_curve"][0]["loss"],
                "loss_end": fitted["loss_curve"][-1]["loss"],
            }
            heads[arm].append(state)
            checkpoints.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_sha256": state["checkpoint_sha256"],
                    "initial_checkpoint_sha256": state["initial_checkpoint_sha256"],
                }
            )
        logistic = fit_logistic_control(full_train, train_labels, seed=seed, steps=steps)
        logistic_state = {
            "seed": seed,
            "checkpoint": logistic["checkpoint"],
            "checkpoint_sha256": logistic["checkpoint_sha256"],
            "initial_checkpoint_sha256": canonical_hash(
                {
                    "coef": np.random.default_rng(seed).normal(0.0, 0.01, size=6).tolist(),
                    "bias": 0.0,
                }
            ),
            "calibration_temperature": 1.0,
            "architecture": {"input_dim": 6, "output_dim": 1},
            "optimizer": {
                "name": "adam",
                "objective": logistic["objective"],
                "learning_rate": logistic["learning_rate"],
                "l2": logistic["l2"],
                "steps": logistic["update_count"],
            },
            "loss_start": logistic["loss_curve"][0]["loss"],
            "loss_end": logistic["loss_curve"][-1]["loss"],
        }
        heads["logistic"].append(logistic_state)
        checkpoints.append(
            {
                "arm": "logistic",
                "seed": seed,
                "checkpoint_sha256": logistic_state["checkpoint_sha256"],
                "initial_checkpoint_sha256": logistic_state["initial_checkpoint_sha256"],
            }
        )
    initial_states = []
    raw_temperature = select_temperature(raw_calibration, calibration_labels)
    for seed in TRAINING_SEEDS:
        residual = initial_gibbs_checkpoint(seed=seed, input_dim=6)
        initial_states.append(
            {
                "seed": seed,
                "base_arm": "temperature",
                "base_temperature": raw_temperature,
                "residual_checkpoint": residual,
                "residual_checkpoint_sha256": canonical_hash(residual),
            }
        )
    bundle: JsonDict = {
        "schema": "carnot.exp7481.frozen_fit_bundle.v1",
        "training_seeds": list(TRAINING_SEEDS),
        "roles_consumed": ["training", "calibration_tuning"],
        "heldout_labels_consumed": False,
        "feature_names": list(FULL_FEATURE_NAMES),
        "training_group_count": len(training),
        "calibration_group_count": len(calibration),
        "training_input_sha256": canonical_hash(training),
        "calibration_input_sha256": canonical_hash(calibration),
        "raw_temperature": raw_temperature,
        "heads": heads,
        "checkpoint_manifest": checkpoints,
        "initial_states_for_exp7483": initial_states,
    }
    bundle["bundle_sha256"] = numeric_bundle_hash(bundle)
    return bundle


def _state_probabilities(
    arm: str, state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Apply one frozen state to its declared feature view."""

    view = {
        "gibbs": "full",
        "logistic": "full",
        "verifier_only_gibbs": "verifier_only",
        "shuffled_label_gibbs": "full",
        "source_removal_gibbs": "source_removal",
    }[arm]
    matrix = np.asarray([row["features"][view] for row in rows], dtype=np.float64)
    logits = (
        _logistic_logits(state["checkpoint"], matrix)
        if arm == "logistic"
        else _gibbs_logits(state["checkpoint"], matrix)
    )
    return sigmoid(logits / float(state["calibration_temperature"]))


def score_numeric_bundle(
    bundle: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Emit one probability row per group, arm, and fitted seed."""

    output: list[JsonDict] = []
    raw_logits = np.asarray([float(row["native_log_odds"]) for row in rows], dtype=np.float64)
    raw_probabilities = sigmoid(raw_logits)
    temperature_probabilities = sigmoid(raw_logits / float(bundle["raw_temperature"]))
    for row, raw, calibrated in zip(
        rows, raw_probabilities, temperature_probabilities, strict=True
    ):
        common = {
            "group_id": row["group_id"],
            "role": row["role"],
            "label": int(row["label"]),
            "failed": False,
        }
        output.append({**common, "arm": "raw_readout", "seed": None, "probability": float(raw)})
        output.append(
            {**common, "arm": "temperature", "seed": None, "probability": float(calibrated)}
        )
    heads = bundle.get("heads")
    if not isinstance(heads, Mapping):
        raise ValueError("frozen_heads_missing")
    for arm in ("logistic", *GIBBS_ARMS):
        states = heads.get(arm)
        if not isinstance(states, list) or len(states) != 5:
            raise ValueError(f"frozen_head_count_invalid:{arm}")
        for state in states:
            probabilities = _state_probabilities(arm, state, rows)
            for row, probability in zip(rows, probabilities, strict=True):
                output.append(
                    {
                        "group_id": row["group_id"],
                        "role": row["role"],
                        "arm": arm,
                        "seed": int(state["seed"]),
                        "label": int(row["label"]),
                        "probability": float(probability),
                        "failed": False,
                    }
                )
    return output


def _average_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Average repeated seeds inside each source group before inference."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("failed") is not True:
            grouped[(str(row.get("role") or ""), str(row["group_id"]), str(row["arm"]))].append(row)
    averaged = []
    for (role, group_id, arm), group_rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in group_rows}
        if len(labels) != 1:
            raise ValueError(f"prediction_label_disagreement:{group_id}:{arm}")
        averaged.append(
            {
                "role": role,
                "group_id": group_id,
                "arm": arm,
                "label": next(iter(labels)),
                "probability": float(np.mean([float(row["probability"]) for row in group_rows])),
                "seed_rows": len(group_rows),
            }
        )
    return averaged


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute tie-aware binary AUROC without a third-party estimator."""

    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    wins = sum(
        float(np.sum(value > negative)) + 0.5 * float(np.sum(value == negative))
        for value in positive
    )
    return float(wins / (len(positive) * len(negative)))


def probability_metrics(labels: Any, probabilities: Any) -> JsonDict:
    """Report proper scores, discrimination, calibration, and class support."""

    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != scores.shape or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("probability_metric_input_invalid")
    clipped = np.clip(scores, 1e-9, 1.0 - 1e-9)
    brier = float(np.mean((clipped - targets) ** 2))
    log_loss = float(np.mean(-(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))))
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        selected = (clipped >= lower) & ((clipped <= upper) if index == 9 else (clipped < upper))
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(clipped[selected])) - float(np.mean(targets[selected]))
            )
    return {
        "n_groups": len(targets),
        "brier": brier,
        "log_loss": log_loss,
        "auroc": _binary_auroc(targets, clipped),
        "ece": ece,
        "class_support": {
            "supported": int(np.sum(targets == 0)),
            "contains_unsupported": int(np.sum(targets == 1)),
        },
        "coverage": 1.0,
        "failures": 0,
    }


def group_loss_deltas(
    rows: Sequence[Mapping[str, Any]], *, candidate: str, control: str, loss: str
) -> list[JsonDict]:
    """Build paired source-group losses after averaging each arm's seed rows."""

    if loss not in {"brier", "log_loss"}:
        raise ValueError("loss_name_invalid")
    averaged = _average_seed_rows(rows)
    indexed = {(row["role"], row["group_id"], row["arm"]): row for row in averaged}
    keys = sorted(
        (role, group_id)
        for role, group_id, arm in indexed
        if arm == candidate and (role, group_id, control) in indexed
    )
    deltas = []
    for role, group_id in keys:
        candidate_row = indexed[(role, group_id, candidate)]
        control_row = indexed[(role, group_id, control)]
        label = int(candidate_row["label"])

        def loss_value(probability: float) -> float:
            clipped = min(max(probability, 1e-9), 1.0 - 1e-9)
            if loss == "brier":
                return (clipped - label) ** 2
            return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))

        candidate_loss = loss_value(float(candidate_row["probability"]))
        control_loss = loss_value(float(control_row["probability"]))
        deltas.append(
            {
                "role": role,
                "group_id": group_id,
                "candidate_loss": candidate_loss,
                "control_loss": control_loss,
                "delta": candidate_loss - control_loss,
            }
        )
    return deltas


def paired_bootstrap(deltas: Sequence[Mapping[str, Any]], *, draws: int, seed: int) -> JsonDict:
    """Resample whole source groups and retain one-sided inference operands."""

    values = np.asarray([float(row["delta"]) for row in deltas], dtype=np.float64)
    if len(values) == 0 or draws <= 0:
        raise ValueError("bootstrap_input_invalid")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    point = float(np.mean(values))
    return {
        "group_count": len(values),
        "draws": draws,
        "seed": seed,
        "delta": point,
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "one_sided_p": float((1 + np.sum(sampled >= 0.0)) / (draws + 1)),
        "bootstrap_means": sampled.tolist(),
    }


def holm_upper_bounds(
    comparisons: Mapping[str, Mapping[str, Any]], *, alpha: float
) -> dict[str, JsonDict]:
    """Apply step-down Holm thresholds and matching one-sided upper bounds."""

    ordered = sorted(comparisons, key=lambda name: (float(comparisons[name]["one_sided_p"]), name))
    family_size = len(ordered)
    cumulative_adjusted = 0.0
    output: dict[str, JsonDict] = {}
    for rank, name in enumerate(ordered, start=1):
        comparison = comparisons[name]
        threshold = alpha / (family_size - rank + 1)
        adjusted = min(1.0, (family_size - rank + 1) * float(comparison["one_sided_p"]))
        cumulative_adjusted = max(cumulative_adjusted, adjusted)
        samples = np.asarray(comparison["bootstrap_means"], dtype=np.float64)
        output[name] = {
            key: deepcopy(value) for key, value in comparison.items() if key != "bootstrap_means"
        }
        output[name].update(
            {
                "holm_rank": rank,
                "holm_alpha": threshold,
                "holm_adjusted_p": cumulative_adjusted,
                "holm_upper": float(np.quantile(samples, 1.0 - threshold)),
            }
        )
    return output


def typed_action(probability: float, policy: Mapping[str, Any]) -> str:
    """Map unsupported-content risk to accept, reject, or escalate."""

    if probability <= float(policy["accept_max"]):
        return "accept"
    if probability >= float(policy["reject_min"]):
        return "reject"
    return "escalate"


def _action_cost(
    action: str,
    label: int,
    *,
    false_accept_cost: float,
    false_reject_cost: float,
    escalation_cost: float,
) -> float:
    """Charge only registered false decisions and escalations."""

    if action == "accept":
        return false_accept_cost if label == 1 else 0.0
    if action == "reject":
        return false_reject_cost if label == 0 else 0.0
    return escalation_cost


def select_cost_policy(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    false_accept_cost: float,
    false_reject_cost: float,
    escalation_cost: float,
) -> JsonDict:
    """Select stable thresholds using calibration rows only."""

    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("policy_input_invalid")
    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    best: tuple[float, float, float, float] | None = None
    for accept_max in candidates:
        for reject_min in candidates:
            if accept_max > reject_min:
                continue
            policy = {"accept_max": accept_max, "reject_min": reject_min}
            actions = [typed_action(float(value), policy) for value in probabilities]
            costs = [
                _action_cost(
                    action,
                    int(label),
                    false_accept_cost=false_accept_cost,
                    false_reject_cost=false_reject_cost,
                    escalation_cost=escalation_cost,
                )
                for action, label in zip(actions, labels, strict=True)
            ]
            non_escalation = float(np.mean([action != "escalate" for action in actions]))
            candidate = (float(np.mean(costs)), -non_escalation, accept_max, reject_min)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {
        "accept_max": best[2],
        "reject_min": best[3],
        "calibration_mean_cost": best[0],
        "calibration_non_escalation": -best[1],
        "selection_role": "calibration_tuning",
        "frozen": True,
    }


def policy_cost_rows(
    probabilities: Sequence[float],
    labels: Sequence[int],
    policy: Mapping[str, Any],
    *,
    false_accept_cost: float,
    false_reject_cost: float,
    escalation_cost: float,
) -> list[JsonDict]:
    """Return one typed action and cost per independent group."""

    output = []
    for probability, label in zip(probabilities, labels, strict=True):
        action = typed_action(float(probability), policy)
        output.append(
            {
                "action": action,
                "cost": _action_cost(
                    action,
                    int(label),
                    false_accept_cost=false_accept_cost,
                    false_reject_cost=false_reject_cost,
                    escalation_cost=escalation_cost,
                ),
                "non_escalated": action != "escalate",
            }
        )
    return output


def _arm_seed_rows(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> dict[int | None, list[Mapping[str, Any]]]:
    """Group one arm by its real seed without inventing replicated controls."""

    output: dict[int | None, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("arm") == arm:
            seed = int(row["seed"]) if row.get("seed") is not None else None
            output[seed].append(row)
    return dict(output)


def evaluate_cost_grid(
    calibration_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
) -> JsonDict:
    """Select policies on calibration and compare all nine external cost cells."""

    candidate_by_seed = _arm_seed_rows(calibration_rows, "gibbs")
    evaluation_candidate = _arm_seed_rows(evaluation_rows, "gibbs")
    comparisons: dict[str, JsonDict] = {}
    pending: list[JsonDict] = []
    for false_accept in FALSE_ACCEPT_COSTS:
        for escalation in ESCALATION_COSTS:
            cell_id = f"fa={false_accept:g}|fr=1|esc={escalation:g}"
            candidate_policies: dict[str, JsonDict] = {}
            candidate_costs: dict[str, list[float]] = defaultdict(list)
            candidate_non_escalation: dict[str, list[float]] = defaultdict(list)
            for fit_seed, rows in candidate_by_seed.items():
                ordered = sorted(rows, key=lambda row: str(row["group_id"]))
                policy = select_cost_policy(
                    [float(row["probability"]) for row in ordered],
                    [int(row["label"]) for row in ordered],
                    false_accept_cost=false_accept,
                    false_reject_cost=FALSE_REJECT_COST,
                    escalation_cost=escalation,
                )
                candidate_policies[str(fit_seed)] = policy
                eval_by_group = {
                    str(row["group_id"]): row for row in evaluation_candidate.get(fit_seed, [])
                }
                for group_id, row in eval_by_group.items():
                    cost_row = policy_cost_rows(
                        [float(row["probability"])],
                        [int(row["label"])],
                        policy,
                        false_accept_cost=false_accept,
                        false_reject_cost=FALSE_REJECT_COST,
                        escalation_cost=escalation,
                    )[0]
                    candidate_costs[group_id].append(float(cost_row["cost"]))
                    candidate_non_escalation[group_id].append(float(cost_row["non_escalated"]))
            simple_candidates = []
            for simple_arm in SIMPLE_ARMS:
                for fit_seed, rows in _arm_seed_rows(calibration_rows, simple_arm).items():
                    ordered = sorted(rows, key=lambda row: str(row["group_id"]))
                    policy = select_cost_policy(
                        [float(row["probability"]) for row in ordered],
                        [int(row["label"]) for row in ordered],
                        false_accept_cost=false_accept,
                        false_reject_cost=FALSE_REJECT_COST,
                        escalation_cost=escalation,
                    )
                    simple_candidates.append(
                        (
                            float(policy["calibration_mean_cost"]),
                            simple_arm,
                            fit_seed,
                            policy,
                        )
                    )
            _, simple_arm, simple_seed, simple_policy = min(
                simple_candidates,
                key=lambda row: (row[0], row[1], -1 if row[2] is None else row[2]),
            )
            simple_evaluation = {
                str(row["group_id"]): row
                for row in _arm_seed_rows(evaluation_rows, simple_arm).get(simple_seed, [])
            }
            delta_rows = []
            for group_id in sorted(set(candidate_costs) & set(simple_evaluation)):
                simple_row = simple_evaluation[group_id]
                simple_cost = policy_cost_rows(
                    [float(simple_row["probability"])],
                    [int(simple_row["label"])],
                    simple_policy,
                    false_accept_cost=false_accept,
                    false_reject_cost=FALSE_REJECT_COST,
                    escalation_cost=escalation,
                )[0]
                candidate_cost = float(np.mean(candidate_costs[group_id]))
                delta_rows.append(
                    {
                        "group_id": group_id,
                        "delta": candidate_cost - float(simple_cost["cost"]),
                    }
                )
            comparison = paired_bootstrap(delta_rows, draws=draws, seed=seed + len(pending))
            comparisons[cell_id] = comparison
            pending.append(
                {
                    "cell_id": cell_id,
                    "false_accept_cost": false_accept,
                    "false_reject_cost": FALSE_REJECT_COST,
                    "escalation_cost": escalation,
                    "candidate_policies": candidate_policies,
                    "best_simple_arm": simple_arm,
                    "best_simple_seed": simple_seed,
                    "best_simple_policy": simple_policy,
                    "candidate_non_escalation": float(
                        np.mean(
                            [
                                value
                                for values in candidate_non_escalation.values()
                                for value in values
                            ]
                        )
                    ),
                }
            )
    adjusted = holm_upper_bounds(comparisons, alpha=0.05)
    cells = []
    for row in pending:
        inference = adjusted[row["cell_id"]]
        passed = (
            row["candidate_non_escalation"] >= 0.20
            and float(inference["delta"]) < 0.0
            and float(inference["holm_upper"]) < 0.0
        )
        cells.append({**row, "comparison": inference, "benefit_passed": passed})
    return {
        "cells": cells,
        "holm_family_size": 9,
        "decision_benefit_score": int(any(row["benefit_passed"] for row in cells)),
        "deployment_certificate_valid": False,
        "scope": "sealed_research_cost_grid_not_universal_deployment",
    }


def reduce_prediction_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_draws: int = BOOTSTRAP_DRAWS
) -> JsonDict:
    """Independently reduce raw probability rows into all registered outcomes."""

    averaged = _average_seed_rows(rows)
    report: dict[str, dict[str, JsonDict]] = {}
    for role in ("calibration_tuning", "internal_test", "external"):
        role_rows = [row for row in averaged if row["role"] == role]
        if not role_rows:
            continue
        report[role] = {}
        for arm in ALL_ARMS:
            arm_rows = [row for row in role_rows if row["arm"] == arm]
            if arm_rows:
                report[role][arm] = probability_metrics(
                    [row["label"] for row in arm_rows],
                    [row["probability"] for row in arm_rows],
                )
    external = [row for row in rows if row.get("role") == "external"]
    brier_raw: dict[str, JsonDict] = {}
    log_loss: dict[str, JsonDict] = {}
    for index, control in enumerate(("temperature", "logistic")):
        brier_raw[control] = paired_bootstrap(
            group_loss_deltas(external, candidate="gibbs", control=control, loss="brier"),
            draws=bootstrap_draws,
            seed=BOOTSTRAP_SEED + index,
        )
        log_loss[control] = paired_bootstrap(
            group_loss_deltas(external, candidate="gibbs", control=control, loss="log_loss"),
            draws=bootstrap_draws,
            seed=BOOTSTRAP_SEED + 100 + index,
        )
    brier = holm_upper_bounds(brier_raw, alpha=0.05)
    log_loss_public = {
        name: {
            key: deepcopy(value) for key, value in comparison.items() if key != "bootstrap_means"
        }
        | {"upper_noninferiority": comparison["ci95"][1]}
        for name, comparison in log_loss.items()
    }
    probability_passed = all(
        float(brier[name]["delta"]) <= -0.01
        and float(brier[name]["holm_upper"]) < 0.0
        and float(log_loss_public[name]["upper_noninferiority"]) <= 0.01
        for name in ("temperature", "logistic")
    )
    calibration = [row for row in rows if row.get("role") == "calibration_tuning"]
    decision = (
        evaluate_cost_grid(
            calibration,
            external,
            draws=bootstrap_draws,
            seed=BOOTSTRAP_SEED + 500,
        )
        if calibration
        else {
            "cells": [],
            "holm_family_size": 9,
            "decision_benefit_score": 0,
            "deployment_certificate_valid": False,
            "scope": "calibration_rows_absent",
        }
    )
    role_group_counts = {
        role: len({str(row["group_id"]) for row in rows if row.get("role") == role})
        for role in ("internal_test", "external")
    }
    evaluation_complete = all(
        role_group_counts[role] > 0
        and all(arm in report.get(role, {}) for arm in ("gibbs", "temperature", "logistic"))
        for role in ("internal_test", "external")
    )
    return {
        "evaluation_complete": evaluation_complete,
        "role_group_counts": role_group_counts,
        "probability_report": report,
        "probability_comparisons": {"brier": brier, "log_loss": log_loss_public},
        "probability_benefit_score": int(probability_passed),
        "decision_report": decision,
        "decision_benefit_score": int(decision["decision_benefit_score"]),
        "failed_prediction_rows": sum(row.get("failed") is True for row in rows),
    }


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - runtime I/O boundary.
    """Load one external JSON object without repairing malformed evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - runtime I/O boundary.
    """Load typed JSON lines and fail on a non-object row."""

    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}")
            rows.append(value)
    return rows


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Write one bounded shard atomically and return its byte identity."""

    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix()
        if path.is_relative_to(REPO_ROOT)
        else str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "size_bytes": path.stat().st_size,
    }


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Record exact expected and observed prerequisite values."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "in" if isinstance(expected, list) else "==",
        "passed": bool(passed),
    }


def upstream_preconditions(fit_path: Path, eval_path: Path) -> list[JsonDict]:
    """Authenticate both current capture producers without absorbing their calls."""

    fit = _load_object(fit_path)
    evaluation = _load_object(eval_path)
    allowed = ["null", "positive"]
    return [
        _precondition(
            "fit_ready",
            str(fit_path),
            "fit_capture_ready_score",
            1,
            fit.get("fit_capture_ready_score"),
            fit.get("fit_capture_ready_score") == 1,
        ),
        _precondition(
            "fit_verdict",
            str(fit_path),
            "verdict_class",
            allowed,
            fit.get("verdict_class"),
            fit.get("verdict_class") in allowed,
        ),
        _precondition(
            "fit_unflagged",
            str(fit_path),
            "flagged_adversarial",
            False,
            fit.get("flagged_adversarial"),
            fit.get("flagged_adversarial") is False,
        ),
        _precondition(
            "eval_ready",
            str(eval_path),
            "evaluation_capture_ready_score",
            1,
            evaluation.get("evaluation_capture_ready_score"),
            evaluation.get("evaluation_capture_ready_score") == 1,
        ),
        _precondition(
            "eval_verdict",
            str(eval_path),
            "verdict_class",
            allowed,
            evaluation.get("verdict_class"),
            evaluation.get("verdict_class") in allowed,
        ),
        _precondition(
            "eval_unflagged",
            str(eval_path),
            "flagged_adversarial",
            False,
            evaluation.get("flagged_adversarial"),
            evaluation.get("flagged_adversarial") is False,
        ),
    ]


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind terminal content while excluding only the checksum itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _field_principles() -> dict[str, str]:
    """Explain why each mandatory field exists without hiding its raw value."""

    return {
        "schema": "Versioned schema with exact roadmap identity prevents silent reader drift.",
        "run_date": "The fixed date and measured clocks prevent a replay from posing as the original run.",
        "preconditions_checked": "Exact resources and observed prerequisite values prevent fitting from absent evidence.",
        "MODEL_SPECS": "An empty model list separates numeric work from current model inference.",
        "model_invoked": "Attempted current calls stay distinct from archived model-shaped evidence.",
        "invocation_counts": "Balanced zero counters prevent old model calls from entering current accounting.",
        "inference_substrate": "The declared numeric work prevents a model-duration claim from being inferred.",
        "inference_substrate_class": "No-model-load classification applies the correct runtime and evidence rules.",
        "execution_venue": "Host identity keeps historical board evidence separate from current CPU work.",
        "duration_s": "Measured components expose missing work without padded runtime.",
        "phase_spans": "Timestamped boundaries expose silent or unfinished phases.",
        "random_seed": "Frozen fitting and bootstrap seeds prevent favorable reruns.",
        "reproducibility_checksum": "The checksum binds code, inputs, roles, checkpoints, rows, and validation scope.",
        "source_artifact_hashes": "Exact bytes and original flags prevent upstream history from being cleaned.",
        "rows": "One fit unit per arm and seed keeps failures and checkpoints auditable.",
        "sample_size_budget": "Independent units stay separate from repeated seed rows.",
        "acceptance_gate_results": "Typed gates keep validity, readiness, and benefit separate.",
        "gate_check_summary": "A failed terminal state names its exact source and observation.",
        "honest_verdict": "A complete terminal prefix distinguishes a valid null from unfinished work.",
        "verdict_class": "The closed enum prevents prose from changing machine classification.",
        "verifier_is_oracle": "False records that human labels, not the learned selector, define evaluation truth.",
        "flagged_adversarial": "Reader flags cannot be cleared to open a scientific gate.",
        "validation_receipts": "Exact commands, exits, and log hashes establish the affected validation scope.",
        "field_principles": "Local explanations keep the evidence understandable without external context.",
        "calibration_complete_score": "Completion records valid fit and full evaluation even when benefit is null.",
        "probability_benefit_score": "Probability quality stays separate from policy usefulness.",
        "decision_benefit_score": "Typed value requires coverage and multiplicity-aware cost improvement.",
        "small_ebm_training": "Architecture, seeds, steps, and checkpoints authenticate numeric fitting.",
        "frozen_policy_manifest": "Immutable policies prevent evaluation-driven selection.",
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
) -> JsonDict:
    """Attach the registered failure-mode principle to every gate."""

    principles = {
        "required_validity": "A positive scientific metric cannot excuse invalid evidence.",
        "readiness": "A valid null must not suppress an independent measurement.",
        "scientific_benefit": "A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
    }
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principles[category],
    }


def _load_prediction_shard(
    value: Mapping[str, Any], root: Path
) -> list[JsonDict]:  # pragma: no cover
    """Reload and authenticate the exact per-group prediction shard."""

    manifest = value.get("prediction_row_shard")
    if not isinstance(manifest, Mapping):
        raise ValueError("prediction_manifest_missing")
    path = Path(str(manifest.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != manifest.get("sha256"):
        raise ValueError("prediction_shard_hash_mismatch")
    rows = _load_jsonl(resolved)
    if len(rows) != manifest.get("rows"):
        raise ValueError("prediction_shard_count_mismatch")
    return rows


def independent_reduce(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> JsonDict:  # pragma: no cover
    """Cold-reduce the persisted group rows without trusting producer summaries."""

    return reduce_prediction_rows(
        _load_prediction_shard(value, root), bootstrap_draws=BOOTSTRAP_DRAWS
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, check_files: bool = True
) -> list[str]:
    """Cold-check identity, accounting, frozen hashes, reductions, and verdict."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_accounting_invalid")
    if (
        value.get("inference_substrate_class") != "no_model_load"
        or value.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    if value.get("deployment_certificate_valid") is not False:
        errors.append("deployment_certificate_forbidden")
    if value.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_invalid")
    if value.get("verdict_class") not in {"positive", "null", "blocked", "disqualified"}:
        errors.append("verdict_class_invalid")
    bundle = value.get("frozen_policy_manifest")
    if not isinstance(bundle, Mapping) or bundle.get("bundle_sha256") != numeric_bundle_hash(
        bundle
    ):
        errors.append("frozen_policy_hash_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if check_files:  # pragma: no cover - exercised by the fresh-process capability E2E.
        try:
            reduced = independent_reduce(value, root=root)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("independent_reduction_failed")
        else:
            if reduced != value.get("independent_reduction"):
                errors.append("independent_reduction_mismatch")
            if value.get("probability_benefit_score") != reduced["probability_benefit_score"]:
                errors.append("probability_score_mismatch")
            if value.get("decision_benefit_score") != reduced["decision_benefit_score"]:
                errors.append("decision_score_mismatch")
        for label, row in (value.get("source_artifact_hashes") or {}).items():
            if not isinstance(row, Mapping):
                errors.append(f"source_hash_row_invalid:{label}")
                continue
            path = Path(str(row.get("path") or label))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{label}")
    return list(dict.fromkeys(errors))


def fixture_artifact() -> JsonDict:
    """Build one minimal cold-valid shape for mutation tests."""

    residual = initial_gibbs_checkpoint(seed=TRAINING_SEEDS[0], input_dim=6)
    bundle: JsonDict = {
        "schema": "carnot.exp7481.frozen_fit_bundle.v1",
        "training_seeds": list(TRAINING_SEEDS),
        "roles_consumed": ["training", "calibration_tuning"],
        "heldout_labels_consumed": False,
        "feature_names": list(FULL_FEATURE_NAMES),
        "heads": {},
        "checkpoint_manifest": [],
        "initial_states_for_exp7483": [
            {
                "seed": TRAINING_SEEDS[0],
                "base_arm": "temperature",
                "base_temperature": 1.0,
                "residual_checkpoint": residual,
                "residual_checkpoint_sha256": canonical_hash(residual),
            }
        ],
    }
    bundle["bundle_sha256"] = numeric_bundle_hash(bundle)
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_fixture",
        "run_date": RUN_DATE,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "compact_numeric_head_training_and_group_reduction",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 0.1,
        "phase_spans": [],
        "random_seed": {"fitting": list(TRAINING_SEEDS), "bootstrap": BOOTSTRAP_SEED},
        "source_artifact_hashes": {},
        "rows": [{"row_type": "fixture", "attempted": True, "completed": True}],
        "sample_size_budget": {
            "planned": 1,
            "attempted": 1,
            "complete": 1,
            "failed": 0,
            "censored": 0,
            "excluded": 0,
            "unstarted": 0,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {"required_checks_passed": True},
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "calibration_complete_score": 1,
        "probability_benefit_score": 0,
        "decision_benefit_score": 0,
        "small_ebm_training": {"performed": True, "current_llm_calls": 0},
        "frozen_policy_manifest": bundle,
        "deployment_certificate_valid": False,
    }
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC timestamp for a real boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7481] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed: int
) -> JsonDict:  # pragma: no cover
    """Record one measured monotonic phase interval."""

    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": time.monotonic() - run_started,
        "completed_units": completed,
    }


def _run_child(
    argv: Sequence[str], *, root: Path, started: float, phase: str, timeout_s: float
) -> int:  # pragma: no cover - owned process boundary.
    """Stream one owned child and print truthful 60-second heartbeats."""

    progress(started, phase, "before_subprocess", command=argv[0])
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    process = subprocess.Popen(
        list(argv),
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(60.0):
            progress(started, phase, "subprocess_pending", pid=process.pid)
            if time.monotonic() - started > timeout_s:
                process.terminate()
                return

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[exp7481:{phase}] {line.rstrip()}", flush=True)
    code = process.wait()
    stop.set()
    monitor.join(timeout=1.0)
    progress(started, phase, "after_subprocess", exit_code=code)
    return int(code)


def _source_hash_row(
    path: Path, root: Path, artifact: Mapping[str, Any] | None = None
) -> JsonDict:  # pragma: no cover
    """Bind one exact input and preserve its original disposition fields."""

    resolved = path if path.is_absolute() else root / path
    return {
        "path": str(path) if path.is_absolute() else path.as_posix(),
        "sha256": sha256_file(resolved),
        "original_flagged_adversarial": (artifact or {}).get("flagged_adversarial"),
        "original_verdict_class": (artifact or {}).get("verdict_class"),
    }


def _collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict]]:  # pragma: no cover
    """Read exact source bytes and authenticate current plus historical states."""

    checks = upstream_preconditions(root / FIT_ARTIFACT, root / EVAL_ARTIFACT)
    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
        Path("python/carnot/models/gibbs/__init__.py"),
        Path("python/carnot/training/platt_scaler.py"),
        Path("python/carnot/experiment_7439_v652_certified_decisions.py"),
        Path("python/carnot/experiment_7412_v650_source_features.py"),
        Path("ops/verifier_gaps.md"),
        Path("openspec/capabilities/autoresearch/spec.md"),
        FIT_ARTIFACT,
        EVAL_ARTIFACT,
        HISTORICAL_CAPSTONE,
        HISTORICAL_NULL,
        COHORT_PREDICTORS,
        COHORT_EVALUATORS,
        FIT_RAW_DIR / "raw-logits-training.jsonl",
        FIT_RAW_DIR / "raw-logits-calibration_tuning.jsonl",
        EVAL_RAW_DIR / "raw-logits-internal_test.jsonl",
        EVAL_RAW_DIR / "raw-logits-external.jsonl",
    )
    hashes: dict[str, JsonDict] = {}
    artifact_paths = {FIT_ARTIFACT, EVAL_ARTIFACT, HISTORICAL_CAPSTONE, HISTORICAL_NULL}
    for relative in paths:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                available,
            )
        )
        if available:
            artifact = _load_object(path) if relative in artifact_paths else None
            hashes[relative.as_posix()] = _source_hash_row(relative, root, artifact)
    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            "openspec/capabilities/autoresearch/spec.md",
            "REQ-*",
            "REQ-AUTO-7481",
            "REQ-AUTO-7481" if "REQ-AUTO-7481" in spec_text else None,
            "REQ-AUTO-7481" in spec_text,
        )
    )
    capstone = _load_object(root / HISTORICAL_CAPSTONE)
    old_rows = capstone.get("rows") or []
    historical = next(
        (
            row
            for row in old_rows
            if isinstance(row, Mapping) and row.get("task_id") == "exp7466-typed-energy-calibration"
        ),
        {},
    )
    checks.append(
        _precondition(
            "v654_missing_producer_preserved",
            HISTORICAL_CAPSTONE.as_posix(),
            "rows[task_id=exp7466].honest_verdict",
            "blocked_missing_declared_producer_evidence",
            historical.get("honest_verdict"),
            historical.get("honest_verdict") == "blocked_missing_declared_producer_evidence",
        )
    )
    old_null = _load_object(root / HISTORICAL_NULL)
    checks.append(
        _precondition(
            "handcrafted_selector_null_preserved",
            HISTORICAL_NULL.as_posix(),
            "honest_verdict",
            "complete_null_no_registered_decision_benefit",
            old_null.get("honest_verdict"),
            old_null.get("honest_verdict") == "complete_null_no_registered_decision_benefit",
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    quarantined = "experiment_id: 7481" in exclusion or "exp7481-typed-calibration" in exclusion
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            quarantined,
            not quarantined,
        )
    )
    return checks, hashes


def _join_feature_rows(
    native: Sequence[Mapping[str, Any]], predictors: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Join by group identity, then discard identity from the numeric vector."""

    by_group = {str(row.get("group_id")): row for row in predictors}
    output = []
    receipts = []
    for row in native:
        predictor = by_group.get(str(row["group_id"]))
        if predictor is None:
            raise ValueError(f"predictor_join_missing:{row['group_id']}")
        features, receipt = project_predictor_features(
            predictor, native_log_odds=float(row["native_log_odds"])
        )
        output.append({**deepcopy(dict(row)), "features": features})
        receipts.append({"group_id": row["group_id"], **receipt})
    return output, receipts


def _fit_worker(input_path: Path, output_path: Path, steps: int) -> int:  # pragma: no cover
    """Run the label-limited child process and write one frozen bundle."""

    payload = _load_object(input_path)
    bundle = fit_numeric_bundle(payload["training"], payload["calibration"], steps=steps)
    atomic_json(output_path, bundle)
    print(
        json.dumps(
            {
                "fit_bundle_sha256": bundle["bundle_sha256"],
                "training_groups": bundle["training_group_count"],
                "calibration_groups": bundle["calibration_group_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the four required fresh-process readers for one exact candidate."""

    python = ".venv/bin/python"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--cold-replay",
                    candidate.as_posix(),
                ),
                "terminal_candidate",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_row_reduction",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--independent-reduce",
                    candidate.as_posix(),
                ),
                "terminal_candidate_rows",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", candidate.as_posix()),
                "terminal_candidate",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    candidate.as_posix(),
                ),
                "terminal_candidate_rows",
            ),
            "required_validation",
            True,
        ),
    ]


def _validation_passed(
    receipts: Sequence[Mapping[str, Any]], *, candidate: bool
) -> bool:  # pragma: no cover
    """Require the frozen affected checks and terminal readers when final."""

    required = set(AFFECTED_CHECK_NAMES) | (set() if candidate else set(TERMINAL_CHECK_NAMES))
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in required
    )


def _build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    reduction: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    candidate: bool,
    flagged: bool,
) -> JsonDict:  # pragma: no cover - terminal assembly.
    """Assemble one schema-complete candidate from measured work."""

    required_validation = _validation_passed(receipts, candidate=candidate)
    prerequisites_passed = all(row.get("passed") is True for row in preconditions)
    evaluation_complete = reduction.get("evaluation_complete") is True
    completion = int(
        prerequisites_passed and required_validation and evaluation_complete and not flagged
    )
    probability = int(reduction.get("probability_benefit_score") == 1 and completion == 1)
    decision = int(reduction.get("decision_benefit_score") == 1 and completion == 1)
    if not prerequisites_passed:
        verdict_class = "blocked"
        honest = "blocked_external_capture_prerequisite"
    elif not required_validation or flagged or not evaluation_complete:
        verdict_class = "disqualified"
        honest = "complete_disqualified_required_validation_or_evaluation"
    elif probability or decision:
        verdict_class = "positive"
        honest = "complete_positive_probability_or_typed_decision_benefit"
    else:
        verdict_class = "null"
        honest = "complete_null_no_registered_probability_or_decision_benefit"
    rows = [
        {
            "row_type": "small_ebm_fit",
            "arm": row["arm"],
            "seed": row["seed"],
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "initial_checkpoint_sha256": row["initial_checkpoint_sha256"],
        }
        for row in bundle["checkpoint_manifest"]
    ]
    group_counts = {"training": 176, "calibration_tuning": 60, **reduction["role_group_counts"]}
    sample_size = {
        "planned": 374,
        "attempted": sum(group_counts.values()),
        "complete": sum(group_counts.values()),
        "failed": 0,
        "censored": 0,
        "excluded": 4,
        "unstarted": 0,
        "independent_group_counts": group_counts,
        "seeds_multiply_group_count": False,
    }
    gates = [
        _gate(
            "authenticated_inputs_and_process_isolation",
            "required_validity",
            True,
            prerequisites_passed and bundle.get("heldout_labels_consumed") is False,
            "==",
            prerequisites_passed and bundle.get("heldout_labels_consumed") is False,
        ),
        _gate(
            "affected_and_terminal_validation",
            "required_validity",
            True,
            required_validation,
            "==",
            required_validation,
        ),
        _gate(
            "calibration_complete",
            "readiness",
            1,
            completion,
            "==",
            completion == 1,
        ),
        _gate(
            "probability_benefit",
            "scientific_benefit",
            1,
            probability,
            "==",
            probability == 1,
        ),
        _gate(
            "decision_benefit",
            "scientific_benefit",
            1,
            decision,
            "==",
            decision == 1,
        ),
    ]
    failed_required = [
        row["check"]
        for row in gates
        if row["category"] in {"required_validity", "readiness"} and row["passed"] is not True
    ]
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "clock_identity": {"utc": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "process_identity": {"pid": os.getpid(), "python": sys.executable},
        "device_identity": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cuda_used": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "compact_numeric_head_training_and_exact_source_group_reduction",
        "inference_substrate_details": {
            "current_llm_calls": 0,
            "archived_model_rows": "hash_bound_inputs_only",
            "qwen_trained": False,
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_components_s": {
            "model_load": 0.0,
            "model_forward": 0.0,
            "generation": 0.0,
            "numeric_fit": next(
                (
                    float(row["end_s"]) - float(row["start_s"])
                    for row in spans
                    if row.get("phase") == "numeric_fit"
                ),
                0.0,
            ),
            "validation": sum(float(row.get("duration_s") or 0.0) for row in receipts),
        },
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": {
            "fitting": list(TRAINING_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "audit": 655_199,
            "ordering": "upstream_frozen_group_order",
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": sample_size,
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "required_checks_passed": not failed_required,
            "failed_required_checks": failed_required,
            "scientific_benefit_passed": bool(probability or decision),
            "blocked_upstream": None,
            "blocked_path": None,
            "blocked_field": None,
            "blocked_expected": None,
            "blocked_observed": None,
        },
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": bool(flagged),
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
            "frozen_before_checks": True,
        },
        "field_principles": _field_principles(),
        "calibration_complete_score": completion,
        "probability_benefit_score": probability,
        "decision_benefit_score": decision,
        "small_ebm_training": {
            "performed": True,
            "receipt_class": "small_ebm_training",
            "current_llm_calls": 0,
            "architectures": {
                "gibbs": {"input_dim": 6, "hidden_dims": [4], "output_dim": 1},
                "logistic": {"input_dim": 6, "output_dim": 1},
            },
            "optimizer": "adam_with_frozen_existing_helper_constants",
            "steps": bundle["heads"]["gibbs"][0]["optimizer"]["steps"],
            "training_seeds": list(TRAINING_SEEDS),
            "checkpoint_manifest": deepcopy(bundle["checkpoint_manifest"]),
            "fit_elapsed_s": next(
                (
                    float(row["end_s"]) - float(row["start_s"])
                    for row in spans
                    if row.get("phase") == "numeric_fit"
                ),
                0.0,
            ),
            "numeric_deadline_s": MAX_NUMERIC_SECONDS,
        },
        "frozen_policy_manifest": deepcopy(dict(bundle)),
        "prediction_row_shard": deepcopy(dict(shard)),
        "independent_reduction": deepcopy(dict(reduction)),
        "probability_report": deepcopy(reduction["probability_report"]),
        "probability_comparisons": deepcopy(reduction["probability_comparisons"]),
        "decision_cost_grid": deepcopy(reduction["decision_report"]),
        "ablation_report": {
            arm: deepcopy(reduction["probability_report"].get("external", {}).get(arm))
            for arm in ("verifier_only_gibbs", "shuffled_label_gibbs", "source_removal_gibbs")
        },
        "source_shuffle_controls": {
            "rows": len(controls),
            "gold_labels_assigned": 0,
            "benefit_eligible": False,
            "use": "source_sensitivity_only",
        },
        "initial_states_for_exp7483": deepcopy(bundle["initial_states_for_exp7483"]),
        "deployment_certificate_valid": False,
        "universal_deployment_certificate": False,
        "external_publication_authorized": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": not candidate,
            "numbered_e2e_scenarios": [],
            "reason": "No shared training, sampling, binding, ARC, telemetry, or Rust code changed.",
        },
        "candidate_artifact": candidate,
    }
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def _blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Publish a precise external prerequisite block without numeric fitting."""

    value = fixture_artifact()
    value.update(
        {
            "status": "blocked_external_capture_prerequisite",
            "honest_verdict": "blocked_external_capture_prerequisite",
            "verdict_class": "blocked",
            "preconditions_checked": [deepcopy(dict(failed))],
            "calibration_complete_score": 0,
            "small_ebm_training": {"performed": False, "current_llm_calls": 0},
            "gate_check_summary": {
                "required_checks_passed": False,
                "failed_required_checks": [failed.get("check")],
                "blocked_upstream": failed.get("upstream"),
                "blocked_path": failed.get("path"),
                "blocked_field": failed.get("field"),
                "blocked_expected": failed.get("expected"),
                "blocked_observed": failed.get("observed"),
            },
        }
    )
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, fit across a process boundary, validate, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes = _collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = _blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    progress(run_started, "feature_projection", "before_benchmark")
    phase_started = time.monotonic()
    predictors = _load_jsonl(root / COHORT_PREDICTORS)
    training_native, training_controls = aggregate_native_rows(
        _load_jsonl(root / FIT_RAW_DIR / "raw-logits-training.jsonl")
    )
    calibration_native, calibration_controls = aggregate_native_rows(
        _load_jsonl(root / FIT_RAW_DIR / "raw-logits-calibration_tuning.jsonl")
    )
    training, _training_projection = _join_feature_rows(training_native, predictors)
    calibration, _calibration_projection = _join_feature_rows(calibration_native, predictors)
    spans.append(
        _span("feature_projection", phase_started, run_started, len(training) + len(calibration))
    )
    progress(
        run_started,
        "feature_projection",
        "after_benchmark",
        completed=len(training) + len(calibration),
    )

    progress(run_started, "numeric_fit", "before_training", planned=25)
    phase_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exp7481-fit-", dir="/tmp") as temporary:
        temporary_root = Path(temporary)
        fit_input = temporary_root / "fit-input.json"
        fit_output = temporary_root / "fit-output.json"
        atomic_json(fit_input, {"training": training, "calibration": calibration})
        code = _run_child(
            (
                sys.executable,
                "-u",
                "-m",
                "carnot.experiment_7481_v655_typed_calibration",
                "--date",
                RUN_DATE,
                "--fit-input",
                str(fit_input),
                "--fit-output",
                str(fit_output),
            ),
            root=root,
            started=run_started,
            phase="numeric_fit",
            timeout_s=MAX_NUMERIC_SECONDS,
        )
        if code != 0:
            raise RuntimeError(f"fit_subprocess_failed:{code}")
        bundle = _load_object(fit_output)
    if bundle.get("bundle_sha256") != numeric_bundle_hash(bundle):
        raise RuntimeError("fit_bundle_hash_invalid")
    atomic_json(root / FIT_BUNDLE_PATH, bundle)
    source_hashes[FIT_BUNDLE_PATH.as_posix()] = _source_hash_row(FIT_BUNDLE_PATH, root)
    spans.append(_span("numeric_fit", phase_started, run_started, 25))
    progress(run_started, "numeric_fit", "after_training", completed=25)

    progress(run_started, "heldout_evaluation", "before_benchmark")
    phase_started = time.monotonic()
    internal_native, internal_controls = aggregate_native_rows(
        _load_jsonl(root / EVAL_RAW_DIR / "raw-logits-internal_test.jsonl")
    )
    external_native, external_controls = aggregate_native_rows(
        _load_jsonl(root / EVAL_RAW_DIR / "raw-logits-external.jsonl")
    )
    internal, _internal_projection = _join_feature_rows(internal_native, predictors)
    external, _external_projection = _join_feature_rows(external_native, predictors)
    prediction_rows = score_numeric_bundle(bundle, [*calibration, *internal, *external])
    shard = _jsonl_write(root / PREDICTION_SHARD, prediction_rows)
    source_hashes[PREDICTION_SHARD.as_posix()] = _source_hash_row(PREDICTION_SHARD, root)
    reduction = reduce_prediction_rows(prediction_rows)
    spans.append(
        _span("heldout_evaluation", phase_started, run_started, len(internal) + len(external))
    )
    progress(
        run_started,
        "heldout_evaluation",
        "after_benchmark",
        completed=len(internal) + len(external),
    )

    source_hashes.update(
        {
            path.as_posix(): _source_hash_row(path, root)
            for path in (
                MODULE_PATH,
                WRAPPER_PATH,
                TEST_PATH,
                Path("openspec/capabilities/autoresearch/spec.md"),
            )
        }
    )
    private_root = Path(tempfile.mkdtemp(prefix="exp7481-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    controls = [*training_controls, *calibration_controls, *internal_controls, *external_controls]
    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        bundle=bundle,
        shard=shard,
        reduction=reduction,
        controls=controls,
        receipts=affected,
        spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        candidate=True,
        flagged=not affected_reduction["passed"] or bool(plan_errors),
    )
    atomic_json(root / TERMINAL_CANDIDATE, candidate)

    progress(run_started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        _terminal_commands(TERMINAL_CANDIDATE),
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        bundle=bundle,
        shard=shard,
        reduction=reduction,
        controls=controls,
        receipts=[*affected, *terminal],
        spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        candidate=False,
        flagged=(
            not affected_reduction["passed"] or not terminal_passed or critical or bool(plan_errors)
        ),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date plus fit-worker and cold-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--fit-input", type=Path)
    parser.add_argument("--fit-output", type=Path)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the measurement, isolated fitter, or one terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.fit_input is not None or args.fit_output is not None:
        if args.fit_input is None or args.fit_output is None:
            raise SystemExit("--fit-input and --fit-output must be supplied together")
        return _fit_worker(args.fit_input, args.fit_output, args.fit_steps)
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value, root=root) if value else ["artifact_unreadable"]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        reduced = independent_reduce(value, root=root) if value else {"error": "unreadable"}
        passed = bool(value) and reduced == value.get("independent_reduction")
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
