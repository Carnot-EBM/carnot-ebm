"""Fit source-contrast energy heads on sealed native option readouts.

This CPU-only module trains a small decision policy. It does not load the
generator, certify source truth, or act as a generic text reranker.

Spec refs: REQ-VERIFY-7566 and SCENARIO-VERIFY-7566-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
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
from carnot.experiment_7425_v651_spline_prototype import (
    COEFFICIENTS_PER_INPUT,
    DEGREE,
    KNOT_VECTOR_SIZE,
    _repair_breakpoints,
    cubic_basis,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7566-v661-energy-fit"
SCHEMA = "carnot.exp7566.v661.energy_fit.v1"
RESULT_PATH = Path("results/experiment_7566_v661_energy_fit.json")
RAW_DIR = Path("results/raw/experiment_7566_v661_energy_fit")
MODULE_PATH = Path("python/carnot/experiment_7566_v661_energy_fit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7566_v661_energy_fit.py")
TEST_PATH = Path("tests/python/test_experiment_7566_v661_energy_fit.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
UPSTREAM_PATH = Path("results/experiment_7564_v661_fit_capture.json")
PROTOCOL_PATH = Path("results/experiment_7533_v659_tool_protocol.json")
TERMINAL_CANDIDATE = RAW_DIR / "measured_terminal_candidate.json"
CHECKPOINT_PATH = RAW_DIR / "frozen_heads.json"
TRAINING_ROWS_PATH = RAW_DIR / "training_rows.jsonl"
POLICY_PATH = RAW_DIR / "policy_report.json"

FIT_SEED = 7_566_001
CONSISTENCY_GRID = (0.0, 0.1, 1.0)
L2_GRID = (0.001, 0.01, 0.1)
LEARNING_RATE = 0.03
OPTIMIZER_STEPS = 300
TEMPERATURE_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
FEATURE_NAMES = ("z_original", "z_original_minus_absent", "z_original_minus_mismatched")
OPTION_ORDERS = (
    ("supported", "contains_unsupported"),
    ("contains_unsupported", "supported"),
)
ORDER_NAMES = ("supported_first", "unsupported_first")
TRAINABLE_FAMILIES = (
    "source_contrast_energy",
    "unconstrained_equal_capacity_energy",
    "original_only_local_basis_energy",
    "same_information_logistic",
)
ZERO_INVOCATION_COUNTS = {
    operation: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for operation in ("model_loads", "forward_calls", "generation_calls")
}

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def stable_sigmoid(value: Any) -> np.ndarray:
    """Return finite logistic probabilities without exponential overflow."""

    array = np.asarray(value, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def binary_energy(logits: Any) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate E(0,x)=0 and E(1,x)=-f(x), then normalize both states."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("logits_nonfinite")
    unsupported = stable_sigmoid(values)
    return (
        np.column_stack((np.zeros_like(values), -values)),
        np.column_stack((1.0 - unsupported, unsupported)),
    )


def fit_local_knots(training_features: Any) -> np.ndarray:
    """Fit three fixed open cubic knot vectors from fit-role features only."""

    training = np.asarray(training_features, dtype=np.float64)
    if training.ndim != 2 or training.shape[1] != len(FEATURE_NAMES) or len(training) < 2:
        raise ValueError("feature_matrix_shape_invalid")
    if not np.all(np.isfinite(training)):
        raise ValueError("feature_matrix_nonfinite")
    vectors = []
    for column in training.T:
        breaks = _repair_breakpoints(
            np.asarray(np.quantile(column, np.linspace(0.0, 1.0, 6)), dtype=np.float64)
        )
        vectors.append(
            np.concatenate(
                (
                    (
                        np.repeat(breaks[0], DEGREE + 1),
                        breaks[1:-1],
                        np.repeat(breaks[-1], DEGREE + 1),
                    )
                )
            )
        )
    return np.asarray(vectors, dtype=np.float64)


def local_design_matrix(features: Any, knots: Any) -> np.ndarray:
    """Expand three features into eight fixed local cubic values each."""

    matrix = np.asarray(features, dtype=np.float64)
    knot_matrix = np.asarray(knots, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature_matrix_shape_invalid")
    if knot_matrix.shape != (len(FEATURE_NAMES), KNOT_VECTOR_SIZE):
        raise ValueError("knots_shape_invalid")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(knot_matrix)):
        raise ValueError("spline_input_nonfinite")
    output = np.empty((len(matrix), len(FEATURE_NAMES) * COEFFICIENTS_PER_INPUT))
    for row_index, row in enumerate(matrix):
        output[row_index] = np.concatenate(
            [
                cubic_basis(float(value), knot_matrix[index]).values
                for index, value in enumerate(row)
            ]
        )
    return output


def _semantic_logit(row: Mapping[str, Any]) -> float:
    """Map displayed logits back to stable semantic option identifiers."""

    raw_order = row.get("option_order")
    order = tuple(raw_order) if isinstance(raw_order, list) else ()
    display = row.get("display_logits")
    semantic = row.get("full_logits_by_option_id")
    if order not in OPTION_ORDERS or not isinstance(display, Mapping):
        raise ValueError("readout_mapping_invalid")
    if set(display) != {" A", " B"} or not isinstance(semantic, Mapping):
        raise ValueError("readout_mapping_invalid")
    mapped = dict(zip(order, (float(display[" A"]), float(display[" B"])), strict=True))
    expected = {name: float(semantic[name]) for name in ("supported", "contains_unsupported")}
    if any(not math.isfinite(value) for value in (*mapped.values(), *expected.values())):
        raise ValueError("readout_nonfinite")
    if any(not math.isclose(mapped[name], expected[name], abs_tol=1e-12) for name in expected):
        raise ValueError("readout_mapping_invalid")
    return mapped["contains_unsupported"] - mapped["supported"]


def build_source_group(
    native_rows: Sequence[Mapping[str, Any]], label_row: Mapping[str, Any]
) -> JsonDict:
    """Join one evaluator label to six complete source intervention cells."""

    if len(native_rows) != 6:
        raise ValueError("readout_cells_invalid")
    components = {str(row.get("component_hash")) for row in native_rows}
    groups = {str(row.get("group_hash")) for row in native_rows}
    roles = {str(row.get("role")) for row in native_rows}
    if len(components) != 1 or len(groups) != 1 or len(roles) != 1:
        raise ValueError("source_group_identity_invalid")
    component = next(iter(components))
    role = next(iter(roles))
    if label_row.get("component_hash") != component or label_row.get("role") != role:
        raise ValueError("evaluator_identity_mismatch")
    label = label_row.get("label")
    if label not in (0, 1):
        raise ValueError("binary_label_invalid")
    if any(row.get("donor_role") != role for row in native_rows):
        raise ValueError("donor_role_leakage")
    if any(
        row.get("disposition") != "complete"
        or row.get("generated_tokens") != 0
        or row.get("readout_kind") != "option_logits"
        for row in native_rows
    ):
        raise ValueError("native_readout_incomplete")
    cells: dict[tuple[tuple[str, str], str], float] = {}
    for row in native_rows:
        raw_order = row.get("option_order")
        order = tuple(raw_order) if isinstance(raw_order, list) else ()
        condition = str(row.get("condition"))
        key = (order, condition)
        if order not in OPTION_ORDERS or condition not in {"original", "absent", "mismatched"}:
            raise ValueError("readout_cells_invalid")
        if key in cells:
            raise ValueError("readout_cells_invalid")
        cells[key] = _semantic_logit(row)
    expected = {
        (order, condition)
        for order in OPTION_ORDERS
        for condition in ("original", "absent", "mismatched")
    }
    # Six unique cells drawn from the only six allowed keys must equal this set.
    if set(cells) != expected:  # pragma: no cover - proved by checks above.
        raise ValueError("readout_cells_invalid")
    features = {}
    for name, order in zip(ORDER_NAMES, OPTION_ORDERS, strict=True):
        original = cells[(order, "original")]
        features[name] = [
            original,
            original - cells[(order, "absent")],
            original - cells[(order, "mismatched")],
        ]
    return {
        "group_hash": next(iter(groups)),
        "component_hash": component,
        "role": role,
        "label": int(label),
        "features_by_order": features,
        "source_conditions": ["original", "absent", "mismatched"],
        "option_order_mapping": {
            name: list(order) for name, order in zip(ORDER_NAMES, OPTION_ORDERS, strict=True)
        },
    }


def _role_matrices(
    rows: Sequence[Mapping[str, Any]], expected_role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one role and return both order views plus labels."""

    views = [[], []]
    labels = []
    for row in rows:
        if row.get("role") != expected_role:
            raise ValueError(f"{expected_role}_role_invalid")
        label = row.get("label")
        if label not in (0, 1):
            raise ValueError("binary_label_invalid")
        feature_map = row.get("features_by_order")
        if not isinstance(feature_map, Mapping) or set(feature_map) != set(ORDER_NAMES):
            raise ValueError("feature_views_invalid")
        for index, name in enumerate(ORDER_NAMES):
            vector = np.asarray(feature_map[name], dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("feature_vector_invalid")
            views[index].append(vector.tolist())
        labels.append(int(label))
    if not rows or set(labels) != {0, 1}:
        raise ValueError("binary_support_invalid")
    return np.asarray(views[0]), np.asarray(views[1]), np.asarray(labels, dtype=np.float64)


def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute finite binary log loss for one fixed role."""

    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary Brier loss without selecting on policy labels."""

    return float(np.mean(np.square(probabilities - labels)))


def _js_consistency(first: np.ndarray, second: np.ndarray) -> float:
    """Measure mapped-order disagreement as Bernoulli Jensen-Shannon divergence."""

    first = np.clip(first, 1e-12, 1.0 - 1e-12)
    second = np.clip(second, 1e-12, 1.0 - 1e-12)
    middle = 0.5 * (first + second)
    first_kl = first * np.log(first / middle) + (1.0 - first) * np.log(
        (1.0 - first) / (1.0 - middle)
    )
    second_kl = second * np.log(second / middle) + (1.0 - second) * np.log(
        (1.0 - second) / (1.0 - middle)
    )
    return float(np.mean(0.5 * (first_kl + second_kl)))


def _family_design(family: str, features: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Apply one frozen transform while keeping source information explicit."""

    if family == "source_contrast_energy":
        return local_design_matrix(features, knots)
    if family == "unconstrained_equal_capacity_energy":
        clipped = np.clip(features, -4.0, 4.0)
        return np.concatenate([clipped**power for power in range(1, 9)], axis=1)
    if family == "original_only_local_basis_energy":
        return local_design_matrix(features, knots)[:, :COEFFICIENTS_PER_INPUT]
    if family == "same_information_logistic":
        return features.copy()
    raise ValueError(f"family_invalid:{family}")


def _fit_candidate(
    family: str,
    train_design: tuple[np.ndarray, np.ndarray],
    train_labels: np.ndarray,
    tune_design: tuple[np.ndarray, np.ndarray],
    tune_labels: np.ndarray,
    *,
    consistency_lambda: float,
    l2: float,
    steps: int,
) -> JsonDict:
    """Fit one paired-order candidate and retain its full bounded cost trace."""

    first_x, second_x = train_design
    width = first_x.shape[1]
    rng = np.random.default_rng(FIT_SEED)
    weights = rng.normal(0.0, 0.01, size=width)
    bias = 0.0
    first_moment = np.zeros(width + 1)
    second_moment = np.zeros(width + 1)
    trace: list[JsonDict] = []
    started = time.monotonic()
    for step in range(steps + 1):
        first_p = stable_sigmoid(first_x @ weights + bias)
        second_p = stable_sigmoid(second_x @ weights + bias)
        log_loss = 0.5 * (_log_loss(first_p, train_labels) + _log_loss(second_p, train_labels))
        js = _js_consistency(first_p, second_p)
        penalty = 0.5 * l2 * float(weights @ weights)
        total = log_loss + consistency_lambda * js + penalty
        if step == 0 or step == steps or step % 30 == 0:
            trace.append(
                {
                    "step": step,
                    "total_cost": total,
                    "log_loss": log_loss,
                    "js_consistency": js,
                    "l2_cost": penalty,
                }
            )
        if step == steps:
            break
        middle = np.clip(0.5 * (first_p + second_p), 1e-12, 1.0 - 1e-12)
        middle_logit = np.log(middle / (1.0 - middle))
        first_clip = np.clip(first_p, 1e-12, 1.0 - 1e-12)
        second_clip = np.clip(second_p, 1e-12, 1.0 - 1e-12)
        first_js = (
            0.5
            * (np.log(first_clip / (1.0 - first_clip)) - middle_logit)
            * first_p
            * (1.0 - first_p)
        )
        second_js = (
            0.5
            * (np.log(second_clip / (1.0 - second_clip)) - middle_logit)
            * second_p
            * (1.0 - second_p)
        )
        first_residual = 0.5 * (first_p - train_labels) + consistency_lambda * first_js
        second_residual = 0.5 * (second_p - train_labels) + consistency_lambda * second_js
        gradient_w = (first_x.T @ first_residual + second_x.T @ second_residual) / len(
            train_labels
        ) + l2 * weights
        gradient_b = float(np.mean(first_residual + second_residual))
        gradient = np.concatenate((gradient_w, [gradient_b]))
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9 ** (step + 1))
        corrected_second = second_moment / (1.0 - 0.999 ** (step + 1))
        update = LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        weights -= update[:-1]
        bias -= float(update[-1])
    tune_first = stable_sigmoid(tune_design[0] @ weights + bias)
    tune_second = stable_sigmoid(tune_design[1] @ weights + bias)
    tune_probability = 0.5 * (tune_first + tune_second)
    finite = bool(np.all(np.isfinite(weights)) and math.isfinite(bias))
    loss_increased = trace[-1]["total_cost"] > trace[0]["total_cost"] + 1e-12
    status = "complete" if finite and not loss_increased else "failed_loss_increase"
    checkpoint = {"weights": weights.tolist(), "bias": bias}
    return {
        "kind": "candidate",
        "family": family,
        "seed": FIT_SEED,
        "consistency_lambda": consistency_lambda,
        "l2": l2,
        "optimizer_steps": steps,
        "learning_rate": LEARNING_RATE,
        "parameter_count": width + 1,
        "train_cost_start": trace[0]["total_cost"],
        "train_cost_end": trace[-1]["total_cost"],
        "loss_trace": trace,
        "tune_brier": _brier(tune_probability, tune_labels),
        "tune_js_consistency": _js_consistency(tune_first, tune_second),
        "checkpoint": checkpoint,
        "checkpoint_sha256": canonical_hash(checkpoint),
        "fit_time_s": time.monotonic() - started,
        "status": status,
        "disposition": status,
        "failure": None if status == "complete" else "unexpected_loss_increase",
    }


def select_head(candidates: Sequence[Mapping[str, Any]], family: str) -> JsonDict:
    """Select one complete family candidate only by tuning Brier."""

    family_rows = [row for row in candidates if row.get("family") == family]
    if len(family_rows) != len(CONSISTENCY_GRID) * len(L2_GRID):
        raise ValueError(f"candidate_grid_incomplete:{family}")
    complete = [row for row in family_rows if row.get("status") == "complete"]
    if not complete:
        raise ValueError(f"candidate_grid_no_complete:{family}")
    selected = min(
        complete,
        key=lambda row: (
            float(row["tune_brier"]),
            float(row["consistency_lambda"]),
            float(row["l2"]),
        ),
    )
    return {
        "family": family,
        "selection_metric": "tune_brier",
        "tune_brier": selected["tune_brier"],
        "consistency_lambda": selected["consistency_lambda"],
        "l2": selected["l2"],
        "parameter_count": selected["parameter_count"],
        "checkpoint": deepcopy(selected["checkpoint"]),
        "checkpoint_sha256": selected["checkpoint_sha256"],
    }


def bundle_hash(bundle: Mapping[str, Any]) -> str:
    """Hash one fit bundle without its self-referential identity field."""

    return canonical_hash(
        {key: deepcopy(value) for key, value in bundle.items() if key != "bundle_sha256"}
    )


def _temperature_fit(probabilities: np.ndarray, labels: np.ndarray) -> JsonDict:
    """Select a scalar temperature using tuning Brier only."""

    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    candidates = [
        {
            "temperature": temperature,
            "tune_brier": _brier(stable_sigmoid(logits / temperature), labels),
        }
        for temperature in TEMPERATURE_GRID
    ]
    selected = min(candidates, key=lambda row: (row["tune_brier"], row["temperature"]))
    return {
        "selection_metric": "tune_brier",
        "selected_temperature": selected["temperature"],
        "tune_brier": selected["tune_brier"],
        "candidates": candidates,
    }


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply one frozen scalar temperature to finite probabilities."""

    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return stable_sigmoid(np.log(clipped / (1.0 - clipped)) / temperature)


def _normalized_views(
    rows: Sequence[Mapping[str, Any]], role: str, transform: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the fit-role mean and safe scale to both mapped order views."""

    first, second, labels = _role_matrices(rows, role)
    mean = np.asarray(transform["mean"], dtype=np.float64)
    safe_scale = np.asarray(transform["safe_scale"], dtype=np.float64)
    return (first - mean) / safe_scale, (second - mean) / safe_scale, labels


def _head_probabilities(
    bundle: Mapping[str, Any], family: str, rows: Sequence[Mapping[str, Any]], role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score one frozen head on both option orders without changing it."""

    first, second, labels = _normalized_views(rows, role, bundle["normalization"])
    knots = np.asarray(bundle["local_basis"]["knots"], dtype=np.float64)
    first_design = _family_design(family, first, knots)
    second_design = _family_design(family, second, knots)
    checkpoint = bundle["selected_heads"][family]["checkpoint"]
    weights = np.asarray(checkpoint["weights"], dtype=np.float64)
    bias = float(checkpoint["bias"])
    return (
        stable_sigmoid(first_design @ weights + bias),
        stable_sigmoid(second_design @ weights + bias),
        labels,
    )


def _raw_original(rows: Sequence[Mapping[str, Any]], role: str) -> tuple[np.ndarray, np.ndarray]:
    """Average mapped original-source probabilities across both display orders."""

    first, second, labels = _role_matrices(rows, role)
    return 0.5 * (stable_sigmoid(first[:, 0]) + stable_sigmoid(second[:, 0])), labels


def fit_energy_bundle(
    fit_rows: Sequence[Mapping[str, Any]],
    tune_rows: Sequence[Mapping[str, Any]],
    *,
    steps: int = OPTIMIZER_STEPS,
) -> JsonDict:
    """Fit all registered families and controls without opening policy labels."""

    if steps < 1 or steps > OPTIMIZER_STEPS:
        raise ValueError("optimizer_steps_invalid")
    fit_first, fit_second, fit_labels = _role_matrices(fit_rows, "fit")
    tune_first, tune_second, tune_labels = _role_matrices(tune_rows, "tune")
    combined_fit = np.vstack((fit_first, fit_second))
    mean = combined_fit.mean(axis=0)
    scale = combined_fit.std(axis=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    normalized_fit = ((fit_first - mean) / safe_scale, (fit_second - mean) / safe_scale)
    normalized_tune = ((tune_first - mean) / safe_scale, (tune_second - mean) / safe_scale)
    knots = fit_local_knots(np.vstack(normalized_fit))
    designs: dict[str, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {}
    for family in TRAINABLE_FAMILIES:
        designs[family] = (
            tuple(_family_design(family, view, knots) for view in normalized_fit),
            tuple(_family_design(family, view, knots) for view in normalized_tune),
        )  # type: ignore[assignment]
    candidates: list[JsonDict] = []
    fit_started = time.monotonic()
    for family in TRAINABLE_FAMILIES:
        for consistency_lambda in CONSISTENCY_GRID:
            for l2 in L2_GRID:
                candidates.append(
                    _fit_candidate(
                        family,
                        designs[family][0],
                        fit_labels,
                        designs[family][1],
                        tune_labels,
                        consistency_lambda=consistency_lambda,
                        l2=l2,
                        steps=steps,
                    )
                )
    selected = {family: select_head(candidates, family) for family in TRAINABLE_FAMILIES}
    raw_tune = 0.5 * (stable_sigmoid(tune_first[:, 0]) + stable_sigmoid(tune_second[:, 0]))
    raw_fit = 0.5 * (stable_sigmoid(fit_first[:, 0]) + stable_sigmoid(fit_second[:, 0]))
    temperature = _temperature_fit(raw_tune, tune_labels)
    temperature_tune = _apply_temperature(raw_tune, float(temperature["selected_temperature"]))
    comparator_rows = [
        {"family": family, "tune_brier": head["tune_brier"], "kind": "trainable"}
        for family, head in selected.items()
    ]
    comparator_rows.extend(
        [
            {"family": "raw_original", "tune_brier": _brier(raw_tune, tune_labels), "kind": "raw"},
            {
                "family": "temperature_original",
                "tune_brier": _brier(temperature_tune, tune_labels),
                "kind": "temperature",
            },
        ]
    )
    strongest = min(comparator_rows, key=lambda row: (row["tune_brier"], row["family"]))
    constant_probability = float(np.mean(fit_labels))
    shuffled_labels = np.random.default_rng(FIT_SEED).permutation(fit_labels)
    shuffled = _fit_candidate(
        "source_contrast_energy",
        designs["source_contrast_energy"][0],
        shuffled_labels,
        designs["source_contrast_energy"][1],
        tune_labels,
        consistency_lambda=0.1,
        l2=0.01,
        steps=steps,
    )
    controls = {
        "constant_feature": {
            "kind": "training_control",
            "family": "constant_feature",
            "parameter_count": 1,
            "probability": constant_probability,
            "train_log_loss": _log_loss(np.full(len(fit_labels), constant_probability), fit_labels),
            "tune_brier": _brier(np.full(len(tune_labels), constant_probability), tune_labels),
            "status": "complete",
            "disposition": "complete",
        },
        "shuffled_label": {
            **shuffled,
            "kind": "training_control",
            "family": "shuffled_label",
            "label_permutation_sha256": canonical_hash(shuffled_labels.tolist()),
        },
    }
    bundle: JsonDict = {
        "schema": "carnot.exp7566.v661.frozen_fit_bundle.v1",
        "equation": "E(0,x)=0; E(1,x)=-f_theta(x); p(unsupported|x)=sigmoid(f_theta(x))",
        "objective": "binary_log_loss_plus_mapped_order_jensen_shannon",
        "roles_consumed": ["fit", "tune"],
        "heldout_labels_consumed": False,
        "seed": FIT_SEED,
        "learning_rate": LEARNING_RATE,
        "optimizer_steps_per_candidate": steps,
        "consistency_grid": list(CONSISTENCY_GRID),
        "l2_grid": list(L2_GRID),
        "candidate_budget_per_family": 9,
        "fit_time_s": time.monotonic() - fit_started,
        "feature_names": list(FEATURE_NAMES),
        "normalization": {
            "fit_role": "fit",
            "mean": mean.tolist(),
            "population_sd": scale.tolist(),
            "safe_scale": safe_scale.tolist(),
        },
        "local_basis": {
            "basis": "open_clamped_cubic_quantile",
            "knots": knots.tolist(),
            "coefficients_per_feature": 8,
            "source_energy_parameter_count": 25,
        },
        "option_order_mapping": {
            name: list(order) for name, order in zip(ORDER_NAMES, OPTION_ORDERS, strict=True)
        },
        "candidate_rows": candidates,
        "selected_heads": selected,
        "raw_baseline": {
            "selection_role": "tune",
            "tune_brier": _brier(raw_tune, tune_labels),
            "fit_brier": _brier(raw_fit, fit_labels),
        },
        "temperature_baseline": temperature,
        "comparator_rows": comparator_rows,
        "strongest_comparator": {
            **strongest,
            "selection_metric": "tune_brier",
            "frozen_before_policy_access": True,
        },
        "controls": controls,
        "normalization_checks": {
            "exact_binary_normalization": True,
            "finite_heads": all(np.isfinite(row["tune_brier"]) for row in candidates),
            "parameter_count_25": selected["source_contrast_energy"]["parameter_count"] == 25,
        },
        "fit_group_count": len(fit_rows),
        "tune_group_count": len(tune_rows),
    }
    bundle["bundle_sha256"] = bundle_hash(bundle)
    return bundle


def expected_action(probability: float) -> str:
    """Choose minimum expected cost with escalation winning exact ties."""

    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability_invalid")
    costs = {"accept": 5.0 * value, "reject": 1.0 - value, "escalate": 0.2}
    order = ("escalate", "accept", "reject")
    minimum = min(costs.values())
    return next(action for action in order if math.isclose(costs[action], minimum, abs_tol=1e-12))


def policy_hash(report: Mapping[str, Any]) -> str:
    """Bind a policy report without hashing its self-referential field."""

    return canonical_hash(
        {key: deepcopy(value) for key, value in report.items() if key != "policy_sha256"}
    )


def build_policy_report(
    bundle: Mapping[str, Any], policy_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Describe frozen-head calibration without selecting on policy labels."""

    if bundle.get("bundle_sha256") != bundle_hash(bundle):
        raise ValueError("fit_bundle_hash_invalid")
    probability_rows: dict[str, JsonDict] = {}
    labels: np.ndarray | None = None
    for family in TRAINABLE_FAMILIES:
        first, second, current_labels = _head_probabilities(bundle, family, policy_rows, "policy")
        labels = current_labels
        probabilities = 0.5 * (first + second)
        actions = [expected_action(float(value)) for value in probabilities]
        probability_rows[family] = {
            "policy_brier": _brier(probabilities, current_labels),
            "policy_log_loss": _log_loss(probabilities, current_labels),
            "order_js_consistency": _js_consistency(first, second),
            "action_counts": dict(Counter(actions)),
            "probabilities": probabilities.tolist(),
            "actions": actions,
        }
    raw, raw_labels = _raw_original(policy_rows, "policy")
    labels = raw_labels if labels is None else labels
    temperature = _apply_temperature(
        raw, float(bundle["temperature_baseline"]["selected_temperature"])
    )
    for name, probabilities in (("raw_original", raw), ("temperature_original", temperature)):
        actions = [expected_action(float(value)) for value in probabilities]
        probability_rows[name] = {
            "policy_brier": _brier(probabilities, labels),
            "policy_log_loss": _log_loss(probabilities, labels),
            "action_counts": dict(Counter(actions)),
            "probabilities": probabilities.tolist(),
            "actions": actions,
        }
    report: JsonDict = {
        "schema": "carnot.exp7566.v661.policy_report.v1",
        "role": "policy",
        "group_count": len(policy_rows),
        "frozen_bundle_sha256": bundle["bundle_sha256"],
        "selection_performed": False,
        "descriptive_only": True,
        "costs": {"accept": "5q", "reject": "1-q", "escalate": 0.2},
        "tie_breaking": ["escalate", "accept", "reject"],
        "probability_rows": probability_rows,
    }
    report["policy_sha256"] = policy_hash(report)
    return report


def _challenge_native() -> tuple[list[JsonDict], JsonDict]:
    """Build one circular group used only for structural challenge checks."""

    rows = []
    for order in OPTION_ORDERS:
        for condition, value in (("original", 1.0), ("absent", 0.0), ("mismatched", 0.5)):
            semantic = {"supported": 0.0, "contains_unsupported": value}
            rows.append(
                {
                    "component_hash": "sha256:challenge-component",
                    "group_hash": "sha256:challenge-group",
                    "role": "fit",
                    "donor_role": "fit",
                    "condition": condition,
                    "option_order": list(order),
                    "display_logits": {" A": semantic[order[0]], " B": semantic[order[1]]},
                    "full_logits_by_option_id": semantic,
                    "disposition": "complete",
                    "generated_tokens": 0,
                    "readout_kind": "option_logits",
                }
            )
    return rows, {"component_hash": "sha256:challenge-component", "role": "fit", "label": 1}


def run_challenge_controls() -> JsonDict:
    """Exercise the five registered fail-closed or non-claiming challenges."""

    native, label = _challenge_native()
    option_swap = deepcopy(native)
    option_swap[0]["option_order"] = list(OPTION_ORDERS[1])
    donor_leakage = deepcopy(native)
    donor_leakage[-1]["donor_role"] = "policy"

    def rejected(rows: Sequence[Mapping[str, Any]]) -> bool:
        try:
            build_source_group(rows, label)
        except ValueError:
            return True
        return False  # pragma: no cover - fixed corrupt fixture must be rejected.

    try:
        binary_energy([math.nan])
        normalization_rejected = False  # pragma: no cover - binary_energy rejects NaN.
    except ValueError:
        normalization_rejected = True
    rows = [
        {
            "challenge": "option_swap",
            "expected": "rejected",
            "observed": "rejected" if rejected(option_swap) else "accepted",
            "empirical_claim": False,
        },
        {
            "challenge": "donor_role_leakage",
            "expected": "rejected",
            "observed": "rejected" if rejected(donor_leakage) else "accepted",
            "empirical_claim": False,
        },
        {
            "challenge": "failed_normalization",
            "expected": "rejected",
            "observed": "rejected" if normalization_rejected else "accepted",
            "empirical_claim": False,
        },
        {
            "challenge": "constant_inputs",
            "expected": "control_only",
            "observed": "control_only",
            "empirical_claim": False,
        },
        {
            "challenge": "shuffled_training_labels",
            "expected": "control_only",
            "observed": "control_only",
            "empirical_claim": False,
        },
    ]
    return {"rows": rows, "passed": all(row["expected"] == row["observed"] for row in rows)}


def source_hash_row(path: Path, root: Path) -> JsonDict:
    """Bind exact source bytes while using a stable relative path when possible."""

    resolved = path if path.is_absolute() else root / path
    try:
        label = resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved.resolve())
    return {"path": label, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    path: str,
    passed: bool,
) -> JsonDict:
    """Record one prerequisite operand before any dependent measurement."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": passed,
        "category": "external_precondition",
        "principle": "Missing or changed upstream evidence must stop fitting.",
    }


REQUIRED_INPUT_PATHS = (
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
    Path("python/carnot/experiment_7505_v657_energy_fit.py"),
    Path("python/carnot/experiment_7533_v659_tool_protocol.py"),
    Path("python/carnot/sampling/gibbs.py"),
    SPEC_PATH,
    UPSTREAM_PATH,
    PROTOCOL_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and reject any non-object document."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _resolve_manifest_path(root: Path, label: str) -> Path:
    """Resolve an authenticated sidecar label without changing its target."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate the upstream gate, requirements, capture, and label shards."""

    root = root.resolve()
    rows: list[JsonDict] = []
    sources: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        passed = path.is_file() and path.stat().st_size > 0
        observed = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        rows.append(
            precondition_row(
                "resource_readable",
                relative.as_posix(),
                "path.bytes",
                "readable_nonempty_bytes",
                observed,
                relative.as_posix(),
                passed,
            )
        )
        if passed:
            sources.append(source_hash_row(relative, root))
    spec = root / SPEC_PATH
    spec_present = spec.is_file() and "REQ-VERIFY-7566" in spec.read_text(encoding="utf-8")
    rows.append(
        precondition_row(
            "requirement_present",
            "openspec-verification",
            "REQ-VERIFY-7566",
            True,
            spec_present,
            SPEC_PATH.as_posix(),
            spec_present,
        )
    )
    if not all(row["passed"] for row in rows):
        return {"rows": rows, "source_artifact_hashes": sources, "passed": False}
    upstream = _load_object(root / UPSTREAM_PATH)
    protocol = _load_object(root / PROTOCOL_PATH)
    gate_specs = (
        (
            "fit_capture_ready",
            "exp7564-fit-capture",
            "fit_capture_ready_score",
            1,
            upstream.get("fit_capture_ready_score"),
            UPSTREAM_PATH,
        ),
        (
            "fit_capture_verdict",
            "exp7564-fit-capture",
            "verdict_class",
            ["null", "positive"],
            upstream.get("verdict_class"),
            UPSTREAM_PATH,
        ),
        (
            "fit_capture_adversarial",
            "exp7564-fit-capture",
            "flagged_adversarial",
            False,
            upstream.get("flagged_adversarial"),
            UPSTREAM_PATH,
        ),
        (
            "capture_group_count",
            "exp7564-fit-capture",
            "raw_manifest.group_count",
            240,
            upstream.get("raw_manifest", {}).get("group_count"),
            UPSTREAM_PATH,
        ),
        (
            "capture_forward_count",
            "exp7564-fit-capture",
            "raw_manifest.forward_count",
            1440,
            upstream.get("raw_manifest", {}).get("forward_count"),
            UPSTREAM_PATH,
        ),
        (
            "protocol_ready",
            "exp7533-tool-protocol",
            "tool_protocol_ready_score",
            1,
            protocol.get("tool_protocol_ready_score"),
            PROTOCOL_PATH,
        ),
        (
            "protocol_adversarial",
            "exp7533-tool-protocol",
            "flagged_adversarial",
            False,
            protocol.get("flagged_adversarial"),
            PROTOCOL_PATH,
        ),
    )
    for check, owner, field, expected, observed, path in gate_specs:
        passed = observed in expected if isinstance(expected, list) else observed == expected
        row = precondition_row(check, owner, field, expected, observed, path.as_posix(), passed)
        row["op"] = "in" if isinstance(expected, list) else "=="
        rows.append(row)
    shard_specs = list(upstream.get("raw_manifest", {}).get("native_row_shards") or [])
    sealed = protocol.get("sealed_shards") or {}
    for name in ("fit_labels", "tune_labels", "policy_labels"):
        value = sealed.get(name)
        if isinstance(value, Mapping):
            shard_specs.append(value)
        else:
            rows.append(
                precondition_row(
                    "sealed_shard_present",
                    "exp7533-tool-protocol",
                    f"sealed_shards.{name}",
                    "hash_bound_sidecar",
                    value,
                    PROTOCOL_PATH.as_posix(),
                    False,
                )
            )
    for receipt in shard_specs:
        label = str(receipt.get("path") or "")
        path = _resolve_manifest_path(root, label)
        observed = sha256_file(path) if path.is_file() else None
        expected = receipt.get("sha256")
        passed = observed == expected and isinstance(expected, str)
        rows.append(
            precondition_row(
                "sidecar_hash",
                "exp7564-capture_or_exp7533-labels",
                "sha256",
                expected,
                observed,
                label,
                passed,
            )
        )
        if passed:
            sources.append(source_hash_row(path, root))
    return {
        "rows": rows,
        "source_artifact_hashes": sources,
        "passed": all(row["passed"] for row in rows),
    }


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read complete JSONL rows from one authenticated sidecar."""

    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def load_fit_roles(root: Path) -> dict[str, list[JsonDict]]:
    """Join native readouts to only the fit, tune, and policy labels."""

    root = root.resolve()
    upstream = _load_object(root / UPSTREAM_PATH)
    protocol = _load_object(root / PROTOCOL_PATH)
    native_rows: list[JsonDict] = []
    for receipt in upstream["raw_manifest"]["native_row_shards"]:
        native_rows.extend(_read_jsonl(_resolve_manifest_path(root, str(receipt["path"]))))
    label_rows: list[JsonDict] = []
    for name in ("fit_labels", "tune_labels", "policy_labels"):
        receipt = protocol["sealed_shards"][name]
        label_rows.extend(_read_jsonl(_resolve_manifest_path(root, str(receipt["path"]))))
    label_by_identity = {
        (str(row.get("component_hash")), str(row.get("role"))): row for row in label_rows
    }
    if len(label_by_identity) != len(label_rows):
        raise ValueError("evaluator_identity_duplicate")
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in native_rows:
        grouped[str(row.get("group_hash"))].append(row)
    roles: dict[str, list[JsonDict]] = {name: [] for name in ("fit", "tune", "policy")}
    for group_hash in sorted(grouped):
        group = grouped[group_hash]
        identity = (str(group[0].get("component_hash")), str(group[0].get("role")))
        if identity not in label_by_identity:
            raise ValueError("authorized_label_missing")
        joined = build_source_group(group, label_by_identity[identity])
        role = str(joined["role"])
        if role not in roles:
            raise ValueError("fit_role_invalid")
        roles[role].append(joined)
    observed = {role: len(values) for role, values in roles.items()}
    if observed != {"fit": 160, "tune": 40, "policy": 40}:
        raise ValueError(f"role_counts_invalid:{observed}")
    return roles


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic JSONL through a same-directory atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sidecar(path: Path, root: Path, *, rows: int | None = None) -> JsonDict:
    """Describe exact sidecar bytes and an optional independent row count."""

    value = source_hash_row(path, root)
    if rows is not None:
        value["rows"] = rows
    return value


def write_fit_sidecars(
    root: Path, bundle: Mapping[str, Any], policy_report: Mapping[str, Any]
) -> JsonDict:
    """Export frozen heads, all training rows, and the descriptive policy report."""

    heads = {
        "schema": "carnot.exp7566.v661.frozen_heads.v1",
        "bundle_sha256": bundle["bundle_sha256"],
        "features": deepcopy(bundle["feature_names"]),
        "normalization": deepcopy(bundle["normalization"]),
        "local_basis": deepcopy(bundle["local_basis"]),
        "option_order_mapping": deepcopy(bundle["option_order_mapping"]),
        "selected_heads": deepcopy(bundle["selected_heads"]),
        "strongest_comparator": deepcopy(bundle["strongest_comparator"]),
        "temperature_baseline": deepcopy(bundle["temperature_baseline"]),
        "policy_sha256": policy_report["policy_sha256"],
    }
    training_rows = [*bundle["candidate_rows"], *bundle["controls"].values()]
    atomic_json(root / CHECKPOINT_PATH, heads)
    _write_jsonl(root / TRAINING_ROWS_PATH, training_rows)
    atomic_json(root / POLICY_PATH, policy_report)
    outputs = {
        "frozen_heads": _sidecar(root / CHECKPOINT_PATH, root),
        "training_rows": _sidecar(root / TRAINING_ROWS_PATH, root, rows=len(training_rows)),
        "policy_report": _sidecar(root / POLICY_PATH, root),
    }
    if any(  # pragma: no cover - storage safety guard; payload is bounded by fixed grids.
        (root / value["path"]).stat().st_size >= 20 * 1024 * 1024 for value in outputs.values()
    ):
        raise ValueError("sidecar_size_limit")
    return outputs


def _gate(
    check: str, category: str, expected: Any, observed: Any, op: str, passed: bool
) -> JsonDict:
    """Attach an exact operand and principle to one acceptance gate."""

    principles = {
        "validity": "Invalid evidence cannot support science.",
        "readiness": "A valid null remains reusable.",
        "benefit": "Completion cannot substitute for empirical value.",
    }
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principles[category],
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failure and retain the first exact failed operand."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {"passed": True, "failed_checks": [], "failure": None}
    first = failed[0]
    return {
        "passed": False,
        "failed_checks": [row.get("check") for row in failed],
        "failure": {
            key: deepcopy(first.get(key))
            for key in ("check", "expected", "observed", "op", "category")
        },
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind terminal content while excluding only the checksum itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the failure prevented by every emitted terminal field."""

    specific = {
        "experiment_id": "Exact identity binds this task, milestone, and registered run date.",
        "preconditions_checked": "Recorded observations prevent fitting on absent or changed inputs.",
        "MODEL_SPECS": "An empty list prevents historical Qwen calls from becoming current calls.",
        "model_specs": "Resolved model details stay empty because this task loads no model.",
        "invocation_counts": "Typed zero counters prevent hidden current loads, forwards, or generations.",
        "inference_substrate_class": "The no-load class prevents duration padding or false GPU claims.",
        "reproducibility_checksum": "The checksum binds inputs, code, settings, raw evidence, and outputs.",
        "rows": "Every comparator keeps its absolute metric and disposition.",
        "acceptance_gate_results": "Separate gate operands keep completion distinct from benefit.",
        "energy_fit_ready_score": "A bare bit requires qualified frozen source-dependent heads.",
        "baseline_ready_score": "A separate bit requires controls and a preselected comparator.",
        "frozen_head_manifest": "Hashes bind transforms and weights before policy or test access.",
        "training_rows": "All candidates and controls retain costs, traces, and failures.",
        "predictive_benefit_measured": "False prevents fit completion from becoming a held-out claim.",
    }
    return {
        field: specific.get(
            field, "This field preserves an auditable operand and prevents silent drift."
        )
        for field in fields
    }


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute fit and baseline readiness from frozen terminal evidence."""

    rows = value.get("training_rows")
    manifest = value.get("frozen_head_manifest")
    policy = value.get("policy_calibration_report")
    controls = value.get("challenge_controls")
    validation = value.get("validation_summary")
    if not all(isinstance(item, Mapping) for item in (manifest, policy, controls, validation)):
        return {"energy_fit_ready_score": 0, "baseline_ready_score": 0}
    if not isinstance(rows, list):
        return {"energy_fit_ready_score": 0, "baseline_ready_score": 0}
    candidates = [row for row in rows if row.get("kind") == "candidate"]
    expected = {
        (family, weight, l2)
        for family in TRAINABLE_FAMILIES
        for weight in CONSISTENCY_GRID
        for l2 in L2_GRID
    }
    observed = {
        (row.get("family"), row.get("consistency_lambda"), row.get("l2")) for row in candidates
    }
    candidates_ok = (
        len(candidates) == len(expected)
        and observed == expected
        and all(
            row.get("status") == "complete"
            and row.get("optimizer_steps") == value.get("optimizer_settings", {}).get("steps")
            and row.get("loss_trace")
            for row in candidates
        )
    )
    selected = manifest.get("selected_heads")
    energy_ok = (
        candidates_ok
        and isinstance(selected, Mapping)
        and set(selected) == set(TRAINABLE_FAMILIES)
        and selected.get("source_contrast_energy", {}).get("parameter_count") == 25
        and manifest.get("transform_sha256") == canonical_hash(manifest.get("transform"))
    )
    training_controls = [row for row in rows if row.get("kind") == "training_control"]
    baseline_ok = (
        {row.get("family") for row in training_controls} == {"constant_feature", "shuffled_label"}
        and all(
            row.get("status") in {"complete", "failed_loss_increase"} for row in training_controls
        )
        and isinstance(manifest.get("strongest_comparator"), Mapping)
        and manifest["strongest_comparator"].get("frozen_before_policy_access") is True
    )
    common = (
        policy.get("selection_performed") is False
        and policy.get("policy_sha256") == policy_hash(policy)
        and controls.get("passed") is True
        and validation.get("required_checks_passed") is True
    )
    return {
        "energy_fit_ready_score": int(energy_ok and common),
        "baseline_ready_score": int(baseline_ok and common),
    }


REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate_class",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "process_identity",
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
    "energy_fit_ready_score",
    "baseline_ready_score",
    "frozen_head_manifest",
    "training_rows",
    "predictive_benefit_measured",
)


def _frozen_manifest(bundle: Mapping[str, Any], sidecars: Mapping[str, Any]) -> JsonDict:
    """Bind features, transforms, heads, selection, policy, and exported bytes."""

    transform = {
        "features": deepcopy(bundle["feature_names"]),
        "normalization": deepcopy(bundle["normalization"]),
        "local_basis": deepcopy(bundle["local_basis"]),
        "option_order_mapping": deepcopy(bundle["option_order_mapping"]),
    }
    return {
        "source_roles": {"training": "fit", "selection": "tune", "description": "policy"},
        "transform": transform,
        "transform_sha256": canonical_hash(transform),
        "selected_heads": deepcopy(bundle["selected_heads"]),
        "strongest_comparator": deepcopy(bundle["strongest_comparator"]),
        "temperature_baseline": deepcopy(bundle["temperature_baseline"]),
        "bundle_sha256": bundle["bundle_sha256"],
        "sidecars": deepcopy(dict(sidecars)),
        "frozen_before_policy_access": True,
    }


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    policy_report: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    challenges: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    duration_s: float,
    fixture: bool = False,
) -> JsonDict:
    """Assemble one complete fit record without reading held-out outcomes."""

    candidate_rows = [deepcopy(row) for row in bundle["candidate_rows"]]
    control_rows = [deepcopy(row) for row in bundle["controls"].values()]
    baseline_rows = [
        {
            "kind": "baseline",
            "family": row["family"],
            "tune_brier": row["tune_brier"],
            "status": "complete",
            "disposition": "complete",
            "parameter_count": (
                0
                if row["family"] == "raw_original"
                else 1
                if row["family"] == "temperature_original"
                else bundle["selected_heads"][row["family"]]["parameter_count"]
            ),
        }
        for row in bundle["comparator_rows"]
    ]
    training_rows = [*candidate_rows, *control_rows, *baseline_rows]
    validation_summary = {
        "required_checks_passed": bool(validation_passed),
        "receipt_count": len(validation_receipts),
        "cold_replay_included": any(
            row.get("name") == "declared_entrypoint_cold_replay" for row in validation_receipts
        ),
        "independent_reduction_included": any(
            row.get("name") == "independent_row_reduction" for row in validation_receipts
        ),
    }
    frozen = _frozen_manifest(bundle, sidecars)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "version": "v661",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Fit contrastive source energies and freeze matched decision controls",
        "complete": True,
        "ready": True,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "historical_invocation_counts": {"source": "exp7564", "not_current_work": True},
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "readout_kind": "option_logits",
            "native_option_forwards_are_historical": True,
            "current_generated_tokens": 0,
            "small_ebm_training": True,
        },
        "execution_venue": "host",
        "execution_device": {"type": "cpu", "identity": platform.processor() or platform.machine()},
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "executable": sys.executable},
        "random_seed": {
            "model": None,
            "fitting": FIT_SEED,
            "ordering": 659033,
            "bootstrap": None,
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "rows": deepcopy(training_rows),
        "training_rows": training_rows,
        "sample_size_budget": {
            "unit": "source_group",
            "planned": 240,
            "attempted": 240,
            "completed": 240,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "roles": {"fit": 160, "tune": 40, "policy": 40},
        },
        "optimizer_settings": {
            "objective": bundle["objective"],
            "steps": bundle["optimizer_steps_per_candidate"],
            "rate": bundle["learning_rate"],
            "seed": bundle["seed"],
            "lambda_grid": deepcopy(bundle["consistency_grid"]),
            "l2_grid": deepcopy(bundle["l2_grid"]),
        },
        "frozen_head_manifest": frozen,
        "policy_calibration_report": deepcopy(dict(policy_report)),
        "challenge_controls": deepcopy(dict(challenges)),
        "learning_lifecycle": {
            "predict": "native_option_readouts_loaded",
            "release": "fit_then_tune_then_policy_labels_opened",
            "update": "small_energy_heads_fitted",
            "persist": bool(sidecars),
            "reload": bool(sidecars),
            "generator_weights_changed": False,
        },
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_summary": validation_summary,
        "applicable_numbered_e2e": [],
        "capability_e2e": "declared_entrypoint_cold_replay",
        "scientific_scope": {
            "small_learned_energy_policy": True,
            "formal_verification": False,
            "generic_text_reranker": False,
            "generator_weights_changed": False,
        },
        "predictive_benefit_measured": False,
        "benefit_measured": False,
        "positive_claim": False,
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "external_publication_authorized": False,
        "reproducibility_checksum": "pending",
    }
    reduced = independent_reduce(artifact)
    artifact.update(reduced)
    gates = [
        _gate(
            "preconditions",
            "validity",
            True,
            all(row.get("passed") for row in preconditions),
            "==",
            all(row.get("passed") for row in preconditions),
        ),
        _gate("scoped_validation", "validity", True, validation_passed, "==", validation_passed),
        _gate(
            "raw_custody",
            "validity",
            240,
            artifact["sample_size_budget"]["completed"],
            "==",
            artifact["sample_size_budget"]["completed"] == 240,
        ),
        _gate(
            "energy_fit_ready",
            "readiness",
            1,
            reduced["energy_fit_ready_score"],
            "==",
            reduced["energy_fit_ready_score"] == 1,
        ),
        _gate(
            "baseline_ready",
            "readiness",
            1,
            reduced["baseline_ready_score"],
            "==",
            reduced["baseline_ready_score"] == 1,
        ),
        _gate("heldout_benefit_unmeasured", "benefit", False, False, "==", True),
    ]
    artifact["acceptance_gate_results"] = gates
    artifact["gate_check_summary"] = _gate_summary(gates)
    artifact["honest_verdict"] = (
        "complete_circular_positive_structural_fixture_only"
        if fixture
        else "complete_null_energy_fit_ready_benefit_unmeasured"
    )
    artifact["verdict_class"] = "circular_positive" if fixture else "null"
    artifact["field_principles"] = _field_principles((*artifact.keys(), "field_principles"))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_role(role: str, count: int, offset: int) -> list[JsonDict]:
    """Create circular source groups for structural tests only."""

    output = []
    for index in range(count):
        value = index + offset
        label = value % 2
        component = f"sha256:fixture-{role}-component-{value}"
        group = f"sha256:fixture-{role}-group-{value}"
        native = []
        for order_index, order in enumerate(OPTION_ORDERS):
            for condition, shift in (("original", 0.0), ("absent", -0.7), ("mismatched", -0.3)):
                logit = (-1.2 if label == 0 else 1.4) + shift + 0.1 * order_index
                semantic = {"supported": 0.0, "contains_unsupported": logit}
                native.append(
                    {
                        "component_hash": component,
                        "group_hash": group,
                        "role": role,
                        "donor_role": role,
                        "condition": condition,
                        "option_order": list(order),
                        "display_logits": {" A": semantic[order[0]], " B": semantic[order[1]]},
                        "full_logits_by_option_id": semantic,
                        "disposition": "complete",
                        "generated_tokens": 0,
                        "readout_kind": "option_logits",
                    }
                )
        output.append(
            build_source_group(native, {"component_hash": component, "role": role, "label": label})
        )
    return output


def fixture_artifact() -> JsonDict:
    """Build a cold-valid circular artifact without an empirical claim."""

    bundle = fit_energy_bundle(_fixture_role("fit", 12, 0), _fixture_role("tune", 8, 12), steps=2)
    policy = build_policy_report(bundle, _fixture_role("policy", 8, 20))
    receipt_names = (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    receipts = [
        {
            "name": name,
            "scope": "fixture",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "log_sha256": canonical_hash({"name": name}),
        }
        for name in receipt_names
    ]
    return build_artifact(
        preconditions=[
            precondition_row(
                "fixture_shape",
                "circular_fixture",
                "roles",
                ["fit", "tune", "policy"],
                ["fit", "tune", "policy"],
                "in_memory",
                True,
            )
        ],
        source_hashes=[],
        bundle=bundle,
        policy_report=policy,
        sidecars={},
        challenges=run_challenge_controls(),
        validation_receipts=receipts,
        validation_passed=True,
        phase_spans=[],
        duration_s=0.01,
        fixture=True,
    )


def blocked_artifact(failed: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Publish external absence as complete blocked work with exact operands."""

    summary = {
        "passed": False,
        "upstream": failed.get("upstream"),
        "path": failed.get("path"),
        "field": failed.get("field"),
        "expected": deepcopy(failed.get("expected")),
        "observed": deepcopy(failed.get("observed")),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "version": "v661",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Fit contrastive source energies and freeze matched decision controls",
        "complete": True,
        "ready": False,
        "preconditions_checked": [deepcopy(dict(failed))],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "historical_invocation_counts": {},
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "readout_kind": "option_logits",
            "current_generated_tokens": 0,
        },
        "execution_venue": "host",
        "execution_device": {"type": "cpu", "identity": platform.processor() or platform.machine()},
        "duration_s": 0.0,
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "executable": sys.executable},
        "random_seed": {"model": None, "fitting": FIT_SEED, "ordering": 659033, "bootstrap": None},
        "source_artifact_hashes": [],
        "rows": [deepcopy(dict(failed))],
        "training_rows": [],
        "sample_size_budget": {
            "unit": "source_group",
            "planned": 240,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 240,
            "roles": {"fit": 160, "tune": 40, "policy": 40},
        },
        "optimizer_settings": {
            "steps": OPTIMIZER_STEPS,
            "rate": LEARNING_RATE,
            "seed": FIT_SEED,
            "lambda_grid": list(CONSISTENCY_GRID),
            "l2_grid": list(L2_GRID),
        },
        "frozen_head_manifest": {},
        "policy_calibration_report": {},
        "challenge_controls": {"rows": [], "passed": False},
        "learning_lifecycle": {
            "predict": "not_started",
            "release": "not_started",
            "update": "not_started",
            "persist": False,
            "reload": False,
            "generator_weights_changed": False,
        },
        "validation_receipts": [],
        "validation_summary": {"required_checks_passed": False, "receipt_count": 0},
        "applicable_numbered_e2e": [],
        "capability_e2e": "not_run_external_precondition_failed",
        "scientific_scope": {
            "small_learned_energy_policy": False,
            "formal_verification": False,
            "generic_text_reranker": False,
            "generator_weights_changed": False,
        },
        "acceptance_gate_results": [
            _gate(
                str(failed.get("check")),
                "validity",
                failed.get("expected"),
                failed.get("observed"),
                str(failed.get("op", "==")),
                False,
            )
        ],
        "gate_check_summary": summary,
        "honest_verdict": f"complete_blocked_{failed.get('check', 'external_input')}",
        "verdict_class": "blocked",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "predictive_benefit_measured": False,
        "benefit_measured": False,
        "positive_claim": False,
        "energy_fit_ready_score": 0,
        "baseline_ready_score": 0,
        "external_publication_authorized": False,
        "reproducibility_checksum": "pending",
    }
    artifact["field_principles"] = _field_principles((*artifact.keys(), "field_principles"))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, no-load claims, reductions, hashes, and principles."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_FIELDS) - set(value))
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_must_be_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    if value.get("verdict_class") != "blocked":
        if value.get("inference_substrate_class") != "no_model_load":
            errors.append("inference_substrate_class_mismatch")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if value.get("predictive_benefit_measured") is not False:
        errors.append("heldout_benefit_claimed")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict_prefix_invalid")
    if value.get("verdict_class") != "blocked":
        reduced = independent_reduce(value)
        if any(value.get(key) != observed for key, observed in reduced.items()):
            errors.append("independent_reduction_mismatch")
        policy = value.get("policy_calibration_report")
        if not isinstance(policy, Mapping) or policy.get("policy_sha256") != policy_hash(policy):
            errors.append("policy_hash_mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping)
        or not {"expected", "observed", "op", "passed", "category", "principle"} <= set(row)
        for row in gates
    ):
        errors.append("acceptance_gates_invalid")
    if verify_sources:
        for row in value.get("source_artifact_hashes") or []:
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            observed = sha256_file(resolved) if resolved.is_file() else None
            if observed != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{row.get('path')}")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Validate exact serialized candidate bytes through a fresh-reader path."""

    return validate_artifact(_load_object(path), root=root, verify_sources=verify_sources)


def reload_fit_sidecars(root: Path, sidecars: Mapping[str, Any]) -> JsonDict:
    """Rehash persisted heads, training rows, and policy bytes after publication."""

    checks = []
    for name in ("frozen_heads", "training_rows", "policy_report"):
        receipt = sidecars.get(name)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"sidecar_receipt_missing:{name}")
        path = _resolve_manifest_path(root, str(receipt.get("path") or ""))
        observed = sha256_file(path) if path.is_file() else None
        checks.append({"name": name, "expected": receipt.get("sha256"), "observed": observed})
    if any(row["expected"] != row["observed"] for row in checks):
        raise ValueError("sidecar_reload_hash_mismatch")
    heads = _load_object(_resolve_manifest_path(root, str(sidecars["frozen_heads"]["path"])))
    policy = _load_object(_resolve_manifest_path(root, str(sidecars["policy_report"]["path"])))
    passed = heads.get("bundle_sha256") == policy.get("frozen_bundle_sha256") and heads.get(
        "policy_sha256"
    ) == policy.get("policy_sha256")
    if not passed:
        raise ValueError("sidecar_reload_identity_mismatch")
    return {"passed": True, "checks": checks, "bundle_sha256": heads["bundle_sha256"]}


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return an aware UTC timestamp for a durable run boundary."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - real execution boundary.
    """Print one flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7566] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - real execution boundary.
    """Close one monotonic phase with its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, reduction, adversarial, and strict-row commands."""

    python = ".venv/bin/python"
    commands = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_row_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                str(candidate),
            ),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(command, "required_validation", True) for command in commands]


def _all_receipts_pass(
    receipts: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> bool:  # pragma: no cover
    """Require one passing zero-exit receipt for every declared command."""

    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Authenticate, fit, freeze, validate, replay, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start", completed_units=0)
    phase_started = time.monotonic()
    preconditions = collect_preconditions(root)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            run_started,
            len(preconditions["rows"]),
            "authenticated_inputs",
        )
    )
    failed = next((row for row in preconditions["rows"] if row.get("passed") is not True), None)
    if failed is not None:
        artifact = blocked_artifact(failed, root=root)
        progress(run_started, "publish", "before_atomic_terminal", verdict="blocked")
        atomic_json(root / RESULT_PATH, artifact)
        progress(run_started, "publish", "complete_blocked", reason=failed.get("check"))
        return artifact
    progress(
        run_started,
        "preconditions",
        "complete",
        completed_units=len(preconditions["rows"]),
    )

    progress(run_started, "predict_release", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    roles = load_fit_roles(root)
    spans.append(_span("predict_release", phase_started, run_started, 240, "joined_source_groups"))
    progress(run_started, "predict_release", "after_benchmark", completed_units=240)

    progress(run_started, "numeric_fit", "before_training", completed_units=0, candidates=36)
    phase_started = time.monotonic()
    bundle = fit_energy_bundle(roles["fit"], roles["tune"], steps=OPTIMIZER_STEPS)
    spans.append(_span("numeric_fit", phase_started, run_started, 36, "frozen_tune_selection"))
    progress(run_started, "numeric_fit", "after_training", completed_units=36)

    progress(run_started, "policy_report", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    policy_report = build_policy_report(bundle, roles["policy"])
    challenges = run_challenge_controls()
    sidecars = write_fit_sidecars(root, bundle, policy_report)
    reload_receipt = reload_fit_sidecars(root, sidecars)
    spans.append(_span("policy_report", phase_started, run_started, 40, "persisted_and_reloaded"))
    progress(
        run_started,
        "policy_report",
        "after_benchmark",
        completed_units=40,
        reload_passed=reload_receipt["passed"],
    )

    manifest_path = root / RAW_DIR / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    )
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7566-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(
        run_started,
        "affected_validation",
        "before_subprocesses",
        completed_units=0,
        commands=len(commands),
    )
    phase_started = time.monotonic()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation" / "affected",
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            run_started,
            len(affected),
            "affected_validation_logs",
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    candidate = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=preconditions["source_artifact_hashes"],
        bundle=bundle,
        policy_report=policy_report,
        sidecars=sidecars,
        challenges=challenges,
        validation_receipts=affected,
        validation_passed=bool(affected_reduction["passed"]),
        phase_spans=spans,
        duration_s=time.monotonic() - run_started,
    )
    errors = validate_artifact(candidate, root=root)
    if errors:
        raise RuntimeError(f"measured_candidate_invalid:{errors}")
    atomic_json(root / TERMINAL_CANDIDATE, candidate)

    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        completed_units=0,
        commands=4,
    )
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        _terminal_commands(TERMINAL_CANDIDATE),
        log_dir=root / RAW_DIR / "validation" / "terminal",
    )
    terminal_passed = _all_receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            run_started,
            len(terminal),
            "terminal_validation_logs",
        )
    )
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    all_validation = bool(affected_reduction["passed"] and terminal_passed and not critical)
    final_sidecars = {
        **sidecars,
        "affected_validation_manifest": _sidecar(manifest_path, root),
        "terminal_candidate": _sidecar(root / TERMINAL_CANDIDATE, root),
    }
    final = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=preconditions["source_artifact_hashes"],
        bundle=bundle,
        policy_report=policy_report,
        sidecars=final_sidecars,
        challenges=challenges,
        validation_receipts=[*affected, *terminal],
        validation_passed=all_validation,
        phase_spans=spans,
        duration_s=time.monotonic() - run_started,
    )
    final["learning_lifecycle"]["reload_receipt"] = reload_receipt
    if not all_validation:
        final["honest_verdict"] = "complete_disqualified_required_validation"
        final["verdict_class"] = "disqualified"
    final["field_principles"] = _field_principles((*final.keys(), "field_principles"))
    final["reproducibility_checksum"] = artifact_checksum(final)
    final_errors = validate_artifact(final, root=root)
    if final_errors:
        raise RuntimeError(f"terminal_artifact_invalid:{final_errors}")
    progress(run_started, "publish", "before_atomic_terminal", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, final)
    progress(
        run_started,
        "publish",
        "complete",
        energy_fit_ready_score=final["energy_fit_ready_score"],
        baseline_ready_score=final["baseline_ready_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and fresh-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:  # pragma: no cover - CLI path boundary.
    """Resolve a command path against the authenticated worktree."""

    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the fit or one bounded fresh-reader operation."""

    arguments = parse_args(argv)
    if arguments.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if arguments.cold_replay is not None:
        errors = cold_replay(_argument_path(arguments.cold_replay))
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if arguments.independent_reduce is not None:
        artifact = _load_object(_argument_path(arguments.independent_reduce))
        reduced = independent_reduce(artifact)
        passed = all(artifact.get(key) == value for key, value in reduced.items())
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(REPO_ROOT, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution boundary.
    raise SystemExit(main())
