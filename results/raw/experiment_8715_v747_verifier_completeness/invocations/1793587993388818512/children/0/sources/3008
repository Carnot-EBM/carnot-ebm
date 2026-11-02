"""Fit calibrated binary energy heads on the sealed V657 feature interface.

This module trains small CPU heads on cached model features. It does not load
Qwen or open test and online labels. Spec refs: REQ-VERIFY-7505 and
SCENARIO-VERIFY-7505-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from carnot.experiment_7504_v657_evidence_interface import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    read_mode,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7505-v657-energy-fit"
SCHEMA = "carnot.exp7505.v657.energy_fit.v1"
RESULT_PATH = Path("results/experiment_7505_v657_energy_fit.json")
RAW_DIR = Path("results/raw/experiment_7505_v657_energy_fit")
MODULE_PATH = Path("python/carnot/experiment_7505_v657_energy_fit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7505_v657_energy_fit.py")
TEST_PATH = Path("tests/python/test_experiment_7505_v657_energy_fit.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
UPSTREAM_ARTIFACT = Path("results/experiment_7504_v657_evidence_interface.json")
UPSTREAM_RAW = Path("results/raw/experiment_7504_v657_evidence_interface")
FEATURE_PATH = UPSTREAM_RAW / "features.jsonl"
NORMALIZATION_PATH = UPSTREAM_RAW / "training_normalization.json"
ACCESS_PATH = UPSTREAM_RAW / "access_exposure_manifest.json"
EVALUATOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/evaluators.jsonl")
CHECKPOINT_PATH = RAW_DIR / "frozen_checkpoints.json"
CALIBRATION_PATH = RAW_DIR / "calibration_rows.jsonl"
FIT_RECEIPT_PATH = RAW_DIR / "fit_receipt.json"
TERMINAL_CANDIDATE = RAW_DIR / "measured_terminal_candidate.json"

TRAINING_SEEDS = (656_101, 656_102, 656_103, 656_104, 656_105)
REGULARIZATION_GRID = (0.0, 0.001, 0.01)
TEMPERATURE_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
FALSE_ACCEPT_COSTS = (1.0, 5.0, 20.0)
ESCALATION_COSTS = (0.1, 0.5, 1.0)
FALSE_REJECT_COST = 1.0
TRAINABLE_ARMS = ("window_gibbs", "whole_only_gibbs", "identical_ten_feature_logistic")
MAX_OPTIMIZER_STEPS = 200
MAX_OPTIMIZER_SECONDS = 900.0
LEARNING_RATE = 0.03
TIE_ORDER = ("accept", "escalate", "reject")

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "process_identity",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "duration_breakdown_s",
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
    "energy_fit_ready_score",
    "baseline_ready_score",
    "small_ebm_training",
    "checkpoint_manifest",
    "calibration_rows",
    "predictive_benefit_measured",
)

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
    Path("python/carnot/experiment_7481_v655_typed_calibration.py"),
    Path("python/carnot/experiment_7468_v654_residual_learner.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    SPEC_PATH,
    UPSTREAM_ARTIFACT,
    FEATURE_PATH,
    NORMALIZATION_PATH,
    ACCESS_PATH,
    EVALUATOR_PATH,
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
    """Evaluate E(0,x)=0 and E(1,x)=-g(x), then normalize both states."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("logits_nonfinite")
    unsupported = stable_sigmoid(values)
    probabilities = np.column_stack((1.0 - unsupported, unsupported))
    return np.column_stack((np.zeros_like(values), -values)), probabilities


def fit_spline_knots(training_features: Any) -> np.ndarray:
    """Fit ten open cubic knot vectors from training features only."""

    training = np.asarray(training_features, dtype=np.float64)
    if training.ndim != 2 or training.shape[1] != len(FEATURE_NAMES) or len(training) < 2:
        raise ValueError("feature_matrix_shape_invalid")
    if not np.all(np.isfinite(training)):
        raise ValueError("feature_matrix_nonfinite")
    vectors = []
    for column in training.T:
        raw = np.quantile(column, np.linspace(0.0, 1.0, 6), method="linear")
        breaks = _repair_breakpoints(np.asarray(raw, dtype=np.float64))
        vectors.append(
            np.concatenate(
                (
                    np.repeat(breaks[0], DEGREE + 1),
                    breaks[1:-1],
                    np.repeat(breaks[-1], DEGREE + 1),
                )
            )
        )
    return np.asarray(vectors, dtype=np.float64)


def spline_design_matrix(features: Any, knots: Any) -> np.ndarray:
    """Reuse the established local cubic basis for all ten frozen features."""

    matrix = np.asarray(features, dtype=np.float64)
    knot_matrix = np.asarray(knots, dtype=np.float64)
    expected_knots = (len(FEATURE_NAMES), KNOT_VECTOR_SIZE)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature_matrix_shape_invalid")
    if knot_matrix.shape != expected_knots:
        raise ValueError("knots_shape_invalid")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(knot_matrix)):
        raise ValueError("spline_input_nonfinite")
    output = np.empty((len(matrix), len(FEATURE_NAMES) * COEFFICIENTS_PER_INPUT))
    for row_index, row in enumerate(matrix):
        parts = [
            cubic_basis(float(value), knot_matrix[index]).values for index, value in enumerate(row)
        ]
        output[row_index] = np.concatenate(parts)
    return output


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute calibration Brier loss on one fixed role."""

    return float(np.mean(np.square(probabilities - labels)))


def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute finite binary log loss for optimization receipts."""

    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))))


def _fit_linear_head(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    regularization: float,
    steps: int,
) -> JsonDict:
    """Fit one log-loss head with deterministic full-batch Adam updates."""

    if steps < 1 or steps > MAX_OPTIMIZER_STEPS:
        raise ValueError("optimizer_steps_invalid")
    if design.ndim != 2 or len(design) != len(labels) or len(design) == 0:
        raise ValueError("optimizer_input_shape_invalid")
    rng = np.random.default_rng(seed)
    coefficient = rng.normal(0.0, 0.01, size=design.shape[1])
    bias = 0.0
    first_moment = np.zeros(design.shape[1] + 1)
    second_moment = np.zeros(design.shape[1] + 1)
    curve: list[JsonDict] = []
    started = time.monotonic()
    for step in range(steps):
        logits = design @ coefficient + bias
        probabilities = stable_sigmoid(logits)
        loss = _log_loss(probabilities, labels) + 0.5 * regularization * float(
            coefficient @ coefficient
        )
        if step == 0 or step % 10 == 0:
            curve.append({"step": step, "loss": loss})
        residual = probabilities - labels
        gradient = np.concatenate(
            ((design.T @ residual) / len(design) + regularization * coefficient, [residual.mean()])
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9 ** (step + 1))
        corrected_second = second_moment / (1.0 - 0.999 ** (step + 1))
        update = LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        coefficient -= update[:-1]
        bias -= float(update[-1])
    final_probabilities = stable_sigmoid(design @ coefficient + bias)
    final_loss = _log_loss(final_probabilities, labels) + 0.5 * regularization * float(
        coefficient @ coefficient
    )
    curve.append({"step": steps, "loss": final_loss})
    checkpoint = {"coefficient": coefficient.tolist(), "bias": bias}
    finite = bool(np.all(np.isfinite(coefficient)) and math.isfinite(bias))
    stable = finite and all(math.isfinite(float(row["loss"])) for row in curve)
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": canonical_hash(checkpoint),
        "loss_curve": curve,
        "loss_start": curve[0]["loss"],
        "loss_end": curve[-1]["loss"],
        "parameter_count": design.shape[1] + 1,
        "optimizer_steps": steps,
        "fit_time_s": time.monotonic() - started,
        "finite": finite,
        "stable": stable,
    }


def _fit_matrices(
    rows: Sequence[Mapping[str, Any]], expected_role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one fit-visible role and return features, labels, and raw controls."""

    features: list[list[float]] = []
    labels: list[int] = []
    whole: list[float] = []
    maximum: list[float] = []
    for row in rows:
        if row.get("role") != expected_role:
            raise ValueError(f"{expected_role.split('_')[0]}_role_invalid")
        vector = np.asarray(row.get("features"), dtype=np.float64)
        if vector.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
            raise ValueError("feature_vector_invalid")
        label = row.get("label")
        if label not in (0, 1):
            raise ValueError("binary_label_invalid")
        raw_whole = float(row.get("raw_whole_expectation"))
        raw_maximum = float(row.get("raw_max_window_probability"))
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in (raw_whole, raw_maximum)
        ):
            raise ValueError("raw_probability_invalid")
        features.append(vector.tolist())
        labels.append(int(label))
        whole.append(raw_whole)
        maximum.append(raw_maximum)
    if not rows or set(labels) != {0, 1}:
        raise ValueError("binary_support_invalid")
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(whole, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
    )


def _select_temperature(probabilities: np.ndarray, labels: np.ndarray) -> JsonDict:
    """Select the raw whole-score temperature by calibration Brier only."""

    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    candidates = []
    for temperature in TEMPERATURE_GRID:
        calibrated = stable_sigmoid(logits / temperature)
        candidates.append(
            {"temperature": temperature, "calibration_brier": _brier(calibrated, labels)}
        )
    selected = min(candidates, key=lambda row: (row["calibration_brier"], row["temperature"]))
    return {
        "candidates": candidates,
        "selected_temperature": selected["temperature"],
        "calibration_brier": selected["calibration_brier"],
        "selection_metric": "calibration_brier",
    }


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply one frozen scalar temperature to probabilities."""

    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return stable_sigmoid(np.log(clipped / (1.0 - clipped)) / temperature)


def _candidate_row(
    arm: str,
    design_train: np.ndarray,
    labels_train: np.ndarray,
    design_calibration: np.ndarray,
    labels_calibration: np.ndarray,
    *,
    seed: int,
    regularization: float,
    steps: int,
) -> JsonDict:
    """Fit and score one equal-budget candidate without reading held-out rows."""

    try:
        fitted = _fit_linear_head(
            design_train,
            labels_train,
            seed=seed,
            regularization=regularization,
            steps=steps,
        )
        logits = design_calibration @ np.asarray(fitted["checkpoint"]["coefficient"]) + float(
            fitted["checkpoint"]["bias"]
        )
        energies, probabilities = binary_energy(logits)
        normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
        passed = (
            fitted["stable"]
            and normalization_error <= 1e-15
            and fitted["parameter_count"] <= 256
            and np.all(np.isfinite(energies))
        )
        return {
            "arm": arm,
            "seed": seed,
            "regularization": regularization,
            **fitted,
            "calibration_brier": _brier(probabilities[:, 1], labels_calibration),
            "normalization_max_abs_error": normalization_error,
            "status": "complete" if passed else "failed_validation",
            "failure": None if passed else "candidate_numeric_validation_failed",
        }
    except Exception as error:  # pragma: no cover - retained candidate failure path.
        return {
            "arm": arm,
            "seed": seed,
            "regularization": regularization,
            "optimizer_steps": steps,
            "parameter_count": design_train.shape[1] + 1,
            "status": "failed",
            "failure": f"{type(error).__name__}:{error}",
        }


def _selected_head(candidates: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    """Freeze one arm by calibration Brier with deterministic tie-breaking."""

    complete = [
        row for row in candidates if row.get("arm") == arm and row.get("status") == "complete"
    ]
    if len(complete) != len(TRAINING_SEEDS) * len(REGULARIZATION_GRID):
        raise ValueError(f"candidate_grid_incomplete:{arm}")
    selected = min(
        complete,
        key=lambda row: (
            float(row["calibration_brier"]),
            float(row["regularization"]),
            int(row["seed"]),
        ),
    )
    return {
        "arm": arm,
        "seed": selected["seed"],
        "regularization": selected["regularization"],
        "calibration_brier": selected["calibration_brier"],
        "selection_metric": "calibration_brier",
        "checkpoint": deepcopy(selected["checkpoint"]),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "parameter_count": selected["parameter_count"],
    }


def _head_probabilities(head: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    """Score one frozen linear-in-design checkpoint as unsupported probability."""

    checkpoint = head["checkpoint"]
    logits = design @ np.asarray(checkpoint["coefficient"], dtype=np.float64) + float(
        checkpoint["bias"]
    )
    return stable_sigmoid(logits)


def bundle_hash(bundle: Mapping[str, Any]) -> str:
    """Hash a fit bundle without its self-referential identity field."""

    return canonical_hash(
        {key: deepcopy(value) for key, value in bundle.items() if key != "bundle_sha256"}
    )


def fit_energy_bundle(
    training: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    steps: int = 120,
) -> JsonDict:
    """Fit all registered candidates using training and calibration labels only."""

    if steps < 1 or steps > MAX_OPTIMIZER_STEPS:
        raise ValueError("optimizer_steps_invalid")
    train_x, train_y, train_whole, _train_maximum = _fit_matrices(training, "training")
    calibration_x, calibration_y, calibration_whole, calibration_maximum = _fit_matrices(
        calibration, "calibration_tuning"
    )
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    normalized_train = (train_x - mean) / safe_scale
    normalized_calibration = (calibration_x - mean) / safe_scale
    knots = fit_spline_knots(normalized_train)
    designs = {
        "window_gibbs": (
            spline_design_matrix(normalized_train, knots),
            spline_design_matrix(normalized_calibration, knots),
        ),
        "whole_only_gibbs": (normalized_train[:, :1], normalized_calibration[:, :1]),
        "identical_ten_feature_logistic": (normalized_train, normalized_calibration),
    }
    started = time.monotonic()
    candidates: list[JsonDict] = []
    for arm in TRAINABLE_ARMS:
        train_design, calibration_design = designs[arm]
        for seed in TRAINING_SEEDS:
            for regularization in REGULARIZATION_GRID:
                if time.monotonic() - started > MAX_OPTIMIZER_SECONDS:  # pragma: no cover
                    raise TimeoutError("optimizer_time_budget_exceeded")
                candidates.append(
                    _candidate_row(
                        arm,
                        train_design,
                        train_y,
                        calibration_design,
                        calibration_y,
                        seed=seed,
                        regularization=regularization,
                        steps=steps,
                    )
                )
    selected = {arm: _selected_head(candidates, arm) for arm in TRAINABLE_ARMS}
    selected_window = _head_probabilities(selected["window_gibbs"], designs["window_gibbs"][1])
    temperature = _select_temperature(calibration_whole, calibration_y)
    temperature_probabilities = _apply_temperature(
        calibration_whole, float(temperature["selected_temperature"])
    )
    simple_candidates = [
        {
            "arm": "raw_whole_expectation",
            "calibration_brier": _brier(calibration_whole, calibration_y),
        },
        {
            "arm": "raw_max_window_probability",
            "calibration_brier": _brier(calibration_maximum, calibration_y),
        },
    ]
    simple_baseline = min(simple_candidates, key=lambda row: (row["calibration_brier"], row["arm"]))
    constant_fit = _fit_linear_head(
        np.ones((len(train_y), 1)),
        train_y,
        seed=TRAINING_SEEDS[0],
        regularization=0.0,
        steps=steps,
    )
    shuffled_y = np.random.default_rng(TRAINING_SEEDS[0]).permutation(train_y)
    shuffled_fit = _fit_linear_head(
        designs["window_gibbs"][0],
        shuffled_y,
        seed=TRAINING_SEEDS[0],
        regularization=float(selected["window_gibbs"]["regularization"]),
        steps=steps,
    )
    policies = freeze_policies(selected_window, temperature_probabilities)
    normalization_checks = {
        "finite_energies": all(row.get("finite") is True for row in candidates),
        "exact_two_state_normalization": all(
            float(row.get("normalization_max_abs_error", math.inf)) <= 1e-15 for row in candidates
        ),
        "stable_optimization": all(row.get("stable") is True for row in candidates),
    }
    normalization_checks["passed"] = all(normalization_checks.values())
    controls = {
        "constant_feature": {
            **constant_fit,
            "passed": constant_fit["finite"] and constant_fit["stable"],
            "evidence_scope": "training_control_only",
        },
        "label_shuffle": {
            **shuffled_fit,
            "passed": shuffled_fit["finite"] and shuffled_fit["stable"],
            "label_permutation_sha256": canonical_hash(shuffled_y.tolist()),
            "evidence_scope": "training_control_only",
        },
    }
    bundle: JsonDict = {
        "schema": "carnot.exp7505.v657.frozen_fit_bundle.v1",
        "equation": "E(0,x)=0; E(1,x)=-g_theta(x); p(unsupported|x)=sigmoid(g_theta(x))",
        "objective": "binary_log_loss",
        "roles_consumed": ["training", "calibration_tuning"],
        "heldout_labels_consumed": False,
        "seeds": list(TRAINING_SEEDS),
        "regularization_grid": list(REGULARIZATION_GRID),
        "optimizer_steps_per_candidate": steps,
        "candidate_budget_per_arm": len(TRAINING_SEEDS) * len(REGULARIZATION_GRID),
        "optimizer_total_time_s": time.monotonic() - started,
        "optimizer_time_limit_s": MAX_OPTIMIZER_SECONDS,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "normalization": {
            "fit_role": "training",
            "mean": mean.tolist(),
            "population_sd": scale.tolist(),
            "safe_scale": safe_scale.tolist(),
        },
        "spline": {
            "basis": "open_clamped_cubic_quantile",
            "knots": knots.tolist(),
            "coefficients_per_input": COEFFICIENTS_PER_INPUT,
            "trainable_parameter_count": len(FEATURE_NAMES) * COEFFICIENTS_PER_INPUT + 1,
        },
        "candidate_rows": candidates,
        "selected_heads": selected,
        "temperature_baseline": temperature,
        "simple_baseline_candidates": simple_candidates,
        "simple_baseline": {**simple_baseline, "selection_metric": "calibration_brier"},
        "raw_scores_unchanged": True,
        "controls": controls,
        "normalization_checks": normalization_checks,
        "frozen_policies": policies,
        "calibration_group_count": len(calibration),
        "training_group_count": len(training),
        "training_input_sha256": canonical_hash(training),
        "calibration_input_sha256": canonical_hash(calibration),
    }
    bundle["bundle_sha256"] = bundle_hash(bundle)
    return bundle


def expected_action(
    probability: float,
    *,
    false_accept_cost: float,
    escalation_cost: float,
    false_reject_cost: float = FALSE_REJECT_COST,
) -> str:
    """Choose the minimum expected-cost action with one frozen tie order."""

    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("probability_invalid")
    costs = {
        "accept": numeric * false_accept_cost,
        "escalate": escalation_cost,
        "reject": (1.0 - numeric) * false_reject_cost,
    }
    return min(TIE_ORDER, key=lambda action: (costs[action], TIE_ORDER.index(action)))


def policy_hash(value: Mapping[str, Any]) -> str:
    """Bind frozen actions without hashing the self-referential field."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "policy_sha256"}
    )


def freeze_policies(window_probabilities: Any, temperature_probabilities: Any) -> JsonDict:
    """Freeze all nine cost cells for both calibrated probability sources."""

    sources = {
        "window_gibbs": np.asarray(window_probabilities, dtype=np.float64),
        "temperature_whole": np.asarray(temperature_probabilities, dtype=np.float64),
    }
    if any(
        values.ndim != 1 or not len(values) or not np.all(np.isfinite(values))
        for values in sources.values()
    ):
        raise ValueError("policy_probabilities_invalid")
    rows: list[JsonDict] = []
    for source, probabilities in sources.items():
        for false_accept_cost in FALSE_ACCEPT_COSTS:
            for escalation_cost in ESCALATION_COSTS:
                actions = [
                    expected_action(
                        float(probability),
                        false_accept_cost=false_accept_cost,
                        escalation_cost=escalation_cost,
                    )
                    for probability in probabilities
                ]
                rows.append(
                    {
                        "probability_source": source,
                        "false_reject_cost": FALSE_REJECT_COST,
                        "false_accept_cost": false_accept_cost,
                        "escalation_cost": escalation_cost,
                        "action_counts": dict(Counter(actions)),
                        "actions": actions,
                        "selection_role": "calibration_tuning",
                        "frozen": True,
                    }
                )
    value: JsonDict = {
        "tie_breaking": list(TIE_ORDER),
        "decision_rule": "minimum_expected_cost_from_calibrated_unsupported_probability",
        "rows": rows,
    }
    value["policy_sha256"] = policy_hash(value)
    return value


def source_hash_row(path: Path, root: Path) -> JsonDict:
    """Bind exact source bytes with a stable repository-relative path."""

    resolved = path if path.is_absolute() else root / path
    try:
        label = resolved.relative_to(root).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _precondition(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    path: str,
    passed: bool,
) -> JsonDict:
    """Record an exact prerequisite failure without repairing producer bytes."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "path": path,
        "passed": passed,
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate required resources and the terminal Exp7504 interface."""

    rows: list[JsonDict] = []
    sources: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        passed = path.is_file() and path.stat().st_size > 0
        rows.append(
            _precondition(
                "resource_readable",
                "worktree_or_Exp7504",
                "path",
                "readable_nonempty_bytes",
                {
                    "exists": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                },
                relative.as_posix(),
                passed,
            )
        )
        if passed:
            sources.append(source_hash_row(relative, root))
    if not all(row["passed"] for row in rows):
        return {"rows": rows, "source_artifact_hashes": sources, "passed": False}
    artifact = json.loads((root / UPSTREAM_ARTIFACT).read_text(encoding="utf-8"))
    checks = (
        ("upstream_terminal_status", "terminal_status", "complete"),
        ("upstream_ready", "evidence_ready_score", 1),
        ("upstream_adversarial", "flagged_adversarial", False),
        ("upstream_model_load", "model_invoked", False),
    )
    for check, field, expected in checks:
        observed = artifact.get(field)
        rows.append(
            _precondition(
                check,
                "Exp7504",
                field,
                expected,
                observed,
                UPSTREAM_ARTIFACT.as_posix(),
                observed == expected,
            )
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        _precondition(
            "relevant_requirement_present",
            "OpenSpec",
            "REQ-VERIFY-7505",
            True,
            "REQ-VERIFY-7505" in spec_text,
            SPEC_PATH.as_posix(),
            "REQ-VERIFY-7505" in spec_text,
        )
    )
    sidecars = artifact.get("raw_sidecars", {})
    for name, relative in (
        ("features", FEATURE_PATH),
        ("training_normalization", NORMALIZATION_PATH),
        ("access_exposure_manifest", ACCESS_PATH),
    ):
        receipt = sidecars.get(name, {}) if isinstance(sidecars, Mapping) else {}
        observed = sha256_file(root / relative)
        expected = receipt.get("sha256") if isinstance(receipt, Mapping) else None
        rows.append(
            _precondition(
                "upstream_sidecar_hash",
                "Exp7504",
                name,
                expected,
                observed,
                relative.as_posix(),
                expected == observed,
            )
        )
    return {
        "rows": rows,
        "source_artifact_hashes": sources,
        "passed": all(row["passed"] for row in rows),
    }


def load_fit_rows(root: Path) -> JsonDict:
    """Use the Exp7504 fit reader, then convert its supported label to unsupported."""

    opened = read_mode(root / FEATURE_PATH, root / EVALUATOR_PATH, mode="fit")
    rows = opened["rows"]
    for row in rows:
        row["label"] = 1 - int(row["label"])
    training = [row for row in rows if row["role"] == "training"]
    calibration = [row for row in rows if row["role"] == "calibration_tuning"]
    if len(training) != 176 or len(calibration) != 60:
        raise ValueError("fit_role_counts_invalid")
    receipt = deepcopy(opened["access_receipt"])
    receipt["label_semantics"] = "one_if_contains_human_unsupported_span"
    return {"training": training, "calibration": calibration, "access_receipt": receipt}


def build_calibration_rows(
    bundle: Mapping[str, Any], calibration: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Keep per-group arm probabilities reviewable without opening held-out labels."""

    matrix, labels, raw_whole, raw_maximum = _fit_matrices(calibration, "calibration_tuning")
    normalization = bundle["normalization"]
    normalized = (matrix - np.asarray(normalization["mean"])) / np.asarray(
        normalization["safe_scale"]
    )
    knots = np.asarray(bundle["spline"]["knots"], dtype=np.float64)
    designs = {
        "window_gibbs": spline_design_matrix(normalized, knots),
        "whole_only_gibbs": normalized[:, :1],
        "identical_ten_feature_logistic": normalized,
    }
    scores = {
        arm: _head_probabilities(bundle["selected_heads"][arm], design)
        for arm, design in designs.items()
    }
    scores["temperature_whole"] = _apply_temperature(
        raw_whole, float(bundle["temperature_baseline"]["selected_temperature"])
    )
    scores["raw_whole_expectation"] = raw_whole
    scores["raw_max_window_probability"] = raw_maximum
    rows = []
    for index, source in enumerate(calibration):
        for arm, probabilities in scores.items():
            rows.append(
                {
                    "group_id": source["group_id"],
                    "role": "calibration_tuning",
                    "arm": arm,
                    "unsupported_probability": float(probabilities[index]),
                    "label": int(labels[index]),
                    "status": "complete",
                    "failed": False,
                    "censored": False,
                }
            )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write one bounded JSONL sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sidecar(path: Path, root: Path, *, rows: int | None = None) -> JsonDict:
    """Describe exact sidecar bytes and optional independent row count."""

    value = source_hash_row(path, root)
    if rows is not None:
        value["rows"] = rows
    return value


def write_fit_sidecars(
    root: Path,
    bundle: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write frozen checkpoints, calibration rows, and a bounded fit receipt."""

    checkpoints = {
        "schema": "carnot.exp7505.v657.checkpoints.v1",
        "bundle_sha256": bundle["bundle_sha256"],
        "selected_heads": deepcopy(bundle["selected_heads"]),
        "candidate_rows": deepcopy(bundle["candidate_rows"]),
        "temperature_baseline": deepcopy(bundle["temperature_baseline"]),
        "simple_baseline": deepcopy(bundle["simple_baseline"]),
        "frozen_policies": deepcopy(bundle["frozen_policies"]),
        "transform_sha256": canonical_hash(
            {
                "feature_names": bundle["feature_names"],
                "normalization": bundle["normalization"],
                "spline": bundle["spline"],
            }
        ),
    }
    fit_receipt = {
        "schema": "carnot.exp7505.v657.fit_receipt.v1",
        "objective": bundle["objective"],
        "equation": bundle["equation"],
        "optimizer": "full_batch_adam",
        "learning_rate": LEARNING_RATE,
        "optimizer_steps_per_candidate": bundle["optimizer_steps_per_candidate"],
        "optimizer_total_time_s": bundle["optimizer_total_time_s"],
        "optimizer_time_limit_s": bundle["optimizer_time_limit_s"],
        "candidate_count": len(bundle["candidate_rows"]),
        "controls": deepcopy(bundle["controls"]),
        "normalization_checks": deepcopy(bundle["normalization_checks"]),
    }
    atomic_json(root / CHECKPOINT_PATH, checkpoints)
    _write_jsonl(root / CALIBRATION_PATH, calibration_rows)
    atomic_json(root / FIT_RECEIPT_PATH, fit_receipt)
    for path in (CHECKPOINT_PATH, CALIBRATION_PATH, FIT_RECEIPT_PATH):
        if (root / path).stat().st_size >= 20 * 1024 * 1024:  # pragma: no cover
            raise ValueError(f"sidecar_size_limit:{path}")
    return {
        "checkpoints": _sidecar(root / CHECKPOINT_PATH, root),
        "calibration_rows": _sidecar(root / CALIBRATION_PATH, root, rows=len(calibration_rows)),
        "fit_receipt": _sidecar(root / FIT_RECEIPT_PATH, root),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
) -> JsonDict:
    """Attach the exact operand and failure prevented to one acceptance gate."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": "A favorable metric cannot replace this independently checked operand.",
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed check while retaining all failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {"passed": True, "failed_checks": [], "failure": None}
    first = failed[0]
    return {
        "passed": False,
        "failed_checks": [row.get("check") for row in failed],
        "failure": {
            key: deepcopy(first.get(key)) for key in ("check", "expected", "observed", "op")
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
        "schema": "Versioned identity prevents readers from mixing incompatible layouts.",
        "run_date": "The registered date prevents a later run from replacing this measurement.",
        "MODEL_SPECS": "An empty list prevents cached Qwen evidence from becoming a current load claim.",
        "model_specs": "The duplicate machine field keeps no-load declarations explicit.",
        "model_invoked": "A false value separates current CPU fitting from historical model inference.",
        "invocation_counts": "Zero reconciled counters prevent hidden current model calls.",
        "inference_substrate": "The exact aggregation phrase identifies upstream cached evidence.",
        "energy_fit_ready_score": "A bare readiness bit reports valid fitted heads without claiming benefit.",
        "baseline_ready_score": "A separate bit keeps temperature-base readiness independent of energy benefit.",
        "small_ebm_training": "The equation, optimizer, seeds, and timing prove actual small-head training.",
        "checkpoint_manifest": "Hashes freeze heads and policies before later label access.",
        "calibration_rows": "Per-group fit-selection rows keep calibration evidence reviewable.",
    }
    return {
        field: specific.get(
            field, "This field preserves an auditable operand and prevents silent evidence drift."
        )
        for field in fields
    }


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute fit and baseline readiness from terminal rows and frozen settings."""

    rows = value.get("rows")
    training = value.get("small_ebm_training")
    policies = value.get("frozen_policy_manifest")
    checkpoints = value.get("checkpoint_manifest")
    if not isinstance(rows, list) or not isinstance(training, Mapping):
        return {"energy_fit_ready_score": 0, "baseline_ready_score": 0}
    candidates = [row for row in rows if row.get("kind") == "candidate"]
    controls = [row for row in rows if row.get("kind") == "training_control"]
    expected_cells = {
        (arm, seed, regularization)
        for arm in TRAINABLE_ARMS
        for seed in TRAINING_SEEDS
        for regularization in REGULARIZATION_GRID
    }
    observed_cells = {
        (row.get("arm"), row.get("seed"), row.get("regularization")) for row in candidates
    }
    candidate_ok = (
        len(candidates) == len(expected_cells)
        and observed_cells == expected_cells
        and all(
            row.get("status") == "complete"
            and int(row.get("optimizer_steps", MAX_OPTIMIZER_STEPS + 1)) <= MAX_OPTIMIZER_STEPS
            and int(row.get("parameter_count", 257)) <= 256
            for row in candidates
        )
    )
    controls_ok = {row.get("arm") for row in controls} == {
        "constant_feature_control",
        "label_shuffle_control",
    } and all(row.get("status") == "complete" for row in controls)
    policy_ok = (
        isinstance(policies, Mapping)
        and policies.get("tie_breaking") == list(TIE_ORDER)
        and isinstance(policies.get("rows"), list)
        and len(policies["rows"]) == 18
        and all(row.get("frozen") is True for row in policies["rows"])
    )
    checkpoint_ok = (
        isinstance(checkpoints, Mapping) and len(checkpoints.get("selected_heads", [])) == 3
    )
    fit_ready = (
        candidate_ok
        and controls_ok
        and policy_ok
        and checkpoint_ok
        and training.get("exact_two_state_normalization_passed") is True
        and training.get("stable_optimization_passed") is True
        and value.get("validation_summary", {}).get("required_checks_passed") is True
    )
    temperature = value.get("temperature_baseline", {})
    baseline_ready = (
        isinstance(temperature, Mapping)
        and temperature.get("selection_metric") == "calibration_brier"
        and len(temperature.get("candidates", [])) == len(TEMPERATURE_GRID)
        and value.get("validation_summary", {}).get("required_checks_passed") is True
    )
    return {"energy_fit_ready_score": int(fit_ready), "baseline_ready_score": int(baseline_ready)}


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    completed_at: str,
    duration_s: float,
    fixture: bool = False,
) -> JsonDict:
    """Assemble one complete fit record without reading held-out outcomes."""

    candidate_rows = [
        {
            "kind": "candidate",
            "arm": row["arm"],
            "seed": row["seed"],
            "regularization": row["regularization"],
            "optimizer_steps": row["optimizer_steps"],
            "parameter_count": row["parameter_count"],
            "calibration_brier": row.get("calibration_brier"),
            "normalization_max_abs_error": row.get("normalization_max_abs_error"),
            "status": row["status"],
            "failure": row.get("failure"),
            "failed": row["status"] != "complete",
            "censored": False,
        }
        for row in bundle["candidate_rows"]
    ]
    control_rows = [
        {
            "kind": "training_control",
            "arm": f"{name}_control",
            "seed": TRAINING_SEEDS[0],
            "status": "complete" if control["passed"] else "failed_validation",
            "failed": not control["passed"],
            "censored": False,
        }
        for name, control in bundle["controls"].items()
    ]
    checkpoint_manifest = {
        "bundle_sha256": bundle["bundle_sha256"],
        "selected_heads": [
            {
                "arm": arm,
                "seed": head["seed"],
                "regularization": head["regularization"],
                "checkpoint_sha256": head["checkpoint_sha256"],
                "parameter_count": head["parameter_count"],
            }
            for arm, head in bundle["selected_heads"].items()
        ],
        "policy_sha256": bundle["frozen_policies"]["policy_sha256"],
        "transform_sha256": canonical_hash(
            {
                "feature_names": bundle["feature_names"],
                "normalization": bundle["normalization"],
                "spline": bundle["spline"],
            }
        ),
        "frozen_before_heldout_label_access": True,
    }
    fit_valid = (
        all(row["status"] == "complete" for row in candidate_rows)
        and all(control["passed"] for control in bundle["controls"].values())
        and bundle["normalization_checks"]["passed"]
    )
    baseline_valid = bundle["temperature_baseline"][
        "selection_metric"
    ] == "calibration_brier" and len(bundle["temperature_baseline"]["candidates"]) == len(
        TEMPERATURE_GRID
    )
    gates = [
        _gate(
            "fit_visible_roles_only",
            "validity",
            False,
            bundle["heldout_labels_consumed"],
            "eq",
            not bundle["heldout_labels_consumed"],
        ),
        _gate("all_candidate_heads_valid", "readiness", True, fit_valid, "eq", fit_valid),
        _gate(
            "temperature_baseline_valid", "readiness", True, baseline_valid, "eq", baseline_valid
        ),
        _gate("required_validation", "validity", True, validation_passed, "eq", validation_passed),
        _gate("heldout_benefit_not_measured", "benefit", False, False, "eq", True),
    ]
    status_ready = fit_valid and baseline_valid and validation_passed
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7505,
        "title": "Fit calibrated conditional Gibbs decisions on sealed window evidence",
        "milestone": MILESTONE,
        "status": "complete",
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "device_identity": {
            "host": platform.node(),
            "machine": platform.machine(),
            "device": "cpu",
        },
        "duration_s": duration_s,
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": duration_s,
            "validation": 0.0,
            "historical_capture": 0.0,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "fitting": list(TRAINING_SEEDS),
            "arrival": None,
            "audit": None,
            "bootstrap": None,
        },
        "source_artifact_hashes": deepcopy(list(source_hashes)),
        "historical_model_provenance": {
            "source": "Exp7504",
            "classification": "historical_cached_Qwen_evidence",
        },
        "rows": [*candidate_rows, *control_rows],
        "sample_size_budget": {
            "planned": len(candidate_rows) + len(control_rows),
            "attempted": len(candidate_rows) + len(control_rows),
            "complete": sum(
                row["status"] == "complete" for row in [*candidate_rows, *control_rows]
            ),
            "completed": sum(
                row["status"] == "complete" for row in [*candidate_rows, *control_rows]
            ),
            "excluded": 0,
            "failed": sum(row["failed"] for row in [*candidate_rows, *control_rows]),
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "registered_fit_candidate_or_control",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "validation_summary": {"required_checks_passed": validation_passed},
        "energy_fit_ready_score": int(status_ready),
        "baseline_ready_score": int(baseline_valid and validation_passed),
        "small_ebm_training": {
            "performed": True,
            "equation": bundle["equation"],
            "loss": bundle["objective"],
            "optimizer": "full_batch_adam",
            "learning_rate": LEARNING_RATE,
            "optimizer_steps_per_candidate": bundle["optimizer_steps_per_candidate"],
            "optimizer_total_time_s": bundle["optimizer_total_time_s"],
            "optimizer_time_limit_s": bundle["optimizer_time_limit_s"],
            "seeds": list(bundle["seeds"]),
            "regularization_grid": list(bundle["regularization_grid"]),
            "trainable_parameter_counts": {
                arm: head["parameter_count"] for arm, head in bundle["selected_heads"].items()
            },
            "stable_optimization_passed": bundle["normalization_checks"]["stable_optimization"],
            "finite_energies_passed": bundle["normalization_checks"]["finite_energies"],
            "exact_two_state_normalization_passed": bundle["normalization_checks"][
                "exact_two_state_normalization"
            ],
            "controls_passed": all(control["passed"] for control in bundle["controls"].values()),
            "current_cpu_work": True,
        },
        "checkpoint_manifest": checkpoint_manifest,
        "calibration_rows": {
            "sidecar": deepcopy(sidecars.get("calibration_rows")),
            "row_count": len(calibration_rows),
            "group_count": len({row["group_id"] for row in calibration_rows}),
            "arms": sorted({row["arm"] for row in calibration_rows}),
            "selection_role": "calibration_tuning",
        },
        "raw_sidecars": deepcopy(dict(sidecars)),
        "temperature_baseline": deepcopy(bundle["temperature_baseline"]),
        "simple_baseline": deepcopy(bundle["simple_baseline"]),
        "frozen_policy_manifest": deepcopy(bundle["frozen_policies"]),
        "label_access_receipt": {
            "label_roles_opened": ["calibration_tuning", "training"],
            "held_out_labels_opened": False,
        },
        "predictive_benefit_measured": False,
        "fixture_behavior": "circular_only" if fixture else "not_applicable",
        "honest_verdict": (
            "complete_circular_positive_fixture_energy_fit_ready"
            if fixture and status_ready
            else "complete_null_energy_fit_ready_heldout_efficacy_unmeasured"
            if status_ready
            else "complete_disqualified_energy_fit_validation_failed"
        ),
        "verdict_class": "circular_positive"
        if fixture and status_ready
        else "null"
        if status_ready
        else "disqualified",
        "verifier_is_oracle": fixture,
        "flagged_adversarial": not validation_passed,
        "affected_validation_manifest": {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "full_python_suite_run": False,
            "numbered_runtime_e2e_applicable": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": validation_passed,
            "numbered_runtime_e2e_applicable": False,
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    verify_sources: bool = True,
) -> list[str]:
    """Cold-check identity, declarations, source bytes, reduction, and checksum."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in value]
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_must_be_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_mismatch")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_mismatch")
    if value.get("predictive_benefit_measured") is not False:
        errors.append("heldout_benefit_claimed")
    reduced = independent_reduce(value)
    if reduced != {
        "energy_fit_ready_score": value.get("energy_fit_ready_score"),
        "baseline_ready_score": value.get("baseline_ready_score"),
    }:
        errors.append("independent_reduction_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if set(value) - set(value.get("field_principles", {})):
        errors.append("field_principles_incomplete")
    if verify_sources:
        for row in value.get("source_artifact_hashes", []):
            path = root / str(row.get("path"))
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{row.get('path')}")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Validate one serialized candidate in a fresh-reader compatible path."""

    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_artifact(value, root=root, verify_sources=verify_sources)


def _fixture_rows(role: str, count: int, offset: int) -> list[JsonDict]:
    """Create a deterministic circular fixture for structural tests only."""

    rows = []
    for index in range(count):
        value = index + offset
        label = value % 2
        rows.append(
            {
                "group_id": f"fixture-{role}-{index}",
                "role": role,
                "features": [
                    (value - count / 2) / count + 0.4 * label,
                    0.2 + 0.5 * label,
                    0.3 + 0.4 * label,
                    0.4 + 0.4 * label,
                    0.01 + 0.02 * (value % 3),
                    0.05 * (value % 4),
                    0.03 * (value % 5),
                    1.0 + 0.1 * (value % 3),
                    5.0 + 0.05 * value,
                    6.0 + 0.02 * value,
                ],
                "raw_whole_expectation": 0.2 + 0.6 * label,
                "raw_max_window_probability": 0.3 + 0.5 * label,
                "label": label,
            }
        )
    return rows


def fixture_artifact() -> JsonDict:
    """Build a cold-valid circular shape without claiming corpus efficacy."""

    training = _fixture_rows("training", 20, 0)
    calibration = _fixture_rows("calibration_tuning", 10, 20)
    bundle = fit_energy_bundle(training, calibration, steps=2)
    calibration_rows = build_calibration_rows(bundle, calibration)
    return build_artifact(
        preconditions=[],
        source_hashes=[],
        bundle=bundle,
        sidecars={},
        calibration_rows=calibration_rows,
        validation_receipts=[{"name": "fixture_validation", "exit_code": 0, "passed": True}],
        validation_passed=True,
        phase_spans=[],
        started_at="2026-09-22T12:00:00+00:00",
        completed_at="2026-09-22T12:00:01+00:00",
        duration_s=1.0,
        fixture=True,
    )


def blocked_artifact(failed: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Publish external absence as a complete blocked result with exact operands."""

    del root
    gate = _gate(
        str(failed.get("check")),
        "validity",
        failed.get("expected"),
        failed.get("observed"),
        "eq",
        False,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7505,
        "title": "Fit calibrated conditional Gibbs decisions on sealed window evidence",
        "milestone": MILESTONE,
        "status": "blocked",
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "preconditions_checked": [deepcopy(dict(failed))],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 0.0,
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": 0.0,
            "validation": 0.0,
            "historical_capture": 0.0,
        },
        "phase_spans": [],
        "random_seed": {
            "fitting": list(TRAINING_SEEDS),
            "arrival": None,
            "audit": None,
            "bootstrap": None,
        },
        "source_artifact_hashes": [],
        "rows": [],
        "sample_size_budget": {
            "planned": 47,
            "attempted": 0,
            "complete": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 47,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [failed.get("check")],
            "check": failed.get("check"),
            "upstream": failed.get("upstream"),
            "artifact_field": failed.get("artifact_field"),
            "expected": failed.get("expected"),
            "observed": failed.get("observed"),
            "path": failed.get("path"),
        },
        "honest_verdict": f"complete_blocked_{failed.get('check')}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "energy_fit_ready_score": 0,
        "baseline_ready_score": 0,
        "small_ebm_training": {"performed": False},
        "checkpoint_manifest": {},
        "calibration_rows": {"row_count": 0},
        "predictive_benefit_measured": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return an aware UTC timestamp for a durable run boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - real execution boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7505] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real execution boundary.
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:
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


def _fit_worker(input_path: Path, output_path: Path, steps: int) -> int:  # pragma: no cover
    """Run bounded numeric fitting in one owned unbuffered child process."""

    value = json.loads(input_path.read_text(encoding="utf-8"))
    bundle = fit_energy_bundle(value["training"], value["calibration"], steps=steps)
    atomic_json(output_path, bundle)
    print(
        json.dumps(
            {
                "candidate_count": len(bundle["candidate_rows"]),
                "bundle_sha256": bundle["bundle_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build cold replay, reduction, adversarial, and strict-row readers."""

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
    """Authenticate, fit, validate, cold-replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
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
    progress(run_started, "preconditions", "complete", checks=len(preconditions["rows"]))

    progress(run_started, "fit_reader", "before_benchmark")
    phase_started = time.monotonic()
    fit_rows = load_fit_rows(root)
    spans.append(
        _span("fit_reader", phase_started, run_started, 236, "training_and_calibration_rows")
    )
    progress(run_started, "fit_reader", "after_benchmark", groups=236)

    progress(run_started, "numeric_fit", "before_training", candidates=45)
    phase_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="carnot-exp7505-fit-", dir="/tmp") as temporary:
        temporary_root = Path(temporary)
        fit_input = temporary_root / "fit-input.json"
        fit_output = temporary_root / "fit-output.json"
        atomic_json(
            fit_input,
            {"training": fit_rows["training"], "calibration": fit_rows["calibration"]},
        )
        fit_spec = validation_scope.CommandSpec(
            "numeric_fit",
            (
                sys.executable,
                "-u",
                "-m",
                "carnot.experiment_7505_v657_energy_fit",
                "--date",
                RUN_DATE,
                "--fit-input",
                str(fit_input),
                "--fit-output",
                str(fit_output),
                "--fit-steps",
                "120",
            ),
            "training_and_calibration_only",
            MAX_OPTIMIZER_SECONDS,
        )
        fit_receipts = run_categorized_commands(
            root,
            [PlannedCommand(fit_spec, "numeric_training", True)],
            log_dir=root / RAW_DIR / "validation" / "fit",
        )
        if not _all_receipts_pass(fit_receipts, ("numeric_fit",)):
            raise RuntimeError("numeric_fit_subprocess_failed")
        bundle = json.loads(fit_output.read_text(encoding="utf-8"))
    if bundle.get("bundle_sha256") != bundle_hash(bundle):
        raise RuntimeError("fit_bundle_hash_invalid")
    spans.append(_span("numeric_fit", phase_started, run_started, 45, "frozen_fit_bundle"))
    progress(run_started, "numeric_fit", "after_training", candidates=45)

    progress(run_started, "calibration_freeze", "before_benchmark")
    phase_started = time.monotonic()
    calibration_rows = build_calibration_rows(bundle, fit_rows["calibration"])
    sidecars = write_fit_sidecars(root, bundle, calibration_rows)
    spans.append(
        _span(
            "calibration_freeze",
            phase_started,
            run_started,
            len(calibration_rows),
            CHECKPOINT_PATH.as_posix(),
        )
    )
    progress(run_started, "calibration_freeze", "after_benchmark", rows=len(calibration_rows))

    source_hashes = list(preconditions["source_artifact_hashes"])
    source_hashes.extend(
        source_hash_row(path, root) for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH)
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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7505-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(run_started, "affected_validation", "before_subprocesses", commands=len(commands))
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
        passed=affected_reduction["passed"],
    )

    candidate = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=source_hashes,
        bundle=bundle,
        sidecars=sidecars,
        calibration_rows=calibration_rows,
        validation_receipts=[*fit_receipts, *affected],
        validation_passed=bool(affected_reduction["passed"]),
        phase_spans=spans,
        started_at=started_at,
        completed_at=utc_now(),
        duration_s=time.monotonic() - run_started,
    )
    candidate_errors = validate_artifact(candidate, root=root)
    if candidate_errors:
        raise RuntimeError(f"measured_candidate_invalid:{candidate_errors}")
    atomic_json(root / TERMINAL_CANDIDATE, candidate)

    progress(run_started, "terminal_validation", "before_subprocesses", commands=4)
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
        passed=terminal_passed,
        critical=critical,
    )
    all_validation_passed = bool(affected_reduction["passed"] and terminal_passed and not critical)
    final = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=source_hashes,
        bundle=bundle,
        sidecars={
            **sidecars,
            "affected_validation_manifest": _sidecar(manifest_path, root),
            "terminal_candidate": _sidecar(root / TERMINAL_CANDIDATE, root),
        },
        calibration_rows=calibration_rows,
        validation_receipts=[*fit_receipts, *affected, *terminal],
        validation_passed=all_validation_passed,
        phase_spans=spans,
        started_at=started_at,
        completed_at=utc_now(),
        duration_s=time.monotonic() - run_started,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and read-only worker or replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fit-input", type=Path)
    parser.add_argument("--fit-output", type=Path)
    parser.add_argument("--fit-steps", type=int, default=120)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the fit or one bounded child and cold-reader operation."""

    arguments = parse_args(argv)
    if arguments.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = arguments.root.resolve()
    if arguments.fit_input is not None or arguments.fit_output is not None:
        if arguments.fit_input is None or arguments.fit_output is None:
            raise SystemExit("--fit-input and --fit-output must be supplied together")
        return _fit_worker(arguments.fit_input, arguments.fit_output, arguments.fit_steps)
    if arguments.cold_replay is not None:
        errors = cold_replay(arguments.cold_replay, root=root)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if arguments.independent_reduce is not None:
        value = json.loads(arguments.independent_reduce.read_text(encoding="utf-8"))
        reduced = independent_reduce(value)
        expected = {
            "energy_fit_ready_score": value.get("energy_fit_ready_score"),
            "baseline_ready_score": value.get("baseline_ready_score"),
        }
        passed = reduced == expected
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(root, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
