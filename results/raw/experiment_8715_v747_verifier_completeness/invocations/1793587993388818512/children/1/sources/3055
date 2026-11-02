"""Fit a bounded proper-loss energy map to cached original forecasts.

This CPU-only experiment fits a nine-knot probability map. It uses cached
Qwen evidence and never changes generator weights. Fit readiness is not a
predictive-benefit claim.

Spec refs: REQ-CL-7576 and SCENARIO-CL-7576-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7561_v661_recalibration_prototype import (
    KNOTS,
    LOG_CLIP,
    MOVEMENT_BOUND,
    PRIMARY_SOLVER,
    RIDGE_MASS,
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
    constraint_errors,
    map_probability,
    quadratic_objective,
    solve_constrained_map,
    statistics_from_examples,
    typed_decision,
)
from carnot.experiment_7566_v661_energy_fit import TEMPERATURE_GRID
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
Solver = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, JsonDict]]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7576-v662-proper-loss-energy"
SCHEMA = "carnot.exp7576.v662.proper_loss_energy.v1"
RESULT_PATH = Path("results/experiment_7576_v662_proper_loss_energy.json")
RAW_DIR = Path("results/raw/experiment_7576_v662_proper_loss_energy")
MODULE_PATH = Path("python/carnot/experiment_7576_v662_proper_loss_energy.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7576_v662_proper_loss_energy.py")
TEST_PATH = Path("tests/python/test_experiment_7576_v662_proper_loss_energy.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
UPSTREAM_REQUALIFICATION_PATH = Path(
    "results/experiment_7574_v662_measurement_requalification.json"
)
UPSTREAM_PROTOCOL_PATH = Path("results/experiment_7575_v662_cached_learning_protocol.json")
HEAD_MANIFEST_PATH = RAW_DIR / "head_manifest.json"
CHECKPOINT_PATH = RAW_DIR / "proper_loss_checkpoint.json"
CONTROL_PATH = RAW_DIR / "qualification_controls.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
AFFECTED_MANIFEST_PATH = RAW_DIR / "affected_validation_manifest.json"
HISTORICAL_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
FIT_SEED = 7_576_001
PERTURBATIONS = (0.0, 0.01, 0.05)
ALLOWED_ROLES = ("fit", "tune", "policy")
EXPECTED_ROLE_COUNTS = {"fit": 160, "tune": 40, "policy": 40}
MODEL_SPECS: list[str] = []
ZERO_INVOCATION_COUNTS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
REQUIRED_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _finite_probability(probability: float) -> float:
    value = float(probability)
    if not math.isfinite(value):
        raise ValueError("probability_not_finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability_out_of_range")
    return value


def normalized_binary_energies(probability: float) -> JsonDict:
    """Return binary energies while preserving the unclipped fitted value."""

    value = _finite_probability(probability)
    log_safe = min(1.0 - LOG_CLIP, max(LOG_CLIP, value))
    return {
        "error": -math.log(log_safe),
        "correct": -math.log1p(-log_safe),
        "log_safe_probability": log_safe,
        "unclipped_probability": value,
    }


def probability_from_energies(energies: Mapping[str, Any]) -> float:
    """Normalize correct and error energies into error probability."""

    if not {"error", "correct"} <= set(energies):
        raise ValueError("energy_pair_invalid")
    error_weight = math.exp(-float(energies["error"]))
    correct_weight = math.exp(-float(energies["correct"]))
    return error_weight / (error_weight + correct_weight)


def _load_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _resolved(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def _source_hash(path: Path, root: Path) -> JsonDict:
    resolved = path if path.is_absolute() else root / path
    try:
        label = resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved.resolve())
    return {"path": label, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _validate_role_rows(rows: Sequence[Mapping[str, Any]], role: str) -> list[JsonDict]:
    """Validate one authorized role before any label enters an objective."""

    if role not in ALLOWED_ROLES:
        raise ValueError(f"forbidden_role_requested:{role}")
    copied = [deepcopy(dict(row)) for row in rows]
    if not copied:
        raise ValueError(f"{role}_rows_empty")
    identities: set[str] = set()
    labels: set[int] = set()
    for row in copied:
        if row.get("role") != role:
            raise ValueError(f"{role}_role_invalid")
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in identities:
            raise ValueError("source_id_duplicate")
        identities.add(source_id)
        _finite_probability(float(row.get("probability")))
        label = row.get("label")
        if label not in (0, 1):
            raise ValueError("binary_label_required")
        labels.add(int(label))
        requests = row.get("request_hashes")
        if not isinstance(requests, list) or len(requests) != 6 or len(set(requests)) != 6:
            raise ValueError("option_order_custody_invalid")
        if row.get("diagnostics_are_labels") is not False:
            raise ValueError("diagnostic_label_leakage")
    if labels != {0, 1}:
        raise ValueError(f"{role}_binary_support_invalid")
    return copied


def load_frozen_fit_roles(
    root: Path,
    source_hashes: list[JsonDict],
    *,
    requested_roles: Sequence[str] = ALLOWED_ROLES,
    expected_counts: Mapping[str, int] = EXPECTED_ROLE_COUNTS,
) -> dict[str, list[JsonDict]]:
    """Read only fit, tune, and policy rows from the frozen Exp7575 sidecar."""

    names = tuple(str(role) for role in requested_roles)
    if not names or any(role not in ALLOWED_ROLES for role in names):
        raise ValueError("forbidden_role_requested")
    root = root.resolve()
    artifact_path = root / UPSTREAM_PROTOCOL_PATH
    artifact = _load_object(artifact_path)
    receipt = artifact.get("raw_sidecars", {}).get("cached_roles", {})
    if not isinstance(receipt, Mapping):
        raise ValueError("cached_roles_receipt_missing")
    path = _resolved(root, str(receipt.get("path") or ""))
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError("cached_roles_hash_invalid")
    roles: dict[str, list[JsonDict]] = {role: [] for role in names}
    observed_rows = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("cached_role_row_invalid")
            observed_rows += 1
            role = str(value.get("role"))
            if role in roles:
                roles[role].append(value)
    if observed_rows != receipt.get("rows"):
        raise ValueError("cached_roles_count_invalid")
    for role in names:
        roles[role] = _validate_role_rows(roles[role], role)
        if len(roles[role]) != expected_counts[role]:
            raise ValueError(f"role_count_invalid:{role}")
    source_hashes.extend((_source_hash(artifact_path, root), _source_hash(path, root)))
    return roles


def _brier(probability: float, label: int) -> float:
    value = _finite_probability(probability)
    if label not in (0, 1):
        raise ValueError("binary_label_required")
    return (value - int(label)) ** 2


def _constraint_residuals(theta: Sequence[float]) -> JsonDict:
    values = np.asarray(theta, dtype=float)
    range_residual = max(0.0, -float(np.min(values)), float(np.max(values)) - 1.0)
    monotone_residual = max(0.0, -float(np.min(np.diff(values))))
    movement_residual = max(0.0, float(np.max(np.abs(values - KNOTS))) - MOVEMENT_BOUND)
    return {
        "range": range_residual,
        "monotonicity": monotone_residual,
        "movement": movement_residual,
        "maximum": max(range_residual, monotone_residual, movement_residual),
    }


def _objective_gradient(theta: np.ndarray, gram: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2.0 * (gram @ theta - target + RIDGE_MASS * (theta - KNOTS))


def solve_range_only_map(gram: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, JsonDict]:
    """Fit the equal-capacity control with range constraints only."""

    result = minimize(
        quadratic_objective,
        KNOTS.copy(),
        args=(np.asarray(gram, dtype=float), np.asarray(target, dtype=float)),
        method=PRIMARY_SOLVER,
        jac=_objective_gradient,
        bounds=Bounds(np.zeros(9), np.ones(9)),
        options={"ftol": SOLVER_TOLERANCE, "maxiter": SOLVER_MAX_ITERATIONS, "disp": False},
    )
    theta = np.asarray(result.x, dtype=float)
    range_residual = max(0.0, -float(np.min(theta)), float(np.max(theta)) - 1.0)
    receipt = {
        "method": f"{PRIMARY_SOLVER}_range_only",
        "tolerance": SOLVER_TOLERANCE,
        "iteration_cap": SOLVER_MAX_ITERATIONS,
        "converged": bool(result.success and range_residual <= 1e-7),
        "status": int(result.status),
        "iterations": int(getattr(result, "nit", 0)),
        "objective": quadratic_objective(theta, gram, target),
        "range_residual": range_residual,
        "message": str(result.message),
    }
    return theta, receipt


def _temperature_probability(probability: float, temperature: float) -> float:
    value = min(1.0 - 1e-12, max(1e-12, _finite_probability(probability)))
    logit = math.log(value / (1.0 - value))
    scaled = logit / float(temperature)
    if scaled >= 0.0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exponent = math.exp(scaled)
    return exponent / (1.0 + exponent)


def _mean_brier(rows: Sequence[Mapping[str, Any]], predictor: Callable[[float], float]) -> float:
    return float(
        np.mean([_brier(predictor(float(row["probability"])), int(row["label"])) for row in rows])
    )


def _objective_record(
    theta: Sequence[float],
    gram: np.ndarray,
    target: np.ndarray,
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> JsonDict:
    values = np.asarray(theta, dtype=float)
    label_constant = float(sum(int(label) ** 2 for label in labels))
    solver_objective = quadratic_objective(values, gram, target)
    brier_sum = float(
        sum(
            _brier(map_probability(float(probability), values), int(label))
            for probability, label in zip(probabilities, labels, strict=True)
        )
    )
    identity_penalty = RIDGE_MASS * float(np.sum((values - KNOTS) ** 2))
    exact = brier_sum + identity_penalty
    return {
        "solver_objective_without_label_constant": solver_objective,
        "label_constant": label_constant,
        "brier_sum": brier_sum,
        "identity_penalty": identity_penalty,
        "exact_objective": exact,
        "objective_reconstruction_residual": abs(exact - (solver_objective + label_constant)),
    }


def _head_record(
    name: str,
    theta: Sequence[float],
    receipt: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
    gram: np.ndarray,
    target: np.ndarray,
    objective_labels: Sequence[int],
    *,
    constraints: str,
) -> JsonDict:
    probabilities = [float(row["probability"]) for row in fit_rows]
    values = np.asarray(theta, dtype=float)
    record = {
        "name": name,
        "theta": values.tolist(),
        "knot_locations": KNOTS.tolist(),
        "parameter_count": 9,
        "constraints": constraints,
        "solver_receipt": deepcopy(dict(receipt)),
        "fit_brier": _mean_brier(
            fit_rows, lambda probability: map_probability(probability, values)
        ),
        "objective": _objective_record(values, gram, target, probabilities, objective_labels),
        "fit_source_ids_sha256": canonical_hash([str(row["source_id"]) for row in fit_rows]),
    }
    record["head_sha256"] = canonical_hash(record)
    return record


def _policy_action_contract(
    bundle: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    probabilities = [float(row["probability"]) for row in rows]
    heads = bundle["heads"]
    head_values = {
        "proper_loss_monotone": [
            map_probability(value, heads["proper_loss_monotone"]["theta"])
            for value in probabilities
        ],
        "raw_original": probabilities,
        "temperature_original": [
            _temperature_probability(value, float(heads["temperature_original"]["temperature"]))
            for value in probabilities
        ],
        "unconstrained_nine_knot": [
            map_probability(value, heads["unconstrained_nine_knot"]["theta"])
            for value in probabilities
        ],
    }
    action_counts = {
        name: dict(Counter(str(typed_decision(value)["action"]) for value in values))
        for name, values in head_values.items()
    }
    tie_checks = [typed_decision(value)["action"] for value in (0.04, 0.8)]
    return {
        "role": "policy",
        "source_count": len(rows),
        "labels_consumed": 0,
        "selection_performed": False,
        "action_counts": action_counts,
        "costs": {"accept": "5q", "reject": "1-q", "escalate": 0.2},
        "tie_breaker": "escalate",
        "tie_checks": tie_checks,
        "passed": tie_checks == ["escalate", "escalate"],
    }


def fit_proper_loss_bundle(
    fit_rows: Sequence[Mapping[str, Any]],
    tune_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_solver: Solver = solve_constrained_map,
) -> JsonDict:
    """Fit every registered head, freeze it, then inspect policy actions."""

    fit = _validate_role_rows(fit_rows, "fit")
    tune = _validate_role_rows(tune_rows, "tune")
    policy = _validate_role_rows(policy_rows, "policy")
    probabilities = [float(row["probability"]) for row in fit]
    labels = [int(row["label"]) for row in fit]
    gram, target = statistics_from_examples(probabilities, labels)
    fallback_used = False
    try:
        candidate_theta, candidate_receipt = candidate_solver(gram, target)
    except Exception as error:  # noqa: BLE001 - a failed registered solver has a defined control.
        candidate_theta = KNOTS.copy()
        candidate_receipt = {
            "method": "registered_solver_exception",
            "converged": False,
            "objective": quadratic_objective(candidate_theta, gram, target),
            "constraint_errors": [type(error).__name__],
            "message": str(error),
        }
        fallback_used = True
    if candidate_receipt.get("converged") is not True or constraint_errors(candidate_theta):
        candidate_theta = KNOTS.copy()
        fallback_used = True
    range_theta, range_receipt = solve_range_only_map(gram, target)
    temperatures = [
        {
            "temperature": float(value),
            "fit_brier": _mean_brier(
                fit,
                lambda probability, current=value: _temperature_probability(probability, current),
            ),
        }
        for value in TEMPERATURE_GRID
    ]
    selected_temperature = min(temperatures, key=lambda row: (row["fit_brier"], row["temperature"]))
    shuffled_labels = np.random.default_rng(FIT_SEED).permutation(labels).astype(int).tolist()
    if shuffled_labels == labels:
        shuffled_labels = labels[1:] + labels[:1]
    shuffled_gram, shuffled_target = statistics_from_examples(probabilities, shuffled_labels)
    shuffled_theta, shuffled_receipt = solve_constrained_map(shuffled_gram, shuffled_target)
    fit_ids_hash = canonical_hash([str(row["source_id"]) for row in fit])
    candidate = _head_record(
        "proper_loss_monotone",
        candidate_theta,
        candidate_receipt,
        fit,
        gram,
        target,
        labels,
        constraints="range_monotonicity_movement_0.10",
    )
    candidate["constraint_residuals"] = _constraint_residuals(candidate_theta)
    candidate["head_sha256"] = canonical_hash(
        {key: value for key, value in candidate.items() if key != "head_sha256"}
    )
    range_only = _head_record(
        "unconstrained_nine_knot",
        range_theta,
        range_receipt,
        fit,
        gram,
        target,
        labels,
        constraints="range_only",
    )
    range_only["constraint_residuals"] = {
        "range": float(range_receipt["range_residual"]),
        "maximum": float(range_receipt["range_residual"]),
    }
    range_only["head_sha256"] = canonical_hash(
        {key: value for key, value in range_only.items() if key != "head_sha256"}
    )
    shuffled = _head_record(
        "shuffled_label_control",
        shuffled_theta,
        shuffled_receipt,
        fit,
        shuffled_gram,
        shuffled_target,
        shuffled_labels,
        constraints="range_monotonicity_movement_0.10",
    )
    shuffled["label_permutation_sha256"] = canonical_hash(shuffled_labels)
    shuffled["head_sha256"] = canonical_hash(
        {key: value for key, value in shuffled.items() if key != "head_sha256"}
    )
    heads: JsonDict = {
        "proper_loss_monotone": candidate,
        "raw_original": {
            "name": "raw_original",
            "parameter_count": 0,
            "fit_brier": _mean_brier(fit, lambda probability: probability),
            "fit_source_ids_sha256": fit_ids_hash,
            "converged": True,
        },
        "temperature_original": {
            "name": "temperature_original",
            "parameter_count": 1,
            "temperature": selected_temperature["temperature"],
            "fit_brier": selected_temperature["fit_brier"],
            "candidate_grid": temperatures,
            "selection_role": "fit",
            "fit_source_ids_sha256": fit_ids_hash,
            "converged": True,
        },
        "unconstrained_nine_knot": range_only,
    }
    for name in ("raw_original", "temperature_original"):
        heads[name]["head_sha256"] = canonical_hash(heads[name])
    comparator_tune_losses = {
        "raw_original": _mean_brier(tune, lambda probability: probability),
        "temperature_original": _mean_brier(
            tune,
            lambda probability: _temperature_probability(
                probability, float(selected_temperature["temperature"])
            ),
        ),
        "unconstrained_nine_knot": _mean_brier(
            tune, lambda probability: map_probability(probability, range_theta)
        ),
    }
    strongest_name = min(
        comparator_tune_losses,
        key=lambda name: (comparator_tune_losses[name], name),
    )
    checkpoint: JsonDict = {
        "schema": "carnot.exp7576.v662.proper_loss_checkpoint.v1",
        "feature_schema": {
            "inputs": ["p_original"],
            "source": "exp7575_frozen_original_probability",
            "source_contrasts_used": False,
            "online_or_test_labels_used": False,
        },
        "objective": {
            "loss": "binary_brier",
            "identity_penalty_lambda": RIDGE_MASS,
            "knot_locations": KNOTS.tolist(),
            "movement_bound": MOVEMENT_BOUND,
            "solver_tolerance": SOLVER_TOLERANCE,
        },
        "fit_source_ids_sha256": fit_ids_hash,
        "heads": deepcopy(heads),
        "shuffled_label_control": deepcopy(shuffled),
        "strongest_comparator": {
            "name": strongest_name,
            "tune_brier": comparator_tune_losses[strongest_name],
            "all_tune_brier": comparator_tune_losses,
            "selected_on_role": "tune",
            "frozen_before_policy_access": True,
        },
    }
    checkpoint_sha256 = canonical_hash(checkpoint)
    candidate_ready = bool(
        not fallback_used
        and candidate_receipt.get("converged") is True
        and candidate["constraint_residuals"]["maximum"] <= 1e-7
        and range_receipt.get("converged") is True
        and shuffled_receipt.get("converged") is True
    )
    baseline_ready = bool(
        range_receipt.get("converged") is True
        and all(head.get("fit_source_ids_sha256") == fit_ids_hash for head in heads.values())
    )
    bundle: JsonDict = {
        **checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "fit_source_ids": [str(row["source_id"]) for row in fit],
        "tune_source_ids": [str(row["source_id"]) for row in tune],
        "fit_source_ids_sha256": fit_ids_hash,
        "tune_source_ids_sha256": canonical_hash([str(row["source_id"]) for row in tune]),
        "candidate_fallback": {
            "used": fallback_used,
            "name": "identity_forecast_control" if fallback_used else None,
            "trained_benefit_claimed": False,
        },
        "proper_loss_fit_ready_score": int(candidate_ready),
        "baseline_ready_score": int(baseline_ready),
        "heads_frozen_before_policy_access": True,
        "trained_benefit_claimed": False,
        "candidate_tune_brier_diagnostic": _mean_brier(
            tune, lambda probability: map_probability(probability, candidate_theta)
        ),
    }
    bundle["policy_action_contract"] = _policy_action_contract(bundle, policy)
    bundle["bundle_sha256"] = canonical_hash(
        {key: deepcopy(value) for key, value in bundle.items() if key != "bundle_sha256"}
    )
    return bundle


def _arm_probability(bundle: Mapping[str, Any], arm: str, probability: float) -> float:
    heads = bundle["heads"]
    if arm == "proper_loss_monotone":
        return map_probability(probability, heads[arm]["theta"])
    if arm == "raw_original":
        return probability
    if arm == "temperature_original":
        return _temperature_probability(probability, float(heads[arm]["temperature"]))
    if arm == "unconstrained_nine_knot":
        return map_probability(probability, heads[arm]["theta"])
    raise ValueError(f"comparison_arm_invalid:{arm}")


def build_comparison_rows(
    bundle: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
    tune_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build one raw Brier row for every fit/tune unit and frozen arm."""

    roles = {
        "fit": _validate_role_rows(fit_rows, "fit"),
        "tune": _validate_role_rows(tune_rows, "tune"),
    }
    arms = (
        "proper_loss_monotone",
        "raw_original",
        "temperature_original",
        "unconstrained_nine_knot",
    )
    rows: list[JsonDict] = []
    for role, sources in roles.items():
        for source in sources:
            label = int(source["label"])
            base = float(source["probability"])
            for arm in arms:
                probability = _arm_probability(bundle, arm, base)
                loss = _brier(probability, label)
                rows.append(
                    {
                        "unit_id": str(source["source_id"]),
                        "arm": arm,
                        "phase": role,
                        "probability": probability,
                        "label": label,
                        "raw_squared_error_numerator": loss,
                        "raw_squared_error_denominator": 1,
                        "brier": loss,
                        "energies": normalized_binary_energies(probability),
                        "metric_direction": "lower_brier_is_better",
                        "seed": FIT_SEED,
                        "censored": False,
                        "provenance": "exp7575_cached_original_probability",
                    }
                )
    return rows


def reduce_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Recompute pooled and role-specific metrics from raw row arithmetic."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("raw_squared_error_denominator") != 1:
            raise ValueError("row_denominator_invalid")
        expected = _brier(float(row["probability"]), int(row["label"]))
        if not math.isclose(expected, float(row.get("raw_squared_error_numerator")), abs_tol=1e-12):
            raise ValueError("row_numerator_invalid")
        if row.get("metric_direction") != "lower_brier_is_better":
            raise ValueError("row_direction_invalid")
        grouped[str(row["arm"])].append(row)
    if not grouped:
        raise ValueError("comparison_rows_empty")
    reduced: dict[str, JsonDict] = {}
    for arm, arm_rows in grouped.items():
        numerator = float(sum(float(row["raw_squared_error_numerator"]) for row in arm_rows))
        by_phase: dict[str, JsonDict] = {}
        for phase in ("fit", "tune"):
            selected = [row for row in arm_rows if row.get("phase") == phase]
            if selected:
                phase_numerator = float(
                    sum(float(row["raw_squared_error_numerator"]) for row in selected)
                )
                by_phase[phase] = {
                    "raw_numerator": phase_numerator,
                    "raw_denominator": len(selected),
                    "mean_brier": phase_numerator / len(selected),
                }
        reduced[arm] = {
            "raw_numerator": numerator,
            "raw_denominator": len(arm_rows),
            "mean_brier": numerator / len(arm_rows),
            "by_phase": by_phase,
        }
    return reduced


def run_qualification_controls(
    bundle: Mapping[str, Any], fit_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Run mathematical and custody checks without creating empirical rows."""

    fit = _validate_role_rows(fit_rows, "fit")
    candidate = bundle["heads"]["proper_loss_monotone"]
    theta = np.asarray(candidate["theta"], dtype=float)
    grid = np.linspace(0.0, 1.0, 1001)
    identity_error = float(max(abs(map_probability(float(value), KNOTS) - value) for value in grid))
    sensitivity: list[JsonDict] = []
    for perturbation in PERTURBATIONS:
        changes = [
            abs(
                map_probability(min(1.0, max(0.0, float(row["probability"]) + perturbation)), theta)
                - map_probability(float(row["probability"]), theta)
            )
            for row in fit
        ]
        sensitivity.append(
            {
                "perturbation": perturbation,
                "mean_absolute_change": float(np.mean(changes)),
                "maximum_absolute_change": float(np.max(changes)),
                "diagnostic_only": True,
                "relabeled_observation": False,
            }
        )
    epsilon = 1e-6
    parameter_changes = []
    for index, location in enumerate(KNOTS):
        changed = KNOTS.copy()
        changed[index] += epsilon
        parameter_changes.append(
            float(
                abs(
                    map_probability(float(location), changed)
                    - map_probability(float(location), KNOTS)
                )
            )
        )
    shuffled = bundle["shuffled_label_control"]
    controls: JsonDict = {
        "identity_reproduction": {
            "grid_size": len(grid),
            "maximum_error": identity_error,
            "passed": bool(identity_error <= 1e-12),
        },
        "candidate_constraints": {
            **deepcopy(candidate["constraint_residuals"]),
            "passed": candidate["constraint_residuals"]["maximum"] <= 1e-7,
        },
        "option_order_mapping": {
            "source": "exp7575_authenticated_six_native_cells",
            "six_cell_rows": sum(len(row["request_hashes"]) == 6 for row in fit),
            "passed": all(len(row["request_hashes"]) == 6 for row in fit),
        },
        "parameter_sensitivity": {
            "epsilon": epsilon,
            "absolute_changes": parameter_changes,
            "all_knots_responsive": bool(all(value > 0.0 for value in parameter_changes)),
        },
        "shuffled_labels": {
            "fit_brier_on_true_labels": shuffled["fit_brier"],
            "label_permutation_sha256": shuffled["label_permutation_sha256"],
            "converged": shuffled["solver_receipt"]["converged"],
            "empirical_benefit_claim": False,
        },
        "input_sensitivity": sensitivity,
    }
    controls["passed"] = bool(
        controls["identity_reproduction"]["passed"]
        and controls["candidate_constraints"]["passed"]
        and controls["option_order_mapping"]["passed"]
        and controls["parameter_sensitivity"]["all_knots_responsive"]
        and controls["shuffled_labels"]["converged"]
    )
    controls["controls_sha256"] = canonical_hash(
        {key: value for key, value in controls.items() if key != "controls_sha256"}
    )
    return controls


def _sidecar_receipt(path: Path, root: Path) -> JsonDict:
    """Bind one persisted sidecar to exact bytes and a stable path label."""

    return _source_hash(path, root)


def _checkpoint_payload(bundle: Mapping[str, Any]) -> JsonDict:
    """Select the frozen learning state without policy labels or runtime metadata."""

    return {
        "schema": bundle["schema"],
        "feature_schema": deepcopy(bundle["feature_schema"]),
        "objective": deepcopy(bundle["objective"]),
        "fit_source_ids_sha256": bundle["fit_source_ids_sha256"],
        "heads": deepcopy(bundle["heads"]),
        "shuffled_label_control": deepcopy(bundle["shuffled_label_control"]),
        "strongest_comparator": deepcopy(bundle["strongest_comparator"]),
    }


def write_fit_sidecars(
    output_root: Path,
    bundle: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, JsonDict]:
    """Persist the checkpoint, controls, and head manifest with exact hashes."""

    output_root = output_root.resolve()
    root = root.resolve()
    checkpoint = {
        "schema": "carnot.exp7576.v662.persisted_checkpoint.v1",
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "checkpoint_payload": _checkpoint_payload(bundle),
        "heads": deepcopy(bundle["heads"]),
        "lifecycle": ["predict", "release", "update", "persist", "reload"],
    }
    if canonical_hash(checkpoint["checkpoint_payload"]) != bundle["checkpoint_sha256"]:
        raise ValueError("checkpoint_hash_invalid")
    checkpoint_path = output_root / CHECKPOINT_PATH
    controls_path = output_root / CONTROL_PATH
    atomic_json(checkpoint_path, checkpoint)
    atomic_json(controls_path, deepcopy(dict(controls)))
    checkpoint_receipt = _sidecar_receipt(checkpoint_path, root)
    controls_receipt = _sidecar_receipt(controls_path, root)
    head_manifest: JsonDict = {
        "schema": "carnot.exp7576.v662.head_manifest.v1",
        "objective": deepcopy(bundle["objective"]),
        "feature_schema": deepcopy(bundle["feature_schema"]),
        "heads": deepcopy(bundle["heads"]),
        "strongest_comparator": deepcopy(bundle["strongest_comparator"]),
        "checkpoint": checkpoint_receipt,
        "qualification_controls": controls_receipt,
        "frozen_before_policy_access": True,
    }
    head_manifest["manifest_sha256"] = canonical_hash(head_manifest)
    manifest_path = output_root / HEAD_MANIFEST_PATH
    atomic_json(manifest_path, head_manifest)
    return {
        "checkpoint": checkpoint_receipt,
        "qualification_controls": controls_receipt,
        "head_manifest": _sidecar_receipt(manifest_path, root),
    }


def reload_checkpoint(path: Path) -> JsonDict:
    """Reload and authenticate one frozen checkpoint in a fresh-reader shape."""

    checkpoint = _load_object(path)
    if canonical_hash(checkpoint.get("checkpoint_payload")) != checkpoint.get("checkpoint_sha256"):
        raise ValueError("checkpoint_hash_invalid")
    if checkpoint.get("heads") != checkpoint["checkpoint_payload"].get("heads"):
        raise ValueError("checkpoint_heads_invalid")
    if checkpoint.get("lifecycle") != ["predict", "release", "update", "persist", "reload"]:
        raise ValueError("checkpoint_lifecycle_invalid")
    return checkpoint


REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7561_v661_recalibration_prototype.py"),
    Path("python/carnot/experiment_7566_v661_energy_fit.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    SPEC_PATH,
    UPSTREAM_REQUALIFICATION_PATH,
    UPSTREAM_PROTOCOL_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "==",
) -> JsonDict:
    """Name both operands of one prerequisite without inventing evidence."""

    passed = observed in expected if op == "in" else observed == expected
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate requirements, owned resources, and frozen upstream bytes."""

    root = root.resolve()
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        observed = path.stat().st_size if path.is_file() else None
        checks.append(
            _precondition(
                "resource_readable",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "positive",
                "positive" if isinstance(observed, int) and observed > 0 else observed,
            )
        )
        if isinstance(observed, int) and observed > 0:
            hashes.append(_source_hash(relative, root))
    spec_path = root / SPEC_PATH
    spec_present = spec_path.is_file() and "REQ-CL-7576" in spec_path.read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "requirement_present",
            "continuous-learning-spec",
            SPEC_PATH.as_posix(),
            "REQ-CL-7576",
            True,
            spec_present,
        )
    )
    if not all(row["passed"] for row in checks):
        return checks, hashes
    requalification = _load_object(root / UPSTREAM_REQUALIFICATION_PATH)
    protocol = _load_object(root / UPSTREAM_PROTOCOL_PATH)
    for check, upstream, path, field, expected, observed, op in (
        (
            "requalification_ready",
            "exp7574",
            UPSTREAM_REQUALIFICATION_PATH,
            "recalibration_ready_score",
            1,
            requalification.get("recalibration_ready_score"),
            "==",
        ),
        (
            "requalification_verdict",
            "exp7574",
            UPSTREAM_REQUALIFICATION_PATH,
            "verdict_class",
            ["null", "positive", "circular_positive"],
            requalification.get("verdict_class"),
            "in",
        ),
        (
            "requalification_unflagged",
            "exp7574",
            UPSTREAM_REQUALIFICATION_PATH,
            "flagged_adversarial",
            False,
            requalification.get("flagged_adversarial"),
            "==",
        ),
        (
            "cached_roles_ready",
            "exp7575",
            UPSTREAM_PROTOCOL_PATH,
            "cached_roles_ready_score",
            1,
            protocol.get("cached_roles_ready_score"),
            "==",
        ),
        (
            "cached_roles_verdict",
            "exp7575",
            UPSTREAM_PROTOCOL_PATH,
            "verdict_class",
            ["null", "positive"],
            protocol.get("verdict_class"),
            "in",
        ),
        (
            "cached_roles_unflagged",
            "exp7575",
            UPSTREAM_PROTOCOL_PATH,
            "flagged_adversarial",
            False,
            protocol.get("flagged_adversarial"),
            "==",
        ),
    ):
        checks.append(
            _precondition(check, upstream, path.as_posix(), field, expected, observed, op=op)
        )
    receipt = protocol.get("raw_sidecars", {}).get("cached_roles", {})
    label = str(receipt.get("path") or "")
    sidecar_path = _resolved(root, label)
    observed_hash = sha256_file(sidecar_path) if sidecar_path.is_file() else None
    checks.append(
        _precondition(
            "cached_roles_hash", "exp7575", label, "sha256", receipt.get("sha256"), observed_hash
        )
    )
    if sidecar_path.is_file() and observed_hash == receipt.get("sha256"):
        hashes.append(_source_hash(sidecar_path, root))
    return checks, hashes


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failure and the first exact failed operand."""

    failures = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _field_principles() -> JsonDict:
    """Carry the interpretation guard beside each high-impact terminal field."""

    return {
        "honest_verdict": "Use a complete_ terminal prefix; completion does not establish benefit.",
        "verdict_class": "Exactly one closed-enum class; null means valid work without established benefit.",
        "flagged_adversarial": "Persist the exact terminal verification outcome; flagged evidence cannot open readiness.",
        "gate_check_summary": "Blocked work names the exact upstream path, field, operator, expected value, and observation.",
        "acceptance_gate_results": "Validity, readiness, and benefit are separate so a valid null remains usable.",
        "rows": "Every comparison unit and arm carries raw arithmetic, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Actual and planned substrates are separate; no model call forbids a live-inference claim.",
        "MODEL_SPECS": "Cached-only work has no current model specification; historical identity stays in source custody.",
        "invocation_counts": "Current loads, forwards, generations, and tokens are counted independently.",
        "duration_s": "Monotonic current work excludes inherited timing and artificial sleeps.",
        "source_artifact_hashes": "Conclusions bind exact source bytes while absent producers remain pre-gate failures.",
        "validation_receipts": "Each check binds command, worktree, exit code, and log hash.",
        "field_principles": "One-line field guards travel with the emitted values.",
        "verifier_is_oracle": "Label-accessing controls cannot support an oracle-distinct positive claim.",
        "proper_loss_fit_ready_score": "One only for a converged frozen candidate and every matched control.",
        "baseline_ready_score": "One only when raw, temperature, and equal-capacity controls use identical fit rows.",
        "head_manifest_path": "The manifest binds objective, knots, comparator choice, feature schema, and sidecars.",
        "fit_brier": "Fit loss is reported without treating it as held-out benefit.",
        "fresh_confirmatory_claim_allowed": "False because V662 cannot erase source exposure.",
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash the terminal record while excluding only its self-reference."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _terminal_base(duration_s: float) -> JsonDict:
    """Return fields common to measured and blocked terminal records."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "no_model_load": True,
        "MODEL_SPECS": [],
        "model_specs": [],
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "planned_inference_substrate": "cached_qwen_feature_learning_without_current_inference",
        "random_seed": FIT_SEED,
        "source_custody": {
            "historical_model_identity": HISTORICAL_MODEL_ID,
            "current_model_calls": 0,
            "generator_weights_changed": False,
        },
        "duration_s": float(duration_s),
        "verifier_is_oracle": True,
        "fresh_confirmatory_claim_allowed": False,
        "submitted_externally": False,
        "field_principles": _field_principles(),
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Publish complete blocked evidence without synthesizing dependent rows."""

    summary = _gate_summary(checks)
    first = summary["first_failure"] or {"check": "unknown_precondition"}
    artifact: JsonDict = {
        **_terminal_base(duration_s),
        "honest_verdict": f"complete_blocked_{first['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": summary,
        "precondition_checks": [deepcopy(dict(row)) for row in checks],
        "acceptance_gate_results": {
            "validity": {"passed": False, "reason": "external_precondition_failed"},
            "readiness": {"passed": False, "reason": "measurement_not_started"},
            "benefit": {"passed": False, "reason": "measurement_not_started"},
        },
        "rows": [],
        "comparative_reduction": {},
        "proper_loss_fit_ready_score": 0,
        "baseline_ready_score": 0,
        "head_manifest_path": None,
        "fit_brier": None,
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [],
        "raw_sidecars": {},
    }
    artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["inference_substrate"] = "blocked_no_run"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and fit loss from raw rows and frozen evidence."""

    reduction = reduce_comparison_rows(artifact.get("rows") or [])
    bundle = artifact.get("fit_bundle") or {}
    heads = bundle.get("heads") or {}
    candidate = heads.get("proper_loss_monotone") or {}
    range_only = heads.get("unconstrained_nine_knot") or {}
    controls = artifact.get("qualification_controls") or {}
    fit_hash = bundle.get("fit_source_ids_sha256")
    required_heads = (
        "proper_loss_monotone",
        "raw_original",
        "temperature_original",
        "unconstrained_nine_knot",
    )
    same_rows = bool(
        fit_hash
        and all(
            (heads.get(name) or {}).get("fit_source_ids_sha256") == fit_hash
            for name in required_heads
        )
    )
    candidate_ready = bool(
        bundle.get("candidate_fallback", {}).get("used") is False
        and candidate.get("solver_receipt", {}).get("converged") is True
        and candidate.get("constraint_residuals", {}).get("maximum", math.inf) <= 1e-7
        and range_only.get("solver_receipt", {}).get("converged") is True
        and bundle.get("shuffled_label_control", {}).get("solver_receipt", {}).get("converged")
        is True
        and bundle.get("heads_frozen_before_policy_access") is True
        and bundle.get("policy_action_contract", {}).get("passed") is True
        and controls.get("passed") is True
    )
    baseline_ready = bool(
        same_rows
        and range_only.get("solver_receipt", {}).get("converged") is True
        and heads.get("raw_original", {}).get("converged") is True
        and heads.get("temperature_original", {}).get("converged") is True
    )
    candidate_fit = reduction["proper_loss_monotone"]["by_phase"]["fit"]["mean_brier"]
    return {
        "proper_loss_fit_ready_score": int(candidate_ready),
        "baseline_ready_score": int(baseline_ready),
        "fit_brier": candidate_fit,
        "comparative_reduction": reduction,
        "benefit_established": False,
    }


def build_artifact(
    *,
    bundle: Mapping[str, Any],
    controls: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    sidecars: Mapping[str, Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Assemble a schema-complete fit record without claiming held-out benefit."""

    artifact: JsonDict = {
        **_terminal_base(duration_s),
        "honest_verdict": "complete_null_proper_loss_heads_frozen_benefit_unmeasured",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": _gate_summary(preconditions),
        "precondition_checks": [deepcopy(dict(row)) for row in preconditions],
        "rows": [deepcopy(dict(row)) for row in rows],
        "fit_bundle": deepcopy(dict(bundle)),
        "qualification_controls": deepcopy(dict(controls)),
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "raw_sidecars": deepcopy(dict(sidecars)),
        "head_manifest_path": sidecars["head_manifest"]["path"],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "fit_brier": float(bundle["heads"]["proper_loss_monotone"]["fit_brier"]),
        "proper_loss_fit_ready_score": int(bundle["proper_loss_fit_ready_score"]),
        "baseline_ready_score": int(bundle["baseline_ready_score"]),
        "fresh_confirmatory_claim_allowed": False,
        "analytical_positive_control_disposition": "not_a_positive_claim",
        "capability_e2e": {
            "learning_lifecycle": ["predict", "release", "update", "persist", "reload"],
            "numbered_runtime_e2e": [],
            "numbered_runtime_e2e_reason": "fit-only cached reporting changes no ARC, binding, or shared sampler runtime",
            "fresh_process_reader_names": [
                str(row.get("name"))
                for row in validation_receipts
                if row.get("name") in TERMINAL_CHECK_NAMES
            ],
        },
    }
    reduced = independent_reduce(artifact)
    artifact["comparative_reduction"] = reduced["comparative_reduction"]
    artifact["proper_loss_fit_ready_score"] = reduced["proper_loss_fit_ready_score"]
    artifact["baseline_ready_score"] = reduced["baseline_ready_score"]
    if not reduced["proper_loss_fit_ready_score"] or not reduced["baseline_ready_score"]:
        artifact["honest_verdict"] = "complete_disqualified_proper_loss_fit_not_ready"
        artifact["verdict_class"] = "disqualified"
    artifact["acceptance_gate_results"] = {
        "validity": {
            "passed": _gate_summary(preconditions)["passed"] and bool(controls.get("passed")),
            "checks": "authenticated_inputs_and_structural_qualifications",
        },
        "readiness": {
            "passed": bool(
                artifact["proper_loss_fit_ready_score"] and artifact["baseline_ready_score"]
            ),
            "checks": "converged_frozen_candidate_and_matched_controls",
        },
        "benefit": {
            "passed": True,
            "established": False,
            "checks": "heldout_benefit_intentionally_unmeasured_in_fit_only_stage",
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validate_sidecars(artifact: Mapping[str, Any], root: Path) -> None:
    """Rehash every persisted head/checkpoint/control byte before trusting it."""

    sidecars = artifact.get("raw_sidecars") or {}
    for name in ("checkpoint", "qualification_controls", "head_manifest"):
        receipt = sidecars.get(name) or {}
        path = _resolved(root, str(receipt.get("path") or ""))
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            raise ValueError(f"sidecar_hash_invalid:{name}")
    checkpoint = reload_checkpoint(_resolved(root, str(sidecars["checkpoint"]["path"])))
    if checkpoint.get("checkpoint_sha256") != artifact.get("fit_bundle", {}).get(
        "checkpoint_sha256"
    ):
        raise ValueError("checkpoint_identity_invalid")
    controls = _load_object(_resolved(root, str(sidecars["qualification_controls"]["path"])))
    control_hash = canonical_hash(
        {key: value for key, value in controls.items() if key != "controls_sha256"}
    )
    if controls.get("controls_sha256") != control_hash:
        raise ValueError("controls_hash_invalid")
    manifest = _load_object(_resolved(root, str(sidecars["head_manifest"]["path"])))
    manifest_hash = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != manifest_hash:
        raise ValueError("head_manifest_hash_invalid")
    if manifest.get("checkpoint", {}).get("sha256") != sidecars["checkpoint"].get("sha256"):
        raise ValueError("head_manifest_checkpoint_invalid")


def _validate_receipts(receipts: Sequence[Mapping[str, Any]]) -> None:
    """Require one passing bounded receipt for each declared validation reader."""

    by_name = {str(row.get("name")): row for row in receipts}
    missing = [
        name for name in (*REQUIRED_VALIDATION_NAMES, *TERMINAL_CHECK_NAMES) if name not in by_name
    ]
    if missing:
        raise ValueError(f"validation_receipt_missing:{missing}")
    for name in (*REQUIRED_VALIDATION_NAMES, *TERMINAL_CHECK_NAMES):
        row = by_name[name]
        if (
            row.get("passed") is not True
            or row.get("exit_code") != 0
            or row.get("timed_out") is True
        ):
            raise ValueError(f"validation_receipt_failed:{name}")
        if not row.get("command") or not row.get("worktree") or not row.get("log_sha256"):
            raise ValueError(f"validation_receipt_incomplete:{name}")


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Cold-check identity, reductions, claims, hashes, and command receipts."""

    artifact = deepcopy(dict(value))
    root = root.resolve()
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        raise ValueError("artifact_checksum_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        raise ValueError("model_specs_not_empty")
    if artifact.get("no_model_load") is not True:
        raise ValueError("no_model_load_invalid")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        raise ValueError("invocation_counts_invalid")
    if artifact.get("fresh_confirmatory_claim_allowed") is not False:
        raise ValueError("freshness_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        raise ValueError("verdict_class_invalid")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        raise ValueError("terminal_prefix_invalid")
    if set(_field_principles()) - set(artifact.get("field_principles") or {}):
        raise ValueError("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") != [] or artifact.get("proper_loss_fit_ready_score") != 0:
            raise ValueError("blocked_measurement_invalid")
        failure = artifact.get("gate_check_summary", {}).get("first_failure") or {}
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not required <= set(failure):
            raise ValueError("blocked_gate_summary_invalid")
        return {"valid": True, "blocked": True}
    _validate_sidecars(artifact, root)
    reduced = independent_reduce(artifact)
    if reduced["comparative_reduction"] != artifact.get("comparative_reduction"):
        raise ValueError("comparative_reduction_mismatch")
    if reduced["proper_loss_fit_ready_score"] != artifact.get("proper_loss_fit_ready_score"):
        raise ValueError("proper_loss_fit_readiness_mismatch")
    if reduced["baseline_ready_score"] != artifact.get("baseline_ready_score"):
        raise ValueError("baseline_readiness_mismatch")
    if not math.isclose(float(artifact.get("fit_brier")), reduced["fit_brier"], abs_tol=1e-12):
        raise ValueError("fit_brier_mismatch")
    expected_class = (
        "null"
        if reduced["proper_loss_fit_ready_score"] and reduced["baseline_ready_score"]
        else "disqualified"
    )
    if artifact.get("verdict_class") != expected_class:
        raise ValueError("verdict_reduction_mismatch")
    if require_validation:
        _validate_receipts(artifact.get("validation_receipts") or [])
    return {"valid": True, "blocked": False, **reduced}


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Read and validate exact serialized bytes through the public cold path."""

    return validate_artifact(_load_object(path), root=root)


def independent_reduce_artifact(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Independently reduce one serialized artifact after sidecar authentication."""

    artifact = _load_object(path)
    validate_artifact(artifact, root=root)
    return independent_reduce(artifact)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit one flushed boundary with truthful monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7576] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _write_affected_manifest(root: Path) -> JsonDict:
    """Freeze the exact validation scope before any measurement begins."""

    files = (
        *AFFECTED_MANIFEST.test_paths,
        *AFFECTED_MANIFEST.changed_modules,
        *AFFECTED_MANIFEST.static_paths,
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "worktree": str(root.resolve()),
        "test_paths": list(AFFECTED_MANIFEST.test_paths),
        "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
        "static_paths": list(AFFECTED_MANIFEST.static_paths),
        "file_hashes": {path: sha256_file(root / path) for path in files},
    }
    atomic_json(root / AFFECTED_MANIFEST_PATH, manifest)
    return manifest


def _prepare_private_parents(command: validation_scope.CommandSpec) -> None:
    """Create private pytest and coverage parents immediately before a child."""

    for argument in command.argv:
        if argument.startswith("--basetemp=") or argument.startswith("--data-file="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(getattr(command, "command_environment", ()))
    if coverage_file := environment.get("COVERAGE_FILE"):
        Path(coverage_file).parent.mkdir(parents=True, exist_ok=True)


def _run_planned(
    root: Path,
    commands: Sequence[PlannedCommand],
    *,
    log_dir: Path,
) -> list[JsonDict]:  # pragma: no cover - bounded capability subprocesses.
    """Run one command at a time so every private parent is freshly present."""

    receipts: list[JsonDict] = []
    for index, planned in enumerate(commands):
        _prepare_private_parents(planned.spec)
        rows = run_categorized_commands(
            root,
            [planned],
            log_dir=log_dir / f"{index:02d}_{planned.spec.name}",
            heartbeat_s=60.0,
        )
        rows[0]["worktree"] = str(root.resolve())
        receipts.extend(rows)
    return receipts


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:
    """Build fresh replay, reduction, adversarial, and strict row readers."""

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
                "--root",
                ".",
                "--cold-replay",
                candidate.as_posix(),
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
                "--root",
                ".",
                "--independent-reduce",
                candidate.as_posix(),
            ),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", candidate.as_posix()),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                candidate.as_posix(),
            ),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(command, "required_terminal_validation", True) for command in commands]


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Authenticate, validate, fit, freeze, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    progress(started, "startup", "flushed_progress", root=root)
    progress(started, "preconditions", "before_authentication")
    checks, source_hashes = collect_preconditions(root)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(
            checks,
            source_hashes,
            duration_s=time.monotonic() - started,
        )
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "publish", "after_atomic_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "after_authentication", completed_units=len(checks))

    progress(started, "validation_manifest", "before_freeze")
    _write_affected_manifest(root)
    progress(started, "validation_manifest", "after_freeze")
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7576-validation-", dir="/tmp"))
    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    affected = _run_planned(
        root,
        [PlannedCommand(command, "required_affected_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation" / "affected",
    )
    affected_reduction = reduce_affected_receipts(root, AFFECTED_MANIFEST, affected)
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise RuntimeError(f"affected_validation_failed:{affected_reduction}")

    progress(started, "model_load", "before_no_model_load_declaration")
    progress(started, "model_load", "after_no_model_load_declaration", model_calls=0)
    progress(started, "cached_role_load", "before_benchmark")
    roles = load_frozen_fit_roles(root, source_hashes)
    progress(
        started,
        "cached_role_load",
        "after_benchmark",
        completed_units=sum(len(rows) for rows in roles.values()),
    )
    progress(started, "proper_loss_fit", "before_benchmark", fit_units=len(roles["fit"]))
    bundle = fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    progress(started, "proper_loss_fit", "after_benchmark", completed_units=len(roles["fit"]))
    progress(started, "qualifications", "before_benchmark")
    controls = run_qualification_controls(bundle, roles["fit"])
    rows = build_comparison_rows(bundle, roles["fit"], roles["tune"])
    progress(started, "qualifications", "after_benchmark", completed_units=len(rows))
    progress(started, "persist_reload", "before_benchmark")
    sidecars = write_fit_sidecars(root, bundle, controls, root=root)
    reload_checkpoint(root / CHECKPOINT_PATH)
    sidecars["affected_validation_manifest"] = _sidecar_receipt(root / AFFECTED_MANIFEST_PATH, root)
    progress(started, "persist_reload", "after_benchmark", completed_units=1)

    deduplicated_hashes = {(str(row["path"]), str(row["sha256"])): row for row in source_hashes}
    candidate = build_artifact(
        bundle=bundle,
        controls=controls,
        rows=rows,
        preconditions=checks,
        source_hashes=list(deduplicated_hashes.values()),
        sidecars=sidecars,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(candidate, root=root, require_validation=False)
    atomic_json(root / TERMINAL_CANDIDATE_PATH, candidate)
    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    terminal = _run_planned(
        root,
        _terminal_commands(TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation" / "terminal",
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    sidecars["terminal_candidate"] = _sidecar_receipt(root / TERMINAL_CANDIDATE_PATH, root)
    final = build_artifact(
        bundle=bundle,
        controls=controls,
        rows=rows,
        preconditions=checks,
        source_hashes=list(deduplicated_hashes.values()),
        sidecars=sidecars,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
    )
    validate_artifact(final, root=root)
    progress(started, "publish", "before_atomic_terminal", verdict=final["verdict_class"])
    atomic_json(root / RESULT_PATH, final)
    validate_artifact(_load_object(root / RESULT_PATH), root=root)
    progress(started, "publish", "after_atomic_terminal", path=RESULT_PATH)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the producer or a bounded serialized-evidence reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        path = _resolved(root, str(args.cold_replay))
        result = validate_artifact(_load_object(path), root=root, require_validation=False)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.independent_reduce is not None:
        path = _resolved(root, str(args.independent_reduce))
        artifact = _load_object(path)
        validate_artifact(artifact, root=root, require_validation=False)
        print(json.dumps(independent_reduce(artifact), sort_keys=True), flush=True)
        return 0
    run_experiment(root, args.date)
    return 0
