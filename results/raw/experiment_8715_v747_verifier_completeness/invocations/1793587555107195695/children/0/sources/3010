"""Evaluate the frozen V657 heads on their sealed held-out source groups.

The module performs CPU-only numerical readout. It writes label-free
predictions before the evidence interface can reveal test labels, so a later
metric cannot influence fitting, checkpoint choice, or the evaluated corpus.

Spec: REQ-VERIFY-7507 and SCENARIO-VERIFY-7507-*.
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
from carnot.experiment_7481_v655_typed_calibration import (
    holm_upper_bounds,
    probability_metrics,
)
from carnot.experiment_7504_v657_evidence_interface import FEATURE_NAMES, read_mode
from carnot.experiment_7505_v657_energy_fit import (
    COEFFICIENTS_PER_INPUT,
    FALSE_ACCEPT_COSTS,
    FALSE_REJECT_COST,
    REGULARIZATION_GRID,
    TIE_ORDER,
    TRAINING_SEEDS,
    expected_action,
    fit_spline_knots,
    spline_design_matrix,
    stable_sigmoid,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7507-v657-static-evaluation"
SCHEMA = "carnot.exp7507.v657.static_evaluation.v1"
RESULT_PATH = Path("results/experiment_7507_v657_static_evaluation.json")
RAW_DIR = Path("results/raw/experiment_7507_v657_static_evaluation")
MODULE_PATH = Path("python/carnot/experiment_7507_v657_static_evaluation.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7507_v657_static_evaluation.py")
TEST_PATH = Path("tests/python/test_experiment_7507_v657_static_evaluation.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
EVIDENCE_ARTIFACT = Path("results/experiment_7504_v657_evidence_interface.json")
FIT_ARTIFACT = Path("results/experiment_7505_v657_energy_fit.json")
FEATURE_PATH = Path("results/raw/experiment_7504_v657_evidence_interface/features.jsonl")
EVALUATOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/evaluators.jsonl")
ACCESS_PATH = Path(
    "results/raw/experiment_7504_v657_evidence_interface/access_exposure_manifest.json"
)
CHECKPOINT_PATH = Path("results/raw/experiment_7505_v657_energy_fit/frozen_checkpoints.json")
PREDICTION_PATH = RAW_DIR / "label_free_predictions.jsonl"
ROW_PATH = RAW_DIR / "evaluation_rows.jsonl"
POLICY_PATH = RAW_DIR / "policy_rows.jsonl"
BOOTSTRAP_PATH = RAW_DIR / "bootstrap_indices.json"
TERMINAL_CANDIDATE = RAW_DIR / "measured_terminal_candidate.json"

FIT_SEEDS = TRAINING_SEEDS
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 657_007
LEARNED_ARMS = (
    "window_gibbs",
    "whole_only_gibbs",
    "identical_ten_feature_logistic",
)
SIMPLE_ARMS = (
    "temperature_whole",
    "raw_whole_expectation",
    "raw_max_window_probability",
)
PROBABILITY_CONTROLS = ("identical_ten_feature_logistic", "whole_only_gibbs")
ESCALATION_COSTS = (0.1, 0.5, 1.0)

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
    "terminal_status",
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
    "policy_rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "static_evaluation_complete_score",
    "static_probability_value_score",
    "selective_decision_value_score",
    "probability_contrasts",
    "evaluator_settings",
    "raw_sidecars",
)


def load_json(path: Path) -> JsonDict:
    """Load one object without repairing malformed evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load object rows so independent reducers can replay exact bytes."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _path_label(path: Path, root: Path) -> str:
    """Use stable repository-relative paths when evidence is in the worktree."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_jsonl_sidecar(path: Path, rows: Sequence[Mapping[str, Any]], *, root: Path) -> JsonDict:
    """Write complete JSONL bytes atomically and return their immutable identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if path.stat().st_size >= 20 * 1024 * 1024:  # pragma: no cover - physical size guard.
        raise ValueError(f"sidecar_size_limit:{path}")
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
    }


def verify_sidecar(reference: Mapping[str, Any], root: Path) -> list[str]:
    """Verify a sidecar's path, byte hash, and row count independently."""

    path = Path(str(reference.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        return [f"sidecar_missing:{path}"]
    if sha256_file(resolved) != reference.get("sha256"):
        return [f"sidecar_sha256_mismatch:{path}"]
    if resolved.suffix == ".jsonl" and len(load_jsonl(resolved)) != reference.get("rows"):
        return [f"sidecar_row_count_mismatch:{path}"]
    return []


def reconstruct_transform(training_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recreate Exp7505's deterministic transform from label-free training rows."""

    matrix = np.asarray([row.get("features") for row in training_rows], dtype=np.float64)
    if matrix.shape != (len(training_rows), 10) or not len(training_rows):
        raise ValueError("training_feature_shape_invalid")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("training_feature_nonfinite")
    mean = matrix.mean(axis=0)
    population_sd = matrix.std(axis=0)
    safe_scale = np.where(population_sd > 0.0, population_sd, 1.0)
    knots = fit_spline_knots((matrix - mean) / safe_scale)
    return {
        "feature_names": list(FEATURE_NAMES),
        "normalization": {
            "fit_role": "training",
            "mean": mean.tolist(),
            "population_sd": population_sd.tolist(),
            "safe_scale": safe_scale.tolist(),
        },
        "spline": {
            "basis": "open_clamped_cubic_quantile",
            "knots": knots.tolist(),
            "coefficients_per_input": COEFFICIENTS_PER_INPUT,
            "trainable_parameter_count": 10 * COEFFICIENTS_PER_INPUT + 1,
        },
    }


def authenticate_transform(
    training_rows: Sequence[Mapping[str, Any]], checkpoints: Mapping[str, Any]
) -> JsonDict:
    """Reconstruct and match the exact transform hash frozen by Exp7505."""

    transform = reconstruct_transform(training_rows)
    if canonical_hash(transform) != checkpoints.get("transform_sha256"):
        raise ValueError("transform_sha256_mismatch")
    return transform


def select_frozen_candidates(
    checkpoints: Mapping[str, Any],
) -> dict[str, list[JsonDict]]:
    """Retain every fit seed at each arm's pre-evaluation regularization."""

    selected_rows = checkpoints.get("selected_heads")
    candidates = checkpoints.get("candidate_rows")
    if not isinstance(selected_rows, Mapping) or not isinstance(candidates, list):
        raise ValueError("frozen_candidate_manifest_missing")
    selected_regularization = {
        str(arm): float(row.get("regularization")) for arm, row in selected_rows.items()
    }
    output: dict[str, list[JsonDict]] = {}
    for arm in LEARNED_ARMS:
        regularization = selected_regularization.get(arm)
        rows = [
            deepcopy(row)
            for row in candidates
            if row.get("arm") == arm
            and row.get("status") == "complete"
            and float(row.get("regularization", math.nan)) == regularization
        ]
        rows.sort(key=lambda row: int(row["seed"]))
        if [int(row.get("seed", -1)) for row in rows] != list(FIT_SEEDS):
            raise ValueError(f"frozen_seed_grid_invalid:{arm}")
        for row in rows:
            checkpoint = row.get("checkpoint")
            if not isinstance(checkpoint, Mapping) or canonical_hash(checkpoint) != row.get(
                "checkpoint_sha256"
            ):
                raise ValueError(f"checkpoint_sha256_mismatch:{arm}:{row.get('seed')}")
        output[arm] = rows
    return output


def _designs(
    rows: Sequence[Mapping[str, Any]], transform: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Apply the authenticated normalization and spline without fitting."""

    matrix = np.asarray([row.get("features") for row in rows], dtype=np.float64)
    if matrix.shape != (len(rows), 10) or not np.all(np.isfinite(matrix)):
        raise ValueError("evaluation_feature_shape_invalid")
    normalization = transform["normalization"]
    normalized = (matrix - np.asarray(normalization["mean"], dtype=np.float64)) / np.asarray(
        normalization["safe_scale"], dtype=np.float64
    )
    return {
        "window_gibbs": spline_design_matrix(
            normalized, np.asarray(transform["spline"]["knots"], dtype=np.float64)
        ),
        "whole_only_gibbs": normalized[:, :1],
        "identical_ten_feature_logistic": normalized,
    }


def _checkpoint_probabilities(checkpoint: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    """Apply one immutable linear-in-design checkpoint."""

    coefficient = np.asarray(checkpoint.get("coefficient"), dtype=np.float64)
    if coefficient.shape != (design.shape[1],) or not np.all(np.isfinite(coefficient)):
        raise ValueError("checkpoint_coefficient_shape_invalid")
    bias = float(checkpoint.get("bias"))
    if not math.isfinite(bias):
        raise ValueError("checkpoint_bias_nonfinite")
    return stable_sigmoid(design @ coefficient + bias)


def _temperature_probability(probability: float, temperature: float) -> float:
    """Apply the scalar temperature frozen on calibration-tuning Brier."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    logit = math.log(clipped / (1.0 - clipped))
    return float(stable_sigmoid(np.asarray([logit / temperature]))[0])


def score_label_free_predictions(
    rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[str, Any],
    transform: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit probabilities without accepting any evaluator label field."""

    if any("label" in row for row in rows):
        raise ValueError("label_visible_during_prediction")
    selected = select_frozen_candidates(checkpoints)
    designs = _designs(rows, transform)
    output: list[JsonDict] = []
    for arm in LEARNED_ARMS:
        for candidate in selected[arm]:
            probabilities = _checkpoint_probabilities(candidate["checkpoint"], designs[arm])
            for row, probability in zip(rows, probabilities, strict=True):
                output.append(
                    {
                        "group_id": str(row["group_id"]),
                        "source_hash": str(row["source_hash"]),
                        "role": str(row["role"]),
                        "arm": arm,
                        "fit_seed": int(candidate["seed"]),
                        "probability": float(probability),
                    }
                )
    temperature = float(checkpoints["temperature_baseline"]["selected_temperature"])
    for row in rows:
        raw_whole = float(row["raw_whole_expectation"])
        raw_maximum = float(row["raw_max_window_probability"])
        controls = {
            "temperature_whole": _temperature_probability(raw_whole, temperature),
            "raw_whole_expectation": raw_whole,
            "raw_max_window_probability": raw_maximum,
        }
        for arm, probability in controls.items():
            output.append(
                {
                    "group_id": str(row["group_id"]),
                    "source_hash": str(row["source_hash"]),
                    "role": str(row["role"]),
                    "arm": arm,
                    "fit_seed": None,
                    "probability": float(probability),
                }
            )
    return sorted(output, key=lambda row: (row["group_id"], row["arm"], row["fit_seed"] or -1))


def fixture_policy_manifest() -> JsonDict:
    """Describe the immutable direct-cost grid without calibration examples."""

    rows = [
        {
            "false_accept_cost": false_accept,
            "false_reject_cost": FALSE_REJECT_COST,
            "escalation_cost": escalation,
            "frozen": True,
        }
        for false_accept in FALSE_ACCEPT_COSTS
        for escalation in ESCALATION_COSTS
    ]
    value: JsonDict = {
        "decision_rule": "minimum_expected_cost_from_calibrated_unsupported_probability",
        "tie_breaking": list(TIE_ORDER),
        "rows": rows,
    }
    value["policy_sha256"] = canonical_hash(value)
    return value


def _cost(action: str, label: int, *, false_accept: float, escalation: float) -> float:
    """Charge a wrong terminal action or the registered escalation fee."""

    if action == "accept":
        return false_accept if label == 1 else 0.0
    if action == "reject":
        return FALSE_REJECT_COST if label == 0 else 0.0
    if action == "escalate":
        return escalation
    raise ValueError("decision_action_invalid")


def loss_fields(probability: float, label: int) -> JsonDict:
    """Return proper per-source losses for unsupported-content probability."""

    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0 or label not in (0, 1):
        raise ValueError("probability_or_label_invalid")
    clipped = min(max(numeric, 1e-9), 1.0 - 1e-9)
    return {
        "brier": (clipped - label) ** 2,
        "log_loss": -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped)),
    }


def attach_losses_and_decisions(row: Mapping[str, Any], label: int) -> JsonDict:
    """Attach labels only after prediction freeze, retaining every cost cell."""

    output = deepcopy(dict(row))
    probability = float(output["probability"])
    output["label"] = int(label)
    output.update(loss_fields(probability, int(label)))
    decisions = []
    for false_accept in FALSE_ACCEPT_COSTS:
        for escalation in ESCALATION_COSTS:
            action = expected_action(
                probability,
                false_accept_cost=false_accept,
                false_reject_cost=FALSE_REJECT_COST,
                escalation_cost=escalation,
            )
            decisions.append(
                {
                    "cell_id": f"fa={false_accept:g}|fr=1|esc={escalation:g}",
                    "false_accept_cost": false_accept,
                    "false_reject_cost": FALSE_REJECT_COST,
                    "escalation_cost": escalation,
                    "action": action,
                    "escalated": action == "escalate",
                    "cost": _cost(
                        action,
                        int(label),
                        false_accept=false_accept,
                        escalation=escalation,
                    ),
                }
            )
    output["decision_costs"] = decisions
    output["status"] = "complete"
    output["failed"] = False
    output["censored"] = False
    return output


def attach_evaluation_labels(
    predictions: Sequence[Mapping[str, Any]], evaluator_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Join the authorized test labels and convert support to unsupported."""

    labels: dict[tuple[str, str], int] = {}
    for row in evaluator_rows:
        if row.get("role") != "test":
            continue
        key = (str(row.get("group_id") or ""), "test")
        if key in labels:
            raise ValueError(f"evaluation_label_duplicate:{key[0]}")
        raw = row.get("label")
        if raw not in (0, 1):
            raise ValueError(f"evaluation_label_invalid:{key[0]}")
        labels[key] = 1 - int(raw)
    predicted_groups = {(str(row["group_id"]), str(row["role"])) for row in predictions}
    if predicted_groups != set(labels):
        raise ValueError("evaluation_group_mismatch")
    return [
        attach_losses_and_decisions(row, labels[(str(row["group_id"]), str(row["role"]))])
        for row in predictions
    ]


def evaluator_settings(
    *,
    draws: int = BOOTSTRAP_DRAWS,
    confirmatory_allowed: bool = True,
    simple_baseline_arm: str = "raw_whole_expectation",
    expected_groups: int | None = None,
) -> JsonDict:
    """Return immutable thresholds without consulting evaluation outcomes."""

    return {
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "minimum_groups": 100,
        "minimum_per_class": 20,
        "minimum_brier_delta": -0.01,
        "maximum_log_loss_delta": 0.01,
        "minimum_non_escalated_coverage": 0.20,
        "holm_alpha": 0.05,
        "probability_candidate": "window_gibbs",
        "probability_controls": list(PROBABILITY_CONTROLS),
        "simple_baseline_arm": simple_baseline_arm,
        "policy_baseline_arm": "temperature_whole",
        "confirmatory_allowed": confirmatory_allowed,
        "expected_groups": expected_groups,
        "inference_unit": "unique_source_group",
        "seed_reduction": "average_losses_within_source_before_paired_inference",
    }


def _averaged_arm_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], JsonDict]:
    """Average repeated fits within a source while preserving real seed counts."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("failed") is not True and row.get("censored") is not True:
            grouped[(str(row["group_id"]), str(row["arm"]))].append(row)
    output: dict[tuple[str, str], JsonDict] = {}
    for key, group in grouped.items():
        labels = {int(row["label"]) for row in group}
        sources = {str(row["source_hash"]) for row in group}
        if len(labels) != 1 or len(sources) != 1:
            raise ValueError(f"source_row_disagreement:{key[0]}:{key[1]}")
        output[key] = {
            "group_id": key[0],
            "arm": key[1],
            "source_hash": next(iter(sources)),
            "label": next(iter(labels)),
            "probability": float(np.mean([float(row["probability"]) for row in group])),
            "brier": float(np.mean([float(row["brier"]) for row in group])),
            "log_loss": float(np.mean([float(row["log_loss"]) for row in group])),
            "seed_rows": len(group),
        }
    return output


def bootstrap_indices(group_count: int, *, draws: int, seed: int) -> np.ndarray:
    """Freeze shared source indices so every contrast resamples the same units."""

    if group_count < 1 or draws < 1:
        raise ValueError("bootstrap_shape_invalid")
    return np.random.default_rng(seed).integers(0, group_count, size=(draws, group_count))


def _paired_inference(deltas: Sequence[float], indices: np.ndarray, *, seed: int) -> JsonDict:
    """Compute one source-paired mean and retain Holm inference operands."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or indices.shape[1] != len(values):
        raise ValueError("paired_inference_shape_invalid")
    means = values[indices].mean(axis=1)
    return {
        "group_count": len(values),
        "draws": len(indices),
        "seed": seed,
        "delta": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "one_sided_p": float((1 + np.sum(means >= 0.0)) / (len(means) + 1)),
        "bootstrap_means": means.tolist(),
    }


def reduce_probability_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
    confirmatory_allowed: bool = True,
    simple_baseline_arm: str = "raw_whole_expectation",
) -> JsonDict:
    """Reduce proper scores and registered source-paired probability gates."""

    averaged = _averaged_arm_rows(rows)
    candidate_groups = sorted(group for group, arm in averaged if arm == "window_gibbs")
    if not candidate_groups:
        raise ValueError("window_gibbs_rows_missing")
    labels = [int(averaged[(group, "window_gibbs")]["label"]) for group in candidate_groups]
    class_support = {
        "supported": labels.count(0),
        "contains_unsupported": labels.count(1),
    }
    support_passed = len(candidate_groups) >= 100 and min(class_support.values()) >= 20
    support = {
        "n_groups": len(candidate_groups),
        "class_support": class_support,
        "minimum_groups": 100,
        "minimum_per_class": 20,
        "passed": support_passed,
    }
    for arm in LEARNED_ARMS:
        counts = {averaged[(group, arm)]["seed_rows"] for group in candidate_groups}
        if counts != {len(FIT_SEEDS)}:
            raise ValueError(f"seed_rows_invalid:{arm}")
    indices = bootstrap_indices(len(candidate_groups), draws=draws, seed=seed)
    raw_comparisons: dict[str, JsonDict] = {}
    for control in PROBABILITY_CONTROLS:
        deltas = [
            averaged[(group, "window_gibbs")]["brier"] - averaged[(group, control)]["brier"]
            for group in candidate_groups
        ]
        raw_comparisons[control] = _paired_inference(deltas, indices, seed=seed)
    brier_contrasts = holm_upper_bounds(raw_comparisons, alpha=0.05)
    simple_deltas = [
        averaged[(group, "window_gibbs")]["log_loss"]
        - averaged[(group, simple_baseline_arm)]["log_loss"]
        for group in candidate_groups
    ]
    log_loss = _paired_inference(simple_deltas, indices, seed=seed)
    log_loss.pop("bootstrap_means")
    log_loss["control"] = simple_baseline_arm
    probability_passed = (
        support_passed
        and confirmatory_allowed
        and all(
            float(row["delta"]) <= -0.01 and float(row["holm_upper"]) < 0.0
            for row in brier_contrasts.values()
        )
        and float(log_loss["delta"]) <= 0.01
    )
    metrics = {}
    for arm in (*LEARNED_ARMS, *SIMPLE_ARMS):
        arm_rows = [averaged[(group, arm)] for group in candidate_groups]
        metrics[arm] = probability_metrics(
            [row["label"] for row in arm_rows], [row["probability"] for row in arm_rows]
        )
    return {
        "support": support,
        "seed_rows_per_learned_arm": len(FIT_SEEDS),
        "metrics": metrics,
        "brier_contrasts": brier_contrasts,
        "log_loss_contrast": log_loss,
        "static_probability_value_score": int(probability_passed),
    }


def _cell(row: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    """Return one retained decision cell without recomputing its action."""

    matches = [cell for cell in row.get("decision_costs", []) if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise ValueError(f"decision_cell_invalid:{cell_id}")
    return matches[0]


def _risk(rows: Sequence[Mapping[str, Any]], cell_id: str) -> float | None:
    """Report error among terminal decisions; escalations are excluded."""

    terminal = [(row, _cell(row, cell_id)) for row in rows if not _cell(row, cell_id)["escalated"]]
    if not terminal:
        return None
    errors = [
        (cell["action"] == "accept" and int(row["label"]) == 1)
        or (cell["action"] == "reject" and int(row["label"]) == 0)
        for row, cell in terminal
    ]
    return float(np.mean(errors))


def _matched_sensitivity(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    cell_id: str,
) -> JsonDict:
    """Compare rejection sensitivity on groups terminal for both policies."""

    candidate_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate:
        candidate_by_group[str(row["group_id"])].append(row)
    baseline_by_group = {str(row["group_id"]): row for row in baseline}
    common = []
    for group, candidate_rows in candidate_by_group.items():
        baseline_row = baseline_by_group[group]
        candidate_terminal = [not _cell(row, cell_id)["escalated"] for row in candidate_rows]
        if (
            int(baseline_row["label"]) == 1
            and all(candidate_terminal)
            and not _cell(baseline_row, cell_id)["escalated"]
        ):
            common.append((candidate_rows, baseline_row))
    if not common:
        return {
            "definition": "positive groups non-escalated by both policies",
            "group_count": 0,
            "candidate": None,
            "baseline": None,
        }
    candidate_value = float(
        np.mean(
            [
                np.mean([_cell(row, cell_id)["action"] == "reject" for row in group])
                for group, _baseline in common
            ]
        )
    )
    baseline_value = float(
        np.mean([_cell(row, cell_id)["action"] == "reject" for _group, row in common])
    )
    return {
        "definition": "positive groups non-escalated by both policies",
        "group_count": len(common),
        "candidate": candidate_value,
        "baseline": baseline_value,
    }


def reduce_policy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
    confirmatory_allowed: bool = True,
) -> JsonDict:
    """Compare every frozen cost cell with one paired simple policy."""

    candidate = [row for row in rows if row.get("arm") == "window_gibbs"]
    baseline = [row for row in rows if row.get("arm") == "temperature_whole"]
    groups = sorted({str(row["group_id"]) for row in candidate})
    if not groups or {str(row["group_id"]) for row in baseline} != set(groups):
        raise ValueError("policy_group_mismatch")
    candidate_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate:
        candidate_by_group[str(row["group_id"])].append(row)
    baseline_by_group = {str(row["group_id"]): row for row in baseline}
    if any(len(candidate_by_group[group]) != len(FIT_SEEDS) for group in groups):
        raise ValueError("policy_seed_rows_invalid")
    indices = bootstrap_indices(len(groups), draws=draws, seed=seed)
    raw: dict[str, JsonDict] = {}
    pending: dict[str, JsonDict] = {}
    policy_rows: list[JsonDict] = []
    for false_accept in FALSE_ACCEPT_COSTS:
        for escalation in ESCALATION_COSTS:
            cell_id = f"fa={false_accept:g}|fr=1|esc={escalation:g}"
            deltas = []
            candidate_coverage = []
            for group in groups:
                candidate_cells = [_cell(row, cell_id) for row in candidate_by_group[group]]
                baseline_cell = _cell(baseline_by_group[group], cell_id)
                deltas.append(
                    float(np.mean([float(cell["cost"]) for cell in candidate_cells]))
                    - float(baseline_cell["cost"])
                )
                candidate_coverage.append(
                    float(np.mean([not bool(cell["escalated"]) for cell in candidate_cells]))
                )
                for row, cell in zip(candidate_by_group[group], candidate_cells, strict=True):
                    policy_rows.append(
                        {
                            "group_id": group,
                            "source_hash": row["source_hash"],
                            "arm": "window_gibbs",
                            "fit_seed": row["fit_seed"],
                            "label": row["label"],
                            **deepcopy(dict(cell)),
                        }
                    )
                policy_rows.append(
                    {
                        "group_id": group,
                        "source_hash": baseline_by_group[group]["source_hash"],
                        "arm": "temperature_whole",
                        "fit_seed": None,
                        "label": baseline_by_group[group]["label"],
                        **deepcopy(dict(baseline_cell)),
                    }
                )
            raw[cell_id] = _paired_inference(deltas, indices, seed=seed)
            pending[cell_id] = {
                "cell_id": cell_id,
                "false_accept_cost": false_accept,
                "false_reject_cost": FALSE_REJECT_COST,
                "escalation_cost": escalation,
                "candidate_coverage": float(np.mean(candidate_coverage)),
                "baseline_coverage": float(
                    np.mean(
                        [
                            not _cell(baseline_by_group[group], cell_id)["escalated"]
                            for group in groups
                        ]
                    )
                ),
                "candidate_risk": _risk(candidate, cell_id),
                "baseline_risk": _risk(baseline, cell_id),
                "coverage_matched_sensitivity": _matched_sensitivity(candidate, baseline, cell_id),
            }
    adjusted = holm_upper_bounds(raw, alpha=0.05)
    cells = []
    for cell_id, row in pending.items():
        comparison = adjusted[cell_id]
        passed = (
            row["candidate_coverage"] >= 0.20
            and float(comparison["delta"]) < 0.0
            and float(comparison["holm_upper"]) < 0.0
        )
        cells.append({**row, "comparison": comparison, "benefit_passed": passed})
    support_passed = len(groups) >= 100
    return {
        "cells": cells,
        "policy_rows": policy_rows,
        "holm_family_size": 9,
        "support_passed": support_passed,
        "selective_decision_value_score": int(
            support_passed and confirmatory_allowed and all(row["benefit_passed"] for row in cells)
        ),
    }


def reduce_rows(rows: Sequence[Mapping[str, Any]], *, settings: Mapping[str, Any]) -> JsonDict:
    """Independently derive readiness and both benefit scores from raw rows."""

    draws = int(settings["bootstrap_draws"])
    seed = int(settings["bootstrap_seed"])
    confirmatory = bool(settings["confirmatory_allowed"])
    probability = reduce_probability_rows(
        rows,
        draws=draws,
        seed=seed,
        confirmatory_allowed=confirmatory,
        simple_baseline_arm=str(settings["simple_baseline_arm"]),
    )
    policy = reduce_policy_rows(
        rows,
        draws=draws,
        seed=seed,
        confirmatory_allowed=confirmatory,
    )
    groups = {str(row["group_id"]) for row in rows}
    expected_groups = settings.get("expected_groups")
    statuses_complete = all(
        row.get("status") == "complete"
        and row.get("failed") is False
        and row.get("censored") is False
        for row in rows
    )
    expected_complete = expected_groups is None or len(groups) == int(expected_groups)
    all_arms = all(
        {str(row["arm"]) for row in rows if str(row["group_id"]) == group}
        == set((*LEARNED_ARMS, *SIMPLE_ARMS))
        for group in groups
    )
    complete = statuses_complete and expected_complete and all_arms
    return {
        "confirmatory_allowed": confirmatory,
        "static_evaluation_complete_score": int(complete),
        "static_probability_value_score": probability["static_probability_value_score"],
        "selective_decision_value_score": policy["selective_decision_value_score"],
        "probability": probability,
        "policy": {key: value for key, value in policy.items() if key != "policy_rows"},
        "policy_rows": policy["policy_rows"],
        "sample_size_budget": {
            "planned": expected_groups if expected_groups is not None else len(groups),
            "attempted": len(groups),
            "completed": len(groups) if statuses_complete else 0,
            "excluded": 0,
            "failed": sum(row.get("failed") is True for row in rows),
            "censored": sum(row.get("censored") is True for row in rows),
            "unstarted": max(0, int(expected_groups or len(groups)) - len(groups)),
            "independent_unit": "unique_source_group",
            "fit_seeds_are_independent_units": False,
            "windows_are_independent_units": False,
        },
    }


def precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    path: str | None = None,
) -> JsonDict:
    """Name the exact prerequisite operand that can block measurement."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path or upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
    }


def _source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes while preserving a stable worktree path."""

    resolved = path if path.is_absolute() else root / path
    return {
        "path": _path_label(resolved, root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "owner": "repository_or_declared_upstream",
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
    Path("results/experiment_7481_v655_typed_calibration.json"),
    Path("python/carnot/experiment_7481_v655_typed_calibration.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    EVIDENCE_ARTIFACT,
    FIT_ARTIFACT,
    FEATURE_PATH,
    ACCESS_PATH,
    CHECKPOINT_PATH,
    EVALUATOR_PATH,
    SPEC_PATH,
)


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate required bytes, terminal states, access, and frozen hashes."""

    root = root.resolve()
    checks: list[JsonDict] = []
    source_hashes: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        readable = path.is_file() and path.stat().st_size > 0
        checks.append(
            precondition_row(
                "resource_readable",
                "worktree_or_upstream",
                "path",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if readable else "missing_or_empty",
                path=relative.as_posix(),
            )
        )
        if readable:
            source_hashes.append(_source_row(path, root))
    evidence = load_json(root / EVIDENCE_ARTIFACT) if (root / EVIDENCE_ARTIFACT).is_file() else {}
    fit = load_json(root / FIT_ARTIFACT) if (root / FIT_ARTIFACT).is_file() else {}
    access = load_json(root / ACCESS_PATH) if (root / ACCESS_PATH).is_file() else {}
    checkpoints = load_json(root / CHECKPOINT_PATH) if (root / CHECKPOINT_PATH).is_file() else {}
    declared = (
        "REQ-VERIFY-7507" in (root / SPEC_PATH).read_text(encoding="utf-8")
        if (root / SPEC_PATH).is_file()
        else False
    )
    checks.extend(
        [
            precondition_row(
                "relevant_requirement_present",
                "OpenSpec",
                "REQ-VERIFY-7507",
                True,
                declared,
                path=SPEC_PATH.as_posix(),
            ),
            precondition_row(
                "evidence_terminal",
                "Exp7504",
                "terminal_status",
                "complete",
                evidence.get("terminal_status"),
                path=EVIDENCE_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "evidence_ready",
                "Exp7504",
                "evidence_ready_score",
                1,
                evidence.get("evidence_ready_score"),
                path=EVIDENCE_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "evidence_unflagged",
                "Exp7504",
                "flagged_adversarial",
                False,
                evidence.get("flagged_adversarial"),
                path=EVIDENCE_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "fit_terminal",
                "Exp7505",
                "terminal_status",
                "complete",
                fit.get("terminal_status"),
                path=FIT_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "fit_ready",
                "Exp7505",
                "energy_fit_ready_score",
                1,
                fit.get("energy_fit_ready_score"),
                path=FIT_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "baseline_ready",
                "Exp7505",
                "baseline_ready_score",
                1,
                fit.get("baseline_ready_score"),
                path=FIT_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "fit_unflagged",
                "Exp7505",
                "flagged_adversarial",
                False,
                fit.get("flagged_adversarial"),
                path=FIT_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "fit_heldout_closed",
                "Exp7505",
                "label_access_receipt.held_out_labels_opened",
                False,
                fit.get("label_access_receipt", {}).get("held_out_labels_opened"),
                path=FIT_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "frozen_before_heldout",
                "Exp7505",
                "checkpoint_manifest.frozen_before_heldout_label_access",
                True,
                fit.get("checkpoint_manifest", {}).get("frozen_before_heldout_label_access"),
                path=FIT_ARTIFACT.as_posix(),
            ),
            precondition_row(
                "equal_access",
                "Exp7504",
                "equal_access.identical_roles",
                True,
                access.get("equal_access", {}).get("identical_roles"),
                path=ACCESS_PATH.as_posix(),
            ),
            precondition_row(
                "evaluator_not_parsed_during_feature_build",
                "Exp7504",
                "evaluator_separation.evaluator_store_parsed_during_feature_build",
                False,
                access.get("evaluator_separation", {}).get(
                    "evaluator_store_parsed_during_feature_build"
                ),
                path=ACCESS_PATH.as_posix(),
            ),
        ]
    )
    sidecar_pairs = (
        (evidence.get("raw_sidecars", {}).get("features", {}), FEATURE_PATH),
        (evidence.get("raw_sidecars", {}).get("access_exposure_manifest", {}), ACCESS_PATH),
        (fit.get("raw_sidecars", {}).get("checkpoints", {}), CHECKPOINT_PATH),
    )
    for reference, path in sidecar_pairs:
        observed = sha256_file(root / path) if (root / path).is_file() else None
        checks.append(
            precondition_row(
                "upstream_sidecar_hash",
                "Exp7504_or_Exp7505",
                path.as_posix(),
                reference.get("sha256"),
                observed,
                path=path.as_posix(),
            )
        )
    return {
        "passed": all(row["passed"] for row in checks),
        "rows": checks,
        "source_artifact_hashes": source_hashes,
        "upstream": {
            "evidence_ready_score": evidence.get("evidence_ready_score"),
            "energy_fit_ready_score": fit.get("energy_fit_ready_score"),
            "frozen_before_heldout_label_access": fit.get("checkpoint_manifest", {}).get(
                "frozen_before_heldout_label_access"
            ),
            "fresh_confirmatory_claim_allowed": access.get("exposure_audit", {}).get(
                "fresh_confirmatory_claim_allowed"
            ),
            "claim_scope": access.get("exposure_audit", {}).get("claim_scope"),
            "bundle_sha256": checkpoints.get("bundle_sha256"),
            "policy_sha256": checkpoints.get("frozen_policies", {}).get("policy_sha256"),
            "transform_sha256": checkpoints.get("transform_sha256"),
        },
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, op: str, passed: bool
) -> JsonDict:
    """Attach the claim boundary to one independently observable operand."""

    principles = {
        "validity": "Favorable metrics cannot excuse invalid evidence.",
        "readiness": "A valid null must not prevent unrelated measurements.",
        "benefit": "Exploratory or favorable metrics cannot become a confirmatory claim.",
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


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate without hiding the remaining failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {"all_passed": True, "failed_checks": [], "first_failure": None}
    first = failed[0]
    return {
        "all_passed": False,
        "failed_checks": [str(row.get("check")) for row in failed],
        "first_failure": {
            "check": first.get("check"),
            "expected": first.get("expected"),
            "observed": first.get("observed"),
            "op": first.get("op"),
        },
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind terminal content while excluding only the checksum itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain what failure each raw field prevents."""

    specific = {
        "schema": "Version and identity prevent readers from applying another experiment contract.",
        "preconditions_checked": "Exact paths and observed values prevent invented upstream readiness.",
        "MODEL_SPECS": "An empty list prevents historical Qwen evidence from becoming a current load claim.",
        "model_specs": "The lowercase mirror prevents schema aliases from disagreeing.",
        "model_invoked": "A bare false separates current calls from historical provenance.",
        "invocation_counts": "Attempted and terminal counts expose unfinished current operations.",
        "rows": "Per-source and per-seed operands prevent pooled or selected-seed claims.",
        "policy_rows": "Per-source cost cells prevent escalation from being counted as accuracy.",
        "sample_size_budget": "Separate counts prevent seeds or windows from inflating support.",
        "static_evaluation_complete_score": "Readiness remains independent of measured benefit.",
        "static_probability_value_score": "Probability value requires every registered support and contrast gate.",
        "selective_decision_value_score": "Decision value requires every registered cost-cell gate.",
        "probability_contrasts": "Paired losses and adjusted bounds prevent pooled-number claims.",
        "evaluator_settings": "Frozen thresholds prevent post-result tuning.",
        "gate_check_summary": "Exact failures distinguish blocked, invalid, and null outcomes.",
        "honest_verdict": "A terminal prefix prevents unfinished work from looking final.",
        "verdict_class": "The closed class separates scientific nulls from operational blocks.",
        "flagged_adversarial": "Actual guard findings cannot be silently cleared.",
        "validation_receipts": "Exact commands and exits bind claims to required checks.",
    }
    return {
        key: specific.get(key, "This raw field preserves identity, scope, or an auditable operand.")
        for key in keys
    }


def _acceptance_gates(
    reduced: Mapping[str, Any], *, validation_passed: bool, preconditions_passed: bool
) -> list[JsonDict]:
    """Keep validity, readiness, and benefit gates independently visible."""

    probability = reduced["probability"]
    policy = reduced["policy"]
    return [
        _gate("preconditions", "validity", True, preconditions_passed, "eq", preconditions_passed),
        _gate("required_validation", "validity", True, validation_passed, "eq", validation_passed),
        _gate(
            "heldout_reduction_complete",
            "readiness",
            1,
            reduced["static_evaluation_complete_score"],
            "eq",
            reduced["static_evaluation_complete_score"] == 1,
        ),
        _gate(
            "primary_support",
            "benefit",
            True,
            probability["support"]["passed"],
            "eq",
            probability["support"]["passed"],
        ),
        _gate(
            "fresh_confirmatory_scope",
            "benefit",
            True,
            bool(reduced.get("confirmatory_allowed")),
            "eq",
            bool(reduced.get("confirmatory_allowed")),
        ),
        _gate(
            "probability_value",
            "benefit",
            1,
            reduced["static_probability_value_score"],
            "eq",
            reduced["static_probability_value_score"] == 1,
        ),
        _gate(
            "all_nine_selective_cells",
            "benefit",
            9,
            sum(row["benefit_passed"] for row in policy["cells"]),
            "eq",
            all(row["benefit_passed"] for row in policy["cells"]),
        ),
        _gate(
            "selective_decision_value",
            "benefit",
            1,
            reduced["selective_decision_value_score"],
            "eq",
            reduced["selective_decision_value_score"] == 1,
        ),
    ]


def classify_terminal(
    complete: int, probability_score: int, decision_score: int, confirmatory_allowed: bool
) -> tuple[str, str]:
    """Classify readiness independently from either registered benefit score."""

    if complete == 0:
        return "complete_disqualified_static_evaluation_required_validation_failed", "disqualified"
    if probability_score or decision_score:
        return "complete_positive_static_evaluation_registered_value_observed", "positive"
    if not confirmatory_allowed:
        return "complete_null_static_evaluation_exploratory_prior_exposure", "null"
    return "complete_null_static_evaluation_no_registered_benefit", "null"


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    settings: Mapping[str, Any],
    raw_sidecars: Mapping[str, Any],
    prediction_receipt: Mapping[str, Any],
    label_access_receipt: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    completed_at: str,
    duration_s: float,
    historical_model_provenance: Mapping[str, Any],
) -> JsonDict:
    """Assemble one terminal candidate from independently reducible operands."""

    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    complete = int(
        preconditions_passed
        and validation_passed
        and reduced["static_evaluation_complete_score"] == 1
    )
    probability_score = int(complete == 1 and reduced["static_probability_value_score"] == 1)
    decision_score = int(complete == 1 and reduced["selective_decision_value_score"] == 1)
    effective = deepcopy(dict(reduced))
    effective["static_evaluation_complete_score"] = complete
    effective["static_probability_value_score"] = probability_score
    effective["selective_decision_value_score"] = decision_score
    gates = _acceptance_gates(
        effective,
        validation_passed=validation_passed,
        preconditions_passed=preconditions_passed,
    )
    verdict, verdict_class = classify_terminal(
        complete,
        probability_score,
        decision_score,
        bool(settings.get("confirmatory_allowed")),
    )
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7507,
        "experiment_id": EXPERIMENT_ID,
        "title": "V657 frozen-head static held-out evaluation",
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "process_identity": {"pid": os.getpid(), "python": sys.executable},
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "inference_substrate_details": {
            "current_work": "CPU frozen-head forward scoring and source-paired reduction",
            "readout_kind": "frozen_small_head_probability",
            "generated_tokens": 0,
        },
        "execution_venue": "host",
        "device_identity": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "current_total": float(duration_s),
            "historical_model_capture": 0.0,
            "authoring_not_in_runtime": True,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "fit_seeds": list(FIT_SEEDS),
            "bootstrap_seed": int(settings["bootstrap_seed"]),
            "bootstrap_draws": int(settings["bootstrap_draws"]),
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "rows": [deepcopy(dict(row)) for row in rows],
        "policy_rows": deepcopy(reduced["policy_rows"]),
        "sample_size_budget": deepcopy(reduced["sample_size_budget"]),
        "preserved_upstream_exclusions": {
            "excluded_groups": 9,
            "censored_groups": 0,
            "source": EVIDENCE_ARTIFACT.as_posix(),
            "scope": "all_520_protocol_groups_before_test_role_selection",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "static_evaluation_complete_score": complete,
        "static_probability_value_score": probability_score,
        "selective_decision_value_score": decision_score,
        "probability_contrasts": {
            "brier": deepcopy(reduced["probability"]["brier_contrasts"]),
            "log_loss": deepcopy(reduced["probability"]["log_loss_contrast"]),
        },
        "probability_metrics": deepcopy(reduced["probability"]["metrics"]),
        "policy_evaluation": deepcopy(reduced["policy"]),
        "evaluator_settings": deepcopy(dict(settings)),
        "raw_sidecars": deepcopy(dict(raw_sidecars)),
        "prediction_freeze": deepcopy(dict(prediction_receipt)),
        "label_access_receipt": deepcopy(dict(label_access_receipt)),
        "historical_model_provenance": deepcopy(dict(historical_model_provenance)),
        "small_ebm_training": {
            "performed_current_run": False,
            "frozen_fit_source": FIT_ARTIFACT.as_posix(),
            "current_cpu_work": "forward_scoring_only",
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
        "capability_e2e": {
            "kind": "declared_entrypoint_plus_fresh_process_cold_replay",
            "numbered_runtime_e2e": "not_applicable_reporting_only",
        },
    }
    value["field_principles"] = _field_principles([*value, "field_principles"])
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute the three scores and registered reports from per-unit rows."""

    if value.get("verdict_class") == "blocked":
        return {
            "static_evaluation_complete_score": 0,
            "static_probability_value_score": 0,
            "selective_decision_value_score": 0,
        }
    rows = value.get("rows")
    settings = value.get("evaluator_settings")
    if not isinstance(rows, list) or not isinstance(settings, Mapping):
        raise ValueError("reduction_operands_missing")
    reduced = reduce_rows(rows, settings=settings)
    validation_passed = any(
        row.get("check") == "required_validation" and row.get("passed") is True
        for row in value.get("acceptance_gate_results", [])
    )
    preconditions_passed = all(
        row.get("passed") is True for row in value.get("preconditions_checked", [])
    )
    complete = int(
        validation_passed
        and preconditions_passed
        and reduced["static_evaluation_complete_score"] == 1
    )
    return {
        "static_evaluation_complete_score": complete,
        "static_probability_value_score": int(
            complete == 1 and reduced["static_probability_value_score"] == 1
        ),
        "selective_decision_value_score": int(
            complete == 1 and reduced["selective_decision_value_score"] == 1
        ),
        "sample_size_budget": reduced["sample_size_budget"],
        "probability_contrasts": {
            "brier": reduced["probability"]["brier_contrasts"],
            "log_loss": reduced["probability"]["log_loss_contrast"],
        },
        "policy_evaluation": reduced["policy"],
        "policy_rows_sha256": canonical_hash(reduced["policy_rows"]),
    }


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    verify_sources: bool = True,
) -> list[str]:
    """Cold-check identity, sidecars, raw reduction, and provenance counters."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in value]
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
        return errors
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("run_date") != RUN_DATE or value.get("milestone") != MILESTONE:
        errors.append("artifact_date_invalid")
    if value.get("terminal_status") != "complete":
        errors.append("terminal_status_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_nonempty")
    if value.get("model_invoked") is not False:
        errors.append("model_invoked_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_invalid")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_invalid")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_invalid")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if any(value.get(field) not in (0, 1) for field in REQUIRED_FIELDS if field.endswith("_score")):
        errors.append("score_not_bare_binary")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if verify_sources:
        for row in value.get("source_artifact_hashes", []):
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file():
                errors.append(f"source_missing:{path}")
            elif sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{path}")
    for reference in value.get("raw_sidecars", {}).values():
        if isinstance(reference, Mapping):
            errors.extend(verify_sidecar(reference, root))
    if value.get("verdict_class") == "blocked":
        expected = independent_reduce(value)
        if any(value.get(key) != observed for key, observed in expected.items()):
            errors.append("blocked_score_mismatch")
        if not str(value.get("honest_verdict")).startswith("complete_blocked_"):
            errors.append("blocked_verdict_invalid")
        return list(dict.fromkeys(errors))
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("evaluation_rows_missing")
        return list(dict.fromkeys(errors))
    row_reference = value.get("raw_sidecars", {}).get("evaluation_rows", {})
    if isinstance(row_reference, Mapping):
        path = Path(str(row_reference.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file() and canonical_hash(load_jsonl(resolved)) != canonical_hash(rows):
            errors.append("evaluation_sidecar_content_mismatch")
    prediction_reference = value.get("prediction_freeze")
    if isinstance(prediction_reference, Mapping):
        path = Path(str(prediction_reference.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file() and any("label" in row for row in load_jsonl(resolved)):
            errors.append("prediction_sidecar_contains_label")
    try:
        reduced = independent_reduce(value)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"independent_reduction_failed:{type(error).__name__}:{error}")
    else:
        for field in (
            "static_evaluation_complete_score",
            "static_probability_value_score",
            "selective_decision_value_score",
            "sample_size_budget",
            "probability_contrasts",
            "policy_evaluation",
        ):
            if value.get(field) != reduced[field]:
                errors.append(f"independent_reduction_mismatch:{field}")
        if canonical_hash(value.get("policy_rows")) != reduced["policy_rows_sha256"]:
            errors.append("independent_reduction_mismatch:policy_rows")
    return list(dict.fromkeys(errors))


def blocked_artifact(failed: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Publish external absence as blocked without fabricating measurements."""

    reason = "".join(
        character if character.isalnum() else "_" for character in str(failed["check"])
    )
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "preconditions_checked": [deepcopy(dict(failed))],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "fit_seeds": list(FIT_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": [],
        "rows": [],
        "policy_rows": [],
        "sample_size_budget": {
            "planned": 116,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 116,
            "independent_unit": "unique_source_group",
        },
        "acceptance_gate_results": [
            _gate(
                str(failed["check"]),
                "validity",
                failed.get("expected"),
                failed.get("observed"),
                str(failed.get("op") or "eq"),
                False,
            )
        ],
        "gate_check_summary": {
            "check": failed.get("check"),
            "upstream": failed.get("upstream"),
            "path": failed.get("path"),
            "field": failed.get("artifact_field"),
            "expected": failed.get("expected"),
            "observed": failed.get("observed"),
        },
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "static_evaluation_complete_score": 0,
        "static_probability_value_score": 0,
        "selective_decision_value_score": 0,
        "probability_contrasts": {"brier": {}, "log_loss": {}},
        "evaluator_settings": evaluator_settings(expected_groups=116),
        "raw_sidecars": {},
    }
    value["field_principles"] = _field_principles([*value, "field_principles"])
    value["reproducibility_checksum"] = artifact_checksum(value)
    if root != REPO_ROOT:
        value["gate_check_summary"]["checked_root"] = str(root.resolve())
        value["field_principles"] = _field_principles([*value, "field_principles"])
        value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def cold_replay(
    path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:  # pragma: no cover - exercised in a fresh process.
    """Load serialized bytes and run the independent artifact validator."""

    return validate_artifact(load_json(path), root=root, verify_sources=verify_sources)


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one aware timestamp for an actual runtime boundary."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - real execution boundary.
    """Flush phase and operation boundaries so the conductor sees liveness."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[progress] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, detail: str
) -> JsonDict:  # pragma: no cover - real execution boundary.
    """Record a measured monotonic phase interval and completed units."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - run_started,
        "ended_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "detail": detail,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the required fresh-process readers for one exact candidate."""

    python = ".venv/bin/python"
    commands = (
        validation_scope.CommandSpec(
            "cold_artifact_replay",
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


def _json_sidecar(  # pragma: no cover - production JSON sidecar boundary.
    path: Path, value: Mapping[str, Any], *, root: Path
) -> JsonDict:
    """Write a bounded JSON object and return its byte identity."""

    atomic_json(path, value)
    if path.stat().st_size >= 20 * 1024 * 1024:
        raise ValueError(f"sidecar_size_limit:{path}")
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _publish_blocked(
    root: Path, failed: Mapping[str, Any], started: float
) -> JsonDict:  # pragma: no cover - external absence boundary.
    """Atomically publish one exact external block and stop dependent work."""

    artifact = blocked_artifact(failed, root=root)
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(started, "publish", "before_atomic_terminal", verdict="blocked")
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "publish", f"complete_blocked_{failed.get('check')}")
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Measure, validate, cold-replay, and atomically publish the evaluator."""

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
            "authenticated_required_inputs",
        )
    )
    failed = next((row for row in preconditions["rows"] if row.get("passed") is not True), None)
    if failed is not None:
        return _publish_blocked(root, failed, run_started)
    progress(run_started, "preconditions", "complete", checks=len(preconditions["rows"]))

    progress(run_started, "prediction", "before_benchmark", expected_groups=116)
    phase_started = time.monotonic()
    checkpoints = load_json(root / CHECKPOINT_PATH)
    predict_opened = read_mode(root / FEATURE_PATH, root / EVALUATOR_PATH, mode="predict")
    training_rows = [row for row in predict_opened["rows"] if row.get("role") == "training"]
    test_rows = [row for row in predict_opened["rows"] if row.get("role") == "test"]
    transform = authenticate_transform(training_rows, checkpoints)
    transform_check = precondition_row(
        "frozen_transform_reconstructed",
        "Exp7505",
        "transform_sha256",
        checkpoints.get("transform_sha256"),
        canonical_hash(transform),
        path=CHECKPOINT_PATH.as_posix(),
    )
    preconditions["rows"].append(transform_check)
    if not transform_check["passed"]:
        return _publish_blocked(root, transform_check, run_started)
    test_count_check = precondition_row(
        "heldout_test_group_count",
        "Exp7504",
        "test_groups",
        116,
        len(test_rows),
        path=FEATURE_PATH.as_posix(),
    )
    preconditions["rows"].append(test_count_check)
    if not test_count_check["passed"]:
        return _publish_blocked(root, test_count_check, run_started)
    predictions = score_label_free_predictions(test_rows, checkpoints, transform)
    prediction_receipt = write_jsonl_sidecar(root / PREDICTION_PATH, predictions, root=root)
    prediction_groups = len({row["group_id"] for row in predictions})
    prediction_check = precondition_row(
        "label_free_prediction_groups",
        "Exp7507",
        "prediction_groups_before_label_access",
        116,
        prediction_groups,
        path=PREDICTION_PATH.as_posix(),
    )
    preconditions["rows"].append(prediction_check)
    if not prediction_check["passed"]:
        return _publish_blocked(root, prediction_check, run_started)
    spans.append(
        _span(
            "prediction",
            phase_started,
            run_started,
            len(predictions),
            PREDICTION_PATH.as_posix(),
        )
    )
    progress(
        run_started,
        "prediction",
        "after_benchmark",
        groups=prediction_groups,
        rows=len(predictions),
        sha256=prediction_receipt["sha256"],
    )

    progress(run_started, "label_access", "before_benchmark", prediction_frozen=True)
    phase_started = time.monotonic()
    evaluated = read_mode(
        root / FEATURE_PATH,
        root / EVALUATOR_PATH,
        mode="evaluate",
        prediction_path=root / PREDICTION_PATH,
        prediction_sha256=str(prediction_receipt["sha256"]),
    )
    rows = attach_evaluation_labels(predictions, evaluated["rows"])
    label_receipt = deepcopy(evaluated["access_receipt"])
    label_receipt["prediction_written_before_label_access"] = True
    label_receipt["prediction_row_count"] = len(predictions)
    label_receipt["evaluation_group_count"] = len({row["group_id"] for row in rows})
    row_receipt = write_jsonl_sidecar(root / ROW_PATH, rows, root=root)
    spans.append(_span("label_access", phase_started, run_started, 116, "authorized_test_labels"))
    progress(
        run_started,
        "label_access",
        "after_benchmark",
        groups=label_receipt["evaluation_group_count"],
    )

    progress(run_started, "reduction", "before_benchmark", draws=BOOTSTRAP_DRAWS)
    phase_started = time.monotonic()
    access = load_json(root / ACCESS_PATH)
    settings = evaluator_settings(
        draws=BOOTSTRAP_DRAWS,
        confirmatory_allowed=bool(
            access.get("exposure_audit", {}).get("fresh_confirmatory_claim_allowed")
        ),
        simple_baseline_arm=str(checkpoints["simple_baseline"]["arm"]),
        expected_groups=116,
    )
    reduced = reduce_rows(rows, settings=settings)
    policy_receipt = write_jsonl_sidecar(root / POLICY_PATH, reduced["policy_rows"], root=root)
    indices = bootstrap_indices(116, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED).tolist()
    bootstrap_receipt = _json_sidecar(
        root / BOOTSTRAP_PATH,
        {
            "schema": "carnot.exp7507.v657.bootstrap_indices.v1",
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "group_count": 116,
            "indices": indices,
        },
        root=root,
    )
    spans.append(
        _span(
            "reduction",
            phase_started,
            run_started,
            BOOTSTRAP_DRAWS,
            "source_paired_bootstrap_draws",
        )
    )
    progress(
        run_started,
        "reduction",
        "after_benchmark",
        probability_score=reduced["static_probability_value_score"],
        decision_score=reduced["selective_decision_value_score"],
    )

    manifest_path = root / RAW_DIR / "affected_validation_manifest.json"
    manifest_receipt = _json_sidecar(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        root=root,
    )
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7507-validation-", dir="/tmp"))
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

    source_hashes = list(preconditions["source_artifact_hashes"])
    source_hashes.extend(_source_row(path, root) for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH))
    evidence = load_json(root / EVIDENCE_ARTIFACT)
    raw_sidecars = {
        "label_free_predictions": prediction_receipt,
        "evaluation_rows": row_receipt,
        "policy_rows": policy_receipt,
        "bootstrap_indices": bootstrap_receipt,
        "affected_validation_manifest": manifest_receipt,
    }
    candidate = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=source_hashes,
        rows=rows,
        reduced=reduced,
        settings=settings,
        raw_sidecars=raw_sidecars,
        prediction_receipt=prediction_receipt,
        label_access_receipt=label_receipt,
        validation_receipts=affected,
        validation_passed=bool(affected_reduction["passed"]),
        phase_spans=spans,
        started_at=started_at,
        completed_at=utc_now(),
        duration_s=time.monotonic() - run_started,
        historical_model_provenance=evidence.get("historical_model_provenance", {}),
    )
    candidate_errors = validate_artifact(candidate, root=root)
    if candidate_errors:
        raise RuntimeError(f"measured_candidate_invalid:{candidate_errors}")
    atomic_json(root / TERMINAL_CANDIDATE, candidate)

    progress(run_started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        _terminal_commands(root / TERMINAL_CANDIDATE),
        log_dir=root / RAW_DIR / "validation" / "terminal",
    )
    terminal_passed = _all_receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    critical = any(
        row.get("name") == "adversarial_verify"
        and "[CRITICAL]" in str(row.get("output_tail") or "")
        for row in terminal
    )
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
    candidate_receipt = {
        "path": TERMINAL_CANDIDATE.as_posix(),
        "sha256": sha256_file(root / TERMINAL_CANDIDATE),
        "bytes": (root / TERMINAL_CANDIDATE).stat().st_size,
    }
    all_validation_passed = bool(affected_reduction["passed"] and terminal_passed and not critical)
    final = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=source_hashes,
        rows=rows,
        reduced=reduced,
        settings=settings,
        raw_sidecars={**raw_sidecars, "terminal_candidate": candidate_receipt},
        prediction_receipt=prediction_receipt,
        label_access_receipt=label_receipt,
        validation_receipts=[*affected, *terminal],
        validation_passed=all_validation_passed,
        phase_spans=spans,
        started_at=started_at,
        completed_at=utc_now(),
        duration_s=time.monotonic() - run_started,
        historical_model_provenance=evidence.get("historical_model_provenance", {}),
    )
    final["flagged_adversarial"] = critical
    final["reproducibility_checksum"] = artifact_checksum(final)
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    encoded_size = len((json.dumps(final, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if encoded_size >= 20 * 1024 * 1024:
        raise RuntimeError(f"terminal_artifact_size_limit:{encoded_size}")
    progress(run_started, "publish", "before_atomic_terminal", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, final)
    progress(
        run_started,
        "publish",
        "complete",
        completion=final["static_evaluation_complete_score"],
        probability_value=final["static_probability_value_score"],
        selective_value=final["selective_decision_value_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date plus read-only cold-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run measurement or one fresh-process reader through the same entrypoint."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = load_json(args.independent_reduce)
        reduced = independent_reduce(value)
        expected = {
            key: value[key]
            for key in (
                "static_evaluation_complete_score",
                "static_probability_value_score",
                "selective_decision_value_score",
            )
        }
        passed = all(reduced[key] == observed for key, observed in expected.items())
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(REPO_ROOT, args.date)
    return 0
