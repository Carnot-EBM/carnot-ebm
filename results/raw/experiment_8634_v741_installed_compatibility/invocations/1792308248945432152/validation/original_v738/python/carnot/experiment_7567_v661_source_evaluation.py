"""Evaluate frozen source-energy heads on sealed injected tool errors.

This module performs CPU-only aggregation. It does not load a language model,
change generator weights, or claim general hallucination detection.

Spec refs: REQ-VERIFY-7567 and SCENARIO-VERIFY-7567-*.
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
from carnot.experiment_7566_v661_energy_fit import local_design_matrix, stable_sigmoid
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7567-v661-source-evaluation"
SCHEMA = "carnot.exp7567.v661.source_evaluation.v1"
RESULT_PATH = Path("results/experiment_7567_v661_source_evaluation.json")
RAW_DIR = Path("results/raw/experiment_7567_v661_source_evaluation")
MODULE_PATH = Path("python/carnot/experiment_7567_v661_source_evaluation.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7567_v661_source_evaluation.py")
TEST_PATH = Path("tests/python/test_experiment_7567_v661_source_evaluation.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
CAPTURE_PATH = Path("results/experiment_7565_v661_test_online_capture.json")
FIT_PATH = Path("results/experiment_7566_v661_energy_fit.json")
LABEL_FREE_PATH = RAW_DIR / "label_free_predictions.jsonl"
LABELED_ROWS_PATH = RAW_DIR / "source_group_rows.jsonl"
STATIC_REPORT_PATH = RAW_DIR / "static_evaluation_report.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
VALIDATION_MANIFEST_PATH = RAW_DIR / "affected_validation_manifest.json"

BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 7_567_001
ORDER_NAMES = ("supported_first", "unsupported_first")
CANDIDATE_ARM = "source_contrast_energy"
ARMS = (
    CANDIDATE_ARM,
    "temperature_original",
    "original_only_local_basis_energy",
    "unconstrained_equal_capacity_energy",
    "same_information_logistic",
    "raw_original",
)
BRIER_COMPARATORS = (
    "temperature_original",
    "original_only_local_basis_energy",
    "unconstrained_equal_capacity_energy",
)
TRAINED_ARMS = (
    "source_contrast_energy",
    "unconstrained_equal_capacity_energy",
    "original_only_local_basis_energy",
    "same_information_logistic",
)
COST_GRID = tuple(
    (false_accept, escalation) for false_accept in (1.0, 2.0, 5.0) for escalation in (0.1, 0.2, 0.5)
)
PRIMARY_FALSE_ACCEPT_COST = 5.0
PRIMARY_ESCALATION_COST = 0.2
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


def load_json(path: Path) -> JsonDict:
    """Read one JSON object without repairing malformed evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read complete JSONL objects and reject any other row type."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _path_label(path: Path, root: Path) -> str:
    """Use a stable worktree-relative label when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def source_hash_row(path: Path, root: Path) -> JsonDict:
    """Bind the exact bytes of one resource used by the evaluation."""

    resolved = path if path.is_absolute() else root / path
    return {
        "path": _path_label(resolved, root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def write_jsonl_sidecar(path: Path, rows: Sequence[Mapping[str, Any]], *, root: Path) -> JsonDict:
    """Write deterministic JSONL through an atomic same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if path.stat().st_size >= 20 * 1024 * 1024:  # pragma: no cover - fixed-size safety guard.
        raise ValueError(f"sidecar_size_limit:{path}")
    return {
        **source_hash_row(path, root),
        "rows": len(rows),
    }


def verify_sidecar(reference: Mapping[str, Any], root: Path) -> list[str]:
    """Verify the path, byte hash, and optional JSONL row count."""

    label = str(reference.get("path") or "")
    path = Path(label)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        return [f"sidecar_missing:{label}"]
    if sha256_file(resolved) != reference.get("sha256"):
        return [f"sidecar_sha256_mismatch:{label}"]
    if resolved.suffix == ".jsonl" and "rows" in reference:
        if len(load_jsonl(resolved)) != reference.get("rows"):
            return [f"sidecar_row_count_mismatch:{label}"]
    return []


def fixture_frozen_heads() -> JsonDict:
    """Build small frozen heads for structural tests, not empirical claims."""

    knots = [[-3.0] * 4 + [-1.5, -0.5, 0.5, 1.5] + [3.0] * 4 for _ in range(3)]
    widths = {
        "source_contrast_energy": 24,
        "unconstrained_equal_capacity_energy": 24,
        "original_only_local_basis_energy": 8,
        "same_information_logistic": 3,
    }
    heads: dict[str, JsonDict] = {}
    for arm, width in widths.items():
        checkpoint = {
            "bias": 0.05,
            "weights": np.linspace(-0.35, 0.45, width, dtype=np.float64).tolist(),
        }
        heads[arm] = {
            "family": arm,
            "checkpoint": checkpoint,
            "checkpoint_sha256": canonical_hash(checkpoint),
        }
    value: JsonDict = {
        "schema": "carnot.exp7567.fixture_heads.v1",
        "bundle_sha256": "sha256:fixture-bundle",
        "normalization": {"mean": [0.0, 0.0, 0.0], "safe_scale": [1.0, 1.0, 1.0]},
        "local_basis": {"knots": knots, "coefficients_per_feature": 8},
        "selected_heads": heads,
        "strongest_comparator": {
            "family": "temperature_original",
            "frozen_before_policy_access": True,
        },
        "temperature_baseline": {"selected_temperature": 0.5},
    }
    return value


def _temperature_probability(probability: float, temperature: float) -> float:
    """Apply a frozen scalar temperature to one finite probability."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    logit = math.log(clipped / (1.0 - clipped))
    return float(stable_sigmoid(np.asarray([logit / temperature]))[0])


def _trained_probability(arm: str, features: np.ndarray, heads: Mapping[str, Any]) -> float:
    """Apply one immutable trained head to one normalized feature view."""

    normalization = heads.get("normalization")
    selected = heads.get("selected_heads")
    basis = heads.get("local_basis")
    if not all(isinstance(value, Mapping) for value in (normalization, selected, basis)):
        raise ValueError("frozen_head_manifest_invalid")
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["safe_scale"], dtype=np.float64)
    if mean.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0.0):
        raise ValueError("normalization_invalid")
    normalized = (features - mean) / scale
    if arm == "source_contrast_energy":
        design = local_design_matrix(normalized[None, :], basis["knots"])[0]
    elif arm == "unconstrained_equal_capacity_energy":
        clipped = np.clip(normalized, -4.0, 4.0)
        design = np.concatenate([clipped**power for power in range(1, 9)])
    elif arm == "original_only_local_basis_energy":
        design = local_design_matrix(normalized[None, :], basis["knots"])[0, :8]
    elif arm == "same_information_logistic":
        design = normalized
    else:
        raise ValueError(f"trained_arm_invalid:{arm}")
    head = selected.get(arm)
    if not isinstance(head, Mapping) or not isinstance(head.get("checkpoint"), Mapping):
        raise ValueError(f"frozen_head_missing:{arm}")
    checkpoint = head["checkpoint"]
    if head.get("checkpoint_sha256") != canonical_hash(checkpoint):
        raise ValueError(f"checkpoint_sha256_mismatch:{arm}")
    weights = np.asarray(checkpoint.get("weights"), dtype=np.float64)
    bias = float(checkpoint.get("bias"))
    if weights.shape != design.shape or not np.all(np.isfinite(weights)) or not math.isfinite(bias):
        raise ValueError(f"checkpoint_shape_invalid:{arm}")
    return float(stable_sigmoid(np.asarray([float(design @ weights) + bias]))[0])


def score_label_free_predictions(
    rows: Sequence[Mapping[str, Any]], heads: Mapping[str, Any]
) -> list[JsonDict]:
    """Score all frozen arms without accepting evaluator outcomes."""

    if any("label" in row or "y" in row for row in rows):
        raise ValueError("label_visible_during_prediction")
    output: list[JsonDict] = []
    seen_components: set[str] = set()
    temperature = float(heads.get("temperature_baseline", {}).get("selected_temperature"))
    strongest = heads.get("strongest_comparator")
    if not isinstance(strongest, Mapping) or strongest.get("family") != "temperature_original":
        raise ValueError("strongest_comparator_invalid")
    for row in rows:
        component = str(row.get("component_hash") or "")
        group = str(row.get("group_hash") or "")
        if not component or not group or row.get("role") != "test":
            raise ValueError("prediction_identity_invalid")
        if component in seen_components:
            raise ValueError("prediction_component_duplicate")
        seen_components.add(component)
        raw_views = row.get("feature_views")
        if not isinstance(raw_views, Mapping) or set(raw_views) != set(ORDER_NAMES):
            raise ValueError("feature_views_invalid")
        features: dict[str, np.ndarray] = {}
        intervention: dict[str, JsonDict] = {}
        for order in ORDER_NAMES:
            vector = np.asarray(raw_views[order], dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("feature_vector_invalid")
            features[order] = vector
            original = float(vector[0])
            absent = original - float(vector[1])
            mismatched = original - float(vector[2])
            intervention[order] = {
                "original_logit": original,
                "absent_logit": absent,
                "mismatched_logit": mismatched,
                "original_confidence": float(stable_sigmoid(np.asarray([original]))[0]),
                "missing_source_confidence": float(stable_sigmoid(np.asarray([absent]))[0]),
                "mismatched_source_confidence": float(stable_sigmoid(np.asarray([mismatched]))[0]),
                "original_minus_absent": float(vector[1]),
                "original_minus_mismatched": float(vector[2]),
            }
        for arm in ARMS:
            option_probabilities: dict[str, float] = {}
            neutral_probabilities: list[float] = []
            for order in ORDER_NAMES:
                vector = features[order]
                raw_probability = float(stable_sigmoid(np.asarray([vector[0]]))[0])
                if arm in TRAINED_ARMS:
                    probability = _trained_probability(arm, vector, heads)
                    neutral = vector.copy()
                    neutral[1:] = 0.0
                    neutral_probabilities.append(_trained_probability(arm, neutral, heads))
                elif arm == "raw_original":
                    probability = raw_probability
                    neutral_probabilities.append(probability)
                elif arm == "temperature_original":
                    probability = _temperature_probability(raw_probability, temperature)
                    neutral_probabilities.append(probability)
                else:  # pragma: no cover - ARMS is a closed constant.
                    raise ValueError(f"arm_invalid:{arm}")
                option_probabilities[order] = probability
            final_probability = float(np.mean(list(option_probabilities.values())))
            neutral_probability = float(np.mean(neutral_probabilities))
            source_dependence_applicable = arm not in {
                "raw_original",
                "temperature_original",
                "original_only_local_basis_energy",
            }
            head = heads.get("selected_heads", {}).get(arm, {})
            head_identity = (
                head.get("checkpoint_sha256")
                if isinstance(head, Mapping)
                else canonical_hash(
                    {
                        "arm": arm,
                        "temperature": temperature if arm == "temperature_original" else None,
                    }
                )
            )
            if arm in {"raw_original", "temperature_original"}:
                head_identity = canonical_hash(
                    {
                        "arm": arm,
                        "temperature": temperature if arm == "temperature_original" else None,
                    }
                )
            output.append(
                {
                    "unit_id": component,
                    "source_component_hash": component,
                    "group_id": group,
                    "role": "test",
                    "tool_type": str(row.get("tool_type") or "unknown"),
                    "arm": arm,
                    "head_identity": head_identity,
                    "option_probabilities": option_probabilities,
                    "probability": final_probability,
                    "source_intervention_diagnostics": {
                        "by_order": intervention,
                        "source_dependence_applicable": source_dependence_applicable,
                        "source_neutral_probability": (
                            neutral_probability if source_dependence_applicable else None
                        ),
                        "source_neutral_absolute_delta": (
                            abs(final_probability - neutral_probability)
                            if source_dependence_applicable
                            else None
                        ),
                        "source_neutral_disposition": (
                            "measured"
                            if source_dependence_applicable
                            else "not_applicable_source_independent_arm"
                        ),
                        "option_order_absolute_gap": abs(
                            option_probabilities[ORDER_NAMES[0]]
                            - option_probabilities[ORDER_NAMES[1]]
                        ),
                        "missing_source_confidence_is_oracle": False,
                    },
                    "status": "prediction_frozen",
                }
            )
    return sorted(output, key=lambda item: (item["unit_id"], item["arm"]))


def loss_fields(probability: float, label: int) -> JsonDict:
    """Compute proper binary losses while clipping only log-loss arithmetic."""

    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0 or label not in (0, 1):
        raise ValueError("probability_or_label_invalid")
    clipped = min(max(numeric, 1e-9), 1.0 - 1e-9)
    return {
        "brier": (numeric - label) ** 2,
        "log_loss": -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped)),
    }


def _decision_cell(
    probability: float, label: int, *, false_accept_cost: float, escalation_cost: float
) -> JsonDict:
    """Choose and score one typed action with escalation winning exact ties."""

    loss_fields(probability, label)
    expected = {
        "accept": false_accept_cost * probability,
        "reject": 1.0 - probability,
        "escalate": escalation_cost,
    }
    tie_order = ("escalate", "accept", "reject")
    action = min(tie_order, key=lambda name: (expected[name], tie_order.index(name)))
    realized = {
        "accept": false_accept_cost if label == 1 else 0.0,
        "reject": 1.0 if label == 0 else 0.0,
        "escalate": escalation_cost,
    }[action]
    return {
        "cell_id": f"fa={false_accept_cost:g}|fr=1|esc={escalation_cost:g}",
        "false_accept_cost": false_accept_cost,
        "false_reject_cost": 1.0,
        "escalation_cost": escalation_cost,
        "action": action,
        "escalated": action == "escalate",
        "realized_cost": float(realized),
    }


def primary_decision(probability: float, label: int) -> JsonDict:
    """Return the registered primary action and its realized cost."""

    cell = _decision_cell(
        probability,
        label,
        false_accept_cost=PRIMARY_FALSE_ACCEPT_COST,
        escalation_cost=PRIMARY_ESCALATION_COST,
    )
    return {"action": cell["action"], "realized_cost": cell["realized_cost"]}


def attach_test_labels(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    prediction_sha256: str,
    expected_upstream_freeze: str,
) -> list[JsonDict]:
    """Join sealed test outcomes only after all prediction bytes freeze."""

    label_by_component: dict[str, Mapping[str, Any]] = {}
    for row in labels:
        component = str(row.get("component_hash") or "")
        if component in label_by_component:
            raise ValueError(f"evaluation_label_duplicate:{component}")
        if (
            not component
            or row.get("role") != "test"
            or row.get("label") not in (0, 1)
            or row.get("prediction_freeze_sha256") != expected_upstream_freeze
        ):
            raise ValueError(f"evaluation_label_invalid:{component}")
        label_by_component[component] = row
    predicted_components = {str(row.get("source_component_hash") or "") for row in predictions}
    if predicted_components != set(label_by_component):
        raise ValueError("evaluation_group_mismatch")
    output: list[JsonDict] = []
    for prediction in predictions:
        component = str(prediction["source_component_hash"])
        label = int(label_by_component[component]["label"])
        probability = float(prediction["probability"])
        row = deepcopy(dict(prediction))
        row.update(loss_fields(probability, label))
        decision = primary_decision(probability, label)
        row.update(
            {
                "label": label,
                "y": label,
                "action": decision["action"],
                "realized_cost": decision["realized_cost"],
                "descriptive_cost_grid": [
                    _decision_cell(
                        probability,
                        label,
                        false_accept_cost=false_accept,
                        escalation_cost=escalation,
                    )
                    for false_accept, escalation in COST_GRID
                ],
                "prediction_freeze_sha256": prediction_sha256,
                "status": "complete",
                "attempted": True,
                "complete": True,
                "failed": False,
                "excluded": False,
                "censored": False,
                "unstarted": False,
            }
        )
        output.append(row)
    return sorted(output, key=lambda item: (item["unit_id"], item["arm"]))


def run_challenge_controls(predictions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report source-neutral shortcuts and mapped-order sensitivity."""

    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        option_probabilities = row.get("option_probabilities")
        if not isinstance(option_probabilities, Mapping) or set(option_probabilities) != set(
            ORDER_NAMES
        ):
            raise ValueError("option_prediction_roster_invalid")
        if any(not math.isfinite(float(option_probabilities[name])) for name in ORDER_NAMES):
            raise ValueError("option_prediction_nonfinite")
        by_arm[str(row.get("arm"))].append(row)
    if set(by_arm) != set(ARMS):
        raise ValueError("arm_roster_invalid")
    sensitivity = {
        arm: {
            "mean_absolute_gap": float(
                np.mean(
                    [
                        float(row["source_intervention_diagnostics"]["option_order_absolute_gap"])
                        for row in arm_rows
                    ]
                )
            ),
            "max_absolute_gap": float(
                max(
                    float(row["source_intervention_diagnostics"]["option_order_absolute_gap"])
                    for row in arm_rows
                )
            ),
            "averaging_rule": "equal_weight_across_two_mapped_orders",
        }
        for arm, arm_rows in by_arm.items()
    }
    candidate_deltas = [
        float(row["source_intervention_diagnostics"]["source_neutral_absolute_delta"])
        for row in by_arm[CANDIDATE_ARM]
    ]
    shortcut = {
        "candidate_mean_absolute_delta": float(np.mean(candidate_deltas)),
        "candidate_changed_group_count": sum(delta > 1e-12 for delta in candidate_deltas),
        "source_neutral_features": "source_contrasts_set_to_zero_before_frozen_transform",
        "empirical_benefit_gate": False,
    }
    passed = (
        len({str(row["unit_id"]) for row in predictions}) > 0
        and all(len(rows) == len(by_arm[CANDIDATE_ARM]) for rows in by_arm.values())
        and shortcut["candidate_changed_group_count"] > 0
    )
    return {
        "passed": passed,
        "source_independent_shortcut": shortcut,
        "option_permutation_sensitivity": sensitivity,
        "missing_source_confidence_is_oracle": False,
        "missing_source_confidence_scope": "diagnostic_only",
    }


def evaluator_settings(
    *,
    draws: int = BOOTSTRAP_DRAWS,
    confirmatory_allowed: bool = False,
    expected_groups: int = 80,
) -> JsonDict:
    """Return registered thresholds without consulting evaluation outcomes."""

    return {
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "expected_groups": expected_groups,
        "minimum_groups": 64,
        "minimum_per_label": 12,
        "brier_improvement": 0.01,
        "maximum_log_loss_deterioration": 0.01,
        "maximum_primary_cost_deterioration": 0.01,
        "minimum_non_escalation_fraction": 0.20,
        "holm_family_alpha": 0.05,
        "probability_candidate": CANDIDATE_ARM,
        "brier_comparators": list(BRIER_COMPARATORS),
        "selected_comparator": "temperature_original",
        "primary_cost_cell": "fa=5|fr=1|esc=0.2",
        "confirmatory_allowed": confirmatory_allowed,
        "inference_unit": "unique_source_component",
        "mapped_orders_are_independent_units": False,
        "arms_are_independent_units": False,
        "no_post_test_refit": True,
        "no_threshold_change": True,
        "no_subgroup_selection": True,
    }


def bootstrap_indices(group_count: int, *, draws: int, seed: int) -> np.ndarray:
    """Freeze shared source-component draws for every paired contrast."""

    if group_count < 1 or draws < 1:
        raise ValueError("bootstrap_shape_invalid")
    return np.random.default_rng(seed).integers(0, group_count, size=(draws, group_count))


def paired_interval(deltas: Sequence[float], indices: np.ndarray) -> JsonDict:
    """Reduce paired differences and retain one-sided inference operands."""

    values = np.asarray(deltas, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or indices.ndim != 2
        or indices.shape[1] != len(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("paired_bootstrap_shape_invalid")
    means = values[indices].mean(axis=1)
    return {
        "direction": "candidate_minus_comparator_lower_is_better",
        "group_count": len(values),
        "draws": len(indices),
        "seed": BOOTSTRAP_SEED,
        "delta": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "upper95": float(np.quantile(means, 0.95)),
        "one_sided_p": float((1 + np.sum(means >= 0.0)) / (len(means) + 1)),
        "bootstrap_means": means.tolist(),
    }


def holm_one_sided(
    comparisons: Mapping[str, Mapping[str, Any]], *, alpha: float
) -> dict[str, JsonDict]:
    """Apply Holm step-down tests and simultaneous upper loss bounds."""

    ordered = sorted(
        comparisons,
        key=lambda name: (float(comparisons[name]["one_sided_p"]), name),
    )
    family_size = len(ordered)
    if family_size < 1:
        raise ValueError("holm_family_empty")
    cumulative = 0.0
    output: dict[str, JsonDict] = {}
    for rank, name in enumerate(ordered, start=1):
        row = comparisons[name]
        means = np.asarray(row.get("bootstrap_means"), dtype=np.float64)
        threshold = alpha / (family_size - rank + 1)
        cumulative = max(
            cumulative,
            min(1.0, (family_size - rank + 1) * float(row["one_sided_p"])),
        )
        output[name] = {
            key: deepcopy(value) for key, value in row.items() if key != "bootstrap_means"
        }
        output[name].update(
            {
                "comparator": name,
                "holm_rank": rank,
                "holm_threshold": threshold,
                "holm_adjusted_p": cumulative,
                "simultaneous_upper95": float(np.quantile(means, 1.0 - threshold)),
            }
        )
    return output


def _validated_group_rows(
    rows: Sequence[Mapping[str, Any]], expected_groups: int
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Reject roster or metric drift before resampling can hide it."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    component_to_group: dict[str, str] = {}
    for row in rows:
        component = str(row.get("source_component_hash") or "")
        group = str(row.get("group_id") or "")
        arm = str(row.get("arm") or "")
        if not component or not group or arm not in ARMS:
            raise ValueError("row_identity_invalid")
        if component in component_to_group and component_to_group[component] != group:
            raise ValueError("source_component_mapping_invalid")
        component_to_group[component] = group
        if arm in grouped[component]:
            raise ValueError("duplicate_group_arm")
        if (
            row.get("status") != "complete"
            or row.get("complete") is not True
            or row.get("failed") is not False
            or row.get("excluded") is not False
            or row.get("censored") is not False
        ):
            raise ValueError("row_disposition_invalid")
        probability = float(row.get("probability"))
        label = int(row.get("label"))
        losses = loss_fields(probability, label)
        if any(
            not math.isclose(float(row.get(name)), value, abs_tol=1e-12)
            for name, value in losses.items()
        ):
            raise ValueError("loss_field_mismatch")
        decision = primary_decision(probability, label)
        if row.get("action") != decision["action"] or not math.isclose(
            float(row.get("realized_cost")), float(decision["realized_cost"]), abs_tol=1e-12
        ):
            raise ValueError("primary_decision_mismatch")
        grouped[component][arm] = row
    if len(grouped) != expected_groups:
        raise ValueError("expected_group_count_mismatch")
    if any(set(arm_rows) != set(ARMS) for arm_rows in grouped.values()):
        raise ValueError("arm_roster_invalid")
    if any(
        len({int(row["label"]) for row in arm_rows.values()}) != 1 for arm_rows in grouped.values()
    ):
        raise ValueError("group_label_disagreement")
    return grouped


def reduce_rows(rows: Sequence[Mapping[str, Any]], *, settings: Mapping[str, Any]) -> JsonDict:
    """Independently reduce complete source groups into registered gates."""

    expected_groups = int(settings["expected_groups"])
    grouped = _validated_group_rows(rows, expected_groups)
    components = sorted(grouped)
    labels = [int(grouped[component][CANDIDATE_ARM]["label"]) for component in components]
    label_counts = {"0": labels.count(0), "1": labels.count(1)}
    support_passed = len(components) >= int(settings["minimum_groups"]) and min(
        label_counts.values()
    ) >= int(settings["minimum_per_label"])
    support = {
        "valid_groups": len(components),
        "minimum_groups": int(settings["minimum_groups"]),
        "label_counts": label_counts,
        "minimum_per_label": int(settings["minimum_per_label"]),
        "passed": support_passed,
    }
    indices = bootstrap_indices(
        len(components),
        draws=int(settings["bootstrap_draws"]),
        seed=int(settings["bootstrap_seed"]),
    )
    raw_brier = {
        comparator: paired_interval(
            [
                float(grouped[component][CANDIDATE_ARM]["brier"])
                - float(grouped[component][comparator]["brier"])
                for component in components
            ],
            indices,
        )
        for comparator in BRIER_COMPARATORS
    }
    brier = holm_one_sided(raw_brier, alpha=float(settings["holm_family_alpha"]))
    selected = str(settings["selected_comparator"])
    log_loss = paired_interval(
        [
            float(grouped[component][CANDIDATE_ARM]["log_loss"])
            - float(grouped[component][selected]["log_loss"])
            for component in components
        ],
        indices,
    )
    log_loss.pop("bootstrap_means")
    log_loss["comparator"] = selected
    primary_cost = paired_interval(
        [
            float(grouped[component][CANDIDATE_ARM]["realized_cost"])
            - float(grouped[component][selected]["realized_cost"])
            for component in components
        ],
        indices,
    )
    primary_cost.pop("bootstrap_means")
    primary_cost["comparator"] = selected
    candidate_non_escalation = float(
        np.mean(
            [grouped[component][CANDIDATE_ARM]["action"] != "escalate" for component in components]
        )
    )
    probability_metrics = {
        arm: {
            "brier": float(np.mean([grouped[item][arm]["brier"] for item in components])),
            "log_loss": float(np.mean([grouped[item][arm]["log_loss"] for item in components])),
            "primary_cost": float(
                np.mean([grouped[item][arm]["realized_cost"] for item in components])
            ),
            "non_escalation_fraction": float(
                np.mean([grouped[item][arm]["action"] != "escalate" for item in components])
            ),
            "n_source_components": len(components),
        }
        for arm in ARMS
    }
    brier_passed = all(
        float(row["delta"]) <= -float(settings["brier_improvement"])
        and float(row["holm_adjusted_p"]) <= float(settings["holm_family_alpha"])
        and float(row["simultaneous_upper95"]) < 0.0
        for row in brier.values()
    )
    log_loss_passed = float(log_loss["upper95"]) <= float(
        settings["maximum_log_loss_deterioration"]
    )
    cost_nonregression = float(primary_cost["upper95"]) <= float(
        settings["maximum_primary_cost_deterioration"]
    )
    confirmatory = settings.get("confirmatory_allowed") is True
    probability_score = int(
        support_passed and confirmatory and brier_passed and log_loss_passed and cost_nonregression
    )
    decision_cost_passed = float(primary_cost["upper95"]) < 0.0
    coverage_passed = candidate_non_escalation >= float(settings["minimum_non_escalation_fraction"])
    decision_score = int(probability_score == 1 and decision_cost_passed and coverage_passed)
    failures = []
    for name, passed in (
        ("source_support", support_passed),
        ("fresh_confirmatory_claim_forbidden", confirmatory),
        ("registered_brier_family", brier_passed),
        ("log_loss_nonregression", log_loss_passed),
        ("primary_cost_nonregression", cost_nonregression),
        ("primary_cost_improvement", decision_cost_passed),
        ("candidate_non_escalation", coverage_passed),
    ):
        if not passed:
            failures.append(name)
    return {
        "static_measurement_complete_score": 1,
        "probability_benefit_score": probability_score,
        "decision_benefit_score": decision_score,
        "support": support,
        "probability_metrics": probability_metrics,
        "paired_intervals": {"brier": brier, "log_loss": log_loss},
        "primary_cost_contrast": primary_cost,
        "candidate_non_escalation_fraction": candidate_non_escalation,
        "failed_benefit_gates": failures,
        "sample_size_budget": {
            "independent_unit": "unique_source_component",
            "planned": expected_groups,
            "attempted": len(components),
            "completed": len(components),
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": max(0, expected_groups - len(components)),
            "orders_are_independent_units": False,
            "arms_are_independent_units": False,
        },
    }


def analytical_panel_rows(group_count: int = 80) -> list[JsonDict]:
    """Create an oracle-defined sensitivity panel outside empirical evidence."""

    output: list[JsonDict] = []
    for index in range(group_count):
        label = index % 2
        for arm in ARMS:
            probability = (0.02 if label == 0 else 0.98) if arm == CANDIDATE_ARM else 0.5
            losses = loss_fields(probability, label)
            decision = primary_decision(probability, label)
            output.append(
                {
                    "unit_id": f"positive-control-{index}",
                    "source_component_hash": f"positive-control-{index}",
                    "group_id": f"positive-control-group-{index}",
                    "role": "analytical_positive_control",
                    "arm": arm,
                    "head_identity": f"positive-control:{arm}",
                    "option_probabilities": {
                        ORDER_NAMES[0]: probability,
                        ORDER_NAMES[1]: probability,
                    },
                    "probability": probability,
                    "label": label,
                    "y": label,
                    **losses,
                    "action": decision["action"],
                    "realized_cost": decision["realized_cost"],
                    "descriptive_cost_grid": [
                        _decision_cell(
                            probability,
                            label,
                            false_accept_cost=false_accept,
                            escalation_cost=escalation,
                        )
                        for false_accept, escalation in COST_GRID
                    ],
                    "source_intervention_diagnostics": {
                        "source_dependence_applicable": arm == CANDIDATE_ARM,
                        "source_neutral_probability": (0.5 if arm == CANDIDATE_ARM else None),
                        "source_neutral_absolute_delta": (
                            abs(probability - 0.5) if arm == CANDIDATE_ARM else None
                        ),
                        "source_neutral_disposition": (
                            "measured"
                            if arm == CANDIDATE_ARM
                            else "not_applicable_analytical_comparator"
                        ),
                        "option_order_absolute_gap": 0.0,
                        "missing_source_confidence_is_oracle": False,
                    },
                    "status": "complete",
                    "attempted": True,
                    "complete": True,
                    "failed": False,
                    "excluded": False,
                    "censored": False,
                    "unstarted": False,
                }
            )
    return output


def run_analytical_positive_control(*, draws: int = BOOTSTRAP_DRAWS) -> JsonDict:
    """Prove the registered evaluation can detect a known analytical effect."""

    reduction = reduce_rows(
        analytical_panel_rows(),
        settings=evaluator_settings(draws=draws, confirmatory_allowed=True),
    )
    passed = (
        reduction["probability_benefit_score"] == 1 and reduction["decision_benefit_score"] == 1
    )
    return {
        "panel": "analytical_oracle_defined_separate_from_empirical_rows",
        "verdict_ceiling": "circular_positive",
        "contributes_empirical_rows": False,
        "draws": draws,
        "passed": passed,
        "probability_benefit_score": reduction["probability_benefit_score"],
        "decision_benefit_score": reduction["decision_benefit_score"],
        "reduction": reduction,
    }


def precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    path: str,
    *,
    op: str = "==",
) -> JsonDict:
    """Retain the exact external operand that can stop evaluation."""

    if op == "in":
        passed = observed in expected
    else:
        passed = observed == expected
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "category": "external_precondition",
        "principle": "Missing or changed upstream evidence must stop before label access.",
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
    Path("python/carnot/experiment_7507_v657_static_evaluation.py"),
    Path("python/carnot/experiment_7508_v657_static_audit.py"),
    Path("python/carnot/experiment_7533_v659_tool_protocol.py"),
    Path("python/carnot/experiment_7565_v661_test_online_capture.py"),
    Path("python/carnot/experiment_7566_v661_energy_fit.py"),
    SPEC_PATH,
    CAPTURE_PATH,
    FIT_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _resolve_reference(root: Path, label: str) -> Path:
    """Resolve one declared sidecar path without changing its target."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate instructions, readiness gates, roles, heads, and sidecars."""

    root = root.resolve()
    rows: list[JsonDict] = []
    sources: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        readable = path.is_file() and path.stat().st_size > 0
        rows.append(
            precondition_row(
                "resource_readable",
                relative.as_posix(),
                "path.bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if readable else None,
                relative.as_posix(),
            )
        )
        if readable:
            sources.append(source_hash_row(relative, root))
    spec_present = (
        "REQ-VERIFY-7567" in (root / SPEC_PATH).read_text(encoding="utf-8")
        if (root / SPEC_PATH).is_file()
        else False
    )
    rows.append(
        precondition_row(
            "requirement_present",
            "openspec-verification",
            "REQ-VERIFY-7567",
            True,
            spec_present,
            SPEC_PATH.as_posix(),
        )
    )
    if not all(row["passed"] for row in rows):
        return {"passed": False, "rows": rows, "source_artifact_hashes": sources}
    capture = load_json(root / CAPTURE_PATH)
    fit = load_json(root / FIT_PATH)
    gate_specs = (
        (
            "test_capture_ready",
            "Exp7565",
            "test_capture_ready_score",
            1,
            capture.get("test_capture_ready_score"),
            CAPTURE_PATH,
            "==",
        ),
        (
            "capture_verdict",
            "Exp7565",
            "verdict_class",
            ["null", "positive"],
            capture.get("verdict_class"),
            CAPTURE_PATH,
            "in",
        ),
        (
            "capture_unflagged",
            "Exp7565",
            "flagged_adversarial",
            False,
            capture.get("flagged_adversarial"),
            CAPTURE_PATH,
            "==",
        ),
        (
            "test_group_count",
            "Exp7565",
            "role_counts.test",
            80,
            capture.get("role_counts", {}).get("test"),
            CAPTURE_PATH,
            "==",
        ),
        (
            "label_after_prediction_freeze",
            "Exp7565",
            "raw_manifest.role_sidecars.label_release_after_prediction_freeze",
            True,
            capture.get("raw_manifest", {})
            .get("role_sidecars", {})
            .get("label_release_after_prediction_freeze"),
            CAPTURE_PATH,
            "==",
        ),
        (
            "energy_fit_ready",
            "Exp7566",
            "energy_fit_ready_score",
            1,
            fit.get("energy_fit_ready_score"),
            FIT_PATH,
            "==",
        ),
        (
            "baseline_ready",
            "Exp7566",
            "baseline_ready_score",
            1,
            fit.get("baseline_ready_score"),
            FIT_PATH,
            "==",
        ),
        (
            "fit_verdict",
            "Exp7566",
            "verdict_class",
            ["null", "positive"],
            fit.get("verdict_class"),
            FIT_PATH,
            "in",
        ),
        (
            "fit_unflagged",
            "Exp7566",
            "flagged_adversarial",
            False,
            fit.get("flagged_adversarial"),
            FIT_PATH,
            "==",
        ),
        (
            "heads_frozen",
            "Exp7566",
            "frozen_head_manifest.frozen_before_policy_access",
            True,
            fit.get("frozen_head_manifest", {}).get("frozen_before_policy_access"),
            FIT_PATH,
            "==",
        ),
        (
            "strongest_comparator",
            "Exp7566",
            "frozen_head_manifest.strongest_comparator.family",
            "temperature_original",
            fit.get("frozen_head_manifest", {}).get("strongest_comparator", {}).get("family"),
            FIT_PATH,
            "==",
        ),
    )
    for check, upstream, field, expected, observed, path, op in gate_specs:
        rows.append(
            precondition_row(
                check,
                upstream,
                field,
                expected,
                observed,
                path.as_posix(),
                op=op,
            )
        )
    schedule = capture.get("capture_schedule_manifest")
    schedule_valid = (
        isinstance(schedule, Mapping)
        and schedule.get("role_counts") == {"test": 80, "online": 160}
        and set(schedule.get("roles") or []) == {"test", "online"}
        and isinstance(schedule.get("role_schedule_hashes", {}).get("test"), str)
    )
    rows.append(
        precondition_row(
            "role_manifest_authenticated",
            "Exp7565",
            "capture_schedule_manifest",
            True,
            schedule_valid,
            CAPTURE_PATH.as_posix(),
        )
    )
    role_sidecars = capture.get("raw_manifest", {}).get("role_sidecars", {})
    fit_sidecars = fit.get("frozen_head_manifest", {}).get("sidecars", {})
    sidecar_specs = (
        ("capture_predictions", role_sidecars.get("predictions")),
        ("capture_test_labels", role_sidecars.get("test_labels")),
        ("frozen_heads", fit_sidecars.get("frozen_heads")),
    )
    authenticated_sidecars: dict[str, JsonDict] = {}
    for name, reference in sidecar_specs:
        if not isinstance(reference, Mapping):
            rows.append(
                precondition_row(
                    "sidecar_hash",
                    "Exp7565_or_Exp7566",
                    name,
                    "hash_bound_sidecar",
                    reference,
                    "missing_reference",
                )
            )
            continue
        label = str(reference.get("path") or "")
        path = _resolve_reference(root, label)
        observed = sha256_file(path) if path.is_file() else None
        expected = reference.get("sha256")
        rows.append(
            precondition_row(
                "sidecar_hash",
                "Exp7565_or_Exp7566",
                name,
                expected,
                observed,
                label,
            )
        )
        if observed == expected and isinstance(expected, str):
            authenticated_sidecars[name] = deepcopy(dict(reference))
            sources.append(source_hash_row(path, root))
    if "frozen_heads" in authenticated_sidecars:
        heads = load_json(
            _resolve_reference(root, str(authenticated_sidecars["frozen_heads"]["path"]))
        )
        expected_bundle = fit.get("frozen_head_manifest", {}).get("bundle_sha256")
        rows.append(
            precondition_row(
                "frozen_head_bundle",
                "Exp7566",
                "frozen_heads.bundle_sha256",
                expected_bundle,
                heads.get("bundle_sha256"),
                str(authenticated_sidecars["frozen_heads"]["path"]),
            )
        )
        rows.append(
            precondition_row(
                "frozen_arm_roster",
                "Exp7566",
                "frozen_heads.selected_heads",
                sorted(TRAINED_ARMS),
                sorted((heads.get("selected_heads") or {}).keys()),
                str(authenticated_sidecars["frozen_heads"]["path"]),
            )
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "source_artifact_hashes": sources,
        "sidecars": authenticated_sidecars,
        "upstream_context": {
            "capture_prediction_freeze_sha256": role_sidecars.get("prediction_freeze_sha256"),
            "capture_schedule_sha256": capture.get("capture_schedule_manifest", {}).get(
                "schedule_sha256"
            ),
            "frozen_bundle_sha256": fit.get("frozen_head_manifest", {}).get("bundle_sha256"),
            "exposure_audit": deepcopy(capture.get("exposure_audit", {})),
        },
    }


def load_label_free_inputs(root: Path, preconditions: Mapping[str, Any]) -> JsonDict:
    """Load only test features and frozen heads while labels remain unopened."""

    sidecars = preconditions.get("sidecars")
    if not isinstance(sidecars, Mapping):
        raise ValueError("authenticated_sidecars_missing")
    prediction_reference = sidecars.get("capture_predictions")
    head_reference = sidecars.get("frozen_heads")
    label_reference = sidecars.get("capture_test_labels")
    if not all(
        isinstance(reference, Mapping)
        for reference in (prediction_reference, head_reference, label_reference)
    ):
        raise ValueError("authenticated_sidecars_missing")
    prediction_path = _resolve_reference(root, str(prediction_reference["path"]))
    features = [row for row in load_jsonl(prediction_path) if row.get("role") == "test"]
    if len(features) != 80 or len({row.get("component_hash") for row in features}) != 80:
        raise ValueError("test_feature_roster_invalid")
    if any("label" in row or "y" in row for row in features):
        raise ValueError("label_visible_during_prediction")
    heads = load_json(_resolve_reference(root, str(head_reference["path"])))
    return {
        "features": features,
        "heads": heads,
        "label_reference": deepcopy(dict(label_reference)),
        "upstream_prediction_freeze_sha256": preconditions.get("upstream_context", {}).get(
            "capture_prediction_freeze_sha256"
        ),
    }


def open_test_labels(root: Path, reference: Mapping[str, Any]) -> list[JsonDict]:
    """Open the sealed test labels only after the caller persists predictions."""

    rows = load_jsonl(_resolve_reference(root, str(reference.get("path") or "")))
    if len(rows) != 80 or any(row.get("role") != "test" for row in rows):
        raise ValueError("test_label_roster_invalid")
    return rows


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
    "static_measurement_complete_score",
    "probability_benefit_score",
    "decision_benefit_score",
    "paired_intervals",
    "positive_control_results",
    "exposure_audit",
)


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
) -> JsonDict:
    """Attach an exact operand and claim-boundary principle to one gate."""

    principles = {
        "validity": "Invalid evidence cannot support science.",
        "readiness": "A valid null remains reusable.",
        "benefit": "Completion cannot substitute for empirical value.",
    }
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principles[category],
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed gate and the first exact operand."""

    failed = [row for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": (
            {
                key: deepcopy(failed[0].get(key))
                for key in ("check", "expected", "observed", "op", "category")
            }
            if failed
            else None
        ),
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain which silent failure each emitted field prevents."""

    specific = {
        "experiment_id": "The exact experiment name prevents cross-task substitution.",
        "preconditions_checked": "Exact operands prevent fabricated fallback data.",
        "MODEL_SPECS": "An empty plan prevents historical model work becoming current work.",
        "model_specs": "Resolved model identity stays empty because this task loads no model.",
        "model_invoked": "The invocation flag prevents aggregation from being called inference.",
        "invocation_counts": "Typed zero counters prevent hidden current model calls.",
        "inference_substrate_class": "The class applies the correct duration floor.",
        "inference_substrate": "The substrate distinguishes aggregation from generation.",
        "execution_venue": "A legal venue value prevents device identity from replacing host.",
        "duration_s": "Measured monotonic time prevents invented runtime claims.",
        "random_seed": "Frozen seeds prevent outcome-dependent resampling.",
        "reproducibility_checksum": "The checksum binds code, settings, rows, and evidence.",
        "rows": "Absolute arm rows prevent missing evidence from becoming zero.",
        "sample_size_budget": "Disposition counts prevent attrition from disappearing.",
        "acceptance_gate_results": "Typed gates keep validity, readiness, and benefit separate.",
        "gate_check_summary": "Exact failures prevent an ambiguous terminal disposition.",
        "honest_verdict": "A terminal prefix prevents a valid null from being retried.",
        "verdict_class": "The closed class prevents blocked evidence from becoming null.",
        "verifier_is_oracle": "The declaration prevents probabilistic energy from certifying truth.",
        "flagged_adversarial": "Actual safety findings cannot be erased to open a gate.",
        "validation_receipts": "Exact command outcomes prevent unrun checks from appearing green.",
        "static_measurement_complete_score": "Readiness stays independent from empirical benefit.",
        "probability_benefit_score": "All registered proper-loss gates must pass together.",
        "decision_benefit_score": "Typed cost and coverage must pass beyond probability benefit.",
        "paired_intervals": "Clustered intervals retain direction, multiplicity, and sample count.",
        "positive_control_results": "Evaluation power must pass before interpreting a null.",
        "exposure_audit": "Prior label access cannot become a fresh confirmatory claim.",
        "field_principles": "Every field names the omission or drift it prevents.",
    }
    return {
        key: specific.get(key, f"Retaining {key} prevents silent omission or scope drift.")
        for key in keys
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind all stable artifact content except the checksum itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute every empirical score from retained source-arm rows."""

    settings = value.get("evaluator_settings")
    rows = value.get("rows")
    if not isinstance(settings, Mapping) or not isinstance(rows, list):
        raise ValueError("independent_reduction_inputs_missing")
    return reduce_rows(rows, settings=settings)


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    challenges: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    exposure_audit: Mapping[str, Any],
    raw_sidecars: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    duration_s: float,
    flagged_adversarial: bool = False,
    fixture: bool = False,
) -> JsonDict:
    """Assemble one independently reducible static evaluation record."""

    reduction = reduce_rows(rows, settings=settings)
    static_complete = int(
        reduction["static_measurement_complete_score"] == 1
        and validation_passed
        and challenges.get("passed") is True
        and positive_control.get("passed") is True
    )
    probability_score = int(static_complete == 1 and reduction["probability_benefit_score"] == 1)
    decision_score = int(probability_score == 1 and reduction["decision_benefit_score"] == 1)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "version": "v661",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Held-out source-grounded injected tool-error evaluation",
        "complete": True,
        "ready": static_complete,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "historical_invocation_counts": {
            "source": "Exp7565 native option forwards",
            "counted_as_current": False,
        },
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "readout_kind": "option_logits",
            "generated_tokens": 0,
            "native_option_forwards_are_historical": True,
            "aggregation": "frozen_head_scoring_and_source_component_bootstrap",
        },
        "execution_venue": "host",
        "execution_device": {
            "type": "cpu",
            "identity": platform.processor() or platform.machine(),
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "executable": sys.executable},
        "random_seed": {
            "model": None,
            "fitting": 7_566_001,
            "ordering": 659_033,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "input_roles": {
            "prediction": "Exp7565 sealed test feature rows",
            "labels": "Exp7565 test labels opened after local prediction freeze",
            "heads": "Exp7566 frozen tune-selected heads",
        },
        "rows": [deepcopy(dict(row)) for row in rows],
        "sample_size_budget": deepcopy(reduction["sample_size_budget"]),
        "evaluator_settings": deepcopy(dict(settings)),
        "probability_metrics": deepcopy(reduction["probability_metrics"]),
        "paired_intervals": deepcopy(reduction["paired_intervals"]),
        "primary_cost_contrast": deepcopy(reduction["primary_cost_contrast"]),
        "candidate_non_escalation_fraction": reduction["candidate_non_escalation_fraction"],
        "failed_benefit_gates": deepcopy(reduction["failed_benefit_gates"]),
        "challenge_controls": deepcopy(dict(challenges)),
        "positive_control_results": deepcopy(dict(positive_control)),
        "exposure_audit": deepcopy(dict(exposure_audit)),
        "raw_sidecars": deepcopy(dict(raw_sidecars)),
        "static_report_frozen_for_exp7569": bool(raw_sidecars.get("static_report")),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_summary": {
            "required_checks_passed": validation_passed,
            "receipt_count": len(validation_receipts),
            "cold_replay_included": any(
                row.get("name") == "declared_entrypoint_cold_replay" for row in validation_receipts
            ),
            "independent_reduction_included": any(
                row.get("name") == "independent_row_reduction" for row in validation_receipts
            ),
        },
        "applicable_numbered_e2e": [],
        "capability_e2e": "declared_entrypoint_and_cold_replay",
        "private_llm_off_real_environment_smoke": "not_applicable_static_read_only_evaluation",
        "learning_lifecycle": {
            "predict": "all_arm_predictions_frozen_without_labels",
            "release": "sealed_test_labels_opened_after_prediction_sidecar",
            "update": "not_applicable_no_learning_or_refit",
            "persist": bool(raw_sidecars),
            "reload": bool(raw_sidecars),
            "generator_weights_changed": False,
        },
        "scientific_scope": {
            "evaluates": "source_grounded_injected_tool_errors",
            "general_hallucination_detection": False,
            "deterministic_correctness": False,
            "source_truth_certification": False,
            "post_test_refit": False,
            "threshold_change": False,
            "subgroup_selection": False,
        },
        "generator_weights_changed": False,
        "external_publication_authorized": False,
        "verifier_is_oracle": False,
        "flagged_adversarial": flagged_adversarial,
        "adversarial_corrections": [],
        "static_measurement_complete_score": static_complete,
        "probability_benefit_score": probability_score,
        "decision_benefit_score": decision_score,
        "reproducibility_checksum": "pending",
    }
    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    confirmatory = settings.get("confirmatory_allowed") is True
    gates = [
        _gate(
            "preconditions",
            "validity",
            True,
            preconditions_passed,
            "==",
            preconditions_passed,
        ),
        _gate(
            "raw_custody",
            "validity",
            80,
            artifact["sample_size_budget"]["completed"],
            "==",
            artifact["sample_size_budget"]["completed"] == 80,
        ),
        _gate(
            "challenge_controls",
            "validity",
            True,
            challenges.get("passed"),
            "==",
            challenges.get("passed") is True,
        ),
        _gate(
            "analytical_positive_control",
            "validity",
            True,
            positive_control.get("passed"),
            "==",
            positive_control.get("passed") is True,
        ),
        _gate(
            "scoped_and_terminal_validation",
            "validity",
            True,
            validation_passed,
            "==",
            validation_passed,
        ),
        _gate(
            "static_measurement_complete",
            "readiness",
            1,
            static_complete,
            "==",
            static_complete == 1,
        ),
        _gate(
            "fresh_confirmatory_claim_allowed",
            "benefit",
            True,
            confirmatory,
            "==",
            confirmatory,
        ),
        _gate(
            "probability_benefit",
            "benefit",
            1,
            probability_score,
            "==",
            probability_score == 1,
        ),
        _gate(
            "decision_benefit",
            "benefit",
            1,
            decision_score,
            "==",
            decision_score == 1,
        ),
    ]
    artifact["acceptance_gate_results"] = gates
    artifact["gate_check_summary"] = _gate_summary(gates)
    if fixture:
        artifact["honest_verdict"] = "complete_circular_positive_analytical_fixture_only"
        artifact["verdict_class"] = "circular_positive"
    elif not validation_passed or flagged_adversarial:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["static_measurement_complete_score"] = 0
        artifact["probability_benefit_score"] = 0
        artifact["decision_benefit_score"] = 0
    elif probability_score == 1:
        artifact["honest_verdict"] = "complete_positive_source_probability_benefit"
        artifact["verdict_class"] = "positive"
    else:
        artifact["honest_verdict"] = "complete_null_source_evaluation_no_supported_benefit"
        artifact["verdict_class"] = "null"
    artifact["positive_claim"] = artifact["verdict_class"] == "positive"
    artifact["field_principles"] = _field_principles((*artifact.keys(), "field_principles"))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def fixture_artifact() -> JsonDict:
    """Build a cold-valid circular fixture for schema and mutation tests."""

    rows = analytical_panel_rows()
    settings = evaluator_settings(draws=64, confirmatory_allowed=True)
    controls = run_challenge_controls(rows)
    positive = run_analytical_positive_control(draws=64)
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
                "fixture",
                "analytical_panel",
                "group_count",
                80,
                80,
                "in_memory",
            )
        ],
        source_hashes=[],
        rows=rows,
        settings=settings,
        challenges=controls,
        positive_control=positive,
        exposure_audit={
            "prior_test_label_access": False,
            "confirmatory_claim_allowed": True,
            "fixture_only": True,
        },
        raw_sidecars={"fixture": {"path": "in_memory"}},
        validation_receipts=receipts,
        validation_passed=True,
        phase_spans=[],
        duration_s=0.01,
        fixture=True,
    )


def blocked_artifact(failed: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Publish external absence as complete blocked work with exact operands."""

    gate = _gate(
        str(failed.get("check") or "external_input"),
        "validity",
        failed.get("expected"),
        failed.get("observed"),
        str(failed.get("op") or "=="),
        False,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "version": "v661",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Held-out source-grounded injected tool-error evaluation",
        "complete": True,
        "ready": 0,
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
            "generated_tokens": 0,
            "blocked_before_prediction": True,
        },
        "execution_venue": "host",
        "execution_device": {
            "type": "cpu",
            "identity": platform.processor() or platform.machine(),
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "executable": sys.executable},
        "random_seed": {
            "model": None,
            "fitting": 7_566_001,
            "ordering": 659_033,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": [],
        "input_roles": {},
        "rows": [deepcopy(dict(failed))],
        "sample_size_budget": {
            "independent_unit": "unique_source_component",
            "planned": 80,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 80,
            "orders_are_independent_units": False,
            "arms_are_independent_units": False,
        },
        "evaluator_settings": evaluator_settings(),
        "probability_metrics": {},
        "paired_intervals": {},
        "primary_cost_contrast": {},
        "candidate_non_escalation_fraction": None,
        "failed_benefit_gates": ["external_precondition"],
        "challenge_controls": {"passed": False, "rows": []},
        "positive_control_results": {"passed": False, "not_run": True},
        "exposure_audit": {},
        "raw_sidecars": {},
        "static_report_frozen_for_exp7569": False,
        "validation_receipts": [],
        "validation_summary": {"required_checks_passed": False, "receipt_count": 0},
        "applicable_numbered_e2e": [],
        "capability_e2e": "not_run_external_precondition_failed",
        "private_llm_off_real_environment_smoke": "not_applicable_static_read_only_evaluation",
        "learning_lifecycle": {
            "predict": "not_started",
            "release": "not_started",
            "update": "not_applicable",
            "persist": False,
            "reload": False,
            "generator_weights_changed": False,
        },
        "scientific_scope": {
            "evaluates": "nothing_external_precondition_failed",
            "general_hallucination_detection": False,
            "deterministic_correctness": False,
            "source_truth_certification": False,
        },
        "generator_weights_changed": False,
        "external_publication_authorized": False,
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "passed": False,
            "failed_count": 1,
            "failed_checks": [failed.get("check")],
            "upstream": failed.get("upstream"),
            "path": failed.get("path"),
            "field": failed.get("field"),
            "expected": deepcopy(failed.get("expected")),
            "observed": deepcopy(failed.get("observed")),
            "op": failed.get("op"),
        },
        "honest_verdict": f"complete_blocked_{failed.get('check', 'external_input')}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "positive_claim": False,
        "static_measurement_complete_score": 0,
        "probability_benefit_score": 0,
        "decision_benefit_score": 0,
        "reproducibility_checksum": "pending",
    }
    artifact["field_principles"] = _field_principles((*artifact.keys(), "field_principles"))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, provenance, raw reduction, scores, and hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_FIELDS) - set(value))
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_must_be_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    blocked = value.get("verdict_class") == "blocked"
    expected_class = "blocked_no_run" if blocked else "no_model_load"
    if value.get("inference_substrate_class") != expected_class:
        errors.append("inference_substrate_class_mismatch")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
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
        errors.append("honest_verdict_prefix_invalid")
    for field in (
        "static_measurement_complete_score",
        "probability_benefit_score",
        "decision_benefit_score",
    ):
        if type(value.get(field)) is not int or value.get(field) not in (0, 1):
            errors.append(f"bare_score_invalid:{field}")
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
    if not blocked:
        try:
            reduced = independent_reduce(value)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"independent_reduction_failed:{exc}")
            errors.append("independent_reduction_mismatch")
        else:
            comparable = {
                "sample_size_budget": reduced["sample_size_budget"],
                "probability_metrics": reduced["probability_metrics"],
                "paired_intervals": reduced["paired_intervals"],
                "primary_cost_contrast": reduced["primary_cost_contrast"],
                "candidate_non_escalation_fraction": reduced["candidate_non_escalation_fraction"],
                "failed_benefit_gates": reduced["failed_benefit_gates"],
            }
            if any(value.get(key) != expected for key, expected in comparable.items()):
                errors.append("independent_reduction_mismatch")
            validation_passed = value.get("validation_summary", {}).get("required_checks_passed")
            positive_passed = value.get("positive_control_results", {}).get("passed")
            challenge_passed = value.get("challenge_controls", {}).get("passed")
            expected_static = int(
                reduced["static_measurement_complete_score"] == 1
                and validation_passed is True
                and positive_passed is True
                and challenge_passed is True
            )
            expected_probability = int(
                expected_static == 1 and reduced["probability_benefit_score"] == 1
            )
            expected_decision = int(
                expected_probability == 1 and reduced["decision_benefit_score"] == 1
            )
            if value.get("verdict_class") != "disqualified" and (
                value.get("static_measurement_complete_score") != expected_static
                or value.get("probability_benefit_score") != expected_probability
                or value.get("decision_benefit_score") != expected_decision
            ):
                errors.append("independent_score_mismatch")
    if verify_sources:
        for row in value.get("source_artifact_hashes") or []:
            path = _resolve_reference(root, str(row.get("path") or ""))
            observed = sha256_file(path) if path.is_file() else None
            if observed != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{row.get('path')}")
        for reference in (value.get("raw_sidecars") or {}).values():
            if isinstance(reference, Mapping):
                errors.extend(verify_sidecar(reference, root))
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Validate exact serialized bytes through the public reader path."""

    return validate_artifact(load_json(path), root=root, verify_sources=verify_sources)


def independent_replay(path: Path) -> list[str]:
    """Recompute retained comparative rows without trusting producer scores."""

    value = load_json(path)
    errors = validate_artifact(value, verify_sources=False)
    if value.get("verdict_class") != "blocked":
        try:
            independent_reduce(value)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"independent_replay_failed:{exc}")
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one aware UTC timestamp for a durable phase boundary."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - real execution boundary.
    """Print one flushed phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7567] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:  # pragma: no cover - real execution boundary.
    """Close one monotonic phase and retain its completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "ended_utc": utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build cold replay, independent reduction, and both terminal guards."""

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
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(command, "required_validation", True) for command in commands]


def _all_receipts_pass(
    receipts: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> bool:  # pragma: no cover - exercised by the real capability E2E.
    """Require exactly one passing zero-exit receipt for each command."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[str(row.get("name"))].append(row)
    return all(
        len(grouped[name]) == 1
        and grouped[name][0].get("passed") is True
        and grouped[name][0].get("exit_code") == 0
        and grouped[name][0].get("timed_out") is not True
        for name in names
    )


def _provisional_terminal_receipts() -> list[JsonDict]:  # pragma: no cover
    """Supply self-check shapes before fresh processes replace them."""

    return [
        {
            "name": name,
            "scope": "terminal_candidate_shape",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "log_sha256": canonical_hash({"name": name, "candidate": True}),
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def _write_static_report(
    path: Path, report: Mapping[str, Any], root: Path
) -> JsonDict:  # pragma: no cover
    """Freeze deterministic science content for the independent Exp7569 audit."""

    encoded = (json.dumps(dict(report), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"immutable_static_report_conflict:{path}")
    if not path.exists():
        atomic_json(path, report)
    if path.stat().st_size >= 20 * 1024 * 1024:
        raise ValueError("static_report_size_limit")
    return source_hash_row(path, root)


def run_experiment(
    root: Path, run_date: str
) -> JsonDict:  # pragma: no cover - capability E2E owns filesystem and subprocesses.
    """Authenticate, predict, release, reduce, validate, and publish atomically."""

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
    failed = next(
        (row for row in preconditions["rows"] if row.get("passed") is not True),
        None,
    )
    if failed is not None:
        artifact = blocked_artifact(failed, root=root)
        artifact["duration_s"] = time.monotonic() - run_started
        artifact["phase_spans"] = spans
        artifact["field_principles"] = _field_principles((*artifact.keys(), "field_principles"))
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        progress(run_started, "publish", "before_atomic_terminal", verdict="blocked")
        atomic_json(root / RESULT_PATH, artifact)
        progress(
            run_started,
            "publish",
            "complete_blocked",
            reason=failed.get("check"),
        )
        return artifact
    progress(
        run_started,
        "preconditions",
        "complete",
        completed_units=len(preconditions["rows"]),
    )

    progress(run_started, "prediction_freeze", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    inputs = load_label_free_inputs(root, preconditions)
    predictions = score_label_free_predictions(inputs["features"], inputs["heads"])
    prediction_reference = write_jsonl_sidecar(
        root / LABEL_FREE_PATH,
        predictions,
        root=root,
    )
    spans.append(
        _span(
            "prediction_freeze",
            phase_started,
            run_started,
            80,
            "all_arm_predictions_persisted",
        )
    )
    progress(
        run_started,
        "prediction_freeze",
        "after_benchmark",
        completed_units=80,
        arm_rows=len(predictions),
    )

    progress(run_started, "label_release", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    labels = open_test_labels(root, inputs["label_reference"])
    labeled_rows = attach_test_labels(
        predictions,
        labels,
        prediction_sha256=str(prediction_reference["sha256"]),
        expected_upstream_freeze=str(inputs["upstream_prediction_freeze_sha256"]),
    )
    row_reference = write_jsonl_sidecar(
        root / LABELED_ROWS_PATH,
        labeled_rows,
        root=root,
    )
    spans.append(
        _span(
            "label_release",
            phase_started,
            run_started,
            80,
            "sealed_labels_joined_after_freeze",
        )
    )
    progress(
        run_started,
        "label_release",
        "after_benchmark",
        completed_units=80,
        arm_rows=len(labeled_rows),
    )

    upstream_exposure = preconditions["upstream_context"].get("exposure_audit", {})
    exposure = {
        **deepcopy(dict(upstream_exposure)),
        "prior_test_label_access": upstream_exposure.get("prior_outcome_inspection") is True,
        "confirmatory_claim_allowed": upstream_exposure.get("confirmatory_claim_allowed") is True,
        "evaluation_claim_scope": "descriptive_reuse_not_fresh_confirmatory",
    }
    settings = evaluator_settings(confirmatory_allowed=bool(exposure["confirmatory_claim_allowed"]))
    challenges = run_challenge_controls(predictions)

    progress(run_started, "positive_control", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    positive_control = run_analytical_positive_control(draws=BOOTSTRAP_DRAWS)
    spans.append(
        _span(
            "positive_control",
            phase_started,
            run_started,
            80,
            "analytical_sensitivity_panel",
        )
    )
    progress(
        run_started,
        "positive_control",
        "after_benchmark",
        completed_units=80,
        passed=positive_control["passed"],
    )

    progress(run_started, "empirical_reduction", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    reduction = reduce_rows(labeled_rows, settings=settings)
    static_report = {
        "schema": "carnot.exp7567.v661.static_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "settings": settings,
        "reduction": reduction,
        "challenge_controls": challenges,
        "positive_control_results": positive_control,
        "exposure_audit": exposure,
        "prediction_sidecar": prediction_reference,
        "row_sidecar": row_reference,
        "frozen_bundle_sha256": inputs["heads"].get("bundle_sha256"),
        "promotion_performed": False,
    }
    static_reference = _write_static_report(
        root / STATIC_REPORT_PATH,
        static_report,
        root,
    )
    raw_sidecars = {
        "label_free_predictions": prediction_reference,
        "source_group_rows": row_reference,
        "static_report": static_reference,
    }
    spans.append(
        _span(
            "empirical_reduction",
            phase_started,
            run_started,
            80,
            "static_report_frozen_for_exp7569",
        )
    )
    progress(
        run_started,
        "empirical_reduction",
        "after_benchmark",
        completed_units=80,
        probability_score=reduction["probability_benefit_score"],
        decision_score=reduction["decision_benefit_score"],
    )

    atomic_json(
        root / VALIDATION_MANIFEST_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    )
    raw_sidecars["affected_validation_manifest"] = source_hash_row(
        root / VALIDATION_MANIFEST_PATH, root
    )
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7567-validation-", dir="/tmp"))
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
            "scoped_validation_logs",
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    provisional = _provisional_terminal_receipts()
    candidate = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=preconditions["source_artifact_hashes"],
        rows=labeled_rows,
        settings=settings,
        challenges=challenges,
        positive_control=positive_control,
        exposure_audit=exposure,
        raw_sidecars=raw_sidecars,
        validation_receipts=[*affected, *provisional],
        validation_passed=bool(affected_reduction["passed"]),
        phase_spans=spans,
        duration_s=time.monotonic() - run_started,
    )
    candidate_errors = validate_artifact(candidate, root=root)
    if candidate_errors:
        raise RuntimeError(f"measured_candidate_invalid:{candidate_errors}")
    atomic_json(root / TERMINAL_CANDIDATE_PATH, candidate)

    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        completed_units=0,
        commands=len(TERMINAL_CHECK_NAMES),
    )
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        _terminal_commands(TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation" / "terminal",
    )
    terminal_passed = _all_receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    adversarial = next(
        (row for row in terminal if row.get("name") == "adversarial_verify"),
        {},
    )
    critical = adversarial.get("passed") is not True or "CRITICAL" in str(
        adversarial.get("output_tail") or ""
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            run_started,
            len(terminal),
            "fresh_process_candidate_checks",
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
    final = build_artifact(
        preconditions=preconditions["rows"],
        source_hashes=preconditions["source_artifact_hashes"],
        rows=labeled_rows,
        settings=settings,
        challenges=challenges,
        positive_control=positive_control,
        exposure_audit=exposure,
        raw_sidecars=raw_sidecars,
        validation_receipts=[*affected, *terminal],
        validation_passed=all_validation,
        phase_spans=spans,
        duration_s=time.monotonic() - run_started,
        flagged_adversarial=critical,
    )
    final_errors = validate_artifact(final, root=root)
    if final_errors:
        raise RuntimeError(f"terminal_artifact_invalid:{final_errors}")
    atomic_json(root / TERMINAL_CANDIDATE_PATH, final)

    progress(
        run_started,
        "promotion_guards",
        "before_subprocesses",
        completed_units=0,
        commands=len(TERMINAL_CHECK_NAMES),
    )
    exact_checks = run_categorized_commands(
        root,
        _terminal_commands(TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation" / "exact_terminal",
    )
    exact_passed = _all_receipts_pass(exact_checks, TERMINAL_CHECK_NAMES)
    exact_adversarial = next(
        (row for row in exact_checks if row.get("name") == "adversarial_verify"),
        {},
    )
    exact_critical = exact_adversarial.get("passed") is not True or "CRITICAL" in str(
        exact_adversarial.get("output_tail") or ""
    )
    progress(
        run_started,
        "promotion_guards",
        "after_subprocesses",
        completed_units=len(exact_checks),
        passed=exact_passed,
        critical=exact_critical,
    )
    if not exact_passed or exact_critical:
        raise RuntimeError("exact_terminal_promotion_guards_failed")
    if (root / TERMINAL_CANDIDATE_PATH).stat().st_size >= 20 * 1024 * 1024:
        raise RuntimeError("terminal_artifact_size_limit")
    progress(
        run_started,
        "publish",
        "before_atomic_terminal",
        path=RESULT_PATH.as_posix(),
    )
    atomic_json(root / RESULT_PATH, final)
    progress(
        run_started,
        "publish",
        "complete",
        static_score=final["static_measurement_complete_score"],
        probability_score=final["probability_benefit_score"],
        decision_score=final["decision_benefit_score"],
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


def _argument_path(path: Path, root: Path) -> Path:
    """Resolve a CLI evidence path against the authenticated worktree."""

    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run measurement or one zero-new-call fresh reader."""

    args = parse_args(argv)
    root = REPO_ROOT.resolve()
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        errors = cold_replay(_argument_path(args.cold_replay, root), root=root)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(_argument_path(args.independent_reduce, root))
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date)
    return 0
