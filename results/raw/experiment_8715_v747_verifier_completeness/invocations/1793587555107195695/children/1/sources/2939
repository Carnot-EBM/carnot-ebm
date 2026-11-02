"""Audit V651 static and online decision evidence as separate branches.

The audit reads authenticated producer files and recomputes their numeric
claims from row evidence. It never treats a human annotation as formal truth.
Spec refs: REQ-REPORT-7428 and SCENARIO-REPORT-7428-*.
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
from carnot.experiment_7412_v650_source_features import (
    SOURCE_FEATURE_NAMES,
    extract_feature_row,
    gibbs_energy,
    probability_from_energy,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_FIELDS,
    EVALUATOR_TOKEN,
    PREDICTOR_FIELDS,
    reload_corpus,
)
from carnot.experiment_7425_v651_spline_prototype import dense_design_vector
from carnot.experiment_7426_v651_static_decisions import load_detail_shards
from carnot.experiment_7427_v651_randomized_feedback import (
    _load_event_rows,
    _load_lineage_rows,
    load_initial_states,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
    write_immutable_sidecar,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7428-v651-decision-audit"
SCHEMA = "carnot.exp7428.v651.decision_audit.v1"
LABEL_AUTHORITY = "human_annotation_source_support"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7428_v651_decision_audit.json")
RAW_DIR = Path("results/raw/experiment_7428_v651_decision_audit")
MODULE_PATH = Path("python/carnot/experiment_7428_v651_decision_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7428_v651_decision_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7428_v651_decision_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
PARITY_TOLERANCE = 1e-10
ONLINE_RATE = 0.01
GRADIENT_CLIP = 1.0
BOOTSTRAP_SEED_STATIC = 6_510_426
BOOTSTRAP_SEED_ONLINE = 6_510_427
TRAINING_SEEDS = (65_101, 65_102, 65_103, 65_104, 65_105)
STATIC_ARMS = (
    "training_prevalence",
    "raw_l2_logistic",
    "gibbs_6_4_1",
    "sparse_spline_49",
    "dense_spline_logistic",
)
ONLINE_ARMS = (
    "frozen_spline",
    "online_sparse_spline",
    "online_gibbs",
    "online_raw_logistic",
    "no_feedback_spline",
)
LEARNER_ARMS = {
    "online_sparse_spline",
    "online_gibbs",
    "online_raw_logistic",
}
ONLINE_ORDERS = ("hash_order", "domain_blocked_shift_order")
ONLINE_DELAYS = (0, 8)
PRIMARY_SCHEDULE = "hybrid_four_plus_four"
PRIMARY_STATIC_ARM = "sparse_spline_49"
PRIMARY_STATIC_CONTROLS = ("raw_l2_logistic", "gibbs_6_4_1")
ALPHA_PER_POLICY_TEST = 0.05 / 90

PRODUCERS: dict[str, JsonDict] = {
    "annotated": {
        "path": Path("results/experiment_7423_v651_annotated_protocol.json"),
        "schema": "carnot.exp7423.v651.annotated_protocol.v1",
        "experiment_id": "exp7423-v651-annotated-protocol",
        "completion_field": "annotated_protocol_ready_score",
        "sha256": "sha256:4d91bd186aa7e0fd320694cc0fbc066a2c84ed85c823beccba47c357f903ba8e",
        "branches": ("static", "online"),
    },
    "spline": {
        "path": Path("results/experiment_7425_v651_spline_prototype.json"),
        "schema": "carnot.exp7425.v651.spline_prototype.v1",
        "experiment_id": "exp7425-spline-prototype",
        "completion_field": "spline_prototype_ready_score",
        "sha256": "sha256:362874817bd332e64e54ddd527b581070ffba06c59ef021c6c104febfff47bce",
        "branches": ("static",),
    },
    "static": {
        "path": Path("results/experiment_7426_v651_static_decisions.json"),
        "schema": "carnot.exp7426.v651.static_decisions.v1",
        "experiment_id": "exp7426-v651-static-decisions",
        "completion_field": "decision_capture_complete_score",
        "sha256": "sha256:82405ac1229cd49a3cb4bc700b4cd978d423c166b51da23f4e4af6b1e8ec7c36",
        "branches": ("static", "online"),
    },
    "online": {
        "path": Path("results/experiment_7427_v651_randomized_feedback.json"),
        "schema": "carnot.exp7427.v651.randomized_feedback.v1",
        "experiment_id": "exp7427-v651-randomized-feedback",
        "completion_field": "online_capture_complete_score",
        "sha256": "sha256:0fa3911cee634a2606cbe8b657f45b3f8bed4b88171a1248246fc2f236384555",
        "branches": ("online",),
    },
}

REQUIRED_ATTACKS = (
    "mislabeled_implicit_true",
    "cross_split_source_alias",
    "response_sibling_cross_split",
    "changed_annotation_offset",
    "annotation_field_in_predictor",
    "future_label_access",
    "propensity_substitution",
    "duplicate_update",
    "hidden_full_feedback_access",
    "omitted_persistence_time",
    "spline_capacity_claim",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "model_duration_s",
    "validation_duration_s",
    "cold_start_duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "static_audit_complete_score",
    "online_audit_complete_score",
    "branch_rows",
    "leakage_attack_rows",
    "small_ebm_training",
    "historical_small_ebm_training",
)

V651_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def binary_log_loss(label: int, probability: float) -> float:
    """Compute finite Bernoulli loss from a label and probability."""

    if label not in {0, 1} or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("binary label and finite probability are required")
    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def _normalized(value: Any) -> str:
    """Normalize text only enough to find exact source and response aliases."""

    return " ".join(str(value or "").casefold().split())


def static_integrity_errors(
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    *,
    raw_responses: Mapping[str, str] | None = None,
) -> list[str]:
    """Check view separation, component splits, labels, siblings, and offsets."""

    errors: list[str] = []
    evaluator_by_key = {str(row.get("row_key")): row for row in evaluators}
    denied = set(EVALUATOR_FIELDS) - set(PREDICTOR_FIELDS)
    source_partitions: dict[str, set[str]] = defaultdict(set)
    response_partitions: dict[str, set[str]] = defaultdict(set)
    group_partitions: dict[str, set[str]] = defaultdict(set)
    for predictor in predictors:
        key = str(predictor.get("row_key") or "")
        if denied.intersection(predictor):
            errors.append("annotation_field_in_predictor")
        partition = str(predictor.get("partition") or "")
        source_partitions[_normalized(predictor.get("source_text"))].add(partition)
        group_partitions[str(predictor.get("group_id") or "")].add(partition)
        evaluator = evaluator_by_key.get(key)
        if evaluator is None:
            errors.append(f"evaluator_row_missing:{key}")
            continue
        if any(
            evaluator.get(field) != predictor.get(field)
            for field in ("group_id", "partition", "task_type")
        ):
            errors.append(f"predictor_evaluator_identity_mismatch:{key}")
        annotations = evaluator.get("annotations")
        if not isinstance(annotations, list):
            errors.append(f"annotation_shape_invalid:{key}")
            continue
        response_id = str(evaluator.get("response_id") or key)
        response_partitions[response_id].add(partition)
        response = (
            str(raw_responses.get(response_id) or "")
            if raw_responses is not None
            else str(predictor.get("response_text") or "")
        )
        offsets_valid = True
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                offsets_valid = False
                continue
            start, end, text = (
                annotation.get("start"),
                annotation.get("end"),
                annotation.get("text"),
            )
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end < start
                or end > len(response)
                or response[start:end] != text
            ):
                offsets_valid = False
        if not offsets_valid:
            errors.append("changed_annotation_offset")
        primary = int(not annotations)
        sensitivity = int(
            not any(
                isinstance(annotation, Mapping) and not bool(annotation.get("implicit_true"))
                for annotation in annotations
            )
        )
        if (
            evaluator.get("primary_label") != primary
            or evaluator.get("implicit_true_excluded_label") != sensitivity
        ):
            errors.append("mislabeled_implicit_true")
    if set(evaluator_by_key) != {str(row.get("row_key")) for row in predictors}:
        errors.append("view_row_keys_mismatch")
    if any(len(partitions) > 1 for text, partitions in source_partitions.items() if text):
        errors.append("cross_split_source_alias")
    if any(len(partitions) > 1 for text, partitions in response_partitions.items() if text):
        errors.append("response_sibling_cross_split")
    if any(len(partitions) > 1 for group, partitions in group_partitions.items() if group):
        errors.append("cross_split_source_alias")
    return list(dict.fromkeys(errors))


def static_metric_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Recompute each stored proper-score contribution from its raw operands."""

    errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            label = int(row["label"])
            probability = float(row["probability"])
            brier = (probability - label) ** 2
            loss = binary_log_loss(label, probability)
            if not math.isclose(brier, float(row["brier_contribution"]), abs_tol=1e-12):
                errors.append(f"static_brier_contribution_mismatch:{index}")
            if not math.isclose(loss, float(row["log_loss_contribution"]), abs_tol=1e-12):
                errors.append(f"static_log_loss_contribution_mismatch:{index}")
            if row.get("action") not in {"accept", "reject", "escalate"}:
                errors.append(f"static_action_invalid:{index}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"static_metric_row_invalid:{index}")
    return errors


def reduce_static_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce Brier, log loss, action counts, and risks without producer code."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("arm"))].append(row)
    by_arm: JsonDict = {}
    for arm, selected in sorted(grouped.items()):
        accepts = [row for row in selected if row.get("action") == "accept"]
        rejects = [row for row in selected if row.get("action") == "reject"]
        actions = accepts + rejects
        by_domain: JsonDict = {}
        for task_type in sorted({str(row.get("task_type")) for row in selected}):
            domain = [row for row in selected if str(row.get("task_type")) == task_type]
            by_domain[task_type] = {
                "rows": len(domain),
                "groups": len({str(row.get("group_id")) for row in domain}),
                "label_counts": {
                    "0": sum(row.get("label") == 0 for row in domain),
                    "1": sum(row.get("label") == 1 for row in domain),
                },
                "brier": float(
                    np.mean(
                        [(float(row["probability"]) - int(row["label"])) ** 2 for row in domain]
                    )
                ),
                "log_loss": float(
                    np.mean(
                        [
                            binary_log_loss(int(row["label"]), float(row["probability"]))
                            for row in domain
                        ]
                    )
                ),
            }
        by_arm[arm] = {
            "rows": len(selected),
            "groups": len({str(row.get("group_id")) for row in selected}),
            "brier": float(
                np.mean([(float(row["probability"]) - int(row["label"])) ** 2 for row in selected])
            ),
            "log_loss": float(
                np.mean(
                    [
                        binary_log_loss(int(row["label"]), float(row["probability"]))
                        for row in selected
                    ]
                )
            ),
            "coverage": len(actions) / len(selected),
            "accept_count": len(accepts),
            "reject_count": len(rejects),
            "escalate_count": len(selected) - len(actions),
            "false_accept_risk": (
                sum(row.get("label") == 0 for row in accepts) / len(accepts) if accepts else None
            ),
            "false_reject_risk": (
                sum(row.get("label") == 1 for row in rejects) / len(rejects) if rejects else None
            ),
            "empirical_decision_risk": (
                sum(
                    (row.get("action") == "accept" and row.get("label") == 0)
                    or (row.get("action") == "reject" and row.get("label") == 1)
                    for row in actions
                )
                / len(actions)
                if actions
                else None
            ),
            "by_domain": by_domain,
        }
    return {"by_arm": by_arm}


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Evaluate the small exact binomial tail used by the confidence bound."""

    return sum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Compute a one-sided exact binomial upper limit independently."""

    if not 0 <= k <= n or n <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("valid binomial counts and alpha are required")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    low, high = k / n, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(k, n, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def spline_capacity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Keep dense/sparse parity inside the shared 49-parameter feature basis."""

    errors: list[str] = []
    for index, row in enumerate(rows):
        if row.get("parameter_count") != 49:
            errors.append(f"spline_parameter_count_mismatch:{index}")
        fields = ("sparse_dense_parameter_gap", "max_probability_gap", "max_gradient_gap")
        if not all(field in row for field in fields):
            errors.append(f"spline_parity_fields_missing:{index}")
            continue
        if any(float(row[field]) > PARITY_TOLERANCE for field in fields):
            errors.append(f"spline_parity_tolerance_exceeded:{index}")
    return errors


def _expected_propensity(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Reconstruct one block's marginal propensity from its selection design."""

    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row.get("observation_id")), row)
    block = list(unique.values())
    block_size = int(block[0].get("block_size", 0)) if block else 0
    schedule = str(block[0].get("schedule")) if block else ""
    if block_size <= 0 or len(block) != block_size:
        return {}
    if schedule == "top_risk_eight":
        return {
            str(row["observation_id"]): 1.0 if row.get("revealed") is True else 0.0 for row in block
        }
    revealed = sum(row.get("revealed") is True for row in block)
    if schedule == "uniform_eight":
        return {str(row["observation_id"]): revealed / block_size for row in block}
    if schedule == "hybrid_four_plus_four":
        deterministic = sum(row.get("selection_component") == "deterministic_top" for row in block)
        random_revealed = sum(
            row.get("revealed") is True and row.get("selection_component") == "uniform_remainder"
            for row in block
        )
        remainder = block_size - deterministic
        return {
            str(row["observation_id"]): (
                1.0
                if row.get("selection_component") == "deterministic_top"
                else random_revealed / remainder
            )
            for row in block
        }
    return {}


def online_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check shared masks, propensity, temporal order, and one-update semantics."""

    errors: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    schedules: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    shared: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for index, row in enumerate(rows):
        key = (
            row.get("ordering"),
            row.get("schedule"),
            row.get("delay"),
            row.get("seed"),
            row.get("arm"),
            row.get("observation_id"),
        )
        if key in seen:
            errors.append(f"duplicate_update:{index}")
        seen.add(key)
        schedule_key = (
            row.get("ordering"),
            row.get("schedule"),
            row.get("delay"),
            row.get("block_index"),
        )
        schedules[schedule_key].append(row)
        shared_key = (*schedule_key, row.get("observation_id"))
        shared_value = (
            row.get("revealed"),
            row.get("propensity"),
            row.get("selection_component"),
        )
        if shared_key in shared and shared[shared_key] != shared_value:
            errors.append(f"propensity_substitution:{index}")
        shared[shared_key] = shared_value
        try:
            prediction_index = int(row["prediction_index"])
            available_at = int(row["available_at"])
            delay = int(row["delay"])
            if available_at != prediction_index + delay or available_at < prediction_index:
                errors.append(f"future_label_access:{index}")
            arrival = row.get("delayed_arrival_index")
            if arrival is not None and int(arrival) < available_at:
                errors.append(f"future_label_access:{index}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"future_label_access:{index}")
        if row.get("prediction_before_feedback") is not True:
            errors.append(f"future_label_access:{index}")
        duration = row.get("journal_duration_s")
        if (
            row.get("prediction_persisted") is not True
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            errors.append(f"omitted_persistence_time:{index}")
        if row.get("update_admitted") is True and row.get("revealed") is not True:
            errors.append(f"hidden_full_feedback_access:{index}")
        if row.get("update_admitted") is True and row.get("arm") not in LEARNER_ARMS:
            errors.append(f"hidden_full_feedback_access:{index}")
        if row.get("update_admitted") is True and row.get("commit_after_prediction") is not True:
            errors.append(f"future_label_access:{index}")
        if row.get("update_admitted") is not True and row.get("new_state_hash") != row.get(
            "state_hash_at_prediction"
        ):
            errors.append(f"state_changed_without_update:{index}")
    for schedule_key, block_rows in schedules.items():
        expected = _expected_propensity(block_rows)
        if not expected:
            errors.append(f"propensity_block_invalid:{schedule_key}")
            continue
        for row in block_rows:
            identity = str(row.get("observation_id"))
            try:
                matches = math.isclose(
                    float(row["propensity"]), expected[identity], rel_tol=0.0, abs_tol=1e-12
                )
            except (KeyError, TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(f"propensity_substitution:{identity}")
                break
    return list(dict.fromkeys(errors))


def reduce_online_metrics(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute full and inverse-propensity proper scores for each condition."""

    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("label") in {0, 1}:
            grouped[
                (
                    str(row.get("ordering")),
                    str(row.get("schedule")),
                    int(row.get("delay", 0)),
                    str(row.get("arm")),
                )
            ].append(row)
    reports: list[JsonDict] = []
    for (ordering, schedule, delay, arm), selected in sorted(grouped.items()):
        groups = len({str(row.get("observation_id")) for row in selected})
        seeds = len({int(row.get("seed", 0)) for row in selected})
        weighted_brier = 0.0
        weighted_log = 0.0
        for row in selected:
            propensity = float(row.get("propensity", 0.0))
            if row.get("revealed") is True and propensity > 0.0:
                weighted_brier += (float(row["probability"]) - int(row["label"])) ** 2 / propensity
                weighted_log += (
                    binary_log_loss(int(row["label"]), float(row["probability"])) / propensity
                )
        action_rows = [row for row in selected if row.get("action") != "escalate"]
        harmful = [
            row
            for row in action_rows
            if (row.get("action") == "accept" and row.get("label") == 0)
            or (row.get("action") == "reject" and row.get("label") == 1)
        ]
        revealed_groups = len(
            {str(row.get("observation_id")) for row in selected if row.get("revealed") is True}
        )
        denominator = max(groups * seeds, 1)
        reports.append(
            {
                "ordering": ordering,
                "schedule": schedule,
                "delay": delay,
                "arm": arm,
                "scored_rows": len(selected),
                "independent_groups": groups,
                "revealed_groups": revealed_groups,
                "reveal_cost": revealed_groups / max(groups, 1),
                "full_brier": float(
                    np.mean(
                        [(float(row["probability"]) - int(row["label"])) ** 2 for row in selected]
                    )
                ),
                "full_log_loss": float(
                    np.mean(
                        [
                            binary_log_loss(int(row["label"]), float(row["probability"]))
                            for row in selected
                        ]
                    )
                ),
                "ipw_brier": weighted_brier / denominator,
                "ipw_log_loss": weighted_log / denominator,
                "coverage": len(action_rows) / max(len(selected), 1),
                "observed_action_risk": len(harmful) / len(action_rows) if action_rows else 0.0,
                "biased_zero_probability_omissions": any(
                    row.get("revealed") is not True and float(row.get("propensity", 0.0)) == 0.0
                    for row in selected
                ),
                "iid_guarantee_asserted": False,
                "conformal_guarantee_asserted": False,
                "fdr_guarantee_asserted": False,
            }
        )
    return reports


def _sigmoid(value: float) -> float:
    """Evaluate a stable scalar sigmoid."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _feature_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Read the six frozen numeric features in protocol order."""

    vector = np.asarray([features[name] for name in SOURCE_FEATURE_NAMES], dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("six finite features are required")
    return vector


def _raw_probability(arm: str, checkpoint: Mapping[str, Any], vector: np.ndarray) -> float:
    """Score one online checkpoint without calling the producer learner."""

    if arm == "online_raw_logistic":
        return _sigmoid(
            float(vector @ np.asarray(checkpoint["coef"], dtype=np.float64))
            + float(checkpoint["bias"])
        )
    if arm == "online_gibbs":
        return probability_from_energy(gibbs_energy(checkpoint, vector))
    knots = np.asarray(checkpoint["knots"], dtype=np.float64)
    design = np.asarray(dense_design_vector(vector, knots)[0], dtype=np.float64)
    return _sigmoid(
        float(design @ np.asarray(checkpoint["coef"], dtype=np.float64).reshape(-1))
        + float(checkpoint["bias"])
    )


def _calibrated_probability(
    arm: str,
    checkpoint: Mapping[str, Any],
    calibration: Mapping[str, Any],
    vector: np.ndarray,
) -> float:
    """Apply the immutable affine calibration to an independently scored head."""

    raw = min(max(_raw_probability(arm, checkpoint, vector), 1e-15), 1.0 - 1e-15)
    logit = math.log(raw) - math.log1p(-raw)
    affine = calibration["affine"]
    calibrated_logit = np.asarray(
        [float(affine["slope"]) * logit + float(affine["intercept"])],
        dtype=np.float64,
    )
    if calibrated_logit[0] >= 0.0:
        return float(1.0 / (1.0 + np.exp(-calibrated_logit))[0])
    exponential = np.exp(calibrated_logit)
    return float((exponential / (1.0 + exponential))[0])


def reapply_numeric_update(
    arm: str,
    checkpoint: Mapping[str, Any],
    calibration: Mapping[str, Any],
    features: Mapping[str, Any],
    label: int,
) -> tuple[JsonDict, JsonDict]:
    """Apply one clipped numeric update without using the producer learner."""

    if arm not in LEARNER_ARMS or label not in {0, 1}:
        raise ValueError("registered learner arm and binary label are required")
    state = deepcopy(dict(checkpoint))
    vector = _feature_vector(features)
    probability = _calibrated_probability(arm, state, calibration, vector)
    scalar = float(calibration["affine"]["slope"]) * (probability - label)
    if arm == "online_raw_logistic":
        gradients = {"coef": scalar * vector, "bias": np.asarray(scalar)}
    elif arm == "online_sparse_spline":
        knots = np.asarray(state["knots"], dtype=np.float64)
        design = np.asarray(dense_design_vector(vector, knots)[0], dtype=np.float64)
        gradients = {"coef": (scalar * design).reshape(6, 8), "bias": np.asarray(scalar)}
    else:
        w1 = np.asarray(state["w1"], dtype=np.float64)
        b1 = np.asarray(state["b1"], dtype=np.float64)
        w_out = np.asarray(state["w_out"], dtype=np.float64)
        hidden_linear = w1 @ vector + b1
        sigmoid = np.asarray([_sigmoid(float(value)) for value in hidden_linear])
        hidden = hidden_linear * sigmoid
        derivative = sigmoid + hidden_linear * sigmoid * (1.0 - sigmoid)
        hidden_gradient = scalar * w_out * derivative
        gradients = {
            "w1": np.outer(hidden_gradient, vector),
            "b1": hidden_gradient,
            "w_out": scalar * hidden,
            "b_out": np.asarray(scalar),
        }
    norm_before = math.sqrt(
        sum(float(np.sum(np.asarray(value, dtype=np.float64) ** 2)) for value in gradients.values())
    )
    scale = min(1.0, GRADIENT_CLIP / norm_before) if norm_before else 1.0
    touched = 0
    change_sq = 0.0
    for key, gradient in gradients.items():
        old = np.asarray(state[key], dtype=np.float64)
        delta = ONLINE_RATE * scale * np.asarray(gradient, dtype=np.float64)
        new = old - delta
        state[key] = float(new) if new.ndim == 0 else new.tolist()
        touched += int(np.count_nonzero(delta))
        change_sq += float(np.sum(delta**2))
    return state, {
        "gradient_norm_before_clip": norm_before,
        "gradient_norm_after_clip": norm_before * scale,
        "coefficient_change_l2": math.sqrt(change_sq),
        "touched_coefficients": touched,
        "pre_update_probability": probability,
    }


def numeric_state_hash(arm: str, seed: int, checkpoint: Mapping[str, Any], count: int) -> str:
    """Bind one numeric state to its arm, seed, checkpoint, and update count."""

    return canonical_hash(
        {"arm": arm, "seed": seed, "checkpoint": checkpoint, "update_count": count}
    )


def replay_trusted_journal(
    arm: str,
    checkpoint: Mapping[str, Any],
    calibration: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Cold replay only active trusted events from the immutable initial state."""

    state = deepcopy(dict(checkpoint))
    for event in events:
        if event.get("active") is True:
            state, _receipt = reapply_numeric_update(
                arm,
                state,
                calibration,
                event["features"],
                int(event["label"]),
            )
    return state


def lineage_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_states: Mapping[tuple[int, str], Mapping[str, Any]],
    calibrations: Mapping[tuple[int, str], Mapping[str, Any]],
    features_by_observation: Mapping[str, Mapping[str, Any]],
    heartbeat: Any = None,
) -> list[str]:
    """Reapply every update and verify parent, event, checkpoint, and state hashes."""

    errors: list[str] = []
    states: dict[tuple[Any, ...], JsonDict] = {}
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    last_heartbeat = time.monotonic()
    for index, row in enumerate(rows):
        try:
            arm, seed = str(row["arm"]), int(row["seed"])
            unit = (
                row.get("ordering"),
                row.get("schedule"),
                int(row.get("delay", 0)),
                seed,
                arm,
            )
            base_key = (seed, arm)
            if unit not in states:
                states[unit] = deepcopy(dict(initial_states[base_key]))
            state = states[unit]
            count = counts[unit]
            expected_parent = numeric_state_hash(arm, seed, state, count)
            if row.get("parent_state_hash") != expected_parent:
                errors.append(f"lineage_parent_hash_mismatch:{index}")
            identity = str(row["observation_id"])
            updated, _receipt = reapply_numeric_update(
                arm,
                state,
                calibrations[base_key],
                features_by_observation[identity],
                int(row["label"]),
            )
            expected_count = count + 1
            expected_state = numeric_state_hash(arm, seed, updated, expected_count)
            if row.get("state_hash") != expected_state:
                errors.append(f"lineage_state_hash_mismatch:{index}")
            if row.get("update_count") != expected_count:
                errors.append(f"lineage_update_count_mismatch:{index}")
            if canonical_hash(row.get("checkpoint")) != canonical_hash(updated):
                errors.append(f"lineage_checkpoint_mismatch:{index}")
            if "event_hash" in row:
                expected_event = canonical_hash(
                    {
                        "observation_id": identity,
                        "label": int(row["label"]),
                        "authority": row.get("label_authority"),
                        "arrival_index": int(row["arrival_index"]),
                    }
                )
                if row.get("event_hash") != expected_event:
                    errors.append(f"lineage_event_hash_mismatch:{index}")
            states[unit], counts[unit] = updated, expected_count
        except (KeyError, TypeError, ValueError, IndexError):
            errors.append(f"lineage_row_invalid:{index}")
        if heartbeat is not None and time.monotonic() - last_heartbeat >= 60.0:
            heartbeat(index + 1)
            last_heartbeat = time.monotonic()
    return errors


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while treating malformed external bytes as absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    branch: str,
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Keep every prerequisite operand explicit, including absent and zero values."""

    return {
        "branch": branch,
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _producer_validation_passed(value: Mapping[str, Any]) -> bool:
    """Require the producer's fixed eight affected checks or its exact summary."""

    summary = value.get("gate_check_summary") or {}
    if (
        summary.get("required_checks_passed") is True
        or summary.get("all_required_gates_passed") is True
    ):
        return True
    names = {
        str(row.get("name"))
        for row in value.get("validation_receipts") or []
        if isinstance(row, Mapping)
        and row.get("required") is True
        and row.get("passed") is True
        and row.get("exit_code") == 0
    }
    return set(validation_scope.REQUIRED_CHECK_NAMES).issubset(names)


def _historical_git_hash_exists(
    root: Path, relative: Path, expected: str
) -> bool:  # pragma: no cover
    """Accept old tracked source bytes without rewriting frozen producer evidence."""

    found = subprocess.run(
        ["git", "log", "--format=%H", "--", relative.as_posix()],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if found.returncode != 0:
        return False
    for commit in found.stdout.splitlines()[:64]:
        shown = subprocess.run(
            ["git", "show", f"{commit}:{relative.as_posix()}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if shown.returncode == 0:
            observed = "sha256:" + __import__("hashlib").sha256(shown.stdout).hexdigest()
            if observed == expected:
                return True
    return False


def collect_preconditions(
    repo_root: Path, *, verify_sources: bool = True
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:
    """Authenticate all four producer identities independently for each branch."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    producers: dict[str, JsonDict] = {}
    for name, definition in PRODUCERS.items():
        relative = definition["path"]
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        producer = _load_object(path) if present else {}
        producers[name] = producer
        observed_hash = sha256_file(path) if producer else None
        if producer:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": observed_hash,
                "original_flagged_adversarial": producer.get("flagged_adversarial"),
            }
        for branch in definition["branches"]:
            identity = str(definition["experiment_id"])
            checks.append(
                _precondition(
                    branch,
                    f"{name}_producer_bytes",
                    identity,
                    relative.as_posix(),
                    "bytes",
                    "readable_nonempty_bytes",
                    "readable_nonempty_bytes" if producer else None,
                )
            )
            if not producer:
                continue
            for field, expected in (
                ("sha256", definition["sha256"]),
                ("schema", definition["schema"]),
                ("experiment_id", identity),
                ("milestone", MILESTONE),
                (str(definition["completion_field"]), 1),
                ("flagged_adversarial", False),
            ):
                observed = observed_hash if field == "sha256" else producer.get(field)
                checks.append(
                    _precondition(
                        branch,
                        f"{name}_{field}",
                        identity,
                        relative.as_posix(),
                        field,
                        expected,
                        observed,
                    )
                )
            verdict = producer.get("verdict_class")
            checks.append(
                _precondition(
                    branch,
                    f"{name}_verdict_eligible",
                    identity,
                    relative.as_posix(),
                    "verdict_class",
                    ["positive", "circular_positive", "null"],
                    verdict,
                    passed=verdict in {"positive", "circular_positive", "null"},
                )
            )
            checks.append(
                _precondition(
                    branch,
                    f"{name}_required_validation",
                    identity,
                    relative.as_posix(),
                    "validation_receipts.required_checks",
                    True,
                    _producer_validation_passed(producer),
                )
            )
        if verify_sources and producer:
            for label, reference in (producer.get("source_artifact_hashes") or {}).items():
                if not isinstance(reference, Mapping):
                    continue
                if isinstance(reference.get("files"), list):
                    for file_row in reference["files"]:
                        cache_path = Path(str(file_row.get("cache_path") or ""))
                        expected_hash = str(file_row.get("sha256") or "")
                        observed_hash = sha256_file(cache_path) if cache_path.is_file() else None
                        for branch in definition["branches"]:
                            checks.append(
                                _precondition(
                                    branch,
                                    f"{name}_source_hash:{label}:{file_row.get('path')}",
                                    str(definition["experiment_id"]),
                                    str(cache_path),
                                    "sha256",
                                    expected_hash,
                                    observed_hash,
                                )
                            )
                    continue
                source_label = str(reference.get("path") or label)
                source = Path(source_label)
                source_path = source if source.is_absolute() else root / source
                expected_hash = str(reference.get("sha256") or "")
                current_hash = sha256_file(source_path) if source_path.is_file() else None
                passed = current_hash == expected_hash
                authenticated_from = "current_bytes"
                if not passed and not source.is_absolute():
                    passed = _historical_git_hash_exists(root, source, expected_hash)
                    authenticated_from = "tracked_historical_bytes" if passed else "unavailable"
                for branch in definition["branches"]:
                    row = _precondition(
                        branch,
                        f"{name}_source_hash:{label}",
                        str(definition["experiment_id"]),
                        source_label,
                        "sha256",
                        expected_hash,
                        current_hash,
                        passed=passed,
                    )
                    row["authenticated_from"] = authenticated_from
                    checks.append(row)
    spec = root / SPEC_PATH
    spec_text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    for branch in ("static", "online"):
        checks.append(
            _precondition(
                branch,
                "driving_requirement",
                SPEC_PATH.as_posix(),
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-REPORT-7428",
                "REQ-REPORT-7428" if "REQ-REPORT-7428" in spec_text else None,
            )
        )
    return checks, hashes, producers


def blocked_branch(branch: str, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Represent external absence without converting it to retryable own work."""

    failed = next((dict(row) for row in preconditions if row.get("passed") is not True), {})
    return {
        "branch": branch,
        "upstream": f"exp742{'6' if branch == 'static' else '7'}-v651",
        "available": False,
        "eligible": False,
        "valid": False,
        "complete": False,
        "value": False,
        "producer_verdict_class": None,
        "label_authority": LABEL_AUTHORITY,
        "reduced_metrics": {},
        "errors": ["external_producer_unavailable"],
        "leakage_attack_rows": [],
        "gate_check_summary": {
            "blocked_upstream": failed.get("upstream"),
            "blocked_path": failed.get("path"),
            "blocked_check": failed.get("check"),
            "blocked_field": failed.get("field"),
            "blocked_expected": failed.get("expected"),
            "blocked_observed": failed.get("observed"),
        },
    }


def classify_terminal(
    branches: Sequence[Mapping[str, Any]],
    *,
    required_validation_passed: bool,
    own_work_complete: bool,
) -> JsonDict:
    """Keep invalid, unavailable, owned-incomplete, value, and valid-null distinct."""

    by_name = {str(row.get("branch")): row for row in branches}
    static = by_name.get("static", {})
    online = by_name.get("online", {})
    static_score = int(
        static.get("available") is True
        and static.get("eligible") is True
        and static.get("valid") is True
        and static.get("complete") is True
    )
    online_score = int(
        online.get("available") is True
        and online.get("eligible") is True
        and online.get("valid") is True
        and online.get("complete") is True
    )
    if any(row.get("available") is True and row.get("valid") is not True for row in branches):
        verdict, honest = "disqualified", "complete_disqualified_invalid_branch_evidence"
    elif not required_validation_passed:
        verdict, honest = "disqualified", "complete_disqualified_required_validation"
    elif any(row.get("available") is not True for row in branches):
        if static_score:
            honest = "blocked_online_external_producer_unavailable_static_audit_preserved"
        elif online_score:
            honest = "blocked_static_external_producer_unavailable_online_audit_preserved"
        else:
            honest = "blocked_external_producer_unavailable"
        verdict = "blocked"
    elif not own_work_complete:
        verdict, honest = "partial", "partial_retryable_owned_audit_work_unfinished"
    elif any(row.get("value") is True for row in branches):
        verdict, honest = "positive", "complete_positive_independent_decision_audit"
    else:
        verdict = "null"
        honest = "complete_null_static_and_online_audits_reproduce_no_registered_benefit"
    return {
        "honest_verdict": honest,
        "verdict_class": verdict,
        "status": honest,
        "static_audit_complete_score": static_score,
        "online_audit_complete_score": online_score,
    }


def _field_principles() -> dict[str, str]:
    """Explain each required field without wrapping ordinary scalar values."""

    principles = {
        field: "This field records measured Exp7428 audit evidence." for field in REQUIRED_FIELDS
    }
    principles.update(
        {
            "schema": "Use a versioned plain top-level schema with terminal identity.",
            "run_date": "Use 20260919 with real UTC and monotonic boundaries.",
            "preconditions_checked": "Name each resource, field, expected value, and observation.",
            "MODEL_SPECS": "List current LLMs; this aggregation has none.",
            "model_invoked": "True only when this run attempts current model use.",
            "invocation_counts": "Count only owned current model loads and generations.",
            "inference_substrate": "Describe current computation as a plain string.",
            "inference_substrate_class": "Classify current aggregation, not historical fitting.",
            "execution_venue": "Use the closed host venue string.",
            "duration_s": "Measure current work and separate validation and cold-start time.",
            "phase_spans": "Record real phase boundaries and completed units.",
            "random_seed": "Freeze fitting and resampling seeds.",
            "reproducibility_checksum": "Bind code, protocol, inputs, raw evidence, and validation scope.",
            "source_artifact_hashes": "Preserve exact input identity and original producer flags.",
            "rows": "Keep every producer comparative unit and disposition.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
            "acceptance_gate_results": "Keep validity checks separate from scientific benefit.",
            "gate_check_summary": "Name exact failed operands without changing missing into zero.",
            "verifier_is_oracle": "Human source-support annotations are fallible and not formal truth.",
            "honest_verdict": "Completed findings start complete_; unavailable prerequisites start blocked_.",
            "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
            "flagged_adversarial": "Preserve critical findings and never use flagged evidence for readiness.",
            "validation_receipts": "Retain exact argv, environment, exit, duration, name, and log hash.",
            "field_principles": "Explain field intent separately from scalar values.",
            "promotion_score": "Always zero; this audit cannot promote or update generator weights.",
            "static_audit_complete_score": "One only for an eligible independent static reconstruction.",
            "online_audit_complete_score": "One only for an eligible independent online reconstruction.",
            "branch_rows": "Keep availability, eligibility, validity, completion, value, metrics, and blockers per branch.",
            "leakage_attack_rows": "Retain one targeted counterexample for each safety property.",
        }
    )
    return principles


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable audit evidence while excluding clocks and the checksum slot."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "model_duration_s",
        "validation_duration_s",
        "cold_start_duration_s",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "phase_spans",
        "current_run_id",
        "current_owner_pid",
    }
    return canonical_hash({key: value[key] for key in sorted(value) if key not in excluded})


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, terminal: bool) -> bool:
    """Require every frozen affected check and, for final records, four readers."""

    required = set(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        required.update(
            {
                "cold_artifact_replay",
                "independent_branch_recompute",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            }
        )
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True and row.get("passed") is True and row.get("exit_code") == 0
    }
    return required.issubset(passed)


def _gate(check: str, category: str, expected: Any, observed: Any) -> JsonDict:
    """Build one plain gate row with validity separate from benefit."""

    return {
        "check": check,
        "category": category,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": "Validity and branch completion do not imply scientific benefit.",
    }


def _build_artifact(
    static: Mapping[str, Any],
    online: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    terminal: bool,
) -> JsonDict:
    """Build one schema-complete record from two independent branch results."""

    branches = [deepcopy(dict(static)), deepcopy(dict(online))]
    validation_ok = _validation_passed(validation_receipts, terminal=terminal)
    classification = classify_terminal(
        branches,
        required_validation_passed=validation_ok,
        own_work_complete=True,
    )
    attacks = [
        deepcopy(dict(row))
        for branch in branches
        for row in branch.get("leakage_attack_rows") or []
    ]
    producer_rows = [
        {"branch": branch["branch"], **deepcopy(dict(row))}
        for branch in branches
        for row in branch.get("unit_rows") or []
    ]
    if not producer_rows:
        producer_rows = [
            {
                "branch": branch["branch"],
                "comparative_unit": f"{branch['branch']}-audit",
                "attempted": branch.get("available") is True,
                "completed": branch.get("complete") is True,
                "failed": branch.get("available") is True and branch.get("valid") is not True,
                "censored": False,
                "unstarted": branch.get("available") is not True,
                "status": "completed" if branch.get("complete") is True else "blocked",
            }
            for branch in branches
        ]
    gates = [
        _gate("static_branch_valid", "validity", True, static.get("valid") is True),
        _gate("online_branch_valid", "validity", True, online.get("valid") is True),
        _gate("required_validation", "completion", True, validation_ok),
        _gate(
            "all_targeted_attacks_rejected",
            "safety",
            True,
            bool(attacks) and all(row.get("passed") is True for row in attacks),
        ),
        _gate("promotion_disabled", "promotion", 0, 0),
    ]
    failed_precondition = next(
        (dict(row) for row in preconditions if row.get("passed") is not True), None
    )
    failed_gate = next((row for row in gates if row["passed"] is not True), None)
    summary = {
        "required_checks_passed": validation_ok,
        "blocked_upstream": (failed_precondition or {}).get("upstream"),
        "blocked_path": (failed_precondition or {}).get("path"),
        "blocked_check": (failed_precondition or {}).get("check")
        or (failed_gate or {}).get("check"),
        "blocked_field": (failed_precondition or {}).get("field"),
        "blocked_expected": (failed_precondition or {}).get("expected")
        if failed_precondition
        else (failed_gate or {}).get("expected"),
        "blocked_observed": (failed_precondition or {}).get("observed")
        if failed_precondition
        else (failed_gate or {}).get("observed"),
    }
    result: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": producer_rows,
        "sample_size_budget": {
            "planned": len(producer_rows),
            "attempted": sum(row.get("attempted") is True for row in producer_rows),
            "completed": sum(row.get("completed") is True for row in producer_rows),
            "failed": sum(row.get("failed") is True for row in producer_rows),
            "censored": sum(row.get("censored") is True for row in producer_rows),
            "unstarted": sum(row.get("unstarted") is True for row in producer_rows),
            "independent_groups": {
                "static": static.get("independent_groups", 0),
                "online": online.get("independent_groups", 0),
            },
            "stop_rule": "audit all authenticated rows; external absence blocks only that branch",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "flagged_adversarial": any(row.get("flagged_adversarial") is True for row in branches),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "branch_rows": branches,
        "leakage_attack_rows": attacks,
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "static_bootstrap_seed": BOOTSTRAP_SEED_STATIC,
            "online_bootstrap_seed": BOOTSTRAP_SEED_ONLINE,
        },
        "model_duration_s": 0.0,
        "validation_duration_s": sum(
            float(row.get("duration_s", 0.0)) for row in validation_receipts
        ),
        "cold_start_duration_s": 0.0,
        "historical_small_ebm_training": {
            "static": deepcopy(static.get("historical_small_ebm_training")),
            "online": deepcopy(online.get("historical_small_ebm_training")),
        },
        **classification,
        **deepcopy(dict(current_receipt)),
    }
    result["small_ebm_training"] = {
        "performed": False,
        "receipt_class": "small_ebm_training",
        "current_llm_calls": 0,
    }
    result["reproducibility_checksum"] = reproducibility_checksum(result)
    return result


def _fixture_receipts() -> list[JsonDict]:
    """Make closed passing receipts only for compact artifact unit tests."""

    names = [
        *validation_scope.REQUIRED_CHECK_NAMES,
        "cold_artifact_replay",
        "independent_branch_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "duration_s": 0.0,
            "command_argv": ["fixture"],
            "command_environment": {},
            "log_sha256": canonical_hash(name),
        }
        for name in names
    ]


def build_artifact_for_test(static: Mapping[str, Any], online: Mapping[str, Any]) -> JsonDict:
    """Build a compact terminal artifact for mutation and cold-reader tests."""

    receipt = build_current_work_receipt(
        run_id="exp7428-fixture",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
        inference_substrate_details={"device": "fixture", "software": "python_numpy"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000_000,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    return _build_artifact(
        static,
        online,
        preconditions=[],
        source_hashes={},
        validation_receipts=_fixture_receipts(),
        current_receipt=receipt,
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        terminal=True,
    )


def validate_artifact(
    value: Any,
    *,
    root: Path = REPO_ROOT,
    verify_source_bytes: bool = True,
    require_terminal: bool = True,
) -> list[str]:
    """Cold-check identity, branch scores, attacks, provenance, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("identity_mismatch")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
    ):
        errors.append("model_declaration_mismatch")
    errors.extend(validate_current_work_receipt(value, events=[], root=root))
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_mismatch")
    branches = value.get("branch_rows") or []
    classification = classify_terminal(
        branches,
        required_validation_passed=_validation_passed(
            value.get("validation_receipts") or [], terminal=require_terminal
        ),
        own_work_complete=True,
    )
    if any(value.get(field) != classification[field] for field in classification):
        if value.get("verdict_class") != classification["verdict_class"]:
            errors.append("terminal_classification_mismatch")
        if (
            value.get("static_audit_complete_score")
            != classification["static_audit_complete_score"]
            or value.get("online_audit_complete_score")
            != classification["online_audit_complete_score"]
        ):
            errors.append("branch_score_mismatch")
    attacks = value.get("leakage_attack_rows") or []
    if {row.get("attack") for row in attacks if isinstance(row, Mapping)} != set(
        REQUIRED_ATTACKS
    ) or not all(row.get("passed") is True for row in attacks if isinstance(row, Mapping)):
        errors.append("leakage_attack_rows_incomplete")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_FIELDS).issubset(principles):
        errors.append("field_principles_incomplete")
    if value.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_mismatch")
    if verify_source_bytes:
        for label, reference in (value.get("source_artifact_hashes") or {}).items():
            if not isinstance(reference, Mapping):
                errors.append(f"source_hash_row_invalid:{label}")
                continue
            path = Path(str(reference.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != reference.get("sha256"):
                errors.append(f"source_hash_mismatch:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(
    path: Path,
    *,
    verify_source_bytes: bool = True,
    require_terminal: bool = True,
) -> list[str]:
    """Reload a candidate in a fresh process and apply strict field checks."""

    value = _load_object(path)
    return (
        validate_artifact(
            value,
            verify_source_bytes=verify_source_bytes,
            require_terminal=require_terminal,
        )
        if value
        else ["artifact_unreadable_or_not_object"]
    )


def build_validation_commands(
    repo_root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    """Freeze the exact Exp7358 affected command plan for this module."""

    return build_command_plan(repo_root, V651_MANIFEST, private_root)


def _nested_close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:  # pragma: no cover
    """Compare nested numeric evidence without hiding nulls or key drift."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
        )
    return left == right


def _static_grouped_intervals(
    rows: Sequence[Mapping[str, Any]], draws: int
) -> JsonDict:  # pragma: no cover - exercised by the capability E2E.
    """Recompute every static contrast from connected source-group draws."""

    groups = sorted({str(row.get("group_id")) for row in rows})
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for arm in STATIC_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        values = {"brier": [], "log_loss": [], "coverage": []}
        for group in groups:
            selected = [row for row in arm_rows if str(row.get("group_id")) == group]
            values["brier"].append(
                float(
                    np.mean(
                        [(float(row["probability"]) - int(row["label"])) ** 2 for row in selected]
                    )
                )
            )
            values["log_loss"].append(
                float(
                    np.mean(
                        [
                            binary_log_loss(int(row["label"]), float(row["probability"]))
                            for row in selected
                        ]
                    )
                )
            )
            values["coverage"].append(
                float(np.mean([row.get("action") != "escalate" for row in selected]))
            )
        vectors[arm] = {key: np.asarray(item, dtype=np.float64) for key, item in values.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED_STATIC)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    raw: dict[str, dict[str, np.ndarray]] = {}
    for control in PRIMARY_STATIC_CONTROLS:
        raw[control] = {}
        for metric in ("brier", "log_loss", "coverage"):
            delta = vectors[PRIMARY_STATIC_ARM][metric] - vectors[control][metric]
            raw[control][metric] = np.mean(delta[indices], axis=1)
    centered = np.column_stack(
        [
            raw[control]["brier"] - np.mean(raw[control]["brier"])
            for control in PRIMARY_STATIC_CONTROLS
        ]
    )
    upper_critical = float(np.quantile(np.max(centered, axis=1), 0.95))
    contrasts: JsonDict = {}
    for control in PRIMARY_STATIC_CONTROLS:
        observed = float(np.mean(vectors[PRIMARY_STATIC_ARM]["brier"] - vectors[control]["brier"]))
        contrasts[control] = {
            "brier_delta": {
                "mean": observed,
                "simultaneous_upper_95": observed + upper_critical,
                "marginal_ci95": [
                    float(np.quantile(raw[control]["brier"], 0.025)),
                    float(np.quantile(raw[control]["brier"], 0.975)),
                ],
            },
            "log_loss_delta": {
                "mean": float(np.mean(raw[control]["log_loss"])),
                "ci95": [
                    float(np.quantile(raw[control]["log_loss"], 0.025)),
                    float(np.quantile(raw[control]["log_loss"], 0.975)),
                ],
            },
            "coverage_delta": {
                "mean": float(np.mean(raw[control]["coverage"])),
                "ci95": [
                    float(np.quantile(raw[control]["coverage"], 0.025)),
                    float(np.quantile(raw[control]["coverage"], 0.975)),
                ],
            },
        }
    return {
        "draws": draws,
        "seed": BOOTSTRAP_SEED_STATIC,
        "resampling_unit": "connected_source_group",
        "effective_groups": len(groups),
        "seed_average_before_resampling": True,
        "primary_contrasts": list(PRIMARY_STATIC_CONTROLS),
        "simultaneous_correction_count": len(PRIMARY_STATIC_CONTROLS),
        "simultaneous_method": "max_centered_statistic_two_primary_brier_contrasts",
        "upper_critical_value": upper_critical,
        "contrasts": contrasts,
    }


def _features_from_source(row: Mapping[str, Any], source_text: str) -> JsonDict:  # pragma: no cover
    """Rebuild the six source proxies without reading an evaluator label."""

    projected = extract_feature_row(
        {
            "row_key": row["row_key"],
            "group_id": row["group_id"],
            "partition": row["partition"],
            "question": "",
            "context": source_text,
            "answer": row.get("response_text", ""),
            "sentence": row.get("response_text", ""),
        }
    )
    return deepcopy(projected["source_features"])


def _condition_static_predictors(
    predictors: Sequence[Mapping[str, Any]], condition: str
) -> list[JsonDict]:  # pragma: no cover
    """Apply a registered source condition while evaluator fields stay unavailable."""

    copied = [deepcopy(dict(row)) for row in predictors]
    if condition == "full_source":
        return copied
    if condition == "source_masked":
        for row in copied:
            row["source_text"] = ""
            row["features"] = _features_from_source(row, "")
        return copied
    if condition != "source_swapped":
        raise ValueError("registered source condition required")
    cohorts: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in copied:
        cohorts[(str(row["partition"]), str(row["task_type"]))].append(row)
    for cohort in cohorts.values():
        groups = sorted({str(row["group_id"]) for row in cohort})
        representatives = {
            group: min(
                (row for row in cohort if str(row["group_id"]) == group),
                key=lambda item: str(item["row_key"]),
            )
            for group in groups
        }
        sources = {group: str(representatives[group]["source_text"]) for group in groups}
        donors = {group: groups[(index + 1) % len(groups)] for index, group in enumerate(groups)}
        for row in cohort:
            source = sources[donors[str(row["group_id"])]]
            row["source_text"] = source
            row["features"] = _features_from_source(row, source)
    return copied


def _static_raw_probability(
    arm: str, checkpoint: Mapping[str, Any], vector: np.ndarray
) -> float:  # pragma: no cover
    """Score a frozen static checkpoint before ensemble calibration."""

    if arm == "training_prevalence":
        return float(checkpoint["probability"])
    if arm == "raw_l2_logistic":
        return _sigmoid(
            float(vector @ np.asarray(checkpoint["coef"], dtype=np.float64))
            + float(checkpoint["bias"])
        )
    if arm == "gibbs_6_4_1":
        return probability_from_energy(gibbs_energy(checkpoint, vector))
    knots = np.asarray(checkpoint["knots"], dtype=np.float64)
    design = np.asarray(dense_design_vector(vector, knots)[0], dtype=np.float64)
    return _sigmoid(
        float(design @ np.asarray(checkpoint["coef"], dtype=np.float64).reshape(-1))
        + float(checkpoint["bias"])
    )


def _apply_static_calibration(
    probability: float, calibration: Mapping[str, Any]
) -> float:  # pragma: no cover
    """Apply the frozen affine calibrator with the producer's float64 order."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    logit = math.log(clipped) - math.log1p(-clipped)
    affine = calibration["affine"]
    value = np.asarray(
        [float(affine["slope"]) * logit + float(affine["intercept"])],
        dtype=np.float64,
    )
    if value[0] >= 0.0:
        return float((1.0 / (1.0 + np.exp(-value)))[0])
    exponential = np.exp(value)
    return float((exponential / (1.0 + exponential))[0])


def _static_confidence_bounds(
    root: Path,
    producer: Mapping[str, Any],
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[str]]:  # pragma: no cover - capability E2E.
    """Score policy rows and recompute every simultaneous exact confidence bound."""

    evaluator_by_key = {str(row["row_key"]): row for row in evaluators}
    payloads: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for manifest in producer.get("checkpoint_manifest") or []:
        if not isinstance(manifest, Mapping):
            continue
        path = root / str(manifest.get("path"))
        if not path.is_file() or sha256_file(path) != manifest.get("sha256"):
            continue
        payload = _load_object(path)
        payloads[(str(manifest["condition"]), str(manifest["arm"]))].append(payload)
    stored = {
        (str(row["condition"]), str(row["arm"])): row
        for row in producer.get("policy_certificates") or []
    }
    output: list[JsonDict] = []
    errors: list[str] = []
    policy_predictors = [row for row in predictors if row.get("partition") == "policy_calibration"]
    for condition in ("full_source", "source_masked", "source_swapped"):
        conditioned = _condition_static_predictors(policy_predictors, condition)
        representatives: dict[str, JsonDict] = {}
        for predictor in conditioned:
            group = str(predictor["group_id"])
            if group not in representatives or str(predictor["row_key"]) < str(
                representatives[group]["row_key"]
            ):
                evaluator = evaluator_by_key[str(predictor["row_key"])]
                representatives[group] = {
                    **predictor,
                    "label": int(evaluator["primary_label"]),
                }
        rows = [representatives[group] for group in sorted(representatives)]
        vectors = [_feature_vector(row["features"]) for row in rows]
        for arm in STATIC_ARMS:
            states = sorted(payloads[(condition, arm)], key=lambda row: int(row["seed"]))
            if len(states) != len(TRAINING_SEEDS):
                errors.append(f"static_checkpoint_set_missing:{condition}:{arm}")
                continue
            probabilities = [
                _apply_static_calibration(
                    float(
                        np.mean(
                            [
                                _static_raw_probability(arm, state["checkpoint"], vector)
                                for state in states
                            ]
                        )
                    ),
                    states[0]["calibration"],
                )
                for vector in vectors
            ]
            certificate = stored.get((condition, arm), {})
            candidates = certificate.get("candidates") or []
            for candidate_index, candidate in enumerate(candidates):
                accept = [
                    row
                    for row, probability in zip(rows, probabilities, strict=True)
                    if probability >= float(candidate["accept_threshold"])
                ]
                reject = [
                    row
                    for row, probability in zip(rows, probabilities, strict=True)
                    if probability <= float(candidate["reject_threshold"])
                ]
                for action, selected, harmful_label in (
                    ("accept", accept, 0),
                    ("reject", reject, 1),
                ):
                    harmful = sum(row["label"] == harmful_label for row in selected)
                    upper = (
                        clopper_pearson_upper(harmful, len(selected), ALPHA_PER_POLICY_TEST)
                        if selected
                        else None
                    )
                    stored_bound = candidate[f"{action}_certificate"]
                    passed = (
                        stored_bound.get("selected_groups") == len(selected)
                        and stored_bound.get("harmful_outcomes") == harmful
                        and _nested_close(stored_bound.get("upper_risk_bound"), upper)
                    )
                    output.append(
                        {
                            "condition": condition,
                            "arm": arm,
                            "candidate_index": candidate_index,
                            "action": action,
                            "selected_groups": len(selected),
                            "harmful_outcomes": harmful,
                            "upper_risk_bound": upper,
                            "passed": passed,
                        }
                    )
                    if not passed:
                        errors.append(
                            f"static_confidence_bound_mismatch:{condition}:{arm}:{candidate_index}:{action}"
                        )
    return output, errors


def _paired_online_intervals(
    rows: Sequence[Mapping[str, Any]], draws: int
) -> list[JsonDict]:  # pragma: no cover - exercised by the capability E2E.
    """Recompute all registered moving-block intervals from raw event rows."""

    output: list[JsonDict] = []
    controls = ("frozen_spline", "online_raw_logistic")
    for order_index, ordering in enumerate(ONLINE_ORDERS):
        for delay in ONLINE_DELAYS:
            for control_index, control in enumerate(controls):
                grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        row.get("ordering") == ordering
                        and row.get("schedule") == PRIMARY_SCHEDULE
                        and row.get("delay") == delay
                        and row.get("arm") in {"online_sparse_spline", control}
                    ):
                        propensity = float(row.get("propensity", 0.0))
                        contribution = (
                            (float(row["probability"]) - int(row["label"])) ** 2 / propensity
                            if row.get("revealed") is True and propensity > 0.0
                            else 0.0
                        )
                        grouped[
                            (
                                int(row["prediction_index"]),
                                str(row["observation_id"]),
                                str(row["arm"]),
                            )
                        ].append(contribution)
                units = sorted({(index, identity) for index, identity, _arm in grouped})
                vector = np.asarray(
                    [
                        np.mean(grouped[(index, identity, "online_sparse_spline")])
                        - np.mean(grouped[(index, identity, control)])
                        for index, identity in units
                    ],
                    dtype=np.float64,
                )
                for block_length in (32, 64):
                    rng = np.random.default_rng(
                        BOOTSTRAP_SEED_ONLINE
                        + order_index * 100
                        + delay * 10
                        + control_index
                        + block_length
                    )
                    block_count = math.ceil(len(vector) / block_length)
                    estimates = np.empty(draws, dtype=np.float64)
                    offsets = np.arange(block_length)
                    for draw in range(draws):
                        starts = rng.integers(0, len(vector), size=block_count)
                        indices = ((starts[:, None] + offsets) % len(vector)).reshape(-1)[
                            : len(vector)
                        ]
                        estimates[draw] = float(np.mean(vector[indices]))
                    output.append(
                        {
                            "ordering": ordering,
                            "schedule": PRIMARY_SCHEDULE,
                            "delay": delay,
                            "control_arm": control,
                            "block_length": block_length,
                            "draws": draws,
                            "seed": BOOTSTRAP_SEED_ONLINE,
                            "simultaneous_comparison_count": 8,
                            "point_delta": float(np.mean(vector)),
                            "upper_brier_delta": float(np.quantile(estimates, 1.0 - 0.05 / 8)),
                            "empirical_replay_only": True,
                        }
                    )
    return output


def _static_attack_rows(
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    raw_responses: Mapping[str, str] | None = None,
) -> list[JsonDict]:  # pragma: no cover - exercised by the capability E2E.
    """Run one private targeted mutation for every static safety property."""

    by_key = {str(row["row_key"]): row for row in predictors}
    annotated = next(row for row in evaluators if row.get("annotations"))
    first_predictor = deepcopy(dict(by_key[str(annotated["row_key"])]))
    first_evaluator = deepcopy(dict(annotated))
    if not any(bool(row.get("implicit_true")) for row in first_evaluator["annotations"]):
        first_evaluator["annotations"][0]["implicit_true"] = True
        first_evaluator["implicit_true_excluded_label"] = 1
    partitions: dict[str, Mapping[str, Any]] = {}
    for row in predictors:
        partitions.setdefault(str(row["partition"]), row)
    cross = list(partitions.values())[:2]
    cases: list[tuple[str, list[JsonDict], list[JsonDict]]] = []
    changed = deepcopy(first_evaluator)
    changed["implicit_true_excluded_label"] = 0
    cases.append(("mislabeled_implicit_true", [first_predictor], [changed]))
    left, right = deepcopy(dict(cross[0])), deepcopy(dict(cross[1]))
    right["source_text"] = left["source_text"]
    cases.append(
        (
            "cross_split_source_alias",
            [left, right],
            [
                deepcopy(dict(evaluator_by))
                for evaluator_by in evaluators
                if evaluator_by["row_key"] in {left["row_key"], right["row_key"]}
            ],
        )
    )
    left, right = deepcopy(dict(cross[0])), deepcopy(dict(cross[1]))
    response_evaluators = [
        deepcopy(dict(evaluator_by))
        for evaluator_by in evaluators
        if evaluator_by["row_key"] in {left["row_key"], right["row_key"]}
    ]
    response_evaluators[1]["response_id"] = response_evaluators[0]["response_id"]
    cases.append(
        (
            "response_sibling_cross_split",
            [left, right],
            response_evaluators,
        )
    )
    changed = deepcopy(first_evaluator)
    changed["annotations"][0]["start"] = int(changed["annotations"][0]["start"]) + 1
    cases.append(("changed_annotation_offset", [first_predictor], [changed]))
    changed_predictor = deepcopy(first_predictor)
    changed_predictor["primary_label"] = 0
    cases.append(("annotation_field_in_predictor", [changed_predictor], [first_evaluator]))
    rows = [
        {
            "attack": name,
            "target": "private_mutation",
            "observation": errors,
            "passed": name in errors,
        }
        for name, predictor_rows, evaluator_rows in cases
        for errors in [
            static_integrity_errors(
                predictor_rows,
                evaluator_rows,
                raw_responses=raw_responses,
            )
        ]
    ]
    changed_parity = deepcopy(dict(parity_rows[0]))
    changed_parity.update(
        {
            "parameter_count": 50,
            "sparse_dense_parameter_gap": changed_parity.pop("max_parameter_gap", 0.0),
        }
    )
    capacity_errors = spline_capacity_errors([changed_parity])
    rows.append(
        {
            "attack": "spline_capacity_claim",
            "target": "shared_49_parameter_basis",
            "observation": capacity_errors,
            "passed": any(
                error.startswith("spline_parameter_count_mismatch") for error in capacity_errors
            ),
        }
    )
    return rows


def _online_attack_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover - exercised by the capability E2E.
    """Run one targeted mutation for every online safety property."""

    targets = {
        "future_label_access": next(
            index for index, row in enumerate(rows) if row.get("revealed") is True
        ),
        "propensity_substitution": next(
            index for index, row in enumerate(rows) if 0.0 < float(row.get("propensity", 0.0)) < 1.0
        ),
        "hidden_full_feedback_access": next(
            index for index, row in enumerate(rows) if row.get("revealed") is not True
        ),
        "omitted_persistence_time": 0,
    }
    output: list[JsonDict] = []
    for attack, target in targets.items():
        changed = list(rows)
        mutation = deepcopy(dict(rows[target]))
        if attack == "future_label_access":
            mutation["available_at"] = int(mutation["prediction_index"]) - 1
        elif attack == "propensity_substitution":
            mutation["propensity"] = float(mutation["propensity"]) / 2.0
        elif attack == "hidden_full_feedback_access":
            mutation["update_admitted"] = True
        else:
            mutation["journal_duration_s"] = None
        changed[target] = mutation
        errors = online_integrity_errors(changed)
        output.append(
            {
                "attack": attack,
                "target": f"raw_event_row:{target}",
                "observation": errors[:8],
                "passed": any(error.startswith(attack) for error in errors),
            }
        )
    duplicate = list(rows)
    duplicate.append(deepcopy(dict(rows[0])))
    errors = online_integrity_errors(duplicate)
    output.append(
        {
            "attack": "duplicate_update",
            "target": "raw_event_identity",
            "observation": errors[:8],
            "passed": any(error.startswith("duplicate_update") for error in errors),
        }
    )
    return output


def _observation_features(
    corpus: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:  # pragma: no cover
    """Rebuild label-blind online observation identities from sealed predictors."""

    representatives: dict[str, Mapping[str, Any]] = {}
    for row in corpus["predictors"]:
        if (
            row.get("partition") != "prospective_stream"
            or row.get("certificate_selected") is not True
        ):
            continue
        group = str(row["group_id"])
        if group not in representatives or str(row["row_key"]) < str(
            representatives[group]["row_key"]
        ):
            representatives[group] = row
    return {
        canonical_hash(
            {
                "group_id": group,
                "row_key": row["row_key"],
                "task_type": row["task_type"],
            }
        ): row["features"]
        for group, row in representatives.items()
    }


def _revocation_errors(
    root: Path,
    states: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:  # pragma: no cover
    """Cold replay replacement, erasure, and rollback from immutable initial state."""

    del root
    initial = next(
        row
        for row in states
        if row["seed"] == TRAINING_SEEDS[0] and row["arm"] == "online_raw_logistic"
    )
    features = {name: 0.4 for name in SOURCE_FEATURE_NAMES}
    replaced = replay_trusted_journal(
        "online_raw_logistic",
        initial["checkpoint"],
        initial["calibration"],
        [{"features": features, "label": 0, "active": True}],
    )
    original = replay_trusted_journal(
        "online_raw_logistic",
        initial["checkpoint"],
        initial["calibration"],
        [{"features": features, "label": 1, "active": True}],
    )
    expected = {
        "replace_label": numeric_state_hash("online_raw_logistic", TRAINING_SEEDS[0], replaced, 1),
        "erase_feedback": numeric_state_hash(
            "online_raw_logistic", TRAINING_SEEDS[0], initial["checkpoint"], 0
        ),
        "rollback_corrupted_state": numeric_state_hash(
            "online_raw_logistic", TRAINING_SEEDS[0], original, 1
        ),
    }
    errors: list[str] = []
    by_operation = {str(row.get("operation")): row for row in rows}
    for operation, state_hash in expected.items():
        row = by_operation.get(operation, {})
        if (
            row.get("trusted_journal_replayed") is not True
            or row.get("state_hash_after") != state_hash
        ):
            errors.append(f"revocation_replay_mismatch:{operation}")
    return errors


def audit_static_branch(
    root: Path,
    protocol: Mapping[str, Any],
    producer: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover - exercised by the capability E2E.
    """Independently reduce the complete static branch from sealed raw rows."""

    if not producer:
        return blocked_branch("static", preconditions)
    corpus = reload_corpus(root / CORPUS_DIR)
    predictors = corpus["predictors"]
    evaluators = corpus["evaluators"]
    ragtruth = (protocol.get("source_artifact_hashes") or {}).get("ragtruth") or {}
    response_receipt = next(
        (row for row in ragtruth.get("files") or [] if row.get("path") == "dataset/response.jsonl"),
        {},
    )
    raw_responses: dict[str, str] = {}
    response_path = Path(str(response_receipt.get("cache_path") or ""))
    if response_path.is_file():
        with response_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                raw_responses[str(row.get("id"))] = str(row.get("response") or "")
    errors = static_integrity_errors(
        predictors,
        evaluators,
        raw_responses=raw_responses if raw_responses else None,
    )
    metric_dir = root / str(producer["metric_row_directory"])
    metric_rows = load_detail_shards(metric_dir, producer["metric_row_shards"])
    errors.extend(static_metric_errors(metric_rows))
    full_rows = [row for row in metric_rows if row.get("condition") == "full_source"]
    metrics = reduce_static_metrics(full_rows)
    if not _nested_close(metrics, producer.get("calibration_metrics")):
        errors.append("static_metric_reduction_mismatch")
    stored_intervals = producer.get("grouped_intervals") or {}
    draws = int(stored_intervals.get("draws", 0))
    intervals = _static_grouped_intervals(full_rows, draws)
    if not _nested_close(intervals, stored_intervals):
        errors.append("static_interval_reduction_mismatch")
    parity_input = [
        {
            **dict(row),
            "parameter_count": 49,
            "sparse_dense_parameter_gap": row.get("max_parameter_gap"),
        }
        for row in producer.get("spline_parity_rows") or []
    ]
    errors.extend(spline_capacity_errors(parity_input))
    confidence_bounds, confidence_errors = _static_confidence_bounds(
        root, producer, predictors, evaluators
    )
    errors.extend(confidence_errors)
    attacks = _static_attack_rows(
        predictors,
        evaluators,
        producer["spline_parity_rows"],
        raw_responses if raw_responses else None,
    )
    if not all(row["passed"] for row in attacks):
        errors.append("static_targeted_attack_failed")
    eligible = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    valid = eligible and not errors and producer.get("flagged_adversarial") is False
    partition_rows = {
        partition: {
            "rows": sum(row.get("partition") == partition for row in predictors),
            "independent_groups": len(
                {row.get("group_id") for row in predictors if row.get("partition") == partition}
            ),
            "label_counts": {
                "0": sum(
                    row.get("partition") == partition and row.get("primary_label") == 0
                    for row in evaluators
                ),
                "1": sum(
                    row.get("partition") == partition and row.get("primary_label") == 1
                    for row in evaluators
                ),
            },
        }
        for partition in sorted({str(row.get("partition")) for row in predictors})
    }
    return {
        "branch": "static",
        "upstream": producer.get("experiment_id"),
        "available": True,
        "eligible": eligible,
        "valid": valid,
        "complete": valid,
        "value": valid and producer.get("decision_value_score") == 1,
        "producer_verdict_class": producer.get("verdict_class"),
        "producer_flagged_adversarial": producer.get("flagged_adversarial"),
        "flagged_adversarial": producer.get("flagged_adversarial"),
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible human source-support judgment; not formal truth",
        "reduced_metrics": metrics,
        "reduced_intervals": intervals,
        "confidence_bound_rows": confidence_bounds,
        "source_component_splits": partition_rows,
        "independent_groups": len({row.get("group_id") for row in predictors}),
        "raw_metric_row_count": len(metric_rows),
        "raw_evidence_hash": canonical_hash(
            {
                "corpus_manifest": corpus["manifest_hash"],
                "metric_shards": producer["metric_row_shards"],
                "probability_shards": producer["probability_row_shards"],
            }
        ),
        "unit_rows": deepcopy(producer.get("rows") or []),
        "historical_small_ebm_training": deepcopy(producer.get("small_ebm_training")),
        "leakage_attack_rows": attacks,
        "errors": list(dict.fromkeys(errors)),
        "protocol_ready_score": protocol.get("annotated_protocol_ready_score"),
    }


def audit_online_branch(
    root: Path,
    static_producer: Mapping[str, Any],
    producer: Mapping[str, Any],
    corpus: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    heartbeat: Any = None,
) -> JsonDict:  # pragma: no cover - exercised by the capability E2E.
    """Independently reduce schedules, updates, checkpoints, and intervals."""

    if not producer:
        return blocked_branch("online", preconditions)
    event_rows = _load_event_rows(producer, root)
    lineage_rows = _load_lineage_rows(producer, root)
    errors = online_integrity_errors(event_rows)
    reports = reduce_online_metrics(event_rows)
    if not _nested_close(reports, producer.get("condition_reports")):
        errors.append("online_metric_reduction_mismatch")
    stored_intervals = producer.get("paired_moving_block_intervals") or []
    draws = int(stored_intervals[0].get("draws", 0)) if stored_intervals else 0
    intervals = _paired_online_intervals(event_rows, draws)
    if not _nested_close(intervals, stored_intervals):
        errors.append("online_interval_reduction_mismatch")
    states = load_initial_states(root, static_producer)
    initial = {(int(row["seed"]), str(row["arm"])): row["checkpoint"] for row in states}
    calibrations = {(int(row["seed"]), str(row["arm"])): row["calibration"] for row in states}
    errors.extend(
        lineage_errors(
            lineage_rows,
            initial_states=initial,
            calibrations=calibrations,
            features_by_observation=_observation_features(corpus),
            heartbeat=heartbeat,
        )
    )
    errors.extend(_revocation_errors(root, states, producer.get("revocation_rows") or []))
    attacks = _online_attack_rows(event_rows)
    if not all(row["passed"] for row in attacks):
        errors.append("online_targeted_attack_failed")
    eligible = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    valid = eligible and not errors and producer.get("flagged_adversarial") is False
    return {
        "branch": "online",
        "upstream": producer.get("experiment_id"),
        "available": True,
        "eligible": eligible,
        "valid": valid,
        "complete": valid,
        "value": valid and producer.get("online_value_score") == 1,
        "producer_verdict_class": producer.get("verdict_class"),
        "producer_flagged_adversarial": producer.get("flagged_adversarial"),
        "flagged_adversarial": producer.get("flagged_adversarial"),
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible human source-support judgment; not formal truth",
        "reduced_metrics": reports,
        "reduced_intervals": intervals,
        "independent_groups": len({row.get("observation_id") for row in event_rows}),
        "raw_event_row_count": len(event_rows),
        "lineage_row_count": len(lineage_rows),
        "raw_evidence_hash": canonical_hash(
            {
                "events": producer["feedback_event_rows"],
                "lineage": producer["checkpoint_lineage"],
            }
        ),
        "unit_rows": deepcopy(producer.get("rows") or []),
        "historical_small_ebm_training": deepcopy(producer.get("small_ebm_training")),
        "leakage_attack_rows": attacks,
        "errors": list(dict.fromkeys(errors)),
    }


def _load_and_audit(
    root: Path, *, heartbeat: Any = None
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    """Load prerequisites once and finish each available branch independently."""

    checks, hashes, producers = collect_preconditions(root)
    by_branch = {
        branch: [row for row in checks if row.get("branch") == branch]
        for branch in ("static", "online")
    }
    static = (
        audit_static_branch(
            root,
            producers["annotated"],
            producers["static"],
            by_branch["static"],
        )
        if producers["static"] and producers["annotated"] and producers["spline"]
        else blocked_branch("static", by_branch["static"])
    )
    if producers["online"] and producers["static"] and producers["annotated"]:
        corpus = reload_corpus(root / CORPUS_DIR)
        online = audit_online_branch(
            root,
            producers["static"],
            producers["online"],
            corpus,
            by_branch["online"],
            heartbeat,
        )
    else:
        online = blocked_branch("online", by_branch["online"])
    return checks, hashes, producers, static, online


def independent_replay(path: Path, root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover
    """Freshly reduce producer rows and compare branch evidence to a candidate."""

    candidate = _load_object(path)
    errors = validate_artifact(candidate, root=root, require_terminal=False)
    _checks, _hashes, _producers, static, online = _load_and_audit(root)
    if not _nested_close([static, online], candidate.get("branch_rows") or []):
        errors.append("fresh_branch_reduction_mismatch")
    return list(dict.fromkeys(errors))


def _utc_now() -> str:  # pragma: no cover
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7428] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": _utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    commands = (
        (
            "cold_artifact_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--cold-replay",
                str(candidate),
                "--preterminal",
            ),
            "completion",
        ),
        (
            "independent_branch_recompute",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--independent-reduce",
                str(candidate),
            ),
            "completion",
        ),
        (
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "completion",
        ),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate", timeout_s=1200.0),
            category,
            True,
        )
        for name, argv, category in commands
    ]


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint.
    """Authenticate, reduce, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    _progress(started, "startup", "start")

    phase_started = time.monotonic()
    _progress(started, "branch_reduction", "before_benchmark")
    heartbeat = lambda units: _progress(  # noqa: E731
        started, "online_lineage", "heartbeat", completed_units=units
    )
    checks, hashes, producers, static, online = _load_and_audit(root, heartbeat=heartbeat)
    spans.append(_span("branch_reduction", phase_started, started, 2))
    _progress(
        started,
        "branch_reduction",
        "after_benchmark",
        static_valid=static["valid"],
        online_valid=online["valid"],
    )

    sidecar = write_immutable_sidecar(
        root / RAW_DIR / "historical_producer_receipts.json",
        scope="historical_model_receipts",
        payload={
            name: {
                "producer_sha256": hashes.get(definition["path"].as_posix(), {}).get("sha256"),
                "small_ebm_training": producers.get(name, {}).get("small_ebm_training"),
            }
            for name, definition in PRODUCERS.items()
        },
        root=root,
    )

    private = Path(tempfile.mkdtemp(prefix="exp7428-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private)
    plan_errors = validate_command_plan(root, V651_MANIFEST, commands)
    phase_started = time.monotonic()
    _progress(started, "affected_validation", "before_subprocesses", errors=len(plan_errors))
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, V651_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    _progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )

    ended_ns = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"exp7428-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
        inference_substrate_details={
            "device": platform.processor() or "cpu",
            "software": "python_numpy",
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=ended_ns - started_ns,
        sidecar_references=[sidecar],
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    candidate = _build_artifact(
        static,
        online,
        preconditions=checks,
        source_hashes=hashes,
        validation_receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in terminal),
    )

    final_ns = time.monotonic_ns()
    final_receipt = build_current_work_receipt(
        run_id=f"exp7428-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
        inference_substrate_details={
            "device": platform.processor() or "cpu",
            "software": "python_numpy",
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=final_ns - started_ns,
        sidecar_references=[sidecar],
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    final = _build_artifact(
        static,
        online,
        preconditions=checks,
        source_hashes=hashes,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        terminal=True,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin public audit and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--skip-source-bytes", action="store_true")
    parser.add_argument("--preterminal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or one requested fresh-process terminal reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(
            args.cold_replay,
            verify_source_bytes=not args.skip_source_bytes,
            require_terminal=not args.preterminal,
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        if args.skip_source_bytes:
            errors = cold_replay(args.independent_reduce, verify_source_bytes=False)
        else:
            errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
