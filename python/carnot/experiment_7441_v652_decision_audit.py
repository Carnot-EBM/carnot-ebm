"""Audit V652 static decisions and delayed learning from raw evidence.

The audit deliberately does not import producer verdict reducers. It reads the
producer rows, repeats the registered arithmetic, and keeps an invalid branch
separate from an available valid branch.

Spec refs: REQ-REPORT-7441 and SCENARIO-REPORT-7441-*.
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
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7441-v652-decision-audit"
SCHEMA = "carnot.exp7441.v652.decision_audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7441_v652_decision_audit.json")
RAW_DIR = Path("results/raw/experiment_7441_v652_decision_audit")
MODULE_PATH = Path("python/carnot/experiment_7441_v652_decision_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7441_v652_decision_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7441_v652_decision_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")

ALPHA_PER_CHECK = 0.05 / 9.0
STATIC_BOOTSTRAP_SEED = 6_520_439
ONLINE_BOOTSTRAP_SEED = 6_520_440
REGISTERED_REVEAL_PROPENSITIES = (0.25, 4.0 / 17.0)
EXPERT_NAMES = (
    "adaptive_gibbs",
    "adaptive_spline",
    "frozen_gibbs",
    "frozen_spline",
)
ORDERS = ("hash_order", "domain_blocked_shift_order")
DELAYS = (0, 8)
PRIMARY_COMPARATORS = (
    "frozen_spline",
    "adaptive_spline",
    "equal_weight_adaptive_mixture",
)
REQUIRED_MUTATIONS = (
    "role_leakage",
    "wrong_group_unit",
    "zero_selected_risk_coercion",
    "future_label",
    "duplicate_event",
    "revoked_label",
    "missing_expert_predictions",
)

PRODUCERS: dict[str, JsonDict] = {
    "selection": {
        "task_id": "exp7436-selection-protocol",
        "experiment_id": "exp7436-v652-selection-protocol",
        "completion_field": "selection_protocol_ready_score",
        "branches": ("static",),
        "allowed_classes": {"null", "positive"},
    },
    "prototype": {
        "task_id": "exp7438-mixture-prototype",
        "experiment_id": "exp7438-mixture-prototype",
        "completion_field": "mixture_prototype_ready_score",
        "branches": ("online",),
        "allowed_classes": {"circular_positive", "null", "positive"},
    },
    "static": {
        "task_id": "exp7439-certified-decisions",
        "experiment_id": "exp7439-v652-certified-decisions",
        "completion_field": "decision_capture_complete_score",
        "branches": ("static", "online"),
        "allowed_classes": {"null", "positive"},
    },
    "online": {
        "task_id": "exp7440-mixture-learning",
        "experiment_id": "exp7440-v652-mixture-learning",
        "completion_field": "online_capture_complete_score",
        "branches": ("online",),
        "allowed_classes": {"null", "positive"},
    },
}

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
    "computation_duration_s",
    "cold_start_duration_s",
    "validation_duration_s",
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
    "static_audit",
    "online_audit",
    "audit_mutation_rows",
    "audited_claim_rows",
)

V652_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def binary_log_loss(label: int, probability: float) -> float:
    """Compute finite Bernoulli loss from a raw label and probability."""

    if label not in {0, 1} or isinstance(label, bool):
        raise ValueError("binary label is required")
    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("finite probability in [0, 1] is required")
    clipped = min(max(numeric, 1e-6), 1.0 - 1e-6)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Evaluate the small exact binomial tail without a producer library."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.fsum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def _validate_binomial(k: int, n: int, alpha: float) -> None:
    if not (0 <= k <= n and n > 0 and 0.0 < alpha < 1.0):
        raise ValueError("valid binomial counts and alpha are required")


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Return a one-sided exact upper risk bound by monotone bisection."""

    _validate_binomial(k, n, alpha)
    if k == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _binomial_cdf(k, n, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """Return a one-sided exact lower success bound by outcome symmetry."""

    _validate_binomial(k, n, alpha)
    if k == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(n - k, n, alpha)


def policy_action(probability: float, policy: Mapping[str, Any]) -> str:
    """Apply a frozen two-threshold policy without consulting any labels."""

    if policy.get("accept_enabled") is True and probability >= float(policy["accept_threshold"]):
        return "accept"
    if policy.get("reject_enabled") is True and probability <= float(policy["reject_threshold"]):
        return "reject"
    return "escalate"


def static_integrity_errors(
    rows: Sequence[Mapping[str, Any]], certificates: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Reject role, group, alpha, empty-risk, and selector evidence drift."""

    errors: list[str] = []
    required = {"stage", "head", "policy_kind", "group_id", "row_key", "label", "probability"}
    seen: set[tuple[str, str, str, str]] = set()
    role_groups: dict[str, set[str]] = defaultdict(set)
    probabilities: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if not required.issubset(row):
            errors.append("static_required_field_missing")
            continue
        stage = str(row["stage"])
        head = str(row["head"])
        kind = str(row["policy_kind"])
        group = str(row["group_id"])
        key = (stage, head, kind, group)
        if key in seen:
            errors.append("duplicate_source_group")
        seen.add(key)
        role_groups[stage].add(group)
        try:
            label = int(row["label"])
            probability = float(row["probability"])
            binary_log_loss(label, probability)
        except (TypeError, ValueError):
            errors.append("static_operand_invalid")
            continue
        probabilities[(stage, head, group)][kind] = probability
    if role_groups.get("certification", set()) & role_groups.get("final_test", set()):
        errors.append("role_group_overlap")
    for variants in probabilities.values():
        if len(variants) > 1 and max(variants.values()) - min(variants.values()) > 1e-12:
            errors.append("selector_probability_mismatch")
    for row in certificates:
        policy = row.get("frozen_policy") or {}
        if (
            row.get("policy_kind") == "tuned"
            and policy.get("selection_partition") != "policy_tuning"
        ):
            errors.append("final_test_threshold_influence")
        certificate = row.get("certificate") or {}
        for name in ("accept_check", "reject_check", "coverage_check"):
            check = certificate.get(name) or {}
            allocated = check.get("alpha_allocated")
            if allocated is not None and not math.isclose(
                float(allocated), ALPHA_PER_CHECK, abs_tol=1e-15
            ):
                errors.append("alpha_allocation_mismatch")
        for name in ("accept_check", "reject_check"):
            check = certificate.get(name) or {}
            if check.get("selected_groups") == 0 and check.get("empirical_risk") is not None:
                errors.append("zero_selected_risk_coercion")
    return list(dict.fromkeys(errors))


def _action_check(
    actions: Sequence[str], labels: Sequence[int], action: str, *, enabled: bool, budget: float
) -> JsonDict:
    selected_labels = [
        label for label, observed in zip(labels, actions, strict=True) if observed == action
    ]
    harmful = sum(
        (action == "accept" and label == 0) or (action == "reject" and label == 1)
        for label in selected_labels
    )
    count = len(selected_labels)
    return {
        "action": action,
        "applicable": enabled,
        "selected_groups": count,
        "harmful_outcomes": harmful,
        "empirical_risk": harmful / count if count else None,
        "upper_risk_bound": (
            clopper_pearson_upper(harmful, count, ALPHA_PER_CHECK) if count and enabled else None
        ),
        "risk_budget": budget,
        "alpha_allocated": ALPHA_PER_CHECK,
        "alpha_spent": ALPHA_PER_CHECK if enabled else 0.0,
    }


def _certificate_reduction(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> JsonDict:
    labels = [int(row["label"]) for row in rows]
    actions = [policy_action(float(row["probability"]), policy) for row in rows]
    accept = _action_check(
        actions, labels, "accept", enabled=policy.get("accept_enabled") is True, budget=0.05
    )
    reject = _action_check(
        actions, labels, "reject", enabled=policy.get("reject_enabled") is True, budget=0.10
    )
    selected = sum(action != "escalate" for action in actions)
    total = len(actions)
    lower = clopper_pearson_lower(selected, total, ALPHA_PER_CHECK) if total else None
    coverage = {
        "selected_groups": selected,
        "total_groups": total,
        "empirical_coverage": selected / total if total else None,
        "lower_coverage_bound": lower,
        "coverage_floor": 0.25,
        "alpha_allocated": ALPHA_PER_CHECK,
        "alpha_spent": ALPHA_PER_CHECK,
    }
    accept_pass = not accept["applicable"] or (
        accept["upper_risk_bound"] is not None and float(accept["upper_risk_bound"]) <= 0.05
    )
    reject_pass = not reject["applicable"] or (
        reject["upper_risk_bound"] is not None and float(reject["upper_risk_bound"]) <= 0.10
    )
    return {
        "accept_check": accept,
        "reject_check": reject,
        "coverage_check": coverage,
        "alpha_allocation": ALPHA_PER_CHECK,
        "valid": bool(accept_pass and reject_pass and lower is not None and float(lower) >= 0.25),
    }


def _utility(action: str, label: int) -> int:
    if action == "accept":
        return 1 if label == 1 else -20
    if action == "reject":
        return 1 if label == 0 else -10
    return 0


def reduce_policy_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce per-source decisions, proper scores, risks, and registered utility."""

    if not rows:
        raise ValueError("policy metric rows must be non-empty")
    actions = [str(row.get("action") or "escalate") for row in rows]
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    utilities = [_utility(action, label) for action, label in zip(actions, labels, strict=True)]
    accept_labels = [
        label for action, label in zip(actions, labels, strict=True) if action == "accept"
    ]
    reject_labels = [
        label for action, label in zip(actions, labels, strict=True) if action == "reject"
    ]
    return {
        "row_count": len(rows),
        "accept_denominator": len(accept_labels),
        "accept_harmful": sum(label == 0 for label in accept_labels),
        "accept_harm": (
            sum(label == 0 for label in accept_labels) / len(accept_labels)
            if accept_labels
            else None
        ),
        "reject_denominator": len(reject_labels),
        "reject_harmful": sum(label == 1 for label in reject_labels),
        "reject_harm": (
            sum(label == 1 for label in reject_labels) / len(reject_labels)
            if reject_labels
            else None
        ),
        "coverage": sum(action != "escalate" for action in actions) / len(rows),
        "brier": float(
            np.mean(
                [
                    (probability - label) ** 2
                    for probability, label in zip(probabilities, labels, strict=True)
                ]
            )
        ),
        "log_loss": float(
            np.mean(
                [
                    binary_log_loss(label, probability)
                    for probability, label in zip(probabilities, labels, strict=True)
                ]
            )
        ),
        "registered_utility": float(np.mean(utilities)),
        "per_source_utility_rows": [
            {
                "group_id": str(row["group_id"]),
                "row_key": str(row["row_key"]),
                "action": action,
                "label": label,
                "utility": utility,
            }
            for row, action, label, utility in zip(rows, actions, labels, utilities, strict=True)
        ],
    }


def paired_coverage_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, JsonDict]:
    """Bootstrap two simultaneous source-group coverage differences."""

    if draws <= 0 or not rows:
        raise ValueError("positive draws and paired rows are required")
    groups = [str(row["group_id"]) for row in rows]
    if len(groups) != len(set(groups)):
        raise ValueError("paired bootstrap requires unique source groups")
    values = np.asarray(
        [
            (
                float(row["tuned_spline"]) - float(row["old_spline"]),
                float(row["tuned_spline"]) - float(row["tuned_logistic"]),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 2), dtype=np.float64)
    for start in range(0, draws, 1_000):
        stop = min(draws, start + 1_000)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = np.mean(values[indices], axis=1)
    names = ("tuned_spline_minus_old", "tuned_spline_minus_logistic")
    return {
        name: {
            "observed": float(np.mean(values[:, index])),
            "lower": float(np.quantile(samples[:, index], 0.0125)),
            "upper": float(np.quantile(samples[:, index], 0.9875)),
            "confidence": 0.95,
            "simultaneous_contrasts": 2,
            "tail_alpha": 0.0125,
            "draws": draws,
            "seed": seed,
            "paired_source_groups": len(values),
        }
        for index, name in enumerate(names)
    }


def reduce_static_evidence(
    rows: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = STATIC_BOOTSTRAP_SEED,
) -> JsonDict:
    """Recompute policy certificates, final metrics, and paired intervals."""

    policies = {(str(row["head"]), str(row["policy_kind"])): row for row in certificates}
    reduced_certificates: dict[str, JsonDict] = {}
    metrics: dict[str, JsonDict] = {}
    for (head, kind), policy_row in sorted(policies.items()):
        certification = [
            row
            for row in rows
            if row.get("stage") == "certification"
            and row.get("head") == head
            and row.get("policy_kind") == kind
        ]
        final = [
            row
            for row in rows
            if row.get("stage") == "final_test"
            and row.get("head") == head
            and row.get("policy_kind") == kind
        ]
        key = f"{head}:{kind}"
        reduced_certificates[key] = _certificate_reduction(
            certification, policy_row["frozen_policy"]
        )
        metrics[key] = reduce_policy_metrics(final)
    indexed = {
        (str(row["head"]), str(row["policy_kind"]), str(row["group_id"])): row
        for row in rows
        if row.get("stage") == "final_test"
    }
    groups = sorted(
        group for head, kind, group in indexed if head == "sparse_spline_49" and kind == "tuned"
    )
    paired = [
        {
            "group_id": group,
            "tuned_spline": int(
                indexed[("sparse_spline_49", "tuned", group)].get("action") != "escalate"
            ),
            "old_spline": int(
                indexed[("sparse_spline_49", "old_fixed", group)].get("action") != "escalate"
            ),
            "tuned_logistic": int(
                indexed[("raw_l2_logistic", "tuned", group)].get("action") != "escalate"
            ),
        }
        for group in groups
    ]
    return {
        "certificates": reduced_certificates,
        "metrics": metrics,
        "paired_coverage_intervals": paired_coverage_intervals(paired, draws=draws, seed=seed),
        "alpha_family": {
            "familywise_alpha": 0.05,
            "policies": 3,
            "checks_per_policy": 3,
            "alpha_per_check": ALPHA_PER_CHECK,
        },
    }


def replay_weight_update(
    weights: Mapping[str, Any], losses: Mapping[str, Any], *, fixed_share: float = 0.01
) -> dict[str, float]:
    """Rebuild one fixed-share update from stored prediction-time losses."""

    if set(weights) != set(EXPERT_NAMES) or set(losses) != set(EXPERT_NAMES):
        raise ValueError("complete expert weights and losses are required")
    numeric_weights = {name: float(weights[name]) for name in EXPERT_NAMES}
    if any(value <= 0.0 or not math.isfinite(value) for value in numeric_weights.values()):
        raise ValueError("complete expert weights must be positive and finite")
    log_posteriors = {
        name: math.log(numeric_weights[name]) - float(losses[name]) for name in EXPERT_NAMES
    }
    maximum = max(log_posteriors.values())
    raw = {name: math.exp(value - maximum) for name, value in log_posteriors.items()}
    denominator = math.fsum(raw.values())
    return {
        name: (1.0 - fixed_share) * raw[name] / denominator + fixed_share / 4.0
        for name in EXPERT_NAMES
    }


def _online_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row.get("ordering")),
        str(row.get("arm")),
        int(row.get("delay", -1)),
        int(row.get("seed", -1)),
        str(row.get("observation_id")),
    )


def online_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check causal order, source units, action claims, and replay operands."""

    errors: list[str] = []
    predictions = [row for row in rows if row.get("row_type") == "prediction"]
    feedback = [row for row in rows if row.get("row_type") == "feedback_event"]
    weights = [row for row in rows if row.get("row_type") == "weight_trajectory"]
    lineage = [row for row in rows if row.get("row_type") == "checkpoint_lineage"]
    prediction_by_key: dict[tuple[str, str, int, int, str], Mapping[str, Any]] = {}
    seen_group_units: set[tuple[str, str, int, int, str]] = set()
    required_prediction = {
        "observation_id",
        "group_id",
        "arm",
        "ordering",
        "delay",
        "seed",
        "prediction_index",
        "probability",
        "label",
        "propensity",
    }
    for row in predictions:
        if not required_prediction.issubset(row):
            errors.append("online_required_field_missing")
            continue
        key = _online_key(row)
        if key in prediction_by_key:
            errors.append("duplicate_prediction_event")
        prediction_by_key[key] = row
        group_unit = (*key[:4], str(row.get("group_id")))
        if group_unit in seen_group_units:
            errors.append("wrong_group_unit")
        seen_group_units.add(group_unit)
        if not any(
            math.isclose(float(row.get("propensity", -1.0)), expected, abs_tol=1e-12)
            for expected in REGISTERED_REVEAL_PROPENSITIES
        ):
            errors.append("nonuniform_reveal_propensity")
        if row.get("label_read_at_prediction") is not False:
            errors.append("future_label_access")
        if row.get("prediction_before_feedback") is not True:
            errors.append("future_label_access")
        if row.get("shadow_only") is not True or row.get("certified_safe") is not False:
            errors.append("adaptive_action_not_shadow_only")
        if row.get("deployed_action") != "escalate":
            errors.append("adaptive_action_not_shadow_only")
        try:
            if not math.isclose(
                float(row.get("loss")),
                binary_log_loss(int(row["label"]), float(row["probability"])),
                abs_tol=1e-12,
            ):
                errors.append("prediction_loss_mismatch")
            if not math.isclose(
                float(row.get("brier")),
                (float(row["probability"]) - int(row["label"])) ** 2,
                abs_tol=1e-12,
            ):
                errors.append("prediction_brier_mismatch")
        except (TypeError, ValueError):
            errors.append("prediction_metric_invalid")
    weight_keys = {_online_key(row) for row in weights}
    feedback_counts = Counter(_online_key(row) for row in feedback)
    for key, count in feedback_counts.items():
        if count > 1:
            errors.append("duplicate_feedback_event")
    for row in feedback:
        key = _online_key(row)
        prediction = prediction_by_key.get(key)
        if prediction is None:
            errors.append("feedback_without_prediction")
            continue
        if int(row.get("arrival_index", -1)) < int(row.get("prediction_index", 0)):
            errors.append("future_label_access")
        if row.get("update_count") != 1 or row.get("probability_source") != "stored_at_prediction":
            errors.append("feedback_update_semantics_invalid")
        if row.get("revoked") is True:
            errors.append("revoked_label_applied")
        if key in weight_keys and not isinstance(row.get("expert_prediction_time_losses"), Mapping):
            errors.append("missing_expert_predictions")
        try:
            expected_loss = binary_log_loss(
                int(row["feedback_label"]), float(prediction["probability"])
            )
            if not math.isclose(
                float(row.get("prediction_time_loss")), expected_loss, abs_tol=1e-12
            ):
                errors.append("stored_prediction_loss_mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append("stored_prediction_loss_invalid")
    weight_counts = Counter(_online_key(row) for row in weights)
    if any(count > 1 for count in weight_counts.values()):
        errors.append("duplicate_weight_update")
    lineage_counts = Counter(_online_key(row) for row in lineage)
    if any(count > 1 for count in lineage_counts.values()):
        errors.append("duplicate_lineage_event")
    feedback_by_key = {_online_key(row): row for row in feedback}
    for row in lineage:
        feedback_row = feedback_by_key.get(_online_key(row))
        if feedback_row is None:
            errors.append("lineage_without_feedback")
        elif any(
            row.get(field) != feedback_row.get(field)
            for field in ("parent_state_hash", "event_hash", "child_state_hash")
        ):
            errors.append("lineage_hash_mismatch")
        if row.get("exactly_once") is not True:
            errors.append("lineage_not_exactly_once")
    for row in rows:
        if row.get("row_type") == "revocation" and row.get("replayed") is not True:
            errors.append("revoked_label_not_replayed")
    return list(dict.fromkeys(errors))


def rebuild_weight_updates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute each recorded weight transition when operands are complete."""

    feedback = {_online_key(row): row for row in rows if row.get("row_type") == "feedback_event"}
    trajectories = [row for row in rows if row.get("row_type") == "weight_trajectory"]
    maximum_gap = 0.0
    replayed = 0
    for trajectory in trajectories:
        event = feedback.get(_online_key(trajectory))
        if event is None or not isinstance(event.get("expert_prediction_time_losses"), Mapping):
            raise ValueError("complete expert prediction losses are required")
        expected = replay_weight_update(
            trajectory["weights_before"], event["expert_prediction_time_losses"]
        )
        observed = trajectory["weights_after"]
        maximum_gap = max(
            maximum_gap,
            max(abs(expected[name] - float(observed[name])) for name in EXPERT_NAMES),
        )
        replayed += 1
    return {"updates_replayed": replayed, "max_weight_gap": maximum_gap}


def _reduce_online_arm(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    losses = [binary_log_loss(int(row["label"]), float(row["probability"])) for row in rows]
    briers = [(float(row["probability"]) - int(row["label"])) ** 2 for row in rows]
    revealed = [row for row in rows if row.get("revealed") is True]
    selected = [row for row in rows if row.get("proposed_action") != "escalate"]
    harmful = sum(
        (row.get("proposed_action") == "accept" and int(row["label"]) == 0)
        or (row.get("proposed_action") == "reject" and int(row["label"]) == 1)
        for row in selected
    )
    changed = [index for index, row in enumerate(rows) if row.get("domain_changed") is True]
    changed_loss = float(np.mean([losses[index] for index in changed])) if changed else None
    return {
        "row_count": len(rows),
        "full_stream_log_loss": float(np.mean(losses)),
        "full_stream_brier": float(np.mean(briers)),
        "revealed_row_count": len(revealed),
        "revealed_only_log_loss": (
            float(
                np.mean(
                    [
                        binary_log_loss(int(row["label"]), float(row["probability"]))
                        for row in revealed
                    ]
                )
            )
            if revealed
            else None
        ),
        "revealed_only_brier": (
            float(
                np.mean([(float(row["probability"]) - int(row["label"])) ** 2 for row in revealed])
            )
            if revealed
            else None
        ),
        "ipw_log_loss": math.fsum(
            binary_log_loss(int(row["label"]), float(row["probability"])) / float(row["propensity"])
            for row in revealed
        )
        / len(rows),
        "ipw_brier": math.fsum(
            (float(row["probability"]) - int(row["label"])) ** 2 / float(row["propensity"])
            for row in revealed
        )
        / len(rows),
        "label_cost": len(revealed) / len(rows),
        "shadow_coverage": len(selected) / len(rows),
        "selected_action_count": len(selected),
        "shadow_harmful_action_rate": harmful / len(selected) if selected else None,
        "deployment_policy": "all_escalate",
        "deployment_coverage": 0.0,
        "deployment_selected_risk": None,
        "domain_change_row_count": len(changed),
        "domain_change_log_loss": changed_loss,
        "domain_change_harm": changed_loss - float(np.mean(losses))
        if changed_loss is not None
        else None,
    }


def reduce_online_metrics(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce proper scores, label cost, and shadow actions by registered cell."""

    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") == "prediction":
            grouped[(str(row["ordering"]), int(row["delay"]), str(row["arm"]))].append(row)
    return [
        {"ordering": ordering, "delay": delay, "arm": arm, **_reduce_online_arm(group)}
        for (ordering, delay, arm), group in sorted(grouped.items())
    ]


def moving_block_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = ONLINE_BOOTSTRAP_SEED,
    block_lengths: Sequence[int] = (32, 64),
) -> list[JsonDict]:
    """Recompute simultaneous moving-block loss intervals from raw predictions."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    values: dict[tuple[str, int, str, str], dict[int, float]] = defaultdict(dict)
    positions: dict[tuple[str, int, str], int] = {}
    allowed = {"learned_mixture", *PRIMARY_COMPARATORS}
    for row in rows:
        if row.get("row_type") != "prediction" or row.get("arm") not in allowed:
            continue
        key = (str(row["ordering"]), int(row["delay"]), str(row["group_id"]), str(row["arm"]))
        values[key][int(row["seed"])] = binary_log_loss(
            int(row["label"]), float(row["probability"])
        )
        positions[(str(row["ordering"]), int(row["delay"]), str(row["group_id"]))] = int(
            row["prediction_index"]
        )
    seed_counts = {len(item) for item in values.values()}
    if len(seed_counts) != 1 or not seed_counts or min(seed_counts) <= 0:
        raise ValueError("moving blocks require complete seed rows")
    fit_seed_count = next(iter(seed_counts))
    contrasts: dict[tuple[str, int, str], np.ndarray] = {}
    for ordering in ORDERS:
        for delay in DELAYS:
            groups = sorted(
                {
                    group
                    for row_order, row_delay, group, _arm in values
                    if row_order == ordering and row_delay == delay
                },
                key=lambda group: positions[(ordering, delay, group)],
            )
            for comparator in PRIMARY_COMPARATORS:
                deltas: list[float] = []
                for group in groups:
                    learned = values.get((ordering, delay, group, "learned_mixture"), {})
                    control = values.get((ordering, delay, group, comparator), {})
                    if set(learned) != set(control) or len(learned) != fit_seed_count:
                        raise ValueError("moving blocks require complete paired rows")
                    deltas.append(
                        float(np.mean([learned[item] - control[item] for item in sorted(learned)]))
                    )
                contrasts[(ordering, delay, comparator)] = np.asarray(deltas, dtype=np.float64)
    keys = sorted(contrasts)
    if len(keys) != 12 or len({len(contrasts[key]) for key in keys}) != 1:
        raise ValueError("moving blocks require all registered cells")
    group_count = len(contrasts[keys[0]])
    rng = np.random.default_rng(seed)
    output: list[JsonDict] = []
    for block_length in block_lengths:
        samples = np.empty((draws, len(keys)), dtype=np.float64)
        blocks = math.ceil(group_count / block_length)
        for draw in range(draws):
            starts = rng.integers(0, group_count, size=blocks)
            indices = np.concatenate(
                [
                    np.arange(start, start + block_length, dtype=np.int64) % group_count
                    for start in starts
                ]
            )[:group_count]
            for index, key in enumerate(keys):
                samples[draw, index] = float(np.mean(contrasts[key][indices]))
        observed = np.asarray([float(np.mean(contrasts[key])) for key in keys])
        simultaneous_margin = float(np.quantile(np.max(samples - observed, axis=1), 0.95))
        for index, (ordering, delay, comparator) in enumerate(keys):
            output.append(
                {
                    "ordering": ordering,
                    "delay": delay,
                    "comparator": comparator,
                    "metric": "log_loss_delta",
                    "observed": float(observed[index]),
                    "lower": float(np.quantile(samples[:, index], 0.025)),
                    "upper": float(observed[index] + simultaneous_margin),
                    "confidence": 0.95,
                    "family_contrasts": 12,
                    "draws": draws,
                    "seed": seed,
                    "block_length": block_length,
                    "paired_source_groups": group_count,
                    "fit_seeds_averaged_before_resampling": fit_seed_count,
                }
            )
    return output


def online_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce negative controls and representation limits from prediction rows."""

    predictions = [row for row in rows if row.get("row_type") == "prediction"]
    no_feedback: dict[tuple[int, str], set[float]] = defaultdict(set)
    losses: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in predictions:
        arm = str(row.get("arm"))
        if arm == "no_feedback_frozen_prior_mixture":
            no_feedback[(int(row["seed"]), str(row["group_id"]))].add(float(row["probability"]))
        if arm in {
            "learned_mixture",
            "shuffled_labels_permutation_1",
            "shuffled_labels_permutation_2",
        }:
            losses[(str(row["ordering"]), int(row["delay"]), arm)].append(
                binary_log_loss(int(row["label"]), float(row["probability"]))
            )
    controls: JsonDict = {
        "no_feedback_equality": all(len(value) == 1 for value in no_feedback.values()),
        "uniform_reveal_propensity": bool(predictions)
        and all(
            any(
                math.isclose(float(row["propensity"]), expected, abs_tol=1e-12)
                for expected in REGISTERED_REVEAL_PROPENSITIES
            )
            for row in predictions
        ),
        "shadow_only_actions": bool(predictions)
        and all(
            row.get("shadow_only") is True
            and row.get("certified_safe") is False
            and row.get("deployed_action") == "escalate"
            for row in predictions
        ),
        "calibration_theorem_applied": False,
        "no_share_regret_theorem_applied": False,
        "energy_representation": "logistic_reexpression_of_probability",
        "new_energy_advantage_claimed": False,
    }
    for index, arm in enumerate(
        ("shuffled_labels_permutation_1", "shuffled_labels_permutation_2"), 1
    ):
        comparisons: list[bool] = []
        for ordering in ORDERS:
            for delay in DELAYS:
                learned = losses[(ordering, delay, "learned_mixture")]
                shuffled = losses[(ordering, delay, arm)]
                if learned or shuffled:
                    comparisons.append(
                        bool(learned)
                        and bool(shuffled)
                        and float(np.mean(learned)) <= float(np.mean(shuffled)) + 1e-12
                    )
        controls[f"shuffled_label_control_{index}"] = all(comparisons)
    return controls


def run_mutation_controls() -> list[JsonDict]:
    """Plant every registered defect in private in-memory evidence."""

    policy = {
        "head": "sparse_spline_49",
        "policy_kind": "tuned",
        "frozen_policy": {
            "accept_enabled": True,
            "accept_threshold": 0.8,
            "reject_enabled": True,
            "reject_threshold": 0.2,
            "selection_partition": "policy_tuning",
        },
        "certificate": {
            name: {"alpha_allocated": ALPHA_PER_CHECK}
            for name in ("accept_check", "reject_check", "coverage_check")
        },
    }
    static_row = {
        "stage": "certification",
        "head": "sparse_spline_49",
        "policy_kind": "tuned",
        "group_id": "g1",
        "row_key": "r1",
        "label": 1,
        "probability": 0.9,
    }
    before = {name: 0.25 for name in EXPERT_NAMES}
    losses = {name: 0.2 + index / 100 for index, name in enumerate(EXPERT_NAMES)}
    feedback = {
        "row_type": "feedback_event",
        "observation_id": "g1",
        "group_id": "g1",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 0,
        "seed": 1,
        "prediction_index": 0,
        "arrival_index": 0,
        "feedback_label": 1,
        "prediction_time_loss": binary_log_loss(1, 0.8),
        "probability_source": "stored_at_prediction",
        "expert_prediction_time_losses": losses,
        "update_count": 1,
    }
    prediction = {
        "row_type": "prediction",
        "observation_id": "g1",
        "group_id": "g1",
        "row_key": "r1",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 0,
        "seed": 1,
        "prediction_index": 0,
        "probability": 0.8,
        "label": 1,
        "propensity": 0.25,
        "label_read_at_prediction": False,
        "prediction_before_feedback": True,
        "shadow_only": True,
        "certified_safe": False,
        "deployed_action": "escalate",
        "loss": binary_log_loss(1, 0.8),
        "brier": 0.04,
    }
    weight = {
        "row_type": "weight_trajectory",
        "observation_id": "g1",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 0,
        "seed": 1,
        "weights_before": before,
        "weights_after": replay_weight_update(before, losses),
    }
    cases: list[tuple[str, bool]] = []
    leaked = deepcopy(policy)
    leaked["frozen_policy"]["selection_partition"] = "final_test"
    cases.append(
        (
            "role_leakage",
            "final_test_threshold_influence" in static_integrity_errors([static_row], [leaked]),
        )
    )
    duplicate = [static_row, {**static_row, "row_key": "r2"}]
    cases.append(
        (
            "wrong_group_unit",
            "duplicate_source_group" in static_integrity_errors(duplicate, [policy]),
        )
    )
    false_zero = deepcopy(policy)
    false_zero["certificate"]["accept_check"].update({"selected_groups": 0, "empirical_risk": 0.0})
    cases.append(
        (
            "zero_selected_risk_coercion",
            "zero_selected_risk_coercion" in static_integrity_errors([static_row], [false_zero]),
        )
    )
    future = deepcopy(feedback)
    future["arrival_index"] = -1
    cases.append(
        (
            "future_label",
            "future_label_access" in online_integrity_errors([prediction, future, weight]),
        )
    )
    cases.append(
        (
            "duplicate_event",
            "duplicate_feedback_event"
            in online_integrity_errors([prediction, feedback, feedback, weight]),
        )
    )
    revoked = deepcopy(feedback)
    revoked["revoked"] = True
    cases.append(
        (
            "revoked_label",
            "revoked_label_applied" in online_integrity_errors([prediction, revoked, weight]),
        )
    )
    missing = deepcopy(feedback)
    del missing["expert_prediction_time_losses"]
    cases.append(
        (
            "missing_expert_predictions",
            "missing_expert_predictions" in online_integrity_errors([prediction, missing, weight]),
        )
    )
    return [
        {
            "attack": name,
            "expected": "rejected",
            "observed": "rejected" if passed else "accepted",
            "passed": passed,
        }
        for name, passed in cases
    ]


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _roadmap_deliverables(path: Path) -> dict[str, Path]:
    """Read task deliverables without importing a roadmap execution helper."""

    if not path.is_file():
        return {}
    wanted = {str(value["task_id"]) for value in PRODUCERS.values()}
    output: dict[str, Path] = {}
    current: str | None = None
    for source_line in path.read_text(encoding="utf-8").splitlines():
        line = source_line.strip()
        if line.startswith("- id:"):
            current = line.split(":", 1)[1].strip()
        elif current in wanted and line.startswith("deliverable:"):
            output[current] = Path(line.split(":", 1)[1].strip())
    return output


def _observed_type(value: Any, *, present: bool) -> str:
    if not present:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and value == 0:
        return "integer_zero"
    return type(value).__name__


def _precondition(
    branch: str,
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    present: bool = True,
    passed: bool | None = None,
) -> JsonDict:
    return {
        "branch": branch,
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "observed_type": _observed_type(observed, present=present),
        "passed": observed == expected if passed is None else bool(passed),
    }


def _producer_validation_passed(value: Mapping[str, Any]) -> bool:
    summary = value.get("gate_check_summary") or {}
    if summary.get("required_checks_passed") is True:
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


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:
    """Locate roadmap-declared producers and preserve exact field observations."""

    root = repo_root.resolve()
    paths = _roadmap_deliverables(root / ROADMAP_PATH)
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    producers: dict[str, JsonDict] = {}
    for name, definition in PRODUCERS.items():
        task_id = str(definition["task_id"])
        relative = paths.get(task_id)
        path_text = relative.as_posix() if relative is not None else None
        for branch in definition["branches"]:
            checks.append(
                _precondition(
                    branch,
                    f"{name}:roadmap_path",
                    task_id,
                    ROADMAP_PATH.as_posix(),
                    "deliverable",
                    "declared_path",
                    "declared_path" if relative is not None else None,
                    present=relative is not None,
                )
            )
        path = root / relative if relative is not None else root / f"results/missing-{name}.json"
        exists = path.is_file() and path.stat().st_size > 0
        producer = _load_object(path) if exists else {}
        producers[name] = producer
        is_object = bool(producer)
        for branch in definition["branches"]:
            checks.append(
                _precondition(
                    branch,
                    f"{name}:artifact_exists",
                    task_id,
                    path_text or str(path),
                    "path",
                    True,
                    exists,
                    present=exists,
                )
            )
            checks.append(
                _precondition(
                    branch,
                    f"{name}:artifact_object",
                    task_id,
                    path_text or str(path),
                    "json_object",
                    True,
                    True if is_object else None,
                    present=is_object,
                )
            )
        if is_object:
            digest = sha256_file(path)
            hashes[relative.as_posix() if relative else str(path)] = {
                "path": relative.as_posix() if relative else str(path),
                "sha256": digest,
                "source_receipt_class": "same_milestone_producer",
                "original_honest_verdict": producer.get("honest_verdict"),
                "original_verdict_class": producer.get("verdict_class"),
                "original_flagged_adversarial": producer.get("flagged_adversarial"),
            }
        fields = (
            ("experiment_id", definition["experiment_id"]),
            ("milestone", MILESTONE),
            (str(definition["completion_field"]), 1),
            ("flagged_adversarial", False),
        )
        for branch in definition["branches"]:
            for field, expected in fields:
                present = field in producer
                checks.append(
                    _precondition(
                        branch,
                        f"{name}:{field}",
                        task_id,
                        relative.as_posix() if relative else str(path),
                        field,
                        expected,
                        producer.get(field),
                        present=present,
                    )
                )
            verdict = producer.get("verdict_class")
            checks.append(
                _precondition(
                    branch,
                    f"{name}:verdict_class",
                    task_id,
                    relative.as_posix() if relative else str(path),
                    "verdict_class",
                    sorted(definition["allowed_classes"]),
                    verdict,
                    present="verdict_class" in producer,
                    passed=verdict in definition["allowed_classes"],
                )
            )
            checks.append(
                _precondition(
                    branch,
                    f"{name}:producer_validation",
                    task_id,
                    relative.as_posix() if relative else str(path),
                    "validation_receipts.required_checks",
                    True,
                    _producer_validation_passed(producer),
                )
            )
    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    for branch in ("static", "online"):
        checks.append(
            _precondition(
                branch,
                "driving_requirement",
                SPEC_PATH.as_posix(),
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-REPORT-7441",
                "REQ-REPORT-7441" if "REQ-REPORT-7441" in spec_text else None,
                present="REQ-REPORT-7441" in spec_text,
            )
        )
    return checks, hashes, producers


def _load_bound_rows(root: Path, manifests: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Load only shards whose exact byte hashes match their producer manifest."""

    output: list[JsonDict] = []
    for manifest in manifests:
        relative = Path(str(manifest.get("path") or ""))
        path = relative if relative.is_absolute() else root / relative
        expected = str(manifest.get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"row_shard_hash_mismatch:{relative}")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row_shard_object_invalid:{relative}")
                output.append(row)
                count += 1
        if count != int(manifest.get("rows", -1)):
            raise ValueError(f"row_shard_count_mismatch:{relative}")
    return output


def blocked_branch(branch: str, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve one unavailable external branch as terminal blocked evidence."""

    failed = next((row for row in checks if row.get("passed") is not True), {})
    return {
        "branch": branch,
        "available": False,
        "valid": False,
        "complete": False,
        "value": False,
        "verdict_class": "blocked",
        "honest_verdict": f"blocked_{branch}_external_producer_unavailable",
        "errors": ["external_producer_unavailable"],
        "gate_check_summary": {
            "blocked_upstream": failed.get("upstream"),
            "blocked_path": failed.get("path"),
            "blocked_check": failed.get("check"),
            "blocked_field": failed.get("field"),
            "blocked_expected": failed.get("expected"),
            "blocked_observed": failed.get("observed"),
        },
    }


def audit_static_branch(
    root: Path, producer: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce every available static row even when the producer result is null."""

    if not producer or any(row.get("passed") is not True for row in checks):
        return blocked_branch("static", checks)
    errors: list[str] = []
    try:
        rows = _load_bound_rows(root, producer.get("probability_row_shards") or [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rows = []
        errors.append(str(exc))
    certificates = producer.get("policy_certificates") or []
    errors.extend(static_integrity_errors(rows, certificates))
    try:
        reduced = reduce_static_evidence(rows, certificates)
    except (KeyError, TypeError, ValueError) as exc:
        reduced = {}
        errors.append(f"static_reduction_failed:{exc}")
    valid = not errors and producer.get("flagged_adversarial") is False
    return {
        "branch": "static",
        "upstream": producer.get("experiment_id"),
        "available": True,
        "valid": valid,
        "complete": True,
        "value": valid and producer.get("decision_value_score") == 1,
        "verdict_class": producer.get("verdict_class") if valid else "disqualified",
        "honest_verdict": (
            str(producer.get("honest_verdict"))
            if valid
            else "complete_disqualified_static_evidence_defect"
        ),
        "producer_verdict_class": producer.get("verdict_class"),
        "producer_flagged_adversarial": producer.get("flagged_adversarial"),
        "raw_row_count": len(rows),
        "raw_evidence_hash": canonical_hash(producer.get("probability_row_shards") or []),
        "reduced_evidence": reduced,
        "errors": list(dict.fromkeys(errors)),
        "gate_check_summary": {
            "blocked_upstream": None,
            "blocked_path": None,
            "blocked_check": None,
            "blocked_field": None,
            "blocked_expected": None,
            "blocked_observed": None,
        },
    }


def audit_online_branch(
    root: Path, producer: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce online metrics and require replayable prediction-time expert evidence."""

    if not producer or any(row.get("passed") is not True for row in checks):
        return blocked_branch("online", checks)
    errors: list[str] = []
    try:
        rows = _load_bound_rows(root, producer.get("row_shards") or [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rows = []
        errors.append(str(exc))
    errors.extend(online_integrity_errors(rows))
    predictions = [row for row in rows if row.get("row_type") == "prediction"]
    try:
        reports = reduce_online_metrics(predictions)
        intervals = moving_block_intervals(predictions)
        controls = online_controls(predictions)
    except (KeyError, TypeError, ValueError) as exc:
        reports, intervals, controls = [], [], {}
        errors.append(f"online_reduction_failed:{exc}")
    try:
        replay = rebuild_weight_updates(rows)
    except (KeyError, TypeError, ValueError) as exc:
        replay = {"updates_replayed": 0, "max_weight_gap": None, "error": str(exc)}
        errors.append("weight_update_replay_incomplete")
    valid = not errors and producer.get("flagged_adversarial") is False
    return {
        "branch": "online",
        "upstream": producer.get("experiment_id"),
        "available": True,
        "valid": valid,
        "complete": True,
        "value": valid and producer.get("online_value_score") == 1,
        "verdict_class": producer.get("verdict_class") if valid else "disqualified",
        "honest_verdict": (
            str(producer.get("honest_verdict"))
            if valid
            else "complete_disqualified_online_missing_prediction_time_expert_evidence"
        ),
        "producer_verdict_class": producer.get("verdict_class"),
        "producer_flagged_adversarial": producer.get("flagged_adversarial"),
        "raw_row_count": len(rows),
        "prediction_row_count": len(predictions),
        "raw_evidence_hash": canonical_hash(producer.get("row_shards") or []),
        "condition_reports": reports,
        "moving_block_intervals": intervals,
        "weight_update_replay": replay,
        "controls": controls,
        "errors": list(dict.fromkeys(errors)),
        "gate_check_summary": {
            "blocked_upstream": None,
            "blocked_path": None,
            "blocked_check": None,
            "blocked_field": None,
            "blocked_expected": None,
            "blocked_observed": None,
        },
    }


def classify_terminal(
    branches: Sequence[Mapping[str, Any]], *, validation_passed: bool
) -> JsonDict:
    """Keep invalid, blocked, positive, and complete null outcomes distinct."""

    if any(row.get("available") is True and row.get("valid") is not True for row in branches):
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_invalid_branch_evidence",
        }
    if not validation_passed:
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_required_validation",
        }
    unavailable = [str(row.get("branch")) for row in branches if row.get("available") is not True]
    if unavailable:
        return {
            "verdict_class": "blocked",
            "honest_verdict": "blocked_" + "_and_".join(unavailable) + "_upstream_science",
        }
    if any(row.get("value") is True for row in branches):
        return {
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_independently_reproduced_decision_value",
        }
    return {
        "verdict_class": "null",
        "honest_verdict": "complete_null_static_and_online_audits_reproduce_no_benefit",
    }


def _field_principles() -> dict[str, str]:
    return {
        "schema": "A versioned top-level identity makes incompatible records fail closed.",
        "run_date": "The scheduled date stays separate from measured UTC boundaries.",
        "preconditions_checked": "Every resource is observed before dependent reduction.",
        "MODEL_SPECS": "An empty list means this audit invoked no current LLM.",
        "model_invoked": "Attempted current model work stays separate from archived evidence.",
        "invocation_counts": "Loads and generations reconcile by terminal state.",
        "inference_substrate": "The string names current host numeric aggregation.",
        "inference_substrate_class": "Aggregation has no model duration floor.",
        "execution_venue": "Host execution stays distinct from CUDA and external devices.",
        "duration_s": "Measured work separates computation, cold reads, and validation.",
        "phase_spans": "Monotonic spans bind progress events and finished units.",
        "random_seed": "Static and moving-block resampling seeds are frozen.",
        "reproducibility_checksum": "The checksum binds code, inputs, rows, and validation scope.",
        "source_artifact_hashes": "Typed hashes preserve producer classes and flags.",
        "rows": "Unit summaries retain completed and failed audit units.",
        "sample_size_budget": "Counts distinguish planned, attempted, completed, and unstarted work.",
        "acceptance_gate_results": "Validity and scientific value use separate gate categories.",
        "gate_check_summary": "Blocked fields preserve missing, null, and zero observations.",
        "verifier_is_oracle": "Human source-support labels are not the deployed verifier.",
        "honest_verdict": "Terminal wording distinguishes invalid evidence from valid nulls.",
        "verdict_class": "Partial is reserved for unfinished work owned by this audit.",
        "flagged_adversarial": "Source flags remain visible and cannot supply readiness.",
        "validation_receipts": "Exact commands, exits, times, and log hashes support publication.",
        "field_principles": "Principles explain fields without wrapping gate scalars.",
        "promotion_score": "This audit cannot promote production or update generator weights.",
        "static_audit": "Policy certification and probability quality retain one disposition.",
        "online_audit": "Causal replay and online effects retain a separate disposition.",
        "audit_mutation_rows": "Private defects show each safety boundary failing closed.",
        "audited_claim_rows": "Each conclusion names its raw source and preserves limits.",
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, terminal: bool) -> bool:
    names = set(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        names.update(
            {
                "cold_artifact_replay",
                "independent_branch_recompute",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            }
        )
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        for name in names
    )


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable audit evidence while excluding clocks and command timing."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    for field in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "model_duration_s",
        "computation_duration_s",
        "cold_start_duration_s",
        "validation_duration_s",
    ):
        payload[field] = 0
    payload["phase_spans"] = []
    payload["validation_receipts"] = []
    return canonical_hash(payload)


def _build_artifact(
    static: Mapping[str, Any],
    online: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    producer_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    candidate: bool,
    fixture: bool = False,
) -> JsonDict:
    validation_ok = _validation_passed(validation_receipts, terminal=not candidate)
    branches = [dict(static), dict(online)]
    classification = classify_terminal(branches, validation_passed=validation_ok)
    failed_branch = next((row for row in branches if row.get("available") is not True), {})
    failed_errors = [
        f"{row.get('branch')}:{error}" for row in branches for error in row.get("errors") or []
    ]
    mutations = run_mutation_controls()
    gates = [
        {
            "check": "required_validation",
            "category": "validation",
            "operator": "==",
            "expected": True,
            "observed": validation_ok,
            "passed": validation_ok,
            "principle": "Only the frozen affected and terminal readers authorize publication.",
        },
        {
            "check": "static_evidence_valid",
            "category": "validity",
            "operator": "==",
            "expected": True,
            "observed": static.get("valid"),
            "passed": static.get("valid") is True,
            "principle": "Static null evidence remains usable only when raw rows reproduce.",
        },
        {
            "check": "online_evidence_valid",
            "category": "validity",
            "operator": "==",
            "expected": True,
            "observed": online.get("valid"),
            "passed": online.get("valid") is True,
            "principle": "Every online update requires complete prediction-time expert evidence.",
        },
        {
            "check": "promotion_disabled",
            "category": "safety",
            "operator": "==",
            "expected": 0,
            "observed": 0,
            "passed": True,
            "principle": "An audit result cannot change production policy.",
        },
    ]
    validation_duration = math.fsum(
        float(row.get("duration_s", 0.0)) for row in validation_receipts
    )
    duration = float(current_receipt["duration_s"])
    result: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": classification["honest_verdict"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **deepcopy(dict(current_receipt)),
        "duration_s": duration,
        "model_duration_s": 0.0,
        "computation_duration_s": max(0.0, duration - validation_duration),
        "cold_start_duration_s": math.fsum(
            float(row.get("duration_s", 0.0))
            for row in validation_receipts
            if row.get("name") in {"cold_artifact_replay", "independent_branch_recompute"}
        ),
        "validation_duration_s": validation_duration,
        "random_seed": {
            "static_paired_bootstrap": STATIC_BOOTSTRAP_SEED,
            "online_moving_block_bootstrap": ONLINE_BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "affected_validation_manifest": {
            "experiment_id": V652_MANIFEST.experiment_id,
            "test_paths": list(V652_MANIFEST.test_paths),
            "changed_modules": list(V652_MANIFEST.changed_modules),
            "static_paths": list(V652_MANIFEST.static_paths),
            "required_command_names": list(validation_scope.REQUIRED_CHECK_NAMES),
        },
        "rows": [deepcopy(dict(row)) for row in producer_rows],
        "sample_size_budget": {
            "planned_independent_units": 2,
            "attempted_independent_units": 2,
            "completed_independent_units": 2,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "unstarted_independent_units": 0,
            "stopping_rule": "audit each available branch once and retain terminal upstream absence",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "required_checks_passed": validation_ok,
            "failed_required_checks": ([] if validation_ok else ["required_validation"]),
            "evidence_defects": failed_errors,
            **{
                "blocked_upstream": (failed_branch.get("gate_check_summary") or {}).get(
                    "blocked_upstream"
                ),
                "blocked_path": (failed_branch.get("gate_check_summary") or {}).get("blocked_path"),
                "blocked_check": (failed_branch.get("gate_check_summary") or {}).get(
                    "blocked_check"
                ),
                "blocked_field": (failed_branch.get("gate_check_summary") or {}).get(
                    "blocked_field"
                ),
                "blocked_expected": (failed_branch.get("gate_check_summary") or {}).get(
                    "blocked_expected"
                ),
                "blocked_observed": (failed_branch.get("gate_check_summary") or {}).get(
                    "blocked_observed"
                ),
            },
        },
        "verifier_is_oracle": False,
        **classification,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "static_audit": deepcopy(dict(static)),
        "online_audit": deepcopy(dict(online)),
        "audit_mutation_rows": mutations,
        "audited_claim_rows": [
            {
                "claim": "static_policy_evidence",
                "source": "exp7439 probability_row_shards",
                "disposition": static.get("verdict_class"),
                "limit": "exploratory reused corpus; no deployment certificate",
            },
            {
                "claim": "delayed_online_learning",
                "source": "exp7440 event row_shards",
                "disposition": online.get("verdict_class"),
                "limit": "shadow-only actions; no inherited calibration or regret theorem",
            },
            {
                "claim": "energy_representation",
                "source": "exp7438 mixture definition",
                "disposition": "representation_equivalence",
                "limit": "negative log odds is a logistic re-expression, not a new energy advantage",
            },
        ],
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
    }
    result["reproducibility_checksum"] = reproducibility_checksum(result)
    return result


def _fake_receipts() -> list[JsonDict]:
    names = [
        *validation_scope.REQUIRED_CHECK_NAMES,
        "cold_artifact_replay",
        "independent_branch_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    return [{"name": name, "passed": True, "exit_code": 0, "duration_s": 0.0} for name in names]


def build_artifact_for_test() -> JsonDict:
    """Build a compact valid terminal artifact for mutation and reader tests."""

    branch = {
        "available": True,
        "valid": True,
        "complete": True,
        "value": False,
        "verdict_class": "null",
        "honest_verdict": "complete_null_fixture",
        "errors": [],
    }
    receipt = build_current_work_receipt(
        run_id="exp7441-fixture",
        owner_pid=1,
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
        inference_substrate_details={"cpu": "fixture", "cuda": None, "external_device": None},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1,
        phase_spans=[],
        small_ebm_training={"performed": False},
    )
    return _build_artifact(
        {**branch, "branch": "static"},
        {**branch, "branch": "online"},
        preconditions=[],
        source_hashes={},
        producer_rows=[],
        validation_receipts=_fake_receipts(),
        current_receipt=receipt,
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        candidate=False,
        fixture=True,
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    """Cold-check identity, branches, mutations, provenance, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("artifact_schedule_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_invalid")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    branches = [value.get("static_audit") or {}, value.get("online_audit") or {}]
    expected = classify_terminal(
        branches,
        validation_passed=_validation_passed(
            value.get("validation_receipts") or [],
            terminal=not bool(value.get("candidate_artifact")),
        ),
    )
    if any(value.get(field) != expected[field] for field in expected):
        errors.append("terminal_classification_mismatch")
    attacks = value.get("audit_mutation_rows") or []
    if {row.get("attack") for row in attacks if isinstance(row, Mapping)} != set(
        REQUIRED_MUTATIONS
    ) or any(row.get("passed") is not True for row in attacks if isinstance(row, Mapping)):
        errors.append("mutation_controls_invalid")
    if not value.get("fixture_artifact"):
        errors.extend(validate_current_work_receipt(value, root=root))
    if verify_source_bytes:
        for label, reference in (value.get("source_artifact_hashes") or {}).items():
            if not isinstance(reference, Mapping):
                errors.append(f"source_reference_invalid:{label}")
                continue
            path = Path(str(reference.get("path") or label))
            resolved = path if path.is_absolute() else root / path
            observed = sha256_file(resolved) if resolved.is_file() else None
            if observed != reference.get("sha256"):
                errors.append(f"source_hash_mismatch:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(
    path: Path, *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    """Reload a candidate in a fresh process and apply strict field checks."""

    value = _load_object(path)
    if not value:
        return ["candidate_artifact_unreadable"]
    return validate_artifact(value, root=root, verify_source_bytes=verify_source_bytes)


def _audit_sources(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict], JsonDict, JsonDict]:
    checks, hashes, producers = collect_preconditions(root)
    by_branch = {
        branch: [row for row in checks if row.get("branch") == branch]
        for branch in ("static", "online")
    }
    static = audit_static_branch(root, producers.get("static") or {}, by_branch["static"])
    online = audit_online_branch(root, producers.get("online") or {}, by_branch["online"])
    return checks, hashes, producers, static, online


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload producer rows and compare both branch reductions to a candidate."""

    value = _load_object(path)
    errors = validate_artifact(value, root=root)
    _checks, _hashes, _producers, static, online = _audit_sources(root)
    if canonical_hash(static) != canonical_hash(value.get("static_audit")):
        errors.append("static_independent_reduction_mismatch")
    if canonical_hash(online) != canonical_hash(value.get("online_audit")):
        errors.append("online_independent_reduction_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    commands = build_command_plan(root, V652_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V652_MANIFEST, commands)
    if plan_errors:
        raise ValueError("validation_plan_invalid:" + ",".join(plan_errors))
    return commands


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:
    commands = (
        (
            "cold_artifact_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--cold-replay",
                str(candidate),
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7441] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": _utc_now(),
    }


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint E2E.
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
    checks, hashes, producers, static, online = _audit_sources(root)
    for relative, receipt_class in (
        (MODULE_PATH, "current_audit_code"),
        (WRAPPER_PATH, "current_audit_entrypoint"),
        (TEST_PATH, "current_audit_tests"),
        (SPEC_PATH, "current_audit_protocol"),
        (ROADMAP_PATH, "current_audit_roadmap"),
    ):
        path = root / relative
        hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "source_receipt_class": receipt_class,
        }
    spans.append(_span("branch_reduction", phase_started, started, 2))
    _progress(
        started,
        "branch_reduction",
        "after_benchmark",
        static_valid=static.get("valid"),
        online_valid=online.get("valid"),
    )

    private = Path(tempfile.mkdtemp(prefix="exp7441-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private)
    phase_started = time.monotonic()
    _progress(started, "affected_validation", "before_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
    )
    affected_reduction = reduce_affected_receipts(root, V652_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    _progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )

    ended_ns = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"exp7441-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
        inference_substrate_details={
            "cpu": platform.processor() or "host_cpu",
            "cuda": None,
            "external_device": None,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=ended_ns - started_ns,
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    producer_rows = [
        {"source": name, **dict(row)}
        for name in ("static", "online")
        for row in (producers.get(name, {}).get("rows") or [])
        if isinstance(row, Mapping)
    ]
    candidate = _build_artifact(
        static,
        online,
        preconditions=checks,
        source_hashes=hashes,
        producer_rows=producer_rows,
        validation_receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        candidate=True,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    terminal_plan = _terminal_commands(candidate_path)
    _progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
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
        run_id=f"exp7441-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
        inference_substrate_details={
            "cpu": platform.processor() or "host_cpu",
            "cuda": None,
            "external_device": None,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=final_ns - started_ns,
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    final = _build_artifact(
        static,
        online,
        preconditions=checks,
        source_hashes=hashes,
        producer_rows=producer_rows,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        candidate=False,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--skip-source-bytes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, verify_source_bytes=not args.skip_source_bytes)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = (
            cold_replay(args.independent_reduce, verify_source_bytes=False)
            if args.skip_source_bytes
            else independent_replay(args.independent_reduce)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
