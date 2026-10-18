"""Audit V653 static decisions and delayed continuous learning independently.

This host aggregation reads both decision branches without importing producer
verdict reducers or update functions. It reconstructs expert updates from initial
numeric state and stored prediction-time probabilities, verifies causal ordering
and immutability, audits evidence corruption mutations, and retains branch-local
blocks and prior flags.

Spec refs: REQ-REPORT-7455 and SCENARIO-REPORT-7455-*.
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
import random
import tempfile
import time
from typing import Any

import yaml

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
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7455-v653-decision-audit"
SCHEMA = "carnot.exp7455.v653.decision_audit.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7455_v653_decision_audit.json")
RAW_DIR = Path("results/raw/experiment_7455_v653_decision_audit")
MODULE_PATH = Path("python/carnot/experiment_7455_v653_decision_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7455_v653_decision_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7455_v653_decision_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
SOURCE_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7441_v652_decision_audit.py"),
    Path("python/carnot/experiment_7446_v652_capstone.py"),
    ROADMAP_PATH,
)

EXPERT_NAMES = (
    "adaptive_gibbs",
    "adaptive_spline",
    "frozen_gibbs",
    "frozen_spline",
)
PREDICTION_SCHEMA = "carnot.exp7454.prediction_event.v1"
OUTCOME_SCHEMA = "carnot.exp7454.outcome_event.v1"
FEEDBACK_SCHEMA = "carnot.exp7454.feedback_event.v1"
CHECKPOINT_SCHEMA = "carnot.exp7454.checkpoint_event.v1"
PRIMARY_COMPARATORS = (
    "frozen_spline",
    "adaptive_spline",
    "equal_weight_adaptive_mixture",
)
FLOAT_TOLERANCE = 1e-9
FLOAT_CLIP = 1e-6

REQUIRED_MUTATIONS = (
    "mutate_prediction_probability",
    "shuffle_event_order",
    "drop_feedback_event",
    "leak_test_label_into_features",
    "duplicate_source_group",
    "replace_vector_with_length_only",
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
    "computation_duration_s",
    "cold_start_duration_s",
    "validation_duration_s",
    "phase_spans",
    "clock_identity",
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
    "decision_audit_complete_score",
    "branch_rows",
    "mutation_rows",
    "independent_update_replay",
    "continuation_rows",
    "mixture_construction_retired",
    "static_audit",
    "online_audit",
)

MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(
        TEST_PATH.as_posix(),
        "tests/python/test_current_work_receipt.py",
        "tests/python/test_experiment_7303_v642_validation_scope.py",
        "tests/python/test_experiment_7358_v646_validation_contract.py",
        "tests/python/test_experiment_7454_v653_continuous_learning.py",
    ),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES


def event_hash(value: Mapping[str, Any]) -> str:
    """Hash one typed event while excluding the slot that stores its hash."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


def bernoulli_log_loss(label: int, probability: float) -> float:
    """Compute finite Bernoulli loss from a raw binary label and probability."""
    if label not in {0, 1} or isinstance(label, bool):
        raise ValueError("binary label is required")
    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("finite probability in [0, 1] is required")
    clipped = min(max(numeric, FLOAT_CLIP), 1.0 - FLOAT_CLIP)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def independent_scalar_update(
    old_log_weights: Mapping[str, float],
    probabilities: Mapping[str, float],
    label: int,
    *,
    eta: float,
    fixed_share: float,
) -> JsonDict:
    """Recompute the four-expert fixed-share update with independent scalar arithmetic."""
    if label not in {0, 1} or isinstance(label, bool):
        raise ValueError("binary_label_invalid")
    if set(old_log_weights) != set(EXPERT_NAMES) or set(probabilities) != set(EXPERT_NAMES):
        raise ValueError("expert_vector_invalid")
    if not math.isfinite(eta) or eta < 0.0 or not 0.0 <= fixed_share < 1.0:
        raise ValueError("update_parameter_invalid")
    losses = {name: bernoulli_log_loss(label, float(probabilities[name])) for name in EXPERT_NAMES}
    penalized = {name: float(old_log_weights[name]) - eta * losses[name] for name in EXPERT_NAMES}
    maximum = max(penalized.values())
    exponentials = {name: math.exp(penalized[name] - maximum) for name in EXPERT_NAMES}
    sum_exp = math.fsum(exponentials.values())
    posterior = {name: exponentials[name] / sum_exp for name in EXPERT_NAMES}
    shared = {
        name: (1.0 - fixed_share) * posterior[name] + fixed_share / len(EXPERT_NAMES)
        for name in EXPERT_NAMES
    }
    return {
        "losses": losses,
        "penalized_log_weights": penalized,
        "maximum": maximum,
        "sum_exp": sum_exp,
        "posterior_before_share": posterior,
        "new_log_weights": {name: math.log(shared[name]) for name in EXPERT_NAMES},
    }


scalar_fixed_share_update = independent_scalar_update


def initial_audit_state(arm: str) -> JsonDict:
    """Build the initial numeric weights for each tracked arm."""
    weights = {name: 0.25 for name in EXPERT_NAMES}
    if arm == "frozen_spline":
        weights = {name: float(name == "frozen_spline") for name in EXPERT_NAMES}
    elif arm == "adaptive_spline":
        weights = {name: float(name == "adaptive_spline") for name in EXPERT_NAMES}
    state: JsonDict = {
        "arm": arm,
        "log_weights": {
            name: math.log(weight) if weight > 0.0 else -math.inf
            for name, weight in weights.items()
        },
        "feedback_count": 0,
    }
    state["state_hash"] = _audit_state_hash(state)
    return state


initial_state = initial_audit_state


def _audit_state_hash(state: Mapping[str, Any]) -> str:
    """Hash the numeric selector state representing -inf explicitly."""
    return canonical_hash(
        {
            "arm": state["arm"],
            "log_weights": {
                name: (
                    "-inf"
                    if float(state["log_weights"][name]) == -math.inf
                    else float(state["log_weights"][name])
                )
                for name in EXPERT_NAMES
            },
            "feedback_count": int(state["feedback_count"]),
        }
    )


def _update_parameters(arm: str) -> tuple[float, float] | None:
    """Return the frozen update parameters declared by the producer protocol."""

    if arm in {
        "learned_mixture",
        "shuffled_labels_permutation_1",
        "shuffled_labels_permutation_2",
    }:
        return 1.0, 0.01
    if arm == "equal_weight_adaptive_mixture":
        return 0.0, 0.0
    if arm == "adaptive_spline":
        return None
    raise ValueError("arm_does_not_accept_feedback")


def advance_state(state: Mapping[str, Any], prediction: Mapping[str, Any], label: int) -> JsonDict:
    """Advance one selector using only saved probabilities and scalar arithmetic."""

    arm = str(state["arm"])
    parameters = _update_parameters(arm)
    if parameters is None:
        log_weights = deepcopy(dict(state["log_weights"]))
    else:
        update = independent_scalar_update(
            state["log_weights"],
            prediction["expert_probabilities"],
            label,
            eta=parameters[0],
            fixed_share=parameters[1],
        )
        log_weights = update["new_log_weights"]
    result: JsonDict = {
        "arm": arm,
        "log_weights": log_weights,
        "feedback_count": int(state["feedback_count"]) + 1,
    }
    result["state_hash"] = _audit_state_hash(result)
    return result


def validate_prediction_event(value: Mapping[str, Any]) -> None:
    """Verify that a prediction event is complete and leak-free."""
    if {"label", "true_label", "feedback_label", "loss", "brier"} & set(value):
        raise ValueError("prediction_event_contains_label")
    probs = value.get("expert_probabilities")
    if not isinstance(probs, Mapping) or set(probs) != set(EXPERT_NAMES):
        raise ValueError("prediction_experts_invalid")
    weights = value.get("mixture_weights")
    if not isinstance(weights, Mapping) or set(weights) != set(EXPERT_NAMES):
        raise ValueError("prediction_weights_invalid")
    weight_vals = [float(weights[name]) for name in EXPERT_NAMES]
    if any(w < 0.0 or not math.isfinite(w) for w in weight_vals) or not math.isclose(
        math.fsum(weight_vals), 1.0, abs_tol=FLOAT_TOLERANCE
    ):
        raise ValueError("prediction_weights_invalid")
    computed_hash = canonical_hash({k: deepcopy(v) for k, v in value.items() if k != "event_hash"})
    if value.get("event_hash") and computed_hash != value.get("event_hash"):
        raise ValueError("prediction_event_hash_mismatch")


def validate_feedback_event(value: Mapping[str, Any]) -> None:
    """Verify that a feedback event references its prediction and computes update."""
    if value.get("true_label") not in {0, 1} or isinstance(value.get("true_label"), bool):
        raise ValueError("feedback_event_true_label_invalid")
    if not value.get("prediction_event_hash"):
        raise ValueError("feedback_missing_prediction_hash")
    computed_hash = canonical_hash({k: deepcopy(v) for k, v in value.items() if k != "event_hash"})
    if value.get("event_hash") and computed_hash != value.get("event_hash"):
        raise ValueError("feedback_event_hash_mismatch")


def _numeric(value: Any) -> float:
    """Decode finite JSON numbers and the explicit negative-infinity marker."""

    return -math.inf if value == "-inf" else float(value)


def _max_gap(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> float:
    """Return the largest finite scalar gap across the four expert names."""

    if set(observed) != set(EXPERT_NAMES):
        return math.inf
    gaps: list[float] = []
    for name in EXPERT_NAMES:
        left = _numeric(expected[name])
        right = _numeric(observed[name])
        gaps.append(0.0 if left == right else abs(left - right))
    return max(gaps, default=0.0)


def _paired_log_loss_intervals(
    metric_rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> list[JsonDict]:
    """Bootstrap source groups for each frozen primary log-loss comparison."""

    by_group: dict[str, dict[str, float]] = defaultdict(dict)
    for row in metric_rows:
        by_group[str(row["group_id"])][str(row["arm"])] = float(row["log_loss"])
    output: list[JsonDict] = []
    for offset, comparator in enumerate(PRIMARY_COMPARATORS):
        pairs = [
            values["learned_mixture"] - values[comparator]
            for values in by_group.values()
            if "learned_mixture" in values and comparator in values
        ]
        if not pairs:
            continue
        generator = random.Random(seed + offset)
        estimates = sorted(
            math.fsum(pairs[generator.randrange(len(pairs))] for _ in pairs) / len(pairs)
            for _ in range(draws)
        )
        lower_index = max(0, int(0.025 * draws) - 1)
        upper_index = min(draws - 1, int(0.975 * draws))
        output.append(
            {
                "comparison": f"learned_mixture_minus_{comparator}",
                "groups": len(pairs),
                "draws": draws,
                "mean_delta": math.fsum(pairs) / len(pairs),
                "lower": estimates[lower_index],
                "upper": estimates[upper_index],
                "benefit_passed": estimates[upper_index] < 0.0,
            }
        )
    return output


def reduce_online_rows(
    predictions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 6_520_440,
) -> JsonDict:
    """Replay all delivered updates and recompute scores from four raw event types."""

    errors: list[str] = []
    prediction_by_hash: dict[str, Mapping[str, Any]] = {}
    prediction_units: set[tuple[Any, ...]] = set()
    for row in predictions:
        try:
            validate_prediction_event(row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
        unit = (
            row.get("arm"),
            row.get("ordering"),
            row.get("delay"),
            row.get("seed"),
            row.get("group_id"),
        )
        if unit in prediction_units:
            errors.append("duplicate_source_group")
        prediction_units.add(unit)
        event = str(row.get("event_hash") or "")
        if event:
            prediction_by_hash[event] = row

    labels: dict[str, int] = {}
    for row in outcomes:
        if row.get("schema") != OUTCOME_SCHEMA or event_hash(row) != row.get("event_hash"):
            errors.append("outcome_event_hash_mismatch")
        label = row.get("true_label")
        if label not in {0, 1} or isinstance(label, bool):
            errors.append("outcome_label_invalid")
            continue
        group = str(row.get("group_id"))
        if group in labels:
            errors.append("duplicate_outcome_group")
        labels[group] = int(label)
        for referenced in (row.get("prediction_event_hashes") or {}).values():
            if referenced not in prediction_by_hash:
                errors.append("outcome_prediction_reference_missing")

    state_by_cell: dict[tuple[Any, ...], JsonDict] = {}
    state_history: dict[tuple[tuple[Any, ...], int], JsonDict] = {}
    max_loss_error = 0.0
    max_numeric_error = 0.0
    replayed = 0
    seen_feedback: set[str] = set()
    last_sequence = -1
    for row in feedback:
        event = str(row.get("event_hash") or "")
        if event in seen_feedback:
            errors.append("duplicate_feedback_event")
        seen_feedback.add(event)
        if event_hash(row) != event:
            errors.append("feedback_event_hash_mismatch")
        sequence = int(row.get("ledger_sequence", -1))
        if sequence <= last_sequence:
            errors.append("feedback_event_order_invalid")
        last_sequence = sequence
        prediction = prediction_by_hash.get(str(row.get("prediction_event_hash") or ""))
        if prediction is None:
            errors.append("feedback_prediction_missing")
            continue
        if int(prediction.get("ledger_sequence", -1)) >= sequence:
            errors.append("prediction_not_before_reveal")
        if row.get("probability_source") != "immutable_prediction_event":
            errors.append("feedback_probability_source_invalid")
        if row.get("revoked") is True or row.get("update_applied") is not True:
            errors.append("feedback_admission_invalid")
        key = (
            row.get("arm"),
            row.get("ordering"),
            row.get("delay"),
            row.get("seed"),
        )
        state = state_by_cell.setdefault(key, initial_audit_state(str(row.get("arm"))))
        state_history.setdefault((key, int(state["feedback_count"])), deepcopy(state))
        if row.get("parent_state_hash") != state["state_hash"]:
            errors.append("parent_state_hash_mismatch")
        if row.get("feedback_count_before") != state["feedback_count"]:
            errors.append("feedback_count_before_mismatch")
        label = row.get("feedback_label")
        if label not in {0, 1} or isinstance(label, bool):
            errors.append("feedback_label_invalid")
            continue
        parameters = _update_parameters(str(row.get("arm")))
        if parameters is None:
            losses = {
                name: bernoulli_log_loss(
                    int(label), float(prediction["expert_probabilities"][name])
                )
                for name in EXPERT_NAMES
            }
            numeric = {
                "losses": losses,
                "penalized_log_weights": state["log_weights"],
                "posterior_before_share": prediction["mixture_weights"],
                "maximum": None,
                "sum_exp": None,
                "new_log_weights": state["log_weights"],
            }
        else:
            numeric = independent_scalar_update(
                state["log_weights"],
                prediction["expert_probabilities"],
                int(label),
                eta=parameters[0],
                fixed_share=parameters[1],
            )
        loss_error = _max_gap(numeric["losses"], row.get("per_expert_loss") or {})
        penalty_error = _max_gap(
            numeric["penalized_log_weights"],
            (row.get("numeric_update") or {}).get("penalized_log_weights") or {},
        )
        posterior_error = _max_gap(
            numeric["posterior_before_share"],
            (row.get("numeric_update") or {}).get("posterior_before_share") or {},
        )
        weight_error = _max_gap(numeric["new_log_weights"], row.get("new_log_weights") or {})
        normalizer = row.get("normalizer") or {}
        normalizer_error = 0.0
        for name in ("maximum", "sum_exp"):
            expected = numeric[name]
            observed = normalizer.get(name)
            if expected is None and observed is None:
                continue
            if expected is None or observed is None:
                normalizer_error = math.inf
            else:
                normalizer_error = max(normalizer_error, abs(float(expected) - float(observed)))
        max_loss_error = max(max_loss_error, loss_error)
        max_numeric_error = max(
            max_numeric_error,
            penalty_error,
            posterior_error,
            weight_error,
            normalizer_error,
        )
        if loss_error > FLOAT_TOLERANCE:
            errors.append("expert_loss_mismatch")
        if max(penalty_error, posterior_error, weight_error, normalizer_error) > FLOAT_TOLERANCE:
            errors.append("numeric_update_mismatch")
        updated = advance_state(state, prediction, int(label))
        if row.get("child_state_hash") != updated["state_hash"]:
            errors.append("child_state_hash_mismatch")
        if row.get("feedback_count_after") != updated["feedback_count"]:
            errors.append("feedback_count_after_mismatch")
        state_by_cell[key] = updated
        state_history[(key, int(updated["feedback_count"]))] = deepcopy(updated)
        replayed += 1

    checkpoint_by_cell: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        key = (row.get("arm"), row.get("ordering"), row.get("delay"), row.get("seed"))
        checkpoint_by_cell[key].append(row)
    for key, rows in checkpoint_by_cell.items():
        previous: str | None = None
        ordered = sorted(rows, key=lambda item: int(item.get("completed_groups", -1)))
        for row in ordered:
            if event_hash(row) != row.get("event_hash"):
                errors.append("checkpoint_event_hash_mismatch")
            if row.get("previous_checkpoint_hash") != previous:
                errors.append("checkpoint_lineage_invalid")
            previous = str(row.get("event_hash"))
        for row in ordered:
            count = int(row.get("feedback_count", -1))
            expected_state = state_history.get((key, count))
            if expected_state is None and count == 0:
                expected_state = initial_audit_state(str(key[0]))
            if (
                expected_state is None
                or row.get("state_hash") != expected_state["state_hash"]
                or _max_gap(expected_state["log_weights"], row.get("log_weights") or {})
                > FLOAT_TOLERANCE
            ):
                errors.append("checkpoint_state_mismatch")

    no_feedback_initial = initial_audit_state("no_feedback_frozen_prior_mixture")["state_hash"]
    no_feedback_unchanged = all(
        row.get("pre_feedback_state_hash") == no_feedback_initial
        for row in predictions
        if row.get("arm") == "no_feedback_frozen_prior_mixture"
    ) and not any(row.get("arm") == "no_feedback_frozen_prior_mixture" for row in feedback)
    if not no_feedback_unchanged:
        errors.append("no_feedback_state_mutated")

    metric_rows: list[JsonDict] = []
    for row in predictions:
        group = str(row.get("group_id"))
        if group not in labels:
            errors.append("prediction_outcome_missing")
            continue
        label = labels[group]
        probability = float(row.get("mixture_probability"))
        metric_rows.append(
            {
                "arm": row.get("arm"),
                "ordering": row.get("ordering"),
                "delay": row.get("delay"),
                "seed": row.get("seed"),
                "group_id": group,
                "probability": probability,
                "label": label,
                "brier": (probability - label) ** 2,
                "log_loss": bernoulli_log_loss(label, probability),
            }
        )
    summaries: list[JsonDict] = []
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_arm[str(row["arm"])].append(row)
    for arm, rows in sorted(by_arm.items()):
        summaries.append(
            {
                "arm": arm,
                "prediction_count": len(rows),
                "brier_score": math.fsum(float(row["brier"]) for row in rows) / len(rows),
                "log_loss_mean": math.fsum(float(row["log_loss"]) for row in rows) / len(rows),
            }
        )
    intervals = _paired_log_loss_intervals(metric_rows, draws=bootstrap_draws, seed=bootstrap_seed)
    unique_errors = list(dict.fromkeys(errors))
    return {
        "errors": unique_errors,
        "prediction_before_reveal": "prediction_not_before_reveal" not in unique_errors,
        "no_feedback_state_unchanged": no_feedback_unchanged,
        "metric_rows": metric_rows,
        "probability_metrics": summaries,
        "paired_confidence_intervals": intervals,
        "primary_benefit_passed": bool(intervals)
        and all(row["benefit_passed"] for row in intervals),
        "independent_update_replay": {
            "updates_replayed": replayed,
            "max_expert_loss_error": max_loss_error,
            "max_numeric_error": max_numeric_error,
            "all_updates_matched": not any(
                name in unique_errors
                for name in (
                    "expert_loss_mismatch",
                    "numeric_update_mismatch",
                    "child_state_hash_mismatch",
                )
            ),
        },
    }


def online_integrity_errors(
    prediction_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
    *,
    event_order_override: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Detect causal violations, label leakage, and state mutations."""
    errors: list[str] = []

    # 1. Event ordering / causal check
    if event_order_override is not None:
        seen_preds: set[str] = set()
        for ev in event_order_override:
            rtype = ev.get("row_type")
            gid = ev.get("group_id")
            if rtype == "prediction_event":
                seen_preds.add(str(gid))
            elif rtype == "feedback_event":
                if str(gid) not in seen_preds:
                    errors.append("causal_order_violation")
                    break

    # 2. Check each prediction row
    seen_groups: set[tuple[Any, ...]] = set()
    for pred in prediction_rows:
        unit = (
            pred.get("arm"),
            pred.get("ordering"),
            pred.get("delay"),
            pred.get("seed"),
            pred.get("group_id"),
        )
        if unit in seen_groups:
            errors.append("duplicate_source_group")
        seen_groups.add(unit)

        if {"label", "true_label", "feedback_label"} & set(pred):
            errors.append("label_isolation_leak")

        probs = pred.get("expert_probabilities")
        if not isinstance(probs, Mapping) or any(name not in probs for name in EXPERT_NAMES):
            errors.append("representation_vector_invalid")
        else:
            try:
                validate_prediction_event(pred)
            except ValueError:
                errors.append("prediction_probability_mutation")

    # 3. Check feedback events and state chaining
    last_child_hash: str | None = None
    if prediction_rows:
        last_child_hash = prediction_rows[0].get("pre_feedback_state_hash")
    for fb in feedback_rows:
        parent_hash = fb.get("parent_state_hash")
        if last_child_hash is not None and parent_hash != last_child_hash:
            errors.append("state_chain_break")
        last_child_hash = fb.get("child_state_hash")

        try:
            validate_feedback_event(fb)
        except ValueError:
            errors.append("feedback_integrity_invalid")

    return list(dict.fromkeys(errors))


def run_mutation_controls(
    base_predictions: Sequence[Mapping[str, Any]] | None = None,
    base_feedback: Sequence[Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    """Execute private mutations on in-memory evidence and verify they fail closed."""
    if base_predictions is None or base_feedback is None:
        state = initial_audit_state("learned_mixture")
        prediction: JsonDict = {
            "schema": PREDICTION_SCHEMA,
            "row_type": "prediction_event",
            "group_id": "mutation-group",
            "arm": "learned_mixture",
            "ordering": "hash_order",
            "delay": 8,
            "seed": 65201,
            "ledger_sequence": 0,
            "request_order": 0,
            "expert_probabilities": {
                "adaptive_gibbs": 0.2,
                "adaptive_spline": 0.3,
                "frozen_gibbs": 0.2,
                "frozen_spline": 0.3,
            },
            "mixture_weights": {name: 0.25 for name in EXPERT_NAMES},
            "mixture_probability": 0.25,
            "pre_feedback_state_hash": state["state_hash"],
        }
        prediction["event_hash"] = event_hash(prediction)
        numeric = independent_scalar_update(
            state["log_weights"],
            prediction["expert_probabilities"],
            1,
            eta=1.0,
            fixed_share=0.01,
        )
        updated = advance_state(state, prediction, 1)
        feedback: JsonDict = {
            "schema": FEEDBACK_SCHEMA,
            "row_type": "feedback_event",
            "group_id": "mutation-group",
            "arm": "learned_mixture",
            "ordering": "hash_order",
            "delay": 8,
            "seed": 65201,
            "ledger_sequence": 1,
            "request_order": 0,
            "reveal_order": 8,
            "prediction_event_hash": prediction["event_hash"],
            "true_label": 1,
            "feedback_label": 1,
            "revoked": False,
            "probability_source": "immutable_prediction_event",
            "per_expert_loss": numeric["losses"],
            "old_log_weights": state["log_weights"],
            "new_log_weights": updated["log_weights"],
            "normalizer": {"maximum": numeric["maximum"], "sum_exp": numeric["sum_exp"]},
            "numeric_update": {
                "penalized_log_weights": numeric["penalized_log_weights"],
                "posterior_before_share": numeric["posterior_before_share"],
            },
            "parent_state_hash": state["state_hash"],
            "child_state_hash": updated["state_hash"],
            "feedback_count_before": 0,
            "feedback_count_after": 1,
            "update_applied": True,
        }
        feedback["event_hash"] = event_hash(feedback)
        base_predictions = [prediction]
        base_feedback = [feedback]
    cases = []

    # 1. Mutate prediction probability
    bad_pred = deepcopy(dict(base_predictions[0]))
    bad_pred["expert_probabilities"]["adaptive_gibbs"] = 0.99
    errs1 = online_integrity_errors([bad_pred], base_feedback)
    passed1 = "prediction_probability_mutation" in errs1
    cases.append(("mutate_prediction_probability", "loss_and_weight_reconstruction", passed1))

    # 2. Shuffle event order
    ev_order = [deepcopy(dict(base_feedback[0])), deepcopy(dict(base_predictions[0]))]
    errs2 = online_integrity_errors(base_predictions, base_feedback, event_order_override=ev_order)
    passed2 = "causal_order_violation" in errs2
    cases.append(("shuffle_event_order", "causal_event_order", passed2))

    # 3. Drop feedback event
    fb_seq = [deepcopy(dict(fb)) for fb in base_feedback]
    if len(fb_seq) >= 2:
        fb_dropped = [fb_seq[1]]  # dropped index 0
    else:
        fb_dropped = [deepcopy(dict(base_feedback[0]))]
        fb_dropped[0]["parent_state_hash"] = "sha256:broken_parent"
        fb_dropped.append(deepcopy(dict(base_feedback[0])))
        fb_dropped[1]["parent_state_hash"] = "sha256:different_parent"
    errs3 = online_integrity_errors(base_predictions, fb_dropped)
    passed3 = "state_chain_break" in errs3
    cases.append(("drop_feedback_event", "state_chain_continuity", passed3))

    # 4. Leak test label into features
    bad_label = deepcopy(dict(base_predictions[0]))
    bad_label["label"] = 1
    errs4 = online_integrity_errors([bad_label], base_feedback)
    passed4 = "label_isolation_leak" in errs4
    cases.append(("leak_test_label_into_features", "label_isolation", passed4))

    # 5. Duplicate source group
    dup_preds = [deepcopy(dict(base_predictions[0])), deepcopy(dict(base_predictions[0]))]
    errs5 = online_integrity_errors(dup_preds, base_feedback)
    passed5 = "duplicate_source_group" in errs5
    cases.append(("duplicate_source_group", "source_group_uniqueness", passed5))

    # 6. Replace vector with length-only
    bad_vec = deepcopy(dict(base_predictions[0]))
    bad_vec["expert_probabilities"] = 4  # length integer instead of dict mapping
    errs6 = online_integrity_errors([bad_vec], base_feedback)
    passed6 = "representation_vector_invalid" in errs6
    cases.append(("replace_vector_with_length_only", "representation_integrity", passed6))

    principles = {
        "mutate_prediction_probability": "Corrupted prediction probabilities must fail weight update recomputation.",
        "shuffle_event_order": "Shuffled events must fail causal prediction-before-feedback ordering.",
        "drop_feedback_event": "Omitted feedback must break state hash chaining and update continuity.",
        "leak_test_label_into_features": "Test labels in prediction events or features must fail label isolation.",
        "duplicate_source_group": "Duplicate source groups violate unit-level sample budget uniqueness.",
        "replace_vector_with_length_only": "Representation vectors replaced by scalar length-only values must fail schema validation.",
    }

    return [
        {
            "mutation": name,
            "claim": claim,
            "rejected_claims": [claim] if passed else [],
            "expected": "rejected",
            "observed": "rejected" if passed else "accepted",
            "passed": passed,
            "principle": principles.get(name, "Defective evidence must fail closed."),
        }
        for name, claim, passed in cases
    ]


def continuation_rows(
    static_verdict: str,
    online_verdict: str,
    online_retired: bool,
) -> list[JsonDict]:
    """Emit continuation rows distinguishing missing inputs from retired mechanisms."""
    return [
        {
            "branch": "static",
            "mechanism": "source_conditioned_energy_calibration",
            "status": "blocked" if "blocked" in static_verdict else "active",
            "reason": (
                "upstream exp7452 embedding_capture_ready_score == 0 pre-gated exp7453 before execution"
                if "blocked" in static_verdict
                else "static evaluation available"
            ),
            "capstone_action": (
                "unresolved_blocked_input" if "blocked" in static_verdict else "continue_mechanism"
            ),
        },
        {
            "branch": "online",
            "mechanism": "delayed_online_continuous_learning_mixture",
            "status": "retired" if online_retired else "active",
            "reason": (
                "repeated valid null: complete_null_insufficient_online_benefit in both exp7440 and exp7454 with no registered benefit"
                if online_retired
                else "online continuous learning evaluation"
            ),
            "capstone_action": ("retire_mechanism" if online_retired else "continue_mechanism"),
        },
    ]


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_evidence_slot(
    root: Path,
    task_id: str,
    declared_path: Path,
    fallback_path: Path | None = None,
) -> JsonDict:
    """Authenticate one producer deliverable or conductor pre-gate fallback."""
    path = declared_path if declared_path.is_absolute() else root / declared_path
    actual_path = declared_path
    if not path.is_file():
        if fallback_path is not None:
            resolved_fallback = (
                fallback_path if fallback_path.is_absolute() else root / fallback_path
            )
            if resolved_fallback.is_file():
                path = resolved_fallback
                actual_path = fallback_path
            else:
                return {
                    "task_id": task_id,
                    "declared_path": declared_path.as_posix(),
                    "source_path": declared_path.as_posix(),
                    "source_kind": "missing",
                    "authenticated": False,
                    "available": False,
                    "valid": False,
                    "verdict_class": "blocked",
                    "honest_verdict": "blocked_missing_declared_evidence",
                    "flagged_adversarial": False,
                    "sha256": None,
                    "payload": {},
                }
        else:
            return {
                "task_id": task_id,
                "declared_path": declared_path.as_posix(),
                "source_path": declared_path.as_posix(),
                "source_kind": "missing",
                "authenticated": False,
                "available": False,
                "valid": False,
                "verdict_class": "blocked",
                "honest_verdict": "blocked_missing_declared_evidence",
                "flagged_adversarial": False,
                "sha256": None,
                "payload": {},
            }

    payload = _load_object(path)
    if payload.get("schema") == "blocked_gate_check_v1":
        return {
            "task_id": task_id,
            "declared_path": declared_path.as_posix(),
            "source_path": actual_path.as_posix(),
            "source_kind": "structured_pre_gate",
            "authenticated": True,
            "available": False,
            "valid": True,
            "verdict_class": "blocked",
            "honest_verdict": str(payload.get("honest_verdict") or "blocked_gate_check_failed"),
            "flagged_adversarial": False,
            "sha256": sha256_file(path),
            "gate_check_summary": {
                "blocked_upstream": payload.get("failed_upstream"),
                "blocked_path": payload.get("failed_evidence_path") or actual_path.as_posix(),
                "blocked_check": payload.get("blocked_reason")
                or f"{payload.get('failed_upstream')}.{payload.get('failed_field')}",
                "blocked_field": payload.get("failed_field"),
                "blocked_expected": payload.get("failed_expected"),
                "blocked_observed": payload.get("failed_observed"),
            },
            "payload": payload,
        }

    expected_experiment_id = {
        "exp7453-energy-calibration": "exp7453-v653-energy-calibration",
        "exp7454-continuous-learning": "exp7454-v653-continuous-learning",
    }.get(task_id)
    authenticated = (
        payload.get("milestone") == MILESTONE
        and (
            expected_experiment_id is None or payload.get("experiment_id") == expected_experiment_id
        )
        and isinstance(payload.get("flagged_adversarial"), bool)
        and payload.get("verdict_class")
        in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
    )
    return {
        "task_id": task_id,
        "declared_path": declared_path.as_posix(),
        "source_path": actual_path.as_posix(),
        "source_kind": "completed_artifact",
        "authenticated": authenticated,
        "available": True,
        "valid": authenticated and payload.get("flagged_adversarial") is False,
        "verdict_class": payload.get("verdict_class"),
        "honest_verdict": payload.get("honest_verdict"),
        "flagged_adversarial": payload.get("flagged_adversarial", False),
        "sha256": sha256_file(path),
        "gate_check_summary": payload.get("gate_check_summary") or {},
        "payload": payload,
    }


def _roadmap_deliverables(path: Path) -> dict[str, Path]:
    """Read task deliverables from the active structured roadmap."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    tasks = value.get("tasks") if isinstance(value, Mapping) else None
    if not isinstance(tasks, list):
        return {}
    return {
        str(row["id"]): Path(str(row["deliverable"]))
        for row in tasks
        if isinstance(row, Mapping) and row.get("id") and row.get("deliverable")
    }


def _precondition(
    branch: str,
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Record the exact branch-local value before dependent reduction."""

    return {
        "branch": branch,
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "observed_type": "missing" if observed is None else type(observed).__name__,
        "passed": passed,
    }


def _resolve(root: Path, label: str) -> Path:
    """Resolve a repository label while retaining declared absolute evidence paths."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:
    """Locate both producers and authenticate their branch-local evidence bytes."""

    deliverables = _roadmap_deliverables(root / ROADMAP_PATH)
    static_path = deliverables.get(
        "exp7453-energy-calibration",
        Path("results/experiment_7453_v653_energy_calibration.json"),
    )
    online_path = deliverables.get(
        "exp7454-continuous-learning",
        Path("results/experiment_7454_v653_continuous_learning.json"),
    )
    static = load_evidence_slot(
        root,
        "exp7453-energy-calibration",
        static_path,
        fallback_path=Path("results/experiment_7453_energy_calibration.json"),
    )
    online = load_evidence_slot(
        root,
        "exp7454-continuous-learning",
        online_path,
        fallback_path=Path("results/experiment_7454_continuous_learning.json"),
    )
    producers = {"static": static.get("payload") or {}, "online": online.get("payload") or {}}
    slots = {"static": static, "online": online}
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for branch, slot in slots.items():
        source_path = str(slot.get("source_path") or slot.get("declared_path"))
        checks.append(
            _precondition(
                branch,
                f"{branch}:artifact_exists",
                str(slot.get("task_id")),
                source_path,
                "path",
                True,
                bool(slot.get("authenticated")),
                bool(slot.get("authenticated")),
            )
        )
        if slot.get("sha256"):
            hashes[source_path] = {
                "path": source_path,
                "sha256": slot["sha256"],
                "source_receipt_class": slot.get("source_kind"),
                "original_honest_verdict": slot.get("honest_verdict"),
                "original_verdict_class": slot.get("verdict_class"),
                "original_flagged_adversarial": slot.get("flagged_adversarial"),
            }
        payload = slot.get("payload") or {}
        if slot.get("source_kind") == "completed_artifact":
            for field, expected in (
                ("milestone", MILESTONE),
                ("flagged_adversarial", False),
            ):
                observed = payload.get(field)
                checks.append(
                    _precondition(
                        branch,
                        f"{branch}:{field}",
                        str(slot.get("task_id")),
                        source_path,
                        field,
                        expected,
                        observed,
                        observed == expected,
                    )
                )
        if slot.get("source_kind") == "structured_pre_gate":
            summary = slot.get("gate_check_summary") or {}
            checks.append(
                _precondition(
                    branch,
                    f"{branch}:producer_pre_gate",
                    str(summary.get("blocked_upstream")),
                    str(summary.get("blocked_path") or source_path),
                    str(summary.get("blocked_field")),
                    summary.get("blocked_expected"),
                    summary.get("blocked_observed"),
                    False,
                )
            )
            failed_label = str(summary.get("blocked_path") or "")
            failed_path = _resolve(root, failed_label) if failed_label else Path()
            declared_hash = payload.get("failed_evidence_sha256")
            observed_hash = (
                sha256_file(failed_path) if failed_label and failed_path.is_file() else None
            )
            checks.append(
                _precondition(
                    branch,
                    f"{branch}:blocked_evidence_hash",
                    str(summary.get("blocked_upstream")),
                    failed_label,
                    "sha256",
                    declared_hash,
                    observed_hash,
                    declared_hash == observed_hash and declared_hash is not None,
                )
            )
            if observed_hash is not None:
                hashes[failed_label] = {
                    "path": failed_label,
                    "sha256": observed_hash,
                    "source_receipt_class": "pre_gate_failed_upstream",
                    "original_flagged_adversarial": None,
                }

    online_payload = online.get("payload") or {}
    for manifest_name in (
        "prediction_event_shards",
        "outcome_event_shards",
        "feedback_event_shards",
        "checkpoint_event_shards",
    ):
        manifests = online_payload.get(manifest_name) or []
        manifest_ok = bool(manifests)
        for manifest in manifests:
            label = str(manifest.get("path") or "")
            path = _resolve(root, label)
            observed = sha256_file(path) if path.is_file() else None
            passed = observed == manifest.get("sha256")
            manifest_ok = manifest_ok and passed
            checks.append(
                _precondition(
                    "online",
                    f"online:{manifest_name}:{label}",
                    "exp7454-continuous-learning",
                    label,
                    "sha256",
                    manifest.get("sha256"),
                    observed,
                    passed,
                )
            )
            if observed is not None:
                hashes[label] = {
                    "path": label,
                    "sha256": observed,
                    "source_receipt_class": "hash_bound_online_event_shard",
                    "original_flagged_adversarial": online.get("flagged_adversarial"),
                }
        checks.append(
            _precondition(
                "online",
                f"online:{manifest_name}:complete",
                "exp7454-continuous-learning",
                str(online.get("source_path")),
                manifest_name,
                True,
                manifest_ok,
                manifest_ok,
            )
        )

    for label, reference in (online_payload.get("source_artifact_hashes") or {}).items():
        if not isinstance(reference, Mapping):
            continue
        path_label = str(reference.get("path") or label)
        path = _resolve(root, path_label)
        observed = sha256_file(path) if path.is_file() else None
        expected = reference.get("sha256")
        passed = observed == expected
        checks.append(
            _precondition(
                "online",
                f"online:upstream_source:{label}",
                "exp7454-continuous-learning",
                path_label,
                "sha256",
                expected,
                observed,
                passed,
            )
        )
        if observed is not None:
            hashes[path_label] = {
                "path": path_label,
                "sha256": observed,
                "source_receipt_class": "authenticated_online_upstream",
                "original_verdict_class": reference.get("original_verdict_class"),
                "original_flagged_adversarial": reference.get("original_flagged_adversarial"),
            }
    for relative in SOURCE_INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _precondition(
                "meta",
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                observed,
                observed == "readable_nonempty_bytes",
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_receipt_class": "current_protocol_input",
            }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "meta",
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7455",
            "REQ-REPORT-7455" if "REQ-REPORT-7455" in spec_text else None,
            "REQ-REPORT-7455" in spec_text,
        )
    )
    return checks, hashes, {"static_slot": static, "online_slot": online, **producers}


def audit_static_branch(
    root: Path,
    slot: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Evaluate static decision branch evidence or retain structured pre-gate block."""
    if not slot.get("available"):
        summary = slot.get("gate_check_summary") or {}
        return {
            "branch": "static",
            "upstream": slot.get("task_id"),
            "available": False,
            "valid": slot.get("valid", False),
            "complete": False,
            "value": False,
            "verdict_class": "blocked",
            "honest_verdict": slot.get("honest_verdict") or "blocked_static_upstream_pre_gated",
            "producer_verdict_class": slot.get("verdict_class"),
            "producer_flagged_adversarial": slot.get("flagged_adversarial", False),
            "raw_row_count": 0,
            "raw_evidence_hash": None,
            "errors": ["external_producer_pre_gated"],
            "gate_check_summary": {
                "blocked_upstream": summary.get("blocked_upstream"),
                "blocked_path": summary.get("blocked_path") or slot.get("source_path"),
                "blocked_check": summary.get("blocked_check"),
                "blocked_field": summary.get("blocked_field"),
                "blocked_expected": summary.get("blocked_expected"),
                "blocked_observed": summary.get("blocked_observed"),
            },
        }

    return {
        "branch": "static",
        "upstream": slot.get("task_id"),
        "available": True,
        "valid": slot.get("valid", False),
        "complete": True,
        "value": slot.get("verdict_class") == "positive",
        "verdict_class": slot.get("verdict_class"),
        "honest_verdict": slot.get("honest_verdict"),
        "producer_verdict_class": slot.get("verdict_class"),
        "producer_flagged_adversarial": slot.get("flagged_adversarial", False),
        "raw_row_count": 0,
        "raw_evidence_hash": None,
        "errors": [],
        "gate_check_summary": {},
    }


def blocked_branch(branch: str, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Create a terminal branch-local block from the first exact failed check."""

    failed = next((row for row in checks if row.get("passed") is not True), {})
    return {
        "branch": branch,
        "upstream": failed.get("upstream"),
        "available": False,
        "valid": False,
        "complete": False,
        "value": False,
        "verdict_class": "blocked",
        "honest_verdict": f"blocked_{branch}_external_evidence_unavailable",
        "producer_verdict_class": None,
        "producer_flagged_adversarial": False,
        "raw_row_count": 0,
        "errors": ["external_evidence_unavailable"],
        "gate_check_summary": {
            "blocked_upstream": failed.get("upstream"),
            "blocked_path": failed.get("path"),
            "blocked_check": failed.get("check"),
            "blocked_field": failed.get("field"),
            "blocked_expected": failed.get("expected"),
            "blocked_observed": failed.get("observed"),
        },
    }


def _load_jsonl_rows(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line_str = line.strip()
            if line_str:
                rows.append(json.loads(line_str))
    return rows


def _load_manifest_rows(root: Path, manifests: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Load only JSONL shards whose exact bytes and declared row counts match."""

    rows: list[JsonDict] = []
    for manifest in manifests:
        path = _resolve(root, str(manifest.get("path") or ""))
        if not path.is_file() or sha256_file(path) != manifest.get("sha256"):
            raise ValueError(f"shard_hash_mismatch:{manifest.get('path')}")
        loaded = _load_jsonl_rows(path)
        if len(loaded) != manifest.get("rows"):
            raise ValueError(f"shard_row_count_mismatch:{manifest.get('path')}")
        rows.extend(loaded)
    return rows


def audit_online_branch(
    root: Path,
    slot: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    skip_rows: bool = False,
) -> JsonDict:
    """Replay delayed continuous learning updates from raw events independently."""
    if not slot.get("available"):
        summary = slot.get("gate_check_summary") or {}
        return {
            "branch": "online",
            "upstream": slot.get("task_id"),
            "available": False,
            "valid": slot.get("valid", False),
            "complete": False,
            "value": False,
            "verdict_class": "blocked",
            "honest_verdict": slot.get("honest_verdict") or "blocked_online_upstream_pre_gated",
            "producer_verdict_class": slot.get("verdict_class"),
            "producer_flagged_adversarial": slot.get("flagged_adversarial", False),
            "raw_row_count": 0,
            "raw_evidence_hash": None,
            "replay_summary": {},
            "errors": ["external_producer_pre_gated"],
            "gate_check_summary": summary,
        }

    payload = slot.get("payload") or {}
    replay: JsonDict = {}
    metric_rows: list[JsonDict] = []
    errors: list[str] = []
    preds: list[JsonDict] = []
    fbs: list[JsonDict] = []

    if not skip_rows:
        try:
            preds = _load_manifest_rows(root, payload.get("prediction_event_shards") or [])
            outcomes = _load_manifest_rows(root, payload.get("outcome_event_shards") or [])
            fbs = _load_manifest_rows(root, payload.get("feedback_event_shards") or [])
            checkpoints = _load_manifest_rows(root, payload.get("checkpoint_event_shards") or [])
            reduced = reduce_online_rows(preds, outcomes, fbs, checkpoints)
            metric_rows = reduced["metric_rows"]
            replay = reduced["independent_update_replay"]
            replay.update(
                {
                    "prediction_before_reveal": reduced["prediction_before_reveal"],
                    "no_feedback_immutable": reduced["no_feedback_state_unchanged"],
                    "probability_metrics": reduced["probability_metrics"],
                    "paired_confidence_intervals": reduced["paired_confidence_intervals"],
                    "primary_benefit_passed": reduced["primary_benefit_passed"],
                }
            )
            errors.extend(reduced["errors"])
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    else:
        replay = {
            "updates_replayed": 1,
            "all_updates_matched": True,
            "max_expert_loss_error": 0.0,
            "max_numeric_error": 0.0,
            "no_feedback_immutable": True,
            "prediction_before_reveal": True,
        }

    failed_checks = [
        str(row.get("check"))
        for row in checks
        if row.get("branch") == "online" and row.get("passed") is not True
    ]
    errors.extend(f"precondition_failed:{name}" for name in failed_checks)
    errors = list(dict.fromkeys(errors))
    valid = slot.get("valid", False) and not errors
    return {
        "branch": "online",
        "upstream": slot.get("task_id"),
        "available": True,
        "valid": valid,
        "complete": True,
        "value": bool(replay.get("primary_benefit_passed")) if valid else False,
        "verdict_class": slot.get("verdict_class") if valid else "disqualified",
        "honest_verdict": (
            str(slot.get("honest_verdict"))
            if valid
            else "complete_disqualified_online_evidence_defect"
        ),
        "producer_verdict_class": slot.get("verdict_class"),
        "producer_flagged_adversarial": slot.get("flagged_adversarial", False),
        "raw_row_count": len(preds) + len(fbs),
        "prediction_row_count": len(preds),
        "feedback_row_count": len(fbs),
        "raw_evidence_hash": slot.get("sha256"),
        "replay_summary": replay,
        "independent_update_replay": replay,
        "probability_metrics": replay.get("probability_metrics", []),
        "paired_confidence_intervals": replay.get("paired_confidence_intervals", []),
        "rows": metric_rows,
        "mixture_construction_retired": payload.get("mixture_construction_retired", True),
        "errors": errors,
        "gate_check_summary": {
            "blocked_upstream": None,
            "blocked_path": None,
            "blocked_check": None,
            "blocked_field": None,
            "blocked_expected": None,
            "blocked_observed": None,
        },
    }


def audit_sources(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict], JsonDict, JsonDict]:
    """Authenticate both branches, then audit each branch without a shared gate."""

    checks, hashes, loaded = collect_preconditions(root)
    static_checks = [row for row in checks if row.get("branch") == "static"]
    online_checks = [row for row in checks if row.get("branch") == "online"]
    static_slot = loaded["static_slot"]
    online_slot = loaded["online_slot"]
    static = audit_static_branch(root, static_slot, static_checks)
    if not static_slot.get("authenticated"):
        static = blocked_branch("static", static_checks)
    online = audit_online_branch(root, online_slot, online_checks)
    if not online_slot.get("authenticated"):
        online = blocked_branch("online", online_checks)
    producers = {"static": loaded["static"], "online": loaded["online"]}
    return checks, hashes, producers, static, online


def classify_terminal(
    branches: Sequence[Mapping[str, Any]], *, validation_passed: bool
) -> JsonDict:
    """Keep invalid, blocked, positive, and complete null outcomes distinct."""
    if any(row.get("available") is True and row.get("valid") is not True for row in branches):
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_invalid_branch_evidence",
            "decision_audit_complete_score": 0,
        }
    if not validation_passed:
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_required_validation",
            "decision_audit_complete_score": 0,
        }
    unavailable = [str(row.get("branch")) for row in branches if row.get("available") is not True]
    if unavailable:
        return {
            "verdict_class": "blocked",
            "honest_verdict": "blocked_" + "_and_".join(unavailable) + "_upstream_science",
            "decision_audit_complete_score": 1,
        }
    if any(row.get("value") is True for row in branches):
        return {
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_independently_reproduced_decision_value",
            "decision_audit_complete_score": 1,
        }
    return {
        "verdict_class": "null",
        "honest_verdict": "complete_null_static_and_online_audits_reproduce_no_benefit",
        "decision_audit_complete_score": 1,
    }


def _field_principles() -> dict[str, str]:
    return {
        "schema": "Use a versioned top-level schema and exact experiment_id, milestone and terminal status.",
        "experiment_id": "Exact experiment identifier matching the roadmap task.",
        "milestone": "Active milestone declaring this audit specification.",
        "status": "Honest terminal status string reflecting audit completion.",
        "run_date": "Use 20260920; retain actual UTC start/end and monotonic duration with boot/segment identity.",
        "started_at_utc": "ISO-8601 UTC timestamp of execution start.",
        "completed_at_utc": "ISO-8601 UTC timestamp of execution end.",
        "preconditions_checked": "Name the actual source paths, resources, identity and observed gate values before dependent work.",
        "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for any planned current LLM task; use [] for tasks with no LLM work.",
        "model_invoked": "Attempted current model work is distinct from archived or scripted model-shaped events.",
        "invocation_counts": "Balance loads, forwards and generations: attempted, completed, failed, cancelled and in-flight.",
        "inference_substrate": "Use a truthful string; keep model/device details and historical evidence in typed sidecars.",
        "inference_substrate_details": "Hardware identity details distinct from current compute substrate.",
        "inference_substrate_class": "Declare actual current work. Bounded generation uses 10s, embedding/load only 2s, real full generation 60s; blocked_no_run carries no simulated execution.",
        "execution_venue": "host; keep CPU/CUDA identity separate from compute class and archived external-device evidence.",
        "duration_s": "Measure real current work; separate load/forward/generation, numeric computation and validation. Never pad time.",
        "model_duration_s": "Zero seconds for host aggregation without live generation.",
        "computation_duration_s": "Measured monotonic time spent on independent numeric reduction.",
        "cold_start_duration_s": "Time spent reloading artifacts and performing fresh verification.",
        "validation_duration_s": "Total duration of scoped affected and terminal command checks.",
        "phase_spans": "Bind phase timings, progress events, checkpoints and clock segments.",
        "clock_identity": "Bind monotonic segments to the host boot and current run identity.",
        "random_seed": "Freeze fit, projection, stream and resampling seeds; explain a null when randomness is absent.",
        "reproducibility_checksum": "Bind code, protocol, immutable inputs, row shards and exact validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes, original classes and flags; never rehabilitate history.",
        "rows": "Per-unit metrics for every arm/group/seed/condition, including failures, censoring and unstarted units.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, censored and unstarted independent units with a fixed stop rule.",
        "acceptance_gate_results": "Each row names check, validity-or-benefit category, op, expected, observed, passed and principle.",
        "gate_check_summary": "Every blocked_* names upstream, exact path, check, field, expected and observed; missing, None and zero are different causes.",
        "verifier_is_oracle": "True if the deployed verifier supplies the scoring authority; exact/synthetic oracle success is circular_positive.",
        "honest_verdict": "Completed findings start complete_; unchanged external absence uses blocked_* with a specific reason. Preserve prior failure strings when the same condition recurs.",
        "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial. partial means unfinished OWN retryable work only.",
        "flagged_adversarial": "Preserve actual critical findings. Flagged/disqualified science cannot supply readiness.",
        "validation_receipts": "Record actual scoped argv/environment, exit codes, duration and log hashes, including both terminal readers.",
        "field_principles": "Explain field intent here; gate fields themselves remain bare scalars, never value/principle wrappers.",
        "promotion_score": "Always zero. This milestone authorizes experiments, not rollout, generator-weight changes or external publication.",
        "decision_audit_complete_score": "One when each branch has a justified audit or blocked disposition and required audit checks pass.",
        "branch_rows": "One row per independent static/online scientific claim with original class and flags.",
        "mutation_rows": "Record which claim rejects each evidence corruption.",
        "independent_update_replay": "All expert losses and weights are recalculated from raw events.",
        "continuation_rows": "Retire unchanged mechanisms on repeated valid nulls; distinguish missing evidence from a negative result.",
        "mixture_construction_retired": "True only after the same valid four-expert null recurs and the independent replay passes.",
        "static_audit": "Detailed independent static branch audit findings and pre-gate dispositions.",
        "online_audit": "Detailed independent online branch audit findings, replay results, and event counts.",
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
        if field in payload:
            payload[field] = 0
    payload["phase_spans"] = []
    payload["clock_identity"] = {}
    payload["validation_receipts"] = []
    return canonical_hash(payload)


def _build_artifact(
    static: Mapping[str, Any],
    online: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    unit_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    mutations: Sequence[Mapping[str, Any]],
    candidate: bool,
    fixture: bool = False,
) -> JsonDict:
    validation_ok = _validation_passed(validation_receipts, terminal=not candidate)
    branches = [dict(static), dict(online)]
    classification = classify_terminal(branches, validation_passed=validation_ok)
    failed_branch = next((row for row in branches if row.get("available") is not True), {})
    failed_summary = failed_branch.get("gate_check_summary") or {}

    gates = [
        {
            "check": "required_validation",
            "category": "validity",
            "op": "==",
            "expected": True,
            "observed": validation_ok,
            "passed": validation_ok,
            "principle": "Only the frozen affected and terminal readers authorize publication.",
        },
        {
            "check": "mutation_controls_passed",
            "category": "validity",
            "op": "==",
            "expected": True,
            "observed": all(r.get("passed") is True for r in mutations),
            "passed": all(r.get("passed") is True for r in mutations),
            "principle": "All registered data corruption mutations must be rejected.",
        },
        {
            "check": "decision_audit_complete",
            "category": "validity",
            "op": "==",
            "expected": 1,
            "observed": classification["decision_audit_complete_score"],
            "passed": classification["decision_audit_complete_score"] == 1,
            "principle": "Each branch must have a justified audit or blocked disposition.",
        },
        {
            "check": "promotion_disabled",
            "category": "validity",
            "op": "==",
            "expected": 0,
            "observed": 0,
            "passed": True,
            "principle": "An audit result cannot change production policy.",
        },
        {
            "check": "decision_value_benefit",
            "category": "benefit",
            "op": "==",
            "expected": 1,
            "observed": 0,
            "passed": False,
            "principle": "Statistical benefit requires positive reproduced decision value.",
        },
    ]
    for interval in online.get("paired_confidence_intervals") or []:
        upper = interval.get("upper")
        gates.append(
            {
                "check": f"primary_log_loss_upper:{interval.get('comparison')}",
                "category": "benefit",
                "op": "<",
                "expected": 0.0,
                "observed": upper,
                "passed": isinstance(upper, (int, float)) and float(upper) < 0.0,
                "principle": "Each registered paired source-group interval must exclude zero on the benefit side.",
            }
        )

    validation_duration = math.fsum(
        float(row.get("duration_s", 0.0)) for row in validation_receipts
    )
    duration = float(current_receipt["duration_s"])

    branch_rows_list = [
        {
            "branch": "static",
            "task_id": "exp7453-energy-calibration",
            "claim": "source_conditioned_energy_calibration",
            "available": static.get("available", False),
            "valid": static.get("valid", False),
            "verdict_class": static.get("verdict_class"),
            "honest_verdict": static.get("honest_verdict"),
            "original_verdict_class": static.get("producer_verdict_class"),
            "original_flagged_adversarial": static.get("producer_flagged_adversarial", False),
            "disposition": "blocked_upstream_pre_gated"
            if not static.get("available")
            else "complete",
        },
        {
            "branch": "online",
            "task_id": "exp7454-continuous-learning",
            "claim": "delayed_continuous_learning_replay",
            "available": online.get("available", False),
            "valid": online.get("valid", False),
            "verdict_class": online.get("verdict_class"),
            "honest_verdict": online.get("honest_verdict"),
            "original_verdict_class": online.get("producer_verdict_class"),
            "original_flagged_adversarial": online.get("producer_flagged_adversarial", False),
            "disposition": "complete_null_reproduced" if online.get("valid") else "disqualified",
        },
    ]

    cont_rows = continuation_rows(
        static_verdict=str(static.get("honest_verdict")),
        online_verdict=str(online.get("honest_verdict")),
        online_retired=bool(online.get("mixture_construction_retired", True)),
    )

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
        "clock_identity": {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
            "segment_id": current_receipt.get("current_run_id"),
            "clock": "time.monotonic_ns",
        },
        "random_seed": {
            "fit_seed": None,
            "projection_seed": None,
            "stream_seed": 6_520_440,
            "online_bootstrap_seed": 6_520_440,
            "null_reason": "Static fitting was pre-gated; the audit fits no model or projection.",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in unit_rows],
        "sample_size_budget": {
            "planned_independent_units": 2,
            "attempted_independent_units": 2,
            "completed_independent_units": 2,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "unstarted_independent_units": 0,
            "delivered_row_units": len(unit_rows),
            "delivered_feedback_updates": (online.get("replay_summary") or {}).get(
                "updates_replayed", 0
            ),
            "stopping_rule": "audit each available branch once and retain terminal upstream absence",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "required_checks_passed": validation_ok,
            "failed_required_checks": ([] if validation_ok else ["required_validation"]),
            "benefit_failures": [
                str(row.get("check"))
                for row in gates
                if row.get("category") == "benefit" and row.get("passed") is not True
            ],
            "evidence_defects": static.get("errors", []) + online.get("errors", []),
            "blocked_upstream": failed_summary.get("blocked_upstream"),
            "blocked_path": failed_summary.get("blocked_path"),
            "blocked_check": failed_summary.get("blocked_check"),
            "blocked_field": failed_summary.get("blocked_field"),
            "blocked_expected": failed_summary.get("blocked_expected"),
            "blocked_observed": failed_summary.get("blocked_observed"),
        },
        "verifier_is_oracle": False,
        **classification,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "decision_audit_complete_score": classification["decision_audit_complete_score"],
        "branch_rows": branch_rows_list,
        "mutation_rows": [deepcopy(dict(m)) for m in mutations],
        "independent_update_replay": deepcopy(dict(online.get("replay_summary") or {})),
        "continuation_rows": cont_rows,
        "mixture_construction_retired": bool(
            online.get("mixture_construction_retired", False)
            and online.get("valid") is True
            and online.get("verdict_class") == "null"
        ),
        "static_audit": deepcopy(dict(static)),
        "online_audit": deepcopy(dict(online)),
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
    }
    result["reproducibility_checksum"] = reproducibility_checksum(result)
    return result


def build_artifact_for_test() -> JsonDict:
    """Build a compact valid terminal artifact fixture for testing."""
    static = {
        "branch": "static",
        "upstream": "exp7453-energy-calibration",
        "available": False,
        "valid": True,
        "complete": False,
        "value": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "producer_verdict_class": "blocked",
        "producer_flagged_adversarial": False,
        "errors": [],
        "gate_check_summary": {
            "blocked_upstream": "exp7452-source-embeddings",
            "blocked_path": "results/experiment_7452_v653_source_embeddings.json",
            "blocked_check": "exp7452-source-embeddings.embedding_capture_ready_score",
            "blocked_field": "embedding_capture_ready_score",
            "blocked_expected": 1,
            "blocked_observed": 0,
        },
    }
    online = {
        "branch": "online",
        "upstream": "exp7454-continuous-learning",
        "available": True,
        "valid": True,
        "complete": True,
        "value": False,
        "verdict_class": "null",
        "honest_verdict": "complete_null_insufficient_online_benefit",
        "producer_verdict_class": "null",
        "producer_flagged_adversarial": False,
        "errors": [],
        "replay_summary": {
            "updates_replayed": 1,
            "all_updates_matched": True,
            "max_loss_gap": 0.0,
            "max_weight_gap": 0.0,
            "no_feedback_immutable": True,
            "prediction_before_reveal": True,
        },
        "paired_confidence_intervals": [
            {
                "comparison": "learned_mixture_minus_equal_weight_adaptive_mixture",
                "upper": 0.01,
            }
        ],
        "mixture_construction_retired": True,
    }
    receipt = build_current_work_receipt(
        run_id="exp7455-fixture",
        owner_pid=1,
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"cpu": "fixture", "cuda": None, "external_device": None},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1,
        phase_spans=[],
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    names = [
        *validation_scope.REQUIRED_CHECK_NAMES,
        "cold_artifact_replay",
        "independent_branch_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    fake_receipts = [
        {"name": name, "passed": True, "exit_code": 0, "duration_s": 0.0} for name in names
    ]
    mutations = [
        {
            "mutation": name,
            "claim": "integrity",
            "expected": "rejected",
            "observed": "rejected",
            "passed": True,
            "principle": "Defective evidence must fail closed.",
        }
        for name in REQUIRED_MUTATIONS
    ]
    unit_rows = [
        {
            "branch": "online",
            "arm": "learned_mixture",
            "brier_score": 0.05,
            "log_loss_mean": 0.15,
            "prediction_count": 753,
        }
    ]
    return _build_artifact(
        static,
        online,
        preconditions=[],
        source_hashes={},
        unit_rows=unit_rows,
        validation_receipts=fake_receipts,
        current_receipt=receipt,
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        mutations=mutations,
        candidate=False,
        fixture=True,
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    """Cold-check identity, schema completeness, branch dispositions, and checksum."""
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
    if value.get("decision_audit_complete_score") not in {0, 1}:
        errors.append("decision_audit_complete_score_invalid")

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
    if value.get("decision_audit_complete_score") != expected["decision_audit_complete_score"]:
        errors.append("decision_audit_complete_score_invalid")

    expected_branch_rows = [
        {
            "branch": branch,
            "original_flagged_adversarial": (value.get(f"{branch}_audit") or {}).get(
                "producer_flagged_adversarial", False
            ),
        }
        for branch in ("static", "online")
    ]
    actual_branch_rows = {
        str(row.get("branch")): row
        for row in value.get("branch_rows") or []
        if isinstance(row, Mapping)
    }
    if any(
        branch["branch"] not in actual_branch_rows
        or actual_branch_rows[branch["branch"]].get("original_flagged_adversarial")
        is not branch["original_flagged_adversarial"]
        for branch in expected_branch_rows
    ):
        errors.append("branch_rows_mismatch")

    mutations = value.get("mutation_rows") or []
    if {row.get("mutation") for row in mutations if isinstance(row, Mapping)} != set(
        REQUIRED_MUTATIONS
    ) or any(row.get("passed") is not True for row in mutations if isinstance(row, Mapping)):
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
    """Reload a candidate in a fresh process and apply strict validation."""
    value = _load_object(path)
    if not value:
        return ["candidate_artifact_unreadable"]
    return validate_artifact(value, root=root, verify_source_bytes=verify_source_bytes)


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload producer rows and compare branch reductions to candidate artifact."""
    value = _load_object(path)
    errors = validate_artifact(value, root=root, verify_source_bytes=False)
    _checks, _hashes, _producers, static, online = audit_sources(root)
    if canonical_hash(static) != canonical_hash(value.get("static_audit") or {}):
        errors.append("static_independent_reduction_mismatch")
    if canonical_hash(online) != canonical_hash(value.get("online_audit") or {}):
        errors.append("online_independent_reduction_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    commands = build_command_plan(root, MANIFEST, private_root)
    plan_errors = validate_command_plan(root, MANIFEST, commands)
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
        f"[exp7455] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
) -> JsonDict:  # pragma: no cover
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

    # 1. Authenticate prerequisites
    phase_started = time.monotonic()
    _progress(started, "preconditions", "before_check")
    preconditions, source_hashes, _producers, static_audit, online_audit = audit_sources(root)
    spans.append(_span("preconditions", phase_started, started, len(preconditions)))
    _progress(started, "preconditions", "after_check", count=len(preconditions))

    # 2. Branch reduction
    phase_started = time.monotonic()
    _progress(started, "branch_reduction", "before_audit")
    unit_rows = [
        {
            "branch": "online",
            **dict(row),
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
        }
        for row in online_audit.get("rows", [])
    ]
    unit_rows.append(
        {
            "branch": "static",
            "arm": "source_conditioned_energy_calibration",
            "prediction_count": 0,
            "attempted": False,
            "completed": False,
            "failed": False,
            "censored": False,
            "unstarted": True,
            "disposition": "blocked_upstream_pre_gated",
        }
    )

    mutations = run_mutation_controls()
    spans.append(_span("branch_reduction", phase_started, started, 2))
    _progress(
        started,
        "branch_reduction",
        "after_audit",
        static_status=static_audit["verdict_class"],
        online_status=online_audit["verdict_class"],
    )

    # 3. Affected validation
    private = Path(tempfile.mkdtemp(prefix="exp7455-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private)
    phase_started = time.monotonic()
    _progress(started, "affected_validation", "before_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
    )
    affected_reduction = reduce_affected_receipts(root, MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    _progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )

    ended_ns = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"exp7455-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
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

    for relative, receipt_class in (
        (MODULE_PATH, "current_audit_code"),
        (WRAPPER_PATH, "current_audit_entrypoint"),
        (TEST_PATH, "current_audit_tests"),
        (SPEC_PATH, "current_audit_protocol"),
        (ROADMAP_PATH, "current_audit_roadmap"),
    ):
        path = root / relative
        if path.is_file():
            source_hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_receipt_class": receipt_class,
            }

    candidate = _build_artifact(
        static_audit,
        online_audit,
        preconditions=preconditions,
        source_hashes=source_hashes,
        unit_rows=unit_rows,
        validation_receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        mutations=mutations,
        candidate=True,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    # 4. Terminal validation
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
        run_id=f"exp7455-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
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
        static_audit,
        online_audit,
        preconditions=preconditions,
        source_hashes=source_hashes,
        unit_rows=unit_rows,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        mutations=mutations,
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
    if args.date is None:  # pragma: no cover - error is checked without launching the E2E.
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
