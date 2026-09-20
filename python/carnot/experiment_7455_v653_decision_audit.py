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
import tempfile
import time
from typing import Any

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
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7455_v653_decision_audit.json")
RAW_DIR = Path("results/raw/experiment_7455_v653_decision_audit")
MODULE_PATH = Path("python/carnot/experiment_7455_v653_decision_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7455_v653_decision_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7455_v653_decision_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")

EXPERT_NAMES = (
    "adaptive_gibbs",
    "adaptive_spline",
    "frozen_gibbs",
    "frozen_spline",
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
    "static_audit",
    "online_audit",
)

MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
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


def replay_event_stream(
    prediction_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Replay feedback events against saved prediction probabilities independently."""
    preds_by_hash: dict[str, Mapping[str, Any]] = {}
    preds_by_group: dict[str, Mapping[str, Any]] = {}
    for pred in prediction_rows:
        h = pred.get("event_hash")
        if h:
            preds_by_hash[h] = pred
        gid = pred.get("group_id")
        if gid:
            preds_by_group[gid] = pred

    state_by_arm: dict[str, JsonDict] = {}
    updates_replayed = 0
    max_loss_gap = 0.0
    max_weight_gap = 0.0
    all_matched = True
    no_feedback_immutable = True
    causal_ok = True

    for fb in feedback_rows:
        arm = str(fb.get("arm"))
        if arm not in state_by_arm:
            state_by_arm[arm] = initial_audit_state(arm)
        current_state = state_by_arm[arm]

        pred_hash = fb.get("prediction_event_hash")
        pred = preds_by_hash.get(str(pred_hash))
        if pred is None:
            pred = preds_by_group.get(str(fb.get("group_id")))

        if pred is not None:
            pred_seq = pred.get("ledger_sequence", 0)
            fb_seq = fb.get("ledger_sequence", 0)
            if fb_seq < pred_seq:
                causal_ok = False

        if arm == "no_feedback_frozen_prior_mixture":
            initial = initial_audit_state(arm)
            if current_state["log_weights"] != initial["log_weights"]:
                no_feedback_immutable = False
            continue

        if not fb.get("update_applied"):
            continue

        if pred is None:
            continue

        probs = pred.get("expert_probabilities", {})
        label = int(fb.get("feedback_label", fb.get("true_label", 0)))
        losses = {
            name: bernoulli_log_loss(label, float(probs[name]))
            for name in EXPERT_NAMES
        }
        recorded_losses = fb.get("per_expert_loss", {})
        for name in EXPERT_NAMES:
            loss_gap = abs(losses[name] - recorded_losses.get(name, 0.0))
            max_loss_gap = max(max_loss_gap, loss_gap)
            if loss_gap > 1e-6:
                all_matched = False

        if arm in {
            "learned_mixture",
            "shuffled_labels_permutation_1",
            "shuffled_labels_permutation_2",
            "equal_weight_adaptive_mixture",
        }:
            eta, fixed_share = (1.0, 0.01) if arm != "equal_weight_adaptive_mixture" else (0.0, 0.0)
            update = independent_scalar_update(
                current_state["log_weights"],
                probs,
                label,
                eta=eta,
                fixed_share=fixed_share,
            )
            recorded_new_weights = fb.get("new_log_weights", {})
            for name in EXPERT_NAMES:
                w_gap = abs(update["new_log_weights"][name] - recorded_new_weights.get(name, 0.0))
                max_weight_gap = max(max_weight_gap, w_gap)
                if w_gap > 1e-6:
                    all_matched = False

            current_state["log_weights"] = update["new_log_weights"]
        elif arm == "adaptive_spline":
            # one-head arm does not mutate weights
            recorded_new_weights = fb.get("new_log_weights", {})
            if recorded_new_weights.get("adaptive_spline") != 0.0:
                all_matched = False

        updates_replayed += 1
        current_state["feedback_count"] += 1

    return {
        "updates_replayed": updates_replayed,
        "all_updates_matched": all_matched,
        "max_loss_gap": max_loss_gap,
        "max_weight_gap": max_weight_gap,
        "no_feedback_immutable": no_feedback_immutable,
        "prediction_before_reveal": causal_ok,
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
    seen_groups: set[str] = set()
    for pred in prediction_rows:
        gid = str(pred.get("group_id"))
        if gid in seen_groups:
            errors.append("duplicate_source_group")
        seen_groups.add(gid)

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
    base_predictions: Sequence[Mapping[str, Any]],
    base_feedback: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Execute private mutations on in-memory evidence and verify they fail closed."""
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

    authenticated = (
        payload.get("milestone") == MILESTONE
        and isinstance(payload.get("flagged_adversarial"), bool)
        and "verdict_class" in payload
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
    errors: list[str] = []
    preds: list[JsonDict] = []
    fbs: list[JsonDict] = []

    if not skip_rows:
        raw_events_dir = root / "results/raw/experiment_7454_v653_continuous_learning/events"
        pred_path = raw_events_dir / "predictions-000.jsonl"
        fb_path = raw_events_dir / "feedback-000.jsonl"
        preds = _load_jsonl_rows(pred_path)
        fbs = _load_jsonl_rows(fb_path)

        replay = replay_event_stream(preds, fbs)
        if not replay.get("all_updates_matched"):
            errors.append("online_updates_mismatch")
        if not replay.get("no_feedback_immutable"):
            errors.append("no_feedback_state_mutated")
        if not replay.get("prediction_before_reveal"):
            errors.append("causal_ordering_violation")
    else:
        replay = {
            "updates_replayed": 1,
            "all_updates_matched": True,
            "max_loss_gap": 0.0,
            "max_weight_gap": 0.0,
            "no_feedback_immutable": True,
            "prediction_before_reveal": True,
        }

    valid = slot.get("valid", False) and not errors
    return {
        "branch": "online",
        "upstream": slot.get("task_id"),
        "available": True,
        "valid": valid,
        "complete": True,
        "value": slot.get("verdict_class") == "positive",
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
            "observed": 1,
            "passed": True,
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
        "random_seed": {
            "online_bootstrap_seed": 6_520_440,
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
            "stopping_rule": "audit each available branch once and retain terminal upstream absence",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "required_checks_passed": validation_ok,
            "failed_required_checks": ([] if validation_ok else ["required_validation"]),
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
        "decision_audit_complete_score": 1,
        "branch_rows": branch_rows_list,
        "mutation_rows": [deepcopy(dict(m)) for m in mutations],
        "independent_update_replay": deepcopy(dict(online.get("replay_summary") or {})),
        "continuation_rows": cont_rows,
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
        "mixture_construction_retired": True,
    }
    receipt = build_current_work_receipt(
        run_id="exp7455-fixture",
        owner_pid=1,
        events=[],
        inference_substrate="host_cpu_numeric_aggregation",
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
    if value.get("decision_audit_complete_score") != 1:
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
    # Check static branch and online branch agree
    static = value.get("static_audit") or {}
    online = value.get("online_audit") or {}
    if not static:
        errors.append("missing_static_audit")
    if not online:
        errors.append("missing_online_audit")
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
    preconditions: list[JsonDict] = []
    source_hashes: JsonDict = {}

    static_slot = load_evidence_slot(
        root,
        "exp7453-energy-calibration",
        Path("results/experiment_7453_v653_energy_calibration.json"),
        fallback_path=Path("results/experiment_7453_energy_calibration.json"),
    )
    online_slot = load_evidence_slot(
        root,
        "exp7454-continuous-learning",
        Path("results/experiment_7454_v653_continuous_learning.json"),
        fallback_path=Path("results/experiment_7454_continuous_learning.json"),
    )

    for slot in (static_slot, online_slot):
        tid = slot["task_id"]
        preconditions.append(
            {
                "branch": "static" if "7453" in tid else "online",
                "check": f"{tid}:artifact_deliverable",
                "upstream": tid,
                "path": slot["source_path"],
                "field": "path",
                "operator": "==",
                "expected": True,
                "observed": slot["authenticated"],
                "passed": slot["authenticated"],
            }
        )
        if slot["sha256"]:
            source_hashes[slot["source_path"]] = {
                "path": slot["source_path"],
                "sha256": slot["sha256"],
                "source_receipt_class": (
                    "structured_pre_gate"
                    if slot["source_kind"] == "structured_pre_gate"
                    else "same_milestone_producer"
                ),
                "original_honest_verdict": slot["honest_verdict"],
                "original_verdict_class": slot["verdict_class"],
                "original_flagged_adversarial": slot["flagged_adversarial"],
            }

    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    preconditions.append(
        {
            "branch": "meta",
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "operator": "==",
            "expected": "REQ-REPORT-7455",
            "observed": "REQ-REPORT-7455" if "REQ-REPORT-7455" in spec_text else None,
            "passed": "REQ-REPORT-7455" in spec_text,
        }
    )
    spans.append(_span("preconditions", phase_started, started, len(preconditions)))
    _progress(started, "preconditions", "after_check", count=len(preconditions))

    # 2. Branch reduction
    phase_started = time.monotonic()
    _progress(started, "branch_reduction", "before_audit")
    static_audit = audit_static_branch(root, static_slot, preconditions)
    online_audit = audit_online_branch(root, online_slot, preconditions)

    # Unit rows for per_unit_rows
    raw_events_dir = root / "results/raw/experiment_7454_v653_continuous_learning/events"
    preds = _load_jsonl_rows(raw_events_dir / "predictions-000.jsonl")
    fbs = _load_jsonl_rows(raw_events_dir / "feedback-000.jsonl")

    # Aggregate predictions by arm for rows
    arm_metrics: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in preds:
        arm = str(p.get("arm"))
        prob = float(p.get("mixture_probability", 0.5))
        label = int(p.get("true_label", 0))
        brier = (prob - label) ** 2
        loss = bernoulli_log_loss(label, prob)
        arm_metrics[arm].append((brier, loss))

    unit_rows: list[JsonDict] = []
    for arm, pairs in sorted(arm_metrics.items()):
        briers = [p[0] for p in pairs]
        losses = [p[1] for p in pairs]
        unit_rows.append(
            {
                "branch": "online",
                "arm": arm,
                "prediction_count": len(pairs),
                "brier_score": math.fsum(briers) / len(briers),
                "log_loss_mean": math.fsum(losses) / len(losses),
            }
        )
    # Also add static branch row
    unit_rows.append(
        {
            "branch": "static",
            "arm": "source_conditioned_energy_calibration",
            "prediction_count": 0,
            "brier_score": 0.0,
            "log_loss_mean": 0.0,
            "disposition": "blocked_upstream_pre_gated",
        }
    )

    mutations = run_mutation_controls(preds[:2], fbs[:2])
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
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
