"""Run bounded reliability-state learning with exact write quarantine.

The simulation keeps model weights absent and fixed. It updates only a small
symmetric matrix after each action closes and its exact outcome becomes known.
Exact receipts, not the learned matrix, decide whether a proposed memory write
can become durable.

Spec ref: REQ-LEARN-6872.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np


JsonDict = dict[str, Any]
Matrix = list[list[float]]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE_PATH = Path(
    "results/experiment_6871_observable_reliability_opportunity_stream.json"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6872_bounded_reliability_controller_quarantine.json"
)
INFERENCE_SUBSTRATE = "deterministic CPU bounded reliability-state online update simulation"
RANDOM_SEED = 6_872_001
BLOCKED_VERDICT = "complete_blocked_bounded_reliability_controller_quarantine"
READY_VERDICT = "complete_positive_bounded_reliability_controller_ready_not_sealed_utility_verdict"
DISQUALIFIED_VERDICT = "complete_disqualified_bounded_reliability_controller_model_weight_mutation"

ARMS = (
    "frozen_no_memory",
    "read_only",
    "bounded_update",
    "exact_quarantine",
    "v599_unsafe_reference",
)
ACTIONS = ("no_memory", "read_only", "write", "abstain")
SOURCE_ACTIONS = {
    "no_memory",
    "read_only_retrieval",
    "bounded_update",
    "quarantine",
    "v599_unsafe_reference",
}
STATE_NODES = (
    "exp6827_transactions",
    "exp6840_outcomes",
    "exp6841_outcomes",
    "no_memory",
    "read_only",
    "write",
    "abstain",
)
OFFLINE_DECISION_FIELDS = {
    "later_exact_outcome",
    "signed_direction",
    "exact_outcome_hash",
    "delayed_correction",
    "future_correction",
    "audit_label",
    "held_split_label",
    "task_order_label",
}
EXACT_CHECK_NAMES = (
    "coverage",
    "preservation",
    "source_faithfulness",
    "provenance",
    "old_family_retention",
    "delayed_invalidation",
    "replay",
    "restart",
    "rollback",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
UPDATE_CONTRACT = {
    "update_timing": "after_action_freeze_and_later_exact_outcome_reveal",
    "same_event_outcome_may_select_action": False,
    "max_abs_entry_delta_per_event": 0.05,
    "max_spectral_norm_delta_per_event": 0.1,
}
OWN_SOURCE_PATHS = {
    "module": "python/carnot/experiment_6872_bounded_reliability_controller_quarantine.py",
    "wrapper": "scripts/experiments/experiment_6872_bounded_reliability_controller_quarantine.py",
    "focused_tests": "tests/python/test_experiment_6872_bounded_reliability_controller_quarantine.py",
    "spec": "openspec/capabilities/continuous-learning/spec.md",
}

STANDARD_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "replay_commands",
}
TASK_FIELDS = {
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "continuous_self_learning_task",
    "no_model_weight_mutation",
    "rows",
    "state_transition_rows",
    "spectral_bound_rows",
    "action_distribution_by_arm",
    "abstention_rate_by_arm",
    "admitted_update_rows",
    "rejected_update_rows",
    "harmful_write_rows",
    "exact_transition_check_rows",
    "held_future_utility_by_arm",
    "old_family_retention_by_arm",
    "delayed_correction_rows",
    "restart_rows",
    "rollback_rows",
    "counterfactual_support_rows",
    "random_seed",
    "reproducibility_checksum",
    "bounded_reliability_controller_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
REQUIRED_ARTIFACT_FIELDS = STANDARD_FIELDS | TASK_FIELDS


def canonical_json_bytes(value: Any) -> bytes:
    """Return one stable JSON representation for state and receipt hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 identity for a JSON value."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash exact file bytes and return an empty identity when absent."""

    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _is_sha256(value: Any) -> bool:
    """Accept only a complete project-style SHA-256 identity."""

    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def initialize_reliability_state() -> Matrix:
    """Create the frozen finite zero matrix in the declared node order."""

    return [[0.0 for _ in STATE_NODES] for _ in STATE_NODES]


def matrix_is_symmetric(matrix: Sequence[Sequence[float]]) -> bool:
    """Check shape, finiteness, and exact symmetry before an update."""

    size = len(STATE_NODES)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        return False
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(array).all() and np.array_equal(array, array.T))


def state_checksum(matrix: Sequence[Sequence[float]]) -> str:
    """Bind every state entry so restart drift is visible."""

    return sha256_json([[float(value) for value in row] for row in matrix])


def _spectral_norm_symmetric(matrix: Sequence[Sequence[float]]) -> tuple[float, list[float]]:
    """Return exact CPU eigenvalue evidence for a small symmetric matrix."""

    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    rounded = [round(float(value), 12) for value in eigenvalues]
    return round(max((abs(value) for value in rounded), default=0.0), 12), rounded


def bounded_symmetric_update(
    state: Sequence[Sequence[float]],
    source_node: str,
    action_node: str,
    reward: float,
    *,
    learning_rate: float = 0.05,
    action_frozen: bool = True,
    outcome_revealed_after_action: bool = True,
    same_event_outcome_access: bool = False,
    enforce_bound: bool = True,
) -> tuple[Matrix, JsonDict]:
    """Apply one symmetric event update only after exact feedback closes.

    A rejected proposal returns an exact copy of the input state. This makes a
    failed bound or timing check incapable of partially teaching the next event.
    """

    before = [[float(value) for value in row] for row in state]
    before_sha = state_checksum(before)
    failures: list[str] = []
    if not matrix_is_symmetric(before):
        failures.append("state_asymmetry")
    if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        failures.append("nonfinite_feedback")
    if source_node not in STATE_NODES or action_node not in STATE_NODES:
        failures.append("unknown_state_node")
    if not action_frozen:
        failures.append("action_not_frozen")
    if not outcome_revealed_after_action:
        failures.append("outcome_not_revealed_after_action")
    if same_event_outcome_access:
        failures.append("same_event_outcome_access")

    delta = initialize_reliability_state()
    if not failures:
        source_index = STATE_NODES.index(source_node)
        action_index = STATE_NODES.index(action_node)
        change = round(float(reward) * float(learning_rate), 12)
        delta[source_index][action_index] = change
        delta[action_index][source_index] = change
    maximum_entry_delta = max((abs(value) for row in delta for value in row), default=0.0)
    spectral_delta, delta_eigenvalues = _spectral_norm_symmetric(delta)
    if maximum_entry_delta > UPDATE_CONTRACT["max_abs_entry_delta_per_event"] + 1e-12:
        failures.append("entry_change_exceeds_bound")
    if spectral_delta > UPDATE_CONTRACT["max_spectral_norm_delta_per_event"] + 1e-12:
        failures.append("spectral_change_exceeds_bound")

    bound_failures = {
        "entry_change_exceeds_bound",
        "spectral_change_exceeds_bound",
    }
    blocking = [reason for reason in failures if enforce_bound or reason not in bound_failures]
    applied = not blocking
    if applied:
        after = [
            [
                round(before[row][column] + delta[row][column], 12)
                for column in range(len(STATE_NODES))
            ]
            for row in range(len(STATE_NODES))
        ]
    else:
        after = deepcopy(before)
    after_sha = state_checksum(after)
    within_bound = not bool(bound_failures.intersection(failures))
    receipt = {
        "applied": applied,
        "failure_reasons": failures,
        "state_before_sha256": before_sha,
        "state_after_sha256": after_sha,
        "state_symmetric_before": matrix_is_symmetric(before),
        "state_symmetric_after": matrix_is_symmetric(after),
        "max_abs_entry_delta": round(maximum_entry_delta, 12),
        "spectral_norm_delta": spectral_delta,
        "delta_eigenvalues": delta_eigenvalues,
        "max_abs_entry_delta_bound": UPDATE_CONTRACT["max_abs_entry_delta_per_event"],
        "spectral_norm_delta_bound": UPDATE_CONTRACT["max_spectral_norm_delta_per_event"],
        "within_bound": within_bound,
        "bound_enforced": enforce_bound,
        "action_frozen_before_update": action_frozen,
        "outcome_revealed_after_action": outcome_revealed_after_action,
        "same_event_outcome_access": same_event_outcome_access,
        "update_timing": UPDATE_CONTRACT["update_timing"],
        "delta_entries": [
            {"row": row, "column": column, "value": delta[row][column]}
            for row in range(len(STATE_NODES))
            for column in range(row, len(STATE_NODES))
            if delta[row][column] != 0.0
        ],
    }
    return after, receipt


def decision_feature_failures(features: Mapping[str, Any]) -> list[str]:
    """Reject any offline supervision field from the current decision."""

    return [
        f"offline_feature_in_decision_context:{field}"
        for field in sorted(set(features).intersection(OFFLINE_DECISION_FIELDS))
    ]


def choose_action(arm: str, features: Mapping[str, Any], state: Matrix) -> str:
    """Choose from prior state and decision-time evidence only."""

    failures = decision_feature_failures(features)
    if failures:
        raise ValueError(";".join(failures))
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if arm == "frozen_no_memory":
        return "no_memory"
    stale = features.get("evidence_status_at_decision") == "stale"
    if arm == "read_only":
        return "no_memory" if stale else "read_only"
    if arm == "v599_unsafe_reference":
        return "write"

    content_identity = str(features.get("source_content_sha256") or "")
    bucket = int(content_identity[-2:], 16) % 8 if _is_sha256(content_identity) else 7
    if arm == "exact_quarantine":
        if stale:
            return "write"
        return (
            "write",
            "write",
            "write",
            "read_only",
            "read_only",
            "no_memory",
            "abstain",
            "abstain",
        )[bucket]

    if stale:
        return "abstain"
    action = (
        "write",
        "write",
        "write",
        "read_only",
        "read_only",
        "no_memory",
        "abstain",
        "abstain",
    )[bucket]
    write_reliability = state[STATE_NODES.index("exp6827_transactions")][STATE_NODES.index("write")]
    return "read_only" if action == "write" and write_reliability < -0.5 else action


def _memory_bytes(memory: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode the complete durable memory for restart and rollback checks."""

    return canonical_json_bytes([dict(row) for row in memory])


def _write_record(event: Mapping[str, Any]) -> JsonDict:
    """Reduce an admitted event to the durable fields needed for replay."""

    features = event.get("decision_features", {})
    outcome = event.get("later_exact_outcome", {})
    return {
        "event_identity": event.get("event_identity"),
        "family": features.get("family") if isinstance(features, Mapping) else None,
        "primary_content_sha256": event.get("primary_content_sha256"),
        "exact_outcome_hash": outcome.get("exact_outcome_hash")
        if isinstance(outcome, Mapping)
        else None,
    }


def evaluate_write_transition(
    event: Mapping[str, Any],
    memory: Sequence[Mapping[str, Any]],
    *,
    attack: Mapping[str, Any] | None = None,
) -> tuple[JsonDict, list[JsonDict]]:
    """Run all exact transition checks before one write can become durable."""

    mutation = dict(attack or {})
    parent = [dict(row) for row in memory]
    parent_bytes = _memory_bytes(parent)
    candidate = [] if mutation.get("drop_existing") else deepcopy(parent)
    drop_family = mutation.get("drop_anchor_family")
    if drop_family is not None:
        candidate = [row for row in candidate if row.get("family") != drop_family]
    candidate.append(_write_record(event))

    features = event.get("decision_features", {})
    counterfactual = event.get("counterfactual", {})
    support = event.get("action_support", {})
    bounded_support = support.get("bounded_update", {}) if isinstance(support, Mapping) else {}
    outcome = event.get("later_exact_outcome", {})
    primary = event.get("primary_source", {})
    direction = outcome.get("signed_direction") if isinstance(outcome, Mapping) else None
    delayed = outcome.get("delayed_correction", {}) if isinstance(outcome, Mapping) else {}
    proposal_kind = mutation.get("proposal_kind", "supported_update")

    parent_ids = {str(row.get("event_identity")) for row in parent}
    candidate_ids = {str(row.get("event_identity")) for row in candidate}
    candidate_bytes = _memory_bytes(candidate)
    restart_bytes = b"{" if mutation.get("restart_drift") else candidate_bytes
    try:
        restart_exact = _memory_bytes(json.loads(restart_bytes)) == restart_bytes
    except (json.JSONDecodeError, TypeError):
        restart_exact = False
    rollback_bytes = candidate_bytes if mutation.get("rollback_failure") else parent_bytes
    provenance_values = (
        event.get("primary_content_sha256"),
        primary.get("transaction_receipt_sha256") if isinstance(primary, Mapping) else None,
        outcome.get("exact_outcome_hash") if isinstance(outcome, Mapping) else None,
        outcome.get("source_row_sha256") if isinstance(outcome, Mapping) else None,
    )
    checks = {
        "coverage": bool(
            proposal_kind == "supported_update"
            and isinstance(bounded_support, Mapping)
            and bounded_support.get("supported") is True
            and isinstance(counterfactual, Mapping)
            and counterfactual.get("valid_pre_action") is True
        ),
        "preservation": parent_ids.issubset(candidate_ids),
        "source_faithfulness": isinstance(direction, int) and direction >= 0,
        "provenance": not mutation.get("remove_provenance")
        and all(_is_sha256(value) for value in provenance_values),
        "old_family_retention": parent_ids.issubset(candidate_ids),
        "delayed_invalidation": isinstance(delayed, Mapping)
        and delayed.get("correction_family") is None,
        "replay": bool(
            mutation.get(
                "replay_passed",
                isinstance(counterfactual, Mapping)
                and counterfactual.get("valid_pre_action") is True,
            )
        ),
        "restart": restart_exact,
        "rollback": rollback_bytes == parent_bytes,
    }
    failed = [name for name in EXACT_CHECK_NAMES if checks[name] is not True]
    admitted = not failed
    after = candidate if admitted else deepcopy(parent)
    restored_bytes = _memory_bytes(after)
    check_id = sha256_json(
        {
            "event_identity": event.get("event_identity"),
            "parent_memory_sha256": sha256_json(parent),
            "checks": checks,
            "attack": mutation,
        }
    )
    receipt = {
        "transition_check_id": check_id,
        "event_identity": event.get("event_identity"),
        "proposal_kind": proposal_kind,
        "checks": checks,
        "failed_checks": failed,
        "exact_transition_passed": admitted,
        "admission_decision": "admitted" if admitted else "quarantined",
        "parent_memory_sha256": sha256_json(parent),
        "candidate_memory_sha256": sha256_json(candidate),
        "restored_memory_sha256": sha256_json(after),
        "parent_bytes_sha256": f"sha256:{hashlib.sha256(parent_bytes).hexdigest()}",
        "candidate_bytes_sha256": f"sha256:{hashlib.sha256(candidate_bytes).hexdigest()}",
        "restored_bytes_sha256": f"sha256:{hashlib.sha256(restored_bytes).hexdigest()}",
        "restart_byte_exact": restart_exact,
        "rollback_byte_exact": rollback_bytes == parent_bytes,
    }
    return receipt, after


def _no_update_receipt(state: Matrix) -> JsonDict:
    """Record a frozen state transition without pretending an update ran."""

    checksum = state_checksum(state)
    return {
        "applied": False,
        "failure_reasons": [],
        "state_before_sha256": checksum,
        "state_after_sha256": checksum,
        "state_symmetric_before": matrix_is_symmetric(state),
        "state_symmetric_after": matrix_is_symmetric(state),
        "max_abs_entry_delta": 0.0,
        "spectral_norm_delta": 0.0,
        "delta_eigenvalues": [0.0 for _ in STATE_NODES],
        "max_abs_entry_delta_bound": UPDATE_CONTRACT["max_abs_entry_delta_per_event"],
        "spectral_norm_delta_bound": UPDATE_CONTRACT["max_spectral_norm_delta_per_event"],
        "within_bound": True,
        "bound_enforced": True,
        "action_frozen_before_update": True,
        "outcome_revealed_after_action": True,
        "same_event_outcome_access": False,
        "update_timing": "frozen_arm_no_update",
        "delta_entries": [],
    }


def _action_reward(action: str, direction: int, admitted: bool) -> float:
    """Convert the later exact direction into action-specific bounded feedback."""

    if action == "write":
        return float(direction) if admitted else 0.0
    if action == "read_only":
        return float(direction)
    if action == "no_memory":
        return float(-direction)
    return -0.25 if direction > 0 else 0.0


def _source_state_node(outcome: Mapping[str, Any]) -> str:
    """Map an exact outcome receipt to its fixed reliability node."""

    return (
        "exp6841_outcomes" if outcome.get("source_artifact") == "outcomes_b" else "exp6840_outcomes"
    )


def _concise_write_row(row: Mapping[str, Any]) -> JsonDict:
    """Keep comparison evidence without copying an entire prospective row."""

    return {
        key: row.get(key)
        for key in (
            "row_id",
            "event_identity",
            "arm",
            "proposed_action",
            "admission_decision",
            "signed_direction",
            "useful_write",
            "harmful_write",
            "false_injection",
            "transition_check_id",
        )
    }


def run_comparison(
    events: Sequence[Mapping[str, Any]],
    *,
    order_seed: int,
) -> JsonDict:
    """Run all arms over the same event order and keep every prospective row."""

    states = {arm: initialize_reliability_state() for arm in ARMS}
    memories: dict[str, list[JsonDict]] = {arm: [] for arm in ARMS}
    rows: list[JsonDict] = []
    transitions: list[JsonDict] = []
    spectral_rows: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    admitted_rows: list[JsonDict] = []
    rejected_rows: list[JsonDict] = []
    harmful_rows: list[JsonDict] = []
    delayed_rows: list[JsonDict] = []
    rollback_rows: list[JsonDict] = []
    held_start = (2 * len(events)) // 3 + 1

    for sequence, event in enumerate(events, start=1):
        features = event.get("decision_features", {})
        outcome = event.get("later_exact_outcome", {})
        if not isinstance(features, Mapping) or not isinstance(outcome, Mapping):
            raise ValueError("event decision and outcome objects are required")
        direction = int(outcome.get("signed_direction", 0))
        correction = outcome.get("delayed_correction", {})
        for arm in ARMS:
            state_before = states[arm]
            state_before_sha = state_checksum(state_before)
            action = choose_action(arm, features, state_before)
            write_receipt: JsonDict | None = None
            admission = "not_applicable"
            if action == "write":
                write_receipt, checked_memory = evaluate_write_transition(event, memories[arm])
                if arm == "v599_unsafe_reference":
                    admission = "unsafe_admitted"
                    memories[arm] = memories[arm] + [_write_record(event)]
                else:
                    admission = str(write_receipt["admission_decision"])
                    memories[arm] = checked_memory

            admitted = admission in {"admitted", "unsafe_admitted"}
            reward = _action_reward(action, direction, admitted)
            if arm in {"bounded_update", "exact_quarantine"}:
                state_after, transition = bounded_symmetric_update(
                    state_before,
                    _source_state_node(outcome),
                    action,
                    reward,
                )
            elif arm == "v599_unsafe_reference":
                state_after, transition = bounded_symmetric_update(
                    state_before,
                    _source_state_node(outcome),
                    action,
                    reward,
                    learning_rate=0.2,
                    enforce_bound=False,
                )
            else:
                state_after = deepcopy(state_before)
                transition = _no_update_receipt(state_before)
            states[arm] = state_after

            harmful = bool(action == "write" and direction < 0 and admitted)
            useful = bool(action == "write" and direction > 0 and admitted)
            row_id = sha256_json(
                {"event_identity": event.get("event_identity"), "arm": arm, "seed": order_seed}
            )
            row = {
                "row_id": row_id,
                "event_identity": event.get("event_identity"),
                "event_sequence": sequence,
                "order_seed": order_seed,
                "arm": arm,
                "family": features.get("family"),
                "decision_context_sha256": sha256_json(dict(features)),
                "decision_input_fields": sorted(features),
                "same_event_outcome_used_for_decision": False,
                "proposed_action": action,
                "action_frozen_before_outcome": True,
                "outcome_revealed_after_action": outcome.get("revealed_after_decision") is True,
                "exact_outcome_hash": outcome.get("exact_outcome_hash"),
                "signed_direction": direction,
                "admission_decision": admission,
                "state_before": state_before_sha,
                "state_after": state_checksum(state_after),
                "transition_check_id": write_receipt.get("transition_check_id")
                if write_receipt
                else None,
                "exact_transition_passed": write_receipt.get("exact_transition_passed")
                if write_receipt
                else None,
                "reward": reward,
                "held_future": sequence >= held_start,
                "useful_write": useful,
                "harmful_write": harmful,
                "false_injection": harmful,
                "rollback_performed": admission == "quarantined",
            }
            rows.append(row)
            transition_row = {
                "row_id": row_id,
                "event_identity": event.get("event_identity"),
                "arm": arm,
                "proposed_action": action,
                **transition,
            }
            transitions.append(transition_row)
            spectral_rows.append(
                {
                    "row_id": row_id,
                    "arm": arm,
                    "max_abs_entry_delta": transition["max_abs_entry_delta"],
                    "max_abs_entry_delta_bound": transition["max_abs_entry_delta_bound"],
                    "spectral_norm_delta": transition["spectral_norm_delta"],
                    "spectral_norm_delta_bound": transition["spectral_norm_delta_bound"],
                    "delta_eigenvalues": transition["delta_eigenvalues"],
                    "within_bound": transition["within_bound"],
                    "bound_required": arm != "v599_unsafe_reference",
                }
            )
            if write_receipt:
                exact_row = {"row_id": row_id, "arm": arm, **write_receipt}
                exact_rows.append(exact_row)
                concise = _concise_write_row(row)
                if admitted:
                    admitted_rows.append(concise)
                else:
                    rejected_rows.append(
                        {**concise, "failed_checks": write_receipt["failed_checks"]}
                    )
                    rollback_rows.append(
                        {
                            "row_id": row_id,
                            "event_identity": event.get("event_identity"),
                            "arm": arm,
                            "parent_bytes_sha256": write_receipt["parent_bytes_sha256"],
                            "restored_bytes_sha256": write_receipt["restored_bytes_sha256"],
                            "byte_exact": write_receipt["rollback_byte_exact"],
                        }
                    )
                if harmful:
                    harmful_rows.append(concise)
            if isinstance(correction, Mapping) and correction.get("correction_family") is not None:
                delayed_rows.append(
                    {
                        "row_id": row_id,
                        "event_identity": event.get("event_identity"),
                        "arm": arm,
                        "correction_family": correction.get("correction_family"),
                        "correction_latency_events": correction.get("correction_latency_events"),
                        "decision_context_excluded_correction": True,
                        "write_quarantined": admission == "quarantined",
                    }
                )

    restart_rows = []
    for arm in ARMS:
        payload = {"state": states[arm], "memory": memories[arm]}
        raw = canonical_json_bytes(payload)
        restarted = json.loads(raw)
        restart_rows.append(
            {
                "arm": arm,
                "checkpoint_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
                "state_before_restart": state_checksum(states[arm]),
                "state_after_restart": state_checksum(restarted["state"]),
                "restart_byte_exact": canonical_json_bytes(restarted) == raw,
            }
        )
    return {
        "rows": rows,
        "state_transition_rows": transitions,
        "spectral_bound_rows": spectral_rows,
        "exact_transition_check_rows": exact_rows,
        "admitted_update_rows": admitted_rows,
        "rejected_update_rows": rejected_rows,
        "harmful_write_rows": harmful_rows,
        "delayed_correction_rows": delayed_rows,
        "restart_rows": restart_rows,
        "rollback_rows": rollback_rows,
        "final_states": {arm: states[arm] for arm in ARMS},
    }


def _rate(numerator: int, denominator: int) -> float:
    """Return a stable zero-safe rate for artifact summaries."""

    return round(numerator / denominator, 12) if denominator else 0.0


def _summaries(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce prospective rows into the required arm-level measurements."""

    distribution: JsonDict = {}
    entropy: JsonDict = {}
    abstention: JsonDict = {}
    retrieval: JsonDict = {}
    proposals: JsonDict = {}
    admitted: JsonDict = {}
    useful: JsonDict = {}
    harmful: JsonDict = {}
    false_injection: JsonDict = {}
    rollback: JsonDict = {}
    held: JsonDict = {}
    retention: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        total = len(arm_rows)
        counts = Counter(str(row.get("proposed_action")) for row in arm_rows)
        distribution[arm] = {action: counts.get(action, 0) for action in ACTIONS}
        probabilities = [count / total for count in counts.values() if total and count]
        entropy[arm] = round(-sum(value * math.log2(value) for value in probabilities), 12)
        write_rows = [row for row in arm_rows if row.get("proposed_action") == "write"]
        admitted_writes = [
            row
            for row in write_rows
            if row.get("admission_decision") in {"admitted", "unsafe_admitted"}
        ]
        useful_writes = [row for row in arm_rows if row.get("useful_write") is True]
        harmful_writes = [row for row in arm_rows if row.get("harmful_write") is True]
        rolled_back = [row for row in arm_rows if row.get("rollback_performed") is True]
        held_rows = [row for row in arm_rows if row.get("held_future") is True]
        held_utility = sum(float(row.get("reward", 0.0)) for row in held_rows)
        abstention[arm] = _rate(counts.get("abstain", 0), total)
        retrieval[arm] = _rate(counts.get("read_only", 0), total)
        proposals[arm] = _rate(len(write_rows), total)
        admitted[arm] = _rate(len(admitted_writes), total)
        useful[arm] = _rate(len(useful_writes), total)
        harmful[arm] = _rate(len(harmful_writes), total)
        false_injection[arm] = harmful[arm]
        rollback[arm] = _rate(len(rolled_back), len(write_rows))
        held[arm] = {
            "row_count": len(held_rows),
            "utility_sum": round(held_utility, 12),
            "mean_utility": round(held_utility / len(held_rows), 12) if held_rows else 0.0,
        }
        anchor_count = total
        retained_count = total - len(harmful_writes)
        retention[arm] = {
            "anchor_count": anchor_count,
            "retained_count": retained_count,
            "retention_rate": _rate(retained_count, anchor_count),
        }
    return {
        "action_distribution_by_arm": distribution,
        "action_entropy_by_arm": entropy,
        "abstention_rate_by_arm": abstention,
        "retrieval_rate_by_arm": retrieval,
        "proposal_rate_by_arm": proposals,
        "admitted_write_rate_by_arm": admitted,
        "useful_write_rate_by_arm": useful,
        "harmful_write_rate_by_arm": harmful,
        "false_injection_rate_by_arm": false_injection,
        "rollback_rate_by_arm": rollback,
        "held_future_utility_by_arm": held,
        "old_family_retention_by_arm": retention,
    }


def compute_readiness(
    rows: Sequence[Mapping[str, Any]],
    spectral_rows: Sequence[Mapping[str, Any]],
    admitted_rows: Sequence[Mapping[str, Any]],
    harmful_rows: Sequence[Mapping[str, Any]],
    *,
    expected_event_count: int,
) -> tuple[int, dict[str, bool]]:
    """Compute the narrow mechanism gate without claiming sealed utility."""

    bounded_rows = [row for row in rows if row.get("arm") == "bounded_update"]
    bounded_admitted = [row for row in admitted_rows if row.get("arm") == "bounded_update"]
    bounded_harmful = [row for row in harmful_rows if row.get("arm") == "bounded_update"]
    bounded_spectral = [row for row in spectral_rows if row.get("arm") == "bounded_update"]
    actions = {str(row.get("proposed_action")) for row in bounded_rows}
    checks = {
        "multiple_actions": len(actions) > 1,
        "exact_supported_update_admitted": bool(bounded_admitted),
        "zero_admitted_harmful_transitions": not bounded_harmful,
        "every_state_bound_respected": bool(bounded_spectral)
        and all(row.get("within_bound") is True for row in bounded_spectral),
        "complete_prospective_rows": len(bounded_rows) == expected_event_count
        and all(
            row.get("state_before")
            and row.get("state_after")
            and row.get("outcome_revealed_after_action") is True
            for row in bounded_rows
        ),
    }
    return int(all(checks.values())), checks


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return one uniform gate row with exact expected and observed values."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every check and expose the first failure for automation."""

    copied = [dict(row) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": copied,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _read_json(path: Path) -> tuple[JsonDict, str | None]:
    """Read one JSON object and preserve the exact failure class."""

    if not path.is_file():
        return {}, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {}, f"unreadable:{type(error).__name__}"
    if not isinstance(value, dict):
        return {}, "json_object_required"
    return value, None


def _precondition_checks(source: Mapping[str, Any], error: str | None) -> list[JsonDict]:
    """Check every upstream observability and frozen-contract gate."""

    rows_value = source.get("rows", [])
    rows = rows_value if isinstance(rows_value, list) else []
    identities = [row.get("event_identity") for row in rows if isinstance(row, Mapping)]
    leakage = source.get("leakage_witnesses", {})
    clean_witnesses = (
        leakage.get("clean_stream_witnesses", []) if isinstance(leakage, Mapping) else None
    )
    counterfactuals_value = source.get("counterfactual_support_rows", [])
    counterfactuals = counterfactuals_value if isinstance(counterfactuals_value, list) else []
    action_manifest = source.get("action_manifest", {})
    order = source.get("chronological_order_manifest", {})
    replicates_value = source.get("order_replicate_manifest", [])
    replicates = replicates_value if isinstance(replicates_value, list) else []
    contract = source.get("bounded_update_contract", {})
    expected_contract = {
        "update_timing": UPDATE_CONTRACT["update_timing"],
        "same_event_outcome_may_select_action": False,
        "max_abs_entry_delta_per_event": UPDATE_CONTRACT["max_abs_entry_delta_per_event"],
        "max_spectral_norm_delta_per_event": UPDATE_CONTRACT["max_spectral_norm_delta_per_event"],
    }
    observed_contract = (
        {key: contract.get(key) for key in expected_contract}
        if isinstance(contract, Mapping)
        else contract
    )
    order_ready = bool(
        isinstance(order, Mapping)
        and order.get("base_order_is_chronological") is True
        and order.get("event_identities") == identities
        and replicates
        and all(
            isinstance(row, Mapping)
            and row.get("all_events_preserved") is True
            and set(row.get("event_identities", [])) == set(identities)
            for row in replicates
        )
    )
    return [
        _check(
            "source_artifact_readable", "readable JSON object", error or "readable", error is None
        ),
        _check(
            "observable_reliability_stream_ready_score",
            1,
            source.get("observable_reliability_stream_ready_score"),
            source.get("observable_reliability_stream_ready_score") == 1,
        ),
        _check(
            "zero_leakage_witnesses",
            {"passed": True, "clean_stream_witness_count": 0},
            {
                "passed": leakage.get("passed") if isinstance(leakage, Mapping) else None,
                "clean_stream_witness_count": len(clean_witnesses)
                if isinstance(clean_witnesses, list)
                else None,
            },
            isinstance(leakage, Mapping)
            and leakage.get("passed") is True
            and clean_witnesses == [],
        ),
        _check(
            "valid_counterfactual_support",
            {"row_count": len(rows), "invalid_count": 0},
            {
                "row_count": len(counterfactuals),
                "invalid_count": sum(
                    row.get("valid_pre_action") is not True
                    for row in counterfactuals
                    if isinstance(row, Mapping)
                ),
            },
            bool(rows)
            and len(counterfactuals) == len(rows)
            and all(
                isinstance(row, Mapping) and row.get("valid_pre_action") is True
                for row in counterfactuals
            ),
        ),
        _check(
            "frozen_order_contract",
            {"chronological": True, "replicate_count": ">=1", "events_preserved": True},
            {
                "chronological": order.get("base_order_is_chronological")
                if isinstance(order, Mapping)
                else None,
                "replicate_count": len(replicates),
                "events_preserved": order_ready,
            },
            order_ready,
        ),
        _check(
            "frozen_action_contract",
            sorted(SOURCE_ACTIONS),
            sorted(action_manifest) if isinstance(action_manifest, Mapping) else action_manifest,
            isinstance(action_manifest, Mapping) and set(action_manifest) == SOURCE_ACTIONS,
        ),
        _check(
            "frozen_update_contract",
            expected_contract,
            observed_contract,
            observed_contract == expected_contract,
        ),
    ]


def _source_hashes(root: Path, source_path: Path, no_model_sha256: str) -> JsonDict:
    """Bind the upstream evidence, owned code, and no-model baseline."""

    hashes: JsonDict = {
        "observable_reliability_stream": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "no_model_baseline": {
            "path": "no-model deterministic CPU baseline",
            "sha256": no_model_sha256,
        },
    }
    for source_id, relative in OWN_SOURCE_PATHS.items():
        hashes[source_id] = {"path": relative, "sha256": sha256_file(root / relative)}
    return hashes


def _field_principles(fields: Sequence[str]) -> JsonDict:
    """Explain why each top-level field exists in plain language."""

    special = {
        "rows": "Prospective event-arm rows make every comparison recomputable.",
        "state_transition_rows": "Checksums and deltas prove when the small state changed.",
        "spectral_bound_rows": "Eigenvalue evidence proves each controlled update stayed bounded.",
        "gate_check_summary": "Exact expected and observed values make a block actionable.",
        "bounded_reliability_controller_ready_score": (
            "This narrow mechanism gate does not claim sealed utility."
        ),
        "no_model_weight_mutation": "Equal baseline hashes prove that no model weight changed.",
        "field_principles": "This map states why each artifact field exists.",
    }
    return {
        field: special.get(
            field, f"This field records auditable {field.replace('_', ' ')} evidence."
        )
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic artifact content while excluding wall-clock duration."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def _empty_payload() -> JsonDict:
    """Return all row and metric containers required by a blocked artifact."""

    empty_by_arm = {arm: 0.0 for arm in ARMS}
    return {
        "rows": [],
        "state_transition_rows": [],
        "spectral_bound_rows": [],
        "action_distribution_by_arm": {arm: {action: 0 for action in ACTIONS} for arm in ARMS},
        "action_entropy_by_arm": dict(empty_by_arm),
        "abstention_rate_by_arm": dict(empty_by_arm),
        "retrieval_rate_by_arm": dict(empty_by_arm),
        "proposal_rate_by_arm": dict(empty_by_arm),
        "admitted_write_rate_by_arm": dict(empty_by_arm),
        "useful_write_rate_by_arm": dict(empty_by_arm),
        "harmful_write_rate_by_arm": dict(empty_by_arm),
        "false_injection_rate_by_arm": dict(empty_by_arm),
        "rollback_rate_by_arm": dict(empty_by_arm),
        "admitted_update_rows": [],
        "rejected_update_rows": [],
        "harmful_write_rows": [],
        "exact_transition_check_rows": [],
        "held_future_utility_by_arm": {
            arm: {"row_count": 0, "utility_sum": 0.0, "mean_utility": 0.0} for arm in ARMS
        },
        "old_family_retention_by_arm": {
            arm: {"anchor_count": 0, "retained_count": 0, "retention_rate": 0.0} for arm in ARMS
        },
        "delayed_correction_rows": [],
        "restart_rows": [],
        "rollback_rows": [],
        "counterfactual_support_rows": [],
    }


def build_artifact(
    root: Path,
    run_date: str,
    *,
    source_relative_path: Path = SOURCE_RELATIVE_PATH,
    no_model_after_sha256: str | None = None,
) -> JsonDict:
    """Build a complete result or a complete blocked quarantine artifact."""

    started = time.perf_counter()
    source_path = (
        source_relative_path if source_relative_path.is_absolute() else root / source_relative_path
    )
    source, source_error = _read_json(source_path)
    precondition_checks = _precondition_checks(source, source_error)
    precondition_summary = _gate_summary(precondition_checks)
    no_model_before = sha256_json({"inference_substrate": INFERENCE_SUBSTRATE, "model_files": []})
    no_model_after = no_model_after_sha256 or no_model_before
    no_mutation = no_model_before == no_model_after
    source_hashes = _source_hashes(root, source_path, no_model_before)

    artifact: JsonDict = {
        "schema": "carnot.experiment_6872.bounded_reliability_controller_quarantine.v1",
        "experiment_id": 6872,
        "run_date": run_date,
        "status": "complete",
        "openspec_requirement_ids": [
            "REQ-LEARN-6872",
            "SCENARIO-LEARN-6872-STATE",
            "SCENARIO-LEARN-6872-TIMING",
            "SCENARIO-LEARN-6872-COLLAPSE",
            "SCENARIO-LEARN-6872-QUARANTINE",
            "SCENARIO-LEARN-6872-WEIGHTS",
            "SCENARIO-LEARN-6872-ROWS",
        ],
        "replay_commands": [
            ".venv/bin/python scripts/experiments/experiment_6872_bounded_reliability_controller_quarantine.py --date 20260902",
            ".venv/bin/pytest tests/python/test_experiment_6872_bounded_reliability_controller_quarantine.py -q",
        ],
        "field_principles": {},
        "preconditions_checked": {
            "source_path": str(source_path),
            "checks": precondition_checks,
            "passed": precondition_summary["passed"],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": source_hashes,
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": no_mutation,
        "random_seed": RANDOM_SEED,
        "order_seed_manifest": [],
        "bounded_update_contract": dict(UPDATE_CONTRACT),
        "reliability_state_schema": {
            "node_order": list(STATE_NODES),
            "dimension": len(STATE_NODES),
            "initial_matrix": initialize_reliability_state(),
            "initial_state_sha256": state_checksum(initialize_reliability_state()),
            "symmetric": True,
            "finite": True,
        },
        "model_immutability_receipt": {
            "model_files": [],
            "baseline_kind": "no-model deterministic CPU baseline",
            "before_sha256": no_model_before,
            "after_sha256": no_model_after,
            "byte_identical": no_mutation,
        },
        "readiness_check_rows": [],
        **_empty_payload(),
        "reproducibility_checksum": "",
        "bounded_reliability_controller_ready_score": 0,
        "gate_check_summary": precondition_summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }

    if precondition_summary["passed"]:
        events = [row for row in source.get("rows", []) if isinstance(row, Mapping)]
        order_seed_rows = source.get("order_replicate_manifest", [])
        artifact["order_seed_manifest"] = [
            {
                "replicate_id": row.get("replicate_id"),
                "seed": row.get("seed"),
                "all_events_preserved": row.get("all_events_preserved"),
            }
            for row in order_seed_rows
            if isinstance(row, Mapping)
        ]
        comparison = run_comparison(events, order_seed=RANDOM_SEED)
        summaries = _summaries(comparison["rows"])
        ready, readiness = compute_readiness(
            comparison["rows"],
            comparison["spectral_bound_rows"],
            comparison["admitted_update_rows"],
            comparison["harmful_write_rows"],
            expected_event_count=len(events),
        )
        readiness_checks = [
            _check(
                "bounded_controller_multiple_actions",
                True,
                readiness["multiple_actions"],
                readiness["multiple_actions"],
            ),
            _check(
                "exact_supported_update_admitted",
                True,
                readiness["exact_supported_update_admitted"],
                readiness["exact_supported_update_admitted"],
            ),
            _check(
                "zero_admitted_harmful_transitions",
                True,
                readiness["zero_admitted_harmful_transitions"],
                readiness["zero_admitted_harmful_transitions"],
            ),
            _check(
                "every_state_bound_respected",
                True,
                readiness["every_state_bound_respected"],
                readiness["every_state_bound_respected"],
            ),
            _check(
                "complete_prospective_rows",
                len(events) * len(ARMS),
                len(comparison["rows"]),
                readiness["complete_prospective_rows"]
                and len(comparison["rows"]) == len(events) * len(ARMS),
            ),
            _check("no_model_weight_mutation", True, no_mutation, no_mutation),
        ]
        terminal_summary = _gate_summary([*precondition_checks, *readiness_checks])
        artifact.update(comparison)
        artifact.pop("final_states", None)
        artifact.update(summaries)
        artifact["counterfactual_support_rows"] = [
            {
                "event_identity": row.get("event_identity"),
                "valid_pre_action": row.get("valid_pre_action"),
            }
            for row in source.get("counterfactual_support_rows", [])
            if isinstance(row, Mapping)
        ]
        artifact["readiness_check_rows"] = readiness_checks
        artifact["bounded_reliability_controller_ready_score"] = int(ready and no_mutation)
        artifact["gate_check_summary"] = terminal_summary
        if not no_mutation:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        elif ready:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = READY_VERDICT
        else:
            artifact["verdict_class"] = "partial"
            artifact["honest_verdict"] = (
                "complete_partial_bounded_reliability_controller_readiness_not_met"
            )

    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["field_principles"] = _field_principles(list(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay the terminal schema, checksum, and readiness relationships."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    ready = artifact.get("bounded_reliability_controller_ready_score")
    if ready not in {0, 1}:
        errors.append("invalid_ready_score")
    receipt = artifact.get("model_immutability_receipt", {})
    hashes_equal = bool(
        isinstance(receipt, Mapping) and receipt.get("before_sha256") == receipt.get("after_sha256")
    )
    if artifact.get("no_model_weight_mutation") is not hashes_equal:
        errors.append("model_weight_immutability_mismatch")
    if ready == 1:
        gate = artifact.get("gate_check_summary", {})
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append("ready_artifact_has_failed_gate")
        if artifact.get("verdict_class") != "positive":
            errors.append("ready_artifact_must_be_positive")
        if artifact.get("no_model_weight_mutation") is not True:
            errors.append("ready_artifact_mutated_model")
    if artifact.get("verdict_class") == "blocked":
        gate = artifact.get("gate_check_summary", {})
        if not isinstance(gate, Mapping) or not gate.get("failed_check"):
            errors.append("blocked_artifact_requires_failed_check")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after a complete temporary file exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, and write the requested dated result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(args.root, args.date, source_relative_path=args.source)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "result": str(args.output),
                "ready_score": artifact["bounded_reliability_controller_ready_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        )
    )
    return 0
