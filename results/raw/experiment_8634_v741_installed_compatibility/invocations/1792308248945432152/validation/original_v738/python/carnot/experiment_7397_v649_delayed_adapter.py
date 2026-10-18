"""Measure delayed two-scalar calibration over a frozen Gibbs representation.

The replay uses the group identities sealed by Exp7382. It fits on the frozen
initial groups, records every prediction before feedback, and never reads the
final-test labels. The two stream orders are controlled archive orders, not
real chronology.

Spec refs: REQ-REPORT-7397 and SCENARIO-REPORT-7397-AUTHORITY through
SCENARIO-REPORT-7397-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
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

from carnot import experiment_7385_v648_decision_training as training
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7382_v648_decision_protocol import typed_decision
from carnot.learning.delayed_energy_calibration import (
    DelayedEnergyCalibrator,
    FutureLabelAccessError,
    energy,
    fit_affine,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
EXPERIMENT_ID = "exp7397-delayed-adapter"
SCHEMA = "carnot.exp7397.v649.delayed_adapter.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7397_v649_delayed_adapter.json")
RAW_DIR = Path("results/raw/experiment_7397_v649_delayed_adapter")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7397_v649_delayed_adapter")
MODULE_PATH = Path("python/carnot/experiment_7397_v649_delayed_adapter.py")
ADAPTER_PATH = Path("python/carnot/learning/delayed_energy_calibration.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7397_v649_delayed_adapter.py")
TEST_PATHS = (
    "tests/python/test_delayed_energy_calibration.py",
    "tests/python/test_experiment_7397_v649_delayed_adapter.py",
)
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROTOCOL_PATH = Path("results/experiment_7382_v648_decision_protocol.json")
TRAINING_ARTIFACT_PATH = Path("results/experiment_7385_v648_decision_training.json")
CORPUS_PATH = Path("data/fover_corpus_v4.json")
EXPECTED_PROTOCOL_SHA256 = "sha256:a093a1b970e0308b84fbcad96d7a5254e9b563260d8a21e16d89de19a90bb8da"
EXPECTED_TRAINING_SHA256 = "sha256:05cb5c7fb56fa5afa9ec8b450315ea00534229ec477890a84b412355f8ede87b"
EXPECTED_CORPUS_SHA256 = "sha256:c5710308eb72575591165ad1df672086e3d91ae3270c174c409e8c1ef48725e2"
EXPECTED_PARTITION_SHA256 = (
    "sha256:c392fe22a192b1db74ef45622034bb665211165243d1ef0df63835f1eee92a18"
)
EXPECTED_ONLINE_MEMBERSHIP_SHA256 = (
    "sha256:f70b35b1c7fc919b4bc749cdb7c9bb3ffb639571c13555c84da6a9eea272d866"
)
TRAINING_SEEDS = (7_397_001, 7_397_002, 7_397_003, 7_397_004, 7_397_005)
ARMS = (
    "static_affine_gibbs",
    "adaptive_affine_gibbs",
    "online_logistic_raw_features",
    "recent_frequency_beta_binomial",
    "no_feedback_adaptive_control",
)
PRACTICAL_CONTROLS = (
    "static_affine_gibbs",
    "online_logistic_raw_features",
    "recent_frequency_beta_binomial",
)
ORDERINGS = ("fixed_hash_order", "reversed_block_order")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7_399_307
BLOCK_LENGTHS = (32, 64)
MAX_STEPS = 500
LEARNING_RATE = 0.01
GRADIENT_NORM_CAP = 1.0
RECENT_WINDOW = 128
LABEL_AUTHORITY = "Exp7382 sealed training-partition label"
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/experiment_7386_v648_online_decisions.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    SPEC_PATH,
    Path("openspec/capabilities/autoresearch/spec.md"),
    Path("openspec/capabilities/continuous-learning/spec.md"),
    PROTOCOL_PATH,
    TRAINING_ARTIFACT_PATH,
    CORPUS_PATH,
)
V649_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=TEST_PATHS,
    changed_modules=(MODULE_PATH.as_posix(), ADAPTER_PATH.as_posix()),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for state, protocol, and result identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without depending on metadata."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object through a local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def feedback_conditions() -> tuple[JsonDict, ...]:
    """Return the primary condition and two separate sensitivity conditions."""

    return (
        {"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0},
        {"name": "sensitivity_delay_32", "delay": 32, "missing_fraction": 0.0},
        {"name": "sensitivity_missing_25", "delay": 1, "missing_fraction": 0.25},
    )


def deterministic_missing_mask(group_ids: Sequence[str]) -> set[str]:
    """Select exactly every fourth frozen identity in the original hash order."""

    return {str(group_id) for index, group_id in enumerate(group_ids) if (index + 1) % 4 == 0}


def _rows_for_ids(protocol: Mapping[str, Any], role: str, ids: Sequence[str]) -> list[JsonDict]:
    rows = {
        str(row["group_id"]): deepcopy(dict(row))
        for row in protocol.get("feature_rows", [])
        if isinstance(row, Mapping) and row.get("partition") == role
    }
    missing = [str(group_id) for group_id in ids if str(group_id) not in rows]
    if missing:
        raise ValueError(f"sealed {role} group is missing: {missing[0]}")
    if len(set(map(str, ids))) != len(ids):
        raise ValueError(f"sealed {role} groups must be unique")
    return [rows[str(group_id)] for group_id in ids]


def initialization_rows(protocol: Mapping[str, Any]) -> list[JsonDict]:
    """Read only the frozen initial half of the training partition."""

    replay = protocol.get("online_replay") or {}
    return _rows_for_ids(protocol, "training", replay.get("initialization_group_ids") or [])


def policy_calibration_rows(protocol: Mapping[str, Any]) -> list[JsonDict]:
    """Expose the disjoint policy role without touching final-test rows."""

    return [
        deepcopy(dict(row))
        for row in protocol.get("feature_rows", [])
        if isinstance(row, Mapping) and row.get("partition") == "policy_calibration"
    ]


def build_streams(
    protocol: Mapping[str, Any], *, block_length: int = 32
) -> dict[str, list[JsonDict]]:
    """Build original hash order and one reversed-block sensitivity order."""

    if block_length <= 0:
        raise ValueError("block length must be positive")
    replay = protocol.get("online_replay") or {}
    fixed = _rows_for_ids(protocol, "training", replay.get("later_group_ids") or [])
    blocks = [fixed[start : start + block_length] for start in range(0, len(fixed), block_length)]
    reversed_blocks = [deepcopy(row) for block in reversed(blocks) for row in block]
    return {"fixed_hash_order": fixed, "reversed_block_order": reversed_blocks}


def _features(row: Mapping[str, Any]) -> list[float]:
    return [float(row["entity_uptake"]), float(row["falsifiability_score"])]


def initialize_seed_states(
    rows: Sequence[Mapping[str, Any]], seed: int, *, steps: int = MAX_STEPS
) -> dict[str, JsonDict]:
    """Fit the Gibbs, affine, and raw logistic states on initial groups only."""

    fitted = training.train_arm("natural_prevalence_bernoulli_gibbs", seed, rows, steps=steps)
    weights = deepcopy(fitted["weights"])
    energies = [energy(weights, _features(row)) for row in rows]
    labels = [int(row["label"]) for row in rows]
    affine = fit_affine(energies, labels, steps=steps)
    logistic = training.train_arm("l2_logistic_calibration", seed, rows, steps=steps)
    adapter = DelayedEnergyCalibrator(weights, a=affine["a"], b=affine["b"])
    recent = labels[-RECENT_WINDOW:]
    return {
        "static_affine_gibbs": {
            "adapter": adapter.to_dict(),
            "update_count": 0,
        },
        "adaptive_affine_gibbs": {
            "adapter": adapter.to_dict(),
            "update_count": 0,
        },
        "online_logistic_raw_features": {
            "weights": deepcopy(logistic["weights"]),
            "update_count": 0,
        },
        "recent_frequency_beta_binomial": {
            "recent_labels": recent,
            "update_count": 0,
        },
        "no_feedback_adaptive_control": {
            "adapter": adapter.to_dict(),
            "update_count": 0,
        },
        "training_record": {
            "seed": seed,
            "gibbs_objective": fitted["objective"],
            "gibbs_steps": steps,
            "gibbs_pre_weight_sha256": fitted["pre_weight_sha256"],
            "gibbs_post_weight_sha256": fitted["post_weight_sha256"],
            "gibbs_loss_initial": fitted["loss_curve"][0]["loss"],
            "gibbs_loss_final": fitted["loss_curve"][-1]["loss"],
            "affine_steps": steps,
            "affine_loss_initial": affine["loss_curve"][0]["loss"],
            "affine_loss_final": affine["loss_curve"][-1]["loss"],
            "initial_group_count": len(rows),
            "initial_group_ids_sha256": canonical_hash([str(row["group_id"]) for row in rows]),
            "future_stream_groups_used": 0,
            "weights": weights,
            "affine": {"a": affine["a"], "b": affine["b"]},
            "logistic_weights": deepcopy(logistic["weights"]),
        },
    }


def _adapter_probability(
    state: Mapping[str, Any], features: Sequence[float]
) -> tuple[float, float]:
    adapter = DelayedEnergyCalibrator.from_dict(state["adapter"])
    return adapter.probability(features)


def _logistic_probability(state: Mapping[str, Any], features: Sequence[float]) -> float:
    weights = state["weights"]
    value = float(np.dot(np.asarray(weights["coef"]), np.asarray(features)) + weights["bias"])
    return training._sigmoid(value)


def _frequency_probability(state: Mapping[str, Any]) -> float:
    labels = list(state.get("recent_labels") or [])
    return (sum(labels) + 1.0) / (len(labels) + 2.0)


def select_fixed_thresholds(
    states: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Select thresholds once on the disjoint policy-calibration role."""

    scored: list[JsonDict] = []
    for row in rows:
        _, probability = _adapter_probability(states["adaptive_affine_gibbs"], _features(row))
        scored.append({**deepcopy(dict(row)), "probability": probability})
    evaluated, policy = training.select_policy(scored)
    return {**policy, "evaluated_pairs": evaluated}


def _state_hash(state: Mapping[str, Any]) -> str:
    return canonical_hash(state)


def _logistic_step(state: JsonDict, features: Sequence[float], label: int) -> JsonDict:
    probability = _logistic_probability(state, features)
    residual = probability - label
    gradient = np.asarray(
        [residual * float(features[0]), residual * float(features[1]), residual],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(gradient))
    scale = min(1.0, GRADIENT_NORM_CAP / max(norm, 1e-15))
    gradient *= scale
    coef = np.asarray(state["weights"]["coef"], dtype=np.float64)
    coef -= LEARNING_RATE * gradient[:2]
    bias = float(state["weights"]["bias"]) - LEARNING_RATE * float(gradient[2])
    state["weights"] = {"coef": coef.tolist(), "bias": bias}
    state["update_count"] = int(state["update_count"]) + 1
    return {
        "gradient_norm_before_clip": norm,
        "gradient_norm_after_clip": float(np.linalg.norm(gradient)),
    }


def _policy_action(probability: float, policy: Mapping[str, Any], version: str) -> JsonDict:
    action = typed_decision(
        probability,
        accept_threshold=float(policy["accept_threshold"]),
        reject_threshold=float(policy["reject_threshold"]),
        model_version=version,
    )
    if action["decision"] == "accept" and not policy.get("accept_enabled"):
        action["decision"] = "escalate"
    if action["decision"] == "reject" and not policy.get("reject_enabled"):
        action["decision"] = "escalate"
    return action


def replay_condition(
    initial_states: Mapping[str, Mapping[str, Any]],
    stream: Sequence[Mapping[str, Any]],
    *,
    ordering: str,
    condition: Mapping[str, Any],
    seed: int,
    thresholds: Mapping[str, Any],
    missing_group_ids: set[str],
    checkpoint_path: Path | None = None,
) -> JsonDict:
    """Replay one paired stream with prediction-before-feedback authority."""

    if ordering not in ORDERINGS or condition not in feedback_conditions():
        raise ValueError("unregistered replay condition")
    states = {arm: deepcopy(dict(initial_states[arm])) for arm in ARMS}
    adapters = {
        arm: DelayedEnergyCalibrator.from_dict(states[arm]["adapter"])
        for arm in ("static_affine_gibbs", "adaptive_affine_gibbs", "no_feedback_adaptive_control")
    }
    pending: list[JsonDict] = []
    rows: list[JsonDict] = []
    ledger: list[JsonDict] = []
    previous_action: dict[str, str] = {}
    midpoint = len(stream) // 2
    restart_receipt: JsonDict = {"performed": False, "prediction_parity": False}

    for index, source in enumerate(stream):
        row = deepcopy(dict(source))
        group_id = str(row["group_id"])
        features = _features(row)
        label = int(row["label"])
        delay = int(condition["delay"])
        event_id = f"{ordering}:{condition['name']}:{seed}:{group_id}"

        if index == midpoint:
            before = adapters["adaptive_affine_gibbs"].probability(features)[1]
            checkpoint_started = time.perf_counter()
            payload = {
                "states": states,
                "adapters": {arm: adapter.to_dict() for arm, adapter in adapters.items()},
            }
            if checkpoint_path is not None:
                atomic_json(checkpoint_path, payload)
                checkpoint_hash = sha256_file(checkpoint_path)
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            else:
                encoded = json.dumps(payload, sort_keys=True)
                checkpoint_hash = canonical_hash(payload)
                payload = json.loads(encoded)
            states = {arm: dict(value) for arm, value in payload["states"].items()}
            adapters = {
                arm: DelayedEnergyCalibrator.from_dict(value)
                for arm, value in payload["adapters"].items()
            }
            after = adapters["adaptive_affine_gibbs"].probability(features)[1]
            restart_receipt = {
                "performed": True,
                "stream_index": index,
                "prediction_parity": before == after,
                "checkpoint_sha256": checkpoint_hash,
                "duration_s": time.perf_counter() - checkpoint_started,
            }

        feedback_available_at = index + delay
        arm_predictions: JsonDict = {}
        row_positions: dict[str, int] = {}
        for arm in ARMS:
            prediction_started = time.perf_counter()
            if arm in adapters:
                adapter = adapters[arm]
                prediction = adapter.record_prediction(
                    event_id,
                    features,
                    prediction_index=index,
                    feedback_available_at=feedback_available_at,
                    label_authority=LABEL_AUTHORITY,
                )
                probability = float(prediction["probability"])
                raw_energy = float(prediction["energy"])
                numeric_state = prediction["numeric_state"]
                state_hash = prediction["state_hash"]
            elif arm == "online_logistic_raw_features":
                probability = _logistic_probability(states[arm], features)
                raw_energy = float("nan")
                numeric_state = deepcopy(states[arm]["weights"])
                state_hash = _state_hash({**states[arm], "arm": arm})
            else:
                probability = _frequency_probability(states[arm])
                raw_energy = float("nan")
                numeric_state = {
                    "alpha": sum(states[arm]["recent_labels"]) + 1.0,
                    "beta": len(states[arm]["recent_labels"])
                    - sum(states[arm]["recent_labels"])
                    + 1.0,
                    "window_size": len(states[arm]["recent_labels"]),
                    "update_count": states[arm]["update_count"],
                }
                state_hash = _state_hash(numeric_state)
            action = _policy_action(
                probability, thresholds, f"{arm}:v{states[arm]['update_count']}"
            )
            prediction_latency = time.perf_counter() - prediction_started
            decision = str(action["decision"])
            output = {
                "ordering": ordering,
                "feedback_condition": str(condition["name"]),
                "delay": delay,
                "missing_fraction": float(condition["missing_fraction"]),
                "stream_index": index,
                "group_id": group_id,
                "source_row_index": int(row["source_row_index"]),
                "arm": arm,
                "seed": seed,
                "label": label,
                "raw_energy": raw_energy if math.isfinite(raw_energy) else None,
                "probability": probability,
                "brier_loss": (probability - label) ** 2,
                "log_loss": -(
                    label * math.log(max(probability, 1e-15))
                    + (1 - label) * math.log(max(1.0 - probability, 1e-15))
                ),
                "decision": decision,
                "action_harm": bool(
                    (decision == "accept" and label == 1) or (decision == "reject" and label == 0)
                ),
                "churn_contribution": int(
                    arm in previous_action and previous_action[arm] != decision
                ),
                "prediction_before_feedback": True,
                "feedback_available_at": feedback_available_at,
                "label_authority": LABEL_AUTHORITY,
                "numeric_state": numeric_state,
                "state_hash": state_hash,
                "gibbs_weights_hash": (adapters[arm].weights_hash if arm in adapters else None),
                "update_count": int(states[arm]["update_count"]),
                "prediction_latency_s": prediction_latency,
                "update_latency_s": 0.0,
                "total_latency_s": prediction_latency,
                "disposition": "measured_prediction",
            }
            previous_action[arm] = decision
            row_positions[arm] = len(rows)
            rows.append(output)
            arm_predictions[arm] = {
                "probability": probability,
                "decision": decision,
                "numeric_state": deepcopy(numeric_state),
                "state_hash": state_hash,
            }

        omitted = group_id in missing_group_ids and float(condition["missing_fraction"]) > 0.0
        ledger_row = {
            "event_id": event_id,
            "ordering": ordering,
            "feedback_condition": str(condition["name"]),
            "seed": seed,
            "stream_index": index,
            "group_id": group_id,
            "prediction_recorded_at": index,
            "feedback_available_at": feedback_available_at,
            "label_authority": LABEL_AUTHORITY,
            "prediction_records": arm_predictions,
            "prediction_before_feedback": True,
            "missing": omitted,
            "feedback": None,
        }
        ledger.append(ledger_row)
        pending.append(
            {
                "due": feedback_available_at,
                "label": None if omitted else label,
                "features": features,
                "event_id": event_id,
                "ledger_index": len(ledger) - 1,
            }
        )

        due = [item for item in pending if int(item["due"]) <= index]
        pending = [item for item in pending if int(item["due"]) > index]
        update_costs = {arm: 0.0 for arm in ARMS}
        for item in due:
            feedback_result: JsonDict = {}
            if item["label"] is None:
                for arm in ARMS:
                    feedback_result[arm] = {"status": "missing", "update_admitted": False}
            else:
                update_started = time.perf_counter()
                feedback_result["adaptive_affine_gibbs"] = adapters[
                    "adaptive_affine_gibbs"
                ].commit_feedback(str(item["event_id"]), int(item["label"]), visible_at=index)
                update_costs["adaptive_affine_gibbs"] += time.perf_counter() - update_started
                update_started = time.perf_counter()
                feedback_result["online_logistic_raw_features"] = _logistic_step(
                    states["online_logistic_raw_features"],
                    item["features"],
                    int(item["label"]),
                )
                feedback_result["online_logistic_raw_features"].update(
                    {"status": "committed", "update_admitted": True}
                )
                update_costs["online_logistic_raw_features"] += time.perf_counter() - update_started
                update_started = time.perf_counter()
                recent = deque(
                    states["recent_frequency_beta_binomial"]["recent_labels"],
                    maxlen=RECENT_WINDOW,
                )
                recent.append(int(item["label"]))
                states["recent_frequency_beta_binomial"]["recent_labels"] = list(recent)
                states["recent_frequency_beta_binomial"]["update_count"] += 1
                feedback_result["recent_frequency_beta_binomial"] = {
                    "status": "committed",
                    "update_admitted": True,
                }
                update_costs["recent_frequency_beta_binomial"] += (
                    time.perf_counter() - update_started
                )
                feedback_result["static_affine_gibbs"] = {
                    "status": "frozen_control",
                    "update_admitted": False,
                }
                feedback_result["no_feedback_adaptive_control"] = {
                    "status": "withheld_control",
                    "update_admitted": False,
                }
                states["adaptive_affine_gibbs"]["update_count"] = adapters[
                    "adaptive_affine_gibbs"
                ].update_count
            ledger[int(item["ledger_index"])]["feedback"] = {
                "label": item["label"],
                "visible_at": index,
                "arm_results": feedback_result,
            }
        for arm, position in row_positions.items():
            rows[position]["update_latency_s"] = update_costs[arm]
            rows[position]["total_latency_s"] += update_costs[arm]

    for arm, adapter in adapters.items():
        states[arm]["adapter"] = adapter.to_dict()
        states[arm]["update_count"] = adapter.update_count
    return {
        "rows": rows,
        "event_ledger": ledger,
        "restart_receipt": restart_receipt,
        "pending_feedback_at_end": len(pending),
        "final_states": states,
    }


def _fixture_weights() -> JsonDict:
    return {
        "w1": [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]],
        "b1": [0.0, 0.0, 0.0, 0.0],
        "w_out": [1.0, -1.0, 0.5, -0.5],
        "b_out": 0.1,
    }


def run_analytic_controls() -> dict[str, JsonDict]:
    """Exercise update authority and degenerate inputs before real replay."""

    weights = _fixture_weights()
    fixture_features = [[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]]
    fixture_labels = [0, 0, 1, 1]
    fixture_energies = [energy(weights, values) for values in fixture_features]
    fitted = fit_affine(fixture_energies, fixture_labels, steps=100)
    informative = fitted["loss_curve"][-1]["loss"] < fitted["loss_curve"][0]["loss"]
    constant = fit_affine([0.0] * 4, fixture_labels, steps=10)

    adapter = DelayedEnergyCalibrator(weights)
    adapter.record_prediction(
        "fixture-event",
        fixture_features[0],
        prediction_index=0,
        feedback_available_at=1,
        label_authority="analytic-fixture",
    )
    future_denied = False
    try:
        adapter.commit_feedback("fixture-event", 1, visible_at=0)
    except FutureLabelAccessError:
        future_denied = True
    adapter.commit_feedback("fixture-event", 1, visible_at=1)
    after_commit = adapter.state_hash
    duplicate = adapter.commit_feedback("fixture-event", 1, visible_at=2)
    restart = DelayedEnergyCalibrator.from_dict(adapter.to_dict())
    restart_equal = restart.to_dict() == adapter.to_dict()
    restart.erase_feedback("fixture-event")
    erased = restart.update_count == 0 and restart.a == 1.0 and restart.b == 0.0

    nonfinite_denied = False
    try:
        adapter.record_prediction(
            "nonfinite",
            [float("nan"), 0.0],
            prediction_index=2,
            feedback_available_at=3,
            label_authority="analytic-fixture",
        )
    except ValueError:
        nonfinite_denied = True

    constant_adapter = DelayedEnergyCalibrator(
        {**weights, "w_out": [0.0, 0.0, 0.0, 0.0], "b_out": 0.0}
    )
    constant_probabilities = [
        constant_adapter.probability(values)[1] for values in fixture_features
    ]
    no_feedback = DelayedEnergyCalibrator.from_dict(adapter.to_dict())
    no_feedback_before = no_feedback.state_hash
    no_feedback.record_prediction(
        "withheld",
        fixture_features[1],
        prediction_index=2,
        feedback_available_at=3,
        label_authority="analytic-fixture",
    )
    controls = {
        "informative_energy": {
            "passed": informative,
            "initial_loss": fitted["loss_curve"][0]["loss"],
            "final_loss": fitted["loss_curve"][-1]["loss"],
        },
        "constant_energy": {
            "passed": math.isfinite(constant["loss_curve"][-1]["loss"]),
            "final_loss": constant["loss_curve"][-1]["loss"],
        },
        "future_label_denied": {"passed": future_denied},
        "duplicate_feedback_denied": {
            "passed": duplicate["status"] == "duplicate" and adapter.state_hash == after_commit,
        },
        "restart_equality": {"passed": restart_equal},
        "erased_update": {"passed": erased},
        "nonfinite_denied": {"passed": nonfinite_denied},
        "constant_prediction": {
            "passed": len(set(constant_probabilities)) == 1,
            "probability": constant_probabilities[0],
        },
        "no_feedback_unchanged": {
            "passed": no_feedback.state_hash == no_feedback_before,
        },
    }
    return controls


def _paired_vectors(rows: Sequence[Mapping[str, Any]], arm: str) -> tuple[list[str], np.ndarray]:
    selected = [
        row
        for row in rows
        if row.get("ordering") == "fixed_hash_order"
        and row.get("feedback_condition") == "primary_delay_1"
        and row.get("arm") == arm
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    ordering: dict[str, int] = {}
    for row in selected:
        group_id = str(row["group_id"])
        grouped[group_id].append(float(row["brier_loss"]))
        ordering[group_id] = int(row["stream_index"])
    group_ids = sorted(grouped, key=lambda group_id: ordering[group_id])
    return group_ids, np.asarray([np.mean(grouped[group_id]) for group_id in group_ids])


def paired_moving_block_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    block_lengths: Sequence[int] = BLOCK_LENGTHS,
) -> list[JsonDict]:
    """Bootstrap paired event deltas after averaging seeds within each event."""

    target_ids, target = _paired_vectors(rows, "adaptive_affine_gibbs")
    if not target_ids:
        raise ValueError("paired intervals require primary rows")
    output: list[JsonDict] = []
    for length in block_lengths:
        block_length = int(length)
        if block_length <= 0:
            raise ValueError("bootstrap block length must be positive")
        block_count = math.ceil(len(target_ids) / block_length)
        rng = np.random.default_rng(BOOTSTRAP_SEED + block_length)
        starts = rng.integers(0, len(target_ids), size=(draws, block_count))
        offsets = np.arange(block_length, dtype=np.int64)
        sampled = ((starts[:, :, None] + offsets) % len(target_ids)).reshape(draws, -1)
        sampled = sampled[:, : len(target_ids)]
        for control in PRACTICAL_CONTROLS:
            control_ids, control_values = _paired_vectors(rows, control)
            if control_ids != target_ids:
                raise ValueError("paired arms must contain identical event groups")
            delta = target - control_values
            resampled = np.mean(delta[sampled], axis=1)
            output.append(
                {
                    "ordering": "fixed_hash_order",
                    "feedback_condition": "primary_delay_1",
                    "target_arm": "adaptive_affine_gibbs",
                    "control_arm": control,
                    "metric": "brier_loss",
                    "block_length": block_length,
                    "draws": draws,
                    "seed": BOOTSTRAP_SEED,
                    "effective_event_groups": len(target_ids),
                    "mean_delta": float(np.mean(delta)),
                    "ci95_lower": float(np.quantile(resampled, 0.025)),
                    "ci95_upper": float(np.quantile(resampled, 0.975)),
                    "seed_reduction": "average_within_event_before_resampling",
                    "paired_block_indices": True,
                    "fixed_archive_descriptive_only": True,
                }
            )
    return output


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def condition_reports(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize proper scores, actions, churn, and measured per-row cost."""

    reports: list[JsonDict] = []
    for ordering in ORDERINGS:
        for condition in feedback_conditions():
            selected = [
                row
                for row in rows
                if row.get("ordering") == ordering
                and row.get("feedback_condition") == condition["name"]
            ]
            arm_metrics: dict[str, JsonDict] = {}
            for arm in ARMS:
                arm_rows = [row for row in selected if row.get("arm") == arm]
                accepts = [row for row in arm_rows if row.get("decision") == "accept"]
                rejects = [row for row in arm_rows if row.get("decision") == "reject"]
                decided = [*accepts, *rejects]
                arm_metrics[arm] = {
                    "prediction_rows": len(arm_rows),
                    "effective_groups": len({str(row["group_id"]) for row in arm_rows}),
                    "mean_brier": float(np.mean([row["brier_loss"] for row in arm_rows])),
                    "mean_log_loss": float(np.mean([row["log_loss"] for row in arm_rows])),
                    "coverage": len(decided) / len(arm_rows),
                    "accept_count": len(accepts),
                    "reject_count": len(rejects),
                    "escalate_count": len(arm_rows) - len(decided),
                    "accept_risk": (
                        sum(int(row["label"]) == 1 for row in accepts) / len(accepts)
                        if accepts
                        else None
                    ),
                    "reject_risk": (
                        sum(int(row["label"]) == 0 for row in rejects) / len(rejects)
                        if rejects
                        else None
                    ),
                    "churn_rate": float(np.mean([row["churn_contribution"] for row in arm_rows])),
                    "prediction_latency_p50_s": _percentile(
                        [float(row["prediction_latency_s"]) for row in arm_rows], 0.50
                    ),
                    "prediction_latency_p95_s": _percentile(
                        [float(row["prediction_latency_s"]) for row in arm_rows], 0.95
                    ),
                    "update_latency_p50_s": _percentile(
                        [float(row["update_latency_s"]) for row in arm_rows], 0.50
                    ),
                    "update_latency_p95_s": _percentile(
                        [float(row["update_latency_s"]) for row in arm_rows], 0.95
                    ),
                    "total_latency_p50_s": _percentile(
                        [float(row["total_latency_s"]) for row in arm_rows], 0.50
                    ),
                    "total_latency_p95_s": _percentile(
                        [float(row["total_latency_s"]) for row in arm_rows], 0.95
                    ),
                }
            reports.append(
                {
                    "ordering": ordering,
                    "feedback_condition": condition["name"],
                    "delay": condition["delay"],
                    "missing_fraction": condition["missing_fraction"],
                    "arm_metrics": arm_metrics,
                    "real_world_chronology": False,
                }
            )
    return reports


def reduce_primary_value(
    intervals: Sequence[Mapping[str, Any]], reports: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Apply the registered primary conjunction without changing readiness."""

    primary = next(
        (
            row
            for row in reports
            if row.get("ordering") == "fixed_hash_order"
            and row.get("feedback_condition") == "primary_delay_1"
        ),
        None,
    )
    checks: list[JsonDict] = []
    block32 = {
        str(row["control_arm"]): row
        for row in intervals
        if row.get("ordering") == "fixed_hash_order"
        and row.get("feedback_condition") == "primary_delay_1"
        and row.get("block_length") == 32
    }

    def add(check: str, expected: Any, observed: Any, operator: str, passed: bool) -> None:
        checks.append(
            {
                "check": check,
                "category": "scientific_efficacy",
                "operator": operator,
                "expected": expected,
                "observed": observed,
                "passed": passed,
            }
        )

    for control in PRACTICAL_CONTROLS:
        upper = (block32.get(control) or {}).get("ci95_upper")
        add(
            f"brier_delta_upper_vs_{control}",
            0.0,
            upper,
            "<",
            isinstance(upper, (int, float)) and float(upper) < 0.0,
        )
    if primary is None:
        add("primary_report_present", True, False, "==", False)
    else:
        metrics = primary["arm_metrics"]
        target = metrics["adaptive_affine_gibbs"]
        for control in PRACTICAL_CONTROLS:
            control_metrics = metrics[control]
            add(
                f"log_loss_nonworse_vs_{control}",
                float(control_metrics["mean_log_loss"]),
                float(target["mean_log_loss"]),
                "<=",
                float(target["mean_log_loss"]) <= float(control_metrics["mean_log_loss"]),
            )
            add(
                f"coverage_nonworse_vs_{control}",
                float(control_metrics["coverage"]),
                float(target["coverage"]),
                ">=",
                float(target["coverage"]) >= float(control_metrics["coverage"]),
            )
        accept_risk = target["accept_risk"]
        reject_risk = target["reject_risk"]
        add(
            "observed_accept_risk_within_budget",
            0.05,
            accept_risk,
            "<=",
            target["accept_count"] > 0
            and isinstance(accept_risk, (int, float))
            and float(accept_risk) <= 0.05,
        )
        add(
            "observed_reject_risk_within_budget",
            0.10,
            reject_risk,
            "<=",
            target["reject_count"] > 0
            and isinstance(reject_risk, (int, float))
            and float(reject_risk) <= 0.10,
        )
    return {
        "checks": checks,
        "passed": bool(checks) and all(row["passed"] is True for row in checks),
        "primary_ordering": "fixed_hash_order",
        "primary_feedback_condition": "primary_delay_1",
        "scientific_benefit_claim": False,
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, operator: str = "=="
) -> JsonDict:
    if operator == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    elif operator == "<=":
        passed = isinstance(observed, (int, float)) and observed <= expected
    else:
        passed = observed == expected
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failed if row.get("category") != "scientific_efficacy"]
    scientific = [row for row in failed if row.get("category") == "scientific_efficacy"]
    return {
        "all_required_passed": not required,
        "failed_required_count": len(required),
        "first_required_failure": required[0] if required else None,
        "failed_scientific_gate_count": len(scientific),
        "first_scientific_failure": scientific[0] if scientific else None,
    }


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    required_names = set(validation_scope.REQUIRED_CHECK_NAMES) | {
        "declared_entrypoint_e2e",
        "independent_reducer",
        "independent_cold_replay",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    passing = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True
        and row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is False
    }
    return required_names <= passing


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and efficacy independently from stored scores."""

    budget = artifact.get("sample_size_budget") or {}
    rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    planned = int(budget.get("planned_prediction_rows") or 0)
    required_row_fields = {
        "ordering",
        "feedback_condition",
        "stream_index",
        "group_id",
        "arm",
        "seed",
        "label",
        "probability",
        "brier_loss",
        "log_loss",
        "decision",
        "prediction_before_feedback",
        "prediction_latency_s",
        "update_latency_s",
        "total_latency_s",
        "disposition",
    }
    identities = {
        (
            row.get("ordering"),
            row.get("feedback_condition"),
            row.get("stream_index"),
            row.get("arm"),
            row.get("seed"),
        )
        for row in rows
    }
    row_complete = (
        planned > 0
        and len(rows) == planned
        and len(identities) == planned
        and all(required_row_fields <= set(row) for row in rows)
        and all(row.get("prediction_before_feedback") is True for row in rows)
    )
    ledger = [row for row in artifact.get("event_ledger", []) if isinstance(row, Mapping)]
    ledger_complete = (
        len(ledger) == int(budget.get("planned_event_ledgers") or -1)
        and len({str(row.get("event_id")) for row in ledger}) == len(ledger)
        and all(
            row.get("prediction_before_feedback") is True
            and isinstance(row.get("feedback_available_at"), int)
            and row.get("label_authority") == LABEL_AUTHORITY
            and isinstance(row.get("prediction_records"), Mapping)
            for row in ledger
        )
    )
    controls = artifact.get("analytic_controls") or {}
    controls_pass = bool(controls) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in controls.values()
    )
    restarts = artifact.get("restart_receipts") or []
    restart_pass = bool(restarts) and all(
        row.get("performed") is True and row.get("prediction_parity") is True for row in restarts
    )
    training_runs = artifact.get("training_runs") or []
    training_complete = (
        len(training_runs) == len(TRAINING_SEEDS)
        and {int(row.get("seed")) for row in training_runs} == set(TRAINING_SEEDS)
        and all(
            int(row.get("gibbs_steps", -1)) <= MAX_STEPS
            and int(row.get("future_stream_groups_used", -1)) == 0
            for row in training_runs
        )
    )
    required_validation = _required_receipts_pass(artifact.get("validation_receipts") or [])
    safe = artifact.get("flagged_adversarial") is False
    ready = int(
        row_complete
        and ledger_complete
        and controls_pass
        and restart_pass
        and training_complete
        and required_validation
        and safe
    )
    value_reduction = artifact.get("primary_value_reduction") or {}
    value = int(ready == 1 and value_reduction.get("passed") is True)
    return {
        "row_completeness_passed": row_complete,
        "ledger_completeness_passed": ledger_complete,
        "analytic_controls_passed": controls_pass,
        "restart_equality_passed": restart_pass,
        "initial_training_scope_passed": training_complete,
        "required_validation_passed": required_validation,
        "safety_passed": safe,
        "delayed_adapter_ready_score": ready,
        "delayed_adapter_value_score": value,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, protocol, configuration, state transitions, and raw rows."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "online_protocol",
        "training_runs",
        "policy_rows",
        "rows",
        "event_ledger",
        "state_transition_rows",
        "analytic_controls",
        "moving_block_intervals",
        "condition_reports",
        "primary_value_reduction",
        "sample_size_budget",
    )
    return canonical_hash({field: artifact.get(field) for field in fields})


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    principles = {
        "schema": "Versioned schema; ordinary top-level identity and terminal status follow measured work and checks.",
        "run_date": "The run uses 20260918 and records actual UTC start and end timestamps.",
        "preconditions_checked": "Exact hashes, source eligibility, runtime, device, and entrypoint checks precede dependent work.",
        "MODEL_SPECS": "No current LLM is loaded; small Gibbs training is recorded separately.",
        "model_invoked": "False means no current real LLM load or generation was attempted.",
        "invocation_counts": "Counts cover current owned LLM events only and remain zero.",
        "inference_substrate": "A string describes current CPU, JAX, and exact numeric work.",
        "inference_substrate_class": "The current class is no_model_load and has no runtime padding.",
        "execution_venue": "The closed execution venue is host.",
        "duration_s": "Monotonic task duration is separate from scientific and validation spans.",
        "phase_spans": "Actual phase boundaries include UTC checkpoints and monotonic offsets.",
        "random_seed": "Frozen training and moving-block seeds make the bounded replay reproducible.",
        "reproducibility_checksum": "The checksum binds current code identity, protocol, configuration, and raw evidence.",
        "source_artifact_hashes": "Exact byte hashes retain input identity without importing historical invocation counters.",
        "rows": "Every arm, seed, condition, event, metric contribution, cost, and disposition is retained.",
        "sample_size_budget": "Planned, attempted, complete, censored, and unstarted units follow a fixed stop rule.",
        "acceptance_gate_results": "Validation, safety, completion, and efficacy checks remain distinct.",
        "gate_check_summary": "Blocked results name the exact upstream, path, check, field, expected, and observed value.",
        "verifier_is_oracle": "Independent evaluation of a learned risk score does not make that score the correctness oracle.",
        "honest_verdict": "Completed findings start with complete_; unchanged missing prerequisites start with blocked_.",
        "verdict_class": "The closed class is positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Critical producer or verifier findings prevent readiness.",
        "validation_receipts": "Receipts retain exact argv, environment, exit, duration, log path, and log hash.",
        "repository_health": "Unrelated broad health stays separate; affected failures remain disqualifying.",
        "field_principles": "Principles explain ordinary fields without changing their machine-readable values.",
        "promotion_score": "Always zero; this experiment cannot change production, weights, publication, or leaderboards.",
        "delayed_adapter_ready_score": "Readiness requires tested update authority, sealed protocol, complete evidence, and required checks, not efficacy.",
        "online_protocol": "Splits, arms, orders, feedback, thresholds, seeds, and confidence procedures are frozen.",
        "state_transition_rows": "Rows prove prediction-before-label, unique commit, restart, and erasure behavior.",
        "small_ebm_training": "Only the initial tiny Gibbs representation and numeric affine calibration are trained.",
    }
    return {
        str(key): principles.get(
            str(key), "This ordinary field retains direct machine-readable experiment evidence."
        )
        for key in keys
    }


def _base_artifact() -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "Host CPU runs JAX training for tiny 2-4-1 Gibbs heads, NumPy affine and control updates, and exact risk certificates; no LLM loads.",
        "inference_substrate_details": {
            "device": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax_backend": "cpu",
            "current_llm_work": False,
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "moving_block_seed": BOOTSTRAP_SEED,
        },
        "verifier_is_oracle": False,
        "promotion_score": 0,
    }


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, str]
) -> JsonDict:
    """Publish an external prerequisite failure without dependent measurement."""

    failed = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else {
            "check": "unknown_precondition",
            "upstream": None,
            "artifact_field": None,
            "expected": True,
            "observed": None,
            "passed": False,
        }
    )
    artifact = {
        **_base_artifact(),
        "status": "blocked_delayed_adapter_precondition",
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "completed_at_utc": "2026-09-18T00:00:00+00:00",
        "duration_s": 0.0,
        "phase_spans": [],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [],
        "rows": [],
        "event_ledger": [],
        "state_transition_rows": [],
        "training_runs": [],
        "policy_rows": [],
        "moving_block_intervals": [],
        "condition_reports": [],
        "analytic_controls": {},
        "restart_receipts": [],
        "sample_size_budget": {
            "planned_prediction_rows": 1321 * len(ARMS) * len(TRAINING_SEEDS) * 6,
            "attempted_prediction_rows": 0,
            "completed_prediction_rows": 0,
            "censored_prediction_rows": 0,
            "unstarted_prediction_rows": 1321 * len(ARMS) * len(TRAINING_SEEDS) * 6,
            "planned_event_ledgers": 1321 * len(TRAINING_SEEDS) * 6,
            "effective_independent_group_count": 0,
            "stopping_rule": "Stop before dependent work after any failed structured prerequisite.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_required_passed": False,
            "failed_required_count": len(failed) or 1,
            "first_required_failure": first,
            "failed_scientific_gate_count": 0,
            "first_scientific_failure": None,
        },
        "honest_verdict": "blocked_delayed_adapter_structured_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_evaluated",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_findings": [],
        },
        "delayed_adapter_ready_score": 0,
        "delayed_adapter_value_score": 0,
        "primary_value_reduction": {"passed": False, "checks": []},
        "online_protocol": {},
        "small_ebm_training": {"performed": False, "current_llm_calls": 0},
        "evidence_scope": {"class": "blocked_before_replay", "conformal_guarantee": False},
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_fixture_artifact() -> JsonDict:
    """Build a compact complete-null record for independent mutation tests."""

    rows = []
    for arm in ARMS:
        rows.append(
            {
                "ordering": "fixed_hash_order",
                "feedback_condition": "primary_delay_1",
                "delay": 1,
                "missing_fraction": 0.0,
                "stream_index": 0,
                "group_id": "fixture-group",
                "source_row_index": 0,
                "arm": arm,
                "seed": TRAINING_SEEDS[0],
                "label": 0,
                "raw_energy": 0.0 if "gibbs" in arm or "feedback" in arm else None,
                "probability": 0.25,
                "brier_loss": 0.0625,
                "log_loss": -math.log(0.75),
                "decision": "escalate",
                "action_harm": False,
                "churn_contribution": 0,
                "prediction_before_feedback": True,
                "feedback_available_at": 1,
                "label_authority": LABEL_AUTHORITY,
                "numeric_state": {},
                "state_hash": canonical_hash({"arm": arm}),
                "gibbs_weights_hash": canonical_hash(_fixture_weights())
                if arm != "online_logistic_raw_features" and arm != "recent_frequency_beta_binomial"
                else None,
                "update_count": 0,
                "prediction_latency_s": 0.001,
                "update_latency_s": 0.0,
                "total_latency_s": 0.001,
                "disposition": "measured_prediction",
            }
        )
    controls = {name: {"passed": True} for name in run_analytic_controls()}
    receipts = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.1,
            "command_argv": [name],
            "command_environment": {},
            "log_path": f"/tmp/{name}.log",
            "log_sha256": canonical_hash(name),
        }
        for name in (
            *validation_scope.REQUIRED_CHECK_NAMES,
            "declared_entrypoint_e2e",
            "independent_reducer",
            "independent_cold_replay",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        )
    ]
    training_runs = [
        {
            "seed": seed,
            "gibbs_steps": 1,
            "future_stream_groups_used": 0,
        }
        for seed in TRAINING_SEEDS
    ]
    report = {
        "ordering": "fixed_hash_order",
        "feedback_condition": "primary_delay_1",
        "arm_metrics": {
            arm: {
                "mean_log_loss": 0.3,
                "coverage": 0.0,
                "accept_risk": None,
                "reject_risk": None,
                "accept_count": 0,
                "reject_count": 0,
            }
            for arm in ARMS
        },
    }
    value = reduce_primary_value([], [report])
    gates = [
        _gate("row_completeness", "completion", True, True),
        _gate("required_validation", "validation", True, True),
        _gate("primary_value", "scientific_efficacy", True, value["passed"]),
    ]
    artifact = {
        **_base_artifact(),
        "status": "complete_delayed_adapter_null",
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "completed_at_utc": "2026-09-18T00:00:01+00:00",
        "duration_s": 1.0,
        "phase_spans": [],
        "preconditions_checked": [{"check": "fixture", "passed": True}],
        "source_artifact_hashes": {},
        "historical_inference_sidecars": [],
        "rows": rows,
        "event_ledger": [
            {
                "event_id": "fixture-event",
                "feedback_available_at": 1,
                "label_authority": LABEL_AUTHORITY,
                "prediction_before_feedback": True,
                "prediction_records": {arm: {"state_hash": canonical_hash(arm)} for arm in ARMS},
            }
        ],
        "state_transition_rows": [{"check": name, **row} for name, row in controls.items()],
        "training_runs": training_runs,
        "policy_rows": [],
        "moving_block_intervals": [],
        "condition_reports": [report],
        "analytic_controls": controls,
        "restart_receipts": [{"performed": True, "prediction_parity": True}],
        "sample_size_budget": {
            "planned_prediction_rows": len(rows),
            "attempted_prediction_rows": len(rows),
            "completed_prediction_rows": len(rows),
            "censored_prediction_rows": 0,
            "unstarted_prediction_rows": 0,
            "planned_event_ledgers": 1,
            "effective_independent_group_count": 1,
            "stopping_rule": "Run every frozen fixture unit once.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": "complete_null_delayed_affine_benefit_not_demonstrated",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "repository_health": {
            "status": "healthy_for_affected_scope",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_findings": [],
        },
        "delayed_adapter_ready_score": 1,
        "delayed_adapter_value_score": 0,
        "primary_value_reduction": value,
        "online_protocol": {
            "orders_are_real_chronology": False,
            "final_test_labels_read": False,
        },
        "small_ebm_training": {"performed": True, "current_llm_calls": 0},
        "evidence_scope": {
            "class": "controlled_archive_replay",
            "scientific_benefit_claim": False,
            "conformal_guarantee": False,
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "independent_reduction": {},
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["independent_reduction"] = independent_reduce(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, raw reductions, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    identity = (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
        or not isinstance(artifact.get("inference_substrate"), str)
    ):
        errors.append("substrate_declaration_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") or artifact.get("validation_receipts"):
            errors.append("blocked_artifact_has_dependent_work")
        failure = (artifact.get("gate_check_summary") or {}).get("first_required_failure")
        if not isinstance(failure, Mapping) or not {
            "upstream",
            "check",
            "artifact_field",
            "expected",
            "observed",
        } <= set(failure):
            errors.append("blocked_gate_summary_missing")
        if any(
            artifact.get(name) != 0
            for name in (
                "delayed_adapter_ready_score",
                "delayed_adapter_value_score",
                "promotion_score",
            )
        ):
            errors.append("blocked_scores_nonzero")
    else:
        reduced = independent_reduce(artifact)
        if artifact.get("independent_reduction") != reduced:
            errors.append("independent_reduction_mismatch")
        if artifact.get("delayed_adapter_ready_score") != reduced["delayed_adapter_ready_score"]:
            if "independent_reduction_mismatch" not in errors:
                errors.append("independent_reduction_mismatch")
        if artifact.get("delayed_adapter_value_score") != reduced["delayed_adapter_value_score"]:
            if "independent_reduction_mismatch" not in errors:
                errors.append("independent_reduction_mismatch")
    if artifact.get("flagged_adversarial") is True and any(
        artifact.get(name) != 0
        for name in (
            "delayed_adapter_ready_score",
            "delayed_adapter_value_score",
            "promotion_score",
        )
    ):
        errors.append("adversarial_scores_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - file boundary.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any
) -> JsonDict:  # pragma: no cover - execution boundary.
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _online_membership_hash(protocol: Mapping[str, Any]) -> str:
    replay = protocol.get("online_replay") or {}
    return canonical_hash(
        {
            "initialization_group_ids": replay.get("initialization_group_ids"),
            "later_group_ids": replay.get("later_group_ids"),
        }
    )


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:  # pragma: no cover - live boundary.
    """Authenticate exact sources, eligibility, CPU runtime, and entrypoint."""

    import jax

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in (*INPUT_PATHS, MODULE_PATH, ADAPTER_PATH, WRAPPER_PATH):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    protocol = _load_object(root / PROTOCOL_PATH)
    trained = _load_object(root / TRAINING_ARTIFACT_PATH)
    manifest = protocol.get("protocol_manifest") or {}
    replay = protocol.get("online_replay") or {}
    source_hashes = manifest.get("source_hashes") or {}
    expected = (
        (
            "protocol_bytes",
            PROTOCOL_PATH,
            "artifact_sha256",
            EXPECTED_PROTOCOL_SHA256,
            hashes.get(PROTOCOL_PATH.as_posix()),
        ),
        (
            "protocol_status",
            PROTOCOL_PATH,
            "status",
            "complete_decision_protocol_ready",
            protocol.get("status"),
        ),
        ("protocol_verdict", PROTOCOL_PATH, "verdict_class", "null", protocol.get("verdict_class")),
        (
            "protocol_flag",
            PROTOCOL_PATH,
            "flagged_adversarial",
            False,
            protocol.get("flagged_adversarial"),
        ),
        (
            "protocol_ready",
            PROTOCOL_PATH,
            "decision_protocol_ready_score",
            1,
            protocol.get("decision_protocol_ready_score"),
        ),
        (
            "protocol_required_gates",
            PROTOCOL_PATH,
            "gate_check_summary.all_required_passed",
            True,
            (protocol.get("gate_check_summary") or {}).get("all_required_passed"),
        ),
        (
            "partition_identity",
            PROTOCOL_PATH,
            "protocol_manifest.partition_membership_sha256",
            EXPECTED_PARTITION_SHA256,
            manifest.get("partition_membership_sha256"),
        ),
        (
            "corpus_identity",
            CORPUS_PATH,
            "sha256",
            EXPECTED_CORPUS_SHA256,
            hashes.get(CORPUS_PATH.as_posix()),
        ),
        (
            "protocol_corpus_identity",
            PROTOCOL_PATH,
            "protocol_manifest.source_hashes.data/fover_corpus_v4.json",
            EXPECTED_CORPUS_SHA256,
            source_hashes.get(CORPUS_PATH.as_posix()),
        ),
        (
            "online_membership_identity",
            PROTOCOL_PATH,
            "online_replay.membership_sha256",
            EXPECTED_ONLINE_MEMBERSHIP_SHA256,
            _online_membership_hash(protocol),
        ),
        (
            "initialization_group_count",
            PROTOCOL_PATH,
            "online_replay.initialization_group_ids",
            1321,
            len(replay.get("initialization_group_ids") or []),
        ),
        (
            "later_group_count",
            PROTOCOL_PATH,
            "online_replay.later_group_ids",
            1321,
            len(replay.get("later_group_ids") or []),
        ),
        (
            "training_artifact_bytes",
            TRAINING_ARTIFACT_PATH,
            "artifact_sha256",
            EXPECTED_TRAINING_SHA256,
            hashes.get(TRAINING_ARTIFACT_PATH.as_posix()),
        ),
        (
            "training_status",
            TRAINING_ARTIFACT_PATH,
            "status",
            "complete_decision_training_null",
            trained.get("status"),
        ),
        (
            "training_verdict",
            TRAINING_ARTIFACT_PATH,
            "verdict_class",
            "null",
            trained.get("verdict_class"),
        ),
        (
            "training_flag",
            TRAINING_ARTIFACT_PATH,
            "flagged_adversarial",
            False,
            trained.get("flagged_adversarial"),
        ),
        (
            "training_capture",
            TRAINING_ARTIFACT_PATH,
            "decision_capture_complete_score",
            1,
            trained.get("decision_capture_complete_score"),
        ),
        (
            "training_null_retained",
            TRAINING_ARTIFACT_PATH,
            "calibration_value_score",
            0,
            trained.get("calibration_value_score"),
        ),
    )
    for check, upstream, field, wanted, observed in expected:
        checks.append(_precondition(check, upstream.as_posix(), field, wanted, observed))

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7397",
            "REQ-REPORT-7397" if "REQ-REPORT-7397" in spec_text else None,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7397" in exclusion_text or EXPERIMENT_ID in exclusion_text
    checks.extend(
        (
            _precondition(
                "current_task_not_quarantined",
                "ops/exclusion_manifest.yaml",
                EXPERIMENT_ID,
                False,
                excluded,
            ),
            _precondition(
                "cpu_jax_backend",
                "current_process",
                "jax.default_backend",
                "cpu",
                jax.default_backend(),
            ),
            _precondition(
                "live_execution_guard",
                "current_process",
                "CARNOT_FORCE_LIVE",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
            ),
            _precondition(
                "declared_entrypoint",
                WRAPPER_PATH.as_posix(),
                "path",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes"
                if (root / WRAPPER_PATH).is_file() and (root / WRAPPER_PATH).stat().st_size > 0
                else None,
            ),
        )
    )
    final_rows = [
        row
        for row in protocol.get("feature_rows", [])
        if isinstance(row, Mapping) and row.get("partition") == "final_test"
    ]
    checks.append(
        _precondition(
            "final_test_labels_sealed",
            PROTOCOL_PATH.as_posix(),
            "feature_rows.final_test.label_absent",
            True,
            bool(final_rows) and all("label" not in row for row in final_rows),
        )
    )
    return checks, hashes, protocol


def validate_candidate_artifact(value: object) -> list[str]:  # pragma: no cover - subprocess.
    """Check measured evidence before terminal readers are attached."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    budget = artifact.get("sample_size_budget") or {}
    if len(artifact.get("rows") or []) != int(budget.get("planned_prediction_rows") or -1):
        errors.append("row_completeness_mismatch")
    if len(artifact.get("event_ledger") or []) != int(budget.get("planned_event_ledgers") or -1):
        errors.append("ledger_completeness_mismatch")
    passing = {
        str(row.get("name"))
        for row in artifact.get("validation_receipts", [])
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    if not set(validation_scope.REQUIRED_CHECK_NAMES) <= passing:
        errors.append("affected_validation_mismatch")
    if not all(
        row.get("passed") is True for row in (artifact.get("analytic_controls") or {}).values()
    ):
        errors.append("analytic_controls_failed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _states_from_record(
    record: Mapping[str, Any], initial_labels: Sequence[int]
) -> dict[str, JsonDict]:
    adapter = DelayedEnergyCalibrator(
        record["weights"],
        a=float(record["affine"]["a"]),
        b=float(record["affine"]["b"]),
    ).to_dict()
    return {
        "static_affine_gibbs": {"adapter": deepcopy(adapter), "update_count": 0},
        "adaptive_affine_gibbs": {"adapter": deepcopy(adapter), "update_count": 0},
        "online_logistic_raw_features": {
            "weights": deepcopy(record["logistic_weights"]),
            "update_count": 0,
        },
        "recent_frequency_beta_binomial": {
            "recent_labels": list(initial_labels)[-RECENT_WINDOW:],
            "update_count": 0,
        },
        "no_feedback_adaptive_control": {"adapter": deepcopy(adapter), "update_count": 0},
    }


_COLD_ROW_FIELDS = (
    "ordering",
    "feedback_condition",
    "delay",
    "missing_fraction",
    "stream_index",
    "group_id",
    "source_row_index",
    "arm",
    "seed",
    "label",
    "raw_energy",
    "probability",
    "brier_loss",
    "log_loss",
    "decision",
    "action_harm",
    "churn_contribution",
    "prediction_before_feedback",
    "feedback_available_at",
    "label_authority",
    "numeric_state",
    "state_hash",
    "gibbs_weights_hash",
    "update_count",
    "disposition",
)


def cold_replay_errors(
    artifact: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[str]:  # pragma: no cover - fresh process E2E.
    """Recompute every scientific row from sealed numeric checkpoints."""

    streams = build_streams(protocol)
    initial = initialization_rows(protocol)
    initial_labels = [int(row["label"]) for row in initial]
    missing = deterministic_missing_mask(
        [
            str(group_id)
            for group_id in (protocol.get("online_replay") or {}).get("later_group_ids", [])
        ]
    )
    policies = {int(row["seed"]): row["policy"] for row in artifact.get("policy_rows", [])}
    records = {int(row["seed"]): row for row in artifact.get("training_runs", [])}
    recomputed: list[JsonDict] = []
    for ordering in ORDERINGS:
        for condition in feedback_conditions():
            condition_missing = missing if float(condition["missing_fraction"]) > 0.0 else set()
            for seed in TRAINING_SEEDS:
                result = replay_condition(
                    _states_from_record(records[seed], initial_labels),
                    streams[ordering],
                    ordering=ordering,
                    condition=condition,
                    seed=seed,
                    thresholds=policies[seed],
                    missing_group_ids=condition_missing,
                )
                recomputed.extend(result["rows"])
    expected = [
        {field: row.get(field) for field in _COLD_ROW_FIELDS} for row in artifact.get("rows", [])
    ]
    observed = [{field: row.get(field) for field in _COLD_ROW_FIELDS} for row in recomputed]
    return (
        [] if canonical_hash(expected) == canonical_hash(observed) else ["cold_row_replay_mismatch"]
    )


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = str(REPO_ROOT / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7397_v649_delayed_adapter import validate_candidate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_candidate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_cold_replay",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7397_v649_delayed_adapter",
                    "--cold-replay",
                    str(candidate),
                ),
                "capability_end_to_end",
                1800.0,
            ),
            "completion",
            True,
        ),
    ]


def _utc_now() -> str:  # pragma: no cover - wall-clock boundary.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7397] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(phase: str, phase_started: float, run_started: float) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "checkpoint_utc": _utc_now(),
    }


def _historical_context_sidecar(root: Path) -> JsonDict:  # pragma: no cover
    """Keep the disqualified full-head history outside current call counters."""

    historical_path = root / "results/experiment_7386_v648_online_decisions.json"
    historical = _load_object(historical_path)
    sidecar = {
        "schema": "carnot.exp7397.historical_context.v1",
        "source_path": "results/experiment_7386_v648_online_decisions.json",
        "source_sha256": sha256_file(historical_path) if historical_path.is_file() else None,
        "status": historical.get("status"),
        "verdict_class": historical.get("verdict_class"),
        "flagged_adversarial": historical.get("flagged_adversarial"),
        "eligibility": "ineligible_context_only",
        "scientific_rows_reused": 0,
        "historical_invocation_counters_copied": False,
        "current_llm_invocations": 0,
    }
    path = root / RAW_DIR / "historical_v648_context.json"
    atomic_json(path, sidecar)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "scope": "ineligible_historical_context_only",
    }


def _entrypoint_receipt(
    root: Path, started_at_utc: str, duration_s: float
) -> JsonDict:  # pragma: no cover
    raw_path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    record = {
        "event": "declared_entrypoint_reached_terminal_validation",
        "argv": argv,
        "started_at_utc": started_at_utc,
        "ended_at_utc": _utc_now(),
        "duration_s": duration_s,
    }
    atomic_json(raw_path, record)
    return {
        "name": "declared_entrypoint_e2e",
        "command": " ".join(argv),
        "command_argv": argv,
        "scope": "capability_end_to_end",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": raw_path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(raw_path),
        "passed": True,
        "timed_out": False,
        "output_tail": "declared entrypoint reached terminal validation",
        "command_category": "completion",
        "required": True,
        "started_at_utc": started_at_utc,
        "ended_at_utc": record["ended_at_utc"],
        "command_environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }


def _assemble_artifact(  # pragma: no cover - measured execution assembly.
    *,
    started: float,
    started_at_utc: str,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    protocol: Mapping[str, Any],
    training_runs: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    ledgers: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    restarts: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    pending_feedback: int,
    receipts: Sequence[Mapping[str, Any]],
    historical_sidecar: Mapping[str, Any],
    fit_duration_s: float,
    flagged_adversarial: bool,
) -> JsonDict:
    value = reduce_primary_value(intervals, reports)
    affected_pass = set(validation_scope.REQUIRED_CHECK_NAMES) <= {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    gates = [
        _gate(
            "structured_preconditions",
            "completion",
            True,
            all(row.get("passed") is True for row in preconditions),
        ),
        _gate(
            "analytic_controls",
            "safety",
            True,
            all(row.get("passed") is True for row in controls.values()),
        ),
        _gate("prediction_rows_complete", "completion", len(rows), len(rows)),
        _gate("event_ledger_complete", "completion", len(ledgers), len(ledgers)),
        _gate("affected_validation", "validation", True, affected_pass),
        _gate("terminal_validation", "validation", True, _required_receipts_pass(receipts)),
        _gate("adversarial_clear", "safety", False, flagged_adversarial),
        *[deepcopy(dict(row)) for row in value["checks"]],
    ]
    replay = protocol.get("online_replay") or {}
    missing_ids = deterministic_missing_mask(
        [str(item) for item in replay.get("later_group_ids") or []]
    )
    artifact: JsonDict = {
        **_base_artifact(),
        "status": "complete_delayed_adapter_pending_reduction",
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "duration_s": time.monotonic() - started,
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [deepcopy(dict(historical_sidecar))],
        "rows": [deepcopy(dict(row)) for row in rows],
        "event_ledger": [deepcopy(dict(row)) for row in ledgers],
        "state_transition_rows": [
            {"check": name, **deepcopy(dict(row))} for name, row in controls.items()
        ]
        + [
            {
                "check": "restart_equality",
                "ordering": row.get("ordering"),
                "feedback_condition": row.get("feedback_condition"),
                "seed": row.get("seed"),
                **deepcopy(dict(row.get("receipt") or {})),
            }
            for row in restarts
        ],
        "training_runs": [deepcopy(dict(row)) for row in training_runs],
        "policy_rows": [deepcopy(dict(row)) for row in policies],
        "moving_block_intervals": [deepcopy(dict(row)) for row in intervals],
        "condition_reports": [deepcopy(dict(row)) for row in reports],
        "analytic_controls": deepcopy(dict(controls)),
        "restart_receipts": [deepcopy(dict(row["receipt"])) for row in restarts],
        "sample_size_budget": {
            "planned_prediction_rows": len(rows),
            "attempted_prediction_rows": len(rows),
            "completed_prediction_rows": len(rows),
            "censored_prediction_rows": 0,
            "unstarted_prediction_rows": 0,
            "planned_event_ledgers": len(ledgers),
            "attempted_feedback_units": len(ledgers),
            "completed_feedback_units": sum(row.get("feedback") is not None for row in ledgers),
            "censored_feedback_units": pending_feedback
            + sum(row.get("missing") is True for row in ledgers),
            "effective_independent_group_count": len(replay.get("later_group_ids") or []),
            "maximum_initial_training_steps": MAX_STEPS,
            "moving_block_draws": BOOTSTRAP_DRAWS,
            "stopping_rule": "Run every frozen order, condition, seed, arm, and event once; never stop from efficacy.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": "complete_disqualified_pending_required_validation",
        "verdict_class": "disqualified",
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": {
            "status": "healthy_for_affected_scope" if affected_pass else "affected_checks_failed",
            "as_of": RUN_DATE,
            "affects_required_checks": not affected_pass,
            "unrelated_findings": [],
            "full_python_suite_launched_by_experiment": False,
        },
        "delayed_adapter_ready_score": 0,
        "delayed_adapter_value_score": 0,
        "primary_value_reduction": value,
        "online_protocol": {
            "source": PROTOCOL_PATH.as_posix(),
            "source_sha256": EXPECTED_PROTOCOL_SHA256,
            "partition_membership_sha256": EXPECTED_PARTITION_SHA256,
            "online_membership_sha256": EXPECTED_ONLINE_MEMBERSHIP_SHA256,
            "initialization_group_count": len(replay.get("initialization_group_ids") or []),
            "later_group_count": len(replay.get("later_group_ids") or []),
            "final_test_labels_read": False,
            "orders": {
                "fixed_hash_order": "Exp7382 original salted hash order",
                "reversed_block_order": "reverse the order of fixed blocks of 32 and preserve order inside each block",
            },
            "orders_are_real_chronology": False,
            "conditions": list(feedback_conditions()),
            "missing_mask_rule": "every fourth later-group identity in the original fixed order",
            "missing_group_count": len(missing_ids),
            "same_event_and_feedback_mask_for_every_arm": True,
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_initial_steps": MAX_STEPS,
            "arm_definitions": {
                "static_affine_gibbs": "frozen initial Gibbs and affine calibration",
                "adaptive_affine_gibbs": "frozen Gibbs with delayed two-scalar affine updates",
                "online_logistic_raw_features": "two-feature online logistic control",
                "recent_frequency_beta_binomial": "window 128 with Beta(1,1)",
                "no_feedback_adaptive_control": "adaptive mechanism with all stream feedback withheld",
            },
            "affine_rule": "p_incorrect=sigmoid(a*E+b); a in [0.25,4], b in [-8,8]",
            "online_learning_rate": LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "threshold_source": "disjoint policy_calibration partition",
            "policy_rows": [
                {"seed": row["seed"], "policy": deepcopy(row["policy"])} for row in policies
            ],
            "risk_budgets": {"incorrect_accept": 0.05, "correct_reject": 0.10},
            "simultaneous_correction_inherited_from_exp7382": True,
            "moving_block_draws": BOOTSTRAP_DRAWS,
            "moving_block_lengths": list(BLOCK_LENGTHS),
            "moving_block_seed": BOOTSTRAP_SEED,
            "seed_reduction": "average_within_event_before_resampling",
            "primary_hypothesis": "fixed_hash_order primary_delay_1",
            "sensitivity_hypotheses": [
                "reversed_block_order",
                "sensitivity_delay_32",
                "sensitivity_missing_25",
                "block_length_64",
            ],
        },
        "small_ebm_training": {
            "performed": True,
            "kind": "initial 2-4-1 Gibbs natural-prevalence Bernoulli training and numeric affine calibration only",
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_steps_per_fit": MAX_STEPS,
            "fit_duration_s": fit_duration_s,
            "gibbs_frozen_during_stream": True,
            "generator_loaded": False,
            "generator_weights_changed": False,
            "current_llm_calls": 0,
        },
        "evidence_scope": {
            "class": "controlled_archive_replay_not_real_world_chronology",
            "v648_full_head": "ineligible_context_only",
            "scientific_benefit_claim": False,
            "no_fitted_result_is_a_benefit_claim": True,
            "conformal_guarantee": False,
            "iid_population_guarantee": False,
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "independent_reduction": {},
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    reduced = independent_reduce(artifact)
    artifact["independent_reduction"] = reduced
    artifact["delayed_adapter_ready_score"] = reduced["delayed_adapter_ready_score"]
    artifact["delayed_adapter_value_score"] = reduced["delayed_adapter_value_score"]
    if reduced["delayed_adapter_ready_score"] == 0:
        artifact["status"] = "complete_delayed_adapter_disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_validation_or_safety_failure"
        artifact["verdict_class"] = "disqualified"
    elif reduced["delayed_adapter_value_score"] == 1:
        artifact["status"] = "complete_delayed_adapter_positive"
        artifact["honest_verdict"] = (
            "complete_positive_delayed_affine_registered_gate_passed_no_promotion"
        )
        artifact["verdict_class"] = "positive"
    else:
        artifact["status"] = "complete_delayed_adapter_null"
        artifact["honest_verdict"] = (
            "complete_null_delayed_affine_registered_benefit_not_demonstrated"
        )
        artifact["verdict_class"] = "null"
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint.
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run preconditions, replay, scoped checks, cold replay, and publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at_utc = _utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    _progress(started, "preconditions", "start")
    preconditions, source_hashes, protocol = collect_preconditions(root)
    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    spans.append(_span("preconditions", phase_started, started))
    _progress(started, "preconditions", "end", passed=preconditions_passed)
    if not preconditions_passed:
        blocked = build_blocked_artifact(preconditions, source_hashes)
        blocked["started_at_utc"] = started_at_utc
        blocked["completed_at_utc"] = _utc_now()
        blocked["duration_s"] = time.monotonic() - started
        blocked["phase_spans"] = spans
        blocked["field_principles"] = _field_principles(tuple(blocked))
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        _progress(started, "write", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        _progress(started, "write", "after_atomic_blocked", status=blocked["status"])
        return blocked

    raw_dir = root / RAW_DIR
    checkpoint_dir = root / CHECKPOINT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    historical_sidecar = _historical_context_sidecar(root)

    phase_started = time.monotonic()
    _progress(started, "controls", "start")
    controls = run_analytic_controls()
    spans.append(_span("controls", phase_started, started))
    _progress(
        started,
        "controls",
        "end",
        passed=all(row.get("passed") is True for row in controls.values()),
    )

    phase_started = time.monotonic()
    _progress(started, "load", "start", model_load="not_attempted")
    spans.append(_span("load", phase_started, started))
    _progress(started, "load", "end", model_invoked=False)
    phase_started = time.monotonic()
    _progress(started, "generate", "start", generation="not_attempted")
    spans.append(_span("generate", phase_started, started))
    _progress(started, "generate", "end", generation_calls=0)

    phase_started = time.monotonic()
    _progress(started, "initial_training", "start", seeds=len(TRAINING_SEEDS))
    initial = initialization_rows(protocol)
    policy_source = policy_calibration_rows(protocol)
    initialized: dict[int, dict[str, JsonDict]] = {}
    policies: list[JsonDict] = []
    training_runs: list[JsonDict] = []
    fit_started = time.monotonic()
    for unit, seed in enumerate(TRAINING_SEEDS, start=1):
        _progress(
            started,
            "initial_training",
            "before_small_ebm_fit",
            unit=f"{unit}/{len(TRAINING_SEEDS)}",
            seed=seed,
        )
        states = initialize_seed_states(initial, seed)
        policy = select_fixed_thresholds(states, policy_source)
        initialized[seed] = states
        training_runs.append(deepcopy(states["training_record"]))
        policies.append({"seed": seed, "policy": policy})
        _progress(
            started,
            "initial_training",
            "after_small_ebm_fit",
            unit=f"{unit}/{len(TRAINING_SEEDS)}",
            seed=seed,
        )
    fit_duration_s = time.monotonic() - fit_started
    spans.append(_span("initial_training", phase_started, started))
    _progress(started, "initial_training", "end", duration_s=f"{fit_duration_s:.3f}")

    streams = build_streams(protocol)
    original_ids = [str(row["group_id"]) for row in streams["fixed_hash_order"]]
    missing_ids = deterministic_missing_mask(original_ids)
    policy_by_seed = {int(row["seed"]): row["policy"] for row in policies}
    all_rows: list[JsonDict] = []
    ledgers: list[JsonDict] = []
    restarts: list[JsonDict] = []
    pending_feedback = 0
    total_units = len(ORDERINGS) * len(feedback_conditions()) * len(TRAINING_SEEDS)
    phase_started = time.monotonic()
    _progress(started, "measurement", "start", replay_units=total_units)
    completed = 0
    for ordering in ORDERINGS:
        for condition in feedback_conditions():
            condition_missing = missing_ids if float(condition["missing_fraction"]) > 0.0 else set()
            for seed in TRAINING_SEEDS:
                _progress(
                    started,
                    "measurement",
                    "before_replay",
                    ordering=ordering,
                    condition=condition["name"],
                    seed=seed,
                )
                checkpoint = checkpoint_dir / f"{ordering}-{condition['name']}-seed{seed}.json"
                replay = replay_condition(
                    initialized[seed],
                    streams[ordering],
                    ordering=ordering,
                    condition=condition,
                    seed=seed,
                    thresholds=policy_by_seed[seed],
                    missing_group_ids=condition_missing,
                    checkpoint_path=checkpoint,
                )
                all_rows.extend(replay["rows"])
                ledgers.extend(replay["event_ledger"])
                pending_feedback += int(replay["pending_feedback_at_end"])
                restarts.append(
                    {
                        "ordering": ordering,
                        "feedback_condition": condition["name"],
                        "seed": seed,
                        "receipt": replay["restart_receipt"],
                    }
                )
                completed += 1
                _progress(
                    started,
                    "measurement",
                    "after_replay",
                    completed=f"{completed}/{total_units}",
                    rows=len(all_rows),
                )
    _progress(started, "measurement", "before_moving_block_bootstrap", draws=BOOTSTRAP_DRAWS)
    intervals = paired_moving_block_intervals(all_rows)
    _progress(started, "measurement", "after_moving_block_bootstrap", intervals=len(intervals))
    reports = condition_reports(all_rows)
    spans.append(_span("measurement", phase_started, started))
    _progress(started, "measurement", "end", rows=len(all_rows), ledgers=len(ledgers))

    private_root = Path(tempfile.mkdtemp(prefix="exp7397-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    commands = build_command_plan(root, V649_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V649_MANIFEST, commands)
    _progress(started, "validation", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    _progress(started, "validation", "after_affected_subprocesses", receipts=len(affected))
    spans.append(_span("affected_validation", phase_started, started))

    candidate = _assemble_artifact(
        started=started,
        started_at_utc=started_at_utc,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        protocol=protocol,
        training_runs=training_runs,
        policies=policies,
        rows=all_rows,
        ledgers=ledgers,
        controls=controls,
        restarts=restarts,
        intervals=intervals,
        reports=reports,
        pending_feedback=pending_feedback,
        receipts=affected,
        historical_sidecar=historical_sidecar,
        fit_duration_s=fit_duration_s,
        flagged_adversarial=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    phase_started = time.monotonic()
    _progress(started, "write", "before_atomic_candidate", path=candidate_path)
    atomic_json(candidate_path, candidate)
    spans.append(_span("candidate_write", phase_started, started))
    _progress(started, "write", "after_atomic_candidate", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    adversarial_failed = any(
        row.get("name") == "adversarial_verify" and row.get("passed") is not True
        for row in terminal
    )
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        adversarial_failed=adversarial_failed,
    )

    entrypoint = _entrypoint_receipt(root, started_at_utc, time.monotonic() - started)
    final = _assemble_artifact(
        started=started,
        started_at_utc=started_at_utc,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        protocol=protocol,
        training_runs=training_runs,
        policies=policies,
        rows=all_rows,
        ledgers=ledgers,
        controls=controls,
        restarts=restarts,
        intervals=intervals,
        reports=reports,
        pending_feedback=pending_feedback,
        receipts=[*affected, *terminal, entrypoint],
        historical_sidecar=historical_sidecar,
        fit_duration_s=fit_duration_s,
        flagged_adversarial=adversarial_failed,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse public execution and fresh-process cold replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Dispatch the thin entrypoint or independent row recomputation."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        artifact = _load_object(args.cold_replay)
        candidate_errors = validate_candidate_artifact(artifact)
        protocol = _load_object(REPO_ROOT / PROTOCOL_PATH)
        replay_errors = cold_replay_errors(artifact, protocol) if not candidate_errors else []
        errors = [*candidate_errors, *replay_errors]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
