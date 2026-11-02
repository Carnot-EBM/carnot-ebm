"""Measure delayed proper-loss learning on frozen cached forecasts.

The learner sees labels only through delayed release blocks. Cached forecasts
carry historical model identity, but this module loads and calls no model.

Spec refs: REQ-CL-7578 and SCENARIO-CL-7578-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.experiment_7534_v659_count_memory import CountConfig, bin_index
from carnot.experiment_7561_v661_recalibration_prototype import (
    KNOTS,
    RecalibrationEventMachine,
    typed_decision,
)
from carnot.experiment_7575_v662_cached_learning_protocol import (
    HISTORICAL_MODEL_ID,
    ORDER_SEEDS,
    ROLE_COUNTS,
    freeze_protocol,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7578-v662-continuous-proper-loss"
SCHEMA = "carnot.exp7578.v662.continuous_proper_loss.v1"
RESULT_PATH = Path("results/experiment_7578_v662_continuous_proper_loss.json")
RAW_DIR = Path("results/raw/experiment_7578_v662_continuous_proper_loss")
MODULE_PATH = Path("python/carnot/experiment_7578_v662_continuous_proper_loss.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7578_v662_continuous_proper_loss.py")
TEST_PATH = Path("tests/python/test_experiment_7578_v662_continuous_proper_loss.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP7574_PATH = Path("results/experiment_7574_v662_measurement_requalification.json")
EXP7575_PATH = Path("results/experiment_7575_v662_cached_learning_protocol.json")
ARMS = ("bounded", "raw", "global_count", "local_count", "shuffled_feedback")
IDENTITY_THETA = KNOTS.tolist()
RETENTION_CHECKPOINTS = (40, 80, 120, 160)
RESTART_POINTS = (80, 120)
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "learning_measurement_complete_score",
    "exploratory_learning_benefit_score",
    "retention_pass_score",
    "causal_event_rows_path",
    "fresh_confirmatory_claim_allowed",
    "state_hashes",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _resolved(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def identity_state_hash() -> str:
    """Name the numerical identity state used at every replay start."""

    return canonical_hash({"theta": IDENTITY_THETA, "sample_count": 0})


def count_config_from_rows(rows: Sequence[Mapping[str, Any]]) -> CountConfig:
    """Derive immutable label-free count priors from fitting probabilities."""

    values = [min(0.9999, max(0.0001, float(row["probability"]))) for row in rows]
    if not values:
        raise ValueError("fit_probabilities_empty")
    buckets: list[list[float]] = [[] for _ in range(8)]
    for value in values:
        buckets[bin_index(value)].append(value)
    means = tuple(
        float(np.mean(bucket)) if bucket else (index + 0.5) / 8
        for index, bucket in enumerate(buckets)
    )
    return CountConfig(means, float(np.mean(values)))


def _brier(probability: float, label: int) -> float:
    value = float(probability)
    if label not in (0, 1) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability_or_label_invalid")
    return (value - label) ** 2


def _action_cost(probability: float, label: int) -> tuple[str, float, bool]:
    decision = typed_decision(probability)
    action = str(decision["action"])
    cost = {
        "accept": 5.0 if label else 0.0,
        "reject": 0.0 if label else 1.0,
        "escalate": 0.2,
    }[action]
    return action, cost, action != "escalate"


def metric_row(
    unit_id: str,
    arm: str,
    probability: float,
    label: int,
    *,
    seed: int,
    phase: str,
    censored: bool = False,
) -> JsonDict:
    """Store the raw operands needed to reproduce proper loss and action cost."""

    loss = _brier(probability, label)
    action, cost, covered = _action_cost(probability, label)
    return {
        "unit_id": str(unit_id),
        "arm": str(arm),
        "phase": str(phase),
        "q": float(probability),
        "label": int(label),
        "typed_action": action,
        "raw_squared_error_numerator": loss,
        "raw_squared_error_denominator": 1,
        "brier": loss,
        "realized_action_cost": cost,
        "non_escalated": covered,
        "metric_direction": "lower_is_better",
        "seed": int(seed),
        "censored": bool(censored),
        "provenance": "exp7575_cached_original_source_forecast",
    }


def _numerical_state(machine: RecalibrationEventMachine) -> JsonDict:
    """Exclude journal growth so movement reflects only learned statistics."""

    return {
        "bounded": machine.constrained.to_payload(),
        "shuffled_feedback": machine.shuffled.to_payload(),
        "global_count": machine.count_arms["global_count"].to_payload(),
        "local_count": machine.count_arms["local_count"].to_payload(),
    }


def _numerical_hash(machine: RecalibrationEventMachine) -> str:
    return canonical_hash(_numerical_state(machine))


def _arm_probabilities(machine: RecalibrationEventMachine, probability: float) -> dict[str, float]:
    return {
        "bounded": machine.constrained.predict(probability),
        "raw": float(probability),
        "global_count": machine.count_arms["global_count"].predict(probability).probability,
        "local_count": machine.count_arms["local_count"].predict(probability).probability,
        "shuffled_feedback": machine.shuffled.predict(probability),
    }


def evaluate_retention(
    machine: RecalibrationEventMachine,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    event_count: int,
) -> JsonDict:
    """Score a read-only evaluator set and return no labels to the learner."""

    before = machine.state_hash()
    numerical_before = _numerical_hash(machine)
    scored = [
        {
            **metric_row(
                str(source["source_id"]),
                arm,
                probability,
                int(source["label"]),
                seed=seed,
                phase="retention",
            ),
            "event_count": event_count,
        }
        for source in rows
        for arm, probability in _arm_probabilities(machine, float(source["probability"])).items()
    ]
    after = machine.state_hash()
    return {
        "event_count": int(event_count),
        "rows": scored,
        "learner_labels_returned": 0,
        "learner_state_hash_before": before,
        "learner_state_hash_after": after,
        "numerical_state_hash_before": numerical_before,
        "numerical_state_hash_after": _numerical_hash(machine),
        "passed": before == after,
    }


def _release_properties(feedback: Sequence[tuple[str, int]]) -> JsonDict:
    labels = [int(label) for _event_id, label in feedback]
    rotated = labels[1:] + labels[:1]
    shufflable = len(set(labels)) > 1
    return {
        "shufflable": shufflable,
        "shuffled_assignment_changed": shufflable and rotated != labels,
        "marginal_labels_preserved": sorted(rotated) == sorted(labels),
        "release_time_preserved": True,
        "original_labels": labels,
        "shuffled_labels": rotated,
    }


def _sealed_arms(prediction: Mapping[str, Any]) -> JsonDict:
    """Rename qualified numerical arms to the measurement contract names."""

    source = prediction["arms"]
    names = {
        "bounded": "constrained",
        "raw": "raw",
        "global_count": "global_count",
        "local_count": "local_count",
        "shuffled_feedback": "shuffled_constrained",
    }
    return {
        arm: {
            "q": float(source[name]["probability"]),
            "typed_action": str(source[name]["decision"]["action"]),
        }
        for arm, name in names.items()
    }


def _append_durable(stream: Any, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _release_block(
    working: RecalibrationEventMachine,
    uninterrupted: RecalibrationEventMachine,
    feedback: Sequence[tuple[str, int]],
    *,
    release_index: int,
    event_count: int,
    checkpoint: Path,
    crash_before_ack: bool,
) -> tuple[JsonDict, JsonDict, RecalibrationEventMachine]:
    """Apply, persist, recover, and acknowledge one legal delayed block."""

    before_theta = working.constrained.theta.copy()
    before_hash = _numerical_hash(working)
    update_started = time.monotonic()
    receipt = working.release(release_index, feedback)
    update_latency = time.monotonic() - update_started
    twin_receipt = uninterrupted.release(release_index, feedback)
    if receipt != twin_receipt:  # pragma: no cover - identical deterministic twins
        raise ValueError("uninterrupted_release_mismatch")
    properties = _release_properties(feedback)
    if (  # pragma: no cover - nonconstant one-step rotation must differ
        properties["shufflable"] and not properties["shuffled_assignment_changed"]
    ):
        raise ValueError("informative_shuffle_did_not_change_assignment")
    if (
        not properties[  # pragma: no cover - rotation preserves the multiset
            "marginal_labels_preserved"
        ]
    ):
        raise ValueError("shuffled_label_marginal_changed")

    working.save(checkpoint)
    persisted_after_release = working.state_hash()
    if crash_before_ack:
        working = RecalibrationEventMachine.load(checkpoint)
    ack_started = time.monotonic()
    working.acknowledge(release_index)
    working.save(checkpoint)
    durable_ack_latency = time.monotonic() - ack_started
    uninterrupted.acknowledge(release_index)
    after_hash = _numerical_hash(working)
    release_row = {
        "operation": "release_update_persist",
        "release_index": release_index,
        "event_count": event_count,
        "event_ids": [event_id for event_id, _label in feedback],
        "feedback": [[event_id, int(label)] for event_id, label in feedback],
        "update_count": len(feedback),
        "state_hash_before": before_hash,
        "state_hash_after": after_hash,
        "persisted_state_hash_before_ack": persisted_after_release,
        "state_movement_l2": float(np.linalg.norm(working.constrained.theta - before_theta)),
        "update_latency_s": update_latency,
        "crash_before_ack_exercised": crash_before_ack,
        **properties,
    }
    ack_row = {
        "operation": "durable_acknowledgment",
        "release_index": release_index,
        "event_count": event_count,
        "ack_count": 1,
        "durable_ack_latency_s": durable_ack_latency,
        "state_hash": working.state_hash(),
    }
    return release_row, ack_row, working


def _release_schedule(event_count: int) -> int | None:
    """Return the block released after eight later events have arrived."""

    if event_count < 16 or event_count % 8:
        return None
    return event_count // 8 - 2


def measure_order(
    online_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    order: Sequence[str],
    count_config: CountConfig,
    *,
    seed: int,
    state_dir: Path,
) -> JsonDict:
    """Replay one order with durable state and an uninterrupted parity twin."""

    by_id = {str(row["source_id"]): row for row in online_rows}
    if len(by_id) != len(online_rows) or set(order) != set(by_id) or len(order) != 160:
        raise ValueError("online_order_roster_invalid")
    if len(retention_rows) != 80:
        raise ValueError("retention_roster_invalid")
    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = state_dir / f"state-{seed}.json"
    event_log = state_dir / f"events-{seed}.jsonl"
    working = RecalibrationEventMachine.create(count_config)
    uninterrupted = RecalibrationEventMachine.create(count_config)
    initial_hash = _numerical_hash(working)
    event_rows: list[JsonDict] = []
    comparison_rows: list[JsonDict] = []
    release_rows: list[JsonDict] = []
    ack_rows: list[JsonDict] = []
    retention_scored: list[JsonDict] = []
    retention_receipts: list[JsonDict] = []
    restart_mismatches = 0
    prediction_mismatches = 0
    retention_mutations = 0

    with event_log.open("w", encoding="utf-8") as durable:
        for event_index, source_id in enumerate(order):
            source = by_id[source_id]
            event_count = event_index + 1
            release_index = event_index // 8
            state_hash = _numerical_hash(working)
            prediction = working.predict(source_id, float(source["probability"]), release_index)
            twin_prediction = uninterrupted.predict(
                source_id, float(source["probability"]), release_index
            )
            if prediction != twin_prediction:  # pragma: no cover - identical deterministic twins
                prediction_mismatches += 1
            sealed = _sealed_arms(prediction)
            event_row = {
                "operation": "sealed_prediction",
                "event_id": source_id,
                "event_count": event_count,
                "release_index": release_index,
                "state_hash_before_prediction": state_hash,
                "original_source_q": float(source["probability"]),
                "arms": sealed,
                "label_available_at_prediction": False,
                "seed": seed,
            }
            _append_durable(durable, event_row)
            event_rows.append(event_row)
            comparison_rows.extend(
                {
                    **metric_row(
                        source_id,
                        arm,
                        values["q"],
                        int(source["label"]),
                        seed=seed,
                        phase="online",
                        censored=event_count > 152,
                    ),
                    "event_count": event_count,
                }
                for arm, values in sealed.items()
            )

            due = _release_schedule(event_count)
            if due is not None:
                block_ids = list(order[due * 8 : due * 8 + 8])
                feedback = [(event_id, int(by_id[event_id]["label"])) for event_id in block_ids]
                release_row, ack_row, working = _release_block(
                    working,
                    uninterrupted,
                    feedback,
                    release_index=due,
                    event_count=event_count,
                    checkpoint=checkpoint,
                    crash_before_ack=due == 2,
                )
                release_rows.append(release_row)
                ack_rows.append(ack_row)

            if event_count in RETENTION_CHECKPOINTS:
                evaluated = evaluate_retention(
                    working, retention_rows, seed=seed, event_count=event_count
                )
                retention_scored.extend(evaluated.pop("rows"))
                retention_receipts.append(evaluated)
                retention_mutations += int(not evaluated["passed"])

            if event_count in RESTART_POINTS:
                working.save(checkpoint)
                working = RecalibrationEventMachine.load(checkpoint)
                restart_mismatches += int(working.to_payload() != uninterrupted.to_payload())

    final_state = working.to_payload()
    twin_state = uninterrupted.to_payload()
    restart_mismatches += int(final_state != twin_state)
    return {
        "seed": seed,
        "initial_theta": IDENTITY_THETA,
        "initial_state_hash": initial_hash,
        "event_rows": event_rows,
        "comparison_rows": comparison_rows,
        "release_rows": release_rows,
        "ack_rows": ack_rows,
        "retention_rows": retention_scored,
        "retention_receipts": retention_receipts,
        "released_event_count": sum(int(row["update_count"]) for row in release_rows),
        "unreleased_tail_count": len(order) - sum(int(row["update_count"]) for row in release_rows),
        "unshufflable_block_count": sum(not row["shufflable"] for row in release_rows),
        "restart_mismatch_count": restart_mismatches,
        "future_prediction_mismatch_count": prediction_mismatches,
        "retention_state_mutation_count": retention_mutations,
        "acknowledged_exactly_once": len(ack_rows)
        == len(release_rows)
        == len({row["release_index"] for row in ack_rows}),
        "final_state": final_state,
        "final_state_hash": working.state_hash(),
        "final_numerical_state_hash": _numerical_hash(working),
        "uninterrupted_final_state_hash": uninterrupted.state_hash(),
        "event_log_path": str(event_log),
    }


def _reject_without_mutation(
    machine: RecalibrationEventMachine, operation: Any, marker: str
) -> bool:
    before = machine.to_payload()
    try:
        operation()
    except ValueError as error:
        return marker in str(error) and before == machine.to_payload()
    return False


def run_lifecycle_controls(
    online_rows: Sequence[Mapping[str, Any]], count_config: CountConfig, root: Path
) -> JsonDict:
    """Exercise invalid order, future labels, duplicates, and pre-ack recovery."""

    root.mkdir(parents=True, exist_ok=True)
    machine = RecalibrationEventMachine.create(count_config)
    rows = list(online_rows[:16])
    for index, row in enumerate(rows):
        machine.predict(str(row["source_id"]), float(row["probability"]), index // 8)
    block0 = [(str(row["source_id"]), int(row["label"])) for row in rows[:8]]
    block1 = [(str(row["source_id"]), int(row["label"])) for row in rows[8:]]
    unchanged = []
    unchanged.append(
        _reject_without_mutation(
            machine, lambda: machine.release(1, block1), "release_out_of_order"
        )
    )
    future = [(block1[0][0], block1[0][1]), *block0[1:]]
    unchanged.append(
        _reject_without_mutation(machine, lambda: machine.release(0, future), "feedback_prerelease")
    )
    machine.release(0, block0)
    checkpoint = root / "crash-before-ack.json"
    machine.save(checkpoint)
    restored = RecalibrationEventMachine.load(checkpoint)
    restored.acknowledge(0)
    restored.save(checkpoint)
    before_duplicate_ack = len(restored.journal)
    restored.acknowledge(0)
    duplicate_ack_noop = len(restored.journal) == before_duplicate_ack
    unchanged.append(
        _reject_without_mutation(
            restored, lambda: restored.release(0, block0), "duplicate_feedback_release"
        )
    )
    return {
        "duplicate_release_rejected": unchanged[2],
        "out_of_order_release_rejected": unchanged[0],
        "future_label_sentinel_rejected": unchanged[1],
        "crash_before_ack_recovered": 0 in restored.acknowledgments
        and restored.next_release_index == 1,
        "duplicate_ack_noop": duplicate_ack_noop,
        "state_unchanged_after_rejections": all(unchanged),
    }


def _bootstrap_one(
    stream: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    count_config: CountConfig,
    *,
    seed: int,
    replay_index: int,
) -> list[JsonDict]:
    """Reset and retrain all arms for one source-component resample."""

    machine = RecalibrationEventMachine.create(count_config)
    initial_hash = identity_state_hash()
    losses: dict[str, float] = defaultdict(float)
    costs: dict[str, float] = defaultdict(float)
    covered: dict[str, int] = defaultdict(int)
    identities: list[str] = []
    labels: list[int] = []
    shufflable_blocks = 0
    for event_index, source in enumerate(stream):
        event_id = f"{seed}:{replay_index}:{event_index}:{source['source_id']}"
        identities.append(event_id)
        labels.append(int(source["label"]))
        prediction = machine.predict(event_id, float(source["probability"]), event_index // 8)
        for arm, values in _sealed_arms(prediction).items():
            loss = _brier(float(values["q"]), int(source["label"]))
            _action, cost, non_escalated = _action_cost(float(values["q"]), int(source["label"]))
            losses[arm] += loss
            costs[arm] += cost
            covered[arm] += int(non_escalated)
        event_count = event_index + 1
        due = _release_schedule(event_count)
        if due is not None:
            start = due * 8
            feedback = list(
                zip(identities[start : start + 8], labels[start : start + 8], strict=True)
            )
            shufflable_blocks += int(_release_properties(feedback)["shufflable"])
            machine.release(due, feedback)
            machine.acknowledge(due)

    retention_losses: dict[str, float] = defaultdict(float)
    for source in retention_rows:
        for arm, probability in _arm_probabilities(machine, float(source["probability"])).items():
            retention_losses[arm] += _brier(probability, int(source["label"]))
    movements = {
        "bounded": float(np.linalg.norm(machine.constrained.theta - KNOTS)),
        "raw": 0.0,
        "global_count": float(len(machine.count_arms["global_count"].processed_event_ids)),
        "local_count": float(len(machine.count_arms["local_count"].processed_event_ids)),
        "shuffled_feedback": float(np.linalg.norm(machine.shuffled.theta - KNOTS)),
    }
    denominator = len(stream)
    retention_denominator = len(retention_rows)
    unit_id = f"order-{seed}-resample-{replay_index:04d}"
    return [
        {
            "unit_id": unit_id,
            "order_seed": seed,
            "resample_index": replay_index,
            "arm": arm,
            "raw_squared_error_numerator": losses[arm],
            "raw_squared_error_denominator": denominator,
            "mean_brier": losses[arm] / denominator,
            "action_cost_numerator": costs[arm],
            "action_cost_denominator": denominator,
            "mean_action_cost": costs[arm] / denominator,
            "coverage_numerator": covered[arm],
            "coverage_denominator": denominator,
            "coverage": covered[arm] / denominator,
            "retention_brier_numerator": retention_losses[arm],
            "retention_denominator": retention_denominator,
            "retention_mean_brier": retention_losses[arm] / retention_denominator,
            "state_movement": movements[arm],
            "initial_state_hash": initial_hash,
            "final_state_hash": _numerical_hash(machine),
            "shufflable_block_count": shufflable_blocks,
            "metric_direction": "lower_brier_and_cost_are_better",
            "seed": seed,
            "censoring": {"prediction_loss": False, "unreleased_tail_events": 8},
            "provenance": "source_component_resample_retrained_from_identity",
        }
        for arm in ARMS
    ]


def causal_bootstrap(
    online_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[str]],
    count_config: CountConfig,
    *,
    replays_per_order: int = 1000,
    order_seeds: Sequence[int] = ORDER_SEEDS,
    checkpoint_path: Path,
    progress_callback: Any | None = None,
) -> JsonDict:
    """Run complete causal retraining for every registered resample."""

    if replays_per_order < 1:
        raise ValueError("replays_per_order_invalid")
    by_id = {str(row["source_id"]): row for row in online_rows}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[JsonDict] = []
    completed = 0
    with checkpoint_path.open("w", encoding="utf-8") as checkpoint:
        for seed in order_seeds:
            order = list(orders.get(str(seed), []))
            if len(order) != len(by_id) or set(order) != set(by_id):
                raise ValueError(f"bootstrap_order_invalid:{seed}")
            rng = np.random.default_rng(seed)
            for replay_index in range(replays_per_order):
                sampled = rng.integers(0, len(order), size=len(order))
                stream = [by_id[order[int(index)]] for index in sampled]
                replay_rows = _bootstrap_one(
                    stream,
                    retention_rows,
                    count_config,
                    seed=seed,
                    replay_index=replay_index,
                )
                all_rows.extend(replay_rows)
                completed += 1
                checkpoint.write(
                    json.dumps(
                        {
                            "completed_replay": completed,
                            "order_seed": seed,
                            "resample_index": replay_index,
                            "row_hash": canonical_hash(replay_rows),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                checkpoint.flush()
                if completed % 50 == 0:
                    os.fsync(checkpoint.fileno())
                if progress_callback is not None:
                    progress_callback(completed, replays_per_order * len(order_seeds))
        os.fsync(checkpoint.fileno())
    return {
        "rows": all_rows,
        "completed_replays": completed,
        "expected_replays": replays_per_order * len(order_seeds),
        "each_resample_retrained": True,
        "source_component_grouping_preserved": True,
        "resampled_adapted_loss_rows": False,
        "checkpoint_path": str(checkpoint_path),
    }


def _interval(values: Sequence[float]) -> JsonDict:
    if not values:
        return {"count": 0, "mean": None, "lower95": None, "upper95": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "lower95": float(np.quantile(values, 0.025)),
        "upper95": float(np.quantile(values, 0.975)),
    }


def _holm_one_sided(contrasts: Mapping[str, Sequence[float]]) -> JsonDict:
    raw = {
        name: (1.0 + sum(value <= 0.0 for value in values)) / (len(values) + 1.0)
        for name, values in contrasts.items()
    }
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, raw[name] * (size - index)))
        adjusted[name] = running
    return {
        name: {
            "one_sided_p": raw[name],
            "holm_adjusted_p": adjusted[name],
            "reject_at_0_05": adjusted[name] <= 0.05,
        }
        for name in raw
    }


def reduce_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_replays: int = 5000,
    expected_order_count: int = 5,
) -> JsonDict:
    """Independently reduce raw per-resample arm numerators and contrasts."""

    by_unit: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    seeds: set[int] = set()
    for row in rows:
        denominator = int(row.get("raw_squared_error_denominator", 0))
        retention_denominator = int(row.get("retention_denominator", 0))
        if denominator <= 0 or retention_denominator <= 0:
            raise ValueError("row_denominator_invalid")
        mean = float(row["raw_squared_error_numerator"]) / denominator
        if not math.isclose(mean, float(row["mean_brier"]), abs_tol=1e-12):
            raise ValueError("mean_brier_mismatch")
        cost = float(row["action_cost_numerator"]) / int(row["action_cost_denominator"])
        if not math.isclose(cost, float(row["mean_action_cost"]), abs_tol=1e-12):
            raise ValueError("mean_action_cost_mismatch")
        retention = float(row["retention_brier_numerator"]) / retention_denominator
        if not math.isclose(retention, float(row["retention_mean_brier"]), abs_tol=1e-12):
            raise ValueError("retention_brier_mismatch")
        key = (str(row["unit_id"]), int(row["seed"]))
        arm = str(row["arm"])
        if arm in by_unit[key]:
            raise ValueError("duplicate_unit_arm")
        by_unit[key][arm] = row
        seeds.add(int(row["seed"]))
    if any(set(arm_rows) != set(ARMS) for arm_rows in by_unit.values()):
        raise ValueError("unit_arm_roster_invalid")

    contrast_values: dict[str, list[float]] = defaultdict(list)
    retention_degradation: list[float] = []
    candidate_coverage: list[float] = []
    for arm_rows in by_unit.values():
        candidate = arm_rows["bounded"]
        candidate_brier = float(candidate["mean_brier"])
        for arm in ("raw", "global_count", "local_count"):
            contrast_values[f"brier_vs_{arm}"].append(
                float(arm_rows[arm]["mean_brier"]) - candidate_brier
            )
        if int(candidate["shufflable_block_count"]) > 0:
            contrast_values["brier_vs_shuffled_feedback"].append(
                float(arm_rows["shuffled_feedback"]["mean_brier"]) - candidate_brier
            )
        contrast_values["action_cost_vs_raw"].append(
            float(arm_rows["raw"]["mean_action_cost"]) - float(candidate["mean_action_cost"])
        )
        retention_degradation.append(
            float(candidate["retention_mean_brier"])
            - float(arm_rows["raw"]["retention_mean_brier"])
        )
        candidate_coverage.append(float(candidate["coverage"]))
    contrasts = {name: _interval(values) for name, values in contrast_values.items()}
    holm_inputs = {
        name: values for name, values in contrast_values.items() if name.startswith("brier_vs_")
    }
    holm = _holm_one_sided(holm_inputs)
    expected_names = {
        "brier_vs_raw",
        "brier_vs_global_count",
        "brier_vs_local_count",
        "brier_vs_shuffled_feedback",
        "action_cost_vs_raw",
    }
    measurement_complete = (
        len(by_unit) == expected_replays
        and len(seeds) == expected_order_count
        and set(contrasts) == expected_names
        and all(value["count"] == expected_replays for value in contrasts.values())
    )
    benefit_passed = measurement_complete and all(
        contrasts[name]["lower95"] is not None
        and float(contrasts[name]["lower95"]) > 0.0
        and holm[name]["reject_at_0_05"]
        for name in holm_inputs
    )
    benefit_passed = bool(
        benefit_passed
        and contrasts["action_cost_vs_raw"]["lower95"] is not None
        and float(contrasts["action_cost_vs_raw"]["lower95"]) > 0.0
        and float(np.quantile(candidate_coverage, 0.025)) >= 0.10
    )
    retention_interval = _interval(retention_degradation)
    retention_passed = bool(
        measurement_complete
        and retention_interval["upper95"] is not None
        and float(retention_interval["upper95"]) <= 0.005
    )
    return {
        "measurement_complete": measurement_complete,
        "benefit_passed": benefit_passed,
        "retention_passed": retention_passed,
        "completed_replays": len(by_unit),
        "order_count": len(seeds),
        "contrasts": contrasts,
        "holm_one_sided": holm,
        "candidate_coverage": _interval(candidate_coverage),
        "retention_degradation": retention_interval,
    }


def precondition_row(
    check: str,
    upstream: str,
    path: str | Path,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Name exact prerequisite operands before dependent measurement starts."""

    return {
        "check": check,
        "upstream": upstream,
        "path": str(path),
        "field": field,
        "op": "eq",
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "required": True,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate independent qualification and every frozen protocol byte."""

    root = root.resolve()
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    artifacts: dict[str, JsonDict] = {}
    for name, relative in (("Experiment 7574", EXP7574_PATH), ("Experiment 7575", EXP7575_PATH)):
        path = root / relative
        value = _load_object(path)
        artifacts[name] = value
        exists = bool(value)
        checks.append(
            precondition_row(
                f"{name.lower().replace(' ', '')}_artifact_exists",
                name,
                relative,
                "path",
                "readable_json_object",
                "readable_json_object" if exists else "missing_or_invalid",
                exists,
            )
        )
        if exists:
            hashes.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    exp7574 = artifacts["Experiment 7574"]
    exp7575 = artifacts["Experiment 7575"]
    for check, upstream, path, field, expected, observed in (
        (
            "exp7574_independent_qualification",
            "Experiment 7574",
            EXP7574_PATH,
            "recalibration_ready_score",
            1,
            exp7574.get("recalibration_ready_score"),
        ),
        (
            "exp7574_compute_envelope",
            "Experiment 7574",
            EXP7574_PATH,
            "benchmark_receipt.fits_2400_seconds_with_reserve",
            True,
            exp7574.get("benchmark_receipt", {}).get("fits_2400_seconds_with_reserve"),
        ),
        (
            "exp7574_not_flagged",
            "Experiment 7574",
            EXP7574_PATH,
            "flagged_adversarial",
            False,
            exp7574.get("flagged_adversarial"),
        ),
        (
            "exp7575_protocol_ready",
            "Experiment 7575",
            EXP7575_PATH,
            "online_protocol_ready_score",
            1,
            exp7575.get("online_protocol_ready_score"),
        ),
        (
            "exp7575_cached_roles_ready",
            "Experiment 7575",
            EXP7575_PATH,
            "cached_roles_ready_score",
            1,
            exp7575.get("cached_roles_ready_score"),
        ),
        (
            "historical_exposure_preserved",
            "Experiment 7575",
            EXP7575_PATH,
            "fresh_confirmatory_claim_allowed",
            False,
            exp7575.get("fresh_confirmatory_claim_allowed"),
        ),
    ):
        checks.append(
            precondition_row(check, upstream, path, field, expected, observed, observed == expected)
        )
    for sidecar_name in ("cached_roles", "frozen_protocol"):
        receipt = exp7575.get("raw_sidecars", {}).get(sidecar_name, {})
        path = _resolved(root, str(receipt.get("path") or "missing"))
        observed = sha256_file(path) if path.is_file() else None
        expected = receipt.get("sha256")
        checks.append(
            precondition_row(
                f"exp7575_{sidecar_name}_hash",
                "Experiment 7575",
                str(receipt.get("path") or path),
                "sha256",
                expected,
                observed,
                expected is not None and observed == expected,
            )
        )
        if path.is_file():
            hashes.append(
                {
                    "path": str(receipt.get("path")),
                    "sha256": observed,
                    "bytes": path.stat().st_size,
                }
            )
    spec = root / SPEC_PATH
    spec_text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    checks.append(
        precondition_row(
            "requirement_present",
            "continuous-learning capability",
            SPEC_PATH,
            "REQ-CL-7578",
            True,
            "REQ-CL-7578" in spec_text,
            "REQ-CL-7578" in spec_text,
        )
    )
    if spec.is_file():
        hashes.append(
            {
                "path": SPEC_PATH.as_posix(),
                "sha256": sha256_file(spec),
                "bytes": spec.stat().st_size,
            }
        )
    return checks, hashes


def validate_roles(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Reject missing, duplicate, relabeled, or nonbinary cached groups."""

    seen: set[str] = set()
    for role, expected in ROLE_COUNTS.items():
        rows = list(roles.get(role, []))
        if len(rows) != expected:
            raise ValueError(f"role_count_invalid:{role}")
        for row in rows:
            source_id = str(row.get("source_id") or "")
            if not source_id or source_id in seen:
                raise ValueError("source_id_duplicate")
            seen.add(source_id)
            if row.get("role") != role:
                raise ValueError(f"role_identity_invalid:{role}")
            if row.get("label") not in (0, 1):
                raise ValueError(f"binary_label_invalid:{role}")
            probability = float(row.get("probability", math.nan))
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability_invalid:{role}")


def load_inputs(root: Path) -> tuple[dict[str, list[JsonDict]], JsonDict]:
    """Read only the hash-bound Exp7575 role and protocol sidecars."""

    root = root.resolve()
    artifact = _load_object(root / EXP7575_PATH)
    if not artifact:
        raise ValueError("exp7575_artifact_missing")
    receipts = artifact.get("raw_sidecars", {})
    role_receipt = receipts.get("cached_roles", {})
    role_path = _resolved(root, str(role_receipt.get("path") or ""))
    if not role_path.is_file() or sha256_file(role_path) != role_receipt.get("sha256"):
        raise ValueError("cached_roles_hash_mismatch")
    rows = _read_jsonl(role_path)
    if len(rows) != int(role_receipt.get("rows", -1)):
        raise ValueError("cached_roles_count_mismatch")
    roles: dict[str, list[JsonDict]] = {name: [] for name in ROLE_COUNTS}
    for row in rows:
        role = str(row.get("role"))
        if role not in roles:
            raise ValueError(f"unknown_role:{role}")
        roles[role].append(row)
    for role_rows in roles.values():
        role_rows.sort(key=lambda row: str(row["source_id"]))
    validate_roles(roles)

    protocol_receipt = receipts.get("frozen_protocol", {})
    protocol_path = _resolved(root, str(protocol_receipt.get("path") or ""))
    if not protocol_path.is_file() or sha256_file(protocol_path) != protocol_receipt.get("sha256"):
        raise ValueError("frozen_protocol_hash_mismatch")
    protocol = _load_object(protocol_path)
    if not protocol:
        raise ValueError("frozen_protocol_invalid")
    rebuilt = freeze_protocol(roles)
    if protocol.get("protocol_sha256") != artifact.get("protocol_sha256"):
        raise ValueError("protocol_artifact_hash_mismatch")
    if rebuilt["protocol_sha256"] != protocol.get("protocol_sha256"):
        raise ValueError("protocol_role_hash_mismatch")
    if protocol.get("order_seeds") != list(ORDER_SEEDS):
        raise ValueError("protocol_order_seed_mismatch")
    return roles, protocol


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], root: Path) -> JsonDict:
    """Publish complete JSONL bytes with a local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def _json_receipt(path: Path, value: Mapping[str, Any], root: Path) -> JsonDict:
    atomic_json(path, dict(value))
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_sidecars(
    root: Path,
    *,
    causal_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    release_ack_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    learned_state: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Store every reconstructive row and final learned state outside summaries."""

    raw = root / RAW_DIR
    return {
        "causal_rows": _write_jsonl(raw / "causal_event_rows.jsonl", causal_rows, root),
        "comparison_rows": _write_jsonl(raw / "comparison_rows.jsonl", comparison_rows, root),
        "release_ack_rows": _write_jsonl(raw / "release_ack_rows.jsonl", release_ack_rows, root),
        "retention_rows": _write_jsonl(raw / "retention_rows.jsonl", retention_rows, root),
        "bootstrap_rows": _write_jsonl(raw / "bootstrap_rows.jsonl", bootstrap_rows, root),
        "learned_state": _json_receipt(raw / "learned_state.json", learned_state, root),
    }


def _receipt_rows(root: Path, name: str, receipt: Mapping[str, Any]) -> list[JsonDict]:
    path = _resolved(root, str(receipt.get("path") or ""))
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"sidecar_hash_mismatch:{name}")
    if path.stat().st_size != int(receipt.get("bytes", -1)):
        raise ValueError(f"sidecar_size_mismatch:{name}")
    rows = _read_jsonl(path)
    if len(rows) != int(receipt.get("rows", -1)):
        raise ValueError(f"sidecar_row_count_mismatch:{name}")
    return rows


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _gate(check: str, category: str, expected: Any, observed: Any, *, op: str = "eq") -> JsonDict:
    passed = {
        "eq": observed == expected,
        "gt": float(observed) > float(expected),
        "le": float(observed) <= float(expected),
    }[op]
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": "Validity, readiness, benefit, and retention remain separate.",
    }


def acceptance_gates(
    reduction: Mapping[str, Any],
    *,
    preconditions_passed: bool,
    lifecycle_passed: bool,
    affected_passed: bool,
    terminal_passed: bool,
) -> list[JsonDict]:
    """Keep a valid null usable by separating each decision class."""

    return [
        _gate("preconditions", "validity", True, preconditions_passed),
        _gate("causal_lifecycle", "validity", True, lifecycle_passed),
        _gate("affected_validation", "validity", True, affected_passed),
        _gate("terminal_readers", "validity", True, terminal_passed),
        _gate("registered_uncertainty", "readiness", True, reduction["measurement_complete"]),
        _gate("informative_feedback_advantage", "benefit", True, reduction["benefit_passed"]),
        _gate("retention_non_regression", "retention", True, reduction["retention_passed"]),
        _gate("fresh_confirmatory_claim", "freshness", False, False),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row["check"] for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def field_principles() -> dict[str, str]:
    values = {
        "honest_verdict": "A complete prefix reports terminal work; it does not establish benefit.",
        "verdict_class": "One closed class distinguishes a valid null from blocked or disqualified work.",
        "flagged_adversarial": "The exact terminal verifier outcome cannot open readiness when flagged.",
        "gate_check_summary": "Blocked evidence names its exact failed operand; other failures remain visible.",
        "acceptance_gate_results": "Separate validity, readiness, benefit, and retention keep a null usable.",
        "rows": "Each replay arm retains raw numerators, denominators, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Actual cached replay is distinct from planned live inference.",
        "MODEL_SPECS": "No current LLM task means an empty current model roster.",
        "invocation_counts": "Loads, forwards, generations, and tokens stay independently zero.",
        "duration_s": "Monotonic current work excludes inherited timing and artificial sleeps.",
        "source_artifact_hashes": "Conclusions bind to exact source bytes, including qualified producers.",
        "validation_receipts": "Each scoped command records command, worktree, exit, and log identity.",
        "verifier_is_oracle": "Label-accessing controls cannot support an oracle-distinct positive claim.",
        "learning_measurement_complete_score": "One requires all orders, uncertainty, and restart checks.",
        "exploratory_learning_benefit_score": "One requires advantage over raw, count, and informative shuffled controls.",
        "retention_pass_score": "One requires the registered retained Brier non-regression limit.",
        "causal_event_rows_path": "The path binds every prediction, release, state update, and durable ack.",
        "fresh_confirmatory_claim_allowed": "Historical exposure makes fresh confirmatory status false.",
        "state_hashes": "State hashes expose reproducible updates and exact restart equality.",
    }
    return values


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    return {
        arm: {
            "raw_squared_error_numerator": float(
                sum(float(row["raw_squared_error_numerator"]) for row in arm_rows)
            ),
            "raw_squared_error_denominator": len(arm_rows),
            "mean_brier": float(
                np.mean([float(row["raw_squared_error_numerator"]) for row in arm_rows])
            ),
            "mean_action_cost": float(
                np.mean([float(row["realized_action_cost"]) for row in arm_rows])
            ),
            "coverage": float(np.mean([bool(row["non_escalated"]) for row in arm_rows])),
        }
        for arm, arm_rows in grouped.items()
    }


def _measurement_metrics(replays: Sequence[Mapping[str, Any]]) -> JsonDict:
    comparisons = [row for replay in replays for row in replay["comparison_rows"]]
    retention = [row for replay in replays for row in replay["retention_rows"]]
    releases = [row for replay in replays for row in replay["release_rows"]]
    acks = [row for replay in replays for row in replay["ack_rows"]]
    return {
        "prequential": _arm_summary(comparisons),
        "retention": _arm_summary(retention),
        "state_movement_l2": _interval([float(row["state_movement_l2"]) for row in releases]),
        "update_latency_s": _interval([float(row["update_latency_s"]) for row in releases]),
        "durable_acknowledgment_latency_s": _interval(
            [float(row["durable_ack_latency_s"]) for row in acks]
        ),
    }


def build_artifact(
    *,
    root: Path,
    sidecars: Mapping[str, Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
    lifecycle_controls: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    expected_replays: int = 5000,
    expected_order_count: int = 5,
) -> JsonDict:
    """Assemble one terminal record from independently reducible evidence."""

    reduction = reduce_bootstrap(
        bootstrap_rows,
        expected_replays=expected_replays,
        expected_order_count=expected_order_count,
    )
    preconditions_passed = all(
        row.get("passed") is True for row in preconditions_checked if row.get("required") is True
    )
    replay_lifecycle = bool(
        len(replays) == expected_order_count
        and all(
            replay.get("restart_mismatch_count") == 0
            and replay.get("future_prediction_mismatch_count") == 0
            and replay.get("retention_state_mutation_count") == 0
            and replay.get("acknowledged_exactly_once") is True
            for replay in replays
        )
    )
    lifecycle_passed = replay_lifecycle and all(
        value is True for value in lifecycle_controls.values()
    )
    affected_passed = _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
    has_terminal_receipts = any(
        row.get("name") in TERMINAL_CHECK_NAMES for row in validation_receipts
    )
    terminal_passed = (
        _receipts_pass(validation_receipts, TERMINAL_CHECK_NAMES) if has_terminal_receipts else True
    )
    gates = acceptance_gates(
        reduction,
        preconditions_passed=preconditions_passed,
        lifecycle_passed=lifecycle_passed,
        affected_passed=affected_passed,
        terminal_passed=terminal_passed,
    )
    validity = all(row["passed"] for row in gates if row["category"] == "validity")
    measurement_complete = bool(validity and reduction["measurement_complete"])
    benefit = bool(measurement_complete and reduction["benefit_passed"])
    retention = bool(measurement_complete and reduction["retention_passed"])
    if not validity:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_required_validation_or_lifecycle"
    elif benefit and retention:
        verdict_class = "positive"
        honest_verdict = "complete_positive_exploratory_continuous_proper_loss"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_continuous_proper_loss_valid"
    flagged = any(
        row.get("name") == "adversarial_verify" and row.get("exit_code") != 0
        for row in validation_receipts
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "status": "complete",
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_identity": HISTORICAL_MODEL_ID,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "empirical_delayed_feedback_replay_cached_original_forecasts_no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "duration_s": float(duration_s),
        "phase_spans": [dict(row) for row in phase_spans],
        "random_seed": ORDER_SEEDS[0],
        "order_seeds": list(ORDER_SEEDS),
        "bootstrap_replays_per_order": expected_replays // expected_order_count,
        "rows": [dict(row) for row in bootstrap_rows],
        "independent_reduction": reduction,
        "measurement_metrics": _measurement_metrics(replays),
        "lifecycle_controls": dict(lifecycle_controls),
        "replay_summaries": [
            {
                key: deepcopy(replay[key])
                for key in (
                    "seed",
                    "released_event_count",
                    "unreleased_tail_count",
                    "unshufflable_block_count",
                    "restart_mismatch_count",
                    "future_prediction_mismatch_count",
                    "retention_state_mutation_count",
                    "acknowledged_exactly_once",
                    "final_state_hash",
                    "final_numerical_state_hash",
                )
            }
            for replay in replays
        ],
        "raw_sidecars": deepcopy(dict(sidecars)),
        "causal_event_rows_path": sidecars["causal_rows"]["path"],
        "state_hashes": {
            "restart_equality": replay_lifecycle,
            "orders": [
                {
                    "seed": replay["seed"],
                    "initial": replay["initial_state_hash"],
                    "final": replay["final_state_hash"],
                    "numerical_final": replay["final_numerical_state_hash"],
                }
                for replay in replays
            ],
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "learning_measurement_complete_score": int(measurement_complete),
        "exploratory_learning_benefit_score": int(benefit),
        "retention_pass_score": int(retention),
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse_after_historical_exposure",
        "deployment_promotion_available": False,
        "verifier_is_oracle": False,
        "positive_claim": benefit,
        "preconditions_checked": [dict(row) for row in preconditions_checked],
        "source_artifact_hashes": [dict(row) for row in source_artifact_hashes],
        "validation_receipts": [dict(row) for row in validation_receipts],
        "field_principles": field_principles(),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _test_roles() -> dict[str, list[JsonDict]]:
    roles: dict[str, list[JsonDict]] = {}
    offsets = {"fit": 0, "tune": 200, "policy": 300, "online": 400, "test": 600}
    for role, count in ROLE_COUNTS.items():
        rows: list[JsonDict] = []
        for index in range(count):
            value = offsets[role] + index
            rows.append(
                {
                    "source_id": f"{role}-{value:03d}",
                    "group_id": f"group-{role}-{value:03d}",
                    "role": role,
                    "official_split": "validation" if role == "test" else "train",
                    "probability": 0.04 + 0.92 * ((value % 17) / 16),
                    "label": int((value * 7 + value // 3) % 5 >= 2),
                    "context_sha256": f"sha256:context-{value}",
                    "response_sha256": f"sha256:response-{value}",
                }
            )
        roles[role] = rows
    return roles


def _test_receipts(root: Path) -> list[JsonDict]:
    return [
        {
            "name": name,
            "command": ["python", name],
            "cwd": str(root.resolve()),
            "exit_code": 0,
            "timed_out": False,
            "log_sha256": f"sha256:{name}",
        }
        for name in (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_test_artifact(root: Path) -> JsonDict:
    """Build compact deterministic evidence for mutation and fresh-reader tests."""

    roles = _test_roles()
    protocol = freeze_protocol(roles)
    config = count_config_from_rows(roles["fit"])
    seeds = ORDER_SEEDS[:2]
    replays = [
        measure_order(
            roles["online"],
            roles["test"],
            protocol["orders"][str(seed)],
            config,
            seed=seed,
            state_dir=root / "state" / str(seed),
        )
        for seed in seeds
    ]
    lifecycle = run_lifecycle_controls(roles["online"], config, root / "controls")
    bootstrap = causal_bootstrap(
        roles["online"],
        roles["test"],
        protocol["orders"],
        config,
        replays_per_order=2,
        order_seeds=seeds,
        checkpoint_path=root / "bootstrap-checkpoint.jsonl",
    )
    causal = [row for replay in replays for row in replay["event_rows"]]
    comparisons = [row for replay in replays for row in replay["comparison_rows"]]
    retention = [row for replay in replays for row in replay["retention_rows"]]
    release_ack = [
        row for replay in replays for row in (*replay["release_rows"], *replay["ack_rows"])
    ]
    learned_state = {str(replay["seed"]): replay["final_state"] for replay in replays}
    sidecars = write_sidecars(
        root,
        causal_rows=causal,
        comparison_rows=comparisons,
        release_ack_rows=release_ack,
        retention_rows=retention,
        bootstrap_rows=bootstrap["rows"],
        learned_state=learned_state,
    )
    preconditions = [precondition_row("fixture", "test", "fixture", "ready", True, True, True)]
    artifact = build_artifact(
        root=root,
        sidecars=sidecars,
        bootstrap_rows=bootstrap["rows"],
        replays=replays,
        lifecycle_controls=lifecycle,
        validation_receipts=_test_receipts(root),
        preconditions_checked=preconditions,
        source_artifact_hashes=[{"path": "fixture", "sha256": canonical_hash(roles), "bytes": 1}],
        duration_s=1.0,
        phase_spans=[],
        expected_replays=4,
        expected_order_count=2,
    )
    artifact["order_seeds"] = list(seeds)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    failed: Mapping[str, Any], checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Publish external absence without synthetic measurement evidence."""

    summary = {
        key: failed.get(key)
        for key in ("check", "upstream", "path", "field", "op", "expected", "observed", "passed")
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "honest_verdict": f"complete_blocked_{failed['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "status": "complete",
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_identity": HISTORICAL_MODEL_ID,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_before_empirical_delayed_feedback_replay",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "duration_s": float(duration_s),
        "rows": [],
        "raw_sidecars": {},
        "causal_event_rows_path": None,
        "state_hashes": {"restart_equality": False, "orders": []},
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "learning_measurement_complete_score": 0,
        "exploratory_learning_benefit_score": 0,
        "retention_pass_score": 0,
        "fresh_confirmatory_claim_allowed": False,
        "deployment_promotion_available": False,
        "verifier_is_oracle": False,
        "positive_claim": False,
        "preconditions_checked": [dict(row) for row in checks],
        "source_artifact_hashes": [],
        "validation_receipts": [],
        "field_principles": field_principles(),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    expected_replays: int = 5000,
    require_terminal: bool = False,
) -> JsonDict:
    """Reject changed claims, rows, sidecars, scores, receipts, or identity."""

    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_mismatch")
    if value.get("verdict_class") == "blocked":
        summary = value.get("gate_check_summary")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            raise ValueError("blocked_gate_summary_invalid")
        if value.get("reproducibility_checksum") != reproducibility_checksum(value):
            raise ValueError("reproducibility_checksum_mismatch")
        return {"measurement_complete": False, "benefit_passed": False, "retention_passed": False}
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        raise ValueError("terminal_prefix_missing")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("current_model_roster_nonempty")
    if value.get("no_model_load") is not True or value.get("model_invoked") is not False:
        raise ValueError("no_model_load_contract_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        raise ValueError("current_invocations_nonzero")
    if value.get("fresh_confirmatory_claim_allowed") is not False:
        raise ValueError("fresh_claim_forbidden")
    if value.get("deployment_promotion_available") is not False:
        raise ValueError("deployment_promotion_forbidden")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_PRINCIPLE_FIELDS) <= set(principles):
        raise ValueError("field_principles_incomplete")
    source_hashes = value.get("source_artifact_hashes")
    if not isinstance(source_hashes, list) or not source_hashes:
        raise ValueError("source_artifact_hashes_missing")

    order_seeds = value.get("order_seeds")
    if not isinstance(order_seeds, list) or not order_seeds:
        raise ValueError("order_seeds_missing")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rows_missing")
    reduction = reduce_bootstrap(
        rows,
        expected_replays=expected_replays,
        expected_order_count=len(order_seeds),
    )
    if value.get("independent_reduction") != reduction:
        raise ValueError("independent_reduction_mismatch")

    expected_measurement = int(reduction["measurement_complete"])
    receipts = value.get("validation_receipts")
    if not isinstance(receipts, list):
        raise ValueError("validation_receipts_missing")
    affected_passed = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    if require_terminal and not terminal_passed:
        raise ValueError("terminal_validation_incomplete")
    lifecycle = value.get("lifecycle_controls")
    lifecycle_passed = isinstance(lifecycle, Mapping) and all(
        item is True for item in lifecycle.values()
    )
    state_hashes = value.get("state_hashes")
    lifecycle_passed = bool(
        lifecycle_passed
        and isinstance(state_hashes, Mapping)
        and state_hashes.get("restart_equality") is True
    )
    preconditions = value.get("preconditions_checked")
    preconditions_passed = isinstance(preconditions, list) and all(
        row.get("passed") is True for row in preconditions if row.get("required") is True
    )
    validity = preconditions_passed and lifecycle_passed and affected_passed
    if require_terminal:
        validity = validity and terminal_passed
    expected_measurement = int(bool(expected_measurement and validity))
    if value.get("learning_measurement_complete_score") != expected_measurement:
        raise ValueError("measurement_score_mismatch")
    if value.get("exploratory_learning_benefit_score") != int(
        bool(expected_measurement and reduction["benefit_passed"])
    ):
        raise ValueError("benefit_score_mismatch")
    if value.get("retention_pass_score") != int(
        bool(expected_measurement and reduction["retention_passed"])
    ):
        raise ValueError("retention_score_mismatch")

    sidecars = value.get("raw_sidecars")
    if not isinstance(sidecars, Mapping):
        raise ValueError("raw_sidecars_missing")
    required_sidecars = {
        "causal_rows",
        "comparison_rows",
        "release_ack_rows",
        "retention_rows",
        "bootstrap_rows",
        "learned_state",
    }
    if set(sidecars) != required_sidecars:
        raise ValueError("sidecar_roster_mismatch")
    bootstrap_rows = _receipt_rows(root, "bootstrap_rows", sidecars["bootstrap_rows"])
    if canonical_hash(bootstrap_rows) != canonical_hash(rows):
        raise ValueError("bootstrap_rows_terminal_mismatch")
    for name in ("causal_rows", "comparison_rows", "release_ack_rows", "retention_rows"):
        _receipt_rows(root, name, sidecars[name])
    learned = sidecars["learned_state"]
    learned_path = _resolved(root, str(learned.get("path") or ""))
    if not learned_path.is_file() or sha256_file(learned_path) != learned.get("sha256"):
        raise ValueError("sidecar_hash_mismatch:learned_state")
    if value.get("causal_event_rows_path") != sidecars["causal_rows"]["path"]:
        raise ValueError("causal_event_rows_path_mismatch")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("reproducibility_checksum_mismatch")
    return reduction


def cold_replay(
    path: Path,
    *,
    root: Path = REPO_ROOT,
    expected_replays: int = 5000,
) -> JsonDict:
    value = _load_object(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(
        value, root=root, expected_replays=expected_replays, require_terminal=True
    )


def independent_reduce(
    path: Path,
    *,
    root: Path = REPO_ROOT,
    expected_replays: int = 5000,
) -> JsonDict:
    value = _load_object(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(
        value, root=root, expected_replays=expected_replays, require_terminal=False
    )


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build serial tests, separate coverage, lint, type, and spec checks."""

    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7578",
    )


def terminal_commands(
    candidate: Path,
    root: Path = REPO_ROOT,
    *,
    allow_preterminal: bool = False,
) -> list[validation_scope.CommandSpec]:
    """Build cold replay, independent reduction, adversarial, and strict readers."""

    python = str(root / ".venv/bin/python")
    cold = [
        python,
        "-u",
        WRAPPER_PATH.as_posix(),
        "--date",
        RUN_DATE,
        "--root",
        str(root),
        "--cold-replay",
        str(candidate),
    ]
    if allow_preterminal:
        cold.append("--allow-preterminal")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay", tuple(cold), "exact_candidate", 300.0
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--root",
                str(root),
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
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
            "exact_candidate",
            300.0,
        ),
    ]


def write_affected_manifest(path: Path) -> None:
    """Freeze exact file scope before any validation command expands it."""

    atomic_json(
        path,
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    payload = {
        "phase": phase,
        "event": event,
        "elapsed_s": round(time.monotonic() - started, 3),
        **details,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _span(
    phase: str, phase_started: float, started: float, units: int
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - started,
        "end_offset_s": ended - started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def run_experiment(  # pragma: no cover - declared entrypoint is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, replay, validate exact bytes, and publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start", root=str(root))
    phase_started = time.monotonic()
    checks, source_hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next((row for row in checks if row["required"] and not row["passed"]), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed, checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "complete", completed_units=len(checks))

    manifest = raw_root / "affected_validation_manifest.json"
    write_affected_manifest(manifest)
    source_hashes.append(
        {
            "path": manifest.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest),
            "bytes": manifest.stat().st_size,
        }
    )
    progress(started, "input_custody", "before_cached_role_load")
    phase_started = time.monotonic()
    roles, protocol = load_inputs(root)
    spans.append(_span("input_custody", phase_started, started, sum(map(len, roles.values()))))
    progress(
        started,
        "input_custody",
        "after_cached_role_load",
        protocol_sha256=protocol["protocol_sha256"],
    )

    count_config = count_config_from_rows(roles["fit"])
    checkpoint_root = Path(tempfile.mkdtemp(prefix="carnot-exp7578-state-", dir="/tmp"))
    replays: list[JsonDict] = []
    progress(started, "causal_orders", "start", orders=len(ORDER_SEEDS))
    phase_started = time.monotonic()
    for completed, seed in enumerate(ORDER_SEEDS, 1):
        progress(started, "causal_orders", "before_order", seed=seed)
        replay = measure_order(
            roles["online"],
            roles["test"],
            protocol["orders"][str(seed)],
            count_config,
            seed=seed,
            state_dir=checkpoint_root / str(seed),
        )
        replays.append(replay)
        progress(
            started,
            "causal_orders",
            "after_order",
            seed=seed,
            completed_units=completed,
            releases=len(replay["release_rows"]),
        )
    spans.append(_span("causal_orders", phase_started, started, len(replays)))

    progress(started, "lifecycle_controls", "start")
    phase_started = time.monotonic()
    lifecycle = run_lifecycle_controls(roles["online"], count_config, checkpoint_root / "controls")
    spans.append(_span("lifecycle_controls", phase_started, started, len(lifecycle)))
    progress(started, "lifecycle_controls", "complete", passed=all(lifecycle.values()))

    bootstrap_checkpoint = raw_root / "bootstrap_checkpoint.jsonl"
    progress(
        started,
        "causal_bootstrap",
        "before_benchmark",
        expected_replays=1000 * len(ORDER_SEEDS),
    )
    phase_started = time.monotonic()
    bootstrap = causal_bootstrap(
        roles["online"],
        roles["test"],
        protocol["orders"],
        count_config,
        replays_per_order=1000,
        order_seeds=ORDER_SEEDS,
        checkpoint_path=bootstrap_checkpoint,
        progress_callback=lambda completed, total: (
            progress(
                started,
                "causal_bootstrap",
                "units_complete",
                completed_units=completed,
                total_units=total,
            )
            if completed % 50 == 0 or completed == total
            else None
        ),
    )
    spans.append(_span("causal_bootstrap", phase_started, started, bootstrap["completed_replays"]))
    progress(
        started,
        "causal_bootstrap",
        "after_benchmark",
        completed_units=bootstrap["completed_replays"],
    )

    causal_rows = [row for replay in replays for row in replay["event_rows"]]
    comparison_rows = [row for replay in replays for row in replay["comparison_rows"]]
    retention_rows = [row for replay in replays for row in replay["retention_rows"]]
    release_ack_rows = [
        row for replay in replays for row in (*replay["release_rows"], *replay["ack_rows"])
    ]
    learned_state = {str(replay["seed"]): replay["final_state"] for replay in replays}
    progress(started, "sidecars", "before_atomic_writes")
    phase_started = time.monotonic()
    sidecars = write_sidecars(
        root,
        causal_rows=causal_rows,
        comparison_rows=comparison_rows,
        release_ack_rows=release_ack_rows,
        retention_rows=retention_rows,
        bootstrap_rows=bootstrap["rows"],
        learned_state=learned_state,
    )
    spans.append(_span("sidecars", phase_started, started, len(sidecars)))
    progress(started, "sidecars", "after_atomic_writes", completed_units=len(sidecars))

    for relative in (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("python/carnot/experiment_7561_v661_recalibration_prototype.py"),
        Path("python/carnot/experiment_7575_v662_cached_learning_protocol.py"),
    ):
        path = root / relative
        if path.is_file():
            source_hashes.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7578-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private_root)
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    affected_passed = _receipts_pass(affected, validation_scope.REQUIRED_CHECK_NAMES)
    progress(started, "affected_validation", "after_subprocesses", passed=affected_passed)

    candidate = build_artifact(
        root=root,
        sidecars=sidecars,
        bootstrap_rows=bootstrap["rows"],
        replays=replays,
        lifecycle_controls=lifecycle,
        validation_receipts=affected,
        preconditions_checked=checks,
        source_artifact_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    if not affected_passed:
        progress(started, "publish", "before_atomic_disqualified_affected")
        atomic_json(destination, candidate)
        progress(started, "publish", "complete_disqualified_affected")
        return candidate
    validate_artifact(candidate, root=root, require_terminal=False)
    measured_candidate = raw_root / "measured_terminal_candidate.json"
    atomic_json(measured_candidate, candidate)

    provisional_commands = terminal_commands(measured_candidate, root, allow_preterminal=True)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        commands=len(provisional_commands),
    )
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        provisional_commands,
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        root=root,
        sidecars=sidecars,
        bootstrap_rows=bootstrap["rows"],
        replays=replays,
        lifecycle_controls=lifecycle,
        validation_receipts=[*affected, *terminal],
        preconditions_checked=checks,
        source_artifact_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    if not terminal_passed:
        progress(started, "publish", "before_atomic_disqualified_terminal")
        atomic_json(destination, final)
        progress(started, "publish", "complete_disqualified_terminal")
        return final
    validate_artifact(final, root=root, require_terminal=True)
    exact_candidate = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_candidate, final)

    exact_commands = terminal_commands(exact_candidate, root)
    progress(started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        exact_commands,
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(started, "exact_candidate_validation", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(started, "publish", "before_atomic_terminal", path=str(destination))
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "complete",
        learning_measurement_complete_score=final["learning_measurement_complete_score"],
        exploratory_learning_benefit_score=final["exploratory_learning_benefit_score"],
        retention_pass_score=final["retention_pass_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer and read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--allow-preterminal", action="store_true")
    parser.add_argument("--expected-replays", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the producer or one strict serialized-evidence reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        if args.allow_preterminal:
            reduction = independent_reduce(
                args.cold_replay, root=root, expected_replays=args.expected_replays
            )
        else:
            reduction = cold_replay(
                args.cold_replay, root=root, expected_replays=args.expected_replays
            )
        print(json.dumps({"event": "cold_replay_passed", **reduction}, sort_keys=True), flush=True)
        return int(not reduction["measurement_complete"])
    if args.independent_reduce is not None:
        reduction = independent_reduce(
            args.independent_reduce, root=root, expected_replays=args.expected_replays
        )
        print(
            json.dumps({"event": "independent_reduction_passed", **reduction}, sort_keys=True),
            flush=True,
        )
        return int(not reduction["measurement_complete"])
    result = run_experiment(root, args.date, output_path=args.output)
    return int(result["verdict_class"] in {"blocked", "disqualified", "partial"})
