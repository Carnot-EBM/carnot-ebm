"""Measure delayed count learning on the authenticated V660 stream.

The learner updates small Beta count tables after delayed labels arrive. It
loads no language model. The module keeps prediction, feedback, persistence,
and evaluation separate so a later label cannot rewrite an earlier forecast.

Spec refs: REQ-CL-7549 and SCENARIO-CL-7549-*.
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
import platform
import random
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.experiment_7504_v657_evidence_interface import load_json, load_jsonl, write_jsonl
from carnot.experiment_7534_v659_count_memory import (
    CountArm,
    CountConfig,
    CountEventMachine,
    canonical_hash,
    normalized_probability,
    sha256_file,
)
from carnot import experiment_7547_v660_count_stream as stream
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7549-count-learning"
SCHEMA = "carnot.exp7549.v660.count_learning.v1"
RESULT_PATH = Path("results/experiment_7549_v660_count_learning.json")
RAW_DIR = Path("results/raw/experiment_7549_v660_count_learning")
MODULE_PATH = Path("python/carnot/experiment_7549_v660_count_learning.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7549_v660_count_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7549_v660_count_learning.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
UPSTREAM_PATH = Path("results/experiment_7547_v660_count_stream.json")
COUNT_PATH = Path("results/experiment_7534_v659_count_memory.json")

ORDER_SEEDS = stream.ORDER_SEEDS
ARMS = ("frozen", "global_count", "local_count", "shuffled_local")
MACHINE_ARMS = {
    "frozen": "frozen",
    "global_count": "global",
    "local_count": "local",
    "shuffled_local": "permuted_local",
}
COMPARATORS = ("frozen", "global_count", "shuffled_local")
RETENTION_CHECKPOINTS = (0, 40, 80, 120, 159)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 7549011
MINIMUM_SOURCES = 128
MINIMUM_PER_LABEL = 12
MINIMUM_BRIER_DELTA = -0.005
MAX_RETENTION_DETERIORATION = 0.01

ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def brier(probability: float, label: int) -> float:
    """Return one proper-loss contribution for an already sealed forecast."""

    return (float(probability) - int(label)) ** 2


def log_loss(probability: float, label: int) -> float:
    """Use a metric-only clip so exact zero never creates an infinite log."""

    value = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return -(int(label) * math.log(value) + (1 - int(label)) * math.log1p(-value))


def typed_decision(probability: float, label: int) -> JsonDict:
    """Choose the registered action and expose expected and realized costs."""

    value = float(probability)
    expected = {"accept": 5.0 * value, "reject": 1.0 - value, "escalate": 0.2}
    minimum = min(expected.values())
    tied = [name for name, cost in expected.items() if math.isclose(cost, minimum, abs_tol=1e-12)]
    action = "escalate" if "escalate" in tied else tied[0]
    realized = {"accept": 5.0 * int(label), "reject": 1.0 - int(label), "escalate": 0.2}
    return {
        "action": action,
        "expected_costs": expected,
        "expected_cost": expected[action],
        "realized_cost": realized[action],
        "tie_break": "escalation_wins",
    }


def _config(value: Mapping[str, Any]) -> CountConfig:
    """Rebuild only the immutable settings sealed by the upstream protocol."""

    return CountConfig(
        tuple(float(item) for item in value["bin_means"]),
        float(value["global_mean"]),
        float(value["kappa"]),
    )


def _counts_hash(machine: CountEventMachine) -> str:
    """Hash only mutable count state so journal growth stays separately visible."""

    return canonical_hash({name: arm.to_payload() for name, arm in sorted(machine.arms.items())})


def _evaluate_retention(
    machine: CountEventMachine,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    arrivals: int,
) -> tuple[list[JsonDict], int]:
    """Score held-out rows through arm copies without returning their labels."""

    before = machine.state_hash()
    output: list[JsonDict] = []
    for source in rows:
        label = int(source["label"])
        baseline = float(source["base_probability"])
        for arm_name, machine_name in MACHINE_ARMS.items():
            prediction = machine.arms[machine_name].predict(baseline)
            probability = prediction.probability
            output.append(
                {
                    "seed": seed,
                    "arrival_checkpoint": arrivals,
                    "group_id": str(source["group_id"]),
                    "source_hash": str(source["source_hash"]),
                    "source_family": str(source.get("source_family") or "unknown"),
                    "arm": arm_name,
                    "base_probability": baseline,
                    "probability": probability,
                    "energies": list(prediction.energies),
                    "normalized_probability": normalized_probability(prediction.energies),
                    "label": label,
                    "brier": brier(probability, label),
                    "log_loss": log_loss(probability, label),
                    "decision": typed_decision(probability, label),
                    "state_hash_before_evaluation": before,
                    "label_returned_to_learner": False,
                    "disposition": "isolated_retention_evaluation",
                }
            )
    return output, int(machine.state_hash() != before)


def _blocks(order: Sequence[str]) -> list[JsonDict]:
    """Use the upstream frozen release schedule without reading labels."""

    output: list[JsonDict] = []
    stream_end = len(order) - 1
    for block_id, start in enumerate(range(0, len(order), stream.BLOCK_SIZE)):
        event_ids = list(order[start : start + stream.BLOCK_SIZE])
        end = start + len(event_ids) - 1
        release_time = end + stream.FEEDBACK_DELAY
        output.append(
            {
                "block_id": block_id,
                "start_time": start,
                "end_time": end,
                "release_time": release_time,
                "event_ids": event_ids,
                "short_final_block": len(event_ids) < stream.BLOCK_SIZE,
                "release_within_stream": release_time <= stream_end,
            }
        )
    return output


def measure_order(
    online_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    retention_sources: Sequence[Mapping[str, Any]],
    order: Sequence[str],
    count_config: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_dir: Path,
    retention_checkpoints: Sequence[int] = RETENTION_CHECKPOINTS,
) -> JsonDict:
    """Replay one order with a continuous twin and a persisted working state."""

    by_group = {str(row.get("group_id")): row for row in online_rows}
    if (
        len(by_group) != len(online_rows)
        or len(order) != len(by_group)
        or set(order) != set(by_group)
        or set(labels) != set(by_group)
    ):
        raise ValueError("measurement_roster_mismatch")
    config = _config(count_config)
    machine = CountEventMachine.create(config)
    continuous = CountEventMachine.create(config)
    blocks = _blocks(order)
    due_by_time = {int(row["release_time"]): row for row in blocks if row["release_within_stream"]}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[JsonDict] = []
    retention: list[JsonDict] = []
    releases: list[JsonDict] = []
    persistence: list[JsonDict] = []
    update_seconds = 0.0
    persistence_seconds = 0.0
    update_count = 0
    chronology_violations = 0
    restart_mismatches = 0
    duplicate_rejections = 0
    retention_mutations = 0
    changed_bindings = 0

    if 0 in retention_checkpoints:
        evaluated, changed = _evaluate_retention(machine, retention_sources, seed=seed, arrivals=0)
        retention.extend(evaluated)
        retention_mutations += changed

    row_indexes: dict[str, list[int]] = defaultdict(list)
    for prediction_time, event_id in enumerate(order):
        source = by_group[event_id]
        baseline = float(source["base_probability"])
        state_before = machine.state_hash()
        counts_before = _counts_hash(machine)
        working_predictions = machine.predict(
            event_id, baseline, release_index=prediction_time // stream.BLOCK_SIZE
        )
        twin_predictions = continuous.predict(
            event_id, baseline, release_index=prediction_time // stream.BLOCK_SIZE
        )
        if {name: dict(value) for name, value in working_predictions.items()} != {
            name: dict(value) for name, value in twin_predictions.items()
        }:
            restart_mismatches += 1
        label = int(labels[event_id])
        for arm_name, machine_name in MACHINE_ARMS.items():
            value = dict(working_predictions[machine_name])
            probability = float(value["probability"])
            row_indexes[event_id].append(len(predictions))
            predictions.append(
                {
                    "seed": seed,
                    "prediction_time": prediction_time,
                    "event_id": event_id,
                    "source_hash": str(source["source_hash"]),
                    "source_family": str(source.get("source_family") or "unknown"),
                    "arm": arm_name,
                    "base_probability": baseline,
                    "probability": probability,
                    "energies": [float(item) for item in value["energies"]],
                    "normalized_probability": normalized_probability(value["energies"]),
                    "bin_index": int(value["bin_index"]),
                    "label": label,
                    "label_available_at_prediction": False,
                    "feedback_update_id": None,
                    "feedback_availability_time": None,
                    "arm_update_applied": False,
                    "counts_hash_before_prediction": counts_before,
                    "state_hash_before_prediction": state_before,
                    "counts_hash_after_feedback": None,
                    "state_hash_after_feedback": None,
                    "brier": brier(probability, label),
                    "log_loss": log_loss(probability, label),
                    "decision": typed_decision(probability, label),
                    "disposition": "predicted_before_feedback",
                }
            )

        due = due_by_time.get(prediction_time)
        if due is not None:
            event_ids = [str(item) for item in due["event_ids"]]
            if any(not row_indexes[event] for event in event_ids):
                chronology_violations += 1
            feedback = [(event, int(labels[event])) for event in event_ids]
            update_started = time.perf_counter()
            receipt = machine.release(int(due["block_id"]), feedback)
            twin_receipt = continuous.release(int(due["block_id"]), feedback)
            machine.acknowledge(int(due["block_id"]))
            continuous.acknowledge(int(due["block_id"]))
            update_seconds += time.perf_counter() - update_started
            update_count += len(event_ids) * 3
            update_id = f"seed-{seed}-release-{int(due['block_id']):03d}"
            permutation = [dict(row) for row in receipt["permutation"]]
            changed = sum(
                int(int(labels[str(row["event_id"])]) != int(row["label"])) for row in permutation
            )
            changed_bindings += changed

            persist_started = time.perf_counter()
            path = checkpoint_dir / f"seed-{seed}-block-{int(due['block_id']):03d}.json"
            machine.save(path)
            reloaded = CountEventMachine.load(path)
            persistence_seconds += time.perf_counter() - persist_started
            payload_equal = machine.to_payload() == reloaded.to_payload()
            uninterrupted_equal = continuous.to_payload() == reloaded.to_payload()
            if not payload_equal or not uninterrupted_equal or receipt != twin_receipt:
                restart_mismatches += 1
            machine = reloaded
            post_state = machine.state_hash()
            post_counts = _counts_hash(machine)

            duplicate_probe = CountEventMachine.from_payload(machine.to_payload())
            try:
                first_id = event_ids[0]
                duplicate_probe.arms["local"].update(
                    first_id, float(by_group[first_id]["base_probability"]), labels[first_id]
                )
            except ValueError as exc:
                duplicate_rejections += int(str(exc).startswith("duplicate_feedback:"))

            for released_id in event_ids:
                for index in row_indexes[released_id]:
                    row = predictions[index]
                    row["feedback_update_id"] = update_id
                    row["feedback_availability_time"] = prediction_time
                    row["arm_update_applied"] = row["arm"] != "frozen"
                    row["counts_hash_after_feedback"] = post_counts
                    row["state_hash_after_feedback"] = post_state
                    row["disposition"] = "released_after_prediction"
            releases.append(
                {
                    **deepcopy(due),
                    "seed": seed,
                    "update_id": update_id,
                    "label_multiset": sorted(int(label) for _event, label in feedback),
                    "permuted_label_multiset": sorted(
                        int(item) for item in receipt["permuted_labels"]
                    ),
                    "source_to_label_permutation": permutation,
                    "changed_label_binding_count": changed,
                    "matched_feedback_budget": {name: len(event_ids) for name in ARMS},
                    "mutable_update_count": len(event_ids) * 3,
                    "disposition": "released_updated_persisted_reloaded",
                }
            )
            persistence.append(
                {
                    "seed": seed,
                    "block_id": int(due["block_id"]),
                    "checkpoint_sha256": sha256_file(path),
                    "state_hash": post_state,
                    "payload_equal_after_reload": payload_equal,
                    "uninterrupted_state_equal": uninterrupted_equal,
                    "duplicate_feedback_rejected": duplicate_rejections == len(persistence) + 1,
                    "disposition": "exact_restart_checked",
                }
            )

        arrivals = prediction_time + 1
        if arrivals in retention_checkpoints:
            evaluated, changed = _evaluate_retention(
                machine, retention_sources, seed=seed, arrivals=arrivals
            )
            retention.extend(evaluated)
            retention_mutations += changed

    released = {str(event_id) for release in releases for event_id in release["event_ids"]}
    for block in blocks:
        if block["release_within_stream"]:
            continue
        releases.append(
            {
                **deepcopy(block),
                "seed": seed,
                "update_id": None,
                "label_multiset": None,
                "permuted_label_multiset": None,
                "source_to_label_permutation": [],
                "changed_label_binding_count": 0,
                "matched_feedback_budget": {name: 0 for name in ARMS},
                "mutable_update_count": 0,
                "disposition": "censored_end_of_stream",
            }
        )
    for row in predictions:
        if row["event_id"] not in released:
            row["disposition"] = "predicted_feedback_censored_end_of_stream"
    return {
        "seed": seed,
        "prediction_update_rows": predictions,
        "retention_rows": retention,
        "release_rows": sorted(releases, key=lambda row: int(row["block_id"])),
        "persistence_receipts": persistence,
        "released_source_count": len(released),
        "censored_source_count": len(order) - len(released),
        "changed_label_binding_count": changed_bindings,
        "chronology_violation_count": chronology_violations,
        "restart_mismatch_count": restart_mismatches,
        "duplicate_feedback_rejection_count": duplicate_rejections,
        "retention_state_mutation_count": retention_mutations,
        "update_timing": {
            "completed_updates": update_count,
            "duration_s": update_seconds,
            "mean_update_s": update_seconds / update_count if update_count else None,
        },
        "persistence_timing": {
            "completed_persists": len(persistence),
            "duration_s": persistence_seconds,
            "mean_persist_s": persistence_seconds / len(persistence) if persistence else None,
        },
        "final_state_hash": machine.state_hash(),
        "disposition": "complete",
    }


def _bootstrap_order(
    sampled: Sequence[str],
    by_group: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, int],
    registered_order: Sequence[str],
    config: CountConfig,
    retention_sources: Sequence[Mapping[str, Any]],
    retention_sample: Sequence[int],
    checkpoints: Sequence[int],
) -> tuple[dict[str, float], list[float], list[float]]:
    """Replay one multiplicity-preserving source draw in registered order."""

    multiplicity = Counter(sampled)
    expanded: list[tuple[str, str]] = []
    for source_id in registered_order:
        for copy_index in range(multiplicity[source_id]):
            expanded.append((f"{source_id}#bootstrap-{copy_index}", source_id))
    arms = {
        "frozen": CountArm.create("frozen", config),
        "global_count": CountArm.create("global", config),
        "local_count": CountArm.create("local", config),
        "shuffled_local": CountArm.create("permuted_local", config),
    }
    losses: dict[str, list[float]] = {name: [] for name in ARMS}
    retention_brier: list[float] = []
    retention_cost: list[float] = []
    block_by_release = {
        int(row["release_time"]): row
        for row in _blocks([clone for clone, _source in expanded])
        if row["release_within_stream"]
    }

    def score_retention() -> None:
        local_brier: list[float] = []
        frozen_brier: list[float] = []
        local_cost: list[float] = []
        frozen_cost: list[float] = []
        for index in retention_sample:
            source = retention_sources[index]
            baseline = float(source["base_probability"])
            label = int(source["label"])
            for name, briers, costs in (
                ("local_count", local_brier, local_cost),
                ("frozen", frozen_brier, frozen_cost),
            ):
                probability = arms[name].predict(baseline).probability
                briers.append(brier(probability, label))
                costs.append(float(typed_decision(probability, label)["realized_cost"]))
        retention_brier.append(
            math.fsum(local_brier) / len(local_brier) - math.fsum(frozen_brier) / len(frozen_brier)
        )
        retention_cost.append(
            math.fsum(local_cost) / len(local_cost) - math.fsum(frozen_cost) / len(frozen_cost)
        )

    if 0 in checkpoints:
        score_retention()
    predictions: dict[str, tuple[str, float]] = {}
    for prediction_time, (clone_id, source_id) in enumerate(expanded):
        baseline = float(by_group[source_id]["base_probability"])
        label = int(labels[source_id])
        predictions[clone_id] = (source_id, baseline)
        for name in ARMS:
            losses[name].append(brier(arms[name].predict(baseline).probability, label))
        due = block_by_release.get(prediction_time)
        if due is not None:
            due_ids = [str(item) for item in due["event_ids"]]
            due_labels = [int(labels[predictions[item][0]]) for item in due_ids]
            rotated = due_labels[1:] + due_labels[:1]
            for position, clone in enumerate(due_ids):
                baseline_due = predictions[clone][1]
                arms["global_count"].update(clone, baseline_due, due_labels[position])
                arms["local_count"].update(clone, baseline_due, due_labels[position])
                arms["shuffled_local"].update(clone, baseline_due, rotated[position])
        if prediction_time + 1 in checkpoints:
            score_retention()
    means = {name: math.fsum(values) / len(values) for name, values in losses.items()}
    return means, retention_brier, retention_cost


def source_cluster_bootstrap(
    online_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    retention_sources: Sequence[Mapping[str, Any]],
    orders: Mapping[int, Sequence[str]],
    count_config: Mapping[str, Any],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    retention_checkpoints: Sequence[int] = RETENTION_CHECKPOINTS,
    progress_hook: Callable[[int], None] | None = None,
) -> list[JsonDict]:
    """Resample source clusters and replay every arm through every order."""

    by_group = {str(row["group_id"]): row for row in online_rows}
    source_ids = list(by_group)
    if not source_ids or set(labels) != set(source_ids):
        raise ValueError("bootstrap_roster_mismatch")
    if any(
        set(order) != set(source_ids) or len(order) != len(source_ids) for order in orders.values()
    ):
        raise ValueError("bootstrap_order_roster_mismatch")
    config = _config(count_config)
    rng = random.Random(bootstrap_seed)
    output: list[JsonDict] = []
    for replicate in range(replicates):
        sampled = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        retention_sample = [
            rng.randrange(len(retention_sources)) for _ in range(len(retention_sources))
        ]
        order_losses: list[dict[str, float]] = []
        retention_brier: list[float] = []
        retention_cost: list[float] = []
        for registered_order in orders.values():
            means, brier_deltas, cost_deltas = _bootstrap_order(
                sampled,
                by_group,
                labels,
                registered_order,
                config,
                retention_sources,
                retention_sample,
                retention_checkpoints,
            )
            order_losses.append(means)
            retention_brier.extend(brier_deltas)
            retention_cost.extend(cost_deltas)
        arm_means = {
            name: math.fsum(row[name] for row in order_losses) / len(order_losses) for name in ARMS
        }
        output.append(
            {
                "replicate": replicate,
                "sampled_source_count": len(sampled),
                "unique_source_count": len(set(sampled)),
                "source_multiplicity_hash": canonical_hash(sorted(Counter(sampled).items())),
                "order_count": len(orders),
                "chronology_replayed": True,
                "online_delta_brier": {
                    comparator: arm_means["local_count"] - arm_means[comparator]
                    for comparator in COMPARATORS
                },
                "retention_brier_deterioration": max(retention_brier),
                "retention_cost_deterioration": max(retention_cost),
                "disposition": "complete_source_cluster_replay",
            }
        )
        if progress_hook is not None and (
            (replicate + 1) % 100 == 0 or replicate + 1 == replicates
        ):
            progress_hook(replicate + 1)
    return output


def _upper(values: Sequence[float], confidence: float) -> float | None:
    """Return the conservative observed quantile for a finite replay sample."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(confidence * len(ordered)) - 1))
    return ordered[index]


def _holm_contrasts(
    means: Mapping[str, float], bootstrap_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply the frozen three-control Holm family to bootstrap replays."""

    base: list[JsonDict] = []
    for comparator in COMPARATORS:
        draws = [float(row["online_delta_brier"][comparator]) for row in bootstrap_rows]
        p_value = (1 + sum(value >= 0.0 for value in draws)) / (len(draws) + 1)
        base.append(
            {
                "comparator": comparator,
                "mean_delta": float(means[comparator]),
                "upper95_delta": _upper(draws, 0.95),
                "one_sided_p": p_value,
                "bootstrap_replicates": len(draws),
            }
        )
    ranked = sorted(enumerate(base), key=lambda item: float(item[1]["one_sided_p"]))
    adjusted = [1.0] * len(base)
    simultaneous: list[float | None] = [None] * len(base)
    running = 0.0
    for rank, (index, row) in enumerate(ranked):
        remaining = len(base) - rank
        running = max(running, min(1.0, remaining * float(row["one_sided_p"])))
        adjusted[index] = running
        draws = [float(item["online_delta_brier"][row["comparator"]]) for item in bootstrap_rows]
        simultaneous[index] = _upper(draws, 1.0 - 0.05 / remaining)
    for index, row in enumerate(base):
        row["holm_adjusted_p"] = adjusted[index]
        row["simultaneous_upper95_delta"] = simultaneous[index]
        row["holm_passed"] = adjusted[index] < 0.05
    return base


def reduce_measurement(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    replay_summaries: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    label_counts: Mapping[str, int],
    expected_sources: int = 159,
) -> JsonDict:
    """Reduce support, effect, retention, and lifecycle gates independently."""

    arm_losses: dict[str, list[float]] = {name: [] for name in ARMS}
    source_ids: set[str] = set()
    for row in prediction_rows:
        arm = str(row.get("arm"))
        if arm in arm_losses and row.get("disposition") in {
            "complete",
            "released_after_prediction",
            "predicted_feedback_censored_end_of_stream",
        }:
            arm_losses[arm].append(float(row["brier"]))
            if arm == "local_count":
                source_ids.add(str(row["event_id"]))
    means = {
        name: math.fsum(values) / len(values) if values else math.nan
        for name, values in arm_losses.items()
    }
    deltas = {comparator: means["local_count"] - means[comparator] for comparator in COMPARATORS}
    contrasts = _holm_contrasts(deltas, bootstrap_rows) if bootstrap_rows else []
    retention_brier_draws = [float(row["retention_brier_deterioration"]) for row in bootstrap_rows]
    retention_cost_draws = [float(row["retention_cost_deterioration"]) for row in bootstrap_rows]
    retention_brier_upper = _upper(retention_brier_draws, 0.95)
    retention_cost_upper = _upper(retention_cost_draws, 0.95)
    support_passed = bool(
        len(source_ids) >= MINIMUM_SOURCES
        and int(label_counts.get("0", 0)) >= MINIMUM_PER_LABEL
        and int(label_counts.get("1", 0)) >= MINIMUM_PER_LABEL
    )
    schedule_passed = bool(
        len(replay_summaries) == len(ORDER_SEEDS)
        and {int(row["seed"]) for row in replay_summaries} == set(ORDER_SEEDS)
        and len(source_ids) == expected_sources
    )
    control_passed = bool(
        schedule_passed
        and all(int(row.get("changed_label_binding_count", 0)) >= 40 for row in replay_summaries)
    )
    chronology_passed = bool(
        schedule_passed
        and all(int(row.get("chronology_violation_count", -1)) == 0 for row in replay_summaries)
    )
    restart_passed = bool(
        schedule_passed
        and all(
            int(row.get("restart_mismatch_count", -1)) == 0
            and int(row.get("duplicate_feedback_rejection_count", 0)) > 0
            for row in replay_summaries
        )
    )
    retention_passed = bool(
        retention_rows
        and retention_brier_upper is not None
        and retention_cost_upper is not None
        and retention_brier_upper <= MAX_RETENTION_DETERIORATION
        and retention_cost_upper <= MAX_RETENTION_DETERIORATION
    )
    uncertainty_passed = bool(
        len(bootstrap_rows) == BOOTSTRAP_REPLICATES
        and len(contrasts) == len(COMPARATORS)
        and all(
            row["simultaneous_upper95_delta"] is not None
            and float(row["simultaneous_upper95_delta"]) < 0.0
            and row["holm_passed"] is True
            for row in contrasts
        )
    )
    effect_passed = bool(
        support_passed
        and control_passed
        and chronology_passed
        and restart_passed
        and retention_passed
        and uncertainty_passed
        and all(delta <= MINIMUM_BRIER_DELTA for delta in deltas.values())
    )
    measurement_complete = bool(
        schedule_passed
        and control_passed
        and chronology_passed
        and restart_passed
        and len(prediction_rows) == expected_sources * len(ORDER_SEEDS) * len(ARMS)
        and len(bootstrap_rows) == BOOTSTRAP_REPLICATES
    )
    return {
        "independent_unit": "online_source_group",
        "complete_online_source_count": len(source_ids),
        "label_counts": {"0": int(label_counts.get("0", 0)), "1": int(label_counts.get("1", 0))},
        "absolute_brier": means,
        "mean_delta_brier": deltas,
        "primary_contrasts": contrasts,
        "retention_brier_upper95_deterioration": retention_brier_upper,
        "retention_cost_upper95_deterioration": retention_cost_upper,
        "support_passed": support_passed,
        "schedule_passed": schedule_passed,
        "control_passed": control_passed,
        "chronology_passed": chronology_passed,
        "restart_passed": restart_passed,
        "retention_passed": retention_passed,
        "uncertainty_passed": uncertainty_passed,
        "effect_passed": effect_passed,
        "measurement_complete": measurement_complete,
    }


REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7534_v659_count_memory.py"),
    Path("python/carnot/experiment_7509_v657_causal_online.py"),
    Path("python/carnot/experiment_7510_v657_causal_audit.py"),
    Path("python/carnot/experiment_7547_v660_count_stream.py"),
    COUNT_PATH,
    UPSTREAM_PATH,
    SPEC_PATH,
    DESIGN_PATH,
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
    passed: bool,
    *,
    required: bool = True,
) -> JsonDict:
    """Record exact prerequisite operands before dependent measurements start."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "eq" if not isinstance(expected, list) else "in",
        "passed": passed,
        "required": required,
    }


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and make malformed external bytes fail closed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Authenticate exact inputs, upstream gates, sidecars, specification, and CPU."""

    checks: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        exists = (root / relative).is_file()
        checks.append(
            _precondition(
                f"required_path:{relative.as_posix()}",
                "worktree",
                relative.as_posix(),
                "exists",
                True,
                exists,
                exists,
            )
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "requirement_registered",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-CL-7549",
            True,
            "REQ-CL-7549" in spec_text,
            "REQ-CL-7549" in spec_text,
        )
    )
    upstream = _load_object(root / UPSTREAM_PATH)
    upstream_checks = (
        (
            "exp7547_cached_stream_ready",
            "cached_stream_ready_score",
            1,
            upstream.get("cached_stream_ready_score") == 1,
        ),
        (
            "exp7547_verdict_class",
            "verdict_class",
            ["positive", "null", "circular_positive"],
            upstream.get("verdict_class") in {"positive", "null", "circular_positive"},
        ),
        (
            "exp7547_not_adversarial",
            "flagged_adversarial",
            False,
            upstream.get("flagged_adversarial") is False,
        ),
    )
    for check, field, expected, passed in upstream_checks:
        checks.append(
            _precondition(
                check,
                "exp7547-count-stream",
                UPSTREAM_PATH.as_posix(),
                field,
                expected,
                upstream.get(field),
                passed,
            )
        )
    for name, receipt in (upstream.get("raw_sidecars") or {}).items():
        relative = Path(str(receipt.get("path") or ""))
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        expected = receipt.get("sha256")
        checks.append(
            _precondition(
                f"exp7547_sidecar:{name}",
                "exp7547-count-stream",
                relative.as_posix(),
                "sha256",
                expected,
                observed,
                observed == expected,
            )
        )
    cpu_count = os.cpu_count() or 0
    checks.append(
        _precondition(
            "host_cpu_available",
            "current_host",
            "/proc/cpuinfo",
            "logical_cpu_count",
            ">=1",
            cpu_count,
            cpu_count >= 1,
        )
    )
    return checks


def load_measurement_inputs(root: Path) -> JsonDict:
    """Rebuild public custody and open labels only after protocol authentication."""

    upstream = _load_object(root / UPSTREAM_PATH)
    stream.validate_artifact(upstream, require_terminal=True, verify_sidecars=True, root=root)
    protocol_path = root / str(upstream["raw_sidecars"]["frozen_protocol"]["path"])
    frozen = load_json(protocol_path)
    public = stream.load_public_role_rows(root)
    rebuilt = stream.freeze_public_protocol(public)
    if rebuilt != frozen:
        raise ValueError("frozen_protocol_rebuild_mismatch")
    labels = stream.load_private_labels(root, public)
    retention_path = root / str(upstream["raw_sidecars"]["retention_rows"]["path"])
    retention = load_jsonl(retention_path)
    if len(retention) != 116:
        raise ValueError("retention_roster_mismatch")
    return {
        "upstream": upstream,
        "frozen": frozen,
        "public": public,
        "labels": labels,
        "retention": retention,
    }


def _sidecar(path: Path, root: Path, *, rows: int) -> JsonDict:
    """Bind a large row set by relative path, exact bytes, hash, and count."""

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def write_measurement_sidecars(
    root: Path,
    predictions: Sequence[Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
    releases: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    persistence: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Publish raw current evidence before a compact terminal record refers to it."""

    raw = root / RAW_DIR
    paths = {
        "prediction_update_rows": raw / "prediction_update_rows.jsonl",
        "retention_rows": raw / "retention_rows.jsonl",
        "release_rows": raw / "release_rows.jsonl",
        "bootstrap_rows": raw / "bootstrap_rows.jsonl",
        "persistence_rows": raw / "persistence_rows.jsonl",
    }
    values: dict[str, Sequence[Mapping[str, Any]]] = {
        "prediction_update_rows": predictions,
        "retention_rows": retention,
        "release_rows": releases,
        "bootstrap_rows": bootstrap,
        "persistence_rows": persistence,
    }
    output: dict[str, JsonDict] = {}
    for name, path in paths.items():
        write_jsonl(path, values[name])
        output[name] = _sidecar(path, root, rows=len(values[name]))
    return output


def _read_sidecar(root: Path, receipt: Mapping[str, Any], name: str) -> list[JsonDict]:
    """Rehash a raw row file before an independent reader trusts its contents."""

    path = root / str(receipt.get("path") or "")
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"raw_sidecar_hash_mismatch:{name}")
    if path.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"raw_sidecar_size_mismatch:{name}")
    rows = load_jsonl(path)
    if len(rows) != receipt.get("rows"):
        raise ValueError(f"raw_sidecar_row_count_mismatch:{name}")
    return rows


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful non-timeout receipt for every named command."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set(names) <= passed


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute every headline from authenticated raw rows and command receipts."""

    prediction_rows = _read_sidecar(
        root, value.get("prediction_update_rows") or {}, "prediction_update_rows"
    )
    retention_rows = _read_sidecar(root, value.get("retention_rows") or {}, "retention_rows")
    bootstrap_rows = _read_sidecar(root, value.get("bootstrap_rows") or {}, "bootstrap_rows")
    raw = reduce_measurement(
        prediction_rows=prediction_rows,
        retention_rows=retention_rows,
        replay_summaries=value.get("replay_summaries") or [],
        bootstrap_rows=bootstrap_rows,
        label_counts=value.get("label_counts") or {},
        expected_sources=int(value.get("expected_online_sources", 159)),
    )
    receipts = value.get("validation_receipts") or []
    affected = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    terminal_present = any(row.get("name") in TERMINAL_CHECK_NAMES for row in receipts)
    raw["affected_validation_passed"] = affected
    raw["terminal_validation_passed"] = terminal
    raw["terminal_validation_present"] = terminal_present
    raw["measurement_complete"] = bool(
        raw["measurement_complete"] and affected and (terminal or not terminal_present)
    )
    return raw


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep raw operands beside each acceptance decision."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Expose validity, readiness, support, and benefit as separate decisions."""

    contrasts = {row["comparator"]: row for row in reduction["primary_contrasts"]}
    gates = [
        _gate(
            "affected_validation",
            "validity",
            True,
            reduction["affected_validation_passed"],
            "eq",
            bool(reduction["affected_validation_passed"]),
            "Invalid scoped checks cannot support science.",
        ),
        _gate(
            "terminal_validation",
            "validity",
            True,
            reduction["terminal_validation_passed"],
            "eq",
            bool(reduction["terminal_validation_passed"]),
            "Fresh readers must accept the exact candidate.",
        ),
        _gate(
            "frozen_schedule_complete",
            "readiness",
            True,
            reduction["schedule_passed"],
            "eq",
            bool(reduction["schedule_passed"]),
            "Five orders cannot become extra independent sources.",
        ),
        _gate(
            "chronology",
            "validity",
            0,
            not reduction["chronology_passed"],
            "eq",
            bool(reduction["chronology_passed"]),
            "Later labels cannot author earlier predictions.",
        ),
        _gate(
            "restart_parity",
            "validity",
            True,
            reduction["restart_passed"],
            "eq",
            bool(reduction["restart_passed"]),
            "Persistence must not change count learning.",
        ),
        _gate(
            "source_support",
            "support",
            {"sources": MINIMUM_SOURCES, "per_label": MINIMUM_PER_LABEL},
            {
                "sources": reduction["complete_online_source_count"],
                "labels": reduction["label_counts"],
            },
            "gte",
            bool(reduction["support_passed"]),
            "Sparse evidence cannot become a benefit claim.",
        ),
        _gate(
            "shuffled_bindings",
            "support",
            40,
            reduction["control_passed"],
            "gte_each_order",
            bool(reduction["control_passed"]),
            "A label-preserving shuffle is not a control.",
        ),
    ]
    for comparator in COMPARATORS:
        row = contrasts.get(comparator, {})
        passed = bool(
            row
            and float(row["mean_delta"]) <= MINIMUM_BRIER_DELTA
            and float(row["simultaneous_upper95_delta"]) < 0.0
            and row["holm_passed"] is True
        )
        gates.append(
            _gate(
                f"brier_benefit_vs_{comparator}",
                "benefit",
                {
                    "mean_delta_lte": MINIMUM_BRIER_DELTA,
                    "simultaneous_upper95_lt": 0.0,
                    "holm_alpha": 0.05,
                },
                row or None,
                "all",
                passed,
                "A noisy or too-small proper-loss change cannot become promotion.",
            )
        )
    gates.extend(
        [
            _gate(
                "retention_brier",
                "benefit",
                MAX_RETENTION_DETERIORATION,
                reduction["retention_brier_upper95_deterioration"],
                "lte",
                bool(reduction["retention_passed"]),
                "Learning must preserve held-out calibration.",
            ),
            _gate(
                "retention_primary_cost",
                "benefit",
                MAX_RETENTION_DETERIORATION,
                reduction["retention_cost_upper95_deterioration"],
                "lte",
                bool(reduction["retention_passed"]),
                "Learning must preserve held-out decisions.",
            ),
            _gate(
                "confirmatory_benefit",
                "benefit",
                0,
                0,
                "eq",
                True,
                "Prior corpus inspection forbids a confirmatory claim.",
            ),
        ]
    )
    return gates


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed gate without hiding a valid completed null."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_checks": [str(row["check"]) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the evidence or claim error prevented by every stored field."""

    specific = {
        "experiment_id": "Binds evidence to the exact task and milestone.",
        "preconditions_checked": "Prevents absent inputs from becoming fabricated measurements.",
        "MODEL_SPECS": "Makes the no-model plan explicit.",
        "model_specs": "Keeps both required model manifests empty and consistent.",
        "model_invoked": "Separates current work from historical cached inference.",
        "inference_substrate_class": "Prevents CPU aggregation from being reported as model work.",
        "inference_substrate": "Names upstream-artifact aggregation as the actual substrate.",
        "execution_venue": "Uses the legal host enum instead of an invented device name.",
        "duration_s": "Keeps current monotonic work separate from historical time.",
        "random_seed": "Prevents outcome-aware order or bootstrap tuning.",
        "reproducibility_checksum": "Binds code, settings, source hashes, and raw rows.",
        "rows": "Keeps absolute arm metrics visible beside comparative effects.",
        "sample_size_budget": "Distinguishes completed, censored, failed, and unstarted units.",
        "acceptance_gate_results": "Keeps validity, readiness, support, and benefit independent.",
        "gate_check_summary": "Names exact failed operands instead of a vague status.",
        "honest_verdict": "Uses one complete terminal disposition.",
        "verdict_class": "Restricts terminal classification to the closed vocabulary.",
        "verifier_is_oracle": "Prevents probabilistic evidence from becoming a correctness proof.",
        "flagged_adversarial": "Preserves adverse findings instead of clearing a gate.",
        "validation_receipts": "Stores exact command scope, exit, log hash, and cold readers.",
        "field_principles": "Makes every artifact field state the failure it prevents.",
        "count_measurement_complete_score": "Keeps auditable completion separate from effect.",
        "exploratory_effect_score": "Requires every registered empirical effect gate.",
        "confirmatory_benefit_score": "Remains zero for the previously inspected corpus.",
        "continuous_self_learning_task": "Identifies real across-event count updates.",
        "prediction_update_rows": "Preserves pre-update forecasts and causal state hashes.",
        "retention_rows": "Keeps held-out calibration and actions visible.",
        "restart_parity_score": "Requires exact resumed predictions and exactly-once feedback.",
    }
    return {
        field: specific.get(field, "Preserves this field for independent drift detection.")
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash every terminal field except the checksum that contains itself."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def build_artifact(
    *,
    root: Path,
    sidecars: Mapping[str, Mapping[str, Any]],
    replay_summaries: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    label_counts: Mapping[str, int],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    expected_sources: int = 159,
) -> JsonDict:
    """Assemble one compact terminal record from independently readable rows."""

    scaffold: JsonDict = {
        "prediction_update_rows": dict(sidecars["prediction_update_rows"]),
        "retention_rows": dict(sidecars["retention_rows"]),
        "bootstrap_rows": dict(sidecars["bootstrap_rows"]),
        "replay_summaries": [deepcopy(dict(row)) for row in replay_summaries],
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "expected_online_sources": expected_sources,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    reduction = independent_reduce(scaffold, root=root)
    required_failure = any(
        row.get("name") in {*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
        and row.get("passed") is not True
        for row in validation_receipts
    )
    complete = bool(reduction["measurement_complete"])
    effect = bool(reduction["effect_passed"] and complete)
    if required_failure or not complete:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_count_learning_validation_or_measurement_invalid"
    elif effect:
        verdict_class = "circular_positive"
        honest_verdict = "complete_circular_positive_exploratory_count_learning_effect"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_count_learning_valid_benefit_gate_failed"
    gates = _acceptance_gates(reduction)
    arm_rows = [
        {
            "unit_id": f"arm-{arm}",
            "arm": arm,
            "absolute_metrics": {
                "mean_brier": reduction["absolute_brier"][arm],
                "prediction_count": int(sidecars["prediction_update_rows"]["rows"]) // len(ARMS),
                "independent_source_groups": reduction["complete_online_source_count"],
            },
            "delta_brier_vs_local": (
                0.0 if arm == "local_count" else -float(reduction["mean_delta_brier"][arm])
            ),
            "disposition": "complete_comparative_arm",
            "failed": False,
            "censored": False,
        }
        for arm in ARMS
    ]
    completed_feedback = sum(int(row["released_source_count"]) for row in replay_summaries)
    censored_feedback = sum(int(row["censored_source_count"]) for row in replay_summaries)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Delayed count learning and retained calibrated decisions",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {"source": UPSTREAM_PATH.as_posix(), "counted_as_current": False},
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "compute_details": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "cwd": str(root.resolve())},
        "random_seed": {
            "order_seeds": list(ORDER_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "source_artifact_hashes": dict(source_hashes),
        "frozen_settings": {
            "block_size": 8,
            "feedback_delay": 8,
            "prior_mass": 8,
            "clip": 1e-4,
            "retention_checkpoints": list(RETENTION_CHECKPOINTS),
            "holm_family_alpha": 0.05,
        },
        **scaffold,
        "release_rows": dict(sidecars["release_rows"]),
        "persistence_rows": dict(sidecars["persistence_rows"]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "independent_reduction": reduction,
        "rows": arm_rows,
        "sample_size_budget": {
            "independent_online_groups": {
                "planned": expected_sources,
                "attempted": expected_sources,
                "completed": reduction["complete_online_source_count"],
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": max(0, expected_sources - reduction["complete_online_source_count"]),
            },
            "ordered_event_instances": {
                "planned": expected_sources * len(ORDER_SEEDS),
                "attempted": expected_sources * len(ORDER_SEEDS),
                "completed": expected_sources * len(ORDER_SEEDS),
                "excluded": 0,
                "failed": 0,
                "censored": censored_feedback,
                "unstarted": 0,
            },
            "feedback_updates": {
                "planned": expected_sources * len(ORDER_SEEDS),
                "attempted": completed_feedback,
                "completed": completed_feedback,
                "excluded": 0,
                "failed": 0,
                "censored": censored_feedback,
                "unstarted": 0,
            },
            "retention_groups": {
                "planned": 116,
                "attempted": 116,
                "completed": 116,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": scaffold["validation_receipts"],
        "repository_health": {
            "unrelated_debt_required_for_this_result": False,
            "unscoped_suite_required": False,
            "unscoped_python_suite": {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "interrupted_after_confirmed_failures": True,
                "observed_before_stop": {
                    "passed": 5291,
                    "failed": 26,
                    "skipped": 6,
                    "collection_errors": 67,
                },
                "representative_unrelated_debt": [
                    "missing unsloth/Qwen3.6-35B-A3B-GGUF registry entry",
                    "stale ARC submission dataset configuration",
                ],
            },
        },
        "capability_e2e": {
            "workflow": "predict_release_update_persist_reload",
            "orders_exercised": len(replay_summaries),
            "passed": bool(
                replay_summaries
                and all(
                    int(row["released_source_count"]) > 0
                    and int(row["restart_mismatch_count"]) == 0
                    and int(row["duplicate_feedback_rejection_count"]) > 0
                    for row in replay_summaries
                )
            ),
            "private_llm_off_real_environment_smoke": "this_authenticated_cached_count_replay",
            "numbered_runtime_e2e": {
                "applicable": [],
                "reason": "No shared sampler, binding, ARC transport, or telemetry changed.",
            },
        },
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "terminal_status": "complete",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
        "no_headroom": False,
        "count_measurement_complete_score": int(complete),
        "exploratory_effect_score": int(effect),
        "confirmatory_benefit_score": 0,
        "continuous_self_learning_task": True,
        "restart_parity_score": int(reduction["restart_passed"]),
        "prior_result": {
            "experiment_id": "exp7509-causal-online",
            "honest_verdict": "complete_null_causal_online_measurement_valid_benefit_gate_failed",
        },
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(
    root: Path,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    effect: bool = False,
) -> JsonDict:
    """Build compact deterministic evidence for mutation and fresh-reader tests."""

    source_count = 159
    predictions = [
        {
            "seed": seed,
            "event_id": f"fixture-{index:03d}",
            "arm": arm,
            "label": index % 2,
            "brier": 0.100 if arm == "local_count" else (0.200 if effect else 0.101),
            "disposition": "complete",
        }
        for seed in ORDER_SEEDS
        for index in range(source_count)
        for arm in ARMS
    ]
    retention = [{"group_id": "fixture-retention", "disposition": "complete"}]
    releases = [{"seed": seed, "disposition": "complete"} for seed in ORDER_SEEDS]
    persistence = [{"seed": seed, "disposition": "complete"} for seed in ORDER_SEEDS]
    bootstrap = [
        {
            "replicate": index,
            "online_delta_brier": {name: -0.100 if effect else -0.001 for name in COMPARATORS},
            "retention_brier_deterioration": 0.0,
            "retention_cost_deterioration": 0.0,
            "disposition": "complete_source_cluster_replay",
        }
        for index in range(BOOTSTRAP_REPLICATES)
    ]
    sidecars = write_measurement_sidecars(
        root, predictions, retention, releases, bootstrap, persistence
    )
    summaries = [
        {
            "seed": seed,
            "released_source_count": 144,
            "censored_source_count": 15,
            "changed_label_binding_count": 50,
            "chronology_violation_count": 0,
            "restart_mismatch_count": 0,
            "duplicate_feedback_rejection_count": 18,
            "retention_state_mutation_count": 0,
            "disposition": "complete",
        }
        for seed in ORDER_SEEDS
    ]
    return build_artifact(
        root=root,
        sidecars=sidecars,
        replay_summaries=summaries,
        validation_receipts=validation_receipts,
        preconditions_checked=[],
        source_hashes={"private_fixture": "sha256:" + "b" * 64},
        label_counts={"0": 80, "1": 79},
        duration_s=0.1,
        phase_spans=[],
    )


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    verify_sidecars: bool = True,
    require_terminal: bool = True,
) -> JsonDict:
    """Reject changed identity, raw rows, scores, receipts, principles, or checksum."""

    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        raise ValueError("artifact_date_or_milestone_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("model_specs_must_be_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        raise ValueError("current_model_invocation_mismatch")
    if value.get("inference_substrate_class") != "no_model_load":
        raise ValueError("inference_substrate_class_mismatch")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        raise ValueError("inference_substrate_mismatch")
    if value.get("execution_venue") != "host":
        raise ValueError("execution_venue_mismatch")
    if value.get("confirmatory_benefit_score") != 0:
        raise ValueError("confirmatory_benefit_score_mismatch")
    if value.get("positive_claim") is not False:
        raise ValueError("positive_claim_mismatch")
    if verify_sidecars:
        for name in (
            "prediction_update_rows",
            "retention_rows",
            "release_rows",
            "bootstrap_rows",
            "persistence_rows",
        ):
            _read_sidecar(root, value.get(name) or {}, name)
    reduction = independent_reduce(value, root=root)
    if reduction != value.get("independent_reduction"):
        raise ValueError("independent_reduction_mismatch")
    if value.get("count_measurement_complete_score") != int(reduction["measurement_complete"]):
        raise ValueError("measurement_complete_score_mismatch")
    if value.get("exploratory_effect_score") != int(
        reduction["effect_passed"] and reduction["measurement_complete"]
    ):
        raise ValueError("exploratory_effect_score_mismatch")
    if value.get("restart_parity_score") != int(reduction["restart_passed"]):
        raise ValueError("restart_parity_score_mismatch")
    required_failure = any(
        row.get("name") in {*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
        and row.get("passed") is not True
        for row in value.get("validation_receipts") or []
    )
    if required_failure:
        raise ValueError("required_validation_failed")
    if require_terminal and not reduction["terminal_validation_passed"]:
        raise ValueError("terminal_validation_missing_or_failed")
    expected_verdict = (
        "circular_positive"
        if reduction["measurement_complete"] and reduction["effect_passed"]
        else "null"
        if reduction["measurement_complete"]
        else "disqualified"
    )
    if value.get("verdict_class") != expected_verdict:
        raise ValueError("verdict_class_mismatch")
    principles = value.get("field_principles") or {}
    if any(field not in principles for field in value):
        raise ValueError("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("reproducibility_checksum_mismatch")
    return reduction


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Load serialized evidence through a strict fresh-reader compatible API."""

    value = _load_object(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(value, root=root, verify_sidecars=True, require_terminal=False)


def build_blocked_artifact(
    failed: Mapping[str, Any], checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Publish external absence once without inventing dependent measurement rows."""

    reason = "".join(
        character if character.isalnum() else "_" for character in str(failed["check"])
    ).strip("_")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Delayed count learning and retained calibrated decisions",
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {"source": UPSTREAM_PATH.as_posix(), "counted_as_current": False},
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "compute_details": {"machine": platform.machine(), "logical_cpu_count": os.cpu_count()},
        "duration_s": float(duration_s),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "cwd": str(REPO_ROOT)},
        "random_seed": {"order_seeds": list(ORDER_SEEDS), "bootstrap_seed": BOOTSTRAP_SEED},
        "source_artifact_hashes": {},
        "rows": [],
        "prediction_update_rows": [],
        "retention_rows": [],
        "release_rows": [],
        "bootstrap_rows": [],
        "persistence_rows": [],
        "replay_summaries": [],
        "label_counts": {},
        "sample_size_budget": {
            "independent_online_groups": {
                "planned": 159,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 159,
            },
            "retention_groups": {
                "planned": 116,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 116,
            },
        },
        "acceptance_gate_results": [
            _gate(
                str(failed["check"]),
                "validity",
                failed.get("expected"),
                failed.get("observed"),
                str(failed.get("op") or "eq"),
                False,
                "Missing external evidence cannot become a measurement.",
            )
        ],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [str(failed["check"])],
            "first_failure": deepcopy(dict(failed)),
        },
        "validation_receipts": [],
        "repository_health": {"unrelated_debt_required_for_this_result": False},
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "terminal_status": "complete",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
        "no_headroom": False,
        "count_measurement_complete_score": 0,
        "exploratory_effect_score": 0,
        "confirmatory_benefit_score": 0,
        "continuous_self_learning_task": True,
        "restart_parity_score": 0,
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def progress(  # pragma: no cover - visible only in the declared entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase and slow-operation boundary with monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7549] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - current monotonic time is an entrypoint boundary.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    """Close one disjoint current-work interval with its completed units."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


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
        coverage_file=private_root / ".coverage.exp7549",
    )


def terminal_commands(
    candidate: Path, root: Path = REPO_ROOT
) -> list[validation_scope.CommandSpec]:
    """Build fresh replay, reduction, adversarial, and strict row readers."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--root",
                str(root),
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
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


def _write_affected_manifest(path: Path) -> None:
    """Freeze exact files before the scoped runner expands any command."""

    atomic_json(
        path,
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )


def _source_hashes(root: Path) -> dict[str, str]:
    """Bind every present required input without treating a path as evidence."""

    return {
        path.as_posix(): sha256_file(root / path)
        for path in REQUIRED_INPUT_PATHS
        if (root / path).is_file()
    }


def _replay_summary(value: Mapping[str, Any]) -> JsonDict:
    """Keep causal counts and timing while moving per-event rows to sidecars."""

    keys = (
        "seed",
        "released_source_count",
        "censored_source_count",
        "changed_label_binding_count",
        "chronology_violation_count",
        "restart_mismatch_count",
        "duplicate_feedback_rejection_count",
        "retention_state_mutation_count",
        "update_timing",
        "persistence_timing",
        "final_state_hash",
        "disposition",
    )
    return {key: deepcopy(value[key]) for key in keys}


def run_experiment(  # pragma: no cover - the declared entrypoint is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, measure, validate in fresh processes, then publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next(
        (row for row in checks if row.get("required") is True and row.get("passed") is not True),
        None,
    )
    if failed is not None:
        blocked = build_blocked_artifact(failed, checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "complete", completed_units=len(checks))

    manifest_path = raw_root / "affected_validation_manifest.json"
    _write_affected_manifest(manifest_path)
    source_hashes = _source_hashes(root)
    source_hashes[manifest_path.relative_to(root).as_posix()] = sha256_file(manifest_path)

    progress(started, "input_custody", "start_before_label_open")
    phase_started = time.monotonic()
    inputs = load_measurement_inputs(root)
    frozen = inputs["frozen"]
    public = inputs["public"]
    labels = inputs["labels"]
    retention_sources = inputs["retention"]
    spans.append(_span("input_custody", phase_started, started, len(public["online"])))
    progress(
        started,
        "input_custody",
        "complete_after_label_open",
        protocol_hash=frozen["protocol_hash"],
    )

    checkpoint_root = Path(tempfile.mkdtemp(prefix="carnot-exp7549-checkpoints-", dir="/tmp"))
    replays: list[JsonDict] = []
    progress(started, "count_learning", "start", orders=len(ORDER_SEEDS))
    phase_started = time.monotonic()
    for completed, seed in enumerate(ORDER_SEEDS, 1):
        progress(started, "count_learning", "before_order", seed=seed)
        replay = measure_order(
            public["online"],
            labels["online"],
            retention_sources,
            frozen["orders"][str(seed)],
            frozen["count_config"],
            seed=seed,
            checkpoint_dir=checkpoint_root / str(seed),
        )
        replays.append(replay)
        progress(
            started,
            "count_learning",
            "after_order",
            completed_units=completed,
            changed_bindings=replay["changed_label_binding_count"],
            seed=seed,
        )
    spans.append(_span("count_learning", phase_started, started, len(replays)))

    orders = {seed: frozen["orders"][str(seed)] for seed in ORDER_SEEDS}
    progress(
        started, "source_cluster_bootstrap", "before_benchmark", replicates=BOOTSTRAP_REPLICATES
    )
    phase_started = time.monotonic()
    bootstrap = source_cluster_bootstrap(
        public["online"],
        labels["online"],
        retention_sources,
        orders,
        frozen["count_config"],
        progress_hook=lambda completed: progress(
            started, "source_cluster_bootstrap", "units_complete", completed_units=completed
        ),
    )
    spans.append(_span("source_cluster_bootstrap", phase_started, started, len(bootstrap)))
    progress(started, "source_cluster_bootstrap", "after_benchmark", completed_units=len(bootstrap))

    predictions = [row for replay in replays for row in replay["prediction_update_rows"]]
    retention = [row for replay in replays for row in replay["retention_rows"]]
    releases = [row for replay in replays for row in replay["release_rows"]]
    persistence = [row for replay in replays for row in replay["persistence_receipts"]]
    progress(started, "sidecars", "before_atomic_writes")
    phase_started = time.monotonic()
    sidecars = write_measurement_sidecars(
        root, predictions, retention, releases, bootstrap, persistence
    )
    spans.append(_span("sidecars", phase_started, started, len(sidecars)))
    progress(started, "sidecars", "after_atomic_writes", completed_units=len(sidecars))

    summaries = [_replay_summary(replay) for replay in replays]
    label_counts = Counter(str(label) for label in labels["online"].values())
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7549-validation-", dir="/tmp"))
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
        replay_summaries=summaries,
        validation_receipts=affected,
        preconditions_checked=checks,
        source_hashes=source_hashes,
        label_counts=label_counts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    if affected_passed:
        validate_artifact(candidate, root=root, verify_sidecars=True, require_terminal=False)
    else:
        progress(started, "publish", "before_atomic_disqualified_affected")
        atomic_json(destination, candidate)
        progress(started, "publish", "complete_disqualified_affected")
        return candidate
    measured_path = raw_root / "measured_terminal_candidate.json"
    atomic_json(measured_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        terminal_commands(measured_path, root),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        root=root,
        sidecars=sidecars,
        replay_summaries=summaries,
        validation_receipts=[*affected, *terminal],
        preconditions_checked=checks,
        source_hashes=source_hashes,
        label_counts=label_counts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    if not terminal_passed:
        progress(started, "publish", "before_atomic_disqualified_terminal")
        atomic_json(destination, final)
        progress(started, "publish", "complete_disqualified_terminal")
        return final
    validate_artifact(final, root=root, verify_sidecars=True, require_terminal=True)
    exact_path = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_path, final)

    progress(started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        terminal_commands(exact_path, root),
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(started, "exact_candidate_validation", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "complete",
        count_measurement_complete_score=final["count_measurement_complete_score"],
        exploratory_effect_score=final["exploratory_effect_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the producer or one strict serialized-evidence reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        reduction = cold_replay(args.cold_replay, root=root)
        print(json.dumps({"event": "cold_replay_passed", **reduction}, sort_keys=True), flush=True)
        return int(not reduction["measurement_complete"])
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        if not value:
            raise ValueError("artifact_unreadable_or_not_object")
        reduction = validate_artifact(
            value, root=root, verify_sidecars=True, require_terminal=False
        )
        print(
            json.dumps({"event": "independent_reduction_passed", **reduction}, sort_keys=True),
            flush=True,
        )
        return int(not reduction["measurement_complete"])
    artifact = run_experiment(root, args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "result": str(args.output or root / RESULT_PATH),
                "count_measurement_complete_score": artifact["count_measurement_complete_score"],
                "exploratory_effect_score": artifact["exploratory_effect_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(artifact["verdict_class"] in {"blocked", "disqualified", "partial"})


if __name__ == "__main__":  # pragma: no cover - wrapper is the public executable.
    raise SystemExit(main())
