"""Measure causal online adaptation on sealed V657 human-label roles.

Predictions use immutable historical Qwen readouts. This run loads no language
model. It updates only small CPU heads after audited labels become available.

Spec ref: REQ-CL-7509 and SCENARIO-CL-7509-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import socket
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.experiment_7504_v657_evidence_interface import load_jsonl
from carnot.experiment_7506_v657_causal_prototype import (
    BoundedResidualHead,
    permute_released_batch,
    select_fixture_hyperparameters,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7509-v657-causal-online"
SCHEMA = "carnot.exp7509.v657.causal_online.v1"
CHECKPOINT_SCHEMA = "carnot.exp7509.v657.online_checkpoint.v1"
RESULT_PATH = Path("results/experiment_7509_v657_causal_online.json")
RAW_DIR = Path("results/raw/experiment_7509_v657_causal_online")
MODULE_PATH = Path("python/carnot/experiment_7509_v657_causal_online.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7509_v657_causal_online.py")
TEST_PATH = Path("tests/python/test_experiment_7509_v657_causal_online.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

FEATURE_PATH = Path("results/raw/experiment_7504_v657_evidence_interface/features.jsonl")
NORMALIZATION_PATH = Path(
    "results/raw/experiment_7504_v657_evidence_interface/training_normalization.json"
)
PREDICTOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/predictors.jsonl")
EVALUATOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/evaluators.jsonl")
SETTINGS_PATH = Path("results/raw/experiment_7505_v657_energy_fit/frozen_checkpoints.json")
UPSTREAM_ARTIFACTS = (
    Path("results/experiment_7504_v657_evidence_interface.json"),
    Path("results/experiment_7505_v657_energy_fit.json"),
    Path("results/experiment_7506_v657_causal_prototype.json"),
)

SCHEDULE_SEEDS = (656201, 656202, 656203, 656204, 656205)
DELAYS = (8, 0)
PRIMARY_DELAY = 8
BLOCK_SIZE = 8
AUDIT_PROBABILITY = 0.25
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH = 16
BOOTSTRAP_SEED = 657009
NUMERIC_BUDGET_S = 1200.0
ARMS = (
    "frozen_base",
    "intercept_brier",
    "affine_brier",
    "local_brier",
    "shuffled_local_brier",
    "local_log_loss",
    "zero_step_local_brier",
)
PRIMARY_COMPARATORS = (
    "frozen_base",
    "intercept_brier",
    "affine_brier",
    "shuffled_local_brier",
    "local_log_loss",
)
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def protocol() -> JsonDict:
    """Return the complete frozen schedule and statistical contract."""

    return {
        "schedule_seeds": list(SCHEDULE_SEEDS),
        "delays": list(DELAYS),
        "primary_delay": PRIMARY_DELAY,
        "block_size": BLOCK_SIZE,
        "audit_probability": AUDIT_PROBABILITY,
        "arms": list(ARMS),
        "primary_contrasts": list(PRIMARY_COMPARATORS),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "seed": BOOTSTRAP_SEED,
            "sensitivity_block_lengths": [8, 32],
        },
        "minimum_sources": 120,
        "minimum_delivered_audit_labels": 20,
        "minimum_permutable_labels": 12,
        "minimum_frozen_delta": -0.01,
        "retention_upper95_max": 0.01,
        "static_benefit_gate": False,
    }


def _stable_unit(seed: int, *parts: str) -> float:
    """Map public identity fields to a stable value in the unit interval."""

    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def build_arrival_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[JsonDict]:
    """Create frozen drift blocks without reading outcomes or probabilities."""

    families: dict[str, list[JsonDict]] = defaultdict(list)
    identities: set[str] = set()
    for source in rows:
        identity = str(source.get("group_id") or "")
        family = str(source.get("source_family") or "")
        if not identity or not family or identity in identities:
            raise ValueError("public_identity_invalid")
        identities.add(identity)
        public = deepcopy(dict(source))
        public.pop("label", None)
        public.pop("secret_label", None)
        families[family].append(public)
    family_order = sorted(families, key=lambda name: (_stable_unit(seed, "family", name), name))
    output: list[JsonDict] = []
    for family in family_order:
        output.extend(
            sorted(
                families[family],
                key=lambda row: (
                    _stable_unit(seed, "member", family, str(row["group_id"])),
                    str(row["group_id"]),
                ),
            )
        )
    return output


def build_audit_mask(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, bool]:
    """Select audits from identity and seed without consulting score or label."""

    return {
        str(row["group_id"]): _stable_unit(seed, "audit", str(row["group_id"])) < AUDIT_PROBABILITY
        for row in rows
    }


def batch_permutation(labels: Sequence[int], event_ids: Sequence[str], *, seed: int) -> JsonDict:
    """Wrap the qualified Exp7506 batch-local permutation with an identity."""

    value = permute_released_batch(labels, event_ids, seed=seed)
    value["permutation_id"] = canonical_hash(
        {"seed": seed, "event_ids": list(event_ids), "result": value}
    )
    return value


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and make malformed external bytes fail closed."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _temperature_probability(probability: float, temperature: float) -> float:
    """Apply the frozen Exp7505 scalar temperature to one native probability."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    logit = math.log(clipped / (1.0 - clipped)) / float(temperature)
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def load_frozen_settings(root: Path) -> JsonDict:
    """Load the registered temperature and Exp7506 fixture-selected head budget."""

    checkpoint = _load_object(root / SETTINGS_PATH)
    temperature = checkpoint.get("temperature_baseline", {}).get("selected_temperature")
    if temperature != 1.5:
        raise ValueError("frozen_temperature_invalid")
    selected = select_fixture_hyperparameters()
    return {
        "temperature": float(temperature),
        "learning_rate": float(selected["selected_learning_rate"]),
        "residual_bound": float(selected["selected_residual_bound"]),
        "selection_receipt": selected,
    }


def load_public_role_rows(root: Path) -> dict[str, list[JsonDict]]:
    """Load four normalized public features while keeping evaluator labels closed."""

    features = load_jsonl(root / FEATURE_PATH)
    predictors = {row["group_id"]: row for row in load_jsonl(root / PREDICTOR_PATH)}
    normalization = _load_object(root / NORMALIZATION_PATH)
    means = np.asarray(normalization.get("mean"), dtype=np.float64)[:4]
    scales = np.asarray(normalization.get("safe_scale"), dtype=np.float64)[:4]
    if means.shape != (4,) or scales.shape != (4,) or np.any(scales <= 0.0):
        raise ValueError("normalization_invalid")
    settings = load_frozen_settings(root)
    output: dict[str, list[JsonDict]] = {
        "training": [],
        "calibration_tuning": [],
        "online": [],
        "test": [],
    }
    for source in features:
        role = str(source.get("role") or "")
        predictor = predictors.get(source.get("group_id"))
        values = np.asarray(source.get("features"), dtype=np.float64)
        if role not in output or predictor is None or values.shape != (10,):
            raise ValueError("public_role_row_invalid")
        row = {
            "group_id": str(source["group_id"]),
            "source_family": str(predictor.get("source_family") or "unknown"),
            "role": role,
            "features": ((values[:4] - means) / scales).tolist(),
            "base_probability": _temperature_probability(
                float(source["raw_whole_expectation"]), settings["temperature"]
            ),
            "source_hash": str(source["source_hash"]),
            "response_hash": str(source["response_hash"]),
        }
        output[role].append(row)
    expected = {"training": 176, "calibration_tuning": 60, "online": 159, "test": 116}
    if {role: len(rows) for role, rows in output.items()} != expected:
        raise ValueError("role_counts_invalid")
    return output


def load_private_labels(root: Path, *, roles: Sequence[str]) -> dict[str, dict[str, int]]:
    """Open only named evaluator roles after the public prediction plan exists."""

    allowed = set(roles)
    if not allowed or not allowed <= {"online", "test"}:
        raise ValueError("private_label_roles_invalid")
    public = load_public_role_rows(root)
    eligible = {role: {str(row["group_id"]) for row in public[role]} for role in roles}
    output = {role: {} for role in roles}
    for row in load_jsonl(root / EVALUATOR_PATH):
        role = str(row.get("role") or "")
        if role in allowed and str(row.get("group_id")) in eligible[role]:
            label = row.get("label")
            if label not in {0, 1}:
                raise ValueError("private_label_invalid")
            output[role][str(row["group_id"])] = 1 - int(label)
    if any(set(output[role]) != eligible[role] for role in roles):
        raise ValueError("private_label_identity_mismatch")
    return output


def _create_heads(
    training_rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, BoundedResidualHead]:
    """Create all six mutable or zero-step heads from training inputs only."""

    training = np.asarray([row["features"] for row in training_rows], dtype=np.float64)
    rate = float(settings["learning_rate"])
    bound = float(settings["residual_bound"])
    declarations = {
        "intercept_brier": ("intercept", "brier", rate),
        "affine_brier": ("affine", "brier", rate),
        "local_brier": ("local", "brier", rate),
        "shuffled_local_brier": ("local", "brier", rate),
        "local_log_loss": ("local", "log_loss", rate),
        "zero_step_local_brier": ("local", "brier", 0.0),
    }
    return {
        arm: BoundedResidualHead.from_training(
            training,
            learning_rate=learning_rate,
            residual_bound=bound,
            loss=loss,
            basis_kind=basis,
        )
        for arm, (basis, loss, learning_rate) in declarations.items()
    }


def _head_hashes(heads: Mapping[str, BoundedResidualHead], frozen_hash: str) -> dict[str, str]:
    """Return every arm state hash with the immutable baseline included."""

    return {arm: frozen_hash if arm == "frozen_base" else heads[arm].state_hash for arm in ARMS}


def _predict(
    arm: str, source: Mapping[str, Any], heads: Mapping[str, BoundedResidualHead]
) -> float:
    """Predict one arm without accepting a label argument."""

    base = float(source["base_probability"])
    if arm == "frozen_base":
        return base
    return float(heads[arm].predict(source["features"], base_probability=base)["probability"])


def _brier(probability: float, label: int) -> float:
    """Return one binary Brier contribution."""

    return (float(probability) - int(label)) ** 2


def _log_loss(probability: float, label: int) -> float:
    """Return one clipped binary log-loss contribution."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(int(label) * math.log(clipped) + (1 - int(label)) * math.log1p(-clipped))


def _checkpoint_payload(
    heads: Mapping[str, BoundedResidualHead],
    *,
    cursor: int,
    predictions: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    reveal_rows: Sequence[Mapping[str, Any]],
    pending_blocks: Mapping[int, Sequence[str]],
) -> JsonDict:
    """Serialize heads and causal queues without adding unrevealed labels."""

    return {
        "schema": CHECKPOINT_SCHEMA,
        "cursor": int(cursor),
        "heads": {arm: head.to_payload() for arm, head in heads.items()},
        "predictions": deepcopy(list(predictions)),
        "updates": deepcopy(list(updates)),
        "reveal_rows": deepcopy(list(reveal_rows)),
        "pending_blocks": {str(key): list(value) for key, value in pending_blocks.items()},
    }


def _restore_checkpoint(
    value: Mapping[str, Any],
) -> tuple[
    dict[str, BoundedResidualHead],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    dict[int, list[str]],
]:
    """Restore the complete replay state from canonical checkpoint bytes."""

    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint_schema_invalid")
    heads = {
        str(arm): BoundedResidualHead.from_payload(payload)
        for arm, payload in value["heads"].items()
    }
    return (
        heads,
        deepcopy(list(value["predictions"])),
        deepcopy(list(value["updates"])),
        deepcopy(list(value["reveal_rows"])),
        {int(key): list(ids) for key, ids in value["pending_blocks"].items()},
    )


def _release_batch(
    *,
    block_id: int,
    event_ids: Sequence[str],
    labels: Mapping[str, int],
    prediction_by_group: Mapping[str, Mapping[str, Any]],
    heads: Mapping[str, BoundedResidualHead],
    schedule_seed: int,
    delay: int,
    update_time: int,
) -> tuple[list[JsonDict], JsonDict]:
    """Apply one audited batch after its fixed availability boundary."""

    aligned = [int(labels[event_id]) for event_id in event_ids]
    permutation = batch_permutation(
        aligned,
        event_ids,
        seed=int(schedule_seed) + int(block_id) + int(delay) * 1000,
    )
    shuffled_by_event = dict(zip(event_ids, permutation["labels"], strict=True))
    shuffled_origins = dict(zip(event_ids, permutation["label_origins"], strict=True))
    release_event_ids = list(event_ids)
    availability = block_id * BLOCK_SIZE + BLOCK_SIZE - 1 + int(delay)
    if update_time < availability:
        raise ValueError("release_before_availability")
    rows: list[JsonDict] = []
    for event_id in event_ids:
        prediction = prediction_by_group[event_id]
        for arm in ARMS:
            before = (
                prediction["prediction_state_hashes"][arm]
                if arm == "frozen_base"
                else heads[arm].state_hash
            )
            if arm == "frozen_base":
                status = "no_update_control"
                origin = event_id
                used_label = int(labels[event_id])
                after = before
            else:
                shuffled = arm == "shuffled_local_brier"
                used_label = int(shuffled_by_event[event_id]) if shuffled else int(labels[event_id])
                origin = str(shuffled_origins[event_id]) if shuffled else event_id
                receipt = heads[arm].update(
                    prediction["prediction_payload"]["features"],
                    used_label,
                    base_probability=float(prediction["prediction_payload"]["base_probability"]),
                )
                status = str(receipt["status"])
                after = heads[arm].state_hash
            rows.append(
                {
                    "group_id": event_id,
                    "arm": arm,
                    "schedule_seed": int(schedule_seed),
                    "delay": int(delay),
                    "block_id": int(block_id),
                    "prediction_time": int(prediction["prediction_time"]),
                    "availability_time": availability,
                    "update_time": int(update_time),
                    "label": used_label,
                    "label_origin": origin,
                    "release_event_ids": release_event_ids,
                    "permutation_identity": permutation["permutation_id"],
                    "permutation_mode": permutation["mode"],
                    "parameter_hash_before": before,
                    "parameter_hash": after,
                    "status": status,
                    "failed": False,
                    "censored": False,
                }
            )
    reveal = {
        "block_id": int(block_id),
        "availability_time": availability,
        "update_time": int(update_time),
        "event_ids": release_event_ids,
        "delivered_label_count": len(event_ids),
        "permutable_label_count": len(event_ids) if int(permutation["changed"]) > 0 else 0,
        "permutation_mode": permutation["mode"],
        "permutation_identity": permutation["permutation_id"],
        "disposition": "released",
    }
    return rows, reveal


def run_schedule(
    training_rows: Sequence[Mapping[str, Any]],
    online_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    *,
    retention_rows: Sequence[Mapping[str, Any]],
    retention_labels: Mapping[str, int],
    schedule_seed: int,
    delay: int,
    checkpoint_dir: Path,
    restart_after_release_block: int | None = None,
    stop_after_arrival: int | None = None,
) -> JsonDict:
    """Replay one schedule while sealing every probability before release."""

    if delay not in DELAYS:
        raise ValueError("delay_invalid")
    settings = load_frozen_settings(REPO_ROOT)
    heads = _create_heads(training_rows, settings)
    ordered = build_arrival_order(online_rows, schedule_seed)
    mask = build_audit_mask(ordered, schedule_seed)
    expected_ids = {str(row["group_id"]) for row in ordered}
    if set(labels) != expected_ids:
        raise ValueError("online_label_identity_mismatch")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frozen_hash = canonical_hash({"temperature": settings["temperature"], "mutable": False})
    predictions: list[JsonDict] = []
    prediction_by_group: dict[str, JsonDict] = {}
    updates: list[JsonDict] = []
    reveal_rows: list[JsonDict] = []
    pending_blocks: dict[int, list[str]] = defaultdict(list)
    checkpoint_hashes: list[JsonDict] = []
    restarted = False

    for arrival, source in enumerate(ordered):
        if stop_after_arrival is not None and arrival > stop_after_arrival:
            break
        identity = str(source["group_id"])
        state_hashes = _head_hashes(heads, frozen_hash)
        payload = {
            "group_id": identity,
            "source_family": str(source["source_family"]),
            "features": deepcopy(source["features"]),
            "base_probability": float(source["base_probability"]),
            "schedule_seed": int(schedule_seed),
            "delay": int(delay),
            "prediction_time": arrival,
        }
        probabilities = {arm: _predict(arm, source, heads) for arm in ARMS}
        sealed = {
            "group_id": identity,
            "source_family": str(source["source_family"]),
            "schedule_seed": int(schedule_seed),
            "delay": int(delay),
            "prediction_time": arrival,
            "block_id": arrival // BLOCK_SIZE,
            "audit_selected": mask[identity],
            "prediction_payload": payload,
            "prediction_hash": canonical_hash(
                {"payload": payload, "probabilities": probabilities, "states": state_hashes}
            ),
            "prediction_state_hashes": state_hashes,
            "probabilities": probabilities,
        }
        predictions.append(sealed)
        prediction_by_group[identity] = sealed
        pending_blocks.setdefault(arrival // BLOCK_SIZE, [])
        if mask[identity]:
            pending_blocks[arrival // BLOCK_SIZE].append(identity)

        due = [
            block_id
            for block_id in sorted(pending_blocks)
            if block_id * BLOCK_SIZE + BLOCK_SIZE - 1 + delay <= arrival
        ]
        for block_id in due:
            event_ids = pending_blocks.pop(block_id)
            if event_ids:
                new_updates, reveal = _release_batch(
                    block_id=block_id,
                    event_ids=event_ids,
                    labels=labels,
                    prediction_by_group=prediction_by_group,
                    heads=heads,
                    schedule_seed=schedule_seed,
                    delay=delay,
                    update_time=arrival,
                )
            else:
                new_updates = []
                reveal = {
                    "block_id": int(block_id),
                    "availability_time": block_id * BLOCK_SIZE + BLOCK_SIZE - 1 + delay,
                    "update_time": arrival,
                    "event_ids": [],
                    "delivered_label_count": 0,
                    "permutable_label_count": 0,
                    "permutation_mode": "empty_audit_noop",
                    "permutation_identity": canonical_hash(
                        [schedule_seed, delay, block_id, "empty_audit_noop"]
                    ),
                    "disposition": "released",
                }
            updates.extend(new_updates)
            reveal_rows.append(reveal)
            checkpoint = _checkpoint_payload(
                heads,
                cursor=arrival + 1,
                predictions=predictions,
                updates=updates,
                reveal_rows=reveal_rows,
                pending_blocks=pending_blocks,
            )
            checkpoint_path = checkpoint_dir / f"schedule-{schedule_seed}-d{delay}-b{block_id}.json"
            atomic_json(checkpoint_path, checkpoint)
            checkpoint_hashes.append(
                {
                    "block_id": block_id,
                    "path": checkpoint_path.as_posix(),
                    "sha256": sha256_file(checkpoint_path),
                }
            )
            if restart_after_release_block == block_id and not restarted:
                restored = _load_object(checkpoint_path)
                heads, predictions, updates, reveal_rows, restored_pending = _restore_checkpoint(
                    restored
                )
                pending_blocks = defaultdict(list, restored_pending)
                prediction_by_group = {str(row["group_id"]): row for row in predictions}
                restarted = True

    last_arrival = len(predictions) - 1
    for block_id, event_ids in sorted(pending_blocks.items()):
        reveal_rows.append(
            {
                "block_id": int(block_id),
                "availability_time": block_id * BLOCK_SIZE + BLOCK_SIZE - 1 + delay,
                "update_time": None,
                "event_ids": list(event_ids),
                "delivered_label_count": 0,
                "permutable_label_count": 0,
                "permutation_mode": "not_released",
                "permutation_identity": None,
                "disposition": "censored",
            }
        )

    per_source: list[JsonDict] = []
    for sealed in predictions:
        identity = str(sealed["group_id"])
        label = int(labels[identity])
        for arm in ARMS:
            probability = float(sealed["probabilities"][arm])
            per_source.append(
                {
                    "group_id": identity,
                    "source_family": sealed["source_family"],
                    "schedule_seed": int(schedule_seed),
                    "delay": int(delay),
                    "arm": arm,
                    "prediction_time": int(sealed["prediction_time"]),
                    "audit_selected": bool(sealed["audit_selected"]),
                    "prediction_payload": deepcopy(sealed["prediction_payload"]),
                    "prediction_hash": sealed["prediction_hash"],
                    "parameter_hash": sealed["prediction_state_hashes"][arm],
                    "probability": probability,
                    "label": label,
                    "brier_loss": _brier(probability, label),
                    "log_loss": _log_loss(probability, label),
                    "status": "complete",
                    "failed": False,
                    "censored": False,
                }
            )

    retention_state_before = canonical_hash(_head_hashes(heads, frozen_hash))
    retention_output: list[JsonDict] = []
    for source in retention_rows:
        identity = str(source["group_id"])
        if identity not in retention_labels:
            raise ValueError("retention_label_identity_mismatch")
        label = int(retention_labels[identity])
        for arm in ARMS:
            probability = _predict(arm, source, heads)
            retention_output.append(
                {
                    "group_id": identity,
                    "source_family": str(source["source_family"]),
                    "schedule_seed": int(schedule_seed),
                    "delay": int(delay),
                    "arm": arm,
                    "probability": probability,
                    "label": label,
                    "brier_loss": _brier(probability, label),
                    "log_loss": _log_loss(probability, label),
                    "used_for_update": False,
                    "used_for_selection_or_rollback": False,
                    "status": "complete",
                    "failed": False,
                    "censored": False,
                }
            )
    retention_state_after = canonical_hash(_head_hashes(heads, frozen_hash))
    chronology = sum(
        int(row["prediction_time"] > row["update_time"])
        + int(row["availability_time"] > row["update_time"])
        + int(row["label_origin"] not in row["release_event_ids"])
        for row in updates
    )
    return {
        "schedule_seed": int(schedule_seed),
        "delay": int(delay),
        "per_source_results": per_source,
        "per_update_rows": updates,
        "retention_rows": retention_output,
        "reveal_counts_by_batch": reveal_rows,
        "checkpoint_hashes": checkpoint_hashes,
        "final_state_hashes": _head_hashes(heads, frozen_hash),
        "pending_queue": {str(key): list(value) for key, value in pending_blocks.items()},
        "chronology_violations": chronology,
        "censored_feedback_count": sum(len(ids) for ids in pending_blocks.values()),
        "delivered_audit_labels": sum(int(row["delivered_label_count"]) for row in reveal_rows),
        "permutable_labels": sum(int(row["permutable_label_count"]) for row in reveal_rows),
        "source_count": len(predictions),
        "last_arrival": last_arrival,
        "retention_state_hash_before": retention_state_before,
        "retention_state_hash_after": retention_state_after,
        "restart_performed": restarted,
    }


def restart_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    """Compare every prediction, update, queue, retention row, and final state."""

    fields = (
        "per_source_results",
        "per_update_rows",
        "retention_rows",
        "reveal_counts_by_batch",
        "final_state_hashes",
        "pending_queue",
    )
    comparisons = {field: left.get(field) == right.get(field) for field in fields}
    return {"passed": all(comparisons.values()), "comparisons": comparisons}


def _source_mean_pairs(
    rows: Sequence[Mapping[str, Any]], *, comparator: str, delay: int
) -> list[JsonDict]:
    """Average schedule replicates inside each source before inference."""

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    families: dict[str, str] = {}
    order: list[str] = []
    for row in rows:
        if int(row.get("delay", -1)) != delay:
            continue
        arm = str(row.get("arm"))
        if arm not in {"local_brier", comparator}:
            continue
        identity = str(row.get("group_id"))
        if identity not in families:
            families[identity] = str(row.get("source_family"))
        if arm == "local_brier" and identity not in order:
            order.append(identity)
        values[(identity, arm)].append(float(row["brier_loss"]))
    pairs: list[JsonDict] = []
    for identity in order:
        local = values.get((identity, "local_brier"), [])
        control = values.get((identity, comparator), [])
        if local and len(local) == len(control):
            pairs.append(
                {
                    "group_id": identity,
                    "source_family": families[identity],
                    "local_mean": math.fsum(local) / len(local),
                    "comparator_mean": math.fsum(control) / len(control),
                    "delta": math.fsum(local) / len(local) - math.fsum(control) / len(control),
                    "schedule_replicates": len(local),
                }
            )
    return pairs


def _moving_block_means(
    pairs: Sequence[Mapping[str, Any]], *, block_length: int, replicates: int, seed: int
) -> list[float]:
    """Resample circular blocks inside each frozen drift stratum."""

    strata: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        strata[str(row["source_family"])].append(float(row["delta"]))
    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _replicate in range(replicates):
        sampled: list[float] = []
        for family in strata:
            values = strata[family]
            remaining = len(values)
            while remaining > 0:
                start = int(rng.integers(0, len(values)))
                take = min(block_length, remaining)
                sampled.extend(values[(start + offset) % len(values)] for offset in range(take))
                remaining -= take
        means.append(math.fsum(sampled) / len(sampled))
    return means


def reduce_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparator: str,
    delay: int,
    block_length: int,
    replicates: int,
    seed: int,
) -> JsonDict:
    """Reduce one registered paired Brier contrast from source means."""

    pairs = _source_mean_pairs(rows, comparator=comparator, delay=delay)
    if not pairs:
        return {
            "comparator": comparator,
            "delay": delay,
            "source_count": 0,
            "replicate_unit": "schedule_seed_mean_within_source",
            "mean_delta": None,
            "upper95_delta": None,
            "one_sided_p": 1.0,
        }
    draws = _moving_block_means(pairs, block_length=block_length, replicates=replicates, seed=seed)
    upper_index = min(len(draws) - 1, math.ceil(0.95 * len(draws)) - 1)
    upper = sorted(draws)[upper_index]
    p_value = (1 + sum(value >= 0.0 for value in draws)) / (len(draws) + 1)
    return {
        "comparator": comparator,
        "delay": delay,
        "source_count": len(pairs),
        "replicate_unit": "schedule_seed_mean_within_source",
        "mean_delta": math.fsum(float(row["delta"]) for row in pairs) / len(pairs),
        "upper95_delta": upper,
        "one_sided_p": p_value,
        "block_length": block_length,
        "replicates": replicates,
        "bootstrap_seed": seed,
        "drift_strata": list(dict.fromkeys(str(row["source_family"]) for row in pairs)),
    }


def _holm_adjust(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Apply Holm correction to the frozen five-contrast family."""

    ranked = sorted(enumerate(rows), key=lambda item: float(item[1]["one_sided_p"]))
    adjusted = [1.0] * len(rows)
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ranked):
        running = max(running, min(1.0, (count - rank) * float(row["one_sided_p"])))
        adjusted[index] = running
    output = []
    for index, row in enumerate(rows):
        value = deepcopy(dict(row))
        value["holm_adjusted_p"] = adjusted[index]
        value["holm_passed"] = adjusted[index] < 0.05
        output.append(value)
    return output


def reduce_measurement(
    per_source_results: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    stream_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Apply support, primary, sensitivity, and retention gates independently."""

    primary = _holm_adjust(
        [
            reduce_contrast(
                per_source_results,
                comparator=comparator,
                delay=PRIMARY_DELAY,
                block_length=BOOTSTRAP_BLOCK_LENGTH,
                replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
            )
            for comparator in PRIMARY_COMPARATORS
        ]
    )
    sensitivities = [
        reduce_contrast(
            per_source_results,
            comparator="frozen_base",
            delay=delay,
            block_length=length,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        for delay, length in ((8, 8), (8, 32), (0, 16))
    ]
    retention = reduce_contrast(
        retention_rows,
        comparator="frozen_base",
        delay=PRIMARY_DELAY,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    primary_streams = [row for row in stream_rows if row.get("delay") == PRIMARY_DELAY]
    support_rows = [
        {
            "schedule_seed": int(row["schedule_seed"]),
            "source_count": int(row["source_count"]),
            "delivered_audit_labels": int(row["delivered_audit_labels"]),
            "permutable_labels": int(row["permutable_labels"]),
            "passed": int(row["source_count"]) >= 120
            and int(row["delivered_audit_labels"]) >= 20
            and int(row["permutable_labels"]) >= 12,
        }
        for row in primary_streams
    ]
    support_passed = len(support_rows) == len(SCHEDULE_SEEDS) and all(
        row["passed"] for row in support_rows
    )
    frozen = next((row for row in primary if row["comparator"] == "frozen_base"), {})
    primary_passed = (
        support_passed
        and frozen.get("mean_delta") is not None
        and float(frozen["mean_delta"]) <= -0.01
        and all(
            row.get("upper95_delta") is not None
            and float(row["upper95_delta"]) < 0.0
            and row["holm_passed"] is True
            for row in primary
        )
    )
    information = next((row for row in primary if row["comparator"] == "shuffled_local_brier"), {})
    information_passed = (
        support_passed
        and information.get("upper95_delta") is not None
        and float(information["upper95_delta"]) < 0.0
        and information.get("holm_passed") is True
    )
    retention_passed = (
        retention.get("upper95_delta") is not None and float(retention["upper95_delta"]) <= 0.01
    )
    return {
        "support_rows": support_rows,
        "support_passed": support_passed,
        "primary_contrasts": primary,
        "primary_passed": primary_passed,
        "causal_information_passed": information_passed,
        "retention_contrast": retention,
        "retention_passed": retention_passed,
        "sensitivity_contrasts": sensitivities,
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
    Path("python/carnot/experiment_7468_v654_residual_learner.py"),
    Path("python/carnot/experiment_7483_v655_continuous_learning.py"),
    Path("results/experiment_7483_v655_continuous_learning.json"),
    Path("python/carnot/experiment_7491_v656_window_protocol.py"),
    Path("python/carnot/experiment_7504_v657_evidence_interface.py"),
    Path("python/carnot/experiment_7505_v657_energy_fit.py"),
    Path("python/carnot/experiment_7506_v657_causal_prototype.py"),
    FEATURE_PATH,
    NORMALIZATION_PATH,
    PREDICTOR_PATH,
    EVALUATOR_PATH,
    SETTINGS_PATH,
    *UPSTREAM_ARTIFACTS,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any, path: str
) -> JsonDict:
    """Record the exact prerequisite value without repairing an upstream."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "path": path,
        "passed": observed == expected,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate instructions, source bytes, specs, and three V657 producers."""

    rows: list[JsonDict] = []
    hashes: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        observed = path.is_file() and path.stat().st_size > 0
        rows.append(
            _precondition(
                "resource_readable",
                "worktree_or_V657_input",
                "path",
                True,
                observed,
                relative.as_posix(),
            )
        )
        if observed:
            hashes.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "owner": "repository_or_declared_historical_input",
                }
            )
    if not all(row["passed"] for row in rows):
        return rows, hashes
    upstream_expectations = (
        (UPSTREAM_ARTIFACTS[0], "Exp7504", "evidence_ready_score", 1),
        (UPSTREAM_ARTIFACTS[1], "Exp7505", "energy_fit_ready_score", 1),
        (UPSTREAM_ARTIFACTS[2], "Exp7506", "causal_update_ready_score", 1),
    )
    for path, name, field, expected in upstream_expectations:
        value = _load_object(root / path)
        for check, checked_field, wanted in (
            ("upstream_ready", field, expected),
            ("upstream_unflagged", "flagged_adversarial", False),
            ("upstream_no_current_model", "model_invoked", False),
        ):
            rows.append(
                _precondition(
                    check,
                    name,
                    checked_field,
                    wanted,
                    value.get(checked_field),
                    path.as_posix(),
                )
            )
    checkpoint = _load_object(root / SETTINGS_PATH)
    rows.append(
        _precondition(
            "frozen_temperature",
            "Exp7505",
            "temperature_baseline.selected_temperature",
            1.5,
            checkpoint.get("temperature_baseline", {}).get("selected_temperature"),
            SETTINGS_PATH.as_posix(),
        )
    )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        _precondition(
            "requirement_present",
            "OpenSpec",
            "REQ-CL-7509",
            True,
            "REQ-CL-7509" in spec_text,
            SPEC_PATH.as_posix(),
        )
    )
    return rows, hashes


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit one flushed phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7509] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def run_measurement(root: Path, checkpoint_root: Path) -> JsonDict:  # pragma: no cover
    """Run all matched schedules, delays, retention scores, and restarts."""

    started = time.monotonic()
    public = load_public_role_rows(root)
    plans = {
        seed: {
            "order": [row["group_id"] for row in build_arrival_order(public["online"], seed)],
            "audit": build_audit_mask(public["online"], seed),
        }
        for seed in SCHEDULE_SEEDS
    }
    plan_hash = canonical_hash(plans)
    labels = load_private_labels(root, roles=("online", "test"))
    per_source: list[JsonDict] = []
    updates: list[JsonDict] = []
    retention: list[JsonDict] = []
    reveal_rows: list[JsonDict] = []
    streams: list[JsonDict] = []
    checkpoint_hashes: list[JsonDict] = []
    parity_rows: list[JsonDict] = []
    for seed, delay in itertools.product(SCHEDULE_SEEDS, DELAYS):
        progress(started, "numeric_replay", "before_schedule", seed=seed, delay=delay)
        result = run_schedule(
            public["training"],
            public["online"],
            labels["online"],
            retention_rows=public["test"],
            retention_labels=labels["test"],
            schedule_seed=seed,
            delay=delay,
            checkpoint_dir=checkpoint_root / f"seed-{seed}-delay-{delay}",
        )
        per_source.extend(result["per_source_results"])
        updates.extend(result["per_update_rows"])
        retention.extend(result["retention_rows"])
        reveal_rows.extend(
            {**row, "schedule_seed": seed, "delay": delay}
            for row in result["reveal_counts_by_batch"]
        )
        checkpoint_hashes.extend(result["checkpoint_hashes"])
        streams.append(
            {
                "unit_id": f"seed-{seed}-delay-{delay}",
                "schedule_seed": seed,
                "delay": delay,
                "source_count": result["source_count"],
                "delivered_audit_labels": result["delivered_audit_labels"],
                "permutable_labels": result["permutable_labels"],
                "censored_feedback_count": result["censored_feedback_count"],
                "chronology_violations": result["chronology_violations"],
                "status": "complete",
                "failed": False,
                "censored": False,
            }
        )
        if delay == PRIMARY_DELAY:
            restarted = run_schedule(
                public["training"],
                public["online"],
                labels["online"],
                retention_rows=public["test"],
                retention_labels=labels["test"],
                schedule_seed=seed,
                delay=delay,
                checkpoint_dir=checkpoint_root / f"restart-seed-{seed}",
                restart_after_release_block=0,
            )
            parity = restart_parity(result, restarted)
            parity_rows.append({"schedule_seed": seed, **parity})
        progress(
            started,
            "numeric_replay",
            "after_schedule",
            seed=seed,
            delay=delay,
            completed_sources=result["source_count"],
        )
        if time.monotonic() - started > NUMERIC_BUDGET_S:
            raise TimeoutError("numeric_budget_exceeded")
    reduction = reduce_measurement(per_source, retention, streams)
    return {
        "public_plan_hash": plan_hash,
        "per_source_results": per_source,
        "per_update_rows": updates,
        "retention_rows": retention,
        "reveal_counts_by_batch": reveal_rows,
        "rows": streams,
        "checkpoint_hashes": checkpoint_hashes,
        "restart_parity_rows": parity_rows,
        "restart_parity_score": int(all(row["passed"] for row in parity_rows)),
        "chronology_violation_count": sum(row["chronology_violations"] for row in streams),
        "measurement_reduction": reduction,
        "computation_duration_s": time.monotonic() - started,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful non-timeout receipt for every exact command name."""

    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep a raw observed operand beside each acceptance decision."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


def independent_reduce(value: Mapping[str, Any], *, require_validation: bool = True) -> JsonDict:
    """Recompute completion, information value, benefit, and verdict from rows."""

    per_source = value.get("per_source_results")
    retention = value.get("retention_rows")
    streams = value.get("rows")
    updates = value.get("per_update_rows")
    parity = value.get("restart_parity_rows")
    if not all(
        isinstance(rows, list) for rows in (per_source, retention, streams, updates, parity)
    ):
        return {
            "causal_evaluation_complete_score": 0,
            "causal_information_value_score": 0,
            "online_benefit_score": 0,
            "restart_parity_score": 0,
            "verdict_class": "disqualified",
            "errors": ["raw_rows_missing"],
        }
    test_fixture = value.get("test_fixture") is True
    expected_sources = 2 if test_fixture else 159
    expected_retention = 2 if test_fixture else 116
    expected_streams = len(SCHEDULE_SEEDS) * len(DELAYS)
    errors: list[str] = []
    if len(streams) != expected_streams or any(
        int(row.get("source_count", -1)) != expected_sources for row in streams
    ):
        errors.append("stream_completion_failed")
    expected_predictions = expected_sources * len(ARMS) * expected_streams
    if len(per_source) != expected_predictions:
        errors.append("prediction_row_count_failed")
    expected_retention_rows = expected_retention * len(ARMS) * expected_streams
    if len(retention) != expected_retention_rows:
        errors.append("retention_row_count_failed")
    chronology = sum(int(row.get("chronology_violations", 1)) for row in streams)
    if chronology != 0:
        errors.append("chronology_failed")
    if any(
        int(row.get("prediction_time", 1)) > int(row.get("update_time", 0))
        or int(row.get("availability_time", 1)) > int(row.get("update_time", 0))
        or row.get("label_origin") not in row.get("release_event_ids", [])
        for row in updates
    ):
        errors.append("update_origin_failed")
    restart_score = int(
        len(parity) == len(SCHEDULE_SEEDS) and all(row.get("passed") is True for row in parity)
    )
    if restart_score != 1:
        errors.append("restart_parity_failed")
    if any(row.get("used_for_update") is not False for row in retention):
        errors.append("retention_update_leakage")
    if any(row.get("used_for_selection_or_rollback") is not False for row in retention):
        errors.append("retention_control_leakage")
    current_calls_valid = (
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
    )
    if not current_calls_valid:
        errors.append("current_model_calls_nonzero")
    receipts = value.get("validation_receipts")
    required_names = (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    validation_passed = not require_validation or (
        isinstance(receipts, list) and _receipts_pass(receipts, required_names)
    )
    if not validation_passed:
        errors.append("required_validation_failed")
    recomputed = reduce_measurement(per_source, retention, streams)
    complete = int(not errors)
    information = int(complete == 1 and recomputed["causal_information_passed"] is True)
    benefit = int(
        complete == 1
        and information == 1
        and recomputed["primary_passed"] is True
        and recomputed["retention_passed"] is True
        and chronology == 0
        and restart_score == 1
    )
    verdict = "positive" if benefit else ("null" if complete else "disqualified")
    return {
        "causal_evaluation_complete_score": complete,
        "causal_information_value_score": information,
        "online_benefit_score": benefit,
        "restart_parity_score": restart_score,
        "verdict_class": verdict,
        "measurement_reduction": recomputed,
        "errors": errors,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the evidence failure that each terminal field prevents."""

    special = {
        "schema": "Version, experiment, milestone, and terminal status prevent reader drift.",
        "run_date": "The fixed date prevents this measurement from moving between runs.",
        "preconditions_checked": "Exact observed prerequisites prevent invented dependent evidence.",
        "MODEL_SPECS": "An empty list prevents historical Qwen evidence from becoming a current load.",
        "model_specs": "The schema mirror prevents aliases from hiding a model load.",
        "invocation_counts": "Balanced zero counters expose attempted or unfinished model work.",
        "per_update_rows": "Origins and times make every causal state transition reconstructible.",
        "per_source_results": "Group-level probabilities prevent aggregate-only benefit claims.",
        "retention_rows": "Evaluation-only labels expose forgetting without controlling the learner.",
        "restart_parity_score": "A bare score prevents partial state recovery from passing restart.",
        "causal_evaluation_complete_score": "Completion stays independent of favorable benefit.",
        "causal_information_value_score": "Only the aligned-versus-legal-shuffle gate gets causal credit.",
        "online_benefit_score": "A full conjunction prevents one favorable contrast from claiming benefit.",
    }
    return {
        field: special.get(
            field, f"The {field} field prevents silent omission or reinterpretation."
        )
        for field in fields
    }


def _reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind all terminal evidence except the self-referential checksum."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def build_artifact(
    measurement: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    require_validation: bool = True,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    started_at_utc: str | None = None,
    duration_s: float = 1e-6,
    test_fixture: bool = False,
) -> JsonDict:
    """Assemble one terminal candidate from raw replay evidence."""

    now = datetime.now(UTC).isoformat()
    settings = load_frozen_settings(REPO_ROOT)
    value: JsonDict = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "status": "complete_pending_reduction",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc or now,
        "completed_at_utc": now,
        "process_identity": {"pid": os.getpid(), "hostname": socket.gethostname()},
        "preconditions_checked": deepcopy([dict(row) for row in preconditions]),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "no_model_load_cpu_prequential_small_head_training",
        "inference_substrate_class": "no_model_load",
        "inference_substrate_details": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
            "historical_readout_source": "cached_Qwen_option_probabilities",
        },
        "execution_venue": "host",
        "duration_s": max(float(duration_s), 1e-6),
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": float(measurement.get("computation_duration_s", 0.0)),
            "validation": max(
                float(duration_s) - float(measurement.get("computation_duration_s", 0.0)), 0.0
            ),
            "historical_capture": 0.0,
        },
        "phase_spans": deepcopy([dict(row) for row in phase_spans]),
        "random_seed": {
            "fitting": 750601,
            "arrival_and_audit": list(SCHEDULE_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": deepcopy([dict(row) for row in source_hashes]),
        "protocol": protocol(),
        "public_plan_hash": measurement.get("public_plan_hash"),
        "rows": deepcopy(list(measurement["rows"])),
        "per_source_results": deepcopy(list(measurement["per_source_results"])),
        "per_update_rows": deepcopy(list(measurement["per_update_rows"])),
        "retention_rows": deepcopy(list(measurement["retention_rows"])),
        "reveal_counts_by_batch": deepcopy(list(measurement["reveal_counts_by_batch"])),
        "checkpoint_hashes": deepcopy(list(measurement.get("checkpoint_hashes", []))),
        "restart_parity_rows": deepcopy(list(measurement["restart_parity_rows"])),
        "validation_receipts": deepcopy([dict(row) for row in validation_receipts]),
        "test_fixture": bool(test_fixture),
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "small_ebm_training": {
            "scope": "current_cpu_work",
            "model_kind": "bounded_residual_heads",
            "learning_rate": settings["learning_rate"],
            "residual_bound": settings["residual_bound"],
            "selected_from_roles": ["analytic_training_fixture", "analytic_calibration_fixture"],
            "online_updates_measured": len(measurement["per_update_rows"]),
            "generator_weights_changed": False,
        },
        "historical_model_provenance": {
            "scope": "historical_only",
            "description": "Cached Qwen option probabilities were inputs; this run made zero calls.",
            "current_model_calls": 0,
        },
        "retention_labels_used_for_selection_or_rollback": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "external_publication_performed": False,
        "push_performed": False,
        "research_conductor_modified": False,
    }
    expected_sources = 2 if test_fixture else 159
    value["sample_size_budget"] = {
        "planned_sources_per_schedule": expected_sources,
        "planned_schedule_delay_units": len(SCHEDULE_SEEDS) * len(DELAYS),
        "planned_prediction_units": expected_sources * len(SCHEDULE_SEEDS) * len(DELAYS),
        "attempted_prediction_units": sum(int(row["source_count"]) for row in value["rows"]),
        "completed_prediction_units": sum(int(row["source_count"]) for row in value["rows"]),
        "excluded_prediction_units": 0,
        "failed_prediction_units": 0,
        "censored_feedback_units": sum(
            int(row["censored_feedback_count"]) for row in value["rows"]
        ),
        "unstarted_prediction_units": 0,
        "sealed_retention_groups": 2 if test_fixture else 116,
    }
    reduction = independent_reduce(value, require_validation=require_validation)
    value["measurement_reduction"] = reduction["measurement_reduction"]
    value["causal_evaluation_complete_score"] = reduction["causal_evaluation_complete_score"]
    value["causal_information_value_score"] = reduction["causal_information_value_score"]
    value["online_benefit_score"] = reduction["online_benefit_score"]
    value["restart_parity_score"] = reduction["restart_parity_score"]
    value["verdict_class"] = reduction["verdict_class"]
    value["honest_verdict"] = (
        "complete_positive_causal_online_benefit_and_retention_passed"
        if reduction["online_benefit_score"] == 1
        else (
            "complete_null_causal_online_measurement_valid_benefit_gate_failed"
            if reduction["causal_evaluation_complete_score"] == 1
            else "complete_disqualified_causal_online_required_validation_failed"
        )
    )
    value["status"] = value["honest_verdict"]
    measured = reduction["measurement_reduction"]
    frozen = next(
        row for row in measured["primary_contrasts"] if row["comparator"] == "frozen_base"
    )
    gates = [
        _gate(
            "valid_complete_measurement",
            "validity",
            1,
            reduction["causal_evaluation_complete_score"],
            "==",
            reduction["causal_evaluation_complete_score"] == 1,
            "Favorable metrics cannot excuse missing rows, chronology, restart, or validation.",
        ),
        _gate(
            "primary_support",
            "support",
            "each seed: sources>=120, delivered>=20, permutable>=12",
            measured["support_rows"],
            "all",
            measured["support_passed"],
            "Weak support closes causal benefit without erasing valid measurement.",
        ),
        _gate(
            "local_brier_effect_size",
            "benefit",
            -0.01,
            frozen["mean_delta"],
            "<=",
            frozen["mean_delta"] is not None and float(frozen["mean_delta"]) <= -0.01,
            "A material-effect floor prevents a tiny interval win from claiming benefit.",
        ),
        _gate(
            "five_holm_primary_contrasts",
            "benefit",
            "upper95<0 and Holm p<0.05 for all five",
            measured["primary_contrasts"],
            "all",
            all(
                row["upper95_delta"] is not None
                and float(row["upper95_delta"]) < 0.0
                and row["holm_passed"] is True
                for row in measured["primary_contrasts"]
            ),
            "All registered controls must lose after family-wise error correction.",
        ),
        _gate(
            "retention",
            "safety",
            0.01,
            measured["retention_contrast"]["upper95_delta"],
            "<=",
            measured["retention_passed"],
            "Static-test forgetting remains disqualifying even after online improvement.",
        ),
        _gate(
            "restart_parity",
            "validity",
            1,
            reduction["restart_parity_score"],
            "==",
            reduction["restart_parity_score"] == 1,
            "Exact restart parity prevents an incomplete checkpoint from passing.",
        ),
    ]
    value["acceptance_gate_results"] = gates
    failed = [row for row in gates if row["passed"] is not True]
    value["gate_check_summary"] = {
        "passed": not failed,
        "failed_checks": [row["check"] for row in failed],
        "first_failure": deepcopy(failed[0]) if failed else None,
    }
    value["raw_evidence_hash"] = canonical_hash(
        {
            "rows": value["rows"],
            "per_source_results": value["per_source_results"],
            "per_update_rows": value["per_update_rows"],
            "retention_rows": value["retention_rows"],
            "restart_parity_rows": value["restart_parity_rows"],
        }
    )
    value["field_principles"] = _field_principles(
        (*value.keys(), "field_principles", "reproducibility_checksum")
    )
    value["reproducibility_checksum"] = _reproducibility_checksum(value)
    return value


def build_test_artifact(*, support_override: bool = False) -> JsonDict:
    """Build deterministic compact evidence for reader and mutation tests."""

    per_source: list[JsonDict] = []
    retention: list[JsonDict] = []
    streams: list[JsonDict] = []
    for delay in DELAYS:
        for seed in SCHEDULE_SEEDS:
            streams.append(
                {
                    "unit_id": f"seed-{seed}-delay-{delay}",
                    "schedule_seed": seed,
                    "delay": delay,
                    "source_count": 2,
                    "delivered_audit_labels": 20 if support_override else 2,
                    "permutable_labels": 12 if support_override else 0,
                    "censored_feedback_count": 1,
                    "chronology_violations": 0,
                    "status": "complete",
                    "failed": False,
                    "censored": False,
                }
            )
            for source_index in range(2):
                identity = f"fixture-{source_index}"
                family = "QA" if source_index == 0 else "Summary"
                for arm in ARMS:
                    loss = 0.10 if arm == "local_brier" else 0.14
                    row = {
                        "group_id": identity,
                        "source_family": family,
                        "schedule_seed": seed,
                        "delay": delay,
                        "arm": arm,
                        "prediction_time": source_index,
                        "audit_selected": True,
                        "prediction_payload": {
                            "group_id": identity,
                            "features": [0.0, 0.0, 0.0, 0.0],
                            "base_probability": 0.5,
                        },
                        "prediction_hash": canonical_hash([identity, seed, delay, arm]),
                        "parameter_hash": canonical_hash([arm, seed, delay]),
                        "probability": 0.5,
                        "label": source_index,
                        "brier_loss": loss,
                        "log_loss": 0.69,
                        "status": "complete",
                        "failed": False,
                        "censored": False,
                    }
                    per_source.append(row)
                    retention.append(
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "group_id",
                                    "source_family",
                                    "schedule_seed",
                                    "delay",
                                    "arm",
                                    "probability",
                                    "label",
                                    "brier_loss",
                                    "log_loss",
                                    "status",
                                    "failed",
                                    "censored",
                                )
                            },
                            "used_for_update": False,
                            "used_for_selection_or_rollback": False,
                        }
                    )
    measurement = {
        "public_plan_hash": canonical_hash("fixture-plan"),
        "rows": streams,
        "per_source_results": per_source,
        "per_update_rows": [
            {
                "group_id": "fixture-0",
                "arm": "local_brier",
                "prediction_time": 0,
                "availability_time": 1,
                "update_time": 1,
                "label_origin": "fixture-0",
                "release_event_ids": ["fixture-0"],
                "status": "committed",
            }
        ],
        "retention_rows": retention,
        "reveal_counts_by_batch": [],
        "checkpoint_hashes": [],
        "restart_parity_rows": [
            {"schedule_seed": seed, "passed": True, "comparisons": {"all": True}}
            for seed in SCHEDULE_SEEDS
        ],
        "restart_parity_score": 1,
        "chronology_violation_count": 0,
        "computation_duration_s": 0.01,
    }
    return build_artifact(
        measurement,
        preconditions=[
            {
                "check": "fixture",
                "upstream": "test",
                "artifact_field": "fixture",
                "expected": True,
                "observed": True,
                "path": "test",
                "passed": True,
            }
        ],
        source_hashes=[],
        require_validation=False,
        duration_s=0.02,
        test_fixture=True,
    )


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    verify_sources: bool = True,
    require_validation: bool = True,
) -> list[str]:
    """Reject changed identity, raw rows, scores, hashes, or required receipts."""

    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(f"field_invalid:{field}")
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
        errors.append("terminal_verdict_prefix_invalid")
    preconditions = value.get("preconditions_checked")
    if not isinstance(preconditions, list) or any(
        not isinstance(row, Mapping) or row.get("passed") is not True for row in preconditions
    ):
        errors.append("preconditions_failed")
    raw_hash = canonical_hash(
        {
            "rows": value.get("rows"),
            "per_source_results": value.get("per_source_results"),
            "per_update_rows": value.get("per_update_rows"),
            "retention_rows": value.get("retention_rows"),
            "restart_parity_rows": value.get("restart_parity_rows"),
        }
    )
    if value.get("raw_evidence_hash") != raw_hash:
        errors.append("raw_evidence_hash_invalid")
    reduced = independent_reduce(value, require_validation=require_validation)
    for field in (
        "causal_evaluation_complete_score",
        "causal_information_value_score",
        "online_benefit_score",
        "restart_parity_score",
        "verdict_class",
        "measurement_reduction",
    ):
        if value.get(field) != reduced.get(field):
            errors.append(f"reduction_mismatch:{field}")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping) or not row.get("principle") for row in gates
    ):
        errors.append("acceptance_gate_principle_missing")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(value):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != _reproducibility_checksum(value):
        errors.append("reproducibility_checksum_invalid")
    sources = value.get("source_artifact_hashes")
    if not isinstance(sources, list):
        errors.append("source_hashes_invalid")
    elif verify_sources:
        for row in sources:
            if not isinstance(row, Mapping):
                errors.append("source_hash_row_invalid")
                continue
            path = root / str(row.get("path") or "")
            if not path.is_file() or row.get("sha256") != sha256_file(path):
                errors.append(f"source_hash_invalid:{row.get('path')}")
    return list(dict.fromkeys(errors))


def cold_replay(
    path: Path,
    *,
    verify_sources: bool = True,
    require_validation: bool = True,
) -> list[str]:
    """Validate serialized evidence through a fresh-reader compatible path."""

    value = _load_object(path)
    if not value:
        return ["artifact_unreadable_or_not_object"]
    return validate_artifact(
        value, verify_sources=verify_sources, require_validation=require_validation
    )


def build_blocked_artifact(
    failed: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a row-free terminal record for one exact external failure."""

    reason = str(failed.get("check") or "external_prerequisite")
    now = datetime.now(UTC).isoformat()
    value: JsonDict = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "status": f"complete_blocked_{reason}",
        "run_date": RUN_DATE,
        "started_at_utc": now,
        "completed_at_utc": now,
        "process_identity": {"pid": os.getpid(), "hostname": socket.gethostname()},
        "preconditions_checked": deepcopy([dict(row) for row in checks]),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "no_model_load_precondition_check_only",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 1e-6,
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": 0.0,
            "validation": 0.0,
            "historical_capture": 0.0,
        },
        "phase_spans": [],
        "random_seed": {
            "fitting": 750601,
            "arrival_and_audit": list(SCHEDULE_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": [],
        "rows": [],
        "per_source_results": [],
        "per_update_rows": [],
        "retention_rows": [],
        "reveal_counts_by_batch": [],
        "checkpoint_hashes": [],
        "restart_parity_rows": [],
        "sample_size_budget": {
            "planned_prediction_units": 1590,
            "attempted_prediction_units": 0,
            "completed_prediction_units": 0,
            "excluded_prediction_units": 0,
            "failed_prediction_units": 0,
            "censored_feedback_units": 0,
            "unstarted_prediction_units": 1590,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [reason],
            "first_failure": deepcopy(dict(failed)),
        },
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "causal_evaluation_complete_score": 0,
        "causal_information_value_score": 0,
        "online_benefit_score": 0,
        "restart_parity_score": 0,
    }
    value["field_principles"] = _field_principles(
        (*value.keys(), "field_principles", "reproducibility_checksum")
    )
    value["reproducibility_checksum"] = _reproducibility_checksum(value)
    return value


def _span(
    phase: str, phase_started: float, run_started: float, completed_units: int
) -> JsonDict:  # pragma: no cover
    """Close one monotonic phase with its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
    }


def _terminal_commands(
    candidate: Path, *, allow_pending_validation: bool
) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build cold replay, reduction, adversarial, and strict reader commands."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pending = ("--allow-pending-validation",) if allow_pending_validation else ()
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                str(candidate),
                *pending,
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
                "--independent-reduce",
                str(candidate),
                *pending,
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300.0,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover
    """Authenticate, measure, validate, cold-replay, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks, hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed, checks)
        progress(started, "publish", "before_atomic_terminal", verdict="blocked")
        atomic_json(destination, blocked)
        progress(started, "publish", f"complete_blocked_{failed.get('check')}")
        return blocked
    progress(started, "preconditions", "complete", checks=len(checks))

    raw_root = root / RAW_DIR
    manifest_path = raw_root / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    )
    checkpoint_root = Path(tempfile.mkdtemp(prefix="carnot-exp7509-checkpoints-", dir="/tmp"))
    progress(started, "measurement", "before_numeric_replay")
    phase_started = time.monotonic()
    measurement = run_measurement(root, checkpoint_root)
    spans.append(_span("measurement", phase_started, started, len(measurement["rows"])))
    progress(
        started,
        "measurement",
        "after_numeric_replay",
        streams=len(measurement["rows"]),
        predictions=len(measurement["per_source_results"]),
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7509-validation-", dir="/tmp"))
    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        VALIDATION_MANIFEST.test_paths,
        VALIDATION_MANIFEST.changed_modules,
        static_paths=VALIDATION_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7509",
    )
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=validation_scope.reduce_required_checks(affected)["required_checks_passed"],
    )

    candidate = build_artifact(
        measurement,
        preconditions=checks,
        source_hashes=hashes,
        validation_receipts=affected,
        require_validation=False,
        phase_spans=spans,
        started_at_utc=started_at,
        duration_s=time.monotonic() - started,
    )
    candidate_path = raw_root / "measured_terminal_candidate.json"
    candidate_errors = validate_artifact(candidate, root=root, require_validation=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")
    atomic_json(candidate_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        _terminal_commands(candidate_path, allow_pending_validation=True),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=_receipts_pass(terminal, TERMINAL_CHECK_NAMES),
    )

    final = build_artifact(
        measurement,
        preconditions=checks,
        source_hashes=hashes,
        validation_receipts=[*affected, *terminal],
        require_validation=True,
        phase_spans=spans,
        started_at_utc=started_at,
        duration_s=time.monotonic() - started,
    )
    final_errors = validate_artifact(final, root=root, require_validation=True)
    if final_errors:
        raise RuntimeError(f"final_candidate_invalid:{final_errors}")
    final_candidate = raw_root / "exact_terminal_candidate.json"
    atomic_json(final_candidate, final)

    progress(started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        _terminal_commands(final_candidate, allow_pending_validation=False),
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
        causal_evaluation_complete_score=final["causal_evaluation_complete_score"],
        online_benefit_score=final["online_benefit_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--allow-pending-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one independent reader through the thin wrapper."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    verify_sources = not bool(args.no_source_check)
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        require_validation = (
            not args.allow_pending_validation and value.get("test_fixture") is not True
        )
        errors = (
            validate_artifact(
                value,
                verify_sources=verify_sources,
                require_validation=require_validation,
            )
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        require_validation = (
            not args.allow_pending_validation and value.get("test_fixture") is not True
        )
        reduction = independent_reduce(value, require_validation=require_validation)
        errors = (
            validate_artifact(
                value,
                verify_sources=verify_sources,
                require_validation=require_validation,
            )
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({**reduction, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors) or reduction["causal_evaluation_complete_score"] != 1)
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "result": RESULT_PATH.as_posix(),
                "causal_evaluation_complete_score": artifact["causal_evaluation_complete_score"],
                "online_benefit_score": artifact["online_benefit_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if artifact["verdict_class"] in {"positive", "null"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
