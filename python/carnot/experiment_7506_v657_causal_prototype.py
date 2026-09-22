"""Qualify a smooth residual learner and causal event machine on fixtures.

The fixtures make update errors easy to detect. They do not estimate benefit
on the sealed V657 online or retention labels.

Spec refs: REQ-KAN-7506, REQ-CL-7506, and their SCENARIO-* items.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import AffectedManifest, PlannedCommand
from carnot.experiment_7468_v654_residual_learner import fit_local_knots, local_design
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7506-v657-causal-prototype"
SCHEMA = "carnot.exp7506.v657.causal_prototype.v1"
CHECKPOINT_SCHEMA = "carnot.exp7506.v657.causal_machine_checkpoint.v1"
RESULT_PATH = Path("results/experiment_7506_v657_causal_prototype.json")
RAW_DIR = Path("results/raw/experiment_7506_v657_causal_prototype")
MODULE_PATH = Path("python/carnot/experiment_7506_v657_causal_prototype.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7506_v657_causal_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7506_v657_causal_prototype.py")
CL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
KAN_SPEC_PATH = Path("openspec/capabilities/kan/spec.md")
UPSTREAM_ARTIFACT = Path("results/experiment_7505_v657_energy_fit.json")
UPSTREAM_CHECKPOINT = Path("results/raw/experiment_7505_v657_energy_fit/frozen_checkpoints.json")
HISTORICAL_AUDIT = Path("results/experiment_7490_v656_historical_audit.json")

LEARNING_RATES = (0.001, 0.01, 0.03)
RESIDUAL_BOUNDS = (0.5, 1.0)
PRIMARY_DELAY = 8
SECONDARY_DELAY = 0
BLOCK_SIZE = 8
AUDIT_PROBABILITY = 0.25
FIT_SEED = 750_601
ARRIVAL_SEED = 750_602
AUDIT_SEED = 750_603
SHUFFLE_SEED = 750_604
BOOTSTRAP_SEED = 750_605
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


def _sigmoid(value: float) -> float:
    """Return a stable scalar logistic probability."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _base_logit(probability: float) -> float:
    """Convert a frozen baseline probability without allowing infinities."""

    value = float(probability)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("base_probability_invalid")
    clipped = min(max(value, 1e-12), 1.0 - 1e-12)
    return math.log(clipped / (1.0 - clipped))


class BoundedResidualHead:
    """Apply one smooth bounded residual above an immutable baseline."""

    def __init__(
        self,
        *,
        knots: np.ndarray | None,
        coefficients: np.ndarray,
        learning_rate: float,
        residual_bound: float,
        loss: str,
        basis_kind: str,
    ) -> None:
        self.knots = None if knots is None else np.asarray(knots, dtype=np.float64).copy()
        self.coefficients = np.asarray(coefficients, dtype=np.float64).copy()
        self.learning_rate = float(learning_rate)
        self.residual_bound = float(residual_bound)
        self.loss = str(loss)
        self.basis_kind = str(basis_kind)
        self.update_count = 0
        self.gradient_clip_count = 0
        if self.coefficients.ndim != 1 or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("coefficients_invalid")
        if self.loss not in {"brier", "log_loss"}:
            raise ValueError("loss_invalid")
        if self.basis_kind not in {"local", "affine", "intercept"}:
            raise ValueError("basis_kind_invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0.0:
            raise ValueError("learning_rate_invalid")
        if not math.isfinite(self.residual_bound) or self.residual_bound <= 0.0:
            raise ValueError("residual_bound_invalid")

    @classmethod
    def from_training(
        cls,
        training_features: Any,
        *,
        learning_rate: float,
        residual_bound: float,
        loss: str,
        basis_kind: str,
    ) -> BoundedResidualHead:
        """Freeze the basis from training inputs before an online label exists."""

        training = np.asarray(training_features, dtype=np.float64)
        if training.ndim != 2 or training.shape[1] != 4 or not np.all(np.isfinite(training)):
            raise ValueError("training_features_invalid")
        knots: np.ndarray | None = None
        if basis_kind == "local":
            knots = fit_local_knots(training)
            design, _support = local_design(training[0], knots)
            parameter_count = len(design) + 1
        elif basis_kind == "affine":
            parameter_count = 5
        elif basis_kind == "intercept":
            parameter_count = 1
        else:
            raise ValueError("basis_kind_invalid")
        return cls(
            knots=knots,
            coefficients=np.zeros(parameter_count, dtype=np.float64),
            learning_rate=learning_rate,
            residual_bound=residual_bound,
            loss=loss,
            basis_kind=basis_kind,
        )

    @property
    def parameter_count(self) -> int:
        """Report the exact trainable capacity for control matching."""

        return int(len(self.coefficients))

    @property
    def state_hash(self) -> str:
        """Bind parameters and fixed update settings."""

        return canonical_hash(self.to_payload())

    def _basis(self, features: Any) -> np.ndarray:
        """Return the frozen basis for one four-feature observation."""

        values = np.asarray(features, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("features_invalid")
        if self.basis_kind == "local":
            if self.knots is None:
                raise ValueError("local_knots_missing")
            design, _support = local_design(values, self.knots)
            return np.concatenate((design, np.ones(1, dtype=np.float64)))
        if self.basis_kind == "affine":
            return np.concatenate((values, np.ones(1, dtype=np.float64)))
        return np.ones(1, dtype=np.float64)

    def predict(self, features: Any, *, base_probability: float) -> JsonDict:
        """Compute the smooth residual without reading an outcome."""

        basis = self._basis(features)
        raw = float(basis @ self.coefficients)
        scaled = math.tanh(raw / self.residual_bound)
        residual = self.residual_bound * scaled
        probability = _sigmoid(_base_logit(base_probability) + residual)
        return {
            "base_probability": float(base_probability),
            "raw_residual": raw,
            "residual": residual,
            "residual_derivative": 1.0 - scaled * scaled,
            "probability": probability,
            "state_hash": self.state_hash,
        }

    def gradient(
        self, features: Any, label: int, *, base_probability: float
    ) -> tuple[np.ndarray, float]:
        """Differentiate the declared Brier or matched log loss exactly."""

        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("label_invalid")
        basis = self._basis(features)
        prediction = self.predict(features, base_probability=base_probability)
        probability = float(prediction["probability"])
        derivative = float(prediction["residual_derivative"])
        if self.loss == "brier":
            multiplier = 2.0 * (probability - label) * probability * (1.0 - probability)
            loss = (probability - label) ** 2
        else:
            multiplier = probability - label
            clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
            loss = -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))
        return multiplier * derivative * basis, float(loss)

    def update(self, features: Any, label: int, *, base_probability: float) -> JsonDict:
        """Replay one gradient against current parameters and reject bad numbers."""

        before = self.predict(features, base_probability=base_probability)
        gradient, loss = self.gradient(features, label, base_probability=base_probability)
        norm = float(np.linalg.norm(gradient))
        scale = min(1.0, 1.0 / max(norm, np.finfo(np.float64).tiny))
        candidate = self.coefficients - self.learning_rate * gradient * scale
        if not np.all(np.isfinite(candidate)):
            raise ValueError("candidate_state_nonfinite")
        self.coefficients = candidate
        self.update_count += int(self.learning_rate > 0.0)
        self.gradient_clip_count += int(scale < 1.0)
        return {
            "status": "zero_step" if self.learning_rate == 0.0 else "committed",
            "gradient_probability": before["probability"],
            "gradient_norm": norm,
            "gradient_scale": scale,
            "loss": loss,
            "state_hash_after": self.state_hash,
            "update_count": self.update_count,
        }

    def to_payload(self) -> JsonDict:
        """Serialize all numeric state needed for exact replay."""

        return {
            "knots": None if self.knots is None else self.knots.tolist(),
            "coefficients": self.coefficients.tolist(),
            "learning_rate": self.learning_rate,
            "residual_bound": self.residual_bound,
            "loss": self.loss,
            "basis_kind": self.basis_kind,
            "update_count": self.update_count,
            "gradient_clip_count": self.gradient_clip_count,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> BoundedResidualHead:
        """Restore a head without fitting or opening any labels."""

        head = cls(
            knots=None if value.get("knots") is None else np.asarray(value["knots"]),
            coefficients=np.asarray(value["coefficients"]),
            learning_rate=float(value["learning_rate"]),
            residual_bound=float(value["residual_bound"]),
            loss=str(value["loss"]),
            basis_kind=str(value["basis_kind"]),
        )
        head.update_count = int(value.get("update_count", 0))
        head.gradient_clip_count = int(value.get("gradient_clip_count", 0))
        return head


def finite_difference_gradient(
    head: BoundedResidualHead,
    features: Any,
    label: int,
    *,
    base_probability: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute centered numeric derivatives without using the analytic path."""

    original = head.coefficients.copy()
    numeric = np.zeros_like(original)
    for index in range(len(original)):
        head.coefficients = original.copy()
        head.coefficients[index] += epsilon
        plus = head.predict(features, base_probability=base_probability)["probability"]
        head.coefficients[index] -= 2.0 * epsilon
        minus = head.predict(features, base_probability=base_probability)["probability"]
        if head.loss == "brier":
            plus_loss = (float(plus) - label) ** 2
            minus_loss = (float(minus) - label) ** 2
        else:
            plus_p = min(max(float(plus), 1e-12), 1.0 - 1e-12)
            minus_p = min(max(float(minus), 1e-12), 1.0 - 1e-12)
            plus_loss = -(label * math.log(plus_p) + (1 - label) * math.log1p(-plus_p))
            minus_loss = -(label * math.log(minus_p) + (1 - label) * math.log1p(-minus_p))
        numeric[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    head.coefficients = original
    return numeric


def training_fixture() -> np.ndarray:
    """Return a bounded training-only support grid for the spline knots."""

    return np.asarray(
        [
            [-2.0, -1.0, 0.0, 1.0],
            [-1.5, -0.5, 0.5, 1.5],
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 0.5, 1.5, 2.5],
            [1.0, 1.0, 2.0, 3.0],
            [2.0, 1.5, 2.5, 3.5],
        ],
        dtype=np.float64,
    )


def _selection_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Build separate analytic training and calibration rows."""

    training: list[JsonDict] = []
    calibration: list[JsonDict] = []
    for index in range(24):
        features = training_fixture()[index % 6] + (index // 6) * 0.03
        row = {
            "features": features.tolist(),
            "base_probability": 0.25 + 0.1 * (index % 5),
            "label": int((index + index // 3) % 2),
        }
        (training if index < 16 else calibration).append(row)
    return training, calibration


def _mean_brier(head: BoundedResidualHead, rows: Sequence[Mapping[str, Any]]) -> float:
    """Score one frozen role without changing the head."""

    return math.fsum(
        (
            float(
                head.predict(row["features"], base_probability=float(row["base_probability"]))[
                    "probability"
                ]
            )
            - int(row["label"])
        )
        ** 2
        for row in rows
    ) / len(rows)


def select_fixture_hyperparameters() -> JsonDict:
    """Select the declared budget before any online or retention labels open."""

    training, calibration = _selection_rows()
    candidates: list[JsonDict] = []
    for learning_rate, bound in itertools.product(LEARNING_RATES, RESIDUAL_BOUNDS):
        head = BoundedResidualHead.from_training(
            training_fixture(),
            learning_rate=learning_rate,
            residual_bound=bound,
            loss="brier",
            basis_kind="local",
        )
        for _epoch in range(3):
            for row in training:
                head.update(
                    row["features"],
                    int(row["label"]),
                    base_probability=float(row["base_probability"]),
                )
        candidates.append(
            {
                "learning_rate": learning_rate,
                "residual_bound": bound,
                "calibration_brier": _mean_brier(head, calibration),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["calibration_brier"], row["learning_rate"], row["residual_bound"]),
    )
    return {
        "learning_rate_candidates": list(LEARNING_RATES),
        "residual_bound_candidates": list(RESIDUAL_BOUNDS),
        "selection_roles": ["training", "calibration_tuning"],
        "online_labels_used": False,
        "retention_labels_used": False,
        "budget_frozen_before_online": True,
        "candidate_rows": candidates,
        "selected_learning_rate": selected["learning_rate"],
        "selected_residual_bound": selected["residual_bound"],
    }


def run_head_controls(selection: Mapping[str, Any]) -> list[JsonDict]:
    """Run local, simple, frozen, and zero-step heads on the same fixture."""

    rows, _calibration = _selection_rows()
    settings = {
        "learning_rate": float(selection["selected_learning_rate"]),
        "residual_bound": float(selection["selected_residual_bound"]),
    }
    declarations = (
        ("local_brier", "local", "brier", settings["learning_rate"]),
        ("local_log_loss", "local", "log_loss", settings["learning_rate"]),
        ("affine_brier", "affine", "brier", settings["learning_rate"]),
        ("intercept_brier", "intercept", "brier", settings["learning_rate"]),
        ("frozen", "local", "brier", 0.0),
        ("zero_step", "local", "brier", 0.0),
    )
    output: list[JsonDict] = []
    for arm, basis_kind, loss, learning_rate in declarations:
        head = BoundedResidualHead.from_training(
            training_fixture(),
            learning_rate=learning_rate,
            residual_bound=settings["residual_bound"],
            loss=loss,
            basis_kind=basis_kind,
        )
        before = head.state_hash
        if arm != "frozen":
            for row in rows[:8]:
                head.update(
                    row["features"],
                    int(row["label"]),
                    base_probability=float(row["base_probability"]),
                )
        output.append(
            {
                "arm": arm,
                "basis_kind": basis_kind,
                "loss": loss,
                "parameter_count": head.parameter_count,
                "state_changed": head.state_hash != before,
                "update_count": head.update_count,
                "gradient_clip_count": head.gradient_clip_count,
            }
        )
    return output


def validate_release_origins(
    released_event_ids: Sequence[str], label_origins: Sequence[str], *, release_time: int
) -> None:
    """Reject any shuffled origin outside the batch that is now available."""

    allowed = set(released_event_ids)
    if any(origin not in allowed for origin in label_origins):
        raise ValueError("origin_not_in_released_batch")
    if release_time < 0:
        raise ValueError("release_time_invalid")


def permute_released_batch(
    labels: Sequence[int], event_ids: Sequence[str], *, seed: int
) -> JsonDict:
    """Permute only one released batch and name unavoidable no-op cases."""

    values = [int(label) for label in labels]
    identities = [str(event_id) for event_id in event_ids]
    if len(values) != len(identities) or not values:
        raise ValueError("released_batch_invalid")
    if any(label not in (0, 1) for label in values) or len(set(identities)) != len(identities):
        raise ValueError("released_batch_invalid")
    if len(values) == 1:
        return {
            "labels": values,
            "label_origins": identities,
            "mode": "singleton_noop",
            "changed": 0,
        }
    if len(set(values)) == 1:
        return {
            "labels": values,
            "label_origins": identities,
            "mode": "identical_labels_noop",
            "changed": 0,
        }
    rng = np.random.default_rng(int(seed))
    permutations = list(itertools.permutations(range(len(values))))
    rng.shuffle(permutations)
    best = max(
        permutations,
        key=lambda order: sum(values[index] != values[pos] for pos, index in enumerate(order)),
    )
    shuffled = [values[index] for index in best]
    origins = [identities[index] for index in best]
    changed = sum(left != right for left, right in zip(values, shuffled, strict=True))
    mode = "derangement" if changed == len(values) else "best_effort_permutation"
    validate_release_origins(identities, origins, release_time=0)
    return {"labels": shuffled, "label_origins": origins, "mode": mode, "changed": changed}


@dataclass
class CausalEventMachine:
    """Keep prediction, withheld labels, releases, and both arms independent."""

    real_head: BoundedResidualHead
    shuffled_head: BoundedResidualHead
    audit_rng: np.random.Generator
    shuffle_seed: int
    delay: int
    order_cursor: int
    predictions: dict[str, JsonDict]
    pending_queue: list[JsonDict]
    update_rows: list[JsonDict]
    batch_rows: list[JsonDict]

    @classmethod
    def create(
        cls,
        training_features: Any,
        *,
        learning_rate: float,
        residual_bound: float,
        delay: int,
    ) -> CausalEventMachine:
        """Create independent real and shuffled heads before stream arrival."""

        if delay not in {PRIMARY_DELAY, SECONDARY_DELAY}:
            raise ValueError("delay_invalid")
        real = BoundedResidualHead.from_training(
            training_features,
            learning_rate=learning_rate,
            residual_bound=residual_bound,
            loss="brier",
            basis_kind="local",
        )
        shuffled = BoundedResidualHead.from_payload(real.to_payload())
        return cls(
            real_head=real,
            shuffled_head=shuffled,
            audit_rng=np.random.default_rng(AUDIT_SEED),
            shuffle_seed=SHUFFLE_SEED,
            delay=delay,
            order_cursor=0,
            predictions={},
            pending_queue=[],
            update_rows=[],
            batch_rows=[],
        )

    @property
    def state_hash(self) -> str:
        """Bind only learner state, not labels that remain unavailable."""

        return canonical_hash(
            {"real": self.real_head.to_payload(), "shuffled": self.shuffled_head.to_payload()}
        )

    def predict(self, event: Mapping[str, Any]) -> JsonDict:
        """Seal immutable features and probabilities before label submission."""

        event_id = str(event.get("event_id") or "")
        arrival = event.get("arrival_index")
        if not event_id or event_id in self.predictions:
            raise ValueError("event_identity_invalid")
        if arrival != self.order_cursor:
            raise ValueError("arrival_order_invalid")
        features = np.asarray(event.get("features"), dtype=np.float64)
        base = float(event.get("base_probability"))
        real = self.real_head.predict(features, base_probability=base)
        shuffled = self.shuffled_head.predict(features, base_probability=base)
        block_id = self.order_cursor // BLOCK_SIZE
        block_end = block_id * BLOCK_SIZE + BLOCK_SIZE - 1
        selected = bool(self.audit_rng.random() < AUDIT_PROBABILITY)
        row = {
            "event_id": event_id,
            "source_id": str(event.get("source_id") or ""),
            "arrival_index": self.order_cursor,
            "block_id": block_id,
            "block_end": block_end,
            "release_time": block_end + self.delay,
            "features": features.tolist(),
            "base_probability": base,
            "real_probability": real["probability"],
            "shuffled_probability": shuffled["probability"],
            "audit_selected": selected,
            "prediction_state_hash": self.state_hash,
        }
        row["prediction_hash"] = canonical_hash(row)
        self.predictions[event_id] = deepcopy(row)
        self.order_cursor += 1
        return deepcopy(row)

    def deliver_label(self, event_id: str, label: int, *, visible_at: int) -> str:
        """Place a selected label behind its fixed release boundary."""

        identity = str(event_id)
        if identity not in self.predictions:
            raise ValueError("prediction_required_before_label")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("label_invalid")
        prediction = self.predictions[identity]
        if not prediction["audit_selected"]:
            return "not_selected"
        if any(row["event_id"] == identity for row in self.pending_queue) or any(
            row["event_id"] == identity and row["arm"] == "real" for row in self.update_rows
        ):
            return "duplicate"
        self.pending_queue.append(
            {
                "event_id": identity,
                "label": int(label),
                "label_origin": identity,
                "received_at": int(visible_at),
                "block_id": prediction["block_id"],
                "release_time": prediction["release_time"],
            }
        )
        return "withheld" if visible_at < prediction["release_time"] else "available"

    def advance(self, visible_at: int) -> list[JsonDict]:
        """Release complete due batches after the current prediction is sealed."""

        due_blocks = sorted(
            {
                int(row["block_id"])
                for row in self.pending_queue
                if int(row["release_time"]) <= int(visible_at)
            }
        )
        released: list[JsonDict] = []
        for block_id in due_blocks:
            batch = sorted(
                [row for row in self.pending_queue if int(row["block_id"]) == block_id],
                key=lambda row: self.predictions[str(row["event_id"])]["arrival_index"],
            )
            labels = [int(row["label"]) for row in batch]
            event_ids = [str(row["event_id"]) for row in batch]
            permutation = permute_released_batch(
                labels, event_ids, seed=self.shuffle_seed + block_id + self.delay * 100
            )
            validate_release_origins(
                event_ids, permutation["label_origins"], release_time=int(visible_at)
            )
            self._apply_batch(batch, permutation, visible_at=int(visible_at))
            batch_row = {
                "block_id": block_id,
                "block_end": block_id * BLOCK_SIZE + BLOCK_SIZE - 1,
                "release_time": block_id * BLOCK_SIZE + BLOCK_SIZE - 1 + self.delay,
                "released_at": int(visible_at),
                "event_ids": event_ids,
                "label_origins": permutation["label_origins"],
                "permutation_mode": permutation["mode"],
                "effective_changes": permutation["changed"],
                "update_count": len(batch),
            }
            self.batch_rows.append(batch_row)
            released.append(deepcopy(batch_row))
            self.pending_queue = [
                row for row in self.pending_queue if int(row["block_id"]) != block_id
            ]
        return released

    def _apply_batch(
        self, batch: Sequence[Mapping[str, Any]], permutation: Mapping[str, Any], *, visible_at: int
    ) -> None:
        """Update both arms in feature order from the same available batch."""

        shuffled_labels = list(permutation["labels"])
        shuffled_origins = list(permutation["label_origins"])
        for index, pending in enumerate(batch):
            event_id = str(pending["event_id"])
            prediction = self.predictions[event_id]
            if visible_at < int(prediction["release_time"]):
                raise ValueError("label_not_available")
            real_receipt = self.real_head.update(
                prediction["features"],
                int(pending["label"]),
                base_probability=float(prediction["base_probability"]),
            )
            shuffled_receipt = self.shuffled_head.update(
                prediction["features"],
                int(shuffled_labels[index]),
                base_probability=float(prediction["base_probability"]),
            )
            common = {
                "event_id": event_id,
                "block_id": int(pending["block_id"]),
                "prediction_hash": prediction["prediction_hash"],
                "prediction_time": prediction["arrival_index"],
                "release_time": prediction["release_time"],
                "updated_at": visible_at,
                "feature_order": prediction["arrival_index"],
            }
            self.update_rows.append(
                {
                    **common,
                    "arm": "real",
                    "label": int(pending["label"]),
                    "label_origin": str(pending["label_origin"]),
                    "status": real_receipt["status"],
                    "gradient_probability": real_receipt["gradient_probability"],
                    "state_hash_after": real_receipt["state_hash_after"],
                }
            )
            self.update_rows.append(
                {
                    **common,
                    "arm": "shuffled",
                    "label": int(shuffled_labels[index]),
                    "label_origin": str(shuffled_origins[index]),
                    "status": shuffled_receipt["status"],
                    "gradient_probability": shuffled_receipt["gradient_probability"],
                    "state_hash_after": shuffled_receipt["state_hash_after"],
                }
            )

    def checkpoint_payload(self) -> JsonDict:
        """Bind models, unavailable labels, audit state, and stream position."""

        return {
            "schema": CHECKPOINT_SCHEMA,
            "models": {
                "real": self.real_head.to_payload(),
                "shuffled": self.shuffled_head.to_payload(),
            },
            "pending_queue": deepcopy(self.pending_queue),
            "audit_rng_state": deepcopy(self.audit_rng.bit_generator.state),
            "shuffle_seed": self.shuffle_seed,
            "delay": self.delay,
            "order_cursor": self.order_cursor,
            "predictions": deepcopy(self.predictions),
            "update_rows": deepcopy(self.update_rows),
            "batch_rows": deepcopy(self.batch_rows),
        }

    def checkpoint_bytes(self) -> str:
        """Return canonical checkpoint bytes for restart comparison."""

        return json.dumps(self.checkpoint_payload(), sort_keys=True, separators=(",", ":"))

    def save_checkpoint(self, path: Path) -> None:
        """Write a complete checkpoint atomically before replacing prior bytes."""

        atomic_json(Path(path), self.checkpoint_payload())

    @classmethod
    def load_checkpoint(cls, path: Path) -> CausalEventMachine:
        """Restore all causal state from one checkpoint."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint_unreadable") from error
        if not isinstance(value, Mapping) or value.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint_schema_invalid")
        models = value.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("checkpoint_models_invalid")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(value["audit_rng_state"])
        return cls(
            real_head=BoundedResidualHead.from_payload(models["real"]),
            shuffled_head=BoundedResidualHead.from_payload(models["shuffled"]),
            audit_rng=rng,
            shuffle_seed=int(value["shuffle_seed"]),
            delay=int(value["delay"]),
            order_cursor=int(value["order_cursor"]),
            predictions={str(key): deepcopy(row) for key, row in value["predictions"].items()},
            pending_queue=deepcopy(list(value["pending_queue"])),
            update_rows=deepcopy(list(value["update_rows"])),
            batch_rows=deepcopy(list(value["batch_rows"])),
        )


def _temperature_probability(raw_probability: float, temperature: float = 1.5) -> float:
    """Apply the Exp7505 frozen scalar temperature to one fixture score."""

    return _sigmoid(_base_logit(raw_probability) / temperature)


def fixture_events() -> list[JsonDict]:
    """Build two fixed blocks with repeated sources and analytic labels."""

    events: list[JsonDict] = []
    support = training_fixture()
    for index in range(16):
        raw_probability = 0.18 + 0.045 * (index % 12)
        events.append(
            {
                "event_id": f"fixture-event-{index:02d}",
                "source_id": f"repeated-source-{index % 5}",
                "arrival_index": index,
                "features": (support[index % len(support)] + (index // 6) * 0.02).tolist(),
                "base_probability": _temperature_probability(raw_probability),
                "label": int(index % 2),
                "label_role": "analytic_online_fixture",
            }
        )
    return events


def _stable_trace(machine: CausalEventMachine) -> JsonDict:
    """Exclude unavailable labels while binding all released causal evidence."""

    return {
        "predictions": [deepcopy(machine.predictions[key]) for key in machine.predictions],
        "updates": deepcopy(machine.update_rows),
        "batches": deepcopy(machine.batch_rows),
        "model_state_hash": machine.state_hash,
        "order_cursor": machine.order_cursor,
    }


def run_causal_fixture(
    events: Sequence[Mapping[str, Any]],
    *,
    delay: int,
    stop_after_release_block: int | None = None,
    checkpoint_path: Path | None = None,
) -> JsonDict:
    """Replay one fixture stream with an optional real checkpoint restart."""

    selection = select_fixture_hyperparameters()
    machine = CausalEventMachine.create(
        training_fixture(),
        learning_rate=float(selection["selected_learning_rate"]),
        residual_bound=float(selection["selected_residual_bound"]),
        delay=delay,
    )
    restart_performed = False
    for index, event in enumerate(events):
        prediction = machine.predict(event)
        machine.deliver_label(
            str(event["event_id"]), int(event["label"]), visible_at=int(event["arrival_index"])
        )
        released = machine.advance(int(event["arrival_index"]))
        if prediction != machine.predictions[str(event["event_id"])]:
            raise ValueError("prediction_rewritten")
        if checkpoint_path is not None and index == 9:
            machine.save_checkpoint(checkpoint_path)
            machine = CausalEventMachine.load_checkpoint(checkpoint_path)
            restart_performed = True
        if stop_after_release_block is not None and any(
            int(row["block_id"]) == stop_after_release_block for row in released
        ):
            break
    else:
        if events:
            last_block = max(int(row["arrival_index"]) // BLOCK_SIZE for row in events)
            machine.advance(last_block * BLOCK_SIZE + BLOCK_SIZE - 1 + delay)

    violations = 0
    batch_events = {int(row["block_id"]): set(row["event_ids"]) for row in machine.batch_rows}
    for row in machine.update_rows:
        allowed = batch_events.get(int(row["block_id"]), set())
        violations += int(
            row["label_origin"] not in allowed or int(row["updated_at"]) < int(row["release_time"])
        )
    predict_before_update = all(
        int(row["prediction_time"]) <= int(row["updated_at"])
        and row["prediction_hash"] == machine.predictions[str(row["event_id"])]["prediction_hash"]
        for row in machine.update_rows
    )
    return {
        "delay": delay,
        "predictions": [deepcopy(machine.predictions[key]) for key in machine.predictions],
        "updates": deepcopy(machine.update_rows),
        "batches": deepcopy(machine.batch_rows),
        "stable_trace": _stable_trace(machine),
        "future_access_violations": violations,
        "predict_before_update": predict_before_update,
        "restart_performed": restart_performed,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "terminal_checkpoint_bytes": machine.checkpoint_bytes(),
        "terminal_state_hash": machine.state_hash,
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
    Path("python/carnot/experiment_7490_v656_historical_audit.py"),
    HISTORICAL_AUDIT,
    Path("tests/python/test_experiment_7496_v656_causal_update_fixture.py"),
    CL_SPEC_PATH,
    KAN_SPEC_PATH,
    Path("results/experiment_7503_v657_contract_methods.json"),
    Path("results/experiment_7504_v657_evidence_interface.json"),
    UPSTREAM_ARTIFACT,
    UPSTREAM_CHECKPOINT,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any, path: str
) -> JsonDict:
    """Record exact prerequisite evidence without repairing external bytes."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "path": path,
        "passed": observed == expected,
    }


def _source_row(path: Path, root: Path) -> JsonDict:
    """Hash one exact input while keeping repository paths portable."""

    resolved = root / path
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate instructions, specs, history, and successful V657 inputs."""

    rows: list[JsonDict] = []
    hashes: list[JsonDict] = []
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        exists = path.is_file() and path.stat().st_size > 0
        rows.append(
            _precondition(
                "resource_readable",
                "worktree_or_upstream",
                "path",
                True,
                exists,
                relative.as_posix(),
            )
        )
        if exists:
            hashes.append(_source_row(relative, root))
    if not all(row["passed"] for row in rows):
        return rows, hashes
    upstream = json.loads((root / UPSTREAM_ARTIFACT).read_text(encoding="utf-8"))
    for check, field, expected in (
        ("upstream_terminal", "terminal_status", "complete"),
        ("upstream_ready", "energy_fit_ready_score", 1),
        ("upstream_adversarial", "flagged_adversarial", False),
        ("upstream_model_invoked", "model_invoked", False),
    ):
        rows.append(
            _precondition(
                check,
                "Exp7505",
                field,
                expected,
                upstream.get(field),
                UPSTREAM_ARTIFACT.as_posix(),
            )
        )
    checkpoint = json.loads((root / UPSTREAM_CHECKPOINT).read_text(encoding="utf-8"))
    temperature = checkpoint.get("temperature_baseline", {}).get("selected_temperature")
    rows.append(
        _precondition(
            "frozen_temperature_available",
            "Exp7505",
            "temperature_baseline.selected_temperature",
            1.5,
            temperature,
            UPSTREAM_CHECKPOINT.as_posix(),
        )
    )
    historical = json.loads((root / HISTORICAL_AUDIT).read_text(encoding="utf-8"))
    count = historical.get("feedback_chronology_summary", {}).get("future_origin_assignment_count")
    rows.append(
        _precondition(
            "historical_future_origin_failure",
            "Exp7490",
            "future_origin_assignment_count",
            376,
            count,
            HISTORICAL_AUDIT.as_posix(),
        )
    )
    for spec, requirement in ((CL_SPEC_PATH, "REQ-CL-7506"), (KAN_SPEC_PATH, "REQ-KAN-7506")):
        observed = requirement in (root / spec).read_text(encoding="utf-8")
        rows.append(
            _precondition(
                "requirement_present", "OpenSpec", requirement, True, observed, spec.as_posix()
            )
        )
    return rows, hashes


def _gradient_checks() -> list[JsonDict]:
    """Measure both labels across interior and saturated smooth states."""

    rows: list[JsonDict] = []
    features = training_fixture()[3]
    for bound, label, raw_scale in itertools.product(
        RESIDUAL_BOUNDS, (0, 1), (0.0, 0.9, -0.9, 8.0, -8.0)
    ):
        head = BoundedResidualHead.from_training(
            training_fixture(),
            learning_rate=0.03,
            residual_bound=bound,
            loss="brier",
            basis_kind="local",
        )
        head.coefficients[-1] = raw_scale * bound
        analytic = head.gradient(features, label, base_probability=0.41)[0]
        numeric = finite_difference_gradient(head, features, label, base_probability=0.41)
        error = float(np.max(np.abs(analytic - numeric)))
        rows.append(
            {
                "unit_id": f"gradient-b{bound}-y{label}-u{raw_scale}",
                "bound": bound,
                "label": label,
                "raw_scale": raw_scale,
                "max_abs_error": error,
                "gradient_norm": float(np.linalg.norm(analytic)),
                "passed": error <= 2e-7,
                "status": "complete",
                "failed": False,
                "censored": False,
            }
        )
    return rows


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep each structural threshold explicit and independently readable."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute structural readiness from raw fixture rows."""

    fixture_rows = value.get("fixture_rows")
    causality_rows = value.get("causality_rows")
    if not isinstance(fixture_rows, Mapping) or not isinstance(causality_rows, list):
        return {"causal_update_ready_score": 0, "errors": ["raw_rows_missing"]}
    gradients = fixture_rows.get("gradient_checks")
    controls = fixture_rows.get("head_controls")
    if not isinstance(gradients, list) or not isinstance(controls, list):
        return {"causal_update_ready_score": 0, "errors": ["fixture_rows_invalid"]}
    errors: list[str] = []
    if not gradients or any(row.get("passed") is not True for row in gradients):
        errors.append("gradient_parity_failed")
    if not any(float(row.get("gradient_norm", 0.0)) > 0.0 for row in gradients):
        errors.append("nonzero_derivative_missing")
    local = [row for row in controls if row.get("arm") in {"local_brier", "local_log_loss"}]
    if len(local) != 2 or len({row.get("parameter_count") for row in local}) != 1:
        errors.append("matched_capacity_failed")
    if any(int(row.get("parameter_count", 257)) > 256 for row in local):
        errors.append("capacity_exceeded")
    static = [row for row in controls if row.get("arm") in {"frozen", "zero_step"}]
    if len(static) != 2 or any(row.get("state_changed") is not False for row in static):
        errors.append("static_control_moved")
    if len(causality_rows) != 2:
        errors.append("delay_rows_missing")
    for row in causality_rows:
        if row.get("future_access_violations") != 0:
            errors.append(f"future_access:{row.get('delay')}")
        if row.get("predict_before_update") is not True:
            errors.append(f"chronology:{row.get('delay')}")
    restart = fixture_rows.get("restart_check")
    if not isinstance(restart, Mapping) or restart.get("passed") is not True:
        errors.append("restart_parity_failed")
    invocations = value.get("invocation_counts")
    if invocations != ZERO_INVOCATION_COUNTS or value.get("model_invoked") is not False:
        errors.append("current_model_calls_nonzero")
    receipts = value.get("validation_receipts")
    if (
        isinstance(receipts, list)
        and receipts
        and any(row.get("exit_code") != 0 or row.get("timed_out") is True for row in receipts)
    ):
        errors.append("required_validation_failed")
    return {"causal_update_ready_score": int(not errors), "errors": errors}


def _reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind all artifact evidence except the checksum field itself."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain how each field prevents one class of evidence drift."""

    special = {
        "schema": "Versioned structure prevents a reader from applying the wrong contract.",
        "run_date": "The fixed run date prevents evidence from moving between milestones.",
        "MODEL_SPECS": "An empty list prevents historical model evidence from becoming a current load.",
        "model_specs": "The lowercase mirror prevents schema aliases from hiding a model load.",
        "model_invoked": "A false flag separates CPU fixture work from current inference.",
        "invocation_counts": "Balanced zero counters expose attempted or unfinished model work.",
        "inference_substrate": "The substrate names the CPU numerical work that produced the rows.",
        "causal_update_ready_score": "A bare structural score stays independent of fixture benefit.",
        "rows": "Per-delay units prevent one delay from hiding a chronology failure in an average.",
        "causality_rows": "Raw release and origin evidence exposes future-label access.",
        "fixture_rows": "Analytic witnesses keep oracle evidence separate from real benefit.",
        "checkpoint_schema": "Whole-machine state makes restart parity meaningful.",
    }
    return {
        field: special.get(
            field, f"The {field} field prevents silent omission or reinterpretation."
        )
        for field in fields
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, non-timeout receipt for every named command."""

    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def build_fixture_artifact(
    scratch_root: Path,
    *,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    validation_required: bool = False,
    terminal_required: bool = False,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    started_at_utc: str | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Build one complete artifact from raw analytic fixture evidence."""

    started = time.monotonic()
    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    checks, hashes = collect_preconditions(REPO_ROOT)
    selection_started = time.monotonic()
    selection = select_fixture_hyperparameters()
    controls = run_head_controls(selection)
    training_elapsed = time.monotonic() - selection_started
    gradients = _gradient_checks()
    events = fixture_events()
    delay_rows = [
        run_causal_fixture(events, delay=delay) for delay in (PRIMARY_DELAY, SECONDARY_DELAY)
    ]
    restarted = run_causal_fixture(
        events, delay=PRIMARY_DELAY, checkpoint_path=scratch / "restart-checkpoint.json"
    )
    primary = delay_rows[0]
    restart_passed = (
        restarted["stable_trace"] == primary["stable_trace"]
        and restarted["terminal_checkpoint_bytes"] == primary["terminal_checkpoint_bytes"]
    )
    causality_rows = [
        {
            "delay": row["delay"],
            "prediction_rows": row["predictions"],
            "release_rows": row["batches"],
            "update_rows": row["updates"],
            "future_access_violations": row["future_access_violations"],
            "predict_before_update": row["predict_before_update"],
            "terminal_state_hash": row["terminal_state_hash"],
        }
        for row in delay_rows
    ]
    rows = [
        {
            "unit_id": f"delay-{row['delay']}",
            "delay": row["delay"],
            "planned_events": len(events),
            "attempted_events": len(row["predictions"]),
            "completed_events": len(row["predictions"]),
            "excluded_events": 0,
            "failed_events": 0,
            "censored_events": 0,
            "released_batches": len(row["batches"]),
            "updates_per_arm": len(row["updates"]) // 2,
            "future_access_violations": row["future_access_violations"],
            "status": "complete",
            "failed": False,
            "censored": False,
        }
        for row in delay_rows
    ]
    required_names = (
        (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
        if terminal_required
        else validation_scope.REQUIRED_CHECK_NAMES
    )
    validation_passed = not validation_required or _receipts_pass(
        validation_receipts, required_names
    )
    max_error = max(float(row["max_abs_error"]) for row in gradients)
    nonzero = any(float(row["gradient_norm"]) > 0.0 for row in gradients)
    chronology_passed = all(
        row["future_access_violations"] == 0 and row["predict_before_update"] is True
        for row in causality_rows
    )
    static_passed = all(
        row["state_changed"] is False for row in controls if row["arm"] in {"frozen", "zero_step"}
    )
    gates = [
        _gate(
            "gradient_parity",
            "validity",
            2e-7,
            max_error,
            "<=",
            max_error <= 2e-7,
            "Finite differences expose an incorrect chain rule.",
        ),
        _gate(
            "nonzero_derivative",
            "validity",
            True,
            nonzero,
            "==",
            nonzero,
            "A nonzero witness prevents a vacuous all-zero gradient.",
        ),
        _gate(
            "causal_chronology",
            "validity",
            0,
            sum(int(row["future_access_violations"]) for row in causality_rows),
            "==",
            chronology_passed,
            "Zero violations prevent future or alternate-delay labels from entering updates.",
        ),
        _gate(
            "restart_parity",
            "validity",
            True,
            restart_passed,
            "==",
            restart_passed,
            "Byte parity proves the whole machine resumed together.",
        ),
        _gate(
            "static_controls",
            "validity",
            True,
            static_passed,
            "==",
            static_passed,
            "Frozen and zero-step controls expose hidden mutation.",
        ),
        _gate(
            "scoped_validation",
            "validity",
            True,
            validation_passed,
            "==",
            validation_passed,
            "Required commands can disqualify favorable fixture metrics.",
        ),
    ]
    first_failure = next((row for row in gates if row["passed"] is not True), None)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed = time.monotonic() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "status": "complete_circular_positive_causal_prototype_ready",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc or now,
        "completed_at_utc": now,
        "process_identity": {"pid": os.getpid(), "hostname": socket.gethostname()},
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_numpy_smooth_residual_fixture",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": max(elapsed, 1e-6),
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": training_elapsed,
            "validation": max(elapsed - training_elapsed, 0.0),
            "historical_capture": 0.0,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "fit": FIT_SEED,
            "arrival": ARRIVAL_SEED,
            "audit": AUDIT_SEED,
            "shuffle": SHUFFLE_SEED,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": hashes,
        "rows": rows,
        "sample_size_budget": {
            "planned": 32,
            "attempted": 32,
            "completed": 32,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "unit": "event_delay_pairs",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "passed": first_failure is None,
            "failed_checks": [row["check"] for row in gates if row["passed"] is not True],
            "first_failure": deepcopy(first_failure),
        },
        "honest_verdict": "complete_circular_positive_causal_prototype_ready_real_learning_deferred_to_exp7509",
        "verdict_class": "circular_positive" if first_failure is None else "disqualified",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_receipts": deepcopy([dict(row) for row in validation_receipts]),
        "causal_update_ready_score": int(first_failure is None),
        "update_equations": {
            "residual": "r=b*tanh(u/b)",
            "probability": "p=sigmoid(logit(p_base)+r)",
            "brier_gradient": "2*(p-y)*p*(1-p)*(1-tanh(u/b)^2)*basis",
            "log_loss_gradient": "(p-y)*(1-tanh(u/b)^2)*basis",
            "basis": "four_feature_open_cubic_spline_plus_bias",
            "maximum_coefficients": 256,
            "observed_local_coefficients": next(
                row["parameter_count"] for row in controls if row["arm"] == "local_brier"
            ),
            "learning_rate_candidates": list(LEARNING_RATES),
            "residual_bound_candidates": list(RESIDUAL_BOUNDS),
        },
        "causality_rows": causality_rows,
        "fixture_rows": {
            "gradient_checks": gradients,
            "head_controls": controls,
            "hyperparameter_selection": selection,
            "restart_check": {
                "passed": restart_passed,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "stable_trace_equal": restarted["stable_trace"] == primary["stable_trace"],
                "terminal_bytes_equal": restarted["terminal_checkpoint_bytes"]
                == primary["terminal_checkpoint_bytes"],
            },
            "claim_limit": "oracle_fixture_circular_positive_only",
        },
        "checkpoint_schema": {
            "name": CHECKPOINT_SCHEMA,
            "required_parts": ["models", "pending_queue", "audit_rng_state", "order_cursor"],
        },
        "small_ebm_training": {
            "scope": "current_cpu_work",
            "roles": ["training", "calibration_tuning"],
            "candidate_count": len(LEARNING_RATES) * len(RESIDUAL_BOUNDS),
            "optimizer_updates": len(LEARNING_RATES) * len(RESIDUAL_BOUNDS) * 3 * 16,
            "elapsed_s": training_elapsed,
            "budget_s": 300.0,
            "within_budget": training_elapsed <= 300.0,
        },
        "frozen_temperature_baseline": {
            "source": UPSTREAM_CHECKPOINT.as_posix(),
            "selected_temperature": 1.5,
            "mutable": False,
        },
        "release_batch_protocol": {
            "block_size": BLOCK_SIZE,
            "audit_probability": AUDIT_PROBABILITY,
            "audit_uses_scores": False,
            "audit_uses_labels": False,
            "primary_delay": PRIMARY_DELAY,
            "secondary_delay": SECONDARY_DELAY,
            "shuffle_scope": "same_released_batch_only",
            "future_label_access_allowed": False,
            "alternate_delay_label_access_allowed": False,
        },
        "historical_model_provenance": {
            "scope": "historical_only",
            "description": "Cached Qwen evidence came from upstream artifacts and was not loaded.",
            "current_model_calls": 0,
        },
        "affected_validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": bool(validation_required and validation_passed),
            "numbered_runtime_e2e_applicable": False,
        },
        "predictive_benefit_measured": False,
        "real_delayed_feedback_evaluation": "deferred_to_Exp7509",
        "retention_labels_used_for_selection_or_rollback": False,
        "importance_anchor_present": False,
        "expert_ensemble_present": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "external_publication_performed": False,
        "push_performed": False,
        "research_conductor_modified": False,
    }
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Fail closed when identity, evidence, sources, or reduction changes."""

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
        "verifier_is_oracle": True,
        "predictive_benefit_measured": False,
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
    if value.get("verdict_class") == "positive":
        errors.append("oracle_positive_forbidden")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_invalid")
    preconditions = value.get("preconditions_checked")
    if not isinstance(preconditions, list) or any(
        row.get("passed") is not True for row in preconditions
    ):
        errors.append("preconditions_failed")
    reduction = independent_reduce(value)
    if value.get("causal_update_ready_score") != reduction["causal_update_ready_score"]:
        errors.append("readiness_reduction_mismatch")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(not row.get("principle") for row in gates):
        errors.append("gate_principle_missing")
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


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and make malformed bytes fail closed."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Validate serialized evidence in a fresh-reader compatible path."""

    value = _load_object(path)
    if not value:
        return ["artifact_unreadable_or_not_object"]
    return validate_artifact(value, root=root, verify_sources=verify_sources)


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit a flushed boundary around each operation that can take time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7506] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed_units: int
) -> JsonDict:  # pragma: no cover
    """Close one measured phase with a monotonic checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
    }


def _terminal_commands(candidate: Path) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build fresh replay, reduction, adversarial, and strict row readers."""

    python = str(REPO_ROOT / ".venv/bin/python")
    common = ("--date", RUN_DATE)
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300.0,
        ),
    ]


def _blocked_artifact(
    failed: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    """Publish exact external absence without inventing dependent measurements."""

    reason = str(failed.get("check") or "external_prerequisite")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
        "preconditions_checked": deepcopy([dict(row) for row in rows]),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_precondition_check_only",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 1e-6,
        "phase_spans": [],
        "random_seed": {"fit": FIT_SEED, "arrival": ARRIVAL_SEED, "audit": AUDIT_SEED},
        "source_artifact_hashes": [],
        "rows": [],
        "sample_size_budget": {
            "planned": 32,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 32,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [reason],
            "first_failure": deepcopy(dict(failed)),
        },
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "causal_update_ready_score": 0,
    }
    value["field_principles"] = _field_principles(
        (*value.keys(), "field_principles", "reproducibility_checksum")
    )
    value["reproducibility_checksum"] = _reproducibility_checksum(value)
    return value


def run_experiment(
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover
    """Authenticate, measure, validate twice, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    run_started = time.monotonic()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "started")
    phase_started = time.monotonic()
    checks, _hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(checks)))
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = _blocked_artifact(failed, checks)
        progress(run_started, "publish", "before_atomic_terminal", verdict="blocked")
        atomic_json(destination, blocked)
        progress(run_started, "publish", f"complete_blocked_{failed.get('check')}")
        return blocked
    progress(run_started, "preconditions", "completed", checks=len(checks))

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
    scratch = Path(tempfile.mkdtemp(prefix="carnot-exp7506-fixture-", dir="/tmp"))
    progress(run_started, "fixture_computation", "before_training")
    phase_started = time.monotonic()
    provisional = build_fixture_artifact(scratch, phase_spans=spans, started_at_utc=started_at)
    spans.append(_span("fixture_computation", phase_started, run_started, 32))
    progress(run_started, "fixture_computation", "after_training", event_units=32)
    if independent_reduce(provisional)["causal_update_ready_score"] != 1:
        raise RuntimeError("fixture_structural_qualification_failed")

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7506-validation-", dir="/tmp"))
    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        VALIDATION_MANIFEST.test_paths,
        VALIDATION_MANIFEST.changed_modules,
        static_paths=VALIDATION_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7506",
    )
    progress(run_started, "affected_validation", "before_subprocesses", commands=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    affected_reduction = validation_scope.reduce_required_checks(affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["required_checks_passed"],
    )

    candidate = build_fixture_artifact(
        scratch,
        validation_receipts=affected,
        validation_required=True,
        phase_spans=spans,
        started_at_utc=started_at,
        duration_s=time.monotonic() - run_started,
    )
    candidate_path = raw_root / "measured_terminal_candidate.json"
    errors = validate_artifact(candidate, root=root)
    if errors:
        raise RuntimeError(f"candidate_invalid:{errors}")
    atomic_json(candidate_path, candidate)

    progress(run_started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    progress(run_started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_fixture_artifact(
        scratch,
        validation_receipts=[*affected, *terminal],
        validation_required=True,
        terminal_required=True,
        phase_spans=spans,
        started_at_utc=started_at,
        duration_s=time.monotonic() - run_started,
    )
    final_errors = validate_artifact(final, root=root)
    if final_errors:
        raise RuntimeError(f"final_candidate_invalid:{final_errors}")
    final_candidate = raw_root / "exact_terminal_candidate.json"
    atomic_json(final_candidate, final)

    progress(run_started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        _terminal_commands(final_candidate),
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(run_started, "exact_candidate_validation", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(run_started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        run_started,
        "publish",
        "complete",
        causal_update_ready_score=final["causal_update_ready_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and fresh-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one strict fresh-process reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    verify_sources = not bool(args.no_source_check)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, verify_sources=verify_sources)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        reduction = independent_reduce(value)
        errors = (
            validate_artifact(value, verify_sources=verify_sources)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({**reduction, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors) or reduction["causal_update_ready_score"] != 1)
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "result": RESULT_PATH.as_posix(),
                "causal_update_ready_score": artifact["causal_update_ready_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if artifact["verdict_class"] in {"circular_positive", "null"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
