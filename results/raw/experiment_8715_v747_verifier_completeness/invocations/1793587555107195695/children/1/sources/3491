"""Update two calibration scalars over a frozen Gibbs energy.

The adapter keeps the learned representation unchanged. It records each
prediction before a trusted label can be used, then accepts at most one
authorized update for that event.

Spec refs: REQ-REPORT-7397, SCENARIO-REPORT-7397-AUTHORITY,
SCENARIO-REPORT-7397-FROZEN, and SCENARIO-REPORT-7397-CONTROLS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
A_MIN = 0.25
A_MAX = 4.0
B_MIN = -8.0
B_MAX = 8.0
ONLINE_LEARNING_RATE = 0.01
GRADIENT_NORM_CAP = 1.0


class FutureLabelAccessError(ValueError):
    """Report an attempt to use a label before its declared release index."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _numeric_tree(value: Any) -> Any:
    """Copy a numeric JSON tree and reject executable or non-finite values."""

    if isinstance(value, Mapping):
        return {str(key): _numeric_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numeric_tree(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Gibbs weights must be a numeric JSON tree")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Gibbs weights must be finite")
    return result


def _validate_weights(weights: Mapping[str, Any]) -> JsonDict:
    required = {"w1", "b1", "w_out", "b_out"}
    if set(weights) != required:
        raise ValueError("Gibbs weights must contain w1, b1, w_out, and b_out")
    clean = _numeric_tree(weights)
    w1 = np.asarray(clean["w1"], dtype=np.float64)
    b1 = np.asarray(clean["b1"], dtype=np.float64)
    w_out = np.asarray(clean["w_out"], dtype=np.float64)
    if w1.ndim != 2 or w1.shape[1] != 2 or w1.shape[0] != 4:
        raise ValueError("Gibbs w1 must have shape 4 by 2")
    if b1.shape != (4,) or w_out.shape != (4,):
        raise ValueError("Gibbs hidden vectors must have length 4")
    return clean


def sigmoid(value: float) -> float:
    """Compute a stable scalar sigmoid for a finite numeric input."""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("sigmoid input must be finite")
    if numeric >= 0.0:
        return 1.0 / (1.0 + math.exp(-numeric))
    exponential = math.exp(numeric)
    return exponential / (1.0 + exponential)


def energy(weights: Mapping[str, Any], features: Sequence[float]) -> float:
    """Evaluate the frozen 2-4-1 SiLU Gibbs representation."""

    clean = _validate_weights(weights)
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("features must contain two finite values")
    hidden_linear = np.asarray(clean["w1"]) @ values + np.asarray(clean["b1"])
    hidden = hidden_linear / (1.0 + np.exp(-hidden_linear))
    result = float(np.asarray(clean["w_out"]) @ hidden + float(clean["b_out"]))
    if not math.isfinite(result):  # pragma: no cover - finite inputs normally guarantee this.
        raise ValueError("energy must be finite")
    return result


def affine_log_loss(energies: Sequence[float], labels: Sequence[int], a: float, b: float) -> float:
    """Return mean Bernoulli log loss for one affine calibration state."""

    if len(energies) != len(labels) or len(energies) == 0:
        raise ValueError("affine inputs must have one non-empty common length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("affine labels must be binary")
    probabilities = np.asarray([sigmoid(float(a) * float(value) + float(b)) for value in energies])
    targets = np.asarray(labels, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped)))


def fit_affine(
    energies: Sequence[float],
    labels: Sequence[int],
    *,
    steps: int = 500,
    learning_rate: float = 0.01,
) -> JsonDict:
    """Fit the bounded affine state on initialization labels only."""

    if len(energies) != len(labels) or not energies:
        raise ValueError("affine inputs must have one non-empty common length")
    if set(labels) != {0, 1}:
        raise ValueError("affine fitting data must contain both labels")
    if steps < 0 or steps > 500:
        raise ValueError("affine steps must be between zero and 500")
    values = np.asarray(energies, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("affine energies must be finite")
    a = 1.0
    b = 0.0
    curve: list[JsonDict] = []
    for step in range(steps):
        probabilities = np.asarray([sigmoid(a * value + b) for value in values])
        curve.append({"step": step, "loss": affine_log_loss(values, labels, a, b)})
        residual = probabilities - targets
        gradient_a = float(np.mean(residual * values))
        gradient_b = float(np.mean(residual))
        a = min(A_MAX, max(A_MIN, a - learning_rate * gradient_a))
        b = min(B_MAX, max(B_MIN, b - learning_rate * gradient_b))
    curve.append({"step": steps, "loss": affine_log_loss(values, labels, a, b)})
    return {
        "a": a,
        "b": b,
        "update_count": steps,
        "loss_curve": curve,
        "bounds": {"a": [A_MIN, A_MAX], "b": [B_MIN, B_MAX]},
        "fitting_role": "frozen_initialization_groups_only",
    }


class DelayedEnergyCalibrator:
    """Commit one bounded affine update for each newly verified event."""

    def __init__(self, weights: Mapping[str, Any], *, a: float = 1.0, b: float = 0.0) -> None:
        self._weights = _validate_weights(weights)
        self._weights_hash = _canonical_hash(self._weights)
        self._initial_a = self._bounded_a(a)
        self._initial_b = self._bounded_b(b)
        self.a = self._initial_a
        self.b = self._initial_b
        self.update_count = 0
        self._predictions: dict[str, JsonDict] = {}
        self._feedback: dict[str, JsonDict] = {}
        self._commit_order: list[str] = []

    @staticmethod
    def _bounded_a(value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("affine state must be finite")
        if not A_MIN <= numeric <= A_MAX:
            raise ValueError("affine a must be within its projection bounds")
        return numeric

    @staticmethod
    def _bounded_b(value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("affine state must be finite")
        if not B_MIN <= numeric <= B_MAX:
            raise ValueError("affine b must be within its projection bounds")
        return numeric

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    @property
    def state_hash(self) -> str:
        return _canonical_hash(self.numeric_state())

    def numeric_state(self) -> JsonDict:
        """Return only durable numeric state used by later predictions."""

        return {
            "a": self.a,
            "b": self.b,
            "update_count": self.update_count,
            "weights_hash": self._weights_hash,
        }

    def probability(self, features: Sequence[float]) -> tuple[float, float]:
        """Return frozen energy and its current calibrated error probability."""

        value = energy(self._weights, features)
        return value, sigmoid(self.a * value + self.b)

    def record_prediction(
        self,
        event_id: str,
        features: Sequence[float],
        *,
        prediction_index: int,
        feedback_available_at: int,
        label_authority: str,
    ) -> JsonDict:
        """Seal an immutable prediction without storing its future label."""

        identity = str(event_id)
        if identity in self._predictions:
            raise ValueError("event prediction already exists")
        if int(feedback_available_at) <= int(prediction_index):
            raise ValueError("feedback must become available after prediction")
        if not label_authority:
            raise ValueError("label authority is required")
        state = self.numeric_state()
        value, probability = self.probability(features)
        record = {
            "event_id": identity,
            "features": [float(item) for item in features],
            "energy": value,
            "probability": probability,
            "prediction_index": int(prediction_index),
            "feedback_available_at": int(feedback_available_at),
            "label_authority": str(label_authority),
            "numeric_state": state,
            "state_hash": _canonical_hash(state),
            "prediction_before_feedback": True,
        }
        self._predictions[identity] = deepcopy(record)
        return deepcopy(record)

    def _apply_update(self, event_id: str, label: int) -> JsonDict:
        prediction = self._predictions[event_id]
        before_hash = self.state_hash
        event_energy = float(prediction["energy"])
        probability_at_update = sigmoid(self.a * event_energy + self.b)
        residual = probability_at_update - label
        raw_a = residual * event_energy
        raw_b = residual
        norm = math.hypot(raw_a, raw_b)
        scale = min(1.0, GRADIENT_NORM_CAP / max(norm, 1e-15))
        gradient_a = raw_a * scale
        gradient_b = raw_b * scale
        self.a = min(A_MAX, max(A_MIN, self.a - ONLINE_LEARNING_RATE * gradient_a))
        self.b = min(B_MAX, max(B_MIN, self.b - ONLINE_LEARNING_RATE * gradient_b))
        self.update_count += 1
        return {
            "state_hash_before": before_hash,
            "state_hash_after": self.state_hash,
            "numeric_state_after": self.numeric_state(),
            "gradient": [gradient_a, gradient_b],
            "gradient_norm_before_clip": norm,
            "gradient_norm_after_clip": math.hypot(gradient_a, gradient_b),
            "learning_rate": ONLINE_LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "probability_at_update": probability_at_update,
            "gibbs_weights_unchanged": self._weights_hash
            == prediction["numeric_state"]["weights_hash"],
        }

    def commit_feedback(self, event_id: str, label: int | None, *, visible_at: int) -> JsonDict:
        """Use one label only after its authority boundary has opened."""

        identity = str(event_id)
        prediction = self._predictions.get(identity)
        if prediction is None:
            return {"event_id": identity, "status": "unknown_event", "update_admitted": False}
        if identity in self._feedback and self._feedback[identity].get("active") is True:
            return {"event_id": identity, "status": "duplicate", "update_admitted": False}
        if int(visible_at) < int(prediction["feedback_available_at"]):
            raise FutureLabelAccessError("feedback is not available at this index")
        if label is None:
            return {"event_id": identity, "status": "missing", "update_admitted": False}
        if label not in (0, 1):
            raise ValueError("feedback label must be binary")
        update = self._apply_update(identity, int(label))
        record = {
            "event_id": identity,
            "label": int(label),
            "visible_at": int(visible_at),
            "label_authority": prediction["label_authority"],
            "status": "committed",
            "update_admitted": True,
            "active": True,
            "commit_sequence": len(self._commit_order),
            **update,
        }
        self._feedback[identity] = deepcopy(record)
        self._commit_order.append(identity)
        return deepcopy(record)

    def erase_feedback(self, event_id: str) -> JsonDict:
        """Rebuild affine state without one update while keeping its audit record."""

        identity = str(event_id)
        record = self._feedback.get(identity)
        if record is None or record.get("active") is not True:
            return {"event_id": identity, "status": "unknown_event", "update_admitted": False}
        record["active"] = False
        record["erased"] = True
        self.a = self._initial_a
        self.b = self._initial_b
        self.update_count = 0
        for committed_id in self._commit_order:
            committed = self._feedback[committed_id]
            if committed.get("active") is True:
                update = self._apply_update(committed_id, int(committed["label"]))
                committed.update(update)
        return {
            "event_id": identity,
            "status": "erased",
            "update_admitted": False,
            "state_hash_after": self.state_hash,
        }

    def to_dict(self) -> JsonDict:
        """Return a code-free checkpoint for a fresh-process restart."""

        return {
            "weights": deepcopy(self._weights),
            "weights_hash": self._weights_hash,
            "initial_a": self._initial_a,
            "initial_b": self._initial_b,
            "a": self.a,
            "b": self.b,
            "update_count": self.update_count,
            "predictions": [deepcopy(self._predictions[key]) for key in self._predictions],
            "feedback": [deepcopy(self._feedback[key]) for key in self._feedback],
            "commit_order": list(self._commit_order),
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DelayedEnergyCalibrator:
        """Restore a checkpoint and reject any changed numeric identity."""

        adapter = cls(value["weights"], a=float(value["initial_a"]), b=float(value["initial_b"]))
        if adapter.weights_hash != value.get("weights_hash"):
            raise ValueError("Gibbs weight hash changed during restart")
        adapter.a = adapter._bounded_a(float(value["a"]))
        adapter.b = adapter._bounded_b(float(value["b"]))
        adapter.update_count = int(value["update_count"])
        adapter._predictions = {
            str(row["event_id"]): deepcopy(dict(row)) for row in value.get("predictions", [])
        }
        adapter._feedback = {
            str(row["event_id"]): deepcopy(dict(row)) for row in value.get("feedback", [])
        }
        adapter._commit_order = [str(item) for item in value.get("commit_order", [])]
        if adapter.state_hash != value.get("state_hash"):
            raise ValueError("affine state hash changed during restart")
        return adapter
