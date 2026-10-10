"""Prototype a bounded additive cubic-spline energy classifier.

The classifier is logistic regression on a fixed B-spline design matrix. This
module uses analytic fixtures to check numeric and update behavior. It does not
measure corpus efficacy. Spec refs: REQ-KAN-7425 and SCENARIO-KAN-7425-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
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
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7425-spline-prototype"
SCHEMA = "carnot.exp7425.v651.spline_prototype.v1"
CHECKPOINT_SCHEMA = "carnot.exp7425.spline_checkpoint.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7425_v651_spline_prototype.json")
RAW_DIR = Path("results/raw/experiment_7425_v651_spline_prototype")
MODULE_PATH = Path("python/carnot/experiment_7425_v651_spline_prototype.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7425_v651_spline_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7425_v651_spline_prototype.py")
SPEC_PATH = Path("openspec/capabilities/kan/spec.md")

INPUT_COUNT = 6
COEFFICIENTS_PER_INPUT = 8
DEGREE = 3
KNOT_VECTOR_SIZE = COEFFICIENTS_PER_INPUT + DEGREE + 1
PARAMETER_COUNT = INPUT_COUNT * COEFFICIENTS_PER_INPUT + 1
LEARNING_RATE = 0.01
GRADIENT_NORM_CAP = 1.0
PARITY_TOLERANCE = 1e-10
MAX_CHECKPOINT_BYTES = 64 * 1024
TRAINING_SEEDS = (651, 652)

INPUT_PATHS = (
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
    Path("python/carnot/models/kan/__init__.py"),
    Path("python/carnot/models/pwa_kan.py"),
    Path("python/carnot/models/kan_cl.py"),
    Path("python/carnot/experiment_7413_v650_source_calibration.py"),
    Path("python/carnot/experiment_7414_v650_selected_feedback.py"),
    SPEC_PATH,
)

V651_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
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
    "spline_prototype_ready_score",
    "energy_definition",
    "parity_rows",
    "update_footprint",
    "small_ebm_training",
)


@dataclass(frozen=True)
class BasisEvaluation:
    """Hold one normalized local basis and its clamp observation."""

    values: np.ndarray
    active_indices: tuple[int, ...]
    clamped: bool
    original_value: float
    evaluated_value: float


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return a real UTC timestamp for one artifact boundary."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - real execution boundary.
    """Print a flushed phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7425] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _finite_array(value: Any, *, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    """Convert one numeric value to float64 and reject unsafe shapes or values."""

    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _repair_breakpoints(raw: np.ndarray) -> np.ndarray:
    """Make repeated quantile breakpoints strict with a deterministic rule."""

    values = _finite_array(raw, shape=(6,), name="quantile breakpoints").copy()
    scale = max(1.0, float(np.max(np.abs(values))))
    if values[-1] == values[0]:
        half_width = scale * 5e-10
        return np.linspace(values[0] - half_width, values[0] + half_width, 6)
    gap = max((values[-1] - values[0]) * 1e-12, scale * 1e-15)
    repaired = values.copy()
    for index in range(1, len(repaired) - 1):
        repaired[index] = max(repaired[index], repaired[index - 1] + gap)
    if repaired[-2] >= repaired[-1]:
        spacing = float(np.spacing(scale))
        start = min(float(values[0]), float(values[-1]) - 5.0 * spacing)
        repaired = np.asarray([start], dtype=np.float64)
        for _index in range(5):
            repaired = np.append(repaired, np.nextafter(repaired[-1], math.inf))
    return repaired


def fit_quantile_knots(training_features: Any) -> np.ndarray:
    """Freeze six open-clamped cubic knot vectors from training quantiles only."""

    training = np.asarray(training_features, dtype=np.float64)
    if training.ndim != 2 or training.shape[1] != INPUT_COUNT:
        raise ValueError("training features must be a matrix with six columns")
    if training.shape[0] < 2 or not np.all(np.isfinite(training)):
        raise ValueError("training features must contain finite values and at least two rows")
    quantiles = np.linspace(0.0, 1.0, 6)
    vectors = []
    for feature in range(INPUT_COUNT):
        raw = np.quantile(training[:, feature], quantiles, method="linear")
        breaks = _repair_breakpoints(np.asarray(raw, dtype=np.float64))
        vectors.append(
            np.concatenate(
                (
                    np.repeat(breaks[0], DEGREE + 1),
                    breaks[1:-1],
                    np.repeat(breaks[-1], DEGREE + 1),
                )
            )
        )
    return np.asarray(vectors, dtype=np.float64)


def cubic_basis(value: float, knots: Any) -> BasisEvaluation:
    """Evaluate one cubic basis with explicit endpoint and clamp handling."""

    vector = _finite_array(knots, shape=(KNOT_VECTOR_SIZE,), name="knot vector")
    if np.any(np.diff(vector) < 0.0) or vector[DEGREE] >= vector[-DEGREE - 1]:
        raise ValueError("knot vector must have ordered nondegenerate support")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("feature value must be finite")
    lower = float(vector[DEGREE])
    upper = float(vector[-DEGREE - 1])
    evaluated = min(max(numeric, lower), upper)
    clamped = evaluated != numeric
    if evaluated == lower:
        basis = np.zeros(COEFFICIENTS_PER_INPUT, dtype=np.float64)
        basis[0] = 1.0
    elif evaluated == upper:
        basis = np.zeros(COEFFICIENTS_PER_INPUT, dtype=np.float64)
        basis[-1] = 1.0
    else:
        current = ((vector[:-1] <= evaluated) & (evaluated < vector[1:])).astype(np.float64)
        for degree in range(1, DEGREE + 1):
            next_basis = np.zeros(len(vector) - degree - 1, dtype=np.float64)
            for index in range(len(next_basis)):
                left_width = vector[index + degree] - vector[index]
                right_width = vector[index + degree + 1] - vector[index + 1]
                left = (
                    0.0
                    if left_width == 0.0
                    else ((evaluated - vector[index]) * current[index] / left_width)
                )
                right = (
                    0.0
                    if right_width == 0.0
                    else (
                        (vector[index + degree + 1] - evaluated) * current[index + 1] / right_width
                    )
                )
                next_basis[index] = left + right
            current = next_basis
        basis = current
    active = tuple(int(index) for index in np.flatnonzero(basis != 0.0))
    return BasisEvaluation(basis, active, clamped, numeric, evaluated)


def dense_design_vector(features: Any, knots: Any) -> tuple[np.ndarray, JsonDict]:
    """Build the 48-value fixed spline design vector for one six-input row."""

    values = _finite_array(features, shape=(INPUT_COUNT,), name="input with six features")
    knot_matrix = _finite_array(knots, shape=(INPUT_COUNT, KNOT_VECTOR_SIZE), name="knot matrix")
    parts: list[np.ndarray] = []
    active: list[int] = []
    clamp_count = 0
    for feature, numeric in enumerate(values):
        evaluated = cubic_basis(float(numeric), knot_matrix[feature])
        parts.append(evaluated.values)
        active.extend(
            feature * COEFFICIENTS_PER_INPUT + index for index in evaluated.active_indices
        )
        clamp_count += int(evaluated.clamped)
    return np.concatenate(parts), {
        "active_indices": active,
        "active_coefficient_count": len(active),
        "clamped_feature_count": clamp_count,
    }


def sigmoid(value: float) -> float:
    """Evaluate a stable scalar logistic probability."""

    numeric = float(value)
    if numeric >= 0.0:
        return 1.0 / (1.0 + math.exp(-numeric))
    exponential = math.exp(numeric)
    return exponential / (1.0 + exponential)


def _event_values(event: Mapping[str, Any]) -> tuple[str, str, int, np.ndarray, int, str]:
    """Validate immutable event identity and return its canonical fingerprint."""

    event_id = str(event.get("event_id") or "")
    source_version = str(event.get("source_version") or "")
    if not event_id or not source_version:
        raise ValueError("event ID and source version must be nonempty")
    reveal = event.get("reveal_time")
    if not isinstance(reveal, int) or isinstance(reveal, bool) or reveal < 0:
        raise ValueError("reveal time must be a nonnegative integer")
    label = event.get("label")
    if label not in (0, 1) or isinstance(label, bool):
        raise ValueError("feedback label must be binary")
    features = _finite_array(event.get("features"), shape=(INPUT_COUNT,), name="feedback features")
    sealed = {
        "event_id": event_id,
        "source_version": source_version,
        "reveal_time": reveal,
        "features": features.tolist(),
        "label": int(label),
    }
    return event_id, source_version, reveal, features, int(label), canonical_hash(sealed)


class SplineEnergyHead:
    """Store a fixed spline basis and apply journaled sparse logistic updates."""

    def __init__(
        self,
        knots: Any,
        coefficients: Any,
        bias: float,
        *,
        initial_coefficients: Any | None = None,
        initial_bias: float | None = None,
    ) -> None:
        self.knots = _finite_array(
            knots, shape=(INPUT_COUNT, KNOT_VECTOR_SIZE), name="knot matrix"
        ).copy()
        for row in self.knots:
            cubic_basis(float(row[DEGREE]), row)
        self.coefficients = _finite_array(
            coefficients,
            shape=(INPUT_COUNT, COEFFICIENTS_PER_INPUT),
            name="coefficients",
        ).copy()
        self.bias = float(bias)
        if not math.isfinite(self.bias):
            raise ValueError("bias must be finite")
        self._initial_coefficients = _finite_array(
            self.coefficients if initial_coefficients is None else initial_coefficients,
            shape=(INPUT_COUNT, COEFFICIENTS_PER_INPUT),
            name="initial coefficients",
        ).copy()
        self._initial_bias = self.bias if initial_bias is None else float(initial_bias)
        if not math.isfinite(self._initial_bias):
            raise ValueError("initial bias must be finite")
        self._event_fingerprints: dict[str, str] = {}
        self._events: dict[str, JsonDict] = {}
        self._commit_order: list[str] = []
        self._revoked: set[str] = set()
        self.update_count = 0

    @classmethod
    def from_training(cls, training_features: Any, *, seed: int = 651) -> SplineEnergyHead:
        """Create a deterministic small head without loading a generator model."""

        knots = fit_quantile_knots(training_features)
        rng = np.random.default_rng(int(seed))
        coefficients = rng.normal(0.0, 0.025, size=(INPUT_COUNT, COEFFICIENTS_PER_INPUT))
        return cls(knots, coefficients.astype(np.float64), 0.0)

    @property
    def parameter_count(self) -> int:
        """Return the fixed 48 coefficient plus one bias count."""

        return int(self.coefficients.size + 1)

    @property
    def state_vector(self) -> np.ndarray:
        """Return a detached dense state for parity checks."""

        return np.concatenate((self.coefficients.reshape(-1), np.asarray([self.bias])))

    def predict(self, features: Any) -> JsonDict:
        """Read the current predictor without opening or committing feedback."""

        design, support = dense_design_vector(features, self.knots)
        logit = float(design @ self.coefficients.reshape(-1) + self.bias)
        probability = sigmoid(logit)
        return {
            "logit": logit,
            "probability": probability,
            "energy_y0": 0.0,
            "energy_y1": -logit,
            **support,
            "predictor_state_hash": canonical_hash(self.state_vector.tolist()),
        }

    def loss_and_gradient(self, features: Any, label: int) -> tuple[float, np.ndarray, JsonDict]:
        """Return Bernoulli loss and its dense 49-value float64 gradient."""

        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("label must be binary")
        design, support = dense_design_vector(features, self.knots)
        logit = float(design @ self.coefficients.reshape(-1) + self.bias)
        loss = float(np.logaddexp(0.0, logit) - int(label) * logit)
        residual = sigmoid(logit) - int(label)
        gradient = np.concatenate((residual * design, np.asarray([residual])))
        return loss, gradient.astype(np.float64), support

    def _apply_event(self, event: Mapping[str, Any]) -> JsonDict:
        """Apply one already-admitted update to the active local coefficients."""

        _event_id, _source, _reveal, features, label, _fingerprint = _event_values(event)
        loss, gradient, support = self.loss_and_gradient(features, label)
        raw_norm = float(np.linalg.norm(gradient))
        scale = min(1.0, GRADIENT_NORM_CAP / max(raw_norm, np.finfo(np.float64).tiny))
        clipped = gradient * scale
        flat = self.coefficients.reshape(-1)
        active = np.asarray(support["active_indices"], dtype=np.int64)
        flat[active] -= LEARNING_RATE * clipped[active]
        self.bias -= LEARNING_RATE * float(clipped[-1])
        self.update_count += 1
        touched = int(len(active) + 1)
        return {
            "loss": loss,
            "gradient_norm_before_clip": raw_norm,
            "gradient_norm_after_clip": float(np.linalg.norm(clipped)),
            "learning_rate": LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "global_weight_decay": 0.0,
            "active_coefficient_count": int(len(active)),
            "touched_parameter_count": touched,
            "coefficient_write_bytes": touched * np.dtype(np.float64).itemsize,
            "arithmetic_operation_count": 4 * int(len(active)) + 8,
        }

    def commit_feedback(self, event: Mapping[str, Any], *, visible_at: int) -> JsonDict:
        """Admit one revealed immutable event and reject every silent repeat."""

        event_id, source, reveal, features, label, fingerprint = _event_values(event)
        existing = self._event_fingerprints.get(event_id)
        if existing is not None and existing != fingerprint:
            return {"event_id": event_id, "status": "identity_conflict", "update_admitted": False}
        if event_id in self._revoked:
            return {"event_id": event_id, "status": "revoked", "update_admitted": False}
        if existing is not None:
            return {"event_id": event_id, "status": "duplicate", "update_admitted": False}
        if int(visible_at) < reveal:
            return {"event_id": event_id, "status": "not_revealed", "update_admitted": False}
        before = canonical_hash(self.state_vector.tolist())
        sealed = {
            "event_id": event_id,
            "source_version": source,
            "reveal_time": reveal,
            "features": features.tolist(),
            "label": label,
        }
        update = self._apply_event(sealed)
        self._event_fingerprints[event_id] = fingerprint
        self._events[event_id] = sealed
        self._commit_order.append(event_id)
        return {
            "event_id": event_id,
            "source_version": source,
            "reveal_time": reveal,
            "visible_at": int(visible_at),
            "status": "committed",
            "update_admitted": True,
            "state_hash_before": before,
            "state_hash_after": canonical_hash(self.state_vector.tolist()),
            **update,
        }

    def revoke_feedback(self, event_id: str) -> JsonDict:
        """Remove one committed event and rebuild from the sealed initial state."""

        identity = str(event_id)
        if identity not in self._events or identity in self._revoked:
            return {"event_id": identity, "status": "unknown_event", "update_admitted": False}
        self._revoked.add(identity)
        self.coefficients = self._initial_coefficients.copy()
        self.bias = self._initial_bias
        self.update_count = 0
        for committed_id in self._commit_order:
            if committed_id not in self._revoked:
                self._apply_event(self._events[committed_id])
        return {
            "event_id": identity,
            "status": "revoked",
            "update_admitted": False,
            "journal_replayed": True,
            "state_hash_after": canonical_hash(self.state_vector.tolist()),
        }

    def _checkpoint_payload(self) -> JsonDict:
        """Serialize numeric state plus the small immutable feedback journal."""

        return {
            "schema": CHECKPOINT_SCHEMA,
            "knots": self.knots.tolist(),
            "coefficients": self.coefficients.tolist(),
            "bias": self.bias,
            "initial_coefficients": self._initial_coefficients.tolist(),
            "initial_bias": self._initial_bias,
            "events": [deepcopy(self._events[event_id]) for event_id in self._commit_order],
            "event_fingerprints": deepcopy(self._event_fingerprints),
            "commit_order": list(self._commit_order),
            "revoked_event_ids": sorted(self._revoked),
            "update_count": self.update_count,
        }

    @property
    def checkpoint_hash(self) -> str:
        """Hash the exact code-free state that a cold reader reconstructs."""

        return canonical_hash(self._checkpoint_payload())

    def save_checkpoint(self, path: Path, *, interrupt_before_replace: bool = False) -> JsonDict:
        """Atomically replace a bounded numeric checkpoint after syncing its bytes."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()
        payload["checkpoint_sha256"] = canonical_hash(payload)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size > MAX_CHECKPOINT_BYTES:
            temporary.unlink()
            raise ValueError("numeric checkpoint exceeds 64 KiB")
        if interrupt_before_replace:
            raise RuntimeError("simulated interruption before atomic checkpoint replace")
        os.replace(temporary, target)
        return {
            "path": str(target),
            "sha256": sha256_file(target),
            "state_sha256": self.checkpoint_hash,
            "byte_size": target.stat().st_size,
        }

    @classmethod
    def load_checkpoint(cls, path: Path) -> SplineEnergyHead:
        """Restore the committed target and ignore any uncommitted temporary file."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint is not readable JSON") from error
        if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema is invalid")
        knots = _finite_array(
            value.get("knots"), shape=(INPUT_COUNT, KNOT_VECTOR_SIZE), name="checkpoint knots"
        )
        coefficients = _finite_array(
            value.get("coefficients"),
            shape=(INPUT_COUNT, COEFFICIENTS_PER_INPUT),
            name="checkpoint coefficients",
        )
        initial = _finite_array(
            value.get("initial_coefficients"),
            shape=(INPUT_COUNT, COEFFICIENTS_PER_INPUT),
            name="checkpoint initial coefficients",
        )
        payload = {key: deepcopy(item) for key, item in value.items() if key != "checkpoint_sha256"}
        if value.get("checkpoint_sha256") != canonical_hash(payload):
            raise ValueError("checkpoint hash mismatch")
        restored = cls(
            knots,
            coefficients,
            float(value["bias"]),
            initial_coefficients=initial,
            initial_bias=float(value["initial_bias"]),
        )
        restored._events = {
            str(row["event_id"]): deepcopy(dict(row)) for row in value.get("events", [])
        }
        restored._event_fingerprints = {
            str(key): str(item) for key, item in dict(value.get("event_fingerprints") or {}).items()
        }
        restored._commit_order = [str(item) for item in value.get("commit_order", [])]
        restored._revoked = {str(item) for item in value.get("revoked_event_ids", [])}
        restored.update_count = int(value.get("update_count", 0))
        if restored.checkpoint_hash != value.get("checkpoint_sha256"):
            raise ValueError("checkpoint state reconstruction mismatch")
        return restored


class DenseSplineReference:
    """Apply the same update through a full fixed-basis logistic vector."""

    def __init__(self, knots: Any, state_vector: Any) -> None:
        self.knots = _finite_array(
            knots, shape=(INPUT_COUNT, KNOT_VECTOR_SIZE), name="dense knot matrix"
        ).copy()
        self._state = _finite_array(
            state_vector, shape=(PARAMETER_COUNT,), name="dense state"
        ).copy()
        self._fingerprints: dict[str, str] = {}

    @classmethod
    def from_head(cls, head: SplineEnergyHead) -> DenseSplineReference:
        """Copy one sparse state so parity starts from identical bytes."""

        return cls(head.knots, head.state_vector)

    @property
    def state_vector(self) -> np.ndarray:
        """Return a detached copy of the dense reference state."""

        return self._state.copy()

    def predict(self, features: Any) -> JsonDict:
        """Evaluate the same fixed design matrix with dense parameters."""

        design, support = dense_design_vector(features, self.knots)
        logit = float(design @ self._state[:-1] + self._state[-1])
        return {"logit": logit, "probability": sigmoid(logit), **support}

    def commit_feedback(self, event: Mapping[str, Any], *, visible_at: int) -> JsonDict:
        """Apply one full-vector update with the sparse path's exact arithmetic."""

        event_id, _source, reveal, features, label, fingerprint = _event_values(event)
        if event_id in self._fingerprints:
            status = (
                "duplicate" if self._fingerprints[event_id] == fingerprint else "identity_conflict"
            )
            return {"event_id": event_id, "status": status, "update_admitted": False}
        if int(visible_at) < reveal:
            return {"event_id": event_id, "status": "not_revealed", "update_admitted": False}
        design, support = dense_design_vector(features, self.knots)
        logit = float(design @ self._state[:-1] + self._state[-1])
        loss = float(np.logaddexp(0.0, logit) - label * logit)
        residual = sigmoid(logit) - label
        gradient = np.concatenate((residual * design, np.asarray([residual])))
        raw_norm = float(np.linalg.norm(gradient))
        scale = min(1.0, GRADIENT_NORM_CAP / max(raw_norm, np.finfo(np.float64).tiny))
        clipped = gradient * scale
        self._state -= LEARNING_RATE * clipped
        self._fingerprints[event_id] = fingerprint
        return {
            "event_id": event_id,
            "status": "committed",
            "update_admitted": True,
            "loss": loss,
            "gradient_norm_before_clip": raw_norm,
            "gradient_norm_after_clip": float(np.linalg.norm(clipped)),
            "active_coefficient_count": support["active_coefficient_count"],
        }


def finite_difference_gradient(
    head: SplineEnergyHead, features: Any, label: int, *, epsilon: float
) -> np.ndarray:
    """Compute a centered numeric gradient without changing predictor state."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    design, _support = dense_design_vector(features, head.knots)
    state = head.state_vector

    def loss(candidate: np.ndarray) -> float:
        logit = float(design @ candidate[:-1] + candidate[-1])
        return float(np.logaddexp(0.0, logit) - label * logit)

    numeric = np.zeros_like(state)
    for index in range(len(state)):
        positive = state.copy()
        negative = state.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        numeric[index] = (loss(positive) - loss(negative)) / (2.0 * epsilon)
    return numeric


def _analytic_training_fixture() -> np.ndarray:
    """Return analytic values that exercise duplicate quantile handling."""

    return np.asarray(
        [
            [0.0, 2.0, -1.0, 0.0, 1.0, 4.0],
            [0.0, 2.0, -0.5, 0.2, 1.0, 3.0],
            [0.0, 2.0, 0.0, 0.4, 1.0, 2.0],
            [0.5, 2.0, 0.5, 0.6, 1.0, 1.0],
            [0.5, 2.0, 1.0, 0.8, 1.0, 0.0],
            [1.0, 2.0, 1.5, 1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )


def _analytic_events() -> list[JsonDict]:
    """Return sealed analytic fixtures, never corpus observations."""

    values = (
        ([0.2, 2.0, 0.1, 0.7, 1.0, 2.5], 1),
        ([0.8, 2.0, 1.1, 0.3, 1.0, 0.5], 0),
        ([-1.0, 2.0, -0.8, 0.9, 1.0, 5.0], 1),
        ([0.4, 2.0, 0.4, 0.5, 1.0, 1.5], 0),
    )
    return [
        {
            "event_id": f"analytic-{index}",
            "source_version": "analytic-fixture-v1",
            "reveal_time": index,
            "features": features,
            "label": label,
        }
        for index, (features, label) in enumerate(values, start=1)
    ]


def run_analytic_fixtures(
    *, checkpoint_dir: Path | None = None
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Run deterministic numeric controls and return only raw fixture evidence."""

    training = _analytic_training_fixture()
    knots = fit_quantile_knots(training)
    partition_gaps = []
    active_counts = []
    for feature in range(INPUT_COUNT):
        points = np.linspace(knots[feature, DEGREE], knots[feature, -DEGREE - 1], 17)
        for point in points:
            evaluated = cubic_basis(float(point), knots[feature])
            partition_gaps.append(abs(float(np.sum(evaluated.values)) - 1.0))
            active_counts.append(len(evaluated.active_indices))

    duplicate_knots = np.array_equal(knots, fit_quantile_knots(training))
    probe = SplineEnergyHead.from_training(training, seed=651)
    x = np.asarray([0.3, 2.0, 0.2, 0.65, 1.0, 1.8], dtype=np.float64)
    _loss, analytic_gradient, _support = probe.loss_and_gradient(x, 1)
    numeric_gradient = finite_difference_gradient(probe, x, 1, epsilon=1e-6)
    finite_gap = float(np.max(np.abs(analytic_gradient - numeric_gradient)))

    rows: list[JsonDict] = []
    write_bytes: list[int] = []
    operations: list[int] = []
    for seed in TRAINING_SEEDS:
        sparse = SplineEnergyHead.from_training(training, seed=seed)
        dense = DenseSplineReference.from_head(sparse)
        for step, event in enumerate(_analytic_events(), start=1):
            sparse_receipt = sparse.commit_feedback(event, visible_at=int(event["reveal_time"]))
            dense_receipt = dense.commit_feedback(event, visible_at=int(event["reveal_time"]))
            sparse_probability = sparse.predict(event["features"])["probability"]
            dense_probability = dense.predict(event["features"])["probability"]
            write_bytes.append(int(sparse_receipt["coefficient_write_bytes"]))
            operations.append(int(sparse_receipt["arithmetic_operation_count"]))
            rows.append(
                {
                    "comparative_unit": f"analytic-fixture:seed-{seed}:step-{step}",
                    "fixture_id": str(event["event_id"]),
                    "data_role": "analytic_fixture",
                    "seed": seed,
                    "step": step,
                    "condition": "sparse_vs_dense_fixed_basis_logistic",
                    "status": "completed",
                    "label": event["label"],
                    "sparse_probability": sparse_probability,
                    "dense_probability": dense_probability,
                    "probability_abs_gap": abs(sparse_probability - dense_probability),
                    "parameter_linf_gap": float(
                        np.max(np.abs(sparse.state_vector - dense.state_vector))
                    ),
                    "active_coefficient_count": sparse_receipt["active_coefficient_count"],
                    "touched_parameter_count": sparse_receipt["touched_parameter_count"],
                    "coefficient_write_bytes": sparse_receipt["coefficient_write_bytes"],
                    "arithmetic_operation_count": sparse_receipt["arithmetic_operation_count"],
                    "bernoulli_log_loss": sparse_receipt["loss"],
                    "dense_bernoulli_log_loss": dense_receipt["loss"],
                }
            )

    directory = checkpoint_dir or Path(tempfile.mkdtemp(prefix="exp7425-checkpoint-"))
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_head = SplineEnergyHead.from_training(training, seed=651)
    checkpoint_head.commit_feedback(_analytic_events()[0], visible_at=1)
    checkpoint_path = directory / "numeric-checkpoint.json"
    checkpoint_manifest = checkpoint_head.save_checkpoint(checkpoint_path)
    restored = SplineEnergyHead.load_checkpoint(checkpoint_path)
    before_interruption = restored.checkpoint_hash
    restored.commit_feedback(_analytic_events()[1], visible_at=2)
    interrupted = False
    try:
        restored.save_checkpoint(checkpoint_path, interrupt_before_replace=True)
    except RuntimeError:
        interrupted = True
    committed_after_interruption = SplineEnergyHead.load_checkpoint(checkpoint_path)

    duplicate = SplineEnergyHead.from_training(training, seed=652)
    admitted = duplicate.commit_feedback(_analytic_events()[0], visible_at=1)
    duplicate_state = duplicate.state_vector.copy()
    duplicate_result = duplicate.commit_feedback(_analytic_events()[0], visible_at=2)
    revoked_result = duplicate.revoke_feedback("analytic-1")
    revoked_state = duplicate.state_vector.copy()
    revoked_repeat = duplicate.commit_feedback(_analytic_events()[0], visible_at=3)

    controls = {
        "partition_of_unity": {
            "observed_max_abs_gap": max(partition_gaps),
            "expected_max_abs_gap": 1e-12,
            "passed": max(partition_gaps) <= 1e-12,
        },
        "active_support": {
            "observed_max_active": max(active_counts),
            "expected_max_active": DEGREE + 1,
            "passed": max(active_counts) <= DEGREE + 1,
        },
        "duplicate_quantiles": {
            "deterministic": duplicate_knots,
            "passed": duplicate_knots,
        },
        "finite_difference": {
            "observed_max_abs_gap": finite_gap,
            "expected_max_abs_gap": 2e-8,
            "passed": finite_gap < 2e-8,
        },
        "dense_sparse_parity": {
            "observed_max_parameter_linf_gap": max(
                float(row["parameter_linf_gap"]) for row in rows
            ),
            "observed_max_probability_abs_gap": max(
                float(row["probability_abs_gap"]) for row in rows
            ),
            "expected_max_gap": PARITY_TOLERANCE,
            "passed": all(
                float(row["parameter_linf_gap"]) <= PARITY_TOLERANCE
                and float(row["probability_abs_gap"]) <= PARITY_TOLERANCE
                for row in rows
            ),
        },
        "feedback_admission": {
            "first_status": admitted["status"],
            "duplicate_status": duplicate_result["status"],
            "revocation_status": revoked_result["status"],
            "revoked_repeat_status": revoked_repeat["status"],
            "duplicate_kept_state": bool(np.array_equal(duplicate_state, duplicate_state)),
            "revocation_changed_state": bool(not np.array_equal(duplicate_state, revoked_state)),
            "passed": (
                admitted["status"] == "committed"
                and duplicate_result["status"] == "duplicate"
                and revoked_result["status"] == "revoked"
                and revoked_repeat["status"] == "revoked"
            ),
        },
        "checkpoint_restart": {
            "state_hash_before": checkpoint_head.checkpoint_hash,
            "state_hash_after": before_interruption,
            "byte_size": checkpoint_manifest["byte_size"],
            "passed": (
                checkpoint_head.checkpoint_hash == before_interruption
                and checkpoint_manifest["byte_size"] <= MAX_CHECKPOINT_BYTES
            ),
        },
        "interrupted_commit": {
            "interruption_observed": interrupted,
            "committed_state_preserved": (
                committed_after_interruption.checkpoint_hash == before_interruption
            ),
            "passed": interrupted
            and committed_after_interruption.checkpoint_hash == before_interruption,
        },
    }
    footprint = {
        "max_active_coefficient_count": max(int(row["active_coefficient_count"]) for row in rows),
        "max_touched_parameter_count": max(int(row["touched_parameter_count"]) for row in rows),
        "max_coefficient_write_bytes": max(write_bytes),
        "max_arithmetic_operation_count": max(operations),
        "float_bytes": np.dtype(np.float64).itemsize,
        "local_coefficient_storage": True,
        "speedup_claimed": False,
        "route": "six_local_spline_banks_plus_bias_with_bounded_float64_arithmetic",
    }
    return rows, controls, footprint, checkpoint_manifest


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Record an exact input check before dependent numeric work."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate the declared source bytes and requirement identity."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                observed,
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": None,
            }
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-KAN-7425",
            "REQ-KAN-7425" if "REQ-KAN-7425" in spec_text else None,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7425" in exclusion or "exp7425-spline-prototype" in exclusion
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    return checks, hashes


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing affected and terminal receipt by frozen name."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(passed)


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Build one plain-valued gate row with its separate evidence principle."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(
    gates: Sequence[Mapping[str, Any]], preconditions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Name the first exact failed upstream field without hiding missing values."""

    failed_precondition = next(
        (dict(row) for row in preconditions if row.get("passed") is not True), None
    )
    failed_gate = next((dict(row) for row in gates if row.get("passed") is not True), None)
    source = failed_precondition or failed_gate
    return {
        "all_required_gates_passed": source is None,
        "blocked_upstream": None if source is None else source.get("upstream", "current_run"),
        "blocked_path": None if source is None else source.get("path", RESULT_PATH.as_posix()),
        "blocked_check": None if source is None else source.get("check"),
        "blocked_field": None if source is None else source.get("field", "passed"),
        "blocked_expected": None if source is None else source.get("expected"),
        "blocked_observed": None if source is None else source.get("observed"),
    }


def _field_principles() -> dict[str, str]:
    """Explain field intent without wrapping the field's actual value."""

    specific = {
        "schema": "A versioned plain top-level schema identifies the terminal record.",
        "run_date": "The scheduled date is separate from actual UTC run boundaries.",
        "preconditions_checked": "Exact path and identity checks precede dependent work.",
        "MODEL_SPECS": "No current LLM use requires an empty model list.",
        "model_invoked": "Only an actual current model attempt can set this true.",
        "invocation_counts": "Owned current load and generation events reduce to counters.",
        "inference_substrate": "The truthful substrate is a string; details stay separate.",
        "inference_substrate_class": "Small numeric fitting is not a model load.",
        "execution_venue": "The closed execution venue is host.",
        "duration_s": "Monotonic duration separates current measured phases.",
        "phase_spans": "Real phase times retain completed units and checkpoints.",
        "random_seed": "Frozen initialization seeds make analytic replay deterministic.",
        "reproducibility_checksum": "The checksum binds code, inputs, rows, and validation scope.",
        "source_artifact_hashes": "Exact byte hashes bind every source input identity.",
        "rows": "Every seed and analytic fixture step remains explicit.",
        "sample_size_budget": "Planned and terminal fixture counts use a fixed stop rule.",
        "acceptance_gate_results": "Validity is separate from scientific benefit.",
        "gate_check_summary": "A failed gate names its exact upstream field and observation.",
        "verifier_is_oracle": "Analytic fixtures share correctness authority and are circular.",
        "honest_verdict": "Completed findings use the complete_ terminal prefix.",
        "verdict_class": "The verdict uses the closed project enum.",
        "flagged_adversarial": "Critical findings cannot supply readiness.",
        "validation_receipts": "Exact scoped commands keep exits, durations, and hashed logs.",
        "field_principles": "Principles explain ordinary fields separately from values.",
        "promotion_score": "No rollout, publication, or generator update occurs.",
        "spline_prototype_ready_score": "All numeric, safety, parity, and validation checks gate readiness.",
        "energy_definition": "The exact two-state energy exposes fixed-basis logistic equivalence.",
        "parity_rows": "Each fixture step records sparse and dense disagreements.",
        "update_footprint": "Measured local writes and operations bound one online update.",
        "small_ebm_training": "Only tiny spline-head updates occur; current LLM calls stay zero.",
    }
    return {
        field: specific.get(field, "This field preserves measured prototype evidence.")
        for field in REQUIRED_FIELDS
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind protocol, exact sources, raw rows, and validation scope."""

    keys = (
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "analytic_controls",
        "energy_definition",
        "update_footprint",
        "checkpoint_manifest",
        "validation_receipts",
    )
    return canonical_hash({key: value.get(key) for key in keys})


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute prototype readiness only from stored raw checks and receipts."""

    if value.get("verdict_class") == "blocked":
        return {"spline_prototype_ready_score": 0, "promotion_score": 0}
    rows = value.get("parity_rows") or []
    controls = value.get("analytic_controls") or {}
    preconditions = value.get("preconditions_checked") or []
    checkpoint = value.get("checkpoint_manifest") or {}
    footprint = value.get("update_footprint") or {}
    parity = bool(rows) and all(
        row.get("status") == "completed"
        and row.get("data_role") == "analytic_fixture"
        and isinstance(row.get("probability_abs_gap"), (int, float))
        and float(row["probability_abs_gap"]) <= PARITY_TOLERANCE
        and isinstance(row.get("parameter_linf_gap"), (int, float))
        and float(row["parameter_linf_gap"]) <= PARITY_TOLERANCE
        and int(row.get("active_coefficient_count", 99)) <= INPUT_COUNT * (DEGREE + 1)
        and int(row.get("touched_parameter_count", 99)) <= INPUT_COUNT * (DEGREE + 1) + 1
        and int(row.get("coefficient_write_bytes", -1))
        == int(row.get("touched_parameter_count", -2)) * np.dtype(np.float64).itemsize
        for row in rows
    )
    checks = (
        bool(preconditions) and all(row.get("passed") is True for row in preconditions),
        bool(controls) and all(row.get("passed") is True for row in controls.values()),
        parity,
        int(checkpoint.get("byte_size", MAX_CHECKPOINT_BYTES + 1)) <= MAX_CHECKPOINT_BYTES,
        footprint.get("speedup_claimed") is False,
        int(footprint.get("max_active_coefficient_count", 99)) <= INPUT_COUNT * (DEGREE + 1),
        _validation_passed(value.get("validation_receipts") or []),
        value.get("MODEL_SPECS") == [],
        value.get("model_invoked") is False,
        value.get("invocation_counts") == ZERO_INVOCATION_COUNTS,
    )
    return {
        "spline_prototype_ready_score": int(all(checks)),
        "promotion_score": 0,
    }


def _build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    footprint: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one internally reducible terminal or pre-terminal artifact."""

    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    controls_passed = bool(controls) and all(row.get("passed") is True for row in controls.values())
    parity_passed = bool(rows) and all(
        float(row.get("probability_abs_gap", math.inf)) <= PARITY_TOLERANCE
        and float(row.get("parameter_linf_gap", math.inf)) <= PARITY_TOLERANCE
        for row in rows
    )
    validation_passed = _validation_passed(validation_receipts)
    checkpoint_passed = int(checkpoint_manifest.get("byte_size", MAX_CHECKPOINT_BYTES + 1)) <= (
        MAX_CHECKPOINT_BYTES
    )
    gates = [
        _gate(
            "preconditions",
            "validity",
            "all",
            True,
            preconditions_passed,
            preconditions_passed,
            "Every exact source must authenticate before fixture work.",
        ),
        _gate(
            "analytic_numeric_controls",
            "validity",
            "all",
            True,
            controls_passed,
            controls_passed,
            "Basis, gradient, feedback, restart, and interruption checks must pass.",
        ),
        _gate(
            "sparse_dense_parity",
            "validity",
            "<=",
            PARITY_TOLERANCE,
            max(
                (
                    max(
                        float(row.get("probability_abs_gap", math.inf)),
                        float(row.get("parameter_linf_gap", math.inf)),
                    )
                    for row in rows
                ),
                default=None,
            ),
            parity_passed,
            "Sparse and dense disagreement is a defect, never a benefit.",
        ),
        _gate(
            "numeric_checkpoint_bound",
            "safety",
            "<=",
            MAX_CHECKPOINT_BYTES,
            checkpoint_manifest.get("byte_size"),
            checkpoint_passed,
            "The numeric checkpoint must fit the fixed prototype memory bound.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_passed,
            validation_passed,
            "All frozen scoped checks and terminal readers must pass.",
        ),
    ]
    valid = (
        preconditions_passed
        and controls_passed
        and parity_passed
        and checkpoint_passed
        and validation_passed
        and not flagged_adversarial
    )
    if valid:
        status = "complete"
        verdict_class = "circular_positive"
        honest_verdict = "complete_circular_spline_numeric_api_ready_analytic_fixtures_only"
    else:
        status = "disqualified"
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_spline_prototype_required_evidence"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **deepcopy(dict(current_receipt)),
        "random_seed": {"head_initialization": list(TRAINING_SEEDS)},
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in rows],
        "sample_size_budget": {
            "planned": len(TRAINING_SEEDS) * len(_analytic_events()),
            "attempted": len(rows),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "censored": sum(row.get("status") == "censored" for row in rows),
            "unstarted": len(TRAINING_SEEDS) * len(_analytic_events()) - len(rows),
            "independent_groups": 0,
            "fixture_groups": len(_analytic_events()),
            "stop_rule": "run every sealed analytic fixture for every registered seed",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates, preconditions),
        "verifier_is_oracle": True,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "spline_prototype_ready_score": 0,
        "energy_definition": {
            "inputs": INPUT_COUNT,
            "basis": "additive_open_clamped_cubic_b_spline_fixed_after_training_quantiles",
            "degree": DEGREE,
            "coefficients_per_input": COEFFICIENTS_PER_INPUT,
            "bias_parameters": 1,
            "trainable_parameter_count": PARAMETER_COUNT,
            "energy_y0": "0",
            "energy_y1": "-f(x)",
            "class_one_probability": "exp(f)/(1+exp(f)) = sigmoid(f)",
            "equivalent_model": "logistic_regression_on_fixed_spline_basis",
            "normalizer_state_count": 2,
        },
        "parity_rows": [deepcopy(dict(row)) for row in rows],
        "update_footprint": deepcopy(dict(footprint)),
        "analytic_controls": deepcopy(dict(controls)),
        "checkpoint_manifest": deepcopy(dict(checkpoint_manifest)),
        "methodology": (
            "Analytic fixtures check a fixed spline design, sparse updates, and dense logistic "
            "parity. They are not corpus rows and provide no real-data efficacy evidence."
        ),
        "methodology_note": (
            "Exact sparse-dense equality is a deterministic implementation invariant. It is "
            "not a statistical performance estimate or learning advantage."
        ),
        "hardware_route": {
            "storage": "local_eight_coefficient_bank_per_input_plus_one_bias",
            "bounded_arithmetic": True,
            "speedup_claimed": False,
        },
    }
    artifact.update(independent_reduce(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_fixture_artifact(
    *, validation_receipts: Sequence[Mapping[str, Any]] | None = None
) -> JsonDict:
    """Build one compact complete artifact for cold-reader mutation tests."""

    with tempfile.TemporaryDirectory(prefix="exp7425-fixture-") as temporary:
        rows, controls, footprint, checkpoint = run_analytic_fixtures(
            checkpoint_dir=Path(temporary)
        )
    receipt = build_current_work_receipt(
        run_id="exp7425-fixture",
        owner_pid=0,
        events=[],
        inference_substrate="numpy_float64_additive_spline_head",
        inference_substrate_details={"device": "cpu", "model_loaded": False},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000,
        phase_spans=[
            {
                "phase": "analytic_fixture",
                "start_s": 0.0,
                "end_s": 0.001,
                "duration_s": 0.001,
                "completed_units": len(rows),
            }
        ],
        small_ebm_training={
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "fixed_basis_additive_spline_logistic",
            "updates_completed": len(rows),
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
    )
    return _build_artifact(
        preconditions=[
            _precondition("fixture_protocol", "analytic-fixture-v1", "fixture", "ready", True, True)
        ],
        source_hashes={},
        rows=rows,
        controls=controls,
        footprint=footprint,
        checkpoint_manifest=checkpoint,
        validation_receipts=list(validation_receipts or []),
        current_receipt=receipt,
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00.001000+00:00",
    )


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish an unavailable source as blocked before dependent work starts."""

    receipt = build_current_work_receipt(
        run_id="exp7425-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="numpy_float64_additive_spline_head",
        inference_substrate_details={"device": "cpu", "dependent_work_started": False},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        small_ebm_training={"performed": False, "current_llm_calls": 0},
    )
    controls: JsonDict = {}
    footprint = {
        "max_active_coefficient_count": 0,
        "max_touched_parameter_count": 0,
        "max_coefficient_write_bytes": 0,
        "max_arithmetic_operation_count": 0,
        "float_bytes": 8,
        "local_coefficient_storage": True,
        "speedup_claimed": False,
        "route": "not_started",
    }
    artifact = _build_artifact(
        preconditions=[deepcopy(dict(failed))],
        source_hashes={},
        rows=[],
        controls=controls,
        footprint=footprint,
        checkpoint_manifest={"byte_size": 0},
        validation_receipts=[],
        current_receipt=receipt,
        started_at_utc=utc_now(),
        completed_at_utc=utc_now(),
    )
    artifact.update(
        {
            "status": "blocked",
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failed.get('check', 'external_prerequisite')}",
            "spline_prototype_ready_score": 0,
            "promotion_score": 0,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, raw reduction, provenance, parity, and checksum."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if value.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("model_declaration_mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_mismatch")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    errors.extend(validate_current_work_receipt(value))
    for index, row in enumerate(value.get("parity_rows") or []):
        probability_gap = row.get("probability_abs_gap")
        parameter_gap = row.get("parameter_linf_gap")
        if (
            not isinstance(probability_gap, (int, float))
            or isinstance(probability_gap, bool)
            or not math.isfinite(float(probability_gap))
            or float(probability_gap) > PARITY_TOLERANCE
            or not isinstance(parameter_gap, (int, float))
            or isinstance(parameter_gap, bool)
            or not math.isfinite(float(parameter_gap))
            or float(parameter_gap) > PARITY_TOLERANCE
        ):
            errors.append(f"parity_tolerance_failed:{index}")
    reduced = independent_reduce(value)
    if value.get("spline_prototype_ready_score") != reduced["spline_prototype_ready_score"]:
        errors.append("spline_prototype_ready_score_mismatch")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_mismatch")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    verdict = value.get("verdict_class")
    if verdict not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    honest = str(value.get("honest_verdict") or "")
    if verdict == "blocked" and not honest.startswith("blocked_"):
        errors.append("blocked_verdict_prefix_invalid")
    if verdict != "blocked" and not honest.startswith("complete_"):
        errors.append("complete_verdict_prefix_invalid")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_oracle_declaration_mismatch")
    return list(dict.fromkeys(errors))


def _load_object(path: Path) -> JsonDict:
    """Read a JSON object and treat missing or malformed bytes as absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def cold_replay(path: Path) -> list[str]:
    """Read and validate one candidate in a fresh process."""

    value = _load_object(path)
    return validate_artifact(value) if value else ["artifact_unreadable_or_not_object"]


def _span(
    phase: str, phase_started: float, run_started: float, completed: int
) -> JsonDict:  # pragma: no cover - real clock boundary.
    """Close one real phase with completed-unit telemetry."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, reduction, adversarial, and strict row readers."""

    python = ".venv/bin/python"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--cold-replay",
                    str(candidate),
                ),
                "capability_end_to_end",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_row_reduction",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--independent-reduce",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
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
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the required public entrypoint.
    """Authenticate, measure, validate, and atomically publish the prototype."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    monotonic_started_ns = time.monotonic_ns()
    started_at_utc = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "start")
    preconditions, source_hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    phase_started = time.monotonic()
    progress(run_started, "analytic_fixtures", "before_benchmark")
    checkpoint_dir = root / RAW_DIR / "numeric_state"
    rows, controls, footprint, checkpoint = run_analytic_fixtures(checkpoint_dir=checkpoint_dir)
    spans.append(_span("analytic_fixtures", phase_started, run_started, len(rows)))
    progress(run_started, "analytic_fixtures", "after_benchmark", completed=len(rows))
    checkpoint_path = Path(str(checkpoint["path"]))
    source_hashes[checkpoint_path.relative_to(root).as_posix()] = {
        "path": checkpoint_path.relative_to(root).as_posix(),
        "sha256": checkpoint["sha256"],
        "original_flagged_adversarial": None,
    }

    raw_dir = root / RAW_DIR
    private_root = Path(tempfile.mkdtemp(prefix="exp7425-validation-", dir="/tmp"))
    commands = build_command_plan(root, V651_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V651_MANIFEST, commands)
    phase_started = time.monotonic()
    progress(
        run_started,
        "affected_validation",
        "before_subprocesses",
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, V651_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "original_flagged_adversarial": None,
        }
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{monotonic_started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="numpy_float64_additive_spline_head",
        inference_substrate_details={
            "device": "cpu",
            "software": "numpy",
            "model_loaded": False,
            "work": "analytic fixed-basis spline updates",
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - monotonic_started_ns,
        phase_spans=spans,
        small_ebm_training={
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "fixed_basis_additive_spline_logistic",
            "updates_completed": len(rows),
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
    )
    candidate = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=rows,
        controls=controls,
        footprint=footprint,
        checkpoint_manifest=checkpoint,
        validation_receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at_utc,
        completed_at_utc=utc_now(),
        flagged_adversarial=not affected_reduction["passed"],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(run_started, "candidate", "before_serialization")
    atomic_json(candidate_path, candidate)
    progress(run_started, "candidate", "after_serialization")

    terminal_commands = _terminal_commands(candidate_path)
    phase_started = time.monotonic()
    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        planned=len(terminal_commands),
    )
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )

    final_receipt = build_current_work_receipt(
        run_id=receipt["current_run_id"],
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_details=receipt["inference_substrate_details"],
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - monotonic_started_ns,
        phase_spans=spans,
        small_ebm_training=receipt["small_ebm_training"],
    )
    final = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=rows,
        controls=controls,
        footprint=footprint,
        checkpoint_manifest=checkpoint,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at_utc,
        completed_at_utc=utc_now(),
        flagged_adversarial=(not affected_reduction["passed"] or not terminal_passed or critical),
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = validate_artifact(value) if value else ["artifact_unreadable_or_not_object"]
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
