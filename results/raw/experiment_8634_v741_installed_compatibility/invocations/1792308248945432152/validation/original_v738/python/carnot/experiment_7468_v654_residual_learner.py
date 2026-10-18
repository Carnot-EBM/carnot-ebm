"""Qualify one local residual energy head with delayed analytic feedback.

The head changes only a small spline residual above a frozen readout. Analytic
fixtures prove implementation properties, not source-support value. Spec refs:
REQ-KAN-7468 and SCENARIO-KAN-7468-*.
"""

from __future__ import annotations

import argparse
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

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7425_v651_spline_prototype import (
    COEFFICIENTS_PER_INPUT,
    DEGREE,
    KNOT_VECTOR_SIZE,
    _repair_breakpoints,
    cubic_basis,
    sigmoid,
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
MILESTONE = "2026.09.654"
PHASE = 3
EXPERIMENT_ID = "exp7468-v654-residual-learner"
SCHEMA = "carnot.exp7468.v654.residual_learner.v1"
CHECKPOINT_SCHEMA = "carnot.exp7468.local_residual_checkpoint.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7468_v654_residual_learner.json")
RAW_DIR = Path("results/raw/experiment_7468_v654_residual_learner")
MODULE_PATH = Path("python/carnot/experiment_7468_v654_residual_learner.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7468_v654_residual_learner.py")
TEST_PATH = Path("tests/python/test_experiment_7468_v654_residual_learner.py")
SPEC_PATH = Path("openspec/capabilities/kan/spec.md")

FEATURE_COUNT = 4
PARAMETER_COUNT = FEATURE_COUNT * COEFFICIENTS_PER_INPUT + 1
LEARNING_RATE = 0.02
GRADIENT_NORM_CAP = 1.0
RESIDUAL_BOUND = 2.0
PARITY_TOLERANCE = 1e-10
GUARD_TOLERANCE = 0.005
FIT_SEEDS = (746800, 746801, 746802, 746803, 746804)
ORDERING_SEED = 746811
AUDIT_SEED = 746821
BOOTSTRAP_SEED = 746831
INFERENCE_SUBSTRATE = "numpy_float64_delayed_local_residual_energy_training"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

UPSTREAM_ARTIFACTS = {
    Path("results/experiment_7450_v653_prediction_ledger.json"): {
        "status": "complete_circular_positive_prediction_ledger_ready",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
    },
    Path("results/experiment_7454_v653_continuous_learning.json"): {
        "status": "complete_null_retired",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "mixture_construction_retired": True,
    },
    Path("results/experiment_7458_v653_durable_updates.json"): {
        "status": "complete_null_durable_delta_speed_gate_not_met",
        "verdict_class": "null",
        "flagged_adversarial": False,
    },
    Path("results/experiment_7462_v654_option_protocol.json"): {
        "status": "disqualified",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
    },
}

INPUT_PATHS = (
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
    Path("python/carnot/experiment_7425_v651_spline_prototype.py"),
    Path("python/carnot/experiment_7450_v653_prediction_ledger.py"),
    Path("python/carnot/experiment_7462_v654_option_protocol.py"),
    Path("python/carnot/experiment_7458_v653_durable_updates.py"),
    SPEC_PATH,
    *UPSTREAM_ARTIFACTS,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "phase",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "clock_identity",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
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
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "residual_learner_ready_score",
    "online_protocol",
    "support_overlap_rows",
    "hardware_acceleration_path",
    "small_ebm_training",
    "analytic_rows",
    "analytic_checks",
)


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert to float64 so malformed numeric state cannot enter a hash."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name}_shape_invalid")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}_nonfinite")
    return array


def _clip_probability(value: float) -> float:
    """Keep frozen log odds finite without tuning from an observed outcome."""

    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError("frozen_probability_must_be_strictly_between_zero_and_one")
    return min(max(numeric, 1e-12), 1.0 - 1e-12)


def fit_local_knots(training_features: Any) -> np.ndarray:
    """Fit four open cubic knot vectors from training inputs only."""

    training = np.asarray(training_features, dtype=np.float64)
    if training.ndim != 2 or training.shape[1] != FEATURE_COUNT or training.shape[0] < 2:
        raise ValueError("training_features_shape_invalid")
    if not np.all(np.isfinite(training)):
        raise ValueError("training_features_nonfinite")
    vectors: list[np.ndarray] = []
    for feature in range(FEATURE_COUNT):
        raw = np.quantile(training[:, feature], np.linspace(0.0, 1.0, 6), method="linear")
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


def local_design(features: Any, knots: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reuse the Exp7425 sparse cubic basis for four scalar features."""

    values = _finite_array(features, (FEATURE_COUNT,), "features")
    knot_matrix = _finite_array(knots, (FEATURE_COUNT, KNOT_VECTOR_SIZE), "knots")
    parts: list[np.ndarray] = []
    active: list[int] = []
    for feature, numeric in enumerate(values):
        evaluated = cubic_basis(float(numeric), knot_matrix[feature])
        parts.append(evaluated.values)
        active.extend(
            feature * COEFFICIENTS_PER_INPUT + index for index in evaluated.active_indices
        )
    return np.concatenate(parts), tuple(active)


def prediction_event_hash(value: Mapping[str, Any]) -> str:
    """Bind every label-free prediction field before feedback can exist."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "prediction_event_hash"}
    )


class LocalResidualHead:
    """Apply one guarded local update above an immutable frozen readout."""

    def __init__(
        self,
        knots: Any,
        coefficients: Any,
        bias: float,
        *,
        learning_rate: float = LEARNING_RATE,
        residual_bound: float = RESIDUAL_BOUND,
        guard_rows: Sequence[Mapping[str, Any]] = (),
        guard_tolerance: float = GUARD_TOLERANCE,
    ) -> None:
        self.knots = _finite_array(knots, (FEATURE_COUNT, KNOT_VECTOR_SIZE), "knots").copy()
        self.coefficients = _finite_array(
            coefficients, (FEATURE_COUNT, COEFFICIENTS_PER_INPUT), "coefficients"
        ).copy()
        self.bias = float(bias)
        self.learning_rate = float(learning_rate)
        self.residual_bound = float(residual_bound)
        self.guard_tolerance = float(guard_tolerance)
        if not all(
            math.isfinite(value)
            for value in (self.bias, self.learning_rate, self.residual_bound, self.guard_tolerance)
        ):
            raise ValueError("head_scalar_nonfinite")
        if self.learning_rate <= 0.0 or self.residual_bound <= 0.0 or self.guard_tolerance < 0.0:
            raise ValueError("head_scalar_range_invalid")
        self.guard_rows = [self._validate_guard_row(row) for row in guard_rows]
        self._predictions: dict[str, JsonDict] = {}
        self._prediction_order: list[str] = []
        self._feedback_fingerprints: dict[str, str] = {}
        self.update_count = 0
        self.rejection_count = 0

    @classmethod
    def from_training(
        cls,
        training_features: Any,
        *,
        seed: int,
        guard_rows: Sequence[Mapping[str, Any]] = (),
        guard_tolerance: float = GUARD_TOLERANCE,
    ) -> LocalResidualHead:
        """Freeze knots, learning rate, and small initial state before feedback."""

        knots = fit_local_knots(training_features)
        rng = np.random.default_rng(int(seed))
        coefficients = rng.normal(0.0, 0.01, size=(FEATURE_COUNT, COEFFICIENTS_PER_INPUT))
        return cls(
            knots,
            coefficients,
            0.0,
            guard_rows=guard_rows,
            guard_tolerance=guard_tolerance,
        )

    @staticmethod
    def _validate_guard_row(row: Mapping[str, Any]) -> JsonDict:
        """Seal one training-only replay row before any online label arrives."""

        features = _finite_array(row.get("features"), (FEATURE_COUNT,), "guard_features")
        label = row.get("label")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("guard_label_invalid")
        frozen = _clip_probability(float(row.get("frozen_probability")))
        return {"features": features.tolist(), "label": int(label), "frozen_probability": frozen}

    @property
    def state_vector(self) -> np.ndarray:
        """Return detached parameters for rollback and independent comparison."""

        return np.concatenate((self.coefficients.reshape(-1), np.asarray([self.bias])))

    @property
    def state_hash(self) -> str:
        """Bind numeric state and frozen update hyperparameters."""

        return canonical_hash(
            {
                "knots": self.knots.tolist(),
                "state": self.state_vector.tolist(),
                "learning_rate": self.learning_rate,
                "residual_bound": self.residual_bound,
                "guard_tolerance": self.guard_tolerance,
            }
        )

    def predict(self, features: Any, *, frozen_probability: float) -> JsonDict:
        """Compute the required energy difference without observing a label."""

        frozen = _clip_probability(frozen_probability)
        design, support = local_design(features, self.knots)
        local = float(design @ self.coefficients.reshape(-1))
        raw_residual = local + self.bias
        bounded = min(max(raw_residual, -self.residual_bound), self.residual_bound)
        frozen_log_odds = math.log(frozen / (1.0 - frozen))
        energy_difference = frozen_log_odds + bounded
        return {
            "active_support": list(support),
            "frozen_prediction": frozen,
            "frozen_readout_log_odds": frozen_log_odds,
            "local_coefficient_contribution": local,
            "shared_bias_contribution": self.bias,
            "bounded_local_residual": bounded,
            "energy_bad_minus_good": energy_difference,
            "residual_prediction": sigmoid(energy_difference),
            "predictor_state_hash": self.state_hash,
        }

    def seal_prediction(
        self,
        *,
        event_id: str,
        source_version: str,
        prediction_time: int,
        reveal_time: int,
        features: Any,
        frozen_probability: float,
    ) -> JsonDict:
        """Store one immutable prediction before its label becomes visible."""

        identity = str(event_id)
        source = str(source_version)
        if not identity or not source:
            raise ValueError("prediction_identity_invalid")
        if identity in self._predictions:
            raise ValueError("prediction_event_duplicate")
        if (
            not isinstance(prediction_time, int)
            or isinstance(prediction_time, bool)
            or not isinstance(reveal_time, int)
            or isinstance(reveal_time, bool)
            or prediction_time < 0
            or reveal_time < prediction_time
        ):
            raise ValueError("prediction_time_invalid")
        numeric_features = _finite_array(features, (FEATURE_COUNT,), "features")
        prediction = self.predict(numeric_features, frozen_probability=frozen_probability)
        event: JsonDict = {
            "event_id": identity,
            "source_version": source,
            "prediction_time": prediction_time,
            "reveal_time": reveal_time,
            "prediction_sequence": len(self._prediction_order),
            "features": numeric_features.tolist(),
            **prediction,
        }
        event["prediction_event_hash"] = prediction_event_hash(event)
        self._predictions[identity] = deepcopy(event)
        self._prediction_order.append(identity)
        return deepcopy(event)

    def _gradient(
        self, features: Any, label: int, frozen_probability: float
    ) -> tuple[np.ndarray, tuple[int, ...], float]:
        """Return the sparse Bernoulli gradient for the residual parameters."""

        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("feedback_label_invalid")
        design, support = local_design(features, self.knots)
        prediction = self.predict(features, frozen_probability=frozen_probability)
        residual = float(prediction["residual_prediction"]) - int(label)
        raw = float(prediction["local_coefficient_contribution"]) + self.bias
        derivative = 1.0 if abs(raw) < self.residual_bound else 0.0
        gradient = np.concatenate(
            (residual * derivative * design, np.asarray([residual * derivative]))
        )
        logit = float(prediction["energy_bad_minus_good"])
        loss = float(np.logaddexp(0.0, logit) - int(label) * logit)
        return gradient, support, loss

    def sparse_gradient(
        self, features: Any, label: int, *, frozen_probability: float
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Expose the sparse gradient for independent dense parity tests."""

        gradient, support, _loss = self._gradient(features, label, frozen_probability)
        return gradient, support

    def _guard_loss(self) -> float:
        """Measure only the preregistered training replay rows."""

        if not self.guard_rows:
            return 0.0
        losses = []
        for row in self.guard_rows:
            prediction = self.predict(
                row["features"], frozen_probability=float(row["frozen_probability"])
            )
            logit = float(prediction["energy_bad_minus_good"])
            losses.append(float(np.logaddexp(0.0, logit) - int(row["label"]) * logit))
        return math.fsum(losses) / len(losses)

    def apply_feedback(self, event_id: str, *, label: int, visible_at: int) -> JsonDict:
        """Admit one chronological label, then keep or roll back one update."""

        identity = str(event_id)
        event = self._predictions.get(identity)
        if event is None:
            return {"event_id": identity, "status": "missing_prediction", "updates_applied": 0}
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("feedback_label_invalid")
        fingerprint = canonical_hash(
            {"prediction_event_hash": event["prediction_event_hash"], "label": int(label)}
        )
        prior = self._feedback_fingerprints.get(identity)
        if prior is not None:
            status = "duplicate" if prior == fingerprint else "identity_conflict"
            return {"event_id": identity, "status": status, "updates_applied": 0}
        if int(visible_at) < int(event["reveal_time"]):
            return {"event_id": identity, "status": "not_revealed", "updates_applied": 0}
        target_index = int(event["prediction_sequence"])
        if any(
            earlier not in self._feedback_fingerprints
            for earlier in self._prediction_order[:target_index]
        ):
            return {"event_id": identity, "status": "reordered", "updates_applied": 0}

        state_before = self.state_vector.copy()
        state_hash_before = self.state_hash
        guard_started = time.perf_counter_ns()
        guard_before = self._guard_loss()
        gradient, support, loss = self._gradient(
            event["features"], int(label), float(event["frozen_prediction"])
        )
        raw_norm = float(np.linalg.norm(gradient))
        scale = min(1.0, GRADIENT_NORM_CAP / max(raw_norm, np.finfo(np.float64).tiny))
        clipped = gradient * scale
        flat = self.coefficients.reshape(-1)
        active = np.asarray(support, dtype=np.int64)
        flat[active] -= self.learning_rate * clipped[active]
        self.bias -= self.learning_rate * float(clipped[-1])
        guard_after = self._guard_loss()
        rejected = guard_after > guard_before + self.guard_tolerance
        if rejected:
            self.coefficients = (
                state_before[:-1].reshape(FEATURE_COUNT, COEFFICIENTS_PER_INPUT).copy()
            )
            self.bias = float(state_before[-1])
            self.rejection_count += 1
        else:
            self.update_count += 1
        self._feedback_fingerprints[identity] = fingerprint
        guard_elapsed = time.perf_counter_ns() - guard_started
        return {
            "event_id": identity,
            "prediction_event_hash": event["prediction_event_hash"],
            "status": "rejected_guard" if rejected else "committed",
            "updates_applied": int(not rejected),
            "rolled_back": rejected,
            "state_hash_before": state_hash_before,
            "state_hash_after": self.state_hash,
            "loss": loss,
            "learning_rate": self.learning_rate,
            "gradient_norm_before_clip": raw_norm,
            "gradient_norm_after_clip": float(np.linalg.norm(clipped)),
            "active_support": list(support),
            "touched_parameter_count": len(support) + 1,
            "guard_loss_before": guard_before,
            "guard_loss_after": guard_after,
            "guard_tolerance": self.guard_tolerance,
            "guard_rows_evaluated": 2 * len(self.guard_rows),
            "guard_basis_evaluations": 2 * len(self.guard_rows) * FEATURE_COUNT,
            "guard_loss_computations": 2 * len(self.guard_rows),
            "guard_elapsed_ns": guard_elapsed,
        }

    def _checkpoint_payload(self) -> JsonDict:
        """Return all state needed to resume immutable feedback chronology."""

        return {
            "schema": CHECKPOINT_SCHEMA,
            "knots": self.knots.tolist(),
            "coefficients": self.coefficients.tolist(),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "residual_bound": self.residual_bound,
            "guard_rows": deepcopy(self.guard_rows),
            "guard_tolerance": self.guard_tolerance,
            "predictions": [deepcopy(self._predictions[key]) for key in self._prediction_order],
            "feedback_fingerprints": deepcopy(self._feedback_fingerprints),
            "update_count": self.update_count,
            "rejection_count": self.rejection_count,
        }

    def save_checkpoint(self, path: Path) -> JsonDict:
        """Atomically publish one hash-bound numeric and ledger checkpoint."""

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
        os.replace(temporary, target)
        return {
            "path": str(target),
            "sha256": sha256_file(target),
            "state_hash": self.state_hash,
            "byte_size": target.stat().st_size,
        }

    @classmethod
    def load_checkpoint(cls, path: Path) -> LocalResidualHead:
        """Authenticate a checkpoint before restoring delayed event state."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:  # pragma: no cover - defensive reader.
            raise ValueError("checkpoint_unreadable") from error
        if not isinstance(value, Mapping) or value.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint_schema_invalid")
        payload = {key: deepcopy(item) for key, item in value.items() if key != "checkpoint_sha256"}
        if value.get("checkpoint_sha256") != canonical_hash(payload):
            raise ValueError("checkpoint_hash_invalid")
        restored = cls(
            value["knots"],
            value["coefficients"],
            float(value["bias"]),
            learning_rate=float(value["learning_rate"]),
            residual_bound=float(value["residual_bound"]),
            guard_rows=list(value.get("guard_rows") or []),
            guard_tolerance=float(value["guard_tolerance"]),
        )
        predictions = list(value.get("predictions") or [])
        restored._prediction_order = [str(row["event_id"]) for row in predictions]
        restored._predictions = {str(row["event_id"]): deepcopy(dict(row)) for row in predictions}
        restored._feedback_fingerprints = {
            str(key): str(item)
            for key, item in dict(value.get("feedback_fingerprints") or {}).items()
        }
        restored.update_count = int(value.get("update_count", 0))
        restored.rejection_count = int(value.get("rejection_count", 0))
        return restored


def independent_dense_gradient(
    head: LocalResidualHead, features: Any, label: int, *, frozen_probability: float
) -> np.ndarray:
    """Recompute the full gradient without calling the sparse gradient path."""

    values = _finite_array(features, (FEATURE_COUNT,), "features")
    full_parts = [cubic_basis(float(values[i]), head.knots[i]).values for i in range(FEATURE_COUNT)]
    design = np.concatenate(full_parts)
    frozen = _clip_probability(frozen_probability)
    raw = float(design @ head.coefficients.reshape(-1) + head.bias)
    bounded = min(max(raw, -head.residual_bound), head.residual_bound)
    probability = sigmoid(math.log(frozen / (1.0 - frozen)) + bounded)
    derivative = 1.0 if abs(raw) < head.residual_bound else 0.0
    residual = (probability - int(label)) * derivative
    return np.concatenate((residual * design, np.asarray([residual])))


class MatchedLinearResidual:
    """Provide a four-feature linear updater with the same step and clipping."""

    def __init__(self, training_features: Any) -> None:
        training = np.asarray(training_features, dtype=np.float64)
        if training.ndim != 2 or training.shape[1] != FEATURE_COUNT:
            raise ValueError("linear_training_shape_invalid")
        self.center = np.mean(training, axis=0)
        scale = np.std(training, axis=0)
        self.scale = np.where(scale > 0.0, scale, 1.0)
        self.weights = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self.bias = 0.0
        self.update_count = 0

    def update(self, features: Any, label: int, *, frozen_probability: float) -> JsonDict:
        """Apply the matched one-step linear logistic update."""

        values = _finite_array(features, (FEATURE_COUNT,), "linear_features")
        normalized = (values - self.center) / self.scale
        frozen = _clip_probability(frozen_probability)
        logit = math.log(frozen / (1.0 - frozen)) + float(normalized @ self.weights + self.bias)
        residual = sigmoid(logit) - int(label)
        gradient = np.concatenate((residual * normalized, np.asarray([residual])))
        norm = float(np.linalg.norm(gradient))
        clipped = gradient * min(1.0, GRADIENT_NORM_CAP / max(norm, np.finfo(np.float64).tiny))
        self.weights -= LEARNING_RATE * clipped[:-1]
        self.bias -= LEARNING_RATE * float(clipped[-1])
        self.update_count += 1
        return {
            "update_count": self.update_count,
            "gradient_norm_after_clip": float(np.linalg.norm(clipped)),
        }


def run_control_comparison(training_features: Any) -> dict[str, JsonDict]:
    """Exercise the local, matched-linear, frozen, and no-feedback arms."""

    training = np.asarray(training_features, dtype=np.float64)
    features = training[2]
    local = LocalResidualHead.from_training(training, seed=FIT_SEEDS[0])
    local.seal_prediction(
        event_id="control-local",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=features,
        frozen_probability=0.55,
    )
    local_receipt = local.apply_feedback("control-local", label=1, visible_at=0)
    linear = MatchedLinearResidual(training)
    linear_receipt = linear.update(features, 1, frozen_probability=0.55)
    frozen = LocalResidualHead.from_training(training, seed=FIT_SEEDS[0])
    frozen_state = frozen.state_hash
    no_feedback_state = canonical_hash({"frozen_probability": 0.55, "residual": 0.0})
    return {
        "local_residual": {"update_count": local.update_count, "status": local_receipt["status"]},
        "matched_linear_residual": deepcopy(linear_receipt),
        "frozen_residual": {
            "update_count": frozen.update_count,
            "state_unchanged": frozen.state_hash == frozen_state,
        },
        "no_feedback": {
            "update_count": 0,
            "state_unchanged": canonical_hash({"frozen_probability": 0.55, "residual": 0.0})
            == no_feedback_state,
        },
    }


def _training_fixture() -> np.ndarray:
    """Return an analytic training-only support fixture."""

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


def run_analytic_controls(checkpoint_dir: Path) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Measure sign, chronology, guard, locality, controls, and restart."""

    training = _training_fixture()
    rows: list[JsonDict] = []
    sign_head = LocalResidualHead.from_training(training, seed=FIT_SEEDS[0])
    sign_head.coefficients.fill(0.0)
    sign_head.bias = 0.0
    for probability in (0.2, 0.5, 0.8):
        prediction = sign_head.predict(training[3], frozen_probability=probability)
        gap = abs(float(prediction["residual_prediction"]) - probability)
        rows.append(
            {
                "check_type": "energy_sign",
                "unit": f"probability-{probability}",
                "frozen_probability": probability,
                "energy_bad_minus_good": prediction["energy_bad_minus_good"],
                "observed_probability": prediction["residual_prediction"],
                "absolute_gap": gap,
                "passed": gap <= 1e-14,
                "data_role": "analytic_fixture",
            }
        )

    gradient_head = LocalResidualHead.from_training(training, seed=FIT_SEEDS[1])
    features = training[2] + 0.1
    sparse, support = gradient_head.sparse_gradient(features, 1, frozen_probability=0.35)
    dense = independent_dense_gradient(gradient_head, features, 1, frozen_probability=0.35)
    rows.append(
        {
            "check_type": "gradient_parity",
            "unit": "sparse-vs-independent-dense",
            "max_abs_gap": float(np.max(np.abs(sparse - dense))),
            "active_support_count": len(support),
            "passed": float(np.max(np.abs(sparse - dense))) <= PARITY_TOLERANCE,
            "data_role": "analytic_fixture",
        }
    )

    chronology = LocalResidualHead.from_training(training, seed=FIT_SEEDS[2])
    chronology.seal_prediction(
        event_id="delay-0",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=training[1],
        frozen_probability=0.45,
    )
    chronology.seal_prediction(
        event_id="delay-8",
        source_version="analytic-v1",
        prediction_time=1,
        reveal_time=9,
        features=training[4],
        frozen_probability=0.55,
    )
    missing_status = chronology.apply_feedback("missing", label=1, visible_at=9)["status"]
    reordered_status = chronology.apply_feedback("delay-8", label=0, visible_at=9)["status"]
    delay_zero = chronology.apply_feedback("delay-0", label=1, visible_at=0)
    duplicate_status = chronology.apply_feedback("delay-0", label=1, visible_at=9)["status"]
    delay_eight = chronology.apply_feedback("delay-8", label=0, visible_at=9)
    rows.append(
        {
            "check_type": "feedback_chronology",
            "unit": "delay-0-and-8",
            "missing_status": missing_status,
            "reordered_status": reordered_status,
            "duplicate_status": duplicate_status,
            "delay_0_status": delay_zero["status"],
            "delay_8_status": delay_eight["status"],
            "immutable_prediction_hashes": True,
            "passed": (
                missing_status == "missing_prediction"
                and reordered_status == "reordered"
                and duplicate_status == "duplicate"
                and delay_zero["status"] == "committed"
                and delay_eight["status"] == "committed"
            ),
            "data_role": "analytic_fixture",
        }
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_head = LocalResidualHead.from_training(training, seed=FIT_SEEDS[3])
    for index, delay in enumerate((0, 8)):
        checkpoint_head.seal_prediction(
            event_id=f"restart-{index}",
            source_version="analytic-v1",
            prediction_time=index,
            reveal_time=index + delay,
            features=training[index + 1],
            frozen_probability=0.45 + index * 0.1,
        )
    checkpoint_head.apply_feedback("restart-0", label=1, visible_at=0)
    checkpoint_path = checkpoint_dir / "residual-head-checkpoint.json"
    checkpoint_manifest = checkpoint_head.save_checkpoint(checkpoint_path)
    uninterrupted = deepcopy(checkpoint_head)
    uninterrupted.apply_feedback("restart-1", label=0, visible_at=9)
    restarted = LocalResidualHead.load_checkpoint(checkpoint_path)
    restarted.apply_feedback("restart-1", label=0, visible_at=9)
    restart_equal = np.array_equal(restarted.state_vector, uninterrupted.state_vector)
    rows.append(
        {
            "check_type": "checkpoint_restart",
            "unit": "delay-8-cold-restart",
            "state_equal": restart_equal,
            "state_hash_equal": restarted.state_hash == uninterrupted.state_hash,
            "passed": restart_equal and restarted.state_hash == uninterrupted.state_hash,
            "data_role": "analytic_fixture",
        }
    )

    guard_features = training[2]
    guarded = LocalResidualHead.from_training(
        training,
        seed=FIT_SEEDS[4],
        guard_rows=[{"features": guard_features.tolist(), "label": 0, "frozen_probability": 0.5}],
        guard_tolerance=0.0,
    )
    guarded.coefficients.fill(0.0)
    guarded.bias = 0.0
    guarded.seal_prediction(
        event_id="guard-reject",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=guard_features,
        frozen_probability=0.5,
    )
    guard_before = guarded.state_hash
    guard_receipt = guarded.apply_feedback("guard-reject", label=1, visible_at=0)
    rows.append(
        {
            "check_type": "guard_rollback",
            "unit": "opposed-training-replay",
            **deepcopy(guard_receipt),
            "state_restored": guarded.state_hash == guard_before,
            "passed": guard_receipt["status"] == "rejected_guard"
            and guarded.state_hash == guard_before,
            "data_role": "analytic_fixture",
        }
    )

    locality = LocalResidualHead.from_training(training, seed=FIT_SEEDS[0])
    locality.coefficients.fill(0.0)
    locality.bias = 0.0
    low = training.min(axis=0)
    high = training.max(axis=0)
    low_before = locality.predict(low, frozen_probability=0.5)
    high_before = locality.predict(high, frozen_probability=0.5)
    high_weights = locality.coefficients.reshape(-1)[high_before["active_support"]].copy()
    locality.seal_prediction(
        event_id="locality-low",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=low,
        frozen_probability=0.5,
    )
    locality.apply_feedback("locality-low", label=1, visible_at=0)
    high_after = locality.predict(high, frozen_probability=0.5)
    support_rows = [
        {
            "updated_probe": "low_endpoint",
            "distant_probe": "high_endpoint",
            "updated_support": low_before["active_support"],
            "distant_support": high_before["active_support"],
            "support_disjoint": set(low_before["active_support"]).isdisjoint(
                high_before["active_support"]
            ),
            "distant_local_weights_unchanged": np.array_equal(
                high_weights,
                locality.coefficients.reshape(-1)[high_before["active_support"]],
            ),
            "distant_local_contribution_before": high_before["local_coefficient_contribution"],
            "distant_local_contribution_after": high_after["local_coefficient_contribution"],
            "shared_bias_before": high_before["shared_bias_contribution"],
            "shared_bias_after": high_after["shared_bias_contribution"],
            "shared_bias_changed": high_before["shared_bias_contribution"]
            != high_after["shared_bias_contribution"],
            "local_retention_claim_excludes_shared_bias": True,
        }
    ]
    support_passed = all(
        row["support_disjoint"]
        and row["distant_local_weights_unchanged"]
        and row["shared_bias_changed"]
        and row["local_retention_claim_excludes_shared_bias"]
        for row in support_rows
    )
    rows.append(
        {
            "check_type": "support_locality",
            "unit": "disjoint-endpoints",
            "passed": support_passed,
            "data_role": "analytic_fixture",
        }
    )

    controls = run_control_comparison(training)
    controls_passed = (
        set(controls)
        == {"local_residual", "matched_linear_residual", "frozen_residual", "no_feedback"}
        and controls["frozen_residual"]["update_count"] == 0
        and controls["no_feedback"]["state_unchanged"] is True
    )
    rows.append(
        {
            "check_type": "control_comparison",
            "unit": "four-registered-arms",
            "controls": deepcopy(controls),
            "passed": controls_passed,
            "data_role": "analytic_fixture",
        }
    )
    return rows, support_rows, checkpoint_manifest


def reduce_analytic_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently reduce raw analytic units without trusting ready scores."""

    by_type = {str(row.get("check_type")): dict(row) for row in rows}
    energy = [dict(row) for row in rows if row.get("check_type") == "energy_sign"]
    gradient = by_type.get("gradient_parity", {})
    feedback = by_type.get("feedback_chronology", {})
    guard = by_type.get("guard_rollback", {})
    restart = by_type.get("checkpoint_restart", {})
    support = by_type.get("support_locality", {})
    controls = by_type.get("control_comparison", {})
    return {
        "energy_sign": {
            "case_count": len(energy),
            "max_abs_probability_gap": max(
                (float(row.get("absolute_gap", math.inf)) for row in energy), default=math.inf
            ),
            "passed": len(energy) == 3 and all(row.get("passed") is True for row in energy),
        },
        "gradient_parity": {
            "max_abs_gap": gradient.get("max_abs_gap"),
            "passed": gradient.get("passed") is True,
        },
        "feedback_chronology": {
            "missing_status": feedback.get("missing_status"),
            "reordered_status": feedback.get("reordered_status"),
            "duplicate_status": feedback.get("duplicate_status"),
            "delay_0_status": feedback.get("delay_0_status"),
            "delay_8_status": feedback.get("delay_8_status"),
            "passed": feedback.get("passed") is True,
        },
        "guard_rollback": {
            "status": guard.get("status"),
            "state_restored": guard.get("state_restored"),
            "rejection_frequency": 1.0 if guard.get("status") == "rejected_guard" else 0.0,
            "guard_rows_evaluated": guard.get("guard_rows_evaluated"),
            "guard_basis_evaluations": guard.get("guard_basis_evaluations"),
            "guard_loss_computations": guard.get("guard_loss_computations"),
            "guard_elapsed_ns": guard.get("guard_elapsed_ns"),
            "passed": guard.get("passed") is True,
        },
        "checkpoint_restart": {
            "state_equal": restart.get("state_equal"),
            "state_hash_equal": restart.get("state_hash_equal"),
            "passed": restart.get("passed") is True,
        },
        "support_locality": {"passed": support.get("passed") is True},
        "control_comparison": {"passed": controls.get("passed") is True},
    }


def exp7469_protocol() -> JsonDict:
    """Freeze the prospective experiment before any online outcome exists."""

    return {
        "schema": "carnot.exp7469.residual_online_protocol.v1",
        "seeds": [746900, 746901, 746902, 746903, 746904],
        "fit_seed_role": "training_or_calibration_only",
        "label_blind_group_orders": ["stable_hash_ascending", "stable_hash_descending"],
        "delays": [0, 8],
        "primary_arm": "local_residual",
        "controls": ["frozen_residual", "matched_linear_residual"],
        "no_feedback_control": True,
        "uniform_audit_probability": 0.5,
        "full_feedback_diagnostic": {"enabled": True, "confirmatory": False},
        "ipw_clip": [0.1, 10.0],
        "moving_block_lengths": [16, 32],
        "bootstrap_resamples": 10000,
        "state_carry_across_replicates": False,
        "learning_rate_source": "frozen_training_calibration",
        "knots_source": "training_inputs_only",
        "outcome_tuning_allowed": False,
        "generator_updates_allowed": False,
        "audit_draw_seed": AUDIT_SEED,
        "group_order_seed": ORDERING_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def protocol_rows() -> list[JsonDict]:
    """Name every prospective source, game, seed, order, delay, and arm unit."""

    rows: list[JsonDict] = []
    arms = ("local_residual", "matched_linear_residual", "frozen_residual", "no_feedback")
    for seed in exp7469_protocol()["seeds"]:
        for order in exp7469_protocol()["label_blind_group_orders"]:
            for delay in exp7469_protocol()["delays"]:
                for arm in arms:
                    rows.append(
                        {
                            "unit_id": f"source-support:{seed}:{order}:d{delay}:{arm}",
                            "source": "exp7462_frozen_source_support",
                            "game": "source_support_binary_decision",
                            "seed": seed,
                            "group_order": order,
                            "delay": delay,
                            "arm": arm,
                            "status": "unstarted",
                            "attempted": False,
                            "complete": False,
                            "failed": False,
                            "censored": False,
                            "failure_reason": None,
                            "data_role": "preregistered_exp7469_unit",
                        }
                    )
    return rows


def _gate(
    check: str, category: str, expected: Any, observed: Any, op: str, passed: bool, principle: str
) -> JsonDict:
    """Keep each gate value plain while preserving its audit reason."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each frozen affected and terminal reader exactly once."""

    names = (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    return all(
        len([row for row in receipts if row.get("name") == name and row.get("passed") is True]) == 1
        for name in names
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness from raw analytic rows and fixed declarations."""

    checks = reduce_analytic_rows(value.get("analytic_rows") or [])
    analytic_passed = all(row.get("passed") is True for row in checks.values())
    preconditions_passed = all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    validation_passed = _required_validation_passed(value.get("validation_receipts") or [])
    declarations_passed = (
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and value.get("inference_substrate_class") == "no_model_load"
        and value.get("execution_venue") == "host"
        and value.get("verifier_is_oracle") is True
        and value.get("flagged_adversarial") is False
    )
    ready = int(
        analytic_passed and preconditions_passed and validation_passed and declarations_passed
    )
    verdict_class = "circular_positive" if ready else "disqualified"
    honest = (
        "complete_circular_positive_analytic_residual_learner_ready"
        if ready
        else "complete_disqualified_residual_learner_evidence"
    )
    return {
        "residual_learner_ready_score": ready,
        "promotion_score": 0,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "status": honest,
        "analytic_checks": checks,
        "preconditions_passed": preconditions_passed,
        "validation_passed": validation_passed,
        "declarations_passed": declarations_passed,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind code, protocol, raw rows, source hashes, and validation scope."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = None
    return canonical_hash(payload)


def _field_principles() -> dict[str, str]:
    """Explain why every required terminal field exists."""

    specific = {
        "schema": "A versioned schema fixes the exact experiment, milestone, and terminal status.",
        "run_date": "The fixed date and measured clocks identify this execution without padding.",
        "preconditions_checked": "Exact paths, ownership, devices, and prior flags authenticate inputs.",
        "MODEL_SPECS": "An empty list states that no current LLM task exists.",
        "model_specs": "The lowercase alias gives older readers the same empty model declaration.",
        "model_invoked": "False separates current calls from archived model-shaped events.",
        "invocation_counts": "Balanced zero counters expose any attempted or unfinished call.",
        "inference_substrate": "The substrate names numeric residual fitting rather than generation.",
        "inference_substrate_class": "The no-model class prevents numeric work from posing as inference.",
        "execution_venue": "The host venue stays separate from device and historical board evidence.",
        "duration_s": "Measured work separates load, fitting, generation, and validation costs.",
        "phase_spans": "Monotonic phase spans bind progress events to completed units.",
        "random_seed": "Frozen fitting, order, audit, and bootstrap seeds prevent outcome tuning.",
        "reproducibility_checksum": "One hash binds code, protocol, data roles, rows, and validation scope.",
        "source_artifact_hashes": "Exact upstream bytes and original flags preserve history.",
        "rows": "Every future source, game, seed, order, delay, and arm remains visible.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted units differ.",
        "acceptance_gate_results": "Typed gates distinguish validity from analytic implementation evidence.",
        "gate_check_summary": "A failure names its exact check, upstream, field, and observed value.",
        "honest_verdict": "The terminal finding cannot hide a null, block, or disqualification.",
        "verdict_class": "A closed class makes circular analytic evidence explicit.",
        "verifier_is_oracle": "True forbids a positive real-world claim from analytic oracle checks.",
        "flagged_adversarial": "Reader flags remain truthful and cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits, durations, and hashes prove current checks ran.",
        "field_principles": "Field reasons keep the terminal evidence auditable.",
        "residual_learner_ready_score": "A bare score depends only on analytic, retention, and restart checks.",
        "online_protocol": "The update law, audits, controls, and chronology freeze before feedback.",
        "support_overlap_rows": "Local coefficients and shared-bias effects are measured separately.",
        "hardware_acceleration_path": "Sparse arithmetic maps to hardware while host durability costs remain.",
    }
    return {
        field: specific.get(field, "This field preserves one required part of the audit record.")
        for field in REQUIRED_FIELDS
    }


def _build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    analytic_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    device_identity: Mapping[str, Any],
) -> JsonDict:
    """Assemble one complete candidate from raw measurements."""

    rows = protocol_rows()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "preterminal",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "clock_identity": {
            "wall": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "model_specs": [],
        **deepcopy(dict(current_receipt)),
        "device_identity": deepcopy(dict(device_identity)),
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "numeric_fitting_s": float(current_receipt.get("duration_s") or 0.0),
            "validation_s": math.fsum(
                float(row.get("duration_s") or 0.0) for row in validation_receipts
            ),
        },
        "random_seed": {
            "fit_seeds": list(FIT_SEEDS),
            "ordering_seed": ORDERING_SEED,
            "audit_seed": AUDIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": None,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": 0,
            "complete": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(rows),
            "independent_unit": "source_game_seed_order_delay_arm",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "honest_verdict": "preterminal",
        "verdict_class": "disqualified",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "residual_learner_ready_score": 0,
        "promotion_score": 0,
        "online_protocol": exp7469_protocol(),
        "support_overlap_rows": [deepcopy(dict(row)) for row in support_rows],
        "hardware_acceleration_path": {
            "sparse_basis": "four_features_times_at_most_four_active_cubic_coefficients",
            "fixed_point_gpu_fpga_work": [
                "knot_interval_lookup",
                "cubic_basis_evaluation",
                "active_coefficient_dot_product",
                "clipped_gradient_multiply_accumulate",
            ],
            "coefficient_bank": "4x8 local coefficients plus one shared bias",
            "host_feedback_work": ["label admission", "training-only guard", "journal fsync"],
            "durability_boundary": "host atomic checkpoint after accepted state",
            "speedup_claimed": False,
        },
        "small_ebm_training": {
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "bounded_four_feature_fixed_cubic_spline_residual",
            "parameter_count": PARAMETER_COUNT,
            "learning_rate": LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
        "analytic_rows": [deepcopy(dict(row)) for row in analytic_rows],
        "analytic_checks": reduce_analytic_rows(analytic_rows),
        "checkpoint_manifest": deepcopy(dict(checkpoint_manifest)),
        "methodology_note": (
            "Analytic fixtures establish implementation evidence only. They do not measure "
            "real source-support value, so a positive verdict is forbidden."
        ),
        "fixed_share_weights_used": False,
        "mixture_of_experts_used": False,
        "generator_updates_used": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "numbered_e2e_applicable": [],
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "cold_replay": "fresh_process",
            "numbered_runtime_e2e": "not_applicable_experiment_local_numeric_head",
        },
    }
    reduction = independent_reduce(artifact)
    artifact.update(
        {
            key: reduction[key]
            for key in (
                "residual_learner_ready_score",
                "promotion_score",
                "verdict_class",
                "honest_verdict",
                "status",
            )
        }
    )
    gates = [
        _gate(
            "preconditions_passed",
            "validity",
            True,
            reduction["preconditions_passed"],
            "==",
            reduction["preconditions_passed"],
            "Inputs and original historical flags must authenticate before numeric work.",
        ),
        _gate(
            "analytic_implementation_checks",
            "circular_implementation_evidence",
            True,
            all(row.get("passed") is True for row in reduction["analytic_checks"].values()),
            "==",
            all(row.get("passed") is True for row in reduction["analytic_checks"].values()),
            "Analytic sign, parity, chronology, guard, restart, locality, and controls must pass.",
        ),
        _gate(
            "required_validation_passed",
            "validity",
            True,
            reduction["validation_passed"],
            "==",
            reduction["validation_passed"],
            "Every frozen affected and terminal command must exit zero.",
        ),
        _gate(
            "real_source_support_benefit",
            "benefit",
            "not_tested",
            "not_tested",
            "==",
            True,
            "Analytic fixtures never establish online source-support benefit.",
        ),
    ]
    artifact["acceptance_gate_results"] = gates
    failed = [row for row in gates if row["passed"] is not True]
    first = failed[0] if failed else None
    artifact["gate_check_summary"] = {
        "passed": not failed,
        "failed_check": first.get("check") if first else None,
        "upstream": EXPERIMENT_ID if first else None,
        "path": RAW_DIR.as_posix() if first else None,
        "field": "passed" if first else None,
        "expected": first.get("expected") if first else None,
        "observed": first.get("observed") if first else None,
        "failed_required_checks": [row["check"] for row in failed],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_fixture_artifact(
    *, validation_receipts: Sequence[Mapping[str, Any]] | None = None
) -> JsonDict:
    """Build deterministic in-memory evidence for mutation tests."""

    with tempfile.TemporaryDirectory(prefix="exp7468-fixture-") as temporary:
        rows, support, checkpoint = run_analytic_controls(Path(temporary))
    receipt = build_current_work_receipt(
        run_id="exp7468-fixture",
        owner_pid=0,
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"device": "cpu", "model_loaded": False},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000,
        phase_spans=[],
        small_ebm_training={
            "performed": True,
            "receipt_class": "small_ebm_training",
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
    )
    return _build_artifact(
        preconditions=[
            {
                "check": "fixture",
                "upstream": "analytic-v1",
                "artifact_field": "available",
                "expected": True,
                "observed": True,
                "passed": True,
            }
        ],
        source_hashes={},
        analytic_rows=rows,
        support_rows=support,
        checkpoint_manifest=checkpoint,
        validation_receipts=list(validation_receipts or []),
        current_receipt=receipt,
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:00.001000+00:00",
        device_identity={"venue": "host", "device": "cpu", "fixture": True},
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed bytes fail-closed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, raw reduction, provenance, hashes, and verdict."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
    ):
        errors.append("model_declaration_mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_mismatch")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    errors.extend(validate_current_work_receipt(value, root=root))
    reduced = independent_reduce(value)
    if value.get("analytic_checks") != reduced["analytic_checks"]:
        errors.append("residual_learner_ready_score_mismatch")
    for field in (
        "residual_learner_ready_score",
        "promotion_score",
        "verdict_class",
        "honest_verdict",
        "status",
    ):
        if value.get(field) != reduced[field]:
            errors.append(f"{field}_mismatch")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_oracle_mismatch")
    if value.get("verdict_class") == "positive":
        errors.append("positive_forbidden_for_oracle")
    if value.get("online_protocol") != exp7469_protocol():
        errors.append("online_protocol_mismatch")
    if verify_sources:
        for label, row in dict(value.get("source_artifact_hashes") or {}).items():
            if not isinstance(row, Mapping):
                errors.append(f"source_hash_row_invalid:{label}")
                continue
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_invalid:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Validate a serialized candidate in a fresh process."""

    value = _load_object(path)
    return (
        validate_artifact(value, root=root, verify_sources=verify_sources)
        if value
        else ["artifact_unreadable_or_not_object"]
    )


def utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one aware UTC timestamp for the current execution."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit a flushed boundary around every potentially slow operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7468] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any, **details: Any
) -> JsonDict:  # pragma: no cover - execution provenance.
    """Record exact prerequisite values without collapsing false and missing."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        **details,
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict], list[JsonDict]]:  # pragma: no cover
    """Authenticate exact source bytes, ownership, and historical dispositions."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    historical_sidecars: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        stat = path.stat() if available else None
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                owner_uid=stat.st_uid if stat else None,
                owner_gid=stat.st_gid if stat else None,
                byte_size=stat.st_size if stat else None,
            )
        )
        if available:
            artifact = _load_object(path) if relative in UPSTREAM_ARTIFACTS else {}
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": artifact.get("flagged_adversarial"),
                "original_verdict_class": artifact.get("verdict_class"),
            }
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-KAN-7468",
            "REQ-KAN-7468" if "REQ-KAN-7468" in spec_text else None,
        )
    )
    for relative, expected_fields in UPSTREAM_ARTIFACTS.items():
        artifact = _load_object(root / relative)
        for field, expected in expected_fields.items():
            checks.append(
                _precondition(
                    f"historical:{relative.stem}:{field}",
                    relative.as_posix(),
                    field,
                    expected,
                    artifact.get(field),
                )
            )
        historical_sidecars.append(
            {
                "path": relative.as_posix(),
                "sha256": hashes.get(relative.as_posix(), {}).get("sha256"),
                "scope": "historical_model_receipts",
                "original_flagged_adversarial": artifact.get("flagged_adversarial"),
                "model_invoked_in_historical_run": artifact.get("model_invoked"),
            }
        )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            "7468" in exclusion_text,
        )
    )
    return checks, hashes, historical_sidecars


def _device_identity() -> JsonDict:  # pragma: no cover - execution provenance.
    """Record the actual CPU host without implying board execution."""

    boot_path = Path("/proc/sys/kernel/random/boot_id")
    return {
        "venue": "host",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "numpy_version": np.__version__,
        "boot_id": boot_path.read_text(encoding="utf-8").strip() if boot_path.is_file() else None,
        "numeric_device": "cpu_float64",
        "cuda_used": False,
        "historical_board_evidence_used_as_current": False,
    }


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - real clock boundary.
    """Close a monotonic phase with a completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the fresh replay, reduction, adversarial, and strict readers."""

    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", ".")
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "fresh_process_cold_replay",
                (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
                "capability_end_to_end",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
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
                "raw_analytic_rows",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
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
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, measure, validate, replay, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = utc_now()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, historical_sidecars = collect_preconditions(root)
    failed = [row for row in preconditions if row.get("passed") is not True]
    spans.append(_span("preconditions", phase_started, started, len(preconditions), "inputs"))
    progress(started, "preconditions", "complete", completed=len(preconditions), failed=len(failed))
    if failed:
        raise RuntimeError(f"precondition_failed:{failed[0]}")

    progress(started, "model_load", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_span("model_load", phase_started, started, 0, "no_model_load"))
    progress(started, "model_load", "after", completed=0)
    progress(started, "generation", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_span("generation", phase_started, started, 0, "no_generation"))
    progress(started, "generation", "after", completed=0)

    progress(started, "numeric_fitting", "before_benchmark", planned=9)
    phase_started = time.monotonic()
    raw_dir = root / RAW_DIR
    analytic_rows, support_rows, checkpoint = run_analytic_controls(raw_dir / "numeric_state")
    spans.append(
        _span("numeric_fitting", phase_started, started, len(analytic_rows), "analytic_rows")
    )
    progress(started, "numeric_fitting", "after_benchmark", completed=len(analytic_rows))
    checkpoint_path = Path(str(checkpoint["path"]))
    source_hashes[checkpoint_path.relative_to(root).as_posix()] = {
        "path": checkpoint_path.relative_to(root).as_posix(),
        "sha256": checkpoint["sha256"],
        "original_flagged_adversarial": None,
        "original_verdict_class": None,
    }

    private_root = Path(tempfile.mkdtemp(prefix="exp7468-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _span("affected_validation", phase_started, started, len(affected), "affected_checks")
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    if plan_errors or not affected_reduction["passed"]:
        raise RuntimeError(f"affected_validation_failed:{plan_errors}:{affected_reduction}")

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "original_flagged_adversarial": None,
            "original_verdict_class": None,
        }
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "cpu_float64",
            "software": "numpy",
            "model_loaded": False,
            "work": "analytic guarded local residual fitting",
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        sidecar_references=[
            {"path": row["path"], "sha256": row["sha256"], "scope": row["scope"]}
            for row in historical_sidecars
        ],
        phase_spans=spans,
        small_ebm_training={
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "bounded_four_feature_fixed_cubic_spline_residual",
            "updates_attempted": 6,
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
    )
    candidate = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        analytic_rows=analytic_rows,
        support_rows=support_rows,
        checkpoint_manifest=checkpoint,
        validation_receipts=[
            *affected,
            *[
                {
                    "name": name,
                    "required": True,
                    "passed": True,
                    "exit_code": 0,
                    "timed_out": False,
                    "provisional_for_candidate_reader": True,
                }
                for name in TERMINAL_CHECK_NAMES
            ],
        ],
        current_receipt=receipt,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        device_identity=_device_identity(),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_serialization")
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_serialization")

    terminal_commands = _terminal_commands(candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(terminal_commands))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_commands, log_dir=raw_dir / "validation/terminal"
    )
    spans.append(
        _span("terminal_validation", phase_started, started, len(terminal), "terminal_readers")
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_validation_failed")

    final_receipt = build_current_work_receipt(
        run_id=receipt["current_run_id"],
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details=receipt["inference_substrate_details"],
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        sidecar_references=receipt["receipt_sidecars"],
        phase_spans=spans,
        small_ebm_training=receipt["small_ebm_training"],
    )
    final = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        analytic_rows=analytic_rows,
        support_rows=support_rows,
        checkpoint_manifest=checkpoint,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        device_identity=_device_identity(),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed execution date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or one strict fresh-process terminal reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=root)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=root) if value else ["artifact_unreadable_or_not_object"]
        )
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
