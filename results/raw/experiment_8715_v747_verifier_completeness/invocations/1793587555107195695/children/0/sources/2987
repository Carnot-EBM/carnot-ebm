"""Prototype a complete per-knot anchor on the Exp7468 residual head.

The code reuses the qualified spline head and immutable prediction ledger. It
adds only diagonal importance, consolidation, and full-vector anchoring.
Analytic fixtures can verify this mechanism but cannot establish held-out
learning value. Spec refs: REQ-KAN-7482 and SCENARIO-KAN-7482-*.
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
import re
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
from carnot.experiment_7468_v654_residual_learner import (
    COEFFICIENTS_PER_INPUT,
    FEATURE_COUNT,
    GRADIENT_NORM_CAP,
    GUARD_TOLERANCE,
    KNOT_VECTOR_SIZE,
    LocalResidualHead,
    _clip_probability,
    _finite_array,
    fit_local_knots,
    local_design,
    prediction_event_hash,
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
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
PHASE = 3
EXPERIMENT_ID = "exp7482-importance-anchor"
SCHEMA = "carnot.exp7482.v655.importance_anchor.v1"
CHECKPOINT_SCHEMA = "carnot.exp7482.importance_anchor_checkpoint.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7482_v655_importance_anchor.json")
RAW_DIR = Path("results/raw/experiment_7482_v655_importance_anchor")
MODULE_PATH = Path("python/carnot/experiment_7482_v655_importance_anchor.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7482_v655_importance_anchor.py")
TEST_PATH = Path("tests/python/test_experiment_7482_v655_importance_anchor.py")
SPEC_PATH = Path("openspec/capabilities/kan/spec.md")

COEFFICIENT_COUNT = FEATURE_COUNT * COEFFICIENTS_PER_INPUT
PARAMETER_COUNT = COEFFICIENT_COUNT + 1
LEARNING_RATE_CANDIDATES = (0.02, 0.04, 0.06)
SELECTED_LEARNING_RATE = 0.04
REPLAY_BUFFER_SIZE = 1
UPDATE_OPPORTUNITIES = 24
FIT_SEEDS = (748200, 748201, 748202)
ORDERING_SEED = 748211
AUDIT_SEED = 748221
BOOTSTRAP_SEED = 748231
FINITE_DIFFERENCE_STEP = 1e-6
FINITE_DIFFERENCE_TOLERANCE = 1e-8
FIXTURE_TIMEOUT_S = 600.0
INFERENCE_SUBSTRATE = "numpy_float64_prequential_importance_anchor_training"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

ARM_CONFIGS: dict[str, dict[str, float | str]] = {
    "unanchored": {"anchor_mode": "none", "anchor_lambda": 0.0, "learning_rate": 0.04},
    "uniform_anchor": {
        "anchor_mode": "uniform",
        "anchor_lambda": 6.0,
        "learning_rate": 0.04,
    },
    "importance_anchor": {
        "anchor_mode": "importance",
        "anchor_lambda": 6.0,
        "learning_rate": 0.04,
    },
    "excessive_anchor": {
        "anchor_mode": "importance",
        "anchor_lambda": 500.0,
        "learning_rate": 0.04,
    },
    "zero_learning_rate": {
        "anchor_mode": "importance",
        "anchor_lambda": 6.0,
        "learning_rate": 0.0,
    },
}

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
    Path("results/experiment_7468_v654_residual_learner.json"): {
        "status": "complete_circular_positive_analytic_residual_learner_ready",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "residual_learner_ready_score": 1,
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
    Path("python/carnot/experiment_7468_v654_residual_learner.py"),
    Path("python/carnot/experiment_7425_v651_spline_prototype.py"),
    Path("python/carnot/experiment_7450_v653_prediction_ledger.py"),
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
    "importance_anchor_ready_score",
    "anchor_definition",
    "active_and_total_work",
    "state_replay_checks",
)


class ImportanceAnchorHead(LocalResidualHead):
    """Add a complete diagonal coefficient anchor to the residual head.

    The data gradient stays local. The anchor gradient is dense because every
    protected coefficient can move back toward its consolidated reference.
    """

    def __init__(
        self,
        knots: Any,
        coefficients: Any,
        bias: float,
        *,
        anchor_mode: str,
        anchor_lambda: float,
        learning_rate: float,
        residual_bound: float = 2.0,
        replay_rows: Sequence[Mapping[str, Any]] = (),
        guard_tolerance: float = GUARD_TOLERANCE,
    ) -> None:
        if anchor_mode not in {"none", "uniform", "importance"}:
            raise ValueError("anchor_mode_invalid")
        scalars = (float(anchor_lambda), float(learning_rate))
        if not all(math.isfinite(value) for value in scalars) or any(
            value < 0.0 for value in scalars
        ):
            raise ValueError("anchor_scalar_range_invalid")
        # Exp7468 requires a positive rate. A tiny constructor value lets this
        # subclass retain its validation while exposing an explicit zero-rate control.
        # guard_tolerance was silently defaulted from the base class before this fix;
        # it is now an explicit, backward-compatible parameter (see mypy override note
        # on from_training below).
        super().__init__(
            knots,
            coefficients,
            bias,
            learning_rate=max(float(learning_rate), np.finfo(np.float64).tiny),
            residual_bound=residual_bound,
            guard_tolerance=guard_tolerance,
        )
        self.learning_rate = float(learning_rate)
        self.anchor_mode = anchor_mode
        self.anchor_lambda = float(anchor_lambda)
        self.importance_accumulator = np.zeros(COEFFICIENT_COUNT, dtype=np.float64)
        self.importance_observations = 0
        self.importance = np.zeros(COEFFICIENT_COUNT, dtype=np.float64)
        self.reference_coefficients = self.coefficients.reshape(-1).copy()
        self.consolidation_count = 0
        self.replay_rows = [self._validate_guard_row(row) for row in replay_rows]

    @classmethod
    def from_training(
        cls,
        training_features: Any,
        *,
        seed: int,
        # Defaults added so this override stays callable with every argument the base
        # LocalResidualHead.from_training accepts (mypy: "Signature of from_training
        # incompatible with supertype"). "none"/0.0/0.0 is an anchor-disabled head,
        # the closest behavioral match to the plain base class. Every existing caller
        # already passes these three explicitly, so this is not a behavior change.
        anchor_mode: str = "none",
        anchor_lambda: float = 0.0,
        learning_rate: float = 0.0,
        guard_rows: Sequence[Mapping[str, Any]] = (),
        guard_tolerance: float = GUARD_TOLERANCE,
    ) -> ImportanceAnchorHead:
        """Initialize the same 32-coefficient head for every comparison arm."""

        knots = fit_local_knots(training_features)
        rng = np.random.default_rng(int(seed))
        coefficients = rng.normal(0.0, 0.01, size=(FEATURE_COUNT, COEFFICIENTS_PER_INPUT))
        return cls(
            knots,
            coefficients,
            0.0,
            anchor_mode=anchor_mode,
            anchor_lambda=anchor_lambda,
            learning_rate=learning_rate,
            # guard_rows is the base class's name for what this subclass calls
            # replay_rows -- both are validated by the same _validate_guard_row.
            replay_rows=guard_rows,
            guard_tolerance=guard_tolerance,
        )

    @property
    def state_hash(self) -> str:
        """Bind predictions to numeric, importance, reference, and replay state."""

        return canonical_hash(
            {
                "knots": self.knots.tolist(),
                "state": self.state_vector.tolist(),
                "learning_rate": self.learning_rate,
                "residual_bound": self.residual_bound,
                "anchor_mode": self.anchor_mode,
                "anchor_lambda": self.anchor_lambda,
                "importance_accumulator": self.importance_accumulator.tolist(),
                "importance_observations": self.importance_observations,
                "importance": self.importance.tolist(),
                "reference_coefficients": self.reference_coefficients.tolist(),
                "consolidation_count": self.consolidation_count,
                "replay_rows": deepcopy(self.replay_rows),
            }
        )

    def seal_prediction(self, **kwargs: Any) -> JsonDict:
        """Retain the feature vector and pre-update probability in the ledger."""

        event = super().seal_prediction(**kwargs)
        identity = str(event["event_id"])
        stored = self._predictions[identity]
        stored["pre_update_probability"] = stored["residual_prediction"]
        stored["prediction_event_hash"] = prediction_event_hash(stored)
        return deepcopy(stored)

    def set_replay_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Freeze the same previously revealed replay rows for every arm."""

        self.replay_rows = [self._validate_guard_row(row) for row in rows]

    def consolidate(self) -> JsonDict:
        """Freeze bounded empirical-Fisher importance and a coefficient reference."""

        if self.importance_observations <= 0:
            raise ValueError("importance_requires_revealed_label")
        estimate = self.importance_accumulator / self.importance_observations
        self.importance = np.clip(estimate, 0.0, 1.0)
        self.reference_coefficients = self.coefficients.reshape(-1).copy()
        self.consolidation_count += 1
        return {
            "importance_min": float(np.min(self.importance)),
            "importance_max": float(np.max(self.importance)),
            "importance_nonzero_count": int(np.count_nonzero(self.importance)),
            "revealed_label_count": self.importance_observations,
            "reference_hash": canonical_hash(self.reference_coefficients.tolist()),
        }

    def _anchor_weights(self) -> np.ndarray:
        """Return the registered diagonal without changing its total scale."""

        if self.anchor_mode == "none":
            return np.zeros(COEFFICIENT_COUNT, dtype=np.float64)
        if self.anchor_mode == "uniform":
            return np.full(COEFFICIENT_COUNT, float(np.mean(self.importance)))
        return self.importance.copy()

    def loss_and_gradient(
        self, features: Any, label: int, *, frozen_probability: float
    ) -> tuple[float, np.ndarray, JsonDict]:
        """Return replay-averaged log loss plus the complete anchor gradient."""

        current_gradient, current_support, current_loss = self._gradient(
            features, label, frozen_probability
        )
        gradients = [current_gradient]
        losses = [current_loss]
        active = set(current_support)
        for row in self.replay_rows:
            replay_gradient, replay_support, replay_loss = self._gradient(
                row["features"], int(row["label"]), float(row["frozen_probability"])
            )
            gradients.append(replay_gradient)
            losses.append(replay_loss)
            active.update(replay_support)
        data_gradient = np.mean(np.asarray(gradients, dtype=np.float64), axis=0)
        data_loss = math.fsum(losses) / len(losses)
        delta = self.coefficients.reshape(-1) - self.reference_coefficients
        weights = self._anchor_weights()
        anchor_loss = self.anchor_lambda * float(np.dot(weights, delta * delta))
        anchor_gradient = 2.0 * self.anchor_lambda * weights * delta
        complete = data_gradient.copy()
        complete[:-1] += anchor_gradient
        work = {
            "active_data_gradient_count": len(active),
            "total_anchor_coefficient_count": COEFFICIENT_COUNT,
            "nonzero_anchor_gradient_count": int(np.count_nonzero(anchor_gradient)),
            "replay_rows_evaluated": len(self.replay_rows),
            "anchor_loss": anchor_loss,
        }
        return data_loss + anchor_loss, complete, work

    def apply_feedback(self, event_id: str, *, label: int, visible_at: int) -> JsonDict:
        """Admit one label, update from prior importance, then observe this label."""

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

        observations_before = self.importance_observations
        state_hash_before = self.state_hash
        current_gradient, current_support, _current_loss = self._gradient(
            event["features"], int(label), float(event["frozen_prediction"])
        )
        loss, gradient, work = self.loss_and_gradient(
            event["features"], int(label), frozen_probability=float(event["frozen_prediction"])
        )
        raw_norm = float(np.linalg.norm(gradient))
        scale = min(1.0, GRADIENT_NORM_CAP / max(raw_norm, np.finfo(np.float64).tiny))
        clipped = gradient * scale
        self.coefficients.reshape(-1)[:] -= self.learning_rate * clipped[:-1]
        self.bias -= self.learning_rate * float(clipped[-1])
        self.importance_accumulator += np.clip(current_gradient[:-1] ** 2, 0.0, 1.0)
        self.importance_observations += 1
        self._feedback_fingerprints[identity] = fingerprint
        self.update_count += 1
        return {
            "event_id": identity,
            "prediction_event_hash": event["prediction_event_hash"],
            "status": "committed",
            "updates_applied": 1,
            "state_hash_before": state_hash_before,
            "state_hash_after": self.state_hash,
            "pre_update_probability": event["pre_update_probability"],
            "prediction_time_features": deepcopy(event["features"]),
            "loss": loss,
            "anchor_loss_before_current_label": work["anchor_loss"],
            "importance_observations_before": observations_before,
            "importance_observations_after": self.importance_observations,
            "active_data_support": sorted(set(current_support)),
            "active_data_gradient_count": work["active_data_gradient_count"],
            "total_anchor_coefficient_count": work["total_anchor_coefficient_count"],
            "nonzero_anchor_gradient_count": work["nonzero_anchor_gradient_count"],
            "replay_rows_evaluated": work["replay_rows_evaluated"],
            "gradient_norm_before_clip": raw_norm,
            "gradient_norm_after_clip": float(np.linalg.norm(clipped)),
        }

    def _checkpoint_payload(self) -> JsonDict:
        """Return every field needed for exact delayed-feedback continuation."""

        return {
            "schema": CHECKPOINT_SCHEMA,
            "knots": self.knots.tolist(),
            "coefficients": self.coefficients.tolist(),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "residual_bound": self.residual_bound,
            "anchor_mode": self.anchor_mode,
            "anchor_lambda": self.anchor_lambda,
            "importance_accumulator": self.importance_accumulator.tolist(),
            "importance_observations": self.importance_observations,
            "importance": self.importance.tolist(),
            "reference_coefficients": self.reference_coefficients.tolist(),
            "consolidation_count": self.consolidation_count,
            "replay_rows": deepcopy(self.replay_rows),
            "predictions": [deepcopy(self._predictions[key]) for key in self._prediction_order],
            "feedback_fingerprints": deepcopy(self._feedback_fingerprints),
            "update_count": self.update_count,
            "rejection_count": self.rejection_count,
        }

    @classmethod
    def load_checkpoint(cls, path: Path) -> ImportanceAnchorHead:
        """Authenticate and restore anchor, ledger, and acknowledged state."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
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
            anchor_mode=str(value["anchor_mode"]),
            anchor_lambda=float(value["anchor_lambda"]),
            learning_rate=float(value["learning_rate"]),
            residual_bound=float(value["residual_bound"]),
            replay_rows=list(value.get("replay_rows") or []),
        )
        restored.importance_accumulator = _finite_array(
            value["importance_accumulator"], (COEFFICIENT_COUNT,), "importance_accumulator"
        ).copy()
        restored.importance_observations = int(value["importance_observations"])
        restored.importance = _finite_array(
            value["importance"], (COEFFICIENT_COUNT,), "importance"
        ).copy()
        restored.reference_coefficients = _finite_array(
            value["reference_coefficients"], (COEFFICIENT_COUNT,), "reference_coefficients"
        ).copy()
        restored.consolidation_count = int(value["consolidation_count"])
        predictions = [dict(row) for row in value.get("predictions") or []]
        restored._prediction_order = [str(row["event_id"]) for row in predictions]
        restored._predictions = {str(row["event_id"]): deepcopy(row) for row in predictions}
        restored._feedback_fingerprints = {
            str(key): str(item)
            for key, item in dict(value.get("feedback_fingerprints") or {}).items()
        }
        restored.update_count = int(value.get("update_count", 0))
        restored.rejection_count = int(value.get("rejection_count", 0))
        return restored


def finite_difference_gradient(
    head: ImportanceAnchorHead,
    features: Any,
    label: int,
    *,
    frozen_probability: float,
    step: float = FINITE_DIFFERENCE_STEP,
) -> np.ndarray:
    """Differentiate the declared total loss without using its gradient path."""

    baseline = head.state_vector.copy()
    numeric = np.zeros_like(baseline)
    for index in range(baseline.size):
        plus = baseline.copy()
        minus = baseline.copy()
        plus[index] += step
        minus[index] -= step
        head.coefficients = plus[:-1].reshape(FEATURE_COUNT, COEFFICIENTS_PER_INPUT).copy()
        head.bias = float(plus[-1])
        plus_loss = head.loss_and_gradient(features, label, frozen_probability=frozen_probability)[
            0
        ]
        head.coefficients = minus[:-1].reshape(FEATURE_COUNT, COEFFICIENTS_PER_INPUT).copy()
        head.bias = float(minus[-1])
        minus_loss = head.loss_and_gradient(features, label, frozen_probability=frozen_probability)[
            0
        ]
        numeric[index] = (plus_loss - minus_loss) / (2.0 * step)
    head.coefficients = baseline[:-1].reshape(FEATURE_COUNT, COEFFICIENTS_PER_INPUT).copy()
    head.bias = float(baseline[-1])
    return numeric


def training_fixture() -> np.ndarray:
    """Return bounded training values with separated endpoint support."""

    base = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    return np.column_stack((base, base**3, np.sin(base), np.tanh(2.0 * base)))


def _binary_loss(head: ImportanceAnchorHead, features: Any, label: int) -> float:
    probability = float(head.predict(features, frozen_probability=0.5)["residual_prediction"])
    probability = _clip_probability(probability)
    return -math.log(probability if label == 1 else 1.0 - probability)


def _prepared_head(seed: int) -> tuple[ImportanceAnchorHead, list[JsonDict]]:
    """Train the shared prior task, then consolidate before any new label."""

    fixture = training_fixture()
    head = ImportanceAnchorHead.from_training(
        fixture,
        seed=seed,
        anchor_mode="none",
        anchor_lambda=0.0,
        learning_rate=SELECTED_LEARNING_RATE,
    )
    replay_rows: list[JsonDict] = []
    for index, features in enumerate((fixture[0], fixture[1], fixture[2], fixture[1])):
        probability = 0.5
        head.seal_prediction(
            event_id=f"prior-{index}",
            source_version="analytic-prior-v1",
            prediction_time=index,
            reveal_time=index,
            features=features,
            frozen_probability=probability,
        )
        head.apply_feedback(f"prior-{index}", label=1, visible_at=index)
        if index < REPLAY_BUFFER_SIZE:
            replay_rows.append(
                {
                    "features": np.asarray(features, dtype=np.float64).tolist(),
                    "label": 1,
                    "frozen_probability": probability,
                }
            )
    head.consolidate()
    return head, replay_rows


def _clone_for_arm(
    prepared: ImportanceAnchorHead, replay_rows: Sequence[Mapping[str, Any]], arm: str
) -> ImportanceAnchorHead:
    config = ARM_CONFIGS[arm]
    clone = ImportanceAnchorHead(
        prepared.knots,
        prepared.coefficients,
        prepared.bias,
        anchor_mode=str(config["anchor_mode"]),
        anchor_lambda=float(config["anchor_lambda"]),
        learning_rate=float(config["learning_rate"]),
        residual_bound=prepared.residual_bound,
        replay_rows=replay_rows,
    )
    clone.importance_accumulator = prepared.importance_accumulator.copy()
    clone.importance_observations = prepared.importance_observations
    clone.importance = prepared.importance.copy()
    clone.reference_coefficients = prepared.reference_coefficients.copy()
    clone.consolidation_count = prepared.consolidation_count
    return clone


def _trial_row(seed: int, arm: str) -> JsonDict:
    """Run one deterministic arm over overlap and disjoint conflict events."""

    prepared, replay_rows = _prepared_head(seed)
    head = _clone_for_arm(prepared, replay_rows, arm)
    fixture = training_fixture()
    overlap_features = fixture[1]
    disjoint_features = fixture[-1]
    _overlap_design, overlap_support = local_design(overlap_features, head.knots)
    _disjoint_design, disjoint_support = local_design(disjoint_features, head.knots)
    retained = np.flatnonzero(head.importance > 0.0)
    before_state = head.state_vector.copy()
    before_loss = (
        math.fsum(
            (_binary_loss(head, overlap_features, 0), _binary_loss(head, disjoint_features, 0))
        )
        / 2.0
    )
    receipts: list[JsonDict] = []
    for index in range(UPDATE_OPPORTUNITIES):
        features = overlap_features if index % 2 == 0 else disjoint_features
        head.seal_prediction(
            event_id=f"adapt-{index}",
            source_version="analytic-adaptation-v1",
            prediction_time=index,
            reveal_time=index,
            features=features,
            frozen_probability=0.5,
        )
        receipts.append(head.apply_feedback(f"adapt-{index}", label=0, visible_at=index))
    after_loss = (
        math.fsum(
            (_binary_loss(head, overlap_features, 0), _binary_loss(head, disjoint_features, 0))
        )
        / 2.0
    )
    drift = float(
        np.linalg.norm(
            head.coefficients.reshape(-1)[retained] - head.reference_coefficients[retained]
        )
    )
    return {
        "unit_id": f"analytic:{seed}:{arm}",
        "seed": seed,
        "arm": arm,
        "status": "complete",
        "attempted": True,
        "complete": True,
        "failed": False,
        "censored": False,
        "excluded": False,
        "failure_reason": None,
        "data_role": "analytic_fixture",
        "parameter_count": PARAMETER_COUNT,
        "coefficient_count": COEFFICIENT_COUNT,
        "replay_buffer_size": len(replay_rows),
        "learning_rate_candidates": list(LEARNING_RATE_CANDIDATES),
        "learning_rate_search_budget": len(LEARNING_RATE_CANDIDATES),
        "selected_learning_rate": SELECTED_LEARNING_RATE,
        "actual_learning_rate": head.learning_rate,
        "update_opportunities": UPDATE_OPPORTUNITIES,
        "updates_completed": len(receipts),
        "anchor_mode": head.anchor_mode,
        "anchor_lambda": head.anchor_lambda,
        "retained_support_count": int(retained.size),
        "retained_support_drift_l2": drift,
        "adaptation_loss_before": before_loss,
        "adaptation_loss_after": after_loss,
        "adaptation_loss_improvement": before_loss - after_loss,
        "state_unchanged": np.array_equal(before_state, head.state_vector),
        "overlap_support_count": len(set(overlap_support) & set(retained.tolist())),
        "disjoint_support_count": len(set(disjoint_support) - set(retained.tolist())),
        "max_active_data_gradient_count": max(
            int(receipt["active_data_gradient_count"]) for receipt in receipts
        ),
        "total_anchor_coefficient_count": max(
            int(receipt["total_anchor_coefficient_count"]) for receipt in receipts
        ),
        "importance_min": float(np.min(head.importance)),
        "importance_max": float(np.max(head.importance)),
        "pre_update_probabilities_recorded": all(
            "pre_update_probability" in receipt for receipt in receipts
        ),
        "prediction_time_features_recorded": all(
            len(receipt.get("prediction_time_features") or []) == FEATURE_COUNT
            for receipt in receipts
        ),
    }


def reduce_trial_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce every seed and arm without trusting stored benefit fields."""

    complete = [dict(row) for row in rows if row.get("complete") is True]
    expected = len(FIT_SEEDS) * len(ARM_CONFIGS)
    by_unit = {(int(row["seed"]), str(row["arm"])): row for row in complete}
    main_arms = ("unanchored", "uniform_anchor", "importance_anchor")
    fair = all(
        len(
            {
                (
                    by_unit[(seed, arm)]["parameter_count"],
                    by_unit[(seed, arm)]["replay_buffer_size"],
                    tuple(by_unit[(seed, arm)]["learning_rate_candidates"]),
                    by_unit[(seed, arm)]["selected_learning_rate"],
                    by_unit[(seed, arm)]["update_opportunities"],
                )
                for arm in main_arms
            }
        )
        == 1
        for seed in FIT_SEEDS
        if all((seed, arm) in by_unit for arm in main_arms)
    ) and all((seed, arm) in by_unit for seed in FIT_SEEDS for arm in main_arms)

    def mean_metric(arm: str, field: str) -> float:
        values = [
            float(by_unit[(seed, arm)][field]) for seed in FIT_SEEDS if (seed, arm) in by_unit
        ]
        return math.fsum(values) / len(values) if len(values) == len(FIT_SEEDS) else math.inf

    unanchored_drift = mean_metric("unanchored", "retained_support_drift_l2")
    importance_drift = mean_metric("importance_anchor", "retained_support_drift_l2")
    uniform_drift = mean_metric("uniform_anchor", "retained_support_drift_l2")
    importance_improvement = mean_metric("importance_anchor", "adaptation_loss_improvement")
    excessive_improvement = mean_metric("excessive_anchor", "adaptation_loss_improvement")
    joint_seed_passes = sum(
        importance_drift != math.inf
        and float(by_unit[(seed, "importance_anchor")]["retained_support_drift_l2"])
        <= 0.8 * float(by_unit[(seed, "unanchored")]["retained_support_drift_l2"])
        and float(by_unit[(seed, "importance_anchor")]["adaptation_loss_improvement"]) >= 0.005
        for seed in FIT_SEEDS
        if (seed, "importance_anchor") in by_unit and (seed, "unanchored") in by_unit
    )
    return {
        "planned_arm_seed_rows": expected,
        "completed_arm_seed_rows": len(complete),
        "fair_main_arm_protocol": fair,
        "overlap_and_disjoint_support_present": len(complete) == expected
        and all(
            int(row.get("overlap_support_count", 0)) > 0
            and int(row.get("disjoint_support_count", 0)) > 0
            for row in complete
        ),
        "zero_rate_unchanged": all(
            row.get("state_unchanged") is True
            for row in complete
            if row.get("arm") == "zero_learning_rate"
        ),
        "unanchored_mean_retained_drift": unanchored_drift,
        "importance_mean_retained_drift": importance_drift,
        "uniform_mean_retained_drift": uniform_drift,
        "importance_to_unanchored_drift_ratio": (
            importance_drift / unanchored_drift if unanchored_drift > 0.0 else math.inf
        ),
        "importance_mean_adaptation_improvement": importance_improvement,
        "excessive_mean_adaptation_improvement": excessive_improvement,
        "retention_effect_passed": importance_drift <= 0.8 * unanchored_drift,
        "uniform_comparator_passed": importance_drift <= uniform_drift,
        "adaptation_passed": importance_improvement >= 0.005,
        "excessive_anchor_blocks_more": excessive_improvement < importance_improvement,
        "joint_seed_passes": joint_seed_passes,
        "all_seed_joint_gate_passed": joint_seed_passes == len(FIT_SEEDS),
    }


def run_analytic_controls(
    checkpoint_dir: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Measure gradient, lifecycle, dense work, restart, and arm outcomes."""

    started = time.monotonic()
    gradient_prepared, gradient_replay = _prepared_head(FIT_SEEDS[0])
    gradient_head = _clone_for_arm(gradient_prepared, gradient_replay, "importance_anchor")
    active = np.flatnonzero(gradient_head.importance > 0.0)
    gradient_head.coefficients.reshape(-1)[active[0]] += 0.17
    features = training_fixture()[3]
    _loss, analytic, work = gradient_head.loss_and_gradient(features, 0, frozen_probability=0.43)
    numeric = finite_difference_gradient(gradient_head, features, 0, frozen_probability=0.43)
    gap = float(np.max(np.abs(analytic - numeric)))

    dense_prepared, _dense_replay = _prepared_head(FIT_SEEDS[0])
    # The dense-work control omits replay so only the complete anchor can move
    # the previously changed coefficient outside the current active support.
    dense_head = _clone_for_arm(dense_prepared, [], "importance_anchor")
    low = training_fixture().min(axis=0)
    high = training_fixture().max(axis=0)
    _low_design, low_support = local_design(low, dense_head.knots)
    _high_design, high_support = local_design(high, dense_head.knots)
    inactive = next(
        index
        for index in low_support
        if index not in high_support and dense_head.importance[index] > 0.0
    )
    reference = float(dense_head.reference_coefficients[inactive])
    dense_head.coefficients.reshape(-1)[inactive] = reference + 0.25
    before = abs(float(dense_head.coefficients.reshape(-1)[inactive]) - reference)
    dense_head.seal_prediction(
        event_id="dense-high",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=high,
        frozen_probability=0.5,
    )
    dense_receipt = dense_head.apply_feedback("dense-high", label=0, visible_at=0)
    after = abs(float(dense_head.coefficients.reshape(-1)[inactive]) - reference)

    chronology = ImportanceAnchorHead.from_training(
        training_fixture(),
        seed=FIT_SEEDS[1],
        anchor_mode="importance",
        anchor_lambda=6.0,
        learning_rate=SELECTED_LEARNING_RATE,
    )
    chronology.seal_prediction(
        event_id="delayed",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=8,
        features=low,
        frozen_probability=0.5,
    )
    chronology.seal_prediction(
        event_id="later",
        source_version="analytic-v1",
        prediction_time=1,
        reveal_time=1,
        features=high,
        frozen_probability=0.5,
    )
    early = chronology.apply_feedback("delayed", label=1, visible_at=7)
    reordered = chronology.apply_feedback("later", label=0, visible_at=1)
    admitted = chronology.apply_feedback("delayed", label=1, visible_at=8)
    duplicate = chronology.apply_feedback("delayed", label=1, visible_at=9)
    final = chronology.apply_feedback("later", label=0, visible_at=9)

    restart, restart_replay = _prepared_head(FIT_SEEDS[2])
    restart.set_replay_rows(restart_replay)
    for index in (1, 2):
        restart.seal_prediction(
            event_id=f"restart-{index}",
            source_version="analytic-v1",
            prediction_time=index,
            reveal_time=index + 8,
            features=high,
            frozen_probability=0.5,
        )
    restart.apply_feedback("restart-1", label=0, visible_at=9)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "importance-anchor-checkpoint.json"
    checkpoint_manifest = restart.save_checkpoint(checkpoint_path)
    left = ImportanceAnchorHead.load_checkpoint(checkpoint_path)
    right = ImportanceAnchorHead.load_checkpoint(checkpoint_path)
    left_receipt = left.apply_feedback("restart-2", label=0, visible_at=10)
    right_receipt = right.apply_feedback("restart-2", label=0, visible_at=10)
    replay_check = {
        "passed": left_receipt == right_receipt and left.state_hash == right.state_hash,
        "terminal_state_hash": left.state_hash,
        "receipt_hash": canonical_hash(left_receipt),
        "checkpoint_manifest": checkpoint_manifest,
        "acknowledged_feedback_count": len(left._feedback_fingerprints),
    }

    rows = [_trial_row(seed, arm) for seed in FIT_SEEDS for arm in ARM_CONFIGS]
    if time.monotonic() - started > FIXTURE_TIMEOUT_S:  # pragma: no cover - hard timeout.
        raise TimeoutError("fixture_fitting_exceeded_600_seconds")
    checks = {
        "gradient_parity": {
            "max_abs_gap": gap,
            "tolerance": FINITE_DIFFERENCE_TOLERANCE,
            "passed": gap <= FINITE_DIFFERENCE_TOLERANCE,
        },
        "chronology": {
            "early_status": early["status"],
            "reordered_status": reordered["status"],
            "admitted_status": admitted["status"],
            "duplicate_status": duplicate["status"],
            "final_status": final["status"],
            "passed": [
                early["status"],
                reordered["status"],
                admitted["status"],
                duplicate["status"],
                final["status"],
            ]
            == ["not_revealed", "reordered", "committed", "duplicate", "committed"],
        },
        "dense_anchor_work": {
            "inactive_index": inactive,
            "inactive_in_current_support": inactive in dense_receipt["active_data_support"],
            "distance_before": before,
            "distance_after": after,
            "active_data_gradient_count": dense_receipt["active_data_gradient_count"],
            "total_anchor_coefficient_count": dense_receipt["total_anchor_coefficient_count"],
            "loss_work_total_anchor_coefficient_count": work["total_anchor_coefficient_count"],
            "passed": inactive not in dense_receipt["active_data_support"]
            and after < before
            and dense_receipt["total_anchor_coefficient_count"] == COEFFICIENT_COUNT,
        },
        "restart": deepcopy(replay_check),
    }
    return rows, checks, replay_check


def anchor_protocol() -> JsonDict:
    """Freeze method, departures, candidates, controls, and fixture thresholds."""

    return {
        "schema": "carnot.exp7482.anchor_protocol.v1",
        "primary_method": {
            "title": "KAN-CL: Per-Knot Importance Regularization for Continual Learning with Kolmogorov-Arnold Networks",
            "arxiv_id": "2605.12306v1",
            "url": "https://arxiv.org/abs/2605.12306",
            "head_anchor_equation": "lambda * sum_i S_i * (c_i - c_i_star)^2",
        },
        "departures": [
            "no_cnn",
            "no_backbone_ewc",
            "no_gradient_mask",
            "prequential_feedback_stream",
        ],
        "importance_estimator": "clip(mean_of_revealed_label_coefficient_gradients_squared,0,1)",
        "consolidation_schedule": "after_prior_fixture_feedback_before_adaptation_feedback",
        "reference": "coefficient_vector_at_consolidation",
        "learning_rate_candidates": list(LEARNING_RATE_CANDIDATES),
        "selected_learning_rate": SELECTED_LEARNING_RATE,
        "selection_data_role": "training_calibration_only",
        "online_outcome_tuning_allowed": False,
        "arm_configs": deepcopy(ARM_CONFIGS),
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "update_opportunities": UPDATE_OPPORTUNITIES,
        "coefficient_cap": 256,
        "actual_coefficient_count": COEFFICIENT_COUNT,
        "generator_weights_frozen": True,
        "qwen_features_frozen": True,
        "fixture_thresholds": {
            "retention_drift_ratio_max": 0.8,
            "adaptation_loss_improvement_min": 0.005,
            "all_seed_joint_pass_count": len(FIT_SEEDS),
            "importance_drift_no_more_than_uniform": True,
            "excessive_anchor_blocks_more": True,
        },
    }


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for each frozen affected and terminal check."""

    return all(
        len([row for row in receipts if row.get("name") == name and row.get("passed") is True]) == 1
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and circular fixture benefit from raw evidence."""

    trial = reduce_trial_rows(value.get("rows") or [])
    checks = dict(value.get("analytic_checks") or {})
    implementation_passed = all(
        isinstance(checks.get(name), Mapping) and checks[name].get("passed") is True
        for name in ("gradient_parity", "chronology", "dense_anchor_work", "restart")
    )
    replay = dict(value.get("state_replay_checks") or {})
    declarations_passed = (
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("execution_venue") == EXECUTION_VENUE
        and value.get("verifier_is_oracle") is True
        and value.get("flagged_adversarial") is False
    )
    preconditions_passed = all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    validation_passed = _required_validation_passed(value.get("validation_receipts") or [])
    ready = int(
        implementation_passed
        and replay.get("passed") is True
        and declarations_passed
        and preconditions_passed
        and validation_passed
    )
    fixture_benefit = (
        trial["completed_arm_seed_rows"] == trial["planned_arm_seed_rows"]
        and trial["fair_main_arm_protocol"]
        and trial["overlap_and_disjoint_support_present"]
        and trial["zero_rate_unchanged"]
        and trial["retention_effect_passed"]
        and trial["uniform_comparator_passed"]
        and trial["adaptation_passed"]
        and trial["excessive_anchor_blocks_more"]
        and trial["all_seed_joint_gate_passed"]
    )
    validity = bool(ready)
    verdict = "circular_positive" if validity and fixture_benefit else "null"
    if not validity:
        verdict = "disqualified"
    honest = {
        "circular_positive": "complete_circular_positive_importance_anchor_fixture_benefit",
        "null": "complete_null_importance_anchor_fixture_benefit_not_met",
        "disqualified": "complete_disqualified_importance_anchor_evidence",
    }[verdict]
    return {
        "importance_anchor_ready_score": ready,
        "fixture_benefit_passed": fixture_benefit,
        "verdict_class": verdict,
        "honest_verdict": honest,
        "status": honest,
        "implementation_passed": implementation_passed,
        "preconditions_passed": preconditions_passed,
        "validation_passed": validation_passed,
        "declarations_passed": declarations_passed,
        "trial_reduction": trial,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Attach one failure-prevention reason to each plain gate value."""

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
    """Keep validity, readiness, and circular benefit as separate decisions."""

    trial = reduction["trial_reduction"]
    validity = bool(
        reduction["preconditions_passed"]
        and reduction["validation_passed"]
        and reduction["declarations_passed"]
        and reduction["implementation_passed"]
    )
    return [
        _gate(
            "required_validity",
            "validity",
            True,
            validity,
            "==",
            validity,
            "A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "importance_anchor_ready_score",
            "readiness",
            1,
            reduction["importance_anchor_ready_score"],
            "==",
            reduction["importance_anchor_ready_score"] == 1,
            "A valid null must not suppress an independent implementation measurement.",
        ),
        _gate(
            "scientific_fixture_benefit",
            "scientific_benefit",
            {
                "support": len(FIT_SEEDS),
                "drift_ratio_max": 0.8,
                "adaptation_improvement_min": 0.005,
                "joint_seed_passes": len(FIT_SEEDS),
            },
            {
                "support": trial["completed_arm_seed_rows"] // len(ARM_CONFIGS),
                "drift_ratio": trial["importance_to_unanchored_drift_ratio"],
                "adaptation_improvement": trial["importance_mean_adaptation_improvement"],
                "joint_seed_passes": trial["joint_seed_passes"],
            },
            "joint_thresholds",
            bool(reduction["fixture_benefit_passed"]),
            "A small sample, favorable seed, or analytic fixture cannot substitute for held-out value.",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed check with its exact expected and observed value."""

    failures = [
        {
            "check": row.get("check"),
            "upstream": "current_exp7482_reduction",
            "field_path": f"acceptance_gate_results.{row.get('check')}",
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {"passed": not failures, "failed_checks": failures}


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind protocol, raw rows, sources, declarations, and validation scope."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = None
    return canonical_hash(payload)


def _field_principles() -> dict[str, str]:
    """Echo the required reason for every terminal field."""

    principles = {
        "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
        "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
        "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
        "MODEL_SPECS": "Use an empty list for numeric work and emit the lowercase alias too.",
        "model_specs": "The lowercase alias keeps older readers aligned with the empty model declaration.",
        "model_invoked": "Any attempted current model call differs from archived or scripted events.",
        "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
        "inference_substrate": "Name the actual numeric learning substrate instead of implying model inference.",
        "inference_substrate_class": "The no-model class keeps numeric work separate from generation.",
        "execution_venue": "Use host and record actual CPU or CUDA identity; historical board evidence is separate.",
        "duration_s": "Measure current work without padding and separate numeric work from validation.",
        "phase_spans": "Timestamped flushed progress and checkpoints expose silent or unfinished operations.",
        "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain any null seed.",
        "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw rows and validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes and their original flags or classes.",
        "rows": "Keep one row per seed and arm, including failures and censoring.",
        "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted units.",
        "acceptance_gate_results": "Typed checks distinguish validity, readiness and fixture benefit.",
        "gate_check_summary": "Every blocked verdict names the failed check, path, expectation and observation.",
        "honest_verdict": "A terminal finding cannot hide a null, block or disqualification.",
        "verdict_class": "The closed class makes circular fixture evidence explicit.",
        "verifier_is_oracle": "True forbids positive status because the analytic fixture defines the answer.",
        "flagged_adversarial": "Real reader flags remain truthful and cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits and log hashes establish the validation scope.",
        "field_principles": "Field and gate reasons make the evidence understandable independently.",
        "importance_anchor_ready_score": "A bare score covers gradients, chronology and restart, never efficacy.",
        "anchor_definition": "The loss, estimator and consolidation schedule make the component falsifiable.",
        "active_and_total_work": "A dense anchor must not masquerade as sparse hardware work.",
        "state_replay_checks": "Delayed updates and acknowledged state must replay exactly.",
    }
    return {
        field: principles.get(field, "This field preserves one required part of the audit record.")
        for field in REQUIRED_FIELDS
    }


def finalize_artifact(artifact: JsonDict) -> JsonDict:
    """Derive all terminal scores, gates, verdicts, and the final checksum."""

    reduction = independent_reduce(artifact)
    artifact.update(
        {
            key: deepcopy(reduction[key])
            for key in (
                "importance_anchor_ready_score",
                "fixture_benefit_passed",
                "verdict_class",
                "honest_verdict",
                "status",
                "trial_reduction",
            )
        }
    )
    gates = _acceptance_gates(reduction)
    artifact["acceptance_gate_results"] = gates
    artifact["gate_check_summary"] = _gate_summary(gates)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    analytic_checks: Mapping[str, Any],
    state_replay_checks: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    device_identity: Mapping[str, Any],
) -> JsonDict:
    """Assemble one reducible candidate without trusting fixture benefit."""

    complete = sum(row.get("complete") is True for row in rows)
    failed = sum(row.get("failed") is True for row in rows)
    censored = sum(row.get("censored") is True for row in rows)
    excluded = sum(row.get("excluded") is True for row in rows)
    work_rows = [dict(row) for row in rows if row.get("complete") is True]
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
            "owner_pid": os.getpid(),
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
        "rows": [deepcopy(dict(row)) for row in rows],
        "sample_size_budget": {
            "planned": len(FIT_SEEDS) * len(ARM_CONFIGS),
            "attempted": len(rows),
            "complete": complete,
            "failed": failed,
            "censored": censored,
            "excluded": excluded,
            "unstarted": len(FIT_SEEDS) * len(ARM_CONFIGS) - len(rows),
            "independent_unit": "analytic_fixture_seed_arm",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "honest_verdict": "preterminal",
        "verdict_class": "disqualified",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "importance_anchor_ready_score": 0,
        "anchor_definition": anchor_protocol(),
        "active_and_total_work": {
            "data_gradient_is_sparse": True,
            "anchor_materialization": "complete_dense_coefficient_vector",
            "coefficient_count": COEFFICIENT_COUNT,
            "max_active_data_gradient_count": max(
                (int(row["max_active_data_gradient_count"]) for row in work_rows), default=0
            ),
            "total_anchor_coefficient_count": COEFFICIENT_COUNT,
            "active_basis_only_cost_claimed": False,
            "lazy_materialization_used": False,
        },
        "state_replay_checks": deepcopy(dict(state_replay_checks)),
        "analytic_checks": deepcopy(dict(analytic_checks)),
        "small_ebm_training": {
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "bounded_four_feature_fixed_cubic_spline_residual_with_diagonal_anchor",
            "parameter_count": PARAMETER_COUNT,
            "coefficient_count": COEFFICIENT_COUNT,
            "learning_rate_candidates": list(LEARNING_RATE_CANDIDATES),
            "selected_learning_rate": SELECTED_LEARNING_RATE,
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
        "historical_comparators": {
            "unanchored_residual": "results/experiment_7468_v654_residual_learner.json",
            "four_expert_mixture": {
                "path": "results/experiment_7454_v653_continuous_learning.json",
                "unchanged": True,
                "mixture_construction_retired": True,
            },
        },
        "qwen_features_frozen": True,
        "generator_weights_frozen": True,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "numbered_e2e_applicable": [],
        "capability_e2e": {
            "entrypoint": WRAPPER_PATH.as_posix(),
            "cold_replay": "fresh_process",
            "numbered_runtime_e2e": "not_applicable_experiment_local_numeric_head",
        },
    }
    return finalize_artifact(artifact)


def build_fixture_artifact(root: Path) -> JsonDict:
    """Build complete private evidence for mutation and cold-reader tests."""

    root.mkdir(parents=True, exist_ok=True)
    rows, checks, replay = run_analytic_controls(root / "state")
    preconditions, hashes, sidecars = collect_preconditions(REPO_ROOT)
    receipts = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": f"fixture:{name}",
            "log_sha256": canonical_hash(name),
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    current = build_current_work_receipt(
        run_id="exp7482-fixture",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"device": "cpu_float64", "model_loaded": False},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=1,
        sidecar_references=sidecars,
        small_ebm_training={"performed": True, "receipt_class": "small_ebm_training"},
    )
    return _build_artifact(
        preconditions=preconditions,
        source_hashes=hashes,
        rows=rows,
        analytic_checks=checks,
        state_replay_checks=replay,
        validation_receipts=receipts,
        current_receipt=current,
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:01+00:00",
        device_identity={"venue": "host", "numeric_device": "cpu_float64"},
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed external bytes fail-closed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, reductions, provenance, hashes, and verdict."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    errors.extend(validate_current_work_receipt(value, root=root))
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_mismatch")
    expected = deepcopy(dict(value))
    finalize_artifact(expected)
    for field in (
        "importance_anchor_ready_score",
        "fixture_benefit_passed",
        "trial_reduction",
        "acceptance_gate_results",
        "gate_check_summary",
        "verdict_class",
        "honest_verdict",
        "status",
    ):
        if value.get(field) != expected.get(field):
            errors.append(f"{field}_mismatch")
    if value.get("verifier_is_oracle") is not True or value.get("verdict_class") == "positive":
        errors.append("verifier_oracle_mismatch")
    if value.get("anchor_definition") != anchor_protocol():
        errors.append("anchor_protocol_mismatch")
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
    """Validate one serialized terminal candidate in a fresh process."""

    value = _load_object(path)
    return (
        validate_artifact(value, root=root, verify_sources=verify_sources)
        if value
        else ["artifact_unreadable_or_not_object"]
    )


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any, **details: Any
) -> JsonDict:
    """Record one exact prerequisite without collapsing false and missing."""

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
) -> tuple[list[JsonDict], dict[str, JsonDict], list[JsonDict]]:
    """Authenticate sources, ownership, historical flags, and exclusions."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    sidecars: list[JsonDict] = []
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
            "REQ-KAN-7482",
            "REQ-KAN-7482" if "REQ-KAN-7482" in spec_text else None,
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
        sidecars.append(
            {
                "path": relative.as_posix(),
                "sha256": hashes.get(relative.as_posix(), {}).get("sha256"),
                "scope": "historical_model_receipts",
            }
        )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            bool(re.search(r"experiment_id:\s*7482\b", exclusion_text)),
        )
    )
    return checks, hashes, sidecars


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one measured UTC boundary for the current run."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and potentially slow operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7482] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _device_identity() -> JsonDict:  # pragma: no cover - host identity varies.
    """Name actual host arithmetic without implying board execution."""

    boot = Path("/proc/sys/kernel/random/boot_id")
    return {
        "venue": "host",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "numpy_version": np.__version__,
        "boot_id": boot.read_text(encoding="utf-8").strip() if boot.is_file() else None,
        "numeric_device": "cpu_float64",
        "cuda_used": False,
        "historical_board_evidence_used_as_current": False,
    }


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - real clock boundary.
    """Close one monotonic phase with its completed-unit checkpoint."""

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
    """Build fresh replay, independent reduction, and strict reader commands."""

    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "capability_end_to_end",
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
            "per_seed_arm_rows",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
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
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by declared capability E2E.
    """Authenticate, fit, validate, replay, and publish one terminal result."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = utc_now()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, sidecars = collect_preconditions(root)
    failures = [row for row in preconditions if row.get("passed") is not True]
    spans.append(_span("preconditions", phase_started, started, len(preconditions), "inputs"))
    progress(started, "preconditions", "complete", failed=len(failures))
    if failures:
        raise RuntimeError(f"precondition_failed:{failures[0]}")

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0, f"no_{phase}"))
        progress(started, phase, "after", completed=0)

    raw_dir = root / RAW_DIR
    progress(started, "numeric_fitting", "before_benchmark", planned=len(FIT_SEEDS))
    phase_started = time.monotonic()
    rows, checks, replay = run_analytic_controls(raw_dir / "numeric_state")
    spans.append(_span("numeric_fitting", phase_started, started, len(rows), "fixture_rows"))
    progress(started, "numeric_fitting", "after_benchmark", completed=len(rows))
    checkpoint = Path(str(replay["checkpoint_manifest"]["path"]))
    source_hashes[checkpoint.relative_to(root).as_posix()] = {
        "path": checkpoint.relative_to(root).as_posix(),
        "sha256": replay["checkpoint_manifest"]["sha256"],
        "original_flagged_adversarial": None,
        "original_verdict_class": None,
    }

    private_root = Path(tempfile.mkdtemp(prefix="exp7482-validation-", dir="/tmp"))
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
            "work": "analytic diagonal importance anchor fitting",
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        sidecar_references=sidecars,
        phase_spans=spans,
        small_ebm_training={
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "local_residual_diagonal_importance_anchor",
            "updates_attempted": len(rows) * UPDATE_OPPORTUNITIES,
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
    )
    provisional = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]
    candidate = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=rows,
        analytic_checks=checks,
        state_replay_checks=replay,
        validation_receipts=[*affected, *provisional],
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
    progress(started, "terminal_validation", "before_subprocesses", planned=4)
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
        sidecar_references=sidecars,
        phase_spans=spans,
        small_ebm_training=receipt["small_ebm_training"],
    )
    final = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=rows,
        analytic_checks=checks,
        state_replay_checks=replay,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process terminal reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    verify_sources = not args.no_source_check
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=root, verify_sources=verify_sources)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=root, verify_sources=verify_sources)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)  # pragma: no cover - E2E boundary.
    return 0  # pragma: no cover - E2E boundary.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
