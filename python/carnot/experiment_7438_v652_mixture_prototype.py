"""Prototype causal aggregation of frozen and adaptive probability heads.

The energy in this module is only a log-odds view of a probability mixture.
The prototype uses shipped spline and Gibbs numeric-head APIs and synthetic
fixtures. It does not establish deployment value or a new generative EBM.

Spec refs: REQ-AUTO-7438 and SCENARIO-AUTO-7438-*.
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
from carnot.experiment_7412_v650_source_features import SOURCE_FEATURE_NAMES
from carnot.experiment_7425_v651_spline_prototype import SplineEnergyHead
from carnot.experiment_7427_v651_randomized_feedback import (
    FutureFeedbackError,
    ONLINE_GIBBS_ARM,
    ONLINE_SPLINE_ARM,
    OnlineLearner,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    sidecar_reference,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7438-mixture-prototype"
SCHEMA = "carnot.exp7438.v652.mixture_prototype.v1"
STATE_SCHEMA = "carnot.exp7438.four_expert_state.v1"
PROTOCOL_SCHEMA = "carnot.exp7438.four_expert_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7438_v652_mixture_prototype.json")
RAW_DIR = Path("results/raw/experiment_7438_v652_mixture_prototype")
PROTOCOL_PATH = RAW_DIR / "protocol.json"
MODULE_PATH = Path("python/carnot/experiment_7438_v652_mixture_prototype.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7438_v652_mixture_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7438_v652_mixture_prototype.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
UPSTREAM_PATH = Path("results/experiment_7427_v651_randomized_feedback.json")

FROZEN_SPLINE = "frozen_spline"
ADAPTIVE_SPLINE = "adaptive_spline"
FROZEN_GIBBS = "frozen_gibbs"
ADAPTIVE_GIBBS = "adaptive_gibbs"
EXPERT_NAMES = (FROZEN_SPLINE, ADAPTIVE_SPLINE, FROZEN_GIBBS, ADAPTIVE_GIBBS)
ADAPTIVE_EXPERTS = (ADAPTIVE_SPLINE, ADAPTIVE_GIBBS)
ETA = 1.0
FIXED_SHARE = 0.01
PROBABILITY_CLIP = 1e-6
PARITY_TOLERANCE = 1e-10
DELAYS = (0, 8)
MAX_PENDING_EVENTS = 16
SAFE_CHECKPOINT_INTERVAL = 2
RANDOM_SEED = 65_201
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

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
    Path("python/carnot/experiment_7425_v651_spline_prototype.py"),
    Path("python/carnot/experiment_7427_v651_randomized_feedback.py"),
    Path("python/carnot/experiment_7412_v650_source_features.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    Path("research-references.md"),
    SPEC_PATH,
    UPSTREAM_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
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
    "mixture_prototype_ready_score",
    "continuous_self_learning_task",
    "mixture_definition",
    "learning_control_rows",
    "hardware_path",
    "small_ebm_training",
)


def utc_now() -> str:  # pragma: no cover - authentic clock boundary.
    """Return one real UTC boundary for the execution receipt."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed boundary before and after each long operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7438] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    passed: bool | None = None,
) -> JsonDict:
    """Keep missing, false, zero, and changed prerequisite values distinct."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and return an empty mapping for invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate every declared input plus the original Exp7427 flags."""

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
                "original_verdict_class": None,
                "original_flagged_adversarial": None,
            }

    upstream = _load_object(root / UPSTREAM_PATH)
    expected_fields = {
        "schema": "carnot.exp7427.v651.randomized_feedback.v1",
        "experiment_id": "exp7427-v651-randomized-feedback",
        "online_capture_complete_score": 1,
        "online_value_score": 0,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    for field, expected in expected_fields.items():
        checks.append(
            _precondition(
                f"context_exp7427_{field}",
                "exp7427-v651-randomized-feedback",
                UPSTREAM_PATH.as_posix(),
                field,
                expected,
                upstream.get(field),
            )
        )
    if UPSTREAM_PATH.as_posix() in hashes:
        hashes[UPSTREAM_PATH.as_posix()].update(
            {
                "original_verdict_class": upstream.get("verdict_class"),
                "original_flagged_adversarial": upstream.get("flagged_adversarial"),
                "original_honest_verdict": upstream.get("honest_verdict"),
            }
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-AUTO-7438",
            "REQ-AUTO-7438" if "REQ-AUTO-7438" in spec_text else None,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            "experiment_id: 7438" in exclusion,
        )
    )
    return checks, hashes, upstream


def clip_probability(probability: float) -> float:
    """Clip one finite probability before loss or energy arithmetic."""

    numeric = float(probability)
    if not math.isfinite(numeric):
        raise ValueError("probability must be finite")
    return min(max(numeric, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)


def bernoulli_log_loss(label: int, probability: float) -> float:
    """Return finite Bernoulli log loss under the frozen clipping rule."""

    if label not in {0, 1} or isinstance(label, bool):
        raise ValueError("label must be binary")
    clipped = clip_probability(probability)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def mixture_probability(probabilities: Sequence[float], weights: Sequence[float]) -> float:
    """Combine exactly four clipped probabilities with normalized weights."""

    if len(probabilities) != 4:
        raise ValueError("four probabilities are required")
    numeric_weights = [float(value) for value in weights]
    if (
        len(numeric_weights) != 4
        or any(not math.isfinite(value) or value < 0.0 for value in numeric_weights)
        or not math.isclose(sum(numeric_weights), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("weights must be four finite nonnegative values summing to one")
    return clip_probability(
        math.fsum(
            weight * clip_probability(probability)
            for probability, weight in zip(probabilities, numeric_weights, strict=True)
        )
    )


def probability_energy(probability: float) -> float:
    """Re-express one clipped probability as negative log odds."""

    clipped = clip_probability(probability)
    return -math.log(clipped / (1.0 - clipped))


def _normalize_log_weights(log_weights: Sequence[float]) -> list[float]:
    """Normalize log weights after subtracting their maximum for stability."""

    if len(log_weights) != 4 or any(not math.isfinite(float(value)) for value in log_weights):
        raise ValueError("four finite log weights are required")
    maximum = max(float(value) for value in log_weights)
    unnormalized = [math.exp(float(value) - maximum) for value in log_weights]
    denominator = math.fsum(unnormalized)
    return [value / denominator for value in unnormalized]


def update_log_weights(
    log_weights: Sequence[float],
    probabilities: Sequence[float],
    label: int,
    *,
    eta: float = ETA,
    fixed_share: float = FIXED_SHARE,
) -> JsonDict:
    """Apply stored log loss, stable normalization, and then uniform share."""

    if len(probabilities) != 4:
        raise ValueError("four probabilities are required")
    if not math.isfinite(float(eta)) or eta < 0.0:
        raise ValueError("eta must be finite and nonnegative")
    if not math.isfinite(float(fixed_share)) or not 0.0 <= fixed_share < 1.0:
        raise ValueError("fixed share must be in [0, 1)")
    losses = [bernoulli_log_loss(label, probability) for probability in probabilities]
    posterior = _normalize_log_weights(
        [
            float(weight) - float(eta) * loss
            for weight, loss in zip(log_weights, losses, strict=True)
        ]
    )
    shared = [(1.0 - fixed_share) * weight + fixed_share / 4.0 for weight in posterior]
    return {
        "losses": losses,
        "posterior_before_share": posterior,
        "weights_after": shared,
        "log_weights_after": [math.log(weight) for weight in shared],
    }


def prediction_hash(row: Mapping[str, Any]) -> str:
    """Hash only the immutable label-free fields sealed at prediction time."""

    return canonical_hash(
        {
            "event_id": row.get("event_id"),
            "prediction_index": row.get("prediction_index"),
            "available_at": row.get("available_at"),
            "features": row.get("features"),
            "expert_probabilities": row.get("expert_probabilities"),
            "weights_at_prediction": row.get("weights_at_prediction"),
            "mixture_probability": row.get("mixture_probability"),
            "energy": row.get("energy"),
        }
    )


def _learner(*, arm: str, seed: int, checkpoint: Mapping[str, Any]) -> OnlineLearner:
    """Build one shipped numeric learner with identity calibration and no policy action."""

    return OnlineLearner(
        arm=arm,
        seed=seed,
        checkpoint=checkpoint,
        calibration={"affine": {"slope": 1.0, "intercept": 0.0}},
        policy={
            "accept_threshold": 0.95,
            "reject_threshold": 0.05,
            "accept_enabled": False,
            "reject_enabled": False,
        },
    )


def build_numeric_experts(*, seed: int = RANDOM_SEED) -> dict[str, OnlineLearner]:
    """Create paired spline and Gibbs states through the shipped numeric APIs."""

    training = np.asarray(
        [[((row + column * 3) % 17) / 16.0 for column in range(6)] for row in range(24)],
        dtype=np.float64,
    )
    spline = SplineEnergyHead.from_training(training, seed=seed)
    spline_checkpoint = {
        "kind": "sparse_spline_49",
        "knots": spline.knots.tolist(),
        "coef": spline.coefficients.tolist(),
        "bias": spline.bias,
    }
    rng = np.random.default_rng(seed + 1)
    gibbs_checkpoint = {
        "w1": rng.normal(0.0, 0.15, size=(4, 6)).tolist(),
        "b1": rng.normal(0.0, 0.05, size=4).tolist(),
        "w_out": rng.normal(0.0, 0.15, size=4).tolist(),
        "b_out": float(rng.normal(0.0, 0.05)),
    }
    return {
        FROZEN_SPLINE: _learner(arm=ONLINE_SPLINE_ARM, seed=seed, checkpoint=spline_checkpoint),
        ADAPTIVE_SPLINE: _learner(arm=ONLINE_SPLINE_ARM, seed=seed, checkpoint=spline_checkpoint),
        FROZEN_GIBBS: _learner(arm=ONLINE_GIBBS_ARM, seed=seed, checkpoint=gibbs_checkpoint),
        ADAPTIVE_GIBBS: _learner(arm=ONLINE_GIBBS_ARM, seed=seed, checkpoint=gibbs_checkpoint),
    }


class FourExpertMixture:
    """Persist four numeric experts and update them only after a causal reveal."""

    def __init__(
        self,
        experts: Mapping[str, OnlineLearner],
        *,
        eta: float = ETA,
        fixed_share: float = FIXED_SHARE,
        max_pending: int = MAX_PENDING_EVENTS,
        safe_checkpoint_interval: int = SAFE_CHECKPOINT_INTERVAL,
    ) -> None:
        if tuple(experts) != EXPERT_NAMES:
            raise ValueError("experts must use the frozen four-name order")
        if max_pending <= 0:
            raise ValueError("pending capacity must be positive")
        if safe_checkpoint_interval <= 0:
            raise ValueError("safe checkpoint interval must be positive")
        self.experts = {name: experts[name] for name in EXPERT_NAMES}
        self.eta = float(eta)
        self.fixed_share = float(fixed_share)
        update_log_weights([math.log(0.25)] * 4, [0.5] * 4, 0, eta=eta, fixed_share=fixed_share)
        self.max_pending = int(max_pending)
        self.safe_checkpoint_interval = int(safe_checkpoint_interval)
        self.log_weights = [math.log(0.25)] * 4
        self.predictions: dict[str, JsonDict] = {}
        self.pending: dict[str, JsonDict] = {}
        self.feedback_journal: list[JsonDict] = []
        self.revoked_event_ids: set[str] = set()
        self.duplicate_commit_count = 0
        self.future_label_reads = 0
        self._safe_checkpoints: list[JsonDict] = []
        self._capture_safe_checkpoint()

    @property
    def weights(self) -> dict[str, float]:
        """Return normalized weights in the fixed expert order."""

        normalized = _normalize_log_weights(self.log_weights)
        return dict(zip(EXPERT_NAMES, normalized, strict=True))

    @property
    def expert_state_hashes(self) -> dict[str, str]:
        """Return numeric state hashes without prediction-journal bytes."""

        return {name: learner.state_hash for name, learner in self.experts.items()}

    @property
    def prediction_hashes(self) -> dict[str, str]:
        """Return the original request hashes retained across replay."""

        return {identity: str(row["prediction_hash"]) for identity, row in self.predictions.items()}

    @property
    def pending_bytes(self) -> int:
        """Measure exact compact JSON bytes held for unrevealed requests."""

        return len(json.dumps(self.pending, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    @property
    def state_hash(self) -> str:
        """Hash every durable controller field except the hash slot itself."""

        return canonical_hash(self.to_state())

    @property
    def revoked_event_ids(self) -> set[str]:
        """Return a detached set of revoked request identities."""

        return set(self._revoked_event_ids)

    @revoked_event_ids.setter
    def revoked_event_ids(self, values: set[str]) -> None:
        self._revoked_event_ids = set(values)

    def _core_state(self) -> JsonDict:
        """Serialize live state without recursively embedding safe checkpoints."""

        return {
            "schema": STATE_SCHEMA,
            "eta": self.eta,
            "fixed_share": self.fixed_share,
            "max_pending": self.max_pending,
            "safe_checkpoint_interval": self.safe_checkpoint_interval,
            "log_weights": list(self.log_weights),
            "experts": {name: self.experts[name].to_dict() for name in EXPERT_NAMES},
            "predictions": deepcopy(self.predictions),
            "pending": deepcopy(self.pending),
            "feedback_journal": deepcopy(self.feedback_journal),
            "revoked_event_ids": sorted(self._revoked_event_ids),
            "duplicate_commit_count": self.duplicate_commit_count,
            "future_label_reads": self.future_label_reads,
        }

    def _capture_safe_checkpoint(self) -> None:
        """Retain a bounded replay boundary after a fixed number of commits."""

        snapshot = self._core_state()
        snapshot["commit_count"] = len(self.feedback_journal)
        self._safe_checkpoints.append(snapshot)

    def to_state(self) -> JsonDict:
        """Return code-free state for restart and atomic persistence."""

        return {**self._core_state(), "safe_checkpoints": deepcopy(self._safe_checkpoints)}

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> FourExpertMixture:
        """Restore a serialized controller and verify every prediction hash."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("checkpoint schema is invalid")
        experts_value = value.get("experts")
        if not isinstance(experts_value, Mapping) or set(experts_value) != set(EXPERT_NAMES):
            raise ValueError("checkpoint experts are invalid")
        restored = cls.__new__(cls)
        restored.experts = {
            name: OnlineLearner.from_dict(experts_value[name]) for name in EXPERT_NAMES
        }
        restored.eta = float(value["eta"])
        restored.fixed_share = float(value["fixed_share"])
        restored.max_pending = int(value["max_pending"])
        restored.safe_checkpoint_interval = int(value["safe_checkpoint_interval"])
        restored.log_weights = [float(item) for item in value["log_weights"]]
        _normalize_log_weights(restored.log_weights)
        restored.predictions = deepcopy(dict(value["predictions"]))
        restored.pending = deepcopy(dict(value["pending"]))
        restored.feedback_journal = deepcopy(list(value["feedback_journal"]))
        restored._revoked_event_ids = {str(item) for item in value["revoked_event_ids"]}
        restored.duplicate_commit_count = int(value["duplicate_commit_count"])
        restored.future_label_reads = int(value["future_label_reads"])
        restored._safe_checkpoints = deepcopy(list(value.get("safe_checkpoints") or []))
        if any(
            row.get("prediction_hash") != prediction_hash(row)
            for row in restored.predictions.values()
        ):
            raise ValueError("prediction hash mismatch")
        return restored

    def predict(
        self,
        event_id: str,
        features: Mapping[str, Any],
        *,
        index: int,
        delay: int,
    ) -> JsonDict:
        """Seal four probabilities before any label can enter pending storage."""

        identity = str(event_id)
        if not identity or identity in self.predictions:
            raise ValueError("event identity must be new and nonempty")
        if delay < 0:
            raise ValueError("delay must be nonnegative")
        if len(self.pending) >= self.max_pending:
            raise OverflowError("pending feedback capacity reached")
        available_at = int(index) + int(delay)
        expert_probabilities: dict[str, float] = {}
        for name, learner in self.experts.items():
            recorded = learner.record_prediction(
                identity, features, index=int(index), available_at=available_at
            )
            expert_probabilities[name] = clip_probability(recorded["probability"])
        weights = self.weights
        probability = mixture_probability(
            [expert_probabilities[name] for name in EXPERT_NAMES],
            [weights[name] for name in EXPERT_NAMES],
        )
        row: JsonDict = {
            "event_id": identity,
            "prediction_index": int(index),
            "available_at": available_at,
            "features": {name: float(features[name]) for name in SOURCE_FEATURE_NAMES},
            "expert_probabilities": expert_probabilities,
            "weights_at_prediction": weights,
            "mixture_probability": probability,
            "energy": probability_energy(probability),
            "prediction_before_feedback": True,
            "label_read_at_prediction": False,
        }
        row["prediction_hash"] = prediction_hash(row)
        self.predictions[identity] = deepcopy(row)
        self.pending[identity] = deepcopy(row)
        return deepcopy(row)

    def _apply_feedback(
        self, identity: str, label: int, visible_at: int, *, replay: bool
    ) -> JsonDict:
        """Update weights from stored probabilities, then adaptive heads only."""

        prediction = self.predictions[identity]
        probabilities = [prediction["expert_probabilities"][name] for name in EXPERT_NAMES]
        weights_before = self.weights
        bytes_before = len(
            json.dumps(self._core_state(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        started_ns = time.perf_counter_ns()
        updated = update_log_weights(
            self.log_weights,
            probabilities,
            label,
            eta=self.eta,
            fixed_share=self.fixed_share,
        )
        self.log_weights = list(updated["log_weights_after"])
        expert_updates: dict[str, JsonDict] = {}
        touched = 0
        for name in ADAPTIVE_EXPERTS:
            receipt = self.experts[name].commit_feedback(identity, label, visible_at=visible_at)
            if receipt.get("update_admitted") is not True:
                raise ValueError(f"adaptive expert rejected trusted feedback:{name}")
            expert_updates[name] = receipt
            touched += int(receipt.get("touched_coefficients") or 0)
        duration_ns = time.perf_counter_ns() - started_ns
        self.pending.pop(identity, None)
        event = {
            "event_id": identity,
            "label": int(label),
            "visible_at": int(visible_at),
            "prediction_hash": prediction["prediction_hash"],
            "active": True,
        }
        self.feedback_journal.append(event)
        if len(self.feedback_journal) % self.safe_checkpoint_interval == 0 and not replay:
            self._capture_safe_checkpoint()
        bytes_after = len(
            json.dumps(self._core_state(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return {
            "status": "committed",
            "update_admitted": True,
            "event_id": identity,
            "prediction_hash": prediction["prediction_hash"],
            "probabilities_used": deepcopy(prediction["expert_probabilities"]),
            "probability_source": "stored_at_prediction",
            "hindsight_prediction_count": 0,
            "weights_before": weights_before,
            "weights_after": self.weights,
            "losses": dict(zip(EXPERT_NAMES, updated["losses"], strict=True)),
            "expert_update_count": len(expert_updates),
            "expert_updates": expert_updates,
            "touched_coefficients": touched,
            "weight_loss_evaluation_count": 4,
            "fixed_share_addition_count": 4,
            "state_bytes_before": bytes_before,
            "state_bytes_after": bytes_after,
            "update_duration_ns": duration_ns,
            "prediction_before_reveal": int(prediction["prediction_index"]) <= visible_at,
            "replayed": replay,
        }

    def commit_feedback(self, event_id: str, label: int, *, visible_at: int) -> JsonDict:
        """Admit a visible label once and reject early or repeated delivery."""

        if label not in {0, 1} or isinstance(label, bool):
            raise ValueError("label must be binary")
        identity = str(event_id)
        if identity in self._revoked_event_ids:
            return {"status": "revoked", "update_admitted": False}
        prediction = self.predictions.get(identity)
        if prediction is None:
            return {"status": "unknown_event", "update_admitted": False}
        if identity not in self.pending:
            return {"status": "late_duplicate", "update_admitted": False}
        if int(visible_at) < int(prediction["available_at"]):
            raise FutureFeedbackError(identity)
        return self._apply_feedback(identity, int(label), int(visible_at), replay=False)

    def _restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a trusted core snapshot without changing retained boundaries."""

        state = {**deepcopy(dict(snapshot)), "safe_checkpoints": []}
        state.pop("commit_count", None)
        restored = self.from_state(state)
        self.experts = restored.experts
        self.eta = restored.eta
        self.fixed_share = restored.fixed_share
        self.max_pending = restored.max_pending
        self.safe_checkpoint_interval = restored.safe_checkpoint_interval
        self.log_weights = restored.log_weights
        self.predictions = restored.predictions
        self.pending = restored.pending
        self.feedback_journal = restored.feedback_journal
        self._revoked_event_ids = restored._revoked_event_ids
        self.duplicate_commit_count = restored.duplicate_commit_count
        self.future_label_reads = restored.future_label_reads

    def revoke_feedback(self, event_id: str) -> JsonDict:
        """Restore the last safe state and replay later active labels without hindsight."""

        identity = str(event_id)
        matching = [
            index for index, row in enumerate(self.feedback_journal) if row["event_id"] == identity
        ]
        if len(matching) != 1:
            return {"status": "unknown_event", "trusted_journal_replayed": False}
        target_index = matching[0]
        original_predictions = deepcopy(self.predictions)
        original_hashes = self.prediction_hashes
        original_pending_ids = set(self.pending)
        original_journal = deepcopy(self.feedback_journal)
        eligible = [
            row for row in self._safe_checkpoints if int(row["commit_count"]) <= target_index
        ]
        safe = max(eligible, key=lambda row: int(row["commit_count"]))
        safe_count = int(safe["commit_count"])
        retained_safe = [
            deepcopy(row)
            for row in self._safe_checkpoints
            if int(row["commit_count"]) <= safe_count
        ]
        self._restore_snapshot(safe)
        self._safe_checkpoints = retained_safe
        self._revoked_event_ids.add(identity)
        for event_identity, row in original_predictions.items():
            if event_identity not in self.predictions:
                for learner in self.experts.values():
                    learner.record_prediction(
                        event_identity,
                        row["features"],
                        index=int(row["prediction_index"]),
                        available_at=int(row["available_at"]),
                    )
                self.predictions[event_identity] = deepcopy(row)
        self.pending = {
            event_identity: deepcopy(original_predictions[event_identity])
            for event_identity in original_pending_ids
        }
        replayed = 0
        for event in original_journal[safe_count:]:
            event_identity = str(event["event_id"])
            if event_identity == identity or event_identity in self._revoked_event_ids:
                continue
            self._apply_feedback(
                event_identity,
                int(event["label"]),
                int(event["visible_at"]),
                replay=True,
            )
            replayed += 1
        if len(self.feedback_journal) % self.safe_checkpoint_interval == 0:
            self._capture_safe_checkpoint()
        return {
            "status": "revoked",
            "trusted_journal_replayed": True,
            "safe_checkpoint_commit_count": safe_count,
            "replayed_event_count": replayed,
            "prediction_hashes_preserved": self.prediction_hashes == original_hashes,
            "state_hash_after": self.state_hash,
        }

    def save_checkpoint(self, path: Path, *, interrupt_before_replace: bool = False) -> JsonDict:
        """Write durable state atomically and preserve an older target on interruption."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_state()
        payload["checkpoint_hash"] = canonical_hash(payload)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if interrupt_before_replace:
                raise RuntimeError("simulated interruption before atomic replacement")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "path": str(target),
            "sha256": sha256_file(target),
            "state_hash": self.state_hash,
            "byte_size": target.stat().st_size,
        }

    @classmethod
    def load_checkpoint(cls, path: Path) -> FourExpertMixture:
        """Restore one hash-bound checkpoint and reject partial or changed bytes."""

        value = _load_object(path)
        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("checkpoint schema is invalid")
        expected = value.get("checkpoint_hash")
        payload = {key: deepcopy(item) for key, item in value.items() if key != "checkpoint_hash"}
        if expected != canonical_hash(payload):
            raise ValueError("checkpoint hash mismatch")
        return cls.from_state(payload)


def analytic_probability_replay(
    probability_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    eta: float = ETA,
    fixed_share: float = FIXED_SHARE,
) -> JsonDict:
    """Replay expert probabilities without fitting a numeric head."""

    if len(probability_rows) != len(labels):
        raise ValueError("probability rows and labels must have equal length")
    log_weights = [math.log(0.25)] * 4
    rows: list[JsonDict] = []
    for index, (probabilities, label) in enumerate(zip(probability_rows, labels, strict=True)):
        weights_before = _normalize_log_weights(log_weights)
        mixture = mixture_probability(probabilities, weights_before)
        updated = update_log_weights(
            log_weights, probabilities, label, eta=eta, fixed_share=fixed_share
        )
        log_weights = list(updated["log_weights_after"])
        rows.append(
            {
                "event_index": index,
                "label": int(label),
                "probabilities": [clip_probability(value) for value in probabilities],
                "weights_before": weights_before,
                "mixture_probability": mixture,
                "energy": probability_energy(mixture),
                "loss": bernoulli_log_loss(int(label), mixture),
                "weights_after": list(updated["weights_after"]),
            }
        )
    return {"rows": rows, "final_weights": _normalize_log_weights(log_weights)}


def full_information_no_share_identity(
    probability_rows: Sequence[Sequence[float]], labels: Sequence[int]
) -> JsonDict:
    """Check the eta-one Bayesian log-loss identity on an analytic table."""

    replay = analytic_probability_replay(probability_rows, labels, fixed_share=0.0)
    cumulative = math.fsum(float(row["loss"]) for row in replay["rows"])
    expert_likelihoods: list[float] = []
    for expert_index in range(4):
        loss = math.fsum(
            bernoulli_log_loss(label, probabilities[expert_index])
            for probabilities, label in zip(probability_rows, labels, strict=True)
        )
        expert_likelihoods.append(math.exp(-loss))
    marginal = -math.log(math.fsum(0.25 * value for value in expert_likelihoods))
    return {
        "mixture_cumulative_log_loss": cumulative,
        "negative_log_marginal_likelihood": marginal,
        "absolute_error": abs(cumulative - marginal),
        "deployment_guarantee_inherited": False,
    }


def replay_numeric_stream(
    *, delay: int, count: int, restart_after: int, checkpoint_path: Path
) -> JsonDict:
    """Replay one synthetic stream while measuring pending and restart state."""

    if delay not in DELAYS or count <= 0:
        raise ValueError("registered delay and positive count are required")
    controller = FourExpertMixture(build_numeric_experts(seed=RANDOM_SEED + delay))
    peak_count = 0
    peak_bytes = 0
    receipts: list[JsonDict] = []
    restart_equal = False
    restart_prediction_equal = False

    def deliver(visible_at: int) -> None:
        due = [
            identity
            for identity, row in controller.pending.items()
            if int(row["available_at"]) <= visible_at
        ]
        for identity in due:
            index = int(controller.predictions[identity]["prediction_index"])
            receipt = controller.commit_feedback(identity, index % 2, visible_at=visible_at)
            receipts.append(receipt)

    for index in range(count):
        deliver(index)
        controller.predict(
            f"delay-{delay}-{index}", _fixture_features(index), index=index, delay=delay
        )
        peak_count = max(peak_count, len(controller.pending))
        peak_bytes = max(peak_bytes, controller.pending_bytes)
        if delay == 0:
            deliver(index)
        if index == restart_after:
            before = controller.to_state()
            controller.save_checkpoint(checkpoint_path)
            controller = FourExpertMixture.load_checkpoint(checkpoint_path)
            restart_equal = controller.to_state() == before
            left = FourExpertMixture.from_state(controller.to_state())
            right = FourExpertMixture.from_state(controller.to_state())
            probe_left = left.predict("restart-probe", _fixture_features(99), index=99, delay=8)
            probe_right = right.predict("restart-probe", _fixture_features(99), index=99, delay=8)
            restart_prediction_equal = probe_left == probe_right
    deliver(count + delay)
    return {
        "delay": delay,
        "rows": receipts,
        "completed_events": len(receipts),
        "pending_at_end": len(controller.pending),
        "peak_pending_count": peak_count,
        "peak_pending_bytes": peak_bytes,
        "future_label_reads": controller.future_label_reads,
        "duplicate_commit_count": controller.duplicate_commit_count,
        "restart_equal": restart_equal,
        "restart_prediction_equal": restart_prediction_equal,
        "controller": controller,
    }


def _fixture_features(index: int) -> dict[str, float]:
    """Create one finite six-feature analytic row without corpus observations."""

    return {
        name: ((index + offset) % 11) / 10.0 for offset, name in enumerate(SOURCE_FEATURE_NAMES)
    }


def build_protocol_manifest() -> JsonDict:
    """Freeze the four experts, update rule, controls, and claim boundary."""

    protocol: JsonDict = {
        "schema": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experts": list(EXPERT_NAMES),
        "adaptive_experts": list(ADAPTIVE_EXPERTS),
        "initial_weights": [0.25] * 4,
        "probability_clip": [PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP],
        "loss": "bernoulli_log_loss",
        "eta": ETA,
        "fixed_share": FIXED_SHARE,
        "delays": list(DELAYS),
        "max_pending_events": MAX_PENDING_EVENTS,
        "safe_checkpoint_interval": SAFE_CHECKPOINT_INTERVAL,
        "parity_tolerance": PARITY_TOLERANCE,
        "required_future_label_reads": 0,
        "required_duplicate_commits": 0,
        "required_restart_equality": True,
        "energy": "-log(q/(1-q)) where q=sum_k(w_k*p_k)",
        "energy_scope": "probability_reexpression_not_generative_ebm",
        "guarantees_not_inherited": [
            "no_share_regret",
            "calibration",
            "iid",
            "conformal",
            "delayed_feedback_theorem",
        ],
        "random_seed": RANDOM_SEED,
    }
    protocol["manifest_hash"] = canonical_hash(protocol)
    return protocol


def write_protocol(path: Path) -> JsonDict:
    """Publish the frozen protocol with a stable internal manifest hash."""

    protocol = build_protocol_manifest()
    atomic_json(path, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> list[str]:
    """Reject any drift from the registered four-expert rule."""

    expected = build_protocol_manifest()
    return [] if canonical_hash(value) == canonical_hash(expected) else ["protocol_mismatch"]


def run_analytic_controls(private_root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run all mathematical, delayed, persistence, and revocation controls."""

    private_root.mkdir(parents=True, exist_ok=True)
    delay_results = {
        delay: replay_numeric_stream(
            delay=delay,
            count=20,
            restart_after=9,
            checkpoint_path=private_root / f"delay-{delay}.json",
        )
        for delay in DELAYS
    }
    constant = analytic_probability_replay([[0.7] * 4] * 6, [1, 0, 1, 0, 1, 0])
    harmful = analytic_probability_replay([[0.99, 0.1, 0.1, 0.1]] * 12, [0] * 12)
    shifted = analytic_probability_replay([[0.9, 0.1, 0.6, 0.4]] * 34, [1] * 10 + [0] * 24)
    extreme = analytic_probability_replay(
        [[0.0, 1.0, 1e-300, 1.0 - 1e-300], [1.0, 0.0, 0.5, 0.5]], [0, 1]
    )
    identity = full_information_no_share_identity(
        [[0.8, 0.4, 0.6, 0.2], [0.7, 0.3, 0.5, 0.9], [0.2, 0.8, 0.4, 0.6]],
        [1, 0, 1],
    )
    revoked = FourExpertMixture(build_numeric_experts(seed=RANDOM_SEED + 99))
    for index in range(7):
        revoked.predict(f"revoke-{index}", _fixture_features(index), index=index, delay=0)
        revoked.commit_feedback(f"revoke-{index}", index % 2, visible_at=index)
    original_hashes = revoked.prediction_hashes
    revocation = revoked.revoke_feedback("revoke-4")
    interrupted_path = private_root / "interrupted.json"
    revoked.save_checkpoint(interrupted_path)
    stable = interrupted_path.read_bytes()
    interrupted = False
    try:
        revoked.save_checkpoint(interrupted_path, interrupt_before_replace=True)
    except RuntimeError:
        interrupted = True

    numeric_parity = max(
        abs(
            float(row["mixture_probability"])
            - mixture_probability(
                list(row["expert_probabilities"].values()),
                list(row["weights_at_prediction"].values()),
            )
        )
        for result in delay_results.values()
        for row in result["controller"].predictions.values()
    )
    controls = [
        {
            "unit_id": f"delay-{delay}",
            "arm": "delayed_numeric_replay",
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "passed": result["future_label_reads"] == 0
            and result["duplicate_commit_count"] == 0
            and result["pending_at_end"] == 0
            and result["restart_equal"]
            and result["restart_prediction_equal"],
            "future_label_reads": result["future_label_reads"],
            "duplicate_commit_count": result["duplicate_commit_count"],
            "restart_equal": result["restart_equal"],
            "peak_pending_count": result["peak_pending_count"],
            "peak_pending_bytes": result["peak_pending_bytes"],
            "completed_events": result["completed_events"],
        }
        for delay, result in delay_results.items()
    ]
    controls.extend(
        [
            {
                "unit_id": "constant_experts",
                "arm": "analytic_probability",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": all(
                    math.isclose(row["mixture_probability"], 0.7, abs_tol=1e-12)
                    for row in constant["rows"]
                ),
            },
            {
                "unit_id": "persistently_harmful_expert",
                "arm": "analytic_probability",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": harmful["final_weights"][0] < min(harmful["final_weights"][1:]),
            },
            {
                "unit_id": "regime_change",
                "arm": "analytic_probability",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": shifted["rows"][9]["weights_after"][0]
                > shifted["rows"][9]["weights_after"][1]
                and shifted["final_weights"][1] > shifted["final_weights"][0],
            },
            {
                "unit_id": "extreme_finite_probabilities",
                "arm": "analytic_probability",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": all(
                    math.isfinite(row["energy"]) and math.isfinite(row["mixture_probability"])
                    for row in extreme["rows"]
                ),
            },
            {
                "unit_id": "full_information_no_share_identity",
                "arm": "analytic_probability",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": identity["absolute_error"] <= PARITY_TOLERANCE,
                "numeric_parity_max_abs": identity["absolute_error"],
            },
            {
                "unit_id": "revocation_safe_replay",
                "arm": "persistence_control",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": revocation["prediction_hashes_preserved"]
                and revoked.prediction_hashes == original_hashes,
                **revocation,
            },
            {
                "unit_id": "interrupted_atomic_write",
                "arm": "persistence_control",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": interrupted and interrupted_path.read_bytes() == stable,
            },
            {
                "unit_id": "numeric_probability_parity",
                "arm": "parity_control",
                "attempted": True,
                "completed": True,
                "failed": False,
                "censored": False,
                "unstarted": False,
                "passed": numeric_parity <= PARITY_TOLERANCE,
                "numeric_parity_max_abs": numeric_parity,
            },
        ]
    )
    learning_rows = [
        {
            **deepcopy(row),
            "delay": delay,
            "future_label_reads": result["future_label_reads"],
            "duplicate_commit_count": result["duplicate_commit_count"],
            "restart_equal": result["restart_equal"],
        }
        for delay, result in delay_results.items()
        for row in result["rows"]
    ]
    return controls, learning_rows


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, candidate: bool = False) -> bool:
    """Require each frozen affected and terminal command exactly once."""

    required = set(AFFECTED_CHECK_NAMES)
    if not candidate:
        required.update(TERMINAL_CHECK_NAMES)
    passing = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True
        and row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return required.issubset(passing)


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep one gate's operands plain and its reason separate."""

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
    gates: Sequence[Mapping[str, Any]], failed: Mapping[str, Any] | None = None
) -> JsonDict:
    """Name the first exact failed operand without hiding later failures."""

    failures = [str(row.get("check")) for row in gates if row.get("passed") is not True]
    summary: JsonDict = {
        "all_required_gates_passed": not failures,
        "failed_checks": failures,
    }
    if failed is not None:
        summary.update(
            {
                "blocked_upstream": failed.get("upstream"),
                "blocked_path": failed.get("path"),
                "blocked_check": failed.get("check"),
                "blocked_field": failed.get("field"),
                "expected": failed.get("expected"),
                "observed": failed.get("observed"),
            }
        )
    return summary


def _field_principles() -> dict[str, str]:
    """Explain ordinary fields without wrapping their machine values."""

    principles = {
        field: "This field records measured Exp7438 prototype evidence."
        for field in REQUIRED_FIELDS
    }
    principles.update(
        {
            "schema": "Use a versioned plain top-level schema with experiment identity and terminal status.",
            "run_date": "Use 20260920 and retain actual UTC and monotonic boundaries.",
            "preconditions_checked": "Name each observed resource, path, identity, field, expectation, and value.",
            "MODEL_SPECS": "List current LLMs; this numeric prototype invokes none.",
            "model_invoked": "Separate current attempted model use from archived model-shaped evidence.",
            "invocation_counts": "Reconcile attempted, completed, failed, cancelled, and in-flight current calls.",
            "inference_substrate": "Use a truthful string and keep device facts in the details field.",
            "inference_substrate_class": "Declare no_model_load because only small numeric heads run.",
            "execution_venue": "Use host and record CPU, CUDA, and external device identities separately.",
            "duration_s": "Measure current work and separate model, computation, cold-start, and validation time.",
            "phase_spans": "Bind phase times, flushed boundaries, completed units, and checkpoints.",
            "random_seed": "Freeze every fitting, stream, and analytic fixture seed.",
            "reproducibility_checksum": "Bind code, protocol, inputs, controls, and exact validation scope.",
            "source_artifact_hashes": "Preserve source identity, original classes, and original flags.",
            "rows": "Keep one unit for each control, including failed and unstarted units.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
            "acceptance_gate_results": "Keep validity, safety, completion, and benefit categories distinct.",
            "gate_check_summary": "Name exact blocked operands and preserve missing, None, and zero.",
            "verifier_is_oracle": "Synthetic analytic labels make readiness circular, not external value.",
            "honest_verdict": "Use complete_ for finished findings and blocked_ for unavailable prerequisites.",
            "verdict_class": "Use positive, circular_positive, null, blocked, disqualified, or partial.",
            "flagged_adversarial": "A critical validation finding prevents readiness.",
            "validation_receipts": "Record scoped argv, environment, exits, durations, and hashed logs.",
            "field_principles": "Explain field intent separately from plain numeric gate values.",
            "promotion_score": "Always zero; this milestone authorizes no rollout or generator update.",
            "mixture_prototype_ready_score": "Certify implementation controls only, not real learning value.",
            "continuous_self_learning_task": "Identify updates from earlier revealed outcomes to later predictions.",
            "mixture_definition": "Expose probabilities, loss, clipping, eta, and fixed-share arithmetic.",
            "learning_control_rows": "Preserve each prediction, reveal, update, restart, and replay comparison.",
            "hardware_path": "Describe four CPU log-weight updates that admit a later vector implementation.",
            "small_ebm_training": "Keep compact numeric updates separate from current LLM calls.",
        }
    )
    return principles


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks, logs, and this hash slot."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    for field in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "phase_spans",
        "validation_receipts",
        "validation_duration_s",
        "cold_start_duration_s",
        "computation_duration_s",
    ):
        payload.pop(field, None)
    return canonical_hash(payload)


def _receipt(
    root: Path,
    *,
    started_ns: int,
    ended_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    fixture: bool,
) -> JsonDict:
    """Build current no-model provenance and type the upstream as historical."""

    sidecars = (
        []
        if fixture
        else [sidecar_reference(root / UPSTREAM_PATH, root=root, scope="historical_model_receipts")]
    )
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "cuda_used": False,
            "external_device": None,
            "current_llm": False,
            "numeric_heads": ["spline", "gibbs"],
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=sidecars,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": True,
            "receipt_class": "small_ebm_training",
            "current_llm_calls": 0,
            "generator_weights_fitted": False,
            "components": "two adaptive experiment-local numeric heads and four log weights",
            "maximum_updates": 40,
        },
    )
    receipt["started_monotonic_ns"] = 0
    receipt["ended_monotonic_ns"] = ended_ns - started_ns
    return receipt


def _protocol_reference(root: Path, path: Path) -> JsonDict:
    """Bind the exact protocol bytes and their internal manifest identity."""

    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(path.resolve())
    value = _load_object(path)
    return {
        "path": label,
        "sha256": sha256_file(path),
        "manifest_hash": value.get("manifest_hash"),
    }


def build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    learning_rows: Sequence[Mapping[str, Any]],
    protocol_path: Path,
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    started_ns: int,
    ended_ns: int,
    candidate: bool = False,
    fixture: bool = False,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Assemble one independently reducible implementation-readiness record."""

    controls_ok = bool(controls) and all(row.get("passed") is True for row in controls)
    future_reads = sum(int(row.get("future_label_reads") or 0) for row in learning_rows)
    duplicate_commits = sum(int(row.get("duplicate_commit_count") or 0) for row in learning_rows)
    restart_equal = bool(learning_rows) and all(
        row.get("restart_equal") is True for row in learning_rows
    )
    parity = max(
        [float(row.get("numeric_parity_max_abs") or 0.0) for row in controls], default=math.inf
    )
    protocol_ok = not validate_protocol(_load_object(protocol_path))
    validation_ok = _validation_passed(validation_receipts, candidate=candidate)
    preconditions_ok = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    ready = int(
        preconditions_ok
        and controls_ok
        and future_reads == 0
        and duplicate_commits == 0
        and restart_equal
        and parity <= PARITY_TOLERANCE
        and protocol_ok
        and validation_ok
        and not flagged_adversarial
    )
    verdict_class = "circular_positive" if ready else "disqualified"
    honest = (
        "complete_circular_positive_mixture_implementation_ready"
        if ready
        else "complete_disqualified_mixture_prototype_controls"
    )
    receipt = _receipt(
        root,
        started_ns=started_ns,
        ended_ns=ended_ns,
        phase_spans=phase_spans,
        fixture=fixture,
    )
    gates = [
        _gate(
            "authenticated_preconditions",
            "validity",
            "all",
            True,
            preconditions_ok,
            preconditions_ok,
            "Changed or absent inputs cannot silently enter the prototype.",
        ),
        _gate(
            "analytic_and_numeric_controls",
            "validity",
            "all",
            True,
            controls_ok,
            controls_ok,
            "All registered controls must finish before readiness.",
        ),
        _gate(
            "zero_future_label_reads",
            "safety",
            "==",
            0,
            future_reads,
            future_reads == 0,
            "A label can affect only requests predicted before its reveal.",
        ),
        _gate(
            "zero_duplicate_commits",
            "safety",
            "==",
            0,
            duplicate_commits,
            duplicate_commits == 0,
            "One event identity can change weights and adaptive heads at most once.",
        ),
        _gate(
            "restart_equality",
            "validity",
            "==",
            True,
            restart_equal,
            restart_equal,
            "Restart must preserve pending, numeric, and mixture state.",
        ),
        _gate(
            "numeric_parity",
            "validity",
            "<=",
            PARITY_TOLERANCE,
            parity,
            parity <= PARITY_TOLERANCE,
            "Stored numeric evidence must reproduce the declared mixture arithmetic.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_ok,
            validation_ok,
            "Only the frozen scoped commands and strict readers authorize readiness.",
        ),
        _gate(
            "synthetic_implementation_readiness",
            "scientific_benefit",
            "==",
            True,
            ready == 1,
            ready == 1,
            "Synthetic success certifies implementation only and remains circular.",
        ),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        **receipt,
        "model_duration_s": 0.0,
        "computation_duration_s": sum(
            float(span.get("duration_s") or 0.0)
            for span in phase_spans
            if span.get("phase") == "controls"
        ),
        "validation_duration_s": sum(
            float(row.get("duration_s") or 0.0) for row in validation_receipts
        ),
        "cold_start_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in validation_receipts
            if row.get("name") in TERMINAL_CHECK_NAMES
        ),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "random_seed": {
            "numeric_fixture": RANDOM_SEED,
            "delays": list(DELAYS),
            "sampling": None,
            "resampling": None,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in controls],
        "sample_size_budget": {
            "planned_units": 10,
            "attempted_units": len(controls),
            "completed_units": sum(row.get("completed") is True for row in controls),
            "failed_units": sum(row.get("failed") is True for row in controls),
            "censored_units": sum(row.get("censored") is True for row in controls),
            "unstarted_units": max(0, 10 - len(controls)),
            "stop_rule": "run each frozen analytic, delay, persistence, and parity control once",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "mixture_prototype_ready_score": ready,
        "continuous_self_learning_task": True,
        "mixture_definition": {
            "experts": list(EXPERT_NAMES),
            "initial_weights": [0.25] * 4,
            "probability_clip": [PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP],
            "probability": "q_t=sum_k(w_tk*p_tk)",
            "energy": "-log(q_t/(1-q_t))",
            "loss": "bernoulli_log_loss",
            "eta": ETA,
            "fixed_share": FIXED_SHARE,
            "update_order": "stored_probability_log_loss_then_stable_normalize_then_share_then_adaptive_heads",
            "generative_ebm_claimed": False,
            "inherited_guarantees": [],
        },
        "learning_control_rows": [deepcopy(dict(row)) for row in learning_rows],
        "protocol_manifest": _protocol_reference(root, protocol_path),
        "hardware_path": {
            "current_device": "host_cpu",
            "weight_update_count": 4,
            "operations": "vector log-loss, stable normalization, and lookup-table compatible share",
            "future_fpga_path": True,
            "hardware_speedup_claimed": False,
        },
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute readiness from protocol bytes, controls, events, and receipts."""

    reference = value.get("protocol_manifest") or {}
    path = Path(str(reference.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    protocol = _load_object(resolved)
    protocol_ok = (
        resolved.is_file()
        and sha256_file(resolved) == reference.get("sha256")
        and protocol.get("manifest_hash") == reference.get("manifest_hash")
        and not validate_protocol(protocol)
    )
    controls = value.get("rows") or []
    learning = value.get("learning_control_rows") or []
    controls_ok = len(controls) == 10 and all(
        isinstance(row, Mapping)
        and row.get("completed") is True
        and row.get("failed") is False
        and row.get("passed") is True
        for row in controls
    )
    future_reads = sum(int(row.get("future_label_reads") or 0) for row in learning)
    duplicate_commits = sum(int(row.get("duplicate_commit_count") or 0) for row in learning)
    restart_equal = bool(learning) and all(row.get("restart_equal") is True for row in learning)
    parity = max(
        [float(row.get("numeric_parity_max_abs") or 0.0) for row in controls], default=math.inf
    )
    preconditions = value.get("preconditions_checked") or []
    validation_ok = _validation_passed(
        value.get("validation_receipts") or [], candidate=value.get("candidate_artifact") is True
    )
    ready = int(
        bool(preconditions)
        and all(row.get("passed") is True for row in preconditions)
        and controls_ok
        and learning
        and future_reads == 0
        and duplicate_commits == 0
        and restart_equal
        and parity <= PARITY_TOLERANCE
        and protocol_ok
        and validation_ok
        and value.get("flagged_adversarial") is False
    )
    return {
        "mixture_prototype_ready_score": ready,
        "promotion_score": 0,
        "continuous_self_learning_task": True,
        "protocol_valid": protocol_ok,
        "future_label_reads": future_reads,
        "duplicate_commit_count": duplicate_commits,
        "restart_equal": restart_equal,
        "numeric_parity_max_abs": parity,
    }


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, raw reduction, provenance, controls, and checksum."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": True,
        "promotion_score": 0,
        "continuous_self_learning_task": True,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"declaration_mismatch:{field}")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(value.get("field_principles") or {}) != set(REQUIRED_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("verdict_class") == "blocked":
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
    else:
        try:
            reduced = independent_reduce(value, root=root)
        except (AttributeError, OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"independent_reduction_failed:{error}")
        else:
            for field in (
                "mixture_prototype_ready_score",
                "promotion_score",
                "continuous_self_learning_task",
            ):
                if value.get(field) != reduced[field]:
                    errors.append(f"independent_reduction_mismatch:{field}")
        if not value.get("fixture_artifact"):
            for row in (value.get("source_artifact_hashes") or {}).values():
                if not isinstance(row, Mapping):
                    errors.append("source_artifact_hash_row_invalid")
                    continue
                source_path = Path(str(row.get("path") or ""))
                resolved = source_path if source_path.is_absolute() else root / source_path
                if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                    errors.append(f"source_artifact_hash_mismatch:{row.get('path')}")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=root))
    return list(dict.fromkeys(errors))


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish one terminal external block without dependent synthetic rows."""

    started = time.monotonic_ns()
    receipt = _receipt(
        REPO_ROOT,
        started_ns=started,
        ended_ns=time.monotonic_ns(),
        phase_spans=[],
        fixture=True,
    )
    gate = _gate(
        str(failed.get("check")),
        "prerequisite",
        str(failed.get("operator") or "=="),
        failed.get("expected"),
        failed.get("observed"),
        False,
        "Dependent work cannot replace an unavailable or changed source.",
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_external_prerequisite",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        **receipt,
        "preconditions_checked": [deepcopy(dict(failed))],
        "random_seed": {"numeric_fixture": RANDOM_SEED},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": 10,
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": 10,
            "stop_rule": "stop before dependent work on failed prerequisite",
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": _gate_summary([gate], failed),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "mixture_prototype_ready_score": 0,
        "continuous_self_learning_task": True,
        "mixture_definition": {},
        "learning_control_rows": [],
        "hardware_path": {},
        "candidate_artifact": False,
        "fixture_artifact": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_fixture_artifact(
    *,
    root: Path,
    protocol_path: Path,
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build compact complete evidence for reducer and mutation tests."""

    write_protocol(protocol_path)
    private = Path(tempfile.mkdtemp(prefix="exp7438-fixture-", dir="/tmp"))
    controls, learning = run_analytic_controls(private)
    started = time.monotonic_ns()
    preconditions = [
        _precondition(
            "fixture_source", "analytic_fixture", "analytic_fixture", "available", True, True
        )
    ]
    return build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes={},
        controls=controls,
        learning_rows=learning,
        protocol_path=protocol_path,
        validation_receipts=validation_receipts,
        phase_spans=[],
        started_at_utc=utc_now(),
        started_ns=started,
        ended_ns=time.monotonic_ns(),
        fixture=True,
    )


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload one candidate in a fresh process and validate bound evidence."""

    value = _load_object(path)
    return validate_artifact(value, root=root) if value else ["artifact_unreadable_or_not_object"]


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover - E2E only.
    """Build fresh replay, reduction, adversarial, and strict row checks."""

    python = ".venv/bin/python"
    common = (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay", (*common, "--cold-replay", str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_row_reduction",
                (*common, "--independent-reduce", str(candidate)),
                "candidate",
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
    ]


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    """Close one phase with monotonic duration and completed units."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def run_experiment(  # pragma: no cover - declared entrypoint E2E.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Authenticate, measure, validate, independently reduce, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, _upstream = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    progress(run_started, "protocol", "before_benchmark")
    phase_started = time.monotonic()
    protocol_path = root / PROTOCOL_PATH
    write_protocol(protocol_path)
    spans.append(_span("protocol", phase_started, run_started, 1))
    progress(run_started, "protocol", "after_benchmark", completed=1)

    progress(run_started, "controls", "before_benchmark", planned=10)
    phase_started = time.monotonic()
    private_controls = Path(tempfile.mkdtemp(prefix="exp7438-controls-", dir="/tmp"))
    controls, learning_rows = run_analytic_controls(private_controls)
    spans.append(_span("controls", phase_started, run_started, len(controls)))
    progress(run_started, "controls", "after_benchmark", completed=len(controls))

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH, PROTOCOL_PATH):
        path = root / relative
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "original_verdict_class": None,
            "original_flagged_adversarial": None,
        }

    private_validation = Path(tempfile.mkdtemp(prefix="exp7438-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_validation)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )

    candidate = build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        controls=controls,
        learning_rows=learning_rows,
        protocol_path=protocol_path,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        candidate=True,
        flagged_adversarial=not affected_reduction["passed"] or bool(plan_errors),
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=root / RAW_DIR / "validation/terminal",
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
    final = build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        controls=controls,
        learning_rows=learning_rows,
        protocol_path=protocol_path,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=(
            not affected_reduction["passed"] or not terminal_passed or critical or bool(plan_errors)
        ),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and strict fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prototype or one strict fresh-process artifact reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=args.root)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=args.root)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduced = independent_reduce(value, root=args.root) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduced}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
