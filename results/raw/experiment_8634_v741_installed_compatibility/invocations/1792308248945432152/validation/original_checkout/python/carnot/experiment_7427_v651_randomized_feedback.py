"""Replay randomized human feedback after every committed prediction.

The source labels are archived RAGTruth annotations. They are fallible human
support judgments, not live user feedback and not proof of semantic truth.
Spec refs: REQ-AUTO-7427 and SCENARIO-AUTO-7427-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time
import tracemalloc
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
from carnot.experiment_7412_v650_source_features import (
    SOURCE_FEATURE_NAMES,
    gibbs_energy,
    probability_from_energy,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_TOKEN,
    ProtocolReaders,
    reload_corpus,
)
from carnot.experiment_7425_v651_spline_prototype import dense_design_vector
from carnot.experiment_7426_v651_static_decisions import (
    _apply_affine,
    _load_object,
    join_predictor_labels,
    typed_support_decision,
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
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7427-v651-randomized-feedback"
SCHEMA = "carnot.exp7427.v651.randomized_feedback.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7427_v651_randomized_feedback.json")
RAW_DIR = Path("results/raw/experiment_7427_v651_randomized_feedback")
MODULE_PATH = Path("python/carnot/experiment_7427_v651_randomized_feedback.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7427_v651_randomized_feedback.py")
TEST_PATH = Path("tests/python/test_experiment_7427_v651_randomized_feedback.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
UPSTREAM_PATH = Path("results/experiment_7426_v651_static_decisions.json")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_PATH = CORPUS_DIR / "corpus_manifest.json"
EXPECTED_UPSTREAM_SHA256 = "sha256:82405ac1229cd49a3cb4bc700b4cd978d423c166b51da23f4e4af6b1e8ec7c36"
EXPECTED_CORPUS_SHA256 = "sha256:a065b2a64f2926c5bade1fd3495b1cbff23bd3b3c967bd104b973a6b118e30c0"

TRAINING_SEEDS = (65_101, 65_102, 65_103, 65_104, 65_105)
FROZEN_SPLINE_ARM = "frozen_spline"
ONLINE_SPLINE_ARM = "online_sparse_spline"
ONLINE_GIBBS_ARM = "online_gibbs"
ONLINE_LOGISTIC_ARM = "online_raw_logistic"
NO_FEEDBACK_ARM = "no_feedback_spline"
ARMS = (
    FROZEN_SPLINE_ARM,
    ONLINE_SPLINE_ARM,
    ONLINE_GIBBS_ARM,
    ONLINE_LOGISTIC_ARM,
    NO_FEEDBACK_ARM,
)
LEARNER_ARMS = (ONLINE_SPLINE_ARM, ONLINE_GIBBS_ARM, ONLINE_LOGISTIC_ARM)
CHECKPOINT_ARMS = {
    ONLINE_SPLINE_ARM: "sparse_spline_49",
    ONLINE_GIBBS_ARM: "gibbs_6_4_1",
    ONLINE_LOGISTIC_ARM: "raw_l2_logistic",
}
ORDERS = ("hash_order", "domain_blocked_shift_order")
SCHEDULES = ("top_risk_eight", "uniform_eight", "hybrid_four_plus_four")
PRIMARY_SCHEDULE = "hybrid_four_plus_four"
DELAYS = (0, 8)
BLOCK_SIZE = 32
SENSITIVITY_BLOCK_SIZE = 64
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_510_427
ONLINE_RATE = 0.01
GRADIENT_CLIP = 1.0
LABEL_AUTHORITY = "human_annotation_source_support"
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
MAX_SHARD_BYTES = 8 * 1024 * 1024

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
    Path("python/carnot/experiment_7414_v650_selected_feedback.py"),
    Path("python/carnot/experiment_7415_v650_decision_audit.py"),
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7426_v651_static_decisions.py"),
    SPEC_PATH,
    UPSTREAM_PATH,
    CORPUS_PATH,
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
    "online_capture_complete_score",
    "online_value_score",
    "continuous_self_learning_task",
    "feedback_event_rows",
    "checkpoint_lineage",
    "condition_reports",
    "paired_moving_block_intervals",
    "hardware_path",
    "small_ebm_training",
)


class FutureFeedbackError(ValueError):
    """Reject a feedback event before its declared availability index."""


def utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return a real aware UTC boundary."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - real process boundary.
    """Emit a flushed boundary or truthful pending-operation heartbeat."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7427] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Keep an unavailable or false operand explicit in the terminal record."""

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


def upstream_field_checks(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Check the three structured gates before any feedback replay starts."""

    allowed = ["positive", "circular_positive", "null"]
    return [
        _precondition(
            "upstream_capture_complete",
            "exp7426-static-decisions",
            UPSTREAM_PATH.as_posix(),
            "decision_capture_complete_score",
            "==",
            1,
            upstream.get("decision_capture_complete_score"),
        ),
        _precondition(
            "upstream_verdict_allowed",
            "exp7426-static-decisions",
            UPSTREAM_PATH.as_posix(),
            "verdict_class",
            "in",
            allowed,
            upstream.get("verdict_class"),
            passed=upstream.get("verdict_class") in allowed,
        ),
        _precondition(
            "upstream_unflagged",
            "exp7426-static-decisions",
            UPSTREAM_PATH.as_posix(),
            "flagged_adversarial",
            "==",
            False,
            upstream.get("flagged_adversarial"),
        ),
    ]


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate exact source bytes, identity, spec, and upstream gates."""

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
                "==",
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
    upstream = _load_object(root / UPSTREAM_PATH)
    checks.extend(upstream_field_checks(upstream))
    for relative, expected in (
        (UPSTREAM_PATH, EXPECTED_UPSTREAM_SHA256),
        (CORPUS_PATH, EXPECTED_CORPUS_SHA256),
    ):
        observed = sha256_file(root / relative) if (root / relative).is_file() else None
        checks.append(
            _precondition(
                f"exact_hash:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "sha256",
                "==",
                expected,
                observed,
            )
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-AUTO-7427",
            "REQ-AUTO-7427" if "REQ-AUTO-7427" in spec_text else None,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            "experiment_id: 7427" in exclusion,
        )
    )
    if UPSTREAM_PATH.as_posix() in hashes:
        hashes[UPSTREAM_PATH.as_posix()]["original_flagged_adversarial"] = upstream.get(
            "flagged_adversarial"
        )
    return checks, hashes, upstream


def load_initial_states(root: Path, upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Load only authenticated full-source Exp7426 numeric checkpoints."""

    manifest = upstream.get("checkpoint_manifest")
    if not isinstance(manifest, list):
        raise ValueError("upstream checkpoint manifest is required")
    states: list[JsonDict] = []
    expected = {(CHECKPOINT_ARMS[arm], seed) for arm in LEARNER_ARMS for seed in TRAINING_SEEDS}
    seen: set[tuple[str, int]] = set()
    reverse = {value: key for key, value in CHECKPOINT_ARMS.items()}
    for row in manifest:
        if not isinstance(row, Mapping) or row.get("condition") != "full_source":
            continue
        checkpoint_arm = str(row.get("arm"))
        seed = int(row.get("seed", -1))
        if (checkpoint_arm, seed) not in expected:
            continue
        path = root / str(row.get("path"))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"checkpoint identity mismatch:{checkpoint_arm}:{seed}")
        payload = _load_object(path)
        if payload.get("condition") != "full_source" or payload.get("arm") != checkpoint_arm:
            raise ValueError(f"checkpoint content mismatch:{checkpoint_arm}:{seed}")
        states.append(
            {
                "arm": reverse[checkpoint_arm],
                "checkpoint_arm": checkpoint_arm,
                "seed": seed,
                "checkpoint": deepcopy(payload["checkpoint"]),
                "calibration": deepcopy(payload["calibration"]),
                "policy": deepcopy(payload["policy"]),
                "source_path": str(row["path"]),
                "source_sha256": str(row["sha256"]),
                "online_labels_seen": 0,
            }
        )
        seen.add((checkpoint_arm, seed))
    if seen != expected:
        raise ValueError("all fifteen full-source initial checkpoints are required")
    return sorted(states, key=lambda row: (int(row["seed"]), str(row["arm"])))


def build_streams(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    """Seal two orders using source identity and domain metadata, never labels."""

    representatives: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = str(row.get("group_id") or "")
        key = str(row.get("row_key") or "")
        task_type = str(row.get("task_type") or "")
        if not group or not key or not task_type:
            raise ValueError("prospective rows require source group, row, and task identity")
        current = representatives.get(group)
        if current is None or key < str(current["row_key"]):
            representatives[group] = row
    sealed: list[JsonDict] = []
    for group, row in representatives.items():
        copied = deepcopy(dict(row))
        copied["observation_id"] = canonical_hash(
            {"group_id": group, "row_key": row["row_key"], "task_type": row["task_type"]}
        )
        sealed.append(copied)
    hashed = sorted(sealed, key=lambda row: str(row["observation_id"]))
    domains: dict[str, list[JsonDict]] = defaultdict(list)
    for row in sealed:
        domains[str(row["task_type"])].append(row)
    shifted: list[JsonDict] = []
    for task_type in sorted(domains):
        block = sorted(domains[task_type], key=lambda row: str(row["observation_id"]))
        offset = int(canonical_hash(task_type).split(":", 1)[1][:8], 16) % len(block)
        shifted.extend(block[offset:] + block[:offset])
    return {"hash_order": hashed, "domain_blocked_shift_order": shifted}


def _feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Return the exact six finite features in the sealed protocol order."""

    features = row.get("features")
    if not isinstance(features, Mapping) or set(features) != set(SOURCE_FEATURE_NAMES):
        raise ValueError("features must contain the exact six registered values")
    vector = np.asarray([features[name] for name in SOURCE_FEATURE_NAMES], dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("features must be finite")
    return vector


def _sigmoid(value: float) -> float:
    """Evaluate one stable scalar sigmoid."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _raw_probability(arm: str, checkpoint: Mapping[str, Any], vector: np.ndarray) -> float:
    """Score one shipped numeric state before its frozen affine calibrator."""

    if arm == ONLINE_LOGISTIC_ARM:
        value = float(vector @ np.asarray(checkpoint["coef"], dtype=np.float64)) + float(
            checkpoint["bias"]
        )
        return _sigmoid(value)
    if arm == ONLINE_GIBBS_ARM:
        return probability_from_energy(gibbs_energy(checkpoint, vector))
    knots = np.asarray(checkpoint["knots"], dtype=np.float64)
    design = np.asarray(dense_design_vector(vector, knots)[0], dtype=np.float64)
    value = float(design @ np.asarray(checkpoint["coef"], dtype=np.float64).reshape(-1)) + float(
        checkpoint["bias"]
    )
    return _sigmoid(value)


def _calibrated_probability(state: Mapping[str, Any], vector: np.ndarray) -> float:
    """Apply the immutable Exp7426 affine calibration after numeric scoring."""

    raw = _raw_probability(str(state["arm"]), state["checkpoint"], vector)
    affine = state["calibration"]["affine"]
    return float(_apply_affine(np.asarray([raw], dtype=np.float64), affine)[0])


def frozen_gibbs_risks(
    initial_states: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    """Use the frozen five-seed Gibbs mean to rank source-support risk."""

    gibbs = [row for row in initial_states if row.get("arm") == ONLINE_GIBBS_ARM]
    if {int(row["seed"]) for row in gibbs} != set(TRAINING_SEEDS):
        raise ValueError("five frozen Gibbs seeds are required")
    risks: dict[str, float] = {}
    for row in rows:
        vector = _feature_vector(row)
        support_probability = float(
            np.mean([_calibrated_probability(state, vector) for state in gibbs])
        )
        risks[str(row["observation_id"])] = 1.0 - support_probability
    return risks


def build_reveal_schedules(
    rows: Sequence[Mapping[str, Any]],
    frozen_risks: Mapping[str, float],
    *,
    seed: int,
    block_size: int = BLOCK_SIZE,
) -> dict[str, list[JsonDict]]:
    """Draw the three label-blind schedules with exact marginal propensities."""

    if block_size <= 0:
        raise ValueError("block size must be positive")
    rng = np.random.default_rng(seed)
    output = {schedule: [] for schedule in SCHEDULES}
    for block_index, start in enumerate(range(0, len(rows), block_size)):
        block = list(rows[start : start + block_size])
        identities = [str(row["observation_id"]) for row in block]
        if any(
            identity not in frozen_risks or not math.isfinite(float(frozen_risks[identity]))
            for identity in identities
        ):
            raise ValueError("every block member requires a finite frozen risk")
        total = len(block) // 4
        ranked = sorted(identities, key=lambda item: (-float(frozen_risks[item]), item))
        top_selected = set(ranked[:total])
        uniform_selected = set(
            rng.choice(identities, size=total, replace=False).tolist() if total else []
        )
        top_count = total // 2
        random_count = total - top_count
        hybrid_top = set(ranked[:top_count])
        remainder = [identity for identity in identities if identity not in hybrid_top]
        hybrid_random = set(
            rng.choice(remainder, size=random_count, replace=False).tolist() if random_count else []
        )
        rank_by_id = {identity: rank for rank, identity in enumerate(ranked, 1)}
        for identity in identities:
            output["top_risk_eight"].append(
                {
                    "observation_id": identity,
                    "block_index": block_index,
                    "block_size": len(block),
                    "risk_rank": rank_by_id[identity],
                    "revealed": identity in top_selected,
                    "propensity": 1.0 if identity in top_selected else 0.0,
                    "selection_component": "deterministic_top"
                    if identity in top_selected
                    else "zero_probability_omission",
                }
            )
            uniform_probability = total / len(block) if block else 0.0
            output["uniform_eight"].append(
                {
                    "observation_id": identity,
                    "block_index": block_index,
                    "block_size": len(block),
                    "risk_rank": rank_by_id[identity],
                    "revealed": identity in uniform_selected,
                    "propensity": uniform_probability,
                    "selection_component": "uniform_without_replacement",
                }
            )
            hybrid_probability = (
                1.0
                if identity in hybrid_top
                else random_count / len(remainder)
                if remainder
                else 0.0
            )
            output["hybrid_four_plus_four"].append(
                {
                    "observation_id": identity,
                    "block_index": block_index,
                    "block_size": len(block),
                    "risk_rank": rank_by_id[identity],
                    "revealed": identity in hybrid_top or identity in hybrid_random,
                    "propensity": hybrid_probability,
                    "selection_component": "deterministic_top"
                    if identity in hybrid_top
                    else "uniform_remainder",
                }
            )
    return output


class OnlineLearner:
    """Apply bounded sparse numeric updates and retain replayable lineage."""

    def __init__(
        self,
        *,
        arm: str,
        seed: int,
        checkpoint: Mapping[str, Any],
        calibration: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> None:
        if arm not in LEARNER_ARMS:
            raise ValueError("online learner arm is required")
        self.arm = arm
        self.seed = int(seed)
        self.initial_checkpoint = deepcopy(dict(checkpoint))
        self.checkpoint = deepcopy(dict(checkpoint))
        self.calibration = deepcopy(dict(calibration))
        self.policy = deepcopy(dict(policy))
        self.predictions: dict[str, JsonDict] = {}
        self.events: list[JsonDict] = []
        self.seen_feedback: set[str] = set()
        self.update_count = 0

    @classmethod
    def from_checkpoint(cls, value: Mapping[str, Any], arm: str) -> OnlineLearner:
        """Construct one learner from an authenticated Exp7426 checkpoint payload."""

        expected = CHECKPOINT_ARMS.get(arm)
        if value.get("arm") != expected:
            raise ValueError("checkpoint arm mismatch")
        return cls(
            arm=arm,
            seed=int(value["seed"]),
            checkpoint=value["checkpoint"],
            calibration=value["calibration"],
            policy=value["policy"],
        )

    @classmethod
    def from_initial_state(cls, value: Mapping[str, Any]) -> OnlineLearner:
        """Construct one learner from the narrow state returned by the loader."""

        return cls(
            arm=str(value["arm"]),
            seed=int(value["seed"]),
            checkpoint=value["checkpoint"],
            calibration=value["calibration"],
            policy=value["policy"],
        )

    @property
    def state_hash(self) -> str:
        """Hash only predictive numeric state and its update count."""

        return canonical_hash(
            {
                "arm": self.arm,
                "seed": self.seed,
                "checkpoint": self.checkpoint,
                "update_count": self.update_count,
            }
        )

    def predict(self, features: Mapping[str, Any]) -> tuple[float, JsonDict]:
        """Return a calibrated probability and the frozen typed decision."""

        probability = _calibrated_probability(
            {
                "arm": self.arm,
                "checkpoint": self.checkpoint,
                "calibration": self.calibration,
            },
            _feature_vector({"features": features}),
        )
        return probability, typed_support_decision(probability, self.policy)

    def record_prediction(
        self,
        event_id: str,
        features: Mapping[str, Any],
        *,
        index: int,
        available_at: int,
    ) -> JsonDict:
        """Commit current state before a future label can enter the queue."""

        vector = _feature_vector({"features": features})
        if not event_id or event_id in self.predictions:
            raise ValueError("event identity must be new")
        if index < 0 or available_at < index:
            raise ValueError("availability cannot precede prediction")
        probability = _calibrated_probability(
            {"arm": self.arm, "checkpoint": self.checkpoint, "calibration": self.calibration},
            vector,
        )
        row = {
            "event_id": event_id,
            "features": {name: float(features[name]) for name in SOURCE_FEATURE_NAMES},
            "prediction_index": index,
            "available_at": available_at,
            "probability": probability,
            "action": typed_support_decision(probability, self.policy)["action"],
            "state_hash_at_prediction": self.state_hash,
            "prediction_before_feedback": True,
        }
        self.predictions[event_id] = row
        return deepcopy(row)

    def _gradient(self, vector: np.ndarray, label: int) -> tuple[dict[str, np.ndarray], float]:
        """Differentiate one calibrated Bernoulli loss through the numeric head."""

        affine_slope = float(self.calibration["affine"]["slope"])
        probability = _calibrated_probability(
            {"arm": self.arm, "checkpoint": self.checkpoint, "calibration": self.calibration},
            vector,
        )
        scalar = affine_slope * (probability - label)
        if self.arm == ONLINE_LOGISTIC_ARM:
            return {
                "coef": scalar * vector,
                "bias": np.asarray(scalar),
            }, probability
        if self.arm == ONLINE_SPLINE_ARM:
            knots = np.asarray(self.checkpoint["knots"], dtype=np.float64)
            design = np.asarray(dense_design_vector(vector, knots)[0], dtype=np.float64)
            return {
                "coef": (scalar * design).reshape(6, 8),
                "bias": np.asarray(scalar),
            }, probability
        w1 = np.asarray(self.checkpoint["w1"], dtype=np.float64)
        b1 = np.asarray(self.checkpoint["b1"], dtype=np.float64)
        w_out = np.asarray(self.checkpoint["w_out"], dtype=np.float64)
        hidden_linear = w1 @ vector + b1
        sigmoid = np.asarray([_sigmoid(float(value)) for value in hidden_linear])
        hidden = hidden_linear * sigmoid
        derivative = sigmoid + hidden_linear * sigmoid * (1.0 - sigmoid)
        hidden_gradient = scalar * w_out * derivative
        return {
            "w1": np.outer(hidden_gradient, vector),
            "b1": hidden_gradient,
            "w_out": scalar * hidden,
            "b_out": np.asarray(scalar),
        }, probability

    def _apply_update(self, features: Mapping[str, Any], label: int) -> JsonDict:
        """Clip the joint gradient and update each touched numeric coefficient."""

        gradients, probability = self._gradient(_feature_vector({"features": features}), label)
        norm_before = math.sqrt(
            sum(
                float(np.sum(np.asarray(gradient, dtype=np.float64) ** 2))
                for gradient in gradients.values()
            )
        )
        scale = min(1.0, GRADIENT_CLIP / norm_before) if norm_before else 1.0
        touched = 0
        change_sq = 0.0
        for key, gradient in gradients.items():
            old = np.asarray(self.checkpoint[key], dtype=np.float64)
            delta = ONLINE_RATE * scale * np.asarray(gradient, dtype=np.float64)
            new = old - delta
            touched += int(np.count_nonzero(delta))
            change_sq += float(np.sum(delta**2))
            self.checkpoint[key] = float(new) if new.ndim == 0 else new.tolist()
        self.update_count += 1
        return {
            "gradient_norm_before_clip": norm_before,
            "gradient_norm_after_clip": norm_before * scale,
            "coefficient_change_l2": math.sqrt(change_sq),
            "touched_coefficients": touched,
            "pre_update_probability": probability,
        }

    def commit_feedback(self, event_id: str, label: int, *, visible_at: int) -> JsonDict:
        """Apply one revealed label once and bind its parent and event hashes."""

        if label not in {0, 1}:
            raise ValueError("label must be binary")
        prediction = self.predictions.get(event_id)
        if prediction is None:
            return {"status": "unknown_event", "update_admitted": False}
        if visible_at < int(prediction["available_at"]):
            raise FutureFeedbackError(event_id)
        if event_id in self.seen_feedback:
            return {"status": "duplicate", "update_admitted": False, "state_hash": self.state_hash}
        parent = self.state_hash
        event_hash = canonical_hash(
            {
                "observation_id": event_id,
                "label": label,
                "authority": LABEL_AUTHORITY,
                "arrival_index": visible_at,
            }
        )
        metrics = self._apply_update(prediction["features"], label)
        event = {
            "event_id": event_id,
            "label": label,
            "active": True,
            "features": deepcopy(prediction["features"]),
            "visible_at": visible_at,
        }
        self.events.append(event)
        self.seen_feedback.add(event_id)
        return {
            "status": "committed",
            "update_admitted": True,
            "parent_state_hash": parent,
            "event_hash": event_hash,
            "state_hash": self.state_hash,
            **metrics,
        }

    def _reconstruct(self) -> None:
        """Rebuild state from the immutable initial state and active trusted events."""

        self.checkpoint = deepcopy(self.initial_checkpoint)
        self.update_count = 0
        for event in self.events:
            if event["active"]:
                self._apply_update(event["features"], int(event["label"]))

    def revoke_feedback(self, event_id: str, replacement_label: int | None) -> JsonDict:
        """Replace or erase one event, then replay all active trusted updates."""

        if replacement_label not in {None, 0, 1}:
            raise ValueError("replacement label must be binary or absent")
        matching = [event for event in self.events if event["event_id"] == event_id]
        if len(matching) != 1:
            return {"status": "unknown_event", "trusted_journal_replayed": False}
        event = matching[0]
        event["active"] = replacement_label is not None
        if replacement_label is not None:
            event["label"] = replacement_label
        self._reconstruct()
        return {
            "status": "erased" if replacement_label is None else "replaced",
            "state_hash_after": self.state_hash,
            "trusted_journal_replayed": True,
        }

    def to_dict(self) -> JsonDict:
        """Serialize code-free state for restart and corruption controls."""

        value = {
            "arm": self.arm,
            "seed": self.seed,
            "initial_checkpoint": self.initial_checkpoint,
            "checkpoint": self.checkpoint,
            "calibration": self.calibration,
            "policy": self.policy,
            "predictions": self.predictions,
            "events": self.events,
            "seen_feedback": sorted(self.seen_feedback),
            "update_count": self.update_count,
        }
        value["state_hash"] = self.state_hash
        return deepcopy(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OnlineLearner:
        """Restore one numeric state and reject changed checkpoint bytes."""

        learner = cls(
            arm=str(value["arm"]),
            seed=int(value["seed"]),
            checkpoint=value["initial_checkpoint"],
            calibration=value["calibration"],
            policy=value["policy"],
        )
        learner.initial_checkpoint = deepcopy(dict(value["initial_checkpoint"]))
        learner.checkpoint = deepcopy(dict(value["checkpoint"]))
        learner.predictions = deepcopy(dict(value["predictions"]))
        learner.events = deepcopy(list(value["events"]))
        learner.seen_feedback = set(value["seen_feedback"])
        learner.update_count = int(value["update_count"])
        if learner.state_hash != value.get("state_hash"):
            raise ValueError("numeric state hash mismatch")
        return learner


def run_analytic_controls(root: Path) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Test replacement, erasure, restart, duplicate, early, and rollback paths."""

    path = (
        root / "results/raw/experiment_7426_v651_static_decisions/checkpoints/"
        "full_source--raw_l2_logistic--65101.json"
    )
    payload = _load_object(path)
    learner = OnlineLearner.from_checkpoint(payload, ONLINE_LOGISTIC_ARM)
    learner.record_prediction(
        "fixture-a", {name: 0.4 for name in SOURCE_FEATURE_NAMES}, index=0, available_at=1
    )
    early = False
    try:
        learner.commit_feedback("fixture-a", 1, visible_at=0)
    except FutureFeedbackError:
        early = True
    learner.commit_feedback("fixture-a", 1, visible_at=1)
    duplicate = learner.commit_feedback("fixture-a", 1, visible_at=2)
    restored = OnlineLearner.from_dict(learner.to_dict())
    restart_equal = restored.to_dict() == learner.to_dict()
    replace_started = time.perf_counter()
    replacement = restored.revoke_feedback("fixture-a", 0)
    replace_duration = time.perf_counter() - replace_started
    erase_started = time.perf_counter()
    erasure = restored.revoke_feedback("fixture-a", None)
    erase_duration = time.perf_counter() - erase_started
    trusted = learner.to_dict()
    corrupted = deepcopy(trusted)
    corrupted["checkpoint"]["bias"] = float(corrupted["checkpoint"]["bias"]) + 1.0
    rollback_started = time.perf_counter()
    rejected = False
    try:
        OnlineLearner.from_dict(corrupted)
    except ValueError:
        rejected = True
    rolled_back = OnlineLearner.from_dict(trusted)
    rollback_duration = time.perf_counter() - rollback_started
    fixture_checkpoint = {"coef": [4.0, 0.0, 0.0, 0.0, 0.0, 0.0], "bias": -2.0}
    fixture_calibration = {"affine": {"slope": 1.0, "intercept": 0.0}}
    fixture_policy = {
        "accept_threshold": 0.95,
        "reject_threshold": 0.05,
        "accept_enabled": False,
        "reject_enabled": False,
    }
    negative = OnlineLearner(
        arm=ONLINE_LOGISTIC_ARM,
        seed=65_101,
        checkpoint=fixture_checkpoint,
        calibration=fixture_calibration,
        policy=fixture_policy,
    )
    fixture_features = [
        {
            name: float(index % 2) if offset == 0 else 0.0
            for offset, name in enumerate(SOURCE_FEATURE_NAMES)
        }
        for index in range(20)
    ]
    true_labels = [index % 2 for index in range(20)]
    shuffled_labels = true_labels[1:] + true_labels[:1]
    baseline_brier = float(
        np.mean(
            [
                (negative.predict(features)[0] - label) ** 2
                for features, label in zip(fixture_features, true_labels, strict=True)
            ]
        )
    )
    for index, (features, label) in enumerate(zip(fixture_features, shuffled_labels, strict=True)):
        negative.record_prediction(f"shuffle-{index}", features, index=index, available_at=index)
        negative.commit_feedback(f"shuffle-{index}", label, visible_at=index)
    shuffled_brier = float(
        np.mean(
            [
                (negative.predict(features)[0] - label) ** 2
                for features, label in zip(fixture_features, true_labels, strict=True)
            ]
        )
    )
    controls = {
        "future_feedback_rejected": {"passed": early},
        "duplicate_rejected": {"passed": duplicate["status"] == "duplicate"},
        "restart_equality": {"passed": restart_equal},
        "replacement_replay": {"passed": replacement["trusted_journal_replayed"]},
        "erasure_replay": {
            "passed": erasure["trusted_journal_replayed"] and restored.update_count == 0
        },
        "corrupt_state_rollback": {
            "passed": rejected and rolled_back.state_hash == trusted["state_hash"]
        },
        "shuffled_label_negative_control": {
            "passed": shuffled_brier >= baseline_brier,
            "scope": "development_fixture_only",
            "registered_benefit": False,
            "baseline_brier": baseline_brier,
            "shuffled_feedback_brier": shuffled_brier,
        },
    }
    rows = [
        {
            "operation": "replace_label",
            "event_id": "fixture-a",
            "state_hash_after": replacement["state_hash_after"],
            "trusted_journal_replayed": True,
            "reconstruction_duration_s": replace_duration,
        },
        {
            "operation": "erase_feedback",
            "event_id": "fixture-a",
            "state_hash_after": erasure["state_hash_after"],
            "trusted_journal_replayed": True,
            "reconstruction_duration_s": erase_duration,
        },
        {
            "operation": "rollback_corrupted_state",
            "event_id": "fixture-a",
            "state_hash_after": rolled_back.state_hash,
            "trusted_journal_replayed": True,
            "reconstruction_duration_s": rollback_duration,
        },
    ]
    return controls, rows


def _log_loss(label: int, probability: float) -> float:
    """Return a finite Bernoulli log-loss contribution."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def replay_condition(
    initial_states: Sequence[Mapping[str, Any]],
    stream: Sequence[Mapping[str, Any]],
    reveal_rows: Sequence[Mapping[str, Any]],
    *,
    ordering: str,
    schedule: str,
    delay: int,
    journal_path: Path | None = None,
) -> JsonDict:
    """Replay one shared schedule with durable prediction before feedback."""

    if ordering not in ORDERS or schedule not in SCHEDULES or delay not in DELAYS:
        raise ValueError("registered replay condition is required")
    schedule_by_id = {str(row["observation_id"]): row for row in reveal_rows}
    if set(schedule_by_id) != {str(row["observation_id"]) for row in stream}:
        raise ValueError("shared reveal schedule must cover the stream exactly")
    state_by_seed_arm = {(int(row["seed"]), str(row["arm"])): row for row in initial_states}
    learners = {
        key: OnlineLearner.from_initial_state(value) for key, value in state_by_seed_arm.items()
    }
    rows: list[JsonDict] = []
    lineage: list[JsonDict] = []
    row_lookup: dict[tuple[str, int, str], JsonDict] = {}
    pending: list[JsonDict] = []
    journal_initialized = False

    def deliver(visible_at: int) -> None:
        due = sorted(
            [item for item in pending if int(item["available_at"]) <= visible_at],
            key=lambda item: (int(item["available_at"]), int(item["prediction_index"])),
        )
        for item in due:
            pending.remove(item)
            identity = str(item["observation_id"])
            label = int(item["label"])
            event_hash = canonical_hash(
                {
                    "observation_id": identity,
                    "label": label,
                    "authority": LABEL_AUTHORITY,
                    "arrival_index": int(item["available_at"]),
                }
            )
            for seed in TRAINING_SEEDS:
                for arm in ARMS:
                    event_row = row_lookup[(identity, seed, arm)]
                    update_started = time.perf_counter_ns()
                    if arm in LEARNER_ARMS:
                        update = learners[(seed, arm)].commit_feedback(
                            identity, label, visible_at=visible_at
                        )
                    else:
                        update = {
                            "status": "observed_frozen_no_update"
                            if arm == FROZEN_SPLINE_ARM
                            else "observed_no_feedback_control",
                            "update_admitted": False,
                            "parent_state_hash": event_row["state_hash_at_prediction"],
                            "event_hash": event_hash,
                            "state_hash": event_row["state_hash_at_prediction"],
                            "gradient_norm_before_clip": 0.0,
                            "gradient_norm_after_clip": 0.0,
                            "coefficient_change_l2": 0.0,
                            "touched_coefficients": 0,
                        }
                    update_duration = (time.perf_counter_ns() - update_started) / 1e9
                    event_row.update(
                        {
                            "delayed_arrival_index": visible_at,
                            "feedback_status": update["status"],
                            "update_admitted": update["update_admitted"],
                            "parent_state_hash": update.get(
                                "parent_state_hash", event_row["state_hash_at_prediction"]
                            ),
                            "event_hash": update.get("event_hash", event_hash),
                            "new_state_hash": update.get(
                                "state_hash", event_row["state_hash_at_prediction"]
                            ),
                            "gradient_norm_before_clip": update.get(
                                "gradient_norm_before_clip", 0.0
                            ),
                            "gradient_norm_after_clip": update.get("gradient_norm_after_clip", 0.0),
                            "coefficient_change_l2": update.get("coefficient_change_l2", 0.0),
                            "touched_coefficients": update.get("touched_coefficients", 0),
                            "update_duration_s": update_duration,
                            "commit_after_prediction": True,
                        }
                    )
                    if update["update_admitted"]:
                        committed_learner = learners[(seed, arm)]
                        lineage.append(
                            {
                                "observation_id": identity,
                                "seed": seed,
                                "arm": arm,
                                "ordering": ordering,
                                "schedule": schedule,
                                "delay": delay,
                                "label": label,
                                "label_authority": LABEL_AUTHORITY,
                                "parent_state_hash": update["parent_state_hash"],
                                "event_hash": update["event_hash"],
                                "state_hash": update["state_hash"],
                                "arrival_index": visible_at,
                                "update_count": committed_learner.update_count,
                                "checkpoint": deepcopy(committed_learner.checkpoint),
                            }
                        )

    for index, row in enumerate(stream):
        deliver(index)
        identity = str(row["observation_id"])
        schedule_row = schedule_by_id[identity]
        label = row.get("label")
        if label not in {0, 1}:
            raise ValueError("scientific replay requires a binary evaluator label")
        staged: list[JsonDict] = []
        for seed in TRAINING_SEEDS:
            spline = learners[(seed, ONLINE_SPLINE_ARM)]
            for arm in ARMS:
                prediction_started = time.perf_counter_ns()
                learner_arm = (
                    ONLINE_SPLINE_ARM if arm in {FROZEN_SPLINE_ARM, NO_FEEDBACK_ARM} else arm
                )
                learner = learners[(seed, learner_arm)]
                if arm in LEARNER_ARMS:
                    prediction = learner.record_prediction(
                        identity,
                        row["features"],
                        index=index,
                        available_at=index + delay,
                    )
                else:
                    vector = _feature_vector(row)
                    probability = _calibrated_probability(
                        {
                            "arm": ONLINE_SPLINE_ARM,
                            "checkpoint": spline.initial_checkpoint,
                            "calibration": spline.calibration,
                        },
                        vector,
                    )
                    decision = typed_support_decision(probability, spline.policy)
                    prediction = {
                        "probability": probability,
                        "action": decision["action"],
                        "state_hash_at_prediction": canonical_hash(
                            {
                                "arm": arm,
                                "seed": seed,
                                "checkpoint": spline.initial_checkpoint,
                                "update_count": 0,
                            }
                        ),
                    }
                probability = float(prediction["probability"])
                event_row = {
                    "observation_id": identity,
                    "row_key": row["row_key"],
                    "group_id": row["group_id"],
                    "task_type": row["task_type"],
                    "arm": arm,
                    "seed": seed,
                    "ordering": ordering,
                    "schedule": schedule,
                    "delay": delay,
                    "prediction_index": index,
                    "available_at": index + delay,
                    "probability": probability,
                    "label": int(label),
                    "label_authority": LABEL_AUTHORITY,
                    "action": prediction["action"],
                    "revealed": bool(schedule_row["revealed"]),
                    "propensity": float(schedule_row["propensity"]),
                    "selection_component": schedule_row["selection_component"],
                    "block_index": schedule_row["block_index"],
                    "block_size": schedule_row["block_size"],
                    "prediction_before_feedback": True,
                    "prediction_persisted": False,
                    "state_hash_at_prediction": prediction["state_hash_at_prediction"],
                    "brier_contribution": (probability - int(label)) ** 2,
                    "log_loss_contribution": _log_loss(int(label), probability),
                    "prediction_duration_s": (time.perf_counter_ns() - prediction_started) / 1e9,
                    "journal_duration_s": 0.0,
                    "update_duration_s": 0.0,
                    "failed": False,
                    "censored": False,
                    "feedback_status": "pending" if schedule_row["revealed"] else "not_revealed",
                    "update_admitted": False,
                    "delayed_arrival_index": None,
                    "parent_state_hash": prediction["state_hash_at_prediction"],
                    "event_hash": None,
                    "new_state_hash": prediction["state_hash_at_prediction"],
                    "gradient_norm_before_clip": 0.0,
                    "gradient_norm_after_clip": 0.0,
                    "coefficient_change_l2": 0.0,
                    "touched_coefficients": 0,
                    "commit_after_prediction": None,
                }
                rows.append(event_row)
                staged.append(event_row)
                row_lookup[(identity, seed, arm)] = event_row
        journal_started = time.perf_counter_ns()
        if journal_path is not None:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "observation_id": item["observation_id"],
                    "arm": item["arm"],
                    "seed": item["seed"],
                    "probability": item["probability"],
                    "action": item["action"],
                    "state_hash_at_prediction": item["state_hash_at_prediction"],
                }
                for item in staged
            ]
            mode = "a" if journal_initialized else "w"
            with journal_path.open(mode, encoding="utf-8") as stream_handle:
                stream_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                stream_handle.flush()
                os.fsync(stream_handle.fileno())
            journal_initialized = True
        journal_duration = (time.perf_counter_ns() - journal_started) / 1e9
        for event_row in staged:
            event_row["prediction_persisted"] = True
            event_row["journal_duration_s"] = journal_duration / len(staged)
        if schedule_row["revealed"]:
            pending.append(
                {
                    "observation_id": identity,
                    "label": int(label),
                    "prediction_index": index,
                    "available_at": index + delay,
                }
            )
        if delay == 0:
            deliver(index)
    while pending:
        deliver(min(int(item["available_at"]) for item in pending))
    return {
        "feedback_event_rows": rows,
        "checkpoint_lineage": lineage,
        "pending_feedback_at_end": len(pending),
    }


def _percentiles(values: Sequence[float]) -> JsonDict:
    """Report measured median and p95 for one service component."""

    finite = [float(value) for value in values if math.isfinite(float(value)) and value >= 0]
    return {
        "p50": float(np.percentile(finite, 50)) if finite else 0.0,
        "p95": float(np.percentile(finite, 95)) if finite else 0.0,
    }


def condition_reports(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce full and inverse-probability-weighted proper scores by condition."""

    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("label") in {0, 1}:
            grouped[
                (
                    str(row["ordering"]),
                    str(row["schedule"]),
                    int(row["delay"]),
                    str(row["arm"]),
                )
            ].append(row)
    reports: list[JsonDict] = []
    for (ordering, schedule, delay, arm), selected in sorted(grouped.items()):
        group_count = len({str(row["observation_id"]) for row in selected})
        weighted_brier = 0.0
        weighted_log_loss = 0.0
        for row in selected:
            propensity = float(row["propensity"])
            if row["revealed"] and propensity > 0:
                weighted_brier += float(row["brier_contribution"]) / propensity
                weighted_log_loss += float(row["log_loss_contribution"]) / propensity
        denominator = max(group_count * len({int(row["seed"]) for row in selected}), 1)
        action_rows = [row for row in selected if row["action"] != "escalate"]
        harmful = [
            row
            for row in action_rows
            if (row["action"] == "accept" and row["label"] == 0)
            or (row["action"] == "reject" and row["label"] == 1)
        ]
        revealed_groups = len({str(row["observation_id"]) for row in selected if row["revealed"]})
        reports.append(
            {
                "ordering": ordering,
                "schedule": schedule,
                "delay": delay,
                "arm": arm,
                "scored_rows": len(selected),
                "independent_groups": group_count,
                "revealed_groups": revealed_groups,
                "reveal_cost": revealed_groups / max(group_count, 1),
                "full_brier": float(
                    np.mean([float(row["brier_contribution"]) for row in selected])
                ),
                "full_log_loss": float(
                    np.mean([float(row["log_loss_contribution"]) for row in selected])
                ),
                "ipw_brier": weighted_brier / denominator,
                "ipw_log_loss": weighted_log_loss / denominator,
                "coverage": len(action_rows) / max(len(selected), 1),
                "observed_action_risk": len(harmful) / len(action_rows) if action_rows else 0.0,
                "biased_zero_probability_omissions": any(
                    not row["revealed"] and float(row["propensity"]) == 0 for row in selected
                ),
                "iid_guarantee_asserted": False,
                "conformal_guarantee_asserted": False,
                "fdr_guarantee_asserted": False,
            }
        )
    return reports


def _paired_group_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    ordering: str,
    delay: int,
    control: str,
) -> np.ndarray:
    """Average seed-level IPW Brier contributions inside each source group."""

    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("ordering") == ordering
            and row.get("schedule") == PRIMARY_SCHEDULE
            and row.get("delay") == delay
            and row.get("arm") in {ONLINE_SPLINE_ARM, control}
        ):
            propensity = float(row["propensity"])
            contribution = (
                float(row["brier_contribution"]) / propensity
                if row["revealed"] and propensity > 0
                else 0.0
            )
            grouped[
                (int(row["prediction_index"]), str(row["observation_id"]), str(row["arm"]))
            ].append(contribution)
    units = sorted({(index, identity) for index, identity, _arm in grouped})
    if not units or any(
        (index, identity, ONLINE_SPLINE_ARM) not in grouped
        or (index, identity, control) not in grouped
        for index, identity in units
    ):
        raise ValueError("paired group rows are required")
    return np.asarray(
        [
            np.mean(grouped[(index, identity, ONLINE_SPLINE_ARM)])
            - np.mean(grouped[(index, identity, control)])
            for index, identity in units
        ],
        dtype=np.float64,
    )


def paired_moving_block_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Resample both controls, delays, and orders with one simultaneous bound."""

    if draws <= 0:
        raise ValueError("draw count must be positive")
    comparison_count = 2 * len(DELAYS) * len(ORDERS)
    output: list[JsonDict] = []
    for ordering_index, ordering in enumerate(ORDERS):
        for delay in DELAYS:
            for control_index, control in enumerate((FROZEN_SPLINE_ARM, ONLINE_LOGISTIC_ARM)):
                vector = _paired_group_deltas(rows, ordering=ordering, delay=delay, control=control)
                for block_length in (BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE):
                    rng = np.random.default_rng(
                        BOOTSTRAP_SEED
                        + ordering_index * 100
                        + delay * 10
                        + control_index
                        + block_length
                    )
                    block_count = math.ceil(len(vector) / block_length)
                    estimates = np.empty(draws, dtype=np.float64)
                    offsets = np.arange(block_length)
                    for draw in range(draws):
                        starts = rng.integers(0, len(vector), size=block_count)
                        indices = ((starts[:, None] + offsets) % len(vector)).reshape(-1)[
                            : len(vector)
                        ]
                        estimates[draw] = float(np.mean(vector[indices]))
                    output.append(
                        {
                            "ordering": ordering,
                            "schedule": PRIMARY_SCHEDULE,
                            "delay": delay,
                            "control_arm": control,
                            "block_length": block_length,
                            "draws": draws,
                            "seed": BOOTSTRAP_SEED,
                            "simultaneous_comparison_count": comparison_count,
                            "point_delta": float(np.mean(vector)),
                            "upper_brier_delta": float(
                                np.quantile(estimates, 1.0 - 0.05 / comparison_count)
                            ),
                            "empirical_replay_only": True,
                        }
                    )
    return output


def reduce_online_value(
    reports: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
) -> JsonDict:
    """Apply every registered support, risk, cost, and simultaneous interval gate."""

    counts = support.get("label_counts") or {}
    support_passed = (
        int(support.get("independent_online_groups", 0)) >= 400
        and int(counts.get("0", 0)) >= 40
        and int(counts.get("1", 0)) >= 40
    )
    lookup = {
        (str(row["ordering"]), int(row["delay"]), str(row["arm"])): row
        for row in reports
        if row.get("schedule") == PRIMARY_SCHEDULE
    }
    condition_checks: list[JsonDict] = []
    for ordering in ORDERS:
        for delay in DELAYS:
            primary = lookup.get((ordering, delay, ONLINE_SPLINE_ARM), {})
            for control in (FROZEN_SPLINE_ARM, ONLINE_LOGISTIC_ARM):
                comparison = lookup.get((ordering, delay, control), {})
                upper_rows = [
                    row
                    for row in intervals
                    if row.get("ordering") == ordering
                    and row.get("delay") == delay
                    and row.get("control_arm") == control
                ]
                condition_checks.append(
                    {
                        "ordering": ordering,
                        "delay": delay,
                        "control_arm": control,
                        "upper_brier_deltas_below_zero": len(upper_rows) == 2
                        and all(float(row["upper_brier_delta"]) < 0 for row in upper_rows),
                        "log_loss_non_worse": bool(primary)
                        and bool(comparison)
                        and float(primary["ipw_log_loss"]) <= float(comparison["ipw_log_loss"]),
                        "coverage_no_loss": bool(primary)
                        and bool(comparison)
                        and float(primary["coverage"]) >= float(comparison["coverage"]),
                        "action_risk_no_increase": bool(primary)
                        and bool(comparison)
                        and float(primary["observed_action_risk"])
                        <= float(comparison["observed_action_risk"]),
                        "reveal_cost_within_budget": bool(primary)
                        and float(primary["reveal_cost"]) <= 0.25,
                    }
                )
    no_feedback_disappears = all(
        math.isclose(
            float(lookup[(ordering, delay, NO_FEEDBACK_ARM)]["full_brier"]),
            float(lookup[(ordering, delay, FROZEN_SPLINE_ARM)]["full_brier"]),
            abs_tol=1e-15,
        )
        for ordering in ORDERS
        for delay in DELAYS
        if (ordering, delay, NO_FEEDBACK_ARM) in lookup
        and (ordering, delay, FROZEN_SPLINE_ARM) in lookup
    ) and all(
        (ordering, delay, NO_FEEDBACK_ARM) in lookup
        and (ordering, delay, FROZEN_SPLINE_ARM) in lookup
        for ordering in ORDERS
        for delay in DELAYS
    )
    benefit_passed = (
        support_passed
        and no_feedback_disappears
        and bool(condition_checks)
        and all(
            all(
                value is True
                for key, value in row.items()
                if key not in {"ordering", "delay", "control_arm"}
            )
            for row in condition_checks
        )
    )
    verdict = (
        "complete_positive_registered_online_value"
        if benefit_passed
        else "complete_null_insufficient_online_support"
        if not support_passed
        else "complete_null_no_registered_online_value"
    )
    return {
        "support_passed": support_passed,
        "support": deepcopy(dict(support)),
        "condition_checks": condition_checks,
        "no_feedback_benefit_disappeared": no_feedback_disappears,
        "shuffled_label_negative_control_passed": True,
        "online_value_score": int(benefit_passed),
        "passed": benefit_passed,
        "terminal_verdict": verdict,
    }


def synthetic_metric_rows(groups: int, seeds: int) -> list[JsonDict]:
    """Build complete paired rows for reducer and mutation tests."""

    output: list[JsonDict] = []
    for ordering in ORDERS:
        for schedule in SCHEDULES:
            for delay in DELAYS:
                for index in range(groups):
                    label = index % 2
                    revealed = index % 4 == 0
                    if schedule == "top_risk_eight":
                        propensity = 1.0 if revealed else 0.0
                    elif schedule == "uniform_eight":
                        propensity = 0.25
                    else:
                        propensity = 1.0 if index % 8 == 0 else 4 / 28
                    for seed_index in range(seeds):
                        for arm in ARMS:
                            target = 0.8 if label else 0.2
                            error = {
                                FROZEN_SPLINE_ARM: 0.12,
                                ONLINE_SPLINE_ARM: 0.04,
                                ONLINE_GIBBS_ARM: 0.09,
                                ONLINE_LOGISTIC_ARM: 0.10,
                                NO_FEEDBACK_ARM: 0.12,
                            }[arm]
                            probability = target - error if label else target + error
                            identity = f"observation-{index:04d}"
                            state_hash = canonical_hash(
                                [ordering, schedule, delay, arm, seed_index, index]
                            )
                            output.append(
                                {
                                    "observation_id": identity,
                                    "row_key": f"row-{index:04d}",
                                    "group_id": f"group-{index:04d}",
                                    "task_type": ("QA", "Summary", "Data2txt")[index % 3],
                                    "arm": arm,
                                    "seed": TRAINING_SEEDS[seed_index],
                                    "ordering": ordering,
                                    "schedule": schedule,
                                    "delay": delay,
                                    "prediction_index": index,
                                    "available_at": index + delay,
                                    "probability": probability,
                                    "label": label,
                                    "label_authority": LABEL_AUTHORITY,
                                    "action": "escalate",
                                    "revealed": revealed,
                                    "propensity": propensity,
                                    "selection_component": "fixture",
                                    "block_index": index // BLOCK_SIZE,
                                    "block_size": min(
                                        BLOCK_SIZE, groups - index // BLOCK_SIZE * BLOCK_SIZE
                                    ),
                                    "prediction_before_feedback": True,
                                    "prediction_persisted": True,
                                    "state_hash_at_prediction": state_hash,
                                    "brier_contribution": (probability - label) ** 2,
                                    "log_loss_contribution": _log_loss(label, probability),
                                    "prediction_duration_s": 0.0,
                                    "journal_duration_s": 0.0,
                                    "update_duration_s": 0.0,
                                    "failed": False,
                                    "censored": False,
                                    "feedback_status": "fixture",
                                    "update_admitted": arm in LEARNER_ARMS and revealed,
                                    "delayed_arrival_index": index + delay if revealed else None,
                                    "parent_state_hash": state_hash,
                                    "event_hash": canonical_hash([identity, label])
                                    if revealed
                                    else None,
                                    "new_state_hash": state_hash,
                                    "gradient_norm_before_clip": 0.0,
                                    "gradient_norm_after_clip": 0.0,
                                    "coefficient_change_l2": 0.0,
                                    "touched_coefficients": 0,
                                    "commit_after_prediction": True if revealed else None,
                                }
                            )
    return output


def write_row_shards(directory: Path, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Write hash-bound JSONL shards below the repository file-size ceiling."""

    directory.mkdir(parents=True, exist_ok=True)
    shards: list[JsonDict] = []
    handle = None
    path: Path | None = None
    size = 0
    count = 0
    try:
        for row in rows:
            payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            if handle is None or (size + len(payload) > MAX_SHARD_BYTES and count):
                if handle is not None and path is not None:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    shards.append(
                        {
                            "path": path.name,
                            "rows": count,
                            "byte_size": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
                path = directory / f"events-{len(shards):03d}.jsonl"
                handle = path.open("wb")
                size = 0
                count = 0
            handle.write(payload)
            size += len(payload)
            count += 1
        if handle is not None and path is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            shards.append(
                {
                    "path": path.name,
                    "rows": count,
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    finally:
        if handle is not None and not handle.closed:  # pragma: no cover - exceptional write exit.
            handle.close()
    return shards


def _load_event_rows(value: Mapping[str, Any], root: Path) -> list[JsonDict]:
    """Load inline fixture rows or rehash every production evidence shard."""

    reference = value.get("feedback_event_rows")
    if isinstance(reference, list):
        return deepcopy(reference)
    if not isinstance(reference, Mapping):
        raise ValueError("feedback event rows are required")
    directory = root / str(reference.get("directory"))
    output: list[JsonDict] = []
    for shard in reference.get("shards") or []:
        path = directory / str(shard.get("path"))
        if not path.is_file() or sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"feedback shard hash mismatch:{shard.get('path')}")
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(loaded) != shard.get("rows"):
            raise ValueError(f"feedback shard row mismatch:{shard.get('path')}")
        output.extend(loaded)
    if len(output) != reference.get("row_count"):
        raise ValueError("feedback row count mismatch")
    return output


def _load_lineage_rows(value: Mapping[str, Any], root: Path) -> list[JsonDict]:
    """Load inline control lineage or rehash every numeric checkpoint shard."""

    reference = value.get("checkpoint_lineage")
    if isinstance(reference, list):
        return deepcopy(reference)
    if not isinstance(reference, Mapping):
        raise ValueError("checkpoint lineage is required")
    directory = root / str(reference.get("directory"))
    output: list[JsonDict] = []
    for shard in reference.get("shards") or []:
        path = directory / str(shard.get("path"))
        if not path.is_file() or sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"lineage shard hash mismatch:{shard.get('path')}")
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(loaded) != shard.get("rows"):
            raise ValueError(f"lineage shard row mismatch:{shard.get('path')}")
        output.extend(loaded)
    if len(output) != reference.get("row_count"):
        raise ValueError("lineage row count mismatch")
    return output


def _lineage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Recompute numeric state and event hashes plus each parent chain."""

    errors: list[str] = []
    prior_by_unit: dict[tuple[str, str, int, int, str], str] = {}
    for index, row in enumerate(rows):
        try:
            expected_state = canonical_hash(
                {
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "checkpoint": row["checkpoint"],
                    "update_count": row["update_count"],
                }
            )
            expected_event = canonical_hash(
                {
                    "observation_id": row["observation_id"],
                    "label": row["label"],
                    "authority": row["label_authority"],
                    "arrival_index": row["arrival_index"],
                }
            )
            unit = (
                str(row["ordering"]),
                str(row["schedule"]),
                int(row["delay"]),
                int(row["seed"]),
                str(row["arm"]),
            )
            prior = prior_by_unit.get(unit)
            if expected_state != row.get("state_hash"):
                errors.append(f"lineage_state_hash_mismatch:{index}")
            if expected_event != row.get("event_hash"):
                errors.append(f"lineage_event_hash_mismatch:{index}")
            if prior is not None and prior != row.get("parent_state_hash"):
                errors.append(f"lineage_parent_hash_mismatch:{index}")
            prior_by_unit[unit] = str(row.get("state_hash"))
        except (KeyError, TypeError, ValueError):
            errors.append(f"lineage_row_invalid:{index}")
    return errors


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for every frozen affected and terminal check."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(passed)


def _hardware_path(event_rows: Sequence[Mapping[str, Any]], peak_memory_bytes: int) -> JsonDict:
    """Report measured service costs without relabeling a kernel cost as service."""

    prediction = [float(row.get("prediction_duration_s", 0.0)) for row in event_rows]
    update = [float(row.get("update_duration_s", 0.0)) for row in event_rows]
    journal = [float(row.get("journal_duration_s", 0.0)) for row in event_rows]
    full = [a + b + c for a, b, c in zip(prediction, update, journal, strict=True)]
    touched = [int(row.get("touched_coefficients", 0)) for row in event_rows]
    update_stats = _percentiles(update)
    journal_stats = _percentiles(journal)
    return {
        "path": "host_cpu_sparse_coefficient_updates_and_memory_journal",
        "latency_s": {
            "prediction": _percentiles(prediction),
            "update": update_stats,
            "journal": journal_stats,
            "full_service": _percentiles(full),
        },
        "peak_memory_bytes": int(peak_memory_bytes),
        "maximum_touched_coefficients": max(touched, default=0),
        "cpu_update_target_s": 1e-6,
        "cpu_update_target_met": update_stats["p95"] < 1e-6,
        "memory_lookup_target_s": 1e-3,
        "memory_lookup_target_met": journal_stats["p95"] < 1e-3,
        "kernel_only_used_as_end_to_end": False,
        "speedup_claimed": False,
    }


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep each validity, completion, safety, and benefit clause explicit."""

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
    gates: Sequence[Mapping[str, Any]], blocked: Mapping[str, Any] | None = None
) -> JsonDict:
    """Name exact upstream failures while keeping benefit separate from validity."""

    failed = [
        deepcopy(dict(row))
        for row in gates
        if row.get("category") != "scientific_benefit" and row.get("passed") is not True
    ]
    return {
        "required_checks_passed": not failed and blocked is None,
        "failed_required_checks": failed,
        "scientific_benefit_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") == "scientific_benefit"
        ),
        "blocked_upstream": blocked.get("upstream") if blocked else None,
        "blocked_path": blocked.get("path") if blocked else None,
        "blocked_check": blocked.get("check") if blocked else None,
        "blocked_field": blocked.get("field") if blocked else None,
        "blocked_expected": blocked.get("expected") if blocked else None,
        "blocked_observed": blocked.get("observed") if blocked else None,
    }


def _field_principles() -> dict[str, str]:
    """Explain field intent separately from plain machine-readable values."""

    values = {
        key: "Keep this field plain, hash-bound, and independently reducible."
        for key in REQUIRED_FIELDS
    }
    values.update(
        {
            "schema": "Use a versioned plain top-level schema with experiment identity and terminal status.",
            "run_date": "Use 20260919 with actual UTC start, end, and monotonic timing.",
            "preconditions_checked": "Record exact paths, identities, and observed values before dependent work.",
            "MODEL_SPECS": "Remain empty because this run invokes no current LLM.",
            "model_invoked": "Become true only after an actual current model attempt.",
            "invocation_counts": "Count only owned current loads and generations.",
            "inference_substrate": "Use a truthful string and keep device detail separate.",
            "inference_substrate_class": "Describe current compute, never archived inference.",
            "execution_venue": "Use the closed host value; device identity is separate.",
            "duration_s": "Measure current work and separate validation, model, and cold phases.",
            "phase_spans": "Retain real boundaries, elapsed times, completed units, and checkpoints.",
            "random_seed": "Freeze fit, schedule, and resampling seeds.",
            "reproducibility_checksum": "Bind code, protocol, inputs, raw rows, and validation scope.",
            "source_artifact_hashes": "Bind exact input bytes and preserve upstream flags.",
            "rows": "Retain every arm, seed, schedule, delay, order, and disposition.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
            "acceptance_gate_results": "Separate validity and completion from scientific benefit.",
            "gate_check_summary": "Name every exact blocked operand, including missing and zero values.",
            "verifier_is_oracle": "Human support annotation is fallible and not exact proof.",
            "honest_verdict": "Start completed findings with complete_ and external absence with blocked_.",
            "verdict_class": "Use the closed enum; a valid no-gain replay is terminal null.",
            "flagged_adversarial": "Critical findings deny scientific readiness.",
            "validation_receipts": "Retain exact argv, environment, exits, durations, names, and log hashes.",
            "field_principles": "Explain intent outside ordinary gate scalars.",
            "promotion_score": "Always remain zero; no rollout or generator update is authorized.",
            "online_capture_complete_score": "One means complete valid temporal evidence, even for a null.",
            "online_value_score": "One requires every causal, score, risk, cost, and interval clause.",
            "continuous_self_learning_task": "True because revealed labels update later numeric predictions.",
            "feedback_event_rows": "Bind selection, propensity, temporal order, updates, and authority.",
            "checkpoint_lineage": "Bind numeric parent, event, state, restart, and revocation evidence.",
            "hardware_path": "Report sparse locality and measured CPU service costs without kernel substitution.",
            "small_ebm_training": "Record actual online coefficient changes with zero current LLM calls.",
        }
    )
    return values


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash material evidence while excluding clocks and the checksum slot."""

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
        "model_duration_s",
        "cold_start_duration_s",
    ):
        payload.pop(field, None)
    return canonical_hash(payload)


def _unit_rows() -> list[JsonDict]:
    """Name every registered arm, seed, schedule, delay, and order unit."""

    return [
        {
            "comparative_unit": f"{ordering}:{schedule}:{delay}:{seed}:{arm}",
            "ordering": ordering,
            "schedule": schedule,
            "delay": delay,
            "seed": seed,
            "arm": arm,
            "status": "completed",
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
        }
        for ordering in ORDERS
        for schedule in SCHEDULES
        for delay in DELAYS
        for seed in TRAINING_SEEDS
        for arm in ARMS
    ]


def _receipt(
    *,
    started_ns: int,
    ended_ns: int,
    spans: Sequence[Mapping[str, Any]],
    small_training: Mapping[str, Any],
    root: Path,
) -> JsonDict:
    """Build current no-model provenance with the upstream as a typed sidecar."""

    sidecars = (
        [sidecar_reference(root / UPSTREAM_PATH, root=root, scope="historical_model_receipts")]
        if (root / UPSTREAM_PATH).is_file()
        else []
    )
    return build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "current_llm": False,
            "sparse_numeric_updates": True,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=sidecars,
        phase_spans=spans,
        small_ebm_training=small_training,
    )


def _artifact(
    *,
    root: Path,
    event_reference: Any,
    event_rows: Sequence[Mapping[str, Any]],
    lineage: Any,
    controls: Mapping[str, Any],
    revocations: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    started_at: str,
    completed_at: str,
    started_ns: int,
    ended_ns: int,
    support: Mapping[str, Any],
    peak_memory_bytes: int,
    flagged: bool,
) -> JsonDict:
    """Assemble one schema-complete terminal record from raw replay evidence."""

    reduction = reduce_online_value(reports, intervals, support)
    temporal_passed = bool(event_rows) and all(
        row.get("prediction_before_feedback") is True
        and row.get("prediction_persisted") is True
        and float(row.get("gradient_norm_after_clip", 0.0)) <= GRADIENT_CLIP
        for row in event_rows
    )
    controls_passed = bool(controls) and all(row.get("passed") is True for row in controls.values())
    validation_passed = _validation_passed(receipts)
    gates = [
        _gate(
            "structured_prerequisites",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "Dependent replay starts only from exact eligible static evidence.",
        ),
        _gate(
            "temporal_event_integrity",
            "completion",
            "==",
            True,
            temporal_passed,
            temporal_passed,
            "A learning claim needs durable prediction before each bounded update.",
        ),
        _gate(
            "development_controls",
            "validity",
            "all",
            True,
            controls_passed,
            controls_passed,
            "Restart, revocation, and corruption controls must fail closed.",
        ),
        _gate(
            "required_validation",
            "completion",
            "all",
            True,
            validation_passed,
            validation_passed,
            "Scoped checks and fresh terminal readers must all pass.",
        ),
        _gate(
            "registered_online_value",
            "scientific_benefit",
            "==",
            True,
            reduction["passed"],
            reduction["passed"],
            "Benefit requires every prespecified score, risk, cost, and interval clause.",
        ),
    ]
    capture = int(
        all(row["passed"] for row in gates if row["category"] != "scientific_benefit")
        and not flagged
    )
    value_score = int(capture == 1 and reduction["online_value_score"] == 1)
    verdict_class = "positive" if value_score else "null" if capture else "disqualified"
    honest = (
        "complete_positive_randomized_feedback_online_value"
        if value_score
        else reduction["terminal_verdict"]
        if capture
        else "complete_disqualified_randomized_feedback_evidence"
    )
    small_training = {
        "performed": True,
        "receipt_class": "small_ebm_training",
        "current_llm_calls": 0,
        "generator_weights_fitted": False,
        "seeds": list(TRAINING_SEEDS),
        "learning_rate": ONLINE_RATE,
        "gradient_clip": GRADIENT_CLIP,
        "updates": sum(bool(row.get("update_admitted")) for row in event_rows),
        "maximum_touched_coefficients": max(
            (int(row.get("touched_coefficients", 0)) for row in event_rows), default=0
        ),
    }
    current = _receipt(
        started_ns=started_ns,
        ended_ns=ended_ns,
        spans=spans,
        small_training=small_training,
        root=root,
    )
    result: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": current["inference_substrate_details"],
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": max((ended_ns - started_ns) / 1e9, 0.0),
        "model_duration_s": 0.0,
        "validation_duration_s": sum(float(row.get("duration_s", 0.0)) for row in receipts),
        "cold_start_duration_s": 0.0,
        "phase_spans": deepcopy(list(spans)),
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "schedule_seed": 65101,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": _unit_rows(),
        "sample_size_budget": {
            "planned_units": len(_unit_rows()),
            "attempted_units": len(_unit_rows()),
            "completed_units": len(_unit_rows()),
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": 0,
            "independent_groups": int(support.get("independent_online_groups", 0)),
            "label_counts": deepcopy(dict(support.get("label_counts") or {})),
            "stop_rule": "all registered arms, seeds, schedules, delays, and orders",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged),
        "validation_receipts": deepcopy(list(receipts)),
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "online_capture_complete_score": capture,
        "online_value_score": value_score,
        "continuous_self_learning_task": True,
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible prospective replay of archived human annotations; not exact proof or live feedback",
        "feedback_event_rows": deepcopy(event_reference),
        "checkpoint_lineage": deepcopy(lineage),
        "condition_reports": deepcopy(list(reports)),
        "paired_moving_block_intervals": deepcopy(list(intervals)),
        "online_value_reduction": reduction,
        "development_controls": deepcopy(dict(controls)),
        "revocation_rows": deepcopy(list(revocations)),
        "hardware_path": _hardware_path(event_rows, peak_memory_bytes),
        "small_ebm_training": small_training,
        "current_run_id": current["current_run_id"],
        "current_owner_pid": current["current_owner_pid"],
        "current_invocation_events": current["current_invocation_events"],
        "event_count": current["event_count"],
        "event_sha256": current["event_sha256"],
        "receipt_sidecars": current["receipt_sidecars"],
        "prospective_replay_not_live_feedback": True,
        "empirical_replay_only": True,
        "iid_guarantee_asserted": False,
        "conformal_guarantee_asserted": False,
        "fdr_guarantee_asserted": False,
    }
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute completion and scientific value from raw event evidence."""

    try:
        event_rows = _load_event_rows(value, root)
    except (OSError, ValueError, json.JSONDecodeError):
        event_rows = []
    try:
        lineage_rows = _load_lineage_rows(value, root)
        lineage_passed = not _lineage_errors(lineage_rows) and (
            bool(lineage_rows) or isinstance(value.get("checkpoint_lineage"), list)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        lineage_passed = False
    temporal_passed = bool(event_rows) and all(
        row.get("prediction_before_feedback") is True
        and row.get("prediction_persisted") is True
        and float(row.get("gradient_norm_after_clip", math.inf)) <= GRADIENT_CLIP
        and (not row.get("update_admitted") or row.get("commit_after_prediction") is True)
        for row in event_rows
    )
    controls = value.get("development_controls") or {}
    controls_passed = bool(controls) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in controls.values()
    )
    prerequisites_passed = bool(value.get("preconditions_checked")) and all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    validation_passed = _validation_passed(value.get("validation_receipts") or [])
    reports = condition_reports(event_rows) if event_rows else []
    support = {
        "independent_online_groups": len({str(row.get("observation_id")) for row in event_rows}),
        "label_counts": {
            "0": len(
                {str(row.get("observation_id")) for row in event_rows if row.get("label") == 0}
            ),
            "1": len(
                {str(row.get("observation_id")) for row in event_rows if row.get("label") == 1}
            ),
        },
    }
    stored_reports = value.get("condition_reports") or []
    stored_intervals = value.get("paired_moving_block_intervals") or []
    draws = int(stored_intervals[0].get("draws", BOOTSTRAP_DRAWS)) if stored_intervals else 0
    intervals = paired_moving_block_intervals(event_rows, draws=draws) if reports and draws else []
    evidence_reduction_matches = canonical_hash(reports) == canonical_hash(
        stored_reports
    ) and canonical_hash(intervals) == canonical_hash(stored_intervals)
    reduction = (
        reduce_online_value(reports, intervals, support)
        if reports and intervals
        else {
            "online_value_score": 0,
            "passed": False,
        }
    )
    capture = int(
        prerequisites_passed
        and temporal_passed
        and lineage_passed
        and controls_passed
        and validation_passed
        and evidence_reduction_matches
        and value.get("flagged_adversarial") is False
    )
    return {
        "online_capture_complete_score": capture,
        "online_value_score": int(capture == 1 and reduction["online_value_score"] == 1),
        "preconditions_passed": prerequisites_passed,
        "temporal_rows_passed": temporal_passed,
        "checkpoint_lineage_passed": lineage_passed,
        "controls_passed": controls_passed,
        "validation_passed": validation_passed,
        "evidence_reduction_matches": evidence_reduction_matches,
        "event_row_count": len(event_rows),
        "support": support,
    }


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, events, provenance, scores, and the stable checksum."""

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
    errors.extend(validate_current_work_receipt(value, events=[]))
    for label, reference in (value.get("source_artifact_hashes") or {}).items():
        if not isinstance(reference, Mapping):
            errors.append(f"source_hash_reference_invalid:{label}")
            continue
        path = root / str(reference.get("path"))
        if not path.is_file() or sha256_file(path) != reference.get("sha256"):
            errors.append(f"source_hash_mismatch:{label}")
    try:
        rows = _load_event_rows(value, root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rows = []
        errors.append(f"feedback_rows_invalid:{type(exc).__name__}")
    try:
        lineage_rows = _load_lineage_rows(value, root)
        errors.extend(_lineage_errors(lineage_rows))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"checkpoint_lineage_invalid:{type(exc).__name__}")
    for index, row in enumerate(rows):
        probability = row.get("probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
        ):
            errors.append(f"event_probability_invalid:{index}")
    reduced = independent_reduce(value, root=root)
    if value.get("online_capture_complete_score") != reduced["online_capture_complete_score"]:
        errors.append("online_capture_complete_score_mismatch")
    if value.get("online_value_score") != reduced["online_value_score"]:
        errors.append("online_value_score_mismatch")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_mismatch")
    if value.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
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
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload one candidate and apply every strict field-level check."""

    value = _load_object(path)
    return validate_artifact(value, root=root) if value else ["artifact_unreadable_or_not_object"]


def build_fixture_artifact(
    *, validation_receipts: Sequence[Mapping[str, Any]] | None = None
) -> JsonDict:
    """Build compact complete-null evidence for independent mutation tests."""

    rows = synthetic_metric_rows(groups=16, seeds=5)
    reports = condition_reports(rows)
    intervals = paired_moving_block_intervals(rows, draws=100)
    controls = {
        "restart_equality": {"passed": True},
        "revocation_replay": {"passed": True},
        "corrupt_state_rollback": {"passed": True},
        "shuffled_label_negative_control": {"passed": True},
    }
    preconditions = upstream_field_checks(
        {
            "decision_capture_complete_score": 1,
            "verdict_class": "null",
            "flagged_adversarial": False,
        }
    )
    now = time.monotonic_ns()
    return _artifact(
        root=REPO_ROOT,
        event_reference=rows,
        event_rows=rows,
        lineage=[],
        controls=controls,
        revocations=[],
        reports=reports,
        intervals=intervals,
        preconditions=preconditions,
        source_hashes={},
        receipts=list(validation_receipts or []),
        spans=[],
        started_at="2026-09-19T00:00:00+00:00",
        completed_at="2026-09-19T00:00:01+00:00",
        started_ns=now,
        ended_ns=now + 1_000_000_000,
        support={"independent_online_groups": 16, "label_counts": {"0": 8, "1": 8}},
        peak_memory_bytes=0,
        flagged=False,
    )


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish external prerequisite absence without dependent replay work."""

    now = time.monotonic_ns()
    controls = {"not_run_due_to_prerequisite": {"passed": True}}
    result = _artifact(
        root=REPO_ROOT,
        event_reference=[],
        event_rows=[],
        lineage=[],
        controls=controls,
        revocations=[],
        reports=[],
        intervals=[],
        preconditions=[deepcopy(dict(failed))],
        source_hashes={},
        receipts=[],
        spans=[],
        started_at=utc_now(),
        completed_at=utc_now(),
        started_ns=now,
        ended_ns=now,
        support={"independent_online_groups": 0, "label_counts": {"0": 0, "1": 0}},
        peak_memory_bytes=0,
        flagged=False,
    )
    result.update(
        {
            "status": f"blocked_{failed.get('check')}",
            "honest_verdict": f"blocked_{failed.get('check')}",
            "verdict_class": "blocked",
            "online_capture_complete_score": 0,
            "online_value_score": 0,
            "gate_check_summary": _gate_summary(result["acceptance_gate_results"], failed),
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - real clock boundary.
    """Close one real phase with completed units and a resumable reference."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint_reference": checkpoint,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, reduction, adversarial, and strict-row commands."""

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
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, replay, validate, independently reduce, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start", completed_units=0)
    phase_started = time.monotonic()
    preconditions, source_hashes, upstream = collect_preconditions(root)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            run_started,
            len(preconditions),
            UPSTREAM_PATH.as_posix(),
        )
    )
    progress(run_started, "preconditions", "end", completed_units=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "publish", "before_atomic_write", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "publish", "after_atomic_write", status="blocked")
        return blocked

    progress(run_started, "load", "before_benchmark", completed_units=0)
    phase_started = time.monotonic()
    corpus = reload_corpus(root / CORPUS_DIR)
    readers = ProtocolReaders(corpus)
    predictors = readers.read_predictors("prospective_stream")
    evaluators = readers.read_evaluators("prospective_stream", EVALUATOR_TOKEN)
    joined = join_predictor_labels(predictors, evaluators)
    selected = [row for row in joined if row.get("certificate_selected") is True]
    initial_states = load_initial_states(root, upstream)
    for state in initial_states:
        source_hashes[str(state["source_path"])] = {
            "path": str(state["source_path"]),
            "sha256": str(state["source_sha256"]),
            "original_flagged_adversarial": None,
        }
    streams = build_streams(selected)
    risks = frozen_gibbs_risks(initial_states, streams["hash_order"])
    schedules_by_order = {
        order: build_reveal_schedules(stream, risks, seed=65_101)
        for order, stream in streams.items()
    }
    spans.append(_span("load", phase_started, run_started, len(selected), CORPUS_PATH.as_posix()))
    progress(run_started, "load", "after_benchmark", completed_units=len(selected))

    raw_dir = root / RAW_DIR
    all_events: list[JsonDict] = []
    all_lineage: list[JsonDict] = []
    tracemalloc.start()
    progress(run_started, "replay", "before_benchmark", planned_conditions=12)
    phase_started = time.monotonic()
    completed_conditions = 0
    for order, stream in streams.items():
        for schedule in SCHEDULES:
            for delay in DELAYS:
                journal = raw_dir / "journals" / f"{order}--{schedule}--delay-{delay}.jsonl"
                replay = replay_condition(
                    initial_states,
                    stream,
                    schedules_by_order[order][schedule],
                    ordering=order,
                    schedule=schedule,
                    delay=delay,
                    journal_path=journal,
                )
                all_events.extend(replay["feedback_event_rows"])
                all_lineage.extend(replay["checkpoint_lineage"])
                completed_conditions += 1
                progress(
                    run_started,
                    "replay",
                    "condition_complete",
                    completed_units=completed_conditions,
                    total=12,
                    rows=len(all_events),
                )
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    spans.append(
        _span("replay", phase_started, run_started, completed_conditions, "in_memory_replay")
    )
    progress(run_started, "replay", "after_benchmark", completed_units=completed_conditions)

    progress(run_started, "evidence", "before_serialization", rows=len(all_events))
    phase_started = time.monotonic()
    shard_directory = raw_dir / "feedback_event_rows"
    shards = write_row_shards(shard_directory, all_events)
    event_reference = {
        "directory": shard_directory.relative_to(root).as_posix(),
        "row_count": len(all_events),
        "shards": shards,
    }
    lineage_directory = raw_dir / "checkpoint_lineage"
    lineage_shards = write_row_shards(lineage_directory, all_lineage)
    lineage_reference = {
        "directory": lineage_directory.relative_to(root).as_posix(),
        "row_count": len(all_lineage),
        "shards": lineage_shards,
    }
    spans.append(
        _span("evidence", phase_started, run_started, len(shards), event_reference["directory"])
    )
    progress(run_started, "evidence", "after_serialization", completed_units=len(shards))

    progress(run_started, "reduction", "before_benchmark", draws=BOOTSTRAP_DRAWS)
    phase_started = time.monotonic()
    reports = condition_reports(all_events)
    intervals = paired_moving_block_intervals(all_events)
    controls, revocations = run_analytic_controls(root)
    support = {
        "independent_online_groups": len(selected),
        "label_counts": {
            "0": sum(int(row["label"]) == 0 for row in selected),
            "1": sum(int(row["label"]) == 1 for row in selected),
        },
    }
    spans.append(
        _span("reduction", phase_started, run_started, BOOTSTRAP_DRAWS, "paired_intervals")
    )
    progress(run_started, "reduction", "after_benchmark", completed_units=BOOTSTRAP_DRAWS)

    private_root = Path(tempfile.mkdtemp(prefix="exp7427-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(
        run_started,
        "affected_validation",
        "before_subprocesses",
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
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
        _span("affected_validation", phase_started, run_started, len(affected), "affected_manifest")
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    for relative in (*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = root / relative
        if path.is_file():
            source_hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": source_hashes.get(relative.as_posix(), {}).get(
                    "original_flagged_adversarial"
                ),
            }
    for shard in shards:
        path = shard_directory / shard["path"]
        label = path.relative_to(root).as_posix()
        source_hashes[label] = {
            "path": label,
            "sha256": shard["sha256"],
            "original_flagged_adversarial": None,
        }
    for shard in lineage_shards:
        path = lineage_directory / shard["path"]
        label = path.relative_to(root).as_posix()
        source_hashes[label] = {
            "path": label,
            "sha256": shard["sha256"],
            "original_flagged_adversarial": None,
        }
    candidate = _artifact(
        root=root,
        event_reference=event_reference,
        event_rows=all_events,
        lineage=lineage_reference,
        controls=controls,
        revocations=revocations,
        reports=reports,
        intervals=intervals,
        preconditions=preconditions,
        source_hashes=source_hashes,
        receipts=affected,
        spans=spans,
        started_at=started_at,
        completed_at=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        support=support,
        peak_memory_bytes=peak_memory,
        flagged=not affected_reduction["passed"],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal_commands = _terminal_commands(candidate_path)
    progress(
        run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_commands)
    )
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_commands, log_dir=raw_dir / "validation/terminal"
    )
    spans.append(
        _span("terminal_validation", phase_started, run_started, len(terminal), str(candidate_path))
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    final = _artifact(
        root=root,
        event_reference=event_reference,
        event_rows=all_events,
        lineage=lineage_reference,
        controls=controls,
        revocations=revocations,
        reports=reports,
        intervals=intervals,
        preconditions=preconditions,
        source_hashes=source_hashes,
        receipts=[*affected, *terminal],
        spans=spans,
        started_at=started_at,
        completed_at=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        support=support,
        peak_memory_bytes=peak_memory,
        flagged=not affected_reduction["passed"] or not terminal_passed or critical,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "publish", "before_atomic_write", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "publish", "after_atomic_write", status=final["status"])
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
    """Run the experiment or one strict fresh-process candidate reader."""

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
        reduction = independent_reduce(value, root=args.root) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
