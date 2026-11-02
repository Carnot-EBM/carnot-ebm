"""Measure fixed selected-feedback adaptation on sealed source features.

The experiment updates only two affine calibration scalars after delayed
machine labels arrive. It does not update the Gibbs representation or claim
that a machine annotation is verified truth.

Spec refs: REQ-AUTO-7414 and SCENARIO-AUTO-7414-01 through 08.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
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
from carnot.experiment_7385_v648_decision_training import fit_affine_transform
from carnot.experiment_7410_v650_source_corpus import CorpusReaders, reload_corpus
from carnot.experiment_7412_v650_source_features import (
    SOURCE_FEATURE_NAMES,
    fit_gibbs_head,
    fit_logistic_control,
    gibbs_energy,
    probability_from_energy,
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
MILESTONE = "2026.09.650"
EXPERIMENT_ID = "exp7414-selected-feedback"
SCHEMA = "carnot.exp7414.v650.selected_feedback.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7414_v650_selected_feedback.json")
RAW_DIR = Path("results/raw/experiment_7414_v650_selected_feedback")
MODULE_PATH = Path("python/carnot/experiment_7414_v650_selected_feedback.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7414_v650_selected_feedback.py")
TEST_PATH = Path("tests/python/test_experiment_7414_v650_selected_feedback.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
UPSTREAM_PATH = Path("results/experiment_7412_v650_source_features.json")
PROTOCOL_PATH = Path("results/raw/experiment_7412_v650_source_features/protocol_manifest.json")
FEATURE_PATH = Path("results/raw/experiment_7412_v650_source_features/source_feature_rows.json")
CORPUS_PATH = Path("results/raw/experiment_7410_v650_source_corpus/corpus_manifest.json")
CORPUS_DIR = CORPUS_PATH.parent

EXPECTED_HASHES = {
    UPSTREAM_PATH: "sha256:b6ff5100e270646ad98961513956bcdc07073668e034f764d1bd64d20fb17874",
    PROTOCOL_PATH: "sha256:cef859226f0dc2f135992ecc594416fc93634e2ffbea44249b0470aa58175bbf",
    FEATURE_PATH: "sha256:a34e26beee7f797bb68fe69af1be00420c1c2cc3bbb1b203ebc89dd59df36864",
    CORPUS_PATH: "sha256:be0f0b29d6216eaf9a2c1a8ddefd98a5761256c2c9a091038baaf901ab4c277a",
}
EXPECTED_PROTOCOL_HASH = "sha256:65a9a5b817f4cfbef1524efa809bc1203402dfeffd5e400b93dabe2f249407af"
EXPECTED_CORPUS_HASH = "sha256:2843bdf316cb6e75f0fcd878704ca3ac937eaa43b2812ef2b73bea3106f7b5d1"

FROZEN_ARM = "frozen_calibrated_gibbs"
ADAPTIVE_ARM = "projected_adaptive_affine_gibbs"
LOGISTIC_ARM = "online_l2_logistic"
FREQUENCY_ARM = "recent_frequency_beta_1_1_window_128"
NO_FEEDBACK_ARM = "no_feedback_adaptive_copy"
ARMS = (FROZEN_ARM, ADAPTIVE_ARM, LOGISTIC_ARM, FREQUENCY_ARM, NO_FEEDBACK_ARM)
CONTROL_ARMS = (FROZEN_ARM, LOGISTIC_ARM, FREQUENCY_ARM)
ORDERINGS = ("hash_order", "reversed_block_order")
FEEDBACK_REGIMES = ("primary_label_blind_75", "frozen_baseline_selected")
DELAYS = (1, 32)
TRAINING_SEEDS = (65_001, 65_002, 65_003, 65_004, 65_005)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_501_407
BLOCK_LENGTHS = (16, 32)
RECENT_WINDOW = 128
ONLINE_RATE = 0.01
L2 = 0.001
A_MIN = 0.25
A_MAX = 4.0
B_MIN = -8.0
B_MAX = 8.0
GRADIENT_NORM_CAP = 1.0
LABEL_AUTHORITY = "machine_annotation"

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
    Path("python/carnot/learning/delayed_energy_calibration.py"),
    Path("python/carnot/experiment_7396_v649_decision_diagnosis.py"),
    Path("python/carnot/experiment_7397_v649_delayed_adapter.py"),
    Path("python/carnot/experiment_7399_v649_online_trial.py"),
    Path("python/carnot/experiment_7410_v650_source_corpus.py"),
    Path("python/carnot/experiment_7412_v650_source_features.py"),
    SPEC_PATH,
    UPSTREAM_PATH,
    PROTOCOL_PATH,
    FEATURE_PATH,
    CORPUS_PATH,
)

V650_MANIFEST = AffectedManifest(
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
    "continuous_self_learning_task",
    "online_capture_complete_score",
    "online_value_score",
    "feedback_event_rows",
    "revocation_rows",
    "condition_reports",
    "paired_moving_block_intervals",
    "hardware_path",
    "label_authority",
)


class FutureLabelAccessError(ValueError):
    """Reject a feedback event that has not reached its release index."""


def utc_now() -> str:  # pragma: no cover - wall-clock boundary.
    """Return one real UTC boundary for the run receipt."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print a flushed phase boundary and its monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7414] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Read a JSON object and treat missing or malformed bytes as absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
) -> JsonDict:
    """Name an exact prerequisite and preserve missing observed values."""

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


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate source bytes and the three structured Exp7412 gates."""

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
        if observed:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": False if relative == UPSTREAM_PATH else None,
            }

    upstream = _load_object(root / UPSTREAM_PATH)
    protocol = _load_object(root / PROTOCOL_PATH)
    features = _load_object(root / FEATURE_PATH)
    corpus = _load_object(root / CORPUS_PATH)
    loaded = {
        "upstream": upstream,
        "protocol": protocol,
        "features": features,
        "corpus": corpus,
    }
    for relative, expected_hash in EXPECTED_HASHES.items():
        path = root / relative
        checks.append(
            _precondition(
                f"exact_hash:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "sha256",
                "==",
                expected_hash,
                sha256_file(path) if path.is_file() else None,
            )
        )
    allowed = ["positive", "circular_positive", "null"]
    verdict = upstream.get("verdict_class")
    checks.extend(
        [
            _precondition(
                "upstream_protocol_ready",
                "exp7412-source-features",
                UPSTREAM_PATH.as_posix(),
                "source_feature_protocol_ready_score",
                "==",
                1,
                upstream.get("source_feature_protocol_ready_score"),
            ),
            _precondition(
                "upstream_verdict_allowed",
                "exp7412-source-features",
                UPSTREAM_PATH.as_posix(),
                "verdict_class",
                "in",
                allowed,
                verdict,
                verdict in allowed,
            ),
            _precondition(
                "upstream_unflagged",
                "exp7412-source-features",
                UPSTREAM_PATH.as_posix(),
                "flagged_adversarial",
                "==",
                False,
                upstream.get("flagged_adversarial"),
            ),
            _precondition(
                "protocol_manifest_hash",
                PROTOCOL_PATH.as_posix(),
                PROTOCOL_PATH.as_posix(),
                "manifest_hash",
                "==",
                EXPECTED_PROTOCOL_HASH,
                protocol.get("manifest_hash"),
            ),
            _precondition(
                "corpus_manifest_hash",
                CORPUS_PATH.as_posix(),
                CORPUS_PATH.as_posix(),
                "manifest_hash",
                "==",
                EXPECTED_CORPUS_HASH,
                corpus.get("manifest_hash"),
            ),
            _precondition(
                "feature_row_count",
                FEATURE_PATH.as_posix(),
                FEATURE_PATH.as_posix(),
                "record_count",
                "==",
                protocol.get("feature_row_count"),
                len(features.get("records") or []),
            ),
        ]
    )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-AUTO-7414",
            "REQ-AUTO-7414" if "REQ-AUTO-7414" in spec else None,
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
            "experiment_id: 7414" in exclusion,
        )
    )
    return checks, hashes, loaded


def _source_vector(row: Mapping[str, Any]) -> list[float]:
    """Return the six frozen source values in protocol order."""

    source = row.get("source_features")
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_FEATURE_NAMES):
        raise ValueError("source features do not match the frozen protocol")
    values = [float(source[name]) for name in SOURCE_FEATURE_NAMES]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("source features must be finite values in [0, 1]")
    return values


def _sigmoid(value: float) -> float:
    """Evaluate the registered stable scalar logistic function."""

    return probability_from_energy(float(value))


def _log_loss(label: int, probability: float) -> float:
    """Return one finite Bernoulli log-loss contribution."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def fixture_weights() -> JsonDict:
    """Return a small fixed 6-4-1 state for independent journal controls."""

    return {
        "w1": [
            [0.2, -0.1, 0.05, 0.1, -0.2, 0.3],
            [-0.1, 0.2, 0.1, -0.05, 0.15, -0.2],
            [0.05, 0.1, -0.2, 0.3, -0.1, 0.2],
            [0.1, -0.2, 0.3, 0.05, 0.2, -0.1],
        ],
        "b1": [0.0, 0.0, 0.0, 0.0],
        "w_out": [0.2, -0.1, 0.15, 0.05],
        "b_out": 0.0,
    }


class SourceAffineAdapter:
    """Journal delayed labels and rebuild two affine scalars after revocation."""

    def __init__(self, weights: Mapping[str, Any], *, a: float = 1.0, b: float = 0.0) -> None:
        gibbs_energy(weights, [0.0] * 6)
        self.weights = deepcopy(dict(weights))
        self.weights_hash = canonical_hash(self.weights)
        self.initial_a = self._a(a)
        self.initial_b = self._b(b)
        self.a = self.initial_a
        self.b = self.initial_b
        self.update_count = 0
        self.predictions: dict[str, JsonDict] = {}
        self.feedback: dict[str, JsonDict] = {}
        self.commit_order: list[str] = []

    @staticmethod
    def _a(value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or not A_MIN <= numeric <= A_MAX:
            raise ValueError("affine a is outside its projection bounds")
        return numeric

    @staticmethod
    def _b(value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or not B_MIN <= numeric <= B_MAX:
            raise ValueError("affine b is outside its projection bounds")
        return numeric

    @property
    def state_hash(self) -> str:
        """Hash only numeric prediction state, not future journal content."""

        return canonical_hash(
            {
                "a": self.a,
                "b": self.b,
                "update_count": self.update_count,
                "weights_hash": self.weights_hash,
            }
        )

    def record_prediction(
        self,
        event_id: str,
        energy: float,
        *,
        prediction_index: int,
        available_at: int,
    ) -> JsonDict:
        """Persist one prediction before its label can become visible."""

        value = float(energy)
        if not math.isfinite(value):
            raise ValueError("energy must be finite")
        if int(available_at) <= int(prediction_index):
            raise ValueError("feedback must become available after prediction")
        identity = str(event_id)
        if identity in self.predictions:
            raise ValueError("event prediction already exists")
        record = {
            "event_id": identity,
            "energy": value,
            "probability": _sigmoid(self.a * value + self.b),
            "prediction_index": int(prediction_index),
            "available_at": int(available_at),
            "state_hash": self.state_hash,
            "prediction_before_feedback": True,
        }
        self.predictions[identity] = deepcopy(record)
        return deepcopy(record)

    def _update(self, event_id: str, label: int) -> JsonDict:
        prediction = self.predictions[event_id]
        before = self.state_hash
        energy = float(prediction["energy"])
        probability = _sigmoid(self.a * energy + self.b)
        residual = probability - label
        raw_a = residual * energy
        raw_b = residual
        raw_norm = math.hypot(raw_a, raw_b)
        scale = min(1.0, GRADIENT_NORM_CAP / max(raw_norm, 1e-15))
        gradient_a = raw_a * scale
        gradient_b = raw_b * scale
        self.a = min(A_MAX, max(A_MIN, self.a - ONLINE_RATE * gradient_a))
        self.b = min(B_MAX, max(B_MIN, self.b - ONLINE_RATE * gradient_b))
        self.update_count += 1
        return {
            "state_hash_before": before,
            "state_hash_after": self.state_hash,
            "gradient_norm_before_clip": raw_norm,
            "gradient_norm_after_clip": math.hypot(gradient_a, gradient_b),
            "learning_rate": ONLINE_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "gibbs_weights_unchanged": True,
        }

    def commit_feedback(self, event_id: str, label: int | None, *, visible_at: int) -> JsonDict:
        """Commit one visible binary label and reject repeats or early access."""

        identity = str(event_id)
        prediction = self.predictions.get(identity)
        if prediction is None:
            return {"event_id": identity, "status": "unknown_event", "update_admitted": False}
        active = self.feedback.get(identity)
        if active is not None and active.get("active") is True:
            return {"event_id": identity, "status": "duplicate", "update_admitted": False}
        if int(visible_at) < int(prediction["available_at"]):
            raise FutureLabelAccessError("feedback is not available at this index")
        if label is None:
            return {"event_id": identity, "status": "missing", "update_admitted": False}
        if label not in (0, 1):
            raise ValueError("feedback label must be binary")
        update = self._update(identity, int(label))
        record = {
            "event_id": identity,
            "label": int(label),
            "visible_at": int(visible_at),
            "label_authority": LABEL_AUTHORITY,
            "status": "committed",
            "update_admitted": True,
            "active": True,
            "commit_sequence": len(self.commit_order),
            **update,
        }
        self.feedback[identity] = deepcopy(record)
        self.commit_order.append(identity)
        return deepcopy(record)

    def _reconstruct(self) -> None:
        """Replay active trusted commits from the sealed initial state."""

        self.a = self.initial_a
        self.b = self.initial_b
        self.update_count = 0
        for identity in self.commit_order:
            record = self.feedback[identity]
            if record.get("active") is True:
                update = self._update(identity, int(record["label"]))
                record.update(update)

    def revoke_feedback(self, event_id: str, replacement_label: int | None) -> JsonDict:
        """Replace or erase one label, then deterministically rebuild state."""

        identity = str(event_id)
        record = self.feedback.get(identity)
        if record is None or record.get("active") is not True:
            return {"event_id": identity, "status": "unknown_event", "update_admitted": False}
        if replacement_label not in (None, 0, 1):
            raise ValueError("replacement label must be binary or absent")
        started = time.perf_counter()
        if replacement_label is None:
            record["active"] = False
            status = "erased"
        else:
            record["label"] = int(replacement_label)
            status = "replaced"
        self._reconstruct()
        return {
            "event_id": identity,
            "status": status,
            "update_admitted": False,
            "state_hash_after": self.state_hash,
            "trusted_journal_replayed": True,
            "reconstruction_duration_s": time.perf_counter() - started,
        }

    def to_dict(self) -> JsonDict:
        """Serialize plain numeric state and its trusted journal."""

        return {
            "weights": deepcopy(self.weights),
            "weights_hash": self.weights_hash,
            "initial_a": self.initial_a,
            "initial_b": self.initial_b,
            "a": self.a,
            "b": self.b,
            "update_count": self.update_count,
            "predictions": [deepcopy(row) for row in self.predictions.values()],
            "feedback": [deepcopy(row) for row in self.feedback.values()],
            "commit_order": list(self.commit_order),
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceAffineAdapter:
        """Restore a checkpoint and reject changed numeric state or weights."""

        adapter = cls(
            value["weights"],
            a=float(value["initial_a"]),
            b=float(value["initial_b"]),
        )
        if adapter.weights_hash != value.get("weights_hash"):
            raise ValueError("Gibbs weights changed during restart")
        adapter.a = adapter._a(float(value["a"]))
        adapter.b = adapter._b(float(value["b"]))
        adapter.update_count = int(value["update_count"])
        adapter.predictions = {
            str(row["event_id"]): deepcopy(dict(row)) for row in value.get("predictions", [])
        }
        adapter.feedback = {
            str(row["event_id"]): deepcopy(dict(row)) for row in value.get("feedback", [])
        }
        adapter.commit_order = [str(item) for item in value.get("commit_order", [])]
        if adapter.state_hash != value.get("state_hash"):
            raise ValueError("affine state hash changed during restart")
        return adapter


def _partition(rows: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    """Select scored rows from one registered development partition."""

    return [row for row in rows if row.get("partition") == name and row.get("label") in {0, 1}]


def _gibbs_logit(weights: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    """Evaluate the frozen source Gibbs scalar before calibration."""

    return gibbs_energy(weights, _source_vector(row))


def _logistic_logit(weights: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    """Evaluate one six-feature linear logistic state."""

    return float(
        np.asarray(_source_vector(row)) @ np.asarray(weights["coef"], dtype=np.float64)
        + float(weights["bias"])
    )


def _policy_action(probability: float, policy: Mapping[str, Any]) -> str:
    """Apply one frozen typed policy with unsupported actions disabled."""

    if probability <= float(policy["accept_threshold"]) and policy.get("accept_enabled") is True:
        return "accept"
    if probability >= float(policy["reject_threshold"]) and policy.get("reject_enabled") is True:
        return "reject"
    return "escalate"


def _select_policy(
    weights: Mapping[str, Any], affine: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Choose one frozen threshold pair on policy groups only."""

    representatives: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = str(row["group_id"])
        current = representatives.get(group)
        if current is None or str(row["row_key"]) < str(current["row_key"]):
            representatives[group] = row
    candidates = []
    for accept in (0.01, 0.025, 0.05):
        for reject in (0.90, 0.95, 0.99):
            actions = []
            selected = list(representatives.values())
            for row in selected:
                probability = _sigmoid(
                    float(affine["slope"]) * _gibbs_logit(weights, row) + float(affine["intercept"])
                )
                actions.append(
                    "accept"
                    if probability <= accept
                    else "reject"
                    if probability >= reject
                    else "escalate"
                )
            accepts = [
                int(row["label"] == 1)
                for row, action in zip(selected, actions, strict=True)
                if action == "accept"
            ]
            rejects = [
                int(row["label"] == 0)
                for row, action in zip(selected, actions, strict=True)
                if action == "reject"
            ]
            accept_enabled = len(accepts) >= 10 and sum(accepts) / len(accepts) <= 0.05
            reject_enabled = len(rejects) >= 10 and sum(rejects) / len(rejects) <= 0.10
            enabled = [
                action
                if (action == "accept" and accept_enabled)
                or (action == "reject" and reject_enabled)
                else "escalate"
                for action in actions
            ]
            coverage = sum(action != "escalate" for action in enabled) / len(selected)
            utility = sum(
                (action == "accept" and row["label"] == 0)
                or (action == "reject" and row["label"] == 1)
                for row, action in zip(selected, enabled, strict=True)
            ) / len(selected)
            candidates.append(
                {
                    "accept_threshold": accept,
                    "reject_threshold": reject,
                    "accept_enabled": accept_enabled,
                    "reject_enabled": reject_enabled,
                    "coverage": coverage,
                    "utility": utility,
                }
            )
    chosen = max(
        candidates,
        key=lambda row: (
            float(row["coverage"]),
            float(row["utility"]),
            -float(row["accept_threshold"]),
            float(row["reject_threshold"]),
        ),
    )
    return {
        **chosen,
        "selection_partition": "policy_calibration",
        "representative_groups": len(representatives),
        "selection_rule": "coverage_then_utility_then_registered_order",
    }


def initialize_seed_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = TRAINING_SEEDS,
    steps: int = 500,
) -> list[JsonDict]:
    """Fit initial Gibbs and logistic states without opening online labels."""

    allowed = {"train", "probability_calibration", "policy_calibration", "online_stream"}
    if any(row.get("partition") not in allowed for row in rows):
        raise ValueError("registered development partitions are required")
    parts = {
        name: _partition(rows, name)
        for name in ("train", "probability_calibration", "policy_calibration")
    }
    if any(not part or {int(row["label"]) for row in part} != {0, 1} for part in parts.values()):
        raise ValueError("initial fitting partitions must contain both labels")
    train_x = np.asarray([_source_vector(row) for row in parts["train"]], dtype=np.float64)
    train_y = np.asarray([int(row["label"]) for row in parts["train"]], dtype=np.float64)
    states = []
    for seed in seeds:
        gibbs = fit_gibbs_head(train_x, train_y, seed=int(seed), steps=steps)
        logistic = fit_logistic_control(train_x, train_y, seed=int(seed), steps=steps)
        calibration_logits = [
            _gibbs_logit(gibbs["checkpoint"], row) for row in parts["probability_calibration"]
        ]
        calibration_labels = [int(row["label"]) for row in parts["probability_calibration"]]
        affine = fit_affine_transform(calibration_logits, calibration_labels, steps=steps)
        policy = _select_policy(gibbs["checkpoint"], affine, parts["policy_calibration"])
        states.append(
            {
                "seed": int(seed),
                "gibbs_weights": gibbs["checkpoint"],
                "affine": {"a": float(affine["slope"]), "b": float(affine["intercept"])},
                "logistic": deepcopy(logistic["checkpoint"]),
                "selected_policy": policy,
                "training_receipt": {
                    "fit_partition": "train",
                    "calibration_partition": "probability_calibration",
                    "policy_partition": "policy_calibration",
                    "online_labels_used_for_initialization": 0,
                    "steps": steps,
                    "architecture": [6, 4, 1],
                    "gibbs_input_hash": canonical_hash(
                        [(row["row_key"], row["group_id"], row["label"]) for row in parts["train"]]
                    ),
                    "generator_weights_fitted": False,
                    "receipt_class": "small_ebm_training",
                },
            }
        )
    return states


def build_streams(
    rows: Sequence[Mapping[str, Any]], *, block_length: int = 16
) -> dict[str, list[JsonDict]]:
    """Seal label-blind group representatives in two constructed orders."""

    representatives: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("partition") != "online_stream":
            continue
        group = str(row.get("group_id") or "")
        key = str(row.get("row_key") or "")
        if not group or not key:
            raise ValueError("online rows require group and row identities")
        current = representatives.get(group)
        if current is None or key < str(current["row_key"]):
            representatives[group] = row
    ordered = []
    for group, row in representatives.items():
        value = deepcopy(dict(row))
        value["observation_id"] = canonical_hash({"group_id": group, "row_key": row["row_key"]})
        ordered.append(value)
    ordered.sort(key=lambda row: str(row["observation_id"]))
    blocks = [
        ordered[index : index + block_length] for index in range(0, len(ordered), block_length)
    ]
    reversed_blocks = [item for block in reversed(blocks) for item in block]
    return {"hash_order": ordered, "reversed_block_order": reversed_blocks}


def primary_availability_mask(observation_ids: Sequence[str]) -> dict[str, bool]:
    """Select a deterministic label-blind 75-percent availability mask."""

    return {
        identity: int(canonical_hash(identity).split(":", 1)[1][:8], 16) % 4 != 0
        for identity in observation_ids
    }


def _logistic_probability(state: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    """Score one row with the current online logistic state."""

    return _sigmoid(_logistic_logit(state, row))


def _logistic_update(state: JsonDict, row: Mapping[str, Any], label: int) -> None:
    """Apply one bounded-cost online L2 logistic update."""

    vector = np.asarray(_source_vector(row), dtype=np.float64)
    coef = np.asarray(state["coef"], dtype=np.float64)
    probability = _sigmoid(float(vector @ coef + float(state["bias"])))
    residual = probability - label
    state["coef"] = (coef - ONLINE_RATE * (residual * vector + L2 * coef)).tolist()
    state["bias"] = float(state["bias"]) - ONLINE_RATE * residual


def _frequency_probability(history: Sequence[int]) -> float:
    """Return the recent-window Beta(1,1) posterior mean."""

    recent = history[-RECENT_WINDOW:]
    return (1 + sum(recent)) / (2 + len(recent))


def replay_condition(
    initial: Mapping[str, Any],
    stream: Sequence[Mapping[str, Any]],
    *,
    ordering: str,
    feedback_regime: str,
    delay: int,
    journal_path: Path | None = None,
) -> JsonDict:
    """Replay one paired condition with prediction before delayed feedback."""

    if ordering not in ORDERINGS or feedback_regime not in FEEDBACK_REGIMES or delay not in DELAYS:
        raise ValueError("registered replay condition is required")
    ids = [str(row["observation_id"]) for row in stream]
    primary_mask = primary_availability_mask(ids)
    weights = initial["gibbs_weights"]
    initial_a = float(initial["affine"]["a"])
    initial_b = float(initial["affine"]["b"])
    adaptive = SourceAffineAdapter(weights, a=initial_a, b=initial_b)
    logistic = deepcopy(dict(initial["logistic"]))
    frequency_history: list[int] = []
    event_rows: list[JsonDict] = []
    pending: list[JsonDict] = []
    by_event_arm: dict[tuple[str, str], JsonDict] = {}

    def reveal(visible_at: int) -> None:
        due = [item for item in pending if int(item["available_at"]) <= visible_at]
        for item in due:
            pending.remove(item)
            label = item["label"]
            admitted = bool(item["mask"] and label in {0, 1})
            for arm in ARMS:
                record = by_event_arm[(str(item["observation_id"]), arm)]
                started = time.perf_counter()
                prior = str(record["state_hash_at_prediction"])
                if not admitted:
                    status = "withheld" if label in {0, 1} else "missing"
                    new_hash = prior
                    update_admitted = False
                elif arm == ADAPTIVE_ARM:
                    result = adaptive.commit_feedback(
                        str(item["observation_id"]), int(label), visible_at=visible_at
                    )
                    status = str(result["status"])
                    new_hash = adaptive.state_hash
                    update_admitted = bool(result["update_admitted"])
                elif arm == LOGISTIC_ARM:
                    _logistic_update(logistic, item["row"], int(label))
                    status = "committed"
                    new_hash = canonical_hash(logistic)
                    update_admitted = True
                elif arm == FREQUENCY_ARM:
                    frequency_history.append(int(label))
                    if len(frequency_history) > RECENT_WINDOW:
                        del frequency_history[:-RECENT_WINDOW]
                    status = "committed"
                    new_hash = canonical_hash(frequency_history)
                    update_admitted = True
                else:
                    status = "frozen_no_update"
                    new_hash = prior
                    update_admitted = False
                record.update(
                    {
                        "reveal_visible_at": visible_at,
                        "feedback_status": status,
                        "update_admitted": update_admitted,
                        "prior_state_hash": prior,
                        "new_state_hash": new_hash,
                        "update_duration_s": time.perf_counter() - started,
                        "commit_after_prediction": True,
                    }
                )

    for index, row in enumerate(stream):
        identity = str(row["observation_id"])
        energy = _gibbs_logit(weights, row)
        frozen_probability = _sigmoid(initial_a * energy + initial_b)
        frozen_action = _policy_action(frozen_probability, initial["selected_policy"])
        mask = (
            primary_mask[identity]
            if feedback_regime == "primary_label_blind_75"
            else frozen_action == "escalate"
        )
        probabilities = {
            FROZEN_ARM: frozen_probability,
            ADAPTIVE_ARM: _sigmoid(adaptive.a * energy + adaptive.b),
            LOGISTIC_ARM: _logistic_probability(logistic, row),
            FREQUENCY_ARM: _frequency_probability(frequency_history),
            NO_FEEDBACK_ARM: frozen_probability,
        }
        adaptive.record_prediction(
            identity,
            energy,
            prediction_index=index,
            available_at=index + delay,
        )
        label = row.get("label")
        for arm in ARMS:
            started = time.perf_counter()
            probability = probabilities[arm]
            decision = _policy_action(probability, initial["selected_policy"])
            state_hash = (
                adaptive.state_hash
                if arm == ADAPTIVE_ARM
                else canonical_hash(logistic)
                if arm == LOGISTIC_ARM
                else canonical_hash(frequency_history)
                if arm == FREQUENCY_ARM
                else canonical_hash(
                    {"weights": weights, "a": initial_a, "b": initial_b, "updates": 0}
                )
            )
            record = {
                "observation_id": identity,
                "row_key": row["row_key"],
                "group_id": row["group_id"],
                "arm": arm,
                "seed": int(initial["seed"]),
                "ordering": ordering,
                "feedback_regime": feedback_regime,
                "delay": delay,
                "prediction_index": index,
                "available_at": index + delay,
                "probability": probability,
                "label": int(label) if label in {0, 1} else None,
                "label_authority": LABEL_AUTHORITY,
                "decision": decision,
                "registered_feedback_mask": mask,
                "diagnostic_learner_selected_mask": decision == "escalate",
                "prediction_before_feedback": True,
                "prediction_persisted": False,
                "state_hash_at_prediction": state_hash,
                "brier_contribution": (probability - int(label)) ** 2 if label in {0, 1} else None,
                "log_loss_contribution": _log_loss(int(label), probability)
                if label in {0, 1}
                else None,
                "prediction_duration_s": time.perf_counter() - started,
                "persistence_duration_s": 0.0,
            }
            event_rows.append(record)
            by_event_arm[(identity, arm)] = record
        if journal_path is not None:
            persistence_started = time.perf_counter()
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            staged = [by_event_arm[(identity, arm)] for arm in ARMS]
            with journal_path.open("a", encoding="utf-8") as stream_handle:
                stream_handle.write(json.dumps(staged, sort_keys=True) + "\n")
                stream_handle.flush()
                os.fsync(stream_handle.fileno())
            persistence_duration = time.perf_counter() - persistence_started
            for staged_row in staged:
                staged_row["prediction_persisted"] = True
                staged_row["persistence_duration_s"] = persistence_duration / len(ARMS)
        pending.append(
            {
                "observation_id": identity,
                "available_at": index + delay,
                "label": label,
                "mask": mask,
                "row": row,
            }
        )
        reveal(index)
    final_index = len(stream) + delay
    reveal(final_index)
    return {
        "feedback_event_rows": event_rows,
        "pending_feedback_at_end": len(pending),
        "frequency_history_at_end": len(frequency_history),
    }


def run_analytic_controls() -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Exercise early, duplicate, restart, replacement, and erased replay controls."""

    controls: dict[str, JsonDict] = {}
    adapter = SourceAffineAdapter(fixture_weights())
    adapter.record_prediction("a", 0.2, prediction_index=0, available_at=1)
    early = False
    try:
        adapter.commit_feedback("a", 1, visible_at=0)
    except FutureLabelAccessError:
        early = True
    committed = adapter.commit_feedback("a", 1, visible_at=1)
    before_duplicate = adapter.state_hash
    duplicate = adapter.commit_feedback("a", 1, visible_at=2)
    restored = SourceAffineAdapter.from_dict(adapter.to_dict())
    controls["future_label_access"] = {"passed": early}
    controls["duplicate_reveal"] = {
        "passed": duplicate["status"] == "duplicate" and adapter.state_hash == before_duplicate
    }
    controls["crash_restart"] = {"passed": restored.to_dict() == adapter.to_dict()}
    controls["prediction_before_update"] = {"passed": committed["update_admitted"] is True}

    replace_started = time.perf_counter()
    replacement = restored.revoke_feedback("a", replacement_label=0)
    replacement_duration = time.perf_counter() - replace_started
    replacement_hash = restored.state_hash
    restarted_replacement = SourceAffineAdapter.from_dict(restored.to_dict())
    erase_started = time.perf_counter()
    erased = restarted_replacement.revoke_feedback("a", replacement_label=None)
    erase_duration = time.perf_counter() - erase_started
    controls["label_replacement"] = {
        "passed": replacement["status"] == "replaced" and restarted_replacement.update_count == 0
    }
    controls["erased_update_replay"] = {
        "passed": erased["status"] == "erased" and restarted_replacement.update_count == 0
    }
    frozen = SourceAffineAdapter(fixture_weights())
    frozen_hash = frozen.state_hash
    frozen.record_prediction("withheld", -0.1, prediction_index=0, available_at=1)
    frozen.commit_feedback("withheld", None, visible_at=1)
    controls["no_feedback_equality"] = {"passed": frozen.state_hash == frozen_hash}
    revocations = [
        {
            "operation": "replace_label",
            "event_id": "a",
            "prior_label": 1,
            "replacement_label": 0,
            "state_hash_after": replacement_hash,
            "trusted_journal_replayed": replacement["trusted_journal_replayed"],
            "reconstruction_duration_s": replacement_duration,
        },
        {
            "operation": "erase_update",
            "event_id": "a",
            "prior_label": 0,
            "replacement_label": None,
            "state_hash_after": restarted_replacement.state_hash,
            "trusted_journal_replayed": erased["trusted_journal_replayed"],
            "reconstruction_duration_s": erase_duration,
        },
    ]
    return controls, revocations


def synthetic_metric_rows(groups: int, seeds: int) -> list[JsonDict]:
    """Build deterministic paired rows for reducer and mutation tests."""

    output = []
    for index in range(groups):
        label = index % 2
        for seed_index in range(seeds):
            for arm_index, arm in enumerate(ARMS):
                base = 0.75 if label else 0.25
                offset = {
                    FROZEN_ARM: 0.08,
                    ADAPTIVE_ARM: -0.04,
                    LOGISTIC_ARM: 0.06,
                    FREQUENCY_ARM: 0.10,
                    NO_FEEDBACK_ARM: 0.08,
                }[arm]
                probability = base + (offset if label == 0 else -offset)
                output.append(
                    {
                        "observation_id": f"observation-{index:04d}",
                        "row_key": f"row-{index:04d}",
                        "group_id": f"group-{index:04d}",
                        "arm": arm,
                        "seed": 65_001 + seed_index,
                        "ordering": "hash_order",
                        "feedback_regime": "primary_label_blind_75",
                        "delay": 1,
                        "prediction_index": index,
                        "available_at": index + 1,
                        "probability": probability,
                        "label": label,
                        "label_authority": LABEL_AUTHORITY,
                        "decision": "escalate",
                        "registered_feedback_mask": index % 4 != 0,
                        "diagnostic_learner_selected_mask": True,
                        "prediction_before_feedback": True,
                        "prediction_persisted": True,
                        "state_hash_at_prediction": canonical_hash([arm_index, seed_index, index]),
                        "prior_state_hash": canonical_hash([arm_index, seed_index, index]),
                        "new_state_hash": canonical_hash([arm_index, seed_index, index]),
                        "feedback_status": "fixture",
                        "update_admitted": arm in {ADAPTIVE_ARM, LOGISTIC_ARM, FREQUENCY_ARM},
                        "commit_after_prediction": True,
                        "brier_contribution": (probability - label) ** 2,
                        "log_loss_contribution": _log_loss(label, probability),
                        "prediction_duration_s": 0.0,
                        "persistence_duration_s": 0.0,
                        "update_duration_s": 0.0,
                    }
                )
    return output


def condition_reports(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce proper scores, typed-action coverage, and observed action risk."""

    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("label") in {0, 1}:
            key = (
                str(row["ordering"]),
                str(row["feedback_regime"]),
                int(row["delay"]),
                str(row["arm"]),
            )
            grouped[key].append(row)
    reports = []
    for (ordering, regime, delay, arm), selected in sorted(grouped.items()):
        action_rows = [row for row in selected if row.get("decision") != "escalate"]
        harmful = [
            row
            for row in action_rows
            if (row["decision"] == "accept" and row["label"] == 1)
            or (row["decision"] == "reject" and row["label"] == 0)
        ]
        reports.append(
            {
                "ordering": ordering,
                "feedback_regime": regime,
                "delay": delay,
                "arm": arm,
                "scored_rows": len(selected),
                "independent_groups": len({str(row["group_id"]) for row in selected}),
                "brier": float(np.mean([float(row["brier_contribution"]) for row in selected])),
                "log_loss": float(
                    np.mean([float(row["log_loss_contribution"]) for row in selected])
                ),
                "coverage": len(action_rows) / len(selected),
                "observed_action_risk": len(harmful) / len(action_rows) if action_rows else 0.0,
                "real_world_chronology": False,
                "iid_guarantee_asserted": False,
                "conformal_guarantee_asserted": False,
            }
        )
    return reports


def _paired_event_deltas(
    rows: Sequence[Mapping[str, Any]], regime: str, control: str
) -> np.ndarray:
    """Average seeds within each event before paired resampling."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("ordering") == "hash_order"
            and row.get("feedback_regime") == regime
            and row.get("delay") == 1
            and row.get("arm") in {ADAPTIVE_ARM, control}
            and row.get("brier_contribution") is not None
        ):
            grouped[(str(row["observation_id"]), str(row["arm"]))].append(
                float(row["brier_contribution"])
            )
    identities = sorted({identity for identity, _arm in grouped})
    if any(
        (identity, ADAPTIVE_ARM) not in grouped or (identity, control) not in grouped
        for identity in identities
    ):
        raise ValueError("paired rows are required for every event")
    return np.asarray(
        [
            np.mean(grouped[(identity, ADAPTIVE_ARM)]) - np.mean(grouped[(identity, control)])
            for identity in identities
        ],
        dtype=np.float64,
    )


def paired_moving_block_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Compute paired circular moving-block intervals for both block lengths."""

    if not rows:
        raise ValueError("paired rows are required")
    if draws <= 0:
        raise ValueError("draws must be positive")
    regimes = sorted(
        {
            str(row["feedback_regime"])
            for row in rows
            if row.get("ordering") == "hash_order" and row.get("delay") == 1
        }
    )
    if not regimes:
        raise ValueError("paired rows are required")
    output = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for regime in regimes:
        vectors = {control: _paired_event_deltas(rows, regime, control) for control in CONTROL_ARMS}
        lengths = {len(vector) for vector in vectors.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("paired rows are required for every control")
        size = next(iter(lengths))
        for block_length in BLOCK_LENGTHS:
            block_count = math.ceil(size / block_length)
            starts = rng.integers(0, size, size=(draws, block_count))
            indices = (
                starts[:, :, None] + np.arange(block_length, dtype=np.int64)[None, None, :]
            ) % size
            indices = indices.reshape(draws, -1)[:, :size]
            draw_values = {
                control: vector[indices].mean(axis=1) for control, vector in vectors.items()
            }
            centered_max = np.maximum.reduce(
                [
                    values - float(np.mean(vectors[control]))
                    for control, values in draw_values.items()
                ]
            )
            simultaneous_margin = float(np.quantile(centered_max, 0.95))
            for control in CONTROL_ARMS:
                observed = float(np.mean(vectors[control]))
                values = draw_values[control]
                output.append(
                    {
                        "feedback_regime": regime,
                        "ordering": "hash_order",
                        "delay": 1,
                        "adaptive_arm": ADAPTIVE_ARM,
                        "control_arm": control,
                        "metric": "brier_delta",
                        "estimate": observed,
                        "lower_95": float(np.quantile(values, 0.025)),
                        "upper_95": float(np.quantile(values, 0.975)),
                        "upper_simultaneous_95": observed + simultaneous_margin,
                        "draws": draws,
                        "block_length": block_length,
                        "seed": BOOTSTRAP_SEED,
                        "seeds_averaged_within_event": True,
                    }
                )
    return output


def reduce_online_value(
    reports: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
) -> JsonDict:
    """Apply the registered support floor and equal-information conjunction."""

    label_counts = support.get("label_counts") or {}
    support_passed = (
        int(support.get("independent_online_groups") or 0) >= 80
        and int(label_counts.get("0") or 0) >= 10
        and int(label_counts.get("1") or 0) >= 10
    )
    primary = {
        str(row["arm"]): row
        for row in reports
        if row.get("ordering") == "hash_order"
        and row.get("feedback_regime") == "primary_label_blind_75"
        and row.get("delay") == 1
    }
    adaptive = primary.get(ADAPTIVE_ARM)
    checks = []
    for control in CONTROL_ARMS:
        comparison = primary.get(control)
        interval = next(
            (
                row
                for row in intervals
                if row.get("feedback_regime") == "primary_label_blind_75"
                and row.get("control_arm") == control
                and row.get("block_length") == 16
            ),
            None,
        )
        checks.append(
            {
                "control_arm": control,
                "brier_simultaneous_upper_below_zero": bool(
                    interval and float(interval["upper_simultaneous_95"]) < 0.0
                ),
                "log_loss_non_worse": bool(
                    adaptive
                    and comparison
                    and float(adaptive["log_loss"]) <= float(comparison["log_loss"])
                ),
                "coverage_not_lower": bool(
                    adaptive
                    and comparison
                    and float(adaptive["coverage"]) >= float(comparison["coverage"])
                ),
                "observed_action_risk_not_higher": bool(
                    adaptive
                    and comparison
                    and float(adaptive["observed_action_risk"])
                    <= float(comparison["observed_action_risk"])
                ),
            }
        )
    passed = support_passed and all(
        all(value is True for key, value in row.items() if key != "control_arm") for row in checks
    )
    verdict = (
        "complete_positive_registered_online_value"
        if passed
        else "complete_null_insufficient_online_support"
        if not support_passed
        else "complete_null_no_registered_online_value"
    )
    return {
        "support_passed": support_passed,
        "support": deepcopy(dict(support)),
        "control_checks": checks,
        "passed": passed,
        "terminal_verdict": verdict,
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one clean receipt for every frozen affected and terminal check."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(passed)


def _percentiles(values: Sequence[float]) -> JsonDict:
    """Report measured median and p95 while accepting zero-cost fixtures."""

    data = np.asarray(list(values) or [0.0], dtype=np.float64)
    return {"p50": float(np.quantile(data, 0.50)), "p95": float(np.quantile(data, 0.95))}


def _hardware_path(
    event_rows: Sequence[Mapping[str, Any]], revocation_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Summarize bounded CPU costs without inventing an acceleration claim."""

    return {
        "path": "cpu_bounded_numeric_updates_fixed_size_memory",
        "speedup_claimed": False,
        "memory_bound": {"recent_frequency_window": RECENT_WINDOW, "gibbs_trainable_weights": 0},
        "latency_s": {
            "prediction": _percentiles(
                [float(row.get("prediction_duration_s") or 0.0) for row in event_rows]
            ),
            "persistence": _percentiles(
                [float(row.get("persistence_duration_s") or 0.0) for row in event_rows]
            ),
            "update": _percentiles(
                [float(row.get("update_duration_s") or 0.0) for row in event_rows]
            ),
            "reconstruction": _percentiles(
                [float(row.get("reconstruction_duration_s") or 0.0) for row in revocation_rows]
            ),
        },
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
    """Keep validity, completion, and benefit as separate ordinary fields."""

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
    """Name the first exact prerequisite failure and all failed required checks."""

    blocked = next((row for row in preconditions if row.get("passed") is not True), None)
    return {
        "blocked_upstream": blocked.get("upstream") if blocked else None,
        "blocked_path": blocked.get("path") if blocked else None,
        "blocked_check": blocked.get("check") if blocked else None,
        "blocked_field": blocked.get("field") if blocked else None,
        "blocked_expected": blocked.get("expected") if blocked else None,
        "blocked_observed": blocked.get("observed") if blocked else None,
        "failed_required_checks": [
            str(row["check"])
            for row in gates
            if row.get("category") != "scientific_benefit" and row.get("passed") is not True
        ],
        "required_checks_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") != "scientific_benefit"
        ),
        "scientific_benefit_passed": any(
            row.get("category") == "scientific_benefit" and row.get("passed") is True
            for row in gates
        ),
    }


def _field_principles() -> dict[str, str]:
    """Explain each ordinary field without wrapping gate scalars in objects."""

    common = {
        "schema": "Versioned ordinary top-level fields identify the record contract.",
        "run_date": "The scheduled date is separate from actual UTC run boundaries.",
        "preconditions_checked": "Exact paths, hashes, and eligibility gates precede labels.",
        "MODEL_SPECS": "No current LLM use requires an empty model list.",
        "model_invoked": "Only current attempted LLM work can set this true.",
        "invocation_counts": "Current owned attempts and terminal events reduce to counters.",
        "inference_substrate": "The substrate is a truthful string with detail kept separate.",
        "inference_substrate_class": "The declared work class is no_model_load.",
        "execution_venue": "The closed venue is host.",
        "duration_s": "Monotonic current duration excludes cited historical work.",
        "phase_spans": "Real phase boundaries retain completed-unit checkpoints.",
        "random_seed": "Frozen fit and resampling seeds make the replay reproducible.",
        "reproducibility_checksum": "The checksum binds protocol, inputs, and raw replay rows.",
        "source_artifact_hashes": "Exact byte hashes preserve each input identity and flag.",
        "rows": "Every arm, seed, and replay condition remains explicit.",
        "sample_size_budget": "Planned and terminal unit counts stay separate from support.",
        "acceptance_gate_results": "Validity and benefit checks use explicit operators.",
        "gate_check_summary": "Blocked fields retain exact expected and observed values.",
        "verifier_is_oracle": "Machine labels are not semantic correctness authority.",
        "honest_verdict": "Completed findings use a complete terminal prefix.",
        "verdict_class": "The verdict uses the closed project enum.",
        "flagged_adversarial": "Critical verification findings disqualify readiness.",
        "validation_receipts": "Each required command keeps argv, duration, exit, and log hash.",
        "field_principles": "Fields explain their evidence role separately.",
        "promotion_score": "No automatic rollout, publication, or generator update occurs.",
        "continuous_self_learning_task": "Predictions precede delayed across-query updates.",
        "online_capture_complete_score": "A valid accounted replay can complete with a null.",
        "online_value_score": "Only the registered equal-information conjunction has value.",
        "feedback_event_rows": "Rows preserve prediction, reveal, commit, masks, and lineage.",
        "revocation_rows": "Replacement and erasure include deterministic reconstruction cost.",
        "hardware_path": "CPU p50 and p95 costs replace an unsupported speedup claim.",
    }
    return {
        field: common.get(field, "This ordinary field preserves measured experiment evidence.")
        for field in REQUIRED_FIELDS
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind fixed protocol settings, source bytes, and material scientific rows."""

    keys = (
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "feedback_event_rows",
        "revocation_rows",
        "condition_reports",
        "paired_moving_block_intervals",
        "online_value_reduction",
        "label_authority",
    )
    return canonical_hash({key: value.get(key) for key in keys})


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute capture and value scores from raw stored evidence."""

    if value.get("verdict_class") == "blocked":
        return {
            "online_capture_complete_score": 0,
            "online_value_score": 0,
            "promotion_score": 0,
        }
    preconditions_passed = all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    controls_passed = all(
        row.get("passed") is True for row in (value.get("analytic_controls") or {}).values()
    )
    events = value.get("feedback_event_rows") or []
    events_valid = bool(events) and all(
        row.get("prediction_before_feedback") is True
        and row.get("prediction_persisted") is True
        and isinstance(row.get("probability"), (int, float))
        and 0.0 <= float(row["probability"]) <= 1.0
        for row in events
    )
    validation_passed = _validation_passed(value.get("validation_receipts") or [])
    value_passed = (value.get("online_value_reduction") or {}).get("passed") is True
    capture = int(preconditions_passed and controls_passed and events_valid and validation_passed)
    return {
        "online_capture_complete_score": capture,
        "online_value_score": int(capture == 1 and value_passed),
        "promotion_score": 0,
    }


def _build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    unit_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    revocation_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one schema-complete terminal or pre-terminal record."""

    value_reduction = reduce_online_value(reports, intervals, support)
    validation_passed = _validation_passed(validation_receipts)
    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    controls_passed = all(row.get("passed") is True for row in controls.values())
    gates = [
        _gate(
            "preconditions",
            "validity",
            "all",
            True,
            preconditions_passed,
            preconditions_passed,
            "Every exact source must authenticate before fitting or replay.",
        ),
        _gate(
            "registered_replay_units",
            "completion",
            "==",
            len(unit_rows),
            sum(row.get("status") == "completed" for row in unit_rows),
            bool(unit_rows) and all(row.get("status") == "completed" for row in unit_rows),
            "Every arm, seed, order, delay, and regime must remain accounted.",
        ),
        _gate(
            "analytic_controls",
            "validity",
            "all",
            True,
            controls_passed,
            controls_passed,
            "Revocation, restart, duplicate, future-label, and no-feedback controls must pass.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_passed,
            validation_passed,
            "All frozen scoped and terminal checks must pass.",
        ),
        _gate(
            "registered_online_value",
            "scientific_benefit",
            "==",
            True,
            value_reduction["passed"],
            bool(value_reduction["passed"]),
            "Value requires simultaneous Brier, log-loss, coverage, and action-risk gates.",
        ),
    ]
    valid_terminal = (
        preconditions_passed and controls_passed and validation_passed and not flagged_adversarial
    )
    if valid_terminal:
        verdict_class = "positive" if value_reduction["passed"] else "null"
        honest_verdict = str(value_reduction["terminal_verdict"])
        status = "complete"
    else:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_selected_feedback_validation"
        status = "disqualified"
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
        "random_seed": {
            "training": list(TRAINING_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "block_lengths": list(BLOCK_LENGTHS),
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in unit_rows],
        "sample_size_budget": {
            "planned": len(unit_rows),
            "attempted": len(unit_rows),
            "completed": sum(row.get("status") == "completed" for row in unit_rows),
            "failed": sum(row.get("status") == "failed" for row in unit_rows),
            "censored": sum(row.get("status") == "censored" for row in unit_rows),
            "unstarted": sum(row.get("status") == "unstarted" for row in unit_rows),
            "independent_groups": int(support.get("independent_online_groups") or 0),
            "label_counts": deepcopy(dict(support.get("label_counts") or {})),
            "stop_rule": "complete every sealed replay condition; do not tune on online labels",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates, preconditions),
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "continuous_self_learning_task": True,
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "feedback_event_rows": [deepcopy(dict(row)) for row in event_rows],
        "revocation_rows": [deepcopy(dict(row)) for row in revocation_rows],
        "analytic_controls": deepcopy(dict(controls)),
        "condition_reports": [deepcopy(dict(row)) for row in reports],
        "paired_moving_block_intervals": [deepcopy(dict(row)) for row in intervals],
        "online_value_reduction": value_reduction,
        "hardware_path": _hardware_path(event_rows, revocation_rows),
        "label_authority": LABEL_AUTHORITY,
        "methodology": (
            "Train source Gibbs and logistic heads on development train groups. Calibrate and "
            "select policy on separate development groups. Predict before delayed machine-label "
            "feedback. Compare fixed equal-information masks in constructed archive replays."
        ),
        "no_iid_guarantee": True,
        "no_conformal_guarantee": True,
        "exact_learning_branch": "Exp7418 remains independent of these machine annotations.",
    }
    reduced = independent_reduce(artifact)
    artifact.update(reduced)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_fixture_artifact(
    *, validation_receipts: Sequence[Mapping[str, Any]] | None = None
) -> JsonDict:
    """Build a compact valid null artifact for independent mutation tests."""

    receipts = list(validation_receipts or [])
    event_rows = synthetic_metric_rows(groups=20, seeds=2)
    reports = condition_reports(event_rows)
    intervals = paired_moving_block_intervals(event_rows, draws=200)
    controls, revocations = run_analytic_controls()
    support = {"independent_online_groups": 20, "label_counts": {"0": 10, "1": 10}}
    unit_rows = [
        {
            "comparative_unit": f"hash_order:primary_label_blind_75:delay1:{arm}:{seed}",
            "ordering": "hash_order",
            "feedback_regime": "primary_label_blind_75",
            "delay": 1,
            "arm": arm,
            "seed": seed,
            "status": "completed",
        }
        for seed in (65_001, 65_002)
        for arm in ARMS
    ]
    receipt = build_current_work_receipt(
        run_id="exp7414-fixture",
        owner_pid=0,
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details={"device": "cpu", "software": "numpy"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000,
        phase_spans=[
            {
                "phase": "fixture",
                "start_s": 0.0,
                "end_s": 0.001,
                "duration_s": 0.001,
                "completed_units": len(unit_rows),
            }
        ],
        small_ebm_training={"performed": True, "generator_weights_fitted": False},
    )
    return _build_artifact(
        preconditions=[
            _precondition("fixture_protocol", "fixture", "fixture", "ready", "==", True, True)
        ],
        source_hashes={},
        unit_rows=unit_rows,
        event_rows=event_rows,
        revocation_rows=revocations,
        controls=controls,
        reports=reports,
        intervals=intervals,
        support=support,
        validation_receipts=receipts,
        current_receipt=receipt,
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00.001000+00:00",
    )


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish one terminal external-prerequisite failure without dependent work."""

    receipt = build_current_work_receipt(
        run_id="exp7414-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details={"device": "cpu", "dependent_work_started": False},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        small_ebm_training={"performed": False},
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(failed))],
        **receipt,
        "random_seed": {
            "training": list(TRAINING_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "block_lengths": list(BLOCK_LENGTHS),
        },
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 200,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 200,
            "independent_groups": 0,
            "label_counts": {"0": 0, "1": 0},
            "stop_rule": "stop before dependent work when an exact prerequisite fails",
        },
        "acceptance_gate_results": [
            _gate(
                "preconditions",
                "validity",
                "all",
                True,
                False,
                False,
                "Every exact source must authenticate before dependent work.",
            )
        ],
        "gate_check_summary": _gate_summary([], [failed]),
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_{failed.get('check', 'external_prerequisite')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "continuous_self_learning_task": True,
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "feedback_event_rows": [],
        "revocation_rows": [],
        "analytic_controls": {},
        "condition_reports": [],
        "paired_moving_block_intervals": [],
        "online_value_reduction": {
            "support_passed": False,
            "passed": False,
            "terminal_verdict": "blocked_external_prerequisite",
        },
        "hardware_path": _hardware_path([], []),
        "label_authority": LABEL_AUTHORITY,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, provenance, event bounds, scores, and checksum."""

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
    errors.extend(validate_current_work_receipt(value))
    for index, row in enumerate(value.get("feedback_event_rows") or []):
        probability = row.get("probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            errors.append(f"event_probability_invalid:{index}")
    reduced = independent_reduce(value)
    if value.get("online_capture_complete_score") != reduced["online_capture_complete_score"]:
        errors.append("online_capture_complete_score_mismatch")
    if value.get("online_value_score") != reduced["online_value_score"]:
        errors.append("online_value_score_mismatch")
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
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> list[str]:
    """Read and validate a candidate in a fresh process."""

    value = _load_object(path)
    return validate_artifact(value) if value else ["artifact_unreadable_or_not_object"]


def _load_real_rows(root: Path) -> list[JsonDict]:  # pragma: no cover - entrypoint boundary.
    """Join frozen source features to development labels without final-test access."""

    features = _load_object(root / FEATURE_PATH).get("records") or []
    feature_by_key = {str(row["row_key"]): deepcopy(dict(row)) for row in features}
    corpus = reload_corpus(root / CORPUS_DIR)
    readers = CorpusReaders(corpus)
    joined: list[JsonDict] = []
    for partition in (
        "train",
        "probability_calibration",
        "policy_calibration",
        "online_stream",
    ):
        predictors = readers.read_predictors(partition)
        labels = readers.read_labels(partition)
        label_by_key = {str(row["row_key"]): row.get("label") for row in labels}
        for predictor in predictors:
            key = str(predictor["row_key"])
            row = deepcopy(feature_by_key[key])
            row.update(
                {
                    "partition": partition,
                    "label": label_by_key[key],
                    "label_authority": LABEL_AUTHORITY,
                }
            )
            joined.append(row)
    return joined


def _write_initial_checkpoints(
    root: Path, states: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:  # pragma: no cover - serialization boundary.
    """Persist code-free initial states that have never seen online labels."""

    directory = root / RAW_DIR / "initial_states"
    manifest = []
    for state in states:
        payload = {
            "schema": "carnot.exp7414.initial_state.v1",
            "seed": state["seed"],
            "gibbs_weights": state["gibbs_weights"],
            "affine": state["affine"],
            "logistic": state["logistic"],
            "selected_policy": state["selected_policy"],
            "training_receipt": state["training_receipt"],
        }
        path = directory / f"seed-{state['seed']}.json"
        atomic_json(path, payload)
        manifest.append(
            {
                "seed": state["seed"],
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
                "online_labels_seen": 0,
            }
        )
    return manifest


def _span(
    phase: str, phase_started: float, run_started: float, completed: int
) -> JsonDict:  # pragma: no cover - clock boundary.
    """Close one real phase with a monotonic completed-unit checkpoint."""

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
    """Build fresh replay, independent reduction, and unchanged strict readers."""

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
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint.
    """Authenticate, train, replay, validate, and atomically publish the result."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    monotonic_started_ns = time.monotonic_ns()
    started_at_utc = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "start")
    preconditions, source_hashes, _loaded = collect_preconditions(root)
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
    progress(run_started, "development_rows", "before_benchmark")
    rows = _load_real_rows(root)
    spans.append(_span("development_rows", phase_started, run_started, len(rows)))
    progress(run_started, "development_rows", "after_benchmark", completed=len(rows))

    states = []
    phase_started = time.monotonic()
    for index, seed in enumerate(TRAINING_SEEDS, start=1):
        progress(
            run_started,
            "small_ebm_training",
            "before_benchmark",
            seed=seed,
            unit=f"{index}/{len(TRAINING_SEEDS)}",
        )
        states.extend(initialize_seed_states(rows, seeds=(seed,), steps=500))
        progress(
            run_started,
            "small_ebm_training",
            "after_benchmark",
            seed=seed,
            completed=index,
        )
    spans.append(_span("small_ebm_training", phase_started, run_started, len(states)))

    phase_started = time.monotonic()
    progress(run_started, "initial_state_persistence", "before_serialization")
    checkpoint_manifest = _write_initial_checkpoints(root, states)
    for row in checkpoint_manifest:
        source_hashes[row["path"]] = {
            "path": row["path"],
            "sha256": row["sha256"],
            "original_flagged_adversarial": None,
        }
    spans.append(
        _span("initial_state_persistence", phase_started, run_started, len(checkpoint_manifest))
    )
    progress(
        run_started,
        "initial_state_persistence",
        "after_serialization",
        completed=len(checkpoint_manifest),
    )

    online_rows = [row for row in rows if row.get("partition") == "online_stream"]
    streams = build_streams(online_rows)
    event_rows: list[JsonDict] = []
    unit_rows: list[JsonDict] = []
    replay_total = len(states) * len(ORDERINGS) * len(FEEDBACK_REGIMES) * len(DELAYS)
    replay_completed = 0
    journal_manifest: list[JsonDict] = []
    journal_directory = root / RAW_DIR / "feedback_journals" / str(monotonic_started_ns)
    phase_started = time.monotonic()
    for state in states:
        for ordering in ORDERINGS:
            for regime in FEEDBACK_REGIMES:
                for delay in DELAYS:
                    replay_completed += 1
                    progress(
                        run_started,
                        "online_replay",
                        "before_benchmark",
                        seed=state["seed"],
                        ordering=ordering,
                        feedback_regime=regime,
                        delay=delay,
                        unit=f"{replay_completed}/{replay_total}",
                    )
                    replay = replay_condition(
                        state,
                        streams[ordering],
                        ordering=ordering,
                        feedback_regime=regime,
                        delay=delay,
                        journal_path=(
                            journal_directory
                            / f"seed-{state['seed']}--{ordering}--{regime}--delay-{delay}.jsonl"
                        ),
                    )
                    event_rows.extend(replay["feedback_event_rows"])
                    journal_path = (
                        journal_directory
                        / f"seed-{state['seed']}--{ordering}--{regime}--delay-{delay}.jsonl"
                    )
                    journal_row = {
                        "seed": state["seed"],
                        "ordering": ordering,
                        "feedback_regime": regime,
                        "delay": delay,
                        "path": journal_path.relative_to(root).as_posix(),
                        "sha256": sha256_file(journal_path),
                        "byte_size": journal_path.stat().st_size,
                    }
                    journal_manifest.append(journal_row)
                    source_hashes[journal_row["path"]] = {
                        "path": journal_row["path"],
                        "sha256": journal_row["sha256"],
                        "original_flagged_adversarial": None,
                    }
                    unit_rows.extend(
                        {
                            "comparative_unit": (
                                f"{ordering}:{regime}:delay{delay}:{arm}:{state['seed']}"
                            ),
                            "ordering": ordering,
                            "feedback_regime": regime,
                            "delay": delay,
                            "arm": arm,
                            "seed": state["seed"],
                            "status": "completed",
                            "initial_state_sha256": next(
                                row["sha256"]
                                for row in checkpoint_manifest
                                if row["seed"] == state["seed"]
                            ),
                            "feedback_journal_sha256": journal_row["sha256"],
                        }
                        for arm in ARMS
                    )
                    progress(
                        run_started,
                        "online_replay",
                        "after_benchmark",
                        completed=replay_completed,
                        total=replay_total,
                    )
    spans.append(_span("online_replay", phase_started, run_started, replay_completed))

    phase_started = time.monotonic()
    progress(run_started, "analytic_controls", "before_benchmark")
    controls, revocations = run_analytic_controls()
    spans.append(_span("analytic_controls", phase_started, run_started, len(controls)))
    progress(run_started, "analytic_controls", "after_benchmark", completed=len(controls))

    phase_started = time.monotonic()
    progress(run_started, "moving_block_intervals", "before_benchmark", draws=BOOTSTRAP_DRAWS)
    reports = condition_reports(event_rows)
    intervals = paired_moving_block_intervals(event_rows)
    spans.append(_span("moving_block_intervals", phase_started, run_started, BOOTSTRAP_DRAWS))
    progress(
        run_started,
        "moving_block_intervals",
        "after_benchmark",
        completed=BOOTSTRAP_DRAWS,
    )

    primary_stream = streams["hash_order"]
    labeled_primary = [row for row in primary_stream if row.get("label") in {0, 1}]
    support = {
        "independent_online_groups": len({str(row["group_id"]) for row in labeled_primary}),
        "label_counts": {
            "0": sum(row.get("label") == 0 for row in labeled_primary),
            "1": sum(row.get("label") == 1 for row in labeled_primary),
        },
    }

    raw_dir = root / RAW_DIR
    private_root = Path(tempfile.mkdtemp(prefix="exp7414-validation-", dir="/tmp"))
    commands = build_command_plan(root, V650_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V650_MANIFEST, commands)
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
    affected_reduction = reduce_affected_receipts(root, V650_MANIFEST, affected)
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
    current_receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{monotonic_started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details={
            "device": "cpu",
            "software": "numpy",
            "work": "small Gibbs fitting and bounded online numeric updates",
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - monotonic_started_ns,
        phase_spans=spans,
        small_ebm_training={
            "performed": True,
            "units": len(states),
            "steps_per_head": 500,
            "generator_weights_fitted": False,
            "online_labels_used_for_initialization": 0,
        },
    )
    candidate = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        unit_rows=unit_rows,
        event_rows=event_rows,
        revocation_rows=revocations,
        controls=controls,
        reports=reports,
        intervals=intervals,
        support=support,
        validation_receipts=affected,
        current_receipt=current_receipt,
        started_at_utc=started_at_utc,
        completed_at_utc=utc_now(),
        flagged_adversarial=not affected_reduction["passed"],
    )
    candidate["initial_state_manifest"] = checkpoint_manifest
    candidate["feedback_journal_manifest"] = journal_manifest
    candidate["reproducibility_checksum"] = reproducibility_checksum(candidate)
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
        run_id=current_receipt["current_run_id"],
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details=current_receipt["inference_substrate_details"],
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - monotonic_started_ns,
        phase_spans=spans,
        small_ebm_training=current_receipt["small_ebm_training"],
    )
    final = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        unit_rows=unit_rows,
        event_rows=event_rows,
        revocation_rows=revocations,
        controls=controls,
        reports=reports,
        intervals=intervals,
        support=support,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at_utc,
        completed_at_utc=utc_now(),
        flagged_adversarial=(not affected_reduction["passed"] or not terminal_passed or critical),
    )
    final["initial_state_manifest"] = checkpoint_manifest
    final["feedback_journal_manifest"] = journal_manifest
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process candidate reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = validate_artifact(value) if value else ["artifact_unreadable_or_not_object"]
        reduced = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduced}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
