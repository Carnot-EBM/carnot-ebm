"""Replay archived feedback through a four-expert online mixture.

The replay is prospective with respect to event order, but the labels come
from an archived human-feedback corpus. The result does not authorize deployed
self-learning or production policy changes.

Spec refs: REQ-AUTO-7440 and SCENARIO-AUTO-7440-01 through -05.
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
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_TOKEN,
    ProtocolReaders,
    reload_corpus,
)
from carnot.experiment_7427_v651_randomized_feedback import (
    ONLINE_GIBBS_ARM,
    ONLINE_SPLINE_ARM,
    OnlineLearner,
    build_streams,
)
from carnot.experiment_7438_v652_mixture_prototype import (
    ADAPTIVE_GIBBS,
    ADAPTIVE_SPLINE,
    EXPERT_NAMES,
    FROZEN_GIBBS,
    FROZEN_SPLINE,
    FourExpertMixture,
    bernoulli_log_loss,
    build_numeric_experts,
    mixture_probability,
    update_log_weights,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7440-v652-mixture-learning"
SCHEMA = "carnot.exp7440.v652.mixture_learning.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7440_v652_mixture_learning.json")
RAW_DIR = Path("results/raw/experiment_7440_v652_mixture_learning")
ROW_DIR = RAW_DIR / "rows"
MODULE_PATH = Path("python/carnot/experiment_7440_v652_mixture_learning.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7440_v652_mixture_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7440_v652_mixture_learning.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
PROTOTYPE_PATH = Path("results/experiment_7438_v652_mixture_prototype.json")
DECISIONS_PATH = Path("results/experiment_7439_v652_certified_decisions.json")
PROTOCOL_PATH = Path("results/raw/experiment_7438_v652_mixture_prototype/protocol.json")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_MANIFEST_PATH = CORPUS_DIR / "corpus_manifest.json"

TRAINING_SEEDS = (65_201, 65_202, 65_203, 65_204, 65_205)
ORDERS = ("hash_order", "domain_blocked_shift_order")
DELAYS = (0, 8)
BLOCK_SIZE = 32
SENSITIVITY_BLOCK_SIZE = 64
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_520_440
LEARNED_MIXTURE = "learned_mixture"
EQUAL_MIXTURE = "equal_weight_adaptive_mixture"
FROZEN_SPLINE_ARM = "frozen_spline"
ADAPTIVE_SPLINE_ARM = "adaptive_spline"
NO_FEEDBACK_MIXTURE = "no_feedback_frozen_prior_mixture"
SHUFFLED_CONTROL_1 = "shuffled_labels_permutation_1"
SHUFFLED_CONTROL_2 = "shuffled_labels_permutation_2"
ARMS = (
    LEARNED_MIXTURE,
    EQUAL_MIXTURE,
    FROZEN_SPLINE_ARM,
    ADAPTIVE_SPLINE_ARM,
    NO_FEEDBACK_MIXTURE,
    SHUFFLED_CONTROL_1,
    SHUFFLED_CONTROL_2,
)
PRIMARY_COMPARATORS = (FROZEN_SPLINE_ARM, ADAPTIVE_SPLINE_ARM, EQUAL_MIXTURE)
INFERENCE_SUBSTRATE = "archived_feedback_numeric_replay"
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
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7427_v651_randomized_feedback.py"),
    Path("python/carnot/experiment_7438_v652_mixture_prototype.py"),
    Path("python/carnot/experiment_7439_v652_certified_decisions.py"),
    SPEC_PATH,
    PROTOTYPE_PATH,
    DECISIONS_PATH,
    PROTOCOL_PATH,
    CORPUS_MANIFEST_PATH,
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
    "weight_trajectory_rows",
    "checkpoint_lineage",
    "shadow_decisions_only",
    "hardware_path",
)


def utc_now() -> str:  # pragma: no cover - authentic clock boundary.
    """Return one aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed phase boundary or long-loop checkpoint."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7440] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and return an empty mapping for invalid bytes."""

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
    passed: bool,
) -> JsonDict:
    """Keep missing, null, zero, and false observations distinct."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def upstream_field_checks(
    prototype: Mapping[str, Any], decisions: Mapping[str, Any]
) -> list[JsonDict]:
    """Apply the six structured same-milestone gates before replay."""

    definitions = (
        (
            "mixture_prototype_ready",
            "exp7438-mixture-prototype",
            PROTOTYPE_PATH,
            "mixture_prototype_ready_score",
            "==",
            1,
            prototype.get("mixture_prototype_ready_score"),
            prototype.get("mixture_prototype_ready_score") == 1,
        ),
        (
            "mixture_prototype_verdict",
            "exp7438-mixture-prototype",
            PROTOTYPE_PATH,
            "verdict_class",
            "in",
            ["circular_positive", "null", "positive"],
            prototype.get("verdict_class"),
            prototype.get("verdict_class") in {"circular_positive", "null", "positive"},
        ),
        (
            "mixture_prototype_unflagged",
            "exp7438-mixture-prototype",
            PROTOTYPE_PATH,
            "flagged_adversarial",
            "==",
            False,
            prototype.get("flagged_adversarial"),
            prototype.get("flagged_adversarial") is False,
        ),
        (
            "decision_capture_complete",
            "exp7439-certified-decisions",
            DECISIONS_PATH,
            "decision_capture_complete_score",
            "==",
            1,
            decisions.get("decision_capture_complete_score"),
            decisions.get("decision_capture_complete_score") == 1,
        ),
        (
            "certified_decisions_verdict",
            "exp7439-certified-decisions",
            DECISIONS_PATH,
            "verdict_class",
            "in",
            ["null", "positive"],
            decisions.get("verdict_class"),
            decisions.get("verdict_class") in {"null", "positive"},
        ),
        (
            "certified_decisions_unflagged",
            "exp7439-certified-decisions",
            DECISIONS_PATH,
            "flagged_adversarial",
            "==",
            False,
            decisions.get("flagged_adversarial"),
            decisions.get("flagged_adversarial") is False,
        ),
    )
    return [
        _precondition(
            check,
            upstream,
            path.as_posix(),
            field,
            operator,
            expected,
            observed,
            passed,
        )
        for check, upstream, path, field, operator, expected, observed, passed in definitions
    ]


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate declared sources, original flags, and checkpoint bytes."""

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
                observed == "readable_nonempty_bytes",
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_verdict_class": None,
                "original_flagged_adversarial": None,
            }
    prototype = _load_object(root / PROTOTYPE_PATH)
    decisions = _load_object(root / DECISIONS_PATH)
    checks.extend(upstream_field_checks(prototype, decisions))
    for relative, value in ((PROTOTYPE_PATH, prototype), (DECISIONS_PATH, decisions)):
        if relative.as_posix() in hashes:
            hashes[relative.as_posix()].update(
                {
                    "original_verdict_class": value.get("verdict_class"),
                    "original_flagged_adversarial": value.get("flagged_adversarial"),
                    "original_honest_verdict": value.get("honest_verdict"),
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
            "==",
            "REQ-AUTO-7440",
            "REQ-AUTO-7440" if "REQ-AUTO-7440" in spec_text else None,
            "REQ-AUTO-7440" in spec_text,
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
            "==",
            False,
            "experiment_id: 7440" in exclusion,
            "experiment_id: 7440" not in exclusion,
        )
    )
    protocol = _load_object(root / PROTOCOL_PATH)
    checks.append(
        _precondition(
            "prototype_protocol_identity",
            "exp7438-mixture-prototype",
            PROTOCOL_PATH.as_posix(),
            "schema",
            "==",
            "carnot.exp7438.four_expert_protocol.v1",
            protocol.get("schema"),
            protocol.get("schema") == "carnot.exp7438.four_expert_protocol.v1",
        )
    )
    manifest = decisions.get("checkpoint_manifest")
    selected = (
        [
            row
            for row in manifest
            if isinstance(row, Mapping) and row.get("head") in {"gibbs_6_4_1", "sparse_spline_49"}
        ]
        if isinstance(manifest, list)
        else []
    )
    checkpoint_ok = len(selected) == 10
    for row in selected:
        path = root / str(row.get("path") or "")
        observed = sha256_file(path) if path.is_file() else None
        checkpoint_ok = checkpoint_ok and observed == row.get("sha256")
        checks.append(
            _precondition(
                f"checkpoint:{row.get('head')}:{row.get('seed')}",
                "exp7439-certified-decisions",
                str(row.get("path")),
                "sha256",
                "==",
                row.get("sha256"),
                observed,
                observed == row.get("sha256"),
            )
        )
        if observed is not None:
            hashes[str(row["path"])] = {
                "path": str(row["path"]),
                "sha256": observed,
                "source_receipt_class": "small_ebm_training",
                "original_flagged_adversarial": False,
            }
    checks.append(
        _precondition(
            "exact_new_fit_checkpoint_count",
            "exp7439-certified-decisions",
            DECISIONS_PATH.as_posix(),
            "checkpoint_manifest",
            "==",
            10,
            len(selected),
            checkpoint_ok,
        )
    )
    return checks, hashes, {"prototype": prototype, "decisions": decisions, "protocol": protocol}


def build_uniform_reveal_schedule(
    rows: Sequence[Mapping[str, Any]], *, seed: int, block_size: int = BLOCK_SIZE
) -> list[JsonDict]:
    """Select one quarter of each block without reading labels."""

    if block_size <= 0:
        raise ValueError("block size must be positive")
    identities = [str(row.get("observation_id") or "") for row in rows]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("observation identities must be unique and nonempty")
    rng = np.random.default_rng(seed)
    output: list[JsonDict] = []
    for block_index, start in enumerate(range(0, len(rows), block_size)):
        block = identities[start : start + block_size]
        reveal_count = len(block) // 4
        chosen = set(
            rng.choice(block, size=reveal_count, replace=False).tolist() if reveal_count else []
        )
        propensity = reveal_count / len(block)
        output.extend(
            {
                "observation_id": identity,
                "block_index": block_index,
                "block_size": len(block),
                "revealed": identity in chosen,
                "propensity": propensity,
                "selection_component": "uniform_without_replacement",
            }
            for identity in block
        )
    return output


def shadow_action(probability: float, thresholds: Mapping[str, Any]) -> str:
    """Propose a typed action while leaving deployment at all-escalate."""

    if probability >= float(thresholds["accept_threshold"]):
        return "accept"
    if probability <= float(thresholds["reject_threshold"]):
        return "reject"
    return "escalate"


def _policy(thresholds: Mapping[str, Any]) -> JsonDict:
    """Build the immutable shadow policy required by numeric learners."""

    return {
        "accept_threshold": float(thresholds["accept_threshold"]),
        "reject_threshold": float(thresholds["reject_threshold"]),
        "accept_enabled": True,
        "reject_enabled": True,
    }


def _learner(
    arm: str,
    seed: int,
    checkpoint: Mapping[str, Any],
    calibration: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> OnlineLearner:
    """Create one independent online learner from authenticated numeric state."""

    return OnlineLearner(
        arm=arm,
        seed=seed,
        checkpoint=checkpoint,
        calibration=calibration,
        policy=_policy(thresholds),
    )


class ReplayMixture(FourExpertMixture):
    """Use prototype math without retaining rollback snapshots for every reveal.

    Exp7438 already tests atomic restart and rollback behavior. This prospective
    replay needs the prediction journal and exactly-once set, but full snapshots
    would copy hundreds of old requests after every second reveal.
    """

    def _capture_safe_checkpoint(self) -> None:
        """Skip prototype rollback snapshots that this immutable replay never uses."""

    def _apply_feedback(
        self, identity: str, label: int, visible_at: int, *, replay: bool
    ) -> JsonDict:
        """Apply stored losses and two adaptive updates without journal-size scans."""

        prediction = self.predictions[identity]
        probabilities = [prediction["expert_probabilities"][name] for name in EXPERT_NAMES]
        weights_before = self.weights
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
        for name in (ADAPTIVE_SPLINE, ADAPTIVE_GIBBS):
            receipt = self.experts[name].commit_feedback(identity, label, visible_at=visible_at)
            if (
                receipt.get("update_admitted") is not True
            ):  # pragma: no cover - inherited learner guard.
                raise ValueError(f"adaptive expert rejected trusted feedback:{name}")
            expert_updates[name] = receipt
            touched += int(receipt.get("touched_coefficients") or 0)
        self.pending.pop(identity, None)
        self.feedback_journal.append(
            {
                "event_id": identity,
                "label": int(label),
                "visible_at": int(visible_at),
                "prediction_hash": prediction["prediction_hash"],
                "active": True,
            }
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
            "update_duration_ns": time.perf_counter_ns() - started_ns,
            "prediction_before_reveal": int(prediction["prediction_index"]) <= visible_at,
            "replayed": replay,
        }


def _controller(state: Mapping[str, Any], *, equal: bool) -> ReplayMixture:
    """Create four independent experts with learned or fixed mixture weights."""

    seed = int(state["seed"])
    thresholds = state["thresholds"]
    experts = {
        FROZEN_SPLINE: _learner(
            ONLINE_SPLINE_ARM,
            seed,
            state["spline_checkpoint"],
            state["spline_calibration"],
            thresholds,
        ),
        ADAPTIVE_SPLINE: _learner(
            ONLINE_SPLINE_ARM,
            seed,
            state["spline_checkpoint"],
            state["spline_calibration"],
            thresholds,
        ),
        FROZEN_GIBBS: _learner(
            ONLINE_GIBBS_ARM,
            seed,
            state["gibbs_checkpoint"],
            state["gibbs_calibration"],
            thresholds,
        ),
        ADAPTIVE_GIBBS: _learner(
            ONLINE_GIBBS_ARM,
            seed,
            state["gibbs_checkpoint"],
            state["gibbs_calibration"],
            thresholds,
        ),
    }
    return ReplayMixture(
        experts,
        eta=0.0 if equal else float(state["eta"]),
        fixed_share=0.0 if equal else float(state["fixed_share"]),
        max_pending=2_000,
    )


def _controller_predictive_hash(controller: FourExpertMixture) -> str:
    """Hash numeric predictive state without the growing request journal."""

    return canonical_hash(
        {
            "log_weights": controller.log_weights,
            "expert_state_hashes": controller.expert_state_hashes,
            "committed_feedback_count": len(controller.feedback_journal),
        }
    )


def build_fixture_states(*, seed: int) -> JsonDict:
    """Build small shipped numeric states for causal unit tests."""

    experts = build_numeric_experts(seed=seed)
    identity = {"affine": {"slope": 1.0, "intercept": 0.0}}
    return {
        "seed": seed,
        "spline_checkpoint": deepcopy(experts[FROZEN_SPLINE].initial_checkpoint),
        "gibbs_checkpoint": deepcopy(experts[FROZEN_GIBBS].initial_checkpoint),
        "spline_calibration": deepcopy(identity),
        "gibbs_calibration": deepcopy(identity),
        "thresholds": {"accept_threshold": 0.8, "reject_threshold": 0.2},
        "eta": 1.0,
        "fixed_share": 0.01,
        "source_checkpoints": [],
    }


def _fixed_predictions(
    state: Mapping[str, Any], features: Mapping[str, Any]
) -> tuple[float, float]:
    """Score frozen spline and frozen four-expert prior without mutation."""

    seed = int(state["seed"])
    thresholds = state["thresholds"]
    spline = _learner(
        ONLINE_SPLINE_ARM,
        seed,
        state["spline_checkpoint"],
        state["spline_calibration"],
        thresholds,
    )
    gibbs = _learner(
        ONLINE_GIBBS_ARM,
        seed,
        state["gibbs_checkpoint"],
        state["gibbs_calibration"],
        thresholds,
    )
    spline_probability = float(spline.predict(features)[0])
    gibbs_probability = float(gibbs.predict(features)[0])
    return spline_probability, mixture_probability(
        [spline_probability, spline_probability, gibbs_probability, gibbs_probability],
        [0.25] * 4,
    )


def replay_cell(
    state: Mapping[str, Any],
    stream: Sequence[Mapping[str, Any]],
    reveal_rows: Sequence[Mapping[str, Any]],
    *,
    ordering: str,
    delay: int,
    seed: int,
) -> JsonDict:
    """Replay one order-delay-seed cell and preserve prediction-time evidence."""

    if ordering not in ORDERS or delay not in DELAYS or seed != int(state["seed"]):
        raise ValueError("registered replay cell is required")
    schedule = {str(row["observation_id"]): row for row in reveal_rows}
    identities = {str(row["observation_id"]) for row in stream}
    if set(schedule) != identities:
        raise ValueError("reveal schedule must cover the stream exactly")
    learned = _controller(state, equal=False)
    equal = _controller(state, equal=True)
    shuffled_one = _controller(state, equal=False)
    shuffled_two = _controller(state, equal=False)
    adaptive = _learner(
        ONLINE_SPLINE_ARM,
        seed,
        state["spline_checkpoint"],
        state["spline_calibration"],
        state["thresholds"],
    )
    controllers = {
        LEARNED_MIXTURE: learned,
        EQUAL_MIXTURE: equal,
        SHUFFLED_CONTROL_1: shuffled_one,
        SHUFFLED_CONTROL_2: shuffled_two,
    }
    true_labels = [int(row["label"]) for row in stream]
    permutation_one = true_labels[1:] + true_labels[:1]
    shift = 7 % len(true_labels) if true_labels else 0
    permutation_two = true_labels[shift:] + true_labels[:shift]
    control_labels = {
        SHUFFLED_CONTROL_1: permutation_one,
        SHUFFLED_CONTROL_2: permutation_two,
    }
    predictions: list[JsonDict] = []
    feedback: list[JsonDict] = []
    weights: list[JsonDict] = []
    lineage: list[JsonDict] = []
    pending: list[JsonDict] = []
    prediction_lookup: dict[tuple[str, str], JsonDict] = {}
    last_domain: str | None = None
    duplicate_updates = 0
    read_duration_ns = 0
    predict_duration_ns = 0
    persist_duration_ns = 0
    update_duration_ns = 0

    def deliver(arrival_index: int) -> None:
        nonlocal duplicate_updates, update_duration_ns
        due = sorted(
            [row for row in pending if int(row["available_at"]) <= arrival_index],
            key=lambda row: (int(row["available_at"]), int(row["prediction_index"])),
        )
        for pending_row in due:
            pending.remove(pending_row)
            identity = str(pending_row["observation_id"])
            true_label = int(pending_row["label"])
            for arm, controller in controllers.items():
                delivered_label = (
                    control_labels[arm][int(pending_row["prediction_index"])]
                    if arm in control_labels
                    else true_label
                )
                parent = _controller_predictive_hash(controller)
                before = controller.weights
                started = time.perf_counter_ns()
                receipt = controller.commit_feedback(
                    identity, delivered_label, visible_at=arrival_index
                )
                update_duration_ns += time.perf_counter_ns() - started
                if (
                    receipt.get("update_admitted") is not True
                ):  # pragma: no cover - guarded by unique schedule identities.
                    duplicate_updates += 1
                child = _controller_predictive_hash(controller)
                prediction = prediction_lookup[(identity, arm)]
                event_hash = canonical_hash(
                    {
                        "observation_id": identity,
                        "arm": arm,
                        "feedback_label": delivered_label,
                        "arrival_index": arrival_index,
                    }
                )
                feedback.append(
                    {
                        "row_type": "feedback_event",
                        "observation_id": identity,
                        "group_id": prediction["group_id"],
                        "arm": arm,
                        "seed": seed,
                        "ordering": ordering,
                        "delay": delay,
                        "prediction_index": prediction["prediction_index"],
                        "request_order": prediction["prediction_index"],
                        "arrival_index": arrival_index,
                        "reveal_probability": prediction["propensity"],
                        "true_label": true_label,
                        "feedback_label": delivered_label,
                        "prediction_commit_hash": prediction["prediction_commit_hash"],
                        "parent_state_hash": parent,
                        "event_hash": event_hash,
                        "child_state_hash": child,
                        "probability_source": "stored_at_prediction",
                        "prediction_time_loss": bernoulli_log_loss(
                            delivered_label, float(prediction["probability"])
                        ),
                        "update_count": int(receipt.get("update_admitted") is True),
                        "update_cost": {
                            "weight_loss_evaluations": receipt.get(
                                "weight_loss_evaluation_count", 4
                            ),
                            "adaptive_expert_updates": receipt.get("expert_update_count", 2),
                            "touched_coefficients": receipt.get("touched_coefficients", 0),
                            "duration_ns": receipt.get("update_duration_ns", 0),
                        },
                    }
                )
                lineage.append(
                    {
                        "row_type": "checkpoint_lineage",
                        "observation_id": identity,
                        "arm": arm,
                        "seed": seed,
                        "ordering": ordering,
                        "delay": delay,
                        "parent_state_hash": parent,
                        "event_hash": event_hash,
                        "child_state_hash": child,
                        "exactly_once": receipt.get("update_admitted") is True,
                    }
                )
                weights.append(
                    {
                        "row_type": "weight_trajectory",
                        "observation_id": identity,
                        "arm": arm,
                        "seed": seed,
                        "ordering": ordering,
                        "delay": delay,
                        "arrival_index": arrival_index,
                        "weights_before": before,
                        "weights_after": controller.weights,
                        "frozen_option_weight_after": controller.weights[FROZEN_SPLINE]
                        + controller.weights[FROZEN_GIBBS],
                    }
                )
            parent = adaptive.state_hash
            started = time.perf_counter_ns()
            receipt = adaptive.commit_feedback(identity, true_label, visible_at=arrival_index)
            update_duration_ns += time.perf_counter_ns() - started
            child = adaptive.state_hash
            prediction = prediction_lookup[(identity, ADAPTIVE_SPLINE_ARM)]
            event_hash = canonical_hash(
                {
                    "observation_id": identity,
                    "arm": ADAPTIVE_SPLINE_ARM,
                    "feedback_label": true_label,
                    "arrival_index": arrival_index,
                }
            )
            feedback.append(
                {
                    "row_type": "feedback_event",
                    "observation_id": identity,
                    "group_id": prediction["group_id"],
                    "arm": ADAPTIVE_SPLINE_ARM,
                    "seed": seed,
                    "ordering": ordering,
                    "delay": delay,
                    "prediction_index": prediction["prediction_index"],
                    "request_order": prediction["prediction_index"],
                    "arrival_index": arrival_index,
                    "reveal_probability": prediction["propensity"],
                    "true_label": true_label,
                    "feedback_label": true_label,
                    "prediction_commit_hash": prediction["prediction_commit_hash"],
                    "parent_state_hash": parent,
                    "event_hash": event_hash,
                    "child_state_hash": child,
                    "probability_source": "stored_at_prediction",
                    "prediction_time_loss": bernoulli_log_loss(
                        true_label, float(prediction["probability"])
                    ),
                    "update_count": int(receipt.get("update_admitted") is True),
                    "update_cost": {
                        "weight_loss_evaluations": 0,
                        "adaptive_expert_updates": 1,
                        "touched_coefficients": receipt.get("touched_coefficients", 0),
                        "duration_ns": 0,
                    },
                }
            )
            lineage.append(
                {
                    "row_type": "checkpoint_lineage",
                    "observation_id": identity,
                    "arm": ADAPTIVE_SPLINE_ARM,
                    "seed": seed,
                    "ordering": ordering,
                    "delay": delay,
                    "parent_state_hash": parent,
                    "event_hash": event_hash,
                    "child_state_hash": child,
                    "exactly_once": receipt.get("update_admitted") is True,
                }
            )

    for index, row in enumerate(stream):
        deliver(index)
        read_started = time.perf_counter_ns()
        identity = str(row["observation_id"])
        label = int(row["label"])
        features = row["features"]
        schedule_row = schedule[identity]
        read_duration_ns += time.perf_counter_ns() - read_started
        predict_started = time.perf_counter_ns()
        learned_row = learned.predict(identity, features, index=index, delay=delay)
        equal_row = equal.predict(identity, features, index=index, delay=delay)
        shuffle_one_row = shuffled_one.predict(identity, features, index=index, delay=delay)
        shuffle_two_row = shuffled_two.predict(identity, features, index=index, delay=delay)
        adaptive_row = adaptive.record_prediction(
            identity, features, index=index, available_at=index + delay
        )
        frozen_spline, frozen_prior = _fixed_predictions(state, features)
        arm_values = {
            LEARNED_MIXTURE: (
                learned_row["mixture_probability"],
                _controller_predictive_hash(learned),
                learned.weights,
            ),
            EQUAL_MIXTURE: (
                equal_row["mixture_probability"],
                _controller_predictive_hash(equal),
                equal.weights,
            ),
            FROZEN_SPLINE_ARM: (
                frozen_spline,
                canonical_hash(
                    {
                        "arm": FROZEN_SPLINE_ARM,
                        "seed": seed,
                        "checkpoint": state["spline_checkpoint"],
                    }
                ),
                None,
            ),
            ADAPTIVE_SPLINE_ARM: (adaptive_row["probability"], adaptive.state_hash, None),
            NO_FEEDBACK_MIXTURE: (
                frozen_prior,
                canonical_hash(
                    {
                        "arm": NO_FEEDBACK_MIXTURE,
                        "seed": seed,
                        "spline": state["spline_checkpoint"],
                        "gibbs": state["gibbs_checkpoint"],
                    }
                ),
                {name: 0.25 for name in EXPERT_NAMES},
            ),
            SHUFFLED_CONTROL_1: (
                shuffle_one_row["mixture_probability"],
                _controller_predictive_hash(shuffled_one),
                shuffled_one.weights,
            ),
            SHUFFLED_CONTROL_2: (
                shuffle_two_row["mixture_probability"],
                _controller_predictive_hash(shuffled_two),
                shuffled_two.weights,
            ),
        }
        predict_duration_ns += time.perf_counter_ns() - predict_started
        domain_changed = last_domain is not None and str(row["task_type"]) != last_domain
        for arm, (probability_value, state_hash, mixture_weights) in arm_values.items():
            probability = float(probability_value)
            proposed = shadow_action(probability, state["thresholds"])
            prediction = {
                "row_type": "prediction",
                "observation_id": identity,
                "row_key": row["row_key"],
                "group_id": row["group_id"],
                "task_type": row["task_type"],
                "arm": arm,
                "seed": seed,
                "ordering": ordering,
                "delay": delay,
                "prediction_index": index,
                "request_order": index,
                "available_at": index + delay,
                "probability": probability,
                "proposed_action": proposed,
                "deployed_action": "escalate",
                "shadow_only": True,
                "certified_safe": False,
                "state_hash_at_prediction": state_hash,
                "weight_vector": mixture_weights,
                "revealed": bool(schedule_row["revealed"]),
                "propensity": float(schedule_row["propensity"]),
                "label": label,
                "label_read_at_prediction": False,
                "prediction_before_feedback": True,
                "domain_changed": domain_changed,
                "loss": bernoulli_log_loss(label, probability),
                "brier": (probability - label) ** 2,
            }
            committed = {
                key: value
                for key, value in prediction.items()
                if key not in {"label", "loss", "brier"}
            }
            prediction["prediction_commit_hash"] = canonical_hash(committed)
            persist_started = time.perf_counter_ns()
            json.dumps(prediction, sort_keys=True, separators=(",", ":"), allow_nan=False)
            persist_duration_ns += time.perf_counter_ns() - persist_started
            predictions.append(prediction)
            prediction_lookup[(identity, arm)] = prediction
        if schedule_row["revealed"]:
            pending.append(
                {
                    "observation_id": identity,
                    "label": label,
                    "prediction_index": index,
                    "available_at": index + delay,
                }
            )
        if delay == 0:
            deliver(index)
        last_domain = str(row["task_type"])
    if pending:
        deliver(max(int(row["available_at"]) for row in pending))
    return {
        "prediction_rows": predictions,
        "feedback_rows": feedback,
        "weight_trajectory_rows": weights,
        "checkpoint_lineage": lineage,
        "future_label_reads": 0,
        "duplicate_updates": duplicate_updates,
        "latency_ns": {
            "read": read_duration_ns,
            "predict": predict_duration_ns,
            "persist": persist_duration_ns,
            "update": update_duration_ns,
        },
    }


def reduce_arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce proper scores, budget, shadow risk, and domain-change harm."""

    if not rows:
        raise ValueError("metric rows must be non-empty")
    losses = [
        float(row.get("loss", bernoulli_log_loss(int(row["label"]), float(row["probability"]))))
        for row in rows
    ]
    briers = [
        float(row.get("brier", (float(row["probability"]) - int(row["label"])) ** 2))
        for row in rows
    ]
    revealed = [row for row in rows if row["revealed"]]
    revealed_losses = [
        bernoulli_log_loss(int(row["label"]), float(row["probability"])) for row in revealed
    ]
    revealed_briers = [(float(row["probability"]) - int(row["label"])) ** 2 for row in revealed]
    selected = [row for row in rows if row["proposed_action"] != "escalate"]
    harmful = sum(
        (row["proposed_action"] == "accept" and int(row["label"]) == 0)
        or (row["proposed_action"] == "reject" and int(row["label"]) == 1)
        for row in selected
    )
    changed = [index for index, row in enumerate(rows) if row["domain_changed"]]
    ipw_loss = math.fsum(
        bernoulli_log_loss(int(row["label"]), float(row["probability"])) / float(row["propensity"])
        for row in revealed
    ) / len(rows)
    ipw_brier = math.fsum(
        (float(row["probability"]) - int(row["label"])) ** 2 / float(row["propensity"])
        for row in revealed
    ) / len(rows)
    changed_loss = float(np.mean([losses[index] for index in changed])) if changed else None
    return {
        "row_count": len(rows),
        "full_stream_log_loss": float(np.mean(losses)),
        "full_stream_brier": float(np.mean(briers)),
        "revealed_row_count": len(revealed),
        "revealed_only_log_loss": float(np.mean(revealed_losses)) if revealed_losses else None,
        "revealed_only_brier": float(np.mean(revealed_briers)) if revealed_briers else None,
        "ipw_log_loss": ipw_loss,
        "ipw_brier": ipw_brier,
        "label_cost": len(revealed) / len(rows),
        "shadow_coverage": len(selected) / len(rows),
        "selected_action_count": len(selected),
        "shadow_harmful_action_rate": harmful / len(selected) if selected else None,
        "deployment_policy": "all_escalate",
        "deployment_coverage": 0.0,
        "deployment_selected_risk": None,
        "domain_change_row_count": len(changed),
        "domain_change_log_loss": changed_loss,
        "domain_change_harm": changed_loss - float(np.mean(losses))
        if changed_loss is not None
        else None,
    }


def condition_reports(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce each arm inside each registered order-delay cell."""

    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ordering"]), int(row["delay"]), str(row["arm"]))].append(row)
    return [
        {"ordering": ordering, "delay": delay, "arm": arm, **reduce_arm_metrics(group)}
        for (ordering, delay, arm), group in sorted(grouped.items())
    ]


def synthetic_metric_rows(*, groups: int, seeds: int) -> list[JsonDict]:
    """Build complete paired cells for bootstrap and mutation tests."""

    output: list[JsonDict] = []
    for ordering in ORDERS:
        for delay in DELAYS:
            for group in range(groups):
                for seed in range(seeds):
                    baseline = 0.45 + 0.02 * ((group + seed) % 5)
                    for arm, delta in (
                        (LEARNED_MIXTURE, -0.04),
                        (FROZEN_SPLINE_ARM, 0.0),
                        (ADAPTIVE_SPLINE_ARM, 0.01),
                        (EQUAL_MIXTURE, 0.02),
                    ):
                        output.append(
                            {
                                "ordering": ordering,
                                "delay": delay,
                                "group_id": f"group-{group:04d}",
                                "prediction_index": group,
                                "seed": seed,
                                "arm": arm,
                                "loss": baseline + delta,
                            }
                        )
    return output


def paired_moving_block_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> list[JsonDict]:
    """Average seeds by group, then form simultaneous moving-block bounds."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    values_by_key: dict[tuple[str, int, str, str], dict[int, float]] = defaultdict(dict)
    positions: dict[tuple[str, int, str], int] = {}
    for row in rows:
        arm = str(row["arm"])
        if arm not in {LEARNED_MIXTURE, *PRIMARY_COMPARATORS}:
            continue
        cell_group_arm = (
            str(row["ordering"]),
            int(row["delay"]),
            str(row["group_id"]),
            arm,
        )
        values_by_key[cell_group_arm][int(row["seed"])] = float(row["loss"])
        positions[(str(row["ordering"]), int(row["delay"]), str(row["group_id"]))] = int(
            row["prediction_index"]
        )
    seed_counts = {len(value) for value in values_by_key.values()}
    expected_keys = len(ORDERS) * len(DELAYS) * len(PRIMARY_COMPARATORS)
    if len(seed_counts) != 1 or not seed_counts or min(seed_counts) <= 0:
        raise ValueError("paired bootstrap requires complete seed rows")
    fit_seed_count = next(iter(seed_counts))
    contrasts: dict[tuple[str, int, str], np.ndarray] = {}
    for ordering in ORDERS:
        for delay in DELAYS:
            groups = sorted(
                {
                    group
                    for order, cell_delay, group, _arm in values_by_key
                    if order == ordering and cell_delay == delay
                },
                key=lambda group: positions[(ordering, delay, group)],
            )
            for comparator in PRIMARY_COMPARATORS:
                deltas: list[float] = []
                for group in groups:
                    learned = values_by_key.get((ordering, delay, group, LEARNED_MIXTURE), {})
                    control = values_by_key.get((ordering, delay, group, comparator), {})
                    if set(learned) != set(control) or len(learned) != fit_seed_count:
                        raise ValueError("paired bootstrap requires complete seed rows")
                    deltas.append(
                        float(np.mean([learned[item] - control[item] for item in sorted(learned)]))
                    )
                contrasts[(ordering, delay, comparator)] = np.asarray(deltas, dtype=np.float64)
    if len(contrasts) != expected_keys or len({len(value) for value in contrasts.values()}) != 1:
        raise ValueError("paired bootstrap requires complete registered cells")
    contrast_keys = sorted(contrasts)
    group_count = len(contrasts[contrast_keys[0]])
    rng = np.random.default_rng(seed)
    output: list[JsonDict] = []
    for block_length in (BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE):
        samples = np.empty((draws, len(contrast_keys)), dtype=np.float64)
        block_count = math.ceil(group_count / block_length)
        for draw in range(draws):
            starts = rng.integers(0, group_count, size=block_count)
            indices = np.concatenate(
                [
                    (np.arange(start, start + block_length, dtype=np.int64) % group_count)
                    for start in starts
                ]
            )[:group_count]
            for contrast_index, key in enumerate(contrast_keys):
                samples[draw, contrast_index] = float(np.mean(contrasts[key][indices]))
        observed = np.asarray([float(np.mean(contrasts[key])) for key in contrast_keys])
        maximum_error = np.max(samples - observed, axis=1)
        simultaneous_margin = float(np.quantile(maximum_error, 0.95))
        for contrast_index, (ordering, delay, comparator) in enumerate(contrast_keys):
            output.append(
                {
                    "ordering": ordering,
                    "delay": delay,
                    "comparator": comparator,
                    "metric": "log_loss_delta",
                    "observed": float(observed[contrast_index]),
                    "lower": float(np.quantile(samples[:, contrast_index], 0.025)),
                    "upper": float(observed[contrast_index] + simultaneous_margin),
                    "confidence": 0.95,
                    "family_contrasts": 12,
                    "draws": draws,
                    "seed": seed,
                    "block_length": block_length,
                    "paired_source_groups": group_count,
                    "fit_seeds_averaged_before_resampling": fit_seed_count,
                }
            )
    return output


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep a machine-readable scalar separate from its explanation."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def reduce_online_value(
    reports: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
) -> JsonDict:
    """Apply every registered comparison and control as one conjunction."""

    by_key = {(str(row["ordering"]), int(row["delay"]), str(row["arm"])): row for row in reports}
    gates: list[JsonDict] = []
    for interval in intervals:
        observed = float(interval["upper"])
        gates.append(
            _gate(
                f"upper_log_loss_delta:{interval['ordering']}:{interval['delay']}:{interval['comparator']}:block{interval['block_length']}",
                "scientific_benefit",
                "<",
                0.0,
                observed,
                observed < 0.0,
                "Every simultaneous upper later-loss delta must be negative.",
            )
        )
    for ordering in ORDERS:
        for delay in DELAYS:
            learned = by_key[(ordering, delay, LEARNED_MIXTURE)]
            for comparator in PRIMARY_COMPARATORS:
                control = by_key[(ordering, delay, comparator)]
                brier_delta = float(learned["full_stream_brier"]) - float(
                    control["full_stream_brier"]
                )
                gates.append(
                    _gate(
                        f"brier_noninferiority:{ordering}:{delay}:{comparator}",
                        "scientific_benefit",
                        "<=",
                        0.001,
                        brier_delta,
                        brier_delta <= 0.001,
                        "Later Brier loss cannot be worse by more than 0.001.",
                    )
                )
                learned_harm = learned["shadow_harmful_action_rate"]
                control_harm = control["shadow_harmful_action_rate"]
                harm_passed = (
                    learned_harm is None
                    if control_harm is None
                    else learned_harm is None or float(learned_harm) <= float(control_harm)
                )
                gates.append(
                    _gate(
                        f"harmful_action_nonincrease:{ordering}:{delay}:{comparator}",
                        "safety",
                        "<=_where_defined",
                        control_harm,
                        learned_harm,
                        harm_passed,
                        "Undefined selected risks remain null; defined harm cannot increase.",
                    )
                )
                coverage_delta = float(learned["shadow_coverage"]) - float(
                    control["shadow_coverage"]
                )
                gates.append(
                    _gate(
                        f"coverage_no_loss:{ordering}:{delay}:{comparator}",
                        "scientific_benefit",
                        ">=",
                        0.0,
                        coverage_delta,
                        coverage_delta >= 0.0,
                        "Shadow coverage cannot fall relative to a primary comparator.",
                    )
                )
                label_delta = float(learned["label_cost"]) - float(control["label_cost"])
                gates.append(
                    _gate(
                        f"label_cost_no_increase:{ordering}:{delay}:{comparator}",
                        "cost",
                        "<=",
                        0.0,
                        label_delta,
                        label_delta <= 1e-12,
                        "Every arm shares the same feedback budget.",
                    )
                )
    for name in ("no_feedback_equality", "shuffled_label_control_1", "shuffled_label_control_2"):
        observed = controls.get(name)
        gates.append(
            _gate(
                name,
                "negative_control",
                "==",
                True,
                observed,
                observed is True,
                "Frozen equality and both fixed shuffled-label controls must pass.",
            )
        )
    value = int(all(row["passed"] for row in gates))
    return {
        "online_capture_complete_score": 1,
        "online_value_score": value,
        "gates": gates,
    }


def synthetic_reduction_inputs(
    *,
    passing: bool,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Build one closed gate table for boundary tests."""

    reports: list[JsonDict] = []
    for ordering in ORDERS:
        for delay in DELAYS:
            for arm in (LEARNED_MIXTURE, *PRIMARY_COMPARATORS):
                learned = arm == LEARNED_MIXTURE
                reports.append(
                    {
                        "ordering": ordering,
                        "delay": delay,
                        "arm": arm,
                        "full_stream_brier": 0.10 if learned else 0.11,
                        "shadow_harmful_action_rate": 0.01 if learned else 0.02,
                        "shadow_coverage": 0.30 if learned else 0.25,
                        "label_cost": 0.25,
                    }
                )
    intervals = [
        {
            "ordering": ordering,
            "delay": delay,
            "comparator": comparator,
            "block_length": block_length,
            "upper": -0.01 if passing else 0.0,
        }
        for block_length in (BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE)
        for ordering in ORDERS
        for delay in DELAYS
        for comparator in PRIMARY_COMPARATORS
    ]
    return (
        reports,
        intervals,
        {
            "no_feedback_equality": True,
            "shuffled_label_control_1": True,
            "shuffled_label_control_2": True,
        },
    )


def _negative_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check frozen equality and that true feedback beats fixed permutations."""

    no_feedback: dict[tuple[int, str], set[float]] = defaultdict(set)
    loss_by_cell_arm: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        if arm == NO_FEEDBACK_MIXTURE:
            no_feedback[(int(row["seed"]), str(row["group_id"]))].add(float(row["probability"]))
        if arm in {LEARNED_MIXTURE, SHUFFLED_CONTROL_1, SHUFFLED_CONTROL_2}:
            loss_by_cell_arm[(str(row["ordering"]), int(row["delay"]), arm)].append(
                float(row["loss"])
            )
    controls: JsonDict = {
        "no_feedback_equality": bool(no_feedback)
        and all(len(values) == 1 for values in no_feedback.values())
    }
    for index, arm in enumerate((SHUFFLED_CONTROL_1, SHUFFLED_CONTROL_2), 1):
        comparisons = []
        for ordering in ORDERS:
            for delay in DELAYS:
                learned = loss_by_cell_arm[(ordering, delay, LEARNED_MIXTURE)]
                shuffled = loss_by_cell_arm[(ordering, delay, arm)]
                comparisons.append(
                    bool(learned)
                    and bool(shuffled)
                    and float(np.mean(learned)) <= float(np.mean(shuffled)) + 1e-12
                )
        controls[f"shuffled_label_control_{index}"] = all(comparisons)
    return controls


def write_row_shards(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    relative_dir: Path = ROW_DIR,
    prefix: str = "events",
    max_bytes: int = MAX_SHARD_BYTES,
) -> list[JsonDict]:  # pragma: no cover - production I/O is checked by cold replay.
    """Write deterministic JSONL shards below the repository size ceiling."""

    directory = root / relative_dir
    directory.mkdir(parents=True, exist_ok=True)
    manifests: list[JsonDict] = []
    buffer: list[str] = []
    size = 0
    part = 0
    for row in rows:
        line = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        encoded = len(line.encode("utf-8"))
        if buffer and size + encoded > max_bytes:
            path = directory / f"{prefix}-{part:03d}.jsonl"
            path.write_text("".join(buffer), encoding="utf-8")
            manifests.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "rows": len(buffer),
                    "size_bytes": path.stat().st_size,
                }
            )
            buffer, size, part = [], 0, part + 1
        buffer.append(line)
        size += encoded
    if buffer:
        path = directory / f"{prefix}-{part:03d}.jsonl"
        path.write_text("".join(buffer), encoding="utf-8")
        manifests.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "rows": len(buffer),
                "size_bytes": path.stat().st_size,
            }
        )
    return manifests


def _load_row_shards(root: Path, manifests: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Rehash each shard before independent reduction."""

    output: list[JsonDict] = []
    for manifest in manifests:
        path = root / str(manifest.get("path") or "")
        if not path.is_file() or sha256_file(path) != manifest.get("sha256"):
            raise ValueError("row_shard_invalid")
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(loaded) != manifest.get("rows") or any(not isinstance(row, dict) for row in loaded):
            raise ValueError("row_shard_invalid")
        output.extend(loaded)
    return output


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute all terminal gates from hash-bound prediction rows."""

    raw = _load_row_shards(root, value.get("row_shards", []))
    predictions = [row for row in raw if row.get("row_type") == "prediction"]
    feedback = [row for row in raw if row.get("row_type") == "feedback_event"]
    lineage = [row for row in raw if row.get("row_type") == "checkpoint_lineage"]
    if not predictions or not feedback or not lineage:
        raise ValueError("required_raw_rows_missing")
    reports = condition_reports(predictions)
    intervals = paired_moving_block_intervals(
        predictions,
        draws=int(value.get("bootstrap_draws", BOOTSTRAP_DRAWS)),
        seed=int(value.get("bootstrap_seed", BOOTSTRAP_SEED)),
    )
    controls = _negative_controls(predictions)
    reduced = reduce_online_value(reports, intervals, controls)
    valid_lineage = all(row.get("exactly_once") is True for row in lineage)
    valid_feedback = all(
        row.get("probability_source") == "stored_at_prediction"
        and int(row.get("arrival_index", -1)) >= int(row.get("prediction_index", 0))
        and row.get("update_count") == 1
        for row in feedback
    )
    reduced.update(
        {
            "condition_reports": reports,
            "paired_moving_block_intervals": intervals,
            "controls": controls,
            "prediction_row_count": len(predictions),
            "feedback_row_count": len(feedback),
            "lineage_row_count": len(lineage),
            "causal_capture_valid": valid_lineage and valid_feedback,
        }
    )
    if not reduced["causal_capture_valid"]:
        reduced["online_capture_complete_score"] = 0
        reduced["online_value_score"] = 0
    return reduced


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable protocol, source, shard, reduction, and validation scope."""

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
    ):
        payload[field] = [] if field in {"phase_spans", "validation_receipts"} else 0
    return canonical_hash(payload)


def _field_principles() -> JsonDict:
    """Explain required artifact fields without wrapping their values."""

    principles = {
        field: "Record measured current-work evidence as a plain machine-readable value."
        for field in REQUIRED_FIELDS
    }
    principles.update(
        {
            "schema": "Use a versioned plain top-level experiment schema.",
            "MODEL_SPECS": "List current LLMs; use an empty list when none is invoked.",
            "model_invoked": "Separate current attempted model work from archived numeric data.",
            "rows": "Keep every registered arm, seed, order, and delay unit explicit.",
            "online_capture_complete_score": "A valid terminal capture remains useful after a null.",
            "online_value_score": "One requires all causal, cost, control, and loss gates.",
            "feedback_event_rows": "Bind request, reveal, arrival, stored loss, and commit order.",
            "weight_trajectory_rows": "Show whether the frozen option gains weight after updates.",
            "checkpoint_lineage": "Bind parent, event, and child states with exactly-once replay.",
            "shadow_decisions_only": "Adaptive probabilities inherit no static certificate.",
            "hardware_path": "Bound four-expert memory, operations, persistence, and feedback cost.",
            "promotion_score": "This milestone authorizes no automatic rollout or weight update.",
        }
    )
    return principles


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, candidate: bool) -> bool:
    """Require the frozen affected set and terminal readers for final output."""

    required = set(AFFECTED_CHECK_NAMES)
    if not candidate:
        required.update(TERMINAL_CHECK_NAMES)
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True and row.get("passed") is True and row.get("exit_code") == 0
    }
    return required.issubset(passed)


def _gate_summary(
    gates: Sequence[Mapping[str, Any]], failed_precondition: Mapping[str, Any] | None = None
) -> JsonDict:
    """Name an exact blocker while keeping valid null benefit separate."""

    if failed_precondition is not None:
        return {
            key: failed_precondition.get(key)
            for key in ("upstream", "path", "check", "field", "operator", "expected", "observed")
        }
    failed = [row["check"] for row in gates if row.get("passed") is not True]
    return {
        "blocked": False,
        "validity_failures": [
            row["check"]
            for row in gates
            if row.get("category") in {"validity", "completion"} and not row.get("passed")
        ],
        "benefit_failures": failed,
    }


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish missing external evidence as a terminal blocked record."""

    now = utc_now()
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_external_prerequisite",
        "run_date": RUN_DATE,
        "started_at_utc": now,
        "completed_at_utc": now,
        "preconditions_checked": [dict(failed)],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {"reason": "external prerequisite unavailable"},
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": BOOTSTRAP_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "row_shards": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "stopping_rule": "blocked before dependent work",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _gate_summary([], failed),
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_external_prerequisite:{failed.get('check')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "continuous_self_learning_task": True,
        "feedback_event_rows": [],
        "weight_trajectory_rows": [],
        "checkpoint_lineage": [],
        "shadow_decisions_only": True,
        "hardware_path": {},
    }
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def _build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    row_shards: Sequence[Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    bootstrap_draws: int,
    candidate: bool,
    fixture: bool,
    latency_ns: Mapping[str, int],
    source_group_count: int,
) -> JsonDict:
    """Assemble a schema-complete record and independently reduce its shards."""

    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "pending_independent_reduction",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "cpu": os.uname().machine,
            "cuda": "not_used",
            "external_device": None,
            "archived_model_events": "typed hash-bound sidecars only",
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": max(0.0, (ended_ns - started_ns) / 1e9),
        "phase_spans": [dict(row) for row in phase_spans],
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "reveal_schedule_seed": BOOTSTRAP_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shuffle_permutations": [1, 7],
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [dict(row) for row in unit_rows],
        "row_shards": [dict(row) for row in row_shards],
        "sample_size_budget": {
            "planned": len(unit_rows),
            "attempted": sum(bool(row.get("attempted")) for row in unit_rows),
            "completed": sum(bool(row.get("completed")) for row in unit_rows),
            "failed": sum(bool(row.get("failed")) for row in unit_rows),
            "censored": sum(bool(row.get("censored")) for row in unit_rows),
            "unstarted": sum(bool(row.get("unstarted")) for row in unit_rows),
            "independent_source_groups": source_group_count,
            "fit_seeds_are_not_independent_corpora": True,
            "stopping_rule": "complete all prespecified order-delay-seed-arm units",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "honest_verdict": "pending_independent_reduction",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "validation_receipts": [dict(row) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "continuous_self_learning_task": True,
        "feedback_event_rows": {"storage": "row_shards", "row_type": "feedback_event"},
        "weight_trajectory_rows": {"storage": "row_shards", "row_type": "weight_trajectory"},
        "checkpoint_lineage": {"storage": "row_shards", "row_type": "checkpoint_lineage"},
        "shadow_decisions_only": True,
        "certified_safe": False,
        "deployment_policy": "all_escalate",
        "deployment_coverage": 0.0,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "moving_block_lengths": [BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE],
        "prespecified_family": {"order_delay_cells": 4, "primary_comparisons": 3, "contrasts": 12},
        "label_budget": {
            "schedule": "uniform_eight_of_32",
            "target_fraction": 0.25,
            "partial_final_block": "floor(n/4)",
        },
        "hardware_path": {
            "expert_count": 4,
            "adaptive_expert_count": 2,
            "bounded_state": True,
            "read_duration_ns": int(latency_ns.get("read", 0)),
            "prediction_duration_ns": int(latency_ns.get("predict", 0)),
            "persistence_duration_ns": int(latency_ns.get("persist", 0)),
            "feedback_update_duration_ns": int(latency_ns.get("update", 0)),
            "feedback_cost_included": True,
            "persistence_cost_included": True,
        },
        "small_ebm_training": {
            "current_training": False,
            "source": "Exp7439 hash-bound checkpoint receipts",
            "receipt_class": "small_ebm_training",
        },
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    }
    reduction = independent_reduce(value, root=root)
    value["independent_reduction"] = reduction
    value["condition_reports"] = reduction["condition_reports"]
    value["paired_moving_block_intervals"] = reduction["paired_moving_block_intervals"]
    value["acceptance_gate_results"] = reduction["gates"]
    value["online_capture_complete_score"] = reduction["online_capture_complete_score"]
    value["online_value_score"] = reduction["online_value_score"]
    validation_ok = _validation_passed(validation_receipts, candidate=candidate)
    value["flagged_adversarial"] = not validation_ok or not reduction["causal_capture_valid"]
    if value["flagged_adversarial"]:
        value["status"] = "disqualified_validation_defect"
        value["honest_verdict"] = "disqualified_validation_defect"
        value["verdict_class"] = "disqualified"
        value["online_capture_complete_score"] = 0
        value["online_value_score"] = 0
    elif reduction["online_value_score"] == 1:
        value["status"] = "complete_positive_online_value"
        value["honest_verdict"] = "complete_positive_online_value"
        value["verdict_class"] = "positive"
    else:
        value["status"] = "complete_null_insufficient_online_benefit"
        value["honest_verdict"] = "complete_null_insufficient_online_benefit"
        value["verdict_class"] = "null"
    value["gate_check_summary"] = _gate_summary(value["acceptance_gate_results"])
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check declarations, shard bytes, reduction, receipts, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("artifact_schedule_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_invalid")
    if value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    if value.get("shadow_decisions_only") is not True or value.get("certified_safe") is not False:
        errors.append("shadow_decision_scope_invalid")
    try:
        reduced = independent_reduce(value, root=root)
    except ValueError as error:
        errors.append(str(error))
    else:
        if reduced != value.get("independent_reduction"):
            errors.append("independent_reduction_mismatch")
        if reduced.get("online_capture_complete_score") != value.get(
            "online_capture_complete_score"
        ):
            errors.append("online_capture_complete_score_mismatch")
        if reduced.get("online_value_score") != value.get("online_value_score"):
            errors.append("online_value_score_mismatch")
    if not _validation_passed(
        value.get("validation_receipts", []), candidate=value.get("candidate_artifact") is True
    ):
        errors.append("required_validation_incomplete")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def build_fixture_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build deterministic complete evidence without external corpus access."""

    rows = _fixture_all_rows(group_count=16)
    manifests = write_row_shards(
        root, rows, relative_dir=Path("fixture_rows"), prefix="fixture", max_bytes=512_000
    )
    unit_rows = _unit_rows()
    return _build_artifact(
        root=root,
        preconditions=[],
        source_hashes={},
        row_shards=manifests,
        unit_rows=unit_rows,
        validation_receipts=validation_receipts,
        phase_spans=[],
        started_at="2026-09-20T00:00:00+00:00",
        started_ns=1,
        ended_ns=2,
        bootstrap_draws=50,
        candidate=False,
        fixture=True,
        latency_ns={"read": 0, "predict": 0, "persist": 0, "update": 0},
        source_group_count=16,
    )


def _fixture_all_rows(group_count: int) -> list[JsonDict]:
    """Replay every registered cell with small deterministic numeric states."""

    base_rows = []
    for index in range(group_count):
        base_rows.append(
            {
                "observation_id": f"fixture-{index:03d}",
                "row_key": f"row-{index:03d}",
                "group_id": f"group-{index:03d}",
                "task_type": ("QA", "Summary", "Data2txt")[index % 3],
                "features": {
                    name: ((index + offset) % 11) / 10.0
                    for offset, name in enumerate(SOURCE_FEATURE_NAMES)
                },
                "label": index % 2,
            }
        )
    output: list[JsonDict] = []
    for ordering in ORDERS:
        stream = base_rows if ordering == "hash_order" else list(reversed(base_rows))
        schedule = build_uniform_reveal_schedule(
            stream, seed=BOOTSTRAP_SEED + ORDERS.index(ordering)
        )
        for delay in DELAYS:
            for seed in TRAINING_SEEDS:
                result = replay_cell(
                    build_fixture_states(seed=seed),
                    stream,
                    schedule,
                    ordering=ordering,
                    delay=delay,
                    seed=seed,
                )
                output.extend(result["prediction_rows"])
                output.extend(result["feedback_rows"])
                output.extend(result["weight_trajectory_rows"])
                output.extend(result["checkpoint_lineage"])
    return output


def _unit_rows() -> list[JsonDict]:
    """Enumerate every prespecified arm, seed, order, and delay unit."""

    return [
        {
            "arm": arm,
            "seed": seed,
            "ordering": ordering,
            "delay": delay,
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "status": "completed",
        }
        for ordering in ORDERS
        for delay in DELAYS
        for seed in TRAINING_SEEDS
        for arm in ARMS
    ]


def _load_initial_states(
    root: Path, decisions: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[JsonDict]:  # pragma: no cover - production input path.
    """Load the exact ten Exp7439 spline and Gibbs fit checkpoints."""

    policies = decisions["policy_certificates"]
    calibration_by_head = {
        str(row["head"]): deepcopy(row["calibration"])
        for row in policies
        if row.get("policy_kind") == "tuned"
    }
    spline_policy = next(
        row["frozen_policy"]
        for row in policies
        if row.get("head") == "sparse_spline_49" and row.get("policy_kind") == "tuned"
    )
    payloads: dict[tuple[str, int], JsonDict] = {}
    source_rows: dict[tuple[str, int], JsonDict] = {}
    for row in decisions["checkpoint_manifest"]:
        head = str(row["head"])
        seed = int(row["seed"])
        if head not in {"gibbs_6_4_1", "sparse_spline_49"}:
            continue
        path = root / str(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("checkpoint identity changed after precondition check")
        payloads[(head, seed)] = _load_object(path)
        source_rows[(head, seed)] = deepcopy(dict(row))
    return [
        {
            "seed": seed,
            "spline_checkpoint": payloads[("sparse_spline_49", seed)]["checkpoint"],
            "gibbs_checkpoint": payloads[("gibbs_6_4_1", seed)]["checkpoint"],
            "spline_calibration": calibration_by_head["sparse_spline_49"],
            "gibbs_calibration": calibration_by_head["gibbs_6_4_1"],
            "thresholds": {
                "accept_threshold": spline_policy["accept_threshold"],
                "reject_threshold": spline_policy["reject_threshold"],
            },
            "eta": protocol["eta"],
            "fixed_share": protocol["fixed_share"],
            "source_checkpoints": [
                source_rows[("sparse_spline_49", seed)],
                source_rows[("gibbs_6_4_1", seed)],
            ],
        }
        for seed in TRAINING_SEEDS
    ]


def _load_streams(
    root: Path,
) -> dict[str, list[JsonDict]]:  # pragma: no cover - production input path.
    """Reload sealed predictors, then authorize archived labels after ordering."""

    reloaded = reload_corpus(root / CORPUS_DIR)
    readers = ProtocolReaders(reloaded)
    predictors = readers.read_predictors("prospective_stream")
    evaluators = readers.read_evaluators("prospective_stream", EVALUATOR_TOKEN)
    evaluator_by_key = {str(row["row_key"]): row for row in evaluators}
    streams = build_streams(predictors)
    output: dict[str, list[JsonDict]] = {}
    for ordering, rows in streams.items():
        joined: list[JsonDict] = []
        for row in rows:
            evaluator = evaluator_by_key[str(row["row_key"])]
            joined.append({**deepcopy(dict(row)), "label": int(evaluator["primary_label"])})
        output[ordering] = joined
    if {len(rows) for rows in output.values()} != {753}:
        raise ValueError("authenticated prospective stream must contain 753 source groups")
    return output


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover - runtime receipt.
    """Close one phase with monotonic offsets and completed units."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - run_started,
        "ended_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": f"{phase}:{units}",
    }


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - E2E subprocess plan.
    """Build fresh replay, reduction, adversarial, and strict row checks."""

    common = (".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
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
                (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    ".venv/bin/python",
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


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared entrypoint E2E.
    """Authenticate, replay, validate, independently reduce, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, loaded = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    progress(run_started, "load", "before_model_load", substrate="no_model_load")
    phase_started = time.monotonic()
    states = _load_initial_states(root, loaded["decisions"], loaded["protocol"])
    streams = _load_streams(root)
    spans.append(_span("load", phase_started, run_started, 753))
    progress(run_started, "load", "after_model_load", completed=753)

    progress(run_started, "replay", "before_benchmark", planned=20)
    phase_started = time.monotonic()
    all_rows: list[JsonDict] = []
    latency = {"read": 0, "predict": 0, "persist": 0, "update": 0}
    completed = 0
    last_heartbeat = time.monotonic()
    for order_index, ordering in enumerate(ORDERS):
        stream = streams[ordering]
        schedule = build_uniform_reveal_schedule(
            stream, seed=BOOTSTRAP_SEED + order_index, block_size=BLOCK_SIZE
        )
        for delay in DELAYS:
            for state in states:
                result = replay_cell(
                    state,
                    stream,
                    schedule,
                    ordering=ordering,
                    delay=delay,
                    seed=int(state["seed"]),
                )
                for name in (
                    "prediction_rows",
                    "feedback_rows",
                    "weight_trajectory_rows",
                    "checkpoint_lineage",
                ):
                    all_rows.extend(result[name])
                for key in latency:
                    latency[key] += int(result["latency_ns"][key])
                completed += 1
                progress(
                    run_started,
                    "replay",
                    "unit_complete",
                    completed=completed,
                    planned=20,
                    ordering=ordering,
                    delay=delay,
                    seed=state["seed"],
                )
                if time.monotonic() - last_heartbeat >= 60:
                    progress(run_started, "replay", "heartbeat", completed=completed, planned=20)
                    last_heartbeat = time.monotonic()
    spans.append(_span("replay", phase_started, run_started, completed))
    progress(run_started, "replay", "after_benchmark", completed=completed)

    progress(run_started, "persist", "before_subprocess", rows=len(all_rows))
    phase_started = time.monotonic()
    row_shards = write_row_shards(root, all_rows)
    for manifest in row_shards:
        source_hashes[manifest["path"]] = {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "original_flagged_adversarial": None,
        }
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "original_flagged_adversarial": None,
        }
    spans.append(_span("persist", phase_started, run_started, len(row_shards)))
    progress(run_started, "persist", "after_subprocess", shards=len(row_shards))

    private_root = Path(tempfile.mkdtemp(prefix="exp7440-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
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
    unit_rows = _unit_rows()
    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        row_shards=row_shards,
        unit_rows=unit_rows,
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        bootstrap_draws=BOOTSTRAP_DRAWS,
        candidate=True,
        fixture=False,
        latency_ns=latency,
        source_group_count=753,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_commands, log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        row_shards=row_shards,
        unit_rows=unit_rows,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        bootstrap_draws=BOOTSTRAP_DRAWS,
        candidate=False,
        fixture=False,
        latency_ns=latency,
        source_group_count=753,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and fresh-process artifact reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the declared experiment or one strict fresh-process reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value, root=root) if value else ["artifact_unreadable"]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        try:
            reduced = independent_reduce(value, root=root)
        except (KeyError, TypeError, ValueError) as error:
            print(json.dumps({"error": str(error)}, sort_keys=True), flush=True)
            return 1
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return 0
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin module CLI.
    raise SystemExit(main())
