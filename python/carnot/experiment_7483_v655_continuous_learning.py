"""Measure controlled prequential learning on sealed V655 source labels.

The experiment reuses the qualified local residual and importance anchor. It
loads no language model. Historical Qwen readouts are immutable input rows.

Spec: REQ-KAN-7483 and SCENARIO-KAN-7483-01 through -09.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
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
from carnot.experiment_7468_v654_residual_learner import fit_local_knots
from carnot.experiment_7481_v655_typed_calibration import (
    aggregate_native_rows,
    project_predictor_features,
    select_temperature,
)
from carnot.experiment_7482_v655_importance_anchor import (
    COEFFICIENTS_PER_INPUT,
    FEATURE_COUNT,
    ImportanceAnchorHead,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7483-continuous-learning"
SCHEMA = "carnot.exp7483.v655.continuous_learning.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7483_v655_continuous_learning.json")
RAW_DIR = Path("results/raw/experiment_7483_v655_continuous_learning")
MODULE_PATH = Path("python/carnot/experiment_7483_v655_continuous_learning.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7483_v655_continuous_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7483_v655_continuous_learning.py")
SPEC_PATH = Path("openspec/capabilities/kan/spec.md")

FIT_ARTIFACT = Path("results/experiment_7479_v655_source_fit_capture.json")
EVAL_ARTIFACT = Path("results/experiment_7480_v655_source_eval_capture.json")
CALIBRATION_ARTIFACT = Path("results/experiment_7481_v655_typed_calibration.json")
ANCHOR_ARTIFACT = Path("results/experiment_7482_v655_importance_anchor.json")
RESIDUAL_ARTIFACT = Path("results/experiment_7468_v654_residual_learner.json")
HISTORICAL_CAPSTONE = Path("results/experiment_7474_v654_capstone.json")
HISTORICAL_NULL = Path("results/experiment_7454_v653_continuous_learning.json")
COHORT_PREDICTORS = Path("results/raw/experiment_7462_v654_option_protocol/cohort_predictors.jsonl")
FIT_RAW_DIR = Path("results/raw/experiment_7479_v655_source_fit_capture")
EVAL_RAW_DIR = Path("results/raw/experiment_7480_v655_source_eval_capture")

MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "numpy_float64_controlled_prequential_residual_learning"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
MAX_NUMERIC_SECONDS = 1_500.0
TEMPERATURE_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
ORDER_SEEDS = (748_301, 748_302, 748_303, 748_304, 748_305)
AUDIT_SEEDS = (748_311, 748_312, 748_313, 748_314, 748_315)
INITIAL_STATE_SEEDS = (655_101, 655_102, 655_103, 655_104, 655_105)
DELAYS = (0, 8)
AUDIT_PROBABILITY = 0.5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7_483_401
RETENTION_BOOTSTRAP_SEED = 7_483_402
LEARNING_RATE_CANDIDATES = (0.005, 0.01, 0.02)
ANCHOR_STRENGTH_CANDIDATES = (0.25, 1.0, 4.0)
REPLAY_BUFFER_SIZE = 8
FEATURE_INDICES = (0, 1, 2, 3)
ARMS = (
    "frozen",
    "affine",
    "unanchored_residual",
    "uniform_anchor",
    "importance_anchor",
    "shuffled_feedback",
    "zero_rate",
)
RESIDUAL_ARMS = ARMS[2:]
COMPARATORS = ("frozen", "affine", "unanchored_residual")

INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "forward_calls_attempted": 0,
    "forward_calls_completed": 0,
    "forward_calls_failed": 0,
    "forward_calls_cancelled": 0,
    "forward_calls_in_flight": 0,
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
    "status",
    "run_date",
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
    "online_complete_score",
    "online_benefit_score",
    "feedback_ledger",
    "retention_rows",
    "service_cost_rows",
    "small_ebm_training",
)


def _stable_unit(seed: int, *parts: str) -> float:
    """Map label-free identity fields to a reproducible value in ``[0, 1)``."""

    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _sigmoid(value: float) -> float:
    """Return a stable scalar logistic value."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(probability: float) -> float:
    """Convert a clipped probability to finite log odds."""

    clipped = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def _brier(probability: float, label: int) -> float:
    """Return one binary Brier loss."""

    return (float(probability) - int(label)) ** 2


def protocol() -> JsonDict:
    """Return the frozen replay and inference contract."""

    return {
        "schema": "carnot.exp7483.protocol.v1",
        "description": "controlled prequential replay, not natural chronological deployment",
        "baseline": "temperature_calibrated_native_readout",
        "selection_roles": ["training", "calibration_tuning"],
        "heldout_selection_allowed": False,
        "online_role": "online",
        "retention_role": "internal_test",
        "order_seeds": list(ORDER_SEEDS),
        "audit_seeds": list(AUDIT_SEEDS),
        "initial_state_seeds": list(INITIAL_STATE_SEEDS),
        "delays": list(DELAYS),
        "label_audit_probability": AUDIT_PROBABILITY,
        "arms": list(ARMS),
        "identical_feedback_opportunities": True,
        "learning_rate_candidates": list(LEARNING_RATE_CANDIDATES),
        "anchor_strength_candidates": list(ANCHOR_STRENGTH_CANDIDATES),
        "feature_indices": list(FEATURE_INDICES),
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "retention_bootstrap_seed": RETENTION_BOOTSTRAP_SEED,
        "replicate_unit": "mean_within_group_before_inference",
        "minimum_independent_online_groups": 120,
        "effect_size_min": 0.01,
        "holm_family": list(COMPARATORS),
        "retention_upper_delta_max": 0.01,
        "numeric_budget_s": MAX_NUMERIC_SECONDS,
        "generator_weights_frozen": True,
        "qwen_features_frozen": True,
    }


def infer_source_family(source_text: str) -> str:
    """Classify the three RAGTruth source shapes without reading a label."""

    try:
        value = json.loads(source_text)
    except json.JSONDecodeError:
        return "article"
    if isinstance(value, Mapping) and {"passages", "question"} <= set(value):
        return "qa"
    if isinstance(value, Mapping) and ("attributes" in value or "address" in value):
        return "review"
    return "article"


def build_stream_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[JsonDict]:
    """Order source-family blocks and their members from identity fields only."""

    families: dict[str, list[JsonDict]] = defaultdict(list)
    for source in rows:
        row = deepcopy(dict(source))
        family = str(row.get("source_family") or "unknown")
        families[family].append(row)
    family_order = sorted(families, key=lambda family: (_stable_unit(seed, family), family))
    output: list[JsonDict] = []
    for family in family_order:
        output.extend(
            sorted(
                families[family],
                key=lambda row: (
                    _stable_unit(seed, family, str(row.get("group_id") or "")),
                    str(row.get("group_id") or ""),
                ),
            )
        )
    return output


def audit_mask(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, bool]:
    """Select feedback with fixed propensity from group identity only."""

    return {
        str(row["group_id"]): _stable_unit(seed, str(row["group_id"])) < AUDIT_PROBABILITY
        for row in rows
    }


class AffineHead:
    """Update one slope and intercept above the frozen calibrated log odds."""

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)
        self.slope = 1.0
        self.intercept = 0.0
        self.update_count = 0

    @property
    def state_hash(self) -> str:
        """Bind both affine parameters and acknowledged update count."""

        return canonical_hash(
            {
                "learning_rate": self.learning_rate,
                "slope": self.slope,
                "intercept": self.intercept,
                "update_count": self.update_count,
            }
        )

    def predict(self, frozen_probability: float) -> float:
        """Apply the current affine correction to frozen log odds."""

        return _sigmoid(self.slope * _logit(frozen_probability) + self.intercept)

    def update(self, frozen_probability: float, label: int) -> None:
        """Take one clipped Brier-gradient step from one revealed label."""

        base_logit = _logit(frozen_probability)
        probability = self.predict(frozen_probability)
        common = 2.0 * (probability - int(label)) * probability * (1.0 - probability)
        gradient = np.asarray([common * base_logit, common], dtype=np.float64)
        norm = float(np.linalg.norm(gradient))
        gradient *= min(1.0, 5.0 / max(norm, np.finfo(np.float64).tiny))
        self.slope -= self.learning_rate * float(gradient[0])
        self.intercept -= self.learning_rate * float(gradient[1])
        self.update_count += 1


def resolve_baseline_state(
    artifact: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Load Exp7481 temperature or reproduce it from fit-role rows only."""

    states = artifact.get("initial_states_for_exp7483")
    valid = (
        artifact.get("calibration_complete_score") == 1
        and artifact.get("flagged_adversarial") is False
        and isinstance(states, list)
        and bool(states)
        and all(state.get("base_arm") == "temperature" for state in states)
    )
    if valid:
        temperatures = {float(state["base_temperature"]) for state in states}
        if len(temperatures) == 1:
            return {
                "branch": "exp7481_frozen_initial_state",
                "temperature": temperatures.pop(),
                "roles_consumed": ["training", "calibration_tuning"],
                "initial_state_count": len(states),
                "initial_state_hash": canonical_hash(states),
            }
    if any(row.get("role") != "training" for row in training):
        raise ValueError("baseline_training_role_invalid")
    if any(row.get("role") != "calibration_tuning" for row in calibration):
        raise ValueError("baseline_calibration_role_invalid")
    logits = [float(row["native_log_odds"]) for row in calibration]
    labels = [int(row["label"]) for row in calibration]
    temperature = select_temperature(logits, labels)
    return {
        "branch": "independent_fit_from_fit_shard",
        "temperature": temperature,
        "roles_consumed": ["training", "calibration_tuning"],
        "initial_state_count": len(INITIAL_STATE_SEEDS),
        "initial_state_hash": canonical_hash(
            {
                "training": [row["group_id"] for row in training],
                "calibration": [row["group_id"] for row in calibration],
                "temperature": temperature,
            }
        ),
    }


def _frozen_probability(row: Mapping[str, Any], temperature: float) -> float:
    """Apply only the frozen scalar temperature to one native readout."""

    return _sigmoid(float(row["native_log_odds"]) / float(temperature))


def _feature_array(row: Mapping[str, Any]) -> np.ndarray:
    """Validate the four-feature view used by the historical residual head."""

    values = np.asarray(row["features"], dtype=np.float64)
    if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("residual_feature_shape_invalid")
    return values


def _prepare_head(
    training: Sequence[Mapping[str, Any]], *, seed: int, temperature: float
) -> tuple[ImportanceAnchorHead, list[JsonDict]]:
    """Estimate importance from training labels without moving the zero residual."""

    matrix = np.asarray([_feature_array(row) for row in training], dtype=np.float64)
    head = ImportanceAnchorHead(
        fit_local_knots(matrix),
        np.zeros((FEATURE_COUNT, COEFFICIENTS_PER_INPUT), dtype=np.float64),
        0.0,
        anchor_mode="none",
        anchor_lambda=0.0,
        learning_rate=0.0,
    )
    replay_rows: list[JsonDict] = []
    for index, row in enumerate(training):
        features = _feature_array(row)
        frozen = _frozen_probability(row, temperature)
        event_id = f"training-{seed}-{index}"
        head.seal_prediction(
            event_id=event_id,
            source_version="v655-training",
            prediction_time=index,
            reveal_time=index,
            features=features,
            frozen_probability=frozen,
        )
        head.apply_feedback(event_id, label=int(row["label"]), visible_at=index)
        if len(replay_rows) < REPLAY_BUFFER_SIZE:
            replay_rows.append(
                {
                    "features": features.tolist(),
                    "label": int(row["label"]),
                    "frozen_probability": frozen,
                }
            )
    head.consolidate()
    return head, replay_rows


def _clone_head(
    prepared: ImportanceAnchorHead,
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    anchor_lambda: float,
    learning_rate: float,
) -> ImportanceAnchorHead:
    """Clone one frozen initial residual state for a registered arm."""

    clone = ImportanceAnchorHead(
        prepared.knots,
        prepared.coefficients,
        prepared.bias,
        anchor_mode=mode,
        anchor_lambda=anchor_lambda,
        learning_rate=learning_rate,
        residual_bound=prepared.residual_bound,
        replay_rows=replay_rows,
    )
    clone.importance_accumulator = prepared.importance_accumulator.copy()
    clone.importance_observations = prepared.importance_observations
    clone.importance = prepared.importance.copy()
    clone.reference_coefficients = prepared.reference_coefficients.copy()
    clone.consolidation_count = prepared.consolidation_count
    return clone


def _pseudo_stream_score(
    training: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
    learning_rate: float,
    anchor_lambda: float,
) -> float:
    """Score one importance setting on a calibration-only pseudo-stream."""

    prepared, replay = _prepare_head(training, seed=INITIAL_STATE_SEEDS[0], temperature=temperature)
    head = _clone_head(
        prepared,
        replay,
        mode="importance",
        anchor_lambda=anchor_lambda,
        learning_rate=learning_rate,
    )
    losses = []
    for index, row in enumerate(calibration):
        features = _feature_array(row)
        frozen = _frozen_probability(row, temperature)
        probability = float(
            head.predict(features, frozen_probability=frozen)["residual_prediction"]
        )
        losses.append(_brier(probability, int(row["label"])))
        event_id = f"calibration-{index}"
        head.seal_prediction(
            event_id=event_id,
            source_version="v655-calibration",
            prediction_time=index,
            reveal_time=index,
            features=features,
            frozen_probability=frozen,
        )
        head.apply_feedback(event_id, label=int(row["label"]), visible_at=index)
    return math.fsum(losses) / len(losses)


def select_settings(
    training: Sequence[Mapping[str, Any]], calibration: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Select rate and anchor strength from fit roles before held-out labels open."""

    baseline = resolve_baseline_state({}, training, calibration)
    temperature = float(baseline["temperature"])
    rows = []
    for learning_rate in LEARNING_RATE_CANDIDATES:
        for anchor_lambda in ANCHOR_STRENGTH_CANDIDATES:
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "anchor_lambda": anchor_lambda,
                    "prequential_brier": _pseudo_stream_score(
                        training,
                        calibration,
                        temperature=temperature,
                        learning_rate=learning_rate,
                        anchor_lambda=anchor_lambda,
                    ),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            float(row["prequential_brier"]),
            float(row["learning_rate"]),
            float(row["anchor_lambda"]),
        ),
    )
    return {
        "selection_roles": ["training", "calibration_tuning"],
        "heldout_labels_consumed": False,
        "base_temperature": temperature,
        "learning_rate": float(selected["learning_rate"]),
        "anchor_lambda": float(selected["anchor_lambda"]),
        "candidate_rows": rows,
        "selection_input_hash": canonical_hash(
            {
                "training": [dict(row) for row in training],
                "calibration": [dict(row) for row in calibration],
            }
        ),
    }


def _predict_arm(
    arm: str,
    row: Mapping[str, Any],
    temperature: float,
    affine: AffineHead,
    heads: Mapping[str, ImportanceAnchorHead],
) -> float:
    """Predict from one arm without exposing the evaluator label."""

    frozen = _frozen_probability(row, temperature)
    if arm == "frozen":
        return frozen
    if arm == "affine":
        return affine.predict(frozen)
    return float(
        heads[arm].predict(_feature_array(row), frozen_probability=frozen)["residual_prediction"]
    )


def _state_hash(
    arm: str, affine: AffineHead, heads: Mapping[str, ImportanceAnchorHead], frozen_hash: str
) -> str:
    """Return the registered state identity for one arm."""

    if arm == "frozen":
        return frozen_hash
    if arm == "affine":
        return affine.state_hash
    return heads[arm].state_hash


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Atomically write a bounded row shard and return its byte identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "size_bytes": path.stat().st_size,
    }


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load a JSON-lines shard and reject non-object rows."""

    output = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("jsonl_row_not_object")
            output.append(value)
    return output


def _retention_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    moment: str,
    order_seed: int,
    delay: int,
    temperature: float,
    affine: AffineHead,
    heads: Mapping[str, ImportanceAnchorHead],
) -> list[JsonDict]:
    """Score retention labels without passing them to an update method."""

    output = []
    for row in rows:
        for arm in ARMS:
            output.append(
                {
                    "group_id": str(row["group_id"]),
                    "source_family": str(row["source_family"]),
                    "role": "internal_test",
                    "order_seed": order_seed,
                    "delay": delay,
                    "arm": arm,
                    "moment": moment,
                    "label": int(row["label"]),
                    "probability": _predict_arm(arm, row, temperature, affine, heads),
                    "used_for_update": False,
                }
            )
    return output


def _service_rows(costs: Mapping[str, list[float]], denominator: int) -> list[JsonDict]:
    """Reduce operation timings on one common event denominator."""

    output = []
    for operation in (
        "prediction",
        "feedback_processing",
        "update",
        "replay_guard",
        "serialization",
        "fsync",
        "restart",
        "no_op",
    ):
        values = list(costs.get(operation) or [0.0])
        output.append(
            {
                "operation": operation,
                "observed_operations": len(costs.get(operation) or []),
                "denominator_events": denominator,
                "total_s": math.fsum(values),
                "mean_s_per_event": math.fsum(values) / denominator,
                "max_s": max(values),
            }
        )
    return output


def run_controlled_replay(
    training: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    online: Sequence[Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    *,
    output_dir: Path,
    order_seeds: Sequence[int] = ORDER_SEEDS,
    audit_seeds: Sequence[int] = AUDIT_SEEDS,
    delays: Sequence[int] = DELAYS,
) -> JsonDict:
    """Replay fair delayed streams and write event, retention, and state shards."""

    if len(order_seeds) != len(audit_seeds):
        raise ValueError("seed_count_mismatch")
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    temperature = float(settings["base_temperature"])
    learning_rate = float(settings["learning_rate"])
    anchor_lambda = float(settings["anchor_lambda"])
    event_rows: list[JsonDict] = []
    retention_rows: list[JsonDict] = []
    stream_rows: list[JsonDict] = []
    checkpoint_rows: list[JsonDict] = []
    crash_checks: list[JsonDict] = []
    costs: dict[str, list[float]] = defaultdict(list)
    chronology_violations = 0
    completed_streams = 0
    frozen_hash = canonical_hash({"temperature": temperature, "mutable": False})

    for replicate_index, (order_seed, audit_seed) in enumerate(
        zip(order_seeds, audit_seeds, strict=True)
    ):
        ordered = build_stream_order(online, int(order_seed))
        selected = audit_mask(ordered, int(audit_seed))
        audited_ids = [str(row["group_id"]) for row in ordered if selected[str(row["group_id"])]]
        shuffled_values = [
            int(row["label"]) for row in ordered if str(row["group_id"]) in set(audited_ids)
        ]
        rng = np.random.default_rng(int(audit_seed))
        rng.shuffle(shuffled_values)
        shuffled_labels = dict(zip(audited_ids, shuffled_values, strict=True))

        for delay in delays:
            if time.monotonic() - started > MAX_NUMERIC_SECONDS:  # pragma: no cover - wall guard.
                raise TimeoutError("numeric_budget_exceeded")
            initial_seed = INITIAL_STATE_SEEDS[replicate_index % len(INITIAL_STATE_SEEDS)]
            prepared, replay_rows = _prepare_head(
                training, seed=initial_seed, temperature=temperature
            )
            heads = {
                "unanchored_residual": _clone_head(
                    prepared,
                    replay_rows,
                    mode="none",
                    anchor_lambda=0.0,
                    learning_rate=learning_rate,
                ),
                "uniform_anchor": _clone_head(
                    prepared,
                    replay_rows,
                    mode="uniform",
                    anchor_lambda=anchor_lambda,
                    learning_rate=learning_rate,
                ),
                "importance_anchor": _clone_head(
                    prepared,
                    replay_rows,
                    mode="importance",
                    anchor_lambda=anchor_lambda,
                    learning_rate=learning_rate,
                ),
                "shuffled_feedback": _clone_head(
                    prepared,
                    replay_rows,
                    mode="importance",
                    anchor_lambda=anchor_lambda,
                    learning_rate=learning_rate,
                ),
                "zero_rate": _clone_head(
                    prepared,
                    replay_rows,
                    mode="importance",
                    anchor_lambda=anchor_lambda,
                    learning_rate=0.0,
                ),
            }
            affine = AffineHead(learning_rate)
            retention_rows.extend(
                _retention_predictions(
                    retention,
                    moment="before",
                    order_seed=int(order_seed),
                    delay=int(delay),
                    temperature=temperature,
                    affine=affine,
                    heads=heads,
                )
            )
            pending: dict[int, list[tuple[Mapping[str, Any], list[JsonDict]]]] = defaultdict(list)
            stream_events: list[JsonDict] = []
            last_update: tuple[str, int] | None = None

            def release(at_time: int) -> None:
                nonlocal chronology_violations, last_update
                for source, arm_events in pending.pop(at_time, []):
                    feedback_started = time.perf_counter()
                    group_id = str(source["group_id"])
                    revealed = selected[group_id]
                    frozen = _frozen_probability(source, temperature)
                    for event in arm_events:
                        arm = str(event["arm"])
                        before = _state_hash(arm, affine, heads, frozen_hash)
                        update_started = time.perf_counter()
                        if not revealed:
                            status = "not_selected_for_audit"
                            accepted = False
                        elif arm == "frozen":
                            status = "no_update_control"
                            accepted = False
                        elif arm == "affine":
                            affine.update(frozen, int(source["label"]))
                            status = "committed"
                            accepted = True
                        else:
                            label = (
                                shuffled_labels[group_id]
                                if arm == "shuffled_feedback"
                                else int(source["label"])
                            )
                            guard_started = time.perf_counter()
                            heads[arm].loss_and_gradient(
                                _feature_array(source), label, frozen_probability=frozen
                            )
                            costs["replay_guard"].append(time.perf_counter() - guard_started)
                            receipt = heads[arm].apply_feedback(
                                str(event["learner_event_id"]),
                                label=label,
                                visible_at=at_time,
                            )
                            status = str(receipt["status"])
                            accepted = status == "committed"
                            if accepted:
                                last_update = (str(event["learner_event_id"]), label)
                        costs["update"].append(time.perf_counter() - update_started)
                        after = _state_hash(arm, affine, heads, frozen_hash)
                        event.update(
                            {
                                "feedback_time": at_time,
                                "label_revealed": revealed,
                                "update_accepted": accepted,
                                "update_status": status,
                                "state_hash_before": before,
                                "state_hash_after": after,
                            }
                        )
                        if (
                            int(event["prediction_time"]) > at_time
                        ):  # pragma: no cover - invariant guard.
                            chronology_violations += 1
                    costs["feedback_processing"].append(time.perf_counter() - feedback_started)

            for event_time, source in enumerate(ordered):
                release(event_time)
                arm_events = []
                for arm in ARMS:
                    prediction_started = time.perf_counter()
                    state_at_prediction = _state_hash(arm, affine, heads, frozen_hash)
                    probability = _predict_arm(arm, source, temperature, affine, heads)
                    learner_event_id = (
                        f"{order_seed}-{delay}-{event_time}"
                        if arm in RESIDUAL_ARMS
                        else "not_applicable"
                    )
                    label_revealed = selected[str(source["group_id"])]
                    if arm in RESIDUAL_ARMS and label_revealed:
                        heads[arm].seal_prediction(
                            event_id=learner_event_id,
                            source_version="v655-online",
                            prediction_time=event_time,
                            reveal_time=event_time + int(delay),
                            features=_feature_array(source),
                            frozen_probability=_frozen_probability(source, temperature),
                        )
                    label_free = {
                        "group_id": str(source["group_id"]),
                        "source_family": str(source["source_family"]),
                        "order_seed": int(order_seed),
                        "audit_seed": int(audit_seed),
                        "delay": int(delay),
                        "event_time": event_time,
                        "arm": arm,
                        "probability": probability,
                        "prediction_state_hash": state_at_prediction,
                    }
                    event = {
                        **label_free,
                        "role": "online",
                        "label": int(source["label"]),
                        "prediction_time": event_time,
                        "propensity": AUDIT_PROBABILITY,
                        "feedback_due_time": event_time + int(delay),
                        "learner_event_id": learner_event_id,
                        "prediction_event_hash": canonical_hash(label_free),
                        "feedback_time": None,
                        "label_revealed": False,
                        "update_accepted": False,
                        "update_status": "pending",
                        "state_hash_before": state_at_prediction,
                        "state_hash_after": state_at_prediction,
                    }
                    costs["prediction"].append(time.perf_counter() - prediction_started)
                    arm_events.append(event)
                    event_rows.append(event)
                    stream_events.append(event)
                pending[event_time + int(delay)].append((source, arm_events))
                if delay == 0:
                    release(event_time)
                no_op_started = time.perf_counter()
                _state_hash("frozen", affine, heads, frozen_hash)
                costs["no_op"].append(time.perf_counter() - no_op_started)
            for at_time in sorted(pending):
                release(at_time)

            retention_rows.extend(
                _retention_predictions(
                    retention,
                    moment="after",
                    order_seed=int(order_seed),
                    delay=int(delay),
                    temperature=temperature,
                    affine=affine,
                    heads=heads,
                )
            )
            importance = heads["importance_anchor"]
            checkpoint_path = output_dir / "checkpoints" / f"state-{order_seed}-{delay}.json"
            serialization_started = time.perf_counter()
            json.dumps(importance._checkpoint_payload(), sort_keys=True, separators=(",", ":"))
            costs["serialization"].append(time.perf_counter() - serialization_started)
            fsync_started = time.perf_counter()
            manifest = importance.save_checkpoint(checkpoint_path)
            costs["fsync"].append(time.perf_counter() - fsync_started)
            restart_started = time.perf_counter()
            restored = ImportanceAnchorHead.load_checkpoint(checkpoint_path)
            costs["restart"].append(time.perf_counter() - restart_started)
            duplicate_equal = True
            if last_update is not None:
                event_id, label = last_update
                left = importance.apply_feedback(event_id, label=label, visible_at=len(ordered) + 8)
                right = restored.apply_feedback(event_id, label=label, visible_at=len(ordered) + 8)
                duplicate_equal = left == right
            replay_passed = (
                importance.state_hash == restored.state_hash
                and canonical_hash(importance._feedback_fingerprints)
                == canonical_hash(restored._feedback_fingerprints)
                and duplicate_equal
            )
            crash_checks.append(
                {
                    "order_seed": int(order_seed),
                    "delay": int(delay),
                    "passed": replay_passed,
                    "terminal_state_hash": importance.state_hash,
                    "acknowledgement_hash": canonical_hash(importance._feedback_fingerprints),
                }
            )
            checkpoint_rows.append(
                {
                    "order_seed": int(order_seed),
                    "delay": int(delay),
                    **manifest,
                }
            )
            for arm in ARMS:
                arm_rows = [row for row in stream_events if row["arm"] == arm]
                stream_rows.append(
                    {
                        "unit_id": f"stream:{order_seed}:{delay}:{arm}",
                        "order_seed": int(order_seed),
                        "delay": int(delay),
                        "arm": arm,
                        "attempted": True,
                        "complete": len(arm_rows) == len(ordered),
                        "failed": False,
                        "censored": False,
                        "excluded": False,
                        "group_count": len(arm_rows),
                        "revealed_label_count": sum(
                            row["label_revealed"] is True for row in arm_rows
                        ),
                        "accepted_update_count": sum(
                            row["update_accepted"] is True for row in arm_rows
                        ),
                        "brier": math.fsum(
                            _brier(float(row["probability"]), int(row["label"])) for row in arm_rows
                        )
                        / len(arm_rows),
                    }
                )
            completed_streams += 1
            print(
                f"[exp7483-evaluator] completed_streams={completed_streams} "
                f"planned_streams={len(order_seeds) * len(delays)} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )

    feedback_ledger = _write_jsonl(output_dir / "feedback-ledger.jsonl", event_rows)
    retention_shard = _write_jsonl(output_dir / "retention-rows.jsonl", retention_rows)
    checkpoint_shard = _write_jsonl(output_dir / "state-checkpoints.jsonl", checkpoint_rows)
    return {
        "rows": stream_rows,
        "feedback_ledger": feedback_ledger,
        "retention_rows": retention_shard,
        "state_checkpoints": checkpoint_shard,
        "service_cost_rows": _service_rows(costs, len(event_rows)),
        "chronology_violations": chronology_violations,
        "retention_label_uses_for_updates": 0,
        "crash_replay": {
            "passed": bool(crash_checks) and all(row["passed"] for row in crash_checks),
            "checks": crash_checks,
        },
        "numeric_elapsed_s": time.monotonic() - started,
    }


def _hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int, alpha: float
) -> JsonDict:
    """Resample source-family blocks, then groups within each sampled block."""

    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["source_family"])].append(float(row["delta"]))
    if not by_family or draws <= 0:
        raise ValueError("bootstrap_input_invalid")
    families = sorted(by_family)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_families = rng.choice(families, size=len(families), replace=True)
        values = []
        for family in selected_families:
            block = np.asarray(by_family[str(family)], dtype=np.float64)
            values.extend(rng.choice(block, size=len(block), replace=True).tolist())
        samples[draw] = float(np.mean(values))
    return {
        "draws": draws,
        "seed": seed,
        "source_family_count": len(families),
        "delta": float(np.mean([float(row["delta"]) for row in rows])),
        "upper_delta": float(np.quantile(samples, 1.0 - alpha)),
        "one_sided_p": float((1 + np.sum(samples >= 0.0)) / (draws + 1)),
        "bootstrap_means": samples.tolist(),
    }


def reduce_online_predictions(
    rows: Sequence[Mapping[str, Any]], *, draws: int, bootstrap_seed: int
) -> JsonDict:
    """Average order replicates within group before block bootstrap and Holm."""

    result: JsonDict = {"delays": {}}
    for delay in sorted({int(row["delay"]) for row in rows}):
        selected = [row for row in rows if int(row["delay"]) == delay]
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(str(row["group_id"]), str(row["arm"]))].append(row)
        averaged: dict[tuple[str, str], JsonDict] = {}
        for key, group_rows in grouped.items():
            labels = {int(row["label"]) for row in group_rows}
            families = {str(row["source_family"]) for row in group_rows}
            if len(labels) != 1 or len(families) != 1:
                raise ValueError("replicate_identity_disagreement")
            averaged[key] = {
                "group_id": key[0],
                "arm": key[1],
                "label": next(iter(labels)),
                "source_family": next(iter(families)),
                "probability": float(np.mean([float(row["probability"]) for row in group_rows])),
                "replicates": len(group_rows),
            }
        group_ids = sorted(
            group_id
            for group_id, arm in averaged
            if arm == "importance_anchor"
            and all((group_id, comparator) in averaged for comparator in COMPARATORS)
        )
        comparisons: dict[str, JsonDict] = {}
        bootstrap_rows: dict[str, JsonDict] = {}
        for comparator_index, comparator in enumerate(COMPARATORS):
            deltas = []
            for group_id in group_ids:
                candidate = averaged[(group_id, "importance_anchor")]
                control = averaged[(group_id, comparator)]
                label = int(candidate["label"])
                deltas.append(
                    {
                        "group_id": group_id,
                        "source_family": candidate["source_family"],
                        "delta": _brier(candidate["probability"], label)
                        - _brier(control["probability"], label),
                    }
                )
            bootstrap_rows[comparator] = _hierarchical_bootstrap(
                deltas,
                draws=draws,
                seed=bootstrap_seed + delay * 100 + comparator_index,
                alpha=0.05,
            )
        ordered = sorted(
            COMPARATORS,
            key=lambda name: (float(bootstrap_rows[name]["one_sided_p"]), name),
        )
        running_adjusted = 0.0
        for rank, comparator in enumerate(ordered, start=1):
            base = bootstrap_rows[comparator]
            threshold = 0.05 / (len(COMPARATORS) - rank + 1)
            means = np.asarray(base.pop("bootstrap_means"), dtype=np.float64)
            running_adjusted = max(
                running_adjusted,
                min(1.0, (len(COMPARATORS) - rank + 1) * float(base["one_sided_p"])),
            )
            comparisons[comparator] = {
                **base,
                "holm_rank": rank,
                "holm_alpha": threshold,
                "holm_adjusted_p": running_adjusted,
                "holm_upper_delta": float(np.quantile(means, 1.0 - threshold)),
            }
        importance_losses = [
            _brier(
                averaged[(group_id, "importance_anchor")]["probability"],
                int(averaged[(group_id, "importance_anchor")]["label"]),
            )
            for group_id in group_ids
        ]
        frozen_losses = [
            _brier(
                averaged[(group_id, "frozen")]["probability"],
                int(averaged[(group_id, "frozen")]["label"]),
            )
            for group_id in group_ids
        ]
        improvement = float(np.mean(frozen_losses) - np.mean(importance_losses))
        replicate_counts = {
            int(averaged[(group_id, "importance_anchor")]["replicates"]) for group_id in group_ids
        }
        result["delays"][str(delay)] = {
            "independent_group_count": len(group_ids),
            "replicates_per_group": next(iter(replicate_counts))
            if len(replicate_counts) == 1
            else None,
            "importance_brier": float(np.mean(importance_losses)),
            "frozen_brier": float(np.mean(frozen_losses)),
            "importance_improvement_vs_frozen": improvement,
            "comparisons": comparisons,
            "passed": len(group_ids) >= 120
            and improvement >= 0.01
            and all(row["holm_upper_delta"] < 0.0 for row in comparisons.values()),
        }
    return result


def reduce_retention_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int, bootstrap_seed: int
) -> JsonDict:
    """Reduce untouched before-and-after retention scores for importance anchoring."""

    output: JsonDict = {"delays": {}}
    for delay in sorted({int(row["delay"]) for row in rows}):
        selected = [
            row for row in rows if int(row["delay"]) == delay and row["arm"] == "importance_anchor"
        ]
        indexed = {
            (str(row["group_id"]), int(row["order_seed"]), str(row["moment"])): row
            for row in selected
        }
        groups = sorted({key[0] for key in indexed})
        group_deltas = []
        for group_id in groups:
            order_seeds = sorted({key[1] for key in indexed if key[0] == group_id})
            deltas = []
            family = "unknown"
            for order_seed in order_seeds:
                before = indexed[(group_id, order_seed, "before")]
                after = indexed[(group_id, order_seed, "after")]
                family = str(before["source_family"])
                label = int(before["label"])
                deltas.append(
                    _brier(float(after["probability"]), label)
                    - _brier(float(before["probability"]), label)
                )
            group_deltas.append(
                {
                    "group_id": group_id,
                    "source_family": family,
                    "delta": float(np.mean(deltas)),
                }
            )
        bootstrap = _hierarchical_bootstrap(
            group_deltas,
            draws=draws,
            seed=bootstrap_seed + delay,
            alpha=0.05,
        )
        bootstrap.pop("bootstrap_means")
        output["delays"][str(delay)] = {
            "independent_group_count": len(groups),
            "mean_brier_delta": bootstrap["delta"],
            "upper_brier_delta": bootstrap["upper_delta"],
            "passed": bootstrap["upper_delta"] <= 0.01,
        }
    output["all_delays_passed"] = bool(output["delays"]) and all(
        row["passed"] for row in output["delays"].values()
    )
    return output


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for each frozen affected and terminal check."""

    return all(
        len([row for row in receipts if row.get("name") == name and row.get("passed") is True]) == 1
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    )


def _shard_valid(reference: Mapping[str, Any]) -> bool:
    """Recompute one row-shard byte hash and row count."""

    path = Path(str(reference.get("path") or ""))
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        return False
    return len(load_jsonl(path)) == int(reference.get("rows", -1))


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute completion and benefit from raw shards and declared receipts."""

    ledger_ref = dict(value.get("feedback_ledger") or {})
    retention_ref = dict(value.get("retention_rows") or {})
    shards_valid = _shard_valid(ledger_ref) and _shard_valid(retention_ref)
    events = load_jsonl(Path(str(ledger_ref["path"]))) if shards_valid else []
    retention = load_jsonl(Path(str(retention_ref["path"]))) if shards_valid else []
    online = (
        reduce_online_predictions(
            events,
            draws=int(value.get("protocol", {}).get("bootstrap_draws", BOOTSTRAP_DRAWS)),
            bootstrap_seed=int(value.get("protocol", {}).get("bootstrap_seed", BOOTSTRAP_SEED)),
        )
        if events
        else {"delays": {}}
    )
    retention_reduction = (
        reduce_retention_rows(
            retention,
            draws=int(value.get("protocol", {}).get("bootstrap_draws", BOOTSTRAP_DRAWS)),
            bootstrap_seed=int(
                value.get("protocol", {}).get("retention_bootstrap_seed", RETENTION_BOOTSTRAP_SEED)
            ),
        )
        if retention
        else {"delays": {}, "all_delays_passed": False}
    )
    chronology_violations = sum(
        row.get("feedback_time") is None
        or int(row["prediction_time"]) > int(row["feedback_time"])
        or (row.get("update_accepted") is True and row.get("label_revealed") is not True)
        for row in events
    )
    opportunities: dict[tuple[int, int, str], set[bool]] = defaultdict(set)
    for row in events:
        opportunities[(int(row["order_seed"]), int(row["delay"]), str(row["group_id"]))].add(
            bool(row["label_revealed"])
        )
    fair_feedback = bool(opportunities) and all(
        len(values) == 1 for values in opportunities.values()
    )
    delay_rows = online.get("delays", {})
    support = all(
        isinstance(delay_rows.get(str(delay)), Mapping)
        and int(delay_rows[str(delay)].get("independent_group_count", 0)) >= 120
        for delay in DELAYS
    )
    complete = int(
        shards_valid
        and support
        and chronology_violations == 0
        and fair_feedback
        and value.get("crash_replay", {}).get("passed") is True
        and value.get("retention_label_uses_for_updates") == 0
        and _validation_passed(value.get("validation_receipts") or [])
        and all(row.get("passed") is True for row in value.get("preconditions_checked") or [])
        and value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == INVOCATION_COUNTS
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("execution_venue") == EXECUTION_VENUE
        and value.get("flagged_adversarial") is False
    )
    benefit = int(
        complete == 1
        and all(delay_rows.get(str(delay), {}).get("passed") is True for delay in DELAYS)
        and retention_reduction["all_delays_passed"] is True
    )
    return {
        "shards_valid": shards_valid,
        "chronology_violations": chronology_violations,
        "fair_feedback_opportunities": fair_feedback,
        "online_reduction": online,
        "retention_reduction": retention_reduction,
        "online_complete_score": complete,
        "online_benefit_score": benefit,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
) -> JsonDict:
    """Attach the registered failure-prevention principle to one gate."""

    principles = {
        "required_validity": "A positive scientific metric cannot excuse invalid evidence.",
        "readiness": "A valid null must not suppress an independent measurement.",
        "scientific_benefit": "A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
    }
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principles[category],
    }


def _acceptance_gates(reduction: Mapping[str, Any], value: Mapping[str, Any]) -> list[JsonDict]:
    """Keep validity, readiness, and scientific benefit independent."""

    online = reduction.get("online_reduction", {}).get("delays", {})
    retention = reduction.get("retention_reduction", {}).get("delays", {})
    service_ops = {row.get("operation") for row in value.get("service_cost_rows") or []}
    expected_ops = {
        "prediction",
        "feedback_processing",
        "update",
        "replay_guard",
        "serialization",
        "fsync",
        "restart",
        "no_op",
    }
    gates = [
        _gate(
            "raw_shards_authenticate",
            "required_validity",
            True,
            reduction.get("shards_valid"),
            "==",
            reduction.get("shards_valid") is True,
        ),
        _gate(
            "feedback_chronology",
            "required_validity",
            0,
            reduction.get("chronology_violations"),
            "==",
            reduction.get("chronology_violations") == 0,
        ),
        _gate(
            "equal_feedback_opportunities",
            "required_validity",
            True,
            reduction.get("fair_feedback_opportunities"),
            "==",
            reduction.get("fair_feedback_opportunities") is True,
        ),
        _gate(
            "crash_replay",
            "required_validity",
            True,
            value.get("crash_replay", {}).get("passed"),
            "==",
            value.get("crash_replay", {}).get("passed") is True,
        ),
        _gate(
            "online_complete_score",
            "readiness",
            1,
            reduction.get("online_complete_score"),
            "==",
            reduction.get("online_complete_score") == 1,
        ),
        _gate(
            "service_cost_operations",
            "readiness",
            sorted(expected_ops),
            sorted(service_ops),
            "superset",
            expected_ops <= service_ops,
        ),
    ]
    for delay in DELAYS:
        row = dict(online.get(str(delay)) or {})
        gates.extend(
            [
                _gate(
                    f"delay_{delay}_independent_groups",
                    "scientific_benefit",
                    120,
                    row.get("independent_group_count"),
                    ">=",
                    int(row.get("independent_group_count", 0)) >= 120,
                ),
                _gate(
                    f"delay_{delay}_brier_improvement",
                    "scientific_benefit",
                    0.01,
                    row.get("importance_improvement_vs_frozen"),
                    ">=",
                    float(row.get("importance_improvement_vs_frozen", -math.inf)) >= 0.01,
                ),
                _gate(
                    f"delay_{delay}_holm_upper_deltas",
                    "scientific_benefit",
                    0.0,
                    {
                        name: item.get("holm_upper_delta")
                        for name, item in row.get("comparisons", {}).items()
                    },
                    "all <",
                    len(row.get("comparisons", {})) == 3
                    and all(
                        item.get("holm_upper_delta", math.inf) < 0.0
                        for item in row.get("comparisons", {}).values()
                    ),
                ),
                _gate(
                    f"delay_{delay}_retention_upper_delta",
                    "scientific_benefit",
                    0.01,
                    retention.get(str(delay), {}).get("upper_brier_delta"),
                    "<=",
                    float(retention.get(str(delay), {}).get("upper_brier_delta", math.inf)) <= 0.01,
                ),
            ]
        )
    return gates


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed gate with its exact expected and observed values."""

    failed = [
        {
            "check": row["check"],
            "category": row["category"],
            "upstream": "current_terminal_candidate",
            "field_path": row["check"],
            "expected": row["expected"],
            "observed": row["observed"],
            "op": row["op"],
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "all_required_validity_passed": all(
            row["passed"] for row in gates if row["category"] == "required_validity"
        ),
        "all_readiness_passed": all(
            row["passed"] for row in gates if row["category"] == "readiness"
        ),
        "all_scientific_benefit_passed": all(
            row["passed"] for row in gates if row["category"] == "scientific_benefit"
        ),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def _field_principles() -> dict[str, str]:
    """Explain why each required terminal field exists."""

    return {
        "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
        "run_date": "Use 20260921; retain measured UTC and monotonic times with clock and process identity.",
        "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
        "MODEL_SPECS": "An empty list distinguishes numeric reducer work from current model tasks.",
        "model_invoked": "Any attempted current model call differs from archived or scripted events.",
        "invocation_counts": "Balanced attempted, complete, failed, cancelled and in-flight counts expose unfinished calls.",
        "inference_substrate": "The substrate names native readout aggregation and numeric learning without implying generation.",
        "inference_substrate_class": "The no-model-load class applies the correct evidence and duration rules.",
        "execution_venue": "Host CPU identity keeps historical board evidence separate.",
        "duration_s": "Measured current work separates numeric learning and validation without padding.",
        "phase_spans": "Timestamped flushed boundaries expose silent or unfinished operations.",
        "random_seed": "Frozen ordering, fitting, audit and bootstrap seeds prevent favorable reruns.",
        "reproducibility_checksum": "The checksum binds code, protocol, roles, model identity, shards and validation scope.",
        "source_artifact_hashes": "Exact upstream bytes and original flags preserve source history.",
        "rows": "One stream, seed and arm row retains failures and censoring while event lists stay sharded.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, excluded and unstarted units stay separate.",
        "acceptance_gate_results": "Typed gates keep validity, support and scientific benefit separate.",
        "gate_check_summary": "Each failed verdict names its exact field, expectation and observation.",
        "honest_verdict": "A complete terminal finding preserves a valid null and any real source block.",
        "verdict_class": "The closed enum prevents prose from changing machine classification.",
        "verifier_is_oracle": "False records that sealed human labels, not the acceptance reader, define truth.",
        "flagged_adversarial": "Reader flags cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits and log hashes establish scoped validation.",
        "field_principles": "Local explanations keep evidence understandable without external context.",
        "online_complete_score": "Full valid prequential evidence stays separate from improvement.",
        "online_benefit_score": "Benefit requires future prediction gains, retention and multiplicity control.",
        "feedback_ledger": "Prediction and label-release records expose leakage and unequal supervision.",
        "retention_rows": "Untouched test support measures forgetting without becoming a training oracle.",
        "service_cost_rows": "Update arithmetic, feedback and durable acknowledgement use one measured denominator.",
        "small_ebm_training": "A separate numeric receipt prevents compact fitting from posing as LLM inference.",
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind terminal content while excluding only its self-referential hash."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _finalize(value: JsonDict) -> JsonDict:
    """Derive scores, gates, verdict, and checksum from raw terminal evidence."""

    reduction = independent_reduce(value)
    value["online_reduction"] = reduction["online_reduction"]
    value["retention_reduction"] = reduction["retention_reduction"]
    value["online_complete_score"] = reduction["online_complete_score"]
    value["online_benefit_score"] = reduction["online_benefit_score"]
    gates = _acceptance_gates(reduction, value)
    value["acceptance_gate_results"] = gates
    value["gate_check_summary"] = _gate_summary(gates)
    if not value["gate_check_summary"]["all_required_validity_passed"]:
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_required_validation"
    elif value["online_complete_score"] != 1:
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "partial_retryable_incomplete_prequential_evidence"
    elif value["online_benefit_score"] == 1:
        value["verdict_class"] = "positive"
        value["honest_verdict"] = "complete_positive_importance_anchored_continuous_learning"
    else:
        value["verdict_class"] = "null"
        value["honest_verdict"] = "complete_null_no_registered_importance_anchor_benefit"
    value["status"] = value["honest_verdict"]
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_artifact(
    *,
    evidence: Mapping[str, Any],
    settings: Mapping[str, Any],
    baseline: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    device_identity: Mapping[str, Any],
) -> JsonDict:
    """Assemble one terminal candidate from hash-bound raw evidence."""

    event_count = int(evidence["feedback_ledger"]["rows"])
    independent_groups = max((int(row["group_count"]) for row in evidence["rows"]), default=0)
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "candidate",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "clock_identity": {"utc": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "process_identity": {"pid": os.getpid(), "ppid": os.getppid()},
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(device_identity)),
        "duration_s": (ended_monotonic_ns - started_monotonic_ns) / 1e9,
        "duration_components_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "numeric_work": float(evidence["numeric_elapsed_s"]),
            "validation": math.fsum(
                float(row.get("duration_s", 0.0)) for row in validation_receipts
            ),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "order": list(ORDER_SEEDS),
            "audit": list(AUDIT_SEEDS),
            "initial_state": list(INITIAL_STATE_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "retention_bootstrap": RETENTION_BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": deepcopy(list(evidence["rows"])),
        "sample_size_budget": {
            "planned": 160,
            "attempted": independent_groups,
            "complete": independent_groups,
            "failed": 0,
            "censored": 0,
            "excluded": 160 - independent_groups,
            "unstarted": 0,
            "independent_unit": "online_source_group",
            "event_rows": event_count,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "honest_verdict": "candidate",
        "verdict_class": "partial",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "online_complete_score": 0,
        "online_benefit_score": 0,
        "feedback_ledger": deepcopy(dict(evidence["feedback_ledger"])),
        "retention_rows": deepcopy(dict(evidence["retention_rows"])),
        "service_cost_rows": deepcopy(list(evidence["service_cost_rows"])),
        "state_checkpoints": deepcopy(dict(evidence["state_checkpoints"])),
        "crash_replay": deepcopy(dict(evidence["crash_replay"])),
        "retention_label_uses_for_updates": evidence["retention_label_uses_for_updates"],
        "small_ebm_training": {
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "local_cubic_residual_with_diagonal_importance",
            "selection_roles": ["training", "calibration_tuning"],
            "selected_learning_rate": settings["learning_rate"],
            "selected_anchor_lambda": settings["anchor_lambda"],
            "numeric_elapsed_s": evidence["numeric_elapsed_s"],
            "current_llm_calls": 0,
            "generator_weights_fitted": False,
        },
        "selection_receipt": deepcopy(dict(settings)),
        "baseline_provenance": deepcopy(dict(baseline)),
        "protocol": protocol(),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "capability_e2e": {
            "entrypoint_run": True,
            "fresh_process_cold_replay_required": True,
            "numbered_runtime_e2e": "not_applicable_experiment_only_change",
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_publication_authorized": False,
    }
    return _finalize(value)


def validate_artifact(value: Mapping[str, Any], *, verify_sources: bool = True) -> list[str]:
    """Cold-check identity, shards, reductions, gates, sources, and checksum."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_mismatch")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != INVOCATION_COUNTS
    ):
        errors.append("invocation_accounting_mismatch")
    if value.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_mismatch")
    for name in ("feedback_ledger", "retention_rows", "state_checkpoints"):
        reference = value.get(name)
        if not isinstance(reference, Mapping) or not _shard_valid(reference):
            errors.append(f"shard_hash_invalid:{name}")
    if not errors:
        expected = deepcopy(dict(value))
        _finalize(expected)
        for field in (
            "online_reduction",
            "retention_reduction",
            "online_complete_score",
            "online_benefit_score",
            "acceptance_gate_results",
            "gate_check_summary",
            "honest_verdict",
            "verdict_class",
            "status",
        ):
            if value.get(field) != expected.get(field):
                errors.append(f"{field}_mismatch")
    if verify_sources:
        for label, reference in dict(value.get("source_artifact_hashes") or {}).items():
            if not isinstance(reference, Mapping):
                errors.append(f"source_hash_row_invalid:{label}")
                continue
            path = Path(str(reference.get("path") or ""))
            if not path.is_file() or sha256_file(path) != reference.get("sha256"):
                errors.append(f"source_hash_invalid:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_fixture_artifact(root: Path) -> JsonDict:
    """Build deterministic synthetic evidence for mutation and cold-reader tests."""

    def rows(role: str, count: int, offset: int) -> list[JsonDict]:
        output = []
        families = ("qa", "review", "article")
        for index in range(count):
            label = (index + offset) % 2
            logit = (-1.0 if label == 0 else 1.0) + 0.05 * ((index % 5) - 2)
            output.append(
                {
                    "group_id": f"{role}-{index:03d}",
                    "source_family": families[index % 3],
                    "role": role,
                    "label": label,
                    "native_log_odds": logit,
                    "features": [logit, index / count, (index % 3) / 2.0, (index % 7) / 6.0],
                }
            )
        return output

    started = time.monotonic_ns()
    training = rows("training", 48, 0)
    calibration = rows("calibration_tuning", 30, 1)
    online = rows("online", 120, 0)
    retention = rows("internal_test", 30, 1)
    settings = select_settings(training, calibration)
    baseline = resolve_baseline_state({}, training, calibration)
    evidence = run_controlled_replay(
        training, calibration, online, retention, settings, output_dir=root / "fixture"
    )
    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": f"fixture:{name}",
            "log_sha256": canonical_hash(name),
            "duration_s": 0.0,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    preconditions = [{"check": "fixture", "expected": True, "observed": True, "passed": True}]
    ended = time.monotonic_ns()
    return build_artifact(
        evidence=evidence,
        settings=settings,
        baseline=baseline,
        preconditions=preconditions,
        source_hashes={},
        validation_receipts=receipts,
        started_at_utc="2026-09-21T00:00:00+00:00",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[],
        device_identity={"venue": "host", "numeric_device": "cpu_float64"},
    )


# Runtime orchestration stays outside unit coverage. The public entrypoint and
# fresh-process readers exercise it with real files and command receipts.
def _load_object(path: Path) -> JsonDict:  # pragma: no cover
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _source_hash(
    path: Path, artifact: Mapping[str, Any] | None = None
) -> JsonDict:  # pragma: no cover
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "original_flagged_adversarial": (artifact or {}).get("flagged_adversarial"),
        "original_verdict_class": (artifact or {}).get("verdict_class"),
    }


def _precondition(
    check: str, path: Path, field: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:  # pragma: no cover
    stat = path.stat() if path.is_file() else None
    return {
        "check": check,
        "upstream": path.as_posix(),
        "path": path.as_posix(),
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "op": "in" if isinstance(expected, list) else "==",
        "passed": bool(passed),
        "owner_uid": stat.st_uid if stat else None,
        "owner_gid": stat.st_gid if stat else None,
        "byte_size": stat.st_size if stat else None,
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict]]:  # pragma: no cover
    """Authenticate required current producers and preserved historical states."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    source_paths = (
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
        Path("python/carnot/experiment_7482_v655_importance_anchor.py"),
        Path("python/carnot/experiment_7468_v654_residual_learner.py"),
        SPEC_PATH,
        FIT_ARTIFACT,
        EVAL_ARTIFACT,
        ANCHOR_ARTIFACT,
        RESIDUAL_ARTIFACT,
        HISTORICAL_CAPSTONE,
        HISTORICAL_NULL,
        COHORT_PREDICTORS,
        FIT_RAW_DIR / "raw-logits-training.jsonl",
        FIT_RAW_DIR / "raw-logits-calibration_tuning.jsonl",
        EVAL_RAW_DIR / "raw-logits-online.jsonl",
        EVAL_RAW_DIR / "raw-logits-internal_test.jsonl",
    )
    artifact_paths = {
        FIT_ARTIFACT,
        EVAL_ARTIFACT,
        CALIBRATION_ARTIFACT,
        ANCHOR_ARTIFACT,
        RESIDUAL_ARTIFACT,
        HISTORICAL_CAPSTONE,
        HISTORICAL_NULL,
    }
    for relative in source_paths:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative,
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                available,
            )
        )
        if available:
            artifact = _load_object(path) if relative in artifact_paths else {}
            hashes[relative.as_posix()] = _source_hash(relative, artifact)
    expected = {
        FIT_ARTIFACT: ("fit_capture_ready_score", 1),
        EVAL_ARTIFACT: ("evaluation_capture_ready_score", 1),
        ANCHOR_ARTIFACT: ("importance_anchor_ready_score", 1),
        RESIDUAL_ARTIFACT: ("residual_learner_ready_score", 1),
    }
    for relative, (field, wanted) in expected.items():
        artifact = _load_object(root / relative)
        checks.append(
            _precondition(
                f"upstream:{relative.stem}:{field}",
                relative,
                field,
                wanted,
                artifact.get(field),
                artifact.get(field) == wanted,
            )
        )
        checks.append(
            _precondition(
                f"upstream:{relative.stem}:flagged",
                relative,
                "flagged_adversarial",
                False,
                artifact.get("flagged_adversarial"),
                artifact.get("flagged_adversarial") is False,
            )
        )
    calibration_path = root / CALIBRATION_ARTIFACT
    calibration = _load_object(calibration_path)
    if calibration_path.is_file():
        hashes[CALIBRATION_ARTIFACT.as_posix()] = _source_hash(CALIBRATION_ARTIFACT, calibration)
        valid = (
            calibration.get("calibration_complete_score") == 1
            and calibration.get("flagged_adversarial") is False
        )
        checks.append(
            _precondition(
                "optional_exp7481_valid_or_absent",
                CALIBRATION_ARTIFACT,
                "calibration_complete_score",
                1,
                calibration.get("calibration_complete_score"),
                valid,
            )
        )
    else:
        checks.append(
            _precondition(
                "optional_exp7481_valid_or_absent",
                CALIBRATION_ARTIFACT,
                "fallback_branch",
                "independent_fit_from_fit_shard",
                "independent_fit_from_fit_shard",
                True,
            )
        )
    capstone = _load_object(root / HISTORICAL_CAPSTONE)
    old = next(
        (
            row
            for row in capstone.get("rows", [])
            if row.get("task_id") == "exp7469-continuous-residual-learning"
        ),
        {},
    )
    checks.append(
        _precondition(
            "historical_exp7469_missing_producer",
            HISTORICAL_CAPSTONE,
            "rows[exp7469].honest_verdict",
            "blocked_missing_declared_producer_evidence",
            old.get("honest_verdict"),
            old.get("honest_verdict") == "blocked_missing_declared_producer_evidence",
        )
    )
    prior = _load_object(root / HISTORICAL_NULL)
    checks.append(
        _precondition(
            "historical_exp7454_null_preserved",
            HISTORICAL_NULL,
            "verdict_class",
            "null",
            prior.get("verdict_class"),
            prior.get("verdict_class") == "null",
        )
    )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH,
            "REQ-*",
            "REQ-KAN-7483",
            "REQ-KAN-7483" if "REQ-KAN-7483" in spec_text else None,
            "REQ-KAN-7483" in spec_text,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7483" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        _precondition(
            "current_task_not_excluded",
            Path("ops/exclusion_manifest.yaml"),
            EXPERIMENT_ID,
            False,
            excluded,
            not excluded,
        )
    )
    return checks, hashes


def _load_experiment_rows(root: Path) -> tuple[list[JsonDict], ...]:  # pragma: no cover
    predictors = load_jsonl(root / COHORT_PREDICTORS)
    by_group = {str(row["group_id"]): row for row in predictors}

    def role_rows(path: Path) -> list[JsonDict]:
        native, _controls = aggregate_native_rows(load_jsonl(root / path))
        output = []
        for row in native:
            predictor = by_group[str(row["group_id"])]
            views, _receipt = project_predictor_features(
                predictor, native_log_odds=float(row["native_log_odds"])
            )
            full = list(views["full"])
            output.append(
                {
                    "group_id": str(row["group_id"]),
                    "source_family": infer_source_family(str(predictor["source_text"])),
                    "role": str(row["role"]),
                    "label": int(row["label"]),
                    "native_log_odds": float(row["native_log_odds"]),
                    "features": [float(full[index]) for index in FEATURE_INDICES],
                }
            )
        return output

    return (
        role_rows(FIT_RAW_DIR / "raw-logits-training.jsonl"),
        role_rows(FIT_RAW_DIR / "raw-logits-calibration_tuning.jsonl"),
        role_rows(EVAL_RAW_DIR / "raw-logits-online.jsonl"),
        role_rows(EVAL_RAW_DIR / "raw-logits-internal_test.jsonl"),
    )


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7483] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _device_identity() -> JsonDict:  # pragma: no cover
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
    }


def _run_child(
    argv: Sequence[str], *, root: Path, started: float, phase: str, timeout_s: float
) -> int:  # pragma: no cover
    progress(started, phase, "before_subprocess", command=" ".join(argv))
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    process = subprocess.Popen(
        list(argv),
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(60.0):
            progress(started, phase, "subprocess_outstanding", pid=process.pid)

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    deadline = time.monotonic() + timeout_s
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if time.monotonic() > deadline:
            process.terminate()
            break
    code = process.wait()
    stop.set()
    monitor.join(timeout=1.0)
    progress(started, phase, "after_subprocess", exit_code=code)
    return int(code)


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
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
            "per_event_shards",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _evaluator_worker(
    input_path: Path, output_path: Path, output_dir: Path
) -> int:  # pragma: no cover
    payload = _load_object(input_path)
    evidence = run_controlled_replay(
        payload["training"],
        payload["calibration"],
        payload["online"],
        payload["retention"],
        payload["settings"],
        output_dir=output_dir,
    )
    atomic_json(output_path, evidence)
    return 0


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Authenticate, fit, replay in a child, validate, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes = collect_preconditions(root)
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

    progress(started, "selection", "before_benchmark")
    phase_started = time.monotonic()
    training, calibration, online, retention = _load_experiment_rows(root)
    calibration_artifact = _load_object(root / CALIBRATION_ARTIFACT)
    baseline = resolve_baseline_state(calibration_artifact, training, calibration)
    settings = select_settings(training, calibration)
    settings["base_temperature"] = baseline["temperature"]
    spans.append(
        _span(
            "selection",
            phase_started,
            started,
            len(settings["candidate_rows"]),
            "training_calibration_only",
        )
    )
    progress(started, "selection", "after_benchmark", candidates=len(settings["candidate_rows"]))

    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress(started, "evaluator", "before_benchmark", online_groups=len(online))
    phase_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exp7483-evaluator-", dir="/tmp") as temporary:
        input_path = Path(temporary) / "input.json"
        output_evidence = Path(temporary) / "output.json"
        atomic_json(
            input_path,
            {
                "training": training,
                "calibration": calibration,
                "online": online,
                "retention": retention,
                "settings": settings,
            },
        )
        code = _run_child(
            (
                sys.executable,
                "-u",
                "-m",
                "carnot.experiment_7483_v655_continuous_learning",
                "--date",
                RUN_DATE,
                "--root",
                str(root),
                "--evaluator-input",
                str(input_path),
                "--evaluator-output",
                str(output_evidence),
                "--evaluator-dir",
                str(raw_dir),
            ),
            root=root,
            started=started,
            phase="evaluator",
            timeout_s=MAX_NUMERIC_SECONDS,
        )
        if code != 0:
            raise RuntimeError(f"evaluator_subprocess_failed:{code}")
        evidence = _load_object(output_evidence)
    spans.append(
        _span(
            "evaluator",
            phase_started,
            started,
            len(evidence["rows"]),
            "controlled_prequential_replay",
        )
    )
    progress(started, "evaluator", "after_benchmark", completed=len(evidence["rows"]))

    for reference in (
        evidence["feedback_ledger"],
        evidence["retention_rows"],
        evidence["state_checkpoints"],
    ):
        relative = Path(reference["path"]).relative_to(root)
        reference["path"] = relative.as_posix()
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": reference["sha256"],
            "original_flagged_adversarial": None,
            "original_verdict_class": None,
        }
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        source_hashes[relative.as_posix()] = _source_hash(relative)

    private_root = Path(tempfile.mkdtemp(prefix="exp7483-validation-", dir="/tmp"))
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
        _span("affected_validation", phase_started, started, len(affected), "frozen_manifest")
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

    provisional = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]
    candidate = build_artifact(
        evidence=evidence,
        settings=settings,
        baseline=baseline,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *provisional],
        started_at_utc=started_utc,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        device_identity=_device_identity(),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_serialization")
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_serialization")

    progress(started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, _terminal_commands(candidate_path), log_dir=raw_dir / "validation/terminal"
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

    final = build_artifact(
        evidence=evidence,
        settings=settings,
        baseline=baseline,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        started_at_utc=started_utc,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        device_identity=_device_identity(),
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--evaluator-input", type=Path)
    parser.add_argument("--evaluator-output", type=Path)
    parser.add_argument("--evaluator-dir", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    root = args.root.resolve()
    if args.evaluator_input is not None:
        if args.evaluator_output is None or args.evaluator_dir is None:
            raise SystemExit("evaluator output and directory are required")
        return _evaluator_worker(args.evaluator_input, args.evaluator_output, args.evaluator_dir)
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
