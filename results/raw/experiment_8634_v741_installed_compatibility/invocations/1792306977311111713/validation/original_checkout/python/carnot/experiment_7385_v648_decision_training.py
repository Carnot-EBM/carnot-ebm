"""Train and evaluate the sealed V648 calibrated-decision heads.

This module consumes the Exp7382 protocol. It trains only the small 2-4-1
decision head and its two controls. The mandated text generator is not loaded
or changed. Final labels stay behind a fresh-process scoring boundary.

Spec refs: REQ-AUTO-7385 and SCENARIO-AUTO-7385-01 through
SCENARIO-AUTO-7385-05.
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
import shutil
import subprocess
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.autoresearch.calibrated_decision_benchmark import _forward_energy
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7382_v648_decision_protocol import (
    ALPHA_PER_TEST,
    ARMS,
    FAMILYWISE_ALPHA,
    SIMULTANEOUS_TEST_COUNT,
    THRESHOLD_PAIRS,
    TRAINING_SEEDS,
    exact_risk_certificate,
    typed_decision,
)
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.training.nce import nce_loss


JsonDict = dict[str, Any]
RUN_DATE = "20260918"
MILESTONE = "2026.09.648"
EXPERIMENT_ID = "exp7385-decision-training"
SCHEMA = "carnot.exp7385.v648.decision_training.v1"
SCORING_REQUEST_SCHEMA = "carnot.exp7385.trusted_score_request.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7385_v648_decision_training.json")
RAW_DIR = Path("results/raw/experiment_7385_v648_decision_training")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7385_v648_decision_training")
MODULE_PATH = Path("python/carnot/experiment_7385_v648_decision_training.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7385_v648_decision_training.py")
TEST_PATH = Path("tests/python/test_experiment_7385_v648_decision_training.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
PROTOCOL_PATH = Path("results/experiment_7382_v648_decision_protocol.json")
CORPUS_PATH = Path("data/fover_corpus_v4.json")
TRUSTED_LABEL_PATH = Path(
    "results/raw/experiment_7382_v648_decision_protocol/trusted_final_test_labels.json"
)

EXPECTED_PROTOCOL_SHA256 = "sha256:a093a1b970e0308b84fbcad96d7a5254e9b563260d8a21e16d89de19a90bb8da"
EXPECTED_PARTITION_SHA256 = (
    "sha256:c392fe22a192b1db74ef45622034bb665211165243d1ef0df63835f1eee92a18"
)
EXPECTED_CORPUS_SHA256 = "sha256:c5710308eb72575591165ad1df672086e3d91ae3270c174c409e8c1ef48725e2"
EXPECTED_TRUSTED_LABEL_SHA256 = (
    "sha256:3c120d1980ca1443b372dc45dbf945cf317f75759fb0159af1e3ec59106b8316"
)
PRIMARY_VALUE_ARM = "natural_prevalence_bernoulli_gibbs"
CONTROL_ARMS = ("training_prevalence", "l2_logistic_calibration")
LEARNED_ARMS = (
    "raw_balanced_nce_gibbs",
    "prior_corrected_nce_gibbs",
    PRIMARY_VALUE_ARM,
)
MAX_STEPS = 500
LEARNING_RATE = 0.01
L2 = 0.001
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7_382_307
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
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    Path("python/carnot/training/nce.py"),
    Path("python/carnot/verify/pcib_probe.py"),
    Path("scripts/_autoresearch_energy_recompute_worker.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    SPEC_PATH,
    Path("ops/verifier_gaps.md"),
    PROTOCOL_PATH,
    CORPUS_PATH,
    TRUSTED_LABEL_PATH,
)
V648_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so checkpoints and reductions have one identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large artifact twice."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_numeric_tree(value: Any) -> None:
    """Reject non-numeric, non-finite, or executable checkpoint payloads."""

    if isinstance(value, bool):
        raise ValueError("numeric state cannot contain booleans")
    if isinstance(value, (int, float, np.number)):
        if not math.isfinite(float(value)):
            raise ValueError("numeric state must be finite")
        return
    if isinstance(value, list):
        for item in value:
            validate_numeric_tree(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("numeric state keys must be strings")
            validate_numeric_tree(item)
        return
    raise ValueError("numeric state contains a non-numeric value")


def _membership_hash(protocol: Mapping[str, Any]) -> str:
    memberships = {
        str(row.get("group_id")): str(row.get("partition"))
        for row in protocol.get("partition_membership", [])
        if isinstance(row, Mapping)
    }
    return canonical_hash(memberships)


def _protocol_gate_rows(
    protocol: Mapping[str, Any], observed_hashes: Mapping[str, str]
) -> list[JsonDict]:
    """Return exact expected and observed values for every upstream gate."""

    manifest = protocol.get("protocol_manifest") or {}
    source_hashes = manifest.get("source_hashes") or {}
    rows: list[JsonDict] = []

    def gate(field: str, expected: Any, observed: Any) -> None:
        rows.append(
            {
                "check": f"exp7382:{field}",
                "upstream": PROTOCOL_PATH.as_posix(),
                "artifact_field": field,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )

    gate("artifact_sha256", EXPECTED_PROTOCOL_SHA256, observed_hashes.get(PROTOCOL_PATH.as_posix()))
    gate("status", "complete_decision_protocol_ready", protocol.get("status"))
    gate("verdict_class", "null", protocol.get("verdict_class"))
    gate("flagged_adversarial", False, protocol.get("flagged_adversarial"))
    gate("decision_protocol_ready_score", 1, protocol.get("decision_protocol_ready_score"))
    gate(
        "gate_check_summary.all_required_passed",
        True,
        (protocol.get("gate_check_summary") or {}).get("all_required_passed"),
    )
    gate(
        "protocol_manifest.partition_membership_sha256",
        EXPECTED_PARTITION_SHA256,
        manifest.get("partition_membership_sha256"),
    )
    gate(
        "recomputed.partition_membership_sha256",
        EXPECTED_PARTITION_SHA256,
        _membership_hash(protocol),
    )
    gate(
        "source_hashes.data/fover_corpus_v4.json",
        EXPECTED_CORPUS_SHA256,
        source_hashes.get(CORPUS_PATH.as_posix()),
    )
    gate(
        "observed.data/fover_corpus_v4.json",
        EXPECTED_CORPUS_SHA256,
        observed_hashes.get(CORPUS_PATH.as_posix()),
    )
    gate(
        "source_hashes.trusted_final_test_labels",
        EXPECTED_TRUSTED_LABEL_SHA256,
        source_hashes.get(TRUSTED_LABEL_PATH.as_posix()),
    )
    gate(
        "observed.trusted_final_test_labels",
        EXPECTED_TRUSTED_LABEL_SHA256,
        observed_hashes.get(TRUSTED_LABEL_PATH.as_posix()),
    )
    memberships = {
        str(row.get("group_id")): str(row.get("partition"))
        for row in protocol.get("partition_membership", [])
        if isinstance(row, Mapping)
    }
    expected_partition_lists = {
        name: sorted(group for group, partition in memberships.items() if partition == name)
        for name in ("training", "probability_calibration", "policy_calibration", "final_test")
    }
    gate(
        "protocol_manifest.group_ids_by_partition",
        expected_partition_lists,
        manifest.get("group_ids_by_partition"),
    )
    feature_rows = [row for row in protocol.get("feature_rows", []) if isinstance(row, Mapping)]
    feature_membership_valid = len(feature_rows) == 6548 and all(
        memberships.get(str(row.get("group_id"))) == row.get("partition") for row in feature_rows
    )
    gate("feature_rows.partition_membership", True, feature_membership_valid)
    final_labels_sealed = all(
        "label" not in row for row in feature_rows if row.get("partition") == "final_test"
    )
    gate("feature_rows.final_labels_sealed", True, final_labels_sealed)
    return rows


def authenticate_protocol(
    protocol: Mapping[str, Any], observed_hashes: Mapping[str, str]
) -> list[str]:
    """Authenticate Exp7382 without promoting its diagnostic inputs."""

    return [
        f"{row['artifact_field']}:expected={row['expected']!r}:observed={row['observed']!r}"
        for row in _protocol_gate_rows(protocol, observed_hashes)
        if row["passed"] is not True
    ]


def _features_and_labels(
    rows: Sequence[Mapping[str, Any]], *, required_partition: str
) -> tuple[np.ndarray, np.ndarray]:
    if not rows or any(row.get("partition") != required_partition for row in rows):
        raise ValueError(f"{required_partition} rows only")
    x = np.asarray(
        [[row.get("entity_uptake"), row.get("falsifiability_score")] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([row.get("label") for row in rows], dtype=np.float32)
    if x.shape != (len(rows), 2) or not np.all(np.isfinite(x)):
        raise ValueError("training features must be finite 2-vectors")
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("fitting data must contain both labels")
    return x, y


def _json_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_tree(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _gibbs_energy(params: Mapping[str, jax.Array], x: jax.Array) -> jax.Array:
    hidden = params["w1"] @ x + params["b1"]
    hidden = hidden * jax.nn.sigmoid(hidden)
    return params["w_out"] @ hidden + params["b_out"]


def _logistic_loss(params: Mapping[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    logits = x @ params["coef"] + params["bias"]
    proper = jnp.mean(jnp.logaddexp(0.0, logits) - y * logits)
    return proper + L2 * jnp.sum(params["coef"] ** 2)


def _balanced_nce_loss(
    params: Mapping[str, jax.Array], correct: jax.Array, incorrect: jax.Array
) -> jax.Array:
    objective = nce_loss(lambda value: _gibbs_energy(params, value), correct, incorrect)
    penalty = sum(jnp.sum(value**2) for key, value in params.items() if key != "b_out")
    return objective + L2 * penalty


def _bernoulli_loss(params: Mapping[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    logits = jax.vmap(lambda value: _gibbs_energy(params, value))(x)
    proper = jnp.mean(jnp.logaddexp(0.0, logits) - y * logits)
    penalty = sum(jnp.sum(value**2) for key, value in params.items() if key != "b_out")
    return proper + L2 * penalty


_LOGISTIC_VALUE_GRAD = jax.jit(jax.value_and_grad(_logistic_loss))
_NCE_VALUE_GRAD = jax.jit(jax.value_and_grad(_balanced_nce_loss))
_BERNOULLI_VALUE_GRAD = jax.jit(jax.value_and_grad(_bernoulli_loss))


def _adam_fit(
    params: Mapping[str, jax.Array],
    value_grad: Any,
    args: tuple[jax.Array, ...],
    *,
    steps: int,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], list[JsonDict]]:
    """Apply the one frozen Adam configuration and retain its real curve."""

    current = {key: jnp.asarray(value) for key, value in params.items()}
    first = {key: jnp.zeros_like(value) for key, value in current.items()}
    second = {key: jnp.zeros_like(value) for key, value in current.items()}
    curve: list[JsonDict] = []
    for update in range(steps):
        loss, gradients = value_grad(current, *args)
        curve.append({"step": update, "loss": float(loss)})
        first = {key: 0.9 * first[key] + 0.1 * gradients[key] for key in current}
        second = {key: 0.999 * second[key] + 0.001 * gradients[key] ** 2 for key in current}
        step_number = update + 1
        current = {
            key: current[key]
            - LEARNING_RATE
            * (first[key] / (1.0 - 0.9**step_number))
            / (jnp.sqrt(second[key] / (1.0 - 0.999**step_number)) + 1e-8)
            for key in current
        }
    final_loss, _ = value_grad(current, *args)
    curve.append({"step": steps, "loss": float(final_loss)})
    return current, first, second, curve


def _initial_gibbs(seed: int) -> dict[str, jax.Array]:
    model = GibbsModel(GibbsConfig(input_dim=2, hidden_dims=[4]), jax.random.PRNGKey(seed))
    weight, bias = model.layers[0]
    return {
        "w1": weight,
        "b1": bias,
        "w_out": model.output_weight,
        "b_out": jnp.asarray(model.output_bias),
    }


def train_arm(
    arm: str,
    seed: int,
    training_rows: Sequence[Mapping[str, Any]],
    *,
    steps: int = MAX_STEPS,
) -> JsonDict:
    """Fit one frozen arm on training rows and retain optimizer evidence."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if steps < 0 or steps > MAX_STEPS:
        raise ValueError(f"steps must be between zero and {MAX_STEPS}")
    x, y = _features_and_labels(training_rows, required_partition="training")
    prevalence = float(np.mean(y))
    if arm == "training_prevalence":
        weights = {"probability": prevalence}
        return {
            "arm": arm,
            "seed": seed,
            "training_partition": "training",
            "training_prevalence": prevalence,
            "weights": weights,
            "optimizer_state": {"step": 0, "m": {}, "v": {}},
            "loss_curve": [
                {
                    "step": 0,
                    "loss": float(
                        -np.mean(y * math.log(prevalence) + (1.0 - y) * math.log1p(-prevalence))
                    ),
                }
            ],
            "update_count": 0,
            "pre_weight_sha256": canonical_hash(weights),
            "post_weight_sha256": canonical_hash(weights),
            "objective": "training_prevalence_constant",
        }

    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    if arm == "l2_logistic_calibration":
        rng = np.random.default_rng(seed)
        initial = {
            "coef": jnp.asarray(rng.normal(0.0, 0.01, size=2), dtype=jnp.float32),
            "bias": jnp.asarray(0.0),
        }
        value_grad = _LOGISTIC_VALUE_GRAD
        args = (x_jax, y_jax)
        objective = "natural_prevalence_l2_log_loss"
    else:
        initial = _initial_gibbs(seed)
        if arm in {"raw_balanced_nce_gibbs", "prior_corrected_nce_gibbs"}:
            value_grad = _NCE_VALUE_GRAD
            args = (x_jax[y_jax == 0], x_jax[y_jax == 1])
            objective = "balanced_nce"
        else:
            value_grad = _BERNOULLI_VALUE_GRAD
            args = (x_jax, y_jax)
            objective = "natural_prevalence_bernoulli_log_loss"
    pre_weights = _json_tree(initial)
    fitted, first, second, curve = _adam_fit(initial, value_grad, args, steps=steps)
    weights = _json_tree(fitted)
    return {
        "arm": arm,
        "seed": seed,
        "training_partition": "training",
        "training_prevalence": prevalence,
        "weights": weights,
        "optimizer_state": {
            "step": steps,
            "m": _json_tree(first),
            "v": _json_tree(second),
        },
        "loss_curve": curve,
        "update_count": steps,
        "pre_weight_sha256": canonical_hash(pre_weights),
        "post_weight_sha256": canonical_hash(weights),
        "objective": objective,
    }


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _raw_energy(unit: Mapping[str, Any], features: Sequence[float]) -> float:
    weights = unit["weights"]
    arm = str(unit["arm"])
    if arm == "training_prevalence":
        return _logit(float(weights["probability"]))
    if arm == "l2_logistic_calibration":
        return float(np.dot(np.asarray(weights["coef"]), np.asarray(features)) + weights["bias"])
    w1 = np.asarray(weights["w1"], dtype=np.float64)
    b1 = np.asarray(weights["b1"], dtype=np.float64)
    w_out = np.asarray(weights["w_out"], dtype=np.float64)
    return _forward_energy(
        w1,
        b1,
        w_out,
        float(weights["b_out"]),
        np.asarray(features, dtype=np.float64),
    )


def _base_logit(unit: Mapping[str, Any], features: Sequence[float]) -> float:
    value = _raw_energy(unit, features)
    if unit["arm"] == "prior_corrected_nce_gibbs":
        value += _logit(float(unit["training_prevalence"]))
    return value


def fit_affine_transform(
    logits: Sequence[float], labels: Sequence[int], *, steps: int = MAX_STEPS
) -> JsonDict:
    """Fit a two-parameter Platt transform on its calibration role."""

    if len(logits) != len(labels) or not logits:
        raise ValueError("affine inputs must have one non-empty common length")
    if set(labels) != {0, 1}:
        raise ValueError("affine fitting data must contain both labels")
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("affine logits must be finite")
    slope = 1.0
    intercept = 0.0
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    curve: list[JsonDict] = []
    for update in range(steps):
        transformed = slope * values + intercept
        probabilities = np.asarray([_sigmoid(float(item)) for item in transformed])
        clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
        loss += L2 * slope * slope
        curve.append({"step": update, "loss": float(loss)})
        residual = probabilities - targets
        gradient = np.asarray(
            [float(np.mean(residual * values) + 2.0 * L2 * slope), float(np.mean(residual))]
        )
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        step_number = update + 1
        update_value = (
            LEARNING_RATE
            * (first / (1.0 - 0.9**step_number))
            / (np.sqrt(second / (1.0 - 0.999**step_number)) + 1e-8)
        )
        slope -= float(update_value[0])
        intercept -= float(update_value[1])
    transformed = slope * values + intercept
    probabilities = np.asarray([_sigmoid(float(item)) for item in transformed])
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    final_loss = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
    final_loss += L2 * slope * slope
    curve.append({"step": steps, "loss": float(final_loss)})
    return {
        "slope": slope,
        "intercept": intercept,
        "update_count": steps,
        "loss_curve": curve,
        "optimizer_state": {"step": steps, "m": first.tolist(), "v": second.tolist()},
        "fitting_partition": "probability_calibration",
    }


def _calibrated_probability(unit: Mapping[str, Any], features: Sequence[float]) -> float:
    affine = unit["affine"]
    return _sigmoid(
        float(affine["slope"]) * _base_logit(unit, features) + float(affine["intercept"])
    )


def direct_numpy_energy_check(
    state: Mapping[str, Any], development_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Compare direct GibbsModel energies with an independent NumPy forward pass."""

    if state.get("arm") not in LEARNED_ARMS:
        return {"performed": False, "passed": True, "max_abs_delta": 0.0, "rows": 0}
    weights = state["weights"]
    model = GibbsModel(GibbsConfig(input_dim=2, hidden_dims=[4]), jax.random.PRNGKey(0))
    model.layers = [(jnp.asarray(weights["w1"]), jnp.asarray(weights["b1"]))]
    model.output_weight = jnp.asarray(weights["w_out"])
    model.output_bias = jnp.asarray(weights["b_out"])
    deltas: list[float] = []
    for row in development_rows:
        features = [float(row["entity_uptake"]), float(row["falsifiability_score"])]
        direct = float(model.energy(jnp.asarray(features)))
        independent = _raw_energy(state, features)
        deltas.append(abs(direct - independent))
    maximum = max(deltas, default=0.0)
    return {
        "performed": True,
        "passed": bool(deltas) and maximum <= 1e-6,
        "max_abs_delta": maximum,
        "rows": len(deltas),
    }


def label_blind_group_representatives(
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Choose the lowest source index per group without consulting labels."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = str(row["group_id"])
        current = selected.get(group)
        if current is None or int(row["source_row_index"]) < int(current["source_row_index"]):
            selected[group] = row
    return [deepcopy(dict(selected[group])) for group in sorted(selected)]


def select_policy(
    probability_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    """Certify all registered pairs, then select by the frozen tie-break rule."""

    if not probability_rows or any(
        row.get("partition") != "policy_calibration" for row in probability_rows
    ):
        raise ValueError("policy selection requires policy-calibration rows only")
    representatives = label_blind_group_representatives(probability_rows)
    evaluated: list[JsonDict] = []
    for index, (accept_threshold, reject_threshold) in enumerate(THRESHOLD_PAIRS):
        initial = [
            typed_decision(
                float(row["probability"]),
                accept_threshold=accept_threshold,
                reject_threshold=reject_threshold,
                model_version="policy-calibration",
            )["decision"]
            for row in representatives
        ]
        accept_harm = [
            int(row["label"] == 1)
            for row, decision in zip(representatives, initial, strict=True)
            if decision == "accept"
        ]
        reject_harm = [
            int(row["label"] == 0)
            for row, decision in zip(representatives, initial, strict=True)
            if decision == "reject"
        ]
        accept_certificate = exact_risk_certificate(accept_harm, risk_budget=0.05)
        reject_certificate = exact_risk_certificate(reject_harm, risk_budget=0.10)
        final_decisions = [
            decision
            if (decision == "accept" and accept_certificate["action_enabled"])
            or (decision == "reject" and reject_certificate["action_enabled"])
            else "escalate"
            for decision in initial
        ]
        harmful = sum(
            (decision == "accept" and row["label"] == 1)
            or (decision == "reject" and row["label"] == 0)
            for row, decision in zip(representatives, final_decisions, strict=True)
        )
        correct = sum(
            (decision == "accept" and row["label"] == 0)
            or (decision == "reject" and row["label"] == 1)
            for row, decision in zip(representatives, final_decisions, strict=True)
        )
        decided = sum(decision != "escalate" for decision in final_decisions)
        evaluated.append(
            {
                "threshold_index": index,
                "accept_threshold": accept_threshold,
                "reject_threshold": reject_threshold,
                "accept_certificate": accept_certificate,
                "reject_certificate": reject_certificate,
                "coverage": decided / len(representatives),
                "utility": (correct - harmful) / len(representatives),
                "representative_groups": len(representatives),
                "simultaneous_test_count": SIMULTANEOUS_TEST_COUNT,
                "alpha_per_test": ALPHA_PER_TEST,
            }
        )
    selected = max(
        evaluated,
        key=lambda row: (
            float(row["coverage"]),
            float(row["utility"]),
            -int(row["threshold_index"]),
        ),
    )
    policy = {
        "threshold_index": selected["threshold_index"],
        "accept_threshold": selected["accept_threshold"],
        "reject_threshold": selected["reject_threshold"],
        "accept_enabled": selected["accept_certificate"]["action_enabled"],
        "reject_enabled": selected["reject_certificate"]["action_enabled"],
        "selection_partition": "policy_calibration",
        "selection_rule": "coverage_then_utility_then_registered_order",
        "simultaneous_test_count": SIMULTANEOUS_TEST_COUNT,
        "alpha_per_test": ALPHA_PER_TEST,
    }
    return evaluated, policy


def checkpoint_payload(state: Mapping[str, Any], *, partition_hash: str) -> JsonDict:
    """Create a code-free numeric checkpoint with its sealed training identity."""

    payload = {
        "schema": "carnot.exp7385.numeric_checkpoint.v1",
        "experiment_id": EXPERIMENT_ID,
        "arm": state["arm"],
        "seed": state["seed"],
        "architecture": {"input_dim": 2, "hidden_dims": [4], "output_dim": 1},
        "optimizer": {
            "name": "adam",
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "maximum_steps": MAX_STEPS,
        },
        "partition_membership_sha256": partition_hash,
        "weights": deepcopy(state["weights"]),
        "optimizer_state": deepcopy(state["optimizer_state"]),
        "update_count": state["update_count"],
        "pre_weight_sha256": state["pre_weight_sha256"],
        "post_weight_sha256": state["post_weight_sha256"],
    }
    validate_numeric_tree(payload["weights"])
    validate_numeric_tree(payload["optimizer_state"])
    return payload


def checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    """Bind every numeric checkpoint field."""

    return canonical_hash(checkpoint)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish JSON through the shared atomic writer."""

    # The local name keeps the checkpoint helper easy to test and discover.
    from carnot.experiment_7358_v646_validation_contract import atomic_json as shared_atomic_json

    shared_atomic_json(path, value)


def scoring_unit(state: Mapping[str, Any]) -> JsonDict:
    """Strip a training record down to the plain state the scorer needs."""

    return {
        "arm": state["arm"],
        "seed": state["seed"],
        "training_prevalence": state["training_prevalence"],
        "weights": deepcopy(state["weights"]),
        "affine": {
            "slope": state["affine"]["slope"],
            "intercept": state["affine"]["intercept"],
        },
        "selected_policy": deepcopy(state["selected_policy"]),
        "model_version": f"exp7385:{state['arm']}:{state['seed']}",
    }


def _unit_id(unit: Mapping[str, Any]) -> str:
    return f"{unit.get('arm')}:{unit.get('seed')}"


def validate_scoring_request(
    request: object,
    *,
    protocol_sha256: str,
    partition_hash: str,
    trusted_label_hash: str,
    expected_units: set[str],
) -> list[str]:
    """Reject malformed scoring state before the trusted label file is read."""

    if not isinstance(request, Mapping):
        return ["request_not_object"]
    errors: list[str] = []
    allowed = {
        "schema",
        "protocol_sha256",
        "partition_membership_sha256",
        "trusted_label_sha256",
        "units",
    }
    if set(request) != allowed:
        errors.append("request_keys_mismatch")
    expected_fields = {
        "schema": SCORING_REQUEST_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "partition_membership_sha256": partition_hash,
        "trusted_label_sha256": trusted_label_hash,
    }
    for field, expected in expected_fields.items():
        if request.get(field) != expected:
            errors.append(f"{field}_mismatch")
    units = request.get("units")
    if not isinstance(units, list):
        return [*errors, "units_not_list"]
    seen: list[str] = []
    unit_keys = {
        "arm",
        "seed",
        "training_prevalence",
        "weights",
        "affine",
        "selected_policy",
        "model_version",
    }
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != unit_keys:
            errors.append("unit_keys_mismatch")
            continue
        identity = _unit_id(unit)
        seen.append(identity)
        if unit.get("arm") not in ARMS or unit.get("seed") not in TRAINING_SEEDS:
            errors.append(f"unit_identity_invalid:{identity}")
        try:
            validate_numeric_tree(unit["weights"])
            validate_numeric_tree(
                {
                    "training_prevalence": unit["training_prevalence"],
                    "slope": unit["affine"]["slope"],
                    "intercept": unit["affine"]["intercept"],
                    "accept_threshold": unit["selected_policy"]["accept_threshold"],
                    "reject_threshold": unit["selected_policy"]["reject_threshold"],
                    "threshold_index": unit["selected_policy"]["threshold_index"],
                }
            )
            arm = str(unit["arm"])
            weights = unit["weights"]
            shape_invalid = (
                (
                    arm == "training_prevalence"
                    and (
                        set(weights) != {"probability"}
                        or np.asarray(weights["probability"]).shape != ()
                    )
                )
                or (
                    arm == "l2_logistic_calibration"
                    and (
                        set(weights) != {"coef", "bias"}
                        or np.asarray(weights["coef"]).shape != (2,)
                    )
                )
                or (
                    arm in LEARNED_ARMS
                    and (
                        set(weights) != {"w1", "b1", "w_out", "b_out"}
                        or np.asarray(weights["w1"]).shape != (4, 2)
                        or np.asarray(weights["b1"]).shape != (4,)
                        or np.asarray(weights["w_out"]).shape != (4,)
                    )
                )
            )
            if shape_invalid:
                errors.append(f"weight_shape_invalid:{identity}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"unit_numeric_invalid:{identity}:{exc}")
    if len(seen) != len(set(seen)):
        errors.append("duplicate_units")
    if set(seen) != expected_units:
        errors.append("unit_set_mismatch")
    return list(dict.fromkeys(errors))


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive = np.asarray(
        [score for label, score in zip(labels, scores, strict=True) if label == 1]
    )
    negative = np.asarray(
        [score for label, score in zip(labels, scores, strict=True) if label == 0]
    )
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both labels")
    wins = sum(
        float(np.sum(value > negative)) + 0.5 * float(np.sum(value == negative))
        for value in positive
    )
    return wins / (len(positive) * len(negative))


def _binary_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise ValueError("PR-AUC requires a positive label")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _decision_for_policy(
    probability: float, policy: Mapping[str, Any], model_version: str
) -> JsonDict:
    result = typed_decision(
        probability,
        accept_threshold=float(policy["accept_threshold"]),
        reject_threshold=float(policy["reject_threshold"]),
        model_version=model_version,
    )
    if result["decision"] == "accept" and policy.get("accept_enabled") is not True:
        result["decision"] = "escalate"
        result["reason"] = "accept_action_uncertified_escalation"
    if result["decision"] == "reject" and policy.get("reject_enabled") is not True:
        result["decision"] = "escalate"
        result["reason"] = "reject_action_uncertified_escalation"
    return result


def score_final_rows(
    units: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    trusted_labels: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Score final rows after exact label identity has been authenticated."""

    features = [row for row in feature_rows if row.get("partition") == "final_test"]
    label_map = {
        (int(row["source_row_index"]), str(row["group_id"])): int(row["label"])
        for row in trusted_labels
    }
    feature_keys = {(int(row["source_row_index"]), str(row["group_id"])) for row in features}
    if feature_keys != set(label_map):
        raise ValueError("final label identity does not match feature identity")
    all_rows: list[JsonDict] = []
    summaries: dict[str, JsonDict] = {}
    for unit in units:
        started = time.monotonic()
        unit_rows: list[JsonDict] = []
        for feature in features:
            key = (int(feature["source_row_index"]), str(feature["group_id"]))
            label = label_map[key]
            vector = [float(feature["entity_uptake"]), float(feature["falsifiability_score"])]
            energy = _raw_energy(unit, vector)
            probability = _calibrated_probability(unit, vector)
            action = _decision_for_policy(
                probability, unit["selected_policy"], str(unit["model_version"])
            )
            clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
            decision = str(action["decision"])
            harmful = (
                label == 1 if decision == "accept" else label == 0 if decision == "reject" else None
            )
            correctness = (
                label == 0 if decision == "accept" else label == 1 if decision == "reject" else None
            )
            unit_rows.append(
                {
                    "unit_id": _unit_id(unit),
                    "arm": unit["arm"],
                    "seed": unit["seed"],
                    "source_row_index": feature["source_row_index"],
                    "group_id": feature["group_id"],
                    "label": label,
                    "raw_energy": energy,
                    "probability": probability,
                    "decision": decision,
                    "confidence_correct": action["confidence_correct"],
                    "reason": action["reason"],
                    "brier_contribution": (probability - label) ** 2,
                    "log_loss_contribution": -(
                        label * math.log(clipped) + (1 - label) * math.log1p(-clipped)
                    ),
                    "correctness": correctness,
                    "risk": {"harmful_action": harmful, "action": decision},
                    "measured_cost": {
                        "current_llm_calls": 0,
                        "energy_forward_parameters": 17 if unit["arm"] in LEARNED_ARMS else 3,
                        "cpu_scoring_duration_s": 0.0,
                    },
                }
            )
        elapsed = time.monotonic() - started
        per_row = elapsed / len(unit_rows) if unit_rows else 0.0
        for row in unit_rows:
            row["measured_cost"]["cpu_scoring_duration_s"] = per_row
        labels = [int(row["label"]) for row in unit_rows]
        probabilities = [float(row["probability"]) for row in unit_rows]
        accepts = [row for row in unit_rows if row["decision"] == "accept"]
        rejects = [row for row in unit_rows if row["decision"] == "reject"]
        summaries[_unit_id(unit)] = {
            "arm": unit["arm"],
            "seed": unit["seed"],
            "effective_groups": len({row["group_id"] for row in unit_rows}),
            "rows": len(unit_rows),
            "prevalence": sum(labels) / len(labels),
            "brier": float(np.mean([row["brier_contribution"] for row in unit_rows])),
            "log_loss": float(np.mean([row["log_loss_contribution"] for row in unit_rows])),
            "auroc": _binary_auroc(labels, probabilities),
            "pr_auc": _binary_pr_auc(labels, probabilities),
            "coverage": (len(accepts) + len(rejects)) / len(unit_rows),
            "utility": sum(
                1 if row["correctness"] is True else -1 if row["correctness"] is False else 0
                for row in unit_rows
            )
            / len(unit_rows),
            "incorrect_accept_risk": (
                sum(row["label"] == 1 for row in accepts) / len(accepts) if accepts else None
            ),
            "correct_reject_risk": (
                sum(row["label"] == 0 for row in rejects) / len(rejects) if rejects else None
            ),
            "accept_count": len(accepts),
            "reject_count": len(rejects),
            "escalate_count": len(unit_rows) - len(accepts) - len(rejects),
            "scoring_duration_s": elapsed,
        }
        all_rows.extend(unit_rows)
    return all_rows, summaries


def _mean_interval(values: np.ndarray) -> JsonDict:
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def paired_group_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, dict[str, JsonDict]]:
    """Bootstrap paired group deltas after averaging seeds within each arm."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), str(row["group_id"]))].append(row)
    groups = sorted({group for _, group in grouped})
    if not groups:
        raise ValueError("paired intervals require rows")
    arm_vectors: dict[str, dict[str, np.ndarray]] = {}
    for arm in (*LEARNED_ARMS, *CONTROL_ARMS):
        if any((arm, group) not in grouped for group in groups):
            continue
        arm_vectors[arm] = {
            "brier": np.asarray(
                [
                    np.mean([float(row["brier_contribution"]) for row in grouped[(arm, group)]])
                    for group in groups
                ]
            ),
            "log_loss": np.asarray(
                [
                    np.mean([float(row["log_loss_contribution"]) for row in grouped[(arm, group)]])
                    for group in groups
                ]
            ),
            "coverage": np.asarray(
                [
                    np.mean([row["decision"] != "escalate" for row in grouped[(arm, group)]])
                    for group in groups
                ]
            ),
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    output: dict[str, dict[str, JsonDict]] = {}
    for learned in LEARNED_ARMS:
        if learned not in arm_vectors:
            continue
        output[learned] = {}
        for control in CONTROL_ARMS:
            if control not in arm_vectors:
                continue
            comparisons: JsonDict = {}
            for metric in ("brier", "log_loss", "coverage"):
                delta = arm_vectors[learned][metric] - arm_vectors[control][metric]
                comparisons[f"{metric}_delta"] = _mean_interval(np.mean(delta[indices], axis=1))
            comparisons["effective_groups"] = len(groups)
            comparisons["draws"] = draws
            comparisons["seed"] = seed
            output[learned][control] = comparisons
    return output


def reduce_calibration_value(
    *,
    paired_intervals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics: Mapping[str, Mapping[str, Any]],
    policy_certified: bool,
) -> JsonDict:
    """Reduce the pre-registered primary conjunction without selecting an arm."""

    primary = metrics.get(PRIMARY_VALUE_ARM) or {}
    comparisons = paired_intervals.get(PRIMARY_VALUE_ARM) or {}
    brier = all(
        float((comparisons.get(control) or {}).get("brier_delta", {}).get("ci95", [math.inf])[-1])
        < 0.0
        for control in CONTROL_ARMS
    )
    log_loss = all(
        float(primary.get("log_loss", math.inf))
        <= float((metrics.get(control) or {}).get("log_loss", -math.inf))
        for control in CONTROL_ARMS
    )
    accept_risk = primary.get("incorrect_accept_risk")
    reject_risk = primary.get("correct_reject_risk")
    observed_risk = (accept_risk is None or float(accept_risk) <= 0.05) and (
        reject_risk is None or float(reject_risk) <= 0.10
    )
    coverage = float(primary.get("coverage", 0.0)) >= 0.25
    logistic = comparisons.get("l2_logistic_calibration") or {}
    coverage_ci = (logistic.get("coverage_delta") or {}).get("ci95", [-math.inf])
    no_coverage_loss = bool(coverage_ci) and float(coverage_ci[0]) >= 0.0
    checks = {
        "brier_ci_below_both_controls": brier,
        "non_worse_mean_log_loss": log_loss,
        "policy_certified": policy_certified,
        "observed_action_risk_within_budget": observed_risk,
        "coverage_at_least_0_25": coverage,
        "no_paired_coverage_loss_vs_logistic": no_coverage_loss,
    }
    passed = all(checks.values())
    return {"checks": checks, "passed": passed, "calibration_value_score": int(passed)}


def _aggregate_arm_metrics(unit_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Average seed metrics while retaining action counts and nullable risks."""

    output: dict[str, JsonDict] = {}
    for arm in ARMS:
        rows = [row for row in unit_metrics.values() if row.get("arm") == arm]
        if not rows:
            continue
        accepts = sum(int(row["accept_count"]) for row in rows)
        rejects = sum(int(row["reject_count"]) for row in rows)
        incorrect_accepts = sum(
            float(row["incorrect_accept_risk"]) * int(row["accept_count"])
            for row in rows
            if row["incorrect_accept_risk"] is not None
        )
        incorrect_rejects = sum(
            float(row["correct_reject_risk"]) * int(row["reject_count"])
            for row in rows
            if row["correct_reject_risk"] is not None
        )
        output[arm] = {
            "seeds": len(rows),
            "effective_groups": rows[0]["effective_groups"],
            "prevalence": float(np.mean([row["prevalence"] for row in rows])),
            "brier": float(np.mean([row["brier"] for row in rows])),
            "log_loss": float(np.mean([row["log_loss"] for row in rows])),
            "auroc": float(np.mean([row["auroc"] for row in rows])),
            "pr_auc": float(np.mean([row["pr_auc"] for row in rows])),
            "coverage": float(np.mean([row["coverage"] for row in rows])),
            "utility": float(np.mean([row["utility"] for row in rows])),
            "incorrect_accept_risk": incorrect_accepts / accepts if accepts else None,
            "correct_reject_risk": incorrect_rejects / rejects if rejects else None,
            "accept_count": accepts,
            "reject_count": rejects,
            "escalate_count": sum(int(row["escalate_count"]) for row in rows),
        }
    return output


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain every ordinary top-level field without wrapping its value."""

    specific = {
        "schema": "Version this record and keep ordinary experiment identity fields.",
        "status": "Use a terminal state only after actual work and required validation.",
        "run_date": "Use 20260918 and retain actual start and end UTC timestamps.",
        "preconditions_checked": "Authenticate exact inputs and producer gates before dependent work.",
        "MODEL_SPECS": "List current LLM work; this experiment performs none.",
        "model_invoked": "Set true for any attempted current LLM load or generation.",
        "invocation_counts": "Retain attempted, completed, failed, cancelled, and live current LLM counts.",
        "inference_substrate": "Describe actual CPU and JAX work, device identity, and host-process lease.",
        "inference_substrate_class": "Use the closed no_model_load class without duration padding.",
        "execution_venue": "Use exactly host and keep device details in inference_substrate.",
        "duration_s": "Measure monotonic elapsed time without invented delay.",
        "phase_spans": "Retain measured read, build, load, generate, evaluate, validate, and write spans.",
        "random_seed": "Freeze training, bootstrap, and replay seeds before outcomes.",
        "reproducibility_checksum": "Bind exact code, settings, protocol, sources, checkpoints, and raw rows.",
        "source_artifact_hashes": "Hash exact producer and data paths while preserving historical provenance.",
        "rows": "Retain every final group, arm, and seed outcome behind each claim.",
        "sample_size_budget": "Separate planned, attempted, completed, censored, and unstarted units.",
        "acceptance_gate_results": "Separate required validation, safety, completion, and scientific efficacy.",
        "gate_check_summary": "Name each failed gate with its path, expected value, and observed value.",
        "verifier_is_oracle": "State that archived labels define truth for this bounded evaluation.",
        "honest_verdict": "Use a complete verdict for finished positive or null science.",
        "verdict_class": "Use the closed terminal class and reserve partial for retryable owned work.",
        "flagged_adversarial": "A critical independent finding disqualifies readiness and promotion.",
        "validation_receipts": "Retain executed arguments, environment, scope, exit, duration, and log hash.",
        "repository_health": "Keep dated unrelated health findings outside required affected checks.",
        "field_principles": "Explain every output field without wrapping ordinary values.",
        "promotion_score": "Remain zero because no automatic rollout or publication is authorized.",
        "decision_capture_complete_score": "Require every sealed final row, checkpoint, and required check, including a null.",
        "calibration_value_score": "Require the complete registered primary Brier, log-loss, risk, and coverage conjunction.",
        "checkpoint_manifest": "List numeric weights, seeds, budgets, partition hash, architecture, and byte hashes.",
        "small_ebm_training": "Record actual small-head updates and CPU/JAX work apart from LLM calls.",
        "calibration_metrics": "Report proper scores, discrimination, prevalence, and typed-policy outcomes.",
        "policy_certificate_rows": "Retain each arm, seed, threshold, action count, error count, bound, and scope.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable scientific evidence while excluding wall-clock receipts."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "protocol_identity",
        "training_runs",
        "checkpoint_manifest",
        "rows",
        "calibration_metrics",
        "paired_group_intervals",
        "policy_certificate_rows",
        "calibration_value_reduction",
        "decision_capture_complete_score",
        "calibration_value_score",
    )
    return canonical_hash({field: artifact.get(field) for field in fields})


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute row completeness, validation, safety, and value from raw fields."""

    budget = artifact.get("sample_size_budget") or {}
    rows = artifact.get("rows") or []
    planned = int(budget.get("planned_final_rows") or 0)
    row_complete = planned > 0 and len(rows) == planned
    if row_complete:
        identities = {
            (row.get("arm"), row.get("seed"), row.get("group_id"))
            for row in rows
            if isinstance(row, Mapping)
        }
        row_complete = len(identities) == planned and all(
            {
                "label",
                "raw_energy",
                "probability",
                "decision",
                "brier_contribution",
                "log_loss_contribution",
                "correctness",
                "risk",
                "measured_cost",
            }
            <= set(row)
            for row in rows
        )
    checkpoints = artifact.get("checkpoint_manifest") or []
    checkpoint_complete = len(checkpoints) == int(budget.get("planned_fits") or 0) > 0
    receipts = [row for row in artifact.get("validation_receipts", []) if isinstance(row, Mapping)]
    required_names = set(validation_scope.REQUIRED_CHECK_NAMES) | {
        "trusted_final_scoring",
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    present = {str(row.get("name")) for row in receipts if row.get("required") is True}
    validation_complete = required_names <= present and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in receipts
        if row.get("name") in required_names
    )
    safe = artifact.get("flagged_adversarial") is False
    capture = int(row_complete and checkpoint_complete and validation_complete and safe)
    value_reduction = artifact.get("calibration_value_reduction") or {}
    value = int(capture == 1 and value_reduction.get("passed") is True)
    return {
        "row_completeness_passed": row_complete,
        "checkpoint_completeness_passed": checkpoint_complete,
        "required_validation_passed": validation_complete,
        "safety_passed": safe,
        "decision_capture_complete_score": capture,
        "calibration_value_score": value,
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, operator: str = "=="
) -> JsonDict:
    if operator == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    else:
        passed = observed == expected
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "operator": operator,
        "passed": passed,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") != "scientific_efficacy"]
    scientific = [row for row in failures if row.get("category") == "scientific_efficacy"]
    return {
        "all_required_passed": not required,
        "failed_required_count": len(required),
        "first_required_failure": required[0] if required else None,
        "failed_scientific_gate_count": len(scientific),
        "first_scientific_failure": scientific[0] if scientific else None,
    }


def build_fixture_artifact() -> JsonDict:
    """Build a complete-null artifact small enough for mutation unit tests."""

    rows = [
        {
            "arm": arm,
            "seed": seed,
            "group_id": "fixture-group",
            "label": 0,
            "raw_energy": 0.0,
            "probability": 0.1,
            "decision": "escalate",
            "brier_contribution": 0.01,
            "log_loss_contribution": -math.log(0.9),
            "correctness": None,
            "risk": {"harmful_action": None},
            "measured_cost": {"current_llm_calls": 0, "cpu_scoring_duration_s": 0.001},
        }
        for arm in ARMS
        for seed in TRAINING_SEEDS
    ]
    checkpoints = [
        {"arm": arm, "seed": seed, "path": f"fixture/{arm}-{seed}.json", "sha256": "sha256:fixture"}
        for arm in ARMS
        for seed in TRAINING_SEEDS
    ]
    receipts = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in (
            *validation_scope.REQUIRED_CHECK_NAMES,
            "trusted_final_scoring",
            "independent_reducer",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        )
    ]
    value_reduction = {
        "checks": {"fixture_scientific_null": False},
        "passed": False,
        "calibration_value_score": 0,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": "complete_decision_training_null",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "completed_at_utc": "2026-09-18T00:00:01+00:00",
        "preconditions_checked": [{"check": "fixture", "passed": True}],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": {
            "value": "host CPU JAX small-head training and numeric scoring",
            "backend": "cpu",
            "resource_lease": "host_process_only",
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 1.0,
        "phase_spans": [],
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "paired_group_bootstrap_seed": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "historical_inference_sidecars": [],
        "protocol_identity": {
            "artifact_sha256": EXPECTED_PROTOCOL_SHA256,
            "partition_membership_sha256": EXPECTED_PARTITION_SHA256,
        },
        "rows": rows,
        "sample_size_budget": {
            "planned_fits": 25,
            "attempted_fits": 25,
            "completed_fits": 25,
            "censored_fits": 0,
            "unstarted_fits": 0,
            "planned_final_rows": 25,
            "completed_final_rows": 25,
            "maximum_optimizer_steps_per_fit": MAX_STEPS,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_required_passed": True,
            "failed_required_count": 0,
            "first_required_failure": None,
            "failed_scientific_gate_count": 1,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null_calibrated_decision_head_no_registered_value",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "repository_health": {
            "status": "not_assessed_beyond_required_checks",
            "affects_required_checks": False,
        },
        "field_principles": {},
        "promotion_score": 0,
        "decision_capture_complete_score": 1,
        "calibration_value_score": 0,
        "checkpoint_manifest": checkpoints,
        "small_ebm_training": {"performed": True, "optimizer_steps_completed": 0, "backend": "cpu"},
        "calibration_metrics": {},
        "policy_certificate_rows": [],
        "training_runs": [],
        "paired_group_intervals": {},
        "calibration_value_reduction": value_reduction,
        "independent_reduction": {},
        "evidence_scope": "fixture_reused_archive_not_external",
        "production_defaults_changed": False,
        "standing_conductor_benchmark_changed": False,
        "active_research_roadmap_changed": False,
    }
    artifact["independent_reduction"] = independent_reduce(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, str]
) -> JsonDict:
    """Publish an external prerequisite failure without dependent work."""

    failed = next(
        (deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True), None
    )
    artifact = build_fixture_artifact()
    artifact.update(
        {
            "status": "blocked_decision_training_precondition",
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
            "source_artifact_hashes": dict(source_hashes),
            "rows": [],
            "sample_size_budget": {
                "planned_fits": 25,
                "attempted_fits": 0,
                "completed_fits": 0,
                "censored_fits": 25,
                "unstarted_fits": 0,
                "planned_final_rows": 0,
                "completed_final_rows": 0,
                "stopping_rule": "Stop before dependent work on an ineligible external prerequisite.",
            },
            "acceptance_gate_results": [],
            "gate_check_summary": {
                "all_required_passed": False,
                "failed_required_count": sum(
                    row.get("passed") is not True for row in preconditions
                ),
                "first_required_failure": failed,
                "failed_scientific_gate_count": 0,
                "first_scientific_failure": None,
            },
            "honest_verdict": "blocked_external_decision_protocol_precondition",
            "verdict_class": "blocked",
            "validation_receipts": [],
            "decision_capture_complete_score": 0,
            "calibration_value_score": 0,
            "checkpoint_manifest": [],
            "training_runs": [],
            "calibration_metrics": {},
            "policy_certificate_rows": [],
            "paired_group_intervals": {},
            "calibration_value_reduction": {
                "checks": {},
                "passed": False,
                "calibration_value_score": 0,
            },
            "independent_reduction": {
                "row_completeness_passed": False,
                "checkpoint_completeness_passed": False,
                "required_validation_passed": False,
                "safety_passed": True,
                "decision_capture_complete_score": 0,
                "calibration_value_score": 0,
            },
        }
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, reductions, scores, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    identity = (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("substrate_class_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") or artifact.get("validation_receipts"):
            errors.append("blocked_artifact_has_dependent_work")
        if (artifact.get("gate_check_summary") or {}).get("first_required_failure") is None:
            errors.append("blocked_gate_summary_missing")
        if artifact.get("decision_capture_complete_score") != 0:
            errors.append("blocked_capture_nonzero")
    else:
        reduced = independent_reduce(artifact)
        if not reduced["row_completeness_passed"]:
            errors.append("row_completeness_mismatch")
        if artifact.get("independent_reduction") != reduced:
            errors.append("independent_reduction_mismatch")
        if (
            artifact.get("decision_capture_complete_score")
            != reduced["decision_capture_complete_score"]
        ):
            errors.append("capture_reduction_mismatch")
        if artifact.get("calibration_value_score") != reduced["calibration_value_score"]:
            errors.append("calibration_value_reduction_mismatch")
    if artifact.get("flagged_adversarial") is True and any(
        artifact.get(field) != 0
        for field in (
            "decision_capture_complete_score",
            "calibration_value_score",
            "promotion_score",
        )
    ):
        errors.append("adversarial_scores_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _utc_now() -> str:  # pragma: no cover - real execution boundary.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7385] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, *, performed: bool = True
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "performed": performed,
    }


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any
) -> JsonDict:  # pragma: no cover
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _load_object(path: Path) -> JsonDict:  # pragma: no cover
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:  # pragma: no cover
    """Authenticate all exact inputs, structured gates, exclusions, and CPU resources."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)
    protocol = _load_object(root / PROTOCOL_PATH)
    checks.extend(_protocol_gate_rows(protocol, hashes))
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-AUTO-7385",
            "REQ-AUTO-7385" if "REQ-AUTO-7385" in spec_text else None,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7385" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    manifest = protocol.get("protocol_manifest") or {}
    checks.extend(
        (
            _precondition(
                "frozen_arms",
                PROTOCOL_PATH.as_posix(),
                "protocol_manifest.arms",
                list(ARMS),
                manifest.get("arms"),
            ),
            _precondition(
                "frozen_seeds",
                PROTOCOL_PATH.as_posix(),
                "protocol_manifest.seeds",
                list(TRAINING_SEEDS),
                manifest.get("seeds"),
            ),
            _precondition(
                "frozen_optimizer_steps",
                PROTOCOL_PATH.as_posix(),
                "protocol_manifest.optimizer.max_steps",
                MAX_STEPS,
                (manifest.get("optimizer") or {}).get("max_steps"),
            ),
            _precondition(
                "cpu_jax_backend",
                "current_process",
                "jax.default_backend",
                "cpu",
                jax.default_backend(),
            ),
            _precondition(
                "live_execution_guard",
                "current_process",
                "CARNOT_FORCE_LIVE",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
            ),
        )
    )
    free_bytes = shutil.disk_usage(root).free
    checks.append(
        {
            "check": "checkpoint_disk_capacity",
            "upstream": str(root),
            "artifact_field": "free_bytes",
            "expected": ">=52428800",
            "observed": free_bytes,
            "passed": free_bytes >= 50 * 1024 * 1024,
        }
    )
    return checks, hashes, protocol


def _calibration_diagnostic(
    unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, calibrated: bool
) -> JsonDict:
    labels = [int(row["label"]) for row in rows]
    probabilities = []
    for row in rows:
        features = [float(row["entity_uptake"]), float(row["falsifiability_score"])]
        probability = (
            _calibrated_probability(unit, features)
            if calibrated
            else _sigmoid(_base_logit(unit, features))
        )
        probabilities.append(probability)
    clipped = np.clip(np.asarray(probabilities), 1e-15, 1.0 - 1e-15)
    targets = np.asarray(labels)
    return {
        "partition": "probability_calibration",
        "brier": float(np.mean((clipped - targets) ** 2)),
        "log_loss": float(
            -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
        ),
        "auroc": _binary_auroc(labels, probabilities),
        "prevalence": float(np.mean(targets)),
        "rows": len(rows),
    }


def _fit_all_units(
    protocol: Mapping[str, Any], repo_root: Path, started: float
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Fit, calibrate, certify, and checkpoint all 25 frozen units."""

    feature_rows = [dict(row) for row in protocol["feature_rows"]]
    training = [row for row in feature_rows if row.get("partition") == "training"]
    probability = [row for row in feature_rows if row.get("partition") == "probability_calibration"]
    policy = [row for row in feature_rows if row.get("partition") == "policy_calibration"]
    checkpoint_root = repo_root / CHECKPOINT_DIR
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    states: list[JsonDict] = []
    manifest_rows: list[JsonDict] = []
    certificate_rows: list[JsonDict] = []
    total = len(ARMS) * len(TRAINING_SEEDS)
    completed = 0
    for arm in ARMS:
        for seed in TRAINING_SEEDS:
            unit_started = time.monotonic()
            _progress(
                started,
                "evaluate",
                "before_small_head_fit",
                arm=arm,
                seed=seed,
                unit=f"{completed + 1}/{total}",
            )
            state = train_arm(arm, seed, training, steps=MAX_STEPS)
            logits = [
                _base_logit(
                    state,
                    [float(row["entity_uptake"]), float(row["falsifiability_score"])],
                )
                for row in probability
            ]
            affine = fit_affine_transform(
                logits, [int(row["label"]) for row in probability], steps=MAX_STEPS
            )
            state["affine"] = affine
            state["calibration_metrics_pre_affine"] = _calibration_diagnostic(
                state, probability, calibrated=False
            )
            state["calibration_metrics_post_affine"] = _calibration_diagnostic(
                state, probability, calibrated=True
            )
            policy_probabilities = [
                {
                    **row,
                    "probability": _calibrated_probability(
                        state,
                        [float(row["entity_uptake"]), float(row["falsifiability_score"])],
                    ),
                }
                for row in policy
            ]
            threshold_rows, selected = select_policy(policy_probabilities)
            state["selected_policy"] = selected
            state["energy_recomputation_check"] = direct_numpy_energy_check(state, probability[:16])
            state["fit_duration_s"] = time.monotonic() - unit_started
            checkpoint = checkpoint_payload(state, partition_hash=EXPECTED_PARTITION_SHA256)
            checkpoint_path = checkpoint_root / f"{arm}_{seed}.json"
            atomic_json(checkpoint_path, checkpoint)
            checkpoint_sha = sha256_file(checkpoint_path)
            state["checkpoint_path"] = checkpoint_path.relative_to(repo_root).as_posix()
            state["checkpoint_sha256"] = checkpoint_sha
            manifest_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": state["checkpoint_path"],
                    "sha256": checkpoint_sha,
                    "architecture": {"input_dim": 2, "hidden_dims": [4], "output_dim": 1},
                    "optimizer_budget": MAX_STEPS,
                    "updates_completed": state["update_count"],
                    "partition_membership_sha256": EXPECTED_PARTITION_SHA256,
                    "pre_weight_sha256": state["pre_weight_sha256"],
                    "post_weight_sha256": state["post_weight_sha256"],
                    "numeric_weights": deepcopy(state["weights"]),
                    "optimizer_state": deepcopy(state["optimizer_state"]),
                    "executable_payload": False,
                }
            )
            for threshold in threshold_rows:
                for action, certificate_key in (
                    ("accept", "accept_certificate"),
                    ("reject", "reject_certificate"),
                ):
                    certificate = threshold[certificate_key]
                    certificate_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "action": action,
                            "threshold_index": threshold["threshold_index"],
                            "accept_threshold": threshold["accept_threshold"],
                            "reject_threshold": threshold["reject_threshold"],
                            "selected_policy": threshold["threshold_index"]
                            == selected["threshold_index"],
                            "selected_group_count": certificate["selected_groups"],
                            "error_count": certificate["harmful_outcomes"],
                            "upper_risk_bound": certificate["upper_risk_bound"],
                            "risk_budget": certificate["risk_budget"],
                            "certified": certificate["certified"],
                            "action_enabled": certificate["action_enabled"],
                            "alpha_familywise": FAMILYWISE_ALPHA,
                            "alpha_per_test": ALPHA_PER_TEST,
                            "certificate_scope": "policy_calibration_one_label_blind_representative_per_group_bonferroni_250",
                        }
                    )
            states.append(state)
            completed += 1
            _progress(
                started,
                "evaluate",
                "after_small_head_fit",
                arm=arm,
                seed=seed,
                updates=state["update_count"],
                unit=f"{completed}/{total}",
                unit_elapsed_s=f"{time.monotonic() - unit_started:.3f}",
            )
    return states, manifest_rows, certificate_rows


def _scoring_request(states: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCORING_REQUEST_SCHEMA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "partition_membership_sha256": EXPECTED_PARTITION_SHA256,
        "trusted_label_sha256": EXPECTED_TRUSTED_LABEL_SHA256,
        "units": [scoring_unit(state) for state in states],
    }


def trusted_score_main(request_path: Path, output_path: Path) -> int:  # pragma: no cover
    """Score final labels once after strict plain-state validation."""

    started = time.monotonic()
    print("[exp7385-trusted-score] phase=validate event=start", flush=True)
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    label_path = REPO_ROOT / TRUSTED_LABEL_PATH
    request = _load_object(request_path)
    protocol = _load_object(protocol_path)
    expected_units = {f"{arm}:{seed}" for arm in ARMS for seed in TRAINING_SEEDS}
    errors = validate_scoring_request(
        request,
        protocol_sha256=sha256_file(protocol_path),
        partition_hash=str(
            (protocol.get("protocol_manifest") or {}).get("partition_membership_sha256")
        ),
        trusted_label_hash=sha256_file(label_path),
        expected_units=expected_units,
    )
    if errors:
        print(f"[exp7385-trusted-score] phase=validate event=reject errors={errors}", flush=True)
        return 2
    print("[exp7385-trusted-score] phase=validate event=end passed=true", flush=True)
    labels_object = _load_object(label_path)
    labels = labels_object.get("rows") or []
    print("[exp7385-trusted-score] phase=evaluate event=before_final_scoring", flush=True)
    rows, metrics = score_final_rows(request["units"], protocol.get("feature_rows") or [], labels)
    result = {
        "schema": "carnot.exp7385.trusted_final_score.v1",
        "request_sha256": sha256_file(request_path),
        "protocol_sha256": sha256_file(protocol_path),
        "trusted_label_sha256": sha256_file(label_path),
        "rows": rows,
        "unit_metrics": metrics,
        "duration_s": time.monotonic() - started,
    }
    atomic_json(output_path, result)
    print(
        f"[exp7385-trusted-score] phase=evaluate event=after_final_scoring rows={len(rows)}",
        flush=True,
    )
    return 0


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:  # pragma: no cover
    names = {str(row.get("name")) for row in receipts if row.get("required") is True}
    expected = set(validation_scope.REQUIRED_CHECK_NAMES)
    return expected <= names and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in receipts
        if row.get("name") in expected
    )


def validate_candidate_artifact(value: object) -> list[str]:  # pragma: no cover
    """Cold-check scientific evidence before terminal readers have run."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_mismatch")
    budget = artifact.get("sample_size_budget") or {}
    if len(artifact.get("rows") or []) != int(budget.get("planned_final_rows") or -1):
        errors.append("row_completeness_mismatch")
    if len(artifact.get("checkpoint_manifest") or []) != 25:
        errors.append("checkpoint_completeness_mismatch")
    if not _required_validation_passed(artifact.get("validation_receipts") or []):
        errors.append("affected_validation_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = str(REPO_ROOT / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7385_v648_decision_training import validate_candidate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_candidate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "candidate"
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


def _training_run_record(state: Mapping[str, Any]) -> JsonDict:
    """Keep loss and calibration evidence while weights remain in checkpoints."""

    fields = (
        "arm",
        "seed",
        "training_partition",
        "training_prevalence",
        "objective",
        "update_count",
        "loss_curve",
        "pre_weight_sha256",
        "post_weight_sha256",
        "affine",
        "calibration_metrics_pre_affine",
        "calibration_metrics_post_affine",
        "selected_policy",
        "energy_recomputation_check",
        "fit_duration_s",
        "checkpoint_path",
        "checkpoint_sha256",
    )
    return {field: deepcopy(state[field]) for field in fields}


def _objective_reports(
    arm_metrics: Mapping[str, Mapping[str, Any]], training_runs: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Name the three learned objectives separately, including prior mismatch."""

    output: JsonDict = {}
    names = {
        "raw_nce_prior_mismatch": "raw_balanced_nce_gibbs",
        "prior_corrected_nce": "prior_corrected_nce_gibbs",
        "log_loss_trained_head": PRIMARY_VALUE_ARM,
    }
    for report, arm in names.items():
        runs = [row for row in training_runs if row.get("arm") == arm]
        output[report] = {
            "arm": arm,
            "final_test_metrics": deepcopy(dict(arm_metrics.get(arm) or {})),
            "probability_calibration_pre_affine": {
                metric: float(
                    np.mean([row["calibration_metrics_pre_affine"][metric] for row in runs])
                )
                for metric in ("brier", "log_loss", "auroc")
            }
            if runs
            else {},
            "probability_calibration_post_affine": {
                metric: float(
                    np.mean([row["calibration_metrics_post_affine"][metric] for row in runs])
                )
                for metric in ("brier", "log_loss", "auroc")
            }
            if runs
            else {},
        }
    return output


def _build_artifact(  # pragma: no cover
    *,
    started: float,
    started_at: str,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    protocol: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    scored: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    flagged_adversarial: bool,
) -> JsonDict:
    final_rows = [deepcopy(dict(row)) for row in scored.get("rows", [])]
    unit_metrics = {
        str(key): deepcopy(dict(value)) for key, value in (scored.get("unit_metrics") or {}).items()
    }
    arm_metrics = _aggregate_arm_metrics(unit_metrics)
    intervals = paired_group_intervals(final_rows, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    primary_states = [state for state in states if state.get("arm") == PRIMARY_VALUE_ARM]
    policy_certified = len(primary_states) == len(TRAINING_SEEDS) and all(
        state["selected_policy"]["accept_enabled"] or state["selected_policy"]["reject_enabled"]
        for state in primary_states
    )
    value_reduction = reduce_calibration_value(
        paired_intervals=intervals,
        metrics=arm_metrics,
        policy_certified=policy_certified,
    )
    training_runs = [_training_run_record(state) for state in states]
    planned_final_rows = (
        len(ARMS)
        * len(TRAINING_SEEDS)
        * int(
            (protocol.get("partition_summary") or {})
            .get("final_test", {})
            .get("effective_groups", 0)
        )
    )
    affected_passed = _required_validation_passed(receipts)
    scoring_passed = any(
        row.get("name") == "trusted_final_scoring"
        and row.get("passed") is True
        and row.get("exit_code") == 0
        for row in receipts
    )
    terminal_names = {
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    terminal_rows = [row for row in receipts if row.get("name") in terminal_names]
    terminal_passed = terminal_names == {str(row.get("name")) for row in terminal_rows} and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in terminal_rows
    )
    row_complete = len(final_rows) == planned_final_rows and planned_final_rows > 0
    checkpoint_complete = len(checkpoints) == len(ARMS) * len(TRAINING_SEEDS)
    safe = terminal_passed and not flagged_adversarial
    provisional_capture = int(
        affected_passed and scoring_passed and row_complete and checkpoint_complete and safe
    )
    provisional_value = int(provisional_capture and value_reduction["passed"])
    if provisional_capture == 0:
        verdict_class = "disqualified"
        honest = "complete_disqualified_decision_training_required_checks_failed"
        status = "complete_decision_training_disqualified"
    elif provisional_value:
        verdict_class = "circular_positive"
        honest = "complete_circular_positive_archive_calibration_value"
        status = "complete_decision_training_circular_positive"
    else:
        verdict_class = "null"
        honest = "complete_null_calibrated_decision_head_no_registered_value"
        status = "complete_decision_training_null"
    gates = [
        _gate(
            "authenticated_protocol",
            "completion",
            True,
            all(row.get("passed") is True for row in preconditions),
        ),
        _gate("all_frozen_fits", "completion", 25, len(states)),
        _gate("numeric_checkpoints", "completion", 25, len(checkpoints)),
        _gate("complete_final_rows", "completion", planned_final_rows, len(final_rows)),
        _gate("required_affected_validation", "required_validation", True, affected_passed),
        _gate("trusted_final_scoring", "safety", True, scoring_passed),
        _gate("independent_terminal_readers", "safety", True, safe),
        *[
            _gate(check, "scientific_efficacy", True, observed)
            for check, observed in value_reduction["checks"].items()
        ],
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": {
            "value": "host CPU JAX small energy-head training and numeric final scoring",
            "backend": f"JAX {jax.__version__} on {jax.default_backend()}",
            "devices": [str(device) for device in jax.devices()],
            "device_identity": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "jax_platform_request": os.environ.get("JAX_PLATFORMS"),
            "resource_lease": "host_process_only_no_gpu_or_llm_lease",
            "work": "small Gibbs and logistic fitting, calibration, numeric scoring, and bootstrap reduction",
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": time.monotonic() - started,
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "paired_group_bootstrap_seed": BOOTSTRAP_SEED,
            "online_reducer_seed": 7_386_307,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [
            {
                "path": PROTOCOL_PATH.as_posix(),
                "sha256": source_hashes.get(PROTOCOL_PATH.as_posix()),
                "label": "authenticated_upstream_protocol_not_current_model_work",
                "original_status": protocol.get("status"),
                "original_verdict_class": protocol.get("verdict_class"),
                "original_flagged_adversarial": protocol.get("flagged_adversarial"),
                "original_MODEL_SPECS": deepcopy(protocol.get("MODEL_SPECS") or []),
                "original_model_invoked": protocol.get("model_invoked"),
                "original_inference_substrate_class": protocol.get("inference_substrate_class"),
                "authorizes_current_llm_inference": False,
            }
        ],
        "protocol_identity": {
            "path": PROTOCOL_PATH.as_posix(),
            "artifact_sha256": source_hashes.get(PROTOCOL_PATH.as_posix()),
            "partition_membership_sha256": (protocol.get("protocol_manifest") or {}).get(
                "partition_membership_sha256"
            ),
            "trusted_label_sha256": source_hashes.get(TRUSTED_LABEL_PATH.as_posix()),
            "arms": list(ARMS),
            "seeds": list(TRAINING_SEEDS),
            "primary_value_arm": PRIMARY_VALUE_ARM,
            "simultaneous_test_count": SIMULTANEOUS_TEST_COUNT,
            "alpha_per_test": ALPHA_PER_TEST,
            "choices_sealed_before_final_test": True,
        },
        "rows": final_rows,
        "sample_size_budget": {
            "planned_fits": 25,
            "attempted_fits": len(states),
            "completed_fits": len(states),
            "censored_fits": 0,
            "unstarted_fits": max(0, 25 - len(states)),
            "planned_final_rows": planned_final_rows,
            "completed_final_rows": len(final_rows),
            "censored_final_rows": max(0, planned_final_rows - len(final_rows)),
            "maximum_optimizer_steps_per_fit": MAX_STEPS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "stopping_rule": "Run every frozen arm and seed once. Do not extend or select from final outcomes.",
            "remaining_work": "none"
            if provisional_capture
            else "required validation or evidence is incomplete",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": {
            "status": "not_assessed_beyond_required_affected_checks",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "field_principles": {},
        "promotion_score": 0,
        "decision_capture_complete_score": provisional_capture,
        "calibration_value_score": provisional_value,
        "checkpoint_manifest": [deepcopy(dict(row)) for row in checkpoints],
        "small_ebm_training": {
            "performed": True,
            "fits_attempted": len(states),
            "fits_completed": len(states),
            "training_optimizer_updates_completed": sum(
                int(state["update_count"]) for state in states
            ),
            "affine_optimizer_updates_completed": sum(
                int(state["affine"]["update_count"]) for state in states
            ),
            "gibbs_fits_completed": sum(state["arm"] in LEARNED_ARMS for state in states),
            "architecture": "2-4-1 Gibbs head; 17 parameters",
            "backend": f"JAX {jax.__version__} {jax.default_backend()}",
            "current_llm_calls": 0,
        },
        "calibration_metrics": {
            "by_arm_seed": unit_metrics,
            "by_arm": arm_metrics,
            "primary_value_arm": PRIMARY_VALUE_ARM,
        },
        "policy_certificate_rows": [deepcopy(dict(row)) for row in certificates],
        "training_runs": training_runs,
        "paired_group_intervals": intervals,
        "calibration_value_reduction": value_reduction,
        "objective_reports": _objective_reports(arm_metrics, training_runs),
        "online_replay_protocol": deepcopy(
            (protocol.get("protocol_manifest") or {}).get("online_replay") or {}
        ),
        "independent_reduction": {},
        "evidence_scope": {
            "archive": "reused_single_FoVer_archive",
            "final_test": "experiment_held_out_not_virgin_external",
            "general_verifier_moat_gap_closed": False,
            "claim_boundary": "bounded calibration evidence only",
        },
        "methodology_note": "Exact 0 or 1 row outcomes can occur for deterministic typed actions; aggregate claims use all groups and paired intervals.",
        "production_defaults_changed": False,
        "standing_conductor_benchmark_changed": False,
        "active_research_roadmap_changed": False,
    }
    artifact["independent_reduction"] = independent_reduce(artifact)
    artifact["decision_capture_complete_score"] = artifact["independent_reduction"][
        "decision_capture_complete_score"
    ]
    artifact["calibration_value_score"] = artifact["independent_reduction"][
        "calibration_value_score"
    ]
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Run sealed fitting, fresh scoring, affected checks, and terminal readers."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = _utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    _progress(started, "read", "start")
    preconditions, source_hashes, protocol = collect_preconditions(root)
    spans.append(_span("read", phase_started, started))
    prerequisites_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    _progress(started, "read", "end", passed=prerequisites_passed)
    if not prerequisites_passed:
        blocked = build_blocked_artifact(preconditions, source_hashes)
        blocked["started_at_utc"] = started_at
        blocked["completed_at_utc"] = _utc_now()
        blocked["duration_s"] = time.monotonic() - started
        blocked["phase_spans"] = spans
        blocked["field_principles"] = _field_principles(tuple(blocked))
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        _progress(started, "write", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        _progress(started, "write", "after_atomic_blocked", path=output_path)
        return blocked

    phase_started = time.monotonic()
    _progress(started, "build", "start", fits=25)
    states, checkpoints, certificates = _fit_all_units(protocol, root, started)
    spans.append(_span("build", phase_started, started))
    _progress(started, "build", "end", fits=len(states), checkpoints=len(checkpoints))

    phase_started = time.monotonic()
    _progress(started, "load", "start", model_load="not_attempted")
    spans.append(_span("load", phase_started, started, performed=False))
    _progress(started, "load", "end", model_invoked=False)
    phase_started = time.monotonic()
    _progress(started, "generate", "start", generation="not_attempted")
    spans.append(_span("generate", phase_started, started, performed=False))
    _progress(started, "generate", "end", calls=0)

    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    request = _scoring_request(states)
    request_path = raw_dir / "sealed_scoring_request.json"
    score_path = raw_dir / "trusted_final_score.json"
    atomic_json(request_path, request)
    source_hashes[request_path.relative_to(root).as_posix()] = sha256_file(request_path)
    phase_started = time.monotonic()
    _progress(started, "evaluate", "before_trusted_final_scoring_subprocess")
    scoring_command = PlannedCommand(
        validation_scope.CommandSpec(
            "trusted_final_scoring",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "-m",
                "carnot.experiment_7385_v648_decision_training",
                "--trusted-score",
                "--request",
                str(request_path),
                "--output",
                str(score_path),
            ),
            "sealed_plain_numeric_state",
            900.0,
        ),
        "scientific_completion",
        True,
    )
    scoring_receipts = run_categorized_commands(
        root,
        [scoring_command],
        log_dir=raw_dir / "validation/scoring",
    )
    scoring_passed = scoring_receipts[0].get("passed") is True and score_path.is_file()
    _progress(
        started,
        "evaluate",
        "after_trusted_final_scoring_subprocess",
        passed=scoring_passed,
    )
    scored = _load_object(score_path) if scoring_passed else {"rows": [], "unit_metrics": {}}
    if score_path.is_file():
        source_hashes[score_path.relative_to(root).as_posix()] = sha256_file(score_path)
    spans.append(_span("evaluate", phase_started, started))

    private = Path(tempfile.mkdtemp(prefix="exp7385-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    commands = build_command_plan(root, V648_MANIFEST, private)
    plan_errors = validate_command_plan(root, V648_MANIFEST, commands)
    _progress(started, "validate", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    spans.append(_span("validate", phase_started, started))
    _progress(
        started,
        "validate",
        "after_affected_subprocesses",
        passed=_required_validation_passed(affected),
    )

    phase_started = time.monotonic()
    candidate = _build_artifact(
        started=started,
        started_at=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        protocol=protocol,
        states=states,
        checkpoints=checkpoints,
        certificates=certificates,
        scored=scored,
        receipts=[*scoring_receipts, *affected],
        flagged_adversarial=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    _progress(started, "write", "before_atomic_candidate", path=candidate_path)
    atomic_json(candidate_path, candidate)
    spans.append(_span("write", phase_started, started))
    _progress(started, "write", "after_atomic_candidate", path=candidate_path)

    phase_started = time.monotonic()
    _progress(started, "validate", "before_terminal_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("validate_terminal", phase_started, started))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    _progress(
        started,
        "validate",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )
    final = _build_artifact(
        started=started,
        started_at=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        protocol=protocol,
        states=states,
        checkpoints=checkpoints,
        certificates=certificates,
        scored=scored,
        receipts=[*scoring_receipts, *affected, *terminal],
        flagged_adversarial=critical or not terminal_passed,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def resume_terminal_validation(  # pragma: no cover - failure recovery entrypoint.
    repo_root: Path,
    artifact_path: Path,
) -> JsonDict:
    """Rebuild terminal metadata from sealed rows without reading final labels again."""

    root = repo_root.resolve()
    started = time.monotonic()
    previous = _load_object(artifact_path)
    _progress(started, "read", "start_terminal_resume", path=artifact_path)
    preconditions, source_hashes, protocol = collect_preconditions(root)
    if not preconditions or any(row.get("passed") is not True for row in preconditions):
        raise RuntimeError("terminal_resume_preconditions_failed")
    source_hashes.update(
        {
            key: value
            for key, value in (previous.get("source_artifact_hashes") or {}).items()
            if key.startswith(RAW_DIR.as_posix())
        }
    )
    checkpoints = [dict(row) for row in previous.get("checkpoint_manifest") or []]
    by_unit = {(row["arm"], row["seed"]): row for row in checkpoints}
    states: list[JsonDict] = []
    for run in previous.get("training_runs") or []:
        checkpoint = by_unit[(run["arm"], run["seed"])]
        state = deepcopy(dict(run))
        state["weights"] = deepcopy(checkpoint["numeric_weights"])
        state["optimizer_state"] = deepcopy(checkpoint["optimizer_state"])
        states.append(state)
    if len(states) != 25 or len(previous.get("rows") or []) != 33_075:
        raise RuntimeError("terminal_resume_sealed_evidence_incomplete")
    base_receipts = [
        dict(row)
        for row in previous.get("validation_receipts") or []
        if row.get("name")
        not in {"independent_reducer", "adversarial_verify", "verdict_row_consistency_strict"}
    ]
    scored = {
        "rows": deepcopy(previous["rows"]),
        "unit_metrics": deepcopy(previous["calibration_metrics"]["by_arm_seed"]),
    }
    spans = [deepcopy(dict(row)) for row in previous.get("phase_spans") or []]
    recovery_span_start = time.monotonic()
    candidate = _build_artifact(
        started=started,
        started_at=str(previous["started_at_utc"]),
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        protocol=protocol,
        states=states,
        checkpoints=checkpoints,
        certificates=previous.get("policy_certificate_rows") or [],
        scored=scored,
        receipts=base_receipts,
        flagged_adversarial=False,
    )
    raw_dir = root / RAW_DIR
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    _progress(started, "validate", "before_resumed_terminal_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal_resume",
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    recovery_duration = time.monotonic() - recovery_span_start
    prior_duration = float(previous.get("duration_s") or 0.0)
    spans.append(
        {
            "phase": "validate_terminal_resume",
            "start_s": prior_duration,
            "end_s": prior_duration + recovery_duration,
            "duration_s": recovery_duration,
            "performed": True,
        }
    )
    _progress(
        started,
        "validate",
        "after_resumed_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )
    final = _build_artifact(
        started=started,
        started_at=str(previous["started_at_utc"]),
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        protocol=protocol,
        states=states,
        checkpoints=checkpoints,
        certificates=previous.get("policy_certificate_rows") or [],
        scored=scored,
        receipts=[*base_receipts, *terminal],
        flagged_adversarial=critical or not terminal_passed,
    )
    final["duration_s"] = float(previous.get("duration_s") or 0.0) + time.monotonic() - started
    final["completed_at_utc"] = _utc_now()
    final["field_principles"] = _field_principles(tuple(final))
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"resumed_terminal_artifact_invalid:{errors}")
    atomic_json(candidate_path, final)
    atomic_json(artifact_path, final)
    _progress(started, "write", "after_resumed_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the public run and private trusted-scoring modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--trusted-score", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--resume-terminal", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Dispatch the declared entrypoint or a strict fresh-process reader."""

    args = parse_args(argv)
    if args.trusted_score:
        if args.request is None:
            raise SystemExit("--trusted-score requires --request")
        return trusted_score_main(args.request, args.output)
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.resume_terminal is not None:
        resume_terminal_validation(REPO_ROOT, args.resume_terminal)
        return 0
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
