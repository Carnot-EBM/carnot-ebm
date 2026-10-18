"""Qualify a small guarded online-update lifecycle without loading a model.

The updater changes only a bounded residual head over cached evidence. It does
not change the generator. Held-out rows decide whether a proposal becomes the
next durable state, so a bad gradient step cannot silently enter later
predictions.

Spec refs: REQ-CL-7603 and SCENARIO-CL-7603-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.664"
EXPERIMENT_ID = "exp7603-v664-guarded-update-fixture"
SCHEMA = "carnot.exp7603.v664.guarded_update_fixture.v1"
RESULT_PATH = Path("results/experiment_7603_v664_guarded_update_fixture.json")
RAW_DIR = Path("results/raw/experiment_7603_v664_guarded_update_fixture")
MODULE_PATH = Path("python/carnot/experiment_7603_v664_guarded_update_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7603_v664_guarded_update_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7603_v664_guarded_update_fixture.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
UPSTREAM_PATH = Path("results/experiment_7575_v662_cached_learning_protocol.json")
CACHED_ROLES_PATH = Path(
    "results/raw/experiment_7575_v662_cached_learning_protocol/cached_roles.jsonl"
)
FROZEN_PROTOCOL_PATH = Path(
    "results/raw/experiment_7575_v662_cached_learning_protocol/frozen_protocol.json"
)
STREAM_SEED = 7_578_001
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "local_compact_energy_heads_on_exact_solver_features_no_llm"
ARMS = ("frozen", "unguarded", "guarded", "guarded_deranged")
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "guarded_update_ready_score",
    "continuous_self_learning_task",
    "update_rule_contract",
    "hardware_path",
)


@dataclass(frozen=True)
class UpdateConfig:
    """Bound the residual head so every accepted snapshot stays portable."""

    feature_count: int = 2
    hidden_units: int = 2
    step_size: float = 0.2
    parameter_bound: float = 0.5
    residual_bound: float = 1.0
    anchor_tolerance: float = 0.002
    lag: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.feature_count <= 8:
            raise ValueError("feature_count_must_be_1_to_8")
        if not 1 <= self.hidden_units <= 8:
            raise ValueError("hidden_units_must_be_1_to_8")
        finite = (
            math.isfinite(self.step_size)
            and math.isfinite(self.parameter_bound)
            and math.isfinite(self.residual_bound)
            and math.isfinite(self.anchor_tolerance)
        )
        if not finite or self.step_size < 0.0:
            raise ValueError("finite_nonnegative_step_size_required")
        if self.parameter_bound <= 0.0:
            raise ValueError("positive_parameter_bound_required")
        if self.residual_bound <= 0.0:
            raise ValueError("positive_residual_bound_required")
        if self.anchor_tolerance < 0.0:
            raise ValueError("nonnegative_anchor_tolerance_required")
        if self.lag != 8:
            raise ValueError("lag_must_equal_eight")


@dataclass(frozen=True)
class ResidualParameters:
    """Store only the small residual head; generator weights are not present."""

    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float

    def to_payload(self) -> JsonDict:
        return {
            "hidden_weights": [list(row) for row in self.hidden_weights],
            "hidden_bias": list(self.hidden_bias),
            "output_weights": list(self.output_weights),
            "output_bias": self.output_bias,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ResidualParameters:
        return cls(
            tuple(tuple(float(item) for item in row) for row in value["hidden_weights"]),
            tuple(float(item) for item in value["hidden_bias"]),
            tuple(float(item) for item in value["output_weights"]),
            float(value["output_bias"]),
        )


def initial_parameters(config: UpdateConfig) -> ResidualParameters:
    """Use a deterministic feature map so the first gradient can use evidence."""

    scale = min(0.05, config.parameter_bound * 0.5)
    weights = tuple(
        tuple(
            scale
            * (-1.0 if (hidden + feature) % 2 else 1.0)
            * (hidden + 1)
            * (feature + 1)
            / (config.hidden_units * config.feature_count)
            for feature in range(config.feature_count)
        )
        for hidden in range(config.hidden_units)
    )
    return ResidualParameters(
        weights,
        (0.0,) * config.hidden_units,
        (0.0,) * config.hidden_units,
        0.0,
    )


def _finite_features(config: UpdateConfig, features: Sequence[float]) -> tuple[float, ...]:
    if len(features) != config.feature_count:
        raise ValueError("feature_count_mismatch")
    values = tuple(float(item) for item in features)
    if not all(math.isfinite(item) for item in values):
        raise ValueError("finite_features_required")
    return values


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = math.exp(value)
    return direct / (1.0 + direct)


class ResidualEnergyHead:
    """Convert a cached base forecast and evidence into two normalized energies."""

    def __init__(self, config: UpdateConfig, parameters: ResidualParameters) -> None:
        self.config = config
        self.parameters = parameters

    def _forward(
        self, base_probability: float, features: Sequence[float]
    ) -> tuple[float, tuple[float, ...], float]:
        base = float(base_probability)
        if not math.isfinite(base) or not 0.0 < base < 1.0:
            raise ValueError("base_probability_must_be_finite_open_interval")
        values = _finite_features(self.config, features)
        hidden = tuple(
            math.tanh(sum(weight * feature for weight, feature in zip(row, values)) + bias)
            for row, bias in zip(self.parameters.hidden_weights, self.parameters.hidden_bias)
        )
        raw = (
            sum(
                weight * activation
                for weight, activation in zip(self.parameters.output_weights, hidden)
            )
            + self.parameters.output_bias
        )
        bound = self.config.residual_bound
        residual = bound * math.tanh(raw / bound)
        logit = math.log(base / (1.0 - base)) + residual
        return _sigmoid(logit), hidden, raw

    def predict(self, base_probability: float, features: Sequence[float]) -> float:
        return self._forward(base_probability, features)[0]

    def energies(self, base_probability: float, features: Sequence[float]) -> tuple[float, float]:
        probability = self.predict(base_probability, features)
        return -math.log1p(-probability), -math.log(probability)


def parameters_are_finite_and_bounded(config: UpdateConfig, parameters: ResidualParameters) -> bool:
    values = [
        *(item for row in parameters.hidden_weights for item in row),
        *parameters.hidden_bias,
        *parameters.output_weights,
        parameters.output_bias,
    ]
    shape_ok = (
        len(parameters.hidden_weights) == config.hidden_units
        and all(len(row) == config.feature_count for row in parameters.hidden_weights)
        and len(parameters.hidden_bias) == config.hidden_units
        and len(parameters.output_weights) == config.hidden_units
    )
    return shape_ok and all(
        math.isfinite(item) and abs(item) <= config.parameter_bound for item in values
    )


@dataclass(frozen=True)
class Proposal:
    """Carry both sides of one pure update so admission can be replayed."""

    prior: ResidualParameters
    candidate: ResidualParameters
    prior_hash: str
    candidate_hash: str
    fit_event_ids: tuple[str, ...]
    gradient_norm: float
    changed: bool
    clipped: bool


def _validated_example(config: UpdateConfig, row: Mapping[str, Any]) -> JsonDict:
    label = row.get("label")
    if label not in (0, 1):
        raise ValueError("binary_label_required")
    return {
        "event_id": str(row["event_id"]),
        "base_probability": float(row["base_probability"]),
        "features": _finite_features(config, row["features"]),
        "label": int(label),
    }


def propose_update(
    config: UpdateConfig,
    prior: ResidualParameters,
    fit_examples: Sequence[Mapping[str, Any]],
) -> Proposal:
    """Return one clipped Brier-gradient step without mutating the prior state."""

    if len(fit_examples) != 4:
        raise ValueError("exactly_four_fit_examples_required")
    examples = [_validated_example(config, row) for row in fit_examples]
    grad_w = [[0.0] * config.feature_count for _ in range(config.hidden_units)]
    grad_b = [0.0] * config.hidden_units
    grad_v = [0.0] * config.hidden_units
    grad_c = 0.0
    head = ResidualEnergyHead(config, prior)
    for row in examples:
        probability, hidden, raw = head._forward(row["base_probability"], row["features"])
        residual_derivative = 1.0 - math.tanh(raw / config.residual_bound) ** 2
        common = (
            2.0
            * (probability - row["label"])
            * probability
            * (1.0 - probability)
            * residual_derivative
            / len(examples)
        )
        grad_c += common
        for unit in range(config.hidden_units):
            grad_v[unit] += common * hidden[unit]
            hidden_common = common * prior.output_weights[unit] * (1.0 - hidden[unit] ** 2)
            grad_b[unit] += hidden_common
            for feature in range(config.feature_count):
                grad_w[unit][feature] += hidden_common * row["features"][feature]

    gradients = [*(item for row in grad_w for item in row), *grad_b, *grad_v, grad_c]
    clipped = False

    def step(value: float, gradient: float) -> float:
        nonlocal clipped
        proposed = value - config.step_size * gradient
        bounded = min(config.parameter_bound, max(-config.parameter_bound, proposed))
        clipped = clipped or bounded != proposed
        return bounded

    candidate = ResidualParameters(
        tuple(
            tuple(step(value, grad_w[unit][feature]) for feature, value in enumerate(row))
            for unit, row in enumerate(prior.hidden_weights)
        ),
        tuple(step(value, grad_b[index]) for index, value in enumerate(prior.hidden_bias)),
        tuple(step(value, grad_v[index]) for index, value in enumerate(prior.output_weights)),
        step(prior.output_bias, grad_c),
    )
    prior_hash = canonical_hash(prior.to_payload())
    candidate_hash = canonical_hash(candidate.to_payload())
    return Proposal(
        prior,
        candidate,
        prior_hash,
        candidate_hash,
        tuple(row["event_id"] for row in examples),
        math.sqrt(sum(item * item for item in gradients)),
        candidate_hash != prior_hash,
        clipped,
    )


def _brier(probability: float, label: int) -> float:
    return (float(probability) - int(label)) ** 2


def _realized_cost(probability: float, label: int) -> float:
    expected = {"accept": 5.0 * probability, "reject": 1.0 - probability, "escalate": 0.2}
    minimum = min(expected.values())
    action = next(
        name
        for name in ("escalate", "accept", "reject")
        if math.isclose(expected[name], minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    return {"accept": 5.0 if label else 0.0, "reject": 0.0 if label else 1.0, "escalate": 0.2}[
        action
    ]


def _mean_metrics(
    config: UpdateConfig,
    parameters: ResidualParameters,
    examples: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    checked = [_validated_example(config, row) for row in examples]
    if not checked:
        raise ValueError("nonempty_examples_required")
    head = ResidualEnergyHead(config, parameters)
    probabilities = [head.predict(row["base_probability"], row["features"]) for row in checked]
    return (
        sum(_brier(probability, row["label"]) for probability, row in zip(probabilities, checked))
        / len(checked),
        sum(
            _realized_cost(probability, row["label"])
            for probability, row in zip(probabilities, checked)
        )
        / len(checked),
    )


@dataclass(frozen=True)
class Admission:
    """Record every guard operand so rejection is independently reproducible."""

    accepted: bool
    checks: dict[str, bool]
    prior_hash: str
    candidate_hash: str
    prior_admission_brier: float
    candidate_admission_brier: float
    prior_admission_cost: float
    candidate_admission_cost: float
    anchor_brier_increase: float


def evaluate_proposal(
    config: UpdateConfig,
    proposal: Proposal,
    admission_examples: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
) -> Admission:
    """Apply the held-out guard without using admission labels for a gradient."""

    if len(admission_examples) != 4:
        raise ValueError("exactly_four_admission_examples_required")
    if len(anchors) != 16:
        raise ValueError("exactly_sixteen_anchor_examples_required")
    prior_brier, prior_cost = _mean_metrics(config, proposal.prior, admission_examples)
    candidate_brier, candidate_cost = _mean_metrics(config, proposal.candidate, admission_examples)
    prior_anchor, _prior_anchor_cost = _mean_metrics(config, proposal.prior, anchors)
    candidate_anchor, _candidate_anchor_cost = _mean_metrics(config, proposal.candidate, anchors)
    increase = candidate_anchor - prior_anchor
    checks = {
        "finite_bounded_parameters": parameters_are_finite_and_bounded(config, proposal.candidate),
        "parameter_change": proposal.changed,
        "lower_admission_brier": candidate_brier < prior_brier,
        "no_admission_cost_increase": candidate_cost <= prior_cost,
        "anchor_brier_increase_at_most_0_002": increase <= config.anchor_tolerance,
    }
    return Admission(
        all(checks.values()),
        checks,
        proposal.prior_hash,
        proposal.candidate_hash,
        prior_brier,
        candidate_brier,
        prior_cost,
        candidate_cost,
        increase,
    )


class UpdateLifecycle:
    """Own predictions, legal releases, proposals, admission, and snapshots."""

    def __init__(
        self,
        *,
        config: UpdateConfig,
        arm: str,
        anchors: Sequence[Mapping[str, Any]],
        state_path: Path,
        parameters: ResidualParameters | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError("arm_invalid")
        if len(anchors) != 16:
            raise ValueError("exactly_sixteen_anchor_examples_required")
        self.config = config
        self.arm = arm
        self.anchors = [deepcopy(dict(row)) for row in anchors]
        self.state_path = Path(state_path)
        self.parameters = parameters or initial_parameters(config)
        self._predictions: dict[str, JsonDict] = {}
        self._prediction_order: list[str] = []
        self._released: dict[str, JsonDict] = {}
        self._release_order: list[str] = []
        self.accepted_update_count = 0
        self.update_history: list[JsonDict] = []

    @property
    def parameter_hash(self) -> str:
        return canonical_hash(self.parameters.to_payload())

    @property
    def prediction_receipts(self) -> dict[str, JsonDict]:
        return deepcopy(self._predictions)

    @property
    def released_event_ids(self) -> list[str]:
        return list(self._release_order)

    def predict(
        self,
        event_id: str,
        base_probability: float,
        features: Sequence[float],
        prediction_index: int,
    ) -> JsonDict:
        identifier = str(event_id)
        if identifier in self._predictions or identifier in self._released:
            raise ValueError(f"duplicate_prediction:{identifier}")
        values = _finite_features(self.config, features)
        started = time.perf_counter_ns()
        probability = ResidualEnergyHead(self.config, self.parameters).predict(
            base_probability, values
        )
        elapsed = max(1, time.perf_counter_ns() - started)
        receipt = {
            "event_id": identifier,
            "base_probability": float(base_probability),
            "features": list(values),
            "prediction_index": int(prediction_index),
            "probability": probability,
            "parameter_hash": self.parameter_hash,
            "inference_ns": elapsed,
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        self._predictions[identifier] = deepcopy(receipt)
        self._prediction_order.append(identifier)
        return deepcopy(receipt)

    def release(
        self, event_id: str, label: int, release_index: int, *, role: str = "stream"
    ) -> JsonDict:
        identifier = str(event_id)
        if role == "evaluation":
            raise PermissionError("evaluator_read_denied")
        if identifier in self._released:
            raise ValueError(f"duplicate_release:{identifier}")
        if identifier not in self._predictions:
            raise ValueError(f"unknown_event:{identifier}")
        if label not in (0, 1):
            raise ValueError("binary_label_required")
        prediction_index = self._predictions[identifier]["prediction_index"]
        if int(release_index) < prediction_index + self.config.lag:
            raise ValueError(f"feedback_too_early:{identifier}")
        release = {
            "event_id": identifier,
            "label": int(label),
            "release_index": int(release_index),
        }
        self._released[identifier] = release
        self._release_order.append(identifier)
        return deepcopy(release)

    def _examples(self, event_ids: Sequence[str]) -> list[JsonDict]:
        examples = []
        for event_id in event_ids:
            prediction = self._predictions[event_id]
            release = self._released[event_id]
            examples.append(
                {
                    "event_id": event_id,
                    "base_probability": prediction["base_probability"],
                    "features": prediction["features"],
                    "label": release["label"],
                }
            )
        return examples

    def propose(self, block_id: str, event_ids: Sequence[str]) -> Proposal:
        if len(event_ids) != 8 or len(set(event_ids)) != 8:
            raise ValueError("exactly_one_eight_event_block_required")
        if any(event_id not in self._released for event_id in event_ids):
            raise ValueError("block_not_fully_released")
        indexes = [self._prediction_order.index(event_id) for event_id in event_ids]
        if indexes != sorted(indexes) or any(
            right != left + 1 for left, right in zip(indexes, indexes[1:])
        ):
            raise ValueError("event_order_mismatch")
        proposal = propose_update(self.config, self.parameters, self._examples(event_ids[:4]))
        self.update_history.append(
            {
                "block_id": str(block_id),
                "stage": "proposed",
                "prior_hash": proposal.prior_hash,
                "candidate_hash": proposal.candidate_hash,
                "fit_event_ids": list(proposal.fit_event_ids),
            }
        )
        return proposal

    def admit(self, proposal: Proposal, event_ids: Sequence[str]) -> JsonDict:
        if proposal.prior_hash != self.parameter_hash:
            raise ValueError("stale_proposal")
        if len(event_ids) != 8:
            raise ValueError("exactly_one_eight_event_block_required")
        started = time.perf_counter_ns()
        assessment = evaluate_proposal(
            self.config, proposal, self._examples(event_ids[4:]), self.anchors
        )
        if self.arm == "frozen":
            accepted, reason = False, "frozen_predictor"
        elif self.arm == "unguarded":
            accepted = bool(assessment.checks["finite_bounded_parameters"] and proposal.changed)
            reason = "unguarded_update" if accepted else "invalid_or_unchanged_proposal"
        else:
            accepted = assessment.accepted
            reason = "guard_accepted" if accepted else "guard_rejected"
        if accepted:
            self.parameters = proposal.candidate
            self.accepted_update_count += 1
        update_ns = max(1, time.perf_counter_ns() - started)
        result = {
            "accepted": accepted,
            "reason": reason,
            "guard_passed": assessment.accepted,
            "prior_hash": proposal.prior_hash,
            "candidate_hash": proposal.candidate_hash,
            "state_hash_after": self.parameter_hash,
            "checks": dict(assessment.checks),
            "prior_admission_brier": assessment.prior_admission_brier,
            "candidate_admission_brier": assessment.candidate_admission_brier,
            "prior_admission_cost": assessment.prior_admission_cost,
            "candidate_admission_cost": assessment.candidate_admission_cost,
            "anchor_brier_increase": assessment.anchor_brier_increase,
            "update_ns": update_ns,
        }
        self.update_history.append({"stage": "admitted", **deepcopy(result)})
        return result

    def _payload(self) -> JsonDict:
        return {
            "schema": "carnot.guarded_update_state.v1",
            "arm": self.arm,
            "config": asdict(self.config),
            "parameters": self.parameters.to_payload(),
            "parameter_hash": self.parameter_hash,
            "predictions": deepcopy(self._predictions),
            "prediction_order": list(self._prediction_order),
            "released": deepcopy(self._released),
            "release_order": list(self._release_order),
            "accepted_update_count": self.accepted_update_count,
            "update_history": deepcopy(self.update_history),
            "generator_weights": "absent_by_schema",
        }

    def persist(self) -> JsonDict:
        """Atomically replace one complete snapshot, then verify its exact bytes."""

        payload = self._payload()
        atomic_json(self.state_path, payload)
        loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        if loaded != payload:
            raise OSError("atomic_snapshot_reload_mismatch")
        return {"path": str(self.state_path), "sha256": sha256_file(self.state_path)}

    @classmethod
    def reload(
        cls,
        *,
        state_path: Path,
        config: UpdateConfig,
        arm: str,
        anchors: Sequence[Mapping[str, Any]],
    ) -> UpdateLifecycle:
        payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
        if payload.get("schema") != "carnot.guarded_update_state.v1":
            raise ValueError("state_schema_invalid")
        if payload.get("arm") != arm or payload.get("config") != asdict(config):
            raise ValueError("state_contract_mismatch")
        machine = cls(
            config=config,
            arm=arm,
            anchors=anchors,
            state_path=state_path,
            parameters=ResidualParameters.from_payload(payload["parameters"]),
        )
        if machine.parameter_hash != payload.get("parameter_hash"):
            raise ValueError("state_parameter_hash_mismatch")
        machine._predictions = deepcopy(payload["predictions"])
        machine._prediction_order = list(payload["prediction_order"])
        machine._released = deepcopy(payload["released"])
        machine._release_order = list(payload["release_order"])
        machine.accepted_update_count = int(payload["accepted_update_count"])
        machine.update_history = deepcopy(payload["update_history"])
        for event_id, receipt in machine._predictions.items():
            unhashed = {key: value for key, value in receipt.items() if key != "receipt_hash"}
            if receipt.get("receipt_hash") != canonical_hash(unhashed):
                raise ValueError(f"prediction_receipt_hash_mismatch:{event_id}")
        return machine


def _load_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    passed = {
        "eq": observed == expected,
        "starts_with": isinstance(observed, str) and str(observed).startswith(str(expected)),
        "gte": isinstance(observed, (int, float)) and observed >= expected,
    }[operator]
    return {
        "check": check,
        "upstream": upstream,
        "path": path.as_posix(),
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "required": True,
        "passed": passed,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate every historical byte before opening labels."""

    root = root.resolve()
    artifact_path = root / UPSTREAM_PATH
    artifact = _load_json(artifact_path)
    role_receipt = (artifact.get("raw_sidecars") or {}).get("cached_roles") or {}
    protocol_receipt = (artifact.get("raw_sidecars") or {}).get("frozen_protocol") or {}
    role_path = root / Path(role_receipt.get("path") or CACHED_ROLES_PATH)
    protocol_path = root / Path(protocol_receipt.get("path") or FROZEN_PROTOCOL_PATH)
    protocol = _load_json(protocol_path)
    frozen = (
        artifact.get("frozen_protocol")
        if isinstance(artifact.get("frozen_protocol"), Mapping)
        else {}
    )
    order = (frozen or protocol).get("orders", {}).get(str(STREAM_SEED), [])
    observed_role_hash = sha256_file(role_path) if role_path.is_file() else None
    observed_protocol_hash = sha256_file(protocol_path) if protocol_path.is_file() else None
    checks = [
        _check(
            "upstream_artifact_exists",
            EXPERIMENT_ID,
            UPSTREAM_PATH,
            "filesystem.is_file",
            "eq",
            True,
            artifact_path.is_file(),
        ),
        _check(
            "upstream_terminal_verdict",
            "exp7575-v662-cached-learning-protocol",
            UPSTREAM_PATH,
            "honest_verdict",
            "starts_with",
            "complete_",
            artifact.get("honest_verdict"),
        ),
        _check(
            "cached_roles_sha256",
            "exp7575-v662-cached-learning-protocol",
            Path(role_receipt.get("path") or CACHED_ROLES_PATH),
            "sha256",
            "eq",
            role_receipt.get("sha256"),
            observed_role_hash,
        ),
        _check(
            "frozen_protocol_sha256",
            "exp7575-v662-cached-learning-protocol",
            Path(protocol_receipt.get("path") or FROZEN_PROTOCOL_PATH),
            "sha256",
            "eq",
            protocol_receipt.get("sha256"),
            observed_protocol_hash,
        ),
        _check(
            "feedback_delay",
            "exp7575-v662-cached-learning-protocol",
            UPSTREAM_PATH,
            "frozen_protocol.feedback_delay",
            "eq",
            8,
            (frozen or protocol).get("feedback_delay"),
        ),
        _check(
            "stream_order_size",
            "exp7575-v662-cached-learning-protocol",
            UPSTREAM_PATH,
            f"frozen_protocol.orders.{STREAM_SEED}.length",
            "gte",
            80,
            len(order) if isinstance(order, list) else 0,
        ),
    ]
    sources = []
    for kind, relative, path in (
        ("producer_artifact", UPSTREAM_PATH, artifact_path),
        ("producer_sidecar", Path(role_receipt.get("path") or CACHED_ROLES_PATH), role_path),
        (
            "producer_protocol",
            Path(protocol_receipt.get("path") or FROZEN_PROTOCOL_PATH),
            protocol_path,
        ),
    ):
        sources.append(
            {
                "kind": kind if path.is_file() else "missing_producer",
                "path": relative.as_posix(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    return checks, sources


def _evidence_features(probability: float) -> list[float]:
    centered = 2.0 * float(probability) - 1.0
    return [centered, centered * centered]


def load_authenticated_inputs(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    checks, _sources = collect_preconditions(root)
    failed = [row for row in checks if not row["passed"]]
    if failed:
        raise ValueError(f"external_precondition_failed:{failed[0]['check']}")
    artifact = _load_json(root / UPSTREAM_PATH)
    role_receipt = artifact["raw_sidecars"]["cached_roles"]
    roles: list[JsonDict] = []
    with (root / role_receipt["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("role") in {"fit", "online"}:
                roles.append(value)
    fit = [row for row in roles if row["role"] == "fit"][:16]
    online = {str(row["source_id"]): row for row in roles if row["role"] == "online"}
    order = artifact["frozen_protocol"]["orders"][str(STREAM_SEED)][:80]
    if len(fit) != 16 or len(order) != 80 or any(str(item) not in online for item in order):
        raise ValueError("authenticated_role_membership_invalid")

    def convert(row: Mapping[str, Any], event_id: str) -> JsonDict:
        probability = float(row["probability"])
        label = row["label"]
        if label not in (0, 1) or not 0.0 < probability < 1.0:
            raise ValueError("authenticated_probability_or_label_invalid")
        return {
            "event_id": event_id,
            "source_id": str(row["source_id"]),
            "base_probability": probability,
            "features": _evidence_features(probability),
            "label": int(label),
        }

    stream_rows = [convert(online[str(source_id)], str(source_id)) for source_id in order]
    anchors = [convert(row, f"anchor:{row['source_id']}") for row in fit]
    return stream_rows, anchors


def metric_row(unit_id: str, arm: str, probability: float, label: int, seed: int) -> JsonDict:
    loss = _brier(probability, label)
    return {
        "unit_id": str(unit_id),
        "arm": str(arm),
        "probability": float(probability),
        "label": int(label),
        "brier": loss,
        "raw_squared_error_numerator": loss,
        "raw_squared_error_denominator": 1,
        "metric_direction": "lower_is_better",
        "seed": int(seed),
        "censored": False,
        "provenance": "exp7575_hash_bound_online_row",
    }


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    required = {
        "unit_id",
        "arm",
        "probability",
        "label",
        "brier",
        "raw_squared_error_numerator",
        "raw_squared_error_denominator",
        "metric_direction",
        "seed",
        "censored",
        "provenance",
    }
    seen: set[tuple[str, str]] = set()
    totals: dict[str, list[float]] = {}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"row_required_field_missing:{sorted(missing)[0]}")
        key = (str(row["unit_id"]), str(row["arm"]))
        if key in seen:
            raise ValueError(f"duplicate_unit_arm:{key}")
        seen.add(key)
        expected = _brier(float(row["probability"]), int(row["label"]))
        if (
            row["raw_squared_error_denominator"] != 1
            or not math.isclose(float(row["brier"]), expected, abs_tol=1e-12)
            or not math.isclose(float(row["raw_squared_error_numerator"]), expected, abs_tol=1e-12)
        ):
            raise ValueError(f"row_arithmetic_mismatch:{key}")
        totals.setdefault(str(row["arm"]), []).append(expected)
    return {
        "row_count": len(rows),
        "arm_summaries": {
            arm: {
                "numerator": sum(values),
                "denominator": len(values),
                "mean_brier": sum(values) / len(values),
            }
            for arm, values in sorted(totals.items())
        },
    }


def _forced_controls(anchors: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Use transparent labels to prove both guard directions, not benefit."""

    config = UpdateConfig(step_size=0.05)
    prior = initial_parameters(config)

    def examples(label: int) -> list[JsonDict]:
        return [
            {
                "event_id": f"forced-{label}-{index}",
                "base_probability": 0.5,
                "features": [1.0, -1.0],
                "label": label,
            }
            for index in range(4)
        ]

    fit = examples(1)
    proposal = propose_update(config, prior, fit)
    accepted = evaluate_proposal(config, proposal, fit, anchors)
    harmful = evaluate_proposal(config, proposal, examples(0), anchors)
    clipped_config = UpdateConfig(step_size=100.0, parameter_bound=0.01)
    clipped = propose_update(clipped_config, initial_parameters(clipped_config), fit)
    zero_config = UpdateConfig(step_size=0.0)
    zero = propose_update(zero_config, initial_parameters(zero_config), fit)
    return {
        "accepted": {**asdict(accepted), "fixture_class": "circular_positive"},
        "harmful": {**asdict(harmful), "fixture_class": "circular_positive"},
        "clipped": {
            "changed": clipped.changed,
            "clipped": clipped.clipped,
            "candidate_hash": clipped.candidate_hash,
            "treatment_candidate_hash": proposal.candidate_hash,
        },
        "no_update": {
            "changed": zero.changed,
            "clipped": zero.clipped,
            "candidate_hash": zero.candidate_hash,
            "treatment_candidate_hash": proposal.candidate_hash,
        },
    }


def run_fixture(
    stream: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    state_dir: Path,
) -> JsonDict:
    """Replay ten blocks with prediction before every delayed release."""

    if len(stream) != 80 or len(anchors) != 16:
        raise ValueError("fixture_requires_80_stream_and_16_anchor_rows")
    state_dir.mkdir(parents=True, exist_ok=True)
    config = UpdateConfig()
    machines = {
        arm: UpdateLifecycle(
            config=config,
            arm=arm,
            anchors=anchors,
            state_path=state_dir / f"{arm}.json",
        )
        for arm in ARMS
    }
    for machine in machines.values():
        machine.persist()
    rows: list[JsonDict] = []
    block_results: list[JsonDict] = []
    releases: list[JsonDict] = []
    inference_ns = 0
    update_ns = 0
    restart_parity = True
    label_counts_preserved = True
    release_times_preserved = True
    original_by_block: dict[int, list[tuple[str, int, int]]] = {}

    for clock in range(88):
        if clock < len(stream):
            source = stream[clock]
            for arm, machine in machines.items():
                receipt = machine.predict(
                    str(source["event_id"]),
                    float(source["base_probability"]),
                    source["features"],
                    clock,
                )
                inference_ns += int(receipt["inference_ns"])
                rows.append(
                    metric_row(
                        str(source["event_id"]),
                        arm,
                        float(receipt["probability"]),
                        int(source["label"]),
                        STREAM_SEED,
                    )
                )
        release_index = clock - config.lag
        if not 0 <= release_index < len(stream):
            continue
        source = stream[release_index]
        event_id = str(source["event_id"])
        label = int(source["label"])
        block = release_index // 8
        original_by_block.setdefault(block, []).append((event_id, label, clock))
        for arm in ("frozen", "unguarded", "guarded"):
            release = machines[arm].release(event_id, label, clock)
            releases.append({"arm": arm, **release, "prediction_index": release_index})
        if release_index % 8 != 7:
            continue

        original = original_by_block[block]
        deranged = original[1:] + original[:1]
        deranged_labels = [item[1] for item in deranged]
        label_counts_preserved &= sorted(deranged_labels) == sorted(item[1] for item in original)
        for (target_id, _target_label, target_time), replacement_label in zip(
            original, deranged_labels
        ):
            release = machines["guarded_deranged"].release(
                target_id, replacement_label, target_time
            )
            releases.append(
                {
                    "arm": "guarded_deranged",
                    **release,
                    "prediction_index": machines["guarded_deranged"].prediction_receipts[target_id][
                        "prediction_index"
                    ],
                }
            )
            release_times_preserved &= release["release_index"] == target_time

        event_ids = [item[0] for item in original]
        for arm in ARMS:
            machine = machines[arm]
            proposal = machine.propose(f"block-{block}", event_ids)
            outcome = machine.admit(proposal, event_ids)
            update_ns += int(outcome["update_ns"])
            snapshot = machine.persist()
            reloaded = UpdateLifecycle.reload(
                state_path=machine.state_path,
                config=config,
                arm=arm,
                anchors=anchors,
            )
            restart_parity &= (
                reloaded.parameter_hash == machine.parameter_hash
                and reloaded.prediction_receipts == machine.prediction_receipts
                and reloaded.released_event_ids == machine.released_event_ids
            )
            machines[arm] = reloaded
            block_results.append(
                {
                    "block_id": block,
                    "arm": arm,
                    "fit_event_ids": event_ids[:4],
                    "admission_event_ids": event_ids[4:],
                    "snapshot": snapshot,
                    **outcome,
                }
            )

    duplicate_rejected = False
    evaluator_denied = False
    try:
        machines["guarded"].release(str(stream[0]["event_id"]), int(stream[0]["label"]), 99)
    except ValueError as error:
        duplicate_rejected = "duplicate_release" in str(error)
    try:
        machines["guarded"].release(
            str(stream[0]["event_id"]), int(stream[0]["label"]), 99, role="evaluation"
        )
    except PermissionError as error:
        evaluator_denied = "evaluator_read_denied" in str(error)
    controls = _forced_controls(anchors)
    causal = all(row["release_index"] - row["prediction_index"] >= config.lag for row in releases)
    peak_snapshot = max(path.stat().st_size for path in state_dir.glob("*.json"))
    return {
        "rows": rows,
        "block_results": block_results,
        "release_rows": releases,
        "causal_timing_passed": causal,
        "restart_parity": restart_parity,
        "accepted_control_passed": bool(controls["accepted"]["accepted"]),
        "rejected_control_passed": not bool(controls["harmful"]["accepted"]),
        "harmful_update_rejected": not bool(controls["harmful"]["accepted"]),
        "duplicate_rejected": duplicate_rejected,
        "evaluator_read_denied": evaluator_denied,
        "deranged_control": {
            "within_block_only": True,
            "release_times_preserved": release_times_preserved,
            "label_counts_preserved": label_counts_preserved,
            "future_origin_shuffle": False,
        },
        "control_results": controls,
        "timing_ns": {"inference": inference_ns, "update": update_ns},
        "peak_snapshot_bytes": peak_snapshot,
        "final_state_hashes": {arm: machine.parameter_hash for arm, machine in machines.items()},
    }


ZERO_INVOCATION_COUNTS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}


def field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "A complete prefix marks terminal work; it does not imply scientific benefit.",
        "verdict_class": "The closed class keeps a fixture-positive result out of empirical headline claims.",
        "flagged_adversarial": "A flagged reader result can never open readiness.",
        "gate_check_summary": "A blocked result names the exact failed operand instead of inventing dependent evidence.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness answer different questions.",
        "rows": "Per-unit operands make every aggregate independently reducible.",
        "sample_size_budget": "Repeated arms do not multiply the 80 independent stream units.",
        "inference_substrate": "The substrate describes the CPU work performed now, not historical model calls.",
        "inference_substrate_class": "The actual closed class selects the correct no-model duration rule.",
        "MODEL_SPECS": "An empty list states that this run loaded no language model.",
        "invocation_counts": "Zero typed counters prevent historical calls from becoming current calls.",
        "duration_s": "Monotonic duration covers only this run and is never padded.",
        "random_seed": "The frozen stream seed fixes order without multiplying source units.",
        "reproducibility_checksum": "The checksum binds configuration, source bytes, rows, and reduction.",
        "source_artifact_hashes": "Producer, sidecar, and conductor records stay distinguishable.",
        "validation_receipts": "Commands, exits, logs, and independent readers remain auditable.",
        "verifier_is_oracle": "Exact fixtures are circular and cannot establish learned semantic correctness.",
        "guarded_update_ready_score": "Readiness requires both guard directions, causal timing, and restart parity.",
        "continuous_self_learning_task": "The task qualifies update mechanics, not empirical value.",
        "update_rule_contract": "The contract freezes delay, block roles, thresholds, and state schema.",
        "hardware_path": "CPU counts and a fixed-point mapping do not imply an unmeasured speedup.",
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def _acceptance_gates(valid: bool, ready: bool) -> dict[str, JsonDict]:
    return {
        "validity": {
            "principle": "Authenticated inputs and passing scoped checks are required before interpretation.",
            "result": valid,
        },
        "readiness": {
            "principle": "Both guard directions, causal timing, and restart parity qualify only the lifecycle.",
            "result": ready,
        },
        "benefit": {
            "principle": "A separate held-out empirical gate is required for a learning-benefit claim.",
            "result": False,
        },
        "retention": {
            "principle": "Final evaluation labels were denied, so this fixture makes no retention claim.",
            "result": False,
        },
        "freshness": {
            "principle": "Cached historical rows cannot establish fresh-population evidence.",
            "result": False,
        },
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _base_fields(duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "random_seed": STREAM_SEED,
        "random_seeds": {"stream_order": STREAM_SEED, "derangement": "cyclic_shift_one"},
        "continuous_self_learning_task": True,
        "generator_weights_immutable": True,
        "verifier_is_oracle": True,
        "field_principles": field_principles(),
        "external_publication_authorized": False,
        "deployment_promotion_available": False,
    }


def build_artifact(
    evidence: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_work_receipt: Mapping[str, Any],
    *,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    required_checks = _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
    lifecycle = all(
        bool(evidence[name])
        for name in (
            "accepted_control_passed",
            "rejected_control_passed",
            "causal_timing_passed",
            "restart_parity",
            "harmful_update_rejected",
            "duplicate_rejected",
            "evaluator_read_denied",
        )
    )
    valid = all(bool(row.get("passed")) for row in checks) and required_checks
    ready = valid and lifecycle
    rows = [deepcopy(dict(row)) for row in evidence["rows"]]
    reduction = reduce_rows(rows)
    gates = _acceptance_gates(valid, ready)
    value = {
        **_base_fields(duration_s),
        "honest_verdict": "complete_circular_positive_guarded_update_lifecycle_ready",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "passed": valid and ready,
            "failed_count": sum(not item["result"] for item in gates.values()),
            "failed_checks": [name for name, item in gates.items() if not item["result"]],
            "scientific_benefit_passed": False,
        },
        "acceptance_gate_results": gates,
        "rows": rows,
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended_independent_units": 80,
            "observed_independent_units": len({row["unit_id"] for row in rows}),
            "excluded_independent_units": 0,
            "censored_independent_units": 0,
            "arms_do_not_multiply_units": True,
        },
        "guarded_update_ready_score": int(ready),
        "fixture_positive_class": "circular_positive",
        "empirical_benefit_claim": None,
        "retention_claim": None,
        "update_rule_contract": {
            "feedback_lag": 8,
            "block_count": 10,
            "block_size": 8,
            "fit_positions": [0, 1, 2, 3],
            "admission_positions": [4, 5, 6, 7],
            "proposal": "one_full_batch_brier_gradient_step",
            "parameter_bound": UpdateConfig().parameter_bound,
            "admission_brier": "candidate_strictly_lower_than_prior",
            "admission_cost": "candidate_less_than_or_equal_to_prior",
            "anchor_brier_increase_max": 0.002,
            "state_schema": "carnot.guarded_update_state.v1",
            "admission_labels_reused": False,
            "final_evaluation_labels_visible": False,
        },
        "hardware_path": {
            "actual": "bounded_cpu_scalar_operations_and_atomic_json_snapshot",
            "estimated_multiply_adds": 80 * len(ARMS) * 2 * 2,
            "peak_snapshot_bytes": int(evidence["peak_snapshot_bytes"]),
            "fixed_point_mapping": "Rust arrays with signed Q2.14 parameters and Q1.15 features",
            "rust_path": "portable fixed-size loops; not implemented or benchmarked here",
            "measured_speedup": None,
            "speedup_claim": False,
        },
        "timing_ns": deepcopy(evidence["timing_ns"]),
        "block_results": deepcopy(evidence["block_results"]),
        "release_rows": deepcopy(evidence["release_rows"]),
        "deranged_control": deepcopy(evidence["deranged_control"]),
        "control_results": deepcopy(evidence["control_results"]),
        "lifecycle_controls": {
            key: bool(evidence[key])
            for key in (
                "accepted_control_passed",
                "rejected_control_passed",
                "causal_timing_passed",
                "restart_parity",
                "harmful_update_rejected",
                "duplicate_rejected",
                "evaluator_read_denied",
            )
        },
        "final_state_hashes": deepcopy(evidence["final_state_hashes"]),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "current_work_receipt": deepcopy(dict(current_work_receipt)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "applicable_e2e": {
            "id": "E2E-007-lifecycle-subset",
            "durable_accepted_state": bool(evidence["restart_parity"]),
            "unsafe_candidates_rejected": bool(evidence["harmful_update_rejected"]),
            "no_model_weight_mutation": True,
            "cerce_certificate_claimed": False,
            "full_e2e_007_applicable": False,
        },
    }
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_blocked_artifact(
    failed: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    first_failure = {
        key: failed.get(key)
        for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
    }
    value = {
        **_base_fields(duration_s),
        "honest_verdict": f"complete_blocked_{failed.get('check', 'external_input')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "passed": False,
            "failed_count": sum(not bool(row.get("passed")) for row in checks),
            "failed_checks": [row.get("check") for row in checks if not row.get("passed")],
            "first_failure": first_failure,
        },
        "acceptance_gate_results": {
            name: {
                "principle": principle,
                "result": False if name in {"validity", "readiness"} else None,
            }
            for name, principle in {
                "validity": "Missing authenticated input prevents measurement.",
                "readiness": "No lifecycle ran, so readiness stays closed.",
                "benefit": "A resource block does not retire the benefit hypothesis.",
                "retention": "No evaluator labels were opened.",
                "freshness": "No fresh evidence was produced.",
            }.items()
        },
        "rows": [],
        "independent_reduction": {"row_count": 0, "arm_summaries": {}},
        "sample_size_budget": {
            "intended_independent_units": 80,
            "observed_independent_units": 0,
            "excluded_independent_units": 80,
            "censored_independent_units": 0,
            "arms_do_not_multiply_units": True,
        },
        "guarded_update_ready_score": 0,
        "update_rule_contract": {
            "feedback_lag": 8,
            "block_count": 10,
            "block_size": 8,
            "state_schema": "carnot.guarded_update_state.v1",
        },
        "hardware_path": {
            "actual": "blocked_no_run",
            "fixed_point_mapping": "not reached",
            "measured_speedup": None,
            "speedup_claim": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [],
        "current_work_receipt": None,
        "phase_spans": [],
        "empirical_benefit_claim": None,
        "retention_claim": None,
    }
    value["inference_substrate"] = "blocked_before_update_fixture"
    value["inference_substrate_class"] = "blocked_no_run"
    value["planned_inference_substrate_class"] = "no_model_load"
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def validate_artifact(
    value: Mapping[str, Any], *, root: Path, require_terminal: bool = False
) -> JsonDict:
    """Recompute claims from raw rows, receipts, and current-work provenance."""

    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_mismatch")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("reproducibility_checksum_mismatch")
    if value.get("verdict_class") == "blocked":
        summary = value.get("gate_check_summary")
        if (
            not str(value.get("honest_verdict", "")).startswith("complete_blocked_")
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or not isinstance(summary.get("first_failure"), Mapping)
        ):
            raise ValueError("blocked_gate_summary_invalid")
        return {"blocked": True, "row_count": 0, "arm_summaries": {}}
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        raise ValueError("terminal_prefix_missing")
    if (
        value.get("verdict_class") != "circular_positive"
        or value.get("verifier_is_oracle") is not True
    ):
        raise ValueError("fixture_verdict_class_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("model_specs_nonempty")
    if (
        value.get("no_model_load") is not True
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or value.get("inference_substrate_class") != "no_model_load"
    ):
        raise ValueError("no_model_contract_invalid")
    if value.get("generator_weights_immutable") is not True:
        raise ValueError("generator_weights_mutated")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_PRINCIPLE_FIELDS) <= set(principles):
        raise ValueError("field_principles_incomplete")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rows_missing")
    reduction = reduce_rows(rows)
    if value.get("independent_reduction") != reduction:
        raise ValueError("independent_reduction_mismatch")
    receipts = value.get("validation_receipts")
    if not isinstance(receipts, list):
        raise ValueError("validation_receipts_missing")
    affected_passed = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    if require_terminal and not terminal_passed:
        raise ValueError("terminal_validation_incomplete")
    lifecycle = value.get("lifecycle_controls")
    lifecycle_passed = isinstance(lifecycle, Mapping) and all(
        item is True for item in lifecycle.values()
    )
    expected_ready = int(affected_passed and lifecycle_passed)
    if value.get("guarded_update_ready_score") != expected_ready:
        raise ValueError("readiness_score_mismatch")
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("observed_independent_units") != 80:
        raise ValueError("sample_size_budget_invalid")
    receipt = value.get("current_work_receipt")
    if not isinstance(receipt, Mapping) or validate_current_work_receipt(receipt, root=root):
        raise ValueError("current_work_receipt_invalid")
    gates = value.get("acceptance_gate_results")
    if (
        not isinstance(gates, Mapping)
        or gates.get("readiness", {}).get("result") is not bool(expected_ready)
        or gates.get("benefit", {}).get("result") is not False
        or gates.get("retention", {}).get("result") is not False
    ):
        raise ValueError("acceptance_gates_invalid")
    return reduction


def cold_replay(path: Path, *, root: Path, require_terminal: bool = True) -> JsonDict:
    value = _load_json(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(value, root=root, require_terminal=require_terminal)


def independent_reduce_artifact(path: Path, *, root: Path) -> JsonDict:
    value = _load_json(path)
    if not value or not isinstance(value.get("rows"), list):
        raise ValueError("artifact_unreadable_or_rows_missing")
    reduction = reduce_rows(value["rows"])
    if reduction != value.get("independent_reduction"):
        raise ValueError("independent_reduction_mismatch")
    validate_artifact(value, root=root, require_terminal=False)
    return reduction


def write_affected_manifest(path: Path) -> None:
    atomic_json(
        path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [MODULE_PATH.as_posix()],
            "static_paths": [WRAPPER_PATH.as_posix()],
        },
    )


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    private_root.mkdir(parents=True, exist_ok=True)
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=private_root / ".coverage.exp7603",
    )


def terminal_commands(
    candidate: Path, root: Path, *, allow_preterminal: bool = False
) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    cold = [
        python,
        "-u",
        WRAPPER_PATH.as_posix(),
        "--root",
        str(root),
        "--date",
        RUN_DATE,
        "--cold-replay",
        str(candidate),
    ]
    if allow_preterminal:
        cold.append("--allow-preterminal")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay", tuple(cold), "exact_candidate", 300.0
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                str(root),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
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
            "exact_candidate",
            300.0,
        ),
    ]


def _test_receipts() -> list[JsonDict]:
    return [
        {
            "name": name,
            "command": f"test-only:{name}",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "log_sha256": canonical_hash({"name": name}),
        }
        for name in (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _current_receipt(duration_ns: int, phase_spans: Sequence[Mapping[str, Any]]) -> JsonDict:
    return build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "model_loaded": False,
            "generator_weights_mutated": False,
            "residual_head_updated": True,
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=max(1, int(duration_ns)),
        phase_spans=phase_spans,
        small_ebm_training={"performed": True, "kind": "bounded_residual_brier_step"},
    )


def build_test_artifact(root: Path, work_dir: Path) -> JsonDict:
    checks, sources = collect_preconditions(root)
    stream, anchors = load_authenticated_inputs(root)
    evidence = run_fixture(stream, anchors, work_dir / "states")
    duration_ns = 10_000_000
    return build_artifact(
        evidence,
        checks,
        sources,
        _test_receipts(),
        _current_receipt(duration_ns, []),
        duration_s=duration_ns / 1_000_000_000,
    )


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    print(
        json.dumps(
            {
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **details,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _span(phase: str, phase_started: float, started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - started,
        "end_s": ended - started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def run_experiment(  # pragma: no cover - the declared CLI is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start", root=str(root))
    phase_started = time.monotonic()
    checks, sources = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next((row for row in checks if not row["passed"]), None)
    if failed is not None:
        blocked = build_blocked_artifact(
            failed, checks, sources, duration_s=time.monotonic() - started
        )
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "after_atomic_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "complete", completed_units=len(checks))

    progress(started, "input_custody", "before_authenticated_load")
    phase_started = time.monotonic()
    stream, anchors = load_authenticated_inputs(root)
    spans.append(_span("input_custody", phase_started, started, len(stream) + len(anchors)))
    progress(
        started,
        "input_custody",
        "after_authenticated_load",
        stream_units=len(stream),
        anchor_units=len(anchors),
    )

    private_state = Path(tempfile.mkdtemp(prefix="carnot-exp7603-state-", dir="/tmp"))
    progress(started, "lifecycle", "before_benchmark", blocks=10, arms=len(ARMS))
    phase_started = time.monotonic()
    evidence = run_fixture(stream, anchors, private_state)
    spans.append(_span("lifecycle", phase_started, started, len(evidence["block_results"])))
    progress(
        started,
        "lifecycle",
        "after_benchmark",
        completed_units=len(evidence["block_results"]),
    )

    manifest = raw_root / "affected_validation_manifest.json"
    progress(started, "manifest", "before_atomic_write")
    write_affected_manifest(manifest)
    progress(started, "manifest", "after_atomic_write")
    sources.append(
        {
            "kind": "conductor_pre_gate_record",
            "path": manifest.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest),
            "bytes": manifest.stat().st_size,
        }
    )
    for relative in (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    ):
        path = root / relative
        if path.is_file():
            sources.append(
                {
                    "kind": "conductor_pre_gate_record",
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )

    private_validation = Path(tempfile.mkdtemp(prefix="carnot-exp7603-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private_validation)
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    affected_passed = _receipts_pass(affected, validation_scope.REQUIRED_CHECK_NAMES)
    progress(started, "affected_validation", "after_subprocesses", passed=affected_passed)
    duration_ns = time.monotonic_ns() - started_ns
    candidate = build_artifact(
        evidence,
        checks,
        sources,
        affected,
        _current_receipt(duration_ns, spans),
        duration_s=duration_ns / 1_000_000_000,
        phase_spans=spans,
    )
    if not affected_passed:
        progress(started, "publish", "before_atomic_disqualified_validation")
        atomic_json(destination, candidate)
        progress(started, "publish", "after_atomic_disqualified_validation")
        return candidate
    validate_artifact(candidate, root=root, require_terminal=False)
    measured = raw_root / "measured_terminal_candidate.json"
    atomic_json(measured, candidate)

    provisional = terminal_commands(measured, root, allow_preterminal=True)
    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        provisional,
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)
    duration_ns = time.monotonic_ns() - started_ns
    final = build_artifact(
        evidence,
        checks,
        sources,
        [*affected, *terminal],
        _current_receipt(duration_ns, spans),
        duration_s=duration_ns / 1_000_000_000,
        phase_spans=spans,
    )
    if not terminal_passed:
        progress(started, "publish", "before_atomic_disqualified_terminal")
        atomic_json(destination, final)
        progress(started, "publish", "after_atomic_disqualified_terminal")
        return final
    validate_artifact(final, root=root, require_terminal=True)
    exact_candidate = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_candidate, final)

    exact_commands = terminal_commands(exact_candidate, root)
    progress(started, "exact_candidate", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        exact_commands,
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(started, "exact_candidate", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(raw_root / "exact_terminal_reader_outcomes.json", {"receipts": exact})
    progress(started, "publish", "before_atomic_terminal", path=str(destination))
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        verdict_class=final["verdict_class"],
        guarded_update_ready_score=final["guarded_update_ready_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--allow-preterminal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        print(
            json.dumps(
                cold_replay(
                    args.cold_replay,
                    root=root,
                    require_terminal=not args.allow_preterminal,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.independent_reduce is not None:
        print(
            json.dumps(
                independent_reduce_artifact(args.independent_reduce, root=root),
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    output = args.output
    if output is not None and not output.is_absolute():
        output = root / output
    run_experiment(root, args.date, output_path=output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
