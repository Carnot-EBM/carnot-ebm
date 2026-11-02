"""Qualify constrained cumulative-Brier recalibration on analytical fixtures.

The learner stores a small matrix and vector instead of past examples. This
prototype checks arithmetic and crash safety only. It does not use untouched
empirical labels, so its fixture gains cannot establish real-world benefit.

Spec refs: REQ-CL-7561 and SCENARIO-CL-7561-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.experiment_7534_v659_count_memory import CountArm, CountConfig
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7561-v661-recalibration-prototype"
SCHEMA = "carnot.exp7561.v661.recalibration_prototype.v1"
STATE_SCHEMA = "carnot.exp7561.v661.recalibration_state.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7561_v661_recalibration_prototype.json")
RAW_DIR = Path("results/raw/experiment_7561_v661_recalibration_prototype")
MODULE_PATH = Path("python/carnot/experiment_7561_v661_recalibration_prototype.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7561_v661_recalibration_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7561_v661_recalibration_prototype.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
V660_RESULT_PATH = Path("results/experiment_7550_v660_count_audit.json")

KNOT_COUNT = 9
KNOTS = np.linspace(0.0, 1.0, KNOT_COUNT)
RIDGE_MASS = 8.0
MOVEMENT_BOUND = 0.10
LOG_CLIP = 1e-6
SOLVER_TOLERANCE = 1e-8
SOLVER_MAX_ITERATIONS = 500
PRIMARY_SOLVER = "SLSQP"
INDEPENDENT_SOLVER = "COBYQA"
SOLVER_OBJECTIVE_TOLERANCE = 1e-7
ORDER_SEEDS = (7_568_001, 7_568_002, 7_568_003, 7_568_004, 7_568_005)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 7_561_011
FEEDBACK_DELAY = 8
RELEASE_BLOCK_SIZE = 8
RETENTION_CHECKPOINTS = (0, 40, 80, 120, 160)
ARMS = ("raw", "global_count", "local_count", "constrained", "shuffled_constrained")
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _finite_probability(probability: float) -> float:
    """Validate a probability before interpolation clamps endpoint noise."""

    value = float(probability)
    if not math.isfinite(value):
        raise ValueError("probability_not_finite")
    return min(1.0, max(0.0, value))


def piecewise_design(probability: float) -> np.ndarray:
    """Return the two nonzero interpolation weights for one probability."""

    value = _finite_probability(probability)
    if value >= 1.0:
        row = np.zeros(KNOT_COUNT)
        row[-1] = 1.0
        return row
    scaled = value * (KNOT_COUNT - 1)
    left = int(math.floor(scaled))
    fraction = scaled - left
    row = np.zeros(KNOT_COUNT)
    row[left] = 1.0 - fraction
    row[left + 1] = fraction
    return row


def map_probability(probability: float, theta: Sequence[float]) -> float:
    """Interpolate one fitted map after validating its nine values."""

    values = np.asarray(theta, dtype=float)
    if values.shape != (KNOT_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("theta_shape_or_finiteness_invalid")
    return float(piecewise_design(probability) @ values)


def binary_energies(probability: float) -> tuple[float, float]:
    """Represent a probability as two normalized binary energies."""

    value = min(1.0 - LOG_CLIP, max(LOG_CLIP, _finite_probability(probability)))
    return (-math.log1p(-value), -math.log(value))


def normalized_probability(energies: Sequence[float]) -> float:
    """Normalize two energies and return the probability of label one."""

    if len(energies) != 2:
        raise ValueError("binary_energy_requires_two_values")
    weights = [math.exp(-float(value)) for value in energies]
    return weights[1] / sum(weights)


def typed_decision(probability: float) -> JsonDict:
    """Choose the lowest registered cost, with escalation winning ties."""

    value = _finite_probability(probability)
    costs = {"accept": 5.0 * value, "reject": 1.0 - value, "escalate": 0.2}
    minimum = min(costs.values())
    action = next(
        name
        for name in ("escalate", "accept", "reject")
        if math.isclose(costs[name], minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    return {"action": action, "expected_costs": costs, "minimum_expected_cost": minimum}


def statistics_from_examples(
    probabilities: Sequence[float], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce examples to the matrix and vector used by cumulative Brier loss."""

    if len(probabilities) != len(labels):
        raise ValueError("probability_label_length_mismatch")
    gram = np.zeros((KNOT_COUNT, KNOT_COUNT))
    target = np.zeros(KNOT_COUNT)
    for probability, label in zip(probabilities, labels):
        if label not in (0, 1):
            raise ValueError("binary_label_required")
        design = piecewise_design(probability)
        gram += np.outer(design, design)
        target += int(label) * design
    return gram, target


def quadratic_objective(theta: Sequence[float], gram: np.ndarray, target: np.ndarray) -> float:
    """Evaluate the label-independent part of the frozen convex objective."""

    values = np.asarray(theta, dtype=float)
    return float(
        values @ gram @ values - 2.0 * target @ values + RIDGE_MASS * np.sum((values - KNOTS) ** 2)
    )


def _objective_gradient(theta: np.ndarray, gram: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2.0 * (gram @ theta - target + RIDGE_MASS * (theta - KNOTS))


def constraint_errors(theta: Sequence[float], *, tolerance: float = 1e-7) -> list[str]:
    """Name each violated map constraint instead of trusting solver success."""

    values = np.asarray(theta, dtype=float)
    errors: list[str] = []
    if values.shape != (KNOT_COUNT,) or not np.all(np.isfinite(values)):
        return ["theta_shape_or_finiteness"]
    if float(np.min(values)) < -tolerance or float(np.max(values)) > 1.0 + tolerance:
        errors.append("probability_range")
    if np.any(np.diff(values) < -tolerance):
        errors.append("monotonicity")
    if float(np.max(np.abs(values - KNOTS))) > MOVEMENT_BOUND + tolerance:
        errors.append("movement_bound")
    return errors


def _validate_statistics(gram: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(gram, dtype=float)
    vector = np.asarray(target, dtype=float)
    if matrix.shape != (KNOT_COUNT, KNOT_COUNT) or vector.shape != (KNOT_COUNT,):
        raise ValueError("sufficient_statistic_shape_invalid")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("sufficient_statistic_not_finite")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("gram_not_symmetric")
    return matrix, vector


def solve_constrained_map(
    gram: np.ndarray, target: np.ndarray, *, method: str = PRIMARY_SOLVER
) -> tuple[np.ndarray, JsonDict]:
    """Solve the frozen nine-variable convex program with a named SciPy method."""

    matrix, vector = _validate_statistics(gram, target)
    if method not in {PRIMARY_SOLVER, INDEPENDENT_SOLVER}:
        raise ValueError(f"solver_method_not_frozen:{method}")
    difference = np.zeros((KNOT_COUNT - 1, KNOT_COUNT))
    for index in range(KNOT_COUNT - 1):
        difference[index, index] = -1.0
        difference[index, index + 1] = 1.0
    lower = np.maximum(0.0, KNOTS - MOVEMENT_BOUND)
    upper = np.minimum(1.0, KNOTS + MOVEMENT_BOUND)
    options: JsonDict
    if method == PRIMARY_SOLVER:
        options = {"ftol": SOLVER_TOLERANCE, "maxiter": SOLVER_MAX_ITERATIONS, "disp": False}
    else:
        options = {
            "feasibility_tol": SOLVER_TOLERANCE,
            "final_tr_radius": SOLVER_TOLERANCE,
            "maxiter": SOLVER_MAX_ITERATIONS,
            "disp": False,
        }
    solver_kwargs: JsonDict = {
        "args": (matrix, vector),
        "method": method,
        "bounds": Bounds(lower, upper),
        "constraints": (LinearConstraint(difference, 0.0, np.inf),),
        "options": options,
    }
    if method == PRIMARY_SOLVER:
        solver_kwargs["jac"] = _objective_gradient
    result = minimize(quadratic_objective, KNOTS.copy(), **solver_kwargs)
    theta = np.asarray(result.x, dtype=float)
    errors = constraint_errors(theta)
    receipt = {
        "method": method,
        "tolerance": SOLVER_TOLERANCE,
        "iteration_cap": SOLVER_MAX_ITERATIONS,
        "converged": bool(result.success and not errors),
        "status": int(result.status),
        "iterations": int(getattr(result, "nit", 0)),
        "objective": quadratic_objective(theta, matrix, vector),
        "constraint_errors": errors,
        "message": str(result.message),
    }
    return theta, receipt


def solve_unconstrained_map(gram: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, JsonDict]:
    """Fit the fixture-only ablation without order or movement constraints."""

    matrix, vector = _validate_statistics(gram, target)
    theta = np.linalg.solve(matrix + RIDGE_MASS * np.eye(KNOT_COUNT), vector + RIDGE_MASS * KNOTS)
    return theta, {
        "method": "analytic_unconstrained",
        "converged": True,
        "objective": quadratic_objective(theta, matrix, vector),
        "constraint_errors": constraint_errors(theta),
    }


@dataclass(frozen=True)
class SolverConfig:
    """Freeze every value that can change the fitted probability map."""

    ridge_mass: float = RIDGE_MASS
    movement_bound: float = MOVEMENT_BOUND
    tolerance: float = SOLVER_TOLERANCE
    iteration_cap: int = SOLVER_MAX_ITERATIONS
    method: str = PRIMARY_SOLVER

    def __post_init__(self) -> None:
        observed = (
            self.ridge_mass,
            self.movement_bound,
            self.tolerance,
            self.iteration_cap,
            self.method,
        )
        expected = (
            RIDGE_MASS,
            MOVEMENT_BOUND,
            SOLVER_TOLERANCE,
            SOLVER_MAX_ITERATIONS,
            PRIMARY_SOLVER,
        )
        if observed != expected:
            raise ValueError("solver_configuration_not_frozen")


class SufficientStatisticMap:
    """Fit a probability map while retaining no raw labeled examples."""

    def __init__(
        self,
        gram: np.ndarray,
        target: np.ndarray,
        theta: np.ndarray,
        sample_count: int,
        processed_event_ids: set[str],
        solver_config: SolverConfig,
        constrained: bool = True,
        last_solver_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        self.gram, self.target = _validate_statistics(gram, target)
        self.theta = np.asarray(theta, dtype=float)
        if self.theta.shape != (KNOT_COUNT,) or not np.all(np.isfinite(self.theta)):
            raise ValueError("theta_shape_or_finiteness_invalid")
        self.sample_count = int(sample_count)
        self.processed_event_ids = set(processed_event_ids)
        self.solver_config = solver_config
        self.constrained = bool(constrained)
        self.last_solver_receipt = deepcopy(dict(last_solver_receipt or {}))

    @classmethod
    def create(cls, *, constrained: bool = True) -> SufficientStatisticMap:
        """Initialize the identity map before any feedback is visible."""

        return cls(
            np.zeros((KNOT_COUNT, KNOT_COUNT)),
            np.zeros(KNOT_COUNT),
            KNOTS.copy(),
            0,
            set(),
            SolverConfig(),
            constrained,
        )

    def predict(self, probability: float) -> float:
        return map_probability(probability, self.theta)

    def update_batch(self, rows: Sequence[tuple[str, float, int]]) -> JsonDict:
        """Validate, solve, and commit one whole release as one transaction."""

        seen: set[str] = set()
        designs: list[np.ndarray] = []
        labels: list[int] = []
        event_ids: list[str] = []
        for event_id, probability, label in rows:
            name = str(event_id)
            if name in seen or name in self.processed_event_ids:
                raise ValueError(f"duplicate_feedback:{name}")
            if label not in (0, 1):
                raise ValueError("binary_label_required")
            seen.add(name)
            event_ids.append(name)
            designs.append(piecewise_design(probability))
            labels.append(int(label))
        candidate_gram = self.gram.copy()
        candidate_target = self.target.copy()
        for design, label in zip(designs, labels):
            candidate_gram += np.outer(design, design)
            candidate_target += label * design
        if self.constrained:
            candidate_theta, receipt = solve_constrained_map(candidate_gram, candidate_target)
        else:
            candidate_theta, receipt = solve_unconstrained_map(candidate_gram, candidate_target)
        if receipt.get("converged") is not True:
            raise RuntimeError("recalibration_solver_did_not_converge")
        self.gram = candidate_gram
        self.target = candidate_target
        self.theta = candidate_theta
        self.sample_count += len(rows)
        self.processed_event_ids.update(event_ids)
        self.last_solver_receipt = deepcopy(receipt)
        return {**deepcopy(receipt), "sample_count": self.sample_count, "update_count": len(rows)}

    def to_payload(self) -> JsonDict:
        """Serialize only sufficient statistics and exact solver metadata."""

        return {
            "schema": "carnot.recalibration.sufficient_statistics.v1",
            "gram_shape": [KNOT_COUNT, KNOT_COUNT],
            "gram": self.gram.tolist(),
            "target": self.target.tolist(),
            "theta": self.theta.tolist(),
            "sample_count": self.sample_count,
            "processed_event_ids": sorted(self.processed_event_ids),
            "solver_config": asdict(self.solver_config),
            "constrained": self.constrained,
            "last_solver_receipt": deepcopy(self.last_solver_receipt),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> SufficientStatisticMap:
        if value.get("schema") != "carnot.recalibration.sufficient_statistics.v1":
            raise ValueError("sufficient_statistic_schema_mismatch")
        config = SolverConfig(**dict(value["solver_config"]))
        return cls(
            np.asarray(value["gram"], dtype=float),
            np.asarray(value["target"], dtype=float),
            np.asarray(value["theta"], dtype=float),
            int(value["sample_count"]),
            {str(item) for item in value["processed_event_ids"]},
            config,
            bool(value["constrained"]),
            value.get("last_solver_receipt") or {},
        )

    def state_hash(self) -> str:
        return canonical_hash(self.to_payload())


def fixture_count_config() -> CountConfig:
    """Use fixed label-free fit-role means for every analytical control."""

    means = tuple((index + 0.5) / 8.0 for index in range(8))
    return CountConfig(means, 0.5, 8.0)


class RecalibrationEventMachine:
    """Seal predictions before delayed feedback changes any learner arm."""

    def __init__(
        self,
        count_config: CountConfig,
        count_arms: Mapping[str, CountArm],
        constrained: SufficientStatisticMap,
        shuffled: SufficientStatisticMap,
        predictions: Mapping[str, Any] | None = None,
        release_receipts: Mapping[str, Any] | None = None,
        acknowledgments: set[int] | None = None,
        next_release_index: int = 0,
        journal: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.count_config = count_config
        self.count_arms = dict(count_arms)
        self.constrained = constrained
        self.shuffled = shuffled
        self._predictions = deepcopy(dict(predictions or {}))
        self.release_receipts = deepcopy(dict(release_receipts or {}))
        self.acknowledgments = set(acknowledgments or set())
        self.next_release_index = int(next_release_index)
        self.journal = [deepcopy(dict(row)) for row in (journal or [])]

    @classmethod
    def create(cls, count_config: CountConfig) -> RecalibrationEventMachine:
        count_arms = {
            "global_count": CountArm.create("global", count_config),
            "local_count": CountArm.create("local", count_config),
        }
        return cls(
            count_config,
            count_arms,
            SufficientStatisticMap.create(),
            SufficientStatisticMap.create(),
        )

    @property
    def predictions(self) -> JsonDict:
        """Return a copy so feedback cannot rewrite a sealed forecast."""

        return deepcopy(self._predictions)

    def _append_journal(self, operation: str, payload: Mapping[str, Any]) -> None:
        previous = self.journal[-1]["entry_hash"] if self.journal else "GENESIS"
        body = {
            "sequence": len(self.journal),
            "operation": operation,
            "payload": deepcopy(dict(payload)),
            "previous_hash": previous,
        }
        self.journal.append({**body, "entry_hash": canonical_hash(body)})

    def predict(self, event_id: str, probability: float, release_index: int) -> JsonDict:
        """Seal all five arms while this event's label is unavailable."""

        name = str(event_id)
        if name in self._predictions:
            raise ValueError(f"prediction_event_duplicate:{name}")
        if release_index < self.next_release_index:
            raise ValueError(f"prediction_release_already_closed:{release_index}")
        baseline = _finite_probability(probability)
        values = {
            "raw": baseline,
            "global_count": self.count_arms["global_count"].predict(baseline).probability,
            "local_count": self.count_arms["local_count"].predict(baseline).probability,
            "constrained": self.constrained.predict(baseline),
            "shuffled_constrained": self.shuffled.predict(baseline),
        }
        arms = {
            arm: {
                "probability": value,
                "energies": list(binary_energies(value)),
                "decision": typed_decision(value),
            }
            for arm, value in values.items()
        }
        row = {
            "event_id": name,
            "base_probability": baseline,
            "release_index": int(release_index),
            "label_available_at_prediction": False,
            "arms": arms,
        }
        self._predictions[name] = deepcopy(row)
        self._append_journal("prediction", row)
        return deepcopy(row)

    @staticmethod
    def _canonical_feedback(feedback: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        rows = [(str(event_id), int(label)) for event_id, label in feedback]
        if len({event_id for event_id, _label in rows}) != len(rows):
            raise ValueError("feedback_event_duplicate_in_release")
        if any(label not in (0, 1) for _event_id, label in rows):
            raise ValueError("binary_label_required")
        return rows

    def release(self, release_index: int, feedback: Sequence[tuple[str, int]]) -> JsonDict:
        """Apply one available block atomically to each matched mutable arm."""

        if release_index < self.next_release_index:
            raise ValueError(f"duplicate_feedback_release:{release_index}")
        if release_index != self.next_release_index:
            raise ValueError(
                f"release_out_of_order:expected={self.next_release_index}:observed={release_index}"
            )
        rows = self._canonical_feedback(feedback)
        for event_id, _label in rows:
            prediction = self._predictions.get(event_id)
            if prediction is None:
                raise ValueError(f"feedback_event_unknown:{event_id}")
            if prediction["release_index"] != release_index:
                raise ValueError(f"feedback_prerelease:{event_id}")
        candidate_counts = {
            name: CountArm.from_payload(arm.to_payload(), self.count_config)
            for name, arm in self.count_arms.items()
        }
        candidate_constrained = SufficientStatisticMap.from_payload(self.constrained.to_payload())
        candidate_shuffled = SufficientStatisticMap.from_payload(self.shuffled.to_payload())
        rotated = rows[1:] + rows[:1] if len(rows) > 1 else rows
        constrained_rows: list[tuple[str, float, int]] = []
        shuffled_rows: list[tuple[str, float, int]] = []
        origins: list[str] = []
        for (event_id, label), (origin_id, shuffled_label) in zip(rows, rotated):
            probability = float(self._predictions[event_id]["base_probability"])
            candidate_counts["global_count"].update(event_id, probability, label)
            candidate_counts["local_count"].update(event_id, probability, label)
            constrained_rows.append((event_id, probability, label))
            shuffled_rows.append((event_id, probability, shuffled_label))
            origins.append(origin_id)
        constrained_receipt = candidate_constrained.update_batch(constrained_rows)
        shuffled_receipt = candidate_shuffled.update_batch(shuffled_rows)
        receipt = {
            "release_index": release_index,
            "feedback": [[event_id, label] for event_id, label in rows],
            "update_count": len(rows),
            "shuffled_update_count": len(rotated),
            "shuffled_label_origins": origins,
            "constrained_solver": constrained_receipt,
            "shuffled_solver": shuffled_receipt,
            "constrained_theta": candidate_constrained.theta.tolist(),
            "shuffled_theta": candidate_shuffled.theta.tolist(),
        }
        self.count_arms = candidate_counts
        self.constrained = candidate_constrained
        self.shuffled = candidate_shuffled
        self.release_receipts[str(release_index)] = deepcopy(receipt)
        self.next_release_index += 1
        self._append_journal("release", receipt)
        return deepcopy(receipt)

    def acknowledge(self, release_index: int) -> None:
        """Record durability only after the matching release was committed."""

        if str(release_index) not in self.release_receipts:
            raise ValueError(f"acknowledgment_without_release:{release_index}")
        if release_index not in self.acknowledgments:
            self.acknowledgments.add(release_index)
            self._append_journal("acknowledgment", {"release_index": release_index})

    def to_payload(self) -> JsonDict:
        """Serialize predictions, receipts, and the complete restart state."""

        return {
            "schema": STATE_SCHEMA,
            "count_config": asdict(self.count_config),
            "arms": {
                "raw": {"kind": "immutable_raw"},
                "global_count": self.count_arms["global_count"].to_payload(),
                "local_count": self.count_arms["local_count"].to_payload(),
                "constrained": self.constrained.to_payload(),
                "shuffled_constrained": self.shuffled.to_payload(),
            },
            "predictions": deepcopy(self._predictions),
            "release_receipts": deepcopy(self.release_receipts),
            "acknowledgments": sorted(self.acknowledgments),
            "next_release_index": self.next_release_index,
            "journal": deepcopy(self.journal),
        }

    @staticmethod
    def _validate_journal(rows: Sequence[Mapping[str, Any]]) -> None:
        previous = "GENESIS"
        for index, row in enumerate(rows):
            body = {
                "sequence": row.get("sequence"),
                "operation": row.get("operation"),
                "payload": row.get("payload"),
                "previous_hash": row.get("previous_hash"),
            }
            if body["sequence"] != index or body["previous_hash"] != previous:
                raise ValueError(f"journal_chain_mismatch:{index}")
            if row.get("entry_hash") != canonical_hash(body):
                raise ValueError(f"journal_hash_mismatch:{index}")
            previous = str(row["entry_hash"])

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> RecalibrationEventMachine:
        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("state_schema_mismatch")
        arms = value.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            raise ValueError("state_arm_set_mismatch")
        journal = list(value["journal"])
        cls._validate_journal(journal)
        config_value = value["count_config"]
        config = CountConfig(
            tuple(float(item) for item in config_value["bin_means"]),
            float(config_value["global_mean"]),
            float(config_value["kappa"]),
        )
        count_arms = {
            name: CountArm.from_payload(arms[name], config)
            for name in ("global_count", "local_count")
        }
        return cls(
            config,
            count_arms,
            SufficientStatisticMap.from_payload(arms["constrained"]),
            SufficientStatisticMap.from_payload(arms["shuffled_constrained"]),
            predictions=value["predictions"],
            release_receipts=value["release_receipts"],
            acknowledgments={int(item) for item in value["acknowledgments"]},
            next_release_index=int(value["next_release_index"]),
            journal=journal,
        )

    def state_hash(self) -> str:
        return canonical_hash(self.to_payload())

    def numerical_state_bytes(self) -> int:
        """Measure the two map statistics without counting replay metadata."""

        state = {
            "constrained": self.constrained.to_payload(),
            "shuffled_constrained": self.shuffled.to_payload(),
            "count_arms": {name: arm.to_payload() for name, arm in sorted(self.count_arms.items())},
        }
        return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())

    def save(self, path: Path) -> None:
        atomic_json(path, self.to_payload())

    @classmethod
    def load(cls, path: Path) -> RecalibrationEventMachine:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("state_payload_not_object")
        return cls.from_payload(value)


def _brier(probability: float, label: int) -> float:
    return (float(probability) - int(label)) ** 2


def analytical_fixtures() -> dict[str, list[JsonDict]]:
    """Construct three deterministic controls without using empirical labels."""

    shifted: list[JsonDict] = []
    no_shift: list[JsonDict] = []
    recurrence: list[JsonDict] = []
    for index in range(160):
        shifted_probability = 0.0 if index < 80 else (index % 8 + 0.5) / 8.0
        shifted_label = 1 if index < 80 else int(index % 8 < round(shifted_probability * 8))
        shifted.append(
            {
                "event_id": f"calibration-shift-{index:03d}",
                "probability": shifted_probability,
                "label": shifted_label,
            }
        )
        probability = 0.25 if index % 2 == 0 else 0.75
        cycle = index % 8
        label = int(cycle in ({0, 4} if probability == 0.25 else {1, 2, 3, 5, 6, 7}))
        no_shift.append(
            {
                "event_id": f"no-shift-{index:03d}",
                "probability": probability,
                "label": label,
            }
        )
        recurrence_probability = 0.15 if (index // 8) % 2 == 0 else 0.85
        recurrence_label = int(recurrence_probability > 0.5)
        if index % 16 in {0, 1}:
            recurrence_label = 1 - recurrence_label
        recurrence.append(
            {
                "event_id": f"recurrence-{index:03d}",
                "probability": recurrence_probability,
                "label": recurrence_label,
            }
        )
    return {
        "calibration_shift": shifted,
        "no_shift": no_shift,
        "recurrence": recurrence,
    }


def _replay_fixture(
    fixture: str, events: Sequence[Mapping[str, Any]], checkpoint_root: Path
) -> JsonDict:
    """Replay delayed releases twice, with one path crossing a cold reload."""

    baseline = RecalibrationEventMachine.create(fixture_count_config())
    restarted = RecalibrationEventMachine.create(fixture_count_config())
    future_rejections = 0
    restart_mismatches = 0
    for block_start in range(0, len(events), RELEASE_BLOCK_SIZE):
        block = events[block_start : block_start + RELEASE_BLOCK_SIZE]
        release_index = block_start // RELEASE_BLOCK_SIZE
        for row in block:
            for machine in (baseline, restarted):
                machine.predict(str(row["event_id"]), float(row["probability"]), release_index)
        if release_index == 0:
            try:
                restarted.release(1, [])
            except ValueError as error:
                future_rejections += int("release_out_of_order" in str(error))
        feedback = [(str(row["event_id"]), int(row["label"])) for row in block]
        baseline.release(release_index, feedback)
        restarted.release(release_index, feedback)
        baseline.acknowledge(release_index)
        restarted.acknowledge(release_index)
        if release_index == 9:
            path = checkpoint_root / f"{fixture}.json"
            restarted.save(path)
            restarted = RecalibrationEventMachine.load(path)
            restart_mismatches += int(restarted.to_payload() != baseline.to_payload())
    restart_mismatches += int(restarted.to_payload() != baseline.to_payload())
    duplicate_rejections = 0
    final_block = events[-RELEASE_BLOCK_SIZE:]
    try:
        restarted.release(
            len(events) // RELEASE_BLOCK_SIZE - 1,
            [(str(row["event_id"]), int(row["label"])) for row in final_block],
        )
    except ValueError as error:
        duplicate_rejections += int("duplicate_feedback_release" in str(error))

    probabilities = [float(row["probability"]) for row in events]
    labels = [int(row["label"]) for row in events]
    gram, target = statistics_from_examples(probabilities, labels)
    unconstrained, unconstrained_receipt = solve_unconstrained_map(gram, target)
    constrained_predictions = [baseline.constrained.predict(value) for value in probabilities]
    raw_brier = float(
        np.mean([_brier(value, label) for value, label in zip(probabilities, labels)])
    )
    constrained_brier = float(
        np.mean([_brier(value, label) for value, label in zip(constrained_predictions, labels)])
    )
    dense = np.linspace(0.0, 1.0, 1001)
    normalized = [
        normalized_probability(binary_energies(baseline.constrained.predict(float(value))))
        for value in dense
    ]
    log_safe = [
        min(1.0 - LOG_CLIP, max(LOG_CLIP, baseline.constrained.predict(float(value))))
        for value in dense
    ]
    return {
        "fixture": fixture,
        "event_count": len(events),
        "raw_brier": raw_brier,
        "constrained_brier": constrained_brier,
        "brier_improvement": raw_brier - constrained_brier,
        "final_theta": baseline.constrained.theta.tolist(),
        "maximum_movement": float(np.max(np.abs(baseline.constrained.theta - KNOTS))),
        "unconstrained_maximum_movement": float(np.max(np.abs(unconstrained - KNOTS))),
        "unconstrained_solver": unconstrained_receipt,
        "parameter_changed": bool(np.max(np.abs(baseline.constrained.theta - KNOTS)) > 1e-6),
        "monotonicity_violations": int(
            np.count_nonzero(np.diff(baseline.constrained.theta) < -1e-8)
        ),
        "movement_violations": int(
            np.count_nonzero(np.abs(baseline.constrained.theta - KNOTS) > 0.1 + 1e-8)
        ),
        "normalization_failures": int(
            np.count_nonzero(np.abs(np.asarray(normalized) - log_safe) > 1e-12)
        ),
        "duplicate_feedback_rejections": duplicate_rejections,
        "future_label_rejections": future_rejections,
        "restart_mismatches": restart_mismatches,
        "final_state_hash": baseline.state_hash(),
        "state_bytes": baseline.numerical_state_bytes(),
        "disposition": "complete",
    }


def run_fixture_panel(checkpoint_root: Path) -> JsonDict:
    """Qualify arithmetic and lifecycle behavior with oracle-defined cases."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows = [
        _replay_fixture(name, events, checkpoint_root)
        for name, events in analytical_fixtures().items()
    ]
    parameter_changes = sum(int(row["parameter_changed"]) for row in rows)
    monotonicity_violations = sum(int(row["monotonicity_violations"]) for row in rows)
    movement_violations = sum(int(row["movement_violations"]) for row in rows)
    normalization_failures = sum(int(row["normalization_failures"]) for row in rows)
    duplicates = sum(int(row["duplicate_feedback_rejections"]) for row in rows)
    futures = sum(int(row["future_label_rejections"]) for row in rows)
    restarts = sum(int(row["restart_mismatches"]) for row in rows)
    shift = next(row for row in rows if row["fixture"] == "calibration_shift")
    gates_passed = bool(
        parameter_changes >= 2
        and monotonicity_violations == 0
        and movement_violations == 0
        and normalization_failures == 0
        and duplicates == len(rows)
        and futures == len(rows)
        and restarts == 0
        and shift["constrained_brier"] < shift["raw_brier"]
        and shift["unconstrained_maximum_movement"] > MOVEMENT_BOUND
    )
    return {
        "rows": rows,
        "fixture_gates_passed": gates_passed,
        "parameter_change_count": parameter_changes,
        "monotonicity_violation_count": monotonicity_violations,
        "movement_violation_count": movement_violations,
        "normalization_failure_count": normalization_failures,
        "duplicate_feedback_rejection_count": duplicates,
        "future_label_rejection_count": futures,
        "restart_mismatch_count": restarts,
        "verifier_is_oracle": True,
    }


def run_numerical_qualification() -> JsonDict:
    """Cross-check solvers and sweep dense single-probability sanity cases."""

    paired_cases = (
        ([0.0] * 32, [0] * 32),
        ([0.0] * 32, [1] * 32),
        ([1.0] * 32, [0] * 32),
        ([1.0] * 32, [1] * 32),
        ([index / 40 for index in range(41)], [index % 2 for index in range(41)]),
    )
    convergence_failures = 0
    constraint_failures = 0
    parity_failures = 0
    objective_deltas: list[float] = []
    solver_rows: list[JsonDict] = []
    for case_index, (probabilities, labels) in enumerate(paired_cases):
        gram, target = statistics_from_examples(probabilities, labels)
        primary, primary_receipt = solve_constrained_map(gram, target)
        independent, independent_receipt = solve_constrained_map(
            gram, target, method=INDEPENDENT_SOLVER
        )
        delta = abs(
            quadratic_objective(primary, gram, target)
            - quadratic_objective(independent, gram, target)
        )
        convergence_failures += int(
            primary_receipt["converged"] is not True or independent_receipt["converged"] is not True
        )
        constraint_failures += int(
            bool(constraint_errors(primary) or constraint_errors(independent))
        )
        parity_failures += int(delta > SOLVER_OBJECTIVE_TOLERANCE)
        objective_deltas.append(delta)
        solver_rows.append(
            {
                "case": case_index,
                "primary": primary_receipt,
                "independent": independent_receipt,
                "objective_delta": delta,
                "maximum_theta_delta": float(np.max(np.abs(primary - independent))),
            }
        )
    dense_failures = 0
    for probability in np.linspace(0.0, 1.0, 101):
        label = int(probability >= 0.5)
        gram, target = statistics_from_examples([float(probability)] * 8, [label] * 8)
        theta, receipt = solve_constrained_map(gram, target)
        dense_failures += int(receipt["converged"] is not True or bool(constraint_errors(theta)))
        value = map_probability(float(probability), theta)
        dense_failures += int(
            abs(
                normalized_probability(binary_energies(value))
                - min(1.0 - LOG_CLIP, max(LOG_CLIP, value))
            )
            > 1e-12
        )
    convergence_failures += dense_failures
    maximum_delta = max(objective_deltas, default=0.0)
    passed = (
        not any((convergence_failures, constraint_failures, parity_failures))
        and maximum_delta <= SOLVER_OBJECTIVE_TOLERANCE
    )
    return {
        "passed": passed,
        "primary_solver": PRIMARY_SOLVER,
        "independent_solver": INDEPENDENT_SOLVER,
        "solver_pair_count": len(solver_rows),
        "dense_scalar_case_count": 101,
        "convergence_failure_count": convergence_failures,
        "constraint_failure_count": constraint_failures,
        "solver_parity_failure_count": parity_failures,
        "maximum_objective_delta": maximum_delta,
        "solver_rows": solver_rows,
    }


def freeze_learning_protocol() -> JsonDict:
    """Freeze future empirical roles and label-blind orders before capture."""

    online_ids = [f"tool-online-{index:03d}" for index in range(160)]
    retention_ids = [f"evaluator-retention-{index:03d}" for index in range(80)]
    orders: dict[str, list[str]] = {}
    for seed in ORDER_SEEDS:
        order = online_ids.copy()
        random.Random(seed).shuffle(order)
        orders[str(seed)] = order
    protocol: JsonDict = {
        "schema": "carnot.exp7561.frozen_learning_protocol.v1",
        "frozen_before_empirical_capture": True,
        "label_blind_freeze": True,
        "online_role": "existing_untouched_tool_online",
        "retention_role": "evaluator_only_test",
        "fit_role": "probability_means_only",
        "online_group_count": 160,
        "retention_group_count": 80,
        "online_group_ids": online_ids,
        "retention_group_ids": retention_ids,
        "order_seeds": list(ORDER_SEEDS),
        "orders": orders,
        "feedback_delay": FEEDBACK_DELAY,
        "release_block_size": RELEASE_BLOCK_SIZE,
        "audit_budget": "full",
        "retention_checkpoints": list(RETENTION_CHECKPOINTS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "controls": ["raw", "global_count", "local_count", "shuffled_constrained"],
        "count_prior_mass": 8.0,
        "fit_role_means_are_label_free": True,
        "objective": {
            "kind": "cumulative_squared_loss_convex_brier",
            "ridge_mass": RIDGE_MASS,
            "not_beta_count_odds_shift": True,
            "not_expert_mixture": True,
            "not_generator_update": True,
        },
        "solver": {
            "primary": PRIMARY_SOLVER,
            "independent": INDEPENDENT_SOLVER,
            "tolerance": SOLVER_TOLERANCE,
            "iteration_cap": SOLVER_MAX_ITERATIONS,
        },
        "map": {
            "knot_locations": KNOTS.tolist(),
            "initial_values": KNOTS.tolist(),
            "monotone": True,
            "movement_bound": MOVEMENT_BOUND,
        },
        "typed_policy_costs": {"accept": "5q", "reject": "1-q", "escalate": 0.2},
        "tie_breaker": "escalate",
        "retention_updates_state": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def frozen_state_schema() -> JsonDict:
    """Describe the bounded numerical state used by a CPU or Rust port."""

    return {
        "schema": STATE_SCHEMA,
        "numerical_state": {
            "gram": {"shape": [9, 9], "dtype": "float64"},
            "target": {"shape": [9], "dtype": "float64"},
            "theta": {"shape": [9], "dtype": "float64"},
            "processed_event_ids": "sorted_string_set",
        },
        "raw_examples_retained": False,
        "restart_metadata": [
            "sealed_predictions",
            "release_receipts",
            "acknowledgments",
            "next_release_index",
            "hash_chained_journal",
        ],
    }


def run_replay_benchmark(
    protocol: Mapping[str, Any],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    progress_hook: Callable[[int], None] | None = None,
) -> JsonDict:
    """Run every registered event and release with deterministic resampling."""

    if replicates <= 0:
        raise ValueError("benchmark_replicates_must_be_positive")
    fixtures = analytical_fixtures()["calibration_shift"]
    probabilities = np.asarray([row["probability"] for row in fixtures], dtype=float)
    labels = np.asarray([row["label"] for row in fixtures], dtype=int)
    online_ids = list(protocol["online_group_ids"])
    positions = {event_id: index for index, event_id in enumerate(online_ids)}
    order_positions = {
        seed: np.asarray([positions[event_id] for event_id in protocol["orders"][str(seed)]])
        for seed in ORDER_SEEDS
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    started = time.monotonic()
    events_completed = 0
    order_replays = 0
    loss_sum = 0.0
    for replicate in range(replicates):
        sampled = rng.integers(0, len(fixtures), size=len(fixtures))
        for seed in ORDER_SEEDS:
            learner = SufficientStatisticMap.create()
            ordered = sampled[order_positions[seed]]
            for block_start in range(0, len(ordered), RELEASE_BLOCK_SIZE):
                block = ordered[block_start : block_start + RELEASE_BLOCK_SIZE]
                updates: list[tuple[str, float, int]] = []
                for offset, source_index in enumerate(block):
                    probability = float(probabilities[source_index])
                    label = int(labels[source_index])
                    loss_sum += _brier(learner.predict(probability), label)
                    arrival = block_start + offset
                    updates.append(
                        (f"rep-{replicate}-seed-{seed}-arrival-{arrival}", probability, label)
                    )
                learner.update_batch(updates)
                events_completed += len(block)
            order_replays += 1
        if progress_hook is not None and ((replicate + 1) % 10 == 0 or replicate + 1 == replicates):
            progress_hook(replicate + 1)
    measured = time.monotonic() - started
    projected = measured * BOOTSTRAP_REPLICATES / replicates
    reserve_s = 300.0
    return {
        "benchmark_scope": "constrained_predict_release_update_fixture_replay",
        "bootstrap_replicates_planned": replicates,
        "bootstrap_replicates_completed": replicates,
        "order_replays_completed": order_replays,
        "events_completed": events_completed,
        "failed": 0,
        "censored": 0,
        "unstarted": 0,
        "mean_prequential_brier": loss_sum / events_completed,
        "measured_duration_s": measured,
        "projected_duration_s": projected,
        "validation_reserve_s": reserve_s,
        "compute_budget_s": 2400.0,
        "fits_2400_seconds_with_reserve": projected + reserve_s <= 2400.0,
    }


def _sidecar(path: Path, root: Path) -> JsonDict:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "scope": "current_work",
    }


def write_raw_sidecars(
    root: Path, protocol: Mapping[str, Any], state_schema: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Publish the protocol and state schema before a terminal artifact refers to them."""

    raw_root = root / RAW_DIR
    protocol_path = raw_root / "frozen_learning_protocol.json"
    schema_path = raw_root / "numerical_state_schema.json"
    atomic_json(protocol_path, protocol)
    atomic_json(schema_path, state_schema)
    return {
        "frozen_learning_protocol": _sidecar(protocol_path, root),
        "numerical_state_schema": _sidecar(schema_path, root),
    }


def read_raw_sidecar(root: Path, receipt: Mapping[str, Any]) -> JsonDict:
    """Rehash one raw object before an independent reader trusts its fields."""

    path = root / str(receipt.get("path") or "")
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError("raw_sidecar_hash_mismatch")
    if path.stat().st_size != receipt.get("bytes"):
        raise ValueError("raw_sidecar_size_mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("raw_sidecar_not_object")
    return dict(value)


REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7534_v659_count_memory.py"),
    Path("python/carnot/experiment_7506_v657_causal_prototype.py"),
    Path("python/carnot/experiment_7549_v660_count_learning.py"),
    V660_RESULT_PATH,
    SPEC_PATH,
    Path("research-references.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _precondition(
    check: str,
    upstream: str,
    path_or_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    required: bool = True,
) -> JsonDict:
    """Record exact operands so a blocked run cannot hide its prerequisite."""

    return {
        "check": check,
        "upstream": upstream,
        "path_or_field": path_or_field,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": bool(passed),
        "required": required,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Authenticate required files, requirements, history, and CPU resources."""

    checks: list[JsonDict] = []
    for path in REQUIRED_INPUT_PATHS:
        present = (root / path).is_file()
        checks.append(
            _precondition(
                f"required_input:{path.as_posix()}",
                path.as_posix(),
                path.as_posix(),
                "present_file",
                "present_file" if present else "missing",
                present,
            )
        )
    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    checks.append(
        _precondition(
            "spec_requirement_present",
            SPEC_PATH.as_posix(),
            "REQ-CL-7561 and SCENARIO-CL-7561-*",
            True,
            "REQ-CL-7561" in spec_text and "SCENARIO-CL-7561-NUMERICAL" in spec_text,
            "REQ-CL-7561" in spec_text and "SCENARIO-CL-7561-NUMERICAL" in spec_text,
        )
    )
    historical = _load_object(root / V660_RESULT_PATH)
    expected_verdict = "complete_null_count_claims_qualified_benefit_gate_failed"
    observed_verdict = historical.get("honest_verdict")
    checks.append(
        _precondition(
            "v660_historical_verdict_preserved",
            V660_RESULT_PATH.as_posix(),
            "honest_verdict",
            expected_verdict,
            observed_verdict,
            observed_verdict == expected_verdict,
        )
    )
    checks.append(
        _precondition(
            "scipy_available",
            "scipy.optimize",
            "minimize",
            True,
            callable(minimize),
            callable(minimize),
        )
    )
    cpu_count = os.cpu_count() or 0
    checks.append(
        _precondition(
            "logical_cpu_available", "host", "logical_cpu_count", ">=1", cpu_count, cpu_count >= 1
        )
    )
    return checks


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for path in REQUIRED_INPUT_PATHS
        if (root / path).is_file()
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = {name: 0 for name in names}
    passed = {name: False for name in names}
    for row in receipts:
        name = str(row.get("name"))
        if name in counts:
            counts[name] += 1
            passed[name] = bool(
                row.get("passed") is True
                and row.get("exit_code") == 0
                and row.get("timed_out") is not True
            )
    return all(counts[name] == 1 and passed[name] for name in names)


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_checks": [str(row["check"]) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def _acceptance_gates(
    panel: Mapping[str, Any],
    numerical: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    preconditions_passed = all(
        row.get("passed") is True for row in preconditions if row.get("required") is True
    )
    affected_passed = _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_present = any(
        str(row.get("name")) in TERMINAL_CHECK_NAMES for row in validation_receipts
    )
    terminal_passed = not terminal_present or _receipts_pass(
        validation_receipts, TERMINAL_CHECK_NAMES
    )
    return [
        _gate(
            "preconditions",
            "validity",
            True,
            preconditions_passed,
            "eq",
            preconditions_passed,
            "Missing required evidence cannot become a measurement.",
        ),
        _gate(
            "numerical_qualification",
            "readiness",
            True,
            numerical.get("passed"),
            "eq",
            numerical.get("passed") is True,
            "A nonconverged map is not ready for empirical use.",
        ),
        _gate(
            "fixture_lifecycle",
            "readiness",
            True,
            panel.get("fixture_gates_passed"),
            "eq",
            panel.get("fixture_gates_passed") is True,
            "Chronology and restart must pass before efficacy is measured.",
        ),
        _gate(
            "registered_replay_count",
            "readiness",
            5000,
            benchmark.get("order_replays_completed"),
            "eq",
            benchmark.get("order_replays_completed") == 5000,
            "The benchmark count cannot shrink after outcomes are visible.",
        ),
        _gate(
            "registered_event_count",
            "readiness",
            800000,
            benchmark.get("events_completed"),
            "eq",
            benchmark.get("events_completed") == 800000,
            "Every replay must retain all 160 events.",
        ),
        _gate(
            "compute_budget_with_reserve",
            "readiness",
            True,
            benchmark.get("fits_2400_seconds_with_reserve"),
            "eq",
            benchmark.get("fits_2400_seconds_with_reserve") is True,
            "Feasibility needs time for both learning and validation.",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            affected_passed,
            "eq",
            affected_passed,
            "Invalid scoped checks cannot support scientific readiness.",
        ),
        _gate(
            "terminal_validation",
            "validity",
            True,
            terminal_passed,
            "eq",
            terminal_passed,
            "Fresh readers must agree before atomic publication.",
        ),
        _gate(
            "empirical_benefit",
            "benefit",
            "not_measured",
            "analytical_fixtures_only",
            "eq",
            False,
            "Oracle-defined fixture gains cannot establish empirical value.",
        ),
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    specific = {
        "experiment_id": "Bind the exact task, milestone, and run date.",
        "preconditions_checked": "Prevent missing inputs from becoming fallback data.",
        "MODEL_SPECS": "Make no-model work explicit.",
        "model_specs": "Keep resolved model identity empty when no model runs.",
        "model_invoked": "Separate current calls from historical evidence.",
        "inference_substrate_class": "Prevent understated model work.",
        "inference_substrate": "Name artifact aggregation as the source of the result.",
        "duration_s": "Keep current monotonic work separate from historical time.",
        "rows": "Retain every compared fixture with an absolute metric.",
        "sample_size_budget": "Keep attempted, failed, censored, and unstarted units visible.",
        "acceptance_gate_results": "Separate validity, readiness, and benefit.",
        "honest_verdict": "Use a closed terminal disposition.",
        "verifier_is_oracle": "Prevent constructed truth from certifying source truth.",
        "recalibration_ready_score": "Require numerical, chronology, and restart qualification.",
        "learning_compute_feasible_score": "Require the complete registered replay to fit its budget.",
        "maximum_prediction_movement": "Keep a mathematical bound distinct from retention evidence.",
        "state_bytes": "Show the small-state hardware path without claiming board speed.",
    }
    return {
        field: specific.get(field, f"Bind {field} so independent replay detects drift.")
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def build_artifact(
    *,
    panel: Mapping[str, Any],
    numerical: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    raw_sidecars: Mapping[str, Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble one terminal record while keeping gate classes independent."""

    affected_passed = _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_present = any(
        str(row.get("name")) in TERMINAL_CHECK_NAMES for row in validation_receipts
    )
    terminal_passed = not terminal_present or _receipts_pass(
        validation_receipts, TERMINAL_CHECK_NAMES
    )
    validation_passed = affected_passed and terminal_passed
    mechanism_ready = bool(
        numerical.get("passed") is True and panel.get("fixture_gates_passed") is True
    )
    compute_feasible = bool(
        benchmark.get("order_replays_completed") == 5000
        and benchmark.get("events_completed") == 800000
        and benchmark.get("fits_2400_seconds_with_reserve") is True
    )
    ready_score = int(mechanism_ready and validation_passed)
    compute_score = int(compute_feasible)
    if not validation_passed:
        honest_verdict = "complete_disqualified_required_validation"
        verdict_class = "disqualified"
    elif ready_score == 1 and compute_score == 1:
        honest_verdict = "complete_circular_positive_fixture_qualification"
        verdict_class = "circular_positive"
    else:
        honest_verdict = "complete_null_recalibration_not_ready"
        verdict_class = "null"
    gates = _acceptance_gates(
        panel, numerical, benchmark, validation_receipts, preconditions_checked
    )
    rows = [deepcopy(dict(row)) for row in panel.get("rows") or []]
    maximum_observed = max((float(row["maximum_movement"]) for row in rows), default=0.0)
    state_bytes = max((int(row["state_bytes"]) for row in rows), default=0)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Jointly constrained cumulative-Brier recalibration prototype",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {
            "source": V660_RESULT_PATH.as_posix(),
            "counted_as_current": False,
        },
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "work": "analytical_fixture_qualification_and_cpu_replay",
            "generated_tokens": 0,
            "model_loads": 0,
        },
        "execution_venue": "host",
        "compute_details": {
            "device_type": "CPU",
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "cwd": str(REPO_ROOT)},
        "random_seed": {
            "model_seed": None,
            "fitting_seed": None,
            "ordering_seeds": list(ORDER_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": dict(source_hashes),
        "input_roles": {
            "historical_context": V660_RESULT_PATH.as_posix(),
            "measurement": "analytical_fixtures",
            "future_empirical": "untouched_tool_online_groups_not_opened",
        },
        "raw_sidecars": deepcopy(dict(raw_sidecars)),
        "rows": rows,
        "numerical_qualification": deepcopy(dict(numerical)),
        "fixture_summary": {key: deepcopy(value) for key, value in panel.items() if key != "rows"},
        "benchmark_receipt": deepcopy(dict(benchmark)),
        "frozen_learning_protocol": {
            "protocol_hash": freeze_learning_protocol()["protocol_hash"] if raw_sidecars else None,
            "empirical_capture_started": False,
        },
        "sample_size_budget": {
            "analytical_fixture_cases": {
                "planned": 3,
                "attempted": len(rows),
                "completed": len(rows),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 3 - len(rows),
            },
            "benchmark_order_replays": {
                "planned": 5000,
                "attempted": int(benchmark.get("order_replays_completed", 0)),
                "completed": int(benchmark.get("order_replays_completed", 0)),
                "excluded": 0,
                "failed": int(benchmark.get("failed", 0)),
                "censored": int(benchmark.get("censored", 0)),
                "unstarted": int(benchmark.get("unstarted", 0)),
            },
            "future_empirical_online_groups": {
                "planned": 160,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 160,
            },
            "future_retention_groups": {
                "planned": 80,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 80,
            },
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": {"unrelated_debt_required_for_this_result": False},
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "terminal_status": "complete",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "positive_claim": False,
        "empirical_benefit_measured": False,
        "confirmatory_benefit_score": 0,
        "recalibration_ready_score": ready_score,
        "learning_compute_feasible_score": compute_score,
        "continuous_self_learning_task": True,
        "maximum_prediction_movement": {
            "bound": MOVEMENT_BOUND,
            "maximum_observed": maximum_observed,
            "mathematical_only_not_retention_certificate": True,
        },
        "state_bytes": state_bytes,
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    failed: Mapping[str, Any], checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Publish an exact external absence without inventing dependent rows."""

    reason = "".join(
        character if character.isalnum() else "_" for character in str(failed["check"])
    ).strip("_")
    gate = _gate(
        str(failed["check"]),
        "validity",
        failed.get("expected"),
        failed.get("observed"),
        str(failed.get("op") or "eq"),
        False,
        "Missing required evidence cannot become substitute measurement data.",
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Jointly constrained cumulative-Brier recalibration prototype",
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {
            "source": V660_RESULT_PATH.as_posix(),
            "counted_as_current": False,
        },
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {"generated_tokens": 0, "model_loads": 0},
        "execution_venue": "host",
        "compute_details": {"device_type": "CPU", "logical_cpu_count": os.cpu_count()},
        "duration_s": float(duration_s),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "cwd": str(REPO_ROOT)},
        "random_seed": {
            "model_seed": None,
            "fitting_seed": None,
            "ordering_seeds": list(ORDER_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": {},
        "input_roles": {"measurement": "blocked_before_fixture_measurement"},
        "raw_sidecars": {},
        "rows": [],
        "numerical_qualification": {},
        "fixture_summary": {},
        "benchmark_receipt": {},
        "frozen_learning_protocol": {"empirical_capture_started": False},
        "sample_size_budget": {
            "analytical_fixture_cases": {
                "planned": 3,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 3,
            },
            "benchmark_order_replays": {
                "planned": 5000,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 5000,
            },
            "future_empirical_online_groups": {
                "planned": 160,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 160,
            },
            "future_retention_groups": {
                "planned": 80,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 80,
            },
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [str(failed["check"])],
            "first_failure": deepcopy(dict(failed)),
        },
        "validation_receipts": [],
        "repository_health": {"unrelated_debt_required_for_this_result": False},
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "terminal_status": "complete",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "positive_claim": False,
        "empirical_benefit_measured": False,
        "confirmatory_benefit_score": 0,
        "recalibration_ready_score": 0,
        "learning_compute_feasible_score": 0,
        "continuous_self_learning_task": True,
        "maximum_prediction_movement": {
            "bound": MOVEMENT_BOUND,
            "maximum_observed": None,
            "mathematical_only_not_retention_certificate": True,
        },
        "state_bytes": 0,
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build compact private evidence for mutation and fresh-reader tests."""

    panel = run_fixture_panel(root / "checkpoints")
    numerical = run_numerical_qualification()
    benchmark = {
        "benchmark_scope": "private_complete_fixture_receipt",
        "bootstrap_replicates_planned": 1000,
        "bootstrap_replicates_completed": 1000,
        "order_replays_completed": 5000,
        "events_completed": 800000,
        "failed": 0,
        "censored": 0,
        "unstarted": 0,
        "mean_prequential_brier": 0.2,
        "measured_duration_s": 0.1,
        "projected_duration_s": 100.0,
        "validation_reserve_s": 300.0,
        "compute_budget_s": 2400.0,
        "fits_2400_seconds_with_reserve": True,
    }
    sidecars = write_raw_sidecars(root, freeze_learning_protocol(), frozen_state_schema())
    checks = [_precondition("private_fixture", "test", "private", True, True, True)]
    return build_artifact(
        panel=panel,
        numerical=numerical,
        benchmark=benchmark,
        raw_sidecars=sidecars,
        validation_receipts=validation_receipts,
        preconditions_checked=checks,
        source_hashes={"private_fixture": "sha256:" + "b" * 64},
        duration_s=0.1,
        phase_spans=[],
    )


def independent_reduce(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> JsonDict:
    """Recompute readiness from raw rows, sidecars, and command receipts."""

    sidecars = value.get("raw_sidecars")
    if not isinstance(sidecars, Mapping) or set(sidecars) != {
        "frozen_learning_protocol",
        "numerical_state_schema",
    }:
        raise ValueError("raw_sidecar_set_mismatch")
    protocol = read_raw_sidecar(root, sidecars["frozen_learning_protocol"])
    state_schema = read_raw_sidecar(root, sidecars["numerical_state_schema"])
    if protocol.get("protocol_hash") != value.get("frozen_learning_protocol", {}).get(
        "protocol_hash"
    ):
        raise ValueError("protocol_hash_mismatch")
    if state_schema.get("schema") != STATE_SCHEMA:
        raise ValueError("numerical_state_schema_mismatch")

    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("fixture_rows_not_list")
    fixture_names = {str(row.get("fixture")) for row in rows if isinstance(row, Mapping)}
    fixture_rows_valid = fixture_names == {"calibration_shift", "no_shift", "recurrence"}
    row_constraint_failures = 0
    for row in rows:
        if not isinstance(row, Mapping):
            row_constraint_failures += 1
            continue
        row_constraint_failures += len(constraint_errors(row.get("final_theta") or []))
        row_constraint_failures += int(
            float(row.get("maximum_movement", 2.0)) > MOVEMENT_BOUND + 1e-7
        )
        row_constraint_failures += int(row.get("normalization_failures") != 0)
        row_constraint_failures += int(row.get("restart_mismatches") != 0)
        row_constraint_failures += int(row.get("duplicate_feedback_rejections") != 1)
        row_constraint_failures += int(row.get("future_label_rejections") != 1)
    shift = next((row for row in rows if row.get("fixture") == "calibration_shift"), {})
    fixture_rows_valid = bool(
        fixture_rows_valid
        and row_constraint_failures == 0
        and shift.get("constrained_brier", math.inf) < shift.get("raw_brier", -math.inf)
        and shift.get("unconstrained_maximum_movement", 0.0) > MOVEMENT_BOUND
    )

    numerical = value.get("numerical_qualification")
    numerical_valid = bool(
        isinstance(numerical, Mapping)
        and numerical.get("passed") is True
        and numerical.get("solver_pair_count", 0) >= 5
        and numerical.get("dense_scalar_case_count", 0) >= 101
        and numerical.get("convergence_failure_count") == 0
        and numerical.get("constraint_failure_count") == 0
        and numerical.get("solver_parity_failure_count") == 0
        and numerical.get("maximum_objective_delta", math.inf) <= SOLVER_OBJECTIVE_TOLERANCE
    )
    benchmark = value.get("benchmark_receipt")
    compute_feasible = bool(
        isinstance(benchmark, Mapping)
        and benchmark.get("bootstrap_replicates_completed") == 1000
        and benchmark.get("order_replays_completed") == 5000
        and benchmark.get("events_completed") == 800000
        and benchmark.get("failed") == 0
        and benchmark.get("censored") == 0
        and benchmark.get("unstarted") == 0
        and benchmark.get("fits_2400_seconds_with_reserve") is True
    )
    receipts = value.get("validation_receipts") or []
    affected_passed = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    preconditions_passed = all(
        row.get("passed") is True
        for row in value.get("preconditions_checked") or []
        if row.get("required") is True
    )
    valid = bool(preconditions_passed and affected_passed and terminal_passed)
    recalibration_ready = bool(valid and numerical_valid and fixture_rows_valid)
    return {
        "valid": valid,
        "preconditions_passed": preconditions_passed,
        "raw_custody_passed": True,
        "affected_validation_passed": affected_passed,
        "terminal_validation_passed": terminal_passed,
        "numerical_qualification_passed": numerical_valid,
        "fixture_lifecycle_passed": fixture_rows_valid,
        "row_constraint_failure_count": row_constraint_failures,
        "recalibration_ready": recalibration_ready,
        "learning_compute_feasible": compute_feasible,
    }


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> JsonDict:
    """Reject changed identity, claims, raw evidence, gates, or checksums."""

    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        raise ValueError("identity_mismatch")
    for field in ("recalibration_ready_score", "learning_compute_feasible_score"):
        if type(value.get(field)) is not int or value.get(field) not in (0, 1):
            raise ValueError(f"score_not_bare_numeric:{field}")
    if value.get("positive_claim") is not False:
        raise ValueError("positive_claim_for_fixture")
    if value.get("verifier_is_oracle") is not True:
        raise ValueError("oracle_declaration_missing")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        raise ValueError("current_invocation_claim_invalid")
    if value.get("inference_substrate_class") != "no_model_load":
        raise ValueError("inference_substrate_class_invalid")
    if value.get("execution_venue") != "host":
        raise ValueError("execution_venue_invalid")
    reduction = independent_reduce(value, root=root, require_terminal=require_terminal)
    expected_ready = int(reduction["recalibration_ready"])
    expected_compute = int(reduction["learning_compute_feasible"])
    if value.get("recalibration_ready_score") != expected_ready:
        raise ValueError("recalibration_ready_score_mismatch")
    if value.get("learning_compute_feasible_score") != expected_compute:
        raise ValueError("learning_compute_feasible_score_mismatch")
    if not reduction["valid"]:
        expected_verdict = ("complete_disqualified_required_validation", "disqualified")
    elif expected_ready and expected_compute:
        expected_verdict = (
            "complete_circular_positive_fixture_qualification",
            "circular_positive",
        )
    else:
        expected_verdict = ("complete_null_recalibration_not_ready", "null")
    if (value.get("honest_verdict"), value.get("verdict_class")) != expected_verdict:
        raise ValueError("verdict_mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value).issubset(principles):
        raise ValueError("field_principles_incomplete")
    if any(not row.get("principle") for row in value.get("acceptance_gate_results") or []):
        raise ValueError("gate_principle_missing")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("checksum_mismatch")
    return reduction


def cold_replay(path: Path, *, root: Path = REPO_ROOT, require_terminal: bool = True) -> JsonDict:
    """Reload and reduce serialized evidence through a fresh-reader API."""

    value = _load_object(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(value, root=root, require_terminal=require_terminal)


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build serial tests, separate coverage, lint, type, and spec checks."""

    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7561",
    )


def terminal_commands(
    candidate: Path, root: Path = REPO_ROOT
) -> list[validation_scope.CommandSpec]:
    """Build cold replay, reduction, adversarial, and strict row readers."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--root",
                str(root),
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--root",
                str(root),
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


def _write_affected_manifest(path: Path) -> None:
    atomic_json(
        path,
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )


def progress(  # pragma: no cover - visible only through the declared entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase and slow-operation boundary with monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7561] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - current time is an entrypoint boundary.
    phase: str, phase_started: float, run_started: float, completed_units: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
    }


def run_experiment(  # pragma: no cover - the entrypoint is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, qualify, benchmark, validate, and publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next(
        (row for row in checks if row.get("required") is True and row.get("passed") is not True),
        None,
    )
    if failed is not None:
        blocked = build_blocked_artifact(failed, checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "complete", completed_units=len(checks))

    progress(started, "protocol", "before_freeze")
    phase_started = time.monotonic()
    protocol = freeze_learning_protocol()
    state_schema = frozen_state_schema()
    sidecars = write_raw_sidecars(root, protocol, state_schema)
    manifest_path = raw_root / "affected_validation_manifest.json"
    _write_affected_manifest(manifest_path)
    spans.append(_span("protocol", phase_started, started, len(sidecars) + 1))
    progress(started, "protocol", "after_freeze", protocol_hash=protocol["protocol_hash"])

    progress(started, "numerical_qualification", "before_solvers")
    phase_started = time.monotonic()
    numerical = run_numerical_qualification()
    spans.append(_span("numerical_qualification", phase_started, started, 106))
    progress(
        started,
        "numerical_qualification",
        "after_solvers",
        passed=numerical["passed"],
    )

    checkpoint_root = Path(tempfile.mkdtemp(prefix="carnot-exp7561-checkpoints-", dir="/tmp"))
    progress(started, "fixture_panel", "before_replays", fixtures=3)
    phase_started = time.monotonic()
    panel = run_fixture_panel(checkpoint_root)
    spans.append(_span("fixture_panel", phase_started, started, len(panel["rows"])))
    progress(
        started,
        "fixture_panel",
        "after_replays",
        completed_units=len(panel["rows"]),
        passed=panel["fixture_gates_passed"],
    )

    progress(
        started,
        "replay_benchmark",
        "before_benchmark",
        replicates=BOOTSTRAP_REPLICATES,
        order_replays=BOOTSTRAP_REPLICATES * len(ORDER_SEEDS),
    )
    phase_started = time.monotonic()
    benchmark = run_replay_benchmark(
        protocol,
        progress_hook=lambda completed: progress(
            started,
            "replay_benchmark",
            "units_complete",
            completed_units=completed,
        ),
    )
    spans.append(
        _span(
            "replay_benchmark",
            phase_started,
            started,
            int(benchmark["order_replays_completed"]),
        )
    )
    progress(
        started,
        "replay_benchmark",
        "after_benchmark",
        completed_units=benchmark["order_replays_completed"],
        feasible=benchmark["fits_2400_seconds_with_reserve"],
    )

    source_hashes = _source_hashes(root)
    source_hashes[manifest_path.relative_to(root).as_posix()] = sha256_file(manifest_path)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7561-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private_root)
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

    candidate = build_artifact(
        panel=panel,
        numerical=numerical,
        benchmark=benchmark,
        raw_sidecars=sidecars,
        validation_receipts=affected,
        preconditions_checked=checks,
        source_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    if not affected_passed:
        progress(started, "publish", "before_atomic_disqualified_affected")
        atomic_json(destination, candidate)
        progress(started, "publish", "complete_disqualified_affected")
        return candidate
    validate_artifact(candidate, root=root, require_terminal=False)
    measured_path = raw_root / "measured_terminal_candidate.json"
    atomic_json(measured_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        terminal_commands(measured_path, root),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        panel=panel,
        numerical=numerical,
        benchmark=benchmark,
        raw_sidecars=sidecars,
        validation_receipts=[*affected, *terminal],
        preconditions_checked=checks,
        source_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    if not terminal_passed:
        progress(started, "publish", "before_atomic_disqualified_terminal")
        atomic_json(destination, final)
        progress(started, "publish", "complete_disqualified_terminal")
        return final
    validate_artifact(final, root=root, require_terminal=True)
    exact_path = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_path, final)

    progress(started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        terminal_commands(exact_path, root),
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(started, "exact_candidate_validation", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "complete",
        recalibration_ready_score=final["recalibration_ready_score"],
        learning_compute_feasible_score=final["learning_compute_feasible_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the producer or one strict serialized-evidence reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        reduction = cold_replay(args.cold_replay, root=root, require_terminal=False)
        print(json.dumps({"event": "cold_replay_passed", **reduction}, sort_keys=True), flush=True)
        return int(not reduction["recalibration_ready"])
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        if not value:
            raise ValueError("artifact_unreadable_or_not_object")
        reduction = validate_artifact(value, root=root, require_terminal=False)
        print(
            json.dumps({"event": "independent_reduction_passed", **reduction}, sort_keys=True),
            flush=True,
        )
        return int(not reduction["recalibration_ready"])
    artifact = run_experiment(root, args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "result": str(args.output or root / RESULT_PATH),
                "recalibration_ready_score": artifact["recalibration_ready_score"],
                "learning_compute_feasible_score": artifact["learning_compute_feasible_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(artifact["verdict_class"] in {"blocked", "disqualified", "partial"})


if __name__ == "__main__":  # pragma: no cover - wrapper is the public executable.
    raise SystemExit(main())
