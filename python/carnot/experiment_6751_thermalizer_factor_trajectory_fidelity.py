"""Build a bounded exact reference for typed stochastic factor compilation.

The reference compiles binary and categorical transition kernels into sparse
log-linear EBM factors. It enumerates every path through depth eight, so no
sampling estimate can hide normalization or accumulated trajectory error. The
official Torx package is only a measured API sidecar. It is not the authority
and it does not imply access to physical thermodynamic hardware.

Spec: REQ-HW-6751, REQ-HW-6751-TYPES, REQ-HW-6751-EXACT,
REQ-HW-6751-MATCHED, REQ-HW-6751-METRICS,
REQ-HW-6751-SERIALIZATION, REQ-HW-6751-PROVENANCE,
REQ-HW-6751-COMPLETION, REQ-HW-6751-BOUNDARY,
SCENARIO-HW-6751-EXACT-COMPILATION,
SCENARIO-HW-6751-REFINEMENT, SCENARIO-HW-6751-FAIL-CLOSED.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from typing import Any

import numpy as np


# Torx imports JAX. Pin its optional sidecar to CPU before that import occurs.
# The internal reference below uses NumPy and does not depend on Torx or JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

JsonDict = dict[str, Any]
Importer = Callable[[str], Any]
DistributionGetter = Callable[[str], Any]
RowBuilder = Callable[[], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6751_thermalizer_factor_trajectory_fidelity.json")
MODULE_PATH = Path("python/carnot/experiment_6751_thermalizer_factor_trajectory_fidelity.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_6751_thermalizer_factor_trajectory_fidelity.py")
TEST_PATH = Path("tests/python/test_experiment_6751_thermalizer_factor_trajectory_fidelity.py")
SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")

EXPERIMENT_ID = "experiment_6751_thermalizer_factor_trajectory_fidelity"
SCHEMA_VERSION = "carnot.experiment_6751.thermalizer_factor_trajectory_fidelity.v1"
RUN_DATE = "20260829"
INFERENCE_SUBSTRATE = "simulator_only_exact_enumeration_no_physical_tsu"
CLAIM_SCOPE = "bounded_cpu_compiler_fidelity_no_physical_tsu_or_performance_claim"
DEPTHS = (1, 2, 4, 8)
ARMS = ("independent_factor", "context_matched", "trajectory_refinement")
PRECISIONS = ("binary32", "fixed_q3_4")
CONTEXT_LABELS = ("left_heavy", "right_heavy")
TRAINING_BUDGET = 96
NORMALIZATION_TOLERANCE = 1.0e-12
REDUCTION_TOLERANCE = 1.0e-12
MAX_ENUMERATED_TRAJECTORIES = 20_000


class CompilerInputError(ValueError):
    """Identify a bounded kernel or state space with ambiguous semantics."""


class CompilerReferenceUnavailable(RuntimeError):
    """Identify an unavailable internal exact path without using a fallback."""


@dataclass(frozen=True)
class PrecisionSpec:
    """Define how fitted factor parameters become serialized numeric values."""

    precision_id: str
    format: str
    quantization_step: float | None
    energy_accumulation: str

    def quantize(self, values: np.ndarray, bound: float) -> np.ndarray:
        """Apply the declared format before any EBM energy is evaluated."""

        clipped = np.clip(np.asarray(values, dtype=np.float64), -bound, bound)
        if self.quantization_step is None:
            return clipped.astype(np.float32).astype(np.float64)
        return np.rint(clipped / self.quantization_step) * self.quantization_step


PRECISION_SPECS: dict[str, PrecisionSpec] = {
    "binary32": PrecisionSpec(
        precision_id="binary32",
        format="IEEE-754 binary32 parameters and logits",
        quantization_step=None,
        energy_accumulation="binary32 logits with binary64 exact normalization",
    ),
    "fixed_q3_4": PrecisionSpec(
        precision_id="fixed_q3_4",
        format="signed Q3.4 fixed point",
        quantization_step=1.0 / 16.0,
        energy_accumulation="Q3.4 parameters with binary64 exact normalization",
    ),
}


@dataclass(frozen=True)
class SeedBundle:
    """Keep factor, fitting, and trajectory tie-breaking seeds explicit."""

    bundle_id: str
    factor_seed: int
    train_seed: int
    trajectory_seed: int

    def receipt(self) -> JsonDict:
        """Return a JSON value that can be copied into each result row."""

        return {
            "seed_bundle_id": self.bundle_id,
            "factor_seed": self.factor_seed,
            "train_seed": self.train_seed,
            "trajectory_seed": self.trajectory_seed,
        }


SEED_BUNDLES = (
    SeedBundle("seed_a", 675101, 675111, 675121),
    SeedBundle("seed_b", 675102, 675112, 675122),
)


@dataclass(frozen=True)
class TypedKernel:
    """Store one finite target conditional and its sparse EBM feature map."""

    factor_id: str
    kind: str
    categories: tuple[str, ...]
    target: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]
    features: tuple[tuple[tuple[float, ...], ...], ...]
    parameter_bound: float
    couplers: tuple[tuple[str, str, str], ...]

    @property
    def n_categories(self) -> int:
        """Return the finite input and output cardinality."""

        return len(self.categories)

    @property
    def n_parameters(self) -> int:
        """Return the fixed factor capacity used by every compiler arm."""

        return len(self.feature_names)

    def validate(self) -> None:
        """Reject malformed tables before they can enter exact enumeration."""

        if self.kind not in {"binary", "categorical"}:
            raise CompilerInputError("kernel kind must be binary or categorical")
        if self.n_categories < 2 or len(set(self.categories)) != self.n_categories:
            raise CompilerInputError("categories must contain unique finite labels")
        target = np.asarray(self.target, dtype=np.float64)
        expected_shape = (self.n_categories, self.n_categories)
        if target.shape != expected_shape:
            raise CompilerInputError(f"target shape must be {expected_shape}")
        if not np.all(np.isfinite(target)) or np.any(target < 0.0):
            raise CompilerInputError("target probabilities must be finite and nonnegative")
        if np.max(np.abs(target.sum(axis=1) - 1.0)) > NORMALIZATION_TOLERANCE:
            raise CompilerInputError("target row normalization failed")
        features = np.asarray(self.features, dtype=np.float64)
        feature_shape = (*expected_shape, self.n_parameters)
        if features.shape != feature_shape or not np.all(np.isfinite(features)):
            raise CompilerInputError(f"feature shape must be {feature_shape} and finite")
        if self.parameter_bound <= 0.0 or not math.isfinite(self.parameter_bound):
            raise CompilerInputError("parameter bound must be finite and positive")
        names = set(self.feature_names)
        for _left, _right, parameter in self.couplers:
            if parameter not in names:
                raise CompilerInputError(f"coupler parameter is not declared: {parameter}")

    def compiled_conditional(self, theta: np.ndarray, precision: PrecisionSpec) -> np.ndarray:
        """Normalize the sparse EBM energy for every clamped input category."""

        self.validate()
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (self.n_parameters,) or not np.all(np.isfinite(values)):
            raise CompilerInputError("parameter vector has the wrong shape or a nonfinite value")
        quantized = precision.quantize(values, self.parameter_bound)
        features = np.asarray(self.features, dtype=np.float64)
        if precision.precision_id == "binary32":
            logits = np.einsum(
                "ijk,k->ij", features.astype(np.float32), quantized.astype(np.float32)
            ).astype(np.float64)
        else:
            logits = np.einsum("ijk,k->ij", features, quantized)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Context:
    """Declare the exact initial law that gives one factor its program context."""

    context_id: str
    label: str
    initial: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryDistribution:
    """Retain every path and probability for one bounded trajectory law."""

    paths: np.ndarray
    probabilities: np.ndarray


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable keys and reject nonfinite JSON values."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CompilerInputError("evidence is not finite canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    """Prefix a SHA-256 digest so evidence is not mistaken for source text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON representation of a bounded evidence value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Keep a missing source distinct from an empty source file."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def spec_anchors(text: str) -> list[str]:
    """Extract stable requirement and scenario identifiers from a specification."""

    return sorted(set(re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)))


def _rounded(value: float) -> float:
    """Keep row JSON stable while retaining more precision than the gates use."""

    result = float(f"{float(value):.15g}")
    return 0.0 if result == 0.0 else result


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row after removing its self-referential digest."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def receipt_hash(receipt: Mapping[str, Any]) -> str:
    """Hash a topology or precision receipt without its digest field."""

    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def frozen_kernels() -> tuple[TypedKernel, ...]:
    """Return one binary and one categorical kernel with limited EBM capacity."""

    binary_features = []
    for input_value, output_value in product(range(2), repeat=2):
        input_spin = -1.0 if input_value == 0 else 1.0
        output_spin = -1.0 if output_value == 0 else 1.0
        binary_features.append((output_spin, input_spin * output_spin))
    binary = TypedKernel(
        factor_id="binary_sticky_transition",
        kind="binary",
        categories=("zero", "one"),
        target=((0.98, 0.02), (0.38, 0.62)),
        feature_names=("output_bias", "input_output_coupler"),
        features=tuple(
            tuple(binary_features[input_value * 2 + output_value] for output_value in range(2))
            for input_value in range(2)
        ),
        parameter_bound=0.75,
        couplers=(
            ("bias", "output:any", "output_bias"),
            ("input:any", "output:any", "input_output_coupler"),
        ),
    )

    categorical_features = []
    for input_value, output_value in product(range(3), repeat=2):
        categorical_features.append(
            (
                1.0 if output_value == 0 else 0.0,
                1.0 if output_value == 1 else 0.0,
                1.0 if output_value == input_value else 0.0,
                1.0 if output_value == (input_value + 1) % 3 else 0.0,
            )
        )
    categorical = TypedKernel(
        factor_id="categorical_ring_transition",
        kind="categorical",
        categories=("red", "green", "blue"),
        target=((0.90, 0.08, 0.02), (0.15, 0.25, 0.60), (0.55, 0.10, 0.35)),
        feature_names=(
            "output_red_bias",
            "output_green_bias",
            "same_category_coupler",
            "forward_ring_coupler",
        ),
        features=tuple(
            tuple(categorical_features[input_value * 3 + output_value] for output_value in range(3))
            for input_value in range(3)
        ),
        parameter_bound=1.0,
        couplers=(
            ("bias", "output:red", "output_red_bias"),
            ("bias", "output:green", "output_green_bias"),
            ("input:category", "output:same", "same_category_coupler"),
            ("input:category", "output:next", "forward_ring_coupler"),
        ),
    )
    for kernel in (binary, categorical):
        kernel.validate()
    return binary, categorical


def frozen_contexts(kernel: TypedKernel) -> tuple[Context, ...]:
    """Return two exact initial laws for the supplied typed kernel."""

    if kernel.factor_id == "binary_sticky_transition":
        initials = ((0.92, 0.08), (0.08, 0.92))
    elif kernel.factor_id == "categorical_ring_transition":
        initials = ((0.86, 0.10, 0.04), (0.04, 0.10, 0.86))
    else:
        raise CompilerInputError(f"no frozen contexts for factor: {kernel.factor_id}")
    return tuple(
        Context(f"{kernel.factor_id}:{label}", label, initial)
        for label, initial in zip(CONTEXT_LABELS, initials, strict=True)
    )


def conditional_kl(
    target: np.ndarray, compiled: np.ndarray, input_weights: np.ndarray
) -> tuple[list[float], float]:
    """Compute exact conditional KL for each input and its context-weighted mean."""

    target_array = np.asarray(target, dtype=np.float64)
    compiled_array = np.asarray(compiled, dtype=np.float64)
    weights = np.asarray(input_weights, dtype=np.float64)
    if target_array.shape != compiled_array.shape or target_array.ndim != 2:
        raise CompilerInputError("conditional arrays must have the same matrix shape")
    if weights.shape != (target_array.shape[0],) or np.any(weights < 0.0):
        raise CompilerInputError("input weights must match the conditional input categories")
    if abs(float(weights.sum()) - 1.0) > NORMALIZATION_TOLERANCE:
        raise CompilerInputError("input weights must normalize")
    if np.any(compiled_array <= 0.0):
        raise CompilerInputError("compiled conditional must have full support")
    mask = target_array > 0.0
    terms = np.zeros_like(target_array)
    terms[mask] = target_array[mask] * np.log(target_array[mask] / compiled_array[mask])
    per_input = terms.sum(axis=1)
    return ([_rounded(value) for value in per_input], _rounded(float(weights @ per_input)))


def _validate_distribution(value: np.ndarray, label: str) -> None:
    """Reject a nonfinite, negative, or non-normalized exact distribution."""

    if value.ndim != 1 or not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise CompilerInputError(f"{label} distribution must be finite and nonnegative")
    if abs(float(value.sum()) - 1.0) > NORMALIZATION_TOLERANCE:
        raise CompilerInputError(f"{label} distribution must normalize")


def enumerate_trajectory_distribution(
    initial: np.ndarray, conditional: np.ndarray, depth: int
) -> TrajectoryDistribution:
    """Enumerate the complete path law instead of estimating it from samples."""

    if depth <= 0:
        raise CompilerInputError("trajectory enumeration requires positive depth")
    matrix = np.asarray(conditional, dtype=np.float64)
    start = np.asarray(initial, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise CompilerInputError("conditional must be a square matrix")
    if start.shape != (matrix.shape[0],):
        raise CompilerInputError("initial distribution must match the conditional categories")
    _validate_distribution(start, "initial")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise CompilerInputError("conditional probabilities must be finite and nonnegative")
    if np.max(np.abs(matrix.sum(axis=1) - 1.0)) > NORMALIZATION_TOLERANCE:
        raise CompilerInputError("conditional rows must normalize")
    path_count = matrix.shape[0] ** (depth + 1)
    if path_count > MAX_ENUMERATED_TRAJECTORIES:
        raise CompilerReferenceUnavailable(
            f"trajectory state space {path_count} exceeds {MAX_ENUMERATED_TRAJECTORIES}"
        )
    paths = np.asarray(list(product(range(matrix.shape[0]), repeat=depth + 1)), dtype=np.int16)
    probabilities = start[paths[:, 0]].astype(np.float64)
    for step in range(depth):
        probabilities *= matrix[paths[:, step], paths[:, step + 1]]
    return TrajectoryDistribution(paths=paths, probabilities=probabilities)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute total variation on two aligned exact finite laws."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise CompilerInputError("total-variation distributions must have equal shape")
    return _rounded(0.5 * float(np.abs(left_array - right_array).sum()))


def context_input_weights(kernel: TypedKernel, context: Context, depth: int) -> np.ndarray:
    """Average exact target inputs over the transitions used at one depth."""

    target = np.asarray(kernel.target, dtype=np.float64)
    marginal = np.asarray(context.initial, dtype=np.float64)
    accumulated = np.zeros(kernel.n_categories, dtype=np.float64)
    for _step in range(depth):
        accumulated += marginal
        marginal = marginal @ target
    return accumulated / float(depth)


def training_context_weights(kernel: TypedKernel, context: Context) -> np.ndarray:
    """Freeze one shared context objective across every reported circuit depth."""

    weights = np.mean(
        np.stack([context_input_weights(kernel, context, depth) for depth in DEPTHS]), axis=0
    )
    return weights / weights.sum()


def candidate_bank(
    kernel: TypedKernel,
    precision: PrecisionSpec,
    seed_bundle: SeedBundle,
    budget: int,
) -> np.ndarray:
    """Create one deterministic parameter bank shared by all three fitting arms."""

    if budget < 3:
        raise CompilerInputError("candidate budget must include zero and both bounds")
    factor_token = int.from_bytes(
        hashlib.sha256(kernel.factor_id.encode("utf-8")).digest()[:4], "little"
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                factor_token,
                seed_bundle.factor_seed,
                seed_bundle.train_seed,
                seed_bundle.trajectory_seed,
            ]
        )
    )
    bank = np.empty((budget, kernel.n_parameters), dtype=np.float64)
    bank[0] = 0.0
    bank[1] = -kernel.parameter_bound
    bank[2] = kernel.parameter_bound
    bank[3:] = rng.uniform(
        -kernel.parameter_bound,
        kernel.parameter_bound,
        size=(budget - 3, kernel.n_parameters),
    )
    return precision.quantize(bank, kernel.parameter_bound)


def _distribution_hash(distribution: TrajectoryDistribution) -> str:
    """Hash exact paths and little-endian binary64 probabilities."""

    paths = np.asarray(distribution.paths, dtype="<i2").tobytes(order="C")
    probabilities = np.asarray(distribution.probabilities, dtype="<f8").tobytes(order="C")
    return sha256_bytes(paths + probabilities)


def _candidate_bank_receipt(kernel: TypedKernel, bank: np.ndarray) -> JsonDict:
    """Bind the fixed capacity and candidate set used by each matched arm."""

    return {
        "factor_capacity": kernel.n_parameters,
        "candidate_count": int(bank.shape[0]),
        "bank_sha256": sha256_bytes(np.asarray(bank, dtype="<f8").tobytes(order="C")),
    }


def _fit_arm(
    kernel: TypedKernel,
    context: Context,
    arm: str,
    precision: PrecisionSpec,
    seed_bundle: SeedBundle,
) -> tuple[np.ndarray, int, float, JsonDict]:
    """Select from one bank using only the objective declared for the arm."""

    if arm not in ARMS:
        raise CompilerInputError(f"unknown compiler arm: {arm}")
    bank = candidate_bank(kernel, precision, seed_bundle, TRAINING_BUDGET)
    target = np.asarray(kernel.target, dtype=np.float64)
    if arm == "independent_factor":
        input_weights = np.full(kernel.n_categories, 1.0 / kernel.n_categories)
    else:
        input_weights = training_context_weights(kernel, context)
    target_trajectories = {
        depth: enumerate_trajectory_distribution(np.asarray(context.initial), target, depth)
        for depth in DEPTHS
    }
    scores = []
    for theta in bank:
        compiled = kernel.compiled_conditional(theta, precision)
        if arm == "trajectory_refinement":
            score = np.mean(
                [
                    total_variation(
                        target_trajectories[depth].probabilities,
                        enumerate_trajectory_distribution(
                            np.asarray(context.initial), compiled, depth
                        ).probabilities,
                    )
                    for depth in DEPTHS
                ]
            )
        else:
            _per_input, score = conditional_kl(target, compiled, input_weights)
        scores.append(float(score))
    selected_index = int(np.argmin(np.asarray(scores, dtype=np.float64)))
    return (
        bank[selected_index],
        selected_index,
        _rounded(scores[selected_index]),
        _candidate_bank_receipt(kernel, bank),
    )


def topology_receipts() -> list[JsonDict]:
    """Serialize category, bias, coupler, bound, and capacity declarations."""

    rows = []
    for kernel in frozen_kernels():
        row: JsonDict = {
            "topology_id": f"sparse_ebm:{kernel.factor_id}",
            "factor_id": kernel.factor_id,
            "factor_kind": kernel.kind,
            "categories": list(kernel.categories),
            "biases": [name for name in kernel.feature_names if "bias" in name],
            "couplers": [
                {"left": left, "right": right, "parameter": parameter}
                for left, right, parameter in kernel.couplers
            ],
            "parameter_names": list(kernel.feature_names),
            "parameter_bound": kernel.parameter_bound,
            "factor_capacity": kernel.n_parameters,
            "sparse": True,
            "target_conditional_sha256": sha256_json(kernel.target),
        }
        row["receipt_sha256"] = receipt_hash(row)
        rows.append(row)
    return rows


def precision_receipts() -> list[JsonDict]:
    """Serialize each numeric format and the rule used to quantize parameters."""

    rows = []
    for precision_id in PRECISIONS:
        spec = PRECISION_SPECS[precision_id]
        row: JsonDict = {
            "precision_id": precision_id,
            "format": spec.format,
            "quantization_step": spec.quantization_step,
            "parameter_rounding": (
                "round_to_nearest_ties_to_even"
                if spec.quantization_step is not None
                else "IEEE-754 binary32 cast"
            ),
            "energy_accumulation": spec.energy_accumulation,
            "probability_normalization": "binary64 exhaustive log-sum-exp",
        }
        row["receipt_sha256"] = receipt_hash(row)
        rows.append(row)
    return rows


def build_rows() -> list[JsonDict]:
    """Fit every matched arm and retain the complete exact Cartesian row set."""

    rows: list[JsonDict] = []
    topology_by_factor = {row["factor_id"]: row for row in topology_receipts()}
    for kernel in frozen_kernels():
        target = np.asarray(kernel.target, dtype=np.float64)
        for context in frozen_contexts(kernel):
            initial = np.asarray(context.initial, dtype=np.float64)
            for precision_id in PRECISIONS:
                precision = PRECISION_SPECS[precision_id]
                for seed_bundle in SEED_BUNDLES:
                    for arm in ARMS:
                        theta, selected_index, objective, bank_receipt = _fit_arm(
                            kernel, context, arm, precision, seed_bundle
                        )
                        compiled = kernel.compiled_conditional(theta, precision)
                        conditional_norm = max(
                            float(np.max(np.abs(target.sum(axis=1) - 1.0))),
                            float(np.max(np.abs(compiled.sum(axis=1) - 1.0))),
                        )
                        for depth in DEPTHS:
                            weights = context_input_weights(kernel, context, depth)
                            per_input_kl, weighted_kl = conditional_kl(target, compiled, weights)
                            target_trajectory = enumerate_trajectory_distribution(
                                initial, target, depth
                            )
                            compiled_trajectory = enumerate_trajectory_distribution(
                                initial, compiled, depth
                            )
                            target_norm = abs(float(target_trajectory.probabilities.sum()) - 1.0)
                            compiled_norm = abs(
                                float(compiled_trajectory.probabilities.sum()) - 1.0
                            )
                            trajectory_tv = total_variation(
                                target_trajectory.probabilities,
                                compiled_trajectory.probabilities,
                            )
                            normalization = {
                                "conditional": _rounded(conditional_norm),
                                "target_trajectory": _rounded(target_norm),
                                "compiled_trajectory": _rounded(compiled_norm),
                            }
                            row_id = ":".join(
                                (
                                    kernel.factor_id,
                                    context.label,
                                    arm,
                                    str(depth),
                                    precision_id,
                                    seed_bundle.bundle_id,
                                )
                            )
                            row: JsonDict = {
                                "row_id": row_id,
                                "factor_id": kernel.factor_id,
                                "factor_kind": kernel.kind,
                                "categories": list(kernel.categories),
                                "context_id": context.context_id,
                                "context_initial": list(context.initial),
                                "arm": arm,
                                "depth": depth,
                                "precision": precision_id,
                                **seed_bundle.receipt(),
                                "topology_id": topology_by_factor[kernel.factor_id]["topology_id"],
                                "topology_receipt_sha256": topology_by_factor[kernel.factor_id][
                                    "receipt_sha256"
                                ],
                                "factor_capacity": kernel.n_parameters,
                                "candidate_evaluations": TRAINING_BUDGET,
                                "candidate_bank_receipt": dict(bank_receipt),
                                "selected_candidate_index": selected_index,
                                "selection_objective": arm,
                                "selection_objective_value": objective,
                                "compiled_parameters": [_rounded(value) for value in theta],
                                "conditional_kl_by_input": {
                                    category: value
                                    for category, value in zip(
                                        kernel.categories, per_input_kl, strict=True
                                    )
                                },
                                "conditional_kl": weighted_kl,
                                "trajectory_tv": trajectory_tv,
                                "trajectory_fidelity_score": _rounded(1.0 - trajectory_tv),
                                "exact_factor_state_count": kernel.n_categories**2,
                                "exact_trajectory_state_count": int(
                                    target_trajectory.probabilities.size
                                ),
                                "target_trajectory_sha256": _distribution_hash(target_trajectory),
                                "compiled_trajectory_sha256": _distribution_hash(
                                    compiled_trajectory
                                ),
                                "normalization_error": normalization,
                                "maximum_normalization_error": _rounded(
                                    max(normalization.values())
                                ),
                            }
                            row["row_sha256"] = row_hash(row)
                            rows.append(row)
    return rows


def _mean(values: Sequence[float]) -> float:
    """Return a stable binary64 mean for one non-empty row reduction."""

    return _rounded(float(math.fsum(values) / len(values)))


def derive_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all public metrics from retained rows only."""

    conditional_groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    trajectory_groups: dict[tuple[int, str], list[float]] = {}
    normalization = {}
    arm_values: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for row in rows:
        conditional_key = (
            str(row["factor_id"]),
            str(row["context_id"]),
            str(row["arm"]),
            str(row["precision"]),
            str(row["seed_bundle_id"]),
        )
        conditional_groups.setdefault(conditional_key, []).append(float(row["conditional_kl"]))
        trajectory_key = (int(row["depth"]), str(row["arm"]))
        trajectory_groups.setdefault(trajectory_key, []).append(float(row["trajectory_tv"]))
        normalization[str(row["row_id"])] = float(row["maximum_normalization_error"])
        arm_values[str(row["arm"])].append(float(row["trajectory_tv"]))

    conditional_rows = [
        {
            "factor_id": key[0],
            "context_id": key[1],
            "arm": key[2],
            "precision": key[3],
            "seed_bundle_id": key[4],
            "mean_conditional_kl": _mean(values),
            "row_count": len(values),
        }
        for key, values in sorted(conditional_groups.items())
    ]
    trajectory_rows = [
        {
            "depth": key[0],
            "arm": key[1],
            "mean_trajectory_tv": _mean(values),
            "minimum_trajectory_tv": _rounded(min(values)),
            "maximum_trajectory_tv": _rounded(max(values)),
            "row_count": len(values),
        }
        for key, values in sorted(trajectory_groups.items())
    ]
    arm_means = {arm: (_mean(values) if values else None) for arm, values in arm_values.items()}
    independent = arm_means["independent_factor"]
    context = arm_means["context_matched"]
    trajectory = arm_means["trajectory_refinement"]
    context_reduced = bool(
        independent is not None
        and context is not None
        and context < independent - REDUCTION_TOLERANCE
    )
    trajectory_reduced = bool(
        independent is not None
        and trajectory is not None
        and trajectory < independent - REDUCTION_TOLERANCE
    )
    refined_values = [value for value in (context, trajectory) if value is not None]
    positive_gate = {
        "independent_mean_trajectory_tv": independent,
        "context_matched_mean_trajectory_tv": context,
        "trajectory_refinement_mean_trajectory_tv": trajectory,
        "best_refined_mean_trajectory_tv": min(refined_values) if refined_values else None,
        "context_reduced": context_reduced,
        "trajectory_reduced": trajectory_reduced,
        "reduction_tolerance": REDUCTION_TOLERANCE,
        "passed": context_reduced or trajectory_reduced,
    }
    return {
        "conditional_kl_by_factor": conditional_rows,
        "trajectory_tv_by_depth": trajectory_rows,
        "normalization_error_by_row": dict(sorted(normalization.items())),
        "positive_result_gate": positive_gate,
    }


def unavailable_sidecar(reason: str) -> JsonDict:
    """Represent skipped or unavailable Torx evidence without inventing a claim."""

    return {
        "available": False,
        "passed": False,
        "distribution": "extro-torx",
        "version": None,
        "module_path": None,
        "api_identity": [],
        "conformance_rows": [],
        "failure": reason,
        "hardware_used": False,
    }


def inspect_torx_sidecar(
    *,
    importer: Importer = importlib.import_module,
    distribution_getter: DistributionGetter = metadata.distribution,
) -> JsonDict:
    """Exercise two installed Torx typed gates and record the observed API."""

    try:
        distribution = distribution_getter("extro-torx")
        torx = importer("torx")
        version = str(distribution.version)
        pnot_class = torx.psc.PNOT
        pdit_class = torx.psc.PditShift
        theta = np.asarray([0.3], dtype=np.float64)
        probability = 1.0 / (1.0 + math.exp(-0.3))
        binary_observed = np.asarray(pnot_class(sites=0).get_matrix(theta), dtype=np.float64)
        binary_expected = np.asarray(
            [[1.0 - probability, probability], [probability, 1.0 - probability]],
            dtype=np.float64,
        )
        categorical_observed = np.asarray(
            pdit_class(sites=0, dims=(3,)).get_matrix(theta), dtype=np.float64
        )
        categorical_expected = (1.0 - probability) * np.eye(3) + probability * np.roll(
            np.eye(3), shift=1, axis=0
        )
        conformance_rows = []
        for api, observed, expected in (
            ("torx.psc.PNOT.get_matrix", binary_observed, binary_expected),
            ("torx.psc.PditShift.get_matrix", categorical_observed, categorical_expected),
        ):
            maximum_delta = float(np.max(np.abs(observed - expected)))
            normalization_error = float(np.max(np.abs(observed.sum(axis=0) - 1.0)))
            conformance_rows.append(
                {
                    "api": api,
                    "shape": list(observed.shape),
                    "maximum_absolute_delta": _rounded(maximum_delta),
                    "normalization_error": _rounded(normalization_error),
                    "passed": maximum_delta <= 1.0e-6 and normalization_error <= 1.0e-6,
                }
            )
        return {
            "available": True,
            "passed": all(row["passed"] for row in conformance_rows),
            "distribution": "extro-torx",
            "version": version,
            "module_path": str(Path(torx.__file__).resolve()),
            "module_sha256": sha256_file(Path(torx.__file__)),
            "api_identity": [
                f"{pnot_class.__module__}.{pnot_class.__qualname__}.get_matrix",
                f"{pdit_class.__module__}.{pdit_class.__qualname__}.get_matrix",
            ],
            "conformance_rows": conformance_rows,
            "failure": None,
            "hardware_used": False,
        }
    except Exception as exc:
        return unavailable_sidecar(f"{exc.__class__.__name__}: {exc}")


TOP_LEVEL_PRINCIPLES: dict[str, str] = {
    "experiment_id": "A stable identifier binds the result to Exp6751.",
    "schema_version": "A versioned schema makes later validation changes explicit.",
    "run_date": "The planning date fixes the experiment boundary.",
    "spec_refs": "Requirement identifiers connect the artifact to its contract.",
    "status": "A terminal state distinguishes completed, null, and owned blocked runs.",
    "prior_failure": "The Exp6684 block and changed internal-reference mechanism remain visible.",
    "field_principles": "Every top-level field states why it belongs in the artifact.",
    "inference_substrate": "The value declares CPU exact simulation and excludes physical TSU use.",
    "duration_s": "A monotonic clock reports real construction time without padding.",
    "random_seed": "Factor, fitting, and trajectory seeds make every candidate bank replayable.",
    "reproducibility_checksum": "The digest binds kernels, configuration, receipts, and retained rows.",
    "hardware_used": "Bare false prevents simulator evidence from becoming a hardware claim.",
    "simulator_used": "Bare true names the exact-enumeration execution path.",
    "compiler_provenance": "Internal authority and optional official conformance remain separate.",
    "frozen_config": "The complete bounded Cartesian design is fixed before metrics are reduced.",
    "rows": "Each factor, context, arm, depth, precision, and seed has one auditable row.",
    "conditional_kl_by_factor": "Only retained rows supply the conditional KL reductions.",
    "trajectory_tv_by_depth": "Only retained rows supply depth-wise accumulated error.",
    "normalization_error_by_row": "Every exact distribution has an explicit unit-mass check.",
    "topology_receipts": "Categories, biases, couplers, bounds, and capacity define sparse factors.",
    "precision_receipts": "Numeric formats and rounding rules expose precision limits.",
    "compiler_fidelity_completed": "True means the full row product and exact gates are complete.",
    "gate_check_summary": "Failed checks name expected and observed evidence for blocked states.",
    "verdict_class": "The closed class separates positive, circular, null, partial, and blocked results.",
    "honest_verdict": "The terminal prefix states the measured compiler result and its boundary.",
    "verifier_is_oracle": "Exact enumeration is the authority for these bounded finite cases.",
    "claim_scope": "The scope forbids physical-device and performance interpretations.",
    "positive_result_gate": "A strict row-derived TV reduction is required for positive credit.",
}


def _random_seed_receipt() -> JsonDict:
    """Collect every explicit seed role without collapsing the three meanings."""

    return {
        "factor_seeds": [bundle.factor_seed for bundle in SEED_BUNDLES],
        "train_seeds": [bundle.train_seed for bundle in SEED_BUNDLES],
        "trajectory_seeds": [bundle.trajectory_seed for bundle in SEED_BUNDLES],
        "bundles": [bundle.receipt() for bundle in SEED_BUNDLES],
    }


def _frozen_config() -> JsonDict:
    """Serialize the complete exact design before any aggregate is interpreted."""

    kernel_count = len(frozen_kernels())
    return {
        "factor_ids": [kernel.factor_id for kernel in frozen_kernels()],
        "context_labels": list(CONTEXT_LABELS),
        "arms": list(ARMS),
        "depths": list(DEPTHS),
        "precisions": list(PRECISIONS),
        "seed_bundle_ids": [bundle.bundle_id for bundle in SEED_BUNDLES],
        "training_budget_per_arm": TRAINING_BUDGET,
        "normalization_tolerance": NORMALIZATION_TOLERANCE,
        "reduction_tolerance": REDUCTION_TOLERANCE,
        "maximum_enumerated_trajectory_states": MAX_ENUMERATED_TRAJECTORIES,
        "expected_row_count": (
            kernel_count
            * len(CONTEXT_LABELS)
            * len(ARMS)
            * len(DEPTHS)
            * len(PRECISIONS)
            * len(SEED_BUNDLES)
        ),
        "selection_objectives": {
            "independent_factor": "uniform-input conditional KL",
            "context_matched": "exact target-visitation conditional KL",
            "trajectory_refinement": "mean exact full-trajectory total variation",
        },
    }


def _internal_provenance() -> JsonDict:
    """Name the exact implementation that remains authoritative."""

    return {
        "identity": "carnot.exp6751.numpy_sparse_ebm_exact_compiler.v1",
        "authority": True,
        "module_path": MODULE_PATH.as_posix(),
        "module_sha256": sha256_file(REPO_ROOT / MODULE_PATH),
        "python": platform.python_version(),
        "numeric_library": f"numpy=={np.__version__}",
        "method": "finite sparse-log-linear candidate fitting plus exhaustive trajectory laws",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind only the frozen compiler inputs, serialization receipts, and rows."""

    return sha256_json(
        {
            "random_seed": artifact["random_seed"],
            "frozen_config": artifact["frozen_config"],
            "topology_receipts": artifact["topology_receipts"],
            "precision_receipts": artifact["precision_receipts"],
            "rows": artifact["rows"],
        }
    )


def _row_grid(rows: Sequence[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    """Reduce rows to the six axes that define the required Cartesian product."""

    return {
        (
            row["factor_id"],
            row["context_id"],
            row["arm"],
            row["depth"],
            row["precision"],
            row["seed_bundle_id"],
        )
        for row in rows
    }


def _expected_grid() -> set[tuple[Any, ...]]:
    """Construct the preregistered Cartesian product independently of result rows."""

    return {
        (
            kernel.factor_id,
            context.context_id,
            arm,
            depth,
            precision,
            seed.bundle_id,
        )
        for kernel in frozen_kernels()
        for context in frozen_contexts(kernel)
        for arm in ARMS
        for depth in DEPTHS
        for precision in PRECISIONS
        for seed in SEED_BUNDLES
    }


def _scientific_gate_errors(
    rows: Sequence[Mapping[str, Any]],
    topology: Sequence[Mapping[str, Any]],
    precision: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return exact failed checks before a completed verdict is selected."""

    errors = []
    expected_grid = _expected_grid()
    observed_grid = _row_grid(rows)
    if len(rows) != len(expected_grid) or observed_grid != expected_grid:
        errors.append(
            {
                "check": "complete_row_product",
                "expected": {"row_count": len(expected_grid)},
                "observed": {"row_count": len(rows), "unique_grid_count": len(observed_grid)},
            }
        )
    invalid_hashes = [row["row_id"] for row in rows if row.get("row_sha256") != row_hash(row)]
    if invalid_hashes:
        errors.append(
            {
                "check": "row_hashes",
                "expected": {"invalid_count": 0},
                "observed": {"invalid_count": len(invalid_hashes), "row_ids": invalid_hashes},
            }
        )
    invalid_normalization = [
        row["row_id"]
        for row in rows
        if float(row["maximum_normalization_error"]) > NORMALIZATION_TOLERANCE
    ]
    if invalid_normalization:
        errors.append(
            {
                "check": "exact_normalization",
                "expected": {"maximum_error": NORMALIZATION_TOLERANCE},
                "observed": {"failed_row_ids": invalid_normalization},
            }
        )
    invalid_topology = [
        row.get("factor_id") for row in topology if row.get("receipt_sha256") != receipt_hash(row)
    ]
    if invalid_topology:
        errors.append(
            {
                "check": "topology_receipts",
                "expected": {"invalid_count": 0},
                "observed": {"factor_ids": invalid_topology},
            }
        )
    invalid_precision = [
        row.get("precision_id")
        for row in precision
        if row.get("receipt_sha256") != receipt_hash(row)
    ]
    if invalid_precision:
        errors.append(
            {
                "check": "precision_receipts",
                "expected": {"invalid_count": 0},
                "observed": {"precision_ids": invalid_precision},
            }
        )
    return errors


def _base_artifact(duration_s: float, torx_sidecar: Mapping[str, Any]) -> JsonDict:
    """Create fields shared by completed and owned-blocked terminal artifacts."""

    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "run_date": RUN_DATE,
        "spec_refs": [
            "REQ-HW-6751",
            "REQ-HW-6751-TYPES",
            "REQ-HW-6751-EXACT",
            "REQ-HW-6751-MATCHED",
            "REQ-HW-6751-METRICS",
            "REQ-HW-6751-SERIALIZATION",
            "REQ-HW-6751-PROVENANCE",
            "REQ-HW-6751-COMPLETION",
            "REQ-HW-6751-BOUNDARY",
            "SCENARIO-HW-6751-EXACT-COMPILATION",
            "SCENARIO-HW-6751-REFINEMENT",
            "SCENARIO-HW-6751-FAIL-CLOSED",
        ],
        "status": "in_progress",
        "prior_failure": {
            "experiment_id": "experiment_6684_torx_typed_factor_parity",
            "status": "blocked_parity_check_failed",
            "root_cause": "installed Torx CPU parity did not pass every end-to-end check",
            "changed_mechanism": (
                "The internal exhaustive compiler is now authoritative. "
                "Installed Torx is an optional measured sidecar."
            ),
        },
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _rounded(duration_s),
        "random_seed": _random_seed_receipt(),
        "reproducibility_checksum": "pending",
        "hardware_used": False,
        "simulator_used": True,
        "compiler_provenance": {
            "internal": _internal_provenance(),
            "official_sidecar": dict(torx_sidecar),
        },
        "frozen_config": _frozen_config(),
        "rows": [],
        "conditional_kl_by_factor": [],
        "trajectory_tv_by_depth": [],
        "normalization_error_by_row": {},
        "topology_receipts": topology_receipts(),
        "precision_receipts": precision_receipts(),
        "compiler_fidelity_completed": False,
        "gate_check_summary": [],
        "verdict_class": "partial",
        "honest_verdict": "complete_partial: artifact construction did not finish",
        "verifier_is_oracle": True,
        "claim_scope": CLAIM_SCOPE,
        "positive_result_gate": {
            "independent_mean_trajectory_tv": None,
            "context_matched_mean_trajectory_tv": None,
            "trajectory_refinement_mean_trajectory_tv": None,
            "best_refined_mean_trajectory_tv": None,
            "context_reduced": False,
            "trajectory_reduced": False,
            "reduction_tolerance": REDUCTION_TOLERANCE,
            "passed": False,
        },
    }


def _finish_common(artifact: JsonDict) -> JsonDict:
    """Add the complete principle map and reproducibility digest last."""

    artifact["field_principles"] = dict(TOP_LEVEL_PRINCIPLES)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    duration_s: float,
    torx_sidecar: Mapping[str, Any],
) -> JsonDict:
    """Reduce exact rows into one terminal compiler-fidelity artifact."""

    artifact = _base_artifact(duration_s, torx_sidecar)
    artifact["rows"] = [dict(row) for row in rows]
    aggregates = derive_aggregates(artifact["rows"])
    artifact.update(aggregates)
    gate_errors = _scientific_gate_errors(
        artifact["rows"], artifact["topology_receipts"], artifact["precision_receipts"]
    )
    artifact["gate_check_summary"] = gate_errors
    completed = not gate_errors
    artifact["compiler_fidelity_completed"] = completed
    if not completed:
        artifact["status"] = "complete_partial"
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "complete_partial: exact compiler gates did not all pass"
    elif artifact["positive_result_gate"]["passed"]:
        artifact["status"] = "complete_circular_positive"
        artifact["verdict_class"] = "circular_positive"
        if artifact["positive_result_gate"]["context_reduced"]:
            artifact["honest_verdict"] = (
                "complete: context matching reduced exact mean trajectory total variation; "
                "oracle-backed circular simulator-only evidence"
            )
        else:
            artifact["honest_verdict"] = (
                "complete: exact-objective trajectory refinement reduced exact mean trajectory "
                "total variation; oracle-backed circular simulator-only evidence"
            )
    else:
        artifact["status"] = "complete_null"
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete: null result; matched refinements did not reduce exact mean trajectory "
            "total variation"
        )
    return _finish_common(artifact)


def _blocked_artifact(reason: str, duration_s: float, torx_sidecar: Mapping[str, Any]) -> JsonDict:
    """Preserve every required field when the owned exact path is unavailable."""

    artifact = _base_artifact(duration_s, torx_sidecar)
    artifact["status"] = "complete_blocked_compiler_reference"
    artifact["gate_check_summary"] = [
        {
            "check": "internal_exact_reference",
            "expected": {"available": True},
            "observed": {"available": False, "reason": reason},
        }
    ]
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"complete_blocked_compiler_reference: {reason}"
    return _finish_common(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute row, receipt, aggregate, completion, and claim-boundary gates."""

    errors: list[str] = []
    missing = sorted(set(TOP_LEVEL_PRINCIPLES) - set(artifact))
    if missing:
        return ["required_fields_missing"]
    if set(artifact["field_principles"]) != set(artifact):
        errors.append("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["hardware_used"] is not False:
        errors.append("hardware_boundary_mismatch")
    if artifact["simulator_used"] is not True:
        errors.append("simulator_boundary_mismatch")
    duration = artifact["duration_s"]
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        errors.append("duration_invalid")
    for row in artifact["topology_receipts"]:
        if row.get("receipt_sha256") != receipt_hash(row):
            errors.append("topology_receipt_hash_mismatch")
            break
    for row in artifact["precision_receipts"]:
        if row.get("receipt_sha256") != receipt_hash(row):
            errors.append("precision_receipt_hash_mismatch")
            break
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")

    blocked = artifact["status"] == "complete_blocked_compiler_reference"
    if blocked:
        if (
            artifact["compiler_fidelity_completed"] is not False
            or artifact["verdict_class"] != "blocked"
            or not str(artifact["honest_verdict"]).startswith("complete_blocked_compiler_reference")
            or not artifact["gate_check_summary"]
        ):
            errors.append("blocked_terminal_state_mismatch")
        return errors

    rows = artifact["rows"]
    expected_grid = _expected_grid()
    if len(rows) != len(expected_grid) or _row_grid(rows) != expected_grid:
        errors.append("row_count_mismatch")
    if any(row.get("row_sha256") != row_hash(row) for row in rows):
        errors.append("row_hash_mismatch")
    derived = derive_aggregates(rows)
    if artifact["conditional_kl_by_factor"] != derived["conditional_kl_by_factor"]:
        errors.append("conditional_kl_by_factor_mismatch")
    if artifact["trajectory_tv_by_depth"] != derived["trajectory_tv_by_depth"]:
        errors.append("trajectory_tv_by_depth_mismatch")
    if artifact["normalization_error_by_row"] != derived["normalization_error_by_row"]:
        errors.append("normalization_error_by_row_mismatch")
    if artifact["positive_result_gate"] != derived["positive_result_gate"]:
        errors.append("positive_result_gate_mismatch")
    normalization_valid = all(
        float(row["maximum_normalization_error"]) <= NORMALIZATION_TOLERANCE for row in rows
    )
    should_complete = (
        len(rows) == len(expected_grid)
        and _row_grid(rows) == expected_grid
        and all(row.get("row_sha256") == row_hash(row) for row in rows)
        and normalization_valid
    )
    if artifact["compiler_fidelity_completed"] is not should_complete:
        errors.append("completion_mismatch")
    if should_complete and artifact["gate_check_summary"]:
        errors.append("completed_gate_summary_not_empty")
    return errors


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the deliverable atomically so partial JSON is never authoritative."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    output_path: Path | str = RESULT_PATH,
    row_builder: RowBuilder = build_rows,
    torx_sidecar: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build and write a completed result or an owned exact-reference block."""

    started = time.perf_counter()
    sidecar = dict(torx_sidecar) if torx_sidecar is not None else inspect_torx_sidecar()
    try:
        rows = row_builder()
    except CompilerReferenceUnavailable as exc:
        artifact = _blocked_artifact(str(exc), time.perf_counter() - started, sidecar)
    else:
        artifact = build_artifact(
            rows=rows,
            duration_s=time.perf_counter() - started,
            torx_sidecar=sidecar,
        )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise CompilerInputError(f"artifact validation failed: {validation_errors}")
    _write_json(Path(output_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded compiler reference from the repository command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--skip-torx-sidecar",
        action="store_true",
        help="Record an explicit skipped sidecar while keeping the internal exact run.",
    )
    args = parser.parse_args(argv)
    sidecar = unavailable_sidecar("skipped_by_cli") if args.skip_torx_sidecar else None
    artifact = run(output_path=args.output, torx_sidecar=sidecar)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "compiler_fidelity_completed": artifact["compiler_fidelity_completed"],
                "verdict_class": artifact["verdict_class"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
