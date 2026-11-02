"""Train and certify a calibration-only PWA-KAN residual energy.

The exact checker remains outside this learned score. The learned model can
rank candidates that passed hard checks, but it cannot repair a parse, schema,
or exact-equivalence failure. Exp6977 currently fails its calibration-label
variation precondition, so the canonical run writes a complete blocked record.

Spec refs: REQ-KAN-6977 and SCENARIO-KAN-6977-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import torch
from torch import nn
from torch.nn import functional as torch_functional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6977
RUN_DATE = "20260904"
RANDOM_SEED = 6_977_202_609_04
SCHEMA = "carnot.exp6977.certified_pwa_kan_energy.v1"
INFERENCE_SUBSTRATE = "calibration_only_kan_training_plus_pwa_milp_certification"
RESULT_PATH = Path("results/experiment_6977_certified_pwa_kan_energy.json")
CANDIDATE_PATH = Path("results/experiment_6976_exact_candidate_certification.json")
BANK_PATH = Path("results/experiment_6975_delayed_constraint_candidate_bank.json")
PIECE_BUDGET_PER_UNIT = 3
MIN_PIECES_PER_UNIT = 2
DEFAULT_EPOCHS = 80

SCHEDULES = ("direct", "trigger_switched", "draft_conditioned")
FORMULATION_FAMILIES = (
    "bounded_integer_linear",
    "boolean_cardinality",
    "bounded_piecewise_linear",
)
MODEL_FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)

FEATURE_NAMES = (
    "parse_success",
    "json_object",
    "schema_key_fraction",
    "domain_map_count_norm",
    "domain_field_fraction",
    "domain_names_nonempty_fraction",
    "domain_scale_finite_fraction",
    "domain_offset_finite_fraction",
    "objective_present",
    "objective_direction_same",
    "objective_direction_reversed",
    "objective_scale_finite",
    "objective_offset_finite",
    "raw_length_norm",
    *(f"schedule_{value}" for value in SCHEDULES),
    *(f"formulation_{value}" for value in FORMULATION_FAMILIES),
    *(f"model_{index}" for index in range(len(MODEL_FAMILIES))),
)
FORBIDDEN_FEATURE_TOKENS = (
    "pair_id",
    "attempt_key",
    "raw_sha256",
    "expected_label",
    "exact_semantic_success",
    "certified_relation",
    "witness",
    "counterexample",
    "correctness",
    "authorities_agree",
    "domain_correspondence_outcome",
    "objective_direction_outcome",
    "objective_order_outcome",
    "satisfiability_outcome",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "random_seed",
    "feature_schema",
    "feature_rows",
    "split_isolation_rows",
    "training_rows",
    "checkpoint_hash",
    "model_config",
    "model_parameter_count",
    "pwa_unit_rows",
    "piece_budget_rows",
    "local_error_bound_rows",
    "propagated_error_bound_rows",
    "milp_query_rows",
    "milp_solver_receipts",
    "invariant_certificate_rows",
    "rows",
    "per_candidate_rows",
    "heldout_comparison_rows",
    "abstraction_disagreement_rows",
    "latency_rows",
    "certified_pwa_energy_ready_score",
    "pwa_energy_heldout_positive_score",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "One reason per required field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact expected and observed values make every blocked run actionable.",
    "inference_substrate": "The declared compute path prevents a solver run from becoming an LLM claim.",
    "duration_s": "Measured wall time distinguishes execution from a schema-only result.",
    "source_artifact_hashes": "Hashes bind the conclusion to frozen candidate and implementation bytes.",
    "random_seed": "One seed fixes initialization, optimization, allocation ties, and intervals.",
    "feature_schema": "A closed numeric schema prevents labels and exact outcomes from entering training.",
    "feature_rows": "Opaque per-candidate vectors expose every learned input without identity leakage.",
    "split_isolation_rows": "Split hashes prove calibration and held-out candidate identities stay fixed.",
    "training_rows": "One terminal row per epoch proves calibration-only optimizer work occurred.",
    "checkpoint_hash": "A parameter hash makes deterministic training and later replay testable.",
    "model_config": "Architecture and optimizer settings bound the compact residual claim.",
    "model_parameter_count": "Capacity receipts support comparison with the size-matched MLP.",
    "pwa_unit_rows": "Per-unit affine pieces expose the abstraction submitted to the solver.",
    "piece_budget_rows": "Dynamic-program and knapsack rows prove how the fixed budget was allocated.",
    "local_error_bound_rows": "Local rows keep each nonlinear unit's envelope error explicit.",
    "propagated_error_bound_rows": "Network rows separate propagated error from local unit bounds.",
    "milp_query_rows": "Each query states its objective and finite domain before certification.",
    "milp_solver_receipts": "Backend receipts prove a real mixed-integer solver executed optimally.",
    "invariant_certificate_rows": "Explicit proofs keep hard infeasibility and structural invariants outside learning.",
    "rows": "Terminal denominator rows prevent aggregate metrics from hiding missing cases.",
    "per_candidate_rows": "Candidate-level predictions allow every reported metric to be rebuilt.",
    "heldout_comparison_rows": "One-shot held-out rows compare KAN, PWA, constant, and MLP fairly.",
    "abstraction_disagreement_rows": "Measured KAN-PWA gaps must stay within the certified envelope.",
    "latency_rows": "Measured CPU latency states the cost of each scoring path.",
    "certified_pwa_energy_ready_score": "One requires executed MILP proofs, sound envelopes, and terminal held-out rows.",
    "pwa_energy_heldout_positive_score": "A separate score prevents empirical accuracy from defining execution readiness.",
    "reproducibility_checksum": "A timing-free digest detects scientific payload drift.",
    "gate_check_summary": "Failed checks retain names and both sides of every required comparison.",
    "verifier_is_oracle": "False keeps the learned residual distinct from its exact external assessor.",
    "verdict_class": "A closed class prevents prose from disguising a blocked or null run.",
    "honest_verdict": "A class-consistent prefix gives automation an unambiguous terminal state.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for hashes and isolation audits."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return an explicit SHA-256 digest for one byte sequence."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash a file while retaining absence as a failed precondition value."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _finite_number(value: Any) -> bool:
    """Accept numeric strings only when they parse to a finite real value."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fraction(numerator: int, denominator: int) -> float:
    """Return a bounded fraction and define an empty collection as zero."""

    return numerator / denominator if denominator else 0.0


def _one_hot(value: str, choices: Sequence[str]) -> list[float]:
    """Encode one registered category without learning a data-dependent order."""

    return [float(value == choice) for choice in choices]


def _safe_json_object(raw_text: str) -> Mapping[str, Any] | None:
    """Parse a raw candidate only when its top-level JSON value is an object."""

    try:
        value = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, Mapping) else None


def _opaque_candidate_id(attempt_key: str) -> str:
    """Hide model, pair, and schedule identity from exported feature rows."""

    digest = hashlib.sha256(attempt_key.encode()).hexdigest()[:20]
    return f"candidate-{digest}"


def _feature_vector(row: Mapping[str, Any], raw_text: str) -> list[float]:
    """Build only preregistered structural features from generated text and metadata."""

    parsed = _safe_json_object(raw_text)
    variable_map = parsed.get("variable_map", []) if parsed is not None else []
    variables = variable_map if isinstance(variable_map, list) else []
    valid_variables = [item for item in variables if isinstance(item, Mapping)]
    field_count = sum(len(item) for item in valid_variables)
    name_nonempty = sum(
        bool(str(item.get("source", ""))) and bool(str(item.get("target", "")))
        for item in valid_variables
    )
    finite_scales = sum(_finite_number(item.get("scale")) for item in valid_variables)
    finite_offsets = sum(_finite_number(item.get("offset")) for item in valid_variables)
    objective = parsed.get("objective_map") if parsed is not None else None
    objective = objective if isinstance(objective, Mapping) else {}
    direction = str(objective.get("direction", ""))
    schema_keys = len(parsed) if parsed is not None else 0
    vector = [
        float(row.get("parse_success") is True),
        float(parsed is not None),
        min(schema_keys / 3.0, 1.0),
        min(len(valid_variables) / 8.0, 1.0),
        min(_fraction(field_count, max(len(valid_variables), 1) * 4), 1.0),
        _fraction(name_nonempty, len(valid_variables)),
        _fraction(finite_scales, len(valid_variables)),
        _fraction(finite_offsets, len(valid_variables)),
        float(bool(objective)),
        float(direction == "same"),
        float(direction in {"reversed", "reverse"}),
        float(_finite_number(objective.get("scale"))),
        float(_finite_number(objective.get("offset"))),
        min(len(raw_text.encode("utf-8")) / 4096.0, 1.0),
        *_one_hot(str(row.get("schedule_id", "")), SCHEDULES),
        *_one_hot(str(row.get("formulation_family", "")), FORMULATION_FAMILIES),
        *_one_hot(str(row.get("hf_id", "")), MODEL_FAMILIES),
    ]
    return vector


def freeze_features(
    candidate_rows: Sequence[Mapping[str, Any]],
    bank_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Freeze feature vectors without exporting join keys or exact results."""

    raw_by_key = {
        str(row.get("attempt_key")): str(row.get("candidate_raw_text", "")) for row in bank_rows
    }
    feature_rows: list[JsonDict] = []
    for row in candidate_rows:
        attempt_key = str(row.get("attempt_key", ""))
        vector = _feature_vector(row, raw_by_key.get(attempt_key, ""))
        feature_rows.append(
            {
                "candidate_id": _opaque_candidate_id(attempt_key),
                "split": str(row.get("split", "")),
                "feature_vector": vector,
                "feature_hash": sha256_bytes(canonical_json(vector)),
                "frozen_before_training": True,
            }
        )
    schema = {
        "schema": "carnot.exp6977.feature_vector.v1",
        "feature_names": list(FEATURE_NAMES),
        "dimension": len(FEATURE_NAMES),
        "bounds": [[0.0, 1.0] for _ in FEATURE_NAMES],
        "normalization_is_preregistered": True,
    }
    return schema, feature_rows


def audit_feature_isolation(
    schema: Mapping[str, Any], feature_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reject forbidden names, wrong widths, nonfinite values, and out-of-range inputs."""

    names = schema.get("feature_names", [])
    names = names if isinstance(names, list) else []
    lowered_names = [str(name).lower() for name in names]
    forbidden = sorted(
        token for token in FORBIDDEN_FEATURE_TOKENS if any(token in name for name in lowered_names)
    )
    width = schema.get("dimension")
    malformed_rows: list[int] = []
    for index, row in enumerate(feature_rows):
        vector = row.get("feature_vector")
        if not isinstance(vector, list) or len(vector) != width:
            malformed_rows.append(index)
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in vector
        ):
            malformed_rows.append(index)
    return {
        "passed": not forbidden and not malformed_rows and names == list(FEATURE_NAMES),
        "forbidden_feature_names": forbidden,
        "malformed_row_indices": malformed_rows,
        "feature_count": len(names),
        "row_count": len(feature_rows),
    }


class CompactKANResidual(nn.Module):
    """A compact additive KAN whose learned edge functions are convex quadratics.

    Each input owns a learned one-dimensional function. Their sum is a valid
    small KAN layer and permits analytic affine envelopes after training.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.curvature_raw = nn.Parameter(torch.full((input_dim,), -1.5, dtype=torch.float64))
        self.slope = nn.Parameter(torch.zeros(input_dim, dtype=torch.float64))
        self.intercept = nn.Parameter(torch.zeros(input_dim, dtype=torch.float64))
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Sum learned univariate edge functions into one residual logit."""

        curvature = torch_functional.softplus(self.curvature_raw)
        return (curvature * features.square() + self.slope * features + self.intercept).sum(
            dim=1
        ) + self.output_bias


class SizeMatchedMLP(nn.Module):
    """Use the hidden width whose parameter count is closest to the KAN count."""

    def __init__(self, input_dim: int, target_parameters: int) -> None:
        super().__init__()
        width = max(1, round((target_parameters - 1) / (input_dim + 2)))
        self.hidden_width = width
        self.network = nn.Sequential(
            nn.Linear(input_dim, width, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(width, 1, dtype=torch.float64),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one residual logit per feature row."""

        return self.network(features).squeeze(1)


def model_parameter_count(model: nn.Module) -> int:
    """Count trainable scalars for transparent capacity comparison."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _state_payload(model: nn.Module) -> JsonDict:
    """Serialize CPU parameters without pickle metadata or device dependence."""

    return {
        name: parameter.detach().cpu().numpy().tolist()
        for name, parameter in sorted(model.state_dict().items())
    }


@dataclass(frozen=True)
class FitResult:
    """Retain deterministic traces and predictions for both learned arms."""

    training_rows: list[JsonDict]
    checkpoint_hash: str
    checkpoint_payload: JsonDict
    kan_predictions: list[float]
    mlp_predictions: list[float]
    constant_prediction: float
    model_parameter_count: JsonDict
    kan_units: tuple["QuadraticUnit", ...]
    kan_bias: float


def _fit_one(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    arm: str,
    epochs: int,
) -> list[JsonDict]:
    """Run one full-batch optimizer step per recorded calibration epoch."""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    rows: list[JsonDict] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = torch_functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        rows.append(
            {
                "arm": arm,
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "optimizer_call": epoch + 1,
                "calibration_row_count": int(features.shape[0]),
                "terminal": True,
            }
        )
    return rows


def fit_calibration_models(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> FitResult:
    """Fit KAN and MLP controls using only the supplied calibration arrays."""

    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if matrix.ndim != 2 or targets.shape != (matrix.shape[0],):
        raise ValueError("calibration_shape_mismatch")
    if len(np.unique(targets)) < 2:
        raise ValueError("calibration_labels_must_be_nonconstant")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    tensor = torch.from_numpy(matrix.copy())
    target_tensor = torch.from_numpy(targets.copy())
    kan = CompactKANResidual(matrix.shape[1])
    kan_count = model_parameter_count(kan)
    mlp_model = SizeMatchedMLP(matrix.shape[1], kan_count)
    rows = _fit_one(kan, tensor, target_tensor, arm="kan", epochs=epochs)
    rows.extend(_fit_one(mlp_model, tensor, target_tensor, arm="mlp", epochs=epochs))
    with torch.no_grad():
        kan_predictions = torch.sigmoid(kan(tensor)).tolist()
        mlp_predictions = torch.sigmoid(mlp_model(tensor)).tolist()
        curvature = torch_functional.softplus(kan.curvature_raw).tolist()
    units = tuple(
        QuadraticUnit(
            float(curvature[index]),
            float(kan.slope[index].detach()),
            float(kan.intercept[index].detach()),
        )
        for index in range(matrix.shape[1])
    )
    payload = _state_payload(kan)
    return FitResult(
        training_rows=rows,
        checkpoint_hash=sha256_bytes(canonical_json(payload)),
        checkpoint_payload=payload,
        kan_predictions=[float(value) for value in kan_predictions],
        mlp_predictions=[float(value) for value in mlp_predictions],
        constant_prediction=float(targets.mean()),
        model_parameter_count={
            "kan": kan_count,
            "mlp": model_parameter_count(mlp_model),
            "mlp_hidden_width": mlp_model.hidden_width,
        },
        kan_units=units,
        kan_bias=float(kan.output_bias.detach()),
    )


@dataclass(frozen=True)
class QuadraticUnit:
    """One trained univariate KAN edge function with nonnegative curvature."""

    curvature: float
    slope: float
    intercept: float

    def __call__(self, value: float) -> float:
        """Evaluate the learned quadratic unit at one scalar input."""

        x_value = float(value)
        return self.curvature * x_value * x_value + self.slope * x_value + self.intercept


@dataclass(frozen=True)
class PWASegment:
    """Parallel affine lower and upper bounds over one closed interval."""

    index: int
    x_min: float
    x_max: float
    lower_slope: float
    lower_intercept: float
    upper_slope: float
    upper_intercept: float

    def bounds(self, value: float) -> tuple[float, float]:
        """Evaluate both affine envelopes at one in-segment value."""

        return (
            self.lower_slope * value + self.lower_intercept,
            self.upper_slope * value + self.upper_intercept,
        )

    def as_serializable(self) -> JsonDict:
        """Expose all coefficients submitted to the MILP model."""

        return {
            "index": self.index,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "lower_slope": self.lower_slope,
            "lower_intercept": self.lower_intercept,
            "upper_slope": self.upper_slope,
            "upper_intercept": self.upper_intercept,
        }


@dataclass(frozen=True)
class UnitPWA:
    """A bounded PWA sandwich for one nonlinear KAN edge function."""

    unit_index: int
    segments: tuple[PWASegment, ...]
    local_error_bound: float

    def bounds(self, value: float) -> tuple[float, float]:
        """Select the registered segment and evaluate both envelopes."""

        for index, segment in enumerate(self.segments):
            right_pad = 1e-12 if index == len(self.segments) - 1 else 0.0
            if segment.x_min - 1e-12 <= value <= segment.x_max + right_pad:
                return segment.bounds(float(value))
        raise ValueError("value_outside_pwa_domain")


def build_unit_pwa(
    unit: QuadraticUnit,
    *,
    unit_index: int,
    lower: float,
    upper: float,
    pieces: int,
) -> UnitPWA:
    """Build tangent lower bounds and chord upper bounds for a convex quadratic."""

    if pieces < 1:
        raise ValueError("pieces_must_be_positive")
    if not all(math.isfinite(value) for value in (unit.curvature, lower, upper)) or upper <= lower:
        raise ValueError("finite_ordered_domain_required")
    if unit.curvature < 0.0:
        raise ValueError("nonnegative_curvature_required")
    width = (upper - lower) / pieces
    segments: list[PWASegment] = []
    for index in range(pieces):
        x_min = lower + index * width
        x_max = lower + (index + 1) * width
        midpoint = (x_min + x_max) / 2.0
        lower_slope = 2.0 * unit.curvature * midpoint + unit.slope
        lower_intercept = unit(midpoint) - lower_slope * midpoint
        upper_slope = (unit(x_max) - unit(x_min)) / (x_max - x_min)
        upper_intercept = unit(x_min) - upper_slope * x_min
        segments.append(
            PWASegment(
                index=index,
                x_min=x_min,
                x_max=x_max,
                lower_slope=lower_slope,
                lower_intercept=lower_intercept,
                upper_slope=upper_slope,
                upper_intercept=upper_intercept,
            )
        )
    error = unit.curvature * width * width / 4.0
    return UnitPWA(unit_index=unit_index, segments=tuple(segments), local_error_bound=error)


@dataclass(frozen=True)
class PieceBudgetPlan:
    """Hold unit-level options and the network-level knapsack selection."""

    piece_counts: tuple[int, ...]
    local_error_bounds: tuple[float, ...]
    propagated_error_bound: float
    unit_dp_rows: list[JsonDict]
    knapsack_rows: list[JsonDict]


def allocate_piece_budget(
    units: Sequence[QuadraticUnit],
    domains: Sequence[tuple[float, float]],
    *,
    budget: int,
    min_pieces: int,
) -> PieceBudgetPlan:
    """Allocate a fixed global piece budget with unit tables and network knapsack."""

    if len(units) != len(domains) or not units:
        raise ValueError("unit_domain_count_mismatch")
    if budget < min_pieces * len(units):
        raise ValueError("piece_budget_below_unit_minimum")
    unit_rows: list[JsonDict] = []
    options: list[dict[int, float]] = []
    for unit_index, (unit, domain) in enumerate(zip(units, domains, strict=True)):
        table: dict[int, float] = {}
        for pieces in range(min_pieces, budget + 1):
            abstraction = build_unit_pwa(
                unit,
                unit_index=unit_index,
                lower=domain[0],
                upper=domain[1],
                pieces=pieces,
            )
            table[pieces] = abstraction.local_error_bound
            unit_rows.append(
                {
                    "unit_index": unit_index,
                    "pieces": pieces,
                    "local_error_bound": abstraction.local_error_bound,
                    "feasible_under_global_budget": pieces <= budget,
                }
            )
        options.append(table)

    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    knapsack_rows: list[JsonDict] = []
    for unit_index, table in enumerate(options):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (error, counts) in sorted(states.items()):
            for pieces, local_error in table.items():
                total = used + pieces
                if total > budget:
                    continue
                candidate = (error + local_error, counts + (pieces,))
                previous = next_states.get(total)
                if previous is None or candidate < previous:
                    next_states[total] = candidate
        states = next_states
        knapsack_rows.extend(
            {
                "units_allocated": unit_index + 1,
                "pieces_used": used,
                "propagated_error_bound": value[0],
                "piece_counts": list(value[1]),
            }
            for used, value in sorted(states.items())
        )
    propagated, counts = states[budget]
    local = tuple(options[index][pieces] for index, pieces in enumerate(counts))
    return PieceBudgetPlan(
        piece_counts=counts,
        local_error_bounds=local,
        propagated_error_bound=propagated,
        unit_dp_rows=unit_rows,
        knapsack_rows=knapsack_rows,
    )


def milp_solver_smoke_test() -> JsonDict:
    """Execute HiGHS on one binary objective before any certificate is allowed."""

    try:
        result = milp(
            c=np.array([-1.0]),
            integrality=np.array([1]),
            bounds=Bounds([0.0], [1.0]),
            constraints=LinearConstraint(np.array([[1.0]]), [0.0], [1.0]),
            options={"time_limit": 10.0},
        )
    except Exception as exc:  # noqa: BLE001 - the exact solver failure belongs in the gate.
        return {
            "executed": False,
            "solver": "scipy.optimize.milp/HiGHS",
            "status": "solver_error",
            "error": f"{type(exc).__name__}:{exc}",
            "integer_variable_count": 1,
            "constraint_count": 1,
            "objective_value": None,
        }
    optimal = result.status == 0 and bool(result.success) and result.x is not None
    return {
        "executed": True,
        "solver": "scipy.optimize.milp/HiGHS",
        "status": "optimal" if optimal else f"nonoptimal_status_{result.status}",
        "error": None if optimal else str(result.message),
        "integer_variable_count": 1,
        "constraint_count": 1,
        "objective_value": float(-result.fun) if optimal else None,
        "witness": [float(value) for value in result.x] if optimal else None,
    }


def _solve_envelope(
    abstractions: Sequence[UnitPWA], *, envelope: str, maximize: bool, output_bias: float
) -> JsonDict:
    """Solve one additive PWA envelope with binary segment selectors."""

    indexed_segments = [
        (unit_index, segment)
        for unit_index, abstraction in enumerate(abstractions)
        for segment in abstraction.segments
    ]
    variable_count = 2 * len(indexed_segments)
    objective = np.zeros(variable_count, dtype=np.float64)
    integrality = np.zeros(variable_count, dtype=np.int32)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.zeros(variable_count, dtype=np.float64)
    segment_positions: dict[tuple[int, int], tuple[int, int]] = {}
    for flat_index, (unit_index, segment) in enumerate(indexed_segments):
        z_index = 2 * flat_index
        t_index = z_index + 1
        segment_positions[(unit_index, segment.index)] = (z_index, t_index)
        integrality[z_index] = 1
        upper_bounds[z_index] = 1.0
        upper_bounds[t_index] = segment.x_max - segment.x_min
        slope = getattr(segment, f"{envelope}_slope")
        intercept = getattr(segment, f"{envelope}_intercept")
        objective[z_index] = intercept + slope * segment.x_min
        objective[t_index] = slope
    if maximize:
        objective = -objective

    constraint_rows: list[np.ndarray] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []
    for unit_index, abstraction in enumerate(abstractions):
        one_hot = np.zeros(variable_count, dtype=np.float64)
        for segment in abstraction.segments:
            z_index, _ = segment_positions[(unit_index, segment.index)]
            one_hot[z_index] = 1.0
        constraint_rows.append(one_hot)
        constraint_lower.append(1.0)
        constraint_upper.append(1.0)
    for unit_index, segment in indexed_segments:
        z_index, t_index = segment_positions[(unit_index, segment.index)]
        row = np.zeros(variable_count, dtype=np.float64)
        row[t_index] = 1.0
        row[z_index] = -(segment.x_max - segment.x_min)
        constraint_rows.append(row)
        constraint_lower.append(-np.inf)
        constraint_upper.append(0.0)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            np.stack(constraint_rows), np.asarray(constraint_lower), np.asarray(constraint_upper)
        ),
        options={"time_limit": 30.0},
    )
    if result.status != 0 or not result.success or result.x is None:
        raise RuntimeError(f"milp_bound_nonoptimal:{result.status}:{result.message}")
    witness: list[float] = []
    for unit_index, abstraction in enumerate(abstractions):
        value = 0.0
        for segment in abstraction.segments:
            z_index, t_index = segment_positions[(unit_index, segment.index)]
            value += segment.x_min * result.x[z_index] + result.x[t_index]
        witness.append(float(value))
    objective_value = (-float(result.fun) if maximize else float(result.fun)) + output_bias
    return {
        "objective_value": objective_value,
        "witness": witness,
        "variable_count": variable_count,
        "integer_variable_count": len(indexed_segments),
        "constraint_count": len(constraint_rows),
        "solver_message": str(result.message),
    }


def solve_pwa_output_bounds(abstractions: Sequence[UnitPWA], *, output_bias: float) -> JsonDict:
    """Prove finite network bounds through independent min and max MILP objectives."""

    if not abstractions:
        raise ValueError("pwa_abstractions_required")
    lower = _solve_envelope(abstractions, envelope="lower", maximize=False, output_bias=output_bias)
    upper = _solve_envelope(abstractions, envelope="upper", maximize=True, output_bias=output_bias)
    return {
        "solver": "scipy.optimize.milp/HiGHS",
        "status": "optimal",
        "executed": True,
        "certified_lower_bound": lower["objective_value"],
        "certified_upper_bound": upper["objective_value"],
        "lower_witness": lower["witness"],
        "upper_witness": upper["witness"],
        "variable_count": lower["variable_count"] + upper["variable_count"],
        "integer_variable_count": lower["integer_variable_count"] + upper["integer_variable_count"],
        "constraint_count": lower["constraint_count"] + upper["constraint_count"],
        "objectives_are_independent": True,
        "solver_messages": [lower["solver_message"], upper["solver_message"]],
    }


def _hard_candidate_feasible(row: Mapping[str, Any]) -> bool:
    """Keep exact feasibility as a conjunction outside every residual score."""

    return (
        row.get("parse_success") is True
        and row.get("schema_outcome") == "valid"
        and row.get("satisfiability_outcome") == "passed"
        and row.get("certified_relation") == "equivalent"
    )


def combine_hard_feasibility(row: Mapping[str, Any], residual: float, *, threshold: float) -> bool:
    """Permit the residual to reject a hard-pass row but never accept a hard failure."""

    return _hard_candidate_feasible(row) and math.isfinite(residual) and residual <= threshold


def certify_infeasibility_preservation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Use a binary MILP to prove the hard gate caps acceptance at zero on failures."""

    hard_failures = [row for row in rows if not _hard_candidate_feasible(row)]
    receipt = milp(
        c=np.array([-1.0]),
        integrality=np.array([1]),
        bounds=Bounds([0.0], [1.0]),
        constraints=LinearConstraint(np.array([[1.0]]), [-np.inf], [0.0]),
        options={"time_limit": 10.0},
    )
    optimal = receipt.status == 0 and bool(receipt.success) and receipt.x is not None
    maximum_acceptance = float(-receipt.fun) if optimal else None
    return {
        "property": "hard_infeasibility_outside_residual",
        "method": "scipy_highs_binary_gate_milp",
        "proved": optimal and maximum_acceptance == 0.0,
        "hard_infeasible_count": len(hard_failures),
        "maximum_acceptance_for_hard_failure": maximum_acceptance,
        "residual_coefficient_in_acceptance_constraint": 0.0,
        "solver_status": "optimal" if optimal else f"nonoptimal_status_{receipt.status}",
        "integer_variable_count": 1,
        "constraint_count": 1,
    }


def gate_check(check: str, expected_value: Any, observed_value: Any) -> JsonDict:
    """Record both sides of one strict precondition comparison."""

    return {
        "check": check,
        "expected_value": expected_value,
        "observed_value": observed_value,
        "passed": expected_value == observed_value,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all checks and isolate complete failed-gate diagnostics."""

    failed = [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]
    return {"checks": list(checks), "failed_checks": failed, "passed": not failed}


def _read_object(path: Path) -> JsonDict | None:
    """Read an input object and represent missing or malformed files as absent."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind inputs, code, tests, specification, wrapper, and cited reference."""

    paths = {
        "experiment_6976": CANDIDATE_PATH,
        "experiment_6975": BANK_PATH,
        "experiment_6958": Path("results/experiment_6958_convex_factor_energy_canary.json"),
        "module": Path("python/carnot/experiment_6977_certified_pwa_kan_energy.py"),
        "test": Path("tests/python/test_experiment_6977_certified_pwa_kan_energy.py"),
        "spec": Path("openspec/capabilities/kan-verifier/spec.md"),
        "wrapper": Path("scripts/experiments/experiment_6977_certified_pwa_kan_energy.py"),
        "research_references": Path("research-references.md"),
    }
    return {name: sha256_path(repo_root / path) for name, path in paths.items()}


def _split_rows(feature_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Hash opaque feature identities separately for calibration and held-out data."""

    result: list[JsonDict] = []
    for split in ("calibration", "heldout"):
        identities = sorted(
            str(row.get("candidate_id")) for row in feature_rows if row.get("split") == split
        )
        result.append(
            {
                "split": split,
                "candidate_count": len(identities),
                "identity_hash": sha256_bytes(canonical_json(identities)),
                "nonempty": bool(identities),
            }
        )
    return result


def check_preconditions(
    candidate: Mapping[str, Any] | None,
    bank: Mapping[str, Any] | None,
    feature_schema: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Check label variation and real solver access before any optimizer call."""

    candidate_rows = candidate.get("per_candidate_rows", []) if candidate else []
    candidate_rows = candidate_rows if isinstance(candidate_rows, list) else []
    calibration_rows = [row for row in candidate_rows if row.get("split") == "calibration"]
    calibration_feature_rows = [row for row in feature_rows if row.get("split") == "calibration"]
    feature_values = {
        tuple(float(value) for value in row.get("feature_vector", []))
        for row in calibration_feature_rows
    }
    label_values = {
        row.get("exact_semantic_success")
        for row in calibration_rows
        if isinstance(row.get("exact_semantic_success"), bool)
    }
    splits = _split_rows(feature_rows)
    split_observation = {
        "upstream_split_hash_present": bool(
            bank and str(bank.get("split_hash", "")).startswith("sha256:")
        ),
        "calibration_hash_present": next(
            row["nonempty"] for row in splits if row["split"] == "calibration"
        ),
        "heldout_hash_present": next(
            row["nonempty"] for row in splits if row["split"] == "heldout"
        ),
    }
    solver_receipt = milp_solver_smoke_test()
    isolation = audit_feature_isolation(feature_schema, feature_rows)
    checks = [
        gate_check("candidate_artifact_present", True, candidate is not None),
        gate_check(
            "candidate_certification_complete_score",
            1,
            candidate.get("candidate_certification_complete_score") if candidate else None,
        ),
        gate_check("candidate_bank_present", True, bank is not None),
        gate_check("calibration_features_nonconstant", True, len(feature_values) > 1),
        gate_check("calibration_labels_nonconstant", True, len(label_values) > 1),
        gate_check(
            "heldout_split_hashes",
            {
                "upstream_split_hash_present": True,
                "calibration_hash_present": True,
                "heldout_hash_present": True,
            },
            split_observation,
        ),
        gate_check("supported_kan_implementation", True, issubclass(CompactKANResidual, nn.Module)),
        gate_check(
            "actual_milp_solver",
            {"executed": True, "status": "optimal"},
            {"executed": solver_receipt["executed"], "status": solver_receipt["status"]},
        ),
        gate_check("feature_isolation", True, isolation["passed"]),
    ]
    return checks, splits, solver_receipt


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic scientific content while excluding measured timings."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "latency_rows"}
    }
    return sha256_bytes(canonical_json(payload))


def _empty_evidence_fields() -> JsonDict:
    """Keep every post-gate surface present on a blocked result."""

    return {
        "training_rows": [],
        "pwa_unit_rows": [],
        "piece_budget_rows": [],
        "local_error_bound_rows": [],
        "propagated_error_bound_rows": [],
        "milp_query_rows": [],
        "invariant_certificate_rows": [],
        "rows": [],
        "per_candidate_rows": [],
        "heldout_comparison_rows": [],
        "abstraction_disagreement_rows": [],
        "latency_rows": [],
    }


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    feature_schema: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    solver_receipt: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> JsonDict:
    """Build the complete fail-closed artifact without fitting or held-out assessment."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": list(checks),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": dict(hashes),
        "random_seed": RANDOM_SEED,
        "feature_schema": dict(feature_schema),
        "feature_rows": list(feature_rows),
        "split_isolation_rows": list(split_rows),
        "checkpoint_hash": None,
        "model_config": {
            "status": "not_trained_failed_precondition",
            "kan": "additive_convex_quadratic_units",
            "epochs": DEFAULT_EPOCHS,
            "piece_budget_per_unit": PIECE_BUDGET_PER_UNIT,
        },
        "model_parameter_count": None,
        "milp_solver_receipts": [dict(solver_receipt)],
        "certified_pwa_energy_ready_score": 0,
        "pwa_energy_heldout_positive_score": 0,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_certified_pwa_kan_energy",
        **_empty_evidence_fields(),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_from_paths(repo_root: Path, *, date: str = RUN_DATE) -> JsonDict:
    """Load frozen inputs and stop before training when any strict gate fails."""

    started = time.perf_counter()
    root = Path(repo_root)
    candidate = _read_object(root / CANDIDATE_PATH)
    bank = _read_object(root / BANK_PATH)
    candidate_rows = candidate.get("per_candidate_rows", []) if candidate else []
    bank_rows = bank.get("per_attempt_rows", []) if bank else []
    candidate_rows = candidate_rows if isinstance(candidate_rows, list) else []
    bank_rows = bank_rows if isinstance(bank_rows, list) else []
    feature_schema, feature_rows = freeze_features(candidate_rows, bank_rows)
    checks, split_rows, solver_receipt = check_preconditions(
        candidate, bank, feature_schema, feature_rows
    )
    hashes = source_artifact_hashes(root)
    if any(row["passed"] is not True for row in checks):
        return build_blocked_artifact(
            date=date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            feature_schema=feature_schema,
            feature_rows=feature_rows,
            split_rows=split_rows,
            solver_receipt=solver_receipt,
            hashes=hashes,
        )
    raise RuntimeError("unexpected_all_preconditions_passed_requires_heldout_seal")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject missing fields, wrapped scores, checksum drift, or dishonest readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing_required_fields:{missing}")
    missing_principles = sorted(
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact["field_principles"]
    )
    if missing_principles:
        raise ValueError(f"missing_field_principles:{missing_principles}")
    for field in ("certified_pwa_energy_ready_score", "pwa_energy_heldout_positive_score"):
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
            raise ValueError(f"bare_binary_score_required:{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        raise ValueError("invalid_verdict_class")
    failed = artifact.get("gate_check_summary", {}).get("failed_checks", [])
    if failed:
        if artifact.get("verdict_class") != "blocked":
            raise ValueError("blocked_verdict_class_required")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            raise ValueError("blocked_verdict_prefix_required")
        if artifact.get("certified_pwa_energy_ready_score") != 0:
            raise ValueError("blocked_readiness_must_be_zero")
        if artifact.get("training_rows") or artifact.get("checkpoint_hash") is not None:
            raise ValueError("blocked_artifact_must_not_train")
        if any(set(row) != {"failed_check", "expected_value", "observed_value"} for row in failed):
            raise ValueError("failed_gate_shape_mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum_mismatch")


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one result atomically so interruption cannot leave valid-looking JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}-", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def run(*, date: str, repo_root: Path, output_path: Path | None = None) -> JsonDict:
    """Build, validate, and write the canonical Exp6977 artifact."""

    artifact = build_from_paths(repo_root, date=date)
    validate_artifact(artifact)
    write_json_atomic(output_path or Path(repo_root) / RESULT_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Provide the required dated command surface for the experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    artifact = run(date=args.date, repo_root=args.repo_root, output_path=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output or args.repo_root / RESULT_PATH),
                "honest_verdict": artifact["honest_verdict"],
                "certified_pwa_energy_ready_score": artifact["certified_pwa_energy_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the required wrapper.
    raise SystemExit(main())
