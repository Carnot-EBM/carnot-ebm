"""Game-blind invariant projection for executable ARC world-model proposals.

The module contains only the reusable numerical mechanism extracted from the
Exp6595 canary.  It does not load archives, inspect source files, choose a
constraint, or evaluate a prediction against the next observation.  A caller
must explicitly provide a frozen quadratic matrix and enable the configuration;
the disabled configuration returns the original engine object unchanged.

Spec refs: REQ-ARC-WMTE-6611, REQ-ARC-WMTE-6611-LIVE,
REQ-ARC-WMTE-6611-FEATURES, SCENARIO-ARC-WMTE-6611-LIVE,
SCENARIO-ARC-WMTE-6611-ORACLE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


DEFAULT_ALPHA = 1.0
DEFAULT_MAX_ITERATIONS = 32
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class LevelSetProjection:
    """The bounded numerical result for one quadratic level-set projection."""

    state: np.ndarray
    distance: float
    iterations: int
    converged: bool
    failure: str | None
    final_residual: float


@dataclass(frozen=True)
class GridProjection:
    """A projected grid plus diagnostics needed by live replay accounting."""

    grid: np.ndarray
    input_features: tuple[float, float]
    predicted_features: tuple[float, float]
    projected_features: tuple[float, float]
    invariant_drift_before: float
    invariant_drift_after: float
    projection_distance: float
    iterations: int
    converged: bool
    failure: str | None


@dataclass(frozen=True)
class InvariantProjectionConfig:
    """An immutable, explicit opt-in configuration for the live proposal seam.

    ``enabled`` is deliberately false.  A true value without a finite 2-by-2
    matrix is rejected when the wrapper is built, so malformed activation
    cannot silently become a different live policy.
    """

    enabled: bool = False
    quadratic_matrix: tuple[tuple[float, float], tuple[float, float]] | None = None
    alpha: float = DEFAULT_ALPHA
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    tolerance: float = DEFAULT_TOLERANCE
    max_projection_distance: float = 2.0


def quadratic_value(state: np.ndarray, matrix: np.ndarray) -> float:
    """Return ``state.T @ matrix @ state`` as a plain float."""

    vector = np.asarray(state, dtype=np.float64)
    quadratic = np.asarray(matrix, dtype=np.float64)
    return float(vector @ quadratic @ vector)


def project_to_level_set(
    state: np.ndarray,
    quadratic_matrix: np.ndarray,
    target: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    max_distance: float | None = None,
) -> LevelSetProjection:
    """Move a vector toward one quadratic level set with bounded Newton steps.

    The update is the general projector math used by Exp6595.  It stops on a
    zero gradient, a fixed iteration ceiling, or an optional Euclidean cost
    budget.  Every non-convergence reason is returned instead of being hidden.
    """

    vector = np.asarray(state, dtype=np.float64).reshape(-1)
    matrix = np.asarray(quadratic_matrix, dtype=np.float64)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("quadratic matrix must match the state dimension")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("state and quadratic matrix must be finite")
    start = vector.copy()
    iterations = 0
    failure: str | None = None
    converged = False
    for iteration in range(1, max(0, int(max_iterations)) + 1):
        residual = quadratic_value(vector, matrix) - float(target)
        if abs(residual) <= float(tolerance):
            converged = True
            break
        gradient = (matrix + matrix.T) @ vector
        gradient_norm_sq = float(gradient @ gradient)
        if gradient_norm_sq <= 1e-18:
            failure = "zero_gradient"
            break
        candidate = vector - float(alpha) * residual * gradient / gradient_norm_sq
        distance = float(np.linalg.norm(candidate - start))
        if max_distance is not None and distance > float(max_distance):
            failure = "cost_budget_exceeded"
            break
        vector = candidate
        iterations = iteration
    final_residual = abs(quadratic_value(vector, matrix) - float(target))
    if not converged and failure is None:
        converged = final_residual <= float(tolerance)
        if not converged:
            failure = "max_iterations"
    return LevelSetProjection(
        state=vector,
        distance=float(np.linalg.norm(vector - start)),
        iterations=iterations,
        converged=converged,
        failure=failure,
        final_residual=final_residual,
    )


def grid_features(grid: np.ndarray) -> np.ndarray:
    """Return two bounded state features without metadata or identity inputs.

    ARC colors are integers in the inclusive range 0..15.  Mean color and root
    mean square retain coarse palette mass while remaining cheap enough for the
    live proposal path.  Dividing by 15 keeps both coordinates near unit scale.
    """

    array = np.asarray(grid, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("grid must be finite and non-empty")
    return np.asarray(
        [float(np.mean(array)) / 15.0, float(np.sqrt(np.mean(array * array))) / 15.0],
        dtype=np.float64,
    )


def _grid_from_features(predicted: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Apply the projected mean and RMS to a proposal, then restore its dtype."""

    source = np.asarray(predicted)
    values = source.astype(np.float64)
    desired_mean = float(np.clip(desired[0] * 15.0, 0.0, 15.0))
    desired_rms = float(np.clip(desired[1] * 15.0, 0.0, 15.0))
    desired_std = float(np.sqrt(max(0.0, desired_rms * desired_rms - desired_mean**2)))
    source_mean = float(np.mean(values))
    source_std = float(np.std(values))
    if source_std <= 1e-12:
        adjusted = np.full(values.shape, desired_mean, dtype=np.float64)
    else:
        adjusted = (values - source_mean) * (desired_std / source_std) + desired_mean
    adjusted = np.clip(np.rint(adjusted), 0.0, 15.0)
    return adjusted.astype(source.dtype, copy=False)


def project_prediction(
    current_grid: np.ndarray,
    predicted_grid: np.ndarray,
    config: InvariantProjectionConfig,
) -> GridProjection:
    """Project one prediction using only the current frame and that prediction."""

    current = np.asarray(current_grid)
    predicted = np.asarray(predicted_grid)
    if current.shape != predicted.shape or current.ndim != 2:
        raise ValueError("current and predicted grids must share one two-dimensional shape")
    input_features = grid_features(current)
    predicted_features = grid_features(predicted)
    if not config.enabled:
        return GridProjection(
            grid=predicted.copy(),
            input_features=tuple(float(value) for value in input_features),
            predicted_features=tuple(float(value) for value in predicted_features),
            projected_features=tuple(float(value) for value in predicted_features),
            invariant_drift_before=0.0,
            invariant_drift_after=0.0,
            projection_distance=0.0,
            iterations=0,
            converged=True,
            failure=None,
        )
    if config.quadratic_matrix is None:
        raise ValueError("enabled invariant projection requires a quadratic matrix")
    matrix = np.asarray(config.quadratic_matrix, dtype=np.float64)
    target = quadratic_value(input_features, matrix)
    drift_before = abs(quadratic_value(predicted_features, matrix) - target)
    level = project_to_level_set(
        predicted_features,
        matrix,
        target,
        alpha=config.alpha,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
        max_distance=config.max_projection_distance,
    )
    output = predicted.copy() if level.failure else _grid_from_features(predicted, level.state)
    output_features = grid_features(output)
    return GridProjection(
        grid=output,
        input_features=tuple(float(value) for value in input_features),
        predicted_features=tuple(float(value) for value in predicted_features),
        projected_features=tuple(float(value) for value in output_features),
        invariant_drift_before=drift_before,
        invariant_drift_after=abs(quadratic_value(output_features, matrix) - target),
        projection_distance=level.distance,
        iterations=level.iterations,
        converged=level.converged,
        failure=level.failure,
    )


def wrap_world_model_engine(
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    config: InvariantProjectionConfig | None = None,
) -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Wrap a proposal engine only when an explicit enabled config is present."""

    frozen = config or InvariantProjectionConfig()
    if not frozen.enabled:
        return engine
    if frozen.quadratic_matrix is None:
        raise ValueError("enabled invariant projection requires a quadratic matrix")

    def projected_engine(grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        prediction = np.asarray(engine(grid, action, data))
        return project_prediction(np.asarray(grid), prediction, frozen).grid

    projected_engine.__name__ = getattr(engine, "__name__", "projected_world_model_engine")
    projected_engine._carnot_invariant_projection_config = frozen  # type: ignore[attr-defined]
    projected_engine._carnot_unprojected_engine = engine  # type: ignore[attr-defined]
    return projected_engine


def norm_matched_random_matrix(reference: np.ndarray, seed: int) -> np.ndarray:
    """Return a seeded symmetric random matrix with the reference Frobenius norm."""

    matrix = np.asarray(reference, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError("reference quadratic matrix must have shape (2, 2)")
    rng = np.random.default_rng(int(seed))
    random = rng.normal(size=(2, 2))
    random = 0.5 * (random + random.T)
    reference_norm = float(np.linalg.norm(matrix))
    random_norm = float(np.linalg.norm(random))
    if random_norm <= 1e-12:
        raise ValueError("random quadratic matrix has zero norm")
    return random * (reference_norm / random_norm)


def config_sha256(config: InvariantProjectionConfig) -> str:
    """Content-address a frozen projector configuration."""

    payload = {
        "enabled": config.enabled,
        "quadratic_matrix": config.quadratic_matrix,
        "alpha": config.alpha,
        "max_iterations": config.max_iterations,
        "tolerance": config.tolerance,
        "max_projection_distance": config.max_projection_distance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
