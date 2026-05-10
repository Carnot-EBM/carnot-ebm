"""Pi-Net-style Douglas-Rachford projection layer for continuous constraints.

The layer is a small JAX prototype for feasible-by-design latent states. It
does not train a neural network; instead, it applies a bounded differentiable
Douglas-Rachford-style loop built from closed-form projections onto affine
equalities and linear half-spaces.

Spec refs: REQ-KONA-039, SCENARIO-KONA-039.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp

JsonDict = dict[str, object]

EXPERIMENT_ID = 1662
SCHEMA = "carnot.models.pinet_layer.v1"
SPEC_REFS = ["REQ-KONA-039", "SCENARIO-KONA-039"]
MODULE_PATH = "python/carnot/models/pinet_layer.py"
ARTIFACT_PATH = "results/experiment_1662_pinet_layer.json"
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_STEPS = 64
DEFAULT_RELAXATION = 1.0
RIDGE = 1e-7
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "schema",
    "experiment_id",
    "spec_refs",
    "module_path",
    "projection_error",
    "convergence_steps",
    "differentiable_projection",
    "honest_verdict",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_RESULT_PATH = _repo_root() / ARTIFACT_PATH


@dataclass(frozen=True)
class LinearConstraintSet:
    """Linear hard constraints for a continuous latent vector.

    Constraints use the convention ``equality_matrix @ z == equality_target``
    and ``inequality_matrix @ z <= inequality_bound``.

    Spec refs: REQ-KONA-039.
    """

    state_dim: int
    equality_matrix: jax.Array
    equality_target: jax.Array
    inequality_matrix: jax.Array
    inequality_bound: jax.Array
    name: str = "linear_constraints"

    @classmethod
    def from_arrays(
        cls,
        *,
        state_dim: int,
        equality_matrix: Sequence[Sequence[float]] | None = None,
        equality_target: Sequence[float] | None = None,
        inequality_matrix: Sequence[Sequence[float]] | None = None,
        inequality_bound: Sequence[float] | None = None,
        name: str = "linear_constraints",
    ) -> "LinearConstraintSet":
        """Build and validate a linear constraint set.

        Spec refs: REQ-KONA-039.
        """

        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        eq = _as_matrix(equality_matrix, state_dim, "equality_matrix")
        eq_target = _as_vector(equality_target, eq.shape[0], "equality_target")
        ineq = _as_matrix(inequality_matrix, state_dim, "inequality_matrix")
        ineq_bound = _as_vector(inequality_bound, ineq.shape[0], "inequality_bound")
        if eq.shape[0] + ineq.shape[0] == 0:
            raise ValueError("at least one hard constraint is required")
        return cls(
            state_dim=state_dim,
            equality_matrix=eq,
            equality_target=eq_target,
            inequality_matrix=ineq,
            inequality_bound=ineq_bound,
            name=name,
        )

    def projection_error(self, state: jax.Array | Sequence[float]) -> jax.Array:
        """Return the max equality residual or positive inequality violation.

        Spec refs: REQ-KONA-039.
        """

        z = jnp.asarray(state, dtype=jnp.float32)
        eq_error = (
            jnp.max(jnp.abs(self.equality_matrix @ z - self.equality_target))
            if self.equality_matrix.shape[0]
            else jnp.array(0.0, dtype=jnp.float32)
        )
        ineq_error = (
            jnp.max(jnp.maximum(self.inequality_matrix @ z - self.inequality_bound, 0.0))
            if self.inequality_matrix.shape[0]
            else jnp.array(0.0, dtype=jnp.float32)
        )
        return jnp.maximum(eq_error, ineq_error)


@dataclass(frozen=True)
class ProjectionCase:
    """One deterministic projection case for Exp 1662.

    Spec refs: SCENARIO-KONA-039.
    """

    name: str
    start: tuple[float, ...]
    constraints: LinearConstraintSet


@dataclass(frozen=True)
class PiNetProjectionResult:
    """Projection diagnostics for a continuous latent state.

    Spec refs: REQ-KONA-039.
    """

    case_name: str
    projected_state: tuple[float, ...]
    initial_projection_error: float
    projection_error: float
    convergence_steps: int
    converged: bool

    def to_json(self) -> JsonDict:
        """Return a JSON-safe row for experiment artifacts."""

        return {
            "case_name": self.case_name,
            "projected_state": list(self.projected_state),
            "initial_projection_error": self.initial_projection_error,
            "projection_error": self.projection_error,
            "convergence_steps": self.convergence_steps,
            "converged": self.converged,
        }


class DouglasRachfordPiNetLayer:
    """Bounded Douglas-Rachford projection layer for linear constraints.

    The internal Douglas-Rachford state is updated by reflecting across the
    equality projection, projecting the reflection through all inequality
    half-spaces, and relaxing toward the resulting point. The public projection
    candidate is the equality projection of the internal state, which converges
    to the intersection for the linear systems used by this prototype.

    Spec refs: REQ-KONA-039, SCENARIO-KONA-039.
    """

    def __init__(
        self,
        constraints: LinearConstraintSet,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        tolerance: float = DEFAULT_TOLERANCE,
        relaxation: float = DEFAULT_RELAXATION,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not (0.0 < relaxation <= 2.0):
            raise ValueError("relaxation must be in (0, 2]")
        self.constraints = constraints
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.relaxation = relaxation

    def project_vector(self, state: jax.Array | Sequence[float]) -> jax.Array:
        """Project a state with a fixed differentiable unrolled loop.

        Spec refs: REQ-KONA-039.
        """

        work = self._validate_state(state)
        for _ in range(self.max_steps):
            work = self._douglas_rachford_step(work)
        return self._project_equalities(work)

    def project(self, state: jax.Array | Sequence[float]) -> PiNetProjectionResult:
        """Project a state and return convergence diagnostics.

        Spec refs: REQ-KONA-039, SCENARIO-KONA-039.
        """

        work = self._validate_state(state)
        initial_error = _as_float(self.constraints.projection_error(work))
        if initial_error <= self.tolerance:
            return self._result(work, initial_error, initial_error, 0)

        candidate = self._project_equalities(work)
        final_error = _as_float(self.constraints.projection_error(candidate))
        steps = self.max_steps
        for step in range(1, self.max_steps + 1):
            work = self._douglas_rachford_step(work)
            candidate = self._project_equalities(work)
            final_error = _as_float(self.constraints.projection_error(candidate))
            if final_error <= self.tolerance:
                steps = step
                break
        return self._result(candidate, initial_error, final_error, steps)

    def _douglas_rachford_step(self, state: jax.Array) -> jax.Array:
        equality_projection = self._project_equalities(state)
        reflected = 2.0 * equality_projection - state
        inequality_projection = self._project_inequalities(reflected)
        return state + self.relaxation * (inequality_projection - equality_projection)

    def _project_equalities(self, state: jax.Array) -> jax.Array:
        if self.constraints.equality_matrix.shape[0] == 0:
            return state
        matrix = self.constraints.equality_matrix
        residual = matrix @ state - self.constraints.equality_target
        gram = matrix @ matrix.T + RIDGE * jnp.eye(matrix.shape[0], dtype=jnp.float32)
        correction = matrix.T @ jnp.linalg.solve(gram, residual)
        return state - correction

    def _project_inequalities(self, state: jax.Array) -> jax.Array:
        projected = state
        for row_index in range(self.constraints.inequality_matrix.shape[0]):
            row = self.constraints.inequality_matrix[row_index]
            bound = self.constraints.inequality_bound[row_index]
            violation = jnp.dot(row, projected) - bound
            step = jnp.maximum(violation, 0.0) / (jnp.dot(row, row) + RIDGE)
            projected = projected - step * row
        return projected

    def _validate_state(self, state: jax.Array | Sequence[float]) -> jax.Array:
        z = jnp.asarray(state, dtype=jnp.float32)
        if z.shape != (self.constraints.state_dim,):
            raise ValueError(f"state must have shape ({self.constraints.state_dim},)")
        return z

    def _result(
        self,
        state: jax.Array,
        initial_error: float,
        final_error: float,
        steps: int,
    ) -> PiNetProjectionResult:
        return PiNetProjectionResult(
            case_name=self.constraints.name,
            projected_state=tuple(float(value) for value in jax.device_get(state)),
            initial_projection_error=initial_error,
            projection_error=final_error,
            convergence_steps=int(steps),
            converged=final_error <= self.tolerance,
        )


def _as_matrix(
    value: Sequence[Sequence[float]] | None,
    state_dim: int,
    name: str,
) -> jax.Array:
    if value is None:
        return jnp.zeros((0, state_dim), dtype=jnp.float32)
    matrix = jnp.asarray(value, dtype=jnp.float32)
    if matrix.ndim != 2 or matrix.shape[1] != state_dim:
        raise ValueError(f"{name} must have shape (n_constraints, {state_dim})")
    return matrix


def _as_vector(value: Sequence[float] | None, size: int, name: str) -> jax.Array:
    if value is None:
        return jnp.zeros((size,), dtype=jnp.float32)
    vector = jnp.asarray(value, dtype=jnp.float32)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return vector


def _as_float(value: jax.Array) -> float:
    return float(jax.device_get(value))


def build_toy_projection_cases() -> tuple[ProjectionCase, ...]:
    """Return deterministic hard-constraint systems for Exp 1662.

    Spec refs: SCENARIO-KONA-039.
    """

    simplex = LinearConstraintSet.from_arrays(
        state_dim=3,
        equality_matrix=[[1.0, 1.0, 1.0]],
        equality_target=[1.0],
        inequality_matrix=[
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        inequality_bound=[0.0, 0.0, 0.0],
        name="simplex_nonnegative_sum_one",
    )
    affine_box = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, -1.0]],
        equality_target=[0.2],
        inequality_matrix=[
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        inequality_bound=[0.55, 0.45, 0.0, 0.0],
        name="affine_box_band",
    )
    budget = LinearConstraintSet.from_arrays(
        state_dim=2,
        inequality_matrix=[
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        inequality_bound=[0.4, 0.0, 0.0],
        name="nonnegative_budget_halfspace",
    )
    return (
        ProjectionCase("simplex_nonnegative_sum_one", (1.4, -0.6, 0.7), simplex),
        ProjectionCase("affine_box_band", (0.9, -0.4), affine_box),
        ProjectionCase("nonnegative_budget_halfspace", (0.7, 0.5), budget),
    )


def _differentiability_check(case: ProjectionCase, max_steps: int) -> bool:
    layer = DouglasRachfordPiNetLayer(case.constraints, max_steps=max_steps)

    def loss(state: jax.Array) -> jax.Array:
        projected = layer.project_vector(state)
        return jnp.sum(projected * projected)

    grad = jax.grad(loss)(jnp.asarray(case.start, dtype=jnp.float32))
    return bool(jnp.all(jnp.isfinite(grad)))


def evaluate_toy_projection_cases(
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> JsonDict:
    """Run the deterministic projection cases and return aggregate metrics.

    Spec refs: SCENARIO-KONA-039.
    """

    case_rows = []
    projection_errors = []
    convergence_steps = []
    differentiable_projection = True
    for case in build_toy_projection_cases():
        layer = DouglasRachfordPiNetLayer(
            case.constraints,
            max_steps=max_steps,
            tolerance=tolerance,
        )
        result = layer.project(case.start)
        case_rows.append(result.to_json())
        projection_errors.append(result.projection_error)
        convergence_steps.append(result.convergence_steps)
        differentiable_projection = differentiable_projection and _differentiability_check(
            case, max_steps
        )

    return {
        "cases_evaluated": len(case_rows),
        "projection_error": max(projection_errors),
        "convergence_steps": max(convergence_steps),
        "differentiable_projection": bool(differentiable_projection),
        "case_results": case_rows,
    }


def build_experiment_1662_artifact(
    *,
    tests_run: Sequence[str] = (),
    max_steps: int = DEFAULT_MAX_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> JsonDict:
    """Build the stable Exp 1662 artifact payload.

    Spec refs: REQ-KONA-039, SCENARIO-KONA-039.
    """

    summary = evaluate_toy_projection_cases(max_steps=max_steps, tolerance=tolerance)
    complete = (
        float(summary["projection_error"]) <= tolerance
        and int(summary["convergence_steps"]) <= max_steps
        and bool(summary["differentiable_projection"])
    )
    now = datetime.now(UTC)
    return {
        "status": "complete" if complete else "blocked",
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_date": now.strftime("%Y-%m-%d"),
        "title": "Pi-net differentiable Douglas-Rachford projection layer",
        "spec_refs": list(SPEC_REFS),
        "module_path": MODULE_PATH,
        "artifact_path": ARTIFACT_PATH,
        "projection_error": float(summary["projection_error"]),
        "convergence_steps": int(summary["convergence_steps"]),
        "cases_evaluated": int(summary["cases_evaluated"]),
        "differentiable_projection": bool(summary["differentiable_projection"]),
        "max_steps": int(max_steps),
        "tolerance": float(tolerance),
        "tests_run": list(tests_run),
        "case_results": summary["case_results"],
        "honest_verdict": (
            "pinet_layer_projection_complete" if complete else "pinet_layer_projection_blocked"
        ),
    }


def write_experiment_1662_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> JsonDict:
    """Write `results/experiment_1662_pinet_layer.json` and return it."""

    artifact = build_experiment_1662_artifact()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


__all__ = [
    "ARTIFACT_PATH",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_RESULT_PATH",
    "DEFAULT_TOLERANCE",
    "DouglasRachfordPiNetLayer",
    "EXPERIMENT_ID",
    "LinearConstraintSet",
    "MODULE_PATH",
    "PiNetProjectionResult",
    "ProjectionCase",
    "REQUIRED_ARTIFACT_FIELDS",
    "SCHEMA",
    "SPEC_REFS",
    "build_experiment_1662_artifact",
    "build_toy_projection_cases",
    "evaluate_toy_projection_cases",
    "write_experiment_1662_artifact",
]
