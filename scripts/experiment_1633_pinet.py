#!/usr/bin/env python3
"""Exp 1633 Pi-Net-style hard projection for continuous constraints.

The prototype is intentionally small and CPU-only. It uses JAX operations to
project continuous latent states onto linear hard constraints before a later
continuous verifier tier would consume them.

Spec refs: REQ-KONA-037, SCENARIO-KONA-037.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

JsonDict = dict[str, Any]

EXPERIMENT_ID = 1633
SCHEMA = "carnot.phase3.pinet_projection.v1"
SPEC_REFS = ["REQ-KONA-037", "SCENARIO-KONA-037"]
DEFAULT_OUTPUT_PATH = Path("results/experiment_1633_pinet.json")
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_STEPS = 96
RIDGE = 1e-7
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "schema",
    "experiment_id",
    "spec_refs",
    "projection_error",
    "convergence_steps",
    "cases_evaluated",
    "differentiable_projection",
    "honest_verdict",
)


@dataclass(frozen=True)
class ContinuousConstraintSystem:
    """Linear hard constraints for a continuous latent state.

    The convention is `equality_matrix @ z == equality_target` and
    `inequality_matrix @ z <= inequality_bound`.

    Spec refs: REQ-KONA-037.
    """

    state_dim: int
    equality_matrix: jax.Array
    equality_target: jax.Array
    inequality_matrix: jax.Array
    inequality_bound: jax.Array
    name: str

    @classmethod
    def from_arrays(
        cls,
        *,
        state_dim: int,
        equality_matrix: Sequence[Sequence[float]] | None = None,
        equality_target: Sequence[float] | None = None,
        inequality_matrix: Sequence[Sequence[float]] | None = None,
        inequality_bound: Sequence[float] | None = None,
        name: str = "continuous_constraints",
    ) -> "ContinuousConstraintSystem":
        """Build and validate a continuous linear constraint system."""

        if state_dim < 1:
            raise ValueError("state_dim must be positive")  # pragma: no cover

        eq = _as_matrix(equality_matrix, state_dim, "equality_matrix")
        eq_target = _as_vector(
            equality_target,
            eq.shape[0],
            "equality_target",
            default_zeros=True,
        )
        ineq = _as_matrix(inequality_matrix, state_dim, "inequality_matrix")
        ineq_bound = _as_vector(
            inequality_bound,
            ineq.shape[0],
            "inequality_bound",
            default_zeros=True,
        )
        if eq.shape[0] + ineq.shape[0] == 0:
            raise ValueError("at least one hard constraint is required")  # pragma: no cover
        return cls(
            state_dim=state_dim,
            equality_matrix=eq,
            equality_target=eq_target,
            inequality_matrix=ineq,
            inequality_bound=ineq_bound,
            name=name,
        )

    def projection_error(self, state: jax.Array) -> jax.Array:
        """Return the max equality residual or positive inequality violation."""

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
    """One deterministic toy case for the Exp 1633 projection smoke."""

    name: str
    start: tuple[float, ...]
    system: ContinuousConstraintSystem


@dataclass(frozen=True)
class ProjectionResult:
    """Projection diagnostics for one continuous latent state."""

    case_name: str
    projected_state: tuple[float, ...]
    initial_projection_error: float
    projection_error: float
    convergence_steps: int
    converged: bool

    def to_json(self) -> JsonDict:
        """Return a JSON-safe row for artifact case results."""

        return {
            "case_name": self.case_name,
            "projected_state": list(self.projected_state),
            "initial_projection_error": self.initial_projection_error,
            "projection_error": self.projection_error,
            "convergence_steps": self.convergence_steps,
            "converged": self.converged,
        }


class PiNetProjectionLayer:
    """JAX projection layer for linear continuous hard constraints.

    The update alternates exact affine equality projection with half-space
    projections. This is the small deterministic analogue of a Pi-Net layer:
    the forward pass returns a state in the declared hard-constraint set.

    Spec refs: REQ-KONA-037, SCENARIO-KONA-037.
    """

    def __init__(
        self,
        system: ContinuousConstraintSystem,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")  # pragma: no cover
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")  # pragma: no cover
        self.system = system
        self.max_steps = max_steps
        self.tolerance = tolerance

    def project_vector(self, state: jax.Array) -> jax.Array:
        """Project a JAX state with a fixed differentiable unrolled loop."""

        z = self._validate_state(state)
        for _ in range(self.max_steps):
            z = self._projection_pass(z)
        return self._project_equalities(z)

    def project(self, state: jax.Array | Sequence[float]) -> ProjectionResult:
        """Project a numeric state and return residual/convergence diagnostics."""

        z = self._validate_state(state)
        initial_error = _as_float(self.system.projection_error(z))
        if initial_error <= self.tolerance:
            return self._result(z, initial_error, initial_error, 0)

        steps = self.max_steps
        final_error = initial_error
        for step in range(1, self.max_steps + 1):
            z = self._projection_pass(z)
            final_error = _as_float(self.system.projection_error(z))
            if final_error <= self.tolerance:
                steps = step
                break

        return self._result(z, initial_error, final_error, steps)

    def _projection_pass(self, state: jax.Array) -> jax.Array:
        z = self._project_equalities(state)
        for row_index in range(self.system.inequality_matrix.shape[0]):
            row = self.system.inequality_matrix[row_index]
            bound = self.system.inequality_bound[row_index]
            violation = jnp.dot(row, z) - bound
            step = jnp.maximum(violation, 0.0) / (jnp.dot(row, row) + RIDGE)
            z = z - step * row
        return self._project_equalities(z)

    def _project_equalities(self, state: jax.Array) -> jax.Array:
        if self.system.equality_matrix.shape[0] == 0:
            return state
        matrix = self.system.equality_matrix
        residual = matrix @ state - self.system.equality_target
        gram = matrix @ matrix.T + RIDGE * jnp.eye(matrix.shape[0], dtype=jnp.float32)
        correction = matrix.T @ jnp.linalg.solve(gram, residual)
        return state - correction

    def _validate_state(self, state: jax.Array | Sequence[float]) -> jax.Array:
        z = jnp.asarray(state, dtype=jnp.float32)
        if z.shape != (self.system.state_dim,):
            raise ValueError(f"state must have shape ({self.system.state_dim},)")
        return z

    def _result(
        self,
        state: jax.Array,
        initial_error: float,
        final_error: float,
        steps: int,
    ) -> ProjectionResult:
        return ProjectionResult(
            case_name=self.system.name,
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


def _as_vector(
    value: Sequence[float] | None,
    size: int,
    name: str,
    *,
    default_zeros: bool,
) -> jax.Array:
    if value is None and default_zeros:
        return jnp.zeros((size,), dtype=jnp.float32)
    vector = jnp.asarray(value, dtype=jnp.float32)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return vector


def _as_float(value: jax.Array) -> float:
    return float(jax.device_get(value))


def build_toy_cases() -> tuple[ProjectionCase, ...]:
    """Return deterministic continuous hard-constraint cases for Exp 1633."""

    simplex = ContinuousConstraintSystem.from_arrays(
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
    affine_box = ContinuousConstraintSystem.from_arrays(
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
    budget = ContinuousConstraintSystem.from_arrays(
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
    layer = PiNetProjectionLayer(case.system, max_steps=max_steps)

    def loss(state: jax.Array) -> jax.Array:
        projected = layer.project_vector(state)
        return jnp.sum(projected * projected)

    grad = jax.grad(loss)(jnp.asarray(case.start, dtype=jnp.float32))
    return bool(jnp.all(jnp.isfinite(grad)))


def evaluate_projection_cases(
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> JsonDict:
    """Run the deterministic projection smoke and return aggregate metrics."""

    case_results = []
    projection_errors = []
    convergence_steps = []
    all_differentiable = True
    for case in build_toy_cases():
        layer = PiNetProjectionLayer(case.system, max_steps=max_steps, tolerance=tolerance)
        result = layer.project(case.start)
        case_results.append(result.to_json())
        projection_errors.append(result.projection_error)
        convergence_steps.append(result.convergence_steps)
        all_differentiable = all_differentiable and _differentiability_check(case, max_steps)

    return {
        "cases_evaluated": len(case_results),
        "projection_error": max(projection_errors) if projection_errors else 0.0,
        "convergence_steps": max(convergence_steps) if convergence_steps else 0,
        "differentiable_projection": bool(all_differentiable),
        "case_results": case_results,
    }


def build_artifact(
    *,
    tests_run: Sequence[str],
    max_steps: int = DEFAULT_MAX_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> JsonDict:
    """Build the terminal Exp 1633 artifact."""

    summary = evaluate_projection_cases(max_steps=max_steps, tolerance=tolerance)
    complete = (
        summary["projection_error"] <= tolerance
        and summary["convergence_steps"] <= max_steps
        and summary["differentiable_projection"]
    )
    artifact: JsonDict = {
        "status": "complete" if complete else "blocked",
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": "1633_pinet_projection",
        "spec_refs": list(SPEC_REFS),
        "projection_error": float(summary["projection_error"]),
        "convergence_steps": int(summary["convergence_steps"]),
        "cases_evaluated": int(summary["cases_evaluated"]),
        "differentiable_projection": bool(summary["differentiable_projection"]),
        "max_steps": int(max_steps),
        "tolerance": float(tolerance),
        "tests_run": list(tests_run),
        "case_results": summary["case_results"],
        "honest_verdict": (
            "pinet_projection_satisfies_hard_constraints"
            if complete
            else "pinet_projection_blocked_or_not_differentiable"
        ),
    }
    validate_artifact(artifact, tolerance=tolerance)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Validate the fields required by REQ-KONA-037."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError("schema mismatch")  # pragma: no cover
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("experiment_id mismatch")  # pragma: no cover
    if artifact["spec_refs"] != SPEC_REFS:
        raise AssertionError("spec_refs mismatch")  # pragma: no cover
    if artifact["projection_error"] < 0.0:
        raise AssertionError("projection_error must be non-negative")  # pragma: no cover
    if artifact["convergence_steps"] < 0:
        raise AssertionError("convergence_steps must be non-negative")  # pragma: no cover
    if artifact["status"] == "complete" and artifact["projection_error"] > tolerance:
        raise AssertionError("projection_error exceeds tolerance for complete artifact")
    if artifact["status"] == "complete" and not artifact["differentiable_projection"]:
        raise AssertionError("differentiable_projection required for complete artifact")


def _write_json(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_experiment(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Run Exp 1633 and write `results/experiment_1633_pinet.json`."""

    artifact = build_artifact(tests_run=tests_run)
    return _write_json(Path(output_path), artifact)


def main() -> None:  # pragma: no cover
    run_experiment(output_path=DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":  # pragma: no cover
    main()
