"""Deterministic SKM-style projection for toy linear certificate constraints.

The helper here is intentionally small: it projects a point onto a finite set
of linear half-spaces by repeatedly selecting the most violated row and applying
the closed-form Euclidean projection onto that row's boundary. Equalities are
encoded as paired inequalities, which keeps the update rule identical for
``Ax <= b`` and ``Ax = b`` certificate checks.

This is a verifier-side smoke baseline only. It trains no model, calls no LLM,
and makes no hardware-correctness claim.

Spec: REQ-VERIFY-1474, SCENARIO-VERIFY-1474.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from carnot.models.ising import IsingConfig, IsingModel
from carnot.verify.z3_math_verifier import Z3MathVerifier


HELPER_PATH = "python/carnot/verify/skm_projection.py"
TOLERANCE = 1e-9
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "toy_cases_evaluated",
    "zero_violation_projection",
    "max_constraint_violation",
    "baseline_verifier_agreement",
    "projection_iterations_p50",
    "projection_iterations_p95",
    "helper_path",
    "tests_run",
    "honest_verdict",
}


@dataclass(frozen=True)
class LinearConstraintSystem:
    """A small linear system in half-space form.

    ``matrix @ x <= bounds`` is the only representation used by the projection
    routine. Equality rows are transformed by ``from_constraints`` into both
    ``a @ x <= c`` and ``-a @ x <= -c`` so the SKM update does not need a
    separate equality branch.

    Spec: REQ-VERIFY-1474.
    """

    matrix: np.ndarray
    bounds: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def from_constraints(
        cls,
        *,
        less_equal: Sequence[tuple[str, Sequence[float], float]] = (),
        equalities: Sequence[tuple[str, Sequence[float], float]] = (),
    ) -> "LinearConstraintSystem":
        """Build a half-space system from inequalities and exact equalities.

        Args:
            less_equal: Named rows of the form ``(name, coefficients, bound)``
                representing ``coefficients @ x <= bound``.
            equalities: Named rows of the form ``(name, coefficients, value)``
                representing ``coefficients @ x == value``.

        Returns:
            A system containing the original inequality rows plus two
            half-space rows for every equality.

        Spec: REQ-VERIFY-1474.
        """
        rows: list[list[float]] = []
        bounds: list[float] = []
        names: list[str] = []

        for name, row, bound in less_equal:
            rows.append([float(value) for value in row])
            bounds.append(float(bound))
            names.append(name)

        for name, row, bound in equalities:
            positive = [float(value) for value in row]
            negative = [-value for value in positive]
            rows.append(positive)
            bounds.append(float(bound))
            names.append(f"{name}<={float(bound)}")
            rows.append(negative)
            bounds.append(-float(bound))
            names.append(f"{name}>={float(bound)}")

        return cls(
            matrix=np.asarray(rows, dtype=float),
            bounds=np.asarray(bounds, dtype=float),
            names=tuple(names),
        )

    def violations(self, vector: Sequence[float]) -> np.ndarray:
        """Return per-row positive violation magnitudes for ``vector``.

        Spec: REQ-VERIFY-1474.
        """
        x = np.asarray(vector, dtype=float)
        return np.maximum(self.matrix @ x - self.bounds, 0.0)

    def max_violation(self, vector: Sequence[float]) -> float:
        """Return the largest positive half-space violation.

        Spec: REQ-VERIFY-1474.
        """
        return float(np.max(self.violations(vector)))

    def is_satisfied(self, vector: Sequence[float], tolerance: float = TOLERANCE) -> bool:
        """Return whether every linear row is satisfied within tolerance.

        Spec: REQ-VERIFY-1474.
        """
        return self.max_violation(vector) <= tolerance


@dataclass(frozen=True)
class ProjectionResult:
    """Output of the bounded SKM projection loop.

    Spec: REQ-VERIFY-1474.
    """

    vector: tuple[float, ...]
    iterations: int
    converged: bool
    max_constraint_violation: float
    violation_history: tuple[float, ...]


@dataclass(frozen=True)
class ToyLinearCase:
    """One deterministic toy linear certificate system.

    Spec: SCENARIO-VERIFY-1474.
    """

    name: str
    system: LinearConstraintSystem
    start: tuple[float, ...]
    expected_solution: tuple[float, ...]
    z3_template: str

    def z3_statement(self, vector: Sequence[float]) -> str:
        """Format projected values as an arithmetic claim for Z3MathVerifier.

        Spec: SCENARIO-VERIFY-1474.
        """
        values = {f"x{idx}": f"{float(value):.12g}" for idx, value in enumerate(vector)}
        return self.z3_template.format(**values)


@dataclass(frozen=True)
class ProjectionSmokeSummary:
    """Aggregate result for the Exp 1474 projection smoke.

    Spec: REQ-VERIFY-1474, SCENARIO-VERIFY-1474.
    """

    toy_cases_evaluated: int
    zero_violation_projection: bool
    max_constraint_violation: float
    baseline_verifier_agreement: bool
    projection_iterations_p50: float
    projection_iterations_p95: float
    case_results: tuple[dict[str, object], ...]

    def to_artifact(self, tests_run: Sequence[str]) -> dict[str, object]:
        """Convert the summary to the required experiment artifact schema.

        Spec: REQ-VERIFY-1474.
        """
        honest_verdict = (
            "complete_cpu_only_zero_violation_baseline_agreement"
            if self.zero_violation_projection and self.baseline_verifier_agreement
            else "complete_cpu_only_projection_or_baseline_disagreement"
        )
        return {
            "status": "complete",
            "toy_cases_evaluated": self.toy_cases_evaluated,
            "zero_violation_projection": self.zero_violation_projection,
            "max_constraint_violation": self.max_constraint_violation,
            "baseline_verifier_agreement": self.baseline_verifier_agreement,
            "projection_iterations_p50": self.projection_iterations_p50,
            "projection_iterations_p95": self.projection_iterations_p95,
            "helper_path": HELPER_PATH,
            "tests_run": list(tests_run),
            "honest_verdict": honest_verdict,
            "case_results": list(self.case_results),
        }


def project_skm(
    system: LinearConstraintSystem,
    start: Sequence[float],
    *,
    max_iterations: int = 64,
    tolerance: float = TOLERANCE,
) -> ProjectionResult:
    """Project ``start`` onto ``system`` with a Motzkin/Kaczmarz half-space loop.

    Each iteration picks the currently most violated row ``a @ x <= b`` and
    moves directly to the closest point on that row's boundary:
    ``x <- x - ((a @ x - b) / ||a||^2) * a``.

    Spec: REQ-VERIFY-1474.
    """
    x = np.asarray(start, dtype=float).copy()
    history: list[float] = []
    iterations = 0

    while iterations <= max_iterations:
        violations = system.violations(x)
        row_index = int(np.argmax(violations))
        max_violation = float(violations[row_index])
        history.append(max_violation)

        if max_violation <= tolerance:
            return ProjectionResult(
                vector=tuple(float(value) for value in x),
                iterations=iterations,
                converged=True,
                max_constraint_violation=max_violation,
                violation_history=tuple(history),
            )

        if iterations == max_iterations:
            return ProjectionResult(
                vector=tuple(float(value) for value in x),
                iterations=iterations,
                converged=False,
                max_constraint_violation=max_violation,
                violation_history=tuple(history),
            )

        row = system.matrix[row_index]
        overshoot = float(row @ x - system.bounds[row_index])
        x = x - (overshoot / float(row @ row)) * row
        iterations += 1

    raise AssertionError("bounded SKM loop exited unexpectedly")  # pragma: no cover


def build_toy_linear_cases() -> tuple[ToyLinearCase, ...]:
    """Return the fixed toy arithmetic/certificate systems for Exp 1474.

    Spec: SCENARIO-VERIFY-1474.
    """
    return (
        ToyLinearCase(
            name="single_answer_equality",
            system=LinearConstraintSystem.from_constraints(
                less_equal=[("answer_nonnegative", [-1.0], 0.0)],
                equalities=[("answer", [1.0], 7.0)],
            ),
            start=(10.0,),
            expected_solution=(7.0,),
            z3_template="3 + 4 = {x0}",
        ),
        ToyLinearCase(
            name="sum_certificate",
            system=LinearConstraintSystem.from_constraints(
                equalities=[
                    ("lhs", [1.0, 0.0, 0.0], 2.0),
                    ("rhs", [0.0, 1.0, 0.0], 3.0),
                    ("total", [0.0, 0.0, 1.0], 5.0),
                ],
            ),
            start=(0.0, 0.0, 20.0),
            expected_solution=(2.0, 3.0, 5.0),
            z3_template="{x0} + {x1} = {x2}",
        ),
        ToyLinearCase(
            name="acceptance_budget_certificate",
            system=LinearConstraintSystem.from_constraints(
                less_equal=[("accepted_plus_violation_within_budget", [1.0, 1.0, -1.0], 0.0)],
                equalities=[
                    ("accepted", [1.0, 0.0, 0.0], 1.0),
                    ("violation", [0.0, 1.0, 0.0], 0.0),
                    ("budget", [0.0, 0.0, 1.0], 1.0),
                ],
            ),
            start=(0.0, 3.0, 0.0),
            expected_solution=(1.0, 0.0, 1.0),
            z3_template="{x0} + {x1} <= {x2}",
        ),
    )


def ising_linear_verdict(
    system: LinearConstraintSystem,
    vector: Sequence[float],
    *,
    tolerance: float = TOLERANCE,
) -> bool:
    """Check row satisfaction through the existing Ising energy primitive.

    A satisfied row maps to spin ``+1`` and a violated row maps to ``-1``. A
    zero-coupling, positive-bias Ising model reaches its minimum energy exactly
    when every row spin is ``+1``.

    Spec: SCENARIO-VERIFY-1474.
    """
    spins = np.where(system.violations(vector) <= tolerance, 1.0, -1.0)
    model = IsingModel(IsingConfig(input_dim=int(spins.size), coupling_init="zeros"))
    model.bias = jnp.ones(int(spins.size))
    energy = float(model.energy(jnp.asarray(spins)))
    return energy <= -float(spins.size) + tolerance


def evaluate_toy_cases(
    cases: Sequence[ToyLinearCase] | None = None,
    *,
    max_iterations: int = 64,
    tolerance: float = TOLERANCE,
) -> ProjectionSmokeSummary:
    """Run projection and Carnot/Z3/Ising baseline agreement on toy cases.

    Spec: SCENARIO-VERIFY-1474.
    """
    selected_cases = tuple(cases or build_toy_linear_cases())
    verifier = Z3MathVerifier()
    case_results: list[dict[str, object]] = []
    max_violations: list[float] = []
    iterations: list[int] = []
    zero_violation_flags: list[bool] = []
    agreement_flags: list[bool] = []

    for case in selected_cases:
        projection = project_skm(
            case.system,
            case.start,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        carnot_verdict = case.system.is_satisfied(projection.vector, tolerance)
        z3_statement = case.z3_statement(projection.vector)
        z3_verdict = verifier.score(z3_statement) == 0.0
        ising_verdict = ising_linear_verdict(case.system, projection.vector, tolerance=tolerance)
        projection_verdict = projection.converged and projection.max_constraint_violation <= tolerance
        baseline_agreement = projection_verdict == carnot_verdict == z3_verdict == ising_verdict

        max_violations.append(projection.max_constraint_violation)
        iterations.append(projection.iterations)
        zero_violation_flags.append(projection_verdict)
        agreement_flags.append(baseline_agreement)
        case_results.append(
            {
                "case": case.name,
                "projected_vector": list(projection.vector),
                "expected_solution": list(case.expected_solution),
                "projection_iterations": projection.iterations,
                "max_constraint_violation": projection.max_constraint_violation,
                "carnot_verdict": carnot_verdict,
                "z3_verdict": z3_verdict,
                "ising_verdict": ising_verdict,
                "z3_statement": z3_statement,
            }
        )

    return ProjectionSmokeSummary(
        toy_cases_evaluated=len(selected_cases),
        zero_violation_projection=all(zero_violation_flags),
        max_constraint_violation=float(max(max_violations)),
        baseline_verifier_agreement=all(agreement_flags),
        projection_iterations_p50=_percentile(iterations, 50.0),
        projection_iterations_p95=_percentile(iterations, 95.0),
        case_results=tuple(case_results),
    )


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = floor(rank)
    upper = ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "HELPER_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "TOLERANCE",
    "LinearConstraintSystem",
    "ProjectionResult",
    "ProjectionSmokeSummary",
    "ToyLinearCase",
    "build_toy_linear_cases",
    "evaluate_toy_cases",
    "ising_linear_verdict",
    "project_skm",
]
