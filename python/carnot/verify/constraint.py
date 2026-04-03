"""Constraint-based verifiable reasoning — JAX implementation.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004, REQ-VERIFY-005
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class ConstraintTerm(Protocol):
    """A single verifiable constraint, expressed as an energy term.

    Satisfied: energy(x) = 0. Violated: energy(x) > 0.

    Spec: REQ-VERIFY-001
    """

    @property
    def name(self) -> str:
        """Human-readable name for this constraint."""
        ...

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute constraint energy. Returns 0.0 if satisfied."""
        ...

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of constraint energy w.r.t. x."""
        ...

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold below which the constraint is considered satisfied."""
        ...

    def is_satisfied(self, x: jax.Array) -> bool:
        """Is this constraint satisfied for configuration x?"""
        ...


@dataclass
class ConstraintReport:
    """Report for a single constraint's evaluation.

    Spec: REQ-VERIFY-002, REQ-VERIFY-003
    """

    name: str
    energy: float
    weighted_energy: float
    satisfied: bool


@dataclass
class Verdict:
    """Verification verdict.

    Spec: REQ-VERIFY-003
    """

    verified: bool
    failing: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete verification result.

    Spec: REQ-VERIFY-003
    """

    total_energy: float
    constraints: list[ConstraintReport]
    verdict: Verdict

    def is_verified(self) -> bool:
        return self.verdict.verified

    def failing_constraints(self) -> list[str]:
        return self.verdict.failing


class BaseConstraint:
    """Base class for constraints with default threshold and is_satisfied.

    Spec: REQ-VERIFY-001
    """

    @property
    def satisfaction_threshold(self) -> float:
        return 1e-6

    def is_satisfied(self, x: jax.Array) -> bool:
        return float(self.energy(x)) <= self.satisfaction_threshold

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Auto-derive gradient via jax.grad.

        Spec: REQ-VERIFY-001
        """
        return jax.grad(self.energy)(x)


class ComposedEnergy:
    """An energy function composed of weighted constraint terms.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004
    """

    def __init__(self, input_dim: int) -> None:
        self._terms: list[tuple[ConstraintTerm, float]] = []
        self._input_dim = input_dim

    def add_constraint(self, term: ConstraintTerm, weight: float) -> None:
        """Add a constraint term with a weight.

        Spec: REQ-VERIFY-004
        """
        self._terms.append((term, weight))

    @property
    def num_constraints(self) -> int:
        return len(self._terms)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def energy(self, x: jax.Array) -> jax.Array:
        """Total weighted energy.

        Spec: REQ-VERIFY-001
        """
        total = jnp.float32(0.0)
        for term, weight in self._terms:
            total = total + weight * term.energy(x)
        return total

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Total gradient from all constraints."""
        return jax.grad(self.energy)(x)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Batched energy via vmap."""
        return jax.vmap(self.energy)(xs)

    def decompose(self, x: jax.Array) -> list[ConstraintReport]:
        """Per-constraint energy decomposition.

        Spec: REQ-VERIFY-002
        """
        reports = []
        for term, weight in self._terms:
            raw_energy = float(term.energy(x))
            reports.append(
                ConstraintReport(
                    name=term.name,
                    energy=raw_energy,
                    weighted_energy=weight * raw_energy,
                    satisfied=term.is_satisfied(x),
                )
            )
        return reports

    def verify(self, x: jax.Array) -> VerificationResult:
        """Produce a full verification result.

        Spec: REQ-VERIFY-003
        """
        reports = self.decompose(x)
        total_energy = sum(r.weighted_energy for r in reports)
        failing = [r.name for r in reports if not r.satisfied]
        verdict = Verdict(verified=len(failing) == 0, failing=failing)
        return VerificationResult(
            total_energy=total_energy,
            constraints=reports,
            verdict=verdict,
        )

    def grad_violated_only(self, x: jax.Array) -> jax.Array:
        """Gradient from only the violated constraints.

        Spec: REQ-VERIFY-005
        """
        grad = jnp.zeros(x.shape)
        for term, weight in self._terms:
            if not term.is_satisfied(x):
                grad = grad + weight * term.grad_energy(x)
        return grad


def repair(
    composed: ComposedEnergy,
    x: jax.Array,
    step_size: float,
    max_steps: int,
) -> tuple[jax.Array, list[VerificationResult]]:
    """Gradient-based repair: descend on violated constraints.

    Spec: REQ-VERIFY-005
    """
    history: list[VerificationResult] = []

    for _ in range(max_steps):
        result = composed.verify(x)
        history.append(result)
        if result.is_verified():
            break
        grad = composed.grad_violated_only(x)
        x = x - step_size * grad

    return x, history
