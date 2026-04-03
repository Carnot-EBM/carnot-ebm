"""Tests for verifiable reasoning — constraint-based verification.

Spec coverage: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004,
               REQ-VERIFY-005, REQ-VERIFY-007,
               SCENARIO-VERIFY-003, SCENARIO-VERIFY-004, SCENARIO-VERIFY-006
"""

import jax
import jax.numpy as jnp
import pytest

from carnot.verify.constraint import (
    BaseConstraint,
    ComposedEnergy,
    repair,
)


class MinValueConstraint(BaseConstraint):
    """x[index] >= min_val. Energy = max(0, min_val - x[index])^2."""

    def __init__(self, name: str, index: int, min_val: float) -> None:
        self._name = name
        self._index = index
        self._min_val = min_val

    @property
    def name(self) -> str:
        return self._name

    def energy(self, x: jax.Array) -> jax.Array:
        violation = jnp.maximum(self._min_val - x[self._index], 0.0)
        return violation**2


class SumConstraint(BaseConstraint):
    """sum(x) == target. Energy = (sum(x) - target)^2."""

    def __init__(self, name: str, target: float) -> None:
        self._name = name
        self._target = target

    @property
    def name(self) -> str:
        return self._name

    def energy(self, x: jax.Array) -> jax.Array:
        return (jnp.sum(x) - self._target) ** 2


class TestConstraintTerm:
    """Tests for REQ-VERIFY-001."""

    def test_satisfied_returns_zero(self) -> None:
        """REQ-VERIFY-001: satisfied constraint returns ~0 energy."""
        c = MinValueConstraint("min", 0, 1.0)
        x = jnp.array([2.0, 0.0])
        assert float(c.energy(x)) < 1e-6
        assert c.is_satisfied(x)

    def test_violated_returns_positive(self) -> None:
        """REQ-VERIFY-001: violated constraint returns > 0 energy."""
        c = MinValueConstraint("min", 0, 1.0)
        x = jnp.array([0.5, 0.0])
        e = float(c.energy(x))
        assert e > 0.0
        assert not c.is_satisfied(x)
        assert abs(e - 0.25) < 1e-5  # (1.0 - 0.5)^2

    def test_auto_gradient(self) -> None:
        """REQ-VERIFY-001: auto-derived gradient via jax.grad."""
        c = SumConstraint("sum", 5.0)
        x = jnp.array([1.0, 2.0])
        grad = c.grad_energy(x)
        # d/dx [(sum(x)-5)^2] = 2*(sum(x)-5) * [1,1] = 2*(-2)*[1,1] = [-4,-4]
        assert jnp.allclose(grad, jnp.array([-4.0, -4.0]), atol=1e-4)


class TestComposedEnergy:
    """Tests for REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004."""

    def test_composition(self) -> None:
        """REQ-VERIFY-004: constraints compose independently."""
        composed = ComposedEnergy(input_dim=2)
        assert composed.num_constraints == 0
        composed.add_constraint(MinValueConstraint("a", 0, 0.0), 1.0)
        assert composed.num_constraints == 1
        composed.add_constraint(SumConstraint("b", 0.0), 2.0)
        assert composed.num_constraints == 2

    def test_weighted_sum(self) -> None:
        """SCENARIO-VERIFY-004: composed energy = sum of weighted terms."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(MinValueConstraint("min", 0, 1.0), 1.0)
        composed.add_constraint(SumConstraint("sum", 3.0), 2.0)

        x = jnp.array([0.5, 1.0])
        total = float(composed.energy(x))
        # min: (1-0.5)^2=0.25*1.0, sum: (1.5-3)^2=2.25*2.0=4.5, total=4.75
        assert abs(total - 4.75) < 0.01

    def test_decomposition_sums_to_total(self) -> None:
        """SCENARIO-VERIFY-003: decomposed sum = total energy."""
        composed = ComposedEnergy(input_dim=3)
        composed.add_constraint(MinValueConstraint("a", 0, 1.0), 1.0)
        composed.add_constraint(MinValueConstraint("b", 1, 2.0), 1.5)
        composed.add_constraint(SumConstraint("c", 6.0), 1.0)

        x = jnp.array([0.5, 1.0, 2.0])
        reports = composed.decompose(x)
        total = float(composed.energy(x))
        decomposed_sum = sum(r.weighted_energy for r in reports)
        assert abs(total - decomposed_sum) < 1e-4
        assert all(r.name for r in reports)

    def test_verify_all_satisfied(self) -> None:
        """REQ-VERIFY-003: VERIFIED when all constraints satisfied."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(MinValueConstraint("min", 0, 1.0), 1.0)
        composed.add_constraint(SumConstraint("sum", 5.0), 1.0)

        x = jnp.array([2.0, 3.0])
        result = composed.verify(x)
        assert result.is_verified()
        assert result.failing_constraints() == []

    def test_verify_with_violations(self) -> None:
        """REQ-VERIFY-003: VIOLATED with specific failing names."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(MinValueConstraint("min", 0, 5.0), 1.0)
        composed.add_constraint(SumConstraint("sum", 10.0), 1.0)

        x = jnp.array([1.0, 2.0])
        result = composed.verify(x)
        assert not result.is_verified()
        failing = result.failing_constraints()
        assert "min" in failing
        assert "sum" in failing


class TestRepair:
    """Tests for REQ-VERIFY-005."""

    def test_repair_reduces_energy(self) -> None:
        """REQ-VERIFY-005: repair reduces total energy."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(MinValueConstraint("a", 0, 3.0), 1.0)
        composed.add_constraint(MinValueConstraint("b", 1, 3.0), 1.0)

        x = jnp.array([0.0, 0.0])
        repaired, history = repair(composed, x, step_size=0.1, max_steps=200)

        initial_e = history[0].total_energy
        final_e = history[-1].total_energy
        assert final_e < initial_e

    def test_repair_preserves_satisfied(self) -> None:
        """REQ-VERIFY-005: satisfied constraints stay satisfied."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(MinValueConstraint("a", 0, 1.0), 1.0)
        composed.add_constraint(MinValueConstraint("b", 1, 5.0), 1.0)

        x = jnp.array([10.0, 0.0])  # a satisfied, b violated
        repaired, _ = repair(composed, x, step_size=0.1, max_steps=100)

        # x[0] should stay >= 1.0 (grad_violated_only ignores satisfied constraint)
        assert float(repaired[0]) >= 1.0
        assert float(repaired[1]) > 0.0


class TestComposedEnergyMethods:
    """Tests for ComposedEnergy utility methods."""

    def test_input_dim(self) -> None:
        """REQ-VERIFY-004: input_dim property works."""
        composed = ComposedEnergy(input_dim=5)
        assert composed.input_dim == 5

    def test_grad_energy(self) -> None:
        """REQ-VERIFY-001: grad_energy computes total gradient."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(SumConstraint("sum", 0.0), 1.0)
        x = jnp.array([1.0, 2.0])
        grad = composed.grad_energy(x)
        # d/dx [(sum(x))^2] = 2*sum(x)*[1,1] = 2*3*[1,1] = [6,6]
        assert jnp.allclose(grad, jnp.array([6.0, 6.0]), atol=0.1)

    def test_energy_batch(self) -> None:
        """REQ-VERIFY-001: energy_batch via vmap."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(SumConstraint("sum", 0.0), 1.0)
        xs = jnp.array([[1.0, 2.0], [0.0, 0.0]])
        energies = composed.energy_batch(xs)
        assert energies.shape == (2,)
        assert float(energies[0]) > 0.0  # sum=3, energy=9
        assert float(energies[1]) < 1e-6  # sum=0, energy=0


class TestDeterminism:
    """Tests for REQ-VERIFY-007."""

    def test_deterministic_energy(self) -> None:
        """SCENARIO-VERIFY-006: identical inputs produce identical results."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(SumConstraint("sum", 5.0), 1.0)

        x = jnp.array([1.0, 2.0])
        e1 = float(composed.energy(x))
        e2 = float(composed.energy(x))
        assert e1 == e2

    def test_deterministic_verification(self) -> None:
        """SCENARIO-VERIFY-006: verification is deterministic."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(SumConstraint("sum", 5.0), 1.0)

        x = jnp.array([1.0, 2.0])
        r1 = composed.verify(x)
        r2 = composed.verify(x)
        assert r1.total_energy == r2.total_energy
        assert r1.verdict.verified == r2.verdict.verified
