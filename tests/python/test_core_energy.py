"""Tests for core energy function protocol and mixins.

Spec coverage: REQ-CORE-002, SCENARIO-CORE-001, SCENARIO-CORE-002, SCENARIO-CORE-003
"""

import jax
import jax.numpy as jnp
import pytest

from carnot.core.energy import AutoGradMixin, EnergyFunction


class QuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * ||x||^2 — standard Gaussian energy."""

    @property
    def input_dim(self) -> int:
        return 0  # accepts any dimension

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)


class TestEnergyProtocol:
    """Tests for REQ-CORE-002: Energy Function Protocol."""

    def test_protocol_check(self) -> None:
        """REQ-CORE-002: QuadraticEnergy satisfies EnergyFunction protocol."""
        model = QuadraticEnergy()
        assert isinstance(model, EnergyFunction)

    def test_energy_single(self) -> None:
        """SCENARIO-CORE-001: Compute energy for single input."""
        model = QuadraticEnergy()
        x = jnp.array([1.0, 2.0, 3.0])
        e = model.energy(x)
        assert jnp.isfinite(e)
        assert jnp.abs(e - 7.0) < 1e-5

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-002: Compute batch energy via vmap."""
        model = QuadraticEnergy()
        xs = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        energies = model.energy_batch(xs)
        assert energies.shape == (2,)
        assert jnp.allclose(energies, jnp.array([0.5, 2.0]), atol=1e-5)
        assert jnp.all(jnp.isfinite(energies))

    def test_grad_energy_auto(self) -> None:
        """SCENARIO-CORE-003: Auto-derived gradient via jax.grad."""
        model = QuadraticEnergy()
        x = jnp.array([1.0, 2.0, 3.0])
        grad = model.grad_energy(x)
        assert grad.shape == x.shape
        # For E(x) = 0.5*||x||^2, grad = x
        assert jnp.allclose(grad, x, atol=1e-5)

    def test_grad_energy_finite_difference(self) -> None:
        """SCENARIO-CORE-003: Gradient consistent with finite-difference."""
        model = QuadraticEnergy()
        x = jnp.array([0.5, -0.3, 0.8])
        grad = model.grad_energy(x)
        eps = 1e-4
        for i in range(3):
            x_p = x.at[i].add(eps)
            x_m = x.at[i].add(-eps)
            fd = (model.energy(x_p) - model.energy(x_m)) / (2 * eps)
            assert jnp.abs(grad[i] - fd) < 1e-3, f"Gradient mismatch at index {i}"

    def test_autograd_mixin_base_energy_raises(self) -> None:
        """REQ-CORE-002: AutoGradMixin.energy raises NotImplementedError."""
        mixin = AutoGradMixin()
        with pytest.raises(NotImplementedError):
            mixin.energy(jnp.array([1.0]))
