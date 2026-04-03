"""Tests for Ising (small tier) model — JAX implementation.

Spec coverage: REQ-TIER-001, REQ-TIER-004, REQ-TIER-005,
               SCENARIO-TIER-001, SCENARIO-TIER-004, SCENARIO-TIER-005, SCENARIO-TIER-006
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.core.energy import EnergyFunction
from carnot.models.ising import IsingConfig, IsingModel


class TestIsingConfig:
    """Tests for REQ-TIER-005: Ising configuration."""

    def test_default_config(self) -> None:
        """REQ-TIER-005: Default Ising config has sensible values."""
        config = IsingConfig()
        assert config.input_dim == 784
        assert config.hidden_dim is None

    def test_validation_zero_dim(self) -> None:
        """SCENARIO-TIER-006: input_dim=0 raises error."""
        config = IsingConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be > 0"):
            config.validate()

    def test_validation_zero_hidden(self) -> None:
        """SCENARIO-TIER-006: hidden_dim=0 raises error."""
        config = IsingConfig(input_dim=10, hidden_dim=0)
        with pytest.raises(ValueError, match="hidden_dim must be > 0"):
            config.validate()


class TestIsingModel:
    """Tests for REQ-TIER-001: Ising model implementation."""

    def test_creation(self) -> None:
        """REQ-TIER-001: Model creates with valid config."""
        model = IsingModel(IsingConfig(input_dim=10))
        assert model.input_dim == 10
        assert model.coupling.shape == (10, 10)
        assert model.bias.shape == (10,)

    def test_coupling_symmetric(self) -> None:
        """REQ-TIER-001: Coupling matrix is symmetric."""
        model = IsingModel(IsingConfig(input_dim=5))
        assert jnp.allclose(model.coupling, model.coupling.T, atol=1e-10)

    def test_energy_finite(self) -> None:
        """SCENARIO-TIER-001: Energy is finite for binary input."""
        model = IsingModel(IsingConfig(input_dim=10))
        x = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-002: Batch energy computation."""
        model = IsingModel(IsingConfig(input_dim=3))
        xs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        energies = model.energy_batch(xs)
        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))

    def test_gradient_finite_difference(self) -> None:
        """SCENARIO-CORE-003: Gradient consistent with finite-difference."""
        model = IsingModel(IsingConfig(input_dim=5))
        x = jnp.array([0.5, -0.3, 0.8, -0.1, 0.4])
        grad = model.grad_energy(x)
        eps = 1e-4
        for i in range(5):
            x_p = x.at[i].add(eps)
            x_m = x.at[i].add(-eps)
            fd = (model.energy(x_p) - model.energy(x_m)) / (2 * eps)
            assert jnp.abs(grad[i] - fd) < 1e-3

    def test_memory_footprint(self) -> None:
        """SCENARIO-TIER-005: Default Ising model < 10MB."""
        model = IsingModel(IsingConfig())
        mb = model.parameter_memory_bytes() / (1024 * 1024)
        assert mb < 10.0, f"Memory footprint should be < 10MB, got {mb:.2f}MB"

    def test_interface_conformance(self) -> None:
        """SCENARIO-TIER-004: Works through EnergyFunction protocol."""
        model = IsingModel(IsingConfig(input_dim=3))
        assert isinstance(model, EnergyFunction)

        # Can be used generically
        def compute_energy(ef: EnergyFunction, x: jax.Array) -> jax.Array:
            return ef.energy(x)

        x = jnp.array([1.0, 2.0, 3.0])
        e = compute_energy(model, x)
        assert jnp.isfinite(e)
