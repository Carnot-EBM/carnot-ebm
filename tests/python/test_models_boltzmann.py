"""Tests for Boltzmann (large tier) model — JAX implementation.

Spec coverage: REQ-TIER-003, REQ-TIER-004, REQ-TIER-005,
               SCENARIO-TIER-003, SCENARIO-TIER-004, SCENARIO-TIER-006
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.core.energy import EnergyFunction
from carnot.models.boltzmann import BoltzmannConfig, BoltzmannModel


class TestBoltzmannConfig:
    """Tests for REQ-TIER-005: Boltzmann configuration."""

    def test_default_config(self) -> None:
        """REQ-TIER-005: sensible defaults."""
        config = BoltzmannConfig()
        assert config.input_dim == 784
        assert config.hidden_dims == [1024, 512, 256, 128]
        assert config.residual is True
        assert config.num_heads == 4

    def test_custom_config(self) -> None:
        """REQ-TIER-005: custom architecture."""
        config = BoltzmannConfig(input_dim=10, hidden_dims=[8, 6, 4], num_heads=2)
        assert config.input_dim == 10
        assert config.hidden_dims == [8, 6, 4]

    def test_validation_zero_dim(self) -> None:
        """SCENARIO-TIER-006: input_dim=0 raises error."""
        config = BoltzmannConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be > 0"):
            config.validate()

    def test_validation_empty_hidden(self) -> None:
        """SCENARIO-TIER-006: empty hidden_dims raises error."""
        config = BoltzmannConfig(input_dim=10, hidden_dims=[])
        with pytest.raises(ValueError, match="hidden_dims must have at least one layer"):
            config.validate()

    def test_validation_zero_hidden(self) -> None:
        """SCENARIO-TIER-006: zero in hidden_dims raises error."""
        config = BoltzmannConfig(input_dim=10, hidden_dims=[8, 0, 4])
        with pytest.raises(ValueError, match="all hidden_dims must be > 0"):
            config.validate()

    def test_validation_zero_heads(self) -> None:
        """SCENARIO-TIER-006: num_heads=0 raises error."""
        config = BoltzmannConfig(input_dim=10, hidden_dims=[8], num_heads=0)
        with pytest.raises(ValueError, match="num_heads must be > 0"):
            config.validate()


class TestBoltzmannModel:
    """Tests for REQ-TIER-003: Boltzmann model implementation."""

    def test_creation(self) -> None:
        """REQ-TIER-003: model creates with valid config."""
        model = BoltzmannModel(BoltzmannConfig(input_dim=10, hidden_dims=[8, 6, 4]))
        assert model.input_dim == 10
        assert len(model.blocks) == 2  # two residual blocks: 8→6, 6→4

    def test_energy_finite(self) -> None:
        """SCENARIO-TIER-003: energy is finite for random input."""
        model = BoltzmannModel(
            BoltzmannConfig(input_dim=10, hidden_dims=[8, 6, 4]),
            key=jrandom.PRNGKey(42),
        )
        x = jrandom.normal(jrandom.PRNGKey(0), (10,))
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-002: batch energy computation."""
        model = BoltzmannModel(BoltzmannConfig(input_dim=5, hidden_dims=[4, 3]))
        xs = jrandom.normal(jrandom.PRNGKey(0), (4, 5))
        energies = model.energy_batch(xs)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_gradient_finite(self) -> None:
        """SCENARIO-CORE-003: gradient is finite and correct shape."""
        model = BoltzmannModel(BoltzmannConfig(input_dim=5, hidden_dims=[4, 3]))
        x = jrandom.normal(jrandom.PRNGKey(0), (5,))
        grad = model.grad_energy(x)
        assert grad.shape == (5,)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_finite_difference(self) -> None:
        """SCENARIO-CORE-003: gradient matches numerical finite difference."""
        model = BoltzmannModel(
            BoltzmannConfig(input_dim=5, hidden_dims=[4, 3]),
            key=jrandom.PRNGKey(42),
        )
        x = jrandom.normal(jrandom.PRNGKey(0), (5,)) * 0.5
        grad = model.grad_energy(x)
        eps = 1e-4
        for i in range(5):
            x_p = x.at[i].add(eps)
            x_m = x.at[i].add(-eps)
            fd = (model.energy(x_p) - model.energy(x_m)) / (2 * eps)
            assert jnp.abs(grad[i] - fd) < 0.05, (
                f"Gradient mismatch at index {i}: analytic={grad[i]}, fd={fd}"
            )

    def test_interface_conformance(self) -> None:
        """SCENARIO-TIER-004: works through EnergyFunction protocol."""
        model = BoltzmannModel(BoltzmannConfig(input_dim=5, hidden_dims=[4, 3]))
        assert isinstance(model, EnergyFunction)

        def compute_energy(ef: EnergyFunction, x: jax.Array) -> jax.Array:
            return ef.energy(x)

        x = jnp.ones(5)
        e = compute_energy(model, x)
        assert jnp.isfinite(e)

    def test_no_residual(self) -> None:
        """REQ-TIER-003: model works without residual connections."""
        model = BoltzmannModel(
            BoltzmannConfig(input_dim=5, hidden_dims=[4, 3], residual=False),
        )
        x = jnp.ones(5)
        e = model.energy(x)
        assert jnp.isfinite(e)
        grad = model.grad_energy(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_same_dim_residual(self) -> None:
        """REQ-TIER-003: residual works when all hidden dims are equal (no projection)."""
        model = BoltzmannModel(
            BoltzmannConfig(input_dim=5, hidden_dims=[4, 4, 4]),
        )
        x = jnp.ones(5)
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_deep_architecture(self) -> None:
        """REQ-TIER-003: deep model (4+ residual blocks) works."""
        model = BoltzmannModel(
            BoltzmannConfig(input_dim=10, hidden_dims=[8, 6, 5, 4, 3]),
        )
        assert len(model.blocks) == 4
        x = jrandom.normal(jrandom.PRNGKey(0), (10,))
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_default_key(self) -> None:
        """REQ-TIER-003: model creates with default PRNG key."""
        model = BoltzmannModel(BoltzmannConfig(input_dim=5, hidden_dims=[4, 3]))
        x = jnp.ones(5)
        assert jnp.isfinite(model.energy(x))
