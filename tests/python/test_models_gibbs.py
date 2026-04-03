"""Tests for Gibbs (medium tier) model — JAX implementation.

Spec coverage: REQ-TIER-002, REQ-TIER-004, REQ-TIER-005,
               SCENARIO-TIER-002, SCENARIO-TIER-004, SCENARIO-TIER-005, SCENARIO-TIER-006
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from carnot.core.energy import EnergyFunction
from carnot.models.gibbs import GibbsConfig, GibbsModel


class TestGibbsConfig:
    """Tests for REQ-TIER-005: Gibbs configuration."""

    def test_default_config(self) -> None:
        """REQ-TIER-005: sensible defaults."""
        config = GibbsConfig()
        assert config.input_dim == 784
        assert config.hidden_dims == [512, 256]
        assert config.activation == "silu"
        assert config.dropout == 0.0

    def test_custom_config(self) -> None:
        """REQ-TIER-005: custom architecture."""
        config = GibbsConfig(input_dim=10, hidden_dims=[8, 6], activation="relu")
        assert config.input_dim == 10
        assert config.hidden_dims == [8, 6]
        assert config.activation == "relu"

    def test_validation_zero_dim(self) -> None:
        """SCENARIO-TIER-006: input_dim=0 raises error."""
        config = GibbsConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be > 0"):
            config.validate()

    def test_validation_empty_hidden(self) -> None:
        """SCENARIO-TIER-006: empty hidden_dims raises error."""
        config = GibbsConfig(input_dim=10, hidden_dims=[])
        with pytest.raises(ValueError, match="hidden_dims must have at least one layer"):
            config.validate()

    def test_validation_zero_hidden(self) -> None:
        """SCENARIO-TIER-006: zero in hidden_dims raises error."""
        config = GibbsConfig(input_dim=10, hidden_dims=[8, 0, 4])
        with pytest.raises(ValueError, match="all hidden_dims must be > 0"):
            config.validate()

    def test_validation_bad_activation(self) -> None:
        """SCENARIO-TIER-006: unknown activation raises error."""
        config = GibbsConfig(input_dim=10, activation="gelu")
        with pytest.raises(ValueError, match="Unknown activation"):
            config.validate()

    def test_validation_bad_dropout(self) -> None:
        """SCENARIO-TIER-006: dropout >= 1.0 raises error."""
        config = GibbsConfig(input_dim=10, dropout=1.0)
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()


class TestGibbsModel:
    """Tests for REQ-TIER-002: Gibbs model implementation."""

    def test_creation(self) -> None:
        """REQ-TIER-002: model creates with valid config."""
        model = GibbsModel(GibbsConfig(input_dim=10, hidden_dims=[8, 4]))
        assert model.input_dim == 10
        assert len(model.layers) == 2  # two hidden layers

    def test_energy_finite(self) -> None:
        """SCENARIO-TIER-002: energy is finite for random input."""
        model = GibbsModel(
            GibbsConfig(input_dim=10, hidden_dims=[8, 4]),
            key=jrandom.PRNGKey(42),
        )
        x = jrandom.normal(jrandom.PRNGKey(0), (10,))
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-002: batch energy computation."""
        model = GibbsModel(GibbsConfig(input_dim=5, hidden_dims=[4, 3]))
        xs = jrandom.normal(jrandom.PRNGKey(0), (4, 5))
        energies = model.energy_batch(xs)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_gradient_finite(self) -> None:
        """SCENARIO-CORE-003: gradient is finite and correct shape."""
        model = GibbsModel(GibbsConfig(input_dim=5, hidden_dims=[4, 3]))
        x = jrandom.normal(jrandom.PRNGKey(0), (5,))
        grad = model.grad_energy(x)
        assert grad.shape == (5,)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_finite_difference_silu(self) -> None:
        """SCENARIO-CORE-003: SiLU gradient matches numerical finite difference."""
        model = GibbsModel(
            GibbsConfig(input_dim=5, hidden_dims=[4, 3], activation="silu"),
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
                f"SiLU gradient mismatch at index {i}: analytic={grad[i]}, fd={fd}"
            )

    def test_gradient_finite_difference_relu(self) -> None:
        """SCENARIO-CORE-003: ReLU gradient matches numerical finite difference."""
        model = GibbsModel(
            GibbsConfig(input_dim=5, hidden_dims=[4, 3], activation="relu"),
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
                f"ReLU gradient mismatch at index {i}: analytic={grad[i]}, fd={fd}"
            )

    def test_gradient_finite_difference_tanh(self) -> None:
        """SCENARIO-CORE-003: Tanh gradient matches numerical finite difference."""
        model = GibbsModel(
            GibbsConfig(input_dim=5, hidden_dims=[4, 3], activation="tanh"),
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
                f"Tanh gradient mismatch at index {i}: analytic={grad[i]}, fd={fd}"
            )

    def test_interface_conformance(self) -> None:
        """SCENARIO-TIER-004: works through EnergyFunction protocol."""
        model = GibbsModel(GibbsConfig(input_dim=5, hidden_dims=[4, 3]))
        assert isinstance(model, EnergyFunction)

        def compute_energy(ef: EnergyFunction, x: jax.Array) -> jax.Array:
            return ef.energy(x)

        x = jnp.ones(5)
        e = compute_energy(model, x)
        assert jnp.isfinite(e)

    def test_deep_architecture(self) -> None:
        """REQ-TIER-002: model with 4 hidden layers works."""
        model = GibbsModel(
            GibbsConfig(input_dim=10, hidden_dims=[8, 6, 5, 4]),
        )
        assert len(model.layers) == 4
        x = jrandom.normal(jrandom.PRNGKey(0), (10,))
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_default_key(self) -> None:
        """REQ-TIER-002: model creates with default PRNG key."""
        model = GibbsModel(GibbsConfig(input_dim=5, hidden_dims=[4, 3]))
        x = jnp.ones(5)
        assert jnp.isfinite(model.energy(x))

    def test_memory_footprint(self) -> None:
        """SCENARIO-TIER-005: memory footprint is reasonable."""
        model = GibbsModel(GibbsConfig(input_dim=100, hidden_dims=[64, 32]))
        mb = model.parameter_memory_bytes() / (1024 * 1024)
        # 100*64 + 64 + 64*32 + 32 + 32 + 1 = ~8.5K params * 4 bytes ≈ 34KB
        assert mb < 1.0, f"Memory footprint should be < 1MB for small model, got {mb:.4f}MB"

    def test_unknown_activation_at_runtime(self) -> None:
        """SCENARIO-TIER-006: unknown activation raises during forward pass."""
        # Bypass config validation to test runtime activation dispatch
        config = GibbsConfig(input_dim=5, hidden_dims=[4])
        config.activation = "gelu"  # sneak past __post_init__
        model = GibbsModel.__new__(GibbsModel)
        model.config = config
        model.layers = [(jnp.ones((4, 5)), jnp.zeros(4))]
        model.output_weight = jnp.zeros(4)
        model.output_bias = 0.0
        with pytest.raises(ValueError, match="Unknown activation"):
            model.energy(jnp.ones(5))

    def test_single_hidden_layer(self) -> None:
        """REQ-TIER-002: model works with a single hidden layer."""
        model = GibbsModel(GibbsConfig(input_dim=5, hidden_dims=[4]))
        assert len(model.layers) == 1
        x = jnp.ones(5)
        e = model.energy(x)
        assert jnp.isfinite(e)
        grad = model.grad_energy(x)
        assert jnp.all(jnp.isfinite(grad))
