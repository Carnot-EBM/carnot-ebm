"""Tests for Energy-Based Transformer (EBT) model — JAX implementation.

Spec coverage: REQ-EBT-001, REQ-EBT-002, REQ-EBT-003,
               SCENARIO-EBT-001, SCENARIO-EBT-002, SCENARIO-EBT-003
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from carnot.core.energy import EnergyFunction
from carnot.models.ebt import EBTConfig, EBTransformer, _gelu, _layer_norm


class TestEBTConfig:
    """Tests for REQ-EBT-001: EBT configuration."""

    def test_default_config(self) -> None:
        """REQ-EBT-001: sensible defaults."""
        config = EBTConfig()
        assert config.n_layers == 4
        assert config.d_model == 64
        assert config.n_heads == 4
        assert config.d_ff == 256
        assert config.vocab_size == 256
        assert config.max_seq_len == 128

    def test_custom_config(self) -> None:
        """REQ-EBT-001: custom architecture."""
        config = EBTConfig(
            n_layers=2, d_model=32, n_heads=2, d_ff=64, vocab_size=100, max_seq_len=16
        )
        assert config.n_layers == 2
        assert config.d_model == 32

    def test_validation_zero_layers(self) -> None:
        """SCENARIO-EBT-001: n_layers=0 raises error."""
        config = EBTConfig(n_layers=0)
        with pytest.raises(ValueError, match="n_layers must be > 0"):
            config.validate()

    def test_validation_zero_d_model(self) -> None:
        """SCENARIO-EBT-001: d_model=0 raises error."""
        config = EBTConfig(d_model=0)
        with pytest.raises(ValueError, match="d_model must be > 0"):
            config.validate()

    def test_validation_zero_n_heads(self) -> None:
        """SCENARIO-EBT-001: n_heads=0 raises error."""
        config = EBTConfig(n_heads=0)
        with pytest.raises(ValueError, match="n_heads must be > 0"):
            config.validate()

    def test_validation_zero_d_ff(self) -> None:
        """SCENARIO-EBT-001: d_ff=0 raises error."""
        config = EBTConfig(d_ff=0)
        with pytest.raises(ValueError, match="d_ff must be > 0"):
            config.validate()

    def test_validation_zero_vocab(self) -> None:
        """SCENARIO-EBT-001: vocab_size=0 raises error."""
        config = EBTConfig(vocab_size=0)
        with pytest.raises(ValueError, match="vocab_size must be > 0"):
            config.validate()

    def test_validation_zero_max_seq_len(self) -> None:
        """SCENARIO-EBT-001: max_seq_len=0 raises error."""
        config = EBTConfig(max_seq_len=0)
        with pytest.raises(ValueError, match="max_seq_len must be > 0"):
            config.validate()

    def test_validation_d_model_not_divisible_by_n_heads(self) -> None:
        """SCENARIO-EBT-001: d_model not divisible by n_heads raises error."""
        config = EBTConfig(d_model=65, n_heads=4)
        with pytest.raises(ValueError, match="d_model.*must be divisible by.*n_heads"):
            config.validate()


class TestHelperFunctions:
    """Tests for helper functions used by EBT."""

    def test_gelu_zero(self) -> None:
        """GELU(0) should be 0."""
        assert jnp.isclose(_gelu(jnp.float32(0.0)), 0.0, atol=1e-6)

    def test_gelu_positive(self) -> None:
        """GELU of large positive is approximately identity."""
        x = jnp.float32(3.0)
        assert jnp.isclose(_gelu(x), x, atol=0.01)

    def test_layer_norm_output_shape(self) -> None:
        """Layer norm preserves shape."""
        x = jrandom.normal(jrandom.PRNGKey(0), (5, 8))
        gamma = jnp.ones(8)
        beta = jnp.zeros(8)
        out = _layer_norm(x, gamma, beta)
        assert out.shape == (5, 8)

    def test_layer_norm_zero_mean(self) -> None:
        """Layer norm output has approximately zero mean per row."""
        x = jrandom.normal(jrandom.PRNGKey(0), (3, 16))
        gamma = jnp.ones(16)
        beta = jnp.zeros(16)
        out = _layer_norm(x, gamma, beta)
        row_means = jnp.mean(out, axis=-1)
        assert jnp.allclose(row_means, 0.0, atol=1e-5)

    def test_layer_norm_unit_variance(self) -> None:
        """Layer norm output has approximately unit variance per row."""
        x = jrandom.normal(jrandom.PRNGKey(0), (3, 16))
        gamma = jnp.ones(16)
        beta = jnp.zeros(16)
        out = _layer_norm(x, gamma, beta)
        row_vars = jnp.var(out, axis=-1)
        assert jnp.allclose(row_vars, 1.0, atol=1e-4)


class TestEBTransformer:
    """Tests for REQ-EBT-002, REQ-EBT-003: EBT model implementation."""

    def _small_config(self) -> EBTConfig:
        """Return a small config for fast testing."""
        return EBTConfig(
            n_layers=2,
            d_model=16,
            n_heads=2,
            d_ff=32,
            vocab_size=50,
            max_seq_len=16,
        )

    def test_creation(self) -> None:
        """REQ-EBT-002: model creates with valid config."""
        config = self._small_config()
        model = EBTransformer(config)
        assert model.input_dim == 16
        assert len(model.layers) == 2

    def test_creation_default_key(self) -> None:
        """REQ-EBT-002: model creates with default PRNG key."""
        model = EBTransformer(self._small_config())
        tokens = jnp.array([1.0, 2.0, 3.0])
        assert jnp.isfinite(model.energy(tokens))

    def test_energy_finite(self) -> None:
        """SCENARIO-EBT-002: energy is finite for random token input."""
        model = EBTransformer(self._small_config(), key=jrandom.PRNGKey(42))
        tokens = jnp.array([1.0, 5.0, 23.0, 7.0, 42.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)
        assert e.shape == ()  # scalar

    def test_energy_single_token(self) -> None:
        """SCENARIO-EBT-002: energy works for a single token."""
        model = EBTransformer(self._small_config(), key=jrandom.PRNGKey(42))
        tokens = jnp.array([10.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)
        assert e.shape == ()

    def test_energy_max_seq_len(self) -> None:
        """SCENARIO-EBT-002: energy works at maximum sequence length."""
        config = self._small_config()
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        tokens = jnp.arange(config.max_seq_len, dtype=jnp.float32) % config.vocab_size
        e = model.energy(tokens)
        assert jnp.isfinite(e)

    def test_grad_energy_shape(self) -> None:
        """REQ-EBT-003: grad_energy returns correct shape (same as input)."""
        model = EBTransformer(self._small_config(), key=jrandom.PRNGKey(42))
        tokens = jnp.array([1.0, 5.0, 23.0, 7.0, 42.0])
        grad = model.grad_energy(tokens)
        assert grad.shape == tokens.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_energy_finite_difference(self) -> None:
        """REQ-EBT-003: gradient matches numerical finite difference."""
        model = EBTransformer(self._small_config(), key=jrandom.PRNGKey(42))
        tokens = jnp.array([1.0, 5.0, 23.0, 7.0, 42.0])
        grad = model.grad_energy(tokens)
        eps = 1e-3
        for i in range(len(tokens)):
            t_p = tokens.at[i].add(eps)
            t_m = tokens.at[i].add(-eps)
            fd = (model.energy(t_p) - model.energy(t_m)) / (2 * eps)
            assert jnp.abs(grad[i] - fd) < 0.1, (
                f"Gradient mismatch at index {i}: analytic={grad[i]}, fd={fd}"
            )

    def test_different_inputs_different_energies(self) -> None:
        """REQ-EBT-002: different token sequences give different energies."""
        model = EBTransformer(self._small_config(), key=jrandom.PRNGKey(42))
        tokens_a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tokens_b = jnp.array([10.0, 20.0, 30.0, 40.0, 49.0])
        e_a = model.energy(tokens_a)
        e_b = model.energy(tokens_b)
        # With random initialization, different inputs should give different energies
        assert not jnp.isclose(e_a, e_b, atol=1e-6), (
            f"Different inputs gave same energy: {e_a} vs {e_b}"
        )

    def test_energy_batch(self) -> None:
        """SCENARIO-EBT-002: batch energy computation works."""
        config = self._small_config()
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        # Batch of 3 sequences, each length 5
        batch = jnp.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0, 49.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ]
        )
        energies = model.energy_batch(batch)
        assert energies.shape == (3,)
        assert jnp.all(jnp.isfinite(energies))

    def test_interface_conformance(self) -> None:
        """REQ-EBT-002: model satisfies EnergyFunction protocol."""
        model = EBTransformer(self._small_config())
        assert isinstance(model, EnergyFunction)

        def compute_energy(ef: EnergyFunction, x: jax.Array) -> jax.Array:
            return ef.energy(x)

        tokens = jnp.array([1.0, 2.0, 3.0])
        e = compute_energy(model, tokens)
        assert jnp.isfinite(e)

    def test_token_clipping(self) -> None:
        """SCENARIO-EBT-002: out-of-range token IDs are clipped safely."""
        config = self._small_config()
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        # Token IDs beyond vocab_size should be clipped
        tokens = jnp.array([0.0, 999.0, -5.0, 25.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)

    def test_deep_architecture(self) -> None:
        """REQ-EBT-002: model with 6 layers works."""
        config = EBTConfig(
            n_layers=6,
            d_model=16,
            n_heads=2,
            d_ff=32,
            vocab_size=50,
            max_seq_len=16,
        )
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        assert len(model.layers) == 6
        tokens = jnp.array([1.0, 2.0, 3.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)

    def test_single_layer(self) -> None:
        """REQ-EBT-002: model with 1 layer works."""
        config = EBTConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            d_ff=32,
            vocab_size=50,
            max_seq_len=16,
        )
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        assert len(model.layers) == 1
        tokens = jnp.array([1.0, 2.0, 3.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)
        grad = model.grad_energy(tokens)
        assert jnp.all(jnp.isfinite(grad))

    def test_single_head(self) -> None:
        """REQ-EBT-002: model with 1 attention head works."""
        config = EBTConfig(
            n_layers=2,
            d_model=16,
            n_heads=1,
            d_ff=32,
            vocab_size=50,
            max_seq_len=16,
        )
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        tokens = jnp.array([1.0, 2.0, 3.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)

    def test_many_heads(self) -> None:
        """REQ-EBT-002: model with many attention heads works."""
        config = EBTConfig(
            n_layers=2,
            d_model=16,
            n_heads=8,
            d_ff=32,
            vocab_size=50,
            max_seq_len=16,
        )
        model = EBTransformer(config, key=jrandom.PRNGKey(42))
        tokens = jnp.array([1.0, 2.0, 3.0])
        e = model.energy(tokens)
        assert jnp.isfinite(e)

    def test_memory_footprint(self) -> None:
        """SCENARIO-EBT-003: memory footprint is reasonable for small model."""
        config = self._small_config()
        model = EBTransformer(config)
        mb = model.parameter_memory_bytes() / (1024 * 1024)
        # Small model should be well under 1MB
        assert mb < 1.0, f"Memory footprint should be < 1MB, got {mb:.4f}MB"
        # But should be positive
        assert model.parameter_memory_bytes() > 0

    def test_reproducibility(self) -> None:
        """REQ-EBT-002: same key produces same model and energy."""
        config = self._small_config()
        model_a = EBTransformer(config, key=jrandom.PRNGKey(42))
        model_b = EBTransformer(config, key=jrandom.PRNGKey(42))
        tokens = jnp.array([1.0, 5.0, 10.0])
        e_a = model_a.energy(tokens)
        e_b = model_b.energy(tokens)
        assert jnp.isclose(e_a, e_b)

    def test_different_keys_different_models(self) -> None:
        """REQ-EBT-002: different keys produce different models."""
        config = self._small_config()
        model_a = EBTransformer(config, key=jrandom.PRNGKey(42))
        model_b = EBTransformer(config, key=jrandom.PRNGKey(99))
        tokens = jnp.array([1.0, 5.0, 10.0])
        e_a = model_a.energy(tokens)
        e_b = model_b.energy(tokens)
        # Different random init should give different energies
        assert not jnp.isclose(e_a, e_b, atol=1e-6)
