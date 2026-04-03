"""Tests for Noise Contrastive Estimation (NCE) training.

Spec coverage: REQ-TRAIN-003
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from carnot.core.energy import AutoGradMixin, EnergyFunction
from carnot.training import nce_loss, nce_loss_stochastic
from carnot.training.nce import HasEnergy


class QuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * ||x||^2. Models a standard Gaussian centered at the origin."""

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)


class ConstantEnergy(AutoGradMixin):
    """E(x) = c for all x. A useless model that cannot discriminate."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        # Return a scalar that does not depend on x, but is still a JAX array.
        return jnp.float32(self._value)


class TestHasEnergyProtocol:
    """Tests for the HasEnergy protocol.

    Spec: REQ-TRAIN-003
    """

    def test_protocol_check_model(self) -> None:
        """REQ-TRAIN-003: QuadraticEnergy satisfies HasEnergy."""
        model = QuadraticEnergy()
        assert isinstance(model, HasEnergy)

    def test_protocol_callable_not_instance(self) -> None:
        """REQ-TRAIN-003: A plain function is not HasEnergy."""

        def f(x: jax.Array) -> jax.Array:
            return jnp.sum(x**2)

        assert not isinstance(f, HasEnergy)


class TestNceLoss:
    """Tests for the nce_loss function.

    Spec: REQ-TRAIN-003
    """

    def test_loss_finite(self) -> None:
        """REQ-TRAIN-003: NCE loss is finite for valid inputs."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.15]])
        noise = jnp.array([[3.0, 2.0], [-2.5, 1.5], [1.0, -3.0]])

        loss = nce_loss(model, data, noise)
        assert jnp.isfinite(loss), f"NCE loss should be finite, got {loss}"

    def test_loss_non_negative(self) -> None:
        """REQ-TRAIN-003: NCE loss is non-negative (sum of -log_sigmoid terms)."""
        model = QuadraticEnergy()
        data = jnp.array([[0.5, 0.5], [-0.5, -0.5]])
        noise = jnp.array([[5.0, 5.0], [-5.0, -5.0]])

        loss = nce_loss(model, data, noise)
        assert loss >= 0.0, f"NCE loss should be non-negative, got {loss}"

    def test_good_model_lower_loss(self) -> None:
        """REQ-TRAIN-003: A model that assigns low energy to data and high energy
        to noise should have lower NCE loss than a constant-energy model."""
        data_near_origin = jnp.array([[0.01, 0.02], [0.0, 0.0], [-0.01, 0.01]])
        noise_far_away = jnp.array([[10.0, 10.0], [-10.0, 8.0], [7.0, -9.0]])

        good_model = QuadraticEnergy()
        bad_model = ConstantEnergy(0.0)

        good_loss = nce_loss(good_model, data_near_origin, noise_far_away)
        bad_loss = nce_loss(bad_model, data_near_origin, noise_far_away)

        assert good_loss < bad_loss, (
            f"Good model should have lower NCE loss: {good_loss} < {bad_loss}"
        )

    def test_noise_high_energy(self) -> None:
        """REQ-TRAIN-003: Noise samples far from data should get high energy."""
        model = QuadraticEnergy()
        noise_far = jnp.array([[100.0, 100.0]])

        energy_val = model.energy(noise_far[0])
        assert energy_val > 1000.0, (
            f"Far noise should have high energy, got {energy_val}"
        )

    def test_well_separated_loss_near_ln2(self) -> None:
        """REQ-TRAIN-003: When data energy ~ 0 and noise energy >> 0,
        data term ≈ softplus(0) = ln(2) and noise term ≈ 0, so total ≈ ln(2)."""
        model = QuadraticEnergy()
        data_origin = jnp.array([[0.0, 0.0], [0.001, 0.0]])
        noise_far = jnp.array([[50.0, 50.0], [-50.0, 50.0]])

        loss = nce_loss(model, data_origin, noise_far)
        expected = jnp.log(2.0)
        assert jnp.abs(loss - expected) < 0.01, (
            f"Loss should be ≈ ln(2) = {expected}, got {loss}"
        )

    def test_callable_energy_fn(self) -> None:
        """REQ-TRAIN-003: nce_loss accepts a raw callable for energy."""
        data = jnp.array([[0.0, 0.0]])
        noise = jnp.array([[10.0, 10.0]])

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        loss = nce_loss(quad_energy, data, noise)
        assert jnp.isfinite(loss)

    def test_jit_compilable(self) -> None:
        """REQ-TRAIN-003: nce_loss is JIT-compilable when using a callable."""

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        @jax.jit
        def compute_loss(data: jax.Array, noise: jax.Array) -> jax.Array:
            return nce_loss(quad_energy, data, noise)

        data = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        noise = jnp.array([[5.0, 5.0], [-3.0, 4.0]])
        loss = compute_loss(data, noise)
        assert jnp.isfinite(loss)

    def test_different_batch_sizes(self) -> None:
        """REQ-TRAIN-003: Data and noise batches can have different sizes."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2]])  # 1 data sample
        noise = jnp.array([[5.0, 5.0], [3.0, 4.0], [-2.0, 6.0]])  # 3 noise samples

        loss = nce_loss(model, data, noise)
        assert jnp.isfinite(loss)


class TestNceLossStochastic:
    """Tests for the nce_loss_stochastic function.

    Spec: REQ-TRAIN-003
    """

    def test_stochastic_finite(self) -> None:
        """REQ-TRAIN-003: Stochastic NCE loss is finite."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        key = jax.random.PRNGKey(42)
        noise_scale = 5.0

        loss = nce_loss_stochastic(model, data, noise_scale, key)
        assert jnp.isfinite(loss)

    def test_stochastic_deterministic_same_key(self) -> None:
        """REQ-TRAIN-003: Same PRNG key gives same loss."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        key = jax.random.PRNGKey(0)
        noise_scale = 5.0

        loss1 = nce_loss_stochastic(model, data, noise_scale, key)
        loss2 = nce_loss_stochastic(model, data, noise_scale, key)
        assert jnp.allclose(loss1, loss2)

    def test_stochastic_different_key_different_loss(self) -> None:
        """REQ-TRAIN-003: Different PRNG keys give different losses."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        noise_scale = 5.0

        loss1 = nce_loss_stochastic(model, data, noise_scale, jax.random.PRNGKey(0))
        loss2 = nce_loss_stochastic(model, data, noise_scale, jax.random.PRNGKey(1))
        assert not jnp.allclose(loss1, loss2, atol=1e-8)

    def test_stochastic_callable(self) -> None:
        """REQ-TRAIN-003: Stochastic NCE loss works with callable."""

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(42)
        data = jnp.array([[0.0, 0.0]])
        noise_scale = 5.0

        loss = nce_loss_stochastic(quad_energy, data, noise_scale, key)
        assert jnp.isfinite(loss)


class TestNceImports:
    """Tests for module imports.

    Spec: REQ-TRAIN-003
    """

    def test_import_from_training(self) -> None:
        """REQ-TRAIN-003: nce_loss importable from carnot.training."""
        from carnot.training import nce_loss as loss_fn
        from carnot.training import nce_loss_stochastic as stoch_fn

        assert callable(loss_fn)
        assert callable(stoch_fn)

    def test_import_from_nce_module(self) -> None:
        """REQ-TRAIN-003: nce_loss importable from carnot.training.nce."""
        from carnot.training.nce import HasEnergy, nce_loss, nce_loss_stochastic

        assert callable(nce_loss)
        assert callable(nce_loss_stochastic)
