"""Tests for denoising score matching training.

Spec coverage: REQ-TRAIN-002, SCENARIO-TRAIN-002
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from carnot.core.energy import AutoGradMixin, EnergyFunction
from carnot.training import dsm_loss, dsm_loss_stochastic
from carnot.training.score_matching import HasGradEnergy


class QuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * ||x||^2. grad_energy(x) = x."""

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)


class ScaledQuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * scale * ||x||^2. grad_energy(x) = scale * x."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * self.scale * jnp.sum(x**2)


class TestHasGradEnergyProtocol:
    """Tests for the HasGradEnergy protocol.

    Spec: REQ-TRAIN-002
    """

    def test_protocol_check(self) -> None:
        """REQ-TRAIN-002: QuadraticEnergy satisfies HasGradEnergy."""
        model = QuadraticEnergy()
        assert isinstance(model, HasGradEnergy)

    def test_protocol_callable_not_instance(self) -> None:
        """REQ-TRAIN-002: A plain function is not HasGradEnergy."""

        def f(x: jax.Array) -> jax.Array:
            return x

        assert not isinstance(f, HasGradEnergy)


class TestDsmLoss:
    """Tests for the dsm_loss function.

    Spec: REQ-TRAIN-002, SCENARIO-TRAIN-002
    """

    def test_loss_finite(self) -> None:
        """SCENARIO-TRAIN-002: DSM loss is finite for valid inputs."""
        model = QuadraticEnergy()
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
        noise = jnp.array([[0.1, -0.1], [0.05, 0.05], [-0.1, 0.0]])
        sigma = 0.5

        loss = dsm_loss(model, batch, noise, sigma)
        assert jnp.isfinite(loss)

    def test_loss_non_negative(self) -> None:
        """REQ-TRAIN-002: DSM loss is non-negative (sum of squares)."""
        model = QuadraticEnergy()
        batch = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        noise = jnp.array([[0.1, -0.1], [-0.05, 0.05]])
        sigma = 0.5

        loss = dsm_loss(model, batch, noise, sigma)
        assert loss >= 0.0

    def test_better_model_lower_loss(self) -> None:
        """SCENARIO-TRAIN-002: Model with score closer to true score has lower loss.

        Data at origin. Noisy point = noise n.
        grad_energy(n) = scale * n. Target = -n / sigma^2.
        diff = scale*n - (-n/sigma^2) = (scale + 1/sigma^2)*n.
        Loss = (scale + 1/sigma^2)^2 * mean(||n||^2).
        Minimized at scale = -1/sigma^2.
        """
        batch = jnp.zeros((3, 2))
        noise = jnp.array([[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]])
        sigma = 1.0

        good_model = ScaledQuadraticEnergy(scale=0.0)
        bad_model = ScaledQuadraticEnergy(scale=5.0)

        good_loss = dsm_loss(good_model, batch, noise, sigma)
        bad_loss = dsm_loss(bad_model, batch, noise, sigma)

        assert good_loss < bad_loss

    def test_perfect_score_zero_loss(self) -> None:
        """SCENARIO-TRAIN-002: Perfect denoiser score gives zero DSM loss.

        For data at origin, noisy = n. Target = -n/sigma^2.
        If grad_energy(x) = -x/sigma^2 (i.e., scale = -1/sigma^2),
        then grad_energy(n) = -n/sigma^2 = target. Loss = 0.
        """
        sigma = 1.0
        perfect_model = ScaledQuadraticEnergy(scale=-1.0 / (sigma * sigma))
        batch = jnp.zeros((2, 2))
        noise = jnp.array([[0.3, -0.2], [-0.1, 0.4]])

        loss = dsm_loss(perfect_model, batch, noise, sigma)
        assert loss < 1e-10

    def test_callable_grad_fn(self) -> None:
        """REQ-TRAIN-002: dsm_loss accepts a raw callable for grad_energy."""
        batch = jnp.zeros((2, 2))
        noise = jnp.array([[0.3, -0.2], [-0.1, 0.4]])
        sigma = 1.0

        # Perfect denoiser as a plain function
        def perfect_grad(x: jax.Array) -> jax.Array:
            return -x / (sigma * sigma)

        loss = dsm_loss(perfect_grad, batch, noise, sigma)
        assert loss < 1e-10

    def test_jit_compilable(self) -> None:
        """REQ-TRAIN-002: dsm_loss is JIT-compilable when using a callable."""
        sigma = 1.0

        def grad_fn(x: jax.Array) -> jax.Array:
            return x  # identity

        @jax.jit
        def compute_loss(batch: jax.Array, noise: jax.Array) -> jax.Array:
            return dsm_loss(grad_fn, batch, noise, sigma)

        batch = jnp.array([[1.0, 2.0]])
        noise = jnp.array([[0.1, -0.1]])
        loss = compute_loss(batch, noise)
        assert jnp.isfinite(loss)


class TestDsmLossStochastic:
    """Tests for the dsm_loss_stochastic function.

    Spec: REQ-TRAIN-002, SCENARIO-TRAIN-002
    """

    def test_stochastic_finite(self) -> None:
        """SCENARIO-TRAIN-002: Stochastic DSM loss is finite."""
        model = QuadraticEnergy()
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        key = jax.random.PRNGKey(42)
        sigma = 0.5

        loss = dsm_loss_stochastic(model, batch, sigma, key)
        assert jnp.isfinite(loss)

    def test_stochastic_deterministic_with_same_key(self) -> None:
        """REQ-TRAIN-002: Same PRNG key gives same loss."""
        model = QuadraticEnergy()
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        key = jax.random.PRNGKey(0)
        sigma = 0.5

        loss1 = dsm_loss_stochastic(model, batch, sigma, key)
        loss2 = dsm_loss_stochastic(model, batch, sigma, key)
        assert jnp.allclose(loss1, loss2)

    def test_stochastic_different_key_different_loss(self) -> None:
        """REQ-TRAIN-002: Different PRNG keys give different losses."""
        model = QuadraticEnergy()
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        sigma = 0.5

        loss1 = dsm_loss_stochastic(model, batch, sigma, jax.random.PRNGKey(0))
        loss2 = dsm_loss_stochastic(model, batch, sigma, jax.random.PRNGKey(1))
        # Very unlikely to be exactly equal
        assert not jnp.allclose(loss1, loss2, atol=1e-8)

    def test_stochastic_callable(self) -> None:
        """REQ-TRAIN-002: Stochastic DSM loss works with callable."""
        sigma = 1.0

        def grad_fn(x: jax.Array) -> jax.Array:
            return x

        key = jax.random.PRNGKey(42)
        batch = jnp.array([[0.0, 0.0]])

        loss = dsm_loss_stochastic(grad_fn, batch, sigma, key)
        assert jnp.isfinite(loss)


class TestDsmImports:
    """Tests for module imports.

    Spec: REQ-TRAIN-002
    """

    def test_import_from_training(self) -> None:
        """REQ-TRAIN-002: dsm_loss importable from carnot.training."""
        from carnot.training import dsm_loss as loss_fn
        from carnot.training import dsm_loss_stochastic as stoch_fn

        assert callable(loss_fn)
        assert callable(stoch_fn)
