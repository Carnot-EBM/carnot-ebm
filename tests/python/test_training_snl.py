"""Tests for Self-Normalised Likelihood (SNL) training.

Spec coverage: REQ-TRAIN-004
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from carnot.core.energy import AutoGradMixin
from carnot.training import snl_loss, snl_loss_stochastic
from carnot.training.snl import HasEnergy


class QuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * ||x||^2. Models a standard Gaussian centered at the origin.

    This is the simplest nontrivial energy function: it assigns low energy
    to points near the origin and high energy to points far away. The
    corresponding probability distribution is a standard Gaussian.
    """

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)


class ConstantEnergy(AutoGradMixin):
    """E(x) = c for all x. A useless model that cannot discriminate.

    Returns the same energy for every input. Used to test that the SNL
    loss differentiates between good and bad models.
    """

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.float32(self._value)


class TestHasEnergyProtocol:
    """Tests for the HasEnergy protocol used by SNL.

    Spec: REQ-TRAIN-004
    """

    def test_protocol_check_model(self) -> None:
        """REQ-TRAIN-004: QuadraticEnergy satisfies HasEnergy."""
        model = QuadraticEnergy()
        assert isinstance(model, HasEnergy)

    def test_protocol_callable_not_instance(self) -> None:
        """REQ-TRAIN-004: A plain function is not HasEnergy."""

        def f(x: jax.Array) -> jax.Array:
            return jnp.sum(x**2)

        assert not isinstance(f, HasEnergy)


class TestSnlLoss:
    """Tests for the snl_loss function.

    Spec: REQ-TRAIN-004
    """

    def test_loss_finite(self) -> None:
        """REQ-TRAIN-004: SNL loss is finite for valid inputs."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.15]])
        proposals = jnp.array([[3.0, 2.0], [-2.5, 1.5], [1.0, -3.0]])
        log_z = jnp.float32(0.0)

        loss = snl_loss(model, data, proposals, log_z)
        assert jnp.isfinite(loss), f"SNL loss should be finite, got {loss}"

    def test_loss_decreases_when_log_z_optimized(self) -> None:
        """REQ-TRAIN-004: Loss decreases when log_z moves toward true value.

        For a quadratic energy E = 0.5||x||^2, the true partition function
        is Z = (2*pi)^(d/2), so log(Z) = d/2 * log(2*pi). For d=2,
        log(Z) = log(2*pi) ~ 1.838.
        """
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.0, 0.0], [-0.1, 0.05]])
        proposals = jnp.array([[2.0, 1.0], [-1.0, 2.0], [0.5, -1.5]])

        # Start with a bad log_z estimate
        bad_log_z = jnp.float32(10.0)
        # Use a better estimate closer to true value log(2*pi) ~ 1.838
        good_log_z = jnp.float32(1.8)

        loss_bad = snl_loss(model, data, proposals, bad_log_z)
        loss_good = snl_loss(model, data, proposals, good_log_z)

        # A better log_z should give a lower or comparable loss
        assert loss_good < loss_bad, (
            f"Better log_z should give lower loss: {loss_good} < {loss_bad}"
        )

    def test_callable_energy_fn(self) -> None:
        """REQ-TRAIN-004: snl_loss accepts a raw callable for energy."""
        data = jnp.array([[0.0, 0.0]])
        proposals = jnp.array([[10.0, 10.0]])
        log_z = jnp.float32(0.0)

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        loss = snl_loss(quad_energy, data, proposals, log_z)
        assert jnp.isfinite(loss)

    def test_has_energy_protocol(self) -> None:
        """REQ-TRAIN-004: snl_loss works with HasEnergy protocol objects."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2]])
        proposals = jnp.array([[3.0, 2.0]])
        log_z = jnp.float32(0.0)

        loss = snl_loss(model, data, proposals, log_z)
        assert jnp.isfinite(loss)

    def test_different_batch_sizes(self) -> None:
        """REQ-TRAIN-004: Data and proposal batches can have different sizes."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2]])  # 1 data sample
        proposals = jnp.array([[5.0, 5.0], [3.0, 4.0], [-2.0, 6.0]])  # 3 proposals
        log_z = jnp.float32(0.0)

        loss = snl_loss(model, data, proposals, log_z)
        assert jnp.isfinite(loss)

    def test_jit_compilable(self) -> None:
        """REQ-TRAIN-004: snl_loss is JIT-compilable when using a callable."""

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        @jax.jit
        def compute_loss(data: jax.Array, proposals: jax.Array, log_z: jax.Array) -> jax.Array:
            return snl_loss(quad_energy, data, proposals, log_z)

        data = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        proposals = jnp.array([[5.0, 5.0], [-3.0, 4.0]])
        log_z = jnp.float32(0.0)
        loss = compute_loss(data, proposals, log_z)
        assert jnp.isfinite(loss)

    def test_gradient_wrt_log_z(self) -> None:
        """REQ-TRAIN-004: Loss is differentiable w.r.t. log_z."""

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        data = jnp.array([[0.1, 0.2], [0.0, 0.0]])
        proposals = jnp.array([[2.0, 1.0], [-1.0, 2.0]])
        log_z = jnp.float32(0.0)

        grad_fn = jax.grad(lambda lz: snl_loss(quad_energy, data, proposals, lz))
        grad_val = grad_fn(log_z)
        assert jnp.isfinite(grad_val), "Gradient w.r.t. log_z should be finite"


class TestSnlLossStochastic:
    """Tests for the snl_loss_stochastic function.

    Spec: REQ-TRAIN-004
    """

    def test_stochastic_finite(self) -> None:
        """REQ-TRAIN-004: Stochastic SNL loss is finite."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        key = jax.random.PRNGKey(42)
        noise_scale = 5.0
        log_z = jnp.float32(0.0)

        loss = snl_loss_stochastic(model, data, noise_scale, log_z, key)
        assert jnp.isfinite(loss)

    def test_stochastic_deterministic_same_key(self) -> None:
        """REQ-TRAIN-004: Same PRNG key gives same loss."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        key = jax.random.PRNGKey(0)
        noise_scale = 5.0
        log_z = jnp.float32(0.0)

        loss1 = snl_loss_stochastic(model, data, noise_scale, log_z, key)
        loss2 = snl_loss_stochastic(model, data, noise_scale, log_z, key)
        assert jnp.allclose(loss1, loss2)

    def test_stochastic_different_key_different_loss(self) -> None:
        """REQ-TRAIN-004: Different PRNG keys give different losses."""
        model = QuadraticEnergy()
        data = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        noise_scale = 5.0
        log_z = jnp.float32(0.0)

        loss1 = snl_loss_stochastic(model, data, noise_scale, log_z, jax.random.PRNGKey(0))
        loss2 = snl_loss_stochastic(model, data, noise_scale, log_z, jax.random.PRNGKey(1))
        assert not jnp.allclose(loss1, loss2, atol=1e-8)

    def test_stochastic_callable(self) -> None:
        """REQ-TRAIN-004: Stochastic SNL loss works with callable."""

        def quad_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(42)
        data = jnp.array([[0.0, 0.0]])
        noise_scale = 5.0
        log_z = jnp.float32(0.0)

        loss = snl_loss_stochastic(quad_energy, data, noise_scale, log_z, key)
        assert jnp.isfinite(loss)


class TestSnlImports:
    """Tests for module imports.

    Spec: REQ-TRAIN-004
    """

    def test_import_from_training(self) -> None:
        """REQ-TRAIN-004: snl_loss importable from carnot.training."""
        from carnot.training import snl_loss as loss_fn
        from carnot.training import snl_loss_stochastic as stoch_fn

        assert callable(loss_fn)
        assert callable(stoch_fn)

    def test_import_from_snl_module(self) -> None:
        """REQ-TRAIN-004: snl_loss importable from carnot.training.snl."""
        from carnot.training.snl import snl_loss, snl_loss_stochastic

        assert callable(snl_loss)
        assert callable(snl_loss_stochastic)
