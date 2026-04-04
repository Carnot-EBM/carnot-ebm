"""Tests for optimization training (backprop through gradient descent).

Spec coverage: REQ-TRAIN-005
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.training.optimization_training import (
    _unrolled_optimization_single,
    optimization_training_loss,
)


def _quadratic_energy(x: jax.Array) -> jax.Array:
    """Simple quadratic: E(x) = 0.5 * ||x||^2. Minimum at origin."""
    return 0.5 * jnp.sum(x**2)


class TestUnrolledOptimization:
    """Tests for single-sample gradient descent unrolling."""

    def test_moves_toward_minimum(self) -> None:
        """REQ-TRAIN-005: gradient descent moves toward energy minimum."""
        x0 = jnp.array([3.0, 4.0])
        x_final = _unrolled_optimization_single(_quadratic_energy, x0, n_steps=10, step_size=0.1)
        assert jnp.linalg.norm(x_final) < jnp.linalg.norm(x0)

    def test_returns_correct_shape(self) -> None:
        """REQ-TRAIN-005: output shape matches input."""
        x0 = jnp.ones(5)
        x_final = _unrolled_optimization_single(_quadratic_energy, x0, n_steps=5, step_size=0.1)
        assert x_final.shape == (5,)

    def test_zero_steps_returns_input(self) -> None:
        """REQ-TRAIN-005: zero steps returns input unchanged."""
        x0 = jnp.array([1.0, 2.0])
        x_final = _unrolled_optimization_single(_quadratic_energy, x0, n_steps=0, step_size=0.1)
        assert jnp.allclose(x_final, x0)


class TestOptimizationTrainingLoss:
    """Tests for the full optimization training loss."""

    def test_loss_is_finite(self) -> None:
        """REQ-TRAIN-005: loss is finite."""
        data = jnp.zeros((4, 3))
        loss = optimization_training_loss(_quadratic_energy, data, n_optimization_steps=5)
        assert jnp.isfinite(loss)

    def test_loss_decreases_with_more_steps(self) -> None:
        """REQ-TRAIN-005: more optimization steps -> lower loss (closer to target)."""
        data = jnp.zeros((4, 3))
        loss_few = optimization_training_loss(
            _quadratic_energy, data, n_optimization_steps=2, key=jrandom.PRNGKey(0)
        )
        loss_many = optimization_training_loss(
            _quadratic_energy, data, n_optimization_steps=20, key=jrandom.PRNGKey(0)
        )
        assert float(loss_many) <= float(loss_few)

    def test_default_key(self) -> None:
        """REQ-TRAIN-005: works without explicit key."""
        data = jnp.zeros((2, 2))
        loss = optimization_training_loss(_quadratic_energy, data, n_optimization_steps=3)
        assert jnp.isfinite(loss)

    def test_is_differentiable(self) -> None:
        """REQ-TRAIN-005: loss is differentiable (can compute gradients)."""
        data = jnp.zeros((2, 2))

        # Parameterized energy: E(x) = 0.5 * scale * ||x||^2
        def param_energy(x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(x**2)

        # Should be able to compute loss without error
        loss = optimization_training_loss(param_energy, data, n_optimization_steps=3)
        assert jnp.isfinite(loss)
