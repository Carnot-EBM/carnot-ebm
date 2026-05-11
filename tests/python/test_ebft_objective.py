"""Tests for Energy-Based Fine-Tuning (EBFT) objective.

Spec coverage: REQ-TRAIN-007
"""

import jax
import jax.numpy as jnp

from carnot.training.ebft_objective import ebft_loss


def test_ebft_loss_value() -> None:
    """REQ-TRAIN-007: Objective calculation."""
    def dummy_energy(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x ** 2, axis=-1)
    
    expert = jnp.array([[1.0, 1.0], [0.0, 1.0]])  # energies: 2.0, 1.0 -> mean 1.5
    rollout = jnp.array([[2.0, 2.0]])             # energies: 8.0 -> mean 8.0
    
    loss = ebft_loss(dummy_energy, expert, rollout)
    
    assert jnp.allclose(loss, 1.5 - 8.0)


def test_ebft_gradient_flow() -> None:
    """REQ-TRAIN-007: Gradient flow and sequence-level feature matching."""
    expert = jnp.array([[1.0, 0.0]])
    rollout = jnp.array([[2.0, 0.0]])
    
    # Learnable parameter theta
    theta = jnp.array([0.5, 0.5])
    
    def energy_fn(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(params * x ** 2, axis=-1)
        
    def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        def bound_energy(x: jnp.ndarray) -> jnp.ndarray:
            return energy_fn(params, x)
        return ebft_loss(bound_energy, expert, rollout)
        
    grad = jax.grad(loss_fn)(theta)
    
    # Expected gradient:
    # E_expert: theta[0]*1^2 + theta[1]*0 = theta[0]
    # E_rollout: theta[0]*2^2 + theta[1]*0 = 4*theta[0]
    # Loss = theta[0] - 4*theta[0] = -3*theta[0]
    # Grad wrt theta[0] = -3.0
    # Grad wrt theta[1] = 0.0
    assert jnp.allclose(grad, jnp.array([-3.0, 0.0]))
