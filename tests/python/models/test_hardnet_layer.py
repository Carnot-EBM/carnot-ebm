"""Tests for the HardNet-style differentiable enforcement layer.

Spec references: REQ-HARDNET-2086, SCENARIO-HARDNET-2086.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from carnot.models.hardnet_layer import HardNetLayer

def test_hardnet_projection_bounds():
    """Verify that outputs are clamped to the given bounds (REQ-HARDNET-2086-1)."""
    layer = HardNetLayer(lower_bound=-1.0, upper_bound=1.0)
    
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    projected = layer(x)
    
    expected = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert jnp.allclose(projected, expected), f"Expected {expected}, got {projected}"

def test_hardnet_differentiability():
    """Verify the layer is fully JAX-differentiable (REQ-HARDNET-2086-2)."""
    layer = HardNetLayer(lower_bound=-1.0, upper_bound=1.0)
    
    def loss_fn(x):
        projected = layer(x)
        return jnp.sum(projected ** 2)
    
    grad_fn = jax.grad(loss_fn)
    x = jnp.array([-2.0, 0.5, 2.0])
    grads = grad_fn(x)
    
    # Forward projected: [-1.0, 0.5, 1.0]
    # d(loss)/d(projected) = 2 * projected = [-2.0, 1.0, 2.0]
    # With STE, d(projected)/d(x) = 1.0 everywhere.
    # Therefore grads should be [-2.0, 1.0, 2.0].
    expected_grads = jnp.array([-2.0, 1.0, 2.0])
    assert jnp.allclose(grads, expected_grads), f"Expected {expected_grads}, got {grads}"

def test_hardnet_optax_compatibility():
    """Verify compatibility with Optax (REQ-HARDNET-2086-3)."""
    layer = HardNetLayer(lower_bound=0.0, upper_bound=1.0)
    
    x = jnp.array([5.0])
    target = jnp.array([0.5])
    
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(x)
    
    def loss_fn(x_val):
        projected = layer(x_val)
        return jnp.sum((projected - target)**2)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(x)
    
    # Gradient shouldn't be 0 even though x is far outside bounds (5.0 > 1.0).
    assert not jnp.allclose(grads, 0.0), "Gradient vanished, STE might be broken."
    
    updates, opt_state = optimizer.update(grads, opt_state, x)
    x_new = optax.apply_updates(x, updates)
    
    assert x_new.shape == x.shape
    # Ensure parameter was actually updated
    assert not jnp.allclose(x, x_new)
