import jax
import jax.numpy as jnp
import pytest
from carnot.models.caffnet_layer import project_affine, CAffNetLayer

def test_project_affine():
    """
    REQ-CAFFNET-3385-1: The layer SHALL project unconstrained logits onto a constrained affine subspace Ax = b.
    SCENARIO-CAFFNET-3385: Exact affine constraint satisfaction
    """
    # A x = b
    # Let's project onto x_1 + x_2 = 1
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    
    layer = CAffNetLayer(A, b)
    
    logits = jnp.array([2.0, 3.0])
    x_proj = layer.apply(logits)
    
    # Assert constraint satisfaction
    assert jnp.allclose(A @ x_proj, b, atol=1e-5)
    
    # The expected projection of [2.0, 3.0] onto x_1 + x_2 = 1:
    # x - A^T (A A^T)^-1 (A x - b)
    # A x - b = 5.0 - 1.0 = 4.0
    # A A^T = 2.0
    # (A A^T)^-1 = 0.5
    # x_proj = [2.0, 3.0] - [1.0, 1.0] * 0.5 * 4.0 = [2.0, 3.0] - [2.0, 2.0] = [0.0, 1.0]
    assert jnp.allclose(x_proj, jnp.array([0.0, 1.0]), atol=1e-5)

def test_caffnet_differentiable():
    """
    REQ-CAFFNET-3385-2: The projection SHALL be fully JAX-differentiable.
    """
    A = jnp.array([[1.0, -1.0]])
    b = jnp.array([0.0]) # x_1 = x_2
    
    layer = CAffNetLayer(A, b)
    
    def loss_fn(logits):
        x_proj = layer.apply(logits)
        return jnp.sum(x_proj ** 2)
        
    grad_fn = jax.grad(loss_fn)
    logits = jnp.array([3.0, 1.0])
    grads = grad_fn(logits)
    
    # Should not be None or NaNs
    assert grads is not None
    assert not jnp.any(jnp.isnan(grads))

def test_caffnet_assertion():
    """
    REQ-CAFFNET-3385-3: The layer SHALL assert 100% hard constraint satisfaction at inference time.
    """
    A = jnp.array([[2.0, 1.0]])
    b = jnp.array([5.0])
    
    layer = CAffNetLayer(A, b)
    logits = jnp.array([0.0, 0.0])
    
    x_proj = layer.apply(logits)
    assert jnp.allclose(A @ x_proj, b, atol=1e-4)

