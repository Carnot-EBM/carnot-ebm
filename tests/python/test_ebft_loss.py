"""Tests for EBFTLoss scaffold.

Spec coverage: REQ-TRAIN-007
"""

import jax
import jax.numpy as jnp
from carnot.training.ebft_loss import EBFTLoss

def test_ebft_loss_scaffold():
    """REQ-TRAIN-007: Test EBFTLoss with a dummy verifier."""
    def dummy_verifier(params: jnp.ndarray, seqs: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(params * seqs ** 2, axis=-1)
        
    loss_fn = EBFTLoss(dummy_verifier)
    params = jnp.array([1.0, 1.0])
    
    expert = jnp.array([[1.0, 0.0]]) # energy = 1.0
    rollout = jnp.array([[2.0, 0.0]]) # energy = 4.0
    
    loss = loss_fn(params, expert, rollout)
    assert jnp.allclose(loss, 1.0 - 4.0)

def test_ebft_loss_gradient():
    """REQ-TRAIN-007: Test gradient flow through EBFTLoss."""
    def dummy_verifier(params: jnp.ndarray, seqs: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(params * seqs, axis=-1)
        
    loss_fn = EBFTLoss(dummy_verifier)
    params = jnp.array([1.0, 2.0])
    
    expert = jnp.array([[1.0, 1.0]])
    rollout = jnp.array([[2.0, 2.0]])
    
    def compute_loss(p):
        return loss_fn(p, expert, rollout)
        
    grad = jax.grad(compute_loss)(params)
    
    # Expert energy: p0*1 + p1*1
    # Rollout energy: p0*2 + p1*2
    # Loss: (p0+p1) - (2p0+2p1) = -p0 - p1
    # Grad wrt p0, p1 = [-1.0, -1.0]
    assert jnp.allclose(grad, jnp.array([-1.0, -1.0]))

def test_ebft_loss_feature_matching_gradient():
    """REQ-TRAIN-007: Test gradient flow for the feature-matching ebft_loss."""
    from carnot.training.ebft_loss import ebft_loss
    
    # Simulate a small neural network predicting features
    def predict_features(params, inputs):
        return jnp.dot(inputs, params)
        
    params = jnp.array([[1.0, -1.0], [0.5, 0.5]])
    inputs = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    target_features = jnp.array([[2.0, 0.0], [2.0, 0.0]])
    
    def loss_fn(p):
        model_features = predict_features(p, inputs)
        return ebft_loss(model_features, target_features)
        
    loss_val = loss_fn(params)
    grad_val = jax.grad(loss_fn)(params)
    
    assert loss_val.shape == ()
    assert grad_val.shape == params.shape
    assert not jnp.all(grad_val == 0)
