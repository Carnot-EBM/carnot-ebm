"""Tests for Muon-OGD."""
import pytest
import jax.numpy as jnp
from carnot.training.muon_ogd import MuonOGD, newton_schulz_matrix_sign

def test_newton_schulz_matrix_sign():
    """Test REQ-LEARN-1827 / SCENARIO-LEARN-1827."""
    G = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    S = newton_schulz_matrix_sign(G, steps=2)
    assert S.shape == G.shape
    
    with pytest.raises(ValueError, match="must be a 2D matrix"):
        newton_schulz_matrix_sign(jnp.array([1.0, 2.0]))

def test_muon_ogd_update():
    """Test REQ-LEARN-1827 / SCENARIO-LEARN-1827."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    
    updated = optimizer.update(params, grads)
    assert updated.shape == params.shape
    
    prior_memory = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    updated_with_memory = optimizer.update(params, grads, prior_memory=prior_memory)
    assert updated_with_memory.shape == params.shape

def test_muon_ogd_update_1d():
    """Test handling of 1D arrays."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    params = jnp.array([1.0, 1.0])
    grads = jnp.array([0.1, 0.2])
    
    updated = optimizer.update(params, grads)
    assert updated.shape == params.shape
    
    prior_memory = jnp.array([1.0, 0.0])
    updated_with_memory = optimizer.update(params, grads, prior_memory=prior_memory)
    assert updated_with_memory.shape == params.shape
