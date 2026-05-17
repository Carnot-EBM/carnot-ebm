"""Tests for CSL loop."""
import pytest
import jax.numpy as jnp
from carnot.training.csl_loop import CSLLoop, run_csl_loop
from carnot.training.muon_ogd import MuonOGD

def test_csl_loop_step():
    """Test REQ-LEARN-1827 / SCENARIO-LEARN-1827."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    csl = CSLLoop(optimizer)
    
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    
    updated1 = csl.step(params, grads)
    assert csl.memory is not None
    assert updated1.shape == params.shape
    
    grads2 = jnp.array([[0.5, 0.6], [0.7, 0.8]])
    updated2 = csl.step(updated1, grads2)
    assert updated2.shape == params.shape

def test_csl_loop_step_zero_grad():
    """Test with zero gradient."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    csl = CSLLoop(optimizer)
    
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    
    updated = csl.step(params, grads)
    assert csl.memory is not None

def test_run_csl_loop():
    """Test REQ-LEARN-1827 / SCENARIO-LEARN-1827."""
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    
    result = run_csl_loop(params, grads)
    assert result["status"] == "success"
    assert result["muon_ogd_applied"] is True
    assert "updated_norm" in result

def test_csl_loop_prem_intrinsic_reward():
    """Test REQ-LEARN-2152 / SCENARIO-LEARN-2152."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    csl = CSLLoop(optimizer)
    
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    prem_intrinsic_reward = jnp.array([[0.05, 0.05], [0.05, 0.05]])
    
    updated_no_reward = csl.step(params, grads)
    
    csl2 = CSLLoop(optimizer)
    updated_with_reward = csl2.step(params, grads, prem_intrinsic_reward=prem_intrinsic_reward)
    
    assert not jnp.allclose(updated_no_reward, updated_with_reward)

def test_run_csl_loop_with_prem():
    """Test REQ-LEARN-2152 / SCENARIO-LEARN-2152."""
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    prem_intrinsic_reward = jnp.array([[0.05, 0.05], [0.05, 0.05]])
    
    result = run_csl_loop(params, grads, prem_intrinsic_reward=prem_intrinsic_reward)
    assert result["status"] == "success"
    assert result["prem_intrinsic_applied"] is True
