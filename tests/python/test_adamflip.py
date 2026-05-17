"""Tests for AdamFLIP optimizer."""
import jax.numpy as jnp
from carnot.training.adamflip import AdamFLIP
from carnot.training.csl_loop import CSLLoop, run_csl_loop
from carnot.training.muon_ogd import MuonOGD

def test_adamflip_update():
    """Test REQ-LEARN-2127: AdamFLIP applies adaptive momentum feedback linearization."""
    optimizer = AdamFLIP(learning_rate=0.1)
    residuals = jnp.array([[0.5, -0.5], [0.1, 0.0]])
    
    # First update
    feedback1 = optimizer.update(residuals)
    assert feedback1.shape == residuals.shape
    assert optimizer.m is not None
    assert optimizer.v is not None
    assert optimizer.t == 1
    
    # Second update
    residuals2 = jnp.array([[0.1, -0.1], [0.0, 0.2]])
    feedback2 = optimizer.update(residuals2)
    assert feedback2.shape == residuals.shape
    assert optimizer.t == 2

def test_csl_loop_with_adamflip():
    """Test REQ-LEARN-2127: CSLLoop applies AdamFLIP to constraint residuals."""
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    residuals = jnp.array([[0.01, -0.01], [0.05, 0.0]])
    
    result = run_csl_loop(params, grads, residuals)
    assert result["status"] == "success"
    assert result["adamflip_applied"] is True
    assert "updated_norm" in result

def test_csl_loop_step_with_residuals():
    """Test REQ-LEARN-2127: CSLLoop step with residuals."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    csl = CSLLoop(optimizer)
    
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    residuals = jnp.array([[0.01, -0.01], [0.05, 0.0]])
    
    updated1 = csl.step(params, grads, residuals)
    assert updated1.shape == params.shape
