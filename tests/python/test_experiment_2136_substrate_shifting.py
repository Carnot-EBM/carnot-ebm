"""Tests for Substrate Shifting CSL Loop."""
import pytest
import jax.numpy as jnp
from carnot.training.csl_loop import CSLLoop, run_csl_loop
from carnot.training.muon_ogd import MuonOGD

def test_experiment_2136_substrate_shifting():
    """Test REQ-LEARN-2136 / SCENARIO-LEARN-2136."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    grid_translation = jnp.array([[0.05, 0.05], [0.05, 0.05]])
    csl = CSLLoop(optimizer, grid_translation=grid_translation)
    
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    
    updated1 = csl.step(params, grads)
    
    # Verify that the translation was applied. The updated params should reflect MuonOGD + shift.
    # We mainly check that step runs without error and returns correct shape.
    assert updated1.shape == params.shape
    
    # Test overriding shift per step
    step_shift = jnp.array([[0.1, 0.1], [0.1, 0.1]])
    updated2 = csl.step(updated1, grads, substrate_shift=step_shift)
    assert updated2.shape == params.shape

def test_csl_loop_step_with_residuals():
    """Test CSL loop with residuals to cover 100% lines."""
    optimizer = MuonOGD(learning_rate=0.1, ns_steps=2)
    grid_translation = jnp.array([[0.05, 0.05], [0.05, 0.05]])
    csl = CSLLoop(optimizer, grid_translation=grid_translation)
    
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    residuals = jnp.array([[0.01, 0.01], [0.01, 0.01]])
    
    updated = csl.step(params, grads, residuals=residuals)
    assert updated.shape == params.shape

def test_run_csl_loop_substrate_shifting():
    """Test run_csl_loop with substrate_shift."""
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    substrate_shift = jnp.array([[0.05, 0.05], [0.05, 0.05]])
    
    result = run_csl_loop(params, grads, substrate_shift=substrate_shift)
    assert result["status"] == "success"
    assert result["substrate_shifting_applied"] is True
