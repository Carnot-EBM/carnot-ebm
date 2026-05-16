import jax
import jax.numpy as jnp
import pytest
from carnot.phase3.eqm_landscape import EqMLandscape, eqm_objective, run_experiment_2095

def test_eqm_objective():
    """
    Test that the EqM objective computes a valid loss.
    Traces to REQ-KONA-2095: EqM Implicit Energy Landscape.
    """
    key = jax.random.PRNGKey(0)
    
    # Dummy energy function: simple quadratic bowl E(x, theta) = theta * ||x||^2
    def energy_fn(theta, x):
        return theta * jnp.sum(x ** 2)

    # Gradient of energy with respect to state x
    def grad_energy_x(theta, x):
        return jax.grad(energy_fn, argnums=1)(theta, x)

    # Dummy target data and equilibrium state
    target_data = jnp.array([[1.0, 1.0], [-1.0, -1.0]])
    eq_state = jnp.array([[0.5, 0.5], [-0.5, -0.5]])
    theta = jnp.array(1.0)
    
    loss = eqm_objective(theta, target_data, eq_state, grad_energy_x)
    
    assert jnp.isfinite(loss)
    assert loss >= 0.0

def test_run_experiment_2095(tmp_path):
    """
    Test the artifact generation for EqM landscape.
    Traces to SCENARIO-KONA-2095-001.
    """
    import os
    import json
    
    # Redirect output to tmp_path
    output_path = tmp_path / "experiment_2095_eqm_landscape.json"
    run_experiment_2095(str(output_path))
    
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = json.load(f)
        
    assert data.get("eqm_landscape_ready") is True
    assert "schema" in data
