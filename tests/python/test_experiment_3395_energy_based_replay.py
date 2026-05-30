"""
Tests for REQ-LEARN-3395 and SCENARIO-LEARN-3395: FR-11 Energy-Guided Sample Selection.
"""
import pytest
import jax
import jax.numpy as jnp
import os
import json
import subprocess
from carnot.models.ising import IsingModel

def test_energy_guided_sample_selection():
    """
    SCENARIO-LEARN-3395: Energy Difference Guides Critical Sample Selection
    Given multiple constraint violations in the memory buffer
    When the Ising energy difference is computed for each sample
    Then the samples with the largest energy difference are selected for replay
    And the nonforgetting metric shows improvement or parity compared to random selection.
    """
    from scripts.experiment_3395_energy_based_replay import ConstraintViolationBuffer, simulate_nonforgetting_metric
    
    model = IsingModel(n_spins=10, seed=42)
    buffer = ConstraintViolationBuffer(capacity=50, model=model)
    
    # We will mock the energy function to strictly control the energy_diff
    key = jax.random.PRNGKey(0)
    
    for i in range(10):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        sample = jax.random.uniform(subkey1, shape=(10,))
        prev = jax.random.uniform(subkey2, shape=(10,))
        buffer.add_violation(sample, prev)
        
    # Manually tweak one to have very high energy diff by overriding in buffer
    buffer.buffer[0]["energy_diff"] = 1000.0
    buffer.buffer[1]["energy_diff"] = 500.0
    
    selected = buffer.select_for_replay(k=2, method="energy")
    
    assert len(selected) == 2
    assert selected[0]["energy_diff"] == 1000.0
    assert selected[1]["energy_diff"] == 500.0
    
    # Test metric
    random_selected = buffer.select_for_replay(k=2, method="random")
    metric_selected = simulate_nonforgetting_metric(selected, 10)
    metric_random = simulate_nonforgetting_metric(random_selected, 10)
    
    # Due to deterministic override, energy should be >= random
    assert metric_selected >= metric_random
    
    # Test ValueError
    with pytest.raises(ValueError):
        buffer.select_for_replay(k=2, method="invalid_method")
        
    assert simulate_nonforgetting_metric([], 10) == 0.0

def test_experiment_3395_execution():
    """
    REQ-LEARN-3395: Full script execution and JSON output verification.
    """
    script_path = "scripts/experiment_3395_energy_based_replay.py"
    result_path = "results/experiment_3395_energy_based_replay.json"
    
    if os.path.exists(result_path):
        os.remove(result_path)
        
    import sys
    # Use the local python to run the script
    subprocess.run([sys.executable, script_path], check=True)
    
    assert os.path.exists(result_path)
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.experiment.v1"
    assert data["experiment"] == 3395
    assert "honest_verdict" in data
    assert "nonforgetting_metric_selected" in data
    assert "nonforgetting_metric_random" in data
    assert data["nonforgetting_metric_selected"] >= 0.0
