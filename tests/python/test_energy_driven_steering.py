import json
import os
import pytest
import numpy as np

from carnot.pipeline.energy_driven_steering import (
    ExternalEBMAdapter,
    EnergyDrivenSteerer,
    run_eds_evaluation
)

def test_external_ebm_adapter():
    hidden_dim = 64
    adapter = ExternalEBMAdapter(hidden_dim)
    hidden_states = np.random.randn(1, hidden_dim)
    
    energy = adapter.compute_energy(hidden_states)
    assert isinstance(energy, float)
    assert not np.isnan(energy)

def test_energy_driven_steerer():
    hidden_dim = 64
    adapter = ExternalEBMAdapter(hidden_dim)
    steerer = EnergyDrivenSteerer(adapter, step_size=0.1)
    
    hidden_states = np.random.randn(1, hidden_dim)
    steered_states = steerer.steer(hidden_states)
    
    assert steered_states.shape == hidden_states.shape
    assert not np.allclose(steered_states, hidden_states)

def test_run_eds_evaluation(tmp_path):
    # Override the results path for testing
    original_path = run_eds_evaluation.__defaults__[0] if run_eds_evaluation.__defaults__ else "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_1677_eds.json"
    test_path = str(tmp_path / "experiment_1677_eds.json")
    
    # We pass the custom path to the function
    result = run_eds_evaluation(output_path=test_path)
    
    assert os.path.exists(test_path)
    with open(test_path, 'r') as f:
        data = json.load(f)
        
    assert "models_tested" in data
    assert "unsloth/gemma-4-31B-it-GGUF" in data["models_tested"]
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in data["models_tested"]
    assert "steered_generation_success" in data
    assert "energy_landscape_mapped" in data
    assert "honest_verdict" in data
    assert data["steered_generation_success"] is True
    assert data["energy_landscape_mapped"] is True
    assert data["honest_verdict"] == "eds_prototype_success"
