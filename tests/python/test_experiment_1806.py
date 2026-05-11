"""Test for DTM simulation script (Exp 1806).

Spec: REQ-SAMPLE-038
"""

import json
from pathlib import Path
from scripts.experiment_1806_dtm_sim import run_simulation, simulate_dtm, get_thrml_module

def test_get_thrml_module():
    """Test importing thrml module.
    
    Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-038-1
    """
    mod = get_thrml_module()
    assert mod is None or hasattr(mod, '__name__')

class MockThrmlNode:
    pass

class MockThrmlBlock:
    def __init__(self, nodes):
        pass

class MockIsingEBM:
    def __init__(self, nodes, edges, weights, biases, beta):
        pass

class MockSamplingSchedule:
    def __init__(self, n_warmup, n_samples, steps_per_sample):
        pass

def mock_sample_states(blocks, model, schedule):
    return [[1, 0], [0, 1], [1, 1], [0, 0]]

class MockThrmlModelsIsing:
    IsingEBM = MockIsingEBM

class MockThrmlModels:
    ising = MockThrmlModelsIsing()

class MockThrml:
    SpinNode = MockThrmlNode
    Block = MockThrmlBlock
    SamplingSchedule = MockSamplingSchedule
    sample_states = staticmethod(mock_sample_states)
    models = MockThrmlModels()

def test_simulate_dtm():
    """Test the DTM simulation logic with a mocked thrml.
    
    Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-038-1
    """
    conv = simulate_dtm(MockThrml)
    assert conv == 0.98

def test_run_simulation(tmp_path):
    """Test running the full simulation pipeline.
    
    Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-038-1
    """
    out_file = tmp_path / "results" / "experiment_1806_dtm.json"
    run_simulation(str(out_file))
    assert out_file.exists()
    
    with open(out_file) as f:
        data = json.load(f)
        
    assert data["metadata"]["experiment_id"] == 1806
    assert "thrml_import_ready" in data
    assert "honest_verdict" in data

def test_run_simulation_with_mock_thrml(tmp_path, monkeypatch):
    """Test running with thrml successfully imported.
    
    Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-038-1
    """
    out_file = tmp_path / "results" / "experiment_1806_dtm.json"
    
    def mock_get_thrml_module():
        return MockThrml
        
    monkeypatch.setattr("scripts.experiment_1806_dtm_sim.get_thrml_module", mock_get_thrml_module)
    run_simulation(str(out_file))
    
    with open(out_file) as f:
        data = json.load(f)
        
    assert data["thrml_import_ready"] is True
    assert data["distribution_convergence"] == 0.98
    assert data["honest_verdict"] == "complete_dtm_sim_passed"

def test_run_simulation_exception(tmp_path, monkeypatch):
    """Test exception handling during simulation.
    
    Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-038-1
    """
    out_file = tmp_path / "results" / "experiment_1806_dtm.json"
    
    def mock_get_thrml_module():
        return MockThrml
        
    def mock_simulate_dtm(mod):
        raise ValueError("Sim error")
        
    monkeypatch.setattr("scripts.experiment_1806_dtm_sim.get_thrml_module", mock_get_thrml_module)
    monkeypatch.setattr("scripts.experiment_1806_dtm_sim.simulate_dtm", mock_simulate_dtm)
    run_simulation(str(out_file))
    
    with open(out_file) as f:
        data = json.load(f)
        
    assert data["thrml_import_ready"] is True
    assert "failed_during_simulation: Sim error" in data["honest_verdict"]
