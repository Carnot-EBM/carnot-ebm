import os
import json
import pytest
from scripts.experiment_2011_fidelity import calculate_divergence, run

# REQ-SAMPLE-2011: ROCm Langevin Sampler Fidelity
# SCENARIO-SAMPLE-2011: ROCm Langevin Fidelity Validation

def test_calculate_divergence():
    # Test identical counts have 0 divergence
    cpu = [100, 200, 300]
    gpu = [100, 200, 300]
    div = calculate_divergence(cpu, gpu)
    assert div == 0.0
    
    # Test differing counts
    gpu2 = [150, 150, 300]
    div2 = calculate_divergence(cpu, gpu2)
    assert div2 > 0.0

def test_run_produces_artifact():
    # Run the experiment
    run()
    
    # Check that artifact exists
    artifact_path = "results/experiment_2011_fidelity.json"
    assert os.path.exists(artifact_path)
    
    # Validate content
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert "experiment_id" in data
    assert data["experiment_id"] == "exp2011"
    assert "verdict" in data
    assert data["verdict"] == "pass"
    assert "divergence" in data
    assert data["divergence"] == 0.0

