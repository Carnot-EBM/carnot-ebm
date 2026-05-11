import os
import json
import jax.numpy as jnp
import pytest
from scripts import experiment_1773_latent_benchmark

def test_latent_benchmark_execution(tmp_path):
    """
    Test that the continuous latent constraint benchmark runs and produces the correct artifact.
    Traces to: REQ-BENCH-1773, SCENARIO-BENCH-1773
    """
    output_path = tmp_path / "experiment_1773_latent_benchmark.json"
    
    # Run the experiment
    experiment_1773_latent_benchmark.main(str(output_path))
    
    # Verify artifact was written
    assert output_path.exists()
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    # Verify required fields
    assert "status" in data
    assert data["status"] == "completed"
    assert "honest_verdict" in data
    assert "model_specs" in data
    assert "unsloth/gemma-4-31B-it-GGUF" in data["model_specs"]
    assert "energy_convergence" in data
    assert "validity_rates" in data
    
    # Energy should converge (first energy > last energy)
    convergence = data["energy_convergence"]
    assert len(convergence) > 1
    assert convergence[0] >= convergence[-1]
