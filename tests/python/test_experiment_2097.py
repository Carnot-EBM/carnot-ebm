import os
import json
import pytest
from scripts.experiment_2097_eqm_eval import run_experiment_2097

def test_experiment_2097_evaluates_eqm_vs_pem():
    """
    REQ-KONA-2097: EqM vs PEM Comparison on Continuous Constraint Graphs
    SCENARIO-KONA-2097: Exp 2097 Evaluates EqM on Continuous Graphs
    """
    output_path = "results/experiment_2097_eqm_eval.json"
    
    # Ensure a clean slate
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Run the experiment
    run_experiment_2097()
    
    # Verify the artifact was generated
    assert os.path.exists(output_path)
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    assert "schema" in data
    assert data["experiment_id"] == "2097"
    assert "eqm_superior" in data
    assert isinstance(data["eqm_superior"], bool)
    assert data["num_instances"] == 50
    assert "avg_pem_energy" in data
    assert "avg_eqm_energy" in data
