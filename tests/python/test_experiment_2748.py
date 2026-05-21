import os
import json

def test_experiment_2748_output():
    # Just verifies that the experiment script produced the correct JSON schema
    output_path = "results/experiment_2748_phase4_fep_factor_graph.json"
    assert os.path.exists(output_path), "Experiment deliverable missing"
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    required_fields = [
        "honest_verdict",
        "fep_factor_graph_computed",
        "fep_auroc",
        "fep_viable",
        "fep_vs_odar_delta",
        "alpha_t_nonzero",
        "alpha_t_mean",
        "n_verifiers",
        "random_seed",
        "duration_s",
        "preconditions_checked"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
        
    assert str(data["honest_verdict"]).startswith("complete:") or str(data["honest_verdict"]).startswith("blocked_")
