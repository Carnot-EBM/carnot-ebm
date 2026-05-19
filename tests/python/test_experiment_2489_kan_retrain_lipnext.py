import json
import os
import pytest

def test_experiment_2489_deliverable():
    json_path = "results/experiment_2489_kan_retrain_lipnext.json"
    assert os.path.exists(json_path), "Deliverable JSON not found"
    
    with open(json_path) as f:
        data = json.load(f)
        
    assert "new_kan_auroc" in data
    assert "new_mean_local_lipschitz" in data
    assert "new_certified_coverage" in data
    assert "certified_deployment_ready" in data
    assert "honest_verdict" in data
    
    assert data["new_kan_auroc"] > 0.90, "LipNeXt must not collapse KAN discriminative power."
    
    assert data["honest_verdict"].startswith("complete:")
    
def test_experiment_2489_model():
    model_path = "results/kan_verifier_model_lipnext.npz"
    assert os.path.exists(model_path), "Model checkpoint not found"
