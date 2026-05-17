import os
import json
from pathlib import Path

def test_experiment_2143_archive():
    artifact_path = Path("results/experiment_2143_archive.json")
    assert artifact_path.exists(), "Deliverable must exist"
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert data["honest_verdict"] == "activation_complete"
    assert data["experiment"] == 2143
