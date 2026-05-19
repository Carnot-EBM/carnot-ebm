import json
import subprocess
from pathlib import Path

def test_blocked_artifact_generation() -> None:
    script_path = Path("scripts/experiment_2510_ensemble_v7.py")
    assert script_path.exists(), "Experiment script not found"
    
    result = subprocess.run(["python3", str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    artifact_path = Path("results/experiment_2510_ensemble_v7.json")
    assert artifact_path.exists(), "Artifact not generated"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "blocked_tier0r_not_implemented"
    assert data["ensemble_v7_auroc"] == 0.0
    assert data["ensemble_v7_auroc_std"] == 0.0
    assert data["ensemble_v6_baseline"] == 0.9750
    assert data["tier0r_group_assignment"] == "Group C"
    assert data["n_seeds"] == 5
    assert "preconditions_checked" in data
    assert "duration_s" in data
    assert data["random_seed"] == 42
