import json
import subprocess
from pathlib import Path

def test_experiment_2514_artifact_generation() -> None:
    script_path = Path("scripts/experiment_2514_kv260_pynq_flash.py")
    assert script_path.exists(), "Experiment script not found"
    
    result = subprocess.run(["python3", str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    artifact_path = Path("results/experiment_2514_kv260_pynq_flash.json")
    assert artifact_path.exists(), "Artifact not generated"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert "terminal: KV260 hwh file successfully generated" in data["honest_verdict"]
    assert data["kv260_hwh_generated"] is True
    assert data["kv260_flash_attempted"] is False
    assert data["kv260_blocker_documented"] is True
    assert data["vivado_version"] == "vivado v2025.2.1 (64-bit)"
    assert "preconditions_checked" in data
    assert "duration_s" in data

if __name__ == "__main__":
    test_experiment_2514_artifact_generation()
    print("Test passed!")
