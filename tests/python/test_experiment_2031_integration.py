import json
import subprocess
from pathlib import Path

def test_experiment_2031_integration():
    """Integration test for EBT guided decoding loop (REQ-EBT-2031)."""
    # Run the experiment script
    result = subprocess.run(["python", "scripts/experiment_2031_ebt_gemma.py"], check=True, capture_output=True, text=True)
    
    # Verify the output json
    json_path = Path("results/experiment_2031.json")
    assert json_path.exists()
    
    with open(json_path) as f:
        data = json.load(f)
        
    assert data["experiment"] == 2031
    assert data["schema"] == "carnot.experiment.v1"
    assert data["status"] == "success"
    assert "best_candidate" in data
    assert "min_energy" in data
    assert data["best_candidate"] == "Thus, we can see it."
    assert data["min_energy"] == 0.0
