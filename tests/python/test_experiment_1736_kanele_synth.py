import os
import json
import subprocess

def test_experiment_1736_artifact():
    """
    Test for REQ-HW-052 and SCENARIO-HW-052.
    Ensures the script experiment_1736_kanele_synth.py successfully creates the artifact.
    """
    script_path = "scripts/experiment_1736_kanele_synth.py"
    artifact_path = "results/experiment_1736_kanele_synth.json"
    
    # Run the script
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # Verify the artifact exists
    assert os.path.exists(artifact_path)
    
    # Verify artifact contents
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == "1736"
    assert data["status"] == "success"
    assert "vivado_available" in data
    assert "bitfile_generated" in data
    assert "utilization" in data
    assert "honest_verdict" in data
