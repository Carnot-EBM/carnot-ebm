import os
import json
import subprocess

def test_req_hw_056_synth_constraints():
    """
    Test that the Makefile synth-constraints target runs and generates
    the JSON artifact.
    REQ-HW-056, SCENARIO-HW-056
    """
    # 1. Run the make target
    res = subprocess.run(["make", "synth-constraints"], capture_output=True, text=True)
    assert res.returncode == 0, f"Synthesis failed: {res.stderr}"

    # 2. Verify we generated the json
    json_path = "results/experiment_1822_rtl_synth.json"
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == 1822
    assert "utilization" in data
    assert "LUT" in data["utilization"]
