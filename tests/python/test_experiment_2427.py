import pytest
import os
import json
import subprocess
import sys

def test_experiment_2427_runs():
    # Run the experiment script
    result = subprocess.run([sys.executable, "/home/ianblenke/github.com/ianblenke/carnot/experiment_2427_kv260_yosys.py"], capture_output=True, text=True)
    assert result.returncode == 0
    
    # Check the json file
    json_path = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_2427_kv260_yosys_v4.json"
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert "synthesis_succeeded" in data
    assert "lut_count" in data
    assert "flip_flop_count" in data
    assert "synthesis_warnings" in data
    assert "synthesis_errors" in data
    assert "yosys_version" in data
    assert "rtl_files_synthesized" in data
    assert "duration_s" in data
    assert "preconditions_checked" in data
    
    assert data["honest_verdict"].startswith("blocked_") or data["synthesis_succeeded"] == True
    assert data["synthesis_succeeded"] == False
    assert data["synthesis_errors"] > 0
