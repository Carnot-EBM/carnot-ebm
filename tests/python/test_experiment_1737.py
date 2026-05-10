import os
import json
import importlib.util
import sys

# Requirements: REQ-HW-053
# Scenarios: SCENARIO-HW-053

def test_experiment_1737_kanele_board():
    # Load the module
    spec = importlib.util.spec_from_file_location(
        "experiment_1737_kanele_board",
        "scripts/experiment_1737_kanele_board.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_1737_kanele_board"] = module
    spec.loader.exec_module(module)
    
    # Run the main function
    module.main()
    
    artifact_path = "results/experiment_1737_kanele_board.json"
    assert os.path.exists(artifact_path)
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert "hardware_latency_us" in data
    assert "throughput_fps" in data
    assert "batch_size" in data
    assert data["experiment"] == "1737"
    assert data["batch_size"] == 1000
    assert data["status"] == "success"
