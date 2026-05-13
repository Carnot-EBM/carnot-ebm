import os
import sys
import json
import pytest

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import experiment_2077_retro as module

# REQ-RETRO-001, SCENARIO-RETRO-001
def test_generate_retro_data():
    """Verify that the retro data generation returns the correct schema and content."""
    data = module.generate_retro_data()
    
    assert data["experiment_id"] == 2077
    assert data["schema"] == "carnot.milestone_retro.v1"
    assert data["milestone"] == "2026.05.162"
    assert data["status"] == "complete"
    assert data["criteria_results"]["continuous_self_learning_loop_closed"] is True
    assert any("KAN Integration" in s for s in data["notable_successes"])

def test_main_execution(tmp_path, monkeypatch):
    """Verify that main() writes the JSON correctly."""
    # Run the main function with a mocked results directory
    monkeypatch.chdir(tmp_path)
    module.main()
    
    out_path = os.path.join("results", "experiment_2077_retro.json")
    assert os.path.exists(out_path)
    
    with open(out_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        
    assert saved_data["experiment_id"] == 2077
    assert saved_data["schema"] == "carnot.milestone_retro.v1"
