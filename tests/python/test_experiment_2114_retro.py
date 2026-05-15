import os
import sys
import json
import pytest

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import experiment_2114_retro as module

# REQ-RETRO-2114, SCENARIO-RETRO-2114
def test_generate_retro_data():
    """Verify that the retro data generation returns the correct schema and content."""
    data = module.generate_retro_data()
    
    assert data["experiment_id"] == 2114
    assert data["schema"] == "carnot.milestone_retro.v1"
    assert data["status"] == "complete"
    assert data["kona_parity_achieved"] is True
    assert "highest_leverage_actions" in data
    assert "estimated_time_savings_pct" in data
    assert "meta_reflection" in data
    assert "honest_verdict" in data

def test_main_execution(tmp_path, monkeypatch):
    """Verify that main() writes the JSON correctly."""
    monkeypatch.chdir(tmp_path)
    module.main()
    
    out_path = os.path.join("results", "experiment_2114_retro.json")
    assert os.path.exists(out_path)
    
    with open(out_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        
    assert saved_data["experiment_id"] == 2114
    assert saved_data["schema"] == "carnot.milestone_retro.v1"
