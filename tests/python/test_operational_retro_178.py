import os
import sys
import json
import pytest

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import operational_retro_178 as module

# REQ-RETRO-178, SCENARIO-RETRO-178
def test_generate_retro_data(monkeypatch):
    """Verify that the retro data generation returns the correct schema and content."""
    # Mock subprocess.run to simulate git log returning non-empty
    import subprocess
    class MockResult:
        def __init__(self, stdout):
            self.stdout = stdout

    def mock_run(cmd, *args, **kwargs):
        if "Activate milestone 2026.05.178" in cmd:
            return MockResult("764f43b45 [conductor] Activate milestone 2026.05.178\\n")
        elif "..HEAD" in cmd:
            return MockResult("deadbeef Some commit\\n")
        return MockResult("")

    monkeypatch.setattr(subprocess, "run", mock_run)

    data = module.generate_retro_data()
    
    assert data["schema"] == "carnot.operational_retro.v64"
    assert data["milestone"] == "2026.05.178"
    assert data["retro_type"] == "operational_full"
    assert "preconditions_checked" in data
    assert "git log [conductor] Activate milestone 2026.05.178..HEAD returns non-empty: True" in data["preconditions_checked"]
    assert "total_wall_time_minutes" in data
    assert "experiments_completed" in data
    assert "compute_bound_experiments_count" in data
    assert "slowest_experiments" in data
    assert "gpu_idle_on_compute_bound_tasks" in data
    assert "summary" in data
    assert "bottlenecks_identified" in data
    assert "improvements_suggested" in data
    assert "top_3_highest_leverage_actions" in data
    assert "estimated_time_savings_pct" in data
    assert "meta_reflection" in data
    assert data["honest_verdict"].startswith("complete:")

def test_main_execution(tmp_path, monkeypatch):
    """Verify that main() writes the JSON correctly."""
    monkeypatch.chdir(tmp_path)
    
    import subprocess
    class MockResult:
        def __init__(self, stdout):
            self.stdout = stdout

    def mock_run(cmd, *args, **kwargs):
        if "Activate milestone 2026.05.178" in cmd:
            return MockResult("764f43b45 [conductor] Activate milestone 2026.05.178\\n")
        elif "..HEAD" in cmd:
            return MockResult("deadbeef Some commit\\n")
        return MockResult("")

    monkeypatch.setattr(subprocess, "run", mock_run)
    
    module.main()
    
    out_path = os.path.join("results", "operational_retro_2026_05_178.json")
    assert os.path.exists(out_path)
    
    with open(out_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        
    assert saved_data["schema"] == "carnot.operational_retro.v64"
    assert saved_data["milestone"] == "2026.05.178"
