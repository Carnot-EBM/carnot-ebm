"""
Tests for Milestone 161 retrospective script (Exp 2065).
"""
import json
import os
import sys
from pathlib import Path

# Add scripts directory to path to allow direct import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_2065_retro import generate_retro_data, main

def test_generate_retro_data():
    """Test the generation of retrospective data for 2065."""
    data = generate_retro_data()
    
    assert data["experiment_id"] == 2065
    assert data["milestone"] == "2026.05.161"
    assert data["status"] == "complete"
    
    assert data["retro_complete"] is True
    assert "Mouth/Brain" in data["notable_successes"][0]
    assert "TSU" in data["notable_successes"][1]
    assert "EBT" in data["notable_successes"][2]
    
    assert data["criteria_met"] == 3
    assert data["criteria_results"]["mouth_brain_separation_implemented"] is True

def test_main_execution(tmp_path, monkeypatch):
    """Test the main function writes the correct JSON file."""
    # Monkeypatch the results directory to use tmp_path
    def mock_makedirs(name, mode=0o777, exist_ok=False):
        pass
    
    monkeypatch.setattr(os, "makedirs", mock_makedirs)
    
    original_join = os.path.join
    
    def mock_join(a, *p):
        if a == "results":
            return original_join(str(tmp_path), *p)
        return original_join(a, *p)
        
    monkeypatch.setattr(os.path, "join", mock_join)
    
    main()
    
    out_file = tmp_path / "experiment_2065_retro.json"
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["experiment_id"] == 2065
    assert data["status"] == "complete"
    assert "milestone_161_retro_filed" in data["honest_verdict"]
