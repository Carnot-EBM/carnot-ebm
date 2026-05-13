"""
Tests for Milestone 160 retrospective script.
"""
import json
import os
import sys
from pathlib import Path

# Add scripts directory to path to allow direct import if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_2052_milestone_160_retro import generate_retro_data, main

def test_generate_retro_data():
    """Test the generation of retrospective data."""
    data = generate_retro_data()
    
    assert data["experiment"] == 2052
    assert data["milestone"] == 160
    assert data["status"] == "success"
    
    retro = data["retrospective_analysis"]
    assert "FAR_latent_transition" in retro
    assert retro["FAR_latent_transition"]["success"] is True
    
    assert "AIA_hardware_metrics" in retro
    assert retro["AIA_hardware_metrics"]["success"] is True
    
    assert "FR_11_self_learning_utility" in retro
    assert retro["FR_11_self_learning_utility"]["success"] is True
    
    assert "overall_conclusion" in retro

def test_main_execution(tmp_path, monkeypatch):
    """Test the main function writes the correct file."""
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
    
    out_file = tmp_path / "experiment_2052_milestone_160_retro.json"
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["experiment"] == 2052
    assert data["status"] == "success"
