"""
Tests for Milestone 159 retrospective script.
"""
import json
import os
import sys
import subprocess
from pathlib import Path

# Add scripts directory to path to allow direct import if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_2038_milestone_159_retro import generate_retro_data, main

def test_generate_retro_data():
    """Test the generation of retrospective data."""
    data = generate_retro_data()
    
    assert data["experiment"] == 2038
    assert data["milestone"] == 159
    assert data["status"] == "success"
    
    retro = data["retrospective_analysis"]
    assert "continuous_latent_refinements" in retro
    assert retro["continuous_latent_refinements"]["success"] is False
    
    assert "formal_verification_architectures" in retro
    assert retro["formal_verification_architectures"]["success"] is False
    
    assert "overall_conclusion" in retro

def test_main_execution(tmp_path, monkeypatch):
    """Test the main function writes the correct file."""
    # Monkeypatch the results directory to use tmp_path
    def mock_makedirs(name, mode=0o777, exist_ok=False):
        pass
    
    monkeypatch.setattr(os, "makedirs", mock_makedirs)
    
    # We will monkeypatch os.path.join to return a path in our tmp_dir
    original_join = os.path.join
    
    def mock_join(a, *p):
        if a == "results":
            return original_join(str(tmp_path), *p)
        return original_join(a, *p)
        
    monkeypatch.setattr(os.path, "join", mock_join)
    
    main()
    
    out_file = tmp_path / "experiment_2038_milestone_159_retro.json"
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["experiment"] == 2038
    assert data["status"] == "success"

