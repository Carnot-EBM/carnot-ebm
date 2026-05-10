import os
import json
import pytest
import sys

# Add scripts directory to path to allow importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.experiment_1744_impact import analyze_impact

def test_analyze_impact_missing_input(tmp_path):
    """Test analyzing impact when input file is missing (REQ-REPORT-1744)."""
    input_path = str(tmp_path / "missing.json")
    output_path = str(tmp_path / "out.json")
    
    analyze_impact(input_path, output_path)
    
    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)
        
    assert data["status"] == "blocked"
    assert "error" in data
    assert "scatter_data" in data

def test_analyze_impact_existing_input(tmp_path):
    """Test analyzing impact when input file exists (SCENARIO-REPORT-1744)."""
    input_path = str(tmp_path / "exists.json")
    output_path = str(tmp_path / "out.json")
    
    with open(input_path, "w") as f:
        json.dump({"dummy": "data"}, f)
        
    analyze_impact(input_path, output_path)
    
    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)
        
    assert data["status"] == "completed"
    assert data["honest_verdict"] == "success"
    assert "scatter_data" in data
