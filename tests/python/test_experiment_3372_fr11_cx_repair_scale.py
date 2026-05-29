"""Tests for Experiment 3372: FR-11 CX Repair loop using Z3 Unsat Cores scaled to 100 cases."""

import pytest
import json
import os
from pathlib import Path
import sys

# Ensure scripts can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_3372_fr11_cx_repair_scale import generate_synthetic_flawed_templates, main

def test_generate_synthetic_flawed_templates_length():
    """Test that exactly 100 templates are generated."""
    templates = generate_synthetic_flawed_templates()
    assert len(templates) == 100
    
    # Check structure of the first template
    assert "variables" in templates[0]
    assert "constraints" in templates[0]
    assert len(templates[0]["constraints"]) == 3

def test_experiment_main_produces_success_rate():
    """Test that main returns 1.0 success rate and writes JSON."""
    
    success_rate = main()
    assert success_rate == 1.0
    
    # Check if the output file was created and contains expected data
    results_path = Path("results") / "experiment_3372_fr11_cx_repair_scale.json"
    assert results_path.exists()
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == "3372_fr11_cx_repair_scale"
    assert data["repair_success_rate"] == 1.0
    assert data["n_cases"] == 100
    assert data["honest_verdict"] == "success"
