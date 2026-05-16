"""Tests for Experiment 2090: CRANE HumanEval Evaluation."""

import sys
import json
import pytest
from pathlib import Path

from scripts.experiment_2090_crane_humaneval import evaluate_crane, evaluate_rigid, main

def test_evaluate_crane():
    """Test evaluate_crane function."""
    pass_rate = evaluate_crane(50)
    assert isinstance(pass_rate, float)
    assert 0.0 <= pass_rate <= 1.0

def test_evaluate_rigid():
    """Test evaluate_rigid function."""
    pass_rate = evaluate_rigid(50)
    assert isinstance(pass_rate, float)
    assert 0.0 <= pass_rate <= 1.0

def test_main(monkeypatch, tmp_path):
    """Test the main execution of experiment 2090."""
    monkeypatch.setattr(sys, "argv", ["scripts/experiment_2090_crane_humaneval.py"])
    
    # Run the main function
    main()
    
    # Check if the deliverable was created
    result_path = Path("results/experiment_2090_crane_humaneval.json")
    assert result_path.exists()
    
    # Verify the contents
    with open(result_path, "r") as f:
        data = json.load(f)
        
    assert data["target"] == "KV260"
    assert data["pipeline_invocations"] == 1
    assert data["simulated_energy_minimized"] is True
    assert "latency_ms" in data
    assert data["honest_verdict"] == "CRANE evaluated vs rigid grammar on 50 HumanEval problems."
    assert "crane_pass_rate" in data
    assert "rigid_pass_rate" in data
    assert data["pass_rate_delta"] > 0
    assert data["success"] is True
