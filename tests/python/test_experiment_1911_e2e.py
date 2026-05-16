"""Tests for Experiment 1911 E2E script.

Spec: REQ-1911-E2E, SCENARIO-1911-E2E
"""
import os
import sys
import json
import pytest

# Add scripts directory to path to allow importing experiment_1911_e2e
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import experiment_1911_e2e

def test_run_evaluation():
    """Test the evaluation function runs successfully."""
    # Given / When
    result = experiment_1911_e2e.run_evaluation()
    
    # Then
    assert result["experiment_id"] == "1911"
    assert result["date"] == "20260516"
    assert result["integration_status"] == "success"
    assert result["fast_slow_variant"]["passed"] is True
    assert result["semantic_grounding"]["verified"] is True
    assert result["muon_ogd"]["available"] is True

def test_main(tmp_path, monkeypatch):
    """Test main function generates the JSON deliverable."""
    # Use tmp_path for results directory
    monkeypatch.chdir(tmp_path)
    
    # When
    experiment_1911_e2e.main()
    
    # Then
    output_file = tmp_path / "results" / "experiment_1911_e2e.json"
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["experiment_id"] == "1911"
