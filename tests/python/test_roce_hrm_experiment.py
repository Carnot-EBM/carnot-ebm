"""Tests for ROCE and HRM experiment 1765.

Spec: REQ-ROCE-HRM-1765, SCENARIO-ROCE-HRM-1765
"""

import os
import json
import tempfile
import sys

# Ensure scripts can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.experiment_1765_roce_hrm import run_experiment

def test_roce_hrm_experiment_output():
    """Test that the experiment script produces a valid JSON file.
    
    Spec: REQ-ROCE-HRM-1765-2
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1765_eval.json")
        model_name = "test-model-4-26B"
        
        report = run_experiment(output_path, model_name)
        
        # Verify returned report
        assert report["experiment_id"] == "1765"
        assert report["model"] == model_name
        assert "constraint_satisfaction_rate" in report
        assert report["total_prompts_evaluated"] == 2
        assert report["total_constraints_extracted"] > 0
        
        # Verify saved file
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            saved_report = json.load(f)
            
        assert saved_report == report
        
        # Verify the HRM details are present
        assert len(saved_report["details"]) == 2
        for item in saved_report["details"]:
            assert "prompt" in item
            assert "constraints_extracted" in item
            assert "score" in item
            assert "hrm_details" in item
            assert "level_1" in item["hrm_details"]
