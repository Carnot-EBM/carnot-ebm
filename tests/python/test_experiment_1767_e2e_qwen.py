import os
import json
import pytest
import tempfile
import sys

# Ensure carnot and scripts are in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

import experiment_1767_e2e_qwen

def test_experiment_1767_e2e_qwen_runs():
    """Test that experiment 1767 runs and creates the expected JSON artifact.
    
    Spec: REQ-PIPELINE-1767
    Spec: SCENARIO-PIPELINE-1767
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1767_e2e_qwen.json")
        model_name = "unsloth/Qwen3.6-35B-A3B-GGUF"
        
        report = experiment_1767_e2e_qwen.run_experiment(output_path, model_name)
        
        # Verify return value
        assert report is not None
        assert report["experiment_id"] == "1767"
        assert report["model"] == model_name
        assert "latency" in report
        assert "parse_rate" in report
        assert "energy_scores" in report
        
        # Verify file artifact
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)
            assert data["experiment_id"] == "1767"
            assert data["model"] == model_name
            assert "latency" in data
            assert "parse_rate" in data
            assert "energy_scores" in data
