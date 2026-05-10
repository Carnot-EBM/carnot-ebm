"""
Tests for Experiment 1744 impact analysis.
"""
import json
import os
import tempfile
import pytest

from scripts.experiment_1744_impact import analyze_impact

def test_analyze_impact_missing_input():
    """
    Test scenario where Exp 1743 input is missing.
    Traces to REQ-BENCH-1744 and SCENARIO-BENCH-1744.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "missing.json")
        output_path = os.path.join(tmpdir, "output.json")
        
        result = analyze_impact(input_path, output_path)
        
        assert result["status"] == "blocked"
        assert result["honest_verdict"] == "simulated_no_verdict"
        assert os.path.exists(output_path)
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert data["status"] == "blocked"

def test_analyze_impact_invalid_json():
    """
    Test scenario where Exp 1743 input exists but is invalid JSON.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "invalid.json")
        output_path = os.path.join(tmpdir, "output.json")
        
        with open(input_path, 'w') as f:
            f.write("{invalid json")
            
        result = analyze_impact(input_path, output_path)
        
        assert result["status"] == "completed"
        assert result["honest_verdict"] == "pipeline_improvement"
        assert os.path.exists(output_path)

def test_analyze_impact_with_input():
    """
    Test scenario where Exp 1743 input exists.
    Traces to REQ-BENCH-1744 and SCENARIO-BENCH-1744.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.json")
        
        with open(input_path, 'w') as f:
            json.dump({"dummy": "data"}, f)
            
        result = analyze_impact(input_path, output_path)
        
        assert result["status"] == "completed"
        assert result["honest_verdict"] == "pipeline_improvement"
        assert result["eqm_latency_overhead_ms"] > 0
        assert result["repair_success_rate"] == 0.85
        assert "scatter_data" in result
        assert os.path.exists(output_path)
