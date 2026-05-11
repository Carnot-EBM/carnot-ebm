import os
import json
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from experiment_1767_e2e_qwen import run_experiment

def test_experiment_1767_run():
    """
    Spec: SCENARIO-E2E-QWEN-1767
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1767_e2e_qwen.json")
        model_name = "unsloth/Qwen3.6-35B-A3B-GGUF"
        
        report = run_experiment(output_path, model_name)
        
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)
            
        assert data["experiment_id"] == "1767"
        assert data["model"] == model_name
        assert "latency_ms" in data
        assert "parse_rate" in data
        assert "energy_score" in data
        assert "total_prompts_evaluated" in data

import subprocess

def test_experiment_1767_main():
    """Test the main block of experiment 1767"""
    result = subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/experiment_1767_e2e_qwen.py'))], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Experiment completed." in result.stdout
