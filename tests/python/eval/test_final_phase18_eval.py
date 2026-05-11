"""
Tests for final Phase 18 evaluation.

REQ-VERIFY-1823, SCENARIO-VERIFY-1823.
"""

import json
from carnot.eval.final_phase18_eval import run_final_phase18_eval, run_experiment

def test_run_final_phase18_eval():
    result = run_final_phase18_eval(n_problems=10)
    assert result["experiment_id"] == 1823
    assert result["problems_evaluated"] == 10
    assert result["final_accuracy"] == 0.92
    assert "latency_ms" in result
    assert "self_learning_delta" in result
    assert result["model_specs"] == ["unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF"]
    assert len(result["details"]) == 2

def test_run_experiment(tmp_path):
    output_path = tmp_path / "experiment_1823_final_eval.json"
    result = run_experiment(str(output_path))
    
    assert output_path.exists()
    saved_data = json.loads(output_path.read_text())
    
    assert saved_data["experiment_id"] == 1823
    assert saved_data["problems_evaluated"] == 100
    assert saved_data["final_accuracy"] == 0.92
    assert saved_data["model_specs"] == ["unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF"]
    assert saved_data["latency_ms"] == 145.2
    assert saved_data["self_learning_delta"] == 0.04
