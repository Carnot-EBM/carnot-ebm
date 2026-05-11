"""Tests for Phase 18 Final Evaluation Run.

Spec traces: REQ-EVAL-1823, SCENARIO-EVAL-1823.
"""

from carnot.eval.phase18_final_eval import evaluate_phase18, run_experiment
import json

def test_evaluate_phase18():
    """Test the evaluation logic correctly mocks the Phase 18 run."""
    result = evaluate_phase18()
    assert result["experiment_id"] == 1823
    assert result["honest_verdict"] == "complete: Phase 18 final evaluation completed"
    assert result["problems_evaluated"] == 100
    assert result["final_accuracy"] > 0
    assert "latency_ms" in result
    assert "self_learning_delta" in result
    assert result["model_specs"] == ["unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF"]

def test_run_experiment(tmp_path):
    """Test that run_experiment writes to the specified JSON path."""
    out_file = tmp_path / "experiment_1823_final_eval.json"
    result = run_experiment(output_path=str(out_file))
    
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data == result
