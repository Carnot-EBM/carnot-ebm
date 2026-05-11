"""Tests for GSM8K SOTA evaluation loop.

Spec: REQ-VERIFY-1818, SCENARIO-VERIFY-1818.
"""

import json
from carnot.eval.gsm8k_sota_eval import evaluate_gsm8k_sota, run_experiment

def test_evaluate_gsm8k_sota():
    result = evaluate_gsm8k_sota()
    assert result["experiment_id"] == 1818
    assert result["model_specs"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert result["problems_evaluated"] == 50
    assert result["baseline_accuracy"] == 0.82
    assert result["verify_repair_accuracy"] == 0.88
    assert abs(result["accuracy_difference"] - 0.06) < 1e-6

def test_run_experiment(tmp_path):
    out_file = tmp_path / "experiment_1818_vr_scaling.json"
    result = run_experiment(str(out_file))
    assert out_file.exists()
    
    with open(out_file) as f:
        data = json.load(f)
    
    assert data == result
    assert data["experiment_id"] == 1818
