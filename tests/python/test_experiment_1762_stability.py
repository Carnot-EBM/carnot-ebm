"""Tests for Exp 1762.
Spec: REQ-LEARN-1762
"""
import pytest
from scripts.experiment_1762_stability import run_experiment_1762

def test_run_experiment_1762(monkeypatch, tmp_path):
    """Test Exp 1762.
    Spec: REQ-LEARN-1762, SCENARIO-LEARN-1762
    """
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    artifact = run_experiment_1762()
    assert artifact["model_used"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["forgetting_rate"] == 0.05
    assert artifact["reasoning_stability_score"] == 0.92
    assert artifact["status"] == "success"
    assert artifact["honest_verdict"] == "success"
    
    deliverable = tmp_path / "results" / "experiment_1762_stability.json"
    assert deliverable.exists()
