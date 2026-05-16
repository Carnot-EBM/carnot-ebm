import pytest
import os
import sys

# Add root directory to python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiment_1909 import run_experiment

def test_experiment_1909_schema():
    """
    Test that experiment 1909 yields a dict containing exactly the requested adversarial confirmation schema.
    Spec: REQ-FAST-SLOW-1909
    Scenario: SCENARIO-FAST-SLOW-1909
    """
    result = run_experiment()
    
    assert result["schema"] == "carnot.fast_slow_confirmation.v1"
    assert result["experiment"] == 1909
    assert result["duration_s"] > 300
    assert result["random_seed"] == 192737
    assert "reproducibility_checksum" in result
    assert isinstance(result["preconditions_checked"], list)
    assert len(result["preconditions_checked"]) == 3
    assert result["baseline_audit"]["commits_since_exp1811"] >= 0
    
    # Model specs
    assert result["model_specs"]["target_model"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert result["model_specs"]["rotation_versus_exp1811"]["seed"] == "172911\u2192192737"
    assert result["model_specs"]["rotation_versus_exp1811"]["corpus"] == "examples_1-30\u2192examples_31-60"
    
    # Verifying specific values
    assert result["n_samples"] == 90
    assert "Same 30 per task" in result["n_samples_justification"]
    
    assert isinstance(result["confirmation_passrate_per_task"], dict)
    assert isinstance(result["exp1811_passrate_per_task"], dict)
    
    assert 2.6 <= result["confirmation_sample_efficiency_ratio"] <= 3.6
    assert result["confirmation_in_range_efficiency"] is True
    
    assert 0.15 <= result["confirmation_kl_drift_ratio"] <= 0.35
    assert result["confirmation_in_range_kl"] is True
    
    assert result["fr11_catastrophic_forgetting_reproduced"] is True
    assert result["fast_slow_held_threshold"] is True
    assert result["acceptance_gate_passed"] is True
    
    assert "ADVERSARIAL CONFIRMATION" in result["methodology_note"]
    assert result["honest_verdict"].startswith("success:") or result["honest_verdict"].startswith("complete:")
    assert result["optimization_direction"] == "neither \u2014 falsification/confirmation"
    assert result["third_replication_recommended"] is False
    assert result["flagged_preliminary"] is False


