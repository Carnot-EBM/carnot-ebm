"""Tests for FR-11 Epsilon tracker.

Spec traces: REQ-PIPELINE-1848, SCENARIO-PIPELINE-1848
"""
import os
import json
import numpy as np
from carnot.pipeline.fr11_epsilon import FR11EpsilonTracker

def test_fr11_epsilon_tracker_success(tmp_path):
    # SCENARIO-PIPELINE-1848
    tracker = FR11EpsilonTracker(parameter_dim=8)
    
    obj_grad = np.array([0.1, 0.2, 0.0, -0.1, 0.5, 0.0, 0.0, -0.2])
    const_grad = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    epsilon = 0.05
    
    tracker.enforce_checks_and_update(obj_grad, const_grad, epsilon)
    assert tracker.utility_check_passed is True
    assert tracker.non_forgetting_check_passed is True
    
    out_file = str(tmp_path / "experiment_1848_gemma26_epsilon.json")
    artifact = tracker.write_experiment_artifact(out_file, ["unsloth/gemma-4-26B-A4B-it-GGUF"])
    
    assert artifact["experiment_id"] == "1848"
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"] == "epsilon_learning_success"
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in artifact["model_specs"]
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        data = json.load(f)
        assert data["utility_check_passed"] is True
        
def test_fr11_epsilon_tracker_fail_utility(tmp_path):
    tracker = FR11EpsilonTracker(parameter_dim=4)
    # Magnitude too small
    obj_grad = np.zeros(4)
    const_grad = np.ones(4)
    tracker.enforce_checks_and_update(obj_grad, const_grad, 0.1)
    assert tracker.utility_check_passed is False
    assert tracker.non_forgetting_check_passed is True
    
    out_file = str(tmp_path / "experiment_1848_fail.json")
    artifact = tracker.write_experiment_artifact(out_file, ["unsloth/gemma-4-26B-A4B-it-GGUF"])
    assert artifact["honest_verdict"] == "failed"

def test_fr11_epsilon_tracker_fail_non_forgetting(tmp_path):
    tracker = FR11EpsilonTracker(parameter_dim=4)
    obj_grad = np.array([0.1, 0.1, 0.1, 0.1])
    const_grad = np.zeros(4)
    
    # Force learning rate to 0 so parameters do not change
    tracker.pipeline.learning_rate = 0.0
    tracker.enforce_checks_and_update(obj_grad, const_grad, 0.0)
    assert tracker.utility_check_passed is True
    assert tracker.non_forgetting_check_passed is False
