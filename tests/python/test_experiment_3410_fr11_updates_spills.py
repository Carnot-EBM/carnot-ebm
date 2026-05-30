import os
import json
import sys
import pytest

# Add scripts to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

import experiment_3410_fr11_updates_spills

def test_detect_spills():
    """Test REQ-LEARN-030 spill detection"""
    outputs = ["has prior", "none"]
    spills = experiment_3410_fr11_updates_spills.detect_spills(outputs)
    assert len(spills) == 2
    assert spills[0] == 0.8
    assert spills[1] == 0.1

def test_update_constraint_templates():
    """Test REQ-LEARN-030 constraint updates"""
    failed = ["a", "b"]
    weights = [0.8, 0.2]
    updated = experiment_3410_fr11_updates_spills.update_constraint_templates(failed, weights)
    assert updated == 1

def test_calculate_scores():
    """Test REQ-LEARN-030 scoring"""
    retention, adaptation = experiment_3410_fr11_updates_spills.calculate_scores(2, 4)
    assert adaptation == 0.5
    assert retention == 1.0 - (0.5 * 0.1)

    r, a = experiment_3410_fr11_updates_spills.calculate_scores(0, 0)
    assert r == 1.0
    assert a == 0.0

def test_run_experiment(tmp_path):
    """Test SCENARIO-LEARN-030 end-to-end experiment run"""
    out_file = tmp_path / "test_out.json"
    result = experiment_3410_fr11_updates_spills.run_experiment(str(out_file))
    
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert data["experiment_id"] == "3410"
    assert data["model_specs"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert "retention_score" in data["metrics"]
    assert "adaptation_score" in data["metrics"]
    assert data["details"]["updated_templates"] == 2
