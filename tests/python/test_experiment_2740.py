import os
import json
import pytest
from experiment_2740 import get_preconditions, run_experiment

def test_preconditions_logic():
    preconditions = get_preconditions()
    assert isinstance(preconditions, list)
    assert len(preconditions) == 4
    for p in preconditions:
        assert 'resource' in p
        assert 'available' in p
        assert 'check' in p

def test_run_experiment_creates_file(tmp_path, monkeypatch):
    # Change working directory so results dir goes to tmp_path
    monkeypatch.chdir(tmp_path)
    run_experiment()
    
    result_path = tmp_path / "results" / "experiment_2740_verifier_energy_debug_v2_live_gpu.json"
    assert result_path.exists(), "JSON artifact must be created"
    
    with open(result_path) as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("blocked_") or data["honest_verdict"].startswith("complete:")
    assert "verifier_discriminative" in data
    assert "model_specs" in data
    assert "preconditions_checked" in data
