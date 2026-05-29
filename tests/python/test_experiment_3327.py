import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.experiment_3327_energy_descent_substrate_bootstrap_v1 import main

def test_blocked_no_sota_models(tmp_path, monkeypatch):
    """Test that missing SOTA models blocks the experiment."""
    # REQ-INFER-SOTA-3327, SCENARIO-INFER-SOTA-3327-001
    deliverable = tmp_path / "results" / "experiment_3327_energy_descent_substrate_bootstrap_v1.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    
    with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", return_value=None):
        with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup"):
            main()
        
    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["status"] == "blocked"
    assert "missing_sota_cache" in data["blocked_reasons"]
    assert data["energy_descent_bootstrap_ready"] is False
    assert data["honest_verdict"] == "blocked_missing_sota_cache"


def test_successful_bootstrap_smoke(tmp_path, monkeypatch):
    """Test successful run of the bootstrap smoke."""
    # REQ-INFER-SOTA-3327, SCENARIO-INFER-SOTA-3327-001
    deliverable = tmp_path / "results" / "experiment_3327_energy_descent_substrate_bootstrap_v1.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_NO_SERVER", "1")  # skip real gpu load
    
    # Mock models
    mock_specs = [
        {"name": "Gemma4-31B-it", "hf_id": "unsloth/gemma-4-31B-it-GGUF", "gpu": 0, "model_path": "/fake/path/model1.gguf"},
    ]
    
    with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", return_value=mock_specs):
        with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1._run_smoke", return_value=(True, [1.0, 0.5], [0.1, 0.9])):
            with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup_gpu", return_value={"all_healthy": True}):
                with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup"):
                    main()
            
    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["status"] == "success"
    assert data["energy_descent_bootstrap_ready"] is True
    assert data["honest_verdict"] == "bootstrap_success"
    assert data["inference_substrate"] == "sota_gguf"
    assert "model_specs" in data
    assert "gpu_status" in data
    assert "random_seed" in data
    assert "duration_s" in data
    assert "reproducibility_checksum" in data
    assert "smoke_improvement_count" in data
    assert data["n_prompts"] == 8

def test_blocked_missing_required_sota_model(tmp_path, monkeypatch):
    """Test that missing the specific REQUIRED_MODELS blocks the experiment."""
    deliverable = tmp_path / "results" / "experiment_3327_energy_descent_substrate_bootstrap_v1.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    
    mock_specs = [
        {"name": "Some-Other-Model", "hf_id": "other/model", "gpu": 0, "model_path": "/fake/path/model1.gguf"},
    ]
    
    with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", return_value=mock_specs):
        with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup"):
            main()
            
    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["status"] == "blocked"
    assert "missing_required_sota_model" in data["blocked_reasons"]
    assert data["honest_verdict"] == "blocked_preconditions"


def test_blocked_gpu_setup_fails(tmp_path, monkeypatch):
    """Test that if gpu setup fails, it correctly blocks the experiment."""
    deliverable = tmp_path / "results" / "experiment_3327_energy_descent_substrate_bootstrap_v1.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    
    mock_specs = [
        {"name": "Gemma4-31B-it", "hf_id": "unsloth/gemma-4-31B-it-GGUF", "gpu": 0, "model_path": "/fake/path/model1.gguf"},
    ]
    
    with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", return_value=mock_specs):
        with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup_gpu", side_effect=RuntimeError("GPU missing")):
            with patch("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup"):
                main()
                
    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["status"] == "blocked"
    assert any("gpu_setup_failed" in reason for reason in data["blocked_reasons"])
    assert data["honest_verdict"] == "blocked_gpu_setup"
