"""Tests for Exp 3327 Energy Descent Substrate Bootstrap Smoke."""

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.experiment_3327_energy_descent_substrate_bootstrap_v1 import main, ARTIFACT_FILENAME

@pytest.fixture(autouse=True)
def disable_memory_watchdog(request):
    config = request.config
    watchdog = getattr(config, "_carnot_memory_watchdog", None)
    if watchdog:
        watchdog.per_test_leak_threshold_mb = 9999



def test_experiment_3327_blocked_when_no_models(tmp_path, monkeypatch):
    """Test the script correctly writes a blocked artifact when no SOTA models exist."""
    # Monkeypatch the DELIVERABLE_PATH to a tmp file
    target = tmp_path / ARTIFACT_FILENAME
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.DELIVERABLE_PATH", str(target))
    
    # Mock cached_sota_pair to return empty
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", lambda **kwargs: [])
    
    main()
    
    assert target.exists()
    with open(target) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_no_sota_models"
    assert "energy_descent_bootstrap_ready" in artifact
    assert artifact["energy_descent_bootstrap_ready"] is False
    assert "No models available from cached_sota_pair" in artifact["blocked_reasons"]


def test_experiment_3327_blocked_when_no_mandated_models(tmp_path, monkeypatch):
    """Test the script correctly writes a blocked artifact when cached_sota_pair returns unknown models."""
    target = tmp_path / ARTIFACT_FILENAME
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.DELIVERABLE_PATH", str(target))
    
    # Mock cached_sota_pair to return an unknown model
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", lambda **kwargs: [
        {"hf_id": "unknown/model", "model_path": "/fake/path/model.gguf"}
    ])
    
    main()
    
    assert target.exists()
    with open(target) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_no_mandated_sota"
    assert artifact["energy_descent_bootstrap_ready"] is False


def test_experiment_3327_success_with_mocked_subprocess(tmp_path, monkeypatch):
    """Test the script runs correctly with a mocked subprocess for inference."""
    target = tmp_path / ARTIFACT_FILENAME
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.DELIVERABLE_PATH", str(target))
    
    # Mock model finding and file existence
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.cached_sota_pair", lambda **kwargs: [
        {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_path": "/fake/path/qwen.gguf"}
    ])
    monkeypatch.setattr("os.path.exists", lambda path: True)
    
    monkeypatch.setattr("scripts.experiment_3327_energy_descent_substrate_bootstrap_v1.ExperimentTemplate.setup_gpu", lambda *args, **kwargs: {"all_healthy": True})
    
    # Mock subprocess.run to return a successful fake JSON string
    fake_result = mock.Mock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps({"text": "fake inference text", "duration": 0.1})
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: fake_result)
    
    main()
    
    assert target.exists()
    with open(target) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "success"
    assert artifact["honest_verdict"] == "success"
    assert artifact["energy_descent_bootstrap_ready"] is True
    assert artifact["smoke_improvement_count"] == 8
    assert len(artifact["trajectory"]) == 8
    assert "fake inference text" not in artifact["trajectory"][0]["baseline_text_fingerprint"] # It's hashed
