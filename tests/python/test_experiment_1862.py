"""Tests for Experiment 1862: E2E S2KAN Verifier."""

import json
import math
from pathlib import Path

from carnot.pipeline.experiment_1862 import run_s2kan_verifier_e2e, write_experiment_1862_artifact

def test_run_s2kan_verifier_e2e():
    """Test the S2KAN verifier E2E proxy."""
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    # sin(pi/2) = 1.0
    result = run_s2kan_verifier_e2e(model_id, math.pi / 2)
    assert result["model_used"] == model_id
    assert abs(float(result["verified_output"]) - 1.0) < 1e-3  # type: ignore
    assert result["constraints_satisfied"] is True

def test_write_experiment_1862_artifact(tmp_path):
    """Test generating the artifact for experiment 1862."""
    output_path = tmp_path / "experiment_1862_e2e.json"
    artifact = write_experiment_1862_artifact(output_path)
    
    assert artifact["schema"] == "carnot.s2kan.experiment_1862.v1"
    assert artifact["status"] == "complete"
    assert artifact["model_used"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["constraints_satisfied"] is True
    
    assert Path(output_path).exists()
    with open(output_path, "r") as f:
        data = json.load(f)
        assert data["experiment_id"] == 1862

def test_write_experiment_1862_artifact_default_path(tmp_path, monkeypatch):
    """Test writing the artifact to the default path."""
    import carnot.pipeline.experiment_1862
    
    # Mock the root path so we don't write to the real results folder during tests
    def mock_repo_root():
        return tmp_path
    monkeypatch.setattr(carnot.pipeline.experiment_1862, "_repo_root", mock_repo_root)
    
    artifact = write_experiment_1862_artifact()
    assert artifact["status"] == "complete"
    assert (tmp_path / "results" / "experiment_1862_e2e.json").exists()

