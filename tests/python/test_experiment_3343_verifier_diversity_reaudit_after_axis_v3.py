"""
Tests for REQ-VERIFY-3343: Cached-Candidate Verifier Diversity Re-Audit After Axis V3
"""

import sys
import os
import json
from pathlib import Path
import resource

import pytest
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.experiment_3343_verifier_diversity_reaudit_after_axis_v3 import main, compute_metrics

def test_compute_metrics():
    # Test compute_metrics function
    np.random.seed(42)
    scores = np.random.binomial(1, 0.5, size=(100, 3)).astype(float)
    lambda_min, k, cov = compute_metrics(scores)
    assert isinstance(lambda_min, float)
    assert isinstance(k, float)
    assert cov.shape == (3, 3)
    assert lambda_min > 0
    assert k > 1.0

def test_experiment_3343_writes_expected_artifact(tmp_path, monkeypatch):
    """
    Test that the experiment script writes the expected JSON artifact
    and satisfies REQ-VERIFY-3343.
    """
    # Create mock 3342 artifact
    mock_3342_path = tmp_path / "results" / "experiment_3342_monitor_provenance_verifier_axis_v1.json"
    mock_3342_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_3342_path, "w") as f:
        json.dump({"monitor_provenance_axis_ready": True}, f)
        
    # Override repo root to tmp_path
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    
    # We must also patch _get_repo_root so it returns tmp_path
    import scripts.experiment_template as ext
    monkeypatch.setattr(ext, "_get_repo_root", lambda: tmp_path)
    
    # Run the experiment
    main()
    
    artifact_path = tmp_path / "results" / "experiment_3343_verifier_diversity_reaudit_after_axis_v3.json"
    assert artifact_path.exists(), "Deliverable JSON must be written"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert "experiment" in data
    assert data["experiment"] == 3343
    assert data["n_cases"] >= 1000
    
    required_fields = [
        "honest_verdict", "inference_substrate", "random_seed", 
        "reproducibility_checksum", "duration_s", "files_updated",
        "n_cases", "lambda_min_sigma_before", "lambda_min_sigma_after",
        "effective_k_before", "effective_k_after", "delta_lambda_min_sigma",
        "delta_effective_k", "collapsed_pairs_after", "diversity_remediation_passed", "blocked_reasons"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
        
    assert data["inference_substrate"] == "cpu"
    assert isinstance(data["collapsed_pairs_after"], list)
    assert data["diversity_remediation_passed"] in [True, False]
    
    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith("failed:")
    
def test_experiment_3343_fails_if_3342_not_ready(tmp_path, monkeypatch):
    # Create mock 3342 artifact but not ready
    mock_3342_path = tmp_path / "results" / "experiment_3342_monitor_provenance_verifier_axis_v1.json"
    mock_3342_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_3342_path, "w") as f:
        json.dump({"monitor_provenance_axis_ready": False}, f)
        
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    import scripts.experiment_template as ext
    monkeypatch.setattr(ext, "_get_repo_root", lambda: tmp_path)
    
    with pytest.raises(RuntimeError, match="experiment 3342 axis not ready"):
        main()
