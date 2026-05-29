"""
Tests for REQ-VERIFY-3329: Cached-Candidate Verifier Ensemble Diversity Audit V2
"""

import sys
import os
import json
from pathlib import Path

import pytest
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.experiment_3329_verifier_ensemble_diversity_audit_v2 import main

def test_experiment_3329_writes_expected_artifact(tmp_path, monkeypatch):
    """
    Test that the experiment script writes the expected JSON artifact
    and satisfies REQ-VERIFY-3329.
    """
    # Override repo root to tmp_path
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    
    # Run the experiment
    main()
    
    artifact_path = tmp_path / "results" / "experiment_3329_verifier_ensemble_diversity_audit_v2.json"
    assert artifact_path.exists(), "Deliverable JSON must be written"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert "experiment" in data
    assert data["experiment"] == 3329
    assert data["n_cases"] >= 1000, "REQ-VERIFY-3329-3: Must have >= 1000 cases"
    
    # REQ-VERIFY-3329-5 fields
    required_fields = [
        "honest_verdict", "inference_substrate", "random_seed", 
        "reproducibility_checksum", "duration_s", "n_cases", 
        "verifier_names", "covariance_methodology", "lambda_min_sigma", 
        "effective_k", "diversity_gate_passed", 
        "verifier_diversity_audit_v2_ready", "collapsed_pairs", "blocked_reasons"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
        
    assert data["inference_substrate"] == "cpu", "REQ-VERIFY-3329-2: Must not require live LLM inference"
    assert isinstance(data["collapsed_pairs"], list)
    assert data["diversity_gate_passed"] in [True, False]
    
    assert data["honest_verdict"] in ["usable for Phase-3 authority", "usable only as diagnostics", "collapsed"]

