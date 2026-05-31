"""
Tests for Experiment 3574: Verifier Factual Hallucination Error Detection
Spec references: REQ-VERIFY-3574, SCENARIO-VERIFY-3574
"""

import json
import os
import sys

# Add scripts directory to path to import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from experiment_3574_verifier_factual_hallucination_error_detection import run_experiment

def test_experiment_creates_valid_json(tmp_path, monkeypatch):
    """
    Test that the experiment runs and creates a valid JSON output
    with the expected factual hallucination metrics.
    
    Ref: REQ-VERIFY-3574
    """
    # Run the experiment
    run_experiment()
    
    # Check that the result file was created
    result_path = "results/experiment_3574_verifier_factual_hallucination_error_detection.json"
    assert os.path.exists(result_path), f"Result file {result_path} was not created"
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    # Verify required schema fields
    required_fields = [
        "honest_verdict",
        "inference_substrate",
        "ensemble_factual_error_detection_auroc",
        "best_single_verifier_auroc",
        "model_confidence_baseline_auroc",
        "ensemble_minus_best_baseline_delta",
        "per_verifier_auroc",
        "n_examples",
        "generalizes_to_facts",
        "constraint_verifiers_inert_on_facts",
        "random_seed",
        "reproducibility_checksum",
        "duration_s"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
        
    # Verify types and constraints
    assert data["n_examples"] >= 100
    assert "SemanticConsistencyVerifier" in data["per_verifier_auroc"]
    assert "IsingVerifier" in data["per_verifier_auroc"]
    
    # Verify the baseline AUROC is realistic
    assert 0.0 <= data["model_confidence_baseline_auroc"] <= 1.0
