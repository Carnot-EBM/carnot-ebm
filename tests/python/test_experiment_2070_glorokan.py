import json
import os
import pytest
import jax.numpy as jnp
from carnot.models.kan.glorokan import CarnotKAN, GloroKANVerifier, run_experiment_2070

def test_carnot_kan_lipschitz_bound():
    """
    Test that GloroKANVerifier correctly computes a positive Lipschitz bound
    for a synthetic CarnotKAN constraint system (REQ-KAN-2070).
    """
    model = CarnotKAN(num_knots=10, degree=3)
    verifier = GloroKANVerifier(model)
    bound = verifier.local_lipschitz_bound()
    
    assert bound > 0.0, "Lipschitz bound should be positive for non-flat splines."

def test_carnot_kan_lipschitz_bound_zero():
    """
    Test that GloroKANVerifier returns 0.0 when there are less than 2 control points.
    """
    model = CarnotKAN(num_knots=1, degree=3)
    verifier = GloroKANVerifier(model)
    bound = verifier.local_lipschitz_bound()
    
    assert bound == 0.0, "Lipschitz bound should be 0.0 for less than 2 control points."

def test_experiment_2070_artifact():
    """
    Test that the experiment 2070 JSON artifact is generated correctly with all
    required schema fields (SCENARIO-KAN-2070).
    """
    artifact_path = "results/experiment_2070_glorokan.json"
    if os.path.exists(artifact_path):
        os.remove(artifact_path)
        
    results = run_experiment_2070()
    
    assert os.path.exists(artifact_path), "Artifact JSON must be written."
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "experiment_2070"
    assert data["status"] == "complete"
    assert data["experiment_id"] == "2070"
    assert "REQ-KAN-2070" in data["spec_traces"]
    assert "local_lipschitz_bound" in data
    assert data["synthetic_constraint_verified"] is True
    assert "honest_verdict" in data
    assert "success" in data["honest_verdict"]
