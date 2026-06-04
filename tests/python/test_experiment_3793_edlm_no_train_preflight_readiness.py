import os
import json
import pytest
from scripts.experiment_3793_edlm_no_train_preflight_readiness import run_preflight

def test_run_preflight(tmp_path, monkeypatch):
    """
    Test the preflight script for the EDLM next thesis.
    Checks that the JSON output conforms to the requirements.
    """
    # Change working directory so results/ goes into tmp_path
    monkeypatch.chdir(tmp_path)
    os.makedirs("results", exist_ok=True)
    
    run_preflight()
    
    result_path = "results/experiment_3793_edlm_no_train_preflight_readiness.json"
    assert os.path.exists(result_path), "Preflight script must write to results directory"
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    assert data["readiness_verdict"] == "go"
    assert "edlm_no_train_preflight_go" in data["honest_verdict"]
    assert data["loop_does_not_commit"] is True
    assert data["reference_impl_fetchable"] is True
    assert data["minimal_kill_gate_sound"] is True
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert data["compute_estimate_gpu_hours"] == 8
    
    prereqs = data["prerequisites_obtainable"]
    assert prereqs["torch_available"] is True
    assert prereqs["diffusion_lib_available"] is True
    assert prereqs["ar_base_fetchable"] is True
    assert prereqs["tiny_corpus_fetchable"] is True
    
    assert "git clone" in data["operator_seed_command"]
