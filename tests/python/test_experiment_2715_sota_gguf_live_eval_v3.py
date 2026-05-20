import json
import os
import pytest

def test_experiment_2715_output_json():
    # Verify that the artifact exists and has the correct fields
    result_path = "results/experiment_2715_sota_gguf_live_eval_v3.json"
    assert os.path.exists(result_path), "JSON deliverable does not exist"
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    required_keys = [
        "honest_verdict", "inference_mode", "n_examples_run",
        "energy_score_distribution", "fast_path_rate", "model_used",
        "models_checked", "cuda_available", "gpu_names", "random_seed",
        "reproducibility_checksum", "duration_s", "preconditions_checked"
    ]
    
    for key in required_keys:
        assert key in data, f"Missing required field: {key}"
        
    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith("blocked_")
    assert data["inference_mode"] in ["live_gpu", "live_cpu", "smoke_only"]
    assert data["random_seed"] == 42
    assert "mean" in data["energy_score_distribution"]
    assert "std" in data["energy_score_distribution"]
    assert "min" in data["energy_score_distribution"]
    assert "max" in data["energy_score_distribution"]
    assert "p25" in data["energy_score_distribution"]
    assert "p75" in data["energy_score_distribution"]
    assert isinstance(data["n_examples_run"], int)
    assert isinstance(data["fast_path_rate"], float)
    assert isinstance(data["cuda_available"], bool)
    
    # Test passed
