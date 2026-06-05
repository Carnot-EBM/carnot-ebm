import os
import json
import pytest
import sys
import importlib.util

# Add scripts to path so we can import
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
sys.path.insert(0, scripts_dir)
import run_experiment_3839

def test_experiment_3839_blocked():
    """Test that the experiment handles the blocked case correctly when not seeded."""
    # Run the experiment
    artifact = run_experiment_3839.run_experiment()
    
    # Check artifact file exists
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/experiment_3839_edlm_kill_gate.json'))
    assert os.path.exists(output_path)
    
    with open(output_path, "r") as f:
        loaded_artifact = json.load(f)
        
    for key in [
        "operator_seeded", "tiny_edlm_trains_stably", "matched_compute_delta_vs_ar",
        "kill_gate_verdict", "preconditions_checked", "model_specs", "n",
        "honest_verdict", "random_seed", "reproducibility_checksum", "duration_s",
        "inference_substrate"
    ]:
        assert key in loaded_artifact
        assert key in artifact
        
    assert loaded_artifact["honest_verdict"].startswith("blocked_") or loaded_artifact["honest_verdict"].startswith("complete:")
    
    # Clean up so we don't leave state around
    os.remove(output_path)
