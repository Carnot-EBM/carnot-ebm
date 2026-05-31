import json
import os
from scripts.experiment_3571_capstone_v328 import run_capstone

def test_run_capstone():
    """Test the capstone aggregation script for v328."""
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    
    artifact = run_capstone()
    
    # Check the required schema fields
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert "inference_substrate" in artifact
    assert "experiments_completed" in artifact
    assert "key_finding" in artifact
    assert "p0_1_status" in artifact
    assert artifact["p0_1_status"] in ["TERMINAL_POSITIVE_GENERALIZED", "TERMINAL_POSITIVE_GRAPH_COLORING_ONLY", "OPEN"]
    assert "route1_second_csp_verdict" in artifact
    assert "route1_robust_verdict" in artifact
    assert "route2_nlmath_terminal_verdict" in artifact
    assert "aggregation_secondary_headline_confirmed" in artifact
    assert "self_learning_p02_verdict" in artifact
    assert "new_secondary_headlines" in artifact
    assert "unmet_gates" in artifact
    assert "paper_v6_safe_claims" in artifact
    assert "paper_v6_forbidden_claims" in artifact
    assert "top_forward_gap" in artifact
    assert "capstone_v328_ready" in artifact
    assert artifact["capstone_v328_ready"] is True
    assert "random_seed" in artifact
    assert artifact["random_seed"] == 20260601
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
    
    # Check that it writes the JSON file
    assert os.path.exists("results/experiment_3571_capstone_v328.json")
    with open("results/experiment_3571_capstone_v328.json", "r") as f:
        saved_artifact = json.load(f)
    assert saved_artifact["random_seed"] == 20260601
    assert saved_artifact["capstone_v328_ready"] is True
