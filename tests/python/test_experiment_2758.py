import json
import os
from experiment_2758 import run

def test_experiment_2758_produces_valid_json():
    # Run the experiment
    run()
    
    # Check the json
    json_path = "results/experiment_2758_weak_strong_policy_fix_v2.json"
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith("blocked_")
    
    if data["honest_verdict"].startswith("complete:"):
        assert data["thresholds_correct"] is True
        assert data["policy_viable_v2"] is True
        assert data["t_low_fixed"] < data["t_high_fixed"]
        assert 20.0 <= data["policy_savings_pct_v2"] <= 60.0
        assert data["false_negative_rate_v2"] <= 0.10
        assert data["fix_method"] in ["label_inversion", "isotonic_monotone"]
        assert data["random_seed"] == 42
