import os
import json
from pathlib import Path
from scripts.experiment_2506_retro_v241 import run_retro

def test_retro_generation():
    data = run_retro()
    
    assert data["n_experiments_completed"] == 8
    assert data["n_missing"] == 1
    assert data["n_blocked"] == 1
    
    assert data["best_241_auroc"] == 0.975
    assert data["auroc_adversarially_verified"] is True
    assert data["phase4_validated_any"] is False
    assert data["arxiv_ready"] is False
    
    assert len(data["top_3_gaps_for_242"]) == 3
    assert len(data["top_3_successes"]) == 3
    assert data["honest_verdict"].startswith("complete:")
    
    assert os.path.exists("results/experiment_2506_retro_v241.json")
    assert os.path.exists("results/operational_retro_2026_05_241.json")
    
    # Check JSON validity
    with open("results/experiment_2506_retro_v241.json", "r") as f:
        loaded = json.load(f)
        assert loaded["schema"] == "carnot.operational_retro.v65"
        
    with open("results/operational_retro_2026_05_241.json", "r") as f:
        loaded = json.load(f)
        assert loaded["honest_verdict"] == data["honest_verdict"]
