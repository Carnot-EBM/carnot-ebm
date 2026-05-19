import os
import json
from scripts.experiment_2517_retro_v242 import run_retro

def test_retro_generation():
    data = run_retro()
    
    assert data["n_experiments_completed"] == 6
    assert data["best_242_auroc"] == 0.975
    assert data["phase4_validated_any"] is True
    assert data["arxiv_ready"] is True
    
    assert len(data["top_3_successes"]) == 3
    assert len(data["top_3_gaps_for_243"]) == 3
    assert data["honest_verdict"].startswith("complete:")
    assert data["schema"] == "carnot.operational_retro.v66"
    
    assert os.path.exists("results/experiment_2517_retro_v242.json")
    assert os.path.exists("results/operational_retro_2026_05_242.json")
    
    with open("results/experiment_2517_retro_v242.json", "r") as f:
        loaded = json.load(f)
        assert loaded["schema"] == "carnot.operational_retro.v66"
        assert loaded["honest_verdict"] == data["honest_verdict"]
