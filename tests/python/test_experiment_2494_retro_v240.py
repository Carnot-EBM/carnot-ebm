import json
import os
import pytest
from scripts.experiment_2494_retro_v240 import generate_retro

def test_generate_retro(tmp_path, monkeypatch):
    # Mock files
    cwd = tmp_path
    monkeypatch.chdir(cwd)
    os.makedirs("results", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    with open("results/experiment_2493_capstone_v240.json", "w") as f:
        json.dump({
            "best_240_auroc": 0.975,
            "auroc_adversarially_verified": False,
            "phase4_validated_any": False,
            "arxiv_ready": False
        }, f)
        
    with open("docs/roadmap.md", "w") as f:
        f.write("| Milestone | Status | # Experiments | Note |\n")
        
    data = generate_retro()
    
    assert data["n_experiments_completed"] == 11
    assert data["best_240_auroc"] == 0.975
    assert len(data["top_3_gaps_for_241"]) == 3
    assert data["honest_verdict"].startswith("complete:")
    
    assert os.path.exists("results/experiment_2494_retro_v240.json")
    assert os.path.exists("results/operational_retro_2026_05_240.json")
    
    with open("docs/roadmap.md", "r") as f:
        content = f.read()
        assert "2026.05.240" in content
