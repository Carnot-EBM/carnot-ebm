import os
import json
from scripts.experiment_2529_retro_v243 import run_retro


def test_retro_generation():
    data = run_retro()

    assert data["n_experiments_completed"] == 7
    assert data["best_243_auroc"] == 0.975
    assert data["phase4_final_status"] == "blocked_precondition"
    assert data["arxiv_ready"] is False
    assert data["operator_recommendation"] == "request_phase4_operator_decision"

    assert len(data["top_3_successes"]) == 3
    assert len(data["top_3_gaps_for_244"]) == 3
    assert data["honest_verdict"].startswith("complete:")
    assert data["schema"] == "carnot.operational_retro.v67"

    assert os.path.exists("results/experiment_2529_retro_v243.json")
    assert os.path.exists("results/operational_retro_2026_05_243.json")

    with open("results/experiment_2529_retro_v243.json", "r") as f:
        loaded = json.load(f)
        assert loaded["schema"] == "carnot.operational_retro.v67"
        assert loaded["honest_verdict"] == data["honest_verdict"]
