import json
import os

def test_experiment_2013_sweep_output():
    # REQ-AUTO-SWEEP-2013
    assert os.path.exists("results/experiment_2013_citation_sweep_cot2meta.json")
    with open("results/experiment_2013_citation_sweep_cot2meta.json", "r") as f:
        data = json.load(f)
    assert data["schema"] == "carnot.routine_citation_sweep.v1"
    assert len(data["new_candidates"]) == 49
