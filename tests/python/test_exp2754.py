import os
import sys

# Ensure python directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from results.experiment_2754_empirical_delta_audit import run_diagnostic

def test_diagnostic_runs_and_returns_correct_fields():
    # Remove the json if it exists so we can verify it gets created
    json_path = "results/experiment_2754_empirical_delta_audit.json"
    if os.path.exists(json_path):
        os.remove(json_path)

    # Run it
    run_diagnostic()

    assert os.path.exists(json_path)

    import json
    with open(json_path, "r") as f:
        data = json.load(f)

    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith("blocked_")
    assert "delta_root_cause_identified" in data
    assert data["root_cause"] in ["H1 definitional", "H2 regression", "H3 ceiling", "unclear"]
    assert "empirical_delta_diagnostic" in data
    assert "n_attempts_total" in data
    assert "n_successes" in data
    assert "n_repair_attempts_definition" in data
    assert "paper_v6_recommendation" in data
    assert "random_seed" in data
    assert "duration_s" in data
    assert "preconditions_checked" in data
