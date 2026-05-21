import json
import pytest
from pathlib import Path

DELIVERABLE_PATH = Path("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_2439_fr11_online_learnability.json")
PATTERNS_PATH = Path("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/constraint_patterns_v4.json")

def test_experiment_2439_deliverable_schema():
    assert DELIVERABLE_PATH.exists(), "Deliverable JSON was not generated."
    
    with open(DELIVERABLE_PATH, "r") as f:
        data = json.load(f)
        
    assert data.get("honest_verdict", "").startswith("Terminal-prefix required.")
    assert data.get("fr11_online_learnability_passed") is True
    assert 0.0 <= data.get("soundness_rate", -1) <= 1.0
    assert 0.0 <= data.get("completeness_rate", -1) <= 1.0
    assert "estimated_littlestone_dim" in data
    assert data.get("cross_domain_retention_rate", -1) >= 0.50
    assert data.get("n_eval_examples") == 30
    assert data.get("random_seed") == 42
    assert "duration_s" in data
    assert data.get("preconditions_checked") is True

def test_experiment_2439_patterns_updated():
    assert PATTERNS_PATH.exists(), "Constraint patterns were not updated."
    with open(PATTERNS_PATH, "r") as f:
        patterns_data = json.load(f)
        
    patterns = patterns_data.get("patterns", [])
    # We should have some patterns or at least the file is valid JSON
    assert isinstance(patterns, list)
    for p in patterns:
        assert "pattern" in p
        assert "confidence" in p
