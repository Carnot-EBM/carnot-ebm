import json
import os
import pytest
from run_experiment_1911 import run

def test_experiment_1911_schema():
    # REQ-PHASE4-CANONICAL-DECISION
    # SCENARIO-DECISION-ARTIFACT-GENERATION
    run()
    assert os.path.exists("results/experiment_1911_phase4_canonical_decision.json")
    with open("results/experiment_1911_phase4_canonical_decision.json", "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.phase4_canonical_decision.v2"
    assert data["experiment"] == 1911
    assert data["honest_verdict"].startswith("success:")
    assert data["acceptance_gate_passed"] is True
    assert "preconditions_checked" in data
    assert "Fast-Slow Variant" in data["canonical_metric_named"]
