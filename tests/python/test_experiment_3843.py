import json
import os
import sys
from pathlib import Path

# Add scripts directory to path to import run_experiment_3843
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import run_experiment_3843

def test_experiment_3843_generation():
    """
    REQ-CAPSTONE-3843: The .353 milestone must be aggregated into a capstone artifact using the Reading-Results discipline.
    SCENARIO-CAPSTONE-3843: The final artifact must accurately state the honest verdict, and paper ready boolean.
    """
    result = run_experiment_3843.run_experiment()
    
    assert "field_provenance" in result
    assert "milestone_summary" in result["field_provenance"]
    assert "honest_verdict" in result
    
    verdict = result["honest_verdict"]
    assert "complete:" in verdict
    assert "capstone_v353" in verdict
    assert "formal_core_CONFIRMED" in verdict
    assert "clean_core_certified_weak" in verdict
    assert "learned_characterized" in verdict
    assert "tier4_viable" in verdict
    assert "edlm_kill_gate_blocked_not_seeded" in verdict
    assert "paper_ready_true" in verdict
    assert "frozen_headline_unchanged" in verdict
    assert "both_energy_routes_bounded" in verdict

    assert result["field_provenance"]["paper_ready"] == "the standing convergence invariant — MUST be true"
    assert result["field_provenance"]["frozen_fover_auroc_unchanged"] == "0.9131 must not have moved"
    assert result["field_provenance"]["both_energy_routes_bounded"] == "the standing strategic conclusion — unchanged this milestone"

    forbidden = result["paper_v6_forbidden_claims"]
    assert "no energy-as-generator beats-AR claim" in forbidden
    assert "no energy-as-selector beats-AR claim" in forbidden
    assert "no KV260 speedup at d in {128,256}" in forbidden
    assert "verifier scoped to the measured math corpora" in forbidden
    
    assert "3836" in str(result["flagged_artifacts_skipped"])
