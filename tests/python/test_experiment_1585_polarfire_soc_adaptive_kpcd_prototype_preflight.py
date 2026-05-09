import json
import os
from pathlib import Path

def test_experiment_1585_polarfire_preflight_artifact():
    """
    SCENARIO-HW-050: PolarFire preflight artifact has required fields.
    """
    artifact_path = Path("results/experiment_1585_polarfire_soc_adaptive_kpcd_prototype_preflight.json")
    assert artifact_path.exists(), "Artifact must exist"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    expected_fields = [
        "status",
        "polarfire_board_available",
        "yosys_polarfire_available",
        "libero_available",
        "reusable_rtl_components_count",
        "preflight_note_path",
        "polarfire_preflight_ready",
        "blocked_reason",
        "no_board_execution_claim",
        "honest_verdict"
    ]
    
    for field in expected_fields:
        assert field in data, f"Missing required field: {field}"
        
    assert data["status"] == "in_progress"
    assert data["polarfire_board_available"] is False
    assert data["libero_available"] is False
    assert data["no_board_execution_claim"] is True
