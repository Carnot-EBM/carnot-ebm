"""Tests for Exp 1989: p-dit Hardware Preflight and Preconditioning."""

import json
from pathlib import Path
from carnot.hardware.pdit_hardware_preflight import run_pdit_preflight, write_artifact

def test_pdit_preflight_artifact_schema(tmp_path: Path) -> None:
    """REQ-HW-057: artifact contains required resource accounting and claim bounds.
    
    SCENARIO-HW-057: executes and writes artifact mapping nodes to p-dits.
    """
    # 1. Test logic
    artifact = run_pdit_preflight(100)
    assert artifact["hardware_execution_claim"] is False
    assert artifact["resource_mapping"]["abstract_nodes"] == 100
    assert artifact["resource_mapping"]["p_dits_required"] == 25
    assert "kona_style_comparison_valid" in artifact["preconditioning_limits"]
    
    # 2. Test artifact generation
    out_file = tmp_path / "experiment_1989_p_dit_hardware_preflight.json"
    write_artifact(str(out_file))
    
    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        
    assert loaded["hardware_execution_claim"] is False
    assert loaded["honest_verdict"] == "p_dit_hardware_preflight_complete_no_hardware_execution_claim"
