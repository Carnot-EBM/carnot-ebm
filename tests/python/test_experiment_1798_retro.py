import os
import json
from pathlib import Path
from scripts.experiment_1798_retro import generate_retro

def test_generate_retro(tmp_path):
    """
    REQ-REPORT-1798: Milestone .138 Phase 4 Operations Retrospective
    SCENARIO-REPORT-1798: Exp 1798 Generates Phase 4 Synthesis Retrospective
    """
    # Create dummy experiment result JSONs
    for i in range(1785, 1798):
        data = {"honest_verdict": f"Success {i}"}
        # Add some details just to have fields
        with open(tmp_path / f"experiment_{i}_mock.json", "w") as f:
            json.dump(data, f)
    
    out_path = tmp_path / "experiment_1798_retro.json"
    
    generate_retro(str(tmp_path), str(out_path))
    
    assert out_path.exists()
    
    with open(out_path, "r") as f:
        data = json.load(f)
        
    assert data["milestone"] == "2026.05.138"
    assert "honest_verdict" in data
    assert "overall_verdict" in data
    
    for i in range(1785, 1798):
        assert f"experiment_{i}_mock.json" in data["honest_verdict"]
        assert data["honest_verdict"][f"experiment_{i}_mock.json"] == f"Success {i}"
