"""Tests for S2KAN Z3 Verification."""

import json
from pathlib import Path
import z3

from carnot.models.s2kan_z3 import verify_s2kan_bounds

def test_s2kan_z3_verification(tmp_path):
    """Test Z3 verification of S2KAN over a bounded domain.
    
    Spec references: REQ-KAN-1859, SCENARIO-KAN-1859
    """
    gate_probs = [0.8, 0.1, 0.1]
    input_lb = -1.0
    input_ub = 1.0
    
    result = verify_s2kan_bounds(gate_probs, input_lb, input_ub)
    
    assert result["status"] == "complete"
    assert result["is_consistent"] is True
    assert "output_lb" in result
    assert "output_ub" in result
    assert result["output_lb"] <= result["output_ub"]

def test_s2kan_z3_transpilation_output(tmp_path):
    """Test generating the artifact for experiment 1859."""
    from carnot.models.s2kan_z3 import write_experiment_1859_artifact
    
    output_path = tmp_path / "experiment_1859_z3_verify.json"
    artifact = write_experiment_1859_artifact(output_path)
    
    assert artifact["schema"] == "carnot.s2kan.experiment_1859.v1"
    assert artifact["status"] == "complete"
    assert artifact["is_consistent"] is True
    assert Path(output_path).exists()
    with open(output_path, "r") as f:
        data = json.load(f)
        assert data["experiment_id"] == 1859
