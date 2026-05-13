"""Tests for the mouth_brain_audit module.

Spec: REQ-PIPELINE-2053, SCENARIO-PIPELINE-2053
"""

import json
from pathlib import Path

from carnot.verify.mouth_brain_audit import run_audit


def test_run_audit_returns_correct_schema():
    """Test that run_audit returns the expected dictionary schema and writes JSON."""
    out_path = Path("results/experiment_2053_mouth_brain_audit.json")
    if out_path.exists():
        out_path.unlink()
        
    result = run_audit()
    
    # Assert dictionary structure
    assert result["experiment_id"] == "2053"
    assert result["title"] == "Mouth/Brain Separation Audit"
    assert "findings" in result
    assert "rust_layer" in result["findings"]
    assert "python_layer" in result["findings"]
    assert "coupling_points" in result["findings"]
    assert "VerifyRepairPipeline._generate" in result["findings"]["coupling_points"]
    assert "recommendation" in result
    assert "honest_verdict" in result
    
    # Assert file was written and is valid JSON
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data == result
