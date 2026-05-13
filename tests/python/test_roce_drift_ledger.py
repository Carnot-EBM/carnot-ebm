"""
Tests for the ROCE Residual Drift Ledger.
"""

import json
from pathlib import Path

from carnot.verifiers.roce_drift_ledger import (
    ResidualDriftLedger,
    RoceValidatorTree,
    compile_roce_validator_trees,
    generate_experiment_artifact
)

def test_compile_roce_validator_trees():
    """Test compiling raw constraints into validator trees."""
    trees = compile_roce_validator_trees([["c1", "c2"], ["c3"]])
    assert len(trees) == 2
    assert trees[0].constraints == ["c1", "c2"]
    assert trees[1].constraints == ["c3"]

def test_residual_drift_ledger_extract_constraints():
    """Test extracting constraints (REQ-ROCE-001)."""
    trees = compile_roce_validator_trees([["c1", "c2"], ["c1", "c3"]])
    ledger = ResidualDriftLedger()
    ledger.extract_constraints(trees)
    assert ledger.constraints == {"c1", "c2", "c3"}

def test_residual_drift_ledger_record_drift_case():
    """Test recording multi-turn metrics (REQ-ROCE-002)."""
    ledger = ResidualDriftLedger()
    ledger.record_drift_case("t1", 2)
    ledger.record_drift_case("t2", 3)
    assert ledger.tracking_metrics["t1"]["drift_count"] == 2
    assert ledger.tracking_metrics["t2"]["drift_count"] == 3
    assert ledger.total_drift_cases == 5

def test_residual_drift_ledger_write_artifact(tmp_path: Path):
    """Test writing artifact and zero_false_accepts logic (REQ-ROCE-003)."""
    ledger = ResidualDriftLedger(zero_false_accepts=True)
    ledger.record_drift_case("t1", 1)
    
    out_file = tmp_path / "artifact.json"
    ledger.write_artifact(str(out_file))
    
    with open(out_file, "r") as f:
        data = json.load(f)
    
    assert data["zero_false_accepts"] is True
    assert data["metrics"]["t1"]["drift_count"] == 1
    assert data["total_drift_cases"] == 1

def test_generate_experiment_artifact(tmp_path: Path):
    """Test end-to-end artifact generation."""
    out_file = tmp_path / "experiment_1992_residual_drift_ledger.json"
    generate_experiment_artifact(str(out_file))
    
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert data["zero_false_accepts"] is True
    assert data["total_drift_cases"] == 7
    assert len(data["constraints"]) == 3
