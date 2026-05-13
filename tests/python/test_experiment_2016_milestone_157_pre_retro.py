"""Tests for Exp 2016 Milestone .157 Pre-Retro Audit."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path to allow importing the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from experiment_2016_milestone_157_pre_retro import audit_artifacts, main


def test_audit_artifacts_success(tmp_path: Path) -> None:
    """Test successful audit when all conditions are met."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(2008, 2016):
        data = {
            "experiment": i,
            "status": "success"
        }
        if i == 2013:
            data["continuous_self_learning_task"] = True
            
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "success"
    assert res["artifacts_exist"] is True
    assert res["valid_schema_confirmed"] is True
    assert res["continuous_learning_compliant"] is True
    assert "All .157 artifacts exist" in res["honest_verdict"]


def test_audit_artifacts_missing(tmp_path: Path) -> None:
    """Test audit when an artifact is missing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(2008, 2015):  # Missing 2015
        data = {
            "experiment": i,
            "status": "success",
            "continuous_self_learning_task": True
        }
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["artifacts_exist"] is False
    assert "Missing artifacts [2015]" in res["honest_verdict"]


def test_audit_artifacts_invalid_schema(tmp_path: Path) -> None:
    """Test audit when an artifact has an invalid schema."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(2008, 2016):
        data = {
            "experiment": i,
            "status": "success",
            "continuous_self_learning_task": True
        }
        if i == 2010:
            data = {"bad": "schema", "continuous_self_learning_task": True}
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["valid_schema_confirmed"] is False
    assert "Invalid schema" in res["honest_verdict"]


def test_audit_artifacts_no_compliance(tmp_path: Path) -> None:
    """Test audit when continuous self-learning compliance is not met."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(2008, 2016):
        data = {
            "experiment": i,
            "status": "success"
        }
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["continuous_learning_compliant"] is False
    assert "Continuous self-learning compliance not confirmed" in res["honest_verdict"]


def test_audit_artifacts_bad_json(tmp_path: Path) -> None:
    """Test audit when an artifact contains invalid JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(2008, 2016):
        data = {
            "experiment": i,
            "status": "success",
            "continuous_self_learning_task": True
        }
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            if i == 2011:
                f.write("not json")
            else:
                json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["valid_schema_confirmed"] is False


def test_main(tmp_path: Path) -> None:
    """Test main execution."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    with patch("os.environ", {"PROJECT_ROOT": str(tmp_path)}):
        main()
        
    out_path = results_dir / "experiment_2016_milestone_157_pre_retro.json"
    assert out_path.exists()
