"""Tests for Exp 2006 Milestone .156 Pre-Retro Audit."""
import json
import os
import sys
from pathlib import Path

# Add scripts directory to path to allow importing the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from experiment_2006_milestone_156_pre_retro import audit_artifacts


def test_audit_artifacts_success(tmp_path: Path) -> None:
    """Test successful audit when all conditions are met."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(1996, 2006):
        data = {
            "experiment": i,
            "status": "success"
        }
        if i == 2000:
            data["models_utilized"] = ["unsloth/Qwen3.6-35B-A3B-GGUF"]
            
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "success"
    assert res["artifacts_exist"] is True
    assert res["valid_schema_confirmed"] is True
    assert res["sota_models_utilized"] is True
    assert "All .156 artifacts exist" in res["honest_verdict"]


def test_audit_artifacts_missing(tmp_path: Path) -> None:
    """Test audit when an artifact is missing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(1996, 2005):  # Missing 2005
        data = {
            "experiment": i,
            "status": "success",
            "models_utilized": ["unsloth/Qwen3.6"]
        }
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["artifacts_exist"] is False
    assert "Missing artifacts [2005]" in res["honest_verdict"]


def test_audit_artifacts_invalid_schema(tmp_path: Path) -> None:
    """Test audit when an artifact has an invalid schema."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(1996, 2006):
        data = {
            "experiment": i,
            "status": "success",
            "models_utilized": ["unsloth/Qwen3.6"]
        }
        if i == 2002:
            data = {"bad": "schema"}
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["valid_schema_confirmed"] is False
    assert "Invalid schema" in res["honest_verdict"]


def test_audit_artifacts_no_sota(tmp_path: Path) -> None:
    """Test audit when SOTA models are not utilized in any artifact."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(1996, 2006):
        data = {
            "experiment": i,
            "status": "success",
            "models_utilized": ["basic-model"]
        }
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["sota_models_utilized"] is False
    assert "SOTA models not confirmed" in res["honest_verdict"]


def test_audit_artifacts_bad_json(tmp_path: Path) -> None:
    """Test audit when an artifact contains invalid JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    for i in range(1996, 2006):
        data = {
            "experiment": i,
            "status": "success",
            "models_utilized": ["unsloth/Qwen3.6"]
        }
        with open(results_dir / f"experiment_{i}_test.json", "w", encoding="utf-8") as f:
            if i == 1999:
                f.write("not json")
            else:
                json.dump(data, f)
            
    res = audit_artifacts(str(tmp_path))
    assert res["status"] == "failure"
    assert res["valid_schema_confirmed"] is False
