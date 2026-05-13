"""Tests for experiment 2027 (Milestone 158 Retrospective)."""
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
import experiment_2027_milestone_158_retro


def test_retro_milestone_158_failure(tmp_path):
    """Test generating retro when tasks failed.
    
    Validates REQ-REPORT-158 and SCENARIO-REPORT-158-A by checking milestone summary.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    
    pre_retro_data = {
        "experiment": 2026,
        "status": "failure",
        "seal_tasks_completed": False,
        "stkan_tasks_completed": False,
        "seal_final_status": "GATE_BLOCK",
        "stkan_final_status": "FAIL",
        "honest_verdict": "Audit complete. SEAL status: GATE_BLOCK. STKAN status: FAIL. Tasks did not complete."
    }
    pre_retro_file.write_text(json.dumps(pre_retro_data))
    
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    assert result["experiment_id"] == 2027
    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.158"
    assert result["status"] == "failure"
    assert result["seal_success"] is False
    assert result["stkan_success"] is False
    assert "recommendations" in result
    assert result["retro_complete"] is True
    assert "honest_verdict" in result

def test_retro_milestone_158_success(tmp_path):
    """Test generating retro when tasks succeeded.
    
    Validates REQ-REPORT-158 by checking milestone summary.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    
    pre_retro_data = {
        "experiment": 2026,
        "status": "success",
        "seal_tasks_completed": True,
        "stkan_tasks_completed": True,
        "seal_final_status": "OK",
        "stkan_final_status": "OK",
        "honest_verdict": "Audit complete. SEAL status: OK. STKAN status: OK. Tasks completed."
    }
    pre_retro_file.write_text(json.dumps(pre_retro_data))
    
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    assert result["status"] == "success"
    assert result["seal_success"] is True
    assert result["stkan_success"] is True

def test_retro_milestone_158_missing_pre_retro(tmp_path):
    """Test generating retro when pre-retro is missing."""
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    assert result["status"] == "failure"
    assert result["seal_success"] is False
    assert result["stkan_success"] is False

def test_main(tmp_path, monkeypatch):
    """Test main function."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro_file.write_text(json.dumps({}))
    
    experiment_2027_milestone_158_retro.main()
    
    out_file = results_dir / "experiment_2027_milestone_158_retro.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["experiment_id"] == 2027
