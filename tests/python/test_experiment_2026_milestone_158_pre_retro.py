"""Tests for experiment 2026 (Milestone 158 Pre-Retro)."""
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
import experiment_2026_milestone_158_pre_retro


def test_audit_milestone_158_failure(tmp_path):
    """Test auditing when tasks failed.
    
    Validates REQ-ORCH-RETRO-001 by checking milestone summary.
    """
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    log_file = ops_dir / "conductor-log.md"
    log_file.write_text(
        "| 2026-05-13 06:36 UTC | Exp 2021: SEAL Self-Adaptive Learning: Synthetic T | GATE_BLOCK |\n"
        "| 2026-05-13 06:42 UTC | Exp 2024: STKAN Spatio-Temporal Constraint Model P | FAIL |\n"
    )
    
    result = experiment_2026_milestone_158_pre_retro.audit_milestone_158(str(tmp_path))
    assert result["experiment"] == 2026
    assert result["status"] == "failure"
    assert result["seal_tasks_completed"] is False
    assert result["stkan_tasks_completed"] is False
    assert result["seal_final_status"] == "GATE_BLOCK"
    assert result["stkan_final_status"] == "FAIL"

def test_audit_milestone_158_success(tmp_path):
    """Test auditing when tasks succeeded.
    
    Validates REQ-ORCH-RETRO-001 by checking milestone summary.
    """
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    log_file = ops_dir / "conductor-log.md"
    log_file.write_text(
        "| 2026-05-13 06:36 UTC | Exp 2021: SEAL Self-Adaptive Learning: Synthetic T | OK |\n"
        "| 2026-05-13 06:42 UTC | Exp 2024: STKAN Spatio-Temporal Constraint Model P | OK |\n"
    )
    
    result = experiment_2026_milestone_158_pre_retro.audit_milestone_158(str(tmp_path))
    assert result["status"] == "success"
    assert result["seal_tasks_completed"] is True
    assert result["stkan_tasks_completed"] is True
    assert result["seal_final_status"] == "OK"
    assert result["stkan_final_status"] == "OK"

def test_audit_milestone_158_missing_log(tmp_path):
    """Test auditing when log is missing."""
    result = experiment_2026_milestone_158_pre_retro.audit_milestone_158(str(tmp_path))
    assert result["status"] == "failure"
    assert result["seal_final_status"] == "UNKNOWN"

def test_main(tmp_path, monkeypatch):
    """Test main function."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    log_file = ops_dir / "conductor-log.md"
    log_file.write_text("")
    
    experiment_2026_milestone_158_pre_retro.main()
    
    out_file = tmp_path / "results" / "experiment_2026_milestone_158_pre_retro.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["experiment"] == 2026
