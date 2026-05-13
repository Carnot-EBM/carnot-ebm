"""Tests for experiment 2037 (Milestone 159 Pre-Retro)."""
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
import experiment_2037_milestone_159_pre_retro


def test_audit_milestone_159_failure(tmp_path):
    """Test auditing when tasks failed.
    
    Validates REQ-ORCH-RETRO-001 by checking milestone summary.
    """
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    log_file = ops_dir / "conductor-log.md"
    log_file.write_text(
        "| 2026-05-13 07:41 UTC | Exp 2031: Continuous Latent EBRM Trace Editing | GATE_BLOCK |\n"
        "| 2026-05-13 07:47 UTC | Exp 2033: KAN Piecewise Affine (PWA) Abstraction | DOOMED_RERUN_BLOCK |\n"
        "| 2026-05-13 07:55 UTC | Exp 2035: GEC Epsilon-Constraint Continual Learnin | DOOMED_RERUN_BLOCK |\n"
    )
    
    result = experiment_2037_milestone_159_pre_retro.audit_milestone_159(str(tmp_path))
    assert result["experiment"] == 2037
    assert result["status"] == "failure"
    assert result["ebrm_tasks_completed"] is False
    assert result["kan_tasks_completed"] is False
    assert result["gec_tasks_completed"] is False
    assert result["ebrm_final_status"] == "GATE_BLOCK"
    assert result["kan_final_status"] == "DOOMED_RERUN_BLOCK"
    assert result["gec_final_status"] == "DOOMED_RERUN_BLOCK"

def test_audit_milestone_159_success(tmp_path):
    """Test auditing when tasks succeeded.
    
    Validates REQ-ORCH-RETRO-001 by checking milestone summary.
    """
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    log_file = ops_dir / "conductor-log.md"
    log_file.write_text(
        "| 2026-05-13 07:41 UTC | Exp 2031: Continuous Latent EBRM Trace Editing | OK |\n"
        "| 2026-05-13 07:47 UTC | Exp 2033: KAN Piecewise Affine (PWA) Abstraction | OK |\n"
        "| 2026-05-13 07:55 UTC | Exp 2035: GEC Epsilon-Constraint Continual Learnin | OK |\n"
    )
    
    result = experiment_2037_milestone_159_pre_retro.audit_milestone_159(str(tmp_path))
    assert result["status"] == "success"
    assert result["ebrm_tasks_completed"] is True
    assert result["kan_tasks_completed"] is True
    assert result["gec_tasks_completed"] is True
    assert result["ebrm_final_status"] == "OK"
    assert result["kan_final_status"] == "OK"
    assert result["gec_final_status"] == "OK"

def test_audit_milestone_159_missing_log(tmp_path):
    """Test auditing when log is missing."""
    result = experiment_2037_milestone_159_pre_retro.audit_milestone_159(str(tmp_path))
    assert result["status"] == "failure"
    assert result["ebrm_final_status"] == "UNKNOWN"
    assert result["kan_final_status"] == "UNKNOWN"
    assert result["gec_final_status"] == "UNKNOWN"

def test_main(tmp_path, monkeypatch):
    """Test main function."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    log_file = ops_dir / "conductor-log.md"
    log_file.write_text("")
    
    experiment_2037_milestone_159_pre_retro.main()
    
    out_file = tmp_path / "results" / "experiment_2037_milestone_159_pre_retro.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["experiment"] == 2037
