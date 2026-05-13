"""
Tests for the milestone .158 retrospective generator.

REQ-REPORT-158: Milestone .158 Retrospective Artifact
SCENARIO-REPORT-158-A: Milestone .158 Retrospective Analyzes SEAL and STKAN
"""

import json
from pathlib import Path
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
import experiment_2027_milestone_158_retro

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))

def test_generate_retro_failure(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    _write(pre_retro, {
        "experiment": 2026,
        "status": "failure",
        "seal_tasks_completed": False,
        "stkan_tasks_completed": False,
        "seal_final_status": "GATE_BLOCK",
        "stkan_final_status": "FAIL",
        "honest_verdict": "Audit complete. SEAL status: GATE_BLOCK. STKAN status: FAIL. Tasks did not complete."
    })
    
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    
    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.158"
    assert result["experiment_id"] == 2027
    assert result["status"] == "complete"
    assert result["seal_success"] is False
    assert result["stkan_success"] is False
    assert "SEAL loop" in result["recommendations"][0]
    assert result["retro_complete"] is True
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Both SEAL and STKAN failed."

def test_generate_retro_success(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    _write(pre_retro, {
        "experiment": 2026,
        "status": "success",
        "seal_tasks_completed": True,
        "stkan_tasks_completed": True,
        "seal_final_status": "OK",
        "stkan_final_status": "OK",
        "honest_verdict": "Audit complete. SEAL status: OK. STKAN status: OK. Tasks completed."
    })
    
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    
    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.158"
    assert result["experiment_id"] == 2027
    assert result["status"] == "complete"
    assert result["seal_success"] is True
    assert result["stkan_success"] is True
    assert "proceed" in result["recommendations"][0].lower()
    assert result["retro_complete"] is True
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Both SEAL and STKAN succeeded."

def test_generate_retro_partial(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    _write(pre_retro, {
        "experiment": 2026,
        "status": "failure",
        "seal_tasks_completed": True,
        "stkan_tasks_completed": False,
        "seal_final_status": "OK",
        "stkan_final_status": "FAIL",
        "honest_verdict": "Audit complete. SEAL status: OK. STKAN status: FAIL."
    })
    
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    assert result["seal_success"] is True
    assert result["stkan_success"] is False
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Partial success."

def test_generate_retro_missing_file(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    assert result["status"] == "failure"
    assert result["honest_verdict"] == "Pre-retro artifact missing"

def test_generate_retro_unreadable_file(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro.write_text("{ unreadable json ")
    result = experiment_2027_milestone_158_retro.generate_retro(str(tmp_path))
    assert result["status"] == "failure"
    assert result["honest_verdict"] == "Pre-retro artifact unreadable"

def test_main(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    _write(pre_retro, {
        "experiment": 2026,
        "status": "success",
        "seal_tasks_completed": True,
        "stkan_tasks_completed": True,
        "seal_final_status": "OK",
        "stkan_final_status": "OK",
        "honest_verdict": "Audit complete."
    })
    
    experiment_2027_milestone_158_retro.main()
    
    out_file = results_dir / "experiment_2027_milestone_158_retro.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["experiment_id"] == 2027
