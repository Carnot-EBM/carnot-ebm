import json
import os
from pathlib import Path
from unittest import mock
import pytest

# Since the script is in scripts/ and not a standard python module,
# we need to import it dynamically or adjust sys.path.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

import experiment_2027_milestone_158_retro as retro

def test_generate_retro_missing_file(tmp_path):
    result = retro.generate_retro(str(tmp_path))
    assert result["retro_complete"] is False
    assert result["honest_verdict"] == "Pre-retro artifact missing"

def test_generate_retro_unreadable_file(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro_file.write_text("{ invalid json")
    
    result = retro.generate_retro(str(tmp_path))
    assert result["retro_complete"] is False
    assert result["honest_verdict"] == "Pre-retro artifact unreadable"

def test_generate_retro_both_success(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro_file.write_text(json.dumps({
        "seal_tasks_completed": True,
        "stkan_tasks_completed": True
    }))
    
    result = retro.generate_retro(str(tmp_path))
    assert result["retro_complete"] is True
    assert result["seal_success"] is True
    assert result["stkan_success"] is True
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Both SEAL and STKAN succeeded."

def test_generate_retro_both_fail(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro_file.write_text(json.dumps({
        "seal_tasks_completed": False,
        "stkan_tasks_completed": False
    }))
    
    result = retro.generate_retro(str(tmp_path))
    assert result["retro_complete"] is True
    assert result["seal_success"] is False
    assert result["stkan_success"] is False
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Both SEAL and STKAN failed."

def test_generate_retro_seal_fail(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro_file.write_text(json.dumps({
        "seal_tasks_completed": False,
        "stkan_tasks_completed": True
    }))
    
    result = retro.generate_retro(str(tmp_path))
    assert result["retro_complete"] is True
    assert result["seal_success"] is False
    assert result["stkan_success"] is True
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Partial success."
    assert "SEAL tasks failed" in result["recommendations"][0]

def test_generate_retro_stkan_fail(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    pre_retro_file.write_text(json.dumps({
        "seal_tasks_completed": True,
        "stkan_tasks_completed": False
    }))
    
    result = retro.generate_retro(str(tmp_path))
    assert result["retro_complete"] is True
    assert result["seal_success"] is True
    assert result["stkan_success"] is False
    assert result["honest_verdict"] == "Milestone .158 retrospective complete. Partial success."
    assert "STKAN tasks failed" in result["recommendations"][0]

@mock.patch("experiment_2027_milestone_158_retro.generate_retro")
@mock.patch("experiment_2027_milestone_158_retro.os.getcwd")
def test_main(mock_getcwd, mock_generate, tmp_path):
    mock_getcwd.return_value = str(tmp_path)
    mock_generate.return_value = {"test": "data"}
    
    with mock.patch("experiment_2027_milestone_158_retro.os.environ", {}):
        retro.main()
        
    out_file = tmp_path / "results" / "experiment_2027_milestone_158_retro.json"
    assert out_file.exists()
    with out_file.open() as f:
        data = json.load(f)
    assert data == {"test": "data"}
