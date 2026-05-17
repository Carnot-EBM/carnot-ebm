import json
from pathlib import Path
import pytest
from scripts.experiment_2154_retro import generate_retro_data, main

def test_generate_retro_data():
    """Test the generation of retrospective data for Milestone 213."""
    data = generate_retro_data()
    assert data["experiment_id"] == 2154
    assert data["schema"] == "carnot.milestone_retro.v1"
    assert data["status"] == "complete"
    assert 2143 in data["completed_experiments"]
    assert 2144 in data["completed_experiments"]
    assert 2147 in data["completed_experiments"]
    assert 2150 in data["completed_experiments"]
    assert 2152 in data["completed_experiments"]
    assert 2145 in data["blocked_experiments"]
    assert 2146 in data["blocked_experiments"]
    assert 2148 in data["blocked_experiments"]
    assert 2149 in data["blocked_experiments"]
    assert 2151 in data["blocked_experiments"]
    assert 2153 in data["blocked_experiments"]
    assert data["completed_task_count"] == 5
    assert data["blocked_task_count"] == 6

def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test the main entrypoint writes the JSON deliverable."""
    monkeypatch.chdir(tmp_path)
    main()
    
    out_path = tmp_path / "results" / "experiment_2154_retro.json"
    assert out_path.exists()
    
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["experiment_id"] == 2154
