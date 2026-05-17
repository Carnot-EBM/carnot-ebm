import json
from pathlib import Path
import pytest
from scripts.experiment_2142_retro import generate_retro_data, main

def test_generate_retro_data():
    """Test the generation of retrospective data for Milestone 212."""
    data = generate_retro_data()
    assert data["experiment_id"] == 2142
    assert data["schema"] == "carnot.milestone_retro.v1"
    assert data["status"] == "complete"
    assert 2133 in data["completed_experiments"]
    assert 2136 in data["completed_experiments"]
    assert 2139 in data["completed_experiments"]
    assert 2134 in data["blocked_experiments"]
    assert 2135 in data["blocked_experiments"]
    assert 2137 in data["blocked_experiments"]
    assert 2140 in data["blocked_experiments"]
    assert data["completed_task_count"] == 3
    assert data["blocked_task_count"] == 4

def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test the main entrypoint writes the JSON deliverable."""
    monkeypatch.chdir(tmp_path)
    main()
    
    out_path = tmp_path / "results" / "experiment_2142_retro.json"
    assert out_path.exists()
    
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["experiment_id"] == 2142
