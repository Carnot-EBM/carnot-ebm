"""Test for experiment_1672_igd_smoke.py. (REQ-IGD-001)"""

import json
from pathlib import Path

from experiment_1672_igd_smoke import main


def test_experiment_1672_igd_smoke(monkeypatch, tmp_path):
    """Test the script produces the correct JSON output."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "results" / "experiment_1672_igd.json"
    
    main()
        
    assert output_file.exists()
    
    with open(output_file) as f:
        data = json.load(f)
        
    assert data["experiment_id"] == 1672
    assert "metrics" in data
    assert "satisfied_clauses" in data["metrics"]
    assert "total_clauses" in data["metrics"]
