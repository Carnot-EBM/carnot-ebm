import os
import json
from carnot.reporting.telemetry_aggregation_3413 import generate_telemetry_aggregation_v39

def test_generate_telemetry_aggregation_v39_spec_coverage(tmp_path, monkeypatch):
    """
    Test that REQ-REPORT-3413 / SCENARIO-REPORT-3413 is satisfied.
    The task is to verify it creates the artifact and has matrix_v39_ready set to true.
    """
    # Monkeypatch the working directory to our tmp_path
    monkeypatch.chdir(tmp_path)
    
    # Run the function
    data = generate_telemetry_aggregation_v39()
    
    # Assert return dictionary
    assert data["matrix_v39_ready"] is True
    assert "tallies" in data
    
    # Assert file creation
    expected_file = tmp_path / "results" / "experiment_3413_telemetry_aggregation_v39.json"
    assert expected_file.exists()
    
    with open(expected_file, "r") as f:
        file_data = json.load(f)
        
    assert file_data["matrix_v39_ready"] is True
    assert "tallies" in file_data
    assert file_data["tallies"]["complete"] == 8
