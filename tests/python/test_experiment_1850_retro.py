"""
Tests for Exp 1850 retrospective script.

References:
- REQ-REPORT-1850: Exp 1850 retrospective must summarize Exp 1839..1849 success/failures and output a JSON.
- SCENARIO-REPORT-1850: It reads valid artifacts, calculates results, and produces a valid output.
"""
import os
import json
import tempfile
from unittest import mock
import pytest

# We will import the script logic here
import scripts.experiment_1850_retro as retro

def test_retro_calculates_results_req_report_1850():
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock some artifact files
        mock_artifacts = [
            ("experiment_1839_activation.json", {"experiment": 1839, "status": "blocked", "title": "Exp 1839", "honest_verdict": "blocked_gate_check_failed", "started_at": "2026-05-11T12:00:00Z"}),
            ("experiment_1840_success.json", {"experiment": 1840, "status": "success", "title": "Exp 1840", "honest_verdict": "success", "finished_at": "2026-05-11T13:00:00Z"}),
            # 1841..1849 will be missing
        ]
        
        for name, data in mock_artifacts:
            with open(os.path.join(tmpdir, name), "w") as f:
                json.dump(data, f)
        
        out_path = os.path.join(tmpdir, "experiment_1850_retro.json")
        
        # Act
        retro.run_retrospective(tmpdir, out_path)
        
        # Assert
        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
            
        assert result["experiment"] == 1850
        assert result["milestone"] == "2026.05.143"
        assert result["schema"] == "carnot.experiment.retro.v1"
        assert result["status"] == "complete"
        assert result["honest_verdict"] == "milestone_complete"
        assert result["criteria_results"]["exp1839"] is False
        assert result["criteria_results"]["exp1840"] is True
        assert result["criteria_results"]["exp1841"] is False  # Missing
        assert result["criteria_details"]["exp1839"]["verdict"] == "blocked_gate_check_failed"
        assert result["criteria_details"]["exp1840"]["verdict"] == "success"
        assert result["criteria_details"]["exp1841"]["status"] == "missing"

def test_retro_handles_parse_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "experiment_1839_bad.json"), "w") as f:
            f.write("invalid json")
            
        out_path = os.path.join(tmpdir, "experiment_1850_retro.json")
        retro.run_retrospective(tmpdir, out_path)
        
        with open(out_path, "r") as f:
            result = json.load(f)
            
        assert result["criteria_results"]["exp1839"] is False
        assert result["criteria_details"]["exp1839"]["status"] == "error"
        assert result["criteria_details"]["exp1839"]["verdict"] == "parse_error"
