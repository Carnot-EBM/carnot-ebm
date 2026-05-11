"""
Tests for Exp 1813 retrospective script.

References:
- REQ-REPORT-1813: Exp 1813 retrospective must summarize Exp 1803..1812 success/failures and list top 3 gaps.
- SCENARIO-REPORT-1813: It reads valid artifacts, calculates ratios, and produces a valid output.
"""
import os
import json
import tempfile
from unittest import mock

# We will import the script logic here
import scripts.experiment_1813_retro as retro

def test_retro_calculates_ratios_and_gaps_req_report_1813():
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock some artifact files
        mock_artifacts = [
            ("experiment_1803.json", {"experiment": 1803, "status": "blocked", "title": "Exp 1803", "honest_verdict": "blocked_gate_check_failed"}),
            ("experiment_1804.json", {"experiment": 1804, "status": "success", "title": "Exp 1804", "honest_verdict": "success"}),
            ("experiment_1805.json", {"experiment": 1805, "status": "failed", "title": "Exp 1805", "honest_verdict": "failed"}),
        ]
        
        for name, data in mock_artifacts:
            with open(os.path.join(tmpdir, name), "w") as f:
                json.dump(data, f)
        
        out_path = os.path.join(tmpdir, "experiment_1813_retro.json")
        
        # Act
        retro.run_retrospective(tmpdir, out_path)
        
        # Assert
        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
            
        assert result["total_artifacts"] == 3
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert result["blocked_count"] == 1
        assert result["success_ratio"] == 1.0 / 3.0
        assert "summary" in result
        assert len(result["top_3_gaps"]) <= 3
