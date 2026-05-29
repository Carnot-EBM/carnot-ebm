import json
import pytest
from pathlib import Path
from unittest.mock import mock_open, patch
from scripts.experiment_3367_fix_gate_status import run_experiment

def test_run_experiment_success(tmp_path):
    # Setup mock files
    exp3355_content = '{"status": "success", "repair_lift": 0.5}'
    exp3357_content = '{"status": "success", "honest_verdict": "complete"}'
    
    def mock_file_open(file, *args, **kwargs):
        if "experiment_3355" in str(file):
            return mock_open(read_data=exp3355_content)()
        elif "experiment_3357" in str(file):
            return mock_open(read_data=exp3357_content)()
        else:
            return mock_open(read_data="{}")()

    with patch("builtins.open", side_effect=mock_file_open):
        result = run_experiment()
        assert result["status"] == "success"

def test_run_experiment_failure(tmp_path):
    # Setup mock files where one fails
    exp3355_content = '{"status": "complete", "repair_lift": 0.5}'
    exp3357_content = '{"status": "success", "honest_verdict": "complete"}'
    
    def mock_file_open(file, *args, **kwargs):
        if "experiment_3355" in str(file):
            return mock_open(read_data=exp3355_content)()
        elif "experiment_3357" in str(file):
            return mock_open(read_data=exp3357_content)()
        else:
            return mock_open(read_data="{}")()

    with patch("builtins.open", side_effect=mock_file_open):
        with pytest.raises(AssertionError, match="exp3355 missing status success"):
            run_experiment()
