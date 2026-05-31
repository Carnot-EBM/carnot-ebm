import os
import json
import tempfile
from unittest import mock

# REQ-HW-3578
# SCENARIO-HW-3578

from carnot.experiment_3578_polarfire_continuity_v16 import (
    run_experiment,
    check_ssh_reachability,
    confirm_continuity
)

def test_experiment_3578_reachable(tmp_path):
    output_file = tmp_path / "experiment_3578_polarfire_continuity_v16.json"
    
    with mock.patch('subprocess.run') as mock_run:
        # Mock responses: 1 for ssh check, 1 for uptime check
        # We need to mock correctly depending on the command
        def side_effect(cmd, *args, **kwargs):
            m = mock.Mock()
            if "true" in cmd:
                m.returncode = 0
            elif "uptime" in cmd:
                m.returncode = 0
                m.stdout = "22:44:51 up 2 days"
            else:
                m.returncode = 1
            return m
            
        mock_run.side_effect = side_effect
        
        run_experiment(str(output_file))
        
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: polarfire_continuity_confirmed_reachable"
        assert data["inference_substrate"] == "hardware_smoke"
        assert data["polarfire_ssh_reachable"] is True
        assert "preconditions_checked" in data
        assert "random_seed" in data
        assert "reproducibility_checksum" in data
        assert "duration_s" in data

def test_experiment_3578_unreachable(tmp_path):
    output_file = tmp_path / "experiment_3578_polarfire_continuity_v16.json"
    
    with mock.patch('subprocess.run') as mock_run:
        m = mock.Mock()
        m.returncode = 255 # Timeout
        mock_run.return_value = m
        
        run_experiment(str(output_file))
        
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: blocked_polarfire_ssh_timeout"
        assert data["inference_substrate"] == "hardware_smoke"
        assert data["polarfire_ssh_reachable"] is False
        assert "preconditions_checked" in data
        assert "random_seed" in data
        assert "reproducibility_checksum" in data
        assert "duration_s" in data

def test_check_ssh_reachability_exception():
    with mock.patch('subprocess.run', side_effect=Exception("Error")):
        assert check_ssh_reachability() is False

def test_confirm_continuity_exception():
    with mock.patch('subprocess.run', side_effect=Exception("Error")):
        assert confirm_continuity() == "unknown"
