"""Tests for experiment 3404 GateMate N=16 bootstrap fix script.
References: REQ-HW-106, SCENARIO-HW-106.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import subprocess

from scripts.experiment_3404_gatemate_n16_bootstrap_fix import main, run_subprocess

@pytest.fixture
def mock_template(tmp_path):
    with patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.ExperimentTemplate") as mock:
        instance = mock.return_value
        instance._output_path = tmp_path / "results" / "experiment_3404_gatemate_n16_bootstrap_fix.json"
        instance._output_path.parent.mkdir(parents=True, exist_ok=True)
        instance.build_result.side_effect = lambda data, status: {"data": data, "status": status}
        yield instance

@patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run")
def test_experiment_3404_success(mock_run, mock_template):
    """Test the success path where all tools succeed.
    References: SCENARIO-HW-106
    """
    mock_run.return_value = MagicMock(stdout="Success", stderr="", returncode=0)
    
    main()
    
    mock_template.setup.assert_called_once()
    assert mock_template._output_path.exists()
    
    with open(mock_template._output_path) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "success"
    assert artifact["data"]["synthesis_success"] is True
    assert artifact["data"]["pnr_success"] is True
    assert artifact["data"]["flash_success"] is True
    assert artifact["data"]["test_success"] is True

@patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run")
def test_experiment_3404_synthesis_fail(mock_run, mock_template):
    """Test the failure path where synthesis fails.
    References: SCENARIO-HW-106
    """
    def side_effect(cmd, **kwargs):
        if "yosys" in cmd:
            raise subprocess.CalledProcessError(1, cmd, output="Fail yosys", stderr="Error")
        return MagicMock(stdout="Success", stderr="", returncode=0)
        
    mock_run.side_effect = side_effect
    
    main()
    
    with open(mock_template._output_path) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "error"
    assert artifact["data"]["synthesis_success"] is False
    assert artifact["data"]["pnr_success"] is False
    assert artifact["data"]["flash_success"] is False
    assert artifact["data"]["test_success"] is False

@patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run")
def test_experiment_3404_flash_fail(mock_run, mock_template):
    """Test the failure path where flash fails (likely in CI environment).
    References: SCENARIO-HW-106
    """
    def side_effect(cmd, **kwargs):
        if "openFPGALoader" in cmd[0]:
            raise subprocess.CalledProcessError(1, cmd, output="Fail flash", stderr="Error")
        return MagicMock(stdout="Success", stderr="", returncode=0)
        
    mock_run.side_effect = side_effect
    
    main()
    
    with open(mock_template._output_path) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "error"
    assert artifact["data"]["synthesis_success"] is True
    assert artifact["data"]["pnr_success"] is True
    assert artifact["data"]["flash_success"] is False
    assert artifact["data"]["test_success"] is False

@patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run")
@patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.os.path.exists")
def test_experiment_3404_hardware_test_exists(mock_exists, mock_run, mock_template):
    """Test the success path where hardware test script exists.
    References: SCENARIO-HW-106
    """
    mock_run.return_value = MagicMock(stdout="Success", stderr="", returncode=0)
    mock_exists.return_value = True
    
    main()
    
    with open(mock_template._output_path) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "success"
    assert artifact["data"]["test_success"] is True

@patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run")
def test_experiment_3404_tool_not_found(mock_run, mock_template):
    """Test the failure path where a tool is not installed (FileNotFoundError).
    References: SCENARIO-HW-106
    """
    mock_run.side_effect = FileNotFoundError(2, "No such file or directory")
    
    main()
    
    with open(mock_template._output_path) as f:
        artifact = json.load(f)
        
    assert artifact["status"] == "error"
    assert artifact["data"]["synthesis_success"] is False
    assert "Command not found" in artifact["data"]["synthesis_log"]

def test_run_subprocess_called_process_error_bytes():
    """Test run_subprocess handles bytes in CalledProcessError."""
    with patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["cmd"], output=b"stdout_bytes", stderr=b"stderr_bytes"
        )
        success, log = run_subprocess(["cmd"])
        assert success is False
        assert "stdout_bytes\nstderr_bytes" in log

def test_run_subprocess_called_process_error_none():
    """Test run_subprocess handles None in CalledProcessError."""
    with patch("scripts.experiment_3404_gatemate_n16_bootstrap_fix.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["cmd"], output=None, stderr=None
        )
        success, log = run_subprocess(["cmd"])
        assert success is False
        assert "\n" in log
