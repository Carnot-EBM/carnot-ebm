import pytest
from unittest.mock import patch, MagicMock
from carnot.hardware.polarfire_continuity_v17 import (
    check_ssh_reachable,
    get_uptime,
    get_dispatch_path,
    perform_continuity_check
)

# REQ-HW-079
# SCENARIO-HW-079

@patch("carnot.hardware.polarfire_continuity_v17.subprocess.run")
def test_check_ssh_reachable_success(mock_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_run.return_value = mock_result
    
    assert check_ssh_reachable() is True
    mock_run.assert_called_once()

@patch("carnot.hardware.polarfire_continuity_v17.subprocess.run")
def test_check_ssh_reachable_failure(mock_run):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_run.return_value = mock_result
    
    assert check_ssh_reachable() is False
    mock_run.assert_called_once()

@patch("carnot.hardware.polarfire_continuity_v17.subprocess.run")
def test_get_uptime_success(mock_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "12345.67 89012.34\n"
    mock_run.return_value = mock_result
    
    assert get_uptime() == "12345.67"

@patch("carnot.hardware.polarfire_continuity_v17.subprocess.run")
def test_get_dispatch_path_success(mock_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "/usr/bin/carnot\n"
    mock_run.return_value = mock_result
    
    assert get_dispatch_path() == "/usr/bin/carnot"

@patch("carnot.hardware.polarfire_continuity_v17.check_ssh_reachable")
@patch("carnot.hardware.polarfire_continuity_v17.get_uptime")
@patch("carnot.hardware.polarfire_continuity_v17.get_dispatch_path")
def test_perform_continuity_check_reachable(mock_dispatch, mock_uptime, mock_ssh):
    mock_ssh.return_value = True
    mock_uptime.return_value = "1234.5"
    mock_dispatch.return_value = "/opt/carnot"
    
    result = perform_continuity_check()
    
    assert result["inference_substrate"] == "hardware_smoke"
    assert result["preconditions_checked"] is True
    assert result["polarfire_ssh_reachable"] is True
    assert result["honest_verdict"] == "complete: polarfire_continuity_confirmed_reachable"
    assert result["polarfire_uptime_s"] == "1234.5"
    assert result["polarfire_carnot_dispatch_path"] == "/opt/carnot"
    assert "duration_s" in result
    assert "random_seed" in result
    assert "reproducibility_checksum" in result

@patch("carnot.hardware.polarfire_continuity_v17.check_ssh_reachable")
def test_perform_continuity_check_unreachable(mock_ssh):
    mock_ssh.return_value = False
    
    result = perform_continuity_check()
    
    assert result["inference_substrate"] == "hardware_smoke"
    assert result["preconditions_checked"] is True
    assert result["polarfire_ssh_reachable"] is False
    assert result["honest_verdict"] == "complete: blocked_polarfire_ssh_timeout"
    assert "polarfire_uptime_s" not in result
    assert "duration_s" in result
    assert "random_seed" in result
    assert "reproducibility_checksum" in result
