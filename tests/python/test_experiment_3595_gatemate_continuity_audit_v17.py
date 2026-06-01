import json
import os
from unittest import mock
import pytest

from carnot.experiment_3595_gatemate_continuity_audit_v17 import run_experiment, check_gatemate_detect

# REQ-FPGA-GATEMATE-CONTINUITY-V17
# SCENARIO-HW-3595: GateMate continuity is audited via JTAG detect only due to known host-IO hang.

@mock.patch("subprocess.run")
def test_check_gatemate_detect_success(mock_run):
    mock_run.return_value = mock.Mock(returncode=0)
    assert check_gatemate_detect() is True
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "openFPGALoader" in args
    assert "--detect" in args

@mock.patch("subprocess.run")
def test_check_gatemate_detect_failure(mock_run):
    mock_run.return_value = mock.Mock(returncode=1)
    assert check_gatemate_detect() is False

@mock.patch("subprocess.run")
def test_check_gatemate_detect_exception(mock_run):
    mock_run.side_effect = Exception("command failed")
    assert check_gatemate_detect() is None

@mock.patch("carnot.experiment_3595_gatemate_continuity_audit_v17.check_gatemate_detect")
def test_run_experiment_success(mock_check, tmp_path):
    mock_check.return_value = True
    
    out_file = tmp_path / "out.json"
    run_experiment(str(out_file))
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
    assert data["inference_substrate"] == "hardware_smoke"
    assert data["gatemate_idcode_detected"] is True
    assert data["known_blocker"] == "flash/smoke host-IO hang"
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data

@mock.patch("carnot.experiment_3595_gatemate_continuity_audit_v17.check_gatemate_detect")
def test_run_experiment_failure(mock_check, tmp_path):
    mock_check.return_value = False
    
    out_file = tmp_path / "out.json"
    run_experiment(str(out_file))
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
    assert data["inference_substrate"] == "hardware_smoke"
    assert data["gatemate_idcode_detected"] is False
    assert data["known_blocker"] == "flash/smoke host-IO hang"
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
