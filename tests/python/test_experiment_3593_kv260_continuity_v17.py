import json
import os
from unittest import mock
import pytest

from carnot.experiment_3593_kv260_continuity_v17 import run_experiment, check_ssh_reachability, get_kv260_overlay

# REQ-FPGA-KV260-CONTINUITY-V17
# SCENARIO-HW-3593: KV260 continuity is audited via SSH reachability only.

@mock.patch("subprocess.run")
def test_ssh_reachability_success(mock_run):
    mock_run.return_value = mock.Mock(returncode=0)
    assert check_ssh_reachability() is True
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "ssh" in args
    assert "kria" in args

@mock.patch("subprocess.run")
def test_ssh_reachability_failure(mock_run):
    mock_run.return_value = mock.Mock(returncode=255)
    assert check_ssh_reachability() is False

@mock.patch("subprocess.run")
def test_ssh_reachability_exception(mock_run):
    mock_run.side_effect = Exception("ssh failed")
    assert check_ssh_reachability() is False

@mock.patch("subprocess.run")
def test_get_overlay_success(mock_run):
    mock_run.return_value = mock.Mock(returncode=0, stdout="k26-starter-kits")
    assert get_kv260_overlay() == "k26-starter-kits"

@mock.patch("subprocess.run")
def test_get_overlay_failure(mock_run):
    mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error")
    assert get_kv260_overlay() is None

@mock.patch("subprocess.run")
def test_get_overlay_exception(mock_run):
    mock_run.side_effect = Exception("xmutil failed")
    assert get_kv260_overlay() is None

@mock.patch("carnot.experiment_3593_kv260_continuity_v17.get_kv260_overlay")
@mock.patch("carnot.experiment_3593_kv260_continuity_v17.check_ssh_reachability")
def test_run_experiment_success(mock_check, mock_get_overlay, tmp_path):
    mock_check.return_value = True
    mock_get_overlay.return_value = "smartcam"
    
    out_file = tmp_path / "out.json"
    run_experiment(str(out_file))
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "complete: kv260_continuity_confirmed_reachable"
    assert data["inference_substrate"] == "hardware_smoke"
    assert isinstance(data["preconditions_checked"], list)
    assert data["preconditions_checked"][0] == {"resource": "kv260_ssh", "available": True}
    assert data["kv260_ssh_reachable"] is True
    assert data["kv260_overlay_loaded"] == "smartcam"
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data

@mock.patch("carnot.experiment_3593_kv260_continuity_v17.check_ssh_reachability")
def test_run_experiment_failure(mock_check, tmp_path):
    mock_check.return_value = False
    
    out_file = tmp_path / "out.json"
    run_experiment(str(out_file))
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "complete: blocked_kv260_ssh_unreachable"
    assert data["inference_substrate"] == "hardware_smoke"
    assert isinstance(data["preconditions_checked"], list)
    assert data["preconditions_checked"][0] == {"resource": "kv260_ssh", "available": False}
    assert data["kv260_ssh_reachable"] is False
    assert data["kv260_overlay_loaded"] is None
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
