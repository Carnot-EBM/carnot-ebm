"""
Tests for GateMate continuity audit experiment script.

References: REQ-HW-3579, SCENARIO-HW-3579
"""
import json
import pytest
import subprocess
from unittest.mock import patch
import builtins

from scripts import experiment_3579_gatemate_continuity_audit_v16

def test_run_experiment_success(tmp_path):
    with patch("scripts.experiment_3579_gatemate_continuity_audit_v16.subprocess.run") as mock_run:
        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="IDCODE: 0x12345678", stderr="")
        mock_run.return_value = mock_result
        
        real_open = builtins.open
        def fake_open(file, mode="r", *args, **kwargs):
            if "results/experiment_3579_gatemate_continuity_audit_v16.json" in str(file):
                return real_open(tmp_path / "test.json", mode, *args, **kwargs)
            return real_open(file, mode, *args, **kwargs)
            
        with patch("builtins.open", fake_open):
            experiment_3579_gatemate_continuity_audit_v16.run_experiment()
            
        with open(tmp_path / "test.json", "r") as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
        assert data["inference_substrate"] == "hardware_smoke"
        assert data["gatemate_idcode_detected"] is True
        assert data["known_blocker"] == ""

def test_run_experiment_file_not_found(tmp_path):
    with patch("scripts.experiment_3579_gatemate_continuity_audit_v16.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        
        real_open = builtins.open
        def fake_open(file, mode="r", *args, **kwargs):
            if "results/experiment_3579_gatemate_continuity_audit_v16.json" in str(file):
                return real_open(tmp_path / "test.json", mode, *args, **kwargs)
            return real_open(file, mode, *args, **kwargs)
            
        with patch("builtins.open", fake_open):
            experiment_3579_gatemate_continuity_audit_v16.run_experiment()
            
        with open(tmp_path / "test.json", "r") as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
        assert data["gatemate_idcode_detected"] is False
        assert "not found" in data["known_blocker"]
        
def test_run_experiment_timeout(tmp_path):
    with patch("scripts.experiment_3579_gatemate_continuity_audit_v16.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="openFPGALoader", timeout=10)
        
        real_open = builtins.open
        def fake_open(file, mode="r", *args, **kwargs):
            if "results/experiment_3579_gatemate_continuity_audit_v16.json" in str(file):
                return real_open(tmp_path / "test.json", mode, *args, **kwargs)
            return real_open(file, mode, *args, **kwargs)
            
        with patch("builtins.open", fake_open):
            experiment_3579_gatemate_continuity_audit_v16.run_experiment()
            
        with open(tmp_path / "test.json", "r") as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
        assert data["gatemate_idcode_detected"] is False
        assert "timed out" in data["known_blocker"]

def test_run_experiment_failure(tmp_path):
    with patch("scripts.experiment_3579_gatemate_continuity_audit_v16.subprocess.run") as mock_run:
        mock_result = subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="error")
        mock_run.return_value = mock_result
        
        real_open = builtins.open
        def fake_open(file, mode="r", *args, **kwargs):
            if "results/experiment_3579_gatemate_continuity_audit_v16.json" in str(file):
                return real_open(tmp_path / "test.json", mode, *args, **kwargs)
            return real_open(file, mode, *args, **kwargs)
            
        with patch("builtins.open", fake_open):
            experiment_3579_gatemate_continuity_audit_v16.run_experiment()
            
        with open(tmp_path / "test.json", "r") as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
        assert data["gatemate_idcode_detected"] is False
        assert "exit code 127" in data["known_blocker"]

def test_run_experiment_exception(tmp_path):
    with patch("scripts.experiment_3579_gatemate_continuity_audit_v16.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("generic error")
        
        real_open = builtins.open
        def fake_open(file, mode="r", *args, **kwargs):
            if "results/experiment_3579_gatemate_continuity_audit_v16.json" in str(file):
                return real_open(tmp_path / "test.json", mode, *args, **kwargs)
            return real_open(file, mode, *args, **kwargs)
            
        with patch("builtins.open", fake_open):
            experiment_3579_gatemate_continuity_audit_v16.run_experiment()
            
        with open(tmp_path / "test.json", "r") as f:
            data = json.load(f)
            
        assert data["honest_verdict"] == "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"
        assert data["gatemate_idcode_detected"] is False
        assert "Exception: generic error" in data["known_blocker"]
