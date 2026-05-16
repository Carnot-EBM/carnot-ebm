import json
import os
import subprocess
from unittest.mock import patch
from scripts.experiment_2008_rocm import main, check_rocminfo, map_memory_limits

def test_experiment_2008_rocm_basic():
    """Verify REQ-HARDWARE-2008 and SCENARIO-HARDWARE-2008 execution"""
    main()
    artifact_path = "results/experiment_2008_rocm_probe.json"
    assert os.path.exists(artifact_path)
    with open(artifact_path, "r") as f:
        data = json.load(f)
    assert data["schema"] == "carnot.hardware.v1"
    assert data["experiment"] == 2008
    assert "honest_verdict" in data
    assert "rocminfo_output" in data
    assert "memory_limits" in data

@patch('subprocess.run')
def test_experiment_2008_rocm_mock_success(mock_run):
    """Test when no hardware is found"""
    mock_run.side_effect = Exception("command not found")
    main()
    with open("results/experiment_2008_rocm_probe.json", "r") as f:
        data = json.load(f)
    assert data["honest_verdict"] == "hardware missing but probe works"

@patch('subprocess.run')
def test_experiment_2008_rocm_hardware_found(mock_run):
    """Test when hardware is found"""
    mock_run.return_value.stdout = "gfx1100 Memory: 24576 MB"
    mock_run.return_value.stderr = ""
    main()
    with open("results/experiment_2008_rocm_probe.json", "r") as f:
        data = json.load(f)
    assert data["honest_verdict"] == "hardware_probe_success"
    assert data["rocminfo_output"] == "gfx1100 Memory: 24576 MB"

def test_map_memory_limits():
    """Test memory extraction logic"""
    output = "Agent 1\n  Name:  TestAgent\n      Size:  12345 (0x0) KB\n"
    limits = map_memory_limits(output)
    assert "parsed_pools" in limits
    assert limits["parsed_pools"][0]["size_kb"] == 12345
    
    limits_empty = map_memory_limits("error")
    assert limits_empty == {"raw_output": "error"}

@patch('subprocess.run')
def test_check_rocminfo_exceptions(mock_run):
    """Cover the exceptions"""
    mock_run.side_effect = [Exception("error1"), Exception("error2")]
    assert check_rocminfo() == "error: error2"

@patch('subprocess.run')
def test_check_rocminfo_fallback(mock_run):
    """Cover fallback to rocminfo"""
    class MockResult:
        stdout = "fallback_success"
    mock_run.side_effect = [Exception("error1"), MockResult()]
    assert check_rocminfo() == "fallback_success"
