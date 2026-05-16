import json
import os
import subprocess
from unittest.mock import patch
from scripts.experiment_1991_egpu_rocm import main, check_rocminfo, check_jax

def test_experiment_1991_egpu_rocm():
    # Run the real main to test the real output
    main()
    
    artifact_path = "results/experiment_1991_egpu_rocm.json"
    assert os.path.exists(artifact_path)
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.hardware.v1"
    assert data["experiment"] == 1991
    assert "honest_verdict" in data
    assert "rocminfo_contains_gfx1100" in data
    assert "jax_devices" in data

@patch('subprocess.run')
def test_experiment_1991_egpu_rocm_coverage(mock_run):
    # Test egpu detected successfully branch
    mock_run.return_value.stdout = "gfx1100"
    mock_run.return_value.stderr = ""
    main()
    with open("results/experiment_1991_egpu_rocm.json", "r") as f:
        data = json.load(f)
    assert data["honest_verdict"] == "egpu_detected_successfully"

@patch('subprocess.run')
def test_check_rocminfo_exceptions(mock_run):
    # Cover the exceptions
    mock_run.side_effect = [Exception("error1"), Exception("error2")]
    assert check_rocminfo() == "error2"

@patch('subprocess.run')
def test_check_jax_exceptions(mock_run):
    # Cover the exceptions
    mock_run.side_effect = Exception("error_jax")
    assert check_jax() == "error_jax"

@patch('subprocess.run')
def test_check_rocminfo_fallback(mock_run):
    # Cover fallback to rocminfo
    class MockResult:
        stdout = "fallback_success"
    mock_run.side_effect = [Exception("error1"), MockResult()]
    assert check_rocminfo() == "fallback_success"
