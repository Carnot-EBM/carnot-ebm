"""Tests for PolarFire continuity script v18 (Exp 3608)."""

import json
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path if needed, but normally pytest running from root can import scripts
import sys
if os.path.abspath("scripts") not in sys.path:
    sys.path.insert(0, os.path.abspath("scripts"))

from scripts import experiment_3608_polarfire_continuity_v18

@pytest.fixture
def temp_results_dir(tmpdir):
    """Provide a temporary results directory."""
    with patch("scripts.experiment_3608_polarfire_continuity_v18.RESULTS_FILE", str(tmpdir.join("experiment_3608_polarfire_continuity_v18.json"))):
        yield tmpdir

@patch("subprocess.run")
def test_experiment_3608_reachable(mock_run, temp_results_dir):
    """
    Test SCENARIO-HW-081: PolarFire is reachable.
    Validates REQ-HW-081.
    """
    # Configure mock
    def mock_subprocess_run(cmd, *args, **kwargs):
        mock_result = MagicMock()
        if "true" in cmd:
            mock_result.returncode = 0
        elif "uptime" in cmd:
            mock_result.returncode = 0
            mock_result.stdout = "10:00  up 10 days"
        elif "which carnot" in cmd:
            mock_result.returncode = 0
            mock_result.stdout = "/usr/local/bin/carnot"
        else:
            mock_result.returncode = 1
            mock_result.stdout = ""
        return mock_result

    mock_run.side_effect = mock_subprocess_run

    # Run the experiment
    experiment_3608_polarfire_continuity_v18.run_experiment()

    # Verify JSON output
    results_file = experiment_3608_polarfire_continuity_v18.RESULTS_FILE
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        data = json.load(f)

    assert data["honest_verdict"] == "complete: polarfire_continuity_confirmed_reachable"
    assert data["inference_substrate"] == "hardware_smoke"
    assert data["polarfire_ssh_reachable"] is True
    assert "polarfire_uptime" in data
    assert "polarfire_dispatch_path" in data
    assert "preconditions_checked" in data
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data

@patch("subprocess.run")
def test_experiment_3608_unreachable(mock_run, temp_results_dir):
    """
    Test SCENARIO-HW-081: PolarFire is unreachable.
    Validates REQ-HW-081.
    """
    # Configure mock to simulate unreachable host
    def mock_subprocess_run(cmd, *args, **kwargs):
        mock_result = MagicMock()
        mock_result.returncode = 255
        mock_result.stdout = ""
        mock_result.stderr = "ssh: connect to host polarfire port 22: Connection timed out"
        return mock_result

    mock_run.side_effect = mock_subprocess_run

    # Run the experiment
    experiment_3608_polarfire_continuity_v18.run_experiment()

    # Verify JSON output
    results_file = experiment_3608_polarfire_continuity_v18.RESULTS_FILE
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        data = json.load(f)

    assert data["honest_verdict"] == "complete: blocked_polarfire_ssh_timeout"
    assert data["inference_substrate"] == "hardware_smoke"
    assert data["polarfire_ssh_reachable"] is False
    assert "polarfire_uptime" not in data  # Optional or null depending on schema
    assert "polarfire_dispatch_path" not in data # Optional or null
    assert "preconditions_checked" in data
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
