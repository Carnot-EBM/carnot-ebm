"""Tests for KAN formal verification smoke test (Experiment 1805).

Traces to REQ-KAN-VERIFY-001
"""
import os
import json
from unittest import mock
import pytest

from scripts.experiment_1805_kan_verify import run_smoke_test, main

def test_run_smoke_test_returns_valid_dict():
    """Verify SCENARIO-KAN-VERIFY-001: 10-constraint smoke test runs MILP and reports metrics."""
    result = run_smoke_test()
    assert "verification_time_ms" in result
    assert "bounds_soundness" in result
    assert result["experiment_id"] == "exp1805"
    assert "status" in result
    assert result["status"] == "complete"

@mock.patch("scripts.experiment_1805_kan_verify.run_smoke_test")
def test_main_writes_json(mock_run, tmp_path):
    """Verify main writes the artifact JSON."""
    mock_run.return_value = {"experiment_id": "exp1805", "fake": "data", "verification_time_ms": 100, "bounds_soundness": True}
    
    # Overwrite RESULT_PATH temporarily
    fake_path = tmp_path / "experiment_1805_smoke.json"
    with mock.patch("scripts.experiment_1805_kan_verify.RESULT_PATH", str(fake_path)):
        main()
        
    assert fake_path.exists()
    data = json.loads(fake_path.read_text())
    assert data["experiment_id"] == "exp1805"
