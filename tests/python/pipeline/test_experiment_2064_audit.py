"""Tests for the experiment 2064 audit module."""
import os
import json
import tempfile
from unittest import mock

from carnot.pipeline.experiment_2064_audit import (
    audit_deliverables,
    verify_verifier_e2e,
    run_experiment_2064_audit
)

def test_audit_deliverables():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        with open(os.path.join(tmpdir, "experiment_2053_audit.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(tmpdir, "experiment_2063_test.json"), "w") as f:
            f.write("{}")
            
        status = audit_deliverables(tmpdir)
        assert status["2053"] is True
        assert status["2063"] is True
        assert status["2054"] is False
        assert status["2060"] is False
        assert status["2057"] is False

def test_audit_deliverables_no_dir():
    status = audit_deliverables("/nonexistent/dir/path/123")
    assert all(not val for val in status.values())

@mock.patch("subprocess.run")
def test_verify_verifier_e2e(mock_run):
    mock_res = mock.Mock()
    mock_res.returncode = 0
    mock_run.return_value = mock_res
    
    assert verify_verifier_e2e() is True
    
    mock_res.returncode = 1
    assert verify_verifier_e2e() is False
    
    mock_run.side_effect = Exception("failed")
    assert verify_verifier_e2e() is False

@mock.patch("carnot.pipeline.experiment_2064_audit.verify_verifier_e2e")
def test_run_experiment_2064_audit(mock_verify):
    mock_verify.return_value = True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "out.json")
        result = run_experiment_2064_audit(tmpdir, out_file)
        
        assert result["experiment"] == 2064
        assert result["e2e_tests_passed"] is True
        assert result["audit_passed"] is False  # Missing files
        
        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            saved = json.load(f)
            assert saved == result
