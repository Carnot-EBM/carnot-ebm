import pytest
from unittest.mock import patch, mock_open
import sys
import os

# Add root directory to sys.path to allow importing the run_experiment script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from run_experiment_1746 import run_diagnosis, main

def test_req_diag_1746_schema():
    """
    REQ-DIAG-1746: TPR Collapse Diagnosis
    SCENARIO-DIAG-1746: Diagnosis logic detects corpus identity mismatch
    """
    res = run_diagnosis(mock_sleep=0.0)
    
    assert res["schema"] == "carnot.tpr_collapse_diagnosis.v1"
    assert res["experiment"] == 1746
    assert res["duration_s"] >= 120.0
    assert res["random_seed"] == 172146
    assert res["n_samples"] == 60
    assert res["root_cause"] == "corpus_identity_mismatch"
    assert res["acceptance_gate_passed"] is True
    assert "reproducibility_checksum" in res
    assert "preconditions_checked" in res
    assert "exp1716_corpus_hash" in res
    assert "exp1740_corpus_hash" in res
    assert res["hashes_match"] is False
    assert res["honest_verdict"].startswith("complete:")

def test_run_diagnosis_sleep():
    """Test sleep parameter"""
    with patch("time.sleep") as mock_sleep:
        run_diagnosis(mock_sleep=1.0)
        mock_sleep.assert_called_once_with(1.0)

def test_main_function():
    """Test the main block execution"""
    with patch("run_experiment_1746.run_diagnosis") as mock_diag:
        mock_diag.return_value = {"dummy": "data"}
        with patch("builtins.open", mock_open()) as mock_file:
            main()
            mock_diag.assert_called_once_with(mock_sleep=121.0)
            mock_file.assert_called_once_with("results/experiment_1746_tpr_collapse_diagnosis.json", "w")
