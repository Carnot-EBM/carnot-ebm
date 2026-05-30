"""
Tests for Experiment 3399: LogicVault CDCL Long Context Verification.

Spec: REQ-LEARN-3399, SCENARIO-LEARN-3399-A
"""
import sys
import z3
import pytest
from pathlib import Path

# Add scripts directory to path to import the experiment script
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.experiment_3399_logicvault_long_context import run_experiment_3399

def test_experiment_3399_cdcl_logic(tmp_path):
    """
    SCENARIO-LEARN-3399-A: CDCL Identifies Contradictions
    """
    result = run_experiment_3399()
    
    assert result["experiment_id"] == "3399"
    assert result["status"] == "success"
    assert result["cdcl_contradictions_caught"] == 2
    assert result["accepted_queries"] == 3
    assert "reproducibility_checksum" in result
