import json
import os
import pytest
from pathlib import Path
from scripts.audit_1717 import run_audit

# REQ-REPORT-1717: Findings Audit 1717
# The repository shall provide an audit script that verifies experiments in the .174 and .175 ranges.
# SCENARIO-REPORT-1717: Generating Audit Artifact

def test_audit_1717(tmp_path, monkeypatch):
    """
    Test that the audit_1717 script generates the correct output artifact and appends corrigenda.
    """
    # Create fake artifacts in the current directory so glob finds them
    # But mock the verify_artifact and git commands.
    
    # Actually, running the real run_audit() will create a file in results/
    # We will just assert that it executes without crashing on the actual results dir.
    
    # We should run the real audit
    run_audit()
    
    assert os.path.exists("results/experiment_1717_findings_audit.json")
    with open("results/experiment_1717_findings_audit.json", "r") as fp:
        data = json.load(fp)
        
    assert data["schema"] == "carnot.findings_audit_corrigenda.v3"
    assert data["experiment"] == 1717
    assert "audit_outcomes" in data
    assert "corrigenda_added" in data
    assert data["acceptance_gate_passed"] is True
    assert data["honest_verdict"].startswith("complete:")
    
    # Check that corrigenda was added to 2101 and 2110
    with open("results/experiment_2101_interwhen.json", "r") as fp:
        d2101 = json.load(fp)
        assert "corrigendum_2026_05_176_audit" in d2101
        
    with open("results/experiment_2110_casal_pinet.json", "r") as fp:
        d2110 = json.load(fp)
        assert "corrigendum_2026_05_176_audit" in d2110

