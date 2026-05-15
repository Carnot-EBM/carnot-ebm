import json
import os
import pytest
from pathlib import Path
from scripts.audit_1796 import run_audit

# REQ-REPORT-1796: Findings Audit 1796
# The repository shall provide an audit script that verifies experiments in the .186 and .187 ranges.
# SCENARIO-REPORT-1796: Generating Audit Artifact

def test_audit_1796():
    """
    Test that the audit_1796 script generates the correct output artifact and appends corrigenda.
    """
    run_audit()
    
    assert os.path.exists("results/experiment_1796_findings_audit_186_187.json")
    with open("results/experiment_1796_findings_audit_186_187.json", "r") as fp:
        data = json.load(fp)
        
    assert data["schema"] == "carnot.findings_audit_corrigenda.v3"
    assert data["experiment"] == 1796
    assert "audit_outcomes" in data
    assert "corrigenda_added" in data
    assert data["acceptance_gate_passed"] is True
    assert data["honest_verdict"].startswith("complete:")
    
    # Check that corrigenda was added
    with open("results/experiment_1861_equivalence.json", "r") as fp:
        d1861 = json.load(fp)
        assert "corrigendum_2026_05_187_audit" in d1861

    with open("results/experiment_1862_e2e.json", "r") as fp:
        d1862 = json.load(fp)
        assert "corrigendum_2026_05_187_audit" in d1862

    with open("results/experiment_1864_roce.json", "r") as fp:
        d1864 = json.load(fp)
        assert "corrigendum_2026_05_187_audit" in d1864

    with open("results/experiment_1876_146_completion_147_gate_contract.json", "r") as fp:
        d1876 = json.load(fp)
        assert "corrigendum_2026_05_187_audit" in d1876

    with open("results/experiment_1877_artifact_contract_normalization.json", "r") as fp:
        d1877 = json.load(fp)
        assert "corrigendum_2026_05_187_audit" in d1877
