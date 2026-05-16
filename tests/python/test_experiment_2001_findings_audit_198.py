import os
import json
import pytest
from carnot.pipeline.findings_audit_2001 import generate_audit_report

def test_generate_audit_report_2001(tmp_path):
    """
    REQ-REPORT-2001: Test that the audit report generation produces the correct
    schema structure and flags the correct number of artifacts for .198 audit.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    out_path = tmp_path / "results/experiment_2001_findings_audit_198.json"
    
    # Run the generator
    report = generate_audit_report(out_path=str(out_path), results_dir=str(results_dir))
    
    # Validate the report
    assert report["schema"] == "carnot.findings_audit_corrigenda.v11"
    assert report["experiment"] == 2001
    assert report["model_specs"]["artifacts_flagged"] == 1
    assert report["acceptance_gate_passed"] is True
    assert "experiment_1980" in report["audit_outcomes"]
    assert report["audit_outcomes"]["experiment_1980"]["classification"] == "REAL_BUG"
    
    assert os.path.exists(out_path)
