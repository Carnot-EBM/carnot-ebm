import os
import json
import pytest
from carnot.pipeline.findings_audit_1984 import generate_audit_report

def test_generate_audit_report(tmp_path):
    """
    Test that the audit report generation produces the correct
    schema structure and flags the correct number of artifacts.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    out_path = tmp_path / "results/experiment_1984_findings_audit_197.json"
    
    # Run the generator
    report = generate_audit_report(out_path=str(out_path), results_dir=str(results_dir))
    
    # Validate the report
    assert report["schema"] == "carnot.findings_audit_corrigenda.v11"
    assert report["experiment"] == 1984
    assert report["model_specs"]["artifacts_flagged"] == 3
    assert report["acceptance_gate_passed"] is True
    assert "experiment_1972" in report["audit_outcomes"]
    assert "experiment_1980" in report["audit_outcomes"]
    assert report["audit_outcomes"]["experiment_1980"]["classification"] == "REAL_BUG"
    
    assert os.path.exists(out_path)
