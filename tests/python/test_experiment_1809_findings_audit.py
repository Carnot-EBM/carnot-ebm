import os
import json
import pytest
from carnot.pipeline.findings_audit_1809 import generate_audit_report

def test_generate_audit_report(tmp_path):
    # Setup mock results
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    mock_result_path = results_dir / "experiment_1800_fast_slow_variant_prototype.json"
    with open(mock_result_path, "w") as f:
        json.dump({
            "honest_verdict": "complete: fast_slow_variant",
            "metric_1": 42.0
        }, f)
        
    out_path = tmp_path / "results/experiment_1809_findings_audit.json"
    
    # Run the generator
    report = generate_audit_report(out_path=str(out_path), results_dir=str(results_dir))
    
    # Validate the report
    assert report["schema"] == "carnot.findings_audit.v1"
    assert report["artifacts_read_count"] == 1
    assert "2026.05.187" in report["milestones_audited"]
    assert report["acceptance_gate_passed"] is True
    assert os.path.exists(out_path)
