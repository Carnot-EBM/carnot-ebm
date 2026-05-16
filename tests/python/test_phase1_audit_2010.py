import os
import json
from carnot.reporting.phase1_audit_2010 import run_audit

def test_audit(tmp_path):
    """
    REQ-REPORT-2010: The pipeline SHALL run a consolidated audit for milestones .198 to .201, 
    generate a Phase 1 dashboard, and write results/experiment_2010_consolidated_audit.json.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create fake preconditions
    with open(results_dir / "experiment_1929_fast_slow_codification.json", "w") as f:
        f.write("{}")
    with open(results_dir / "experiment_1931_huggingface_mirror.json", "w") as f:
        f.write("{}")
        
    log_path = tmp_path / "conductor-log.md"
    with open(log_path, "w") as f:
        f.write("No issues")
        
    dashboard_path = tmp_path / "phase-1-dashboard.md"
    
    # First run (dashboard doesn't exist)
    artifact = run_audit(results_dir=str(results_dir), log_path=str(log_path), dashboard_path=str(dashboard_path))
        
    assert artifact["schema"] == "carnot.consolidated_audit_dashboard.v1"
    assert artifact["experiment"] == 2010
    assert artifact["acceptance_gate_passed"] is True
    assert "complete:" in artifact["honest_verdict"]
    assert artifact["dashboard_has_emojis"] is False
    assert os.path.exists(dashboard_path)

    # Second run (dashboard exists, and log has SKIP)
    with open(log_path, "w") as f:
        f.write("SKIP 2026-05-16T05:10:00")
        
    artifact2 = run_audit(results_dir=str(results_dir), log_path=str(log_path), dashboard_path=str(dashboard_path))
    assert artifact2["skip_cascade_observed"] is True

    # Test missing adversarial_verify by changing cwd to tmp_path
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        artifact3 = run_audit(results_dir="results", log_path="missing.md", dashboard_path="missing.md")
        assert "adversarial_verify_present" not in artifact3["preconditions_checked"]
    finally:
        os.chdir(orig_cwd)
