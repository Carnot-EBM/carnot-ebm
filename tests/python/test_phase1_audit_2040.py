import json
import os
from pathlib import Path
from carnot.reporting.phase1_audit_2040 import run_audit

def test_phase1_audit_2040_module(tmp_path):
    # Setup paths
    results_dir = tmp_path / "results"
    log_path = tmp_path / "ops" / "conductor-log.md"
    dashboard_path = tmp_path / "ops" / "phase-1-dashboard.md"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_path.parent, exist_ok=True)
    
    # Create fake files to test preconditions
    with open(results_dir / "experiment_1929_fast_slow_codification.json", "w") as f:
        f.write("{}")
    with open(results_dir / "experiment_1931_huggingface_mirror.json", "w") as f:
        f.write("{}")
        
    with open(log_path, "w") as f:
        f.write("SKIP-cascade due to unhealed pre-test failures starting after a 600s stall failure")

    # Run audit
    artifact = run_audit(str(results_dir), str(log_path), str(dashboard_path))
    
    # Verifications
    assert artifact["schema"] == "carnot.phase1_dashboard_audit.v2"
    assert artifact["experiment"] == 2040
    assert "exp1929_exists" in artifact["preconditions_checked"]
    assert "exp1931_exists" in artifact["preconditions_checked"]
    assert artifact["n_samples"] == 6
    assert artifact["phase_1_ship_percentage"] == 80
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    
    # Check JSON file written
    out_json = results_dir / "experiment_2040_phase1_dashboard_audit.json"
    assert out_json.exists()
    
    # Check Dashboard written
    assert dashboard_path.exists()
    with open(dashboard_path) as f:
        content = f.read()
        assert "80% (4/5 prongs shipped)" in content

    # Run audit again to cover the path where dashboard already exists
    artifact2 = run_audit(str(results_dir), str(log_path), str(dashboard_path))
    assert artifact2["experiment"] == 2040
