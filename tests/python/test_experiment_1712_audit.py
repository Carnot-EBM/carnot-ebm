import json
import os
from python.carnot.experiment_1712_audit import generate_audit_report

def test_experiment_1712_audit_deliverable():
    # Generate the report
    generate_audit_report()
    
    filepath = "results/experiment_1712_findings_audit_174.json"
    assert os.path.exists(filepath), f"{filepath} does not exist"
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data["schema"] == "carnot.findings_audit_corrigenda.v2"
    assert data["experiment"] == 1712
    assert data["n_samples"] == 19
    assert data["acceptance_gate_passed"] is True
    assert data["honest_verdict"].startswith("complete:")
    assert "2101" in data["audit_outcomes"]
    assert "2110" in data["audit_outcomes"]
    
    assert "recovered_in_175" in data["skip_recovery_summary"]
    assert "missed_in_174_carry_forward" in data["skip_recovery_summary"]
    
    # Check that corrigenda were added to the target files
    for fname in data["corrigenda_added"]:
        fpath = os.path.join("results", fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                target_data = json.load(f)
            assert "corrigendum" in target_data

