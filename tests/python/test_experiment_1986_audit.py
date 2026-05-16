import json
import os

def test_experiment_1986_audit_json_validity():
    json_path = "results/experiment_1986_findings_audit_198_199.json"
    assert os.path.exists(json_path), "JSON artifact must exist"
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.findings_audit_corrigenda.v12"
    assert data["experiment"] == 1986
    assert data["duration_s"] > 30
    assert "model_specs" in data
    assert data["n_samples"] == 3
    assert data["acceptance_gate_passed"] is True
    assert data["honest_verdict"].startswith("complete:")
    
    # Check that we proposed .201+ follow-ups
    assert "1998_live_it_baselines_gsm8k" in data["audit_outcomes"]
    assert ".201+" in data["audit_outcomes"]["1998_live_it_baselines_gsm8k"]["rationale"]
