import os
import json
import pytest
from carnot.phase4_report import generate_report

def test_generate_report(tmp_path):
    os.makedirs(tmp_path / "results", exist_ok=True)
    
    with open(tmp_path / "results/experiment_2474_phase4_odar_empirical.json", "w") as f:
        json.dump({"odar_energy_auroc": 0.55, "pearson_r": 0.2}, f)
    with open(tmp_path / "results/experiment_2455_odar_free_energy_routing.json", "w") as f:
        json.dump({"odar_routing_implemented": True}, f)
        
    res = generate_report(workspace_dir=str(tmp_path))
    
    assert res["phase4_hold_status"] == "partially_validated"
    assert res["phase4_claim_supported"] is True
    assert res["report_written"] is True
    assert res["odar_energy_auroc"] == 0.55
    assert res["preconditions_checked"] is True
    
    assert os.path.exists(tmp_path / "docs/research-notes/phase4-empirical-validation-report.md")
    assert os.path.exists(tmp_path / "results/experiment_2480_phase4_empirical_report.json")

def test_generate_report_no_files(tmp_path):
    res = generate_report(workspace_dir=str(tmp_path))
    assert res["phase4_hold_status"] == "empirical_evidence_pending"
    assert res["phase4_claim_supported"] is False
    assert res["preconditions_checked"] is False
