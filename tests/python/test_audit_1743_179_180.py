import os
import json
from scripts.audit_1743_179_180 import generate_audit_artifact

def test_generate_audit_artifact(tmp_path):
    output_path = tmp_path / "test_artifact.json"
    data = generate_audit_artifact(output_path)
    assert os.path.exists(output_path)
    assert data["experiment"] == 1743
    assert data["schema"] == "carnot.findings_audit_corrigenda.v5"
    assert "1724" in data["audit_outcomes"]
    
    with open(output_path, "r") as f:
        loaded = json.load(f)
    assert loaded["acceptance_gate_passed"] is True
    assert loaded["honest_verdict"].startswith("TERMINAL_VERDICT:")
