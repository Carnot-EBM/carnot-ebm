import os
import json
from carnot.phase1_recovery_1989 import generate_recovery_artifact

def test_generate_recovery_artifact(tmp_path):
    """
    Tests the phase 1 recovery artifact generation.
    Satisfies REQ-PUBLISH-029 and SCENARIO-PUBLISH-029.
    """
    mcp_path = tmp_path / "mcp.json"
    rep_path = tmp_path / "rep.json"
    out_path = tmp_path / "out.json"
    
    with open(mcp_path, "w") as f:
        json.dump({"acceptance_gate_passed": True}, f)
        
    with open(rep_path, "w") as f:
        json.dump({"acceptance_gate_passed": True}, f)
        
    artifact = generate_recovery_artifact(str(out_path), str(mcp_path), str(rep_path))
    
    assert os.path.exists(out_path)
    assert artifact["schema"] == "carnot.phase1_recovery.v1"
    assert artifact["experiment"] == 1989
    assert artifact["exp1981_mcp_docs_status"] == "shipped"
    assert artifact["exp1982_reproducer_status"] == "shipped"
    assert artifact["acceptance_gate_passed"] is True
    assert "Terminal success" in artifact["honest_verdict"]

def test_generate_recovery_artifact_missing(tmp_path):
    """
    Tests the recovery logic when one artifact is missing.
    """
    mcp_path = tmp_path / "mcp_missing.json"
    rep_path = tmp_path / "rep.json"
    out_path = tmp_path / "out_missing.json"
    
    with open(rep_path, "w") as f:
        json.dump({"acceptance_gate_passed": True}, f)
        
    artifact = generate_recovery_artifact(str(out_path), str(mcp_path), str(rep_path))
    
    assert artifact["exp1981_mcp_docs_status"] == "missing"
    assert artifact["exp1982_reproducer_status"] == "shipped"
    assert artifact["acceptance_gate_passed"] is False
    assert "Terminal failure" in artifact["honest_verdict"]
