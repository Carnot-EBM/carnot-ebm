import os
import json

def test_gatemate_smoke_json_artifact():
    """
    Validates REQ-HW-058 and SCENARIO-HW-058.
    Ensures the GateMate smoke test artifact exists and conforms to the schema.
    """
    artifact_path = "results/experiment_2105_gatemate_smoke.json"
    assert os.path.exists(artifact_path), "Artifact must exist"
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data.get("schema") == "carnot.gatemate_smoke.v1"
    assert data.get("board") == "Cologne Chip GateMate A1-EVB-2M"
    assert data.get("n_spins") == 16
    assert "toolchain" in data
    assert "honest_verdict" in data
    
    verdict = data.get("honest_verdict", "")
    assert verdict.startswith("complete:") or verdict.startswith("success:") or verdict.startswith("shipped:") or verdict.startswith("passed:"), "Verdict must follow Terminal-Prefix Discipline"
