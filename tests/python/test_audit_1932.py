import json
from pathlib import Path

def test_audit_1932_artifact_exists_and_schema_valid():
    """Verify that experiment_1932_findings_audit_195.json exists and has correct schema."""
    artifact_path = Path("results/experiment_1932_findings_audit_195.json")
    assert artifact_path.exists(), "Artifact file must exist"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.findings_audit_corrigenda.v10", "Schema must be v10"
    assert data["experiment"] == 1932, "Experiment ID must be 1932"
    assert "honest_verdict" in data, "Must have honest_verdict"
    assert data["honest_verdict"].startswith("success:"), "Verdict must be terminal success:"
    assert data["n_samples_justification"] == "Audit; n is flagged count.", "Sample justification required"
    assert data["acceptance_gate_passed"] is True, "Must pass acceptance gate"
    assert "experiment_1923" in data["audit_outcomes"], "Must flag experiment_1923"
