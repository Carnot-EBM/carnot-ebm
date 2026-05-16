import json
from pathlib import Path

def test_audit_1912_artifact_exists_and_schema_valid():
    """Verify that experiment_1912_findings_audit_193.json exists and has correct schema."""
    artifact_path = Path("results/experiment_1912_findings_audit_193.json")
    assert artifact_path.exists(), "Artifact file must exist"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.findings_audit_corrigenda.v9", "Schema must be v9"
    assert data["experiment"] == 1912, "Experiment ID must be 1912"
    assert "honest_verdict" in data, "Must have honest_verdict"
    assert data["honest_verdict"].startswith("success:"), "Verdict must be terminal success:"
    assert "skip_cascade_diagnosis" in data, "Must diagnose SKIP cascade"
    assert data["skip_cascade_diagnosis"]["root_cause"] != "", "Root cause must not be empty"
    assert data["n_samples_justification"] == "Audit; n is flagged count.", "Sample justification required"
