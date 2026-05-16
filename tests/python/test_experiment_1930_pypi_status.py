import json
from pathlib import Path

def test_experiment_1930_pypi_status_artifact():
    artifact_path = Path("results/experiment_1930_pypi_status.json")
    assert artifact_path.exists(), "Artifact missing"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.pypi_workflow_status_check.v1"
    assert data["experiment"] == 1930
    assert "honest_verdict" in data
    assert data["actionable_next_step"] in ["operator_approve", "verify_install", "investigate_workflow_failure", "no_action"]
    assert data["workflow_run_status"] in ["waiting", "in_progress", "success", "failure", "not_triggered"]
