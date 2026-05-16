import json
from pathlib import Path

# REQ-PUBLISH-026: PyPI Workflow Status Re-check
# SCENARIO-PUBLISH-028: PyPI Workflow Status Re-check correctly verifies and reports wait state
def test_experiment_1987_pypi_status_recheck_artifact():
    artifact_path = Path("results/experiment_1987_pypi_status_recheck.json")
    assert artifact_path.exists(), "Artifact missing"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.pypi_workflow_status_recheck.v1"
    assert data["experiment"] == 1987
    assert "honest_verdict" in data
    assert data["state_change_from_exp1930"] in ["no_change", "now_succeeded", "now_failed"]
    assert data["workflow_run_status"] in ["waiting", "in_progress", "success", "failure"]
    assert "Operator: approve at" in data["actionable_next_step"] or data["actionable_next_step"] == "none"
