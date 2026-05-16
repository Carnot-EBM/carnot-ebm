import json
from pathlib import Path

# REQ-PUBLISH-026: PyPI Workflow Status Re-check
# SCENARIO-PUBLISH-029: PyPI Workflow Final Re-check correctly verifies and reports timeout state
def test_experiment_2011_pypi_final_recheck_artifact():
    artifact_path = Path("results/experiment_2011_pypi_final_recheck.json")
    assert artifact_path.exists(), "Artifact missing"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.pypi_workflow_final_recheck.v1"
    assert data["experiment"] == 2011
    assert "honest_verdict" in data
    assert data["state_change_from_prior"] in ["no_change", "now_succeeded", "now_failed", "now_cancelled"]
    assert data["workflow_run_status"] in ["waiting", "in_progress", "success", "failure", "cancelled"]
    assert "re-trigger via workflow_dispatch" in data["actionable_next_step"] or data["actionable_next_step"] == "none"
