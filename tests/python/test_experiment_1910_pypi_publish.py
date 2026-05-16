import json
from pathlib import Path

def test_experiment_1910_pypi_publish_artifact():
    artifact_path = Path("results/experiment_1910_pypi_publish.json")
    assert artifact_path.exists(), "Artifact missing"
    
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.pypi_publish.v4_ci_tagged_release"
    assert data["experiment"] == 1910
    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("blocked_") or "OK" in data["honest_verdict"] or "FAIL" in data["honest_verdict"]
