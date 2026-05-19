import json
from pathlib import Path
import pytest

def test_experiment_2515_artifact_schema():
    """Verify the deliverable artifact for experiment 2515 has all required fields."""
    artifact_path = Path("results/experiment_2515_paper_writethrough.json")
    if not artifact_path.exists():
        pytest.skip("Artifact not yet generated.")
        
    with open(artifact_path) as f:
        data = json.load(f)
        
    assert "honest_verdict" in data, "Terminal-prefix required."
    assert "paper_updated" in data, "Must track if paper was modified."
    assert "sections_updated" in data, "Audit trail required."
    assert "arxiv_ready" in data, "Final gate check result required."
    assert "arxiv_gates" in data, "Individual gate status required."
    assert "corrigendum_resolved" in data, "Must document resolved issues."
    assert "preconditions_checked" in data, "Must record verified resources."
    assert "duration_s" in data, "Wall-clock measurement required."
    
    assert data["paper_updated"] is True, "Paper must actually be modified."
    assert len(data["sections_updated"]) >= 1, "At least one section must be updated."
