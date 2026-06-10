"""Tests for Experiment 2070 — Milestone Activation.

Spec: REQ-DOCS-001
"""

import json
from pathlib import Path


def test_roadmap_doc_renamed():
    """Verify that the vNEXT roadmap document was renamed to the current milestone."""
    repo_root = Path(__file__).parent.parent.parent
    roadmap_path = repo_root / "openspec/change-proposals/research-roadmap-v2026.05.207.md"
    assert roadmap_path.exists(), "Renamed roadmap doc must exist"
    
    old_roadmap_path = repo_root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    assert not old_roadmap_path.exists(), "Old roadmap doc should not exist"


def test_result_file_has_required_fields():
    """The written JSON result file contains all required schema fields."""
    result_path = (
        Path(__file__).parent.parent.parent / "results" / "experiment_2070_activation.json"
    )
    assert result_path.exists(), "Result file must exist after experiment run"

    data = json.loads(result_path.read_text())

    required = [
        "experiment",
        "run_date",
        "status",
        "honest_verdict",
    ]
    for field in required:
        assert field in data, f"Missing required field: {field}"

    assert data["experiment"] == 2070
    assert data["honest_verdict"] == "activation_complete"
