"""Test milestone .142 retrospective reporting.

References:
- REQ-REPORT-1838
- SCENARIO-REPORT-1838
"""

import json
from pathlib import Path

from carnot.reporting import milestone_retro_142


def test_milestone_retro_142_generation(tmp_path: Path):
    """Test generating the Phase 19 Final Evaluation Retrospective.
    
    Validates REQ-REPORT-1838.
    """
    # Create fake results directory and source artifacts
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Fake some artifacts
    (results_dir / "experiment_1825_activation.json").write_text(json.dumps({
        "status": "success",
        "honest_verdict": "milestone_142_activated"
    }))
    (results_dir / "experiment_1832_zero_violation.json").write_text(json.dumps({
        "status": "complete",
        "honest_verdict": "success_zero_violation_implemented"
    }))
    
    out_path = results_dir / "experiment_1838_retro.json"
    
    artifact = milestone_retro_142.run(results_dir=results_dir, out_path=out_path)
    
    assert artifact["experiment"] == 1838
    assert artifact["milestone"] == "2026.05.142"
    assert artifact["honest_verdict"] == "milestone_complete"
    
    # Verify file was written
    assert out_path.exists()
    written_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert written_data["experiment"] == 1838

def test_milestone_retro_142_defaults(monkeypatch, tmp_path: Path):
    """Test the default arguments."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Patch REPO_ROOT to point to our tmp_path parent
    monkeypatch.setattr(milestone_retro_142, "REPO_ROOT", tmp_path)
    
    # Create a malformed json file to trigger JSONDecodeError
    (results_dir / "experiment_1825_activation.json").write_text("{ invalid json")
    
    artifact = milestone_retro_142.run()
    assert artifact["experiment"] == 1838
    assert (results_dir / "experiment_1838_retro.json").exists()
