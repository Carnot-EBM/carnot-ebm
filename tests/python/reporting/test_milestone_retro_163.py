"""Test milestone 163 retrospective reporting.

References:
- REQ-REPORT-2089
"""

import json
from pathlib import Path

from carnot.reporting import milestone_retro_163


def test_milestone_retro_163_generation(tmp_path: Path):
    """Test generating the Milestone 163 Retrospective.
    
    Validates REQ-REPORT-2089.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    out_path = results_dir / "experiment_2089_retro.json"
    
    artifact = milestone_retro_163.run(results_dir=results_dir, out_path=out_path)
    
    assert artifact["experiment_id"] == 2089
    assert artifact["milestone"] == "163"
    assert "smt_jepa_scaffolding" in artifact["honest_verdict"]
    assert artifact["criteria_met"] == 2
    
    # Verify file was written
    assert out_path.exists()
    written_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert written_data["experiment_id"] == 2089

def test_milestone_retro_163_defaults(monkeypatch, tmp_path: Path):
    """Test the default arguments."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    monkeypatch.setattr(milestone_retro_163, "REPO_ROOT", tmp_path)
    
    artifact = milestone_retro_163.run()
    assert artifact["experiment_id"] == 2089
    assert (results_dir / "experiment_2089_retro.json").exists()
