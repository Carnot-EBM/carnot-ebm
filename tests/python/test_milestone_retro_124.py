"""Tests for the Exp 1626 milestone .124 retrospective.

Spec: REQ-REPORT-124, SCENARIO-REPORT-124.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_124 as retro124


def test_milestone_retro_124_produces_artifact(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Write a dummy source file for an existing task
    (results_dir / "experiment_1614_archive.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "success"})
    )
    
    # Let others be missing, it should handle it gracefully
    
    out_path = results_dir / "experiment_1626_retro.json"
    artifact = retro124.run(results_dir, out_path)
    
    assert out_path.exists()
    assert artifact["experiment"] == 1626
    assert artifact["status"] == "complete"
    assert artifact["criteria_results"]["exp1614"] is True
    assert artifact["criteria_results"]["exp1615"] is False
    assert artifact["criteria_details"]["exp1614"]["status"] == "complete"
    assert artifact["criteria_details"]["exp1615"]["status"] == "MISSING"

def test_milestone_retro_124_default_paths(tmp_path: Path, monkeypatch) -> None:
    # Use monkeypatch to point REPO_ROOT to tmp_path for the test
    monkeypatch.setattr(retro124, "REPO_ROOT", tmp_path)
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    (results_dir / "experiment_1614_archive.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "success"})
    )
    
    artifact = retro124.run()
    
    out_path = results_dir / "experiment_1626_retro.json"
    assert out_path.exists()
    assert artifact["experiment"] == 1626
