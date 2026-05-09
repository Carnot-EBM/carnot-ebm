"""Tests for the Exp 1639 milestone .125 retrospective.

Spec: REQ-REPORT-125, SCENARIO-REPORT-125.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_125 as retro125


def test_milestone_retro_125_produces_artifact(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Write a dummy source file for an existing task
    (results_dir / "experiment_1627_nabla_debug.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "success"})
    )
    
    # Let others be missing, it should handle it gracefully
    
    out_path = results_dir / "experiment_1639_retro.json"
    artifact = retro125.run(results_dir, out_path)
    
    assert out_path.exists()
    assert artifact["experiment"] == 1639
    assert artifact["status"] == "complete"
    assert artifact["criteria_results"]["exp1627"] is True
    assert artifact["criteria_results"]["exp1628"] is False
    assert artifact["criteria_details"]["exp1627"]["status"] == "complete"
    assert artifact["criteria_details"]["exp1628"]["status"] == "MISSING"

def test_milestone_retro_125_default_paths(tmp_path: Path, monkeypatch) -> None:
    # Use monkeypatch to point REPO_ROOT to tmp_path for the test
    monkeypatch.setattr(retro125, "REPO_ROOT", tmp_path)
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    (results_dir / "experiment_1627_nabla_debug.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "success"})
    )
    
    artifact = retro125.run()
    
    out_path = results_dir / "experiment_1639_retro.json"
    assert out_path.exists()
    assert artifact["experiment"] == 1639
