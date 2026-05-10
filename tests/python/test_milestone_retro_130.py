"""Tests for the milestone .130 retrospective generator."""

import json
from datetime import datetime, timezone, UTC
from typing import Generator

import pytest
from pathlib import Path

from carnot.reporting.milestone_retro_130 import run, load_result

@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a temporary directory with mock experiment results."""
    # Create complete result
    (tmp_path / "experiment_1682_archive.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "archive_success"})
    )
    # Create partial result
    (tmp_path / "experiment_1683_self_play.json").write_text(
        json.dumps({"status": "failed", "honest_verdict": "failed_self_play"})
    )
    # The others will be missing
    yield tmp_path


def test_load_result_missing(tmp_path: Path) -> None:
    """Test load_result with missing file."""
    assert load_result(tmp_path, "does_not_exist.json") == {}


def test_load_result_exists(temp_results_dir: Path) -> None:
    """Test load_result with existing file."""
    res = load_result(temp_results_dir, "experiment_1682_archive.json")
    assert res == {"status": "complete", "honest_verdict": "archive_success"}


def test_run(temp_results_dir: Path) -> None:
    """Test the main run function."""
    out_path = temp_results_dir / "retro.json"
    artifact = run(results_dir=temp_results_dir, out_path=out_path)

    # Check generated artifact
    assert artifact["experiment"] == 1695
    assert artifact["schema"] == "carnot.experiment.v1"
    assert artifact["title"] == "Milestone 2026.05.130 Retrospective"
    assert artifact["status"] == "complete"
    assert artifact["criteria_total"] == 13
    
    # We mocked 1 complete, 1 failed, 11 missing
    assert artifact["criteria_met"] == 1
    assert artifact["honest_verdict"] == "milestone_130_retrospective_filed_1_of_13_complete"
    
    # Check details
    assert artifact["criteria_results"]["exp1682"] is True
    assert artifact["criteria_results"]["exp1683"] is False
    assert artifact["criteria_results"]["exp1684"] is False
    
    assert artifact["criteria_details"]["exp1682"]["status"] == "complete"
    assert artifact["criteria_details"]["exp1683"]["status"] == "failed"
    assert artifact["criteria_details"]["exp1684"]["status"] == "MISSING"

    # Check file was written
    assert out_path.exists()
    written_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert written_data["experiment"] == 1695

def test_run_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test the main run function with default paths."""
    # We patch REPO_ROOT to tmp_path to avoid dirtying real directories
    import carnot.reporting.milestone_retro_130
    monkeypatch.setattr(carnot.reporting.milestone_retro_130, "REPO_ROOT", tmp_path)
    
    # Create empty results dir
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    artifact = run()
    assert artifact["criteria_met"] == 0
    assert artifact["criteria_total"] == 13
    assert artifact["experiment"] == 1695
    
    out_path = results_dir / "experiment_1695_retro.json"
    assert out_path.exists()
