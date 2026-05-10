"""Tests for the experiment 1676 retrospective script."""

import json
from pathlib import Path
import pytest

from scripts.experiment_1676_retro import run, load_result

def test_load_result_existing(tmp_path):
    """Test loading an existing result file."""
    data = {"status": "complete", "honest_verdict": "success"}
    file_path = tmp_path / "test_result.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    
    result = load_result(tmp_path, "test_result.json")
    assert result == data

def test_load_result_missing(tmp_path):
    """Test loading a non-existent result file."""
    result = load_result(tmp_path, "missing_result.json")
    assert result == {}

def test_run_retro(tmp_path):
    """Test running the retrospective generation."""
    # Create mock result files
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Complete
    (results_dir / "experiment_1666_nsvif.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "ok"}), encoding="utf-8"
    )
    # Incomplete/Missing status
    (results_dir / "experiment_1667_ebcn.json").write_text(
        json.dumps({"status": "failed", "honest_verdict": "error"}), encoding="utf-8"
    )
    
    out_path = tmp_path / "out" / "retro.json"
    
    artifact = run(results_dir=results_dir, out_path=out_path)
    
    assert out_path.exists()
    saved_artifact = json.loads(out_path.read_text(encoding="utf-8"))
    
    assert artifact == saved_artifact
    assert artifact["experiment"] == 1676
    assert artifact["status"] == "complete"
    assert artifact["criteria_met"] == 1  # only exp1666 is complete
    assert artifact["criteria_total"] == 10
    assert "exp1666" in artifact["criteria_results"]
    assert artifact["criteria_results"]["exp1666"] is True
    assert artifact["criteria_results"]["exp1667"] is False
    assert artifact["criteria_results"]["exp1668"] is False # missing file returns {}

def test_run_retro_default_paths(monkeypatch, tmp_path):
    """Test running the retrospective with default paths."""
    # Monkeypatch REPO_ROOT in the script module
    import scripts.experiment_1676_retro as retro_module
    monkeypatch.setattr(retro_module, "REPO_ROOT", tmp_path)
    
    # Create the expected results dir
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_1666_nsvif.json").write_text(
        json.dumps({"status": "complete", "honest_verdict": "ok"}), encoding="utf-8"
    )
    
    # Should use tmp_path / "results"
    artifact = run()
    
    assert artifact["experiment"] == 1676
    out_path = tmp_path / "results" / "experiment_1676_retro.json"
    assert out_path.exists()
