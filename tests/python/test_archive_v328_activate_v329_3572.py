"""Tests for the exp3572 archive v328 / activate v329 artifact.

Spec: REQ-REPORT-3572
"""
from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.archive_v328_activate_v329_3572 import write_artifact

def test_write_artifact_returns_path(tmp_path, monkeypatch):
    """write_artifact() returns a Path pointing at the written file."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    assert isinstance(result, Path)
    assert result.exists()

def test_artifact_has_required_fields(tmp_path, monkeypatch):
    """The artifact file contains required fields."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    
    assert payload["honest_verdict"] == "complete: archived_v328_p01_honest_negative_g2_closed_v329_verifier_pivot_active"
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert payload["p01_recorded_as"] == "honest-negative"
    assert payload["g2_recorded_as"] == "closed/paper_ready"
    assert isinstance(payload["n_tasks_archived"], int)
    assert payload["random_seed"] == 3572
    assert isinstance(payload["reproducibility_checksum"], str)
    assert payload["duration_s"] == 0.1
