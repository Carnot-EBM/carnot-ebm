"""Tests for the exp3416 archive v314 / activate v315 artifact.

Spec: REQ-REPORT-3416, SCENARIO-REPORT-3416.
"""
from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.archive_v314_activate_v315_3416 import write_artifact


def test_write_artifact_returns_path(tmp_path, monkeypatch):
    """write_artifact() returns a Path pointing at the written file."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    assert isinstance(result, Path)
    assert result.exists()


def test_artifact_is_valid_json(tmp_path, monkeypatch):
    """The artifact file contains valid JSON."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert isinstance(payload, dict)


def test_honest_verdict_starts_with_complete(tmp_path, monkeypatch):
    """honest_verdict must start with 'complete:' per Verdict Terminal-Prefix Discipline."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["honest_verdict"].startswith("complete:")


def test_archive_activate_ready_flag(tmp_path, monkeypatch):
    """archive_v314_activate_v315_ready must be True."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["archive_v314_activate_v315_ready"] is True


def test_inference_substrate_is_aggregation(tmp_path, monkeypatch):
    """Inference substrate must be aggregation_from_upstream_artifacts (no LLM invoked)."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_flagged_adversarial_artifacts_quarantined(tmp_path, monkeypatch):
    """exp3397 and exp3405 must appear in flagged_adversarial_artifacts."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    flagged_ids = [f["experiment_id"] for f in payload["flagged_adversarial_artifacts"]]
    assert "exp3397" in flagged_ids
    assert "exp3405" in flagged_ids


def test_archived_and_activated_milestones(tmp_path, monkeypatch):
    """Correct milestones must be named in the artifact."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["archived_milestone"] == "2026.05.314"
    assert payload["activated_milestone"] == "2026.05.315"


def test_next_top_gap_mentions_depth_over_breadth(tmp_path, monkeypatch):
    """next_top_gap must reference the P0.1/P0.2/Kona/G2 depth block."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    gap = payload["next_top_gap"].lower()
    assert "p0.1" in gap or "depth" in gap
