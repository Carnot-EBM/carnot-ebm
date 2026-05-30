"""Tests for the exp3425 archive v315 / activate v316 artifact.

Spec: REQ-REPORT-3425, SCENARIO-REPORT-3425.
"""
from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.archive_v315_activate_v316_3425 import write_artifact


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
    """archive_v315_activate_v316_ready must be True."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["archive_v315_activate_v316_ready"] is True


def test_inference_substrate_is_aggregation(tmp_path, monkeypatch):
    """Inference substrate must be aggregation_from_upstream_artifacts (no LLM invoked)."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_flagged_adversarial_artifacts_include_exp3312(tmp_path, monkeypatch):
    """exp3312 (flagged P0.1) must appear in flagged_adversarial_artifacts."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    flagged_ids = [f["experiment_id"] for f in payload["flagged_adversarial_artifacts"]]
    assert "exp3312" in flagged_ids


def test_archived_and_activated_milestones(tmp_path, monkeypatch):
    """Correct milestones must be named in the artifact."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["archived_milestone"] == "2026.05.315"
    assert payload["activated_milestone"] == "2026.05.316"


def test_missing_artifacts_include_kona_and_injection(tmp_path, monkeypatch):
    """Kona gate and ensemble-vs-injection must be listed as missing."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    missing = payload["missing_artifacts"]
    assert any("kona" in m for m in missing)
    assert any("injection" in m or "ensemble" in m for m in missing)


def test_depth_forcing_function_remains_active(tmp_path, monkeypatch):
    """depth_forcing_function_can_relax must be False — capstone was premature."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["depth_forcing_function_can_relax"] is False
    assert payload["depth_forcing_function_active"] is True


def test_g2_remains_unmet(tmp_path, monkeypatch):
    """G2 must still be unmet — external reproducer has not run."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["g2"] is False
    assert "G2" in payload["unmet_gates"]
    assert payload["paper_ready"] is False
