"""Tests for the exp3436 archive v316 / activate v317 artifact.

Spec: REQ-REPORT-3436, SCENARIO-REPORT-3436.
"""
from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.archive_v316_activate_v317_3436 import write_artifact


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
    """archive_v316_activate_v317_ready must be True."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["archive_v316_activate_v317_ready"] is True


def test_inference_substrate_is_aggregation(tmp_path, monkeypatch):
    """Inference substrate must be aggregation_from_upstream_artifacts (no LLM invoked)."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_archived_and_activated_milestones(tmp_path, monkeypatch):
    """Correct milestones must be named in the artifact."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["archived_milestone"] == "2026.05.316"
    assert payload["activated_milestone"] == "2026.05.317"


def test_missing_artifacts_include_p02_kona_injection(tmp_path, monkeypatch):
    """P0.2, Kona gate, and injection must be listed as missing (gemini-cli crash)."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    missing = payload["missing_artifacts"]
    assert any("kona" in m.lower() for m in missing)
    assert any("injection" in m.lower() or "ensemble" in m.lower() for m in missing)
    assert any("p0.2" in m.lower() or "null-space" in m.lower() for m in missing)


def test_capstone_is_flagged(tmp_path, monkeypatch):
    """Capstone exp3435 must be listed as flagged_adversarial."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["capstone_flagged_adversarial"] is True
    flagged_ids = [f["experiment_id"] for f in payload["flagged_adversarial_artifacts"]]
    assert "exp3435" in flagged_ids


def test_depth_forcing_function_remains_active(tmp_path, monkeypatch):
    """depth_forcing_function_can_relax must be False — P0.1 harness broken + G2 unmet."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["depth_forcing_function_can_relax"] is False
    assert payload["depth_forcing_function_active"] is True


def test_g2_remains_unmet(tmp_path, monkeypatch):
    """G2 must still be unmet — cleanroom CI gate failed."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["g2"] is False
    assert "G2" in payload["unmet_gates"]
    assert payload["paper_ready"] is False


def test_p0_1_v2_is_clean(tmp_path, monkeypatch):
    """P0.1 v2 must be marked clean (run was authentic, harness was broken)."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    assert payload["p0_1_v2_is_clean"] is True


def test_gemini_cli_outage_note_present(tmp_path, monkeypatch):
    """The artifact must document the gemini-cli outage as root cause of 3 missing tasks."""
    monkeypatch.chdir(tmp_path)
    result = write_artifact()
    payload = json.loads(result.read_text())
    note = payload.get("gemini_cli_outage_note", "")
    assert "gemini" in note.lower()
    assert "crash" in note.lower() or "error" in note.lower()
