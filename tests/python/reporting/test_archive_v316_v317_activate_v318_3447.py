"""Tests for carnot.reporting.archive_v316_v317_activate_v318_3447.

REQ-REPORT-3447: Archive milestones .316 and .317, activate .318.
SCENARIO-REPORT-3447-01: write_artifact produces a JSON file with all required
  schema fields including archive_v316_v317_activate_v318_ready=True.
SCENARIO-REPORT-3447-02: honest_verdict starts with the required terminal prefix.
SCENARIO-REPORT-3447-03: inference_substrate is aggregation_from_upstream_artifacts.
SCENARIO-REPORT-3447-04: P0.1 v3 retirement is recorded with root_cause and fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.archive_v316_v317_activate_v318_3447 import write_artifact


# ── helpers ────────────────────────────────────────────────────────────────


@pytest.fixture()
def artifact(tmp_path, monkeypatch) -> dict:
    """Run write_artifact with the working directory set to tmp_path."""
    monkeypatch.chdir(tmp_path)
    out = write_artifact()
    return json.loads(out.read_text())


# ── SCENARIO-REPORT-3447-01 ────────────────────────────────────────────────


def test_required_fields_present(artifact):
    """REQ-REPORT-3447 SCENARIO-REPORT-3447-01: all required fields must exist."""
    required = {
        "schema",
        "experiment_id",
        "inference_substrate",
        "honest_verdict",
        "archived_milestones",
        "activated_milestone",
        "archive_v316_v317_activate_v318_ready",
        "g1", "g2", "g3", "g4",
        "unmet_gates",
        "paper_ready",
    }
    missing = required - set(artifact)
    assert not missing, f"Missing required fields: {missing}"


def test_archive_ready_flag(artifact):
    """REQ-REPORT-3447: archive_v316_v317_activate_v318_ready must be True."""
    assert artifact["archive_v316_v317_activate_v318_ready"] is True


def test_archived_milestones(artifact):
    """Both .316 and .317 must appear in archived_milestones."""
    milestones = artifact["archived_milestones"]
    assert "2026.05.316" in milestones
    assert "2026.05.317" in milestones


def test_activated_milestone(artifact):
    """Activated milestone must be .318."""
    assert artifact["activated_milestone"] == "2026.05.318"


# ── SCENARIO-REPORT-3447-02 ────────────────────────────────────────────────


def test_honest_verdict_terminal_prefix(artifact):
    """REQ-REPORT-3447 SCENARIO-REPORT-3447-02: honest_verdict must start with 'complete:'."""
    assert artifact["honest_verdict"].startswith("complete:")


# ── SCENARIO-REPORT-3447-03 ────────────────────────────────────────────────


def test_inference_substrate(artifact):
    """REQ-REPORT-3447 SCENARIO-REPORT-3447-03: inference_substrate must be aggregation."""
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


# ── SCENARIO-REPORT-3447-04 ────────────────────────────────────────────────


def test_p0_1_retirement_recorded(artifact):
    """REQ-REPORT-3447 SCENARIO-REPORT-3447-04: P0.1 v3 retirement with root_cause and fix."""
    retired = artifact.get("experiments_retired_317", [])
    assert len(retired) >= 1
    p01 = next((e for e in retired if "3437" in str(e.get("experiment_id", ""))), None)
    assert p01 is not None, "exp3437 retirement must be recorded"
    assert "root_cause" in p01
    assert "fix_for_318" in p01


def test_p0_1_hypothesis_unanswered(artifact):
    """P0.1 hypothesis must still be marked unanswered after 3x timeout."""
    assert artifact.get("p0_1_hypothesis_answered") is False


# ── G-gate state ───────────────────────────────────────────────────────────


def test_g1_g3_g4_met(artifact):
    """G1, G3, G4 must be True; G2 is the sole unmet gate."""
    assert artifact["g1"] is True
    assert artifact["g3"] is True
    assert artifact["g4"] is True


def test_g2_not_met(artifact):
    """G2 must be False — external reproducer still pending."""
    assert artifact["g2"] is False


def test_unmet_gates_contains_g2(artifact):
    """unmet_gates list must include G2."""
    assert "G2" in artifact["unmet_gates"]


def test_paper_not_ready(artifact):
    """paper_ready must be False while G2 is unmet."""
    assert artifact["paper_ready"] is False


# ── forward gap ────────────────────────────────────────────────────────────


def test_next_top_gap_present(artifact):
    """next_top_gap field must be non-empty and mention P0.1 decoupling."""
    gap = artifact.get("next_top_gap", "")
    assert gap
    assert "P0.1" in gap or "p0_1" in gap.lower()


def test_depth_forcing_function_active(artifact):
    """Depth-Over-Breadth forcing function must remain active for .318."""
    assert artifact.get("depth_forcing_function_active") is True
    assert artifact.get("depth_forcing_function_can_relax") is False
