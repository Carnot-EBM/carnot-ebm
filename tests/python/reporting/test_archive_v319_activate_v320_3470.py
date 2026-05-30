"""Tests for carnot.reporting.archive_v319_activate_v320_3470.

REQ-REPORT-3470: Archive milestone .319, activate .320.

SCENARIO-REPORT-3470-01: write_artifact produces a JSON file with all required
  schema fields including archive_v319_activate_v320_ready=True.
SCENARIO-REPORT-3470-02: honest_verdict starts with the required terminal prefix.
SCENARIO-REPORT-3470-03: inference_substrate is aggregation_from_upstream_artifacts.
SCENARIO-REPORT-3470-04: .319 key finding (trained energy AUROC 0.629, ceiling tie) recorded.
SCENARIO-REPORT-3470-05: G-gate state is G1/G3/G4=True, G2=False.
SCENARIO-REPORT-3470-06: depth_forcing_function remains active.
SCENARIO-REPORT-3470-07: retro JSON is written with correct schema.
SCENARIO-REPORT-3470-08: changelog entry is appended (not replacing content).
SCENARIO-REPORT-3470-09: compute_milestone_stats returns correct aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.archive_v319_activate_v320_3470 import (
    compute_milestone_stats,
    write_artifact,
)


# ── fixture ────────────────────────────────────────────────────────────────


@pytest.fixture()
def artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run write_artifact in a tmp_path that mirrors ops/ and results/."""
    # Create minimal changelog so append_changelog_entry can read it.
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    changelog = ops_dir / "changelog.md"
    changelog.write_text("# Carnot — Changelog\n\nPrior content.\n", encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    monkeypatch.chdir(tmp_path)
    out = write_artifact(repo_root=tmp_path)
    return json.loads(out.read_text())


# ── SCENARIO-REPORT-3470-01 ────────────────────────────────────────────────


def test_required_fields_present(artifact: dict) -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-01: all required fields must exist."""
    required = {
        "schema",
        "experiment_id",
        "inference_substrate",
        "honest_verdict",
        "archived_milestone",
        "activated_milestone",
        "archive_v319_activate_v320_ready",
        "g1", "g2", "g3", "g4",
        "unmet_gates",
        "paper_ready",
        "next_top_gap",
        "retro_path",
    }
    missing = required - set(artifact)
    assert not missing, f"Missing required fields: {missing}"


def test_archive_ready_flag(artifact: dict) -> None:
    """REQ-REPORT-3470: archive_v319_activate_v320_ready must be True."""
    assert artifact["archive_v319_activate_v320_ready"] is True


def test_archived_milestone(artifact: dict) -> None:
    """Archived milestone must be .319."""
    assert artifact["archived_milestone"] == "2026.05.319"


def test_activated_milestone(artifact: dict) -> None:
    """Activated milestone must be .320."""
    assert artifact["activated_milestone"] == "2026.05.320"


# ── SCENARIO-REPORT-3470-02 ────────────────────────────────────────────────


def test_honest_verdict_terminal_prefix(artifact: dict) -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-02: honest_verdict must start with 'complete:'."""
    assert artifact["honest_verdict"].startswith("complete:")


# ── SCENARIO-REPORT-3470-03 ────────────────────────────────────────────────


def test_inference_substrate(artifact: dict) -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-03: substrate must be aggregation_from_upstream_artifacts."""
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


# ── SCENARIO-REPORT-3470-04 ────────────────────────────────────────────────


def test_trained_energy_auroc_recorded(artifact: dict) -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-04: AUROC 0.629 from exp3461 must be present."""
    auroc = artifact.get("trained_energy_correctness_auroc_319")
    assert auroc is not None
    assert abs(auroc - 0.629401) < 1e-4


def test_auroc_lift_recorded(artifact: dict) -> None:
    """Lift over untrained baseline (+0.113) must be recorded."""
    lift = artifact.get("trained_energy_auroc_lift_319")
    assert lift is not None
    assert abs(lift - 0.113401) < 1e-4


def test_p01_hypothesis_not_answered(artifact: dict) -> None:
    """P0.1 hypothesis must remain False — ceiling tie is not the final answer."""
    assert artifact["p01_hypothesis_answered"] is False


def test_p01_next_step_mentions_headroom(artifact: dict) -> None:
    """p01_next_step must mention headroom or a benchmark with non-saturated SC."""
    step = artifact.get("p01_next_step", "").lower()
    assert "headroom" in step or "sc ~" in step or "ceiling" in step


# ── SCENARIO-REPORT-3470-05 ────────────────────────────────────────────────


def test_g1_g3_g4_met(artifact: dict) -> None:
    """G1, G3, G4 must be True; G2 is the sole unmet gate."""
    assert artifact["g1"] is True
    assert artifact["g3"] is True
    assert artifact["g4"] is True


def test_g2_not_met(artifact: dict) -> None:
    """G2 must be False — external reproducer still pending."""
    assert artifact["g2"] is False


def test_unmet_gates_contains_g2(artifact: dict) -> None:
    """unmet_gates list must include G2."""
    assert "G2" in artifact["unmet_gates"]


def test_paper_not_ready(artifact: dict) -> None:
    """paper_ready must be False while G2 is unmet."""
    assert artifact["paper_ready"] is False


# ── SCENARIO-REPORT-3470-06 ────────────────────────────────────────────────


def test_depth_forcing_function_active(artifact: dict) -> None:
    """Depth-Over-Breadth forcing function must remain active for .320."""
    assert artifact.get("depth_forcing_function_active") is True
    assert artifact.get("depth_forcing_function_can_relax") is False


def test_next_top_gap_present(artifact: dict) -> None:
    """next_top_gap must be non-empty and reference P0.1."""
    gap = artifact.get("next_top_gap", "")
    assert gap
    assert "P0.1" in gap or "p0_1" in gap.lower()


# ── SCENARIO-REPORT-3470-07 ────────────────────────────────────────────────


def test_retro_file_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-07: retro JSON must be written."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    (ops_dir / "changelog.md").write_text(
        "# Carnot — Changelog\n", encoding="utf-8"
    )
    (tmp_path / "results").mkdir()
    monkeypatch.chdir(tmp_path)
    write_artifact(repo_root=tmp_path)

    retro_path = tmp_path / "results" / "operational_retro_2026_05_319.json"
    assert retro_path.exists()
    retro = json.loads(retro_path.read_text())
    assert retro.get("schema") == "carnot.operational_retro.v65"
    assert retro.get("milestone") == "2026.05.319"


# ── SCENARIO-REPORT-3470-08 ────────────────────────────────────────────────


def test_changelog_entry_appended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-08: changelog must be updated without deleting prior content."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    existing_text = "# Carnot — Changelog\n\nPrior milestone entry.\n"
    changelog = ops_dir / "changelog.md"
    changelog.write_text(existing_text, encoding="utf-8")
    (tmp_path / "results").mkdir()

    monkeypatch.chdir(tmp_path)
    write_artifact(repo_root=tmp_path)

    updated = changelog.read_text()
    assert "Prior milestone entry." in updated, "Existing content was removed — violation!"
    assert ".319" in updated
    assert ".320" in updated


# ── SCENARIO-REPORT-3470-09 ────────────────────────────────────────────────


def test_compute_milestone_stats() -> None:
    """REQ-REPORT-3470 SCENARIO-REPORT-3470-09: stats function returns correct aggregates."""
    stats = compute_milestone_stats()
    # 11 experiments: exp3459-exp3469 (capstone exp3469 included)
    assert stats["experiments_completed"] == 11
    assert stats["flagged_adversarial_count"] == 2
    assert stats["flagged_ids"] == ["exp3460", "exp3462"]
    assert stats["compute_bound_experiments_count"] == 1
    assert stats["compute_bound_ids"] == ["exp3459"]
    assert stats["slowest_experiment_id"] == "exp3459"
    assert stats["total_wall_time_minutes"] > 0
