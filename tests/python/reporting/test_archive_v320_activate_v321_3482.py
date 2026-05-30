"""Tests for carnot.reporting.archive_v320_activate_v321_3482.

REQ-REPORT-3482: Archive milestone .320, activate .321.

SCENARIO-REPORT-3482-01: write_artifact produces a JSON file with all required
  schema fields including archive_v320_activate_v321_ready=True.
SCENARIO-REPORT-3482-02: honest_verdict starts with the required terminal prefix.
SCENARIO-REPORT-3482-03: inference_substrate is aggregation_from_upstream_artifacts.
SCENARIO-REPORT-3482-04: .320 key finding (P0.1 blocked — benchmark difficulty
  mismatch; SC=0.265 below headroom band floor) recorded.
SCENARIO-REPORT-3482-05: G-gate state is G1/G3/G4=True, G2=False.
SCENARIO-REPORT-3482-06: depth_forcing_function remains active (not relaxed).
SCENARIO-REPORT-3482-07: retro JSON is written with correct schema.
SCENARIO-REPORT-3482-08: changelog entry is appended (not replacing content).
SCENARIO-REPORT-3482-09: compute_milestone_stats returns correct aggregates.
SCENARIO-REPORT-3482-10: FR-11 depth collapse finding (N=200, onset=138) recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.archive_v320_activate_v321_3482 import (
    compute_milestone_stats,
    write_artifact,
)


# ── fixture ────────────────────────────────────────────────────────────────


@pytest.fixture()
def artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run write_artifact in a tmp_path that mirrors ops/ and results/."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    changelog = ops_dir / "changelog.md"
    changelog.write_text("# Carnot — Changelog\n\nPrior content.\n", encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    monkeypatch.chdir(tmp_path)
    out = write_artifact(repo_root=tmp_path)
    return json.loads(out.read_text())


@pytest.fixture()
def retro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run write_artifact and return the retro JSON."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    (ops_dir / "changelog.md").write_text("# Changelog\n", encoding="utf-8")
    (tmp_path / "results").mkdir()

    monkeypatch.chdir(tmp_path)
    write_artifact(repo_root=tmp_path)
    retro_path = tmp_path / "results" / "operational_retro_2026_05_320.json"
    return json.loads(retro_path.read_text())


# ── SCENARIO-REPORT-3482-01: required fields ───────────────────────────────


def test_required_fields_present(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-01: all required fields must exist."""
    required = {
        "schema",
        "experiment_id",
        "inference_substrate",
        "honest_verdict",
        "archived_milestone",
        "activated_milestone",
        "archive_v320_activate_v321_ready",
        "g1", "g2", "g3", "g4",
        "unmet_gates",
        "paper_ready",
        "next_top_gap",
        "retro_path",
    }
    missing = required - set(artifact)
    assert not missing, f"Missing required fields: {missing}"


def test_archive_ready_flag(artifact: dict) -> None:
    """REQ-REPORT-3482: archive_v320_activate_v321_ready must be True."""
    assert artifact["archive_v320_activate_v321_ready"] is True


def test_archived_milestone(artifact: dict) -> None:
    """Archived milestone must be .320."""
    assert artifact["archived_milestone"] == "2026.05.320"


def test_activated_milestone(artifact: dict) -> None:
    """Activated milestone must be .321."""
    assert artifact["activated_milestone"] == "2026.05.321"


# ── SCENARIO-REPORT-3482-02: verdict prefix ────────────────────────────────


def test_verdict_terminal_prefix(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-02: verdict must start with terminal prefix."""
    verdict = artifact["honest_verdict"]
    prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "passed_",
                "shipped:", "shipped_")
    assert any(verdict.startswith(p) for p in prefixes), (
        f"honest_verdict does not start with a terminal prefix: {verdict!r}"
    )


# ── SCENARIO-REPORT-3482-03: inference substrate ──────────────────────────


def test_inference_substrate(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-03: substrate must be aggregation_from_upstream_artifacts."""
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


# ── SCENARIO-REPORT-3482-04: key P0.1 finding ────────────────────────────


def test_p01_not_answered(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-04: P0.1 remains open (not answered)."""
    assert artifact["p01_hypothesis_answered"] is False


def test_corpus_sc_below_band(artifact: dict) -> None:
    """SCENARIO-REPORT-3482-04: SC accuracy recorded below headroom band floor."""
    assert artifact["corpus_sc_accuracy_320"] == pytest.approx(0.2647)
    assert artifact["corpus_sc_in_headroom_band"] is False


def test_p01_v6_blocked_recorded(artifact: dict) -> None:
    """SCENARIO-REPORT-3482-04: P0.1 v6 blocked verdict is documented."""
    assert "blocked" in artifact["p01_v6_finding"].lower()


# ── SCENARIO-REPORT-3482-05: G-gate state ────────────────────────────────


def test_g_gates(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-05: G1/G3/G4=True, G2=False."""
    assert artifact["g1"] is True
    assert artifact["g2"] is False
    assert artifact["g3"] is True
    assert artifact["g4"] is True
    assert "G2" in artifact["unmet_gates"]
    assert artifact["paper_ready"] is False


# ── SCENARIO-REPORT-3482-06: depth forcing function ───────────────────────


def test_depth_forcing_function_active(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-06: depth forcing function must not relax."""
    assert artifact["depth_forcing_function_active"] is True
    assert artifact["depth_forcing_function_can_relax"] is False


# ── SCENARIO-REPORT-3482-07: retro JSON written ───────────────────────────


def test_retro_schema(retro: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-07: retro must use v65 schema."""
    assert retro["schema"] == "carnot.operational_retro.v65"
    assert retro["milestone"] == "2026.05.320"


def test_retro_key_finding(retro: dict) -> None:
    """SCENARIO-REPORT-3482-07: retro captures P0.1 block and FR-11 positive."""
    assert "p01_status" in retro
    assert "OPEN" in retro["p01_status"]
    assert retro["fr11_depth_collapse_finding"] != ""


def test_retro_g2_status(retro: dict) -> None:
    """SCENARIO-REPORT-3482-07: retro records G2 package as verified but external run pending."""
    assert retro["g2_status"] == "self_contained_package_verified_external_run_pending"
    assert retro["g2_independent_reproducer"] is False


# ── SCENARIO-REPORT-3482-08: changelog appended ───────────────────────────


def test_changelog_appended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-08: changelog entry appended without deleting prior content."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    prior = "# Carnot — Changelog\n\nPrior content that must not be deleted.\n"
    changelog = ops_dir / "changelog.md"
    changelog.write_text(prior, encoding="utf-8")
    (tmp_path / "results").mkdir()

    monkeypatch.chdir(tmp_path)
    write_artifact(repo_root=tmp_path)

    new_content = changelog.read_text(encoding="utf-8")
    assert "Prior content that must not be deleted." in new_content, (
        "Changelog append must not delete prior content"
    )
    assert "2026.05.320" in new_content
    assert "archive_v320_activate_v321_ready=true" in new_content
    assert ".321 is active" in new_content


# ── SCENARIO-REPORT-3482-09: compute_milestone_stats ──────────────────────


def test_compute_milestone_stats_keys() -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-09: stats dict has expected keys."""
    stats = compute_milestone_stats()
    required_keys = {
        "completed_ids", "flagged_ids", "blocked_ids",
        "total_wall_s", "live_llm_wall_s",
        "n_completed", "n_flagged", "n_blocked",
    }
    assert required_keys <= set(stats)


def test_compute_milestone_stats_flagged() -> None:
    """SCENARIO-REPORT-3482-09: exp3473 (TAUTOLOGY) is the only flagged experiment."""
    stats = compute_milestone_stats()
    assert "exp3473" in stats["flagged_ids"]
    assert stats["n_flagged"] == 1


def test_compute_milestone_stats_live_llm_wall() -> None:
    """SCENARIO-REPORT-3482-09: live LLM wall time comes from exp3471 only."""
    stats = compute_milestone_stats()
    # exp3471 is the only live_llm_inference experiment with duration_s=1358.769
    assert stats["live_llm_wall_s"] == pytest.approx(1358.769)


def test_compute_milestone_stats_all_experiments() -> None:
    """SCENARIO-REPORT-3482-09: all 11 milestone experiments are included."""
    stats = compute_milestone_stats()
    assert stats["n_completed"] == 11


# ── SCENARIO-REPORT-3482-10: FR-11 depth collapse finding ────────────────


def test_fr11_collapse_finding(artifact: dict) -> None:
    """REQ-REPORT-3482 SCENARIO-REPORT-3482-10: FR-11 collapse finding recorded."""
    assert artifact["fr11_depth_collapse_confirmed_n200"] is True
    assert artifact["fr11_collapse_onset_iteration"] == 138
    assert artifact["fr11_arm_b_entropy_beta"] == pytest.approx(0.5)
    assert "entropy_beta" in artifact["fr11_phase5_mandatory_action"].lower()
