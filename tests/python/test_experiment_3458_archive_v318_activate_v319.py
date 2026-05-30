"""Tests for the .318 archive / .319 activation module (exp3458).

Spec: REQ-REPORT-3458, SCENARIO-REPORT-3458-VERDICTS,
SCENARIO-REPORT-3458-STATS, SCENARIO-REPORT-3458-RETRO,
SCENARIO-REPORT-3458-CHANGELOG.

These tests verify the pure-computation functions in the module without
touching the filesystem (the write_artifact side-effecting function is
exercised by a single integration test against tmp_path).  Every assertion
is derivable from the published .318 artifact files — no live model needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.archive_v318_activate_v319_3458 import (
    append_changelog_entry,
    build_deliverable_payload,
    build_retro_payload,
    compute_milestone_stats,
    write_artifact,
)


# ---------------------------------------------------------------------------
# compute_milestone_stats — REQ-REPORT-3458
# ---------------------------------------------------------------------------


def test_stats_experiment_count() -> None:
    """SCENARIO-REPORT-3458-STATS: six experiments completed in .318."""
    stats = compute_milestone_stats()
    assert stats["experiments_completed"] == 6


def test_stats_compute_bound_count() -> None:
    """SCENARIO-REPORT-3458-STATS: exactly one live-LLM experiment (exp3448)."""
    stats = compute_milestone_stats()
    assert stats["compute_bound_experiments_count"] == 1
    assert "exp3448" in stats["compute_bound_ids"]


def test_stats_flagged_adversarial_count() -> None:
    """SCENARIO-REPORT-3458-STATS: two experiments flagged (exp3449, exp3452)."""
    stats = compute_milestone_stats()
    assert stats["flagged_adversarial_count"] == 2
    assert "exp3449" in stats["flagged_ids"]
    assert "exp3452" in stats["flagged_ids"]


def test_stats_total_wall_time_plausible() -> None:
    """SCENARIO-REPORT-3458-STATS: total wall time > 17 minutes (exp3448 alone = 17 min)."""
    stats = compute_milestone_stats()
    # exp3448 took 1041.724 s = 17.36 min; other exps add ~1.5 min more
    assert stats["total_wall_time_minutes"] > 17


def test_stats_slowest_is_exp3448() -> None:
    """SCENARIO-REPORT-3458-STATS: the live-LLM corpus builder is the slowest experiment."""
    stats = compute_milestone_stats()
    assert stats["slowest_experiment_id"] == "exp3448"
    assert stats["slowest_experiment_duration_s"] > 1000


# ---------------------------------------------------------------------------
# build_retro_payload — REQ-REPORT-3458
# ---------------------------------------------------------------------------


def test_retro_schema_is_v65() -> None:
    """SCENARIO-REPORT-3458-RETRO: retro upgrades from v64 to v65."""
    stats = compute_milestone_stats()
    retro = build_retro_payload(stats)
    assert retro["schema"] == "carnot.operational_retro.v65"


def test_retro_honest_verdict_terminal_prefix() -> None:
    """SCENARIO-REPORT-3458-VERDICTS: retro honest_verdict must start with 'complete:'."""
    stats = compute_milestone_stats()
    retro = build_retro_payload(stats)
    assert retro["honest_verdict"].startswith("complete:")


def test_retro_energy_auroc_matches_exp3450() -> None:
    """SCENARIO-REPORT-3458-RETRO: energy_correctness_auroc is the exp3450 value (0.516~)."""
    stats = compute_milestone_stats()
    retro = build_retro_payload(stats)
    # exp3450 measured 0.5160115448048378 — within float tolerance
    assert abs(retro["energy_correctness_auroc_318"] - 0.5160) < 0.001


def test_retro_g2_remains_unmet() -> None:
    """SCENARIO-REPORT-3458-RETRO: G2 is the sole unmet publication gate."""
    stats = compute_milestone_stats()
    retro = build_retro_payload(stats)
    assert retro["g2"] is False
    assert retro["g1"] is True
    assert retro["g3"] is True
    assert retro["g4"] is True
    assert retro["unmet_gates"] == ["G2"]
    assert retro["paper_ready"] is False


def test_retro_flagged_experiments_have_mechanistic_explanations() -> None:
    """SCENARIO-REPORT-3458-RETRO: flagged entries document mechanistic cause (not just 'bug')."""
    stats = compute_milestone_stats()
    retro = build_retro_payload(stats)
    flagged = retro["flagged_adversarial_experiments"]
    assert len(flagged) == 2
    for entry in flagged:
        # Every flagged experiment must have a mechanistic explanation and a valid claim.
        assert "mechanistic_explanation" in entry
        assert len(entry["mechanistic_explanation"]) > 40
        assert "valid_directional_claim" in entry
        assert entry["is_real_finding"] is True


def test_retro_key_finding_mentions_auroc_and_trained_reranker() -> None:
    """SCENARIO-REPORT-3458-RETRO: key finding names the AUROC and the forward action."""
    stats = compute_milestone_stats()
    retro = build_retro_payload(stats)
    kf = retro["key_finding_p01"]
    assert "0.516" in kf
    assert "trained" in kf.lower() or "EORM" in kf


# ---------------------------------------------------------------------------
# build_deliverable_payload — REQ-REPORT-3458
# ---------------------------------------------------------------------------


def test_deliverable_honest_verdict_terminal_prefix() -> None:
    """SCENARIO-REPORT-3458-VERDICTS: deliverable verdict must start with 'complete:'."""
    stats = compute_milestone_stats()
    payload = build_deliverable_payload(stats)
    assert payload["honest_verdict"].startswith("complete:")


def test_deliverable_archive_flag_is_true() -> None:
    """SCENARIO-REPORT-3458-VERDICTS: archive_v318_activate_v319_ready must be True."""
    stats = compute_milestone_stats()
    payload = build_deliverable_payload(stats)
    assert payload["archive_v318_activate_v319_ready"] is True


def test_deliverable_p01_hypothesis_not_answered() -> None:
    """SCENARIO-REPORT-3458-VERDICTS: p01_hypothesis_answered is False — trained reranker pending."""
    stats = compute_milestone_stats()
    payload = build_deliverable_payload(stats)
    assert payload["p01_hypothesis_answered"] is False


def test_deliverable_inference_substrate_is_aggregation() -> None:
    """SCENARIO-REPORT-3458-RETRO: deliverable uses aggregation substrate (no live model)."""
    stats = compute_milestone_stats()
    payload = build_deliverable_payload(stats)
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_deliverable_is_json_serializable() -> None:
    """SCENARIO-REPORT-3458-RETRO: the deliverable dict must serialize to valid JSON."""
    stats = compute_milestone_stats()
    payload = build_deliverable_payload(stats)
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)
    assert reloaded["archive_v318_activate_v319_ready"] is True


def test_deliverable_milestone_fields() -> None:
    """SCENARIO-REPORT-3458-RETRO: archived and activated milestones are correct."""
    stats = compute_milestone_stats()
    payload = build_deliverable_payload(stats)
    assert payload["archived_milestone"] == "2026.05.318"
    assert payload["activated_milestone"] == "2026.05.319"


# ---------------------------------------------------------------------------
# append_changelog_entry — SCENARIO-REPORT-3458-CHANGELOG
# ---------------------------------------------------------------------------


def test_changelog_append_does_not_remove_existing_content(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3458-CHANGELOG: existing changelog content is preserved."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    changelog = ops_dir / "changelog.md"
    original = "# Carnot — Changelog\n\n## 2026-05-30 (Old Entry)\n\n- old content\n"
    changelog.write_text(original, encoding="utf-8")

    append_changelog_entry(tmp_path)

    result = changelog.read_text(encoding="utf-8")
    assert "# Carnot — Changelog" in result
    assert "old content" in result


def test_changelog_append_inserts_318_entry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3458-CHANGELOG: the .318 archive entry is present after append."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    changelog = ops_dir / "changelog.md"
    changelog.write_text("# Carnot — Changelog\n", encoding="utf-8")

    append_changelog_entry(tmp_path)

    result = changelog.read_text(encoding="utf-8")
    assert "2026.05.318" in result
    assert "archive_v318_activate_v319_ready" in result


def test_changelog_header_line_survives_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3458-CHANGELOG: the first '# Carnot' header line stays at top."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    changelog = ops_dir / "changelog.md"
    changelog.write_text("# Carnot — Changelog\n\nexisting\n", encoding="utf-8")

    append_changelog_entry(tmp_path)

    result = changelog.read_text(encoding="utf-8")
    assert result.startswith("# Carnot — Changelog\n")


# ---------------------------------------------------------------------------
# write_artifact integration test — REQ-REPORT-3458
# ---------------------------------------------------------------------------


def test_write_artifact_produces_all_files(tmp_path: Path) -> None:
    """REQ-REPORT-3458: write_artifact writes the retro, changelog, and deliverable."""
    # Set up the directory structure write_artifact expects.
    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text(
        "# Carnot — Changelog\n", encoding="utf-8"
    )

    deliverable_path = write_artifact(repo_root=tmp_path)

    # Deliverable was returned and exists.
    assert deliverable_path.exists()
    assert deliverable_path.name == "experiment_3458_archive_v318_activate_v319.json"

    # Retro was written.
    retro_path = tmp_path / "results" / "operational_retro_2026_05_318.json"
    assert retro_path.exists()
    retro = json.loads(retro_path.read_text(encoding="utf-8"))
    assert retro["schema"] == "carnot.operational_retro.v65"

    # Changelog was updated.
    changelog_text = (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8")
    assert "2026.05.318" in changelog_text

    # Deliverable JSON is valid and has required fields.
    payload = json.loads(deliverable_path.read_text(encoding="utf-8"))
    assert payload["archive_v318_activate_v319_ready"] is True
    assert payload["honest_verdict"].startswith("complete:")


def test_write_artifact_default_repo_root_resolves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-3458: write_artifact(repo_root=None) falls back to path relative to the module file."""
    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text("# Carnot — Changelog\n", encoding="utf-8")

    import carnot.reporting.archive_v318_activate_v319_3458 as mod

    # Patch the module's Path so parents[3] resolves to tmp_path.
    fake_file = tmp_path / "python" / "carnot" / "reporting" / "archive_v318_activate_v319_3458.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.touch()

    monkeypatch.setattr(mod, "__file__", str(fake_file))

    deliverable_path = mod.write_artifact(repo_root=None)
    assert deliverable_path.exists()
    payload = json.loads(deliverable_path.read_text(encoding="utf-8"))
    assert payload["archive_v318_activate_v319_ready"] is True
