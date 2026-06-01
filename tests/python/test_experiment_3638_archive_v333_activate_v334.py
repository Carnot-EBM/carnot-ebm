"""Tests for Exp 3638 milestone .333 archive and .334 activation."""

import json
from pathlib import Path

from carnot.reporting import archive_v333_activate_v334_3638 as mod


TERMINAL_VERDICT = (
    "complete: "
    "archived_v333_gemini_quota_total_wipeout_zero_artifacts_"
    "cross_domain_question_still_open_v334_active_paper_ready_true"
)


def _write_fixture(root: Path, complete_text: str) -> None:
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "scripts").mkdir()
    (root / "research-roadmap.yaml").write_text(
        'milestone: "2026.06.334"\n'
        'milestone_title: "Run the .333 science on a working backend"\n'
        "tasks:\n"
        "  - id: exp3638-archive-v333-activate-v334\n",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        "| 2026-06-01 06:09 UTC | Milestone 2026.06.333 activated | OK | 14 tasks queued |\n"
        "| 2026-06-01 06:12 UTC | Archive milestone .332 honestly | FAIL | "
        "Gemini CLI error: .js:345500:14) |\n"
        "| 2026-06-01 08:06 UTC | Milestone 2026.06.334 activated | OK | 14 tasks queued |\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# Carnot North Star\n\nG1-G4 gate frame is the publication invariant.\n",
        encoding="utf-8",
    )
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor must remain untouched\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")


def test_run_writes_artifact_and_corrects_existing_archive(tmp_path: Path) -> None:
    """REQ-REPORT-3638 / SCENARIO-REPORT-3638: .333 is archived as infra wipeout."""

    _write_fixture(
        tmp_path,
        "- id: 2026.06.332\n"
        "  finding: prior archive may be stale\n"
        "- id: 2026.06.333\n"
        "  finding: incorrect leftover success\n"
        "  tasks:\n"
        "  - id: exp3624-archive-v332-activate-v333\n"
        "    result: OK (conductor)\n",
    )

    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    out_path = mod.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["v333_outcome_recorded_as"] == (
        "total_infrastructure_wipeout_zero_artifacts_science_never_ran"
    )
    assert artifact["gemini_quota_crash_cascade_recorded"] == (
        "gemini quota 429 + gemini-cli .js:345500:14 crash + "
        "GEMINI_FORCE_EXPERIMENTS coercion"
    )
    assert artifact["cross_domain_question_still_open"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["n_tasks_archived"] == 14
    assert artifact["random_seed"] == 3638
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] >= 0.0001
    assert artifact["zero_artifacts_found"] is True
    assert artifact["exp3624_archive_task_never_landed"] is True
    assert artifact["v332_full_archive_may_be_leftover"] is True
    assert artifact["activated_milestone"] == "2026.06.334"

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert "TOTAL INFRASTRUCTURE WIPEOUT" in complete
    assert "result: FAIL (gemini quota crash; no artifact landed)" in complete
    assert "result: OK (conductor)" not in complete
    assert complete.count("- id: 2026.06.333") == 1
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_research_complete_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3638: missing .333 archive is appended once and stays stable."""

    _write_fixture(
        tmp_path,
        "- id: 2026.06.332\n"
        "  finding: prior archive may be stale because exp3624 never landed\n",
    )

    first_path = mod.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = mod.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.333") == 1
    assert first_artifact == second_artifact


def test_main_uses_current_directory_and_unknown_milestone_is_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-REPORT-3638: CLI entrypoint writes the same archive artifact."""

    _write_fixture(
        tmp_path,
        "- id: 2026.06.332\n"
        "  finding: prior archive may be stale because exp3624 never landed\n",
    )

    assert mod._read_active_milestone(tmp_path) == "2026.06.334"
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")
    assert mod._read_active_milestone(tmp_path) == "unknown"
    (tmp_path / "research-roadmap.yaml").write_text(
        'milestone: "2026.06.334"\n',
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    mod.main()

    artifact = json.loads(
        (tmp_path / "results" / "experiment_3638_archive_v333_activate_v334.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
