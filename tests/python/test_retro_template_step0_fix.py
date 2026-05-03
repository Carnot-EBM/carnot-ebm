"""Tests for the Exp 1204 retro-template STEP 0 documentation fix.

Spec traces: REQ-REPORT-014, SCENARIO-REPORT-011.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import retro_template_step0_fix as exp1204


KNOWN_ISSUE_TEXT = """\
## MANDATORY-NEXT-MILESTONE PRIORITIES (.86 planner -- hard pickup per CLAUDE.md)

### NEW 2026-05-03 (13:55Z): Retro Task Boundary Too Tight (artifact_not_updated_past_bootstrap)

**Background:** Heavy retro work does not fit the configured max_turns budget.

**Mandatory .94 fix (one or more of):**

D. **Recommended: A + B combined.** Longer budget + explicit early-write instruction.
"""

ROADMAP_TEXT = """\
- id: exp1215-milestone-retro-94
  title: "Milestone 2026.04.94 Retrospective -- STEP 0 Pattern, claude/opus, max_turns:100"
  agent_type: claude
  model: opus
  max_turns: 100
  prompt: |
    STEP 0 -- WRITE SKELETON ARTIFACT IMMEDIATELY:
    Write skeleton artifact (STEP 0) to results/experiment_1215_milestone_retro_94.json
"""


def test_resolution_note_is_added_without_pruning_req_report_014() -> None:
    """REQ-REPORT-014: the resolution line is added while old text remains."""
    updated, note_added = exp1204.add_resolution_note(KNOWN_ISSUE_TEXT)

    assert note_added is True
    assert exp1204.RESOLUTION_NOTE in updated
    assert "**Background:** Heavy retro work" in updated
    assert "D. **Recommended: A + B combined.**" in updated
    assert updated.count(exp1204.RESOLUTION_NOTE) == 1


def test_resolution_note_is_idempotent_req_report_014() -> None:
    """REQ-REPORT-014: an existing resolution note is not duplicated."""
    already_resolved = KNOWN_ISSUE_TEXT.replace(
        "\n**Background:**",
        f"\n{exp1204.RESOLUTION_NOTE}\n\n**Background:**",
    )

    updated, note_added = exp1204.add_resolution_note(already_resolved)

    assert note_added is False
    assert updated == already_resolved
    assert updated.count(exp1204.RESOLUTION_NOTE) == 1


def test_run_updates_known_issues_and_writes_artifact_scenario_report_011(tmp_path: Path) -> None:
    """SCENARIO-REPORT-011: Exp 1204 writes the required successful artifact."""
    known_issues = tmp_path / "ops" / "known-issues.md"
    roadmap = tmp_path / "research-roadmap.yaml"
    out = tmp_path / "results" / "experiment_1204_retro_template_step0_fix.json"
    known_issues.parent.mkdir()
    known_issues.write_text(KNOWN_ISSUE_TEXT, encoding="utf-8")
    roadmap.write_text(ROADMAP_TEXT, encoding="utf-8")

    artifact = exp1204.run(known_issues_path=known_issues, roadmap_path=roadmap, out_path=out)

    written_known_issues = known_issues.read_text(encoding="utf-8")
    written_artifact = json.loads(out.read_text(encoding="utf-8"))
    assert exp1204.RESOLUTION_NOTE in written_known_issues
    assert artifact == written_artifact
    assert written_artifact["retro_boundary_issue_found"] is True
    assert written_artifact["resolution_note_added"] is True
    assert written_artifact["known_issues_file_updated"] is True
    assert written_artifact["retro_template_updated"] is True
    assert written_artifact["exp1215_step0_pattern_found"] is True
    assert written_artifact["exp1215_opus_100_found"] is True
    assert written_artifact["honest_verdict"] == "template_updated"


def test_run_reports_already_resolved_without_rewrite_req_report_014(tmp_path: Path) -> None:
    """REQ-REPORT-014: already-resolved entries produce an honest no-op artifact."""
    known_issues = tmp_path / "known-issues.md"
    roadmap = tmp_path / "research-roadmap.yaml"
    out = tmp_path / "artifact.json"
    known_issues.write_text(f"{KNOWN_ISSUE_TEXT}\n{exp1204.RESOLUTION_NOTE}\n", encoding="utf-8")
    roadmap.write_text(ROADMAP_TEXT, encoding="utf-8")

    artifact = exp1204.run(known_issues_path=known_issues, roadmap_path=roadmap, out_path=out)

    assert known_issues.read_text(encoding="utf-8").count(exp1204.RESOLUTION_NOTE) == 1
    assert artifact["resolution_note_added"] is False
    assert artifact["known_issues_file_updated"] is False
    assert artifact["retro_template_updated"] is False
    assert artifact["exp1215_step0_pattern_found"] is True
    assert artifact["honest_verdict"] == "already_resolved"


def test_run_blocks_when_issue_is_missing_scenario_report_011(tmp_path: Path) -> None:
    """SCENARIO-REPORT-011: missing known-issues evidence is reported as blocked."""
    out = tmp_path / "nested" / "artifact.json"

    artifact = exp1204.run(
        known_issues_path=tmp_path / "missing-known-issues.md",
        roadmap_path=tmp_path / "missing-roadmap.yaml",
        out_path=out,
    )

    assert out.exists()
    assert artifact["retro_boundary_issue_found"] is False
    assert artifact["resolution_note_added"] is False
    assert artifact["known_issues_file_updated"] is False
    assert artifact["retro_template_updated"] is False
    assert artifact["exp1215_step0_pattern_found"] is False
    assert artifact["exp1215_opus_100_found"] is False
    assert artifact["honest_verdict"] == "blocked"


def test_build_artifact_blocks_unresolved_noop_req_report_014() -> None:
    """REQ-REPORT-014: a no-op without an existing resolution remains blocked."""
    artifact = exp1204.build_artifact(
        KNOWN_ISSUE_TEXT,
        ROADMAP_TEXT,
        resolution_note_added=False,
        known_issues_file_updated=False,
    )

    assert artifact["retro_boundary_issue_found"] is True
    assert artifact["exp1215_step0_pattern_found"] is True
    assert artifact["honest_verdict"] == "blocked"


def test_main_returns_status_from_artifact_req_report_014(tmp_path: Path) -> None:
    """REQ-REPORT-014: the CLI returns success only for non-blocked artifacts."""
    known_issues = tmp_path / "known-issues.md"
    roadmap = tmp_path / "research-roadmap.yaml"
    out = tmp_path / "artifact.json"
    known_issues.write_text(KNOWN_ISSUE_TEXT, encoding="utf-8")
    roadmap.write_text(ROADMAP_TEXT, encoding="utf-8")

    ok_code = exp1204.main(
        [
            "--known-issues",
            str(known_issues),
            "--roadmap",
            str(roadmap),
            "--out",
            str(out),
        ]
    )
    blocked_code = exp1204.main(
        [
            "--known-issues",
            str(tmp_path / "missing.md"),
            "--roadmap",
            str(roadmap),
            "--out",
            str(tmp_path / "blocked.json"),
        ]
    )

    assert ok_code == 0
    assert blocked_code == 1
