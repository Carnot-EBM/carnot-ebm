"""Tests for the Exp 1455 known-issues mandatory priority audit.

Spec: REQ-REPORT-041, SCENARIO-REPORT-041.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.known_issues_mandatory_priority_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    ActivePriority,
    PriorityDecision,
    extract_active_priority_entries,
    run,
    write_in_progress_artifact,
)


KNOWN_ISSUES_FIXTURE = """# Carnot - Known Issues

**Last Updated:** 2026-04-30

## MANDATORY-NEXT-MILESTONE PRIORITIES (.86 planner - hard pickup per CLAUDE.md)

### NEW 2026-05-06 (16:30Z): SCOPE REDUCTION MILESTONE (.111 - preempts all other priorities)

Scope reduction body.

### NEW 2026-05-02 (18:50Z): Paper Integrity Audit - 18 Issues Block Publication

Paper integrity body.

### NEW 2026-05-03 (19:50Z): CRITICAL - Pre-Commit staged_files_only is Causing Silent Data Loss

Pre-commit body.

### NEW 2026-05-03 (20:35Z): NRGPT Frozen-Prefix Evaluation (optional, .95 or .96)

Optional body.

### REVISED 2026-04-30: Phase-2 Hardware Story Re-Scope (HIGH PRIORITY - paper-shaping)

Hardware body.

## MANDATORY-NEXT-MILESTONE PRIORITIES (.82 planner - historical)

### NEW 2026-04-29: historical priority

Historical body must remain untouched.
"""


def _fixture_decisions() -> dict[str, PriorityDecision]:
    return {
        "scope reduction milestone (.111 - preempts all other priorities)": PriorityDecision(
            status="keep",
            active_priority_id="scope",
            rationale="Scope reduction is the controlling .112 governance gate.",
        ),
        "paper integrity audit - 18 issues block publication": PriorityDecision(
            status="keep",
            active_priority_id="paper",
            rationale="Publication remains blocked until integrity issues close.",
        ),
        "critical - pre-commit staged_files_only is causing silent data loss": PriorityDecision(
            status="superseded",
            active_priority_id="artifact_hygiene",
            rationale="Superseded by the shipped fail-forward pre-commit fix.",
        ),
        "nrgpt frozen-prefix evaluation (optional, .95 or .96)": PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Explicitly optional and not part of the active scope-reduction lane.",
        ),
        "phase-2 hardware story re-scope (high priority - paper-shaping)": PriorityDecision(
            status="consolidate",
            active_priority_id="hardware",
            rationale="Folded into the smaller hardware portfolio narrowing decision.",
        ),
    }


def test_req_report_041_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-041: the workflow seeds the deliverable before auditing."""

    out_path = tmp_path / "results" / "experiment_1455_known_issues_mandatory_priority_audit.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["initial_priority_count"] == 0
    assert artifact["active_priority_count"] == 0
    assert artifact["trim_fraction"] == 0.0
    assert artifact["honest_verdict"] == "in_progress"


def test_req_report_041_extracts_only_current_active_priority_block() -> None:
    """REQ-REPORT-041: historical mandatory sections are not counted as active."""

    entries = extract_active_priority_entries(KNOWN_ISSUES_FIXTURE)

    assert len(entries) == 5
    assert entries[0].marker == "NEW 2026-05-06 (16:30Z)"
    assert entries[0].title == "SCOPE REDUCTION MILESTONE (.111 - preempts all other priorities)"
    assert entries[-1].marker == "REVISED 2026-04-30"
    assert all("historical priority" not in entry.title for entry in entries)


def test_scenario_report_041_writes_audit_index_and_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-041: trim to <=10 active items while preserving history."""

    known_issues_path = tmp_path / "ops" / "known-issues.md"
    audit_path = tmp_path / "ops" / "mandatory_priority_audit.md"
    index_path = tmp_path / "ops" / "active-priorities.md"
    out_path = tmp_path / "results" / "experiment_1455_known_issues_mandatory_priority_audit.json"
    known_issues_path.parent.mkdir(parents=True)
    known_issues_path.write_text(KNOWN_ISSUES_FIXTURE, encoding="utf-8")

    artifact = run(
        root=tmp_path,
        known_issues_path=known_issues_path,
        audit_path=audit_path,
        index_path=index_path,
        out_path=out_path,
        decisions=_fixture_decisions(),
        active_priorities=[
            ActivePriority(
                priority_id="scope",
                title="Scope control",
                source_entries=["SCOPE REDUCTION MILESTONE (.111 - preempts all other priorities)"],
                next_action="Finish .112 scope-reduction tasks before expanding lineages.",
            ),
            ActivePriority(
                priority_id="paper",
                title="Paper integrity",
                source_entries=["Paper Integrity Audit - 18 Issues Block Publication"],
                next_action="Keep publication hold until paper-critical issues are evidence-clean.",
            ),
        ],
    )
    written_artifact = json.loads(out_path.read_text(encoding="utf-8"))
    audit_text = audit_path.read_text(encoding="utf-8")
    index_text = index_path.read_text(encoding="utf-8")
    known_text = known_issues_path.read_text(encoding="utf-8")

    assert artifact == written_artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["initial_priority_count"] == 5
    assert artifact["active_priority_count"] == 2
    assert artifact["trim_fraction"] == 0.6
    assert artifact["priority_audit_path"] == "ops/mandatory_priority_audit.md"
    assert artifact["active_priorities_index_path"] == "ops/active-priorities.md"
    assert artifact["known_issues_updated"] is True
    assert len(artifact["retired_or_consolidated_priorities"]) == 2
    assert "superseded" in audit_text
    assert "parked" in audit_text
    assert "consolidate" in audit_text
    assert "Scope control" in index_text
    assert "Paper integrity" in index_text
    assert "historical priority" not in index_text
    assert "## CURRENT ACTIVE PRIORITIES (20260507 audit)" in known_text
    assert "Historical body must remain untouched." in known_text
