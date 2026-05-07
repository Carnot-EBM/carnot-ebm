"""Tests for the Exp 1461 comparator cite/retire audit.

Spec: REQ-REPORT-045, SCENARIO-REPORT-045.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.comparator_cite_retire_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_COMPARATORS,
    STATUS_CLARIFICATION_SENTINEL,
    apply_references_status_clarification,
    build_artifact,
    default_decisions,
    render_decision_table,
    run,
    write_in_progress_artifact,
)


def test_req_report_045_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-045: seed the deliverable before terminal audit work."""

    out_path = tmp_path / "results" / "experiment_1461_comparator_integration_cite_retire_audit.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["comparator_decision_count"] == 0
    assert artifact["cite_count"] == 0
    assert artifact["retire_count"] == 0
    assert artifact["watchlist_count"] == 0
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_report_045_counts_decisions_and_retirement_reopen_conditions() -> None:
    """SCENARIO-REPORT-045: every required comparator receives one decision."""

    decisions = default_decisions()
    artifact = build_artifact(
        decisions=decisions,
        decision_table_path="docs/research-notes/comparator_cite_retire_audit.md",
        references_updated=True,
    )
    table = render_decision_table(decisions)

    assert [decision.comparator for decision in decisions] == REQUIRED_COMPARATORS
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["comparator_decision_count"] == 10
    assert artifact["cite_count"] == 6
    assert artifact["retire_count"] == 1
    assert artifact["watchlist_count"] == 3
    assert artifact["decision_table_path"] == "docs/research-notes/comparator_cite_retire_audit.md"
    assert artifact["references_updated"] is True
    assert "paper-v6 cites only the six comparator rows" in artifact[
        "paper_related_work_implications"
    ]
    assert artifact["honest_verdict"] == (
        "comparator_scope_narrowed_6_cite_1_retire_3_watchlist"
    )

    retired = [decision for decision in decisions if decision.decision == "retire"]
    assert [decision.comparator for decision in retired] == ["GStack"]
    assert retired[0].future_reopen_condition.startswith("Reopen only if")
    assert "| Abstract-CoT | cite |" in table
    assert "| LARQL | future_watchlist |" in table
    assert "| GStack | retire |" in table
    assert "Related Work / discrete-latent reasoning" in table


def test_req_report_045_references_clarification_is_idempotent() -> None:
    """REQ-REPORT-045: references update is narrow and repeatable."""

    references = "# Research References & Future Considerations\n\nExisting comparator notes.\n"
    first, first_updated = apply_references_status_clarification(
        references,
        default_decisions(),
    )
    second, second_updated = apply_references_status_clarification(
        first,
        default_decisions(),
    )

    assert first_updated is True
    assert second_updated is False
    assert second == first
    assert first.count(STATUS_CLARIFICATION_SENTINEL) == 1
    assert "GStack: retired from active scope" in first
    assert "No unrelated references were added" in first


def test_req_report_045_run_writes_table_references_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-045: run writes the markdown table, refs note, and JSON."""

    out_path = tmp_path / "results" / "experiment_1461_comparator_integration_cite_retire_audit.json"
    table_path = tmp_path / "docs" / "research-notes" / "comparator_cite_retire_audit.md"
    refs_path = tmp_path / "research-references.md"
    refs_path.write_text("# Research References & Future Considerations\n", encoding="utf-8")

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        decision_table_path=table_path,
        references_path=refs_path,
    )

    written_artifact = json.loads(out_path.read_text(encoding="utf-8"))
    table = table_path.read_text(encoding="utf-8")
    references = refs_path.read_text(encoding="utf-8")

    assert artifact == written_artifact
    assert written_artifact["status"] == "complete"
    assert written_artifact["comparator_decision_count"] == 10
    assert written_artifact["cite_count"] == 6
    assert written_artifact["retire_count"] == 1
    assert written_artifact["watchlist_count"] == 3
    assert written_artifact["decision_table_path"] == (
        "docs/research-notes/comparator_cite_retire_audit.md"
    )
    assert written_artifact["references_updated"] is True
    assert "Abstract-CoT" in table
    assert "ontology-constrained reasoning" in table
    assert STATUS_CLARIFICATION_SENTINEL in references
