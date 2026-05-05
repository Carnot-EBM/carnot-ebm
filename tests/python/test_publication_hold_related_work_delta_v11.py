"""Tests for the Exp 1321 publication-hold related-work delta artifact.

Spec traces: REQ-PUBLISH-016, SCENARIO-PUBLISH-017.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.publication_hold_related_work_delta_v11 import (
    REQUIRED_FIELDS,
    build_related_work_delta,
    extract_20260505_reference_entries,
    run,
    validate_artifact,
    write_in_progress_artifact,
)


REFERENCES_TEXT = """# Research References

## 2026-05-05 Post-.101 Planning Sweep (Milestone 2026.04.102)

### ConstraintBench: Direct Constrained Optimization Feasibility
- **Paper:** arXiv 2602.22465, "ConstraintBench: Benchmarking LLM Constraint
  Reasoning on Direct Optimization."
- **Source:** https://arxiv.org/abs/2602.22465
- **What:** Feasibility is the dominant bottleneck.
- **Relevance to Carnot:** Better certificate stress tests.

### SATQuest: PySAT-Backed Logical Reasoning Verifier
- **Paper:** arXiv 2509.00930 / OpenReview ICLR 2026 Workshop LLM Reasoning.
- **Source:** https://arxiv.org/abs/2509.00930
- **What:** SAT-derived reasoning tasks with PySAT verification.
- **Relevance to Carnot:** Open verifier benchmark.

## 2026-05-05 Planning Sweep (Milestone 2026.04.101)

### QueryBandits for Online Hallucination Mitigation
- **Paper:** arXiv 2602.20332, "No One Size Fits All."
- **Source:** https://huggingface.co/papers/2602.20332
- **What:** Contextual bandits choose query rewrites online.
- **Relevance to Carnot:** Static rewrite policies are unsafe.

### Undated Internal Note
- **Paper:** internal note without a publication year.
- **Source:** local
- **What:** Should not be counted as a new material reference.

## 2026-05-04 Earlier Sweep

### Older Entry
- **Paper:** arXiv 2601.00001.
"""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_hold_project(project: Path, *, references_text: str = REFERENCES_TEXT) -> None:
    _write_json(
        project / "results" / "experiment_1307_arxiv_v10_hold_receipt_v2.json",
        {
            "status": "complete",
            "publication_state": "operator_hold",
            "operator_hold_active": True,
            "credentialed_submission_attempted": False,
            "honest_verdict": "operator_hold_active_no_local_arxiv_receipt",
        },
    )
    known_issues = project / "ops" / "known-issues.md"
    known_issues.parent.mkdir(parents=True)
    known_issues.write_text(
        "## PUBLICATION HOLD\n\narXiv submission is ON HOLD until operator review.\n",
        encoding="utf-8",
    )
    (project / "research-references.md").write_text(references_text, encoding="utf-8")


def test_write_in_progress_artifact_has_required_fields_req_publish_016(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-016: the runner writes a durable in-progress skeleton first."""

    out_path = tmp_path / "results" / "experiment_1321_publication_hold_related_work_delta_v11.json"

    artifact = write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "in_progress"
    assert set(REQUIRED_FIELDS).issubset(artifact)
    assert artifact["credentialed_submission_attempted"] is False


def test_extract_20260505_reference_entries_req_publish_016() -> None:
    """REQ-PUBLISH-016: only 2026-05-05 material 2025--2026 entries are counted."""

    entries = extract_20260505_reference_entries(REFERENCES_TEXT)

    assert [entry["title"] for entry in entries] == [
        "ConstraintBench: Direct Constrained Optimization Feasibility",
        "SATQuest: PySAT-Backed Logical Reasoning Verifier",
        "QueryBandits for Online Hallucination Mitigation",
    ]
    assert all(entry["source"] for entry in entries)


def test_related_work_delta_summarizes_reference_clusters_req_publish_016() -> None:
    """REQ-PUBLISH-016: the written delta is compact but names the new references."""

    delta = build_related_work_delta(extract_20260505_reference_entries(REFERENCES_TEXT))

    assert "2026-05-05 Related-Work Delta (Exp 1321)" in delta
    assert "ConstraintBench" in delta
    assert "SATQuest" in delta
    assert "QueryBandits" in delta
    assert "operator hold" in delta


def test_related_work_delta_covers_hardware_and_other_clusters_req_publish_016() -> None:
    """REQ-PUBLISH-016: compact synthesis covers hardware and uncategorized updates."""

    delta = build_related_work_delta(
        [
            {"title": "KAN Hardware and Analog Paths: lmKAN, RM/BOP/NABS, and aKAN"},
            {"title": "Unclassified 2026 Publication Context"},
        ]
    )

    assert "hardware-portable energy and KAN context" in delta
    assert "other publication-context updates" in delta


def test_run_writes_notes_delta_and_preserves_operator_hold_scenario_publish_017(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-017: notes update does not attempt submission or lift hold."""

    project = tmp_path
    notes = project / "docs" / "research-notes" / "literature-priority-audit.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("# Existing Literature Audit\n", encoding="utf-8")
    _write_hold_project(project)

    artifact = run(project_root=project, related_work_notes_path=notes)

    written = json.loads(
        (
            project / "results" / "experiment_1321_publication_hold_related_work_delta_v11.json"
        ).read_text(encoding="utf-8")
    )
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["publication_state"] == "operator_hold"
    assert artifact["operator_hold_active"] is True
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["related_work_delta_written"] is True
    assert artifact["new_references_count"] == 3
    assert (
        artifact["honest_verdict"]
        == "operator_hold_active_related_work_delta_written_no_submission"
    )
    assert (
        artifact["related_work_delta_target"] == "docs/research-notes/literature-priority-audit.md"
    )
    assert "QueryBandits" in notes.read_text(encoding="utf-8")


def test_missing_notes_keeps_delta_in_artifact_req_publish_016(tmp_path: Path) -> None:
    """REQ-PUBLISH-016: absent notes files use an artifact-only related-work delta."""

    project = tmp_path
    _write_hold_project(project)

    artifact = run(project_root=project, related_work_notes_path=project / "docs" / "missing.md")

    assert artifact["related_work_delta_written"] is True
    assert artifact["related_work_delta_target"] == "artifact"
    assert "ConstraintBench" in artifact["related_work_delta"]


def test_submitted_exp1307_state_is_preserved_req_publish_016(tmp_path: Path) -> None:
    """REQ-PUBLISH-016: a prior submitted state stays submitted without new submission."""

    project = tmp_path
    _write_json(
        project / "results" / "experiment_1307_arxiv_v10_hold_receipt_v2.json",
        {"publication_state": "submitted", "operator_hold_active": False},
    )
    known_issues = project / "ops" / "known-issues.md"
    known_issues.parent.mkdir(parents=True)
    known_issues.write_text("No active publication hold.\n", encoding="utf-8")
    (project / "research-references.md").write_text(REFERENCES_TEXT, encoding="utf-8")

    artifact = run(project_root=project, related_work_notes_path=project / "docs" / "missing.md")

    assert artifact["publication_state"] == "submitted"
    assert artifact["operator_hold_active"] is False
    assert artifact["credentialed_submission_attempted"] is False


def test_missing_exp1307_without_hold_records_blocked_req_publish_016(tmp_path: Path) -> None:
    """REQ-PUBLISH-016: absent prior hold evidence cannot invent a submission."""

    project = tmp_path
    known_issues = project / "ops" / "known-issues.md"
    known_issues.parent.mkdir(parents=True)
    known_issues.write_text("No active publication hold.\n", encoding="utf-8")
    (project / "research-references.md").write_text(REFERENCES_TEXT, encoding="utf-8")

    artifact = run(project_root=project, related_work_notes_path=project / "docs" / "missing.md")

    assert artifact["publication_state"] == "blocked"
    assert artifact["operator_hold_active"] is False
    assert (
        artifact["source_artifacts"][0] == "results/experiment_1307_arxiv_v10_hold_receipt_v2.json"
    )


def test_validate_artifact_rejects_submission_attempt_req_publish_016() -> None:
    """REQ-PUBLISH-016: credentialed submission attempts are forbidden."""

    artifact = {
        "status": "complete",
        "publication_state": "operator_hold",
        "operator_hold_active": True,
        "credentialed_submission_attempted": True,
        "related_work_delta_written": True,
        "new_references_count": 1,
        "honest_verdict": "bad",
    }

    with pytest.raises(AssertionError, match="credentialed_submission_attempted"):
        validate_artifact(artifact)
