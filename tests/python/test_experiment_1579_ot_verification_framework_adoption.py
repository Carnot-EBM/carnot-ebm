"""Tests for Exp 1579 paper-v6 OT verification framework adoption.

Spec: REQ-PUBLISH-023, SCENARIO-PUBLISH-025.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_ot_verification_framework_adoption as exp1579


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json"
)
NOTE_PATH = (
    REPO_ROOT
    / "docs"
    / "research-notes"
    / "paper-v6-ot-verification-framework-adoption.md"
)
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "publication" / "spec.md"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "adoption_note_path",
    "ot_framework_adopted",
    "claim_conflict_count",
    "paper_patch_applied",
    "no_publication_trigger",
    "honest_verdict",
}


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _note_text() -> str:
    return NOTE_PATH.read_text(encoding="utf-8")


def test_req_publish_023_spec_anchor_exists() -> None:
    """REQ-PUBLISH-023, SCENARIO-PUBLISH-025: Exp 1579 is spec-anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PUBLISH-023" in spec
    assert "SCENARIO-PUBLISH-025" in spec
    assert "experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json" in spec
    assert "paper-v6-ot-verification-framework-adoption.md" in spec


def test_req_publish_023_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-PUBLISH-023: the workflow seeds the required artifact before analysis."""

    out_path = tmp_path / exp1579.DEFAULT_OUT_REL
    artifact = exp1579.write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert exp1579.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["adoption_note_path"] == ""
    assert artifact["ot_framework_adopted"] is False
    assert artifact["claim_conflict_count"] == 0
    assert artifact["paper_patch_applied"] is False
    assert artifact["no_publication_trigger"] is True


def test_scenario_publish_025_default_mappings_and_conflicts_are_complete() -> None:
    """SCENARIO-PUBLISH-025: mappings cover OT terms and conflict boundaries."""

    mappings = exp1579.default_ot_mappings()
    conflicts = exp1579.default_claim_conflicts()

    assert {"coverage", "ROC", "sub-optimality"} <= set(mappings)
    assert "generator proposal mass" in mappings["coverage"]["carnot_mapping"]
    assert "Youden" in mappings["ROC"]["paper_term"]
    assert "finite-K" in mappings["sub-optimality"]["carnot_mapping"]
    assert len(conflicts) >= 4
    assert all(row["claim_id"].startswith("CONFLICT-") for row in conflicts)
    assert all(row["softened_boundary"].startswith("Paper-v6 should") for row in conflicts)
    assert any("finite-K" in row["reason"] for row in conflicts)
    assert any("ROC" in row["reason"] for row in conflicts)


def test_req_publish_023_run_writes_note_and_complete_artifact(tmp_path: Path) -> None:
    """REQ-PUBLISH-023: run writes in-progress then terminal local artifacts."""

    out_path = tmp_path / exp1579.DEFAULT_OUT_REL
    note_path = tmp_path / exp1579.DEFAULT_NOTE_REL
    writes: list[str] = []

    artifact = exp1579.run(
        root=tmp_path,
        out_path=out_path,
        note_path=note_path,
        write_observer=lambda _path, payload: writes.append(str(payload["status"])),
    )
    note = note_path.read_text(encoding="utf-8")

    assert writes == ["in_progress", "complete"]
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["adoption_note_path"] == exp1579.DEFAULT_NOTE_REL.as_posix()
    assert artifact["ot_framework_adopted"] is True
    assert artifact["claim_conflict_count"] == len(exp1579.default_claim_conflicts())
    assert artifact["paper_patch_applied"] is False
    assert artifact["no_publication_trigger"] is True
    assert "## Patch Plan" in note
    assert "docs/papers/paper-v6/main.tex is absent" in note


def test_req_publish_023_validator_rejects_overclaiming_artifacts() -> None:
    """REQ-PUBLISH-023: validation rejects publication triggers and weak ledgers."""

    artifact = exp1579.build_artifact(
        adoption_note_path=exp1579.DEFAULT_NOTE_REL.as_posix(),
        claim_conflicts=exp1579.default_claim_conflicts(),
        paper_patch_applied=False,
    )
    exp1579.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1579.validate_artifact(missing)

    with pytest.raises(ValueError, match="no_publication_trigger"):
        exp1579.validate_artifact(dict(artifact, no_publication_trigger=False))

    with pytest.raises(ValueError, match="ot_framework_adopted"):
        exp1579.validate_artifact(dict(artifact, ot_framework_adopted=False))

    with pytest.raises(ValueError, match="claim_conflict_count"):
        exp1579.validate_artifact(dict(artifact, claim_conflict_count=1))

    with pytest.raises(ValueError, match="honest_verdict"):
        exp1579.validate_artifact(dict(artifact, honest_verdict="complete"))


def test_scenario_publish_025_checked_in_note_and_artifact_are_terminal() -> None:
    """SCENARIO-PUBLISH-025: checked-in deliverables adopt vocabulary safely."""

    artifact = _artifact()
    text = _note_text()

    exp1579.validate_artifact(artifact)
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["adoption_note_path"] == NOTE_PATH.relative_to(REPO_ROOT).as_posix()
    assert artifact["ot_framework_adopted"] is True
    assert artifact["claim_conflict_count"] >= 4
    assert artifact["paper_patch_applied"] is False
    assert artifact["no_publication_trigger"] is True
    assert "arXiv:2510.18982" in text
    assert "## Coverage Mapping" in text
    assert "## ROC Mapping" in text
    assert "## Sub-optimality Mapping" in text
    assert "## Conflict Ledger" in text
    assert text.count("### CONFLICT-") == artifact["claim_conflict_count"]
    assert "finite-K" in text
    assert "Youden" in text
    assert "does not trigger publication" in text
