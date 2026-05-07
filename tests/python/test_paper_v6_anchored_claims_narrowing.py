"""Tests for Exp 1462 paper-v6 anchored-claims narrowing.

Spec: REQ-PUBLISH-021, SCENARIO-PUBLISH-023.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_anchored_claims_narrowing as exp1462


def _write_source_inputs(root: Path) -> None:
    for rel_path, text in exp1462.REQUIRED_SOURCE_INPUTS.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps({"status": "complete", "honest_verdict": rel_path.as_posix()}),
                encoding="utf-8",
            )
        else:
            path.write_text(text, encoding="utf-8")


def _paper_text() -> str:
    return (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Introduction}\n"
        "Introductory context.\n"
        "\\section{Related Work}\n"
        "Related context.\n"
        "\\appendix\n"
        "\\section{Existing Appendix}\n"
        "Existing appendix context.\n"
        "\\end{document}\n"
    )


def test_req_publish_021_spec_anchor_exists() -> None:
    """REQ-PUBLISH-021, SCENARIO-PUBLISH-023: Exp 1462 is spec-anchored."""

    spec = (
        exp1462.REPO_ROOT / "openspec" / "capabilities" / "publication" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-PUBLISH-021" in spec
    assert "SCENARIO-PUBLISH-023" in spec
    assert "experiment_1462_paper_v6_anchored_claims_narrowing.json" in spec


def test_req_publish_021_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-PUBLISH-021: the workflow seeds the required artifact before analysis."""

    out_path = tmp_path / "results" / exp1462.OUTPUT_FILENAME

    artifact = exp1462.write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert exp1462.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["paper_source_path"] is None
    assert artifact["anchored_claim_count"] == 0
    assert artifact["paper_updated"] is False
    assert artifact["arxiv_submission_triggered"] is False


def test_scenario_publish_023_default_claims_are_fully_anchored() -> None:
    """SCENARIO-PUBLISH-023: every retained claim has artifacts and theory."""

    claims = exp1462.default_anchored_claims()
    moved = exp1462.default_unanchored_claims_moved()

    assert 3 <= len(claims) <= 5
    assert len(moved) >= 3
    for claim in claims:
        assert claim["claim_id"].startswith("CLAIM-")
        assert claim["empirical_artifact_paths"]
        assert claim["theoretical_support"]
        assert claim["claim_boundary"].startswith("Does not claim")
    for unsupported in moved:
        assert unsupported["destination"] in {"appendix", "future_work"}
        assert unsupported["reason"]


def test_scenario_publish_023_updates_paper_idempotently(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-023: paper text gets one anchored section and future-work note."""

    claims = exp1462.default_anchored_claims()
    moved = exp1462.default_unanchored_claims_moved()

    updated, first_changed = exp1462.update_paper_text(_paper_text(), claims, moved)
    updated_again, second_changed = exp1462.update_paper_text(updated, claims, moved)

    assert first_changed is True
    assert second_changed is False
    assert updated_again == updated
    assert updated.count("\\section{Anchored Claims}") == 1
    assert updated.count("\\section{Unsupported Territory Moved to Appendix/Future Work}") == 1
    assert "CLAIM-1" in updated
    assert "Extropic" in updated
    assert updated.index("\\section{Anchored Claims}") < updated.index("\\section{Related Work}")
    assert updated.index("\\section{Unsupported Territory") > updated.index("\\appendix")


def test_req_publish_021_validator_rejects_unanchored_or_submission_artifacts() -> None:
    """REQ-PUBLISH-021: complete artifacts cannot hide weak claims or submission."""

    artifact = exp1462.build_artifact(
        paper_source_path="docs/arxiv-paper/main.tex",
        anchored_claims=exp1462.default_anchored_claims(),
        unanchored_claims_moved=exp1462.default_unanchored_claims_moved(),
        claim_matrix_path="docs/research-notes/paper_v6_anchored_claim_matrix.md",
        paper_updated=True,
        source_input_status={},
    )
    exp1462.validate_artifact(artifact)

    weak_claim = dict(artifact["anchored_claims"][0], empirical_artifact_paths=[])
    with pytest.raises(ValueError, match="empirical_artifact_paths"):
        exp1462.validate_artifact(
            dict(artifact, anchored_claims=[weak_claim, *artifact["anchored_claims"][1:]])
        )

    too_many = dict(
        artifact,
        anchored_claims=[
            *artifact["anchored_claims"],
            dict(artifact["anchored_claims"][0], claim_id="CLAIM-5"),
            dict(artifact["anchored_claims"][0], claim_id="CLAIM-6"),
        ],
    )
    with pytest.raises(ValueError, match="between 3 and 5"):
        exp1462.validate_artifact(too_many)

    with pytest.raises(ValueError, match="arxiv_submission_triggered"):
        exp1462.validate_artifact(dict(artifact, arxiv_submission_triggered=True))


def test_req_publish_021_validator_rejects_schema_and_boundary_drift() -> None:
    """REQ-PUBLISH-021: schema and claim-boundary drift fail loudly."""

    artifact = exp1462.build_artifact(
        paper_source_path="docs/arxiv-paper/main.tex",
        anchored_claims=exp1462.default_anchored_claims(),
        unanchored_claims_moved=exp1462.default_unanchored_claims_moved(),
        claim_matrix_path="docs/research-notes/paper_v6_anchored_claim_matrix.md",
        paper_updated=True,
        source_input_status={},
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1462.validate_artifact(missing)

    with pytest.raises(ValueError, match="status"):
        exp1462.validate_artifact(dict(artifact, status="in_progress"))

    with pytest.raises(ValueError, match="anchored_claim_count"):
        exp1462.validate_artifact(dict(artifact, anchored_claim_count=3))

    with pytest.raises(ValueError, match="unanchored_claims_moved"):
        exp1462.validate_artifact(dict(artifact, unanchored_claims_moved=[]))

    bad_destination = dict(artifact["unanchored_claims_moved"][0], destination="body")
    with pytest.raises(ValueError, match="unsupported destination"):
        exp1462.validate_artifact(
            dict(artifact, unanchored_claims_moved=[bad_destination])
        )

    missing_reason = dict(artifact["unanchored_claims_moved"][0], reason="")
    with pytest.raises(ValueError, match="missing reason"):
        exp1462.validate_artifact(dict(artifact, unanchored_claims_moved=[missing_reason]))

    with pytest.raises(ValueError, match="claim_matrix_path"):
        exp1462.validate_artifact(dict(artifact, claim_matrix_path=""))

    with pytest.raises(ValueError, match="paper_updated"):
        exp1462.validate_artifact(dict(artifact, paper_source_path=None, paper_updated=True))

    with pytest.raises(ValueError, match="honest_verdict"):
        exp1462.validate_artifact(dict(artifact, honest_verdict="complete"))

    missing_claim_field = dict(artifact["anchored_claims"][0])
    missing_claim_field.pop("title")
    with pytest.raises(ValueError, match="missing fields"):
        exp1462.validate_artifact(
            dict(
                artifact,
                anchored_claims=[missing_claim_field, *artifact["anchored_claims"][1:]],
            )
        )

    no_theory = dict(artifact["anchored_claims"][0], theoretical_support=[])
    with pytest.raises(ValueError, match="theoretical_support"):
        exp1462.validate_artifact(
            dict(artifact, anchored_claims=[no_theory, *artifact["anchored_claims"][1:]])
        )

    weak_boundary = dict(artifact["anchored_claims"][0], claim_boundary="Maybe later.")
    with pytest.raises(ValueError, match="negative boundary"):
        exp1462.validate_artifact(
            dict(
                artifact,
                anchored_claims=[weak_boundary, *artifact["anchored_claims"][1:]],
            )
        )


def test_req_publish_021_source_input_and_block_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-021: malformed inputs and block replacement remain deterministic."""

    malformed = (
        tmp_path
        / "results"
        / "experiment_1454_experiment_artifact_signal_noise_classifier.json"
    )
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")

    status = exp1462.inspect_source_inputs(tmp_path)
    assert status[malformed.relative_to(tmp_path).as_posix()]["parse_error"] == (
        "json_decode_error"
    )

    replaced = exp1462._replace_or_insert_block(
        f"{exp1462.ANCHOR_START}\nold\n{exp1462.ANCHOR_END}\nnext",
        start_marker=exp1462.ANCHOR_START,
        end_marker=exp1462.ANCHOR_END,
        block="new\n",
        fallback_marker="missing",
    )
    appended = exp1462._replace_or_insert_block(
        "plain text",
        start_marker="start",
        end_marker="end",
        block="block\n",
        fallback_marker="missing",
    )

    assert replaced == "new\nnext"
    assert appended == "plain text\n\nblock\n"


def test_scenario_publish_023_run_updates_paper_and_writes_matrix(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-023: run writes matrix, paper section, and complete JSON."""

    _write_source_inputs(tmp_path)
    paper_path = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(_paper_text(), encoding="utf-8")
    out_path = tmp_path / "results" / exp1462.OUTPUT_FILENAME
    matrix_path = tmp_path / exp1462.CLAIM_MATRIX_REL

    artifact = exp1462.run(
        root=tmp_path,
        out_path=out_path,
        claim_matrix_path=matrix_path,
        paper_candidates=[paper_path],
    )

    written_artifact = json.loads(out_path.read_text(encoding="utf-8"))
    paper = paper_path.read_text(encoding="utf-8")
    matrix = matrix_path.read_text(encoding="utf-8")

    assert written_artifact == artifact
    assert artifact["status"] == "complete"
    assert artifact["paper_source_path"] == "docs/arxiv-paper/main.tex"
    assert artifact["paper_updated"] is True
    assert artifact["anchored_claim_count"] == 4
    assert artifact["arxiv_submission_triggered"] is False
    assert "\\section{Anchored Claims}" in paper
    assert "CLAIM-4" in matrix
    assert "results/experiment_1460_hardware_portfolio_narrowing.json" in matrix


def test_scenario_publish_023_run_without_paper_keeps_artifact_honest(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-021: missing paper source still writes a complete claim matrix."""

    _write_source_inputs(tmp_path)
    out_path = tmp_path / "results" / exp1462.OUTPUT_FILENAME

    artifact = exp1462.run(root=tmp_path, out_path=out_path)

    assert artifact["status"] == "complete"
    assert artifact["paper_source_path"] is None
    assert artifact["paper_updated"] is False
    assert artifact["anchored_claim_count"] == 4
    assert (tmp_path / exp1462.CLAIM_MATRIX_REL).exists()
    assert artifact["honest_verdict"] == (
        "paper_v6_narrowed_to_4_anchored_claims_no_paper_source_updated_false"
    )
