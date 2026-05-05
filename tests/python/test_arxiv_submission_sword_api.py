"""Tests for the Exp 1390 arXiv SWORD submission runner.

Spec traces: REQ-PUBLISH-019, SCENARIO-PUBLISH-020, SCENARIO-PUBLISH-021.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting import arxiv_submission_sword_api as exp1390


def _write_ready_bundle(root: Path) -> Path:
    bundle = root / "results" / "arxiv_bundle_v11.tar.gz"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"not empty")
    return bundle


def _write_paper(root: Path) -> None:
    paper = root / "docs" / "arxiv-paper" / "main.tex"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(
        "\n".join(
            [
                r"\title{Carnot: Test Submission}",
                r"\author{Ian Blenke \\ \texttt{ian@blenke.com}}",
                r"\begin{abstract}",
                r"A concise abstract for the submission workflow with $k^* \leq 3.125$.",
                r"\end{abstract}",
            ]
        ),
        encoding="utf-8",
    )


def test_in_progress_artifact_contains_required_fields(tmp_path: Path) -> None:
    """REQ-PUBLISH-019: the runner writes the required placeholder first."""
    out_path = tmp_path / "results" / "experiment_1390_arxiv_submission_sword_api.json"

    artifact = exp1390.write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    for field in (
        "status",
        "bundle_path",
        "submission_attempted",
        "submission_method",
        "arxiv_id_if_submitted",
        "submission_result",
        "manual_checklist_generated",
        "manual_checklist_path",
        "honest_verdict",
    ):
        assert field in written


def test_missing_credentials_generate_manual_checklist(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-020: no credentials produces browser-ready instructions."""
    _write_ready_bundle(tmp_path)
    _write_paper(tmp_path)

    artifact = exp1390.run(project_root=tmp_path, environ={})

    assert artifact["status"] == "complete"
    assert artifact["submission_attempted"] is False
    assert artifact["submission_method"] == "manual_checklist_no_credentials"
    assert artifact["submission_result"] == "manual_checklist_generated"
    assert artifact["manual_checklist_generated"] is True
    checklist = tmp_path / artifact["manual_checklist_path"]
    text = checklist.read_text(encoding="utf-8")
    assert "https://arxiv.org/submit" in text
    assert "results/arxiv_bundle_v11.tar.gz" in text
    assert "Carnot: Test Submission" in text
    assert "ian@blenke.com" in text
    assert "cs.LG" in text
    assert "CC-BY-4.0" in text
    assert r"$k^* \leq 3.125$" in text


def test_credentials_post_bundle_and_extract_arxiv_id(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-021: credentials trigger the SWORD POST path."""
    _write_ready_bundle(tmp_path)
    _write_paper(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_post(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status_code=201,
            text="Created deposit for arXiv:2605.01234v1",
            headers={"location": "https://arxiv.org/abs/2605.01234"},
        )

    artifact = exp1390.run(
        project_root=tmp_path,
        environ={
            "ARXIV_SWORD_USERNAME": "user",
            "ARXIV_SWORD_PASSWORD": "pass",
        },
        http_post=fake_post,
    )

    assert artifact["status"] == "complete"
    assert artifact["submission_attempted"] is True
    assert artifact["submission_method"] == "sword_api"
    assert artifact["submission_result"] == "submitted"
    assert artifact["arxiv_id_if_submitted"] == "2605.01234"
    assert artifact["manual_checklist_generated"] is False
    assert len(calls) == 1
    assert calls[0]["url"] == exp1390.SWORD_DEPOSIT_URL
    assert calls[0]["auth"] == ("user", "pass")
    assert "Carnot: Test Submission" in str(calls[0]["data"])


def test_missing_bundle_blocks_without_submission(tmp_path: Path) -> None:
    """REQ-PUBLISH-019: an absent bundle blocks before network submission."""
    _write_paper(tmp_path)

    artifact = exp1390.run(
        project_root=tmp_path,
        environ={
            "ARXIV_SWORD_USERNAME": "user",
            "ARXIV_SWORD_PASSWORD": "pass",
        },
        http_post=lambda **_kwargs: SimpleNamespace(status_code=201, text="submitted"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["submission_attempted"] is False
    assert artifact["submission_result"] == "not_attempted_bundle_missing_or_empty"
    assert artifact["manual_checklist_generated"] is False
