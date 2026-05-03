"""Tests for scripts/paper_claim_audit.py.

Spec coverage:
  REQ-PUBLISH-009 — Paper Numerical Claim Audit
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import paper_claim_audit as audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_claims_normalizes_latex_units_for_req_publish_009() -> None:
    """REQ-PUBLISH-009: LaTeX paper claims are normalized before extraction."""
    tex = (
        "Latency was $24.83\\,\\mu s$ (exp1068). "
        "Compression was 5.03$\\times$ and accuracy was 100\\% (exp1148)."
    )

    claims = audit.extract_claims(tex)

    assert [claim.raw_value for claim in claims] == ["24.83", "5.03", "100"]
    assert [claim.unit for claim in claims] == ["µs", "×", "%"]


def test_audit_verifies_matching_artifact_values_for_req_publish_009(tmp_path: Path) -> None:
    """REQ-PUBLISH-009: cited claims match numeric fields in result artifacts."""
    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        "The board measured 24.83µs (exp1068). "
        "The compressed verifier was 5.03x smaller (exp1148). "
        "Accuracy reached 100% (exp1147).",
        encoding="utf-8",
    )
    _write_json(tmp_path / "results" / "experiment_1068_latency.json", {"latency_us": 24.83})
    _write_json(
        tmp_path / "results" / "experiment_1148_compression.json", {"size_reduction": 5.026}
    )
    _write_json(tmp_path / "results" / "experiment_1147_projection.json", {"accuracy": 1.0})

    report = audit.audit_paper_claims(paper, tmp_path / "results")

    assert report["n_claims_total"] == 3
    assert report["n_claims_with_artifact_citation"] == 3
    assert report["n_claims_verified"] == 3
    assert report["n_mismatches"] == 0
    assert report["passes"] is True


def test_audit_reports_mismatch_and_low_citation_ratio_for_req_publish_009(tmp_path: Path) -> None:
    """REQ-PUBLISH-009: mismatches and under-cited claim sets fail the audit."""
    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        "Latency was 24.83µs (exp1068). Accuracy was 91%. Delta was 8.5pp.",
        encoding="utf-8",
    )
    _write_json(tmp_path / "results" / "experiment_1068_latency.json", {"latency_us": 99.0})

    report = audit.audit_paper_claims(paper, tmp_path / "results")

    assert report["n_claims_total"] == 3
    assert report["n_claims_with_artifact_citation"] == 1
    assert report["n_mismatches"] == 1
    assert report["passes"] is False
    assert report["mismatches"][0]["exp_id"] == "1068"


def test_main_exits_nonzero_when_req_publish_009_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-PUBLISH-009: CLI exits with code 1 for failing audits."""
    paper = tmp_path / "main.tex"
    paper.write_text("Latency was 24.83µs (exp1068).", encoding="utf-8")
    _write_json(tmp_path / "results" / "experiment_1068_latency.json", {"latency_us": 1.0})

    monkeypatch.setattr(
        audit.sys, "argv", ["paper_claim_audit.py", str(paper), str(tmp_path / "results")]
    )

    with pytest.raises(SystemExit) as excinfo:
        audit.main()

    assert excinfo.value.code == 1
    assert json.loads(capsys.readouterr().out)["n_mismatches"] == 1
