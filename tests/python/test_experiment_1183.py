"""Tests for Experiment 1183 paper-v5 recompile and arXiv bundle gate.

Spec traces: REQ-PUBLISH-010, SCENARIO-PUBLISH-009, SCENARIO-PUBLISH-010.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest

exp1183 = importlib.import_module("scripts.experiment_1183_paper_v5_recompile_arxiv_bundle_v6")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_paper(path: Path, text: str | None = None) -> None:
    paper = text or "\n".join(
        [
            r"\begin{abstract}Abstract text.\end{abstract}",
            r"\section{Introduction}",
            r"\section{Carnot Architectural Framework}",
            r"\section{Theoretical Bounds of Verification Composition}",
            r"\section{Hardware Acceleration \& Sampling Limits}",
            r"\section{Empirical Realities \& Anomalies}",
            r"\section{Phase 4: Carnot as Active Inference}",
            r"\section{Decentralization \& Deployment Sovereignty}",
            r"\section{Conclusion \& Roadmap}",
            r"\bibliography{carnot}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper, encoding="utf-8")


def _write_prereqs(results_dir: Path) -> None:
    _write_json(
        results_dir / "experiment_1180_paper_v5_critical_issues_1_5.json",
        {
            "honest_verdict": "all_5_critical_resolved",
            "critical_issues_fixed": 5,
            "acceptance_gate": True,
            "figure_integrity_script_active": True,
            "status": "success",
        },
    )
    _write_json(
        results_dir / "experiment_1181_paper_v5_high_issues_6_10.json",
        {
            "honest_verdict": "all_5_high_resolved",
            "high_severity_fixed": 5,
            "4_test_passes_high": True,
            "status": "success",
        },
    )


def test_count_banned_strings_matches_required_grep_terms() -> None:
    """REQ-PUBLISH-010: fabricated constants are counted for the final gate."""
    text = "11680 and 76130 and 76,130 and 15.6x are banned"
    assert exp1183.count_banned_strings(text) == 4
    assert exp1183.count_banned_strings("measured 249x on exp1094") == 0


def test_check_sections_present_requires_abstract_main_sections_and_references(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-010: PDF readability proxy checks paper structure."""
    paper = tmp_path / "main.tex"
    _write_paper(paper)
    assert exp1183.check_sections_present(paper.read_text(encoding="utf-8")) == []

    missing = exp1183.check_sections_present(r"\section{Introduction}")
    assert "Abstract" in missing
    assert "References" in missing


def test_parse_json_object_from_audit_output() -> None:
    """REQ-PUBLISH-010: audit command output is normalized into the artifact."""
    output = 'debug line\n{"n_mismatches": 0, "passes": true}\n'
    assert exp1183.parse_json_object(output) == {"n_mismatches": 0, "passes": True}
    assert exp1183.parse_json_object('log {bad json\n{"ok": true}') == {"ok": True}
    assert exp1183.parse_json_object("no json here") == {}


def test_helper_branches_for_gate_counts_and_parse_errors(tmp_path: Path) -> None:
    """REQ-PUBLISH-010: gate helpers handle malformed and alternate artifacts."""
    bad = tmp_path / "experiment_1180_bad.json"
    bad.write_text("{", encoding="utf-8")
    (tmp_path / "experiment_1181_dir.json").mkdir()

    status = exp1183.check_prerequisite_gates(tmp_path)

    assert status["required"]["exp1180"]["reason"] == "gate_not_true"
    assert exp1183._count_complete(True, 5) is True
    assert exp1183._count_complete(5.0, 5) is True
    assert exp1183._count_complete("5/5", 5) is True
    assert exp1183._count_complete(object(), 5) is False
    assert exp1183._artifact_gate_true({"status": "partial"}, "exp1180") is False
    assert exp1183._artifact_gate_true({"phase_1_critical_fixes_landed": "5/5"}, "exp1180") is True
    assert exp1183._artifact_gate_true({}, "unknown") is False


def test_known_remaining_issues_variants(tmp_path: Path) -> None:
    """REQ-PUBLISH-010: medium/low issue notes mirror exp1182 status."""
    assert exp1183.known_remaining_issues(tmp_path) == [
        "exp1182 medium/low issues are not merged into the v6 gate record"
    ]

    exp1182 = tmp_path / "experiment_1182_paper_v5_medium_low_issues_11_18.json"
    _write_json(exp1182, {"honest_verdict": "all_8_medium_low_resolved"})
    assert exp1183.known_remaining_issues(tmp_path) == []

    _write_json(
        exp1182, {"honest_verdict": "partial_fix", "issue_11_thinkprm_citation_fixed": True}
    )
    remaining = exp1183.known_remaining_issues(tmp_path)
    assert "ISSUE-12 FoVer holdout n disclosure" in remaining

    _write_json(
        exp1182,
        {
            "honest_verdict": "partial_fix",
            "issue_11_thinkprm_citation_fixed": True,
            "issue_12_holdout_n_stated": True,
            "issue_13_nrgpt_disclosure_added": True,
            "issue_14_soskan_auroc_reconciled": True,
            "issue_15_fig2_caveat_added": True,
            "issue_16_bibliography_ok": True,
            "issue_17_k15_caption_tightened": True,
            "issue_18_hardware_scope_added": True,
        },
    )
    assert exp1183.known_remaining_issues(tmp_path) == [
        "exp1182 artifact did not report all medium/low issues resolved"
    ]


def test_prerequisite_gate_requires_exp1180_and_exp1181(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-009: missing exp1180 blocks before recompile."""
    _write_json(
        tmp_path / "experiment_1181_paper_v5_high_issues_6_10.json",
        {"honest_verdict": "all_5_high_resolved", "high_severity_fixed": 5},
    )

    status = exp1183.check_prerequisite_gates(tmp_path)

    assert status["all_required_gates_true"] is False
    assert status["required"]["exp1180"]["gate_true"] is False
    assert status["required"]["exp1181"]["gate_true"] is True


def test_prerequisite_gate_accepts_successful_artifacts(tmp_path: Path) -> None:
    """REQ-PUBLISH-010: exp1180 and exp1181 true gates permit the bundle step."""
    _write_prereqs(tmp_path)

    status = exp1183.check_prerequisite_gates(tmp_path)

    assert status["all_required_gates_true"] is True
    assert status["required"]["exp1180"]["gate_true"] is True
    assert status["required"]["exp1181"]["gate_true"] is True


def test_audit_runners_normalize_missing_output_and_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PUBLISH-010: audit runners always return structured status."""
    missing = tmp_path / "missing.py"
    monkeypatch.setattr(exp1183, "FIGURE_AUDIT_SCRIPT", missing)
    assert exp1183.run_figure_integrity_audit().returncode == 127
    monkeypatch.setattr(exp1183, "CLAIM_AUDIT_SCRIPT", missing)
    assert exp1183.run_paper_claim_audit().returncode == 127

    script = tmp_path / "audit.py"
    script.write_text("print('audit')\n", encoding="utf-8")
    monkeypatch.setattr(exp1183, "FIGURE_AUDIT_SCRIPT", script)
    monkeypatch.setattr(exp1183, "CLAIM_AUDIT_SCRIPT", script)

    monkeypatch.setattr(
        exp1183,
        "_run_script",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"untraced_constants": 0}',
            stderr="",
        ),
    )
    assert exp1183.run_figure_integrity_audit().passed is True

    monkeypatch.setattr(
        exp1183,
        "_run_script",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"n_mismatches": 0, "passes": true}',
            stderr="",
        ),
    )
    assert exp1183.run_paper_claim_audit().passed is True

    monkeypatch.setattr(
        exp1183,
        "_run_script",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not json", stderr="stderr"),
    )
    assert exp1183.run_figure_integrity_audit().report["raw_stdout_tail"] == "not json"
    assert exp1183.run_paper_claim_audit().passed is False

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="audit", timeout=1)

    monkeypatch.setattr(exp1183, "_run_script", _raise_timeout)
    assert exp1183.run_figure_integrity_audit().returncode == 124
    assert exp1183.run_paper_claim_audit().returncode == 124


def test_compile_latex_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PUBLISH-010: LaTeX compile reports tool and command failures honestly."""
    paper = tmp_path / "main.tex"
    pdf = tmp_path / "main.pdf"
    paper.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")

    monkeypatch.setattr(
        exp1183.shutil, "which", lambda name: None if name == "pdflatex" else "/bin/bibtex"
    )
    assert exp1183.try_compile_latex(paper, pdf)["pdflatex_available"] is False

    monkeypatch.setattr(
        exp1183.shutil, "which", lambda name: "/bin/pdflatex" if name == "pdflatex" else None
    )
    assert exp1183.try_compile_latex(paper, pdf)["bibtex_available"] is False

    monkeypatch.setattr(exp1183.shutil, "which", lambda name: f"/bin/{name}")

    def _raise_oserror(*args, **kwargs):
        raise OSError("no exec")

    monkeypatch.setattr(exp1183.subprocess, "run", _raise_oserror)
    assert "no exec" in exp1183.try_compile_latex(paper, pdf)["log_tail"]

    monkeypatch.setattr(
        exp1183.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="bad", stderr="err"),
    )
    assert exp1183.try_compile_latex(paper, pdf)["compiled"] is False

    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(
        exp1183.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    assert exp1183.try_compile_latex(paper, pdf)["compiled"] is True


def test_build_arxiv_bundle_includes_sources_and_pdf_png_figures(tmp_path: Path) -> None:
    """REQ-PUBLISH-010: source bundle contains arXiv source and renderable figures."""
    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    bib = tmp_path / "docs" / "arxiv-paper" / "carnot.bib"
    figures = tmp_path / "docs" / "arxiv-paper" / "figures"
    paper.parent.mkdir(parents=True)
    figures.mkdir(parents=True)
    paper.write_text("paper", encoding="utf-8")
    bib.write_text("bib", encoding="utf-8")
    (figures / "fig1.pdf").write_bytes(b"pdf")
    (figures / "fig2.png").write_bytes(b"png")
    (figures / "ignore.txt").write_text("ignore", encoding="utf-8")

    bundle = tmp_path / "bundle.tar.gz"
    exp1183.build_arxiv_bundle(
        bundle_path=bundle,
        paper_tex=paper,
        paper_bib=bib,
        arxiv_figures_dir=figures,
        docs_figures_dir=tmp_path / "missing-docs-figures",
    )

    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
    assert "docs/arxiv-paper/main.tex" in names
    assert "docs/arxiv-paper/carnot.bib" in names
    assert "docs/arxiv-paper/figures/fig1.pdf" in names
    assert "docs/arxiv-paper/figures/fig2.png" in names
    assert "docs/arxiv-paper/figures/ignore.txt" not in names


def test_run_stops_before_audits_compile_or_bundle_when_prereq_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PUBLISH-009: missing prior gate writes schema but has no side effects."""
    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    bib = tmp_path / "docs" / "arxiv-paper" / "carnot.bib"
    output = tmp_path / "results" / "experiment_1183_paper_v5_recompile_arxiv_bundle_v6.json"
    _write_paper(paper)
    bib.parent.mkdir(parents=True, exist_ok=True)
    bib.write_text("@article{x,title={x}}\n", encoding="utf-8")

    def _boom(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError("exp1183 proceeded despite missing prerequisite gate")

    monkeypatch.setattr(exp1183, "run_figure_integrity_audit", _boom)
    monkeypatch.setattr(exp1183, "run_paper_claim_audit", _boom)
    monkeypatch.setattr(exp1183, "try_compile_latex", _boom)
    monkeypatch.setattr(exp1183, "build_arxiv_bundle", _boom)

    artifact = exp1183.run(
        paper_tex=paper,
        paper_bib=bib,
        paper_pdf=tmp_path / "docs" / "arxiv-paper" / "main.pdf",
        results_dir=tmp_path / "results",
        output_path=output,
        bundle_path=tmp_path / "bundle.tar.gz",
    )

    assert output.exists()
    assert artifact["prerequisites_met"] is False
    assert artifact["arxiv_bundle_v6_ready"] is False
    assert artifact["4_test_full_pass"] is False
    assert artifact["honest_verdict"] == "audit_failures_remain"
    assert set(exp1183.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)


def test_run_records_bundle_with_audit_failure_without_blocking_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PUBLISH-010: audit failures are recorded while the tarball is built."""
    results = tmp_path / "results"
    _write_prereqs(results)
    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    bib = tmp_path / "docs" / "arxiv-paper" / "carnot.bib"
    pdf = tmp_path / "docs" / "arxiv-paper" / "main.pdf"
    _write_paper(paper)
    bib.parent.mkdir(parents=True, exist_ok=True)
    bib.write_text("@article{x,title={x}}\n", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    monkeypatch.setattr(
        exp1183,
        "run_figure_integrity_audit",
        lambda: exp1183.AuditRun(
            returncode=1,
            stdout='{"untraced_constants": 2}',
            stderr="",
            report={"untraced_constants": 2},
            passed=False,
        ),
    )
    monkeypatch.setattr(
        exp1183,
        "run_paper_claim_audit",
        lambda: exp1183.AuditRun(
            returncode=0,
            stdout='{"n_mismatches": 0, "passes": true}',
            stderr="",
            report={"n_mismatches": 0, "passes": True},
            passed=True,
        ),
    )
    monkeypatch.setattr(
        exp1183,
        "try_compile_latex",
        lambda *args, **kwargs: {
            "compiled": True,
            "pdflatex_available": True,
            "bibtex_available": True,
            "output_pdf_exists": True,
            "log_tail": "",
        },
    )

    bundle = tmp_path / "bundle.tar.gz"

    def _fake_bundle(*args, **kwargs):
        bundle.write_bytes(b"bundle")
        return str(bundle)

    monkeypatch.setattr(exp1183, "build_arxiv_bundle", _fake_bundle)

    artifact = exp1183.run(
        paper_tex=paper,
        paper_bib=bib,
        paper_pdf=pdf,
        results_dir=results,
        output_path=results / "experiment_1183_paper_v5_recompile_arxiv_bundle_v6.json",
        bundle_path=bundle,
    )

    assert artifact["arxiv_bundle_v6_ready"] is True
    assert artifact["figure_audit_untraced_constants"] == 2
    assert artifact["claim_audit_n_mismatches"] == 0
    assert artifact["4_test_full_pass"] is False
    assert artifact["honest_verdict"] == "audit_failures_remain"


def test_run_records_clean_ready_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PUBLISH-010: successful audits and bundle produce the ready verdict."""
    results = tmp_path / "results"
    _write_prereqs(results)
    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    bib = tmp_path / "docs" / "arxiv-paper" / "carnot.bib"
    pdf = tmp_path / "docs" / "arxiv-paper" / "main.pdf"
    _write_paper(paper)
    bib.parent.mkdir(parents=True, exist_ok=True)
    bib.write_text("@article{x,title={x}}\n", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    clean_figure = exp1183.AuditRun(0, "{}", "", {"untraced_constants": 0}, True)
    clean_claims = exp1183.AuditRun(0, "{}", "", {"n_mismatches": 0, "passes": True}, True)
    monkeypatch.setattr(exp1183, "run_figure_integrity_audit", lambda: clean_figure)
    monkeypatch.setattr(exp1183, "run_paper_claim_audit", lambda: clean_claims)
    monkeypatch.setattr(
        exp1183,
        "try_compile_latex",
        lambda *args, **kwargs: {
            "compiled": True,
            "pdflatex_available": True,
            "bibtex_available": True,
            "output_pdf_exists": True,
            "log_tail": "",
        },
    )
    bundle = tmp_path / "bundle.tar.gz"
    monkeypatch.setattr(
        exp1183,
        "build_arxiv_bundle",
        lambda *args, **kwargs: bundle.write_bytes(b"bundle") and str(bundle),
    )

    artifact = exp1183.run(
        paper_tex=paper,
        paper_bib=bib,
        paper_pdf=pdf,
        results_dir=results,
        output_path=results / "experiment_1183_paper_v5_recompile_arxiv_bundle_v6.json",
        bundle_path=bundle,
    )

    assert artifact["pdf_compiles_without_error"] is True
    assert artifact["arxiv_bundle_v6_ready"] is True
    assert artifact["4_test_full_pass"] is True
    assert artifact["honest_verdict"] == "arxiv_bundle_v6_ready"
