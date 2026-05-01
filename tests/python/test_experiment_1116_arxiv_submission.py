"""Tests for ``scripts/experiment_1116_arxiv_submission.py``.

Spec: REQ-PUB-001 — arXiv submission for Position Paper v3 (milestone .87).

These tests cover the helper functions and ``build_artifact``
integration path added for Exp 1116. They do *not* attempt to actually
upload to arXiv -- that requires interactive browser auth and is
explicitly the residual manual step the experiment documents.

Why these specific tests:

* ``find_latex_engine`` returns ``"none"`` on the dev machine that
  produced this artifact, but a future runner may install pdflatex or
  tectonic; we exercise both branches via ``shutil.which``
  monkeypatching so behaviour is pinned in either environment.
* ``author_identity_filled`` is the gate that decides whether the
  honest verdict can be ``bundle_ready_for_manual_upload`` rather than
  ``bundle_only_pdf_failed``; both the placeholder-detection path and
  the canonical-author path need explicit coverage.
* ``manual_submission_checklist`` shape changes based on whether a
  local PDF was produced; we test both branches because the checklist
  is the operator-facing artifact.
* ``build_artifact`` is the integration test: every schema field the
  .87 roadmap required must be present and well-typed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1116_arxiv_submission.py"


def _load_module():
    """Load the experiment script as a module without package install.

    The conductor exposes ``scripts/`` via ``PYTHONPATH`` only when
    invoked through its own pipeline; tests run under pytest without
    that setup, so we hand-load the file by spec.
    """
    spec = importlib.util.spec_from_file_location("exp1116", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1116"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1116():
    return _load_module()


def test_find_latex_engine_none(exp1116, monkeypatch):
    monkeypatch.setattr(exp1116.shutil, "which", lambda _: None)
    assert exp1116.find_latex_engine() == "none"


def test_find_latex_engine_prefers_pdflatex(exp1116, monkeypatch):
    monkeypatch.setattr(exp1116.shutil, "which", lambda name: "/usr/bin/" + name)
    assert exp1116.find_latex_engine() == "pdflatex"


def test_find_latex_engine_falls_back_to_tectonic(exp1116, monkeypatch):
    available = {"tectonic"}
    monkeypatch.setattr(
        exp1116.shutil,
        "which",
        lambda name: "/usr/bin/" + name if name in available else None,
    )
    assert exp1116.find_latex_engine() == "tectonic"


def test_author_identity_filled_canonical(exp1116, tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text(
        r"\documentclass{article}"
        + "\n"
        + r"\author{Ian Blenke \\ \texttt{icblenke@gmail.com}}"
        + "\n"
        + r"\begin{document}\end{document}",
        encoding="utf-8",
    )
    assert exp1116.author_identity_filled(tex) is True


def test_author_identity_filled_rejects_placeholder(exp1116, tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text(
        r"\author{Ian Blenke \\ \texttt{icblenke@gmail.com}}"
        + "\n"
        + r"% PLACEHOLDER text still present",
        encoding="utf-8",
    )
    assert exp1116.author_identity_filled(tex) is False


def test_author_identity_filled_rejects_missing_block(exp1116, tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text(r"\documentclass{article}", encoding="utf-8")
    assert exp1116.author_identity_filled(tex) is False


def test_author_identity_filled_missing_file(exp1116, tmp_path):
    assert exp1116.author_identity_filled(tmp_path / "absent.tex") is False


def test_manual_checklist_includes_arxiv_url(exp1116):
    steps = exp1116.manual_submission_checklist(pdf_compiled=True)
    joined = " ".join(steps)
    assert "https://arxiv.org/submit" in joined
    assert "cs.LG" in joined
    assert "cs.NE" in joined


def test_manual_checklist_flags_missing_pdf(exp1116):
    steps = exp1116.manual_submission_checklist(pdf_compiled=False)
    assert any("Local PDF compilation skipped" in s for s in steps)


def test_build_artifact_required_schema_keys(exp1116):
    artifact = exp1116.build_artifact(
        bundle_ok=True,
        missing_files=[],
        author_filled=True,
        engine="none",
        pdf_compiled=False,
        pdf_path=None,
        tarball_size=12345,
    )
    required = {
        "arxiv_bundle_created",
        "arxiv_bundle_path",
        "pdf_compiled",
        "pdf_path",
        "author_identity_filled",
        "latex_engine_used",
        "arxiv_submitted",
        "arxiv_submission_id",
        "submission_deadline",
        "manual_steps_remaining",
        "honest_verdict",
    }
    assert required.issubset(artifact.keys())
    assert artifact["submission_deadline"] == "2026-05-15"
    assert artifact["arxiv_submitted"] is False
    assert isinstance(artifact["manual_steps_remaining"], list)


def test_build_artifact_verdict_bundle_ready(exp1116):
    art = exp1116.build_artifact(
        bundle_ok=True,
        missing_files=[],
        author_filled=True,
        engine="none",
        pdf_compiled=False,
        pdf_path=None,
        tarball_size=1,
    )
    assert art["honest_verdict"] == "bundle_ready_for_manual_upload"


def test_build_artifact_verdict_pdf_compiled(exp1116):
    art = exp1116.build_artifact(
        bundle_ok=True,
        missing_files=[],
        author_filled=True,
        engine="pdflatex",
        pdf_compiled=True,
        pdf_path="docs/arxiv-paper/main.pdf",
        tarball_size=1,
    )
    assert art["honest_verdict"] == "pdf_compiled_upload_pending"


def test_build_artifact_verdict_failed_when_author_missing(exp1116):
    art = exp1116.build_artifact(
        bundle_ok=True,
        missing_files=[],
        author_filled=False,
        engine="none",
        pdf_compiled=False,
        pdf_path=None,
        tarball_size=1,
    )
    assert art["honest_verdict"] == "bundle_only_pdf_failed"
