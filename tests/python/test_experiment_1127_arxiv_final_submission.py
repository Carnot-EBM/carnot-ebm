"""Tests for ``scripts/experiment_1127_arxiv_final_submission.py``.

Spec: REQ-PUB-001 -- arXiv submission for Position Paper v3
(milestone .88 follow-up to exp1116).

These tests cover the helpers and ``build_artifact`` integration
path added for Exp 1127. They never attempt a real arXiv upload
(that requires interactive browser auth and is the residual manual
step the experiment documents) and they never invoke a real LaTeX
engine (compile path is mocked via ``subprocess.run`` so the suite
runs in any environment, with or without tectonic installed).

Coverage rationale, in the order the tests appear:

* ``verify_tarball`` is the gate that decides whether the verdict
  can be ``pdf_compiled_upload_pending`` at all -- a missing or
  truncated tarball forces ``all_blocked_manual_only``. Both the
  happy path (build a complete tarball in ``tmp_path`` and confirm
  it verifies) and the unhappy paths (missing file, missing member,
  zero-byte file, corrupt gzip) need explicit coverage.
* ``find_latex_engine`` priority order matters: tectonic is
  preferred over pdflatex because tectonic is self-contained. We
  pin both branches via ``shutil.which`` monkeypatching so the test
  outcome does not depend on what is actually installed on the host.
* ``classify_verdict`` is the verdict mapping; it is called from
  ``build_artifact`` but is also a pure function over three
  booleans/strings, so it gets dedicated tests for every
  (bundle_ok, engine, pdf_compiled) tuple that resolves to a
  distinct verdict.
* ``manual_submission_steps`` shape changes based on (bundle_ok,
  pdf_compiled). All three branches (bundle broken, bundle ok +
  pdf, bundle ok + no pdf) need coverage because the checklist is
  the operator-facing artifact -- a regression in step ordering or
  text would silently mislead a future submitter.
* ``build_artifact`` is the integration test: every required
  schema key from the milestone .88 task description must be
  present and well-typed, and the embedded verdict must match
  ``classify_verdict``'s output for the same inputs.
* ``attempt_pdf_compile`` is exercised against a mocked
  ``subprocess.run`` so we can prove the script tolerates a
  missing engine (``"none"``) without raising and that it correctly
  reports the PDF when one is created on disk.
"""

from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1127_arxiv_final_submission.py"


def _load_module():
    """Load the experiment script as a module without package install.

    Mirrors the loader used in ``test_experiment_1116_arxiv_submission.py``
    so the two suites stay symmetric and a future maintainer can
    copy-paste between them safely.
    """
    spec = importlib.util.spec_from_file_location("exp1127", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1127"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1127():
    return _load_module()


def _build_complete_tarball(target: Path) -> None:
    """Write a tarball that contains every member ``verify_tarball`` checks.

    Used by the happy-path test. The file contents themselves do
    not matter for verification (tar member-name listing is what we
    check), but we still write small payloads so the tarball is a
    realistic gzipped archive rather than an empty file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    src = target.parent / "src"
    src.mkdir(exist_ok=True)
    (src / "main.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
    (src / "carnot.bib").write_text("@misc{x, title={x}}\n", encoding="utf-8")
    figures = src / "figures"
    figures.mkdir(exist_ok=True)
    for i in range(1, 8):
        (figures / f"fig{i}.pdf").write_bytes(b"%PDF-1.4\n%mocked\n")
    with tarfile.open(target, "w:gz") as tf:
        tf.add(src / "main.tex", arcname="main.tex")
        tf.add(src / "carnot.bib", arcname="carnot.bib")
        tf.add(figures, arcname="figures")


def test_verify_tarball_happy_path(exp1127, tmp_path):
    target = tmp_path / "carnot-arxiv-v3.tar.gz"
    _build_complete_tarball(target)
    ok, missing = exp1127.verify_tarball(target)
    assert ok is True
    assert missing == []


def test_verify_tarball_missing_file(exp1127, tmp_path):
    target = tmp_path / "absent.tar.gz"
    ok, missing = exp1127.verify_tarball(target)
    assert ok is False
    assert "main.tex" in missing


def test_verify_tarball_zero_bytes(exp1127, tmp_path):
    target = tmp_path / "empty.tar.gz"
    target.touch()
    ok, missing = exp1127.verify_tarball(target)
    assert ok is False
    assert "main.tex" in missing


def test_verify_tarball_corrupt(exp1127, tmp_path):
    target = tmp_path / "corrupt.tar.gz"
    target.write_bytes(b"not a real gzip file")
    ok, missing = exp1127.verify_tarball(target)
    assert ok is False
    assert missing  # full expected list returned on parse failure


def test_verify_tarball_missing_member(exp1127, tmp_path):
    target = tmp_path / "incomplete.tar.gz"
    src = tmp_path / "src2"
    src.mkdir()
    (src / "main.tex").write_text("x", encoding="utf-8")
    with tarfile.open(target, "w:gz") as tf:
        tf.add(src / "main.tex", arcname="main.tex")
    ok, missing = exp1127.verify_tarball(target)
    assert ok is False
    assert "carnot.bib" in missing
    assert "figures/fig1.pdf" in missing


def test_find_latex_engine_none(exp1127, monkeypatch):
    monkeypatch.setattr(exp1127.shutil, "which", lambda _: None)
    assert exp1127.find_latex_engine() == "none"


def test_find_latex_engine_prefers_tectonic(exp1127, monkeypatch):
    monkeypatch.setattr(exp1127.shutil, "which", lambda name: "/usr/bin/" + name)
    assert exp1127.find_latex_engine() == "tectonic"


def test_find_latex_engine_falls_back_to_pdflatex(exp1127, monkeypatch):
    available = {"pdflatex", "xelatex"}
    monkeypatch.setattr(
        exp1127.shutil,
        "which",
        lambda name: "/usr/bin/" + name if name in available else None,
    )
    # Tectonic missing -> next in priority list is pdflatex.
    assert exp1127.find_latex_engine() == "pdflatex"


def test_classify_verdict_blocked_when_bundle_missing(exp1127):
    assert exp1127.classify_verdict(False, "tectonic", True) == "all_blocked_manual_only"
    assert exp1127.classify_verdict(False, "none", False) == "all_blocked_manual_only"


def test_classify_verdict_pdf_compiled(exp1127):
    assert exp1127.classify_verdict(True, "tectonic", True) == "pdf_compiled_upload_pending"


def test_classify_verdict_bundle_ready_no_pdf(exp1127):
    assert exp1127.classify_verdict(True, "none", False) == "bundle_ready_tectonic_install_failed"


def test_manual_steps_bundle_broken(exp1127):
    steps = exp1127.manual_submission_steps(pdf_compiled=False, bundle_ok=False)
    assert len(steps) == 1
    assert "rerun experiment_1116" in steps[0]


def test_manual_steps_bundle_ok_pdf_compiled(exp1127):
    steps = exp1127.manual_submission_steps(pdf_compiled=True, bundle_ok=True)
    joined = " ".join(steps)
    assert "main.pdf" in joined
    assert "https://arxiv.org/submit" in joined
    assert "cs.LG" in joined
    assert "cs.NE" in joined
    assert "ian@blenke.com" in joined


def test_manual_steps_bundle_ok_no_pdf(exp1127):
    steps = exp1127.manual_submission_steps(pdf_compiled=False, bundle_ok=True)
    joined = " ".join(steps)
    assert "compile server-side" in joined
    assert "https://arxiv.org/submit" in joined


def test_build_artifact_required_schema_keys(exp1127):
    artifact = exp1127.build_artifact(
        bundle_ok=True,
        missing_files=[],
        engine="none",
        pdf_compiled=False,
        pdf_path=None,
        compile_log_tail="",
        tarball_size=12345,
    )
    required = {
        "arxiv_bundle_verified",
        "pdf_compiled",
        "pdf_path",
        "latex_engine_used",
        "arxiv_submitted",
        "arxiv_submission_id",
        "manual_steps_remaining",
        "submission_deadline",
        "honest_verdict",
    }
    assert required.issubset(artifact.keys())
    assert artifact["submission_deadline"] == "2026-05-15"
    assert artifact["arxiv_submitted"] is False
    assert artifact["arxiv_submission_id"] is None
    assert isinstance(artifact["manual_steps_remaining"], list)
    assert artifact["author_email"] == "ian@blenke.com"


def test_build_artifact_verdict_pdf_compiled(exp1127):
    art = exp1127.build_artifact(
        bundle_ok=True,
        missing_files=[],
        engine="tectonic",
        pdf_compiled=True,
        pdf_path="docs/arxiv-paper/main.pdf",
        compile_log_tail="ok",
        tarball_size=1,
    )
    assert art["honest_verdict"] == "pdf_compiled_upload_pending"
    assert art["pdf_path"] == "docs/arxiv-paper/main.pdf"


def test_build_artifact_verdict_bundle_ready(exp1127):
    art = exp1127.build_artifact(
        bundle_ok=True,
        missing_files=[],
        engine="none",
        pdf_compiled=False,
        pdf_path=None,
        compile_log_tail="",
        tarball_size=1,
    )
    assert art["honest_verdict"] == "bundle_ready_tectonic_install_failed"


def test_build_artifact_verdict_blocked(exp1127):
    art = exp1127.build_artifact(
        bundle_ok=False,
        missing_files=["main.tex"],
        engine="none",
        pdf_compiled=False,
        pdf_path=None,
        compile_log_tail="",
        tarball_size=0,
    )
    assert art["honest_verdict"] == "all_blocked_manual_only"
    assert art["missing_bundle_files"] == ["main.tex"]


def test_attempt_pdf_compile_no_engine(exp1127):
    ok, path, log = exp1127.attempt_pdf_compile("none")
    assert ok is False
    assert path is None
    assert log == ""


def test_attempt_pdf_compile_engine_creates_pdf(exp1127, monkeypatch, tmp_path):
    """If the engine produces a non-empty PDF on disk, we return success.

    We monkeypatch ARXIV_DIR / PDF_PATH so the test does not touch
    the real ``docs/arxiv-paper`` tree, then mock ``subprocess.run``
    to "succeed" without invoking a real engine. The script must
    notice the PDF that the mock dropped on disk and return
    ``(True, relative_path, log_tail)``.
    """
    fake_arxiv = tmp_path / "arxiv-paper"
    fake_arxiv.mkdir()
    fake_pdf = fake_arxiv / "main.pdf"

    monkeypatch.setattr(exp1127, "ARXIV_DIR", fake_arxiv)
    monkeypatch.setattr(exp1127, "PDF_PATH", fake_pdf)

    class _FakeProc:
        stdout = "compiled ok"
        stderr = ""

    def _fake_run(*_args, **_kwargs):
        fake_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
        return _FakeProc()

    monkeypatch.setattr(exp1127.subprocess, "run", _fake_run)
    ok, path, log = exp1127.attempt_pdf_compile("tectonic")
    assert ok is True
    assert path is not None and path.endswith("main.pdf")
    assert "compiled ok" in log


def test_attempt_pdf_compile_engine_fails_silently(exp1127, monkeypatch, tmp_path):
    """If the engine does NOT produce a PDF, we report failure without raising."""
    fake_arxiv = tmp_path / "arxiv-paper"
    fake_arxiv.mkdir()
    fake_pdf = fake_arxiv / "main.pdf"

    monkeypatch.setattr(exp1127, "ARXIV_DIR", fake_arxiv)
    monkeypatch.setattr(exp1127, "PDF_PATH", fake_pdf)

    class _FakeProc:
        stdout = "missing.sty not found"
        stderr = "fatal"

    monkeypatch.setattr(exp1127.subprocess, "run", lambda *a, **k: _FakeProc())
    ok, path, log = exp1127.attempt_pdf_compile("pdflatex")
    assert ok is False
    assert path is None
    assert "missing.sty" in log or "fatal" in log


def test_attempt_pdf_compile_handles_timeout(exp1127, monkeypatch, tmp_path):
    """A subprocess timeout is downgraded to a clean failure, never raised."""
    import subprocess as _sp

    fake_arxiv = tmp_path / "arxiv-paper"
    fake_arxiv.mkdir()
    monkeypatch.setattr(exp1127, "ARXIV_DIR", fake_arxiv)
    monkeypatch.setattr(exp1127, "PDF_PATH", fake_arxiv / "main.pdf")

    def _raise_timeout(*_a, **_k):
        raise _sp.TimeoutExpired(cmd="tectonic", timeout=1)

    monkeypatch.setattr(exp1127.subprocess, "run", _raise_timeout)
    ok, path, log = exp1127.attempt_pdf_compile("tectonic")
    assert ok is False
    assert path is None
    assert "engine_invocation_failed" in log


def test_allowed_verdicts_set_complete(exp1127):
    """Sanity-check the verdict set is exactly the four values the milestone defined."""
    assert exp1127.ALLOWED_VERDICTS == frozenset(
        {
            "submitted",
            "pdf_compiled_upload_pending",
            "bundle_ready_tectonic_install_failed",
            "all_blocked_manual_only",
        }
    )
