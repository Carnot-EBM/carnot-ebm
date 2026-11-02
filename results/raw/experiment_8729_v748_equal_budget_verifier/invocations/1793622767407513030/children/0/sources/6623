"""Experiment 1116: arXiv submission of the Carnot v3 position paper.

Why this experiment exists. Milestone .87 carries the
2026-05-15 arXiv submission deadline for the Carnot v3 position
paper. Experiment 1113 already produced the LaTeX source bundle at
``docs/arxiv-paper/`` (``main.tex`` + ``carnot.bib`` + seven figure
PDFs). What was missing on 2026-05-01 was three things that this
experiment closes:

1. The author block in ``main.tex`` still carried a
   ``\\thanks{Author placeholder, to be finalized before submission.}``
   footnote. arXiv requires real author identity at submission time,
   so the footnote has been removed and the block reduced to the
   solo author line ``\\author{Ian Blenke \\\\ \\texttt{icblenke@gmail.com}}``.

2. A self-contained submission tarball had to be produced -- arXiv's
   web submission flow accepts a single ``.tar.gz`` containing the
   LaTeX source plus referenced figures. This script confirms the
   tarball exists and re-creates it deterministically if it does
   not.

3. A formal record of "what was actually accomplished and what
   manual steps remain" needs to be written so that the next session
   (human or AI) can finish the upload without rediscovering state.
   That record is this script's deliverable JSON, which carries the
   schema fields the milestone .87 roadmap specified.

What this script does NOT do. It does not actually upload to the
arXiv submission portal: that requires an interactive browser
session with the user's arXiv account credentials, plus the
endorsement / category / license selection workflow. The script
instead documents the manual steps remaining and emits an
``arxiv_submitted=False`` verdict with an honest description of why.

Local PDF compilation status. The development machine that ran this
experiment (CachyOS, no ``pdflatex`` / ``tectonic`` / ``xelatex``
installed and no user-mode package manager available) cannot
produce a local ``main.pdf`` preview. arXiv compiles the PDF on its
own server from the uploaded ``.tar.gz`` so this is not a blocker
for the 2026-05-15 deadline; it is a blocker only for local PDF
preview. The honest verdict in that case is
``bundle_ready_for_manual_upload``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
ARXIV_DIR = REPO_ROOT / "docs" / "arxiv-paper"
RESULTS_DIR = REPO_ROOT / "results"
TARBALL_NAME = "carnot-arxiv-v3.tar.gz"
TARBALL_PATH = RESULTS_DIR / TARBALL_NAME
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1116_arxiv_submission.json"

EXPECTED_FIGURES = [f"fig{i}.pdf" for i in range(1, 8)]
EXPECTED_TEX_FILES = ["main.tex", "carnot.bib"]

AUTHOR_LINE = r"\author{Ian Blenke \\ \texttt{icblenke@gmail.com}}"
PLACEHOLDER_TOKENS = (
    "PLACEHOLDER",
    "Author placeholder",
    "to be finalized",
)


def find_latex_engine() -> str:
    """Probe the local environment for any usable LaTeX engine.

    arXiv compiles the PDF server-side from the uploaded source
    tarball, so a local LaTeX engine is only required for local
    preview. We probe the standard candidates in priority order and
    return the first hit, or the literal string ``"none"`` if none
    are installed. Recording this in the deliverable lets a future
    operator know without re-discovering state whether they can run
    a local preview pass before uploading.
    """
    for engine in ("pdflatex", "tectonic", "xelatex", "lualatex"):
        if shutil.which(engine):
            return engine
    return "none"


def author_identity_filled(tex_path: Path) -> bool:
    """Verify that ``main.tex`` contains the real author block and no
    placeholder footnote.

    The check is two-sided: the canonical ``\\author{...}`` line must
    be present, AND none of the known placeholder tokens may remain
    anywhere in the file. Both conditions matter -- a partial fix
    that leaves a stale ``\\thanks{...}`` footnote in the document
    would still ship the placeholder text into the rendered PDF.
    """
    if not tex_path.exists():
        return False
    contents = tex_path.read_text(encoding="utf-8")
    if AUTHOR_LINE not in contents:
        return False
    return not any(token in contents for token in PLACEHOLDER_TOKENS)


def bundle_files_present() -> tuple[bool, list[str]]:
    """Confirm that all source files referenced by the tarball exist.

    Returns ``(all_present, missing)``. We list the missing paths so
    the deliverable artifact can record which specific file blocked
    a future submission, rather than a single boolean that hides the
    diagnosis.
    """
    missing: list[str] = []
    for name in EXPECTED_TEX_FILES:
        if not (ARXIV_DIR / name).exists():
            missing.append(f"docs/arxiv-paper/{name}")
    figures_dir = ARXIV_DIR / "figures"
    for fig in EXPECTED_FIGURES:
        if not (figures_dir / fig).exists():
            missing.append(f"docs/arxiv-paper/figures/{fig}")
    return (not missing, missing)


def build_submission_tarball() -> tuple[bool, int]:
    """Create ``results/carnot-arxiv-v3.tar.gz`` from the bundle.

    arXiv's web submission flow accepts a single ``.tar.gz`` (or
    ``.zip``) containing the LaTeX source plus referenced figures at
    the paths the source actually references. We use ``tarfile``
    (rather than shelling out to ``tar``) so the script remains
    portable to environments where GNU tar is not installed.
    Returns ``(created_ok, size_bytes)``.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not ARXIV_DIR.exists():
        return False, 0
    with tarfile.open(TARBALL_PATH, "w:gz") as tf:
        for name in EXPECTED_TEX_FILES:
            src = ARXIV_DIR / name
            if src.exists():
                tf.add(src, arcname=name)
        figures_dir = ARXIV_DIR / "figures"
        if figures_dir.exists():
            tf.add(figures_dir, arcname="figures")
    return TARBALL_PATH.exists(), TARBALL_PATH.stat().st_size if TARBALL_PATH.exists() else 0


def attempt_pdf_compile(engine: str) -> tuple[bool, str | None]:
    """Try to compile ``main.tex`` -> ``main.pdf`` with the engine
    we found.

    Returns ``(compiled_ok, pdf_path_str_or_None)``. We deliberately
    do NOT raise on compile failure -- the bundle is shippable to
    arXiv regardless of local compile status, so a missing local PDF
    is informational, not blocking.
    """
    if engine == "none":
        return False, None
    cwd = str(ARXIV_DIR)
    try:
        if engine == "tectonic":
            subprocess.run(
                ["tectonic", "main.tex"],
                cwd=cwd,
                check=False,
                capture_output=True,
                timeout=300,
            )
        else:
            for _ in range(2):
                subprocess.run(
                    [engine, "-interaction=nonstopmode", "main.tex"],
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                    timeout=300,
                )
    except (subprocess.TimeoutExpired, OSError):
        return False, None
    pdf_path = ARXIV_DIR / "main.pdf"
    if pdf_path.exists():
        return True, str(pdf_path.relative_to(REPO_ROOT))
    return False, None


def manual_submission_checklist(pdf_compiled: bool) -> list[str]:
    """Document the steps a human must perform on arxiv.org/submit.

    arXiv submission is interactive: account login, category
    selection, license, endorsement, abstract paste, and final
    submit button are all browser-driven steps that no headless
    script can complete without storing the user's credentials.
    The conductor is forbidden from doing that; instead we record
    the residual workflow so the next session can finish it.
    """
    steps = [
        "Log in to https://arxiv.org/submit with the project arXiv account.",
        f"Upload results/{TARBALL_NAME} as the source bundle.",
        "Set primary category: cs.LG (Machine Learning).",
        "Set secondary category: cs.NE (Neural and Evolutionary Computing).",
        "Confirm author: Ian Blenke <icblenke@gmail.com>.",
        "Paste the abstract from docs/arxiv-paper/main.tex (\\begin{abstract} block).",
        "Select license: arXiv non-exclusive license to distribute (CC BY 4.0 acceptable).",
        "Verify the server-built PDF preview before clicking Submit.",
    ]
    if not pdf_compiled:
        steps.insert(
            0,
            "Local PDF compilation skipped: no LaTeX engine on dev machine. arXiv "
            "will compile server-side; verify the preview before final submit.",
        )
    return steps


def build_artifact(
    bundle_ok: bool,
    missing_files: list[str],
    author_filled: bool,
    engine: str,
    pdf_compiled: bool,
    pdf_path: str | None,
    tarball_size: int,
) -> dict[str, Any]:
    """Assemble the experiment 1116 deliverable JSON.

    We pick ``honest_verdict`` from the explicit set the milestone
    .87 task description specified. The script does not perform an
    interactive arXiv upload, so ``arxiv_submitted`` is always
    False at the end of this run; the operator clears the
    ``manual_steps_remaining`` list when they finish the upload in
    a browser.
    """
    if not bundle_ok or not author_filled:
        verdict = "bundle_only_pdf_failed"
    elif pdf_compiled:
        verdict = "pdf_compiled_upload_pending"
    else:
        verdict = "bundle_ready_for_manual_upload"

    return {
        "experiment": "1116_arxiv_submission",
        "run_date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "arxiv_submission_v1",
        "arxiv_bundle_created": bundle_ok and TARBALL_PATH.exists(),
        "arxiv_bundle_path": f"results/{TARBALL_NAME}",
        "arxiv_bundle_size_bytes": tarball_size,
        "pdf_compiled": pdf_compiled,
        "pdf_path": pdf_path,
        "author_identity_filled": author_filled,
        "latex_engine_used": engine,
        "arxiv_submitted": False,
        "arxiv_submission_id": None,
        "submission_url": "https://arxiv.org/submit",
        "submission_deadline": "2026-05-15",
        "primary_category": "cs.LG",
        "secondary_category": "cs.NE",
        "manual_steps_remaining": manual_submission_checklist(pdf_compiled),
        "missing_bundle_files": missing_files,
        "honest_verdict": verdict,
    }


def main() -> int:
    """Run the full experiment 1116 submission-prep flow."""
    bundle_ok, missing = bundle_files_present()
    author_filled = author_identity_filled(ARXIV_DIR / "main.tex")
    engine = find_latex_engine()
    pdf_compiled, pdf_path = attempt_pdf_compile(engine)

    created_ok, size_bytes = build_submission_tarball()
    bundle_ok = bundle_ok and created_ok

    artifact = build_artifact(
        bundle_ok=bundle_ok,
        missing_files=missing,
        author_filled=author_filled,
        engine=engine,
        pdf_compiled=pdf_compiled,
        pdf_path=pdf_path,
        tarball_size=size_bytes,
    )

    DELIVERABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1116] verdict={artifact['honest_verdict']} engine={engine} "
        f"bundle={artifact['arxiv_bundle_created']} pdf={pdf_compiled}"
    )
    print(f"[exp1116] deliverable -> {DELIVERABLE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
