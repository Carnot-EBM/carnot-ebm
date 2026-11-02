"""Experiment 1127: final arXiv submission preparation for Carnot v3.

Why this experiment exists. Experiment 1116 produced the LaTeX
source bundle ``results/carnot-arxiv-v3.tar.gz`` and confirmed the
author identity in ``docs/arxiv-paper/main.tex``, but on the dev
machine where it ran no LaTeX engine was installed, so it could not
produce a local ``main.pdf`` preview. The ``honest_verdict`` it
emitted was ``bundle_ready_for_manual_upload`` -- shippable to arXiv
(the arXiv server compiles the PDF itself) but not previewable
locally before clicking Submit.

The submission deadline is 2026-05-15. As of this run that is ~13
days away, so closing the local-preview gap matters because any
LaTeX error that the arXiv server would surface only after upload
costs an extra round-trip we do not have time to spend casually.

What this experiment closes vs. what it cannot.

This experiment:

1. Verifies the existing tarball at ``results/carnot-arxiv-v3.tar.gz``
   is present and non-empty, and that its contents match the file
   list the bundle is supposed to ship (``main.tex``, ``carnot.bib``,
   seven figure PDFs).
2. Probes the local environment for a usable LaTeX engine in
   priority order ``tectonic`` -> ``pdflatex`` -> ``xelatex`` ->
   ``lualatex``. Tectonic is preferred because it is self-contained
   (downloads packages on first run; no system-wide TeX Live
   install required) and ships in CachyOS' default repos.
3. Attempts a local PDF compile. Tectonic's first run downloads font
   metrics, .tfm files, the bibtex style, and PostScript Type 1
   fonts on demand; that takes roughly 60-180 s. A cached second
   run is < 10 s. We allow a 240 s timeout to cover the cold path.
4. Records the resulting state in a deliverable JSON with an
   ``honest_verdict`` drawn from a closed set so the conductor and
   future operators can act on it without re-discovering state.

This experiment does NOT actually upload to arxiv.org. arXiv's
submission flow is interactive: it requires a logged-in browser
session against the user's arXiv account, plus the
license/category/endorsement flow with a final human-clicked
Submit button. The conductor has no credentials and cannot drive
that flow safely. Instead the script writes the exact remaining
manual steps so the next session (human or AI) can complete the
upload without re-discovering them.

Honest verdict semantics. The set of allowed verdicts is fixed by
the milestone .88 task description:

    submitted
        The script actually uploaded the bundle to arXiv and got a
        submission ID back. This script never produces this verdict;
        it is reserved for a future interactive flow.
    pdf_compiled_upload_pending
        Tarball verified, PDF compiled locally, ready for human
        upload. This is the happy path for this experiment.
    bundle_ready_tectonic_install_failed
        Tarball verified but no usable LaTeX engine on the host.
        arXiv will compile server-side so this is still shippable;
        it just blocks local preview.
    all_blocked_manual_only
        Something more fundamental broke (tarball missing, bundle
        contents incomplete) and a human must intervene before any
        upload can proceed.

Why this script does not extend ExperimentTemplate. The template
exists to remove cold-start boilerplate for *inference* experiments
(GPU pre-warm, batched inference, checkpoint resume). This
experiment is a pure file-system + subprocess workflow with no
inference and no GPU; pulling in the template would add more
machinery (EnvPropagationGuard, atexit GPU teardown) than the
script needs. We mirror the lightweight standalone shape of
``experiment_1116_arxiv_submission.py`` instead.
"""

from __future__ import annotations

import datetime as _dt
import json
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
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1127_arxiv_final_submission.json"
PDF_PATH = ARXIV_DIR / "main.pdf"

EXPECTED_FIGURES = [f"figures/fig{i}.pdf" for i in range(1, 8)]
EXPECTED_TEX_FILES = ["main.tex", "carnot.bib"]
EXPECTED_BUNDLE_MEMBERS = EXPECTED_TEX_FILES + EXPECTED_FIGURES

LATEX_ENGINE_PRIORITY: tuple[str, ...] = ("tectonic", "pdflatex", "xelatex", "lualatex")
TECTONIC_TIMEOUT_S = 240

ALLOWED_VERDICTS: frozenset[str] = frozenset(
    {
        "submitted",
        "pdf_compiled_upload_pending",
        "bundle_ready_tectonic_install_failed",
        "all_blocked_manual_only",
    }
)


def verify_tarball(path: Path) -> tuple[bool, list[str]]:
    """Confirm the arXiv bundle tarball exists and contains every expected file.

    Returns ``(verified_ok, missing_members)``. We list missing
    members rather than returning a single boolean because the next
    operator needs to know exactly which file is absent to fix it,
    not just that "the bundle is broken". Empty ``missing_members``
    with ``verified_ok=True`` means every expected file is present.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, list(EXPECTED_BUNDLE_MEMBERS)
    try:
        with tarfile.open(path, "r:gz") as tf:
            members = set(tf.getnames())
    except (tarfile.ReadError, OSError):
        return False, list(EXPECTED_BUNDLE_MEMBERS)
    missing = [m for m in EXPECTED_BUNDLE_MEMBERS if m not in members]
    return (not missing), missing


def find_latex_engine() -> str:
    """Probe ``$PATH`` for the first usable LaTeX engine.

    Priority order is ``tectonic`` first (self-contained, no system-
    wide TeX install needed) then the system-TeX-Live engines
    pdflatex / xelatex / lualatex. Returns the engine name or the
    literal string ``"none"`` when nothing is installed -- the
    deliverable JSON records that string verbatim so the verdict
    classifier can branch on it.
    """
    for engine in LATEX_ENGINE_PRIORITY:
        if shutil.which(engine):
            return engine
    return "none"


def attempt_pdf_compile(engine: str) -> tuple[bool, str | None, str]:
    """Compile ``main.tex`` -> ``main.pdf`` with the chosen engine.

    Returns ``(compiled_ok, pdf_path_or_None, last_log_tail)``.
    ``last_log_tail`` is the trailing ~500 chars of stderr+stdout
    from the engine, suitable for embedding in the artifact when
    diagnosing a failed compile.

    The script intentionally does NOT raise on engine failure. The
    bundle is shippable to arXiv regardless of local-compile state
    (arXiv's server compiles its own PDF), so a failed local compile
    is informational only -- it just downgrades the verdict from
    ``pdf_compiled_upload_pending`` to
    ``bundle_ready_tectonic_install_failed``.
    """
    if engine == "none":
        return False, None, ""
    cwd = str(ARXIV_DIR)
    try:
        if engine == "tectonic":
            proc = subprocess.run(
                ["tectonic", "main.tex"],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=TECTONIC_TIMEOUT_S,
            )
        else:
            # pdflatex/xelatex/lualatex need two passes for refs/bibtex.
            proc = subprocess.run(
                [engine, "-interaction=nonstopmode", "main.tex"],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=TECTONIC_TIMEOUT_S,
            )
            if PDF_PATH.exists():
                proc = subprocess.run(
                    [engine, "-interaction=nonstopmode", "main.tex"],
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=TECTONIC_TIMEOUT_S,
                )
        log_tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-500:]
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, None, f"engine_invocation_failed: {exc!r}"
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
        # Prefer a repo-relative path so the deliverable JSON stays portable
        # across machines; fall back to the absolute path when the file is
        # outside the repo root (mostly only happens in tmp-path tests).
        try:
            pdf_path_str = str(PDF_PATH.relative_to(REPO_ROOT))
        except ValueError:
            pdf_path_str = str(PDF_PATH)
        return True, pdf_path_str, log_tail
    return False, None, log_tail


def manual_submission_steps(pdf_compiled: bool, bundle_ok: bool) -> list[str]:
    """Build the exact browser-side checklist a human must perform.

    Steps are ordered as the human will see them in the arXiv
    submission UI: login -> upload -> classify -> license -> verify
    preview -> submit. When a local PDF preview was successfully
    compiled, we add a leading verification step pointing at the
    local file so the operator catches any rendering surprise
    BEFORE clicking Submit. When the bundle itself is broken we
    invert the list to lead with the fixup step.
    """
    if not bundle_ok:
        return [
            "Bundle missing or incomplete -- rerun experiment_1116 to rebuild "
            f"results/{TARBALL_NAME} before attempting upload.",
        ]
    steps: list[str] = []
    if pdf_compiled:
        steps.append(
            f"Open {PDF_PATH.relative_to(REPO_ROOT)} locally and verify all 7 "
            "figures render and the abstract reads correctly. If any figure "
            "shows a missing-image box, halt and re-run the figure-generation "
            "scripts before upload."
        )
    else:
        steps.append(
            "Local PDF compile failed or no LaTeX engine available. arXiv will "
            "compile server-side; verify the server-built preview carefully "
            "before clicking Submit."
        )
    steps.extend(
        [
            "Log in to https://arxiv.org/submit with the project arXiv account.",
            f"Upload results/{TARBALL_NAME} as the source bundle.",
            "Set primary category: cs.LG (Machine Learning).",
            "Set secondary category: cs.NE (Neural and Evolutionary Computing).",
            "Confirm author: Ian Blenke <ian@blenke.com>.",
            "Paste the abstract from docs/arxiv-paper/main.tex (\\begin{abstract} block).",
            "Select license: arXiv non-exclusive license to distribute (CC BY 4.0 acceptable).",
            "Verify the server-built PDF preview before clicking Submit.",
            "Record the returned arXiv submission ID in "
            f"results/{DELIVERABLE_PATH.name} (arxiv_submission_id field) "
            "and flip arxiv_submitted=True for the next-session handoff.",
        ]
    )
    return steps


def classify_verdict(bundle_ok: bool, engine: str, pdf_compiled: bool) -> str:
    """Map the observed state to one of the four allowed honest verdicts.

    The mapping is deterministic and total: any (bundle_ok, engine,
    pdf_compiled) tuple resolves to exactly one verdict. We never
    emit ``submitted`` here because this script does not perform an
    interactive upload -- that verdict is reserved for whatever
    future code actually drives the arXiv portal.
    """
    if not bundle_ok:
        return "all_blocked_manual_only"
    if pdf_compiled:
        return "pdf_compiled_upload_pending"
    return "bundle_ready_tectonic_install_failed"


def build_artifact(
    bundle_ok: bool,
    missing_files: list[str],
    engine: str,
    pdf_compiled: bool,
    pdf_path: str | None,
    compile_log_tail: str,
    tarball_size: int,
) -> dict[str, Any]:
    """Assemble the deliverable JSON for experiment 1127.

    All required schema fields from the milestone .88 task
    description are populated. The verdict is classified in a
    single helper for testability. We pin ``arxiv_submitted=False``
    and ``arxiv_submission_id=None`` because no upload is
    performed; a future session that completes the manual upload
    can edit those two fields in place without re-running anything.
    """
    verdict = classify_verdict(bundle_ok, engine, pdf_compiled)
    assert verdict in ALLOWED_VERDICTS, f"verdict {verdict!r} not in allowed set"
    return {
        "experiment": "1127_arxiv_final_submission",
        "run_date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "arxiv_submission_v1",
        "arxiv_bundle_verified": bundle_ok,
        "arxiv_bundle_path": f"results/{TARBALL_NAME}",
        "arxiv_bundle_size_bytes": tarball_size,
        "missing_bundle_files": missing_files,
        "pdf_compiled": pdf_compiled,
        "pdf_path": pdf_path,
        "latex_engine_used": engine,
        "compile_log_tail": compile_log_tail,
        "arxiv_submitted": False,
        "arxiv_submission_id": None,
        "submission_url": "https://arxiv.org/submit",
        "submission_deadline": "2026-05-15",
        "primary_category": "cs.LG",
        "secondary_category": "cs.NE",
        "author_name": "Ian Blenke",
        "author_email": "ian@blenke.com",
        "manual_steps_remaining": manual_submission_steps(pdf_compiled, bundle_ok),
        "honest_verdict": verdict,
    }


def main() -> int:
    """Run the full experiment 1127 submission-prep flow."""
    bundle_ok, missing = verify_tarball(TARBALL_PATH)
    tarball_size = TARBALL_PATH.stat().st_size if TARBALL_PATH.exists() else 0
    engine = find_latex_engine()
    pdf_compiled, pdf_path, log_tail = attempt_pdf_compile(engine)

    artifact = build_artifact(
        bundle_ok=bundle_ok,
        missing_files=missing,
        engine=engine,
        pdf_compiled=pdf_compiled,
        pdf_path=pdf_path,
        compile_log_tail=log_tail,
        tarball_size=tarball_size,
    )

    DELIVERABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1127] verdict={artifact['honest_verdict']} engine={engine} "
        f"bundle={bundle_ok} pdf={pdf_compiled} size={tarball_size}"
    )
    print(f"[exp1127] deliverable -> {DELIVERABLE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
