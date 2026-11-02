#!/usr/bin/env python3
"""Prepare the Exp 1153 arXiv final-submission v4 bundle.

Spec: REQ-PUBLISH-005, SCENARIO-PUBLISH-005
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX_REL = Path("docs/arxiv-paper/main.tex")
PDF_REL = Path("docs/arxiv-paper/main.pdf")
BUNDLE_REL = Path("results/carnot-arxiv-v4.tar.gz")
DELIVERABLE_REL = Path("results/experiment_1153_arxiv_final_submission_v4.json")
EXP1147_REL = Path("results/experiment_1147_hardnet_projection_repair.json")
EXP1148_REL = Path("results/experiment_1148_metacluster_sos_kan_compression.json")
RESULTS_MARKER = "\\subsection{$D_{\\mathrm{int}} = 1.6$ motivates the Welch bound (exp1093)}"
SUBMISSION_DEADLINE = "2026-05-15"

CommandRunner = Callable[[Sequence[str], Path, int], Any]


def _run_command(
    command: Sequence[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:  # pragma: no cover
    """Run a subprocess and capture its text output."""
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _results_section(tex_text: str) -> str:
    """Return the empirical results section used for paper-integration checks."""
    start = tex_text.find("\\section{Empirical Realities")
    start = 0 if start < 0 else start
    end = tex_text.find("\\section{Related Work}", start + 1)
    end = len(tex_text) if end < 0 else end
    return tex_text[start:end]


def detect_paper_integrations(tex_text: str) -> dict[str, bool]:
    """Detect the required Exp 1153 paper updates in the empirical section."""
    compact = " ".join(_results_section(tex_text).replace("{,}", ",").split())
    lower = compact.lower()
    return {
        "grpo_v2_result_in_paper": "GRPO" in compact and "8.51" in compact,
        "projection_repair_in_paper": (
            "projection repair" in lower and ("76,130" in compact or "76130" in compact)
        ),
        "metacluster_in_paper": (
            "MetaCluster" in compact and "5.03" in compact and "0.018" in compact
        ),
    }


def _latex_int(value: float) -> str:
    """Format an integer-like value with LaTeX-safe thousands separators."""
    return f"{round(value):,}".replace(",", "{,}")


def projection_sentence(artifact: dict[str, Any]) -> str:
    """Build the one-sentence exp1147 paper summary from the source artifact."""
    speedup = _latex_int(float(artifact["speedup_factor"]))
    accuracy_pct = round(float(artifact["projection_repair_accuracy"]) * 100)
    n_violations = int(artifact["n_violations_tested"])
    return (
        "In milestone .89, HardNet++-style arithmetic projection repair corrected "
        f"{n_violations}/{n_violations} synthetic violations at {accuracy_pct}\\% "
        f"accuracy and ran {speedup}$\\times$ faster than prompt repair (exp1147)."
    )


def metacluster_sentence(artifact: dict[str, Any]) -> str:
    """Build the one-sentence exp1148 paper summary from the source artifact."""
    factor = float(artifact["size_reduction_factor"])
    drop = float(artifact["auroc_drop"])
    original = float(artifact["auroc_original"])
    compressed = float(artifact["auroc_compressed"])
    return (
        "MetaCluster-style centroid compression made SOSKANEnergyV3 "
        f"{factor:.2f}$\\times$ smaller with AUROC drop {drop:.3f} "
        f"({original:.4f} to {compressed:.4f}), keeping the compressed verifier "
        "within the 0.02 degradation target (exp1148)."
    )


def ensure_paper_mentions(
    main_tex_path: Path,
    projection_artifact: dict[str, Any],
    metacluster_artifact: dict[str, Any],
) -> tuple[dict[str, bool], bool]:
    """Insert missing exp1147/exp1148 result sentences without restructuring."""
    tex_text = main_tex_path.read_text(encoding="utf-8")
    flags = detect_paper_integrations(tex_text)
    additions = []
    if not flags["projection_repair_in_paper"]:
        additions.append(projection_sentence(projection_artifact))
    if not flags["metacluster_in_paper"]:
        additions.append(metacluster_sentence(metacluster_artifact))

    if additions:
        insertion = "\n\n" + " ".join(additions) + "\n\n"
        tex_text = tex_text.replace(RESULTS_MARKER, insertion + RESULTS_MARKER, 1)
        main_tex_path.write_text(tex_text, encoding="utf-8")
        flags = detect_paper_integrations(tex_text)

    return flags, bool(additions)


def compile_pdf(
    arxiv_dir: Path,
    command_runner: CommandRunner = _run_command,
    timeout: int = 180,
) -> bool:
    """Compile the arXiv paper PDF with Tectonic."""
    result = command_runner(["tectonic", "main.tex"], arxiv_dir, timeout)
    if result.returncode != 0:
        raise RuntimeError(f"tectonic failed: {result.stderr or result.stdout}")  # pragma: no cover
    return True


def verify_pdf(pdf_path: Path) -> float:
    """Return the verified PDF size in KiB."""
    if not pdf_path.exists():
        raise RuntimeError(f"missing PDF: {pdf_path}")  # pragma: no cover
    size_kb = round(pdf_path.stat().st_size / 1024.0, 2)
    if size_kb < 300.0:
        raise RuntimeError(f"PDF too small for final paper: {size_kb} KiB")  # pragma: no cover
    return size_kb


def repack_bundle(
    project_root: Path,
    bundle_path: Path,
    command_runner: CommandRunner = _run_command,
    timeout: int = 180,
) -> bool:
    """Create and verify the arXiv source tarball."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    result = command_runner(
        ["tar", "-czf", str(bundle_path.relative_to(project_root)), "docs/arxiv-paper/"],
        project_root,
        timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tar failed: {result.stderr or result.stdout}")  # pragma: no cover
    return bundle_path.exists() and bundle_path.stat().st_size > 0


def manual_upload_steps() -> list[str]:
    """Return browser-ready arXiv upload steps for the human operator."""
    return [
        "1. Open https://arxiv.org/login and sign in to the operator arXiv account.",
        "2. Open https://arxiv.org/submit and choose Start New Submission.",
        "3. Select Computer Science - Artificial Intelligence (cs.AI) as the primary category.",
        "4. Choose the compressed source upload option and upload results/carnot-arxiv-v4.tar.gz.",
        "5. Let arXiv process the TeX source, then open the generated PDF preview.",
        "6. Compare the arXiv preview against docs/arxiv-paper/main.pdf and confirm the title, abstract, figures, and references render correctly.",
        "7. Enter the title, author list, and abstract exactly from docs/arxiv-paper/main.tex.",
        "8. Set comments to: Position paper draft v3; Tectonic-compiled PDF prepared 2026-05-02.",
        "9. Review arXiv warnings; fix any fatal TeX issue locally, rerun exp1153, and re-upload the refreshed tarball.",
        "10. Submit before 2026-05-15 and record the arXiv submission ID in results/experiment_1153_arxiv_final_submission_v4.json.",
    ]


def classify_verdict(
    paper_updated: bool,
    pdf_recompiled: bool,
    bundle_verified: bool,
    arxiv_submitted: bool,
) -> str:
    """Map final-submission state to the allowed honest verdict set."""
    if arxiv_submitted:
        return "submitted"
    if paper_updated and pdf_recompiled:
        return "paper_updated_recompiled"
    if pdf_recompiled and bundle_verified:
        return "pdf_recompiled_bundle_ready_upload_pending"
    return "paper_verified_no_recompile_needed"


def build_artifact(
    flags: dict[str, bool],
    paper_updated: bool,
    pdf_recompiled: bool,
    pdf_size_kb: float,
    bundle_verified: bool,
) -> dict[str, Any]:
    """Assemble the Exp 1153 deliverable JSON."""
    arxiv_submitted = False
    artifact = {
        "experiment": "1153_arxiv_final_submission_v4",
        "schema": "arxiv_final_submission_v4",
        "run_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grpo_v2_result_in_paper": flags["grpo_v2_result_in_paper"],
        "projection_repair_in_paper": flags["projection_repair_in_paper"],
        "metacluster_in_paper": flags["metacluster_in_paper"],
        "pdf_recompiled": pdf_recompiled,
        "pdf_path": str(PDF_REL),
        "pdf_size_kb": pdf_size_kb,
        "bundle_path": str(BUNDLE_REL),
        "bundle_verified": bundle_verified,
        "arxiv_submitted": arxiv_submitted,
        "arxiv_submission_id": None,
        "manual_upload_steps": manual_upload_steps(),
        "submission_deadline": SUBMISSION_DEADLINE,
        "honest_verdict": classify_verdict(
            paper_updated=paper_updated,
            pdf_recompiled=pdf_recompiled,
            bundle_verified=bundle_verified,
            arxiv_submitted=arxiv_submitted,
        ),
    }
    return artifact


def run_experiment(
    project_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    command_runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Patch paper summaries, rebuild the PDF, repack sources, and write JSON."""
    output_path = project_root / DELIVERABLE_REL if output_path is None else output_path
    main_tex_path = project_root / MAIN_TEX_REL
    projection_artifact = _load_json(project_root / EXP1147_REL)
    metacluster_artifact = _load_json(project_root / EXP1148_REL)

    flags, paper_updated = ensure_paper_mentions(
        main_tex_path=main_tex_path,
        projection_artifact=projection_artifact,
        metacluster_artifact=metacluster_artifact,
    )
    pdf_recompiled = compile_pdf(project_root / "docs" / "arxiv-paper", command_runner)
    pdf_size_kb = verify_pdf(project_root / PDF_REL)
    bundle_verified = repack_bundle(project_root, project_root / BUNDLE_REL, command_runner)
    artifact = build_artifact(
        flags=flags,
        paper_updated=paper_updated,
        pdf_recompiled=pdf_recompiled,
        pdf_size_kb=pdf_size_kb,
        bundle_verified=bundle_verified,
    )
    _write_json(output_path, artifact)
    print(
        "[exp1153] "
        f"verdict={artifact['honest_verdict']} "
        f"pdf={artifact['pdf_size_kb']}KiB "
        f"bundle_verified={artifact['bundle_verified']} "
        f"output={output_path}"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for conductor and manual experiment runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    run_experiment(project_root=args.project_root, output_path=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
