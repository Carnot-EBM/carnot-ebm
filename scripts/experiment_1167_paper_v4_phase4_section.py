#!/usr/bin/env python3
"""Experiment 1167: verify the Phase 4 arXiv Section 7 revision.

Spec: REQ-PUBLISH-006, SCENARIO-PUBLISH-006
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
BUNDLE_REL = Path("results/carnot-arxiv-v5.tar.gz")
DELIVERABLE_REL = Path("results/experiment_1167_paper_v4_phase4_section.json")
EXP1165_REL = Path("results/experiment_1165_phase4_active_inference_pilot_v1.json")
EXP1166_REL = Path("results/experiment_1166_arc_agi3_leaderboard_themesis_outreach.json")

SECTION_TITLE = "\\section{Phase 4: Carnot as Active Inference (Empirical Comparison)}"
NEXT_SECTION_TITLE = "\\section{Decentralization \\& Deployment Sovereignty}"
SUBSECTION_MARKERS = (
    "\\subsection{Theoretical equivalence",
    "\\subsection{Phase 4 pilot results",
    "\\subsection{ARC-AGI-3 leaderboard context",
    "\\subsection{Gap analysis and future work",
)
REQUIRED_ARTIFACT_FIELDS = {
    "section7_expanded",
    "n_subsections_added",
    "phase4_results_in_paper",
    "leaderboard_comparison_in_paper",
    "theoretical_equivalence_stated",
    "pdf_recompiled",
    "pdf_size_kb",
    "bundle_path",
    "paper_ready_for_arxiv_hold_lift",
    "honest_verdict",
}

CommandRunner = Callable[[Sequence[str], Path, int], Any]


def _run_command(
    command: Sequence[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:  # pragma: no cover
    """Run a command and return captured text output for diagnostics."""
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


def section7_text(tex_text: str) -> str:
    """Return the revised Section 7 text used for integration checks."""
    start = tex_text.find(SECTION_TITLE)
    if start < 0:
        start = tex_text.find("\\section{Related Work}")
    start = 0 if start < 0 else start
    end = tex_text.find(NEXT_SECTION_TITLE, start + 1)
    end = len(tex_text) if end < 0 else end
    return tex_text[start:end]


def _metric_token(value: object) -> str:
    """Format source-artifact numbers the way they appear in the paper."""
    return f"{float(value):.6f}"


def detect_phase4_section(
    tex_text: str,
    exp1165: dict[str, Any],
    exp1166: dict[str, Any],
) -> dict[str, bool | int]:
    """Detect the required Phase 4 Section 7 paper integrations."""
    section = section7_text(tex_text)
    compact = "".join(section.split())
    lower = section.lower()
    action_count_ratio = _metric_token(exp1165["action_count_ratio"])
    seed_iq_score = f"{float(exp1166['seed_iq_score']):.2f}"

    n_subsections = sum(1 for marker in SUBSECTION_MARKERS if marker in section)
    theoretical = "F(z)=\\sum_kw_kE_k(z)" in compact and (
        "variational-free-energy" in lower or "variational free energy" in lower
    )
    phase4_results = action_count_ratio in section and (
        "solved_rate" in section or "solved\\_rate" in section
    )
    leaderboard = (
        "Seed IQ" in section
        and "ARC-AGI-3" in section
        and (seed_iq_score in section or "1.00" in section)
        and ("frontier LLM" in section or "frontier autoregressive" in lower)
    )

    return {
        "section7_expanded": SECTION_TITLE in section and n_subsections >= 4,
        "n_subsections_added": n_subsections,
        "phase4_results_in_paper": phase4_results,
        "leaderboard_comparison_in_paper": leaderboard,
        "theoretical_equivalence_stated": theoretical,
    }


def compile_pdf(
    arxiv_dir: Path,
    command_runner: CommandRunner = _run_command,
    timeout: int = 180,
) -> bool:
    """Compile the paper PDF with Tectonic."""
    result = command_runner(["tectonic", "main.tex"], arxiv_dir, timeout)
    if result.returncode != 0:
        raise RuntimeError(f"tectonic failed: {result.stderr or result.stdout}")
    return True


def verify_pdf(pdf_path: Path, minimum_size_bytes: int) -> float:
    """Return the PDF size in KiB after checking it did not shrink."""
    if not pdf_path.exists():
        raise RuntimeError(f"missing PDF: {pdf_path}")
    size_bytes = pdf_path.stat().st_size
    if size_bytes < minimum_size_bytes:
        raise RuntimeError(f"PDF smaller than previous build: {size_bytes} < {minimum_size_bytes}")
    return round(size_bytes / 1024.0, 2)


def repack_bundle(
    project_root: Path,
    bundle_path: Path,
    command_runner: CommandRunner = _run_command,
    timeout: int = 180,
) -> bool:
    """Create and verify the arXiv v5 source tarball."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    result = command_runner(
        ["tar", "-czf", str(bundle_path.relative_to(project_root)), "docs/arxiv-paper/"],
        project_root,
        timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tar failed: {result.stderr or result.stdout}")
    return bundle_path.exists() and bundle_path.stat().st_size > 0


def classify_verdict(
    section_ready: bool,
    pdf_recompiled: bool,
    bundle_verified: bool,
) -> str:
    """Map paper, PDF, and bundle state to the required closed verdict set."""
    if not section_ready:
        return "partial_expansion_only"
    if not pdf_recompiled or not bundle_verified:
        return "section_expanded_pdf_recompile_failed"
    return "paper_v4_phase4_complete_arxiv_ready"


def build_artifact(
    flags: dict[str, bool | int],
    pdf_recompiled: bool,
    pdf_size_kb: float,
    bundle_verified: bool,
) -> dict[str, Any]:
    """Assemble the Exp 1167 deliverable JSON."""
    section_ready = (
        bool(flags["section7_expanded"])
        and int(flags["n_subsections_added"]) >= 4
        and bool(flags["phase4_results_in_paper"])
        and bool(flags["leaderboard_comparison_in_paper"])
        and bool(flags["theoretical_equivalence_stated"])
    )
    paper_ready = section_ready and pdf_recompiled and bundle_verified
    artifact = {
        "schema": "carnot.paper_v4_phase4_section.v1",
        "experiment": 1167,
        "run_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "section7_expanded": bool(flags["section7_expanded"]),
        "n_subsections_added": int(flags["n_subsections_added"]),
        "phase4_results_in_paper": bool(flags["phase4_results_in_paper"]),
        "leaderboard_comparison_in_paper": bool(flags["leaderboard_comparison_in_paper"]),
        "theoretical_equivalence_stated": bool(flags["theoretical_equivalence_stated"]),
        "pdf_recompiled": pdf_recompiled,
        "pdf_size_kb": pdf_size_kb,
        "bundle_path": str(BUNDLE_REL),
        "bundle_verified": bundle_verified,
        "paper_ready_for_arxiv_hold_lift": paper_ready,
        "honest_verdict": classify_verdict(section_ready, pdf_recompiled, bundle_verified),
    }
    assert REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    return artifact


def run_experiment(
    project_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    command_runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Verify Section 7, rebuild the PDF, repack sources, and write JSON."""
    output_path = project_root / DELIVERABLE_REL if output_path is None else output_path
    main_tex_path = project_root / MAIN_TEX_REL
    pdf_path = project_root / PDF_REL
    previous_pdf_size = pdf_path.stat().st_size if pdf_path.exists() else 0

    tex_text = main_tex_path.read_text(encoding="utf-8")
    exp1165 = _load_json(project_root / EXP1165_REL)
    exp1166 = _load_json(project_root / EXP1166_REL)
    flags = detect_phase4_section(tex_text, exp1165, exp1166)
    pdf_recompiled = compile_pdf(project_root / "docs" / "arxiv-paper", command_runner)
    pdf_size_kb = verify_pdf(pdf_path, previous_pdf_size)
    bundle_verified = repack_bundle(project_root, project_root / BUNDLE_REL, command_runner)
    artifact = build_artifact(flags, pdf_recompiled, pdf_size_kb, bundle_verified)
    _write_json(output_path, artifact)
    print(
        "[exp1167] "
        f"verdict={artifact['honest_verdict']} "
        f"ready={artifact['paper_ready_for_arxiv_hold_lift']} "
        f"pdf={artifact['pdf_size_kb']}KiB "
        f"output={output_path}"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for the conductor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    run_experiment(project_root=args.project_root, output_path=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
