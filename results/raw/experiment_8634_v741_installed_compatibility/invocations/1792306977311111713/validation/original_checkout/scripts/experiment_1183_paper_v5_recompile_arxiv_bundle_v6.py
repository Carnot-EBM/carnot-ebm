#!/usr/bin/env python3
"""Experiment 1183: paper-v5 recompile and arXiv bundle-v6 gate record.

Spec traces: REQ-PUBLISH-010, SCENARIO-PUBLISH-009, SCENARIO-PUBLISH-010.

This runner is intentionally gate-first.  It writes the required artifact even
when prior publication-integrity gates are missing, but it does not run audits,
compile LaTeX, or build a bundle unless Exp 1180 and Exp 1181 both report true
acceptance gates.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_TEX = REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
PAPER_PDF = REPO_ROOT / "docs" / "arxiv-paper" / "main.pdf"
PAPER_BIB = REPO_ROOT / "docs" / "arxiv-paper" / "carnot.bib"
ARXIV_FIGURES_DIR = REPO_ROOT / "docs" / "arxiv-paper" / "figures"
DOCS_FIGURES_DIR = REPO_ROOT / "docs" / "figures"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "experiment_1183_paper_v5_recompile_arxiv_bundle_v6.json"
ARXIV_BUNDLE_PATH = Path(f"/tmp/carnot_arxiv_v6_{datetime.now().strftime('%Y%m%d')}.tar.gz")

FIGURE_AUDIT_SCRIPT = REPO_ROOT / "scripts" / "figure_integrity_audit.py"
CLAIM_AUDIT_SCRIPT = REPO_ROOT / "scripts" / "paper_claim_audit.py"

BANNED_PATTERNS = (r"11680", r"76130", r"76,130", r"15\.6x")
REQUIRED_SECTION_PATTERNS = {
    "Abstract": re.compile(r"\\begin\{abstract\}", re.IGNORECASE),
    "Introduction": re.compile(r"\\section\{Introduction\}", re.IGNORECASE),
    "Section 1": re.compile(r"\\section\{Introduction\}", re.IGNORECASE),
    "Section 2": re.compile(r"\\section\{Carnot Architectural Framework\}", re.IGNORECASE),
    "Section 3": re.compile(r"\\section\{Theoretical Bounds", re.IGNORECASE),
    "Section 4": re.compile(r"\\section\{Hardware Acceleration", re.IGNORECASE),
    "Section 5": re.compile(r"\\section\{Empirical Realities", re.IGNORECASE),
    "Section 6": re.compile(r"\\section\{Phase 4:", re.IGNORECASE),
    "Section 7": re.compile(r"\\section\{Decentralization", re.IGNORECASE),
    "Conclusion": re.compile(r"\\section\{Conclusion", re.IGNORECASE),
    "References": re.compile(
        r"\\bibliography\{|\\begin\{thebibliography\}|\\section\*?\{References\}",
        re.IGNORECASE,
    ),
}

EXP1180_PATTERNS = ("experiment_1180*.json",)
EXP1181_PATTERNS = (
    "experiment_1181_paper_v5_high_issues_6_10.json",
    "experiment_1181*.json",
)
EXP1182_PATTERNS = (
    "experiment_1182_paper_v5_medium_low_issues_11_18.json",
    "experiment_1182*.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "pdf_compiles_without_error",
    "arxiv_bundle_v6_ready",
    "arxiv_bundle_path",
    "figure_audit_untraced_constants",
    "claim_audit_n_mismatches",
    "known_remaining_issues",
    "fabricated_constants_remaining",
    "paper_word_count",
    "4_test_full_pass",
    "honest_verdict",
)
READY_VERDICT = "arxiv_bundle_v6_ready"
COMPILATION_FAILED_VERDICT = "compilation_failed"
AUDIT_FAILURE_VERDICT = "audit_failures_remain"
ALLOWED_VERDICTS = {READY_VERDICT, COMPILATION_FAILED_VERDICT, AUDIT_FAILURE_VERDICT}


@dataclass(frozen=True)
class AuditRun:
    """Normalized result from an audit subprocess."""

    returncode: int
    stdout: str
    stderr: str
    report: dict[str, Any]
    passed: bool


def count_banned_strings(tex: str) -> int:
    """Count the exact banned strings required by the final grep gate."""
    return sum(len(re.findall(pattern, tex)) for pattern in BANNED_PATTERNS)


def check_sections_present(tex: str) -> list[str]:
    """Return missing major paper sections used as the PDF readability proxy."""
    return [
        section for section, pattern in REQUIRED_SECTION_PATTERNS.items() if not pattern.search(tex)
    ]


def parse_json_object(output: str) -> dict[str, Any]:
    """Return the first JSON object embedded in command output, or an empty dict."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", output):
        try:
            value, _ = decoder.raw_decode(output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON artifact; parse errors become a small error payload."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_parse_error": str(exc)}


def _find_first_artifact(results_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    """Return the first artifact matching any accepted filename pattern."""
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(results_dir.glob(pattern)):
            if path not in seen and path.is_file():
                return path
            seen.add(path)
    return None


def _count_complete(value: Any, target: int) -> bool:
    """Return True when a scalar or fraction-like string reports target/target."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return int(value) == target
    if isinstance(value, str):
        compact = value.strip().lower()
        return compact in {str(target), f"{target}/{target}", "true", "yes", "complete"}
    return False


def _has_failure_words(payload: dict[str, Any]) -> bool:
    """Return True when verdict/status text clearly marks a failed prior gate."""
    text = " ".join(
        str(payload.get(key, "")).lower() for key in ("honest_verdict", "status", "verdict")
    )
    return any(word in text for word in ("blocked", "partial", "failed", "failure"))


def _artifact_gate_true(payload: dict[str, Any], experiment: str) -> bool:
    """Interpret the prior experiment's acceptance gate conservatively."""
    if payload.get("_parse_error") or _has_failure_words(payload):
        return False

    if payload.get("acceptance_gate") is True or payload.get("acceptance_gate_passed") is True:
        return True

    verdict = str(payload.get("honest_verdict", "")).lower()
    if experiment == "exp1180":
        return bool(
            "all_5_critical_resolved" in verdict
            or _count_complete(payload.get("critical_issues_fixed"), 5)
            or _count_complete(payload.get("phase_1_critical_fixes_landed"), 5)
            or payload.get("4_test_passes_critical") is True
        )
    if experiment == "exp1181":
        return bool(
            "all_5_high_resolved" in verdict
            or _count_complete(payload.get("high_severity_fixed"), 5)
            or payload.get("4_test_passes_high") is True
        )
    return False


def check_prerequisite_gates(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Check that Exp 1180 and Exp 1181 acceptance gates are present and true."""
    required: dict[str, dict[str, Any]] = {}
    for experiment, patterns in {
        "exp1180": EXP1180_PATTERNS,
        "exp1181": EXP1181_PATTERNS,
    }.items():
        artifact_path = _find_first_artifact(results_dir, patterns)
        if artifact_path is None:
            required[experiment] = {
                "path": "",
                "gate_true": False,
                "reason": "missing_artifact",
            }
            continue

        payload = _load_json(artifact_path)
        gate_true = _artifact_gate_true(payload, experiment)
        required[experiment] = {
            "path": str(artifact_path),
            "gate_true": gate_true,
            "reason": "ok" if gate_true else "gate_not_true",
            "honest_verdict": payload.get("honest_verdict", "unknown"),
            "status": payload.get("status", "unknown"),
        }

    return {
        "all_required_gates_true": all(item["gate_true"] for item in required.values()),
        "required": required,
    }


def known_remaining_issues(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Return medium/low issue notes from exp1182 if those fixes are not complete."""
    path = _find_first_artifact(results_dir, EXP1182_PATTERNS)
    if path is None:
        return ["exp1182 medium/low issues are not merged into the v6 gate record"]

    payload = _load_json(path)
    if str(payload.get("honest_verdict", "")).lower() == "all_8_medium_low_resolved":
        return []

    issue_names = {
        "issue_11_thinkprm_citation_fixed": "ISSUE-11 ThinkPRM citation",
        "issue_12_holdout_n_stated": "ISSUE-12 FoVer holdout n disclosure",
        "issue_13_nrgpt_disclosure_added": "ISSUE-13 NRGPT non-monotonicity disclosure",
        "issue_14_soskan_auroc_reconciled": "ISSUE-14 SOS-KAN AUROC reconciliation",
        "issue_15_fig2_caveat_added": "ISSUE-15 Figure 2 binormal caveat",
        "issue_16_bibliography_ok": "ISSUE-16 bibliography audit",
        "issue_17_k15_caption_tightened": "ISSUE-17 k=15 caption clarity",
        "issue_18_hardware_scope_added": "ISSUE-18 hardware scope caveat",
    }
    remaining = [label for key, label in issue_names.items() if payload.get(key) is not True]
    return remaining or ["exp1182 artifact did not report all medium/low issues resolved"]


def _run_script(script: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    """Run a Python script from the repository root."""
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def run_figure_integrity_audit() -> AuditRun:
    """Run the figure integrity audit and normalize its result."""
    if not FIGURE_AUDIT_SCRIPT.exists():
        report = {
            "available": False,
            "reason": "scripts/figure_integrity_audit.py not found",
            "untraced_constants": 0,
        }
        return AuditRun(127, "", report["reason"], report, False)

    try:
        result = _run_script(FIGURE_AUDIT_SCRIPT, timeout_s=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report = {"error": str(exc), "untraced_constants": 0}
        return AuditRun(124, "", str(exc), report, False)

    report = parse_json_object(result.stdout) or {
        "raw_stdout_tail": result.stdout[-1000:],
        "raw_stderr_tail": result.stderr[-1000:],
    }
    untraced = _numeric_count(
        report.get("untraced_constants", report.get("n_untraced_constants", 0))
    )
    passed = result.returncode == 0 and untraced == 0 and report.get("passes", True) is not False
    return AuditRun(result.returncode, result.stdout, result.stderr, report, passed)


def run_paper_claim_audit() -> AuditRun:
    """Run the paper numerical-claim audit and normalize its result."""
    if not CLAIM_AUDIT_SCRIPT.exists():
        report = {
            "available": False,
            "reason": "scripts/paper_claim_audit.py not found",
            "n_mismatches": 0,
            "passes": False,
        }
        return AuditRun(127, "", report["reason"], report, False)

    try:
        result = _run_script(CLAIM_AUDIT_SCRIPT, timeout_s=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report = {"error": str(exc), "n_mismatches": 0, "passes": False}
        return AuditRun(124, "", str(exc), report, False)

    report = parse_json_object(result.stdout) or {
        "raw_stdout_tail": result.stdout[-1000:],
        "raw_stderr_tail": result.stderr[-1000:],
        "passes": False,
    }
    passed = result.returncode == 0 and report.get("passes", True) is not False
    return AuditRun(result.returncode, result.stdout, result.stderr, report, passed)


def _numeric_count(value: Any) -> int:
    """Convert an audit count field into an integer."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, list | tuple | set):
        return len(value)
    return 0


def try_compile_latex(
    paper_tex: Path = PAPER_TEX,
    paper_pdf: Path = PAPER_PDF,
) -> dict[str, Any]:
    """Attempt the preferred pdflatex + bibtex pipeline."""
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None:
        return {
            "compiled": False,
            "pdflatex_available": False,
            "bibtex_available": bibtex is not None,
            "output_pdf_exists": paper_pdf.exists(),
            "log_tail": "pdflatex not available; source-bundle fallback required",
        }
    if bibtex is None:
        return {
            "compiled": False,
            "pdflatex_available": True,
            "bibtex_available": False,
            "output_pdf_exists": paper_pdf.exists(),
            "log_tail": "bibtex not available; source-bundle fallback required",
        }

    commands = (
        [pdflatex, "-interaction=nonstopmode", paper_tex.name],
        [bibtex, paper_tex.stem],
        [pdflatex, "-interaction=nonstopmode", paper_tex.name],
        [pdflatex, "-interaction=nonstopmode", paper_tex.name],
    )
    logs: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=str(paper_tex.parent),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "compiled": False,
                "pdflatex_available": True,
                "bibtex_available": True,
                "output_pdf_exists": paper_pdf.exists(),
                "log_tail": str(exc),
            }
        logs.append((result.stdout + "\n" + result.stderr)[-2000:])
        if result.returncode != 0:
            return {
                "compiled": False,
                "pdflatex_available": True,
                "bibtex_available": True,
                "output_pdf_exists": paper_pdf.exists(),
                "log_tail": "\n".join(logs)[-3000:],
            }

    return {
        "compiled": paper_pdf.exists(),
        "pdflatex_available": True,
        "bibtex_available": True,
        "output_pdf_exists": paper_pdf.exists(),
        "log_tail": "\n".join(logs)[-3000:],
    }


def build_arxiv_bundle(
    bundle_path: Path = ARXIV_BUNDLE_PATH,
    paper_tex: Path = PAPER_TEX,
    paper_bib: Path = PAPER_BIB,
    arxiv_figures_dir: Path = ARXIV_FIGURES_DIR,
    docs_figures_dir: Path = DOCS_FIGURES_DIR,
) -> str:
    """Create the arXiv source tarball and return its path."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz") as tar:
        if paper_tex.exists():
            tar.add(paper_tex, arcname="docs/arxiv-paper/main.tex")
        if paper_bib.exists():
            tar.add(paper_bib, arcname="docs/arxiv-paper/carnot.bib")
        for root, arc_root in (
            (arxiv_figures_dir, "docs/arxiv-paper/figures"),
            (docs_figures_dir, "docs/figures"),
        ):
            if not root.exists():
                continue
            for figure in sorted(root.iterdir()):
                if figure.suffix.lower() in {".pdf", ".png"}:
                    tar.add(figure, arcname=f"{arc_root}/{figure.name}")
    return str(bundle_path)


def _artifact_summary(report: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Keep the artifact readable by preserving only high-signal audit fields."""
    return {key: report[key] for key in keys if key in report}


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write a stable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    paper_tex: Path = PAPER_TEX,
    paper_bib: Path = PAPER_BIB,
    paper_pdf: Path = PAPER_PDF,
    results_dir: Path = RESULTS_DIR,
    output_path: Path = OUTPUT_PATH,
    bundle_path: Path = ARXIV_BUNDLE_PATH,
) -> dict[str, Any]:
    """Run exp1183 and write the required gate-record artifact."""
    t0 = time.monotonic()
    tex = paper_tex.read_text(encoding="utf-8") if paper_tex.exists() else ""
    fabricated_remaining = count_banned_strings(tex)
    missing_sections = check_sections_present(tex)
    prereqs = check_prerequisite_gates(results_dir)
    remaining_issues = known_remaining_issues(results_dir)

    artifact: dict[str, Any] = {
        "experiment": 1183,
        "title": "Paper v5 recompile and arXiv bundle v6",
        "run_date": datetime.now(UTC).isoformat(),
        "schema": "experiment_result_v1",
        "pdf_compiles_without_error": False,
        "arxiv_bundle_v6_ready": False,
        "arxiv_bundle_path": str(bundle_path),
        "figure_audit_untraced_constants": 0,
        "claim_audit_n_mismatches": 0,
        "known_remaining_issues": remaining_issues,
        "fabricated_constants_remaining": fabricated_remaining,
        "paper_word_count": len(tex.split()),
        "4_test_full_pass": False,
        "missing_sections": missing_sections,
        "prerequisites_met": prereqs["all_required_gates_true"],
        "prereq_status": prereqs,
        "honest_verdict": AUDIT_FAILURE_VERDICT,
        "status": "blocked",
        "duration_s": 0.0,
    }

    if not prereqs["all_required_gates_true"]:
        missing = [
            name for name, info in prereqs["required"].items() if info.get("gate_true") is not True
        ]
        artifact["blocked_reason"] = "prior acceptance gates incomplete: " + ", ".join(missing)
        artifact["duration_s"] = round(time.monotonic() - t0, 3)
        _write_artifact(output_path, artifact)
        return artifact

    figure_audit = run_figure_integrity_audit()
    claim_audit = run_paper_claim_audit()
    compile_report = try_compile_latex(paper_tex=paper_tex, paper_pdf=paper_pdf)
    built_bundle_path = build_arxiv_bundle(
        bundle_path=bundle_path,
        paper_tex=paper_tex,
        paper_bib=paper_bib,
    )
    bundle_exists = Path(built_bundle_path).exists()

    figure_untraced = _numeric_count(
        figure_audit.report.get(
            "untraced_constants",
            figure_audit.report.get("n_untraced_constants", 0),
        )
    )
    claim_mismatches = _numeric_count(claim_audit.report.get("n_mismatches", 0))
    audits_pass = figure_audit.passed and claim_audit.passed
    compile_failed_after_attempt = (
        compile_report.get("pdflatex_available") is True
        and compile_report.get("compiled") is not True
    )

    arxiv_ready = bool(
        bundle_exists
        and paper_tex.exists()
        and paper_bib.exists()
        and fabricated_remaining == 0
        and not missing_sections
        and not compile_failed_after_attempt
    )
    if not arxiv_ready:
        verdict = COMPILATION_FAILED_VERDICT
    elif not audits_pass:
        verdict = AUDIT_FAILURE_VERDICT
    else:
        verdict = READY_VERDICT

    artifact.update(
        {
            "pdf_compiles_without_error": compile_report.get("compiled") is True,
            "arxiv_bundle_v6_ready": arxiv_ready,
            "arxiv_bundle_path": built_bundle_path,
            "figure_audit_untraced_constants": figure_untraced,
            "claim_audit_n_mismatches": claim_mismatches,
            "4_test_full_pass": audits_pass and fabricated_remaining == 0,
            "latex_compile_summary": compile_report,
            "figure_audit_summary": _artifact_summary(
                figure_audit.report,
                (
                    "available",
                    "passes",
                    "untraced_constants",
                    "n_untraced_constants",
                    "reason",
                    "error",
                ),
            ),
            "paper_claim_audit_summary": _artifact_summary(
                claim_audit.report,
                (
                    "passes",
                    "n_claims_total",
                    "n_claims_with_artifact_citation",
                    "n_claims_verified",
                    "n_mismatches",
                    "citation_ratio",
                    "reason",
                    "error",
                ),
            ),
            "honest_verdict": verdict,
            "status": "success" if verdict == READY_VERDICT else "partial",
            "duration_s": round(time.monotonic() - t0, 3),
        }
    )
    assert artifact["honest_verdict"] in ALLOWED_VERDICTS
    _write_artifact(output_path, artifact)
    return artifact


def main() -> None:
    """CLI entrypoint."""
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if (
        not artifact.get("prerequisites_met")
        or artifact["honest_verdict"] == COMPILATION_FAILED_VERDICT
    ):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
