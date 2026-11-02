"""Build the Exp 2945 Phase-4 VFE firewall verification artifact.

Spec refs: REQ-REPORT-2945, SCENARIO-REPORT-2945.

The verifier is a mechanical citation-context scan. It does not decide whether
Phase-4 is useful in general; it only records places where Phase-4 VFE
citations appear close enough to hardware language that the paper needs the
operator's firewall paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.phase4_vfe_firewall_verification.v1"
ARTIFACT = "experiment_2945_phase4_vfe_firewall_verification_v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2945_phase4_vfe_firewall_verification_v1.json")

LATEX_ROOT_REL_PATH = Path("docs/arxiv-paper")
CAPSTONE_LIMIT = 10
CONTEXT_RADIUS_LINES = 2
HARDWARE_CONTEXT_REGEX = r"\b(?:hardware|FPGA|KV260|Glauber)\b"


@dataclass(frozen=True)
class Phase4Pattern:
    label: str
    regex: str
    flags: int = re.IGNORECASE

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.regex, self.flags)


PHASE4_PATTERNS: tuple[Phase4Pattern, ...] = (
    Phase4Pattern("exp2550", r"\bexp2550\b"),
    Phase4Pattern("exp2748", r"\bexp2748\b"),
    Phase4Pattern("exp2753", r"\bexp2753\b"),
    Phase4Pattern("exp2766", r"\bexp2766\b"),
    Phase4Pattern("Phase-4 active inference", r"\bPhase[- ]4 active inference\b"),
    Phase4Pattern("variational free energy", r"\bvariational free energy\b"),
    Phase4Pattern("FEP factor graph", r"\bFEP factor graph\b"),
    Phase4Pattern("FEP aggregator", r"\bFEP aggregator\b"),
)

FIREWALL_PARAGRAPH_DRAFT = {
    "principle": (
        "Operator-integrable LaTeX snippet stating that Phase-4 VFE bounds apply "
        "only to RTX 3090 continuous-sampler deployment."
    ),
    "latex": (
        "\\paragraph{Phase-4 VFE scope firewall.} "
        "The Phase-4 variational-free-energy bounds reported in exp2550, exp2748, "
        "exp2753, and exp2766 apply only to the RTX 3090 continuous-sampler "
        "deployment used in those upstream artifacts. They do not establish a "
        "partition function, entropy guarantee, or deployment claim for KV260 "
        "synchronous Glauber execution, FPGA implementations, or any other "
        "hardware substrate whose dynamics can collapse into limit cycles. "
        "Accordingly, Phase-4 VFE evidence must not be used to defend KV260 or "
        "FPGA-deployment claims in this paper."
    ),
}


def select_latex_paths(root: Path | str = REPO_ROOT) -> list[Path]:
    """REQ-REPORT-2945: select checked-in paper-v6 LaTeX source files."""

    root_path = Path(root)
    latex_root = root_path / LATEX_ROOT_REL_PATH
    if not latex_root.is_dir():
        return []
    return [
        path.relative_to(root_path)
        for path in sorted(latex_root.rglob("*.tex"))
        if path.is_file()
    ]


def select_capstone_paths(
    root: Path | str = REPO_ROOT,
    *,
    limit: int = CAPSTONE_LIMIT,
) -> list[Path]:
    """REQ-REPORT-2945: select recent capstones deterministically by experiment."""

    root_path = Path(root)
    paths = list((root_path / "results").glob("experiment_*capstone*.json"))
    return [
        path.relative_to(root_path)
        for path in sorted(paths, key=_capstone_sort_key, reverse=True)[:limit]
    ]


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    context_radius: int = CONTEXT_RADIUS_LINES,
) -> dict[str, Any]:
    """REQ-REPORT-2945: scan paper LaTeX and capstones for firewall violations."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    files_to_scan = [*select_latex_paths(root_path), *select_capstone_paths(root_path)]
    violations = scan_paths(root_path, files_to_scan, context_radius=context_radius)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _honest_verdict(len(violations)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "files_scanned": [path.as_posix() for path in files_to_scan],
        "phase_4_regexes": [
            {"label": pattern.label, "regex": pattern.regex}
            for pattern in PHASE4_PATTERNS
        ],
        "hardware_context_regex": HARDWARE_CONTEXT_REGEX,
        "context_radius_lines": context_radius,
        "firewall_violations": violations,
        "n_violations": len(violations),
        "firewall_paragraph_draft": dict(FIREWALL_PARAGRAPH_DRAFT),
        "cited_upstream_artifacts": cited_upstream_artifacts(root_path, files_to_scan),
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the Exp 2945 terminal artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    _write_json(out_path, artifact)
    return out_path


def scan_paths(
    root: Path,
    rel_paths: list[Path],
    *,
    context_radius: int = CONTEXT_RADIUS_LINES,
) -> list[dict[str, Any]]:
    """Scan existing text inputs and return schema-shaped firewall violations."""

    violations: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        abs_path = root / rel_path
        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError:
            continue
        violations.extend(scan_text(text, rel_path, context_radius=context_radius))
    return violations


def scan_text(
    text: str,
    rel_path: Path,
    *,
    context_radius: int = CONTEXT_RADIUS_LINES,
) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-2945: record Phase-4 citations near hardware language."""

    violations: list[dict[str, Any]] = []
    lines = text.splitlines()
    hardware_re = re.compile(HARDWARE_CONTEXT_REGEX, re.IGNORECASE)
    for line_index, line in enumerate(lines):
        for pattern in PHASE4_PATTERNS:
            for match in pattern.compiled().finditer(line):
                snippet = _context_snippet(lines, line_index, context_radius)
                if not hardware_re.search(snippet):
                    continue
                violations.append(
                    {
                        "file": rel_path.as_posix(),
                        "line": line_index + 1,
                        "phase_4_citation": match.group(0).strip(),
                        "hardware_context_snippet": snippet,
                    }
                )
    return violations


def cited_upstream_artifacts(root: Path, scanned_paths: list[Path]) -> list[dict[str, Any]]:
    """List every scanned input with a checksum for provenance."""

    return [
        {
            "path": rel_path.as_posix(),
            "role": "scanned_input",
            "sha256": sha256_file(root / rel_path),
        }
        for rel_path in scanned_paths
    ]


def sha256_file(path: Path) -> str | None:
    """Return a file checksum, or None for absent scan inputs."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_snippet(lines: list[str], line_index: int, radius: int) -> str:
    radius = max(0, int(radius))
    start = max(0, line_index - radius)
    end = min(len(lines), line_index + radius + 1)
    snippet = " ".join(line.strip() for line in lines[start:end])
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) <= 500:
        return snippet
    return snippet[:497].rstrip() + "..."


def _write_json(path: Path, payload: Mapping[str, Any] | Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capstone_sort_key(path: Path) -> tuple[int, str]:
    return (_experiment_number(path), path.name)


def _experiment_number(path: Path) -> int:
    match = re.search(r"experiment_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _honest_verdict(n_violations: int) -> str:
    if n_violations == 0:
        return "complete: phase4_vfe_firewall_no_violations"
    return (
        "complete: phase4_vfe_firewall_violations_found; "
        f"operator_firewall_paragraph_required; n_violations={n_violations}"
    )
