"""Build the Exp 2944 paper-v6 narrowing-discipline audit artifact.

Spec refs: REQ-REPORT-2944, SCENARIO-REPORT-2944.

The audit is a mechanical guard for CLAUDE.md's Paper-v6 Narrowing Discipline.
It scans only the named paper/doc targets plus the ten latest capstone
artifacts, records every forbidden-phrase hit, and rewrites only autonomous
capstone JSON string content where a prior autonomous task repeated a retracted
claim. Operator-curated documentation is never modified by this module.
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
SCHEMA = "carnot.paper_v6_narrowing_audit.v1"
ARTIFACT = "experiment_2944_paper_v6_narrowing_audit_v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2944_paper_v6_narrowing_audit_v1.json")

POLICY_SOURCE_REL_PATH = Path("CLAUDE.md")
DOC_TARGETS: tuple[Path, ...] = (
    Path("docs/arxiv-paper/main.tex"),
    Path("docs/technical-report.md"),
    Path("docs/technical-report.html"),
    Path("docs/index.html"),
)
SUGGESTED_LINT_SCRIPT_PATH = {
    "path": "scripts/paper_v6_narrowing_lint.py",
    "principle": (
        "Path to a proposed pre-commit hook that would catch future violations. "
        "Just suggest; do not commit the hook."
    ),
}


@dataclass(frozen=True)
class ForbiddenPattern:
    retracted_claim_id: str
    retracted_claim: str
    regex: str
    suggested_fix: str
    replacement: str
    flags: int = re.IGNORECASE

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.regex, self.flags)


FORBIDDEN_PATTERNS: tuple[ForbiddenPattern, ...] = (
    ForbiddenPattern(
        retracted_claim_id="#2",
        retracted_claim="KV260 samples reach Boltzmann thermalization",
        regex=r"\b(?:thermalization|equilibrium samples|Boltzmann-distributed energies)\b",
        suggested_fix='Use "fixed-compute heuristic budget" instead.',
        replacement="fixed-compute heuristic budget",
    ),
    ForbiddenPattern(
        retracted_claim_id="#3",
        retracted_claim="KV260 hardware speedup over CPU at current d",
        regex=(
            r"\b(?:KV260 hardware speedup|FPGA acceleration over CPU|"
            r"Carnot's verifier ensemble runs faster on KV260)\b"
        ),
        suggested_fix=(
            'Replace with "POC functional simulator anchoring future high-N deployment".'
        ),
        replacement="POC functional simulator anchoring future high-N deployment",
    ),
    ForbiddenPattern(
        retracted_claim_id="#6",
        retracted_claim="Phase-4 VFE bounds validate KV260 deployment",
        regex=(
            r"\b(?:(?:exp2550|exp2748|exp2753|exp2766)[^\n]{0,120}"
            r"(?:FPGA|KV260|hardware)[^\n]{0,80}(?:deploy|deployment)|"
            r"(?:FPGA|KV260|hardware)[^\n]{0,120}(?:deploy|deployment)[^\n]{0,120}"
            r"(?:exp2550|exp2748|exp2753|exp2766))\b"
        ),
        suggested_fix=(
            "State that Phase-4 VFE bounds apply only to continuous-sampler "
            "(RTX 3090) deployment, and add the paper firewall paragraph."
        ),
        replacement=(
            "Phase-4 VFE bounds apply only to continuous-sampler "
            "(RTX 3090) deployment"
        ),
    ),
    ForbiddenPattern(
        retracted_claim_id="#7",
        retracted_claim="Extropic Z1 / photonic as future production target",
        regex=(
            r"\b(?:(?:Extropic Z1|photonic)[^\n]{0,100}"
            r"(?:future production target|production target|future target|hardware target)|"
            r"(?:future production target|production target|future target|hardware target)"
            r"[^\n]{0,100}(?:Extropic Z1|photonic))\b"
        ),
        suggested_fix=(
            'Replace with "digital ASICs, spatial FPGAs, or bespoke digital '
            'Ising machines" as the future production target.'
        ),
        replacement="digital ASICs, spatial FPGAs, or bespoke digital Ising machines",
    ),
    ForbiddenPattern(
        retracted_claim_id="#8",
        retracted_claim="Verifier ensemble generalizes universally across modalities",
        regex=(
            r"\b(?:the verifier ensemble generalizes|verifier ensemble generalizes|"
            r"the verifier ensemble works on novel corpora|"
            r"verifier ensemble works on novel corpora)\b"
        ),
        suggested_fix="Scope the claim to the six corpora in cross-corpus matrix v9+.",
        replacement="the verifier ensemble is scoped to the measured six-corpus matrix",
    ),
    ForbiddenPattern(
        retracted_claim_id="#9",
        retracted_claim="Hardware sovereignty via commodity FPGA",
        regex=r"\bhardware sovereignty\b",
        suggested_fix='Replace with "local edge deployability".',
        replacement="local edge deployability",
    ),
    ForbiddenPattern(
        retracted_claim_id="#10",
        retracted_claim="The five-paper_ready streak as scientific maturity",
        regex=(
            r"(?:\.271/\.272/\.273/\.274/\.275\s+paper_ready=true|"
            r"five[- ]paper_ready streak|paper_ready=true[^\n]{0,80}streak|"
            r"streak[^\n]{0,80}paper_ready=true)"
        ),
        suggested_fix="Treat the streak as an infrastructure / MLOps signal only.",
        replacement="CI-loop discipline metric",
    ),
)


def select_capstone_paths(root: Path | str = REPO_ROOT, limit: int = 10) -> list[Path]:
    """REQ-REPORT-2944: select the ten latest capstones by experiment number."""

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
    auto_fix_capstones: bool = True,
) -> dict[str, Any]:
    """REQ-REPORT-2944: scan target files and optionally narrow capstone strings."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    files_to_scan = [*DOC_TARGETS, *select_capstone_paths(root_path)]
    hits = scan_paths(root_path, files_to_scan)
    changed_capstones: set[str] = set()
    if auto_fix_capstones:
        changed_capstones = auto_fix_capstone_hits(root_path, hits)

    operator_hits = [
        hit for hit in hits if Path(str(hit["file"])) in set(DOC_TARGETS)
    ]
    fixed_hits = [
        hit for hit in hits if str(hit["file"]) in changed_capstones
    ]
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _honest_verdict(len(hits), len(operator_hits), len(fixed_hits)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "files_scanned": [path.as_posix() for path in files_to_scan],
        "forbidden_regexes": [
            {
                "retracted_claim_id": pattern.retracted_claim_id,
                "retracted_claim": pattern.retracted_claim,
                "regex": pattern.regex,
                "suggested_fix": pattern.suggested_fix,
            }
            for pattern in FORBIDDEN_PATTERNS
        ],
        "per_file_hits": hits,
        "n_total_hits": len(hits),
        "n_operator_curated_hits_left_for_operator": len(operator_hits),
        "n_autonomous_artifact_hits_auto_fixed": len(fixed_hits),
        "suggested_lint_script_path": dict(SUGGESTED_LINT_SCRIPT_PATH),
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
    """Write the Exp 2944 terminal artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    _write_json(out_path, artifact)
    return out_path


def scan_paths(root: Path, rel_paths: list[Path]) -> list[dict[str, Any]]:
    """Scan files line-by-line and return schema-shaped hit rows."""

    hits: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        abs_path = root / rel_path
        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError:
            continue
        hits.extend(scan_text(text, rel_path))
    return hits


def scan_text(text: str, rel_path: Path) -> list[dict[str, Any]]:
    """Return forbidden-phrase hits for one text buffer."""

    hits: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in FORBIDDEN_PATTERNS:
            for match in pattern.compiled().finditer(line):
                hits.append(
                    {
                        "file": rel_path.as_posix(),
                        "line": line_no,
                        "matched_phrase": match.group(0).strip(),
                        "retracted_claim_id": pattern.retracted_claim_id,
                        "suggested_fix": pattern.suggested_fix,
                    }
                )
    return hits


def auto_fix_capstone_hits(root: Path, hits: list[dict[str, Any]]) -> set[str]:
    """Apply narrowing replacements only to capstone JSON string content."""

    capstone_paths = sorted(
        {
            Path(str(hit["file"]))
            for hit in hits
            if _is_capstone_rel_path(Path(str(hit["file"])))
        }
    )
    changed: set[str] = set()
    for rel_path in capstone_paths:
        abs_path = root / rel_path
        try:
            payload = json.loads(abs_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        narrowed, did_change = _narrow_json_strings(payload)
        if did_change:
            _write_json(abs_path, narrowed)
            changed.add(rel_path.as_posix())
    return changed


def cited_upstream_artifacts(root: Path, scanned_paths: list[Path]) -> list[dict[str, Any]]:
    """List the policy source and scanned inputs with checksums."""

    cited_paths = [POLICY_SOURCE_REL_PATH, *scanned_paths]
    return [
        {
            "path": rel_path.as_posix(),
            "role": "policy_source" if rel_path == POLICY_SOURCE_REL_PATH else "scanned_input",
            "sha256": sha256_file(root / rel_path),
        }
        for rel_path in cited_paths
    ]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _narrow_json_strings(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        narrowed = _apply_replacements(value)
        return narrowed, narrowed != value
    if isinstance(value, list):
        changed = False
        items: list[Any] = []
        for item in value:
            narrowed_item, item_changed = _narrow_json_strings(item)
            items.append(narrowed_item)
            changed = changed or item_changed
        return items, changed
    if isinstance(value, dict):
        changed = False
        narrowed_dict: dict[Any, Any] = {}
        for key, item in value.items():
            narrowed_item, item_changed = _narrow_json_strings(item)
            narrowed_dict[key] = narrowed_item
            changed = changed or item_changed
        return narrowed_dict, changed
    return value, False


def _apply_replacements(text: str) -> str:
    narrowed = text
    for pattern in FORBIDDEN_PATTERNS:
        narrowed = pattern.compiled().sub(pattern.replacement, narrowed)
    return narrowed


def _write_json(path: Path, payload: Mapping[str, Any] | Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capstone_sort_key(path: Path) -> tuple[int, str]:
    return (_experiment_number(path), path.name)


def _experiment_number(path: Path) -> int:
    match = re.search(r"experiment_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _is_capstone_rel_path(path: Path) -> bool:
    return path.parent == Path("results") and "capstone" in path.name and path.suffix == ".json"


def _honest_verdict(total_hits: int, operator_hits: int, fixed_hits: int) -> str:
    if total_hits == 0:
        return "complete: paper_v6_narrowing_audit_no_matches"
    return (
        "complete: paper_v6_narrowing_audit_recorded_matches; "
        f"operator_curated_hits_left={operator_hits}; "
        f"autonomous_capstone_hits_fixed={fixed_hits}"
    )
