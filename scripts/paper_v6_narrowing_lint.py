#!/usr/bin/env python3
"""Paper-v6 narrowing lint — mechanical G3 prose guard.

This linter scans the current paper targets and tracked `results/paper_v6_*.json`
artifacts for the forbidden Paper-v6 Narrowing Discipline phrasings in
CLAUDE.md. It is a static text lint with one narrow context heuristic: a match
is allowed only when a nearby local window contains explicit retraction or
negation markers such as "retracted", "repinned", "downward", "not", or
"remove". That filter is intentionally shallow; it cannot prove semantic intent
or catch all paraphrases. The operator-facing paper targets should avoid live
uses of retired prose and retired numerical values; CLAUDE.md, lint scripts, and
immutable historical research-log paths are exempt because they document the
rules themselves.

Usage:
    python scripts/paper_v6_narrowing_lint.py
    python scripts/paper_v6_narrowing_lint.py --path docs/technical-report.md

Exit codes:
    0 — clean
    1 — at least one forbidden narrowing phrase or number found
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_TARGETS = (
    Path("docs/arxiv-paper/main.tex"),
    Path("docs/technical-report.md"),
)
RESULTS_ARTIFACT_RE = re.compile(r"^results/paper_v6_[^/]*\.json$")

ALLOWLIST_FILES = {
    "CLAUDE.md",
    "scripts/publication_gate.py",
    "scripts/paper_v6_narrowing_lint.py",
}

EXEMPT_PATH_PATTERNS = [
    re.compile(r"^docs/research-notes/"),
    re.compile(r"^docs/research-log"),
    re.compile(r"^research-log"),
]

RETRACTION_CONTEXT_RE = re.compile(
    r"retract|repin|downward|\bv2 headline\b|\bnot\b|remove|narrowed|"
    r"superseded|deprecated|prior \d|earlier v\d",
    re.IGNORECASE,
)
RETRACTION_CONTEXT_CHARS = 120


@dataclass(frozen=True)
class PatternSpec:
    name: str
    regex: str
    why: str

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.regex, re.IGNORECASE | re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class LintHit:
    path: Path
    line_no: int
    pattern_name: str
    pattern_regex: str
    matched_text: str
    why: str


# Seeded from scripts/publication_gate.py and extended to cover the CLAUDE.md
# Paper-v6 Narrowing Discipline table plus the retracted numerical values.
FORBIDDEN_PATTERNS = [
    PatternSpec("RETRACTED_FOVER_AUROC_09857", r"\b0\.9857\b", "retracted FoVer v2 headline AUROC"),
    PatternSpec("RETRACTED_HIVE_DELTA_00621", r"\+0\.0621\b", "retracted HIVE comparator delta"),
    PatternSpec(
        "RETRACTED_HIVE_PEER_0924",
        r"(?:HIVE\s+peer\s+0\.924|0\.924\s+HIVE\s+peer)",
        "retracted HIVE peer-comparator framing",
    ),
    PatternSpec("HARDWARE_SOVEREIGNTY", r"hardware sovereignty", "use local edge deployability instead"),
    PatternSpec("THERMALIZATION", r"\bthermaliz\w*", "retracted KV260 Boltzmann-thermalization claim"),
    PatternSpec("EQUILIBRIUM_SAMPLES", r"equilibrium samples", "retracted KV260 equilibrium claim"),
    PatternSpec(
        "BOLTZMANN_DISTRIBUTED_ENERGIES",
        r"Boltzmann-distributed energies",
        "retracted KV260 equilibrium claim",
    ),
    PatternSpec(
        "KV260_SPEEDUP_NUMBER",
        r"\b(?:11680|12788|13000)\s*[x×]\b",
        "retracted KV260 speedup figure",
    ),
    PatternSpec(
        "HUMANEVAL_0_TO_36",
        r"0\s*%\s*(?:to|→|->)\s*36\s*%",
        "unsupported 35B HumanEval claim",
    ),
    PatternSpec("REPLACEMENT_GRADE", r"replacement-grade", "prompt-injection replacement was refuted"),
    PatternSpec("KV260_HARDWARE_SPEEDUP", r"KV260 hardware speedup", "KV260 is not a current CPU speedup claim"),
    PatternSpec("FPGA_ACCELERATION_OVER_CPU", r"FPGA acceleration over CPU", "unsupported current FPGA speedup claim"),
    PatternSpec(
        "VERIFIER_RUNS_FASTER_ON_KV260",
        r"Carnot[’']s verifier ensemble runs faster on KV260",
        "unsupported current KV260 verifier speedup claim",
    ),
    PatternSpec(
        "PHASE4_FPGA_FIREWALL",
        r"(?:exp(?:2550|2748|2753|2766).{0,120}\b(?:KV260|FPGA|deployment)\b|"
        r"\b(?:KV260|FPGA|deployment)\b.{0,120}exp(?:2550|2748|2753|2766))",
        "Phase-4 active-inference artifacts do not defend FPGA deployment claims",
    ),
    PatternSpec(
        "ANALOG_PRODUCTION_TARGET",
        r"(?:Extropic\s+Z1|photonic).{0,80}(?:production target|future production|deployment target)",
        "post-pivot Boolean architecture cannot cite analog substrates as production targets",
    ),
    PatternSpec(
        "UNSCOPED_VERIFIER_GENERALIZES",
        r"\bthe verifier ensemble generalizes\b",
        "scope verifier generalization to measured corpora",
    ),
    PatternSpec(
        "UNSCOPED_NOVEL_CORPORA",
        r"\bthe verifier ensemble works on novel corpora\b",
        "scope verifier generalization to measured corpora",
    ),
    PatternSpec(
        "PAPER_READY_STREAK",
        r"(?:five[- ]paper_ready streak|\.271/\.272/\.273/\.274/\.275\s+paper_ready=true)",
        "paper_ready streak is not scientific maturity evidence",
    ),
]

COMPILED_PATTERNS = [(spec, spec.compile()) for spec in FORBIDDEN_PATTERNS]


def _rel_string(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def should_skip(path: Path, root: Path = PROJECT_ROOT) -> bool:
    rel = _rel_string(path, root)
    if rel in ALLOWLIST_FILES:
        return True
    return any(pattern.search(rel) for pattern in EXEMPT_PATH_PATTERNS)


def list_tracked_files(root: Path = PROJECT_ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def discover_targets(root: Path = PROJECT_ROOT) -> list[Path]:
    targets: list[Path] = []
    for rel in PAPER_TARGETS:
        path = root / rel
        if path.exists():
            targets.append(path)
    for rel in list_tracked_files(root):
        if RESULTS_ARTIFACT_RE.match(rel):
            path = root / rel
            if path.exists():
                targets.append(path)
    return targets


def _line_no_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _display_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:160]


def _is_retraction_context(text: str, start: int, end: int) -> bool:
    window = text[
        max(0, start - RETRACTION_CONTEXT_CHARS) : min(len(text), end + RETRACTION_CONTEXT_CHARS)
    ]
    return bool(RETRACTION_CONTEXT_RE.search(window))


def scan_file(path: Path, root: Path = PROJECT_ROOT) -> list[LintHit]:
    if should_skip(path, root):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    hits: list[LintHit] = []
    for spec, pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            if _is_retraction_context(text, match.start(), match.end()):
                continue
            hits.append(
                LintHit(
                    path=path,
                    line_no=_line_no_for_offset(text, match.start()),
                    pattern_name=spec.name,
                    pattern_regex=spec.regex,
                    matched_text=_display_match(match.group(0)),
                    why=spec.why,
                )
            )
    return hits


def scan_paths(paths: list[Path], root: Path = PROJECT_ROOT) -> list[LintHit]:
    hits: list[LintHit] = []
    for path in paths:
        hits.extend(scan_file(path, root=root))
    return hits


def _format_hit(hit: LintHit) -> str:
    return (
        f"{hit.path}:{hit.line_no} [{hit.pattern_name}] "
        f"match={hit.matched_text!r} pattern={hit.pattern_regex!r} why={hit.why}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--path", type=Path, action="append", dest="paths")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    paths = [p.resolve() for p in args.paths] if args.paths else discover_targets(root)
    hits = scan_paths(paths, root=root)

    if not hits:
        if args.verbose:
            print(f"paper_v6_narrowing_lint: clean ({len(paths)} files scanned)")
        return 0

    print(
        f"paper_v6_narrowing_lint: {len(hits)} violation(s) across "
        f"{len({hit.path for hit in hits})} file(s)."
    )
    for hit in hits:
        print(_format_hit(hit))
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
