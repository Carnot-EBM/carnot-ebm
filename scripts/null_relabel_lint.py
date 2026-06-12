#!/usr/bin/env python3
"""Null-relabel lint — surface result artifacts a roadmap/doc cites whose own
honest_verdict is a NULL/negative or that carry flagged_adversarial=true.

ORIGIN: 2026-06-12 adversarial review. The project's credibility rests on its
NEGATIVES being trustworthy. The failure mode this catches: a planner/roadmap
cites an artifact as SUPPORTING a forward claim ("the Sudoku RFT-beats-SFT
beachhead, 3/3 seeds") when the artifact's own verdict is
`energy_teacher_no_lift_dteacher+0.0112` (+1.1%, gold control broken) -- i.e.
quietly UPGRADING a null into a positive in prose. This is the sibling of the
Verdict Terminal-Prefix lint and the summarize_artifact reading discipline,
applied at the citation layer.

WHAT IT DOES: scans a roadmap/doc for `results/*.json` references (and bare
`experiment_NNNN` / `expNNNN` tokens that resolve to a results artifact), loads
each artifact's honest_verdict + flagged_adversarial, and reports any that are
NULL/negative or flagged. It does NOT judge the surrounding framing (that is a
human/operator call) -- it SURFACES the citations so the framing can be checked.
A roadmap that cites a null HONESTLY (naming it a null) is fine; one that cites
it as a win is the bug -- this report is what makes that auditable.

USAGE:
  python3 scripts/null_relabel_lint.py [file ...]      # default: research-roadmap-next.yaml
  python3 scripts/null_relabel_lint.py --strict ...    # exit 1 if any null/flagged citation found

Exit code: 0 by default (warn-level surfacing). With --strict, 1 if any found.
Stdlib-only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# Verdict substrings that mean the artifact is NOT a clean positive. Lowercased match.
NEGATIVE_TOKENS = (
    "no_lift", "no_transfer", "no_win", "no_solve", "no_generalization", "no_improvement",
    "no_delta", "not_decision_grade", "_n0", "n0_", "accumulating", "partial", "retired",
    "blocked", "flat", "refuted", "degenerate", "uninformative", "inverted", "insufficient",
    "still_wrong", "still_", "below", "regression", "negative", "plateau", "collapsed",
    "marginal", "unconfirmed", "no_cross_game", "absent", "ceiling_saturated", "sim2real_ceiling",
)

# Bare experiment-id references -> resolve to a results artifact path.
EXP_ID_RE = re.compile(r"\b(?:exp|experiment_?)(\d{3,4})\b", re.IGNORECASE)
RESULTS_PATH_RE = re.compile(r"results/[\w./-]+\.json")


def _load_verdict(path: Path) -> tuple[str, bool] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    verdict = str(d.get("honest_verdict") or d.get("verdict") or "")
    flagged = bool(d.get("flagged_adversarial"))
    return verdict, flagged


def _resolve_exp(num: str) -> Path | None:
    # find a results/experiment_<num>*.json (skip checkpoints)
    hits = sorted(
        p for p in RESULTS.glob(f"experiment_{int(num)}_*.json")
        if "checkpoint" not in p.name
    )
    if hits:
        return hits[0]
    hits = sorted(p for p in RESULTS.glob(f"*{int(num)}*.json") if "checkpoint" not in p.name)
    return hits[0] if hits else None


def lint_file(path: Path) -> list[tuple[int, str, str, str]]:
    """Return [(lineno, artifact_name, verdict_or_FLAGGED, snippet)] for null/flagged citations."""
    findings: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        cands: list[Path] = []
        for m in RESULTS_PATH_RE.finditer(line):
            p = REPO / m.group(0)
            if p.exists():
                cands.append(p)
        for m in EXP_ID_RE.finditer(line):
            p = _resolve_exp(m.group(1))
            if p is not None:
                cands.append(p)
        for p in cands:
            key = f"{lineno}:{p.name}"
            if key in seen:
                continue
            seen.add(key)
            vf = _load_verdict(p)
            if vf is None:
                continue
            verdict, flagged = vf
            vlow = verdict.lower()
            is_neg = any(tok in vlow for tok in NEGATIVE_TOKENS)
            if flagged or is_neg:
                tag = "FLAGGED_ADVERSARIAL" if flagged else verdict[:80]
                findings.append((lineno, p.name, tag, line.strip()[:100]))
    return findings


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        files = ["research-roadmap-next.yaml"]
    total = 0
    for f in files:
        path = REPO / f if not Path(f).is_absolute() else Path(f)
        if not path.exists():
            continue
        findings = lint_file(path)
        if findings:
            print(f"\n{f}: {len(findings)} citation(s) of NULL/FLAGGED artifacts "
                  f"-- verify they are NOT framed as supporting a forward claim:")
            for lineno, name, tag, snippet in findings:
                print(f"  L{lineno}: {name} [{tag}]")
                print(f"        ↳ {snippet}")
            total += len(findings)
    if total == 0:
        print("null-relabel lint: clean (no null/flagged artifact citations found).")
        return 0
    print(f"\nnull-relabel lint: {total} null/flagged citation(s) surfaced for framing review.")
    print("This is a SURFACING report, not an auto-judgment -- confirm each is cited honestly "
          "(named as a null), not relabeled as a positive/beachhead/win.")
    return 1 if strict else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
