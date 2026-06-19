#!/usr/bin/env python3
"""ARC Level-Up Attempt Guarantee lint.

Operator directive 2026-06-19: "ensure we ALWAYS have at least one level-up attempt across all games
once a roadmap." The ARC-AGI-3 submission sprint's only headline metric is reproducible_total_levels
growing monotonically -- but a planner can drift to all-meta work (library induction, benchmarks,
registry hygiene, SOTA ingestion) and ship a roadmap with ZERO concrete level-bank attempts, which is
pure churn (north-star sec.1). This lint GUARANTEES, mechanically, that every ARC-sprint roadmap
contains at least MIN level-up SOLVE attempts.

A "level-up attempt" = a task whose acceptance gate requires actually BANKING a reproducible level:
its prompt asserts `offline_reproduced=true` AND a new-level condition (reproduced_levels>=1, OR a
"NEW <game> level reproduced" / "level > N" deepen, OR a first-contact/+1-level/L1 solve). A task that
only RE-solves an already-banked level via a generic operator (generalization validation), or that
benchmarks / induces a library / reconciles the registry, does NOT count.

Usage:
  python3 scripts/arc_levelup_guarantee_lint.py [roadmap.yaml] [--min N]
Exit 0 if >= MIN level-up attempts; exit 1 otherwise (refuse the roadmap). Prints the qualifying
tasks + the games targeted (so target ROTATION across games is auditable -- the soft half of the rule).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# The 25 public survey games (targets the planner rotates through over milestones).
_GAMES = re.compile(
    r"\b(ar25|bp35|cd82|cn04|dc22|ft09|g50t|ka59|lf52|lp85|ls20|m0r0|r11l|re86|s5i5|sb26|sc25|sk48|"
    r"sp80|su15|tn36|tr87|tu93|vc33|wa30)\b"
)


def _is_levelup_attempt(prompt: str) -> bool:
    """True if the task's acceptance gate requires banking a NEW reproducible level (not just a
    generic re-solve of an already-banked level, and not a meta/benchmark/hygiene task)."""
    p = prompt.lower()
    if "offline_reproduced" not in p:
        return False
    bank_signals = (
        "reproduced_levels>=1",
        "reproduced_levels >= 1",
        "reproduced_levels > 0",
        "new sc25 level",
        "new level reproduced",
        " level > 1",
        " level>1",
        "+1 reproducible level",
        "+1 level",
        "first-contact",  # first-contact a never-attempted game = a new-level attempt
        "+1 deeper",
        "deeper level",
    )
    # A genuine new-level signal is REQUIRED. A pure generic RE-solve of an already-banked level
    # (generalization validation: "re-solves via the generic operator, NOT its own recipe") carries no
    # bank_signal, so it is correctly NOT counted -- no fragile exclusion needed.
    return any(s in p for s in bank_signals)


def lint_roadmap(path: Path, minimum: int) -> int:
    d = yaml.safe_load(path.read_text())
    tasks = d.get("tasks", []) or []
    attempts = []
    for t in tasks:
        prompt = t.get("prompt") or ""
        if _is_levelup_attempt(prompt):
            games = sorted(set(_GAMES.findall(prompt.lower())))
            attempts.append((t.get("id", "?"), games))
    print(f"milestone: {d.get('milestone')}  tasks: {len(tasks)}  level-up attempts: {len(attempts)}")
    for tid, games in attempts:
        print(f"  LEVEL-UP  {tid:42} targets={games or ['(game chosen at runtime)']}")
    if len(attempts) < minimum:
        print(
            f"\nFAIL: only {len(attempts)} level-up attempt(s) (< {minimum} required). This roadmap "
            f"would produce no new reproducible level = churn. The planner MUST add a SOLVE task whose "
            f"gate is offline_reproduced=true AND reproduced_levels>=1 (first-contact an unsolved game, "
            f"or +1 deeper on a solved one). See CLAUDE.md 'ARC Level-Up Attempt Guarantee'."
        )
        return 1
    # soft audit: which games get a level-up attempt (rotation visibility)
    touched = sorted({g for _, gs in attempts for g in gs})
    print(f"\nOK: {len(attempts)} >= {minimum}. Games with a level-up attempt this roadmap: {touched or '(runtime-chosen)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roadmap", nargs="?", default="research-roadmap.yaml")
    ap.add_argument("--min", type=int, default=1, help="minimum level-up attempts required (default 1)")
    args = ap.parse_args()
    path = Path(args.roadmap)
    if not path.exists():
        print(f"roadmap not found: {path}")
        return 2
    return lint_roadmap(path, args.min)


if __name__ == "__main__":
    raise SystemExit(main())
