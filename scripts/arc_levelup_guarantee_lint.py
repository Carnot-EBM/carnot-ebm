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

**RETIRED 2026-07-17 (operator directive: "now that the registry of public levels is complete, we
should not continue to try and solve them anymore").** All 25 public survey games now show
`full_game_clear: true` in `ops/arc_solve_registry.yaml` (183/183 known levels; bp35 and lf52 were the
last two, solved same-day). The requirement this lint enforces -- "every roadmap must attempt to bank a
NEW reproducible level" -- is now structurally unsatisfiable: there is no remaining unsolved level in
the public set to attempt. Left unpatched, this lint would permanently HARD-BLOCK conductor milestone
activation (see `scripts/research_conductor.py`'s `_activate_next_roadmap`, date-gated through
2026-11-01) the moment a planner correctly stopped proposing dead-end solve tasks. `lint_roadmap` below
now checks the registry FIRST and passes trivially (with a clear retirement message, exit 0) once every
public game is cleared, regardless of `--min`. See CLAUDE.md "ARC Level-Up Attempt Guarantee" and
"ARC-AGI-3 November-Submission Standing Floor" for the corresponding rule-level retirement.

**FOLLOW-UP 2026-07-17 (operator answered "redirect to generalization research"):** once retired, this
lint additionally checks (via `count_generalization_attempts` / `_is_generalization_attempt`) whether the
roadmap contains a task targeting the new "ARC-AGI-3 Generalization-Testing Floor" (held-out /
leave-one-game-out live-path measurement, an `arc_solver_kit.py` reusable-primitive hardening, or
cross-game gotcha mining). This check is WARN-ONLY -- it never returns non-zero -- because the detection
heuristic is new and unproven; see CLAUDE.md "ARC-AGI-3 Generalization-Testing Floor" for the full rule
and the rationale for keeping this soft rather than promoting it to a hard gate immediately.

Usage:
  python3 scripts/arc_levelup_guarantee_lint.py [roadmap.yaml] [--min N]
Exit 0 if >= MIN level-up attempts (or if the public game set is fully cleared, see above); exit 1
otherwise (refuse the roadmap). Prints the qualifying tasks + the games targeted (so target ROTATION
across games is auditable -- the soft half of the rule).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# The 25 public survey games (targets the planner rotates through over milestones).
_GAME_NAMES = (
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "dc22",
    "ft09",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "vc33",
    "wa30",
)
_GAMES = re.compile(r"\b(" + "|".join(_GAME_NAMES) + r")\b")

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "ops" / "arc_solve_registry.yaml"


def _all_public_games_cleared(registry_path: Path = _REGISTRY_PATH) -> bool:
    """True if every one of the 25 public survey games shows full_game_clear: true in the registry.

    Returns False (fail open to the original enforcement) if the registry is missing, malformed, or
    any tracked game is not yet cleared -- this check must never itself cause a false "retired" pass.
    """
    try:
        reg = yaml.safe_load(registry_path.read_text())
        games = {g.get("game"): g for g in (reg.get("games") or []) if isinstance(g, dict)}
    except Exception:
        return False
    return all(bool(games.get(name, {}).get("full_game_clear")) for name in _GAME_NAMES)


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


_GENERALIZATION_SIGNALS = (
    "held-out",
    "held out",
    "leave-one-game-out",
    "leave one game out",
    "generalization",
    "generalisation",  # British spelling, seen in some operator prompts
    "transfer to",
    "transfer-test",
    "unseen game",
    "never-adaptered",
    "never adaptered",
    "arc_solver_kit",  # a reusable-primitive change is task-class 2 of the redirected floor
    "general_gotchas",  # cross-game gotcha mining is task-class 3
    # Task-class 4 (added 2026-08-22): supervisor refinement from the redirect
    # ledger. The trajectory supervisor's receipts now carry per-arm
    # fired/helped outcomes (REQ-ARC-WMTE-6640); a task that reads them and
    # refines the arm table is a qualifying floor task and must not draw a
    # spurious zero-qualifying-tasks warning.
    "trajectory supervisor",
    "redirect ledger",
    "arm_outcomes",
)


def _is_generalization_attempt(prompt: str) -> bool:
    """True if the task's prompt targets the 2026-07-17 ARC-AGI-3 Generalization-Testing Floor: held-out
    / leave-one-game-out measurement against the LIVE scored path, a reusable-primitive hardening in
    arc_solver_kit.py, cross-game gotcha mining into a shared primitive, or (task-class 4,\n    2026-08-22) supervisor refinement from the redirect ledger. See CLAUDE.md 'ARC-AGI-3
    Generalization-Testing Floor'. Heuristic and UNPROVEN -- kept WARN-only in lint_roadmap, never a hard
    gate, until real compliant task prompts establish what this detection should actually match."""
    p = prompt.lower()
    if not _GAMES.search(p) and "arc" not in p and "arc-agi" not in p:
        return False  # must be ARC-scoped at all before checking for the generalization signal
    return any(s in p for s in _GENERALIZATION_SIGNALS)


def count_generalization_attempts(path: Path) -> int:
    """Count tasks in `path` matching the redirected generalization-testing floor. Returns 0 on any
    read/parse failure (soft check; callers must not hard-fail on this alone)."""
    try:
        d = yaml.safe_load(path.read_text())
        tasks = d.get("tasks", []) or []
    except Exception:
        return 0
    return sum(1 for t in tasks if _is_generalization_attempt(t.get("prompt") or ""))


def lint_roadmap(path: Path, minimum: int) -> int:
    if _all_public_games_cleared(_REGISTRY_PATH):
        print(
            "RETIRED: all 25 public survey games show full_game_clear: true in "
            "ops/arc_solve_registry.yaml (183/183 known levels). The level-up-attempt requirement is "
            "moot -- there is no remaining unsolved public level to attempt. Per the 2026-07-17 operator "
            "directive, roadmaps are no longer required to propose public-game solve tasks. Passing "
            "without checking task content."
        )
        gen_count = count_generalization_attempts(path)
        if gen_count < 1:
            print(
                "WARN (soft, non-blocking): 0 generalization-testing-floor tasks detected this roadmap. "
                "Per CLAUDE.md 'ARC-AGI-3 Generalization-Testing Floor' (2026-07-17, operator-redirected "
                "from the retired public-solving floor), consider reserving >=1 slot for held-out/"
                "leave-one-game-out live-path measurement, an arc_solver_kit.py primitive hardening, or "
                "cross-game gotcha mining. This is a heuristic prompt-text match and may under-count a "
                "genuinely compliant task worded differently -- verify by eye before treating this warning "
                "as authoritative."
            )
        else:
            print(
                f"OK (soft): {gen_count} generalization-testing-floor task(s) detected this roadmap."
            )
        return 0
    d = yaml.safe_load(path.read_text())
    tasks = d.get("tasks", []) or []
    attempts = []
    for t in tasks:
        prompt = t.get("prompt") or ""
        if _is_levelup_attempt(prompt):
            games = sorted(set(_GAMES.findall(prompt.lower())))
            attempts.append((t.get("id", "?"), games))
    print(
        f"milestone: {d.get('milestone')}  tasks: {len(tasks)}  level-up attempts: {len(attempts)}"
    )
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
    print(
        f"\nOK: {len(attempts)} >= {minimum}. Games with a level-up attempt this roadmap: {touched or '(runtime-chosen)'}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roadmap", nargs="?", default="research-roadmap.yaml")
    ap.add_argument(
        "--min", type=int, default=1, help="minimum level-up attempts required (default 1)"
    )
    args = ap.parse_args()
    path = Path(args.roadmap)
    if not path.exists():
        print(f"roadmap not found: {path}")
        return 2
    return lint_roadmap(path, args.min)


if __name__ == "__main__":
    raise SystemExit(main())
