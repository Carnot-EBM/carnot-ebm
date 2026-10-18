"""CAPTURED GAP-FILL HEURISTICS — the autolearning store.

When the LLM gap-filler (scripts/arc_gap_fill.py) dynamically writes a per-game
goal-distance heuristic that HELPS the deterministic search SOLVE a game
(reproduction-gated), the heuristic CODE is captured here as `<game>_goal_distance.py`.
Future runs then REUSE it PRE-GENERATED (no LLM call, deterministic, instant), and the
offline competition submission BUNDLES these files (no internet at eval). Each successful
gap-fill compounds — the system gets faster and more capable per game over time. This is
the ARC-specific instance of the project's self-learning ethos (CLAUDE.md "ARC Solve
Reproducibility + Solver-Reuse"; the LLM-as-gap-filler reframe 2026-06-17): the LLM writes
the small per-game DELTA once, it is verified by the reproduction gate, then it is a
durable deterministic asset.

Capture contract: only heuristics that PASSED the reproduction gate are saved here, so a
loaded heuristic is trustworthy by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

GAP_FILL_DIR = Path(__file__).resolve().parent


def save_heuristic(game: str, code: str, meta: str = "") -> Path:
    """Persist a reproduction-gated goal_distance heuristic for `game` (autolearning)."""
    p = GAP_FILL_DIR / f"{_slug(game)}_goal_distance.py"
    header = (f"# AUTO-CAPTURED gap-fill heuristic for {game} (autolearning; reproduction-gated).\n"
              f"# {meta}\n# Reuse pre-generated; bundle for the offline competition submission.\n")
    p.write_text(header + code.strip() + "\n")
    return p


def load_heuristic(game: str) -> Optional[Callable[[Any], float]]:
    """Return the captured goal_distance(grid)->float for `game`, or None if not captured."""
    p = GAP_FILL_DIR / f"{_slug(game)}_goal_distance.py"
    if not p.exists():
        return None
    ns: dict = {}
    try:
        exec(p.read_text(), ns)
    except Exception:
        return None
    fn = ns.get("goal_distance")
    return fn if callable(fn) else None


def captured_games() -> list[str]:
    return sorted(p.name[: -len("_goal_distance.py")]
                  for p in GAP_FILL_DIR.glob("*_goal_distance.py"))


def _slug(game: str) -> str:
    return str(game).split("-", 1)[0]
