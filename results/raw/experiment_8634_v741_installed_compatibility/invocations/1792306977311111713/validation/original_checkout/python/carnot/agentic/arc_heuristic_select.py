"""Heuristic SELECTION learning for ARC-AGI-3 search — the "when (and when NOT) to use which
goal-distance heuristic" brain. This is the learning layer that was missing: we had several
heuristics (cell-count, 8-connected misplaced-region count) and per-game captures, but no
encoded rule for CHOOSING between them, and nothing wired into the live solve loop.

Why selection matters (measured 2026-06-17): the heuristics are NOT interchangeable.
  - `misplaced_region_distance` (8-conn wrong-region count) wins on HIGH-cell-impact games
    where one action flips many cells (r11l/m0r0/sk48): it is move-aligned (one action ≈ fixes
    one region) so A* finds the OPTIMAL path with far fewer expansions (r11l -88%).
  - `cell_count_distance` (Hamming) wins on LOW-cell-impact games (su15) where cell-count ≈
    move-count, and over-estimates move-distance on high-impact games (greedy → suboptimal).
  - pure BFS (no heuristic) is the complete fallback and is correct when NO win-state/target is
    known (a goal-distance heuristic needs a target — on a first-ever solve there is none).

The DISCRIMINATING feature is the per-action CELL IMPACT (median cells changed per move),
measurable from a handful of banked transitions: su15≈20 (cell-count better) vs r11l≈70 /
sk48≈74 / m0r0≈100 (region-count better). `recommend_order` encodes that learned rule.

For DYNAMIC ADAPTATION to UNSEEN games, `select_best` does not merely trust the rule — it runs
the small portfolio in the recommended order, reproduction-gates each, picks the winner
(fewest expansions among reproduced; solving-where-others-fail wins outright), and BANKS it to
`gap_fills/` so the choice becomes a reusable, compounding asset. Try-and-measure beats predict
on a novel game; the rule just orders the portfolio so the likely winner runs first.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

import numpy as np

from .arc_graph_explore import (
    cell_count_distance,
    graph_explore_solve_v2,
    misplaced_region_distance,
    trajectory_labels,
)

# Learned threshold separating low- vs high-cell-impact games. Derived from the 2026-06-17
# 8-game A/B: su15≈20 / cd82,sp80 low (cell-count fine) vs r11l≈70 / sk48≈74 / m0r0≈100
# (region-count wins). The portfolio selector makes the exact value non-critical — it only
# orders which heuristic is TRIED first.
HIGH_IMPACT_CELLS = 40

# The heuristic catalog. None = pure BFS (no heuristic). New heuristics register here.
HEURISTIC_NAMES = ("region_count", "cell_count", "bfs")


def factory(name: str, win) -> Optional[Callable[[Any], float]]:
    """Return the goal_distance(grid) factory for a heuristic name, or None for BFS."""
    if name == "region_count":
        return misplaced_region_distance(win, connectivity=8)
    if name == "cell_count":
        return cell_count_distance(win)
    if name == "bfs":
        return None
    raise ValueError(f"unknown heuristic {name!r}")


def per_action_cell_impact(transitions) -> float:
    """Median number of cells a single action changes, over the banked transitions. This is the
    feature that predicts which heuristic to use — high impact ⇒ cell-count over-estimates
    move-distance ⇒ prefer region-count."""
    diffs = [int((np.asarray(t.grid) != np.asarray(t.next_grid)).sum())
             for t in transitions
             if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))]
    return float(np.median(diffs)) if diffs else 0.0


def recommend_order(step_cells: float, has_target: bool) -> list[str]:
    """THE LEARNED RULE — ordered heuristic names to try for a game.

    - No win-state/target known (e.g. a first-ever solve) ⇒ ['bfs'] only: a goal-distance
      heuristic is meaningless without a target, so do NOT use one (this is the 'when NOT to').
    - High per-action cell-impact ⇒ region-count first (move-aligned), cell-count, then BFS.
    - Low per-action cell-impact ⇒ cell-count first, region-count, then BFS.
    BFS is always included last as the complete fallback (it never fails to be correct)."""
    if not has_target:
        return ["bfs"]
    if step_cells >= HIGH_IMPACT_CELLS:
        return ["region_count", "cell_count", "bfs"]
    return ["cell_count", "region_count", "bfs"]


def _run(env_factory, game, win, mask_hud, budget, max_depth, name) -> dict:
    """Run one heuristic over a fresh env; return solve metrics (reproduction-gated)."""
    from . import arc_solver_kit as kit
    from .arc_agi3_live_adapter import _game_action
    from .arc_agi3_world_model import grid_of
    from arcengine import GameAction

    gd = factory(name, win)
    heuristic = None if gd is None else (lambda frame, _gd=gd: _gd(grid_of(frame)))
    env = env_factory()
    stats: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=max_depth,
                                       heuristic=heuristic, mask_hud=mask_hud, stats=stats)
    reproduced = False
    if traj:
        def apply(env, label, frame):
            s = json.loads(label)
            return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))
        reproduced = bool(kit.reproduce(game, trajectory_labels(traj), apply,
                                        claimed_level=lvl)["reproduced"])
    return {"heuristic": name, "solved": bool(traj), "reproduced": reproduced,
            "actions": len(traj) if traj else 0, "expansions": stats.get("expansions"),
            "trajectory": traj if reproduced else None}


def select_best(game: str, win, transitions, *, mask_hud: bool = False, budget: int = 8000,
                max_depth: int = 60, env_factory: Optional[Callable[[], Any]] = None,
                bank: bool = True) -> dict:
    """DYNAMIC ADAPTATION: run the recommended heuristic portfolio on `game` (which must have a
    known `win`-state), reproduction-gate each, and return the winner. A heuristic that SOLVES
    where another doesn't wins outright; otherwise fewest expansions among reproduced wins. The
    winning heuristic is BANKED to gap_fills/ (the selection becomes a reusable asset) unless
    bank=False. Returns {chosen, step_cells, order, results:[...], rule}."""
    if env_factory is None:
        from . import arc_solver_kit as kit
        def env_factory():  # noqa: E306
            arc = kit.offline_arcade()
            return arc.make(game, scorecard_id=arc.open_scorecard())

    win = np.asarray(win)
    step_cells = per_action_cell_impact(transitions)
    order = recommend_order(step_cells, has_target=True)

    results = [_run(env_factory, game, win, mask_hud, budget, max_depth, name) for name in order]
    reproduced = [r for r in results if r["reproduced"]]

    chosen = None
    if reproduced:
        # fewest actions first (path-optimality), then fewest expansions (efficiency), then PREFER
        # BFS on a tie — a heuristic only "wins" if it STRICTLY beats plain BFS; if it merely ties,
        # BFS is the honest choice (no heuristic needed, no gap_fills file to bundle). This keeps
        # the no-headroom games (cd82/sp80/lp85-L1) correctly labelled bfs.
        chosen = min(reproduced, key=lambda r: (r["actions"], r["expansions"] or 1 << 30,
                                                0 if r["heuristic"] == "bfs" else 1))

    if bank and chosen is not None and chosen["heuristic"] != "bfs":
        _bank(game, win, chosen["heuristic"], chosen)

    return {"game": game, "chosen": chosen["heuristic"] if chosen else None,
            "step_cells": step_cells, "order": order,
            "rule": ("high-impact→region_count" if step_cells >= HIGH_IMPACT_CELLS
                     else "low-impact→cell_count"),
            "results": [{k: v for k, v in r.items() if k != "trajectory"} for r in results]}


def select_and_learn(game: str, win, transitions, *, mask_hud: bool = False, budget: int = 8000,
                     max_depth: int = 60, env_factory: Optional[Callable[[], Any]] = None
                     ) -> Optional[str]:
    """LIVE LEARNING hook: run the heuristic portfolio (select_best -> banks the winning heuristic
    to gap_fills/), THEN record the (features -> winner) to the router ledger so the TRAINED router
    (arc_router) improves with every game we solve. This is how the learning phase stays current
    for dynamic adaptation to unseen games. Returns the chosen heuristic name. Guarded: ledger
    recording must never break the solve."""
    out = select_best(game, win, transitions, mask_hud=mask_hud, budget=budget,
                      max_depth=max_depth, env_factory=env_factory, bank=True)
    try:
        from . import arc_router
        bfs_exp = next((r["expansions"] for r in out["results"] if r["heuristic"] == "bfs"), None)
        feats = arc_router.extract_features(game, win, transitions, bfs_exp)
        outcomes = {r["heuristic"]: {"reproduced": r["reproduced"], "actions": r["actions"],
                                     "expansions": r["expansions"]} for r in out["results"]}
        arc_router.record(game, feats, out["chosen"], outcomes, mask_hud=mask_hud)
    except Exception:
        pass
    return out["chosen"]


def bank_for_solved_game(game: str, win, transitions, *, mask_hud: bool = False,
                         budget: int = 8000, max_depth: int = 60,
                         env_factory: Optional[Callable[[], Any]] = None) -> Optional[str]:
    """LIVE-LOOP HOOK (cheap): once a game is solved and a win-state is known, pick the
    RULE-recommended heuristic (by measured per-action cell-impact), reproduction-gate it with a
    SINGLE search, and BANK it to gap_fills/ if it reproduces. One extra search (vs select_best's
    full portfolio) — light enough to run on every fresh solve so the learning updates live.
    Returns the banked heuristic name, or None if it didn't reproduce. Never raises on a bad
    heuristic (the caller should still guard, but this is best-effort)."""
    if env_factory is None:
        from . import arc_solver_kit as kit
        def env_factory():  # noqa: E306
            arc = kit.offline_arcade()
            return arc.make(game, scorecard_id=arc.open_scorecard())
    win = np.asarray(win)
    name = recommend_order(per_action_cell_impact(transitions), has_target=True)[0]
    r = _run(env_factory, game, win, mask_hud, budget, max_depth, name)
    if r["reproduced"]:
        _bank(game, win, name, r)
        return name
    return None


def _bank(game: str, win, name: str, metrics: dict) -> None:
    """Persist the winning heuristic to gap_fills/ as a self-contained, reusable asset."""
    from . import gap_fills
    if name == "region_count":
        body = ("    return float(ndi.label(np.asarray(grid) != WIN, "
                "structure=np.ones((3, 3), dtype=int))[1])\n")
        imports = "import numpy as np\nimport scipy.ndimage as ndi\n"
    elif name == "cell_count":
        body = "    return float((np.asarray(grid) != WIN).sum())\n"
        imports = "import numpy as np\n"
    else:
        return
    code = (f"{imports}WIN = np.array({np.asarray(win).tolist()}, dtype=np.int16)\n\n"
            f"def goal_distance(grid):\n{body}")
    gap_fills.save_heuristic(
        game, code,
        meta=(f"selected '{name}' by portfolio (step_cells-routed); reproduced "
              f"{metrics['actions']}-action solve @ {metrics['expansions']} exp"))
