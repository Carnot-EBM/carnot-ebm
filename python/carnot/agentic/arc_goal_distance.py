"""Goal-distance gradient for the navigation subclass of the ARC-AGI-3 hard tail (2026-06-21).

The diagnosis: ls20/tu93/m0r0 are AVATAR-NAVIGATION games (move an avatar with ACTION1-4 to reach a goal),
and the registry says they "need a goal-distance heuristic to avoid the 4^13 blind search". Systematic BFS
DOES find them but at ~2k expansions (near-0 efficiency); a DIRECTED A* with an avatar->goal Manhattan
heuristic finds the 13-18 action win in ~path-length expansions -> EFFICIENT recovery, the only thing that
makes a hard-tail win actually SCORE.

This builds a GENERAL (game-agnostic) detector + heuristic -- it transfers to hidden navigation games (a
toolkit member the feature-router would route to), not a per-game recipe:

  1. CALIBRATE: probe ACTION1-4 from reset; the AVATAR is the object whose cells MOVE (a colour that
     vanishes at one cell and reappears at an adjacent one). Record its colour.
  2. GOAL CANDIDATES: the rare, static, non-background, non-avatar cells the avatar might need to reach.
  3. goal_distance(frame) = min over goal candidates of Manhattan(avatar-centroid, candidate). Lower=closer.
  4. Plug into graph_explore_solve_v2(env, heuristic=goal_distance) -> A* ordered by depth + w*distance,
     which KEEPS BFS completeness (the depth term avoids the greedy local-minimum trap) while pulling the
     frontier toward the goal. The env VERIFIES (the win fires only at the true goal); a wrong candidate
     just wastes a little ordering, never blocks completeness.

Honest scope: works when there is a single coherent avatar + a reachable static goal (the nav subclass).
Click-puzzle (su15) and stateful pick/place (wa30) are out of scope (other toolkit members). No LLM.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# ACTION id -> (dy, dx) intended grid move, for matching a moving object to a direction.
_DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _centroid(grid: np.ndarray, color: int) -> Optional[tuple]:
    ys, xs = np.where(np.asarray(grid) == color)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def calibrate_avatar_goal(env, cell: int, *, warmup: bool = False, n_probe: int = 8) -> dict:
    """Probe ACTION1-4 to find the AVATAR colour (the object that moves) + GOAL-candidate cells."""
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
    from carnot.agentic.arc_graph_explore import _warm
    from carnot.agentic.arc_executable_world_model import to_logical

    f = _warm(env, warmup)
    g0 = to_logical(grid_of(f), cell)
    bg = _bg(g0)
    move_votes: dict[int, int] = {}
    probes = 0
    for aid in (1, 2, 3, 4, 1, 2, 3, 4)[:n_probe]:
        nf = env.step(_game_action(GameAction, aid), data=None)
        probes += 1
        if nf is None:
            f = _warm(env, warmup); g0 = to_logical(grid_of(f), cell); continue
        try:
            g1 = to_logical(grid_of(nf), cell)
        except Exception:
            f = _warm(env, warmup); g0 = to_logical(grid_of(f), cell); continue
        if g1.shape == g0.shape and not np.array_equal(g0, g1):
            dy, dx = _DIRS[aid]
            # a colour whose centroid shifts in the action's direction is the avatar
            for color in np.unique(g0):
                color = int(color)
                if color == bg:
                    continue
                c0 = _centroid(g0, color)
                c1 = _centroid(g1, color)
                if c0 is None or c1 is None:
                    continue
                sy, sx = c1[0] - c0[0], c1[1] - c0[1]
                if (dy != 0 and np.sign(sy) == np.sign(dy) and abs(sy) > 0.2) or \
                   (dx != 0 and np.sign(sx) == np.sign(dx) and abs(sx) > 0.2):
                    move_votes[color] = move_votes.get(color, 0) + 1
        g0 = g1; f = nf
        if _levels_completed(nf) > 0:
            break
    avatar = max(move_votes, key=move_votes.get) if move_votes else None
    # goal candidates: rare non-bg non-avatar cells in the (reset) start grid
    f = _warm(env, warmup)
    gs = to_logical(grid_of(f), cell)
    goals: list[tuple] = []
    if avatar is not None:
        vals, counts = np.unique(gs, return_counts=True)
        order = sorted(zip(vals.tolist(), counts.tolist()), key=lambda vc: vc[1])  # rarest first
        for v, _c in order:
            v = int(v)
            if v in (bg, avatar):
                continue
            ys, xs = np.where(gs == v)
            goals.append((float(ys.mean()), float(xs.mean())))
            if len(goals) >= 4:
                break
    return {"avatar": avatar, "goals": goals, "bg": bg, "probes": probes}


def make_goal_distance(avatar_color: int, goals: list[tuple], cell: int):
    """A goal_distance(frame) -> float heuristic: min Manhattan from the avatar centroid to a goal cell.
    Lower = closer to a win. Returns a large constant when the avatar is not visible (penalize losing it)."""
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import to_logical

    def goal_distance(frame_or_grid) -> float:
        try:
            g = frame_or_grid if isinstance(frame_or_grid, np.ndarray) else to_logical(grid_of(frame_or_grid), cell)
        except Exception:
            return 1e6
        a = _centroid(np.asarray(g), avatar_color)
        if a is None or not goals:
            return 1e6
        return float(min(abs(a[0] - gy) + abs(a[1] - gx) for gy, gx in goals))

    return goal_distance


def goal_distance_solve(game: str, *, budget: int = 4000, heuristic_weight: float = 2.0,
                        warmup: bool = False) -> dict:
    """Calibrate avatar+goal, then run graph_explore_solve_v2 A*-ordered by the goal-distance heuristic."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_graph_explore import _warm

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)
    try:
        cell = detect_cell(grid_of(f))
    except Exception:
        return {"game": game, "levels_reached": 0, "error": "degenerate start"}
    calib = calibrate_avatar_goal(env, cell, warmup=warmup)
    if calib["avatar"] is None or not calib["goals"]:
        return {"game": game, "levels_reached": 0, "avatar": calib["avatar"],
                "n_goals": len(calib["goals"]), "note": "no avatar/goal detected -> not a nav game"}
    heur = make_goal_distance(calib["avatar"], calib["goals"], cell)
    env2 = arc.make(game, scorecard_id=arc.open_scorecard())   # fresh env for the directed search
    r = graph_explore_solve_v2(env2, start_level=0, max_expansions=budget,
                               heuristic=heur, heuristic_weight=heuristic_weight)
    traj, lvl = (r[0], r[1]) if isinstance(r, tuple) else (None, 0)
    return {
        "game": game,
        "levels_reached": int(lvl),
        "win_traj_len": len(traj) if traj else 0,
        "avatar_color": int(calib["avatar"]),
        "n_goal_candidates": len(calib["goals"]),
        "heuristic_weight": heuristic_weight,
        "executor": "goal_distance_a_star",
    }
