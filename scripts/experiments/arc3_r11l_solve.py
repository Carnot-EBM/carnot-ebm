"""M2-v5b: FIRST-SOLVE attempt on r11l (the survey's easiest inducible target).

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). The win-condition survey
(results/arc3_win_condition_survey.json) picked r11l: a click-to-drag placement puzzle whose goal is
DIRECTLY OBSERVABLE — small colored PIECES must be dragged onto matching GRAY (color-2) TEMPLATE
outlines; win when all pieces are placed (no spatial planning, unlike vc33). Mechanic (induced from
play; confirmed against source for understanding): a click selects the piece under the cursor; a 2nd
click places the selected piece centered at the cursor (if the spot is valid); the level completes
when every piece sits on its target.

Solver: perceive PIECES (connected components of non-background, non-gray colors) + TEMPLATES (gray
connected components), match each piece to the template of most similar shape (bbox + filled-area),
then for each match execute [click piece centroid, click template centroid] in the REAL env. The REAL
env confirms the solve via level_completed (ground truth). If the shape-matching doesn't win, fall back
to env-feedback search over piece->template assignments (resetting on lose, within the action budget).
Fully offline. No LLM, no GPU. The model/verifier framing: perception proposes placements, the real
env verifies; a future version routes the placements through the induced world-model as a pre-check.

  .venv/bin/python scripts/experiments/arc3_r11l_solve.py --game r11l-495a7899 --budget 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402


def _background(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(grid, colors):
    """Connected components (4-neighbour) restricted to `colors`. Returns list of dicts with color,
    cells, centroid (y,x), bbox (h,w), area."""
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    target = np.isin(grid, list(colors))
    for i in range(h):
        for j in range(w):
            if target[i, j] and not seen[i, j]:
                col = int(grid[i, j])
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and target[ny, nx]
                            and not seen[ny, nx]
                            and grid[ny, nx] == col
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                comps.append(
                    {
                        "color": col,
                        "cells": cells,
                        "area": len(cells),
                        "centroid": (sum(ys) // len(cells), sum(xs) // len(cells)),
                        "bbox": (max(ys) - min(ys) + 1, max(xs) - min(xs) + 1),
                    }
                )
    return comps


def _perceive(grid, gray=2, min_area=2, max_piece_area=80):
    bg = _background(grid)
    pieces = [
        c
        for c in _components(grid, set(range(16)) - {bg, gray})
        if min_area <= c["area"] <= max_piece_area
    ]
    templates = [c for c in _components(grid, {gray}) if c["area"] >= min_area]
    return bg, pieces, templates


def _shape_dist(a, b):
    (ah, aw), (bh, bw) = a["bbox"], b["bbox"]
    return abs(ah - bh) + abs(aw - bw) + abs(a["area"] - b["area"]) * 0.2


def _match(pieces, templates):
    """Greedy shape-match each piece to a distinct template."""
    used = set()
    pairs = []
    for p in sorted(pieces, key=lambda c: -c["area"]):
        best, bd = None, 1e9
        for ti, t in enumerate(templates):
            if ti in used:
                continue
            d = _shape_dist(p, t)
            if d < bd:
                bd, best = d, ti
        if best is not None:
            used.add(best)
            pairs.append((p, templates[best]))
    return pairs


def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})


def _attempt(env, GameAction, GameState, order, budget_left, log):
    """Execute click-piece -> click-target for each (piece,target) in `order`; return (levels, actions)."""
    f = None
    actions = 0
    for p, t in order:
        if actions + 2 > budget_left:
            break
        py, px = p["centroid"]
        ty, tx = t["centroid"]
        f = _click(env, GameAction, py, px)
        actions += 1  # select the piece
        f = _click(env, GameAction, ty, tx)
        actions += 1  # place it on its template
        lv = int(getattr(f, "levels_completed", 0) or 0)
        st = getattr(f, "state", None)
        log.append(
            {
                "piece": p["centroid"],
                "target": t["centroid"],
                "color": p["color"],
                "level": lv,
                "state": str(st),
            }
        )
        if st in (GameState.WIN, GameState.GAME_OVER):
            break
    return f, actions


def run(game="r11l-495a7899", budget=60, max_resets=8, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState

    started = time.time()
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    info = {
        getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or [])
        for e in arc.get_environments()
    }
    win_levels = len(info.get(game, []))

    env = arc.make(game)
    f = env.reset()
    grid = grid_of(f)
    bg, pieces, templates = _perceive(grid)
    n_pieces = len(pieces)
    n_templates = len(templates)
    max_level = 0
    total_actions = 0
    solve_log = []
    attempts = []

    # primary: shape-matched assignment
    orders = [_match(pieces, templates)]
    # fallbacks: if few pieces, try permutations of templates (env-feedback search)
    if 0 < n_pieces <= 4 and n_templates >= n_pieces:
        for perm in list(permutations(range(n_templates), n_pieces))[:max_resets]:
            orders.append([(pieces[i], templates[perm[i]]) for i in range(n_pieces)])

    for ai, order in enumerate(orders):
        if not order:
            continue
        f = env.reset()
        log = []
        f, used = _attempt(env, GameAction, GameState, order, budget, log)
        total_actions += used
        lv = int(getattr(f, "levels_completed", 0) or 0)
        attempts.append(
            {"attempt": ai, "n_pairs": len(order), "actions": used, "level_reached": lv}
        )
        if lv > max_level:
            max_level = lv
            solve_log = log
        if lv > 0:
            break

    solved = max_level > 0
    verdict = (
        f"complete: r11l_first_solve_levels{max_level}_of{win_levels}_solved{solved}"
        f"_pieces{n_pieces}_templates{n_templates}_attempts{len(attempts)}"
    )
    art = {
        "experiment": "arc3_r11l_solve",
        "title": "arc3_m2v5b_r11l_first_solve_attempt",
        "honest_verdict": verdict,
        # LEGAL substrate per CLAUDE.md's Inference-Substrate table. This script previously
        # wrote "offline_arc_agi3_perception_planner_real_env_confirmed", which is not in
        # that table, so every re-run recreated an artifact the ARC artifact lint rejects
        # (the exp3946 writer had the same defect, fixed 2026-07-27; see commit 0a6329fb45's
        # sibling). Honest: this script steps the offline Arcade sim; no LLM import exists.
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "game": game,
        "win_levels": win_levels,
        "ACCURACY_levels_solved": max_level,
        "solved": solved,
        "n_pieces_perceived": n_pieces,
        "n_templates_perceived": n_templates,
        "total_actions": total_actions,
        "attempts": attempts,
        "solve_log": solve_log[:40],
        "real_env_confirmed": True,
        "budget": budget,
        "no_llm_used": True,
        "no_gpu_used": True,
        "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1),
        "note": (
            "M2-v5b first-solve attempt on r11l (survey's easiest target). Perception matches pieces "
            "to gray templates by shape; the REAL env confirms via level_completed. solved=True = the "
            "FIRST ARC-AGI-3 solve. Quota-gate: online play only when an offline solve beats the TRM "
            "baseline + best prior Carnot submission (operator-gated)."
        ),
    }
    if write:
        (REPO / "results" / "arc3_r11l_solve.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8"
        )
    print(f"-> {verdict}")
    print(
        f"   pieces={n_pieces} templates={n_templates} attempts={len(attempts)} max_level={max_level}"
    )
    for a in attempts[:6]:
        print(
            f"     attempt {a['attempt']}: {a['n_pairs']} pairs, {a['actions']} actions -> level {a['level_reached']}"
        )
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="r11l-495a7899")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--max_resets", type=int, default=8)
    args = ap.parse_args()
    art = run(game=args.game, budget=args.budget, max_resets=args.max_resets)
    raise SystemExit(0 if art["ACCURACY_levels_solved"] > 0 else 1)
