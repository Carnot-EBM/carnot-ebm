"""Piece-3 diagnostic: does RANDOM/diverse exploration beat the structured explorer's 1/11 first-win?
(2026-06-21)

The structured depth_first_ride explorer reaches a first level-up on only 1/11 unseen games. But a random
walk over the salient candidates reaches r11l's level 1 easily (372/600 steps post-win) where the
structured explorer scored 0 -- the structure RIDES ONE BRANCH and misses easy wins. This measures whether
a simple random explorer (random among the top-K salient candidates, reset-on-degenerate, no give-up)
catches the structure-missed games and lifts the first-win count. If it does, the lever is exploration
DIVERSITY (the structured ride is over-committed), not a better value signal. Pure env-stepping, no model,
no LLM; fast.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np

GAMES = (sys.argv[1].split(",") if len(sys.argv) > 1
         else ["r11l", "lp85", "ls20", "wa30", "cd82", "sp80", "su15", "tu93", "cn04", "m0r0", "sk48"])
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 12


def _ok(grid_of, f):
    try:
        return np.asarray(grid_of(f)).ndim == 2
    except Exception:
        return False


def run(game: str) -> dict:
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_graph_explore import _warm, rich_action_candidates
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from arcengine import GameAction

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    rng = random.Random(0)
    start_level = _levels_completed(f)
    levels = start_level
    first_levelup = None
    degen = 0
    for i in range(BUDGET):
        if not _ok(grid_of, f):
            degen += 1
            f = _warm(env, False)
            continue
        cs = rich_action_candidates(f)
        if not cs:
            f = _warm(env, False)
            continue
        c = cs[rng.randrange(min(len(cs), TOPK))]
        nf = env.step(_game_action(GameAction, int(c.action_id)), data=c.data)
        if nf is None:
            f = _warm(env, False)
            continue
        lvl = _levels_completed(nf)
        if lvl > levels:
            levels = lvl
            if first_levelup is None:
                first_levelup = i + 1
        f = nf
    return {"game": game, "levels_reached": int(levels - start_level),
            "first_levelup_actions": first_levelup, "degenerate_frames": degen}


def main() -> int:
    print(f"== RANDOM explorer first-win (budget={BUDGET}, topK={TOPK}) vs structured floor 1/11 ==", flush=True)
    print(f"{'game':6} {'levels':>6} {'1st_lvlup':>9} {'degen':>5}", flush=True)
    rows = []
    won = 0
    for g in GAMES:
        try:
            r = run(g)
        except Exception as e:
            r = {"game": g, "error": f"{type(e).__name__}: {str(e)[:80]}", "levels_reached": 0}
        won += int(r.get("levels_reached", 0) > 0)
        rows.append(r)
        print(f"{g:6} {r.get('levels_reached', 0):>6} {str(r.get('first_levelup_actions')):>9} "
              f"{r.get('degenerate_frames', '?'):>5} {r.get('error', '')}", flush=True)
    print(f"\nRANDOM first-win: {won}/{len(GAMES)}  (structured explorer floor = 1/11)", flush=True)
    out = {
        "experiment": "arc_random_explore_measure",
        "honest_verdict": f"complete_random_explore_firstwin_{won}_of_{len(GAMES)}",
        "budget": BUDGET, "topK": TOPK, "structured_floor": "1/11", "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates", "verifier_is_oracle": False,
    }
    (REPO / "results" / "arc_random_explore_measure.json").write_text(json.dumps(out, indent=2, default=str))
    print("-> results/arc_random_explore_measure.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
