"""Outer-loop sweep: run the adapter-free systematic explorer (graph_explore_solve_v2,
rich candidates) over EVERY ARC-AGI-3 game with 0 levels solved, reproduction-gate
any solve, and report which games advanced. Zero quota (offline sim).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels
from carnot.agentic.arc_agi3_live_adapter import _game_action

import os
# 15 still-unsolved games (cd82/sp80/su15/tu93 solved in the 2026-06-17 sweep; tn36
# errored on its click schema — keep it for an E1 retry now that candidates are
# salience-ordered and HUD-masking is available).
UNSOLVED = "ar25 bp35 cn04 dc22 ft09 g50t ka59 lf52 m0r0 re86 s5i5 sb26 sk48 tn36 tr87".split()
MAX_EXPANSIONS = int(os.environ.get("ARC_MAX_EXPANSIONS", "4000"))
MAX_DEPTH = int(os.environ.get("ARC_MAX_DEPTH", "45"))
MASK_HUD = os.environ.get("ARC_MASK_HUD", "0") == "1"   # E1 status-bar masking


def apply(env, label, frame):
    s = json.loads(label)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def attempt(game: str, warmup: bool):
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    t0 = time.time()
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=MAX_EXPANSIONS,
                                       max_depth=MAX_DEPTH, warmup=warmup, mask_hud=MASK_HUD)
    return traj, lvl, time.time() - t0


def main() -> int:
    print(f"== adapter-free explorer sweep over {len(UNSOLVED)} unsolved games ==", flush=True)
    solved, results = [], {}
    for i, game in enumerate(UNSOLVED, 1):
        try:
            traj, lvl, dt = attempt(game, warmup=False)
            if traj is None:                       # retry once with warm-up (first-action-consumed games)
                traj, lvl, dt2 = attempt(game, warmup=True)
                dt += dt2
            if traj:
                labels = trajectory_labels(traj)
                gate = kit.reproduce(game, labels, apply, claimed_level=lvl)
                ok = bool(gate["reproduced"])
                results[game] = {"reached_level": lvl, "moves": len(traj), "offline_reproduced": ok}
                if ok:
                    solved.append((game, lvl, len(traj)))
                    (REPO / "results" / f"arc_explore_trajectory_{game}.json").write_text(
                        json.dumps({"game": game, "reached_level": lvl, "trajectory": traj}, indent=2))
                print(f"  [{i:2}/{len(UNSOLVED)}] {game}: SOLVED L{lvl} in {len(traj)} actions "
                      f"(reproduced={ok}) [{dt:.0f}s]", flush=True)
            else:
                results[game] = {"reached_level": lvl, "no_advance": True}
                print(f"  [{i:2}/{len(UNSOLVED)}] {game}: no-advance (best L{lvl}) [{dt:.0f}s]", flush=True)
        except Exception as e:
            results[game] = {"error": repr(e)[:120]}
            print(f"  [{i:2}/{len(UNSOLVED)}] {game}: ERROR {repr(e)[:80]}", flush=True)

    print(f"\n== SWEEP RESULT: {len(solved)} new game(s) advanced adapter-free ==", flush=True)
    for g, lvl, mv in solved:
        print(f"  + {g} -> L{lvl} ({mv} actions)", flush=True)
    (REPO / "results" / "arc_explore_sweep.json").write_text(
        json.dumps({"unsolved_attempted": len(UNSOLVED), "newly_solved": solved, "results": results},
                   indent=2))
    print("  wrote results/arc_explore_sweep.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
