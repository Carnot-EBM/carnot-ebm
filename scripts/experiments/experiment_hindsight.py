#!/usr/bin/env python3
"""Hindsight goal-induction — ground the goal on an OBSERVED win, then deepen.

The sprint proved goal-induction from a STATIC first-contact grid is fundamentally
limited (even Claude can't do it — sb26). But the chicken-and-egg has a breaker: once
exploration REACHES a level-up, you have a (start, win) PAIR, and inducing the goal from
a before/after pair (you SEE what the win achieves) is far more tractable than from
structure alone. This tests that: blind-solve L1 to OBSERVE a win, dump the (L1-start,
L1-win) pair so a strong reasoner can induce the level-invariant goal predicate, then
apply that goal to DEEPEN to L2 (where the same mechanic recurs) -> bank a new level.

Phase 1 (no --goal-code): blind graph_explore to L1, capture + print the (start, win)
pair + the level-up action. Phase 2 (--goal-code FILE): load the pair-induced
goal_progress, search L2 from the L1-complete state (prefix=L1 trajectory) using it vs
blind, reproduction-gated. Honest, OFFLINE, zero quota. verifier_is_oracle: false.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels, _warm
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_hindsight.json"


def _logical(frame, cell):
    return np.asarray(to_logical(grid_of(frame), cell))


def _apply(env, label, frame):
    s = json.loads(label)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def blind_solve_l1(game: str, cell, budget: int):
    """Blind graph_explore to L1; replay the winning trajectory to capture the start grid, the
    win grid (right after the level-up), and the L1 trajectory labels."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=80, stats=st)
    if not traj or int(lvl) < 1:
        return None
    labels = trajectory_labels(traj)
    env2 = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env2, False)
    start_grid = _logical(f, cell)
    win_grid = None
    win_at = None
    prev = _levels_completed(f)
    for i, lab in enumerate(labels):
        f = _apply(env2, lab, f)
        if f is None:
            break
        if _levels_completed(f) > prev:
            win_grid = _logical(f, cell)
            win_at = i + 1
            break
        prev = _levels_completed(f)
    return {"labels": labels, "raw_traj": list(traj), "lvl": int(lvl), "start_grid": start_grid,
            "win_grid": win_grid, "win_at": win_at,
            "win_action": labels[win_at - 1] if win_at else None,
            "expansions": st.get("expansions")}


def main() -> int:
    import scipy.ndimage as _ndi

    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, required=True)
    ap.add_argument("--l1-budget", type=int, default=4000)
    ap.add_argument("--l2-budget", type=int, default=4000)
    ap.add_argument("--max-depth", type=int, default=120)
    ap.add_argument("--goal-code", type=str, default="", help="pair-induced goal_progress(grid) .py file")
    args = ap.parse_args()
    t0 = time.time()

    arc = kit.offline_arcade()
    f0 = _warm(arc.make(args.game, scorecard_id=arc.open_scorecard()), False)
    cell = detect_cell(grid_of(f0))

    print(f"== hindsight {args.game}: blind-solve L1 to observe a win ==", flush=True)
    l1 = blind_solve_l1(args.game, cell, args.l1_budget)
    if l1 is None or l1["win_grid"] is None:
        artifact = {"experiment": "experiment_hindsight", "game": args.game,
                    "honest_verdict": "blocked_blind_could_not_reach_L1",
                    "verifier_is_oracle": False, "inference_substrate": "offline_arc_search",
                    "duration_s": round(time.time() - t0, 1)}
        OUT.write_text(json.dumps(artifact, indent=2))
        print(f"  blind did NOT reach L1 within {args.l1_budget} -> {OUT}")
        return 0
    print(f"  reached L{l1['lvl']} in {l1['win_at']} actions (win action: {l1['win_action']}); "
          f"L1 trajectory length {len(l1['labels'])}", flush=True)

    if not args.goal_code:
        # Phase 1: dump the (start, win) PAIR for pair-based goal induction.
        print(f"\n===== L1-START ({args.game}) =====\n{e3.to_ascii(l1['start_grid'])}")
        print(f"\n===== L1-WIN ({args.game}) =====\n{e3.to_ascii(l1['win_grid'])}")
        print(f"\n===== DELTA (start -> win), (row,col,from,to) =====\n"
              f"{e3._delta(l1['start_grid'], l1['win_grid'], cap=60)}")
        print(f"===== win_action: {l1['win_action']} | available next: "
              f"{list(getattr(f0, 'available_actions', []) or range(1,7))} =====\n", flush=True)
        artifact = {"experiment": "experiment_hindsight", "game": args.game, "phase": "dump_pair",
                    "honest_verdict": "complete: l1_observed_pair_dumped_for_induction",
                    "l1_level": l1["lvl"], "l1_actions": l1["win_at"],
                    "verifier_is_oracle": False, "inference_substrate": "offline_arc_search",
                    "duration_s": round(time.time() - t0, 1)}
        OUT.write_text(json.dumps(artifact, indent=2))
        return 0

    # Phase 2: load the pair-induced goal; search L2 from the L1-complete state (prefix=L1 traj).
    code = Path(args.goal_code).read_text()
    ns: dict = {"np": np, "ndi": _ndi}
    exec(code, ns)
    gp = ns["goal_progress"]
    assert math.isfinite(float(gp(l1["win_grid"]))), "goal_progress not finite on the L1-win grid"

    def heuristic(frame):
        try:
            return float(gp(_logical(frame, cell)))
        except Exception:
            return 1e9

    res = {}
    for label, hf in [("hindsight_goal", heuristic), ("blind", None)]:
        env = arc.make(args.game, scorecard_id=arc.open_scorecard())
        st: dict = {}
        traj, lvl = graph_explore_solve_v2(env, 1, max_expansions=args.l2_budget,
                                           max_depth=args.max_depth, prefix=list(l1["raw_traj"]),
                                           heuristic=hf, stats=st)
        reached2 = bool(traj) and int(lvl) >= 2
        repro = False
        if reached2:
            g = kit.reproduce(args.game, trajectory_labels(traj), _apply, claimed_level=int(lvl))
            repro = bool(g["reproduced"])
        res[label] = {"reached_L2": reached2, "level": int(lvl), "offline_reproduced": repro,
                      "expansions": st.get("expansions"), "actions": len(traj) if traj else 0}
        print(f"  [{label:14}] reached_L2={reached2} repro={repro} (L{res[label]['level']}) "
              f"exp={st.get('expansions')}", flush=True)

    hg, bl = res["hindsight_goal"], res["blind"]
    won = hg["offline_reproduced"] and not bl["offline_reproduced"]
    verdict = ("success: hindsight_pair_goal_deepened_to_L2_above_blind" if won
               else "complete: hindsight_pair_goal_no_L2_advantage_honest_null_gap_sharpened")
    artifact = {"experiment": "experiment_hindsight", "game": args.game, "phase": "deepen_L2",
                "honest_verdict": verdict, "verifier_is_oracle": False,
                "inference_substrate": "offline_arc_search",
                "hindsight_deepened_where_blind_did_not": won,
                "hindsight_goal": hg, "blind": bl, "goal_code": code.strip()[:1200],
                "l1_actions": l1["win_at"], "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
