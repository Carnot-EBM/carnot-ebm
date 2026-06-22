#!/usr/bin/env python3
"""Verifier-as-Q-head, STEP 6: MILESTONE-DISTANCE re-labeling (the deepening fix, from Blind Squirrel).

v5 showed the per-level value does NOT deepen: trained on "steps-to-L1-win", at the L1-complete state
it predicts ~0 and has no L2 signal. The fix the 2nd-place leaderboard team (Blind Squirrel) uses:
re-label states by DISTANCE-TO-THE-NEXT-MILESTONE across a MULTI-LEVEL trace, so the value is
LEVEL-AGNOSTIC -- at L1-complete it predicts steps-to-L2, supplying the cross-boundary gradient.

This trains TWO SpatialValueNets on the SAME L0->L2 trace and compares routing L2 from L1-complete:
  - L1_ONLY: labels = steps-to-L1 only (the v5 baseline; states past L1 get ~0).
  - MILESTONE: labels = steps-to-NEXT-level-up (steps-to-L1 before L1, steps-to-L2 between L1 and L2).
If the MILESTONE value reaches L2 (from L1-complete, high weight) where the L1_ONLY value + blind do
NOT, milestone-relabeling cracks our deepening wall. Honest, OFFLINE, CPU. verifier_is_oracle: false.

NOTE: this needs a multi-level (L2-reachable) seed trace; run only on a game the L2 probe confirmed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels, _warm
from carnot.agentic.arc_value_net import _to_grid

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_value_q_head_v6_milestone.json"

_spec = importlib.util.spec_from_file_location(
    "vqh3", str(REPO / "scripts" / "experiments" / "experiment_value_q_head_v3.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
v2 = v3.v2
SpatialValueNet = v3.SpatialValueNet


def _ok(frame):
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def multilevel_trace(game: str, l1_budget: int, l2_budget: int):
    """Blind-solve L1, deepen to L2; replay the full L0->L2 path recording (grid, level) per state +
    the level-up indices. Returns None if L2 not reached."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    traj1, lvl1 = graph_explore_solve_v2(env, 0, max_expansions=l1_budget, max_depth=80)
    if not traj1 or int(lvl1) < 1:
        return None
    env2 = arc.make(game, scorecard_id=arc.open_scorecard())
    traj2, lvl2 = graph_explore_solve_v2(env2, 1, max_expansions=l2_budget, max_depth=120,
                                         prefix=list(traj1))
    if not traj2 or int(lvl2) < 2:
        return {"reached": int(lvl2), "l1_labels": trajectory_labels(traj1), "l1_traj": list(traj1)}
    labels = trajectory_labels(traj2)
    env3 = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env3, False)
    grids = [_to_grid(f)]
    levels = [_levels_completed(f)]
    levelups = []
    prev = levels[0]
    for i, lab in enumerate(labels):
        f = v2._apply(env3, lab, f)
        if f is None or not _ok(f):
            break
        grids.append(_to_grid(f))
        lv = _levels_completed(f)
        levels.append(lv)
        if lv > prev:
            levelups.append(i + 1)
        prev = lv
    return {"reached": int(lvl2), "labels": labels, "l1_labels": trajectory_labels(traj1),
            "l1_traj": list(traj1), "grids": grids, "levels": levels, "levelups": levelups}


def label_l1_only(grids, levels, levelups):
    """v5 baseline: steps-to-FIRST-levelup; states at/after L1 get 0 (no deeper signal)."""
    w1 = levelups[0]
    out = []
    for i, g in enumerate(grids):
        out.append((g, float(max(0, w1 - i)) if i <= w1 else 0.0))
    return out


def label_milestone(grids, levels, levelups):
    """Milestone re-labeling: distance to the NEXT level-up (level-agnostic)."""
    out = []
    for i, g in enumerate(grids):
        nxt = next((w for w in levelups if w > i), None)
        out.append((g, float(nxt - i) if nxt is not None else 0.0))
    return out


def deepen(game, prefix, budget, heuristic, weight):
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 1, max_expansions=budget, max_depth=120,
                                       prefix=list(prefix), heuristic=heuristic,
                                       heuristic_weight=weight, stats=st)
    r2 = bool(traj) and int(lvl) >= 2
    repro = (r2 and bool(kit.reproduce(game, trajectory_labels(traj), v2._apply,
                                      claimed_level=int(lvl))["reproduced"]))
    return {"reached_L2": bool(r2), "level": int(lvl), "reproduced": bool(repro), "exp": st.get("expansions")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, required=True)
    ap.add_argument("--l1-budget", type=int, default=4000)
    ap.add_argument("--l2-seed-budget", type=int, default=12000, help="budget to REACH L2 once (the seed)")
    ap.add_argument("--l2-test-budget", type=int, default=4000)
    ap.add_argument("--weight", type=float, default=10.0)
    ap.add_argument("--far-rollouts", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    game = args.game
    rng = random.Random(args.seed)

    tr = multilevel_trace(game, args.l1_budget, args.l2_seed_budget)
    if tr is None:
        verdict = "blocked_blind_no_L1"
    elif "grids" not in tr or len(tr.get("levelups", [])) < 2:
        # replay must reproduce BOTH level-ups; <2 means a non-idempotent-reset game (e.g. tu93 parity)
        n = len(tr.get("levelups", [])) if isinstance(tr, dict) else 0
        verdict = f"blocked_replay_unstable_only_{n}_levelups_reached_L{tr.get('reached', 0)}"
    else:
        prefix = [json.loads(l) for l in tr["l1_labels"]]
        hard = v2.hard_negatives(game, tr["l1_labels"], tr["levelups"][0], 3, 8.0, rng)
        far = v2.far_negatives(game, args.far_rollouts, 45, 60.0, rng)
        rng.shuffle(far)
        far = far[: max(len(hard), 20)]
        negs_g = [g for g, _ in hard] + [g for g, _ in far]
        negs_v = [v for _, v in hard] + [v for _, v in far]

        l1pos = label_l1_only(tr["grids"], tr["levels"], tr["levelups"])
        mspos = label_milestone(tr["grids"], tr["levels"], tr["levelups"])
        l1net = SpatialValueNet(device="cpu").fit([g for g, _ in l1pos] + negs_g,
                                                  [v for _, v in l1pos] + negs_v, epochs=args.epochs, seed=args.seed)
        msnet = SpatialValueNet(device="cpu").fit([g for g, _ in mspos] + negs_g,
                                                  [v for _, v in mspos] + negs_v, epochs=args.epochs, seed=args.seed)
        # at the L1-complete state, what does each value predict? (milestone should see steps-to-L2 > 0)
        l1c = tr["grids"][tr["levelups"][0]]
        print(f"  L1-complete value: L1_only={l1net.predict_grid(l1c):.2f} (should be ~0) | "
              f"MILESTONE={msnet.predict_grid(l1c):.2f} (should be ~steps-to-L2 = {tr['levelups'][1]-tr['levelups'][0]})", flush=True)

        blind = deepen(game, prefix, args.l2_test_budget, None, 1.0)
        l1r = deepen(game, prefix, args.l2_test_budget, l1net, args.weight)
        msr = deepen(game, prefix, args.l2_test_budget, msnet, args.weight)
        cracks = msr["reproduced"] and not l1r["reproduced"]
        print(f"  [{game}] L2-routing from L1-complete: BLIND repro={blind['reproduced']} L{blind['level']} | "
              f"L1_ONLY repro={l1r['reproduced']} L{l1r['level']} exp={l1r['exp']} | "
              f"MILESTONE repro={msr['reproduced']} L{msr['level']} exp={msr['exp']} | "
              f"milestone_cracks_deepening={cracks}", flush=True)
        if cracks:
            verdict = "success: milestone_relabeling_cracks_deepening_reaches_L2_where_blind_and_l1only_fail"
        elif msr["reproduced"] and blind["reproduced"]:
            verdict = "complete: L2_trivially_blind_reachable_no_deepening_advantage_demonstrable_on_this_seed"
        elif msr["reproduced"]:
            verdict = "complete: milestone_value_reaches_L2_but_no_unique_crack_vs_l1only"
        else:
            verdict = "complete: milestone_relabeling_no_L2_routing_honest_null_gap_sharpened"
        tr_summary = {"l2_seed_reached": True, "levelups": tr["levelups"],
                      "l1_complete_value_l1only": round(l1net.predict_grid(l1c), 2),
                      "l1_complete_value_milestone": round(msnet.predict_grid(l1c), 2),
                      "blind_L2": blind, "l1_only_L2": l1r, "milestone_L2": msr,
                      "milestone_cracks_deepening": bool(cracks)}

    artifact = {"experiment": "experiment_value_q_head_v6_milestone", "game": game,
                "honest_verdict": verdict, "verifier_is_oracle": False,
                "inference_substrate": "offline_arc_search_plus_cpu_cnn_train",
                "random_seed": args.seed, "mechanism": "milestone-distance re-labeling (Blind Squirrel)",
                "duration_s": round(time.time() - t0, 1)}
    if "tr_summary" in dir():
        artifact.update(tr_summary)
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
