#!/usr/bin/env python3
"""Verifier-as-Q-head, STEP 4: make the strong spatial value ROUTE by tuning the A* weight.

Step 3 gave a value that DISCRIMINATES sharply (on-path 13 << far 43) but did NOT route better than
the flat global-pool value (1.16x vs 1.21x). The A* orders by `depth + heuristic_weight * value`;
the spatial value's wide range (13-43) at the default weight=1.0 mis-balances vs the depth term.
This sweeps heuristic_weight: if SOME weight routes the strong value much better (or reaches L2), the
value-Q-head is validated (the value is good, the search just needed tuning). If NO weight helps,
the value doesn't generalize to the search frontier (a deeper problem). Honest, OFFLINE, CPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_value_q_head_v4.json"

_spec = importlib.util.spec_from_file_location(
    "vqh3", str(REPO / "scripts" / "experiments" / "experiment_value_q_head_v3.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
v2 = v3.v2
SpatialValueNet = v3.SpatialValueNet


def search_w(game: str, budget: int, heuristic, weight: float):
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=80,
                                       heuristic=heuristic, heuristic_weight=weight, stats=st)
    won = bool(traj) and int(lvl) >= 1
    repro = (won and bool(kit.reproduce(game, trajectory_labels(traj), v2._apply,
                                        claimed_level=int(lvl))["reproduced"]))
    return {"won": won, "level": int(lvl), "reproduced": bool(repro), "exp": st.get("expansions")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="ls20")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--weights", type=str, default="0.05,0.1,0.3,1.0,3.0,10.0")
    ap.add_argument("--hard-branch", type=int, default=3)
    ap.add_argument("--far-rollouts", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    game = args.game
    rng = random.Random(args.seed)
    weights = [float(w) for w in args.weights.split(",")]

    blind = v2.search(game, args.budget, None)
    print(f"  blind: exp={blind['expansions']} L{blind['reached_level']}", flush=True)
    tr = v2.solve_trace(game, args.budget, None)
    if not tr["pos"]:
        OUT.write_text(json.dumps({"experiment": "experiment_value_q_head_v4", "game": game,
                                   "honest_verdict": "blocked_blind_no_L1"}, indent=2)); return 0
    pos = tr["pos"]
    hard = v2.hard_negatives(game, tr["labels"], tr["win_at"], args.hard_branch, 8.0, rng)
    far = v2.far_negatives(game, args.far_rollouts, 45, 60.0, rng)
    rng.shuffle(far)
    far = far[: max(len(hard), 20)]
    grids = [g for g, _ in pos] + [g for g, _ in hard] + [g for g, _ in far]
    values = [v for _, v in pos] + [v for _, v in hard] + [v for _, v in far]
    net = SpatialValueNet(device="cpu").fit(grids, values, epochs=args.epochs, seed=args.seed)
    on = float(np.mean([net.predict_grid(g) for g, v in pos if v <= 2]))
    farv = float(np.mean([net.predict_grid(g) for g, _ in far]))
    print(f"  spatial value: on-path={on:.1f} far={farv:.1f} (discrimination {farv - on:.1f})", flush=True)

    sweep = []
    for w in weights:
        r = search_w(game, args.budget, net, w)
        speedup = round(blind["expansions"] / r["exp"], 2) if r["reproduced"] and r["exp"] else None
        sweep.append({"weight": w, "won": r["reproduced"], "exp": r["exp"], "level": r["level"], "speedup": speedup})
        print(f"  weight={w:<5} won={r['reproduced']} exp={r['exp']} L{r['level']} speedup={speedup}", flush=True)

    valid = [s for s in sweep if s["speedup"]]
    best = max(valid, key=lambda s: s["speedup"]) if valid else None
    deepest = max((s["level"] for s in sweep), default=0)
    if deepest >= 2:
        verdict = "success: spatial_value_weight_tuned_reached_L2"
    elif best and best["speedup"] >= 1.5:
        verdict = "success: spatial_value_weight_tuned_strong_routing_speedup"
    elif best and best["speedup"] > 1.25:
        verdict = "complete: spatial_value_weight_tuned_improved_routing_modest"
    else:
        verdict = "complete: no_weight_routes_strong_value_well_frontier_generalization_gap"

    artifact = {"experiment": "experiment_value_q_head_v4", "game": game, "honest_verdict": verdict,
                "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_cpu_cnn_train",
                "random_seed": args.seed, "blind_expansions": blind["expansions"],
                "spatial_discrimination": round(farv - on, 2), "best_weight": best,
                "deepest_level": deepest, "sweep": sweep, "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}\n  best={best} deepest_level={deepest} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
