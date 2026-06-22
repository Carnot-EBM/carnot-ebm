#!/usr/bin/env python3
"""Verifier-as-Q-head, STEP 5 (the prize): does the tuned value reach a level BLIND CANNOT?

Step 4 showed the spatial value routes L1 7.6x faster than blind. The real prize: that efficiency
should let it go DEEPER. This trains the value on a game's L1 trace, then searches for L2 from the
L1-complete state (prefix=L1 trajectory, high heuristic_weight) vs blind. If the value reaches L2
where blind cannot, the verifier-as-Q-head cracks levels the whole session's hand/LLM goals could
not. Honest, OFFLINE, CPU. verifier_is_oracle: false.
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
OUT = REPO / "results" / "experiment_value_q_head_v5.json"

_spec = importlib.util.spec_from_file_location(
    "vqh3", str(REPO / "scripts" / "experiments" / "experiment_value_q_head_v3.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
v2 = v3.v2
SpatialValueNet = v3.SpatialValueNet


def deepen(game: str, prefix, budget: int, heuristic, weight: float):
    """Search for L2 from the L1-complete state (prefix = the L1 trajectory of action dicts)."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 1, max_expansions=budget, max_depth=120,
                                       prefix=list(prefix), heuristic=heuristic,
                                       heuristic_weight=weight, stats=st)
    reached2 = bool(traj) and int(lvl) >= 2
    repro = (reached2 and bool(kit.reproduce(game, trajectory_labels(traj), v2._apply,
                                            claimed_level=int(lvl))["reproduced"]))
    return {"reached_L2": bool(reached2), "level": int(lvl), "reproduced": bool(repro),
            "exp": st.get("expansions")}


def run_game(game: str, args, rng) -> dict:
    tr = v2.solve_trace(game, args.l1_budget, None)
    if not tr["pos"]:
        print(f"  [{game}] blind could not reach L1 -> skip", flush=True)
        return {"game": game, "honest_verdict": "blocked_blind_no_L1"}
    prefix = [json.loads(l) for l in tr["labels"]]
    pos = tr["pos"]
    hard = v2.hard_negatives(game, tr["labels"], tr["win_at"], 3, 8.0, rng)
    far = v2.far_negatives(game, args.far_rollouts, 45, 60.0, rng)
    rng.shuffle(far)
    far = far[: max(len(hard), 20)]
    grids = [g for g, _ in pos] + [g for g, _ in hard] + [g for g, _ in far]
    values = [v for _, v in pos] + [v for _, v in hard] + [v for _, v in far]
    net = SpatialValueNet(device="cpu").fit(grids, values, epochs=args.epochs, seed=args.seed)
    on = float(np.mean([net.predict_grid(g) for g, v in pos if v <= 2])) if any(v <= 2 for _, v in pos) else 0.0
    farv = float(np.mean([net.predict_grid(g) for g, _ in far]))

    blind = deepen(game, prefix, args.l2_budget, None, 1.0)
    value = deepen(game, prefix, args.l2_budget, net, args.weight)
    cracks = value["reproduced"] and not blind["reproduced"]
    print(f"  [{game}] L1 in {tr['win_at']} actions | value(on={on:.1f},far={farv:.1f}) | "
          f"VALUE->L2: reached={value['reproduced']} L{value['level']} exp={value['exp']} | "
          f"BLIND->L2: reached={blind['reproduced']} L{blind['level']} | value_cracks_L2={cracks}", flush=True)
    return {"game": game, "l1_actions": tr["win_at"], "discrimination": round(farv - on, 2),
            "value_L2": value, "blind_L2": blind, "value_cracks_L2_where_blind_fails": bool(cracks)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=str, default="ls20,lp85")
    ap.add_argument("--l1-budget", type=int, default=3000)
    ap.add_argument("--l2-budget", type=int, default=4000)
    ap.add_argument("--weight", type=float, default=10.0)
    ap.add_argument("--far-rollouts", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    rows = []
    for game in games:
        rng = random.Random(args.seed + hash(game) % 9999)
        rows.append(run_game(game, args, rng))

    cracked = [r for r in rows if r.get("value_cracks_L2_where_blind_fails")]
    both = [r for r in rows if r.get("value_L2", {}).get("reproduced") and r.get("blind_L2", {}).get("reproduced")]
    if cracked:
        verdict = "success: value_q_head_cracks_L2_where_blind_fails_the_prize"
    elif any(r.get("value_L2", {}).get("reproduced") for r in rows):
        verdict = "complete: value_q_head_reaches_L2_but_so_does_blind_no_unique_crack"
    else:
        verdict = "complete: value_q_head_no_L2_deepening_honest_null_gap_sharpened"

    artifact = {"experiment": "experiment_value_q_head_v5", "honest_verdict": verdict,
                "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_cpu_cnn_train",
                "random_seed": args.seed, "weight": args.weight, "games": games,
                "n_games_value_cracks_L2": len(cracked), "rows": rows,
                "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}\n  value cracks L2 (where blind fails) on {len(cracked)}/{len(rows)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
