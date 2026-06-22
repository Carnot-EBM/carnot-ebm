#!/usr/bin/env python3
"""Verifier-as-Q-head, STEP 2: strengthen the learned value via HARD NEGATIVES + self-play rounds.

Step 1 (experiment_value_q_head) validated the CNN value ROUTES (1.15x on ls20, 3 seeds) but with
MARGINAL discrimination (near-win 58.5 ~= off-path 59.8) -- data-limited: 14 positives from one
blind trace + RANDOM-FAR negatives that only teach "on-blind-path vs noise", not the fine local
gradient. This step adds the two fixes:

  1. HARD NEGATIVES -- for each ON-PATH state, branch with a few WRONG actions -> one-step-off-path
     states labeled WORSE than the on-path next state. This teaches the value the LOCAL routing
     gradient ("the on-path action beats its alternatives"), the discrimination far-negatives lack.
  2. SELF-PLAY ROUNDS -- accumulate traces across rounds (round 0 = blind trace; later rounds = the
     VALUE-routed trace, which the improving value may find via a shorter/different path) -> more +
     more diverse positives -> stronger dense gradient.

Measures per round: 3-way discrimination (on-path < hard-neg < far) + routing speedup vs blind +
max level reached. Hypothesis: discrimination + speedup GROW over rounds, and the value starts
reaching deeper. Honest, OFFLINE, CPU. verifier_is_oracle: false.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import (
    graph_explore_solve_v2, trajectory_labels, rich_action_candidates, _warm,
)
from carnot.agentic.arc_value_net import ValueNet, _to_grid

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_value_q_head_v2.json"


def _ok(frame) -> bool:
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _apply(env, label, frame):
    s = json.loads(label)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def solve_trace(game: str, budget: int, heuristic):
    """graph_explore (blind if heuristic=None, else value-routed); return the winning trajectory
    labels + the per-state (grid64, steps_to_go) ON-PATH positives + expansions + level."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=80,
                                       heuristic=heuristic, stats=st)
    if not traj or int(lvl) < 1:
        return {"labels": None, "pos": [], "expansions": st.get("expansions"), "level": int(lvl)}
    labels = trajectory_labels(traj)
    env2 = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env2, False)
    grids = [_to_grid(f)]
    prev = _levels_completed(f)
    win_at = None
    for i, lab in enumerate(labels):
        f = _apply(env2, lab, f)
        if f is None or not _ok(f):
            break
        grids.append(_to_grid(f))
        if _levels_completed(f) > prev:
            win_at = i + 1
            break
    if win_at is None:
        return {"labels": labels, "pos": [], "expansions": st.get("expansions"), "level": int(lvl)}
    grids = grids[: win_at + 1]
    pos = [(g, float(win_at - i)) for i, g in enumerate(grids)]
    repro = bool(kit.reproduce(game, labels[:win_at], _apply, claimed_level=1)["reproduced"])
    return {"labels": labels[:win_at], "pos": pos, "expansions": st.get("expansions"),
            "level": int(lvl), "reproduced": repro, "win_at": win_at}


def hard_negatives(game: str, win_labels, win_at: int, n_branch: int, penalty: float, rng):
    """For each ON-PATH prefix, take a WRONG action -> a one-step-off-path state (HARD negative),
    labeled = on-path-steps-to-go + penalty (worse than the on-path next state, the local gradient)."""
    arc = kit.offline_arcade()
    out = []
    for i in range(win_at):
        for _ in range(n_branch):
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            f = _warm(env, False)
            ok = True
            for lab in win_labels[:i]:                 # replay to the on-path state at index i
                f = _apply(env, lab, f)
                if f is None or not _ok(f):
                    ok = False
                    break
            if not ok or not _ok(f):
                continue
            cands = rich_action_candidates(f)
            onpath = json.loads(win_labels[i]) if i < len(win_labels) else None
            wrong = [c for c in cands if onpath is None or int(c.action_id) != onpath.get("action")
                     or c.data != onpath.get("data")]
            if not wrong:
                continue
            c = wrong[rng.randrange(min(len(wrong), 8))]
            nf = env.step(_game_action(GameAction, int(c.action_id)), data=c.data)
            if nf is None or not _ok(nf):
                continue
            steps_to_go = float(win_at - i)
            out.append((_to_grid(nf), steps_to_go + penalty))     # worse than on-path next (w-i-1)
    return out


def far_negatives(game: str, n_rollouts: int, max_len: int, off_path_value: float, rng):
    arc = kit.offline_arcade()
    out = []
    for _ in range(n_rollouts):
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        for _ in range(max_len):
            if not _ok(f):
                break
            cands = rich_action_candidates(f)
            if not cands:
                break
            c = cands[rng.randrange(min(len(cands), 8))]
            nf = env.step(_game_action(GameAction, int(c.action_id)), data=c.data)
            if nf is None or not _ok(nf):
                break
            f = nf
            out.append(_to_grid(f))
    return [(g, off_path_value) for g in out]


def search(game: str, budget: int, heuristic):
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=80,
                                       heuristic=heuristic, stats=st)
    won = bool(traj) and int(lvl) >= 1
    repro = bool(traj) and bool(kit.reproduce(game, trajectory_labels(traj), _apply,
                                              claimed_level=int(lvl))["reproduced"]) if won else False
    return {"won": won, "reached_level": int(lvl), "offline_reproduced": repro,
            "expansions": st.get("expansions")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="ls20")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--hard-branch", type=int, default=3)
    ap.add_argument("--hard-penalty", type=float, default=8.0)
    ap.add_argument("--far-rollouts", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=45)
    ap.add_argument("--off-path-value", type=float, default=60.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    game = args.game
    rng = random.Random(args.seed)

    blind = search(game, args.budget, None)
    print(f"  blind baseline: won={blind['offline_reproduced']} exp={blind['expansions']} "
          f"L{blind['reached_level']}", flush=True)

    pos_all: list = []
    hard_all: list = []
    far_all = far_negatives(game, args.far_rollouts, args.max_len, args.off_path_value, rng)
    vnet = None
    rounds = []
    for r in range(args.rounds):
        # positive trace: blind on round 0, value-routed afterwards (improving value may find a new path)
        heur = None if (r == 0 or vnet is None) else vnet
        tr = solve_trace(game, args.budget, heur)
        if tr["pos"]:
            pos_all += tr["pos"]
            hard_all += hard_negatives(game, tr["labels"], tr["win_at"], args.hard_branch,
                                       args.hard_penalty, rng)
        if len(pos_all) < 5:
            print(f"  round {r}: no positives yet (blind/value couldn't solve) -> stop", flush=True)
            break
        # balance far negatives to ~= hard negatives so the local gradient isn't swamped
        rng.shuffle(far_all)
        far_use = far_all[: max(len(hard_all), 20)]
        grids = [g for g, _ in pos_all] + [g for g, _ in hard_all] + [g for g, _ in far_use]
        values = [v for _, v in pos_all] + [v for _, v in hard_all] + [v for _, v in far_use]
        vnet = ValueNet(device="cpu").fit(grids, values, epochs=args.epochs, seed=args.seed)
        on_mean = float(np.mean([vnet.predict_grid(g) for g, v in pos_all if v <= 2]) if any(v <= 2 for _, v in pos_all) else 0.0)
        hard_mean = float(np.mean([vnet.predict_grid(g) for g, _ in hard_all])) if hard_all else None
        far_mean = float(np.mean([vnet.predict_grid(g) for g, _ in far_use])) if far_use else None
        routed = search(game, args.budget, vnet)
        speedup = (round(blind["expansions"] / routed["expansions"], 2)
                   if routed["offline_reproduced"] and routed["expansions"] else None)
        rd = {"round": r, "n_pos": len(pos_all), "n_hard": len(hard_all), "n_far": len(far_use),
              "on_path_value": round(on_mean, 2), "hard_neg_value": round(hard_mean, 2) if hard_mean else None,
              "far_value": round(far_mean, 2) if far_mean else None,
              "routed_won": routed["offline_reproduced"], "routed_exp": routed["expansions"],
              "routed_level": routed["reached_level"], "speedup": speedup}
        rounds.append(rd)
        print(f"  round {r}: pos={len(pos_all)} hard={len(hard_all)} | values on={on_mean:.1f} "
              f"hard={hard_mean if hard_mean is None else round(hard_mean,1)} "
              f"far={far_mean if far_mean is None else round(far_mean,1)} | routed: won={routed['offline_reproduced']} "
              f"exp={routed['expansions']} L{routed['reached_level']} speedup={speedup}", flush=True)

    best = max((rd["speedup"] or 0) for rd in rounds) if rounds else 0
    deepest = max((rd["routed_level"] for rd in rounds), default=0)
    grew = (len(rounds) >= 2 and (rounds[-1]["speedup"] or 0) > (rounds[0]["speedup"] or 0))
    local_grad = any(rd["hard_neg_value"] is not None and rd["on_path_value"] < rd["hard_neg_value"] < (rd["far_value"] or 1e9)
                     for rd in rounds)
    if deepest >= 2:
        verdict = "success: value_q_head_selfplay_reached_L2_dense_routing"
    elif best >= 1.5:
        verdict = "success: value_q_head_selfplay_strong_routing_speedup"
    elif local_grad and best > 1.0:
        verdict = "complete: value_q_head_hard_negatives_learned_local_gradient_modest_routing"
    else:
        verdict = "complete: value_q_head_selfplay_no_strengthening_honest_null_gap_sharpened"

    artifact = {"experiment": "experiment_value_q_head_v2", "game": game, "honest_verdict": verdict,
                "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_cpu_cnn_train",
                "random_seed": args.seed, "blind_expansions": blind["expansions"],
                "best_speedup": best, "deepest_routed_level": deepest,
                "local_gradient_learned": local_grad, "speedup_grew_over_rounds": grew,
                "rounds": rounds, "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}\n  best_speedup={best} deepest_level={deepest} "
          f"local_gradient={local_grad} grew={grew} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
