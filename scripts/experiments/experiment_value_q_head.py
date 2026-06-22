#!/usr/bin/env python3
"""Verifier-as-Q-head, STEP 1: does a LEARNED CNN value ROUTE the live search?

The session's unifying finding: every hand/LLM goal yields a SPARSE/terminal signal; the
deep levels need a DENSE per-step gradient. A learned value V(state)=steps-to-go IS that
dense signal. The prior finding (results/arc_offline_to_live_bridge_v2.json): a LINEAR head
over 5-41 hand-features CANNOT route (actively misleading). The CNN ValueNet (arc_value_net)
sees the GRID directly -- hypothesis: it has the capacity to route IF trained on enough data
(on-path positives + off-path negatives). That hypothesis was never conclusively tested for
live routing. This tests it.

FOUNDATION TEST: collect win-traces (positives: steps-to-go) + off-path negatives by
exploration, train ValueNet, then route graph_explore with it vs blind BFS. The learned
value WINS if it reaches the level with FEWER expansions (the dense routing the linear head
lacked). Testbed ls20: blind reaches L1 but at ~2k expansions (near-0 efficiency, registry),
so there is real room for a router to win. Reproduction-gated, OFFLINE, CPU (tiny net, no
3090 contention). verifier_is_oracle: false (the value ESTIMATES distance-to-win, learned).
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
OUT = REPO / "results" / "experiment_value_q_head.json"


def _ok(frame) -> bool:
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _apply(env, label, frame):
    s = json.loads(label)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def collect_training_data(game: str, n_rollouts: int, max_len: int, off_path_value: float, rng):
    """Salient-random rollouts -> (grid64, steps_to_go) pairs. A rollout that reaches a level-up at
    step w labels its states 0..w with steps_to_go = w - i (ON-PATH positives, dropping toward the
    win). A rollout that never advances labels its states with `off_path_value` (NEGATIVES — the
    discrimination the linear head lacked). Returns grids, values, n_win_rollouts."""
    arc = kit.offline_arcade()
    grids: list = []
    values: list = []
    n_win = 0
    for _ in range(n_rollouts):
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        start = _levels_completed(f)
        traj_grids = [_to_grid(f)]
        win_at = None
        for i in range(max_len):
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
            traj_grids.append(_to_grid(f))
            if _levels_completed(f) > start:
                win_at = i + 1
                break
        if win_at is not None:
            n_win += 1
            for i, g in enumerate(traj_grids[: win_at + 1]):
                grids.append(g)
                values.append(float(win_at - i))      # steps-to-go (0 at the win)
        else:
            for g in traj_grids:                        # off-path: far from a win
                grids.append(g)
                values.append(float(off_path_value))
    return grids, values, n_win


def blind_positive_trace(game: str, budget: int):
    """Bootstrap positives where random exploration can't reach a win: blind graph_explore finds L1
    (inefficiently), and its winning path -> (grid64, steps_to_go) is the ON-PATH trace the value
    should learn to route along. Returns (grids, values) or ([],[]) if blind can't solve."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=80)
    if not traj or int(lvl) < 1:
        return [], []
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
        return [], []
    grids = grids[: win_at + 1]
    values = [float(win_at - i) for i in range(len(grids))]   # steps-to-go, 0 at the win
    return grids, values


def search(game: str, budget: int, heuristic) -> dict:
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=80,
                                       heuristic=heuristic, stats=st)
    won = bool(traj) and int(lvl) >= 1
    repro = False
    if won:
        g = kit.reproduce(game, trajectory_labels(traj), _apply, claimed_level=int(lvl))
        repro = bool(g["reproduced"])
    return {"won": won, "reached_level": int(lvl), "offline_reproduced": repro,
            "expansions": st.get("expansions"), "actions": len(traj) if traj else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=str, default="ls20")
    ap.add_argument("--rollouts", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--off-path-value", type=float, default=60.0)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    rows = []
    for game in games:
        rng = random.Random(args.seed + hash(game) % 9999)
        t1 = time.time()
        # POSITIVES: random-reached wins (easy games) + a blind-solve trace (hard games like ls20
        # where random never wins). NEGATIVES: off-path random states, BALANCED to ~3x positives so
        # the CNN learns discrimination, not the majority label.
        rg, rv, n_rand_win = collect_training_data(game, args.rollouts, args.max_len,
                                                   args.off_path_value, rng)
        rand_pos = [(g, v) for g, v in zip(rg, rv) if v < args.off_path_value]
        rand_neg = [g for g, v in zip(rg, rv) if v >= args.off_path_value]
        bpg, bpv = blind_positive_trace(game, args.budget)
        positives = rand_pos + list(zip(bpg, bpv))
        n_win = n_rand_win + (1 if bpg else 0)
        if len(positives) < 5:
            rows.append({"game": game, "honest_verdict": "blocked_no_win_traces_for_training",
                         "n_win_rollouts": n_win, "n_positives": len(positives)})
            print(f"  [{game}] only {len(positives)} positive states (random_wins={n_rand_win}, "
                  f"blind_trace={'yes' if bpg else 'no'}) -> can't train (chicken-egg)", flush=True)
            continue
        rng.shuffle(rand_neg)
        negatives = rand_neg[: max(10, 3 * len(positives))]
        grids = [g for g, _ in positives] + negatives
        values = [v for _, v in positives] + [args.off_path_value] * len(negatives)
        print(f"  [{game}] training on {len(positives)} positives (rand={len(rand_pos)}, "
              f"blind={len(bpg)}) + {len(negatives)} negatives", flush=True)
        vnet = ValueNet(device="cpu").fit(grids, values, epochs=args.epochs, seed=args.seed)
        # sanity: does the value DISCRIMINATE (near-win states score lower than off-path)?
        near = float(np.mean([vnet.predict_grid(g) for g, v in zip(grids, values) if v <= 2]))
        far = float(np.mean([vnet.predict_grid(g) for g, v in zip(grids, values) if v >= args.off_path_value]))

        learned = search(game, args.budget, vnet)            # value-routed
        blind = search(game, args.budget, None)              # blind BFS
        # routing win: reaches the level with FEWER expansions (or reaches it where blind doesn't)
        both_won = learned["offline_reproduced"] and blind["offline_reproduced"]
        routes_better = (
            (learned["offline_reproduced"] and not blind["offline_reproduced"])
            or (both_won and learned["expansions"] is not None and blind["expansions"] is not None
                and learned["expansions"] < blind["expansions"])
        )
        row = {
            "game": game, "n_win_rollouts": n_win, "n_states": len(grids),
            "value_discriminates": bool(near < far), "near_win_value_mean": round(near, 2),
            "off_path_value_mean": round(far, 2), "train_loss": round(vnet.last_train_loss, 3),
            "learned": learned, "blind": blind,
            "value_routes_better_than_blind": bool(routes_better),
            "expansion_speedup": (round(blind["expansions"] / learned["expansions"], 2)
                                  if both_won and learned["expansions"] else None),
            "secs": round(time.time() - t1, 1),
        }
        rows.append(row)
        print(f"  [{game}] discriminates={row['value_discriminates']} "
              f"(near={near:.1f}<far={far:.1f}) | learned: won={learned['offline_reproduced']} "
              f"exp={learned['expansions']} L{learned['reached_level']} | blind: won={blind['offline_reproduced']} "
              f"exp={blind['expansions']} | routes_better={routes_better} "
              f"speedup={row['expansion_speedup']} [{row['secs']}s]", flush=True)

    valid = [r for r in rows if "learned" in r]
    n_better = sum(1 for r in valid if r["value_routes_better_than_blind"])
    n_discrim = sum(1 for r in valid if r["value_discriminates"])
    if not valid:
        verdict = "blocked_no_trainable_game"
    elif n_better >= 1:
        verdict = "success: learned_cnn_value_routes_better_than_blind_q_head_foundation_validated"
    elif n_discrim >= 1:
        verdict = "complete: learned_value_discriminates_but_does_not_route_better_honest_null_gap_sharpened"
    else:
        verdict = "complete: learned_value_no_discrimination_honest_null_gap_sharpened"

    artifact = {
        "experiment": "experiment_value_q_head", "honest_verdict": verdict,
        "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_cpu_cnn_train",
        "random_seed": args.seed, "games": games, "budget": args.budget,
        "n_games_value_routes_better": n_better, "n_games_value_discriminates": n_discrim,
        "prior_finding": "linear head over hand-features CANNOT route (arc_offline_to_live_bridge_v2); this tests the CNN",
        "rows": rows, "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"  value-routes-better on {n_better}/{len(valid)}; discriminates {n_discrim}. -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
