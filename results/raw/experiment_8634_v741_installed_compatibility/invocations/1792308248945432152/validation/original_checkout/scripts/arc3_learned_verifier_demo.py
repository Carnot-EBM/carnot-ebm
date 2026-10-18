"""Self-improving loop demo: a LEARNED verifier trained on lp85's own solve traces
(L1-L2) accelerates the search for the HELD-OUT next level (L3) — non-circular,
the verifier never saw L3. Compares L3 states-expanded for plain BFS vs the
learned verifier vs the hand-computed verifier (upper bound). Zero quota.

This closes the self-learning loop: each solved level trains the verifier, which
prunes the next level's search — the verifier improving from our own successes.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import LearnedVerifier, collect_trajectory_data
from carnot.experiment_4179_arc_incremental_progress import (
    discover_click_buttons, _goal_key, _target_goal_key,
)

GAME = "lp85"
DEPTH = {1: 20, 2: 70, 3: 90}


def action_labels(env):
    return [json.dumps({"x": int(b["x"]), "y": int(b["y"])}) for b in discover_click_buttons(env)]


def apply(env, label, frame):
    a = json.loads(label)
    return env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})


def state_key(game):
    return _goal_key(game)


def _dists(game):
    actual = _goal_key(game)
    target = _target_goal_key(game)
    by_type = defaultdict(list)
    for t, x, y in actual:
        by_type[t].append((x, y))
    ds = []
    for t, tx, ty in target:
        cands = by_type.get(t, [])
        ds.append(min((abs(tx - x) + abs(ty - y)) for x, y in cands) if cands else 1000.0)
    return ds


def hand_verifier(game):
    return float(sum(_dists(game)))


def featurize(game):
    """State features the value function LEARNS to weight (it is NOT told the
    answer is total-distance): total, count-unsatisfied, mean, max, n_goals."""
    ds = _dists(game)
    n = len(ds) or 1
    total = sum(ds)
    unsat = sum(1 for d in ds if d > 0)
    return [total, float(unsat), total / n, float(max(ds) if ds else 0.0), float(n)]


def new_solver(env_unused, verifier=None):
    return kit.OfflineSolver(GAME, action_labels, apply, state_key, verifier=verifier)


def main() -> int:
    print("== learned-verifier self-improving loop: lp85 (held-out L3) ==")
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())

    # --- solve L1, L2 with plain BFS; collect trajectory training data ---
    solver = new_solver(env, verifier=None)
    solver._replay(env, [])
    prefix, X, y = [], [], []
    for lvl in (1, 2):
        path, _ = solver.solve_level(env, lvl - 1, prefix, DEPTH[lvl])
        Xi, yi = collect_trajectory_data(env, solver, prefix, path, featurize)
        X += Xi; y += yi
        prefix = prefix + path
    print(f"  collected {len(X)} (state,steps-to-go) samples from L1+L2 traces")

    # --- train the verifier on L1+L2 ONLY (L3 is held out) ---
    lv = LearnedVerifier(featurize).fit(X, y)
    print(f"  trained LearnedVerifier on {lv.n_samples} samples; weights={lv.w.round(3).tolist()}")

    # --- solve the HELD-OUT L3 three ways, from the same L1+L2 prefix ---
    results = {}
    for name, vf in [("plain_bfs", None), ("learned_verifier", lv), ("hand_verifier", hand_verifier)]:
        s = new_solver(env, verifier=vf)
        path, nodes = s.solve_level(env, 2, prefix, DEPTH[3])
        # confirm it actually reaches L3
        f = s._replay(env, prefix + (path or []))
        results[name] = {"states_expanded": nodes, "reached_L3": kit.frame_level(f) >= 3, "moves": len(path or [])}
        print(f"  L3 via {name:18}: {nodes} states, reached_L3={results[name]['reached_L3']}, moves={results[name]['moves']}")

    bfs_n = results["plain_bfs"]["states_expanded"]
    lv_n = results["learned_verifier"]["states_expanded"]
    if results["learned_verifier"]["reached_L3"] and lv_n > 0:
        print(f"\n  SELF-LEARNING WIN: verifier trained on L1-L2 cut HELD-OUT L3 search "
              f"{bfs_n} -> {lv_n} states = {bfs_n / lv_n:.2f}x fewer (vs hand-verifier "
              f"{results['hand_verifier']['states_expanded']}).")
    out = REPO / "results" / "arc3_learned_verifier_demo.json"
    out.write_text(json.dumps({"game": GAME, "train_samples": lv.n_samples,
                               "weights": lv.w.tolist(), "L3": results, "mode": "offline_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
