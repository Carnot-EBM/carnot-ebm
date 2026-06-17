"""CROSS-GAME value transfer — train ONE verifier on the pooled solve trajectories
of ALL reproduced games (game-agnostic frame features -> normalized steps-to-go),
then use it to guide the deep/sparse explorer (graph_explore_solve_v3) on a game it
has never solved (wa30). This is the lever for the hard tail: a transferred value
head gives a deep game a search heuristic before it has its own trajectory.

Honest test: cross-game transfer over GENERIC grid features is a weak signal; this
measures whether it actually helps wa30 vs novelty-only. Zero quota.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import LearnedVerifier, cross_game_features
from carnot.agentic.arc_agi3_live_adapter import _game_action
from carnot.agentic.arc_graph_explore import graph_explore_solve_v3

# reproduced solves with replayable trajectories
TRAJ_SOURCES = {
    "r11l": ("results/arc_explore_trajectory_r11l.json", "trajectory"),
    "ls20": ("results/arc_explore_trajectory_ls20.json", "trajectory"),
    "lp85": ("results/arc3_lp85_offline_resolve.json", "solution"),
}


def _steps(d: dict, key: str) -> list[dict]:
    seq = d.get(key) or []
    out = []
    for s in seq:
        action = int(s.get("action", 6))
        if "data" in s:
            data = s["data"]
        elif "x" in s and "y" in s:
            data = {"x": int(s["x"]), "y": int(s["y"])}
        else:
            data = None
        out.append({"action": action, "data": data})
    return out


def _apply(env, step):
    return env.step(_game_action(GameAction, step["action"]), data=step.get("data"))


def collect_pooled():
    X, y, per_game = [], [], {}
    arc = kit.offline_arcade()
    for game, (path, key) in TRAJ_SOURCES.items():
        try:
            d = json.load(open(REPO / path))
        except Exception:
            continue
        steps = _steps(d, key)
        if not steps:
            continue
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = env.reset()
        n = len(steps)
        for i, st in enumerate(steps):
            X.append(cross_game_features(f))
            y.append((n - i) / n)            # NORMALIZED fraction-of-trajectory remaining
            f = _apply(env, st)
        per_game[game] = n
    return X, y, per_game


def main() -> int:
    print("== cross-game value transfer: train on pooled solves, guide v3 on wa30 ==")
    X, y, per_game = collect_pooled()
    print(f"  pooled {len(X)} states from {per_game}")
    if not X:
        print("  no trajectories — abort"); return 1
    cg = LearnedVerifier(cross_game_features).fit(X, y)
    ckpt = REPO / "models" / "arc_verifier_cross_game.json"
    cg.save(ckpt, meta={"trained_games": list(per_game), "feature_names": "cross_game_features",
                        "provenance": "pooled normalized steps-to-go"})
    print(f"  trained cross-game verifier on {cg.n_samples} states; weights={cg.w.round(3).tolist()}")
    print(f"  checkpoint: models/{ckpt.name}")

    # A/B on wa30: novelty-only v3 vs cross-game-verifier-guided v3
    import time
    arc = kit.offline_arcade()
    for label, vf in [("novelty_only", None), ("cross_game_verifier", cg)]:
        env = arc.make("wa30", scorecard_id=arc.open_scorecard())
        t0 = time.time()
        traj, lvl = graph_explore_solve_v3(env, 0, max_expansions=20000, max_depth=80, verifier=vf)
        print(f"  wa30 [{label:20}]: {'SOLVED L'+str(lvl)+' /'+str(len(traj)) if traj else 'no-advance best L'+str(lvl)} "
              f"[{time.time()-t0:.0f}s]")
    out = REPO / "results" / "arc_cross_game_verifier.json"
    out.write_text(json.dumps({"pooled_states": len(X), "per_game": per_game,
                               "weights": cg.w.tolist(), "checkpoint": f"models/{ckpt.name}",
                               "mode": "cross_game_value_transfer_offline_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
