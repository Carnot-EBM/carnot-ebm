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

import importlib.util

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import LearnedVerifier, cross_game_features, cross_game_features_v2
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import graph_explore_solve_v3


def _metaharness():
    """Reuse the metaharness's banked-trajectory registry (GAME_ARTIFACTS / RESOLVED_ARTIFACTS /
    load_actions / normalize / WARMUP_GAMES) so the cross-game corpus stays in sync with the single
    source of truth for which solve trajectory each game has -- no parallel trajectory list."""
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)
    return mh


def collect_pooled(featurize=cross_game_features):
    """Pool FRAME-ONLY (features, steps-to-next-level-up) over ALL banked solves. The label is normalized
    steps to the NEXT level-up (lower == closer to advancing a level) -- exactly what the live explorer's
    frontier ordering wants -- pooled across games so a never-seen game inherits a progress heuristic.
    Replays the banked action lists offline; reads only FRAMES (live-legal). `featurize` selects v1 (5
    scalars) or v2 (richer spatial)."""
    mh = _metaharness()
    arc = kit.offline_arcade()
    X, y, per_game = [], [], {}
    for game in sorted(mh.GAME_ARTIFACTS):
        src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS[game])
        actions = mh.load_actions(src)
        if not actions:
            continue
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = env.reset()
        if game in mh.WARMUP_GAMES and actions:           # consume the swallowed first step (sc25)
            aid, data = mh.normalize(actions[0])
            if aid is not None:
                f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
        # replay, recording (frame, level) before each action
        seq = []
        for a in actions:
            aid, data = mh.normalize(a)
            if aid is None:
                continue
            seq.append((featurize(f), _levels_completed(f)))
            f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
            if f is None:
                break
        seq.append((featurize(f), _levels_completed(f)))
        # label each state with normalized steps to the NEXT level-up (segment by level transitions)
        n = len(seq)
        # find, for each index, the distance to the next index where the level increases
        next_up = [None] * n
        d = None
        run = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and seq[i + 1][1] > seq[i][1]:    # a level-up happens at i+1
                d = 0
                run = 1
            elif d is not None:
                d += 1
                run += 1
            next_up[i] = (d, run)
        for i in range(n):
            if next_up[i] is None or next_up[i][0] is None:
                continue                                   # tail after the last level-up: no label
            dist, seg = next_up[i]
            X.append(seq[i][0])
            y.append(dist / max(1, seg))                   # normalized steps-to-next-level-up in [0,1]
        per_game[game] = n
    return X, y, per_game


def main() -> int:
    print("== BRIDGE: train ONE cross-game value head on ALL banked solves (offline->live) ==")
    X, y, per_game = collect_pooled()
    print(f"  pooled {len(X)} frame-only states from {len(per_game)} games: {per_game}")
    if not X:
        print("  no trajectories — abort"); return 1
    cg = LearnedVerifier(cross_game_features).fit(X, y)
    ckpt = REPO / "models" / "arc_verifier_cross_game.json"
    cg.save(ckpt, meta={"trained_games": list(per_game), "feature_names": "cross_game_features",
                        "provenance": "BRIDGE v1: pooled normalized steps-to-next-level-up over all "
                                      "banked solve trajectories (frame-only); the live explorer loads "
                                      "this to route its frontier on UNSEEN games"})
    print(f"  trained cross-game value head on {cg.n_samples} states; weights={cg.w.round(3).tolist()}")
    print(f"  checkpoint: models/{ckpt.name}  (the live StepwiseExplorer loads this)")
    out = REPO / "results" / "arc_cross_game_verifier.json"
    out.write_text(json.dumps({"pooled_states": len(X), "per_game": per_game,
                               "weights": cg.w.tolist(), "checkpoint": f"models/{ckpt.name}",
                               "feature_names": "cross_game_features",
                               "label": "normalized_steps_to_next_level_up",
                               "mode": "bridge_cross_game_value_head_offline_to_live"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")

    if "--ab" in sys.argv:        # optional slow sanity A/B on wa30 (a held-out deep game)
        import time
        arc = kit.offline_arcade()
        for label, vf in [("novelty_only", None), ("cross_game_value_head", cg)]:
            env = arc.make("wa30", scorecard_id=arc.open_scorecard())
            t0 = time.time()
            traj, lvl = graph_explore_solve_v3(env, 0, max_expansions=20000, max_depth=80, verifier=vf)
            print(f"  wa30 [{label:22}]: {'SOLVED L'+str(lvl)+' /'+str(len(traj)) if traj else 'no-advance best L'+str(lvl)} "
                  f"[{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
