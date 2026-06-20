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

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import LearnedVerifier, cross_game_features, cross_game_features_v2
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import graph_explore_solve_v3
from carnot.agentic.arc_variant_generator import reflect_grid


def _metaharness():
    """Reuse the metaharness's banked-trajectory registry (GAME_ARTIFACTS / RESOLVED_ARTIFACTS /
    load_actions / normalize / WARMUP_GAMES) so the cross-game corpus stays in sync with the single
    source of truth for which solve trajectory each game has -- no parallel trajectory list."""
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)
    return mh


# Reflection axes used for augmentation, in order: horizontal (1) then vertical (0). COLOR-PERMUTATION
# is deliberately NOT used: cross_game_features / _v2 are color-AGNOSTIC by design (nonzero fraction,
# #colors, dominant-color *fraction*, nonzero-occupancy map) so a color bijection leaves every feature
# identical -- recolor variants are exact duplicates (validated empirically: weights unchanged). Only a
# POSITION-changing transform (reflection) actually diversifies the spatial / occupancy features, and it
# keeps the steps-to-go label fixed -- so it teaches the verifier reflection-invariance, a valid bias
# toward the held-out, re-laid-out ~110 eval games.
_AUG_AXES = [1, 0]


def _replay_seq(env, game, actions, mh):
    """Replay the banked action list on the REAL offline env; return [(frame, level), ...]. The trajectory
    (and thus the level-up structure) is fixed here; augmentation transforms only the OBSERVED frame at
    featurize time, never the actions, so every augmented copy keeps known-correct labels."""
    f = env.reset()
    if game in mh.WARMUP_GAMES and actions:               # consume the swallowed first step (sc25)
        aid, data = mh.normalize(actions[0])
        if aid is not None:
            f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
    seq = []
    for a in actions:
        aid, data = mh.normalize(a)
        if aid is None:
            continue
        seq.append((f, _levels_completed(f)))
        f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
        if f is None:
            break
    seq.append((f, _levels_completed(f)))
    return seq


def _steps_to_next_up(levels):
    """For a level sequence, return per-index (dist, run) = steps to the next level-up (None past the
    last level-up). Segmented by level transitions so the normalizer is the segment length."""
    n = len(levels)
    next_up = [None] * n
    d = None
    run = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and levels[i + 1] > levels[i]:        # a level-up happens at i+1
            d = 0
            run = 1
        elif d is not None:
            d += 1
            run += 1
        next_up[i] = (d, run)
    return next_up


def _featurize_reflected(frame, featurize, axis):
    """Featurize `frame` with its grid stack reflected on `axis` (None = identity). Reflection moves
    positions, so the v2 occupancy map + spatial features genuinely change while colors/counts are kept."""
    if axis is None:
        return featurize(frame)
    stack = np.array(frame.frame)
    if stack.ndim == 2:
        stack = stack[None, ...]
    out = np.stack([reflect_grid(stack[i], axis=axis) for i in range(stack.shape[0])])
    f2 = frame.model_copy() if hasattr(frame, "model_copy") else frame.copy()
    object.__setattr__(f2, "frame", out.tolist())
    return featurize(f2)


def collect_pooled(featurize=cross_game_features, variants=0):
    """Pool FRAME-ONLY (features, steps-to-next-level-up) over ALL banked solves. The label is normalized
    steps to the NEXT level-up (lower == closer to advancing a level) -- exactly what the live explorer's
    frontier ordering wants -- pooled across games so a never-seen game inherits a progress heuristic.
    Replays the banked action lists offline; reads only FRAMES (live-legal). `featurize` selects v1 (5
    scalars) or v2 (richer spatial).

    REFLECTION AUGMENTATION (variants=K, K<=2: +horizontal, +vertical): replay each banked solve ONCE,
    then featurize each frame under the identity AND K reflections. Reflection keeps the real win-logic /
    level-ups (the trajectory is fixed at replay), so labels are known-correct, while it MOVES positions
    so the spatial / v2 occupancy features genuinely differ -- K+1x labeled training points that are NOT
    duplicates. This directly attacks the bottleneck that the corpus is otherwise hard-capped by the 25
    public games we have solved (the 25 are ALL the public games; the ~110 eval games are held out by
    design), and biases the verifier toward reflection-invariance for the re-laid-out eval games."""
    mh = _metaharness()
    arc = kit.offline_arcade()
    axes = [None] + _AUG_AXES[: max(0, variants)]
    X, y, per_game = [], [], {}
    for game in sorted(mh.GAME_ARTIFACTS):
        src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS[game])
        actions = mh.load_actions(src)
        if not actions:
            continue
        seq = _replay_seq(arc.make(game, scorecard_id=arc.open_scorecard()), game, actions, mh)
        next_up = _steps_to_next_up([lv for _, lv in seq])
        cnt = 0
        for axis in axes:
            for i in range(len(seq)):
                if next_up[i] is None or next_up[i][0] is None:
                    continue                               # tail after the last level-up: no label
                dist, seg = next_up[i]
                X.append(_featurize_reflected(seq[i][0], featurize, axis))
                y.append(dist / max(1, seg))               # normalized steps-to-next-level-up in [0,1]
                cnt += 1
        per_game[game] = cnt
    return X, y, per_game


def _arg_int(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return default


def main() -> int:
    # --variants K reflection-augments the corpus (K<=2: +horizontal,+vertical) -> K+1x training points
    # with known-correct labels on reflected layouts. NOTE: augmentation only adds signal with the v2
    # features (the 6x6 occupancy map changes under reflection); the v1 5-scalar features are
    # reflection-AND-recolor-invariant by design, so variants>0 auto-selects v2 (also what the live
    # agent loads via load_cross_game_value_head).
    variants = _arg_int("--variants", 0)
    use_v2 = "--v2" in sys.argv or variants > 0
    featurize = cross_game_features_v2 if use_v2 else cross_game_features
    fname = "cross_game_features_v2" if use_v2 else "cross_game_features"
    print(f"== BRIDGE: train ONE cross-game value head ({fname}) on ALL banked solves "
          f"[variants={variants}] ==")
    X, y, per_game = collect_pooled(featurize=featurize, variants=variants)
    aug = f" (reflection-augmented x{variants + 1})" if variants else ""
    print(f"  pooled {len(X)} frame-only states from {len(per_game)} games{aug}: {per_game}")
    if not X:
        print("  no trajectories — abort"); return 1
    cg = LearnedVerifier(featurize).fit(X, y)
    base = "arc_verifier_cross_game_v2" if use_v2 else "arc_verifier_cross_game"
    suffix = f"_aug{variants}" if variants else ""
    ckpt = REPO / "models" / f"{base}{suffix}.json"
    cg.save(ckpt, meta={"trained_games": list(per_game), "feature_names": fname,
                        "variants_per_game": variants,
                        "provenance": "BRIDGE: pooled normalized steps-to-next-level-up over all banked "
                                      "solve trajectories (frame-only)"
                                      + (f"; REFLECTION-augmented x{variants + 1} (known-correct labels on "
                                         "reflected layouts; diversifies the v2 occupancy features -> "
                                         "reflection-invariance for the held-out eval games)" if variants else "")
                                      + "; the live explorer loads this to route its frontier on UNSEEN games"})
    print(f"  trained cross-game value head on {cg.n_samples} states; weights[:6]={cg.w[:6].round(3).tolist()}")
    print(f"  checkpoint: models/{ckpt.name}  (the live StepwiseExplorer loads this)")
    out = REPO / "results" / f"{base.replace('arc_verifier_', 'arc_')}{suffix}.json"
    out.write_text(json.dumps({"pooled_states": len(X), "per_game": per_game,
                               "variants_per_game": variants,
                               "weights": cg.w.tolist(), "checkpoint": f"models/{ckpt.name}",
                               "feature_names": fname,
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
