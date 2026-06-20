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
import hashlib
import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import importlib.util

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import (
    DiscriminativeVerifier,
    LearnedVerifier,
    cross_game_feature_slices_v3,
    cross_game_features,
    cross_game_features_v2,
    cross_game_features_v3,
)
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import graph_explore_solve_v3
from carnot.agentic.arc_variant_generator import reflect_grid

_SIMPLE_ACTIONS = [1, 2, 3, 4, 5]  # non-click actions usable as off-path probes without coordinates


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


def _reflect_frame(frame, axis):
    if frame is None or axis is None:
        return frame
    if axis is None:
        return frame
    stack = np.array(frame.frame)
    if stack.ndim == 2:
        stack = stack[None, ...]
    out = np.stack([reflect_grid(stack[i], axis=axis) for i in range(stack.shape[0])])
    f2 = frame.model_copy() if hasattr(frame, "model_copy") else frame.copy()
    object.__setattr__(f2, "frame", out.tolist())
    return f2


def _featurize_with_context(featurize, frame, previous_frame=None, action_id=None, goal_frame=None):
    params = inspect.signature(featurize).parameters
    kwargs = {}
    if "previous_frame" in params:
        kwargs["previous_frame"] = previous_frame
    if "action_id" in params:
        kwargs["action_id"] = action_id
    if "goal_frame" in params:
        kwargs["goal_frame"] = goal_frame
    return featurize(frame, **kwargs)


def _featurize_reflected(frame, featurize, axis, previous_frame=None, action_id=None, goal_frame=None):
    """Featurize `frame` with reflected context on `axis` (None = identity)."""
    f2 = _reflect_frame(frame, axis)
    prev2 = _reflect_frame(previous_frame, axis)
    goal2 = _reflect_frame(goal_frame, axis)
    return _featurize_with_context(featurize, f2, previous_frame=prev2, action_id=action_id, goal_frame=goal2)


def _goal_frame_for_index(seq, next_up, i):
    if not seq:
        return None
    if i < len(next_up) and next_up[i] is not None and next_up[i][0] is not None:
        target = min(len(seq) - 1, i + int(next_up[i][0]) + 1)
        return seq[target][0]
    return seq[-1][0]


def _previous_context(seq, norm, i):
    prev = seq[i - 1][0] if i > 0 else None
    action = norm[i - 1][0] if 0 < i <= len(norm) else None
    return prev, action


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
        norm = [(aid, d) for aid, d in (mh.normalize(a) for a in actions) if aid is not None]
        seq = _replay_seq(arc.make(game, scorecard_id=arc.open_scorecard()), game, actions, mh)
        next_up = _steps_to_next_up([lv for _, lv in seq])
        cnt = 0
        for axis in axes:
            for i in range(len(seq)):
                if next_up[i] is None or next_up[i][0] is None:
                    continue                               # tail after the last level-up: no label
                dist, seg = next_up[i]
                prev, action = _previous_context(seq, norm, i)
                goal = _goal_frame_for_index(seq, next_up, i)
                X.append(_featurize_reflected(
                    seq[i][0], featurize, axis, previous_frame=prev, action_id=action, goal_frame=goal))
                y.append(dist / max(1, seg))               # normalized steps-to-next-level-up in [0,1]
                cnt += 1
        per_game[game] = cnt
    return X, y, per_game


def collect_discriminative(featurize=cross_game_features_v2, neg_per_game=14, seed=0):
    """POS = on-winning-path states (banked replay); NEG = OFF-PATH states reached by ONE non-gold action
    from a sampled on-path prefix (replay-from-reset, correct for non-deepcopy games). A non-gold action
    that ACCIDENTALLY levels up is skipped (it is not off-path). This is the "off-path negatives" corpus the
    steps-to-go regressor never sees -- the signal a win-reachability classifier needs. Returns X, y
    (1=on-path, 0=off-path), per_game pos/neg counts."""
    mh = _metaharness()
    arc = kit.offline_arcade()
    rng = np.random.default_rng(seed)
    X, y, per_game = [], [], {}
    for game in sorted(mh.GAME_ARTIFACTS):
        src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS[game])
        actions = mh.load_actions(src)
        if not actions:
            continue
        norm = [(aid, d) for aid, d in (mh.normalize(a) for a in actions) if aid is not None]
        if not norm:
            continue
        # positives: every on-winning-path frame
        seq = _replay_seq(arc.make(game, scorecard_id=arc.open_scorecard()), game, actions, mh)
        next_up = _steps_to_next_up([lv for _, lv in seq])
        npos = 0
        for j, (f, _lv) in enumerate(seq):
            prev, action = _previous_context(seq, norm, j)
            goal = _goal_frame_for_index(seq, next_up, j)
            X.append([float(v) for v in _featurize_with_context(
                featurize, f, previous_frame=prev, action_id=action, goal_frame=goal)])
            y.append(1.0)
            npos += 1
        # negatives: from sampled on-path prefixes, take one non-gold action -> off-path frame
        k = min(neg_per_game, len(norm))
        idxs = sorted(rng.choice(len(norm), size=k, replace=False).tolist())
        nneg = 0
        for i in idxs:
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            f = env.reset()
            if game in mh.WARMUP_GAMES and actions:
                a0, d0 = mh.normalize(actions[0])
                if a0 is not None:
                    f = env.step(getattr(GameAction, f"ACTION{a0}"), data=d0)
            for aid, d in norm[:i]:
                f = env.step(getattr(GameAction, f"ACTION{aid}"), data=d)
                if f is None:
                    break
            if f is None:
                continue
            lvl0 = _levels_completed(f)
            gold = norm[i][0]
            choices = [a for a in _SIMPLE_ACTIONS if a != gold] + [6]  # +click
            aid2 = int(rng.choice(choices))
            data2 = {"x": int(rng.integers(0, 64)), "y": int(rng.integers(0, 64))} if aid2 == 6 else None
            f2 = env.step(getattr(GameAction, f"ACTION{aid2}"), data=data2)
            if f2 is None or _levels_completed(f2) > lvl0:   # None or accidentally won -> not a negative
                continue
            goal = _goal_frame_for_index(seq, next_up, min(i, len(seq) - 1))
            X.append([float(v) for v in _featurize_with_context(
                featurize, f2, previous_frame=f, action_id=aid2, goal_frame=goal)])
            y.append(0.0)
            nneg += 1
        per_game[game] = {"pos": npos, "neg": nneg}
    return X, y, per_game


def _auroc(scores, labels):
    """Rank-based AUROC (no sklearn). scores high => predicted positive."""
    pos = [s for s, l in zip(scores, labels) if l == 1.0]
    neg = [s for s, l in zip(scores, labels) if l == 0.0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1.0)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _arg_int(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return default


def _subset_rows(X, ranges):
    out = []
    for row in X:
        vals = []
        for lo, hi in ranges:
            vals.extend(row[lo:hi])
        out.append(vals)
    return out


def _discriminative_metrics(X, y, per_game):
    games = list(per_game)
    bounds, cur = {}, 0
    for g in sorted(per_game):
        n = per_game[g]["pos"] + per_game[g]["neg"]
        bounds[g] = (cur, cur + n)
        cur += n

    aurocs = []
    for held in games:
        lo, hi = bounds[held]
        if (hi - lo) < 4 or sum(y[lo:hi]) in (0, hi - lo):
            continue
        trX = X[:lo] + X[hi:]
        trY = y[:lo] + y[hi:]
        clf = DiscriminativeVerifier(lambda v: v).fit(trX, trY)
        Z = (np.asarray(X[lo:hi]) - clf.mu) / clf.sd
        Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        a = _auroc((Z @ clf.w).tolist(), y[lo:hi])
        if a == a:
            aurocs.append(a)

    mean_auroc = sum(aurocs) / len(aurocs) if aurocs else float("nan")
    full = DiscriminativeVerifier(lambda v: v).fit(X, y)
    Zi = np.hstack([(np.asarray(X) - full.mu) / full.sd, np.ones((len(X), 1))])
    in_sample = _auroc((Zi @ full.w).tolist(), y)
    return {
        "in_sample_auroc": in_sample,
        "loo_auroc": mean_auroc,
        "n_held_out_games": len(aurocs),
        "n_pos": int(sum(y)),
        "n_neg": int(len(y) - sum(y)),
    }


def _v3_feature_class_metrics(X, y, per_game):
    slices = cross_game_feature_slices_v3()
    v2_range = [slices["v2"]]
    v2_metrics = _discriminative_metrics(_subset_rows(X, v2_range), y, per_game)
    out = {"v2": v2_metrics["loo_auroc"], "v3_full": _discriminative_metrics(X, y, per_game)["loo_auroc"]}
    for name in ("object_relational", "frame_delta", "action_conditioned", "predicate_distance"):
        rows = _subset_rows(X, [slices["v2"], slices[name]])
        out[f"v2_plus_{name}"] = _discriminative_metrics(rows, y, per_game)["loo_auroc"]
    return out


def _reproduced_levels_from_registry():
    p = REPO / "ops" / "arc_solve_registry.yaml"
    if not p.exists():
        return 0
    return sum(int(m.group(1)) for m in re.finditer(r"levels_reproduced:\s*(\d+)", p.read_text()))


def _clean_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


_EXP4476_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with a terminal prefix complete:/complete_/success:/success_/passed:/passed_/"
        "shipped:/shipped_ so the reconciler classifies it as terminal (Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit declaration (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right floor."
    ),
    "offline_reproduced": (
        "a solve not reproducible offline is wasted effort -- only reproduced levels count "
        "(ARC Solve Reproducibility)."
    ),
    "reproduced_levels": (
        "headline metric reproducible_total_levels grows monotonically; report the count banked, "
        "real-env-confirmed."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified before launching; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
}


def _checksum_payload(payload):
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()


def _build_exp4476_artifact(
    v2_metrics,
    v3_metrics,
    feature_class_loo_auroc,
    value_head_routing_measure,
    tests_pass,
    preconditions_checked,
    reproduced_levels,
):
    target = 0.6
    v2 = _clean_float(v2_metrics.get("loo_auroc"))
    v3 = _clean_float(v3_metrics.get("loo_auroc"))
    loo_gate_passed = bool(v3 is not None and v3 >= target)
    if loo_gate_passed:
        honest_verdict = f"success: cross_game_features_v3_loo_auroc_{v3:.3f}_passes_gate"
    elif v2 is not None and v3 is not None and v3 > v2:
        honest_verdict = f"complete: cross_game_features_v3_honest_null_improved_{v2:.3f}_to_{v3:.3f}_below_gate"
    else:
        suffix = "nan" if v3 is None else f"{v3:.3f}"
        honest_verdict = f"complete: cross_game_features_v3_honest_null_no_transfer_{suffix}"

    feature_class_loo_auroc = {
        k: _clean_float(v) for k, v in sorted(feature_class_loo_auroc.items())
    }
    feature_class_deltas = {
        k: (None if v is None or v2 is None else float(v - v2))
        for k, v in feature_class_loo_auroc.items()
        if k != "v2"
    }
    payload = {
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "offline_reproduced": bool(reproduced_levels > 0),
        "reproduced_levels": int(reproduced_levels),
        "preconditions_checked": dict(preconditions_checked),
        "v2_baseline_loo_auroc": v2,
        "v2_baseline_in_sample_auroc": _clean_float(v2_metrics.get("in_sample_auroc")),
        "v3_loo_auroc": v3,
        "v3_in_sample_auroc": _clean_float(v3_metrics.get("in_sample_auroc")),
        "target_loo_auroc": target,
        "loo_gate_passed": loo_gate_passed,
        "feature_class_loo_auroc": feature_class_loo_auroc,
        "feature_class_deltas": feature_class_deltas,
        "value_head_routing_measure": dict(value_head_routing_measure),
        "tests_pass": bool(tests_pass),
        "field_principles": dict(_EXP4476_FIELD_PRINCIPLES),
        "spec_refs": ["REQ-LEARN-4476", "SCENARIO-LEARN-4476-FEATURES", "SCENARIO-LEARN-4476-GATE"],
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def _main_discriminative() -> int:
    """Train + LEAVE-ONE-GAME-OUT validate the win-reachability classifier on off-path negatives. The LOO
    AUROC answers the load-bearing question: do the OFF-PATH negatives carry TRANSFERABLE discriminative
    signal (does on-path vs off-path separate on a game the classifier never trained on)? AUROC>0.5 ==
    yes; this is the discrimination the steps-to-go regressor structurally cannot provide."""
    use_v2 = "--v2" in sys.argv
    feat = cross_game_features_v2 if use_v2 else cross_game_features_v3
    fname = "cross_game_features_v2" if use_v2 else "cross_game_features_v3"
    neg_per_game = _arg_int("--neg-per-game", 14)
    seed = _arg_int("--seed", 0)
    print(f"== DISCRIMINATIVE: win-reachability classifier on off-path negatives ({fname}) ==")
    X, y, per_game = collect_discriminative(featurize=feat, neg_per_game=neg_per_game, seed=seed)
    npos, nneg = int(sum(y)), int(len(y) - sum(y))
    print(f"  collected {npos} on-path POS + {nneg} off-path NEG from {len(per_game)} games")
    if nneg < 10 or npos < 10:
        print("  too few examples -- abort"); return 1

    metrics = _discriminative_metrics(X, y, per_game)
    mean_auroc = metrics["loo_auroc"]
    full = DiscriminativeVerifier(feat).fit(X, y)
    in_sample = metrics["in_sample_auroc"]
    print(f"  IN-SAMPLE AUROC (per-game feature ceiling): {in_sample:.3f}")
    print(f"  LEAVE-ONE-GAME-OUT AUROC (cross-game transfer): {mean_auroc:.3f} "
          f"over {metrics['n_held_out_games']} games")
    if in_sample > 0.6 and mean_auroc < 0.55:
        verdict = ("PER-GAME ONLY: off-path negatives ARE separable within a game (in-sample>0.6) but do "
                   "NOT transfer cross-game (loo~chance) -> train a discriminative head ONLINE per game "
                   "during exploration; a cross-game pre-trained head is NOT usable (needs richer "
                   "relational/delta/action-conditioned features, not just more negatives).")
    elif mean_auroc >= 0.55:
        verdict = "TRANSFERS: off-path negatives carry cross-game discriminative signal."
    else:
        verdict = "NULL: off-path negatives do not separate even in-sample -- negatives too weak or features too coarse."
    print(f"  VERDICT: {verdict}")
    ckpt = REPO / "models" / ("arc_discriminative_verifier_v2.json" if use_v2 else
                              "arc_discriminative_verifier_v3.json")
    games = list(per_game)
    full.save(ckpt, meta={"trained_games": games, "feature_names": fname,
                          "provenance": f"win-reachability classifier; in_sample_auroc={in_sample:.3f}, "
                                        f"loo_auroc={mean_auroc:.3f}. {verdict}"})
    out = REPO / "results" / ("arc_discriminative_verifier_v2.json" if use_v2 else
                              "arc_discriminative_verifier_v3.json")
    payload = {"n_pos": npos, "n_neg": nneg, "per_game": per_game,
               "in_sample_auroc": in_sample, "loo_auroc": mean_auroc,
               "n_held_out_games": metrics["n_held_out_games"], "verdict": verdict,
               "checkpoint": f"models/{ckpt.name}", "feature_names": fname,
               "honest_verdict": "complete_discriminative_cross_game_transfer_measured",
               "inference_substrate": "verifier_ensemble_against_cached_candidates",
               "mode": "discriminative_win_reachability_off_path_negatives"}
    out.write_text(json.dumps(payload, indent=2))
    (REPO / "results" / "arc_discriminative_verifier.json").write_text(json.dumps(payload, indent=2))

    if not use_v2:
        slices = cross_game_feature_slices_v3()
        v2_metrics = _discriminative_metrics(_subset_rows(X, [slices["v2"]]), y, per_game)
        feature_class_loo = _v3_feature_class_metrics(X, y, per_game)
        value_head_routing = {
            "ran": False,
            "artifact": "results/arc3_value_routing_v2.json",
            "note": "existing value-head routing gate is run separately; Exp4476 records this handle",
        }
        exp4476 = _build_exp4476_artifact(
            v2_metrics=v2_metrics,
            v3_metrics=metrics,
            feature_class_loo_auroc=feature_class_loo,
            value_head_routing_measure=value_head_routing,
            tests_pass=False,
            preconditions_checked={
                "banked_trajectories": bool(per_game),
                "offline_arcade": True,
                "feature_names": fname,
                "neg_per_game": neg_per_game,
                "seed": seed,
            },
            reproduced_levels=_reproduced_levels_from_registry(),
        )
        exp_out = REPO / "results" / "experiment_4476_verifier_features_v3_loo_gate.json"
        exp_out.write_text(json.dumps(exp4476, indent=2))
        print(f"  wrote {exp_out.relative_to(REPO)}")
    print(f"  checkpoint: models/{ckpt.name}; wrote {out.relative_to(REPO)}")
    return 0


def main() -> int:
    if "--discriminative" in sys.argv:
        return _main_discriminative()
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
