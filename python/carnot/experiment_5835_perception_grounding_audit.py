#!/usr/bin/env python3
"""PERCEPTION-GROUNDING AUDIT (REQ-ARC-PERC-5835): does an OBJECT-LEVEL frame representation predict
cross-game progress where the shipped ORDER-1 (frame-only, position/color-stat) representation sits at
chance? (operator directive 2026-07-24 "Let's target perception next").

WHY. Two live A/Bs just came back null -- the generator switch (exp5834: dense-27B == 9B on live levels)
and goal-exemplar grading (exp5832) -- both because the agent can't perceive well enough to reach the
boundaries where induction/goal-grading would bite. The standing diagnosis (GAP-4891 /
project_arc_live_agent_learning_gaps) is that the shipped featurizer -- cross_game_features_v2, 41
frame-only order-1 features -- sits at LEAVE-ONE-GAME-OUT chance for predicting progress, which is why the
live value head is pinned inert (SUBMITTED_VALUE_WEIGHT=1e-12). BUT there is NO committed numeric artifact
for that claim (it's prose in known-issues + docstrings). Before building/shipping the (already-built,
shipped-OFF) classical color-blob/salience front-end into the live path, this AUDIT measures, on a clean
labeled corpus, whether an object-level representation carries the cross-game progress signal order-1
lacks. Diagnostic-first, so we don't ship a 4th null.

METHOD. Substrate = the 144-trajectory human-replay corpus (all 25 public games), staged shards yielding
{frame, level_progress in [0,1], env}. Two FRAME-ONLY feature sets: (A) order1 = cross_game_features_v2
(41 features, the shipped representation); (B) object = a frame-only object vector from
connected_color_blobs + translation-invariant shape signatures + ColorBlobSaliencePrior tiers (the classical
front-end's perception). Label = level_progress (binary >=0.5 for AUC; continuous for Spearman). Evaluation
= LEAVE-ONE-GAME-OUT (train on 24 games, test on the held-out game) -- this IS the generalization measurement
the ARC generalization floor calls for. Metric = mean LOO AUC + mean LOO Spearman per feature set.

GATE (pre-registered): object mean-LOO-AUC exceeds order1 mean-LOO-AUC by >= 0.05 AND object AUC >= 0.60,
while order1 AUC ~ chance (<= 0.56). Clears -> object-level perception is the confirmed fix direction
(justifies solving the m0r0 regression to ship the front-end). Else HONEST-NEGATIVE: object representation
also fails to carry cross-game progress signal -> perception REPRESENTATION is not the lever either; the
bottleneck is deeper (dynamics/goal induction, or the label), needs a rethink -- do NOT ship the front-end
on this evidence.

inference_substrate: aggregation_from_upstream_artifacts (CPU feature-regression on a cached corpus; NO LLM,
NO GPU). verifier_is_oracle: False. solve_provenance: development_proxy. NEVER submits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

CORPUS_DIR = ROOT / "data" / "arc_public_demo_human_replay_corpus"
CAP_PER_GAME = int(os.environ.get("PERC_CAP_PER_GAME", "3000"))  # subsample cap per game for speed
PROGRESS_THRESH = float(os.environ.get("PERC_PROGRESS_THRESH", "0.5"))
OUT = ROOT / "results" / "experiment_5835_perception_grounding_audit.json"


def _object_features(frame) -> list[float]:
    """Frame-only OBJECT-LEVEL representation: the classical color-blob/salience front-end's perception.
    Translation-invariant shape diversity is the direct GAP-4891 counter to order-1 position-only stats."""
    from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior, connected_color_blobs

    try:
        blobs = connected_color_blobs(frame)
    except Exception:
        blobs = []
    if not blobs:
        return [0.0] * 54  # 18 abstractions + 36 salient-object occupancy map
    prior = ColorBlobSaliencePrior()
    n = len(blobs)
    fa = 1.0
    fs = blobs[0].frame_shape
    if fs:
        fa = float(max(1, int(fs[0]) * int(fs[1])))
        diag = float(max(1, max(int(fs[0]), int(fs[1]))))
    else:
        diag = 1.0
    areafr = [b.area_fraction for b in blobs]
    heights = [float(b.height) for b in blobs]
    widths = [float(b.width) for b in blobs]
    aspects = [float(b.height) / max(1.0, float(b.width)) for b in blobs]
    colors = {int(b.color) for b in blobs}
    cxs = [float(b.centroid[1]) for b in blobs]
    cys = [float(b.centroid[0]) for b in blobs]

    def _shape_sig(b) -> int:
        ys = [y for (y, _x) in b.cells]
        xs = [x for (_y, x) in b.cells]
        y0, x0 = min(ys), min(xs)
        return hash(frozenset((y - y0, x - x0) for (y, x) in b.cells))

    shapes = {_shape_sig(b) for b in blobs}
    tiers = [int(prior.tier(b)) for b in blobs]
    tier_hist = [sum(1 for t in tiers if t == k) / n for k in range(5)]
    # 6x6 SALIENT-OBJECT occupancy map -> spatial fairness vs v2's raw-nonzero 6x6 map. Salient = tier<=2
    # (buttons/interactables, not status-bars/backgrounds), so this is a SALIENCE-FILTERED spatial signature,
    # not a copy of the raw-occupancy map. Isolates whether object/salience abstraction adds signal.
    salient_map = [0.0] * 36
    if fs:
        h, w = int(fs[0]), int(fs[1])
        sal = np.zeros((h, w), dtype=float)
        for b, t in zip(blobs, tiers):
            if t <= 2:
                for (yy, xx) in b.cells:
                    if 0 <= yy < h and 0 <= xx < w:
                        sal[yy, xx] = 1.0
        idx = 0
        for by in range(6):
            for bx in range(6):
                y0, y1 = h * by // 6, max(h * by // 6 + 1, h * (by + 1) // 6)
                x0, x1 = w * bx // 6, max(w * bx // 6 + 1, w * (bx + 1) // 6)
                block = sal[y0:y1, x0:x1]
                salient_map[idx] = float(block.mean()) if block.size else 0.0
                idx += 1
    return [
        n / 32.0,                                   # object count (normalized)
        len(shapes) / max(1, n),                    # distinct-shape fraction (translation-invariant)
        len(shapes) / 32.0,                         # distinct-shape count (normalized)
        len(colors) / 16.0,                         # color diversity
        float(np.mean(areafr)), float(np.max(areafr)), float(np.std(areafr)),  # size distribution
        float(np.mean(aspects)), float(np.std(aspects)),                       # shape aspect
        float(np.std(cxs)) / diag, float(np.std(cys)) / diag,                  # object spatial spread
        float(np.mean(heights)) / diag, float(np.mean(widths)) / diag,         # object scale
        *tier_hist,                                 # 5 salience-tier fractions
    ] + salient_map                                 # + 36 salient-object occupancy = 54 total


def main() -> int:
    t0 = time.time()
    from carnot.agentic.arc_human_replay_corpus import load_training_shards
    from carnot.agentic.arc_value_learner import cross_game_features_v2

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import spearmanr
    except Exception as e:
        art = {
            "experiment": "experiment_5835_perception_grounding_audit",
            "honest_verdict": f"complete: blocked_missing_dependency_{type(e).__name__}",
            "error": str(e)[:200], "duration_s": round(time.time() - t0, 2),
        }
        OUT.write_text(json.dumps(art, indent=2))
        print("BLOCKED: missing sklearn/scipy ->", e)
        return 0

    # ---- Load corpus, group by game, subsample per game. ----
    by_game: dict[str, dict[str, list]] = {}
    n_seen = 0
    n_skipped = 0
    for ex in load_training_shards(CORPUS_DIR):
        env = str(ex.get("env") or "")
        frame = ex.get("frame")
        lp = ex.get("level_progress")
        if not env or frame is None or lp is None:
            n_skipped += 1
            continue
        g = by_game.setdefault(env, {"order1": [], "object": [], "y": [], "lp": []})
        if len(g["y"]) >= CAP_PER_GAME:
            continue
        try:
            f1 = cross_game_features_v2(frame)
            fo = _object_features(frame)
        except Exception:
            n_skipped += 1
            continue
        g["order1"].append(f1)
        g["object"].append(fo)
        g["lp"].append(float(lp))
        g["y"].append(1 if float(lp) >= PROGRESS_THRESH else 0)
        n_seen += 1

    games = sorted(by_game.keys())
    # Precompute concatenated arrays per game.
    for env in games:
        g = by_game[env]
        g["order1"] = np.asarray(g["order1"], dtype=float)
        g["object"] = np.asarray(g["object"], dtype=float)
        g["combo"] = np.concatenate([g["order1"], g["object"]], axis=1) if len(g["y"]) else np.zeros((0, 0))
        g["y"] = np.asarray(g["y"], dtype=int)
        g["lp"] = np.asarray(g["lp"], dtype=float)

    ARMS = ["order1", "object", "combo"]

    def _loo(arm: str) -> dict:
        aucs, spears, folds = [], [], []
        for held in games:
            Xtr = np.concatenate([by_game[e][arm] for e in games if e != held], axis=0)
            ytr = np.concatenate([by_game[e]["y"] for e in games if e != held], axis=0)
            Xte = by_game[held][arm]
            yte = by_game[held]["y"]
            lpte = by_game[held]["lp"]
            if len(np.unique(ytr)) < 2 or len(yte) < 20:
                continue
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
            prob = clf.predict_proba(sc.transform(Xte))[:, 1]
            auc = float(roc_auc_score(yte, prob)) if len(np.unique(yte)) == 2 else None
            sp = spearmanr(prob, lpte).correlation if len(lpte) > 5 else None
            folds.append({"game": held, "n": int(len(yte)),
                          "auc": None if auc is None else round(auc, 4),
                          "spearman": None if sp is None or (isinstance(sp, float) and math.isnan(sp))
                          else round(float(sp), 4)})
            if auc is not None:
                aucs.append(auc)
            if sp is not None and not (isinstance(sp, float) and math.isnan(sp)):
                spears.append(float(sp))
        return {
            "mean_loo_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "mean_loo_spearman": round(float(np.mean(spears)), 4) if spears else None,
            "n_folds_scored": len(aucs), "per_game": folds,
        }

    results = {arm: _loo(arm) for arm in ARMS}

    a1 = results["order1"]["mean_loo_auc"]
    ao = results["object"]["mean_loo_auc"]
    ac = results["combo"]["mean_loo_auc"]
    auc_delta_obj = round(ao - a1, 4) if isinstance(a1, float) and isinstance(ao, float) else None
    gate_clears = bool(
        isinstance(a1, float) and isinstance(ao, float)
        and (ao - a1) >= 0.05 and ao >= 0.60 and a1 <= 0.56
    )

    art = {
        "experiment": "experiment_5835_perception_grounding_audit",
        "experiment_id": "REQ-ARC-PERC-5835",
        "run_date": "2026-07-24",
        "title": "Perception-grounding audit: does an object-level frame representation predict cross-game "
                 "progress where order-1 (frame-only) sits at chance? Leave-one-game-out on the human-replay corpus.",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "operator_directive": "2026-07-24 'Let's target perception next' -- fresh directive resuming the "
                              "perception line under the ARC generalization-testing floor (leave-one-game-out).",
        "random_seed": 5835,
        "config": {"corpus": str(CORPUS_DIR.relative_to(ROOT)), "cap_per_game": CAP_PER_GAME,
                   "progress_threshold": PROGRESS_THRESH, "n_games": len(games),
                   "n_examples_used": int(n_seen), "n_skipped": int(n_skipped),
                   "arms": {"order1": "cross_game_features_v2 (41 frame-only)",
                            "object": "connected_color_blobs + translation-invariant shapes + salience tiers "
                                      "+ 6x6 salient-object occupancy map (54)",
                            "combo": "order1 + object"}},
        "methodology_note": (
            "Leave-one-game-out logistic regression predicting level_progress>=%.2f from FRAME-ONLY features on "
            "the 25-public-game human-replay corpus (cap %d/game). AUC + Spearman(prob, continuous level_progress) "
            "averaged over held-out games. order1 = the shipped cross_game_features_v2; object = the classical "
            "color-blob/salience front-end's frame perception. NO LLM, NO GPU. Produces the missing committed "
            "baseline number for the GAP-4891 order-1=LOO-chance claim." % (PROGRESS_THRESH, CAP_PER_GAME)
        ),
        "per_arm": results,
        "order1_mean_loo_auc": a1,
        "object_mean_loo_auc": ao,
        "combo_mean_loo_auc": ac,
        "object_minus_order1_auc": auc_delta_obj,
        "gate_clears": gate_clears,
        "games": games,
        "duration_s": round(time.time() - t0, 1),
    }
    art["honest_verdict"] = (
        f"complete_perception_grounding_audit_order1_auc_{a1}_object_auc_{ao}"
        f"_delta_{auc_delta_obj}_gate_clears_{gate_clears}"
    )
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    OUT.write_text(json.dumps(art, indent=2, default=str))
    print(f"\n=== PERCEPTION AUDIT: order1 LOO-AUC={a1} | object LOO-AUC={ao} (delta {auc_delta_obj}) | "
          f"combo={ac} | GATE_CLEARS={gate_clears} ===")
    print(f"games={len(games)} examples={n_seen} skipped={n_skipped}")
    print("wrote", OUT, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
