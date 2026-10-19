"""Within-game depth fit: train on other levels of ONE game, test on a held-out level of it.

Leave-one-level-out ridge on banked solve traces. CPU only, offline sim.
Usage: python <this file> <out.json>. See docs/research-notes/depth-calibration-cross-game-verifier-2026-10-18.md."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))
OUT = Path(sys.argv[1])
import arc_cross_game_verifier_train as trainer
from carnot.agentic.arc_value_learner import cross_game_features, cross_game_features_v3


def spearman(a, b):
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return lambda Q: ((Q - mu) / sd) @ w + y.mean()


res = {}
for name, feat in [("v1", cross_game_features), ("v3", cross_game_features_v3)]:
    X, y, per_game, dists = trainer.collect_pooled(featurize=feat, variants=0, return_dist=True)
    X = np.array(X, float)
    y = np.array(y, float)
    dd = np.array(dists, float)
    games = np.concatenate([[g] * n for g, n in per_game.items()])
    # a level segment ends at the row whose steps-to-level-up is 0
    seg = np.zeros(len(y), int)
    for g in per_game:
        idx = np.where(games == g)[0]
        s = 0
        for i in idx:
            seg[i] = s
            if dd[i] == 0:
                s += 1
    pred = np.full_like(y, np.nan)
    base = np.full_like(y, np.nan)
    for g in per_game:
        for s in np.unique(seg[games == g]):
            te = (games == g) & (seg == s)
            tr = (games == g) & (seg != s)
            if tr.sum() < 20 or len(np.unique(seg[tr])) < 2:
                continue
            pred[te] = np.clip(ridge(X[tr], y[tr], 10.0)(X[te]), 0, 1)
            base[te] = y[tr].mean()
    ok = ~np.isnan(pred)
    per = {}
    for g in per_game:
        m = ok & (games == g)
        if m.sum() >= 10:
            per[g] = dict(
                n=int(m.sum()),
                spearman=spearman(pred[m], y[m]),
                mae_model=float(np.abs(pred[m] - y[m]).mean()),
                mae_const=float(np.abs(base[m] - y[m]).mean()),
            )
    # rank inside each held-out level: the question the live explorer asks
    within = []
    for g in per_game:
        for s in np.unique(seg[games == g]):
            m = ok & (games == g) & (seg == s)
            if m.sum() >= 8:
                within.append(spearman(pred[m], y[m]))
    within = np.array([w for w in within if w is not None])
    sp = np.array([v["spearman"] for v in per.values() if v["spearman"] is not None])
    res[name] = dict(
        n_scored=int(ok.sum()),
        n_games_scored=len(per),
        pooled_spearman=spearman(pred[ok], y[ok]),
        mae_model=float(np.abs(pred[ok] - y[ok]).mean()),
        mae_const=float(np.abs(base[ok] - y[ok]).mean()),
        per_game_spearman_median=float(np.median(sp)),
        per_game_spearman_positive=int((sp > 0).sum()),
        n_levels_within=int(len(within)),
        within_level_spearman_median=float(np.median(within)),
        within_level_spearman_positive=int((within > 0).sum()),
        per_game=per,
    )
    print(name, json.dumps({k: v for k, v in res[name].items() if k != "per_game"}), flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE", flush=True)
