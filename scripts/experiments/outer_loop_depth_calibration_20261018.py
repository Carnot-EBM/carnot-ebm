"""Does a cross-game distance estimate get less reliable far from the goal?

Leave-one-game-out ridge fit on pooled banked solve traces. CPU only, offline sim.
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
    print(f"collect {name}", flush=True)
    X, y, per_game, dists = trainer.collect_pooled(featurize=feat, variants=0, return_dist=True)
    X = np.array(X, float)
    y = np.array(y, float)
    games = np.concatenate([[g] * n for g, n in per_game.items()])
    assert len(games) == len(y), (len(games), len(y))
    print(name, X.shape, per_game, flush=True)
    pred = np.zeros_like(y)
    base = np.zeros_like(y)
    for g in per_game:
        te = games == g
        f = ridge(X[~te], y[~te], 10.0)
        pred[te] = np.clip(f(X[te]), 0, 1)
        base[te] = y[~te].mean()
    dd = np.array(dists, float)  # raw steps to the next level-up
    ly = np.log1p(dd)
    lp = np.zeros_like(ly)
    for g in per_game:
        te = games == g
        lp[te] = ridge(X[~te], ly[~te], 10.0)(X[te])
    lb = np.array([ly[games != g].mean() for g in games])
    raw = []
    for lo, hi in [(0, 2), (2, 6), (6, 16), (16, 1e9)]:
        m = (dd >= lo) & (dd < hi)
        raw.append(
            dict(
                raw_steps_bucket=[lo, hi],
                n=int(m.sum()),
                mae_log_model=float(np.abs(lp[m] - ly[m]).mean()),
                mae_log_const=float(np.abs(lb[m] - ly[m]).mean()),
                bias_log=float((lp[m] - ly[m]).mean()),
                spearman_within=spearman(lp[m], ly[m]),
            )
        )
    edges = [0, 0.25, 0.5, 0.75, 1.0000001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y >= lo) & (y < hi)
        rows.append(
            dict(
                true_depth_bucket=[lo, min(hi, 1.0)],
                n=int(m.sum()),
                mae_model=float(np.abs(pred[m] - y[m]).mean()),
                mae_mean_baseline=float(np.abs(base[m] - y[m]).mean()),
                bias_pred_minus_true=float((pred[m] - y[m]).mean()),
                spearman_within=spearman(pred[m], y[m]),
            )
        )
    rel = []
    qs = np.quantile(pred, np.linspace(0, 1, 6))
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (pred >= lo) & (pred <= hi)
        rel.append(
            dict(
                pred_range=[float(lo), float(hi)],
                n=int(m.sum()),
                mean_pred=float(pred[m].mean()),
                mean_true=float(y[m].mean()),
            )
        )
    pg = {
        g: dict(
            n=int((games == g).sum()),
            spearman=spearman(pred[games == g], y[games == g]),
            mae=float(np.abs(pred[games == g] - y[games == g]).mean()),
        )
        for g in per_game
    }
    res[name] = dict(
        n=int(len(y)),
        n_games=len(per_game),
        overall_spearman=spearman(pred, y),
        overall_mae_model=float(np.abs(pred - y).mean()),
        overall_mae_baseline=float(np.abs(base - y).mean()),
        by_true_depth=rows,
        raw_depth_logfit=raw,
        overall_spearman_log=spearman(lp, ly),
        dist_raw=dict(min=float(dd.min()), median=float(np.median(dd)), max=float(dd.max())),
        reliability_by_pred_quintile=rel,
        per_game=pg,
    )
    print(json.dumps(res[name], indent=1), flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE", flush=True)
