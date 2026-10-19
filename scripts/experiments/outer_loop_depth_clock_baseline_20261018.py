"""Clock baseline for the within-game depth fit: is the frame signal more than a step counter?

Same leave-one-level-out split as outer_loop_depth_within_game_20261018.py. Compares ridge on
(a) the clock only (steps since the level started), (b) v1 frame features, (c) v1 plus the clock.
CPU only. Usage: python <this file> <out.json>."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))
OUT = Path(sys.argv[1])
import arc_cross_game_verifier_train as trainer
from carnot.agentic.arc_value_learner import cross_game_features


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


X, y, per_game, dists = trainer.collect_pooled(
    featurize=cross_game_features, variants=0, return_dist=True
)
X = np.array(X, float)
y = np.array(y, float)
dd = np.array(dists, float)
games = np.concatenate([[g] * n for g, n in per_game.items()])
seg = np.zeros(len(y), int)
clock = np.zeros(len(y))
for g in per_game:
    s, t = 0, 0
    for i in np.where(games == g)[0]:
        seg[i], clock[i] = s, t
        t += 1
        if dd[i] == 0:
            s, t = s + 1, 0
C = clock[:, None]
sets = {"clock_only": C, "v1": X, "v1_plus_clock": np.hstack([X, C])}
out = {}
for name, F in sets.items():
    pred = np.full_like(y, np.nan)
    base = np.full_like(y, np.nan)
    for g in per_game:
        for s in np.unique(seg[games == g]):
            te = (games == g) & (seg == s)
            tr = (games == g) & (seg != s)
            if tr.sum() < 20 or len(np.unique(seg[tr])) < 2:
                continue
            pred[te] = np.clip(ridge(F[tr], y[tr], 10.0)(F[te]), 0, 1)
            base[te] = y[tr].mean()
    ok = ~np.isnan(pred)
    per = {}
    for g in per_game:
        m = ok & (games == g)
        if m.sum() >= 10:
            per[g] = dict(
                spearman=spearman(pred[m], y[m]),
                mae_model=float(np.abs(pred[m] - y[m]).mean()),
                mae_const=float(np.abs(base[m] - y[m]).mean()),
            )
    out[name] = dict(
        n_scored=int(ok.sum()),
        pooled_spearman=spearman(pred[ok], y[ok]),
        mae_model=float(np.abs(pred[ok] - y[ok]).mean()),
        mae_const=float(np.abs(base[ok] - y[ok]).mean()),
        per_game=per,
    )
    print(name, json.dumps({k: v for k, v in out[name].items() if k != "per_game"}), flush=True)
    for g, v in per.items():
        print("  ", g, json.dumps(v), flush=True)
OUT.write_text(json.dumps(out, indent=1))
print("DONE", flush=True)
