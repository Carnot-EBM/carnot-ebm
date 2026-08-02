#!/usr/bin/env python3
"""Power for the clustered design, computed BEFORE pre-registration and BEFORE any GPU time.

WHY IT IS HERE RATHER THAN IN THE WRITE-UP. "State minimum reachable p BEFORE results" is
only half an honest answer for a permutation test: with within-game label permutation the
number of distinct assignments is C(2R,R)^20, so the attainable minimum p is ~0 and quoting
it would be meaningless reassurance. The number that actually decides whether this run can
say anything is POWER at the observed control base rate, so that is what is computed.

Control base rate is taken from the pre-flight on the frozen 116-engine corpus
(pre/preflight_outcomes.json), NOT assumed.

The test simulated here is exactly the test the analysis will run: statistic = mean over
games of (rate_treatment - rate_control), null distribution by permuting the arm label
WITHIN each game.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NGAMES = 20


def stat(y: np.ndarray, arm: np.ndarray) -> float:
    """Mean over games of (treatment rate - control rate). y/arm are (games, cells)."""
    t = np.where(arm == 1, y, np.nan)
    c = np.where(arm == 0, y, np.nan)
    return float(np.nanmean(np.nanmean(t, axis=1) - np.nanmean(c, axis=1)))


def perm_p(y: np.ndarray, arm: np.ndarray, rng: np.random.Generator, n: int = 4000) -> float:
    obs = stat(y, arm)
    cnt = 0
    for _ in range(n):
        sh = np.array([rng.permutation(row) for row in arm])
        if abs(stat(y, sh)) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1)


def simulate(p_ctrl: float, p_trt: float, reps: int, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    arm = np.array([[0] * reps + [1] * reps] * NGAMES)
    hits = 0
    for _ in range(trials):
        # game-level heterogeneity: the pre-flight shows the outcome is concentrated in a
        # minority of games (variance in 9/20), so a homogeneous-Bernoulli sim would be
        # optimistic. Draw each game's rate from a Beta with the target mean and heavy
        # dispersion, which reproduces "most games flat, a few carry it".
        conc = 1.2
        gc = rng.beta(p_ctrl * conc, (1 - p_ctrl) * conc, NGAMES)
        gt = rng.beta(p_trt * conc, (1 - p_trt) * conc, NGAMES)
        y = np.zeros((NGAMES, 2 * reps))
        y[:, :reps] = (rng.random((NGAMES, reps)) < gc[:, None]).astype(float)
        y[:, reps:] = (rng.random((NGAMES, reps)) < gt[:, None]).astype(float)
        if perm_p(y, arm, rng, n=1500) < 0.05:
            hits += 1
    return hits / trials


def main() -> int:
    pre = json.loads((HERE / "pre" / "preflight_outcomes.json").read_text())
    ok = [r for r in pre if r.get("status") == "ok"]
    base = {}
    for k in ("O4_discriminates_heldout", "O2_fires_pre_win", "O1_fires_post_win"):
        base[k] = sum(1 for r in ok if r["outcomes"][k]) / len(ok)
    print(json.dumps({"control_base_rates_from_preflight": base}, indent=1))

    p_ctrl = base["O4_discriminates_heldout"]
    out = {"p_ctrl": round(p_ctrl, 4), "grid": []}
    for reps in (3, 4):
        for p_trt in (0.20, 0.30, 0.40, 0.50):
            pw = simulate(p_ctrl, p_trt, reps, trials=300, seed=hash((reps, p_trt)) % 2**31)
            out["grid"].append({"reps": reps, "p_trt": p_trt, "power": round(pw, 3)})
            print(f"  reps={reps} p_trt={p_trt:.2f}  power={pw:.3f}")
    (HERE / "pre" / "power.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
