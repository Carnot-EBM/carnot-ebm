#!/usr/bin/env python3
"""Power of the stage-1 game-clustered permutation test at the REALISED design.

PROVENANCE, STATED PLAINLY. This was written and run AFTER the first LLM call of the measured
run, so it is NOT part of the pre-registration and is not presented as such. Its inputs are
exclusively the FROZEN-CORPUS pre-flight (`pre/frozen_corpus_clusters.json`, 116 world models
from a run that finished on 2026-08-01) and the realised design constants. NOT ONE CELL OF THE
MEASURED RUN ENTERS IT. It exists because the pre-registration expressed power as an
effect-size argument in prose, and a prose argument is worth less than a simulation when the
question is "would this design have seen a real effect".

WHY BETA-HETEROGENEOUS AND NOT HOMOGENEOUS BERNOULLI. A homogeneous simulation assumes every
game has the same underlying rate, which is optimistic and flatters the design: the real
pre-flight shows a few games carrying most of the tropes and most games near zero. Games are
drawn with per-game rates from a Beta matched to the pre-flight mean, so the simulated
between-game variance is the variance the analysis will actually face.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

N_GAMES = 20
REPS = 3  # the realised stage-1 replicate count after the wall-budget revision
# 600 simulated experiments per effect size. The Monte-Carlo standard error on a power
# estimate near 0.5 is then ~0.02, which is finer than any decision this table informs; the
# inner permutation is the expensive loop, so buying more sims costs real time for no change
# in what the table says.
SIMS = 600
PERM_DRAWS = 2000  # inner permutation draws; enough to resolve alpha=0.05 in a power sim


def _beta_params(mean: float, conc: float = 2.0) -> tuple[float, float]:
    """Beta(a, b) with the requested mean and a deliberately LOW concentration, i.e. high
    between-game spread. Low concentration is the pessimistic choice."""
    return max(mean * conc, 0.05), max((1 - mean) * conc, 0.05)


def one_trial(p_ctrl: float, p_trt: float) -> float:
    """Simulate one experiment and return its permutation p-value."""
    a_c, b_c = _beta_params(p_ctrl)
    a_t, b_t = _beta_params(p_trt)
    ctrl, trt = [], []
    for _ in range(N_GAMES):
        # The SAME game draws a control rate and a treatment rate whose ranks are coupled, so a
        # game that is trope-prone in control is trope-prone in treatment too. Independent draws
        # would wash out between-game correlation and overstate power.
        u = RNG.random()
        from scipy.stats import beta as _beta  # noqa: PLC0415

        pc = float(_beta.ppf(u, a_c, b_c))
        pt = float(_beta.ppf(u, a_t, b_t))
        ctrl.append(RNG.random(REPS) < pc)
        trt.append(RNG.random(REPS) < pt)
    ctrl = [c.astype(float) for c in ctrl]
    trt = [t.astype(float) for t in trt]
    obs = float(np.mean([t.mean() - c.mean() for t, c in zip(trt, ctrl, strict=True)]))
    acc = np.zeros(PERM_DRAWS)
    for t, c in zip(trt, ctrl, strict=True):
        pool = np.concatenate([t, c])
        idx = np.argsort(RNG.random((PERM_DRAWS, len(pool))), axis=1)
        perm = pool[idx]
        acc += perm[:, : len(t)].mean(axis=1) - perm[:, len(t) :].mean(axis=1)
    acc /= N_GAMES
    return float((np.sum(np.abs(acc) >= abs(obs) - 1e-12) + 1) / (PERM_DRAWS + 1))


def main() -> int:
    frozen = json.loads((HERE / "frozen_corpus_clusters.json").read_text())
    split = [r for r in frozen if r["split"]]
    trope_base = sum(
        1 for r in split if r["cluster"] in {"C_UNIFORMITY", "B_COLOUR_ELIMINATION"}
    ) / len(split)
    decl_base = sum(1 for r in split if r["cluster"] in {"A_DECLINED", "D_NO_PREDICATE"}) / len(
        split
    )
    out: dict = {
        "provenance": "computed AFTER the run started, from FROZEN-CORPUS pre-flight inputs "
        "only; no cell of the measured run enters this file",
        "design": {
            "n_games": N_GAMES,
            "replicates_per_arm": REPS,
            "sims_per_effect_size": SIMS,
            "monte_carlo_se_near_p0.5": round((0.25 / SIMS) ** 0.5, 4),
            "inner_permutation_draws": PERM_DRAWS,
            "alpha": 0.05,
        },
        "base_rates_from_the_23_frozen_split_induce_cells": {
            "trope": round(trope_base, 4),
            "declined": round(decl_base, 4),
            "why_the_split_cells": "the goal-only prompt is the prompt used on split-induce "
            "cells, so those 23 are the closest frozen analogue of a stage-1 control cell",
        },
        "power": {},
    }
    for label, base, targets in (
        ("TROPE", trope_base, [0.40, 0.30, 0.20, 0.10, 0.05]),
        ("DECLINED", decl_base, [0.30, 0.20, 0.10, 0.05, 0.02]),
    ):
        rows = {}
        for tgt in targets:
            ps = [one_trial(base, tgt) for _ in range(SIMS)]
            rows[f"{base:.3f}->{tgt:.2f}"] = round(float(np.mean(np.asarray(ps) < 0.05)), 3)
        out["power"][label] = rows
    out["reading"] = (
        "Read the DECLINED row first: it is the PRIMARY. Its control base rate on the frozen "
        "split cells is low, so only a large absolute reduction is detectable and a null on the "
        "primary is weak evidence against a small effect. The TROPE row is where the design has "
        "real power, which is why the pre-registration named it as the secondary the mechanism "
        "can actually bite on."
    )
    (HERE / "power.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
