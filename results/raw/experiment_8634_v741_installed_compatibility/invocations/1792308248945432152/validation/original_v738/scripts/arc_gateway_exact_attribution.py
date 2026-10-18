#!/usr/bin/env python3
"""EXACT per-level reset attribution: a live measurement that resolves the bound.

WHY A LIVE RE-RUN IS NECESSARY
------------------------------
The 1401 persisted early-stop rows record only a WHOLE-RUN `n_resets`. The
gateway's per-level charge is a DIFFERENCE of cumulative CHARGED counts, so
recovering the correction needs the resets attributed PER LEVEL. From the rows
alone the answer is only a bound, and that bound is wide (a vc33 b4000 cell:
offline 2.0897 vs worst case 0.2254 -- an 89% swing, useless as a conclusion).

`scripts/arc_leaderboard_eval.py:run_game` is now instrumented to record
`resets_before_levelups` / `level_up_charged` / `efficiency_gateway_charged`
(2026-07-26). This driver re-runs MATCHED cells (same game, budget, seed, same
shipped flag configuration) through that instrumentation and reports, per cell:

  * `efficiency`                    -- offline accounting, resets FREE (the unit
                                       every historical number is in)
  * `efficiency_gateway_charged`    -- gateway accounting, resets CHARGED (the
                                       only unit the competition score uses)
  * the bound from the persisted row, so we can state whether reality sits at
    the optimistic end, the pessimal end, or in between.

CLOCKS. This script performs a LIVE measurement: substrate
`offline_arcade_live_agent_runtime_self_discovery_no_llm`, and it publishes its
own `measurement_wall_s`. It is NOT an aggregation pass over persisted rows.

It never submits anything and never rewrites a historical artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "python")):
    if p not in sys.path:
        sys.path.insert(0, p)


def run_one(game: str, seed: int, budget: int) -> dict:
    """One live cell through the instrumented run_game.

    MATCHED to the recorded `S_llmoff_*` arm, cell for cell, per
    `arc_scored_path_lever_harness.run_cell`: `CARNOT_ARC_DISABLE_INDUCTION=1`
    (LLM off -- the recorded rows all carry `llm_enabled: false`), `random.seed`
    + `np.random.seed(seed % 2**32)`, and `frontier_discipline_seed=seed`. An
    UNMATCHED re-run would compare two different agents and the reset delta
    would be confounded by the configuration difference. It also stalls: with
    induction enabled the policy blocks on an llama-server health check.
    """
    import random

    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    t0 = time.time()
    policy = E3AgentPolicy(game, frontier_discipline_seed=seed)
    r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
    wall = time.time() - t0

    # The reset count must agree with the frames-vs-actions identity, or the
    # instrumentation is not seeing every reset.
    frames = len(r.get("frame_sequence") or [])
    charged = int(r.get("charged_actions") or 0)
    actions = int(r.get("actions") or 0)
    resets = int(r.get("n_resets_run_game") or 0)
    return {
        "game": game,
        "seed": seed,
        "budget": budget,
        "wall_s": round(wall, 3),
        "levels": r.get("levels"),
        "reached": r.get("reached"),
        "offline_actions": actions,
        "n_resets": resets,
        "charged_actions": charged,
        "frames_recorded": frames,
        # identity assertions -- a dead channel must not read as a clean null
        "identity_charged_eq_actions_plus_resets": charged == actions + resets,
        "level_up_actions_offline": r.get("per_level")
        and [p["agent_actions"] for p in r["per_level"] if p.get("completed")],
        "resets_before_levelups": r.get("resets_before_levelups"),
        "level_up_charged": r.get("level_up_charged"),
        "efficiency_offline": r.get("efficiency"),
        "efficiency_gateway_charged": r.get("efficiency_gateway_charged"),
        "efficiency_optimism_vs_gateway": r.get("efficiency_optimism_vs_gateway"),
        "per_level": r.get("per_level"),
        "per_level_gateway": r.get("per_level_gateway"),
        "baselines_nonzero": bool(
            r.get("per_level") and any(p.get("human_actions") for p in r["per_level"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="vc33,sp80,tu93,cd82,lp85")
    ap.add_argument("--budgets", default="400")
    ap.add_argument("--seeds", default="20260724")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    games = [g for g in args.games.split(",") if g]
    budgets = [int(b) for b in args.budgets.split(",") if b]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    t0 = time.time()
    cells = []
    # BUDGET IS THE INNERMOST LOOP (CLAUDE.md: the swept parameter must be
    # innermost, so drift over wall-clock cannot masquerade as a budget effect).
    for game in games:
        for seed in seeds:
            for budget in budgets:
                try:
                    cells.append(run_one(game, seed, budget))
                except Exception as exc:  # a failure is recorded, never silent
                    cells.append(
                        {
                            "game": game,
                            "seed": seed,
                            "budget": budget,
                            "error": repr(exc)[:300],
                        }
                    )
                c = cells[-1]
                print(
                    f"{game} b={budget} s={seed} lv={c.get('levels')} "
                    f"res={c.get('n_resets')} off={c.get('efficiency_offline')} "
                    f"gw={c.get('efficiency_gateway_charged')} "
                    f"rbl={c.get('resets_before_levelups')} wall={c.get('wall_s')}",
                    flush=True,
                )
    wall = time.time() - t0
    out = {
        "measurement": "arc_gateway_exact_attribution",
        "measurement_wall_s": round(wall, 3),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "n_cells": len(cells),
        "cells": cells,
        "rows_checksum": hashlib.sha256(
            json.dumps(cells, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    path = args.out or os.path.join(
        REPO, "results/early_stop_sweep_20260726/rows_exact_attribution.json"
    )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {path} ({len(cells)} cells, {wall:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
