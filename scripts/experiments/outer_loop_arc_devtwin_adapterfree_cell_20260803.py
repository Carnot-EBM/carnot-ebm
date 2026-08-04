#!/usr/bin/env python3
"""ONE CELL of the DEV-TWIN half of the adapter-free measurement: first-level acquisition,
hand-built `GameAdapter` vs the generic adapter-free explorer.

WHY THIS EXISTS ALONGSIDE THE SCORED-PATH CELL. `arc_game_adapters` is ABSENT from the scored
`E3AgentPolicy` import closure (independently verified: 55 files, adapters not among them), so
"turn the adapter off" is a no-op there. But the 183/183 public clear was NOT produced by the
scored path -- it was produced by `scripts/arc_loop_solve.py`, the offline dev twin, THROUGH
those adapters. So the dev twin is where the hand-tuning actually lives and is the only place
the question "how much of 183/183 is the adapter?" can be asked at all.

`scripts/arc_loop_solve.py` is one of the two canonical live entrypoints named in CLAUDE.md's
ARC Live-Path Reachability Discipline, and it already ships the two arms as first-class modes:

  control_adapter_on    `solve_adaptered(game, target_level=1)` -- verifier-routed best-first
                        search over the adapter's hand-built action vocabulary / state key /
                        warmup / depth cap, warm-started from `models/arc_verifier_<game>.json`
                        when one exists (a checkpoint trained on THAT game's own solve traces).
  treatment_adapter_free `solve_via_explore(game)` -- the `--ignore-adapter` path: generic
                        graph-explore first contact, no per-game vocabulary, no per-game
                        verifier, no warmup.

MATCHED ON THE OBJECTIVE, NOT ON THE SEARCH BUDGET, and this is stated rather than hidden. Both
arms are asked for exactly the same thing: bank the FIRST level from level 0, reproduction-gated.
Their search budgets are each arm's own shipped configuration (the adapter's `depth_caps` vs
graph-explore's `max_expansions=6000 / max_depth=60`), because forcing one arm onto the other's
budget would be measuring the budget, not the knowledge. The search COST of each arm is recorded
next to its outcome so the reader can see what the outcome cost.

TRACKED STATE IS NOT WRITTEN. Both `arc_loop_solve` paths write by default:
`models/arc_verifier_<game>.json` (a tracked checkpoint -- `solve_via_explore` OVERWRITES it,
which would destroy the control arm's own warm start and mutate the research record),
`results/arc_explore_trajectory_<game>.json`, plus `gap_fills/<game>_goal_distance.py` and
`ops/arc_router_ledger.json` via `arc_heuristic_select.select_and_learn`. All four are
redirected or neutralized here. Per CLAUDE.md's Test-Run Record Integrity Discipline a
measurement must not rewrite the record it is measuring.

Neutralizing `select_and_learn` cannot change the outcome: it runs only AFTER the reproduction
gate has already returned, and its return value is reported (`heuristic_learned`) but never read
back into the solve.

CPU ONLY. No generator, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True)
    ap.add_argument(
        "--arm", required=True, choices=["control_adapter_on", "treatment_adapter_free"]
    )
    ap.add_argument("--target-level", type=int, default=1)
    ap.add_argument("--max-expansions", type=int, default=6000)
    ap.add_argument("--max-depth", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t0 = time.time()
    row: dict[str, Any] = {
        "game": args.game,
        "arm": args.arm,
        "target_level": args.target_level,
        "status": "started",
    }

    # The temp tree lives INSIDE the repo, deliberately. `arc_loop_solve` reports its written
    # paths as `ckpt.relative_to(REPO)`, which raises for any path outside the checkout -- so a
    # /tmp redirect crashes the very function under measurement. Keeping it under the repo (and
    # deleting it in `finally`) redirects the writes without patching `loop.REPO`, which is also
    # what `load_live_spatial_value_head(root=REPO, ...)` reads and must stay pointed at the real
    # checkout. Prefix is distinctive so a leftover from a killed process is obvious in
    # `git status` and cannot be mistaken for a deliverable.
    tmp = tempfile.mkdtemp(prefix=f".arc_devtwin_tmp_{args.game}_{args.arm}_", dir=str(REPO))
    os.environ["CARNOT_ARC_E3_DIR"] = str(Path(tmp) / "e3")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    sys.path.insert(0, str(REPO / "python"))
    sys.path.insert(0, str(REPO / "scripts"))
    import logging

    logging.disable(logging.INFO)

    try:
        import arc_loop_solve as loop

        # ---- REDIRECT EVERY WRITE INTO THE TEMP DIR ------------------------------------------
        # `_ckpt_path` and `solve_via_explore` both build their paths from these two module
        # globals, so repointing them is sufficient for the checkpoint + trajectory writes.
        real_ckpt_dir = loop.CKPT_DIR
        loop.CKPT_DIR = Path(tmp) / "models"
        loop.RESULTS = Path(tmp) / "results"
        loop.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        loop.RESULTS.mkdir(parents=True, exist_ok=True)
        # The CONTROL legitimately reads the game's own learned verifier checkpoint -- that is
        # part of the per-game knowledge under test (leak C2). Copy it in so the control keeps
        # its warm start while writing only to temp.
        if args.arm == "control_adapter_on":
            src = real_ckpt_dir / f"arc_verifier_{args.game}.json"
            if src.is_file():
                shutil.copy2(src, loop.CKPT_DIR / src.name)
                row["control_warm_start_checkpoint_present"] = True
            else:
                row["control_warm_start_checkpoint_present"] = False

        # `select_and_learn` writes `gap_fills/<game>_goal_distance.py` and appends to
        # `ops/arc_router_ledger.json`, both tracked. It runs strictly AFTER the reproduction
        # gate and feeds nothing back into the solve, so neutralizing it cannot change any
        # number reported here.
        from carnot.agentic import arc_heuristic_select as hsel

        hsel.select_and_learn = lambda *a, **k: None  # type: ignore[assignment]
        row["select_and_learn_neutralized"] = True

        if args.arm == "control_adapter_on":
            from carnot.agentic import arc_game_adapters as adapters

            if adapters.get_adapter(args.game) is None:
                # Named, in the denominator, never silently dropped.
                row["status"] = "blocked_no_adapter_registered"
                row["cell_wall_s"] = round(time.time() - t0, 2)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(row, indent=2))
                shutil.rmtree(tmp, ignore_errors=True)
                print(json.dumps({k: row[k] for k in ("game", "arm", "status")}), flush=True)
                return 1
            out = loop.solve_adaptered(args.game, args.target_level)
            row["result"] = {
                "reached_level": out.get("reached_level"),
                "moves": out.get("moves"),
                "search_cost": out.get("states_expanded"),
                "search_cost_unit": "states_expanded",
                "offline_reproduced": out.get("offline_reproduced"),
                "reproduced_levels": out.get("reproduced_levels"),
                "verifier_src": out.get("verifier_src"),
                "reproduction_gate": out.get("reproduction_gate"),
            }
        else:
            out = loop.solve_via_explore(
                args.game, max_expansions=args.max_expansions, max_depth=args.max_depth
            )
            if out is None:
                # `solve_via_explore` returns None when the generic explorer never advanced a
                # level within its budget. That is the measurement, not a failure.
                row["result"] = {
                    "reached_level": 0,
                    "moves": None,
                    "search_cost": args.max_expansions,
                    "search_cost_unit": "max_expansions_exhausted",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "no_advance": True,
                }
            else:
                row["result"] = {
                    "reached_level": out.get("reached_level"),
                    "moves": out.get("moves"),
                    "search_cost": args.max_expansions,
                    "search_cost_unit": "max_expansions_budget",
                    "offline_reproduced": out.get("offline_reproduced"),
                    "reproduced_levels": out.get("reproduced_levels"),
                    "no_advance": False,
                }
        row["banked_levels"] = (
            int(row["result"].get("reproduced_levels") or 0)
            if row["result"].get("offline_reproduced")
            else 0
        )
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "blocked_cell_exception"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["traceback"] = traceback.format_exc()[-1500:]
    finally:
        row["cell_wall_s"] = round(time.time() - t0, 2)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(row, indent=2))
        shutil.rmtree(tmp, ignore_errors=True)

    print(
        json.dumps(
            {
                "game": args.game,
                "arm": args.arm,
                "status": row["status"],
                "banked": row.get("banked_levels"),
                "reached": (row.get("result") or {}).get("reached_level"),
                "wall_s": row["cell_wall_s"],
            }
        ),
        flush=True,
    )
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
