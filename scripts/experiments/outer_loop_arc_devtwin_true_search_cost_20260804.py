"""Measure the TRUE expansion count of the adapter-free generic explorer (exp6094).

WHY THIS EXISTS
---------------
`scripts/experiments/outer_loop_arc_devtwin_adapterfree_cell_20260803.py` recorded
`search_cost = args.max_expansions` unconditionally in BOTH treatment branches (its lines 166
and 176) and never passed `stats=` to `graph_explore_solve_v2`. So every adapter-free row in
exp6093 carried the search BUDGET, not a measurement, and 15 of them additionally carried the
label `max_expansions_exhausted` -- an assertion the code could not make, because the only thing
it observed was `traj is None`. The control arm's `states_expanded` was always real, so the
per-game table was printing a measurement and a constant side by side as if they were
comparable.

`graph_explore_solve_v2` will report the real expansion count if handed a `stats` dict. This
script re-runs the identical call with `stats={}` and prints what the search actually did, one
JSON object per invocation.

WHAT IT DOES NOT DO
-------------------
It does not call `solve_via_explore`, which would reproduce, train a verifier, and seed an
adapter -- i.e. it would WRITE. This is read-only against the offline arcade: no reproduce, no
verifier training, no adapter seeding, nothing persisted. CPU only, no generator, no GPU, no
game played.

Usage:  python outer_loop_arc_devtwin_true_search_cost_20260804.py <game> <max_expansions>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Derive the repo root from this file rather than hardcoding an absolute path (CLAUDE.md
# "Test-Run Record Integrity Discipline" rule 4: an absolute path baked into source is a defect,
# and independently a G2 reproducibility defect because a fresh clone would point at the
# operator's checkout).
REPO = Path(__file__).resolve().parents[2]

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # this probe must never take a GPU
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    game = argv[1]
    budget = int(argv[2])

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    stats: dict = {}
    t0 = time.time()
    traj, lvl = graph_explore_solve_v2(
        env, 0, max_expansions=budget, max_depth=60, warmup=False, stats=stats
    )
    wall = time.time() - t0

    expansions = stats.get("expansions")
    print(
        json.dumps(
            {
                "game": game,
                "budget": budget,
                "true_expansions": expansions,
                "distinct_states": stats.get("states"),
                # The distinction the original cell could not draw: a search that STOPPED because
                # it ran out of budget (a depth limit) versus one that stopped because its
                # frontier emptied (a representation limit). Only the second is a collapse.
                "budget_exhausted": expansions is not None and int(expansions) >= budget,
                "advanced": traj is not None,
                "reached_level": lvl,
                "wall_s": round(wall, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
