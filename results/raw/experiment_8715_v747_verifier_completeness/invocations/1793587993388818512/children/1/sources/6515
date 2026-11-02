"""Second-stage diagnostic: does a SYSTEMATIC BFS (graph_explore_solve_v2)
reach a level-up where the random/hybrid walk did not? This is the decisive
STRUCTURE-vs-(BUDGET|CANDIDATES) discriminator:

  - BFS finds the win that random missed  => STRUCTURE (depth-first/greedy is
    the bottleneck; a systematic frontier finds it).
  - BFS also fails but reaches MANY states => BUDGET (deeper/longer needed) or
    the win-region is genuinely unreached.
  - BFS fails AND the state space is tiny/exhausted => CANDIDATES (winning
    action TYPE not in the set).

Also reports: best_level reached, expansions used, distinct states in the BFS
graph, and whether the budget was hit (capped) vs the frontier emptied
(exhausted).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "python")

from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

GAMES = {
    "ls20": "ls20-9607627b",
    "wa30": "wa30-ee6fef47",
    "su15": "su15-1944f8ab",
    "tu93": "tu93-0768757b",
    "cn04": "cn04-2fe56bfb",
    "m0r0": "m0r0-492f87ba",
    "sk48": "sk48-d8078629",
}


def run(short, gid, max_expansions=8000, max_depth=60):
    ar = offline_arcade()
    env = ar.make(gid, save_recording=False, include_frame_data=True)
    stats: dict = {}
    t0 = time.time()
    traj, lvl = graph_explore_solve_v2(
        env, start_level=0, max_expansions=max_expansions,
        max_depth=max_depth, warmup=False, stats=stats,
    )
    wall = round(time.time() - t0, 2)
    solved = traj is not None
    return {
        "game": short,
        "game_id": gid,
        "bfs_solved": solved,
        "bfs_reached_level": int(lvl),
        "bfs_traj_len": (len(traj) if traj else 0),
        "bfs_expansions": stats.get("expansions"),
        "bfs_distinct_states": stats.get("states"),
        "max_expansions": max_expansions,
        "budget_capped": (stats.get("expansions", 0) >= max_expansions) and not solved,
        "wall_s": wall,
    }


def main():
    max_exp = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    only = sys.argv[2] if len(sys.argv) > 2 else None
    out = {}
    for short, gid in GAMES.items():
        if only and short != only:
            continue
        print(f"### BFS {short} ...", flush=True)
        try:
            res = run(short, gid, max_expansions=max_exp)
        except Exception as e:
            import traceback
            res = {"game": short, "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-800:]}
        out[short] = res
        print(json.dumps(res, indent=2), flush=True)
    with open("results/diag_hardtail_bfs.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("WROTE results/diag_hardtail_bfs.json", flush=True)


if __name__ == "__main__":
    main()
