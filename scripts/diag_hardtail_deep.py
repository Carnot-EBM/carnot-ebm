"""Third-stage: for the 4 games where 8k-BFS was budget-capped (wa30, cn04,
m0r0, sk48), distinguish BUDGET (deeper/longer systematic search finds it)
from deep-STRUCTURE (the win-region is unreached even by a larger systematic
search; a goal heuristic is required). Runs:
  - higher-budget pure BFS (30000 expansions)
  - A*-heuristic BFS with misplaced_region_distance toward the L0 win-grid.

The win grid for the heuristic is unknown a-priori for an unsolved game, so we
ONLY use the heuristic where the registry records a known solve length (m0r0,
sk48) by attempting the heuristic with a 'reach any level-up' objective via the
plain higher-budget BFS; the A* arm here is the pure-novelty escalation. The
decisive output is simply: does ANY systematic arm reach a level-up, and how
many states/expansions did the win require?
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "python")

from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.agentic.arc_graph_explore import (
    graph_explore_solve_v2,
    graph_explore_solve_v3,
)

CAPPED = {
    "wa30": "wa30-ee6fef47",
    "cn04": "cn04-2fe56bfb",
    "m0r0": "m0r0-492f87ba",
    "sk48": "sk48-d8078629",
}


def run_bfs(gid, max_expansions, max_depth):
    ar = offline_arcade()
    env = ar.make(gid, save_recording=False, include_frame_data=True)
    stats: dict = {}
    t0 = time.time()
    traj, lvl = graph_explore_solve_v2(
        env, start_level=0, max_expansions=max_expansions,
        max_depth=max_depth, warmup=False, stats=stats,
    )
    return {
        "solved": traj is not None,
        "level": int(lvl),
        "traj_len": len(traj) if traj else 0,
        "expansions": stats.get("expansions"),
        "states": stats.get("states"),
        "capped": (stats.get("expansions", 0) >= max_expansions) and traj is None,
        "wall_s": round(time.time() - t0, 1),
    }


def run_v3(gid, max_expansions, max_depth):
    """v3 = value/novelty-guided best-first; only frame-CHANGING transitions
    enqueued (skips wall-bump no-ops), depth-biased to push deeper. Designed
    for deep games where uniform BFS exhausts before the win."""
    ar = offline_arcade()
    env = ar.make(gid, save_recording=False, include_frame_data=True)
    stats: dict = {}
    t0 = time.time()
    traj, lvl = graph_explore_solve_v3(
        env, start_level=0, max_expansions=max_expansions,
        max_depth=max_depth, warmup=False, stats=stats,
    )
    return {
        "solved": traj is not None,
        "level": int(lvl),
        "traj_len": len(traj) if traj else 0,
        "expansions": stats.get("expansions"),
        "states": stats.get("states"),
        "capped": (stats.get("expansions", 0) >= max_expansions) and traj is None,
        "wall_s": round(time.time() - t0, 1),
    }


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out = {}
    for short, gid in CAPPED.items():
        if only and short != only:
            continue
        print(f"### DEEP {short} ...", flush=True)
        res = {"game": short, "game_id": gid}
        try:
            res["bfs_30k_d120"] = run_bfs(gid, 30000, 120)
        except Exception as e:
            res["bfs_30k_d120"] = {"error": f"{type(e).__name__}: {e}"}
        print(short, "bfs30k:", json.dumps(res["bfs_30k_d120"]), flush=True)
        try:
            res["v3_novelty_30k_d120"] = run_v3(gid, 30000, 120)
        except Exception as e:
            res["v3_novelty_30k_d120"] = {"error": f"{type(e).__name__}: {e}"}
        print(short, "v3:", json.dumps(res["v3_novelty_30k_d120"]), flush=True)
        out[short] = res
    with open("results/diag_hardtail_deep.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("WROTE results/diag_hardtail_deep.json", flush=True)


if __name__ == "__main__":
    main()
