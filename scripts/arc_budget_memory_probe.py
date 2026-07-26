"""ONE cell in ONE clean process, reporting the MEMORY the retained search graph costs.

WHY A SEPARATE PROCESS PER CELL. CPython does not return freed arena pages to the OS reliably, so a
second cell in the same process reuses the first's freed pages and its delta reads far too low. One
process per cell is the only way to attribute an RSS delta to that cell.

WHY THE DECOMPOSITION IS  shared_libs + n_games * per_game  AND NOT  import_time_rss + delta.
The competition framework's `Swarm.main()` (agents/swarm.py:76-99) builds one agent + one Thread per
game and starts EVERY thread before joining any, so all N games are live in ONE address space at
once. What is SHARED across those threads is the interpreter plus the imported libraries; what is
PER-THREAD is that game's env object, its policy, and its retained search graph. So the two terms
must be split at the boundary "everything importable" vs "everything the cell allocates".

THE MEASUREMENT DEFECT THIS AVOIDS (found while writing it). The first version snapshotted RSS
right after `import arc_scored_path_lever_harness` and reported the delta as the graph cost. That
read 846 MiB at budget 400 -- because the harness imports numpy, arc_leaderboard_eval, arcengine and
the per-game env module LAZILY, inside `run_cell` (lines 606-610). The 846 MiB was overwhelmingly
library import cost, which is SHARED across the swarm's threads and must not be multiplied by 110.
This version imports every one of those modules EAGERLY before the baseline snapshot, so the
baseline is the genuinely shared term and the delta is genuinely per-thread.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))


def rss_kib() -> int:
    """CURRENT RSS in KiB. ru_maxrss is a high-water mark, which is the wrong term for a set of
    concurrent threads that each RETAIN their graph to the end of the run."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)


def peak_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    a = ap.parse_args(argv)

    # EAGER import of everything run_cell imports lazily, so the baseline below is the SHARED term.
    import arc_scored_path_lever_harness as harness  # noqa: E402
    import numpy  # noqa: E402,F401
    import arc_leaderboard_eval  # noqa: E402,F401
    import arcengine  # noqa: E402,F401
    import arc_agi.scorecard  # noqa: E402,F401
    from carnot.agentic import arc_competition_agent  # noqa: E402,F401
    from carnot.agentic import arc_game_adapters  # noqa: E402,F401

    shared_libs_rss = rss_kib()
    base_peak = peak_kib()
    t0 = time.time()
    row = harness.run_cell(
        a.game,
        a.seed,
        budget=a.budget,
        proposer=None,
        llm=False,
        extra_kwargs=dict(harness.ARMS["S"]),
        arm=f"S_llmoff_mem_b{a.budget}",
    )
    after_rss = rss_kib()
    after_peak = peak_kib()
    print(
        json.dumps(
            {
                "game": a.game,
                "seed": a.seed,
                "budget": a.budget,
                "ran": row.get("ran"),
                "levels": row.get("levels"),
                "actions": row.get("actions"),
                "nodes_total": row.get("nodes_total"),
                "nodes_with_frame": row.get("nodes_with_frame"),
                "wall_s": round(time.time() - t0, 2),
                "shared_libs_rss_mib": round(shared_libs_rss / 1024.0, 1),
                "after_rss_mib": round(after_rss / 1024.0, 1),
                # PER-THREAD term: env + policy + retained graph. This is what gets multiplied by n_games.
                "per_game_delta_mib": round((after_rss - shared_libs_rss) / 1024.0, 1),
                "per_game_delta_peak_mib": round((after_peak - base_peak) / 1024.0, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
