"""From-scratch OFFLINE solver for lp85 (priority: make deeper ARC solves
offline-reproducible). No replay of live recordings — BFS over the offline env
(a deterministic simulator) using env-adaptive button discovery, checking the
real levels_completed signal. Zero quota.

lp85: click-only [ACTION6]; win = every moveable piece aligned with its goal
sprite at (x+1, y+1). L1 budget = 13 moves.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

from carnot.agentic.arc_agi3_live_adapter import _levels_completed
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
from carnot.experiment_4179_arc_incremental_progress import discover_click_buttons

MAX_DEPTH = 16          # L1 budget is 13; allow a little slack
MAX_NODES = 40000


def make_env(arcade):
    env = arcade.make("lp85", scorecard_id=arcade.open_scorecard())
    return env, env.reset()


def actions_at(env) -> list[dict]:
    try:
        return discover_click_buttons(env)
    except Exception:
        return []


def replay(env, path: list[dict]):
    f = env.reset()
    for b in path:
        f = env.step(GameAction.ACTION6, data={"x": int(b["x"]), "y": int(b["y"])})
    return f


def main() -> int:
    print("== lp85 FROM-SCRATCH offline BFS solver (zero quota) ==")
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    env, f0 = make_env(arc)
    print(f"reset: level={_levels_completed(f0)} buttons@base={[b.get('button') for b in actions_at(env)]}")

    TARGET = 3                  # the deeper level we need reproducible
    BUDGET = {1: 16, 2: 64, 3: 84}  # per-level move budget (survey + slack)
    full_solution: list[dict] = []
    cur_level = 0
    while cur_level < TARGET:
        # BFS from the current solved prefix to advance exactly one level
        prefix = list(full_solution)
        base_f = replay(env, prefix)
        seen = {frame_hash(grid_of(base_f))}
        frontier = deque([[]])
        nodes = 0
        found = None
        depth_cap = BUDGET.get(cur_level + 1, 84)
        while frontier and nodes < MAX_NODES and found is None:
            path = frontier.popleft()
            if len(path) >= depth_cap:
                continue
            replay(env, prefix + path)
            for b in actions_at(env):
                f2 = env.step(GameAction.ACTION6, data={"x": int(b["x"]), "y": int(b["y"])})
                nodes += 1
                if f2 is None:
                    env = make_env(arc)[0]
                    continue
                if _levels_completed(f2) >= cur_level + 1:
                    found = path + [b]
                    break
                h = frame_hash(grid_of(f2))
                if h not in seen:
                    seen.add(h)
                    frontier.append(path + [b])
                replay(env, prefix + path)
            else:
                continue
        if found is None:
            print(f"\n  STUCK at L{cur_level} -> L{cur_level+1}: no path in {nodes} nodes "
                  f"(depth_cap={depth_cap}, states_seen={len(seen)})")
            break
        full_solution += found
        cur_level += 1
        print(f"  solved L{cur_level}: +{len(found)} moves (total {len(full_solution)}), {nodes} nodes")

    print(f"\n  lp85 FROM-SCRATCH offline result: reached L{cur_level} in {len(full_solution)} moves")
    if cur_level >= 1:
        import json
        out = REPO / "results" / "arc3_lp85_offline_resolve.json"
        out.write_text(json.dumps({
            "game": "lp85", "reached_level": cur_level, "moves": len(full_solution),
            "solution": [{"action": 6, "x": int(b["x"]), "y": int(b["y"]), "button": b.get("button")}
                         for b in full_solution],
            "mode": "from_scratch_offline_bfs_no_quota",
        }, indent=2))
        print(f"  wrote {out.relative_to(REPO)}")
    return 0 if cur_level >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
