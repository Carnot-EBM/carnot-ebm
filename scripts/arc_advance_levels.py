"""Outer-loop INCREMENTAL-PROGRESS driver: for every ARC-AGI-3 game we already
solve, pin the known winning trajectory as a PREFIX and explore only the frontier
BEYOND it to advance +1 level (reproduction-gated). Per CLAUDE.md "ARC-AGI-3
Incremental-Progress Scoping" (advance +1, bank progress) and "ARC Solve
Reproducibility" (only reproduced levels count). Zero quota (offline sim).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels
from carnot.agentic.arc_agi3_live_adapter import _game_action

# game -> (current best reproduced level, prefix source, key-path into the json)
SOLVED = {
    "r11l": (1, "results/arc_explore_trajectory_r11l.json", ("trajectory",)),
    "ls20": (1, "results/arc_explore_trajectory_ls20.json", ("trajectory",)),
    "cd82": (1, "results/arc_explore_trajectory_cd82.json", ("trajectory",)),
    "sp80": (1, "results/arc_explore_trajectory_sp80.json", ("trajectory",)),
    "su15": (1, "results/arc_explore_trajectory_su15.json", ("trajectory",)),
    "tu93": (1, "results/arc_explore_trajectory_tu93.json", ("trajectory",)),
    "lp85": (3, "results/arc3_lp85_offline_resolve.json", ("solution",)),
    "wa30": (1, "results/experiment_4275_arc_incremental_progress_new_game.json",
             ("solve_trace", "actions")),
}
SUFFIX_DEPTH = 45        # actions to explore BEYOND the prefix
MAX_EXPANSIONS = 4000


def _dig(d: dict, keys: tuple) -> list:
    cur = d
    for k in keys:
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            return []
    return cur if isinstance(cur, list) else []


def _normalize_step(a: dict) -> dict:
    """-> {"action": int, "data": {x,y}|None} for any of our recorded formats."""
    if "data" in a:                                   # arc_explore_trajectory format
        return {"action": int(a["action"]), "data": a.get("data")}
    x = a.get("x", a.get("world_x"))
    y = a.get("y", a.get("world_y"))
    has_xy = x is not None and y is not None
    aid = a.get("action")
    if aid is None:
        aid = 6 if has_xy else None
    data = {"x": int(x), "y": int(y)} if has_xy else None
    return {"action": int(aid), "data": data}


def load_prefix(src: str, keys: tuple) -> list:
    d = json.load(open(REPO / src))
    return [_normalize_step(s) for s in _dig(d, keys)]


def apply(env, label, frame):
    s = json.loads(label)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def main() -> int:
    print(f"== incremental-progress: advance +1 level on {len(SOLVED)} solved games ==", flush=True)
    advanced, results = [], {}
    for game, (best, src, keys) in SOLVED.items():
        try:
            prefix = load_prefix(src, keys)
            if not prefix:
                results[game] = {"error": "empty prefix"}
                print(f"  {game}: empty prefix from {src}", flush=True)
                continue
            # sanity: prefix actually reaches `best` offline
            pgate = kit.reproduce(game, trajectory_labels(prefix), apply, claimed_level=best)
            if not pgate["reproduced"]:
                results[game] = {"prefix_reaches": pgate["reached_level"], "prefix_ok": False}
                print(f"  {game}: prefix does NOT reproduce L{best} (reached L{pgate['reached_level']}) — skip", flush=True)
                continue
            arc = kit.offline_arcade()
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            t0 = time.time()
            traj, lvl = graph_explore_solve_v2(
                env, start_level=best, prefix=prefix,
                max_expansions=MAX_EXPANSIONS, max_depth=len(prefix) + SUFFIX_DEPTH)
            dt = time.time() - t0
            if traj and lvl > best:
                gate = kit.reproduce(game, trajectory_labels(traj), apply, claimed_level=lvl)
                ok = bool(gate["reproduced"])
                results[game] = {"from_level": best, "reached_level": lvl,
                                 "moves": len(traj), "offline_reproduced": ok}
                if ok:
                    advanced.append((game, best, lvl, len(traj)))
                    (REPO / "results" / f"arc_advance_trajectory_{game}_L{lvl}.json").write_text(
                        json.dumps({"game": game, "reached_level": lvl, "trajectory": traj}, indent=2))
                print(f"  {game}: L{best} -> L{lvl} in {len(traj)} actions (reproduced={ok}) [{dt:.0f}s]", flush=True)
            else:
                results[game] = {"from_level": best, "no_advance": True}
                print(f"  {game}: no-advance beyond L{best} [{dt:.0f}s]", flush=True)
        except Exception as e:
            results[game] = {"error": repr(e)[:120]}
            print(f"  {game}: ERROR {repr(e)[:80]}", flush=True)

    print(f"\n== RESULT: {len(advanced)} game(s) advanced +>=1 level ==", flush=True)
    for g, frm, lvl, mv in advanced:
        print(f"  + {g}: L{frm} -> L{lvl} ({mv} actions)", flush=True)
    (REPO / "results" / "arc_advance_levels.json").write_text(
        json.dumps({"games_attempted": len(SOLVED), "advanced": advanced, "results": results}, indent=2))
    print("  wrote results/arc_advance_levels.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
