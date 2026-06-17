"""Verifier-routed search demo (the north-star efficiency loop): the SAME offline
solver, with vs without a verifier guiding the search, on lp85. The verifier is a
goal-distance ENERGY on the game state (LOWER = closer to the win); best-first
ordering by it should expand FEWER states than plain BFS to reach each level —
the verifier-routed action-efficiency win, now standing in the harness
(arc_solver_kit.OfflineSolver). Zero quota.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction

from carnot.agentic import arc_solver_kit as kit
from carnot.experiment_4179_arc_incremental_progress import (
    discover_click_buttons, _goal_key, _target_goal_key,
)

GAME = "lp85"


def action_labels(env):
    # env-DISCOVERED action vocabulary (gotcha #5), encoded as click coords
    return [json.dumps({"x": int(b["x"]), "y": int(b["y"])}) for b in discover_click_buttons(env)]


def apply(env, label, frame):
    a = json.loads(label)
    return env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})


def state_key(game):
    return _goal_key(game)  # goal-relevant dedup (not full grid) — keeps lp85 tractable


def verifier(game):
    """Goal-distance ENERGY: how far the actual goal sprites are from where the
    win wants them (each piece's +1,+1). 0 ⇒ win. This is the score the search
    descends — the verifier routing the solver."""
    actual = _goal_key(game)            # [(type, x, y), ...]
    target = _target_goal_key(game)     # [(type, x+1, y+1), ...]
    by_type = defaultdict(list)
    for t, x, y in actual:
        by_type[t].append((x, y))
    total = 0.0
    for t, tx, ty in target:
        cands = by_type.get(t, [])
        total += min((abs(tx - x) + abs(ty - y)) for x, y in cands) if cands else 1000.0
    return total


def run(use_verifier: bool, target_level: int = 3):
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        GAME, action_labels, apply, state_key,
        verifier=(verifier if use_verifier else None),
    )
    # solve level-by-level, summing states expanded
    f = solver._replay(env, [])
    cur = kit.frame_level(f)
    full, total_states = [], 0
    per_level = {}
    for lvl in range(cur + 1, target_level + 1):
        path, nodes = solver.solve_level(env, cur, full, depth_cap={1: 20, 2: 70, 3: 90}.get(lvl, 90))
        total_states += nodes
        per_level[f"L{lvl}"] = nodes
        if path is None:
            break
        f = solver._replay(env, full + path)
        cur = kit.frame_level(f)
        full += path
    return {"reached": cur, "moves": len(full), "states_expanded": total_states, "per_level": per_level}


def main() -> int:
    print("== verifier-routed search demo: lp85 (BFS vs verifier-routed best-first) ==")
    bfs = run(use_verifier=False)
    print(f"  plain BFS            : reached L{bfs['reached']} in {bfs['moves']} moves, "
          f"{bfs['states_expanded']} states expanded {bfs['per_level']}")
    vr = run(use_verifier=True)
    print(f"  verifier-routed best : reached L{vr['reached']} in {vr['moves']} moves, "
          f"{vr['states_expanded']} states expanded {vr['per_level']}")
    if bfs["reached"] == vr["reached"] and bfs["states_expanded"] > 0:
        ratio = bfs["states_expanded"] / max(1, vr["states_expanded"])
        print(f"\n  EFFICIENCY: verifier-routed search expanded {vr['states_expanded']} vs "
              f"{bfs['states_expanded']} states -> {ratio:.2f}x fewer (same level reached)")
    out = REPO / "results" / "arc3_verifier_routed_search_demo.json"
    out.write_text(json.dumps({"game": GAME, "bfs": bfs, "verifier_routed": vr,
                               "mode": "offline_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
