"""From-scratch OFFLINE solver for sc25 (priority: deeper ARC solves
offline-reproducible). No replay of live recordings.

sc25 mechanics (reverse-engineered from environment_files/sc25/.../sc25.py):
  - 3x3 cast grid: clicking the `clzbxlm-sptivk-slsrhr` cell sprite at coords
    (24+5c, 49+5r) TOGGLES xhhaqjfncnp[r][c] (camera is identity). cast = select
    `sptivk-{spell}` sprite, set xhhaqjfncnp==zzpoabuniyn[spell], re-click to fire.
  - player `pluyoo` (= self.plnqvukupu, auto-set on level load) moves via
    ACTION1/2/3/4; reaching the exit `exydhv` triggers next_level. Step budget.
  - NOTE: env._game deepcopy-injection is BROKEN for sc25 (unlike lp85), so we
    BFS by REPLAY-FROM-RESET on the real env; the level is read from the FRAME.
Zero quota.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

from carnot.agentic.arc_agi3_live_adapter import _levels_completed

TARGET = 5
MAX_NODES = 30000
DEPTH = {1: 24, 2: 30, 3: 30, 4: 30, 5: 30}
PHASES = ["vbublqskwzw", "ggotuphkheh", "obrrczymkxn", "wmnlnlscbpq", "jwlqyoqyagv", "agzbtzaakna"]
MOVE_IDS = (1, 2, 3, 4)


def busy(g):
    return any(getattr(g, p, {}).get("acyylh") for p in PHASES) or getattr(g, "eycwbtepcvs", False)


def resolve(env, f):
    for _ in range(80):
        if not busy(env._game):
            break
        f = env.step(GameAction.ACTION5)
    return f


def spell_sprite(g, spell):
    return next((s for s in g.current_level.get_sprites()
                 if str(getattr(s, "name", "")) == f"sptivk-{spell}"), None)


def castable_spells(g):
    pats = getattr(g, "zzpoabuniyn", {})
    cur = g.xhhaqjfncnp
    return [sp for sp, pat in pats.items()
            if spell_sprite(g, sp) is not None
            and any(bool(pat[r][c]) != bool(cur[r][c]) for r in range(3) for c in range(3))]


def do_cast(env, spell, f):
    ss = spell_sprite(env._game, spell)
    if ss is None:
        return f
    f = env.step(GameAction.ACTION6, data={"x": int(ss.x), "y": int(ss.y)})
    pat = env._game.zzpoabuniyn[spell]
    for r in range(3):
        for c in range(3):
            if bool(pat[r][c]) != bool(env._game.xhhaqjfncnp[r][c]):
                f = env.step(GameAction.ACTION6, data={"x": 24 + 5 * c, "y": 49 + 5 * r})
    ss = spell_sprite(env._game, spell)
    if ss is not None:
        f = env.step(GameAction.ACTION6, data={"x": int(ss.x), "y": int(ss.y)})
    return resolve(env, f)


def do_move(env, m, f):
    f = env.step(getattr(GameAction, f"ACTION{m}"))
    return resolve(env, f)


def apply(env, label, f):
    if label.startswith("cast:"):
        return do_cast(env, label.split(":", 1)[1], f)
    return do_move(env, int(label[-1]), f)


def replay(env, path):
    f = env.reset()
    f = env.step(GameAction.ACTION5)  # the first action after reset is consumed (no-op); warm it up
    for label in path:
        f = apply(env, label, f)
    return f


def state_key(g):
    p = getattr(g, "plnqvukupu", None)
    pp = (int(p.x), int(p.y)) if p else None
    facing = getattr(g, "jdmucabyqar", None)  # player facing — turn vs move is state-dependent
    sprites = tuple(sorted((str(getattr(s, "name", "")), int(getattr(s, "x", 0)), int(getattr(s, "y", 0)))
                           for s in g.current_level.get_sprites()))
    return (pp, facing, str(g.xhhaqjfncnp), sprites)


def actions(env):
    return ["cast:" + sp for sp in castable_spells(env._game)] + [f"move{m}" for m in MOVE_IDS]


def solve_one(env, start_level, depth_cap, prefix):
    replay(env, prefix)
    seen = {state_key(env._game)}
    frontier = deque([[]])
    nodes = 0
    while frontier and nodes < MAX_NODES:
        path = frontier.popleft()
        if len(path) >= depth_cap:
            continue
        replay(env, prefix + path)
        for label in actions(env):
            f2 = apply(env, label, None)
            nodes += 1
            if _levels_completed(f2) > start_level:
                return path + [label], nodes
            k = state_key(env._game)
            if k not in seen:
                seen.add(k)
                frontier.append(path + [label])
            replay(env, prefix + path)  # restore for next sibling
    return None, nodes


def main() -> int:
    print("== sc25 FROM-SCRATCH offline solver v3 (replay-from-reset, zero quota) ==")
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    env = arc.make("sc25", scorecard_id=arc.open_scorecard())
    f = env.reset()
    cur = _levels_completed(f)
    print(f"reset: level={cur} castable={castable_spells(env._game)}")

    full = []
    for lvl in range(cur + 1, TARGET + 1):
        path, nodes = solve_one(env, cur, DEPTH.get(lvl, 30), full)
        if path is None:
            print(f"  STUCK L{cur}->L{lvl} ({nodes} nodes)")
            break
        f = replay(env, full + path)
        cur = _levels_completed(f)
        full += path
        print(f"  solved L{cur}: +{len(path)} actions (total {len(full)}, {nodes} nodes)")
        if cur < lvl:
            print("  WARN under-target; stop"); break

    print(f"\n  sc25 result: reached L{cur} via {len(full)} composite actions")
    out = REPO / "results" / "arc3_sc25_offline_resolve.json"
    out.write_text(json.dumps({"game": "sc25", "reached_level": cur, "composite_actions": full,
                               "mode": "from_scratch_offline_replay_bfs_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0 if cur >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
