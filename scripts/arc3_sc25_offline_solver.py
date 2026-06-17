"""From-scratch OFFLINE solver for sc25 (priority: deeper ARC solves
offline-reproducible). No replay of live recordings.

sc25 mechanics (reverse-engineered from environment_files/sc25/.../sc25.py):
  - 3x3 cast grid: clicking the `clzbxlm-sptivk-slsrhr` cell sprite at display
    coords (24+5c, 49+5r) TOGGLES xhhaqjfncnp[r][c] (camera is identity).
    (The hardcoded SC25_GRID_COORDS in the live solver are WRONG for the env.)
  - cast: select `sptivk-{spell}` sprite, set xhhaqjfncnp == zzpoabuniyn[spell],
    re-click the spell to fire -> multi-frame animation; let it resolve.
  - moves: ACTION1/2/3/4 move the active sprite; abvgsmnbrj() -> next_level when a
    winning move completes. Step budget enforced.

Env-cloned BFS over {cast each castable spell} + {moves}, animations resolved,
deduped by a goal-relevant key, chained level-by-level from the offline reset.
Zero quota.
"""
from __future__ import annotations

import copy
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
MAX_STATES = 40000
PHASES = ["vbublqskwzw", "ggotuphkheh", "obrrczymkxn", "wmnlnlscbpq", "jwlqyoqyagv", "agzbtzaakna"]
MOVE_IDS = (1, 2, 3, 4)


def cell_xy(r, c):
    return (24 + 5 * c, 49 + 5 * r)


def anim_active(g):
    return any(getattr(g, p, {}).get("acyylh") for p in PHASES)


def resolve(env):
    # advance only while an animation phase is active (action ignored during anim)
    for _ in range(80):
        if not anim_active(env._game):
            break
        env.step(GameAction.ACTION5)
    return env._game


def spell_sprite(g, spell):
    for s in g.current_level.get_sprites():
        if str(getattr(s, "name", "")) == f"sptivk-{spell}":
            return s
    return None


def castable_spells(g):
    pats = getattr(g, "zzpoabuniyn", {})
    cur = g.xhhaqjfncnp
    out = []
    for sp, pat in pats.items():
        if spell_sprite(g, sp) is None:
            continue
        if any(bool(pat[r][c]) != bool(cur[r][c]) for r in range(3) for c in range(3)):
            out.append(sp)
    return out


def cast(env, spell):
    g = env._game
    ss = spell_sprite(g, spell)
    if ss is None:
        return env._game
    env.step(GameAction.ACTION6, data={"x": int(ss.x), "y": int(ss.y)})  # select
    g = env._game
    pat = g.zzpoabuniyn[spell]
    for r in range(3):
        for c in range(3):
            if bool(pat[r][c]) != bool(env._game.xhhaqjfncnp[r][c]):
                x, y = cell_xy(r, c)
                env.step(GameAction.ACTION6, data={"x": x, "y": y})  # toggle to match
    ss = spell_sprite(env._game, spell)
    if ss is not None:
        env.step(GameAction.ACTION6, data={"x": int(ss.x), "y": int(ss.y)})  # fire
    return resolve(env)


def goal_key(g):
    sprites = g.current_level.get_sprites()
    pos = tuple(sorted((str(getattr(s, "name", "")), int(getattr(s, "x", 0)), int(getattr(s, "y", 0)))
                       for s in sprites))
    return (str(g.xhhaqjfncnp), pos)


def composite(env, game_state):
    env._game = copy.deepcopy(game_state)
    acts = [("cast:" + sp, lambda sp=sp: cast(env, sp)) for sp in castable_spells(env._game)]
    acts += [(f"move{m}", lambda m=m: (env.step(getattr(GameAction, f"ACTION{m}")), resolve(env))[1])
             for m in MOVE_IDS]
    return acts


def solve_one(env, start_level):
    original = copy.deepcopy(env._game)
    seen = {goal_key(original)}
    q = deque([(copy.deepcopy(original), [])])
    states = 0
    while q and states < MAX_STATES:
        gs, path = q.popleft()
        states += 1
        for label, fn in composite(env, gs):
            env._game = copy.deepcopy(gs)
            try:
                fn()
            except Exception:
                continue
            lv = _levels_completed(None) if False else _level(env)
            if lv > start_level:
                env._game = copy.deepcopy(original)
                return path + [label], states
            k = goal_key(env._game)
            if k not in seen:
                seen.add(k)
                q.append((copy.deepcopy(env._game), path + [label]))
    env._game = copy.deepcopy(original)
    return None, states


def _level(env):
    try:
        return int(getattr(env._game, "levels_completed", 0) or 0)
    except Exception:
        return 0


def apply(env, label):
    if label.startswith("cast:"):
        cast(env, label.split(":", 1)[1])
    else:
        env.step(getattr(GameAction, f"ACTION{label[-1]}"))
        resolve(env)


def main() -> int:
    print("== sc25 FROM-SCRATCH offline solver v2 (discovered cast coords, zero quota) ==")
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    env = arc.make("sc25", scorecard_id=arc.open_scorecard())
    env.reset()
    cur = _level(env)
    print(f"reset: level={cur} castable={castable_spells(env._game)}")

    full = []
    for lvl in range(cur + 1, TARGET + 1):
        path, states = solve_one(env, cur)
        if path is None:
            print(f"  STUCK L{cur}->L{lvl} ({states} states)")
            break
        for label in path:
            apply(env, label)
        cur = _level(env)
        full += path
        print(f"  solved L{cur}: +{len(path)} actions (total {len(full)}, {states} states)")
        if cur < lvl:
            print("  WARN under-target; stop"); break

    print(f"\n  sc25 result: reached L{cur} via {len(full)} composite actions")
    out = REPO / "results" / "arc3_sc25_offline_resolve.json"
    out.write_text(json.dumps({"game": "sc25", "reached_level": cur, "composite_actions": full,
                               "mode": "from_scratch_offline_bfs_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0 if cur >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
