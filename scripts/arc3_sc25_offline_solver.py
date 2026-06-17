"""From-scratch OFFLINE solver for sc25 (priority: deeper ARC solves
offline-reproducible). No replay of live recordings. Env-cloned BFS over sc25's
real action units -- cast each available spell (_cast_spell_plan) + moves
(ACTION1/2/3) -- deduped by a goal-relevant state key, chained level-by-level
from the offline reset. Zero quota.
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
import carnot.experiment_4236_arc_incremental_progress as sc

TARGET = 5
MAX_STATES = 60000
MOVE_IDS = (1, 2, 3)


def available_spells(game) -> list[str]:
    # castable = has a selector sprite AND a pattern that isn't already satisfied
    sprites = game.current_level.get_sprites() if hasattr(game, "current_level") else []
    selectable = {str(getattr(s, "name", ""))[len("sptivk-"):]
                  for s in sprites if str(getattr(s, "name", "")).startswith("sptivk-")}
    patterns = getattr(game, "zzpoabuniyn", {})
    cur = getattr(game, "xhhaqjfncnp", [[False] * 3 for _ in range(3)])
    out = []
    for sp in sorted(selectable):
        pat = patterns.get(sp) if isinstance(patterns, dict) else None
        if not isinstance(pat, list):
            continue
        # only a cast that would actually change the cast-state is a useful action
        if any(bool(pat[r][c]) and not bool(cur[r][c]) for r in range(3) for c in range(3)):
            out.append(sp)
    return out


def goal_key(game) -> tuple:
    sprites = game.current_level.get_sprites() if hasattr(game, "current_level") else []
    pos = tuple(sorted((str(getattr(s, "name", "")), int(getattr(s, "x", 0)), int(getattr(s, "y", 0)))
                       for s in sprites))
    cast = str(getattr(game, "xhhaqjfncnp", None))
    return (cast, pos)


def apply_cast(env, spell) -> object:
    f = None
    for step in sc._cast_spell_plan(env._game, spell):
        f = env.step(GameAction.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
    return f


def composite_actions(env):
    """Yield (label, callable->frame) for each spell-cast and move from current state."""
    acts = []
    for sp in available_spells(env._game):
        acts.append(("cast:" + sp, lambda sp=sp: apply_cast(env, sp)))
    for m in MOVE_IDS:
        acts.append((f"move{m}", lambda m=m: env.step(getattr(GameAction, f"ACTION{m}"))))
    return acts


def solve_one_level(env, start_level: int) -> list | None:
    original = copy.deepcopy(env._game)
    seen = {goal_key(original)}
    queue = deque([(copy.deepcopy(original), [])])
    states = 0
    while queue and states < MAX_STATES:
        game_state, path = queue.popleft()
        states += 1
        for label, fn in composite_actions_for(env, game_state):
            env._game = copy.deepcopy(game_state)
            try:
                f = fn()
            except Exception:
                continue
            lv = _levels_completed(f) if f is not None else start_level
            np_ = path + [label]
            if lv > start_level:
                env._game = copy.deepcopy(original)
                return np_
            k = goal_key(env._game)
            if k not in seen:
                seen.add(k)
                queue.append((copy.deepcopy(env._game), np_))
    env._game = copy.deepcopy(original)
    return None


def composite_actions_for(env, game_state):
    """Composite actions enumerated against a given game_state (set before calling fn)."""
    env._game = copy.deepcopy(game_state)
    spells = available_spells(env._game)
    acts = []
    for sp in spells:
        acts.append(("cast:" + sp, lambda sp=sp: apply_cast(env, sp)))
    for m in MOVE_IDS:
        acts.append((f"move{m}", lambda m=m: env.step(getattr(GameAction, f"ACTION{m}"))))
    return acts


def main() -> int:
    print("== sc25 FROM-SCRATCH offline solver (spell-cast BFS, zero quota) ==")
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    env = arc.make("sc25", scorecard_id=arc.open_scorecard())
    f = env.reset()
    cur = _levels_completed(f)
    print(f"reset: level={cur}  spells@base={available_spells(env._game)}")

    full: list[str] = []
    for lvl in range(cur + 1, TARGET + 1):
        path = solve_one_level(env, cur)
        if path is None:
            print(f"  STUCK L{cur}->L{lvl}")
            break
        # apply the re-derived composite path to the real env
        for label in path:
            if label.startswith("cast:"):
                apply_cast(env, label.split(":", 1)[1])
            else:
                f = env.step(getattr(GameAction, f"ACTION{label[-1]}"))
        cur = _levels_completed(env.reset() if False else f if f is not None else f)
        cur = _levels_completed(f)
        full += path
        print(f"  solved L{cur}: +{len(path)} composite actions (total {len(full)})")
        if cur < lvl:
            print(f"  WARN level={cur}<{lvl}; stop"); break

    print(f"\n  sc25 FROM-SCRATCH offline result: reached L{cur} via {len(full)} composite actions")
    out = REPO / "results" / "arc3_sc25_offline_resolve.json"
    out.write_text(json.dumps({"game": "sc25", "reached_level": cur,
                               "composite_actions": full, "mode": "from_scratch_offline_bfs_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0 if cur >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
