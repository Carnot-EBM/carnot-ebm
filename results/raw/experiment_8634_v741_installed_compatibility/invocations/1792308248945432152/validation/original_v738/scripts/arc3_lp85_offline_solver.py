"""From-scratch OFFLINE solver for lp85 (priority: make deeper ARC solves
offline-reproducible). No replay of live recordings. Reuses the existing
goal-key-deduped env-cloned searcher `plan_observed_suffix` (exp4179) and chains
it level-by-level from the offline reset, so each level is RE-DERIVED for the
offline layout. Zero quota.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

from carnot.experiment_4179_arc_incremental_progress import (
    plan_observed_suffix,
    _levels_completed,
)

TARGET = 3
# per-level move budget -> search depth cap (survey: L1=13, L2=60, L3=80) + slack
DEPTH = {1: 20, 2: 70, 3: 90}


def main() -> int:
    print("== lp85 FROM-SCRATCH offline solver (plan_observed_suffix, zero quota) ==")
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    env = arc.make("lp85", scorecard_id=arc.open_scorecard())
    f = env.reset()
    print(f"reset: level={_levels_completed(f, env)}")

    full: list[dict] = []
    cur = 0
    for lvl in range(1, TARGET + 1):
        path, trace = plan_observed_suffix(env, GameAction, start_level=cur,
                                           max_depth=DEPTH.get(lvl, 90))
        if not path or not trace.get("found"):
            print(f"  STUCK L{cur}->L{lvl}: expanded={trace.get('expanded_states')} "
                  f"transitions={trace.get('observed_transition_count')}")
            break
        # apply the re-derived path to advance the real env
        for step in path:
            f = env.step(GameAction.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
        cur = _levels_completed(f, env)
        full += path
        print(f"  solved L{cur}: +{len(path)} moves (total {len(full)}), expanded={trace.get('expanded_states')}")
        if cur < lvl:
            print(f"  WARN: applied path but level={cur} < {lvl}; stop")
            break

    print(f"\n  lp85 FROM-SCRATCH offline result: reached L{cur} in {len(full)} moves")
    out = REPO / "results" / "arc3_lp85_offline_resolve.json"
    out.write_text(json.dumps({
        "game": "lp85", "reached_level": cur, "moves": len(full),
        "solution": [{"action": 6, "x": int(s["x"]), "y": int(s["y"]), "button": s.get("button")} for s in full],
        "mode": "from_scratch_offline_plan_observed_suffix_no_quota",
    }, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0 if cur >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
