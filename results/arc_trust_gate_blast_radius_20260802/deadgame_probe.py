"""WORST CASE at the REAL cap: a plan of `plan_max_depth_default()` (=80) steps.

Measures how many of those 80 real actions are fired while the game is already in
GAME_OVER -- i.e. actions the agent spends on a corpse because nothing between the
trust gate and the environment ever asks whether the plan is still worth executing.
80 is not a chosen number: it is `plan_in_model`'s shipped depth cap, and the scored
agent's own MAX_ACTIONS is 400, so this is 20% of one game's entire scored budget.
"""

from __future__ import annotations

import json
import sys

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, REPO + "/python")


def probe(game: str, plan_len: int, seed: int = 20260802) -> dict:
    import random

    import numpy as np
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    random.seed(seed)
    np.random.seed(seed)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    latest = env.reset()
    frames = [latest]
    pol = E3AgentPolicy(game, proposer=None, explore_budget=24)
    pol.plan = [{"action": (i % 4) + 1, "data": None} for i in range(plan_len)]
    pol.pi = 0
    pol.phase = "execute"
    pol.induced = True
    pol._execute_plan_from_current = True

    steps = dead_steps = 0
    first_dead = None
    while pol.phase == "execute" and pol.pi < len(pol.plan):
        kind, data = pol.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
        else:
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
        frames.append(latest)
        steps += 1
        st = str(getattr(latest, "state", ""))
        if "GAME_OVER" in st:
            if first_dead is None:
                first_dead = steps
            else:
                dead_steps += 1
    return {
        "game": game,
        "plan_len": plan_len,
        "plan_steps_executed": steps,
        "first_game_over_at_step": first_dead,
        "steps_executed_into_dead_game": dead_steps,
        "pct_of_scored_budget_400": round(100.0 * steps / 400.0, 1),
    }


if __name__ == "__main__":
    dest = sys.argv[1]
    out = []
    for g in sys.argv[2:]:
        try:
            out.append(probe(g, 80))
        except Exception as exc:
            out.append({"game": g, "plan_len": 80, "error": repr(exc)[:300]})
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=2)
    print("WROTE", dest, len(out))
