"""STRUCTURAL PROBE: once a plan is installed, how many real actions does it cost?

Q1/Q2 of the blast-radius task. Instruments the CALLEE (`_next_plan_move`) and reads the
caller off the stack, per the "availability is not delivery" rule -- we do NOT conclude
from reading the execute branch that it has no divergence check; we drive the real policy
with a deliberately-wrong plan installed and COUNT how many steps it consumes and whether
anything ever interrupts it.

The plan is a sequence of actions the model believes reaches a win. We install it directly
(the same attribute `_induce_and_plan` assigns, `self.plan`) and then drive `next_move`
exactly as the harness does, feeding back REAL frames from the offline arcade. If a
downstream check existed, execution would stop early on the first frame that disagrees
with the model.
"""

from __future__ import annotations

import json
import sys
import traceback

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, REPO + "/python")


def probe(game: str, plan_len: int = 25, seed: int = 20260802) -> dict:
    import random

    import numpy as np
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    random.seed(seed)
    np.random.seed(seed)

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    latest = env.reset()
    frames = [latest]

    pol = E3AgentPolicy(game, proposer=None, explore_budget=24)

    # ---- instrument the CALLEE, not the call site -------------------------------------
    calls: list[dict] = []
    real_next_plan_move = pol._next_plan_move

    def spy():
        st = traceback.extract_stack()
        caller = st[-2].name  # who invoked _next_plan_move
        mv = real_next_plan_move()
        calls.append({"caller": caller, "pi_after": int(pol.pi), "move": str(mv[0])})
        return mv

    pol._next_plan_move = spy  # type: ignore[method-assign]

    # ---- install a deliberately-WRONG plan ---------------------------------------------
    # Action 1..4 cycled: a plausible nav plan that no correct model would emit for an
    # arbitrary board. This stands in for "the plan a 0.2-scoring engine produced".
    plan = [{"action": (i % 4) + 1, "data": None} for i in range(plan_len)]
    pol.plan = list(plan)
    pol.pi = 0
    pol.phase = "execute"
    pol.induced = True
    pol._execute_plan_from_current = True

    lvl0 = _level_of(latest)
    actions = 0
    stopped_at = None
    for _step in range(plan_len * 3):  # generous: room to observe post-plan behaviour
        mv = pol.next_move(frames, latest)
        kind, data = mv
        if kind == "RESET":
            latest = env.reset()
        else:
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
        frames.append(latest)
        actions += 1
        if pol.phase != "execute" and stopped_at is None:
            stopped_at = {"action_index": actions, "pi": int(pol.pi), "phase": pol.phase}
            break

    plan_steps_consumed = sum(1 for c in calls if c["caller"] == "_next_move_routed")
    return {
        "game": game,
        "plan_len": plan_len,
        "plan_steps_consumed": len(calls),
        "plan_steps_from_execute_branch": plan_steps_consumed,
        "pi_final": int(pol.pi),
        "phase_final": pol.phase,
        "actions_spent": actions,
        "left_execute_early": stopped_at,
        "callers_seen": sorted({c["caller"] for c in calls}),
        "level_before": lvl0,
        "level_after": _level_of(latest),
        "explored_out": bool(getattr(pol.explorer, "explored_out", False)),
    }


if __name__ == "__main__":
    dest = sys.argv[1]
    out = []
    for g in sys.argv[2:]:
        for pl in (5, 25, 60):
            try:
                out.append(probe(g, plan_len=pl))
            except Exception as exc:
                out.append({"game": g, "plan_len": pl, "error": repr(exc)[:400]})
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=2)
    print("WROTE", dest, len(out))
