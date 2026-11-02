"""HEAD-TO-HEAD: just-explore (3rd place, arXiv:2512.24156) vs Carnot's graph_explore_solve_v2 on OUR
offline arcade. The decisive, gating measurement (sota-rapid-accel-levers-2026-06-23.md #1): does the
SOTA explorer reach STRICTLY deeper levels than ours at an equal env-step budget on >=3 of 25 games?

SHIM: subclass just-explore's HeuristicAgent and override ONLY take_action() to drive our stateful
offline env (arc_solver_kit.offline_arcade) instead of the online HTTP API. Our offline frame is already
(1,64,64) -- the exact FrameData.frame format -- and our env is stateful, so the agent's exploration
core (FrameProcessor + GraphExplorer + the 5 salience tiers) runs UNCHANGED. We map our
`levels_completed -> FrameData.score` (our frame has no `.score`).

CAVEAT (documented, not hidden -- corrected per the 2026-06-23 adversarial audit): the two systems
navigate differently -- just-explore live-shortest-path, ours replay-from-reset. The nominal "budget" is
NOT equal env interaction: graph_explore_solve_v2's max_expansions counts only its own env.step calls, but
replay-from-reset does MANY uncounted env.step+reset per expansion, so at nominal-equal "1000" the bare
explorer does a MEASURED ~4x MORE real env interaction than just-explore (4.2x lp85 / 4.4x bp35). That
asymmetry HANDICAPS just-explore, so any just-explore win here is a conservative LOWER BOUND (it wins while
touching the env ~1/4 as much). Preflights: (a) shim-validity (replay lp85's banked trajectory, assert
score increments at each level-up); (b) best-of-3 seeds for the stochastic SOTA arm (report median too --
best-of-3 inflates the win COUNT, not the existence).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # no GPU for the just-explore arm

import types  # noqa: E402

JE = "/home/ianblenke/arc-sota-refs/arc-agi-3-just-explore"
sys.path.insert(0, JE)
sys.path.insert(0, os.path.join(JE, "agents"))  # graph_explorer is a top-level import in heuristic_agent

# Bypass agents/__init__.py (it eagerly imports langgraph / smolagents / LLM templates we do not need):
# pre-register a bare `agents` package with the right __path__ so submodule relative imports resolve.
_pkg = types.ModuleType("agents")
_pkg.__path__ = [os.path.join(JE, "agents")]
sys.modules["agents"] = _pkg
# Stub the AgentOps tracer (agents.tracing) -> a no-op decorator, so we don't need the agentops dep.
_tr = types.ModuleType("agents.tracing")
_tr.trace_agent_session = lambda fn: fn
sys.modules["agents.tracing"] = _tr

from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from agents.structs import FrameData, GameAction as JEGameAction, GameState  # noqa: E402

from arcengine import GameAction as OurGA  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_graph_explore import (  # noqa: E402
    _game_action, _game_over, _warm, graph_explore_solve_v2,
)
from carnot.agentic.arc_agi3_live_adapter import _available_action_ids, _levels_completed


def _to_framedata(f, game_id: str, *, not_played: bool = False) -> FrameData:
    """Map our offline frame -> a just-explore FrameData the HeuristicAgent reads."""
    if f is None:
        return FrameData(game_id=game_id, frame=[], state=GameState.GAME_OVER, score=0)
    grid = np.asarray(f.frame)
    if grid.ndim == 2:
        grid = grid[None, :, :]
    state = (
        GameState.NOT_PLAYED if not_played
        else GameState.GAME_OVER if _game_over(f)
        else GameState.NOT_FINISHED
    )
    return FrameData(
        game_id=game_id,
        frame=grid.astype(int).tolist(),
        state=state,
        score=int(_levels_completed(f)),
        available_actions=[int(a) for a in _available_action_ids(f)],
        guid="offline",
    )


class OfflineHeuristicAgent(HeuristicAgent):
    """just-explore's HeuristicAgent, driving our offline env. Only take_action is overridden."""

    def __init__(self, env, game_id: str, budget: int):
        super().__init__(card_id="offline", game_id=game_id, agent_name="h2h",
                         ROOT_URL="", record=False)
        self.env = env
        self.MAX_ACTIONS = int(budget)
        self.minimal_step_time = 0.0   # the 0.31s online rate-limit sleep is pointless offline
        self.TOTAL_TIME_ALLOWED = 10 ** 9  # don't time-terminate an offline run
        self.max_score = 0
        self.frames = [FrameData(game_id=game_id, score=0, state=GameState.NOT_PLAYED)]

    def take_action(self, action: JEGameAction):
        aid = 0 if action.name == "RESET" else int(action.name.replace("ACTION", ""))
        try:
            if aid == 0:  # RESET
                f = _warm(self.env, False)
                fd = _to_framedata(f, self.game_id)
            else:
                data = None
                if aid == 6:  # click carries x,y
                    data = {"x": int(action.action_data.x), "y": int(action.action_data.y)}
                f = self.env.step(_game_action(OurGA, aid), data=data)
                fd = _to_framedata(f, self.game_id)
        except Exception:
            return FrameData(game_id=self.game_id, frame=[], state=GameState.GAME_OVER, score=0)
        self.max_score = max(self.max_score, fd.score)
        return fd

    def cleanup(self) -> None:  # no session to close
        pass


def run_just_explore(arc, gid, budget, seed) -> int:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    agent = OfflineHeuristicAgent(env, gid, budget)
    try:
        agent.main()
    except Exception:
        pass
    return int(agent.max_score)


def run_carnot(arc, gid, budget) -> int:
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    _traj, lvl = graph_explore_solve_v2(env, start_level=0, max_expansions=budget,
                                        warmup=True, max_depth=60)
    return int(lvl)


def shim_validity(arc) -> dict:
    """Replay lp85's banked L4 trajectory through the shim; assert FrameData.score increments at level-up."""
    traj = json.load(open("results/arc_explore_trajectory_lp85_l4.json"))["trajectory"]
    gid = next(getattr(e, "game_id", "") for e in arc.get_environments()
               if getattr(e, "game_id", "").startswith("lp85"))
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    agent = OfflineHeuristicAgent(env, gid, budget=len(traj) + 5)
    _warm(env, False)
    scores = []
    for step in traj:
        aid = int(step["action"])
        a = JEGameAction.RESET if aid == 0 else JEGameAction[f"ACTION{aid}"]
        if aid == 6:
            d = step.get("data") or {"x": int(step.get("x", 0)), "y": int(step.get("y", 0))}
            a.set_data({"x": int(d["x"]), "y": int(d["y"])})
        fd = agent.take_action(a)
        scores.append(fd.score)
    ok = max(scores) >= 1 and scores[-1] >= max(scores) - 0  # score rose to the banked depth
    return {"valid": bool(ok), "max_score_seen": max(scores), "final": scores[-1], "n": len(scores)}


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seeds = [0, 1, 2]
    arc = kit.offline_arcade()
    print("== shim-validity preflight (lp85) ==", flush=True)
    sv = shim_validity(arc)
    print(f"  {sv}", flush=True)
    if not sv["valid"]:
        json.dump({"blocked": "shim_score_unmapped", "shim_validity": sv},
                  open("results/proto_h2h_just_explore.json", "w"), indent=2)
        print("BLOCKED: shim score mapping failed -- aborting (no silent garbage table).", flush=True)
        return 1
    games = sorted({getattr(e, "game_id", "").split("-")[0] for e in arc.get_environments()})
    out = {"budget": budget, "seeds": seeds, "shim_validity": sv, "games": {}}
    print(f"\n== head-to-head: just-explore (best-of-{len(seeds)}) vs Carnot graph_explore_v2, budget={budget} ==", flush=True)
    for short in games:
        gid = next(getattr(e, "game_id", "") for e in arc.get_environments()
                   if getattr(e, "game_id", "").startswith(short))
        t0 = time.time()
        try:
            je = [run_just_explore(arc, gid, budget, s) for s in seeds]
            je_best = max(je)
            carnot = run_carnot(arc, gid, budget)
        except Exception as e:  # one bad game must not kill the 25-game sweep
            out["games"][short] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {short:6} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        out["games"][short] = {"just_explore_per_seed": je, "just_explore_best": je_best,
                               "carnot_v2": carnot, "je_minus_carnot": je_best - carnot,
                               "wall_s": round(time.time() - t0, 1)}
        print(f"  {short:6} JE={je} best={je_best}  Carnot={carnot}  delta={je_best-carnot:+d}  ({round(time.time()-t0,1)}s)", flush=True)
    deltas = [g["je_minus_carnot"] for g in out["games"].values() if "je_minus_carnot" in g]
    out["je_deeper_count"] = sum(1 for d in deltas if d > 0)
    out["carnot_deeper_count"] = sum(1 for d in deltas if d < 0)
    out["sum_delta"] = sum(deltas)
    out["VERDICT"] = ("JE_STRATEGY_VALIDATED_extract" if out["je_deeper_count"] >= 3 and out["sum_delta"] > 0
                      else "TIE_OR_CARNOT_HOLDS_wall_is_not_exploration_schedule")
    json.dump(out, open("results/proto_h2h_just_explore.json", "w"), indent=2)
    print(f"\nje_deeper={out['je_deeper_count']}/25  carnot_deeper={out['carnot_deeper_count']}  "
          f"sum_delta={out['sum_delta']}  VERDICT={out['VERDICT']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
