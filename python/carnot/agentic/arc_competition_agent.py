"""CarnotAgent — an ARC Prize 2026 / ARC-AGI-3 competition agent (the Kaggle
submission shape). The competition runs an Agent subclass step-wise and OFFLINE:
each turn the harness calls `choose_action(frames, latest_frame) -> GameAction` and
`is_done(frames, latest_frame) -> bool`. No internet at eval time.

This module is FRAMEWORK-AGNOSTIC so it is testable offline without the
ARC-AGI-3-Agents package: the decision logic lives in `CarnotAgentPolicy`, and
`make_carnot_agent(Agent)` adapts it onto the real `Agent` base class at submission
time (a thin subclass). The validation harness `scripts/arc_competition_validate.py`
drives the policy through our offline sims (environment_files), mimicking the
competition loop, to confirm the agent scores BEFORE any submission.

Policy (v1 — recognize-and-replay): the harness gives the agent its `game_id` at
construction. For a game we have an OFFLINE-REPRODUCED solution (the 13-level
registry), the agent replays that banked action sequence (Mode-1; no search, no
internet — ideal for the offline eval IF the eval games == the public 25). For an
UNKNOWN game (hidden eval), it falls back to a step-wise systematic explorer
(navigate-by-RESET-replay, take untested salient actions) — the online form of
graph_explore_solve_v2. The replay path is the validated v1; the explore fallback is
the generalizing path for held-out games.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[3]

# the 11 reproduced games and their target (offline-reproduced) level
CLAIMED = {"r11l": 1, "lp85": 3, "ls20": 1, "wa30": 1, "cd82": 1, "sp80": 1,
           "su15": 1, "tu93": 1, "cn04": 1, "m0r0": 1, "sk48": 1}
MAX_ACTIONS = 200


def load_solutions() -> dict[str, list[dict]]:
    """game-short -> [{"action": int, "data": {x,y}|None}] for every banked solution,
    via the metaharness's loader (single source of truth for the trajectories)."""
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)  # type: ignore
    sols: dict[str, list[dict]] = {}
    for short in CLAIMED:
        src = mh.RESOLVED_ARTIFACTS.get(short, mh.GAME_ARTIFACTS.get(short))
        if not src:
            continue
        steps = []
        for a in mh.load_actions(src):
            aid, data = mh.normalize(a)
            if aid is not None:
                steps.append({"action": int(aid), "data": data})
        sols[short] = steps
    return sols


def _level_of(frame: Any) -> int:
    if frame is None:
        return 0
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed
    try:
        return _levels_completed(frame)
    except Exception:
        return 0


class StepwiseExplorer:
    """Generic, game-AGNOSTIC step-wise solver — the competition asset (eval games are
    UNSEEN, so replay is useless; this is the only thing that scores). It is
    `graph_explore_solve_v2` turned inside-out into an EVENT LOOP the harness drives one
    action per turn: maintain a state-transition graph, expand the current state's
    untested SALIENT actions DEPTH-first (no navigation cost), and when the current
    state is exhausted or dead-ends, navigate to the shallowest frontier state with
    untested actions by RESET + replaying its path (deepcopy/jump is impossible live;
    RESET-replay is the only navigation). Action-efficient (scoring rewards fewer
    actions: min(human/agent,1)^2). Stops at the first level-up (+1 incremental unit) or
    when fully explored."""

    def __init__(self, target_levels: int = 1, max_depth: int = 45) -> None:
        self.graph: dict[str, dict] = {}     # hash -> {"path": [...], "untested": [...]}
        self.root: Optional[str] = None
        self.cur: Optional[str] = None
        self.start_level: Optional[int] = None
        self.best_level = 0
        self.target_levels = target_levels   # stop after this many levels beyond start
        self.max_depth = max_depth           # cap DFS branch length -> forces backtrack
        self.pending: list[dict] = []        # queued nav/probe actions
        self.awaiting: Optional[dict] = None # last probe, to attribute its result
        self.explored_out = False

    @staticmethod
    def _hash(frame) -> str:
        from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
        return frame_hash(grid_of(frame))

    @staticmethod
    def _candidates(frame) -> list[dict]:
        from carnot.agentic.arc_graph_explore import rich_action_candidates
        return [{"action": int(c.action_id), "data": c.data} for c in rich_action_candidates(frame)]

    @staticmethod
    def _game_over(frame) -> bool:
        from carnot.agentic.arc_agi3_live_adapter import _game_over
        try:
            return bool(_game_over(frame))
        except Exception:
            return False

    def _ingest(self, latest) -> None:
        if latest is None:
            return
        h = self._hash(latest)
        lvl = _level_of(latest)
        if self.start_level is None:
            self.start_level = lvl
        self.best_level = max(self.best_level, lvl)
        if self.awaiting is not None:
            o = self.awaiting
            self.awaiting = None
            if not self._game_over(latest) and h not in self.graph:
                opath = self.graph.get(o["origin"], {}).get("path", [])
                self.graph[h] = {"path": opath + [{"action": o["action"], "data": o["data"]}],
                                 "untested": self._candidates(latest)}
        self.cur = h
        if self.root is None:
            self.root = h
            self.graph.setdefault(h, {"path": [], "untested": self._candidates(latest)})

    def _frontier(self) -> Optional[str]:
        best = None
        for h, node in self.graph.items():
            if node["untested"]:
                if best is None or len(node["path"]) < len(self.graph[best]["path"]):
                    best = h
        return best

    def _serve(self) -> tuple:
        item = self.pending.pop(0)
        if item.get("probe"):
            self.awaiting = {"origin": item["origin"], "action": item["kind"], "data": item["data"]}
        if item["kind"] == "RESET":
            return ("RESET", None)
        return (item["kind"], item["data"])

    def next_move(self, frames, latest) -> tuple:
        if self.root is None and latest is None:      # bootstrap: RESET to get the first frame
            return ("RESET", None)
        self._ingest(latest)
        if self.pending:
            return self._serve()
        over = latest is not None and self._game_over(latest)
        cur_node = self.graph.get(self.cur) if not over else None
        # 1) Ride the current branch DEPTH-first while it has untested SALIENT actions and
        #    is under the depth cap (no navigation cost — reaches deep wins like wa30/sp80).
        if cur_node and cur_node["untested"] and len(cur_node["path"]) < self.max_depth:
            a = cur_node["untested"].pop(0)
            self.awaiting = {"origin": self.cur, "action": a["action"], "data": a["data"]}
            return (a["action"], a["data"])
        # 2) Current exhausted / dead-end / depth-capped: navigate to the SHALLOWEST
        #    frontier state with an untested action by RESET + replay (the only live nav).
        th = self._frontier()
        if th is None:
            self.explored_out = True
            return (None, None)
        node = self.graph[th]
        a = node["untested"].pop(0)
        self.pending = [{"kind": "RESET", "data": None, "probe": False}]
        for step in node["path"]:
            self.pending.append({"kind": step["action"], "data": step["data"], "probe": False})
        self.pending.append({"kind": a["action"], "data": a["data"], "probe": True, "origin": th})
        return self._serve()

    def is_done(self, frames, latest) -> bool:
        if self.start_level is not None and self.best_level >= self.start_level + self.target_levels:
            return True
        return self.explored_out


class CarnotAgentPolicy:
    """Framework-agnostic decision logic. `next_move` yields ("RESET",None) once, then
    the banked plan one step at a time, then (None,None) when exhausted. `is_done`
    stops when the target level is reached or the plan is spent."""

    def __init__(self, game_id: str, solutions: Optional[dict] = None,
                 target_level: Optional[int] = None, force_explore: bool = False) -> None:
        self.short = str(game_id).split("-", 1)[0]
        sols = solutions if solutions is not None else load_solutions()
        self.plan = [] if force_explore else sols.get(self.short, [])
        self.i = 0
        self.reset_sent = False
        self.target = target_level if target_level is not None else CLAIMED.get(self.short, 1)
        self.has_plan = bool(self.plan)
        # eval games are UNSEEN -> no banked plan -> the generic step-wise explorer runs.
        self.explorer: Optional[StepwiseExplorer] = None if self.has_plan else StepwiseExplorer()

    def next_move(self, frames, latest_frame) -> tuple:
        """-> ("RESET", None) | (action_id:int, data:dict|None) | (None, None)."""
        if self.explorer is not None:                 # unknown game: generic solver
            return self.explorer.next_move(frames, latest_frame)
        if not self.reset_sent:                       # known game: replay banked solution
            self.reset_sent = True
            return ("RESET", None)
        if self.i < len(self.plan):
            s = self.plan[self.i]
            self.i += 1
            return (int(s["action"]), s.get("data"))
        return (None, None)

    def is_done(self, frames, latest_frame) -> bool:
        if self.explorer is not None:
            return self.explorer.is_done(frames, latest_frame)
        if _level_of(latest_frame) >= self.target:
            return True
        return self.reset_sent and self.i >= len(self.plan)


def make_carnot_agent(base_cls):
    """Adapt CarnotAgentPolicy onto the real ARC-AGI-3-Agents `Agent` base class.
    Submission: `from agents.agent import Agent; CarnotAgent = make_carnot_agent(Agent)`.
    The base class supplies game_id/arc_env/etc.; we only implement the two abstract
    methods by delegating to the policy."""

    class CarnotAgent(base_cls):  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._policy = CarnotAgentPolicy(getattr(self, "game_id", ""), load_solutions())

        def is_done(self, frames, latest_frame) -> bool:
            return self._policy.is_done(frames, latest_frame)

        def choose_action(self, frames, latest_frame):
            from arcengine import GameAction
            kind, data = self._policy.next_move(frames, latest_frame)
            if kind == "RESET" or kind is None:
                return GameAction.RESET
            act = getattr(GameAction, f"ACTION{kind}")
            return act.set_data(data) if data else act

    return CarnotAgent
