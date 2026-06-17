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

    def __init__(self, target_levels: int = 1, max_depth: int = 45, hud_mask=None) -> None:
        self.hud_mask = hud_mask             # E1: mask step-counter cells out of node identity
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
        self.adj: dict[str, list] = {}       # known forward edges: hash -> [(action_dict, next_hash)]

    def _hash(self, frame) -> str:
        from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
        g = grid_of(frame)
        if self.hud_mask is not None and getattr(self.hud_mask, "shape", None) == g.shape:
            g = g.copy()
            g[self.hud_mask] = 0             # collapse counter/timer cells so equal game states dedup
        return frame_hash(g)

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
            if not self._game_over(latest):
                act = {"action": o["action"], "data": o["data"]}
                # record the forward edge for frontier-distance navigation (only if the
                # action actually CHANGED state — a no-op self-edge is useless to navigate)
                if h != o["origin"]:
                    self.adj.setdefault(o["origin"], []).append((act, h))
                if h not in self.graph:
                    opath = self.graph.get(o["origin"], {}).get("path", [])
                    self.graph[h] = {"path": opath + [act], "untested": self._candidates(latest)}
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

    def _shortest_path(self, src: Optional[str], dst: str) -> Optional[list]:
        """Frontier-distance navigation: BFS over the KNOWN forward edges from src to dst.
        Returns the action sequence to walk there WITHOUT a RESET (cheaper than replay-
        from-root), or None if dst isn't forward-reachable from src in the known graph."""
        from collections import deque
        if src is None or src == dst:
            return [] if src == dst else None
        seen = {src}
        q = deque([(src, [])])
        while q:
            node, path = q.popleft()
            for act, nxt in self.adj.get(node, []):
                if nxt in seen:
                    continue
                npath = path + [act]
                if nxt == dst:
                    return npath
                seen.add(nxt)
                q.append((nxt, npath))
        return None

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
        # 1) DEPTH-first ride: expand the current state's untested SALIENT actions while
        #    under the depth cap (no nav cost; reaches the deep wins lp85/sp80 need —
        #    BFS-order regressed those, so the deep-ride stays the search order).
        if cur_node and cur_node["untested"] and len(cur_node["path"]) < self.max_depth:
            a = cur_node["untested"].pop(0)
            self.awaiting = {"origin": self.cur, "action": a["action"], "data": a["data"]}
            return (a["action"], a["data"])
        # 2) Current exhausted / dead-end / depth-capped: go to the shallowest frontier,
        #    navigating via known forward edges if reachable (FRONTIER-DISTANCE, cheap)
        #    else RESET + replay from root. This only changes nav COST, not search order.
        th = self._frontier()
        if th is None:
            self.explored_out = True
            return (None, None)
        node = self.graph[th]
        a = node["untested"].pop(0)
        fwd = self._shortest_path(self.cur, th) if not over else None
        if fwd is not None:
            self.pending = [{"kind": s["action"], "data": s["data"], "probe": False} for s in fwd]
        else:
            self.pending = [{"kind": "RESET", "data": None, "probe": False}]
            self.pending += [{"kind": s["action"], "data": s["data"], "probe": False} for s in node["path"]]
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
                 target_level: Optional[int] = None, force_explore: bool = False,
                 hud_mask=None) -> None:
        self.short = str(game_id).split("-", 1)[0]
        sols = solutions if solutions is not None else load_solutions()
        self.plan = [] if force_explore else sols.get(self.short, [])
        self.i = 0
        self.reset_sent = False
        self.target = target_level if target_level is not None else CLAIMED.get(self.short, 1)
        self.has_plan = bool(self.plan)
        # eval games are UNSEEN -> no banked plan -> the generic step-wise explorer runs.
        self.explorer: Optional[StepwiseExplorer] = (
            None if self.has_plan else StepwiseExplorer(hud_mask=hud_mask))

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


class E3AgentPolicy:
    """E3-mode agent: the STRONG choose_action. Phase machine driven step-wise by the
    harness — EXPLORE (collect transitions from its own play) -> INDUCE (an OFFLINE
    local proposer writes a world model from those transitions) -> VERIFY (Carnot
    WorldModelVerifier grounds it) -> PLAN (search to a win INSIDE the verified model)
    -> EXECUTE (replay the plan; on divergence, back to EXPLORE). The proposer is
    INJECTED and defaults to the offline-legal local one, never a closed online API, so
    this is competition-legal (no internet at eval) AND decentralized.

    The induce/verify/plan quality (esp. from a small LOCAL model) is the open milestone
    the focused loop measures next; the EXPLORE + collect + verify wiring is exercised
    here, and EXECUTE re-uses the same env interface as the explorer."""

    def __init__(self, game_id: str, proposer=None, explore_budget: int = 80,
                 target_levels: int = 1) -> None:
        self.short = str(game_id).split("-", 1)[0]
        self.explorer = StepwiseExplorer(target_levels=target_levels)
        self.transitions: list = []          # (grid_before, action, data, grid_after) self-collected
        self.explore_budget = explore_budget
        self.proposer = proposer             # default set lazily to LocalGGUFProposer
        self.phase = "explore"
        self.plan: list = []
        self.pi = 0
        self._prev = None                    # last (grid, action_id, data) for transition pairing
        self.cell = 1
        self.induced = False
        self.root_grid = None                # the reset-state logical grid; plan_in_model starts here

    def _proposer(self):
        if self.proposer is None:
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
            self.proposer = LocalGGUFProposer()
        return self.proposer

    def next_move(self, frames, latest):
        from carnot.agentic.arc_executable_world_model import to_logical, detect_cell
        # collect a transition from the last action's outcome
        if self._prev is not None and latest is not None:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import Transition
            g0, aid, data = self._prev
            self.transitions.append(Transition(g0, aid, data,
                                               to_logical(grid_of(latest), self.cell), 0, _level_of(latest)))
        if self.phase == "explore":
            mv = self.explorer.next_move(frames, latest)
            if latest is not None:
                from carnot.agentic.arc_agi3_world_model import grid_of
                self.cell = detect_cell(grid_of(latest))
                if self.root_grid is None and self.explorer.root is not None:
                    self.root_grid = to_logical(grid_of(latest), self.cell)
                if mv[0] not in ("RESET", None):
                    self._prev = (to_logical(grid_of(latest), self.cell), int(mv[0]), mv[1])
                else:
                    self._prev = None
            # VERIFIER-ROUTED CASCADE escalation: hand off to the tier-3 LLM (E3 induction)
            # only when the cheap tier-1 explorer has STALLED — spent its transition budget
            # without a level-up, or fully explored out. If the explorer WON, is_done ends
            # the episode (tier-1 success; no costly escalation). This is the router: cheap
            # first, escalate the hard tail.
            won = self.explorer.best_level > (self.explorer.start_level or 0)
            stalled = len(self.transitions) >= self.explore_budget or self.explorer.explored_out
            if stalled and not won:
                self.phase = "induce"
            return mv
        if self.phase == "induce" and not self.induced:
            self.induced = True
            self._induce_and_plan()
            self.phase = "execute" if self.plan else "explore"
            self._prev = None
            return ("RESET", None) if self.plan else self.explorer.next_move(frames, latest)
        if self.phase == "execute" and self.pi < len(self.plan):
            step = self.plan[self.pi]; self.pi += 1
            return (step["action"], step.get("data"))
        # plan exhausted / no model -> keep exploring
        self.phase = "explore"
        return self.explorer.next_move(frames, latest)

    def _induce_and_plan(self):
        from carnot.agentic import arc_executable_world_model as e3
        try:
            ok, _ = self._proposer().induce(self.short, self.transitions, self.cell)
            if not ok or self.root_grid is None:
                return
            engine, is_done = e3.load_engine(self.short)
            vr = e3.WorldModelVerifier(self.transitions).score(engine)
            if vr.accuracy < 0.5:            # too weak to trust for planning
                return
            # plan ENTIRELY in the model (zero real actions); execute phase RESETs then
            # replays this plan in the real env, halting on divergence.
            plan = e3.plan_in_model(engine, is_done, self.root_grid)
            if plan:
                self.plan = plan
        except Exception:
            return

    def is_done(self, frames, latest):
        return self.explorer.is_done(frames, latest) and self.phase == "explore"


def make_carnot_agent(base_cls, cascade: bool = True, proposer=None):
    """Adapt the Carnot policy onto the real ARC-AGI-3-Agents `Agent` base class.
    Submission: `from agents.agent import Agent; CarnotAgent = make_carnot_agent(Agent)`.

    cascade=True (DEFAULT, the competition path): the VERIFIER-ROUTED CASCADE
    (E3AgentPolicy) — tier-1 training-free explorer; on STALL escalate to tier-3 E3
    induction with the bundled open proposer (the verifier routes + grounds). This is the
    unified choose_action the hard eval needs. cascade=False: pure recognize-and-replay
    (dev/known games only — useless on the hidden eval)."""

    class CarnotAgent(base_cls):  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            gid = getattr(self, "game_id", "")
            self._policy = (E3AgentPolicy(gid, proposer=proposer) if cascade
                            else CarnotAgentPolicy(gid, load_solutions()))

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
