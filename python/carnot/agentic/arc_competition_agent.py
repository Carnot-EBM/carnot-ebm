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

import carnot.agentic.arc_strategy_router as arc_strategy_router
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel

REPO = Path(__file__).resolve().parents[3]

# the 11 reproduced games and their target (offline-reproduced) level
CLAIMED = {
    "r11l": 1,
    "lp85": 3,
    "ls20": 1,
    "wa30": 1,
    "cd82": 1,
    "sp80": 1,
    "su15": 1,
    "tu93": 1,
    "cn04": 1,
    "m0r0": 1,
    "sk48": 1,
}
MAX_ACTIONS = 200
SUBMITTED_VALUE_WEIGHT = 5.0
SUBMITTED_TARGET_LEVELS = 5
SUBMITTED_SEARCH_MODE = "depth_first_ride"
SUBMITTED_GRAPH_EXPLORE_BUDGET = 80
SUBMITTED_ROUTED_EXPLORE_BUDGET = 24
_DEFAULT_VALUE_HEAD = object()


def load_solutions() -> dict[str, list[dict]]:
    """game-short -> [{"action": int, "data": {x,y}|None}] for every banked solution,
    via the metaharness's loader (single source of truth for the trajectories)."""
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py")
    )
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


def _action_key(action_id: int, data: Any) -> tuple:
    if int(action_id) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action_id),)


def _frame_only_mechanic_hint(frame: Any) -> Optional[str]:
    """Cheap frame-only mechanic hint used when the live route has no registry class.

    The full probe-based detector lives at the I/O edge (`scripts/arc3_frame_induction.py`).
    The submitted policy cannot touch `env._game`, so this hint only inspects the rendered
    grid and returns a positive class when a dense lower-board editor palette is visible.
    Unknown frames return None and keep the registry/default route.
    """
    if frame is None:
        return None
    try:
        import numpy as np
        from carnot.agentic.arc_agi3_world_model import grid_of

        grid = grid_of(frame)
        if getattr(grid, "shape", None) is None or grid.ndim != 2:
            return None
        h, w = grid.shape
        lower = grid[int(h * 0.55):, :]
        if lower.size == 0:
            return None
        vals, counts = np.unique(lower, return_counts=True)
        bg = int(vals[counts.argmax()])
        active = lower != bg
        density = float(active.mean())
        columns = int(np.count_nonzero(active.any(axis=0)))
        rows = int(np.count_nonzero(active.any(axis=1)))
        if density >= 0.08 and columns >= max(12, w // 5) and rows >= 5:
            return "program_editor"
    except Exception:
        return None
    return None


def _route_explore_budget(route: dict[str, Any]) -> int:
    if route.get("uses_goal_distance_heuristic") is False:
        return SUBMITTED_ROUTED_EXPLORE_BUDGET
    return SUBMITTED_GRAPH_EXPLORE_BUDGET


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

    def __init__(
        self,
        target_levels: int = 1,
        max_depth: int = 45,
        hud_mask=None,
        value_head=None,
        value_weight: float = 0.0,
        search_mode: str = "depth_first_ride",
    ) -> None:
        self.hud_mask = hud_mask  # E1: mask step-counter cells out of node identity
        # BRIDGE: a frame-only cross-game value head (frame -> predicted steps-to-next-level-up, LOWER =
        # closer). Trained offline on ALL banked solves (the offline->live distillation). The frontier is
        # ordered A*-style: priority = depth + value_weight*value. value_weight=0 (default) -> pure BFS
        # (value only breaks ties; can't regress). value_weight>0 -> the value head NUDGES the search
        # toward predicted-closer states (the A* routing that unlocked cn04 in graph_explore_solve_v2).
        self.value_head = value_head
        self.value_weight = float(value_weight)
        # SEARCH MODE: "depth_first_ride" (default, proven) rides the current branch depth-first (action-
        # efficient; load-bearing for the deep wins lp85/sp80). "best_first" ALWAYS expands the globally-
        # best A*-value frontier (depth + value_weight*value) -- this is the graph_explore_solve_v2 search
        # form where the cross-game value head's routing actually helped (it unlocked cn04). best_first
        # only beats the ride when the value head is good enough to route the deep wins itself; measure it.
        self.search_mode = search_mode
        self.graph: dict[
            str, dict
        ] = {}  # hash -> {"path": [...], "untested": [...], "value": float}
        self.root: Optional[str] = None
        self.cur: Optional[str] = None
        self.start_level: Optional[int] = None
        self.best_level = 0
        self.target_levels = target_levels  # stop after this many levels beyond start
        self.max_depth = max_depth  # cap DFS branch length -> forces backtrack
        self.pending: list[dict] = []  # queued nav/probe actions
        self.awaiting: Optional[dict] = None  # last probe, to attribute its result
        self.explored_out = False
        self.adj: dict[str, list] = {}  # known forward edges: hash -> [(action_dict, next_hash)]

    def _hash(self, frame) -> str:
        from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash

        g = grid_of(frame)
        if self.hud_mask is not None and getattr(self.hud_mask, "shape", None) == g.shape:
            g = g.copy()
            g[self.hud_mask] = 0  # collapse counter/timer cells so equal game states dedup
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

    def _value(self, frame) -> float:
        """Frame-only learned progress score (predicted steps-to-next-level-up; LOWER == closer).
        0.0 when no value head -> the frontier falls back to shallowest-first. Never crashes the loop."""
        if self.value_head is None:
            return 0.0
        try:
            return float(self.value_head(frame))
        except Exception:
            return 0.0

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
                    self.graph[h] = {
                        "path": opath + [act],
                        "untested": self._candidates(latest),
                        "value": self._value(latest),
                    }
        self.cur = h
        if self.root is None:
            self.root = h
            self.graph.setdefault(
                h, {"path": [], "untested": self._candidates(latest), "value": self._value(latest)}
            )

    def _frontier(self) -> Optional[str]:
        # BRIDGE: A*-style frontier order -- priority = depth + value_weight*value. value_weight=0 is
        # depth-primary (pure BFS; value only breaks ties -> provably cannot regress). value_weight>0
        # lets the value head NUDGE toward predicted-closer states (the routing that unlocked cn04 in
        # graph_explore at weight 5). A full value-OVERRIDE (ignoring depth) measurably REGRESSED the
        # baseline (the weak head misroutes from shallow wins), so the blend keeps depth load-bearing.
        use_value = self.value_head is not None
        w = self.value_weight
        best = None
        best_key = None
        for h, node in self.graph.items():
            if not node["untested"]:
                continue
            depth = len(node["path"])
            if use_value:
                key = (depth + w * node.get("value", 0.0), depth)
            else:
                key = (depth,)
            if best is None or key < best_key:
                best, best_key = h, key
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
        if self.root is None and latest is None:  # bootstrap: RESET to get the first frame
            return ("RESET", None)
        self._ingest(latest)
        if self.pending:
            return self._serve()
        over = latest is not None and self._game_over(latest)
        cur_node = self.graph.get(self.cur) if not over else None
        # 1) DEPTH-first ride (search_mode="depth_first_ride", default): expand the current state's
        #    untested SALIENT actions while under the depth cap (no nav cost; reaches the deep wins
        #    lp85/sp80 need — BFS-order regressed those). best_first SKIPS this and always expands the
        #    globally-best A*-value frontier (step 2) so the value head drives the search order.
        if (
            self.search_mode == "depth_first_ride"
            and cur_node
            and cur_node["untested"]
            and len(cur_node["path"]) < self.max_depth
        ):
            a = cur_node["untested"].pop(0)
            self.awaiting = {"origin": self.cur, "action": a["action"], "data": a["data"]}
            return (a["action"], a["data"])
        # 2) Expand the best frontier (A*-value order). In best_first this is the primary step; in
        #    depth_first_ride it fires when the current node is exhausted / dead-end / depth-capped.
        th = self._frontier()
        if th is None:
            self.explored_out = True
            return (None, None)
        node = self.graph[th]
        a = node["untested"].pop(0)
        if (
            th == self.cur and not over
        ):  # best frontier IS the current state -> expand in place (no nav)
            self.awaiting = {"origin": self.cur, "action": a["action"], "data": a["data"]}
            return (a["action"], a["data"])
        fwd = self._shortest_path(self.cur, th) if not over else None
        if fwd is not None:
            self.pending = [{"kind": s["action"], "data": s["data"], "probe": False} for s in fwd]
        else:
            self.pending = [{"kind": "RESET", "data": None, "probe": False}]
            self.pending += [
                {"kind": s["action"], "data": s["data"], "probe": False} for s in node["path"]
            ]
        self.pending.append({"kind": a["action"], "data": a["data"], "probe": True, "origin": th})
        return self._serve()

    def is_done(self, frames, latest) -> bool:
        if (
            self.start_level is not None
            and self.best_level >= self.start_level + self.target_levels
        ):
            return True
        return self.explored_out


def load_cross_game_value_head():
    """BRIDGE loader: the frame-only cross-game value head (frame -> predicted steps-to-next-level-up),
    trained offline on ALL banked solves by scripts/arc_cross_game_verifier_train.py. Returns a callable
    the StepwiseExplorer routes its frontier with on an UNSEEN game, or None if not yet trained. This is
    the offline->live distillation: continued offline solves retrain it and the live agent inherits them."""
    from pathlib import Path

    models = Path(__file__).resolve().parents[3] / "models"
    try:
        from carnot.agentic.arc_value_learner import (
            LearnedVerifier,
            cross_game_features,
            cross_game_features_v2,
        )

        # prefer the RICHER v2 head (spatial occupancy; it routed cn04 where v1's 5 scalars could not)
        v2 = models / "arc_verifier_cross_game_v2.json"
        if v2.exists():
            v = LearnedVerifier.load(v2, cross_game_features_v2)
            return lambda frame: v(frame)
        v1 = models / "arc_verifier_cross_game.json"
        if v1.exists():
            v = LearnedVerifier.load(v1, cross_game_features)
            return lambda frame: v(frame)
    except Exception:
        return None
    return None


class CarnotAgentPolicy:
    """Framework-agnostic decision logic. `next_move` yields ("RESET",None) once, then
    the banked plan one step at a time, then (None,None) when exhausted. `is_done`
    stops when the target level is reached or the plan is spent."""

    def __init__(
        self,
        game_id: str,
        solutions: Optional[dict] = None,
        target_level: Optional[int] = None,
        force_explore: bool = False,
        hud_mask=None,
        value_head=None,
        value_weight: float = 0.0,
        search_mode: str = "depth_first_ride",
    ) -> None:
        self.short = str(game_id).split("-", 1)[0]
        sols = solutions if solutions is not None else load_solutions()
        self.plan = [] if force_explore else sols.get(self.short, [])
        self.i = 0
        self.reset_sent = False
        self.target = target_level if target_level is not None else CLAIMED.get(self.short, 1)
        self.has_plan = bool(self.plan)
        # eval games are UNSEEN -> no banked plan -> the generic step-wise explorer runs (value_head +
        # value_weight A*-route its frontier when provided -- the offline->live bridge).
        self.explorer: Optional[StepwiseExplorer] = (
            None
            if self.has_plan
            else StepwiseExplorer(
                hud_mask=hud_mask,
                value_head=value_head,
                value_weight=value_weight,
                search_mode=search_mode,
            )
        )

    def next_move(self, frames, latest_frame) -> tuple:
        """-> ("RESET", None) | (action_id:int, data:dict|None) | (None, None)."""
        if self.explorer is not None:  # unknown game: generic solver
            return self.explorer.next_move(frames, latest_frame)
        if not self.reset_sent:  # known game: replay banked solution
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

    def __init__(
        self,
        game_id: str,
        proposer=None,
        explore_budget: Optional[int] = None,
        target_levels: int = SUBMITTED_TARGET_LEVELS,
        value_head: Any = _DEFAULT_VALUE_HEAD,
        value_weight: float = SUBMITTED_VALUE_WEIGHT,
        search_mode: str = SUBMITTED_SEARCH_MODE,
        mechanic_detector=None,
    ) -> None:
        self.short = str(game_id).split("-", 1)[0]
        if value_head is _DEFAULT_VALUE_HEAD:
            value_head = load_cross_game_value_head()
        self.strategy_route = arc_strategy_router.route_for_game(self.short)
        self._route_from_frame_checked = False
        self.mechanic_detector = mechanic_detector or _frame_only_mechanic_hint
        self.dsl_model = ObjectDeltaModel(self.short)
        self.dsl_energy: Optional[dict[str, Any]] = None
        self._dsl_transitions: list[tuple[Any, tuple, Any]] = []
        self.explorer = StepwiseExplorer(
            target_levels=target_levels,
            value_head=value_head,
            value_weight=value_weight,
            search_mode=search_mode,
        )
        self.transitions: list = []  # (grid_before, action, data, grid_after) self-collected
        self.explore_budget = (
            int(explore_budget)
            if explore_budget is not None
            else _route_explore_budget(self.strategy_route)
        )
        self.proposer = proposer  # default set lazily to LocalGGUFProposer
        self.phase = "explore"
        self.plan: list = []
        self.pi = 0
        self._prev = None  # last (grid, action_id, data) for transition pairing
        self.cell = 1
        self.induced = False
        self.root_grid = None  # the reset-state logical grid; plan_in_model starts here

    def _maybe_route_from_frame(self, latest: Any) -> None:
        if self._route_from_frame_checked or latest is None:
            return
        self._route_from_frame_checked = True
        try:
            mechanic = self.mechanic_detector(latest) if self.mechanic_detector else None
        except Exception:
            mechanic = None
        if isinstance(mechanic, dict):
            mechanic = mechanic.get("mechanic")
        if not mechanic or mechanic == "unknown":
            return
        routed = arc_strategy_router.route_for_game(self.short, mechanic=str(mechanic))
        if routed.get("name") != self.strategy_route.get("name"):
            self.strategy_route = routed
            self.explore_budget = min(self.explore_budget, _route_explore_budget(routed))

    def _fit_dsl_model(self) -> None:
        if not self._dsl_transitions:
            return
        try:
            self.dsl_model = ObjectDeltaModel(self.short).fit(self._dsl_transitions)
            self.dsl_energy = self.dsl_model.consistency_energy(self._dsl_transitions)
        except Exception:
            self.dsl_energy = {"energy": None, "n_heldout": len(self._dsl_transitions)}

    def _proposer(self):
        if self.proposer is None:
            import os
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

            # Live-submission generator (validated 2026-06-19): Qwen3.5-9B-MTP + MTP + 8-bit KV + /no_think,
            # n_predict>=2048. 5.9GB Q4 fits 16GB; 62.5% Layer-B grounding (DeepSeek-Flash 25%, gemma verbose).
            # Kaggle deploy: set CARNOT_ARC_GGUF_PATH to the bundled /kaggle/input/.../Qwen3.5-9B-Q4_K_M.gguf;
            # CARNOT_ARC_MTP=0 disables MTP if a tight-VRAM box needs the ~4GB the self-draft costs.
            self.proposer = LocalGGUFProposer(
                repo_substr="Qwen3.5-9B-MTP",
                model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
                mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
                kv_quant="q8_0",
                no_think_prefix="/no_think\n",
                max_tokens=2560,
            )
        return self.proposer

    def next_move(self, frames, latest):
        from carnot.agentic.arc_executable_world_model import to_logical, detect_cell

        self._maybe_route_from_frame(latest)
        # collect a transition from the last action's outcome
        if self._prev is not None and latest is not None:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import Transition

            g0, aid, data = self._prev
            g1 = to_logical(grid_of(latest), self.cell)
            self.transitions.append(
                Transition(
                    g0, aid, data, g1, 0, _level_of(latest)
                )
            )
            self._dsl_transitions.append((g0, _action_key(aid, data), g1))
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
            if stalled and not won and not self.induced:  # escalate ONCE; then tier-1 fallback
                self.phase = "induce"
            return mv
        if self.phase == "induce" and not self.induced:
            self.induced = True
            self._induce_and_plan()
            self.phase = "execute" if self.plan else "explore"
            self._prev = None
            return ("RESET", None) if self.plan else self.explorer.next_move(frames, latest)
        if self.phase == "execute" and self.pi < len(self.plan):
            step = self.plan[self.pi]
            self.pi += 1
            return (step["action"], step.get("data"))
        # plan exhausted / no model -> keep exploring
        self.phase = "explore"
        return self.explorer.next_move(frames, latest)

    def _induce_and_plan(self):
        from carnot.agentic import arc_executable_world_model as e3

        try:
            self._fit_dsl_model()
            ok, _ = self._proposer().induce(self.short, self.transitions, self.cell)
            if not ok or self.root_grid is None:
                return
            engine, is_done = e3.load_engine(self.short)
            vr = e3.WorldModelVerifier(self.transitions).score(engine)
            if vr.accuracy < 0.5:  # too weak to trust for planning
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


# ============================================================================================
# SINGLE SOURCE OF TRUTH for WHAT SHIPS. The 0.08 incident (2026-06-19): the offline eval measured
# STRONGER opt-in configs (explorer_bf unlocked cn04) while the SUBMITTED default shipped bare BFS,
# and nobody caught it because "better" was opt-in-only and the headline metric was banked-replay
# levels, not the submitted path. RULE (enforced by test_arc_submitted_agent_parity.py): the
# STRONGEST measured config MUST be this declared submitted config -- improvements go HERE, never to
# an opt-in-only eval flag. The eval's submission-baseline + the parity test both read this dict, so
# the shipped agent and the measured baseline can never silently diverge again.
SUBMITTED_AGENT_CONFIG = {
    "policy": "E3AgentPolicy",          # the verifier-routed cascade (NOT cascade=False banked-replay)
    "cascade": True,
    # explorer config the live agent actually runs.
    "value_weight": SUBMITTED_VALUE_WEIGHT,
    "target_levels": SUBMITTED_TARGET_LEVELS,
    "search_mode": SUBMITTED_SEARCH_MODE,
    "graph_explore_budget": SUBMITTED_GRAPH_EXPLORE_BUDGET,
    "routed_explore_budget": SUBMITTED_ROUTED_EXPLORE_BUDGET,
    "router_wired": True,
    "world_model_dsl_wired": True,
}


def make_carnot_agent(base_cls, cascade: bool = True, proposer=None):
    """Adapt the Carnot policy onto the real ARC-AGI-3-Agents `Agent` base class.
    Submission: `from agents.agent import Agent; CarnotAgent = make_carnot_agent(Agent)`.

    cascade=True (DEFAULT, the competition path): the VERIFIER-ROUTED CASCADE
    (E3AgentPolicy) — tier-1 training-free explorer; on STALL escalate to tier-3 E3
    induction with the bundled open proposer (the verifier routes + grounds). This is the
    unified choose_action the hard eval needs. cascade=False: pure recognize-and-replay
    (dev/known games only — useless on the hidden eval).

    The exact shipped config is declared in SUBMITTED_AGENT_CONFIG (single source of truth);
    test_arc_submitted_agent_parity.py asserts this function's default policy matches it, so a
    silent divergence between what we measure and what we ship cannot recur (the 0.08 incident)."""

    class CarnotAgent(base_cls):  # type: ignore
        # The framework's Agent.MAX_ACTIONS default is 80 ("avoid looping forever"),
        # which is far too low for the eval: our deepest banked replay (lp85 -> L5)
        # alone needs well over 80 actions, and the held-out-game explore fallback
        # needs room to probe + solve several levels. 80 would truncate even our best
        # known game. The real bound is the eval's wall-clock budget (<=12h across all
        # games), not this per-game loop guard; Playback overrides it to 1e6 for the
        # same reason, so it is an intended override point. 400 comfortably covers our
        # multi-level replays + explore while staying well inside the time budget.
        MAX_ACTIONS = 400

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            gid = getattr(self, "game_id", "")
            self._policy = (
                E3AgentPolicy(
                    gid,
                    proposer=proposer,
                    target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
                    value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
                    search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
                )
                if cascade
                else CarnotAgentPolicy(gid, load_solutions())
            )

        def is_done(self, frames, latest_frame) -> bool:
            return self._policy.is_done(frames, latest_frame)

        def choose_action(self, frames, latest_frame):
            from arcengine import GameAction

            kind, data = self._policy.next_move(frames, latest_frame)
            if kind == "RESET" or kind is None:
                return GameAction.RESET
            act = getattr(GameAction, f"ACTION{kind}")
            if data:
                # CRITICAL: GameAction.set_data() RETURNS the inner ComplexAction, NOT
                # the enum member. The framework's do_action_request reads
                # `action.action_data` off the object choose_action returns, so we must
                # mutate the enum in place and return the ENUM. Returning set_data()'s
                # result (a ComplexAction, which has no .action_data) crashes every
                # coordinate/click action against the real harness. game_id is a required
                # ComplexAction field (Playback injects it too), so carry it through.
                act.set_data({"game_id": getattr(self, "game_id", ""), **data})
            return act

    return CarnotAgent
