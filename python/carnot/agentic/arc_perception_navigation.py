#!/usr/bin/env python3
"""Autonomous perception-grounded navigation solving for the live agent (REQ-ARC-WMTE-5839).

The chain (REQ-ARC-WMTE-5831..5838) established, end-to-end: perception is the ARC wall; fixing it flips goal
induction; and an AUTONOMOUS perception-derived player->goal verifier fed into Carnot's verifier-routed
`OfflineSolver` reproduces a navigation solve (tu93 L3) with ZERO per-game hand-RE -- EXCEPT one input,
`branch_mode` (whether the env's reset is idempotent). This module removes that last input and packages the
whole autonomous pipeline behind the scored-agent policy interface:

  - `recon`               : short exploration that moves the player, off the agent's OWN frames.
  - `derive_verifier`     : (player, goal) from motion via `derive_navigation_pair` -> a player->goal
                            Manhattan verifier. None if not a clean navigation game.
  - `auto_branch_mode`    : replay the same probe path twice; if the resulting state differs, the env's
                            reset is non-idempotent -> 'fresh_env', else 'replay'. This is the last per-game
                            input, now DERIVED.
  - `build_nav_adapter`   : a fully GENERIC navigation GameAdapter (directional moves, env.step, full-grid
                            state key) + the perception verifier + the detected branch_mode.
  - `solve_navigation`    : run the verifier-routed `OfflineSolver` + the real reproduction gate.
  - `PerceptionNavigationPolicy` : wraps the autonomous solve behind the scored `run_game` interface
                            (`is_done` / `next_move`), so the SCORED driver can carry a self-discovered
                            navigation solve -- the same plan-then-replay shape `CarnotAgentPolicy` uses, but
                            with the plan self-discovered from perception rather than hand-banked.

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm. verifier_is_oracle: False (a
player->goal Manhattan heuristic over observable colours; never reads the win predicate). All colours are
DERIVED FROM MOTION, never hardcoded, never read from source -- oracle-distinct, sovereignty-safe.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

GENERIC_DEPTH_CAPS = {1: 40, 2: 60, 3: 80, 4: 90, 5: 90}


def _grid2d(frame: Any) -> np.ndarray:
    from carnot.agentic.arc_agi3_world_model import grid_of

    g = grid_of(frame)
    if getattr(g, "ndim", 2) == 1:
        s = int(round(g.size**0.5))
        if s * s == g.size:
            g = g.reshape(s, s)
    return g


def build_perception_verifier(player_color: int, goal_color: int):
    """A player->goal Manhattan verifier over the RAW frame grid (lower == closer). Mirrors the tu93
    hand_verifier exactly, but with autonomously-derived colours."""
    pc, gc = int(player_color), int(goal_color)

    def _cent(g: np.ndarray, c: int):
        ys, xs = np.where(g == c)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    def verifier(game, frame=None):  # noqa: ARG001 -- OfflineSolver passes (game, frame)
        if frame is None:
            return 1000.0
        g = _grid2d(frame)
        p, t = _cent(g, pc), _cent(g, gc)
        if p is None or t is None:
            return 1000.0
        return abs(p[0] - t[0]) + abs(p[1] - t[1])

    return verifier


def recon(game: str, cycles: int = 3):
    """Short exploration that actually MOVES the player; returns the agent's own perception Transitions."""
    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import _available_action_ids, _game_action, _game_over
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_entity_hud_perception import Transition
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    cell = detect_cell(grid_of(frame))
    trans: list = []
    for a in [4, 2, 3, 1, 4, 2, 3, 1, 4, 4, 2, 2, 1, 1, 3, 3] * int(cycles):
        if _game_over(frame):
            break
        av = set(_available_action_ids(frame) or [1, 2, 3, 4])
        if a not in av:
            a = next(iter(av), 1)
        before = to_logical(grid_of(frame), cell)
        frame = env.step(_game_action(GameAction, a))
        after = to_logical(grid_of(frame), cell)
        trans.append(Transition(before=before, action=a, after=after))
    return trans


def derive_verifier(transitions):
    """(verifier, (player, goal)) from the agent's own transitions, or (None, None) if not a clean nav game."""
    from carnot.agentic.arc_entity_hud_perception import derive_navigation_pair

    pair = derive_navigation_pair(transitions)
    if pair is None:
        return None, None
    return build_perception_verifier(pair[0], pair[1]), pair


def _generic_action_labels(env, frame=None, path=None):
    av = list(getattr(frame, "available_actions", []) or []) if frame is not None else []
    moves = [a for a in av if a in (1, 2, 3, 4)] or [1, 2, 3, 4]
    return [json.dumps({"action": int(a)}) for a in moves]


def _generic_apply(env, label, frame):
    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import _game_action

    return env.step(_game_action(GameAction, json.loads(label)["action"]))


def _generic_state_key(game, frame=None):
    if frame is None:
        return None
    return _grid2d(frame).tobytes()


def auto_branch_mode(game: str, probe_labels: Optional[list] = None) -> str:
    """Derive the OfflineSolver branch_mode: replay the SAME probe path twice from a fresh reset of one env;
    if the resulting state differs, the env's reset is non-idempotent (a parity/hidden-state toggle) and the
    search must evaluate each candidate on a fresh env ('fresh_env'); otherwise 'replay' is correct and
    cheaper. This removes the last per-game input from the autonomous navigation solver (tu93 -> fresh_env)."""
    from carnot.agentic import arc_solver_kit as kit

    labels = probe_labels or [json.dumps({"action": a}) for a in (4, 2, 3, 1, 4, 2)]
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())

    def _replay_state():
        f = env.reset()
        for lbl in labels:
            f = _generic_apply(env, lbl, f)
        return _generic_state_key(game, f), kit.frame_level(f)

    s1, l1 = _replay_state()
    s2, l2 = _replay_state()
    return "fresh_env" if (s1 != s2 or l1 != l2) else "replay"


def build_nav_adapter(verifier, branch_mode: str):
    """A fully GENERIC navigation GameAdapter -- no per-game code -- carrying the perception verifier."""
    from carnot.agentic.arc_game_adapters import GameAdapter

    return GameAdapter(
        game="__perception_nav__",
        action_labels=_generic_action_labels,
        apply=_generic_apply,
        state_key=_generic_state_key,
        featurize=None,
        hand_verifier=verifier,
        warmup_label=None,
        depth_caps=dict(GENERIC_DEPTH_CAPS),
        branch_mode=branch_mode,
    )


def solve_navigation(game: str, target_level: int = 3, cycles: int = 3) -> dict:
    """The full autonomous pipeline: recon -> derive verifier -> auto branch_mode -> generic
    verifier-routed OfflineSolver -> reproduction gate. Returns a result dict; `path` is the solved action
    labels (None if not a navigation game / no solve)."""
    from carnot.agentic import arc_solver_kit as kit

    trans = recon(game, cycles=cycles)
    verifier, pair = derive_verifier(trans)
    if verifier is None:
        return {"game": game, "is_navigation_game": False, "pair": None, "path": None,
                "reached": 0, "reproduced": False}
    branch = auto_branch_mode(game)
    ad = build_nav_adapter(verifier, branch)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        game, ad.action_labels, ad.apply, ad.state_key,
        warmup_label=ad.warmup_label, verifier=ad.hand_verifier, branch_mode=ad.branch_mode,
    )
    f = solver._replay(env, [])
    cur = kit.frame_level(f)
    full: list = []
    for lvl in range(cur + 1, target_level + 1):
        path, _nodes = solver.solve_level(env, cur, full, ad.depth_caps.get(lvl, 90))
        if path is None:
            break
        search_reached = kit.frame_level(solver.last_frame)
        f = solver._replay(env, full + path)
        cur = max(kit.frame_level(f), search_reached)
        full += path
    gate = kit.reproduce(game, full, ad.apply, warmup_label=ad.warmup_label, claimed_level=cur)
    reproduced = bool(gate.get("reproduced")) if isinstance(gate, dict) else False
    reached = int(gate.get("reached_level", cur)) if isinstance(gate, dict) else int(cur)
    return {"game": game, "is_navigation_game": True, "pair": list(pair), "branch_mode": branch,
            "search_reached": int(cur), "reproduced": reproduced,
            "reproduced_level": reached if reproduced else 0, "path": full, "path_len": len(full)}


class PerceptionNavigationPolicy:
    """Scored-agent policy that self-discovers a navigation solve from perception and replays it -- the same
    `is_done`/`next_move` interface `run_game` and the ARC-AGI-3 Agent drive, and the same plan-then-replay
    shape `CarnotAgentPolicy` uses, but with the plan SELF-DISCOVERED (perception -> verifier-routed search)
    rather than hand-banked. Returns ("RESET", None) once, then the discovered action sequence, then
    (None, None). Falls back to a no-op plan if the game is not a clean navigation game."""

    def __init__(self, game_id: str, target_level: int = 3):
        self.game = str(game_id).split("-", 1)[0]
        self.solve = solve_navigation(self.game, target_level=target_level)
        # decode the label path -> action-int sequence for next_move
        self.plan = [int(json.loads(lbl)["action"]) for lbl in (self.solve.get("path") or [])]
        self.i = 0
        self.reset_sent = False
        self.target = target_level

    def is_done(self, frames=None, latest=None) -> bool:
        return self.reset_sent and self.i >= len(self.plan)

    def next_move(self, frames=None, latest=None):
        if not self.reset_sent:
            self.reset_sent = True
            return ("RESET", None)
        if self.i < len(self.plan):
            a = self.plan[self.i]
            self.i += 1
            return (int(a), None)
        return (None, None)
