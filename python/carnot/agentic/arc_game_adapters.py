"""Per-game ADAPTERS for the standing ARC learning loop — the output of each
game's (irreducible) per-game reverse-engineering, captured as a reusable plug-in
for arc_solver_kit.OfflineSolver. A new game, once its win/action/state delta is
RE'd, registers an adapter here and is then solvable + reproducible + learnable by
the standing loop (scripts/arc_loop_solve.py) with no further bespoke code.

An adapter provides the four game-specific callables the kit needs:
  action_labels(env) -> [str]   : env-discovered action vocabulary
  apply(env, label, frame)      : execute one action, return the new frame
  state_key(game)               : the dedup key (every load-bearing piece of state)
  featurize(game) -> [float]    : features for the LEARNED verifier (optional)
plus optional warmup_label and a hand verifier (goal-distance) for cold start.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from arcengine import GameAction


@dataclass
class GameAdapter:
    game: str
    action_labels: Callable[[Any], Sequence[str]]
    apply: Callable[[Any, str, Any], Any]
    state_key: Callable[[Any], Any]
    featurize: Optional[Callable[[Any], Sequence[float]]] = None
    hand_verifier: Optional[Callable[[Any], float]] = None
    warmup_label: Optional[str] = None
    depth_caps: dict = field(default_factory=lambda: {})
    # how the OfflineSolver navigates between search nodes: "replay" (default; replay-from-reset) or
    # "deepcopy" (snapshot/restore env._game per node). Use "deepcopy" only for a game whose env is
    # deepcopy-injectable AND whose replay-from-reset doesn't faithfully reproduce the searched state.
    branch_mode: str = "replay"


# ---------------- lp85 (reference adapter; click-only rotation puzzle) ----------------
def _lp85():
    from carnot.experiment_4179_arc_incremental_progress import (
        discover_click_buttons, _goal_key, _target_goal_key,
    )

    def action_labels(env):
        return [json.dumps({"x": int(b["x"]), "y": int(b["y"])}) for b in discover_click_buttons(env)]

    def apply(env, label, frame):
        a = json.loads(label)
        return env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})

    def _dists(game):
        actual = _goal_key(game)
        target = _target_goal_key(game)
        by_type = defaultdict(list)
        for t, x, y in actual:
            by_type[t].append((x, y))
        return [min((abs(tx - x) + abs(ty - y)) for x, y in by_type.get(t, [])) if by_type.get(t) else 1000.0
                for t, tx, ty in target]

    def featurize(game):
        ds = _dists(game)
        n = len(ds) or 1
        return [sum(ds), float(sum(1 for d in ds if d > 0)), sum(ds) / n, float(max(ds) if ds else 0), float(n)]

    return GameAdapter(
        game="lp85", action_labels=action_labels, apply=apply, state_key=_goal_key,
        featurize=featurize, hand_verifier=lambda g: float(sum(_dists(g))),
        depth_caps={1: 20, 2: 70, 3: 90},
    )


# ---------------- tu93 (4-direction keyboard maze; frame-based RE) ----------------
def _tu93():
    """tu93 -- a 4-direction keyboard maze (ACTION1-4). FRAME-BASED RE (no internal-state read): the
    PLAYER is the moving colour-9 sprite, the GOAL is the static colour-14 marker (RE'd 2026-06-17 by
    motion: only colour-9 + the colour-4 key drift across moves; colour-14 is static). The
    hand_verifier is the player->goal Manhattan distance (goal-distance-routed best-first search).

    VALIDATED: solves L1 reproducibly (18 moves, offline_reproduced=True). KNOWN LIMITATION at L2+: tu93
    carries HIDDEN/RNG state (beyond the rendered grid) that the OfflineSolver's REUSED single env
    accumulates across the search, so a verifier-found L2 path does NOT reproduce on a fresh env (the
    reproduction gate correctly rejects it -- no false claim). graph_explore avoids this by DEEPCOPYING
    the env per node; the OfflineSolver does not. Deep-solving tu93 via this adapter therefore needs a
    deepcopy-per-node (fresh-env) mode in arc_solver_kit.OfflineSolver -- a kit-level change, not an
    adapter fix. featurize is None (the learned verifier is fed env._game internals via
    collect_trajectory_data, which this frame-based RE doesn't read)."""
    import numpy as np
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    PLAYER, GOAL = 9, 14

    def _grid2d(frame):
        g = grid_of(frame)
        if g.ndim == 1:                                # some stepped frames flatten -> reshape square
            s = int(round(g.size ** 0.5))
            if s * s == g.size:
                g = g.reshape(s, s)
        return g

    def _centroid(g, col):
        ys, xs = np.where(g == col)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    def action_labels(env, frame=None, path=None):
        av = list(getattr(frame, "available_actions", []) or []) if frame is not None else []
        moves = [a for a in av if a in (1, 2, 3, 4)] or [1, 2, 3, 4]
        return [json.dumps({"action": int(a)}) for a in moves]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, json.loads(label)["action"]))

    def state_key(game, frame=None):
        # frame-based dedup: the FULL grid (player position + maze + key + the blocked-move counter).
        # Do NOT mask any region -- the corner counter is LOAD-BEARING (masking it collapses distinct
        # states and yields a non-reproducing path; the full-grid hash is what graph_explore reproduces
        # tu93 with).
        if frame is None:
            return None
        return _grid2d(frame).tobytes()

    def hand_verifier(game, frame=None):
        if frame is None:
            return 1000.0
        g = _grid2d(frame)
        p, t = _centroid(g, PLAYER), _centroid(g, GOAL)
        if p is None or t is None:
            return 1000.0
        return abs(p[0] - t[0]) + abs(p[1] - t[1])     # lower == closer to the goal

    return GameAdapter(
        game="tu93", action_labels=action_labels, apply=apply, state_key=state_key,
        featurize=None, hand_verifier=hand_verifier, warmup_label=None,
        depth_caps={1: 40, 2: 60, 3: 80, 4: 90, 5: 90},
    )


_BUILDERS = {"lp85": _lp85, "tu93": _tu93}


def get_adapter(game: str) -> Optional[GameAdapter]:
    """Return the adapter for `game`, or None if it hasn't been RE'd/registered yet."""
    b = _BUILDERS.get(game)
    return b() if b else None


def adaptered_games() -> list[str]:
    return sorted(_BUILDERS)
