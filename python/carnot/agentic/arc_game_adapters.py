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

    branch_mode='fresh_env' is LOAD-BEARING here (gotcha #7): tu93's env.reset() is NON-IDEMPOTENT --
    it leaves a parity-toggling hidden state (same path, 6 reset+replays -> levels [1,2,1,2,1,2]). The
    default reuse-one-env 'replay' search therefore detects parity-CONTINGENT 'wins' that fail the
    fresh-env reproduction gate (the gate correctly rejects them -- no false claim), so 'replay' only
    reproduces L1. Evaluating EVERY candidate on a brand-new env (fresh_env mode) makes each see the
    same pristine parity-0 the gate uses, so found paths reproduce. (deepcopy mode does NOT work for
    tu93 -- its env._game is not deepcopy-injectable, gotcha #3, like sc25.)

    VALIDATED 2026-06-17: with branch_mode='fresh_env' the adapter DEEP-SOLVES to L3 reproducibly
    (47 moves, offline_reproduced=True), vs L1 under replay. featurize is None (the learned verifier
    is fed env._game internals via collect_trajectory_data, which this frame-based RE doesn't read)."""
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
        branch_mode="fresh_env",   # gotcha #7: tu93 reset is non-idempotent -> fresh env per node
    )


# ---------------- tr87 (glyph-substitution configuration puzzle; RE'd 2026-06-17) ----------------
def _tr87():
    """tr87 -- a GLYPH-SUBSTITUTION configuration puzzle (RE'd 2026-06-17, frame + internal-state).

    Mechanic: a row of 5 EDITABLE glyphs (sprite series 'B', each a value 1-7) must be set to match a
    TARGET row (series 'A', values 1-7) THROUGH a substitution rule. The visible top reference grid IS
    the rule -- it pairs A-values with B-values (e.g. A4<->B3). Win = for every position i,
    value(editable_i) == rule_map[value(target_i)]. ACTION1/ACTION2 cycle the SELECTED glyph's value
    (-1/+1 mod 7); ACTION3/ACTION4 move the selector among the 5 glyphs. A move budget (128) decrements
    per action; running out loses. (Later levels add alter_rules / tree_translation / double_translation
    twists; L1-L5 of the base mechanic are handled here.)

    VALIDATED: solves L1 reproducibly (15 moves, offline_reproduced=True). The hand_verifier reads the
    game's internal config -- the rule map (cifzvbcuwqe) + target (zvojhrjxxm) + current (ztgmtnnufb) --
    and returns the count of positions NOT yet at their rule-mapped required value (0 == win). This is
    the SAME internal-state-reading pattern as the lp85 adapter (_goal_key); it routes the best-first
    search to set each glyph to its target. Frame-only perception (classifying glyph bitmaps + decoding
    the rule grid from pixels) is a future upgrade; the solve is reproduction-gated regardless (the gate
    replays ACTIONS, not internal reads). branch_mode='replay' (tr87's reset is idempotent + the config
    is a deterministic function of the action prefix). state_key is the full-grid hash so the win
    animation frames stay distinct (the search can traverse them to the level-up)."""
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def _val(s):
        return int(s.name[-1])                          # glyph value = trailing digit of the sprite name

    def _mismatches(game):
        # rule map: A-value -> B-value (the top reference grid, read once per call)
        amap = {int(lhs[0].name[-1]): int(rhs[0].name[-1]) for lhs, rhs in game.cifzvbcuwqe}
        cur = [_val(s) for s in game.ztgmtnnufb]         # current editable B-values
        req = [amap.get(_val(s)) for s in game.zvojhrjxxm]  # required B-value per target A-value
        return sum(1 for c, r in zip(cur, req) if c != r)

    def action_labels(env, frame=None, path=None):
        return [json.dumps({"action": a}) for a in (1, 2, 3, 4)]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, json.loads(label)["action"]))

    def state_key(game, frame=None):
        # full-grid hash: distinguishes every (selector, glyph-values) config AND the win-animation
        # frames (where the config is frozen but the grid changes), so the search reaches the level-up.
        return frame_hash(grid_of(frame)) if frame is not None else None

    def hand_verifier(game, frame=None):
        # goal-distance = positions not yet at their rule-mapped target value (0 == win). Internal-state
        # read (the lp85 pattern); routes the best-first search. Guarded so a malformed level never crashes.
        try:
            return float(_mismatches(game))
        except Exception:
            return 1000.0

    return GameAdapter(
        game="tr87", action_labels=action_labels, apply=apply, state_key=state_key,
        featurize=None, hand_verifier=hand_verifier, warmup_label=None,
        depth_caps={1: 40, 2: 60, 3: 90, 4: 90, 5: 90}, branch_mode="replay",
    )


_BUILDERS = {"lp85": _lp85, "tu93": _tu93, "tr87": _tr87}


def get_adapter(game: str) -> Optional[GameAdapter]:
    """Return the adapter for `game`, or None if it hasn't been RE'd/registered yet."""
    b = _BUILDERS.get(game)
    return b() if b else None


def adaptered_games() -> list[str]:
    return sorted(_BUILDERS)
