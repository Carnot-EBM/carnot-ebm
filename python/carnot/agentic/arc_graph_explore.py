"""Adapter-FREE graph-explore solver for first contact with an un-adaptered ARC
game (Family-A, cf. arXiv:2512.24156). No per-game reverse-engineering: it explores
the offline sim's state-transition graph using the generic salience-prioritized
action candidates (`_action_candidates`: object-centroid clicks + keyboard actions)
and a `GameGraph`, taking untested actions / novel states until a level-up.

This is the fallback the standing loop (scripts/arc_loop_solve.py) uses when a game
has no adapter yet: advance it adapter-free, CAPTURE the winning trajectory, then
that trajectory seeds the game's adapter + trains its verifier (so the next time
it's solved by the efficient verifier-routed loop, not blind exploration).

A basic explorer (random-restart greedy-novelty); it will crack the easier games
and is the right architecture for the rest — upgradeable toward the full SOTA
(frame segmentation + shortest-path-to-untested-state-action) without changing the
loop wiring.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from carnot.agentic.arc_agi3_live_adapter import (
    _action_candidates, _game_action, _game_over, _levels_completed,
)
from carnot.agentic.arc_agi3_world_model import GameGraph, frame_hash, grid_of


def _warm(env, do_warmup):
    f = env.reset()
    if do_warmup:
        # some games consume the first post-reset action (e.g. sc25); burn it
        ids = [c.action_id for c in _action_candidates(f)]
        if ids:
            from arcengine import GameAction
            f = env.step(_game_action(GameAction, ids[0]), data=None)
    return f


def graph_explore_solve(env: Any, start_level: int = 0, *, max_actions: int = 140,
                        restarts: int = 60, warmup: bool = False, seed: int = 0) -> tuple[Optional[list], int]:
    """Explore adapter-free until a level beyond `start_level` completes. Returns
    (trajectory, reached_level). trajectory = [{"action": id, "data": {...}|None}]."""
    from arcengine import GameAction
    rng = random.Random(seed)
    graph = GameGraph("explore")
    global_tested: set = set()           # (state_hash, action_key) tried across restarts
    best_level = start_level

    for _ in range(restarts):
        f = _warm(env, warmup)
        cur = frame_hash(grid_of(f))
        graph.see_node(cur, f)
        traj: list = []
        for _step in range(max_actions):
            cands = _action_candidates(f)
            if not cands:
                break
            fresh = [c for c in cands if (cur, c.key) not in global_tested]
            pool = fresh if fresh else cands
            sel = pool[0] if fresh else pool[rng.randrange(len(pool))]
            global_tested.add((cur, sel.key))
            nf = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                          reasoning={"policy": "graph_explore_adapter_free"})
            if nf is None:
                break
            traj.append({"action": int(sel.action_id), "data": sel.data})
            lvl = _levels_completed(nf)
            if lvl > start_level:
                return traj, lvl                       # solved +1, return the winning trajectory
            best_level = max(best_level, lvl)
            if _game_over(nf):
                break                                  # dead end; restart
            f = nf
            cur = frame_hash(grid_of(f))
            graph.see_node(cur, f)
    return None, best_level


def trajectory_labels(traj: list) -> list[str]:
    """Encode a captured trajectory as replayable labels (for the reproduction gate
    / a trajectory-replay adapter)."""
    import json
    return [json.dumps(step) for step in traj]
