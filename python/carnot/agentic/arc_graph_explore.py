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
    ArcAction, _action_candidates, _available_action_ids, _game_action, _game_over, _levels_completed,
)
from carnot.agentic.arc_agi3_world_model import GameGraph, frame_hash, grid_of, objects


def rich_action_candidates(frame: Any, max_click: int = 48) -> list:
    """Like _action_candidates but WITHOUT the 12-click cap — every detected object
    is a click candidate (the winning clicks for e.g. r11l are objects #15/#27 that
    the cap dropped). Keyboard actions unchanged."""
    ids = _available_action_ids(frame)
    out = [ArcAction(a, None, "available_keyboard_action") for a in ids if a != 6]
    if 6 in ids:
        grid = grid_of(frame)
        pts = [(int(x), int(y)) for y, x in objects(grid)]
        if not pts:
            h, w = grid.shape
            pts = [(w // 2, h // 2)]
        seen: set = set()
        for x, y in pts[:max_click]:
            p = (max(0, int(x)), max(0, int(y)))
            if p in seen:
                continue
            seen.add(p)
            out.append(ArcAction(6, {"x": p[0], "y": p[1]}, "object_click"))
    return out


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


def graph_explore_solve_v2(env: Any, start_level: int = 0, *, max_expansions: int = 6000,
                           warmup: bool = False, max_depth: int = 60,
                           prefix: Optional[list] = None) -> tuple[Optional[list], int]:
    """SYSTEMATIC graph-explore (toward arXiv:2512.24156): maintain a directed
    state-transition graph and take the SHORTEST PATH to a state with an untested
    state-action pair (BFS frontier), navigating by replay-from-reset (deepcopy-
    injection is unreliable). Complete over the reachable state-action space up to
    the budget — far stronger than greedy-restart. Returns (trajectory, reached_level).

    `prefix` (optional) is a KNOWN winning trajectory that gets the env to a starting
    state (e.g. the L1 solution); the search is ROOTED at the post-prefix state and
    only explores the frontier BEYOND it. Pair with `start_level` = the level the
    prefix reaches, so the search returns the full prefix+suffix trajectory to the
    NEXT level. This is the INCREMENTAL-PROGRESS lever: pin what we know, explore only
    the new frontier — far cheaper than re-discovering the early levels from L0.
    """
    from collections import deque
    from arcengine import GameAction

    prefix = list(prefix or [])

    def _candidates(frame):
        return rich_action_candidates(frame)   # all objects, no 12-cap (fixes r11l)

    def replay(path):
        f = _warm(env, warmup)
        for act in path:
            f = env.step(_game_action(GameAction, act["action"]), data=act.get("data"))
        return f

    f0 = replay(prefix)                 # root at the post-prefix state (L0 if no prefix)
    h0 = frame_hash(grid_of(f0))
    states = {h0: {"path": list(prefix), "untested": _candidates(f0)}}
    frontier = deque([h0])              # BFS order ⇒ shortest path first
    best = start_level
    expansions = 0
    while frontier and expansions < max_expansions:
        h = frontier[0]
        st = states[h]
        if not st["untested"] or len(st["path"]) >= max_depth:
            frontier.popleft()
            continue
        sel = st["untested"].pop(0)
        replay(st["path"])              # navigate to this state
        nf = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                      reasoning={"policy": "graph_explore_v2_shortest_path"})
        expansions += 1
        if nf is None:
            continue
        traj = st["path"] + [{"action": int(sel.action_id), "data": sel.data}]
        lvl = _levels_completed(nf)
        if lvl > start_level:
            return traj, lvl
        best = max(best, lvl)
        if _game_over(nf):
            continue
        nh = frame_hash(grid_of(nf))
        if nh not in states:           # new state ⇒ add to graph + frontier
            states[nh] = {"path": traj, "untested": _candidates(nf)}
            frontier.append(nh)
    return None, best


def graph_explore_solve_v3(env: Any, start_level: int = 0, *, max_expansions: int = 30000,
                           warmup: bool = False, max_depth: int = 80,
                           verifier=None) -> tuple[Optional[list], int]:
    """Value/novelty-guided graph-explore for DEEP games (e.g. wa30 ~33-deep keyboard)
    where uniform BFS exhausts its budget before reaching the win. Best-first over the
    frontier by: an optional VERIFIER (predicted steps-to-go on the frame, the learned
    verifier feeding back) else count-based NOVELTY (least-visited coarse-region first)
    with a depth bias to push deeper. Only frame-CHANGING transitions are enqueued
    (skips wall-bump no-ops that waste the budget). Replay-navigation. Returns
    (trajectory, reached_level)."""
    import heapq
    import itertools
    from arcengine import GameAction

    def replay(path):
        f = _warm(env, warmup)
        for act in path:
            f = env.step(_game_action(GameAction, act["action"]), data=act.get("data"))
        return f

    def coarse(frame):
        g = grid_of(frame)
        return (int((g != 0).sum()) // 8, len(set(g.flatten().tolist())))

    def priority(frame, depth):
        if verifier is not None:
            return float(verifier(frame))          # lower predicted steps-to-go = better
        return float(region_visits[coarse(frame)] - 0.25 * depth)  # novelty, push deeper

    f0 = _warm(env, warmup)
    h0 = frame_hash(grid_of(f0))
    region_visits: dict = {coarse(f0): 1}
    states = {h0: {"path": [], "untested": rich_action_candidates(f0)}}
    counter = itertools.count()
    heap = [(priority(f0, 0), next(counter), h0)]
    best = start_level
    expansions = 0
    while heap and expansions < max_expansions:
        _, _, h = heapq.heappop(heap)
        st = states.get(h)
        if st is None or not st["untested"] or len(st["path"]) >= max_depth:
            continue
        # expand ALL untested actions of this state (re-push if any remain handled by new states)
        f_here = replay(st["path"])
        here_hash = frame_hash(grid_of(f_here))
        while st["untested"]:
            sel = st["untested"].pop(0)
            replay(st["path"])
            nf = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                          reasoning={"policy": "graph_explore_v3_value_guided"})
            expansions += 1
            if nf is None:
                continue
            traj = st["path"] + [{"action": int(sel.action_id), "data": sel.data}]
            lvl = _levels_completed(nf)
            if lvl > start_level:
                return traj, lvl
            best = max(best, lvl)
            if _game_over(nf):
                continue
            nh = frame_hash(grid_of(nf))
            if nh == here_hash or nh in states:
                continue                            # no-op (wall bump) or seen ⇒ skip
            states[nh] = {"path": traj, "untested": rich_action_candidates(nf)}
            reg = coarse(nf)
            region_visits[reg] = region_visits.get(reg, 0) + 1
            heapq.heappush(heap, (priority(nf, len(traj)), next(counter), nh))
            if expansions >= max_expansions:
                break
    return None, best


def trajectory_labels(traj: list) -> list[str]:
    """Encode a captured trajectory as replayable labels (for the reproduction gate
    / a trajectory-replay adapter)."""
    import json
    return [json.dumps(step) for step in traj]
