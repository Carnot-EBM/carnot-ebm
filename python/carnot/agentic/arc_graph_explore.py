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


def _components_detailed(grid) -> list:
    """Connected non-background components with (centroid_y, centroid_x, area, color).
    Same 4-neighbour flood fill as world_model.objects(), but also returns area+color
    so candidates can be ordered by VISUAL SALIENCE (segment size × color rarity) —
    the key ingredient from the graph-explore SOTA (arXiv:2512.24156) that lets the
    search try the most salient interactive elements first instead of treating all
    objects uniformly."""
    import numpy as np
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[counts.argmax()])
    mask = grid != bg
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                cy = sum(c[0] for c in cells) // len(cells)
                cx = sum(c[1] for c in cells) // len(cells)
                comps.append((cy, cx, len(cells), int(grid[i, j])))
    return comps


def rich_action_candidates(frame: Any, max_click: int = 48, by_salience: bool = True) -> list:
    """Every detected object is a click candidate (no 12-click cap — the winning
    clicks for e.g. r11l are objects #15/#27 that the cap dropped). Keyboard actions
    unchanged.

    `by_salience` (default on, E1 / arXiv:2512.24156): order the click candidates by
    VISUAL SALIENCE = segment area × color-rarity, so the explorer tries large,
    rare-colored (interactive-looking) objects before small, common-colored (HUD /
    background-texture) ones. Pure ordering change — the trajectory it ultimately
    records is still a valid deterministic replay; it just reaches the win within a
    smaller budget. Set False for the legacy raster order."""
    ids = _available_action_ids(frame)
    out = [ArcAction(a, None, "available_keyboard_action") for a in ids if a != 6]
    if 6 in ids:
        grid = grid_of(frame)
        comps = _components_detailed(grid)
        if by_salience and comps:
            from collections import Counter
            color_cells = Counter(int(v) for v in grid.flatten().tolist())
            # salience: big segments + globally-rare colors score highest
            comps.sort(key=lambda c: c[2] * (1.0 + 1.0 / (1 + color_cells.get(c[3], 0))),
                       reverse=True)
        pts = [(int(cx), int(cy)) for (cy, cx, _area, _color) in comps]
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


def discover_hud_mask(env, warmup: bool, n_probe: int = 4):
    """Deterministically find STEP-DRIVEN HUD cells (score / timer / move-counter) so
    they can be masked OUT of the node-identity hash (E1 / arXiv:2512.24156 status-bar
    masking). A HUD counter advances the SAME way regardless of which action is taken;
    a board cell changes DIFFERENTLY per action. So: probe several distinct first
    actions from reset, and mark any cell that (a) changed from the reset frame AND
    (b) took an IDENTICAL value across all probes. Those are action-invariant = HUD.

    Computed ONCE at search start (a static mask) so node identity stays stationary —
    a drifting mask would alias states mid-search. Returns a bool mask or None."""
    import numpy as np
    from arcengine import GameAction
    base = grid_of(_warm(env, warmup))
    cands = [c for c in _action_candidates(_warm(env, warmup))]
    seen_keys, probes = set(), []
    for c in cands:
        if c.key in seen_keys:
            continue
        seen_keys.add(c.key)
        f = _warm(env, warmup)
        nf = env.step(_game_action(GameAction, c.action_id), data=c.data)
        if nf is None:
            continue
        g = grid_of(nf)
        if g.shape == base.shape:
            probes.append(g)
        if len(probes) >= n_probe:
            break
    if len(probes) < 2:
        return None
    same = np.logical_and.reduce([p == probes[0] for p in probes[1:]])
    changed = probes[0] != base
    mask = same & changed
    return mask if bool(mask.any()) else None


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
                           prefix: Optional[list] = None,
                           mask_hud: bool = False,
                           heuristic=None, heuristic_weight: float = 1.0,
                           stats: Optional[dict] = None
                           ) -> tuple[Optional[list], int]:
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

    `heuristic` (optional `goal_distance(frame_or_grid) -> float`, lower = closer to a
    win): when provided, the frontier is ordered A*-style by `depth + heuristic_weight *
    heuristic(frame)` instead of FIFO. This KEEPS v2's completeness — the depth (g) term
    prevents the greedy-best-first local-minimum trap that makes a pure-heuristic order
    (v3) fail on games like cn04 — while reaching the win with FEWER expansions. A
    goal-distance heuristic's value is EFFICIENCY in a search that already reaches the
    win (the lp85 pattern), NOT making the search solve a game it structurally can't.
    This is the plug-in slot for an LLM-written / captured gap-fill heuristic
    (scripts/arc_gap_fill.py, python/carnot/agentic/gap_fills/). When None, the search
    is byte-for-byte the original pure-BFS (no regression to the proven solves).
    """
    from collections import deque
    from arcengine import GameAction

    prefix = list(prefix or [])

    # E1: optionally mask step-driven HUD cells out of node identity so a ticking
    # score/timer doesn't make every state look new (state-explosion) or alias states.
    hud = discover_hud_mask(env, warmup) if mask_hud else None

    def node_id(frame):
        g = grid_of(frame)
        if hud is not None and hud.shape == g.shape:
            g = g.copy()
            g[hud] = 0
        return frame_hash(g)

    def _candidates(frame):
        return rich_action_candidates(frame)   # salience-ordered, all objects (fixes r11l)

    def replay(path):
        f = _warm(env, warmup)
        for act in path:
            f = env.step(_game_action(GameAction, act["action"]), data=act.get("data"))
        return f

    f0 = replay(prefix)                 # root at the post-prefix state (L0 if no prefix)
    h0 = node_id(f0)
    states = {h0: {"path": list(prefix), "untested": _candidates(f0)}}
    best = start_level
    expansions = 0

    def _ret(traj, lvl):
        # record search cost so an A/B can measure the heuristic's EFFICIENCY win
        # (fewer expansions to the same win) — not just the action count, which ties
        # whenever both arms find the shortest path.
        if stats is not None:
            stats["expansions"] = expansions
            stats["states"] = len(states)
        return traj, lvl

    if heuristic is None:
        # --- pure BFS (UNCHANGED from the original; preserves the proven 8/11 solves) ---
        frontier = deque([h0])          # BFS order ⇒ shortest path first
        while frontier and expansions < max_expansions:
            h = frontier[0]
            st = states[h]
            if not st["untested"] or len(st["path"]) >= max_depth:
                frontier.popleft()
                continue
            sel = st["untested"].pop(0)
            replay(st["path"])          # navigate to this state
            nf = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                          reasoning={"policy": "graph_explore_v2_shortest_path"})
            expansions += 1
            if nf is None:
                continue
            traj = st["path"] + [{"action": int(sel.action_id), "data": sel.data}]
            lvl = _levels_completed(nf)
            if lvl > start_level:
                return _ret(traj, lvl)
            best = max(best, lvl)
            if _game_over(nf):
                continue
            nh = node_id(nf)
            if nh not in states:        # new state ⇒ add to graph + frontier
                states[nh] = {"path": traj, "untested": _candidates(nf)}
                frontier.append(nh)
        return _ret(None, best)

    # --- A*-style heuristic-guided best-first (COMPLETE + efficient) ---
    import heapq
    import itertools

    def _h(frame) -> float:
        try:
            return heuristic_weight * float(heuristic(frame))
        except Exception:
            return 1e9                   # a broken heuristic must never crash the search

    counter = itertools.count()
    # priority = g (depth) + h (weighted goal-distance); root popped first regardless
    heap = [(len(states[h0]["path"]) + _h(f0), next(counter), h0)]
    while heap and expansions < max_expansions:
        _, _, h = heapq.heappop(heap)
        st = states.get(h)
        if st is None or not st["untested"] or len(st["path"]) >= max_depth:
            continue
        # fully expand this (most-promising) state's untested actions (A* graph search:
        # each state expanded once, in priority order)
        while st["untested"]:
            sel = st["untested"].pop(0)
            replay(st["path"])          # navigate to this state
            nf = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                          reasoning={"policy": "graph_explore_v2_heuristic_guided"})
            expansions += 1
            if nf is not None:
                traj = st["path"] + [{"action": int(sel.action_id), "data": sel.data}]
                lvl = _levels_completed(nf)
                if lvl > start_level:
                    return _ret(traj, lvl)
                best = max(best, lvl)
                if not _game_over(nf):
                    nh = node_id(nf)
                    if nh not in states:    # new state ⇒ add with A* priority g+h
                        states[nh] = {"path": traj, "untested": _candidates(nf)}
                        heapq.heappush(heap, (len(traj) + _h(nf), next(counter), nh))
            if expansions >= max_expansions:
                break
    return _ret(None, best)


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
