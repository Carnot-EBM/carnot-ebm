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
    ArcAction,
    _action_candidates,
    _available_action_ids,
    _game_action,
    _game_over,
    _levels_completed,
)
from carnot.agentic.arc_frame_change_predictor import (
    ActionEffectExpansionPrior,
    prune_arc_actions,
    prune_arc_actions_by_prior_quantile,
    rank_arc_actions,
)
from carnot.agentic.arc_agi3_world_model import GameGraph, frame_hash, grid_of, objects
from carnot.agentic.arc_energy_fitness_qd import coerce_qd_generator
from carnot.agentic.arc_goal_energy_live import make_goal_energy_heuristic


def _components_detailed(grid) -> list:
    """Connected non-background components with (centroid_y, centroid_x, area, color).
    Same 4-neighbour flood fill as world_model.objects(), but also returns area+color
    so candidates can be ordered by VISUAL SALIENCE (segment size × color rarity) —
    the key ingredient from the graph-explore SOTA (arXiv:2512.24156) that lets the
    search try the most salient interactive elements first instead of treating all
    objects uniformly."""
    from carnot.agentic.arc_solver_kit import object_centric_digest

    comps = []
    for comp in object_centric_digest(grid)["components"]:
        cx, cy = comp["centroid"]
        comps.append((int(cy), int(cx), int(comp["area"]), int(comp["color"])))
    return comps


# just-explore's 5-tier salience schedule (heuristic_agent.py:frame_segments_to_action_groups,
# arXiv:2512.24156). The 2026-06-23 offline head-to-head showed this SCHEDULE reaches first-wins our
# flat area*rarity sort misses on 5 games (bp35/ft09/m0r0/r11l/vc33). It front-loads BUTTON-LIKE objects
# (salient colour AND medium bounding-box) and defers very-large / dull / status-bar segments, where our
# flat sort up-ranks the largest area first. Constants are just-explore's verbatim.
_TIER_SALIENT_COLORS = frozenset(range(6, 16))  # {6..15} (non-salient = {0..5})
_TIER_STATUS_BAR_COLOR = 16
_TIER_MIN_WIDTH = 2
_TIER_MAX_WIDTH = 32


def _tier_ordered_click_points(grid) -> list:
    """Object-click (x, y) points ordered by just-explore's 5 salience tiers (T0 first):
    T0 salient AND medium-width, T1 medium-width, T2 salient, T3 other, T4 status-bar. Stable
    secondary sort by descending area. x = centroid[0], y = centroid[1] (matching the flat path)."""
    from carnot.agentic.arc_solver_kit import object_centric_digest

    def _tier(comp) -> int:
        bb = comp["bbox"]  # [min_row, min_col, max_row, max_col]
        h = bb[2] - bb[0] + 1
        w = bb[3] - bb[1] + 1
        color = int(comp["color"])
        salient = color in _TIER_SALIENT_COLORS
        medium = _TIER_MIN_WIDTH <= w <= _TIER_MAX_WIDTH and _TIER_MIN_WIDTH <= h <= _TIER_MAX_WIDTH
        if color == _TIER_STATUS_BAR_COLOR:
            return 4
        if salient and medium:
            return 0
        if medium:
            return 1
        if salient:
            return 2
        return 3

    comps = object_centric_digest(grid)["components"]
    comps_sorted = sorted(comps, key=lambda c: (_tier(c), -int(c["area"])))
    return [(int(c["centroid"][0]), int(c["centroid"][1])) for c in comps_sorted]


def rich_action_candidates(
    frame: Any,
    max_click: int = 48,
    by_salience: bool = True,
    frame_change_scorer: Any | None = None,
    frame_change_prune_threshold: float | None = None,
    action_prior: Any | None = None,
    action_prior_prune_quantile: float | None = None,
    structural_energy_scorer: Any | None = None,
    candidate_router: Any | None = None,
    previous_frame: Any | None = None,
) -> list:
    """Every detected object is a click candidate (no 12-click cap — the winning
    clicks for e.g. r11l are objects #15/#27 that the cap dropped). Keyboard actions
    unchanged.

    `by_salience` (default on, E1 / arXiv:2512.24156): order the click candidates by
    VISUAL SALIENCE = segment area × color-rarity, so the explorer tries large,
    rare-colored (interactive-looking) objects before small, common-colored (HUD /
    background-texture) ones. Pure ordering change — the trajectory it ultimately
    records is still a valid deterministic replay; it just reaches the win within a
    smaller budget. Set False for the legacy raster order.

    REQ-ARC-FCP-4491/4493: when a frame-change scorer, human behavior prior, or
    structural energy scorer is supplied, rank the same candidate set by predicted
    action effect while preserving this salience/raster order as the stable tie-break.

    REQ-ARC-FCP-4511: when ``frame_change_prune_threshold`` is supplied with a
    frame-change scorer, predicted no-op candidates are removed before the
    explorer ever expands them.

    REQ-ARC-FCP-4512: when ``action_prior_prune_quantile`` is supplied with an
    action prior, the bottom prior-likelihood quantile is removed before
    expansion while retaining at least one candidate.

    REQ-CAPSTONE-4556: when ``candidate_router`` is supplied, apply its learned
    cross-game ordering as the final candidate-router pass. A scoring failure
    keeps the bare order so the live solver has a no-regression fallback."""
    ids = _available_action_ids(frame)
    out = [ArcAction(a, None, "available_keyboard_action") for a in ids if a != 6]
    if 6 in ids:
        import os

        grid = grid_of(frame)
        # CARNOT_ARC_TIER_SCHEDULE=1 orders object-clicks by just-explore's 5 salience tiers (button-like
        # medium-width salient objects first) instead of the flat area*rarity sort. Default off -> the
        # path below is byte-identical to the proven order (parity preserved; the SUBMITTED agent unchanged
        # until the A/B greenlights it). A scoring failure falls back to the flat order (no-regression).
        if os.environ.get("CARNOT_ARC_TIER_SCHEDULE") == "1":
            try:
                pts = _tier_ordered_click_points(grid) or None  # None -> fall back to the flat order
            except Exception:
                pts = None
        else:
            pts = None
        if pts is None:
            comps = _components_detailed(grid)
            if by_salience and comps:
                from collections import Counter

                color_cells = Counter(int(v) for v in grid.flatten().tolist())
                # salience: big segments + globally-rare colors score highest
                comps.sort(
                    key=lambda c: c[2] * (1.0 + 1.0 / (1 + color_cells.get(c[3], 0))), reverse=True
                )
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
    if frame_change_scorer is not None and frame_change_prune_threshold is not None:
        out, _diagnostics = prune_arc_actions(
            frame,
            out,
            scorer=frame_change_scorer,
            threshold=frame_change_prune_threshold,
        )
    if action_prior is not None and action_prior_prune_quantile is not None:
        out, _diagnostics = prune_arc_actions_by_prior_quantile(
            frame,
            out,
            prior=action_prior,
            prune_quantile=action_prior_prune_quantile,
        )
    ranker_present = (
        frame_change_scorer is not None
        or action_prior is not None
        or structural_energy_scorer is not None
    )
    if candidate_router is not None and out:
        try:
            if hasattr(candidate_router, "rank"):
                ranked = candidate_router.rank(frame, out, previous_frame=previous_frame)
            else:
                ranked = candidate_router(frame, out)
            out = list(ranked)
        except Exception:
            pass
    if (
        structural_energy_scorer is not None
        and frame_change_scorer is None
        and action_prior is None
    ):
        scored = []
        for index, candidate in enumerate(out):
            try:
                if hasattr(structural_energy_scorer, "candidate_delta_energy"):
                    delta_energy = structural_energy_scorer.candidate_delta_energy(frame, candidate)
                else:
                    delta_energy = structural_energy_scorer(frame, candidate)
                score = -float(delta_energy)
            except Exception:
                score = 0.0
        scored.append((score, index, candidate))
        scored.sort(key=lambda row: (-row[0], row[1]))
        out = [candidate for _score, _index, candidate in scored]
    elif ranker_present:
        out = rank_arc_actions(
            frame,
            out,
            scorer=frame_change_scorer,
            prior=action_prior,
            structural_energy_scorer=structural_energy_scorer,
        )
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


def graph_explore_solve(
    env: Any,
    start_level: int = 0,
    *,
    max_actions: int = 140,
    restarts: int = 60,
    warmup: bool = False,
    seed: int = 0,
) -> tuple[Optional[list], int]:
    """Explore adapter-free until a level beyond `start_level` completes. Returns
    (trajectory, reached_level). trajectory = [{"action": id, "data": {...}|None}]."""
    from arcengine import GameAction

    rng = random.Random(seed)
    graph = GameGraph("explore")
    global_tested: set = set()  # (state_hash, action_key) tried across restarts
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
            nf = env.step(
                _game_action(GameAction, sel.action_id),
                data=sel.data,
                reasoning={"policy": "graph_explore_adapter_free"},
            )
            if nf is None:
                break
            traj.append({"action": int(sel.action_id), "data": sel.data})
            lvl = _levels_completed(nf)
            if lvl > start_level:
                return traj, lvl  # solved +1, return the winning trajectory
            best_level = max(best_level, lvl)
            if _game_over(nf):
                break  # dead end; restart
            f = nf
            cur = frame_hash(grid_of(f))
            graph.see_node(cur, f)
    return None, best_level


def graph_explore_solve_v2(
    env: Any,
    start_level: int = 0,
    *,
    max_expansions: int = 6000,
    warmup: bool = False,
    max_depth: int = 60,
    prefix: Optional[list] = None,
    mask_hud: bool = False,
    heuristic=None,
    heuristic_weight: float = 1.0,
    goal_energy=None,
    goal_energy_alpha: float = 0.9,
    goal_energy_beta: float = 0.1,
    emit_plan_only_when_goal_predicate_fires: bool = False,
    expansion_priority=None,
    frame_change_scorer=None,
    frame_change_prune_threshold: float | None = None,
    action_prior=None,
    action_prior_prune_quantile: float | None = None,
    action_effect_expansion_prior: Any | bool | None = None,
    qd_generator: Any | bool | None = None,
    candidate_router=None,
    structural_energy_scorer=None,
    move_pruner=None,
    stats: Optional[dict] = None,
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

    `expansion_priority` is the generic REQ-CAPSTONE-4569 hook: a learned frontier-node
    scorer (lower = expand earlier). It uses the same bounded best-first queue as
    `heuristic`, but is named for the verifier-guided expansion use case to keep it
    distinct from action candidate re-ranking.

    `action_effect_expansion_prior` is the REQ-ARC-FCP-4641 hook: when enabled
    with the same frame-change scorer used for candidate ranking, frontier
    states whose remaining untested actions are predicted to change the frame
    are expanded before predicted no-op branches.

    `goal_energy` is the REQ-ARC-WMTE-4640 hook: Exp4020's visible-state
    goal-satisfaction energy can be convex-combined with the navigation heuristic as
    alpha*navigation + beta*goal_energy. When
    `emit_plan_only_when_goal_predicate_fires` is true, a level-up trajectory is
    returned only if the visible predicate fires on the terminal frame.

    `qd_generator` is the REQ-ARC-WMTE-4653 hook: an additive MAP-Elites
    multi-action sequence generator. It injects a generated sequence into the
    same scored pool while leaving primitive actions available as the fallback.
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

    def _candidates(frame, previous_frame=None):
        return rich_action_candidates(
            frame,
            frame_change_scorer=frame_change_scorer,
            frame_change_prune_threshold=frame_change_prune_threshold,
            action_prior=action_prior,
            action_prior_prune_quantile=action_prior_prune_quantile,
            structural_energy_scorer=structural_energy_scorer,
            candidate_router=candidate_router,
            previous_frame=previous_frame,
        )  # salience-ordered, all objects (fixes r11l)

    def replay(path):
        f = _warm(env, warmup)
        for act in path:
            f = env.step(_game_action(GameAction, act["action"]), data=act.get("data"))
        return f

    f0 = replay(prefix)  # root at the post-prefix state (L0 if no prefix)
    h0 = node_id(f0)
    states = {h0: {"path": list(prefix), "untested": _candidates(f0), "frame": f0}}
    best = start_level
    expansions = 0
    qd_search_generator = coerce_qd_generator(
        qd_generator,
        action_effect_scorer=frame_change_scorer,
        goal_energy=goal_energy,
    )
    qd_sequences_injected = 0
    qd_actions_injected = 0
    move_pruned = 0

    def _label(action_id, data):
        return {"action": int(action_id), "data": data}

    def _should_prune(frame, label) -> bool:
        nonlocal move_pruned
        if move_pruner is None:
            return False
        try:
            pruned = bool(move_pruner.should_prune(frame, label))
        except Exception:
            return False
        if pruned:
            move_pruned += 1
        return pruned

    def _observe(frame_before, label, frame_after, leveled_up: bool) -> None:
        if move_pruner is None or frame_after is None:
            return
        try:
            move_pruner.observe(frame_before, label, frame_after, leveled_up)
        except Exception:
            pass

    def _ret(traj, lvl):
        # record search cost so an A/B can measure the heuristic's EFFICIENCY win
        # (fewer expansions to the same win) — not just the action count, which ties
        # whenever both arms find the shortest path.
        if stats is not None:
            stats["expansions"] = expansions
            stats["states"] = len(states)
            stats["max_expansions"] = int(max_expansions)
            stats["proposal_prior_enabled"] = structural_energy_scorer is not None
            stats["expansion_priority_enabled"] = (
                expansion_priority is not None
                or heuristic is not None
                or action_effect_frontier_prior is not None
            )
            stats["action_effect_expansion_prior_enabled"] = (
                action_effect_frontier_prior is not None
            )
            stats["goal_energy_enabled"] = goal_energy is not None
            stats["goal_energy_alpha"] = (
                float(goal_energy_alpha) if goal_energy is not None else 0.0
            )
            stats["goal_energy_beta"] = float(goal_energy_beta) if goal_energy is not None else 0.0
            stats["goal_predicate_gate_enabled"] = bool(emit_plan_only_when_goal_predicate_fires)
            stats.setdefault("goal_predicate_plan_emitted", False)
            stats["qd_generation_enabled"] = qd_search_generator is not None
            stats["qd_sequences_injected"] = int(qd_sequences_injected)
            stats["qd_actions_injected"] = int(qd_actions_injected)
            stats["move_pruner_enabled"] = move_pruner is not None
            stats["move_pruned"] = int(move_pruned)
            if move_pruner is not None and hasattr(move_pruner, "stats"):
                try:
                    stats["move_pruner_stats"] = move_pruner.stats()
                except Exception:
                    stats["move_pruner_stats"] = None
            if qd_search_generator is not None and hasattr(qd_search_generator, "diagnostics"):
                stats["qd_generation_diagnostics"] = qd_search_generator.diagnostics()
        return traj, lvl

    if hasattr(action_effect_expansion_prior, "frontier_priority"):
        action_effect_frontier_prior = action_effect_expansion_prior
    elif action_effect_expansion_prior and frame_change_scorer is not None:
        action_effect_frontier_prior = ActionEffectExpansionPrior(frame_change_scorer)
    else:
        action_effect_frontier_prior = None

    navigation_scorer = expansion_priority if expansion_priority is not None else heuristic
    if goal_energy is not None:
        priority_scorer = make_goal_energy_heuristic(
            navigation_energy=navigation_scorer,
            goal_energy=goal_energy,
            alpha=float(goal_energy_alpha),
            beta=float(goal_energy_beta),
        )
    else:
        priority_scorer = navigation_scorer

    def _predicate_allows_emit(frame) -> bool:
        if not emit_plan_only_when_goal_predicate_fires:
            return True
        predicate = getattr(priority_scorer, "predicate_fires", None)
        allowed = bool(callable(predicate) and predicate(frame))
        if stats is not None and not allowed:
            stats["goal_predicate_rejected_levelups"] = (
                int(stats.get("goal_predicate_rejected_levelups") or 0) + 1
            )
        return allowed

    def _mark_goal_plan_emitted() -> None:
        if stats is not None and emit_plan_only_when_goal_predicate_fires:
            stats["goal_predicate_plan_emitted"] = True

    def _next_qd_sequence(frame, node: dict) -> list[dict]:
        nonlocal qd_sequences_injected, qd_actions_injected
        if qd_search_generator is None or node.get("qd_sequence_injected"):
            return []
        candidates = list(node.get("untested") or [])
        if not candidates:
            return []
        try:
            sequence = qd_search_generator.best_sequence(
                frame,
                candidates,
                goal_energy=goal_energy,
                action_effect_scorer=frame_change_scorer,
                min_len=2,
            )
        except Exception:
            return []
        rows = [dict(step) for step in sequence if step.get("action") is not None]
        if len(rows) < 2:
            return []
        node["qd_sequence_injected"] = True
        qd_sequences_injected += 1
        qd_actions_injected += len(rows)
        return rows

    def _apply_qd_sequence(state: dict, frame_here, sequence: list[dict], *, policy: str):
        nonlocal expansions, best
        traj = list(state["path"])
        nf = frame_here
        for step in sequence:
            nf = env.step(
                _game_action(GameAction, int(step["action"])),
                data=step.get("data"),
                reasoning={"policy": policy, "generator": "energy_fitness_qd"},
            )
            expansions += 1
            if nf is None:
                return True, None
            traj = traj + [{"action": int(step["action"]), "data": step.get("data")}]
            lvl = _levels_completed(nf)
            if lvl > start_level and _predicate_allows_emit(nf):
                _mark_goal_plan_emitted()
                return True, _ret(traj, lvl)
            best = max(best, lvl)
            if _game_over(nf) or expansions >= max_expansions:
                return True, None
        if nf is not None and not _game_over(nf):
            nh = node_id(nf)
            if nh not in states:
                states[nh] = {
                    "path": traj,
                    "untested": _candidates(nf, previous_frame=frame_here),
                    "frame": nf,
                }
                return True, ("new_state", nh)
        return True, None

    if priority_scorer is None and action_effect_frontier_prior is None:
        # --- pure BFS (UNCHANGED from the original; preserves the proven 8/11 solves) ---
        frontier = deque([h0])  # BFS order ⇒ shortest path first
        while frontier and expansions < max_expansions:
            h = frontier[0]
            st = states[h]
            if not st["untested"] or len(st["path"]) >= max_depth:
                frontier.popleft()
                continue
            f_here = replay(st["path"])  # navigate to this state
            qd_sequence = _next_qd_sequence(f_here, st)
            if qd_sequence:
                handled, result = _apply_qd_sequence(
                    st,
                    f_here,
                    qd_sequence,
                    policy="graph_explore_v2_qd_sequence",
                )
                if isinstance(result, tuple) and result and result[0] != "new_state":
                    return result
                if isinstance(result, tuple) and result and result[0] == "new_state":
                    frontier.append(result[1])
                if handled and expansions >= max_expansions:
                    break
                continue
            sel = st["untested"].pop(0)
            label = _label(sel.action_id, sel.data)
            if _should_prune(f_here, label):
                continue
            nf = env.step(
                _game_action(GameAction, sel.action_id),
                data=sel.data,
                reasoning={"policy": "graph_explore_v2_shortest_path"},
            )
            expansions += 1
            if nf is None:
                continue
            traj = st["path"] + [{"action": int(sel.action_id), "data": sel.data}]
            lvl = _levels_completed(nf)
            _observe(f_here, label, nf, lvl > start_level)
            if lvl > start_level and _predicate_allows_emit(nf):
                _mark_goal_plan_emitted()
                return _ret(traj, lvl)
            best = max(best, lvl)
            if _game_over(nf):
                continue
            nh = node_id(nf)
            if nh not in states:  # new state ⇒ add to graph + frontier
                states[nh] = {
                    "path": traj,
                    "untested": _candidates(nf, previous_frame=f_here),
                    "frame": nf,
                }
                frontier.append(nh)
        return _ret(None, best)

    # --- A*-style heuristic-guided best-first (COMPLETE + efficient) ---
    import heapq
    import itertools

    def _priority_value(frame, candidates) -> float:
        value = 0.0
        if priority_scorer is not None:
            try:
                if hasattr(priority_scorer, "frontier_priority"):
                    base = priority_scorer.frontier_priority(frame, candidates)
                else:
                    base = priority_scorer(frame)
                value += heuristic_weight * float(base)
            except Exception:
                value += 1e9  # a broken heuristic must never crash the search
        if action_effect_frontier_prior is not None:
            try:
                value += float(action_effect_frontier_prior.frontier_priority(frame, candidates))
            except Exception:
                pass
        return float(value)

    counter = itertools.count()
    # priority = g (depth) + h (weighted goal-distance); root popped first regardless
    heap = [
        (
            len(states[h0]["path"]) + _priority_value(f0, states[h0]["untested"]),
            next(counter),
            h0,
        )
    ]
    while heap and expansions < max_expansions:
        _, _, h = heapq.heappop(heap)
        st = states.get(h)
        if st is None or not st["untested"] or len(st["path"]) >= max_depth:
            continue
        # fully expand this (most-promising) state's untested actions (A* graph search:
        # each state expanded once, in priority order)
        while st["untested"]:
            f_here = replay(st["path"])  # navigate to this state
            qd_sequence = _next_qd_sequence(f_here, st)
            if qd_sequence:
                handled, result = _apply_qd_sequence(
                    st,
                    f_here,
                    qd_sequence,
                    policy="graph_explore_v2_heuristic_qd_sequence",
                )
                if isinstance(result, tuple) and result and result[0] != "new_state":
                    return result
                if isinstance(result, tuple) and result and result[0] == "new_state":
                    nh = result[1]
                    new_state = states[nh]
                    heapq.heappush(
                        heap,
                        (
                            len(new_state["path"])
                            + _priority_value(new_state["frame"], new_state["untested"]),
                            next(counter),
                            nh,
                        ),
                    )
                if handled and expansions >= max_expansions:
                    break
                continue
            sel = st["untested"].pop(0)
            label = _label(sel.action_id, sel.data)
            if _should_prune(f_here, label):
                continue
            nf = env.step(
                _game_action(GameAction, sel.action_id),
                data=sel.data,
                reasoning={"policy": "graph_explore_v2_heuristic_guided"},
            )
            expansions += 1
            if nf is not None:
                traj = st["path"] + [{"action": int(sel.action_id), "data": sel.data}]
                lvl = _levels_completed(nf)
                _observe(f_here, label, nf, lvl > start_level)
                if lvl > start_level and _predicate_allows_emit(nf):
                    _mark_goal_plan_emitted()
                    return _ret(traj, lvl)
                best = max(best, lvl)
                if not _game_over(nf):
                    nh = node_id(nf)
                    if nh not in states:  # new state ⇒ add with A* priority g+h
                        states[nh] = {
                            "path": traj,
                            "untested": _candidates(nf, previous_frame=f_here),
                            "frame": nf,
                        }
                        heapq.heappush(
                            heap,
                            (
                                len(traj)
                                + _priority_value(nf, states[nh]["untested"]),
                                next(counter),
                                nh,
                            ),
                        )
            if expansions >= max_expansions:
                break
    return _ret(None, best)


def cell_count_distance(win):
    """Baseline goal heuristic: `goal_distance(grid) -> float` = the number of cells differing
    from the win-state (`(grid != win).sum()`, Hamming distance). It is move-distance-accurate
    ONLY in LOW-cell-impact games (where one action changes few cells, so cell-count ≈ move
    count — e.g. su15, where it slightly beats region-count). In HIGH-cell-impact games it
    over-estimates move-distance and sends A* greedy → use `misplaced_region_distance` instead.
    The `arc_heuristic_select` router picks between the two by per-action cell impact."""
    import numpy as np

    win_arr = np.asarray(win)

    def goal_distance(grid) -> float:
        return float((np.asarray(grid) != win_arr).sum())

    return goal_distance


def misplaced_region_distance(win, connectivity: int = 8):
    """MOVE-DISTANCE-aware goal heuristic factory. Returns `goal_distance(grid) -> float` =
    the number of CONNECTED COMPONENTS in the `(grid != win)` mask — how many distinct
    "wrong regions" remain between `grid` and the win-state.

    WHY this beats a raw cell-count. A cell-count `(grid != win).sum()` over-estimates
    move-distance in games where one action changes MANY cells (one r11l click flips ~hundreds
    of cells): "1375 cells wrong" is a terrible proxy for "3 MOVES to win", so A* goes greedy
    and commits to a fast-but-SUBOPTIMAL path, and no `heuristic_weight` rescues it (proven by a
    weight sweep, 2026-06-17). The region count is instead MOVE-ALIGNED: each game action
    typically fixes one localized region, so the count drops ~1 per move. That gives BOTH the
    right SCALE (≈ moves) and a real GRADIENT, so A* (depth + h) finds the OPTIMAL path with far
    fewer expansions. 8-connectivity (diagonal) empirically beats 4-conn — it merges
    diagonally-touching wrong cells into one region, matching how an action groups its changes.

    Pass to `graph_explore_solve_v2(..., heuristic=lambda frame: gd(grid_of(frame)))`.

    Validated 2026-06-17 (v2-A*, budget 8000, vs pure BFS):
      r11l  OPTIMAL 3 actions @ 257 exp  (BFS 3 @ 2236  -> -88% expansions)
      m0r0  15-action solve   @ 6188 exp (BFS exhausts 8000 / no solve; 15 = registry-optimal)
      sk48  14 actions        @ 2496 exp (cell-count fails entirely; BFS 14 @ 4365 -> -43%)
      su15  7 actions         @ 1574 exp (helps; cell-count slightly better here at 1406)
    The 3 high-cell-impact games are exactly where cell-count could not win — this heuristic
    is the move-distance lever that unlocks them."""
    import numpy as np
    import scipy.ndimage as ndi

    win_arr = np.asarray(win)
    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else None

    def goal_distance(grid) -> float:
        return float(ndi.label(np.asarray(grid) != win_arr, structure=structure)[1])

    return goal_distance


def graph_explore_solve_v3(
    env: Any,
    start_level: int = 0,
    *,
    max_expansions: int = 30000,
    warmup: bool = False,
    max_depth: int = 80,
    verifier=None,
    stats: Optional[dict] = None,
) -> tuple[Optional[list], int]:
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
            return float(verifier(frame))  # lower predicted steps-to-go = better
        return float(region_visits[coarse(frame)] - 0.25 * depth)  # novelty, push deeper

    f0 = _warm(env, warmup)
    h0 = frame_hash(grid_of(f0))
    region_visits: dict = {coarse(f0): 1}
    states = {h0: {"path": [], "untested": rich_action_candidates(f0)}}
    counter = itertools.count()
    heap = [(priority(f0, 0), next(counter), h0)]
    best = start_level
    expansions = 0

    def _ret(traj, lvl):
        if stats is not None:
            stats["expansions"] = expansions
            stats["states"] = len(states)
        return traj, lvl

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
            nf = env.step(
                _game_action(GameAction, sel.action_id),
                data=sel.data,
                reasoning={"policy": "graph_explore_v3_value_guided"},
            )
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
            nh = frame_hash(grid_of(nf))
            if nh == here_hash or nh in states:
                continue  # no-op (wall bump) or seen ⇒ skip
            states[nh] = {"path": traj, "untested": rich_action_candidates(nf)}
            reg = coarse(nf)
            region_visits[reg] = region_visits.get(reg, 0) + 1
            heapq.heappush(heap, (priority(nf, len(traj)), next(counter), nh))
            if expansions >= max_expansions:
                break
    return _ret(None, best)


def trajectory_labels(traj: list) -> list[str]:
    """Encode a captured trajectory as replayable labels (for the reproduction gate
    / a trajectory-replay adapter)."""
    import json

    return [json.dumps(step) for step in traj]
