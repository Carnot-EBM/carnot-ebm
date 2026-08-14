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

import json
import os
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


def _components_detailed(grid, *, emit_grid_fallback: bool = False) -> list:
    """Connected non-background components with (centroid_y, centroid_x, area, color, is_grid_fallback).
    NOTE: the trailing ``is_grid_fallback`` field was added in the GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS
    fix (2f0760307); consumers MUST unpack defensively (index or ``*_``), never a rigid 4-tuple assignment --
    a rigid unpack crashed plan_in_model's ``_model_candidates`` for months of grids with components.
    Same 4-neighbour flood fill as world_model.objects(), but also returns area+color
    so candidates can be ordered by VISUAL SALIENCE (segment size × color rarity) —
    the key ingredient from the graph-explore SOTA (arXiv:2512.24156) that lets the
    search try the most salient interactive elements first instead of treating all
    objects uniformly.

    ``emit_grid_fallback`` (REQ-ARC-FCP-5757 follow-up, GAP-ARC-BP35-CLICK-CANDIDATE-
    GENERATION-MISS, 2026-07-23; OFF by default): threads through to
    ``object_centric_digest``'s ``emit_grid_fallback_for_background`` -- see that
    function's docstring for the full diagnosis and rationale."""
    from carnot.agentic.arc_solver_kit import object_centric_digest

    comps = []
    for comp in object_centric_digest(grid, emit_grid_fallback_for_background=emit_grid_fallback)[
        "components"
    ]:
        cx, cy = comp["centroid"]
        comps.append(
            (
                int(cy),
                int(cx),
                int(comp["area"]),
                int(comp["color"]),
                bool(comp.get("is_grid_fallback")),
            )
        )
    return comps


# just-explore's 5-tier salience schedule (heuristic_agent.py:frame_segments_to_action_groups,
# arXiv:2512.24156). The 2026-06-23 offline head-to-head showed this SCHEDULE reaches first-wins our
# flat area*rarity sort misses on 5 games (bp35/ft09/m0r0/r11l/vc33). It front-loads BUTTON-LIKE objects
# (salient colour AND medium bounding-box) and defers very-large / dull / status-bar segments, where our
# flat sort up-ranks the largest area first. Constants are just-explore's verbatim.
# REQ-ARC-WMTE-5950: the fallback stream for click-pixel sampling when a caller enables the
# feature without supplying its own RNG (seeded, so a run is still reproducible; see the call
# site in rich_action_candidates for why it must be one shared instance and not a fresh one).
_CLICK_PIXEL_FALLBACK_RNG = random.Random(20260725)

_TIER_SALIENT_COLORS = frozenset(range(6, 16))  # {6..15} (non-salient = {0..5})
_TIER_STATUS_BAR_COLOR = 16
_TIER_MIN_WIDTH = 2
_TIER_MAX_WIDTH = 32


def _action_prior_click_points(
    action_prior: Any | None,
    frame: Any,
    *,
    max_click: int,
) -> list[tuple[int, int]] | None:
    """REQ-ARC-FCP-5397: let live action priors shape click generation before caps."""

    prior = action_prior
    if prior is not None and not hasattr(prior, "click_points"):
        prior = getattr(prior, "base_prior", None)
    if prior is None or not hasattr(prior, "click_points"):
        return None
    try:
        points = prior.click_points(frame, max_points=max_click)
    except Exception:
        return None
    return [(int(x), int(y)) for x, y in points]


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


def _small_object_first_click_points(grid, *, small_area_max: int = 8) -> list[tuple[int, int]]:
    """REQ-ARC-FCP-5758: surface TINY interactive targets ahead of large decoration.

    Diagnosis (exp5757 attribution + exp5758 per-object dump): on the stalled games
    the winning object-clicks are consistently very SMALL objects -- r11l's low-ranked
    winner is a single pixel (area=1), su15's repeatedly-clicked winner is a single
    pixel (area=1, color=3), r11l's other winners are area 4 and 12. The shipped default
    orders object clicks by VISUAL SALIENCE = ``area * (1 + 1/(1 + global_color_pixels))``
    which is AREA-DOMINATED: a 240-pixel decorative region (salience ~240) always outranks
    a 1-pixel interactive target (salience ~1) regardless of colour rarity, so the winning
    clicks sink to ranks 13-22 of ~27-34 candidates and rarely get tried within budget.

    This orders object clicks in TWO bands so the tiny targets are NOT buried:
      1. a SMALL band (object area <= ``small_area_max``) ordered by colour-rarity
         (rarest colour first; larger-within-small breaks ties), tried FIRST;
      2. every remaining object in the PROVEN salience order (large decorations last).

    Pure reordering of the SAME candidate set (no new clicks, no dropped clicks) -- the
    trajectory the explorer ultimately records is still a valid deterministic replay; it
    just reaches the small-target clicks within a smaller budget. This is genuinely
    DIFFERENT from ``CARNOT_ARC_TIER_SCHEDULE`` (that front-loads MEDIUM-width w,h in
    [2,32] objects, which EXCLUDES the 1x1 winners here -- and it was already A/B-NULL,
    results/proto_tier_ab.json) and from the area*rarity default (which buries them)."""
    from collections import Counter

    from carnot.agentic.arc_solver_kit import object_centric_digest

    color_cells = Counter(int(v) for v in grid.flatten().tolist())
    small: list[tuple[float, int, int, int]] = []  # (rarity_key, -area, x, y)
    rest: list[tuple[float, int, int]] = []  # (-salience, x, y)
    for comp in object_centric_digest(grid)["components"]:
        cx, cy = comp["centroid"]
        x, y = int(cx), int(cy)
        area = int(comp["area"])
        color = int(comp["color"])
        gpc = color_cells.get(color, 0)
        rarity = 1.0 / (1.0 + gpc)  # larger = rarer colour
        if area <= small_area_max:
            small.append((-rarity, -area, x, y))  # rarest first, larger-within-small first
        else:
            salience = area * (1.0 + rarity)
            rest.append((-salience, x, y))
    small.sort()
    rest.sort()
    return [(x, y) for _r, _a, x, y in small] + [(x, y) for _s, x, y in rest]


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
    click_pixel_sampling: bool = False,
    click_pixel_samples_per_component: int = 1,
    click_pixel_rng: Any | None = None,
    click_pixel_diagnostics_out: dict | None = None,
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
    keeps the bare order so the live solver has a no-regression fallback.

    REQ-ARC-WMTE-5950 (``click_pixel_sampling``, default OFF -> byte-identical to the
    proven behaviour): replace each object's single truncated CENTROID click coordinate
    with a uniform random pixel OF THAT OBJECT (the just-explore generation rule; see
    ``carnot.agentic.arc_component_sampling`` for the full rationale and the measured
    defects it addresses -- most importantly that the truncated centroid is not a member
    of its own object on 100% of 204 measured real r11l states). Applied LAST among the
    point producers, so it composes with whichever ordering lever is active and varies
    ONLY the coordinate. ``click_pixel_samples_per_component`` > 1 emits several pixels
    per object; the object list is bounded by ``max_click`` BEFORE expansion so that
    raising k cannot silently shrink the number of reachable objects (which would turn a
    coordinate experiment into a budget experiment).

    ``click_pixel_diagnostics_out``: an OUT-parameter dict the sampler's per-call
    ``SamplingDiagnostics`` is written into (``coordinates_changed``, ``unresolved``,
    ``errors``, ...). It is an out-parameter rather than a second return value because this
    function's list return type is consumed in dozens of places, and the alternative --
    discarding the diagnostics, as the first implementation did -- made a TOTALLY DEAD
    sampler indistinguishable from a working one in the emitted artifact (verified by
    patching ``component_partition`` to raise: the arm still reported rows_sampled=1,
    errors=0 while emitting the unmodified centroid). ``coordinates_changed == 0`` with
    ``errors > 0`` is now the mechanical signature of a dead sampler."""
    ids = _available_action_ids(frame)
    out = [ArcAction(a, None, "available_keyboard_action") for a in ids if a != 6]
    if 6 in ids:
        import os

        pts_already_bounded = False

        grid = grid_of(frame)
        pts = _action_prior_click_points(action_prior, frame, max_click=max_click)
        # CARNOT_ARC_TIER_SCHEDULE=1 orders object-clicks by just-explore's 5 salience tiers (button-like
        # medium-width salient objects first) instead of the flat area*rarity sort. Default off -> the
        # path below is byte-identical to the proven order (parity preserved; the SUBMITTED agent unchanged
        # until the A/B greenlights it). A scoring failure falls back to the flat order (no-regression).
        if pts is None and os.environ.get("CARNOT_ARC_TIER_SCHEDULE") == "1":
            try:
                pts = (
                    _tier_ordered_click_points(grid) or None
                )  # None -> fall back to the flat order
            except Exception:
                pts = None
        # CARNOT_ARC_SMALL_OBJECT_FIRST=1 (REQ-ARC-FCP-5758) surfaces TINY interactive
        # targets (single-pixel / area<=8 objects) ahead of the large decorative regions
        # the area-dominated default salience buries them under -- the exp5757-diagnosed
        # click-ranking gap on r11l/su15. Default off -> byte-identical to the proven order
        # (the SUBMITTED agent is unchanged until an A/B greenlights it). A failure falls
        # back to the flat order (no-regression).
        if pts is None and os.environ.get("CARNOT_ARC_SMALL_OBJECT_FIRST") == "1":
            try:
                pts = _small_object_first_click_points(grid) or None
            except Exception:
                pts = None
        if pts is None:
            # CARNOT_ARC_GRID_FALLBACK_CANDIDATES=1 (REQ-ARC-FCP-5757 follow-up,
            # GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS, 2026-07-23) tiles the excluded
            # background color into coarse click candidates so individually-meaningful cells
            # that share the most-common color (e.g. bp35's same-row blocker tiles) are not
            # PERMANENTLY invisible to every downstream mechanism regardless of search depth.
            # Default off -> byte-identical to the proven order (parity preserved; the
            # SUBMITTED agent unchanged until an A/B greenlights it). The kwarg is only ever
            # PASSED when the flag is actually on, so a test double patched against the
            # pre-5757-followup 1-arg _components_detailed(grid) signature is unaffected.
            grid_fallback_on = os.environ.get("CARNOT_ARC_GRID_FALLBACK_CANDIDATES") == "1"
            if grid_fallback_on:
                comps = _components_detailed(grid, emit_grid_fallback=True)
            else:
                comps = _components_detailed(grid)

            # Unpacked defensively (not `cy, cx, area, color, is_fb = c`): several existing tests
            # (e.g. test_req_arc_fcp_4491/4511/4512/4547/4568) monkeypatch _components_detailed
            # with a fake returning the pre-5757-followup 4-tuple `(cy, cx, area, color)` shape,
            # to test OTHER rich_action_candidates behavior (rankers, priors) unrelated to grid
            # fallback -- a hard `len(c) == 5` assumption here would break every one of them.
            def _is_fallback(c) -> bool:
                return bool(c[4]) if len(c) > 4 else False

            # Grid-fallback tiles are, BY CONSTRUCTION, small pieces of the single most-common
            # color -- the exact combination (small area * near-zero color-rarity) the salience
            # sort below ranks lowest. Left in the same pool, they are reliably pushed past the
            # `max_click` cap by genuine higher-salience objects and NEVER actually surface as
            # candidates (measured directly: 55 fallback tiles generated, 0 survived a top-48
            # cut on a real bp35 state -- the bug this split fixes). Splitting them into their
            # OWN small reserved budget guarantees the fallback mechanism can't be silently
            # starved by unrelated objects it structurally can never outrank.
            real_comps = [c for c in comps if not _is_fallback(c)] if grid_fallback_on else comps
            fallback_comps = [c for c in comps if _is_fallback(c)] if grid_fallback_on else []
            if by_salience and real_comps:
                from collections import Counter

                color_cells = Counter(int(v) for v in grid.flatten().tolist())
                # salience: big segments + globally-rare colors score highest
                real_comps.sort(
                    key=lambda c: c[2] * (1.0 + 1.0 / (1 + color_cells.get(c[3], 0))), reverse=True
                )
            pts = [(int(c[1]), int(c[0])) for c in real_comps]
            if fallback_comps:
                # SMALLEST tile first, not largest: measured directly on bp35 that the tile
                # actually covering a real winning click (area=9, a small interior sliver
                # against other objects) sat near the BOTTOM of an area-descending sort behind
                # ~50 large, genuinely-empty corner/edge tiles of the suppressed background --
                # mirrors REQ-ARC-FCP-5758's own established finding elsewhere in this file
                # ("winning object-clicks are consistently very SMALL objects").
                fallback_comps.sort(key=lambda c: c[2])  # smaller tile first
                # No further sub-cap here: object_centric_digest's own grid_fallback_max_tiles
                # (default 64) is already the size guard. A first attempt additionally capped
                # this at max_click//2 and measured a REAL, but smaller, improvement on bp35's
                # actual reproduction-gated winning trajectory (21/57 generation misses -> 16/57)
                # -- the covering tile for several targets simply didn't make that tighter cut.
                # Removing the redundant sub-cap (keeping ALL emitted fallback tiles, bounded
                # only by the digest's own 64-tile ceiling) measured a LARGER improvement on the
                # SAME replay (21/57 -> 15/57 at the real default max_click=48; a synthetic,
                # artificially-widened max_click=200 run -- NOT the real live budget -- closed it
                # completely, confirming the underlying mechanism is sound and the remaining
                # misses at the real budget are a genuine candidate-budget-sharing tradeoff
                # against real, higher-salience objects, not a bug in the fallback tiling itself).
                pts = pts[:max_click] + [(int(c[1]), int(c[0])) for c in fallback_comps]
                pts_already_bounded = True  # don't let the loop's own [:max_click] slice
                # below cut the appended fallback tail back off -- that would silently undo
                # the whole point of the reserved fallback budget above.
        if not pts:
            h, w = grid.shape
            pts = [(w // 2, h // 2)]
        if click_pixel_sampling:
            # REQ-ARC-WMTE-5950. Bound the OBJECT list first, then expand each surviving
            # object into k sampled pixels, then tell the loop below not to re-cut. Doing
            # it in this order is the whole point: expanding first and capping after would
            # divide the reachable object count by k, so a k>1 arm would be measuring a
            # smaller candidate budget rather than a different coordinate rule.
            from carnot.agentic.arc_component_sampling import sample_component_click_points

            bounded = pts if pts_already_bounded else pts[:max_click]
            # The fallback RNG is a MODULE-LEVEL instance, not a fresh Random(0) per call.
            # A fresh instance would restart the same stream on every invocation, so a caller
            # that enables the flag via the env override (and therefore passes no rng) would
            # get the SAME "random" pixel for a given object on every frame -- silently
            # turning a with-replacement sampler back into a fixed-point one. The live path
            # always passes the explorer's own seeded stream; this only guards direct callers.
            sampled, cps_diag = sample_component_click_points(
                grid,
                bounded,
                rng=click_pixel_rng if click_pixel_rng is not None else _CLICK_PIXEL_FALLBACK_RNG,
                samples_per_component=max(1, int(click_pixel_samples_per_component)),
            )
            # The diagnostics are the mechanism's ONLY activity witness -- discarding them
            # (as the first implementation did with `_diag`) is what let a dead sampler read
            # as an active one. Written to the caller's dict when one was supplied.
            if click_pixel_diagnostics_out is not None:
                try:
                    click_pixel_diagnostics_out.update(cps_diag.as_dict())
                except Exception:
                    pass
            pts = sampled
            pts_already_bounded = True
        seen: set = set()
        for x, y in pts if pts_already_bounded else pts[:max_click]:
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
    frontier_seed_bank: Any | None = None,
    candidate_router=None,
    structural_energy_scorer=None,
    move_pruner=None,
    state_key_action_suffix_k: Optional[int] = None,
    collision_certified_state_key_suffix: bool | None = None,
    collision_certified_state_key_suffix_max_k: int = 4,
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

    `frontier_seed_bank` is the REQ-REPORT-5198 MAP hook: a bounded pre-search
    map can provide replayable landmark/affordance trajectories that are tried
    before flat primitive expansion at matching frontier nodes. Seeded actions
    still consume the same expansion budget and return the same replayable
    trajectory format, so `arc_solver_kit.reproduce()` remains the solve gate.
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

    # --- Non-Markov observation aliasing fix (DEFAULT OFF; REQ-ARC-GE-6110) ---
    #
    # WHY: on some games the visible grid does not expose all of the game's state, so two
    # BEHAVIOURALLY DISTINCT states hash to the same node key. The measured worst case is
    # sc25 (exp6094): every one of the root's candidate actions is visually inert on its
    # FIRST application (the game consumes it — hidden "started" state advances, the frame
    # does not), so every successor aliases into the root node, the root's untested list
    # drains, and the whole search terminates at 24 expansions having "discovered" 1 state —
    # identically at a 6000 and a 30000 budget. The frontier collapse is a REPRESENTATION
    # limit, not a search wall: no budget helps.
    #
    # THE FIX: append the last k actions of the arriving path to the node key — the classic
    # k-th-order-Markov remedy for a non-Markov observation. Same frame reached under a
    # different recent-action suffix = a different node, so a visually-inert-but-state-
    # advancing action creates a NEW frontier node instead of aliasing into its parent.
    # Purely frame-and-own-action derived: no game ids, no per-game constants, nothing the
    # live agent does not see about its own behaviour.
    #
    # THE COST (why it defaults OFF): k>0 inflates the state space — the same true state
    # reached under different suffixes becomes several nodes, each re-expanded. That spends
    # budget, so a game that was fine at k=0 can regress. Enable via the env flag
    # CARNOT_ARC_STATE_KEY_SUFFIX_K=<k> or the explicit parameter; k=0 (the default) is
    # byte-for-byte the original single-hash identity.
    if state_key_action_suffix_k is None:
        try:
            state_key_action_suffix_k = int(
                os.environ.get("CARNOT_ARC_STATE_KEY_SUFFIX_K", "0") or "0"
            )
        except ValueError:
            state_key_action_suffix_k = 0
    suffix_k = max(0, int(state_key_action_suffix_k))

    if collision_certified_state_key_suffix is None:
        collision_certified_state_key_suffix = (
            os.environ.get("CARNOT_ARC_COLLISION_CERTIFIED_STATE_KEY_SUFFIX") == "1"
        )
    certified_suffix_enabled = bool(collision_certified_state_key_suffix) and suffix_k == 0
    from carnot.agentic.arc_state_key_certifier import StateKeyCollisionCertifier

    collision_certifier = StateKeyCollisionCertifier(
        enabled=certified_suffix_enabled,
        max_suffix_k=collision_certified_state_key_suffix_max_k,
    )

    def _action_suffix(path) -> str:
        if not suffix_k:
            return ""
        parts = []
        for step in (path or [])[-suffix_k:]:
            data = step.get("data")
            # json with sorted keys = a stable, hashable canonical form for click coords etc.
            parts.append(
                f"{int(step['action'])}"
                + (f"@{json.dumps(data, sort_keys=True, default=str)}" if data else "")
            )
        return "|k:" + ";".join(parts)

    def _node_key(frame, path, observation_history) -> str:
        # k=0 returns node_id(frame) EXACTLY (empty suffix) — the shipped default is
        # unchanged, which is what the both-directions test asserts.
        base_key = node_id(frame)
        if suffix_k:
            return base_key + _action_suffix(path)
        return collision_certifier.state_key(base_key, observation_history, path)

    def _base_from_state_key(key: str) -> str:
        for marker in ("|k:", "|certk:"):
            if marker in key:
                return key.split(marker, 1)[0]
        return key

    def _history_action_label(step) -> str:
        data = step.get("data")
        if data is None:
            return str(int(step["action"]))
        return f"{int(step['action'])}@{json.dumps(data, sort_keys=True, default=str)}"

    def _obs_history_root(frame, path) -> tuple[str, ...]:
        tokens: list[str] = [f"obs:{node_id(frame)}"]
        if path:
            tokens.insert(0, f"prefix_len:{len(path)}")
            tokens.extend(f"prefix_act:{_history_action_label(step)}" for step in path)
        return tuple(tokens)

    def _obs_history_next(parent_history, label, frame_after) -> tuple[str, ...]:
        return tuple(parent_history or ()) + (
            f"act:{_history_action_label(label)}",
            f"obs:{node_id(frame_after)}",
        )

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
    obs0 = _obs_history_root(f0, prefix)
    h0 = _node_key(f0, prefix, obs0)
    states = {
        h0: {"path": list(prefix), "untested": _candidates(f0), "frame": f0, "obs_history": obs0}
    }
    best = start_level
    expansions = 0
    qd_search_generator = coerce_qd_generator(
        qd_generator,
        action_effect_scorer=frame_change_scorer,
        goal_energy=goal_energy,
    )
    qd_sequences_injected = 0
    qd_actions_injected = 0
    frontier_seed_sequences_injected = 0
    frontier_seed_actions_injected = 0
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
            stats["state_key_action_suffix_k"] = int(suffix_k)
            # How many distinct VISIBLE frames the graph holds, independent of suffix
            # splits. states == distinct_frames when k=0; the gap between them is the
            # state-space inflation the suffix key paid for its de-aliasing.
            stats["distinct_frames"] = (
                len({_base_from_state_key(k) for k in states})
                if (suffix_k or certified_suffix_enabled)
                else len(states)
            )
            cert_diag = collision_certifier.diagnostics()
            cert_rows = collision_certifier.certificate_rows()
            stats["state_key_collision_certified_suffix_enabled"] = bool(certified_suffix_enabled)
            stats["state_key_collision_certificate_count"] = int(len(cert_rows))
            stats["state_key_collision_certificates"] = cert_rows
            stats["state_key_collision_diagnostics"] = cert_diag
            stats["state_key_effective_suffix_max_k"] = max(
                int(suffix_k), int(cert_diag.get("max_suffix_k_used") or 0)
            )
            stats["state_key_collision_hash_substitution_detected"] = bool(
                cert_diag.get("hash_substitution_detected")
            )
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
            stats["frontier_seed_enabled"] = frontier_seed_bank is not None
            stats["frontier_seed_sequences_injected"] = int(frontier_seed_sequences_injected)
            stats["frontier_seed_actions_injected"] = int(frontier_seed_actions_injected)
            if frontier_seed_bank is not None and hasattr(frontier_seed_bank, "diagnostics"):
                try:
                    stats["frontier_seed_diagnostics"] = frontier_seed_bank.diagnostics()
                except Exception:
                    stats["frontier_seed_diagnostics"] = None
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

    def _frontier_seed_sequences(frame, node: dict) -> list[list[dict]]:
        if frontier_seed_bank is None:
            return []
        candidates = list(node.get("untested") or [])
        path = list(node.get("path") or [])
        try:
            if hasattr(frontier_seed_bank, "frontier_seed_sequences"):
                raw = frontier_seed_bank.frontier_seed_sequences(
                    frame,
                    candidates,
                    path=path,
                    root_path_length=len(prefix),
                    goal_energy=goal_energy,
                )
            elif hasattr(frontier_seed_bank, "best_sequence"):
                raw = [frontier_seed_bank.best_sequence(frame, candidates, goal_energy=goal_energy)]
            elif callable(frontier_seed_bank):
                raw = frontier_seed_bank(frame, candidates)
            else:
                raw = []
        except TypeError:
            try:
                raw = frontier_seed_bank.frontier_seed_sequences(frame, candidates)
            except Exception:
                raw = []
        except Exception:
            raw = []
        sequences: list[list[dict]] = []
        for seq in raw or []:
            rows = []
            for step in seq or []:
                if not isinstance(step, dict) or step.get("action") is None:
                    continue
                rows.append({"action": int(step["action"]), "data": step.get("data")})
            if rows:
                sequences.append(rows)
        return sequences

    def _next_frontier_seed_sequence(frame, node: dict) -> list[dict]:
        nonlocal frontier_seed_sequences_injected, frontier_seed_actions_injected
        if frontier_seed_bank is None or node.get("frontier_seed_exhausted"):
            return []
        cursor = int(node.get("frontier_seed_cursor", 0) or 0)
        sequences = _frontier_seed_sequences(frame, node)
        if cursor >= len(sequences):
            node["frontier_seed_exhausted"] = True
            return []
        rows = sequences[cursor]
        node["frontier_seed_cursor"] = cursor + 1
        frontier_seed_sequences_injected += 1
        frontier_seed_actions_injected += len(rows)
        return rows

    def _apply_frontier_seed_sequence(
        state: dict, frame_here, sequence: list[dict], *, policy: str
    ):
        nonlocal expansions, best
        traj = list(state["path"])
        nf = frame_here
        obs_history = tuple(state.get("obs_history") or ())
        for step in sequence:
            label = _label(step["action"], step.get("data"))
            if _should_prune(nf, label):
                return True, None
            before = nf
            nf = env.step(
                _game_action(GameAction, int(step["action"])),
                data=step.get("data"),
                reasoning={"policy": policy, "generator": "map_landmark_prestage"},
            )
            expansions += 1
            if nf is None:
                return True, None
            traj = traj + [{"action": int(step["action"]), "data": step.get("data")}]
            obs_history = _obs_history_next(obs_history, label, nf)
            lvl = _levels_completed(nf)
            _observe(before, label, nf, lvl > start_level)
            if lvl > start_level and _predicate_allows_emit(nf):
                _mark_goal_plan_emitted()
                return True, _ret(traj, lvl)
            best = max(best, lvl)
            if _game_over(nf) or expansions >= max_expansions:
                return True, None
        if nf is not None and not _game_over(nf):
            nh = _node_key(nf, traj, obs_history)
            if nh not in states:
                states[nh] = {
                    "path": traj,
                    "untested": _candidates(nf, previous_frame=frame_here),
                    "frame": nf,
                    "obs_history": obs_history,
                }
                return True, ("new_state", nh)
        return True, None

    def _apply_qd_sequence(state: dict, frame_here, sequence: list[dict], *, policy: str):
        nonlocal expansions, best
        traj = list(state["path"])
        nf = frame_here
        obs_history = tuple(state.get("obs_history") or ())
        for step in sequence:
            label = _label(step["action"], step.get("data"))
            nf = env.step(
                _game_action(GameAction, int(step["action"])),
                data=step.get("data"),
                reasoning={"policy": policy, "generator": "energy_fitness_qd"},
            )
            expansions += 1
            if nf is None:
                return True, None
            traj = traj + [{"action": int(step["action"]), "data": step.get("data")}]
            obs_history = _obs_history_next(obs_history, label, nf)
            lvl = _levels_completed(nf)
            if lvl > start_level and _predicate_allows_emit(nf):
                _mark_goal_plan_emitted()
                return True, _ret(traj, lvl)
            best = max(best, lvl)
            if _game_over(nf) or expansions >= max_expansions:
                return True, None
        if nf is not None and not _game_over(nf):
            nh = _node_key(nf, traj, obs_history)
            if nh not in states:
                states[nh] = {
                    "path": traj,
                    "untested": _candidates(nf, previous_frame=frame_here),
                    "frame": nf,
                    "obs_history": obs_history,
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
            frontier_seed_sequence = _next_frontier_seed_sequence(f_here, st)
            if frontier_seed_sequence:
                handled, result = _apply_frontier_seed_sequence(
                    st,
                    f_here,
                    frontier_seed_sequence,
                    policy="graph_explore_v2_map_frontier_seed",
                )
                if isinstance(result, tuple) and result and result[0] != "new_state":
                    return result
                if isinstance(result, tuple) and result and result[0] == "new_state":
                    frontier.append(result[1])
                if handled and expansions >= max_expansions:
                    break
                continue
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
            obs_history = _obs_history_next(st.get("obs_history"), label, nf)
            lvl = _levels_completed(nf)
            _observe(f_here, label, nf, lvl > start_level)
            if lvl > start_level and _predicate_allows_emit(nf):
                _mark_goal_plan_emitted()
                return _ret(traj, lvl)
            best = max(best, lvl)
            if _game_over(nf):
                continue
            nh = _node_key(nf, traj, obs_history)
            if nh not in states:  # new state ⇒ add to graph + frontier
                states[nh] = {
                    "path": traj,
                    "untested": _candidates(nf, previous_frame=f_here),
                    "frame": nf,
                    "obs_history": obs_history,
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
            frontier_seed_sequence = _next_frontier_seed_sequence(f_here, st)
            if frontier_seed_sequence:
                handled, result = _apply_frontier_seed_sequence(
                    st,
                    f_here,
                    frontier_seed_sequence,
                    policy="graph_explore_v2_heuristic_map_frontier_seed",
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
                obs_history = _obs_history_next(st.get("obs_history"), label, nf)
                lvl = _levels_completed(nf)
                _observe(f_here, label, nf, lvl > start_level)
                if lvl > start_level and _predicate_allows_emit(nf):
                    _mark_goal_plan_emitted()
                    return _ret(traj, lvl)
                best = max(best, lvl)
                if not _game_over(nf):
                    nh = _node_key(nf, traj, obs_history)
                    if nh not in states:  # new state ⇒ add with A* priority g+h
                        states[nh] = {
                            "path": traj,
                            "untested": _candidates(nf, previous_frame=f_here),
                            "frame": nf,
                            "obs_history": obs_history,
                        }
                        heapq.heappush(
                            heap,
                            (
                                len(traj) + _priority_value(nf, states[nh]["untested"]),
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
