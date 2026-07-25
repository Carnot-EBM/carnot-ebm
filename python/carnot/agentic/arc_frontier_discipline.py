"""Frontier DISCIPLINE for the live ARC-AGI-3 explorer -- the two search-ORDER mechanisms
grafted from the no-LLM "just-explore" solver (arXiv:2512.24156, 3rd place on the ARC-AGI-3
Preview private leaderboard, which solved a median of ~16 private levels with NO language
model and NO learned scorer at all).

=====================================================================================
WHY THIS MODULE EXISTS (read this before changing anything here)
=====================================================================================

Carnot's live click decision is a single line -- ``lst.pop(0)`` inside
``StepwiseExplorer._pop_untested`` -- over a candidate list ordered by a STATIC
"area x colour-rarity" salience sort. The learned candidate router that is nominally
supposed to reorder that list is COORDINATE-BLIND: it featurises only the integer action
TYPE, so all N click targets on a frame receive one identical score and the router's
``rank()`` is a stable no-op. The practical consequence, verified across four recent
experiments, is that improving the QUALITY of information the agent has (a bigger
inducer, a goal-predicate veto, richer frame features) could not move the decision those
experiments targeted, because the decision was never actually reorderable.

So the lever is not "more model" -- it is the SEARCH ORDER itself. just-explore's whole
advantage is a *discipline* over the order in which an exhaustive explorer spends its
actions, and that discipline has two halves. Carnot already ports one small piece (the
5-tier salience predicate, behind ``CARNOT_ARC_TIER_SCHEDULE``) and that piece measured
a clean NULL -- because Carnot applied it as a flat SORT KEY, which is emphatically NOT
what the reference does with it. The reference uses the tiers as an EXHAUSTION BARRIER.
The barrier, and the distance gradient below, are what this module supplies.

-------------------------------------------------------------------------------------
MECHANISM (a) -- STRICT GLOBAL PRIORITY-TIER EXHAUSTION
-------------------------------------------------------------------------------------
Reference: ``graph_explorer.py`` ``NodeInfo.has_open_group`` / ``GraphExplorer._active_group``
/ ``GraphExplorer._maybe_advance_group``; stated in the paper as "Algorithm 1: Hierarchical
Action Selection".

A *sort* says "try tier-0 things before tier-3 things, at this node". A *barrier* says
"do not touch a tier-3 action ANYWHERE in the entire graph until every tier-0/1/2 action
EVERYWHERE has been tried". Those are wildly different search orders once the graph has
more than one node -- and the second one is the reference's. Four properties are
load-bearing and all four are reproduced here:

  1. ONE GLOBAL priority level ``p`` for the whole explorer (not one per node).
  2. Eligibility is CUMULATIVE: an action is selectable iff ``tier(a) <= p``. Never
     ``tier(a) == p`` -- lower tiers stay selectable forever once admitted.
  3. Advancement is not "this node ran out"; it is "NOTHING anywhere is still open at
     ``p``". The reference expresses this as unreachability (its distance to any open
     node is INFINITY). See ``next_active_tier`` for why Carnot's faithful analogue is
     global set-exhaustion instead.
  4. Advancement can skip MULTIPLE tiers in one step (it is a ``while``, not an ``if``),
     because tier ``p+1`` may itself be empty everywhere.

WHY a barrier would beat a sort: the sort lets the depth-first ride descend a branch
whose tier-0 options are gone, committing budget to tier-3 junk deep in one subtree while
untried BUTTON-LIKE (tier-0) objects sit unexplored in sibling subtrees. The barrier
forces the cheap, high-yield tier to be swept globally first. That is a hypothesis, not a
result -- which is exactly why this ships default-OFF behind an A/B.

-------------------------------------------------------------------------------------
MECHANISM (b) -- MULTI-SOURCE FRONTIER-DISTANCE GRADIENT
-------------------------------------------------------------------------------------
Reference: ``graph_explorer.py`` ``GraphExplorer._rebuild_distances``, consumed by
``choose_edge``'s else-branch.

The reference runs a MULTI-SOURCE REVERSE breadth-first search seeded simultaneously from
EVERY node that still has open work (all at distance 0), walked BACKWARD over only
CONFIRMED-WORKING edges. Every node thereby learns (i) how many hops it is from the
nearest node with something left to try, and (ii) the first action along that route. The
agent then simply walks downhill.

Carnot instead selects its frontier target by DEPTH FROM ROOT (shallowest-first, i.e. a
position-independent global BFS order) and only afterwards computes how to get there --
falling back to RESET + replaying the whole path from the root when no forward walk
exists. Depth-from-root and distance-from-here are different objectives: the shallowest
open node can be arbitrarily far from where the agent is standing, and every such choice
pays a full replay. Ordering by the gradient instead means the agent converges on the
NEAREST open node, which is both a different search order AND strictly cheaper in real
environment interactions (the metric the live scorer squares).

Note carefully that Carnot ALREADY has the walking half (exact forward path -> partial
forward path -> reset+replay). This module deliberately does NOT re-implement navigation.
It supplies only the TARGET-SELECTION criterion that was missing.

-------------------------------------------------------------------------------------
WHAT THIS MODULE DELIBERATELY DOES *NOT* DO
-------------------------------------------------------------------------------------
It does not re-port the 5-tier salience predicate. That already exists verbatim at
``arc_graph_explore._tier_ordered_click_points`` and has already been A/B'd to a null as
a sort. This module REUSES that predicate's constants and re-derives the same tiers only
so it can gate on them; re-deriving a second, subtly-different tier predicate would make
the A/B uninterpretable.

It also does not replace within-tier ordering with any Carnot score. Three prior
experiments swapped the reference's within-tier UNIFORM-RANDOM draw for a Carnot-scored
argmax / epsilon-greedy / softmax / percentile-defer, and every single arm LOST solves.
Carnot's live ``pop(0)`` is the fully-greedy end of that same spectrum, so it is entirely
possible that the greedy draw is itself part of the defect and that the barrier only pays
off when paired with a uniform-random draw. ``select_index`` therefore supports BOTH
draws (deterministic-first and uniform-random-among-eligible) so the A/B can separate
"the barrier helped" from "merely breaking the static tie helped".

The uniform draw is faithful ONLY at ``top_k=None``: the reference's ``choose_edge`` does
``random.choice(untested_edges)`` over every untested edge in groups ``0..active_group``,
with no top-k restriction whatsoever. ``top_k`` exists here so a top-n restriction can be
measured as its own explicitly-declared arm; passing a ``top_k`` is a DEVIATION from the
reference and any experiment that does so must record it in ``spec_deviations``. (The wiring
originally defaulted this to the unrelated hybrid-diversity knob's value of 8, which made
"the reference's uniform draw" arm silently a top-8 draw coupled to a foreign env var --
caught by adversarial review 2026-07-24.)

-------------------------------------------------------------------------------------
DESIGN CONSTRAINT: EVERYTHING HERE IS PURE
-------------------------------------------------------------------------------------
Every function takes explicit state in and returns an ordering/decision out. Nothing here
touches an environment, a frame stream, an LLM, or the explorer's mutable internals. That
is what makes the policies unit-testable without running a game, and it is what lets the
wiring in ``arc_competition_agent.StepwiseExplorer`` stay a thin, auditable adapter.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "TIER_COUNT",
    "DEFAULT_TIER",
    "STATUS_BAR_TIER",
    "click_tier_map",
    "tier_map_for_frame",
    "row_tier",
    "annotate_tiers",
    "TierExhaustionPolicy",
    "FrontierDistanceField",
    "reverse_adjacency",
    "frontier_distance_field",
    "nearest_open_node",
    "gradient_frontier_target",
]


# just-explore's schedule has exactly five priority groups; tier 0 is the most eligible.
TIER_COUNT = 5
# Anything we cannot classify (keyboard/simple actions, click coordinates that do not land
# on a detected object centroid, rows created before the flag was flipped) is treated as
# tier 0 -- i.e. ALWAYS eligible. This fails OPEN on purpose: an unclassified action being
# selectable too early only costs ordering, whereas an unclassified action being deferred
# forever would silently make part of the action space unreachable. The reference makes the
# same choice for its keyboard actions, which it forces into group 0 explicitly.
DEFAULT_TIER = 0
# The reference reserves its LAST group for status-bar segments (HUD chrome, the least
# likely thing to be interactive).
STATUS_BAR_TIER = TIER_COUNT - 1


def click_tier_map(grid: Any, *, include_cells: bool = False) -> dict[tuple[int, int], int]:
    """Map each detected object's click point ``(x, y)`` to its just-explore priority tier.

    ``include_cells`` (REQ-ARC-WMTE-5950, default False -> byte-identical to the proven
    map) additionally keys EVERY cell of every object, not just its truncated centroid.

    WHY THIS IS A CORRECTNESS CO-CHANGE, NOT AN OPTIMIZATION. This map is centroid-keyed,
    and ``row_tier`` falls back to ``DEFAULT_TIER`` = 0 = ALWAYS ELIGIBLE on a miss. So the
    moment click coordinates stop being centroids -- which is exactly what
    REQ-ARC-WMTE-5950's per-object pixel sampling does -- every click row misses the map
    and the tier barrier silently becomes a no-op. That was measured directly before this
    parameter existed: a sampled non-centroid pixel missed the centroid-keyed map for
    11/11 (r11l), 37/37 (lp85), 63/63 (bp35) and 113/113 (sc25) probed multi-pixel objects,
    i.e. 100%. Shipping the sampler without this would have quietly disabled the barrier on
    precisely the CLICK games the barrier was measured to help.

    Per-cell keys cannot collide between distinct objects (objects are disjoint pixel
    sets) and are bounded by the frame's cell count, so the map stays small. Where a key
    IS contested -- one object's centroid landing on another object's cell -- the MOST
    ELIGIBLE (numerically lowest) tier wins, the same fail-open rule the centroid-only map
    already used for centroid collisions: an over-eager click costs ordering, whereas a
    wrongly-deferred one can cost reachability.

    WHY reuse ``arc_graph_explore``'s predicate instead of writing a fresh one: that module
    already ports the reference's ``frame_segments_to_action_groups`` verbatim (same
    salient-colour set, same medium-width bounds, same status-bar colour), and it is the
    function that produced the already-measured tier-SORT null. Deriving the tiers from a
    second, independently-written predicate here would mean a difference in the A/B result
    could be a difference in the predicate rather than in the discipline -- an
    uninterpretable experiment. So the barrier gates on exactly the same tiers the sort
    used; the ONLY thing that changes between the null and this graft is how the tiers are
    consumed.

    The ``(x, y)`` key convention (x = truncated centroid column-mean, y = truncated
    centroid row-mean) is chosen to match the click coordinates the candidate generator
    actually emits, so lookups hit rather than silently falling back to ``DEFAULT_TIER``.
    """

    from carnot.agentic.arc_graph_explore import (
        _TIER_MAX_WIDTH,
        _TIER_MIN_WIDTH,
        _TIER_SALIENT_COLORS,
        _TIER_STATUS_BAR_COLOR,
    )
    from carnot.agentic.arc_solver_kit import object_centric_digest

    out: dict[tuple[int, int], int] = {}
    cells_by_signature: dict[str, tuple[tuple[int, int], ...]] = {}
    if include_cells:
        # Reuse the sampler's partition, which is verified component-for-component
        # identical to object_centric_digest's (tests/python/test_arc_component_sampling.py).
        # Matching on the digest's own signature string (colour + area + bbox) keeps the two
        # views aligned without depending on iteration order -- the digest SORTS its
        # components, so a positional zip would silently mis-pair them.
        try:
            from carnot.agentic.arc_component_sampling import component_partition

            part = component_partition(grid)
            for idx, cells in enumerate(part.cells):
                color = int(part.colors[idx])
                ys = [y for _x, y in cells]
                xs = [x for x, _y in cells]
                sig = f"c{color}:a{len(cells)}:bbox{min(ys)},{min(xs)},{max(ys)},{max(xs)}"
                cells_by_signature.setdefault(sig, cells)
        except Exception:
            # Fails to the centroid-only map = today's behaviour. Never raises into the
            # live agent's candidate path.
            cells_by_signature = {}

    def _keep(key: tuple[int, int], tier_value: int) -> None:
        prev = out.get(key)
        out[key] = tier_value if prev is None else min(prev, tier_value)

    for comp in object_centric_digest(grid)["components"]:
        bbox = comp["bbox"]  # [min_row, min_col, max_row, max_col]
        height = int(bbox[2]) - int(bbox[0]) + 1
        width = int(bbox[3]) - int(bbox[1]) + 1
        color = int(comp["color"])
        salient = color in _TIER_SALIENT_COLORS
        medium = (
            _TIER_MIN_WIDTH <= width <= _TIER_MAX_WIDTH
            and _TIER_MIN_WIDTH <= height <= _TIER_MAX_WIDTH
        )
        if color == _TIER_STATUS_BAR_COLOR:
            tier = STATUS_BAR_TIER
        elif salient and medium:
            tier = 0
        elif medium:
            tier = 1
        elif salient:
            tier = 2
        else:
            tier = 3
        if cells_by_signature:
            for cell in cells_by_signature.get(str(comp.get("signature", "")), ()):
                _keep(cell, tier)
        cx, cy = comp["centroid"]
        # Two components can share a truncated centroid. Keep the MOST eligible tier for
        # that point: the click is a single action and it will in fact hit whichever object
        # occupies that cell, so deferring it on account of a colliding duller object would
        # be strictly wrong.
        _keep((int(cx), int(cy)), tier)
    return out


def tier_map_for_frame(frame: Any, *, include_cells: bool = False) -> dict[tuple[int, int], int]:
    """``click_tier_map`` for a live frame; ``{}`` when the frame has no usable grid.

    Returning an empty map (rather than raising) is deliberate: an empty map means every
    row falls back to ``DEFAULT_TIER`` = 0 = always eligible, so a frame we cannot segment
    degrades to today's un-gated behaviour instead of stalling the search.
    """

    if frame is None:
        return {}
    try:
        from carnot.agentic.arc_graph_explore import grid_of

        grid = grid_of(frame)
    except Exception:
        return {}
    if grid is None:
        return {}
    try:
        return click_tier_map(grid, include_cells=include_cells)
    except Exception:
        return {}


def row_tier(
    row: Mapping[str, Any],
    tier_by_xy: Mapping[tuple[int, int], int],
    *,
    default: int = DEFAULT_TIER,
) -> int:
    """Priority tier for one candidate row ``{"action": int, "data": dict|None}``.

    Non-click actions (keyboard / simple moves) get tier 0, matching the reference, which
    unconditionally adds every available simple action to group 0. Rationale: those actions
    are few, always available, and cheap to test, so there is never a reason to defer them.
    """

    try:
        action_id = int(row.get("action"))
    except Exception:
        return int(default)
    if action_id != 6:
        return 0
    data = row.get("data")
    if not isinstance(data, Mapping):
        return int(default)
    try:
        key = (int(data["x"]), int(data["y"]))
    except Exception:
        return int(default)
    return int(tier_by_xy.get(key, default))


def annotate_tiers(
    rows: Sequence[Mapping[str, Any]],
    frame: Any,
    *,
    tier_by_xy: Mapping[tuple[int, int], int] | None = None,
    include_cells: bool = False,
) -> list[dict[str, Any]]:
    """Stamp ``row["tier"]`` on each candidate row. NEVER reorders.

    ``include_cells`` is forwarded to ``tier_map_for_frame``; pass it True whenever the
    click coordinates being stamped may be per-object SAMPLED pixels rather than centroids
    (REQ-ARC-WMTE-5950), or every row silently reads back as tier 0.

    Not reordering is the whole point of separating this from the already-nulled sort: the
    existing ranker pipeline (frame-change scorer, goal guidance, epistemic ledger, ...)
    keeps deciding order WITHIN a tier, and the tier only decides WHETHER a row is eligible
    yet. Rows are copied so an unstamped caller's list is never mutated underneath it.
    """

    mapping = (
        tier_map_for_frame(frame, include_cells=include_cells) if tier_by_xy is None else tier_by_xy
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["tier"] = row_tier(new_row, mapping)
        out.append(new_row)
    return out


@dataclass(frozen=True)
class TierExhaustionPolicy:
    """MECHANISM (a): the strict GLOBAL priority-tier exhaustion barrier, as a pure policy.

    Holds no mutable state. The caller owns ``active_tier``; every method takes it as an
    argument and returns a decision. That is what makes the barrier testable without an
    environment, and it is why a per-node implementation cannot accidentally masquerade as
    a global one -- ``next_active_tier`` is *given* every node's rows at once.
    """

    tier_count: int = TIER_COUNT

    def eligible_indices(self, rows: Sequence[Mapping[str, Any]], active_tier: int) -> list[int]:
        """Indices of rows selectable at ``active_tier``, in the caller's existing order.

        CUMULATIVE (``tier <= active_tier``), matching the reference's ``has_open_group``,
        which loops ``range(group_id + 1)``. A ``== active_tier`` reading would make already
        admitted lower tiers spuriously unreachable once the barrier advanced.
        """

        limit = int(active_tier)
        return [idx for idx, row in enumerate(rows) if int(row.get("tier", DEFAULT_TIER)) <= limit]

    def node_has_open_tier(
        self, rows: Sequence[Mapping[str, Any]] | None, active_tier: int
    ) -> bool:
        """Does this node still have work admitted by the barrier? (``has_open_group``)"""

        if not rows:
            return False
        limit = int(active_tier)
        return any(int(row.get("tier", DEFAULT_TIER)) <= limit for row in rows)

    def select_index(
        self,
        rows: Sequence[Mapping[str, Any]],
        active_tier: int,
        *,
        rng: Any | None = None,
        top_k: int | None = None,
    ) -> Optional[int]:
        """Choose which eligible row to expand. ``None`` when nothing is eligible.

        ``rng is None`` (default) -> the FIRST eligible row, preserving Carnot's proven
        greedy ``pop(0)`` semantics restricted to the admitted tiers. This is the
        conservative arm: with the barrier off it is byte-identical to today.

        ``rng`` supplied -> UNIFORM RANDOM among the eligible rows (optionally among only
        the first ``top_k`` of them), which is what the reference actually does
        (``random.choice`` over untested edges in groups ``0..p``). This arm exists because
        three prior Carnot experiments showed that replacing that uniform draw with ANY
        Carnot-scored ordering lost solves -- and Carnot's live ``pop(0)`` IS such a
        replacement (the fully-greedy one). So "greedy draw" is a live suspect, and the A/B
        must be able to vary it independently of the barrier or the result is
        uninterpretable.
        """

        eligible = self.eligible_indices(rows, active_tier)
        if not eligible:
            return None
        if rng is None:
            return eligible[0]
        pool = eligible if top_k is None else eligible[: max(1, int(top_k))]
        return pool[rng.randrange(len(pool))]

    def next_active_tier(
        self,
        nodes_rows: Iterable[Sequence[Mapping[str, Any]] | None],
        active_tier: int,
    ) -> int:
        """Advance the GLOBAL barrier, possibly by several tiers. Pure.

        The reference triggers advancement on UNREACHABILITY: it advances while the current
        node's distance to any open node is INFINITY. That trigger does not port literally,
        and the difference matters enough to state explicitly. In the reference, a node with
        remaining work can genuinely become unreachable, because its only navigation is
        walking forward over discovered edges. Carnot can ALWAYS reach any node, because it
        can RESET and replay that node's recorded path from the root -- so a Carnot distance
        is never infinite and the reference's literal trigger would never fire.

        The faithful analogue of "no open node is reachable" in a graph where everything is
        reachable is therefore "no open node EXISTS": advance only when NOTHING anywhere is
        still open at the active tier. That preserves the semantic the mechanism is actually
        for -- no tier ``p+1`` action anywhere before every tier ``<= p`` action everywhere
        -- which is precisely the property a per-node sort does not have.

        Looping (rather than a single increment) mirrors the reference's ``while``: tier
        ``p+1`` may itself be empty across the whole graph, and stopping there would stall
        the search for one whole decision per empty tier.
        """

        tier = int(active_tier)
        ceiling = max(0, int(self.tier_count) - 1)
        # Materialise once: the caller may hand us a generator and we need several passes.
        rows_list = list(nodes_rows)
        while tier < ceiling and not any(self.node_has_open_tier(rows, tier) for rows in rows_list):
            tier += 1
        return tier


@dataclass(frozen=True)
class FrontierDistanceField:
    """Output of the multi-source reverse BFS: hops-to-nearest-open-node, plus next hop.

    ``distance[u]`` = minimum number of KNOWN-WORKING forward edges from ``u`` to any node
    that still has open work. ``next_hop[u] = (action, v)`` is the first edge along such a
    minimal route. A node absent from ``distance`` is unreachable-to-any-open-node over the
    known edges (the reference's INFINITY).
    """

    distance: dict[Hashable, int] = field(default_factory=dict)
    next_hop: dict[Hashable, tuple[Mapping[str, Any], Hashable]] = field(default_factory=dict)

    def is_open(self, node: Hashable) -> bool:
        """True iff ``node`` is itself a gradient source (distance 0 = has open work)."""

        return self.distance.get(node) == 0


def reverse_adjacency(
    forward_adj: Mapping[Hashable, Iterable[tuple[Mapping[str, Any], Hashable]]],
) -> dict[Hashable, list[tuple[Mapping[str, Any], Hashable]]]:
    """Invert ``origin -> [(action, target)]`` into ``target -> [(action, origin)]``.

    Provided as a pure helper so the reverse index can be rebuilt from scratch in a test
    (or after any graph surgery) rather than only maintained incrementally. The explorer
    maintains its own incremental copy for cost reasons; this is the reference definition
    that copy must agree with.
    """

    rev: dict[Hashable, list[tuple[Mapping[str, Any], Hashable]]] = {}
    for origin, edges in forward_adj.items():
        for action, target in edges:
            rev.setdefault(target, []).append((action, origin))
    return rev


def frontier_distance_field(
    reverse_adj: Mapping[Hashable, Iterable[tuple[Mapping[str, Any], Hashable]]],
    open_nodes: Iterable[Hashable],
) -> FrontierDistanceField:
    """MECHANISM (b): the MULTI-SOURCE reverse BFS over known-working edges.

    Seeds EVERY open node at distance 0 simultaneously and relaxes backwards, so one pass
    labels the whole graph with "how far am I from the nearest thing worth trying, and
    which action starts me down that route". This is the reference's
    ``GraphExplorer._rebuild_distances``.

    Multi-source is the load-bearing word. The naive alternative -- run a separate
    single-source search per frontier candidate to score it -- is what Carnot does today in
    its navigation tie-break, and it costs one full traversal PER candidate PER decision.
    One multi-source pass answers the same question for every node at once, so the graft is
    also strictly cheaper than the code it displaces.

    "Known-working edges" is likewise load-bearing: the reverse index must be built only
    from transitions that actually changed state (the reference records reverse edges only
    on ``success == 1``). Including failed/no-op actions would let the gradient promise a
    route that does not exist.
    """

    dist: dict[Hashable, int] = {}
    nxt: dict[Hashable, tuple[Mapping[str, Any], Hashable]] = {}
    queue: deque[Hashable] = deque()
    for node in open_nodes:
        if node in dist:
            continue
        dist[node] = 0
        queue.append(node)
    while queue:
        v = queue.popleft()
        dv = dist[v]
        for action, u in reverse_adj.get(v, ()) or ():
            if u in dist:
                continue  # BFS: the first label is already minimal
            dist[u] = dv + 1
            nxt[u] = (action, v)
            queue.append(u)
    return FrontierDistanceField(distance=dist, next_hop=nxt)


def nearest_open_node(
    field_: FrontierDistanceField,
    start: Optional[Hashable],
    *,
    max_hops: int | None = None,
) -> Optional[Hashable]:
    """Follow the gradient downhill from ``start`` to the open node it leads to.

    Returns ``start`` itself when ``start`` is already open (distance 0), and ``None`` when
    ``start`` is unreachable to any open node over the known-working edges -- which is
    exactly the condition that makes MECHANISM (a) advance its barrier in the reference.

    ``max_hops`` bounds the walk purely as a defensive guard against a malformed field
    (BFS next-hop pointers are acyclic by construction, so a well-formed field terminates
    in ``distance[start]`` steps).
    """

    if start is None:
        return None
    if start not in field_.distance:
        return None
    limit = field_.distance[start] if max_hops is None else int(max_hops)
    node: Hashable = start
    for _ in range(int(limit) + 1):
        if field_.distance.get(node) == 0:
            return node
        step = field_.next_hop.get(node)
        if step is None:
            return None
        node = step[1]
    return node if field_.distance.get(node) == 0 else None


def gradient_frontier_target(
    forward_adj: Mapping[Hashable, Iterable[tuple[Mapping[str, Any], Hashable]]] | None,
    reverse_adj: Mapping[Hashable, Iterable[tuple[Mapping[str, Any], Hashable]]] | None,
    open_nodes: Iterable[Hashable],
    current: Optional[Hashable],
) -> Optional[Hashable]:
    """End-to-end MECHANISM (b) decision: which open node should we go to from ``current``?

    Answers "the navigation-NEAREST one", replacing Carnot's "the shallowest-from-root one".
    ``None`` means the gradient has no opinion (nothing open, or no known-working route from
    here), in which case the caller must fall back to its existing ordering -- the gradient
    is a preference, never a veto, so it can never make a reachable node unreachable.

    ``forward_adj`` is accepted so callers that do not maintain an incremental reverse index
    can pass only the forward one and have it inverted here; if both are given, the supplied
    reverse index wins (it is the caller's incrementally-maintained copy).

    CALLER OBLIGATION (learned the hard way, 2026-07-24). ``current`` is normally itself a
    member of ``open_nodes``, and a source of the multi-source BFS sits at distance 0 -- so this
    function returns ``current`` unchanged whenever ``current`` still has open work. That is
    correct as a pure answer to "which open node is nearest", but it means a caller who expands a
    returned ``current`` IN PLACE, bypassing whatever depth/backtrack cap it applies elsewhere,
    turns this preference into a cap-cancelling intervention rather than a re-ordering one. Such
    a caller must exclude its own current node from ``open_nodes`` when it has already decided
    not to expand there. ``StepwiseExplorer._gradient_frontier_target`` does exactly that for its
    ``max_depth`` cap and counts the exclusions.
    """

    opens = list(open_nodes)
    if not opens:
        return None
    rev = reverse_adj
    if rev is None:
        rev = reverse_adjacency(forward_adj or {})
    field_ = frontier_distance_field(rev, opens)
    return nearest_open_node(field_, current)
