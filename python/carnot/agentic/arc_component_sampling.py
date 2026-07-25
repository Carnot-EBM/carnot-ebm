"""REQ-ARC-WMTE-5950 -- per-component CLICK-TARGET SAMPLING (the just-explore
generation rule), as a pure, testable function.

=====================================================================================
WHAT THIS IS, IN PLAIN ENGLISH
=====================================================================================
When the live agent decides to click on a detected object, it has to pick an actual
pixel. Today it always picks the object's CENTROID (the average of the object's cell
coordinates, truncated to an integer). One object => exactly one clickable coordinate,
forever.

The just-explore reference solver (Family-A graph explore, arXiv:2512.24156) does
something different and almost embarrassingly simple: it picks a UNIFORM RANDOM PIXEL
belonging to that object, and it draws a FRESH one every time it revisits that object.
One object => every one of its pixels is reachable, across visits.

This module implements that rule. It is deliberately PURE (grid in, coordinates out) so
it can be unit-tested without an environment, a game, or a search.

=====================================================================================
WHY THIS IS WORTH IMPLEMENTING (the measured lead, not a hunch)
=====================================================================================
1. The reference wins game r11l in 9 of 9 measured cells; our baseline explorer wins it
   in 0 of its measured cells. Critically, the reference STILL wins r11l 3/3 when its
   entire frontier-search discipline is deleted (``favor_frontier_search=False`` ->
   plain ``random.choice``). So whatever the reference has that we lack survives
   deleting its search ordering -- which points at its CANDIDATE GENERATION.

2. Our centroid is frequently not even a pixel OF THE OBJECT IT CAME FROM. Measured on
   204 distinct real r11l states reached by walking the offline env: 204 of 204 states
   (100%) contained at least one object whose truncated centroid lies OUTSIDE that
   object's own cells, mean 5.94 such objects per state. Concretely on the r11l reset
   frame, a colour-15 object of area 20 has a centroid that lands on the frame's single
   rarest pixel (a lone colour-6 cell) -- so "click that object" actually clicks a
   different thing entirely. This is a plain defect, independent of any search policy:
   a C-shaped, ring-shaped, or diagonal object has a hollow centre.

3. Truncated centroids COLLIDE, and the live candidate builder de-duplicates by (x, y),
   silently dropping the colliding candidates (measured: r11l 37 objects -> 34 click
   rows; sc25 22 -> 20). Distinct objects are disjoint pixel sets, so a per-object pixel
   sample can never collide -- so once each colliding object is actually SAMPLED, the drop
   goes to zero.

   CORRECTION (2026-07-25, found by adversarial review before any flag flip): the first
   implementation of this module did NOT deliver that property, it INVERTED it into a worse
   defect. ``index_by_centroid`` kept only the FIRST claimant of a colliding truncated
   centroid, and resolution consulted that map before cell containment, so BOTH of the two
   generated points resolved to the SAME object and the second object received ZERO click
   candidates. Flag OFF, the single de-duplicated row at least landed on whichever object
   occupied that cell; flag ON, that object became UNREACHABLE. Measured on real offline
   reset frames (10 games, 441 objects): 9 objects (2.0%) lost all reachability, including
   3 of r11l's 37 and 4 of dc22's 35, and 5 of the 9 were <=8px -- precisely the
   small-object class ``arc_graph_explore.py``'s REQ-ARC-FCP-5758 identifies as carrying
   the winning clicks. The fix is OCCURRENCE-AWARE resolution (see
   ``sample_component_click_points``): every claimant of a contested centroid key is
   recorded, and the Nth occurrence of that key in one call resolves to the Nth claimant,
   so each colliding object gets exactly one slot and none is starved. Which input slot
   maps to which colliding object may permute within the contested set -- that is
   deliberate and harmless, because every one of those slots carries the IDENTICAL input
   coordinate, so there is no per-slot ordering to preserve among them in the first place.

4. On some games the click coordinate is the PARAMETER of the move, not merely a
   selector. r11l's own registry entry (``ops/arc_solve_registry.yaml``) records:
   "clicking elsewhere moves the selected handle to the click minus half-handle size".
   There, collapsing an object to one fixed coordinate collapses a pixel-continuous
   action space to ~41 frozen coordinates -- a genuine CAPABILITY loss, not a ranking
   inefficiency.

=====================================================================================
WHAT THIS IS *NOT* CLAIMED TO FIX (read this before crediting it with r11l)
=====================================================================================
A parallel investigation of r11l specifically found that r11l's failure is NOT a
candidate-generation miss: a winning 3-click sequence for r11l level 1 exists ENTIRELY
inside our current candidate set (each click present at ranks 22/34, 6/26, 25/26 at the
exact state it must be issued from). r11l's actual defect is STATE-IDENTITY ALIASING --
it renders a monotone step counter into frame column 0, our automatic HUD mask resolves
to None on that game, so every action mints a brand-new graph node (measured 1392 nodes
over 31 true game states, 44.9x inflation) and one wall-blocked inert click is re-popped
1371 times out of 1956 actions. Masking column 0 out of node identity, changing nothing
else, flips r11l from 0 to 1 level on 3 of 3 seeds.

So: this module is a GENERAL lever, justified by items 2-4 above (which are
policy-independent structural defects measured on real frames). It is NOT the r11l fix,
and any artifact using it must set its headline gate on a full-corpus regression result
rather than on r11l. Forcing this fix onto that diagnosis would put a wrong causal story
on the record.

=====================================================================================
PARTITION FIDELITY (why we do not just use ``connected_color_blobs`` as-is)
=====================================================================================
The live click generator's objects come from ``arc_solver_kit.object_centric_digest``,
which excludes the single most-common colour WHOLESALE as background. ``connected_color_
blobs`` instead suppresses only components above ``max_component_fraction``. Those are
DIFFERENT partitions, so sampling from blobs would change WHICH objects are candidates
and confound a coordinate-only A/B with a set-membership change.

We therefore reuse ``connected_color_blobs`` (vectorized, cached, and already the
project's cell-carrying primitive) with ``max_component_fraction=1.0`` so it suppresses
nothing, and then apply the digest's own most-common-colour background rule ourselves.
The result is verified component-for-component identical to ``object_centric_digest``'s
partition in ``tests/python/test_arc_component_sampling.py`` -- that equivalence test is
the guard against this drifting into a silent set-membership change.

=====================================================================================
HIDDEN-GAME LEGALITY
=====================================================================================
Everything here consumes ONLY the rendered frame grid. No game source, no per-game
constant, no offline ground truth, no exhaustive calibration. It is therefore legal on
an unseen hidden game (CLAUDE.md "ARC Live-Path Reachability Discipline"). The reference
implementation was READ for its rule and re-implemented here from the described
behaviour; no reference code is copied (it is a separate repository under its own
licence).
"""

from __future__ import annotations

import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "ComponentPartition",
    "component_partition",
    "sample_component_click_points",
    "redraw_component_pixel",
    "SamplingDiagnostics",
]


@dataclass(frozen=True)
class ComponentPartition:
    """The frame's non-background objects, each with its full cell set.

    ``cells[i]`` is object ``i``'s cells as ``(x, y)`` pairs -- the SAME (x, y) order the
    live click rows use (``{"x": ..., "y": ...}``), deliberately not the (row, col) order
    the underlying grid uses, so no caller has to remember to swap and none can silently
    forget to.

    ``index_by_centroid`` maps each object's TRUNCATED centroid ``(int(cx), int(cy))`` to
    its index. That key is exactly what today's candidate generator emits as the click
    coordinate, so a generated point can be traced back to the object that produced it
    even when the centroid does not lie on that object (measured: that is the common
    case, not the exception). When two objects share a truncated centroid this map keeps
    the FIRST claimant, so it stays a plain ``key -> index`` mapping for every caller that
    only needs a stable single answer; ``centroid_claimants`` below is the complete record.

    ``centroid_claimants`` maps each truncated centroid to the FULL tuple of object indices
    that produced it, in object-index order. This exists because keeping only the first
    claimant made every other claimant UNREACHABLE once sampling was enabled (see the
    module docstring's CORRECTION). Any resolution that has to hand out one object per
    generated point must consult THIS map, not ``index_by_centroid``.

    ``index_by_cell`` maps every object cell to its index, which is how an ALREADY-SAMPLED
    coordinate (a real member pixel) is resolved back to its object for a redraw.
    """

    cells: tuple[tuple[tuple[int, int], ...], ...]
    colors: tuple[int, ...]
    index_by_centroid: Mapping[tuple[int, int], int]
    index_by_cell: Mapping[tuple[int, int], int]
    background_color: Optional[int]
    centroid_claimants: Mapping[tuple[int, int], tuple[int, ...]] = ()  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.cells)

    def claimants_for_centroid(self, x: int, y: int) -> tuple[int, ...]:
        """Every object whose truncated centroid is ``(x, y)``, in object-index order.

        Empty tuple when the point is not any object's centroid. Falls back to the
        single-claimant map when ``centroid_claimants`` was not populated, so a partition
        built by older code (or by a test double) still resolves rather than silently
        reporting that no object claims a key it demonstrably does.
        """

        key = (int(x), int(y))
        claims = self.centroid_claimants or {}
        hit = claims.get(key) if hasattr(claims, "get") else None
        if hit:
            return tuple(int(i) for i in hit)
        single = self.index_by_centroid.get(key)
        return (int(single),) if single is not None else ()

    def index_for_point(self, x: int, y: int) -> Optional[int]:
        """Which object does the click coordinate ``(x, y)`` belong to?

        Resolution order, most-specific first:

        1. The point IS a truncated centroid -> the object that produced it. Checked
           first because the point may be a centroid that lies outside its own object
           (the 100%-of-r11l-states case), where containment would resolve it to the
           WRONG object or to nothing at all.
        2. The point is a member cell -> the object containing it. This is the path a
           previously-sampled pixel takes on a redraw.
        3. Neither -> ``None``. The caller must then leave the coordinate untouched
           (fail OPEN to today's behaviour) rather than guess: points can legitimately
           come from other producers (grid-fallback background tiles, a learned action
           prior, the centre-of-grid fallback) that this partition does not describe.
        """

        key = (int(x), int(y))
        hit = self.index_by_centroid.get(key)
        if hit is not None:
            return int(hit)
        hit = self.index_by_cell.get(key)
        if hit is not None:
            return int(hit)
        return None

    def sample_pixel(self, index: int, rng: random.Random) -> tuple[int, int]:
        """One UNIFORM random member pixel of object ``index``, as ``(x, y)``.

        Uniform over cells (not over the bounding box, and not the geometric centre):
        a hollow or L-shaped object's bounding-box centre may be background, whereas
        every cell drawn here is guaranteed to be part of the object. Drawing WITH
        REPLACEMENT is the caller's business -- this function keeps no memory of what it
        returned, exactly like the reference, whose only bookkeeping is per-object and
        never per-pixel.
        """

        pool = self.cells[int(index)]
        if not pool:  # pragma: no cover -- empty objects are never constructed
            raise ValueError(f"object {index} has no cells")
        return pool[rng.randrange(len(pool))]

    def sample_pixels(self, index: int, rng: random.Random, k: int) -> list[tuple[int, int]]:
        """``k`` draws for object ``index``, de-duplicated but never padded.

        De-duplicated because two identical coordinates in one candidate list are two
        rows for literally the same action -- the live builder would drop the second
        anyway, and keeping it would burn a slot in the ``max_click`` budget for nothing.
        NOT padded up to ``k``: an object with 3 cells genuinely only has 3 distinct
        clicks, and inventing more would mean sampling a different object's pixels.
        """

        k = max(1, int(k))
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        pool = self.cells[int(index)]
        # Bounded attempt count: with replacement, hitting k distinct cells is not
        # guaranteed in k draws. 4*k attempts keeps this O(k) while making a short return
        # rare for the small k values this feature uses (default 1).
        attempts = 4 * k
        while len(out) < min(k, len(pool)) and attempts > 0:
            attempts -= 1
            point = self.sample_pixel(index, rng)
            if point in seen:
                continue
            seen.add(point)
            out.append(point)
        if not out:  # pragma: no cover -- only reachable if pool is empty
            out = [self.sample_pixel(index, rng)]
        return out


@dataclass(frozen=True)
class SamplingDiagnostics:
    """Per-call counters, so an A/B arm can never be an UNINSTRUMENTED arm.

    This exists because of a real prior measurement failure in this project: an entire
    control arm reported ``states_expanded=None`` on all 225 of its rows, so a 72-97%
    crashed arm read as a legitimate null across 975 cells. Every mechanism added after
    that must be able to say how often it actually did anything.

    ``coordinates_changed`` is the ONE counter that answers "did this mechanism actually do
    anything". The others describe HOW points were resolved; a dead sampler can still report
    a healthy-looking ``points_in``/``points_out``. Added 2026-07-25 after adversarial review
    showed that a totally dead sampler (``component_partition`` patched to raise on every
    call) was INDISTINGUISHABLE from a working one in the emitted artifact: the arm reported
    ``click_pixel_rows_sampled=1, click_pixel_errors=0`` while emitting the unmodified
    centroid. ``click_pixel_rows_sampled`` counts click rows PRESENT, not coordinates
    REPLACED, so it is not an activity counter at all.
    """

    points_in: int = 0
    points_out: int = 0
    resolved_via_centroid: int = 0
    resolved_via_cell: int = 0
    unresolved: int = 0
    centroid_outside_own_component: int = 0
    errors: int = 0
    coordinates_changed: int = 0
    contested_centroid_points: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "points_in": int(self.points_in),
            "points_out": int(self.points_out),
            "resolved_via_centroid": int(self.resolved_via_centroid),
            "resolved_via_cell": int(self.resolved_via_cell),
            "unresolved": int(self.unresolved),
            "centroid_outside_own_component": int(self.centroid_outside_own_component),
            "errors": int(self.errors),
            "coordinates_changed": int(self.coordinates_changed),
            "contested_centroid_points": int(self.contested_centroid_points),
        }


_PARTITION_CACHE_MAX_SIZE = 8
_partition_cache: "OrderedDict[tuple, ComponentPartition]" = OrderedDict()


def _as_grid(grid: Any):
    """Coerce a frame / nested list / array to a 2-D integer grid.

    Mirrors ``arc_color_blob_salience._as_grid``: a live ARC frame is a LIST of grids
    (one per rendered sub-frame) and the LAST one is the current observation.
    """

    import numpy as np

    arr = np.asarray(grid.frame if hasattr(grid, "frame") else grid)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D ARC grid, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def component_partition(grid: Any) -> ComponentPartition:
    """Objects of ``grid`` under the SAME rule the live click generator uses.

    Cached (bounded LRU keyed on grid bytes) because within one ``next_move()`` every
    candidate shares the same frame, and the search may resolve the same node's frame
    many times across redraws. Same reasoning and same bound as
    ``arc_color_blob_salience._cached_blobs_and_counts``.
    """

    arr = _as_grid(grid)
    key = (arr.shape, arr.tobytes())
    cached = _partition_cache.get(key)
    if cached is not None:
        _partition_cache.move_to_end(key)
        return cached
    part = _build_partition(arr)
    _partition_cache[key] = part
    if len(_partition_cache) > _PARTITION_CACHE_MAX_SIZE:
        _partition_cache.popitem(last=False)
    return part


def _build_partition(arr) -> ComponentPartition:
    from carnot.agentic.arc_color_blob_salience import connected_color_blobs

    counts = Counter(int(v) for v in arr.flatten().tolist())
    # object_centric_digest's background rule, reproduced exactly: the single most-common
    # colour is excluded wholesale. numpy's argmax over np.unique's counts breaks ties by
    # the SMALLEST colour value (unique() is sorted ascending), so Counter.most_common --
    # which breaks ties by insertion order -- would silently disagree on a tied frame.
    background: Optional[int] = None
    if counts:
        background = min((c for c in counts if counts[c] == max(counts.values())), default=None)

    # max_component_fraction=1.0 disables blob-size suppression, so the ONLY exclusion is
    # the background-colour rule above. That is what makes this partition equal to the
    # digest's (asserted in tests) instead of merely similar to it.
    blobs = connected_color_blobs(arr, min_pixels=1, max_component_fraction=1.0)

    cells: list[tuple[tuple[int, int], ...]] = []
    colors: list[int] = []
    by_centroid: dict[tuple[int, int], int] = {}
    claimants: dict[tuple[int, int], list[int]] = {}
    by_cell: dict[tuple[int, int], int] = {}
    for blob in blobs:
        if background is not None and int(blob.color) == int(background):
            continue
        index = len(cells)
        # ColorBlob.cells are (row, col); the click convention is (x, y) = (col, row).
        pts = tuple(sorted((int(x), int(y)) for (y, x) in blob.cells))
        cells.append(pts)
        colors.append(int(blob.color))
        cy, cx = blob.centroid  # ColorBlob.centroid is (row_mean, col_mean)
        ckey = (int(cx), int(cy))
        # A truncated centroid can be shared by two objects. ``by_centroid`` keeps the FIRST
        # claimant so it remains a stable single-answer map, but EVERY claimant is recorded in
        # ``claimants``. Recording only the first is what made colliding objects unreachable
        # once sampling was enabled -- see the module docstring's CORRECTION. This map is the
        # complete record; the resolution policy lives in sample_component_click_points.
        by_centroid.setdefault(ckey, index)
        claimants.setdefault(ckey, []).append(index)
        for point in pts:
            by_cell.setdefault(point, index)
    return ComponentPartition(
        cells=tuple(cells),
        colors=tuple(colors),
        index_by_centroid=by_centroid,
        index_by_cell=by_cell,
        background_color=background,
        centroid_claimants={k: tuple(v) for k, v in claimants.items()},
    )


def sample_component_click_points(
    grid: Any,
    points: Sequence[tuple[int, int]],
    *,
    rng: random.Random,
    samples_per_component: int = 1,
    partition: Optional[ComponentPartition] = None,
) -> tuple[list[tuple[int, int]], SamplingDiagnostics]:
    """Replace each already-CHOSEN click point with uniform pixel(s) of its own object.

    Contract, and why each half of it matters:

    * ORDER IS PRESERVED. Each input point expands IN PLACE. Every existing ordering
      lever (the flat area x colour-rarity salience sort, the 5-tier schedule, the
      small-object-first sort, a learned action prior, the grid-fallback tail) has
      already run by the time this is called, so preserving order is what makes this
      lever ORTHOGONAL to all of them -- it varies the coordinate and nothing else.
    * UNRESOLVED POINTS PASS THROUGH UNCHANGED. Fails open to today's coordinate. A
      point may come from a producer this partition does not describe, and dropping it
      would be a silent capability loss.
    * ``samples_per_component`` > 1 emits several pixels for the same object. The caller
      is responsible for having already bounded the object count, because expanding by
      k AFTER a per-POINT cap would divide the reachable object count by k -- turning a
      coordinate experiment into a budget experiment.
    * RESOLUTION IS OCCURRENCE-AWARE, so a CONTESTED truncated centroid does not starve an
      object. Two objects can share a truncated centroid; the generator then emits that same
      coordinate once per object. Handing every occurrence to the first claimant (the
      original implementation) left the other claimant with ZERO click candidates -- a
      REACHABILITY REGRESSION versus flag-off, measured at 2.0% of objects on real reset
      frames and concentrated in the small-object class that carries winning clicks. So this
      function keeps a per-call occurrence counter per centroid key: the Nth occurrence of a
      contested key resolves to the Nth claimant. Each colliding object therefore gets
      exactly one slot. The assignment of slot-to-object may permute within the contested
      set, which costs nothing: every slot in that set carries the same input coordinate, so
      there is no per-slot ordering among them to preserve.
    """

    try:
        part = partition if partition is not None else component_partition(grid)
    except Exception:
        # A frame we cannot segment degrades to today's coordinates rather than
        # stalling the search. Counted, never silent. coordinates_changed stays 0, which is
        # what makes a dead sampler mechanically detectable downstream.
        return [(int(x), int(y)) for x, y in points], SamplingDiagnostics(
            points_in=len(points), points_out=len(points), unresolved=len(points), errors=1
        )

    out: list[tuple[int, int]] = []
    resolved_centroid = 0
    resolved_cell = 0
    unresolved = 0
    centroid_outside = 0
    changed = 0
    contested = 0
    # How many times this call has already consumed each contested centroid key.
    consumed: dict[tuple[int, int], int] = {}
    for raw_x, raw_y in points:
        x, y = int(raw_x), int(raw_y)
        key = (x, y)
        claims = part.claimants_for_centroid(x, y)
        if claims:
            # Provenance path (centroid). Hand out claimants in order, one per occurrence,
            # cycling only if the same key appears more often than there are claimants.
            nth = consumed.get(key, 0)
            consumed[key] = nth + 1
            index = int(claims[nth % len(claims)])
            resolved_centroid += 1
            if len(claims) > 1:
                contested += 1
            if key not in part.cells[index]:
                centroid_outside += 1
        else:
            hit = part.index_by_cell.get(key)
            if hit is None:
                unresolved += 1
                out.append(key)
                continue
            index = int(hit)
            resolved_cell += 1
        drawn = part.sample_pixels(index, rng, samples_per_component)
        changed += sum(1 for point in drawn if point != key)
        out.extend(drawn)
    return out, SamplingDiagnostics(
        points_in=len(points),
        points_out=len(out),
        resolved_via_centroid=resolved_centroid,
        resolved_via_cell=resolved_cell,
        unresolved=unresolved,
        centroid_outside_own_component=centroid_outside,
        coordinates_changed=changed,
        contested_centroid_points=contested,
    )


def redraw_component_pixel(
    grid: Any,
    x: int,
    y: int,
    *,
    rng: random.Random,
    partition: Optional[ComponentPartition] = None,
) -> tuple[Optional[int], Optional[tuple[int, int]]]:
    """One fresh draw for the object owning ``(x, y)``: ``(object_index, (x, y))``.

    This is the WITH-REPLACEMENT half of the reference's rule. The reference keeps no
    record of which pixel it clicked -- only of which OBJECT produced a state change --
    so revisiting an object yields a brand-new pixel. Returns ``(None, None)`` when the
    coordinate cannot be attributed to an object, so the caller can decline to redraw
    instead of guessing.
    """

    try:
        part = partition if partition is not None else component_partition(grid)
    except Exception:
        return None, None
    index = part.index_for_point(int(x), int(y))
    if index is None:
        return None, None
    return index, part.sample_pixel(index, rng)


def centroid_outside_component_rate(grids: Iterable[Any]) -> dict[str, float]:
    """Diagnostic: how often is an object's truncated centroid NOT one of its own cells?

    Kept in the shipped module (not a throwaway probe) because it is the measurement that
    motivates this whole mechanism, and a future reader should be able to re-run it on any
    corpus of frames rather than take the 204-state r11l number on trust.
    """

    n_grids = 0
    n_components = 0
    n_outside = 0
    grids_with_any = 0
    for grid in grids:
        try:
            part = component_partition(grid)
        except Exception:
            continue
        n_grids += 1
        outside_here = 0
        # Iterate CLAIMANTS, not index_by_centroid: two objects sharing a truncated centroid
        # are two components, and counting the key once under-reports the component total
        # (and therefore the rate's denominator) on exactly the frames that motivated the
        # occurrence-aware fix.
        for key, claims in (part.centroid_claimants or {}).items():
            for index in claims:
                n_components += 1
                if key not in part.cells[int(index)]:
                    outside_here += 1
        n_outside += outside_here
        if outside_here:
            grids_with_any += 1
    return {
        "n_grids": float(n_grids),
        "n_components": float(n_components),
        "n_centroid_outside": float(n_outside),
        "frac_grids_with_any_outside": (float(grids_with_any) / n_grids) if n_grids else 0.0,
        "mean_outside_per_grid": (float(n_outside) / n_grids) if n_grids else 0.0,
    }
