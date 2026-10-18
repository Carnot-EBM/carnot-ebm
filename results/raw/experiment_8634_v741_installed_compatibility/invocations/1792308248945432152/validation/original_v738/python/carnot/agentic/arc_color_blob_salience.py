"""Classical color-blob salience for ARC live exploration.

Spec refs: REQ-ARC-FCP-5360, SCENARIO-ARC-FCP-5360,
REQ-ARC-FCP-5373, SCENARIO-ARC-FCP-5373,
REQ-ARC-FCP-5397, SCENARIO-ARC-FCP-5397,
REQ-ARC-FCP-5591, SCENARIO-ARC-FCP-5591.

The live ARC agent already has learned frame-diff and action-prior hooks. This
module adds the cheap perception-grounded fallback those hooks were missing:
segment the rendered frame into single-color connected components and rank
button-like blobs before large dull regions and status-bar artifacts.

``object_hash``/``blob_topology`` (REQ-ARC-FCP-5591) extend that base with the
two sub-components ``ops/known-issues.md``'s 2026-07-11 task 10 entry
identified from an independent read of a real top-3 ARC-AGI-3 competitor's
open-sourced segmentation code: a translation-invariant object-identity
signature, and a containment tree + adjacency graph over the blob list. Both
are purely additive (new functions over the existing ``ColorBlob``/
``connected_color_blobs`` primitives, no changes to their behavior or
signatures) -- this file's existing tier/score/click_points outputs are
unaffected.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np
from scipy import ndimage


SALIENT_COLORS = frozenset(range(6, 16))
STATUS_BAR_COLOR = 16
_FOUR_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# REQ-ARC-FCP-5699 item-2 re-validation (2026-07-16): a bounded module-level per-frame
# cache for connected_color_blobs()'s output. ColorBlobSaliencePrior is a frozen
# dataclass shared across many callers via the generic two-arg `score(frame, candidate)`
# protocol (arc_frame_change_predictor.rank_arc_actions -> _prior_value is one such
# caller) that does NOT have access to action_tier_rows()'s own per-call blobs/
# color_counts cache args. Profiling a real lp85 episode found this uncached path calls
# connected_color_blobs() 8176 times for only 500 actions (once per CANDIDATE, not once
# per FRAME) -- 23.1s of a 43.3s total run, the dominant remaining cost even after
# vectorizing connected_color_blobs() itself (which alone cut lp85 budget=500 from 68s to
# 29.7s but was not sufficient). This cache is keyed on frame content (not object
# identity, since candidate-generation code may re-wrap the same underlying grid in a new
# object per call) via a cheap grid hash, bounded to a small size since within one
# next_move() all candidates share the SAME current frame -- only a handful of distinct
# frames are ever live at once.
_BLOB_CACHE_MAX_SIZE = 8
_blob_cache: "OrderedDict[tuple, tuple[list[ColorBlob], Counter]]" = OrderedDict()


def _cached_blobs_and_counts(
    grid: np.ndarray, *, min_pixels: int, max_component_fraction: float
) -> tuple[list[ColorBlob], Counter]:
    key = (
        grid.shape,
        grid.tobytes(),
        int(min_pixels),
        float(max_component_fraction),
    )
    cached = _blob_cache.get(key)
    if cached is not None:
        _blob_cache.move_to_end(key)
        return cached
    blobs = connected_color_blobs(
        grid, min_pixels=min_pixels, max_component_fraction=max_component_fraction
    )
    color_counts = Counter(int(value) for value in grid.flatten().tolist())
    _blob_cache[key] = (blobs, color_counts)
    if len(_blob_cache) > _BLOB_CACHE_MAX_SIZE:
        _blob_cache.popitem(last=False)
    return blobs, color_counts


@dataclass(frozen=True)
class ColorBlob:
    """One same-color connected component visible in the rendered frame."""

    color: int
    pixel_count: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cells: frozenset[tuple[int, int]]
    frame_shape: tuple[int, int] | None = None

    @property
    def height(self) -> int:
        return int(self.bbox[2] - self.bbox[0] + 1)

    @property
    def width(self) -> int:
        return int(self.bbox[3] - self.bbox[1] + 1)

    def contains_xy(self, x: int, y: int) -> bool:
        return (int(y), int(x)) in self.cells

    @property
    def area_fraction(self) -> float:
        if self.frame_shape is None:
            return 0.0
        area = int(self.frame_shape[0]) * int(self.frame_shape[1])
        if area <= 0:
            return 0.0
        return float(self.pixel_count) / float(area)


def _as_grid(frame: Any) -> np.ndarray:
    arr = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D ARC grid, got shape {arr.shape}")  # pragma: no cover
    return arr.astype(np.int16, copy=False)


def connected_color_blobs(
    frame: Any,
    *,
    min_pixels: int = 1,
    max_component_fraction: float = 0.45,
) -> list[ColorBlob]:
    """Return same-color components while suppressing huge background fills.

    REQ-ARC-FCP-5699 item-2 re-validation (2026-07-16): vectorized via
    ``scipy.ndimage.label`` (one call per DISTINCT color, 4-connectivity structure
    matching the original flood-fill's up/down/left/right neighbor rule), replacing
    a pure-Python per-cell BFS. Verified field-for-field equivalent to the prior
    implementation across 200 randomized grids (varying size, color count,
    min_pixels, max_component_fraction) plus a realistic structured ARC-frame
    shape -- see tests/python/test_arc_color_blob_salience_vectorized.py.
    Measured ~5.9x faster on a realistic structured frame (16 blobs: 2.9ms ->
    0.5ms); on a PATHOLOGICAL near-uniform-random grid (thousands of
    near-singleton blobs, not representative of a designed ARC puzzle frame) the
    per-blob Python object construction can make this SLOWER than the original --
    an honest, measured tradeoff, not hidden. Real ARC games render structured
    puzzle grids (backgrounds + designed shapes), not per-pixel noise, so the
    realistic-frame speedup is what governs actual live-agent performance; the
    matched-budget A/B against real games is the authoritative test.
    """

    grid = _as_grid(frame)
    height, width = grid.shape
    max_pixels = int(max(1, math.floor(height * width * float(max_component_fraction))))
    blobs: list[ColorBlob] = []
    for color in np.unique(grid):
        labeled, n_components = ndimage.label(grid == color, structure=_FOUR_CONNECTIVITY)
        if n_components == 0:
            continue
        # find_objects()[i] is the tight bounding-box slice for label i+1; comparing
        # labeled[slice] == label_id isolates exactly that label's cells within the
        # slice (other labels sharing the same bounding box, if any, don't match).
        for label_id, slice_yx in enumerate(ndimage.find_objects(labeled), start=1):
            if slice_yx is None:
                continue
            sub = labeled[slice_yx] == label_id
            count = int(sub.sum())
            if count < int(min_pixels) or count > max_pixels:
                continue
            ys_local, xs_local = np.nonzero(sub)
            y0, x0 = slice_yx[0].start, slice_yx[1].start
            ys = ys_local + y0
            xs = xs_local + x0
            blobs.append(
                ColorBlob(
                    color=int(color),
                    pixel_count=count,
                    bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
                    centroid=(float(ys.mean()), float(xs.mean())),
                    cells=frozenset(zip(ys.tolist(), xs.tolist())),
                    frame_shape=(height, width),
                )
            )
    return sorted(blobs, key=lambda blob: (blob.bbox, blob.color, blob.pixel_count))


@dataclass(frozen=True)
class ColorBlobSaliencePrior:
    """Five-tier single-color component prior for live ARC candidate ranking."""

    salient_colors: frozenset[int] = field(default_factory=lambda: SALIENT_COLORS)
    status_bar_color: int = STATUS_BAR_COLOR
    min_width: int = 2
    max_width: int = 32
    min_pixels: int = 1
    max_component_fraction: float = 0.45
    keyboard_score: float = 0.0
    status_bar_deprioritization: bool = True
    large_flat_deprioritization: bool = True
    status_bar_min_width_fraction: float = 0.75
    status_bar_max_height: int = 2
    large_flat_min_fraction: float = 0.20
    large_flat_min_pixels: int = 16
    button_like_max_area: int = 64
    button_like_max_aspect: float = 3.0

    def _blob_sort_key(
        self, blob: ColorBlob
    ) -> tuple[float, float, int, tuple[int, int, int, int]]:
        tier = int(self.tier(blob))
        return (
            float(tier),
            -float(self.button_likelihood(blob)),
            -int(blob.pixel_count),
            tuple(int(value) for value in blob.bbox),
        )

    def is_status_bar_like(self, blob: ColorBlob) -> bool:
        """Return true for status-colored or frame-edge status-strip components."""

        if not self.status_bar_deprioritization:
            return False
        if int(blob.color) == int(self.status_bar_color):
            return True
        if blob.frame_shape is None:
            return False
        frame_height, frame_width = int(blob.frame_shape[0]), int(blob.frame_shape[1])
        touches_frame_edge = int(blob.bbox[0]) == 0 or int(blob.bbox[2]) == frame_height - 1
        spans_status_width = float(blob.width) >= (
            float(frame_width) * float(self.status_bar_min_width_fraction)
        )
        return bool(
            touches_frame_edge
            and spans_status_width
            and int(blob.height) <= int(self.status_bar_max_height)
        )

    def is_large_flat_blob(self, blob: ColorBlob) -> bool:
        """Return true for broad visual fields that should not outrank buttons."""

        if not self.large_flat_deprioritization:
            return False
        if int(blob.pixel_count) < int(self.large_flat_min_pixels):
            return False
        aspect = max(
            float(blob.width) / max(1.0, float(blob.height)),
            float(blob.height) / max(1.0, float(blob.width)),
        )
        return bool(
            float(blob.area_fraction) >= float(self.large_flat_min_fraction)
            or aspect >= float(self.button_like_max_aspect)
        )

    def is_button_like_blob(self, blob: ColorBlob) -> bool:
        """Return true for compact, roughly rectangular blobs likely to be controls."""

        if self.is_status_bar_like(blob) or self.is_large_flat_blob(blob):
            return False
        if int(blob.pixel_count) > int(self.button_like_max_area):
            return False
        aspect = max(
            float(blob.width) / max(1.0, float(blob.height)),
            float(blob.height) / max(1.0, float(blob.width)),
        )
        return bool(
            int(self.min_width) <= blob.width <= int(self.max_width)
            and int(self.min_width) <= blob.height <= int(self.max_width)
            and aspect <= float(self.button_like_max_aspect)
        )

    def button_likelihood(self, blob: ColorBlob) -> float:
        """Return a small morphology score used only to order blobs inside a tier."""

        if self.is_status_bar_like(blob):
            return 0.0
        score = 0.0
        if int(blob.color) in self.salient_colors:
            score += 0.35
        if self.is_button_like_blob(blob):
            score += 1.0
        area = max(1.0, float(self.button_like_max_area))
        score += 0.15 * min(1.0, float(blob.pixel_count) / area)
        return float(score)

    def tier(self, blob: ColorBlob) -> int:
        """Return 0..4 where lower means earlier exploration priority."""

        if self.is_status_bar_like(blob):
            return 4
        medium = int(self.min_width) <= blob.width <= int(self.max_width) and int(
            self.min_width
        ) <= blob.height <= int(self.max_width)
        salient = int(blob.color) in self.salient_colors
        if not self.large_flat_deprioritization:
            if salient and medium:
                return 0
            if medium:
                return 1
            if salient:
                return 2
            return 3
        if self.is_large_flat_blob(blob):
            return 3
        if salient and self.is_button_like_blob(blob):
            return 0
        if medium:
            return 1
        if salient:
            return 2
        return 3

    def tier_rows(self, frame: Any) -> list[dict[str, Any]]:
        """Emit connected-component tier evidence for live-path diagnostics."""

        grid = _as_grid(frame)
        blobs = connected_color_blobs(
            grid,
            min_pixels=self.min_pixels,
            max_component_fraction=1.0
            if self.large_flat_deprioritization
            else self.max_component_fraction,
        )
        rows: list[dict[str, Any]] = []
        for blob in sorted(blobs, key=self._blob_sort_key):
            tier = int(self.tier(blob))
            rows.append(
                {
                    "tier": tier,
                    "color": int(blob.color),
                    "pixel_count": int(blob.pixel_count),
                    "bbox": [int(value) for value in blob.bbox],
                    "centroid_y": float(blob.centroid[0]),
                    "centroid_x": float(blob.centroid[1]),
                    "button_like": bool(self.is_button_like_blob(blob)),
                    "button_likelihood": float(self.button_likelihood(blob)),
                    "salient_color": int(blob.color) in self.salient_colors,
                    "status_bar": bool(self.is_status_bar_like(blob)),
                    "large_flat": bool(self.is_large_flat_blob(blob)),
                    "non_status_region": not self.is_status_bar_like(blob),
                }
            )
        return rows

    def click_points(self, frame: Any, *, max_points: int | None = None) -> list[tuple[int, int]]:
        """Return blob-centroid click points in salience-tier generation order."""

        rows = self.tier_rows(frame)
        points: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for row in rows:
            point = (int(row["centroid_x"]), int(row["centroid_y"]))
            if point in seen:
                continue
            seen.add(point)
            points.append(point)
            if max_points is not None and len(points) >= int(max_points):
                break
        return points

    def _candidate_action_id(self, candidate: Any) -> int:
        value = (
            candidate.get("action", candidate.get("action_id", 0))
            if isinstance(candidate, Mapping)
            else getattr(candidate, "action_id", 0)
        )
        return int(value or 0)

    def _candidate_data(self, candidate: Any) -> Mapping[str, Any]:
        data = (
            candidate.get("data")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "data", None)
        )
        return data if isinstance(data, Mapping) else {}

    def _blob_for_click(self, blobs: Sequence[ColorBlob], x: int, y: int) -> ColorBlob | None:
        for blob in blobs:
            if blob.contains_xy(x, y):
                return blob
        if not blobs:
            return None
        return min(
            blobs,
            key=lambda blob: math.dist((float(y), float(x)), blob.centroid),
        )

    def action_tier_rows(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        """Return candidate-tier diagnostics in the same order the prior prefers."""

        grid = _as_grid(frame)
        blobs = connected_color_blobs(
            grid,
            min_pixels=self.min_pixels,
            max_component_fraction=1.0
            if self.large_flat_deprioritization
            else self.max_component_fraction,
        )
        # Computed once and passed to every score() call below -- see score()'s docstring
        # for why per-candidate recomputation of these two frame-level quantities was a
        # severe (O(candidates x grid_cells)) performance bug.
        color_counts = Counter(int(value) for value in grid.flatten().tolist())
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            action_id = self._candidate_action_id(candidate)
            data = self._candidate_data(candidate)
            source = (
                candidate.get("source", "")
                if isinstance(candidate, Mapping)
                else getattr(candidate, "source", "")
            )
            if action_id != 6 or "x" not in data or "y" not in data:
                rows.append(
                    {
                        "index": int(index),
                        "action": int(action_id),
                        "data": dict(data),
                        "source": str(source),
                        "tier": None,
                        "score": float(
                            self.score(frame, candidate, blobs=blobs, color_counts=color_counts)
                        ),
                    }
                )
                continue
            blob = self._blob_for_click(blobs, int(data["x"]), int(data["y"]))
            if blob is None:
                tier = None
                row = {
                    "index": int(index),
                    "action": int(action_id),
                    "data": dict(data),
                    "source": str(source),
                    "tier": tier,
                    "score": 0.0,
                }
            else:
                tier = int(self.tier(blob))
                row = {
                    "index": int(index),
                    "action": int(action_id),
                    "data": dict(data),
                    "source": str(source),
                    "tier": tier,
                    "score": float(
                        self.score(frame, candidate, blobs=blobs, color_counts=color_counts)
                    ),
                    "color": int(blob.color),
                    "button_like": bool(self.is_button_like_blob(blob)),
                    "button_likelihood": float(self.button_likelihood(blob)),
                    "status_bar": bool(self.is_status_bar_like(blob)),
                    "large_flat": bool(self.is_large_flat_blob(blob)),
                    "non_status_region": not self.is_status_bar_like(blob),
                }
            rows.append(row)
        return sorted(
            rows,
            key=lambda row: (
                99 if row.get("tier") is None else int(row["tier"]),
                -float(row.get("score") or 0.0),
                int(row["index"]),
            ),
        )

    def score(
        self,
        frame: Any,
        candidate: Any,
        *,
        blobs: Sequence[ColorBlob] | None = None,
        color_counts: Counter | None = None,
    ) -> float:
        """Score a live candidate; higher values are tried earlier.

        `blobs`/`color_counts` are an optional per-frame cache: computing them is
        O(grid cells) via a flood-fill, and `action_tier_rows` calls `score()` once per
        candidate action (up to one per grid cell on a click-heavy game) -- recomputing
        the SAME frame's decomposition from scratch on every call made a single
        `next_move()` on a large grid (e.g. lp85's 64x64) take O(candidates x grid_cells),
        multi-minute-plus in practice (found 2026-07-14 via a real hang: the offline
        submission gate timed out on 7/8 canonical games with SUBMITTED_COLOR_BLOB_
        SALIENCE_ENABLED=True). Callers outside this module (arc_frame_change_predictor,
        arc_geometric_salience, arc_discriminative_router, etc.) use the generic
        `score(frame, candidate)` two-arg protocol shared across action-prior classes;
        omitting the cache args here preserves that exact call signature.
        """

        if self._candidate_action_id(candidate) != 6:
            return float(self.keyboard_score)
        data = self._candidate_data(candidate)
        if "x" not in data or "y" not in data:
            return 0.0
        grid = _as_grid(frame)
        x = int(data["x"])
        y = int(data["y"])
        if blobs is None or color_counts is None:
            # REQ-ARC-FCP-5699 item-2: the generic two-arg callers (this branch) hit the
            # module-level per-frame cache instead of recomputing from scratch every call
            # -- see _cached_blobs_and_counts's docstring for the 8176-calls-for-500-
            # actions profiling finding this closes.
            cached_blobs, cached_counts = _cached_blobs_and_counts(
                grid,
                min_pixels=self.min_pixels,
                max_component_fraction=1.0
                if self.large_flat_deprioritization
                else self.max_component_fraction,
            )
            if blobs is None:
                blobs = cached_blobs
            if color_counts is None:
                color_counts = cached_counts
        blob = self._blob_for_click(blobs, x, y)
        if blob is None:
            return 0.0
        if len(color_counts) <= 1:
            return 0.0
        tier = self.tier(blob)
        tier_score = max(0, 4 - int(tier)) * 1000.0
        area_score = min(float(blob.pixel_count), 999.0) / 1000.0
        rarity_score = 1.0 / float(1 + color_counts.get(int(blob.color), 0))
        return float(tier_score + area_score + rarity_score)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": "single_color_connected_component_tiers",
            "connected_component_salience_enabled": True,
            "salience_tiers_emitted": True,
            "salient_colors": sorted(int(color) for color in self.salient_colors),
            "status_bar_color": int(self.status_bar_color),
            "max_component_fraction": float(self.max_component_fraction),
            "status_bar_deprioritization": bool(self.status_bar_deprioritization),
            "large_flat_deprioritization": bool(self.large_flat_deprioritization),
        }


def blob_at_click(blobs: Sequence[ColorBlob], x: int, y: int) -> ColorBlob | None:
    """REQ-ARC-FCP-5595: free-function form of ``ColorBlobSaliencePrior._blob_for_click`` --
    the blob a click at ``(x, y)`` lands in, or the nearest-centroid blob as a fallback (a
    click can land on a shared-color background gap between two components). Promoted to a
    module-level function (identical logic, callers pass a full blob list) so callers
    outside ``ColorBlobSaliencePrior`` -- e.g. ``arc_inert_click_pruner.InertClickSigPruner``
    -- can reuse the exact same lookup without depending on the prior's tier/score machinery.
    """

    for blob in blobs:
        if blob.contains_xy(x, y):
            return blob
    if not blobs:
        return None
    return min(blobs, key=lambda blob: math.dist((float(y), float(x)), blob.centroid))


def object_hash(blob: ColorBlob) -> str:
    """REQ-ARC-FCP-5591: translation-invariant identity signature for a blob.

    The signature is the blob's color plus its cell-shape pattern, normalized
    so the top-left of its bounding box is the origin. Two blobs with the same
    shape and color hash identically regardless of WHERE they sit in the
    frame, so an object's identity can be tracked across frames even after it
    moves -- the position-INVARIANT feature ``tier``/``score`` (both keyed on
    a click's absolute (x, y) or a blob's absolute bbox) do not provide. This
    directly attacks the GAP-4891 / ``project_arc_live_agent_learning_gaps``
    binding constraint that frame-only, position-only (order-1) features sit
    at LOO=chance on held-out games.
    """

    cells = blob.cells
    min_y = min(y for y, _ in cells)
    min_x = min(x for _, x in cells)
    normalized = tuple(sorted((y - min_y, x - min_x) for y, x in cells))
    payload = repr((int(blob.color), normalized)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def blob_topology(frame: Any) -> dict[str, Any]:
    """REQ-ARC-FCP-5591: containment tree + adjacency graph over a frame's FULL
    (unfiltered) connected-component partition.

    Reuses ``connected_color_blobs`` unmodified (called here with
    ``min_pixels=1, max_component_fraction=1.0`` so every pixel belongs to
    exactly one returned blob -- the ``ColorBlobSaliencePrior`` tier/score
    methods intentionally filter blobs for RANKING purposes, but a
    containment/adjacency computation needs the complete partition to be
    correct). Blobs are already returned in top-left-cell order by
    ``connected_color_blobs``, which is a stable, unique ordering within one
    frame -- that list position is this function's blob ``id``.

    Containment: for each blob ``b``, flood-fill the grid's complement of
    ``b`` inward from the frame border; any blob whose cells are never
    reached is enclosed by ``b``. A blob's ``parent`` is its INNERMOST
    encloser (the encloser that is itself most deeply enclosed) -- this
    yields a clean nesting tree rather than every blob listing every
    ancestor. A single representative cell per blob is sufficient to test
    reachability because a blob's cells are one connected 4-adjacency region
    that is either fully reached or fully unreached together.

    Returns a dict:
      - ``blobs``: the full blob list, ``blobs[i]`` is blob id ``i``.
      - ``object_hashes``: ``{id: object_hash(blobs[id])}`` for convenience.
      - ``children``: ``{id: sorted [child ids directly enclosed by id]}``.
      - ``adjacency_list``: sorted ``[i, j]`` id pairs for blobs sharing a
        4-connected edge (includes parent/child pairs, since they physically
        touch).
    """

    grid = _as_grid(frame)
    height, width = grid.shape
    blobs = connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)
    n = len(blobs)

    comp_id = np.full((height, width), -1, dtype=np.int32)
    for blob_id, blob in enumerate(blobs):
        for y, x in blob.cells:
            comp_id[y, x] = blob_id

    adj_pairs: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            cid = int(comp_id[y, x])
            if y + 1 < height:
                other = int(comp_id[y + 1, x])
                if other != cid:
                    adj_pairs.add((min(cid, other), max(cid, other)))
            if x + 1 < width:
                other = int(comp_id[y, x + 1])
                if other != cid:
                    adj_pairs.add((min(cid, other), max(cid, other)))
    adjacency_list = sorted(adj_pairs)

    enclosers: list[set[int]] = [set() for _ in range(n)]
    for b in range(n):
        reached = np.zeros((height, width), dtype=bool)
        stack: list[tuple[int, int]] = []
        for y in range(height):
            for x in (0, width - 1):
                if int(comp_id[y, x]) != b and not reached[y, x]:
                    reached[y, x] = True
                    stack.append((y, x))
        for x in range(width):
            for y in (0, height - 1):
                if int(comp_id[y, x]) != b and not reached[y, x]:
                    reached[y, x] = True
                    stack.append((y, x))
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and not reached[ny, nx]
                    and int(comp_id[ny, nx]) != b
                ):
                    reached[ny, nx] = True
                    stack.append((ny, nx))
        for a in range(n):
            if a == b:
                continue
            ay, ax = next(iter(blobs[a].cells))
            if not reached[ay, ax]:
                enclosers[a].add(b)

    encloser_counts = {index: len(encs) for index, encs in enumerate(enclosers)}
    children: dict[int, list[int]] = {blob_id: [] for blob_id in range(n)}
    for a in range(n):
        if enclosers[a]:
            parent = max(enclosers[a], key=lambda e: (encloser_counts.get(e, 0), -e))
            children[parent].append(a)
    for child_ids in children.values():
        child_ids.sort()

    return {
        "blobs": blobs,
        "object_hashes": {blob_id: object_hash(blob) for blob_id, blob in enumerate(blobs)},
        "children": children,
        "adjacency_list": [[i, j] for i, j in adjacency_list],
    }
