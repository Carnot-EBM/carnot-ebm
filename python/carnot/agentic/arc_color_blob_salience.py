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

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np


SALIENT_COLORS = frozenset(range(6, 16))
STATUS_BAR_COLOR = 16


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
    """Return same-color components while suppressing huge background fills."""

    grid = _as_grid(frame)
    height, width = grid.shape
    max_pixels = int(max(1, math.floor(height * width * float(max_component_fraction))))
    seen = np.zeros((height, width), dtype=bool)
    blobs: list[ColorBlob] = []
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0]:
                continue
            color = int(grid[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = y + dy
                    nx = x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < int(min_pixels) or len(cells) > max_pixels:
                continue
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            blobs.append(
                ColorBlob(
                    color=color,
                    pixel_count=len(cells),
                    bbox=(min(ys), min(xs), max(ys), max(xs)),
                    centroid=(float(sum(ys)) / len(cells), float(sum(xs)) / len(cells)),
                    cells=frozenset(cells),
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
                        "score": float(self.score(frame, candidate)),
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
                    "score": float(self.score(frame, candidate)),
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

    def score(self, frame: Any, candidate: Any) -> float:
        """Score a live candidate; higher values are tried earlier."""

        if self._candidate_action_id(candidate) != 6:
            return float(self.keyboard_score)
        data = self._candidate_data(candidate)
        if "x" not in data or "y" not in data:
            return 0.0
        grid = _as_grid(frame)
        x = int(data["x"])
        y = int(data["y"])
        blobs = connected_color_blobs(
            grid,
            min_pixels=self.min_pixels,
            max_component_fraction=1.0
            if self.large_flat_deprioritization
            else self.max_component_fraction,
        )
        blob = self._blob_for_click(blobs, x, y)
        if blob is None:
            return 0.0
        color_counts = Counter(int(value) for value in grid.flatten().tolist())
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
