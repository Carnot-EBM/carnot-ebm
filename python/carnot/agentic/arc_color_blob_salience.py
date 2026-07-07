"""Classical color-blob salience for ARC live exploration.

Spec refs: REQ-ARC-FCP-5360, SCENARIO-ARC-FCP-5360,
REQ-ARC-FCP-5373, SCENARIO-ARC-FCP-5373.

The live ARC agent already has learned frame-diff and action-prior hooks. This
module adds the cheap perception-grounded fallback those hooks were missing:
segment the rendered frame into single-color connected components and rank
button-like blobs before large dull regions and status-bar artifacts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
            "salient_colors": sorted(int(color) for color in self.salient_colors),
            "status_bar_color": int(self.status_bar_color),
            "max_component_fraction": float(self.max_component_fraction),
            "status_bar_deprioritization": bool(self.status_bar_deprioritization),
            "large_flat_deprioritization": bool(self.large_flat_deprioritization),
        }
