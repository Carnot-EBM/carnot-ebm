"""Regression tests for the vectorized `connected_color_blobs` re-implementation.

REQ-ARC-FCP-5699 item-2 re-validation (2026-07-16): `connected_color_blobs` was
rewritten to use `scipy.ndimage.label` per color instead of a pure-Python per-cell
flood-fill, to address the measured slowdown (lp85 budget=500 took 68s vs baseline's
~7761 actions/115s) that led to `SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED = False`.
These tests verify the vectorized version is field-for-field equivalent to the
original pure-Python flood-fill, using a fresh from-scratch oracle re-implementation
(not a copy of the shipped code) so the test cannot pass by sharing a bug with the
implementation it is checking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carnot.agentic.arc_color_blob_salience import connected_color_blobs


def _oracle_blobs(
    grid: np.ndarray, *, min_pixels: int = 1, max_component_fraction: float = 0.45
) -> list[tuple]:
    """Independent pure-Python flood-fill oracle -- deliberately NOT the module under
    test, so an equivalence pass means real behavioral agreement, not a shared bug."""

    height, width = grid.shape
    max_pixels = int(max(1, math.floor(height * width * float(max_component_fraction))))
    seen = np.zeros((height, width), dtype=bool)
    out: list[tuple] = []
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
                    ny, nx = y + dy, x + dx
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
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            out.append(
                (
                    color,
                    len(cells),
                    (min(ys), min(xs), max(ys), max(xs)),
                    (float(sum(ys)) / len(cells), float(sum(xs)) / len(cells)),
                    frozenset(cells),
                    (height, width),
                )
            )
    return sorted(out, key=lambda t: (t[2], t[0], t[1]))


def _as_tuples(blobs) -> list[tuple]:
    return [(b.color, b.pixel_count, b.bbox, b.centroid, b.cells, b.frame_shape) for b in blobs]


@pytest.mark.parametrize("seed", range(30))
def test_vectorized_matches_oracle_random_grids(seed: int) -> None:
    rng = np.random.RandomState(seed)
    h, w = rng.randint(1, 65), rng.randint(1, 65)
    n_colors = rng.randint(1, 17)
    grid = rng.randint(0, n_colors, size=(h, w)).astype(np.int16)
    min_pixels = int(rng.choice([1, 1, 2, 3]))
    max_frac = float(rng.choice([0.45, 1.0, 0.1, 0.9]))

    got = _as_tuples(
        connected_color_blobs(grid, min_pixels=min_pixels, max_component_fraction=max_frac)
    )
    want = _oracle_blobs(grid, min_pixels=min_pixels, max_component_fraction=max_frac)
    assert got == want


def test_vectorized_matches_oracle_realistic_structured_frame() -> None:
    """A realistic ARC-like frame (background + a handful of rectangular objects +
    a status bar), the regime that actually matters for live-agent performance --
    not uniform random noise, which real ARC games do not render."""

    grid = np.zeros((64, 64), dtype=np.int16)
    rng = np.random.RandomState(7)
    for _ in range(15):
        y0, x0 = rng.randint(0, 55, size=2)
        h, w = rng.randint(2, 8, size=2)
        color = rng.randint(1, 12)
        grid[y0 : y0 + h, x0 : x0 + w] = color
    grid[0:2, :] = 16

    got = _as_tuples(connected_color_blobs(grid))
    want = _oracle_blobs(grid)
    assert got == want
    assert len(got) > 1  # sanity: this frame is not degenerate


def test_vectorized_handles_single_color_grid() -> None:
    grid = np.full((10, 10), 3, dtype=np.int16)
    got = _as_tuples(connected_color_blobs(grid, max_component_fraction=1.0))
    want = _oracle_blobs(grid, max_component_fraction=1.0)
    assert got == want
    assert len(got) == 1


def test_vectorized_handles_1x1_grid() -> None:
    grid = np.array([[5]], dtype=np.int16)
    got = _as_tuples(connected_color_blobs(grid, max_component_fraction=1.0))
    want = _oracle_blobs(grid, max_component_fraction=1.0)
    assert got == want


def test_vectorized_min_pixels_filters_singletons() -> None:
    rng = np.random.RandomState(3)
    grid = rng.randint(0, 8, size=(20, 20)).astype(np.int16)
    got = _as_tuples(connected_color_blobs(grid, min_pixels=3, max_component_fraction=1.0))
    want = _oracle_blobs(grid, min_pixels=3, max_component_fraction=1.0)
    assert got == want
    assert all(pixel_count >= 3 for _, pixel_count, *_ in got)
