"""Regression tests for the vectorized `object_centric_digest` re-implementation.

REQ-ARC-OCD-VECTORIZE-1 (2026-08-07): `object_centric_digest` was rewritten to use
`scipy.ndimage.label` per non-background color instead of a pure-Python per-cell
flood-fill, to address a measured cost (3.6s self-time over 2648 calls, ~1.4ms/call
in a live sp80 run -- the dominant per-step cost on the sp80/ft09 Kaggle gate-timeout
regression). Same technique already proven once in this codebase for the sibling
function `connected_color_blobs` (REQ-ARC-FCP-5699 item-2, 2026-07-16) -- this test
file mirrors tests/python/test_arc_color_blob_salience_vectorized.py's structure.

These tests verify the vectorized version is field-for-field equivalent to the
original pure-Python flood-fill. HONEST NOTE (adversarial review, 2026-08-07): the
oracle below is a close transcription of the pre-rewrite algorithm (same variable
names, same stack-based DFS), not an independently-derived re-implementation --
an earlier draft of this docstring overclaimed "from scratch... not a copy" and
was corrected. It still validly compares two GENUINELY DIFFERENT techniques (a
per-cell BFS vs. scipy's per-color labeling), so a bug specific to one technique
would still be caught; it is just not immune to a bug in the shared CONCEPTION of
what the algorithm should do (e.g. if both were wrong about background exclusion
in the same way, this test would not catch that).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_solver_kit import object_centric_digest


def _oracle_digest(grid: np.ndarray) -> list[tuple]:
    """A separate flood-fill implementation, deliberately NOT the module under test --
    it closely transcribes the ORIGINAL object_centric_digest algorithm (pre-2026-08-07)
    rather than being independently derived (see the module docstring's honest note),
    so treat this as "does the new technique agree with the old one" rather than "two
    unrelated authors agree", but a real comparison of two different techniques either
    way."""

    arr = np.asarray(grid)
    vals, counts = np.unique(arr, return_counts=True)
    background = int(vals[counts.argmax()]) if len(vals) else 0
    mask = arr != background
    seen = np.zeros_like(mask, dtype=bool)
    h, w = arr.shape
    out: list[tuple] = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            color = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not seen[ny, nx]
                        and int(arr[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            bbox = (min(ys), min(xs), max(ys), max(xs))
            area = len(cells)
            out.append((color, area, bbox, (sum(xs) / area, sum(ys) / area)))
    return sorted(out, key=lambda t: (-t[1], t[0], t[2]))


def _as_tuples(digest: dict) -> list[tuple]:
    return [
        (c["color"], c["area"], tuple(c["bbox"]), tuple(c["centroid"]))
        for c in digest["components"]
    ]


@pytest.mark.parametrize("seed", range(30))
def test_vectorized_matches_oracle_random_grids(seed: int) -> None:
    rng = np.random.RandomState(seed)
    h, w = rng.randint(1, 65), rng.randint(1, 65)
    n_colors = rng.randint(1, 17)
    grid = rng.randint(0, n_colors, size=(h, w)).astype(np.int16)

    got = _as_tuples(object_centric_digest(grid))
    want = _oracle_digest(grid)
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

    got = _as_tuples(object_centric_digest(grid))
    want = _oracle_digest(grid)
    assert got == want
    assert len(got) > 1  # sanity: this frame is not degenerate


def test_vectorized_handles_single_color_grid() -> None:
    """A grid with only the background color: no non-background components exist."""
    grid = np.full((10, 10), 3, dtype=np.int16)
    digest = object_centric_digest(grid)
    assert digest["components"] == []
    assert digest["background_color"] == 3


def test_vectorized_handles_1x1_grid() -> None:
    grid = np.array([[5]], dtype=np.int16)
    digest = object_centric_digest(grid)
    # A 1x1 grid's single pixel IS the most-common color -> it is background,
    # and background is always excluded -> zero components.
    assert digest["components"] == []
    assert digest["background_color"] == 5


def test_background_pixels_are_excluded_from_every_component() -> None:
    """The most-common color is background and must never appear as a component's
    own color, regardless of how many disjoint regions it forms."""
    rng = np.random.RandomState(11)
    grid = rng.randint(0, 5, size=(30, 30)).astype(np.int16)
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[counts.argmax()])

    digest = object_centric_digest(grid)
    assert digest["background_color"] == background
    assert all(c["color"] != background for c in digest["components"])


def test_component_count_and_shape_are_reported() -> None:
    grid = np.zeros((5, 5), dtype=np.int16)
    grid[1, 1] = 7
    grid[3, 3] = 7
    digest = object_centric_digest(grid)
    assert digest["shape"] == [5, 5]
    assert digest["component_count"] == len(digest["components"]) == 2


@pytest.mark.parametrize("seed", range(10))
def test_vectorized_matches_oracle_grid_fallback_enabled(seed: int) -> None:
    """emit_grid_fallback_for_background is untouched by this rewrite (it operates on
    the background mask, after the vectorized component-discovery loop) -- confirm
    the rewrite didn't accidentally change its behaviour either."""
    rng = np.random.RandomState(seed + 100)
    h, w = rng.randint(4, 40), rng.randint(4, 40)
    n_colors = rng.randint(1, 8)
    grid = rng.randint(0, n_colors, size=(h, w)).astype(np.int16)

    with_fallback = object_centric_digest(grid, emit_grid_fallback_for_background=True)
    without_fallback = object_centric_digest(grid, emit_grid_fallback_for_background=False)

    fallback_rows = [c for c in with_fallback["components"] if c.get("is_grid_fallback")]
    non_fallback_rows = [c for c in with_fallback["components"] if not c.get("is_grid_fallback")]
    assert _as_tuples({"components": non_fallback_rows}) == _as_tuples(without_fallback)
    # Every fallback row's color is the background color, by construction.
    assert all(c["color"] == with_fallback["background_color"] for c in fallback_rows)
