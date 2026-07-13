"""Tests for arc_color_blob_salience.object_hash / blob_topology.

Spec refs: REQ-ARC-FCP-5591, SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY,
SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_color_blob_salience import (
    blob_topology,
    connected_color_blobs,
    object_hash,
)


class _Frame:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame


def test_req_arc_fcp_5591_object_hash_matches_across_translated_twins() -> None:
    """SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY: same shape+color -> same hash."""

    grid = np.zeros((10, 10), dtype=np.int16)
    grid[1:4, 1:4] = 5
    grid[6:9, 6:9] = 5
    blobs = connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)
    fives = [b for b in blobs if b.color == 5]

    assert len(fives) == 2
    assert object_hash(fives[0]) == object_hash(fives[1])


def test_req_arc_fcp_5591_object_hash_differs_by_shape_and_by_color() -> None:
    """SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY: different shape or color -> different hash."""

    grid = np.zeros((10, 10), dtype=np.int16)
    grid[1:4, 1:4] = 5  # 3x3 square, color 5
    grid[6:8, 6:9] = 5  # 2x3 rectangle, color 5 -- different shape, same color
    grid[1:4, 6:9] = 7  # 3x3 square, color 7 -- same shape, different color
    blobs = connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)

    square5 = next(b for b in blobs if b.color == 5 and b.height == 3 and b.width == 3)
    rect5 = next(b for b in blobs if b.color == 5 and b.height == 2 and b.width == 3)
    square7 = next(b for b in blobs if b.color == 7)

    assert object_hash(square5) != object_hash(rect5)
    assert object_hash(square5) != object_hash(square7)


def test_req_arc_fcp_5591_blob_topology_containment_and_adjacency() -> None:
    """SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY: ring encloses inner blob, they are adjacent."""

    grid = np.zeros((10, 10), dtype=np.int16)
    grid[1:6, 1:6] = 3  # outer ring color
    grid[2:5, 2:5] = 5  # inner blob, fully enclosed by the ring
    frame = _Frame(grid)

    topo = blob_topology(frame)
    blobs = topo["blobs"]
    ring_id = next(i for i, b in enumerate(blobs) if b.color == 3)
    inner_id = next(i for i, b in enumerate(blobs) if b.color == 5)

    assert inner_id in topo["children"][ring_id]
    assert [ring_id, inner_id] in topo["adjacency_list"] or [inner_id, ring_id] in [
        [b, a] for a, b in topo["adjacency_list"]
    ]


def test_req_arc_fcp_5591_blob_topology_distinguishes_enclosed_from_free_twin() -> None:
    """SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY: only the enclosed twin gets a parent;
    both twins still share the same object_hash despite differing topological position."""

    grid = np.zeros((10, 10), dtype=np.int16)
    grid[1:6, 1:6] = 3  # enclosing ring
    grid[2:5, 2:5] = 5  # enclosed twin
    grid[7:10, 7:10] = 5  # free-standing twin, same shape and color, not enclosed
    frame = _Frame(grid)

    topo = blob_topology(frame)
    blobs = topo["blobs"]
    ring_id = next(i for i, b in enumerate(blobs) if b.color == 3)
    fives = [i for i, b in enumerate(blobs) if b.color == 5]
    assert len(fives) == 2

    enclosed_id = next(i for i in fives if i in topo["children"][ring_id])
    free_id = next(i for i in fives if i != enclosed_id)

    all_children = [child for kids in topo["children"].values() for child in kids]
    assert free_id not in all_children
    assert topo["object_hashes"][enclosed_id] == topo["object_hashes"][free_id]


def test_req_arc_fcp_5591_blob_topology_full_partition_covers_every_pixel() -> None:
    """REQ-ARC-FCP-5591: blob_topology uses the FULL unfiltered partition -- every pixel
    belongs to exactly one returned blob, unlike ColorBlobSaliencePrior's filtered tiers."""

    grid = np.zeros((6, 6), dtype=np.int16)
    grid[0:2, 0:2] = 1
    grid[3:5, 3:5] = 2
    frame = _Frame(grid)

    topo = blob_topology(frame)
    total_pixels = sum(b.pixel_count for b in topo["blobs"])
    assert total_pixels == grid.size
