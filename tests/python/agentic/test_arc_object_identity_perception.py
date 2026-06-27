"""Tests for the Exp 4841 object-identity perception prototype.

Spec refs: REQ-ARC-WMTE-4841,
SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE,
SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_object_identity_perception import (
    TrackerConfig,
    measure_correspondence,
    object_relational_features,
    segment_frame,
    track_object_identities,
)


def _grid_with_recolored_object(color: int, *, x_offset: int = 0) -> np.ndarray:
    grid = np.zeros((12, 12), dtype=np.int16)
    grid[2:5, 2 + x_offset : 5 + x_offset] = color
    grid[8, 8:11] = 3
    return grid


def test_req_arc_wmte_4841_segments_connected_components_without_global_color_identity() -> None:
    """REQ-ARC-WMTE-4841: segmentation exposes shape/connectivity object slots."""

    components = segment_frame(_grid_with_recolored_object(2), TrackerConfig(min_pixels=1))

    shapes = sorted((component.pixel_count, component.bbox) for component in components)
    assert shapes == [(3, (8, 8, 8, 10)), (9, (2, 2, 4, 4))]
    assert {component.shape_key for component in components} == {
        "3:1:111",
        "3:3:111/111/111",
    }


def test_scenario_arc_wmte_4841_shape_motion_beats_color_centroid_on_recolor() -> None:
    """SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE: recolor-invariant matching is measured."""

    frames = [
        _grid_with_recolored_object(2, x_offset=0),
        _grid_with_recolored_object(7, x_offset=1),
        _grid_with_recolored_object(9, x_offset=2),
    ]

    measurement = measure_correspondence(frames, TrackerConfig(min_pixels=1))

    assert measurement.n_frames == 3
    assert measurement.shape_motion_score == 1.0
    assert measurement.color_centroid_baseline_score < 0.5
    assert measurement.recovered is True


def test_scenario_arc_wmte_4841_tracks_support_object_relational_features() -> None:
    """REQ-ARC-WMTE-4841: stable tracks expose object-relational offsets."""

    frames = [
        _grid_with_recolored_object(2, x_offset=0),
        _grid_with_recolored_object(7, x_offset=1),
    ]
    tracked = track_object_identities(frames, TrackerConfig(min_pixels=1))

    assert len(tracked.frames) == 2
    assert {obj.track_id for obj in tracked.frames[0]} == {obj.track_id for obj in tracked.frames[1]}

    relations = object_relational_features(tracked.frames[1])
    assert relations
    assert all("nearest_object_id" in row for row in relations)
    assert any(row["offset_to_nearest"] != [0.0, 0.0] for row in relations)
