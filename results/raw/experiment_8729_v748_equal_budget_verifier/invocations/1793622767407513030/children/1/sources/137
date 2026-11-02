"""Grid-grounded object identity tracking for ARC rendered frames.

Spec refs: REQ-ARC-WMTE-4841,
SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE,
SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    """Deterministic thresholds for the prototype tracker."""

    min_pixels: int = 1
    max_component_fraction: float = 0.45
    max_match_distance: float = 18.0
    min_match_score: float = 0.45
    recovery_margin: float = 0.10
    positive_control_min_persistence: float = 0.35


@dataclass(frozen=True)
class Component:
    """One connected same-color component in a rendered grid."""

    component_id: int
    color: int
    pixel_count: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cells: frozenset[tuple[int, int]]
    normalized_cells: frozenset[tuple[int, int]]
    shape_key: str


@dataclass(frozen=True)
class TrackedObject:
    """A component plus its persistent identity assigned by shape/motion matching."""

    track_id: int
    frame_index: int
    component: Component

    @property
    def centroid(self) -> tuple[float, float]:
        return self.component.centroid

    @property
    def pixel_count(self) -> int:
        return self.component.pixel_count

    @property
    def shape_key(self) -> str:
        return self.component.shape_key


@dataclass(frozen=True)
class TrackedSequence:
    """Tracked objects grouped both by frame and by stable track id."""

    frames: list[list[TrackedObject]]
    tracks: dict[int, list[TrackedObject]]


@dataclass(frozen=True)
class CorrespondenceMeasurement:
    """Quantitative tracker-vs-baseline correspondence on one frame sequence."""

    shape_motion_score: float
    color_centroid_baseline_score: float
    n_frames: int
    n_transition_pairs: int
    eligible_objects: int
    shape_motion_matches: int
    color_centroid_matches: int
    recovered: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape_motion_score": self.shape_motion_score,
            "color_centroid_baseline_score": self.color_centroid_baseline_score,
            "n_frames": self.n_frames,
            "n_transition_pairs": self.n_transition_pairs,
            "eligible_objects": self.eligible_objects,
            "shape_motion_matches": self.shape_motion_matches,
            "color_centroid_matches": self.color_centroid_matches,
            "recovered": self.recovered,
        }


def _as_grid(frame: Any) -> np.ndarray:
    arr = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def _shape_key(normalized_cells: Iterable[tuple[int, int]], height: int, width: int) -> str:
    cells = set(normalized_cells)
    rows = []
    for y in range(height):
        rows.append("".join("1" if (y, x) in cells else "0" for x in range(width)))
    return f"{width}:{height}:{'/'.join(rows)}"


def segment_frame(frame: Any, config: TrackerConfig | None = None) -> list[Component]:
    """Segment one rendered grid into same-color connected components.

    Color is allowed to define component boundaries, but it is not used as the
    identity key by the shape/motion tracker. Very large components are treated
    as rendered background/board fill and excluded from object correspondence.
    """

    cfg = config or TrackerConfig()
    grid = _as_grid(frame)
    h, w = grid.shape
    max_pixels = int(max(1, math.floor(h * w * cfg.max_component_fraction)))
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for y0 in range(h):
        for x0 in range(w):
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
                        0 <= ny < h
                        and 0 <= nx < w
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            pixel_count = len(cells)
            if pixel_count < cfg.min_pixels or pixel_count > max_pixels:
                continue
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            ymin, ymax = min(ys), max(ys)
            xmin, xmax = min(xs), max(xs)
            normalized = frozenset((y - ymin, x - xmin) for y, x in cells)
            height = ymax - ymin + 1
            width = xmax - xmin + 1
            component = Component(
                component_id=len(components),
                color=color,
                pixel_count=pixel_count,
                bbox=(ymin, xmin, ymax, xmax),
                centroid=(float(sum(ys)) / pixel_count, float(sum(xs)) / pixel_count),
                cells=frozenset(cells),
                normalized_cells=normalized,
                shape_key=_shape_key(normalized, height, width),
            )
            components.append(component)

    return sorted(components, key=lambda item: (item.bbox, item.color, item.pixel_count))


def _centroid_distance(left: Component, right: Component) -> float:
    return math.dist(left.centroid, right.centroid)


def _shape_similarity(left: Component, right: Component) -> float:
    if left.normalized_cells == right.normalized_cells:
        return 1.0
    union = left.normalized_cells | right.normalized_cells
    if not union:
        return 0.0
    return len(left.normalized_cells & right.normalized_cells) / len(union)


def _overlap_similarity(left: Component, right: Component) -> float:
    union = left.cells | right.cells
    if not union:
        return 0.0
    return len(left.cells & right.cells) / len(union)


def _size_similarity(left: Component, right: Component) -> float:
    bigger = max(left.pixel_count, right.pixel_count)
    if bigger <= 0:
        return 0.0
    return min(left.pixel_count, right.pixel_count) / bigger


def _motion_similarity(left: Component, right: Component, config: TrackerConfig) -> float:
    return max(0.0, 1.0 - (_centroid_distance(left, right) / config.max_match_distance))


def _shape_motion_score(left: Component, right: Component, config: TrackerConfig) -> float:
    return (
        0.45 * _shape_similarity(left, right)
        + 0.25 * _overlap_similarity(left, right)
        + 0.20 * _motion_similarity(left, right, config)
        + 0.10 * _size_similarity(left, right)
    )


def _greedy_match(
    previous: list[Component],
    current: list[Component],
    config: TrackerConfig,
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(previous):
        for right_index, right in enumerate(current):
            score = _shape_motion_score(left, right, config)
            if score >= config.min_match_score:
                candidates.append((score, left_index, right_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: dict[int, int] = {}
    for _score, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches[left_index] = right_index
    return matches


def _color_centroid_match(
    previous: list[Component],
    current: list[Component],
    config: TrackerConfig,
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(previous):
        for right_index, right in enumerate(current):
            if left.color != right.color:
                continue
            distance = _centroid_distance(left, right)
            if distance > config.max_match_distance:
                continue
            size = _size_similarity(left, right)
            if size < 0.25:
                continue
            score = (1.0 - (distance / config.max_match_distance)) + size
            candidates.append((score, left_index, right_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: dict[int, int] = {}
    for _score, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches[left_index] = right_index
    return matches


def _active_component_indices(
    previous: list[Component],
    before: np.ndarray,
    after: np.ndarray,
) -> list[int]:
    changed = before != after
    if not bool(changed.any()):
        return []
    changed_cells = set(zip(*np.where(changed), strict=False))
    active = [
        index
        for index, component in enumerate(previous)
        if component.cells & changed_cells
    ]
    return active


def track_object_identities(
    frames: Iterable[Any],
    config: TrackerConfig | None = None,
) -> TrackedSequence:
    """Assign stable track ids to segmented objects over a frame sequence."""

    cfg = config or TrackerConfig()
    grids = [_as_grid(frame) for frame in frames]
    if not grids:
        return TrackedSequence(frames=[], tracks={})

    segmented = [segment_frame(grid, cfg) for grid in grids]
    next_track_id = 1
    tracked_frames: list[list[TrackedObject]] = []

    first_frame: list[TrackedObject] = []
    previous_ids: dict[int, int] = {}
    for index, component in enumerate(segmented[0]):
        previous_ids[index] = next_track_id
        first_frame.append(TrackedObject(next_track_id, 0, component))
        next_track_id += 1
    tracked_frames.append(first_frame)

    for frame_index in range(1, len(segmented)):
        previous = segmented[frame_index - 1]
        current = segmented[frame_index]
        matches = _greedy_match(previous, current, cfg)
        current_ids: dict[int, int] = {}
        for prev_index, curr_index in matches.items():
            current_ids[curr_index] = previous_ids[prev_index]
        frame_objects: list[TrackedObject] = []
        for curr_index, component in enumerate(current):
            track_id = current_ids.get(curr_index)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
            frame_objects.append(TrackedObject(track_id, frame_index, component))
        previous_ids = {index: obj.track_id for index, obj in enumerate(frame_objects)}
        tracked_frames.append(frame_objects)

    tracks: dict[int, list[TrackedObject]] = {}
    for frame_objects in tracked_frames:
        for obj in frame_objects:
            tracks.setdefault(obj.track_id, []).append(obj)
    return TrackedSequence(frames=tracked_frames, tracks=tracks)


def measure_correspondence(
    frames: Iterable[Any],
    config: TrackerConfig | None = None,
) -> CorrespondenceMeasurement:
    """Measure active-object correspondence against a color-centroid baseline."""

    cfg = config or TrackerConfig()
    grids = [_as_grid(frame) for frame in frames]
    if len(grids) < 2:
        return CorrespondenceMeasurement(0.0, 0.0, len(grids), 0, 0, 0, 0, False)

    eligible = 0
    shape_matches = 0
    color_matches = 0
    pairs = 0
    for before, after in zip(grids, grids[1:], strict=False):
        previous = segment_frame(before, cfg)
        current = segment_frame(after, cfg)
        active_indices = _active_component_indices(previous, before, after)
        if not active_indices:
            continue
        pairs += 1
        eligible += len(active_indices)
        shape = _greedy_match(previous, current, cfg)
        color = _color_centroid_match(previous, current, cfg)
        shape_matches += sum(1 for index in active_indices if index in shape)
        color_matches += sum(1 for index in active_indices if index in color)

    shape_score = shape_matches / eligible if eligible else 0.0
    color_score = color_matches / eligible if eligible else 0.0
    recovered = bool(eligible and shape_score >= color_score + cfg.recovery_margin)
    return CorrespondenceMeasurement(
        shape_motion_score=round(shape_score, 6),
        color_centroid_baseline_score=round(color_score, 6),
        n_frames=len(grids),
        n_transition_pairs=pairs,
        eligible_objects=eligible,
        shape_motion_matches=shape_matches,
        color_centroid_matches=color_matches,
        recovered=recovered,
    )


def object_relational_features(objects: Iterable[TrackedObject]) -> list[dict[str, Any]]:
    """Expose simple object-relational offsets for later goal-grounding consumers."""

    frame_objects = list(objects)
    rows: list[dict[str, Any]] = []
    for obj in frame_objects:
        nearest: TrackedObject | None = None
        nearest_distance = float("inf")
        for other in frame_objects:
            if other.track_id == obj.track_id:
                continue
            distance = math.dist(obj.centroid, other.centroid)
            if distance < nearest_distance:
                nearest = other
                nearest_distance = distance
        overlaps = [
            other.track_id
            for other in frame_objects
            if other.track_id != obj.track_id
            and bool(obj.component.cells & other.component.cells)
        ]
        if nearest is None:
            offset = [0.0, 0.0]
            nearest_id = None
        else:
            offset = [
                round(nearest.centroid[0] - obj.centroid[0], 6),
                round(nearest.centroid[1] - obj.centroid[1], 6),
            ]
            nearest_id = nearest.track_id
        rows.append(
            {
                "object_id": obj.track_id,
                "shape_key": obj.shape_key,
                "pixel_count": obj.pixel_count,
                "centroid": [round(obj.centroid[0], 6), round(obj.centroid[1], 6)],
                "nearest_object_id": nearest_id,
                "offset_to_nearest": offset,
                "overlaps_object_ids": sorted(overlaps),
            }
        )
    return rows


def track_summary(tracked: TrackedSequence) -> dict[int, dict[str, float | int]]:
    """Summarize track persistence and motion for controls and artifacts."""

    n_frames = len(tracked.frames)
    out: dict[int, dict[str, float | int]] = {}
    for track_id, objects in tracked.tracks.items():
        ordered = sorted(objects, key=lambda obj: obj.frame_index)
        motion = 0.0
        for left, right in zip(ordered, ordered[1:], strict=False):
            motion += math.dist(left.centroid, right.centroid)
        out[track_id] = {
            "frames_present": len({obj.frame_index for obj in ordered}),
            "persistence": round((len({obj.frame_index for obj in ordered}) / n_frames), 6)
            if n_frames
            else 0.0,
            "total_motion": round(motion, 6),
            "mean_pixels": round(
                sum(obj.pixel_count for obj in ordered) / len(ordered), 6
            ),
        }
    return out


def config_fingerprint(config: TrackerConfig | None = None) -> dict[str, Any]:
    """Stable representation of tracker parameters for artifact checksums."""

    cfg = config or TrackerConfig()
    return {
        "min_pixels": cfg.min_pixels,
        "max_component_fraction": cfg.max_component_fraction,
        "max_match_distance": cfg.max_match_distance,
        "min_match_score": cfg.min_match_score,
        "recovery_margin": cfg.recovery_margin,
        "positive_control_min_persistence": cfg.positive_control_min_persistence,
    }
