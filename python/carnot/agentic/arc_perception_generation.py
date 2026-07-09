"""Classical ARC perception pass for runtime candidate generation.

Spec refs: REQ-ARC-FCP-5508, SCENARIO-ARC-FCP-5508.

This helper keeps the ARC live path honest by using only what the agent can see
at runtime: rendered frames, the action it just took, and the next rendered
frame. It extracts simple connected-component and color-blob structure, overlays
that structure with observed motion, and turns the result into click/action
affordances that the existing `E3AgentPolicy` can consume as both an
`action_prior` and a short-prefix generator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carnot.agentic.arc_color_blob_salience import (
    ColorBlobSaliencePrior,
    connected_color_blobs,
    _as_grid,
)


REQUIRED_PERCEPTION_FEATURES = (
    "connected_components",
    "color_blobs",
    "sprite_overlays",
    "salient_motion",
    "action_affordances",
)
TRAJECTORY_TAXONOMY_KEYS = (
    "factual",
    "referential",
    "logical",
    "procedural",
    "scope_based",
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _level(frame: Any) -> int:
    for name in ("levels_completed", "level_progress", "level"):
        value = getattr(frame, name, None)
        if value is not None:
            return _as_int(value)
    return 0


def _candidate_action(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        return _as_int(candidate.get("action", candidate.get("action_id", 0)))
    return _as_int(getattr(candidate, "action", getattr(candidate, "action_id", 0)))


def _candidate_data(candidate: Any) -> Mapping[str, Any]:
    data = candidate.get("data") if isinstance(candidate, Mapping) else getattr(candidate, "data", None)
    return data if isinstance(data, Mapping) else {}


def _point(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["x"]), int(row["y"])


def _changed_mask(before: Any | None, after: Any) -> tuple[np.ndarray, bool]:
    if before is None:
        return np.zeros_like(_as_grid(after), dtype=bool), False
    lhs = _as_grid(before)
    rhs = _as_grid(after)
    if lhs.shape != rhs.shape:
        return np.zeros_like(rhs, dtype=bool), True
    return lhs != rhs, False


def _overlap_count(mask: np.ndarray, bbox: Sequence[int]) -> int:
    if mask.size == 0:
        return 0
    y0, x0, y1, x1 = [int(value) for value in bbox]
    clipped = mask[max(0, y0) : min(mask.shape[0], y1 + 1), max(0, x0) : min(mask.shape[1], x1 + 1)]
    return int(np.count_nonzero(clipped))


def _taxonomy_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in TRAJECTORY_TAXONOMY_KEYS}
    for row in rows:
        failure = str(row.get("failure_kind") or "")
        if failure in counts:
            counts[failure] += 1
    return counts


@dataclass
class ClassicalPerceptionGenerator:
    """Runtime-grounded action prior and prefix generator for ARC frames."""

    base_prior: ColorBlobSaliencePrior = field(default_factory=ColorBlobSaliencePrior)
    max_candidates: int = 8
    keyboard_score: float = 0.0
    motion_bonus: float = 750.0
    _last_features: dict[str, Any] = field(default_factory=dict, init=False)
    _taxonomy_steps: list[dict[str, Any]] = field(default_factory=list, init=False)
    _candidate_generation_count: int = field(default=0, init=False)

    def for_path(self, _path: Sequence[Mapping[str, Any]]) -> "ClassicalPerceptionGenerator":
        """Return the same stateful generator for a live frontier path."""

        return self

    @property
    def perception_features_enabled(self) -> list[str]:
        return list(REQUIRED_PERCEPTION_FEATURES)

    def inspect(self, frame: Any, *, previous_frame: Any | None = None) -> dict[str, Any]:
        """Extract connected components, overlays, motion, and action affordances."""

        grid = _as_grid(frame)
        mask, shape_mismatch = _changed_mask(previous_frame, frame)
        component_rows = self._component_rows(frame)
        blob_rows = self.base_prior.tier_rows(frame)
        sprite_rows = self._sprite_overlay_rows(blob_rows, mask)
        motion_rows = self._motion_rows(mask, shape_mismatch=shape_mismatch)
        action_rows = self._action_affordance_rows(blob_rows, motion_rows, grid)
        self._last_features = {
            "connected_component_rows": component_rows,
            "color_blob_rows": blob_rows,
            "sprite_overlay_rows": sprite_rows,
            "motion_affordance_rows": motion_rows,
            "action_affordance_rows": action_rows,
            "shape_mismatch": bool(shape_mismatch),
        }
        self._candidate_generation_count += 1
        return dict(self._last_features)

    def click_points(self, frame: Any, *, max_points: int | None = None) -> list[tuple[int, int]]:
        """Return generated click points in perception-salience order."""

        limit = self.max_candidates if max_points is None else int(max_points)
        if not self._last_features:
            self.inspect(frame)
        rows = self._last_features.get("action_affordance_rows", [])
        points: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for row in rows:
            point = _point(row)
            if point in seen:
                continue
            seen.add(point)
            points.append(point)
            if len(points) >= limit:
                break
        return points

    def tier_rows(self, frame: Any) -> list[dict[str, Any]]:
        return self.base_prior.tier_rows(frame)

    def action_tier_rows(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        """Return candidate diagnostics sorted in the order this prior prefers."""

        self.inspect(frame)
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            action = _candidate_action(candidate)
            data = _candidate_data(candidate)
            row = {
                "index": int(index),
                "action": int(action),
                "data": dict(data),
                "score": float(self.score(frame, candidate)),
                "source": str(candidate.get("source", "") if isinstance(candidate, Mapping) else ""),
            }
            rows.append(row)
        return sorted(rows, key=lambda row: (-float(row["score"]), int(row["index"])))

    def score(self, frame: Any, candidate: Any) -> float:
        """Score a live candidate; higher scores are consumed earlier."""

        action = _candidate_action(candidate)
        if action != 6:
            return float(self.keyboard_score)
        data = _candidate_data(candidate)
        if "x" not in data or "y" not in data:
            return 0.0
        base = float(self.base_prior.score(frame, candidate))
        motion = 0.0
        for row in self._last_features.get("motion_affordance_rows", []):
            distance = abs(int(data["x"]) - int(row["x"])) + abs(int(data["y"]) - int(row["y"]))
            motion = max(motion, max(0.0, 1.0 - float(distance) / 20.0))
        return float(base + self.motion_bonus * motion)

    def best_sequence(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
        min_len: int = 2,
    ) -> tuple[dict[str, Any], ...]:
        """Generate a short prefix from the current perception-ranked candidates."""

        del goal_energy, action_effect_scorer
        ordered = self.action_tier_rows(frame, candidates)
        sequence: list[dict[str, Any]] = []
        for row in ordered:
            sequence.append({"action": int(row["action"]), "data": dict(row["data"]) or None})
            if len(sequence) >= int(min_len):
                break
        return tuple(sequence)

    def observe_transition(self, before: Any, action: int, data: Any, after: Any) -> None:
        """Record one action effect and classify any trajectory failure mode."""

        mask, shape_mismatch = _changed_mask(before, after)
        changed = int(np.count_nonzero(mask))
        level_before = _level(before)
        level_after = _level(after)
        failure = ""
        if shape_mismatch:
            failure = "procedural"
        elif int(action) == 6 and not (isinstance(data, Mapping) and "x" in data and "y" in data):
            failure = "referential"
        elif changed == 0 and level_after <= level_before:
            failure = "factual"
        elif changed > 0 and level_after <= level_before:
            failure = "logical"
        self._taxonomy_steps.append(
            {
                "step": int(len(self._taxonomy_steps) + 1),
                "action": int(action),
                "data": dict(data) if isinstance(data, Mapping) else data,
                "changed_cells": int(changed),
                "level_before": int(level_before),
                "level_after": int(level_after),
                "failure_kind": failure,
            }
        )

    def reset(self, *, level: int | None = None, reset_to_prior: bool = True) -> None:
        """Clear transient feature context on an intentional live-agent reset."""

        del level
        if reset_to_prior:
            self._last_features = {}

    def record_scope_failure(self, reason: str) -> None:
        self._taxonomy_steps.append(
            {
                "step": int(len(self._taxonomy_steps) + 1),
                "failure_kind": "scope_based",
                "reason": str(reason),
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": "classical_connected_component_color_blob_perception_generation",
            "perception_features_enabled": self.perception_features_enabled,
            "runtime_observation_steps": int(
                len([row for row in self._taxonomy_steps if row.get("action") is not None])
            ),
            "candidate_generation_count": int(self._candidate_generation_count),
            "last_feature_counts": {
                "connected_components": len(self._last_features.get("connected_component_rows", [])),
                "color_blobs": len(self._last_features.get("color_blob_rows", [])),
                "sprite_overlays": len(self._last_features.get("sprite_overlay_rows", [])),
                "salient_motion": len(self._last_features.get("motion_affordance_rows", [])),
                "action_affordances": len(self._last_features.get("action_affordance_rows", [])),
            },
            "trajectory_taxonomy_counts": _taxonomy_counts(self._taxonomy_steps),
            "trajectory_taxonomy_steps": [dict(row) for row in self._taxonomy_steps],
        }

    def as_dict(self) -> dict[str, Any]:
        diagnostics = self.diagnostics()
        diagnostics["generation_stage_action_prioritization"] = True
        return diagnostics

    def _component_rows(self, frame: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, blob in enumerate(connected_color_blobs(frame, max_component_fraction=1.0)):
            rows.append(
                {
                    "component_id": int(index),
                    "color": int(blob.color),
                    "pixel_count": int(blob.pixel_count),
                    "bbox": [int(value) for value in blob.bbox],
                    "centroid_y": float(blob.centroid[0]),
                    "centroid_x": float(blob.centroid[1]),
                }
            )
        return rows

    def _sprite_overlay_rows(
        self,
        blob_rows: Sequence[Mapping[str, Any]],
        mask: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(blob_rows):
            if row.get("status_bar"):
                continue
            overlap = _overlap_count(mask, row.get("bbox", [0, 0, 0, 0]))
            if row.get("button_like") or overlap > 0:
                rows.append(
                    {
                        "sprite_id": int(index),
                        "overlay_kind": "motion_overlay" if overlap > 0 else "blob_sprite_candidate",
                        "color": int(row.get("color", 0)),
                        "bbox": [int(value) for value in row.get("bbox", [0, 0, 0, 0])],
                        "x": int(row.get("centroid_x", 0)),
                        "y": int(row.get("centroid_y", 0)),
                        "motion_overlap": int(overlap),
                        "tier": row.get("tier"),
                    }
                )
        return rows

    def _motion_rows(self, mask: np.ndarray, *, shape_mismatch: bool) -> list[dict[str, Any]]:
        if shape_mismatch or not np.any(mask):
            return []
        motion_grid = mask.astype(np.int16)
        rows: list[dict[str, Any]] = []
        for index, blob in enumerate(connected_color_blobs(motion_grid, max_component_fraction=1.0)):
            if int(blob.color) != 1:
                continue
            rows.append(
                {
                    "motion_id": int(index),
                    "x": int(blob.centroid[1]),
                    "y": int(blob.centroid[0]),
                    "changed_pixels": int(blob.pixel_count),
                    "bbox": [int(value) for value in blob.bbox],
                }
            )
        return rows

    def _action_affordance_rows(
        self,
        blob_rows: Sequence[Mapping[str, Any]],
        motion_rows: Sequence[Mapping[str, Any]],
        grid: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        color_counts = Counter(int(value) for value in grid.flatten().tolist())
        for row in motion_rows:
            point = _point(row)
            if point not in seen:
                seen.add(point)
                rows.append(
                    {
                        "action": 6,
                        "data": {"x": point[0], "y": point[1]},
                        "x": point[0],
                        "y": point[1],
                        "source": "salient_motion",
                        "score": 5000.0 + float(row.get("changed_pixels", 0)),
                    }
                )
        for row in blob_rows:
            if row.get("status_bar") or row.get("large_flat"):
                continue
            point = (int(row.get("centroid_x", 0)), int(row.get("centroid_y", 0)))
            if point in seen:
                continue
            color = int(row.get("color", 0))
            rarity = 1.0 / float(1 + color_counts.get(color, 0))
            score = 4000.0 - float(row.get("tier", 4)) * 100.0 + rarity
            seen.add(point)
            rows.append(
                {
                    "action": 6,
                    "data": {"x": point[0], "y": point[1]},
                    "x": point[0],
                    "y": point[1],
                    "source": "color_blob_sprite_overlay",
                    "score": float(score),
                }
            )
        rows.sort(key=lambda row: (-float(row["score"]), int(row["y"]), int(row["x"])))
        return rows[: max(1, int(self.max_candidates))]
