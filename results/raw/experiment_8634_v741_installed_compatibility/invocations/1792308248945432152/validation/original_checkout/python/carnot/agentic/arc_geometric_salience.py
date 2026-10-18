"""Geometry-aware ARC action salience for live exploration.

Spec refs: REQ-ARC-FCP-5385, SCENARIO-ARC-FCP-5385.

The prior is intentionally small and frame-only. It starts from the existing
color-blob salience score, then adds a geometric tie-break from the agent's own
observed before/after transitions. That keeps the signal useful for hidden
games: the agent only needs rendered frames, the action it just took, and the
next rendered frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior, _as_grid


@dataclass
class GeometricSaliencePrior:
    """Live action prior that adds geodesic and hyperbolic geometry to blob salience."""

    base_prior: ColorBlobSaliencePrior = field(default_factory=ColorBlobSaliencePrior)
    geodesic_weight: float = 250.0
    hyperbolic_weight: float = 25.0
    max_transition_anchors: int = 8
    enabled: bool = True
    source: str = "geometric_geodesic_blob_salience"
    _transition_anchors: list[tuple[float, float]] = field(default_factory=list, init=False)

    @property
    def observed_transition_count(self) -> int:
        return len(self._transition_anchors)

    @property
    def hyperbolic_or_geodesic_ranking_enabled(self) -> bool:
        return bool(self.enabled)

    def for_path(self, _path: list[Mapping[str, Any]]) -> "GeometricSaliencePrior":
        """Return the live prior for a graph path without cloning transition memory."""

        return self

    def observe_transition(
        self,
        before: Any,
        _action_id: int,
        _data: Mapping[str, Any] | None,
        after: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        """Add a geodesic anchor from the agent's own observed frame change."""

        anchor = self._changed_centroid(before, after)
        if anchor is None:
            return
        self._transition_anchors.append(anchor)
        keep = max(1, int(self.max_transition_anchors))
        if len(self._transition_anchors) > keep:
            del self._transition_anchors[: len(self._transition_anchors) - keep]

    def reset(self, *_args: Any, reset_to_prior: bool = False, **_kwargs: Any) -> None:
        """Clear transition anchors when the live policy intentionally resets levels."""

        if reset_to_prior:
            self._transition_anchors.clear()

    def score(self, frame: Any, candidate: Any) -> float:
        """Score one live candidate, with higher values tried earlier."""

        base = float(self.base_prior.score(frame, candidate))
        if not self.enabled or self._candidate_action_id(candidate) != 6:
            return base
        data = self._candidate_data(candidate)
        if "x" not in data or "y" not in data:
            return base
        try:
            grid = _as_grid(frame)
        except Exception:
            return base
        height, width = int(grid.shape[0]), int(grid.shape[1])
        y = float(int(data["y"]))
        x = float(int(data["x"]))
        geodesic = self._geodesic_proximity(y, x, height, width)
        hyperbolic = self._hyperbolic_centrality(y, x, height, width)
        return float(
            base + float(self.geodesic_weight) * geodesic + self.hyperbolic_weight * hyperbolic
        )

    def diagnostics(self) -> dict[str, Any]:
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "base_prior": self.base_prior.as_dict(),
            "hyperbolic_or_geodesic_ranking_enabled": bool(
                self.hyperbolic_or_geodesic_ranking_enabled
            ),
            "geodesic_weight": float(self.geodesic_weight),
            "hyperbolic_weight": float(self.hyperbolic_weight),
            "geodesic_anchor_count": int(len(self._transition_anchors)),
            "transition_anchors": [[float(y), float(x)] for y, x in self._transition_anchors],
        }

    @staticmethod
    def _candidate_action_id(candidate: Any) -> int:
        value = (
            candidate.get("action", candidate.get("action_id", 0))
            if isinstance(candidate, Mapping)
            else getattr(candidate, "action_id", 0)
        )
        return int(value or 0)

    @staticmethod
    def _candidate_data(candidate: Any) -> Mapping[str, Any]:
        data = (
            candidate.get("data")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "data", None)
        )
        return data if isinstance(data, Mapping) else {}

    @staticmethod
    def _changed_centroid(before: Any, after: Any) -> tuple[float, float] | None:
        try:
            lhs = _as_grid(before)
            rhs = _as_grid(after)
        except Exception:
            return None
        if lhs.shape != rhs.shape:
            return None
        changed = np.argwhere(lhs != rhs)
        if changed.size == 0:
            return None
        return float(np.mean(changed[:, 0])), float(np.mean(changed[:, 1]))

    def _geodesic_proximity(self, y: float, x: float, height: int, width: int) -> float:
        if not self._transition_anchors:
            return 0.0
        max_dist = max(1.0, float(max(0, height - 1) + max(0, width - 1)))
        distance = min(abs(y - ay) + abs(x - ax) for ay, ax in self._transition_anchors)
        return float(max(0.0, 1.0 - min(1.0, distance / max_dist)))

    @staticmethod
    def _hyperbolic_centrality(y: float, x: float, height: int, width: int) -> float:
        if height <= 1 or width <= 1:
            return 0.0
        ny = (2.0 * y / float(height - 1)) - 1.0
        nx = (2.0 * x / float(width - 1)) - 1.0
        radius = min(0.999999, math.sqrt(nx * nx + ny * ny) / math.sqrt(2.0))
        distance = math.log((1.0 + radius) / max(1e-12, 1.0 - radius))
        return float(1.0 / (1.0 + distance))
