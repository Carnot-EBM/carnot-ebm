"""Bounded MAP-style landmark prestage for ARC graph exploration.

This is the small Exp 5198 prototype of MAP's task-specific cognitive map:
before the solver search runs, spend a fixed novelty-exploration budget from
the same offline state, record reachable regions and action effects, and expose
replayable landmark trajectories that `graph_explore_solve_v2` can try before
flat primitive expansion. It is not an oracle: seeded trajectories still consume
the normal expansion budget and any banked level must pass reproduce().
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time

import numpy as np

from carnot.agentic.arc_agi3_live_adapter import _game_action, _game_over, _levels_completed
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of as default_grid_of
from carnot.agentic.arc_graph_explore import _warm, rich_action_candidates


def _grid2d(frame: Any, grid_of: Callable[[Any], Any]) -> Optional[np.ndarray]:
    try:
        grid = np.asarray(grid_of(frame))
    except Exception:
        return None
    if grid.ndim == 3 and grid.shape[0] == 1:
        grid = grid[0]
    return grid if grid.ndim == 2 else None


def _data_key(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, dict):
        return tuple(sorted((str(k), _data_key(v)) for k, v in data.items()))
    if isinstance(data, list):
        return tuple(_data_key(v) for v in data)
    if isinstance(data, (str, int, float, bool)):
        return data
    return repr(data)


def _action_key(action: Any, data: Any) -> tuple[int, Any]:
    return (int(action), _data_key(data))


def _changed_bbox(g0: np.ndarray, g1: np.ndarray) -> list[int] | None:
    if g0.shape != g1.shape:
        return None
    changed = g0 != g1
    if not bool(changed.any()):
        return None
    rows, cols = np.where(changed)
    return [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())]


def _touches_region(g0: np.ndarray, g1: np.ndarray, region: np.ndarray | None) -> bool:
    if region is None or g0.shape != g1.shape or g0.shape != region.shape:
        return False
    return bool(((g0 != g1) & region).any())


def _energy(goal_energy: Callable[[Any], float] | None, frame: Any) -> float | None:
    if goal_energy is None:
        return None
    try:
        return float(goal_energy(frame))
    except Exception:
        return None


def _clean_step(action: int, data: Any) -> dict[str, Any]:
    return {"action": int(action), "data": data}


def _serialise_action_key(key: tuple[int, Any]) -> str:
    return f"{key[0]}:{repr(key[1])}"


@dataclass
class ArcLandmarkMap:
    """Task-specific map facts plus replayable seed trajectories for graph search."""

    root_hash: str
    grid_of: Callable[[Any], Any] = default_grid_of
    map_overhead_steps: int = 0
    map_overhead_wall_s: float = 0.0
    reachable_regions: list[dict[str, Any]] = field(default_factory=list)
    effect_deltas: dict[tuple[int, Any], dict[str, Any]] = field(default_factory=dict)
    relational_landmarks: list[dict[str, Any]] = field(default_factory=list)
    seed_sequences: list[list[dict[str, Any]]] = field(default_factory=list)
    max_sequences_per_node: int = 12

    verifier_is_oracle = False

    @property
    def reachable_region_count(self) -> int:
        return len(self.reachable_regions)

    def frontier_seed_sequences(
        self,
        frame: Any,
        candidates: list[Any],
        *,
        path: list[dict[str, Any]] | None = None,
        root_path_length: int = 0,
        goal_energy: Callable[[Any], float] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Return replayable seed trajectories for the matching root frontier node."""
        grid = _grid2d(frame, self.grid_of)
        if grid is None or frame_hash(grid) != self.root_hash:
            return []
        candidate_keys = {_action_key(c.action_id, c.data) for c in candidates if hasattr(c, "action_id")}
        out: list[list[dict[str, Any]]] = []
        for sequence in self.seed_sequences:
            if not sequence:
                continue
            first = sequence[0]
            if candidate_keys and _action_key(first["action"], first.get("data")) not in candidate_keys:
                continue
            out.append([dict(step) for step in sequence])
            if len(out) >= self.max_sequences_per_node:
                break
        return out

    def diagnostics(self) -> dict[str, Any]:
        return {
            "reachable_region_count": self.reachable_region_count,
            "effect_action_classes": len(self.effect_deltas),
            "relational_landmarks": len(self.relational_landmarks),
            "seed_sequences": len(self.seed_sequences),
            "map_overhead_steps": int(self.map_overhead_steps),
            "map_overhead_wall_s": round(float(self.map_overhead_wall_s), 4),
            "verifier_is_oracle": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.diagnostics(),
            "reachable_regions": list(self.reachable_regions[:32]),
            "effect_deltas": {
                _serialise_action_key(key): dict(value)
                for key, value in sorted(self.effect_deltas.items(), key=lambda item: repr(item[0]))
            },
            "relational_landmarks": list(self.relational_landmarks[:32]),
            "seed_sequences": [list(sequence) for sequence in self.seed_sequences[:16]],
        }


def _rank_seed_landmarks(landmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        energy_after = row.get("energy_after")
        energy_value = float(energy_after) if energy_after is not None else 1e9
        energy_drop = float(row.get("energy_drop") or 0.0)
        return (
            0 if row.get("leveled_up") else 1,
            0 if row.get("target_touch") else 1,
            energy_value,
            len(row.get("sequence") or []),
            -energy_drop,
        )

    return sorted(landmarks, key=key)


def _unique_sequences(landmarks: list[dict[str, Any]], max_sequences: int) -> list[list[dict[str, Any]]]:
    seen: set[tuple[tuple[int, Any], ...]] = set()
    out: list[list[dict[str, Any]]] = []
    for row in _rank_seed_landmarks(landmarks):
        sequence = [dict(step) for step in row.get("sequence") or []]
        signature = tuple(_action_key(step["action"], step.get("data")) for step in sequence)
        if not sequence or signature in seen:
            continue
        seen.add(signature)
        out.append(sequence)
        if len(out) >= max_sequences:
            break
    return out


def build_landmark_map(
    env: Any,
    *,
    start_level: int,
    prefix: list[dict[str, Any]] | None = None,
    max_steps: int = 750,
    max_depth: int = 20,
    warmup: bool = False,
    grid_of: Callable[[Any], Any] = default_grid_of,
    goal_energy: Callable[[Any], float] | None = None,
    target_region: np.ndarray | None = None,
    max_seed_sequences: int = 16,
) -> ArcLandmarkMap:
    """Build a bounded novelty map from the post-prefix state."""
    from arcengine import GameAction

    started = time.time()
    prefix = list(prefix or [])
    budget = max(1, int(max_steps))
    target = None if target_region is None else np.asarray(target_region, dtype=bool)

    def replay(suffix: list[dict[str, Any]]):
        frame = _warm(env, warmup)
        for step in [*prefix, *suffix]:
            frame = env.step(
                _game_action(GameAction, int(step["action"])),
                data=step.get("data"),
                reasoning={"policy": "map_landmark_prestage_replay"},
            )
        return frame

    root_frame = replay([])
    root_grid = _grid2d(root_frame, grid_of)
    root_hash = frame_hash(root_grid) if root_grid is not None else ""
    reachable_regions = [
        {
            "hash": root_hash,
            "depth": 0,
            "grid_shape": list(root_grid.shape) if root_grid is not None else [],
            "source": "post_prefix_root",
        }
    ]
    effect_deltas: dict[tuple[int, Any], dict[str, Any]] = {}
    landmarks: list[dict[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {
        root_hash: {
            "path": [],
            "frame": root_frame,
            "untested": rich_action_candidates(root_frame),
        }
    }
    frontier: deque[str] = deque([root_hash])
    steps = 0

    while frontier and steps < budget:
        node_hash = frontier[0]
        node = nodes[node_hash]
        if not node["untested"] or len(node["path"]) >= int(max_depth):
            frontier.popleft()
            continue
        frame_before = replay(node["path"])
        grid_before = _grid2d(frame_before, grid_of)
        if grid_before is None:
            frontier.popleft()
            continue
        before_energy = _energy(goal_energy, frame_before)
        candidate = node["untested"].pop(0)
        step = _clean_step(candidate.action_id, candidate.data)
        frame_after = env.step(
            _game_action(GameAction, int(candidate.action_id)),
            data=candidate.data,
            reasoning={"policy": "map_landmark_prestage_novelty_probe"},
        )
        steps += 1
        if frame_after is None:
            continue
        grid_after = _grid2d(frame_after, grid_of)
        if grid_after is None:
            continue
        after_energy = _energy(goal_energy, frame_after)
        changed = grid_before != grid_after if grid_before.shape == grid_after.shape else np.ones((), dtype=bool)
        changed_cells = int(changed.sum()) if hasattr(changed, "sum") else 0
        target_touch = _touches_region(grid_before, grid_after, target)
        leveled_up = _levels_completed(frame_after) > int(start_level)
        action_key = _action_key(candidate.action_id, candidate.data)
        tally = effect_deltas.setdefault(
            action_key,
            {
                "observations": 0,
                "changed_cells_total": 0,
                "changed_cells_min": None,
                "changed_cells_max": 0,
                "target_touches": 0,
                "levelups": 0,
                "new_state_hits": 0,
                "max_energy_drop": 0.0,
            },
        )
        tally["observations"] += 1
        tally["changed_cells_total"] += changed_cells
        tally["changed_cells_max"] = max(int(tally["changed_cells_max"]), changed_cells)
        current_min = tally["changed_cells_min"]
        tally["changed_cells_min"] = changed_cells if current_min is None else min(int(current_min), changed_cells)
        if target_touch:
            tally["target_touches"] += 1
        if leveled_up:
            tally["levelups"] += 1
        if before_energy is not None and after_energy is not None:
            tally["max_energy_drop"] = max(float(tally["max_energy_drop"]), float(before_energy - after_energy))

        next_path = [*node["path"], step]
        next_hash = frame_hash(grid_after)
        if next_hash not in nodes and not _game_over(frame_after):
            tally["new_state_hits"] += 1
            nodes[next_hash] = {
                "path": next_path,
                "frame": frame_after,
                "untested": rich_action_candidates(frame_after, previous_frame=frame_before),
            }
            frontier.append(next_hash)
            reachable_regions.append(
                {
                    "hash": next_hash,
                    "depth": len(next_path),
                    "changed_bbox": _changed_bbox(grid_before, grid_after),
                    "changed_cells": changed_cells,
                    "target_touch": target_touch,
                }
            )

        energy_drop = (
            float(before_energy - after_energy)
            if before_energy is not None and after_energy is not None
            else 0.0
        )
        if target_touch or energy_drop > 0.0 or leveled_up:
            landmarks.append(
                {
                    "source_hash": node_hash,
                    "target_hash": next_hash,
                    "sequence": next_path,
                    "action": int(candidate.action_id),
                    "data": candidate.data,
                    "depth": len(next_path),
                    "changed_bbox": _changed_bbox(grid_before, grid_after),
                    "changed_cells": changed_cells,
                    "target_touch": target_touch,
                    "leveled_up": leveled_up,
                    "energy_before": before_energy,
                    "energy_after": after_energy,
                    "energy_drop": energy_drop,
                }
            )

    return ArcLandmarkMap(
        root_hash=root_hash,
        grid_of=grid_of,
        map_overhead_steps=steps,
        map_overhead_wall_s=time.time() - started,
        reachable_regions=reachable_regions,
        effect_deltas=effect_deltas,
        relational_landmarks=landmarks,
        seed_sequences=_unique_sequences(landmarks, int(max_seed_sequences)),
    )
