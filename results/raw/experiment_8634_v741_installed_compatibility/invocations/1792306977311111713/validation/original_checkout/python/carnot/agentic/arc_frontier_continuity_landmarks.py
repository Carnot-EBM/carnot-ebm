"""Runtime-log frontier continuity and landmark seeds for ARC graph search.

Spec refs: REQ-REPORT-5216, SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS.

The objects here are deliberately observation-only. They accept frames and
action paths already collected by the agent/runtime environment, then expose
replayable seed sequences through `graph_explore_solve_v2`'s existing
`frontier_seed_bank` hook. They do not inspect game source and they do not run
an exhaustive ground-truth search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of as default_grid_of


ActionStep = dict[str, Any]


def _grid2d(frame: Any, grid_of: Callable[[Any], Any]) -> np.ndarray | None:
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
    if isinstance(data, Mapping):
        return tuple(sorted((str(k), _data_key(v)) for k, v in data.items()))
    if isinstance(data, list | tuple):
        return tuple(_data_key(v) for v in data)
    if isinstance(data, str | int | float | bool):
        return data
    return repr(data)


def _action_key(action: Any, data: Any = None) -> tuple[int, Any]:
    if isinstance(action, Mapping):
        return int(action.get("action", 0) or 0), _data_key(action.get("data"))
    return int(action), _data_key(data)


def _clean_step(step: Mapping[str, Any] | Any) -> ActionStep:
    if isinstance(step, Mapping):
        return {"action": int(step.get("action", 0) or 0), "data": step.get("data")}
    return {"action": int(step), "data": None}


def _clean_path(path: Sequence[Any] | None) -> list[ActionStep]:
    return [_clean_step(step) for step in list(path or []) if step is not None]


def _path_signature(path: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, Any], ...]:
    return tuple(_action_key(step) for step in path)


def _path_suffix(path: Sequence[ActionStep], prefix: Sequence[ActionStep]) -> list[ActionStep]:
    path_rows = _clean_path(path)
    prefix_rows = _clean_path(prefix)
    if _path_signature(path_rows[: len(prefix_rows)]) == _path_signature(prefix_rows):
        return [dict(step) for step in path_rows[len(prefix_rows) :]]
    return [dict(step) for step in path_rows]


def _structural_signature(grid: np.ndarray) -> tuple[Any, ...]:
    vals, counts = np.unique(grid, return_counts=True)
    palette = tuple(sorted((int(v), int(c)) for v, c in zip(vals.tolist(), counts.tolist())))
    nonzero = int(np.count_nonzero(grid))
    return (tuple(int(v) for v in grid.shape), len(vals), nonzero, palette[:16])


def _compatible_signature(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if not left or not right:
        return False
    if left[0] != right[0]:
        return False
    # Exact palette matches are strongest. If palette differs, still allow a
    # same-shape/same-palette-size match so a later level can reuse the same
    # action structure when colors or object positions drift.
    return left[3] == right[3] or left[1] == right[1]


def _energy(goal_energy: Callable[[Any], float] | None, frame: Any) -> float | None:
    if goal_energy is None:
        return None
    try:
        return float(goal_energy(frame))
    except Exception:
        return None


def _level(row: Mapping[str, Any], key: str, frame_key: str) -> int:
    if key in row:
        return int(row.get(key) or 0)
    frame = row.get(frame_key)
    return int(getattr(frame, "levels_completed", 0) or 0)


def _candidate_keys(candidates: Sequence[Any]) -> set[tuple[int, Any]]:
    keys: set[tuple[int, Any]] = set()
    for candidate in candidates:
        if hasattr(candidate, "action_id"):
            keys.add(_action_key(candidate.action_id, getattr(candidate, "data", None)))
        elif isinstance(candidate, Mapping):
            keys.add(_action_key(candidate))
    return keys


def _first_action_allowed(sequence: Sequence[ActionStep], candidates: Sequence[Any]) -> bool:
    if not sequence:
        return False
    keys = _candidate_keys(candidates)
    return not keys or _action_key(sequence[0]) in keys


def _unique_sequences(sequences: Sequence[Sequence[ActionStep]], limit: int) -> list[list[ActionStep]]:
    seen: set[tuple[tuple[int, Any], ...]] = set()
    out: list[list[ActionStep]] = []
    for sequence in sequences:
        rows = [dict(step) for step in sequence if step.get("action") is not None]
        signature = _path_signature(rows)
        if not rows or signature in seen:
            continue
        seen.add(signature)
        out.append(rows)
        if len(out) >= int(limit):
            break
    return out


@dataclass(frozen=True)
class LandmarkSeed:
    """A two-stage seed: first reach a landmark, then continue toward a level-up."""

    landmark_hash: str
    landmark_signature: tuple[Any, ...]
    reach_sequence: list[ActionStep]
    goal_sequence: list[ActionStep]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "landmark_hash": self.landmark_hash,
            "reach_len": len(self.reach_sequence),
            "goal_len": len(self.goal_sequence),
            "score": round(float(self.score), 4),
        }


@dataclass
class FrontierContinuityLandmarkBank:
    """Seed-bank consumed by `graph_explore_solve_v2(frontier_seed_bank=...)`."""

    root_signature: tuple[Any, ...]
    root_hash: str
    continuity_sequences: list[list[ActionStep]] = field(default_factory=list)
    landmarks: list[LandmarkSeed] = field(default_factory=list)
    grid_of: Callable[[Any], Any] = default_grid_of
    max_sequences_per_node: int = 4
    enable_frontier_continuity: bool = True
    enable_landmark_decomposition: bool = True

    verifier_is_oracle = False

    def _frame_facts(self, frame: Any) -> tuple[str, tuple[Any, ...]] | None:
        grid = _grid2d(frame, self.grid_of)
        if grid is None:
            return None
        return frame_hash(grid), _structural_signature(grid)

    def frontier_seed_sequences(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        path: list[ActionStep] | None = None,
        root_path_length: int = 0,
        goal_energy: Callable[[Any], float] | None = None,
    ) -> list[list[ActionStep]]:
        """Return replayable sequences for compatible root or landmark states."""

        facts = self._frame_facts(frame)
        if facts is None:
            return []
        current_hash, current_signature = facts
        path_len = len(path or [])
        sequences: list[list[ActionStep]] = []

        if (
            path_len <= int(root_path_length)
            and _compatible_signature(current_signature, self.root_signature)
        ):
            if self.enable_frontier_continuity:
                sequences.extend(self.continuity_sequences)
            if self.enable_landmark_decomposition:
                sequences.extend(seed.reach_sequence for seed in self.landmarks)

        if self.enable_landmark_decomposition:
            for seed in self.landmarks:
                if current_hash == seed.landmark_hash and seed.goal_sequence:
                    sequences.append(seed.goal_sequence)

        allowed = [
            sequence for sequence in sequences if _first_action_allowed(sequence, list(candidates))
        ]
        return _unique_sequences(allowed, self.max_sequences_per_node)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "root_hash": self.root_hash,
            "continuity_sequence_count": len(self.continuity_sequences),
            "landmark_count": len(self.landmarks),
            "max_sequences_per_node": int(self.max_sequences_per_node),
            "enable_frontier_continuity": bool(self.enable_frontier_continuity),
            "enable_landmark_decomposition": bool(self.enable_landmark_decomposition),
            "verifier_is_oracle": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.diagnostics(),
            "continuity_sequences": [
                [dict(step) for step in sequence] for sequence in self.continuity_sequences[:8]
            ],
            "landmarks": [seed.as_dict() for seed in self.landmarks[:8]],
        }


def build_frontier_continuity_landmark_bank(
    *,
    root_frame: Any,
    transition_logs: Sequence[Mapping[str, Any]],
    grid_of: Callable[[Any], Any] = default_grid_of,
    goal_energy: Callable[[Any], float] | None = None,
    max_sequences: int = 8,
    max_sequence_len: int = 16,
    enable_frontier_continuity: bool = True,
    enable_landmark_decomposition: bool = True,
) -> FrontierContinuityLandmarkBank:
    """Build a seed bank from runtime transition logs only."""

    root_grid = _grid2d(root_frame, grid_of)
    root_hash = frame_hash(root_grid) if root_grid is not None else ""
    root_signature = _structural_signature(root_grid) if root_grid is not None else ()

    levelup_paths: list[list[ActionStep]] = []
    for row in transition_logs:
        after_level = _level(row, "level_after", "frame_after")
        before_level = _level(row, "level_before", "frame_before")
        if after_level > before_level:
            levelup_paths.append(_clean_path(row.get("path_after")))

    continuity: list[list[ActionStep]] = []
    landmark_candidates: list[LandmarkSeed] = []
    for row in transition_logs:
        before = row.get("frame_before")
        after = row.get("frame_after")
        before_grid = _grid2d(before, grid_of)
        after_grid = _grid2d(after, grid_of)
        if before_grid is None or after_grid is None:
            continue
        before_signature = _structural_signature(before_grid)
        path_before = _clean_path(row.get("path_before"))
        path_after = _clean_path(row.get("path_after"))

        if _compatible_signature(root_signature, before_signature):
            for full_path in levelup_paths:
                suffix = _path_suffix(full_path, path_before)
                if 0 < len(suffix) <= int(max_sequence_len):
                    continuity.append(suffix)

        before_energy = _energy(goal_energy, before)
        after_energy = _energy(goal_energy, after)
        energy_drop = (
            max(0.0, before_energy - after_energy)
            if before_energy is not None and after_energy is not None
            else 0.0
        )
        changed = (
            int(np.count_nonzero(before_grid != after_grid))
            if before_grid.shape == after_grid.shape
            else 0
        )
        if not enable_landmark_decomposition or (energy_drop <= 0.0 and changed <= 0):
            continue
        reach = _path_suffix(path_after, path_before)
        if not reach or len(reach) > int(max_sequence_len):
            continue
        for full_path in levelup_paths:
            if _path_signature(full_path[: len(path_after)]) != _path_signature(path_after):
                continue
            goal = _path_suffix(full_path, path_after)
            if not goal or len(goal) > int(max_sequence_len):
                continue
            score = float(energy_drop) + 0.01 * float(changed)
            landmark_candidates.append(
                LandmarkSeed(
                    landmark_hash=frame_hash(after_grid),
                    landmark_signature=_structural_signature(after_grid),
                    reach_sequence=reach,
                    goal_sequence=goal,
                    score=score,
                )
            )
            break

    landmark_candidates.sort(key=lambda seed: (-seed.score, len(seed.reach_sequence)))
    return FrontierContinuityLandmarkBank(
        root_signature=root_signature,
        root_hash=root_hash,
        continuity_sequences=_unique_sequences(continuity, max_sequences),
        landmarks=landmark_candidates[: int(max_sequences)],
        grid_of=grid_of,
        max_sequences_per_node=max(1, int(max_sequences)),
        enable_frontier_continuity=bool(enable_frontier_continuity),
        enable_landmark_decomposition=bool(enable_landmark_decomposition),
    )
