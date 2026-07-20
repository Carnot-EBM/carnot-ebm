"""Game-blind causal primitives for the ARC live E3 path.

The module holds small, inspectable live components distilled from Exp5740's
causal audit. They are not game solvers. A primitive may rank or prune live
candidate actions only from the current agent's visible before/action/after
receipts, and it must tolerate hidden games where no public-game identity or
source code is available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np


SOURCE_LEAK_KEYS = frozenset(
    {
        "source_file",
        "source_rule",
        "game_source",
        "solution_code",
        "hidden_state",
        "per_game_adapter",
        "adapter_label",
        "outer_loop_bfs",
        "hand_authored_model",
    }
)
GAME_IDENTITY_KEYS = frozenset(
    {"game", "game_id", "game_name", "source_game", "registry_game", "registry_provenance"}
)


def _candidate_action(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        value = candidate.get("action", candidate.get("action_id", 0))
    else:
        value = getattr(candidate, "action", getattr(candidate, "action_id", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_data(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get("data")
    return getattr(candidate, "data", None)


def _action_signature(action: int, data: Any) -> tuple[Any, ...]:
    if int(action) == 6 and isinstance(data, Mapping):
        return (6, int(data.get("x", -1)), int(data.get("y", -1)))
    return (int(action),)


def _frame_grid(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    try:
        from carnot.agentic.arc_agi3_world_model import grid_of

        return grid_of(frame)
    except Exception:
        return None


def _state_signature(frame: Any) -> str:
    grid = _frame_grid(frame)
    if grid is None:
        return "unknown"
    try:
        from carnot.agentic.arc_agi3_world_model import frame_hash

        return frame_hash(grid)
    except Exception:
        shape = tuple(getattr(grid, "shape", ()))
        total = int(np.asarray(grid).sum()) if getattr(grid, "size", 0) else 0
        return f"shape={shape};sum={total}"


def _frames_equal(before: Any, after: Any) -> bool:
    left = _frame_grid(before)
    right = _frame_grid(after)
    if left is None or right is None:
        return False
    return bool(np.array_equal(left, right))


@dataclass
class BoundaryCollisionPrimitive:
    """Live ranker for Exp5740's `boundary_or_collision` primitive.

    The primitive watches for visible no-change transitions at the same visible
    state/action signature. Once an action has produced no visible effect from
    that state, the candidate remains available but is ranked after candidates
    without that no-effect receipt. This keeps the behavior game-blind: no
    walls, maps, coordinates from source, or per-game rules are imported.
    """

    primitive_id: str = "boundary_or_collision"
    no_effect_threshold: int = 1
    _no_effect_counts: dict[tuple[str, tuple[Any, ...]], int] = field(default_factory=dict)
    _receipts: list[dict[str, Any]] = field(default_factory=list)

    def game_blind_receipt(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        detected_source = 0
        detected_identity = 0
        for row in rows:
            keys = set(row)
            if keys & SOURCE_LEAK_KEYS:
                detected_source += 1
            if keys & GAME_IDENTITY_KEYS:
                detected_identity += 1
        return {
            "primitive_id": self.primitive_id,
            "learner_visible_rows": len(list(rows)),
            "detected_source_leak_canary_count": detected_source,
            "detected_game_identity_leak_canary_count": detected_identity,
            "admitted_source_leak_count": 0,
            "admitted_game_identity_leak_count": 0,
            "source_keys_rejected": sorted(SOURCE_LEAK_KEYS),
            "game_identity_keys_rejected": sorted(GAME_IDENTITY_KEYS),
        }

    def observe_transition(
        self,
        before: Any,
        action: int,
        data: Any,
        after: Any,
        *,
        leveled_up: bool = False,
    ) -> dict[str, Any]:
        no_visible_change = _frames_equal(before, after) and not bool(leveled_up)
        state = _state_signature(before)
        action_key = _action_signature(int(action), data)
        key = (state, action_key)
        if no_visible_change:
            self._no_effect_counts[key] = self._no_effect_counts.get(key, 0) + 1
        receipt = {
            "primitive_id": self.primitive_id,
            "state_signature": state,
            "action_signature": list(action_key),
            "no_visible_change": bool(no_visible_change),
            "count": int(self._no_effect_counts.get(key, 0)),
            "rank_after_threshold": bool(
                self._no_effect_counts.get(key, 0) >= self.no_effect_threshold
            ),
        }
        self._receipts.append(receipt)
        return receipt

    def rank_candidates(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        state = _state_signature(frame)
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            action = _candidate_action(candidate)
            data = _candidate_data(candidate)
            action_key = _action_signature(action, data)
            count = self._no_effect_counts.get((state, action_key), 0)
            if isinstance(candidate, Mapping):
                row = dict(candidate)
                row.setdefault("action", action)
                row.setdefault("data", data)
            else:
                row = {"action": action, "data": data}
            row["_generic_causal_primitive_order"] = index
            row["_generic_causal_primitive_no_effect_count"] = count
            rows.append(row)
        rows.sort(
            key=lambda row: (
                int(row.get("_generic_causal_primitive_no_effect_count", 0))
                >= self.no_effect_threshold,
                int(row.get("_generic_causal_primitive_order", 0)),
            )
        )
        for row in rows:
            row.pop("_generic_causal_primitive_order", None)
        return rows

    def diagnostics(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "blocked_signature_count": sum(
                1 for count in self._no_effect_counts.values() if count >= self.no_effect_threshold
            ),
            "observation_receipt_count": len(self._receipts),
            "receipts_hash_input": json.dumps(self._receipts, sort_keys=True, default=str),
        }


def coerce_generic_causal_primitive(value: Any) -> Any | None:
    if value in (None, False):
        return None
    if value is True or value == "boundary_or_collision":
        return BoundaryCollisionPrimitive()
    return value
