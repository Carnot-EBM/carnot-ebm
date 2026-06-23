"""Dense online curiosity for ARC live exploration.

Spec refs: REQ-CAPSTONE-4628, SCENARIO-CAPSTONE-4628.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.agentic.arc_live_ttt import LiveTTTWorldModel, action_key


@dataclass(frozen=True)
class CuriosityEvent:
    """One online transition's reducible prediction-error improvement."""

    origin_hash: str
    next_hash: str
    action_key: tuple
    before_error: float
    after_error: float
    raw_progress: float
    aleatoric_estimate: float
    bonus: float


def _per_cell_error(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape:  # pragma: no cover - defensive live guard.
        return 1.0
    return float(np.mean(predicted != target))


class DenseCuriosityProgress:
    """Curiosity-Critic style progress from the live world model's own error field.

    The reward is not raw surprise. It is the drop in per-cell prediction error
    after the online model observes a transition, minus an aleatoric estimate for
    repeated state/action keys that produce conflicting next states.
    """

    def __init__(
        self,
        game: str = "?",
        *,
        bonus_weight: float = 0.15,
        backup_discount: float = 0.5,
        noise_weight: float = 1.0,
        aleatoric_decay: float = 0.75,
    ) -> None:
        self.game = str(game)
        self.bonus_weight = float(bonus_weight)
        self.backup_discount = float(backup_discount)
        self.noise_weight = float(noise_weight)
        self.aleatoric_decay = float(aleatoric_decay)
        self._world_model = LiveTTTWorldModel(self.game, refit_every=1, min_transitions=1)
        self._seen_next: dict[tuple[bytes, tuple], set[bytes]] = {}
        self._aleatoric: dict[tuple[bytes, tuple], float] = {}
        self._state_values: dict[str, float] = {}
        self._edge_values: dict[tuple[str, str], float] = {}
        self._events = 0
        self._conflicts = 0

    def observe_transition(
        self,
        origin_hash: str,
        next_hash: str,
        grid: Any,
        action: int,
        data: Any,
        next_grid: Any,
        *,
        level_before: int = 0,
        level_after: int = 0,
    ) -> CuriosityEvent:
        """Observe one live transition and return its dense progress event."""

        grid_arr = np.asarray(grid)
        next_arr = np.asarray(next_grid)
        akey = action_key(int(action), data)
        key = (grid_arr.tobytes(), akey)

        before_pred = np.asarray(self._world_model.engine(grid_arr, int(action), data))
        before_error = _per_cell_error(before_pred, next_arr)

        seen_next = self._seen_next.setdefault(key, set())
        next_digest = next_arr.tobytes()
        conflict = 1.0 if seen_next and next_digest not in seen_next else 0.0

        self._world_model.observe(
            grid_arr,
            int(action),
            data,
            next_arr,
            level_before=level_before,
            level_after=level_after,
        )
        seen_next.add(next_digest)

        after_pred = np.asarray(self._world_model.engine(grid_arr, int(action), data))
        after_error = _per_cell_error(after_pred, next_arr)
        raw_progress = max(0.0, before_error - after_error)

        prior_noise = self._aleatoric.get(key, 0.0)
        current_noise = max(after_error, conflict)
        smoothed_noise = (
            self.aleatoric_decay * prior_noise + (1.0 - self.aleatoric_decay) * current_noise
        )
        aleatoric = max(current_noise, smoothed_noise)
        self._aleatoric[key] = aleatoric
        if conflict > 0.0:
            self._conflicts += 1

        bonus = max(0.0, raw_progress - self.noise_weight * aleatoric) * self.bonus_weight
        self._events += 1
        self.backup_edge(origin_hash, next_hash, bonus)
        return CuriosityEvent(
            origin_hash=str(origin_hash),
            next_hash=str(next_hash),
            action_key=akey,
            before_error=round(float(before_error), 6),
            after_error=round(float(after_error), 6),
            raw_progress=round(float(raw_progress), 6),
            aleatoric_estimate=round(float(aleatoric), 6),
            bonus=round(float(bonus), 6),
        )

    def backup_edge(self, origin_hash: str, next_hash: str, reward: float) -> None:
        """Single-step value backup over one explored graph edge."""

        reward = max(0.0, float(reward))
        self.record_state_bonus(next_hash, reward)
        next_value = self._state_values.get(str(next_hash), 0.0)
        backed_value = reward + self.backup_discount * next_value
        origin = str(origin_hash)
        if backed_value > self._state_values.get(origin, 0.0):
            self._state_values[origin] = backed_value
        edge = (origin, str(next_hash))
        if reward > self._edge_values.get(edge, 0.0):
            self._edge_values[edge] = reward

    def record_state_bonus(self, node_hash: str, bonus: float) -> None:
        """Record a directly observed state bonus for tests and live diagnostics."""

        node = str(node_hash)
        value = max(0.0, float(bonus))
        if value > self._state_values.get(node, 0.0):
            self._state_values[node] = value

    def score_state(self, node_hash: str) -> float:
        return float(self._state_values.get(str(node_hash), 0.0))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "game": self.game,
            "prediction_error_events": int(self._events),
            "aleatoric_conflicts": int(self._conflicts),
            "state_values": int(len(self._state_values)),
            "edge_values": int(len(self._edge_values)),
            "bonus_weight": float(self.bonus_weight),
            "backup_discount": float(self.backup_discount),
            "dense_signal_source": (
                "LiveTTT per-cell prediction error improvement with online aleatoric conflict "
                "suppression"
            ),
            "verifier_is_oracle": False,
        }
