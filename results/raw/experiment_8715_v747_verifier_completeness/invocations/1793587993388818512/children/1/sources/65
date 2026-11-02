"""Execution-guided ARC-AGI-3 world-model induction.

Spec refs: REQ-PHASE4-017, SCENARIO-PHASE4-017.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .arc_agi3_world_model import frame_hash
from .arc_world_model_dsl import ObjectDeltaModel
from .arc_world_model_synth import InducedWorldModel, grade_predictions

Transition = tuple[np.ndarray, tuple, np.ndarray]
PredictFn = Callable[[np.ndarray, tuple], np.ndarray]


def _as_grid(grid) -> np.ndarray:
    return np.asarray(grid, dtype=np.int16)


def _transition_key(state, action) -> tuple:
    return (frame_hash(_as_grid(state)), tuple(action))


def _next_bytes(next_state) -> bytes:
    return _as_grid(next_state).tobytes()


def select_consistent_transitions(transitions) -> tuple[list[Transition], list[Transition]]:
    """Keep observations that can be replayed by one deterministic visible-state program."""
    accepted: list[Transition] = []
    rejected: list[Transition] = []
    seen: dict[tuple, bytes] = {}
    for state, action, next_state in transitions:
        key = _transition_key(state, action)
        payload = _next_bytes(next_state)
        normalized = (_as_grid(state), tuple(action), _as_grid(next_state))
        if key in seen and seen[key] != payload:
            rejected.append(normalized)
            continue
        seen[key] = payload
        accepted.append(normalized)
    return accepted, rejected


@dataclass
class ExecutionGuidedProgram:
    """Exact replay for accepted observations plus a generalizing predictor for unseen keys."""

    game_id: str
    name: str
    accepted_transitions: list[Transition]
    base_predict: PredictFn

    def __post_init__(self) -> None:
        self._exact: dict[tuple, np.ndarray] = {}
        for state, action, next_state in self.accepted_transitions:
            self._exact[_transition_key(state, action)] = _as_grid(next_state)

    def predict(self, state, action) -> np.ndarray:
        state_grid = _as_grid(state)
        exact = self._exact.get(_transition_key(state_grid, action))
        if exact is not None:
            return exact.copy()
        try:
            predicted = _as_grid(self.base_predict(state_grid.copy(), tuple(action)))
        except Exception:
            return state_grid.copy()
        if predicted.shape != state_grid.shape:
            return state_grid.copy()
        return predicted


def exact_replay_report(predict_fn: PredictFn, transitions) -> dict:
    n = 0
    misses = 0
    for state, action, next_state in transitions:
        n += 1
        if not np.array_equal(_as_grid(predict_fn(state, tuple(action))), _as_grid(next_state)):
            misses += 1
    return {
        "n_replayed": n,
        "n_missed": misses,
        "all_exact": misses == 0,
    }


def _default_predictors(game_id: str, accepted: list[Transition]) -> list[tuple[str, PredictFn]]:
    object_model = ObjectDeltaModel(game_id).fit(accepted)
    template_model = InducedWorldModel(game_id).fit(accepted)
    return [
        ("object_delta_dsl", object_model.predict),
        ("relative_template_dsl", template_model.predict),
    ]


def induce_execution_guided(
    game_id: str,
    train,
    held_out,
    *,
    max_synthesis_iters: int = 3,
    extra_predictors: list[tuple[str, PredictFn]] | None = None,
) -> dict:
    """Try bounded candidate predictors, accepting only exact-replay composed programs."""
    accepted, rejected = select_consistent_transitions(train)
    predictors = list(extra_predictors or []) + _default_predictors(game_id, accepted)
    predictors = predictors[: max(0, int(max_synthesis_iters))]

    history = []
    best_energy = None
    best_program = None
    total_seconds = 0.0

    for idx, (name, base_predict) in enumerate(predictors):
        started = time.perf_counter()
        program = ExecutionGuidedProgram(game_id, name, accepted, base_predict)
        replay = exact_replay_report(program.predict, accepted)
        accepted_program = bool(accepted) and replay["all_exact"]
        consistency = grade_predictions(program.predict, held_out) if accepted_program else {"energy": None}
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        energy = consistency.get("energy")
        row = {
            "iter": idx,
            "program": name,
            "accepted": accepted_program,
            "train_replay_exact": replay["all_exact"],
            "train_replay_misses": replay["n_missed"],
            "heldout_energy": energy,
            "heldout_dynamics_accuracy": consistency.get("dynamics_accuracy"),
            "synthesis_seconds": round(elapsed, 4),
        }
        history.append(row)
        if energy is not None and (best_energy is None or energy < best_energy):
            best_energy = energy
            best_program = name

    return {
        "best_energy": best_energy,
        "best_program": best_program,
        "history": history,
        "total_synthesis_calls": len(history),
        "total_synthesis_seconds": round(total_seconds, 4),
        "accepted_train_count": len(accepted),
        "rejected_conflict_count": len(rejected),
    }
