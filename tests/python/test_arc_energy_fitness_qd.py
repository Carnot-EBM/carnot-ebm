"""Tests for the live energy-fitness QD action-sequence generator.

Spec refs: REQ-ARC-WMTE-4653, SCENARIO-ARC-WMTE-4653.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic import arc_competition_agent as comp
from carnot.agentic.arc_energy_fitness_qd import (
    EnergyFitnessQDConfig,
    EnergyFitnessQDGenerator,
    SequenceEvaluation,
    fitness_from_evaluation,
    mutate_sequence,
    shared_state_crossover,
)
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


def _frame(state: str, actions: tuple[int, ...] = (1, 2)) -> SimpleNamespace:
    values = {"root": 0, "trap": 1, "near": 2, "win": 9}
    return SimpleNamespace(
        frame=np.array([[values[state]]], dtype=np.int16),
        levels_completed=1 if state == "win" else 0,
        available_actions=list(actions),
        state=state,
    )


class ActionScore:
    def candidate_score(self, _frame: Any, candidate: Any) -> float:
        if isinstance(candidate, dict):
            action_id = int(candidate.get("action", candidate.get("action_id", 0)))
        else:
            action_id = int(getattr(candidate, "action_id", 0))
        return 0.95 if action_id == 2 else 0.05


class _QDToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return _frame("root", (1, 2))

    @staticmethod
    def _action_id(action: Any) -> int:
        if hasattr(action, "value"):
            return int(action.value)
        text = str(action)
        if "ACTION" in text:
            return int(text.rsplit("ACTION", 1)[-1])
        return int(action)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        action_id = self._action_id(action)
        transitions = {
            ("root", 1): "trap",
            ("root", 2): "near",
            ("trap", 1): "trap",
            ("trap", 2): "trap",
            ("near", 1): "trap",
            ("near", 2): "win",
        }
        self.state = transitions[(self.state, action_id)]
        return _frame(self.state, () if self.state == "win" else (1, 2))


def test_req_arc_wmte_4653_fitness_uses_energy_effect_and_efficiency() -> None:
    """REQ-ARC-WMTE-4653: sequence fitness combines goal-energy, effect, and efficiency."""

    evaluation = SequenceEvaluation(
        sequence=({"action": 2, "data": None}, {"action": 2, "data": None}),
        behavior_descriptor=(2, 9, 0),
        goal_energy_start=1.0,
        goal_energy_end=0.1,
        action_effect_cell_recall=0.8,
        won=True,
        actions_to_win=2,
        state_trace=("root", "near", "win"),
        generated_by="fixture",
    )

    fitness = fitness_from_evaluation(evaluation)

    assert fitness.components["goal_energy_delta"] == 0.9
    assert fitness.components["action_effect_cell_recall"] == 0.8
    assert fitness.components["first_win_efficiency"] == 0.5
    assert fitness.total == 2.2
    assert fitness.verifier_is_oracle is False


def test_req_arc_wmte_4653_mutation_and_shared_state_crossover() -> None:
    """REQ-ARC-WMTE-4653: QD supports insert/delete/swap/splice and shared-state crossover."""

    sequence = (
        {"action": 1, "data": None},
        {"action": 2, "data": None},
    )
    pool = (
        {"action": 3, "data": None},
        {"action": 4, "data": None},
    )

    assert len(mutate_sequence(sequence, pool, operation="insert", index=1)) == 3
    assert mutate_sequence(sequence, pool, operation="delete", index=0) == (
        {"action": 2, "data": None},
    )
    assert mutate_sequence(sequence, pool, operation="swap", index=0)[0]["action"] in {3, 4}
    assert mutate_sequence(sequence, pool, operation="splice", index=1) == (
        {"action": 1, "data": None},
        {"action": 3, "data": None},
    )

    crossed = shared_state_crossover(
        left=({"action": 1, "data": None}, {"action": 2, "data": None}),
        left_states=("root", "shared", "dead"),
        right=({"action": 4, "data": None}, {"action": 5, "data": None}),
        right_states=("alt", "shared", "win"),
    )

    assert crossed == ({"action": 1, "data": None}, {"action": 5, "data": None})


def test_scenario_arc_wmte_4653_graph_explore_injects_generated_sequence() -> None:
    """SCENARIO-ARC-WMTE-4653: graph_explore can execute a QD sequence primitives miss."""

    scorer = ActionScore()
    generator = EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(
            enabled=True,
            random_seed=4653,
            max_sequence_len=2,
            mutation_rounds=8,
            archive_size=8,
        ),
        action_effect_scorer=scorer,
    )
    baseline_stats: dict[str, Any] = {}
    baseline, baseline_level = graph_explore_solve_v2(
        _QDToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        frame_change_scorer=scorer,
        qd_generator=None,
        stats=baseline_stats,
    )
    qd_stats: dict[str, Any] = {}
    qd_path, qd_level = graph_explore_solve_v2(
        _QDToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        frame_change_scorer=scorer,
        qd_generator=generator,
        stats=qd_stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert qd_level == 1
    assert qd_path == [{"action": 2, "data": None}, {"action": 2, "data": None}]
    assert qd_stats["qd_generation_enabled"] is True
    assert qd_stats["qd_sequences_injected"] >= 1


def test_scenario_arc_wmte_4653_stepwise_e3_path_accepts_qd_generator() -> None:
    """SCENARIO-ARC-WMTE-4653: E3/StepwiseExplorer can inject a generated sequence live."""

    scorer = ActionScore()
    generator = EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(enabled=True, random_seed=4653, max_sequence_len=2),
        action_effect_scorer=scorer,
    )
    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=scorer,
        qd_generator=generator,
        goal_bias=lambda frame: 1.0 if getattr(frame, "state", "") != "win" else 0.0,
    )

    move = explorer.next_move([], _frame("root", (1, 2)))

    assert move == (2, None)
    assert explorer.pending == [{"kind": 2, "data": None, "probe": True}]
    assert explorer.qd_generation_diagnostics()["sequences_injected"] == 1
