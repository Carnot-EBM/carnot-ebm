"""Tests for the Exp4020 graded goal-energy live wire.

Spec refs: REQ-ARC-WMTE-4640, SCENARIO-ARC-WMTE-4640.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.agentic.arc_goal_energy_live import (
    GoalSatisfactionEnergy,
    load_exp4020_goal_energy,
    make_goal_energy_heuristic,
    make_uniform_goal_energy,
)
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


PREDICATE_CODE = 'def is_goal(state):\n    return state["unsatisfied_targets"] == 0\n'


def test_req_arc_wmte_4640_goal_energy_is_graded_fraction_not_binary() -> None:
    """REQ-ARC-WMTE-4640: Exp4020 is compiled into fraction-satisfied energy."""

    energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)

    assert energy({"total_targets": 4, "satisfied_targets": 0, "unsatisfied_targets": 4}) == 1.0
    assert energy({"total_targets": 4, "satisfied_targets": 1, "unsatisfied_targets": 3}) == 0.75
    assert energy({"total_targets": 4, "satisfied_targets": 3, "unsatisfied_targets": 1}) == 0.25
    assert energy({"total_targets": 4, "satisfied_targets": 4, "unsatisfied_targets": 0}) == 0.0
    assert energy.predicate_fires(
        {"total_targets": 4, "satisfied_targets": 4, "unsatisfied_targets": 0}
    )


def test_req_arc_wmte_4640_convex_combines_navigation_and_goal_energy() -> None:
    """REQ-ARC-WMTE-4640: the search bias is alpha*navigation + beta*goal energy."""

    goal = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    heuristic = make_goal_energy_heuristic(
        navigation_energy=lambda state: float(state["navigation"]),
        goal_energy=goal,
        alpha=0.7,
        beta=0.3,
    )

    state = {
        "navigation": 10.0,
        "total_targets": 4,
        "satisfied_targets": 2,
        "unsatisfied_targets": 2,
    }
    assert heuristic(state) == pytest.approx(0.7 * 10.0 + 0.3 * 0.5)


class _GoalEnergyToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return self._frame()

    def _frame(self) -> Any:
        levels = 1 if self.state == "win" else 0
        visible = {
            "root": {"total_targets": 2, "satisfied_targets": 0, "unsatisfied_targets": 2},
            "bad": {"total_targets": 2, "satisfied_targets": 0, "unsatisfied_targets": 2},
            "trap": {"total_targets": 2, "satisfied_targets": 0, "unsatisfied_targets": 2},
            "near": {"total_targets": 2, "satisfied_targets": 1, "unsatisfied_targets": 1},
            "win": {"total_targets": 2, "satisfied_targets": 2, "unsatisfied_targets": 0},
        }[self.state]
        values = {"root": 0, "bad": 1, "trap": 2, "near": 3, "win": 9}
        actions = [] if self.state == "win" else ([1] if self.state == "near" else [1, 2])
        return SimpleNamespace(
            frame=np.array([[values[self.state]]], dtype=np.int16),
            available_actions=actions,
            levels_completed=levels,
            goal_state=visible,
            nav_energy=1.0,
        )

    @staticmethod
    def _action_id(action: Any) -> int:
        if hasattr(action, "value"):
            return int(action.value)
        text = str(action)
        if "ACTION" in text:
            return int(text.rsplit("ACTION", 1)[-1])
        return int(action)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        aid = self._action_id(action)
        transitions = {
            ("root", 1): "bad",
            ("root", 2): "near",
            ("bad", 1): "trap",
            ("bad", 2): "trap",
            ("trap", 1): "trap",
            ("trap", 2): "trap",
            ("near", 1): "win",
        }
        self.state = transitions[(self.state, aid)]
        return self._frame()


def test_scenario_arc_wmte_4640_graph_explore_uses_goal_energy_to_generate_winner() -> None:
    """SCENARIO-ARC-WMTE-4640: dense goal energy guides expansion, predicate gates emit."""

    baseline, baseline_level = graph_explore_solve_v2(
        _GoalEnergyToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        heuristic=lambda frame: 1.0,
    )
    stats: dict[str, Any] = {}
    goal = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    guided, guided_level = graph_explore_solve_v2(
        _GoalEnergyToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        heuristic=lambda frame: float(frame.nav_energy),
        goal_energy=goal,
        goal_energy_alpha=0.5,
        goal_energy_beta=0.5,
        emit_plan_only_when_goal_predicate_fires=True,
        stats=stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert guided == [{"action": 2, "data": None}, {"action": 1, "data": None}]
    assert guided_level == 1
    assert stats["goal_energy_enabled"] is True
    assert stats["goal_predicate_gate_enabled"] is True
    assert stats["goal_predicate_plan_emitted"] is True


def test_scenario_arc_wmte_4640_uniform_energy_is_deterministic_ablation() -> None:
    """SCENARIO-ARC-WMTE-4640: uniform-energy control is deterministic and bounded."""

    energy = make_uniform_goal_energy(seed=4640)
    state = {"variant_signature": "g1~color01", "total_targets": 2}

    assert energy(state) == energy(dict(state))
    assert 0.0 <= energy(state) <= 1.0


def test_req_arc_wmte_4640_goal_energy_helpers_cover_defensive_branches(tmp_path) -> None:
    """REQ-ARC-WMTE-4640: helper branches stay deterministic and non-crashing."""

    artifact = {"goal_predicate_code": PREDICATE_CODE}
    path = tmp_path / "results" / "experiment_4020_goal_induction_separation.json"
    path.parent.mkdir()
    path.write_text(json.dumps(artifact), encoding="utf-8")

    energy = GoalSatisfactionEnergy.from_artifact(artifact)
    loaded = GoalSatisfactionEnergy.from_artifact_path(path)

    assert loaded({"total_targets": 2, "satisfied_targets": 1, "unsatisfied_targets": 1}) == 0.5
    assert load_exp4020_goal_energy(tmp_path) is not None
    assert load_exp4020_goal_energy(tmp_path / "missing") is None
    assert energy(SimpleNamespace(visible_goal_state={"total_targets": 0})) == 1.0
    assert energy(SimpleNamespace(target_group_state={"satisfied_targets": 1, "unsatisfied_targets": 1})) == 0.5
    assert energy(object()) == 1.0
    assert make_uniform_goal_energy().predicate_fires({}) is False
    assert 0.0 <= make_uniform_goal_energy()(object()) <= 1.0

    heuristic = make_goal_energy_heuristic(
        navigation_energy=None,
        goal_energy=energy,
        alpha=0.0,
        beta=1.0,
    )
    components = heuristic.components({"total_targets": 2, "satisfied_targets": 1, "unsatisfied_targets": 1})
    assert components["navigation"] == 0.0
    assert components["goal_energy"] == 0.5
    assert heuristic.predicate_fires({"total_targets": 2, "satisfied_targets": 2, "unsatisfied_targets": 0})

    broken = GoalSatisfactionEnergy(lambda _state: (_ for _ in ()).throw(RuntimeError("boom")), PREDICATE_CODE)
    assert broken.predicate_fires({"total_targets": 1, "satisfied_targets": 1, "unsatisfied_targets": 0}) is False

    with pytest.raises(ValueError):
        GoalSatisfactionEnergy.from_artifact({})
    with pytest.raises(ValueError):
        make_goal_energy_heuristic(
            navigation_energy=None,
            goal_energy=energy,
            alpha=0.2,
            beta=0.2,
        )
