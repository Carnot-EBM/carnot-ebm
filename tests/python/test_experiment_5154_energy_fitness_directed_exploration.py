"""Tests for Experiment 5154 energy-fitness directed exploration.

Spec refs: REQ-ARC-WMTE-5154, SCENARIO-ARC-WMTE-5154-LIVE-ENERGY-QD.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _frame() -> SimpleNamespace:
    grid = np.zeros((16, 16), dtype=np.int16)
    grid[4, 4] = 7
    grid[10, 11] = 3
    return SimpleNamespace(grid=grid, levels_completed=0)


def test_req_arc_wmte_5154_spec_declares_energy_fitness_pilot() -> None:
    """REQ-ARC-WMTE-5154: OpenSpec anchors the energy-fitness pilot artifact."""

    text = SPEC.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-5154" in text
    assert "SCENARIO-ARC-WMTE-5154-LIVE-ENERGY-QD" in text
    assert "results/experiment_5154_energy_fitness_directed_exploration_v472.json" in text
    for field in (
        "energy_signal_source",
        "reproducible_levels_delta",
        "offline_reproduced",
        "live_path_reachable",
        "winning_trajectory_surfaced",
    ):
        assert field in text


def test_scenario_arc_wmte_5154_live_qd_reports_real_goal_energy_source() -> None:
    """SCENARIO-ARC-WMTE-5154-LIVE-ENERGY-QD: diagnostics name the true energy source."""

    from carnot.agentic.arc_energy_fitness_qd import (
        EnergyFitnessQDConfig,
        EnergyFitnessQDGenerator,
    )
    from carnot.agentic.arc_goal_energy_live import GOAL_ENERGY_SOURCE

    class Exp4020LikeEnergy:
        source = GOAL_ENERGY_SOURCE

        def __call__(self, _frame: Any) -> float:
            return 0.75

    class EffectScorer:
        def candidate_score(self, _frame: Any, candidate: dict[str, Any]) -> float:
            data = candidate.get("data") or {}
            return 1.0 if data.get("x") == 4 and data.get("y") == 4 else 0.25

    generator = EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(
            random_seed=5154,
            use_energy_fitness=True,
            candidate_pool_max_new=4,
            mutation_rounds=4,
        ),
        goal_energy=Exp4020LikeEnergy(),
        action_effect_scorer=EffectScorer(),
    )
    pool = generator.generate_candidate_pool(
        _frame(),
        [
            {"action": 6, "data": {"x": 4, "y": 4}},
            {"action": 6, "data": {"x": 11, "y": 10}},
            {"action": 1, "data": None},
        ],
        arm_label="energy-QD",
    )

    generated = [row for row in pool if row.get("generated_by") == "energy-QD"]
    diagnostics = generator.diagnostics()["candidate_pool"]

    assert generated
    assert diagnostics["energy_signal_source"] == GOAL_ENERGY_SOURCE
    assert diagnostics["fitness_signal_kind"] == "goal_energy_fitness"
    assert diagnostics["verifier_is_oracle"] is False
    assert all(row["energy_signal_source"] == GOAL_ENERGY_SOURCE for row in generated)


def test_scenario_arc_wmte_5154_artifact_schema_accepts_honest_null() -> None:
    """SCENARIO-ARC-WMTE-5154-LIVE-ENERGY-QD: a null reports zero level delta honestly."""

    from carnot import experiment_5154_energy_fitness_directed_exploration_v472 as exp
    from carnot.agentic.arc_goal_energy_live import GOAL_ENERGY_SOURCE

    artifact = exp.build_artifact(
        preconditions_checked=exp.ok_preconditions_for_tests(),
        live_path_check={"passed": True, "stdout_tail": "OK: all solver-like ARC modules"},
        energy_arm={
            "game": "bp35",
            "reached_level": 0,
            "actions": 195,
            "winning_trajectory_surfaced": False,
            "qd_generation_diagnostics": {
                "enabled": True,
                "generator": {
                    "candidate_pool": {
                        "energy_signal_source": GOAL_ENERGY_SOURCE,
                        "fitness_signal_kind": "goal_energy_fitness",
                        "verifier_is_oracle": False,
                    }
                },
                "verifier_is_oracle": False,
            },
        },
        no_energy_control={
            "game": "bp35",
            "reached_level": 0,
            "actions": 195,
            "winning_trajectory_surfaced": False,
        },
        reproduction_gate={
            "game": "bp35",
            "claimed_level": 0,
            "reproduced": False,
            "reached_level": 0,
        },
        duration_s=60.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert "winning_trajectory_not_surfaced" in artifact["honest_verdict"]
    assert artifact["energy_signal_source"] == GOAL_ENERGY_SOURCE
    assert artifact["reproducible_levels_delta"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["live_path_reachable"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["verifier_is_oracle"] is False
    assert exp.artifact_schema_errors(artifact) == []
