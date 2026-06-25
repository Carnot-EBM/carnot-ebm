from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _frame() -> SimpleNamespace:
    grid = (np.arange(64 * 64, dtype=np.int16).reshape(64, 64) % 9).astype(np.int16)
    return SimpleNamespace(grid=grid, levels_completed=0, available_actions=[])


def _seed_candidates() -> list[dict]:
    return [
        {"action": 6, "data": {"x": 16, "y": 16}},
        {"action": 6, "data": {"x": 40, "y": 40}},
        {"action": 1, "data": None},
        {"action": 5, "data": None},
    ]


def test_req_arc_wmte_4738_spec_declares_valid_qd_gate() -> None:
    """REQ-ARC-WMTE-4738: OpenSpec anchors the valid QD generation test."""

    text = SPEC.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4738" in text
    assert "SCENARIO-ARC-WMTE-4738-NON-DEGENERATE-QD-ARMS" in text
    assert "results/experiment_4738_energy_fitness_qd_generation_valid_test.json" in text
    for field in (
        "arms_non_degenerate",
        "arm_pool_jaccard",
        "novel_candidates_generated",
        "energy_qd_vs_naive_delta",
        "positive_control_passed",
    ):
        assert field in text


def test_req_arc_wmte_4738_qd_candidate_pool_is_non_degenerate() -> None:
    """REQ-ARC-WMTE-4738: QD/random/naive pools differ and QD emits novel candidates."""

    from carnot.agentic.arc_energy_fitness_qd import (
        EnergyFitnessQDConfig,
        EnergyFitnessQDGenerator,
        candidate_signature_set,
        pool_jaccard,
    )

    def effect_score(_frame, row):
        data = row.get("data") or {}
        return ((int(data.get("x", 0)) + 2 * int(data.get("y", 0))) % 17) / 16.0

    naive = _seed_candidates()
    random_pool = EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(random_seed=17, use_energy_fitness=False)
    ).generate_candidate_pool(_frame(), naive, action_effect_scorer=effect_score)
    qd = EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(random_seed=17, use_energy_fitness=True)
    )
    qd_pool = qd.generate_candidate_pool(_frame(), naive, action_effect_scorer=effect_score)

    assert pool_jaccard(naive, random_pool) < 1.0
    assert pool_jaccard(naive, qd_pool) < 1.0
    assert pool_jaccard(random_pool, qd_pool) < 1.0
    assert len(candidate_signature_set(qd_pool) - candidate_signature_set(naive)) > 0

    diagnostics = qd.diagnostics()["candidate_pool"]
    assert diagnostics["arms_non_degenerate"] is True
    assert diagnostics["novel_candidates_generated"] > 0
    assert diagnostics["cpu_generation_ms"] >= 0.0
    assert diagnostics["verifier_is_oracle"] is False


@pytest.mark.memory_watchdog_skip
def test_scenario_arc_wmte_4738_stepwise_explorer_uses_live_qd_candidate_hook(
    monkeypatch,
) -> None:
    """SCENARIO-ARC-WMTE-4738-NON-DEGENERATE-QD-ARMS: live candidates call QD hook."""

    from carnot.agentic.arc_competition_agent import StepwiseExplorer
    from carnot.agentic import arc_graph_explore

    class ArcAction:
        def __init__(self, action_id, data=None):
            self.action_id = action_id
            self.data = data

    class FakeQD:
        def __init__(self):
            self.called = False

        def best_sequence(self, *_args, **_kwargs):
            return ()

        def generate_candidate_pool(self, _frame, candidates, **kwargs):
            self.called = True
            assert kwargs["arm_label"] == "energy-QD"
            return [dict(row) for row in candidates] + [
                {"action": 6, "data": {"x": 9, "y": 9}, "generated_by": "energy-QD"}
            ]

        def diagnostics(self):
            return {"called": self.called, "verifier_is_oracle": False}

    monkeypatch.setattr(
        arc_graph_explore,
        "rich_action_candidates",
        lambda *args, **kwargs: [ArcAction(1), ArcAction(6, {"x": 1, "y": 2})],
    )

    qd = FakeQD()
    explorer = StepwiseExplorer(
        frame_change_scorer=None,
        candidate_router=None,
        qd_generator=qd,
        goal_bias=lambda _frame: 0.0,
    )

    rows = explorer._candidates(_frame())

    assert qd.called is True
    assert {"action": 6, "data": {"x": 9, "y": 9}, "generated_by": "energy-QD"} in rows
    assert explorer.qd_generation_diagnostics()["generator"]["called"] is True


def test_scenario_arc_wmte_4738_artifact_schema_accepts_flat_nondegenerate_null() -> None:
    """SCENARIO-ARC-WMTE-4738-HELDOUT-NULL-OR-LIFT: flat valid null is annotated."""

    from carnot import experiment_4738_energy_fitness_qd_generation_valid_test as exp

    artifact = exp.build_artifact(
        preconditions_checked=exp.ok_preconditions_for_tests(),
        nondegeneracy={
            "arms_non_degenerate": True,
            "arm_pool_jaccard": {
                "naive-search__random-mutation": 0.5,
                "naive-search__energy-QD": 0.25,
                "random-mutation__energy-QD": 0.75,
            },
            "novel_candidates_generated": 4,
            "cpu_generation_ms": 0.12,
        },
        naive_measurement=exp.measurement_from_attempts(
            [{"attempted": True, "first_win": False, "variant_signature": "lp85~color01"}]
        ),
        random_measurement=exp.measurement_from_attempts(
            [{"attempted": True, "first_win": False, "variant_signature": "lp85~color01"}]
        ),
        qd_measurement=exp.measurement_from_attempts(
            [{"attempted": True, "first_win": False, "variant_signature": "lp85~color01"}]
        ),
        multi_level_probe={
            "goal_free_l2_reached": False,
            "offline_reproduced": False,
            "reproduced_levels": 0,
        },
        live_path_check={"passed": True},
        parity_test={"passed": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        duration_s=60.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["arms_non_degenerate"] is True
    assert artifact["energy_qd_vs_naive_delta"] == 0.0
    assert artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
    assert exp.artifact_schema_errors(artifact) == []
