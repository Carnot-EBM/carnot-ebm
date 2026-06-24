"""Tests for Exp 4689 program-synthesis action-effect proposal filtering.

Spec refs: REQ-ARC-WMTE-4689,
SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION,
SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING,
SCENARIO-ARC-WMTE-4689-COVERAGE-CONTROL.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot import experiment_4689_program_synthesis_action_effect_proposal_filter as exp4689
from carnot.agentic import arc_competition_agent as comp
from carnot.agentic.arc_executable_world_model import ProgrammaticExpert, Transition
from carnot.agentic.arc_program_synthesis_filter import (
    ActionEffectProposalFilter,
    induce_action_effect_proposal_filter,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


class _FixtureProposer:
    def induce_programmatic_experts(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "stable_action_1_rewrite",
                "object_class": "color_1",
                "kind": "color_rewrite",
                "action": 1,
                "from_color": 1,
                "to_color": 2,
            },
            {
                "name": "prefix_overfit_action_2_rewrite",
                "object_class": "color_3",
                "kind": "color_rewrite",
                "action": 2,
                "from_color": 3,
                "to_color": 4,
            },
        ]


def _transition(before: int, action: int, after: int) -> Transition:
    return Transition(
        grid=np.array([[before]], dtype=np.int16),
        action=action,
        data=None,
        next_grid=np.array([[after]], dtype=np.int16),
        level_before=0,
        level_after=0,
    )


def _stable_expert() -> ProgrammaticExpert:
    def _precondition(grid: np.ndarray, action: int, _data: Any) -> bool:
        return int(action) == 1 and bool(np.any(np.asarray(grid) == 1))

    def _effect(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[out == 1] = 2
        return out

    return ProgrammaticExpert(
        name="stable_action_1_rewrite",
        object_class="color_1",
        precondition=_precondition,
        effect=_effect,
        action=1,
        trust=1.0,
        heldout_correct=1,
        heldout_total=1,
    )


def test_req_arc_wmte_4689_spec_declares_heldout_filter_fields() -> None:
    """REQ-ARC-WMTE-4689: OpenSpec declares held-out counts and coverage-control fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4689" in spec
    assert "SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION" in spec
    assert "SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING" in spec
    for field in (
        "candidate_generation_coverage_filter",
        "candidate_generation_coverage_blind_baseline",
        "heldout_programs_kept",
        "heldout_programs_rejected",
    ):
        assert field in spec


def test_scenario_arc_wmte_4689_heldout_rejection_keeps_only_stable_programs() -> None:
    """SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION: prefix-overfit programs are rejected."""

    transitions = [
        _transition(1, 1, 2),
        _transition(1, 1, 2),
        _transition(3, 2, 4),
        _transition(3, 2, 5),
    ]

    result = induce_action_effect_proposal_filter(
        game="fixture",
        transitions=transitions,
        proposer=_FixtureProposer(),
        trust_threshold=1.0,
        heldout_fraction=0.25,
    )

    assert result.heldout_programs_kept == 1
    assert result.heldout_programs_rejected == 1
    assert [expert.name for expert in result.proposal_filter.experts] == ["stable_action_1_rewrite"]
    assert result.program_trust_weights == [
        {
            "name": "stable_action_1_rewrite",
            "object_class": "color_1",
            "trust": 1.0,
            "heldout_correct": 1,
            "heldout_total": 1,
            "kept": True,
        },
        {
            "name": "prefix_overfit_action_2_rewrite",
            "object_class": "color_3",
            "trust": 0.0,
            "heldout_correct": 0,
            "heldout_total": 1,
            "kept": False,
        },
    ]


def test_scenario_arc_wmte_4689_filter_prunes_to_trusted_effect_actions() -> None:
    """SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING: only trusted effect actions survive."""

    proposal_filter = ActionEffectProposalFilter(
        game="fixture",
        experts=[_stable_expert()],
        program_trust_weights=[],
        heldout_programs_kept=1,
        heldout_programs_rejected=1,
    )
    candidates = [
        {"action": 1, "data": None},
        {"action": 2, "data": None},
        {"action": 6, "data": {"x": 0, "y": 0}},
    ]

    pruned = proposal_filter.filter_candidates(np.array([[1]], dtype=np.int16), candidates)

    assert pruned == [{"action": 1, "data": None}]
    diagnostics = proposal_filter.diagnostics()
    assert diagnostics["candidate_sets_scored"] == 1
    assert diagnostics["candidates_pruned"] == 2
    assert diagnostics["fallback_no_match"] == 0


def test_scenario_arc_wmte_4689_stepwise_explorer_uses_live_filter() -> None:
    """SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING: StepwiseExplorer prunes before ranking."""

    proposal_filter = ActionEffectProposalFilter(
        game="fixture",
        experts=[_stable_expert()],
        program_trust_weights=[],
        heldout_programs_kept=1,
        heldout_programs_rejected=0,
    )
    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
        program_synthesis_filter=proposal_filter,
    )
    frame = SimpleNamespace(frame=np.array([[1]], dtype=np.int16), available_actions=[1, 2])

    candidates = explorer._candidates(frame, path=[])

    assert candidates == [{"action": 1, "data": None}]
    assert explorer.program_synthesis_filter_diagnostics()["candidates_pruned"] == 1


def test_scenario_arc_wmte_4689_artifact_schema_records_counts_and_null() -> None:
    """SCENARIO-ARC-WMTE-4689-COVERAGE-CONTROL: artifact records coverage and held-out counts."""

    artifact = exp4689.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_games=["bp35"],
        candidate_generation_coverage_filter=0.0,
        candidate_generation_coverage_blind_baseline=0.0,
        heldout_programs_kept=1,
        heldout_programs_rejected=1,
        live_first_win_rate_filter=0.0,
        live_baseline_blind_proposal={"first_win_rate": 0.04, "source": "fixture"},
        live_lift_ci={"metric": "first_win_rate_delta", "low": 0.0, "high": 0.0, "n_boot": 64},
        bare_control_passed=True,
        offline_reproduced=False,
        duration_s=60.0,
        target_arm_results={"candidate_generation_probe": {"rows": []}},
    )

    assert artifact["heldout_programs_kept"] == 1
    assert artifact["heldout_programs_rejected"] == 1
    assert artifact["coverage_delta"] == 0.0
    assert artifact["first_win_rate_delta"] == -0.04
    assert artifact["null_methodology_note"]
    assert artifact["residual_bridge_gap"] in {
        "heldout_transitions_too_sparse",
        "program_cannot_target_winning_action",
    }
    assert exp4689.artifact_schema_errors(artifact) == []
