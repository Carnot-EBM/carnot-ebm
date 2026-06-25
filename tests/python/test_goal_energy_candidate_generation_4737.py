"""Tests for Exp 4737 goal-energy candidate-generation guidance.

Spec refs: REQ-ARC-WMTE-4737,
SCENARIO-ARC-WMTE-4737-NON-DEGENERATE-GOAL-GUIDANCE,
SCENARIO-ARC-WMTE-4737-HELDOUT-NULL-OR-LIFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
PREDICATE_CODE = 'def is_goal(state):\n    return state["unsatisfied_targets"] == 0\n'


def _candidate(action: int) -> dict[str, Any]:
    return {"action": action, "data": None}


def _measurement(first_win: float, signature: str = "aa00~color01") -> dict[str, Any]:
    solved = bool(first_win > 0.0)
    return {
        "variant_attempts": [
            {
                "game": signature.split("~", 1)[0],
                "variant_signature": signature,
                "attempted": True,
                "solved": solved,
                "first_win": solved,
                "reached_level": 1 if solved else 0,
                "reproduction_gate": {"reproduced": solved, "reached_level": 1 if solved else 0},
            }
        ],
        "variant_attempts_count": 1,
        "variant_solved_count": 1 if solved else 0,
        "first_win_rate": float(first_win),
        "solve_rate": float(first_win),
        "variant_signatures": [signature],
    }


def test_req_arc_wmte_4737_spec_declares_guidance_contract() -> None:
    """REQ-ARC-WMTE-4737: OpenSpec declares the valid-test artifact and principles."""

    from carnot import experiment_4737_goal_energy_candidate_generation_valid_test as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4737" in spec
    assert "SCENARIO-ARC-WMTE-4737-NON-DEGENERATE-GOAL-GUIDANCE" in spec
    assert "SCENARIO-ARC-WMTE-4737-HELDOUT-NULL-OR-LIFT" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4737_guidance_scores_real_candidate_states_and_reranks() -> None:
    """REQ-ARC-WMTE-4737: candidate guidance scores predictor states, not cached attempts."""

    from carnot.agentic.arc_goal_energy_live import (
        GoalEnergyCandidateGuidance,
        GoalSatisfactionEnergy,
    )

    energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    predicted = {
        1: {"total_targets": 4, "satisfied_targets": 0, "unsatisfied_targets": 4},
        2: {"total_targets": 4, "satisfied_targets": 4, "unsatisfied_targets": 0},
        3: {"total_targets": 4, "satisfied_targets": 2, "unsatisfied_targets": 2},
    }

    def predictor(_frame: Any, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return predicted[int(candidate["action"])]

    guidance = GoalEnergyCandidateGuidance(
        goal_energy=energy,
        transition_predictor=predictor,
        alpha=0.0,
        beta=1.0,
    )
    ranked = guidance.rank_candidates(object(), [_candidate(1), _candidate(2), _candidate(3)])
    diagnostics = guidance.diagnostics()

    assert [row["action"] for row in ranked] == [2, 3, 1]
    assert diagnostics["candidate_states_scored"] == 3
    assert diagnostics["real_candidate_state_evidence"] is True
    assert diagnostics["candidate_pool_differs_from_baseline"] is True
    assert diagnostics["arms_non_degenerate"] is True
    assert diagnostics["goal_energy_score_variance"] == pytest.approx(np.var([1.0, 0.0, 0.5]))
    assert ranked[0]["goal_energy_score"] == 0.0
    assert ranked[0]["predicted_candidate_state_hash"]


def test_req_arc_wmte_4737_degenerate_guidance_is_detected_without_reordering() -> None:
    """REQ-ARC-WMTE-4737: zero score variance is a harness bug, not lift evidence."""

    from carnot.agentic.arc_goal_energy_live import GoalEnergyCandidateGuidance

    guidance = GoalEnergyCandidateGuidance(
        goal_energy=lambda _state: 1.0,
        transition_predictor=lambda _frame, candidate: {"state_id": candidate["action"]},
    )
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    ranked = guidance.rank_candidates(object(), candidates)
    diagnostics = guidance.diagnostics()

    assert ranked == candidates
    assert diagnostics["candidate_states_scored"] == 3
    assert diagnostics["goal_energy_score_variance"] == 0.0
    assert diagnostics["candidate_pool_differs_from_baseline"] is False
    assert diagnostics["arms_non_degenerate"] is False


@pytest.mark.memory_watchdog_skip
def test_scenario_arc_wmte_4737_stepwise_explorer_applies_guidance_in_live_candidates() -> None:
    """SCENARIO-ARC-WMTE-4737-NON-DEGENERATE-GOAL-GUIDANCE: E3 candidate path is wired."""

    from carnot.agentic.arc_competition_agent import StepwiseExplorer
    from carnot.agentic.arc_goal_energy_live import GoalEnergyCandidateGuidance

    frame = SimpleNamespace(
        frame=np.zeros((4, 4), dtype=np.int16),
        available_actions=[1, 2],
        levels_completed=0,
    )

    def predictor(_frame: Any, candidate: Mapping[str, Any]) -> Mapping[str, float]:
        return {"goal_distance": 0.0 if int(candidate["action"]) == 2 else 1.0}

    guidance = GoalEnergyCandidateGuidance(
        goal_energy=lambda state: float(state["goal_distance"]),
        transition_predictor=predictor,
        alpha=0.0,
        beta=1.0,
    )
    explorer = StepwiseExplorer(
        goal_bias=None,
        goal_candidate_guidance=guidance,
        frame_change_scorer=None,
        candidate_router=None,
    )

    assert [row["action"] for row in explorer._candidates(frame)] == [2, 1]
    assert explorer.goal_candidate_guidance_diagnostics()["arms_non_degenerate"] is True


def test_scenario_arc_wmte_4737_artifact_null_delta_markers_are_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4737-HELDOUT-NULL-OR-LIFT: flat non-degenerate arms emit markers."""

    from carnot import experiment_4737_goal_energy_candidate_generation_valid_test as mod

    nondegenerate = {
        "arms_non_degenerate": True,
        "candidate_pool_differs_from_baseline": True,
        "goal_energy_score_variance": 0.125,
        "cpu_scoring_ms_per_candidate": 0.02,
        "diagnostics": {"candidate_states_scored": 3},
    }
    artifact = mod.build_artifact(
        preconditions_checked={
            "ok": True,
            "cuda_available": True,
            "qwen_cached": True,
            "offline_arcade": True,
            "qwen_props_verified": True,
        },
        nondegeneracy=nondegenerate,
        baseline_measurement=_measurement(0.0),
        goal_energy_measurement=_measurement(0.0),
        multi_level_probe={"goal_free_l2_reached": False, "offline_reproduced": False, "reproduced_levels": 0},
        live_path_check={"passed": True},
        parity_test={"passed": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        duration_s=60.0,
    )

    output = tmp_path / mod.RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["arms_non_degenerate"] is True
    assert artifact["goal_energy_vs_baseline_delta"] == 0.0
    assert artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["proposer_served_model"] == "Qwen3.5-9B-MTP"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
