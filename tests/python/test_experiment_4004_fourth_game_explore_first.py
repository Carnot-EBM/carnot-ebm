"""Tests for Exp 4004 ARC-AGI-3 fourth-game explore-first pruning.

Spec refs: REQ-PHASE4-024, SCENARIO-PHASE4-024.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import carnot.agentic.arc_fourth_game_explore_first as helper
from carnot.agentic.arc_fourth_game_explore_first import (
    REQUIRED_ARTIFACT_FIELDS,
    AttemptResult,
    CandidateGame,
    TransitionObservation,
    artifact_schema_errors,
    build_fourth_game_artifact,
    induce_model_from_observations,
    prune_candidates_after_induction,
    select_candidate_order,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4004_fourth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _attempt(
    game_id: str = "tn36-ef4dde99",
    *,
    solved: bool = True,
    first_solve_at_action: int = 9,
) -> AttemptResult:
    return AttemptResult(
        game_id=game_id,
        baseline_actions=32,
        target_selection_reason="smallest non-spatial L0 baseline with observable target properties",
        exploration_actions_used=4,
        dynamics_induced=True,
        first_solve_at_action=first_solve_at_action if solved else -1,
        levels_completed=1 if solved else 0,
        actions_vs_baseline=(first_solve_at_action / 32.0) if solved else 0.0,
        induced_mechanic=(
            "Observed button clicks change visible sprite pose; induced a grounded "
            "button-to-property transition model before verifier pruning."
        ),
        real_env_confirmed=True,
        observed_dynamics=[
            {
                "action_key": [6, 11, 54],
                "n_changed": 12,
                "level_delta": 0,
                "game_over": False,
            }
        ],
        pruner_decisions=[
            {
                "action_key": [6, 11, 54],
                "energy": 0.0,
                "retained": True,
                "reason": "executed-consistency",
            }
        ],
        solve_log=[{"action": "click", "x": 11, "y": 54}],
        failure_reason="" if solved else "no verifier-retained real-env level-up",
    )


def test_req_phase4_024_spec_declares_explore_first_contract() -> None:
    """REQ-PHASE4-024: OpenSpec declares Exp 4004 and the required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-024" in spec
    assert "SCENARIO-PHASE4-024" in spec
    assert "experiment_4004_fourth_game_explore_first.json" in spec
    assert "Exp 3993's prune-before-induce failure" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_024_candidate_order_prefers_non_spatial_observable_baseline() -> None:
    """REQ-PHASE4-024: target selection excludes spatial candidates before sorting by L0 baseline."""

    candidates = [
        CandidateGame("su15-1944f8ab", 22, False, True, "counting target zone but spatial survey"),
        CandidateGame("tn36-ef4dde99", 32, True, True, "visible target pose and buttons"),
        CandidateGame("dc22-fdcac232", 59, True, True, "visible player and goal"),
    ]

    ordered = select_candidate_order(candidates)

    assert [item.game_id for item in ordered] == ["tn36-ef4dde99", "dc22-fdcac232"]
    assert ordered[0].selection_reason.startswith("selected: non-spatial")


def test_req_phase4_024_pruner_requires_induced_model_after_observed_transitions() -> None:
    """REQ-PHASE4-024: pruning is unavailable until positive-budget exploration induced dynamics."""

    start = np.zeros((3, 3), dtype=np.int16)
    changed = start.copy()
    changed[1, 1] = 5
    observations = [
        TransitionObservation(
            before=start,
            action_key=(6, 1, 1),
            after=changed,
            level_delta=0,
            game_over=False,
        )
    ]

    with pytest.raises(ValueError, match="induced model"):
        prune_candidates_after_induction(None, start, [((6, 1, 1), changed)], energy_threshold=0.2)

    model = induce_model_from_observations("tn36-ef4dde99", observations)
    decisions = prune_candidates_after_induction(
        model,
        start,
        [((6, 1, 1), changed), ((6, 2, 2), start)],
        energy_threshold=0.2,
    )

    assert model.n_train == 1
    assert decisions[0]["retained"] is True
    assert decisions[0]["energy"] == 0.0
    assert decisions[1]["retained"] is False


def test_req_phase4_024_induction_requires_at_least_one_observation() -> None:
    """REQ-PHASE4-024: an induced model cannot be fabricated from zero observed transitions."""

    with pytest.raises(ValueError, match="observed transition"):
        induce_model_from_observations("tn36-ef4dde99", [])


def test_req_phase4_024_artifact_schema_requires_bare_scalars() -> None:
    """REQ-PHASE4-024: Exp 4004 artifacts keep the fourth-game evidence auditable."""

    artifact = build_fourth_game_artifact(
        [_attempt()],
        seed=4004,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert artifact["honest_verdict"] == "success: fourth_game_solved_tn36-ef4dde99_at_action9"
    assert artifact["ACCURACY_levels_solved"] == 1
    assert artifact["exploration_actions_used"] == 4
    assert artifact["dynamics_induced"] is True
    assert artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["ACCURACY_levels_solved"] = "1"
    bad["exploration_actions_used"] = 0
    bad["dynamics_induced"] = False
    bad["real_env_confirmed"] = 1
    bad["honest_verdict"] = "done"
    bad["actions_vs_baseline"] = "0.3"

    errors = artifact_schema_errors(bad)

    assert any("ACCURACY_levels_solved" in err for err in errors)
    assert any("exploration_actions_used" in err for err in errors)
    assert any("dynamics_induced" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("honest_verdict" in err for err in errors)
    assert any("actions_vs_baseline" in err for err in errors)


def test_req_phase4_024_artifact_schema_covers_no_solve_and_missing_fields(monkeypatch) -> None:
    """SCENARIO-PHASE4-024: no-solve and invalid artifacts still report schema issues precisely."""

    no_solve = build_fourth_game_artifact(
        [_attempt(solved=False)],
        seed=4004,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert no_solve["honest_verdict"].startswith("complete: fourth_game_no_solve_")
    assert no_solve["game_solved"] == "none"
    assert no_solve["first_solve_at_action"] == -1
    assert artifact_schema_errors(no_solve) == []

    missing_errors = artifact_schema_errors(
        {
            "honest_verdict": "complete: malformed",
            "duration_s": "slow",
            "games_attempted": "tn36",
            "game_solved": 5,
            "induced_mechanic": 7,
            "inference_substrate": 9,
            "dynamics_induced": True,
            "real_env_confirmed": False,
            "actions_vs_baseline": 0.0,
        }
    )

    assert any("missing required field ACCURACY_levels_solved" in err for err in missing_errors)
    assert any("game_solved" in err for err in missing_errors)
    assert any("duration_s" in err for err in missing_errors)
    assert any("games_attempted" in err for err in missing_errors)

    success_bad = dict(no_solve)
    success_bad["honest_verdict"] = "success: fourth_game_solved_fake_at_action0"
    success_bad["exploration_actions_used"] = 0
    success_bad["dynamics_induced"] = False
    success_bad["first_solve_at_action"] = 0
    success_bad["real_env_confirmed"] = False

    success_errors = artifact_schema_errors(success_bad)

    assert any("exploration_actions_used must be >0 for success" in err for err in success_errors)
    assert any("dynamics_induced must be true for success" in err for err in success_errors)
    assert any("first_solve_at_action" in err for err in success_errors)

    with pytest.raises(ValueError, match="honest_verdict"):
        helper.blocked_artifact(seed=4004, started=0.0, inference_substrate="test", verdict="bad")

    monkeypatch.setattr(helper, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        helper.build_fourth_game_artifact(
            [_attempt()],
            seed=4004,
            started=0.0,
            inference_substrate="test_substrate",
        )


def test_scenario_phase4_024_blocked_offline_arcade_writes_artifact(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-024: unavailable offline Arcade writes a blocked artifact."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unavailable() -> object:
        raise RuntimeError("offline missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", unavailable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["real_env_confirmed"] is False
    assert artifact["ACCURACY_levels_solved"] == 0
    assert artifact["dynamics_induced"] is False
    assert artifact_schema_errors(artifact) == []
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_scenario_phase4_024_success_uses_mocked_explore_first_attempt(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-024: a solved attempt writes the required success verdict."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_attempts", lambda arc, budget, exploration_budget: [_attempt()])

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: fourth_game_solved_tn36-ef4dde99_at_action9"
    assert artifact["game_solved"] == "tn36-ef4dde99"
    assert artifact["exploration_actions_used"] == 4
    assert artifact["dynamics_induced"] is True
    assert artifact["real_env_confirmed"] is True
    assert artifact_schema_errors(artifact) == []
