"""Tests for Exp 4015 ARC-AGI-3 fifth-game explore-first pruning.

Spec refs: REQ-PHASE4-027, SCENARIO-PHASE4-027.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_fifth_game_explore_first as helper
from carnot.agentic.arc_fifth_game_explore_first import (
    REQUIRED_ARTIFACT_FIELDS,
    AttemptResult,
    CandidateGame,
    artifact_schema_errors,
    build_fifth_game_artifact,
    select_fifth_candidate_order,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4015_fifth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _attempt(
    game_id: str = "tn36-ef4dde99",
    *,
    solved: bool = True,
    first_solve_at_action: int = 7,
) -> AttemptResult:
    return AttemptResult(
        game_id=game_id,
        baseline_actions=32,
        target_selection_reason=(
            "selected: non-spatial directly-observable target; L0 baseline_actions=32"
        ),
        exploration_actions_used=4,
        dynamics_induced=True,
        first_solve_at_action=first_solve_at_action if solved else -1,
        levels_completed=1 if solved else 0,
        actions_vs_baseline=(first_solve_at_action / 32.0) if solved else 0.0,
        induced_mechanic=(
            "Observed TN36 program-bit toggles and the execute button; induced a "
            "button-to-program-state transition model before verifier pruning."
        ),
        real_env_confirmed=True,
        observed_dynamics=[
            {
                "action_key": [6, 24, 41],
                "n_changed": 4,
                "level_delta": 0,
                "game_over": False,
            }
        ],
        pruner_decisions=[
            {
                "action_key": [6, 39, 41],
                "energy": 0.0,
                "retained": True,
                "reason": "executed-consistency",
            }
        ],
        solve_log=[{"action": "click", "x": 36, "y": 55}],
        failure_reason="" if solved else "no verifier-retained real-env level-up",
    )


def test_req_phase4_027_spec_declares_fifth_game_contract() -> None:
    """REQ-PHASE4-027: OpenSpec declares Exp 4015 and its required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-027" in spec
    assert "SCENARIO-PHASE4-027" in spec
    assert "experiment_4015_fifth_game_explore_first.json" in spec
    assert "exclude already solved games" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_027_candidate_order_excludes_solved_games() -> None:
    """REQ-PHASE4-027: target selection filters banked solves before baseline sorting."""

    candidates = [
        CandidateGame("su15-1944f8ab", 22, True, True, "fourth game already solved"),
        CandidateGame("tn36-ef4dde99", 32, True, True, "visible target pose and buttons"),
        CandidateGame("dc22-fdcac232", 59, True, True, "visible player and goal"),
        CandidateGame("vc33-5430563c", 64, False, True, "PSPACE spatial trap"),
        CandidateGame("lp85-305b61c3", 17, True, True, "second game already solved"),
    ]

    ordered = select_fifth_candidate_order(candidates)

    assert [item.game_id for item in ordered] == ["tn36-ef4dde99", "dc22-fdcac232"]
    assert ordered[0].selection_reason.startswith("selected: non-spatial")
    assert "already solved" not in ordered[0].selection_reason


def test_req_phase4_027_artifact_schema_requires_bare_scalars() -> None:
    """REQ-PHASE4-027: Exp 4015 success artifacts keep fifth-game evidence auditable."""

    artifact = build_fifth_game_artifact(
        [_attempt()],
        seed=4015,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert artifact["honest_verdict"] == "success: fifth_game_solved_tn36-ef4dde99_at_action7"
    assert artifact["ACCURACY_levels_solved"] == 1
    assert artifact["game_solved"] == "tn36-ef4dde99"
    assert artifact["exploration_actions_used"] == 4
    assert artifact["dynamics_induced"] is True
    assert artifact["actions_vs_baseline"] == 0.2188
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


def test_scenario_phase4_027_no_solve_and_blocked_artifacts() -> None:
    """SCENARIO-PHASE4-027: no-solve and blocked artifacts retain schema-valid evidence."""

    no_solve = build_fifth_game_artifact(
        [_attempt(solved=False)],
        seed=4015,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert no_solve["honest_verdict"].startswith("complete: fifth_game_no_solve_")
    assert no_solve["game_solved"] == "none"
    assert no_solve["first_solve_at_action"] == -1
    assert artifact_schema_errors(no_solve) == []

    blocked = helper.blocked_artifact(
        seed=4015,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert blocked["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert blocked["precondition_blocked"] is True
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    with pytest.raises(ValueError, match="honest_verdict"):
        helper.blocked_artifact(seed=4015, started=0.0, inference_substrate="test", verdict="bad")


def test_scenario_phase4_027_build_rejects_schema_errors(monkeypatch) -> None:
    """SCENARIO-PHASE4-027: helper refuses to emit a malformed fifth-game artifact."""

    monkeypatch.setattr(helper, "artifact_schema_errors", lambda artifact: ["forced schema error"])

    with pytest.raises(ValueError, match="forced schema error"):
        helper.build_fifth_game_artifact(
            [_attempt()],
            seed=4015,
            started=0.0,
            inference_substrate="test_substrate",
        )


def test_scenario_phase4_027_blocked_offline_arcade_writes_artifact(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-027: unavailable offline Arcade writes a blocked artifact."""

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
    assert json.loads(written.read_text(encoding="utf-8"))["honest_verdict"] == (
        "blocked_arc_offline_env_unavailable"
    )
