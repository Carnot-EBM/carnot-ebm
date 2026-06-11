"""Tests for Exp 4014 ARC-AGI-3 explore-first level-wall re-induction.

Spec refs: REQ-PHASE4-026, SCENARIO-PHASE4-026.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic.arc_level_wall_explore_first import (
    BANKED_FRONTIER,
    REQUIRED_ARTIFACT_FIELDS,
    LevelWallResult,
    TransitionObservation,
    artifact_schema_errors,
    build_level_wall_artifact,
    count_validated_candidates,
    induce_model_from_level_observations,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4014_break_level_wall_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _wall(
    short_game: str,
    levels_completed: int,
    *,
    validated: int = 0,
    saved: int = 0,
    exploration: int = 2,
    real: bool = True,
) -> LevelWallResult:
    return LevelWallResult(
        short_game=short_game,
        game_id=f"{short_game}-fake",
        banked_level=BANKED_FRONTIER[short_game],
        target_level=BANKED_FRONTIER[short_game] + 1,
        levels_completed=levels_completed,
        first_fail_level=None if levels_completed > BANKED_FRONTIER[short_game] else BANKED_FRONTIER[short_game] + 1,
        exploration_actions_used=exploration,
        observed_dynamics=[
            {
                "action_key": [6, 1, 2],
                "level_delta": 0,
                "n_changed": 1,
                "game_over": False,
            }
        ],
        dynamics_induced=True,
        candidate_validations=[
            {
                "candidate_id": "observed-candidate",
                "heldout_energy": 0.0,
                "heldout_n": 1,
                "selected": validated > 0,
            }
        ],
        committed_actions=[],
        verifier_validated_count=validated,
        actions_saved_vs_openloop=saved,
        real_env_confirmed=real,
        stall_reason="no verifier-validated candidate after explore-first",
        solve_log=[],
    )


def test_req_phase4_026_spec_declares_explore_first_level_wall_contract() -> None:
    """REQ-PHASE4-026: OpenSpec declares Exp 4014 and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-026" in spec
    assert "SCENARIO-PHASE4-026" in spec
    assert "experiment_4014_break_level_wall_explore_first.json" in spec
    assert "spend a fixed positive exploratory action budget" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_026_induction_requires_observed_level_transitions() -> None:
    """REQ-PHASE4-026: per-level induction is blocked before exploration."""

    start = np.zeros((3, 3), dtype=np.int16)
    changed = start.copy()
    changed[1, 1] = 7
    observations = [
        TransitionObservation(
            before=start,
            action_key=(6, 1, 1),
            after=changed,
            level_delta=0,
            game_over=False,
        )
    ]

    with pytest.raises(ValueError, match="observed transition"):
        induce_model_from_level_observations("lp85-fake", [])

    model = induce_model_from_level_observations("lp85-fake", observations)

    assert model.n_train == 1


def test_req_phase4_026_artifact_schema_requires_bare_fields() -> None:
    """REQ-PHASE4-026: Exp 4014 artifacts keep level-wall fields auditable."""

    artifact = build_level_wall_artifact(
        [_wall("lp85", 1), _wall("sc25", 1), _wall("r11l", 3)],
        seed=4014,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert artifact["honest_verdict"].startswith("complete: level_walls_hold_")
    assert artifact["ACCURACY_total_levels_solved"] == 5
    assert artifact["new_levels_this_task"] == 0
    assert artifact["exploration_actions_used"] == 6
    assert artifact["explore_first_found_validated_candidate"] is False
    assert artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["ACCURACY_total_levels_solved"] = "5"
    bad["per_game_max_level"] = {"lp85": "1"}
    bad["explore_first_found_validated_candidate"] = 1
    bad["exploration_actions_used"] = "6"
    bad["real_env_confirmed"] = 1
    bad["honest_verdict"] = "done"

    errors = artifact_schema_errors(bad)

    assert any("ACCURACY_total_levels_solved" in err for err in errors)
    assert any("per_game_max_level" in err for err in errors)
    assert any("explore_first_found_validated_candidate" in err for err in errors)
    assert any("exploration_actions_used" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("honest_verdict" in err for err in errors)


def test_scenario_phase4_026_success_reports_broken_wall_and_validated_candidate() -> None:
    """SCENARIO-PHASE4-026: a real-env level-up uses the required success verdict."""

    artifact = build_level_wall_artifact(
        [_wall("lp85", 2, validated=1, saved=3), _wall("sc25", 1), _wall("r11l", 3)],
        seed=4014,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert artifact["ACCURACY_total_levels_solved"] == 6
    assert artifact["new_levels_this_task"] == 1
    assert artifact["per_game_max_level"] == {"lp85": 2, "sc25": 1, "r11l": 3}
    assert artifact["explore_first_found_validated_candidate"] is True
    assert artifact["verifier_validated_count"] == 1
    assert artifact["actions_saved_vs_openloop"] == 3
    assert artifact["honest_verdict"] == "success: broke_wall_lp85_to_L2_total6"
    assert artifact_schema_errors(artifact) == []


def test_req_phase4_026_validated_counter_requires_selected_heldout_energy() -> None:
    """REQ-PHASE4-026: only selected held-out verifier passes count as candidates."""

    rows = [
        {"candidate_id": "kept", "heldout_energy": 0.0, "heldout_n": 1, "selected": True},
        {"candidate_id": "demo-only", "heldout_energy": None, "heldout_n": 0, "selected": True},
        {"candidate_id": "rejected", "heldout_energy": 0.5, "heldout_n": 1, "selected": True},
        {"candidate_id": "not-selected", "heldout_energy": 0.0, "heldout_n": 1, "selected": False},
    ]

    assert count_validated_candidates(rows) == 1


def test_scenario_phase4_026_blocks_when_offline_arcade_unavailable(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-026: unavailable offline Arcade writes a blocked artifact."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unavailable() -> object:
        raise RuntimeError("offline missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", unavailable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["real_env_confirmed"] is False
    assert artifact["ACCURACY_total_levels_solved"] == 0
    assert artifact_schema_errors(artifact) == []
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_scenario_phase4_026_success_uses_mocked_real_env_results(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-026: mocked explore-first level-up writes the success artifact."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(
        exp,
        "_run_level_walls",
        lambda arc, budget, exploration_budget: [
            _wall("lp85", 2, validated=1, saved=3),
            _wall("sc25", 1),
            _wall("r11l", 3),
        ],
    )

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: broke_wall_lp85_to_L2_total6"
    assert artifact["new_levels_this_task"] == 1
    assert artifact["explore_first_found_validated_candidate"] is True
    assert artifact_schema_errors(artifact) == []
