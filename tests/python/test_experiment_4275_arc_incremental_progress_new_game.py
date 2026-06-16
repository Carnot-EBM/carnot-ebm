"""Tests for Exp 4275 ARC-AGI-3 offline new-game incremental progress.

Spec refs: REQ-PHASE4-069, SCENARIO-PHASE4-069.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4275_arc_incremental_progress_new_game as exp
from carnot.experiment_4275_arc_incremental_progress_new_game import (
    PRIOR_TOTAL_LEVELS,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_FIELD_PRINCIPLES,
    SC25_GAME_ID,
    WA30_GAME_ID,
    SolverOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    compute_reproducibility_checksum,
    load_environment_baselines,
    make_model_specs,
    select_best_headroom_unattempted_game,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _survey() -> dict[str, object]:
    return {
        "ranked_targets": [{"game": "r11l"}, {"game": "sc25"}, {"game": "lp85"}, {"game": "tn36"}],
        "per_game_surveys": [
            {"game": "r11l", "is_spatial_planning": False, "n_levels": 6, "win_difficulty": "medium"},
            {"game": "sc25", "is_spatial_planning": False, "n_levels": 6, "win_difficulty": "medium"},
            {"game": "lp85", "is_spatial_planning": False, "n_levels": 8, "win_difficulty": "medium"},
            {"game": "tn36", "is_spatial_planning": False, "n_levels": 7, "win_difficulty": "medium"},
            {"game": "lf52", "is_spatial_planning": True, "n_levels": 10, "win_difficulty": "hard"},
            {"game": "wa30", "is_spatial_planning": True, "n_levels": 9, "win_difficulty": "medium"},
        ],
    }


def _baselines() -> dict[str, tuple[str, list[int]]]:
    return {
        "r11l": ("r11l-495a7899", [22, 33, 51, 26, 52, 49]),
        "sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143, 50]),
        "lp85": ("lp85-305b61c3", [17, 38, 31, 16, 41, 60, 26, 159]),
        "tn36": ("tn36-ef4dde99", [32, 72, 26, 40, 30, 55, 62]),
        "lf52": ("lf52-271a04aa", [32, 81, 60, 71, 205, 148, 244, 109, 164, 225]),
        "wa30": (WA30_GAME_ID, [71, 119, 183, 98, 368, 68, 79, 442, 415]),
    }


def _prior_best() -> dict[str, object]:
    return {
        "experiment": "experiment_4249_arc_incremental_progress",
        "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L5_total19",
        "target_game": SC25_GAME_ID,
        "target_level": 5,
        "total_levels_solved": 19,
        "new_levels_solved_this_task": 1,
        "real_env_confirmed": True,
        "verifier_validated": True,
    }


def _wall() -> dict[str, object]:
    return {
        "experiment": "experiment_4261_arc_incremental_progress",
        "honest_verdict": "complete: incremental_progress_no_advance_sc25-635fd71a_L6_no_verifier_validated_level_up_candidate",
        "total_levels": 19,
        "total_levels_solved": 19,
        "levels_completed": 0,
        "new_levels_solved_this_task": 0,
        "game_advanced": "none",
        "target_game": SC25_GAME_ID,
        "target_level": 6,
        "real_env_confirmed": False,
    }


def _set_encoder() -> dict[str, object]:
    return {
        "random_seed": 4244,
        "model_type": "deepsets_pooled_context_set_encoder",
        "model_specs": {
            "architecture": "deepsets_pooled_context_set_encoder",
            "status": "trained",
            "feature_set": ["vote_weight", "set_candidate_count", "grid_entropy"],
        },
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="wa30",
        game_id=WA30_GAME_ID,
        target_level=1,
        prior_level=0,
        baseline_actions=71,
        n_levels=9,
        survey_rank=5,
        selection_mode="best_headroom_unattempted_non_sc25",
        selection_reason=(
            "selected wa30 L1 because it is the highest-headroom unattempted non-sc25 game "
            "with medium survey difficulty, 9 levels, a local fixture, and hardened set-encoder routing available"
        ),
        headroom_score=2019,
        excluded_game_prefixes=exp.SOLVED_PREFIXES_BEFORE_4275,
    )


def _outcome(*, advanced: bool) -> SolverOutcome:
    return SolverOutcome(
        target_game=WA30_GAME_ID,
        target_level=1,
        prior_level=0,
        final_level_completed=1 if advanced else 0,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        executed_real_env_actions=33 if advanced else 0,
        exploration_actions_used=33,
        observed_transition_count=33,
        action_plan=exp.WA30_L1_ACTION_PLAN if advanced else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "hardened-set-encoder-route", "retained": advanced},
            {"phase": "act", "levels_completed": 1 if advanced else 0},
        ],
        solver_trace={"world_model": "GameGraph", "candidate_route": "wa30_l1_three_box_target_strip"},
        failure_reason="" if advanced else "no_verifier_routed_level_up_candidate",
    )


def test_req_phase4_069_spec_declares_exp4275_contract() -> None:
    """REQ-PHASE4-069: OpenSpec declares the Exp 4275 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-069" in spec
    assert "SCENARIO-PHASE4-069" in spec
    assert "experiment_4275_arc_incremental_progress_new_game.json" in spec
    assert "blocked_arc_fixtures_missing" in spec
    assert "sc25-635fd71a" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_069_selects_wa30_as_best_unattempted_non_sc25_headroom() -> None:
    """REQ-PHASE4-069: selection skips sc25 and solved games, then picks WA30."""

    target = select_best_headroom_unattempted_game(
        _survey(),
        _baselines(),
        _prior_best(),
        _wall(),
        _set_encoder(),
    )

    assert target == _target()

    with pytest.raises(ValueError, match="Exp 4249 prior best evidence unavailable"):
        select_best_headroom_unattempted_game(_survey(), _baselines(), {**_prior_best(), "real_env_confirmed": False}, _wall(), _set_encoder())
    with pytest.raises(ValueError, match="Exp 4261 sc25 wall evidence unavailable"):
        select_best_headroom_unattempted_game(_survey(), _baselines(), _prior_best(), {**_wall(), "target_game": WA30_GAME_ID}, _set_encoder())
    with pytest.raises(ValueError, match="hardened set-encoder routing artifact unavailable"):
        select_best_headroom_unattempted_game(_survey(), _baselines(), _prior_best(), _wall(), {"model_specs": {"status": "missing"}})
    with pytest.raises(ValueError, match="no unattempted non-sc25 headroom candidate"):
        select_best_headroom_unattempted_game(_survey(), {"sc25": (SC25_GAME_ID, [1])}, _prior_best(), _wall(), _set_encoder())


def test_scenario_phase4_069_checksum_is_deterministic_and_trajectory_bound() -> None:
    """SCENARIO-PHASE4-069: reproducibility checksum binds route inputs and trajectory."""

    specs = make_model_specs(_target(), _set_encoder())
    checksum = compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_best_artifact=_prior_best(),
        wall_artifact=_wall(),
        set_encoder_artifact=_set_encoder(),
        random_seed=4275,
    )

    assert len(checksum) == 64
    assert checksum == compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_best_artifact=_prior_best(),
        wall_artifact=_wall(),
        set_encoder_artifact=_set_encoder(),
        random_seed=4275,
    )
    assert checksum != compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=False),
        model_specs=specs,
        prior_best_artifact=_prior_best(),
        wall_artifact=_wall(),
        set_encoder_artifact=_set_encoder(),
        random_seed=4275,
    )


def test_scenario_phase4_069_artifact_schema_accepts_success_complete_and_blocked() -> None:
    """SCENARIO-PHASE4-069: only real-env-confirmed non-sc25 evidence increments levels."""

    success = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_best(),
        _wall(),
        _set_encoder(),
        random_seed=4275,
        duration_s=0.2,
    )

    assert success["honest_verdict"] == "success: incremental_progress_wa30-ee6fef47_advanced_to_L1_total20"
    assert success["total_levels"] == PRIOR_TOTAL_LEVELS + 1
    assert success["levels_completed"] == 1
    assert success["game_advanced"] == WA30_GAME_ID
    assert success["target_game"] != SC25_GAME_ID
    assert success["model_specs"]["trm_training"] is False
    assert success["model_specs"]["hardened_set_encoder_routing"]["source_experiment"] == 4244
    assert artifact_schema_errors(success) == []

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_best(),
        _wall(),
        _set_encoder(),
        random_seed=4275,
        duration_s=0.2,
    )
    assert no_advance["honest_verdict"].startswith("complete: incremental_progress_no_advance_wa30-ee6fef47_L1")
    assert no_advance["total_levels"] == PRIOR_TOTAL_LEVELS
    assert no_advance["levels_completed"] == 0
    assert no_advance["game_advanced"] == "none"
    assert no_advance["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_advance) == []

    blocked = blocked_artifact(target_game="none", target_level=0, random_seed=4275, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert blocked["acceptance_gate_passed"] is True
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_069_schema_rejects_fabricated_or_sc25_increment() -> None:
    """SCENARIO-PHASE4-069: schema rejects self-reported and sc25 progress."""

    fabricated = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_best(),
        _wall(),
        _set_encoder(),
        random_seed=4275,
        duration_s=0.0,
    )
    fabricated["real_env_confirmed"] = False
    fabricated["levels_completed"] = 0
    fabricated["total_levels"] = PRIOR_TOTAL_LEVELS
    fabricated["game_advanced"] = "none"
    fabricated["verifier_validated"] = False
    fabricated["action_plan"] = []

    errors = artifact_schema_errors(fabricated)

    assert any("real_env_confirmed must be true for success" in err for err in errors)
    assert any("levels_completed must be one for scoped success" in err for err in errors)
    assert any("total_levels must be 20 for scoped success" in err for err in errors)
    assert any("game_advanced must equal target_game for success" in err for err in errors)
    assert any("verifier_validated must be true for success" in err for err in errors)
    assert any("success requires a real action_plan" in err for err in errors)

    sc25 = {**fabricated, "real_env_confirmed": True, "verifier_validated": True, "action_plan": [{"action": 1}], "total_levels": 20, "levels_completed": 1, "target_game": SC25_GAME_ID, "game_advanced": SC25_GAME_ID}
    assert any("success target_game must not be sc25" in err for err in artifact_schema_errors(sc25))

    malformed = {**fabricated, "honest_verdict": 4275, "random_seed": "4275", "reproducibility_checksum": "bad", "model_specs": []}
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors(malformed))
    assert any("random_seed must be a bare int" in err for err in artifact_schema_errors(malformed))
    assert any("reproducibility_checksum must be a sha256 hex string" in err for err in artifact_schema_errors(malformed))
    assert any("model_specs must be a dict" in err for err in artifact_schema_errors(malformed))


def test_req_phase4_069_defensive_helpers_and_baseline_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PHASE4-069: preconditions and fixture helpers fail closed."""

    fixture = tmp_path / "environment_files" / "wa30" / "ee6fef47"
    fixture.mkdir(parents=True)
    fixture.joinpath("metadata.json").write_text(
        json.dumps({"game_id": WA30_GAME_ID, "baseline_actions": [71, 119]}),
        encoding="utf-8",
    )
    assert load_environment_baselines(tmp_path / "environment_files") == {"wa30": (WA30_GAME_ID, [71, 119])}

    monkeypatch.setattr(exp, "REPO", tmp_path)
    assert exp._fixture_available("not-a-game-id") is False
    assert exp._fixture_available(WA30_GAME_ID) is False
    fixture.joinpath("wa30.py").write_text("# marker\n", encoding="utf-8")
    assert exp._fixture_available(WA30_GAME_ID) is True

    monkeypatch.delattr(exp.world_model, "GameGraph", raising=False)
    with pytest.raises(RuntimeError, match="offline ARC world-model solver import unavailable"):
        exp._require_offline_solver()


def test_scenario_phase4_069_real_wa30_l1_route_confirms_level_counter() -> None:
    """SCENARIO-PHASE4-069: the WA30 L1 route is confirmed by the real offline env."""

    outcome = exp._run_wa30_l1_frontier(_target(), _set_encoder())

    assert outcome.advanced is True
    assert outcome.final_level_completed == 1
    assert outcome.executed_real_env_actions == 33
    assert outcome.observed_transition_count == 33
    assert outcome.action_plan == exp.WA30_L1_ACTION_PLAN
    assert any(row.get("phase") == "hardened-set-encoder-route" and row.get("retained") is True for row in outcome.phase_trace)


def test_scenario_phase4_069_runner_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-069: runner writes solver-derived non-sc25 level evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(json.dumps(_survey()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4249_arc_incremental_progress.json").write_text(json.dumps(_prior_best()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4261_arc_incremental_progress.json").write_text(json.dumps(_wall()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json").write_text(json.dumps(_set_encoder()), encoding="utf-8")
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_run_selected_frontier", lambda target, set_encoder: _outcome(advanced=True))

    artifact = exp.run(write=True)

    assert artifact["total_levels"] == 20
    assert artifact["levels_completed"] == 1
    assert artifact["game_advanced"] == WA30_GAME_ID
    written = json.loads((tmp_path / "results" / "experiment_4275_arc_incremental_progress_new_game.json").read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_phase4_069_runner_blocks_or_converts_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-069: missing inputs block; solver exceptions do not fabricate progress."""

    (tmp_path / "results").mkdir()
    monkeypatch.setattr(exp, "REPO", tmp_path)
    blocked = exp.run(write=True)
    assert blocked["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert blocked["total_levels"] == PRIOR_TOTAL_LEVELS

    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(json.dumps(_survey()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4249_arc_incremental_progress.json").write_text(json.dumps(_prior_best()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4261_arc_incremental_progress.json").write_text(json.dumps(_wall()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json").write_text(json.dumps(_set_encoder()), encoding="utf-8")
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())

    def _raise_solver_error(target: TargetSelection, set_encoder: dict[str, object]) -> SolverOutcome:
        raise ValueError("boom")

    monkeypatch.setattr(exp, "_run_selected_frontier", _raise_solver_error)
    no_advance = exp.run(write=False)

    assert no_advance["honest_verdict"].startswith("complete: incremental_progress_no_advance_wa30-ee6fef47_L1")
    assert no_advance["levels_completed"] == 0
    assert no_advance["real_env_confirmed"] is False


def test_req_phase4_069_entrypoint_exists() -> None:
    """REQ-PHASE4-069: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4275_arc_incremental_progress_new_game.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4275_arc_incremental_progress_new_game" in entrypoint.read_text(encoding="utf-8")
