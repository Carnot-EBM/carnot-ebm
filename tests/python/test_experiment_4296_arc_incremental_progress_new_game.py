"""Tests for Exp 4296 ARC-AGI-3 offline different-game incremental progress.

Spec refs: REQ-PHASE4-071, SCENARIO-PHASE4-071.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.experiment_4296_arc_incremental_progress_new_game as exp
from carnot.experiment_4296_arc_incremental_progress_new_game import (
    LS20_GAME_ID,
    PRIOR_TOTAL_LEVELS,
    R11L_GAME_ID,
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
        "top_pick": "r11l",
        "ranked_targets": [{"game": "r11l"}, {"game": "ls20"}, {"game": "wa30"}, {"game": "sc25"}],
        "per_game_surveys": [
            {
                "game": "r11l",
                "is_spatial_planning": False,
                "n_levels": 6,
                "win_difficulty": "medium",
            },
            {"game": "ls20", "is_spatial_planning": True, "n_levels": 7, "win_difficulty": "hard"},
            {
                "game": "wa30",
                "is_spatial_planning": True,
                "n_levels": 9,
                "win_difficulty": "medium",
            },
            {
                "game": "sc25",
                "is_spatial_planning": False,
                "n_levels": 6,
                "win_difficulty": "medium",
            },
            {
                "game": "lp85",
                "is_spatial_planning": False,
                "n_levels": 8,
                "win_difficulty": "medium",
            },
        ],
    }


def _baselines() -> dict[str, tuple[str, list[int]]]:
    return {
        "r11l": (R11L_GAME_ID, [22, 33, 51, 26, 52, 49]),
        "ls20": (LS20_GAME_ID, [22, 123, 73, 84, 96, 192, 186]),
        "wa30": (WA30_GAME_ID, [71, 119, 183, 98, 368, 68, 79, 442, 415]),
        "sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143, 50]),
        "lp85": ("lp85-305b61c3", [17, 60, 80, 150, 80, 80, 80, 80]),
    }


def _prior_ls20() -> dict[str, object]:
    return {
        "experiment": "experiment_4285_arc_incremental_progress_new_game",
        "honest_verdict": "success: incremental_progress_ls20-9607627b_advanced_to_L1_total21",
        "total_levels": 21,
        "total_levels_solved": 21,
        "levels_completed": 1,
        "new_levels_solved_this_task": 1,
        "game_advanced": LS20_GAME_ID,
        "target_game": LS20_GAME_ID,
        "target_level": 1,
        "real_env_confirmed": True,
        "verifier_validated": True,
    }


def _prior_wa30() -> dict[str, object]:
    return {
        "experiment": "experiment_4275_arc_incremental_progress_new_game",
        "honest_verdict": "success: incremental_progress_wa30-ee6fef47_advanced_to_L1_total20",
        "total_levels": 20,
        "total_levels_solved": 20,
        "levels_completed": 1,
        "game_advanced": WA30_GAME_ID,
        "target_game": WA30_GAME_ID,
        "target_level": 1,
        "real_env_confirmed": True,
        "verifier_validated": True,
    }


def _sc25_wall() -> dict[str, object]:
    return {
        "experiment": "experiment_4261_arc_incremental_progress",
        "honest_verdict": "complete: incremental_progress_no_advance_sc25-635fd71a_L6_no_verifier_validated_level_up_candidate",
        "total_levels": 19,
        "total_levels_solved": 19,
        "levels_completed": 0,
        "game_advanced": "none",
        "target_game": SC25_GAME_ID,
        "target_level": 6,
        "real_env_confirmed": False,
    }


def _set_encoder() -> dict[str, object]:
    return {
        "random_seed": 4244,
        "model_type": "standardized_deepsets_context_temperature_calibrated",
        "model_specs": {
            "architecture": "deepsets_pooled_context_set_encoder",
            "status": "trained",
            "feature_set": ["vote_weight", "set_candidate_count", "grid_entropy"],
        },
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="r11l",
        game_id=R11L_GAME_ID,
        target_level=1,
        prior_level=0,
        baseline_actions=22,
        n_levels=6,
        survey_rank=0,
        selection_mode="survey_top_pick_non_ls20_non_wa30_non_sc25",
        selection_reason=(
            "selected r11l L1 because the 25-game survey marks it as the top directly observable "
            "non-spatial headroom target, local fixtures are present, and hardened set-encoder routing is ready"
        ),
        headroom_score=2078,
        excluded_game_prefixes=exp.EXCLUDED_GAME_PREFIXES,
    )


def _outcome(*, advanced: bool) -> SolverOutcome:
    return SolverOutcome(
        target_game=R11L_GAME_ID,
        target_level=1,
        prior_level=0,
        final_level_completed=1 if advanced else 0,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        executed_real_env_actions=4 if advanced else 0,
        exploration_actions_used=2,
        observed_transition_count=4 if advanced else 0,
        action_plan=[
            {"action": 6, "x": 7, "y": 36, "role": "select_piece"},
            {"action": 6, "x": 34, "y": 20, "role": "place_piece"},
            {"action": 6, "x": 27, "y": 59, "role": "select_piece"},
            {"action": 6, "x": 42, "y": 20, "role": "place_piece"},
        ]
        if advanced
        else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "hardened-set-encoder-route", "retained": advanced},
            {"phase": "act", "levels_completed": 1 if advanced else 0},
        ],
        solver_trace={"world_model": "GameGraph", "candidate_route": "r11l_l1_click_select_place"},
        failure_reason="" if advanced else "no_verifier_routed_level_up_candidate",
    )


def test_req_phase4_071_spec_declares_exp4296_contract() -> None:
    """REQ-PHASE4-071: OpenSpec declares the Exp 4296 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-071" in spec
    assert "SCENARIO-PHASE4-071" in spec
    assert "experiment_4296_arc_incremental_progress_new_game.json" in spec
    assert "blocked_arc_fixtures_missing" in spec
    assert "ls20-9607627b" in spec
    assert "wa30-ee6fef47" in spec
    assert "sc25-635fd71a" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_071_selects_r11l_top_pick_after_recent_attempts_are_excluded() -> None:
    """REQ-PHASE4-071: selection skips ls20, wa30, and sc25 while honoring survey headroom."""

    target = select_best_headroom_unattempted_game(
        _survey(),
        _baselines(),
        _prior_ls20(),
        _prior_wa30(),
        _sc25_wall(),
        _set_encoder(),
    )

    assert target == _target()
    assert target.game not in {"ls20", "wa30", "sc25"}

    with pytest.raises(ValueError, match="Exp 4285 ls20 progress evidence unavailable"):
        select_best_headroom_unattempted_game(
            _survey(),
            _baselines(),
            {**_prior_ls20(), "real_env_confirmed": False},
            _prior_wa30(),
            _sc25_wall(),
            _set_encoder(),
        )
    with pytest.raises(ValueError, match="Exp 4275 wa30 progress evidence unavailable"):
        select_best_headroom_unattempted_game(
            _survey(),
            _baselines(),
            _prior_ls20(),
            {**_prior_wa30(), "target_game": LS20_GAME_ID},
            _sc25_wall(),
            _set_encoder(),
        )
    with pytest.raises(ValueError, match="Exp 4261 sc25 wall evidence unavailable"):
        select_best_headroom_unattempted_game(
            _survey(),
            _baselines(),
            _prior_ls20(),
            _prior_wa30(),
            {**_sc25_wall(), "real_env_confirmed": True},
            _set_encoder(),
        )
    with pytest.raises(ValueError, match="hardened set-encoder routing artifact unavailable"):
        select_best_headroom_unattempted_game(
            _survey(),
            _baselines(),
            _prior_ls20(),
            _prior_wa30(),
            _sc25_wall(),
            {"model_specs": {"status": "missing"}},
        )
    with pytest.raises(
        ValueError, match="no unattempted non-ls20 non-wa30 non-sc25 headroom candidate"
    ):
        select_best_headroom_unattempted_game(
            _survey(),
            {"r11l": (R11L_GAME_ID, [])},
            _prior_ls20(),
            _prior_wa30(),
            _sc25_wall(),
            _set_encoder(),
        )


def test_scenario_phase4_071_checksum_is_deterministic_and_trajectory_bound() -> None:
    """SCENARIO-PHASE4-071: reproducibility checksum binds route inputs and trajectory."""

    specs = make_model_specs(_target(), _set_encoder())
    checksum = compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_ls20_artifact=_prior_ls20(),
        prior_wa30_artifact=_prior_wa30(),
        sc25_wall_artifact=_sc25_wall(),
        set_encoder_artifact=_set_encoder(),
        random_seed=4296,
    )

    assert len(checksum) == 64
    assert checksum == compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_ls20_artifact=_prior_ls20(),
        prior_wa30_artifact=_prior_wa30(),
        sc25_wall_artifact=_sc25_wall(),
        set_encoder_artifact=_set_encoder(),
        random_seed=4296,
    )
    assert checksum != compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=False),
        model_specs=specs,
        prior_ls20_artifact=_prior_ls20(),
        prior_wa30_artifact=_prior_wa30(),
        sc25_wall_artifact=_sc25_wall(),
        set_encoder_artifact=_set_encoder(),
        random_seed=4296,
    )


def test_scenario_phase4_071_artifact_schema_accepts_success_complete_and_blocked() -> None:
    """SCENARIO-PHASE4-071: only real-env-confirmed non-recent evidence increments levels."""

    success = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_ls20(),
        _prior_wa30(),
        _sc25_wall(),
        _set_encoder(),
        random_seed=4296,
        duration_s=0.2,
    )

    assert (
        success["honest_verdict"]
        == "success: incremental_progress_r11l-495a7899_advanced_to_L1_total22"
    )
    assert success["total_levels"] == PRIOR_TOTAL_LEVELS + 1
    assert success["levels_completed"] == 1
    assert success["game_advanced"] == R11L_GAME_ID
    assert success["target_game"] not in {LS20_GAME_ID, WA30_GAME_ID, SC25_GAME_ID}
    assert success["model_specs"]["trm_training"] is False
    assert success["model_specs"]["hardened_set_encoder_routing"]["source_experiment"] == 4244
    assert artifact_schema_errors(success) == []

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_ls20(),
        _prior_wa30(),
        _sc25_wall(),
        _set_encoder(),
        random_seed=4296,
        duration_s=0.2,
    )
    assert no_advance["honest_verdict"].startswith(
        "complete: incremental_progress_no_advance_r11l-495a7899_L1"
    )
    assert no_advance["total_levels"] == PRIOR_TOTAL_LEVELS
    assert no_advance["levels_completed"] == 0
    assert no_advance["game_advanced"] == "none"
    assert no_advance["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_advance) == []

    blocked = blocked_artifact(target_game="none", target_level=0, random_seed=4296, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert blocked["acceptance_gate_passed"] is True
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_071_schema_rejects_fabricated_or_excluded_increment() -> None:
    """SCENARIO-PHASE4-071: schema rejects self-reported, ls20, wa30, and sc25 progress."""

    fabricated = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_ls20(),
        _prior_wa30(),
        _sc25_wall(),
        _set_encoder(),
        random_seed=4296,
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
    assert any("total_levels must be 22 for scoped success" in err for err in errors)
    assert any("game_advanced must equal target_game for success" in err for err in errors)
    assert any("verifier_validated must be true for success" in err for err in errors)
    assert any("success requires a real action_plan" in err for err in errors)

    sc25 = {
        **fabricated,
        "real_env_confirmed": True,
        "verifier_validated": True,
        "action_plan": [{"action": 1}],
        "total_levels": 22,
        "levels_completed": 1,
        "target_game": SC25_GAME_ID,
        "game_advanced": SC25_GAME_ID,
    }
    wa30 = {**sc25, "target_game": WA30_GAME_ID, "game_advanced": WA30_GAME_ID}
    ls20 = {**sc25, "target_game": LS20_GAME_ID, "game_advanced": LS20_GAME_ID}
    assert any(
        "success target_game must not be ls20, wa30, or sc25" in err
        for err in artifact_schema_errors(sc25)
    )
    assert any(
        "success target_game must not be ls20, wa30, or sc25" in err
        for err in artifact_schema_errors(wa30)
    )
    assert any(
        "success target_game must not be ls20, wa30, or sc25" in err
        for err in artifact_schema_errors(ls20)
    )

    malformed = {
        **fabricated,
        "honest_verdict": 4296,
        "random_seed": "4296",
        "reproducibility_checksum": "bad",
        "model_specs": [],
    }
    assert any(
        "honest_verdict must be a string" in err for err in artifact_schema_errors(malformed)
    )
    assert any("random_seed must be a bare int" in err for err in artifact_schema_errors(malformed))
    assert any(
        "reproducibility_checksum must be a sha256 hex string" in err
        for err in artifact_schema_errors(malformed)
    )
    assert any("model_specs must be a dict" in err for err in artifact_schema_errors(malformed))

    malformed_prefix = {
        **fabricated,
        "honest_verdict": "invalid",
        "game_advanced": [],
        "field_principles": [],
    }
    assert any(
        "honest_verdict must be terminal-prefixed" in err
        for err in artifact_schema_errors(malformed_prefix)
    )
    assert any(
        "game_advanced must be a string" in err for err in artifact_schema_errors(malformed_prefix)
    )
    assert any(
        "field_principles must be a dict" in err for err in artifact_schema_errors(malformed_prefix)
    )

    wrong_principle = {
        **fabricated,
        "field_principles": {**REQUIRED_FIELD_PRINCIPLES, "total_levels": "wrong"},
    }
    assert any(
        "field_principles missing exact total_levels" in err
        for err in artifact_schema_errors(wrong_principle)
    )

    assert any("missing required field total_levels" in err for err in artifact_schema_errors({}))

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_ls20(),
        _prior_wa30(),
        _sc25_wall(),
        _set_encoder(),
        random_seed=4296,
        duration_s=0.0,
    )
    no_advance.update(
        {
            "total_levels": 22,
            "levels_completed": 1,
            "game_advanced": R11L_GAME_ID,
            "real_env_confirmed": True,
            "target_game": LS20_GAME_ID,
        }
    )
    no_advance_errors = artifact_schema_errors(no_advance)
    assert any("total_levels must remain 21 for no-advance" in err for err in no_advance_errors)
    assert any("levels_completed must be zero for no-advance" in err for err in no_advance_errors)
    assert any('game_advanced must be "none" for no-advance' in err for err in no_advance_errors)
    assert any(
        "real_env_confirmed must be false for no-advance" in err for err in no_advance_errors
    )
    assert any(
        "no-advance target_game must not be ls20, wa30, or sc25" in err for err in no_advance_errors
    )

    blocked = blocked_artifact(target_game="none", target_level=0, random_seed=4296, duration_s=0.0)
    blocked.update({"total_levels": 22, "levels_completed": 1, "game_advanced": R11L_GAME_ID})
    blocked_errors = artifact_schema_errors(blocked)
    assert any("total_levels must remain 21 for blocked verdict" in err for err in blocked_errors)
    assert any("levels_completed must be zero for blocked verdict" in err for err in blocked_errors)
    assert any('game_advanced must be "none" for blocked verdict' in err for err in blocked_errors)


def test_req_phase4_071_defensive_helpers_and_baseline_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PHASE4-071: preconditions and fixture helpers fail closed."""

    fixture = tmp_path / "environment_files" / "r11l" / "495a7899"
    fixture.mkdir(parents=True)
    fixture.joinpath("metadata.json").write_text(
        json.dumps({"game_id": R11L_GAME_ID, "baseline_actions": [22, "bad", 33]}),
        encoding="utf-8",
    )
    bad = tmp_path / "environment_files" / "bad" / "00000000"
    bad.mkdir(parents=True)
    bad.joinpath("metadata.json").write_text("{bad json", encoding="utf-8")
    nodash = tmp_path / "environment_files" / "nodash" / "00000000"
    nodash.mkdir(parents=True)
    nodash.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "nodash", "baseline_actions": [1]}), encoding="utf-8"
    )
    assert load_environment_baselines(tmp_path / "environment_files") == {
        "r11l": (R11L_GAME_ID, [22, 33])
    }

    monkeypatch.setattr(exp, "REPO", tmp_path)
    assert exp._fixture_available("badgameid") is False
    assert exp._fixture_available(R11L_GAME_ID) is False
    fixture.joinpath("r11l.py").write_text("# marker\n", encoding="utf-8")
    assert exp._fixture_available(R11L_GAME_ID) is True
    assert exp._hardened_set_encoder_ready({"model_specs": []}) is False

    monkeypatch.delattr(exp.world_model, "GameGraph", raising=False)
    with pytest.raises(RuntimeError, match="offline ARC world-model solver import unavailable"):
        exp._require_offline_solver()


def test_req_phase4_071_internal_error_paths_and_route_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-071: schema builders and dispatch helpers expose closed failure paths."""

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        blocked_artifact(target_game="none", target_level=0, random_seed=4296, duration_s=0.0)
    with pytest.raises(ValueError, match="forced"):
        build_artifact(
            _outcome(advanced=True),
            _target(),
            _prior_ls20(),
            _prior_wa30(),
            _sc25_wall(),
            _set_encoder(),
            random_seed=4296,
            duration_s=0.0,
        )
    monkeypatch.undo()

    failed_route = exp._verify_hardened_set_encoder_route(
        target=_target(),
        set_encoder_artifact={"model_specs": []},
        predicted_final_level=0,
        observed_transition_count=0,
    )
    assert failed_route["retained"] is False
    assert failed_route["score"] == 0.0

    fallback = exp._run_selected_frontier(
        TargetSelection(
            game="lp85",
            game_id="lp85-305b61c3",
            target_level=1,
            prior_level=0,
            baseline_actions=17,
            n_levels=8,
            survey_rank=1,
            selection_mode="test",
            selection_reason="test",
            headroom_score=1,
            excluded_game_prefixes=exp.EXCLUDED_GAME_PREFIXES,
        ),
        _set_encoder(),
    )
    assert fallback.advanced is False
    assert fallback.failure_reason == "selected_frontier_adapter_unavailable"

    monkeypatch.setattr(
        exp, "_run_r11l_l1_frontier", lambda target, set_encoder: _outcome(advanced=True)
    )
    assert exp._run_selected_frontier(_target(), _set_encoder()).advanced is True


def test_scenario_phase4_071_real_r11l_l1_route_confirms_level_counter() -> None:
    """SCENARIO-PHASE4-071: the R11L L1 route is confirmed by the real offline env."""

    outcome = exp._run_r11l_l1_frontier(_target(), _set_encoder())

    assert outcome.advanced is True
    assert outcome.final_level_completed == 1
    assert outcome.executed_real_env_actions == len(outcome.action_plan)
    assert outcome.observed_transition_count == len(outcome.action_plan)
    assert outcome.action_plan[0]["role"] == "select_piece"
    assert any(
        row.get("phase") == "hardened-set-encoder-route" and row.get("retained") is True
        for row in outcome.phase_trace
    )


def test_scenario_phase4_071_runner_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-071: runner writes solver-derived non-recent level evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(_survey()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4285_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_ls20()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4275_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_wa30()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4261_arc_incremental_progress.json").write_text(
        json.dumps(_sc25_wall()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json").write_text(
        json.dumps(_set_encoder()), encoding="utf-8"
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(
        exp, "_run_selected_frontier", lambda target, set_encoder: _outcome(advanced=True)
    )

    artifact = exp.run(write=True)

    assert artifact["total_levels"] == 22
    assert artifact["levels_completed"] == 1
    assert artifact["game_advanced"] == R11L_GAME_ID
    written = json.loads(
        (tmp_path / "results" / "experiment_4296_arc_incremental_progress_new_game.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_phase4_071_runner_blocks_or_converts_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-071: missing inputs block; solver exceptions do not fabricate progress."""

    (tmp_path / "results").mkdir()
    monkeypatch.setattr(exp, "REPO", tmp_path)
    blocked = exp.run(write=True)
    assert blocked["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert blocked["total_levels"] == PRIOR_TOTAL_LEVELS

    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(_survey()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4285_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_ls20()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4275_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_wa30()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4261_arc_incremental_progress.json").write_text(
        json.dumps(_sc25_wall()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json").write_text(
        json.dumps(_set_encoder()), encoding="utf-8"
    )
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {})
    blocked_empty_baselines = exp.run(write=False)
    assert blocked_empty_baselines["honest_verdict"] == "blocked_arc_fixtures_missing"

    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: False)
    blocked_missing_fixture = exp.run(write=False)
    assert blocked_missing_fixture["honest_verdict"] == "blocked_arc_fixtures_missing"

    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)

    def _raise_solver_error(
        target: TargetSelection, set_encoder: dict[str, object]
    ) -> SolverOutcome:
        raise ValueError("boom")

    monkeypatch.setattr(exp, "_run_selected_frontier", _raise_solver_error)
    no_advance = exp.run(write=False)

    assert no_advance["honest_verdict"].startswith(
        "complete: incremental_progress_no_advance_r11l-495a7899_L1"
    )
    assert no_advance["levels_completed"] == 0
    assert no_advance["real_env_confirmed"] is False


def test_req_phase4_071_entrypoint_exists() -> None:
    """REQ-PHASE4-071: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4296_arc_incremental_progress_new_game.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4296_arc_incremental_progress_new_game" in entrypoint.read_text(
        encoding="utf-8"
    )


def test_req_phase4_071_main_prints_terminal_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-PHASE4-071: CLI main delegates to the preconditioned runner."""

    monkeypatch.setattr(
        sys, "argv", ["experiment_4296_arc_incremental_progress_new_game.py", "--seed", "7"]
    )
    monkeypatch.setattr(
        exp, "run", lambda *, seed, write: {"honest_verdict": f"complete: seed_{seed}_{write}"}
    )

    exp.main()

    assert "-> complete: seed_7_True" in capsys.readouterr().out
