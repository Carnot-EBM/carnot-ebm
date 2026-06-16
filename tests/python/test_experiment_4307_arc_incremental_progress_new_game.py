"""Tests for Exp 4307 ARC-AGI-3 offline new-game incremental progress.

Spec refs: REQ-PHASE4-072, SCENARIO-PHASE4-072.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.experiment_4307_arc_incremental_progress_new_game as exp
from carnot.experiment_4307_arc_incremental_progress_new_game import (
    PRIOR_TOTAL_LEVELS,
    R11L_GAME_ID,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_FIELD_PRINCIPLES,
    RE86_GAME_ID,
    SolverOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_arc_env_unreachable_artifact,
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
        "ranked_targets": [{"game": "r11l"}, {"game": "sc25"}, {"game": "lp85"}],
        "per_game_surveys": [
            {"game": "r11l", "is_spatial_planning": False, "n_levels": 6, "win_difficulty": "medium"},
            {"game": "lp85", "is_spatial_planning": False, "n_levels": 8, "win_difficulty": "medium"},
            {"game": "re86", "is_spatial_planning": True, "n_levels": 8, "win_difficulty": "hard"},
            {"game": "ka59", "is_spatial_planning": True, "n_levels": 8, "win_difficulty": "hard"},
        ],
    }


def _baselines() -> dict[str, tuple[str, list[int]]]:
    return {
        "r11l": (R11L_GAME_ID, [22, 33, 51, 26, 52, 49]),
        "lp85": ("lp85-305b61c3", [17, 38, 31, 16, 41, 60, 26, 159]),
        "re86": (RE86_GAME_ID, [26, 42, 86, 108, 189, 139, 424, 241]),
        "ka59": ("ka59-38d34dbb", [28, 45, 60]),
    }


def _prior_4296() -> dict[str, object]:
    return {
        "experiment": "experiment_4296_arc_incremental_progress_new_game",
        "honest_verdict": "success: incremental_progress_r11l-495a7899_advanced_to_L1_total22",
        "total_levels": 22,
        "total_levels_solved": 22,
        "levels_completed": 1,
        "game_advanced": R11L_GAME_ID,
        "target_game": R11L_GAME_ID,
        "target_level": 1,
        "real_env_confirmed": True,
        "verifier_validated": True,
    }


def _set_encoder_4291() -> dict[str, object]:
    return {
        "random_seed": 4291,
        "cross_generator_holds": True,
        "non_degenerate_guards_pass": True,
        "verifier_is_oracle": False,
        "model_specs": {
            "set_encoder_config": {
                "architecture": "deepsets_pooled_context_set_encoder",
                "status": "trained",
                "feature_set": ["vote_weight", "grid_entropy"],
            }
        },
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="re86",
        game_id=RE86_GAME_ID,
        target_level=1,
        prior_level=0,
        baseline_actions=26,
        n_levels=8,
        survey_rank=3,
        selection_mode="best_headroom_unattempted_non_recent_lowest_baseline",
        selection_reason=(
            "selected re86 L1 because it is the lowest-baseline unattempted survey game "
            "after excluding r11l/ls20/wa30/sc25 and banked prior prefixes, with a local "
            "fixture and hardened Exp 4291 set-encoder routing available"
        ),
        headroom_score=974,
        excluded_game_prefixes=exp.EXCLUDED_GAME_PREFIXES,
    )


def _preconditions(*, env: bool = True) -> list[dict[str, object]]:
    return [
        {"resource": "offline_solver_import", "available": True, "detail": "GameGraph import OK"},
        {"resource": "arc3_win_condition_survey", "available": True, "detail": "loaded"},
        {"resource": "prior_exp4296_progress", "available": True, "detail": "total_levels=22"},
        {"resource": "hardened_set_encoder_routing_exp4291", "available": True, "detail": "ready"},
        {
            "resource": "offline_arc_env",
            "available": env,
            "detail": "reset levels_completed=0" if env else "reset failed",
        },
    ]


def _outcome(*, advanced: bool) -> SolverOutcome:
    return SolverOutcome(
        target_game=RE86_GAME_ID,
        target_level=1,
        prior_level=0,
        final_level_completed=1 if advanced else 0,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        executed_real_env_actions=3 if advanced else 0,
        exploration_actions_used=1,
        observed_transition_count=3 if advanced else 0,
        action_plan=[{"action": 4, "role": "move"}, {"action": 6, "x": 2, "y": 3, "role": "click"}]
        if advanced
        else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "hardened-set-encoder-route", "retained": advanced},
            {"phase": "act", "levels_completed": 1 if advanced else 0},
        ],
        solver_trace={"world_model": "GameGraph", "candidate_route": "re86_l1_candidate"},
        failure_reason="" if advanced else "selected_frontier_adapter_unavailable",
    )


def test_req_phase4_072_spec_declares_exp4307_contract() -> None:
    """REQ-PHASE4-072: OpenSpec declares the Exp 4307 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-072" in spec
    assert "SCENARIO-PHASE4-072" in spec
    assert "experiment_4307_arc_incremental_progress_new_game.json" in spec
    assert "blocked_arc_env_unreachable" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_072_selects_re86_after_recent_and_banked_prefixes() -> None:
    """REQ-PHASE4-072: selection skips recent attempts and banked prefixes."""

    target = select_best_headroom_unattempted_game(
        _survey(), _baselines(), _prior_4296(), _set_encoder_4291()
    )

    assert target == _target()
    assert target.game not in {"r11l", "ls20", "wa30", "sc25", "lp85"}

    with pytest.raises(ValueError, match="Exp 4296 progress evidence unavailable"):
        select_best_headroom_unattempted_game(
            _survey(), _baselines(), {**_prior_4296(), "total_levels": 21}, _set_encoder_4291()
        )
    with pytest.raises(ValueError, match="hardened Exp 4291 set-encoder routing artifact unavailable"):
        select_best_headroom_unattempted_game(
            _survey(), _baselines(), _prior_4296(), {"model_specs": {"status": "missing"}}
        )
    with pytest.raises(ValueError, match="no unattempted non-r11l non-ls20 non-wa30 non-sc25"):
        select_best_headroom_unattempted_game(
            _survey(), {"r11l": (R11L_GAME_ID, [1])}, _prior_4296(), _set_encoder_4291()
        )
    with pytest.raises(ValueError, match="no unattempted non-r11l non-ls20 non-wa30 non-sc25"):
        select_best_headroom_unattempted_game(
            _survey(),
            {"re86": (RE86_GAME_ID, []), "ka59": ("ka59-38d34dbb", [])},
            _prior_4296(),
            _set_encoder_4291(),
            excluded_prefixes=(),
        )


def test_scenario_phase4_072_checksum_is_deterministic_and_trajectory_bound() -> None:
    """SCENARIO-PHASE4-072: reproducibility checksum binds inputs and env outcome."""

    specs = make_model_specs(_target(), _set_encoder_4291())
    checksum = compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=False),
        model_specs=specs,
        prior_4296_artifact=_prior_4296(),
        set_encoder_artifact=_set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
    )

    assert len(checksum) == 64
    assert checksum == compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=False),
        model_specs=specs,
        prior_4296_artifact=_prior_4296(),
        set_encoder_artifact=_set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
    )
    assert checksum != compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_4296_artifact=_prior_4296(),
        set_encoder_artifact=_set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
    )


def test_scenario_phase4_072_artifact_schema_accepts_success_no_advance_and_blocked() -> None:
    """SCENARIO-PHASE4-072: only real-env-confirmed evidence increments total levels."""

    success = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_4296(),
        _set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
        duration_s=0.2,
    )

    assert success["honest_verdict"] == "success: incremental_progress_re86-8af5384d_advanced_to_L1_total23"
    assert success["total_levels"] == PRIOR_TOTAL_LEVELS + 1
    assert success["levels_completed"] == 1
    assert success["game_advanced"] == RE86_GAME_ID
    assert success["real_env_confirmed"] is True
    assert success["model_specs"]["hardened_set_encoder_routing"]["source_experiment"] == 4291
    assert artifact_schema_errors(success) == []

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_4296(),
        _set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
        duration_s=0.2,
    )
    assert no_advance["honest_verdict"].startswith(
        "complete: incremental_progress_no_advance_re86-8af5384d_L1"
    )
    assert no_advance["total_levels"] == PRIOR_TOTAL_LEVELS
    assert no_advance["levels_completed"] == 0
    assert no_advance["game_advanced"] == RE86_GAME_ID
    assert no_advance["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_advance) == []

    blocked = blocked_arc_env_unreachable_artifact(
        target_game="none",
        target_level=0,
        reason="reset_failed",
        preconditions_checked=_preconditions(env=False),
        random_seed=4307,
        duration_s=0.0,
    )
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked["total_levels"] == PRIOR_TOTAL_LEVELS
    assert blocked["levels_completed"] == 0
    assert blocked["game_advanced"] == "none"
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_072_schema_rejects_fabricated_or_recent_success() -> None:
    """SCENARIO-PHASE4-072: malformed success artifacts cannot inflate the count."""

    fabricated = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_4296(),
        _set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
        duration_s=0.0,
    )
    fabricated.update(
        {
            "real_env_confirmed": False,
            "levels_completed": 0,
            "total_levels": PRIOR_TOTAL_LEVELS,
            "verifier_validated": False,
            "action_plan": [],
            "target_game": R11L_GAME_ID,
            "game_advanced": R11L_GAME_ID,
        }
    )

    errors = artifact_schema_errors(fabricated)

    assert any("real_env_confirmed must be true for success" in err for err in errors)
    assert any("levels_completed must be one for scoped success" in err for err in errors)
    assert any("total_levels must be 23 for scoped success" in err for err in errors)
    assert any("success target_game must not be r11l, ls20, wa30, or sc25" in err for err in errors)
    assert any("verifier_validated must be true for success" in err for err in errors)
    assert any("success requires a real action_plan" in err for err in errors)

    malformed = {
        **fabricated,
        "honest_verdict": 4307,
        "random_seed": "4307",
        "reproducibility_checksum": "bad",
        "model_specs": [],
        "preconditions_checked": [],
        "game_advanced": [],
        "total_levels": 21,
    }
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors(malformed))
    assert any("random_seed must be a bare int" in err for err in artifact_schema_errors(malformed))
    assert any("game_advanced must be a string" in err for err in artifact_schema_errors(malformed))
    assert any("total_levels must be monotonic from 22" in err for err in artifact_schema_errors(malformed))
    assert any(
        "reproducibility_checksum must be a sha256 hex string" in err
        for err in artifact_schema_errors(malformed)
    )
    assert any("model_specs must be a dict" in err for err in artifact_schema_errors(malformed))
    assert any(
        "preconditions_checked must include offline_arc_env and survey load"
        in err
        for err in artifact_schema_errors(malformed)
    )

    wrong_principle = {
        **fabricated,
        "honest_verdict": "invalid",
        "field_principles": {**REQUIRED_FIELD_PRINCIPLES, "total_levels": "wrong"},
    }
    assert any(
        "honest_verdict must be terminal-prefixed" in err
        for err in artifact_schema_errors(wrong_principle)
    )
    assert any(
        "field_principles missing exact total_levels" in err
        for err in artifact_schema_errors(wrong_principle)
    )
    assert any(
        "field_principles must be a dict" in err
        for err in artifact_schema_errors({**fabricated, "field_principles": []})
    )
    assert any("missing required field total_levels" in err for err in artifact_schema_errors({}))

    success_mismatch = {
        **fabricated,
        "real_env_confirmed": True,
        "verifier_validated": True,
        "action_plan": [{"action": 4}],
        "total_levels": 23,
        "levels_completed": 1,
        "target_game": RE86_GAME_ID,
        "game_advanced": "other-00000000",
    }
    assert any(
        "game_advanced must equal target_game for success" in err
        for err in artifact_schema_errors(success_mismatch)
    )

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_4296(),
        _set_encoder_4291(),
        preconditions_checked=_preconditions(),
        random_seed=4307,
        duration_s=0.0,
    )
    no_advance.update(
        {
            "total_levels": 23,
            "levels_completed": 1,
            "game_advanced": "none",
            "target_game": R11L_GAME_ID,
            "real_env_confirmed": True,
        }
    )
    no_advance_errors = artifact_schema_errors(no_advance)
    assert any("total_levels must remain 22 for no-advance" in err for err in no_advance_errors)
    assert any("levels_completed must be zero for no-advance" in err for err in no_advance_errors)
    assert any(
        "game_advanced must keep target_game for no-advance attribution" in err
        for err in no_advance_errors
    )
    assert any(
        "no-advance target_game must not be r11l, ls20, wa30, or sc25" in err
        for err in no_advance_errors
    )
    assert any("real_env_confirmed must be false for no-advance" in err for err in no_advance_errors)

    blocked = blocked_arc_env_unreachable_artifact(
        target_game="none",
        target_level=0,
        reason="reset_failed",
        preconditions_checked=_preconditions(env=False),
        random_seed=4307,
        duration_s=0.0,
    )
    blocked.update({"total_levels": 23, "levels_completed": 1, "game_advanced": RE86_GAME_ID})
    blocked_errors = artifact_schema_errors(blocked)
    assert any("total_levels must remain 22 for blocked verdict" in err for err in blocked_errors)
    assert any("levels_completed must be zero for blocked verdict" in err for err in blocked_errors)
    assert any('game_advanced must be "none" for blocked verdict' in err for err in blocked_errors)


def test_req_phase4_072_defensive_helpers_and_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PHASE4-072: local fixture helpers and import checks fail closed."""

    fixture = tmp_path / "environment_files" / "re86" / "8af5384d"
    fixture.mkdir(parents=True)
    fixture.joinpath("metadata.json").write_text(
        json.dumps({"game_id": RE86_GAME_ID, "baseline_actions": [26, "bad", 42]}),
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
        "re86": (RE86_GAME_ID, [26, 42])
    }

    monkeypatch.setattr(exp, "REPO", tmp_path)
    assert exp._fixture_available("badgameid") is False
    assert exp._fixture_available(RE86_GAME_ID) is False
    fixture.joinpath("re86.py").write_text("# marker\n", encoding="utf-8")
    assert exp._fixture_available(RE86_GAME_ID) is True
    assert exp._reason_slug("No Route-Found") == "no_route_found"
    assert exp._set_encoder_config({"model_specs": []}) == {}

    class Frame:
        levels_completed = None
        level_completed = 2

    class Env:
        class _game:
            _current_level_index = 1

    assert exp._levels_completed(Frame(), Env()) == 2

    monkeypatch.delattr(exp.world_model, "GameGraph", raising=False)
    with pytest.raises(RuntimeError, match="offline ARC world-model solver import unavailable"):
        exp._require_offline_solver()


def test_req_phase4_072_internal_error_paths_and_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-072: artifact builders and no-adapter frontier fail closed."""

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        blocked_arc_env_unreachable_artifact(
            target_game="none",
            target_level=0,
            reason="bad",
            preconditions_checked=_preconditions(env=False),
            random_seed=4307,
            duration_s=0.0,
        )
    with pytest.raises(ValueError, match="forced"):
        build_artifact(
            _outcome(advanced=False),
            _target(),
            _prior_4296(),
            _set_encoder_4291(),
            preconditions_checked=_preconditions(),
            random_seed=4307,
            duration_s=0.0,
        )
    monkeypatch.undo()

    failed = exp._failed_outcome(_target(), "unit_test_failure", final_level=0)
    assert failed.advanced is False
    assert failed.failure_reason == "unit_test_failure"

    class DummyArcade:
        def make(self, game_id: str) -> object:
            assert game_id == RE86_GAME_ID
            return self

        def reset(self) -> object:
            return [[0, 0], [0, 1]]

    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: DummyArcade())
    frame, env, level = exp._reset_offline_env(_target())
    assert frame == [[0, 0], [0, 1]]
    assert env.__class__ is DummyArcade
    assert level == 0

    outcome = exp._run_selected_frontier(_target(), _set_encoder_4291(), [[0, 0], [0, 1]], object())
    assert outcome.advanced is False
    assert outcome.failure_reason == "selected_frontier_adapter_unavailable"
    assert outcome.phase_trace[1]["retained"] is False


def test_scenario_phase4_072_runner_writes_no_advance_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-072: runner writes solver-derived no-advance evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(_survey()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4296_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_4296()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4291_arcgen_cross_generator_nondegenerate.json").write_text(
        json.dumps(_set_encoder_4291()), encoding="utf-8"
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "_reset_offline_env", lambda target: (object(), object(), 0))
    monkeypatch.setattr(
        exp, "_run_selected_frontier", lambda target, set_encoder, frame, env: _outcome(advanced=False)
    )

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"].startswith("complete: incremental_progress_no_advance")
    assert artifact["total_levels"] == PRIOR_TOTAL_LEVELS
    assert artifact["levels_completed"] == 0
    assert artifact["game_advanced"] == RE86_GAME_ID
    written = json.loads(
        (tmp_path / "results" / "experiment_4307_arc_incremental_progress_new_game.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_phase4_072_runner_blocks_when_env_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-072: unreachable offline env produces the required blocker."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(_survey()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4296_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_4296()), encoding="utf-8"
    )
    (tmp_path / "results" / "experiment_4291_arcgen_cross_generator_nondegenerate.json").write_text(
        json.dumps(_set_encoder_4291()), encoding="utf-8"
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)

    def _raise_env(target: TargetSelection) -> tuple[object, object, int]:
        raise RuntimeError("offline env down")

    monkeypatch.setattr(exp, "_reset_offline_env", _raise_env)
    artifact = exp.run(write=False)

    assert artifact["honest_verdict"] == "blocked_arc_env_unreachable"
    assert artifact["total_levels"] == PRIOR_TOTAL_LEVELS
    assert any(
        row["resource"] == "offline_arc_env" and row["available"] is False
        for row in artifact["preconditions_checked"]
    )

    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {})
    artifact_empty_baselines = exp.run(write=True)
    assert artifact_empty_baselines["honest_verdict"] == "blocked_arc_env_unreachable"

    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: False)
    artifact_missing_fixture = exp.run(write=False)
    assert artifact_missing_fixture["honest_verdict"] == "blocked_arc_env_unreachable"

    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "_reset_offline_env", lambda target: (object(), object(), 0))

    def _raise_solver(
        target: TargetSelection, set_encoder: dict[str, object], frame: object, env: object
    ) -> SolverOutcome:
        raise RuntimeError("solver broke")

    monkeypatch.setattr(exp, "_run_selected_frontier", _raise_solver)
    no_advance = exp.run(write=False)
    assert no_advance["honest_verdict"].startswith(
        "complete: incremental_progress_no_advance_re86-8af5384d_L1"
    )
    assert no_advance["levels_completed"] == 0


def test_req_phase4_072_entrypoint_and_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-PHASE4-072: the required command path delegates to the runner."""

    entrypoint = REPO / "results" / "experiment_4307_arc_incremental_progress_new_game.py"
    assert entrypoint.exists()
    assert "carnot.experiment_4307_arc_incremental_progress_new_game" in entrypoint.read_text(
        encoding="utf-8"
    )

    monkeypatch.setattr(
        sys, "argv", ["experiment_4307_arc_incremental_progress_new_game.py", "--seed", "7"]
    )
    monkeypatch.setattr(
        exp,
        "run",
        lambda *, seed, write: {"honest_verdict": f"complete: seed_{seed}_{write}"},
    )

    exp.main()

    assert "-> complete: seed_7_True" in capsys.readouterr().out
