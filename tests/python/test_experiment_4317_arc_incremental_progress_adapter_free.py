"""Tests for Exp 4317 ARC-AGI-3 adapter-free incremental progress.

Spec refs: REQ-PHASE4-073, SCENARIO-PHASE4-073.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.experiment_4317_arc_incremental_progress_adapter_free as exp
from carnot.experiment_4317_arc_incremental_progress_adapter_free import (
    CD82_GAME_ID,
    PRIOR_TOTAL_LEVELS,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_FIELD_PRINCIPLES,
    SolverOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_arc_env_unreachable_artifact,
    blocked_arc_solver_cannot_act_artifact,
    build_artifact,
    compute_reproducibility_checksum,
    load_environment_baselines,
    load_saved_trajectories,
    make_model_specs,
    select_best_headroom_adapter_free_game,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _survey() -> dict[str, object]:
    return {
        "top_pick": "r11l",
        "ranked_targets": [{"game": "r11l"}, {"game": "sc25"}, {"game": "lp85"}, {"game": "tn36"}],
        "per_game_surveys": [
            {"game": "r11l", "is_spatial_planning": False, "n_levels": 6, "win_difficulty": "medium"},
            {"game": "cd82", "is_spatial_planning": False, "n_levels": 6, "win_difficulty": "medium"},
            {"game": "su15", "is_spatial_planning": True, "n_levels": 9, "win_difficulty": "medium"},
            {"game": "tu93", "is_spatial_planning": True, "n_levels": 9, "win_difficulty": "hard"},
        ],
    }


def _baselines() -> dict[str, tuple[str, list[int]]]:
    return {
        "r11l": ("r11l-495a7899", [22, 33, 51]),
        "cd82": (CD82_GAME_ID, [55, 8, 41, 21, 23, 23]),
        "su15": ("su15-1944f8ab", [22, 42, 26]),
        "tu93": ("tu93-0768757b", [19, 16, 34]),
    }


def _trajectories(*, cd82_level: int = 1) -> dict[str, dict[str, object]]:
    return {
        "cd82": {
            "game": "cd82",
            "reached_level": cd82_level,
            "trajectory": [
                {"action": 3, "data": None},
                {"action": 2, "data": None},
                {"action": 2, "data": None},
                {"action": 4, "data": None},
                {"action": 5, "data": None},
            ],
            "path": "results/arc_explore_trajectory_cd82.json",
        },
        "su15": {
            "game": "su15",
            "reached_level": 1,
            "trajectory": [{"action": 6, "data": {"x": 10, "y": 53}}],
            "path": "results/arc_explore_trajectory_su15.json",
        },
    }


def _prior_4296() -> dict[str, object]:
    return {
        "experiment": "experiment_4296_arc_incremental_progress_new_game",
        "honest_verdict": "success: incremental_progress_r11l-495a7899_advanced_to_L1_total22",
        "total_levels": 22,
        "total_levels_solved": 22,
        "levels_completed": 1,
        "game_advanced": "r11l-495a7899",
        "target_game": "r11l-495a7899",
        "real_env_confirmed": True,
    }


def _prior_4307() -> dict[str, object]:
    return {
        "experiment": "experiment_4307_arc_incremental_progress_new_game",
        "honest_verdict": "complete: incremental_progress_no_advance_re86-8af5384d_L1_selected_frontier_adapter_unavailable",
        "total_levels": 22,
        "levels_completed": 0,
        "exploration_actions_used": 0,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {"kind": "GATE_PASSED_WITHOUT_DATA", "detail": "exploration_actions_used=0"}
        ],
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="cd82",
        game_id=CD82_GAME_ID,
        target_level=1,
        prior_level=0,
        baseline_actions=55,
        n_levels=6,
        survey_rank=4,
        selection_mode="adapter_free_cached_graph_explore_nonspatial_headroom",
        selection_reason=(
            "selected cd82 L1 because it is a non-excluded survey game with a saved "
            "adapter-free graph-explore trajectory, local fixture metadata, and the "
            "highest nonspatial headroom score after the exp4307 adapter-dependency flag"
        ),
        headroom_score=10945,
        trajectory_path="results/arc_explore_trajectory_cd82.json",
        excluded_game_prefixes=exp.EXCLUDED_GAME_PREFIXES,
    )


def _preconditions(*, env: bool = True) -> list[dict[str, object]]:
    return [
        {"resource": "offline_solver_import", "available": True, "detail": "GameGraph import OK"},
        {"resource": "arc_solver_kit", "available": True, "detail": "reproduce gate import OK"},
        {
            "resource": "adapter_free_graph_explore_solver",
            "available": True,
            "detail": "graph_explore_solve_v2 import OK",
        },
        {"resource": "arc3_win_condition_survey", "available": True, "detail": "loaded"},
        {"resource": "prior_exp4296_progress", "available": True, "detail": "total_levels=22"},
        {
            "resource": "prior_exp4307_flag",
            "available": True,
            "detail": "flagged exploration_actions_used=0",
        },
        {
            "resource": "frontier_adapter_dependency_absent",
            "available": True,
            "detail": "no set-encoder or frontier adapter required",
        },
        {
            "resource": "offline_arc_env",
            "available": env,
            "detail": "reset levels_completed=0" if env else "reset failed",
        },
    ]


def _recommendation() -> dict[str, object]:
    return {
        "target_game": "cd82",
        "recommended": [{"game": "r11l", "similarity": 2.0, "solver": "registry"}],
        "general_gotchas": [{"id": "level_on_frame_not_game"}],
    }


def _reproduction_gate(*, reproduced: bool = True, reached: int = 1) -> dict[str, object]:
    return {
        "game": CD82_GAME_ID,
        "reached_level": reached,
        "claimed_level": 1,
        "reproduced": reproduced,
        "mode": "offline_reproduction_gate_no_quota",
    }


def _outcome(*, advanced: bool, explored: int = 5) -> SolverOutcome:
    return SolverOutcome(
        target_game=CD82_GAME_ID,
        target_level=1,
        prior_level=0,
        final_level_completed=1 if advanced else 0,
        real_env_confirmed=advanced,
        offline_reproduced=advanced,
        reproduced_levels=1 if advanced else 0,
        executed_real_env_actions=explored,
        exploration_actions_used=explored,
        observed_transition_count=explored,
        action_plan=list(_trajectories()["cd82"]["trajectory"]),
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "adapter-free-graph-explore", "trajectory_source": "cached"},
            {"phase": "reproduce", "reproduced": advanced},
        ],
        solver_trace={"policy": "graph_explore_v2_shortest_path", "trajectory_source": "cached"},
        reproduction_gate=_reproduction_gate(reproduced=advanced, reached=1 if advanced else 0),
        failure_reason="" if advanced else "trajectory_did_not_advance",
    )


def test_req_phase4_073_spec_declares_exp4317_contract() -> None:
    """REQ-PHASE4-073: OpenSpec declares the Exp 4317 anti-flag contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-073" in spec
    assert "SCENARIO-PHASE4-073" in spec
    assert "experiment_4317_arc_incremental_progress_adapter_free.json" in spec
    assert "exploration_actions_used>0" in spec
    assert "arc_solver_kit.reproduce()" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_073_selects_cd82_adapter_free_and_rejects_bad_inputs() -> None:
    """REQ-PHASE4-073: selection uses adapter-free solved trajectory, not an adapter."""

    target = select_best_headroom_adapter_free_game(
        _survey(), _baselines(), _trajectories(), _prior_4296(), _prior_4307()
    )

    assert target == _target()
    assert target.game not in {"r11l", "ls20", "wa30", "sc25"}

    with pytest.raises(ValueError, match="Exp 4296 progress evidence unavailable"):
        select_best_headroom_adapter_free_game(
            _survey(), _baselines(), _trajectories(), {**_prior_4296(), "total_levels": 21}, _prior_4307()
        )
    with pytest.raises(ValueError, match="Exp 4307 flagged zero-action stall evidence unavailable"):
        select_best_headroom_adapter_free_game(
            _survey(), _baselines(), _trajectories(), _prior_4296(), {**_prior_4307(), "flagged_adversarial": False}
        )
    with pytest.raises(ValueError, match="no adapter-free reproduced trajectory candidate"):
        select_best_headroom_adapter_free_game(
            _survey(), _baselines(), _trajectories(cd82_level=0), _prior_4296(), _prior_4307()
        )
    with pytest.raises(ValueError, match="no adapter-free reproduced trajectory candidate"):
        select_best_headroom_adapter_free_game(
            _survey(), {"r11l": ("r11l-495a7899", [22])}, {}, _prior_4296(), _prior_4307()
        )


def test_scenario_phase4_073_checksum_and_model_specs_are_adapter_free() -> None:
    """SCENARIO-PHASE4-073: checksum binds the trajectory and reproduce result."""

    specs = make_model_specs(_target(), _recommendation())
    checksum = compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_4296_artifact=_prior_4296(),
        prior_4307_artifact=_prior_4307(),
        preconditions_checked=_preconditions(),
        recommendation=_recommendation(),
        random_seed=4317,
    )

    assert specs["frontier_adapter_free"] is True
    assert specs["frontier_adapter_dependency"] == "none"
    assert "set_encoder" not in json.dumps(specs)
    assert len(checksum) == 64
    assert checksum == compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_4296_artifact=_prior_4296(),
        prior_4307_artifact=_prior_4307(),
        preconditions_checked=_preconditions(),
        recommendation=_recommendation(),
        random_seed=4317,
    )
    assert checksum != compute_reproducibility_checksum(
        target=_target(),
        outcome=_outcome(advanced=False),
        model_specs=specs,
        prior_4296_artifact=_prior_4296(),
        prior_4307_artifact=_prior_4307(),
        preconditions_checked=_preconditions(),
        recommendation=_recommendation(),
        random_seed=4317,
    )


def test_scenario_phase4_073_artifact_schema_accepts_success_no_advance_and_blocked() -> None:
    """SCENARIO-PHASE4-073: only explored and reproduced evidence increments total."""

    success = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_4296(),
        _prior_4307(),
        recommendation=_recommendation(),
        preconditions_checked=_preconditions(),
        random_seed=4317,
        duration_s=0.2,
    )

    assert success["honest_verdict"] == "success: adapter_free_incremental_progress_cd82-fb555c5d_advanced_to_L1_total23"
    assert success["total_levels"] == PRIOR_TOTAL_LEVELS + 1
    assert success["levels_completed"] == 1
    assert success["exploration_actions_used"] == 5
    assert success["offline_reproduced"] is True
    assert success["model_specs"]["frontier_adapter_free"] is True
    assert artifact_schema_errors(success) == []

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_4296(),
        _prior_4307(),
        recommendation=_recommendation(),
        preconditions_checked=_preconditions(),
        random_seed=4317,
        duration_s=0.2,
    )
    assert no_advance["honest_verdict"].startswith(
        "complete: adapter_free_no_advance_cd82-fb555c5d_L1"
    )
    assert no_advance["total_levels"] == PRIOR_TOTAL_LEVELS
    assert no_advance["levels_completed"] == 0
    assert no_advance["game_advanced"] == CD82_GAME_ID
    assert no_advance["offline_reproduced"] is False
    assert no_advance["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_advance) == []

    blocked_env = blocked_arc_env_unreachable_artifact(
        target_game="none",
        target_level=0,
        reason="reset_failed",
        preconditions_checked=_preconditions(env=False),
        random_seed=4317,
        duration_s=0.0,
    )
    assert blocked_env["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked_env["total_levels"] == PRIOR_TOTAL_LEVELS
    assert blocked_env["game_advanced"] == "none"
    assert blocked_env["offline_reproduced"] is False
    assert artifact_schema_errors(blocked_env) == []

    blocked_act = blocked_arc_solver_cannot_act_artifact(
        target_game=CD82_GAME_ID,
        target_level=1,
        reason="empty_trajectory",
        preconditions_checked=_preconditions(),
        random_seed=4317,
        duration_s=0.0,
    )
    assert blocked_act["honest_verdict"] == "blocked_arc_solver_cannot_act"
    assert blocked_act["game_advanced"] == "none"
    assert artifact_schema_errors(blocked_act) == []


def test_scenario_phase4_073_schema_rejects_exp4307_zero_action_flag_and_fabrication() -> None:
    """SCENARIO-PHASE4-073: zero-action and unreproduced advances cannot pass."""

    fabricated = build_artifact(
        _outcome(advanced=True),
        _target(),
        _prior_4296(),
        _prior_4307(),
        recommendation=_recommendation(),
        preconditions_checked=_preconditions(),
        random_seed=4317,
        duration_s=0.0,
    )
    fabricated.update(
        {
            "real_env_confirmed": False,
            "offline_reproduced": False,
            "levels_completed": 0,
            "total_levels": PRIOR_TOTAL_LEVELS,
            "exploration_actions_used": 0,
            "action_plan": [],
            "target_game": "r11l-495a7899",
            "game_advanced": "r11l-495a7899",
        }
    )

    errors = artifact_schema_errors(fabricated)

    assert any("exploration_actions_used must be positive for success" in err for err in errors)
    assert any("offline_reproduced must be true for success" in err for err in errors)
    assert any("real_env_confirmed must be true for success" in err for err in errors)
    assert any("levels_completed must be one for scoped success" in err for err in errors)
    assert any("success target_game must not be r11l, ls20, wa30, or sc25" in err for err in errors)
    assert any("success requires a real action_plan" in err for err in errors)

    zero_action_no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_4296(),
        _prior_4307(),
        recommendation=_recommendation(),
        preconditions_checked=_preconditions(),
        random_seed=4317,
        duration_s=0.0,
    )
    zero_action_no_advance["exploration_actions_used"] = 0
    assert any(
        "exploration_actions_used must be positive for no-advance"
        in err
        for err in artifact_schema_errors(zero_action_no_advance)
    )

    malformed = {
        **fabricated,
        "honest_verdict": 4317,
        "random_seed": "4317",
        "reproducibility_checksum": "bad",
        "preconditions_checked": [],
        "offline_reproduced": "true",
        "exploration_actions_used": True,
        "model_specs": {"frontier_adapter_free": False},
        "field_principles": [],
        "total_levels": 21,
        "game_advanced": [],
    }
    malformed_errors = artifact_schema_errors(malformed)
    assert any("honest_verdict must be a string" in err for err in malformed_errors)
    assert any("random_seed must be a bare int" in err for err in malformed_errors)
    assert any("offline_reproduced must be a bare bool" in err for err in malformed_errors)
    assert any("exploration_actions_used must be a bare int" in err for err in malformed_errors)
    assert any("game_advanced must be a string" in err for err in malformed_errors)
    assert any("total_levels must be monotonic from 22" in err for err in malformed_errors)
    assert any("reproducibility_checksum must be a sha256 hex string" in err for err in malformed_errors)
    assert any(
        "preconditions_checked must include offline env, survey, kit, and adapter-free solver"
        in err
        for err in malformed_errors
    )
    assert any("model_specs must declare frontier_adapter_free true" in err for err in malformed_errors)
    assert any("field_principles must be a dict" in err for err in malformed_errors)

    wrong_principle = {
        **fabricated,
        "honest_verdict": "invalid",
        "field_principles": {**REQUIRED_FIELD_PRINCIPLES, "total_levels": "wrong"},
    }
    assert any("honest_verdict must be terminal-prefixed" in err for err in artifact_schema_errors(wrong_principle))
    assert any(
        "field_principles missing exact total_levels"
        in err
        for err in artifact_schema_errors(wrong_principle)
    )
    assert any("missing required field total_levels" in err for err in artifact_schema_errors({}))


def test_req_phase4_073_fixture_loading_and_execution_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PHASE4-073: local fixture helpers and trajectory execution are grounded."""

    fixture = tmp_path / "environment_files" / "cd82" / "fb555c5d"
    fixture.mkdir(parents=True)
    fixture.joinpath("metadata.json").write_text(
        json.dumps({"game_id": CD82_GAME_ID, "baseline_actions": [55, "bad", 8]}),
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
        "cd82": (CD82_GAME_ID, [55, 8])
    }

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_explore_trajectory_cd82.json").write_text(
        json.dumps({"game": "cd82", "reached_level": 1, "trajectory": [{"action": 5, "data": None}]}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "arc_explore_trajectory_bad.json").write_text("{bad", encoding="utf-8")
    assert load_saved_trajectories(tmp_path / "results")["cd82"]["reached_level"] == 1

    monkeypatch.setattr(exp, "REPO", tmp_path)
    assert exp._fixture_available("badgameid") is False
    assert exp._fixture_available(CD82_GAME_ID) is False
    fixture.joinpath("cd82.py").write_text("# marker\n", encoding="utf-8")
    assert exp._fixture_available(CD82_GAME_ID) is True
    assert exp._reason_slug("No Route-Found") == "no_route_found"

    class Frame:
        levels_completed = None
        level_completed = 2

    assert exp._frame_level(Frame()) == 2

    class FakeEnv:
        def __init__(self) -> None:
            self.actions: list[tuple[object, object]] = []

        def reset(self) -> object:
            return {"levels_completed": 0}

        def step(self, action: object, data: object = None, reasoning: object = None) -> object:
            self.actions.append((action, data))
            return {"levels_completed": 1 if len(self.actions) >= 2 else 0}

    monkeypatch.setattr(exp, "_arc_action", lambda action_id: f"ACTION{action_id}")
    outcome = exp.execute_cached_trajectory(
        FakeEnv(),
        _target(),
        [{"action": 3, "data": None}, {"action": 5, "data": {"x": 1}}],
        reproduction_gate=_reproduction_gate(),
    )
    assert outcome.advanced is True
    assert outcome.final_level_completed == 1
    assert outcome.exploration_actions_used == 2

    failed = exp._failed_outcome(_target(), "unit_test_failure", final_level=0, explored=3)
    assert failed.advanced is False
    assert failed.exploration_actions_used == 3


def test_scenario_phase4_073_runner_writes_success_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-073: runner writes a reproduction-gated adapter-free success."""

    _write_required_files(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "recommend_approach", lambda game: _recommendation())
    monkeypatch.setattr(exp.kit, "reproduce", lambda *args, **kwargs: _reproduction_gate())
    monkeypatch.setattr(exp, "_arc_action", lambda action_id: f"ACTION{action_id}")

    class FakeArcade:
        def open_scorecard(self) -> str:
            return "offline-scorecard"

        def make(self, game_id: str, scorecard_id: str | None = None) -> object:
            assert game_id == CD82_GAME_ID
            return FakeEnv()

    class FakeEnv:
        def __init__(self) -> None:
            self.actions = 0

        def reset(self) -> object:
            return {"levels_completed": 0}

        def step(self, action: object, data: object = None, reasoning: object = None) -> object:
            self.actions += 1
            return {"levels_completed": 1 if self.actions >= 5 else 0}

    monkeypatch.setattr(exp.kit, "offline_arcade", lambda: FakeArcade())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"].startswith("success: adapter_free_incremental_progress")
    assert artifact["total_levels"] == 23
    assert artifact["levels_completed"] == 1
    assert artifact["exploration_actions_used"] == 5
    assert artifact["offline_reproduced"] is True
    assert artifact["submitted_to_leaderboard"] is False
    written = json.loads(
        (tmp_path / "results" / "experiment_4317_arc_incremental_progress_adapter_free.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_phase4_073_runner_blocks_and_handles_no_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-073: runner blocks honestly or records real-action no-advance."""

    _write_required_files(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: _baselines())
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "recommend_approach", lambda game: _recommendation())

    def _raise_env(target: TargetSelection) -> tuple[object, object, int]:
        raise RuntimeError("offline env down")

    monkeypatch.setattr(exp, "_reset_offline_env", _raise_env)
    blocked_env = exp.run(write=False)
    assert blocked_env["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked_env["game_advanced"] == "none"

    monkeypatch.setattr(exp, "_reset_offline_env", lambda target: (object(), object(), 0))
    monkeypatch.setattr(exp, "load_saved_trajectories", lambda root: {})
    blocked_act = exp.run(write=False)
    assert blocked_act["honest_verdict"] == "blocked_arc_solver_cannot_act"

    monkeypatch.setattr(exp, "load_saved_trajectories", lambda root: _trajectories())
    monkeypatch.setattr(
        exp,
        "execute_cached_trajectory",
        lambda env, target, trajectory, reproduction_gate: _outcome(advanced=False),
    )
    monkeypatch.setattr(exp.kit, "reproduce", lambda *args, **kwargs: _reproduction_gate(reproduced=False, reached=0))
    no_advance = exp.run(write=False)
    assert no_advance["honest_verdict"].startswith("complete: adapter_free_no_advance_cd82-fb555c5d_L1")
    assert no_advance["exploration_actions_used"] == 5
    assert no_advance["offline_reproduced"] is False


def test_req_phase4_073_internal_error_paths_and_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-PHASE4-073: builders fail closed and the command path delegates to run."""

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        blocked_arc_env_unreachable_artifact(
            target_game="none",
            target_level=0,
            reason="bad",
            preconditions_checked=_preconditions(env=False),
            random_seed=4317,
            duration_s=0.0,
        )
    with pytest.raises(ValueError, match="forced"):
        blocked_arc_solver_cannot_act_artifact(
            target_game=CD82_GAME_ID,
            target_level=1,
            reason="bad",
            preconditions_checked=_preconditions(),
            random_seed=4317,
            duration_s=0.0,
        )
    with pytest.raises(ValueError, match="forced"):
        build_artifact(
            _outcome(advanced=True),
            _target(),
            _prior_4296(),
            _prior_4307(),
            recommendation=_recommendation(),
            preconditions_checked=_preconditions(),
            random_seed=4317,
            duration_s=0.0,
        )
    monkeypatch.undo()

    monkeypatch.delattr(exp.world_model, "GameGraph", raising=False)
    with pytest.raises(RuntimeError, match="offline ARC world-model solver import unavailable"):
        exp._require_adapter_free_solver()
    monkeypatch.setattr(exp.world_model, "GameGraph", object, raising=False)

    entrypoint = REPO / "results" / "experiment_4317_arc_incremental_progress_adapter_free.py"
    assert entrypoint.exists()
    assert "carnot.experiment_4317_arc_incremental_progress_adapter_free" in entrypoint.read_text(
        encoding="utf-8"
    )

    monkeypatch.setattr(
        sys, "argv", ["experiment_4317_arc_incremental_progress_adapter_free.py", "--seed", "7"]
    )
    monkeypatch.setattr(
        exp,
        "run",
        lambda *, seed, write: {"honest_verdict": f"complete: seed_{seed}_{write}"},
    )

    exp.main()

    assert "-> complete: seed_7_True" in capsys.readouterr().out


def _write_required_files(root: Path) -> None:
    (root / "results").mkdir()
    (root / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(_survey()), encoding="utf-8"
    )
    (root / "results" / "experiment_4296_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_4296()), encoding="utf-8"
    )
    (root / "results" / "experiment_4307_arc_incremental_progress_new_game.json").write_text(
        json.dumps(_prior_4307()), encoding="utf-8"
    )
    (root / "results" / "arc_explore_trajectory_cd82.json").write_text(
        json.dumps(_trajectories()["cd82"]), encoding="utf-8"
    )
