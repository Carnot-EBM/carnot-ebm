"""Tests for Exp 4261 ARC-AGI-3 incremental progress headroom attempt.

Spec refs: REQ-PHASE4-067, SCENARIO-PHASE4-067.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4261_arc_incremental_progress as exp
from carnot.experiment_4261_arc_incremental_progress import (
    PRIOR_TOTAL_LEVELS,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_FIELD_PRINCIPLES,
    R11L_GAME_ID,
    SC25_GAME_ID,
    SolverOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    compute_reproducibility_checksum,
    make_model_specs,
    select_best_headroom_target,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _survey() -> dict[str, object]:
    return {"per_game_surveys": [{"game": "r11l"}, {"game": "sc25"}]}


def _prior_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4249_arc_incremental_progress",
        "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L5_total19",
        "target_game": SC25_GAME_ID,
        "target_level": 5,
        "total_levels_solved": 19,
        "levels_completed": 5,
        "new_levels_solved_this_task": 1,
        "real_env_confirmed": True,
        "verifier_validated": True,
        "action_plan": [{"action": 6, "kind": "spell_select", "spell": "fibcey", "x": 4, "y": 23}],
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=6,
        prior_level=5,
        baseline_actions=50,
        selection_mode="sc25_l6_after_exp4249_L5",
        selection_reason="selected sc25 L6 because Exp 4249 banked SC25 L5 and local metadata exposes a sixth baseline",
    )


def _outcome(*, advanced: bool) -> SolverOutcome:
    return SolverOutcome(
        target_game=SC25_GAME_ID,
        target_level=6,
        prior_level=5,
        final_level_completed=6 if advanced else 5,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        replay_actions_used=260,
        executed_real_env_actions=18 if advanced else 0,
        exploration_actions_used=41,
        action_plan=[{"action": 1, "kind": "move"}] if advanced else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 5},
            {"phase": "explore", "source": "copied_env_sc25_L6_margin_route"},
            {"phase": "margin-triggered-verify", "retained": advanced, "margin": 0.42 if advanced else -0.1},
            {"phase": "act", "levels_completed": 6 if advanced else 5},
        ],
        solver_trace={"world_model": "GameGraph", "candidate_count": 3},
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_067_spec_declares_exp4261_contract() -> None:
    """REQ-PHASE4-067: OpenSpec declares the Exp 4261 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-067" in spec
    assert "SCENARIO-PHASE4-067" in spec
    assert "experiment_4261_arc_incremental_progress.json" in spec
    assert "blocked_arc_fixtures_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_067_selects_best_local_headroom_after_exp4249() -> None:
    """REQ-PHASE4-067: target selection prefers SC25 L6 when local headroom exists."""

    baselines = {
        "r11l": (R11L_GAME_ID, [22, 33, 51, 26, 52, 49]),
        "sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143, 50]),
    }

    assert select_best_headroom_target(_survey(), baselines, _prior_artifact()) == _target()
    assert select_best_headroom_target(
        _survey(),
        {"r11l": (R11L_GAME_ID, [22, 33, 51, 26, 52, 49])},
        _prior_artifact(),
    ) == TargetSelection(
        game="r11l",
        game_id=R11L_GAME_ID,
        target_level=5,
        prior_level=4,
        baseline_actions=52,
        selection_mode="r11l_l5_survey_headroom_fallback",
        selection_reason="selected r11l L5 fallback because SC25 L6 local headroom was unavailable",
    )

    with pytest.raises(ValueError, match=".393 Exp 4249 success evidence unavailable"):
        select_best_headroom_target(_survey(), baselines, {**_prior_artifact(), "real_env_confirmed": False})
    with pytest.raises(ValueError, match="no local headroom candidate"):
        select_best_headroom_target(_survey(), {"sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143])}, _prior_artifact())


def test_scenario_phase4_067_checksum_is_deterministic_and_trajectory_bound() -> None:
    """SCENARIO-PHASE4-067: reproducibility checksum binds inputs and trajectory."""

    target = _target()
    specs = make_model_specs(target)
    checksum = compute_reproducibility_checksum(
        target=target,
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_artifact=_prior_artifact(),
        random_seed=4261,
    )

    assert len(checksum) == 64
    assert checksum == compute_reproducibility_checksum(
        target=target,
        outcome=_outcome(advanced=True),
        model_specs=specs,
        prior_artifact=_prior_artifact(),
        random_seed=4261,
    )
    assert checksum != compute_reproducibility_checksum(
        target=target,
        outcome=_outcome(advanced=False),
        model_specs=specs,
        prior_artifact=_prior_artifact(),
        random_seed=4261,
    )


def test_scenario_phase4_067_artifact_schema_accepts_success_complete_and_blocked() -> None:
    """SCENARIO-PHASE4-067: success increments one level; no-advance stays terminal and honest."""

    success = build_artifact(_outcome(advanced=True), _target(), _prior_artifact(), random_seed=4261, duration_s=0.2)

    assert success["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L6_total20"
    assert success["total_levels"] == PRIOR_TOTAL_LEVELS + 1
    assert success["levels_completed"] == 1
    assert success["game_advanced"] == SC25_GAME_ID
    assert success["real_env_confirmed"] is True
    assert success["acceptance_gate_passed"] is True
    assert success["model_specs"]["trm_training"] is False
    assert artifact_schema_errors(success) == []

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_artifact(),
        random_seed=4261,
        duration_s=0.2,
    )
    assert no_advance["honest_verdict"].startswith("complete: incremental_progress_no_advance_sc25-635fd71a_L6")
    assert no_advance["total_levels"] == PRIOR_TOTAL_LEVELS
    assert no_advance["levels_completed"] == 0
    assert no_advance["game_advanced"] == "none"
    assert no_advance["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_advance) == []

    blocked = blocked_artifact(target_game="none", target_level=0, random_seed=4261, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert blocked["acceptance_gate_passed"] is True
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_067_schema_rejects_fabricated_increment() -> None:
    """SCENARIO-PHASE4-067: schema rejects success fields without real-env confirmation."""

    fabricated = build_artifact(_outcome(advanced=True), _target(), _prior_artifact(), random_seed=4261, duration_s=0.0)
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
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4261}))


def test_scenario_phase4_067_schema_reports_all_terminal_contract_errors() -> None:
    """SCENARIO-PHASE4-067: schema errors are explicit for malformed terminal artifacts."""

    no_advance = build_artifact(
        _outcome(advanced=False),
        _target(),
        _prior_artifact(),
        random_seed=4261,
        duration_s=0.0,
    )
    malformed = {
        **no_advance,
        "honest_verdict": "later maybe",
        "total_levels": "19",
        "levels_completed": "0",
        "random_seed": "4261",
        "game_advanced": 4261,
        "reproducibility_checksum": "not-a-sha",
        "model_specs": [],
        "field_principles": [],
    }

    errors = artifact_schema_errors(malformed)

    assert any("honest_verdict must be terminal-prefixed" in err for err in errors)
    assert any("total_levels must be a bare int" in err for err in errors)
    assert any("levels_completed must be a bare int" in err for err in errors)
    assert any("random_seed must be a bare int" in err for err in errors)
    assert any("game_advanced must be a string" in err for err in errors)
    assert any("reproducibility_checksum must be a sha256 hex string" in err for err in errors)
    assert any("model_specs must be a dict" in err for err in errors)
    assert any("field_principles must be a dict" in err for err in errors)

    wrong_principles = {**no_advance, "field_principles": {"honest_verdict": "wrong"}}
    assert any("field_principles missing exact honest_verdict" in err for err in artifact_schema_errors(wrong_principles))

    bad_complete = {
        **no_advance,
        "total_levels": 20,
        "levels_completed": 1,
        "game_advanced": SC25_GAME_ID,
        "real_env_confirmed": True,
    }
    complete_errors = artifact_schema_errors(bad_complete)
    assert any("total_levels must remain 19 for no-advance" in err for err in complete_errors)
    assert any("levels_completed must be zero for no-advance" in err for err in complete_errors)
    assert any('game_advanced must be "none" for no-advance' in err for err in complete_errors)
    assert any("real_env_confirmed must be false for no-advance" in err for err in complete_errors)

    bad_blocked = {
        **blocked_artifact(target_game="none", target_level=0, random_seed=4261, duration_s=0.0),
        "total_levels": 20,
        "levels_completed": 1,
        "game_advanced": SC25_GAME_ID,
    }
    blocked_errors = artifact_schema_errors(bad_blocked)
    assert any("total_levels must remain 19 for blocked verdict" in err for err in blocked_errors)
    assert any("levels_completed must be zero for blocked verdict" in err for err in blocked_errors)
    assert any('game_advanced must be "none" for blocked verdict' in err for err in blocked_errors)


def test_req_phase4_067_defensive_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PHASE4-067: precondition and internal consistency guards fail closed."""

    monkeypatch.setattr(exp.previous, "load_environment_baselines", lambda root: {"sc25": (SC25_GAME_ID, [])})
    assert exp.load_environment_baselines(Path("/tmp/unused")) == {"sc25": (SC25_GAME_ID, [])}

    monkeypatch.delattr(exp.world_model, "GameGraph", raising=False)
    with pytest.raises(RuntimeError, match="offline ARC world-model solver import unavailable"):
        exp._require_offline_solver()


def test_req_phase4_067_internal_schema_guard_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PHASE4-067: artifact builders reject internally malformed artifacts."""

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])

    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game="none", target_level=0, random_seed=4261, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(advanced=True), _target(), _prior_artifact(), random_seed=4261, duration_s=0.0)


def test_scenario_phase4_067_private_frontier_and_fixture_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-067: fixture and frontier helpers preserve closed-world routing."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    assert exp._fixture_available("not-a-game-id") is False
    (tmp_path / "environment_files" / "sc25" / "635fd71a").mkdir(parents=True)
    (tmp_path / "environment_files" / "sc25" / "635fd71a" / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "environment_files" / "sc25" / "635fd71a" / "sc25.py").write_text("# fixture\n", encoding="utf-8")
    assert exp._fixture_available(SC25_GAME_ID) is True

    monkeypatch.setattr(exp, "_run_sc25_l6_frontier", lambda target, prior: _outcome(advanced=True))
    assert exp._run_selected_frontier(_target(), _prior_artifact()).advanced is True

    fallback = TargetSelection(
        game="r11l",
        game_id=R11L_GAME_ID,
        target_level=5,
        prior_level=4,
        baseline_actions=52,
        selection_mode="r11l_l5_survey_headroom_fallback",
        selection_reason="selected r11l L5 fallback because SC25 L6 local headroom was unavailable",
    )
    assert exp._run_selected_frontier(fallback, _prior_artifact()).failure_reason == "fallback_frontier_not_attempted_this_window"


def test_scenario_phase4_067_runner_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-067: runner writes solver-derived real-env evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(json.dumps(_survey()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4249_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {"sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143, 50])})
    monkeypatch.setattr(exp, "_run_selected_frontier", lambda target, prior: _outcome(advanced=True))

    artifact = exp.run(write=True)

    assert artifact["total_levels"] == 20
    assert artifact["levels_completed"] == 1
    assert artifact["game_advanced"] == SC25_GAME_ID
    written = json.loads((tmp_path / "results" / "experiment_4261_arc_incremental_progress.json").read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_phase4_067_runner_blocks_when_fixtures_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-067: missing fixtures stop with blocked_arc_fixtures_missing."""

    (tmp_path / "results").mkdir()
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert artifact["total_levels"] == PRIOR_TOTAL_LEVELS
    assert artifact["levels_completed"] == 0
    assert (tmp_path / "results" / "experiment_4261_arc_incremental_progress.json").exists()


def test_scenario_phase4_067_runner_blocks_when_selected_fixture_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-067: selected target fixture absence produces a blocked artifact."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(json.dumps(_survey()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4249_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {"sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143, 50])})
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: False)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_fixtures_missing"
    assert artifact["target_game"] == SC25_GAME_ID
    assert artifact["target_level"] == 6


def test_scenario_phase4_067_runner_converts_solver_exception_to_no_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-067: solver adapter exceptions do not fabricate progress."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(json.dumps(_survey()), encoding="utf-8")
    (tmp_path / "results" / "experiment_4249_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {"sc25": (SC25_GAME_ID, [36, 6, 32, 83, 143, 50])})

    def _raise_solver_error(target: TargetSelection, prior: dict[str, object]) -> SolverOutcome:
        raise ValueError("boom")

    monkeypatch.setattr(exp, "_run_selected_frontier", _raise_solver_error)

    artifact = exp.run(write=False)

    assert artifact["honest_verdict"].startswith("complete: incremental_progress_no_advance_sc25-635fd71a_L6")
    assert artifact["levels_completed"] == 0
    assert artifact["real_env_confirmed"] is False


def test_req_phase4_067_entrypoint_exists() -> None:
    """REQ-PHASE4-067: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4261_arc_incremental_progress.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4261_arc_incremental_progress" in entrypoint.read_text(encoding="utf-8")
