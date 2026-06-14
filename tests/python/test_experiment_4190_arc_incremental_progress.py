"""Tests for Exp 4190 ARC-AGI-3 hardened GAP-4 incremental progress.

Spec refs: REQ-PHASE4-055, SCENARIO-PHASE4-055.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import carnot.experiment_4190_arc_incremental_progress as exp
from carnot.experiment_4190_arc_incremental_progress import (
    INFERENCE_SUBSTRATE,
    LP85_GAME_ID,
    PRIOR_TOTAL_LEVELS_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    FrontierOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    execute_plan_until_level,
    gap4_hardening_ready,
    select_deeper_level_target,
    validate_hardened_gap4_heldout_replay,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


@dataclass
class FakeAction:
    ACTION6: int = 6


class FakeEnv:
    def __init__(self, start_level: int = 2, advance_after: int = 3) -> None:
        self._level = start_level
        self._advance_after = advance_after
        self._steps = 0

    def step(self, action: int, data: dict[str, int]) -> SimpleNamespace:
        assert action == FakeAction.ACTION6
        assert "x" in data and "y" in data
        self._steps += 1
        if self._steps >= self._advance_after:
            self._level = 3
        return SimpleNamespace(levels_completed=self._level)


def _prior_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4179_arc_incremental_progress",
        "honest_verdict": "success: incremental_progress_lp85-305b61c3_advanced_to_L2_total14",
        "target_game": LP85_GAME_ID,
        "target_level": 2,
        "total_levels_solved": 14,
        "levels_completed": 2,
        "real_env_confirmed": True,
        "action_plan": [{"button": "button_A_R", "x": 38, "y": 16}],
    }


def _hardening_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4187_gap4_graded_execution_gate_hardening",
        "honest_verdict": "complete: gap4_graded_relaxation_adds_nothing_on_arc1",
        "vote_aware_guard_blocked_mispromotion": True,
        "gross_recovery_ledger": {"recovered": 4, "lost": 0},
        "gate_fire_count": 16,
        "guard_block_count": 1,
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="lp85",
        game_id=LP85_GAME_ID,
        target_level=3,
        prior_level=2,
        baseline_actions=31,
        selection_mode="deeper_level_after_lp85_L2_success",
        selection_reason="selected lp85 L3 as the next deeper already-solved-game level after Exp 4179 reached L2",
    )


def _outcome(*, advanced: bool) -> FrontierOutcome:
    verification = validate_hardened_gap4_heldout_replay(
        start_level=2,
        final_level=3 if advanced else 2,
        heldout_transition_count=8,
        predicted_level=3,
        gap4_artifact=_hardening_artifact(),
    )
    return FrontierOutcome(
        target_game=LP85_GAME_ID,
        target_level=3,
        prior_level=2,
        final_level_completed=3 if advanced else 2,
        replay_actions_used=13,
        executed_real_env_actions=16 if advanced else 0,
        exploration_actions_used=13,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        verification_decisions=[verification],
        action_plan=[{"button": "button_A_L", "x": 34, "y": 40}] if advanced else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "replay", "levels_completed": 2},
            {"phase": "explore", "buttons_observed": 4},
            {"phase": "induce", "mechanic": "button permutation over visible goals"},
            verification,
            {"phase": "act", "levels_completed": 3 if advanced else 2},
        ],
        induced_mechanic="lp85 L3 observed button-permutation mechanic with visible goal-overlap predicate",
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_055_spec_declares_exp4190_contract() -> None:
    """REQ-PHASE4-055: OpenSpec declares the Exp 4190 terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-055" in spec
    assert "SCENARIO-PHASE4-055" in spec
    assert "experiment_4190_arc_incremental_progress.json" in spec
    assert "lp85-305b61c3" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_055_selects_lp85_l3_after_prior_l2_success() -> None:
    """REQ-PHASE4-055: target selection prefers lp85 L3 over spatial fallback."""

    survey = {"top_pick": "r11l", "ranked_targets": [{"game": "r11l"}, {"game": "lp85"}]}
    baselines = {"lp85": (LP85_GAME_ID, [17, 38, 31])}

    assert select_deeper_level_target(survey, baselines, _prior_artifact(), _hardening_artifact()) == _target()

    bad_prior = {**_prior_artifact(), "levels_completed": 1}
    with pytest.raises(ValueError, match="Exp 4179 L2 success evidence unavailable"):
        select_deeper_level_target(survey, baselines, bad_prior, _hardening_artifact())
    with pytest.raises(ValueError, match="hardened GAP-4 verifier evidence unavailable"):
        select_deeper_level_target(survey, baselines, _prior_artifact(), {"gross_recovery_ledger": {"lost": 1}})
    with pytest.raises(ValueError, match="lp85 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {}, _prior_artifact(), _hardening_artifact())
    with pytest.raises(ValueError, match="lp85 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {"lp85": ("lp85-wrong", [17, 38])}, _prior_artifact(), _hardening_artifact())


def test_scenario_phase4_055_artifacts_validate_success_no_solve_and_blocked() -> None:
    """SCENARIO-PHASE4-055: only hardened-verified real-env evidence increments levels."""

    success = build_artifact(_outcome(advanced=True), _target(), random_seed=4190, duration_s=0.25)

    assert success["honest_verdict"] == "success: incremental_progress_lp85-305b61c3_advanced_to_L3_total15"
    assert success["total_games_solved"] == 13
    assert success["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED + 1
    assert success["levels_completed"] == 3
    assert success["new_levels_solved_this_task"] == 1
    assert success["real_env_confirmed"] is True
    assert success["verifier_validated"] is True
    assert success["requirements"] == REQUIREMENTS
    assert success["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(success) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in success

    no_solve = build_artifact(_outcome(advanced=False), _target(), random_seed=4190, duration_s=0.25)
    assert no_solve["honest_verdict"] == (
        "complete: incremental_progress_no_solve_lp85-305b61c3_L3_no_verifier_validated_level_up_candidate"
    )
    assert no_solve["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert no_solve["new_levels_solved_this_task"] == 0
    assert no_solve["real_env_confirmed"] is False
    assert no_solve["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(target_game=LP85_GAME_ID, target_level=3, random_seed=4190, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_055_hardened_gap4_and_l3_execution_stop() -> None:
    """SCENARIO-PHASE4-055: hardened GAP-4 retained suffixes precede L3 execution."""

    assert gap4_hardening_ready(_hardening_artifact()) is True
    assert gap4_hardening_ready({"gross_recovery_ledger": {"lost": 1}}) is False

    retained = validate_hardened_gap4_heldout_replay(2, 3, 8, 3, gap4_artifact=_hardening_artifact())
    rejected = validate_hardened_gap4_heldout_replay(2, 2, 8, 3, gap4_artifact=_hardening_artifact())
    unready = validate_hardened_gap4_heldout_replay(2, 3, 8, 3, gap4_artifact={})
    assert retained["retained"] is True
    assert retained["energy"] == 0.0
    assert retained["verifier"] == "hardened_gap4_heldout_executed_consistency_deeper_level_replay"
    assert rejected["retained"] is False
    assert unready["retained"] is False
    assert unready["hardened_gap4_ready"] is False

    plan = [
        {"button": "button_A_L", "x": 34, "y": 40},
        {"button": "button_A_L", "x": 34, "y": 40},
        {"button": "button_B_L", "x": 22, "y": 40},
    ]
    final_level, executed, trace = execute_plan_until_level(
        FakeEnv(start_level=2, advance_after=3),
        FakeAction,
        plan,
        prior_level=2,
        target_level=3,
    )
    assert final_level == 3
    assert executed == 3
    assert [step["levels_completed"] for step in trace] == [2, 2, 3]

    assert exp._fixture_available("malformedid") is False


def test_scenario_phase4_055_replay_and_copy_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-055: prior replay and held-out copy validation are deterministic."""

    monkeypatch.setattr(
        exp,
        "_replay_lp85_l1",
        lambda env, action: (SimpleNamespace(levels_completed=1), 2, [{"phase": "replay", "levels_completed": 1}]),
    )
    env = FakeEnv(start_level=1, advance_after=1)
    frontier, actions, trace = exp._replay_prior_lp85_frontier(
        env,
        FakeAction,
        {"action_plan": [{"button": "button_A_R", "x": 38, "y": 16}]},
        _target(),
    )
    assert frontier == 3
    assert actions == 3
    assert trace[-1]["source"] == "banked_exp4179_lp85_L2_replay"

    monkeypatch.setattr(
        exp,
        "_replay_lp85_l1",
        lambda env, action: (SimpleNamespace(levels_completed=2), 1, [{"phase": "replay", "levels_completed": 2}]),
    )
    frontier_ready, actions_ready, trace_ready = exp._replay_prior_lp85_frontier(
        FakeEnv(start_level=2),
        FakeAction,
        {"action_plan": []},
        _target(),
    )
    assert (frontier_ready, actions_ready, trace_ready[-1]["levels_completed"]) == (2, 1, 2)

    copy_env = FakeEnv(start_level=2, advance_after=2)
    copy_env._game = {"level": 2}
    decision = exp._validate_suffix_on_copy(
        copy_env,
        FakeAction,
        start_level=2,
        target_level=3,
        action_plan=[{"button": "button_A_L", "x": 34, "y": 40}, {"button": "button_B_L", "x": 22, "y": 40}],
        gap4_artifact=_hardening_artifact(),
    )
    assert decision["retained"] is True
    assert decision["validated_prefix_transition_count"] == 1
    assert decision["validated_total_transition_count"] == 2
    assert copy_env._game == {"level": 2}


def test_scenario_phase4_055_schema_rejects_fabricated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-055: schema guards reject malformed or fabricated increments."""

    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4190}))
    assert any("honest_verdict must be terminal-prefixed" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("total_levels_solved must be a bare int" in err for err in artifact_schema_errors({"total_levels_solved": True}))
    assert any("real_env_confirmed must be a bare bool" in err for err in artifact_schema_errors({"real_env_confirmed": 1}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4190}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("inference_substrate must equal" in err for err in artifact_schema_errors({"inference_substrate": "wrong"}))
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))
    assert any("field_principles must be a dict" in err for err in artifact_schema_errors({"field_principles": []}))
    assert any(
        "field_principles missing honest_verdict" in err
        for err in artifact_schema_errors({"field_principles": {"total_levels_solved": "x"}})
    )

    bad_success = {
        "honest_verdict": "success: incremental_progress_lp85-305b61c3_advanced_to_L3_total15",
        "target_game": LP85_GAME_ID,
        "target_level": 3,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "new_levels_solved_this_task": 0,
        "levels_completed": 2,
        "real_env_confirmed": False,
        "verifier_validated": False,
        "verification_decisions": [{"retained": False}],
        "action_plan": [],
        "solve_trace": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "requirements": REQUIREMENTS,
        "field_principles": dict(exp.REQUIRED_FIELD_PRINCIPLES),
    }
    errors = artifact_schema_errors(bad_success)
    assert any("real_env_confirmed must be true for success" in err for err in errors)
    assert any("verifier_validated must be true for success" in err for err in errors)
    assert any("total_levels_solved must increment" in err for err in errors)
    assert any("levels_completed must reach target_level" in err for err in errors)
    assert any("success requires a retained hardened GAP-4 verifier decision" in err for err in errors)
    assert any("success requires a validated action_plan" in err for err in errors)

    bad_complete = {**bad_success, "honest_verdict": "complete: incremental_progress_no_solve_lp85-305b61c3_L3_x"}
    bad_complete["real_env_confirmed"] = True
    bad_complete["total_levels_solved"] = PRIOR_TOTAL_LEVELS_SOLVED + 1
    bad_complete["new_levels_solved_this_task"] = 1
    complete_errors = artifact_schema_errors(bad_complete)
    assert any("total_levels_solved must remain" in err for err in complete_errors)
    assert any("new_levels_solved_this_task must be zero" in err for err in complete_errors)
    assert any("real_env_confirmed must be false" in err for err in complete_errors)

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(advanced=True), _target(), random_seed=4190, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game=LP85_GAME_ID, target_level=3, random_seed=4190, duration_s=0.0)


def test_scenario_phase4_055_runner_writes_terminal_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-055: runner writes blocked and real-env-confirmed artifacts."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    missing = exp.run(write=True)
    assert missing["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert (tmp_path / "results" / "experiment_4190_arc_incremental_progress.json").exists()

    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps({"top_pick": "r11l", "ranked_targets": [{"game": "r11l"}, {"game": "lp85"}]}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4179_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json").write_text(
        json.dumps(_hardening_artifact()),
        encoding="utf-8",
    )
    no_fixture = exp.run(write=True)
    assert no_fixture["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir = tmp_path / "environment_files" / "lp85" / "305b61c3"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": LP85_GAME_ID, "baseline_actions": [17, 38, 31]}),
        encoding="utf-8",
    )
    missing_py = exp.run(write=True)
    assert missing_py["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir.joinpath("lp85.py").write_text("# marker\n", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_lp85_frontier", lambda arcade, target, prior, gap4: _outcome(advanced=True))
    success = exp.run(write=True)
    assert success["honest_verdict"] == "success: incremental_progress_lp85-305b61c3_advanced_to_L3_total15"
    written = json.loads((tmp_path / "results" / "experiment_4190_arc_incremental_progress.json").read_text())
    assert written == success

    monkeypatch.setattr(
        exp,
        "_run_lp85_frontier",
        lambda arcade, target, prior, gap4: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    errored = exp.run(write=False)
    assert errored["honest_verdict"].startswith("complete: incremental_progress_no_solve_lp85-305b61c3_L3")
    assert "offline_run_failed_runtimeerror" in errored["honest_verdict"]


def test_results_entrypoint_exists() -> None:
    """REQ-PHASE4-055: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4190_arc_incremental_progress.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4190_arc_incremental_progress" in entrypoint.read_text(encoding="utf-8")
