"""Tests for Exp 4179 ARC-AGI-3 deeper-level incremental progress.

Spec refs: REQ-PHASE4-054, SCENARIO-PHASE4-054.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import carnot.experiment_4179_arc_incremental_progress as exp
from carnot.experiment_4179_arc_incremental_progress import (
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
    discover_click_buttons,
    plan_observed_suffix,
    select_deeper_level_target,
    validate_gap4_heldout_replay,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


@dataclass
class FakeSprite:
    name: str
    x: int
    y: int
    tags: list[str]


class FakeCamera:
    def display_to_grid(self, x: int, y: int) -> tuple[int, int] | None:
        return (int(x), int(y)) if 0 <= int(x) < 64 and 0 <= int(y) < 64 else None


class FakeLevel:
    def __init__(self, sprites: list[FakeSprite]) -> None:
        self._sprites = sprites

    def get_sprites_by_tag(self, tag: str) -> list[FakeSprite]:
        return [sprite for sprite in self._sprites if tag in sprite.tags]


class FakeAction:
    ACTION6 = 6


class FakeGame:
    def __init__(self) -> None:
        self.level_index = 1
        self.levels_completed = 1
        self.progress = 0
        self.camera = FakeCamera()
        self.current_level = self._level()

    def _level(self) -> FakeLevel:
        return FakeLevel(
            [
                FakeSprite("button_a", 10, 10, ["button_A_R", "sys_click"]),
                FakeSprite("button_c", 20, 20, ["button_C_R", "sys_click"]),
                FakeSprite("goal_1", 1 + self.progress, 1, ["goal"]),
                FakeSprite("goal_2", 5, 5 + self.progress, ["goal"]),
                FakeSprite("piece_1", 4, 1, ["bghvgbtwcb"]),
                FakeSprite("piece_2", 5, 7, ["bghvgbtwcb"]),
            ]
        )

    def click(self, x: int, y: int) -> None:
        tags_by_click = {(10, 10): "button_A_R", (20, 20): "button_C_R"}
        expected = ["button_A_R", "button_C_R", "button_A_R"]
        tag = tags_by_click.get((int(x), int(y)))
        if tag == expected[self.progress]:
            self.progress += 1
        else:
            self.progress = 0
        if self.progress >= len(expected):
            self.levels_completed = 2
            self.level_index = 2
        self.current_level = self._level()


class FakeEnv:
    def __init__(self) -> None:
        self._game = FakeGame()

    def step(self, action: int, data: dict[str, int]) -> SimpleNamespace:
        assert action == FakeAction.ACTION6
        self._game.click(int(data["x"]), int(data["y"]))
        return SimpleNamespace(levels_completed=self._game.levels_completed)


def _target() -> TargetSelection:
    return TargetSelection(
        game="lp85",
        game_id=LP85_GAME_ID,
        target_level=2,
        prior_level=1,
        baseline_actions=38,
        selection_mode="deeper_level_after_strict_nonspatial_exhaustion",
        selection_reason="selected lp85 L2 as the next deeper level after prior r11l L5 and lp85 L2 stalls",
    )


def _outcome(*, advanced: bool) -> FrontierOutcome:
    verification = validate_gap4_heldout_replay(
        start_level=1,
        final_level=2 if advanced else 1,
        heldout_transition_count=2,
        predicted_level=2,
    )
    return FrontierOutcome(
        target_game=LP85_GAME_ID,
        target_level=2,
        prior_level=1,
        final_level_completed=2 if advanced else 1,
        replay_actions_used=5,
        executed_real_env_actions=8 if advanced else 0,
        exploration_actions_used=5,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        verification_decisions=[verification],
        action_plan=[{"button": "button_A_R", "x": 38, "y": 16}] if advanced else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 1},
            {"phase": "explore", "buttons_observed": 6},
            {"phase": "induce", "mechanic": "button permutation over visible goals"},
            verification,
            {"phase": "act", "levels_completed": 2 if advanced else 1},
        ],
        induced_mechanic="lp85 observed button-permutation mechanic with visible goal-overlap predicate",
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_054_spec_declares_exp4179_contract() -> None:
    """REQ-PHASE4-054: OpenSpec declares the Exp 4179 terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-054" in spec
    assert "SCENARIO-PHASE4-054" in spec
    assert "experiment_4179_arc_incremental_progress.json" in spec
    assert "lp85-305b61c3" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_054_selects_lp85_deeper_level_after_r11l_block() -> None:
    """REQ-PHASE4-054: the selected unsolved level prefers lp85 L2 over spatial fallback."""

    survey = {"top_pick": "r11l", "ranked_targets": [{"game": "r11l"}, {"game": "lp85"}]}
    baselines = {"r11l": ("r11l-495a7899", [22, 33, 51, 26, 52]), "lp85": (LP85_GAME_ID, [17, 38])}

    target = select_deeper_level_target(survey, baselines)

    assert target == _target()

    with pytest.raises(ValueError, match="lp85 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {"r11l": ("r11l-495a7899", [22])})


def test_scenario_phase4_054_artifacts_validate_success_no_solve_and_blocked() -> None:
    """SCENARIO-PHASE4-054: only verified real-env evidence increments solved levels."""

    success = build_artifact(_outcome(advanced=True), _target(), random_seed=4179, duration_s=0.25)

    assert success["honest_verdict"] == "success: incremental_progress_lp85-305b61c3_advanced_to_L2_total14"
    assert success["total_games_solved"] == 13
    assert success["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED + 1
    assert success["levels_completed"] == 2
    assert success["new_levels_solved_this_task"] == 1
    assert success["real_env_confirmed"] is True
    assert success["verifier_validated"] is True
    assert success["requirements"] == REQUIREMENTS
    assert success["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(success) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in success

    no_solve = build_artifact(_outcome(advanced=False), _target(), random_seed=4179, duration_s=0.25)
    assert no_solve["honest_verdict"] == (
        "complete: incremental_progress_no_solve_lp85-305b61c3_L2_no_verifier_validated_level_up_candidate"
    )
    assert no_solve["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert no_solve["new_levels_solved_this_task"] == 0
    assert no_solve["real_env_confirmed"] is False
    assert no_solve["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(target_game=LP85_GAME_ID, target_level=2, random_seed=4179, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_054_gap4_and_planner_use_observed_transitions() -> None:
    """SCENARIO-PHASE4-054: GAP-4 retained suffixes come from copied-env observations."""

    retained = validate_gap4_heldout_replay(1, 2, 2, 2)
    rejected = validate_gap4_heldout_replay(1, 1, 2, 2)
    no_heldout = validate_gap4_heldout_replay(1, 2, 0, 2)
    assert retained["retained"] is True
    assert retained["energy"] == 0.0
    assert rejected["retained"] is False
    assert no_heldout["retained"] is False

    env = FakeEnv()
    buttons = discover_click_buttons(env)
    plan, trace = plan_observed_suffix(env, FakeAction, start_level=1, max_depth=5)

    assert [button["button"] for button in buttons] == ["button_A_R", "button_C_R"]
    assert [step["button"] for step in plan] == ["button_A_R", "button_C_R", "button_A_R"]
    assert trace["observed_transition_count"] > 0
    assert trace["expanded_states"] > 0


def test_scenario_phase4_054_schema_rejects_fabricated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-054: schema guards reject malformed or fabricated increments."""

    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4179}))
    assert any("honest_verdict must be terminal-prefixed" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("total_levels_solved must be a bare int" in err for err in artifact_schema_errors({"total_levels_solved": True}))
    assert any("real_env_confirmed must be a bare bool" in err for err in artifact_schema_errors({"real_env_confirmed": 1}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4179}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))
    assert any("field_principles must be a dict" in err for err in artifact_schema_errors({"field_principles": []}))

    bad_success = {
        "honest_verdict": "success: incremental_progress_lp85-305b61c3_advanced_to_L2_total14",
        "target_game": LP85_GAME_ID,
        "target_level": 2,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "new_levels_solved_this_task": 0,
        "levels_completed": 1,
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
    assert any("success requires a retained GAP-4 verifier decision" in err for err in errors)
    assert any("success requires a validated action_plan" in err for err in errors)

    bad_complete = {**bad_success, "honest_verdict": "complete: incremental_progress_no_solve_lp85-305b61c3_L2_x"}
    bad_complete["real_env_confirmed"] = True
    bad_complete["total_levels_solved"] = PRIOR_TOTAL_LEVELS_SOLVED + 1
    complete_errors = artifact_schema_errors(bad_complete)
    assert any("total_levels_solved must remain" in err for err in complete_errors)
    assert any("real_env_confirmed must be false" in err for err in complete_errors)

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(advanced=True), _target(), random_seed=4179, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game=LP85_GAME_ID, target_level=2, random_seed=4179, duration_s=0.0)


def test_scenario_phase4_054_runner_writes_terminal_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-054: runner writes blocked and real-env-confirmed artifacts."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    missing = exp.run(write=True)
    assert missing["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert (tmp_path / "results" / "experiment_4179_arc_incremental_progress.json").exists()

    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text("{bad json", encoding="utf-8")
    malformed = exp.run(write=True)
    assert malformed["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps({"top_pick": "r11l", "ranked_targets": [{"game": "r11l"}, {"game": "lp85"}]}),
        encoding="utf-8",
    )
    no_fixture = exp.run(write=True)
    assert no_fixture["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir = tmp_path / "environment_files" / "lp85" / "305b61c3"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": LP85_GAME_ID, "baseline_actions": [17, 38]}),
        encoding="utf-8",
    )
    missing_py = exp.run(write=True)
    assert missing_py["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir.joinpath("lp85.py").write_text("# marker\n", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_lp85_frontier", lambda arcade, target: _outcome(advanced=True))
    success = exp.run(write=True)
    assert success["honest_verdict"] == "success: incremental_progress_lp85-305b61c3_advanced_to_L2_total14"
    written = json.loads((tmp_path / "results" / "experiment_4179_arc_incremental_progress.json").read_text())
    assert written == success

    monkeypatch.setattr(exp, "_run_lp85_frontier", lambda arcade, target: (_ for _ in ()).throw(RuntimeError("boom")))
    errored = exp.run(write=False)
    assert errored["honest_verdict"].startswith("complete: incremental_progress_no_solve_lp85-305b61c3_L2")
    assert "offline_run_failed_runtimeerror" in errored["honest_verdict"]


def test_results_entrypoint_exists() -> None:
    """REQ-PHASE4-054: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4179_arc_incremental_progress.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4179_arc_incremental_progress" in entrypoint.read_text(encoding="utf-8")
