"""Tests for Exp 4224 ARC-AGI-3 SC25 L3 incremental progress.

Spec refs: REQ-PHASE4-061, SCENARIO-PHASE4-061.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import carnot.experiment_4224_arc_incremental_progress as exp
from carnot.experiment_4224_arc_incremental_progress import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_LEVELS_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    SC25_GAME_ID,
    FrontierOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    explore_sc25_l3_precast_suffix,
    select_deeper_level_target,
    validate_hardened_gap4_l3_suffix,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class FakeAction:
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION6 = 6


@dataclass
class FakeSprite:
    x: int
    y: int
    scale: int = 2


class FakeSc25Game:
    def __init__(self) -> None:
        self.zzpoabuniyn = {
            "l1": [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ],
            "tevyeq": [
                [True, True, False],
                [False, True, False],
                [False, False, False],
            ],
            "fibcey": [
                [False, True, False],
                [False, True, False],
                [False, True, False],
            ],
        }
        self.set_level(0)

    def set_level(self, level: int) -> None:
        self._current_level_index = int(level)
        self.jlpticwjyvy = ["l1", "tevyeq", "fibcey"][min(level, 2) : min(level, 2) + 1]
        self.ijhfdcamokt = self.jlpticwjyvy[0]
        self.xhhaqjfncnp = [[False for _ in range(3)] for _ in range(3)]
        self.rrinmfkkstu = 0
        self.txyqmqkitgl = None
        self.pattern_ready = False
        self.moves_after_pattern = 0
        self.eycwbtepcvs = False
        self.xelyxfeemol = 0
        self.ufpevlpokkj = 0
        self.wihhwrkolym = 0
        self.jdmucabyqar = 0
        self.barrier_removed = False
        self.l3_progress: list[int] = []
        self.plnqvukupu = FakeSprite(35, 22, 2)


class FakeSc25Env:
    def __init__(self) -> None:
        self._game = FakeSc25Game()

    def reset(self) -> SimpleNamespace:
        self._game.set_level(0)
        return SimpleNamespace(levels_completed=0)

    def step(self, action: object, data: dict[str, int] | None = None) -> SimpleNamespace:
        action_id = exp.previous._action_id(action)
        game = self._game
        if action_id in (1, 2, 3, 4):
            game.jdmucabyqar = {1: 0, 2: 1, 3: 2, 4: 3}[action_id]
            game.rrinmfkkstu += 1
            if game._current_level_index < 2 and game.pattern_ready and action_id == 1:
                game.moves_after_pattern += 1
                needed = 1 if game._current_level_index == 0 else 2
                if game.moves_after_pattern >= needed:
                    game.set_level(game._current_level_index + 1)
            elif game._current_level_index == 2 and game.barrier_removed:
                game.l3_progress.append(action_id)
                if game.l3_progress == [2, 2, 3, 3, 3, 2, 3]:
                    game.set_level(3)
        elif action_id == 6 and data:
            coord_to_cell = {
                (25, 50): (0, 0),
                (30, 50): (0, 1),
                (35, 50): (0, 2),
                (25, 55): (1, 0),
                (30, 55): (1, 1),
                (35, 55): (1, 2),
                (25, 60): (2, 0),
                (30, 60): (2, 1),
                (35, 60): (2, 2),
            }
            row, col = coord_to_cell[(int(data["x"]), int(data["y"]))]
            game.xhhaqjfncnp[row][col] = not game.xhhaqjfncnp[row][col]
            game.rrinmfkkstu += 1
            if game.xhhaqjfncnp == game.zzpoabuniyn[game.jlpticwjyvy[0]]:
                game.xhhaqjfncnp = [[False for _ in range(3)] for _ in range(3)]
                if game._current_level_index < 2:
                    game.pattern_ready = True
                elif game.jdmucabyqar == 3:
                    game.barrier_removed = True
        return SimpleNamespace(levels_completed=game._current_level_index)


class FakeArcade:
    def __init__(self) -> None:
        self.env = FakeSc25Env()

    def make(self, game_id: str) -> FakeSc25Env:
        assert game_id == SC25_GAME_ID
        return self.env


def _hardening_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4187_gap4_graded_execution_gate_hardening",
        "vote_aware_guard_blocked_mispromotion": True,
        "gross_recovery_ledger": {"recovered": 4, "lost": 0},
    }


def _prior_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4213_arc_incremental_progress",
        "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L2_total16",
        "target_game": SC25_GAME_ID,
        "target_level": 2,
        "total_levels_solved": 16,
        "levels_completed": 2,
        "new_levels_solved_this_task": 1,
        "real_env_confirmed": True,
        "action_plan": [
            {"action": 6, "kind": "pattern_click", "row": 0, "col": 0, "x": 25, "y": 50},
            {"action": 6, "kind": "pattern_click", "row": 0, "col": 1, "x": 30, "y": 50},
            {"action": 6, "kind": "pattern_click", "row": 1, "col": 1, "x": 30, "y": 55},
            {"action": 1, "kind": "move"},
            {"action": 1, "kind": "move"},
        ],
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=3,
        prior_level=2,
        baseline_actions=32,
        selection_mode="deeper_sc25_frontier_after_exp4213_L2",
        selection_reason="selected sc25 L3 after Exp 4213 banked sc25 L2 with hardened GAP-4 evidence",
    )


def _outcome(*, advanced: bool) -> FrontierOutcome:
    validation = validate_hardened_gap4_l3_suffix(
        start_level=2,
        final_level=3 if advanced else 2,
        heldout_transition_count=5,
        predicted_level=3,
        gap4_artifact=_hardening_artifact(),
    )
    return FrontierOutcome(
        target_game=SC25_GAME_ID,
        target_level=3,
        prior_level=2,
        final_level_completed=3 if advanced else 2,
        replay_actions_used=22,
        executed_real_env_actions=11 if advanced else 0,
        exploration_actions_used=44,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        verification_decisions=[validation],
        action_plan=[
            {"action": 4, "kind": "precast_face_right"},
            {"action": 6, "kind": "pattern_click", "x": 30, "y": 50},
            {"action": 2, "kind": "move"},
        ]
        if advanced
        else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "replay", "source": "sc25_L1_reestablishment"},
            {"phase": "replay", "source": "sc25_L2_banked_suffix"},
            {"phase": "explore", "precast_action": 4},
            {"phase": "induce", "mechanic": "fibcey fire unlock"},
            validation,
            {"phase": "act", "levels_completed": 3 if advanced else 2},
        ],
        induced_mechanic="sc25 L3 pre-cast facing plus fibcey fire unlock followed by exit-touch movement",
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_061_spec_declares_exp4224_contract() -> None:
    """REQ-PHASE4-061: OpenSpec declares the Exp 4224 terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-061" in spec
    assert "SCENARIO-PHASE4-061" in spec
    assert "experiment_4224_arc_incremental_progress.json" in spec
    assert "sc25-635fd71a" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_061_selects_sc25_l3_after_exp4213_l2() -> None:
    """REQ-PHASE4-061: target selection prefers the SC25 L3 frontier."""

    survey = {"per_game_surveys": [{"game": "sc25"}]}
    baselines = {"sc25": (SC25_GAME_ID, [36, 6, 32])}

    assert select_deeper_level_target(survey, baselines, _prior_artifact(), _hardening_artifact()) == _target()

    with pytest.raises(ValueError, match="Exp 4213 sc25 L2 success evidence unavailable"):
        select_deeper_level_target(survey, baselines, {**_prior_artifact(), "real_env_confirmed": False}, _hardening_artifact())
    with pytest.raises(ValueError, match="hardened GAP-4 verifier evidence unavailable"):
        select_deeper_level_target(survey, baselines, _prior_artifact(), {"gross_recovery_ledger": {"lost": 1}})
    with pytest.raises(ValueError, match="sc25 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {"sc25": (SC25_GAME_ID, [36, 6])}, _prior_artifact(), _hardening_artifact())


def test_scenario_phase4_061_l3_precast_exploration_finds_suffix() -> None:
    """SCENARIO-PHASE4-061: copied exploration induces the pre-cast facing rule."""

    arcade = FakeArcade()
    env = arcade.make(SC25_GAME_ID)
    env.reset()
    l1_plan, _ = exp.previous.plan_sc25_suffix_bounded(env, FakeAction, target_level=1)
    exp.previous.execute_plan_until_level(env, FakeAction, l1_plan, prior_level=0, target_level=1, phase="replay")
    exp.previous.execute_plan_until_level(
        env,
        FakeAction,
        _prior_artifact()["action_plan"],  # type: ignore[arg-type]
        prior_level=1,
        target_level=2,
        phase="replay",
    )

    plan, trace = explore_sc25_l3_precast_suffix(env, FakeAction, target_level=3, max_depth=16, max_expansions=256)

    assert trace["found"] is True
    assert trace["precast_action"] == 4
    assert plan[0]["kind"] == "precast_face_right"
    assert [step["action"] for step in plan[-7:]] == [2, 2, 3, 3, 3, 2, 3]


def test_scenario_phase4_061_artifact_schema_accepts_success_and_complete() -> None:
    """SCENARIO-PHASE4-061: only hardened-verified real-env evidence increments levels."""

    success = build_artifact(_outcome(advanced=True), _target(), random_seed=4224, duration_s=0.2)

    assert success["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L3_total17"
    assert success["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED + 1
    assert success["levels_completed"] == 3
    assert success["real_env_confirmed"] is True
    assert success["acceptance_gate_passed"] is True
    assert success["inference_substrate"] == INFERENCE_SUBSTRATE
    assert success["solve_trace"]["actions"][0]["kind"] == "precast_face_right"
    assert artifact_schema_errors(success) == []

    no_solve = build_artifact(_outcome(advanced=False), _target(), random_seed=4224, duration_s=0.2)
    assert no_solve["honest_verdict"].startswith("complete: incremental_progress_no_solve_sc25-635fd71a_L3")
    assert no_solve["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert no_solve["new_levels_solved_this_task"] == 0
    assert no_solve["real_env_confirmed"] is False
    assert no_solve["acceptance_gate_passed"] is True

    blocked = blocked_artifact(target_game=SC25_GAME_ID, target_level=3, random_seed=4224, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["acceptance_gate_passed"] is False
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_061_validation_and_schema_reject_fabrication() -> None:
    """SCENARIO-PHASE4-061: validation must precede acting and schema rejects inflation."""

    retained = validate_hardened_gap4_l3_suffix(
        start_level=2,
        final_level=3,
        heldout_transition_count=4,
        predicted_level=3,
        gap4_artifact=_hardening_artifact(),
    )
    rejected = validate_hardened_gap4_l3_suffix(
        start_level=2,
        final_level=2,
        heldout_transition_count=4,
        predicted_level=3,
        gap4_artifact=_hardening_artifact(),
    )

    assert retained["retained"] is True
    assert retained["verifier"] == exp.HARDENED_VERIFIER
    assert rejected["retained"] is False
    assert rejected["energy"] == 1.0
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4224}))
    fabricated = build_artifact(_outcome(advanced=True), _target(), random_seed=4224, duration_s=0.0)
    fabricated["real_env_confirmed"] = False
    assert any("real_env_confirmed must be true for success" in err for err in artifact_schema_errors(fabricated))


def test_scenario_phase4_061_runner_writes_real_env_confirmed_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-061: runner writes replay-explore-verify-act evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "environment_files" / "sc25" / "635fd71a").mkdir(parents=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps({"per_game_surveys": [{"game": "sc25"}]}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4213_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json").write_text(
        json.dumps(_hardening_artifact()),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {"sc25": (SC25_GAME_ID, [36, 6, 32])})
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: FakeArcade())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L3_total17"
    assert artifact["total_levels_solved"] == 17
    assert artifact["levels_completed"] == 3
    assert artifact["real_env_confirmed"] is True
    assert [row["phase"] for row in artifact["phase_trace"] if row["phase"] in {"observe", "induce", "hardened-gap4-verify"}]
    written = json.loads((tmp_path / "results" / "experiment_4224_arc_incremental_progress.json").read_text(encoding="utf-8"))
    assert written["solve_trace"]["actions"][0]["kind"] == "precast_face_right"


def test_req_phase4_061_entrypoint_exists() -> None:
    """REQ-PHASE4-061: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4224_arc_incremental_progress.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4224_arc_incremental_progress" in entrypoint.read_text(encoding="utf-8")
