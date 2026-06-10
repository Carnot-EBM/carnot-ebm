import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4005_arcmemo_solve_transfer_v3 as exp


SU15_STEPS = [
    {"action": "click", "x": 8, "y": 54},
    {"action": "click", "x": 12, "y": 50},
    {"action": "click", "x": 16, "y": 46},
    {"action": "click", "x": 20, "y": 42},
    {"action": "click", "x": 24, "y": 38},
    {"action": "click", "x": 28, "y": 34},
    {"action": "click", "x": 32, "y": 30},
    {"action": "click", "x": 36, "y": 26},
    {"action": "click", "x": 40, "y": 22},
    {"action": "click", "x": 44, "y": 18},
]


class FakeAction:
    def __init__(self, value: int):
        self.value = value


class FakeActions:
    ACTION1 = FakeAction(1)
    ACTION2 = FakeAction(2)
    ACTION3 = FakeAction(3)
    ACTION4 = FakeAction(4)
    ACTION6 = FakeAction(6)


class FakeFrame:
    def __init__(self, levels_completed: int = 0, tick: int = 0):
        self.levels_completed = levels_completed
        self.frame = np.array([[levels_completed, tick % 251]], dtype=np.uint8)
        self.available_actions = [1, 2, 3, 4, 6]
        self.state = None


class FakeEnv:
    required = [(row["x"], row["y"]) for row in SU15_STEPS]

    def __init__(self):
        self.step_count = 0
        self.clicks = []
        self._game = self
        self.levels_completed = 0

    def reset(self):
        self.clicks = []
        self.levels_completed = 0
        return FakeFrame(0, self.step_count)

    def step(self, action, data=None):
        self.step_count += 1
        if getattr(action, "value", action) == 6 and data:
            self.clicks.append((int(data["x"]), int(data["y"])))
        if self.clicks == self.required:
            self.levels_completed = 1
        return FakeFrame(self.levels_completed, self.step_count)


class FakeArcade:
    def __init__(self):
        self.envs = []

    def get_environments(self):
        return [object()]

    def make(self, _game_id):
        env = FakeEnv()
        self.envs.append(env)
        return env


class EmptyArcade:
    def get_environments(self):
        return []


class QuickSolveEnv(FakeEnv):
    required = [(8, 54)]


class QuickSolveArcade(FakeArcade):
    def make(self, _game_id):
        env = QuickSolveEnv()
        self.envs.append(env)
        return env


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", "utf-8")


def _minimal_repo(
    tmp_path: Path,
    *,
    shared: bool = True,
    solved_4004: bool = True,
    include_solve_log: bool = True,
) -> None:
    _write_json(
        tmp_path / "results" / "experiment_4004_fourth_game_explore_first.json",
        {
            "ACCURACY_levels_solved": 1 if solved_4004 else 0,
            "game_solved": "su15-1944f8ab" if solved_4004 else "none",
            "real_env_confirmed": True,
            "attempt_details": [
                {
                    "game_id": "su15-1944f8ab",
                    "levels_completed": 1 if solved_4004 else 0,
                    "first_solve_at_action": 14 if solved_4004 else -1,
                    "solve_log": SU15_STEPS if include_solve_log else [],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_4003_scale_level_frontier.json",
        {
            "new_levels_this_task": 0,
            "per_game_max_level": {"r11l": 3, "lp85": 1, "sc25": 1},
            "per_game_new_levels": {"r11l": 0, "lp85": 0, "sc25": 0},
            "real_env_confirmed": True,
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_3946_r11l_first_solve.json",
        {
            "real_env_confirmed": True,
            "game": "r11l-495a7899",
            "induced_select_place_mechanic": "Click selects a piece and a later click places it.",
        },
    )
    if shared:
        _write_json(
            tmp_path / "results" / "experiment_3954_second_game_solve.json",
            {
                "real_env_confirmed": True,
                "game_solved": "lp85-305b61c3",
                "induced_mechanic": "Buttons permute a set of pieces.",
            },
        )
        _write_json(
            tmp_path / "results" / "experiment_3966_third_game_first_solve.json",
            {
                "real_env_confirmed": True,
                "game_solved": "sc25-635fd71a",
                "induced_mechanic": "Pattern matching by clicked cells and then navigation to an exit.",
                "solve_log": [
                    {"action": "click", "x": 25, "y": 50},
                    {"action": "click", "x": 30, "y": 50},
                    {"action": "left"},
                ],
            },
        )


def test_spec_declares_arcmemo_solve_transfer_v3_requirement():
    """REQ-PHASE4-025: OpenSpec declares Exp 4005 before implementation."""
    spec = (REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md").read_text("utf-8")
    assert "REQ-PHASE4-025" in spec
    assert "SCENARIO-PHASE4-025" in spec
    assert "experiment_4005_arcmemo_solve_transfer_v3.json" in spec


def test_target_prefers_exp4004_new_fourth_game(tmp_path):
    """REQ-PHASE4-025: target selection prefers the solved new Exp 4004 content."""
    _minimal_repo(tmp_path)

    target = exp.select_target_game(tmp_path)
    records = exp.build_concept_memory(tmp_path)

    assert target == ("su15", "su15-1944f8ab", "experiment_4004_fourth_game_explore_first.json")
    assert exp.positive_control_shared_structure(records, "su15") is True
    assert all({"name", "when_it_applies", "effect"} <= set(record) for record in records)


def test_target_falls_back_to_4003_r11l_then_reheldout_sc25(tmp_path):
    """REQ-PHASE4-025: target selection falls back only when newer content did not advance."""
    _minimal_repo(tmp_path, solved_4004=False)
    _write_json(
        tmp_path / "results" / "experiment_4003_scale_level_frontier.json",
        {
            "new_levels_this_task": 1,
            "per_game_max_level": {"r11l": 4, "lp85": 1, "sc25": 1},
            "per_game_new_levels": {"r11l": 1, "lp85": 0, "sc25": 0},
            "real_env_confirmed": True,
        },
    )
    assert exp.select_target_game(tmp_path) == (
        "r11l",
        "r11l-495a7899",
        "experiment_4003_scale_level_frontier.json",
    )

    (tmp_path / "results" / "experiment_4003_scale_level_frontier.json").unlink()
    assert exp.select_target_game(tmp_path) == ("sc25", "sc25-635fd71a", "reheld_out_sc25")


def test_run_writes_solve_transfer_v3_win_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: memory-seeded solving reuses a concept and costs fewer real steps."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is True
    assert artifact["actions_cold_start"] == 14
    assert artifact["actions_with_memory"] == 10
    assert artifact["attempts_cold_start"] == 2
    assert artifact["attempts_with_memory"] == 1
    assert artifact["target_game"] == "su15-1944f8ab"
    assert artifact["concept_reused"] == "pattern_match_then_navigate"
    assert artifact["positive_control_shared_structure"] is True
    assert artifact["real_env_confirmed"] is True
    assert artifact["honest_verdict"] == "success: arcmemo_solve_transfer_v3_14to10_actions"
    written = tmp_path / "results" / "experiment_4005_arcmemo_solve_transfer_v3.json"
    assert json.loads(written.read_text("utf-8"))["solve_transfer_win"] is True


def test_positive_control_failure_stops_before_solve(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: fewer than two same-family banked concepts cannot claim transfer."""
    _minimal_repo(tmp_path, shared=False)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["target_game"] == "su15-1944f8ab"
    assert artifact["positive_control_shared_structure"] is False
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_positive_control_failed"


def test_blocked_offline_env_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: absent offline Arcade blocks without fabricating env confirmation."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=EmptyArcade(), _actions=FakeActions)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["target_game"] == "unknown"
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / "experiment_4005_arcmemo_solve_transfer_v3.json"
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_new_target_without_replayable_concept_stops(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: a new target without replay evidence is a bounded no-transfer result."""
    _minimal_repo(tmp_path, include_solve_log=False)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["target_game"] == "su15-1944f8ab"
    assert artifact["concept_reused"] is None
    assert artifact["positive_control_shared_structure"] is True
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_content_no_replayable_retrieved_concept"


def test_no_transfer_when_real_env_confirmation_fails(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: unconfirmed solves remain complete rather than success."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    measured = {"actions": 3, "attempts": 1, "solved": False, "levels_completed": 0}
    monkeypatch.setattr(exp, "_cold_su15_no_memory_solve", lambda *_args, **_kwargs: measured)
    monkeypatch.setattr(exp, "_execute_plan", lambda *_args, **_kwargs: measured)

    artifact = exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact["concept_reused"] is None
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_content_real_env_solve_not_confirmed"


def test_no_transfer_when_memory_is_not_cheaper(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: confirmed equal-cost memory reuse is a bounded no-transfer finding."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    measured = {"actions": 10, "attempts": 1, "solved": True, "levels_completed": 1}
    monkeypatch.setattr(exp, "_cold_su15_no_memory_solve", lambda *_args, **_kwargs: measured)
    monkeypatch.setattr(exp, "_execute_plan", lambda *_args, **_kwargs: measured)

    artifact = exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["real_env_confirmed"] is True
    assert artifact["concept_reused"] == "pattern_match_then_navigate"
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_content_memory_not_cheaper"


def test_helpers_cover_replay_and_schema_edges(monkeypatch, tmp_path):
    """REQ-PHASE4-025: helper branches preserve deterministic target and schema semantics."""
    _minimal_repo(tmp_path)

    records = exp.build_concept_memory(tmp_path)
    assert exp._target_family("su15") == "click_state_transform"
    assert exp._target_family("missing") == ""
    assert exp.positive_control_shared_structure(records, "missing") is False
    assert exp._retrieve_concept(records, "sc25")["name"] == "pattern_match_then_navigate"
    assert exp._retrieve_concept(records, "su15")["name"] == "pattern_match_then_navigate"
    assert exp._retrieve_concept([], "su15") is None
    assert exp._memory_steps_for_target(tmp_path, "su15", "su15-1944f8ab") == SU15_STEPS
    assert exp._steps_from_exp4004_solve_log(tmp_path, "missing-game") == []

    _minimal_repo(tmp_path, solved_4004=False)
    assert exp._steps_from_exp4004_solve_log(tmp_path, "su15-1944f8ab") == []

    _minimal_repo(tmp_path)
    assert exp._memory_steps_for_target(tmp_path, "sc25", "sc25-635fd71a") == [
        {"action": "click", "x": 25, "y": 50},
        {"action": "click", "x": 30, "y": 50},
        {"action": "left"},
    ]
    assert exp._memory_steps_for_target(tmp_path, "r11l", "r11l-495a7899") == []
    errors = exp.artifact_schema_errors(
        {
            "solve_transfer_win": "no",
            "actions_cold_start": "1",
            "actions_with_memory": "1",
            "attempts_cold_start": "1",
            "attempts_with_memory": "1",
            "target_game": 7,
            "positive_control_shared_structure": "yes",
            "real_env_confirmed": "yes",
            "random_seed": "42",
            "honest_verdict": "bad",
            "duration_s": "fast",
            "inference_substrate": 9,
        }
    )
    assert "actions_cold_start must be a bare int" in errors
    assert "solve_transfer_win must be a bare bool" in errors
    assert "target_game must be a bare string" in errors
    assert "duration_s must be a bare number" in errors

    monkeypatch.setattr(exp, "REQUIRED_ARTIFACT_FIELDS", ("missing",))
    with pytest.raises(ValueError, match="missing required field"):
        exp._empty_artifact(42, 0.0, "blocked_arc_offline_env_unavailable")


def test_cold_su15_records_exploration_solve_edge():
    """SCENARIO-PHASE4-025: cold-start accounting stops if exploration itself solves."""
    measured = exp._cold_su15_no_memory_solve(
        QuickSolveArcade(),
        FakeActions,
        "su15-1944f8ab",
        exp.GameGraph("quick"),
        [SU15_STEPS[0]],
        exploration_steps=1,
    )

    assert measured == {"actions": 1, "attempts": 1, "solved": True, "levels_completed": 1}


def test_run_sc25_fallback_branch(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: re-held-out fallback still measures a two-arm comparison."""
    _minimal_repo(tmp_path, solved_4004=False)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(
        exp,
        "_cold_sc25_search",
        lambda *_args, **_kwargs: {"actions": 8, "attempts": 2, "solved": True, "levels_completed": 1},
    )
    monkeypatch.setattr(
        exp,
        "_execute_plan",
        lambda *_args, **_kwargs: {"actions": 3, "attempts": 1, "solved": True, "levels_completed": 1},
    )

    artifact = exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["target_game"] == "sc25-635fd71a"
    assert artifact["honest_verdict"] == "success: arcmemo_solve_transfer_v3_8to3_actions"


def test_run_unsupported_frontier_branch_stops(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-025: unsupported advanced targets stop before fabricated replay."""
    _minimal_repo(tmp_path, solved_4004=False)
    _write_json(
        tmp_path / "results" / "experiment_4003_scale_level_frontier.json",
        {
            "new_levels_this_task": 1,
            "per_game_new_levels": {"r11l": 1},
            "real_env_confirmed": True,
        },
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_memory_steps_for_target", lambda *_args, **_kwargs: [SU15_STEPS[0]])

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["target_game"] == "r11l-495a7899"
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_content_no_replayable_retrieved_concept"


def test_run_schema_validation_failure_is_not_silent(monkeypatch, tmp_path):
    """REQ-PHASE4-025: final artifact schema validation is enforced."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["schema broke"])

    with pytest.raises(ValueError, match="schema broke"):
        exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)
