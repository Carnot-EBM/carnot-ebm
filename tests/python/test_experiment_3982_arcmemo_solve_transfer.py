import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3982_arcmemo_solve_transfer as exp


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
    def __init__(self):
        self.step_count = 0
        self.clicks = []
        self.solved = False
        self._game = self
        self.levels_completed = 0

    def reset(self):
        self.clicks = []
        self.solved = False
        self.levels_completed = 0
        return FakeFrame(0, self.step_count)

    def step(self, action, data=None):
        self.step_count += 1
        if getattr(action, "value", action) == 6 and data:
            self.clicks.append((int(data["x"]), int(data["y"])))
        elif getattr(action, "value", action) == 3 and self.clicks == [(25, 50), (30, 50)]:
            self.solved = True
        self.levels_completed = 1 if self.solved else 0
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", "utf-8")


def _minimal_repo(tmp_path: Path, shared: bool = True) -> None:
    _write_json(
        tmp_path / "results" / "experiment_3981_fourth_game_first_solve.json",
        {
            "ACCURACY_levels_solved": 0,
            "game_solved": "none",
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


def test_spec_declares_arcmemo_solve_transfer_requirement():
    """REQ-PHASE4-020: OpenSpec declares Exp 3982 before implementation."""
    spec = (REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md").read_text("utf-8")
    assert "REQ-PHASE4-020" in spec
    assert "SCENARIO-PHASE4-020" in spec
    assert "experiment_3982_arcmemo_solve_transfer.json" in spec


def test_memory_records_have_shared_positive_control(monkeypatch, tmp_path):
    """REQ-PHASE4-020: banked solved mechanics expose a shared concept family."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    records = exp.build_concept_memory(tmp_path)

    assert [record["name"] for record in records] == [
        "select_then_place",
        "permute_set_by_button",
        "pattern_match_then_navigate",
    ]
    assert exp.positive_control_shared_structure(records) is True
    assert all({"name", "when_it_applies", "effect"} <= set(record) for record in records)


def test_run_writes_solve_transfer_win_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-020: memory-seeded solving reuses a concrete concept and costs fewer steps."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    arcade = FakeArcade()

    artifact = exp.run(
        write=True,
        _arc_client=arcade,
        _actions=FakeActions,
        cold_combo_limit=4,
    )

    assert artifact["solve_transfer_win"] is True
    assert artifact["actions_cold_start"] == 41
    assert artifact["actions_with_memory"] == 3
    assert artifact["attempts_cold_start"] == 4
    assert artifact["attempts_with_memory"] == 1
    assert artifact["concept_reused"] == "pattern_match_then_navigate"
    assert artifact["positive_control_shared_structure"] is True
    assert artifact["real_env_confirmed"] is True
    assert artifact["honest_verdict"] == "success: arcmemo_solve_transfer_41to3_actions"
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    written = tmp_path / "results" / "experiment_3982_arcmemo_solve_transfer.json"
    assert json.loads(written.read_text("utf-8"))["solve_transfer_win"] is True


def test_positive_control_failure_stops_before_solve(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-020: no shared family yields an honest no-transfer artifact."""
    _minimal_repo(tmp_path, shared=False)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["positive_control_shared_structure"] is False
    assert artifact["actions_cold_start"] == 0
    assert artifact["actions_with_memory"] == 0
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_positive_control_failed"
    written = tmp_path / "results" / "experiment_3982_arcmemo_solve_transfer.json"
    assert json.loads(written.read_text("utf-8"))["positive_control_shared_structure"] is False


def test_blocked_offline_env_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-020: absent offline Arcade blocks without claiming a solve."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    class EmptyArcade:
        def get_environments(self):
            return []

    artifact = exp.run(write=True, _arc_client=EmptyArcade(), _actions=FakeActions)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["solve_transfer_win"] is False
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / "experiment_3982_arcmemo_solve_transfer.json"
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_helper_branches_for_target_and_actions(tmp_path):
    """REQ-PHASE4-020: helper branches preserve deterministic action and target semantics."""
    _minimal_repo(tmp_path, shared=False)
    _write_json(
        tmp_path / "results" / "experiment_3981_fourth_game_first_solve.json",
        {
            "ACCURACY_levels_solved": 1,
            "game_solved": "su15-1944f8ab",
            "real_env_confirmed": True,
            "induced_mechanic": "Click objects until target counts match.",
        },
    )

    assert exp._levels_completed(SimpleNamespace(), SimpleNamespace(_game=SimpleNamespace(levels_completed=2))) == 2
    records = exp.build_concept_memory(tmp_path)
    assert records[-1]["name"] == "object_click_count_match"
    assert exp._target_game(tmp_path) == ("su15", "su15-1944f8ab")
    assert exp._action_for_step(FakeActions, {"action": "up"}) == (FakeActions.ACTION1, None)
    assert exp._action_for_step(FakeActions, {"action": "down"}) == (FakeActions.ACTION2, None)
    assert exp._action_for_step(FakeActions, {"action": "right"}) == (FakeActions.ACTION4, None)
    with pytest.raises(ValueError, match="unknown action"):
        exp._action_for_step(FakeActions, {"action": "spin"})
    assert exp._steps_from_solve_log(tmp_path, "su15") == []
    assert exp._retrieve_concept([{"name": "fallback", "family": "click_state_transform"}], "missing") == {
        "name": "fallback",
        "family": "click_state_transform",
    }
    assert exp._retrieve_concept([], "missing") is None

    no_solve = exp._cold_sc25_search(FakeArcade(), FakeActions, "sc25-635fd71a", exp.GameGraph("none"), 0)
    assert no_solve == {"actions": 0, "attempts": 0, "solved": False, "levels_completed": 0, "combo": None}


def test_no_retrievable_concept_blocks(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-020: solved fourth-game target still blocks if no replayable concept exists."""
    _write_json(
        tmp_path / "results" / "experiment_3946_r11l_first_solve.json",
        {"real_env_confirmed": True, "induced_select_place_mechanic": "Click selects then applies."},
    )
    _write_json(
        tmp_path / "results" / "experiment_3981_fourth_game_first_solve.json",
        {
            "ACCURACY_levels_solved": 1,
            "game_solved": "su15-1944f8ab",
            "real_env_confirmed": True,
            "induced_mechanic": "Object clicks solve counts.",
        },
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_no_retrievable_concept"
    assert artifact["positive_control_shared_structure"] is True
    written = tmp_path / "results" / "experiment_3982_arcmemo_solve_transfer.json"
    assert json.loads(written.read_text("utf-8"))["concept_reused"] is None


class NeverSolveEnv(FakeEnv):
    def step(self, action, data=None):
        self.step_count += 1
        self.levels_completed = 0
        return FakeFrame(0, self.step_count)


class NeverSolveArcade(FakeArcade):
    def make(self, _game_id):
        env = NeverSolveEnv()
        self.envs.append(env)
        return env


def test_no_transfer_when_real_env_not_confirmed(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-020: failed real-env confirmation never reports transfer."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=False, _arc_client=NeverSolveArcade(), _actions=FakeActions, cold_combo_limit=1)

    assert artifact["real_env_confirmed"] is False
    assert artifact["solve_transfer_win"] is False
    assert artifact["concept_reused"] is None
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_real_env_solve_not_confirmed"


def test_no_transfer_when_memory_not_cheaper(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-020: equal confirmed costs are complete, not success."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    measured = {"actions": 3, "attempts": 1, "solved": True, "levels_completed": 1}
    monkeypatch.setattr(exp, "_cold_sc25_search", lambda *_args, **_kwargs: {**measured, "combo": 3})
    monkeypatch.setattr(exp, "_execute_plan", lambda *_args, **_kwargs: measured)

    artifact = exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["real_env_confirmed"] is True
    assert artifact["solve_transfer_win"] is False
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_memory_not_cheaper"
