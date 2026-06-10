import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3994_arcmemo_solve_transfer_v2 as exp


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


class EmptyArcade:
    def get_environments(self):
        return []


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", "utf-8")


def _minimal_repo(tmp_path: Path, shared: bool = True, solved_3993: bool = False) -> None:
    _write_json(
        tmp_path / "results" / "experiment_3993_fourth_game_verifier_pruned.json",
        {
            "ACCURACY_levels_solved": 1 if solved_3993 else 0,
            "game_solved": "su15-1944f8ab" if solved_3993 else "none",
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


def test_spec_declares_arcmemo_solve_transfer_v2_requirement():
    """REQ-PHASE4-022: OpenSpec declares Exp 3994 before implementation."""
    spec = (REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md").read_text("utf-8")
    assert "REQ-PHASE4-022" in spec
    assert "SCENARIO-PHASE4-022" in spec
    assert "experiment_3994_arcmemo_solve_transfer_v2.json" in spec


def test_target_falls_back_to_sc25_when_exp3993_has_no_solve(tmp_path):
    """REQ-PHASE4-022: no-solve Exp 3993 re-holds out sc25 as the transfer target."""
    assert exp.select_target_game(tmp_path) == ("sc25", "sc25-635fd71a", "reheld_out_sc25")
    _minimal_repo(tmp_path)

    target = exp.select_target_game(tmp_path)
    records = exp.build_concept_memory(tmp_path)

    assert target == ("sc25", "sc25-635fd71a", "reheld_out_sc25")
    assert exp.positive_control_shared_structure(records) is True
    assert all({"name", "when_it_applies", "effect"} <= set(record) for record in records)


def test_run_writes_solve_transfer_v2_win_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-022: memory-seeded solving reuses a concept and costs fewer real steps."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions, cold_combo_limit=4)

    assert artifact["solve_transfer_win"] is True
    assert artifact["actions_cold_start"] == 41
    assert artifact["actions_with_memory"] == 3
    assert artifact["attempts_cold_start"] == 4
    assert artifact["attempts_with_memory"] == 1
    assert artifact["target_game"] == "sc25-635fd71a"
    assert artifact["concept_reused"] == "pattern_match_then_navigate"
    assert artifact["positive_control_shared_structure"] is True
    assert artifact["real_env_confirmed"] is True
    assert artifact["honest_verdict"] == "success: arcmemo_solve_transfer_v2_41to3_actions"
    written = tmp_path / "results" / "experiment_3994_arcmemo_solve_transfer_v2.json"
    assert json.loads(written.read_text("utf-8"))["solve_transfer_win"] is True


def test_solved_exp3993_target_without_replayable_concept_stops(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-022: a new target without replayable concept evidence is bounded."""
    _minimal_repo(tmp_path, solved_3993=True)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["target_game"] == "su15-1944f8ab"
    assert artifact["positive_control_shared_structure"] is True
    assert artifact["concept_reused"] is None
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_game_no_retrievable_concept"


def test_positive_control_failure_stops_before_solve(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-022: fewer than two shared-family concepts cannot claim transfer."""
    _minimal_repo(tmp_path, shared=False)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["target_game"] == "sc25-635fd71a"
    assert artifact["positive_control_shared_structure"] is False
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_positive_control_failed"


def test_blocked_offline_env_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-022: absent offline Arcade blocks without fabricating env confirmation."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True, _arc_client=EmptyArcade(), _actions=FakeActions)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["target_game"] == "unknown"
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / "experiment_3994_arcmemo_solve_transfer_v2.json"
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_no_transfer_when_real_env_confirmation_fails(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-022: unconfirmed solves remain complete rather than success."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    measured = {"actions": 3, "attempts": 1, "solved": False, "levels_completed": 0}
    monkeypatch.setattr(exp, "_cold_sc25_search", lambda *_args, **_kwargs: {**measured, "combo": None})
    monkeypatch.setattr(exp, "_execute_plan", lambda *_args, **_kwargs: measured)

    artifact = exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact["concept_reused"] is None
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_game_real_env_solve_not_confirmed"


def test_no_transfer_when_memory_is_not_cheaper(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-022: confirmed equal-cost memory reuse is a bounded no-transfer finding."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    measured = {"actions": 3, "attempts": 1, "solved": True, "levels_completed": 1}
    monkeypatch.setattr(exp, "_cold_sc25_search", lambda *_args, **_kwargs: {**measured, "combo": 3})
    monkeypatch.setattr(exp, "_execute_plan", lambda *_args, **_kwargs: measured)

    artifact = exp.run(write=False, _arc_client=FakeArcade(), _actions=FakeActions)

    assert artifact["solve_transfer_win"] is False
    assert artifact["real_env_confirmed"] is True
    assert artifact["concept_reused"] == "pattern_match_then_navigate"
    assert artifact["honest_verdict"] == "complete: arcmemo_solve_no_transfer_to_new_game_memory_not_cheaper"
