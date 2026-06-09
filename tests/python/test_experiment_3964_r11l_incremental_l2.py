"""Tests for Exp 3964 r11l incremental real-env solve.

Spec coverage: REQ-PHASE4-013, SCENARIO-PHASE4-013.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "experiments" / "experiment_3964_r11l_incremental_l2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_3964_r11l_incremental_l2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeAction:
    ACTION6 = 6


class FakeState:
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"
    PLAYING = "PLAYING"


class FakeSprite:
    def __init__(self, y: int, x: int, height: int = 2, width: int = 2) -> None:
        self.y = y
        self.x = x
        self.height = height
        self.width = width


class FakeGame:
    def __init__(self, levels: list[list[tuple[tuple[int, int], tuple[int, int]]]]) -> None:
        self.levels_completed = 0
        self.yfbjozweime = False
        self._levels = levels
        self.kacotwgjcyq: dict[str, dict[str, object]] = {}
        self.load_level(0)

    def load_level(self, index: int) -> None:
        groups: dict[str, dict[str, object]] = {}
        if index < len(self._levels):
            for i, (piece, target) in enumerate(self._levels[index]):
                groups[f"group-{index}-{i}"] = {
                    "lecfirgqbwunn": [FakeSprite(piece[0], piece[1])],
                    "gosubdcyegamj": FakeSprite(target[0], target[1]),
                }
        self.kacotwgjcyq = groups


class FakeEnv:
    def __init__(
        self,
        levels: list[list[tuple[tuple[int, int], tuple[int, int]]]],
        *,
        fail_zero_based_levels: set[int] | None = None,
    ) -> None:
        self._game = FakeGame(levels)
        self._levels = levels
        self._fail_zero_based_levels = fail_zero_based_levels or set()
        self.actions: list[tuple[int, int]] = []
        self.attempted_levels: list[int] = []
        self._clicks_this_level = 0

    def reset(self):
        self._game.levels_completed = 0
        self._game.load_level(0)
        self._clicks_this_level = 0
        return self._frame()

    def step(self, action: int, data: dict[str, int]):
        assert action == FakeAction.ACTION6
        self.actions.append((data["y"], data["x"]))
        if data["x"] == -1 and data["y"] == -1:
            return self._frame()

        current_level = self._game.levels_completed
        if current_level not in self.attempted_levels:
            self.attempted_levels.append(current_level)

        self._clicks_this_level += 1
        needed_clicks = len(self._levels[current_level]) * 2
        if self._clicks_this_level >= needed_clicks:
            if current_level not in self._fail_zero_based_levels:
                self._game.levels_completed += 1
                self._game.load_level(self._game.levels_completed)
            self._clicks_this_level = 0
        return self._frame()

    def _frame(self):
        state = FakeState.WIN if self._game.levels_completed >= len(self._levels) else FakeState.PLAYING
        return SimpleNamespace(levels_completed=self._game.levels_completed, state=state, frame=[[0]])


def test_solver_reperceives_each_level_and_reports_incremental_artifact() -> None:
    """REQ-PHASE4-013: the solver re-perceives after each confirmed level-up."""
    mod = load_module()
    env = FakeEnv(
        [
            [((10, 1), (20, 30)), ((12, 4), (20, 30))],
            [((30, 2), (7, 40)), ((32, 4), (7, 40)), ((34, 6), (7, 40))],
            [((50, 9), (15, 15))],
        ]
    )

    solved = mod.solve_incremental_levels(env, FakeAction, FakeState, budget=30, stop_after_level=3)
    artifact = mod.build_result_artifact(
        solved,
        game="r11l-495a7899",
        budget=30,
        duration_s=0.25,
        precondition_blocked=False,
    )

    assert solved.levels_completed == 3
    assert solved.per_level_actions == [4, 6, 2]
    assert solved.baseline_actions_ref == [4, 6, 2]
    assert solved.first_fail_level is None
    assert [summary["n_pairs"] for summary in solved.level_summaries] == [2, 3, 1]
    assert env.attempted_levels == [0, 1, 2]
    assert artifact["ACCURACY_levels_solved"] == 3
    assert artifact["new_levels_solved_this_task"] == 2
    assert artifact["real_env_confirmed"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_solver_stops_at_first_failed_incremental_level() -> None:
    """SCENARIO-PHASE4-013: the scoped run stops at L3 when L3 does not solve."""
    mod = load_module()
    env = FakeEnv(
        [
            [((10, 1), (20, 30)), ((12, 4), (20, 30))],
            [((30, 2), (7, 40))],
            [((50, 9), (15, 15))],
            [((5, 5), (9, 9))],
        ],
        fail_zero_based_levels={2},
    )

    solved = mod.solve_incremental_levels(env, FakeAction, FakeState, budget=30, stop_after_level=3)
    artifact = mod.build_result_artifact(
        solved,
        game="r11l-495a7899",
        budget=30,
        duration_s=0.5,
        precondition_blocked=False,
    )

    assert solved.levels_completed == 2
    assert solved.new_levels_solved_this_task == 1
    assert solved.first_fail_level == 3
    assert env.attempted_levels == [0, 1, 2]
    assert artifact["ACCURACY_levels_solved"] == 2
    assert artifact["first_fail_level"] == 3
    assert "levels2_of6" in artifact["honest_verdict"]


def test_blocked_artifact_has_required_schema_and_terminal_prefix(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-013: unavailable offline ARC writes a blocked terminal artifact."""
    mod = load_module()

    artifact = mod.build_blocked_artifact(
        game="r11l-495a7899",
        budget=30,
        duration_s=0.1,
        honest_verdict="blocked_arc_offline_env_unavailable",
    )
    output = mod.write_result_artifact(artifact, tmp_path / "experiment_3964.json")

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["ACCURACY_levels_solved"] == 1
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact["per_level_actions"] == []
    assert artifact["baseline_actions_ref"] == []
    assert artifact["first_fail_level"] is None
    assert artifact["real_env_confirmed"] is False
    assert artifact["inference_substrate"] == "offline_arc_agi3_perception_planner_real_env_confirmed"
