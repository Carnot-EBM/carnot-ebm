"""Tests for Exp 3965 lp85 incremental real-env solve.

Spec coverage: REQ-PHASE4-014, SCENARIO-PHASE4-014.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "experiments" / "experiment_3965_lp85_incremental_l2.py"
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_3965_lp85_incremental_l2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeAction:
    ACTION6 = 6


@dataclass(frozen=True)
class FakeLevel:
    buttons: tuple[tuple[int, int], ...]
    sequence: tuple[tuple[int, int], ...]
    fail_on_complete: bool = False


class FakeGame:
    def __init__(self, levels: list[FakeLevel]) -> None:
        self.levels = levels
        self.levels_completed = 0
        self.progress = 0
        self.last_button_index = 0
        self.frame = self._grid()

    def _level(self) -> FakeLevel:
        return self.levels[min(self.levels_completed, len(self.levels) - 1)]

    def _grid(self) -> np.ndarray:
        grid = np.zeros((16, 16), dtype=np.int16)
        level = self._level()
        for index, (y, x) in enumerate(level.buttons, start=1):
            grid[y, x] = index
        grid[0, 0] = self.progress + 3
        grid[0, 1] = self.last_button_index
        grid[0, 2] = self.levels_completed
        return grid

    def click(self, y: int, x: int) -> None:
        level = self._level()
        if (y, x) not in level.buttons:
            self.frame = self._grid()
            return

        self.last_button_index = level.buttons.index((y, x)) + 1
        expected = level.sequence[self.progress] if self.progress < len(level.sequence) else None
        self.progress = self.progress + 1 if (y, x) == expected else 0
        if self.progress >= len(level.sequence):
            if not level.fail_on_complete:
                self.levels_completed += 1
            self.progress = 0
            self.last_button_index = 0
        self.frame = self._grid()


class FakeLp85Env:
    def __init__(self, levels: list[FakeLevel]) -> None:
        self._levels = levels
        self._game = FakeGame(levels)

    def reset(self):
        self._game = FakeGame(self._levels)
        return self._frame()

    def step(self, action: int, data: dict[str, int]):
        assert action == FakeAction.ACTION6
        self._game.click(int(data["y"]), int(data["x"]))
        return self._frame()

    def _frame(self):
        return SimpleNamespace(
            frame=self._game.frame,
            levels_completed=self._game.levels_completed,
            state="WIN" if self._game.levels_completed >= len(self._levels) else "PLAYING",
        )


class FakeArcade:
    def __init__(self, env: FakeLp85Env) -> None:
        self.env = env

    def make(self, game: str) -> FakeLp85Env:
        assert game == "lp85-305b61c3"
        return self.env


def test_spec_declares_lp85_incremental_contract() -> None:
    """REQ-PHASE4-014: OpenSpec declares the lp85 L2 contract before implementation."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-014" in spec
    assert "SCENARIO-PHASE4-014" in spec
    assert "experiment_3965_lp85_incremental_l2.json" in spec
    assert "blocked_arc_offline_env_unavailable" in spec


def test_solver_reperceives_l2_and_builds_required_artifact() -> None:
    """REQ-PHASE4-014: L2 uses newly perceived buttons and records required bare fields."""
    mod = load_module()
    env = FakeLp85Env(
        [
            FakeLevel(buttons=((4, 4),), sequence=((4, 4), (4, 4))),
            FakeLevel(buttons=((8, 2), (8, 7)), sequence=((8, 7), (8, 2), (8, 7))),
        ]
    )

    result = mod.solve_incremental_levels(
        env,
        FakeAction,
        budget=20,
        baseline_actions=[17, 38, 31],
        stop_after_level=2,
    )
    artifact = mod.build_result_artifact(
        result,
        game="lp85-305b61c3",
        budget=20,
        duration_s=0.25,
        precondition_blocked=False,
    )

    assert result.levels_completed == 2
    assert result.new_levels_solved_this_task == 1
    assert result.per_level_actions == [2, 3]
    assert result.baseline_actions_ref == [17, 38]
    assert result.first_fail_level is None
    assert result.induced_mechanic_held is True
    assert [summary["n_buttons"] for summary in result.level_summaries] == [1, 2]
    assert [entry["level"] for entry in result.solve_log] == [1, 1, 2, 2, 2]
    assert [entry["click"] for entry in result.solve_log[-3:]] == [[8, 7], [8, 2], [8, 7]]
    assert artifact["ACCURACY_levels_solved"] == 2
    assert artifact["new_levels_solved_this_task"] == 1
    assert artifact["induced_mechanic_held"] is True
    assert artifact["real_env_confirmed"] is True
    assert artifact["per_level_actions"] == [2, 3]
    assert artifact["baseline_actions_ref"] == [17, 38]
    assert artifact["honest_verdict"].startswith("complete:")


def test_solver_stops_after_l2_failure_without_chasing_more_levels() -> None:
    """SCENARIO-PHASE4-014: after L2 fails, the scoped run stops at L1."""
    mod = load_module()
    env = FakeLp85Env(
        [
            FakeLevel(buttons=((3, 3),), sequence=((3, 3),)),
            FakeLevel(buttons=((9, 3),), sequence=((9, 3),), fail_on_complete=True),
            FakeLevel(buttons=((12, 12),), sequence=((12, 12),)),
        ]
    )

    result = mod.solve_incremental_levels(
        env,
        FakeAction,
        budget=8,
        baseline_actions=[17, 38, 31],
        stop_after_level=2,
        max_plan_depth=2,
    )
    artifact = mod.build_result_artifact(
        result,
        game="lp85-305b61c3",
        budget=8,
        duration_s=0.5,
        precondition_blocked=False,
    )

    assert result.levels_completed == 1
    assert result.new_levels_solved_this_task == 0
    assert result.first_fail_level == 2
    assert result.per_level_actions == [1]
    assert result.baseline_actions_ref == [17]
    assert result.induced_mechanic_held is False
    assert artifact["ACCURACY_levels_solved"] == 1
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact["induced_mechanic_held"] is False
    assert "lp85_levels1" in artifact["honest_verdict"]


def test_baseline_loader_and_blocked_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-014: blocked offline runs still write the terminal schema."""
    mod = load_module()
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps(
            {
                "games": [
                    {"game_id": "other", "baseline_actions": [99]},
                    {"game_id": "lp85-305b61c3", "baseline_actions": [17, 38]},
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = mod.build_blocked_artifact(
        game="lp85-305b61c3",
        budget=20,
        duration_s=0.1,
        honest_verdict="blocked_arc_offline_env_unavailable",
    )
    output = mod.write_result_artifact(artifact, tmp_path / "experiment_3965.json")

    assert mod.load_baseline_actions("lp85-305b61c3", probe) == [17, 38]
    assert mod.load_baseline_actions("missing", probe) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["ACCURACY_levels_solved"] == 1
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact["per_level_actions"] == []
    assert artifact["baseline_actions_ref"] == []
    assert artifact["induced_mechanic_held"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_defensive_branches_for_counters_budget_and_missing_baselines(tmp_path: Path) -> None:
    """REQ-PHASE4-014: helper fallbacks stay explicit and fail closed."""
    mod = load_module()

    assert mod._levels_completed(SimpleNamespace(frame=[[0]]), SimpleNamespace(_game=SimpleNamespace(levels_completed=3))) == 3
    assert mod._levels_completed(SimpleNamespace(frame=[[0]]), SimpleNamespace(_game=SimpleNamespace(_current_level_index=4))) == 4
    assert mod.load_baseline_actions("lp85-305b61c3", tmp_path / "missing.json") == []
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.load_baseline_actions("lp85-305b61c3", malformed) == []

    env = FakeLp85Env([FakeLevel(buttons=((4, 4),), sequence=((4, 4),))])
    frame = env.reset()
    assert mod.plan_permutation_clicks(
        env,
        FakeAction,
        mod.grid_of(frame),
        [(4, 4)],
        0,
        max_depth=0,
    ) is None

    budget_result = mod.solve_incremental_levels(
        FakeLp85Env([FakeLevel(buttons=((4, 4),), sequence=((4, 4),))]),
        FakeAction,
        budget=0,
        baseline_actions=[17],
        stop_after_level=1,
    )
    assert budget_result.first_fail_level == 1

    too_expensive = mod.solve_incremental_levels(
        FakeLp85Env([FakeLevel(buttons=((4, 4),), sequence=((4, 4), (4, 4)))]),
        FakeAction,
        budget=1,
        baseline_actions=[17],
        stop_after_level=1,
    )
    assert too_expensive.first_fail_level == 1

    stale_env = FakeLp85Env([FakeLevel(buttons=((4, 4),), sequence=((4, 4), (4, 4)))])
    original_planner = mod.plan_permutation_clicks
    mod.plan_permutation_clicks = lambda *args, **kwargs: [(4, 4)]
    try:
        stale_result = mod.solve_incremental_levels(
            stale_env,
            FakeAction,
            budget=2,
            baseline_actions=[17],
            stop_after_level=1,
        )
    finally:
        mod.plan_permutation_clicks = original_planner
    assert stale_result.first_fail_level == 1


def test_fake_import_paths_for_offline_arcade_and_game_action(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-014: real-env import wiring is covered with fake modules."""
    mod = load_module()
    env = FakeLp85Env(
        [
            FakeLevel(buttons=((4, 4),), sequence=((4, 4),)),
            FakeLevel(buttons=((8, 2),), sequence=((8, 2),)),
        ]
    )

    arc_agi = ModuleType("arc_agi")
    arc_agi_base = ModuleType("arc_agi.base")
    arcengine = ModuleType("arcengine")
    arcengine_enums = ModuleType("arcengine.enums")

    class FakeOperationMode:
        OFFLINE = "offline"

    class ImportArcade:
        def __init__(self, arc_api_key: str, operation_mode: str, environments_dir: str) -> None:
            self.arc_api_key = arc_api_key
            self.operation_mode = operation_mode
            self.environments_dir = environments_dir

        def make(self, game: str) -> FakeLp85Env:
            assert game == "lp85-305b61c3"
            return env

    arc_agi.Arcade = ImportArcade
    arc_agi_base.OperationMode = FakeOperationMode
    arcengine_enums.GameAction = FakeAction
    monkeypatch.setitem(sys.modules, "arc_agi", arc_agi)
    monkeypatch.setitem(sys.modules, "arc_agi.base", arc_agi_base)
    monkeypatch.setitem(sys.modules, "arcengine", arcengine)
    monkeypatch.setitem(sys.modules, "arcengine.enums", arcengine_enums)

    arcade = mod._make_offline_arcade()
    assert arcade.operation_mode == "offline"
    artifact = mod.run(
        budget=10,
        arcade_factory=lambda: arcade,
        baseline_actions=[17, 38],
        artifact_path=tmp_path / "import_success.json",
    )
    assert artifact["ACCURACY_levels_solved"] == 2


def test_run_writes_success_and_blocked_artifacts(tmp_path: Path) -> None:
    """REQ-PHASE4-014: run() writes either success or blocked artifacts honestly."""
    mod = load_module()
    env = FakeLp85Env(
        [
            FakeLevel(buttons=((4, 4),), sequence=((4, 4),)),
            FakeLevel(buttons=((8, 2),), sequence=((8, 2),)),
        ]
    )
    success_path = tmp_path / "success.json"
    blocked_path = tmp_path / "blocked.json"

    success = mod.run(
        budget=10,
        arcade_factory=lambda: FakeArcade(env),
        game_action=FakeAction,
        baseline_actions=[17, 38],
        artifact_path=success_path,
    )
    blocked = mod.run(
        budget=10,
        arcade_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
        game_action=FakeAction,
        artifact_path=blocked_path,
    )

    assert json.loads(success_path.read_text(encoding="utf-8")) == success
    assert success["ACCURACY_levels_solved"] == 2
    assert success["new_levels_solved_this_task"] == 1
    assert success["honest_verdict"].startswith("complete:")
    assert json.loads(blocked_path.read_text(encoding="utf-8")) == blocked
    assert blocked["honest_verdict"].startswith("blocked_arc_offline_env_unavailable")
