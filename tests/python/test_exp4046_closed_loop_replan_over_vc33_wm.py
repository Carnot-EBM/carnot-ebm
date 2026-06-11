"""Tests for Exp 4046 closed-loop vc33 replanning.

Spec refs: REQ-PHASE4-040, SCENARIO-PHASE4-040.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot.agentic.arc_vc33_closed_loop_replan import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    EnvObservation,
    artifact_schema_errors,
    blocked_artifact,
    bounded_receding_horizon_search,
    build_exp4046_artifact,
    grid_divergence,
    is_degenerate_repeat_plan,
    run,
    run_closed_loop_controller,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _aligned_grid() -> np.ndarray:
    grid = np.full((16, 16), 3, dtype=np.int16)
    grid[1, :] = 7
    grid[3:5, 4:6] = 11
    grid[10:12, 4:6] = 11
    return grid


def _misaligned_grid() -> np.ndarray:
    grid = _aligned_grid()
    grid[10:12, 4:6] = 3
    grid[10:12, 9:11] = 11
    grid[1, 12:] = 5
    return grid


def _goal(state: dict[str, object]) -> bool:
    return bool(state["target_color_pairs"] > 0 and state["misaligned_target_pairs"] == 0)


def _write_preconditions(root: Path, *, precision: float = 1.0) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4034_vc33_goal_predicate_induction.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: vc33_goal_predicate_induced_heldout_precision_1.000",
                "goal_predicate_heldout_precision": precision,
                "goal_predicate_code": (
                    "def is_goal(state):\n"
                    '    return state["target_color_pairs"] > 0 and '
                    'state["misaligned_target_pairs"] == 0\n'
                ),
                "game": "vc33",
                "inference_substrate": "offline_arc_agi3_goal_predicate_induction_no_oracle",
            }
        ),
        encoding="utf-8",
    )
    (results / "arc3_vc33_world_model_program.py").write_text(
        "def predict(grid, action):\n"
        "    g = grid.copy()\n"
        "    if action[0] == 6 and int(action[1]) == 9 and int(action[2]) == 10:\n"
        "        g[10:12, 9:11] = 3\n"
        "        g[10:12, 4:6] = 11\n"
        "    return g\n",
        encoding="utf-8",
    )


class FakeVC33Env:
    def __init__(
        self, start: np.ndarray, transitions: dict[tuple[int, int, int], EnvObservation]
    ) -> None:
        self.start = EnvObservation(start.copy(), 0)
        self.transitions = transitions
        self.actions: list[tuple[int, int, int]] = []
        self.current = self.start

    def reset(self) -> EnvObservation:
        self.current = self.start
        return self.current

    def step(self, action: tuple[int, int, int]) -> EnvObservation:
        self.actions.append(tuple(action))
        self.current = self.transitions.get(
            tuple(action), EnvObservation(self.current.grid.copy(), 0)
        )
        return self.current


def test_req_phase4_040_spec_declares_exp4046_contract() -> None:
    """REQ-PHASE4-040: OpenSpec declares closed-loop vc33 grounding fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-040" in spec
    assert "SCENARIO-PHASE4-040" in spec
    assert "experiment_4046_closed_loop_replan_over_vc33_wm.json" in spec
    assert "blocked_arc_env_unreachable" in spec
    assert "blocked_vc33_goal_predicate_or_wm_missing" in spec
    assert "single-action-repeat" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_040_receding_horizon_executes_first_step_and_confirms_real_env() -> None:
    """SCENARIO-PHASE4-040: controller executes one real action, re-observes, and confirms."""

    start = _misaligned_grid()
    solved = _aligned_grid()

    def predict(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        return solved.copy() if tuple(action) == (6, 9, 10) else grid.copy()

    env = FakeVC33Env(start, {(6, 9, 10): EnvObservation(solved, 1)})
    outcome = run_closed_loop_controller(
        env,
        predict=predict,
        is_goal=_goal,
        horizon=3,
        max_plan_expansions=64,
        max_real_steps=3,
    )

    assert env.actions == [(6, 9, 10)]
    assert outcome.real_env_confirmed is True
    assert outcome.levels_completed_after == 1
    assert outcome.closed_loop_broke_wall is True
    assert outcome.per_step_wm_real_divergence_rate == 0.0
    assert outcome.divergence_gate_fired_count == 0


def test_scenario_phase4_040_divergence_gate_stops_wm_exploit() -> None:
    """SCENARIO-PHASE4-040: WM-real mismatch fires the trust gate and reports a ceiling."""

    start = _misaligned_grid()
    wm_solved = _aligned_grid()
    real_after = start.copy()
    real_after[5, 5] = 8

    def predict(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        return wm_solved.copy() if tuple(action) == (6, 9, 10) else grid.copy()

    env = FakeVC33Env(start, {(6, 9, 10): EnvObservation(real_after, 0)})
    outcome = run_closed_loop_controller(
        env,
        predict=predict,
        is_goal=_goal,
        divergence_threshold=0.01,
        horizon=3,
        max_plan_expansions=64,
        max_real_steps=3,
    )
    artifact = build_exp4046_artifact(
        outcome,
        duration_s=0.25,
        goal_predicate_precision=1.0,
    )

    assert outcome.real_env_confirmed is False
    assert outcome.closed_loop_broke_wall is False
    assert outcome.divergence_gate_fired_count == 1
    assert outcome.per_step_wm_real_divergence_rate > 0.01
    assert artifact["honest_verdict"].startswith(
        "complete: closed_loop_no_solve_vc33_wm_sim2real_ceiling_divergence_"
    )
    assert artifact["per_step_wm_real_divergence_rate"] == outcome.per_step_wm_real_divergence_rate


def test_scenario_phase4_040_refuses_degenerate_repeat_plan() -> None:
    """SCENARIO-PHASE4-040: a single-action-repeat WM goal plan is rejected."""

    start = _misaligned_grid()
    progress = start.copy()
    progress[0, 0] = 4
    solved = _aligned_grid()
    repeated = (6, 9, 10)

    def predict(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        if tuple(action) != repeated:
            return grid.copy()
        if int(grid[0, 0]) == 4:
            return solved.copy()
        next_grid = grid.copy()
        next_grid[0, 0] = 4
        return next_grid

    plan = bounded_receding_horizon_search(
        start,
        predict=predict,
        is_goal=_goal,
        horizon=3,
        max_expansions=64,
    )
    env = FakeVC33Env(start, {repeated: EnvObservation(progress, 0)})
    outcome = run_closed_loop_controller(
        env,
        predict=predict,
        is_goal=_goal,
        horizon=3,
        max_plan_expansions=64,
        max_real_steps=2,
    )

    assert plan.solved_in_wm is True
    assert plan.actions == (repeated, repeated)
    assert is_degenerate_repeat_plan(plan.actions) is True
    assert outcome.degenerate_plan_refused is True
    assert outcome.actions_taken == ()
    assert outcome.bottleneck == "degenerate_plan_refused"


def test_req_phase4_040_artifact_schema_and_blocked_paths_are_bare(tmp_path: Path) -> None:
    """REQ-PHASE4-040: Exp 4046 artifacts preserve required bare fields."""

    start = _misaligned_grid()
    solved = _aligned_grid()
    outcome = run_closed_loop_controller(
        FakeVC33Env(start, {(6, 9, 10): EnvObservation(solved, 1)}),
        predict=lambda grid, action: solved.copy() if tuple(action) == (6, 9, 10) else grid.copy(),
        is_goal=_goal,
        horizon=3,
        max_plan_expansions=64,
        max_real_steps=3,
    )
    artifact = build_exp4046_artifact(outcome, duration_s=0.125, goal_predicate_precision=1.0)

    assert artifact["honest_verdict"] == "complete: closed_loop_solved_vc33_L1_real_env_confirmed"
    assert artifact["new_levels_solved_this_task"] == 1
    assert artifact["closed_loop_broke_wall"] is True
    assert artifact["per_step_wm_real_divergence_rate"] == 0.0
    assert artifact["divergence_gate_fired_count"] == 0
    assert artifact["real_env_confirmed"] is True
    assert artifact["degenerate_plan_refused"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["honest_verdict"] = "done"
    bad["new_levels_solved_this_task"] = True
    bad["closed_loop_broke_wall"] = 1
    bad["per_step_wm_real_divergence_rate"] = "0"
    bad["divergence_gate_fired_count"] = False
    bad["real_env_confirmed"] = 1
    bad["degenerate_plan_refused"] = 0
    bad["inference_substrate"] = None

    errors = artifact_schema_errors(bad)
    assert any("honest_verdict" in err for err in errors)
    assert any("new_levels_solved_this_task" in err for err in errors)
    assert any("closed_loop_broke_wall" in err for err in errors)
    assert any("per_step_wm_real_divergence_rate" in err for err in errors)
    assert any("divergence_gate_fired_count" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("degenerate_plan_refused" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)
    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))

    blocked_env = blocked_artifact(
        "blocked_arc_env_unreachable", 0.1, errors=["arc offline missing"]
    )
    blocked_wm = blocked_artifact("blocked_vc33_goal_predicate_or_wm_missing", 0.1)
    output = run(
        repo_root=tmp_path,
        env_factory=lambda root: (_ for _ in ()).throw(RuntimeError("no arc")),
        write=True,
    )

    assert blocked_env["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked_wm["honest_verdict"] == "blocked_vc33_goal_predicate_or_wm_missing"
    assert artifact_schema_errors(blocked_env) == []
    assert output["honest_verdict"] == "blocked_arc_env_unreachable"
    assert (tmp_path / "results" / "experiment_4046_closed_loop_replan_over_vc33_wm.json").exists()


def test_scenario_phase4_040_run_blocks_without_goal_predicate_or_wm_and_writes_confirmed(
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-040: run blocks after env reachability or writes confirmed solve."""

    start = _misaligned_grid()
    solved = _aligned_grid()
    missing = run(
        repo_root=tmp_path,
        env_factory=lambda root: FakeVC33Env(start, {}),
        write=False,
    )

    assert missing["honest_verdict"] == "blocked_vc33_goal_predicate_or_wm_missing"

    _write_preconditions(tmp_path)
    confirmed = run(
        repo_root=tmp_path,
        env_factory=lambda root: FakeVC33Env(start, {(6, 9, 10): EnvObservation(solved, 1)}),
        write=True,
        horizon=3,
        max_plan_expansions=64,
        max_real_steps=3,
    )

    assert confirmed["honest_verdict"] == "complete: closed_loop_solved_vc33_L1_real_env_confirmed"
    assert confirmed["real_env_confirmed"] is True
    assert artifact_schema_errors(confirmed) == []
    assert (
        json.loads(
            (
                tmp_path / "results" / "experiment_4046_closed_loop_replan_over_vc33_wm.json"
            ).read_text(encoding="utf-8")
        )
        == confirmed
    )


def test_req_phase4_040_grid_divergence_handles_shape_mismatch() -> None:
    """REQ-PHASE4-040: divergence is a normalized per-step WM-real diagnostic."""

    same = np.zeros((2, 2), dtype=np.int16)
    one_changed = same.copy()
    one_changed[0, 0] = 1

    assert grid_divergence(same, same.copy()) == 0.0
    assert grid_divergence(same, one_changed) == 0.25
    assert grid_divergence(same, np.zeros((1, 4), dtype=np.int16)) == 1.0
    assert is_degenerate_repeat_plan([(6, 1, 1)]) is False
    assert is_degenerate_repeat_plan([(6, 1, 1), (6, 1, 2)]) is False
    assert (
        grid_divergence(np.zeros((0, 0), dtype=np.int16), np.zeros((0, 0), dtype=np.int16)) == 0.0
    )


def test_req_phase4_040_planner_and_controller_cover_defensive_paths(tmp_path: Path) -> None:
    """REQ-PHASE4-040: bounded planner/controller failure paths stay terminal."""

    start = _misaligned_grid()
    solved = _aligned_grid()

    already_goal = bounded_receding_horizon_search(
        solved,
        predict=lambda grid, action: grid.copy(),
        is_goal=_goal,
    )
    zero_horizon = bounded_receding_horizon_search(
        start,
        predict=lambda grid, action: solved.copy(),
        is_goal=_goal,
        horizon=0,
    )

    def broken_predict(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        raise RuntimeError("wm failed")

    broken_plan = bounded_receding_horizon_search(
        start,
        predict=broken_predict,
        is_goal=_goal,
        max_expansions=4,
    )

    def constant_child(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        child = grid.copy()
        child[0, 0] = 4
        return child

    duplicate_child = bounded_receding_horizon_search(
        start,
        predict=constant_child,
        is_goal=_goal,
        horizon=1,
        max_expansions=8,
    )

    no_action = run_closed_loop_controller(
        FakeVC33Env(start, {}),
        predict=lambda grid, action: grid.copy(),
        is_goal=_goal,
        max_real_steps=2,
    )

    calls = {"count": 0}

    def raises_after_plan(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        if tuple(action) == (6, 9, 10):
            calls["count"] += 1
            if calls["count"] > 1:
                raise RuntimeError("commit prediction failed")
            return solved.copy()
        return grid.copy()

    predict_failed = run_closed_loop_controller(
        FakeVC33Env(start, {(6, 9, 10): EnvObservation(solved, 1)}),
        predict=raises_after_plan,
        is_goal=_goal,
        max_real_steps=2,
    )
    exhausted = run_closed_loop_controller(
        FakeVC33Env(start, {}),
        predict=lambda grid, action: grid.copy(),
        is_goal=_goal,
        max_real_steps=0,
    )
    no_solve_artifact = build_exp4046_artifact(
        no_action, duration_s=0.0, goal_predicate_precision=1.0
    )
    blocked_wm_written = run(
        repo_root=tmp_path,
        env_factory=lambda root: FakeVC33Env(start, {}),
        write=True,
    )

    assert already_goal.solved_in_wm is True
    assert already_goal.actions == ()
    assert zero_horizon.bottleneck == "frontier_exhausted"
    assert broken_plan.bottleneck == "frontier_exhausted"
    assert duplicate_child.bottleneck == "horizon_exhausted"
    assert duplicate_child.actions
    assert no_action.bottleneck == "frontier_exhausted"
    assert predict_failed.bottleneck == "wm_predict_failed"
    assert exhausted.bottleneck == "max_real_steps_exhausted"
    assert (
        no_solve_artifact["honest_verdict"]
        == "complete: closed_loop_no_solve_vc33_frontier_exhausted"
    )
    assert blocked_wm_written["honest_verdict"] == "blocked_vc33_goal_predicate_or_wm_missing"
    assert (tmp_path / "results" / "experiment_4046_closed_loop_replan_over_vc33_wm.json").exists()
