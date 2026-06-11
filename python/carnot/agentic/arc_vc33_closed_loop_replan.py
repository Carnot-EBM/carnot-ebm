"""Exp 4046 closed-loop replanning over the verified vc33 world model.

Spec refs: REQ-PHASE4-040, SCENARIO-PHASE4-040.

This module keeps the planner grounded in the real environment.  It uses the
verified vc33 `predict(grid, action)` program only for a short lookahead, then
executes one action, re-observes the real grid, and compares the model's next
state against that observation before any further planning.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from carnot.agentic.arc_vc33_hierarchical_search import (
    DEFAULT_MAX_BRANCHING,
    TARGET_GAME,
    TARGET_LEVEL,
    component_landmark_click_actions,
    grid_state_features,
    load_exp4035_preconditions,
    vc33_goal_distance_heuristic,
)


RESULT_NAME = "experiment_4046_closed_loop_replan_over_vc33_wm.json"
INFERENCE_SUBSTRATE = "offline_arc_agi3_closed_loop_replanning_with_real_env_grounding"
PRIOR_WALL_LEVEL = 0
DEFAULT_HORIZON = 3
DEFAULT_MAX_REAL_STEPS = 25
DEFAULT_MAX_PLAN_EXPANSIONS = 256
DEFAULT_DIVERGENCE_THRESHOLD = 0.02
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "new_levels_solved_this_task",
    "closed_loop_broke_wall",
    "per_step_wm_real_divergence_rate",
    "divergence_gate_fired_count",
    "real_env_confirmed",
    "degenerate_plan_refused",
    "inference_substrate",
)

Action = tuple[int, int, int]
PredictFn = Callable[[np.ndarray, Action], np.ndarray]
GoalPredicate = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class EnvObservation:
    """One real vc33 observation after reset or a single action."""

    grid: np.ndarray
    levels_completed: int


class VC33Env(Protocol):
    def reset(self) -> EnvObservation:
        """Return the current level's initial observation."""

    def step(self, action: Action) -> EnvObservation:
        """Execute one click action and return the re-observed state."""


@dataclass(frozen=True)
class RecedingHorizonPlan:
    """One short-horizon WM plan from the current real observation."""

    solved_in_wm: bool
    actions: tuple[Action, ...]
    nodes_expanded: int
    bottleneck: str
    final_grid: np.ndarray


@dataclass(frozen=True)
class ClosedLoopOutcome:
    """Grounded controller evidence used to build the terminal artifact."""

    actions_taken: tuple[Action, ...]
    levels_completed_after: int
    real_env_confirmed: bool
    closed_loop_broke_wall: bool
    per_step_wm_real_divergence_rate: float
    divergence_gate_fired_count: int
    degenerate_plan_refused: bool
    nodes_expanded: int
    steps: int
    bottleneck: str
    wm_goal_claims: int
    divergence_threshold: float


def _grid_key(grid: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int16))
    shape = ",".join(str(dim) for dim in arr.shape).encode("ascii")
    return hashlib.sha1(arr.tobytes() + shape).hexdigest()[:16]


def _goal_holds(grid: np.ndarray, is_goal: GoalPredicate) -> bool:
    return bool(is_goal(grid_state_features(np.asarray(grid, dtype=np.int16))))


def grid_divergence(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Return normalized cell disagreement between WM prediction and reality."""

    left = np.asarray(predicted)
    right = np.asarray(observed)
    if left.shape != right.shape:
        return 1.0
    if left.size == 0:
        return 0.0
    return float(np.mean(left != right))


def is_degenerate_repeat_plan(actions: tuple[Action, ...] | list[Action]) -> bool:
    """Detect WM goal plans made only of one repeated action."""

    if len(actions) < 2:
        return False
    first = tuple(actions[0])
    return all(tuple(action) == first for action in actions[1:])


def bounded_receding_horizon_search(
    start_grid: np.ndarray,
    *,
    predict: PredictFn,
    is_goal: GoalPredicate,
    horizon: int = DEFAULT_HORIZON,
    max_expansions: int = DEFAULT_MAX_PLAN_EXPANSIONS,
    max_branching: int = DEFAULT_MAX_BRANCHING,
) -> RecedingHorizonPlan:
    """Run a bounded best-first WM lookahead and return a first-step plan."""

    horizon = max(0, int(horizon))
    max_expansions = max(0, int(max_expansions))
    max_branching = max(1, int(max_branching))
    start = np.asarray(start_grid, dtype=np.int16)
    if _goal_holds(start, is_goal):
        return RecedingHorizonPlan(True, (), 0, "", start.copy())

    counter = 0
    start_state = grid_state_features(start)
    start_score = float(vc33_goal_distance_heuristic(start_state))
    frontier: list[tuple[float, int, int, np.ndarray, tuple[Action, ...]]] = [
        (start_score, 0, counter, start.copy(), ())
    ]
    best_depth: dict[str, int] = {_grid_key(start): 0}
    best_actions: tuple[Action, ...] = ()
    best_grid = start.copy()
    best_score = start_score
    nodes_expanded = 0
    last_grid = start.copy()

    while frontier and nodes_expanded < max_expansions:
        _, depth, _, grid, actions = heapq.heappop(frontier)
        key = _grid_key(grid)
        if depth != best_depth.get(key):  # pragma: no cover - stale queue defense
            continue
        last_grid = grid
        state = grid_state_features(grid)
        score = float(vc33_goal_distance_heuristic(state))
        if actions and score < best_score:
            best_score = score
            best_actions = actions
            best_grid = grid.copy()
        if bool(is_goal(state)):
            return RecedingHorizonPlan(True, actions, nodes_expanded, "", grid.copy())
        if depth >= horizon:
            continue

        nodes_expanded += 1
        for action in component_landmark_click_actions(grid, max_actions=max_branching):
            try:
                predicted = np.asarray(predict(grid.copy(), action), dtype=np.int16)
            except Exception:
                continue
            if predicted.shape != grid.shape or np.array_equal(predicted, grid):
                continue
            child_depth = depth + 1
            child_key = _grid_key(predicted)
            if child_depth >= best_depth.get(child_key, 1_000_000_000):
                continue
            best_depth[child_key] = child_depth
            child_state = grid_state_features(predicted)
            priority = child_depth + float(vc33_goal_distance_heuristic(child_state))
            counter += 1
            heapq.heappush(
                frontier,
                (priority, child_depth, counter, predicted.copy(), actions + (tuple(action),)),
            )

    if best_actions:
        return RecedingHorizonPlan(
            False,
            best_actions,
            nodes_expanded,
            "horizon_exhausted",
            best_grid.copy(),
        )
    bottleneck = "expansion_bound_exhausted" if frontier else "frontier_exhausted"
    return RecedingHorizonPlan(False, (), nodes_expanded, bottleneck, last_grid.copy())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def run_closed_loop_controller(
    env: VC33Env,
    *,
    predict: PredictFn,
    is_goal: GoalPredicate,
    horizon: int = DEFAULT_HORIZON,
    max_plan_expansions: int = DEFAULT_MAX_PLAN_EXPANSIONS,
    max_branching: int = DEFAULT_MAX_BRANCHING,
    max_real_steps: int = DEFAULT_MAX_REAL_STEPS,
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    initial_observation: EnvObservation | None = None,
) -> ClosedLoopOutcome:
    """Execute receding-horizon WM plans one real action at a time."""

    observation = initial_observation if initial_observation is not None else env.reset()
    baseline_level = int(observation.levels_completed)
    actions_taken: list[Action] = []
    divergences: list[float] = []
    gate_fired = 0
    total_nodes = 0
    degenerate_refused = False
    wm_goal_claims = 0
    bottleneck = "max_real_steps_exhausted"

    for _ in range(max(0, int(max_real_steps))):
        plan = bounded_receding_horizon_search(
            observation.grid,
            predict=predict,
            is_goal=is_goal,
            horizon=horizon,
            max_expansions=max_plan_expansions,
            max_branching=max_branching,
        )
        total_nodes += plan.nodes_expanded
        if plan.solved_in_wm:
            wm_goal_claims += 1
        if plan.solved_in_wm and is_degenerate_repeat_plan(plan.actions):
            degenerate_refused = True
            bottleneck = "degenerate_plan_refused"
            break
        if not plan.actions:
            bottleneck = plan.bottleneck or "no_action_available"
            break

        action = plan.actions[0]
        try:
            predicted_next = np.asarray(predict(observation.grid.copy(), action), dtype=np.int16)
        except Exception:
            bottleneck = "wm_predict_failed"
            break
        next_observation = env.step(action)
        actions_taken.append(action)
        divergence = grid_divergence(predicted_next, next_observation.grid)
        divergences.append(divergence)
        if divergence > float(divergence_threshold):
            gate_fired += 1

        observation = next_observation
        if int(observation.levels_completed) > baseline_level:
            bottleneck = ""
            break
        if divergence > float(divergence_threshold):
            bottleneck = "wm_real_divergence_gate_fired"
            break
    else:
        bottleneck = "max_real_steps_exhausted"

    confirmed = int(observation.levels_completed) > baseline_level
    return ClosedLoopOutcome(
        actions_taken=tuple(actions_taken),
        levels_completed_after=int(observation.levels_completed),
        real_env_confirmed=bool(confirmed),
        closed_loop_broke_wall=bool(confirmed),
        per_step_wm_real_divergence_rate=round(_mean(divergences), 6),
        divergence_gate_fired_count=int(gate_fired),
        degenerate_plan_refused=bool(degenerate_refused),
        nodes_expanded=int(total_nodes),
        steps=len(actions_taken),
        bottleneck=bottleneck,
        wm_goal_claims=int(wm_goal_claims),
        divergence_threshold=float(divergence_threshold),
    )


def _actions_as_json(actions: tuple[Action, ...]) -> list[list[int]]:
    return [[int(part) for part in action] for action in actions]


def build_exp4046_artifact(
    outcome: ClosedLoopOutcome,
    *,
    duration_s: float,
    goal_predicate_precision: float,
) -> dict[str, Any]:
    """Normalize closed-loop evidence into the required result artifact."""

    if outcome.real_env_confirmed:
        verdict = (
            f"complete: closed_loop_solved_vc33_L{outcome.levels_completed_after}_"
            "real_env_confirmed"
        )
        new_levels = max(1, int(outcome.levels_completed_after) - PRIOR_WALL_LEVEL)
    else:
        new_levels = 0
        if outcome.bottleneck == "wm_real_divergence_gate_fired" or (
            outcome.per_step_wm_real_divergence_rate > 0.0
            and outcome.divergence_gate_fired_count > 0
        ):
            verdict = (
                "complete: closed_loop_no_solve_vc33_wm_sim2real_ceiling_divergence_"
                f"{outcome.per_step_wm_real_divergence_rate:.3f}"
            )
        else:
            bottleneck = outcome.bottleneck or "no_real_env_confirmation"
            verdict = f"complete: closed_loop_no_solve_vc33_{bottleneck}"

    return {
        "experiment": "experiment_4046_closed_loop_replan_over_vc33_wm",
        "schema": "carnot.experiment_4046_closed_loop_replan_over_vc33_wm.v1",
        "game": TARGET_GAME,
        "target_level": TARGET_LEVEL,
        "prior_wall_level": PRIOR_WALL_LEVEL,
        "honest_verdict": verdict,
        "new_levels_solved_this_task": int(new_levels),
        "closed_loop_broke_wall": bool(outcome.closed_loop_broke_wall),
        "per_step_wm_real_divergence_rate": float(outcome.per_step_wm_real_divergence_rate),
        "divergence_gate_fired_count": int(outcome.divergence_gate_fired_count),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "degenerate_plan_refused": bool(outcome.degenerate_plan_refused),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levels_completed_after": int(outcome.levels_completed_after),
        "action_plan": _actions_as_json(outcome.actions_taken),
        "action_count": len(outcome.actions_taken),
        "nodes_expanded": int(outcome.nodes_expanded),
        "steps": int(outcome.steps),
        "wm_goal_claims": int(outcome.wm_goal_claims),
        "divergence_threshold": float(outcome.divergence_threshold),
        "goal_predicate_heldout_precision": float(goal_predicate_precision),
        "duration_s": round(float(duration_s), 3),
        "bottleneck": outcome.bottleneck,
        "diagnosis": (
            "Closed-loop grounding broke the vc33 wall where open-loop WM search failed."
            if outcome.real_env_confirmed
            else "Closed-loop grounding did not produce a real-env-confirmed vc33 solve."
        ),
    }


def blocked_artifact(
    honest_verdict: str,
    duration_s: float,
    *,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Return a terminal blocked artifact with the Exp 4046 required fields."""

    return {
        "experiment": "experiment_4046_closed_loop_replan_over_vc33_wm",
        "schema": "carnot.experiment_4046_closed_loop_replan_over_vc33_wm.v1",
        "game": TARGET_GAME,
        "target_level": TARGET_LEVEL,
        "prior_wall_level": PRIOR_WALL_LEVEL,
        "honest_verdict": str(honest_verdict),
        "new_levels_solved_this_task": 0,
        "closed_loop_broke_wall": False,
        "per_step_wm_real_divergence_rate": 0.0,
        "divergence_gate_fired_count": 0,
        "real_env_confirmed": False,
        "degenerate_plan_refused": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levels_completed_after": PRIOR_WALL_LEVEL,
        "action_plan": [],
        "action_count": 0,
        "nodes_expanded": 0,
        "steps": 0,
        "wm_goal_claims": 0,
        "duration_s": round(float(duration_s), 3),
        "precondition_errors": list(errors or []),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors for Exp 4046 artifacts."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be a terminal-prefix string")
    for field in ("new_levels_solved_this_task", "divergence_gate_fired_count"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    for field in ("closed_loop_broke_wall", "real_env_confirmed", "degenerate_plan_refused"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")
    if (
        "per_step_wm_real_divergence_rate" in artifact
        and type(artifact["per_step_wm_real_divergence_rate"]) is not float
    ):
        errors.append("per_step_wm_real_divergence_rate must be a bare float")
    if "inference_substrate" in artifact and type(artifact["inference_substrate"]) is not str:
        errors.append("inference_substrate must be a bare string")
    return errors


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Write stable JSON for downstream conductor reconciliation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class ArcVC33EnvAdapter:  # pragma: no cover - exercised by required experiment command
    """Small adapter around the ARC SDK's vc33 environment."""

    def __init__(self, repo_root: Path) -> None:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
        from arcengine.enums import GameAction

        self._game_action = GameAction
        self._arc = Arcade(
            arc_api_key="",
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(repo_root / "environment_files"),
        )
        self._env = self._arc.make("vc33-5430563c")

    def reset(self) -> EnvObservation:
        return self._observe(self._env.reset())

    def step(self, action: Action) -> EnvObservation:
        frame = self._env.step(
            self._game_action.ACTION6,
            data={"x": int(action[1]), "y": int(action[2])},
        )
        return self._observe(frame)

    @staticmethod
    def _observe(frame: Any) -> EnvObservation:
        from carnot.agentic.arc_agi3_world_model import grid_of

        return EnvObservation(
            grid=grid_of(frame),
            levels_completed=int(getattr(frame, "levels_completed", 0) or 0),
        )


def _make_vc33_env_adapter(repo_root: Path) -> VC33Env:  # pragma: no cover - real ARC precondition
    return ArcVC33EnvAdapter(repo_root)


def run(
    *,
    repo_root: Path | None = None,
    env_factory: Callable[[Path], VC33Env] | None = None,
    write: bool = True,
    horizon: int = DEFAULT_HORIZON,
    max_plan_expansions: int = DEFAULT_MAX_PLAN_EXPANSIONS,
    max_branching: int = DEFAULT_MAX_BRANCHING,
    max_real_steps: int = DEFAULT_MAX_REAL_STEPS,
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    """Run Exp 4046 and optionally write its terminal artifact."""

    started = time.time()
    root = repo_root or Path(__file__).resolve().parents[3]
    make_env = env_factory or _make_vc33_env_adapter
    try:
        env = make_env(root)
        initial = env.reset()
    except Exception as exc:
        artifact = blocked_artifact(
            "blocked_arc_env_unreachable",
            time.time() - started,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        if write:
            write_artifact(artifact, root / "results" / RESULT_NAME)
        return artifact

    preconditions = load_exp4035_preconditions(root)
    if not preconditions.ok or preconditions.predict is None or preconditions.is_goal is None:
        artifact = blocked_artifact(
            "blocked_vc33_goal_predicate_or_wm_missing",
            time.time() - started,
            errors=preconditions.errors,
        )
        if write:
            write_artifact(artifact, root / "results" / RESULT_NAME)
        return artifact

    outcome = run_closed_loop_controller(
        env,
        predict=preconditions.predict,
        is_goal=preconditions.is_goal,
        horizon=horizon,
        max_plan_expansions=max_plan_expansions,
        max_branching=max_branching,
        max_real_steps=max_real_steps,
        divergence_threshold=divergence_threshold,
        initial_observation=initial,
    )
    artifact = build_exp4046_artifact(
        outcome,
        duration_s=time.time() - started,
        goal_predicate_precision=preconditions.goal_predicate_precision,
    )
    schema_errors = artifact_schema_errors(artifact)
    if schema_errors:  # pragma: no cover - defensive contract guard
        raise ValueError("; ".join(schema_errors))
    if write:
        write_artifact(artifact, root / "results" / RESULT_NAME)
    return artifact
