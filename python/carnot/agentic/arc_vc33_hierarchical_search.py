"""Exp 4035 hierarchical search over the verified vc33 world model.

Spec refs: REQ-PHASE4-037, SCENARIO-PHASE4-037.

The module keeps the Exp 4035 claim narrow: the verified `predict(grid, action)`
program is the simulator, the Exp 4034 predicate is the terminal goal, and the
planner uses grid-derived landmarks plus a coded distance heuristic.  It does
not replay known vc33 coordinate traces or install per-level macros; a solve
only counts after the same action list advances the offline environment's
`levels_completed` counter.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from carnot.agentic.arc_goal_predicate_separation import compile_goal_predicate
from carnot.agentic.arc_heuristic_search_over_verified_wm import (
    SearchResult,
    best_first_search,
)
from carnot.agentic.arc_vc33_goal_predicate_induction import (
    TARGET_COLORS,
    _component_bboxes,
    vc33_grid_state_features,
)


RESULT_NAME = "experiment_4035_hierarchical_search_over_vc33_wm.json"
INFERENCE_SUBSTRATE = "offline_arc_agi3_planning_search_over_verified_world_model"
DEFAULT_MAX_EXPANSIONS = 4096
DEFAULT_MAX_BRANCHING = 96
TARGET_GAME = "vc33"
TARGET_LEVEL = 1
PRIOR_WALL_LEVEL = 0
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "new_levels_solved_this_task",
    "search_layer_generalizes",
    "heuristic_was_non_bespoke",
    "nodes_expanded",
    "branching_factor",
    "subgoal_decomposition_used",
    "real_env_confirmed",
    "inference_substrate",
)
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
PredictFn = Callable[[np.ndarray, tuple[int, ...]], np.ndarray]
GoalPredicate = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class Subgoal:
    """One ordered landmark predicate for the hierarchical planner."""

    name: str
    predicate: GoalPredicate


@dataclass(frozen=True)
class Exp4035Preconditions:
    """Loaded Exp 4034 predicate plus verified vc33 simulator, or errors."""

    ok: bool
    predict: PredictFn | None
    is_goal: GoalPredicate | None
    goal_predicate_precision: float
    goal_predicate_code: str
    errors: list[str]


@dataclass(frozen=True)
class GridSearchOutcome:
    """Aggregated hierarchical search evidence for artifact construction."""

    search: SearchResult
    nodes_expanded: int
    branching_factor: float
    subgoals_attempted: int
    subgoals_reached: int


def _grid_key(grid: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int16))
    shape = ",".join(str(dim) for dim in arr.shape).encode("ascii")
    return hashlib.sha1(arr.tobytes() + shape).hexdigest()[:16]


def progress_bar_gap(grid: np.ndarray) -> int:
    """Estimate remaining progress from a rasterized top-row bar, when present."""

    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return 0
    width = int(arr.shape[1])
    top = arr[0]
    old = top[0]
    consumed = 0
    for col in range(width - 1, -1, -1):
        if top[col] == old:
            break
        consumed += 1
    steps_total = max(1, int(np.floor(width * 25.0 / 32.0 + 0.5)))
    step = int(np.floor(consumed * steps_total / float(width) + 0.5))
    return max(0, steps_total - step)


def _target_alignment_distance(grid: np.ndarray) -> int:
    total = 0
    for color in TARGET_COLORS:
        boxes = _component_bboxes(np.asarray(grid), int(color))
        if len(boxes) < 2:
            continue
        centers = [(x0 + x1, y0 + y1) for x0, y0, x1, y1, _ in boxes]
        best = None
        for idx, (cx, cy) in enumerate(centers):
            for other_cx, other_cy in centers[idx + 1 :]:
                distance = min(abs(cx - other_cx), abs(cy - other_cy))
                best = distance if best is None else min(best, distance)
        total += int(best or 0)
    return int(total)


def grid_state_features(grid: np.ndarray) -> dict[str, Any]:
    """Build the feature dictionary consumed by Exp 4034's goal predicate."""

    features = dict(vc33_grid_state_features(np.asarray(grid)))
    missing_pair = 0 if bool(features.get("has_target_pair")) else 1
    misaligned = int(features.get("misaligned_target_pairs", 0) or 0)
    unsatisfied = missing_pair + misaligned
    features["unsatisfied_targets"] = int(unsatisfied)
    features["unmet_goal_components"] = int(unsatisfied)
    features["manhattan_to_target"] = _target_alignment_distance(np.asarray(grid))
    features["progress_bar_delta_to_goal"] = progress_bar_gap(np.asarray(grid))
    return features


def vc33_goal_distance_heuristic(state: dict[str, Any]) -> float:
    """Rank states by unmet predicate components, then distance and progress."""

    unmet = float(state.get("unmet_goal_components", state.get("unsatisfied_targets", 0)) or 0)
    manhattan = float(state.get("manhattan_to_target", 0) or 0)
    progress_gap_value = float(state.get("progress_bar_delta_to_goal", 0) or 0)
    return unmet * 1000.0 + manhattan + 0.01 * max(0.0, progress_gap_value)


def _component_points(box: tuple[int, int, int, int, int]) -> list[tuple[int, int]]:
    x0, y0, x1, y1, size = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    points = [
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1),
        (cx, cy),
        (x0, cy),
        (x1, cy),
        (cx, y0),
        (cx, y1),
    ]
    if size > 32:
        xs = sorted({x0, (2 * x0 + x1) // 3, cx, (x0 + 2 * x1) // 3, x1})
        ys = sorted({y0, (2 * y0 + y1) // 3, cy, (y0 + 2 * y1) // 3, y1})
        points.extend((x, y) for x in xs for y in ys)
    return points


def component_landmark_click_actions(grid: np.ndarray, *, max_actions: int = DEFAULT_MAX_BRANCHING) -> list[tuple[int, int, int]]:
    """Return deterministic click landmarks from visible grid components."""

    arr = np.asarray(grid)
    values, counts = np.unique(arr, return_counts=True)
    count_by_value = {int(value): int(count) for value, count in zip(values, counts)}
    target_set = {int(color) for color in TARGET_COLORS}
    ordered_values = sorted(
        (int(value) for value in values),
        key=lambda value: (0 if value in target_set else 1, count_by_value[value], value),
    )
    actions: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    height, width = arr.shape
    for color in ordered_values:
        boxes = sorted(_component_bboxes(arr, color), key=lambda box: (box[4], box[1], box[0]))
        for box in boxes:
            for x, y in _component_points(box):
                if not (0 <= x < width and 0 <= y < height) or y == 0:
                    continue
                action = (6, int(x), int(y))
                if action in seen:
                    continue
                seen.add(action)
                actions.append(action)
                if len(actions) >= max_actions:
                    return actions
    return actions


def decompose_goal_predicate(start_features: dict[str, Any]) -> list[Subgoal]:
    """Create ordered landmarks from the visible components of the goal predicate."""

    misaligned = int(start_features.get("misaligned_target_pairs", 0) or 0)
    subgoals: list[Subgoal] = []
    for target in range(max(0, misaligned - 1), -1, -1):
        subgoals.append(
            Subgoal(
                name=f"reduce_misaligned_target_pairs_to_{target}",
                predicate=lambda state, target=target: int(
                    state.get("misaligned_target_pairs", 0) or 0
                )
                <= target,
            )
        )
    subgoals.append(
        Subgoal(
            name="full_goal_predicate",
            predicate=lambda state: int(state.get("unsatisfied_targets", 0) or 0) == 0,
        )
    )
    return subgoals


class VC33VerifiedWorldModel:
    """Adapter that lets generic best-first search expand vc33 grid states."""

    def __init__(
        self,
        start_grid: np.ndarray,
        predict: PredictFn,
        *,
        max_branching: int = DEFAULT_MAX_BRANCHING,
    ) -> None:
        self.predict = predict
        self.max_branching = int(max(1, max_branching))
        self._grids: dict[str, np.ndarray] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self.branch_counts: list[int] = []
        self.start_state = self._register(np.asarray(start_grid))

    def _register(self, grid: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(grid, dtype=np.int16)
        key = _grid_key(arr)
        if key not in self._states:
            state = grid_state_features(arr)
            state["state_id"] = key
            state["grid_key"] = key
            self._grids[key] = arr.copy()
            self._states[key] = state
        return self._states[key]

    def grid_for(self, state: dict[str, Any]) -> np.ndarray:
        return self._grids[str(state["grid_key"])]

    def next_states(self, state: dict[str, Any]) -> list[tuple[tuple[int, int, int], dict[str, Any]]]:
        grid = self.grid_for(state)
        children: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
        for action in component_landmark_click_actions(grid, max_actions=self.max_branching):
            try:
                predicted = np.asarray(self.predict(grid.copy(), action), dtype=np.int16)
            except Exception:
                continue
            if predicted.shape != grid.shape or np.array_equal(predicted, grid):
                continue
            children.append((action, self._register(predicted)))
        self.branch_counts.append(len(children))
        return children

    def branching_factor(self) -> float:
        if not self.branch_counts:
            return 0.0
        return float(sum(self.branch_counts) / len(self.branch_counts))


def hierarchical_best_first_search(
    model: VC33VerifiedWorldModel,
    subgoals: list[Subgoal],
    *,
    is_goal: GoalPredicate,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
) -> GridSearchOutcome:
    """Plan to each subgoal in order with bounded best-first search."""

    hard_bound = min(50000, max(0, int(max_expansions)))
    current = model.start_state
    actions: list[Any] = []
    nodes_expanded = 0
    reached = 0
    for subgoal in subgoals:
        if subgoal.predicate(current):
            reached += 1
            continue
        remaining = max(0, hard_bound - nodes_expanded)
        result = best_first_search(
            current,
            next_states=model.next_states,
            is_goal=subgoal.predicate,
            heuristic=vc33_goal_distance_heuristic,
            max_expansions=remaining,
        )
        nodes_expanded += result.nodes_expanded
        if not result.solved:
            search = SearchResult(
                solved=False,
                actions=actions,
                nodes_expanded=nodes_expanded,
                final_state=result.final_state,
                bottleneck=result.bottleneck or "subgoal_unreachable",
                max_expansions=hard_bound,
            )
            return GridSearchOutcome(
                search=search,
                nodes_expanded=nodes_expanded,
                branching_factor=model.branching_factor(),
                subgoals_attempted=len(subgoals),
                subgoals_reached=reached,
            )
        actions.extend(result.actions)
        current = result.final_state
        reached += 1

    solved = bool(is_goal(current))
    search = SearchResult(
        solved=solved,
        actions=actions,
        nodes_expanded=nodes_expanded,
        final_state=current,
        bottleneck="" if solved else "final_goal_unreached_after_subgoals",
        max_expansions=hard_bound,
    )
    return GridSearchOutcome(
        search=search,
        nodes_expanded=nodes_expanded,
        branching_factor=model.branching_factor(),
        subgoals_attempted=len(subgoals),
        subgoals_reached=reached,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_predict(path: Path) -> PredictFn:
    namespace: dict[str, Any] = {
        "np": np,
        "__builtins__": {
            "abs": abs,
            "bool": bool,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "sum": sum,
            "zip": zip,
        },
    }
    exec(path.read_text(encoding="utf-8"), namespace)
    predict = namespace.get("predict")
    if not callable(predict):
        raise ValueError("world-model program did not define predict")
    return predict


def load_exp4035_preconditions(repo_root: Path) -> Exp4035Preconditions:
    """Load Exp 4034's predicate and the vc33 verified world-model program."""

    results = repo_root / "results"
    errors: list[str] = []
    predicate_code = ""
    precision = 0.0
    is_goal: GoalPredicate | None = None
    predict: PredictFn | None = None

    try:
        goal_payload = _load_json_object(results / "experiment_4034_vc33_goal_predicate_induction.json")
        precision = float(goal_payload.get("goal_predicate_heldout_precision", 0.0) or 0.0)
        predicate_code = str(goal_payload.get("goal_predicate_code") or "")
        if precision < 0.5:
            errors.append(f"exp4034 goal predicate precision below gate: {precision:.3f}")
        if not predicate_code:
            errors.append("exp4034 goal_predicate_code is empty")
        else:
            is_goal = compile_goal_predicate(predicate_code)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"could not load exp4034 goal predicate: {type(exc).__name__}")

    try:
        predict = _load_predict(results / "arc3_vc33_world_model_program.py")
    except (OSError, ValueError, SyntaxError, NameError) as exc:
        errors.append(f"could not load vc33 world model: {type(exc).__name__}")

    return Exp4035Preconditions(
        ok=not errors,
        predict=predict if not errors else None,
        is_goal=is_goal if not errors else None,
        goal_predicate_precision=precision,
        goal_predicate_code=predicate_code,
        errors=errors,
    )


def _actions_as_json(actions: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for action in actions:
        if isinstance(action, tuple):
            out.append(list(action))
        else:
            out.append(action)
    return out


def build_exp4035_artifact(
    outcome: GridSearchOutcome,
    *,
    real_env_confirmed: bool,
    levels_completed_after: int,
    duration_s: float,
    goal_predicate_precision: float,
    action_plan: list[Any],
) -> dict[str, Any]:
    """Normalize hierarchical search evidence into the required artifact."""

    confirmed = bool(outcome.search.solved and real_env_confirmed)
    if confirmed:
        verdict = f"complete: search_layer_solved_vc33_L{TARGET_LEVEL}_real_env_confirmed"
        bottleneck = ""
        new_levels = max(1, int(levels_completed_after) - PRIOR_WALL_LEVEL)
    else:
        bottleneck = outcome.search.bottleneck or "real_env_confirmation_failed"
        if outcome.search.solved and not real_env_confirmed:
            bottleneck = "real_env_confirmation_failed"
        verdict = f"complete: search_layer_no_solve_vc33_{bottleneck}"
        new_levels = 0

    return {
        "experiment": "experiment_4035_hierarchical_search_over_vc33_wm",
        "schema": "carnot.experiment_4035_hierarchical_search_over_vc33_wm.v1",
        "game": TARGET_GAME,
        "target_level": TARGET_LEVEL,
        "prior_wall_level": PRIOR_WALL_LEVEL,
        "honest_verdict": verdict,
        "new_levels_solved_this_task": int(new_levels),
        "search_layer_generalizes": bool(confirmed),
        "heuristic_was_non_bespoke": True,
        "nodes_expanded": int(outcome.nodes_expanded),
        "branching_factor": round(float(outcome.branching_factor), 6),
        "subgoal_decomposition_used": bool(outcome.subgoals_attempted > 0),
        "real_env_confirmed": bool(real_env_confirmed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "max_expansions": int(outcome.search.max_expansions),
        "max_branching": DEFAULT_MAX_BRANCHING,
        "subgoals_attempted": int(outcome.subgoals_attempted),
        "subgoals_reached": int(outcome.subgoals_reached),
        "search_found_plan": bool(outcome.search.solved),
        "action_plan": _actions_as_json(action_plan),
        "action_count": len(action_plan),
        "levels_completed_after": int(levels_completed_after),
        "goal_predicate_heldout_precision": float(goal_predicate_precision),
        "heuristic_used": "coded_unmet_goal_components_plus_manhattan_alignment_plus_progress_bar_delta",
        "bottleneck": bottleneck,
        "duration_s": round(float(duration_s), 3),
        "generality_verdict": (
            "general_heuristic_solved_second_game"
            if confirmed
            else "general_heuristic_did_not_break_vc33_wall"
        ),
        "diagnosis": (
            "The same bounded search architecture reached a real-env-confirmed vc33 solve without "
            "a vc33 coordinate macro."
            if confirmed
            else "The non-bespoke hierarchical heuristic did not produce a real-env-confirmed vc33 "
            "solve under the verified world model; this does not support generalization beyond r11l."
        ),
    }


def blocked_artifact(duration_s: float, errors: list[str] | None = None) -> dict[str, Any]:
    """Return the required blocked Exp 4035 artifact without running search."""

    return {
        "experiment": "experiment_4035_hierarchical_search_over_vc33_wm",
        "schema": "carnot.experiment_4035_hierarchical_search_over_vc33_wm.v1",
        "honest_verdict": "blocked_vc33_goal_predicate_or_wm_missing",
        "new_levels_solved_this_task": 0,
        "search_layer_generalizes": False,
        "heuristic_was_non_bespoke": True,
        "nodes_expanded": 0,
        "branching_factor": 0.0,
        "subgoal_decomposition_used": False,
        "real_env_confirmed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "max_expansions": 0,
        "precondition_errors": list(errors or []),
        "duration_s": round(float(duration_s), 3),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors for Exp 4035 artifacts."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be a terminal-prefix string")
    for field in ("new_levels_solved_this_task", "nodes_expanded"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    for field in (
        "search_layer_generalizes",
        "heuristic_was_non_bespoke",
        "subgoal_decomposition_used",
        "real_env_confirmed",
    ):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")
    if "branching_factor" in artifact and type(artifact["branching_factor"]) is not float:
        errors.append("branching_factor must be a bare float")
    if "inference_substrate" in artifact and type(artifact["inference_substrate"]) is not str:
        errors.append("inference_substrate must be a bare string")
    if (
        isinstance(artifact.get("nodes_expanded"), int)
        and isinstance(artifact.get("max_expansions"), int)
        and artifact["nodes_expanded"] > artifact["max_expansions"]
    ):
        errors.append("nodes_expanded must not exceed max_expansions")
    return errors


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Write stable JSON for downstream conductor reconciliation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _initial_vc33_grid(repo_root: Path) -> np.ndarray:  # pragma: no cover - exercised by experiment command
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    from carnot.agentic.arc_agi3_world_model import grid_of

    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    env = arc.make("vc33-5430563c")
    return grid_of(env.reset())


def _execute_plan_in_real_env(
    repo_root: Path,
    actions: list[Any],
) -> tuple[bool, int]:  # pragma: no cover - exercised by experiment command
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction

    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    env = arc.make("vc33-5430563c")
    frame = env.reset()
    best_level = int(getattr(frame, "levels_completed", 0) or 0)
    for action in actions:
        if not action or int(action[0]) != 6:
            continue
        frame = env.step(GameAction.ACTION6, data={"x": int(action[1]), "y": int(action[2])})
        best_level = max(best_level, int(getattr(frame, "levels_completed", 0) or 0))
    return best_level > PRIOR_WALL_LEVEL, best_level


def run(
    *,
    repo_root: Path | None = None,
    write: bool = True,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_branching: int = DEFAULT_MAX_BRANCHING,
) -> dict[str, Any]:
    """Run Exp 4035 and optionally write its terminal artifact."""

    started = time.time()
    root = repo_root or Path(__file__).resolve().parents[3]
    preconditions = load_exp4035_preconditions(root)
    if not preconditions.ok or preconditions.predict is None or preconditions.is_goal is None:
        artifact = blocked_artifact(time.time() - started, preconditions.errors)
        if write:
            write_artifact(artifact, root / "results" / RESULT_NAME)
        return artifact

    start_grid = _initial_vc33_grid(root)
    model = VC33VerifiedWorldModel(start_grid, preconditions.predict, max_branching=max_branching)
    subgoals = decompose_goal_predicate(model.start_state)
    outcome = hierarchical_best_first_search(
        model,
        subgoals,
        is_goal=preconditions.is_goal,
        max_expansions=max_expansions,
    )
    real_env_confirmed = False
    levels_completed_after = PRIOR_WALL_LEVEL
    if outcome.search.solved:
        real_env_confirmed, levels_completed_after = _execute_plan_in_real_env(
            root,
            list(outcome.search.actions),
        )
    artifact = build_exp4035_artifact(
        outcome,
        real_env_confirmed=real_env_confirmed,
        levels_completed_after=levels_completed_after,
        duration_s=time.time() - started,
        goal_predicate_precision=preconditions.goal_predicate_precision,
        action_plan=list(outcome.search.actions),
    )
    artifact["max_branching"] = int(max_branching)
    schema_errors = artifact_schema_errors(artifact)
    if schema_errors:  # pragma: no cover - defensive contract guard
        raise ValueError("; ".join(schema_errors))
    if write:
        write_artifact(artifact, root / "results" / RESULT_NAME)
    return artifact
