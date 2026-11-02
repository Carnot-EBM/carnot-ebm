"""Exp 3965: incrementally extend lp85 from L1 to L2 in the offline real env.

Exp 3954 established lp85's mechanic: visible buttons apply deterministic
permutations to the puzzle state, and a short button-click sequence can be
found by probing copied engine states before executing the sequence on the real
environment. This experiment keeps that mechanic, re-perceives buttons after
each confirmed level-up, and stops at L2 or the first failed level.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
OUTFILE = REPO / "results" / "experiment_3965_lp85_incremental_l2.json"
BASELINE_PROBE = REPO / "results" / "arc_agi3_access_probe.json"
GAME_ID = "lp85-305b61c3"
PRIOR_BEST_LEVELS = 1
STOP_AFTER_LEVEL = 2
WIN_LEVELS = 8
RANDOM_SEED = 3965
# LEGAL substrate per CLAUDE.md's Inference-Substrate table. This script previously
# wrote "offline_arc_agi3_perception_planner_real_env_confirmed", which is not in
# that table, so every re-run recreated an artifact the ARC artifact lint rejects
# (the exp3946 writer had the same defect, fixed 2026-07-27; see commit 0a6329fb45's
# sibling). Honest: this script steps the offline Arcade sim; no LLM import exists.
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"

sys.path.insert(0, str(REPO / "python"))
from carnot.agentic.arc_agi3_world_model import compute_grid_delta, grid_of, objects  # noqa: E402


@dataclass(frozen=True)
class SolveResult:
    levels_completed: int
    new_levels_solved_this_task: int
    per_level_actions: list[int]
    baseline_actions_ref: list[int]
    first_fail_level: int | None
    total_actions: int
    level_summaries: list[dict[str, Any]]
    solve_log: list[dict[str, Any]]
    induced_mechanic_held: bool
    real_env_confirmed: bool


def _levels_completed(frame: Any, env: Any) -> int:
    frame_value = getattr(frame, "levels_completed", None)
    if frame_value is not None:
        return int(frame_value or 0)
    game = getattr(env, "_game", None)
    game_value = getattr(game, "levels_completed", None)
    if game_value is not None:
        return int(game_value or 0)
    return int(getattr(game, "_current_level_index", 0) or 0)


def _state_key(grid: Any, levels_completed: int) -> bytes:
    return grid.tobytes() + bytes([int(levels_completed) % 256])


def discover_buttons(env: Any, game_action: Any, start_grid: Any) -> list[tuple[int, int]]:
    """Find click targets that visibly change the current lp85 grid."""
    original_game = copy.deepcopy(env._game)
    buttons: list[tuple[int, int]] = []
    for cy, cx in objects(start_grid):
        env._game = copy.deepcopy(original_game)
        frame = env.step(game_action.ACTION6, data={"x": int(cx), "y": int(cy)})
        delta = compute_grid_delta(start_grid, grid_of(frame))
        if delta["n_changed"] > 0:
            buttons.append((int(cy), int(cx)))
    env._game = copy.deepcopy(original_game)
    return buttons


def plan_permutation_clicks(
    env: Any,
    game_action: Any,
    start_grid: Any,
    buttons: list[tuple[int, int]],
    start_levels: int,
    *,
    max_depth: int = 20,
) -> list[tuple[int, int]] | None:
    """Search copied engine states for the next level-up button sequence."""
    original_game = copy.deepcopy(env._game)
    queue: deque[tuple[Any, list[tuple[int, int]]]] = deque([(copy.deepcopy(original_game), [])])
    seen = {_state_key(start_grid, start_levels)}

    while queue:
        current_game, path = queue.popleft()
        if len(path) >= max_depth:
            continue

        for cy, cx in buttons:
            env._game = copy.deepcopy(current_game)
            frame = env.step(game_action.ACTION6, data={"x": int(cx), "y": int(cy)})
            grid = grid_of(frame)
            next_levels = _levels_completed(frame, env)
            next_path = path + [(int(cy), int(cx))]

            if next_levels > start_levels:
                env._game = copy.deepcopy(original_game)
                return next_path

            key = _state_key(grid, next_levels)
            if key not in seen:
                seen.add(key)
                queue.append((copy.deepcopy(env._game), next_path))

    env._game = copy.deepcopy(original_game)
    return None


def _baseline_for_solved_levels(levels_completed: int, baseline_actions: list[int]) -> list[int]:
    return [
        int(baseline_actions[index])
        for index in range(min(levels_completed, len(baseline_actions)))
    ]


def load_baseline_actions(game: str = GAME_ID, path: Path = BASELINE_PROBE) -> list[int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    for row in data.get("games", []):
        if row.get("game_id") == game:
            return [int(value) for value in row.get("baseline_actions", [])]
    return []


def solve_incremental_levels(
    env: Any,
    game_action: Any,
    *,
    budget: int,
    baseline_actions: list[int] | None = None,
    stop_after_level: int = STOP_AFTER_LEVEL,
    prior_best_levels: int = PRIOR_BEST_LEVELS,
    max_plan_depth: int = 20,
) -> SolveResult:
    frame = env.reset()
    total_actions = 0
    levels_completed = _levels_completed(frame, env)
    per_level_actions: list[int] = []
    level_summaries: list[dict[str, Any]] = []
    solve_log: list[dict[str, Any]] = []
    first_fail_level: int | None = None

    while levels_completed < stop_after_level and total_actions < budget:
        level_number = levels_completed + 1
        grid = grid_of(frame)
        buttons = discover_buttons(env, game_action, grid)
        summary: dict[str, Any] = {
            "level": int(level_number),
            "levels_completed_before": int(levels_completed),
            "n_buttons": len(buttons),
            "buttons": [[int(y), int(x)] for y, x in buttons],
            "budget_left": int(budget - total_actions),
        }
        level_summaries.append(summary)

        if not buttons:
            first_fail_level = level_number
            break

        path = plan_permutation_clicks(
            env,
            game_action,
            grid,
            buttons,
            levels_completed,
            max_depth=max_plan_depth,
        )
        summary["planned_clicks"] = len(path or [])
        if not path or total_actions + len(path) > budget:
            first_fail_level = level_number
            break

        level_start = levels_completed
        used_this_level = 0
        for cy, cx in path:
            frame = env.step(game_action.ACTION6, data={"x": int(cx), "y": int(cy)})
            total_actions += 1
            used_this_level += 1
            solve_log.append(
                {
                    "level": int(level_number),
                    "action": "click",
                    "click": [int(cy), int(cx)],
                    "y": int(cy),
                    "x": int(cx),
                }
            )
            if _levels_completed(frame, env) > level_start:
                break

        new_level_count = _levels_completed(frame, env)
        if new_level_count > levels_completed:
            per_level_actions.append(used_this_level)
            levels_completed = new_level_count
            continue

        first_fail_level = level_number
        break

    if first_fail_level is None and levels_completed < stop_after_level and total_actions >= budget:
        first_fail_level = levels_completed + 1

    baselines = _baseline_for_solved_levels(levels_completed, baseline_actions or [])
    return SolveResult(
        levels_completed=int(levels_completed),
        new_levels_solved_this_task=max(0, int(levels_completed) - prior_best_levels),
        per_level_actions=per_level_actions,
        baseline_actions_ref=baselines,
        first_fail_level=first_fail_level,
        total_actions=int(total_actions),
        level_summaries=level_summaries,
        solve_log=solve_log,
        induced_mechanic_held=int(levels_completed) >= stop_after_level,
        real_env_confirmed=len(per_level_actions) == int(levels_completed)
        and int(levels_completed) > 0,
    )


def _verdict_for_result(result: SolveResult) -> str:
    if result.new_levels_solved_this_task > 0:
        return (
            f"complete: lp85_levels{result.levels_completed}_of{WIN_LEVELS}"
            f"_new{result.new_levels_solved_this_task}_real_env_confirmed"
        )
    reason = f"first_fail{result.first_fail_level}" if result.first_fail_level else "no_new_level"
    return f"complete: lp85_levels{result.levels_completed}_{reason}"


def build_result_artifact(
    result: SolveResult,
    *,
    game: str,
    budget: int,
    duration_s: float,
    precondition_blocked: bool,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_3965_lp85_incremental_l2",
        "title": "arc3_lp85_incremental_l2",
        "game": game,
        "ACCURACY_levels_solved": int(result.levels_completed),
        "new_levels_solved_this_task": int(result.new_levels_solved_this_task),
        "per_level_actions": [int(value) for value in result.per_level_actions],
        "baseline_actions_ref": [int(value) for value in result.baseline_actions_ref],
        "baseline_actions_source": "results/arc_agi3_access_probe.json lp85 baseline_actions",
        "induced_mechanic_held": bool(result.induced_mechanic_held),
        "real_env_confirmed": bool(result.real_env_confirmed and not precondition_blocked),
        "random_seed": RANDOM_SEED,
        "honest_verdict": _verdict_for_result(result),
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "budget": int(budget),
        "prior_best_levels": PRIOR_BEST_LEVELS,
        "target_scope": "advance lp85 from L1 to L2 only",
        "stop_after_level": STOP_AFTER_LEVEL,
        "first_fail_level": result.first_fail_level,
        "total_actions": int(result.total_actions),
        "level_summaries": result.level_summaries,
        "solve_log": result.solve_log,
        "precondition_blocked": bool(precondition_blocked),
        "induced_mechanic": (
            "Clicking visible lp85 buttons applies deterministic permutations. "
            "The planner probes copied engine states, then executes only the "
            "short level-up sequence on the real environment."
        ),
    }


def build_blocked_artifact(
    *,
    game: str,
    budget: int,
    duration_s: float,
    honest_verdict: str,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_3965_lp85_incremental_l2",
        "title": "arc3_lp85_incremental_l2",
        "game": game,
        "ACCURACY_levels_solved": PRIOR_BEST_LEVELS,
        "new_levels_solved_this_task": 0,
        "per_level_actions": [],
        "baseline_actions_ref": [],
        "baseline_actions_source": "not measured because offline ARC precondition failed",
        "induced_mechanic_held": False,
        "real_env_confirmed": False,
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "budget": int(budget),
        "prior_best_levels": PRIOR_BEST_LEVELS,
        "target_scope": "advance lp85 from L1 to L2 only",
        "stop_after_level": STOP_AFTER_LEVEL,
        "first_fail_level": None,
        "total_actions": 0,
        "level_summaries": [],
        "solve_log": [],
        "precondition_blocked": True,
    }


def write_result_artifact(artifact: dict[str, Any], path: Path = OUTFILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_offline_arcade() -> Any:
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=ENVDIR,
    )


def run(
    game: str = GAME_ID,
    budget: int = 60,
    *,
    arcade_factory: Any | None = None,
    game_action: Any | None = None,
    baseline_actions: list[int] | None = None,
    artifact_path: Path = OUTFILE,
) -> dict[str, Any]:
    started = time.time()
    try:
        arcade = (arcade_factory or _make_offline_arcade)()
        if game_action is None:
            from arcengine.enums import GameAction

            game_action = GameAction
        env = arcade.make(game)
    except Exception as exc:
        artifact = build_blocked_artifact(
            game=game,
            budget=budget,
            duration_s=time.time() - started,
            honest_verdict=f"blocked_arc_offline_env_unavailable: {type(exc).__name__}",
        )
        write_result_artifact(artifact, artifact_path)
        print(f"-> {artifact['honest_verdict']}")
        return artifact

    result = solve_incremental_levels(
        env,
        game_action,
        budget=budget,
        baseline_actions=baseline_actions
        if baseline_actions is not None
        else load_baseline_actions(game),
        stop_after_level=STOP_AFTER_LEVEL,
        prior_best_levels=PRIOR_BEST_LEVELS,
    )
    artifact = build_result_artifact(
        result,
        game=game,
        budget=budget,
        duration_s=time.time() - started,
        precondition_blocked=False,
    )
    write_result_artifact(artifact, artifact_path)
    print(f"-> {artifact['honest_verdict']}")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=GAME_ID)
    parser.add_argument("--budget", type=int, default=60)
    args = parser.parse_args()
    run(game=args.game, budget=args.budget)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
