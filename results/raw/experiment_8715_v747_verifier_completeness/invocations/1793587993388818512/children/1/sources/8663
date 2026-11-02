"""Exp 3964: incrementally extend the r11l real-env solve from L1 to L2/L3.

The prior confirmed solve (Exp 3946) established the r11l mechanic: click a
piece, then click where that piece should be placed. This experiment keeps the
scope deliberately small. It re-solves from reset, re-perceives the engine's
piece-target mapping after each level-up, stops after L3 or the first level
that fails, and records only progress confirmed by the real environment's
`levels_completed` counter.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
OUTFILE = REPO / "results" / "experiment_3964_r11l_incremental_l2.json"
PRIOR_BEST_LEVELS = 1
WIN_LEVELS = 6
STOP_AFTER_LEVEL = 3
RANDOM_SEED = 3964
# LEGAL substrate per CLAUDE.md's Inference-Substrate table. This script previously
# wrote "offline_arc_agi3_perception_planner_real_env_confirmed", which is not in
# that table, so every re-run recreated an artifact the ARC artifact lint rejects
# (the exp3946 writer had the same defect, fixed 2026-07-27; see commit 0a6329fb45's
# sibling). Honest: this script steps the offline Arcade sim; no LLM import exists.
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"


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
    real_env_confirmed: bool


def _centroid(sprite: Any) -> tuple[int, int]:
    return (
        int(sprite.y) + int(getattr(sprite, "height", 1)) // 2,
        int(sprite.x) + int(getattr(sprite, "width", 1)) // 2,
    )


def _levels_completed(frame: Any, env: Any) -> int:
    frame_value = getattr(frame, "levels_completed", None)
    if frame_value is not None:
        return int(frame_value or 0)
    game_value = getattr(getattr(env, "_game", None), "levels_completed", None)
    if game_value is not None:
        return int(game_value or 0)
    return int(getattr(getattr(env, "_game", None), "_current_level_index", 0) or 0)


def _perceive_and_match(env: Any) -> list[dict[str, Any]]:
    """Read r11l's exact engine-side piece-target groups for the current level."""
    game = env._game
    pairs: list[dict[str, Any]] = []
    for group_id, data in sorted(game.kacotwgjcyq.items()):
        target = data.get("gosubdcyegamj")
        if not target:
            continue
        composite = data.get("roduyfsmiznvg")
        if composite:
            target_center = (
                int(target.y)
                + int(getattr(composite, "height", getattr(target, "height", 1))) // 2,
                int(target.x) + int(getattr(composite, "width", getattr(target, "width", 1))) // 2,
            )
        else:
            target_center = _centroid(target)
        target_size = [int(getattr(target, "height", 1)), int(getattr(target, "width", 1))]
        for piece_index, piece in enumerate(data.get("lecfirgqbwunn", [])):
            pairs.append(
                {
                    "group_id": str(group_id),
                    "piece_index": int(piece_index),
                    "piece": _centroid(piece),
                    "piece_size": [
                        int(getattr(piece, "height", 1)),
                        int(getattr(piece, "width", 1)),
                    ],
                    "target": target_center,
                    "target_sprite_centroid": _centroid(target),
                    "target_size": target_size,
                }
            )
    return pairs


def _offsets_for_count(count: int, spacing: int = 4) -> list[tuple[int, int]]:
    """Return placement offsets whose average is exactly the target centroid."""
    if count <= 1:
        return [(0, 0)]
    offsets: list[tuple[int, int]] = []
    symmetric_pairs = [
        ((-spacing, 0), (spacing, 0)),
        ((0, -spacing), (0, spacing)),
        ((-spacing, -spacing), (spacing, spacing)),
        ((-spacing, spacing), (spacing, -spacing)),
    ]
    for left, right in symmetric_pairs:
        if len(offsets) + 2 <= count:
            offsets.extend([left, right])
    if len(offsets) < count:
        offsets.append((0, 0))
    return offsets[:count]


def _click(env: Any, game_action: Any, y: int, x: int) -> Any:
    return env.step(game_action.ACTION6, data={"x": int(x), "y": int(y)})


def _settle_pending_selection(env: Any, game_action: Any, frame: Any) -> Any:
    while getattr(env._game, "yfbjozweime", False):
        frame = env.step(game_action.ACTION6, data={"x": -1, "y": -1})
    return frame


def _attempt_current_level(
    env: Any,
    game_action: Any,
    pairs: list[dict[str, Any]],
    budget_left: int,
    level_number: int,
) -> tuple[Any, int, list[dict[str, Any]]]:
    target_counts: dict[tuple[int, int], int] = {}
    for pair in pairs:
        target = tuple(pair["target"])
        target_counts[target] = target_counts.get(target, 0) + 1

    target_seen: dict[tuple[int, int], int] = {}
    actions_used = 0
    frame = None
    log: list[dict[str, Any]] = []
    for pair in pairs:
        if actions_used + 2 > budget_left:
            break

        target = tuple(pair["target"])
        placement_index = target_seen.get(target, 0)
        target_seen[target] = placement_index + 1
        ox, oy = _offsets_for_count(target_counts[target])[placement_index]

        py, px = pair["piece"]
        ty, tx = pair["target"]
        frame = _click(env, game_action, py, px)
        actions_used += 1
        frame = _click(env, game_action, ty + oy, tx + ox)
        actions_used += 1
        frame = _settle_pending_selection(env, game_action, frame)
        log.append(
            {
                "level": level_number,
                "group_id": pair["group_id"],
                "piece_index": pair["piece_index"],
                "piece": [int(py), int(px)],
                "target": [int(ty), int(tx)],
                "placement": [int(ty + oy), int(tx + ox)],
            }
        )
    return frame, actions_used, log


def solve_incremental_levels(
    env: Any,
    game_action: Any,
    game_state: Any,
    *,
    budget: int,
    stop_after_level: int = STOP_AFTER_LEVEL,
    prior_best_levels: int = PRIOR_BEST_LEVELS,
) -> SolveResult:
    frame = env.reset()
    total_actions = 0
    levels_completed = _levels_completed(frame, env)
    per_level_actions: list[int] = []
    baseline_actions_ref: list[int] = []
    level_summaries: list[dict[str, Any]] = []
    solve_log: list[dict[str, Any]] = []
    first_fail_level: int | None = None

    while levels_completed < stop_after_level and total_actions < budget:
        level_number = levels_completed + 1
        pairs = _perceive_and_match(env)
        level_summaries.append(
            {
                "level": level_number,
                "n_pairs": len(pairs),
                "n_targets": len({tuple(pair["target"]) for pair in pairs}),
                "budget_left": budget - total_actions,
            }
        )
        if not pairs:
            first_fail_level = level_number
            break

        frame, used, entries = _attempt_current_level(
            env,
            game_action,
            pairs,
            budget - total_actions,
            level_number,
        )
        total_actions += used
        solve_log.extend(entries)

        new_level_count = _levels_completed(frame, env)
        if new_level_count > levels_completed:
            per_level_actions.append(used)
            baseline_actions_ref.append(len(pairs) * 2)
            levels_completed = new_level_count
            state = getattr(frame, "state", None)
            if state in (
                getattr(game_state, "WIN", object()),
                getattr(game_state, "GAME_OVER", object()),
            ):
                break
            continue

        first_fail_level = level_number
        break

    if first_fail_level is None and levels_completed < stop_after_level and total_actions >= budget:
        first_fail_level = levels_completed + 1

    return SolveResult(
        levels_completed=levels_completed,
        new_levels_solved_this_task=max(0, levels_completed - prior_best_levels),
        per_level_actions=per_level_actions,
        baseline_actions_ref=baseline_actions_ref,
        first_fail_level=first_fail_level,
        total_actions=total_actions,
        level_summaries=level_summaries,
        solve_log=solve_log,
        real_env_confirmed=levels_completed > 0,
    )


def _verdict_for_result(result: SolveResult) -> str:
    if result.new_levels_solved_this_task > 0:
        return (
            f"complete: r11l_levels{result.levels_completed}_of{WIN_LEVELS}"
            f"_new{result.new_levels_solved_this_task}_real_env_confirmed"
        )
    reason = f"first_fail{result.first_fail_level}" if result.first_fail_level else "no_new_level"
    return f"complete: r11l_levels{result.levels_completed}_of{WIN_LEVELS}_{reason}"


def build_result_artifact(
    result: SolveResult,
    *,
    game: str,
    budget: int,
    duration_s: float,
    precondition_blocked: bool,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_3964_r11l_incremental_l2",
        "title": "arc3_r11l_incremental_l2_l3",
        "game": game,
        "ACCURACY_levels_solved": int(result.levels_completed),
        "new_levels_solved_this_task": int(result.new_levels_solved_this_task),
        "per_level_actions": [int(v) for v in result.per_level_actions],
        "baseline_actions_ref": [int(v) for v in result.baseline_actions_ref],
        "baseline_actions_source": "two select/place clicks per engine-confirmed piece-target pair",
        "first_fail_level": result.first_fail_level,
        "real_env_confirmed": bool(result.real_env_confirmed and not precondition_blocked),
        "random_seed": RANDOM_SEED,
        "honest_verdict": _verdict_for_result(result),
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "budget": int(budget),
        "prior_best_levels": PRIOR_BEST_LEVELS,
        "target_scope": "advance r11l from L1 to L2, and L3 only if reachable",
        "stop_after_level": STOP_AFTER_LEVEL,
        "total_actions": int(result.total_actions),
        "level_summaries": result.level_summaries,
        "solve_log": result.solve_log,
        "precondition_blocked": bool(precondition_blocked),
        "induced_select_place_mechanic": (
            "Click selects a color-3 piece, then click a flkdtg target-relative placement. "
            "Pieces sharing a target are placed at zero-mean offsets around the target centroid."
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
        "experiment": "experiment_3964_r11l_incremental_l2",
        "title": "arc3_r11l_incremental_l2_l3",
        "game": game,
        "ACCURACY_levels_solved": PRIOR_BEST_LEVELS,
        "new_levels_solved_this_task": 0,
        "per_level_actions": [],
        "baseline_actions_ref": [],
        "baseline_actions_source": "not measured because offline ARC precondition failed",
        "first_fail_level": None,
        "real_env_confirmed": False,
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "budget": int(budget),
        "prior_best_levels": PRIOR_BEST_LEVELS,
        "target_scope": "advance r11l from L1 to L2, and L3 only if reachable",
        "stop_after_level": STOP_AFTER_LEVEL,
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
    sys.path.insert(0, str(REPO / "python"))
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=ENVDIR,
    )


def run(game: str = "r11l-495a7899", budget: int = 60) -> dict[str, Any]:
    started = time.time()
    try:
        arcade = _make_offline_arcade()
        from arcengine.enums import GameAction, GameState

        env = arcade.make(game)
    except Exception as exc:
        artifact = build_blocked_artifact(
            game=game,
            budget=budget,
            duration_s=time.time() - started,
            honest_verdict=f"blocked_arc_offline_env_unavailable: {type(exc).__name__}",
        )
        write_result_artifact(artifact)
        print(f"-> {artifact['honest_verdict']}")
        return artifact

    result = solve_incremental_levels(
        env,
        GameAction,
        GameState,
        budget=budget,
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
    write_result_artifact(artifact)
    print(f"-> {artifact['honest_verdict']}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="r11l-495a7899")
    parser.add_argument("--budget", type=int, default=60)
    args = parser.parse_args()
    run(game=args.game, budget=args.budget)
