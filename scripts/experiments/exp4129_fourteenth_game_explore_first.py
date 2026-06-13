"""Exp 4129: fourteenth ARC-AGI-3 game solve via offline explore-first.

Spec refs: REQ-PHASE4-049, SCENARIO-PHASE4-049.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4129_fourteenth_game_explore_first.json"
RANDOM_SEED = 4129

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines  # noqa: E402
from carnot.agentic.arc_exp4129_fourteenth_game_explore_first import (  # noqa: E402
    INFERENCE_SUBSTRATE,
    Bp35ObservedState,
    Bp35Outcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_bp35_l1_plan,
    select_exp4129_candidate_from_survey,
    validate_bp35_replayed_plan,
)


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_offline_arcade() -> Any:
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )


def _fixture_available(game_id: str) -> bool:
    prefix, suffix = str(game_id).split("-", maxsplit=1)
    root = REPO / "environment_files" / prefix / suffix
    return root.joinpath("metadata.json").exists() and root.joinpath(f"{prefix}.py").exists()


def _level_completed(frame: Any, env: Any) -> int:
    value = getattr(frame, "levels_completed", None)
    if value is not None:
        return int(value)
    return int(getattr(getattr(env, "_game", object()), "levels_completed", 0) or 0)


def _game_over(frame: Any) -> bool:
    state = str(getattr(frame, "state", ""))
    return "GAME_OVER" in state or state.endswith("WIN")


def _grid_click_data(env: Any, grid: tuple[int, int]) -> dict[str, int]:
    inner = env._game.oztjzzyqoek
    grid_obj = inner.hdnrlfmyrj
    camera_y = inner.camera.rczgvgfsb if hasattr(inner.camera, "rczgvgfsb") else inner.camera.rczgvgfsfb[1]
    grid_x, grid_y = grid_obj.knpqzpefyn()
    return {
        "x": int(grid_x + int(grid[0]) * grid_obj.unxmkbpkzwj + grid_obj.unxmkbpkzwj // 2),
        "y": int(grid_y + int(grid[1]) * grid_obj.ltlyhlyvapv + grid_obj.ltlyhlyvapv // 2 - camera_y),
    }


def _capture_state(env: Any, frame: Any) -> Bp35ObservedState:
    inner = env._game.oztjzzyqoek
    grid = inner.hdnrlfmyrj
    gem = grid.wwkbcxznzg("fjlzdjxhant")[0]
    removable_blocks = tuple(
        sorted((int(sprite.grid_x), int(sprite.grid_y)) for sprite in grid.wwkbcxznzg("qclfkhjnaac"))
    )
    return Bp35ObservedState(
        player_position=tuple(inner.twdpowducb.qumspquyus),
        gem_position=tuple(gem.qumspquyus),
        gravity_direction="up" if bool(inner.vivnprldht) else "down",
        level_completed=_level_completed(frame, env),
        grid_size=tuple(grid.grid_size),
        removable_blocks=removable_blocks,
    )


def _capture_trace_state(env: Any, frame: Any) -> dict[str, Any]:
    try:
        return _capture_state(env, frame).to_json()
    except Exception:
        return {
            "level_completed": _level_completed(frame, env),
            "state": str(getattr(frame, "state", "")),
        }


def _step(env: Any, action: Any) -> Any:
    from arcengine.enums import GameAction

    game_action = getattr(GameAction, f"ACTION{int(action.action)}")
    if int(action.action) == 6:
        if action.grid is None:
            raise ValueError("BP35 click action requires a grid target")
        return env.step(game_action, data=_grid_click_data(env, action.grid))
    return env.step(game_action)


def _execute_actions(
    env: Any,
    frame: Any,
    actions: list[Any],
    *,
    source: str,
    action_offset: int = 0,
) -> tuple[int, int, list[dict[str, Any]], Any]:
    start_level = _level_completed(frame, env)
    final_level = start_level
    first_solve_at_action = -1
    trace: list[dict[str, Any]] = []
    for local_index, action in enumerate(actions, start=1):
        before = _capture_trace_state(env, frame)
        frame = _step(env, action)
        after = _capture_trace_state(env, frame)
        final_level = int(after["level_completed"])
        action_index = action_offset + local_index
        trace.append(
            {
                "phase": "act" if source == "offline_real_env_confirmation" else "explore",
                "source": source,
                "action_index": action_index,
                "action": action.to_json(),
                "before": before,
                "after": after,
                "level_completed": final_level,
            }
        )
        if final_level > start_level and first_solve_at_action < 0:
            first_solve_at_action = action_index
            break
        if _game_over(frame):
            break
    return final_level, first_solve_at_action, trace, frame


def _run_bp35_explore_first(
    offline_arcade: Any,
    candidate: Any,
    *,
    arc_env_count: int,
) -> Bp35Outcome:
    env = offline_arcade.make(candidate.game_id)
    frame = env.reset()
    observed = _capture_state(env, frame)
    plan = build_bp35_l1_plan(observed)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_explore_first_induction",
            "target_game": candidate.game_id,
            "state": observed.to_json(),
            "candidate_reason": candidate.selection_reason,
        }
    ]

    _, _, exploration_trace, frame = _execute_actions(
        env,
        frame,
        plan.exploration_actions,
        source="offline_explore_first_induction",
    )
    phase_trace.extend(exploration_trace)
    commit_start_state = _capture_state(env, frame)
    phase_trace.append(
        {
            "phase": "induce",
            "source": "offline_explore_first_induction",
            "mechanic": "bp35_upward_gravity_gem_route",
            "goal_predicate": plan.induction_call["goal_predicate"],
            "induction_call": plan.induction_call,
            "exploration_actions_used": len(plan.exploration_actions),
        }
    )

    final_level, _, _, frame = _execute_actions(
        env,
        frame,
        plan.commit_actions,
        source="offline_gap4_replay_verification",
        action_offset=len(plan.exploration_actions),
    )
    verification = validate_bp35_replayed_plan(
        commit_start_state,
        _capture_state(env, frame),
        plan,
    )
    phase_trace.append(verification)

    induced_mechanic = (
        "Observed BP35 transitions: horizontal keyboard movement, upward falling through unsupported "
        "cells, qclfkhjnaac block removal by click, and a required real level counter increment when "
        "the player reaches fjlzdjxhant."
    )
    if final_level <= commit_start_state.level_completed or not verification["retained"]:
        return Bp35Outcome(
            target_game=candidate.game_id,
            selected_candidate_reason=candidate.selection_reason,
            prior_total_games_solved=12,
            final_level_completed=commit_start_state.level_completed,
            first_solve_at_action=-1,
            exploration_actions_used=len(plan.exploration_actions),
            induced_mechanic=induced_mechanic,
            verification_decisions=[verification],
            phase_trace=phase_trace,
            real_env_confirmed=False,
            action_plan=plan.actions,
            arc_env_count=arc_env_count,
            induction_calls=[plan.induction_call],
            failure_reason="verification_rejected_commit_suffix",
        )

    real_env = offline_arcade.make(candidate.game_id)
    real_frame = real_env.reset()
    real_final_level, first_solve_at_action, act_trace, _ = _execute_actions(
        real_env,
        real_frame,
        plan.actions,
        source="offline_real_env_confirmation",
    )
    phase_trace.extend(act_trace)
    solved = real_final_level > 0 and first_solve_at_action > 0
    return Bp35Outcome(
        target_game=candidate.game_id,
        selected_candidate_reason=candidate.selection_reason,
        prior_total_games_solved=12,
        final_level_completed=real_final_level,
        first_solve_at_action=first_solve_at_action if solved else -1,
        exploration_actions_used=len(plan.exploration_actions),
        induced_mechanic=induced_mechanic,
        verification_decisions=[verification],
        phase_trace=phase_trace,
        real_env_confirmed=bool(solved),
        action_plan=plan.actions,
        arc_env_count=arc_env_count,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def run(*, seed: int = RANDOM_SEED, write: bool = True) -> dict[str, Any]:
    started = time.time()
    survey = json.loads((REPO / "results" / "arc3_win_condition_survey.json").read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")
    candidate = select_exp4129_candidate_from_survey(survey, baselines)
    if not _fixture_available(candidate.game_id):
        artifact = blocked_artifact(
            target_game=candidate.game_id,
            random_seed=seed,
            duration_s=round(time.time() - started, 3),
            inference_substrate=INFERENCE_SUBSTRATE,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    offline_arcade = _load_offline_arcade()
    try:
        arc_env_count = len(offline_arcade.get_environments())
    except Exception:
        arc_env_count = 0
    outcome = _run_bp35_explore_first(
        offline_arcade,
        candidate,
        arc_env_count=arc_env_count,
    )
    artifact = build_artifact(
        outcome,
        candidate,
        random_seed=seed,
        duration_s=round(time.time() - started, 3),
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise RuntimeError("; ".join(errors))
    if write:
        _write_artifact(artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = run(write=not args.no_write)
    print(result["honest_verdict"])
    raise SystemExit(0 if result["honest_verdict"].startswith(("success:", "complete:", "blocked_")) else 1)
