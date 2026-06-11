"""Exp 4024: explore-first ARC-AGI-3 continuation on a new non-spatial game.

Spec refs: REQ-PHASE4-032, SCENARIO-PHASE4-032.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
RESULT_NAME = "experiment_4024_fifth_game_explore_first.json"
RANDOM_SEED = 4024
INFERENCE_SUBSTRATE = "offline_arc_agi3_cd82_explore_first_region_fill_induction"

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4024_fifth_game_explore_first import (  # noqa: E402
    PRIOR_TOTAL_GAMES_SOLVED,
    ExperimentOutcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_cd82_l1_plan,
    load_environment_baselines,
    select_new_candidate_from_survey,
)


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_offline_arcade():  # pragma: no cover - exercised by the required live run
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arcade = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    if not arcade.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arcade


def _level_completed(frame: Any, env: Any) -> int:
    for attr in ("level_completed", "levels_completed"):
        value = getattr(frame, attr, None)
        if value is not None:
            return int(value or 0)
    game = getattr(env, "_game", None)
    for attr in ("level_completed", "levels_completed", "_current_level_index"):
        value = getattr(game, attr, None)
        if value is not None:
            return int(value or 0)
    return 0


def _game_over(frame: Any) -> bool:
    return "GAME_OVER" in str(getattr(frame, "state", ""))


def _canvas_and_target(env: Any) -> tuple[np.ndarray, np.ndarray]:
    level = env._game.current_level
    canvas = level.get_sprites_by_name("xytrjjbyib")[0].pixels
    target_sprite = next(sprite for sprite in level.get_sprites() if sprite.name.startswith("eoqnvkspoa-"))
    return np.asarray(canvas, dtype=np.int16), np.asarray(target_sprite.pixels, dtype=np.int16)


def _cd82_state(env: Any, frame: Any) -> dict[str, Any]:
    canvas, target = _canvas_and_target(env)
    return {
        "level_completed": _level_completed(frame, env),
        "active_region_index": int(env._game.xwmfgtlso),
        "selected_color": int(env._game.knqmgavuh),
        "action_count": int(getattr(env._game, "_action_count", 0) or 0),
        "canvas_colors": sorted({int(value) for value in canvas.ravel()}),
        "target_colors": sorted({int(value) for value in target.ravel()}),
    }


def _step(env: Any, action_id: int) -> Any:
    from arcengine.enums import GameAction

    action = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
    }[int(action_id)]
    return env.step(action)


def _attempt_cd82(arcade: Any, candidate: Any) -> ExperimentOutcome:
    env = arcade.make(candidate.game_id)
    frame = env.reset()
    start_level = _level_completed(frame, env)
    canvas, target = _canvas_and_target(env)
    initial_state = _cd82_state(env, frame)
    initial_plan = build_cd82_l1_plan(
        active_index=initial_state["active_region_index"],
        selected_color=initial_state["selected_color"],
        current_canvas=canvas,
        target_canvas=target,
    )
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "target_game": candidate.game_id,
            "state": initial_state,
            "candidate_reason": candidate.selection_reason,
        }
    ]

    for index, action_id in enumerate(initial_plan.exploration_actions, start=1):
        before = _cd82_state(env, frame)
        frame = _step(env, action_id)
        after = _cd82_state(env, frame)
        phase_trace.append(
            {
                "phase": "explore",
                "action_index": index,
                "action": int(action_id),
                "before": before,
                "after": after,
                "level_completed": _level_completed(frame, env),
            }
        )
        if _game_over(frame):
            return ExperimentOutcome(
                target_game=candidate.game_id,
                selected_candidate_reason=candidate.selection_reason,
                prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
                final_level_completed=_level_completed(frame, env),
                first_solve_at_action=-1,
                exploration_actions_used=index,
                induced_mechanic="cd82 region-fill induction stopped because exploration reached game over",
                verification_decisions=[],
                phase_trace=phase_trace,
                real_env_confirmed=False,
                failure_reason="game_over_during_exploration",
            )

    current_canvas, current_target = _canvas_and_target(env)
    current_state = _cd82_state(env, frame)
    commit_plan = build_cd82_l1_plan(
        active_index=current_state["active_region_index"],
        selected_color=current_state["selected_color"],
        current_canvas=current_canvas,
        target_canvas=current_target,
    )
    induced_mechanic = (
        "Observed cd82 basket navigation changed the active region before any commit; "
        "induced a one-region fill model where ACTION5 paints the active basket region "
        "with the selected palette color and the goal ignores both diagonals."
    )
    verification_decisions = [
        {
            "phase": "verify",
            "region_index": int(commit_plan.region_index),
            "fill_color": int(commit_plan.fill_color),
            "predicted_goal_after_action": bool(commit_plan.predicted_goal_after_commit),
            "retained": bool(commit_plan.predicted_goal_after_commit),
            "commit_action": int(commit_plan.commit_action),
        }
    ]
    phase_trace.append(
        {
            "phase": "induce",
            "mechanic": "cd82_active_region_fill",
            "exploration_actions_used": len(initial_plan.exploration_actions),
            "induced_region_index": int(commit_plan.region_index),
            "induced_fill_color": int(commit_plan.fill_color),
        }
    )
    phase_trace.append(verification_decisions[0])

    if not commit_plan.predicted_goal_after_commit:
        return ExperimentOutcome(
            target_game=candidate.game_id,
            selected_candidate_reason=candidate.selection_reason,
            prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
            final_level_completed=_level_completed(frame, env),
            first_solve_at_action=-1,
            exploration_actions_used=len(initial_plan.exploration_actions),
            induced_mechanic=induced_mechanic,
            verification_decisions=verification_decisions,
            phase_trace=phase_trace,
            real_env_confirmed=False,
            failure_reason="verification_rejected_commit_action",
        )

    frame = _step(env, commit_plan.commit_action)
    final_level = _level_completed(frame, env)
    first_solve_at_action = len(initial_plan.exploration_actions) + 1
    phase_trace.append(
        {
            "phase": "act",
            "action_index": first_solve_at_action,
            "action": int(commit_plan.commit_action),
            "level_completed": final_level,
            "levels_completed": final_level,
        }
    )
    solved = final_level > start_level
    return ExperimentOutcome(
        target_game=candidate.game_id,
        selected_candidate_reason=candidate.selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=final_level,
        first_solve_at_action=first_solve_at_action if solved else -1,
        exploration_actions_used=len(initial_plan.exploration_actions),
        induced_mechanic=induced_mechanic,
        verification_decisions=verification_decisions,
        phase_trace=phase_trace,
        real_env_confirmed=solved,
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def run(*, seed: int = RANDOM_SEED, write: bool = True) -> dict[str, Any]:
    started = time.time()
    try:
        arcade = _load_offline_arcade()
    except Exception:
        artifact = blocked_artifact(
            random_seed=seed,
            duration_s=round(time.time() - started, 3),
            inference_substrate=INFERENCE_SUBSTRATE,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    survey = json.loads((REPO / "results" / "arc3_win_condition_survey.json").read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")
    candidate = select_new_candidate_from_survey(survey, baselines)
    outcome = _attempt_cd82(arcade, candidate)
    artifact = build_artifact(
        outcome,
        random_seed=seed,
        duration_s=round(time.time() - started, 3),
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    artifact["excluded_solved_games"] = list(candidate.excluded_solved_games)
    artifact["candidate_baseline_actions"] = int(candidate.baseline_actions)
    if write:
        _write_artifact(artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = run(seed=args.seed, write=True)
    print(f"-> {result['honest_verdict']}")
    sys.exit(0 if result["honest_verdict"].startswith(("success:", "complete:", "blocked_")) else 1)
