"""Exp 4110: twelfth ARC-AGI-3 game solve via offline explore-first.

Spec refs: REQ-PHASE4-047, SCENARIO-PHASE4-047.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4110_twelfth_game_explore_first.json"
RANDOM_SEED = 4110

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines  # noqa: E402
from carnot.agentic.arc_exp4110_twelfth_game_explore_first import (  # noqa: E402
    INFERENCE_SUBSTRATE,
    Tu93ObservedState,
    Tu93Outcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_tu93_l1_plan,
    observe_tu93_state_from_env,
    select_exp4110_candidate_from_survey,
    validate_tu93_replayed_plan,
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


def _capture_state(env: Any, frame: Any) -> Tu93ObservedState:
    return observe_tu93_state_from_env(env, level_completed=_level_completed(frame, env))


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


def _predicted_final_state(start_state: Tu93ObservedState, *, level_completed: int) -> Tu93ObservedState:
    return Tu93ObservedState(
        player_position=start_state.target_position,
        target_position=start_state.target_position,
        map_origin=start_state.map_origin,
        map_pixels=start_state.map_pixels,
        remaining_steps=start_state.remaining_steps,
        level_completed=int(level_completed),
    )


def _run_tu93_explore_first(
    offline_arcade: Any,
    candidate: Any,
    *,
    arc_env_count: int,
) -> Tu93Outcome:
    env = offline_arcade.make(candidate.game_id)
    frame = env.reset()
    observed = _capture_state(env, frame)
    plan = build_tu93_l1_plan(observed)
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
            "mechanic": "tu93_lattice_navigation_to_target",
            "goal_predicate": plan.induction_call["goal_predicate"],
            "induction_call": plan.induction_call,
            "exploration_actions_used": len(plan.exploration_actions),
        }
    )

    final_level, _, _, _ = _execute_actions(
        env,
        frame,
        plan.commit_actions,
        source="offline_gap4_replay_verification",
        action_offset=len(plan.exploration_actions),
    )
    verification = validate_tu93_replayed_plan(
        commit_start_state,
        _predicted_final_state(commit_start_state, level_completed=final_level),
        plan,
    )
    phase_trace.append(verification)

    induced_mechanic = (
        "Observed TU93 keyboard transitions on a 6-pixel lattice; induced that each accepted "
        "direction advances the player one visible path-node toward the 0015msvpvzxhqf target, "
        "with a real level counter increment required before claiming success."
    )
    if not verification["retained"]:
        return Tu93Outcome(
            target_game=candidate.game_id,
            selected_candidate_reason=candidate.selection_reason,
            prior_total_games_solved=11,
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
    return Tu93Outcome(
        target_game=candidate.game_id,
        selected_candidate_reason=candidate.selection_reason,
        prior_total_games_solved=11,
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
    candidate = select_exp4110_candidate_from_survey(survey, baselines)
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
    outcome = _run_tu93_explore_first(
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
