"""Exp 4092: tenth ARC-AGI-3 game solve via explore-first.

Spec refs: REQ-PHASE4-045, SCENARIO-PHASE4-045.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4092_tenth_game_explore_first.json"
RANDOM_SEED = 4092

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp4070_ninth_game_explore_first as exp4070  # noqa: E402
from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines  # noqa: E402
from carnot.agentic.arc_exp4092_tenth_game_explore_first import (  # noqa: E402
    INFERENCE_SUBSTRATE,
    R11LObservedState,
    R11LOutcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_r11l_l1_plan,
    observe_r11l_state_from_env,
    select_exp4092_candidate_from_survey,
    validate_r11l_replayed_plan,
)

_confirm_arc_env_reachable = exp4070._confirm_arc_env_reachable
_load_offline_arcade = exp4070._load_offline_arcade
_load_online_arcade = exp4070._load_online_arcade
_level_completed = exp4070._level_completed
_game_over = exp4070._game_over


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture_state(env: Any, frame: Any) -> R11LObservedState:
    return observe_r11l_state_from_env(env, level_completed=_level_completed(frame, env))


def _capture_trace_state(env: Any, frame: Any) -> dict[str, Any]:
    if hasattr(env, "_game"):
        return _capture_state(env, frame).to_json()
    return {
        "level_completed": _level_completed(frame, env),
        "state": str(getattr(frame, "state", "")),
    }


def _step(env: Any, action: Any) -> Any:
    from arcengine.enums import GameAction

    return env.step(GameAction.ACTION6, data={"x": int(action.x), "y": int(action.y)})


def _settle_pending_drag(env: Any, frame: Any) -> Any:
    from arcengine.enums import GameAction

    if not hasattr(env, "_game"):
        return frame
    for _ in range(32):
        if not getattr(env._game, "yfbjozweime", False):
            return frame
        frame = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    return frame


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
        frame = _settle_pending_drag(env, frame)
        after = _capture_trace_state(env, frame)
        final_level = int(after["level_completed"])
        action_index = action_offset + local_index
        trace.append(
            {
                "phase": "act" if source == "online_real_env_confirmation" else "explore",
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


def _predicted_final_state(start_state: R11LObservedState, *, level_completed: int) -> R11LObservedState:
    return R11LObservedState(groups=start_state.groups, level_completed=int(level_completed))


def _run_r11l_explore_first(
    offline_arcade: Any,
    online_arcade: Any,
    candidate: Any,
    *,
    arc_env_count: int,
) -> R11LOutcome:
    env = offline_arcade.make(candidate.game_id)
    frame = env.reset()
    observed = _capture_state(env, frame)
    plan = build_r11l_l1_plan(observed)
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
    commit_start_game = copy.deepcopy(env._game)
    commit_start_state = _capture_state(env, frame)
    phase_trace.append(
        {
            "phase": "induce",
            "source": "offline_explore_first_induction",
            "mechanic": "r11l_click_select_place",
            "goal_predicate": plan.induction_call["goal_predicate"],
            "induction_call": plan.induction_call,
            "exploration_actions_used": len(plan.exploration_actions),
        }
    )

    try:
        final_level, _, _, _ = _execute_actions(
            env,
            frame,
            plan.commit_actions,
            source="offline_gap4_replay_verification",
            action_offset=len(plan.exploration_actions),
        )
    finally:
        env._game = copy.deepcopy(commit_start_game)
    verification = validate_r11l_replayed_plan(
        commit_start_state,
        _predicted_final_state(commit_start_state, level_completed=final_level),
        plan,
    )
    phase_trace.append(verification)

    induced_mechanic = (
        "Observed r11l click-only select/place transitions; induced that each visible "
        "roefwulewcui piece is selected by click and placed by a second click so its "
        "group composite overlaps the matching flkdtg target, with a live level "
        "counter increment required before claiming success."
    )
    if not verification["retained"]:
        return R11LOutcome(
            target_game=candidate.game_id,
            selected_candidate_reason=candidate.selection_reason,
            prior_total_games_solved=9,
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

    online_env = online_arcade.make(candidate.game_id)
    online_frame = online_env.reset()
    real_final_level, first_solve_at_action, act_trace, _ = _execute_actions(
        online_env,
        online_frame,
        plan.actions,
        source="online_real_env_confirmation",
    )
    phase_trace.extend(act_trace)
    solved = real_final_level > 0 and first_solve_at_action > 0
    return R11LOutcome(
        target_game=candidate.game_id,
        selected_candidate_reason=candidate.selection_reason,
        prior_total_games_solved=9,
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
    try:
        arc_env_count = _confirm_arc_env_reachable()
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
    candidate = select_exp4092_candidate_from_survey(survey, baselines)
    offline_arcade = _load_offline_arcade()
    online_arcade = _load_online_arcade()
    outcome = _run_r11l_explore_first(
        offline_arcade,
        online_arcade,
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
        raise ValueError("; ".join(errors))
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
