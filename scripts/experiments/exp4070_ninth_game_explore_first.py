"""Exp 4070: ninth ARC-AGI-3 game solve via explore-first.

Spec refs: REQ-PHASE4-042, SCENARIO-PHASE4-042.
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
ENVDIR = str(REPO / "environment_files")
RESULT_NAME = "experiment_4070_ninth_game_explore_first.json"
RANDOM_SEED = 4070

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (  # noqa: E402
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    ExperimentOutcome,
    Ft09Action,
    Ft09Cell,
    Ft09Constraint,
    Ft09ObservedState,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_ft09_l1_plan,
    load_environment_baselines,
    select_ninth_candidate_from_survey,
    validate_replayed_plan,
)


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _confirm_arc_env_reachable() -> int:  # pragma: no cover - exercised by required live run
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arcade = Arcade(arc_api_key="", operation_mode=OperationMode.ONLINE, environments_dir=ENVDIR)
    environments = arcade.get_environments()
    if not environments:
        raise RuntimeError("live ARC catalog returned no environments")
    return int(len(environments))


def _load_offline_arcade():  # pragma: no cover - exercised by required live run
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arcade = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    if not arcade.get_environments():
        raise RuntimeError("offline ARC catalog returned no environments")
    return arcade


def _load_online_arcade():  # pragma: no cover - exercised by required live run
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(arc_api_key="", operation_mode=OperationMode.ONLINE, environments_dir=ENVDIR)


def _level_completed(frame: Any, env: Any) -> int:
    for attr in ("levels_completed", "level_completed"):
        value = getattr(frame, attr, None)
        if value is not None:
            return int(value or 0)
    game = getattr(env, "_game", None)
    for attr in ("levels_completed", "level_completed", "_current_level_index", "level_index"):
        value = getattr(game, attr, None)
        if value is not None:
            return int(value or 0)
    return 0


def _game_over(frame: Any) -> bool:
    return "GAME_OVER" in str(getattr(frame, "state", ""))


def _step(env: Any, action: Ft09Action) -> Any:
    from arcengine.enums import GameAction

    if int(action.action) == 6:
        return env.step(
            GameAction.ACTION6,
            data={"x": int(action.x), "y": int(action.y)},
        )
    raise ValueError(f"unsupported ft09 action {action.action}")


def _pattern_tuple(sprite: Any) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return tuple(tuple(int(value) for value in row) for row in sprite.pixels.tolist())  # type: ignore[return-value]


def _capture_state(env: Any, frame: Any) -> Ft09ObservedState:
    game = env._game
    constraints = tuple(
        Ft09Constraint(
            grid=(int(sprite.x), int(sprite.y)),
            center_color=int(sprite.pixels[1][1]),
            pattern=_pattern_tuple(sprite),
        )
        for sprite in sorted(game.gig, key=lambda item: (int(item.y), int(item.x)))
    )
    cells = tuple(
        Ft09Cell(
            grid=(int(sprite.x), int(sprite.y)),
            color=int(sprite.pixels[1][1]),
            kind=str(sprite.name),
        )
        for sprite in sorted([*game.fhc, *game.mou], key=lambda item: (int(item.y), int(item.x)))
    )
    return Ft09ObservedState(
        constraints=constraints,
        cells=cells,
        color_cycle=tuple(int(color) for color in game.gqb),
        level_completed=_level_completed(frame, env),
    )


def _capture_trace_state(env: Any, frame: Any) -> dict[str, Any]:
    if hasattr(env, "_game"):
        return _capture_state(env, frame).to_json()
    return {
        "level_completed": _level_completed(frame, env),
        "state": str(getattr(frame, "state", "")),
    }


def _execute_actions(
    env: Any,
    frame: Any,
    actions: list[Ft09Action],
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


def _predicted_final_state(
    start_state: Ft09ObservedState,
    plan: Any,
    *,
    level_completed: int,
) -> Ft09ObservedState:
    return Ft09ObservedState(
        constraints=start_state.constraints,
        cells=tuple(
            Ft09Cell(grid=cell.grid, color=plan.predicted_cell_colors[tuple(cell.grid)], kind=cell.kind)
            for cell in start_state.cells
        ),
        color_cycle=start_state.color_cycle,
        level_completed=int(level_completed),
    )


def _run_ft09_explore_first(
    offline_arcade: Any,
    online_arcade: Any,
    candidate: Any,
    *,
    arc_env_count: int,
) -> ExperimentOutcome:
    env = offline_arcade.make(candidate.game_id)
    frame = env.reset()
    observed = _capture_state(env, frame)
    plan = build_ft09_l1_plan(observed)
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
            "mechanic": "ft09_local_constraint_color_cycle",
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
    verification = validate_replayed_plan(
        commit_start_state,
        _predicted_final_state(commit_start_state, plan, level_completed=final_level),
        plan,
    )
    phase_trace.append(verification)

    induced_mechanic = (
        "Observed ft09 Hkx cells cycle colors on click; induced a local non-navigation "
        "constraint model where bsT zero pixels require neighboring cells to equal the "
        "constraint center color, non-zero pixels require inequality, and a real level "
        "counter increment is required before the plan is retained."
    )
    if not verification["retained"]:
        return ExperimentOutcome(
            target_game=candidate.game_id,
            selected_candidate_reason=candidate.selection_reason,
            prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
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
    return ExperimentOutcome(
        target_game=candidate.game_id,
        selected_candidate_reason=candidate.selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
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
    candidate = select_ninth_candidate_from_survey(survey, baselines)
    offline_arcade = _load_offline_arcade()
    online_arcade = _load_online_arcade()
    outcome = _run_ft09_explore_first(
        offline_arcade,
        online_arcade,
        candidate,
        arc_env_count=arc_env_count,
    )
    artifact = build_artifact(
        outcome,
        random_seed=seed,
        duration_s=round(time.time() - started, 3),
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    artifact["excluded_solved_games"] = list(candidate.excluded_solved_games)
    artifact["candidate_baseline_actions"] = int(candidate.baseline_actions)
    artifact["selected_candidate_reason"] = candidate.selection_reason
    artifact["selection_mode"] = candidate.selection_mode
    artifact["survey_is_spatial_planning"] = bool(candidate.survey_is_spatial_planning)
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
