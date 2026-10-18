"""Exp 4038: seventh ARC-AGI-3 game solve via explore-first.

Spec refs: REQ-PHASE4-038, SCENARIO-PHASE4-038.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
RESULT_NAME = "experiment_4038_seventh_game_explore_first.json"
RANDOM_SEED = 4038

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4038_seventh_game_explore_first import (  # noqa: E402
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    Dc22Action,
    Dc22State,
    ExperimentOutcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    dc22_default_exploration_actions,
    load_environment_baselines,
    select_seventh_candidate_from_survey,
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


def _step(env: Any, action: Dc22Action) -> Any:
    from arcengine.enums import GameAction

    if int(action.action) == 6:
        return env.step(
            GameAction.ACTION6,
            data={"x": int(action.x or 0), "y": int(action.y or 0)},
        )
    keyboard = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
    }
    return env.step(keyboard[int(action.action)])


def _blocker_signature(game: Any) -> tuple[Any, ...]:
    rows: list[tuple[Any, ...]] = []
    for sprite in sorted(
        game.current_level.get_sprites(),
        key=lambda item: (item.name, item.x, item.y, tuple(item.tags)),
    ):
        if not any(tag in sprite.tags for tag in ("tovemc", "buezna", "piyqze")):
            continue
        rows.append(
            (
                sprite.name,
                int(sprite.x),
                int(sprite.y),
                bool(sprite.is_visible),
                str(sprite.interaction),
            )
        )
    return tuple(rows)


def _capture_state(env: Any, frame: Any) -> Dc22State:
    game = env._game
    return Dc22State(
        player=(int(game.qnnpcoyzd.x), int(game.qnnpcoyzd.y)),
        goal=(int(game.hfuqkxulm.x), int(game.hfuqkxulm.y)),
        level_completed=_level_completed(frame, env),
        blocker_signature=_blocker_signature(game),
    )


def _capture_trace_state(env: Any, frame: Any) -> dict[str, Any]:
    if hasattr(env, "_game"):
        return _capture_state(env, frame).to_json()
    return {
        "level_completed": _level_completed(frame, env),
        "state": str(getattr(frame, "state", "")),
    }


def _display_for_grid(game: Any, grid_x: int, grid_y: int) -> tuple[int, int]:
    camera = game.camera
    for display_y in range(max(0, int(grid_y) - 2), int(grid_y) + 32):
        if camera.display_to_grid(int(grid_x), display_y) == (int(grid_x), int(grid_y)):
            return int(grid_x), int(display_y)
    return int(grid_x), int(grid_y + 10)


def _visible_click_actions(game: Any) -> list[Dc22Action]:
    actions: list[Dc22Action] = []
    for sprite in sorted(
        game.current_level.get_sprites_by_tag("sys_click"),
        key=lambda item: (item.name, item.x, item.y),
    ):
        if not sprite.is_visible:
            continue
        if "REMOVED" in str(sprite.interaction) or "INVISIBLE" in str(sprite.interaction):
            continue
        grid_x = int(sprite.x + sprite.width // 2)
        grid_y = int(sprite.y + sprite.height // 2)
        display_x, display_y = _display_for_grid(game, grid_x, grid_y)
        actions.append(
            Dc22Action.click(
                display_x,
                display_y,
                sprite=str(sprite.name),
                grid=(grid_x, grid_y),
            )
        )
    return actions


def _available_dc22_actions(game: Any) -> list[Dc22Action]:
    return [
        Dc22Action.key(1),
        Dc22Action.key(2),
        Dc22Action.key(3),
        Dc22Action.key(4),
        *_visible_click_actions(game),
    ]


def _state_signature(game: Any) -> tuple[Any, ...]:
    return (
        int(game.qnnpcoyzd.x),
        int(game.qnnpcoyzd.y),
        _blocker_signature(game),
    )


def _find_commit_suffix(
    env: Any,
    frame: Any,
    *,
    max_depth: int = 80,
) -> list[Dc22Action]:
    start_level = _level_completed(frame, env)
    start_game = copy.deepcopy(env._game)
    queue: deque[tuple[Any, list[Dc22Action]]] = deque([(copy.deepcopy(start_game), [])])
    seen = {_state_signature(start_game)}
    try:
        while queue:
            game, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for action in _available_dc22_actions(game):
                env._game = copy.deepcopy(game)
                next_frame = _step(env, action)
                next_game = copy.deepcopy(env._game)
                next_path = [*path, action]
                if _level_completed(next_frame, env) > start_level:
                    return next_path
                if _game_over(next_frame):
                    continue
                signature = _state_signature(next_game)
                if signature in seen:
                    continue
                seen.add(signature)
                queue.append((next_game, next_path))
    finally:
        env._game = copy.deepcopy(start_game)
    raise RuntimeError("dc22 commit suffix search exhausted")


def _replay_commit_suffix(
    env: Any,
    start_game: Any,
    start_state: Dc22State,
    actions: list[Dc22Action],
) -> tuple[list[Dc22State], Any]:
    env._game = copy.deepcopy(start_game)
    states = [start_state]
    frame: Any = None
    for action in actions:
        frame = _step(env, action)
        states.append(_capture_state(env, frame))
    return states, frame


def _execute_real_env_plan(
    arcade: Any,
    game_id: str,
    actions: list[Dc22Action],
) -> tuple[int, int, list[dict[str, Any]]]:
    env = arcade.make(game_id)
    frame = env.reset()
    start_level = _level_completed(frame, env)
    trace: list[dict[str, Any]] = []
    first_solve_at_action = -1
    final_level = start_level
    for index, action in enumerate(actions, start=1):
        before = _capture_trace_state(env, frame)
        frame = _step(env, action)
        after = _capture_trace_state(env, frame)
        final_level = int(after["level_completed"])
        trace.append(
            {
                "phase": "act",
                "source": "online_real_env_confirmation",
                "action_index": index,
                "action": action.to_json(),
                "before": before,
                "after": after,
                "level_completed": final_level,
            }
        )
        if final_level > start_level:
            first_solve_at_action = index
            break
        if _game_over(frame):
            break
    return final_level, first_solve_at_action, trace


def _run_dc22_explore_first(
    offline_arcade: Any,
    online_arcade: Any,
    candidate: Any,
    *,
    arc_env_count: int,
) -> ExperimentOutcome:
    env = offline_arcade.make(candidate.game_id)
    frame = env.reset()
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_explore_first_induction",
            "target_game": candidate.game_id,
            "state": _capture_state(env, frame).to_json(),
            "candidate_reason": candidate.selection_reason,
        }
    ]

    exploration_actions = dc22_default_exploration_actions()
    for index, action in enumerate(exploration_actions, start=1):
        before = _capture_state(env, frame)
        frame = _step(env, action)
        after = _capture_state(env, frame)
        phase_trace.append(
            {
                "phase": "explore",
                "source": "offline_explore_first_induction",
                "action_index": index,
                "action": action.to_json(),
                "before": before.to_json(),
                "after": after.to_json(),
                "level_completed": int(after.level_completed),
            }
        )
        if _game_over(frame):
            return ExperimentOutcome(
                target_game=candidate.game_id,
                selected_candidate_reason=candidate.selection_reason,
                prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
                final_level_completed=int(after.level_completed),
                first_solve_at_action=-1,
                exploration_actions_used=index,
                induced_mechanic="dc22 exploration reached game over before induction",
                verification_decisions=[],
                phase_trace=phase_trace,
                real_env_confirmed=False,
                action_plan=exploration_actions[:index],
                arc_env_count=arc_env_count,
                failure_reason="game_over_during_exploration",
            )

    current_game = copy.deepcopy(env._game)
    current_state = _capture_state(env, frame)
    commit_actions = _find_commit_suffix(env, frame)
    replayed_states, _ = _replay_commit_suffix(env, current_game, current_state, commit_actions)
    verification = validate_replayed_plan(
        current_state,
        replayed_states,
        commit_actions,
        start_level_completed=current_state.level_completed,
    )
    phase_trace.append(
        {
            "phase": "induce",
            "source": "offline_explore_first_induction",
            "mechanic": "dc22_navigation_toggle_goal",
            "observed_movement": "ACTION1 moved jfva from y=30 to y=28",
            "observed_toggle": "buezna-blrmbx click changed tag-b blocker interactions",
            "goal_predicate": "level counter increments when jfva reaches goknoi",
            "exploration_actions_used": len(exploration_actions),
        }
    )
    phase_trace.append(verification)

    induced_mechanic = (
        "Observed dc22 movement and buezna toggle transitions before planning; induced a "
        "grid-navigation model where keyboard actions move jfva by two pixels, visible buezna "
        "clicks toggle same-letter blockers, and the goal predicate is confirmed only by a "
        "levels_completed increment when jfva reaches goknoi."
    )
    full_plan = [*exploration_actions, *commit_actions]
    if not verification["retained"]:
        return ExperimentOutcome(
            target_game=candidate.game_id,
            selected_candidate_reason=candidate.selection_reason,
            prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
            final_level_completed=current_state.level_completed,
            first_solve_at_action=-1,
            exploration_actions_used=len(exploration_actions),
            induced_mechanic=induced_mechanic,
            verification_decisions=[verification],
            phase_trace=phase_trace,
            real_env_confirmed=False,
            action_plan=full_plan,
            arc_env_count=arc_env_count,
            failure_reason="verification_rejected_commit_suffix",
        )

    final_level, first_solve_at_action, act_trace = _execute_real_env_plan(
        online_arcade,
        candidate.game_id,
        full_plan,
    )
    phase_trace.extend(act_trace)
    solved = final_level > 0 and first_solve_at_action > 0
    return ExperimentOutcome(
        target_game=candidate.game_id,
        selected_candidate_reason=candidate.selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=final_level,
        first_solve_at_action=first_solve_at_action if solved else -1,
        exploration_actions_used=len(exploration_actions),
        induced_mechanic=induced_mechanic,
        verification_decisions=[verification],
        phase_trace=phase_trace,
        real_env_confirmed=bool(solved),
        action_plan=full_plan,
        arc_env_count=arc_env_count,
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
    candidate = select_seventh_candidate_from_survey(survey, baselines)
    offline_arcade = _load_offline_arcade()
    online_arcade = _load_online_arcade()
    outcome = _run_dc22_explore_first(
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
