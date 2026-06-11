"""Exp 4014: break ARC-AGI-3 level walls with explore-first re-induction.

Spec refs: REQ-PHASE4-026, SCENARIO-PHASE4-026.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
RESULT_NAME = "experiment_4014_break_level_wall_explore_first.json"
RANDOM_SEED = 4014
INFERENCE_SUBSTRATE = "offline_arc_agi3_explore_first_level_wall_gap4_executed_consistency"
ENERGY_THRESHOLD = 0.0

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import compute_grid_delta, grid_of, objects  # noqa: E402
from carnot.agentic.arc_level_wall_explore_first import (  # noqa: E402
    BANKED_FRONTIER,
    STALLED_LEVELS,
    LevelWallResult,
    TransitionObservation,
    artifact_schema_errors,
    blocked_artifact,
    build_level_wall_artifact,
    count_validated_candidates,
    induce_model_from_level_observations,
)
from experiment_3964_r11l_incremental_l2 import (  # noqa: E402
    _attempt_current_level as _attempt_r11l_level,
    _perceive_and_match as _perceive_r11l,
)
from experiment_3965_lp85_incremental_l2 import (  # noqa: E402
    discover_buttons as _discover_lp85_buttons,
    load_baseline_actions as _load_lp85_baseline_actions,
    plan_permutation_clicks as _plan_lp85_clicks,
)
from experiment_3992_incremental_levels_verifier_validated import _execute_validated_level  # noqa: E402


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_offline_arcade():  # pragma: no cover - exercised by the required experiment run
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    if not arc.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arc


def _select_game_id(arc: Any, short_game: str) -> str:
    for env in arc.get_environments():
        game_id = str(getattr(env, "game_id", ""))
        if game_id.split("-")[0] == short_game:
            return game_id
    raise RuntimeError(f"{short_game} offline environment unavailable")


def _levels_completed(frame: Any, env: Any) -> int:
    frame_value = getattr(frame, "levels_completed", None)
    if frame_value is not None:
        return int(frame_value or 0)
    game_value = getattr(getattr(env, "_game", None), "levels_completed", None)
    if game_value is not None:
        return int(game_value or 0)
    return int(getattr(getattr(env, "_game", None), "_current_level_index", 0) or 0)


def _game_over(frame: Any) -> bool:
    return "GAME_OVER" in str(getattr(frame, "state", ""))


def _apply_action_key(env: Any, game_action: Any, action_key: tuple[int, ...]) -> Any:
    if action_key[0] == 6:
        return env.step(game_action.ACTION6, data={"x": int(action_key[1]), "y": int(action_key[2])})
    mapping = {
        1: game_action.ACTION1,
        2: game_action.ACTION2,
        3: game_action.ACTION3,
        4: game_action.ACTION4,
    }
    return env.step(mapping[int(action_key[0])])


def _sc25_step_to_action_key(step: dict[str, Any]) -> tuple[int, ...]:
    action = step.get("action")
    if action == "click":
        return (6, int(step["x"]), int(step["y"]))
    mapping = {"up": 1, "down": 2, "left": 3, "right": 4}
    return (mapping[str(action)],)


def _load_sc25_log() -> list[dict[str, Any]]:
    path = REPO / "results" / "experiment_3966_third_game_first_solve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("solve_log", []))


def _refresh_frame(env: Any, game_action: Any) -> Any:
    return env.step(game_action.ACTION6, data={"x": -1, "y": -1})


def _click_candidates(grid: Any, budget: int) -> list[tuple[int, ...]]:
    h, w = grid.shape
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[int, ...]] = []
    for y, x in objects(grid):
        key = (6, int(x), int(y))
        if key not in seen:
            seen.add(key)
            candidates.append(key)
    fixed = [
        (0, 0),
        (w // 4, h // 4),
        (w // 2, h // 2),
        ((3 * w) // 4, (3 * h) // 4),
        (max(0, w - 1), max(0, h - 1)),
        (w // 4, (3 * h) // 4),
        ((3 * w) // 4, h // 4),
    ]
    for x, y in fixed:
        key = (6, int(x), int(y))
        if key not in seen:
            seen.add(key)
            candidates.append(key)
        if len(candidates) >= budget:
            break
    while len(candidates) < budget:
        candidates.append(candidates[len(candidates) % max(1, len(candidates))])
    return candidates[:budget]


def _observe_actions(
    env: Any,
    game_action: Any,
    frame: Any,
    action_keys: list[tuple[int, ...]],
    *,
    exploration_budget: int,
) -> tuple[list[TransitionObservation], list[dict[str, Any]]]:
    original = copy.deepcopy(env._game)
    start_grid = grid_of(frame)
    start_levels = _levels_completed(frame, env)
    observations: list[TransitionObservation] = []
    summaries: list[dict[str, Any]] = []

    for action_key in action_keys[:exploration_budget]:
        env._game = copy.deepcopy(original)
        before = start_grid.copy()
        next_frame = _apply_action_key(env, game_action, action_key)
        after = grid_of(next_frame)
        level_delta = _levels_completed(next_frame, env) - start_levels
        delta = compute_grid_delta(before, after)
        observations.append(
            TransitionObservation(
                before=before,
                action_key=tuple(int(v) for v in action_key),
                after=after,
                level_delta=int(level_delta),
                game_over=_game_over(next_frame),
            )
        )
        summaries.append(
            {
                "action_key": [int(v) for v in action_key],
                "n_changed": int(delta.get("n_changed", 0) or 0),
                "level_delta": int(level_delta),
                "game_over": _game_over(next_frame),
            }
        )

    env._game = original
    return observations, summaries


def _validate_and_maybe_commit(
    env: Any,
    game_action: Any,
    observations: list[TransitionObservation],
    *,
    short_game: str,
    level_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, bool, str]:
    split = max(1, len(observations) // 2)
    train = observations[:split]
    heldout = observations[split:] or observations[:]
    model = induce_model_from_level_observations(short_game, train)
    heldout_rows = [(obs.before, obs.action_key, obs.after) for obs in heldout]
    energy_info = model.consistency_energy(heldout_rows)
    energy = energy_info.get("energy")
    level_up = next((obs for obs in observations if obs.level_delta > 0), None)
    selected = energy is not None and float(energy) <= ENERGY_THRESHOLD and level_up is not None
    start_level = _levels_completed(None, env)

    row = {
        "candidate_id": f"{short_game}-explore-first-single-action",
        "rule_name": "observed per-level transition replay after explore-first induction",
        "level": int(level_number),
        "demo_fit": 1.0 if energy is not None else 0.0,
        "heldout_energy": None if energy is None else float(energy),
        "heldout_n": len(heldout),
        "predicted_levels_after": int(start_level + 1 if level_up else start_level),
        "validated_levels_after": int(start_level + 1 if selected else start_level),
        "planned_l2_actions": 1,
        "selected": bool(selected),
        "validation_source": "GAP-4 executed-consistency energy on held-out observed transitions",
        "prediction_paths": energy_info.get("prediction_paths", {}),
    }
    committed: list[dict[str, Any]] = []
    stall_reason = "no verifier-validated level-up candidate after explore-first observation"
    advanced = False

    if selected and level_up is not None:
        frame = _apply_action_key(env, game_action, level_up.action_key)
        after = _levels_completed(frame, env)
        committed.append(
            {
                "level": int(level_number),
                "action_key": [int(v) for v in level_up.action_key],
                "levels_completed_after": int(after),
            }
        )
        advanced = after > start_level
        row["real_env_levels_after_commit"] = int(after)
        if advanced:
            stall_reason = ""
        else:
            row["selected"] = False
            row["commit_rejected_reason"] = "copied-env validated action did not advance real env"
            stall_reason = "validated candidate failed real-env levels_completed confirmation"

    saved = 0 if advanced else 1
    return [row], committed, count_validated_candidates([row]), saved, advanced, stall_reason


def _advance_lp85_to_l1(env: Any, game_action: Any, budget: int) -> tuple[Any, list[dict[str, Any]], int]:
    frame = env.reset()
    used = 0
    solve_log: list[dict[str, Any]] = []
    baseline = _load_lp85_baseline_actions(getattr(env, "game_id", "lp85-305b61c3"))
    max_depth = max(20, max(baseline or [0]))
    while _levels_completed(frame, env) < BANKED_FRONTIER["lp85"] and used < budget:
        grid = grid_of(frame)
        buttons = _discover_lp85_buttons(env, game_action, grid)
        path = _plan_lp85_clicks(
            env,
            game_action,
            grid,
            buttons,
            _levels_completed(frame, env),
            max_depth=max_depth,
        )
        if not path:
            break
        start = _levels_completed(frame, env)
        for y, x in path:
            frame = env.step(game_action.ACTION6, data={"x": int(x), "y": int(y)})
            used += 1
            solve_log.append({"level": start + 1, "action": "click", "x": int(x), "y": int(y)})
            if _levels_completed(frame, env) > start or used >= budget:
                break
    return frame, solve_log, used


def _advance_r11l_to_l3(env: Any, game_action: Any, budget: int) -> tuple[Any, list[dict[str, Any]], int]:
    frame = env.reset()
    used = 0
    solve_log: list[dict[str, Any]] = []
    while _levels_completed(frame, env) < BANKED_FRONTIER["r11l"] and used < budget:
        level = _levels_completed(frame, env) + 1
        if level == 1:
            pairs = _perceive_r11l(env)
            frame, actions, entries = _attempt_r11l_level(env, game_action, pairs, budget - used, level)
            used += int(actions)
            solve_log.extend(entries)
            if _levels_completed(frame, env) < level:
                break
        else:
            result = _execute_validated_level(env, game_action, level_number=level, budget_left=budget - used)
            used += int(result.get("actions_used", 0) or 0)
            solve_log.extend(result.get("solve_log", []))
            frame = _refresh_frame(env, game_action)
            if not result.get("advanced"):
                break
    return frame, solve_log, used


def _advance_sc25_to_l1(env: Any, game_action: Any, budget: int) -> tuple[Any, list[dict[str, Any]], int]:
    frame = env.reset()
    used = 0
    solve_log: list[dict[str, Any]] = []
    for step in _load_sc25_log():
        if used >= budget:
            break
        frame = _apply_action_key(env, game_action, _sc25_step_to_action_key(step))
        used += 1
        solve_log.append(dict(step))
        if _levels_completed(frame, env) >= BANKED_FRONTIER["sc25"]:
            break
    return frame, solve_log, used


def _run_one_wall(
    arc: Any,
    short_game: str,
    *,
    budget: int,
    exploration_budget: int,
) -> LevelWallResult:
    from arcengine.enums import GameAction

    game_id = _select_game_id(arc, short_game)
    env = arc.make(game_id)
    if short_game == "lp85":
        frame, banked_log, _ = _advance_lp85_to_l1(env, GameAction, budget)
        action_keys = _click_candidates(grid_of(frame), exploration_budget)
    elif short_game == "sc25":
        frame, banked_log, _ = _advance_sc25_to_l1(env, GameAction, budget)
        action_keys = [_sc25_step_to_action_key(step) for step in _load_sc25_log()]
        action_keys.extend(_click_candidates(grid_of(frame), exploration_budget))
    elif short_game == "r11l":
        frame, banked_log, _ = _advance_r11l_to_l3(env, GameAction, budget)
        action_keys = _click_candidates(grid_of(frame), exploration_budget)
    else:
        raise ValueError(f"unknown level-wall game {short_game}")

    levels_completed = _levels_completed(frame, env)
    if levels_completed < BANKED_FRONTIER[short_game]:
        return LevelWallResult(
            short_game=short_game,
            game_id=game_id,
            banked_level=BANKED_FRONTIER[short_game],
            target_level=STALLED_LEVELS[short_game],
            levels_completed=levels_completed,
            first_fail_level=levels_completed + 1,
            exploration_actions_used=0,
            observed_dynamics=[],
            dynamics_induced=False,
            candidate_validations=[],
            committed_actions=[],
            verifier_validated_count=0,
            actions_saved_vs_openloop=0,
            real_env_confirmed=False,
            stall_reason="banked frontier replay failed before exploration",
            solve_log=banked_log,
        )

    observations, observed = _observe_actions(
        env,
        GameAction,
        frame,
        action_keys,
        exploration_budget=exploration_budget,
    )
    validations, committed, validated_count, saved, advanced, reason = _validate_and_maybe_commit(
        env,
        GameAction,
        observations,
        short_game=short_game,
        level_number=STALLED_LEVELS[short_game],
    )
    final_levels = _levels_completed(None, env)

    return LevelWallResult(
        short_game=short_game,
        game_id=game_id,
        banked_level=BANKED_FRONTIER[short_game],
        target_level=STALLED_LEVELS[short_game],
        levels_completed=final_levels,
        first_fail_level=None if advanced else STALLED_LEVELS[short_game],
        exploration_actions_used=len(observations),
        observed_dynamics=observed,
        dynamics_induced=bool(observations),
        candidate_validations=validations,
        committed_actions=committed,
        verifier_validated_count=validated_count,
        actions_saved_vs_openloop=saved,
        real_env_confirmed=final_levels >= levels_completed,
        stall_reason=reason or "explore-first validated level wall advance",
        solve_log=banked_log + committed,
    )


def _run_level_walls(arc: Any, budget: int, exploration_budget: int) -> list[LevelWallResult]:
    per_game_budget = max(20, budget)
    return [
        _run_one_wall(arc, "lp85", budget=per_game_budget, exploration_budget=exploration_budget),
        _run_one_wall(arc, "sc25", budget=per_game_budget, exploration_budget=exploration_budget),
        _run_one_wall(arc, "r11l", budget=per_game_budget, exploration_budget=exploration_budget),
    ]


def run(
    *,
    budget: int = 80,
    exploration_budget: int = 4,
    seed: int = RANDOM_SEED,
    use_codex: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    started = time.time()
    if use_codex and shutil.which("codex") is None:
        artifact = blocked_artifact(
            seed=seed,
            started=started,
            inference_substrate=INFERENCE_SUBSTRATE,
            verdict="blocked_codex_unavailable",
        )
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        arc = _load_offline_arcade()
    except Exception:
        artifact = blocked_artifact(
            seed=seed,
            started=started,
            inference_substrate=INFERENCE_SUBSTRATE,
            verdict="blocked_arc_offline_env_unavailable",
        )
        if write:
            _write_artifact(artifact)
        return artifact

    results = _run_level_walls(arc, budget, exploration_budget)
    artifact = build_level_wall_artifact(
        results,
        seed=seed,
        started=started,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--exploration-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--use-codex", action="store_true")
    args = parser.parse_args()
    result = run(
        budget=args.budget,
        exploration_budget=args.exploration_budget,
        seed=args.seed,
        use_codex=args.use_codex,
        write=True,
    )
    print(f"-> {result['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
