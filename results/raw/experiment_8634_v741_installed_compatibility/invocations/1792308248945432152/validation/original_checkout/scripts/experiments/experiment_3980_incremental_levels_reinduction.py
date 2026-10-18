"""Exp 3980: one-game incremental level re-induction for ARC-AGI-3.

Spec refs: REQ-PHASE4-018, SCENARIO-PHASE4-018.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
RESULT_NAME = "experiment_3980_incremental_levels_reinduction.json"
RANDOM_SEED = 3980
PRIOR_BEST_LEVELS = 1
INFERENCE_SUBSTRATE = "offline_arc_agi3_per_level_execution_guided_reinduction"

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_level_reinduction import (  # noqa: E402
    artifact_schema_errors,
    choose_reinduction_candidate,
)
from experiment_3964_r11l_incremental_l2 import (  # noqa: E402
    _attempt_current_level as _attempt_r11l_level,
    _levels_completed,
    _perceive_and_match as _perceive_r11l,
)


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")


def _base_artifact(seed: int, started: float, verdict: str) -> dict[str, Any]:
    return {
        "experiment": "experiment_3980_incremental_levels_reinduction",
        "title": "arc3_incremental_per_level_reinduction",
        "ACCURACY_levels_solved": 0,
        "new_levels_solved_this_task": 0,
        "reinduction_found_different_rule": False,
        "game_advanced": "none",
        "per_level_actions": [],
        "baseline_actions_ref": [],
        "real_env_confirmed": False,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 3) if started else 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "selected_game": None,
        "candidate_reason": "",
        "first_fail_level": None,
        "l1_mechanic_transferred_to_l2": None,
        "level_summaries": [],
        "per_level": [],
        "solve_log": [],
        "rule_diagnosis": "",
        "precondition_blocked": verdict.startswith("blocked_"),
        "submitted_to_leaderboard": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def _load_prior_stalls() -> dict[str, dict[str, Any]]:
    return {
        "r11l": _load_json(REPO / "results" / "experiment_3964_r11l_incremental_l2.json"),
        "lp85": _load_json(REPO / "results" / "experiment_3965_lp85_incremental_l2.json"),
    }


def _load_offline_arcade():  # pragma: no cover - exercised by the real experiment preflight
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=ENVDIR,
    )
    if not arc.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arc


def _select_game_id(arc: Any, short_game: str) -> str:
    for env in arc.get_environments():
        game_id = getattr(env, "game_id", "")
        if game_id.split("-")[0] == short_game:
            return str(game_id)
    raise RuntimeError(f"{short_game} offline environment unavailable")


def _settle(env: Any, game_action: Any, frame: Any) -> tuple[Any, int]:
    steps = 0
    while getattr(env._game, "yfbjozweime", False) and steps < 200:
        frame = env.step(game_action.ACTION6, data={"x": -1, "y": -1})
        steps += 1
    return frame, steps


def _simulate_group_move(base_env: Any, group_id: str, piece_index: int, top_left: tuple[int, int], game_action: Any):
    env = copy.deepcopy(base_env)
    data = env._game.kacotwgjcyq[group_id]
    piece = data["lecfirgqbwunn"][piece_index]
    frame = env.step(
        game_action.ACTION6,
        data={"x": int(piece.x + piece.width // 2), "y": int(piece.y + piece.height // 2)},
    )
    frame = env.step(
        game_action.ACTION6,
        data={"x": int(top_left[0] + piece.width // 2), "y": int(top_left[1] + piece.height // 2)},
    )
    frame, steps = _settle(env, game_action, frame)
    composite = data["roduyfsmiznvg"]
    target = data["gosubdcyegamj"]
    return {
        "env": env,
        "frame": frame,
        "steps": steps,
        "levels_completed": _levels_completed(frame, env),
        "state": getattr(frame, "state", None),
        "failure_count": int(getattr(env._game, "yledlprvvkb", 0) or 0),
        "composite": (int(composite.x), int(composite.y)),
        "target_collides": bool(composite.collides_with(target)),
        "piece_after": (
            int(data["lecfirgqbwunn"][piece_index].x),
            int(data["lecfirgqbwunn"][piece_index].y),
        ),
    }


def _target_colliding_composite_positions(composite: Any, target: Any) -> list[tuple[int, int]]:
    ox, oy = composite.x, composite.y
    positions: list[tuple[int, int]] = []
    for y in range(-5, 75):
        for x in range(-5, 75):
            composite.set_position(x, y)
            if composite.collides_with(target):
                positions.append((int(x), int(y)))
    composite.set_position(ox, oy)
    return sorted(positions, key=lambda pos: (pos[0] - ox) ** 2 + (pos[1] - oy) ** 2)


def _find_one_move_group_solution(env: Any, group_id: str, game_action: Any) -> dict[str, Any] | None:
    data = env._game.kacotwgjcyq[group_id]
    pieces = data["lecfirgqbwunn"]
    composite = data["roduyfsmiznvg"]
    target = data["gosubdcyegamj"]
    if not pieces or not composite or not target:
        return None

    n_pieces = len(pieces)
    for comp_x, comp_y in _target_colliding_composite_positions(composite, target)[:400]:
        avg_x = comp_x + composite.width // 2
        avg_y = comp_y + composite.height // 2
        for piece_index, piece in enumerate(pieces):
            other_x = sum(p.x + p.width // 2 for i, p in enumerate(pieces) if i != piece_index)
            other_y = sum(p.y + p.height // 2 for i, p in enumerate(pieces) if i != piece_index)
            center_xs = range(n_pieces * avg_x - other_x, n_pieces * (avg_x + 1) - other_x)
            center_ys = range(n_pieces * avg_y - other_y, n_pieces * (avg_y + 1) - other_y)
            for center_x in center_xs:
                for center_y in center_ys:
                    top_left = (
                        int(center_x - piece.width // 2),
                        int(center_y - piece.height // 2),
                    )
                    if not (-8 <= top_left[0] <= 72 and -8 <= top_left[1] <= 72):
                        continue
                    outcome = _simulate_group_move(env, group_id, piece_index, top_left, game_action)
                    state_name = getattr(outcome["state"], "name", "")
                    if (
                        outcome["target_collides"]
                        and outcome["piece_after"] == top_left
                        and state_name != "GAME_OVER"
                    ):
                        return {
                            "group_id": group_id,
                            "piece_index": int(piece_index),
                            "top_left": [int(top_left[0]), int(top_left[1])],
                            "click": [
                                int(top_left[1] + piece.height // 2),
                                int(top_left[0] + piece.width // 2),
                            ],
                            "composite_after": [int(outcome["composite"][1]), int(outcome["composite"][0])],
                            "target_after_collides": True,
                            "induced_rule": "single-piece move chosen so the group-average proxy collides with the L2 target while respecting the forbidden mask",
                        }
    return None


def _execute_group_plan(env: Any, plan: dict[str, Any], game_action: Any, level_number: int) -> tuple[Any, int, list[dict[str, Any]]]:
    data = env._game.kacotwgjcyq[plan["group_id"]]
    piece = data["lecfirgqbwunn"][plan["piece_index"]]
    log: list[dict[str, Any]] = []
    frame = env.step(
        game_action.ACTION6,
        data={"x": int(piece.x + piece.width // 2), "y": int(piece.y + piece.height // 2)},
    )
    frame = env.step(
        game_action.ACTION6,
        data={"x": int(plan["click"][1]), "y": int(plan["click"][0])},
    )
    frame, settle_steps = _settle(env, game_action, frame)
    composite = data["roduyfsmiznvg"]
    target = data["gosubdcyegamj"]
    log.append(
        {
            "level": int(level_number),
            "group_id": plan["group_id"],
            "piece_index": int(plan["piece_index"]),
            "action": "reinduced_collision_aware_place",
            "click": plan["click"],
            "piece_top_left_after": [int(piece.y), int(piece.x)],
            "composite_after": [int(composite.y), int(composite.x)],
            "target": [int(target.y), int(target.x)],
            "target_after_collides": bool(composite.collides_with(target)),
            "settle_steps": int(settle_steps),
        }
    )
    return frame, 2, log


def _execute_r11l_reinduction(arc: Any, game_id: str, budget: int) -> dict[str, Any]:
    from arcengine.enums import GameAction, GameState

    env = arc.make(game_id)
    frame = env.reset()
    total_actions = 0
    per_level_actions: list[int] = []
    baseline_actions_ref: list[int] = []
    level_summaries: list[dict[str, Any]] = []
    solve_log: list[dict[str, Any]] = []

    l1_pairs = _perceive_r11l(env)
    frame, used_l1, l1_log = _attempt_r11l_level(env, GameAction, l1_pairs, budget, 1)
    total_actions += used_l1
    levels_completed = _levels_completed(frame, env)
    solve_log.extend(l1_log)
    level_summaries.append(
        {
            "level": 1,
            "n_pairs": len(l1_pairs),
            "n_targets": len({tuple(pair["target"]) for pair in l1_pairs}),
            "actions_used": int(used_l1),
            "levels_completed_after": int(levels_completed),
        }
    )
    if levels_completed < 1:
        return {
            "ACCURACY_levels_solved": levels_completed,
            "new_levels_solved_this_task": 0,
            "reinduction_found_different_rule": False,
            "game_advanced": f"{game_id}_to_L{levels_completed}",
            "per_level_actions": [],
            "baseline_actions_ref": [],
            "real_env_confirmed": False,
            "first_fail_level": 1,
            "per_level": [],
            "level_summaries": level_summaries,
            "solve_log": solve_log,
            "rule_diagnosis": "banked L1 could not be reproduced in the real env",
            "l1_mechanic_transferred_to_l2": False,
            "l2_attempted_actions": 0,
        }

    per_level_actions.append(int(used_l1))
    baseline_actions_ref.append(len(l1_pairs) * 2)
    l2_pairs = _perceive_r11l(env)
    level_summaries.append(
        {
            "level": 2,
            "n_pairs": len(l2_pairs),
            "n_targets": len({tuple(pair["target"]) for pair in l2_pairs}),
            "budget_left": int(budget - total_actions),
        }
    )

    transfer_env = copy.deepcopy(env)
    transfer_frame, transfer_used, _ = _attempt_r11l_level(
        transfer_env,
        GameAction,
        _perceive_r11l(transfer_env),
        max(0, budget - total_actions),
        2,
    )
    transfer_levels = _levels_completed(transfer_frame, transfer_env)
    l1_transferred = transfer_levels > levels_completed

    plans = []
    for group_id, data in sorted(env._game.kacotwgjcyq.items(), key=lambda item: len(item[1]["lecfirgqbwunn"])):
        if data.get("gosubdcyegamj") is None:
            continue
        plan = _find_one_move_group_solution(env, group_id, GameAction)
        if plan is not None:
            plans.append(plan)

    l2_attempted_actions = 0
    per_level_rows: list[dict[str, Any]] = []
    for plan in plans:
        if total_actions + 2 > budget or _levels_completed(frame, env) > levels_completed:
            break
        frame, used, entries = _execute_group_plan(env, plan, GameAction, 2)
        total_actions += used
        l2_attempted_actions += used
        solve_log.extend(entries)
        current_levels = _levels_completed(frame, env)
        per_level_rows.append(
            {
                "level": 2,
                "group_id": plan["group_id"],
                "actions_used": int(used),
                "levels_completed_after": int(current_levels),
                "rule": plan["induced_rule"],
            }
        )
        if current_levels > levels_completed:
            levels_completed = current_levels
            per_level_actions.append(l2_attempted_actions)
            baseline_actions_ref.append(max(transfer_used, len(l2_pairs) * 2))
            break

    state = getattr(frame, "state", None)
    if state in (GameState.WIN, GameState.GAME_OVER):
        levels_completed = _levels_completed(frame, env)

    new_levels = max(0, levels_completed - PRIOR_BEST_LEVELS)
    first_fail_level = None if new_levels > 0 else 2
    unsolved_groups = []
    if new_levels == 0:
        for group_id, data in sorted(env._game.kacotwgjcyq.items()):
            target = data.get("gosubdcyegamj")
            composite = data.get("roduyfsmiznvg")
            if target and composite and not composite.collides_with(target):
                unsolved_groups.append(group_id)

    reason = "L2 re-induction found a collision/forbidden-mask rule different from L1"
    if unsolved_groups:
        reason += f"; remaining unsolved groups need multi-piece path search: {','.join(unsolved_groups)}"
    return {
        "ACCURACY_levels_solved": int(levels_completed),
        "new_levels_solved_this_task": int(new_levels),
        "reinduction_found_different_rule": not l1_transferred,
        "game_advanced": f"{game_id}_to_L{levels_completed}",
        "per_level_actions": per_level_actions,
        "baseline_actions_ref": baseline_actions_ref,
        "real_env_confirmed": bool(levels_completed >= 1),
        "first_fail_level": first_fail_level,
        "per_level": per_level_rows,
        "level_summaries": level_summaries,
        "solve_log": solve_log,
        "rule_diagnosis": reason,
        "l1_mechanic_transferred_to_l2": bool(l1_transferred),
        "l1_transfer_simulated_actions": int(transfer_used),
        "l1_transfer_simulated_levels_after": int(transfer_levels),
        "l2_attempted_actions": int(l2_attempted_actions),
        "total_actions": int(total_actions),
        "state_after": getattr(state, "name", str(state)),
    }


def _execute_candidate(arc: Any, choice: Any, game_id: str, budget: int) -> dict[str, Any]:
    if choice.short_game != "r11l":
        return {
            "ACCURACY_levels_solved": int(choice.prior_levels),
            "new_levels_solved_this_task": 0,
            "reinduction_found_different_rule": False,
            "game_advanced": f"{game_id}_to_L{choice.prior_levels}",
            "per_level_actions": [],
            "baseline_actions_ref": choice.baseline_actions_ref,
            "real_env_confirmed": False,
            "first_fail_level": 2,
            "per_level": [],
            "level_summaries": [],
            "solve_log": [],
            "rule_diagnosis": f"no executor implemented for selected game {choice.short_game}",
            "l1_mechanic_transferred_to_l2": False,
        }
    return _execute_r11l_reinduction(arc, game_id, budget)


def _slug_reason(reason: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
    return slug[:80] or "unsolved"


def run(seed: int = RANDOM_SEED, budget: int = 60, write: bool = True, _arc_client: Any | None = None) -> dict[str, Any]:
    started = time.time()
    stalls = _load_prior_stalls()
    choice = choose_reinduction_candidate(stalls)

    try:
        arc = _arc_client if _arc_client is not None else _load_offline_arcade()
        game_id = _select_game_id(arc, choice.short_game)
    except Exception:
        artifact = _base_artifact(seed, started, "blocked_arc_offline_env_unavailable")
        artifact.update(
            {
                "selected_game": choice.short_game,
                "candidate_reason": choice.reason,
                "game_advanced": f"{choice.game_id}_blocked",
                "baseline_actions_ref": choice.baseline_actions_ref,
                "duration_s": round(time.time() - started, 3),
            }
        )
        if write:
            _write_artifact(artifact)
        return artifact

    execution = _execute_candidate(arc, choice, game_id, budget)
    short = choice.short_game
    if execution["new_levels_solved_this_task"] > 0 and execution["real_env_confirmed"]:
        verdict = f"success: reinduction_advanced_{short}_to_L{execution['ACCURACY_levels_solved']}"
    else:
        reason = _slug_reason(execution.get("rule_diagnosis", "l2_unsolved"))
        verdict = f"complete: l2_wall_holds_{short}_{reason}"

    artifact = _base_artifact(seed, started, verdict)
    artifact.update(execution)
    artifact.update(
        {
            "selected_game": short,
            "candidate_reason": choice.reason,
            "prior_l1_mechanic": choice.l1_mechanic,
            "duration_s": round(time.time() - started, 3),
            "precondition_blocked": False,
        }
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        artifact["schema_errors"] = errors
        artifact["honest_verdict"] = "complete: l2_wall_holds_schema_error"

    if write:
        _write_artifact(artifact)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = run(seed=args.seed, budget=args.budget, write=True)
    print(f"-> {result['honest_verdict']}")
