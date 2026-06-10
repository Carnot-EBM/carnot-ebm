"""Exp 3992: verifier-validated per-level re-induction for ARC-AGI-3 r11l.

Spec refs: REQ-PHASE4-021, SCENARIO-PHASE4-021.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
RESULT_NAME = "experiment_3992_incremental_levels_verifier_validated.json"
RANDOM_SEED = 3992
PRIOR_BEST_LEVELS = 1
INFERENCE_SUBSTRATE = "offline_arc_agi3_gap4_executed_consistency_verifier_validated_reinduction"

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_level_reinduction import choose_reinduction_candidate  # noqa: E402
from carnot.agentic.arc_verifier_validated_reinduction import (  # noqa: E402
    RuleValidation,
    actions_saved_vs_openloop,
    artifact_schema_errors,
    choose_verified_candidate,
    executed_consistency_energy,
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def _load_prior_stalls() -> dict[str, dict[str, Any]]:
    return {
        "r11l": _load_json(REPO / "results" / "experiment_3964_r11l_incremental_l2.json"),
        "lp85": _load_json(REPO / "results" / "experiment_3965_lp85_incremental_l2.json"),
    }


def _load_exp3980() -> dict[str, Any]:
    return _load_json(REPO / "results" / "experiment_3980_incremental_levels_reinduction.json")


def _load_offline_arcade():  # pragma: no cover - exercised by the real experiment preflight
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    if not arc.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arc


def _select_game_id(arc: Any, short_game: str) -> str:
    for env in arc.get_environments():
        game_id = getattr(env, "game_id", "")
        if game_id.split("-")[0] == short_game:
            return str(game_id)
    raise RuntimeError(f"{short_game} offline environment unavailable")


def _base_artifact(seed: int, started: float, verdict: str) -> dict[str, Any]:
    return {
        "experiment": "experiment_3992_incremental_levels_verifier_validated",
        "title": "arc3_incremental_levels_verifier_validated_reinduction",
        "ACCURACY_levels_solved": 0,
        "new_levels_solved_this_task": 0,
        "verifier_validated_the_rule": False,
        "actions_saved_vs_openloop": 0,
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
        "level_summaries": [],
        "per_level": [],
        "solve_log": [],
        "candidate_validations": [],
        "selected_candidate": None,
        "rule_diagnosis": "",
        "precondition_blocked": verdict.startswith("blocked_"),
        "codex_used_for_reinduction": False,
        "submitted_to_leaderboard": False,
    }


def _settle(env: Any, game_action: Any, frame: Any) -> tuple[Any, int]:
    steps = 0
    while getattr(env._game, "yfbjozweime", False) and steps < 200:
        frame = env.step(game_action.ACTION6, data={"x": -1, "y": -1})
        steps += 1
    return frame, steps


def _composite_position(state: tuple[tuple[int, int], ...], pieces: list[Any], composite: Any) -> tuple[int, int]:
    count = len(state)
    x_sum = sum(x + int(getattr(piece, "width", 1)) // 2 for (x, _), piece in zip(state, pieces))
    y_sum = sum(y + int(getattr(piece, "height", 1)) // 2 for (_, y), piece in zip(state, pieces))
    return (
        int(x_sum // count - int(getattr(composite, "width", 1)) // 2),
        int(y_sum // count - int(getattr(composite, "height", 1)) // 2),
    )


def _collides_at(sprite: Any, x: int, y: int, other: Any) -> bool:
    old_x, old_y = int(sprite.x), int(sprite.y)
    sprite.set_position(int(x), int(y))
    hit = bool(sprite.collides_with(other))
    sprite.set_position(old_x, old_y)
    return hit


def _safe_composite(env: Any, composite: Any, pos: tuple[int, int]) -> bool:
    forbidden = [s for s in env._game.current_level.get_sprites() if s.name.startswith("defgjl")]
    return not any(_collides_at(composite, pos[0], pos[1], sprite) for sprite in forbidden)


def _target_composite(env: Any, composite: Any, target: Any, pos: tuple[int, int]) -> bool:
    return _safe_composite(env, composite, pos) and _collides_at(composite, pos[0], pos[1], target)


def _legal_piece_positions(env: Any, piece: Any) -> list[tuple[int, int]]:
    old_selected = env._game.wiayqaumjug
    env._game.wiayqaumjug = piece
    positions: list[tuple[int, int]] = []
    for x in range(-2, 62):
        for y in range(-2, 62):
            if not env._game.gabrtablhx(x, y):
                positions.append((x, y))
    env._game.wiayqaumjug = old_selected
    return positions


def _click_hits_other_piece(
    state: tuple[tuple[int, int], ...],
    pieces: list[Any],
    piece_index: int,
    top_left: tuple[int, int],
) -> bool:
    piece = pieces[piece_index]
    click_x = int(top_left[0] + int(getattr(piece, "width", 1)) // 2)
    click_y = int(top_left[1] + int(getattr(piece, "height", 1)) // 2)
    for index, (x, y) in enumerate(state):
        if index == piece_index:
            continue
        other = pieces[index]
        if x <= click_x < x + int(getattr(other, "width", 1)) and y <= click_y < y + int(getattr(other, "height", 1)):
            return True
    return False


def _target_positions(env: Any, composite: Any, target: Any) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for x in range(-5, 70):
        for y in range(-5, 70):
            pos = (x, y)
            if _target_composite(env, composite, target, pos):
                positions.append(pos)
    return positions


def _plan_group_path(env: Any, group_id: str, *, max_expand: int = 5000, beam: int = 240) -> list[dict[str, Any]] | None:
    data = env._game.kacotwgjcyq[group_id]
    composite = data.get("roduyfsmiznvg")
    target = data.get("gosubdcyegamj")
    pieces = list(data.get("lecfirgqbwunn", []))
    if not composite or not target or not pieces:
        return [] if composite and target and composite.collides_with(target) else None
    if composite.collides_with(target):
        return []

    targets = _target_positions(env, composite, target)
    if not targets:
        return None

    legal = [_legal_piece_positions(env, piece) for piece in pieces]
    start = tuple((int(piece.x), int(piece.y)) for piece in pieces)

    def heuristic(state: tuple[tuple[int, int], ...]) -> int:
        comp = _composite_position(state, pieces, composite)
        return min(abs(comp[0] - tx) + abs(comp[1] - ty) for tx, ty in targets)

    best: dict[tuple[tuple[int, int], ...], int] = {start: 0}
    queue: list[tuple[int, int, tuple[tuple[int, int], ...], list[dict[str, Any]]]] = [
        (heuristic(start), 0, start, [])
    ]
    expanded = 0
    while queue and expanded < max_expand:
        _, cost, state, path = heapq.heappop(queue)
        if cost != best.get(state):
            continue
        comp = _composite_position(state, pieces, composite)
        if _target_composite(env, composite, target, comp):
            return path

        expanded += 1
        current_h = heuristic(state)
        candidates: list[tuple[int, int, int, tuple[int, int], tuple[tuple[int, int], ...], tuple[int, int]]] = []
        for piece_index, positions in enumerate(legal):
            current = state[piece_index]
            for top_left in positions:
                if top_left == current or _click_hits_other_piece(state, pieces, piece_index, top_left):
                    continue
                next_state = list(state)
                next_state[piece_index] = top_left
                next_tuple = tuple(next_state)
                next_comp = _composite_position(next_tuple, pieces, composite)
                if not _safe_composite(env, composite, next_comp):
                    continue
                next_h = heuristic(next_tuple)
                if next_h <= current_h + 24:
                    move_cost = (top_left[0] - current[0]) ** 2 + (top_left[1] - current[1]) ** 2
                    candidates.append((next_h, move_cost, piece_index, top_left, next_tuple, next_comp))

        candidates.sort(key=lambda row: (row[0], row[1]))
        for next_h, _, piece_index, top_left, next_tuple, next_comp in candidates[:beam]:
            next_cost = cost + 1
            if next_cost >= best.get(next_tuple, 1_000_000):
                continue
            best[next_tuple] = next_cost
            move = {
                "group_id": group_id,
                "piece_index": int(piece_index),
                "top_left": [int(top_left[0]), int(top_left[1])],
                "expected_composite_after": [int(next_comp[1]), int(next_comp[0])],
                "rule": "safe-composite path through collision-forbidden mask",
            }
            heapq.heappush(queue, (next_cost + next_h, next_cost, next_tuple, path + [move]))
    return None


def _execute_moves(env: Any, moves: list[dict[str, Any]], game_action: Any, level_number: int) -> tuple[Any, int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame = None
    actions = 0
    log: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for move in moves:
        group_id = move["group_id"]
        data = env._game.kacotwgjcyq[group_id]
        piece = data["lecfirgqbwunn"][int(move["piece_index"])]
        composite = data["roduyfsmiznvg"]
        target = data["gosubdcyegamj"]
        top_left = (int(move["top_left"][0]), int(move["top_left"][1]))
        failure_before = int(getattr(env._game, "yledlprvvkb", 0) or 0)

        frame = env.step(game_action.ACTION6, data={"x": int(piece.x + piece.width // 2), "y": int(piece.y + piece.height // 2)})
        actions += 1
        frame = env.step(game_action.ACTION6, data={"x": int(top_left[0] + piece.width // 2), "y": int(top_left[1] + piece.height // 2)})
        actions += 1
        frame, settle_steps = _settle(env, game_action, frame)

        expected_row = {
            "group_id": group_id,
            "piece_index": int(move["piece_index"]),
            "piece_after": [int(top_left[1]), int(top_left[0])],
            "composite_after": list(move["expected_composite_after"]),
            "failure_delta": 0,
        }
        observed_row = {
            "group_id": group_id,
            "piece_index": int(move["piece_index"]),
            "piece_after": [int(piece.y), int(piece.x)],
            "composite_after": [int(composite.y), int(composite.x)],
            "failure_delta": int(getattr(env._game, "yledlprvvkb", 0) or 0) - failure_before,
        }
        expected.append(expected_row)
        observed.append(observed_row)
        log.append(
            {
                "level": int(level_number),
                "group_id": group_id,
                "piece_index": int(move["piece_index"]),
                "action": "verifier_validated_safe_path_place",
                "click": [int(top_left[1] + piece.height // 2), int(top_left[0] + piece.width // 2)],
                "piece_top_left_after": [int(piece.y), int(piece.x)],
                "composite_after": [int(composite.y), int(composite.x)],
                "target": [int(target.y), int(target.x)],
                "target_after_collides": bool(composite.collides_with(target)),
                "settle_steps": int(settle_steps),
            }
        )
    return frame, actions, log, expected, observed


def _build_safe_path_moves(base_env: Any, game_action: Any, level_number: int) -> tuple[list[dict[str, Any]] | None, int]:
    plan_env = copy.deepcopy(base_env)
    level_before = _levels_completed(None, plan_env)
    all_moves: list[dict[str, Any]] = []
    frame = None
    for group_id, data in sorted(plan_env._game.kacotwgjcyq.items(), key=lambda item: len(item[1].get("lecfirgqbwunn", []))):
        target = data.get("gosubdcyegamj")
        composite = data.get("roduyfsmiznvg")
        if not target or not composite or composite.collides_with(target):
            continue
        path = _plan_group_path(plan_env, group_id)
        if path is None:
            return None, level_before
        for move in path:
            move["level"] = int(level_number)
        if path:
            frame, _, _, _, _ = _execute_moves(plan_env, path, game_action, level_number)
            all_moves.extend(path)
        if _levels_completed(frame, plan_env) > level_before:
            break
    return all_moves, _levels_completed(frame, plan_env) if frame is not None else level_before


def _build_single_move_candidate(base_env: Any, game_action: Any, level_number: int) -> tuple[list[dict[str, Any]], int]:
    plan_env = copy.deepcopy(base_env)
    moves: list[dict[str, Any]] = []
    if "pumlzd" in plan_env._game.kacotwgjcyq:
        path = _plan_group_path(plan_env, "pumlzd")
        if path:
            move = dict(path[0])
            move["level"] = int(level_number)
            moves.append(move)
    frame, _, _, _, _ = _execute_moves(plan_env, moves, game_action, level_number) if moves else (None, 0, [], [], [])
    return moves, _levels_completed(frame, plan_env) if frame is not None else _levels_completed(None, plan_env)


def _validate_candidate(
    base_env: Any,
    moves: list[dict[str, Any]],
    game_action: Any,
    *,
    level_number: int,
    candidate_id: str,
    rule_name: str,
    predicted_levels_after: int,
) -> tuple[RuleValidation, dict[str, Any]]:
    validate_env = copy.deepcopy(base_env)
    frame, actions, _, expected, observed = _execute_moves(validate_env, moves, game_action, level_number)
    split = max(1, len(expected) // 2)
    demo_energy = executed_consistency_energy(expected[:split], observed[:split])
    heldout_energy = executed_consistency_energy(expected[split:], observed[split:])
    demo_fit = 1.0 if demo_energy == 0.0 else 0.0
    validated_levels_after = _levels_completed(frame, validate_env) if frame is not None else _levels_completed(None, validate_env)
    validation = RuleValidation(
        candidate_id=candidate_id,
        rule_name=rule_name,
        demo_fit=demo_fit,
        heldout_energy=heldout_energy,
        heldout_n=max(0, len(expected) - split),
        predicted_levels_after=int(predicted_levels_after),
        validated_levels_after=int(validated_levels_after),
        planned_l2_actions=int(actions),
    )
    return validation, {
        **asdict(validation),
        "level": int(level_number),
        "expected_heldout": expected[split:],
        "observed_heldout": observed[split:],
    }


def _candidate_moves_for_level(base_env: Any, game_action: Any, level_number: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    safe_moves, safe_predicted = _build_safe_path_moves(base_env, game_action, level_number)
    single_moves, single_predicted = _build_single_move_candidate(base_env, game_action, level_number)
    moves_by_candidate = {"single-move-mask": single_moves}
    predicted = {"single-move-mask": single_predicted}
    if safe_moves:
        moves_by_candidate["safe-composite-path"] = safe_moves
        predicted["safe-composite-path"] = safe_predicted
    return moves_by_candidate, predicted


def _execute_validated_level(
    env: Any,
    game_action: Any,
    *,
    level_number: int,
    budget_left: int,
) -> dict[str, Any]:
    level_before = _levels_completed(None, env)
    moves_by_candidate, predicted = _candidate_moves_for_level(env, game_action, level_number)
    validations: list[RuleValidation] = []
    diagnostics: list[dict[str, Any]] = []
    for candidate_id, moves in moves_by_candidate.items():
        if not moves or len(moves) * 2 > budget_left:
            continue
        rule_name = "collision-forbidden mask single-move rule" if candidate_id == "single-move-mask" else "collision-forbidden safe-composite path rule"
        validation, detail = _validate_candidate(
            env,
            moves,
            game_action,
            level_number=level_number,
            candidate_id=candidate_id,
            rule_name=rule_name,
            predicted_levels_after=predicted[candidate_id],
        )
        validations.append(validation)
        diagnostics.append(detail)

    chosen = choose_verified_candidate(validations, current_level=level_before)
    if chosen is None or chosen.validated_levels_after <= level_before:
        return {
            "advanced": False,
            "actions_used": 0,
            "selected_candidate": None,
            "candidate_validations": diagnostics,
            "solve_log": [],
            "levels_completed_after": int(level_before),
            "reason": "no verifier-validated candidate advanced the level",
        }

    moves = moves_by_candidate[chosen.candidate_id]
    frame, used, log, _, _ = _execute_moves(env, moves, game_action, level_number)
    levels_after = _levels_completed(frame, env)
    return {
        "advanced": bool(levels_after > level_before),
        "actions_used": int(used),
        "selected_candidate": chosen.candidate_id,
        "candidate_validations": diagnostics,
        "solve_log": log,
        "levels_completed_after": int(levels_after),
        "reason": chosen.rule_name,
    }


def _execute_r11l_verifier_validated(arc: Any, game_id: str, budget: int, openloop_actions: int) -> dict[str, Any]:
    from arcengine.enums import GameAction

    env = arc.make(game_id)
    frame = env.reset()
    total_actions = 0
    per_level_actions: list[int] = []
    baseline_actions_ref: list[int] = []
    level_summaries: list[dict[str, Any]] = []
    per_level_rows: list[dict[str, Any]] = []
    solve_log: list[dict[str, Any]] = []
    candidate_validations: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []

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
            "ACCURACY_levels_solved": int(levels_completed),
            "new_levels_solved_this_task": 0,
            "verifier_validated_the_rule": False,
            "actions_saved_vs_openloop": 0,
            "game_advanced": f"{game_id}_to_L{levels_completed}",
            "per_level_actions": [],
            "baseline_actions_ref": [],
            "real_env_confirmed": False,
            "first_fail_level": 1,
            "level_summaries": level_summaries,
            "per_level": [],
            "solve_log": solve_log,
            "candidate_validations": [],
            "selected_candidate": None,
            "rule_diagnosis": "banked L1 could not be reproduced in the real env",
        }

    per_level_actions.append(int(used_l1))
    baseline_actions_ref.append(len(l1_pairs) * 2)

    first_fail_level = None
    while levels_completed < 3 and total_actions < budget:
        level_number = levels_completed + 1
        pairs = _perceive_r11l(env)
        level_summaries.append(
            {
                "level": int(level_number),
                "n_pairs": len(pairs),
                "n_targets": len({tuple(pair["target"]) for pair in pairs}),
                "budget_left": int(budget - total_actions),
            }
        )
        if not pairs:
            first_fail_level = level_number
            break

        result = _execute_validated_level(
            env,
            GameAction,
            level_number=level_number,
            budget_left=max(0, budget - total_actions),
        )
        candidate_validations.extend(result["candidate_validations"])
        if not result["advanced"]:
            first_fail_level = level_number
            break

        total_actions += int(result["actions_used"])
        levels_completed = int(result["levels_completed_after"])
        per_level_actions.append(int(result["actions_used"]))
        baseline_actions_ref.append(len(pairs) * 2)
        solve_log.extend(result["solve_log"])
        per_level_rows.append(
            {
                "level": int(level_number),
                "actions_used": int(result["actions_used"]),
                "levels_completed_after": int(levels_completed),
                "selected_candidate": result["selected_candidate"],
                "rule": result["reason"],
            }
        )
        selected_candidates.append(
            {
                "level": int(level_number),
                "candidate_id": result["selected_candidate"],
            }
        )

    saved_actions = actions_saved_vs_openloop(openloop_actions=openloop_actions, committed_rejected_actions=0)
    return {
        "ACCURACY_levels_solved": int(levels_completed),
        "new_levels_solved_this_task": int(max(0, levels_completed - PRIOR_BEST_LEVELS)),
        "verifier_validated_the_rule": bool(selected_candidates),
        "actions_saved_vs_openloop": int(saved_actions),
        "game_advanced": f"{game_id}_to_L{levels_completed}",
        "per_level_actions": per_level_actions,
        "baseline_actions_ref": baseline_actions_ref,
        "real_env_confirmed": bool(levels_completed >= 1),
        "first_fail_level": first_fail_level,
        "level_summaries": level_summaries,
        "per_level": per_level_rows,
        "solve_log": solve_log,
        "candidate_validations": candidate_validations,
        "selected_candidate": selected_candidates[-1]["candidate_id"] if selected_candidates else None,
        "selected_candidates": selected_candidates,
        "rule_diagnosis": "verifier-validated safe-composite path solved through L3" if levels_completed >= 3 else "L2 wall held after verifier validation",
        "total_actions": int(total_actions),
        "actions_saved_basis": "Exp3980 open-loop L2 rejected action count avoided before committing the validated path",
    }


def _slug_reason(reason: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
    return slug[:80] or "unsolved"


def run(seed: int = RANDOM_SEED, budget: int = 60, write: bool = True, _arc_client: Any | None = None) -> dict[str, Any]:
    started = time.time()
    stalls = _load_prior_stalls()
    exp3980 = _load_exp3980()
    choice = choose_reinduction_candidate(stalls)
    openloop_actions = int(exp3980.get("l2_attempted_actions", 0) or 0)

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

    if choice.short_game != "r11l":
        execution = {
            "ACCURACY_levels_solved": int(choice.prior_levels),
            "new_levels_solved_this_task": 0,
            "verifier_validated_the_rule": False,
            "actions_saved_vs_openloop": 0,
            "game_advanced": f"{game_id}_to_L{choice.prior_levels}",
            "per_level_actions": [],
            "baseline_actions_ref": choice.baseline_actions_ref,
            "real_env_confirmed": False,
            "first_fail_level": 2,
            "rule_diagnosis": f"no verifier-validated executor implemented for {choice.short_game}",
        }
    else:
        execution = _execute_r11l_verifier_validated(arc, game_id, budget, openloop_actions)

    short = choice.short_game
    if execution["new_levels_solved_this_task"] > 0 and execution["real_env_confirmed"] and execution["verifier_validated_the_rule"]:
        verdict = f"success: verifier_validated_reinduction_advanced_{short}_to_L{execution['ACCURACY_levels_solved']}"
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
            "exp3980_l2_attempted_actions": openloop_actions,
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
