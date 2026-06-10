"""Exp 4003: scale verifier-validated ARC-AGI-3 level frontiers.

Spec refs: REQ-PHASE4-023, SCENARIO-PHASE4-023.
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
RESULT_NAME = "experiment_4003_scale_level_frontier.json"
RANDOM_SEED = 4003
INFERENCE_SUBSTRATE = "offline_arc_agi3_gap4_executed_consistency_verifier_validated_frontier_scaling"

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import compute_grid_delta, grid_of  # noqa: E402
from carnot.agentic.arc_scale_level_frontier import (  # noqa: E402
    BANKED_FRONTIER,
    GameFrontierResult,
    artifact_schema_errors,
    build_frontier_artifact,
    count_validated_rules,
)
from carnot.agentic.arc_verifier_validated_reinduction import executed_consistency_energy  # noqa: E402
from experiment_3964_r11l_incremental_l2 import (  # noqa: E402
    _attempt_current_level as _attempt_r11l_level,
    _levels_completed,
    _perceive_and_match as _perceive_r11l,
)
from experiment_3965_lp85_incremental_l2 import (  # noqa: E402
    load_baseline_actions as _load_lp85_baseline_actions,
    solve_incremental_levels as _solve_lp85_incremental,
)
from experiment_3992_incremental_levels_verifier_validated import _execute_validated_level  # noqa: E402


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_artifact(seed: int, started: float, verdict: str) -> dict[str, Any]:
    return {
        "experiment": "experiment_4003_scale_level_frontier",
        "title": "arc3_verifier_validated_frontier_scaling",
        "ACCURACY_total_levels_solved": 0,
        "new_levels_this_task": 0,
        "per_game_max_level": {},
        "verifier_validated_count": 0,
        "actions_saved_vs_openloop": 0,
        "per_level_actions": {},
        "baseline_actions_ref": {},
        "real_env_confirmed": False,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 3) if started else 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "banked_frontier": dict(BANKED_FRONTIER),
        "per_game_new_levels": {},
        "first_fail_level": {},
        "stall_reasons": {},
        "level_summaries": {},
        "solve_log": {},
        "candidate_validations": {},
        "precondition_blocked": verdict.startswith("blocked_"),
        "real_env_confirmed_source": "levels_completed",
    }


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


def _with_selected(rows: list[dict[str, Any]], selected: str | None) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["selected"] = bool(selected and item.get("candidate_id") == selected)
        marked.append(item)
    return marked


def _saved_actions(rows: list[dict[str, Any]], selected: str | None) -> int:
    del selected
    return sum(
        int(row.get("planned_l2_actions", 0) or 0)
        for row in rows
        if row.get("selected") is not True
    )


def _execute_r11l_frontier(arc: Any, budget: int) -> GameFrontierResult:
    from arcengine.enums import GameAction

    game_id = _select_game_id(arc, "r11l")
    env = arc.make(game_id)
    frame = env.reset()
    total_actions = 0
    per_level_actions: list[int] = []
    baseline_actions_ref: list[int] = []
    level_summaries: list[dict[str, Any]] = []
    solve_log: list[dict[str, Any]] = []
    candidate_validations: list[dict[str, Any]] = []
    first_fail_level: int | None = None
    stall_reason = ""

    pairs = _perceive_r11l(env)
    frame, used, entries = _attempt_r11l_level(env, GameAction, pairs, budget, 1)
    total_actions += int(used)
    levels_completed = _levels_completed(frame, env)
    solve_log.extend(entries)
    level_summaries.append(
        {
            "level": 1,
            "n_pairs": len(pairs),
            "n_targets": len({tuple(pair["target"]) for pair in pairs}),
            "actions_used": int(used),
            "levels_completed_after": int(levels_completed),
        }
    )
    if levels_completed >= 1:
        per_level_actions.append(int(used))
        baseline_actions_ref.append(len(pairs) * 2)
    else:
        first_fail_level = 1
        stall_reason = "banked L1 replay failed"

    scoped_stop_level = BANKED_FRONTIER["r11l"] + 2
    while first_fail_level is None and levels_completed < scoped_stop_level and total_actions < budget:
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
            first_fail_level = int(level_number)
            stall_reason = "no perceivable r11l piece-target pairs"
            break

        result = _execute_validated_level(
            env,
            GameAction,
            level_number=int(level_number),
            budget_left=max(0, budget - total_actions),
        )
        marked = _with_selected(result["candidate_validations"], result.get("selected_candidate"))
        candidate_validations.extend(marked)
        if not result["advanced"]:
            first_fail_level = int(level_number)
            stall_reason = str(result["reason"])
            break

        total_actions += int(result["actions_used"])
        levels_completed = int(result["levels_completed_after"])
        per_level_actions.append(int(result["actions_used"]))
        baseline_actions_ref.append(len(pairs) * 2)
        solve_log.extend(result["solve_log"])

    if first_fail_level is None and levels_completed < scoped_stop_level:
        first_fail_level = int(levels_completed + 1)
        stall_reason = "r11l budget exhausted before next scoped frontier"

    return GameFrontierResult(
        short_game="r11l",
        game_id=game_id,
        banked_level=BANKED_FRONTIER["r11l"],
        levels_completed=int(levels_completed),
        first_fail_level=first_fail_level,
        per_level_actions=per_level_actions,
        baseline_actions_ref=baseline_actions_ref,
        verifier_validated_count=count_validated_rules(candidate_validations),
        actions_saved_vs_openloop=_saved_actions(candidate_validations, None),
        real_env_confirmed=bool(levels_completed >= min(BANKED_FRONTIER["r11l"], 1)),
        stall_reason=stall_reason or "scoped r11l frontier reached",
        level_summaries=level_summaries,
        solve_log=solve_log,
        candidate_validations=candidate_validations,
    )


def _apply_clicks(env: Any, game_action: Any, clicks: list[tuple[int, int]]) -> tuple[Any, int]:
    frame = None
    for y, x in clicks:
        frame = env.step(game_action.ACTION6, data={"x": int(x), "y": int(y)})
    return frame, len(clicks)


def _validate_clicks(
    env: Any,
    game_action: Any,
    *,
    candidate_id: str,
    level_number: int,
    clicks: list[tuple[int, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    before = _levels_completed(None, env)
    validate_env = copy.deepcopy(env)
    frame, planned_actions = _apply_clicks(validate_env, game_action, clicks)
    after = _levels_completed(frame, validate_env)
    expected = [{"levels_completed_after": before + 1}]
    observed = [{"levels_completed_after": after}]
    energy = executed_consistency_energy(expected, observed)
    selected = after > before and energy == 0.0
    log = [{"level": int(level_number), "action": "click", "y": int(y), "x": int(x)} for y, x in clicks]
    row = {
        "candidate_id": candidate_id,
        "rule_name": "bounded copied-env click-sequence verifier",
        "level": int(level_number),
        "demo_fit": 1.0,
        "heldout_energy": energy,
        "heldout_n": 1,
        "predicted_levels_after": before + 1,
        "validated_levels_after": after,
        "planned_l2_actions": int(planned_actions),
        "selected": bool(selected),
        "expected_heldout": expected,
        "observed_heldout": observed,
    }
    return row, log, int(planned_actions), int(after)


def _coarse_changing_clicks(env: Any, game_action: Any, frame: Any, *, step: int = 4) -> list[tuple[int, int]]:
    original = copy.deepcopy(env._game)
    start_grid = grid_of(frame)
    start_levels = _levels_completed(frame, env)
    clicks: list[tuple[int, int]] = []
    for y in range(0, 64, step):
        for x in range(0, 64, step):
            env._game = copy.deepcopy(original)
            next_frame = env.step(game_action.ACTION6, data={"x": int(x), "y": int(y)})
            delta = compute_grid_delta(start_grid, grid_of(next_frame))
            if int(delta.get("n_changed", 0) or 0) > 0 or _levels_completed(next_frame, env) > start_levels:
                clicks.append((int(y), int(x)))
    env._game = original
    return clicks


def _execute_lp85_frontier(arc: Any, budget: int) -> GameFrontierResult:
    from arcengine.enums import GameAction

    game_id = _select_game_id(arc, "lp85")
    env = arc.make(game_id)
    baseline_actions = _load_lp85_baseline_actions(game_id)
    result = _solve_lp85_incremental(
        env,
        GameAction,
        budget=budget,
        baseline_actions=baseline_actions,
        stop_after_level=2,
        prior_best_levels=BANKED_FRONTIER["lp85"],
        max_plan_depth=20,
    )
    levels_completed = int(result.levels_completed)
    candidate_validations: list[dict[str, Any]] = []
    stall_reason = "lp85 visible-button L1 mechanic exposed no validated L2 level-up"
    first_fail_level = result.first_fail_level
    solve_log = list(result.solve_log)
    per_level_actions = list(result.per_level_actions)
    baseline_ref = list(result.baseline_actions_ref)

    if levels_completed == BANKED_FRONTIER["lp85"]:
        row, _, _, _ = _validate_clicks(
            env,
            GameAction,
            candidate_id="lp85-bounded-l2-click-probe",
            level_number=2,
            clicks=[(16, 20)],
        )
        candidate_validations.append(row)

    return GameFrontierResult(
        short_game="lp85",
        game_id=game_id,
        banked_level=BANKED_FRONTIER["lp85"],
        levels_completed=levels_completed,
        first_fail_level=first_fail_level,
        per_level_actions=per_level_actions,
        baseline_actions_ref=baseline_ref,
        verifier_validated_count=count_validated_rules(candidate_validations),
        actions_saved_vs_openloop=_saved_actions(candidate_validations, None),
        real_env_confirmed=bool(levels_completed >= BANKED_FRONTIER["lp85"]),
        stall_reason=stall_reason if levels_completed == BANKED_FRONTIER["lp85"] else "lp85 L2 validated",
        level_summaries=list(result.level_summaries),
        solve_log=solve_log,
        candidate_validations=candidate_validations,
    )


def _load_sc25_log() -> list[dict[str, Any]]:
    path = REPO / "results" / "experiment_3966_third_game_first_solve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("solve_log", []))


def _apply_sc25_step(env: Any, game_action: Any, step: dict[str, Any]) -> Any:
    action = step.get("action")
    if action == "click":
        return env.step(game_action.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
    mapping = {
        "up": game_action.ACTION1,
        "down": game_action.ACTION2,
        "left": game_action.ACTION3,
        "right": game_action.ACTION4,
    }
    return env.step(mapping[str(action)])


def _execute_sc25_frontier(arc: Any, budget: int) -> GameFrontierResult:
    from arcengine.enums import GameAction

    game_id = _select_game_id(arc, "sc25")
    env = arc.make(game_id)
    frame = env.reset()
    banked_log = _load_sc25_log()
    solve_log: list[dict[str, Any]] = []
    used = 0
    for step in banked_log:
        if used >= budget:
            break
        frame = _apply_sc25_step(env, GameAction, step)
        used += 1
        solve_log.append(dict(step))
        if _levels_completed(frame, env) >= BANKED_FRONTIER["sc25"]:
            break

    levels_completed = _levels_completed(frame, env)
    per_level_actions = [used] if levels_completed >= 1 else []
    baseline_ref = [len(banked_log)] if levels_completed >= 1 else []
    first_fail_level: int | None = None
    stall_reason = ""
    candidate_validations: list[dict[str, Any]] = []

    if levels_completed >= BANKED_FRONTIER["sc25"] and used + len(banked_log) <= budget:
        before = levels_completed
        validate_env = copy.deepcopy(env)
        validate_frame = frame
        for step in banked_log:
            validate_frame = _apply_sc25_step(validate_env, GameAction, step)
            if _levels_completed(validate_frame, validate_env) > before:
                break
        after = _levels_completed(validate_frame, validate_env)
        energy = executed_consistency_energy([{"levels_completed_after": before + 1}], [{"levels_completed_after": after}])
        selected = after > before and energy == 0.0
        row = {
            "candidate_id": "sc25-replay-l1-pattern-on-l2",
            "rule_name": "bounded replay of banked pattern-toggle plus navigation",
            "level": 2,
            "demo_fit": 1.0,
            "heldout_energy": energy,
            "heldout_n": 1,
            "predicted_levels_after": before + 1,
            "validated_levels_after": after,
            "planned_l2_actions": len(banked_log),
            "selected": bool(selected),
        }
        candidate_validations.append(row)
        if selected:
            for step in banked_log:
                frame = _apply_sc25_step(env, GameAction, step)
                used += 1
                solve_log.append(dict(step, level=2))
                if _levels_completed(frame, env) > before:
                    break
            levels_completed = _levels_completed(frame, env)
            per_level_actions.append(used - per_level_actions[0])
            baseline_ref.append(len(banked_log))
        else:
            first_fail_level = 2
            stall_reason = "banked sc25 L1 replay did not validate on L2"
    elif levels_completed < BANKED_FRONTIER["sc25"]:
        first_fail_level = 1
        stall_reason = "banked sc25 L1 replay failed"
    else:
        first_fail_level = 2
        stall_reason = "sc25 budget exhausted before L2 validation"

    return GameFrontierResult(
        short_game="sc25",
        game_id=game_id,
        banked_level=BANKED_FRONTIER["sc25"],
        levels_completed=int(levels_completed),
        first_fail_level=first_fail_level,
        per_level_actions=per_level_actions,
        baseline_actions_ref=baseline_ref,
        verifier_validated_count=count_validated_rules(candidate_validations),
        actions_saved_vs_openloop=_saved_actions(candidate_validations, None),
        real_env_confirmed=bool(levels_completed >= BANKED_FRONTIER["sc25"]),
        stall_reason=stall_reason or "sc25 L2 validated",
        level_summaries=[
            {
                "level": 1,
                "actions_used": int(per_level_actions[0]) if per_level_actions else 0,
                "levels_completed_after": min(1, int(levels_completed)),
            },
            {
                "level": 2,
                "candidate": "sc25-replay-l1-pattern-on-l2",
                "levels_completed_after": int(levels_completed),
            },
        ],
        solve_log=solve_log,
        candidate_validations=candidate_validations,
    )


def _run_all_frontiers(arc: Any, budget: int) -> list[GameFrontierResult]:
    per_game_budget = max(20, budget)
    return [
        _execute_r11l_frontier(arc, per_game_budget),
        _execute_lp85_frontier(arc, per_game_budget),
        _execute_sc25_frontier(arc, per_game_budget),
    ]


def run(seed: int = RANDOM_SEED, budget: int = 80, write: bool = True) -> dict[str, Any]:
    started = time.time()
    try:
        arc = _load_offline_arcade()
    except Exception:
        artifact = _base_artifact(seed, started, "blocked_arc_offline_env_unavailable")
        artifact["duration_s"] = round(time.time() - started, 3)
        if write:
            _write_artifact(artifact)
        return artifact

    results = _run_all_frontiers(arc, budget)
    artifact = build_frontier_artifact(
        results,
        seed=seed,
        started=started,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    artifact["precondition_blocked"] = False
    artifact["real_env_confirmed_source"] = "levels_completed"
    errors = artifact_schema_errors(artifact)
    if errors:
        artifact["schema_errors"] = errors
        artifact["honest_verdict"] = "complete: level_frontier_holds_schema_error"

    if write:
        _write_artifact(artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = run(seed=args.seed, budget=args.budget, write=True)
    print(f"-> {result['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
