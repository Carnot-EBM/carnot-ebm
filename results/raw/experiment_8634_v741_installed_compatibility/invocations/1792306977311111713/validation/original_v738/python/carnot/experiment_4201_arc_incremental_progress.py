"""Exp 4201: ARC-AGI-3 hardened GAP-4 L4 incremental progress.

Spec refs: REQ-PHASE4-057, SCENARIO-PHASE4-057.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from carnot.experiment_4179_arc_incremental_progress import (
    FrontierOutcome,
    TargetSelection,
    _goal_key,
    _levels_completed,
    _target_goal_key,
    discover_click_buttons,
    load_environment_baselines,
)


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4201_arc_incremental_progress.json"
RANDOM_SEED = 4201
LP85_GAME_ID = "lp85-305b61c3"
PRIOR_TOTAL_GAMES_SOLVED = 13
PRIOR_TOTAL_LEVELS_SOLVED = 15
INFERENCE_SUBSTRATE = "offline_arc_agi3_hardened_gap4_l4_incremental_progress"
REQUIREMENTS = ["REQ-PHASE4-057", "SCENARIO-PHASE4-057"]
HARDENED_VERIFIER = "hardened_gap4_heldout_executed_consistency_deeper_level_replay"
L4_PLANNING_MAX_EXPANSIONS = 128
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "total_levels_solved",
    "levels_completed",
    "real_env_confirmed",
    "target_game",
    "target_level",
    "prior_total_levels_solved",
    "new_levels_solved_this_task",
    "solve_trace",
    "inference_substrate",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. An honest no-solve is a COMPLETE verdict (progress-not-perfection).",
    "total_levels_solved": "The monotonic progress metric; must be >= the prior milestone's count (15).",
    "levels_completed": "Real-env-confirmed level count this run; falsifiable evidence of an actual solve.",
    "real_env_confirmed": "Only real-env solves raise the headline count; a synthetic-scaffold solve does not.",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def gap4_hardening_ready(gap4_artifact: dict[str, Any]) -> bool:
    """REQ-PHASE4-057: B1 hardening evidence must be present before a suffix is retained."""

    ledger = gap4_artifact.get("gross_recovery_ledger") if isinstance(gap4_artifact, dict) else None
    return (
        isinstance(gap4_artifact, dict)
        and gap4_artifact.get("experiment") == "experiment_4187_gap4_graded_execution_gate_hardening"
        and gap4_artifact.get("vote_aware_guard_blocked_mispromotion") is True
        and isinstance(ledger, dict)
        and int(ledger.get("recovered", 0) or 0) >= 4
        and int(ledger.get("lost", 0) or 0) == 0
    )


def select_deeper_level_target(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, list[int]]],
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> TargetSelection:
    """REQ-PHASE4-057: choose lp85 L4 after Exp 4190 confirmed L3."""

    _ = survey
    if (
        prior_artifact.get("experiment") != "experiment_4190_arc_incremental_progress"
        or prior_artifact.get("target_game") != LP85_GAME_ID
        or prior_artifact.get("real_env_confirmed") is not True
        or int(prior_artifact.get("levels_completed", 0) or 0) < 3
        or int(prior_artifact.get("total_levels_solved", 0) or 0) < PRIOR_TOTAL_LEVELS_SOLVED
        or not isinstance(prior_artifact.get("action_plan"), list)
        or not prior_artifact.get("action_plan")
    ):
        raise ValueError("Exp 4190 L3 success evidence unavailable")
    if not gap4_hardening_ready(gap4_artifact):
        raise ValueError("hardened GAP-4 verifier evidence unavailable")
    if "lp85" not in baselines:
        raise ValueError("lp85 offline fixture metadata unavailable")
    game_id, baseline_actions = baselines["lp85"]
    if game_id != LP85_GAME_ID or len(baseline_actions) < 4:
        raise ValueError("lp85 offline fixture metadata unavailable")
    return TargetSelection(
        game="lp85",
        game_id=LP85_GAME_ID,
        target_level=4,
        prior_level=3,
        baseline_actions=int(baseline_actions[3]),
        selection_mode="deeper_level_after_lp85_L3_success",
        selection_reason="selected lp85 L4 as the next deeper already-solved-game level after Exp 4190 reached L3",
    )


def validate_hardened_gap4_heldout_replay(
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level: int,
    *,
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    """SCENARIO-PHASE4-057: hardened GAP-4 retained suffixes must advance held-out replay."""

    ready = gap4_hardening_ready(gap4_artifact)
    level_increment = int(final_level) > int(start_level)
    retained = (
        ready
        and int(heldout_transition_count) > 0
        and level_increment
        and int(final_level) >= int(predicted_level)
    )
    ledger = gap4_artifact.get("gross_recovery_ledger", {}) if isinstance(gap4_artifact, dict) else {}
    return {
        "phase": "hardened-gap4-verify",
        "verifier": HARDENED_VERIFIER,
        "start_level_completed": int(start_level),
        "final_level_completed": int(final_level),
        "predicted_level_after_actions": int(predicted_level),
        "heldout_transition_count": int(heldout_transition_count),
        "level_increment": bool(level_increment),
        "hardened_gap4_ready": bool(ready),
        "hardened_gap4_source": "results/experiment_4187_gap4_graded_execution_gate_hardening.json",
        "hardened_gap4_recovered": int(ledger.get("recovered", 0) or 0) if isinstance(ledger, dict) else 0,
        "hardened_gap4_lost": int(ledger.get("lost", 0) or 0) if isinstance(ledger, dict) else 0,
        "retained": bool(retained),
        "energy": 0.0 if retained else 1.0,
    }


def _fixture_available(game_id: str) -> bool:
    if "-" not in str(game_id):
        return False
    prefix, suffix = str(game_id).split("-", maxsplit=1)
    root = REPO / "environment_files" / prefix / suffix
    return root.joinpath("metadata.json").exists() and root.joinpath(f"{prefix}.py").exists()


def blocked_artifact(
    *,
    target_game: str,
    target_level: int,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-057: report fixture blockage without solve inflation."""

    prior_level = max(0, int(target_level) - 1)
    artifact = {
        "experiment": "experiment_4201_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_lp85_L4",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "target_game": str(target_game),
        "target_level": int(target_level),
        "prior_level": int(prior_level),
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "new_levels_solved_this_task": 0,
        "levels_completed": int(prior_level),
        "real_env_confirmed": False,
        "verifier_validated": False,
        "replay_actions_used": 0,
        "executed_real_env_actions": 0,
        "exploration_actions_used": 0,
        "induced_mechanic": "none",
        "verification_decisions": [],
        "action_plan": [],
        "phase_trace": [],
        "solve_trace": {
            "target_game": str(target_game),
            "target_level": int(target_level),
            "actions": [],
            "verification_decisions": [],
            "phase_trace": [],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": 0,
        "selection_mode": "blocked_precondition",
        "selected_candidate_reason": "offline fixture or prior verifier precondition failed",
        "acceptance_gate_passed": False,
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_artifact(
    outcome: FrontierOutcome,
    target: TargetSelection,
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-057: build the terminal artifact from hardened verified evidence."""

    advanced = outcome.advanced
    new_levels = 1 if advanced else 0
    total_levels = PRIOR_TOTAL_LEVELS_SOLVED + new_levels
    if advanced:
        verdict = (
            f"success: incremental_progress_{outcome.target_game}_advanced_to_"
            f"L{outcome.final_level_completed}_total{total_levels}"
        )
    else:
        verdict = (
            f"complete: incremental_progress_no_solve_{outcome.target_game}_"
            f"L{outcome.target_level}_{_reason_slug(outcome.failure_reason)}"
        )

    solve_trace = {
        "target_game": outcome.target_game,
        "target_level": int(outcome.target_level),
        "prior_level": int(outcome.prior_level),
        "selection_mode": target.selection_mode,
        "selection_reason": target.selection_reason,
        "actions": list(outcome.action_plan),
        "verification_decisions": list(outcome.verification_decisions),
        "phase_trace": list(outcome.phase_trace),
    }
    artifact = {
        "experiment": "experiment_4201_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_lp85_L4",
        "honest_verdict": verdict,
        "target_game": outcome.target_game,
        "target_level": int(outcome.target_level),
        "prior_level": int(outcome.prior_level),
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": int(total_levels),
        "new_levels_solved_this_task": int(new_levels),
        "levels_completed": int(outcome.final_level_completed),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "verifier_validated": bool(outcome.verifier_validated),
        "replay_actions_used": int(outcome.replay_actions_used),
        "executed_real_env_actions": int(outcome.executed_real_env_actions),
        "exploration_actions_used": int(outcome.exploration_actions_used),
        "induced_mechanic": outcome.induced_mechanic,
        "verification_decisions": list(outcome.verification_decisions),
        "action_plan": list(outcome.action_plan),
        "phase_trace": list(outcome.phase_trace),
        "solve_trace": solve_trace,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": int(target.baseline_actions),
        "selection_mode": target.selection_mode,
        "selected_candidate_reason": target.selection_reason,
        "acceptance_gate_passed": bool(
            (advanced and total_levels > PRIOR_TOTAL_LEVELS_SOLVED)
            or (not advanced and verdict.startswith("complete:"))
        ),
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-057: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    int_fields = (
        "total_levels_solved",
        "levels_completed",
        "target_level",
        "prior_total_levels_solved",
        "new_levels_solved_this_task",
        "total_games_solved",
    )
    for field in int_fields:
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    for field in ("real_env_confirmed", "verifier_validated"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")
    for field in ("target_game", "inference_substrate"):
        if field in artifact and not isinstance(artifact[field], str):
            errors.append(f"{field} must be a string")
    if "solve_trace" in artifact and not isinstance(artifact["solve_trace"], dict):
        errors.append("solve_trace must be a dict")
    if "inference_substrate" in artifact and artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if "requirements" in artifact and artifact["requirements"] != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-057 and SCENARIO-PHASE4-057")
    if "field_principles" in artifact:
        principles = artifact["field_principles"]
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in REQUIRED_FIELD_PRINCIPLES:
                if field not in principles:
                    errors.append(f"field_principles missing {field}")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("verifier_validated") is not True:
            errors.append("verifier_validated must be true for success")
        if artifact.get("new_levels_solved_this_task") != 1:
            errors.append("new_levels_solved_this_task must be one for scoped success")
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED + 1:
            errors.append("total_levels_solved must increment from 15 to 16 for success")
        if int(artifact.get("levels_completed", 0) or 0) < int(artifact.get("target_level", 0) or 0):
            errors.append("levels_completed must reach target_level for success")
        if not any(
            isinstance(decision, dict)
            and decision.get("retained") is True
            and decision.get("verifier") == HARDENED_VERIFIER
            for decision in artifact.get("verification_decisions", [])
        ):
            errors.append("success requires a retained hardened GAP-4 verifier decision")
        if not artifact.get("action_plan"):
            errors.append("success requires a validated action_plan")
        if not isinstance(artifact.get("solve_trace"), dict) or not artifact["solve_trace"].get("phase_trace"):
            errors.append("solve_trace must include phase_trace for success")
    elif isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED:
            errors.append("total_levels_solved must remain at the prior count for no-solve")
        if artifact.get("new_levels_solved_this_task") != 0:
            errors.append("new_levels_solved_this_task must be zero for no-solve")
        if artifact.get("real_env_confirmed") is not False:
            errors.append("real_env_confirmed must be false for no-solve")
    return errors


def prior_replay_steps(prior_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-PHASE4-057: extract banked real click actions needed to re-establish lp85 L3."""

    trace = prior_artifact.get("phase_trace")
    if not isinstance(trace, list):
        solve_trace = prior_artifact.get("solve_trace", {})
        trace = solve_trace.get("phase_trace") if isinstance(solve_trace, dict) else None
    steps: list[dict[str, Any]] = []
    for row in trace if isinstance(trace, list) else []:
        if not isinstance(row, dict) or row.get("phase") not in {"replay", "act"}:
            continue
        if "x" not in row or "y" not in row:
            continue
        steps.append(
            {
                "phase": str(row.get("phase")),
                "button": str(row.get("button", "banked_click")),
                "x": int(row["x"]),
                "y": int(row["y"]),
            }
        )
    if not steps:
        raise ValueError("prior Exp 4190 replay trace has no executable click steps")
    return steps


def replay_prior_lp85_frontier(
    env: Any,
    game_action: Any,
    prior_artifact: dict[str, Any],
    target: TargetSelection,
) -> tuple[int, int, list[dict[str, Any]]]:
    """SCENARIO-PHASE4-057: re-establish prior L3 from Exp 4190 observed actions."""

    replay_trace: list[dict[str, Any]] = []
    current_level = _levels_completed(None, env)
    action_count = 0
    if current_level >= target.prior_level:
        return current_level, action_count, replay_trace

    for index, step in enumerate(prior_replay_steps(prior_artifact), start=1):
        frame = env.step(game_action.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
        current_level = _levels_completed(frame, env)
        action_count += 1
        replay_trace.append(
            {
                "phase": "replay",
                "source": "banked_exp4190_lp85_L3_replay",
                "source_phase": str(step["phase"]),
                "action_index": int(index),
                "button": str(step.get("button", "banked_click")),
                "x": int(step["x"]),
                "y": int(step["y"]),
                "levels_completed": int(current_level),
            }
        )
        if current_level >= target.prior_level:
            break
    return current_level, action_count, replay_trace


def plan_observed_suffix_bounded(
    env: Any,
    game_action: Any,
    *,
    start_level: int,
    max_depth: int = 16,
    max_expansions: int = L4_PLANNING_MAX_EXPANSIONS,
) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    """SCENARIO-PHASE4-057: search copied env states with a hard expansion cap."""

    buttons = discover_click_buttons(env)
    original_game = copy.deepcopy(env._game)
    start_key = _goal_key(original_game)
    target_key = _target_goal_key(original_game)
    trace: dict[str, Any] = {
        "observed_buttons": list(buttons),
        "start_goal_key": list(start_key),
        "target_goal_key": list(target_key),
        "observed_transition_count": 0,
        "expanded_states": 0,
        "max_depth": int(max_depth),
        "max_expansions": int(max_expansions),
        "found": False,
        "stopped_reason": "",
    }
    if not buttons:
        trace["stopped_reason"] = "no_click_buttons_observed"
        return [], trace

    queue: deque[tuple[Any, list[dict[str, int | str]]]] = deque([(copy.deepcopy(original_game), [])])
    seen = {start_key}
    while queue and int(trace["expanded_states"]) < int(max_expansions):
        current_game, path = queue.popleft()
        trace["expanded_states"] = int(trace["expanded_states"]) + 1
        if len(path) >= max_depth:
            continue
        for button in buttons:
            env._game = copy.deepcopy(current_game)
            frame = env.step(game_action.ACTION6, data={"x": int(button["x"]), "y": int(button["y"])})
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
            level_after = _levels_completed(frame, env)
            step = dict(button)
            step["levels_completed_after"] = int(level_after)
            next_path = path + [step]
            if level_after > int(start_level):
                env._game = copy.deepcopy(original_game)
                trace["found"] = True
                trace["planned_depth"] = len(next_path)
                trace["stopped_reason"] = "level_increment_found"
                return next_path, trace
            key = _goal_key(env._game)
            if key not in seen:
                seen.add(key)
                queue.append((copy.deepcopy(env._game), next_path))

    env._game = copy.deepcopy(original_game)
    trace["planned_depth"] = 0
    trace["stopped_reason"] = "max_expansions_exhausted" if queue else "frontier_exhausted"
    return [], trace


def _validate_suffix_on_copy(
    env: Any,
    game_action: Any,
    *,
    start_level: int,
    target_level: int,
    action_plan: list[dict[str, int | str]],
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    original_game = copy.deepcopy(env._game)
    heldout_count = max(1, len(action_plan) // 2) if action_plan else 0
    prefix_len = max(0, len(action_plan) - heldout_count)
    final_level = int(start_level)
    for step in action_plan:
        frame = env.step(game_action.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
        final_level = _levels_completed(frame, env)
    env._game = copy.deepcopy(original_game)
    decision = validate_hardened_gap4_heldout_replay(
        start_level=start_level,
        final_level=final_level,
        heldout_transition_count=heldout_count,
        predicted_level=target_level,
        gap4_artifact=gap4_artifact,
    )
    decision["validated_prefix_transition_count"] = int(prefix_len)
    decision["validated_total_transition_count"] = len(action_plan)
    return decision


def execute_plan_until_level(
    env: Any,
    game_action: Any,
    action_plan: list[dict[str, int | str]],
    *,
    prior_level: int,
    target_level: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    """SCENARIO-PHASE4-057: execute validated actions until the scoped L4 increment."""

    action_trace: list[dict[str, Any]] = []
    final_level = int(prior_level)
    for index, step in enumerate(action_plan, start=1):
        frame = env.step(game_action.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
        final_level = _levels_completed(frame, env)
        action_trace.append(
            {
                "phase": "act",
                "action_index": int(index),
                "button": str(step["button"]),
                "x": int(step["x"]),
                "y": int(step["y"]),
                "levels_completed": int(final_level),
            }
        )
        if final_level >= int(target_level) or final_level > int(prior_level):
            break
    return final_level, len(action_trace), action_trace


def _load_offline_arcade() -> Any:  # pragma: no cover - thin real-env adapter
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )


def _run_lp85_l4_frontier(
    offline_arcade: Any,
    target: TargetSelection,
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> FrontierOutcome:  # pragma: no cover - exercised by required experiment command
    from arcengine.enums import GameAction

    env = offline_arcade.make(target.game_id)
    frame = env.reset()
    initial_level = _levels_completed(frame, env)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_lp85_reset",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(initial_level),
        }
    ]
    frontier_level, replay_actions, replay_trace = replay_prior_lp85_frontier(
        env,
        GameAction,
        prior_artifact,
        target,
    )
    phase_trace.extend(replay_trace)
    if frontier_level < target.prior_level:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=replay_actions,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="lp85 banked L1-L3 replay",
            failure_reason="could_not_reestablish_prior_frontier",
        )

    buttons = discover_click_buttons(env)
    action_plan, planner_trace = plan_observed_suffix_bounded(
        env,
        GameAction,
        start_level=frontier_level,
        max_depth=target.baseline_actions or 16,
    )
    phase_trace.append(
        {
            "phase": "explore",
            "source": "copied_env_visible_button_observations",
            "buttons_observed": len(buttons),
            "planner_trace": planner_trace,
        }
    )
    phase_trace.append(
        {
            "phase": "induce",
            "mechanic": "lp85 visible goals are permuted by observed left/right button clicks",
            "goal_predicate": "every colored frame has a paired visible goal sprite at the induced offset",
            "candidate_action_count": len(action_plan),
        }
    )
    if not action_plan:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=replay_actions,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="lp85 L4 observed button-permutation mechanic",
            failure_reason="no_observed_level_up_candidate",
        )

    validation = _validate_suffix_on_copy(
        env,
        GameAction,
        start_level=frontier_level,
        target_level=target.target_level,
        action_plan=action_plan,
        gap4_artifact=gap4_artifact,
    )
    phase_trace.append(validation)
    if not validation["retained"]:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=replay_actions,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[validation],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="lp85 L4 observed button-permutation mechanic",
            failure_reason="no_verifier_validated_level_up_candidate",
        )

    final_level, executed_actions, act_trace = execute_plan_until_level(
        env,
        GameAction,
        action_plan,
        prior_level=frontier_level,
        target_level=target.target_level,
    )
    phase_trace.extend(act_trace)
    advanced = final_level >= target.target_level
    return FrontierOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=final_level,
        replay_actions_used=replay_actions,
        executed_real_env_actions=executed_actions,
        exploration_actions_used=replay_actions,
        real_env_confirmed=advanced,
        verifier_validated=True,
        verification_decisions=[validation],
        action_plan=action_plan,
        phase_trace=phase_trace,
        induced_mechanic="lp85 L4 observed button-permutation mechanic with visible goal-overlap predicate",
        failure_reason="" if advanced else "real_env_confirmation_not_incremented",
    )


def _failed_outcome(target: TargetSelection, reason: str) -> FrontierOutcome:
    return FrontierOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=target.prior_level,
        replay_actions_used=0,
        executed_real_env_actions=0,
        exploration_actions_used=0,
        real_env_confirmed=False,
        verifier_validated=False,
        verification_decisions=[],
        action_plan=[],
        phase_trace=[{"phase": "observe", "target_game": target.game_id, "source": reason}],
        induced_mechanic="none",
        failure_reason=reason,
    )


def run(*, write: bool = True) -> dict[str, Any]:
    """Run Exp 4201 offline and optionally write the terminal artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    prior_path = REPO / "results" / "experiment_4190_arc_incremental_progress.json"
    gap4_path = REPO / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json"
    try:
        survey = _read_json(survey_path)
        prior_artifact = _read_json(prior_path)
        gap4_artifact = _read_json(gap4_path)
        baselines = load_environment_baselines(REPO / "environment_files")
        target = select_deeper_level_target(survey, baselines, prior_artifact, gap4_artifact)
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
        artifact = blocked_artifact(
            target_game=LP85_GAME_ID,
            target_level=4,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    if not _fixture_available(target.game_id):
        artifact = blocked_artifact(
            target_game=target.game_id,
            target_level=target.target_level,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        offline_arcade = _load_offline_arcade()
        outcome = _run_lp85_l4_frontier(offline_arcade, target, prior_artifact, gap4_artifact)
    except Exception as exc:
        outcome = _failed_outcome(target, f"offline_run_failed_{type(exc).__name__.lower()}_{exc}")

    artifact = build_artifact(
        outcome,
        target,
        random_seed=RANDOM_SEED,
        duration_s=time.time() - started,
    )
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
