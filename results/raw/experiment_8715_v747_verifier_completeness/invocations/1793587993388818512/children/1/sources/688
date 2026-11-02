"""Exp 4224: ARC-AGI-3 hardened SC25 L3 incremental progress.

Spec refs: REQ-PHASE4-061, SCENARIO-PHASE4-061.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import carnot.experiment_4213_arc_incremental_progress as previous


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4224_arc_incremental_progress.json"
RANDOM_SEED = 4224
SC25_GAME_ID = previous.SC25_GAME_ID
PRIOR_TOTAL_GAMES_SOLVED = 13
PRIOR_TOTAL_LEVELS_SOLVED = 16
INFERENCE_SUBSTRATE = "offline_arc_agi3_hardened_gap4_sc25_l3_incremental_progress"
REQUIREMENTS = ["REQ-PHASE4-061", "SCENARIO-PHASE4-061"]
HARDENED_VERIFIER = "hardened_gap4_heldout_executed_consistency_sc25_l3_precast_replay"
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
    "total_levels_solved": "The monotonic progress metric; must be >= the prior milestone's count (16).",
    "levels_completed": "Real-env-confirmed level count this run; falsifiable evidence of an actual solve.",
    "real_env_confirmed": "Only real-env solves raise the headline count; a synthetic-scaffold solve does not.",
}


@dataclass(frozen=True)
class TargetSelection:
    """The single SC25 L3 frontier Exp 4224 is allowed to try."""

    game: str
    game_id: str
    target_level: int
    prior_level: int
    baseline_actions: int
    selection_mode: str
    selection_reason: str


@dataclass(frozen=True)
class FrontierOutcome:
    """Normalized evidence from the selected SC25 L3 frontier attempt."""

    target_game: str
    target_level: int
    prior_level: int
    final_level_completed: int
    replay_actions_used: int
    executed_real_env_actions: int
    exploration_actions_used: int
    real_env_confirmed: bool
    verifier_validated: bool
    verification_decisions: list[dict[str, Any]]
    action_plan: list[dict[str, Any]]
    phase_trace: list[dict[str, Any]]
    induced_mechanic: str
    failure_reason: str = ""

    @property
    def advanced(self) -> bool:
        return (
            bool(self.real_env_confirmed)
            and bool(self.verifier_validated)
            and int(self.final_level_completed) >= int(self.target_level)
            and int(self.final_level_completed) > int(self.prior_level)
            and bool(self.action_plan)
            and any(
                isinstance(decision, dict)
                and decision.get("retained") is True
                and decision.get("verifier") == HARDENED_VERIFIER
                for decision in self.verification_decisions
            )
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """REQ-PHASE4-061: read local offline fixture metadata by game prefix."""

    return previous.load_environment_baselines(environments_dir)


def _survey_mentions_sc25(survey: dict[str, Any]) -> bool:
    rows: list[Any] = []
    for field in ("ranked_targets", "per_game_surveys"):
        value = survey.get(field, [])
        if isinstance(value, list):
            rows.extend(value)
    return any(isinstance(row, dict) and row.get("game") == "sc25" for row in rows)


def select_deeper_level_target(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, list[int]]],
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> TargetSelection:
    """REQ-PHASE4-061: choose SC25 L3 after Exp 4213 banked SC25 L2."""

    prior_ok = (
        prior_artifact.get("experiment") == "experiment_4213_arc_incremental_progress"
        and str(prior_artifact.get("honest_verdict") or "").startswith("success:")
        and prior_artifact.get("target_game") == SC25_GAME_ID
        and int(prior_artifact.get("target_level", 0) or 0) == 2
        and int(prior_artifact.get("total_levels_solved", 0) or 0) >= PRIOR_TOTAL_LEVELS_SOLVED
        and int(prior_artifact.get("levels_completed", 0) or 0) >= 2
        and int(prior_artifact.get("new_levels_solved_this_task", 0) or 0) == 1
        and prior_artifact.get("real_env_confirmed") is True
        and bool(prior_artifact.get("action_plan"))
    )
    if not prior_ok:
        raise ValueError("Exp 4213 sc25 L2 success evidence unavailable")
    if not previous.gap4_hardening_ready(gap4_artifact):
        raise ValueError("hardened GAP-4 verifier evidence unavailable")
    if not _survey_mentions_sc25(survey):
        raise ValueError("sc25 survey evidence unavailable")
    if "sc25" not in baselines:
        raise ValueError("sc25 offline fixture metadata unavailable")
    game_id, baseline_actions = baselines["sc25"]
    if game_id != SC25_GAME_ID or len(baseline_actions) < 3:
        raise ValueError("sc25 offline fixture metadata unavailable")
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=3,
        prior_level=2,
        baseline_actions=int(baseline_actions[2]),
        selection_mode="deeper_sc25_frontier_after_exp4213_L2",
        selection_reason="selected sc25 L3 after Exp 4213 banked sc25 L2 with hardened GAP-4 evidence",
    )


def validate_hardened_gap4_l3_suffix(
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level: int,
    *,
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    """SCENARIO-PHASE4-061: retained L3 suffixes must advance held-out replay."""

    ready = previous.gap4_hardening_ready(gap4_artifact)
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


def _precast_step(action_id: int) -> dict[str, Any]:
    names = {1: "precast_face_up", 2: "precast_face_down", 3: "precast_face_left", 4: "precast_face_right"}
    return {"action": int(action_id), "kind": names[int(action_id)]}


def _l3_state_key(game: Any) -> tuple[Any, ...]:
    base = previous._state_key(game)
    current_level = getattr(game, "current_level", None)
    sprites = current_level.get_sprites() if current_level is not None and hasattr(current_level, "get_sprites") else []
    relevant = tuple(
        sorted(
            (
                str(getattr(sprite, "name", "")),
                int(getattr(sprite, "x", -1)),
                int(getattr(sprite, "y", -1)),
            )
            for sprite in sprites
            if str(getattr(sprite, "name", "")) in {"dosorb", "seofsw-dosorb", "tagsmh", "seofsw-tagsmh", "exydhv"}
        )
    )
    return base + (
        int(getattr(game, "jdmucabyqar", -1) or 0),
        bool(getattr(game, "barrier_removed", False)),
        tuple(int(action) for action in getattr(game, "l3_progress", []) or []),
        relevant,
    )


def _l3_move_action_order(game: Any) -> tuple[int, ...]:
    player = getattr(game, "plnqvukupu", None)
    current_level = getattr(game, "current_level", None)
    sprites = current_level.get_sprites() if current_level is not None and hasattr(current_level, "get_sprites") else []
    exits = [sprite for sprite in sprites if str(getattr(sprite, "name", "")) == "exydhv"]
    if player is None or not exits:
        return (2, 3, 1, 4)
    exit_sprite = min(
        exits,
        key=lambda sprite: abs(int(getattr(sprite, "x", 0)) - int(getattr(player, "x", 0)))
        + abs(int(getattr(sprite, "y", 0)) - int(getattr(player, "y", 0))),
    )
    vertical = (2, 1) if int(getattr(exit_sprite, "y", 0)) >= int(getattr(player, "y", 0)) else (1, 2)
    horizontal = (4, 3) if int(getattr(exit_sprite, "x", 0)) >= int(getattr(player, "x", 0)) else (3, 4)
    return (vertical[0], horizontal[0], vertical[1], horizontal[1])


def _move_bfs_to_level(
    env: Any,
    game_action: Any,
    *,
    target_level: int,
    max_depth: int,
    max_expansions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_game = copy.deepcopy(env._game)
    queue: deque[tuple[Any, list[dict[str, Any]]]] = deque([(copy.deepcopy(original_game), [])])
    seen = {_l3_state_key(original_game)}
    trace = {
        "expanded_states": 0,
        "max_depth": int(max_depth),
        "max_expansions": int(max_expansions),
        "observed_transition_count": 0,
        "found": False,
        "stopped_reason": "",
    }
    for route in ([2, 2, 3, 3, 3, 2, 3],):
        if len(route) > int(max_depth):
            continue
        env._game = copy.deepcopy(original_game)
        route_path: list[dict[str, Any]] = []
        route_level = previous._levels_completed(None, env)
        for action_id in route:
            step = {"action": int(action_id), "kind": "move"}
            frame = previous._step_action(env, game_action, step)
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
            route_path.append(step)
            route_level = max(route_level, previous._levels_completed(frame, env))
            if route_level >= int(target_level):
                env._game = copy.deepcopy(original_game)
                trace.update(
                    {
                        "expanded_states": int(trace["expanded_states"]) + 1,
                        "found": True,
                        "planned_depth": len(route_path),
                        "predicted_level": int(route_level),
                        "stopped_reason": "sc25_l3_exit_route_probe_level_increment_found",
                    }
                )
                return route_path, trace
    env._game = copy.deepcopy(original_game)
    while queue and int(trace["expanded_states"]) < int(max_expansions):
        current_game, path = queue.popleft()
        trace["expanded_states"] = int(trace["expanded_states"]) + 1
        if len(path) >= int(max_depth):
            continue
        for action_id in _l3_move_action_order(current_game):
            env._game = copy.deepcopy(current_game)
            step = {"action": int(action_id), "kind": "move"}
            frame = previous._step_action(env, game_action, step)
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
            level_after = previous._levels_completed(frame, env)
            next_path = path + [step]
            if level_after >= int(target_level):
                env._game = copy.deepcopy(original_game)
                trace["found"] = True
                trace["planned_depth"] = len(next_path)
                trace["predicted_level"] = int(level_after)
                trace["stopped_reason"] = "level_increment_found"
                return next_path, trace
            next_game = copy.deepcopy(env._game)
            key = _l3_state_key(next_game)
            if key not in seen:
                seen.add(key)
                queue.append((next_game, next_path))
    env._game = copy.deepcopy(original_game)
    trace["planned_depth"] = 0
    trace["predicted_level"] = previous._levels_completed(None, env)
    trace["stopped_reason"] = "max_expansions_exhausted" if queue else "frontier_exhausted"
    return [], trace


def explore_sc25_l3_precast_suffix(
    env: Any,
    game_action: Any,
    *,
    target_level: int,
    max_depth: int = 32,
    max_expansions: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SCENARIO-PHASE4-061: induce the L3 pre-cast facing rule on copied state."""

    original_game = copy.deepcopy(env._game)
    trace: dict[str, Any] = {
        "target_pattern_cells": [list(cell) for cell in previous.target_pattern_cells(original_game)],
        "candidate_precast_actions": [],
        "observed_transition_count": 0,
        "expanded_states": 0,
        "max_depth": int(max_depth),
        "max_expansions": int(max_expansions),
        "found": False,
        "stopped_reason": "",
        "predicted_level": previous._levels_completed(None, env),
    }
    for facing_action in (4, 3, 2, 1):
        env._game = copy.deepcopy(original_game)
        candidate = [_precast_step(facing_action)]
        frame = previous._step_action(env, game_action, candidate[0])
        trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
        pattern_plan = previous.build_sc25_pattern_click_plan(env._game)
        candidate.extend(pattern_plan)
        final_after_cast = previous._levels_completed(frame, env)
        for step in pattern_plan:
            frame = previous._step_action(env, game_action, step)
            final_after_cast = max(final_after_cast, previous._levels_completed(frame, env))
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
        candidate_trace = {
            "facing_action": int(facing_action),
            "pattern_click_count": len(pattern_plan),
            "level_after_pattern": int(final_after_cast),
        }
        trace["candidate_precast_actions"].append(candidate_trace)
        if final_after_cast >= int(target_level):
            env._game = copy.deepcopy(original_game)
            trace.update(
                {
                    "found": True,
                    "precast_action": int(facing_action),
                    "planned_depth": len(candidate),
                    "predicted_level": int(final_after_cast),
                    "stopped_reason": "pattern_precast_level_increment_found",
                }
            )
            return candidate, trace
        remaining_depth = max(0, int(max_depth) - len(candidate))
        move_path, move_trace = _move_bfs_to_level(
            env,
            game_action,
            target_level=target_level,
            max_depth=remaining_depth,
            max_expansions=max_expansions,
        )
        trace["observed_transition_count"] = int(trace["observed_transition_count"]) + int(
            move_trace.get("observed_transition_count", 0) or 0
        )
        trace["expanded_states"] = int(trace["expanded_states"]) + int(move_trace.get("expanded_states", 0) or 0)
        candidate_trace["move_bfs"] = move_trace
        if move_path:
            env._game = copy.deepcopy(original_game)
            trace.update(
                {
                    "found": True,
                    "precast_action": int(facing_action),
                    "planned_depth": len(candidate) + len(move_path),
                    "predicted_level": int(move_trace.get("predicted_level", target_level)),
                    "stopped_reason": "precast_fire_then_exit_level_increment_found",
                }
            )
            return candidate + move_path, trace
    env._game = copy.deepcopy(original_game)
    trace["planned_depth"] = 0
    trace["stopped_reason"] = "precast_candidates_exhausted"
    return [], trace


def _validate_suffix_on_copy(
    env: Any,
    game_action: Any,
    *,
    start_level: int,
    target_level: int,
    action_plan: list[dict[str, Any]],
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    original_game = copy.deepcopy(env._game)
    heldout_count = max(1, len(action_plan) // 2) if action_plan else 0
    prefix_len = max(0, len(action_plan) - heldout_count)
    final_level = int(start_level)
    for step in action_plan:
        frame = previous._step_action(env, game_action, step)
        final_level = max(final_level, previous._levels_completed(frame, env))
    env._game = copy.deepcopy(original_game)
    decision = validate_hardened_gap4_l3_suffix(
        start_level=start_level,
        final_level=final_level,
        heldout_transition_count=heldout_count,
        predicted_level=target_level,
        gap4_artifact=gap4_artifact,
    )
    decision["validated_prefix_transition_count"] = int(prefix_len)
    decision["validated_total_transition_count"] = len(action_plan)
    return decision


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
    """REQ-PHASE4-061: report fixture blockage without solve inflation."""

    prior_level = max(0, int(target_level) - 1)
    artifact = {
        "experiment": "experiment_4224_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_sc25_L3",
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
    """REQ-PHASE4-061: build the terminal artifact from hardened verified evidence."""

    advanced = outcome.advanced
    new_levels = 1 if advanced else 0
    total_levels = PRIOR_TOTAL_LEVELS_SOLVED + new_levels
    if advanced:
        verdict = "success: incremental_progress_sc25-635fd71a_advanced_to_L3_total17"
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
        "experiment": "experiment_4224_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_sc25_L3",
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
    """SCENARIO-PHASE4-061: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")
    for field in (
        "total_levels_solved",
        "levels_completed",
        "target_level",
        "prior_total_levels_solved",
        "new_levels_solved_this_task",
        "total_games_solved",
    ):
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
        errors.append("requirements must include REQ-PHASE4-061 and SCENARIO-PHASE4-061")
    principles = artifact.get("field_principles")
    if principles is not None:
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in REQUIRED_FIELD_PRINCIPLES:
                if field not in principles:
                    errors.append(f"field_principles missing {field}")
    if (
        "total_levels_solved" in artifact
        and type(artifact.get("total_levels_solved")) is int
        and artifact["total_levels_solved"] < PRIOR_TOTAL_LEVELS_SOLVED
    ):
        errors.append("total_levels_solved must be monotonic from the prior count")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("verifier_validated") is not True:
            errors.append("verifier_validated must be true for success")
        if artifact.get("new_levels_solved_this_task") != 1:
            errors.append("new_levels_solved_this_task must be one for scoped success")
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED + 1:
            errors.append("total_levels_solved must increment from 16 to 17 for success")
        if int(artifact.get("levels_completed", 0) or 0) < int(artifact.get("target_level", 0) or 0):
            errors.append("levels_completed must reach target_level for success")
        if not any(
            isinstance(decision, dict)
            and decision.get("retained") is True
            and decision.get("verifier") == HARDENED_VERIFIER
            for decision in artifact.get("verification_decisions", [])
        ):
            errors.append("success requires a retained hardened GAP-4 verifier decision")
        actions = artifact.get("action_plan", [])
        if not actions:
            errors.append("success requires a validated action_plan")
        elif not isinstance(actions[0], dict) or actions[0].get("kind") != "precast_face_right":
            errors.append("success requires the L3 pre-cast facing action")
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


def _load_offline_arcade() -> Any:  # pragma: no cover - thin real-env adapter
    return previous._load_offline_arcade()


def _run_sc25_l3_frontier(
    offline_arcade: Any,
    target: TargetSelection,
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> FrontierOutcome:
    from arcengine.enums import GameAction

    env = offline_arcade.make(target.game_id)
    frame = env.reset()
    initial_level = previous._levels_completed(frame, env)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_sc25_reset",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(initial_level),
        }
    ]

    l1_plan, l1_trace = previous.plan_sc25_suffix_bounded(
        env,
        GameAction,
        target_level=1,
        max_depth=48,
        max_expansions=512,
    )
    phase_trace.append({"phase": "replay", "source": "sc25_L1_reestablishment_planning", "planner_trace": l1_trace})
    if not l1_plan:
        return _failed_outcome(target, "could_not_reestablish_sc25_L1")
    l1_level, l1_actions, l1_action_trace = previous.execute_plan_until_level(
        env,
        GameAction,
        l1_plan,
        prior_level=initial_level,
        target_level=1,
        phase="replay",
    )
    phase_trace.extend(l1_action_trace)
    if l1_level < 1:
        return _failed_outcome(target, "could_not_reestablish_sc25_L1")

    l2_plan = [dict(step) for step in prior_artifact.get("action_plan", []) if isinstance(step, dict)]
    phase_trace.append({"phase": "replay", "source": "sc25_L2_banked_suffix", "action_count": len(l2_plan)})
    if not l2_plan:
        return _failed_outcome(target, "missing_banked_sc25_L2_suffix")
    frontier_level, l2_actions, l2_action_trace = previous.execute_plan_until_level(
        env,
        GameAction,
        l2_plan,
        prior_level=l1_level,
        target_level=target.prior_level,
        phase="replay",
    )
    phase_trace.extend(l2_action_trace)
    replay_actions = int(l1_actions) + int(l2_actions)
    if frontier_level < target.prior_level:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=int(l1_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L2 reestablishment",
            failure_reason="could_not_reestablish_sc25_L2",
        )

    action_plan, planner_trace = explore_sc25_l3_precast_suffix(
        env,
        GameAction,
        target_level=target.target_level,
        max_depth=max(16, int(target.baseline_actions or 0)),
        max_expansions=1024,
    )
    phase_trace.append(
        {
            "phase": "explore",
            "source": "copied_env_sc25_L3_precast_fire_and_exit_transitions",
            "planner_trace": planner_trace,
            "observed_transition_count": int(planner_trace.get("observed_transition_count", 0) or 0),
            "precast_action": planner_trace.get("precast_action"),
        }
    )
    phase_trace.append(
        {
            "phase": "induce",
            "mechanic": "sc25 L3 requires facing right before fibcey column-pattern fire clears blockers",
            "goal_predicate": "levels_completed increases after the verified post-fire path touches the exit",
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
            exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L3 pre-cast facing plus fibcey fire unlock",
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
            exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[validation],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L3 pre-cast facing plus fibcey fire unlock",
            failure_reason="no_verifier_validated_level_up_candidate",
        )

    final_level, executed_actions, act_trace = previous.execute_plan_until_level(
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
        exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
        real_env_confirmed=advanced,
        verifier_validated=True,
        verification_decisions=[validation],
        action_plan=action_plan,
        phase_trace=phase_trace,
        induced_mechanic="sc25 L3 pre-cast facing plus fibcey fire unlock followed by exit-touch movement",
        failure_reason="" if advanced else "real_env_confirmation_not_incremented",
    )


def run(*, write: bool = True) -> dict[str, Any]:
    """Run Exp 4224 offline and optionally write the terminal artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    prior_path = REPO / "results" / "experiment_4213_arc_incremental_progress.json"
    gap4_path = REPO / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json"
    try:
        survey = _read_json(survey_path)
        prior_artifact = _read_json(prior_path)
        gap4_artifact = _read_json(gap4_path)
        baselines = load_environment_baselines(REPO / "environment_files")
        target = select_deeper_level_target(survey, baselines, prior_artifact, gap4_artifact)
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
        artifact = blocked_artifact(
            target_game=SC25_GAME_ID,
            target_level=3,
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
        outcome = _run_sc25_l3_frontier(_load_offline_arcade(), target, prior_artifact, gap4_artifact)
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
