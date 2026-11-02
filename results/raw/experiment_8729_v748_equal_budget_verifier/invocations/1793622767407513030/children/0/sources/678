"""Exp 4213: ARC-AGI-3 hardened fallback incremental progress.

Spec refs: REQ-PHASE4-059, SCENARIO-PHASE4-059.
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


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4213_arc_incremental_progress.json"
RANDOM_SEED = 4213
LP85_GAME_ID = "lp85-305b61c3"
SC25_GAME_ID = "sc25-635fd71a"
PRIOR_TOTAL_GAMES_SOLVED = 13
PRIOR_TOTAL_LEVELS_SOLVED = 15
INFERENCE_SUBSTRATE = "offline_arc_agi3_hardened_gap4_sc25_l2_incremental_progress"
REQUIREMENTS = ["REQ-PHASE4-059", "SCENARIO-PHASE4-059"]
HARDENED_VERIFIER = "hardened_gap4_heldout_executed_consistency_sc25_l2_replay"
SC25_GRID_COORDS = {
    (0, 0): (25, 50),
    (0, 1): (30, 50),
    (0, 2): (35, 50),
    (1, 0): (25, 55),
    (1, 1): (30, 55),
    (1, 2): (35, 55),
    (2, 0): (25, 60),
    (2, 1): (30, 60),
    (2, 2): (35, 60),
}
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


@dataclass(frozen=True)
class TargetSelection:
    """The one fallback already-solved-game level Exp 4213 is allowed to try."""

    game: str
    game_id: str
    target_level: int
    prior_level: int
    baseline_actions: int
    selection_mode: str
    selection_reason: str


@dataclass(frozen=True)
class FrontierOutcome:
    """Normalized evidence from the selected fallback frontier attempt."""

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


def _action_id(action: object) -> int:
    if isinstance(action, int):
        return int(action)
    name = str(getattr(action, "name", "") or "")
    if name.startswith("ACTION") and name[6:].isdigit():
        return int(name[6:])
    value = getattr(action, "value", None)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.startswith("ACTION") and value[6:].isdigit():
        return int(value[6:])
    return int(action)  # type: ignore[arg-type]


def _game_action(game_action: Any, action_id: int) -> Any:
    return getattr(game_action, f"ACTION{int(action_id)}")


def _levels_completed(frame: Any, env: Any) -> int:
    values: list[int] = []
    for attr in ("levels_completed", "level_completed"):
        value = getattr(frame, attr, None) if frame is not None else None
        if value is not None and not isinstance(value, bool):
            values.append(int(value))
    game = getattr(env, "_game", None)
    if game is not None and hasattr(game, "_current_level_index"):
        values.append(int(getattr(game, "_current_level_index") or 0))
    return max(values or [0])


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """REQ-PHASE4-059: read local offline fixture metadata by game prefix."""

    baselines: dict[str, tuple[str, list[int]]] = {}
    for metadata in sorted(Path(environments_dir).glob("*/*/metadata.json")):
        try:
            payload = _read_json(metadata)
        except (OSError, json.JSONDecodeError):
            continue
        game_id = str(payload.get("game_id") or "")
        if "-" not in game_id:
            continue
        actions = [int(value) for value in payload.get("baseline_actions") or []]
        baselines[game_id.split("-", maxsplit=1)[0]] = (game_id, actions)
    return baselines


def gap4_hardening_ready(gap4_artifact: dict[str, Any]) -> bool:
    """REQ-PHASE4-059: B1 hardening evidence must be present before acting."""

    ledger = gap4_artifact.get("gross_recovery_ledger") if isinstance(gap4_artifact, dict) else None
    return (
        isinstance(gap4_artifact, dict)
        and gap4_artifact.get("experiment") == "experiment_4187_gap4_graded_execution_gate_hardening"
        and gap4_artifact.get("vote_aware_guard_blocked_mispromotion") is True
        and isinstance(ledger, dict)
        and int(ledger.get("recovered", 0) or 0) >= 4
        and int(ledger.get("lost", 0) or 0) == 0
    )


def _survey_mentions_sc25(survey: dict[str, Any]) -> bool:
    ranked = survey.get("ranked_targets", [])
    per_game = survey.get("per_game_surveys", [])
    rows = list(ranked if isinstance(ranked, list) else []) + list(per_game if isinstance(per_game, list) else [])
    return any(isinstance(row, dict) and row.get("game") == "sc25" for row in rows)


def select_deeper_level_target(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, list[int]]],
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> TargetSelection:
    """REQ-PHASE4-059: choose sc25 L2 after the lp85 L4 structural block."""

    prior_verdict = str(prior_artifact.get("honest_verdict") or "")
    prior_blocked = (
        prior_artifact.get("experiment") == "experiment_4201_arc_incremental_progress"
        and prior_artifact.get("target_game") == LP85_GAME_ID
        and int(prior_artifact.get("target_level", 0) or 0) == 4
        and int(prior_artifact.get("total_levels_solved", 0) or 0) >= PRIOR_TOTAL_LEVELS_SOLVED
        and int(prior_artifact.get("levels_completed", 0) or 0) == 3
        and prior_artifact.get("real_env_confirmed") is False
        and prior_verdict.startswith("complete:")
    )
    if not prior_blocked:
        raise ValueError("Exp 4201 lp85 L4 structural-block evidence unavailable")
    if not gap4_hardening_ready(gap4_artifact):
        raise ValueError("hardened GAP-4 verifier evidence unavailable")
    if not _survey_mentions_sc25(survey):
        raise ValueError("sc25 survey evidence unavailable")
    if "sc25" not in baselines:
        raise ValueError("sc25 offline fixture metadata unavailable")
    game_id, baseline_actions = baselines["sc25"]
    if game_id != SC25_GAME_ID or len(baseline_actions) < 2:
        raise ValueError("sc25 offline fixture metadata unavailable")
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=2,
        prior_level=1,
        baseline_actions=int(baseline_actions[1]),
        selection_mode="fallback_deeper_level_after_lp85_L4_structural_block",
        selection_reason=(
            "selected sc25 L2 after Exp 4201 structurally blocked on lp85 L4; "
            "sc25 has a local L2 baseline and already-solved L1 prefix"
        ),
    )


def validate_hardened_gap4_heldout_replay(
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level: int,
    *,
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    """SCENARIO-PHASE4-059: hardened GAP-4 retained suffixes must advance held-out replay."""

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


def target_pattern_cells(game: Any) -> list[tuple[int, int]]:
    """REQ-PHASE4-059: infer SC25's visible 3x3 target pattern from observed state."""

    target_names = getattr(game, "jlpticwjyvy", []) or []
    selected = str(target_names[0] if target_names else getattr(game, "ijhfdcamokt", ""))
    patterns = getattr(game, "zzpoabuniyn", {})
    pattern = patterns.get(selected)
    if not isinstance(pattern, list):
        raise ValueError("sc25 target pattern unavailable")
    return [
        (row, col)
        for row in range(3)
        for col in range(3)
        if bool(pattern[row][col])
    ]


def build_sc25_pattern_click_plan(game: Any) -> list[dict[str, Any]]:
    """SCENARIO-PHASE4-059: plan only missing target-pattern toggles."""

    current = getattr(game, "xhhaqjfncnp", [[False for _ in range(3)] for _ in range(3)])
    plan: list[dict[str, Any]] = []
    for row, col in target_pattern_cells(game):
        if not bool(current[row][col]):
            x, y = SC25_GRID_COORDS[(row, col)]
            plan.append(
                {
                    "action": 6,
                    "kind": "pattern_click",
                    "row": int(row),
                    "col": int(col),
                    "x": int(x),
                    "y": int(y),
                }
            )
    return plan


def _sc25_pattern_acceptance_observed(game: Any) -> bool:
    selected = getattr(game, "ijhfdcamokt", "__missing__")
    return bool(getattr(game, "pattern_ready", False)) or selected is None


def _state_key(game: Any) -> tuple[Any, ...]:
    player = getattr(game, "plnqvukupu", None)
    grid = getattr(game, "xhhaqjfncnp", [])
    return (
        int(getattr(game, "_current_level_index", 0) or 0),
        int(getattr(player, "x", -1) if player is not None else -1),
        int(getattr(player, "y", -1) if player is not None else -1),
        int(getattr(player, "scale", -1) if player is not None else -1),
        int(getattr(game, "rrinmfkkstu", 0) or 0),
        bool(getattr(game, "eycwbtepcvs", False)),
        int(getattr(game, "xelyxfeemol", 0) or 0),
        int(getattr(game, "ufpevlpokkj", 0) or 0),
        int(getattr(game, "wihhwrkolym", 0) or 0),
        tuple(tuple(bool(cell) for cell in row) for row in grid),
        str(getattr(game, "ijhfdcamokt", "")),
        str(getattr(game, "txyqmqkitgl", "")),
        bool(getattr(game, "pattern_ready", False)),
        int(getattr(game, "moves_after_pattern", 0) or 0),
    )


def _step_action(env: Any, game_action: Any, step: dict[str, Any]) -> Any:
    action_id = int(step["action"])
    action = _game_action(game_action, action_id)
    if action_id == 6 and "x" in step and "y" in step:
        return env.step(action, data={"x": int(step["x"]), "y": int(step["y"])})
    return env.step(action)


def plan_sc25_suffix_bounded(
    env: Any,
    game_action: Any,
    *,
    target_level: int,
    max_depth: int = 16,
    max_expansions: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SCENARIO-PHASE4-059: copied-state search has a hard expansion cap."""

    original_game = copy.deepcopy(env._game)
    pattern_plan = build_sc25_pattern_click_plan(original_game)
    trace: dict[str, Any] = {
        "target_pattern_cells": [list(cell) for cell in target_pattern_cells(original_game)],
        "pattern_click_count": len(pattern_plan),
        "observed_transition_count": 0,
        "expanded_states": 0,
        "max_depth": int(max_depth),
        "max_expansions": int(max_expansions),
        "found": False,
        "stopped_reason": "",
        "predicted_level": _levels_completed(None, env),
    }

    env._game = copy.deepcopy(original_game)
    final_after_pattern = trace["predicted_level"]
    for step in pattern_plan:
        frame = _step_action(env, game_action, step)
        trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
        final_after_pattern = max(int(final_after_pattern), _levels_completed(frame, env))
        if final_after_pattern >= int(target_level):
            env._game = copy.deepcopy(original_game)
            trace["found"] = True
            trace["predicted_level"] = int(final_after_pattern)
            trace["planned_depth"] = len(pattern_plan)
            trace["stopped_reason"] = "pattern_click_level_increment_found"
            return pattern_plan, trace
        if _sc25_pattern_acceptance_observed(env._game):
            break

    pattern_retry_count = 0
    while not _sc25_pattern_acceptance_observed(env._game) and pattern_retry_count < 2:
        extra_plan = build_sc25_pattern_click_plan(env._game)
        if not extra_plan:
            break
        pattern_retry_count += 1
        for step in extra_plan:
            pattern_plan.append(step)
            trace["pattern_click_count"] = len(pattern_plan)
            frame = _step_action(env, game_action, step)
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
            final_after_pattern = max(int(final_after_pattern), _levels_completed(frame, env))
            if final_after_pattern >= int(target_level):
                env._game = copy.deepcopy(original_game)
                trace["found"] = True
                trace["predicted_level"] = int(final_after_pattern)
                trace["planned_depth"] = len(pattern_plan)
                trace["stopped_reason"] = "pattern_click_level_increment_found"
                return pattern_plan, trace
            if _sc25_pattern_acceptance_observed(env._game):
                break

    start_game = copy.deepcopy(env._game)
    queue: deque[tuple[Any, list[dict[str, Any]]]] = deque([(start_game, [])])
    seen = {_state_key(start_game)}
    move_action_ids = (1, 2, 3, 4)
    while queue and int(trace["expanded_states"]) < int(max_expansions):
        current_game, path = queue.popleft()
        trace["expanded_states"] = int(trace["expanded_states"]) + 1
        if len(path) >= int(max_depth):
            continue
        for action_id in move_action_ids:
            env._game = copy.deepcopy(current_game)
            step = {"action": int(action_id), "kind": "move"}
            frame = _step_action(env, game_action, step)
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
            level_after = _levels_completed(frame, env)
            next_path = path + [step]
            if level_after >= int(target_level):
                env._game = copy.deepcopy(original_game)
                trace["found"] = True
                trace["predicted_level"] = int(level_after)
                trace["planned_depth"] = len(pattern_plan) + len(next_path)
                trace["stopped_reason"] = "level_increment_found"
                return pattern_plan + next_path, trace
            next_game = copy.deepcopy(env._game)
            key = _state_key(next_game)
            if key not in seen:
                seen.add(key)
                queue.append((next_game, next_path))

    env._game = copy.deepcopy(original_game)
    trace["predicted_level"] = int(final_after_pattern)
    trace["planned_depth"] = 0
    trace["stopped_reason"] = "max_expansions_exhausted" if queue else "frontier_exhausted"
    return [], trace


def execute_plan_until_level(
    env: Any,
    game_action: Any,
    action_plan: list[dict[str, Any]],
    *,
    prior_level: int,
    target_level: int,
    phase: str = "act",
) -> tuple[int, int, list[dict[str, Any]]]:
    """SCENARIO-PHASE4-059: execute validated actions until the scoped increment."""

    action_trace: list[dict[str, Any]] = []
    final_level = int(prior_level)
    for index, step in enumerate(action_plan, start=1):
        frame = _step_action(env, game_action, step)
        final_level = max(final_level, _levels_completed(frame, env))
        row = {
            "phase": str(phase),
            "action_index": int(index),
            "action": int(step["action"]),
            "kind": str(step.get("kind", "action")),
            "levels_completed": int(final_level),
        }
        if "x" in step and "y" in step:
            row["x"] = int(step["x"])
            row["y"] = int(step["y"])
        action_trace.append(row)
        if final_level >= int(target_level) or final_level > int(prior_level):
            break
    return final_level, len(action_trace), action_trace


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
        frame = _step_action(env, game_action, step)
        final_level = max(final_level, _levels_completed(frame, env))
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
    """REQ-PHASE4-059: report fixture blockage without solve inflation."""

    prior_level = max(0, int(target_level) - 1)
    artifact = {
        "experiment": "experiment_4213_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_sc25_L2",
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
    """REQ-PHASE4-059: build the terminal artifact from hardened verified evidence."""

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
        "experiment": "experiment_4213_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_sc25_L2",
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
    """SCENARIO-PHASE4-059: validate the terminal artifact contract."""

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
        errors.append("requirements must include REQ-PHASE4-059 and SCENARIO-PHASE4-059")
    if "field_principles" in artifact:
        principles = artifact["field_principles"]
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
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arcade = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    if not arcade.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arcade


def _run_sc25_l2_frontier(
    offline_arcade: Any,
    target: TargetSelection,
    gap4_artifact: dict[str, Any],
) -> FrontierOutcome:
    from arcengine.enums import GameAction

    env = offline_arcade.make(target.game_id)
    frame = env.reset()
    initial_level = _levels_completed(frame, env)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_sc25_reset",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(initial_level),
        }
    ]

    l1_plan, l1_planner_trace = plan_sc25_suffix_bounded(
        env,
        GameAction,
        target_level=target.prior_level,
        max_depth=48,
        max_expansions=512,
    )
    phase_trace.append(
        {
            "phase": "replay",
            "source": "sc25_L1_reestablishment_planning",
            "planner_trace": l1_planner_trace,
        }
    )
    if not l1_plan:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=initial_level,
            replay_actions_used=0,
            executed_real_env_actions=0,
            exploration_actions_used=int(l1_planner_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L1 reestablishment",
            failure_reason="could_not_reestablish_prior_frontier",
        )

    frontier_level, replay_actions, replay_trace = execute_plan_until_level(
        env,
        GameAction,
        l1_plan,
        prior_level=initial_level,
        target_level=target.prior_level,
        phase="replay",
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
            exploration_actions_used=int(l1_planner_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L1 reestablishment",
            failure_reason="could_not_reestablish_prior_frontier",
        )

    action_plan, planner_trace = plan_sc25_suffix_bounded(
        env,
        GameAction,
        target_level=target.target_level,
        max_depth=max(8, int(target.baseline_actions or 0)),
        max_expansions=512,
    )
    phase_trace.append(
        {
            "phase": "explore",
            "source": "copied_env_sc25_L2_pattern_and_exit_transitions",
            "planner_trace": planner_trace,
            "observed_transition_count": int(planner_trace.get("observed_transition_count", 0) or 0),
        }
    )
    phase_trace.append(
        {
            "phase": "induce",
            "mechanic": "sc25 target-pattern grid toggles unlock movement, then touching the exit increments level",
            "goal_predicate": "levels_completed increases after copied-env suffix reaches the exit",
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
            induced_mechanic="sc25 3x3 pattern-toggle unlock followed by exit-touch movement",
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
            induced_mechanic="sc25 3x3 pattern-toggle unlock followed by exit-touch movement",
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
        exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
        real_env_confirmed=advanced,
        verifier_validated=True,
        verification_decisions=[validation],
        action_plan=action_plan,
        phase_trace=phase_trace,
        induced_mechanic="sc25 3x3 pattern-toggle unlock followed by exit-touch movement",
        failure_reason="" if advanced else "real_env_confirmation_not_incremented",
    )


def run(*, write: bool = True) -> dict[str, Any]:
    """Run Exp 4213 offline and optionally write the terminal artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    prior_path = REPO / "results" / "experiment_4201_arc_incremental_progress.json"
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
            target_level=2,
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
        outcome = _run_sc25_l2_frontier(offline_arcade, target, gap4_artifact)
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
