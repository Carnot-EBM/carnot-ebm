"""Shared contracts for Exp 4014 explore-first ARC-AGI-3 level-wall attempts.

Spec refs: REQ-PHASE4-026, SCENARIO-PHASE4-026.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.agentic.arc_world_model_synth import InducedWorldModel


BANKED_FRONTIER = {"lp85": 1, "sc25": 1, "r11l": 3}
STALLED_LEVELS = {"lp85": 2, "sc25": 2, "r11l": 4}

REQUIRED_ARTIFACT_FIELDS = (
    "ACCURACY_total_levels_solved",
    "new_levels_this_task",
    "per_game_max_level",
    "explore_first_found_validated_candidate",
    "verifier_validated_count",
    "actions_saved_vs_openloop",
    "exploration_actions_used",
    "real_env_confirmed",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")


@dataclass(frozen=True)
class TransitionObservation:
    """One observed per-level transition collected before rule induction."""

    before: np.ndarray
    action_key: tuple[int, ...]
    after: np.ndarray
    level_delta: int
    game_over: bool


@dataclass(frozen=True)
class LevelWallResult:
    """One scoped game result for the Exp 4014 aggregate artifact."""

    short_game: str
    game_id: str
    banked_level: int
    target_level: int
    levels_completed: int
    first_fail_level: int | None
    exploration_actions_used: int
    observed_dynamics: list[dict[str, Any]]
    dynamics_induced: bool
    candidate_validations: list[dict[str, Any]]
    committed_actions: list[dict[str, Any]]
    verifier_validated_count: int
    actions_saved_vs_openloop: int
    real_env_confirmed: bool
    stall_reason: str
    solve_log: list[dict[str, Any]]

    @property
    def new_levels(self) -> int:
        return max(0, int(self.levels_completed) - int(self.banked_level))


def _duration(started: float) -> float:
    return round(time.time() - started, 3) if started else 0.0


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:80] or "unknown"


def count_validated_candidates(rows: list[dict[str, Any]], *, energy_threshold: float = 0.0) -> int:
    """Count selected candidates that passed held-out executed consistency."""

    return sum(
        1
        for row in rows
        if row.get("selected") is True
        and row.get("heldout_energy") is not None
        and float(row.get("heldout_energy", 1.0)) <= energy_threshold
        and int(row.get("heldout_n", 0) or 0) > 0
    )


def induce_model_from_level_observations(
    game_id: str,
    observations: list[TransitionObservation],
) -> InducedWorldModel:
    """Build a grounded world model only after per-level transitions exist."""

    if not observations:
        raise ValueError("at least one observed transition is required before level-rule induction")
    transitions = [(obs.before, obs.action_key, obs.after) for obs in observations]
    return InducedWorldModel(game_id).fit(transitions)


def blocked_artifact(
    *,
    seed: int,
    started: float,
    inference_substrate: str,
    verdict: str,
) -> dict[str, Any]:
    """Build a schema-valid blocked artifact for failed preconditions."""

    artifact = {
        "experiment": "experiment_4014_break_level_wall_explore_first",
        "title": "arc3_explore_first_level_wall_reinduction",
        "ACCURACY_total_levels_solved": 0,
        "new_levels_this_task": 0,
        "per_game_max_level": {},
        "explore_first_found_validated_candidate": False,
        "verifier_validated_count": 0,
        "actions_saved_vs_openloop": 0,
        "exploration_actions_used": 0,
        "real_env_confirmed": False,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": _duration(started),
        "inference_substrate": inference_substrate,
        "banked_frontier": dict(BANKED_FRONTIER),
        "stalled_levels": dict(STALLED_LEVELS),
        "wall_results": {},
        "observed_dynamics": {},
        "candidate_validations": {},
        "precondition_blocked": True,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_level_wall_artifact(
    results: list[LevelWallResult],
    *,
    seed: int,
    started: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """Aggregate scoped explore-first level-wall results into the terminal artifact."""

    per_game = {row.short_game: int(row.levels_completed) for row in results}
    new_by_game = {row.short_game: row.new_levels for row in results}
    total_levels = sum(per_game.values())
    new_levels = sum(new_by_game.values())
    validations = sum(int(row.verifier_validated_count) for row in results)
    saved = sum(int(row.actions_saved_vs_openloop) for row in results)
    exploration = sum(int(row.exploration_actions_used) for row in results)
    confirmed = bool(results) and all(row.real_env_confirmed for row in results)
    found_validated = validations > 0 and new_levels > 0

    if new_levels > 0:
        order = {name: index for index, name in enumerate(STALLED_LEVELS)}
        leader = sorted(results, key=lambda row: (-row.new_levels, order.get(row.short_game, 99)))[0]
        verdict = f"success: broke_wall_{leader.short_game}_to_L{leader.levels_completed}_total{total_levels}"
    else:
        parts = [
            f"{row.short_game}_L{row.first_fail_level or row.target_level}_{_slug(row.stall_reason)}"
            for row in results
        ]
        verdict = f"complete: level_walls_hold_{'_'.join(parts)}_total{total_levels}"

    artifact = {
        "experiment": "experiment_4014_break_level_wall_explore_first",
        "title": "arc3_explore_first_level_wall_reinduction",
        "ACCURACY_total_levels_solved": int(total_levels),
        "new_levels_this_task": int(new_levels),
        "per_game_max_level": per_game,
        "explore_first_found_validated_candidate": bool(found_validated),
        "verifier_validated_count": int(validations),
        "actions_saved_vs_openloop": int(saved),
        "exploration_actions_used": int(exploration),
        "real_env_confirmed": bool(confirmed),
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": _duration(started),
        "inference_substrate": inference_substrate,
        "banked_frontier": dict(BANKED_FRONTIER),
        "stalled_levels": dict(STALLED_LEVELS),
        "per_game_new_levels": new_by_game,
        "first_fail_level": {row.short_game: row.first_fail_level for row in results},
        "stall_reasons": {row.short_game: row.stall_reason for row in results},
        "observed_dynamics": {row.short_game: row.observed_dynamics for row in results},
        "candidate_validations": {row.short_game: row.candidate_validations for row in results},
        "committed_actions": {row.short_game: row.committed_actions for row in results},
        "solve_log": {row.short_game: row.solve_log for row in results},
        "dynamics_induced": {row.short_game: row.dynamics_induced for row in results},
        "wall_results": {
            row.short_game: {
                "game_id": row.game_id,
                "banked_level": int(row.banked_level),
                "target_level": int(row.target_level),
                "levels_completed": int(row.levels_completed),
                "first_fail_level": row.first_fail_level,
                "exploration_actions_used": int(row.exploration_actions_used),
                "verifier_validated_count": int(row.verifier_validated_count),
                "actions_saved_vs_openloop": int(row.actions_saved_vs_openloop),
                "real_env_confirmed": bool(row.real_env_confirmed),
                "stall_reason": row.stall_reason,
            }
            for row in results
        },
        "precondition_blocked": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def _map_int_errors(name: str, value: Any) -> list[str]:
    if not isinstance(value, dict) or any(type(key) is not str or type(item) is not int for key, item in value.items()):
        return [f"{name} must be a map of string to bare int"]
    return []


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Validate the bare-field terminal schema required by REQ-PHASE4-026."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    for field in (
        "ACCURACY_total_levels_solved",
        "new_levels_this_task",
        "verifier_validated_count",
        "actions_saved_vs_openloop",
        "exploration_actions_used",
        "random_seed",
    ):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")

    for field in ("explore_first_found_validated_candidate", "real_env_confirmed"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")

    for field in ("honest_verdict", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    if "per_game_max_level" in artifact:
        errors.extend(_map_int_errors("per_game_max_level", artifact["per_game_max_level"]))

    if "duration_s" in artifact and type(artifact["duration_s"]) not in (int, float):
        errors.append("duration_s must be a bare number")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str):
        if not verdict.startswith(TERMINAL_PREFIXES):
            errors.append("honest_verdict must start with complete:/success:/blocked_")
        exploration_actions = artifact.get("exploration_actions_used", 0)
        if (
            not verdict.startswith("blocked_")
            and isinstance(exploration_actions, int)
            and exploration_actions <= 0
        ):
            errors.append("exploration_actions_used must be >0 for non-blocked attempts")
        if verdict.startswith("success:"):
            if artifact.get("new_levels_this_task", 0) <= 0:
                errors.append("new_levels_this_task must be >0 for success")
            if artifact.get("explore_first_found_validated_candidate") is not True:
                errors.append("explore_first_found_validated_candidate must be true for success")
            if artifact.get("real_env_confirmed") is not True:
                errors.append("real_env_confirmed must be true for success")
    return errors
