"""Shared contracts for Exp 4003 ARC-AGI-3 frontier scaling.

Spec refs: REQ-PHASE4-023, SCENARIO-PHASE4-023.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

BANKED_FRONTIER = {"r11l": 3, "lp85": 1, "sc25": 1}

REQUIRED_ARTIFACT_FIELDS = (
    "ACCURACY_total_levels_solved",
    "new_levels_this_task",
    "per_game_max_level",
    "verifier_validated_count",
    "actions_saved_vs_openloop",
    "per_level_actions",
    "baseline_actions_ref",
    "real_env_confirmed",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")


@dataclass(frozen=True)
class GameFrontierResult:
    """One scoped game frontier result for the Exp 4003 aggregate artifact."""

    short_game: str
    game_id: str
    banked_level: int
    levels_completed: int
    first_fail_level: int | None
    per_level_actions: list[int]
    baseline_actions_ref: list[int]
    verifier_validated_count: int
    actions_saved_vs_openloop: int
    real_env_confirmed: bool
    stall_reason: str
    level_summaries: list[dict[str, Any]]
    solve_log: list[dict[str, Any]]
    candidate_validations: list[dict[str, Any]]

    @property
    def new_levels(self) -> int:
        return max(0, int(self.levels_completed) - int(self.banked_level))


def _duration(started: float) -> float:
    return round(time.time() - started, 3) if started else 0.0


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:80] or "unknown"


def count_validated_rules(rows: list[dict[str, Any]], *, energy_threshold: float = 0.0) -> int:
    """Count selected candidates that passed held-out executed consistency."""

    return sum(
        1
        for row in rows
        if row.get("selected") is True
        and row.get("heldout_energy") is not None
        and float(row.get("heldout_energy", 1.0)) <= energy_threshold
        and int(row.get("heldout_n", 0) or 0) > 0
    )


def build_frontier_artifact(
    results: list[GameFrontierResult],
    *,
    seed: int,
    started: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """Aggregate scoped per-game results into the required Exp 4003 artifact."""

    per_game = {row.short_game: int(row.levels_completed) for row in results}
    new_by_game = {row.short_game: row.new_levels for row in results}
    total_levels = sum(per_game.values())
    new_levels = sum(new_by_game.values())
    actions = {row.short_game: [int(value) for value in row.per_level_actions] for row in results}
    baselines = {row.short_game: [int(value) for value in row.baseline_actions_ref] for row in results}
    validations = sum(int(row.verifier_validated_count) for row in results)
    saved = sum(int(row.actions_saved_vs_openloop) for row in results)
    confirmed = bool(results) and all(row.real_env_confirmed for row in results)

    if new_levels > 0:
        ordered = sorted(
            results,
            key=lambda row: (-row.new_levels, list(BANKED_FRONTIER).index(row.short_game), -row.levels_completed),
        )
        leader = ordered[0]
        verdict = f"success: scaled_level_frontier_{leader.short_game}_to_L{leader.levels_completed}_total{total_levels}"
    else:
        held = next((row for row in results if row.first_fail_level is not None), results[0])
        reason = f"{held.short_game}_L{held.first_fail_level}_{_slug(held.stall_reason)}"
        verdict = f"complete: level_frontier_holds_{reason}_total{total_levels}"

    return {
        "experiment": "experiment_4003_scale_level_frontier",
        "title": "arc3_verifier_validated_frontier_scaling",
        "ACCURACY_total_levels_solved": int(total_levels),
        "new_levels_this_task": int(new_levels),
        "per_game_max_level": per_game,
        "verifier_validated_count": int(validations),
        "actions_saved_vs_openloop": int(saved),
        "per_level_actions": actions,
        "baseline_actions_ref": baselines,
        "real_env_confirmed": confirmed,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": _duration(started),
        "inference_substrate": inference_substrate,
        "banked_frontier": dict(BANKED_FRONTIER),
        "per_game_new_levels": new_by_game,
        "first_fail_level": {row.short_game: row.first_fail_level for row in results},
        "stall_reasons": {row.short_game: row.stall_reason for row in results},
        "level_summaries": {row.short_game: row.level_summaries for row in results},
        "solve_log": {row.short_game: row.solve_log for row in results},
        "candidate_validations": {row.short_game: row.candidate_validations for row in results},
    }


def _map_int_errors(name: str, value: Any) -> list[str]:
    if not isinstance(value, dict) or any(type(key) is not str or type(item) is not int for key, item in value.items()):
        return [f"{name} must be a map of string to bare int"]
    return []


def _map_int_list_errors(name: str, value: Any) -> list[str]:
    if not isinstance(value, dict) or any(
        type(key) is not str or not isinstance(items, list) or any(type(item) is not int for item in items)
        for key, items in value.items()
    ):
        return [f"{name} must be a map of string to lists of bare ints"]
    return []


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Validate the bare-field terminal schema required by REQ-PHASE4-023."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    for field in (
        "ACCURACY_total_levels_solved",
        "new_levels_this_task",
        "verifier_validated_count",
        "actions_saved_vs_openloop",
        "random_seed",
    ):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")

    if "real_env_confirmed" in artifact and type(artifact["real_env_confirmed"]) is not bool:
        errors.append("real_env_confirmed must be a bare bool")

    for field in ("honest_verdict", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    if "per_game_max_level" in artifact:
        errors.extend(_map_int_errors("per_game_max_level", artifact["per_game_max_level"]))

    for field in ("per_level_actions", "baseline_actions_ref"):
        if field in artifact:
            errors.extend(_map_int_list_errors(field, artifact[field]))

    if "duration_s" in artifact and type(artifact["duration_s"]) not in (int, float):
        errors.append("duration_s must be a bare number")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete:/success:/blocked_")
    return errors
