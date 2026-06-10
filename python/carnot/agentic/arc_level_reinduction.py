"""Helpers for one-level ARC-AGI-3 re-induction experiments.

Spec refs: REQ-PHASE4-018, SCENARIO-PHASE4-018.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_ARTIFACT_FIELDS = (
    "ACCURACY_levels_solved",
    "new_levels_solved_this_task",
    "reinduction_found_different_rule",
    "game_advanced",
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
class ReinductionCandidate:
    short_game: str
    game_id: str
    first_fail_level: int
    prior_levels: int
    baseline_actions_ref: list[int]
    l1_mechanic: str
    reason: str
    score: int


def _level_summary(payload: dict[str, Any], level: int) -> dict[str, Any]:
    for row in payload.get("level_summaries", []):
        if int(row.get("level", -1) or -1) == level:
            return dict(row)
    return {}


def _score_stall(short_game: str, payload: dict[str, Any]) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    first_fail = int(payload.get("first_fail_level", 0) or 0)
    if first_fail == 2:
        score += 10
        reasons.append("stalled exactly at L2")

    l2 = _level_summary(payload, 2)
    if int(l2.get("n_pairs", 0) or 0) > 0 and int(l2.get("n_targets", 0) or 0) > 0:
        score += 25
        reasons.append("visible L2 piece-target structure remains perceivable")
    if "n_buttons" in l2 and int(l2.get("n_buttons", 0) or 0) == 0:
        score -= 20
        reasons.append("L2 exposes no visible buttons for the L1 button mechanic")
    if short_game == "r11l":
        score += 3
        reasons.append("prior solver already exposes engine-side piece-target groups")
    return score, "; ".join(reasons) or "no L2 re-induction signal"


def choose_reinduction_candidate(stalls: dict[str, dict[str, Any]]) -> ReinductionCandidate:
    candidates: list[ReinductionCandidate] = []
    for short_game, payload in stalls.items():
        score, reason = _score_stall(short_game, payload)
        mechanic = (
            payload.get("induced_select_place_mechanic")
            or payload.get("induced_mechanic")
            or "unknown L1 mechanic"
        )
        candidates.append(
            ReinductionCandidate(
                short_game=short_game,
                game_id=str(payload.get("game") or payload.get("game_solved") or short_game),
                first_fail_level=int(payload.get("first_fail_level", 0) or 0),
                prior_levels=int(payload.get("ACCURACY_levels_solved", 0) or 0),
                baseline_actions_ref=[int(v) for v in payload.get("baseline_actions_ref", [])],
                l1_mechanic=str(mechanic),
                reason=reason,
                score=score,
            )
        )
    if not candidates:
        raise ValueError("no prior stall diagnostics supplied")
    return max(candidates, key=lambda row: (row.score, -row.first_fail_level, row.short_game))


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    int_fields = ("ACCURACY_levels_solved", "new_levels_solved_this_task", "random_seed")
    for field in int_fields:
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")

    bool_fields = ("reinduction_found_different_rule", "real_env_confirmed")
    for field in bool_fields:
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")

    text_fields = ("game_advanced", "honest_verdict", "inference_substrate")
    for field in text_fields:
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    for field in ("per_level_actions", "baseline_actions_ref"):
        if field not in artifact:
            continue
        value = artifact[field]
        if not isinstance(value, list) or any(type(item) is not int for item in value):
            errors.append(f"{field} must be a list of bare ints")

    if "duration_s" in artifact and type(artifact["duration_s"]) not in (int, float):
        errors.append("duration_s must be a bare number")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete:/success:/blocked_")
    return errors
