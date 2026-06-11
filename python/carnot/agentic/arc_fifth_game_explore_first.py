"""Exp 4015 helpers for fifth-game explore-first ARC-AGI-3 pruning.

Spec refs: REQ-PHASE4-027, SCENARIO-PHASE4-027.
"""

from __future__ import annotations

import time
from typing import Any

from carnot.agentic.arc_fourth_game_explore_first import (
    REQUIRED_ARTIFACT_FIELDS,
    AttemptResult,
    CandidateGame,
    TransitionObservation,
    artifact_schema_errors,
    induce_model_from_observations,
    prune_candidates_after_induction,
    select_candidate_order,
)

SOLVED_GAME_PREFIXES = ("r11l", "lp85", "sc25", "su15")


def select_fifth_candidate_order(
    candidates: list[CandidateGame],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_GAME_PREFIXES,
) -> list[CandidateGame]:
    """REQ-PHASE4-027: reject banked solves, then apply the non-spatial baseline order."""

    remaining = [
        item
        for item in candidates
        if not any(item.game_id == prefix or item.game_id.startswith(f"{prefix}-") for prefix in solved_prefixes)
    ]
    return select_candidate_order(remaining)


def _attempt_to_json(attempt: AttemptResult) -> dict[str, Any]:
    return {
        "game_id": attempt.game_id,
        "baseline_actions": int(attempt.baseline_actions),
        "target_selection_reason": attempt.target_selection_reason,
        "exploration_actions_used": int(attempt.exploration_actions_used),
        "dynamics_induced": bool(attempt.dynamics_induced),
        "first_solve_at_action": int(attempt.first_solve_at_action),
        "levels_completed": int(attempt.levels_completed),
        "actions_vs_baseline": float(attempt.actions_vs_baseline),
        "induced_mechanic": attempt.induced_mechanic,
        "real_env_confirmed": bool(attempt.real_env_confirmed),
        "observed_dynamics": list(attempt.observed_dynamics),
        "pruner_decisions": list(attempt.pruner_decisions),
        "solve_log": list(attempt.solve_log),
        "failure_reason": attempt.failure_reason,
    }


def _duration(started: float) -> float:
    return round(time.time() - started, 3) if started else 0.0


def blocked_artifact(
    *,
    seed: int,
    started: float,
    inference_substrate: str,
    verdict: str = "blocked_arc_offline_env_unavailable",
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4015_fifth_game_explore_first",
        "title": "arc3_fifth_game_explore_first_verifier_pruned",
        "ACCURACY_levels_solved": 0,
        "game_solved": "none",
        "exploration_actions_used": 0,
        "dynamics_induced": False,
        "first_solve_at_action": -1,
        "actions_vs_baseline": 0.0,
        "induced_mechanic": "none",
        "games_attempted": [],
        "real_env_confirmed": False,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": _duration(started),
        "inference_substrate": inference_substrate,
        "attempt_details": [],
        "verifier_pruner_used": False,
        "precondition_blocked": True,
        "excluded_solved_games": list(SOLVED_GAME_PREFIXES),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_fifth_game_artifact(
    attempts: list[AttemptResult],
    *,
    seed: int,
    started: float,
    inference_substrate: str,
) -> dict[str, Any]:
    solved = next((attempt for attempt in attempts if attempt.solved), None)
    attempted_rows = [_attempt_to_json(attempt) for attempt in attempts]
    total_exploration = sum(int(attempt.exploration_actions_used) for attempt in attempts)
    any_induced = any(bool(attempt.dynamics_induced) for attempt in attempts)

    if solved is not None:
        levels_solved = int(solved.levels_completed)
        game_solved = solved.game_id
        first_solve = int(solved.first_solve_at_action)
        actions_vs_baseline = float(solved.actions_vs_baseline)
        mechanic = solved.induced_mechanic
        real_env_confirmed = bool(solved.real_env_confirmed)
        verdict = f"success: fifth_game_solved_{game_solved}_at_action{first_solve}"
    else:
        levels_solved = 0
        game_solved = "none"
        first_solve = -1
        actions_vs_baseline = 0.0
        real_env_confirmed = bool(attempts)
        reasons = [attempt.failure_reason for attempt in attempts if attempt.failure_reason]
        reason = reasons[0] if reasons else "no_candidate_solved"
        reason_slug = "_".join(str(reason).lower().replace("-", "_").split())
        mechanic = " | ".join(attempt.induced_mechanic for attempt in attempts) if attempts else "none"
        verdict = f"complete: fifth_game_no_solve_{reason_slug}"

    artifact = {
        "experiment": "experiment_4015_fifth_game_explore_first",
        "title": "arc3_fifth_game_explore_first_verifier_pruned",
        "ACCURACY_levels_solved": levels_solved,
        "game_solved": game_solved,
        "exploration_actions_used": int(total_exploration),
        "dynamics_induced": bool(any_induced),
        "first_solve_at_action": int(first_solve),
        "actions_vs_baseline": round(float(actions_vs_baseline), 4),
        "induced_mechanic": mechanic,
        "games_attempted": attempted_rows,
        "real_env_confirmed": bool(real_env_confirmed),
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": _duration(started),
        "inference_substrate": inference_substrate,
        "attempt_details": attempted_rows,
        "verifier_pruner_used": any(bool(attempt.pruner_decisions) for attempt in attempts),
        "precondition_blocked": False,
        "excluded_solved_games": list(SOLVED_GAME_PREFIXES),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
