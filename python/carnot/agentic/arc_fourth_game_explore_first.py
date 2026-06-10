"""Exp 4004 helpers for explore-first ARC-AGI-3 verifier pruning.

Spec refs: REQ-PHASE4-024, SCENARIO-PHASE4-024.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from carnot.agentic.arc_world_model_synth import InducedWorldModel, grade_predictions


REQUIRED_ARTIFACT_FIELDS = (
    "ACCURACY_levels_solved",
    "game_solved",
    "exploration_actions_used",
    "dynamics_induced",
    "first_solve_at_action",
    "actions_vs_baseline",
    "induced_mechanic",
    "games_attempted",
    "real_env_confirmed",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)


@dataclass(frozen=True)
class CandidateGame:
    game_id: str
    baseline_actions: int
    non_spatial: bool
    directly_observable_goal: bool
    survey_reason: str
    selection_reason: str = ""


@dataclass(frozen=True)
class TransitionObservation:
    before: np.ndarray
    action_key: tuple[int, ...]
    after: np.ndarray
    level_delta: int
    game_over: bool


@dataclass(frozen=True)
class AttemptResult:
    game_id: str
    baseline_actions: int
    target_selection_reason: str
    exploration_actions_used: int
    dynamics_induced: bool
    first_solve_at_action: int
    levels_completed: int
    actions_vs_baseline: float
    induced_mechanic: str
    real_env_confirmed: bool
    observed_dynamics: list[dict[str, Any]]
    pruner_decisions: list[dict[str, Any]]
    solve_log: list[dict[str, Any]]
    failure_reason: str = ""

    @property
    def solved(self) -> bool:
        return self.levels_completed > 0 and self.first_solve_at_action >= 0


def select_candidate_order(candidates: list[CandidateGame]) -> list[CandidateGame]:
    """REQ-PHASE4-024: reject spatial targets, then sort by L0 baseline."""
    eligible = [
        item
        for item in candidates
        if item.non_spatial and item.directly_observable_goal
    ]
    ordered = sorted(eligible, key=lambda item: (item.baseline_actions, item.game_id))
    return [
        replace(
            item,
            selection_reason=(
                "selected: non-spatial directly-observable target; "
                f"L0 baseline_actions={item.baseline_actions}; {item.survey_reason}"
            ),
        )
        for item in ordered
    ]


def induce_model_from_observations(
    game_id: str,
    observations: list[TransitionObservation],
) -> InducedWorldModel:
    """SCENARIO-PHASE4-024: build the grounded model only from observed transitions."""
    if not observations:
        raise ValueError("at least one observed transition is required before induction")
    transitions = [
        (obs.before, obs.action_key, obs.after)
        for obs in observations
    ]
    return InducedWorldModel(game_id).fit(transitions)


def prune_candidates_after_induction(
    model: InducedWorldModel | None,
    current_grid: np.ndarray,
    candidates: list[tuple[tuple[int, ...], np.ndarray]],
    *,
    energy_threshold: float,
) -> list[dict[str, Any]]:
    """REQ-PHASE4-024: apply executed-consistency pruning only after induction."""
    if model is None or int(getattr(model, "n_train", 0) or 0) <= 0:
        raise ValueError("an induced model with observed transitions is required before pruning")

    decisions: list[dict[str, Any]] = []
    for action_key, observed_next in candidates:
        graded = grade_predictions(model.predict, [(current_grid, action_key, observed_next)])
        energy = graded.get("energy")
        retained = energy is not None and float(energy) <= energy_threshold
        decisions.append(
            {
                "action_key": [int(v) for v in action_key],
                "energy": None if energy is None else float(energy),
                "retained": bool(retained),
                "reason": "executed-consistency" if retained else "low-consistency-or-no-dynamics",
            }
        )
    return decisions


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
        "experiment": "experiment_4004_fourth_game_explore_first",
        "title": "arc3_fourth_game_explore_first_verifier_pruned",
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
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_fourth_game_artifact(
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
        verdict = f"success: fourth_game_solved_{game_solved}_at_action{first_solve}"
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
        verdict = f"complete: fourth_game_no_solve_{reason_slug}"

    artifact = {
        "experiment": "experiment_4004_fourth_game_explore_first",
        "title": "arc3_fourth_game_explore_first_verifier_pruned",
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
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    int_fields = ("ACCURACY_levels_solved", "exploration_actions_used", "first_solve_at_action", "random_seed")
    for field in int_fields:
        if field in artifact and not isinstance(artifact[field], int):
            errors.append(f"{field} must be a bare int")

    str_fields = ("game_solved", "induced_mechanic", "honest_verdict", "inference_substrate")
    for field in str_fields:
        if field in artifact and not isinstance(artifact[field], str):
            errors.append(f"{field} must be a string")

    bool_fields = ("dynamics_induced", "real_env_confirmed")
    for field in bool_fields:
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")

    if "actions_vs_baseline" in artifact and not isinstance(artifact["actions_vs_baseline"], float):
        errors.append("actions_vs_baseline must be a bare float")
    if "duration_s" in artifact and not isinstance(artifact["duration_s"], float):
        errors.append("duration_s must be a bare float")
    if "games_attempted" in artifact and not isinstance(artifact["games_attempted"], list):
        errors.append("games_attempted must be a list")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str):
        if not (verdict.startswith("success:") or verdict.startswith("complete:") or verdict.startswith("blocked_")):
            errors.append("honest_verdict must start with success:, complete:, or blocked_")
        if not verdict.startswith("blocked_") and artifact.get("games_attempted"):
            if artifact.get("exploration_actions_used", 0) <= 0:
                errors.append("exploration_actions_used must be >0 for non-blocked attempts")
            if artifact.get("dynamics_induced") is not True:
                errors.append("dynamics_induced must be true for non-blocked attempts")
        if verdict.startswith("success:"):
            if artifact.get("exploration_actions_used", 0) <= 0:
                errors.append("exploration_actions_used must be >0 for success")
            if artifact.get("dynamics_induced") is not True:
                errors.append("dynamics_induced must be true for success")
            if artifact.get("real_env_confirmed") is not True:
                errors.append("real_env_confirmed must be true for success")
            if artifact.get("first_solve_at_action", -1) <= 0:
                errors.append("first_solve_at_action must be >0 for success")
    return errors
