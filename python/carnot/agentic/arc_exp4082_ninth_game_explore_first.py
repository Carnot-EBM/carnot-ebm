"""Exp 4082 helpers for the ARC-AGI-3 ninth-game explore-first retry.

Spec refs: REQ-PHASE4-044, SCENARIO-PHASE4-044.
"""

from __future__ import annotations

from typing import Any

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    SOLVED_GAME_PREFIXES,
    ExperimentOutcome,
    SelectedCandidate,
    artifact_schema_errors as _base_artifact_schema_errors,
    blocked_artifact as _base_blocked_artifact,
    build_artifact as _base_build_artifact,
    select_ninth_candidate_from_survey,
)

REQUIREMENTS = ["REQ-PHASE4-044", "SCENARIO-PHASE4-044"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "game_solved",
    "total_games_solved",
    "first_solve_at_action",
    "actions_vs_baseline",
    "real_env_confirmed",
    "inference_substrate",
)


def select_exp4082_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_GAME_PREFIXES,
) -> SelectedCandidate:
    """REQ-PHASE4-044: choose the 4082 target using the established explore-first filter."""

    return select_ninth_candidate_from_survey(
        survey,
        baselines,
        solved_prefixes=solved_prefixes,
    )


def compute_actions_vs_baseline(
    first_solve_at_action: int,
    baseline_actions: int,
    *,
    solved: bool,
) -> float:
    """REQ-PHASE4-044: normalize confirmed solve depth against the L0 action baseline."""

    if not solved:
        return 0.0
    if int(baseline_actions) <= 0:
        raise ValueError("baseline_actions must be positive for a solved action ratio")
    if int(first_solve_at_action) <= 0:
        raise ValueError("first_solve_at_action must be positive for a solved action ratio")
    return round(float(first_solve_at_action) / float(baseline_actions), 4)


def _field_principles(artifact: dict[str, Any]) -> dict[str, str]:
    principles = dict(artifact.get("field_principles", {}))
    principles.update(
        {
            "first_solve_at_action": "real confirmed action index where the first level counter increment occurred",
            "actions_vs_baseline": "confirmed solve actions divided by the selected game's L0 baseline action count",
            "requirements": "OpenSpec requirement and scenario anchors for the Exp 4082 retry",
        }
    )
    return principles


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-044: validate the Exp 4082 terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    for error in _base_artifact_schema_errors(artifact):
        if error not in errors:
            errors.append(error)

    if "first_solve_at_action" in artifact and type(artifact["first_solve_at_action"]) is not int:
        errors.append("first_solve_at_action must be a bare int")
    if "actions_vs_baseline" in artifact and type(artifact["actions_vs_baseline"]) is not float:
        errors.append("actions_vs_baseline must be a bare float")
    if "inference_substrate" in artifact and artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if "requirements" in artifact and not all(req in artifact["requirements"] for req in REQUIREMENTS):
        errors.append("requirements must include REQ-PHASE4-044 and SCENARIO-PHASE4-044")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if int(artifact.get("first_solve_at_action", 0) or 0) <= 0:
            errors.append("first_solve_at_action must be positive for success")
        actions_vs_baseline = artifact.get("actions_vs_baseline", 0.0)
        if isinstance(actions_vs_baseline, float) and actions_vs_baseline <= 0.0:
            errors.append("actions_vs_baseline must be positive for success")
    return errors


def build_artifact(
    outcome: ExperimentOutcome,
    candidate: SelectedCandidate,
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-044: convert confirmed ft09 evidence into the Exp 4082 artifact."""

    artifact = _base_build_artifact(
        outcome,
        random_seed=random_seed,
        duration_s=duration_s,
        inference_substrate=inference_substrate,
    )
    artifact.update(
        {
            "experiment": "experiment_4082_ninth_game_explore_first",
            "title": "arc3_ninth_game_explore_first_retry_ft09",
            "requirements": list(REQUIREMENTS),
            "candidate_baseline_actions": int(candidate.baseline_actions),
            "excluded_solved_games": list(candidate.excluded_solved_games),
            "selected_candidate_reason": candidate.selection_reason,
            "selection_mode": candidate.selection_mode,
            "survey_is_spatial_planning": bool(candidate.survey_is_spatial_planning),
            "actions_vs_baseline": compute_actions_vs_baseline(
                int(outcome.first_solve_at_action),
                int(candidate.baseline_actions),
                solved=outcome.solved,
            ),
        }
    )
    artifact["field_principles"] = _field_principles(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_artifact(
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-044: report the blocked live-ARC precondition without solve inflation."""

    artifact = _base_blocked_artifact(
        random_seed=random_seed,
        duration_s=duration_s,
        inference_substrate=inference_substrate,
    )
    artifact.update(
        {
            "experiment": "experiment_4082_ninth_game_explore_first",
            "title": "arc3_ninth_game_explore_first_retry_ft09",
            "requirements": list(REQUIREMENTS),
            "candidate_baseline_actions": 0,
            "actions_vs_baseline": 0.0,
        }
    )
    artifact["field_principles"] = _field_principles(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
