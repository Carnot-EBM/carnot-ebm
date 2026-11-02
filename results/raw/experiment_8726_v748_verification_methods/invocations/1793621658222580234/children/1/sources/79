"""Exp 4120 helpers for the ARC-AGI-3 thirteenth-game strict non-spatial attempt.

Spec refs: REQ-PHASE4-048, SCENARIO-PHASE4-048.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    SelectedCandidate,
)

REQUIREMENTS = ["REQ-PHASE4-048", "SCENARIO-PHASE4-048"]
PRIOR_TOTAL_GAMES_SOLVED = 12
TARGET_TOTAL_GAMES_SOLVED = 13
SOLVED_PREFIXES_BEFORE_THIRTEENTH = (
    "r11l",
    "lp85",
    "sc25",
    "su15",
    "tn36",
    "cd82",
    "dc22",
    "sb26",
    "ft09",
    "s5i5",
    "tu93",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "game_solved",
    "target_game",
    "total_games_solved",
    "levels_completed",
    "first_solve_at_action",
    "actions_vs_baseline",
    "real_env_confirmed",
    "solve_trace",
    "inference_substrate",
)
STRICT_EXHAUSTION_VERDICT = "complete: thirteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates"


@dataclass(frozen=True)
class SurveyCandidate:
    """One survey row joined with its local offline baseline metadata."""

    game: str
    game_id: str
    baseline_actions: int
    is_spatial_planning: bool
    win_difficulty: str

    def to_json(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "game_id": self.game_id,
            "baseline_actions": int(self.baseline_actions),
            "is_spatial_planning": bool(self.is_spatial_planning),
            "win_difficulty": self.win_difficulty,
        }


@dataclass(frozen=True)
class SelectionReport:
    """Auditable Exp 4120 candidate filtering result before any environment action."""

    solved_prefixes: tuple[str, ...]
    strict_nonspatial_candidates: tuple[SurveyCandidate, ...]
    unsolved_strict_nonspatial_candidates: tuple[SurveyCandidate, ...]
    remaining_unsolved_offline_candidates: tuple[SurveyCandidate, ...]
    offline_candidate_count: int

    @property
    def no_unsolved_strict_nonspatial_candidates(self) -> bool:
        return len(self.unsolved_strict_nonspatial_candidates) == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "solved_prefixes": list(self.solved_prefixes),
            "strict_nonspatial_candidates": [
                candidate.to_json() for candidate in self.strict_nonspatial_candidates
            ],
            "unsolved_strict_nonspatial_candidates": [
                candidate.to_json() for candidate in self.unsolved_strict_nonspatial_candidates
            ],
            "remaining_unsolved_offline_candidates": [
                candidate.to_json() for candidate in self.remaining_unsolved_offline_candidates
            ],
            "offline_candidate_count": int(self.offline_candidate_count),
            "no_unsolved_strict_nonspatial_candidates": self.no_unsolved_strict_nonspatial_candidates,
        }


class NoUnsolvedNonSpatialCandidate(ValueError):
    """Raised when Exp 4120's strict non-spatial target set is exhausted."""

    def __init__(self, report: SelectionReport) -> None:
        super().__init__("no unsolved strict non-spatial candidates remain")
        self.report = report


def _survey_candidate_from_row(
    row: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
) -> SurveyCandidate:
    game = str(row.get("game", ""))
    game_id, baseline_actions = baselines[game]
    return SurveyCandidate(
        game=game,
        game_id=game_id,
        baseline_actions=int(baseline_actions),
        is_spatial_planning=bool(row.get("is_spatial_planning")),
        win_difficulty=str(row.get("win_difficulty", "unknown")),
    )


def build_selection_report(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_THIRTEENTH,
) -> SelectionReport:
    """REQ-PHASE4-048: filter the local survey for unsolved strict non-spatial targets."""

    candidates = [
        _survey_candidate_from_row(row, baselines)
        for row in survey.get("per_game_surveys", [])
        if str(row.get("game", "")) != "vc33" and str(row.get("game", "")) in baselines
    ]
    strict_nonspatial = tuple(
        sorted(
            (candidate for candidate in candidates if candidate.is_spatial_planning is False),
            key=lambda candidate: candidate.game,
        )
    )
    unsolved_strict = tuple(
        sorted(
            (candidate for candidate in strict_nonspatial if candidate.game not in solved_prefixes),
            key=lambda candidate: (candidate.baseline_actions, candidate.game),
        )
    )
    remaining_unsolved = tuple(
        sorted(
            (candidate for candidate in candidates if candidate.game not in solved_prefixes),
            key=lambda candidate: (candidate.baseline_actions, candidate.game),
        )
    )
    return SelectionReport(
        solved_prefixes=tuple(solved_prefixes),
        strict_nonspatial_candidates=strict_nonspatial,
        unsolved_strict_nonspatial_candidates=unsolved_strict,
        remaining_unsolved_offline_candidates=remaining_unsolved,
        offline_candidate_count=len(candidates),
    )


def select_exp4120_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_THIRTEENTH,
) -> SelectedCandidate:
    """REQ-PHASE4-048: choose only an unsolved strict non-spatial candidate."""

    report = build_selection_report(survey, baselines, solved_prefixes=solved_prefixes)
    if report.no_unsolved_strict_nonspatial_candidates:
        raise NoUnsolvedNonSpatialCandidate(report)
    candidate = report.unsolved_strict_nonspatial_candidates[0]
    return SelectedCandidate(
        game=candidate.game,
        game_id=candidate.game_id,
        baseline_actions=int(candidate.baseline_actions),
        survey_is_spatial_planning=False,
        win_difficulty=candidate.win_difficulty,
        selection_mode="strict_survey_non_spatial",
        selection_reason=(
            f"selected: {candidate.game} is the lowest-baseline unsolved strict non-spatial "
            f"offline fixture, L0 baseline_actions={candidate.baseline_actions}"
        ),
        excluded_solved_games=tuple(solved_prefixes),
    )


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefixed. An honest no-solve is a COMPLETE verdict.",
        "total_games_solved": "The monotonic progress metric; must be >= the prior milestone's count.",
        "levels_completed": "Real-env-confirmed level count; the falsifiable evidence of an actual solve.",
        "real_env_confirmed": "Only real-env solves raise the headline count.",
        "first_solve_at_action": "Real confirmed action index where the first level counter increment occurred.",
        "actions_vs_baseline": "Confirmed solve actions divided by the selected game's L0 baseline action count.",
        "inference_substrate": "Declares the offline explore-first induction and verifier substrate.",
        "requirements": "OpenSpec requirement and scenario anchors for the Exp 4120 run.",
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-048: validate the Exp 4120 terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if "honest_verdict" in artifact:
        if not isinstance(verdict, str):
            errors.append("honest_verdict must be a string")
        elif not (
            verdict.startswith("success:")
            or verdict.startswith("complete:")
            or verdict.startswith("blocked_")
        ):
            errors.append("honest_verdict must start with success:, complete:, or blocked_")

    if "game_solved" in artifact and type(artifact["game_solved"]) is not bool:
        errors.append("game_solved must be a bare bool")
    if "target_game" in artifact and not isinstance(artifact["target_game"], str):
        errors.append("target_game must be a string")
    if "total_games_solved" in artifact and type(artifact["total_games_solved"]) is not int:
        errors.append("total_games_solved must be a bare int")
    if "levels_completed" in artifact and type(artifact["levels_completed"]) is not int:
        errors.append("levels_completed must be a bare int")
    if "first_solve_at_action" in artifact and type(artifact["first_solve_at_action"]) is not int:
        errors.append("first_solve_at_action must be a bare int")
    if "actions_vs_baseline" in artifact and type(artifact["actions_vs_baseline"]) is not float:
        errors.append("actions_vs_baseline must be a bare float")
    if "real_env_confirmed" in artifact and type(artifact["real_env_confirmed"]) is not bool:
        errors.append("real_env_confirmed must be a bare bool")
    if "solve_trace" in artifact and not isinstance(artifact["solve_trace"], dict):
        errors.append("solve_trace must be a dict")
    if "inference_substrate" in artifact and artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if "requirements" in artifact and not all(req in artifact["requirements"] for req in REQUIREMENTS):
        errors.append("requirements must include REQ-PHASE4-048 and SCENARIO-PHASE4-048")

    if verdict == STRICT_EXHAUSTION_VERDICT:
        if artifact.get("game_solved") is not False:
            errors.append("game_solved must be false for strict non-spatial exhaustion")
        if artifact.get("target_game") != "none":
            errors.append("target_game must be none for strict non-spatial exhaustion")
        if artifact.get("total_games_solved") != PRIOR_TOTAL_GAMES_SOLVED:
            errors.append("total_games_solved must remain at 12 for no-solve")
        if artifact.get("levels_completed") != 0:
            errors.append("levels_completed must be zero for no-solve")
        if artifact.get("first_solve_at_action") != -1:
            errors.append("first_solve_at_action must be -1 for no-solve")
        if artifact.get("actions_vs_baseline") != 0.0:
            errors.append("actions_vs_baseline must be 0.0 for no-solve")
        if artifact.get("real_env_confirmed") is not False:
            errors.append("real_env_confirmed must be false for no-solve")
    return errors


def build_no_solve_artifact(
    report: SelectionReport,
    *,
    random_seed: int,
    duration_s: float,
    offline_driver_available: bool,
    arc_env_count: int,
) -> dict[str, Any]:
    """REQ-PHASE4-048: construct the honest terminal artifact when no strict target remains."""

    artifact = {
        "experiment": "experiment_4120_thirteenth_game_explore_first",
        "title": "arc3_thirteenth_game_strict_nonspatial_explore_first",
        "honest_verdict": STRICT_EXHAUSTION_VERDICT,
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "solve_trace": {
            "actions": [],
            "exploration_actions": [],
            "commit_actions": [],
            "induction_calls": [],
            "verification_decisions": [],
            "phase_trace": [
                {
                    "phase": "select",
                    "source": "offline_survey_strict_nonspatial_filter",
                    "selection_report": report.to_json(),
                },
                {
                    "phase": "stop",
                    "reason": "no_unsolved_strict_nonspatial_candidates",
                    "acted_in_environment": False,
                },
            ],
            "selection_report": report.to_json(),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "requirements": list(REQUIREMENTS),
        "prior_total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "level_completed": 0,
        "levels_completed": 0,
        "first_solve_at_action": -1,
        "exploration_actions_used": 0,
        "induced_mechanic": "none_no_unsolved_strict_nonspatial_candidate",
        "verification_decisions": [],
        "phase_trace": [],
        "action_plan": [],
        "arc_env_count": int(arc_env_count),
        "random_seed": int(random_seed),
        "duration_s": float(duration_s),
        "candidate_baseline_actions": 0,
        "actions_vs_baseline": 0.0,
        "nonspatial_candidates_exhausted": report.no_unsolved_strict_nonspatial_candidates,
        "remaining_unsolved_offline_candidates": [
            candidate.to_json() for candidate in report.remaining_unsolved_offline_candidates
        ],
        "strict_nonspatial_candidates": [
            candidate.to_json() for candidate in report.strict_nonspatial_candidates
        ],
        "offline_driver_available": bool(offline_driver_available),
        "acceptance_gate_passed": True,
        "failure_reason": "no_unsolved_strict_nonspatial_candidates",
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_artifact(*, random_seed: int, duration_s: float, reason: str) -> dict[str, Any]:
    """REQ-PHASE4-048: report the blocked offline-fixture precondition without solve inflation."""

    artifact = {
        "experiment": "experiment_4120_thirteenth_game_explore_first",
        "title": "arc3_thirteenth_game_strict_nonspatial_explore_first",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "solve_trace": {
            "actions": [],
            "induction_calls": [],
            "verification_decisions": [],
            "phase_trace": [{"phase": "stop", "reason": reason}],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "requirements": list(REQUIREMENTS),
        "prior_total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "level_completed": 0,
        "levels_completed": 0,
        "first_solve_at_action": -1,
        "exploration_actions_used": 0,
        "induced_mechanic": "none",
        "verification_decisions": [],
        "phase_trace": [],
        "action_plan": [],
        "arc_env_count": 0,
        "random_seed": int(random_seed),
        "duration_s": float(duration_s),
        "candidate_baseline_actions": 0,
        "actions_vs_baseline": 0.0,
        "acceptance_gate_passed": True,
        "failure_reason": reason,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
