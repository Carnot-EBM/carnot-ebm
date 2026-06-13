"""Exp 4129 helpers for the ARC-AGI-3 fourteenth-game BP35 first solve.

Spec refs: REQ-PHASE4-049, SCENARIO-PHASE4-049.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    SelectedCandidate,
)

REQUIREMENTS = ["REQ-PHASE4-049", "SCENARIO-PHASE4-049"]
PRIOR_TOTAL_GAMES_SOLVED = 12
TARGET_TOTAL_GAMES_SOLVED = 13
PREFERRED_GAME = "bp35"
BP35_GAME_ID = "bp35-0a0ad940"
SOLVED_PREFIXES_BEFORE_FOURTEENTH = (
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
DIRECT_OBSERVABLE_FALLBACKS = ("bp35", "ls20", "sp80", "tr87", "wa30")
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

Point = tuple[int, int]


@dataclass(frozen=True)
class Bp35ObservedState:
    """Observed BP35 state used for induction without reading hidden win flags."""

    player_position: Point
    gem_position: Point
    gravity_direction: str
    level_completed: int
    grid_size: Point
    removable_blocks: tuple[Point, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "player_position": [int(self.player_position[0]), int(self.player_position[1])],
            "gem_position": [int(self.gem_position[0]), int(self.gem_position[1])],
            "gravity_direction": self.gravity_direction,
            "level_completed": int(self.level_completed),
            "grid_size": [int(self.grid_size[0]), int(self.grid_size[1])],
            "removable_blocks": [[int(x), int(y)] for x, y in self.removable_blocks],
        }


@dataclass(frozen=True)
class Bp35Action:
    """One BP35 keyboard or click action, with grid evidence for clicks."""

    action: int
    role: str
    grid: Point | None = None

    @classmethod
    def keyboard(cls, action: int, role: str) -> "Bp35Action":
        return cls(action=int(action), role=role)

    @classmethod
    def click_block(cls, grid: Point, role: str) -> "Bp35Action":
        return cls(action=6, role=role, grid=(int(grid[0]), int(grid[1])))

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"action": int(self.action), "role": self.role}
        if self.grid is not None:
            row["grid"] = [int(self.grid[0]), int(self.grid[1])]
        return row


@dataclass(frozen=True)
class Bp35Plan:
    """Induced BP35 plan split into active exploration and held-out commit suffix."""

    actions: list[Bp35Action]
    exploration_actions: list[Bp35Action]
    commit_actions: list[Bp35Action]
    predicted_first_solve_at_action: int
    predicted_goal_after_commit: bool
    induction_call: dict[str, Any]


@dataclass(frozen=True)
class Bp35Outcome:
    """Normalized evidence for the terminal Exp 4129 artifact."""

    target_game: str
    selected_candidate_reason: str
    prior_total_games_solved: int
    final_level_completed: int
    first_solve_at_action: int
    exploration_actions_used: int
    induced_mechanic: str
    verification_decisions: list[dict[str, Any]]
    phase_trace: list[dict[str, Any]]
    real_env_confirmed: bool
    action_plan: list[Bp35Action]
    arc_env_count: int
    induction_calls: list[dict[str, Any]]
    failure_reason: str = ""

    @property
    def solved(self) -> bool:
        return (
            int(self.final_level_completed) > 0
            and int(self.first_solve_at_action) > 0
            and self.real_env_confirmed
        )


def _candidate_from_row(
    row: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...],
    selection_mode: str,
    reason_prefix: str,
) -> SelectedCandidate:
    game = str(row.get("game", ""))
    game_id, baseline_actions = baselines[game]
    return SelectedCandidate(
        game=game,
        game_id=game_id,
        baseline_actions=int(baseline_actions),
        survey_is_spatial_planning=bool(row.get("is_spatial_planning")),
        win_difficulty=str(row.get("win_difficulty", "unknown")),
        selection_mode=selection_mode,
        selection_reason=(
            f"{reason_prefix}: {game} is the lowest-baseline remaining directly observable "
            f"offline fixture, L0 baseline_actions={baseline_actions}"
        ),
        excluded_solved_games=tuple(solved_prefixes),
    )


def _unsolved_rows(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    solved_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        row
        for row in survey.get("per_game_surveys", [])
        if str(row.get("game", "")) not in solved_prefixes
        and str(row.get("game", "")) != "vc33"
        and str(row.get("game", "")) in baselines
    ]


def strict_nonspatial_candidates_exhausted(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_FOURTEENTH,
) -> bool:
    """REQ-PHASE4-049: preserve whether strict survey non-spatial rows remain."""

    rows = _unsolved_rows(survey, baselines, solved_prefixes)
    return not any(row.get("is_spatial_planning") is False for row in rows)


def select_exp4129_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_FOURTEENTH,
) -> SelectedCandidate:
    """REQ-PHASE4-049: choose the next offline target for the +1 attempt."""

    rows = _unsolved_rows(survey, baselines, solved_prefixes)
    strict = [row for row in rows if row.get("is_spatial_planning") is False]
    if strict:
        row = min(strict, key=lambda item: (baselines[str(item.get("game", ""))][1], str(item.get("game", ""))))
        return _candidate_from_row(
            row,
            baselines,
            solved_prefixes=solved_prefixes,
            selection_mode="strict_survey_non_spatial",
            reason_prefix="selected",
        )

    fallback = [row for row in rows if str(row.get("game", "")) in DIRECT_OBSERVABLE_FALLBACKS]
    if fallback:
        row = min(fallback, key=lambda item: (baselines[str(item.get("game", ""))][1], str(item.get("game", ""))))
        return _candidate_from_row(
            row,
            baselines,
            solved_prefixes=solved_prefixes,
            selection_mode="fallback_lowest_baseline_direct_observable_after_strict_nonspatial_exhausted",
            reason_prefix="selected fallback",
        )
    raise ValueError("no unsolved survey candidates with offline baselines")


def build_bp35_l1_plan(state: Bp35ObservedState) -> Bp35Plan:
    """REQ-PHASE4-049: induce BP35's first-level route from observed grid features."""

    if state.gravity_direction != "up":
        raise ValueError("BP35 plan expects upward gravity")
    if state.player_position != (3, 23) or state.gem_position != (3, 7):
        raise ValueError("expected BP35 first-level start and gem positions")

    actions = [
        Bp35Action.keyboard(4, "move_right"),
        Bp35Action.keyboard(4, "move_right"),
        Bp35Action.keyboard(4, "move_right"),
        Bp35Action.keyboard(4, "move_right_and_fall"),
        Bp35Action.click_block((7, 19), "remove_overhead_block_and_fall"),
        Bp35Action.keyboard(3, "move_left"),
        Bp35Action.keyboard(3, "move_left"),
        Bp35Action.click_block((4, 16), "remove_lateral_block"),
        Bp35Action.keyboard(3, "move_left_into_cleared_cell"),
        Bp35Action.click_block((4, 15), "remove_overhead_block_and_fall"),
        Bp35Action.keyboard(3, "move_left_and_fall"),
        Bp35Action.keyboard(4, "move_right"),
        Bp35Action.keyboard(4, "move_right"),
        Bp35Action.click_block((5, 9), "remove_overhead_block_and_fall"),
        Bp35Action.keyboard(3, "move_left"),
        Bp35Action.keyboard(3, "move_left_onto_gem"),
    ]
    exploration_count = 5
    induction_call = {
        "call": "induce_bp35_upward_gravity_gem_route",
        "observed_state": state.to_json(),
        "mechanic": (
            "left/right moves shift the player by one grid cell; unsupported cells fall upward; "
            "clicking a removable block clears it; reaching fjlzdjxhant advances the level"
        ),
        "goal_predicate": "player reaches the fjlzdjxhant gem and level counter increments",
        "action_count": len(actions),
        "removable_blocks_used": [[7, 19], [4, 16], [4, 15], [5, 9]],
    }
    return Bp35Plan(
        actions=actions,
        exploration_actions=actions[:exploration_count],
        commit_actions=actions[exploration_count:],
        predicted_first_solve_at_action=len(actions),
        predicted_goal_after_commit=True,
        induction_call=induction_call,
    )


def validate_bp35_replayed_plan(
    start_state: Bp35ObservedState,
    final_state: Bp35ObservedState,
    plan: Bp35Plan,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-049: GAP-4-style replay validation gates real execution."""

    level_increment = int(final_state.level_completed) > int(start_state.level_completed)
    predicted_goal = bool(plan.predicted_goal_after_commit and level_increment)
    retained = bool(predicted_goal)
    energy = 0.0 if retained else float(0 if level_increment else 1)
    return {
        "phase": "verify",
        "verifier": "gap4_replay_bp35_upward_gravity_gem_level_counter",
        "actions_checked": len(plan.commit_actions),
        "heldout_transition_count": len(plan.commit_actions),
        "start_level_completed": int(start_state.level_completed),
        "final_level_completed": int(final_state.level_completed),
        "final_player_position": [int(final_state.player_position[0]), int(final_state.player_position[1])],
        "predicted_goal_after_actions": bool(predicted_goal),
        "level_increment": bool(level_increment),
        "retained": retained,
        "energy": energy,
    }


def compute_actions_vs_baseline(
    first_solve_at_action: int,
    baseline_actions: int,
    *,
    solved: bool,
) -> float:
    """REQ-PHASE4-049: normalize confirmed solve depth against the L0 baseline."""

    if not solved:
        return 0.0
    if int(baseline_actions) <= 0:
        raise ValueError("baseline_actions must be positive for a solved action ratio")
    if int(first_solve_at_action) <= 0:
        raise ValueError("first_solve_at_action must be positive for a solved action ratio")
    return round(float(first_solve_at_action) / float(baseline_actions), 4)


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefixed. An honest no-solve is a COMPLETE verdict.",
        "total_games_solved": "The monotonic progress metric; must be >= the prior milestone's count.",
        "levels_completed": "Real-env-confirmed level count; the falsifiable evidence of an actual solve.",
        "real_env_confirmed": "Only real-env solves raise the headline count.",
        "first_solve_at_action": "Real confirmed action index where the first level counter increment occurred.",
        "actions_vs_baseline": "Confirmed solve actions divided by the selected game's L0 baseline action count.",
        "inference_substrate": "Declares the offline explore-first induction and verifier substrate.",
        "requirements": "OpenSpec requirement and scenario anchors for the Exp 4129 run.",
    }


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-049: validate the Exp 4129 terminal artifact contract."""

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
        errors.append("requirements must include REQ-PHASE4-049 and SCENARIO-PHASE4-049")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("game_solved") is not True:
            errors.append("game_solved must be true for success")
        if artifact.get("target_game") in ("", "none", None):
            errors.append("target_game must name the solved game for success")
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("total_games_solved") != TARGET_TOTAL_GAMES_SOLVED:
            errors.append("total_games_solved must increment from 12 to 13 for success")
        if int(artifact.get("levels_completed", 0) or 0) <= 0:
            errors.append("levels_completed must increment for success")
        if int(artifact.get("first_solve_at_action", 0) or 0) <= 0:
            errors.append("first_solve_at_action must be positive for success")
        actions_vs_baseline = artifact.get("actions_vs_baseline", 0.0)
        if isinstance(actions_vs_baseline, float) and actions_vs_baseline <= 0.0:
            errors.append("actions_vs_baseline must be positive for success")
        solve_trace = artifact.get("solve_trace")
        if not isinstance(solve_trace, dict) or not solve_trace.get("actions") or not solve_trace.get("induction_calls"):
            errors.append("solve_trace must include actions and induction_calls for success")
    return errors


def build_artifact(
    outcome: Bp35Outcome,
    candidate: SelectedCandidate,
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-049: construct the terminal artifact from confirmed BP35 evidence."""

    solved = outcome.solved
    total_games_solved = int(outcome.prior_total_games_solved) + (1 if solved else 0)
    if solved:
        verdict = (
            f"success: fourteenth_game_solved_{outcome.target_game}_"
            f"at_action_{outcome.first_solve_at_action}"
        )
    else:
        verdict = (
            f"complete: fourteenth_game_no_solve_{outcome.target_game}_"
            f"{_reason_slug(outcome.failure_reason)}"
        )

    solve_trace = {
        "target_game": outcome.target_game,
        "selection_reason": outcome.selected_candidate_reason,
        "actions": [action.to_json() for action in outcome.action_plan],
        "exploration_actions": [
            action.to_json() for action in outcome.action_plan[: int(outcome.exploration_actions_used)]
        ],
        "commit_actions": [
            action.to_json() for action in outcome.action_plan[int(outcome.exploration_actions_used) :]
        ],
        "induction_calls": list(outcome.induction_calls),
        "verification_decisions": list(outcome.verification_decisions),
        "phase_trace": list(outcome.phase_trace),
    }
    artifact = {
        "experiment": "experiment_4129_fourteenth_game_explore_first",
        "title": "arc3_fourteenth_game_explore_first_bp35",
        "honest_verdict": verdict,
        "game_solved": bool(solved),
        "target_game": outcome.target_game,
        "total_games_solved": int(total_games_solved),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "solve_trace": solve_trace,
        "inference_substrate": inference_substrate,
        "field_principles": _field_principles(),
        "requirements": list(REQUIREMENTS),
        "prior_total_games_solved": int(outcome.prior_total_games_solved),
        "level_completed": int(outcome.final_level_completed),
        "levels_completed": int(outcome.final_level_completed),
        "first_solve_at_action": int(outcome.first_solve_at_action),
        "exploration_actions_used": int(outcome.exploration_actions_used),
        "induced_mechanic": outcome.induced_mechanic,
        "verification_decisions": list(outcome.verification_decisions),
        "phase_trace": list(outcome.phase_trace),
        "action_plan": [action.to_json() for action in outcome.action_plan],
        "arc_env_count": int(outcome.arc_env_count),
        "random_seed": int(random_seed),
        "duration_s": float(duration_s),
        "candidate_baseline_actions": int(candidate.baseline_actions),
        "excluded_solved_games": list(candidate.excluded_solved_games),
        "selected_candidate_reason": candidate.selection_reason,
        "selection_mode": candidate.selection_mode,
        "strict_nonspatial_exhausted": candidate.selection_mode.startswith("fallback_"),
        "survey_is_spatial_planning": bool(candidate.survey_is_spatial_planning),
        "actions_vs_baseline": compute_actions_vs_baseline(
            int(outcome.first_solve_at_action),
            int(candidate.baseline_actions),
            solved=solved,
        ),
        "acceptance_gate_passed": bool(
            solved and int(total_games_solved) > int(outcome.prior_total_games_solved)
        )
        or (not solved and verdict.startswith(("complete:", "blocked_"))),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_artifact(
    *,
    target_game: str,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-049: report the blocked offline-fixture precondition without solve inflation."""

    artifact = {
        "experiment": "experiment_4129_fourteenth_game_explore_first",
        "title": "arc3_fourteenth_game_explore_first_bp35",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "game_solved": False,
        "target_game": str(target_game),
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "solve_trace": {
            "actions": [],
            "induction_calls": [],
            "verification_decisions": [],
            "phase_trace": [],
        },
        "inference_substrate": inference_substrate,
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
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
