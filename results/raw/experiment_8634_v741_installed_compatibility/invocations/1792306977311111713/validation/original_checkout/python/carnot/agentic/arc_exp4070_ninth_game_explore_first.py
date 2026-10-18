"""Exp 4070 helpers for the ninth ARC-AGI-3 explore-first solve.

Spec refs: REQ-PHASE4-042, SCENARIO-PHASE4-042.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INFERENCE_SUBSTRATE = "offline_arc_agi3_explore_first_first_solve"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "game_solved",
    "target_game",
    "total_games_solved",
    "real_env_confirmed",
    "solve_trace",
    "inference_substrate",
)
SOLVED_GAME_PREFIXES = ("r11l", "lp85", "sc25", "su15", "tn36", "cd82", "dc22", "sb26")
PRIOR_TOTAL_GAMES_SOLVED = 8


@dataclass(frozen=True)
class SelectedCandidate:
    """Survey row selected for the next one-game ARC-AGI-3 increment."""

    game: str
    game_id: str
    baseline_actions: int
    survey_is_spatial_planning: bool
    win_difficulty: str
    selection_mode: str
    selection_reason: str
    excluded_solved_games: tuple[str, ...]


@dataclass(frozen=True)
class Ft09Action:
    """Compact click action record for the Exp 4070 trace."""

    action: int
    x: int
    y: int
    grid: tuple[int, int]
    role: str = "cycle_cell"
    target_color: int | None = None

    @classmethod
    def click_cell(
        cls,
        grid: tuple[int, int],
        *,
        target_color: int | None = None,
        role: str = "cycle_cell",
    ) -> "Ft09Action":
        return cls(
            action=6,
            x=int(grid[0]) * 2,
            y=int(grid[1]) * 2,
            grid=(int(grid[0]), int(grid[1])),
            role=role,
            target_color=None if target_color is None else int(target_color),
        )

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "action": int(self.action),
            "x": int(self.x),
            "y": int(self.y),
            "grid": [int(self.grid[0]), int(self.grid[1])],
        }
        if self.role:
            row["role"] = self.role
        if self.target_color is not None:
            row["target_color"] = int(self.target_color)
        return row


@dataclass(frozen=True)
class Ft09Cell:
    """Clickable ft09 color cell, represented by its grid position and center color."""

    grid: tuple[int, int]
    color: int
    kind: str

    def to_json(self) -> dict[str, Any]:
        return {
            "grid": [int(self.grid[0]), int(self.grid[1])],
            "color": int(self.color),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Ft09Constraint:
    """ft09 bsT local rule: zero pixels require equality, non-zero pixels require inequality."""

    grid: tuple[int, int]
    center_color: int
    pattern: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

    def required_neighbors(self) -> list[tuple[tuple[int, int], str, int]]:
        required: list[tuple[tuple[int, int], str, int]] = []
        for row_index, row in enumerate(self.pattern):
            for col_index, value in enumerate(row):
                if row_index == 1 and col_index == 1:
                    continue
                grid = (
                    int(self.grid[0]) + (col_index - 1) * 4,
                    int(self.grid[1]) + (row_index - 1) * 4,
                )
                relation = "equal" if int(value) == 0 else "not_equal"
                required.append((grid, relation, int(self.center_color)))
        return required

    def to_json(self) -> dict[str, Any]:
        return {
            "grid": [int(self.grid[0]), int(self.grid[1])],
            "center_color": int(self.center_color),
            "pattern": [[int(value) for value in row] for row in self.pattern],
        }


@dataclass(frozen=True)
class Ft09ObservedState:
    """Small verifier state: constraints, clickable cells, color cycle, and level counter."""

    constraints: tuple[Ft09Constraint, ...]
    cells: tuple[Ft09Cell, ...]
    color_cycle: tuple[int, ...]
    level_completed: int

    @property
    def cell_colors(self) -> dict[tuple[int, int], int]:
        return {tuple(cell.grid): int(cell.color) for cell in self.cells}

    @property
    def violation_count(self) -> int:
        violations = 0
        colors = self.cell_colors
        for constraint in self.constraints:
            for grid, relation, color in constraint.required_neighbors():
                if grid not in colors:
                    continue
                observed = colors[grid]
                if relation == "equal" and observed != color:
                    violations += 1
                if relation == "not_equal" and observed == color:
                    violations += 1
        return violations

    def to_json(self) -> dict[str, Any]:
        return {
            "constraints": [constraint.to_json() for constraint in self.constraints],
            "cells": [cell.to_json() for cell in self.cells],
            "cell_colors": {
                f"{int(grid[0])},{int(grid[1])}": int(color)
                for grid, color in sorted(self.cell_colors.items())
            },
            "color_cycle": [int(color) for color in self.color_cycle],
            "violation_count": int(self.violation_count),
            "level_completed": int(self.level_completed),
        }


@dataclass(frozen=True)
class Ft09Plan:
    """Induced ft09 L1 plan split into exploration and commit suffix."""

    actions: list[Ft09Action]
    exploration_actions: list[Ft09Action]
    commit_actions: list[Ft09Action]
    predicted_cell_colors: dict[tuple[int, int], int]
    predicted_goal_after_commit: bool
    induction_call: dict[str, Any]


@dataclass(frozen=True)
class ExperimentOutcome:
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
    action_plan: list[Ft09Action]
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


def load_environment_baselines(environment_dir: Path) -> dict[str, tuple[str, int]]:
    """REQ-PHASE4-042: read local ARC metadata so selection uses measured L0 baselines."""

    baselines: dict[str, tuple[str, int]] = {}
    for metadata_path in sorted(environment_dir.glob("*/*/metadata.json")):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline_values = data.get("baseline_actions") or []
        if not baseline_values:
            continue
        game_id = str(data["game_id"])
        game = game_id.split("-", 1)[0]
        baselines[game] = (game_id, int(baseline_values[0]))
    return baselines


def _is_click_local_constraint_non_navigation(row: dict[str, Any]) -> bool:
    actions = str(row.get("available_actions", "")).lower()
    recipe = str(row.get("first_solve_recipe", "")).lower()
    summary = str(row.get("win_condition_summary", "")).lower()
    text = f"{recipe} {summary}"
    if "click-only" not in actions:
        return False
    if "constraint" not in text and "csp" not in text:
        return False
    blocked_terms = (
        "drag",
        "resize",
        "gravity",
        "pathfinding",
        "pushing",
        "sokoban",
        "collision",
        "navigate",
        "move sprites",
    )
    return not any(term in text for term in blocked_terms)


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
            f"{reason_prefix}: {game} is unsolved, win_difficulty="
            f"{row.get('win_difficulty', 'unknown')}, L0 baseline_actions={baseline_actions}"
        ),
        excluded_solved_games=tuple(solved_prefixes),
    )


def select_ninth_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_GAME_PREFIXES,
) -> SelectedCandidate:
    """REQ-PHASE4-042: choose strict non-spatial first, then the safest local-constraint fallback."""

    rows = [
        row
        for row in survey.get("per_game_surveys", [])
        if str(row.get("game", "")) not in solved_prefixes
        and str(row.get("game", "")) != "vc33"
        and str(row.get("game", "")) in baselines
    ]
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

    fallback = [row for row in rows if _is_click_local_constraint_non_navigation(row)]
    if fallback:
        row = min(fallback, key=lambda item: (baselines[str(item.get("game", ""))][1], str(item.get("game", ""))))
        return _candidate_from_row(
            row,
            baselines,
            solved_prefixes=solved_prefixes,
            selection_mode="fallback_click_local_constraint_non_navigation",
            reason_prefix="selected fallback",
        )
    raise ValueError("no unsolved non-spatial or local-constraint non-navigation survey candidates")


def _next_color(current: int, color_cycle: tuple[int, ...]) -> int:
    index = color_cycle.index(int(current))
    return int(color_cycle[(index + 1) % len(color_cycle)])


def build_ft09_l1_plan(state: Ft09ObservedState) -> Ft09Plan:
    """REQ-PHASE4-042: induce the ft09 local-constraint color-cycle plan from observations."""

    if len(state.color_cycle) < 2:
        raise ValueError("color_cycle must contain at least two colors")

    predicted_colors = dict(state.cell_colors)
    actions: list[Ft09Action] = []
    cell_by_grid = {tuple(cell.grid): cell for cell in state.cells}
    for constraint in state.constraints:
        for grid, relation, target_color in constraint.required_neighbors():
            if grid not in cell_by_grid:
                if relation == "equal":
                    raise ValueError(f"missing clickable cell at {grid}")
                continue
            current = predicted_colors[grid]
            if relation == "equal" and current != target_color:
                while current != target_color:
                    current = _next_color(current, state.color_cycle)
                    actions.append(Ft09Action.click_cell(grid, target_color=target_color))
                predicted_colors[grid] = current
            if relation == "not_equal" and current == target_color:
                while current == target_color:
                    current = _next_color(current, state.color_cycle)
                    actions.append(Ft09Action.click_cell(grid, target_color=current))
                predicted_colors[grid] = current

    predicted_state = Ft09ObservedState(
        constraints=state.constraints,
        cells=tuple(
            Ft09Cell(grid=cell.grid, color=predicted_colors[tuple(cell.grid)], kind=cell.kind)
            for cell in state.cells
        ),
        color_cycle=state.color_cycle,
        level_completed=state.level_completed,
    )
    induction_call = {
        "call": "induce_ft09_local_constraint_color_cycle",
        "observed_constraints": [constraint.to_json() for constraint in state.constraints],
        "observed_color_cycle": [int(color) for color in state.color_cycle],
        "mechanic": "clicking a visible Hkx cell cycles its center color through the level color cycle",
        "goal_predicate": "all bsT neighbor equality/inequality constraints hold",
    }
    return Ft09Plan(
        actions=actions,
        exploration_actions=actions[:1],
        commit_actions=actions[1:],
        predicted_cell_colors=predicted_colors,
        predicted_goal_after_commit=predicted_state.violation_count == 0,
        induction_call=induction_call,
    )


def validate_replayed_plan(
    start_state: Ft09ObservedState,
    final_state: Ft09ObservedState,
    plan: Ft09Plan,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-042: GAP-4-style replay validation gates live execution."""

    predicted_matches = dict(plan.predicted_cell_colors) == dict(final_state.cell_colors)
    final_violation_count = int(final_state.violation_count)
    level_increment = int(final_state.level_completed) > int(start_state.level_completed)
    retained = bool(predicted_matches and final_violation_count == 0 and level_increment)
    energy = 0.0 if retained else float(final_violation_count + (0 if level_increment else 1))
    return {
        "phase": "verify",
        "verifier": "gap4_replay_ft09_local_constraint_level_counter",
        "actions_checked": len(plan.commit_actions),
        "start_violation_count": int(start_state.violation_count),
        "final_violation_count": final_violation_count,
        "predicted_cell_colors_match": bool(predicted_matches),
        "predicted_goal_after_actions": bool(predicted_matches and final_violation_count == 0),
        "level_increment": bool(level_increment),
        "retained": retained,
        "energy": energy,
    }


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "terminal prefix records success, complete no-solve, or blocked precondition",
        "game_solved": "bare boolean for whether this task added exactly one new solved game",
        "target_game": "auditable game id selected from the survey and environment metadata",
        "total_games_solved": "monotonic ARC accuracy counter and north-star metric",
        "real_env_confirmed": "a solve is real only when the live environment level counter confirms it",
        "solve_trace": "full observe/explore/induce/verify/act trace for Exp 4072 concept-transfer analysis",
        "inference_substrate": "declares the offline explore-first induction and verifier substrate",
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-042: validate required terminal artifact fields."""

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
    if "real_env_confirmed" in artifact and type(artifact["real_env_confirmed"]) is not bool:
        errors.append("real_env_confirmed must be a bare bool")
    if "solve_trace" in artifact and not isinstance(artifact["solve_trace"], dict):
        errors.append("solve_trace must be a dict")
    if "inference_substrate" in artifact and not isinstance(artifact["inference_substrate"], str):
        errors.append("inference_substrate must be a string")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("game_solved") is not True:
            errors.append("game_solved must be true for success")
        if artifact.get("target_game") in ("", "none", None):
            errors.append("target_game must name the solved game for success")
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("total_games_solved") != PRIOR_TOTAL_GAMES_SOLVED + 1:
            errors.append("total_games_solved must increment by one for success")
        if int(artifact.get("level_completed", 0) or 0) <= 0:
            errors.append("level_completed must increment for success")
        if int(artifact.get("first_solve_at_action", 0) or 0) <= 0:
            errors.append("first_solve_at_action must be positive for success")
        if int(artifact.get("exploration_actions_used", 0) or 0) <= 0:
            errors.append("exploration_actions_used must be positive for success")
        solve_trace = artifact.get("solve_trace")
        if not isinstance(solve_trace, dict) or not solve_trace.get("actions") or not solve_trace.get("induction_calls"):
            errors.append("solve_trace must include actions and induction_calls for success")
    return errors


def build_artifact(
    outcome: ExperimentOutcome,
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-042: construct the terminal artifact from confirmed outcome evidence."""

    solved = outcome.solved
    total_games_solved = int(outcome.prior_total_games_solved) + (1 if solved else 0)
    if solved:
        verdict = (
            f"success: ninth_game_solved_{outcome.target_game}_"
            f"at_action_{outcome.first_solve_at_action}"
        )
    else:
        verdict = (
            f"complete: ninth_game_no_solve_{outcome.target_game}_"
            f"{_reason_slug(outcome.failure_reason)}"
        )

    solve_trace = {
        "principle": "exp4072 reads this trace to measure cross-game concept-library transfer",
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
        "experiment": "experiment_4070_ninth_game_explore_first",
        "title": "arc3_ninth_game_explore_first_ft09",
        "honest_verdict": verdict,
        "game_solved": bool(solved),
        "target_game": outcome.target_game,
        "total_games_solved": int(total_games_solved),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "solve_trace": solve_trace,
        "inference_substrate": inference_substrate,
        "field_principles": _field_principles(),
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
    }
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
    """REQ-PHASE4-042: report the mandated blocked verdict on live ARC precondition failure."""

    artifact = {
        "experiment": "experiment_4070_ninth_game_explore_first",
        "title": "arc3_ninth_game_explore_first_ft09",
        "honest_verdict": "blocked_arc_env_unreachable",
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "solve_trace": {
            "principle": "exp4072 trace unavailable because live ARC precondition failed",
            "actions": [],
            "induction_calls": [],
            "verification_decisions": [],
            "phase_trace": [],
        },
        "inference_substrate": inference_substrate,
        "field_principles": _field_principles(),
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
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
