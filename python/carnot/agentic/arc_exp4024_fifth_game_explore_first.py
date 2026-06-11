"""Exp 4024 helpers for the ARC-AGI-3 explore-first continuation.

Spec refs: REQ-PHASE4-032, SCENARIO-PHASE4-032.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "game_solved",
    "total_games_solved",
    "real_env_confirmed",
    "inference_substrate",
)
SOLVED_GAME_PREFIXES = ("r11l", "lp85", "sc25", "su15", "tn36")
PRIOR_TOTAL_GAMES_SOLVED = 5
CD82_RING_POSITIONS = {
    0: (0, 1),
    1: (0, 2),
    2: (1, 2),
    3: (2, 2),
    4: (2, 1),
    5: (2, 0),
    6: (1, 0),
    7: (0, 0),
}
CD82_POSITION_TO_INDEX = {value: key for key, value in CD82_RING_POSITIONS.items()}


@dataclass(frozen=True)
class SelectedCandidate:
    game: str
    game_id: str
    baseline_actions: int
    non_spatial: bool
    win_difficulty: str
    selection_reason: str
    excluded_solved_games: tuple[str, ...]


@dataclass(frozen=True)
class Cd82Plan:
    region_index: int
    fill_color: int
    actions: list[int]
    exploration_actions: list[int]
    commit_action: int
    predicted_canvas: np.ndarray
    predicted_goal_after_commit: bool


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
    failure_reason: str = ""

    @property
    def solved(self) -> bool:
        return (
            self.final_level_completed > 0
            and self.first_solve_at_action > 0
            and self.real_env_confirmed
        )


def load_environment_baselines(environment_dir: Path) -> dict[str, tuple[str, int]]:
    """REQ-PHASE4-032: read local ARC metadata so survey choices use real L0 baselines."""

    baselines: dict[str, tuple[str, int]] = {}
    for metadata_path in sorted(environment_dir.glob("*/*/metadata.json")):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline_values = data.get("baseline_actions") or []
        if not baseline_values:  # pragma: no cover - defensive for incomplete downloaded games
            continue
        game = str(data["game_id"]).split("-", 1)[0]
        baselines[game] = (str(data["game_id"]), int(baseline_values[0]))
    return baselines


def select_new_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_GAME_PREFIXES,
) -> SelectedCandidate:
    """REQ-PHASE4-032: choose the lowest-baseline unsolved non-spatial survey target."""

    eligible: list[SelectedCandidate] = []
    for row in survey.get("per_game_surveys", []):
        game = str(row.get("game", ""))
        if game in solved_prefixes or not bool(row.get("is_spatial_planning") is False):
            continue
        if game not in baselines:
            continue
        game_id, baseline_actions = baselines[game]
        eligible.append(
            SelectedCandidate(
                game=game,
                game_id=game_id,
                baseline_actions=baseline_actions,
                non_spatial=True,
                win_difficulty=str(row.get("win_difficulty", "unknown")),
                selection_reason=(
                    f"selected: {game} is unsolved, survey non-spatial, "
                    f"win_difficulty={row.get('win_difficulty', 'unknown')}, "
                    f"L0 baseline_actions={baseline_actions}"
                ),
                excluded_solved_games=solved_prefixes,
            )
        )
    if not eligible:
        raise ValueError("no unsolved non-spatial survey candidates with metadata baselines")
    return min(eligible, key=lambda item: (item.baseline_actions, item.game))


def cd82_goal_mask(size: int = 10) -> np.ndarray:
    """SCENARIO-PHASE4-032: cd82 validates all target pixels except both diagonals."""

    mask = np.ones((size, size), dtype=bool)
    for index in range(size):
        mask[index, index] = False
        mask[index, size - 1 - index] = False
    return mask


def cd82_region_mask(region_index: int) -> np.ndarray:
    """Return the pixel region painted by a cd82 basket index."""

    mask = np.zeros((10, 10), dtype=bool)
    if region_index == 0:
        mask[0:5, :] = True
    elif region_index == 2:
        mask[:, 5:10] = True
    elif region_index == 4:
        mask[5:10, :] = True
    elif region_index == 6:
        mask[:, 0:5] = True
    elif region_index == 1:
        for row in range(10):
            mask[row, row:10] = True
    elif region_index == 3:
        for row in range(10):
            mask[row, 9 - row:10] = True
    elif region_index == 5:
        for row in range(10):
            mask[row, 0 : row + 1] = True
    elif region_index == 7:
        for row in range(10):
            mask[row, 0 : 10 - row] = True
    else:
        raise ValueError(f"unknown cd82 region index {region_index}")
    return mask


def apply_cd82_region_fill(canvas: np.ndarray, region_index: int, fill_color: int) -> np.ndarray:
    """Apply the induced cd82 one-region fill model to a 10x10 canvas."""

    predicted = np.array(canvas, dtype=np.int16, copy=True)
    predicted[cd82_region_mask(region_index)] = int(fill_color)
    return predicted


def move_basket_index(active_index: int, action_id: int) -> int:
    """Move the cd82 active basket around the 3x3 ring for actions 1-4."""

    row, col = CD82_RING_POSITIONS[int(active_index)]
    if action_id == 1:
        next_pos = (max(0, row - 1), col)
    elif action_id == 2:
        next_pos = (min(2, row + 1), col)
    elif action_id == 3:
        next_pos = (row, max(0, col - 1))
    elif action_id == 4:
        next_pos = (row, min(2, col + 1))
    else:
        return int(active_index)
    if next_pos == (1, 1):
        return int(active_index)
    return int(CD82_POSITION_TO_INDEX.get(next_pos, active_index))


def basket_navigation_actions(start_index: int, target_index: int) -> list[int]:
    """Find the shortest keyboard path between cd82 basket indices."""

    queue: deque[tuple[int, list[int]]] = deque([(int(start_index), [])])
    seen = {int(start_index)}
    while queue:
        index, path = queue.popleft()
        if index == int(target_index):
            return path
        for action_id in (4, 2, 3, 1):
            next_index = move_basket_index(index, action_id)
            if next_index not in seen:
                seen.add(next_index)
                queue.append((next_index, [*path, action_id]))
    raise ValueError(f"no cd82 basket path from {start_index} to {target_index}")


def _single_region_fill(
    current_canvas: np.ndarray,
    target_canvas: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    current = np.asarray(current_canvas, dtype=np.int16)
    target = np.asarray(target_canvas, dtype=np.int16)
    if current.shape != (10, 10) or target.shape != (10, 10):
        raise ValueError("cd82 canvases must be 10x10")

    goal_mask = cd82_goal_mask()
    candidate_colors = sorted({int(value) for value in target[goal_mask].ravel()})
    for region_index in range(8):
        for fill_color in candidate_colors:
            predicted = apply_cd82_region_fill(current, region_index, fill_color)
            if np.array_equal(predicted[goal_mask], target[goal_mask]):
                return region_index, fill_color, predicted
    raise ValueError("target difference is not covered by a single cd82 region fill")


def build_cd82_l1_plan(
    *,
    active_index: int,
    selected_color: int,
    current_canvas: np.ndarray,
    target_canvas: np.ndarray,
) -> Cd82Plan:
    """REQ-PHASE4-032: induce the cd82 L1 region fill and verify it reaches the goal."""

    region_index, fill_color, predicted = _single_region_fill(current_canvas, target_canvas)
    if int(selected_color) != fill_color:
        raise ValueError("selected cd82 palette color does not match the induced fill color")
    exploration_actions = basket_navigation_actions(active_index, region_index)
    actions = [*exploration_actions, 5]
    goal_mask = cd82_goal_mask()
    return Cd82Plan(
        region_index=region_index,
        fill_color=fill_color,
        actions=actions,
        exploration_actions=exploration_actions,
        commit_action=5,
        predicted_canvas=predicted,
        predicted_goal_after_commit=bool(
            np.array_equal(predicted[goal_mask], np.asarray(target_canvas)[goal_mask])
        ),
    )


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-032: validate the mandatory terminal artifact schema."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    if "honest_verdict" in artifact:
        verdict = artifact["honest_verdict"]
        if not isinstance(verdict, str):
            errors.append("honest_verdict must be a string")
        elif not (
            verdict.startswith("success:")
            or verdict.startswith("complete:")
            or verdict.startswith("blocked_")
        ):
            errors.append("honest_verdict must start with success:, complete:, or blocked_")

    for field in ("game_solved", "real_env_confirmed"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")
    if "total_games_solved" in artifact and not isinstance(artifact["total_games_solved"], int):
        errors.append("total_games_solved must be a bare int")
    if "inference_substrate" in artifact and not isinstance(artifact["inference_substrate"], str):
        errors.append("inference_substrate must be a string")

    if artifact.get("honest_verdict", "").startswith("success:"):
        if artifact.get("game_solved") is not True:
            errors.append("game_solved must be true for success")
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if int(artifact.get("level_completed", 0) or 0) <= 0:
            errors.append("level_completed must increment for success")
    return errors


def build_artifact(
    outcome: ExperimentOutcome,
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """Build the terminal Exp 4024 artifact from an observed real-env outcome."""

    solved = outcome.solved
    total_games_solved = outcome.prior_total_games_solved + (1 if solved else 0)
    if solved:
        verdict = f"success: fifth_game_solved_{outcome.target_game}_at_action_{outcome.first_solve_at_action}"
    else:
        verdict = (
            f"complete: fifth_game_no_solve_{outcome.target_game}_"
            f"{_reason_slug(outcome.failure_reason)}"
        )
    artifact = {
        "experiment": "experiment_4024_fifth_game_explore_first",
        "title": "arc3_fifth_game_explore_first_cd82_continuation",
        "honest_verdict": verdict,
        "game_solved": bool(solved),
        "total_games_solved": int(total_games_solved),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "inference_substrate": inference_substrate,
        "target_game": outcome.target_game,
        "selected_candidate_reason": outcome.selected_candidate_reason,
        "prior_total_games_solved": int(outcome.prior_total_games_solved),
        "level_completed": int(outcome.final_level_completed),
        "levels_completed": int(outcome.final_level_completed),
        "first_solve_at_action": int(outcome.first_solve_at_action),
        "exploration_actions_used": int(outcome.exploration_actions_used),
        "induced_mechanic": outcome.induced_mechanic,
        "verification_decisions": list(outcome.verification_decisions),
        "phase_trace": list(outcome.phase_trace),
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
    """Return the honest blocked artifact when the offline ARC environment is unavailable."""

    artifact = {
        "experiment": "experiment_4024_fifth_game_explore_first",
        "title": "arc3_fifth_game_explore_first_cd82_continuation",
        "honest_verdict": "blocked_arc_offline_env_unavailable",
        "game_solved": False,
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "inference_substrate": inference_substrate,
        "target_game": "none",
        "selected_candidate_reason": "offline ARC environment unavailable before survey target execution",
        "prior_total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "level_completed": 0,
        "levels_completed": 0,
        "first_solve_at_action": -1,
        "exploration_actions_used": 0,
        "induced_mechanic": "none",
        "verification_decisions": [],
        "phase_trace": [],
        "random_seed": int(random_seed),
        "duration_s": float(duration_s),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
