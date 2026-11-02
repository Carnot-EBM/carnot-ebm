"""Exp 4038 helpers for the seventh ARC-AGI-3 explore-first solve.

Spec refs: REQ-PHASE4-038, SCENARIO-PHASE4-038.
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
    "inference_substrate",
)
SOLVED_GAME_PREFIXES = ("r11l", "lp85", "sc25", "su15", "tn36", "cd82")
PRIOR_TOTAL_GAMES_SOLVED = 6


@dataclass(frozen=True)
class SelectedCandidate:
    """Survey row selected for the next one-game ARC-AGI-3 increment."""

    game: str
    game_id: str
    baseline_actions: int
    non_spatial: bool
    win_difficulty: str
    selection_reason: str
    excluded_solved_games: tuple[str, ...]


@dataclass(frozen=True)
class Dc22Action:
    """A compact action record that can be serialized into the Exp 4038 trace."""

    action: int
    x: int | None = None
    y: int | None = None
    sprite: str = ""
    grid: tuple[int, int] | None = None

    @classmethod
    def key(cls, action: int) -> "Dc22Action":
        return cls(action=int(action))

    @classmethod
    def click(
        cls,
        x: int,
        y: int,
        *,
        sprite: str = "",
        grid: tuple[int, int] | None = None,
    ) -> "Dc22Action":
        return cls(action=6, x=int(x), y=int(y), sprite=str(sprite), grid=grid)

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"action": int(self.action)}
        if self.x is not None and self.y is not None:
            row["x"] = int(self.x)
            row["y"] = int(self.y)
        if self.sprite:
            row["sprite"] = self.sprite
        if self.grid is not None:
            row["grid"] = [int(self.grid[0]), int(self.grid[1])]
        return row


@dataclass(frozen=True)
class Dc22State:
    """Small verifier state: player, goal, level counter, and blocker signature."""

    player: tuple[int, int]
    goal: tuple[int, int]
    level_completed: int
    blocker_signature: tuple[Any, ...] = ()

    @property
    def distance_to_goal(self) -> int:
        return abs(int(self.player[0]) - int(self.goal[0])) + abs(
            int(self.player[1]) - int(self.goal[1])
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "player": [int(self.player[0]), int(self.player[1])],
            "goal": [int(self.goal[0]), int(self.goal[1])],
            "level_completed": int(self.level_completed),
            "blocker_signature": list(self.blocker_signature),
            "distance_to_goal": int(self.distance_to_goal),
        }


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
    action_plan: list[Dc22Action]
    arc_env_count: int
    failure_reason: str = ""

    @property
    def solved(self) -> bool:
        return (
            int(self.final_level_completed) > 0
            and int(self.first_solve_at_action) > 0
            and self.real_env_confirmed
        )


def load_environment_baselines(environment_dir: Path) -> dict[str, tuple[str, int]]:
    """REQ-PHASE4-038: read local ARC metadata so selection uses measured L0 baselines."""

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


def select_seventh_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_GAME_PREFIXES,
) -> SelectedCandidate:
    """REQ-PHASE4-038: choose the easiest unsolved non-spatial target and avoid vc33."""

    eligible: list[SelectedCandidate] = []
    for row in survey.get("per_game_surveys", []):
        game = str(row.get("game", ""))
        if game in solved_prefixes or game == "vc33":
            continue
        if row.get("is_spatial_planning") is not False:
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
                excluded_solved_games=tuple(solved_prefixes),
            )
        )
    if not eligible:
        raise ValueError("no unsolved non-spatial survey candidates with metadata baselines")
    return min(eligible, key=lambda item: (item.baseline_actions, item.game))


def dc22_default_exploration_actions() -> list[Dc22Action]:
    """REQ-PHASE4-038: spend a positive probe prefix before inducing the solve plan."""

    return [
        Dc22Action.key(1),
        Dc22Action.click(48, 36, sprite="buezna-blrmbx", grid=(48, 26)),
    ]


def validate_replayed_plan(
    start_state: Dc22State,
    replayed_states: list[Dc22State],
    actions: list[Dc22Action],
    *,
    start_level_completed: int,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-038: GAP-4-style replay validation gates real execution.

    The verifier is intentionally small: it does not accept a visual-looking
    goal by itself. It retains a plan only when replay reaches a state whose
    level counter is higher than the counter before the commit suffix.
    """

    if len(replayed_states) != len(actions) + 1:
        raise ValueError("replayed plan must include one more state than actions")
    if replayed_states[0] != start_state:
        raise ValueError("replayed plan must start from the supplied start state")

    final = replayed_states[-1]
    level_increment = int(final.level_completed) > int(start_level_completed)
    visual_goal = tuple(final.player) == tuple(final.goal)
    retained = bool(level_increment)
    final_distance = int(final.distance_to_goal)
    energy = 0.0 if retained else float(final_distance + 1)
    return {
        "phase": "verify",
        "verifier": "gap4_replay_level_counter",
        "actions_checked": len(actions),
        "start_distance": int(start_state.distance_to_goal),
        "final_distance": final_distance,
        "predicted_goal_after_actions": bool(level_increment or visual_goal),
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
        "real_env_confirmed": "a solve is real only when the environment level counter confirms it",
        "inference_substrate": "declares the offline explore-first induction and verifier substrate",
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-038: validate the required terminal artifact fields."""

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
    return errors


def build_artifact(
    outcome: ExperimentOutcome,
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-038: construct the terminal artifact from confirmed outcome evidence."""

    solved = outcome.solved
    total_games_solved = int(outcome.prior_total_games_solved) + (1 if solved else 0)
    if solved:
        verdict = (
            f"success: seventh_game_solved_{outcome.target_game}_"
            f"at_action_{outcome.first_solve_at_action}"
        )
    else:
        verdict = (
            f"complete: seventh_game_no_solve_{outcome.target_game}_"
            f"{_reason_slug(outcome.failure_reason)}"
        )
    artifact = {
        "experiment": "experiment_4038_seventh_game_explore_first",
        "title": "arc3_seventh_game_explore_first_dc22",
        "honest_verdict": verdict,
        "game_solved": bool(solved),
        "target_game": outcome.target_game,
        "total_games_solved": int(total_games_solved),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
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
    """REQ-PHASE4-038: report the mandated blocked verdict on live ARC precondition failure."""

    artifact = {
        "experiment": "experiment_4038_seventh_game_explore_first",
        "title": "arc3_seventh_game_explore_first_dc22",
        "honest_verdict": "blocked_arc_env_unreachable",
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
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
