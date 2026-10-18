"""Exp 4049 helpers for the eighth ARC-AGI-3 explore-first solve.

Spec refs: REQ-PHASE4-041, SCENARIO-PHASE4-041.
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
SOLVED_GAME_PREFIXES = ("r11l", "lp85", "sc25", "su15", "tn36", "cd82", "dc22")
PRIOR_TOTAL_GAMES_SOLVED = 7


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
class Sb26Action:
    """Compact action record that can be serialized into the Exp 4049 trace."""

    action: int
    x: int | None = None
    y: int | None = None
    sprite: str = ""
    role: str = ""
    color: int | None = None

    @classmethod
    def click(
        cls,
        x: int,
        y: int,
        *,
        sprite: str = "",
        role: str = "",
        color: int | None = None,
    ) -> "Sb26Action":
        return cls(action=6, x=int(x), y=int(y), sprite=sprite, role=role, color=color)

    @classmethod
    def action5(cls) -> "Sb26Action":
        return cls(action=5, role="validate")

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"action": int(self.action)}
        if self.x is not None and self.y is not None:
            row["x"] = int(self.x)
            row["y"] = int(self.y)
        if self.sprite:
            row["sprite"] = self.sprite
        if self.role:
            row["role"] = self.role
        if self.color is not None:
            row["color"] = int(self.color)
        return row


@dataclass(frozen=True)
class Sb26ClickTarget:
    """Visible sb26 colored item that can be clicked and moved into a slot."""

    x: int
    y: int
    color: int
    name: str

    @property
    def center(self) -> tuple[int, int]:
        return int(self.x) + 3, int(self.y) + 3


@dataclass(frozen=True)
class Sb26Slot:
    """Visible sb26 frame slot; color is None when the slot is empty."""

    x: int
    y: int
    color: int | None

    @property
    def center(self) -> tuple[int, int]:
        return int(self.x) + 3, int(self.y) + 3


@dataclass(frozen=True)
class Sb26ObservedState:
    """Small verifier state: target colors, slot colors, loose items, and level counter."""

    target_colors: tuple[int, ...]
    slots: tuple[Sb26Slot, ...]
    items: tuple[Sb26ClickTarget, ...]
    level_completed: int

    @property
    def slot_colors(self) -> tuple[int | None, ...]:
        return tuple(slot.color for slot in self.slots)

    @property
    def loose_item_colors(self) -> tuple[int, ...]:
        return tuple(item.color for item in self.items)

    @property
    def remaining_mismatches(self) -> int:
        mismatches = 0
        for target, slot in zip(self.target_colors, self.slots, strict=True):
            if slot.color != target:
                mismatches += 1
        return mismatches

    def to_json(self) -> dict[str, Any]:
        return {
            "target_colors": [int(color) for color in self.target_colors],
            "slot_colors": [None if color is None else int(color) for color in self.slot_colors],
            "loose_item_colors": [int(color) for color in self.loose_item_colors],
            "level_completed": int(self.level_completed),
        }


@dataclass(frozen=True)
class Sb26Plan:
    """Induced sb26 L1 plan split into exploration and commit suffix."""

    actions: list[Sb26Action]
    exploration_actions: list[Sb26Action]
    commit_actions: list[Sb26Action]
    predicted_slot_colors: tuple[int, ...]
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
    action_plan: list[Sb26Action]
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
    """REQ-PHASE4-041: read local ARC metadata so selection uses measured L0 baselines."""

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


def _is_click_sequence_non_navigation(row: dict[str, Any]) -> bool:
    actions = str(row.get("available_actions", "")).lower()
    recipe = str(row.get("first_solve_recipe", "")).lower()
    summary = str(row.get("win_condition_summary", "")).lower()
    if "click" not in actions:
        return False
    if "[5" not in actions and "action5" not in recipe and "action 5" not in recipe:
        return False
    blocked_terms = (
        "drag",
        "resize",
        "gravity",
        "pathfinding",
        "pushing",
        "sokoban",
        "collision",
        "navigate a grid",
        "wall adjacent",
    )
    return not any(term in f"{recipe} {summary}" for term in blocked_terms)


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


def select_eighth_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_GAME_PREFIXES,
) -> SelectedCandidate:
    """REQ-PHASE4-041: choose strict non-spatial first, then the safest click-sequence fallback."""

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

    fallback = [row for row in rows if _is_click_sequence_non_navigation(row)]
    if fallback:
        row = min(fallback, key=lambda item: (baselines[str(item.get("game", ""))][1], str(item.get("game", ""))))
        return _candidate_from_row(
            row,
            baselines,
            solved_prefixes=solved_prefixes,
            selection_mode="fallback_click_sequence_non_navigation",
            reason_prefix="selected fallback",
        )
    raise ValueError("no unsolved non-spatial or click-sequence non-navigation survey candidates")


def build_sb26_l1_plan(state: Sb26ObservedState) -> Sb26Plan:
    """REQ-PHASE4-041: induce the sb26 color-sequence slot plan from observations."""

    if len(state.target_colors) != len(state.slots):
        raise ValueError("target color count and slot count must match")

    remaining_items = list(state.items)
    actions: list[Sb26Action] = []
    for target_color, slot in zip(state.target_colors, state.slots, strict=True):
        selected_index = next(
            (index for index, item in enumerate(remaining_items) if int(item.color) == int(target_color)),
            None,
        )
        if selected_index is None:
            raise ValueError(f"no available item for target color {target_color}")
        item = remaining_items.pop(selected_index)
        item_x, item_y = item.center
        slot_x, slot_y = slot.center
        actions.append(
            Sb26Action.click(
                item_x,
                item_y,
                sprite=item.name,
                role="select_item",
                color=int(target_color),
            )
        )
        actions.append(Sb26Action.click(slot_x, slot_y, role="place_slot", color=int(target_color)))

    actions.append(Sb26Action.action5())
    predicted = tuple(int(color) for color in state.target_colors)
    induction_call = {
        "call": "induce_sb26_color_sequence_slot_matching",
        "observed_target_colors": list(predicted),
        "observed_item_colors": [int(color) for color in state.loose_item_colors],
        "mechanic": "click an item to select it, click an empty slot to place it, ACTION5 validates left-to-right colors",
        "goal_predicate": "slot_colors == target_colors and ACTION5 increments level counter",
    }
    return Sb26Plan(
        actions=actions,
        exploration_actions=actions[:2],
        commit_actions=actions[2:],
        predicted_slot_colors=predicted,
        predicted_goal_after_commit=True,
        induction_call=induction_call,
    )


def validate_replayed_plan(
    start_state: Sb26ObservedState,
    final_state: Sb26ObservedState,
    plan: Sb26Plan,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-041: GAP-4-style replay validation gates live execution."""

    slot_sequence_matches = tuple(final_state.slot_colors) == tuple(start_state.target_colors)
    predicted_matches = tuple(plan.predicted_slot_colors) == tuple(start_state.target_colors)
    level_increment = int(final_state.level_completed) > int(start_state.level_completed)
    retained = bool(slot_sequence_matches and predicted_matches and level_increment)
    mismatches = sum(
        1
        for target, observed in zip(start_state.target_colors, final_state.slot_colors, strict=True)
        if observed != target
    )
    energy = 0.0 if retained else float(mismatches + (0 if level_increment else 1))
    return {
        "phase": "verify",
        "verifier": "gap4_replay_sb26_level_counter",
        "actions_checked": len(plan.commit_actions),
        "slot_sequence_matches": bool(slot_sequence_matches),
        "predicted_goal_after_actions": bool(predicted_matches and slot_sequence_matches),
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
        "solve_trace": "full observe/explore/induce/verify/act trace for Exp 4050 concept-transfer analysis",
        "inference_substrate": "declares the offline explore-first induction and verifier substrate",
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-041: validate required terminal artifact fields."""

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
    """REQ-PHASE4-041: construct the terminal artifact from confirmed outcome evidence."""

    solved = outcome.solved
    total_games_solved = int(outcome.prior_total_games_solved) + (1 if solved else 0)
    if solved:
        verdict = (
            f"success: eighth_game_solved_{outcome.target_game}_"
            f"at_action_{outcome.first_solve_at_action}"
        )
    else:
        verdict = (
            f"complete: eighth_game_no_solve_{outcome.target_game}_"
            f"{_reason_slug(outcome.failure_reason)}"
        )

    solve_trace = {
        "principle": "exp4050 reads this trace to measure cross-game concept-library transfer",
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
        "experiment": "experiment_4049_eighth_game_explore_first",
        "title": "arc3_eighth_game_explore_first_sb26",
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
    """REQ-PHASE4-041: report the mandated blocked verdict on live ARC precondition failure."""

    artifact = {
        "experiment": "experiment_4049_eighth_game_explore_first",
        "title": "arc3_eighth_game_explore_first_sb26",
        "honest_verdict": "blocked_arc_env_unreachable",
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "solve_trace": {
            "principle": "exp4050 trace unavailable because live ARC precondition failed",
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
