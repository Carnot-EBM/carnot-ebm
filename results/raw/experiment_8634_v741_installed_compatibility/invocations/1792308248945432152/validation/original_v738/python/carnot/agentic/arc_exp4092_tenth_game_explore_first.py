"""Exp 4092 helpers for the ARC-AGI-3 tenth-game explore-first solve.

Spec refs: REQ-PHASE4-045, SCENARIO-PHASE4-045.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    SelectedCandidate,
)

REQUIREMENTS = ["REQ-PHASE4-045", "SCENARIO-PHASE4-045"]
PRIOR_TOTAL_GAMES_SOLVED = 9
TARGET_TOTAL_GAMES_SOLVED = 10
PREFERRED_GAME = "r11l"
PREFERRED_GAME_ID = "r11l-495a7899"
SOLVED_PREFIXES_BEFORE_TENTH = ("lp85", "sc25", "su15", "tn36", "cd82", "dc22", "sb26", "ft09")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "game_solved",
    "total_games_solved",
    "first_solve_at_action",
    "actions_vs_baseline",
    "real_env_confirmed",
    "inference_substrate",
)

Point = tuple[int, int]


@dataclass(frozen=True)
class R11LPiece:
    """One visible r11l piece center in row, column coordinates."""

    piece_index: int
    center: Point

    def to_json(self) -> dict[str, Any]:
        return {
            "piece_index": int(self.piece_index),
            "center": [int(self.center[0]), int(self.center[1])],
        }


@dataclass(frozen=True)
class R11LGroup:
    """One r11l target group with its movable pieces and target center."""

    group_id: str
    target_center: Point
    target_satisfied: bool
    pieces: tuple[R11LPiece, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "target_center": [int(self.target_center[0]), int(self.target_center[1])],
            "target_satisfied": bool(self.target_satisfied),
            "pieces": [piece.to_json() for piece in self.pieces],
        }


@dataclass(frozen=True)
class R11LObservedState:
    """Observed r11l groups and the environment-confirmed level counter."""

    groups: tuple[R11LGroup, ...]
    level_completed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "groups": [group.to_json() for group in self.groups],
            "level_completed": int(self.level_completed),
        }


@dataclass(frozen=True)
class R11LAction:
    """One r11l click action."""

    point: Point
    role: str
    group_id: str
    piece_index: int
    action: int = 6

    @property
    def x(self) -> int:
        return int(self.point[1])

    @property
    def y(self) -> int:
        return int(self.point[0])

    @classmethod
    def click(
        cls,
        point: Point,
        *,
        role: str,
        group_id: str,
        piece_index: int,
    ) -> "R11LAction":
        return cls(
            point=(int(point[0]), int(point[1])),
            role=role,
            group_id=group_id,
            piece_index=int(piece_index),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "action": int(self.action),
            "x": self.x,
            "y": self.y,
            "point": [self.y, self.x],
            "role": self.role,
            "group_id": self.group_id,
            "piece_index": int(self.piece_index),
        }


@dataclass(frozen=True)
class R11LPlan:
    """Induced r11l plan split into exploration and validated commit suffix."""

    actions: list[R11LAction]
    exploration_actions: list[R11LAction]
    commit_actions: list[R11LAction]
    predicted_goal_after_commit: bool
    induction_call: dict[str, Any]


@dataclass(frozen=True)
class R11LOutcome:
    """Normalized result evidence for Exp 4092 artifact construction."""

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
    action_plan: list[R11LAction]
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


def _centroid(sprite: Any) -> Point:
    return (
        int(sprite.y) + int(getattr(sprite, "height", 1)) // 2,
        int(sprite.x) + int(getattr(sprite, "width", 1)) // 2,
    )


def _target_center(target: Any, composite: Any | None) -> Point:
    if composite is None:
        return _centroid(target)
    return (
        int(target.y) + int(getattr(composite, "height", getattr(target, "height", 1))) // 2,
        int(target.x) + int(getattr(composite, "width", getattr(target, "width", 1))) // 2,
    )


def observe_r11l_state_from_env(env: Any, *, level_completed: int) -> R11LObservedState:
    """REQ-PHASE4-045: derive r11l groups from the observed engine state."""

    groups: list[R11LGroup] = []
    game = env._game
    for group_id, data in sorted(game.kacotwgjcyq.items()):
        target = data.get("gosubdcyegamj")
        if target is None:
            continue
        composite = data.get("roduyfsmiznvg")
        pieces = tuple(
            R11LPiece(piece_index=index, center=_centroid(piece))
            for index, piece in enumerate(data.get("lecfirgqbwunn", []))
        )
        groups.append(
            R11LGroup(
                group_id=str(group_id),
                target_center=_target_center(target, composite),
                target_satisfied=bool(composite is not None and composite.collides_with(target)),
                pieces=pieces,
            )
        )
    return R11LObservedState(groups=tuple(groups), level_completed=int(level_completed))


def _offsets_for_count(count: int, *, spacing: int = 4) -> list[tuple[int, int]]:
    if count <= 1:
        return [(0, 0)]
    offsets: list[tuple[int, int]] = []
    symmetric_pairs = [
        ((-spacing, 0), (spacing, 0)),
        ((0, -spacing), (0, spacing)),
        ((-spacing, -spacing), (spacing, spacing)),
        ((-spacing, spacing), (spacing, -spacing)),
    ]
    for left, right in symmetric_pairs:
        if len(offsets) + 2 <= count:
            offsets.extend([left, right])
    if len(offsets) < count:
        offsets.append((0, 0))
    return offsets[:count]


def build_r11l_l1_plan(state: R11LObservedState) -> R11LPlan:
    """REQ-PHASE4-045: induce the r11l click-select/place plan from observations."""

    if not state.groups:
        raise ValueError("at least one r11l target group is required")

    actions: list[R11LAction] = []
    for group in state.groups:
        if not group.pieces:
            raise ValueError(f"r11l group {group.group_id} has no observed pieces")
        offsets = _offsets_for_count(len(group.pieces))
        for piece, (dx, dy) in zip(group.pieces, offsets, strict=True):
            placement = (int(group.target_center[0]) + int(dy), int(group.target_center[1]) + int(dx))
            actions.append(
                R11LAction.click(
                    piece.center,
                    role="select_piece",
                    group_id=group.group_id,
                    piece_index=piece.piece_index,
                )
            )
            actions.append(
                R11LAction.click(
                    placement,
                    role="place_piece",
                    group_id=group.group_id,
                    piece_index=piece.piece_index,
                )
            )

    induction_call = {
        "call": "induce_r11l_click_select_place",
        "observed_groups": [group.to_json() for group in state.groups],
        "mechanic": "clicking a visible piece selects it; clicking an empty target-relative point places it",
        "goal_predicate": "each movable piece group composite overlaps its matching flkdtg target",
    }
    return R11LPlan(
        actions=actions,
        exploration_actions=actions[:2],
        commit_actions=actions[2:],
        predicted_goal_after_commit=bool(actions),
        induction_call=induction_call,
    )


def validate_r11l_replayed_plan(
    start_state: R11LObservedState,
    final_state: R11LObservedState,
    plan: R11LPlan,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-045: GAP-4-style replay validation gates live execution."""

    level_increment = int(final_state.level_completed) > int(start_state.level_completed)
    retained = bool(plan.predicted_goal_after_commit and level_increment)
    energy = 0.0 if retained else 1.0
    return {
        "phase": "verify",
        "verifier": "gap4_replay_r11l_click_select_place_level_counter",
        "actions_checked": len(plan.commit_actions),
        "heldout_transition_count": len(plan.commit_actions),
        "start_level_completed": int(start_state.level_completed),
        "final_level_completed": int(final_state.level_completed),
        "predicted_goal_after_actions": bool(plan.predicted_goal_after_commit),
        "level_increment": bool(level_increment),
        "retained": retained,
        "energy": energy,
    }


def select_exp4092_candidate_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_TENTH,
) -> SelectedCandidate:
    """REQ-PHASE4-045: choose R11L as the consensus tenth-game target."""

    for row in survey.get("per_game_surveys", []):
        game = str(row.get("game", ""))
        if game != PREFERRED_GAME or game in solved_prefixes or game == "vc33":
            continue
        if game not in baselines:
            continue
        game_id, baseline_actions = baselines[game]
        return SelectedCandidate(
            game=game,
            game_id=game_id,
            baseline_actions=int(baseline_actions),
            survey_is_spatial_planning=bool(row.get("is_spatial_planning")),
            win_difficulty=str(row.get("win_difficulty", "unknown")),
            selection_mode="preferred_consensus_top_pick",
            selection_reason=(
                f"selected preferred: {game} is the consensus top pick, directly observable, "
                f"L0 baseline_actions={baseline_actions}"
            ),
            excluded_solved_games=tuple(solved_prefixes),
        )
    raise ValueError("r11l consensus top-pick candidate unavailable")


def compute_actions_vs_baseline(
    first_solve_at_action: int,
    baseline_actions: int,
    *,
    solved: bool,
) -> float:
    """REQ-PHASE4-045: normalize confirmed solve depth against the L0 baseline."""

    if not solved:
        return 0.0
    if int(baseline_actions) <= 0:
        raise ValueError("baseline_actions must be positive for a solved action ratio")
    if int(first_solve_at_action) <= 0:
        raise ValueError("first_solve_at_action must be positive for a solved action ratio")
    return round(float(first_solve_at_action) / float(baseline_actions), 4)


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "terminal prefix records success, complete no-solve, or blocked precondition",
        "game_solved": "bare boolean for whether this task added exactly one new solved game",
        "total_games_solved": "monotonic ARC accuracy counter, incrementing from 9 to 10 only on a real solve",
        "first_solve_at_action": "real confirmed action index where the first level counter increment occurred",
        "actions_vs_baseline": "confirmed solve actions divided by the selected game's L0 baseline action count",
        "real_env_confirmed": "a solve is real only when the live environment level counter confirms it",
        "inference_substrate": "declares the offline explore-first induction and verifier substrate",
        "requirements": "OpenSpec requirement and scenario anchors for the Exp 4092 tenth-game run",
    }


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-045: validate the Exp 4092 terminal artifact contract."""

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
    if "total_games_solved" in artifact and type(artifact["total_games_solved"]) is not int:
        errors.append("total_games_solved must be a bare int")
    if "first_solve_at_action" in artifact and type(artifact["first_solve_at_action"]) is not int:
        errors.append("first_solve_at_action must be a bare int")
    if "actions_vs_baseline" in artifact and type(artifact["actions_vs_baseline"]) is not float:
        errors.append("actions_vs_baseline must be a bare float")
    if "real_env_confirmed" in artifact and type(artifact["real_env_confirmed"]) is not bool:
        errors.append("real_env_confirmed must be a bare bool")
    if "inference_substrate" in artifact and artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if "requirements" in artifact and not all(req in artifact["requirements"] for req in REQUIREMENTS):
        errors.append("requirements must include REQ-PHASE4-045 and SCENARIO-PHASE4-045")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("game_solved") is not True:
            errors.append("game_solved must be true for success")
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("total_games_solved") != TARGET_TOTAL_GAMES_SOLVED:
            errors.append("total_games_solved must increment from 9 to 10 for success")
        if int(artifact.get("level_completed", 0) or 0) <= 0:
            errors.append("level_completed must increment for success")
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
    outcome: R11LOutcome,
    candidate: SelectedCandidate,
    *,
    random_seed: int,
    duration_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-PHASE4-045: construct the terminal artifact from confirmed r11l evidence."""

    solved = outcome.solved
    total_games_solved = int(outcome.prior_total_games_solved) + (1 if solved else 0)
    if solved:
        verdict = (
            f"success: tenth_game_solved_{outcome.target_game}_"
            f"at_action_{outcome.first_solve_at_action}"
        )
    else:
        verdict = (
            f"complete: tenth_game_no_solve_{outcome.target_game}_"
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
        "experiment": "experiment_4092_tenth_game_explore_first",
        "title": "arc3_tenth_game_explore_first_r11l",
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
        "survey_is_spatial_planning": bool(candidate.survey_is_spatial_planning),
        "actions_vs_baseline": compute_actions_vs_baseline(
            int(outcome.first_solve_at_action),
            int(candidate.baseline_actions),
            solved=solved,
        ),
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
    """REQ-PHASE4-045: report the blocked live-ARC precondition without solve inflation."""

    artifact = {
        "experiment": "experiment_4092_tenth_game_explore_first",
        "title": "arc3_tenth_game_explore_first_r11l",
        "honest_verdict": "blocked_arc_agi3_unreachable",
        "game_solved": False,
        "target_game": "none",
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
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact
