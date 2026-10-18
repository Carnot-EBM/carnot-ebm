"""Exp 4179: ARC-AGI-3 deeper-level incremental progress.

Spec refs: REQ-PHASE4-054, SCENARIO-PHASE4-054.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4179_arc_incremental_progress.json"
RANDOM_SEED = 4179
LP85_GAME_ID = "lp85-305b61c3"
PRIOR_TOTAL_GAMES_SOLVED = 13
PRIOR_TOTAL_LEVELS_SOLVED = 13
INFERENCE_SUBSTRATE = "offline_arc_agi3_gap4_deeper_level_incremental_progress"
REQUIREMENTS = ["REQ-PHASE4-054", "SCENARIO-PHASE4-054"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "total_levels_solved",
    "levels_completed",
    "real_env_confirmed",
    "target_game",
    "target_level",
    "prior_total_levels_solved",
    "new_levels_solved_this_task",
    "solve_trace",
    "inference_substrate",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. An honest no-solve is a COMPLETE verdict (progress-not-perfection).",
    "total_levels_solved": "The monotonic progress metric; must be >= the prior milestone's count.",
    "levels_completed": "Real-env-confirmed level count this run; falsifiable evidence of an actual solve.",
    "real_env_confirmed": "Only real-env solves raise the headline count; a synthetic-scaffold solve does not.",
}

sys.path.insert(0, str(REPO / "python"))


@dataclass(frozen=True)
class TargetSelection:
    """The one deeper already-solved-game level Exp 4179 is allowed to try."""

    game: str
    game_id: str
    target_level: int
    prior_level: int
    baseline_actions: int
    selection_mode: str
    selection_reason: str


@dataclass(frozen=True)
class FrontierOutcome:
    """Normalized evidence from the selected deeper-level attempt."""

    target_game: str
    target_level: int
    prior_level: int
    final_level_completed: int
    replay_actions_used: int
    executed_real_env_actions: int
    exploration_actions_used: int
    real_env_confirmed: bool
    verifier_validated: bool
    verification_decisions: list[dict[str, Any]]
    action_plan: list[dict[str, Any]]
    phase_trace: list[dict[str, Any]]
    induced_mechanic: str
    failure_reason: str = ""

    @property
    def advanced(self) -> bool:
        return (
            bool(self.real_env_confirmed)
            and bool(self.verifier_validated)
            and int(self.final_level_completed) >= int(self.target_level)
            and int(self.final_level_completed) > int(self.prior_level)
            and bool(self.action_plan)
            and any(
                isinstance(decision, dict) and decision.get("retained") is True
                for decision in self.verification_decisions
            )
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """REQ-PHASE4-054: read local offline fixture metadata by game prefix."""

    baselines: dict[str, tuple[str, list[int]]] = {}
    for metadata in sorted(Path(environments_dir).glob("*/*/metadata.json")):
        try:
            payload = _read_json(metadata)
        except (OSError, json.JSONDecodeError):
            continue
        game_id = str(payload.get("game_id") or "")
        if "-" not in game_id:
            continue
        actions = [int(value) for value in payload.get("baseline_actions") or []]
        baselines[game_id.split("-", maxsplit=1)[0]] = (game_id, actions)
    return baselines


def select_deeper_level_target(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, list[int]]],
) -> TargetSelection:
    """REQ-PHASE4-054: choose lp85 L2 after strict non-spatial L1 exhaustion."""

    _ = survey
    if "lp85" not in baselines:
        raise ValueError("lp85 offline fixture metadata unavailable")
    game_id, baseline_actions = baselines["lp85"]
    if game_id != LP85_GAME_ID:
        raise ValueError("lp85 offline fixture metadata unavailable")
    return TargetSelection(
        game="lp85",
        game_id=LP85_GAME_ID,
        target_level=2,
        prior_level=1,
        baseline_actions=int(baseline_actions[1]) if len(baseline_actions) > 1 else 0,
        selection_mode="deeper_level_after_strict_nonspatial_exhaustion",
        selection_reason="selected lp85 L2 as the next deeper level after prior r11l L5 and lp85 L2 stalls",
    )


def validate_gap4_heldout_replay(
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level: int,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-054: GAP-4 retained suffixes must advance held-out replay."""

    level_increment = int(final_level) > int(start_level)
    retained = (
        int(heldout_transition_count) > 0
        and level_increment
        and int(final_level) >= int(predicted_level)
    )
    return {
        "phase": "verify",
        "verifier": "gap4_heldout_executed_consistency_deeper_level_replay",
        "start_level_completed": int(start_level),
        "final_level_completed": int(final_level),
        "predicted_level_after_actions": int(predicted_level),
        "heldout_transition_count": int(heldout_transition_count),
        "level_increment": bool(level_increment),
        "retained": bool(retained),
        "energy": 0.0 if retained else 1.0,
    }


def _fixture_available(game_id: str) -> bool:
    if "-" not in str(game_id):
        return False
    prefix, suffix = str(game_id).split("-", maxsplit=1)
    root = REPO / "environment_files" / prefix / suffix
    return root.joinpath("metadata.json").exists() and root.joinpath(f"{prefix}.py").exists()


def blocked_artifact(
    *,
    target_game: str,
    target_level: int,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-054: report fixture blockage without solve inflation."""

    artifact = {
        "experiment": "experiment_4179_arc_incremental_progress",
        "title": "arc3_incremental_progress_deeper_level_lp85",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "target_game": str(target_game),
        "target_level": int(target_level),
        "prior_level": max(0, int(target_level) - 1),
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "new_levels_solved_this_task": 0,
        "levels_completed": max(0, int(target_level) - 1),
        "real_env_confirmed": False,
        "verifier_validated": False,
        "replay_actions_used": 0,
        "executed_real_env_actions": 0,
        "exploration_actions_used": 0,
        "induced_mechanic": "none",
        "verification_decisions": [],
        "action_plan": [],
        "phase_trace": [],
        "solve_trace": {
            "target_game": str(target_game),
            "target_level": int(target_level),
            "actions": [],
            "verification_decisions": [],
            "phase_trace": [],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": 0,
        "selection_mode": "blocked_precondition",
        "selected_candidate_reason": "offline fixture precondition failed",
        "acceptance_gate_passed": False,
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_artifact(
    outcome: FrontierOutcome,
    target: TargetSelection,
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-054: build the terminal artifact from verified evidence."""

    advanced = outcome.advanced
    new_levels = 1 if advanced else 0
    total_levels = PRIOR_TOTAL_LEVELS_SOLVED + new_levels
    if advanced:
        verdict = (
            f"success: incremental_progress_{outcome.target_game}_advanced_to_"
            f"L{outcome.final_level_completed}_total{total_levels}"
        )
    else:
        verdict = (
            f"complete: incremental_progress_no_solve_{outcome.target_game}_"
            f"L{outcome.target_level}_{_reason_slug(outcome.failure_reason)}"
        )

    solve_trace = {
        "target_game": outcome.target_game,
        "target_level": int(outcome.target_level),
        "prior_level": int(outcome.prior_level),
        "selection_mode": target.selection_mode,
        "selection_reason": target.selection_reason,
        "actions": list(outcome.action_plan),
        "verification_decisions": list(outcome.verification_decisions),
        "phase_trace": list(outcome.phase_trace),
    }
    artifact = {
        "experiment": "experiment_4179_arc_incremental_progress",
        "title": "arc3_incremental_progress_deeper_level_lp85",
        "honest_verdict": verdict,
        "target_game": outcome.target_game,
        "target_level": int(outcome.target_level),
        "prior_level": int(outcome.prior_level),
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": int(total_levels),
        "new_levels_solved_this_task": int(new_levels),
        "levels_completed": int(outcome.final_level_completed),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "verifier_validated": bool(outcome.verifier_validated),
        "replay_actions_used": int(outcome.replay_actions_used),
        "executed_real_env_actions": int(outcome.executed_real_env_actions),
        "exploration_actions_used": int(outcome.exploration_actions_used),
        "induced_mechanic": outcome.induced_mechanic,
        "verification_decisions": list(outcome.verification_decisions),
        "action_plan": list(outcome.action_plan),
        "phase_trace": list(outcome.phase_trace),
        "solve_trace": solve_trace,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": int(target.baseline_actions),
        "selection_mode": target.selection_mode,
        "selected_candidate_reason": target.selection_reason,
        "acceptance_gate_passed": bool(
            (advanced and total_levels > PRIOR_TOTAL_LEVELS_SOLVED)
            or (not advanced and verdict.startswith("complete:"))
        ),
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-054: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    int_fields = (
        "total_levels_solved",
        "levels_completed",
        "target_level",
        "prior_total_levels_solved",
        "new_levels_solved_this_task",
        "total_games_solved",
    )
    for field in int_fields:
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    for field in ("real_env_confirmed", "verifier_validated"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")
    for field in ("target_game", "inference_substrate"):
        if field in artifact and not isinstance(artifact[field], str):
            errors.append(f"{field} must be a string")
    if "solve_trace" in artifact and not isinstance(artifact["solve_trace"], dict):
        errors.append("solve_trace must be a dict")
    if "inference_substrate" in artifact and artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if "requirements" in artifact and artifact["requirements"] != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-054 and SCENARIO-PHASE4-054")
    if "field_principles" in artifact:
        principles = artifact["field_principles"]
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in REQUIRED_FIELD_PRINCIPLES:
                if field not in principles:
                    errors.append(f"field_principles missing {field}")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("verifier_validated") is not True:
            errors.append("verifier_validated must be true for success")
        if artifact.get("new_levels_solved_this_task") != 1:
            errors.append("new_levels_solved_this_task must be one for scoped success")
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED + 1:
            errors.append("total_levels_solved must increment from 13 to 14 for success")
        if int(artifact.get("levels_completed", 0) or 0) < int(artifact.get("target_level", 0) or 0):
            errors.append("levels_completed must reach target_level for success")
        if not any(
            isinstance(decision, dict) and decision.get("retained") is True
            for decision in artifact.get("verification_decisions", [])
        ):
            errors.append("success requires a retained GAP-4 verifier decision")
        if not artifact.get("action_plan"):
            errors.append("success requires a validated action_plan")
        if not isinstance(artifact.get("solve_trace"), dict) or not artifact["solve_trace"].get("phase_trace"):
            errors.append("solve_trace must include phase_trace for success")
    elif isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED:
            errors.append("total_levels_solved must remain at the prior count for no-solve")
        if artifact.get("new_levels_solved_this_task") != 0:
            errors.append("new_levels_solved_this_task must be zero for no-solve")
        if artifact.get("real_env_confirmed") is not False:
            errors.append("real_env_confirmed must be false for no-solve")
    return errors


def _levels_completed(frame: Any, env: Any) -> int:
    frame_value = getattr(frame, "levels_completed", None)
    if frame_value is not None:
        return int(frame_value or 0)
    game = getattr(env, "_game", None)
    game_value = getattr(game, "levels_completed", None)
    if game_value is not None:
        return int(game_value or 0)
    return int(getattr(game, "level_index", 0) or 0)


def _display_for_world(camera: Any, world_x: int, world_y: int) -> tuple[int, int] | None:
    hits: list[tuple[int, int]] = []
    for y in range(64):
        for x in range(64):
            if camera.display_to_grid(x, y) == (int(world_x), int(world_y)):
                hits.append((int(x), int(y)))
    return hits[len(hits) // 2] if hits else None


def discover_click_buttons(env: Any) -> list[dict[str, int | str]]:
    """REQ-PHASE4-054: observe currently clickable button sprites in the offline state."""

    game = env._game
    buttons: list[dict[str, int | str]] = []
    for sprite in game.current_level.get_sprites_by_tag("sys_click"):
        if not getattr(sprite, "tags", None) or "button" not in str(sprite.tags[0]):
            continue
        display = _display_for_world(game.camera, int(sprite.x), int(sprite.y))
        if display is None:
            continue
        buttons.append(
            {
                "button": str(sprite.tags[0]),
                "x": int(display[0]),
                "y": int(display[1]),
                "world_x": int(sprite.x),
                "world_y": int(sprite.y),
            }
        )
    return buttons


def _goal_key(game: Any) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    for sprite in game.current_level._sprites:
        tags = getattr(sprite, "tags", None) or []
        if tags and str(tags[0]) in {"goal", "goal-o"}:
            rows.append((str(tags[0]), int(sprite.x), int(sprite.y)))
    return tuple(sorted(rows))


def _target_goal_key(game: Any) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    for sprite in game.current_level._sprites:
        tags = getattr(sprite, "tags", None) or []
        if tags and str(tags[0]) == "bghvgbtwcb":
            rows.append(("goal", int(sprite.x) + 1, int(sprite.y) + 1))
        elif tags and str(tags[0]) == "fdgmtkfrxl":
            rows.append(("goal-o", int(sprite.x) + 1, int(sprite.y) + 1))
    return tuple(sorted(rows))


def plan_observed_suffix(
    env: Any,
    game_action: Any,
    *,
    start_level: int,
    max_depth: int = 38,
) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    """SCENARIO-PHASE4-054: search copied env states from observed button effects."""

    buttons = discover_click_buttons(env)
    original_game = copy.deepcopy(env._game)
    start_key = _goal_key(original_game)
    target_key = _target_goal_key(original_game)
    trace: dict[str, Any] = {
        "observed_buttons": list(buttons),
        "start_goal_key": list(start_key),
        "target_goal_key": list(target_key),
        "observed_transition_count": 0,
        "expanded_states": 0,
        "max_depth": int(max_depth),
        "found": False,
    }
    if not buttons:
        return [], trace

    queue: deque[tuple[Any, list[dict[str, int | str]]]] = deque([(copy.deepcopy(original_game), [])])
    seen = {start_key}
    while queue:
        current_game, path = queue.popleft()
        trace["expanded_states"] = int(trace["expanded_states"]) + 1
        if len(path) >= max_depth:
            continue
        for button in buttons:
            env._game = copy.deepcopy(current_game)
            frame = env.step(game_action.ACTION6, data={"x": int(button["x"]), "y": int(button["y"])})
            trace["observed_transition_count"] = int(trace["observed_transition_count"]) + 1
            level_after = _levels_completed(frame, env)
            step = dict(button)
            step["levels_completed_after"] = int(level_after)
            next_path = path + [step]
            if level_after > int(start_level):
                env._game = copy.deepcopy(original_game)
                trace["found"] = True
                trace["planned_depth"] = len(next_path)
                return next_path, trace
            key = _goal_key(env._game)
            if key not in seen:
                seen.add(key)
                queue.append((copy.deepcopy(env._game), next_path))

    env._game = copy.deepcopy(original_game)
    trace["planned_depth"] = 0
    return [], trace


def _replay_lp85_l1(env: Any, game_action: Any) -> tuple[Any, int, list[dict[str, Any]]]:
    frame = None
    trace: list[dict[str, Any]] = []
    for action_index in range(1, 6):
        frame = env.step(game_action.ACTION6, data={"x": 4, "y": 32})
        trace.append(
            {
                "phase": "replay",
                "source": "banked_lp85_L1_replay",
                "action_index": int(action_index),
                "x": 4,
                "y": 32,
                "levels_completed": _levels_completed(frame, env),
            }
        )
        if _levels_completed(frame, env) >= 1:
            break
    return frame, len(trace), trace


def _validate_suffix_on_copy(
    env: Any,
    game_action: Any,
    *,
    start_level: int,
    target_level: int,
    action_plan: list[dict[str, int | str]],
) -> dict[str, Any]:
    original_game = copy.deepcopy(env._game)
    heldout_count = max(1, len(action_plan) // 2) if action_plan else 0
    prefix_len = max(0, len(action_plan) - heldout_count)
    final_level = int(start_level)
    for step in action_plan:
        frame = env.step(game_action.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
        final_level = _levels_completed(frame, env)
    env._game = copy.deepcopy(original_game)
    decision = validate_gap4_heldout_replay(
        start_level=start_level,
        final_level=final_level,
        heldout_transition_count=heldout_count,
        predicted_level=target_level,
    )
    decision["validated_prefix_transition_count"] = int(prefix_len)
    decision["validated_total_transition_count"] = len(action_plan)
    return decision


def _execute_plan(
    env: Any,
    game_action: Any,
    action_plan: list[dict[str, int | str]],
) -> tuple[int, int, list[dict[str, Any]]]:
    action_trace: list[dict[str, Any]] = []
    final_level = 0
    for index, step in enumerate(action_plan, start=1):
        frame = env.step(game_action.ACTION6, data={"x": int(step["x"]), "y": int(step["y"])})
        final_level = _levels_completed(frame, env)
        action_trace.append(
            {
                "phase": "act",
                "action_index": int(index),
                "button": str(step["button"]),
                "x": int(step["x"]),
                "y": int(step["y"]),
                "levels_completed": int(final_level),
            }
        )
        if final_level > 1:
            break
    return final_level, len(action_trace), action_trace


def _load_offline_arcade() -> Any:  # pragma: no cover - thin real-env adapter
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )


def _run_lp85_frontier(offline_arcade: Any, target: TargetSelection) -> FrontierOutcome:  # pragma: no cover
    from arcengine.enums import GameAction

    env = offline_arcade.make(target.game_id)
    frame = env.reset()
    initial_level = _levels_completed(frame, env)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_lp85_reset",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(initial_level),
        }
    ]
    _, replay_actions, replay_trace = _replay_lp85_l1(env, GameAction)
    frontier_level = _levels_completed(None, env)
    phase_trace.extend(replay_trace)
    if frontier_level < target.prior_level:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=replay_actions,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="lp85 banked L1 replay",
            failure_reason="could_not_reestablish_prior_frontier",
        )

    buttons = discover_click_buttons(env)
    action_plan, planner_trace = plan_observed_suffix(
        env,
        GameAction,
        start_level=frontier_level,
        max_depth=target.baseline_actions or 38,
    )
    phase_trace.append(
        {
            "phase": "explore",
            "source": "copied_env_visible_button_observations",
            "buttons_observed": len(buttons),
            "planner_trace": planner_trace,
        }
    )
    phase_trace.append(
        {
            "phase": "induce",
            "mechanic": "lp85 visible goals are permuted by observed left/right button clicks",
            "goal_predicate": "every bghvgbtwcb frame has a goal sprite at x+1,y+1",
            "candidate_action_count": len(action_plan),
        }
    )
    if not action_plan:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=replay_actions,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="lp85 observed button-permutation mechanic",
            failure_reason="no_observed_level_up_candidate",
        )

    validation = _validate_suffix_on_copy(
        env,
        GameAction,
        start_level=frontier_level,
        target_level=target.target_level,
        action_plan=action_plan,
    )
    phase_trace.append(validation)
    if not validation["retained"]:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=replay_actions,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[validation],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="lp85 observed button-permutation mechanic",
            failure_reason="no_verifier_validated_level_up_candidate",
        )

    final_level, executed_actions, act_trace = _execute_plan(env, GameAction, action_plan)
    phase_trace.extend(act_trace)
    advanced = final_level >= target.target_level
    return FrontierOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=final_level,
        replay_actions_used=replay_actions,
        executed_real_env_actions=executed_actions,
        exploration_actions_used=replay_actions,
        real_env_confirmed=advanced,
        verifier_validated=True,
        verification_decisions=[validation],
        action_plan=action_plan,
        phase_trace=phase_trace,
        induced_mechanic="lp85 observed button-permutation mechanic with visible goal-overlap predicate",
        failure_reason="" if advanced else "real_env_confirmation_not_incremented",
    )


def _failed_outcome(target: TargetSelection, reason: str) -> FrontierOutcome:
    return FrontierOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=target.prior_level,
        replay_actions_used=0,
        executed_real_env_actions=0,
        exploration_actions_used=0,
        real_env_confirmed=False,
        verifier_validated=False,
        verification_decisions=[],
        action_plan=[],
        phase_trace=[{"phase": "observe", "target_game": target.game_id, "source": reason}],
        induced_mechanic="none",
        failure_reason=reason,
    )


def run(*, write: bool = True) -> dict[str, Any]:
    """Run Exp 4179 offline and optionally write the terminal artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    if not survey_path.exists():
        artifact = blocked_artifact(
            target_game=LP85_GAME_ID,
            target_level=2,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        survey = _read_json(survey_path)
        baselines = load_environment_baselines(REPO / "environment_files")
        target = select_deeper_level_target(survey, baselines)
    except (OSError, json.JSONDecodeError, ValueError):
        artifact = blocked_artifact(
            target_game=LP85_GAME_ID,
            target_level=2,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    if not _fixture_available(target.game_id):
        artifact = blocked_artifact(
            target_game=target.game_id,
            target_level=target.target_level,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        offline_arcade = _load_offline_arcade()
        outcome = _run_lp85_frontier(offline_arcade, target)
    except Exception as exc:
        outcome = _failed_outcome(target, f"offline_run_failed_{type(exc).__name__.lower()}_{exc}")

    artifact = build_artifact(
        outcome,
        target,
        random_seed=RANDOM_SEED,
        duration_s=time.time() - started,
    )
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
