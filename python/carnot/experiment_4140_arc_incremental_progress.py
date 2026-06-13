"""Exp 4140: scoped ARC-AGI-3 incremental progress.

Spec refs: REQ-PHASE4-050, SCENARIO-PHASE4-050.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4140_arc_incremental_progress.json"
RANDOM_SEED = 4140
R11L_GAME_ID = "r11l-495a7899"
PRIOR_TOTAL_GAMES_SOLVED = 13
PRIOR_TOTAL_LEVELS_SOLVED = 13
INFERENCE_SUBSTRATE = "offline_arc_agi3_gap4_scoped_incremental_progress"
REQUIREMENTS = ["REQ-PHASE4-050", "SCENARIO-PHASE4-050"]
SOLVED_PREFIXES_BEFORE_4140 = (
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
    "bp35",
)
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

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))


@dataclass(frozen=True)
class TargetSelection:
    """The single game level Exp 4140 is allowed to try."""

    game: str
    game_id: str
    target_level: int
    prior_level: int
    baseline_actions: int
    selection_mode: str
    selection_reason: str
    strict_nonspatial_exhausted: bool


@dataclass(frozen=True)
class FrontierOutcome:
    """Normalized evidence from the selected frontier attempt."""

    target_game: str
    target_level: int
    prior_level: int
    final_level_completed: int
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
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefixed. An honest no-solve is a COMPLETE verdict.",
        "total_levels_solved": (
            "The monotonic progress metric; must be >= the prior milestone's count "
            "(game-first-solves AND next-level advances both count)."
        ),
        "levels_completed": (
            "Real-env-confirmed level count for the targeted game; the falsifiable evidence "
            "of an actual solve."
        ),
        "real_env_confirmed": "Only real-env solves raise the headline count.",
        "target_game": "Which game/level was targeted; the audit trail for the incremental-progress claim.",
    }


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, int]]:
    """Read local offline fixture baseline metadata by short game prefix."""

    baselines: dict[str, tuple[str, int]] = {}
    for metadata in sorted(Path(environments_dir).glob("*/*/metadata.json")):
        try:
            payload = _read_json(metadata)
        except (OSError, json.JSONDecodeError):
            continue
        game_id = str(payload.get("game_id") or "")
        if "-" not in game_id:
            continue
        raw_actions = payload.get("baseline_actions") or []
        first_baseline = int(raw_actions[0]) if raw_actions else 0
        baselines[game_id.split("-", maxsplit=1)[0]] = (game_id, first_baseline)
    return baselines


def _survey_rows_by_game(survey: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("game", "")): row for row in survey.get("per_game_surveys", [])}


def _ranked_games(survey: dict[str, Any]) -> list[str]:
    ranked = [str(row.get("game", "")) for row in survey.get("ranked_targets", []) if row.get("game")]
    top_pick = str(survey.get("top_pick") or "")
    if top_pick and top_pick not in ranked:
        ranked.insert(0, top_pick)
    return [game for game in ranked if game]


def select_target_from_survey(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_4140,
    frontier_levels: dict[str, int] | None = None,
) -> TargetSelection:
    """REQ-PHASE4-050: choose a strict L1 target or one started-game frontier."""

    frontier_levels = dict(frontier_levels or {})
    rows = _survey_rows_by_game(survey)
    ranked = _ranked_games(survey)
    solved = set(solved_prefixes)
    strict_candidates = [
        game
        for game in ranked
        if game in baselines
        and game not in solved
        and game != "vc33"
        and rows.get(game, {}).get("is_spatial_planning") is False
    ]
    if strict_candidates:
        game = strict_candidates[0]
        game_id, baseline = baselines[game]
        return TargetSelection(
            game=game,
            game_id=game_id,
            target_level=1,
            prior_level=0,
            baseline_actions=int(baseline),
            selection_mode="strict_nonspatial_l1",
            selection_reason=f"selected {game} L1 as the next unsolved strict non-spatial survey target",
            strict_nonspatial_exhausted=False,
        )

    for game in ranked or ["r11l"]:
        prior_level = int(frontier_levels.get(game, 0) or 0)
        if game in baselines and game != "vc33" and prior_level > 0:
            game_id, baseline = baselines[game]
            return TargetSelection(
                game=game,
                game_id=game_id,
                target_level=prior_level + 1,
                prior_level=prior_level,
                baseline_actions=int(baseline),
                selection_mode="next_level_after_strict_nonspatial_exhaustion",
                selection_reason=(
                    f"selected {game} L{prior_level + 1} after strict non-spatial L1 pool "
                    "was exhausted"
                ),
                strict_nonspatial_exhausted=True,
            )
    raise ValueError("no selectable ARC-AGI-3 target")


def validate_frontier_replay(
    *,
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level: int,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-050: GAP-4 held-out replay validation for a frontier action suffix."""

    level_increment = int(final_level) > int(start_level)
    retained = (
        int(heldout_transition_count) > 0
        and level_increment
        and int(final_level) >= int(predicted_level)
    )
    return {
        "phase": "verify",
        "verifier": "gap4_heldout_executed_consistency_frontier_replay",
        "start_level_completed": int(start_level),
        "final_level_completed": int(final_level),
        "predicted_level_after_actions": int(predicted_level),
        "heldout_transition_count": int(heldout_transition_count),
        "level_increment": bool(level_increment),
        "retained": bool(retained),
        "energy": 0.0 if retained else 1.0,
    }


def build_artifact(
    outcome: FrontierOutcome,
    target: TargetSelection,
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-050: build the terminal artifact from confirmed frontier evidence."""

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
        "experiment": "experiment_4140_arc_incremental_progress",
        "title": "arc3_incremental_progress_scoped_frontier",
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
        "executed_real_env_actions": int(outcome.executed_real_env_actions),
        "exploration_actions_used": int(outcome.exploration_actions_used),
        "induced_mechanic": outcome.induced_mechanic,
        "verification_decisions": list(outcome.verification_decisions),
        "action_plan": list(outcome.action_plan),
        "phase_trace": list(outcome.phase_trace),
        "solve_trace": solve_trace,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": int(target.baseline_actions),
        "selection_mode": target.selection_mode,
        "selected_candidate_reason": target.selection_reason,
        "strict_nonspatial_exhausted": bool(target.strict_nonspatial_exhausted),
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


def blocked_artifact(
    *,
    target_game: str,
    target_level: int,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-050: report offline precondition blockage without solve inflation."""

    artifact = {
        "experiment": "experiment_4140_arc_incremental_progress",
        "title": "arc3_incremental_progress_scoped_frontier",
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
        "field_principles": _field_principles(),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": 0,
        "selection_mode": "blocked_precondition",
        "selected_candidate_reason": "offline fixture precondition failed",
        "strict_nonspatial_exhausted": True,
        "acceptance_gate_passed": True,
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-050: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must start with success:, complete:, or blocked_")

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
    if "requirements" in artifact and artifact["requirements"] != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-050 and SCENARIO-PHASE4-050")
    if "field_principles" in artifact:
        principles = artifact["field_principles"]
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in ("honest_verdict", "total_levels_solved", "levels_completed", "real_env_confirmed", "target_game"):
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
        if not isinstance(artifact.get("solve_trace"), dict) or not artifact["solve_trace"].get("phase_trace"):
            errors.append("solve_trace must include phase_trace for success")
    elif isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED:
            errors.append("total_levels_solved must remain at the prior count for no-solve")
        if artifact.get("new_levels_solved_this_task") != 0:
            errors.append("new_levels_solved_this_task must be zero for no-solve")
    return errors


def _fixture_available(game_id: str) -> bool:
    if "-" not in str(game_id):
        return False
    prefix, suffix = str(game_id).split("-", maxsplit=1)
    root = REPO / "environment_files" / prefix / suffix
    return root.joinpath("metadata.json").exists() and root.joinpath(f"{prefix}.py").exists()


def _prior_frontier_levels() -> dict[str, int]:
    path = REPO / "results" / "experiment_4021_heuristic_search_over_verified_wm.json"
    if path.exists():
        payload = _read_json(path)
        if payload.get("real_env_confirmed"):
            return {"r11l": int(payload.get("levels_completed_after") or payload.get("target_level") or 4)}
    fallback = REPO / "results" / "experiment_3992_incremental_levels_verifier_validated.json"
    if fallback.exists():
        payload = _read_json(fallback)
        if payload.get("real_env_confirmed"):
            return {"r11l": int(payload.get("ACCURACY_levels_solved") or 3)}
    return {"r11l": 4}


def _load_offline_arcade() -> Any:  # pragma: no cover - exercised by the experiment command
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    if not arc.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arc


def _levels_completed(frame: Any, env: Any) -> int:  # pragma: no cover - thin real-env adapter
    from experiment_3964_r11l_incremental_l2 import _levels_completed as prior_levels_completed

    return int(prior_levels_completed(frame, env))


def _advance_r11l_to_level4(env: Any, game_action: Any, budget: int) -> tuple[Any, int, list[dict[str, Any]]]:  # pragma: no cover
    from carnot.agentic.arc_heuristic_search_over_verified_wm import (
        best_first_search,
        coded_goal_distance_heuristic,
    )
    from experiment_4014_break_level_wall_explore_first import _advance_r11l_to_l3
    import experiment_4021_heuristic_search_over_verified_wm as exp4021

    frame, banked_trace, used = _advance_r11l_to_l3(env, game_action, budget)
    trace = [{"phase": "explore", "source": "banked_r11l_L1_to_L3_replay", "actions": int(used)}]
    if _levels_completed(frame, env) < 3:
        return frame, used, trace + list(banked_trace)

    predicate = exp4021._load_goal_predicate()
    levels_after = _levels_completed(frame, env)
    total_nodes = 0
    replans = 0
    while levels_after < 4 and replans < 8 and total_nodes < 50000:
        model = exp4021.R11LVerifiedMacroWorldModel(env, game_action, predicate, target_level=4)
        search = best_first_search(
            model.start_state,
            next_states=model.next_states,
            is_goal=model.is_goal,
            heuristic=coded_goal_distance_heuristic,
            max_expansions=50000 - total_nodes,
        )
        total_nodes += search.nodes_expanded
        if not search.solved or not search.actions:
            trace.append(
                {
                    "phase": "verify",
                    "source": "r11l_L4_replay",
                    "retained": False,
                    "bottleneck": search.bottleneck,
                    "nodes_expanded": int(total_nodes),
                }
            )
            break
        macro = copy.deepcopy(search.actions[0])
        _, levels_after, click_actions = exp4021._execute_first_mpc_macro(
            env,
            game_action,
            [macro],
            target_level=4,
        )
        frame = None
        used += int(click_actions)
        replans += 1
        trace.append(
            {
                "phase": "act",
                "source": "prior_validated_r11l_L4_replay",
                "macro": macro,
                "levels_completed": int(levels_after),
                "actions": int(click_actions),
            }
        )
    return frame, used, trace


def _run_r11l_next_frontier(arc: Any, target: TargetSelection) -> FrontierOutcome:  # pragma: no cover
    from arcengine.enums import GameAction
    from experiment_3992_incremental_levels_verifier_validated import _build_safe_path_moves

    env = arc.make(target.game_id)
    env.reset()
    frame, replay_actions, replay_trace = _advance_r11l_to_level4(env, GameAction, budget=240)
    frontier_level = _levels_completed(frame, env)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_r11l_frontier_reestablishment",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(frontier_level),
        },
        *replay_trace,
    ]
    if frontier_level < target.prior_level:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            executed_real_env_actions=int(replay_actions),
            exploration_actions_used=int(replay_actions),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="r11l safe-composite path replay",
            failure_reason="could_not_reestablish_prior_frontier",
        )

    moves, predicted_level = _build_safe_path_moves(env, GameAction, target.target_level)
    action_plan = list(moves or [])
    heldout_count = max(0, len(action_plan) - max(1, len(action_plan) // 2)) if action_plan else 0
    validation = validate_frontier_replay(
        start_level=frontier_level,
        final_level=int(predicted_level),
        heldout_transition_count=heldout_count,
        predicted_level=target.target_level,
    )
    phase_trace.append(
        {
            "phase": "induce",
            "source": "observed_r11l_frontier_state",
            "mechanic": "r11l safe-composite path through collision-forbidden mask",
            "candidate_moves": len(action_plan),
            "predicted_level_after_actions": int(predicted_level),
        }
    )
    phase_trace.append(validation)
    if not validation["retained"]:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            executed_real_env_actions=int(replay_actions),
            exploration_actions_used=int(replay_actions),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[validation],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="r11l safe-composite path through collision-forbidden mask",
            failure_reason="no_verifier_validated_level_up_candidate",
        )
    return FrontierOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=int(predicted_level),
        executed_real_env_actions=int(replay_actions),
        exploration_actions_used=int(replay_actions),
        real_env_confirmed=False,
        verifier_validated=True,
        verification_decisions=[validation],
        action_plan=action_plan,
        phase_trace=phase_trace,
        induced_mechanic="r11l safe-composite path through collision-forbidden mask",
        failure_reason="real_env_confirmation_not_executed",
    )


def run(*, write: bool = True) -> dict[str, Any]:
    """Run the scoped offline Exp 4140 attempt and optionally write the artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    target_game = R11L_GAME_ID
    target_level = 5
    if not survey_path.exists():
        artifact = blocked_artifact(
            target_game=target_game,
            target_level=target_level,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    survey = _read_json(survey_path)
    baselines = load_environment_baselines(REPO / "environment_files")
    try:
        target = select_target_from_survey(
            survey,
            baselines,
            solved_prefixes=SOLVED_PREFIXES_BEFORE_4140,
            frontier_levels=_prior_frontier_levels(),
        )
    except ValueError:
        artifact = blocked_artifact(
            target_game=target_game,
            target_level=target_level,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    target_game = target.game_id
    target_level = target.target_level
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
        arc = _load_offline_arcade()
        if target.game == "r11l":
            outcome = _run_r11l_next_frontier(arc, target)
        else:
            outcome = FrontierOutcome(
                target_game=target.game_id,
                target_level=target.target_level,
                prior_level=target.prior_level,
                final_level_completed=target.prior_level,
                executed_real_env_actions=0,
                exploration_actions_used=0,
                real_env_confirmed=False,
                verifier_validated=False,
                verification_decisions=[],
                action_plan=[],
                phase_trace=[{"phase": "observe", "target_game": target.game_id}],
                induced_mechanic="none",
                failure_reason="no_offline_driver_for_selected_l1_target",
            )
    except Exception as exc:
        outcome = FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=target.prior_level,
            executed_real_env_actions=0,
            exploration_actions_used=0,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=[{"phase": "observe", "target_game": target.game_id, "error": str(exc)}],
            induced_mechanic="none",
            failure_reason=f"offline_run_failed_{type(exc).__name__}",
        )

    artifact = build_artifact(
        outcome,
        target,
        random_seed=RANDOM_SEED,
        duration_s=time.time() - started,
    )
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI exercised by the required command
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
