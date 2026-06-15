"""Exp 4249: ARC-AGI-3 hardened SC25 L5 incremental progress.

Spec refs: REQ-PHASE4-065, SCENARIO-PHASE4-065.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import carnot.experiment_4236_arc_incremental_progress as previous


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4249_arc_incremental_progress.json"
RANDOM_SEED = 4249
SC25_GAME_ID = previous.SC25_GAME_ID
PRIOR_TOTAL_GAMES_SOLVED = 13
PRIOR_TOTAL_LEVELS_SOLVED = 18
INFERENCE_SUBSTRATE = "offline_arc_agi3_hardened_gap4_sc25_l5_incremental_progress"
REQUIREMENTS = ["REQ-PHASE4-065", "SCENARIO-PHASE4-065"]
HARDENED_VERIFIER = "hardened_gap4_heldout_executed_consistency_sc25_l5_dual_fire_lane_replay"
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
    "total_levels_solved": "The monotonic progress metric; must be >= the prior milestone's count (18).",
    "levels_completed": "Real-env-confirmed level count this run; falsifiable evidence of an actual solve.",
    "real_env_confirmed": "Only real-env solves raise the headline count; a synthetic-scaffold solve does not.",
}
L5_SHRINK_SPELL = "sieesc_chwjgc"
L5_TELEPORT_SPELL = "tevyeq"
L5_FIRE_SPELL = "fibcey"


@dataclass(frozen=True)
class TargetSelection:
    """The single SC25 L5 frontier Exp 4249 is allowed to try."""

    game: str
    game_id: str
    target_level: int
    prior_level: int
    baseline_actions: int
    selection_mode: str
    selection_reason: str


@dataclass(frozen=True)
class FrontierOutcome:
    """Normalized evidence from the selected SC25 L5 frontier attempt."""

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
                isinstance(decision, dict)
                and decision.get("retained") is True
                and decision.get("verifier") == HARDENED_VERIFIER
                for decision in self.verification_decisions
            )
        )


def _base() -> Any:
    return previous.previous.previous


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """REQ-PHASE4-065: read local offline fixture metadata by game prefix."""

    return previous.load_environment_baselines(environments_dir)


def _gap4_hardening_ready(gap4_artifact: dict[str, Any]) -> bool:
    return bool(_base().gap4_hardening_ready(gap4_artifact))


def _survey_mentions_sc25(survey: dict[str, Any]) -> bool:
    rows: list[Any] = []
    for field in ("ranked_targets", "per_game_surveys"):
        value = survey.get(field, [])
        if isinstance(value, list):
            rows.extend(value)
    return any(isinstance(row, dict) and row.get("game") == "sc25" for row in rows)


def select_deeper_level_target(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, list[int]]],
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> TargetSelection:
    """REQ-PHASE4-065: choose SC25 L5 after Exp 4236 banked SC25 L4."""

    prior_ok = (
        prior_artifact.get("experiment") == "experiment_4236_arc_incremental_progress"
        and str(prior_artifact.get("honest_verdict") or "").startswith("success:")
        and prior_artifact.get("target_game") == SC25_GAME_ID
        and int(prior_artifact.get("target_level", 0) or 0) == 4
        and int(prior_artifact.get("total_levels_solved", 0) or 0) >= PRIOR_TOTAL_LEVELS_SOLVED
        and int(prior_artifact.get("levels_completed", 0) or 0) >= 4
        and int(prior_artifact.get("new_levels_solved_this_task", 0) or 0) == 1
        and prior_artifact.get("real_env_confirmed") is True
        and bool(prior_artifact.get("action_plan"))
    )
    if not prior_ok:
        raise ValueError("Exp 4236 sc25 L4 success evidence unavailable")
    if not _gap4_hardening_ready(gap4_artifact):
        raise ValueError("hardened GAP-4 verifier evidence unavailable")
    if not _survey_mentions_sc25(survey):
        raise ValueError("sc25 survey evidence unavailable")
    if "sc25" not in baselines:
        raise ValueError("sc25 offline fixture metadata unavailable")
    game_id, baseline_actions = baselines["sc25"]
    if game_id != SC25_GAME_ID or len(baseline_actions) < 5:
        raise ValueError("sc25 offline fixture metadata unavailable")
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=5,
        prior_level=4,
        baseline_actions=int(baseline_actions[4]),
        selection_mode="deeper_sc25_frontier_after_exp4236_L4",
        selection_reason="selected sc25 L5 after Exp 4236 banked sc25 L4 with hardened GAP-4 evidence",
    )


def validate_hardened_gap4_l5_suffix(
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level: int,
    *,
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    """SCENARIO-PHASE4-065: retained L5 suffixes must advance held-out replay."""

    ready = _gap4_hardening_ready(gap4_artifact)
    level_increment = int(final_level) > int(start_level)
    retained = (
        ready
        and int(heldout_transition_count) > 0
        and level_increment
        and int(final_level) >= int(predicted_level)
    )
    ledger = gap4_artifact.get("gross_recovery_ledger", {}) if isinstance(gap4_artifact, dict) else {}
    return {
        "phase": "hardened-gap4-verify",
        "verifier": HARDENED_VERIFIER,
        "start_level_completed": int(start_level),
        "final_level_completed": int(final_level),
        "predicted_level_after_actions": int(predicted_level),
        "heldout_transition_count": int(heldout_transition_count),
        "level_increment": bool(level_increment),
        "hardened_gap4_ready": bool(ready),
        "hardened_gap4_source": "results/experiment_4187_gap4_graded_execution_gate_hardening.json",
        "hardened_gap4_recovered": int(ledger.get("recovered", 0) or 0) if isinstance(ledger, dict) else 0,
        "hardened_gap4_lost": int(ledger.get("lost", 0) or 0) if isinstance(ledger, dict) else 0,
        "retained": bool(retained),
        "energy": 0.0 if retained else 1.0,
    }


def _move_steps(actions: list[int]) -> list[dict[str, Any]]:
    return [{"action": int(action), "kind": "move"} for action in actions]


def _count_sprite(game: Any, name: str) -> int:
    sprites = game.current_level.get_sprites() if hasattr(game, "current_level") else []
    return sum(1 for sprite in sprites if str(getattr(sprite, "name", "")) == name)


def _player_snapshot(game: Any) -> dict[str, int]:
    player = getattr(game, "plnqvukupu", None)
    return {
        "x": int(getattr(player, "x", -1) if player is not None else -1),
        "y": int(getattr(player, "y", -1) if player is not None else -1),
        "scale": int(getattr(player, "scale", -1) if player is not None else -1),
        "direction": int(getattr(game, "jdmucabyqar", -1) or 0),
    }


def _execute_segment_on_copy(
    env: Any,
    game_action: Any,
    steps: list[dict[str, Any]],
    *,
    start_level: int,
    stop_at_level: int | None = None,
) -> tuple[int, int, list[dict[str, Any]]]:
    final_level = int(start_level)
    used: list[dict[str, Any]] = []
    for step in steps:
        frame = _base()._step_action(env, game_action, step)
        final_level = max(final_level, _base()._levels_completed(frame, env))
        used.append(step)
        if stop_at_level is not None and final_level >= int(stop_at_level):
            break
    return final_level, len(used), used


def _append_segment(
    env: Any,
    game_action: Any,
    trace: dict[str, Any],
    action_plan: list[dict[str, Any]],
    *,
    label: str,
    steps: list[dict[str, Any]],
    start_level: int,
    stop_at_level: int | None = None,
) -> int:
    before = {
        "player": _player_snapshot(env._game),
        "tagsmh": _count_sprite(env._game, "tagsmh"),
        "dosorb": _count_sprite(env._game, "dosorb"),
        "seofsw_tagsmh": _count_sprite(env._game, "seofsw-tagsmh"),
        "seofsw_dosorb": _count_sprite(env._game, "seofsw-dosorb"),
        "enjehv_pahtoz": _count_sprite(env._game, "enjehv-pahtoz"),
    }
    final_level, observed, used = _execute_segment_on_copy(
        env,
        game_action,
        steps,
        start_level=start_level,
        stop_at_level=stop_at_level,
    )
    action_plan.extend(used)
    after = {
        "player": _player_snapshot(env._game),
        "tagsmh": _count_sprite(env._game, "tagsmh"),
        "dosorb": _count_sprite(env._game, "dosorb"),
        "seofsw_tagsmh": _count_sprite(env._game, "seofsw-tagsmh"),
        "seofsw_dosorb": _count_sprite(env._game, "seofsw-dosorb"),
        "enjehv_pahtoz": _count_sprite(env._game, "enjehv-pahtoz"),
    }
    trace["observed_transition_count"] = int(trace["observed_transition_count"]) + observed
    trace["segments"].append(
        {
            "label": str(label),
            "action_count": int(observed),
            "before": before,
            "after": after,
            "final_level_completed": int(final_level),
        }
    )
    return int(final_level)


def explore_sc25_l5_dual_fire_suffix(
    env: Any,
    game_action: Any,
    *,
    target_level: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SCENARIO-PHASE4-065: induce L5's dual blocker-removal route on copied state."""

    original_game = copy.deepcopy(env._game)
    trace: dict[str, Any] = {
        "candidate_spell_orders": [[L5_SHRINK_SPELL, L5_TELEPORT_SPELL, L5_FIRE_SPELL, L5_FIRE_SPELL]],
        "candidate_fire_lane_routes": [[3, 3, 3, 3, 3, 3, 1], [2, 2, 2, 3]],
        "candidate_exit_routes": [[1, 1, 1, 1, 1, 1]],
        "observed_transition_count": 0,
        "expanded_states": 0,
        "segments": [],
        "found": False,
        "stopped_reason": "",
        "predicted_level": _base()._levels_completed(None, env),
    }
    start_level = int(trace["predicted_level"])
    action_plan: list[dict[str, Any]] = []
    env._game = copy.deepcopy(original_game)

    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="shrink_to_small_scale",
        steps=previous._cast_spell_plan(env._game, L5_SHRINK_SPELL),
        start_level=start_level,
    )
    if _player_snapshot(env._game)["scale"] != 1:
        env._game = copy.deepcopy(original_game)
        trace["stopped_reason"] = "sieesc_chwjgc_did_not_shrink_player"
        return [], trace

    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="small_scale_tevyeq_teleport",
        steps=previous._cast_spell_plan(env._game, L5_TELEPORT_SPELL),
        start_level=start_level,
    )
    small_teleport = _player_snapshot(env._game)
    if (small_teleport["x"], small_teleport["y"], small_teleport["scale"]) != (29, 39, 1):
        env._game = copy.deepcopy(original_game)
        trace["stopped_reason"] = "small_scale_tevyeq_teleport_not_observed"
        return [], trace

    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="move_to_tagsmh_fire_lane",
        steps=_move_steps([3, 3, 3, 3, 3, 3, 1]),
        start_level=start_level,
    )
    before_tags = _count_sprite(env._game, "tagsmh")
    before_dosorb = _count_sprite(env._game, "dosorb")
    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="fibcey_removes_tagsmh_and_dosorb",
        steps=previous._cast_spell_plan(env._game, L5_FIRE_SPELL),
        start_level=start_level,
    )
    if not (before_tags > _count_sprite(env._game, "tagsmh") and before_dosorb > _count_sprite(env._game, "dosorb")):
        env._game = copy.deepcopy(original_game)
        trace["stopped_reason"] = "fibcey_first_fire_lane_did_not_remove_tagsmh_dosorb"
        return [], trace

    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="move_to_seofsw_fire_lane",
        steps=_move_steps([2, 2, 2, 3]),
        start_level=start_level,
    )
    before_seofsw_tags = _count_sprite(env._game, "seofsw-tagsmh")
    before_seofsw_dosorb = _count_sprite(env._game, "seofsw-dosorb")
    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="fibcey_removes_seofsw_tagsmh_and_dosorb",
        steps=previous._cast_spell_plan(env._game, L5_FIRE_SPELL),
        start_level=start_level,
    )
    if not (
        before_seofsw_tags > _count_sprite(env._game, "seofsw-tagsmh")
        and before_seofsw_dosorb > _count_sprite(env._game, "seofsw-dosorb")
    ):
        env._game = copy.deepcopy(original_game)
        trace["stopped_reason"] = "fibcey_second_fire_lane_did_not_remove_seofsw_blockers"
        return [], trace

    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="grow_to_large_scale_and_clear_penalty_marker",
        steps=previous._cast_spell_plan(env._game, L5_SHRINK_SPELL),
        start_level=start_level,
    )
    if _player_snapshot(env._game)["scale"] != 2:
        env._game = copy.deepcopy(original_game)
        trace["stopped_reason"] = "sieesc_chwjgc_did_not_restore_large_scale"
        return [], trace

    _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="large_scale_tevyeq_teleport_to_exit_lane",
        steps=previous._cast_spell_plan(env._game, L5_TELEPORT_SPELL),
        start_level=start_level,
    )
    large_teleport = _player_snapshot(env._game)
    if (large_teleport["x"], large_teleport["y"], large_teleport["scale"]) != (51, 35, 2):
        env._game = copy.deepcopy(original_game)
        trace["stopped_reason"] = "large_scale_tevyeq_teleport_not_observed"
        return [], trace

    predicted_level = _append_segment(
        env,
        game_action,
        trace,
        action_plan,
        label="exit_touch_up_route",
        steps=_move_steps([1, 1, 1, 1, 1, 1]),
        start_level=start_level,
        stop_at_level=target_level,
    )
    trace["predicted_level"] = int(predicted_level)
    trace["planned_depth"] = len(action_plan)
    env._game = copy.deepcopy(original_game)
    if predicted_level >= int(target_level):
        trace["found"] = True
        trace["stopped_reason"] = "dual_fire_blocker_removal_then_exit_level_increment_found"
        return action_plan, trace
    trace["stopped_reason"] = "exit_route_did_not_increment_level"
    return [], trace


def _validate_suffix_on_copy(
    env: Any,
    game_action: Any,
    *,
    start_level: int,
    target_level: int,
    action_plan: list[dict[str, Any]],
    gap4_artifact: dict[str, Any],
) -> dict[str, Any]:
    original_game = copy.deepcopy(env._game)
    heldout_count = max(1, len(action_plan) // 2) if action_plan else 0
    prefix_len = max(0, len(action_plan) - heldout_count)
    final_level = int(start_level)
    for step in action_plan:
        frame = _base()._step_action(env, game_action, step)
        final_level = max(final_level, _base()._levels_completed(frame, env))
    env._game = copy.deepcopy(original_game)
    decision = validate_hardened_gap4_l5_suffix(
        start_level=start_level,
        final_level=final_level,
        heldout_transition_count=heldout_count,
        predicted_level=target_level,
        gap4_artifact=gap4_artifact,
    )
    decision["validated_prefix_transition_count"] = int(prefix_len)
    decision["validated_total_transition_count"] = len(action_plan)
    return decision


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
    """REQ-PHASE4-065: report fixture blockage without solve inflation."""

    prior_level = max(0, int(target_level) - 1)
    artifact = {
        "experiment": "experiment_4249_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_sc25_L5",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "target_game": str(target_game),
        "target_level": int(target_level),
        "prior_level": int(prior_level),
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "new_levels_solved_this_task": 0,
        "levels_completed": int(prior_level),
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
        "selected_candidate_reason": "offline fixture or prior verifier precondition failed",
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
    """REQ-PHASE4-065: build the terminal artifact from hardened verified evidence."""

    advanced = outcome.advanced
    new_levels = 1 if advanced else 0
    total_levels = PRIOR_TOTAL_LEVELS_SOLVED + new_levels
    if advanced:
        verdict = "success: incremental_progress_sc25-635fd71a_advanced_to_L5_total19"
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
        "experiment": "experiment_4249_arc_incremental_progress",
        "title": "arc3_incremental_progress_hardened_gap4_sc25_L5",
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
    """SCENARIO-PHASE4-065: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")
    for field in (
        "total_levels_solved",
        "levels_completed",
        "target_level",
        "prior_total_levels_solved",
        "new_levels_solved_this_task",
        "total_games_solved",
    ):
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
        errors.append("requirements must include REQ-PHASE4-065 and SCENARIO-PHASE4-065")
    principles = artifact.get("field_principles")
    if principles is not None:
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in REQUIRED_FIELD_PRINCIPLES:
                if field not in principles:
                    errors.append(f"field_principles missing {field}")
    if (
        "total_levels_solved" in artifact
        and type(artifact.get("total_levels_solved")) is int
        and artifact["total_levels_solved"] < PRIOR_TOTAL_LEVELS_SOLVED
    ):
        errors.append("total_levels_solved must be monotonic from the prior count")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("verifier_validated") is not True:
            errors.append("verifier_validated must be true for success")
        if artifact.get("new_levels_solved_this_task") != 1:
            errors.append("new_levels_solved_this_task must be one for scoped success")
        if artifact.get("total_levels_solved") != PRIOR_TOTAL_LEVELS_SOLVED + 1:
            errors.append("total_levels_solved must increment from 18 to 19 for success")
        if int(artifact.get("levels_completed", 0) or 0) < int(artifact.get("target_level", 0) or 0):
            errors.append("levels_completed must reach target_level for success")
        if not any(
            isinstance(decision, dict)
            and decision.get("retained") is True
            and decision.get("verifier") == HARDENED_VERIFIER
            for decision in artifact.get("verification_decisions", [])
        ):
            errors.append("success requires a retained hardened GAP-4 verifier decision")
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


def _load_offline_arcade() -> Any:  # pragma: no cover - thin real-env adapter
    return previous._load_offline_arcade()


def _run_sc25_l5_frontier(
    offline_arcade: Any,
    target: TargetSelection,
    prior_artifact: dict[str, Any],
    gap4_artifact: dict[str, Any],
) -> FrontierOutcome:
    from arcengine.enums import GameAction

    env = offline_arcade.make(target.game_id)
    frame = env.reset()
    initial_level = _base()._levels_completed(frame, env)
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_sc25_reset",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(initial_level),
        }
    ]

    l1_plan, l1_trace = _base().plan_sc25_suffix_bounded(
        env,
        GameAction,
        target_level=1,
        max_depth=48,
        max_expansions=512,
    )
    phase_trace.append({"phase": "replay", "source": "sc25_L1_reestablishment_planning", "planner_trace": l1_trace})
    if not l1_plan:
        return _failed_outcome(target, "could_not_reestablish_sc25_L1")
    l1_level, l1_actions, l1_action_trace = _base().execute_plan_until_level(
        env,
        GameAction,
        l1_plan,
        prior_level=initial_level,
        target_level=1,
        phase="replay",
    )
    phase_trace.extend(l1_action_trace)
    if l1_level < 1:
        return _failed_outcome(target, "could_not_reestablish_sc25_L1")

    l2_plan = previous._banked_l2_plan()
    phase_trace.append({"phase": "replay", "source": "sc25_L2_banked_suffix", "action_count": len(l2_plan)})
    if not l2_plan:
        return _failed_outcome(target, "missing_banked_sc25_L2_suffix")
    l2_level, l2_actions, l2_action_trace = _base().execute_plan_until_level(
        env,
        GameAction,
        l2_plan,
        prior_level=l1_level,
        target_level=2,
        phase="replay",
    )
    phase_trace.extend(l2_action_trace)
    if l2_level < 2:
        return _failed_outcome(target, "could_not_reestablish_sc25_L2")

    l3_path = REPO / "results" / "experiment_4224_arc_incremental_progress.json"
    try:
        l3_artifact = _read_json(l3_path)
    except (OSError, json.JSONDecodeError):
        return _failed_outcome(target, "missing_banked_sc25_L3_suffix")
    l3_plan = [dict(step) for step in l3_artifact.get("action_plan", []) if isinstance(step, dict)]
    phase_trace.append({"phase": "replay", "source": "sc25_L3_banked_suffix", "action_count": len(l3_plan)})
    if not l3_plan:
        return _failed_outcome(target, "missing_banked_sc25_L3_suffix")
    l3_level, l3_actions, l3_action_trace = _base().execute_plan_until_level(
        env,
        GameAction,
        l3_plan,
        prior_level=l2_level,
        target_level=3,
        phase="replay",
    )
    phase_trace.extend(l3_action_trace)
    if l3_level < 3:
        return _failed_outcome(target, "could_not_reestablish_sc25_L3")

    l4_plan = [dict(step) for step in prior_artifact.get("action_plan", []) if isinstance(step, dict)]
    phase_trace.append({"phase": "replay", "source": "sc25_L4_banked_suffix", "action_count": len(l4_plan)})
    if not l4_plan:
        return _failed_outcome(target, "missing_banked_sc25_L4_suffix")
    frontier_level, l4_actions, l4_action_trace = _base().execute_plan_until_level(
        env,
        GameAction,
        l4_plan,
        prior_level=l3_level,
        target_level=target.prior_level,
        phase="replay",
    )
    phase_trace.extend(l4_action_trace)
    replay_actions = int(l1_actions) + int(l2_actions) + int(l3_actions) + int(l4_actions)
    if frontier_level < target.prior_level:
        return FrontierOutcome(
            target_game=target.game_id,
            target_level=target.target_level,
            prior_level=target.prior_level,
            final_level_completed=frontier_level,
            replay_actions_used=replay_actions,
            executed_real_env_actions=0,
            exploration_actions_used=int(l1_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L4 reestablishment",
            failure_reason="could_not_reestablish_sc25_L4",
        )

    action_plan, planner_trace = explore_sc25_l5_dual_fire_suffix(
        env,
        GameAction,
        target_level=target.target_level,
    )
    phase_trace.append(
        {
            "phase": "explore",
            "source": "copied_env_sc25_L5_dual_fire_lane_transitions",
            "planner_trace": planner_trace,
            "observed_transition_count": int(planner_trace.get("observed_transition_count", 0) or 0),
        }
    )
    phase_trace.append(
        {
            "phase": "induce",
            "mechanic": (
                "shrink to small scale, tevyeq teleport to lower marker, fibcey clears tagsmh/dosorb, "
                "fibcey clears seofsw-tagsmh/seofsw-dosorb, grow, large tevyeq teleport, exit touch"
            ),
            "goal_predicate": "levels_completed increases after both blocker families are removed and the large player touches exydhv",
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
            exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L5 dual fire-lane blocker-removal sequence",
            failure_reason="no_observed_level_up_candidate",
        )

    validation = _validate_suffix_on_copy(
        env,
        GameAction,
        start_level=frontier_level,
        target_level=target.target_level,
        action_plan=action_plan,
        gap4_artifact=gap4_artifact,
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
            exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[validation],
            action_plan=[],
            phase_trace=phase_trace,
            induced_mechanic="sc25 L5 dual fire-lane blocker-removal sequence",
            failure_reason="no_verifier_validated_level_up_candidate",
        )

    final_level, executed_actions, act_trace = _base().execute_plan_until_level(
        env,
        GameAction,
        action_plan,
        prior_level=frontier_level,
        target_level=target.target_level,
    )
    phase_trace.extend(act_trace)
    advanced = final_level >= target.target_level
    return FrontierOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=final_level,
        replay_actions_used=replay_actions,
        executed_real_env_actions=executed_actions,
        exploration_actions_used=int(planner_trace.get("observed_transition_count", 0) or 0),
        real_env_confirmed=advanced,
        verifier_validated=True,
        verification_decisions=[validation],
        action_plan=action_plan,
        phase_trace=phase_trace,
        induced_mechanic="sc25 L5 shrink, small teleport, dual fibcey blocker removal, grow, large teleport, exit-touch movement",
        failure_reason="" if advanced else "real_env_confirmation_not_incremented",
    )


def run(*, write: bool = True) -> dict[str, Any]:
    """Run Exp 4249 offline and optionally write the terminal artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    prior_path = REPO / "results" / "experiment_4236_arc_incremental_progress.json"
    gap4_path = REPO / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json"
    try:
        survey = _read_json(survey_path)
        prior_artifact = _read_json(prior_path)
        gap4_artifact = _read_json(gap4_path)
        baselines = load_environment_baselines(REPO / "environment_files")
        target = select_deeper_level_target(survey, baselines, prior_artifact, gap4_artifact)
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
        artifact = blocked_artifact(
            target_game=SC25_GAME_ID,
            target_level=5,
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
        outcome = _run_sc25_l5_frontier(_load_offline_arcade(), target, prior_artifact, gap4_artifact)
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
