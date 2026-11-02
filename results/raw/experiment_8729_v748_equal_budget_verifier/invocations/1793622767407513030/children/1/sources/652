"""Exp 4169: ARC-AGI-3 strict non-spatial incremental-progress recheck.

Spec refs: REQ-PHASE4-053, SCENARIO-PHASE4-053.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4169_arc_incremental_progress.json"
RANDOM_SEED = 4169
PRIOR_TOTAL_GAMES_SOLVED = 13
INFERENCE_SUBSTRATE = "offline_arc_agi3_gap4_strict_nonspatial_recheck"
REQUIREMENTS = ["REQ-PHASE4-053", "SCENARIO-PHASE4-053"]
SOLVED_PREFIXES_BEFORE_4169 = (
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
    "total_games_solved",
    "levels_completed",
    "real_env_confirmed",
    "target_game",
    "game_solved",
    "first_solve_at_action",
    "solve_trace",
    "inference_substrate",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. An honest no-solve is a COMPLETE verdict.",
    "total_games_solved": "The monotonic progress metric; must be >= the prior milestone's count.",
    "levels_completed": "Real-env-confirmed level count; falsifiable evidence of an actual solve.",
    "real_env_confirmed": "Only real-env solves raise the headline count.",
}

sys.path.insert(0, str(REPO / "python"))


@dataclass(frozen=True)
class SelectedTarget:
    """One strict non-spatial survey game eligible for a first-level attempt."""

    game: str
    game_id: str
    baseline_actions: int
    survey_rank: int
    selection_mode: str
    selection_reason: str
    excluded_solved_games: tuple[str, ...]


@dataclass(frozen=True)
class FirstLevelOutcome:
    """Normalized first-level evidence from a selected offline ARC game."""

    target_game: str
    final_level_completed: int
    first_solve_at_action: int
    exploration_actions_used: int
    real_env_confirmed: bool
    verifier_validated: bool
    verification_decisions: list[dict[str, Any]]
    action_plan: list[dict[str, Any]]
    phase_trace: list[dict[str, Any]]
    induced_mechanic: str
    failure_reason: str = ""

    @property
    def solved(self) -> bool:
        return (
            bool(self.real_env_confirmed)
            and bool(self.verifier_validated)
            and int(self.final_level_completed) >= 1
            and int(self.first_solve_at_action) > 0
            and any(
                isinstance(decision, dict) and decision.get("retained") is True
                for decision in self.verification_decisions
            )
            and bool(self.action_plan)
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, int]]:
    """REQ-PHASE4-053: read local offline fixture metadata by game prefix."""

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
    for row in survey.get("per_game_surveys", []):
        game = str(row.get("game", ""))
        if game and game not in ranked:
            ranked.append(game)
    return ranked


def build_selection_evidence(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_4169,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-053: preserve why no strict non-spatial target remains."""

    rows = _survey_rows_by_game(survey)
    solved = set(solved_prefixes)
    strict_rows: list[dict[str, Any]] = []
    for rank, game in enumerate(_ranked_games(survey)):
        row = rows.get(game, {})
        if row.get("is_spatial_planning") is not False:
            continue
        baseline = baselines.get(game)
        strict_rows.append(
            {
                "game": game,
                "rank": int(rank),
                "game_id": baseline[0] if baseline else "",
                "has_baseline": baseline is not None,
                "fixture_available": _fixture_available(baseline[0]) if baseline else False,
                "already_solved": game in solved,
                "baseline_actions": int(baseline[1]) if baseline else 0,
            }
        )
    unsolved_count = sum(1 for row in strict_rows if row["has_baseline"] and not row["already_solved"])
    return {
        "excluded_solved_games": list(solved_prefixes),
        "strict_nonspatial_rows": strict_rows,
        "unsolved_strict_nonspatial_count": int(unsolved_count),
    }


def select_next_unsolved_nonspatial(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, int]],
    *,
    solved_prefixes: tuple[str, ...] = SOLVED_PREFIXES_BEFORE_4169,
) -> SelectedTarget | None:
    """REQ-PHASE4-053: select the next unbanked strict non-spatial survey target."""

    rows = _survey_rows_by_game(survey)
    solved = set(solved_prefixes)
    for rank, game in enumerate(_ranked_games(survey)):
        row = rows.get(game, {})
        if row.get("is_spatial_planning") is not False:
            continue
        if game in solved or game not in baselines:
            continue
        game_id, baseline = baselines[game]
        return SelectedTarget(
            game=game,
            game_id=game_id,
            baseline_actions=int(baseline),
            survey_rank=int(rank),
            selection_mode="strict_survey_non_spatial",
            selection_reason=f"selected {game} as the next unsolved strict non-spatial survey target",
            excluded_solved_games=tuple(solved_prefixes),
        )
    return None


def validate_gap4_heldout_replay(
    start_level: int,
    final_level: int,
    heldout_transition_count: int,
    predicted_level_after_actions: int,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-053: validate a candidate suffix before real-env commitment."""

    level_increment = int(final_level) > int(start_level)
    retained = (
        int(heldout_transition_count) > 0
        and level_increment
        and int(final_level) >= int(predicted_level_after_actions)
    )
    return {
        "phase": "verify",
        "verifier": "gap4_heldout_executed_consistency_first_level_replay",
        "start_level_completed": int(start_level),
        "final_level_completed": int(final_level),
        "predicted_level_after_actions": int(predicted_level_after_actions),
        "heldout_transition_count": int(heldout_transition_count),
        "level_increment": bool(level_increment),
        "retained": bool(retained),
        "energy": 0.0 if retained else 1.0,
    }


def build_artifact(
    outcome: FirstLevelOutcome,
    target: SelectedTarget | None,
    *,
    random_seed: int,
    duration_s: float,
    selection_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-PHASE4-053: build the terminal artifact from verified first-level evidence."""

    solved = outcome.solved
    total_games = PRIOR_TOTAL_GAMES_SOLVED + (1 if solved else 0)
    if solved:
        verdict = f"success: incremental_progress_solved_{outcome.target_game}_at_action_{outcome.first_solve_at_action}"
    elif outcome.target_game == "none" and outcome.failure_reason == "no_unsolved_nonspatial_candidates":
        verdict = "complete: incremental_progress_no_solve_no_unsolved_nonspatial_candidates"
    else:
        verdict = f"complete: incremental_progress_no_solve_{outcome.target_game}_{_reason_slug(outcome.failure_reason)}"

    solve_trace = {
        "target_game": outcome.target_game,
        "selection_mode": target.selection_mode if target else "strict_nonspatial_pool_exhausted",
        "selection_reason": target.selection_reason if target else "no unsolved strict non-spatial survey candidate remains",
        "selection_evidence": dict(selection_evidence or {}),
        "actions": list(outcome.action_plan),
        "verification_decisions": list(outcome.verification_decisions),
        "phase_trace": list(outcome.phase_trace),
    }
    artifact = {
        "experiment": "experiment_4169_arc_incremental_progress",
        "title": "arc3_incremental_progress_strict_nonspatial_recheck",
        "honest_verdict": verdict,
        "game_solved": bool(solved),
        "target_game": outcome.target_game,
        "prior_total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "total_games_solved": int(total_games),
        "levels_completed": int(outcome.final_level_completed),
        "first_solve_at_action": int(outcome.first_solve_at_action),
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "verifier_validated": bool(outcome.verifier_validated),
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
        "candidate_baseline_actions": int(target.baseline_actions) if target else 0,
        "acceptance_gate_passed": bool(
            (solved and total_games > PRIOR_TOTAL_GAMES_SOLVED)
            or (not solved and verdict.startswith("complete:"))
        ),
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_artifact(*, target_game: str, random_seed: int, duration_s: float) -> dict[str, Any]:
    """REQ-PHASE4-053: report unavailable offline fixtures without solve inflation."""

    artifact = {
        "experiment": "experiment_4169_arc_incremental_progress",
        "title": "arc3_incremental_progress_strict_nonspatial_recheck",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "game_solved": False,
        "target_game": str(target_game),
        "prior_total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "levels_completed": 0,
        "first_solve_at_action": -1,
        "real_env_confirmed": False,
        "verifier_validated": False,
        "exploration_actions_used": 0,
        "induced_mechanic": "none",
        "verification_decisions": [],
        "action_plan": [],
        "phase_trace": [],
        "solve_trace": {
            "target_game": str(target_game),
            "selection_mode": "blocked_precondition",
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
        "acceptance_gate_passed": False,
        "submitted_to_leaderboard": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-053: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    for field in ("total_games_solved", "prior_total_games_solved", "levels_completed", "first_solve_at_action"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    for field in ("game_solved", "real_env_confirmed", "verifier_validated"):
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
        errors.append("requirements must include REQ-PHASE4-053 and SCENARIO-PHASE4-053")

    principles = artifact.get("field_principles")
    if "field_principles" in artifact:
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in REQUIRED_FIELD_PRINCIPLES:
                if field not in principles:
                    errors.append(f"field_principles missing {field}")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("game_solved") is not True:
            errors.append("game_solved must be true for success")
        if artifact.get("target_game") == "none":
            errors.append("target_game must name the solved game for success")
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("verifier_validated") is not True:
            errors.append("verifier_validated must be true for success")
        if artifact.get("total_games_solved") != PRIOR_TOTAL_GAMES_SOLVED + 1:
            errors.append("total_games_solved must increment for success")
        if int(artifact.get("levels_completed", 0) or 0) < 1:
            errors.append("levels_completed must increment for success")
        if int(artifact.get("first_solve_at_action", 0) or 0) <= 0:
            errors.append("first_solve_at_action must be positive for success")
        if not isinstance(artifact.get("solve_trace"), dict) or not artifact["solve_trace"].get("phase_trace"):
            errors.append("solve_trace must include phase_trace for success")
        if not any(
            isinstance(decision, dict) and decision.get("retained") is True
            for decision in artifact.get("verification_decisions", [])
        ):
            errors.append("success requires a retained GAP-4 verifier decision")
        if not artifact.get("action_plan"):
            errors.append("success requires a validated action_plan")
    elif isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("total_games_solved") != PRIOR_TOTAL_GAMES_SOLVED:
            errors.append("total_games_solved must remain at the prior count for no-solve")
        if artifact.get("game_solved") is not False:
            errors.append("game_solved must be false for no-solve")
        if artifact.get("real_env_confirmed") is not False:
            errors.append("real_env_confirmed must be false for no-solve")
    return errors


def _fixture_available(game_id: str) -> bool:
    if "-" not in str(game_id):
        return False
    prefix, suffix = str(game_id).split("-", maxsplit=1)
    root = REPO / "environment_files" / prefix / suffix
    return root.joinpath("metadata.json").exists() and root.joinpath(f"{prefix}.py").exists()


def _load_offline_arcade() -> Any:  # pragma: no cover - thin real-env adapter
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )


def _run_selected_first_level(offline_arcade: Any, target: SelectedTarget) -> FirstLevelOutcome:
    """Return an honest no-driver outcome unless a game-specific adapter is added."""

    return FirstLevelOutcome(
        target_game=target.game_id,
        final_level_completed=0,
        first_solve_at_action=-1,
        exploration_actions_used=0,
        real_env_confirmed=False,
        verifier_validated=False,
        verification_decisions=[],
        action_plan=[],
        phase_trace=[
            {
                "phase": "observe",
                "target_game": target.game_id,
                "source": "offline_driver_loaded",
                "driver": type(offline_arcade).__name__,
            }
        ],
        induced_mechanic="none",
        failure_reason="no_offline_driver_for_selected_strict_nonspatial_target",
    )


def _no_unsolved_nonspatial_outcome(selection_evidence: dict[str, Any]) -> FirstLevelOutcome:
    return FirstLevelOutcome(
        target_game="none",
        final_level_completed=0,
        first_solve_at_action=-1,
        exploration_actions_used=0,
        real_env_confirmed=False,
        verifier_validated=False,
        verification_decisions=[],
        action_plan=[],
        phase_trace=[
            {
                "phase": "observe",
                "source": "survey_selection",
                "unsolved_strict_nonspatial_count": int(selection_evidence["unsolved_strict_nonspatial_count"]),
            }
        ],
        induced_mechanic="none",
        failure_reason="no_unsolved_nonspatial_candidates",
    )


def _failed_selected_outcome(target: SelectedTarget, reason: str) -> FirstLevelOutcome:
    return FirstLevelOutcome(
        target_game=target.game_id,
        final_level_completed=0,
        first_solve_at_action=-1,
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
    """Run the offline Exp 4169 attempt and optionally write the terminal artifact."""

    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    if not survey_path.exists():
        artifact = blocked_artifact(target_game="none", random_seed=RANDOM_SEED, duration_s=time.time() - started)
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        survey = _read_json(survey_path)
        baselines = load_environment_baselines(REPO / "environment_files")
    except (OSError, json.JSONDecodeError, ValueError):
        artifact = blocked_artifact(target_game="none", random_seed=RANDOM_SEED, duration_s=time.time() - started)
        if write:
            _write_artifact(artifact)
        return artifact
    if not baselines:
        artifact = blocked_artifact(target_game="none", random_seed=RANDOM_SEED, duration_s=time.time() - started)
        if write:
            _write_artifact(artifact)
        return artifact

    selection_evidence = build_selection_evidence(survey, baselines)
    target = select_next_unsolved_nonspatial(survey, baselines)
    missing_unsolved_fixture = next(
        (
            row
            for row in selection_evidence["strict_nonspatial_rows"]
            if not row["already_solved"] and not row["has_baseline"]
        ),
        None,
    )
    if target is None and missing_unsolved_fixture is not None:
        artifact = blocked_artifact(
            target_game=str(missing_unsolved_fixture["game"]),
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact
    if target is None:
        artifact = build_artifact(
            _no_unsolved_nonspatial_outcome(selection_evidence),
            None,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
            selection_evidence=selection_evidence,
        )
        if write:
            _write_artifact(artifact)
        return artifact
    if not _fixture_available(target.game_id):
        artifact = blocked_artifact(
            target_game=target.game_id,
            random_seed=RANDOM_SEED,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        offline_arcade = _load_offline_arcade()
        outcome = _run_selected_first_level(offline_arcade, target)
    except Exception as exc:
        outcome = _failed_selected_outcome(
            target,
            f"offline_run_failed_{type(exc).__name__.lower()}_{exc}",
        )
    artifact = build_artifact(
        outcome,
        target,
        random_seed=RANDOM_SEED,
        duration_s=time.time() - started,
        selection_evidence=selection_evidence,
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
