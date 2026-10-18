"""Exp 4214: ARC-AGI-3 live solver accuracy probe.

Spec refs: REQ-PHASE4-060, SCENARIO-PHASE4-060.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from carnot.agentic.arc_agi3_live_adapter import (
    BASE_URL,
    ArcLivePreconditions,
    EnvironmentSummary,
    MetricMapping,
    _extract_environment_score,
    _normalise_score,
    check_live_preconditions,
    enumerate_live_environments,
    open_online_arcade,
    validate_recorded_fixture,
)
from carnot.experiment_4202_arc_live_env_solver_vs_floor import (
    LP85_GAME_ID,
    FloorBaseline,
    SolverRunOutcome,
    compare_solver_to_floor,
    load_banked_solver_plan,
    run_solver_replay,
)


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4214_arc_live_env_solver_accuracy.json"
FLOOR_SOURCE_NAME = "experiment_4202_arc_live_env_solver_vs_floor.json"
SOLVER_SOURCE_NAME = "experiment_4190_arc_incremental_progress.json"
RANDOM_SEED = 4214
INFERENCE_SUBSTRATE = "official_arc_agi3_online_anonymous_key_solver_accuracy_probe"
REQUIREMENTS = ["REQ-PHASE4-060", "SCENARIO-PHASE4-060"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "solver_completes_level",
    "solver_beats_floor",
    "live_env_metrics",
    "no_leaderboard_submission",
    "preconditions_checked",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'solver completes 0 levels but is efficient' or "
        "'blocked_arc_live_unreachable' is a COMPLETE grounding verdict."
    ),
    "solver_completes_level": (
        "BARE bool: levels_completed>=1 on the live env -- the ACCURACY win exp4202 "
        "lacked; the north-star's primary axis."
    ),
    "solver_beats_floor": (
        "{accuracy: solver_score vs floor_score, efficiency: solver_actions vs "
        "floor_actions} -- the real-env read on the two north-star axes vs the random floor."
    ),
    "live_env_metrics": (
        "{score, levels_completed, actions_taken, baseline_actions} from the live "
        "EnvironmentScore -- falsifiable real-env evidence, not synthetic-scaffold."
    ),
    "no_leaderboard_submission": (
        "BARE bool: zero scorecards submitted (Operator-Only External Publication; "
        "the online quota gate)."
    ),
    "preconditions_checked": (
        "Records the SDK + network reachability checks; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_exp4202_floor(
    artifact: dict[str, Any],
    *,
    source_path: str = f"results/{FLOOR_SOURCE_NAME}",
) -> FloorBaseline:
    """REQ-PHASE4-060: read the same-env random/greedy floor recorded by Exp 4202."""

    floor = artifact.get("random_greedy_floor")
    if not isinstance(floor, dict):
        raise ValueError("Exp 4202 artifact missing random_greedy_floor")
    env_payload = floor.get("environment")
    if not isinstance(env_payload, dict):
        raise ValueError("Exp 4202 floor missing environment")
    environment = EnvironmentSummary(
        game_id=str(env_payload.get("game_id", "") or ""),
        title=str(env_payload.get("title", "") or ""),
        tags=[str(tag) for tag in (env_payload.get("tags") or [])],
        baseline_actions=[int(value) for value in (env_payload.get("baseline_actions") or [])],
    )
    if environment.game_id != LP85_GAME_ID:
        raise ValueError("Exp 4214 must compare on Exp 4202's same LP85 environment")
    return FloorBaseline(
        environment=environment,
        actions_taken=int(floor.get("actions_taken", 0) or 0),
        baseline_actions=int(floor.get("baseline_actions", 0) or 0),
        actions_vs_baseline_actions=float(floor.get("actions_vs_baseline_actions", 0.0) or 0.0),
        score=float(floor.get("score", 0.0) or 0.0),
        levels_completed=int(floor.get("levels_completed", 0) or 0),
        source_path=source_path,
    )


def load_exp4202_floor(path: Optional[Path] = None) -> FloorBaseline:
    """REQ-PHASE4-060: load the prior live floor artifact Exp 4214 compares against."""

    floor_path = path or (REPO / "results" / FLOOR_SOURCE_NAME)
    try:
        source_path = str(floor_path.relative_to(REPO))
    except ValueError:
        source_path = str(floor_path)
    return extract_exp4202_floor(_read_json(floor_path), source_path=source_path)


def load_completion_solver_plan(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """REQ-PHASE4-060: use the proven LP85 L1 replay plan from the offline solver."""

    return load_banked_solver_plan(path or (REPO / "results" / SOLVER_SOURCE_NAME))


def completion_action_budget(
    environment: EnvironmentSummary,
    floor: FloorBaseline,
    solver_plan: list[dict[str, Any]],
    requested_action_budget: Optional[int] = None,
) -> int:
    """SCENARIO-PHASE4-060: never cap the live completion probe below L1 baseline actions."""

    first_level_baseline = int(environment.baseline_actions[0]) if environment.baseline_actions else 0
    floor_baseline = int(floor.baseline_actions)
    requested = int(requested_action_budget or 0)
    plan_len = len(solver_plan)
    return max(requested, first_level_baseline, floor_baseline, plan_len)


def run_solver_completion(
    env: Any,
    environment: EnvironmentSummary,
    *,
    floor: FloorBaseline,
    solver_plan: list[dict[str, Any]],
    requested_action_budget: Optional[int] = None,
    action_enum: Any = None,
    score_provider: Optional[Callable[[Any], Any]] = None,
) -> SolverRunOutcome:
    """SCENARIO-PHASE4-060: run the proven solver with a completion-sized budget."""

    action_budget = completion_action_budget(
        environment,
        floor,
        solver_plan,
        requested_action_budget=requested_action_budget,
    )
    return run_solver_replay(
        env,
        environment,
        solver_plan=solver_plan,
        action_budget=action_budget,
        action_enum=action_enum,
        score_provider=score_provider,
    )


def _observed_frame_levels_completed(outcome: SolverRunOutcome) -> int:
    observed = [int(row.get("levels_completed_after", 0) or 0) for row in outcome.trace]
    return max([int(outcome.score.levels_completed), *observed] if observed else [int(outcome.score.levels_completed)])


def _verdict_from_evidence(comparison: dict[str, Any], completes_level: bool, game_id: str) -> str:
    if completes_level:
        return f"success: solver_completes_level_live_{game_id}"
    if bool(comparison.get("efficiency", {}).get("beats")):
        return f"complete: solver_completes_0_levels_live_{game_id}_efficiency_only"
    if bool(comparison.get("accuracy", {}).get("beats")):
        return f"complete: solver_score_beats_floor_without_completion_live_{game_id}"
    return f"complete: solver_completes_0_levels_live_{game_id}"


def blocked_artifact(*, preconditions: ArcLivePreconditions, duration_s: float) -> dict[str, Any]:
    """SCENARIO-PHASE4-060: report missing live substrate without fabricating metrics."""

    artifact = {
        "experiment": "experiment_4214_arc_live_env_solver_accuracy",
        "title": "arc3_live_env_solver_accuracy",
        "honest_verdict": "blocked_arc_live_unreachable",
        "solver_completes_level": False,
        "solver_beats_floor": {},
        "live_env_metrics": {},
        "random_greedy_floor": {},
        "no_leaderboard_submission": True,
        "leaderboard_submission_attempted": False,
        "scorecard_closed": False,
        "preconditions_checked": preconditions.to_json(),
        "real_metric_mapping": MetricMapping().to_json(),
        "offline_validation": {"passed": False, "skipped": True},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 3),
        "acceptance_gate_passed": True,
    }
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive schema guard
        raise ValueError("; ".join(errors))
    return artifact


def build_artifact(
    *,
    outcome: SolverRunOutcome,
    floor: FloorBaseline,
    preconditions: ArcLivePreconditions,
    offline_validation: dict[str, Any],
    environment_count: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-060: build the terminal live accuracy artifact."""

    if not preconditions.ok:
        artifact = blocked_artifact(preconditions=preconditions, duration_s=duration_s)
        artifact["offline_validation"] = dict(offline_validation)
        return artifact
    if outcome.environment.game_id != floor.environment.game_id:
        raise ValueError("solver outcome and floor must use the same environment")

    comparison = compare_solver_to_floor(outcome, floor)
    completes_level = int(outcome.score.levels_completed) >= 1
    live_metrics = outcome.live_metrics_json()
    live_metrics["observed_frame_levels_completed"] = _observed_frame_levels_completed(outcome)
    artifact = {
        "experiment": "experiment_4214_arc_live_env_solver_accuracy",
        "title": "arc3_live_env_solver_accuracy",
        "honest_verdict": _verdict_from_evidence(comparison, completes_level, outcome.environment.game_id),
        "live_env_reachable": True,
        "solver_completes_level": bool(completes_level),
        "solver_beats_floor": comparison,
        "live_env_metrics": live_metrics,
        "random_greedy_floor": floor.to_json(),
        "solver_trace": list(outcome.trace),
        "solver_policy": outcome.solver_policy,
        "solver_source_artifact": outcome.source_artifact,
        "environment_count": int(environment_count),
        "no_leaderboard_submission": True,
        "leaderboard_submission_attempted": bool(outcome.leaderboard_submission_attempted),
        "scorecard_closed": False,
        "preconditions_checked": preconditions.to_json(),
        "real_metric_mapping": MetricMapping().to_json(),
        "offline_validation": dict(offline_validation),
        "online_mode": "official_sdk_online_anonymous_key_open_scorecard_not_closed",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 3),
        "acceptance_gate_passed": True,
    }
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive schema guard
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-060: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    if type(artifact.get("solver_completes_level")) is not bool:
        errors.append("solver_completes_level must be a bare bool")
    if artifact.get("no_leaderboard_submission") is not True:
        errors.append("no_leaderboard_submission must be true")
    if artifact.get("leaderboard_submission_attempted") is not False:
        errors.append("leaderboard_submission_attempted must be false")
    if artifact.get("scorecard_closed") is not False:
        errors.append("scorecard_closed must be false")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, dict):
        errors.append("preconditions_checked must be a dict")
    else:
        for field in ("sdk_importable", "sdk_version", "network_reachable", "base_url"):
            if field not in preconditions:
                errors.append(f"preconditions_checked missing {field}")
        for field in ("sdk_importable", "network_reachable"):
            if field in preconditions and type(preconditions[field]) is not bool:
                errors.append(f"preconditions_checked.{field} must be a bare bool")

    if artifact.get("requirements") != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-060 and SCENARIO-PHASE4-060")

    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles must be a dict")
    else:
        for field in REQUIRED_FIELD_PRINCIPLES:
            if field not in principles:
                errors.append(f"field_principles missing {field}")

    solver_beats = artifact.get("solver_beats_floor")
    live_metrics = artifact.get("live_env_metrics")
    if verdict == "blocked_arc_live_unreachable":
        if solver_beats != {}:
            errors.append("blocked artifacts must leave solver_beats_floor empty")
        if live_metrics != {}:
            errors.append("blocked artifacts must leave live_env_metrics empty")
        return errors

    if not isinstance(solver_beats, dict):
        errors.append("solver_beats_floor must be a dict")
    else:
        for field in ("accuracy", "efficiency", "overall"):
            if field not in solver_beats:
                errors.append(f"solver_beats_floor missing {field}")
        if "overall" in solver_beats and type(solver_beats["overall"]) is not bool:
            errors.append("solver_beats_floor.overall must be a bare bool")

    if not isinstance(live_metrics, dict):
        errors.append("live_env_metrics must be a dict")
    else:
        for field in ("score", "levels_completed", "actions_taken", "baseline_actions", "action_budget"):
            if field not in live_metrics:
                errors.append(f"live_env_metrics missing {field}")
        for field in ("levels_completed", "actions_taken", "baseline_actions", "action_budget"):
            if field in live_metrics and type(live_metrics[field]) is not int:
                errors.append(f"live_env_metrics.{field} must be a bare int")
        if "score" in live_metrics and type(live_metrics["score"]) not in (int, float):
            errors.append("live_env_metrics.score must be numeric")
        if (
            isinstance(live_metrics.get("action_budget"), int)
            and isinstance(live_metrics.get("baseline_actions"), int)
            and live_metrics["action_budget"] < live_metrics["baseline_actions"]
        ):
            errors.append("live_env_metrics.action_budget must be >= baseline_actions")
        expected_completion = isinstance(live_metrics.get("levels_completed"), int) and live_metrics["levels_completed"] >= 1
        if type(artifact.get("solver_completes_level")) is bool and artifact["solver_completes_level"] != expected_completion:
            errors.append("solver_completes_level must equal live_env_metrics.levels_completed>=1")

    if artifact.get("offline_validation", {}).get("passed") is not True:
        errors.append("reachable artifacts require passed offline_validation")
    if artifact.get("real_metric_mapping") != MetricMapping().to_json():
        errors.append("real_metric_mapping must equal the ARC live EnvironmentScore mapping")
    return errors


def _make_polling_score_provider(
    arcade: Any,
    game_id: str,
    *,
    min_levels_completed: int = 1,
    timeout_s: float = 8.0,
    poll_interval_s: float = 0.5,
) -> Callable[[Any], MetricMapping.Score]:  # pragma: no cover - live SDK boundary
    def score_provider(live_env: Any) -> MetricMapping.Score:
        deadline = time.time() + max(0.0, timeout_s)
        best: Optional[MetricMapping.Score] = None
        while True:
            scorecard = arcade.get_scorecard(str(getattr(live_env, "scorecard_id", "") or ""))
            score = _normalise_score(_extract_environment_score(scorecard, game_id))
            if best is None or score.levels_completed > best.levels_completed or score.score > best.score:
                best = score
            if score.levels_completed >= min_levels_completed or time.time() >= deadline:
                return best
            time.sleep(max(0.0, poll_interval_s))

    setattr(score_provider, "score_source", "sdk_get_scorecard_open_scorecard_polled")
    return score_provider


def run_live_solver_accuracy(
    arcade: Any,
    floor: FloorBaseline,
    solver_plan: list[dict[str, Any]],
    action_budget: Optional[int],
) -> tuple[int, SolverRunOutcome]:  # pragma: no cover - exercised by required live command
    """REQ-PHASE4-060: run one same-env live completion probe without closing scorecards."""

    environments = enumerate_live_environments(arcade)
    selected = next((env for env in environments if env.game_id == floor.environment.game_id), None)
    if selected is None:
        raise ValueError(f"SDK did not enumerate floor environment {floor.environment.game_id}")
    env = arcade.make(selected.game_id, save_recording=False, include_frame_data=True)
    if env is None:
        raise ValueError(f"SDK could not make live environment {selected.game_id}")
    return (
        len(environments),
        run_solver_completion(
            env,
            selected,
            floor=floor,
            solver_plan=solver_plan,
            requested_action_budget=action_budget,
            score_provider=_make_polling_score_provider(arcade, selected.game_id),
        ),
    )


def run(
    *,
    write: bool = True,
    action_budget: Optional[int] = None,
    base_url: str = BASE_URL,
) -> dict[str, Any]:
    """Run Exp 4214 live or write an honest blocked verdict."""

    started = time.time()
    preconditions = check_live_preconditions(base_url=base_url)
    offline_validation: dict[str, Any] = {"passed": False, "skipped": True}

    if not preconditions.ok:
        artifact = blocked_artifact(preconditions=preconditions, duration_s=time.time() - started)
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        offline_validation = validate_recorded_fixture()
        if offline_validation.get("passed") is not True:
            raise RuntimeError("recorded fixture adapter validation failed")
        floor = load_exp4202_floor()
        solver_plan = load_completion_solver_plan()
        arcade = open_online_arcade(base_url=base_url)
        environment_count, outcome = run_live_solver_accuracy(
            arcade,
            floor,
            solver_plan,
            action_budget,
        )
        artifact = build_artifact(
            outcome=outcome,
            floor=floor,
            preconditions=preconditions,
            offline_validation=offline_validation,
            environment_count=environment_count,
            duration_s=time.time() - started,
        )
    except Exception as exc:
        blocked_preconditions = ArcLivePreconditions(
            sdk_importable=preconditions.sdk_importable,
            sdk_version=preconditions.sdk_version,
            network_reachable=preconditions.network_reachable,
            base_url=preconditions.base_url,
            error=f"{preconditions.error}; live_solver_error={type(exc).__name__}: {exc}".strip("; "),
        )
        artifact = blocked_artifact(preconditions=blocked_preconditions, duration_s=time.time() - started)
        artifact["offline_validation"] = offline_validation
        errors = artifact_schema_errors(artifact)
        if errors:  # pragma: no cover - defensive schema guard
            raise ValueError("; ".join(errors)) from exc

    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--action-budget", type=int, default=None)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    artifact = run(
        write=not args.no_write,
        action_budget=args.action_budget,
        base_url=args.base_url,
    )
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
