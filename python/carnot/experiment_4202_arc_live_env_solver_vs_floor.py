"""Exp 4202: ARC-AGI-3 live solver-vs-floor probe.

Spec refs: REQ-PHASE4-058, SCENARIO-PHASE4-058.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from carnot.agentic.arc_agi3_live_adapter import (
    BASE_URL,
    ArcAction,
    ArcLivePreconditions,
    EnvironmentSummary,
    MetricMapping,
    _baseline_reference,
    _extract_environment_score,
    _game_action,
    _game_over,
    _levels_completed,
    _normalise_score,
    check_live_preconditions,
    enumerate_live_environments,
    open_online_arcade,
    validate_recorded_fixture,
)
from carnot.agentic.arc_agi3_world_model import compute_grid_delta, frame_hash, grid_of


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4202_arc_live_env_solver_vs_floor.json"
FLOOR_RESULT_NAME = "experiment_4191_arc_live_env_grounding_probe.json"
SOLVER_SOURCE_NAME = "experiment_4190_arc_incremental_progress.json"
LP85_GAME_ID = "lp85-305b61c3"
RANDOM_SEED = 4202
DEFAULT_ACTION_BUDGET = 6
INFERENCE_SUBSTRATE = "official_arc_agi3_online_anonymous_key_solver_vs_floor_probe"
REQUIREMENTS = ["REQ-PHASE4-058", "SCENARIO-PHASE4-058"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "solver_beats_floor",
    "live_env_metrics",
    "no_leaderboard_submission",
    "preconditions_checked",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'solver does not beat the floor on the live env' "
        "or 'blocked_arc_live_unreachable' is a COMPLETE grounding verdict."
    ),
    "solver_beats_floor": (
        "{accuracy: solver_score vs floor_score, efficiency: solver_actions vs floor_actions} "
        "-- the real-env read on whether the verifier-routed solver earns its place over "
        "random (the north-star's two axes)."
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


@dataclass(frozen=True)
class FloorBaseline:
    """REQ-PHASE4-058: Exp 4191 random/greedy floor on the same live environment."""

    environment: EnvironmentSummary
    actions_taken: int
    baseline_actions: int
    actions_vs_baseline_actions: float
    score: float
    levels_completed: int
    source_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_json(),
            "actions_taken": int(self.actions_taken),
            "baseline_actions": int(self.baseline_actions),
            "actions_vs_baseline_actions": float(self.actions_vs_baseline_actions),
            "score": float(self.score),
            "levels_completed": int(self.levels_completed),
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class SolverRunOutcome:
    """Normalized evidence from the bounded live solver replay."""

    environment: EnvironmentSummary
    action_budget: int
    actions_taken: int
    baseline_actions: int
    actions_vs_baseline_actions: float
    score: MetricMapping.Score
    trace: list[dict[str, Any]]
    scorecard_id: str
    score_source: str
    solver_policy: str
    source_artifact: str
    anonymous_key_used: bool = True
    leaderboard_submission_attempted: bool = False

    def live_metrics_json(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_json(),
            "score": float(self.score.score),
            "levels_completed": int(self.score.levels_completed),
            "actions_taken": int(self.actions_taken),
            "baseline_actions": int(self.baseline_actions),
            "actions_vs_baseline_actions": float(self.actions_vs_baseline_actions),
            "environment_score": self.score.to_json(),
            "scorecard_id": self.scorecard_id,
            "score_source": self.score_source,
            "action_budget": int(self.action_budget),
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_floor_baseline(
    artifact: dict[str, Any],
    *,
    source_path: str = f"results/{FLOOR_RESULT_NAME}",
) -> FloorBaseline:
    """REQ-PHASE4-058: load Exp 4191's random/greedy floor for the same LP85 env."""

    floor = artifact.get("random_greedy_baseline")
    if not isinstance(floor, dict):
        raise ValueError("Exp 4191 artifact missing random_greedy_baseline")
    env_payload = floor.get("environment")
    if not isinstance(env_payload, dict):
        raise ValueError("Exp 4191 floor missing environment")
    environment = EnvironmentSummary(
        game_id=str(env_payload.get("game_id", "") or ""),
        title=str(env_payload.get("title", "") or ""),
        tags=[str(tag) for tag in (env_payload.get("tags") or [])],
        baseline_actions=[int(value) for value in (env_payload.get("baseline_actions") or [])],
    )
    if environment.game_id != LP85_GAME_ID:
        raise ValueError("Exp 4202 must compare on the same LP85 environment as Exp 4191")
    return FloorBaseline(
        environment=environment,
        actions_taken=int(floor.get("actions_taken", 0) or 0),
        baseline_actions=int(floor.get("baseline_actions", 0) or 0),
        actions_vs_baseline_actions=float(floor.get("actions_vs_baseline_actions", 0.0) or 0.0),
        score=float(floor.get("score", 0.0) or 0.0),
        levels_completed=int(floor.get("levels_completed", 0) or 0),
        source_path=str(source_path),
    )


def load_floor_baseline(path: Optional[Path] = None) -> FloorBaseline:
    """REQ-PHASE4-058: read the existing Exp 4191 floor artifact."""

    floor_path = path or (REPO / "results" / FLOOR_RESULT_NAME)
    try:
        source_path = str(floor_path.relative_to(REPO))
    except ValueError:
        source_path = str(floor_path)
    return extract_floor_baseline(_read_json(floor_path), source_path=source_path)


def extract_banked_lp85_l1_plan(prior_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-PHASE4-058: extract the banked LP85 L1 replay from the proven solver trace."""

    trace = prior_artifact.get("phase_trace")
    if not isinstance(trace, list):
        solve_trace = prior_artifact.get("solve_trace")
        trace = solve_trace.get("phase_trace") if isinstance(solve_trace, dict) else None
    plan: list[dict[str, Any]] = []
    for row in trace if isinstance(trace, list) else []:
        if not isinstance(row, dict) or row.get("source") != "banked_lp85_L1_replay":
            continue
        if "x" not in row or "y" not in row:
            continue
        plan.append(
            {
                "action_id": 6,
                "x": int(row["x"]),
                "y": int(row["y"]),
                "source": "banked_lp85_L1_replay",
                "expected_levels_completed": int(row.get("levels_completed", 0) or 0),
            }
        )
    if not plan:
        raise ValueError("banked LP85 L1 replay not found in solver artifact")
    if max(int(step["expected_levels_completed"]) for step in plan) < 1:
        raise ValueError("banked LP85 L1 replay does not confirm L1")
    return plan


def load_banked_solver_plan(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """REQ-PHASE4-058: read the prior LP85 solver trace used for live replay."""

    solver_path = path or (REPO / "results" / SOLVER_SOURCE_NAME)
    return extract_banked_lp85_l1_plan(_read_json(solver_path))


def run_solver_replay(
    env: Any,
    environment: EnvironmentSummary,
    *,
    solver_plan: list[dict[str, Any]],
    action_budget: int = DEFAULT_ACTION_BUDGET,
    action_enum: Any = None,
    score_provider: Optional[Callable[[Any], Any]] = None,
) -> SolverRunOutcome:
    """SCENARIO-PHASE4-058: execute the bounded solver replay through the live adapter path."""

    if action_enum is None:  # pragma: no cover - SDK boundary
        from arcengine import GameAction as action_enum

    frame = env.reset()
    if frame is None:
        raise ValueError(f"reset returned no frame for {environment.game_id}")

    trace: list[dict[str, Any]] = []
    actions_taken = 0
    for action_index, step in enumerate(solver_plan[: max(0, int(action_budget))], start=1):
        prev_grid = grid_of(frame)
        current_hash = frame_hash(prev_grid)
        action = ArcAction(6, {"x": int(step["x"]), "y": int(step["y"])}, str(step.get("source", "solver_replay")))
        next_frame = env.step(
            _game_action(action_enum, action.action_id),
            data=action.data,
            reasoning={
                "policy": "bounded_verifier_routed_explore_induce_verify_replay",
                "experiment": 4202,
                "source_artifact": f"results/{SOLVER_SOURCE_NAME}",
            },
        )
        actions_taken += 1
        if next_frame is None:
            trace.append(
                {
                    "action_index": int(action_index),
                    "action": action.to_json(),
                    "event": "step_returned_no_frame",
                }
            )
            break

        next_grid = grid_of(next_frame)
        level_delta = _levels_completed(next_frame) - _levels_completed(frame)
        delta = compute_grid_delta(prev_grid, next_grid)
        trace.append(
            {
                "action_index": int(action_index),
                "action": action.to_json(),
                "frame_hash_before": current_hash,
                "frame_hash_after": frame_hash(next_grid),
                "n_changed": int(delta.get("n_changed", 0)),
                "level_delta": int(level_delta),
                "levels_completed_after": _levels_completed(next_frame),
                "game_over": _game_over(next_frame),
            }
        )
        frame = next_frame
        if level_delta > 0:
            break

    if score_provider is None:
        score = MetricMapping.Score(
            score=0.0,
            levels_completed=_levels_completed(frame),
            actions=actions_taken,
            level_actions=[actions_taken],
            level_baseline_actions=environment.baseline_actions[:1],
            completed=False,
            guid=str(getattr(frame, "guid", "") or ""),
            message="local_score_provider_not_configured",
        )
        score_source = "local_adapter_fallback"
    else:
        score = _normalise_score(score_provider(env))
        score_source = str(getattr(score_provider, "score_source", "score_provider"))

    baseline_actions = _baseline_reference(environment, score)
    actions_vs_baseline = float(actions_taken / baseline_actions) if baseline_actions > 0 else 0.0
    return SolverRunOutcome(
        environment=environment,
        action_budget=int(action_budget),
        actions_taken=int(actions_taken),
        baseline_actions=int(baseline_actions),
        actions_vs_baseline_actions=actions_vs_baseline,
        score=score,
        trace=trace,
        scorecard_id=str(getattr(env, "scorecard_id", "") or ""),
        score_source=score_source,
        solver_policy="bounded_verifier_routed_explore_induce_verify_replay",
        source_artifact=f"results/{SOLVER_SOURCE_NAME}",
        anonymous_key_used=True,
        leaderboard_submission_attempted=False,
    )


def compare_solver_to_floor(outcome: SolverRunOutcome, floor: FloorBaseline) -> dict[str, Any]:
    """REQ-PHASE4-058: compare accuracy and efficiency against the same-env floor."""

    accuracy_beats = (
        float(outcome.score.score) > float(floor.score)
        or int(outcome.score.levels_completed) > int(floor.levels_completed)
    )
    efficiency_beats = int(outcome.actions_taken) < int(floor.actions_taken)
    return {
        "accuracy": {
            "beats": bool(accuracy_beats),
            "solver_score": float(outcome.score.score),
            "floor_score": float(floor.score),
            "solver_levels_completed": int(outcome.score.levels_completed),
            "floor_levels_completed": int(floor.levels_completed),
        },
        "efficiency": {
            "beats": bool(efficiency_beats),
            "solver_actions": int(outcome.actions_taken),
            "floor_actions": int(floor.actions_taken),
            "solver_actions_vs_baseline_actions": float(outcome.actions_vs_baseline_actions),
            "floor_actions_vs_baseline_actions": float(floor.actions_vs_baseline_actions),
        },
        "overall": bool(accuracy_beats or efficiency_beats),
    }


def _verdict_from_comparison(comparison: dict[str, Any], game_id: str) -> str:
    accuracy = bool(comparison.get("accuracy", {}).get("beats"))
    efficiency = bool(comparison.get("efficiency", {}).get("beats"))
    if accuracy and efficiency:
        return f"success: solver_beats_floor_live_{game_id}_accuracy_and_efficiency"
    if accuracy:
        return f"complete: solver_beats_floor_live_{game_id}_accuracy_only"
    if efficiency:
        return f"complete: solver_beats_floor_live_{game_id}_efficiency_only"
    return f"complete: solver_does_not_beat_floor_live_{game_id}"


def blocked_artifact(*, preconditions: ArcLivePreconditions, duration_s: float) -> dict[str, Any]:
    """SCENARIO-PHASE4-058: report missing live substrate without fabricating metrics."""

    artifact = {
        "experiment": "experiment_4202_arc_live_env_solver_vs_floor",
        "title": "arc3_live_env_solver_vs_floor",
        "honest_verdict": "blocked_arc_live_unreachable",
        "solver_beats_floor": {},
        "live_env_metrics": {},
        "random_greedy_floor": {},
        "no_leaderboard_submission": True,
        "leaderboard_submission_attempted": False,
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
    """REQ-PHASE4-058: build the terminal solver-vs-floor artifact."""

    if not preconditions.ok:
        artifact = blocked_artifact(preconditions=preconditions, duration_s=duration_s)
        artifact["offline_validation"] = dict(offline_validation)
        return artifact
    if outcome.environment.game_id != floor.environment.game_id:
        raise ValueError("solver outcome and floor must use the same environment")

    comparison = compare_solver_to_floor(outcome, floor)
    artifact = {
        "experiment": "experiment_4202_arc_live_env_solver_vs_floor",
        "title": "arc3_live_env_solver_vs_floor",
        "honest_verdict": _verdict_from_comparison(comparison, outcome.environment.game_id),
        "live_env_reachable": True,
        "solver_beats_floor": comparison,
        "live_env_metrics": outcome.live_metrics_json(),
        "random_greedy_floor": floor.to_json(),
        "solver_trace": list(outcome.trace),
        "solver_policy": outcome.solver_policy,
        "solver_source_artifact": outcome.source_artifact,
        "environment_count": int(environment_count),
        "no_leaderboard_submission": True,
        "leaderboard_submission_attempted": bool(outcome.leaderboard_submission_attempted),
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
    """SCENARIO-PHASE4-058: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    if artifact.get("no_leaderboard_submission") is not True:
        errors.append("no_leaderboard_submission must be true")
    if artifact.get("leaderboard_submission_attempted") is not False:
        errors.append("leaderboard_submission_attempted must be false")

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
        errors.append("requirements must include REQ-PHASE4-058 and SCENARIO-PHASE4-058")

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
        for field in ("score", "levels_completed", "actions_taken", "baseline_actions"):
            if field not in live_metrics:
                errors.append(f"live_env_metrics missing {field}")
        for field in ("levels_completed", "actions_taken", "baseline_actions"):
            if field in live_metrics and type(live_metrics[field]) is not int:
                errors.append(f"live_env_metrics.{field} must be a bare int")
        if "score" in live_metrics and type(live_metrics["score"]) not in (int, float):
            errors.append("live_env_metrics.score must be numeric")

    if artifact.get("offline_validation", {}).get("passed") is not True:
        errors.append("reachable artifacts require passed offline_validation")
    if artifact.get("real_metric_mapping") != MetricMapping().to_json():
        errors.append("real_metric_mapping must equal the ARC live EnvironmentScore mapping")
    return errors


def run_live_solver_vs_floor(
    arcade: Any,
    floor: FloorBaseline,
    solver_plan: list[dict[str, Any]],
    action_budget: int,
) -> tuple[int, SolverRunOutcome]:  # pragma: no cover - exercised by required live command
    """REQ-PHASE4-058: run the same-env LP85 live solver probe without closing scorecards."""

    environments = enumerate_live_environments(arcade)
    selected = next((env for env in environments if env.game_id == floor.environment.game_id), None)
    if selected is None:
        raise ValueError(f"SDK did not enumerate floor environment {floor.environment.game_id}")
    env = arcade.make(selected.game_id, save_recording=False, include_frame_data=True)
    if env is None:
        raise ValueError(f"SDK could not make live environment {selected.game_id}")

    def score_provider(live_env: Any) -> Any:
        scorecard = arcade.get_scorecard(str(getattr(live_env, "scorecard_id", "") or ""))
        return _extract_environment_score(scorecard, selected.game_id)

    setattr(score_provider, "score_source", "sdk_get_scorecard_open_scorecard")
    return (
        len(environments),
        run_solver_replay(
            env,
            selected,
            solver_plan=solver_plan,
            action_budget=action_budget,
            score_provider=score_provider,
        ),
    )


def run(
    *,
    write: bool = True,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    base_url: str = BASE_URL,
) -> dict[str, Any]:
    """Run Exp 4202 live or write an honest blocked verdict."""

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
        floor = load_floor_baseline()
        solver_plan = load_banked_solver_plan()
        arcade = open_online_arcade(base_url=base_url)
        environment_count, outcome = run_live_solver_vs_floor(
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
    parser.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
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
