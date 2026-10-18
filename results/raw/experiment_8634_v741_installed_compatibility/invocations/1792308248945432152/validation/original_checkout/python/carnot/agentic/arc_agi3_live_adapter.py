"""Live ARC-AGI-3 adapter for the bounded Exp 4191 grounding probe.

Spec refs: REQ-PHASE4-056, SCENARIO-PHASE4-056.

This module keeps the live SDK contact deliberately small: enumerate games,
reset one selected environment, spend a bounded random/greedy action budget,
and read the SDK's open-scorecard EnvironmentScore. It never closes a
scorecard or enters competition mode.
"""

from __future__ import annotations

import importlib.metadata
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import numpy as np

from carnot.agentic.arc_agi3_world_model import (
    GameGraph,
    action_key,
    compute_grid_delta,
    frame_hash,
    grid_of,
    objects,
)


REPO = Path(__file__).resolve().parents[3]
RESULT_NAME = "experiment_4191_arc_live_env_grounding_probe.json"
BASE_URL = "https://three.arcprize.org"
RANDOM_SEED = 4191
DEFAULT_ACTION_BUDGET = 6
INFERENCE_SUBSTRATE = "official_arc_agi3_online_anonymous_key_reachability_probe"
REQUIREMENTS = ["REQ-PHASE4-056", "SCENARIO-PHASE4-056"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "live_env_reachable",
    "real_metric_mapping",
    "random_greedy_baseline",
    "no_leaderboard_submission",
    "preconditions_checked",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest blocked_arc_live_unreachable is a COMPLETE grounding verdict; "
        "so is a successful reachability + baseline probe."
    ),
    "live_env_reachable": (
        "Bare bool: the live SDK connected with an anonymous key and returned a real EnvironmentScore; "
        "the §0 grounding the whole north-star harness needs."
    ),
    "real_metric_mapping": (
        "{accuracy: EnvironmentScore.score/levels_completed, efficiency: actions_vs_baseline_actions} "
        "— the REAL-env metric pipeline the offline proof feeds into."
    ),
    "random_greedy_baseline": (
        "The reference numbers (score + actions) for the probed env; the floor any Carnot run "
        "must beat before operator-gated online submission."
    ),
    "no_leaderboard_submission": (
        "Bare bool: zero scorecards were submitted (Operator-Only External Publication; the online quota gate)."
    ),
    "preconditions_checked": (
        "Records the SDK + network reachability checks; pre-empts the silent-missing-resource fabrication mode."
    ),
}


@dataclass(frozen=True)
class ArcLivePreconditions:
    """REQ-PHASE4-056: preflight record for the SDK import and network checks."""

    sdk_importable: bool
    sdk_version: str
    network_reachable: bool
    base_url: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.sdk_importable and self.network_reachable)

    def to_json(self) -> dict[str, Any]:
        return {
            "sdk_importable": bool(self.sdk_importable),
            "sdk_version": str(self.sdk_version),
            "network_reachable": bool(self.network_reachable),
            "base_url": str(self.base_url),
            "error": str(self.error),
        }


@dataclass(frozen=True)
class EnvironmentSummary:
    """Small serializable view of SDK EnvironmentInfo."""

    game_id: str
    title: str
    tags: list[str] = field(default_factory=list)
    baseline_actions: list[int] = field(default_factory=list)

    @classmethod
    def from_info(cls, info: Any) -> "EnvironmentSummary":
        return cls(
            game_id=str(getattr(info, "game_id", "") or ""),
            title=str(getattr(info, "title", "") or ""),
            tags=[str(tag) for tag in (getattr(info, "tags", None) or [])],
            baseline_actions=[int(value) for value in (getattr(info, "baseline_actions", None) or [])],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "title": self.title,
            "tags": list(self.tags),
            "baseline_actions": [int(value) for value in self.baseline_actions],
        }


@dataclass(frozen=True)
class ArcAction:
    """One normalized ARC action, keyed the same way as the offline GameGraph."""

    action_id: int
    data: Optional[dict[str, int]]
    source: str

    @property
    def key(self) -> tuple:
        return action_key(self.action_id, self.data)

    def to_json(self) -> dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "data": dict(self.data) if self.data else None,
            "source": self.source,
            "action_key": list(self.key),
        }


class MetricMapping:
    """REQ-PHASE4-056: names the real north-star metric fields."""

    accuracy = "EnvironmentScore.score/levels_completed"
    efficiency = "actions_vs_baseline_actions"

    @dataclass(frozen=True)
    class Score:
        score: float
        levels_completed: int
        actions: int
        level_actions: list[int] = field(default_factory=list)
        level_baseline_actions: list[int] = field(default_factory=list)
        completed: bool = False
        resets: int | None = None
        guid: str = ""
        message: str = ""

        def to_json(self) -> dict[str, Any]:
            return {
                "score": float(self.score),
                "levels_completed": int(self.levels_completed),
                "actions": int(self.actions),
                "level_actions": [int(value) for value in self.level_actions],
                "level_baseline_actions": [int(value) for value in self.level_baseline_actions],
                "completed": bool(self.completed),
                "resets": None if self.resets is None else int(self.resets),
                "guid": self.guid,
                "message": self.message,
            }

    def to_json(self) -> dict[str, str]:
        return {
            "accuracy": self.accuracy,
            "efficiency": self.efficiency,
        }


@dataclass(frozen=True)
class LiveProbeOutcome:
    """Normalized output from the bounded live random/greedy probe."""

    environment: EnvironmentSummary
    action_budget: int
    actions_taken: int
    baseline_actions: int
    actions_vs_baseline_actions: float
    score: MetricMapping.Score
    trace: list[dict[str, Any]]
    scorecard_id: str
    score_source: str
    anonymous_key_used: bool
    leaderboard_submission_attempted: bool = False

    def baseline_json(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_json(),
            "action_budget": int(self.action_budget),
            "actions_taken": int(self.actions_taken),
            "baseline_actions": int(self.baseline_actions),
            "actions_vs_baseline_actions": float(self.actions_vs_baseline_actions),
            "score": float(self.score.score),
            "levels_completed": int(self.score.levels_completed),
            "environment_score": self.score.to_json(),
            "trace": list(self.trace),
            "scorecard_id": self.scorecard_id,
            "score_source": self.score_source,
            "anonymous_key_used": bool(self.anonymous_key_used),
            "leaderboard_submission_attempted": bool(self.leaderboard_submission_attempted),
        }


def check_live_preconditions(base_url: str = BASE_URL, timeout_s: float = 10.0) -> ArcLivePreconditions:
    """REQ-PHASE4-056: verify SDK import and ARC endpoint reachability."""

    sdk_importable = False
    sdk_version = "missing"
    errors: list[str] = []
    try:
        __import__("arc_agi")
        sdk_importable = True
        try:
            sdk_version = importlib.metadata.version("arc-agi")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = "version_unknown"
    except Exception as exc:  # pragma: no cover - exercised by monkeypatch in callers if needed
        errors.append(f"sdk_import_error={type(exc).__name__}: {exc}")

    network_reachable = False
    try:
        response = requests.get(base_url, timeout=timeout_s, allow_redirects=False)
        network_reachable = response.status_code < 500
    except requests.RequestException as exc:
        errors.append(f"network_error={type(exc).__name__}: {exc}")

    return ArcLivePreconditions(
        sdk_importable=sdk_importable,
        sdk_version=sdk_version,
        network_reachable=network_reachable,
        base_url=base_url,
        error="; ".join(errors),
    )


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("carnot.arc_agi3_live_probe")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


def open_online_arcade(base_url: str = BASE_URL) -> Any:  # pragma: no cover - thin SDK boundary
    """REQ-PHASE4-056: instantiate the official SDK in online anonymous-key mode."""

    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        arc_base_url=base_url,
        operation_mode=OperationMode.ONLINE,
        environments_dir="",
        recordings_dir=str(REPO / "recordings" / "arc_live_probe"),
        logger=_quiet_logger(),
    )


def enumerate_live_environments(arcade: Any) -> list[EnvironmentSummary]:
    """REQ-PHASE4-056: enumerate environments returned by the live SDK."""

    return [
        summary
        for summary in (EnvironmentSummary.from_info(info) for info in arcade.get_environments())
        if summary.game_id
    ]


def choose_probe_environment(environments: list[EnvironmentSummary]) -> EnvironmentSummary:
    """SCENARIO-PHASE4-056: choose a low-budget easy environment for bounded probing."""

    if not environments:
        raise ValueError("no live ARC-AGI-3 environments returned by SDK")

    by_prefix = {env.game_id.split("-", maxsplit=1)[0]: env for env in environments}
    for preferred in ("lp85", "sb26", "s5i5"):
        if preferred in by_prefix:
            return by_prefix[preferred]

    def key(env: EnvironmentSummary) -> tuple[int, str]:
        positives = [value for value in env.baseline_actions if value > 0]
        return (min(positives) if positives else 10**9, env.game_id)

    return sorted(environments, key=key)[0]


def _available_action_ids(frame: Any) -> list[int]:
    out: list[int] = []
    for raw in getattr(frame, "available_actions", []) or []:
        if hasattr(raw, "value"):
            value = int(raw.value)
        elif isinstance(raw, str) and raw.upper().startswith("ACTION"):
            value = int(raw.upper().replace("ACTION", "", 1))
        else:
            value = int(raw)
        if value != 0 and value not in out:
            out.append(value)
    return out


def _action_candidates(frame: Any) -> list[ArcAction]:
    action_ids = _available_action_ids(frame)
    candidates: list[ArcAction] = []

    for action_id in action_ids:
        if action_id != 6:
            candidates.append(ArcAction(action_id, None, "available_keyboard_action"))

    if 6 in action_ids:
        grid = grid_of(frame)
        click_points = [(int(x), int(y)) for y, x in objects(grid)]
        if not click_points:
            h, w = grid.shape
            click_points = [(w // 2, h // 2)]
        seen: set[tuple[int, int]] = set()
        for x, y in click_points[:12]:
            point = (max(0, int(x)), max(0, int(y)))
            if point in seen:
                continue
            seen.add(point)
            candidates.append(ArcAction(6, {"x": point[0], "y": point[1]}, "object_centroid_click"))

    return candidates


def _game_action(action_enum: Any, action_id: int) -> Any:
    if hasattr(action_enum, "from_id"):
        return action_enum.from_id(int(action_id))
    attr = f"ACTION{int(action_id)}"
    if hasattr(action_enum, attr):
        return getattr(action_enum, attr)
    return int(action_id)


def _levels_completed(frame: Any) -> int:
    return int(getattr(frame, "levels_completed", 0) or 0)


def _game_over(frame: Any) -> bool:
    state = str(getattr(frame, "state", "") or "").upper()
    return "GAME_OVER" in state or "LOSE" in state


def _normalise_score(score: Any) -> MetricMapping.Score:
    if isinstance(score, MetricMapping.Score):
        return score
    if isinstance(score, dict):
        getter = score.get
    else:
        getter = lambda name, default=None: getattr(score, name, default)
    return MetricMapping.Score(
        score=float(getter("score", 0.0) or 0.0),
        levels_completed=int(getter("levels_completed", 0) or 0),
        actions=int(getter("actions", 0) or 0),
        level_actions=[int(value) for value in (getter("level_actions", []) or [])],
        level_baseline_actions=[
            int(value) for value in (getter("level_baseline_actions", []) or [])
        ],
        completed=bool(getter("completed", False) or False),
        resets=None if getter("resets", None) is None else int(getter("resets", 0) or 0),
        guid=str(getter("guid", "") or ""),
        message=str(getter("message", "") or ""),
    )


def _extract_environment_score(scorecard: Any, game_id: str) -> Any:
    if scorecard is None:
        raise ValueError("SDK returned no scorecard")

    env_score = None
    if hasattr(scorecard, "find_environment"):
        env_score = scorecard.find_environment(game_id)
        if env_score is None and "-" in game_id:
            env_score = scorecard.find_environment(game_id.split("-", maxsplit=1)[0])

    if env_score is None:
        for candidate in getattr(scorecard, "environments", []) or []:
            candidate_id = str(getattr(candidate, "id", "") or "")
            if candidate_id == game_id or game_id.startswith(candidate_id) or candidate_id.startswith(game_id):
                env_score = candidate
                break

    if env_score is None:
        raise ValueError(f"scorecard did not include environment {game_id}")

    runs = getattr(env_score, "runs", None) or []
    if runs:
        return runs[0]
    return env_score


def _baseline_reference(environment: EnvironmentSummary, score: MetricMapping.Score) -> int:
    completed = max(0, int(score.levels_completed) - 1)
    if score.level_baseline_actions:
        idx = min(completed, len(score.level_baseline_actions) - 1)
        value = int(score.level_baseline_actions[idx])
        if value > 0:
            return value
    if environment.baseline_actions:
        idx = min(completed, len(environment.baseline_actions) - 1)
        return int(environment.baseline_actions[idx])
    return 0


def run_random_greedy_baseline(
    env: Any,
    environment: EnvironmentSummary,
    *,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    random_seed: int = RANDOM_SEED,
    action_enum: Any = None,
    score_provider: Optional[Callable[[Any], Any]] = None,
) -> LiveProbeOutcome:
    """SCENARIO-PHASE4-056: execute the bounded random/greedy baseline through the adapter."""

    if action_enum is None:  # pragma: no cover - SDK boundary
        from arcengine import GameAction as action_enum

    rng = random.Random(random_seed)
    graph = GameGraph(environment.game_id)
    frame = env.reset()
    if frame is None:
        raise ValueError(f"reset returned no frame for {environment.game_id}")

    trace: list[dict[str, Any]] = []
    actions_taken = 0
    used_action_keys: set[tuple] = set()
    graph.see_node(frame_hash(grid_of(frame)), frame)

    for action_index in range(1, max(0, int(action_budget)) + 1):
        prev_grid = grid_of(frame)
        current_hash = frame_hash(prev_grid)
        candidates = _action_candidates(frame)
        if not candidates:
            trace.append({"action_index": action_index, "event": "no_available_actions"})
            break

        by_key = {candidate.key: candidate for candidate in candidates}
        globally_new = [candidate for candidate in candidates if candidate.key not in used_action_keys]
        if globally_new:
            selected = globally_new[0]
        elif untested := graph.untested(current_hash, list(by_key)):
            selected = by_key[untested[0]]
        else:
            selected = candidates[rng.randrange(len(candidates))]
        used_action_keys.add(selected.key)

        next_frame = env.step(
            _game_action(action_enum, selected.action_id),
            data=selected.data,
            reasoning={"policy": "bounded_random_greedy_baseline", "experiment": 4191},
        )
        actions_taken += 1
        if next_frame is None:
            trace.append(
                {
                    "action_index": action_index,
                    "action": selected.to_json(),
                    "event": "step_returned_no_frame",
                }
            )
            break

        next_grid = grid_of(next_frame)
        next_hash = frame_hash(next_grid)
        delta = compute_grid_delta(prev_grid, next_grid)
        level_delta = _levels_completed(next_frame) - _levels_completed(frame)
        graph.record(current_hash, selected.key, next_hash, delta, level_delta, _game_over(next_frame))
        graph.see_node(next_hash, next_frame)
        trace.append(
            {
                "action_index": action_index,
                "action": selected.to_json(),
                "frame_hash_before": current_hash,
                "frame_hash_after": next_hash,
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
    return LiveProbeOutcome(
        environment=environment,
        action_budget=int(action_budget),
        actions_taken=int(actions_taken),
        baseline_actions=int(baseline_actions),
        actions_vs_baseline_actions=actions_vs_baseline,
        score=score,
        trace=trace,
        scorecard_id=str(getattr(env, "scorecard_id", "") or ""),
        score_source=score_source,
        anonymous_key_used=True,
        leaderboard_submission_attempted=False,
    )


def run_live_reachability_probe(
    arcade: Any,
    *,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    random_seed: int = RANDOM_SEED,
) -> tuple[int, LiveProbeOutcome]:  # pragma: no cover - exercised by required experiment command
    """REQ-PHASE4-056: enumerate live envs and run one bounded baseline without closing scorecards."""

    environments = enumerate_live_environments(arcade)
    selected = choose_probe_environment(environments)
    env = arcade.make(selected.game_id, save_recording=False, include_frame_data=True)
    if env is None:
        raise ValueError(f"SDK could not make live environment {selected.game_id}")

    def score_provider(live_env: Any) -> Any:
        scorecard = arcade.get_scorecard(str(getattr(live_env, "scorecard_id", "") or ""))
        return _extract_environment_score(scorecard, selected.game_id)

    setattr(score_provider, "score_source", "sdk_get_scorecard_open_scorecard")
    return (
        len(environments),
        run_random_greedy_baseline(
            env,
            selected,
            action_budget=action_budget,
            random_seed=random_seed,
            score_provider=score_provider,
        ),
    )


class _RecordedActionEnum:
    RESET = type("ResetAction", (), {"value": 0, "name": "RESET"})()

    @staticmethod
    def from_id(action_id: int) -> Any:
        return type("RecordedAction", (), {"value": int(action_id), "name": f"ACTION{int(action_id)}"})()


class _RecordedFixtureEnv:
    def __init__(self) -> None:
        self.scorecard_id = "recorded-fixture-open-scorecard"
        self._level = 0
        self._index = 0
        self.actions: list[int] = []
        self.frames = [
            _frame(np.zeros((3, 3), dtype=np.int16), 0, [1, 2], "recorded-guid"),
            _frame(np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]], dtype=np.int16), 0, [1, 2], "recorded-guid"),
            _frame(np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=np.int16), 1, [1, 2], "recorded-guid"),
        ]

    def reset(self) -> Any:
        self._level = 0
        self._index = 0
        self.actions.clear()
        return self.frames[self._index]

    def step(self, action: Any, data: Optional[dict[str, int]] = None, reasoning: Optional[dict[str, Any]] = None) -> Any:
        del data, reasoning
        action_id = int(getattr(action, "value", action))
        self.actions.append(action_id)
        self._index = min(self._index + 1, 2)
        if action_id == 2:
            self._level = 1
        return _frame(self.frames[self._index].frame, self._level, [1, 2], "recorded-guid")


def _frame(grid: Any, levels_completed: int, available_actions: list[int], guid: str) -> Any:
    return type(
        "RecordedFrame",
        (),
        {
            "frame": grid,
            "levels_completed": int(levels_completed),
            "available_actions": list(available_actions),
            "state": "PLAYING",
            "guid": guid,
        },
    )()


def validate_recorded_fixture() -> dict[str, Any]:
    """REQ-PHASE4-056: prove the adapter path against a recorded offline fixture first."""

    env = _RecordedFixtureEnv()

    def score_provider(recorded_env: _RecordedFixtureEnv) -> MetricMapping.Score:
        return MetricMapping.Score(
            score=25.0,
            levels_completed=recorded_env._level,
            actions=len(recorded_env.actions),
            level_actions=[len(recorded_env.actions)],
            level_baseline_actions=[4],
            completed=False,
            resets=1,
            guid="recorded-guid",
        )

    setattr(score_provider, "score_source", "recorded_fixture_score")
    outcome = run_random_greedy_baseline(
        env,
        EnvironmentSummary("recorded-fixture", "Recorded Fixture", ["keyboard"], [4]),
        action_budget=2,
        random_seed=RANDOM_SEED,
        action_enum=_RecordedActionEnum,
        score_provider=score_provider,
    )
    observed = [int(step["action"]["action_id"]) for step in outcome.trace if "action" in step]
    expected = [1, 2]
    passed = (
        observed == expected
        and outcome.score.levels_completed == 1
        and outcome.score.actions == 2
        and outcome.trace[-1]["level_delta"] == 1
    )
    return {
        "passed": bool(passed),
        "fixture": "recorded_two_action_keyboard_level_up",
        "expected_action_ids": expected,
        "observed_action_ids": observed,
        "score": outcome.score.to_json(),
        "actions_vs_baseline_actions": outcome.actions_vs_baseline_actions,
        "trace": outcome.trace,
    }


def blocked_artifact(*, preconditions: ArcLivePreconditions, duration_s: float) -> dict[str, Any]:
    """REQ-PHASE4-056: terminal blocked artifact for unavailable live ARC."""

    artifact = {
        "experiment": "experiment_4191_arc_live_env_grounding_probe",
        "title": "arc3_live_env_grounding_probe",
        "honest_verdict": "blocked_arc_live_unreachable",
        "live_env_reachable": False,
        "real_metric_mapping": MetricMapping().to_json(),
        "random_greedy_baseline": {},
        "no_leaderboard_submission": True,
        "preconditions_checked": preconditions.to_json(),
        "offline_validation": {"passed": False, "skipped": True},
        "environment_count": 0,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "acceptance_gate_passed": True,
        "leaderboard_submission_attempted": False,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_artifact(
    *,
    outcome: LiveProbeOutcome,
    preconditions: ArcLivePreconditions,
    offline_validation: dict[str, Any],
    environment_count: int,
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-056: terminal artifact for a reachable live baseline probe."""

    live_env_reachable = bool(
        preconditions.ok
        and offline_validation.get("passed") is True
        and not outcome.leaderboard_submission_attempted
    )
    no_submission = not outcome.leaderboard_submission_attempted
    if live_env_reachable:
        verdict = f"complete: arc_live_env_reachable_random_greedy_baseline_{outcome.environment.game_id}"
    else:
        verdict = "blocked_arc_live_unreachable"
    artifact = {
        "experiment": "experiment_4191_arc_live_env_grounding_probe",
        "title": "arc3_live_env_grounding_probe",
        "honest_verdict": verdict,
        "live_env_reachable": live_env_reachable,
        "real_metric_mapping": MetricMapping().to_json(),
        "random_greedy_baseline": outcome.baseline_json() if live_env_reachable else {},
        "no_leaderboard_submission": bool(no_submission),
        "preconditions_checked": preconditions.to_json(),
        "offline_validation": dict(offline_validation),
        "environment_count": int(environment_count),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "acceptance_gate_passed": bool((live_env_reachable or verdict.startswith("blocked_")) and no_submission),
        "leaderboard_submission_attempted": bool(outcome.leaderboard_submission_attempted),
        "online_mode": "official_sdk_online_anonymous_key_open_scorecard_not_closed",
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-056: validate the terminal artifact contract."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("complete:", "success:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    for field_name in ("live_env_reachable", "no_leaderboard_submission"):
        if field_name in artifact and type(artifact[field_name]) is not bool:
            errors.append(f"{field_name} must be a bare bool")

    if artifact.get("no_leaderboard_submission") is not True:
        errors.append("no_leaderboard_submission must be true")
    if artifact.get("leaderboard_submission_attempted") is True:
        errors.append("leaderboard_submission_attempted must be false")

    mapping = artifact.get("real_metric_mapping")
    if mapping != MetricMapping().to_json():
        errors.append("real_metric_mapping must equal the official EnvironmentScore mapping")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, dict):
        errors.append("preconditions_checked must be a dict")
    else:
        for field_name in ("sdk_importable", "sdk_version", "network_reachable", "base_url"):
            if field_name not in preconditions:
                errors.append(f"preconditions_checked missing {field_name}")
        for field_name in ("sdk_importable", "network_reachable"):
            if field_name in preconditions and type(preconditions[field_name]) is not bool:
                errors.append(f"preconditions_checked.{field_name} must be a bare bool")

    baseline = artifact.get("random_greedy_baseline")
    if not isinstance(baseline, dict):
        errors.append("random_greedy_baseline must be a dict")
    elif artifact.get("live_env_reachable") is True:
        for field_name in (
            "environment",
            "action_budget",
            "actions_taken",
            "baseline_actions",
            "actions_vs_baseline_actions",
            "score",
            "levels_completed",
            "environment_score",
        ):
            if field_name not in baseline:
                errors.append(f"random_greedy_baseline missing {field_name}")
        if int(baseline.get("actions_taken", 0) or 0) < 0:
            errors.append("random_greedy_baseline.actions_taken must be non-negative")
        if int(baseline.get("action_budget", 0) or 0) <= 0:
            errors.append("random_greedy_baseline.action_budget must be positive")

    if "requirements" in artifact and artifact["requirements"] != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-056 and SCENARIO-PHASE4-056")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles must be a dict")
    else:
        for field_name in REQUIRED_FIELD_PRINCIPLES:
            if field_name not in principles:
                errors.append(f"field_principles missing {field_name}")

    if isinstance(verdict, str) and verdict.startswith("blocked_"):
        if artifact.get("live_env_reachable") is not False:
            errors.append("blocked artifacts must set live_env_reachable false")
    if isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("live_env_reachable") is not True:
            errors.append("complete reachable artifacts must set live_env_reachable true")
        offline_validation = artifact.get("offline_validation")
        if not isinstance(offline_validation, dict) or offline_validation.get("passed") is not True:
            errors.append("complete reachable artifacts require passed offline_validation")
    return errors
