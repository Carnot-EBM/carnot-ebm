"""Exp 4475: wire the stronger generic ARC stack into the submitted default.

Spec refs: REQ-REPORT-4475-LIVE-STACK,
SCENARIO-REPORT-4475-LIVE-STACK-PARITY.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.agentic.arc_competition_agent import (
    E3AgentPolicy,
    SUBMITTED_AGENT_CONFIG,
    _level_of,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4475_wire_stronger_generic_stack.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "offline_reproduced",
    "reproduced_levels",
    "preconditions_checked",
    "before_generic_solve_rate",
    "after_generic_solve_rate",
    "generic_solve_rate_delta",
    "submitted_agent_config",
    "tests_pass",
    "field_principles",
    "spec_refs",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "MUST start with a terminal prefix complete:/complete_/success:/success_/"
            "passed:/passed_/shipped:/shipped_ so the reconciler classifies it as terminal "
            "(Verdict Terminal-Prefix Discipline)."
        )
    },
    "inference_substrate": {
        "principle": (
            "explicit declaration (live_llm_inference | "
            "verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) "
            "so adversarial_verify applies the right floor."
        )
    },
    "offline_reproduced": {
        "principle": (
            "a solve not reproducible offline is wasted effort -- only reproduced levels count "
            "(ARC Solve Reproducibility)."
        )
    },
    "reproduced_levels": {
        "principle": (
            "headline metric reproducible_total_levels grows monotonically; report the count "
            "banked, real-env-confirmed."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified before launching; pre-empts the "
            "silent-missing-resource fabrication mode."
        )
    },
    "before_generic_solve_rate": {
        "principle": "bare float: held-out generic solve-rate for the old bare submitted stack.",
    },
    "after_generic_solve_rate": {
        "principle": "bare float: held-out generic solve-rate for the exact submitted default.",
    },
    "generic_solve_rate_delta": {
        "principle": "bare float: after-before solve-rate delta; this is the headline, not replay levels.",
    },
    "submitted_agent_config": {
        "principle": "the single source of truth for what ships in make_carnot_agent.",
    },
    "tests_pass": {
        "principle": "bare bool: the focused tests backing this integration passed.",
    },
    "field_principles": {
        "principle": "schema self-description so artifact review checks field intent.",
    },
    "spec_refs": {
        "principle": "OpenSpec anchors that the tests and artifact claim to satisfy.",
    },
    "reproducibility_checksum": {
        "principle": "sha256 over the stable measurement/config payload.",
    },
}


class _BlockedEnvGame:
    """Proxy that blocks `env._game` while preserving the public reset/step API."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        if name == "_game":
            raise AttributeError("env._game is blocked for frame-only live-legal evaluation")
        return getattr(self._env, name)


class _NoopProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return False, {}


def _stable_hash(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rate(solved: int, attempted: int) -> float:
    return round(float(solved) / float(attempted), 10) if attempted else 0.0


def _duration(started_at: float, ended_at: float) -> float:
    return max(0.0, round(float(ended_at - started_at), 6))


def make_baseline_policy(game: str) -> E3AgentPolicy:
    """Old submitted behavior: first-level target, no router refresh, inert value weight."""
    return E3AgentPolicy(
        game,
        proposer=_NoopProposer(),
        explore_budget=80,
        target_levels=1,
        value_head=None,
        value_weight=0.0,
        search_mode="depth_first_ride",
        mechanic_detector=lambda _frame: None,
    )


def make_submitted_policy(game: str) -> E3AgentPolicy:
    """Exact submitted-default config, with local proposer disabled for cached-candidate scoring."""
    return E3AgentPolicy(
        game,
        proposer=_NoopProposer(),
        target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
        value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
        search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
    )


def run_policy_game(
    game: str,
    policy: E3AgentPolicy,
    *,
    arcade: Any,
    game_action: Any,
    budget: int,
) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary
    env = _BlockedEnvGame(arcade.make(game, scorecard_id=arcade.open_scorecard()))
    frames: list[Any] = []
    latest = None
    start_level = None
    reached = 0
    actions = 0
    for _ in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(game_action, f"ACTION{kind}"), data=data)
            actions += 1
        if latest is None:
            break
        frames.append(latest)
        if start_level is None:
            start_level = _level_of(latest)
        reached = _level_of(latest)
        if start_level is not None and reached >= start_level + policy.explorer.target_levels:
            break
    solved = bool(start_level is not None and reached > start_level)
    return {
        "game": game,
        "solved": solved,
        "start_level": int(start_level or 0),
        "reached_level": int(reached),
        "levels_delta": int(max(0, reached - (start_level or 0))),
        "actions": int(actions),
    }


def run_offline_benchmark(
    games: Sequence[str],
    *,
    budget: int = 400,
    baseline_factory: Callable[[str], E3AgentPolicy] = make_baseline_policy,
    submitted_factory: Callable[[str], E3AgentPolicy] = make_submitted_policy,
) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit

    arcade = kit.offline_arcade()
    before_rows = [
        run_policy_game(game, baseline_factory(game), arcade=arcade, game_action=GameAction, budget=budget)
        for game in games
    ]
    after_rows = [
        run_policy_game(game, submitted_factory(game), arcade=arcade, game_action=GameAction, budget=budget)
        for game in games
    ]
    before_solved = sum(1 for row in before_rows if row["solved"])
    after_solved = sum(1 for row in after_rows if row["solved"])
    attempted = len(games)
    return {
        "games": list(games),
        "budget": int(budget),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "before_solved": before_solved,
        "after_solved": after_solved,
        "attempted_games": attempted,
        "before_generic_solve_rate": _rate(before_solved, attempted),
        "after_generic_solve_rate": _rate(after_solved, attempted),
    }


def build_artifact(
    *,
    before_generic_solve_rate: float,
    after_generic_solve_rate: float,
    before_solved: int,
    after_solved: int,
    attempted_games: int,
    reproduced_levels: int,
    offline_reproduced: bool,
    preconditions_checked: Mapping[str, Any],
    tests_pass: bool,
    duration_s: float,
    benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    delta = round(float(after_generic_solve_rate) - float(before_generic_solve_rate), 10)
    checksum_payload = {
        "before_generic_solve_rate": float(before_generic_solve_rate),
        "after_generic_solve_rate": float(after_generic_solve_rate),
        "generic_solve_rate_delta": delta,
        "before_solved": int(before_solved),
        "after_solved": int(after_solved),
        "attempted_games": int(attempted_games),
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "preconditions_checked": dict(preconditions_checked),
        "benchmark": dict(benchmark or {}),
    }
    return {
        "experiment": "experiment_4475_wire_stronger_generic_stack",
        "schema": "carnot.exp4475.wire_stronger_generic_stack.v1",
        "honest_verdict": (
            "complete: submitted_default_stronger_generic_stack_wired"
            if tests_pass
            else "complete: submitted_default_stronger_generic_stack_tests_failed"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": int(reproduced_levels),
        "preconditions_checked": dict(preconditions_checked),
        "before_generic_solve_rate": float(before_generic_solve_rate),
        "after_generic_solve_rate": float(after_generic_solve_rate),
        "generic_solve_rate_delta": delta,
        "before_solved": int(before_solved),
        "after_solved": int(after_solved),
        "attempted_games": int(attempted_games),
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "tests_pass": bool(tests_pass),
        "benchmark": dict(benchmark or {}),
        "duration_s": float(duration_s),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": [
            "REQ-REPORT-4475-LIVE-STACK",
            "SCENARIO-REPORT-4475-LIVE-STACK-PARITY",
            "SCENARIO-REPORT-4475-LIVE-STACK-FORWARD-NAV",
        ],
        "leaderboard_submission": False,
        "reproducibility_checksum": _stable_hash(checksum_payload),
        "result_path": RESULT_RELATIVE_PATH,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict missing terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be verifier_ensemble_against_cached_candidates")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("tests_pass")) is not bool:
        errors.append("tests_pass must be bare bool")
    if artifact.get("submitted_agent_config") != SUBMITTED_AGENT_CONFIG:
        errors.append("submitted_agent_config diverges from arc_competition_agent.SUBMITTED_AGENT_CONFIG")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = principles.get(field)
            if not isinstance(row, Mapping) or not row.get("principle"):
                errors.append(f"missing field principle: {field}")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:  # pragma: no cover - integration entrypoint
    started = time.time()
    games = ("lp85", "r11l", "tn36", "cn04", "sb26", "sc25")
    preconditions = {
        "arcade_import": False,
        "submitted_agent_config_loaded": bool(SUBMITTED_AGENT_CONFIG),
        "router_wired": SUBMITTED_AGENT_CONFIG.get("router_wired") is True,
        "world_model_dsl_wired": SUBMITTED_AGENT_CONFIG.get("world_model_dsl_wired") is True,
        "env_game_blocked": True,
    }
    benchmark: dict[str, Any] = {"games": list(games), "blocked_reason": ""}
    try:
        benchmark = run_offline_benchmark(games, budget=400)
        preconditions["arcade_import"] = True
    except Exception as exc:
        benchmark = {"games": list(games), "blocked_reason": type(exc).__name__, "error": str(exc)}
    attempted = int(benchmark.get("attempted_games") or len(games))
    before_rate = float(benchmark.get("before_generic_solve_rate") or 0.0)
    after_rate = float(benchmark.get("after_generic_solve_rate") or 0.0)
    before_solved = int(benchmark.get("before_solved") or 0)
    after_solved = int(benchmark.get("after_solved") or 0)
    artifact = build_artifact(
        before_generic_solve_rate=before_rate,
        after_generic_solve_rate=after_rate,
        before_solved=before_solved,
        after_solved=after_solved,
        attempted_games=attempted,
        reproduced_levels=sum(row.get("levels_delta", 0) for row in benchmark.get("after_rows", [])),
        offline_reproduced=not bool(benchmark.get("blocked_reason")),
        preconditions_checked=preconditions,
        tests_pass=False,
        duration_s=_duration(started, time.time()),
        benchmark=benchmark,
    )
    write_artifact(REPO_ROOT, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
