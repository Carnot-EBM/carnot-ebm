"""Experiment 4524: stop after the scored ARC level target.

Spec refs: REQ-ARC-FCP-4524, SCENARIO-ARC-FCP-4524.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4524_stop_after_levelup.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
CANONICAL_ACTION_METRIC = {
    "field": "actions",
    "definition": "total_actions_on_solved_games",
}
DEFAULT_GATE_BUDGET = 8000
DEFAULT_MAX_WORKERS = 8
RANDOM_SEED = 4524
BIG_ACTIONS = 1_000_000_000
STOP_AT_SCORED_TARGET_LEVELS = 1
RUN_TO_COMPLETION_CONTROL_TARGET_LEVELS = 5
SUBMITTED_AGENT_CONFIG_BASELINE = {
    "policy": "E3AgentPolicy",
    "cascade": True,
    "value_weight": 0.0,
    "target_levels": STOP_AT_SCORED_TARGET_LEVELS,
    "search_mode": "depth_first_ride",
    "graph_explore_budget": 80,
    "routed_explore_budget": 24,
    "lazy_value_top_k": 4,
    "frontier_batch_size": 1,
    "navigation_cost_tiebreak": False,
    "router_wired": True,
    "world_model_dsl_wired": True,
    "online_discriminative": True,
}
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
REQUIREMENTS = ("REQ-ARC-FCP-4524",)
SCENARIOS = ("SCENARIO-ARC-FCP-4524",)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; success: stop_after_levelup_core_actions_<n>_below_control OR complete: no_overrun_or_drops_solve_honest_null.",
    "inference_substrate": "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor).",
    "core_solves_preserved": "stopping earlier MUST NOT drop a CORE solve (set-containment over {lp85,m0r0,sp80,vc33}).",
    "levels_per_game_preserved": "HARD gate: per-game best_level before vs after -- stopping early MUST NOT shed any game's banked level depth (the CORE gate checks the game set only; the competition scores total LEVELS, so a level-depth regression that the gate would PASS must be caught here).",
    "median_actions_on_core_control": "the run-to-completion baseline, same action field.",
    "median_actions_on_core_best": "the headline -- did stopping at the scored target cut total actions.",
    "action_field_used": "single action field both conditions measured on (A3 metric-mismatch guard).",
    "positive_control_passed": "proves the harness detects a real reduction.",
    "false_negative_risk_checked": "a null is valid only with the control present.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "gate_games",
    "core_games",
    "scored_target_levels",
    "control_target_levels",
    "control_measurement",
    "stop_measurement",
    "per_game_gap",
    "positive_control",
    "submitted_agent_config_before",
    "submitted_agent_config_after",
    "local_gate_budget",
    "leaderboard_submission",
    "result_path",
    "duration_s",
)


def _kit() -> Any:  # pragma: no cover - import boundary for offline ARC SDK.
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def _submitted_agent_config() -> dict[str, Any]:  # pragma: no cover - heavy submitted-agent boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return dict(SUBMITTED_AGENT_CONFIG)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - SDK boundary.
    """REQ-ARC-FCP-4524: verify local resources before measuring."""

    root_path = Path(root)
    spec_path = (
        root_path
        / "openspec"
        / "capabilities"
        / "arc-human-replay-frame-change"
        / "spec.md"
    )
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "spec_has_req_4524": spec_path.exists()
        and "REQ-ARC-FCP-4524" in spec_path.read_text(encoding="utf-8"),
    }
    try:
        _kit().offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import"])
    return checks


def _json_action_label(action_id: int, data: Any) -> str:  # pragma: no cover - SDK boundary.
    return json.dumps({"action": int(action_id), "data": data}, sort_keys=True)


def _apply_json_action_label(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - SDK boundary.
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    payload = json.loads(label)
    return env.step(_game_action(GameAction, int(payload["action"])), data=payload.get("data"))


def _run_game(  # pragma: no cover - SDK boundary.
    game: str,
    *,
    budget: int,
    target_levels: int,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    started = time.perf_counter()
    submitted_config = _submitted_agent_config()
    arc_kit = _kit()
    arc = arc_kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=None,
        target_levels=int(target_levels),
        value_weight=float(submitted_config["value_weight"]),
        search_mode=str(submitted_config["search_mode"]),
        lazy_value_top_k=int(submitted_config["lazy_value_top_k"]),
        frontier_batch_size=submitted_config["frontier_batch_size"],
        navigation_cost_tiebreak=bool(submitted_config["navigation_cost_tiebreak"]),
    )
    frames: list[Any] = []
    latest = None
    actions = 0
    start_level: int | None = None
    best_level = 0
    current_segment: list[str] = []
    first_levelup_segment: list[str] = []
    actions_to_reach_levels: dict[str, int] = {}
    error: str | None = None

    try:
        for _ in range(int(budget)):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                current_segment = []
            elif kind is None:
                break
            else:
                latest = env.step(_game_action(GameAction, int(kind)), data=data)
                actions += 1
                current_segment.append(_json_action_label(int(kind), data))
            if start_level is None:
                start_level = arc_kit.frame_level(latest)
                best_level = int(start_level)
            frames.append(latest)
            if latest is None:
                break
            reached_now = int(arc_kit.frame_level(latest))
            if start_level is not None and reached_now > best_level:
                for level in range(best_level + 1, reached_now + 1):
                    relative = int(level - start_level)
                    actions_to_reach_levels.setdefault(str(relative), int(actions))
                best_level = reached_now
                if not first_levelup_segment:
                    first_levelup_segment = list(current_segment)
    except Exception as exc:
        error = repr(exc)

    try:
        reached = int(arc_kit.frame_level(latest))
    except Exception:
        reached = int(best_level or start_level or 0)
    best_level = max(int(best_level), int(reached))
    levels = max(0, int(best_level) - int(start_level or 0))
    reproduction = None
    if error is None and levels >= 1 and first_levelup_segment:
        reproduction = arc_kit.reproduce(
            game,
            first_levelup_segment,
            _apply_json_action_label,
            claimed_level=int((start_level or 0) + 1),
        )
    reproduced = None if reproduction is None else bool(reproduction.get("reproduced"))
    solved = bool(error is None and levels >= 1 and reproduced is True)
    return {
        "game": game,
        "timed_out": False,
        "solved": solved,
        "levels": int(levels if solved else 0),
        "best_level": int(levels if solved else 0),
        "reached": int(reached),
        "actions": int(actions),
        "actions_to_reach_levels": actions_to_reach_levels if solved else {},
        "actions_to_scored_target": actions_to_reach_levels.get(str(STOP_AT_SCORED_TARGET_LEVELS)),
        "reproduced": reproduced,
        "reproduction": reproduction,
        "target_levels": int(target_levels),
        "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
        "error": error,
    }


def _actions_by_game(measurement: Mapping[str, Any]) -> dict[str, int]:
    actions = measurement.get("actions_by_game")
    if isinstance(actions, Mapping):
        return {str(game): int(value) for game, value in actions.items() if value is not None}
    return {
        str(row["game"]): int(row["actions"])
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping) and row.get("solved") is True and row.get("actions") is not None
    }


def _best_level_by_game(measurement: Mapping[str, Any]) -> dict[str, int]:
    levels = measurement.get("best_level_by_game")
    if isinstance(levels, Mapping):
        return {str(game): int(value) for game, value in levels.items() if value is not None}
    return {
        str(row["game"]): int(row.get("best_level") or row.get("levels") or 0)
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping)
    }


def _solved_games(measurement: Mapping[str, Any]) -> set[str]:
    solved = measurement.get("solved_games")
    if solved is not None:
        return {str(game) for game in solved}
    return {
        str(row["game"])
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping) and row.get("solved") is True
    }


def _median_actions_on_core(measurement: Mapping[str, Any]) -> float:
    actions = _actions_by_game(measurement)
    return float(median([actions.get(game, BIG_ACTIONS) for game in CORE_GAMES]))


def _action_metric_field(measurement: Mapping[str, Any]) -> str:
    metric = measurement.get("action_metric")
    if isinstance(metric, Mapping) and metric.get("field"):
        return str(metric["field"])
    return "actions"


def action_metric_compatibility_error(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> str | None:
    control_field = _action_metric_field(control)
    treatment_field = _action_metric_field(treatment)
    if control_field != treatment_field:
        return f"action metric mismatch control={control_field} treatment={treatment_field}"
    if control_field != "actions":
        return f"action metric must be actions, got {control_field}"
    return None


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_levels: int,
    games: Sequence[str] = GATE_GAMES,
) -> dict[str, Any]:
    solved_rows = [row for row in rows if row.get("solved") is True]
    actions_by_game = {
        str(row["game"]): int(row["actions"])
        for row in solved_rows
        if row.get("actions") is not None
    }
    best_level_by_game = {
        str(row["game"]): int(row.get("best_level") or row.get("levels") or 0)
        for row in rows
    }
    solved_actions = list(actions_by_game.values())
    return {
        "policy": "e3",
        "games": list(games),
        "per_game": [dict(row) for row in rows],
        "action_metric": dict(CANONICAL_ACTION_METRIC),
        "target_levels": int(target_levels),
        "solved_count": int(len(solved_rows)),
        "solved_games": sorted(actions_by_game),
        "actions_by_game": actions_by_game,
        "best_level_by_game": best_level_by_game,
        "median_actions_on_solved": float(median(solved_actions)) if solved_actions else None,
        "median_actions_on_core": _median_actions_on_core({"actions_by_game": actions_by_game}),
        "total_actions_on_solved": int(sum(solved_actions)) if solved_actions else None,
        "timed_out_count": sum(1 for row in rows if row.get("timed_out") is True),
    }


def measure_config(  # pragma: no cover - SDK boundary.
    *,
    games: Sequence[str] = GATE_GAMES,
    budget: int = DEFAULT_GATE_BUDGET,
    max_workers: int = DEFAULT_MAX_WORKERS,
    target_levels: int,
) -> dict[str, Any]:
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            rows = list(
                executor.map(
                    lambda game: _run_game(
                        str(game),
                        budget=int(budget),
                        target_levels=int(target_levels),
                    ),
                    games,
                )
            )
        summary = summarize_rows(rows, target_levels=int(target_levels), games=games)
        summary["budget"] = int(budget)
        return summary
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _core_solves_preserved(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> bool:
    control_core = set(CORE_GAMES) & _solved_games(control)
    return control_core.issubset(_solved_games(treatment))


def _levels_per_game_preserved(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    before = _best_level_by_game(control)
    after = _best_level_by_game(treatment)
    by_game = {
        game: {"before": int(before.get(game, 0)), "after": int(after.get(game, 0))}
        for game in GATE_GAMES
    }
    lost = [
        game
        for game, row in by_game.items()
        if int(row["after"]) < int(row["before"])
    ]
    return {
        "passed": not lost,
        "before_after": by_game,
        "lost_level_depth_games": lost,
    }


def _per_game_gap(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> list[dict[str, Any]]:
    treatment_actions = _actions_by_game(treatment)
    rows_by_game = {
        str(row.get("game")): row
        for row in control.get("per_game", []) or []
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for game in GATE_GAMES:
        row = rows_by_game.get(game, {})
        reach = dict(row.get("actions_to_reach_levels") or {})
        total = row.get("actions")
        target_action = reach.get(str(STOP_AT_SCORED_TARGET_LEVELS))
        overrun = None
        if total is not None and target_action is not None:
            overrun = int(total) - int(target_action)
        out.append(
            {
                "game": game,
                "control_actions_to_reach_levels": reach,
                "control_total_actions": None if total is None else int(total),
                "stop_total_actions": treatment_actions.get(game),
                "overrun_after_scored_target": overrun,
            }
        )
    return out


def positive_control_from_control(control: Mapping[str, Any]) -> dict[str, Any]:
    control_median = _median_actions_on_core(control)
    improved_actions = {
        game: max(1, int(action) - 1000)
        for game, action in _actions_by_game(control).items()
        if game in CORE_GAMES
    }
    improved = {"actions_by_game": improved_actions, "action_metric": dict(CANONICAL_ACTION_METRIC)}
    improved_median = _median_actions_on_core(improved)
    return {
        "passed": bool(improved_actions and improved_median < control_median),
        "control_median": float(control_median),
        "improved_median": float(improved_median),
    }


def _honest_verdict(
    *,
    metric_error: str | None,
    core_preserved: bool,
    level_depth: Mapping[str, Any],
    control_median: float,
    treatment_median: float,
) -> str:
    if metric_error is not None:
        return "complete: action_metric_mismatch_honest_null"
    if not core_preserved:
        return "complete: stop_after_levelup_drops_core_solve_honest_null"
    if level_depth.get("passed") is not True:
        return "complete: stop_after_levelup_drops_level_depth_honest_null"
    if treatment_median < control_median:
        return f"success: stop_after_levelup_core_actions_{int(treatment_median)}_below_control"
    return "complete: no_overrun_or_drops_solve_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    control_measurement: Mapping[str, Any],
    stop_measurement: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
    submitted_agent_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4524: assemble the terminal stop-policy artifact."""

    metric_error = action_metric_compatibility_error(control_measurement, stop_measurement)
    core_preserved = _core_solves_preserved(control_measurement, stop_measurement)
    level_depth = _levels_per_game_preserved(control_measurement, stop_measurement)
    control_median = _median_actions_on_core(control_measurement)
    treatment_median = _median_actions_on_core(stop_measurement)
    verdict = _honest_verdict(
        metric_error=metric_error,
        core_preserved=core_preserved,
        level_depth=level_depth,
        control_median=control_median,
        treatment_median=treatment_median,
    )
    accepted = verdict.startswith("success:")
    before_config = dict(submitted_agent_config or SUBMITTED_AGENT_CONFIG_BASELINE)
    after_config = dict(before_config)
    if accepted:
        after_config["target_levels"] = STOP_AT_SCORED_TARGET_LEVELS

    artifact = {
        "experiment": "experiment_4524_stop_after_levelup",
        "schema": "carnot.arc_stop_after_levelup_4524.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "core_solves_preserved": bool(core_preserved),
        "levels_per_game_preserved": dict(level_depth),
        "median_actions_on_core_control": float(control_median),
        "median_actions_on_core_best": float(treatment_median),
        "action_field_used": "actions" if metric_error is None else f"invalid: {metric_error}",
        "positive_control_passed": bool(positive_control.get("passed")),
        "positive_control": dict(positive_control),
        "false_negative_risk_checked": bool(positive_control.get("passed")),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "gate_games": list(GATE_GAMES),
        "core_games": list(CORE_GAMES),
        "scored_target_levels": int(STOP_AT_SCORED_TARGET_LEVELS),
        "control_target_levels": int(control_measurement.get("target_levels") or 0),
        "control_measurement": dict(control_measurement),
        "stop_measurement": dict(stop_measurement),
        "per_game_gap": _per_game_gap(control_measurement, stop_measurement),
        "submitted_agent_config_before": before_config,
        "submitted_agent_config_after": after_config,
        "local_gate_budget": int(DEFAULT_GATE_BUDGET),
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _blocked_artifact(  # pragma: no cover - missing-resource boundary.
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    empty = summarize_rows([], target_levels=STOP_AT_SCORED_TARGET_LEVELS)
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        control_measurement=empty,
        stop_measurement=empty,
        positive_control={"passed": False},
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = "blocked_offline_arcade_import"
    artifact["false_negative_risk_checked"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-FCP-4524")
    if artifact.get("action_field_used") != "actions" and not blocked:
        errors.append("action_field_used must be actions")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("positive_control_passed") is not True and not blocked:
        errors.append("positive_control_passed must be true")
    if artifact.get("false_negative_risk_checked") is not True and not blocked:
        errors.append("false_negative_risk_checked must be true")
    level_depth = artifact.get("levels_per_game_preserved")
    if not isinstance(level_depth, Mapping) or "before_after" not in level_depth:
        errors.append("levels_per_game_preserved must include before_after")
    if str(verdict).startswith("success:"):
        if artifact.get("core_solves_preserved") is not True:
            errors.append("success cannot drop a CORE solve")
        if not isinstance(level_depth, Mapping) or level_depth.get("passed") is not True:
            errors.append("success cannot drop level depth")
        if float(artifact.get("median_actions_on_core_best") or BIG_ACTIONS) >= float(
            artifact.get("median_actions_on_core_control") or 0
        ):
            errors.append("success must reduce median CORE actions")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(  # pragma: no cover - SDK boundary.
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    measure: Callable[..., dict[str, Any]] = measure_config,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4524: measure stop-at-target versus run-to-completion."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    if preconditions.get("offline_arcade_import") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=preconditions,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        control = measure(
            games=GATE_GAMES,
            budget=DEFAULT_GATE_BUDGET,
            max_workers=DEFAULT_MAX_WORKERS,
            target_levels=RUN_TO_COMPLETION_CONTROL_TARGET_LEVELS,
        )
        stopped = measure(
            games=GATE_GAMES,
            budget=DEFAULT_GATE_BUDGET,
            max_workers=DEFAULT_MAX_WORKERS,
            target_levels=STOP_AT_SCORED_TARGET_LEVELS,
        )
        artifact = build_artifact(
            preconditions_checked=preconditions,
            control_measurement=control,
            stop_measurement=stopped,
            positive_control=positive_control_from_control(control),
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
            submitted_agent_config=_submitted_agent_config(),
        )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - script wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script wrapper.
    raise SystemExit(main())
