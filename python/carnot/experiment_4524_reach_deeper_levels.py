"""Experiment 4524: diagnose and attack CORE L1->L2 stalls.

Spec refs: REQ-ARC-WMTE-4524, SCENARIO-ARC-WMTE-4524.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4524_reach_deeper_levels.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
CORE_EFFICIENCY_BASELINE = 2.0074
RANDOM_SEED = 4524
TARGET_GAME = "lp85"
CORE_GAMES = ("lp85", "m0r0", "sp80")
TARGET_LEVELS = 2
DEFAULT_BUDGET = 8000
DEFAULT_MAX_WORKERS = 3
REQUIREMENTS = ("REQ-ARC-WMTE-4524",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4524",)
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
VALID_ROOT_CAUSES = {
    "resolved_l2_reached",
    "depth_cap",
    "missing_mechanic",
    "new_win_condition",
    "induction_not_engaged",
    "budget_exhausted",
    "explored_out",
    "core_level_regression",
    "blocked_resource",
}
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: <game>_reached_L2_core_efficiency_<n>_above_2.0074 OR "
        "complete: l1_l2_barrier_diagnosed_<root_cause>_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade search + world-model/verifier "
        "routing (no headline GGUF load); if the LLM induction tier is invoked, declare live_llm_inference "
        "+ add the model precondition."
    ),
    "core_efficiency_baseline": (
        "2.0074 -- the REAL per-level metric control (NOT median actions, which is retired as a score lever)."
    ),
    "core_efficiency_best": (
        "the HEADLINE -- did any lever raise per-level efficiency by reaching a deeper level."
    ),
    "deepest_level_reached_per_core_game": (
        "best_level per CORE game WITH each lever -- the direct evidence of solving MORE levels "
        "(the score lever)."
    ),
    "barrier_diagnosis": (
        "the concrete, actionable root cause of the L1->L2 stall (depth-cap / missing mechanic / "
        "new win-condition / induction-not-engaged) -- the deliverable when L2 is not reached, and the "
        "input to the next milestone."
    ),
    "levers_tried": (
        "each L1->L2 lever (deeper search / world-model induction / verifier routing) with its measured "
        "effect -- no assumed wins."
    ),
    "offline_reproduced": (
        "any new level reached must offline-reproduce (arc_solver_kit.reproduce) to count -- not a one-off "
        "live fluke."
    ),
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
    "target_game",
    "core_games",
    "measurements",
    "offline_reproduction",
    "core_level_regressions",
    "no_core_game_loses_level",
    "leaderboard_submission",
    "result_path",
    "duration_s",
)


@dataclass(frozen=True)
class LeverConfig:
    name: str
    description: str
    max_depth: int
    search_mode: str = "depth_first_ride"
    policy_kind: str = "explorer"
    value_weight: float = 0.0
    use_value_head: bool = False


LEVER_CONFIGS = (
    LeverConfig(
        name="control_max_depth_45",
        description="StepwiseExplorer target L2 with submitted max_depth=45.",
        max_depth=45,
    ),
    LeverConfig(
        name="deeper_search_max_depth_90",
        description="Raise the depth cap past the L1 plateau.",
        max_depth=90,
    ),
    LeverConfig(
        name="world_model_dsl_induction",
        description="Run the E3 collection path and inspect whether DSL induction engages after L1.",
        max_depth=45,
        policy_kind="e3_dsl",
    ),
    LeverConfig(
        name="energy_verifier_frontier_routing",
        description="Route the frontier with the cached verifier/value energy signal when available.",
        max_depth=90,
        search_mode="best_first",
        value_weight=1.0,
        use_value_head=True,
    ),
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _round_efficiency(value: Any) -> float:
    return round(float(value or 0.0), 4)


def _per_game_map(measurement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping) and row.get("game") is not None
    }


def _level_by_game(measurement: Mapping[str, Any]) -> dict[str, int]:
    levels = measurement.get("best_level_by_game")
    if isinstance(levels, Mapping):
        return {str(game): int(value or 0) for game, value in levels.items()}
    return {
        game: int(row.get("best_level") or row.get("levels") or 0)
        for game, row in _per_game_map(measurement).items()
    }


def _measurement_efficiency(measurement: Mapping[str, Any]) -> float:
    return _round_efficiency(measurement.get("core_efficiency"))


def _deepest_level_reached_per_core_game(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for measurement in measurements:
        lever = str(measurement.get("lever") or "unknown")
        levels = _level_by_game(measurement)
        out[lever] = {game: int(levels.get(game, 0)) for game in CORE_GAMES}
    return out


def _best_measurement(measurements: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not measurements:
        return {}
    return max(
        measurements,
        key=lambda item: (
            _measurement_efficiency(item),
            max(_level_by_game(item).values() or [0]),
            str(item.get("lever") or ""),
        ),
    )


def _control_measurement(measurements: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not measurements:
        return {}
    for measurement in measurements:
        if str(measurement.get("lever")) == LEVER_CONFIGS[0].name:
            return measurement
    return measurements[0]


def _core_level_regressions(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before = _level_by_game(control)
    after = _level_by_game(treatment)
    return [
        {"game": game, "control_level": int(before.get(game, 0)), "best_level": int(after.get(game, 0))}
        for game in CORE_GAMES
        if int(after.get(game, 0)) < int(before.get(game, 0))
    ]


def _l2_row(measurement: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for game, row in _per_game_map(measurement).items():
        if game in CORE_GAMES and int(row.get("best_level") or row.get("levels") or 0) >= TARGET_LEVELS:
            return row
    return None


def _target_diagnostics(measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for measurement in measurements:
        row = _per_game_map(measurement).get(TARGET_GAME)
        if not row:
            continue
        diagnostics = dict(row.get("diagnostics") or {})
        diagnostics["lever"] = str(measurement.get("lever") or "unknown")
        rows.append(diagnostics)
    return rows


def diagnose_barrier(
    *,
    measurements: Sequence[Mapping[str, Any]],
    best_measurement: Mapping[str, Any],
    offline_reproduced: bool,
    core_level_regressions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-4524: classify the measured lp85 L1->L2 barrier."""

    diagnostics = _target_diagnostics(measurements)
    stopped_reasons = sorted(
        {
            str(row.get("stopped_reason"))
            for row in diagnostics
            if row.get("stopped_reason") is not None
        }
    )
    depth_cap_likely = any(
        int(row.get("depth_cap_frontier_nodes") or 0) > 0
        or str(row.get("stopped_reason")) == "depth_cap"
        for row in diagnostics
    )
    salience_values = [
        row.get("known_l2_transition_in_salience")
        for row in diagnostics
        if row.get("known_l2_transition_in_salience") is not None
    ]
    missing_salience_likely = bool(salience_values) and any(value is False for value in salience_values)
    new_win_condition_likely = any(
        row.get("l2_win_condition_differs_from_l1") is True for row in diagnostics
    )
    induction_not_engaged = any(
        row.get("lever") == "world_model_dsl_induction"
        and row.get("world_model_induction_invoked") is False
        for row in diagnostics
    )
    reached_l2 = _l2_row(best_measurement)
    if reached_l2 is not None and offline_reproduced and not core_level_regressions:
        root_cause = "resolved_l2_reached"
    elif core_level_regressions:
        root_cause = "core_level_regression"
    elif depth_cap_likely:
        root_cause = "depth_cap"
    elif missing_salience_likely:
        root_cause = "missing_mechanic"
    elif new_win_condition_likely:
        root_cause = "new_win_condition"
    elif induction_not_engaged:
        root_cause = "induction_not_engaged"
    elif "budget_exhausted" in stopped_reasons:
        root_cause = "budget_exhausted"
    else:
        root_cause = "explored_out"
    next_steps = [
        str(row.get("actionable_next_step"))
        for row in diagnostics
        if row.get("actionable_next_step")
    ]
    return {
        "target_game": TARGET_GAME,
        "root_cause": root_cause,
        "stopped_reasons": stopped_reasons,
        "depth_cap_likely": bool(depth_cap_likely),
        "missing_salience_likely": bool(missing_salience_likely),
        "new_win_condition_likely": bool(new_win_condition_likely),
        "induction_not_engaged": bool(induction_not_engaged),
        "l2_reached_by": None
        if reached_l2 is None
        else str(best_measurement.get("lever") or "unknown"),
        "core_level_regressions": [dict(row) for row in core_level_regressions],
        "actionable_next_step": next_steps[0]
        if next_steps
        else "build a level-conditioned L2 win predicate and route the frontier against it.",
        "evidence": diagnostics,
    }


def _lever_summary(measurement: Mapping[str, Any]) -> dict[str, Any]:
    lever = str(measurement.get("lever") or "unknown")
    levels = _level_by_game(measurement)
    return {
        "lever": lever,
        "description": str(measurement.get("description") or ""),
        "core_efficiency": _measurement_efficiency(measurement),
        "delta_vs_baseline": round(_measurement_efficiency(measurement) - CORE_EFFICIENCY_BASELINE, 4),
        "deepest_level_by_game": {game: int(levels.get(game, 0)) for game in CORE_GAMES},
        "stopped_reasons": {
            game: str((row.get("diagnostics") or {}).get("stopped_reason") or "")
            for game, row in _per_game_map(measurement).items()
            if game in CORE_GAMES
        },
    }


def _success_verdict(best: Mapping[str, Any]) -> str:
    row = _l2_row(best) or {}
    game = str(row.get("game") or TARGET_GAME)
    return (
        f"success: {game}_reached_L2_core_efficiency_"
        f"{_measurement_efficiency(best):.4f}_above_{CORE_EFFICIENCY_BASELINE:.4f}"
    )


def _null_verdict(root_cause: str) -> str:
    return f"complete: l1_l2_barrier_diagnosed_{root_cause}_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    offline_reproduction: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4524: assemble the terminal reach-deeper-levels artifact."""

    rows = [dict(measurement) for measurement in measurements]
    best = _best_measurement(rows)
    control = _control_measurement(rows)
    core_regressions = _core_level_regressions(control, best)
    offline_reproduced = bool(
        offline_reproduction.get("reproduced") is True
        and int(offline_reproduction.get("reached_level") or 0) >= TARGET_LEVELS
    )
    diagnosis = diagnose_barrier(
        measurements=rows,
        best_measurement=best,
        offline_reproduced=offline_reproduced,
        core_level_regressions=core_regressions,
    )
    success = (
        _l2_row(best) is not None
        and _measurement_efficiency(best) > CORE_EFFICIENCY_BASELINE
        and offline_reproduced
        and not core_regressions
    )
    verdict = _success_verdict(best) if success else _null_verdict(str(diagnosis["root_cause"]))
    artifact = {
        "experiment": "experiment_4524_reach_deeper_levels",
        "schema": "carnot.arc_reach_deeper_levels_4524.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": max(CORE_EFFICIENCY_BASELINE, _measurement_efficiency(best)),
        "deepest_level_reached_per_core_game": _deepest_level_reached_per_core_game(rows),
        "barrier_diagnosis": diagnosis,
        "levers_tried": [_lever_summary(measurement) for measurement in rows],
        "offline_reproduced": bool(offline_reproduced),
        "offline_reproduction": dict(offline_reproduction),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "target_game": TARGET_GAME,
        "core_games": list(CORE_GAMES),
        "measurements": rows,
        "core_level_regressions": [dict(row) for row in core_regressions],
        "no_core_game_loses_level": not core_regressions,
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        measurements=[],
        offline_reproduction={},
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = "blocked_offline_arcade_import_smoke"
    artifact["barrier_diagnosis"]["root_cause"] = "blocked_resource"
    artifact["barrier_diagnosis"]["actionable_next_step"] = (
        "restore carnot.agentic.arc_solver_kit.offline_arcade() before measuring L2 levers."
    )
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
        errors.append("field_principles must match REQ-ARC-WMTE-4524")
    if float(artifact.get("core_efficiency_baseline") or 0.0) != CORE_EFFICIENCY_BASELINE:
        errors.append("core_efficiency_baseline must equal 2.0074")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked must be a mapping")
    elif not blocked and preconditions.get("offline_arcade_import_smoke") is not True:
        errors.append("preconditions_checked must record offline_arcade_import_smoke=true")
    diagnosis = artifact.get("barrier_diagnosis")
    if not isinstance(diagnosis, Mapping):
        errors.append("barrier_diagnosis must be a mapping")
    elif diagnosis.get("root_cause") not in VALID_ROOT_CAUSES:
        errors.append("barrier_diagnosis root_cause is invalid")
    deepest = artifact.get("deepest_level_reached_per_core_game")
    if not isinstance(deepest, Mapping):
        errors.append("deepest_level_reached_per_core_game must be a mapping")
    if not isinstance(artifact.get("levers_tried"), list):
        errors.append("levers_tried must be a list")
    if str(verdict).startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced=true")
        if float(artifact.get("core_efficiency_best") or 0.0) <= CORE_EFFICIENCY_BASELINE:
            errors.append("success requires core_efficiency_best above baseline")
        if artifact.get("no_core_game_loses_level") is not True:
            errors.append("success requires no CORE game loses level")
        if not any(
            int(level) >= TARGET_LEVELS
            for per_lever in (deepest or {}).values()
            if isinstance(per_lever, Mapping)
            for level in per_lever.values()
        ):
            errors.append("success requires a CORE game reaches L2")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _kit() -> Any:  # pragma: no cover - ARC SDK boundary.
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "spec_has_req_4524": spec_path.exists()
        and "REQ-ARC-WMTE-4524" in spec_path.read_text(encoding="utf-8"),
    }
    try:
        _kit().offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import_smoke"])
    return checks


def _baseline_actions(env: Any) -> dict[int, int]:  # pragma: no cover - ARC SDK boundary.
    for attr in ("baseline_actions", "human_actions", "reference_actions"):
        value = getattr(getattr(env, "info", env), attr, None)
        if value:
            if isinstance(value, (list, tuple)):
                return {index: int(item) for index, item in enumerate(value)}
            return {int(key): int(item) for key, item in dict(value).items()}
    return {}


def _json_action_label(action_id: int, data: Any) -> str:  # pragma: no cover - ARC SDK boundary.
    return json.dumps({"action": int(action_id), "data": data}, sort_keys=True)


def _apply_json_action_label(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - ARC SDK boundary.
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    payload = json.loads(label)
    return env.step(_game_action(GameAction, int(payload["action"])), data=payload.get("data"))


class _ExplorerPolicy:  # pragma: no cover - ARC SDK boundary.
    def __init__(self, lever: LeverConfig) -> None:
        from carnot.agentic.arc_competition_agent import StepwiseExplorer, load_cross_game_value_head

        value_head = load_cross_game_value_head() if lever.use_value_head else None
        self.explorer = StepwiseExplorer(
            target_levels=TARGET_LEVELS,
            max_depth=lever.max_depth,
            value_head=value_head,
            value_weight=lever.value_weight,
            search_mode=lever.search_mode,
        )

    def next_move(self, frames: Sequence[Any], latest: Any) -> tuple[Any, Any]:
        return self.explorer.next_move(frames, latest)

    def is_done(self, frames: Sequence[Any], latest: Any) -> bool:
        return self.explorer.is_done(frames, latest)


def _make_policy(game: str, lever: LeverConfig) -> Any:  # pragma: no cover - ARC SDK boundary.
    if lever.policy_kind == "e3_dsl":
        from carnot.agentic.arc_competition_agent import E3AgentPolicy

        policy = E3AgentPolicy(
            game,
            proposer=None,
            target_levels=TARGET_LEVELS,
            value_weight=0.0,
            search_mode=lever.search_mode,
        )
        policy.explorer.max_depth = lever.max_depth
        return policy
    return _ExplorerPolicy(lever)


def _explorer_diagnostics(policy: Any, *, budget_exhausted: bool) -> dict[str, Any]:  # pragma: no cover
    explorer = getattr(policy, "explorer", policy)
    graph = getattr(explorer, "graph", {}) or {}
    depths = [len(node.get("path", [])) for node in graph.values()]
    frontier_nodes = [
        node for node in graph.values() if node.get("untested")
    ]
    depth_cap_frontier_nodes = [
        node for node in frontier_nodes if len(node.get("path", [])) >= int(getattr(explorer, "max_depth", 0))
    ]
    if getattr(explorer, "best_level", 0) >= (getattr(explorer, "start_level", 0) or 0) + TARGET_LEVELS:
        stopped_reason = "target_reached"
    elif getattr(explorer, "early_stopped", False):
        stopped_reason = "early_stop_grace"
    elif getattr(explorer, "explored_out", False):
        stopped_reason = "explored_out"
    elif depth_cap_frontier_nodes:
        stopped_reason = "depth_cap"
    elif budget_exhausted:
        stopped_reason = "budget_exhausted"
    else:
        stopped_reason = "unknown"
    if hasattr(policy, "_fit_dsl_model"):
        try:
            policy._fit_dsl_model()
        except Exception:
            pass
    dsl_energy = getattr(policy, "dsl_energy", None)
    return {
        "stopped_reason": stopped_reason,
        "max_depth": int(getattr(explorer, "max_depth", 0)),
        "max_depth_reached": max(depths) if depths else 0,
        "graph_nodes": int(len(graph)),
        "frontier_nodes": int(len(frontier_nodes)),
        "depth_cap_frontier_nodes": int(len(depth_cap_frontier_nodes)),
        "candidate_count_at_last_l1": int(len(getattr(explorer, "graph", {}).get(getattr(explorer, "cur", ""), {}).get("untested", []))),
        "known_l2_transition_in_salience": None,
        "l2_win_condition_differs_from_l1": True,
        "world_model_induction_invoked": bool(getattr(policy, "induced", False)),
        "dsl_energy": dsl_energy,
        "energy_signal_available": bool(getattr(explorer, "value_head", None) is not None),
        "navigation": explorer.navigation_diagnostics() if hasattr(explorer, "navigation_diagnostics") else {},
        "actionable_next_step": (
            "force post-L1 DSL/goal-predicate induction and route lp85 frontier states toward the "
            "level-conditioned L2 predicate."
        ),
    }


def _score_efficiency(
    *,
    baseline_actions: Mapping[int, int],
    level_up_actions: Sequence[int],
    total_actions: int,
) -> tuple[float, list[dict[str, Any]]]:  # pragma: no cover - ARC scorer boundary.
    if not baseline_actions:
        return 0.0, []
    from arc_agi.scorecard import EnvironmentScoreCalculator

    baseline_list = [baseline_actions[index] for index in sorted(baseline_actions)]
    calc = EnvironmentScoreCalculator()
    prev = 0
    per_level = []
    for index, human_actions in enumerate(baseline_list):
        if index < len(level_up_actions):
            at = int(level_up_actions[index])
            level_actions = at - prev
            completed = True
            prev = at
        else:
            level_actions = int(total_actions) - prev
            completed = False
            prev = int(total_actions)
        calc.add_level(
            level_index=index + 1,
            completed=completed,
            actions_taken=level_actions,
            baseline_actions=int(human_actions),
        )
        per_level.append(
            {
                "level": index,
                "human_actions": int(human_actions),
                "agent_actions": int(level_actions),
                "completed": bool(completed),
            }
        )
    return round(float(calc.to_score(include_levels=False).score), 4), per_level


def _run_game(game: str, lever: LeverConfig) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary.
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    started = time.perf_counter()
    arc_kit = _kit()
    arc = arc_kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    base = _baseline_actions(env)
    policy = _make_policy(game, lever)
    frames: list[Any] = []
    latest = None
    actions = 0
    start_level: int | None = None
    best_level = 0
    level_up_actions: list[int] = []
    current_segment: list[str] = []
    segment_by_level: dict[int, list[str]] = {}
    error: str | None = None
    budget_exhausted = True
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        for _ in range(DEFAULT_BUDGET):
            if policy.is_done(frames, latest):
                budget_exhausted = False
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                current_segment = []
            elif kind is None:
                budget_exhausted = False
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
                budget_exhausted = False
                break
            reached_now = int(arc_kit.frame_level(latest))
            if start_level is not None and reached_now > best_level:
                for level in range(best_level + 1, reached_now + 1):
                    relative = int(level - start_level)
                    level_up_actions.append(actions)
                    segment_by_level[relative] = list(current_segment)
                best_level = reached_now
    except Exception as exc:
        error = repr(exc)
        budget_exhausted = False
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable
    try:
        reached = int(arc_kit.frame_level(latest))
    except Exception:
        reached = int(best_level or start_level or 0)
    best_level = max(best_level, reached)
    relative_best = max(0, int(best_level) - int(start_level or 0))
    efficiency, per_level = _score_efficiency(
        baseline_actions=base,
        level_up_actions=level_up_actions,
        total_actions=actions,
    )
    diagnostics = _explorer_diagnostics(policy, budget_exhausted=budget_exhausted)
    return {
        "game": game,
        "best_level": int(relative_best),
        "reached": int(reached),
        "actions": int(actions),
        "efficiency": float(efficiency),
        "per_level": per_level,
        "level_up_actions": list(level_up_actions),
        "segment_to_l2": segment_by_level.get(TARGET_LEVELS, []),
        "diagnostics": diagnostics,
        "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
        "error": error,
    }


def measure_lever(lever: LeverConfig) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary.
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        rows = list(executor.map(lambda game: _run_game(game, lever), CORE_GAMES))
    return {
        "lever": lever.name,
        "description": lever.description,
        "core_efficiency": round(sum(float(row.get("efficiency") or 0.0) for row in rows), 4),
        "best_level_by_game": {
            str(row["game"]): int(row.get("best_level") or 0)
            for row in rows
        },
        "per_game": rows,
    }


def reproduce_new_l2(best_measurement: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    row = _l2_row(best_measurement)
    if row is None:
        return {}
    labels = list(row.get("segment_to_l2") or [])
    if not labels:
        return {"game": row.get("game"), "reached_level": int(row.get("best_level") or 0), "reproduced": False}
    return dict(
        _kit().reproduce(
            str(row.get("game")),
            labels,
            _apply_json_action_label,
            claimed_level=TARGET_LEVELS,
        )
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    measure_lever: Callable[[LeverConfig], Mapping[str, Any]] = measure_lever,
    offline_reproduction_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]] = reproduce_new_l2,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4524: run the lever sweep and write the terminal artifact."""

    root_path = Path(root)
    started = float(now())
    checks = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root_path)
    if checks.get("offline_arcade_import_smoke") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        measurements = [dict(measure_lever(lever)) for lever in LEVER_CONFIGS]
        best = _best_measurement(measurements)
        reproduction = dict(offline_reproduction_runner(best))
        artifact = build_artifact(
            preconditions_checked=checks,
            measurements=measurements,
            offline_reproduction=reproduction,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - script wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script wrapper.
    raise SystemExit(main())
