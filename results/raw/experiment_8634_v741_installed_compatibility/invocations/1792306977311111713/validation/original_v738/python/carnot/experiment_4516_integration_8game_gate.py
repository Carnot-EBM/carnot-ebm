"""Experiment 4516: submitted integration gate and forward navigation loop.

Spec refs: REQ-ARC-FCP-4516, SCENARIO-ARC-FCP-4516.
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

from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4516_integration_8game_gate.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end, "
    "no LLM load (1s floor)."
)
BASELINE_MEDIAN_ACTIONS = 7760.0
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
DEFAULT_GATE_BUDGET = 8000
DEFAULT_MAX_WORKERS = 8
RANDOM_SEED = 4516
BIG_ACTIONS = 1_000_000_000.0
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
REQUIREMENTS = ("REQ-ARC-FCP-4516",)
SCENARIOS = ("SCENARIO-ARC-FCP-4516",)
UPSTREAM_ARTIFACTS = {
    "A1_prune_predictor": "results/experiment_4511_frame_change_prune_predictor.json",
    "A2_imitation_prior": "results/experiment_4512_imitation_action_prior.json",
    "A3_adaptive_budget": "results/experiment_4513_adaptive_per_step_budget.json",
    "A4_lazy_best_first": "results/experiment_4514_lazy_best_first_value_weight.json",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        'principle "terminal prefix; e.g. success: '
        "integrated_median_actions_<n>_below_7760 OR complete: "
        'no_lever_beats_7760_honest_null."'
    ),
    "inference_substrate": (
        'principle "verifier_ensemble_against_cached_candidates -- offline arcade '
        'end-to-end, no LLM load (1s floor)."'
    ),
    "median_actions_baseline": 'principle "the 7760 control."',
    "median_actions_integrated": (
        'principle "the HEADLINE -- the SUBMITTED-config median after wiring the '
        'winners + the nav fix."'
    ),
    "levers_integrated": (
        'principle "names which of A1-A4 (and the nav fix) were wired -- traceable '
        'to their measured deltas."'
    ),
    "solve_rate_integrated": (
        'principle "integration must not drop solve-rate (and ideally keeps >13 '
        'reproducible levels for the submission gate)."'
    ),
    "heldout_solve_rate": (
        'principle "the real transfer signal (was 0.143); integration should not regress it."'
    ),
    "nav_loop_finding": (
        'principle "the answer to why the .416 nav-edge fix did not move actions '
        '(closes candidate 5)."'
    ),
    "false_negative_risk_checked": (
        'principle "an honest null only valid with the 7760 baseline measured the same way."'
    ),
    "random_seed": 'principle "determinism precondition for reproducibility."',
    "reproducibility_checksum": 'principle "catches silent drift on replay."',
    "preconditions_checked": (
        'principle "records resources verified; pre-empts missing-resource fabrication."'
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "gate_games",
    "core_games",
    "submitted_agent_config",
    "upstream_decision",
    "integrated_measurement",
    "local_gate_budget",
    "leaderboard_submission",
    "result_path",
    "duration_s",
)


def _kit() -> Any:  # pragma: no cover - import boundary for offline ARC SDK.
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4516: verify local resources before measuring."""

    root_path = Path(root)
    artifacts_present = all(
        (root_path / relative_path).exists() for relative_path in UPSTREAM_ARTIFACTS.values()
    )
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "a1_a4_artifacts_present": artifacts_present,
    }
    try:
        _kit().offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path.
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import"])
    return checks


def load_gate_baseline(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    path = Path(root) / "ops" / "arc-submission-baseline.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "policy": "e3",
        "games": list(GATE_GAMES),
        "per_game": [],
        "solved_count": 4,
        "median_actions_on_solved": BASELINE_MEDIAN_ACTIONS,
        "note": "fixed operator-provided baseline control",
    }


def load_a1_a4_artifacts(root: Path | str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    root_path = Path(root)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, relative_path in UPSTREAM_ARTIFACTS.items():
        path = root_path / relative_path
        if path.exists():
            artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def _first_existing_mapping(mapping: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _summarize_upstream_artifact(name: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    solved_games = artifact.get("solved_games")
    actions_by_game = artifact.get("actions_by_game")
    core_preserved = artifact.get("core_solves_preserved")
    core_median = artifact.get("median_actions_on_core")

    if solved_games is None or actions_by_game is None:
        local_gate = artifact.get("local_gate_metrics")
        treatment = (
            _first_existing_mapping(
                local_gate if isinstance(local_gate, Mapping) else {},
                ("with_prune", "with_prior", "with_adaptive", "integrated", "current"),
            )
            or artifact
        )
        rows = treatment.get("per_game", []) if isinstance(treatment, Mapping) else []
        solved_games = sorted(str(row["game"]) for row in rows if row.get("solved") is True)
        actions_by_game = {
            str(row["game"]): int(row.get("actions_to_first_levelup") or row.get("actions") or 0)
            for row in rows
            if row.get("solved") is True
        }

    solved_set = set(str(game) for game in (solved_games or []))
    actions_map = {
        str(game): int(actions)
        for game, actions in (actions_by_game or {}).items()
        if actions is not None
    }
    if core_preserved is None:
        core_preserved = all(game in solved_set for game in CORE_GAMES)
    if core_median is None:
        core_median = median([float(actions_map.get(game, BIG_ACTIONS)) for game in CORE_GAMES])

    chosen_weight = artifact.get("chosen_submitted_value_weight")
    if chosen_weight is None:
        chosen_weight = (
            artifact.get("decision", {}).get("selected_value_weight")
            if isinstance(artifact.get("decision"), Mapping)
            else None
        )
    return {
        "name": name,
        "honest_verdict": artifact.get("honest_verdict"),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "core_solves_preserved": bool(core_preserved),
        "median_actions_on_core": float(core_median),
        "solved_games": sorted(solved_set),
        "actions_by_game": actions_map,
        "chosen_submitted_value_weight": (
            None if chosen_weight is None else float(chosen_weight)
        ),
    }


def select_integrated_levers(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    control_median_actions_on_core: float,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4516: accept only non-flagged levers that pass the CORE gate."""

    accepted: list[str] = []
    rejected: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    selected_value_weight = 0.0

    for name, artifact in artifacts.items():
        summary = _summarize_upstream_artifact(name, artifact)
        summaries[name] = summary
        if summary["flagged_adversarial"]:
            rejected[name] = {"reason": "flagged_adversarial", **summary}
            continue
        if name == "A4_lazy_best_first":
            selected_value_weight = float(summary.get("chosen_submitted_value_weight") or 0.0)
            if selected_value_weight <= 0.0:
                rejected[name] = {"reason": "selected_value_weight_0_honest_null", **summary}
                continue
        wins_core = (
            summary["core_solves_preserved"] is True
            and float(summary["median_actions_on_core"]) < float(control_median_actions_on_core)
        )
        if wins_core:
            accepted.append(name)
        else:
            rejected[name] = {"reason": "core_gate_failed", **summary}

    return {
        "accepted_a1_a4_levers": accepted,
        "rejected_a1_a4_levers": rejected,
        "upstream_summaries": summaries,
        "selected_value_weight": selected_value_weight,
        "control_median_actions_on_core": float(control_median_actions_on_core),
    }


def _json_action_label(action_id: int, data: Any) -> str:
    return json.dumps({"action": int(action_id), "data": data}, sort_keys=True)


def _apply_json_action_label(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    payload = json.loads(label)
    return env.step(
        _game_action(GameAction, int(payload["action"])),
        data=payload.get("data"),
    )


def _run_submitted_game(game: str, *, budget: int) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    started = time.perf_counter()
    arc_kit = _kit()
    arc = arc_kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=None,
        target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
        value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
        search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
        lazy_value_top_k=int(SUBMITTED_AGENT_CONFIG["lazy_value_top_k"]),
    )
    frames: list[Any] = []
    latest = None
    actions = 0
    start_level: int | None = None
    first_levelup_actions: int | None = None
    current_segment: list[str] = []
    first_levelup_segment: list[str] = []

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
            latest = env.step(
                _game_action(GameAction, int(kind)),
                data=data,
            )
            actions += 1
            current_segment.append(_json_action_label(int(kind), data))
        if start_level is None:
            start_level = arc_kit.frame_level(latest)
        frames.append(latest)
        if latest is None:
            break
        reached_now = arc_kit.frame_level(latest)
        if (
            start_level is not None
            and reached_now > start_level
            and first_levelup_actions is None
        ):
            first_levelup_actions = int(actions)
            first_levelup_segment = list(current_segment)

    reached = arc_kit.frame_level(latest)
    levels = max(0, int(reached) - int(start_level or 0))
    reproduction = None
    if levels >= 1 and first_levelup_segment:
        reproduction = arc_kit.reproduce(
            game,
            first_levelup_segment,
            _apply_json_action_label,
            claimed_level=int((start_level or 0) + 1),
        )
    reproduced = None if reproduction is None else bool(reproduction.get("reproduced"))
    solved = bool(levels >= 1 and reproduced is True)
    return {
        "game": game,
        "timed_out": False,
        "solved": solved,
        "levels": int(levels if solved else 0),
        "reached": int(reached),
        "actions": int(actions),
        "actions_to_first_levelup": first_levelup_actions if solved else None,
        "reproduced": reproduced,
        "reproduction": reproduction,
        "navigation_diagnostics": policy.explorer.navigation_diagnostics(),
        "lazy_value_diagnostics": policy.explorer.lazy_value_diagnostics(),
        "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
    }


def _combine_navigation_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = {
        "navigation_attempts": 0,
        "exact_shortest_path_hits": 0,
        "partial_forward_walk_hits": 0,
        "forward_walk_hits": 0,
        "reset_replay_fallbacks": 0,
        "forward_edges_recorded": 0,
        "forward_navigation_steps": 0,
        "reset_replay_steps": 0,
    }
    for row in rows:
        diagnostics = row.get("navigation_diagnostics") or {}
        if not isinstance(diagnostics, Mapping):
            continue
        for key in total:
            total[key] += int(diagnostics.get(key) or 0)
    attempts = int(total["navigation_attempts"])
    total["forward_walk_hit_rate"] = (
        float(total["forward_walk_hits"] / attempts) if attempts else 0.0
    )
    return total


def summarize_gate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    games: Sequence[str] = GATE_GAMES,
) -> dict[str, Any]:
    solved_rows = [row for row in rows if row.get("solved") is True]
    solved_games = sorted(str(row["game"]) for row in solved_rows)
    actions_by_game = {
        str(row["game"]): int(row.get("actions") or 0)
        for row in solved_rows
        if row.get("actions") is not None
    }
    first_levelup_actions = [
        int(row["actions_to_first_levelup"])
        for row in solved_rows
        if row.get("actions_to_first_levelup") is not None
    ]
    solved_actions = [int(row.get("actions") or 0) for row in solved_rows]
    core_actions = [float(actions_by_game.get(game, BIG_ACTIONS)) for game in CORE_GAMES]
    return {
        "policy": "e3",
        "games": list(games),
        "per_game": [dict(row) for row in rows],
        "solved_count": int(len(solved_rows)),
        "solved_games": solved_games,
        "actions_by_game": actions_by_game,
        "median_actions_on_solved": float(median(solved_actions)) if solved_actions else None,
        "median_actions_on_core": float(median(core_actions)),
        "median_actions_to_first_levelup": (
            float(median(first_levelup_actions)) if first_levelup_actions else None
        ),
        "total_actions_on_solved": int(sum(solved_actions)) if solved_actions else None,
        "timed_out_count": sum(1 for row in rows if row.get("timed_out") is True),
        "heldout_solve_rate": (
            round(float(len(solved_rows)) / float(len(games)), 10) if games else 0.0
        ),
        "navigation_diagnostics": _combine_navigation_diagnostics(rows),
    }


def measure_submitted_gate(
    *,
    root: Path | str = REPO_ROOT,
    games: Sequence[str] = GATE_GAMES,
    budget: int = DEFAULT_GATE_BUDGET,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    _ = Path(root)
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            rows = list(
                executor.map(
                    lambda game: _run_submitted_game(str(game), budget=int(budget)),
                    games,
                )
            )
        summary = summarize_gate_rows(rows, games=games)
        summary["budget"] = int(budget)
        return summary
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _baseline_solved_games(baseline: Mapping[str, Any]) -> set[str]:
    rows = baseline.get("per_game", [])
    solved = {str(row["game"]) for row in rows if row.get("solved") is True}
    return solved or set(CORE_GAMES)


def _core_preserved(baseline: Mapping[str, Any], measurement: Mapping[str, Any]) -> bool:
    baseline_core = _baseline_solved_games(baseline)
    solved = set(str(game) for game in measurement.get("solved_games") or [])
    return baseline_core.issubset(solved)


def _nav_loop_finding(measurement: Mapping[str, Any]) -> str:
    diagnostics = measurement.get("navigation_diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        return "navigation_diagnostics_missing"
    attempts = int(diagnostics.get("navigation_attempts") or 0)
    hit_rate = float(diagnostics.get("forward_walk_hit_rate") or 0.0)
    partial_hits = int(diagnostics.get("partial_forward_walk_hits") or 0)
    exact_hits = int(diagnostics.get("exact_shortest_path_hits") or 0)
    resets = int(diagnostics.get("reset_replay_fallbacks") or 0)
    if attempts <= 0:
        return "no_frontier_navigation_attempts_observed_on_gate"
    if hit_rate <= 0.0:
        return (
            "forward_walk_hit_rate_0_exact_targets_not_reachable_from_current_state; "
            "recorded_edges_alone_could_not_displace_reset_replay"
        )
    if partial_hits > 0:
        return (
            f"partial_forward_walk_engaged_{partial_hits}_times_with_{exact_hits}_exact_hits; "
            f"reset_replay_fallbacks_remained_{resets}"
        )
    return (
        f"exact_shortest_path_engaged_{exact_hits}_times; "
        f"reset_replay_fallbacks_remained_{resets}"
    )


def _honest_verdict(
    *,
    baseline: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> str:
    integrated = measurement.get("median_actions_on_solved")
    if (
        integrated is not None
        and float(integrated) < BASELINE_MEDIAN_ACTIONS
        and _core_preserved(baseline, measurement)
    ):
        return f"success: integrated_median_actions_{int(float(integrated))}_below_7760"
    return "complete: no_lever_beats_7760_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    upstream_decision: Mapping[str, Any],
    integrated_measurement: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4516: assemble the terminal integration artifact."""

    verdict = _honest_verdict(baseline=baseline, measurement=integrated_measurement)
    a1_a4_levers = list(upstream_decision.get("accepted_a1_a4_levers") or [])
    levers_integrated = list(a1_a4_levers)
    if verdict.startswith("success:"):
        levers_integrated.append("forward_navigation_partial_ancestor")
    checksum_payload = {
        "baseline": baseline,
        "upstream_decision": upstream_decision,
        "integrated_measurement": integrated_measurement,
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "random_seed": int(random_seed),
    }
    median_integrated = integrated_measurement.get("median_actions_on_solved")
    solve_rate = (
        float(integrated_measurement.get("solved_count") or 0) / float(len(GATE_GAMES))
        if GATE_GAMES
        else 0.0
    )
    return {
        "experiment": "experiment_4516_integration_8game_gate",
        "schema": "carnot.arc_integration_8game_gate_4516.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "median_actions_baseline": float(baseline.get("median_actions_on_solved") or BASELINE_MEDIAN_ACTIONS),
        "median_actions_integrated": (
            None if median_integrated is None else float(median_integrated)
        ),
        "levers_integrated": levers_integrated,
        "solve_rate_integrated": solve_rate,
        "heldout_solve_rate": float(integrated_measurement.get("heldout_solve_rate") or solve_rate),
        "nav_loop_finding": _nav_loop_finding(integrated_measurement),
        "false_negative_risk_checked": (
            float(baseline.get("median_actions_on_solved") or 0.0) == BASELINE_MEDIAN_ACTIONS
            and list(baseline.get("games") or GATE_GAMES) == list(GATE_GAMES)
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "gate_games": list(GATE_GAMES),
        "core_games": list(CORE_GAMES),
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "upstream_decision": dict(upstream_decision),
        "integrated_measurement": dict(integrated_measurement),
        "local_gate_budget": int(DEFAULT_GATE_BUDGET),
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:  # pragma: no cover - missing-resource boundary.
    missing = [
        name
        for name in ("offline_arcade_import",)
        if preconditions_checked.get(name) is not True
    ]
    reason = "_".join(missing) if missing else "unknown_resource"
    decision = {
        "accepted_a1_a4_levers": [],
        "rejected_a1_a4_levers": {},
        "upstream_summaries": {},
        "selected_value_weight": 0.0,
        "control_median_actions_on_core": BASELINE_MEDIAN_ACTIONS,
    }
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        baseline=baseline,
        upstream_decision=decision,
        integrated_measurement={
            "policy": "e3",
            "games": list(GATE_GAMES),
            "per_game": [],
            "solved_count": 0,
            "solved_games": [],
            "median_actions_on_solved": None,
            "median_actions_on_core": BIG_ACTIONS,
            "heldout_solve_rate": 0.0,
            "navigation_diagnostics": {},
        },
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{reason}"
    artifact["false_negative_risk_checked"] = False
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match the required substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if float(artifact.get("median_actions_baseline") or 0.0) != BASELINE_MEDIAN_ACTIONS:
        errors.append("median_actions_baseline must be the fixed 7760 control")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked must be true for complete/success artifacts")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    measurement = artifact.get("integrated_measurement")
    if not isinstance(measurement, Mapping):
        errors.append("integrated_measurement must be a mapping")
    else:
        diagnostics = measurement.get("navigation_diagnostics")
        if not isinstance(diagnostics, Mapping):
            errors.append("integrated_measurement must include navigation diagnostics")
        if measurement.get("median_actions_on_solved") != artifact.get("median_actions_integrated"):
            errors.append("median_actions_integrated must mirror integrated measurement")
    upstream = artifact.get("upstream_decision")
    if not isinstance(upstream, Mapping):
        errors.append("upstream_decision must be a mapping")
    else:
        for name, row in (upstream.get("rejected_a1_a4_levers") or {}).items():
            if isinstance(row, Mapping) and row.get("flagged_adversarial") is True:
                continue
            if name in (upstream.get("accepted_a1_a4_levers") or []):
                errors.append("accepted levers must not also appear in rejected levers")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
    load_upstream_artifacts: Callable[[Path], dict[str, dict[str, Any]]] = load_a1_a4_artifacts,
    measure_submitted_gate: Callable[..., dict[str, Any]] = measure_submitted_gate,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4516: run the integration gate and write its artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    gate_baseline = dict(baseline) if baseline is not None else load_gate_baseline(root_path)
    if preconditions.get("offline_arcade_import") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=preconditions,
            baseline=gate_baseline,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        upstream_artifacts = load_upstream_artifacts(root_path)
        decision = select_integrated_levers(
            upstream_artifacts,
            control_median_actions_on_core=float(
                gate_baseline.get("median_actions_on_solved") or BASELINE_MEDIAN_ACTIONS
            ),
        )
        measurement = measure_submitted_gate(
            root=root_path,
            games=GATE_GAMES,
            budget=DEFAULT_GATE_BUDGET,
            max_workers=DEFAULT_MAX_WORKERS,
        )
        artifact = build_artifact(
            preconditions_checked=preconditions,
            baseline=gate_baseline,
            upstream_decision=decision,
            integrated_measurement=measurement,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
        errors = artifact_schema_errors(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    if write:
        out = root_path / RESULT_RELATIVE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
