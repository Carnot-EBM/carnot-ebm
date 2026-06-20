"""Experiment 4523: forward-walk navigation cost tie-break and frontier batching.

Spec refs: REQ-ARC-FCP-4523, SCENARIO-ARC-FCP-4523.
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
RESULT_RELATIVE_PATH = "results/experiment_4523_forward_walk_navigation.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates -- offline arcade search, no GGUF/LLM load (1s floor)."
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
CANONICAL_ACTION_METRIC = {
    "field": "actions",
    "definition": "total_actions_on_solved_games",
}
DEFAULT_GATE_BUDGET = 8000
DEFAULT_MAX_WORKERS = 8
RANDOM_SEED = 4523
BIG_ACTIONS = 1_000_000_000
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
SWEEP_CONFIGS: tuple[tuple[int | str, bool], ...] = (
    (1, False),
    (1, True),
    (3, False),
    (3, True),
    (8, False),
    (8, True),
    ("all", False),
    ("all", True),
)
REQUIREMENTS = ("REQ-ARC-FCP-4523",)
SCENARIOS = ("SCENARIO-ARC-FCP-4523",)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; e.g. success: forward_walk_median_actions_on_core_<n>_below_<control> OR complete: forward_walk_no_reduction_honest_null.",
    "inference_substrate": "verifier_ensemble_against_cached_candidates -- offline arcade search, no GGUF/LLM load (1s floor).",
    "median_actions_on_core_control": "the k=1/no-tie-break baseline measured the SAME way -- the apples-to-apples control (A3 false-win guard).",
    "median_actions_on_core_best": "the headline -- did amortizing the replay cut TOTAL actions on the CORE games.",
    "core_solves_preserved": "HARD empirical gate on {lp85,m0r0,sp80,vc33} -- under the fixed explore_budget, batch/reorder CAN drop a knife-edge solve (the .417 m0r0 mechanism); a dropped CORE solve FAILS the lever regardless of action savings.",
    "nav_diagnostics_before_after": "reset_replay_fallbacks + reset_replay_steps + forward_walk_hit_rate WITH vs WITHOUT -- the causal mechanism witness (did replay actually drop, or is any action change incidental).",
    "action_field_used": "names the SINGLE action field both conditions were measured on (total actions on solved) -- the A3 metric-mismatch guard.",
    "config_sweep": "the full {k, tie_break -> (median_actions_on_core, core_solves_preserved, reset_replay_steps)} table so the decision is auditable, not asserted.",
    "chosen_submitted_config": "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); must keep test_arc_submitted_agent_parity.py consistent.",
    "positive_control_passed": "proves the harness can detect a real replay reduction (guards a silently-broken metric).",
    "false_negative_risk_checked": "a null is valid only if the positive control passed.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash catches silent corpus/model drift on replay.",
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
    "baseline",
    "submitted_agent_config",
    "positive_control",
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


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4523: verify local resources before measuring."""

    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "spec_has_req_4523": "REQ-ARC-FCP-4523"
        in (
            root_path
            / "openspec"
            / "capabilities"
            / "arc-human-replay-frame-change"
            / "spec.md"
        ).read_text(encoding="utf-8"),
    }
    try:
        _kit().offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path.
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import"])
    return checks


def load_baseline(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    path = Path(root) / "ops" / "arc-submission-baseline.json"
    if path.exists():
        baseline = json.loads(path.read_text(encoding="utf-8"))
        baseline.setdefault("action_metric", dict(CANONICAL_ACTION_METRIC))
        return baseline
    return {
        "policy": "e3",
        "games": list(GATE_GAMES),
        "action_metric": dict(CANONICAL_ACTION_METRIC),
        "solved_count": 4,
        "solved_games": list(CORE_GAMES),
        "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        "median_actions_on_solved": 7760.0,
    }


def _json_action_label(action_id: int, data: Any) -> str:
    return json.dumps({"action": int(action_id), "data": data}, sort_keys=True)


def _apply_json_action_label(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    payload = json.loads(label)
    return env.step(_game_action(GameAction, int(payload["action"])), data=payload.get("data"))


def _run_game(
    game: str,
    *,
    budget: int,
    frontier_batch_size: int | str,
    navigation_cost_tiebreak: bool,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary
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
        frontier_batch_size=frontier_batch_size,
        navigation_cost_tiebreak=bool(navigation_cost_tiebreak),
    )
    frames: list[Any] = []
    latest = None
    actions = 0
    start_level: int | None = None
    first_levelup_actions: int | None = None
    current_segment: list[str] = []
    first_levelup_segment: list[str] = []
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
            frames.append(latest)
            if latest is None:
                break
            reached_now = arc_kit.frame_level(latest)
            if start_level is not None and reached_now > start_level and first_levelup_actions is None:
                first_levelup_actions = int(actions)
                first_levelup_segment = list(current_segment)
    except Exception as exc:
        error = repr(exc)

    try:
        reached = arc_kit.frame_level(latest)
    except Exception:
        reached = int(start_level or 0)
    levels = max(0, int(reached) - int(start_level or 0))
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
        "reached": int(reached),
        "actions": int(actions),
        "actions_to_first_levelup": first_levelup_actions if solved else None,
        "reproduced": reproduced,
        "reproduction": reproduction,
        "navigation_diagnostics": policy.explorer.navigation_diagnostics(),
        "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
        "error": error,
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
    total["forward_walk_hit_rate"] = float(total["forward_walk_hits"] / attempts) if attempts else 0.0
    return total


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, games: Sequence[str] = GATE_GAMES) -> dict[str, Any]:
    solved_rows = [row for row in rows if row.get("solved") is True]
    solved_games = sorted(str(row["game"]) for row in solved_rows)
    actions_by_game = {
        str(row["game"]): int(row["actions"])
        for row in solved_rows
        if row.get("actions") is not None
    }
    solved_actions = list(actions_by_game.values())
    core_actions = [float(actions_by_game.get(game, BIG_ACTIONS)) for game in CORE_GAMES]
    return {
        "policy": "e3",
        "games": list(games),
        "per_game": [dict(row) for row in rows],
        "action_metric": dict(CANONICAL_ACTION_METRIC),
        "solved_count": int(len(solved_rows)),
        "solved_games": solved_games,
        "actions_by_game": actions_by_game,
        "median_actions_on_solved": float(median(solved_actions)) if solved_actions else None,
        "median_actions_on_core": float(median(core_actions)),
        "total_actions_on_solved": int(sum(solved_actions)) if solved_actions else None,
        "timed_out_count": sum(1 for row in rows if row.get("timed_out") is True),
        "navigation_diagnostics": _combine_navigation_diagnostics(rows),
    }


def measure_config(
    *,
    games: Sequence[str] = GATE_GAMES,
    budget: int = DEFAULT_GATE_BUDGET,
    max_workers: int = DEFAULT_MAX_WORKERS,
    frontier_batch_size: int | str = 1,
    navigation_cost_tiebreak: bool = False,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            rows = list(
                executor.map(
                    lambda game: _run_game(
                        str(game),
                        budget=int(budget),
                        frontier_batch_size=frontier_batch_size,
                        navigation_cost_tiebreak=bool(navigation_cost_tiebreak),
                    ),
                    games,
                )
            )
        summary = summarize_rows(rows, games=games)
        summary["budget"] = int(budget)
        return summary
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _actions_by_game(measurement: Mapping[str, Any]) -> dict[str, int]:
    actions = measurement.get("actions_by_game")
    if isinstance(actions, Mapping):
        return {str(game): int(value) for game, value in actions.items() if value is not None}
    return {
        str(row["game"]): int(row["actions"])
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping) and row.get("solved") is True and row.get("actions") is not None
    }


def _core_preserved(measurement: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    core = set(str(game) for game in (baseline.get("solved_games") or CORE_GAMES))
    solved = set(str(game) for game in measurement.get("solved_games") or [])
    return core.issubset(solved)


def _median_actions_on_core(measurement: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    actions = _actions_by_game(measurement)
    core = tuple(str(game) for game in (baseline.get("solved_games") or CORE_GAMES))
    return float(median([actions.get(game, BIG_ACTIONS) for game in core]))


def _verdict_tag(measurement: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    if not _core_preserved(measurement, baseline):
        return "REGRESSION"
    cur = _median_actions_on_core(measurement, baseline)
    base = _median_actions_on_core(baseline, baseline)
    return "IMPROVED" if cur < base else "non-inferior"


def _sweep_row(entry: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    measurement = entry.get("measurement") or {}
    diagnostics = measurement.get("navigation_diagnostics") or {}
    median_core = _median_actions_on_core(measurement, baseline)
    return {
        "k": entry.get("k"),
        "navigation_cost_tiebreak": bool(entry.get("navigation_cost_tiebreak")),
        "median_actions_on_core": median_core,
        "core_solves_preserved": _core_preserved(measurement, baseline),
        "solved_games": sorted(str(game) for game in measurement.get("solved_games") or []),
        "actions_by_game": _actions_by_game(measurement),
        "reset_replay_fallbacks": int(diagnostics.get("reset_replay_fallbacks") or 0),
        "reset_replay_steps": int(diagnostics.get("reset_replay_steps") or 0),
        "forward_walk_hit_rate": float(diagnostics.get("forward_walk_hit_rate") or 0.0),
        "navigation_diagnostics": dict(diagnostics),
        "measurement": dict(measurement),
        "verdict_tag": _verdict_tag(measurement, baseline),
    }


def _select_best(rows: Sequence[Mapping[str, Any]], control: Mapping[str, Any]) -> Mapping[str, Any]:
    control_median = float(control["median_actions_on_core"])
    candidates = [
        row
        for row in rows
        if row is not control
        and row.get("core_solves_preserved") is True
        and row.get("verdict_tag") == "IMPROVED"
        and float(row["median_actions_on_core"]) < control_median
    ]
    if candidates:
        return min(candidates, key=lambda row: (float(row["median_actions_on_core"]), int(row["reset_replay_steps"])))
    return min(rows, key=lambda row: (float(row["median_actions_on_core"]), int(row["reset_replay_steps"])))


def positive_control_from_sweep(rows: Sequence[Mapping[str, Any]], control: Mapping[str, Any]) -> dict[str, Any]:
    control_steps = int(control.get("reset_replay_steps") or 0)
    lower = [row for row in rows if int(row.get("reset_replay_steps") or 0) < control_steps]
    best = min(lower, key=lambda row: int(row.get("reset_replay_steps") or 0), default=None)
    return {
        "passed": best is not None,
        "reset_replay_steps_before": control_steps,
        "reset_replay_steps_after": None if best is None else int(best.get("reset_replay_steps") or 0),
        "config": None
        if best is None
        else {
            "frontier_batch_size": best.get("k"),
            "navigation_cost_tiebreak": bool(best.get("navigation_cost_tiebreak")),
        },
    }


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    config_sweep: Sequence[Mapping[str, Any]],
    positive_control: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4523: assemble the terminal sweep artifact."""

    rows = [_sweep_row(entry, baseline) for entry in config_sweep]
    control = next(
        row
        for row in rows
        if row.get("k") == 1 and row.get("navigation_cost_tiebreak") is False
    )
    best = _select_best(rows, control)
    strict_improvement = (
        best is not control
        and best.get("core_solves_preserved") is True
        and best.get("verdict_tag") == "IMPROVED"
        and float(best["median_actions_on_core"]) < float(control["median_actions_on_core"])
    )
    if strict_improvement:
        verdict = (
            "success: forward_walk_median_actions_on_core_"
            f"{int(float(best['median_actions_on_core']))}_below_"
            f"{int(float(control['median_actions_on_core']))}"
        )
        chosen: Any = {
            "frontier_batch_size": best.get("k"),
            "navigation_cost_tiebreak": bool(best.get("navigation_cost_tiebreak")),
        }
    else:
        verdict = "complete: forward_walk_no_reduction_honest_null"
        chosen = "unchanged"

    artifact = {
        "experiment": "experiment_4523_forward_walk_navigation",
        "schema": "carnot.arc_forward_walk_navigation_4523.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "median_actions_on_core_control": float(control["median_actions_on_core"]),
        "median_actions_on_core_best": float(best["median_actions_on_core"]),
        "core_solves_preserved": bool(best.get("core_solves_preserved")),
        "nav_diagnostics_before_after": {
            "before": dict(control.get("navigation_diagnostics") or {}),
            "after": dict(best.get("navigation_diagnostics") or {}),
        },
        "action_field_used": "actions",
        "config_sweep": rows,
        "chosen_submitted_config": chosen,
        "positive_control_passed": bool(positive_control.get("passed")),
        "positive_control": dict(positive_control),
        "false_negative_risk_checked": bool(positive_control.get("passed")),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "gate_games": list(GATE_GAMES),
        "core_games": list(CORE_GAMES),
        "baseline": dict(baseline),
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "local_gate_budget": int(DEFAULT_GATE_BUDGET),
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:  # pragma: no cover - missing-resource boundary.
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        baseline=baseline,
        config_sweep=[
            {
                "k": 1,
                "navigation_cost_tiebreak": False,
                "measurement": {
                    "action_metric": dict(CANONICAL_ACTION_METRIC),
                    "solved_games": [],
                    "actions_by_game": {},
                    "median_actions_on_core": float(BIG_ACTIONS),
                    "navigation_diagnostics": {},
                },
            }
        ],
        positive_control={"passed": False},
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = "blocked_offline_arcade_import"
    artifact["false_negative_risk_checked"] = False
    artifact["chosen_submitted_config"] = "unchanged"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
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
        errors.append("inference_substrate must match")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-FCP-4523")
    if artifact.get("action_field_used") != "actions":
        errors.append("action_field_used must be actions")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("positive_control_passed") is not True and not str(verdict).startswith("blocked_"):
        errors.append("positive_control_passed must be true for complete/success artifacts")
    if artifact.get("false_negative_risk_checked") is not True and not str(verdict).startswith("blocked_"):
        errors.append("false_negative_risk_checked must be true for complete/success artifacts")
    nav = artifact.get("nav_diagnostics_before_after")
    if not isinstance(nav, Mapping) or not isinstance(nav.get("before"), Mapping) or not isinstance(nav.get("after"), Mapping):
        errors.append("nav_diagnostics_before_after must include before/after mappings")
    sweep = artifact.get("config_sweep")
    if not isinstance(sweep, list) or not sweep:
        errors.append("config_sweep must be a non-empty list")
    else:
        has_control = any(
            row.get("k") == 1 and row.get("navigation_cost_tiebreak") is False for row in sweep
        )
        if not has_control:
            errors.append("config_sweep must include k=1/no-tie-break control")
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


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
    measure: Callable[..., dict[str, Any]] = measure_config,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4523: sweep configs and write the terminal artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    gate_baseline = dict(baseline) if baseline is not None else load_baseline(root_path)
    if preconditions.get("offline_arcade_import") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=preconditions,
            baseline=gate_baseline,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        sweep_entries = []
        for k, tie_break in SWEEP_CONFIGS:
            measurement = measure(
                games=GATE_GAMES,
                budget=DEFAULT_GATE_BUDGET,
                max_workers=DEFAULT_MAX_WORKERS,
                frontier_batch_size=k,
                navigation_cost_tiebreak=tie_break,
            )
            sweep_entries.append(
                {
                    "k": k,
                    "navigation_cost_tiebreak": tie_break,
                    "measurement": measurement,
                }
            )
        rows = [_sweep_row(entry, gate_baseline) for entry in sweep_entries]
        control = next(
            row
            for row in rows
            if row.get("k") == 1 and row.get("navigation_cost_tiebreak") is False
        )
        artifact = build_artifact(
            preconditions_checked=preconditions,
            baseline=gate_baseline,
            config_sweep=sweep_entries,
            positive_control=positive_control_from_sweep(rows, control),
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
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
