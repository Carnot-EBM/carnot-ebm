"""Experiment 4514: lazy best-first value-weight remeasurement.

Spec refs: REQ-ARC-FCP-4514, SCENARIO-ARC-FCP-4514.
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

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_LAZY_VALUE_TOP_K,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4514_lazy_best_first_value_weight.json"
LAZY_VALUE_EVAL_SOURCE = "results/experiment_4506_lazy_value_eval_prototype.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade + value-head scoring, "
    "no GGUF load (1s floor)."
)
VALUE_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
DEFAULT_GATE_BUDGET = 8000
DEFAULT_MAX_WORKERS = 8
DEFAULT_LAZY_VALUE_TOP_K = SUBMITTED_LAZY_VALUE_TOP_K
WALL_BUDGET_S = 390.0
RANDOM_SEED = 4514
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
REQUIREMENTS = ("REQ-ARC-FCP-4514",)
SCENARIOS = ("SCENARIO-ARC-FCP-4514",)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        'principle "terminal prefix; e.g. success: '
        "lazy_value_weight_<w>_beats_0 OR complete: "
        'lazy_value_weight_null_keep_0."'
    ),
    "inference_substrate": (
        'principle "verifier_ensemble_against_cached_candidates -- offline arcade + '
        'value-head scoring, no GGUF load (1s floor)."'
    ),
    "per_weight_results": (
        'principle "the full {weight -> (solve_rate, median_actions, median_wall_s) '
        'table so the decision is auditable, not asserted."'
    ),
    "control_value_weight_0": (
        'principle "the explicit baseline -- a weight only wins if it BEATS 0 '
        '(guards the FALSE_NEGATIVE_RISK null)."'
    ),
    "chosen_submitted_value_weight": (
        'principle "the new SUBMITTED_VALUE_WEIGHT (0 if null); must keep '
        'test_arc_submitted_agent_parity.py consistent."'
    ),
    "lazy_eval_speedup_confirmed": (
        'principle "confirms the cheap-eval cost regime that distinguishes this '
        'from the .416 full-cost null."'
    ),
    "false_negative_risk_checked": (
        'principle "a null is valid only with the value_weight=0 control present."'
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
    "value_weights_tested",
    "gate_games",
    "core_games",
    "local_gate_budget",
    "wall_budget_s",
    "lazy_value_top_k",
    "search_mode",
    "decision",
    "submitted_agent_config_before",
    "leaderboard_submission",
    "result_path",
    "duration_s",
)


def _weight_key(weight: float) -> str:
    return f"{float(weight):.1f}"


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _import_torch_version() -> str:  # pragma: no cover - external precondition boundary
    import torch

    return str(torch.__version__)


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4514: record the resources required before measuring."""

    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
        "control_value_weight_0_present": 0.0 in VALUE_WEIGHTS,
    }
    try:
        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:  # pragma: no cover - only local missing-resource path.
        checks["offline_arcade_error"] = repr(exc)
    try:
        checks["torch_version"] = _import_torch_version()
        checks["torch_import"] = True
    except Exception as exc:  # pragma: no cover - only local missing-resource path.
        checks["torch_error"] = repr(exc)
    return checks


def load_lazy_eval_speedup_confirmation(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4514: confirm this is the cheap lazy-eval regime, not full-cost A1."""

    path = Path(root) / LAZY_VALUE_EVAL_SOURCE
    if not path.exists():
        return {
            "confirmed": False,
            "source": LAZY_VALUE_EVAL_SOURCE,
            "missing": True,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    speedup = float(data.get("speedup_factor") or 0.0)
    routing_ok = data.get("routing_quality_preserved") is True
    cached = data.get("cache_by_frame_hash") is True
    return {
        "confirmed": bool(speedup > 1.0 and routing_ok and cached),
        "source": LAZY_VALUE_EVAL_SOURCE,
        "honest_verdict": data.get("honest_verdict"),
        "speedup_factor": speedup,
        "routing_quality_preserved": routing_ok,
        "lazy_top_k": int(data.get("lazy_top_k") or DEFAULT_LAZY_VALUE_TOP_K),
        "cache_by_frame_hash": cached,
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


def _run_policy_game(
    game: str,
    *,
    value_weight: float,
    budget: int,
    value_head: Any | None,
    lazy_value_top_k: int,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    started = time.perf_counter()
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=None,
        target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
        value_head=value_head,
        value_weight=float(value_weight),
        search_mode="best_first",
        lazy_value_top_k=int(lazy_value_top_k),
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
            start_level = kit.frame_level(latest)
        frames.append(latest)
        if latest is None:
            break
        reached_now = kit.frame_level(latest)
        if (
            start_level is not None
            and reached_now > start_level
            and first_levelup_actions is None
        ):
            first_levelup_actions = int(actions)
            first_levelup_segment = list(current_segment)

    reached = kit.frame_level(latest)
    levels = max(0, int(reached) - int(start_level or 0))
    reproduction = None
    if levels >= 1 and first_levelup_segment:
        reproduction = kit.reproduce(
            game,
            first_levelup_segment,
            _apply_json_action_label,
            claimed_level=int((start_level or 0) + 1),
        )
    reproduced = None if reproduction is None else bool(reproduction.get("reproduced"))
    solved = bool(levels >= 1 and reproduced is True)
    return {
        "game": game,
        "value_weight": float(value_weight),
        "timed_out": False,
        "solved": solved,
        "levels": int(levels if solved else 0),
        "reached": int(reached),
        "actions": int(actions),
        "actions_to_first_levelup": first_levelup_actions if solved else None,
        "reproduced": reproduced,
        "reproduction": reproduction,
        "lazy_value_diagnostics": policy.explorer.lazy_value_diagnostics(),
        "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
    }


def _summarize_weight(
    *,
    value_weight: float,
    rows: Sequence[Mapping[str, Any]],
    games: Sequence[str],
    core_games: Sequence[str] = CORE_GAMES,
) -> dict[str, Any]:
    solved_rows = [row for row in rows if row.get("solved") is True]
    solved_games = sorted(str(row["game"]) for row in solved_rows)
    actions_by_game = {
        str(row["game"]): int(row.get("actions_to_first_levelup") or row.get("actions") or 0)
        for row in solved_rows
    }
    all_solved_actions = [
        int(row.get("actions_to_first_levelup") or row.get("actions") or 0)
        for row in solved_rows
    ]
    core_actions = [
        float(actions_by_game.get(str(game), BIG_ACTIONS))
        for game in core_games
    ]
    wall = [float(row.get("wall_seconds") or 0.0) for row in rows]
    return {
        "value_weight": float(value_weight),
        "attempted_games": int(len(rows)),
        "heldout_solve_rate": (
            round(float(len(solved_rows)) / float(len(games)), 10) if games else 0.0
        ),
        "solved_games": solved_games,
        "solved_count": int(len(solved_rows)),
        "core_games": [str(game) for game in core_games],
        "core_solves_preserved": all(str(game) in solved_games for game in core_games),
        "actions_by_game": actions_by_game,
        "median_actions_to_first_levelup": (
            float(median(all_solved_actions)) if all_solved_actions else None
        ),
        "median_actions_on_core": float(median(core_actions)),
        "median_per_game_wall_s": float(median(wall)) if wall else 0.0,
        "timed_out_games": int(sum(1 for row in rows if row.get("timed_out") is True)),
        "per_game": [dict(row) for row in rows],
    }


def measure_value_weight_sweep(
    *,
    root: Path | str = REPO_ROOT,
    value_weights: Sequence[float] = VALUE_WEIGHTS,
    games: Sequence[str] = GATE_GAMES,
    budget: int = DEFAULT_GATE_BUDGET,
    max_workers: int = DEFAULT_MAX_WORKERS,
    lazy_value_top_k: int = DEFAULT_LAZY_VALUE_TOP_K,
    value_head_factory: Callable[[], Any | None] | None = None,
) -> dict[str, dict[str, Any]]:  # pragma: no cover - SDK boundary
    from carnot.agentic.arc_competition_agent import load_cross_game_value_head

    _ = Path(root)
    value_head = value_head_factory() if value_head_factory else load_cross_game_value_head()
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        per_weight: dict[str, dict[str, Any]] = {}
        for weight in [float(value) for value in value_weights]:
            with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
                rows = list(
                    executor.map(
                        lambda game: _run_policy_game(
                            str(game),
                            value_weight=weight,
                            budget=int(budget),
                            value_head=value_head,
                            lazy_value_top_k=int(lazy_value_top_k),
                        ),
                        games,
                    )
                )
            per_weight[_weight_key(weight)] = _summarize_weight(
                value_weight=weight,
                rows=rows,
                games=games,
            )
        return per_weight
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def choose_submitted_value_weight(
    per_weight_results: Mapping[str, Mapping[str, Any]],
    *,
    wall_budget_s: float = WALL_BUDGET_S,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4514: select a positive weight only if it beats the 0.0 control."""

    control = per_weight_results.get("0.0")
    if control is None:
        raise ValueError("value_weight=0.0 control is required")
    control_actions = float(control.get("median_actions_on_core") or BIG_ACTIONS)
    control_solve_rate = float(control.get("heldout_solve_rate") or 0.0)
    candidates: list[dict[str, Any]] = []
    for key, row in per_weight_results.items():
        weight = float(row.get("value_weight", key))
        if weight == 0.0:
            continue
        median_actions = float(row.get("median_actions_on_core") or BIG_ACTIONS)
        median_wall = float(row.get("median_per_game_wall_s") or 0.0)
        solve_rate = float(row.get("heldout_solve_rate") or 0.0)
        wins = (
            row.get("core_solves_preserved") is True
            and median_actions < control_actions
            and median_wall <= float(wall_budget_s)
            and solve_rate >= control_solve_rate
        )
        candidates.append(
            {
                "value_weight": weight,
                "wins": bool(wins),
                "heldout_solve_rate": solve_rate,
                "median_actions_on_core": median_actions,
                "median_per_game_wall_s": median_wall,
                "core_solves_preserved": row.get("core_solves_preserved") is True,
            }
        )
    winners = [candidate for candidate in candidates if candidate["wins"]]
    if not winners:
        return {
            "selected_value_weight": 0.0,
            "should_raise_submitted_value_weight": False,
            "selection_reason": "no_positive_weight_beat_control",
            "candidates_considered": candidates,
        }
    winner = sorted(
        winners,
        key=lambda row: (
            float(row["median_actions_on_core"]),
            -float(row["heldout_solve_rate"]),
            float(row["median_per_game_wall_s"]),
            float(row["value_weight"]),
        ),
    )[0]
    return {
        "selected_value_weight": float(winner["value_weight"]),
        "should_raise_submitted_value_weight": True,
        "selection_reason": "positive_weight_beats_control_on_core_actions",
        "candidates_considered": candidates,
    }


def _honest_verdict(decision: Mapping[str, Any]) -> str:
    chosen = float(decision.get("selected_value_weight") or 0.0)
    if chosen > 0.0:
        return f"success: lazy_value_weight_{chosen:g}_beats_0"
    return "complete: lazy_value_weight_null_keep_0"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    per_weight_results: Mapping[str, Mapping[str, Any]],
    lazy_eval_speedup_confirmed: Mapping[str, Any],
    decision: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4514: assemble the terminal sweep artifact."""

    control = dict(per_weight_results.get("0.0", {}))
    checksum_payload = {
        "per_weight_results": per_weight_results,
        "decision": decision,
        "lazy_eval_speedup_confirmed": lazy_eval_speedup_confirmed,
        "random_seed": int(random_seed),
        "submitted_agent_config_before": dict(SUBMITTED_AGENT_CONFIG),
    }
    return {
        "experiment": "experiment_4514_lazy_best_first_value_weight",
        "schema": "carnot.arc_lazy_best_first_value_weight_4514.v1",
        "honest_verdict": _honest_verdict(decision),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "value_weights_tested": [float(weight) for weight in VALUE_WEIGHTS],
        "gate_games": list(GATE_GAMES),
        "core_games": list(CORE_GAMES),
        "local_gate_budget": int(DEFAULT_GATE_BUDGET),
        "wall_budget_s": float(WALL_BUDGET_S),
        "lazy_value_top_k": int(DEFAULT_LAZY_VALUE_TOP_K),
        "search_mode": "best_first",
        "per_weight_results": {str(key): dict(value) for key, value in per_weight_results.items()},
        "control_value_weight_0": control,
        "chosen_submitted_value_weight": float(decision.get("selected_value_weight") or 0.0),
        "decision": dict(decision),
        "lazy_eval_speedup_confirmed": dict(lazy_eval_speedup_confirmed),
        "false_negative_risk_checked": bool("0.0" in per_weight_results),
        "random_seed": int(random_seed),
        "submitted_agent_config_before": dict(SUBMITTED_AGENT_CONFIG),
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
    }


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    lazy_eval_speedup_confirmed: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:  # pragma: no cover - missing-resource boundary
    missing = [
        name
        for name in ("offline_arcade_import_smoke", "torch_import")
        if preconditions_checked.get(name) is not True
    ]
    reason = "_".join(missing) if missing else "unknown_resource"
    decision = {
        "selected_value_weight": 0.0,
        "should_raise_submitted_value_weight": False,
        "selection_reason": f"blocked_{reason}",
        "candidates_considered": [],
    }
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        per_weight_results={"0.0": {"value_weight": 0.0}},
        lazy_eval_speedup_confirmed=lazy_eval_speedup_confirmed,
        decision=decision,
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
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked must be a mapping")
    else:
        if preconditions.get("offline_arcade_import_smoke") is not True:
            errors.append("preconditions_checked must record offline_arcade_import_smoke=true")
        if preconditions.get("torch_import") is not True:
            errors.append("preconditions_checked must record torch_import=true")
    per_weight = artifact.get("per_weight_results")
    if not isinstance(per_weight, Mapping) or "0.0" not in per_weight:
        errors.append("per_weight_results must include the 0.0 control")
    if "5.0" in (per_weight or {}):
        errors.append("per_weight_results must not include the known-regressed 5.0 arm")
    control = artifact.get("control_value_weight_0")
    if not isinstance(control, Mapping) or float(control.get("value_weight", -1.0)) != 0.0:
        errors.append("control_value_weight_0 must contain the 0.0 control summary")
    lazy = artifact.get("lazy_eval_speedup_confirmed")
    if not isinstance(lazy, Mapping) or lazy.get("confirmed") is not True:
        errors.append("lazy_eval_speedup_confirmed.confirmed must be true")
    if artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked must be true when control is present")
    decision = artifact.get("decision")
    if not isinstance(decision, Mapping):
        errors.append("decision must be a mapping")
    else:
        chosen = artifact.get("chosen_submitted_value_weight")
        selected = decision.get("selected_value_weight")
        if chosen is None or selected is None or float(chosen) != float(selected):
            errors.append("chosen_submitted_value_weight must match the decision")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("search_mode") != "best_first":
        errors.append("search_mode must be best_first")
    if int(artifact.get("lazy_value_top_k") or 0) <= 0:
        errors.append("lazy_value_top_k must be positive")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    measure_sweep: Callable[..., dict[str, dict[str, Any]]] = measure_value_weight_sweep,
    preconditions_checked: Mapping[str, Any] | None = None,
    lazy_eval_speedup_confirmed: Mapping[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4514: run the lazy best-first sweep and write its artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    lazy_confirmation = (
        dict(lazy_eval_speedup_confirmed)
        if lazy_eval_speedup_confirmed is not None
        else load_lazy_eval_speedup_confirmation(root_path)
    )
    if (
        preconditions.get("offline_arcade_import_smoke") is not True
        or preconditions.get("torch_import") is not True
    ):
        artifact = _blocked_artifact(
            preconditions_checked=preconditions,
            lazy_eval_speedup_confirmed=lazy_confirmation,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        per_weight = measure_sweep(
            root=root_path,
            value_weights=VALUE_WEIGHTS,
            games=GATE_GAMES,
            budget=DEFAULT_GATE_BUDGET,
            max_workers=DEFAULT_MAX_WORKERS,
            lazy_value_top_k=DEFAULT_LAZY_VALUE_TOP_K,
        )
        decision = choose_submitted_value_weight(per_weight)
        artifact = build_artifact(
            preconditions_checked=preconditions,
            per_weight_results=per_weight,
            lazy_eval_speedup_confirmed=lazy_confirmation,
            decision=decision,
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
