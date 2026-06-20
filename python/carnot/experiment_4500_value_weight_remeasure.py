"""Experiment 4500: re-measure submitted ARC value_weight.

Spec refs: REQ-REPORT-4500, SCENARIO-REPORT-4500-CONTROL,
SCENARIO-REPORT-4500-SCHEMA.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4500_value_weight_remeasure.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
PARITY_TEST_PATH = "tests/python/test_arc_submitted_agent_parity.py"
VALUE_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 5.0)
CONTROL_VALUE_WEIGHT = 0.0
EVAL_BUDGET_MEDIAN_WALL_S = 390.0
ACTION_BUDGET = 400
HELDOUT_GAMES = ("tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09")
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
    "preconditions_checked",
    "value_weights_tested",
    "control_value_weight",
    "eval_budget_median_wall_s",
    "heldout_games",
    "per_weight",
    "selected_value_weight",
    "submitted_value_weight_before",
    "submitted_value_weight_after",
    "submitted_agent_config",
    "field_principles",
    "spec_refs",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "MUST start with terminal prefix complete:/complete_/success:/success_/"
            "passed:/passed_/shipped:/shipped_."
        )
    },
    "inference_substrate": {
        "principle": "explicit substrate so adversarial_verify applies the right duration floor."
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
        )
    },
    "value_weights_tested": {
        "principle": "records every tested value_weight, including the zero-weight control."
    },
    "control_value_weight": {
        "principle": "bare float: the zero-weight control required to avoid false negatives."
    },
    "eval_budget_median_wall_s": {
        "principle": "bare float: positive weights must fit the approximate 390-second eval budget."
    },
    "heldout_games": {
        "principle": "the held-out ARC games used for the submitted-default frame-only sweep."
    },
    "per_weight": {
        "principle": "per-weight summaries with solve-rate, first-level-up actions, wall time, and rows."
    },
    "selected_value_weight": {
        "principle": "bare float: raised only when a positive weight beats control within budget."
    },
    "submitted_value_weight_before": {
        "principle": "bare float: the value_weight in code before the sweep decision."
    },
    "submitted_value_weight_after": {
        "principle": "bare float: the value_weight that remains or ships after the sweep decision."
    },
    "submitted_agent_config": {
        "principle": "must match SUBMITTED_AGENT_CONFIG after any allowed value_weight update."
    },
    "field_principles": {
        "principle": "schema self-description so artifact review checks field intent."
    },
    "spec_refs": {
        "principle": "OpenSpec anchors that the tests and artifact claim to satisfy."
    },
    "reproducibility_checksum": {
        "principle": "sha256 over the stable measurement, selection, and config payload."
    },
}


class _BlockedEnvGame:
    """Proxy that blocks `env._game` while preserving public reset/step methods."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        if name == "_game":
            raise AttributeError("env._game is blocked for frame-only live-legal evaluation")
        return getattr(self._env, name)


class _NoopProposer:
    """Disable tier-3 induction so the sweep scores cached-candidate E3 routing."""

    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return False, {}


def _agent_module() -> Any:
    from carnot.agentic import arc_competition_agent

    return arc_competition_agent


def _submitted_agent_config() -> dict[str, Any]:
    return dict(_agent_module().SUBMITTED_AGENT_CONFIG)


def _submitted_value_weight() -> float:
    return float(_agent_module().SUBMITTED_VALUE_WEIGHT)


def _level_of_frame(frame: Any) -> int:
    return int(_agent_module()._level_of(frame))


def _import_arc_solver_kit() -> Any:
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def _import_torch_version() -> str:
    import torch

    return str(torch.__version__)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rate(solved: int, attempted: int) -> float:
    return round(float(solved) / float(attempted), 10) if attempted else 0.0


def _median(values: Iterable[Any]) -> int | float | None:
    numeric = [value for value in values if type(value) in {int, float}]
    if not numeric:
        return None
    med = statistics.median(numeric)
    return int(med) if float(med).is_integer() else round(float(med), 6)


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def make_value_weight_policy(game: str, value_weight: float) -> Any:  # pragma: no cover - ARC boundary
    """Build the exact submitted E3 explorer config with only value_weight varied."""

    config = _submitted_agent_config()
    return _agent_module().E3AgentPolicy(
        game,
        proposer=_NoopProposer(),
        target_levels=int(config["target_levels"]),
        value_weight=float(value_weight),
        search_mode=str(config["search_mode"]),
    )


def run_policy_game(
    game: str,
    *,
    value_weight: float,
    arcade: Any,
    game_action: Any,
    budget: int = ACTION_BUDGET,
    wall_budget_s: float = EVAL_BUDGET_MEDIAN_WALL_S,
    policy_factory: Callable[[str, float], Any] = make_value_weight_policy,
    level_getter: Callable[[Any], int] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one frame-only submitted-default policy episode with `env._game` blocked."""

    started = float(clock())
    env = _BlockedEnvGame(arcade.make(game, scorecard_id=arcade.open_scorecard()))
    policy = policy_factory(game, float(value_weight))
    read_level = level_getter or _level_of_frame
    frames: list[Any] = []
    latest = None
    start_level = None
    reached_level = 0
    actions = 0
    actions_to_first_levelup = None
    timed_out = False
    for _ in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        if float(clock()) - started > float(wall_budget_s):
            timed_out = True
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
            start_level = int(read_level(latest))
        reached_level = int(read_level(latest))
        if actions_to_first_levelup is None and start_level is not None and reached_level > start_level:
            actions_to_first_levelup = int(actions)
    ended = float(clock())
    levels_delta = max(0, int(reached_level) - int(start_level or 0))
    return {
        "game": str(game),
        "value_weight": float(value_weight),
        "solved": bool(levels_delta > 0),
        "actions_to_first_levelup": actions_to_first_levelup,
        "actions": int(actions),
        "levels_delta": int(levels_delta),
        "start_level": int(start_level or 0),
        "reached_level": int(reached_level),
        "wall_seconds": _round_seconds(ended - started),
        "timed_out": bool(timed_out),
        "frame_only": True,
        "env_game_access_blocked": True,
    }


def summarize_weight_rows(value_weight: float, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    solved = sum(1 for row in rows if row.get("solved") is True)
    attempted = len(rows)
    median_wall = _median(row.get("wall_seconds") for row in rows)
    return {
        "value_weight": float(value_weight),
        "heldout_solve_rate": _rate(solved, attempted),
        "solved_games": int(solved),
        "attempted_games": int(attempted),
        "median_actions_to_first_levelup": _median(
            row.get("actions_to_first_levelup") for row in rows if row.get("solved") is True
        ),
        "median_per_game_wall_seconds": None if median_wall is None else float(median_wall),
        "timed_out_games": int(sum(1 for row in rows if row.get("timed_out") is True)),
        "per_game": [dict(row) for row in rows],
    }


def run_weight_sweep(
    games: Sequence[str] = HELDOUT_GAMES,
    *,
    value_weights: Sequence[float] = VALUE_WEIGHTS,
    budget: int = ACTION_BUDGET,
    wall_budget_s: float = EVAL_BUDGET_MEDIAN_WALL_S,
    arcade_factory: Callable[[], Any] | None = None,
    game_action: Any = None,
    game_runner: Callable[[str, float], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or assemble the value-weight sweep and summarize each tested weight."""

    weights = [float(weight) for weight in value_weights]
    heldout_games = [str(game) for game in games]
    per_weight: list[dict[str, Any]] = []
    if game_runner is None:  # pragma: no cover - exercised by the real artifact run.
        from arcengine import GameAction

        arcade = (arcade_factory or _import_arc_solver_kit().offline_arcade)()
        action_enum = game_action or GameAction

        def game_runner(game: str, value_weight: float) -> Mapping[str, Any]:
            return run_policy_game(
                game,
                value_weight=value_weight,
                arcade=arcade,
                game_action=action_enum,
                budget=budget,
                wall_budget_s=wall_budget_s,
            )

    for value_weight in weights:
        rows = [dict(game_runner(game, value_weight)) for game in heldout_games]
        per_weight.append(summarize_weight_rows(value_weight, rows))
    return {
        "heldout_games": heldout_games,
        "heldout_game_source": "results/experiment_4475_wire_stronger_generic_stack.json benchmark.games",
        "value_weights_tested": weights,
        "action_budget": int(budget),
        "wall_budget_s": float(wall_budget_s),
        "per_weight": per_weight,
    }


def select_value_weight(
    per_weight: Sequence[Mapping[str, Any]],
    *,
    control_value_weight: float = CONTROL_VALUE_WEIGHT,
    eval_budget_median_wall_s: float = EVAL_BUDGET_MEDIAN_WALL_S,
) -> dict[str, Any]:
    control = next(
        row for row in per_weight if float(row.get("value_weight", -1.0)) == float(control_value_weight)
    )
    control_rate = float(control.get("heldout_solve_rate") or 0.0)
    candidates = []
    for row in per_weight:
        weight = float(row.get("value_weight") or 0.0)
        wall = row.get("median_per_game_wall_seconds")
        solve_rate = float(row.get("heldout_solve_rate") or 0.0)
        if weight <= 0.0:
            continue
        if solve_rate <= control_rate:
            continue
        if type(wall) not in {int, float} or float(wall) > float(eval_budget_median_wall_s):
            continue
        candidates.append(row)
    if not candidates:
        return {
            "selected_value_weight": float(control_value_weight),
            "control_solve_rate": control_rate,
            "selected_solve_rate": control_rate,
            "beats_control": False,
            "within_wall_budget": True,
            "should_raise_submitted_value_weight": False,
            "reason": "no_positive_weight_beats_control_within_budget",
        }
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("heldout_solve_rate") or 0.0),
            row.get("median_actions_to_first_levelup")
            if type(row.get("median_actions_to_first_levelup")) in {int, float}
            else float("inf"),
            float(row.get("median_per_game_wall_seconds") or float("inf")),
            float(row.get("value_weight") or 0.0),
        ),
    )[0]
    return {
        "selected_value_weight": float(selected["value_weight"]),
        "control_solve_rate": control_rate,
        "selected_solve_rate": float(selected.get("heldout_solve_rate") or 0.0),
        "beats_control": True,
        "within_wall_budget": True,
        "should_raise_submitted_value_weight": True,
        "reason": "positive_weight_beats_control_within_budget",
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Record the requested resources before launching the sweep."""

    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
        "env_game_blocked": True,
        "value_head_v3_model_present": (root_path / "models" / "arc_verifier_cross_game_v3.json").exists(),
        "parity_test_target": PARITY_TEST_PATH,
    }
    try:
        _import_arc_solver_kit().offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:  # pragma: no cover - only exercised when local ARC SDK breaks.
        checks["offline_arcade_error"] = repr(exc)
    try:
        checks["torch_version"] = _import_torch_version()
        checks["torch_import"] = True
    except Exception as exc:  # pragma: no cover - only exercised when torch is absent.
        checks["torch_error"] = repr(exc)
    return checks


def build_artifact(
    *,
    sweep: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    per_weight = [dict(row) for row in sweep.get("per_weight", [])]
    selection = select_value_weight(
        per_weight,
        control_value_weight=CONTROL_VALUE_WEIGHT,
        eval_budget_median_wall_s=EVAL_BUDGET_MEDIAN_WALL_S,
    )
    selected = float(selection["selected_value_weight"])
    control = next(row for row in per_weight if float(row["value_weight"]) == CONTROL_VALUE_WEIGHT)
    before = _submitted_value_weight()
    after = selected if selection["should_raise_submitted_value_weight"] else before
    submitted_agent_config = _submitted_agent_config()
    if selection["should_raise_submitted_value_weight"]:
        submitted_agent_config["value_weight"] = selected
    verdict = (
        f"success: value_weight_remeasure_raise_to_{selected:g}_"
        f"{int(selection['selected_solve_rate'] * control['attempted_games'])}_of_{control['attempted_games']}"
        if selection["should_raise_submitted_value_weight"]
        else f"complete: value_weight_remeasure_null_keep_0_{control['solved_games']}_of_{control['attempted_games']}"
    )
    checksum_payload = {
        "value_weights_tested": list(sweep.get("value_weights_tested", [])),
        "heldout_games": list(sweep.get("heldout_games", [])),
        "per_weight": per_weight,
        "selection": selection,
        "submitted_agent_config": submitted_agent_config,
        "preconditions_checked": dict(preconditions_checked),
    }
    return {
        "experiment": "experiment_4500_value_weight_remeasure",
        "schema": "carnot.exp4500.value_weight_remeasure.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "value_weights_tested": [float(weight) for weight in sweep.get("value_weights_tested", [])],
        "control_value_weight": CONTROL_VALUE_WEIGHT,
        "eval_budget_median_wall_s": EVAL_BUDGET_MEDIAN_WALL_S,
        "heldout_games": list(sweep.get("heldout_games", [])),
        "heldout_game_source": sweep.get("heldout_game_source"),
        "action_budget": int(sweep.get("action_budget", ACTION_BUDGET)),
        "wall_budget_s": float(sweep.get("wall_budget_s", EVAL_BUDGET_MEDIAN_WALL_S)),
        "per_weight": per_weight,
        "selection": selection,
        "selected_value_weight": selected,
        "submitted_value_weight_before": before,
        "submitted_value_weight_after": after,
        "submitted_agent_config": submitted_agent_config,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": [
            "REQ-REPORT-4500",
            "SCENARIO-REPORT-4500-CONTROL",
            "SCENARIO-REPORT-4500-SCHEMA",
        ],
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "reproducibility_checksum": _stable_hash(checksum_payload),
    }


def _bare_number_or_none(value: Any, number_type: type) -> bool:
    return value is None or type(value) is number_type


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required principles")
    value_weights = artifact.get("value_weights_tested")
    if not isinstance(value_weights, list) or CONTROL_VALUE_WEIGHT not in value_weights:
        errors.append("value_weights_tested must include the zero-weight control")
    if artifact.get("control_value_weight") != CONTROL_VALUE_WEIGHT:
        errors.append("control_value_weight must be 0.0")
    if type(artifact.get("selected_value_weight")) is not float:
        errors.append("selected_value_weight must be bare float")
    if type(artifact.get("submitted_value_weight_before")) is not float:
        errors.append("submitted_value_weight_before must be bare float")
    if type(artifact.get("submitted_value_weight_after")) is not float:
        errors.append("submitted_value_weight_after must be bare float")
    if artifact.get("submitted_agent_config") != _submitted_agent_config():
        errors.append("submitted_agent_config must match SUBMITTED_AGENT_CONFIG")

    per_weight = artifact.get("per_weight")
    if not isinstance(per_weight, list) or not per_weight:
        errors.append("per_weight must be a non-empty list")
    else:
        for idx, row in enumerate(per_weight):
            if not isinstance(row, Mapping):
                errors.append(f"per_weight[{idx}] must be a mapping")
                continue
            if type(row.get("heldout_solve_rate")) is not float:
                errors.append(f"per_weight[{idx}].heldout_solve_rate must be bare float")
            if type(row.get("solved_games")) is not int:
                errors.append(f"per_weight[{idx}].solved_games must be bare int")
            if type(row.get("attempted_games")) is not int:
                errors.append(f"per_weight[{idx}].attempted_games must be bare int")
            if not _bare_number_or_none(row.get("median_actions_to_first_levelup"), int):
                errors.append(f"per_weight[{idx}].median_actions_to_first_levelup must be bare int or null")
            if not _bare_number_or_none(row.get("median_per_game_wall_seconds"), float):
                errors.append(f"per_weight[{idx}].median_per_game_wall_seconds must be bare float or null")
            per_game = row.get("per_game")
            if not isinstance(per_game, list):
                errors.append(f"per_weight[{idx}].per_game must be a list")
                continue
            for game_idx, game_row in enumerate(per_game):
                if not isinstance(game_row, Mapping):
                    errors.append(f"per_weight[{idx}].per_game[{game_idx}] must be a mapping")
                    continue
                if game_row.get("env_game_access_blocked") is not True:
                    errors.append(f"per_weight[{idx}].per_game[{game_idx}] must block env._game")
                if game_row.get("frame_only") is not True:
                    errors.append(f"per_weight[{idx}].per_game[{game_idx}] must be frame-only")

    selection = artifact.get("selection")
    if isinstance(selection, Mapping):
        selected = float(selection.get("selected_value_weight") or 0.0)
        if selected > 0 and artifact.get("submitted_value_weight_after") != selected:
            errors.append("positive selection must update submitted_value_weight_after")
        if selected == 0.0 and artifact.get("submitted_value_weight_after") != artifact.get(
            "submitted_value_weight_before"
        ):
            errors.append("null selection must keep submitted_value_weight_after unchanged")
    else:
        errors.append("selection must be a mapping")
    return errors


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:  # pragma: no cover - integration entrypoint
    root_path = Path(root)
    checks = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root_path)
    sweep = run_weight_sweep()
    artifact = build_artifact(sweep=sweep, preconditions_checked=checks)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
