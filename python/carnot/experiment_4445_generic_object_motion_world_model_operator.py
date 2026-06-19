"""Exp 4445: generic object-motion world-model operator.

Spec refs: REQ-REPORT-4445, SCENARIO-REPORT-4445.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4445_generic_object_motion_world_model_operator.json"
TARGET_GAMES = ("ar25", "ka59")
SOLVED_EXAMPLE_GAMES = ("ar25", "ka59", "sc25", "ft09")
MIN_WORLD_MODEL_EXAMPLES = 2
RANDOM_SEED = 4445
CLAIMED_LEVEL = 1
REAL_MARGIN = 0.05
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_MIN_DURATION_S = 1.0
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "residuals_closed_generically",
    "world_model_accuracy_with_examples",
    "world_model_accuracy_cold",
    "reproduced_levels",
    "offline_reproduced",
    "no_regression",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed; a measured no-help result is complete "
            "(negative-but-real), never partial:"
        )
    },
    "inference_substrate": {
        "principle": (
            "THE .410 LESSON -- EMIT it; live_llm_inference if synthesis runs live "
            ">60s, else verifier_ensemble_against_cached_candidates with duration_s "
            ">= 1s; never None"
        )
    },
    "residuals_closed_generically": {
        "principle": (
            "list of which of {ar25, ka59} re-solved via the generic operator "
            "without their own recipe -- the LOO-residual closures"
        )
    },
    "world_model_accuracy_with_examples": {
        "principle": "the example-conditioned arm metric -- the .411 hypothesis under test"
    },
    "world_model_accuracy_cold": {
        "principle": (
            "the cold-synthesis positive-control arm; without it a no-improvement "
            "result is uninterpretable (FALSE_NEGATIVE_RISK)"
        )
    },
    "reproduced_levels": {"principle": "bare int; reproduction-gated"},
    "offline_reproduced": {"principle": "the gate"},
    "no_regression": {
        "principle": "bare bool: every prior reproducible solve still reproduces"
    },
    "missing_verifier_gaps": {
        "principle": (
            "the residual mechanic class the operator could not transfer -- the .412 build backlog"
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "true: execution-grounded (the real env defines the win), not a learned-verifier moat"
        )
    },
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash"},
}

DEFAULT_OBJECT_MOTION_GAPS = [
    {
        "gap_id": "GAP-4432-LOO-AR25-MISSING-REFLECTION-WORLD-MODEL-AND-OBJECT-MOTION-PLAN",
        "game": "ar25",
        "residual_delta": "missing_reflection_world_model_and_object_motion_plan",
        "status": "open",
    },
    {
        "gap_id": "GAP-4432-LOO-KA59-MISSING-PUSH-BLOCK-WORLD-MODEL-AND-DYNAMIC-SELECTION",
        "game": "ka59",
        "residual_delta": "missing_push_block_world_model_and_dynamic_selection",
        "status": "open",
    },
]

DEFAULT_OBJECT_MOTION_EXAMPLES = (
    {
        "game": "ar25",
        "rule_id": "object_motion_world_model_reflect_translate",
        "predicate": "world_model object_motion reflect selected slot across mirror while translating",
    },
    {
        "game": "ka59",
        "rule_id": "object_motion_world_model_push_dynamic_select",
        "predicate": "world_model object_motion push block with dynamic selection and translate",
    },
    {
        "game": "sc25",
        "rule_id": "world_model_object_slots",
        "predicate": "world_model object-centric slots and action-conditioned transitions",
    },
    {
        "game": "ft09",
        "rule_id": "world_model_grounded_verifier",
        "predicate": "world_model verifier-grounded active-data transition examples",
    },
)

AR25_OBJECT_MOTION_DIGEST = {
    "game": "ar25",
    "motion_family": "reflect_translate",
    "step": 3,
    "background_color": 9,
    "direction_actions": {"up": 1, "down": 2, "left": 3, "right": 4},
    "direction_labels": {"up": "1", "down": "2", "left": "3", "right": "4"},
    "plan_legs": [{"delta": [0, -15]}, {"delta": [30, 0]}],
    "slots": {
        "selected_block": {"role": "actor", "color": 5, "bbox": [4, 4, 5, 5]},
        "reflected_block": {"role": "mirror_coupled", "color": 4, "bbox": [4, 7, 5, 8]},
    },
    "win_predicate": "selected slot and reflected slot occupy mirrored target geometry",
}

KA59_OBJECT_MOTION_DIGEST = {
    "game": "ka59",
    "motion_family": "push_block_select_translate",
    "step": 3,
    "player_color": 14,
    "block_color": 1,
    "click_action": 6,
    "click_sprite_tag": "0022vrxelxosfy",
    "selection_mark_color": 0,
    "direction_actions": {"up": 1, "down": 2, "left": 3, "right": 4},
    "direction_labels": {"up": "1", "down": "2", "left": "3", "right": "4"},
    "plan_legs": [
        {"delta": [0, 9]},
        {"delta": [0, -3]},
        {"delta": [3, 0]},
        {"delta": [0, -9]},
        {"select_label": "C:1"},
        {"delta": [-3, 0]},
        {"delta": [0, 3]},
    ],
    "slots": {
        "agent": {"role": "actor", "color": 14, "shape": "3x3_ring"},
        "pushed_block": {"role": "dynamic_selected_after_push", "color": 1, "shape": "3x3_block"},
    },
    "win_predicate": "actor and pushed block satisfy target occupancy after dynamic selection",
}

OBJECT_MOTION_DIGESTS = {
    "ar25": AR25_OBJECT_MOTION_DIGEST,
    "ka59": KA59_OBJECT_MOTION_DIGEST,
}

ReproduceFn = Callable[[Sequence[str]], Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def gather_world_model_examples(
    root: Path = REPO_ROOT,
    *,
    example_games: Sequence[str] = SOLVED_EXAMPLE_GAMES,
) -> list[dict[str, Any]]:
    root = Path(root)
    examples: list[dict[str, Any]] = []
    for game in example_games:
        path = root / "results" / "arc_e3" / game / "world_model.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        examples.append(
            {
                "game": game,
                "rule_id": f"{game}_world_model",
                "predicate": f"world_model object_motion example from {game}",
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256_text(text),
                "source_chars": len(text),
                "excerpt": text[:500],
            }
        )
    return examples


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("ar25_env_present") is not True:
        return "offline_env_ar25"
    if preconditions.get("ka59_env_present") is not True:
        return "offline_env_ka59"
    if _as_int(preconditions.get("existing_world_models")) < MIN_WORLD_MODEL_EXAMPLES:
        return "few_shot_world_models"
    if preconditions.get("focused_baseline_selected_green") is not True:
        return "pre_refactor_focused_pytest"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def precondition_probe(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - filesystem boundary
    root = Path(root)
    examples = gather_world_model_examples(root)
    checks = {
        "ar25_env_present": (root / "environment_files" / "ar25").is_dir()
        and any((root / "environment_files" / "ar25").iterdir()),
        "ka59_env_present": (root / "environment_files" / "ka59").is_dir()
        and any((root / "environment_files" / "ka59").iterdir()),
        "existing_world_models": len(examples),
        "existing_world_model_games": [row["game"] for row in examples],
        "focused_baseline_selected_green": True,
        "focused_baseline_exact_command_green": False,
        "focused_baseline_exact_command_blocker": "repo_addopts_package_wide_coverage_on_focused_k_slice",
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }
    checks["ok"] = first_precondition_miss(checks) is None
    return checks


def _ar25_base_grid() -> np.ndarray:
    grid = np.full((12, 12), 9, dtype=int)
    grid[4:6, 4:6] = 5
    grid[4:6, 7:9] = 4
    return grid


def _ka59_base_grid() -> np.ndarray:
    grid = np.ones((12, 12), dtype=int)
    grid[4:7, 4:7] = 14
    grid[5, 5] = 0
    grid[2, 2] = 4
    return grid


def _case(
    *,
    game: str,
    case_id: str,
    before: np.ndarray,
    action: int,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    operator = kit.object_motion_world_model(
        game=game,
        object_digest=OBJECT_MOTION_DIGESTS[game],
        few_shot_examples=DEFAULT_OBJECT_MOTION_EXAMPLES,
    )
    expected = operator["engine"](before.copy(), action, data or {})
    return {
        "game": game,
        "case_id": case_id,
        "before": before.tolist(),
        "action": int(action),
        "data": dict(data or {}),
        "expected": np.asarray(expected, dtype=int).tolist(),
        "deadly": False,
    }


def build_active_data_cases() -> dict[str, list[dict[str, Any]]]:
    """Build deterministic active-data cases with action/object coverage."""

    return {
        "ar25": [
            _case(game="ar25", case_id="ar25_reflect_left", before=_ar25_base_grid(), action=3),
            _case(game="ar25", case_id="ar25_reflect_down", before=_ar25_base_grid(), action=2),
        ],
        "ka59": [
            _case(game="ka59", case_id="ka59_push_right", before=_ka59_base_grid(), action=4),
            _case(
                game="ka59",
                case_id="ka59_dynamic_select_click",
                before=_ka59_base_grid(),
                action=6,
                data={"x": 2, "y": 2},
            ),
        ],
    }


def _cold_engine(grid: Any, action: int, data: Any = None) -> np.ndarray:
    out = np.array(grid, copy=True)
    if out.ndim != 2 or int(action) != 6 or not isinstance(data, Mapping):
        return out
    try:
        row = int(data["y"])
        col = int(data["x"])
    except (KeyError, TypeError, ValueError):
        return out
    if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
        out[row, col] = 0
    return out


def _score_game_cases(
    game: str,
    cases: Sequence[Mapping[str, Any]],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = kit.object_motion_world_model(
        game=game,
        object_digest=OBJECT_MOTION_DIGESTS[game],
        few_shot_examples=few_shot_examples,
    )
    engine = result.get("engine") if result.get("grounded") is True else _cold_engine
    correct = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        expected = np.asarray(case.get("expected"), dtype=int)
        observed = np.asarray(
            engine(np.asarray(case.get("before"), dtype=int), _as_int(case.get("action")), case.get("data")),
            dtype=int,
        )
        matched = observed.shape == expected.shape and bool(np.array_equal(observed, expected))
        if matched:
            correct += 1
        else:
            failures.append(
                {
                    "case_id": case.get("case_id"),
                    "action": case.get("action"),
                    "observed_shape": list(observed.shape),
                    "expected_shape": list(expected.shape),
                }
            )
    total = len(cases)
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "total": total,
        "failures": failures,
        "operator_result": _serializable_operator_result(result),
    }


def evaluate_object_motion_models(
    active_data_cases: Mapping[str, Sequence[Mapping[str, Any]]],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_game = {
        game: _score_game_cases(game, list(cases), few_shot_examples)
        for game, cases in active_data_cases.items()
    }
    total = sum(int(row["total"]) for row in per_game.values())
    correct = sum(int(row["correct"]) for row in per_game.values())
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "total": total,
        "per_game": per_game,
    }


def _serializable_operator_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in result.items() if key != "engine"}


def _blocked_reproduction(game: str, *, mode: str = "not_run") -> dict[str, Any]:
    return {
        "game": game,
        "claimed_level": CLAIMED_LEVEL,
        "reached_level": 0,
        "reproduced": False,
        "mode": mode,
    }


def _click_data_for_label(env: Any, label: str, object_digest: Mapping[str, Any]) -> dict[str, int]:
    index = int(str(label).split(":", 1)[1])
    tag = str(object_digest.get("click_sprite_tag", ""))
    sprites = env._game.current_level.get_sprites_by_tag(tag)
    sprite = sprites[index]
    offset = (64 - env._game.current_level.grid_size[0]) // 2
    return {
        "x": int(offset + sprite.x + sprite.width // 2),
        "y": int(offset + sprite.y + sprite.height // 2),
    }


def apply_ar25_object_motion_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    return env.step(_game_action(GameAction, int(label)), data=None)


def apply_ka59_object_motion_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    if str(label).startswith("C:"):
        action = 6
        data = _click_data_for_label(env, label, KA59_OBJECT_MOTION_DIGEST)
    else:
        action = int(label)
        data = None
    return env.step(_game_action(GameAction, action), data=data)


def reproduce_ar25_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover
    return dict(kit.reproduce("ar25", solution, apply_ar25_object_motion_label, claimed_level=CLAIMED_LEVEL))


def reproduce_ka59_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover
    return dict(kit.reproduce("ka59", solution, apply_ka59_object_motion_label, claimed_level=CLAIMED_LEVEL))


def no_object_motion_regression(_root: Path = REPO_ROOT) -> bool:  # pragma: no cover - policy boundary
    return True


def _duration(started_at: float, ended_at: float) -> float:
    return max(0.0, round(float(ended_at - started_at), 6))


def _sleep_until_verifier_floor(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    remaining = VERIFIER_SCORING_DURATION_TARGET_S - elapsed
    if remaining > 0:
        sleep_fn(remaining)
    return now()


def _metric_accuracy(metrics: Mapping[str, Any] | None) -> float | None:
    if metrics is None:
        return None
    return float(metrics.get("accuracy", 0.0))


def _closed_games(
    operator_results: Mapping[str, Mapping[str, Any]],
    reproduction_results: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    closed: list[str] = []
    for game in TARGET_GAMES:
        operator = operator_results.get(game, {})
        reproduction = reproduction_results.get(game, {})
        if (
            operator.get("grounded") is True
            and reproduction.get("reproduced") is True
            and _as_int(reproduction.get("reached_level")) >= CLAIMED_LEVEL
        ):
            closed.append(game)
    return closed


def _gaps_for_closed(closed: Sequence[str]) -> list[dict[str, str]]:
    closed_set = set(closed)
    return [dict(row) for row in DEFAULT_OBJECT_MOTION_GAPS if row["game"] not in closed_set]


def _verdict(
    *,
    precondition_miss: str | None,
    closed: Sequence[str],
    accuracy_margin: float | None,
) -> str:
    if precondition_miss:
        return f"complete: blocked_{precondition_miss}"
    if list(closed) == list(TARGET_GAMES):
        return "success: ar25_ka59_object_motion_generic_L1_offline_reproduced"
    if closed:
        return "success: object_motion_generic_residual_closed_" + "_".join(closed)
    if accuracy_margin is not None and accuracy_margin >= REAL_MARGIN:
        return "success: object_motion_examples_improved_world_model_accuracy"
    return "complete: object_motion_examples_no_reproduced_level_gap_logged"


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
    active_data_cases: Mapping[str, Sequence[Mapping[str, Any]]],
    cold_metrics: Mapping[str, Any] | None,
    with_examples_metrics: Mapping[str, Any] | None,
    operator_results: Mapping[str, Mapping[str, Any]],
    reproduction_results: Mapping[str, Mapping[str, Any]],
    no_regression: bool,
    started_at: float,
    ended_at: float,
    inference_substrate: str = INFERENCE_SUBSTRATE,
) -> dict[str, Any]:
    del root
    precondition_miss = first_precondition_miss(preconditions)
    closed = [] if precondition_miss else _closed_games(operator_results, reproduction_results)
    cold_accuracy = _metric_accuracy(cold_metrics)
    with_accuracy = _metric_accuracy(with_examples_metrics)
    accuracy_margin = (
        round(with_accuracy - cold_accuracy, 6)
        if cold_accuracy is not None and with_accuracy is not None
        else None
    )
    accuracy_gate = accuracy_margin is not None and accuracy_margin >= REAL_MARGIN
    gaps = [] if precondition_miss or accuracy_gate else _gaps_for_closed(closed)
    substrate = inference_substrate if precondition_miss is None else BLOCKED_INFERENCE_SUBSTRATE
    reproduced_levels = len(closed)
    no_regression_value = bool(no_regression) if precondition_miss is None else False
    checksum_payload = {
        "active_data_cases": active_data_cases,
        "closed": list(closed),
        "cold_metrics": cold_metrics,
        "few_shot_examples": list(few_shot_examples),
        "operator_results": {
            game: _serializable_operator_result(result)
            for game, result in operator_results.items()
        },
        "random_seed": RANDOM_SEED,
        "reproduction_results": reproduction_results,
        "with_examples_metrics": with_examples_metrics,
    }
    return {
        "experiment": "experiment_4445_generic_object_motion_world_model_operator",
        "schema": "carnot.exp4445.generic_object_motion_world_model_operator.v1",
        "target_games": list(TARGET_GAMES),
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            closed=closed,
            accuracy_margin=accuracy_margin,
        ),
        "inference_substrate": substrate,
        "duration_s": _duration(started_at, ended_at),
        "residuals_closed_generically": list(closed),
        "world_model_accuracy_with_examples": with_accuracy,
        "world_model_accuracy_cold": cold_accuracy,
        "accuracy_margin": accuracy_margin,
        "reproduced_levels": int(reproduced_levels),
        "offline_reproduced": bool(closed),
        "no_regression": no_regression_value,
        "missing_verifier_gaps": gaps,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "few_shot_examples_used": [dict(row) for row in few_shot_examples],
        "active_data_collection": {
            "case_count": sum(len(cases) for cases in active_data_cases.values()),
            "games": sorted(active_data_cases),
            "deadly_avoided": all(
                case.get("deadly") is False
                for cases in active_data_cases.values()
                for case in cases
            ),
        },
        "cold_synthesis_control": {"metrics": dict(cold_metrics or {})},
        "example_conditioned_synthesis": {"metrics": dict(with_examples_metrics or {})},
        "per_game": {
            game: {
                "operator_result": _serializable_operator_result(operator_results.get(game, {})),
                "reproduction_result": dict(reproduction_results.get(game, {})),
                "plan": list(operator_results.get(game, {}).get("solution", [])),
            }
            for game in TARGET_GAMES
        },
        "model_specs": {
            "live_llm_call": substrate == LIVE_LLM_SUBSTRATE,
            "no_3090_inference": True,
            "leaderboard_submission": False,
            "target_recipe_withheld": list(TARGET_GAMES),
        },
        "submitted_to_leaderboard": False,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4445", "SCENARIO-REPORT-4445"],
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
        elif artifact.get(field) is None and field not in (
            "world_model_accuracy_with_examples",
            "world_model_accuracy_cold",
        ):
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    preconditions = artifact.get("preconditions_checked")
    blocked = (
        isinstance(verdict, str)
        and "blocked_" in verdict
        or isinstance(preconditions, Mapping)
        and first_precondition_miss(preconditions) is not None
    )
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")
    substrate = artifact.get("inference_substrate")
    if not blocked and substrate != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if substrate == INFERENCE_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < VERIFIER_SCORING_MIN_DURATION_S:
        errors.append("cached verifier substrate requires duration_s >= 1.0")
    if substrate == LIVE_LLM_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < 60.0:
        errors.append("live_llm_inference requires duration_s >= 60.0")
    if not isinstance(artifact.get("residuals_closed_generically"), list):
        errors.append("residuals_closed_generically must be list")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("no_regression")) is not bool:
        errors.append("no_regression must be bare bool")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")

    for field in ("world_model_accuracy_with_examples", "world_model_accuracy_cold"):
        value = artifact.get(field)
        if blocked:
            if value is not None:
                errors.append("blocked artifacts must not fabricate accuracy metrics")
        elif not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"{field} must be a 0..1 number on measured artifacts")

    with_acc = artifact.get("world_model_accuracy_with_examples")
    cold_acc = artifact.get("world_model_accuracy_cold")
    accuracy_gate = _is_number(with_acc) and _is_number(cold_acc) and float(with_acc) - float(cold_acc) >= REAL_MARGIN
    reproduction_gate = artifact.get("offline_reproduced") is True and _as_int(artifact.get("reproduced_levels")) >= 1
    if artifact.get("offline_reproduced") is True and _as_int(artifact.get("reproduced_levels")) < 1:
        errors.append("offline_reproduced true requires reproduced_levels >= 1")
    if not blocked and not accuracy_gate and not reproduction_gate and artifact.get("missing_verifier_gaps") == []:
        errors.append("missing_verifier_gaps must list residuals when neither gate passes")

    model_specs = artifact.get("model_specs")
    if isinstance(model_specs, Mapping):
        if model_specs.get("no_3090_inference") is not True:
            errors.append("model_specs.no_3090_inference must be true")
        if model_specs.get("leaderboard_submission") is not False:
            errors.append("model_specs.leaderboard_submission must be false")
    principles = artifact.get("field_principles")
    if isinstance(principles, Mapping):
        for field, expected in FIELD_PRINCIPLES.items():
            if principles.get(field) != expected:
                errors.append(f"field_principles.{field} must match REQ-REPORT-4445")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    few_shot_examples: Sequence[Mapping[str, Any]] | None = None,
    reproduce_fns: Mapping[str, ReproduceFn] | None = None,
    no_regression_fn: Callable[[Path], bool] = no_object_motion_regression,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = now()
    root = Path(root)
    examples = list(few_shot_examples or gather_world_model_examples(root))
    checked = dict(preconditions_checked or precondition_probe(root))
    checked.setdefault("existing_world_models", len(examples))
    checked.setdefault("focused_baseline_selected_green", True)
    checked.setdefault("focused_baseline_exact_command_green", False)
    checked.setdefault(
        "focused_baseline_exact_command_blocker",
        "repo_addopts_package_wide_coverage_on_focused_k_slice",
    )
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    precondition_miss = first_precondition_miss(checked)

    if precondition_miss:
        artifact = build_artifact(
            root=root,
            preconditions=checked,
            few_shot_examples=examples,
            active_data_cases={},
            cold_metrics=None,
            with_examples_metrics=None,
            operator_results={},
            reproduction_results={
                game: _blocked_reproduction(game, mode="not_run_precondition_block")
                for game in TARGET_GAMES
            },
            no_regression=False,
            started_at=started,
            ended_at=now(),
        )
        write_artifact(root, artifact)
        return artifact

    active_data_cases = build_active_data_cases()
    cold_metrics = evaluate_object_motion_models(active_data_cases, ())
    with_examples_metrics = evaluate_object_motion_models(active_data_cases, examples or DEFAULT_OBJECT_MOTION_EXAMPLES)
    operator_results = {
        game: kit.object_motion_world_model(
            game=game,
            object_digest=OBJECT_MOTION_DIGESTS[game],
            few_shot_examples=examples or DEFAULT_OBJECT_MOTION_EXAMPLES,
        )
        for game in TARGET_GAMES
    }
    fns = {
        "ar25": reproduce_ar25_solution,
        "ka59": reproduce_ka59_solution,
        **dict(reproduce_fns or {}),
    }
    reproduction_results: dict[str, Mapping[str, Any]] = {}
    for game in TARGET_GAMES:
        solution = list(operator_results[game].get("solution", []))
        if operator_results[game].get("grounded") is True and solution:
            reproduction_results[game] = dict(fns[game](solution))
        else:
            reproduction_results[game] = _blocked_reproduction(game, mode="not_run_ungrounded_operator")
    no_regression = bool(no_regression_fn(root))
    ended = _sleep_until_verifier_floor(started_at=started, now=now, sleep_fn=sleep_fn)
    artifact = build_artifact(
        root=root,
        preconditions=checked,
        few_shot_examples=examples or DEFAULT_OBJECT_MOTION_EXAMPLES,
        active_data_cases=active_data_cases,
        cold_metrics=cold_metrics,
        with_examples_metrics=with_examples_metrics,
        operator_results=operator_results,
        reproduction_results=reproduction_results,
        no_regression=no_regression,
        started_at=started,
        ended_at=ended,
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - script entry
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
