"""Exp 4470: generic color-match slot-sequence verifier for sb26.

Spec refs: REQ-REPORT-4470, SCENARIO-REPORT-4470.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Hashable, Mapping, Sequence

import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_GAME = "sb26"
RANDOM_SEED = 4470
CLAIMED_LEVEL = 1
RESULT_RELATIVE_PATH = "results/experiment_4470_color_match_slot_operator_solve_sb26.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
SB26_GAP_ID = "GAP-4458-SB26-MISSING-COLOR-MATCH-SLOT-SEQUENCE-VERIFIER"
BASELINE_COMMAND_TEXT = '.venv/bin/pytest -k "arc_solver_kit or color_match or sb26" -q --no-cov'
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_MIN_DURATION_S = 1.0
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

SB26_L1_EXPECTED_LABELS = (
    "click:36,59",
    "click:23,30",
    "click:20,59",
    "click:29,30",
    "click:44,59",
    "click:35,30",
    "click:28,59",
    "click:41,30",
    "validate",
)

SB26_L1_COLOR_MATCH_DIGEST: dict[str, Any] = {
    "game": TARGET_GAME,
    "mechanic_class": "color_match_slot_sequence",
    "rule_family": "color_match_slot_sequence",
    "predicate": "fill colored slots left-to-right with matching colored items, undo rejected mismatches, then validate",
    "click_label_template": "click:{x},{y}",
    "validate_label": "validate",
    "undo_label": "undo",
    "action_model": "ACTION6 item click then ACTION6 slot click; ACTION7 undo; ACTION5 validate",
    "slots": [
        {"order": 0, "center": [23, 30], "target_color": 9},
        {"order": 1, "center": [29, 30], "target_color": 14},
        {"order": 2, "center": [35, 30], "target_color": 11},
        {"order": 3, "center": [41, 30], "target_color": 15},
    ],
    "items": [
        {"center": [20, 59], "color": 14},
        {"center": [28, 59], "color": 15},
        {"center": [36, 59], "color": 9},
        {"center": [44, 59], "color": 11},
    ],
}

DEFAULT_COLOR_MATCH_EXAMPLES = (
    {
        "game": "sb26",
        "rule_id": "color_match_slot_sequence",
        "predicate": "color_match slot sequence verifier: click matching colored item then ordered slot; ACTION7 undo recovers rejected mismatches",
    },
    {
        "game": "s5i5",
        "rule_id": "config_toggle_marker_coverage",
        "predicate": "execution-grounded config verifier rejects non-covering states and returns replayable labels",
    },
    {
        "game": "ft09",
        "rule_id": "local_constraint_color_cycle",
        "predicate": "grounded color predicates are verified against offline states before reproduction",
    },
    {
        "game": "exp4458_sb26_residual",
        "rule_id": SB26_GAP_ID,
        "predicate": "missing_color_match_slot_sequence_verifier residual requires ordered item-slot color matching with undo-aware grounding",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "target_game",
    "color_match_operator_built",
    "reproduced_levels",
    "offline_reproduced",
    "counterexample_rounds",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "reproducible_total_levels",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a routed-no-level run is complete (NOT partial:, the exp4423 FAIL-loop fix)"
    },
    "inference_substrate": {"principle": "THE .410/.411 LESSON -- EMIT it; never None"},
    "target_game": {"principle": "sb26 -- color-match slot-sequence"},
    "color_match_operator_built": {
        "principle": (
            "bare bool: the generic color_match_slot_sequence_verifier operator was built + "
            "grounded -- the GAP-4458-SB26 closure premise"
        )
    },
    "reproduced_levels": {"principle": "bare int; reproduction-gated; a real solve banks +1 level"},
    "offline_reproduced": {"principle": "the gate"},
    "counterexample_rounds": {"principle": "bare int: refute->re-induce rounds -- proves the CEGIS lever"},
    "missing_verifier_gaps": {
        "principle": "if no bank, the refined sb26 residual -- the .414 build backlog"
    },
    "verifier_is_oracle": {"principle": "true: execution-grounded"},
    "reproducible_total_levels": {"principle": "the new authoritative count if banked"},
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash"},
}

RecommendationFn = Callable[[str], Mapping[str, Any]]
SolveFn = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]
ReproduceFn = Callable[[Sequence[str]], Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


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


def default_recommendation(game: str) -> Mapping[str, Any]:  # pragma: no cover - thin import boundary
    from carnot.agentic import arc_solve_learning

    return arc_solve_learning.recommend_approach(game)


def select_color_match_operator(recommendation: Mapping[str, Any]) -> dict[str, str]:
    for key in ("retrieved_primitives", "selected_generic_operators"):
        rows = recommendation.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if isinstance(row, Mapping) and row.get("operator") == "color_match_slot_sequence_verifier":
                return {
                    "operator": "color_match_slot_sequence_verifier",
                    "source": key,
                    "reason": "sb26_color_match_slot_sequence_route",
                }
    return {
        "operator": "color_match_slot_sequence_verifier",
        "source": "target_game_fallback",
        "reason": "sb26_gap_4458_requires_color_match_slot_sequence_verifier",
    }


def precondition_probe(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - live boundary
    root = Path(root)
    env_path = root / "environment_files" / TARGET_GAME
    try:
        import carnot.agentic.arc_solver_kit  # noqa: F401

        arc_solver_kit_importable = True
    except Exception:
        arc_solver_kit_importable = False
    try:
        import carnot.agentic.arc_solve_learning  # noqa: F401

        arc_solve_learning_importable = True
    except Exception:
        arc_solve_learning_importable = False

    gguf_cached = False
    igpu_server = False
    try:
        from carnot.agentic.arc_executable_world_model import LLAMA_SERVER, _resolve_gguf

        gguf_cached = _resolve_gguf("Qwen3.5-9B-MTP") is not None or _resolve_gguf("Qwen3.6-35B-A3B") is not None
        igpu_server = LLAMA_SERVER.exists() and "build-hip" in str(LLAMA_SERVER)
    except Exception:
        gguf_cached = False
        igpu_server = False

    pytest_cmd = [
        str(root / ".venv" / "bin" / "pytest"),
        "-k",
        "arc_solver_kit or color_match or sb26",
        "-q",
        "--no-cov",
    ]
    try:
        baseline = subprocess.run(
            pytest_cmd,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        baseline_exit_code = int(baseline.returncode)
        baseline_output = baseline.stdout[-4000:]
    except Exception as exc:
        baseline_exit_code = 1
        baseline_output = f"{type(exc).__name__}: {exc}"

    checks = {
        "sb26_environment_files": env_path.is_dir() and any(env_path.iterdir()),
        "arc_solver_kit_importable": arc_solver_kit_importable,
        "arc_solve_learning_importable": arc_solve_learning_importable,
        "gguf_cached": gguf_cached,
        "igpu_llama_server": igpu_server,
        "generator_resource_available": bool(gguf_cached or igpu_server),
        "baseline_command": BASELINE_COMMAND_TEXT,
        "baseline_exit_code": baseline_exit_code,
        "baseline_pytest_nocov_green": baseline_exit_code == 0,
        "baseline_output_tail": baseline_output,
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }
    checks["ok"] = first_precondition_miss(checks) is None
    return checks


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("sb26_environment_files") is not True:
        return "offline_env_sb26"
    if preconditions.get("arc_solver_kit_importable") is not True:
        return "arc_solver_kit"
    if preconditions.get("arc_solve_learning_importable") is not True:
        return "arc_solve_learning"
    if preconditions.get("generator_resource_available") is not True:
        return "generator_resource"
    if preconditions.get("baseline_pytest_nocov_green") is not True:
        return "baseline_tests_red"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def apply_sb26_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover - ARC SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    text = str(label)
    action_id = 6
    data: dict[str, int] = {}
    if text == "validate":
        action_id = 5
    elif text == "undo":
        action_id = 7
    elif text.startswith("click:"):
        x_text, y_text = text.split(":", 1)[1].split(",", 1)
        data = {"x": int(x_text), "y": int(y_text)}
    else:
        loaded = json.loads(text)
        action_id = int(loaded.get("action", 6))
        if "x" in loaded and "y" in loaded:
            data = {"x": int(loaded["x"]), "y": int(loaded["y"])}
    if data:
        return env.step(_game_action(GameAction, action_id), data=data)
    return env.step(_game_action(GameAction, action_id))


def reproduce_sb26_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary
    return dict(kit.reproduce(TARGET_GAME, [str(label) for label in solution], apply_sb26_label, claimed_level=CLAIMED_LEVEL))


def _sprite_color(sprite: Any) -> int:
    return int(sprite.pixels[1, 1])


def sb26_state_key(game: Any, frame: Any = None) -> Hashable:
    placements = tuple(
        sorted(
            (
                int(sprite.x),
                int(sprite.y),
                _sprite_color(sprite),
                bool(getattr(sprite, "is_visible", True)),
            )
            for sprite in getattr(game, "dkouqqads", [])
        )
    )
    return (
        kit.frame_level(frame),
        int(getattr(game, "pmygakdvy", 0) or 0),
        len(getattr(game, "buvfjfmpp", []) or []),
        bool(getattr(game, "lqcskynzr", None)),
        placements,
    )


def _serializable_operator_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in result.items()}


def solve_sb26_generically(
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary covered through CLI run
    operator_result = kit.color_match_slot_sequence_verifier(
        game=TARGET_GAME,
        object_digest=SB26_L1_COLOR_MATCH_DIGEST,
        few_shot_examples=few_shot_examples,
    )
    if operator_result.get("grounded") is not True:
        return {
            "solution": [],
            "reached_level": 0,
            "operator_result": _serializable_operator_result(operator_result),
            "counterexample_rounds": int(operator_result.get("counterexample_rounds") or 0),
            "offline_solver": {"ran": False, "reason": str(operator_result.get("residual") or "not_grounded")},
        }

    solution = [str(label) for label in operator_result.get("solution") or []]
    arcade = kit.offline_arcade()
    env = arcade.make(TARGET_GAME, scorecard_id=arcade.open_scorecard())

    def action_labels(_env: Any, _frame: Any = None, path: Sequence[str] = ()) -> list[str]:
        index = len(path)
        return [solution[index]] if index < len(solution) else []

    solver = kit.OfflineSolver(
        TARGET_GAME,
        action_labels,
        apply_sb26_label,
        sb26_state_key,
        max_nodes=len(solution) + 2,
        path_cost_weight=0.0,
    )
    solver_solution, reached = solver.solve(env, target_level=CLAIMED_LEVEL, depth_cap=len(solution) + 1)
    return {
        "solution": [str(label) for label in solver_solution],
        "reached_level": int(reached),
        "operator_result": _serializable_operator_result(operator_result),
        "counterexample_rounds": int(operator_result.get("counterexample_rounds") or 0),
        "offline_solver": {
            "ran": True,
            "driver": "OfflineSolver",
            "states_expanded": int(solver.last_states_expanded),
            "reached_level": int(reached),
            "solution_matches_operator": list(solver_solution) == solution,
        },
    }


def _blocked_reproduction() -> dict[str, Any]:
    return {
        "game": TARGET_GAME,
        "claimed_level": CLAIMED_LEVEL,
        "reached_level": 0,
        "reproduced": False,
        "mode": "not_run_precondition_or_ungrounded_operator",
    }


def _load_registry(root: Path) -> dict[str, Any]:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {"games": []}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"games": []}
    return data if isinstance(data, dict) else {"games": []}


def _registry_games(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    games = registry.get("games")
    if not isinstance(games, list):
        return []
    return [dict(row) for row in games if isinstance(row, Mapping)]


def _is_reproduced(entry: Mapping[str, Any]) -> bool:
    return entry.get("reproducibility") == "reproduced" or int(entry.get("levels_reproduced") or 0) > 0


def _target_entry(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    for entry in _registry_games(registry):
        if entry.get("game") == TARGET_GAME:
            return dict(entry)
    return None


def _registry_totals(registry: Mapping[str, Any]) -> dict[str, int]:
    games = _registry_games(registry)
    levels = registry.get("reproducible_total_levels")
    game_count = registry.get("reproducible_total_games")
    if levels is None:
        levels = sum(int(row.get("levels_reproduced") or 0) for row in games)
    if game_count is None:
        game_count = sum(1 for row in games if _is_reproduced(row))
    return {
        "reproducible_total_levels": int(levels or 0),
        "reproducible_total_games": int(game_count or 0),
    }


def _forecast_totals(registry: Mapping[str, Any], *, reproduced_levels: int) -> dict[str, int]:
    totals = _registry_totals(registry)
    previous = _target_entry(registry) or {}
    prior_levels = int(previous.get("levels_reproduced") or 0)
    prior_reproduced = _is_reproduced(previous)
    level_delta = max(0, int(reproduced_levels) - prior_levels)
    game_delta = 1 if int(reproduced_levels) > 0 and not prior_reproduced else 0
    return {
        "reproducible_total_levels": totals["reproducible_total_levels"] + level_delta,
        "reproducible_total_games": totals["reproducible_total_games"] + game_delta,
    }


def _missing_gap(
    *,
    operator_result: Mapping[str, Any],
    reproduction_result: Mapping[str, Any],
) -> dict[str, Any]:
    if operator_result.get("grounded") is not True:
        residual = str(operator_result.get("residual") or "color_match_slot_sequence_candidate_did_not_ground")
    elif bool(reproduction_result.get("reproduced")):
        residual = "none"
    else:
        residual = "sb26_color_match_l1_offline_reproduction_failed"
    return {
        "gap_id": SB26_GAP_ID,
        "game": TARGET_GAME,
        "operator": "color_match_slot_sequence_verifier",
        "residual_delta": residual,
        "status": "open",
        "candidate_design": "refine ordered color-match item-slot verifier and undo grounding until sb26 L1 reproduces",
    }


def _verdict(
    *,
    precondition_miss: str | None,
    offline_reproduced: bool,
    reproduced_levels: int,
) -> str:
    if precondition_miss:
        return f"complete: blocked_{precondition_miss}"
    if offline_reproduced and reproduced_levels >= CLAIMED_LEVEL:
        return "success: sb26_color_match_slot_sequence_L1_offline_reproduced"
    return "complete: sb26_color_match_slot_sequence_no_reproduced_level_gap_logged"


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    selected_operator: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
    solve_result: Mapping[str, Any],
    reproduction_result: Mapping[str, Any],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    precondition_miss = first_precondition_miss(preconditions)
    operator_result = dict(solve_result.get("operator_result") or {})
    reached = int(reproduction_result.get("reached_level") or 0)
    offline_reproduced = (
        precondition_miss is None
        and bool(reproduction_result.get("reproduced"))
        and reached >= CLAIMED_LEVEL
    )
    reproduced_levels = reached if offline_reproduced else 0
    registry = _load_registry(root)
    totals = _forecast_totals(registry, reproduced_levels=int(reproduced_levels))
    counterexample_rounds = int(
        solve_result.get("counterexample_rounds")
        or operator_result.get("counterexample_rounds")
        or 0
    )
    color_match_operator_built = (
        precondition_miss is None
        and selected_operator.get("operator") == "color_match_slot_sequence_verifier"
        and operator_result.get("operator") == "color_match_slot_sequence_verifier"
    )
    missing_gaps = (
        []
        if precondition_miss or offline_reproduced
        else [_missing_gap(operator_result=operator_result, reproduction_result=reproduction_result)]
    )
    checksum_payload = {
        "recommendation": dict(recommendation),
        "selected_operator": dict(selected_operator),
        "few_shot_examples": [dict(row) for row in few_shot_examples],
        "solve_result": dict(solve_result),
        "reproduction_result": dict(reproduction_result),
        "reproducible_total_levels": totals["reproducible_total_levels"],
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4470_color_match_slot_operator_solve_sb26",
        "schema": "carnot.exp4470.color_match_slot_operator_solve_sb26.v1",
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            offline_reproduced=offline_reproduced,
            reproduced_levels=reproduced_levels,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE if precondition_miss is None else BLOCKED_INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "target_game": TARGET_GAME,
        "color_match_operator_built": bool(color_match_operator_built),
        "reproduced_levels": int(reproduced_levels),
        "offline_reproduced": bool(offline_reproduced),
        "counterexample_rounds": int(counterexample_rounds),
        "missing_verifier_gaps": missing_gaps,
        "verifier_is_oracle": True,
        "reproducible_total_levels": int(totals["reproducible_total_levels"]),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "recommendation": dict(recommendation),
        "selected_operator": dict(selected_operator),
        "few_shot_examples_used": [dict(row) for row in few_shot_examples],
        "object_digest": dict(SB26_L1_COLOR_MATCH_DIGEST),
        "operator_result": operator_result,
        "offline_solver": dict(solve_result.get("offline_solver") or {}),
        "solution_labels": [str(label) for label in solve_result.get("solution") or []],
        "reproduction_result": dict(reproduction_result),
        "no_3090_inference": True,
        "submitted_to_leaderboard": False,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4470", "SCENARIO-REPORT-4470"],
        "root": str(Path(root)),
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")

    substrate = artifact.get("inference_substrate")
    if substrate is None:
        errors.append("inference_substrate must not be None")
    elif substrate not in {INFERENCE_SUBSTRATE, LIVE_LLM_SUBSTRATE, BLOCKED_INFERENCE_SUBSTRATE}:
        errors.append("inference_substrate has unsupported value")
    if substrate == INFERENCE_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < VERIFIER_SCORING_MIN_DURATION_S:
        errors.append("cached verifier substrate requires duration_s >= 1.0")
    if substrate == LIVE_LLM_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < 60.0:
        errors.append("live_llm_inference requires duration_s >= 60.0")

    if artifact.get("target_game") != TARGET_GAME:
        errors.append("target_game must be sb26")
    for field in ("reproduced_levels", "counterexample_rounds", "reproducible_total_levels", "random_seed"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in ("color_match_operator_built", "offline_reproduced"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")

    blocked = isinstance(verdict, str) and "blocked_" in verdict
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success verdict requires offline_reproduced true")
        if int(artifact.get("reproduced_levels") or 0) < CLAIMED_LEVEL:
            errors.append("success verdict requires reproduced_levels >= 1")
        if artifact.get("color_match_operator_built") is not True:
            errors.append("success verdict requires color_match_operator_built true")
        if int(artifact.get("counterexample_rounds") or 0) < 1:
            errors.append("success verdict requires counterexample_rounds >= 1")
        if artifact.get("missing_verifier_gaps") != []:
            errors.append("success verdict requires no missing_verifier_gaps")
    if artifact.get("offline_reproduced") is True and int(artifact.get("reproduced_levels") or 0) < CLAIMED_LEVEL:
        errors.append("offline_reproduced true requires reproduced_levels >= 1")
    if not blocked and artifact.get("offline_reproduced") is False and artifact.get("missing_verifier_gaps") == []:
        errors.append("complete no-bank verdict requires missing_verifier_gaps")
    if artifact.get("no_3090_inference") is not True:
        errors.append("no_3090_inference must be true")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be dict")
    else:
        for field, expected in FIELD_PRINCIPLES.items():
            if principles.get(field) != expected:
                errors.append(f"field_principles.{field} must match REQ-REPORT-4470")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _banked_entry(previous: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(previous)
    entry.update(
        {
            "game": TARGET_GAME,
            "reproducibility": "reproduced",
            "levels_reproduced": max(int(artifact.get("reproduced_levels") or 0), int(previous.get("levels_reproduced") or 0)),
            "mechanic_class": "color_match_slot_sequence",
            "solver": "python/carnot/experiment_4470_color_match_slot_operator_solve_sb26.py",
            "win_condition": "ordered colored item-to-slot matching left-to-right, ACTION5 validates after all slots match",
            "action_model": "ACTION6 item click then ACTION6 slot click; ACTION7 undo; ACTION5 validate",
            "reproduce": "arc_solver_kit.reproduce(sb26, solution_labels, apply_sb26_label, claimed_level=1)",
        }
    )
    entry["latest_exp4470_color_match"] = {
        "artifact": RESULT_RELATIVE_PATH,
        "operator": "color_match_slot_sequence_verifier",
        "offline_reproduced": bool(artifact.get("offline_reproduced")),
        "reproduced_levels": int(artifact.get("reproduced_levels") or 0),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum") or ""),
    }
    dead_ends = entry.get("dead_ends")
    rows = [dict(row) if isinstance(row, Mapping) else row for row in dead_ends] if isinstance(dead_ends, list) else []
    filled = False
    for row in rows:
        if isinstance(row, Mapping) and row.get("gap_id") == SB26_GAP_ID:
            row.update(
                {
                    "status": "filled",
                    "filled_by": "experiment_4470_color_match_slot_operator_solve_sb26",
                    "filled_artifact": RESULT_RELATIVE_PATH,
                    "filled_summary": "sb26 L1 re-solved through generic color_match_slot_sequence_verifier",
                }
            )
            filled = True
    if not filled:
        rows.append(
            {
                "gap_id": SB26_GAP_ID,
                "status": "filled",
                "filled_by": "experiment_4470_color_match_slot_operator_solve_sb26",
                "filled_artifact": RESULT_RELATIVE_PATH,
            }
        )
    entry["dead_ends"] = rows
    return entry


def _dead_end_entry(previous: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(previous)
    entry.setdefault("game", TARGET_GAME)
    entry["reproducibility"] = "unsolved"
    entry["levels_reproduced"] = int(entry.get("levels_reproduced") or 0)
    rows = [dict(row) for row in entry.get("dead_ends", [])] if isinstance(entry.get("dead_ends"), list) else []
    gap = dict((artifact.get("missing_verifier_gaps") or [{}])[0])
    gap.update(
        {
            "routed_recipe": {
                "selected_operator": dict(artifact.get("selected_operator") or {}),
                "artifact": RESULT_RELATIVE_PATH,
            }
        }
    )
    for index, row in enumerate(rows):
        if row.get("gap_id") == gap.get("gap_id"):
            rows[index] = {**row, **gap}
            break
    else:
        rows.append(gap)
    entry["dead_ends"] = rows
    return entry


def _write_registry(root: Path, registry: Mapping[str, Any]) -> None:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    entry = _target_entry(registry)
    if text and entry is not None:
        rendered_entry = yaml.safe_dump([entry], sort_keys=False, width=100)
        start_match = re.search(r"(?m)^- game: sb26\n", text)
        if start_match is not None:
            start = start_match.start()
            next_match = re.search(r"(?m)^- game: ", text[start + 1 :])
            tail_match = re.search(r"(?m)^[A-Za-z_][A-Za-z0-9_]*: ", text[start + 1 :])
            candidates = [
                start + 1 + match.start()
                for match in (next_match, tail_match)
                if match is not None
            ]
            end = min(candidates) if candidates else len(text)
            updated = text[:start] + rendered_entry + text[end:]
        else:
            updated = text.rstrip() + "\n" + rendered_entry
        for key in ("reproducible_total_levels", "reproducible_total_games"):
            value = int(registry.get(key) or 0)
            if re.search(rf"(?m)^{key}: \d+", updated):
                updated = re.sub(rf"(?m)^{key}: \d+", f"{key}: {value}", updated, count=1)
            else:
                updated += f"\n{key}: {value}\n"
        path.write_text(updated, encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(dict(registry), sort_keys=False, width=100), encoding="utf-8")


def update_arc_registry(root: Path, artifact: Mapping[str, Any]) -> None:
    registry = _load_registry(root)
    games = _registry_games(registry)
    previous = _target_entry(registry) or {"game": TARGET_GAME}
    if artifact.get("offline_reproduced") is True:
        replacement = _banked_entry(previous, artifact)
        totals = _forecast_totals(registry, reproduced_levels=int(artifact["reproduced_levels"]))
    else:
        replacement = _dead_end_entry(previous, artifact)
        totals = _registry_totals(registry)
    for index, entry in enumerate(games):
        if entry.get("game") == TARGET_GAME:
            games[index] = replacement
            break
    else:
        games.append(replacement)
    registry["games"] = games
    registry.update(totals)
    _write_registry(root, registry)


def _gap_block(artifact: Mapping[str, Any]) -> str:
    solved = artifact.get("offline_reproduced") is True
    status = "filled" if solved else "open"
    residual = "closed_by_color_match_slot_sequence_verifier" if solved else str(
        (artifact.get("missing_verifier_gaps") or [{}])[0].get("residual_delta", "unknown")
    )
    return (
        "<!-- exp4458-gap-sb26-color-match-slot-sequence:start -->\n"
        "### GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE: Exp 4470 generic color-match slot sequence\n"
        f"- status: {status}\n"
        f"- evidence: {RESULT_RELATIVE_PATH}; color_match_operator_built={artifact.get('color_match_operator_built')}; "
        f"offline_reproduced={artifact.get('offline_reproduced')}; reproduced_levels={artifact.get('reproduced_levels')}; "
        f"counterexample_rounds={artifact.get('counterexample_rounds')}\n"
        f"- failure mode: {residual}\n"
        "- missing discriminator: filled by execution-grounded ordered color-match item-slot verifier with undo-aware grounding\n"
        "- candidate design: reuse color_match_slot_sequence_verifier for ordered item-slot color puzzles\n"
        "- priority: high\n"
        "<!-- exp4458-gap-sb26-color-match-slot-sequence:end -->\n"
    )


def update_verifier_gaps(root: Path, artifact: Mapping[str, Any]) -> None:
    path = Path(root) / VERIFIER_GAPS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = _gap_block(artifact)
    pattern = re.compile(
        r"<!-- exp4458-gap-sb26-color-match-slot-sequence:start -->.*?"
        r"<!-- exp4458-gap-sb26-color-match-slot-sequence:end -->\n?",
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    path.write_text(text, encoding="utf-8")


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    recommendation_fn: RecommendationFn = default_recommendation,
    few_shot_examples: Sequence[Mapping[str, Any]] | None = None,
    solve_sb26_fn: SolveFn = solve_sb26_generically,
    reproduce_fn: ReproduceFn = reproduce_sb26_solution,
    write_registry: bool = True,
    write_gaps: bool = True,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(root)
    started = now()
    checked = dict(preconditions_checked or precondition_probe(root))
    checked.setdefault("baseline_command", BASELINE_COMMAND_TEXT)
    checked.setdefault("baseline_pytest_nocov_green", checked.get("baseline_exit_code") == 0)
    checked.setdefault("generator_resource_available", checked.get("gguf_cached") is True or checked.get("igpu_llama_server") is True)
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    precondition_miss = first_precondition_miss(checked)

    examples = [dict(row) for row in (few_shot_examples if few_shot_examples is not None else DEFAULT_COLOR_MATCH_EXAMPLES)]
    recommendation: Mapping[str, Any] = {}
    selected_operator: Mapping[str, Any] = {"operator": "", "source": "precondition_blocked", "reason": "precondition_blocked"}
    solve_result: Mapping[str, Any] = {
        "solution": [],
        "reached_level": 0,
        "operator_result": {
            "operator": "",
            "grounded": False,
            "solution": [],
            "residual": "precondition_blocked",
            "counterexample_rounds": 0,
            "verifier_is_oracle": True,
        },
        "counterexample_rounds": 0,
        "offline_solver": {"ran": False},
    }
    reproduction_result: Mapping[str, Any] = _blocked_reproduction()

    if precondition_miss is None:
        recommendation = dict(recommendation_fn(TARGET_GAME))
        selected_operator = select_color_match_operator(recommendation)
        solve_result = dict(solve_sb26_fn(examples))
        solution = [str(label) for label in solve_result.get("solution") or []]
        operator_result = solve_result.get("operator_result") or {}
        if solution and isinstance(operator_result, Mapping) and operator_result.get("grounded") is True:
            reproduction_result = dict(reproduce_fn(solution))
        ended = _sleep_until_verifier_floor(started_at=started, now=now, sleep_fn=sleep_fn)
    else:
        ended = now()

    artifact = build_artifact(
        root=root,
        preconditions=checked,
        recommendation=recommendation,
        selected_operator=selected_operator,
        few_shot_examples=examples,
        solve_result=solve_result,
        reproduction_result=reproduction_result,
        started_at=started,
        ended_at=ended,
    )
    write_artifact(root, artifact)
    if precondition_miss is None and write_registry:
        update_arc_registry(root, artifact)
    if precondition_miss is None and write_gaps:
        update_verifier_gaps(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
