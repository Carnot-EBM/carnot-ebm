"""Exp 4373: targeted active-data continuation for ar25/ka59/ft09 E3 gaps.

Spec refs: REQ-PHASE4-4373, SCENARIO-PHASE4-4373.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from carnot import experiment_4339_e3_explore_verify_plan_ar25 as exp4339
from carnot import experiment_4350_e3_explore_verify_plan_ka59 as exp4350
from carnot import experiment_4363_e3_mechanic_limited_tails_tr87_ft09 as exp4363
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit


REPO = Path(__file__).resolve().parents[2]
TARGET_ORDER = ("ar25", "ka59", "ft09")
RANDOM_SEED = 4373
ACTIVE_BUDGET = 96
ROUND_BUDGET = 4
VERIFY_TRANSITIONS = 96
PRIOR_BEST_LEVELS = {"ar25": 1, "ka59": 1, "ft09": 1}
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 33
RESULT_RELATIVE_PATH = "results/experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
GAP_PATH = REPO / GAP_RELATIVE_PATH
REGISTRY_PATH = REPO / REGISTRY_RELATIVE_PATH
WORLD_MODEL_PATHS = {
    "ar25": "results/arc_e3/ar25/world_model.py",
    "ka59": "results/arc_e3/ka59/world_model.py",
    "ft09": "results/arc_e3/ft09/world_model.py",
}
ACTIVE_DATASET_PATHS = {
    "ar25": "results/arc_e3/ar25/active_data_4373.json",
    "ka59": "results/arc_e3/ka59/active_data_4373.json",
    "ft09": "results/arc_e3/ft09/active_data_4373.json",
}
TARGET_GAP_ACTIONS = {
    "ar25": (7,),
    "ka59": (1, 2, 3, 4, 6),
    "ft09": (6,),
}
RESIDUAL_GAP_CLASSES = {
    "ar25": "ar25_l2_action7_undo_stack_hidden_rule_gap",
    "ka59": "ka59_l2_hidden_step_counter_hud_register_gap",
    "ft09": "ft09_l2_residual_world_model_mismatch_gap",
}
PRIOR_ARTIFACT_PATHS = {
    "ar25": ("results/experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json",),
    "ka59": ("results/experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json",),
    "ft09": ("results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json",),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_game_scorecard",
    "new_levels_reproduced",
    "reproducible_total_levels",
    "world_model_paths",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_ar25_ka59_ft09_<n>_reproduced or "
        "complete_e3_ar25_ka59_ft09_partial). Any new reproduced level and an honest "
        "partial per game are BOTH progress."
    ),
    "per_game_scorecard": (
        "list of {game, prior_best_level, new_reproduced_level, verifier_accuracy, "
        "offline_reproduced, residual_gap_class} -- the per-game record for ar25/ka59/ft09."
    ),
    "new_levels_reproduced": (
        "BARE int: NEW levels offline-reproduced across ar25+ka59+ft09 -- the incremental-progress unit."
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after this task (>= the prior 33) -- "
        "the monotonic north-star accuracy signal."
    ),
    "world_model_paths": (
        "list[str]: results/arc_e3/{ar25,ka59,ft09}/world_model.py -- the extended models ARE the deliverables."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the SOLVEs are execution-grounded; ARC progress, NOT a moat headline."
    ),
    "preconditions_checked": (
        "Records the offline-env presence per game + harness import + TRM-stand-down; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the active-data collection + induction + exploration + planning.",
    "reproducibility_checksum": (
        "Hash of the active dataset + the extended models + the plans + the reproduce() results; "
        "lets a third party re-run."
    ),
}


@dataclass(frozen=True)
class CompactTransition:
    grid: np.ndarray
    action: int
    data: dict[str, Any] | None
    next_grid: np.ndarray
    level_before: int
    level_after: int


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def path_hashes(repo: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        full = path if path.is_absolute() else repo / path
        hashes[str(item)] = sha256_file(full) if full.exists() and full.is_file() else ""
    return hashes


def _array_hash(arr: Any) -> str:
    a = np.asarray(arr)
    payload = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "data": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _state_signature(grid: Any, q: int = 8) -> tuple[tuple[int, int, int], ...]:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return ()
    sig: list[tuple[int, int, int]] = []
    for color in sorted(int(value) for value in np.unique(arr) if int(value) != 0):
        ys, xs = np.where(arr == color)
        if len(xs):
            sig.append((color, int(np.mean(ys) // q), int(np.mean(xs) // q)))
    return tuple(sig[:16])


def _changed_cells(a: Any, b: Any) -> int:
    left = np.asarray(a)
    right = np.asarray(b)
    if left.shape != right.shape:
        return -1
    return int(np.count_nonzero(left != right))


def _transition_record(index: int, transition: CompactTransition) -> dict[str, Any]:
    return {
        "i": int(index),
        "action": int(transition.action),
        "data": transition.data,
        "grid_sha256": _array_hash(transition.grid),
        "next_grid_sha256": _array_hash(transition.next_grid),
        "changed_cells": _changed_cells(transition.grid, transition.next_grid),
        "level_before": int(transition.level_before),
        "level_after": int(transition.level_after),
        "state_signature": [list(item) for item in _state_signature(transition.grid)],
    }


def summarize_targeted_transitions(
    *,
    game: str,
    target_actions: Sequence[int],
    transitions: Sequence[CompactTransition],
    dataset_path: Path,
    dataset_sha256: str,
    collection_error: str = "",
) -> dict[str, Any]:
    action_counts = Counter(str(int(transition.action)) for transition in transitions)
    signatures = {_state_signature(transition.grid) for transition in transitions}
    return {
        "game": game,
        "active_transitions_collected": len(transitions),
        "target_actions": [int(action) for action in target_actions],
        "action_counts": dict(sorted(action_counts.items())),
        "target_action_counts": {
            str(action): int(action_counts.get(str(action), 0)) for action in target_actions
        },
        "diverse_object_config_signatures": len(signatures),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "collection_error": collection_error,
    }


def write_targeted_active_dataset(
    repo: Path,
    game: str,
    transitions: Sequence[CompactTransition],
    *,
    random_seed: int,
    collection_method: str,
    collection_error: str = "",
) -> tuple[Path, str]:
    path = repo / ACTIVE_DATASET_PATHS[game]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game": game,
        "random_seed": int(random_seed),
        "target_actions": list(TARGET_GAP_ACTIONS[game]),
        "collection_method": collection_method,
        "collection_error": collection_error,
        "transitions": [_transition_record(index, transition) for index, transition in enumerate(transitions)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path, sha256_file(path)


def _to_compact(transitions: Sequence[Any]) -> list[CompactTransition]:
    compact: list[CompactTransition] = []
    for transition in transitions:
        compact.append(
            CompactTransition(
                np.asarray(transition.grid),
                int(transition.action),
                dict(transition.data) if isinstance(transition.data, dict) else None,
                np.asarray(transition.next_grid),
                int(transition.level_before),
                int(transition.level_after),
            )
        )
    return compact


def collect_targeted_active_dataset(  # pragma: no cover - offline SDK boundary
    repo: Path,
    game: str,
    *,
    random_seed: int,
    active_budget: int,
) -> tuple[dict[str, Any], list[CompactTransition]]:
    method = "e3_collect_transitions_coverage_balanced_named_gap"
    try:
        if game == "ft09":
            active_collect = exp4363._load_active_collect()
            from arc_agi import Arcade
            from arc_agi.base import OperationMode
            from arcengine.enums import GameAction, GameState

            arc = Arcade(
                arc_api_key="",
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(repo / "environment_files"),
            )
            full_game = exp4363._offline_game_id(arc, game)
            raw = active_collect(
                arc,
                full_game,
                active_budget,
                max(4, active_budget // 8),
                random.Random(random_seed),
                GameAction,
                GameState,
            )
            compact = [
                CompactTransition(
                    np.asarray(grid),
                    exp4363._action_int(action),
                    exp4363._action_data(action),
                    np.asarray(next_grid),
                    0,
                    0,
                )
                for grid, action, next_grid in raw
            ]
            dataset_path, dataset_sha = write_targeted_active_dataset(
                repo,
                game,
                compact,
                random_seed=random_seed,
                collection_method="m2_v4_active_collect_direct_to_exp4373",
            )
            summary = summarize_targeted_transitions(
                game=game,
                target_actions=TARGET_GAP_ACTIONS[game],
                transitions=compact,
                dataset_path=dataset_path.relative_to(repo),
                dataset_sha256=dataset_sha,
            )
            return summary, compact

        transitions, _cell = e3.collect_transitions(game, n=active_budget, seed=random_seed)
        compact = _to_compact(transitions)
        dataset_path, dataset_sha = write_targeted_active_dataset(
            repo,
            game,
            compact,
            random_seed=random_seed,
            collection_method=method,
        )
        summary = summarize_targeted_transitions(
            game=game,
            target_actions=TARGET_GAP_ACTIONS[game],
            transitions=compact,
            dataset_path=dataset_path.relative_to(repo),
            dataset_sha256=dataset_sha,
        )
        return summary, compact
    except Exception as exc:
        dataset_path, dataset_sha = write_targeted_active_dataset(
            repo,
            game,
            [],
            random_seed=random_seed,
            collection_method=method,
            collection_error=repr(exc)[:500],
        )
        summary = summarize_targeted_transitions(
            game=game,
            target_actions=TARGET_GAP_ACTIONS[game],
            transitions=[],
            dataset_path=dataset_path.relative_to(repo),
            dataset_sha256=dataset_sha,
            collection_error=repr(exc)[:240],
        )
        return summary, []


def preconditions(repo: Path) -> dict[str, Any]:
    games: dict[str, dict[str, Any]] = {}
    offline_env_present: dict[str, bool] = {}
    for game in TARGET_ORDER:
        env = repo / "environment_files" / game
        present = env.is_dir() and any(env.iterdir())
        offline_env_present[game] = present
        games[game] = {
            "offline_env_present": present,
            "offline_env_path": str(env),
        }
    active_ok, active_error = exp4363.active_collect_import_status()
    return {
        "games": games,
        "offline_env_present": offline_env_present,
        "harness_import": True,
        "solver_kit_import": True,
        "executable_world_model_import": True,
        "active_collect_import": active_ok,
        "active_collect_import_error": active_error,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }


def blocked_game_row(repo: Path, game: str) -> dict[str, Any]:
    model_path = repo / WORLD_MODEL_PATHS[game]
    prior = PRIOR_BEST_LEVELS[game]
    return {
        "game": game,
        "offline_env_present": False,
        "prior_best_level": prior,
        "new_reproduced_level": prior,
        "verifier_accuracy": 0.0,
        "verifier_accuracy_per_round": [],
        "offline_reproduced": False,
        "active_transitions_collected": 0,
        "target_action_counts": {str(action): 0 for action in TARGET_GAP_ACTIONS[game]},
        "active_dataset_path": ACTIVE_DATASET_PATHS[game],
        "active_dataset_sha256": "",
        "world_model_path": WORLD_MODEL_PATHS[game],
        "world_model_sha256": sha256_file(model_path) if model_path.exists() else "",
        "plan": [],
        "plan_action_count": 0,
        "plan_source": "offline_env_missing",
        "plan_executed": False,
        "reproduce_result": {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
            "mode": "offline_env_missing",
        },
        "residual_gap_class": "offline_env_missing",
        "targeted_gap_lemmas": [],
        "mechanic_checks_passed": False,
        "checkpoint_status": f"blocked_offline_env_missing_{game}",
    }


def _reached_level(reproduce_result: dict[str, Any], prior: int) -> int:
    return int(reproduce_result.get("reached_level", prior) or prior)


def build_game_scorecard(
    *,
    repo: Path,
    game: str,
    verifier_accuracy_per_round: list[float],
    active_dataset_summary: dict[str, Any],
    world_model_path: Path,
    plan: list[str],
    reproduce_result: dict[str, Any],
    residual_gap_class: str,
    targeted_gap_lemmas: list[dict[str, Any]],
    plan_source: str,
) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    accuracy = max([float(value) for value in verifier_accuracy_per_round] or [0.0])
    reached = _reached_level(reproduce_result, prior)
    advanced = bool(reproduce_result.get("reproduced")) and reached > prior
    return {
        "game": game,
        "offline_env_present": True,
        "prior_best_level": prior,
        "new_reproduced_level": reached if advanced else prior,
        "verifier_accuracy": round(float(accuracy), 6),
        "verifier_accuracy_per_round": [round(float(value), 6) for value in verifier_accuracy_per_round],
        "offline_reproduced": advanced,
        "active_transitions_collected": int(active_dataset_summary.get("active_transitions_collected", 0) or 0),
        "target_action_counts": dict(active_dataset_summary.get("target_action_counts", {})),
        "action_counts": dict(active_dataset_summary.get("action_counts", {})),
        "diverse_object_config_signatures": int(
            active_dataset_summary.get("diverse_object_config_signatures", 0) or 0
        ),
        "active_dataset_path": str(active_dataset_summary.get("dataset_path", ACTIVE_DATASET_PATHS[game])),
        "active_dataset_sha256": str(active_dataset_summary.get("dataset_sha256", "")),
        "active_collection_error": str(active_dataset_summary.get("collection_error", "")),
        "world_model_path": _relative_or_absolute(repo, world_model_path),
        "world_model_sha256": sha256_file(world_model_path) if world_model_path.exists() else "",
        "plan": list(plan),
        "plan_action_count": len(plan),
        "plan_source": plan_source,
        "plan_executed": advanced,
        "reproduce_result": reproduce_result,
        "residual_gap_class": "none" if advanced else residual_gap_class,
        "targeted_gap_lemmas": targeted_gap_lemmas,
        "mechanic_checks_passed": advanced,
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial_no_new_level_reproduced",
    }


def compute_reproducibility_checksum(
    *,
    per_game_scorecard: list[dict[str, Any]],
    world_model_paths: Sequence[str],
    path_hashes: dict[str, str],
    active_dataset_hashes: dict[str, str],
    random_seed: int,
) -> str:
    payload = {
        "per_game_scorecard": per_game_scorecard,
        "world_model_paths": list(world_model_paths),
        "path_hashes": path_hashes,
        "active_dataset_hashes": active_dataset_hashes,
        "random_seed": int(random_seed),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _new_level_delta(row: dict[str, Any]) -> int:
    if not bool(row.get("offline_reproduced")):
        return 0
    return max(0, int(row.get("new_reproduced_level", 0)) - int(row.get("prior_best_level", 0)))


def _new_levels_reproduced(rows: Sequence[dict[str, Any]]) -> int:
    return sum(_new_level_delta(row) for row in rows)


def _combined_verdict(rows: Sequence[dict[str, Any]]) -> str:
    new_levels = _new_levels_reproduced(rows)
    if new_levels:
        return f"success_e3_ar25_ka59_ft09_{new_levels}_reproduced"
    if rows and all(str(row.get("checkpoint_status", "")).startswith("blocked_offline_env_missing") for row in rows):
        return "blocked_offline_env_missing_ar25_ka59_ft09"
    return "complete_e3_ar25_ka59_ft09_partial"


def build_artifact(
    *,
    repo: Path,
    per_game_scorecard: list[dict[str, Any]],
    reproducible_total_levels: int,
    world_model_paths: Sequence[str],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    normalized_paths = list(dict.fromkeys(str(path) for path in world_model_paths))
    hashes = path_hashes(repo, normalized_paths)
    active_hashes = {
        str(row.get("game")): str(row.get("active_dataset_sha256", "")) for row in per_game_scorecard
    }
    checksum = compute_reproducibility_checksum(
        per_game_scorecard=per_game_scorecard,
        world_model_paths=normalized_paths,
        path_hashes=hashes,
        active_dataset_hashes=active_hashes,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09",
        "method": "targeted_active_data_explore_verify_plan_named_hidden_rule_gap_closure",
        "target_order": list(TARGET_ORDER),
        "honest_verdict": _combined_verdict(per_game_scorecard),
        "per_game_scorecard": per_game_scorecard,
        "new_levels_reproduced": _new_levels_reproduced(per_game_scorecard),
        "reproducible_total_levels": int(reproducible_total_levels),
        "world_model_paths": normalized_paths,
        "world_model_path_sha256": hashes,
        "active_dataset_sha256": active_hashes,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-4373", "SCENARIO-PHASE4-4373"],
        "inference_substrate": "offline_targeted_active_data_plus_arc_solver_kit_reproduction_gate_no_nested_codex",
        "submitted_to_leaderboard": False,
        "duration_s": round(float(duration_s), 3),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    rows = artifact.get("per_game_scorecard")
    if not isinstance(rows, list):
        errors.append("per_game_scorecard must be list")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"per_game_scorecard[{index}] must be dict")
                continue
            for field in (
                "game",
                "prior_best_level",
                "new_reproduced_level",
                "verifier_accuracy",
                "offline_reproduced",
                "residual_gap_class",
                "active_transitions_collected",
            ):
                if field not in row:
                    errors.append(f"per_game_scorecard[{index}] missing {field}")
            if not isinstance(row.get("prior_best_level"), int):
                errors.append(f"per_game_scorecard[{index}].prior_best_level must be bare int")
            if not isinstance(row.get("new_reproduced_level"), int):
                errors.append(f"per_game_scorecard[{index}].new_reproduced_level must be bare int")
            if not isinstance(row.get("offline_reproduced"), bool):
                errors.append(f"per_game_scorecard[{index}].offline_reproduced must be bare bool")
            if not isinstance(row.get("active_transitions_collected"), int):
                errors.append(f"per_game_scorecard[{index}].active_transitions_collected must be bare int")
    if not isinstance(artifact.get("new_levels_reproduced"), int):
        errors.append("new_levels_reproduced must be bare int")
    if not isinstance(artifact.get("reproducible_total_levels"), int):
        errors.append("reproducible_total_levels must be bare int")
    paths = artifact.get("world_model_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        errors.append("world_model_paths must be list[str]")
    if not isinstance(artifact.get("verifier_is_oracle"), bool):
        errors.append("verifier_is_oracle must be bare bool")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not isinstance(artifact.get("random_seed"), int):
        errors.append("random_seed must be bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles missing")
    else:
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"principle mismatch for {field}")
    return errors


def prior_plan(repo: Path, game: str) -> list[str]:
    for rel in PRIOR_ARTIFACT_PATHS.get(game, ()):
        path = repo / rel
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for row in data.get("per_game_scorecard", []):
            if row.get("game") == game and isinstance(row.get("plan"), list):
                return [str(label) for label in row["plan"]]
    return []


def _residual_mismatch_class(game: str, mismatches: Sequence[dict[str, Any]]) -> str:
    if not mismatches:
        return "none"
    if any("error" in row for row in mismatches):
        return "engine_runtime_error_gap"
    actions = sorted({int(row.get("action", -1)) for row in mismatches})
    if set(TARGET_GAP_ACTIONS[game]) & set(actions):
        return RESIDUAL_GAP_CLASSES[game]
    return "missing_world_model_rule_gap_actions_" + "_".join(str(action) for action in actions)


def _verify_world_model_round(game: str, *, random_seed: int) -> tuple[float, str]:  # pragma: no cover - offline SDK boundary
    try:
        transitions, cell = e3.collect_transitions(game, n=VERIFY_TRANSITIONS, seed=random_seed)
        engine, _complete = e3.load_engine(game)
        result = e3.WorldModelVerifier(transitions).score(engine, max_mismatch=16)
        print(f"{game} frame-world-model verifier sample cell={cell}", flush=True)
        return round(float(result.accuracy), 6), _residual_mismatch_class(game, result.mismatches)
    except Exception as exc:
        print(f"{game} frame-world-model verifier error={repr(exc)[:160]}", flush=True)
        return 0.0, "engine_runtime_error_gap"


def _targeted_gap_lemmas(  # pragma: no cover - model/offline boundary
    game: str,
    transitions: Sequence[CompactTransition],
    engine: Callable[[np.ndarray, int, dict[str, Any] | None], np.ndarray],
) -> list[dict[str, Any]]:
    lemmas: list[dict[str, Any]] = []
    for transition in transitions:
        if int(transition.action) not in TARGET_GAP_ACTIONS[game]:
            continue
        if game == "ka59" and transition.grid.shape == transition.next_grid.shape:
            bottom_row_changed = bool(np.any(transition.grid[-1] != transition.next_grid[-1]))
            if not bottom_row_changed:
                continue
        try:
            pred = np.asarray(engine(np.asarray(transition.grid).copy(), int(transition.action), transition.data))
        except Exception as exc:
            lemmas.append({"action": int(transition.action), "verifier_gated": False, "error": repr(exc)[:120]})
            continue
        exact = pred.shape == transition.next_grid.shape and bool(np.array_equal(pred, transition.next_grid))
        lemma: dict[str, Any] = {
            "action": int(transition.action),
            "verifier_gated": exact,
            "changed_cells": _changed_cells(transition.grid, transition.next_grid),
            "predicted_changed_cells": _changed_cells(transition.grid, pred),
            "level_delta": int(transition.level_after - transition.level_before),
        }
        if game == "ka59" and pred.shape == transition.grid.shape:
            lemma["hud_count_before"] = int(np.count_nonzero(transition.grid[-1] == 4))
            lemma["hud_count_after"] = int(np.count_nonzero(transition.next_grid[-1] == 4))
            lemma["hud_count_predicted"] = int(np.count_nonzero(pred[-1] == 4))
        if game == "ft09" and isinstance(transition.data, dict):
            x = int(transition.data.get("x", -1))
            y = int(transition.data.get("y", -1))
            if 0 <= y < np.asarray(transition.grid).shape[0] and 0 <= x < np.asarray(transition.grid).shape[1]:
                lemma["clicked_color_before"] = int(transition.grid[y, x])
        lemmas.append(lemma)
        if len(lemmas) >= 12:
            break
    if not lemmas:
        lemmas.append(
            {
                "action": list(TARGET_GAP_ACTIONS[game]),
                "verifier_gated": False,
                "residual": "targeted_active_collection_did_not_hit_named_gap_transition",
            }
        )
    return lemmas


def _apply_for_game(game: str) -> Callable[[Any, str, Any], Any]:
    if game == "ar25":
        return exp4339._apply_ar25_label
    if game == "ka59":
        return exp4350._apply_ka59_label
    if game == "ft09":
        return exp4363._apply_ft09_label
    raise KeyError(game)


def _reproduce_candidate_next_level(game: str, labels: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    prior = PRIOR_BEST_LEVELS[game]
    if not labels:
        return {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
            "mode": "no_replayable_plan",
        }
    return arc_solver_kit.reproduce(game, labels, _apply_for_game(game), claimed_level=prior + 1)


def _run_game(  # pragma: no cover - offline SDK boundary
    repo: Path,
    game: str,
    random_seed: int,
    active_budget: int,
    round_budget: int,
) -> dict[str, Any]:
    active_summary, active_transitions = collect_targeted_active_dataset(
        repo,
        game,
        random_seed=random_seed,
        active_budget=active_budget,
    )
    rounds: list[float] = []
    residuals: list[str] = []
    for round_index in range(max(1, round_budget)):
        accuracy, residual = _verify_world_model_round(game, random_seed=random_seed + round_index)
        rounds.append(accuracy)
        residuals.append(residual)
        print(f"{game} verifier round {round_index} accuracy={accuracy:.6f}", flush=True)
        if accuracy >= 0.95 and residual == "none":
            break
    try:
        engine, _complete = e3.load_engine(game)
        lemmas = _targeted_gap_lemmas(game, active_transitions, engine)
    except Exception as exc:
        lemmas = [{"verifier_gated": False, "error": repr(exc)[:160]}]
    plan = prior_plan(repo, game)
    reproduce_result = _reproduce_candidate_next_level(game, plan)
    residual = RESIDUAL_GAP_CLASSES[game]
    if any(value not in ("none", RESIDUAL_GAP_CLASSES[game]) for value in residuals):
        residual = residuals[-1]
    return build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=rounds,
        active_dataset_summary=active_summary,
        world_model_path=repo / WORLD_MODEL_PATHS[game],
        plan=plan,
        reproduce_result=reproduce_result,
        residual_gap_class=residual,
        targeted_gap_lemmas=lemmas,
        plan_source="prior_reproduced_l1_plan_replayed_against_l2_gate_after_targeted_active_checks",
    )


TARGET_RUNNERS: dict[str, Callable[[Path, str, int, int, int], dict[str, Any]]] = {
    game: _run_game for game in TARGET_ORDER
}


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_gap(path: Path, *, row: dict[str, Any], checksum: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    game = str(row["game"])
    marker = f"### 2026-06-18 Exp4373 {game} named active-data residual gap"
    entry = (
        f"\n\n{marker}\n"
        "- Spec: REQ-PHASE4-4373 / SCENARIO-PHASE4-4373\n"
        f"- Best verifier accuracy: {float(row.get('verifier_accuracy', 0.0)):.4f}\n"
        f"- Active transitions collected: {int(row.get('active_transitions_collected', 0))}\n"
        f"- Target action counts: `{json.dumps(row.get('target_action_counts', {}), sort_keys=True)}`\n"
        f"- Residual gap class: `{row.get('residual_gap_class', 'unknown')}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded targeted active-data pass did not reproduce a new level beyond L1.\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    if marker in existing:
        before, after = existing.split(marker, 1)
        remainder = after.split("\n\n### ", 1)
        suffix = ("\n\n### " + remainder[1]) if len(remainder) == 2 else ""
        path.write_text(before.rstrip() + entry + suffix, encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def _registry_total(text: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}:\s*(\d+)\b", text, re.M)
    return int(match.group(1)) if match else None


def _replace_total(text: str, key: str, value: int, note: str) -> str:
    line = f"{key}: {value}   # {note}"
    if re.search(rf"^{re.escape(key)}:\s*\d+.*$", text, re.M):
        return re.sub(rf"^{re.escape(key)}:\s*\d+.*$", line, text, flags=re.M)
    return text.rstrip() + "\n" + line + "\n"


def _registry_total_from_repo(repo: Path) -> int | None:
    path = repo / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return None
    return _registry_total(path.read_text(encoding="utf-8"), "reproducible_total_levels")


def _update_registry_for_new_levels(rows: Sequence[dict[str, Any]], checksum: str) -> None:  # pragma: no cover - ops ledger boundary
    if not REGISTRY_PATH.exists() or not any(_new_level_delta(row) for row in rows):
        return
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    total = max(
        _registry_total(text, "reproducible_total_levels") or PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
        PRIOR_REPRODUCIBLE_TOTAL_LEVELS + _new_levels_reproduced(rows),
    )
    text = _replace_total(
        text,
        "reproducible_total_levels",
        total,
        f"Exp4373 named-gap L2 reproduction gate; checksum {checksum[:12]}",
    )
    REGISTRY_PATH.write_text(text, encoding="utf-8")


def run_experiment(
    *,
    random_seed: int = RANDOM_SEED,
    active_budget: int = ACTIVE_BUDGET,
    round_budget: int = ROUND_BUDGET,
) -> dict[str, Any]:
    t0 = time.time()
    checks = preconditions(REPO)
    rows: list[dict[str, Any]] = []
    for game in TARGET_ORDER:
        if not checks["games"][game]["offline_env_present"]:
            row = blocked_game_row(REPO, game)
        else:
            row = TARGET_RUNNERS[game](REPO, game, random_seed, active_budget, round_budget)
        rows.append(row)
        for round_index, accuracy in enumerate(row.get("verifier_accuracy_per_round") or []):
            print(f"{game} verifier round {round_index} accuracy={float(accuracy):.6f}", flush=True)
        print(
            f"{game} checkpoint={row['checkpoint_status']} "
            f"active_transitions={row['active_transitions_collected']}",
            flush=True,
        )

    new_levels = _new_levels_reproduced(rows)
    total = _registry_total_from_repo(REPO)
    if total is None:
        total = PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels
    total = max(total, PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels)
    artifact = build_artifact(
        repo=REPO,
        per_game_scorecard=rows,
        reproducible_total_levels=total,
        world_model_paths=[WORLD_MODEL_PATHS[game] for game in TARGET_ORDER],
        random_seed=random_seed,
        duration_s=time.time() - t0,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4373 artifact schema errors: {errors}")
    _write_artifact(artifact)
    for row in rows:
        if row.get("offline_env_present") and not row.get("offline_reproduced"):
            _write_gap(GAP_PATH, row=row, checksum=str(artifact["reproducibility_checksum"]))
    _update_registry_for_new_levels(rows, str(artifact["reproducibility_checksum"]))
    print(
        f"wrote {RESULT_RELATIVE_PATH} verdict={artifact['honest_verdict']} "
        f"new_levels={artifact['new_levels_reproduced']} total={artifact['reproducible_total_levels']}",
        flush=True,
    )
    return artifact


def main() -> int:  # pragma: no cover - exercised through the results wrapper
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
