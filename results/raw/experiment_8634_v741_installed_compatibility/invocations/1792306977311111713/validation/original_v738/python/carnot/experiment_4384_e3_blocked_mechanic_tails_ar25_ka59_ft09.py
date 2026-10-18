"""Exp 4384: targeted active-data plus Mind-Studio K-step lookahead for E3 tails.

Spec refs: REQ-PHASE4-4384, SCENARIO-PHASE4-4384.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing as mp
import queue
import random
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from carnot import experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09 as base
from carnot.agentic import arc_executable_world_model as e3


REPO = Path(__file__).resolve().parents[2]
TARGET_ORDER = ("ar25", "ka59", "ft09")
RANDOM_SEED = 4384
ACTIVE_BUDGET = 96
ROUND_BUDGET = 4
LOOKAHEAD_K = 3
LOOKAHEAD_THRESHOLD = 0.95
VERIFY_THRESHOLD = 0.95
DEFAULT_GAME_WALL_TIME_S = 30.0
PRIOR_BEST_LEVELS = {"ar25": 1, "ka59": 1, "ft09": 1}
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 34
RESULT_RELATIVE_PATH = "results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
GAP_PATH = REPO / GAP_RELATIVE_PATH
REGISTRY_PATH = REPO / REGISTRY_RELATIVE_PATH
WORLD_MODEL_PATHS = dict(base.WORLD_MODEL_PATHS)
ACTIVE_DATASET_PATHS = {
    "ar25": "results/arc_e3/ar25/active_data_4384.json",
    "ka59": "results/arc_e3/ka59/active_data_4384.json",
    "ft09": "results/arc_e3/ft09/active_data_4384.json",
}
TARGET_GAP_ACTIONS = dict(base.TARGET_GAP_ACTIONS)
RESIDUAL_GAP_CLASSES = dict(base.RESIDUAL_GAP_CLASSES)
CompactTransition = base.CompactTransition

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
        "lookahead_fidelity, offline_reproduced, residual_gap_class} -- the per-game "
        "record for ar25/ka59/ft09."
    ),
    "new_levels_reproduced": (
        "BARE int: NEW levels offline-reproduced across ar25+ka59+ft09 -- the incremental-progress unit."
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after this task (>= the prior 34) -- "
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
    "random_seed": "Determinism precondition for the active-data collection + lookahead-fidelity + induction + planning.",
    "reproducibility_checksum": (
        "Hash of the active dataset + the extended models + the plans + the reproduce() results; "
        "lets a third party re-run."
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_hashes(repo: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        full = path if path.is_absolute() else repo / path
        hashes[str(item)] = sha256_file(full) if full.exists() and full.is_file() else ""
    return hashes


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


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
        "transitions": [
            base._transition_record(index, transition)
            for index, transition in enumerate(transitions)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path, sha256_file(path)


def collect_targeted_active_dataset(  # pragma: no cover - offline SDK boundary
    repo: Path,
    game: str,
    *,
    random_seed: int,
    active_budget: int,
) -> tuple[dict[str, Any], list[CompactTransition]]:
    method = "e3_collect_transitions_coverage_balanced_named_gap_mind_studio_4384"
    try:
        if game == "ft09":
            active_collect = base.exp4363._load_active_collect()
            from arc_agi import Arcade
            from arc_agi.base import OperationMode
            from arcengine.enums import GameAction, GameState

            arc = Arcade(
                arc_api_key="",
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(repo / "environment_files"),
            )
            full_game = base.exp4363._offline_game_id(arc, game)
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
                    base.exp4363._action_int(action),
                    base.exp4363._action_data(action),
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
                collection_method="m2_v4_active_collect_direct_to_exp4384",
            )
        else:
            transitions, _cell = e3.collect_transitions(game, n=active_budget, seed=random_seed)
            compact = base._to_compact(transitions)
            dataset_path, dataset_sha = write_targeted_active_dataset(
                repo,
                game,
                compact,
                random_seed=random_seed,
                collection_method=method,
            )
        summary = base.summarize_targeted_transitions(
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
        summary = base.summarize_targeted_transitions(
            game=game,
            target_actions=TARGET_GAP_ACTIONS[game],
            transitions=[],
            dataset_path=dataset_path.relative_to(repo),
            dataset_sha256=dataset_sha,
            collection_error=repr(exc)[:240],
        )
        return summary, []


def _named_register(game: str, grid: Any) -> np.ndarray:
    arr = np.asarray(grid)
    if game == "ka59" and arr.ndim == 2 and arr.shape[0] > 0:
        return arr[-1:, :].copy()
    return arr.copy()


def compute_k_step_lookahead_fidelity(
    *,
    game: str,
    transitions: Sequence[CompactTransition],
    engine: Callable[[np.ndarray, int, dict[str, Any] | None], np.ndarray],
    k: int,
) -> float:
    if k <= 0 or not transitions:
        return 0.0
    total = 0
    correct = 0
    for start in range(len(transitions)):
        current = np.asarray(transitions[start].grid).copy()
        for offset in range(k):
            index = start + offset
            if index >= len(transitions):
                break
            transition = transitions[index]
            if offset and not np.array_equal(
                np.asarray(transitions[index - 1].next_grid),
                np.asarray(transition.grid),
            ):
                break
            total += 1
            try:
                predicted = np.asarray(engine(current.copy(), int(transition.action), transition.data))
            except Exception:
                break
            actual_register = _named_register(game, transition.next_grid)
            predicted_register = _named_register(game, predicted)
            if predicted_register.shape == actual_register.shape and np.array_equal(
                predicted_register,
                actual_register,
            ):
                correct += 1
            current = predicted
    return round(correct / total, 6) if total else 0.0


def write_skill_file(
    repo: Path,
    game: str,
    *,
    random_seed: int,
    lookahead_k: int,
) -> str:
    rel = Path("results") / "arc_e3" / game / "skill_4384.json"
    payload = {
        "game": game,
        "spec_refs": ["REQ-PHASE4-4384", "SCENARIO-PHASE4-4384"],
        "mind_studio_source": "arXiv:2606.16070",
        "method": "targeted_active_data_plus_named_register_k_step_lookahead_fidelity",
        "prior_best_level": PRIOR_BEST_LEVELS[game],
        "target_level": PRIOR_BEST_LEVELS[game] + 1,
        "lookahead_k": int(lookahead_k),
        "random_seed": int(random_seed),
        "active_data_target_actions": list(TARGET_GAP_ACTIONS[game]),
        "entropy_selected_traces": [f"{game}:named_gap_active_rollout:L1->L2"],
        "target_named_gap": RESIDUAL_GAP_CLASSES[game],
        "lookahead_fidelity_target": "named_register_k_step_rollout_matches_env_before_planning",
        "verifier_is_oracle": True,
    }
    full = repo / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rel.as_posix()


def _reached_level(reproduce_result: dict[str, Any], prior: int) -> int:
    return int(reproduce_result.get("reached_level", prior) or prior)


def build_game_scorecard(
    *,
    repo: Path,
    game: str,
    verifier_accuracy_per_round: list[float],
    lookahead_fidelity_per_round: list[float],
    active_dataset_summary: dict[str, Any],
    world_model_path: Path,
    skill_file_path: str,
    plan: list[str],
    reproduce_result: dict[str, Any],
    residual_gap_class: str,
    targeted_gap_lemmas: list[dict[str, Any]],
    plan_source: str,
    mechanic_checks_passed: bool,
) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    accuracy = max([float(value) for value in verifier_accuracy_per_round] or [0.0])
    fidelity = max([float(value) for value in lookahead_fidelity_per_round] or [0.0])
    reached = _reached_level(reproduce_result, prior)
    advanced = bool(reproduce_result.get("reproduced")) and reached > prior
    return {
        "game": game,
        "offline_env_present": True,
        "prior_best_level": prior,
        "new_reproduced_level": reached if advanced else prior,
        "verifier_accuracy": round(float(accuracy), 6),
        "verifier_accuracy_per_round": [round(float(value), 6) for value in verifier_accuracy_per_round],
        "lookahead_fidelity": round(float(fidelity), 6),
        "lookahead_fidelity_per_round": [round(float(value), 6) for value in lookahead_fidelity_per_round],
        "lookahead_k": LOOKAHEAD_K,
        "lookahead_status": "accepted_for_planning" if fidelity >= LOOKAHEAD_THRESHOLD else "honest_partial_prefix_only",
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
        "mind_studio_skill_file": skill_file_path,
        "mind_studio_skill_sha256": sha256_file(repo / skill_file_path) if (repo / skill_file_path).exists() else "",
        "plan": list(plan),
        "plan_action_count": len(plan),
        "plan_source": plan_source,
        "plan_executed": advanced,
        "reproduce_result": reproduce_result,
        "residual_gap_class": "none" if advanced else residual_gap_class,
        "targeted_gap_lemmas": targeted_gap_lemmas,
        "mechanic_checks_passed": bool(mechanic_checks_passed and fidelity >= LOOKAHEAD_THRESHOLD),
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial_no_new_level_reproduced",
    }


def blocked_game_row(repo: Path, game: str, *, skill_file_path: str | None = None) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    model_path = repo / WORLD_MODEL_PATHS[game]
    skill = skill_file_path or f"results/arc_e3/{game}/skill_4384.json"
    return {
        "game": game,
        "offline_env_present": False,
        "prior_best_level": prior,
        "new_reproduced_level": prior,
        "verifier_accuracy": 0.0,
        "verifier_accuracy_per_round": [],
        "lookahead_fidelity": 0.0,
        "lookahead_fidelity_per_round": [],
        "lookahead_k": LOOKAHEAD_K,
        "lookahead_status": "offline_env_missing",
        "offline_reproduced": False,
        "active_transitions_collected": 0,
        "target_action_counts": {str(action): 0 for action in TARGET_GAP_ACTIONS[game]},
        "active_dataset_path": ACTIVE_DATASET_PATHS[game],
        "active_dataset_sha256": "",
        "world_model_path": WORLD_MODEL_PATHS[game],
        "world_model_sha256": sha256_file(model_path) if model_path.exists() else "",
        "mind_studio_skill_file": skill,
        "mind_studio_skill_sha256": sha256_file(repo / skill) if (repo / skill).exists() else "",
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


def timeout_game_row(game: str, target_wall_time_s: float) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    return {
        "game": game,
        "offline_env_present": True,
        "prior_best_level": prior,
        "new_reproduced_level": prior,
        "verifier_accuracy": 0.0,
        "verifier_accuracy_per_round": [0.0],
        "lookahead_fidelity": 0.0,
        "lookahead_fidelity_per_round": [0.0],
        "lookahead_k": LOOKAHEAD_K,
        "lookahead_status": "wall_time_cap_exhausted",
        "offline_reproduced": False,
        "active_transitions_collected": 0,
        "target_action_counts": {str(action): 0 for action in TARGET_GAP_ACTIONS[game]},
        "active_dataset_path": ACTIVE_DATASET_PATHS[game],
        "active_dataset_sha256": "",
        "world_model_path": WORLD_MODEL_PATHS[game],
        "world_model_sha256": "",
        "mind_studio_skill_file": f"results/arc_e3/{game}/skill_4384.json",
        "mind_studio_skill_sha256": "",
        "plan": [],
        "plan_action_count": 0,
        "plan_source": "wall_time_cap_exhausted",
        "plan_executed": False,
        "reproduce_result": {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
            "timeout_s": target_wall_time_s,
        },
        "residual_gap_class": "wall_time_cap_exhausted",
        "targeted_gap_lemmas": [],
        "mechanic_checks_passed": False,
        "checkpoint_status": "honest_partial_wall_time_cap_exhausted",
    }


def exception_game_row(game: str, exc: str) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    return {
        **timeout_game_row(game, 0.0),
        "plan_source": "target_runner_exception",
        "reproduce_result": {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
            "exception": exc.splitlines()[-1] if exc else "unknown_exception",
        },
        "residual_gap_class": "target_runner_exception",
        "checkpoint_status": "honest_partial_target_exception",
        "exception_traceback_tail": exc.splitlines()[-8:],
    }


def compute_reproducibility_checksum(
    *,
    per_game_scorecard: list[dict[str, Any]],
    world_model_paths: Sequence[str],
    skill_file_paths: Sequence[str],
    path_hashes: dict[str, str],
    active_dataset_hashes: dict[str, str],
    random_seed: int,
    game_wall_time_s: float | None,
    lookahead_k: int,
) -> str:
    payload = {
        "per_game_scorecard": per_game_scorecard,
        "world_model_paths": list(world_model_paths),
        "skill_file_paths": list(skill_file_paths),
        "path_hashes": path_hashes,
        "active_dataset_hashes": active_dataset_hashes,
        "random_seed": int(random_seed),
        "game_wall_time_s": game_wall_time_s,
        "lookahead_k": int(lookahead_k),
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
    skill_file_paths: Sequence[str],
    random_seed: int,
    game_wall_time_s: float | None,
    lookahead_k: int,
    duration_s: float,
) -> dict[str, Any]:
    normalized_paths = list(dict.fromkeys(str(path) for path in world_model_paths))
    normalized_skills = list(dict.fromkeys(str(path) for path in skill_file_paths))
    hashes = path_hashes(repo, normalized_paths + normalized_skills)
    active_hashes = {
        str(row.get("game")): str(row.get("active_dataset_sha256", "")) for row in per_game_scorecard
    }
    checksum = compute_reproducibility_checksum(
        per_game_scorecard=per_game_scorecard,
        world_model_paths=normalized_paths,
        skill_file_paths=normalized_skills,
        path_hashes=hashes,
        active_dataset_hashes=active_hashes,
        random_seed=random_seed,
        game_wall_time_s=game_wall_time_s,
        lookahead_k=lookahead_k,
    )
    return {
        "experiment": "experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09",
        "method": "targeted_active_data_plus_mind_studio_named_register_k_step_lookahead_fidelity",
        "target_order": list(TARGET_ORDER),
        "game_wall_time_s": game_wall_time_s,
        "lookahead_k": int(lookahead_k),
        "honest_verdict": _combined_verdict(per_game_scorecard),
        "per_game_scorecard": per_game_scorecard,
        "new_levels_reproduced": _new_levels_reproduced(per_game_scorecard),
        "reproducible_total_levels": int(reproducible_total_levels),
        "world_model_paths": normalized_paths,
        "world_model_path_sha256": {path: hashes.get(path, "") for path in normalized_paths},
        "mind_studio_skill_paths": normalized_skills,
        "mind_studio_skill_sha256": {path: hashes.get(path, "") for path in normalized_skills},
        "active_dataset_sha256": active_hashes,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-4384", "SCENARIO-PHASE4-4384"],
        "inference_substrate": "offline_targeted_active_data_plus_mind_studio_lookahead_no_nested_codex",
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
                "lookahead_fidelity",
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
            fidelity = row.get("lookahead_fidelity")
            if isinstance(fidelity, bool) or not isinstance(fidelity, (int, float)):
                errors.append(f"per_game_scorecard[{index}].lookahead_fidelity must be bare number")
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
    cap = artifact.get("game_wall_time_s")
    if cap is not None and not isinstance(cap, (int, float)):
        errors.append("game_wall_time_s must be numeric")
    if not isinstance(artifact.get("lookahead_k"), int):
        errors.append("lookahead_k must be int")
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


def _imports_ok() -> dict[str, bool]:
    checks = {
        "harness_import": "carnot.agentic.arc_executable_world_model",
        "solver_kit_import": "carnot.agentic.arc_solver_kit",
        "active_data_module_import": "carnot.experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09",
    }
    out: dict[str, bool] = {}
    for key, module in checks.items():
        try:
            importlib.import_module(module)
            out[key] = True
        except Exception:
            out[key] = False
    return out


def _research_conductor_modified(repo: Path) -> bool:
    if not (repo / ".git").exists():
        return False
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--short", "--", "scripts/research_conductor.py"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return bool(out.stdout.strip())


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
    imports = _imports_ok()
    active_ok, active_error = base.exp4363.active_collect_import_status()
    return {
        "games": games,
        "offline_env_present": offline_env_present,
        **imports,
        "executable_world_model_import": imports.get("harness_import", False),
        "active_collect_import": active_ok,
        "active_collect_import_error": active_error,
        "lookahead_fidelity_enabled": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": _research_conductor_modified(repo),
    }


def _reproduce_candidate_next_level(game: str, labels: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    prior = PRIOR_BEST_LEVELS[game]
    if not labels:
        return {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
            "mode": "lookahead_or_mechanic_gate_failed_no_replayable_plan",
        }
    return base._reproduce_candidate_next_level(game, labels)


def _run_game(  # pragma: no cover - offline SDK boundary
    repo: Path,
    game: str,
    random_seed: int,
    active_budget: int,
    round_budget: int,
    lookahead_k: int,
    skill_file_path: str,
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
        accuracy, residual = base._verify_world_model_round(game, random_seed=random_seed + round_index)
        rounds.append(accuracy)
        residuals.append(residual)
        print(f"{game} verifier round {round_index} accuracy={accuracy:.6f}", flush=True)
        if accuracy >= VERIFY_THRESHOLD and residual == "none":
            break
    try:
        engine, _complete = e3.load_engine(game)
        lookahead_fidelity = compute_k_step_lookahead_fidelity(
            game=game,
            transitions=active_transitions,
            engine=engine,
            k=lookahead_k,
        )
        lemmas = base._targeted_gap_lemmas(game, active_transitions, engine)
    except Exception as exc:
        lookahead_fidelity = 0.0
        lemmas = [{"verifier_gated": False, "error": repr(exc)[:160]}]
    mechanic_checks_passed = (
        max(rounds or [0.0]) >= VERIFY_THRESHOLD
        and lookahead_fidelity >= LOOKAHEAD_THRESHOLD
        and (not residuals or residuals[-1] == "none")
        and all(bool(row.get("verifier_gated")) for row in lemmas if isinstance(row, dict))
    )
    if mechanic_checks_passed:
        plan = base.prior_plan(repo, game)
        plan_source = "lookahead_and_mechanic_checks_passed_prior_l1_plan_replayed_against_l2_gate"
    else:
        plan = []
        plan_source = "planning_blocked_until_named_register_lookahead_fidelity_passes"
    reproduce_result = _reproduce_candidate_next_level(game, plan)
    residual = RESIDUAL_GAP_CLASSES[game]
    if any(value not in ("none", RESIDUAL_GAP_CLASSES[game]) for value in residuals):
        residual = residuals[-1]
    return build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=rounds,
        lookahead_fidelity_per_round=[lookahead_fidelity for _ in rounds] or [lookahead_fidelity],
        active_dataset_summary=active_summary,
        world_model_path=repo / WORLD_MODEL_PATHS[game],
        skill_file_path=skill_file_path,
        plan=plan,
        reproduce_result=reproduce_result,
        residual_gap_class=residual,
        targeted_gap_lemmas=lemmas,
        plan_source=plan_source,
        mechanic_checks_passed=mechanic_checks_passed,
    )


TARGET_RUNNERS: dict[str, Callable[[Path, str, int, int, int, int, str], dict[str, Any]]] = {
    game: _run_game for game in TARGET_ORDER
}


def _target_worker(  # pragma: no cover - multiprocessing boundary
    game: str,
    repo: str,
    random_seed: int,
    active_budget: int,
    round_budget: int,
    lookahead_k: int,
    skill_file_path: str,
    out_queue: mp.Queue,
) -> None:
    try:
        row = TARGET_RUNNERS[game](
            Path(repo),
            game,
            random_seed,
            active_budget,
            round_budget,
            lookahead_k,
            skill_file_path,
        )
        out_queue.put({"ok": True, "row": row})
    except Exception:
        out_queue.put({"ok": False, "traceback": traceback.format_exc()})


def _run_game_with_cap(  # pragma: no cover - multiprocessing/offline boundary
    game: str,
    repo: Path,
    random_seed: int,
    active_budget: int,
    round_budget: int,
    lookahead_k: int,
    skill_file_path: str,
    game_wall_time_s: float | None,
) -> dict[str, Any]:
    if game_wall_time_s is None:
        try:
            return TARGET_RUNNERS[game](
                repo,
                game,
                random_seed,
                active_budget,
                round_budget,
                lookahead_k,
                skill_file_path,
            )
        except Exception:
            return exception_game_row(game, traceback.format_exc())
    out_queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_target_worker,
        args=(game, str(repo), random_seed, active_budget, round_budget, lookahead_k, skill_file_path, out_queue),
    )
    proc.start()
    proc.join(float(game_wall_time_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return timeout_game_row(game, float(game_wall_time_s))
    try:
        payload = out_queue.get_nowait()
    except queue.Empty:
        return exception_game_row(game, f"{game} runner exited without result")
    if payload.get("ok"):
        return payload["row"]
    return exception_game_row(game, str(payload.get("traceback", "")))


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_checkpoint(rows: list[dict[str, Any]], random_seed: int, lookahead_k: int) -> None:
    checkpoint_path = RESULT_PATH.with_suffix(".checkpoint.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "experiment": "experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09",
                "checkpoint_kind": "per_game_partial",
                "completed_games": [row["game"] for row in rows],
                "per_game_scorecard": rows,
                "lookahead_k": int(lookahead_k),
                "random_seed": int(random_seed),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_gap(path: Path, *, row: dict[str, Any], checksum: str) -> None:  # pragma: no cover - ops ledger boundary
    path.parent.mkdir(parents=True, exist_ok=True)
    game = str(row["game"])
    marker = f"### 2026-06-18 Exp4384 {game} Mind-Studio lookahead residual gap"
    entry = (
        f"\n\n{marker}\n"
        "- Spec: REQ-PHASE4-4384 / SCENARIO-PHASE4-4384\n"
        f"- Best verifier accuracy: {float(row.get('verifier_accuracy', 0.0)):.4f}\n"
        f"- K-step lookahead fidelity: {float(row.get('lookahead_fidelity', 0.0)):.4f}\n"
        f"- Active transitions collected: {int(row.get('active_transitions_collected', 0))}\n"
        f"- Target action counts: `{json.dumps(row.get('target_action_counts', {}), sort_keys=True)}`\n"
        f"- Residual gap class: `{row.get('residual_gap_class', 'unknown')}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded active-data plus K-step named-register fidelity did not reproduce a new level beyond L1.\n"
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
    text = base._replace_total(
        text,
        "reproducible_total_levels",
        total,
        f"Exp4384 Mind-Studio named-gap L2 reproduction gate; checksum {checksum[:12]}",
    )
    REGISTRY_PATH.write_text(text, encoding="utf-8")


def run_experiment(
    *,
    random_seed: int = RANDOM_SEED,
    active_budget: int = ACTIVE_BUDGET,
    round_budget: int = ROUND_BUDGET,
    game_wall_time_s: float | None = DEFAULT_GAME_WALL_TIME_S,
    lookahead_k: int = LOOKAHEAD_K,
) -> dict[str, Any]:
    t0 = time.time()
    checks = preconditions(REPO)
    rows: list[dict[str, Any]] = []
    skill_paths: list[str] = []
    _write_checkpoint(rows, random_seed, lookahead_k)
    for game in TARGET_ORDER:
        _write_checkpoint(rows, random_seed, lookahead_k)
        skill_path = write_skill_file(REPO, game, random_seed=random_seed, lookahead_k=lookahead_k)
        skill_paths.append(skill_path)
        if not checks["games"][game]["offline_env_present"]:
            row = blocked_game_row(REPO, game, skill_file_path=skill_path)
        else:
            row = _run_game_with_cap(
                game,
                REPO,
                random_seed,
                active_budget,
                round_budget,
                lookahead_k,
                skill_path,
                game_wall_time_s,
            )
            row.setdefault("mind_studio_skill_file", skill_path)
            row.setdefault("lookahead_k", lookahead_k)
        rows.append(row)
        accuracy_rounds = row.get("verifier_accuracy_per_round") or [row.get("verifier_accuracy", 0.0)]
        fidelity_rounds = row.get("lookahead_fidelity_per_round") or [row.get("lookahead_fidelity", 0.0)]
        for round_index, accuracy in enumerate(accuracy_rounds):
            fidelity = fidelity_rounds[min(round_index, len(fidelity_rounds) - 1)]
            print(
                f"{game} verifier round {round_index} accuracy={float(accuracy):.6f} "
                f"lookahead_fidelity={float(fidelity):.6f}",
                flush=True,
            )
        print(
            f"{game} checkpoint={row['checkpoint_status']} "
            f"active_transitions={row.get('active_transitions_collected', 0)}",
            flush=True,
        )
        _write_checkpoint(rows, random_seed, lookahead_k)

    new_levels = _new_levels_reproduced(rows)
    total = _registry_total_from_repo(REPO)
    if total is None:
        total = PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels
    total = max(total, PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels)
    artifact = build_artifact(
        repo=REPO,
        per_game_scorecard=rows,
        reproducible_total_levels=total,
        world_model_paths=list(WORLD_MODEL_PATHS.values()),
        skill_file_paths=skill_paths,
        random_seed=random_seed,
        game_wall_time_s=game_wall_time_s,
        lookahead_k=lookahead_k,
        duration_s=time.time() - t0,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4384 artifact schema errors: {errors}")
    for row in rows:
        if row.get("offline_env_present") and not row.get("offline_reproduced"):
            _write_gap(GAP_PATH, row=row, checksum=artifact["reproducibility_checksum"])
    _update_registry_for_new_levels(rows, artifact["reproducibility_checksum"])
    _write_artifact(artifact)
    print(
        f"wrote {RESULT_RELATIVE_PATH} verdict={artifact['honest_verdict']} "
        f"new_levels={artifact['new_levels_reproduced']} total={artifact['reproducible_total_levels']}",
        flush=True,
    )
    return artifact


def main() -> int:  # pragma: no cover - exercised through results wrapper in operator runs
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
