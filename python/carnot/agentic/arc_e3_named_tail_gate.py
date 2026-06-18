"""Offline ARC E3 named-tail retry for Exp 4395.

Spec refs: REQ-PHASE4-4395, SCENARIO-PHASE4-4395.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_ORDER = ("ar25", "ka59", "ft09")
PRIOR_BEST_LEVELS = {"ar25": 1, "ka59": 1, "ft09": 1}
PRIOR_REPRODUCIBLE_TOTAL = 34
FIDELITY_GATE = 0.95
LOOKAHEAD_K = 3
RANDOM_SEED = 4395
TARGET_WALL_TIME_S = 30.0
PRIOR_ARTIFACT = "results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"
RESULT_ARTIFACT = "results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"

WORLD_MODEL_PATHS = {
    "ar25": "results/arc_e3/ar25/world_model.py",
    "ka59": "results/arc_e3/ka59/world_model.py",
    "ft09": "results/arc_e3/ft09/world_model.py",
}

RESIDUAL_GAP_CLASSES = {
    "ar25": "ar25_l2_action7_undo_stack_hidden_rule_gap",
    "ka59": "ka59_l2_object_relevance_step_counter_hud_register_gap",
    "ft09": "ft09_l2_residual_world_model_mismatch_gap",
}

NAMED_REGISTERS = {
    "ar25": "action7_undo_stack",
    "ka59": "step_counter_hud_register_object_relevance",
    "ft09": "coverage_balanced_residual_world_model",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_ar25_ka59_ft09_<n>_reproduced or "
        "complete_e3_ar25_ka59_ft09_partial). Any new reproduced level and an honest "
        "partial per game are BOTH progress."
    ),
    "per_game_scorecard": (
        "list of {game, prior_best_level, new_reproduced_level, lookahead_fidelity, "
        "verifier_accuracy, offline_reproduced, residual_gap_class} -- the per-game "
        "record for ar25/ka59/ft09 (ka59 includes the object-relevance discriminator outcome)."
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
        "Hash of the object-relevance discriminator + the extended models + the plans + "
        "the reproduce() results; lets a third party re-run."
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Filesystem configuration for the Exp 4395 offline checkpoint."""

    repo_root: Path
    registry_path: Path
    prior_artifact_path: Path
    artifact_path: Path
    env_root: Path
    random_seed: int = RANDOM_SEED
    fidelity_gate: float = FIDELITY_GATE
    lookahead_k: int = LOOKAHEAD_K
    target_wall_time_s: float = TARGET_WALL_TIME_S

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> "ExperimentConfig":
        root = Path(repo_root).resolve()
        return cls(
            repo_root=root,
            registry_path=root / "ops" / "arc_solve_registry.yaml",
            prior_artifact_path=root / PRIOR_ARTIFACT,
            artifact_path=root / RESULT_ARTIFACT,
            env_root=root / "environment_files",
        )


def load_prior_scorecards(prior_artifact_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(prior_artifact_path.read_text(encoding="utf-8"))
    return {
        str(card["game"]): card
        for card in data.get("per_game_scorecard", [])
        if card.get("game") in TARGET_ORDER
    }


def read_registry_total(registry_path: Path) -> int:
    if not registry_path.exists():
        return PRIOR_REPRODUCIBLE_TOTAL
    match = re.search(
        r"^reproducible_total_levels:\s*(\d+)\b",
        registry_path.read_text(encoding="utf-8"),
        re.M,
    )
    return int(match.group(1)) if match else PRIOR_REPRODUCIBLE_TOTAL


def target_env_available(env_root: Path, game: str) -> bool:
    game_dir = env_root / game
    return game_dir.is_dir() and any(game_dir.iterdir())


def check_harness_imports() -> dict[str, bool]:  # pragma: no cover - import boundary.
    modules = {
        "arc_solver_kit": "carnot.agentic.arc_solver_kit",
        "arc_executable_world_model": "carnot.agentic.arc_executable_world_model",
    }
    status: dict[str, bool] = {}
    for label, module_name in modules.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            status[label] = False
        else:
            status[label] = True
    return status


def _rounds(card: dict[str, Any], key: str, fallback: float) -> list[float]:
    values = card.get(f"{key}_per_round") or [fallback]
    return [float(value) for value in values]


def _preconditions(config: ExperimentConfig, import_status: dict[str, bool]) -> dict[str, Any]:
    offline_envs = {
        game: {
            "available": target_env_available(config.env_root, game),
            "offline_env_path": str(config.env_root / game),
            "status": (
                "available"
                if target_env_available(config.env_root, game)
                else f"blocked_offline_env_missing_{game}"
            ),
        }
        for game in TARGET_ORDER
    }
    return {
        "offline_envs": offline_envs,
        "harness_import": all(import_status.values()),
        "arc_solver_kit_import": bool(import_status.get("arc_solver_kit")),
        "executable_world_model_import": bool(import_status.get("arc_executable_world_model")),
        "trm_training": "stood_down_not_invoked",
        "leaderboard_submission": False,
        "research_conductor_modified": False,
        "per_register_lookahead_fidelity_enabled": True,
    }


def _write_skill_file(
    config: ExperimentConfig,
    game: str,
    prior_card: dict[str, Any],
    gate_passed: bool,
    object_relevance: dict[str, Any] | None,
) -> str:
    skill_dir = config.repo_root / "results" / "arc_e3" / game
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "skill_4395.json"
    payload = {
        "experiment": "experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09",
        "spec_refs": ["REQ-PHASE4-4395", "SCENARIO-PHASE4-4395"],
        "game": game,
        "method": "per_register_lookahead_fidelity_gate",
        "named_register": NAMED_REGISTERS[game],
        "prior_best_level": PRIOR_BEST_LEVELS[game],
        "target_level": PRIOR_BEST_LEVELS[game] + 1,
        "random_seed": config.random_seed,
        "lookahead_k": config.lookahead_k,
        "fidelity_gate": config.fidelity_gate,
        "planning_gate": "passed" if gate_passed else "blocked_until_named_register_fidelity_passes",
        "object_relevance_discriminator": object_relevance,
        "rounds": [
            {
                "round": index + 1,
                "verifier_accuracy": verifier,
                "lookahead_fidelity": fidelity,
            }
            for index, (verifier, fidelity) in enumerate(
                zip(
                    _rounds(prior_card, "verifier_accuracy", float(prior_card.get("verifier_accuracy", 0.0))),
                    _rounds(prior_card, "lookahead_fidelity", float(prior_card.get("lookahead_fidelity", 0.0))),
                )
            )
        ],
        "world_model_path": WORLD_MODEL_PATHS[game],
    }
    skill_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(skill_path.relative_to(config.repo_root))


def build_ka59_object_relevance_discriminator(prior_card: dict[str, Any], fidelity_gate: float) -> dict[str, Any]:
    lemmas = [row for row in prior_card.get("targeted_gap_lemmas", []) if isinstance(row, dict)]
    hud_lemmas = [row for row in lemmas if "hud_count_after" in row and "hud_count_before" in row]
    exact_hud_predictions = sum(
        1 for row in hud_lemmas if row.get("hud_count_after") == row.get("hud_count_predicted")
    )
    hud_prediction_accuracy = exact_hud_predictions / max(1, len(hud_lemmas))
    prior_fidelity = float(prior_card.get("lookahead_fidelity", 0.0))
    planning_allowed = prior_fidelity >= fidelity_gate and hud_prediction_accuracy >= fidelity_gate
    return {
        "blocker_class": "object_relevance_not_clicks_or_multi_object_push",
        "provenance_commits": ["f0b078247", "6fba583c7"],
        "commit_findings": [
            "click_capable_piece_centroid_exploration_built_no_ka59_advance",
            "adaptive_multi_object_push_state_built_no_regression_ka59_remains_over_fragmented",
        ],
        "active_object_candidates": [
            "selected_block",
            "second_movable_block",
            "agent_plus_second_movable_block",
            "all_piece_objects",
        ],
        "selected_object_hypothesis": "agent_plus_second_movable_block",
        "hud_lemma_count": len(hud_lemmas),
        "hud_prediction_accuracy": round(hud_prediction_accuracy, 6),
        "prior_lookahead_fidelity": prior_fidelity,
        "lookahead_fidelity_after_discriminator": prior_fidelity,
        "planning_allowed": planning_allowed,
        "outcome": (
            "fidelity_gate_passed"
            if planning_allowed
            else "object_relevance_hypothesis_recorded_but_named_register_fidelity_still_below_gate"
        ),
    }


def _reproduction_result(
    game: str,
    prior_best: int,
    target_level: int,
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None,
) -> tuple[dict[str, Any], bool]:
    if reproduction_runner is None:
        return (
            {
                "game": game,
                "claimed_level": target_level,
                "reached_level": prior_best,
                "reproduced": False,
                "reason": "reproduction_runner_not_configured",
            },
            False,
        )
    result = reproduction_runner(game, target_level)
    reached = int(result.get("reached_level", prior_best))
    reproduced = bool(result.get("reproduced")) and reached >= target_level
    return result, reproduced


def _scorecard(
    config: ExperimentConfig,
    game: str,
    prior_card: dict[str, Any],
    preconditions: dict[str, Any],
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None,
) -> dict[str, Any]:
    prior_best = PRIOR_BEST_LEVELS[game]
    target_level = prior_best + 1
    fidelity = float(prior_card.get("lookahead_fidelity", 0.0))
    verifier = float(prior_card.get("verifier_accuracy", fidelity))
    object_relevance = (
        build_ka59_object_relevance_discriminator(prior_card, config.fidelity_gate)
        if game == "ka59"
        else None
    )
    gate_passed = fidelity >= config.fidelity_gate
    if object_relevance is not None:
        gate_passed = gate_passed and bool(object_relevance["planning_allowed"])
    skill_path = _write_skill_file(config, game, prior_card, gate_passed, object_relevance)
    env_status = preconditions["offline_envs"][game]

    if not env_status["available"]:
        checkpoint_status = f"blocked_offline_env_missing_{game}"
        reproduce_result = {
            "game": game,
            "claimed_level": target_level,
            "reached_level": prior_best,
            "reproduced": False,
            "reason": checkpoint_status,
        }
        offline_reproduced = False
    elif not gate_passed:
        checkpoint_status = "honest_partial_fidelity_gate_not_met"
        reproduce_result = prior_card.get(
            "reproduce_result",
            {
                "game": game,
                "claimed_level": target_level,
                "reached_level": prior_best,
                "reproduced": False,
                "reason": checkpoint_status,
            },
        )
        offline_reproduced = False
    else:
        reproduce_result, offline_reproduced = _reproduction_result(
            game,
            prior_best,
            target_level,
            reproduction_runner,
        )
        checkpoint_status = (
            "offline_reproduced_new_level"
            if offline_reproduced
            else "honest_partial_reproduction_gate_not_proven"
        )

    return {
        "game": game,
        "prior_best_level": prior_best,
        "target_level": target_level,
        "new_reproduced_level": target_level if offline_reproduced else prior_best,
        "named_register": NAMED_REGISTERS[game],
        "lookahead_k": config.lookahead_k,
        "lookahead_fidelity": fidelity,
        "lookahead_fidelity_per_round": _rounds(prior_card, "lookahead_fidelity", fidelity),
        "fidelity_gate_passed": gate_passed,
        "verifier_accuracy": verifier,
        "verifier_accuracy_per_round": _rounds(prior_card, "verifier_accuracy", verifier),
        "offline_reproduced": offline_reproduced,
        "checkpoint_status": checkpoint_status,
        "residual_gap_class": "none" if offline_reproduced else RESIDUAL_GAP_CLASSES[game],
        "object_relevance_discriminator": object_relevance,
        "active_transitions_collected": int(prior_card.get("active_transitions_collected", 0)),
        "target_action_counts": prior_card.get("target_action_counts", {}),
        "targeted_gap_lemmas": prior_card.get("targeted_gap_lemmas", []),
        "active_dataset_sha256": str(prior_card.get("active_dataset_sha256", "")),
        "world_model_path": WORLD_MODEL_PATHS[game],
        "mind_studio_skill_file": skill_path,
        "reproduce_result": reproduce_result,
        "plan": prior_card.get("plan", []),
        "target_wall_time_s": config.target_wall_time_s,
    }


def _new_levels(scorecards: list[dict[str, Any]]) -> int:
    return sum(
        max(0, int(card["new_reproduced_level"]) - int(card["prior_best_level"]))
        for card in scorecards
        if card["offline_reproduced"]
    )


def _world_model_paths() -> list[str]:
    return [WORLD_MODEL_PATHS[game] for game in TARGET_ORDER]


def _checksum(repo_root: Path, artifact: dict[str, Any], paths: list[str]) -> str:
    material = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode("utf-8"))
    for raw_path in sorted(paths):
        full = repo_root / raw_path
        digest.update(raw_path.encode("utf-8"))
        if full.exists() and full.is_file():
            digest.update(full.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    required = [
        "honest_verdict",
        "per_game_scorecard",
        "new_levels_reproduced",
        "reproducible_total_levels",
        "world_model_paths",
        "verifier_is_oracle",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
    ]
    errors = [f"missing:{field}" for field in required if field not in artifact]
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_not_true")
    if not isinstance(artifact.get("new_levels_reproduced"), int):
        errors.append("new_levels_reproduced_not_int")
    if not isinstance(artifact.get("reproducible_total_levels"), int):
        errors.append("reproducible_total_levels_not_int")
    if artifact.get("world_model_paths") != _world_model_paths():
        errors.append("world_model_paths_not_named_tail_models")
    rows = artifact.get("per_game_scorecard")
    if isinstance(rows, list):
        games = [row.get("game") for row in rows if isinstance(row, dict)]
        if games != list(TARGET_ORDER):
            errors.append("per_game_scorecard_order_wrong")
        for row in rows:
            if isinstance(row, dict) and row.get("game") == "ka59" and not row.get("object_relevance_discriminator"):
                errors.append("ka59_missing_object_relevance_discriminator")
    return errors


def run_experiment(
    config: ExperimentConfig,
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None = None,
    write_artifact: bool = True,
    import_checker: Callable[[], dict[str, bool]] = check_harness_imports,
) -> dict[str, Any]:
    start = time.monotonic()
    prior_cards = load_prior_scorecards(config.prior_artifact_path)
    preconditions = _preconditions(config, import_checker())
    scorecards = [
        _scorecard(
            config,
            game,
            prior_cards.get(game, {}),
            preconditions,
            reproduction_runner,
        )
        for game in TARGET_ORDER
    ]
    new_levels = _new_levels(scorecards)
    world_paths = _world_model_paths()
    skill_paths = [card["mind_studio_skill_file"] for card in scorecards]
    artifact = {
        "experiment": "experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09",
        "artifact_path": str(config.artifact_path),
        "honest_verdict": (
            f"success_e3_ar25_ka59_ft09_{new_levels}_reproduced"
            if new_levels
            else "complete_e3_ar25_ka59_ft09_partial"
        ),
        "method": "offline_e3_named_tail_object_relevance_per_register_fidelity_gate",
        "target_order": list(TARGET_ORDER),
        "target_wall_time_s": config.target_wall_time_s,
        "lookahead_k": config.lookahead_k,
        "fidelity_gate": config.fidelity_gate,
        "per_game_scorecard": scorecards,
        "new_levels_reproduced": new_levels,
        "reproducible_total_levels": max(
            read_registry_total(config.registry_path),
            PRIOR_REPRODUCIBLE_TOTAL + new_levels,
        ),
        "world_model_paths": world_paths,
        "mind_studio_skill_paths": skill_paths,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions,
        "random_seed": config.random_seed,
        "duration_s": round(time.monotonic() - start, 6),
        "inference_substrate": "offline_arc_e3_harness_no_leaderboard_no_nested_codex",
        "submitted_to_leaderboard": False,
        "spec_refs": ["REQ-PHASE4-4395", "SCENARIO-PHASE4-4395"],
        "field_principles": FIELD_PRINCIPLES,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(config.repo_root, artifact, world_paths + skill_paths)
    errors = artifact_schema_errors(artifact)
    if errors:
        artifact["schema_errors"] = errors
    if write_artifact:
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        config.artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
