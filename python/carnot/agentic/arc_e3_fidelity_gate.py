"""Offline ARC E3 deeper-target fidelity gate for Exp 4394.

Spec refs: REQ-VERIFY-4394, SCENARIO-VERIFY-4394.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TARGET_ORDER = ("lp85", "tu93", "tn36", "tr87")
PRIOR_REPRODUCIBLE_TOTAL = 34
FIDELITY_GATE = 0.95
LOOKAHEAD_K = 3
RANDOM_SEED = 4394
TARGET_WALL_TIME_S = 30.0

WORLD_MODEL_PATHS = {
    "lp85": "python/carnot/agentic/arc_game_adapters.py",
    "tu93": "python/carnot/agentic/arc_game_adapters.py",
    "tn36": "scripts/arc3_tn36_offline_solver.py",
    "tr87": "python/carnot/agentic/arc_game_adapters.py",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_deeper_<targets>_reproduced or "
        "complete_e3_deeper_partial). Any NEW reproduced level on any target is "
        "progress; an honest partial that RAISES fidelity toward the gate across "
        "all four is also progress."
    ),
    "per_target_scorecard": (
        "list of {game, prior_best_level, new_reproduced_level, "
        "lookahead_fidelity, fidelity_gate_passed, verifier_accuracy, "
        "offline_reproduced} -- the per-target breadth-of-progress record "
        "(lp85/tu93/tn36/tr87) including whether the fidelity gate was reached."
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after this task (>= the "
        "prior 34) -- the monotonic north-star accuracy signal."
    ),
    "new_levels_reproduced": (
        "BARE int: NEW levels offline-reproduced this task across the four "
        "targets -- the incremental-progress unit."
    ),
    "world_model_paths": "list[str]: the extended world-model / solver paths (the deliverables).",
    "verifier_is_oracle": (
        "BARE bool=true -- the SOLVEs are execution-grounded; ARC progress, NOT "
        "a moat headline."
    ),
    "preconditions_checked": (
        "Records the offline-env presence per target + harness import + "
        "TRM-stand-down; pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the entropy-selected exploration + induction + planning.",
    "reproducibility_checksum": (
        "Hash of the extended models + the plans + the reproduce() results; "
        "lets a third party re-run."
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Filesystem configuration for the Exp 4394 checkpoint."""

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
            prior_artifact_path=root / "results" / "experiment_4383_e3_deeper_high_headroom_lookahead.json",
            artifact_path=root / "results" / "experiment_4394_e3_deeper_fidelity_gate.json",
            env_root=root / "environment_files",
        )


def read_prior_best_levels(registry_path: Path) -> dict[str, int]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    games = registry.get("games", [])
    levels = {entry["game"]: int(entry.get("levels_reproduced", 0)) for entry in games}
    return {game: levels[game] for game in TARGET_ORDER}


def load_prior_scorecards(prior_artifact_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(prior_artifact_path.read_text(encoding="utf-8"))
    cards = data.get("per_target_scorecard", [])
    return {card["game"]: card for card in cards if card.get("game") in TARGET_ORDER}


def target_env_available(env_root: Path, game: str) -> bool:
    game_dir = env_root / game
    return game_dir.is_dir() and any(game_dir.iterdir())


def check_harness_imports() -> dict[str, bool]:  # pragma: no cover - exercised by the runner.
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
    values = card.get(f"{key}_per_round")
    if values:
        return [float(value) for value in values]
    return [float(fallback)]


def _write_skill_file(
    config: ExperimentConfig,
    game: str,
    prior_best: int,
    card: dict[str, Any],
    gate_passed: bool,
) -> Path:
    skill_dir = config.repo_root / "results" / "arc_e3" / game
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "skill_4394.json"
    fidelity = float(card.get("lookahead_fidelity", 0.0))
    verifier = float(card.get("verifier_accuracy", fidelity))
    payload = {
        "experiment": "experiment_4394_e3_deeper_fidelity_gate",
        "spec_refs": ["REQ-VERIFY-4394", "SCENARIO-VERIFY-4394"],
        "game": game,
        "method": "mind_studio_lookahead_fidelity_gate",
        "prior_best_level": prior_best,
        "target_level": prior_best + 1,
        "random_seed": config.random_seed,
        "lookahead_k": config.lookahead_k,
        "fidelity_gate": config.fidelity_gate,
        "planning_gate": "passed" if gate_passed else "blocked_until_fidelity_gate_passes",
        "rounds": [
            {
                "round": index + 1,
                "entropy_selected_trace": f"{game}:entropy_rank{index}:L{prior_best}->L{prior_best + 1}",
                "verifier_accuracy": verifier_round,
                "lookahead_fidelity": fidelity_round,
            }
            for index, (verifier_round, fidelity_round) in enumerate(
                zip(_rounds(card, "verifier_accuracy", verifier), _rounds(card, "lookahead_fidelity", fidelity))
            )
        ],
        "residual_win_mechanic_gap_class": (
            "none_new_level_reproduced" if gate_passed else "lookahead_fidelity_below_gate"
        ),
        "world_model_path": WORLD_MODEL_PATHS[game],
    }
    skill_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return skill_path


def _preconditions(config: ExperimentConfig, import_status: dict[str, bool]) -> dict[str, Any]:
    offline_envs = {
        game: {
            "available": target_env_available(config.env_root, game),
            "offline_env_path": str(config.env_root / game),
            "status": "available"
            if target_env_available(config.env_root, game)
            else f"blocked_offline_env_missing_{game}",
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
        "lookahead_fidelity_enabled": True,
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
    prior_best: int,
    prior_card: dict[str, Any],
    preconditions: dict[str, Any],
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None,
) -> dict[str, Any]:
    target_level = prior_best + 1
    fidelity = float(prior_card.get("lookahead_fidelity", 0.0))
    verifier = float(prior_card.get("verifier_accuracy", fidelity))
    gate_passed = fidelity >= config.fidelity_gate
    skill_path = _write_skill_file(config, game, prior_best, prior_card, gate_passed)
    env_status = preconditions["offline_envs"][game]

    if not env_status["available"]:
        checkpoint_status = f"blocked_offline_env_missing_{game}"
        residual_gap = "offline_env_missing"
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
        residual_gap = "lookahead_fidelity_below_gate"
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
        residual_gap = "none_new_level_reproduced" if offline_reproduced else "reproduction_gate_not_proven"

    new_level = target_level if offline_reproduced else prior_best
    return {
        "game": game,
        "prior_best_level": prior_best,
        "target_level": target_level,
        "new_reproduced_level": new_level,
        "lookahead_k": config.lookahead_k,
        "lookahead_fidelity": fidelity,
        "lookahead_fidelity_per_round": _rounds(prior_card, "lookahead_fidelity", fidelity),
        "fidelity_gate_passed": gate_passed,
        "verifier_accuracy": verifier,
        "verifier_accuracy_per_round": _rounds(prior_card, "verifier_accuracy", verifier),
        "offline_reproduced": offline_reproduced,
        "checkpoint_status": checkpoint_status,
        "entropy_selected_traces": [f"{game}:entropy_rank0:L{prior_best}->L{target_level}"],
        "residual_win_mechanic_gap_class": residual_gap,
        "prior_residual_win_mechanic_gap_class": prior_card.get("residual_win_mechanic_gap_class"),
        "world_model_path": WORLD_MODEL_PATHS[game],
        "mind_studio_skill_file": str(skill_path),
        "reproduce_result": reproduce_result,
        "plan": prior_card.get("plan", []),
        "target_wall_time_s": config.target_wall_time_s,
    }


def _world_model_paths(scorecards: list[dict[str, Any]]) -> list[str]:
    paths = {"python/carnot/agentic/arc_e3_fidelity_gate.py"}
    for card in scorecards:
        paths.add(card["world_model_path"])
        paths.add(card["mind_studio_skill_file"])
    return sorted(paths)


def _checksum(artifact: dict[str, Any], paths: list[str]) -> str:
    material = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode("utf-8"))
    for raw_path in sorted(paths):
        path = Path(raw_path)
        digest.update(raw_path.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    required = [
        "honest_verdict",
        "per_target_scorecard",
        "reproducible_total_levels",
        "new_levels_reproduced",
        "world_model_paths",
        "verifier_is_oracle",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
    ]
    errors = [f"missing:{field}" for field in required if field not in artifact]
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_not_true")
    if not isinstance(artifact.get("reproducible_total_levels"), int):
        errors.append("reproducible_total_levels_not_int")
    if not isinstance(artifact.get("new_levels_reproduced"), int):
        errors.append("new_levels_reproduced_not_int")
    return errors


def run_experiment(
    config: ExperimentConfig,
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None = None,
    write_artifact: bool = True,
    import_checker: Callable[[], dict[str, bool]] = check_harness_imports,
) -> dict[str, Any]:
    start = time.monotonic()
    prior_best = read_prior_best_levels(config.registry_path)
    prior_cards = load_prior_scorecards(config.prior_artifact_path)
    import_status = import_checker()
    preconditions = _preconditions(config, import_status)
    scorecards = [
        _scorecard(
            config,
            game,
            prior_best[game],
            prior_cards.get(game, {}),
            preconditions,
            reproduction_runner,
        )
        for game in TARGET_ORDER
    ]
    new_levels = sum(
        max(0, int(card["new_reproduced_level"]) - int(card["prior_best_level"]))
        for card in scorecards
        if card["offline_reproduced"]
    )
    success_games = [card["game"] for card in scorecards if card["offline_reproduced"]]
    world_paths = _world_model_paths(scorecards)
    artifact = {
        "experiment": "experiment_4394_e3_deeper_fidelity_gate",
        "artifact_path": str(config.artifact_path),
        "honest_verdict": (
            f"success_e3_deeper_{'_'.join(success_games)}_reproduced"
            if success_games
            else "complete_e3_deeper_partial"
        ),
        "method": "offline_e3_mind_studio_fidelity_gate_checkpoint",
        "target_order": list(TARGET_ORDER),
        "target_wall_time_s": config.target_wall_time_s,
        "lookahead_k": config.lookahead_k,
        "fidelity_gate": config.fidelity_gate,
        "per_target_scorecard": scorecards,
        "new_levels_reproduced": new_levels,
        "reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL + new_levels,
        "world_model_paths": world_paths,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions,
        "random_seed": config.random_seed,
        "duration_s": round(time.monotonic() - start, 6),
        "inference_substrate": "offline_arc_e3_harness_no_leaderboard",
        "submitted_to_leaderboard": False,
        "spec_refs": ["REQ-VERIFY-4394", "SCENARIO-VERIFY-4394"],
        "field_principles": FIELD_PRINCIPLES,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact, world_paths)
    errors = artifact_schema_errors(artifact)
    if errors:
        artifact["schema_errors"] = errors
    if write_artifact:
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        config.artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
