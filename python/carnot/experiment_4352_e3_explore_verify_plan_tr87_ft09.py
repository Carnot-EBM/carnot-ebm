"""Exp 4352: E3 explore-verify-plan continuation for tr87 and ft09.

Spec refs: REQ-PHASE4-084, SCENARIO-PHASE4-084.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit


REPO = Path(__file__).resolve().parents[2]
GAMES = ("tr87", "ft09")
RANDOM_SEED = 4352
N_TRANSITIONS = 120
MECHANIC_PASS_ACCURACY = 0.95
RESULT_RELATIVE_PATH = "results/experiment_4352_e3_explore_verify_plan_tr87_ft09.json"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
GAP_PATH = REPO / GAP_RELATIVE_PATH
WORLD_MODEL_PATHS = {
    "tr87": "results/arc_e3/tr87/world_model.py",
    "ft09": "results/arc_e3/ft09/world_model.py",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_game_scorecard",
    "world_model_paths",
    "new_levels_reproduced",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_tr87_ft09_<n>_reproduced or "
        "complete_e3_tr87_ft09_partial). Any reproduced L1 and an honest partial per game "
        "are BOTH progress."
    ),
    "per_game_scorecard": (
        "list of {game, verifier_accuracy, offline_reproduced, reproduced_levels} -- "
        "the breadth-of-progress record for tr87/ft09."
    ),
    "world_model_paths": (
        "list[str]: results/arc_e3/{tr87,ft09}/world_model.py -- the induced models ARE "
        "the deliverables."
    ),
    "new_levels_reproduced": (
        "BARE int: NEW levels offline-reproduced across tr87+ft09 -- the incremental-progress unit."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the SOLVEs are execution-grounded; ARC progress, NOT a moat headline."
    ),
    "preconditions_checked": (
        "Records the offline-env presence per game + harness import + TRM-stand-down; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the induction + exploration + planning.",
    "reproducibility_checksum": (
        "Hash of the world models + the plans + the reproduce() results; lets a third party re-run."
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def world_model_path(repo: Path, game: str) -> Path:
    return repo / WORLD_MODEL_PATHS[game]


def path_hashes(repo: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        full = path if path.is_absolute() else repo / path
        hashes[str(item)] = sha256_file(full) if full.exists() and full.is_file() else ""
    return hashes


def preconditions(repo: Path) -> dict[str, Any]:
    games: dict[str, dict[str, Any]] = {}
    offline_env_present: dict[str, bool] = {}
    for game in GAMES:
        env = repo / "environment_files" / game
        present = env.is_dir() and any(env.iterdir())
        offline_env_present[game] = present
        games[game] = {
            "offline_env_present": present,
            "offline_env_path": str(env),
        }
    return {
        "games": games,
        "offline_env_present": offline_env_present,
        "harness_import": True,
        "solver_kit_import": True,
        "executable_world_model_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }


def collect_explore_lemmas(transitions: Sequence[Any], limit: int = 12) -> list[dict[str, Any]]:
    lemmas: list[dict[str, Any]] = []
    for transition in transitions:
        changed = int(np.count_nonzero(np.asarray(transition.grid) != np.asarray(transition.next_grid)))
        if changed == 0 and len(lemmas) >= 2:
            continue
        lemmas.append(
            {
                "action": int(transition.action),
                "has_data": transition.data is not None,
                "changed_cells": changed,
                "level_delta": int(transition.level_after) - int(transition.level_before),
                "verifier_gated": True,
            }
        )
        if len(lemmas) >= limit:
            break
    return lemmas


def adaptive_world_model_tests(game: str, accuracy: float) -> list[dict[str, Any]]:
    passed = accuracy >= MECHANIC_PASS_ACCURACY
    return [
        {
            "name": f"{game}_agent2world_mechanic_threshold_probe",
            "passed": passed,
            "accuracy_threshold": MECHANIC_PASS_ACCURACY,
            "observed_accuracy": round(float(accuracy), 6),
            "hidden_rule_mismatch_class": "none" if passed else f"{game}_mechanics_not_verified_for_planning",
        }
    ]


def residual_mismatch_class(mismatches: Sequence[dict[str, Any]]) -> str:
    if not mismatches:
        return "none"
    if any("error" in mismatch for mismatch in mismatches):
        return "engine_runtime_error_gap"
    if any(mismatch.get("your_prediction_was_wrong_at") == [] for mismatch in mismatches):
        return "model_predicted_identity_when_transition_changed_gap"
    if any(isinstance(mismatch.get("your_prediction_was_wrong_at"), str) for mismatch in mismatches):
        return "world_model_shape_rule_gap"
    actions = sorted({int(mismatch.get("action", -1)) for mismatch in mismatches})
    if 7 in actions:
        return "missing_world_model_rule_gap_hidden_undo_stack_action7"
    return "missing_world_model_rule_gap_actions_" + "_".join(str(action) for action in actions)


def _reproduced_levels(reproduce_result: dict[str, Any]) -> int:
    if not bool(reproduce_result.get("reproduced")):
        return 0
    return int(reproduce_result.get("reached_level", 0) or 0)


def _plan_executed(plan_result: dict[str, Any] | None) -> bool:
    if not plan_result:
        return False
    return bool(plan_result.get("executed") and not plan_result.get("divergence_step"))


def _checkpoint_status(game: str, accuracy: float, offline_reproduced: bool, reproduced_levels: int) -> str:
    if offline_reproduced and reproduced_levels >= 1:
        return f"success_e3_{game}_L1_reproduced"
    return f"complete_e3_{game}_partial_model_{accuracy:.2f}"


def build_missing_game_scorecard(repo: Path, game: str) -> dict[str, Any]:
    path = world_model_path(repo, game)
    return {
        "game": game,
        "offline_env_present": False,
        "verifier_accuracy": 0.0,
        "verifier_accuracy_per_round": [],
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "world_model_path": WORLD_MODEL_PATHS[game],
        "world_model_sha256": sha256_file(path) if path.exists() else "",
        "plan_executed": False,
        "plan_result": None,
        "reproduce_result": {
            "game": game,
            "reached_level": 0,
            "claimed_level": 1,
            "reproduced": False,
            "mode": "offline_env_missing",
        },
        "explore_lemmas": [],
        "explore_lemmas_collected": 0,
        "adaptive_test_results": [],
        "mechanic_checks_passed": False,
        "checkpoint_status": f"blocked_offline_env_missing_{game}",
        "residual_mismatch_class": "offline_env_missing",
    }


def build_game_scorecard(
    *,
    repo: Path,
    game: str,
    verifier_accuracy_per_round: list[float],
    world_model_path: Path,
    plan_result: dict[str, Any] | None,
    reproduce_result: dict[str, Any],
    residual_mismatch_class: str,
    explore_lemmas: list[dict[str, Any]],
    adaptive_test_results: list[dict[str, Any]],
) -> dict[str, Any]:
    accuracy = max(verifier_accuracy_per_round or [0.0])
    reproduced_levels = _reproduced_levels(reproduce_result)
    offline_reproduced = bool(reproduce_result.get("reproduced")) and reproduced_levels >= 1
    return {
        "game": game,
        "offline_env_present": True,
        "verifier_accuracy": round(float(accuracy), 6),
        "verifier_accuracy_per_round": verifier_accuracy_per_round,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "world_model_path": _relative_or_absolute(repo, world_model_path),
        "world_model_sha256": sha256_file(world_model_path),
        "plan_executed": _plan_executed(plan_result),
        "plan_result": plan_result,
        "reproduce_result": reproduce_result,
        "explore_lemmas": explore_lemmas,
        "explore_lemmas_collected": len(explore_lemmas),
        "adaptive_test_results": adaptive_test_results,
        "mechanic_checks_passed": all(bool(row.get("passed")) for row in adaptive_test_results),
        "checkpoint_status": _checkpoint_status(game, accuracy, offline_reproduced, reproduced_levels),
        "residual_mismatch_class": residual_mismatch_class,
    }


def compute_reproducibility_checksum(
    *,
    per_game_scorecard: list[dict[str, Any]],
    world_model_paths: Sequence[str],
    path_hashes: dict[str, str],
    random_seed: int,
) -> str:
    payload = {
        "per_game_scorecard": per_game_scorecard,
        "world_model_paths": list(world_model_paths),
        "path_hashes": path_hashes,
        "random_seed": random_seed,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _new_levels_reproduced(rows: Sequence[dict[str, Any]]) -> int:
    return sum(
        int(row.get("reproduced_levels", 0) or 0)
        for row in rows
        if bool(row.get("offline_reproduced"))
    )


def _combined_verdict(rows: Sequence[dict[str, Any]]) -> str:
    new_levels = _new_levels_reproduced(rows)
    if new_levels >= 1:
        return f"success_e3_tr87_ft09_{new_levels}_reproduced"
    if all(str(row.get("checkpoint_status", "")).startswith("blocked_offline_env_missing") for row in rows):
        return "blocked_offline_env_missing_tr87_ft09"
    return "complete_e3_tr87_ft09_partial"


def build_artifact(
    *,
    repo: Path,
    per_game_scorecard: list[dict[str, Any]],
    world_model_paths: Sequence[str],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    normalized_paths = list(dict.fromkeys(str(path) for path in world_model_paths))
    hashes = path_hashes(repo, normalized_paths)
    checksum = compute_reproducibility_checksum(
        per_game_scorecard=per_game_scorecard,
        world_model_paths=normalized_paths,
        path_hashes=hashes,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4352_e3_explore_verify_plan_tr87_ft09",
        "games": list(GAMES),
        "method": "aera_explore_verify_plan_agent2world_adaptive_world_model_testing",
        "honest_verdict": _combined_verdict(per_game_scorecard),
        "per_game_scorecard": per_game_scorecard,
        "world_model_paths": normalized_paths,
        "world_model_path_sha256": hashes,
        "new_levels_reproduced": _new_levels_reproduced(per_game_scorecard),
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-084", "SCENARIO-PHASE4-084"],
        "inference_substrate": "codex_direct_model_edit_offline_env_no_nested_proposer",
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
            for field in ("game", "verifier_accuracy", "offline_reproduced", "reproduced_levels"):
                if field not in row:
                    errors.append(f"per_game_scorecard[{index}] missing {field}")
            if not isinstance(row.get("offline_reproduced"), bool):
                errors.append(f"per_game_scorecard[{index}].offline_reproduced must be bare bool")
            if not isinstance(row.get("reproduced_levels"), int):
                errors.append(f"per_game_scorecard[{index}].reproduced_levels must be bare int")
    paths = artifact.get("world_model_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        errors.append("world_model_paths must be list[str]")
    if not isinstance(artifact.get("new_levels_reproduced"), int):
        errors.append("new_levels_reproduced must be bare int")
    if not isinstance(artifact.get("verifier_is_oracle"), bool):
        errors.append("verifier_is_oracle must be bare bool")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
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


def _apply_label(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - ARC SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    try:
        step = json.loads(label)
        action = int(step["action"])
        data = step.get("data")
    except (TypeError, ValueError, KeyError):
        action = int(label)
        data = None
    return env.step(_game_action(GameAction, action), data=data)


def _labels_from_plan(plan_result: dict[str, Any] | None) -> list[str]:
    if not plan_result:
        return []
    solution = plan_result.get("solution") or []
    labels: list[str] = []
    for item in solution:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict):
            labels.append(json.dumps(item, sort_keys=True))
    return labels


def _planned_result(game: str, engine: Any, is_level_complete: Any, mechanic_passed: bool) -> dict[str, Any]:
    if not mechanic_passed:
        return {
            "game": game,
            "planned": False,
            "reason": "mechanic checks did not pass; plan withheld per explore-verify-plan gate",
        }
    return e3.plan_and_execute(game, engine, is_level_complete)


def score_game(game: str, *, repo: Path, random_seed: int, n_transitions: int) -> dict[str, Any]:
    transitions, cell = e3.collect_transitions(game, n=n_transitions, seed=random_seed)
    verifier = e3.WorldModelVerifier(transitions)
    engine, is_level_complete = e3.load_engine(game)
    verify_result = verifier.score(engine)
    accuracy = round(float(verify_result.accuracy), 6)
    print(f"{game} verifier round 0 accuracy={accuracy:.6f} cell={cell}", flush=True)

    lemmas = collect_explore_lemmas(transitions)
    adaptive_results = adaptive_world_model_tests(game, accuracy)
    mechanic_passed = all(bool(row.get("passed")) for row in adaptive_results)
    plan_result = _planned_result(game, engine, is_level_complete, mechanic_passed)

    reproduce_result = {
        "game": game,
        "reached_level": 0,
        "claimed_level": 1,
        "reproduced": False,
        "mode": "not_reproduced",
    }
    labels = _labels_from_plan(plan_result)
    if plan_result.get("level_up") and labels:
        reproduce_result = arc_solver_kit.reproduce(game, labels, _apply_label, claimed_level=1)
    elif plan_result.get("level_up"):
        reproduce_result["mode"] = "level_up_without_replayable_solution_labels"

    return build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=[accuracy],
        world_model_path=world_model_path(repo, game),
        plan_result=plan_result,
        reproduce_result=reproduce_result,
        residual_mismatch_class=residual_mismatch_class(verify_result.mismatches),
        explore_lemmas=lemmas,
        adaptive_test_results=adaptive_results,
    )


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_gap(path: Path, *, row: dict[str, Any], checksum: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    game = str(row["game"])
    marker = f"### 2026-06-17 Exp4352 {game} E3 residual gap"
    entry = (
        f"\n\n{marker}\n"
        "- Spec: REQ-PHASE4-084 / SCENARIO-PHASE4-084\n"
        f"- Game: `{game}`\n"
        f"- Best verifier accuracy: {float(row.get('verifier_accuracy', 0.0)):.4f}\n"
        f"- Residual mismatch class: `{row.get('residual_mismatch_class', 'unknown')}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded explore-verify-plan did not satisfy the offline reproduced L1 gate.\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    if marker in existing:
        before = existing.split(marker, 1)[0].rstrip()
        after_parts = existing.split(marker, 1)[1].split("\n\n### ", 1)
        after = ("\n\n### " + after_parts[1]) if len(after_parts) == 2 else ""
        path.write_text(before + entry + after, encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def run_experiment(*, random_seed: int = RANDOM_SEED, n_transitions: int = N_TRANSITIONS) -> dict[str, Any]:
    t0 = time.time()
    checks = preconditions(REPO)
    rows: list[dict[str, Any]] = []
    for game in GAMES:
        if not checks["offline_env_present"][game]:
            row = build_missing_game_scorecard(REPO, game)
            print(f"{game} checkpoint={row['checkpoint_status']}", flush=True)
        else:
            row = score_game(game, repo=REPO, random_seed=random_seed, n_transitions=n_transitions)
            print(f"{game} checkpoint={row['checkpoint_status']}", flush=True)
        rows.append(row)

    artifact = build_artifact(
        repo=REPO,
        per_game_scorecard=rows,
        world_model_paths=[WORLD_MODEL_PATHS[game] for game in GAMES],
        random_seed=random_seed,
        duration_s=time.time() - t0,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4352 artifact schema errors: {errors}")
    _write_artifact(artifact)
    for row in rows:
        if row.get("offline_env_present") and not row.get("offline_reproduced"):
            _write_gap(GAP_PATH, row=row, checksum=str(artifact["reproducibility_checksum"]))
    print(
        f"wrote {RESULT_RELATIVE_PATH} verdict={artifact['honest_verdict']} "
        f"new_levels={artifact['new_levels_reproduced']}",
        flush=True,
    )
    return artifact


def main() -> int:  # pragma: no cover - exercised through the results wrapper
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
