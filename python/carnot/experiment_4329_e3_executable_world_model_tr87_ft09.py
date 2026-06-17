"""Exp 4329: offline E3 executable-world-model attempt on ARC-AGI-3 tr87 and ft09.

Spec refs: REQ-PHASE4-076, SCENARIO-PHASE4-076.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit


REPO = Path(__file__).resolve().parents[2]
GAMES = ("tr87", "ft09")
RANDOM_SEED = 4329
N_TRANSITIONS = 120
RESULT_RELATIVE_PATH = "results/experiment_4329_e3_executable_world_model_tr87_ft09.json"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
GAP_PATH = REPO / GAP_RELATIVE_PATH

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_game_scorecard",
    "reproduced_levels_total",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_tr87_ft09_<n>_L1_reproduced or "
        "complete_e3_tr87_ft09_partial). Any reproduced L1 OR an honest partial per game is progress."
    ),
    "per_game_scorecard": (
        "Per game (tr87, ft09): offline_reproduced + reproduced_levels + best "
        "verifier_accuracy + world_model_sha256 -- breadth-of-progress, guards the "
        "one-game-dominates failure mode."
    ),
    "reproduced_levels_total": (
        "BARE int: total offline-reproduced levels across tr87+ft09 this task -- the +1+ "
        "incremental-progress unit."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the SOLVES are execution-grounded; ARC progress, NOT a moat headline."
    ),
    "preconditions_checked": (
        "Records the per-game offline-env presence + harness import + TRM-stand-down; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the induction + planning.",
    "reproducibility_checksum": (
        "Hash of the world models + the plans + the reproduce() results; lets a third party re-run."
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def world_model_relative_path(game: str) -> str:
    return f"results/arc_e3/{game}/world_model.py"


def world_model_path(repo: Path, game: str) -> Path:
    return repo / world_model_relative_path(game)


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preconditions(repo: Path) -> dict[str, Any]:
    env_presence = {}
    env_paths = {}
    for game in GAMES:
        env = repo / "environment_files" / game
        env_presence[game] = env.is_dir() and any(env.iterdir())
        env_paths[game] = str(env)
    return {
        "offline_env_present": env_presence,
        "offline_env_paths": env_paths,
        "harness_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }


def _reproduced_levels(reproduce_result: dict[str, Any]) -> int:
    if not bool(reproduce_result.get("reproduced")):
        return 0
    return int(reproduce_result.get("reached_level", 0) or 0)


def _plan_executed(plan_result: dict[str, Any] | None) -> bool:
    if not plan_result:
        return False
    return bool(plan_result.get("executed") and not plan_result.get("divergence_step"))


def _divergence_step(plan_result: dict[str, Any] | None) -> Any:
    if not plan_result:
        return None
    return plan_result.get("divergence_step")


def _game_status(game: str, best_accuracy: float, reproduced_levels: int, offline_reproduced: bool) -> str:
    if offline_reproduced and reproduced_levels >= 1:
        return f"success_e3_{game}_L1_reproduced"
    return f"complete_e3_{game}_partial_model_{best_accuracy:.2f}"


def residual_mismatch_class(mismatches: list[dict[str, Any]]) -> str:
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


def build_missing_game_scorecard(repo: Path, game: str) -> dict[str, Any]:
    path = world_model_path(repo, game)
    return {
        "game": game,
        "status": f"blocked_offline_env_missing_{game}",
        "offline_env_present": False,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "verifier_accuracy_per_round": [],
        "best_verifier_accuracy": 0.0,
        "world_model_path": world_model_relative_path(game),
        "world_model_sha256": sha256_file(path) if path.exists() else "",
        "plan_executed": False,
        "plan_executed_detail": {"divergence_step": None, "plan_result": None},
        "plan_result": None,
        "reproduce_result": {
            "game": game,
            "reached_level": 0,
            "claimed_level": 1,
            "reproduced": False,
            "mode": "offline_env_missing",
        },
        "residual_mismatch_class": "offline_env_missing",
    }


def build_game_scorecard(
    *,
    repo: Path,
    game: str,
    status: str,
    verifier_accuracy_per_round: list[float],
    world_model_path: Path,
    plan_result: dict[str, Any] | None,
    reproduce_result: dict[str, Any],
    residual_mismatch_class: str,
) -> dict[str, Any]:
    best_accuracy = max(verifier_accuracy_per_round or [0.0])
    reproduced_levels = _reproduced_levels(reproduce_result)
    offline_reproduced = bool(reproduce_result.get("reproduced")) and reproduced_levels >= 1
    return {
        "game": game,
        "status": status,
        "offline_env_present": True,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "verifier_accuracy_per_round": verifier_accuracy_per_round,
        "best_verifier_accuracy": best_accuracy,
        "world_model_path": _relative_or_absolute(repo, world_model_path),
        "world_model_sha256": sha256_file(world_model_path),
        "plan_executed": _plan_executed(plan_result),
        "plan_executed_detail": {
            "divergence_step": _divergence_step(plan_result),
            "plan_result": plan_result,
        },
        "plan_result": plan_result,
        "reproduce_result": reproduce_result,
        "residual_mismatch_class": residual_mismatch_class,
    }


def compute_reproducibility_checksum(
    *,
    per_game_scorecard: dict[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "per_game_scorecard": per_game_scorecard,
        "random_seed": random_seed,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _combined_verdict(reproduced_levels_total: int) -> str:
    if reproduced_levels_total >= 1:
        return f"success_e3_tr87_ft09_{reproduced_levels_total}_L1_reproduced"
    return "complete_e3_tr87_ft09_partial"


def blocked_artifact(repo: Path, *, random_seed: int) -> dict[str, Any]:
    per_game = {game: build_missing_game_scorecard(repo, game) for game in GAMES}
    checksum = compute_reproducibility_checksum(
        per_game_scorecard=per_game,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4329_e3_executable_world_model_tr87_ft09",
        "games": list(GAMES),
        "honest_verdict": "blocked_offline_env_missing_tr87_ft09",
        "per_game_scorecard": per_game,
        "reproduced_levels_total": 0,
        "verifier_is_oracle": True,
        "preconditions_checked": {
            **preconditions(repo),
            "offline_env_present": {game: False for game in GAMES},
        },
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "inference_substrate": "codex_direct_model_edit_offline_env_no_nested_proposer",
        "submitted_to_leaderboard": False,
    }


def build_artifact(
    *,
    repo: Path,
    per_game_scorecard: dict[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    reproduced_levels_total = sum(
        int(per_game_scorecard.get(game, {}).get("reproduced_levels", 0) or 0)
        for game in GAMES
    )
    checksum = compute_reproducibility_checksum(
        per_game_scorecard=per_game_scorecard,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4329_e3_executable_world_model_tr87_ft09",
        "games": list(GAMES),
        "method": "executable_world_model_verify_plan_reproduce_checkpointed",
        "honest_verdict": _combined_verdict(reproduced_levels_total),
        "per_game_scorecard": per_game_scorecard,
        "reproduced_levels_total": reproduced_levels_total,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "inference_substrate": "codex_direct_model_edit_offline_env_no_nested_proposer",
        "submitted_to_leaderboard": False,
        "duration_s": round(duration_s, 3),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    scorecard = artifact.get("per_game_scorecard")
    if not isinstance(scorecard, dict):
        errors.append("per_game_scorecard must be dict")
        scorecard = {}
    for game in GAMES:
        row = scorecard.get(game)
        if not isinstance(row, dict):
            errors.append(f"missing {game} scorecard")
            continue
        for field in ("offline_reproduced", "plan_executed"):
            if not isinstance(row.get(field), bool):
                errors.append(f"{game}.{field} must be bare bool")
        if not isinstance(row.get("reproduced_levels"), int):
            errors.append(f"{game}.reproduced_levels must be bare int")
        if "world_model_sha256" not in row:
            errors.append(f"{game}.world_model_sha256 missing")
        if "best_verifier_accuracy" not in row:
            errors.append(f"{game}.best_verifier_accuracy missing")
    if not isinstance(artifact.get("reproduced_levels_total"), int):
        errors.append("reproduced_levels_total must be bare int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles missing")
    else:
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"principle mismatch for {field}")
    return errors


def _write_gap(path: Path, *, game: str, best_accuracy: float, mismatch_class: str, checksum: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n\n### 2026-06-17 Exp4329 {game} E3 residual gap\n"
        "- Spec: REQ-PHASE4-076 / SCENARIO-PHASE4-076\n"
        f"- Game: `{game}`\n"
        f"- Best verifier accuracy: {best_accuracy:.4f}\n"
        f"- Residual mismatch class: `{mismatch_class}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded executable-world-model run did not satisfy the offline reproduced L1 gate.\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    marker = f"### 2026-06-17 Exp4329 {game} E3 residual gap"
    if marker in existing:
        before = existing.split(marker, 1)[0].rstrip()
        after_parts = existing.split(marker, 1)[1].split("\n\n### ", 1)
        after = ("\n\n### " + after_parts[1]) if len(after_parts) == 2 else ""
        path.write_text(before + entry + after, encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _apply_noop(_env: Any, _label: str, frame: Any) -> Any:
    return frame


def _score_game(game: str, *, repo: Path, random_seed: int, n_transitions: int) -> dict[str, Any]:
    transitions, cell = e3.collect_transitions(game, n=n_transitions, seed=random_seed)
    verifier = e3.WorldModelVerifier(transitions)
    engine, is_level_complete = e3.load_engine(game)
    verify_result = verifier.score(engine)
    accuracies = [round(float(verify_result.accuracy), 6)]
    print(f"game={game} verifier round 0 accuracy={accuracies[-1]:.6f} cell={cell}", flush=True)

    plan_result = e3.plan_and_execute(game, engine, is_level_complete)
    print(f"game={game} plan result={plan_result}", flush=True)

    reproduce_result = {"game": game, "reached_level": 0, "claimed_level": 1, "reproduced": False}
    if plan_result.get("level_up"):
        reproduce_result = arc_solver_kit.reproduce(
            game,
            plan_result.get("solution", []),
            _apply_noop,
            claimed_level=1,
        )

    mismatch_class = residual_mismatch_class(verify_result.mismatches)
    reproduced_levels = _reproduced_levels(reproduce_result)
    offline_reproduced = bool(reproduce_result.get("reproduced")) and reproduced_levels >= 1
    status = _game_status(game, max(accuracies or [0.0]), reproduced_levels, offline_reproduced)
    return build_game_scorecard(
        repo=repo,
        game=game,
        status=status,
        verifier_accuracy_per_round=accuracies,
        world_model_path=world_model_path(repo, game),
        plan_result=plan_result,
        reproduce_result=reproduce_result,
        residual_mismatch_class=mismatch_class,
    )


def run_experiment(*, random_seed: int = RANDOM_SEED, n_transitions: int = N_TRANSITIONS) -> dict[str, Any]:
    t0 = time.time()
    checks = preconditions(REPO)
    present_games = [game for game in GAMES if checks["offline_env_present"][game]]
    if not present_games:
        artifact = blocked_artifact(REPO, random_seed=random_seed)
        _write_artifact(artifact)
        print("blocked_offline_env_missing_tr87_ft09", flush=True)
        return artifact

    per_game: dict[str, Any] = {}
    for game in GAMES:
        if not checks["offline_env_present"][game]:
            per_game[game] = build_missing_game_scorecard(REPO, game)
            print(f"game={game} blocked_offline_env_missing_{game}", flush=True)
            continue
        per_game[game] = _score_game(
            game,
            repo=REPO,
            random_seed=random_seed,
            n_transitions=n_transitions,
        )

    artifact = build_artifact(
        repo=REPO,
        per_game_scorecard=per_game,
        random_seed=random_seed,
        duration_s=time.time() - t0,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4329 artifact schema errors: {errors}")
    _write_artifact(artifact)
    for game, row in per_game.items():
        if row.get("offline_env_present") and not row.get("offline_reproduced"):
            _write_gap(
                GAP_PATH,
                game=game,
                best_accuracy=float(row.get("best_verifier_accuracy", 0.0)),
                mismatch_class=str(row.get("residual_mismatch_class", "unknown")),
                checksum=str(artifact["reproducibility_checksum"]),
            )
    print(f"wrote {RESULT_RELATIVE_PATH} verdict={artifact['honest_verdict']}", flush=True)
    return artifact


def main() -> int:  # pragma: no cover - exercised through results wrapper in operator runs
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
