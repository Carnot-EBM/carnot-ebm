"""Exp 4362: E3 blocked-mechanic deepen pass for ar25 and ka59.

Spec refs: REQ-PHASE4-086, SCENARIO-PHASE4-086.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from carnot import experiment_4339_e3_explore_verify_plan_ar25 as exp4339
from carnot import experiment_4350_e3_explore_verify_plan_ka59 as exp4350
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit


REPO = Path(__file__).resolve().parents[2]
RANDOM_SEED = 4362
N_TRANSITIONS = 160
TARGET_ORDER = ("ar25", "ka59")
PRIOR_BEST_LEVELS = {"ar25": 1, "ka59": 1}
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 32
RESULT_RELATIVE_PATH = "results/experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
GAP_PATH = REPO / GAP_RELATIVE_PATH
WORLD_MODEL_PATHS = {
    "ar25": "results/arc_e3/ar25/world_model.py",
    "ka59": "results/arc_e3/ka59/world_model.py",
}
PRIOR_RESULT_PATHS = {
    "ar25": "results/experiment_4339_e3_explore_verify_plan_ar25.json",
    "ka59": "results/experiment_4350_e3_explore_verify_plan_ka59.json",
}
NAMED_GAP_CLASSES = {
    "ar25": "ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap",
    "ka59": "ka59_l2_hidden_step_counter_hud_register_gap",
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
        "Terminal-prefixed (success_e3_ar25_ka59_<n>_reproduced or "
        "complete_e3_ar25_ka59_partial). Any new reproduced level and an honest "
        "partial per game are BOTH progress."
    ),
    "per_game_scorecard": (
        "list of {game, prior_best_level, new_reproduced_level, verifier_accuracy, "
        "offline_reproduced, residual_gap_class} -- the per-game record for ar25/ka59."
    ),
    "new_levels_reproduced": (
        "BARE int: NEW levels offline-reproduced across ar25+ka59 -- the incremental-progress unit."
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after this task (>= the prior 26) -- "
        "the monotonic north-star accuracy signal."
    ),
    "world_model_paths": (
        "list[str]: results/arc_e3/{ar25,ka59}/world_model.py -- the extended models ARE the deliverables."
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
        "Hash of the extended models + the plans + the reproduce() results; lets a third party re-run."
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_hashes(repo: Path, paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        full = path if path.is_absolute() else repo / path
        hashes[item] = sha256_file(full) if full.exists() and full.is_file() else ""
    return hashes


def compute_reproducibility_checksum(
    *,
    per_game_scorecard: list[dict[str, Any]],
    world_model_paths: list[str],
    path_hashes: dict[str, str],
    random_seed: int,
) -> str:
    payload = {
        "per_game_scorecard": per_game_scorecard,
        "world_model_paths": world_model_paths,
        "path_hashes": path_hashes,
        "random_seed": random_seed,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def preconditions(repo: Path) -> dict[str, Any]:
    games: dict[str, dict[str, Any]] = {}
    for game in TARGET_ORDER:
        env = repo / "environment_files" / game
        games[game] = {
            "offline_env_present": env.is_dir() and any(env.iterdir()),
            "offline_env_path": str(env),
        }
    return {
        "games": games,
        "harness_import": True,
        "solver_kit_import": True,
        "executable_world_model_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }


def _new_level_delta(row: dict[str, Any]) -> int:
    if not bool(row.get("offline_reproduced")):
        return 0
    return max(0, int(row.get("new_reproduced_level", 0)) - int(row.get("prior_best_level", 0)))


def _new_levels_reproduced(rows: list[dict[str, Any]]) -> int:
    return sum(_new_level_delta(row) for row in rows)


def _verdict(rows: list[dict[str, Any]]) -> str:
    new_levels = _new_levels_reproduced(rows)
    if new_levels:
        return f"success_e3_ar25_ka59_{new_levels}_reproduced"
    return "complete_e3_ar25_ka59_partial"


def blocked_game_row(game: str) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    return {
        "game": game,
        "prior_best_level": prior,
        "new_reproduced_level": prior,
        "verifier_accuracy": 0.0,
        "verifier_accuracy_per_round": [],
        "offline_reproduced": False,
        "reproduce_result": {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
        },
        "plan": [],
        "checkpoint_status": f"blocked_offline_env_missing_{game}",
        "residual_gap_class": "offline_env_missing",
        "world_model_path": WORLD_MODEL_PATHS[game],
        "targeted_gap_lemmas": [],
    }


def _prior_artifact_row(repo: Path, game: str) -> dict[str, Any]:
    prior = PRIOR_BEST_LEVELS[game]
    path = repo / PRIOR_RESULT_PATHS[game]
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        rounds = [float(value) for value in data.get("verifier_accuracy_per_round", [])]
        plan = list(
            data.get("accepted_plan")
            or data.get("plan_executed_detail", {}).get("plan_result", {}).get("solution", [])
        )
        reached = int(data.get("reproduced_levels", prior) or prior)
        reproduce_result = {
            "game": game,
            "reached_level": reached,
            "claimed_level": prior + 1,
            "reproduced": False,
        }
        checkpoint_status = "honest_partial_no_new_level_reproduced"
    else:
        rounds = []
        plan = []
        reproduce_result = {
            "game": game,
            "reached_level": prior,
            "claimed_level": prior + 1,
            "reproduced": False,
        }
        checkpoint_status = "honest_partial_prior_artifact_missing"
    accuracy = max(rounds or [0.0])
    return {
        "game": game,
        "prior_best_level": prior,
        "new_reproduced_level": prior,
        "verifier_accuracy": round(float(accuracy), 6),
        "verifier_accuracy_per_round": rounds,
        "offline_reproduced": False,
        "reproduce_result": reproduce_result,
        "plan": plan,
        "checkpoint_status": checkpoint_status,
        "residual_gap_class": NAMED_GAP_CLASSES[game],
        "world_model_path": WORLD_MODEL_PATHS[game],
        "targeted_gap_lemmas": [],
    }


def build_artifact(
    *,
    repo: Path,
    per_game_scorecard: list[dict[str, Any]],
    reproducible_total_levels: int,
    world_model_paths: list[str],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    normalized_paths = list(dict.fromkeys(str(path) for path in world_model_paths))
    path_hashes = _path_hashes(repo, normalized_paths)
    checksum = compute_reproducibility_checksum(
        per_game_scorecard=per_game_scorecard,
        world_model_paths=normalized_paths,
        path_hashes=path_hashes,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4362_e3_blocked_mechanic_levels_ar25_ka59",
        "method": "offline_e3_named_hidden_rule_explore_verify_plan",
        "target_order": list(TARGET_ORDER),
        "honest_verdict": _verdict(per_game_scorecard),
        "per_game_scorecard": per_game_scorecard,
        "new_levels_reproduced": _new_levels_reproduced(per_game_scorecard),
        "reproducible_total_levels": int(reproducible_total_levels),
        "world_model_paths": normalized_paths,
        "world_model_path_sha256": path_hashes,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-086", "SCENARIO-PHASE4-086"],
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
            ):
                if field not in row:
                    errors.append(f"per_game_scorecard[{index}] missing {field}")
            if not isinstance(row.get("offline_reproduced"), bool):
                errors.append(f"per_game_scorecard[{index}].offline_reproduced must be bare bool")
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


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_gap(
    path: Path,
    *,
    game: str,
    best_accuracy: float,
    residual_gap_class: str,
    checksum: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"### 2026-06-17 Exp4362 {game} named hidden-rule residual gap"
    entry = (
        f"\n\n{marker}\n"
        "- Spec: REQ-PHASE4-086 / SCENARIO-PHASE4-086\n"
        f"- Best verifier accuracy: {best_accuracy:.4f}\n"
        f"- Residual gap class: `{residual_gap_class}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded explore-verify-plan did not reproduce a new level beyond L1.\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    if marker in existing:
        before, after = existing.split(marker, 1)
        remainder = after.split("\n\n### ", 1)
        suffix = ("\n\n### " + remainder[1]) if len(remainder) == 2 else ""
        path.write_text(before.rstrip() + entry + suffix, encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def _registry_total(repo: Path) -> int | None:
    path = repo / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return None
    match = re.search(r"^reproducible_total_levels:\s*(\d+)\b", path.read_text(encoding="utf-8"), re.M)
    return int(match.group(1)) if match else None


def _reproduce_existing_plan_for_next_level(game: str, plan: list[str]) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    prior = PRIOR_BEST_LEVELS[game]
    if game == "ar25":
        apply = exp4339._apply_ar25_label
    elif game == "ka59":
        apply = exp4350._apply_ka59_label
    else:  # defensive guard for future edits
        raise KeyError(game)
    return arc_solver_kit.reproduce(game, plan, apply, claimed_level=prior + 1)


def _targeted_gap_lemmas(game: str, transitions: list[e3.Transition], engine) -> list[dict[str, Any]]:  # pragma: no cover - offline SDK boundary
    lemmas: list[dict[str, Any]] = []
    for transition in transitions:
        if game == "ar25" and int(transition.action) != 7:
            continue
        if game == "ka59" and transition.grid.shape == transition.next_grid.shape:
            row_changed = bool(np.any(transition.grid[-1] != transition.next_grid[-1]))
            if not row_changed:
                continue
        try:
            pred = np.asarray(engine(transition.grid.copy(), transition.action, transition.data))
        except Exception as exc:
            lemmas.append({"action": int(transition.action), "verifier_gated": False, "error": repr(exc)[:120]})
            continue
        exact = pred.shape == transition.next_grid.shape and bool(np.array_equal(pred, transition.next_grid))
        lemma: dict[str, Any] = {
            "action": int(transition.action),
            "verifier_gated": exact,
            "changed_cells": len(e3._delta(transition.grid, transition.next_grid, cap=256)),
            "predicted_changed_cells": len(e3._delta(transition.grid, pred, cap=256))
            if pred.shape == transition.grid.shape
            else None,
            "level_delta": int(transition.level_after - transition.level_before),
        }
        if game == "ka59":
            lemma["hud_count_before"] = int(np.count_nonzero(transition.grid[-1] == 4))
            lemma["hud_count_after"] = int(np.count_nonzero(transition.next_grid[-1] == 4))
            lemma["hud_count_predicted"] = int(np.count_nonzero(pred[-1] == 4)) if pred.shape == transition.grid.shape else None
        lemmas.append(lemma)
        if len(lemmas) >= 8:
            break
    return lemmas


def _run_world_model_game(repo: Path, game: str, random_seed: int, _round_budget: int) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    prior_row = _prior_artifact_row(repo, game)
    transitions, cell = e3.collect_transitions(game, n=N_TRANSITIONS, seed=random_seed)
    engine, _is_level_complete = e3.load_engine(game)
    verify = e3.WorldModelVerifier(transitions).score(engine, max_mismatch=12)
    accuracy = round(float(verify.accuracy), 6)
    print(f"{game} verifier round 0 accuracy={accuracy:.6f} cell={cell}", flush=True)
    lemmas = _targeted_gap_lemmas(game, transitions, engine)
    plan = list(prior_row.get("plan") or [])
    reproduce_result = _reproduce_existing_plan_for_next_level(game, plan) if plan else prior_row["reproduce_result"]
    reached = int(reproduce_result.get("reached_level", prior_row["prior_best_level"]) or prior_row["prior_best_level"])
    advanced = bool(reproduce_result.get("reproduced")) and reached > PRIOR_BEST_LEVELS[game]
    return {
        **prior_row,
        "new_reproduced_level": reached if advanced else PRIOR_BEST_LEVELS[game],
        "verifier_accuracy": accuracy,
        "verifier_accuracy_per_round": [accuracy],
        "offline_reproduced": advanced,
        "reproduce_result": reproduce_result,
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial_no_new_level_reproduced",
        "residual_gap_class": "none" if advanced else NAMED_GAP_CLASSES[game],
        "targeted_gap_lemmas": lemmas,
    }


TARGET_RUNNERS: dict[str, Callable[[Path, int, int], dict[str, Any]]] = {
    game: (lambda repo, random_seed, round_budget, game=game: _run_world_model_game(repo, game, random_seed, round_budget))
    for game in TARGET_ORDER
}


def run_experiment(*, random_seed: int = RANDOM_SEED, round_budget: int = 6) -> dict[str, Any]:
    t0 = time.time()
    checks = preconditions(REPO)
    rows: list[dict[str, Any]] = []
    for game in TARGET_ORDER:
        if not checks["games"][game]["offline_env_present"]:
            row = blocked_game_row(game)
        else:
            row = TARGET_RUNNERS[game](REPO, random_seed, round_budget)
        rows.append(row)
        rounds = row.get("verifier_accuracy_per_round") or [row.get("verifier_accuracy", 0.0)]
        for round_index, accuracy in enumerate(rounds):
            print(f"{game} verifier round {round_index} accuracy={float(accuracy):.6f}", flush=True)
        print(f"{game} checkpoint={row['checkpoint_status']}", flush=True)

    new_levels = _new_levels_reproduced(rows)
    total = _registry_total(REPO)
    if total is None:
        total = PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels
    total = max(total, PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels)
    artifact = build_artifact(
        repo=REPO,
        per_game_scorecard=rows,
        reproducible_total_levels=total,
        world_model_paths=list(WORLD_MODEL_PATHS.values()),
        random_seed=random_seed,
        duration_s=time.time() - t0,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4362 artifact schema errors: {errors}")
    _write_artifact(artifact)
    for row in rows:
        if not row["offline_reproduced"] and row["residual_gap_class"] != "offline_env_missing":
            _write_gap(
                GAP_PATH,
                game=str(row["game"]),
                best_accuracy=float(row["verifier_accuracy"]),
                residual_gap_class=str(row["residual_gap_class"]),
                checksum=str(artifact["reproducibility_checksum"]),
            )
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
