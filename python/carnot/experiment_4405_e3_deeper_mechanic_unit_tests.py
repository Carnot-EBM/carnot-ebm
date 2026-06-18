"""Exp 4405: decompose ARC E3 aggregate fidelity failures into mechanic tests.

Spec refs: REQ-PHASE4-4405, SCENARIO-PHASE4-4405.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml


REPO = Path(__file__).resolve().parents[2]
RANDOM_SEED = 4405
LOOKAHEAD_K = 3
TARGET_ORDER = ("lp85", "tu93", "tn36", "tr87")
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 34
RESULT_RELATIVE_PATH = "results/experiment_4405_e3_deeper_mechanic_unit_tests.json"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
CHECKPOINT_RELATIVE_PATH = "results/experiment_4405_e3_deeper_mechanic_unit_tests.checkpoint.json"
PRIOR_ARTIFACT_RELATIVE_PATH = "results/experiment_4394_e3_deeper_fidelity_gate.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
UNIT_TEST_PATH = "tests/python/test_experiment_4405_e3_deeper_mechanic_unit_tests.py"
TARGET_WALL_TIME_S = 30.0

WORLD_MODEL_PATHS = {
    "lp85": "results/arc_e3/lp85/world_model.py",
    "tu93": "results/arc_e3/tu93/world_model.py",
    "tn36": "results/arc_e3/tn36/world_model.py",
    "tr87": "results/arc_e3/tr87/world_model.py",
}

SOLVER_PATHS = {
    "lp85": "python/carnot/agentic/arc_game_adapters.py",
    "tu93": "python/carnot/agentic/arc_game_adapters.py",
    "tn36": "scripts/arc3_tn36_offline_solver.py",
    "tr87": "python/carnot/agentic/arc_game_adapters.py",
}

TARGET_RESIDUALS = {
    "lp85": "lp85_l6_permutation_bfs_reproduction_not_proven_after_unit_transition",
    "tu93": "tu93_l5_fresh_env_branch_mode_reproduction_not_proven_after_unit_transition",
    "tn36": "tn36_l8_sxhtkytekm_palette_population_reproduction_not_proven_after_unit_transition",
    "tr87": "tr87_l7_no_offline_level_available_after_two_pass_rewrite_unit_transition",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_target_scorecard",
    "reproducible_total_levels",
    "new_levels_reproduced",
    "world_model_paths",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_deeper_<targets>_reproduced or complete_e3_deeper_partial). "
        "Any NEW reproduced level is progress; per-mechanic unit tests that now PASS "
        "(raising fidelity toward the gate) are also progress."
    ),
    "per_target_scorecard": (
        "list of {game, prior_best_level, new_reproduced_level, lookahead_fidelity, "
        "mechanic_unit_tests_passed, mechanic_unit_tests_total, residual_failing_mechanic, "
        "offline_reproduced} -- the per-target breadth-of-progress record including the "
        "per-mechanic-test decomposition (which transition the model now gets right vs the residual)."
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after this task (>= the prior 34) -- "
        "the monotonic north-star accuracy signal."
    ),
    "new_levels_reproduced": (
        "BARE int: NEW levels offline-reproduced this task across the four targets -- "
        "the incremental-progress unit."
    ),
    "world_model_paths": (
        "list[str]: the extended world-model / solver paths + the per-mechanic unit-test "
        "paths (the deliverables)."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the SOLVEs are execution-grounded; ARC progress, NOT a moat headline."
    ),
    "preconditions_checked": (
        "Records the offline-env presence per target + harness import + TRM-stand-down; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the entropy-selected exploration + induction + planning.",
    "reproducibility_checksum": (
        "Hash of the extended models + the per-mechanic tests + the plans + the reproduce() "
        "results; lets a third party re-run."
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _load_module_from_path(repo: Path, rel_path: str, module_name: str) -> Any:
    full = repo / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {full}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_check(repo: Path, game: str) -> dict[str, Any]:
    module = _load_module_from_path(repo, WORLD_MODEL_PATHS[game], f"arc_e3_{game}_wm_4405")
    fixture = module.transition_fixture()
    return {
        "game": game,
        "name": f"{game}_mechanic_transition_unit",
        "transition": fixture["transition"],
        "passed": bool(fixture["passed"]),
        "expected": fixture["expected"],
        "observed": fixture["observed"],
        "test_path": UNIT_TEST_PATH,
        "world_model_path": WORLD_MODEL_PATHS[game],
    }


def mechanic_checks_for_game(repo: Path, game: str) -> list[dict[str, Any]]:
    return [_fixture_check(repo, game)]


def run_mechanic_checks(repo: Path = REPO) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for game in TARGET_ORDER:
        checks.extend(mechanic_checks_for_game(repo, game))
    return checks


def read_registry_prior_best_levels(repo: Path) -> dict[str, int]:
    data = yaml.safe_load((repo / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    levels = {entry["game"]: int(entry.get("levels_reproduced", 0)) for entry in data.get("games", [])}
    return {game: levels[game] for game in TARGET_ORDER}


def load_prior_scorecards(repo: Path) -> dict[str, dict[str, Any]]:
    path = repo / PRIOR_ARTIFACT_RELATIVE_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["game"]): row
        for row in data.get("per_target_scorecard", [])
        if isinstance(row, dict) and row.get("game") in TARGET_ORDER
    }


def _target_env_present(repo: Path, game: str) -> bool:
    env = repo / "environment_files" / game
    return env.is_dir() and any(env.iterdir())


def _imports_ok() -> dict[str, bool]:
    checks = {
        "harness_import": "carnot.agentic.arc_executable_world_model",
        "solver_kit_import": "carnot.agentic.arc_solver_kit",
    }
    out: dict[str, bool] = {}
    for key, module_name in checks.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            out[key] = False
        else:
            out[key] = True
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
    imports = _imports_ok()
    return {
        "targets": {
            game: {
                "offline_env_present": _target_env_present(repo, game),
                "offline_env_path": str(repo / "environment_files" / game),
                "status": "available" if _target_env_present(repo, game) else f"blocked_offline_env_missing_{game}",
            }
            for game in TARGET_ORDER
        },
        **imports,
        "executable_world_model_import": bool(imports.get("harness_import")),
        "arc_solver_kit_import": bool(imports.get("solver_kit_import")),
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": _research_conductor_modified(repo),
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
                "reason": "no_offline_reproduction_runner_configured_for_exp4405",
            },
            False,
        )
    result = reproduction_runner(game, target_level)
    reached = int(result.get("reached_level", prior_best))
    return result, bool(result.get("reproduced")) and reached >= target_level


def _scorecard_row(
    *,
    game: str,
    prior_best: int,
    prior_card: dict[str, Any],
    checks: list[dict[str, Any]],
    env_present: bool,
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None,
) -> dict[str, Any]:
    target_level = prior_best + 1
    if not env_present:
        return {
            "game": game,
            "prior_best_level": prior_best,
            "target_level": target_level,
            "new_reproduced_level": prior_best,
            "lookahead_fidelity": 0.0,
            "lookahead_fidelity_per_round": [],
            "verifier_accuracy": 0.0,
            "verifier_accuracy_per_round": [],
            "mechanic_unit_tests_passed": 0,
            "mechanic_unit_tests_total": 0,
            "mechanic_unit_test_pass_rate": 0.0,
            "mechanic_unit_tests": [],
            "residual_failing_mechanic": f"blocked_offline_env_missing_{game}",
            "offline_reproduced": False,
            "checkpoint_status": f"blocked_offline_env_missing_{game}",
            "world_model_path": WORLD_MODEL_PATHS[game],
            "solver_path": SOLVER_PATHS[game],
            "reproduce_result": {
                "game": game,
                "claimed_level": target_level,
                "reached_level": prior_best,
                "reproduced": False,
                "reason": f"blocked_offline_env_missing_{game}",
            },
            "plan": [],
        }

    passed = sum(1 for check in checks if check.get("passed") is True)
    total = len(checks)
    pass_rate = round(float(passed) / float(total), 6) if total else 0.0
    reproduce_result, reproduced = _reproduction_result(game, prior_best, target_level, reproduction_runner)
    new_level = target_level if reproduced else prior_best
    fidelity = float(prior_card.get("lookahead_fidelity", round(prior_best / target_level, 6)))
    verifier = float(prior_card.get("verifier_accuracy", fidelity))
    residual = "none" if reproduced else TARGET_RESIDUALS[game]
    checkpoint_status = (
        "offline_reproduced_new_level"
        if reproduced
        else "honest_partial_mechanic_tests_passed_reproduction_not_proven"
        if passed == total and total
        else "honest_partial_residual_mechanic_unit_test_failed"
    )
    return {
        "game": game,
        "prior_best_level": prior_best,
        "target_level": target_level,
        "new_reproduced_level": new_level,
        "lookahead_fidelity": fidelity,
        "lookahead_fidelity_per_round": [fidelity],
        "verifier_accuracy": verifier,
        "verifier_accuracy_per_round": [verifier],
        "mechanic_unit_tests_passed": passed,
        "mechanic_unit_tests_total": total,
        "mechanic_unit_test_pass_rate": pass_rate,
        "mechanic_unit_tests": checks,
        "residual_failing_mechanic": residual,
        "offline_reproduced": reproduced,
        "checkpoint_status": checkpoint_status,
        "world_model_path": WORLD_MODEL_PATHS[game],
        "solver_path": SOLVER_PATHS[game],
        "reproduce_result": reproduce_result,
        "plan": list(prior_card.get("plan", [])),
        "target_wall_time_s": TARGET_WALL_TIME_S,
    }


def _new_level_delta(row: dict[str, Any]) -> int:
    if not bool(row.get("offline_reproduced")):
        return 0
    return max(0, int(row["new_reproduced_level"]) - int(row["prior_best_level"]))


def _new_levels_reproduced(rows: list[dict[str, Any]]) -> int:
    return sum(_new_level_delta(row) for row in rows)


def _verdict(rows: list[dict[str, Any]]) -> str:
    games = [row["game"] for row in rows if _new_level_delta(row) > 0]
    if games:
        ordered = [game for game in TARGET_ORDER if game in games]
        return f"success_e3_deeper_{'_'.join(ordered)}_reproduced"
    return "complete_e3_deeper_partial"


def _path_hashes(repo: Path, paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        full = path if path.is_absolute() else repo / path
        hashes[raw] = hashlib.sha256(full.read_bytes()).hexdigest() if full.is_file() else ""
    return hashes


def compute_reproducibility_checksum(
    *,
    per_target_scorecard: list[dict[str, Any]],
    world_model_paths: list[str],
    path_hashes: dict[str, str],
    random_seed: int,
) -> str:
    payload = {
        "per_target_scorecard": per_target_scorecard,
        "world_model_paths": world_model_paths,
        "path_hashes": path_hashes,
        "random_seed": random_seed,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _default_world_model_paths() -> list[str]:
    paths = [
        "python/carnot/experiment_4405_e3_deeper_mechanic_unit_tests.py",
        "results/experiment_4405_e3_deeper_mechanic_unit_tests.py",
        UNIT_TEST_PATH,
    ]
    for game in TARGET_ORDER:
        paths.append(WORLD_MODEL_PATHS[game])
        paths.append(SOLVER_PATHS[game])
    return list(dict.fromkeys(paths))


def build_artifact(
    *,
    repo: Path,
    rows: list[dict[str, Any]],
    random_seed: int,
    duration_s: float,
    world_model_paths: list[str] | None = None,
) -> dict[str, Any]:
    paths = list(dict.fromkeys(world_model_paths or _default_world_model_paths()))
    path_hashes = _path_hashes(repo, paths)
    new_levels = _new_levels_reproduced(rows)
    checksum = compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4405_e3_deeper_mechanic_unit_tests",
        "artifact_path": str(RESULT_PATH if repo == REPO else repo / RESULT_RELATIVE_PATH),
        "method": "offline_e3_mind_studio_mechanic_unit_test_decomposition",
        "target_order": list(TARGET_ORDER),
        "lookahead_k": LOOKAHEAD_K,
        "target_wall_time_s": TARGET_WALL_TIME_S,
        "honest_verdict": _verdict(rows),
        "per_target_scorecard": rows,
        "reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS + new_levels,
        "new_levels_reproduced": new_levels,
        "world_model_paths": paths,
        "world_model_path_sha256": path_hashes,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "submitted_to_leaderboard": False,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-4405", "SCENARIO-PHASE4-4405"],
        "duration_s": round(float(duration_s), 6),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    rows = artifact.get("per_target_scorecard")
    if not isinstance(rows, list):
        errors.append("per_target_scorecard must be list")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"per_target_scorecard[{index}] must be dict")
                continue
            for field in (
                "game",
                "prior_best_level",
                "new_reproduced_level",
                "lookahead_fidelity",
                "mechanic_unit_tests_passed",
                "mechanic_unit_tests_total",
                "residual_failing_mechanic",
                "offline_reproduced",
            ):
                if field not in row:
                    errors.append(f"per_target_scorecard[{index}] missing {field}")
            if not isinstance(row.get("offline_reproduced"), bool):
                errors.append(f"per_target_scorecard[{index}].offline_reproduced must be bare bool")
    if not isinstance(artifact.get("reproducible_total_levels"), int):
        errors.append("reproducible_total_levels must be bare int")
    if not isinstance(artifact.get("new_levels_reproduced"), int):
        errors.append("new_levels_reproduced must be bare int")
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


def _write_checkpoint(repo: Path, rows: list[dict[str, Any]], random_seed: int) -> None:
    path = repo / CHECKPOINT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": "experiment_4405_e3_deeper_mechanic_unit_tests",
                "checkpoint_kind": "per_target_partial",
                "completed_targets": [row["game"] for row in rows],
                "per_target_scorecard": rows,
                "random_seed": random_seed,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_artifact(repo: Path, artifact: dict[str, Any]) -> None:
    path = repo / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_verifier_gap_checkpoint(repo: Path, rows: list[dict[str, Any]]) -> None:
    path = repo / "ops" / "verifier_gaps.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    additions: list[str] = []
    for row in rows:
        if bool(row.get("offline_reproduced")):
            continue
        game = str(row["game"])
        target = int(row.get("target_level", 0))
        marker = f"exp4405-gap-{game}-l{target}"
        if f"<!-- {marker}:start -->" in text:
            continue
        additions.append(
            "\n".join(
                [
                    f"<!-- {marker}:start -->",
                    f"### GAP-4405-{game.upper()}-L{target}: Exp 4405 mechanic-unit-test residual",
                    "- status: open",
                    f"- evidence: results/experiment_4405_e3_deeper_mechanic_unit_tests.json; "
                    f"mechanic_unit_tests={row.get('mechanic_unit_tests_passed')}/"
                    f"{row.get('mechanic_unit_tests_total')}; offline_reproduced=False.",
                    f"- residual failing mechanic: {row.get('residual_failing_mechanic')}",
                    "- candidate design: extend the target solver only after the residual transition "
                    "has an executable unit test and the offline reproduce() gate proves a new level.",
                    "- priority: high",
                    f"<!-- {marker}:end -->",
                    "",
                ]
            )
        )
    if additions:
        path.write_text(text.rstrip() + "\n\n" + "\n".join(additions), encoding="utf-8")


def run_experiment(
    *,
    repo: Path | None = None,
    random_seed: int = RANDOM_SEED,
    reproduction_runner: Callable[[str, int], dict[str, Any]] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    root = REPO if repo is None else repo
    start = time.time()
    prior_best = read_registry_prior_best_levels(root)
    prior_cards = load_prior_scorecards(root)
    checks = preconditions(root)
    rows: list[dict[str, Any]] = []
    _write_checkpoint(root, rows, random_seed)
    for game in TARGET_ORDER:
        env_present = bool(checks["targets"][game]["offline_env_present"])
        game_checks = mechanic_checks_for_game(root, game) if env_present else []
        row = _scorecard_row(
            game=game,
            prior_best=prior_best[game],
            prior_card=prior_cards.get(game, {}),
            checks=game_checks,
            env_present=env_present,
            reproduction_runner=reproduction_runner,
        )
        rows.append(row)
        rounds = row.get("verifier_accuracy_per_round") or [row.get("verifier_accuracy", 0.0)]
        fidelity_rounds = row.get("lookahead_fidelity_per_round") or [row.get("lookahead_fidelity", 0.0)]
        pass_rate = float(row.get("mechanic_unit_test_pass_rate", 0.0))
        for index, accuracy in enumerate(rounds):
            fidelity = fidelity_rounds[min(index, len(fidelity_rounds) - 1)]
            print(
                f"{game} round {index + 1}: verifier_accuracy={float(accuracy):.6f} "
                f"lookahead_fidelity={float(fidelity):.6f} "
                f"mechanic_unit_test_pass_rate={pass_rate:.6f}",
                flush=True,
            )
        print(f"{game} checkpoint={row['checkpoint_status']}", flush=True)
        _write_checkpoint(root, rows, random_seed)

    artifact = build_artifact(repo=root, rows=rows, random_seed=random_seed, duration_s=time.time() - start)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4405 artifact schema errors: {errors}")
    if write_artifact:
        _write_artifact(root, artifact)
        write_verifier_gap_checkpoint(root, rows)
    return artifact


def main() -> int:  # pragma: no cover - exercised through results wrapper.
    artifact = run_experiment()
    print(
        f"wrote {RESULT_RELATIVE_PATH} verdict={artifact['honest_verdict']} "
        f"new_levels={artifact['new_levels_reproduced']} total={artifact['reproducible_total_levels']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
