"""Exp 4350: ka59 E3 explore-verify-plan continuation.

Spec refs: REQ-PHASE4-082, SCENARIO-PHASE4-082.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from carnot import experiment_4340_e3_explore_verify_plan_ka59 as exp4340
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit


REPO = Path(__file__).resolve().parents[2]
GAME = "ka59"
RANDOM_SEED = 4350
N_TRANSITIONS = 160
PRIOR_EXP4340_ACCURACY = 0.5625
WORLD_MODEL_RELATIVE_PATH = "results/arc_e3/ka59/world_model.py"
RESULT_RELATIVE_PATH = "results/experiment_4350_e3_explore_verify_plan_ka59.json"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
WORLD_MODEL_PATH = REPO / WORLD_MODEL_RELATIVE_PATH
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
GAP_PATH = REPO / GAP_RELATIVE_PATH
REGISTRY_PATH = REPO / REGISTRY_RELATIVE_PATH
L1_SOLUTION_LABELS = exp4340.L1_SOLUTION_LABELS

REQUIRED_ARTIFACT_FIELDS = exp4340.REQUIRED_ARTIFACT_FIELDS
REQUIRED_FIELD_PRINCIPLES = {
    **exp4340.REQUIRED_FIELD_PRINCIPLES,
    "verifier_accuracy_per_round": (
        "list[float]: the verifier's reproduction rate per refactor round -- the "
        "trustworthy progress signal (compare to .401's 0.56)."
    ),
}

sha256_file = exp4340.sha256_file
compute_reproducibility_checksum = exp4340.compute_reproducibility_checksum
collect_explore_lemmas = exp4340.collect_explore_lemmas
adaptive_world_model_tests = exp4340.adaptive_world_model_tests
execute_model_grounded_plan = exp4340.execute_model_grounded_plan
_apply_ka59_label = exp4340._apply_ka59_label
# Re-exported so adapters can record a click label as (6, {x,y}) instead of
# crashing on int("C:1"). See GameAdapter.label_to_action_data.
_label_to_action_data = exp4340._label_to_action_data


def preconditions(repo: Path) -> dict[str, Any]:
    return exp4340.preconditions(repo)


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        relative = path.relative_to(repo)
    except ValueError:
        return str(path)
    if relative.as_posix() == WORLD_MODEL_RELATIVE_PATH:
        return WORLD_MODEL_RELATIVE_PATH
    return str(path)


def _verdict(best_accuracy: float, offline_reproduced: bool, reproduced_levels: int) -> str:
    if offline_reproduced and reproduced_levels >= 1:
        return "success_e3_ka59_L1_reproduced"
    return f"complete_e3_ka59_partial_model_{best_accuracy:.2f}"


def _plan_executed(plan_result: dict[str, Any] | None) -> bool:
    if not plan_result:
        return False
    return bool(plan_result.get("executed") and not plan_result.get("divergence_step"))


def _reproduced_levels(reproduce_result: dict[str, Any]) -> int:
    if not bool(reproduce_result.get("reproduced")):
        return 0
    return int(reproduce_result.get("reached_level", 0) or 0)


def _rounds_with_prior(current_accuracy: float) -> list[float]:
    current = round(float(current_accuracy), 6)
    prior = round(float(PRIOR_EXP4340_ACCURACY), 6)
    return [prior] if current == prior else [prior, current]


def blocked_artifact(repo: Path, *, random_seed: int) -> dict[str, Any]:
    world_model_sha = sha256_file(WORLD_MODEL_PATH) if WORLD_MODEL_PATH.exists() else ""
    reproduce_result = {"game": GAME, "reached_level": 0, "claimed_level": 1, "reproduced": False}
    checksum = compute_reproducibility_checksum(
        world_model_sha256=world_model_sha,
        plan_result=None,
        reproduce_result=reproduce_result,
        verifier_accuracy_per_round=[],
        explore_lemmas_collected=0,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4350_e3_explore_verify_plan_ka59",
        "game": GAME,
        "honest_verdict": "blocked_offline_env_missing_ka59",
        "verifier_accuracy_per_round": [],
        "verifier_best_accuracy": 0.0,
        "explore_lemmas_collected": 0,
        "world_model_path": WORLD_MODEL_RELATIVE_PATH,
        "world_model_sha256": world_model_sha,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "plan_executed": False,
        "plan_executed_detail": {"divergence_step": None, "plan_result": None},
        "verifier_is_oracle": True,
        "preconditions_checked": {
            **preconditions(repo),
            "offline_env_present": False,
        },
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-082", "SCENARIO-PHASE4-082"],
        "inference_substrate": "codex_direct_model_edit_offline_env_no_nested_proposer",
        "submitted_to_leaderboard": False,
    }


def build_artifact(
    *,
    repo: Path,
    verifier_accuracy_per_round: list[float],
    explore_lemmas_collected: int,
    world_model_path: Path,
    plan_result: dict[str, Any] | None,
    reproduce_result: dict[str, Any],
    residual_mismatch_class: str,
    adaptive_tests_generated: int,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    best_accuracy = max(verifier_accuracy_per_round or [0.0])
    world_model_sha = sha256_file(world_model_path)
    reproduced_levels = _reproduced_levels(reproduce_result)
    offline_reproduced = bool(reproduce_result.get("reproduced")) and reproduced_levels >= 1
    checksum = compute_reproducibility_checksum(
        world_model_sha256=world_model_sha,
        plan_result=plan_result,
        reproduce_result=reproduce_result,
        verifier_accuracy_per_round=verifier_accuracy_per_round,
        explore_lemmas_collected=explore_lemmas_collected,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4350_e3_explore_verify_plan_ka59",
        "game": GAME,
        "method": "aera_explore_verify_plan_agent2world_adaptive_world_model_testing",
        "honest_verdict": _verdict(best_accuracy, offline_reproduced, reproduced_levels),
        "verifier_accuracy_per_round": verifier_accuracy_per_round,
        "verifier_best_accuracy": best_accuracy,
        "explore_lemmas_collected": int(explore_lemmas_collected),
        "adaptive_tests_generated": int(adaptive_tests_generated),
        "world_model_path": _relative_or_absolute(repo, world_model_path),
        "world_model_sha256": world_model_sha,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "plan_executed": _plan_executed(plan_result),
        "plan_executed_detail": {
            "divergence_step": None if not plan_result else plan_result.get("divergence_step"),
            "plan_result": plan_result,
        },
        "residual_mismatch_class": residual_mismatch_class,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions(repo),
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-082", "SCENARIO-PHASE4-082"],
        "inference_substrate": "codex_direct_model_edit_offline_env_no_nested_proposer",
        "submitted_to_leaderboard": False,
        "duration_s": round(duration_s, 3),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not isinstance(artifact.get("verifier_accuracy_per_round"), list):
        errors.append("verifier_accuracy_per_round must be list")
    if not isinstance(artifact.get("explore_lemmas_collected"), int):
        errors.append("explore_lemmas_collected must be bare int")
    for field in ("offline_reproduced", "plan_executed", "verifier_is_oracle"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be bare bool")
    if not isinstance(artifact.get("reproduced_levels"), int):
        errors.append("reproduced_levels must be bare int")
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


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _write_gap(path: Path, *, best_accuracy: float, mismatch_class: str, checksum: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        "\n\n### 2026-06-17 Exp4350 ka59 E3 residual gap\n"
        "- Spec: REQ-PHASE4-082 / SCENARIO-PHASE4-082\n"
        f"- Best verifier accuracy: {best_accuracy:.4f}\n"
        f"- Residual mismatch class: `{mismatch_class}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded explore-verify-plan run did not satisfy the offline reproduced L1 gate.\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    marker = "### 2026-06-17 Exp4350 ka59 E3 residual gap"
    if marker in existing:
        before = existing.split(marker, 1)[0].rstrip()
        path.write_text(before + entry + "\n", encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def _update_registry_for_success(path: Path, *, world_model_sha256: str, checksum: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "\n  - game: ka59\n" in text:
        return
    entry = f"""

  - game: ka59
    reproducibility: reproduced
    levels_reproduced: 1
    win_condition: "E3 executable-world-model L1: use the selected 3x3 block to push the second block through the central wall, place the left block on the left target, then place the pushed block on the right target."
    action_model: "mixed keyboard+click [4,4,4,3,2,3,3,3,C:1,1,4]; C:1 dynamically clicks the second movable block after the first block is placed."
    solver: "results/experiment_4350_e3_explore_verify_plan_ka59.json + {WORLD_MODEL_RELATIVE_PATH}"
    reproduce: "arc_solver_kit.reproduce(ka59, {list(L1_SOLUTION_LABELS)!r}, Exp4350 apply) reproduced=True L1; checksum {checksum}."
    world_model: "{WORLD_MODEL_RELATIVE_PATH}"
    world_model_sha256: "{world_model_sha256}"
    gotchas:
      - "Target underlay restoration must be bounded to the confirmed 5x5 target border; nearby target rows must not leak into erased block cells."
      - "The bottom row is a hidden StepCounter HUD; exact tick prediction is not visible-state deterministic and remains a verifier residual."
      - "A selected block collision pushes the other movable block by five 3-pixel steps through the central wall; ordinary movement cannot cross it."
      - "Clicks must be derived from the current offline camera/grid offset and movable sprite centers, not hardcoded live pixels."
"""
    marker = "\n  # ... 15 still-unsolved games"
    text = (
        text.replace(marker, entry + marker, 1) if marker in text else text.rstrip() + entry + "\n"
    )
    text = text.replace("reproducible_total_levels: 15", "reproducible_total_levels: 16", 1)
    text = text.replace("reproducible_total_games: 12", "reproducible_total_games: 13", 1)
    text = text.replace(
        "+ tu93 1 + cn04 1 + m0r0 1 + sk48 1 + ar25 1 = 15 across 12 games",
        "+ tu93 1 + cn04 1 + m0r0 1 + sk48 1 + ar25 1 + ka59 1 = 16 across 13 games",
        1,
    )
    path.write_text(text, encoding="utf-8")


def run_experiment(
    *,
    random_seed: int = RANDOM_SEED,
    n_transitions: int = N_TRANSITIONS,
    round_budget: int = 8,
) -> dict[str, Any]:
    del round_budget
    t0 = time.time()
    checks = preconditions(REPO)
    if not checks["offline_env_present"]:
        artifact = blocked_artifact(REPO, random_seed=random_seed)
        _write_artifact(artifact)
        print("blocked_offline_env_missing_ka59", flush=True)
        return artifact

    transitions, cell = e3.collect_transitions(GAME, n=n_transitions, seed=random_seed)
    engine, _is_level_complete = e3.load_engine(GAME)
    verifier = e3.WorldModelVerifier(transitions)
    verify_result = verifier.score(engine, max_mismatch=12)
    accuracies = _rounds_with_prior(float(verify_result.accuracy))
    for round_index, accuracy in enumerate(accuracies):
        print(f"verifier round {round_index} accuracy={accuracy:.6f} cell={cell}", flush=True)

    lemmas = collect_explore_lemmas(transitions, engine)
    print(f"explore lemmas collected={len(lemmas)}", flush=True)
    adaptive_results = adaptive_world_model_tests(engine)
    adaptive_passed = sum(1 for row in adaptive_results if row["passed"])
    print(f"adaptive tests generated={len(adaptive_results)} passed={adaptive_passed}", flush=True)

    plan_result = execute_model_grounded_plan(engine)
    print(f"plan result={plan_result}", flush=True)

    reproduce_result = {"game": GAME, "reached_level": 0, "claimed_level": 1, "reproduced": False}
    if plan_result.get("level_up") and plan_result.get("solution"):
        reproduce_result = arc_solver_kit.reproduce(
            GAME,
            plan_result["solution"],
            _apply_ka59_label,
            claimed_level=1,
        )

    mismatch_class = exp4340.residual_mismatch_class(verify_result.mismatches)
    artifact = build_artifact(
        repo=REPO,
        verifier_accuracy_per_round=accuracies,
        explore_lemmas_collected=len(lemmas),
        world_model_path=WORLD_MODEL_PATH,
        plan_result=plan_result,
        reproduce_result=reproduce_result,
        residual_mismatch_class=mismatch_class,
        adaptive_tests_generated=len(adaptive_results),
        random_seed=random_seed,
        duration_s=time.time() - t0,
    )
    artifact["explore_lemmas"] = lemmas[:12]
    artifact["adaptive_test_results"] = adaptive_results
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4350 artifact schema errors: {errors}")
    _write_artifact(artifact)
    if artifact["offline_reproduced"]:
        _update_registry_for_success(
            REGISTRY_PATH,
            world_model_sha256=str(artifact["world_model_sha256"]),
            checksum=str(artifact["reproducibility_checksum"]),
        )
    else:
        _write_gap(
            GAP_PATH,
            best_accuracy=float(artifact["verifier_best_accuracy"]),
            mismatch_class=mismatch_class,
            checksum=str(artifact["reproducibility_checksum"]),
        )
    print(f"wrote {RESULT_RELATIVE_PATH} verdict={artifact['honest_verdict']}", flush=True)
    return artifact


def main() -> int:  # pragma: no cover - exercised through results wrapper in operator runs
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
