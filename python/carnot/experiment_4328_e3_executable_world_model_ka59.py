"""Exp 4328: offline E3 executable-world-model attempt on ARC-AGI-3 ka59.

Spec refs: REQ-PHASE4-075, SCENARIO-PHASE4-075.
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
ENV_DIR = REPO / "environment_files"
GAME = "ka59"
RANDOM_SEED = 4328
N_TRANSITIONS = 160
WORLD_MODEL_RELATIVE_PATH = "results/arc_e3/ka59/world_model.py"
RESULT_RELATIVE_PATH = "results/experiment_4328_e3_executable_world_model_ka59.json"
GAP_RELATIVE_PATH = "ops/verifier_gaps.md"
WORLD_MODEL_PATH = REPO / WORLD_MODEL_RELATIVE_PATH
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
GAP_PATH = REPO / GAP_RELATIVE_PATH

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_accuracy_per_round",
    "world_model_path",
    "world_model_sha256",
    "offline_reproduced",
    "reproduced_levels",
    "plan_executed",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_e3_ka59_L1_reproduced or "
        "complete_e3_ka59_partial_model_<acc>). A reproduced L1 and an honest partial "
        "are BOTH progress."
    ),
    "verifier_accuracy_per_round": (
        "list[float]: the verifier's reproduction rate per refactor round -- the "
        "trustworthy progress signal for the induced model."
    ),
    "world_model_path": (
        "results/arc_e3/ka59/world_model.py -- the induced model IS the deliverable."
    ),
    "world_model_sha256": "Hash of the induced world model -- auditable/reproducible.",
    "offline_reproduced": (
        "BARE bool: the real env reaches L1 via the induced-model plan, re-gated -- only "
        "reproduced levels count."
    ),
    "reproduced_levels": (
        "BARE int: levels offline-reproduced on ka59 (target >=1) -- the +1 "
        "incremental-progress unit."
    ),
    "plan_executed": (
        "BARE bool + divergence_step: execution-grounded confirmation; halt-on-divergence."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the SOLVE is execution-grounded (real env defines the win); "
        "ARC progress, NOT a moat headline."
    ),
    "preconditions_checked": (
        "Records the offline-env presence + harness import + TRM-stand-down; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the induction + planning.",
    "reproducibility_checksum": (
        "Hash of the world model + the plan + the reproduce() result; lets a third party re-run."
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_reproducibility_checksum(
    *,
    world_model_sha256: str,
    plan_result: dict[str, Any] | None,
    reproduce_result: dict[str, Any],
    verifier_accuracy_per_round: list[float],
    random_seed: int,
) -> str:
    payload = {
        "world_model_sha256": world_model_sha256,
        "plan_result": plan_result,
        "reproduce_result": reproduce_result,
        "verifier_accuracy_per_round": verifier_accuracy_per_round,
        "random_seed": random_seed,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def preconditions(repo: Path) -> dict[str, Any]:
    env = repo / "environment_files" / GAME
    return {
        "offline_env_present": env.is_dir() and any(env.iterdir()),
        "offline_env_path": str(env),
        "harness_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }


def _verdict(best_accuracy: float, offline_reproduced: bool, reproduced_levels: int) -> str:
    if offline_reproduced and reproduced_levels >= 1:
        return "success_e3_ka59_L1_reproduced"
    return f"complete_e3_ka59_partial_model_{best_accuracy:.2f}"


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


def _plan_executed(plan_result: dict[str, Any] | None) -> bool:
    if not plan_result:
        return False
    return bool(plan_result.get("executed") and not plan_result.get("divergence_step"))


def _divergence_step(plan_result: dict[str, Any] | None) -> Any:
    if not plan_result:
        return None
    return plan_result.get("divergence_step")


def _reproduced_levels(reproduce_result: dict[str, Any]) -> int:
    if not bool(reproduce_result.get("reproduced")):
        return 0
    return int(reproduce_result.get("reached_level", 0) or 0)


def blocked_artifact(repo: Path, *, random_seed: int) -> dict[str, Any]:
    world_model_sha = sha256_file(WORLD_MODEL_PATH) if WORLD_MODEL_PATH.exists() else ""
    reproduce_result = {"game": GAME, "reached_level": 0, "claimed_level": 1, "reproduced": False}
    checksum = compute_reproducibility_checksum(
        world_model_sha256=world_model_sha,
        plan_result=None,
        reproduce_result=reproduce_result,
        verifier_accuracy_per_round=[],
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4328_e3_executable_world_model_ka59",
        "game": GAME,
        "honest_verdict": "blocked_offline_env_missing_ka59",
        "verifier_accuracy_per_round": [],
        "verifier_best_accuracy": 0.0,
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
        "inference_substrate": "codex_direct_model_edit_offline_env_no_nested_proposer",
        "submitted_to_leaderboard": False,
    }


def build_artifact(
    *,
    repo: Path,
    verifier_accuracy_per_round: list[float],
    world_model_path: Path,
    plan_result: dict[str, Any] | None,
    reproduce_result: dict[str, Any],
    residual_mismatch_class: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    best_accuracy = max(verifier_accuracy_per_round or [0.0])
    world_model_sha = sha256_file(world_model_path)
    reproduced_levels = _reproduced_levels(reproduce_result)
    offline_reproduced = bool(reproduce_result.get("reproduced")) and reproduced_levels >= 1
    plan_executed = _plan_executed(plan_result)
    checksum = compute_reproducibility_checksum(
        world_model_sha256=world_model_sha,
        plan_result=plan_result,
        reproduce_result=reproduce_result,
        verifier_accuracy_per_round=verifier_accuracy_per_round,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4328_e3_executable_world_model_ka59",
        "game": GAME,
        "method": "executable_world_model_verify_plan_reproduce",
        "honest_verdict": _verdict(best_accuracy, offline_reproduced, reproduced_levels),
        "verifier_accuracy_per_round": verifier_accuracy_per_round,
        "verifier_best_accuracy": best_accuracy,
        "world_model_path": _relative_or_absolute(repo, world_model_path),
        "world_model_sha256": world_model_sha,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "plan_executed": plan_executed,
        "plan_executed_detail": {
            "divergence_step": _divergence_step(plan_result),
            "plan_result": plan_result,
        },
        "residual_mismatch_class": residual_mismatch_class,
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
    if not isinstance(artifact.get("verifier_accuracy_per_round"), list):
        errors.append("verifier_accuracy_per_round must be list")
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


def _write_gap(path: Path, *, best_accuracy: float, mismatch_class: str, checksum: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        "\n\n### 2026-06-17 Exp4328 ka59 E3 residual gap\n"
        "- Spec: REQ-PHASE4-075 / SCENARIO-PHASE4-075\n"
        f"- Best verifier accuracy: {best_accuracy:.4f}\n"
        f"- Residual mismatch class: `{mismatch_class}`\n"
        f"- Reproducibility checksum: `{checksum}`\n"
        "- Gap: bounded executable-world-model run did not satisfy the offline reproduced L1 gate.\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    marker = "### 2026-06-17 Exp4328 ka59 E3 residual gap"
    if marker in existing:
        before = existing.split(marker, 1)[0].rstrip()
        path.write_text(before + entry + "\n", encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _apply_noop(_env: Any, _label: str, frame: Any) -> Any:
    return frame


def run_experiment(*, random_seed: int = RANDOM_SEED, n_transitions: int = N_TRANSITIONS) -> dict[str, Any]:
    t0 = time.time()
    checks = preconditions(REPO)
    if not checks["offline_env_present"]:
        artifact = blocked_artifact(REPO, random_seed=random_seed)
        _write_artifact(artifact)
        print("blocked_offline_env_missing_ka59", flush=True)
        return artifact

    transitions, cell = e3.collect_transitions(GAME, n=n_transitions, seed=random_seed)
    verifier = e3.WorldModelVerifier(transitions)
    engine, is_level_complete = e3.load_engine(GAME)
    verify_result = verifier.score(engine)
    accuracies = [round(float(verify_result.accuracy), 6)]
    print(f"verifier round 0 accuracy={accuracies[-1]:.6f} cell={cell}", flush=True)

    plan_result = e3.plan_and_execute(GAME, engine, is_level_complete)
    print(f"plan result={plan_result}", flush=True)

    reproduce_result = {"game": GAME, "reached_level": 0, "claimed_level": 1, "reproduced": False}
    if plan_result.get("level_up"):
        reproduce_result = arc_solver_kit.reproduce(
            GAME,
            plan_result.get("solution", []),
            _apply_noop,
            claimed_level=1,
        )

    mismatch_class = residual_mismatch_class(verify_result.mismatches)
    artifact = build_artifact(
        repo=REPO,
        verifier_accuracy_per_round=accuracies,
        world_model_path=WORLD_MODEL_PATH,
        plan_result=plan_result,
        reproduce_result=reproduce_result,
        residual_mismatch_class=mismatch_class,
        random_seed=random_seed,
        duration_s=time.time() - t0,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp4328 artifact schema errors: {errors}")
    _write_artifact(artifact)
    if not artifact["offline_reproduced"]:
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
