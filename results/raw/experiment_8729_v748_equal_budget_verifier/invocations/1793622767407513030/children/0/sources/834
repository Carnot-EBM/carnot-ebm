"""Exp 4424: single solved-game lookahead-fidelity repair.

Spec refs: REQ-PHASE4-4424, SCENARIO-PHASE4-4424.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GAME = "sc25"
RANDOM_SEED = 4424
LOOKAHEAD_K = 3
RESULT_RELATIVE_PATH = "results/experiment_4424_deeper_solved_game.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
WORLD_MODEL_RELATIVE_PATH = "results/arc_e3/sc25/world_model.py"
UNIT_TEST_PATH = "tests/python/test_experiment_4424_deeper_solved_game.py"
SPEC_REFS = ("REQ-PHASE4-4424", "SCENARIO-PHASE4-4424")

REQUIRED_ARTIFACT_FIELDS = (
    "offline_reproduced",
    "reproduced_levels",
    "verifier_is_oracle",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "offline_reproduced": (
        "Bare bool: true only when arc_solver_kit.reproduce or the injected "
        "equivalent reaches the target level beyond the registry prior best."
    ),
    "reproduced_levels": "Bare int: levels reached by the execution-grounded reproduction gate.",
    "verifier_is_oracle": (
        "Bare bool=true: solve claims are execution-grounded by the offline ARC "
        "environment, not by a heuristic world-model prediction."
    ),
    "honest_verdict": (
        "Terminal-prefixed: success only for +1 reproduced level; complete for "
        "passing per-mechanic tests with a residual solve gap."
    ),
}

ReproductionRunner = Callable[[Path, int, int], Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _load_registry(root: Path) -> Mapping[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, Mapping) else {"games": []}


def read_prior_best_level(root: Path = REPO_ROOT) -> int:
    """REQ-PHASE4-4424: read sc25's authoritative prior from the registry."""
    games = _load_registry(root).get("games")
    if isinstance(games, Sequence):
        for row in games:
            if isinstance(row, Mapping) and row.get("game") == GAME:
                return int(row.get("levels_reproduced") or 0)
    return 0  # pragma: no cover - defensive missing-registry fallback


def _load_world_model(root: Path):
    path = root / WORLD_MODEL_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("sc25_world_model_4424", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _json_runs(runs: Any) -> list[list[int]]:
    return [[int(row), int(col0), int(col1), int(value)] for row, col0, col1, value in runs or ()]


def sc25_l2_hud_cleanup_check(root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-PHASE4-4424: executable unit check for the rollout-derived L2 mismatch."""
    world_model = _load_world_model(root)
    fixture = world_model.l2_first_cast_cleanup_fixture()
    expected = _json_runs(fixture.get("expected_runs"))
    observed = _json_runs(fixture.get("observed_runs"))
    return {
        "game": GAME,
        "name": "l2_first_cast_hud_cleanup",
        "transition": str(fixture.get("transition")),
        "mechanic": "hud_cleanup_on_first_l2_cast",
        "derived_from_rollout_trace": True,
        "rollout_k": LOOKAHEAD_K,
        "before_hash": str(fixture.get("before_hash")),
        "action": int(fixture.get("action") or 0),
        "data_key": list(fixture.get("data_key") or []),
        "expected_runs": expected,
        "observed_runs": observed,
        "passed": bool(fixture.get("passed")) and expected == observed,
        "test_path": UNIT_TEST_PATH,
        "world_model_path": WORLD_MODEL_RELATIVE_PATH,
    }


def reproduction_gap_check(
    reproduce_result: Mapping[str, Any],
    *,
    prior_best_level: int,
    target_level: int,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-4424: residual solve gate reported as its own row."""
    reached = int(reproduce_result.get("reached_level") or 0)
    reproduced = bool(reproduce_result.get("reproduced")) and reached >= target_level
    return {
        "game": GAME,
        "name": "l2_complete_route_reproduction",
        "transition": "sc25:L2:complete_route_reproduction",
        "mechanic": "plan_reaches_target_level",
        "derived_from_rollout_trace": False,
        "passed": reproduced,
        "gap_class": "none" if reproduced else "sc25_l2_route_search_still_missing_after_hud_cleanup",
        "expected": f"arc_solver_kit.reproduce reaches L{target_level}",
        "observed": f"reached L{reached} from prior L{prior_best_level}",
    }


def default_reproduction_runner(root: Path, prior_best_level: int, target_level: int) -> Mapping[str, Any]:  # pragma: no cover
    """Replay the prior sc25 L1 plan through the offline reproduction gate."""
    from carnot.agentic import arc_solver_kit
    from carnot.experiment_4341_e3_sc25_reproduction import L1_SOLUTION_LABELS, _apply_sc25_label

    result = arc_solver_kit.reproduce(
        GAME,
        L1_SOLUTION_LABELS,
        _apply_sc25_label,
        warmup_label="warmup",
        claimed_level=target_level,
    )
    return {
        **dict(result),
        "prior_best_level": prior_best_level,
        "solution_labels": list(L1_SOLUTION_LABELS),
        "reproduction_scope": "prior_l1_path_replayed_against_l2_target",
        "root": str(root),
    }


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("success:", "complete:", "blocked:", "failed:"))


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _residual_gap(checks: Sequence[Mapping[str, Any]]) -> str:
    for check in checks:
        if check.get("passed") is False:
            return str(check.get("gap_class") or check.get("transition") or "unknown_mechanic_gap")
    return "none"


def _honest_verdict(offline_reproduced: bool, target_level: int) -> str:
    if offline_reproduced:
        return f"success: {GAME}_L{target_level}_offline_reproduced"
    return "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap"


def compute_reproducibility_checksum(artifact_payload: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "game": artifact_payload.get("game"),
            "prior_best_level": artifact_payload.get("prior_best_level"),
            "target_level": artifact_payload.get("target_level"),
            "per_mechanic_tests": artifact_payload.get("per_mechanic_tests"),
            "reproduce_result": artifact_payload.get("reproduce_result"),
            "residual_failing_mechanic": artifact_payload.get("residual_failing_mechanic"),
            "random_seed": artifact_payload.get("random_seed"),
            "world_model_sha256": artifact_payload.get("world_model_sha256"),
            "unit_test_path": UNIT_TEST_PATH,
        }
    )


def preconditions(root: Path) -> dict[str, Any]:
    env_path = root / "environment_files" / GAME
    return {
        "selected_game": GAME,
        "offline_env_present": env_path.is_dir() and any(env_path.iterdir()),
        "offline_env_path": str(env_path),
        "registry_path": REGISTRY_RELATIVE_PATH,
        "world_model_path": WORLD_MODEL_RELATIVE_PATH,
        "unit_test_path": UNIT_TEST_PATH,
        "leaderboard_submission": False,
        "trm_training_stood_down": True,
    }


def build_artifact(
    *,
    root: Path,
    prior_best_level: int,
    target_level: int,
    mechanic_checks: Sequence[Mapping[str, Any]],
    reproduce_result: Mapping[str, Any],
) -> dict[str, Any]:
    reached = int(reproduce_result.get("reached_level") or 0)
    offline_reproduced = bool(reproduce_result.get("reproduced")) and reached >= target_level
    reproduced_levels = reached
    new_levels = max(0, reproduced_levels - prior_best_level) if offline_reproduced else 0
    checks = [dict(check) for check in mechanic_checks]
    residual_gap = _residual_gap(checks)
    pass_count = sum(1 for check in checks if check.get("passed") is True)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4424_deeper_solved_game",
        "schema": "carnot.exp4424.deeper_solved_game.v1",
        "game": GAME,
        "selected_target_reason": "sc25 next because lp85/tu93 timed out and local tn36 has no L8 fixture",
        "prior_best_level": int(prior_best_level),
        "target_level": int(target_level),
        "lookahead_k": LOOKAHEAD_K,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "new_levels_reproduced": new_levels,
        "verifier_is_oracle": True,
        "honest_verdict": _honest_verdict(offline_reproduced, target_level),
        "per_mechanic_tests": checks,
        "per_mechanic_tests_passed": pass_count,
        "per_mechanic_tests_total": len(checks),
        "per_mechanic_test_pass_rate": round(pass_count / max(1, len(checks)), 6),
        "residual_failing_mechanic": residual_gap,
        "reproduce_result": dict(reproduce_result),
        "world_model_paths": [
            WORLD_MODEL_RELATIVE_PATH,
            "python/carnot/experiment_4424_deeper_solved_game.py",
            UNIT_TEST_PATH,
        ],
        "world_model_sha256": _file_sha256(root / WORLD_MODEL_RELATIVE_PATH),
        "preconditions_checked": preconditions(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "submitted_to_leaderboard": False,
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """REQ-PHASE4-4424: typed validation for the terminal artifact."""
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be an int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not isinstance(artifact.get("per_mechanic_tests"), list) or not artifact.get("per_mechanic_tests"):
        errors.append("per_mechanic_tests must include at least one row")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run_experiment(
    root: Path = REPO_ROOT,
    *,
    reproduction_runner: ReproductionRunner = default_reproduction_runner,
    mechanic_checks: Sequence[Mapping[str, Any]] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-4424: build and optionally persist the terminal artifact."""
    prior = read_prior_best_level(root)
    target = prior + 1
    reproduce_result = reproduction_runner(root, prior, target)
    checks = list(mechanic_checks) if mechanic_checks is not None else [sc25_l2_hud_cleanup_check(root)]
    if mechanic_checks is None:  # pragma: no cover - covered by the operator artifact run
        checks.append(
            reproduction_gap_check(
                reproduce_result,
                prior_best_level=prior,
                target_level=target,
            )
        )
    artifact = build_artifact(
        root=root,
        prior_best_level=prior,
        target_level=target,
        mechanic_checks=checks,
        reproduce_result=reproduce_result,
    )
    if write:
        write_artifact(root, artifact)
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run_experiment()
    print(f"{artifact['honest_verdict']} wrote {RESULT_RELATIVE_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
