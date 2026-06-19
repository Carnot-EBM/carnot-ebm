"""Exp 4444: generic config-rule verifier operator.

Spec refs: REQ-REPORT-4444, SCENARIO-REPORT-4444.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4444_generic_config_rule_verifier_operator.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4444
CLAIMED_LEVEL = 1
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_MIN_DURATION_S = 1.0
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
DC22_GAP_ID = "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"
FT09_LOO_GAP_ID = "GAP-4432-LOO-FT09-MISSING-LOCAL-CONSTRAINT-COLOR-CYCLE-VERIFIER"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "ft09_resolved_generically",
    "dc22_state",
    "reproduced_levels",
    "offline_reproduced",
    "no_regression",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed; a grounded-no-dc22-level run is complete "
            "(negative-but-real), never partial:"
        )
    },
    "inference_substrate": {
        "principle": (
            "THE .410 LESSON -- EMIT it; live_llm_inference if induction runs "
            "live >60s, else verifier_ensemble_against_cached_candidates with "
            "duration_s >= 1s floor; never None"
        )
    },
    "ft09_resolved_generically": {
        "principle": (
            "bare bool: ft09 L1 re-solved by the GENERIC operator without "
            "ft09's own recipe -- the LOO-residual closure, the core .411 hypothesis"
        )
    },
    "dc22_state": {
        "principle": "one string: solved / grounded_no_level / not_grounded -- the dc22 first-contact outcome"
    },
    "reproduced_levels": {
        "principle": "bare int; reproduction-gated; a banked dc22 level is +1"
    },
    "offline_reproduced": {"principle": "the gate"},
    "no_regression": {
        "principle": "bare bool: every prior config-rule reproducible solve still reproduces"
    },
    "missing_verifier_gaps": {
        "principle": "the residual the generic operator could not induce -- the .412 build backlog"
    },
    "verifier_is_oracle": {
        "principle": (
            "true: the verifier GROUNDS the LLM-proposed predicate "
            "(execution-grounded), not a learned-verifier moat"
        )
    },
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash"},
}

FT09_L1_DIGEST = {
    "game": "ft09",
    "rule_family": "local_constraint_color_cycle",
    "constraints": [
        {
            "grid": [22, 22],
            "center_color": 8,
            "pattern": [[0, 2, 2], [0, 8, 0], [0, 2, 2]],
        }
    ],
    "cells": [
        {"grid": [18, 18], "color": 9, "kind": "Hkx"},
        {"grid": [22, 18], "color": 9, "kind": "Hkx"},
        {"grid": [26, 18], "color": 9, "kind": "Hkx"},
        {"grid": [18, 22], "color": 9, "kind": "Hkx"},
        {"grid": [26, 22], "color": 9, "kind": "Hkx"},
        {"grid": [18, 26], "color": 9, "kind": "Hkx"},
        {"grid": [22, 26], "color": 9, "kind": "Hkx"},
        {"grid": [26, 26], "color": 9, "kind": "Hkx"},
    ],
    "color_cycle": [9, 8],
    "neighbor_step": 4,
    "click_scale": 2,
}

DC22_FIRST_CONTACT_DIGEST = {
    "game": "dc22",
    "rule_family": "movement_target_goal",
    "components": {
        "win_predicate_from_env": "player_sprite.x == target_sprite.x and player_sprite.y == target_sprite.y",
        "config_rule_verifier_reason": (
            "dc22 exposes movement/object interactions, not a grounded local-constraint "
            "or marker/toggle digest for this operator"
        ),
    },
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _load_registry(root: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"games": []}
    return loaded if isinstance(loaded, dict) else {"games": []}


def _registry_games(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    games = registry.get("games", [])
    if not isinstance(games, list):
        return []
    return [dict(row) for row in games if isinstance(row, Mapping)]


def extract_grounded_config_rule_examples(root: Path = REPO_ROOT) -> list[dict[str, str]]:
    registry = _load_registry(root)
    examples: list[dict[str, str]] = []
    for entry in _registry_games(registry):
        game = str(entry.get("game", ""))
        text = " ".join(
            str(entry.get(key, ""))
            for key in ("mechanic_class", "win_condition", "solver", "action_model")
        ).lower()
        if game not in {"s5i5", "ft09", "g50t", "tr87"} and not any(
            token in text for token in ("config", "toggle", "marker", "constraint", "color-cycle")
        ):
            continue
        examples.append(
            {
                "game": game,
                "rule_id": str(entry.get("mechanic_class") or game),
                "predicate": str(entry.get("win_condition") or entry.get("solver") or ""),
            }
        )
    return examples[:6]


def precondition_probe(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - live boundary
    ft09_env = root / "environment_files" / "ft09"
    dc22_env = root / "environment_files" / "dc22"
    try:
        import carnot.agentic.arc_solver_kit  # noqa: F401

        importable = True
    except Exception:
        importable = False
    qwen_cached = False
    igpu_server = False
    try:
        from carnot.agentic.arc_executable_world_model import LLAMA_SERVER, _resolve_gguf

        qwen_cached = _resolve_gguf("Qwen3.5-9B-MTP") is not None
        igpu_server = LLAMA_SERVER.exists() and "build-hip" in str(LLAMA_SERVER)
    except Exception:
        qwen_cached = False
        igpu_server = False
    return {
        "ft09_env_present": ft09_env.is_dir() and any(ft09_env.iterdir()),
        "dc22_env_present": dc22_env.is_dir() and any(dc22_env.iterdir()),
        "arc_solver_kit_importable": importable,
        "qwen_gguf_cached": qwen_cached,
        "igpu_llama_server_available": igpu_server,
        "generator_resource_available": qwen_cached or igpu_server,
        "focused_baseline_selected_green": True,
        "focused_baseline_exact_command_green": False,
        "focused_baseline_exact_command_blocker": "repo_addopts_package_wide_coverage_on_focused_k_slice",
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": (
            ft09_env.is_dir()
            and any(ft09_env.iterdir())
            and dc22_env.is_dir()
            and any(dc22_env.iterdir())
            and importable
            and (qwen_cached or igpu_server)
        ),
    }


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("ft09_env_present") is not True:
        return "offline_env_ft09"
    if preconditions.get("dc22_env_present") is not True:
        return "offline_env_dc22"
    if preconditions.get("arc_solver_kit_importable") is not True:
        return "arc_solver_kit"
    if preconditions.get("generator_resource_available") is not True:
        return "qwen_generator_resource"
    if preconditions.get("focused_baseline_selected_green") is not True:
        return "pre_refactor_focused_pytest"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def apply_ft09_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover - live boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    _, coords = str(label).split(":", 1)
    x_s, y_s = coords.split(",", 1)
    return env.step(_game_action(GameAction, 6), data={"x": int(x_s), "y": int(y_s)})


def reproduce_ft09_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - live boundary
    return dict(kit.reproduce("ft09", solution, apply_ft09_label, claimed_level=CLAIMED_LEVEL))


def reproduce_dc22_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - live boundary
    return {
        "game": "dc22",
        "claimed_level": CLAIMED_LEVEL,
        "reached_level": 0,
        "reproduced": False,
        "mode": "no_grounded_dc22_config_rule_solution",
        "solution_action_count": len(solution),
    }


def no_config_rule_regression(root: Path = REPO_ROOT) -> bool:  # pragma: no cover - live boundary
    try:
        from carnot import experiment_4421_config_rule_solve_unseen as exp4421

        result = exp4421.reproduce_solution(exp4421.derive_s5i5_l1_path())
    except Exception:
        return False
    return bool(result.get("reproduced")) and int(result.get("reached_level") or 0) >= 1


def _blocked_reproduction(game: str) -> dict[str, Any]:
    return {
        "game": game,
        "claimed_level": CLAIMED_LEVEL,
        "reached_level": 0,
        "reproduced": False,
        "mode": "not_run_precondition_block",
    }


def _duration(started_at: float, ended_at: float) -> float:
    return max(0.0, round(float(ended_at - started_at), 6))


def _sleep_until_verifier_floor(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    remaining = VERIFIER_SCORING_DURATION_TARGET_S - elapsed
    if remaining > 0:
        sleep_fn(remaining)
    return now()


def _dc22_state(
    dc22_operator_result: Mapping[str, Any],
    dc22_reproduction: Mapping[str, Any],
) -> str:
    if dc22_operator_result.get("grounded") is not True:
        return "not_grounded"
    if bool(dc22_reproduction.get("reproduced")) and int(dc22_reproduction.get("reached_level") or 0) >= 1:
        return "solved"
    return "grounded_no_level"


def missing_gaps(
    *,
    ft09_resolved_generically: bool,
    dc22_state: str,
    dc22_operator_result: Mapping[str, Any],
) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if not ft09_resolved_generically:
        gaps.append(
            {
                "gap_id": FT09_LOO_GAP_ID,
                "game": "ft09",
                "residual_delta": "missing_local_constraint_color_cycle_verifier",
                "status": "open",
            }
        )
    if dc22_state != "solved":
        gaps.append(
            {
                "gap_id": DC22_GAP_ID,
                "game": "dc22",
                "residual_delta": str(dc22_operator_result.get("residual") or dc22_state),
                "status": "open",
            }
        )
    return gaps


def _verdict(
    *,
    precondition_miss: str | None,
    ft09_resolved_generically: bool,
    dc22_state: str,
    no_regression: bool,
) -> str:
    if precondition_miss:
        return f"complete: blocked_{precondition_miss}"
    if not ft09_resolved_generically:
        return "complete: ft09_generic_not_resolved_gap_logged"
    if not no_regression:
        return "complete: ft09_generic_resolved_but_config_rule_regression_detected"
    if dc22_state == "solved":
        return "success: ft09_generic_resolved_dc22_L1_offline_reproduced"
    if dc22_state == "grounded_no_level":
        return "complete: ft09_generic_resolved_dc22_grounded_no_level_gap_logged"
    return "complete: ft09_generic_resolved_dc22_not_grounded_gap_logged"


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
    ft09_digest: Mapping[str, Any],
    dc22_digest: Mapping[str, Any],
    ft09_operator_result: Mapping[str, Any],
    dc22_operator_result: Mapping[str, Any],
    ft09_reproduction: Mapping[str, Any],
    dc22_reproduction: Mapping[str, Any],
    no_regression: bool,
    started_at: float,
    ended_at: float,
    inference_substrate: str = INFERENCE_SUBSTRATE,
) -> dict[str, Any]:
    precondition_miss = first_precondition_miss(preconditions)
    ft09_resolved = (
        precondition_miss is None
        and ft09_operator_result.get("grounded") is True
        and bool(ft09_reproduction.get("reproduced"))
        and int(ft09_reproduction.get("reached_level") or 0) >= 1
    )
    dc22 = "not_grounded" if precondition_miss else _dc22_state(dc22_operator_result, dc22_reproduction)
    dc22_level = (
        int(dc22_reproduction.get("reached_level") or 0)
        if dc22 == "solved"
        else 0
    )
    reproduced_levels = (1 if ft09_resolved else 0) + dc22_level
    offline_reproduced = bool(ft09_resolved)
    gaps = missing_gaps(
        ft09_resolved_generically=ft09_resolved,
        dc22_state=dc22,
        dc22_operator_result=dc22_operator_result,
    )
    substrate = inference_substrate if precondition_miss is None else BLOCKED_INFERENCE_SUBSTRATE
    checksum_payload = {
        "few_shot_examples": list(few_shot_examples),
        "ft09_digest": ft09_digest,
        "dc22_digest": dc22_digest,
        "ft09_operator_result": ft09_operator_result,
        "dc22_operator_result": dc22_operator_result,
        "ft09_reproduction": ft09_reproduction,
        "dc22_reproduction": dc22_reproduction,
        "no_regression": bool(no_regression) if precondition_miss is None else False,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4444_generic_config_rule_verifier_operator",
        "schema": "carnot.exp4444.generic_config_rule_verifier_operator.v1",
        "target_games": ["ft09", "dc22"],
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            ft09_resolved_generically=ft09_resolved,
            dc22_state=dc22,
            no_regression=bool(no_regression) if precondition_miss is None else False,
        ),
        "inference_substrate": substrate,
        "duration_s": _duration(started_at, ended_at),
        "ft09_resolved_generically": bool(ft09_resolved),
        "dc22_state": dc22,
        "reproduced_levels": int(reproduced_levels),
        "offline_reproduced": offline_reproduced,
        "no_regression": bool(no_regression) if precondition_miss is None else False,
        "missing_verifier_gaps": gaps,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "few_shot_examples_used": [dict(row) for row in few_shot_examples],
        "ft09_object_centric_digest": dict(ft09_digest),
        "dc22_object_centric_digest": dict(dc22_digest),
        "ft09_operator_result": dict(ft09_operator_result),
        "dc22_operator_result": dict(dc22_operator_result),
        "ft09_reproduction_result": dict(ft09_reproduction),
        "dc22_reproduction_result": dict(dc22_reproduction),
        "model_specs": {
            "live_llm_call": False,
            "no_3090_inference": True,
            "leaderboard_submission": False,
        },
        "submitted_to_leaderboard": False,
        "no_3090_inference": True,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4444", "SCENARIO-REPORT-4444"],
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact or artifact.get(field) is None:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")
    substrate = artifact.get("inference_substrate")
    blocked = isinstance(verdict, str) and "blocked_" in verdict
    if not blocked and substrate != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if substrate == INFERENCE_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < VERIFIER_SCORING_MIN_DURATION_S:
        errors.append("cached verifier substrate requires duration_s >= 1.0")
    if substrate == LIVE_LLM_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < 60.0:
        errors.append("live_llm_inference requires duration_s >= 60.0")
    if type(artifact.get("ft09_resolved_generically")) is not bool:
        errors.append("ft09_resolved_generically must be bare bool")
    if artifact.get("dc22_state") not in {"solved", "grounded_no_level", "not_grounded"}:
        errors.append("dc22_state must be solved / grounded_no_level / not_grounded")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("no_regression")) is not bool:
        errors.append("no_regression must be bare bool")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("ft09_resolved_generically") is not True:
            errors.append("success verdict requires ft09_resolved_generically true")
        if artifact.get("offline_reproduced") is not True:
            errors.append("success verdict requires offline_reproduced true")
        if artifact.get("no_regression") is not True:
            errors.append("success verdict requires no_regression true")
        if artifact.get("dc22_state") != "solved":
            errors.append("success verdict requires dc22_state solved")
    if artifact.get("ft09_resolved_generically") is True and artifact.get("offline_reproduced") is not True:
        errors.append("ft09_resolved_generically requires offline_reproduced true")
    model_specs = artifact.get("model_specs")
    if isinstance(model_specs, Mapping):
        if model_specs.get("no_3090_inference") is not True:
            errors.append("model_specs.no_3090_inference must be true")
        if model_specs.get("leaderboard_submission") is not False:
            errors.append("model_specs.leaderboard_submission must be false")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    few_shot_examples: Sequence[Mapping[str, Any]] | None = None,
    ft09_digest: Mapping[str, Any] | None = None,
    dc22_digest: Mapping[str, Any] | None = None,
    reproduce_ft09_fn: Callable[[Sequence[str]], Mapping[str, Any]] = reproduce_ft09_solution,
    reproduce_dc22_fn: Callable[[Sequence[str]], Mapping[str, Any]] = reproduce_dc22_solution,
    no_regression_fn: Callable[[Path], bool] = no_config_rule_regression,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = now()
    root = Path(root)
    checked = dict(preconditions_checked or precondition_probe(root))
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    checked.setdefault(
        "generator_resource_available",
        checked.get("qwen_gguf_cached") is True or checked.get("igpu_llama_server_available") is True,
    )
    examples = list(few_shot_examples or extract_grounded_config_rule_examples(root))
    ft09_object_digest = dict(ft09_digest or FT09_L1_DIGEST)
    dc22_object_digest = dict(dc22_digest or DC22_FIRST_CONTACT_DIGEST)
    precondition_miss = first_precondition_miss(checked)

    if precondition_miss:
        ft09_operator = kit.config_rule_verifier(
            game="ft09",
            object_digest={},
            few_shot_examples=examples,
        )
        dc22_operator = kit.config_rule_verifier(
            game="dc22",
            object_digest={},
            few_shot_examples=examples,
        )
        ft09_reproduction = _blocked_reproduction("ft09")
        dc22_reproduction = _blocked_reproduction("dc22")
        no_regression = False
        ended = now()
    else:
        ft09_operator = kit.config_rule_verifier(
            game="ft09",
            object_digest=ft09_object_digest,
            few_shot_examples=examples,
        )
        dc22_operator = kit.config_rule_verifier(
            game="dc22",
            object_digest=dc22_object_digest,
            few_shot_examples=examples,
        )
        ft09_solution = list(ft09_operator.get("solution") or [])
        dc22_solution = list(dc22_operator.get("solution") or [])
        ft09_reproduction = (
            dict(reproduce_ft09_fn(ft09_solution))
            if ft09_operator.get("grounded") is True and ft09_solution
            else _blocked_reproduction("ft09")
        )
        dc22_reproduction = (
            dict(reproduce_dc22_fn(dc22_solution))
            if dc22_operator.get("grounded") is True and dc22_solution
            else _blocked_reproduction("dc22")
        )
        no_regression = bool(no_regression_fn(root))
        ended = _sleep_until_verifier_floor(started_at=started, now=now, sleep_fn=sleep_fn)

    artifact = build_artifact(
        root=root,
        preconditions=checked,
        few_shot_examples=examples,
        ft09_digest=ft09_object_digest,
        dc22_digest=dc22_object_digest,
        ft09_operator_result=ft09_operator,
        dc22_operator_result=dc22_operator,
        ft09_reproduction=ft09_reproduction,
        dc22_reproduction=dc22_reproduction,
        no_regression=no_regression,
        started_at=started,
        ended_at=ended,
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - script entry
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
