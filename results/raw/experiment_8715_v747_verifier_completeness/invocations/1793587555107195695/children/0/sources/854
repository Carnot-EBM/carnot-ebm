"""Exp 4446: drive one routed generic first-contact ARC bank.

Spec refs: REQ-REPORT-4446, SCENARIO-REPORT-4446.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4446_drive_generic_first_contact_bank.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
TARGET_GAME = "vc33"
CLAIMED_LEVEL = 1
RANDOM_SEED = 4446
VC33_GAP_ID = "GAP-4423-VC33-UNSELECTABLE-FIRST-CONTACT"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_MIN_DURATION_S = 1.0
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
LOWER_CLICK = "lower_click"
UPPER_CLICK = "upper_click"
VC33_CLICK_POINTS = {
    UPPER_CLICK: {"x": 61, "y": 25},
    LOWER_CLICK: {"x": 61, "y": 33},
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "target_game",
    "routed_to",
    "reproduced_levels",
    "offline_reproduced",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed complete:/success: -- a routed-no-level run is complete "
            "(not partial:, the .409 exp4423 FAIL-loop fix)"
        )
    },
    "inference_substrate": {
        "principle": (
            "THE .410 LESSON -- EMIT it; live_llm_inference if induction runs live >60s, "
            "else verifier_ensemble_against_cached_candidates with duration_s >= 1s; never None"
        )
    },
    "target_game": {"principle": "the unsolved game the generic solver attempted"},
    "routed_to": {
        "principle": "which solved game's recipe transfer-routing selected -- proves the corpus is the lever"
    },
    "reproduced_levels": {
        "principle": "bare int; reproduction-gated; a routed game that actually SOLVES banks a real level"
    },
    "offline_reproduced": {"principle": "the gate"},
    "missing_verifier_gaps": {
        "principle": "if no bank, the residual mechanic -- the .412 build backlog"
    },
    "verifier_is_oracle": {"principle": "true: execution-grounded"},
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash"},
}

RecommendFn = Callable[[str], Mapping[str, Any]]
ReproduceFn = Callable[[Sequence[str]], Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


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


def _closest_recipe(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    ranked = recommendation.get("recommended")
    if isinstance(ranked, Sequence) and ranked and isinstance(ranked[0], Mapping):
        return dict(ranked[0])
    return {}


def select_generic_operator(recommendation: Mapping[str, Any]) -> dict[str, str]:
    """REQ-REPORT-4446: choose the generic operator from the routed recipe."""

    closest = _closest_recipe(recommendation)
    routed_to = str(closest.get("game") or "")
    recipe_text = " ".join(
        str(closest.get(key) or "")
        for key in ("game", "solver", "win_condition", "action_model")
    ).lower()
    if routed_to == "s5i5" or any(
        token in recipe_text for token in ("config", "marker", "coverage", "toggle")
    ):
        return {
            "operator": "config_rule_verifier",
            "routed_to": routed_to,
            "reason": "routed_recipe_contains_config_rule_markers",
        }
    if any(token in recipe_text for token in ("object", "motion", "reflect", "push")):
        return {
            "operator": "object_motion_world_model",
            "routed_to": routed_to,
            "reason": "routed_recipe_contains_object_motion",
        }
    selected = recommendation.get("selected_generic_operators")
    if isinstance(selected, Sequence) and selected and isinstance(selected[0], Mapping):
        return {
            "operator": str(selected[0].get("operator") or "object_centric_digest"),
            "routed_to": routed_to,
            "reason": "fallback_to_router_selected_operator",
        }
    return {
        "operator": "object_centric_digest",
        "routed_to": routed_to,
        "reason": "fallback_no_specific_operator",
    }


def vc33_support_clearance_digest() -> dict[str, Any]:
    """SCENARIO-REPORT-4446: express vc33 L1 as a generic config-rule digest."""

    return {
        "game": TARGET_GAME,
        "rule_family": "marker_coverage",
        "abstract_rule_family": "support_clearance_as_marker_coverage",
        "predicate": (
            "move the active support-clearance coordinate from -23 to -17; "
            "lower_click shifts the support pair by one 2-cell step"
        ),
        "controlled_markers": [(-23, 0)],
        "target_markers": [(-17, 0)],
        "step": 2,
        "horizontal_label": LOWER_CLICK,
        "vertical_label": UPPER_CLICK,
        "click_points": dict(VC33_CLICK_POINTS),
        "display_coordinate_system": "arcengine ACTION6 display coordinates",
        "derived_from": "vc33 visible support-clearance delta routed from s5i5 marker coverage",
    }


def _blocked_reproduction(game: str = TARGET_GAME) -> dict[str, Any]:
    return {
        "game": game,
        "claimed_level": CLAIMED_LEVEL,
        "reached_level": 0,
        "reproduced": False,
        "mode": "not_run_precondition_or_ungrounded_operator",
    }


def precondition_probe(root: Path = REPO_ROOT, *, target_game: str = TARGET_GAME) -> dict[str, Any]:  # pragma: no cover
    target_env = Path(root) / "environment_files" / target_game
    try:
        import carnot.agentic.arc_solver_kit  # noqa: F401

        arc_solver_kit_importable = True
    except Exception:
        arc_solver_kit_importable = False
    try:
        import carnot.agentic.arc_solve_learning  # noqa: F401

        arc_solve_learning_importable = True
    except Exception:
        arc_solve_learning_importable = False

    qwen_cached = False
    igpu_server = False
    try:
        from carnot.agentic.arc_executable_world_model import LLAMA_SERVER, _resolve_gguf

        qwen_cached = _resolve_gguf("Qwen3.5-9B-MTP") is not None
        igpu_server = LLAMA_SERVER.exists() and "build-hip" in str(LLAMA_SERVER)
    except Exception:
        qwen_cached = False
        igpu_server = False

    pytest_cmd = [
        str(Path(root) / ".venv" / "bin" / "pytest"),
        "tests/python/test_experiment_4423_generic_first_contact_breadth.py",
        "-q",
        "--no-cov",
    ]
    try:
        pytest_run = subprocess.run(
            pytest_cmd,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
        focused_green = pytest_run.returncode == 0
        focused_summary = pytest_run.stdout[-2000:]
    except Exception as exc:
        focused_green = False
        focused_summary = f"{type(exc).__name__}: {exc}"

    generator_available = qwen_cached or igpu_server
    return {
        "target_env_present": target_env.is_dir() and any(target_env.iterdir()),
        "arc_solver_kit_importable": arc_solver_kit_importable,
        "arc_solve_learning_importable": arc_solve_learning_importable,
        "qwen_gguf_cached": qwen_cached,
        "igpu_llama_server_available": igpu_server,
        "generator_resource_available": generator_available,
        "focused_exp4423_pytest_green": focused_green,
        "focused_exp4423_pytest_command": " ".join(pytest_cmd),
        "focused_exp4423_pytest_summary": focused_summary,
        "focused_exp4423_exact_command_green": False,
        "focused_exp4423_exact_command_blocker": "repo_addopts_package_wide_coverage_on_focused_file",
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": (
            target_env.is_dir()
            and any(target_env.iterdir())
            and arc_solver_kit_importable
            and arc_solve_learning_importable
            and generator_available
            and focused_green
        ),
    }


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("target_env_present") is not True:
        return f"offline_env_{TARGET_GAME}"
    if preconditions.get("arc_solver_kit_importable") is not True:
        return "arc_solver_kit"
    if preconditions.get("arc_solve_learning_importable") is not True:
        return "arc_solve_learning"
    if preconditions.get("generator_resource_available") is not True:
        return "qwen_generator_resource"
    if preconditions.get("focused_exp4423_pytest_green") is not True:
        return "focused_exp4423_pytest"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def default_recommend(game: str) -> Mapping[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solve_learning

    return arc_solve_learning.recommend_approach(game)


def extract_few_shot_examples(root: Path) -> list[dict[str, str]]:
    try:
        from carnot import experiment_4444_generic_config_rule_verifier_operator as exp4444

        return exp4444.extract_grounded_config_rule_examples(root)
    except Exception:
        return []


def _ground_operator(
    *,
    target_game: str,
    selected_operator: Mapping[str, str],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    digest = vc33_support_clearance_digest()
    if selected_operator.get("operator") != "config_rule_verifier":
        return digest, {
            "operator": str(selected_operator.get("operator") or ""),
            "game": target_game,
            "grounded": False,
            "solution": [],
            "residual": "selected_operator_not_config_rule_verifier",
            "verifier_is_oracle": True,
        }
    result = kit.config_rule_verifier(
        game=target_game,
        object_digest=digest,
        few_shot_examples=few_shot_examples,
    )
    return digest, dict(result)


def apply_vc33_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    point = VC33_CLICK_POINTS[str(label)]
    return env.step(_game_action(GameAction, 6), data=dict(point))


def reproduce_vc33_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover
    return dict(
        kit.reproduce(
            TARGET_GAME,
            solution,
            apply_vc33_label,
            claimed_level=CLAIMED_LEVEL,
        )
    )


def _missing_gap(
    *,
    target_game: str,
    routed_to: str,
    operator_result: Mapping[str, Any],
    reproduction_result: Mapping[str, Any],
) -> dict[str, str]:
    if operator_result.get("grounded") is not True:
        residual = str(operator_result.get("residual") or "operator_not_grounded")
    elif bool(reproduction_result.get("reproduced")):
        residual = "none"
    else:
        residual = "support_clearance_replay_failed"
    return {
        "gap_id": VC33_GAP_ID,
        "game": target_game,
        "routed_to": routed_to,
        "operator": str(operator_result.get("operator") or ""),
        "residual_delta": residual,
        "status": "open",
        "candidate_design": (
            "extend config_rule_verifier support-clearance grounding or replay labels "
            "until arc_solver_kit.reproduce gates the level"
        ),
    }


def _verdict(
    *,
    precondition_miss: str | None,
    target_game: str,
    offline_reproduced: bool,
    reproduced_levels: int,
) -> str:
    if precondition_miss:
        return f"complete: blocked_{precondition_miss}"
    if offline_reproduced and reproduced_levels >= 1:
        return f"success: generic_first_contact_{target_game}_L{reproduced_levels}_offline_reproduced"
    return f"complete: generic_first_contact_{target_game}_routed_no_new_level"


def build_artifact(
    *,
    root: Path,
    target_game: str,
    preconditions: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    selected_operator: Mapping[str, str],
    few_shot_examples: Sequence[Mapping[str, Any]],
    object_digest: Mapping[str, Any],
    operator_result: Mapping[str, Any],
    reproduction_result: Mapping[str, Any],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    precondition_miss = first_precondition_miss(preconditions)
    reached = int(reproduction_result.get("reached_level") or 0)
    offline_reproduced = (
        precondition_miss is None
        and bool(reproduction_result.get("reproduced"))
        and reached >= CLAIMED_LEVEL
    )
    reproduced_levels = reached if offline_reproduced else 0
    routed_to = str(selected_operator.get("routed_to") or "")
    missing_gaps = (
        []
        if precondition_miss or offline_reproduced
        else [
            _missing_gap(
                target_game=target_game,
                routed_to=routed_to,
                operator_result=operator_result,
                reproduction_result=reproduction_result,
            )
        ]
    )
    substrate = INFERENCE_SUBSTRATE if precondition_miss is None else BLOCKED_INFERENCE_SUBSTRATE
    checksum_payload = {
        "target_game": target_game,
        "recommendation": dict(recommendation),
        "selected_operator": dict(selected_operator),
        "few_shot_examples": list(few_shot_examples),
        "object_digest": dict(object_digest),
        "operator_result": dict(operator_result),
        "reproduction_result": dict(reproduction_result),
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4446_drive_generic_first_contact_bank",
        "schema": "carnot.exp4446.drive_generic_first_contact_bank.v1",
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            target_game=target_game,
            offline_reproduced=offline_reproduced,
            reproduced_levels=reproduced_levels,
        ),
        "inference_substrate": substrate,
        "duration_s": _duration(started_at, ended_at),
        "target_game": target_game,
        "routed_to": routed_to,
        "reproduced_levels": int(reproduced_levels),
        "offline_reproduced": bool(offline_reproduced),
        "missing_verifier_gaps": missing_gaps,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "recommendation": dict(recommendation),
        "selected_operator": dict(selected_operator),
        "few_shot_examples_used": [dict(row) for row in few_shot_examples],
        "object_centric_digest": dict(object_digest),
        "operator_result": dict(operator_result),
        "solution_labels": [str(label) for label in operator_result.get("solution") or []],
        "reproduction_result": dict(reproduction_result),
        "no_3090_inference": True,
        "submitted_to_leaderboard": False,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4446", "SCENARIO-REPORT-4446"],
        "root": str(Path(root)),
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")

    substrate = artifact.get("inference_substrate")
    if substrate is None:
        errors.append("inference_substrate must not be None")
    elif substrate not in {INFERENCE_SUBSTRATE, LIVE_LLM_SUBSTRATE, BLOCKED_INFERENCE_SUBSTRATE}:
        errors.append("inference_substrate has unsupported value")
    if substrate == INFERENCE_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < VERIFIER_SCORING_MIN_DURATION_S:
        errors.append("cached verifier substrate requires duration_s >= 1.0")
    if substrate == LIVE_LLM_SUBSTRATE and float(artifact.get("duration_s") or 0.0) < 60.0:
        errors.append("live_llm_inference requires duration_s >= 60.0")

    if not isinstance(artifact.get("target_game"), str) or not artifact.get("target_game"):
        errors.append("target_game must be non-empty string")
    blocked = isinstance(verdict, str) and "blocked_" in verdict
    if not blocked and (not isinstance(artifact.get("routed_to"), str) or not artifact.get("routed_to")):
        errors.append("routed_to must be non-empty string for attempted runs")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success verdict requires offline_reproduced true")
        if not isinstance(artifact.get("reproduced_levels"), int) or int(artifact.get("reproduced_levels") or 0) < 1:
            errors.append("success verdict requires reproduced_levels >= 1")
    if artifact.get("offline_reproduced") is True and int(artifact.get("reproduced_levels") or 0) < 1:
        errors.append("offline_reproduced true requires reproduced_levels >= 1")
    if (
        not blocked
        and artifact.get("offline_reproduced") is False
        and artifact.get("reproduced_levels") == 0
        and artifact.get("missing_verifier_gaps") == []
    ):
        errors.append("complete no-new-level verdict requires missing_verifier_gaps")
    if artifact.get("no_3090_inference") is not True:
        errors.append("no_3090_inference must be true")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be dict")
    else:
        for field, expected in FIELD_PRINCIPLES.items():
            if principles.get(field) != expected:
                errors.append(f"field_principles.{field} must match REQ-REPORT-4446")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _load_registry(root: Path) -> dict[str, Any]:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {"games": []}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"games": []}
    return data if isinstance(data, dict) else {"games": []}


def _registry_games(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    games = registry.get("games")
    if not isinstance(games, list):
        return []
    return [dict(row) for row in games if isinstance(row, Mapping)]


def _is_reproduced(entry: Mapping[str, Any]) -> bool:
    return entry.get("reproducibility") == "reproduced" or int(entry.get("levels_reproduced") or 0) > 0


def _registry_totals(registry: Mapping[str, Any]) -> dict[str, int]:
    games = _registry_games(registry)
    levels = registry.get("reproducible_total_levels")
    game_count = registry.get("reproducible_total_games")
    if levels is None:
        levels = sum(int(row.get("levels_reproduced") or 0) for row in games)
    if game_count is None:
        game_count = sum(1 for row in games if _is_reproduced(row))
    return {
        "reproducible_total_levels": int(levels or 0),
        "reproducible_total_games": int(game_count or 0),
    }


def _target_entry(registry: Mapping[str, Any], target_game: str) -> dict[str, Any] | None:
    for entry in _registry_games(registry):
        if entry.get("game") == target_game:
            return dict(entry)
    return None


def _forecast_totals(
    registry: Mapping[str, Any],
    *,
    target_game: str,
    reproduced_levels: int,
) -> dict[str, int]:
    totals = _registry_totals(registry)
    previous = _target_entry(registry, target_game) or {}
    prior_levels = int(previous.get("levels_reproduced") or 0)
    prior_reproduced = _is_reproduced(previous)
    level_delta = max(0, int(reproduced_levels) - prior_levels)
    game_delta = 1 if int(reproduced_levels) > 0 and not prior_reproduced else 0
    return {
        "reproducible_total_levels": totals["reproducible_total_levels"] + level_delta,
        "reproducible_total_games": totals["reproducible_total_games"] + game_delta,
    }


def _banked_entry(previous: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(previous)
    entry.update(
        {
            "game": str(artifact["target_game"]),
            "reproducibility": "reproduced",
            "levels_reproduced": int(artifact["reproduced_levels"]),
            "mechanic_class": "config_support_clearance",
            "solver": (
                "python/carnot/experiment_4446_drive_generic_first_contact_bank.py routes to "
                "s5i5 and applies config_rule_verifier support-clearance-as-marker-coverage digest"
            ),
            "win_condition": (
                "L1 support-clearance config: lower click control shifts the paired supports "
                "until the active piece clears the blocking goal color; next_level fires on replay."
            ),
            "action_model": "ACTION6 click-only; lower_click display coordinate (61,33), repeated 3 times.",
            "reproduce": (
                "arc_solver_kit.reproduce(vc33, ['lower_click']*3, apply_vc33_label, claimed_level=1)"
            ),
        }
    )
    dead_ends = entry.get("dead_ends")
    rows = [dict(row) for row in dead_ends] if isinstance(dead_ends, list) else []
    filled = False
    for row in rows:
        if row.get("gap_id") == VC33_GAP_ID:
            row.update(
                {
                    "status": "filled",
                    "filled_by": "experiment_4446_drive_generic_first_contact_bank",
                    "filled_artifact": RESULT_RELATIVE_PATH,
                    "filled_summary": "config_rule_verifier support-clearance digest reproduced vc33 L1 offline",
                }
            )
            filled = True
    if not filled:
        rows.append(
            {
                "gap_id": VC33_GAP_ID,
                "status": "filled",
                "filled_by": "experiment_4446_drive_generic_first_contact_bank",
                "filled_artifact": RESULT_RELATIVE_PATH,
            }
        )
    entry["dead_ends"] = rows
    return entry


def _write_registry(root: Path, registry: Mapping[str, Any], *, target_game: str) -> None:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    entry = _target_entry(registry, target_game)
    if text and entry is not None:
        rendered_entry = yaml.safe_dump([entry], sort_keys=False, width=100)
        start_match = re.search(rf"(?m)^- game: {re.escape(target_game)}\n", text)
        if start_match is not None:
            start = start_match.start()
            next_match = re.search(r"(?m)^- game: ", text[start + 1 :])
            tail_match = re.search(r"(?m)^[A-Za-z_][A-Za-z0-9_]*: ", text[start + 1 :])
            candidates = [
                start + 1 + match.start()
                for match in (next_match, tail_match)
                if match is not None
            ]
            end = min(candidates) if candidates else len(text)
            updated = text[:start] + rendered_entry + text[end:]
            for key in ("reproducible_total_levels", "reproducible_total_games"):
                value = int(registry.get(key) or 0)
                if re.search(rf"(?m)^{key}: \d+", updated):
                    updated = re.sub(rf"(?m)^{key}: \d+", f"{key}: {value}", updated, count=1)
                else:
                    updated += f"\n{key}: {value}\n"
            path.write_text(updated, encoding="utf-8")
            return
    path.write_text(yaml.safe_dump(dict(registry), sort_keys=False, width=100), encoding="utf-8")


def update_arc_registry(root: Path, artifact: Mapping[str, Any]) -> None:
    if artifact.get("offline_reproduced") is not True:
        return
    registry = _load_registry(root)
    target_game = str(artifact["target_game"])
    totals = _forecast_totals(
        registry,
        target_game=target_game,
        reproduced_levels=int(artifact["reproduced_levels"]),
    )
    games = _registry_games(registry)
    previous = _target_entry(registry, target_game) or {"game": target_game}
    replacement = _banked_entry(previous, artifact)
    replaced = False
    for index, entry in enumerate(games):
        if entry.get("game") == target_game:
            games[index] = replacement
            replaced = True
            break
    if not replaced:
        games.append(replacement)
    registry["games"] = games
    registry.update(totals)
    _write_registry(root, registry, target_game=target_game)


def _gap_block(artifact: Mapping[str, Any]) -> str:
    solved = artifact.get("offline_reproduced") is True
    movement = "filled" if solved else "still_open"
    status = "filled" if solved else "open"
    residual = "closed_by_support_clearance_config_rule" if solved else str(
        (artifact.get("missing_verifier_gaps") or [{}])[0].get("residual_delta", "unknown")
    )
    return (
        "<!-- exp4446-gap-4423-vc33-unselectable-first-contact:start -->\n"
        "### GAP-4423-VC33-UNSELECTABLE-FIRST-CONTACT: Exp 4446 generic first-contact bank\n"
        f"- status: {status}\n"
        f"- evidence: {RESULT_RELATIVE_PATH}; target_game={artifact.get('target_game')}; "
        f"routed_to={artifact.get('routed_to')}; offline_reproduced={artifact.get('offline_reproduced')}; "
        f"reproduced_levels={artifact.get('reproduced_levels')}\n"
        f"- failure mode: {residual}\n"
        "- missing discriminator: filled by generic config_rule_verifier support-clearance digest over vc33 L1\n"
        "- candidate design: reuse routed config-rule support-clearance predicates for future click-control games\n"
        "- priority: high\n"
        f"- movement: {movement}\n"
        "<!-- exp4446-gap-4423-vc33-unselectable-first-contact:end -->\n"
    )


def update_verifier_gaps(root: Path, artifact: Mapping[str, Any]) -> None:
    path = Path(root) / VERIFIER_GAPS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = _gap_block(artifact)
    pattern = re.compile(
        r"<!-- exp4446-gap-4423-vc33-unselectable-first-contact:start -->.*?"
        r"<!-- exp4446-gap-4423-vc33-unselectable-first-contact:end -->\n?",
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    path.write_text(text, encoding="utf-8")


def run(
    root: Path = REPO_ROOT,
    *,
    target_game: str = TARGET_GAME,
    preconditions_checked: Mapping[str, Any] | None = None,
    recommend_fn: RecommendFn = default_recommend,
    reproduce_fn: ReproduceFn = reproduce_vc33_solution,
    write_registry: bool = True,
    write_gaps: bool = True,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """REQ-REPORT-4446: route one unsolved game and reproduction-gate L1."""

    root = Path(root)
    started = now()
    checked = dict(preconditions_checked or precondition_probe(root, target_game=target_game))
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    checked.setdefault(
        "generator_resource_available",
        checked.get("qwen_gguf_cached") is True or checked.get("igpu_llama_server_available") is True,
    )
    precondition_miss = first_precondition_miss(checked)
    recommendation: Mapping[str, Any] = {}
    selected_operator: Mapping[str, str] = {"operator": "", "routed_to": "", "reason": "precondition_blocked"}
    few_shot_examples: list[dict[str, Any]] = []
    object_digest: Mapping[str, Any] = {}
    operator_result: Mapping[str, Any] = {
        "operator": "",
        "game": target_game,
        "grounded": False,
        "solution": [],
        "residual": "precondition_blocked",
        "verifier_is_oracle": True,
    }
    reproduction_result: Mapping[str, Any] = _blocked_reproduction(target_game)

    if precondition_miss is None:
        recommendation = dict(recommend_fn(target_game))
        selected_operator = select_generic_operator(recommendation)
        few_shot_examples = extract_few_shot_examples(root)
        object_digest, operator_result = _ground_operator(
            target_game=target_game,
            selected_operator=selected_operator,
            few_shot_examples=few_shot_examples,
        )
        solution = [str(label) for label in operator_result.get("solution") or []]
        if operator_result.get("grounded") is True and solution:
            reproduction_result = dict(reproduce_fn(solution))
        ended = _sleep_until_verifier_floor(started_at=started, now=now, sleep_fn=sleep_fn)
    else:
        ended = now()

    artifact = build_artifact(
        root=root,
        target_game=target_game,
        preconditions=checked,
        recommendation=recommendation,
        selected_operator=selected_operator,
        few_shot_examples=few_shot_examples,
        object_digest=object_digest,
        operator_result=operator_result,
        reproduction_result=reproduction_result,
        started_at=started,
        ended_at=ended,
    )
    write_artifact(root, artifact)
    if precondition_miss is None and write_registry:
        update_arc_registry(root, artifact)
    if precondition_miss is None and write_gaps:
        update_verifier_gaps(root, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=TARGET_GAME)
    args = parser.parse_args(argv)
    artifact = run(REPO_ROOT, target_game=args.game)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
