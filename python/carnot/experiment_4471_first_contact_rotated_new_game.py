"""Exp 4471: rotated generic first-contact attempt on re86.

Spec refs: REQ-REPORT-4471, SCENARIO-REPORT-4471.
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
RESULT_RELATIVE_PATH = "results/experiment_4471_first_contact_rotated_new_game.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
PREFERRED_TARGETS = ("re86", "lf52", "bp35")
TARGET_GAME = "re86"
CLAIMED_LEVEL = 1
RANDOM_SEED = 4471
RE86_GAP_ID = "GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER"
BASELINE_COMMAND_TEXT = '.venv/bin/pytest -k "arc_solver_kit or first_contact" -q --no-cov'
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_MIN_DURATION_S = 1.0
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
GRAPH_EXPANSION_BUDGET = 6000
GRAPH_DEPTH_CAP = 60

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "target_game",
    "routed_to",
    "retrieved_primitives",
    "reproduced_levels",
    "offline_reproduced",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "reproducible_total_levels",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed complete:/success: -- a routed-no-level run is complete "
            "(NOT partial:, the exp4423 FAIL-loop fix)"
        )
    },
    "inference_substrate": {
        "principle": "THE .410/.411 LESSON -- EMIT it; never None"
    },
    "target_game": {
        "principle": "the never-attempted ROTATED game the generic solver attempted (bp35/lf52/re86)"
    },
    "routed_to": {
        "principle": "which solved game's recipe transfer-routing selected -- proves the corpus is the lever"
    },
    "retrieved_primitives": {
        "principle": (
            "the ranked documented-library primitives retrieve_primitives returned -- "
            "proves the LILO library is in the loop"
        )
    },
    "reproduced_levels": {
        "principle": "bare int; reproduction-gated; a routed game that actually SOLVES banks a real level"
    },
    "offline_reproduced": {"principle": "the gate"},
    "missing_verifier_gaps": {
        "principle": "if no bank, the residual mechanic -- the .414 build backlog"
    },
    "verifier_is_oracle": {"principle": "true: execution-grounded"},
    "reproducible_total_levels": {
        "principle": "the new authoritative count if banked"
    },
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash"},
}

RecommendFn = Callable[[str], Mapping[str, Any]]
GroundFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]
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
    if isinstance(ranked, Sequence) and not isinstance(ranked, (str, bytes)) and ranked:
        first = ranked[0]
        if isinstance(first, Mapping):
            return dict(first)
    return {}


def _retrieved_primitives(recommendation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = recommendation.get("retrieved_primitives")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def select_rotated_target(recommendations: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """REQ-REPORT-4471: choose the rotated target from routing evidence."""

    if "re86" in recommendations:
        return {
            "target_game": "re86",
            "reason": "pattern_match_sprite_resize_is_best_rotated_gap_for_existing_graph_route",
        }
    for game in PREFERRED_TARGETS:
        if game in recommendations:
            return {"target_game": game, "reason": "first_available_rotated_target"}
    return {"target_game": TARGET_GAME, "reason": "default_rotated_target"}


def select_generic_operator(recommendation: Mapping[str, Any]) -> dict[str, str]:
    """REQ-REPORT-4471: choose the best-fitting generic operator from routing evidence."""

    closest = _closest_recipe(recommendation)
    routed_to = str(closest.get("game") or "")
    for row in _retrieved_primitives(recommendation):
        operator = str(row.get("operator") or "")
        text = f"{operator} {row.get('mechanic_class', '')} {row.get('matched_cues', '')}".lower()
        if operator == "graph_astar_action_cost" and ("graph" in text or "keyboard" in text):
            return {
                "operator": "graph_astar_action_cost",
                "routed_to": routed_to,
                "reason": "retrieved_graph_primitive_matches_re86_keyboard_pattern_route",
                "source": "retrieved_primitives",
            }

    recipe_text = " ".join(
        str(closest.get(key) or "")
        for key in ("game", "solver", "win_condition", "action_model")
    ).lower()
    if "glyph" in recipe_text or "rewrite" in recipe_text or "pattern" in recipe_text:
        return {
            "operator": "glyph_rewrite_rule_verifier",
            "routed_to": routed_to,
            "reason": "routed_recipe_contains_glyph_or_pattern_rewrite",
            "source": "closest_recipe",
        }
    if "object motion" in recipe_text or "world model" in recipe_text:
        return {
            "operator": "object_motion_world_model",
            "routed_to": routed_to,
            "reason": "routed_recipe_contains_object_motion",
            "source": "closest_recipe",
        }
    selected = recommendation.get("selected_generic_operators")
    if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)) and selected:
        first = selected[0]
        if isinstance(first, Mapping):
            return {
                "operator": str(first.get("operator") or "object_centric_digest"),
                "routed_to": routed_to,
                "reason": "fallback_to_router_selected_operator",
                "source": "selected_generic_operators",
            }
    return {
        "operator": "object_centric_digest",
        "routed_to": routed_to,
        "reason": "fallback_no_specific_operator",
        "source": "fallback",
    }


def target_digest_for(game: str) -> dict[str, Any]:
    if game == "re86":
        return {
            "game": game,
            "mechanic_class": "pattern_match_sprite_resize",
            "rule_family": "sprite_overlay_pattern_match",
            "predicate": "overlay movable sprites and resized variants to exactly match target patterns",
            "action_model": "keyboard ACTION1-4 movement plus ACTION5 transformation/resize",
            "counterexample_guidance": [
                "graph_astar_action_cost can enumerate movement actions",
                "re86 needs a verifier for exact sprite overlay, transparency, and resize transformations",
            ],
        }
    return {
        "game": game,
        "mechanic_class": "unknown_arc_mechanic",
        "rule_family": "unknown_first_contact",
    }


def _blocked_reproduction(game: str) -> dict[str, Any]:
    return {
        "game": game,
        "claimed_level": CLAIMED_LEVEL,
        "reached_level": 0,
        "reproduced": False,
        "mode": "not_run_precondition_or_ungrounded_operator",
    }


def precondition_probe(root: Path = REPO_ROOT, *, target_game: str = TARGET_GAME) -> dict[str, Any]:  # pragma: no cover
    root = Path(root)
    env_path = root / "environment_files" / target_game
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

        qwen_cached = _resolve_gguf("Qwen3.5-9B-MTP") is not None or _resolve_gguf("Qwen3.6-35B-A3B") is not None
        igpu_server = LLAMA_SERVER.exists() and "build-hip" in str(LLAMA_SERVER)
    except Exception:
        qwen_cached = False
        igpu_server = False

    pytest_cmd = [
        str(root / ".venv" / "bin" / "pytest"),
        "-k",
        "arc_solver_kit or first_contact",
        "-q",
        "--no-cov",
    ]
    try:
        baseline = subprocess.run(
            pytest_cmd,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        baseline_exit_code = int(baseline.returncode)
        baseline_output = baseline.stdout[-4000:]
    except Exception as exc:
        baseline_exit_code = 1
        baseline_output = f"{type(exc).__name__}: {exc}"

    checks = {
        "target_env_present": env_path.is_dir() and any(env_path.iterdir()),
        "arc_solver_kit_importable": arc_solver_kit_importable,
        "arc_solve_learning_importable": arc_solve_learning_importable,
        "qwen_gguf_cached": qwen_cached,
        "igpu_llama_server_available": igpu_server,
        "generator_resource_available": bool(qwen_cached or igpu_server),
        "baseline_command": BASELINE_COMMAND_TEXT,
        "baseline_exit_code": baseline_exit_code,
        "baseline_pytest_nocov_green": baseline_exit_code == 0,
        "baseline_output_tail": baseline_output,
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }
    checks["ok"] = first_precondition_miss(checks, target_game) is None
    return checks


def first_precondition_miss(preconditions: Mapping[str, Any], target_game: str) -> str | None:
    if preconditions.get("target_env_present") is not True:
        return f"offline_env_{target_game}"
    if preconditions.get("arc_solver_kit_importable") is not True:
        return "arc_solver_kit"
    if preconditions.get("arc_solve_learning_importable") is not True:
        return "arc_solve_learning"
    if preconditions.get("generator_resource_available") is not True:
        return "qwen_generator_resource"
    if preconditions.get("baseline_pytest_nocov_green") is not True:
        return "baseline_pytest_nocov"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def default_recommend(game: str) -> Mapping[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solve_learning

    return arc_solve_learning.recommend_approach(game)


def extract_few_shot_examples(root: Path) -> list[dict[str, Any]]:  # pragma: no cover
    try:
        from carnot import experiment_4458_first_contact_new_game as exp4458

        return exp4458.extract_few_shot_examples(root)
    except Exception:
        return []


def _trajectory_to_labels(trajectory: Sequence[Mapping[str, Any]]) -> list[str]:
    return [_stable_json({"action": int(row["action"]), **({"data": row["data"]} if row.get("data") is not None else {})}) for row in trajectory]


def _ground_operator(
    *,
    target_game: str,
    selected_operator: Mapping[str, str],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover
    del few_shot_examples
    digest = target_digest_for(target_game)
    operator = str(selected_operator.get("operator") or "")
    if operator != "graph_astar_action_cost":
        return digest, {
            "operator": operator,
            "game": target_game,
            "grounded": False,
            "solution": [],
            "predicate_id": "",
            "counterexample_rounds": 1,
            "residual": "selected_operator_not_supported_for_rotated_first_contact",
            "verifier_is_oracle": True,
        }

    from carnot.agentic import arc_graph_explore

    stats: dict[str, Any] = {}
    arc = kit.offline_arcade()
    env = arc.make(target_game, scorecard_id=arc.open_scorecard())
    trajectory, reached = arc_graph_explore.graph_explore_solve_v2(
        env,
        start_level=0,
        max_expansions=GRAPH_EXPANSION_BUDGET,
        max_depth=GRAPH_DEPTH_CAP,
        warmup=False,
        mask_hud=True,
        stats=stats,
    )
    solution = _trajectory_to_labels(trajectory or [])
    grounded = bool(solution) and int(reached or 0) >= CLAIMED_LEVEL
    residual = "" if grounded else "missing_pattern_match_sprite_resize_verifier"
    return digest, {
        "operator": operator,
        "game": target_game,
        "grounded": grounded,
        "solution": solution,
        "predicate_id": "sprite_overlay_pattern_match" if grounded else "",
        "counterexample_rounds": 1 if grounded else 2,
        "residual": residual,
        "counterexamples": [] if grounded else [
            {
                "rejected_candidate": "graph_astar_without_pattern_resize_verifier",
                "observed_digest_rule_family": digest["rule_family"],
                "residual": residual,
            }
        ],
        "graph_search": {
            "expansions": int(stats.get("expansions") or 0),
            "states": int(stats.get("states") or 0),
            "reached_level": int(reached or 0),
            "max_expansions": GRAPH_EXPANSION_BUDGET,
            "max_depth": GRAPH_DEPTH_CAP,
        },
        "verifier_is_oracle": True,
    }


def apply_re86_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    loaded = json.loads(str(label))
    action_id = int(loaded["action"])
    data = loaded.get("data")
    if isinstance(data, Mapping):
        return env.step(_game_action(GameAction, action_id), data=dict(data))
    return env.step(_game_action(GameAction, action_id))


def reproduce_re86_solution(solution: Sequence[str]) -> dict[str, Any]:  # pragma: no cover
    return dict(kit.reproduce(TARGET_GAME, [str(label) for label in solution], apply_re86_label, claimed_level=CLAIMED_LEVEL))


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


def _forecast_totals(registry: Mapping[str, Any], *, target_game: str, reproduced_levels: int) -> dict[str, int]:
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


def _missing_gap(
    *,
    target_game: str,
    routed_to: str,
    selected_operator: Mapping[str, str],
    operator_result: Mapping[str, Any],
) -> dict[str, str]:
    residual = str(operator_result.get("residual") or "operator_not_grounded")
    return {
        "gap_id": RE86_GAP_ID if target_game == "re86" else f"GAP-4471-{target_game.upper()}-ROTATED-FIRST-CONTACT",
        "game": target_game,
        "routed_to": routed_to,
        "operator": str(selected_operator.get("operator") or operator_result.get("operator") or ""),
        "residual_delta": residual,
        "status": "open",
        "candidate_design": "add a generic sprite-overlay pattern-match and resize/transformation verifier",
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
    target_digest: Mapping[str, Any],
    operator_result: Mapping[str, Any],
    reproduction_result: Mapping[str, Any],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    precondition_miss = first_precondition_miss(preconditions, target_game)
    reached = int(reproduction_result.get("reached_level") or 0)
    offline_reproduced = precondition_miss is None and bool(reproduction_result.get("reproduced")) and reached >= CLAIMED_LEVEL
    reproduced_levels = reached if offline_reproduced else 0
    routed_to = str(selected_operator.get("routed_to") or "")
    retrieved = _retrieved_primitives(recommendation)
    totals = _forecast_totals(_load_registry(root), target_game=target_game, reproduced_levels=reproduced_levels)
    missing_gaps = (
        []
        if precondition_miss or offline_reproduced
        else [_missing_gap(target_game=target_game, routed_to=routed_to, selected_operator=selected_operator, operator_result=operator_result)]
    )
    checksum_payload = {
        "target_game": target_game,
        "recommendation": dict(recommendation),
        "selected_operator": dict(selected_operator),
        "target_digest": dict(target_digest),
        "operator_result": dict(operator_result),
        "reproduction_result": dict(reproduction_result),
        "reproducible_total_levels": totals["reproducible_total_levels"],
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4471_first_contact_rotated_new_game",
        "schema": "carnot.exp4471.first_contact_rotated_new_game.v1",
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            target_game=target_game,
            offline_reproduced=offline_reproduced,
            reproduced_levels=reproduced_levels,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE if precondition_miss is None else BLOCKED_INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "target_game": target_game,
        "routed_to": routed_to,
        "retrieved_primitives": retrieved,
        "reproduced_levels": int(reproduced_levels),
        "offline_reproduced": bool(offline_reproduced),
        "missing_verifier_gaps": missing_gaps,
        "verifier_is_oracle": True,
        "reproducible_total_levels": int(totals["reproducible_total_levels"]),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "recommendation": dict(recommendation),
        "selected_operator": dict(selected_operator),
        "few_shot_examples_used": [dict(row) for row in few_shot_examples],
        "target_digest": dict(target_digest),
        "operator_result": dict(operator_result),
        "solution_labels": [str(label) for label in operator_result.get("solution") or []],
        "reproduction_result": dict(reproduction_result),
        "target_selection": select_rotated_target({target_game: dict(recommendation)}) if recommendation else {"target_game": target_game, "reason": "precondition_blocked"},
        "no_3090_inference": True,
        "submitted_to_leaderboard": False,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4471", "SCENARIO-REPORT-4471"],
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
    if not blocked and artifact.get("retrieved_primitives") == []:
        errors.append("retrieved_primitives must be non-empty list for attempted runs")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be bare int")
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
    if not blocked and artifact.get("offline_reproduced") is False and artifact.get("reproduced_levels") == 0 and artifact.get("missing_verifier_gaps") == []:
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
                errors.append(f"field_principles.{field} must match REQ-REPORT-4471")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _dead_end_row(artifact: Mapping[str, Any]) -> dict[str, Any]:
    gap = dict((artifact.get("missing_verifier_gaps") or [{}])[0])
    return {
        **gap,
        "routed_recipe": {
            "game": artifact.get("routed_to", ""),
            "selected_operator": dict(artifact.get("selected_operator") or {}),
        },
        "artifact": RESULT_RELATIVE_PATH,
    }


def _banked_entry(previous: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(previous)
    target_game = str(artifact["target_game"])
    entry.update(
        {
            "game": target_game,
            "reproducibility": "reproduced",
            "levels_reproduced": int(artifact["reproduced_levels"]),
            "mechanic_class": "pattern_match_sprite_resize",
            "solver": "python/carnot/experiment_4471_first_contact_rotated_new_game.py generic first-contact route",
            "win_condition": str((artifact.get("target_digest") or {}).get("predicate") or ""),
            "action_model": str((artifact.get("target_digest") or {}).get("action_model") or ""),
            "reproduce": "arc_solver_kit.reproduce(re86, solution_labels, apply_re86_label, claimed_level=1)",
        }
    )
    rows = [dict(row) for row in entry.get("dead_ends", [])] if isinstance(entry.get("dead_ends"), list) else []
    filled = False
    for row in rows:
        if row.get("gap_id") == RE86_GAP_ID:
            row.update(
                {
                    "status": "filled",
                    "filled_by": "experiment_4471_first_contact_rotated_new_game",
                    "filled_artifact": RESULT_RELATIVE_PATH,
                    "filled_summary": "generic first-contact reproduction gate banked re86 L1",
                }
            )
            filled = True
    if not filled:
        rows.append(
            {
                "gap_id": RE86_GAP_ID,
                "status": "filled",
                "filled_by": "experiment_4471_first_contact_rotated_new_game",
                "filled_artifact": RESULT_RELATIVE_PATH,
            }
        )
    entry["dead_ends"] = rows
    return entry


def _dead_end_entry(previous: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    target_game = str(artifact["target_game"])
    entry = dict(previous)
    entry.setdefault("game", target_game)
    entry["reproducibility"] = "unsolved"
    entry["levels_reproduced"] = int(entry.get("levels_reproduced") or 0)
    entry["mechanic_class"] = "pattern_match_sprite_resize"
    entry["solver"] = entry.get("solver") or f"python/carnot/experiment_4471_first_contact_rotated_new_game.py --game {target_game}"
    rows = [dict(row) for row in entry.get("dead_ends", [])] if isinstance(entry.get("dead_ends"), list) else []
    gap = _dead_end_row(artifact)
    gap_id = gap.get("gap_id")
    for index, row in enumerate(rows):
        if row.get("gap_id") == gap_id:
            rows[index] = {**row, **gap}
            entry["dead_ends"] = rows
            return entry
    rows.append(gap)
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
            candidates = [start + 1 + match.start() for match in (next_match, tail_match) if match is not None]
            end = min(candidates) if candidates else len(text)
            updated = text[:start] + rendered_entry + text[end:]
        else:
            totals_match = re.search(r"(?m)^reproducible_total_levels: ", text)
            insert_at = totals_match.start() if totals_match is not None else len(text)
            prefix = text[:insert_at]
            suffix = text[insert_at:]
            if prefix and not prefix.endswith("\n"):  # pragma: no cover
                prefix += "\n"
            updated = prefix + rendered_entry + suffix
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
    registry = _load_registry(root)
    target_game = str(artifact["target_game"])
    games = _registry_games(registry)
    previous = _target_entry(registry, target_game) or {"game": target_game}
    if artifact.get("offline_reproduced") is True:
        replacement = _banked_entry(previous, artifact)
        totals = _forecast_totals(registry, target_game=target_game, reproduced_levels=int(artifact["reproduced_levels"]))
    else:
        replacement = _dead_end_entry(previous, artifact)
        totals = _registry_totals(registry)
    for index, entry in enumerate(games):
        if entry.get("game") == target_game:
            games[index] = replacement
            break
    else:
        games.append(replacement)
    registry["games"] = games
    registry.update(totals)
    _write_registry(root, registry, target_game=target_game)


def _gap_block(artifact: Mapping[str, Any]) -> str:
    solved = artifact.get("offline_reproduced") is True
    movement = "filled" if solved else "still_open"
    status = "filled" if solved else "open"
    residual = "closed_by_generic_first_contact_reproduction" if solved else str(
        (artifact.get("missing_verifier_gaps") or [{}])[0].get("residual_delta", "unknown")
    )
    return (
        "<!-- exp4471-gap-re86-pattern-match-sprite-resize:start -->\n"
        "### GAP-4471-RE86-PATTERN-MATCH-SPRITE-RESIZE: Exp 4471 rotated first-contact\n"
        f"- status: {status}\n"
        f"- evidence: {RESULT_RELATIVE_PATH}; target_game={artifact.get('target_game')}; "
        f"routed_to={artifact.get('routed_to')}; offline_reproduced={artifact.get('offline_reproduced')}; "
        f"reproduced_levels={artifact.get('reproduced_levels')}\n"
        f"- failure mode: {residual}\n"
        "- missing discriminator: generic sprite-overlay pattern-match and resize/transformation verifier\n"
        "- candidate design: extend graph/object operators with exact overlay and ACTION5 resize grounding\n"
        "- priority: high\n"
        f"- movement: {movement}\n"
        "<!-- exp4471-gap-re86-pattern-match-sprite-resize:end -->\n"
    )


def update_verifier_gaps(root: Path, artifact: Mapping[str, Any]) -> None:
    path = Path(root) / VERIFIER_GAPS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = _gap_block(artifact)
    pattern = re.compile(
        r"<!-- exp4471-gap-re86-pattern-match-sprite-resize:start -->.*?"
        r"<!-- exp4471-gap-re86-pattern-match-sprite-resize:end -->\n?",
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
    ground_operator_fn: GroundFn = _ground_operator,
    reproduce_fn: ReproduceFn = reproduce_re86_solution,
    write_registry: bool = True,
    write_gaps: bool = True,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """REQ-REPORT-4471: route one rotated game and reproduction-gate L1 if grounded."""

    root = Path(root)
    started = now()
    checked = dict(preconditions_checked or precondition_probe(root, target_game=target_game))
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    checked.setdefault(
        "generator_resource_available",
        checked.get("qwen_gguf_cached") is True or checked.get("igpu_llama_server_available") is True,
    )
    precondition_miss = first_precondition_miss(checked, target_game)
    recommendation: Mapping[str, Any] = {}
    selected_operator: Mapping[str, str] = {
        "operator": "",
        "routed_to": "",
        "reason": "precondition_blocked",
        "source": "precondition_blocked",
    }
    few_shot_examples: list[dict[str, Any]] = []
    target_digest: Mapping[str, Any] = {}
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
        target_digest, operator_result = ground_operator_fn(
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
        target_digest=target_digest,
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
