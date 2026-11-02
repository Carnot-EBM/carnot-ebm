"""Exp 4432: leave-one-out generic ARC solve baseline.

Spec refs: REQ-REPORT-4432, SCENARIO-REPORT-4432.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4432_loo_generic_solve_benchmark.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4432
SPEC_REFS = ("REQ-REPORT-4432", "SCENARIO-REPORT-4432")
TARGET_CANDIDATES = ("tr87", "tu93", "lp85", "vc33", "sc25", "ka59", "ar25", "ft09")
MIN_HELDOUT_GAMES = 6
GENERIC_SOLVE_GATE = 2
MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "generic_loo_solve_count",
    "per_game",
    "offline_reproduced",
    "missing_verifier_gaps",
    "random_seed",
    "reproducibility_checksum",
    "verifier_is_oracle",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal-prefixed (this benchmark ALWAYS completes with a measurement -- "
        "positive, partial, or zero -- so it is terminal, never partial:)"
    ),
    "generic_loo_solve_count": (
        "bare int: the count of games re-solved from the example corpus WITHOUT "
        "their own recipe -- the headline generic-capability number"
    ),
    "per_game": (
        "list of {game, solved_without_own_recipe, routed_to, residual_delta} -- "
        "shows EXACTLY which mechanics the example corpus does and does not transfer"
    ),
    "offline_reproduced": (
        "every claimed re-solve is reproduction-gated -- a non-reproducible solve "
        "does not count (ARC Solve Reproducibility Discipline)"
    ),
    "missing_verifier_gaps": (
        "each residual_delta the generic path could not induce is a "
        "missing-verifier/missing-primitive gap -- the .410 A2-A5 build backlog"
    ),
    "random_seed": "determinism precondition for a third party to re-run the benchmark",
    "reproducibility_checksum": (
        "content hash of the example-corpus snapshot used, so the LOO result is reproducible"
    ),
}

RESIDUAL_DELTA_BY_GAME = {
    "tr87": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
    "tu93": "missing_maze_goal_distance_and_fresh_env_reset_induction",
    "lp85": "missing_alignment_goal_key_and_button_discovery_verifier",
    "sc25": "missing_cast_grid_spell_shrink_tank_exit_verifier",
    "ka59": "missing_push_block_world_model_and_dynamic_selection",
    "ar25": "missing_reflection_world_model_and_object_motion_plan",
    "ft09": "missing_local_constraint_color_cycle_verifier",
}

RouteFn = Callable[[str, Mapping[str, Any], Path], Mapping[str, Any]]
AttemptFn = Callable[[str, Path, Mapping[str, Any]], Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def load_registry(root: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {"games": []}
    return loaded if isinstance(loaded, dict) else {"games": []}


def environment_games(root: Path = REPO_ROOT) -> set[str]:
    env_dir = root / "environment_files"
    if not env_dir.is_dir():
        return set()
    return {path.name for path in env_dir.iterdir() if path.is_dir()}


def _registry_games(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    games = registry.get("games")
    if not isinstance(games, list):
        return []
    return [row for row in games if isinstance(row, Mapping)]


def _entry_by_game(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("game")): row for row in _registry_games(registry) if row.get("game")}


def _is_reproduced_entry(entry: Mapping[str, Any]) -> bool:
    return entry.get("reproducibility") == "reproduced" and _as_int(entry.get("levels_reproduced")) > 0


def select_heldout_games(
    registry: Mapping[str, Any],
    *,
    env_games: set[str] | None = None,
    candidates: Sequence[str] = TARGET_CANDIDATES,
    min_k: int = MIN_HELDOUT_GAMES,
) -> list[str]:
    """REQ-REPORT-4432: choose reproduced registry games for leave-one-out folds."""

    by_game = _entry_by_game(registry)
    available_envs = env_games if env_games is not None else set(candidates)
    selected = [
        game
        for game in candidates
        if game in available_envs and game in by_game and _is_reproduced_entry(by_game[game])
    ]
    if len(selected) < min_k:
        raise ValueError(f"leave-one-out benchmark requires at least {min_k} reproduced games")
    return selected


def registry_without_game(registry: Mapping[str, Any], target: str) -> dict[str, Any]:
    """SCENARIO-REPORT-4432: return an in-memory registry view with target recipe withheld."""

    heldout = copy.deepcopy(dict(registry))
    games = heldout.get("games")
    if isinstance(games, list):
        heldout["games"] = [
            row for row in games if not (isinstance(row, Mapping) and row.get("game") == target)
        ]
    return heldout


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_env_files_present") is not True:
        return "offline_env_files"
    if preconditions.get("arc_solver_kit_import") is not True:
        return "arc_solver_kit_import"
    if preconditions.get("arc_solve_learning_import") is not True:
        return "arc_solve_learning_import"
    return None


def check_preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - live import boundary
    envs = environment_games(root)
    checks: dict[str, Any] = {
        "offline_env_files_present": bool(envs),
        "offline_env_file_count": len(envs),
        "arc_solver_kit_import": False,
        "arc_solve_learning_import": False,
        "qwen_gguf_cached": MODEL_CACHE.is_dir() and any(MODEL_CACHE.iterdir()),
    }
    try:
        from carnot.agentic import arc_solver_kit  # noqa: F401

        checks["arc_solver_kit_import"] = True
    except Exception as exc:
        checks["arc_solver_kit_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic import arc_solve_learning  # noqa: F401

        checks["arc_solve_learning_import"] = True
    except Exception as exc:
        checks["arc_solve_learning_import_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = _first_precondition_miss(checks) is None
    return checks


def recommend_approach_without_target(
    target: str,
    registry_view: Mapping[str, Any],
    _root: Path = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - live routing boundary
    """Call arc_solve_learning.recommend_approach against a temporary held-out registry."""

    from carnot.agentic import arc_solve_learning

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="carnot_loo_", dir="/tmp", delete=False) as fh:
        yaml.safe_dump(dict(registry_view), fh, sort_keys=False)
        temp_registry = Path(fh.name)
    old_registry = arc_solve_learning.REGISTRY
    try:
        arc_solve_learning.REGISTRY = temp_registry
        recommendation = dict(arc_solve_learning.recommend_approach(target))
    finally:
        arc_solve_learning.REGISTRY = old_registry
        temp_registry.unlink(missing_ok=True)

    recommended = recommendation.get("recommended")
    if isinstance(recommended, list):
        recommendation["recommended"] = [
            dict(row) for row in recommended if isinstance(row, Mapping) and row.get("game") != target
        ]
    recommendation["loo_target_recipe_withheld"] = target
    return recommendation


def _trajectory_apply(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover - ARC SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, step["action"]), data=step.get("data"))


def run_adapter_free_attempt(
    target: str,
    root: Path,
    route_result: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary
    """Run the adapter-withheld standing-loop equivalent without writing trajectory seeds."""

    from carnot.agentic import arc_solver_kit
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels

    arc = arc_solver_kit.offline_arcade()
    env = arc.make(target, scorecard_id=arc.open_scorecard())
    stats: dict[str, Any] = {}
    trajectory, reached = graph_explore_solve_v2(env, 0, max_expansions=6000, max_depth=60, stats=stats)
    labels = trajectory_labels(trajectory) if trajectory else []
    gate: dict[str, Any] = {
        "game": target,
        "claimed_level": reached,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if labels:
        gate = dict(arc_solver_kit.reproduce(target, labels, _trajectory_apply, claimed_level=reached))
    offline_reproduced = bool(gate.get("reproduced")) and _as_int(gate.get("reached_level")) >= 1
    return {
        "game": target,
        "mode": "standing_arc_loop_adapter_withheld_graph_explore",
        "route_used": dict(route_result),
        "solution_labels": labels,
        "search_reached_level": _as_int(reached),
        "stats": stats,
        "reproduction_gate": gate,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": _as_int(gate.get("reached_level")) if offline_reproduced else 0,
    }


def routed_to(route_result: Mapping[str, Any]) -> str:
    recommended = route_result.get("recommended")
    if isinstance(recommended, list) and recommended and isinstance(recommended[0], Mapping):
        return str(recommended[0].get("game") or "")
    return ""


def _slug(value: Any) -> str:
    text = str(value or "").lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(chars).split("_") if part)


def residual_delta_for(
    target: str,
    registry: Mapping[str, Any],
    *,
    solved: bool,
    blocked_reason: str | None = None,
) -> str:
    if solved:
        return "none"
    if blocked_reason:
        return blocked_reason
    if target in RESIDUAL_DELTA_BY_GAME:
        return RESIDUAL_DELTA_BY_GAME[target]
    entry = _entry_by_game(registry).get(target, {})
    mechanic = entry.get("mechanic_class")
    if mechanic:
        return f"missing_{_slug(mechanic)}_verifier_or_primitive"
    if entry.get("world_model"):
        return "missing_executable_world_model_transfer"
    return "missing_generic_goal_discriminator"


def _attempt_solved(attempt: Mapping[str, Any]) -> bool:
    gate = attempt.get("reproduction_gate")
    return (
        isinstance(gate, Mapping)
        and gate.get("reproduced") is True
        and _as_int(gate.get("reached_level")) >= 1
        and attempt.get("offline_reproduced") is True
    )


def _primitive_hashes(root: Path) -> dict[str, str]:
    rel_paths = (
        "scripts/arc_loop_solve.py",
        "python/carnot/agentic/arc_graph_explore.py",
        "python/carnot/agentic/arc_solver_kit.py",
        "python/carnot/agentic/arc_solve_learning.py",
    )
    hashes: dict[str, str] = {}
    for rel_path in rel_paths:
        path = root / rel_path
        if path.exists():
            hashes[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _example_corpus_snapshot(
    registry: Mapping[str, Any],
    heldout_games: Sequence[str],
    root: Path,
) -> dict[str, Any]:
    return {
        "heldout_games": list(heldout_games),
        "fold_example_games": {
            target: [
                str(row.get("game"))
                for row in _registry_games(registry_without_game(registry, target))
                if _is_reproduced_entry(row)
            ]
            for target in heldout_games
        },
        "general_gotchas": registry.get("general_gotchas", []),
        "primitive_hashes": _primitive_hashes(root),
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    started_at: float,
    ended_at: float,
    root: Path,
) -> dict[str, Any]:
    snapshot = {
        "blocked_reason": reason,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4432_loo_generic_solve_benchmark",
        "schema": "carnot.exp4432.loo_generic_solve_benchmark.v1",
        "honest_verdict": f"blocked_{reason}",
        "generic_loo_solve_count": 0,
        "heldout_games": [],
        "loo_gate_passed": False,
        "per_game": [],
        "attempts": [],
        "offline_reproduced": False,
        "missing_verifier_gaps": [],
        "preconditions_checked": dict(preconditions_checked),
        "verifier_is_oracle": True,
        "verifier_oracle_note": "execution-grounded reproduction gate; no leaderboard submission",
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "result_path": RESULT_RELATIVE_PATH,
        "leaderboard_submission": False,
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "reproducibility_checksum": _sha256(snapshot | {"primitive_hashes": _primitive_hashes(root)}),
    }


def build_artifact(
    *,
    root: Path,
    registry: Mapping[str, Any],
    heldout_games: Sequence[str],
    attempts: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    per_game: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for attempt in attempts:
        game = str(attempt.get("game") or "")
        route = attempt.get("route_result") if isinstance(attempt.get("route_result"), Mapping) else {}
        solved = _attempt_solved(attempt)
        residual = residual_delta_for(
            game,
            registry,
            solved=solved,
            blocked_reason=str(attempt.get("blocked_reason")) if attempt.get("blocked_reason") else None,
        )
        routed = routed_to(route)
        row = {
            "game": game,
            "solved_without_own_recipe": solved,
            "routed_to": routed,
            "residual_delta": residual,
        }
        per_game.append(row)
        if not solved:
            gaps.append(
                {
                    "game": game,
                    "routed_to": routed,
                    "residual_delta": residual,
                    "attempt_mode": attempt.get("mode", ""),
                }
            )

    solve_count = sum(1 for row in per_game if row["solved_without_own_recipe"])
    gate_passed = solve_count >= GENERIC_SOLVE_GATE
    gate_word = "passed" if gate_passed else "failed"
    snapshot = _example_corpus_snapshot(registry, heldout_games, root)
    return {
        "experiment": "experiment_4432_loo_generic_solve_benchmark",
        "schema": "carnot.exp4432.loo_generic_solve_benchmark.v1",
        "honest_verdict": (
            f"complete: generic_loo_solve_count_{solve_count}_of_{len(heldout_games)}_gate_{gate_word}"
        ),
        "generic_loo_solve_count": solve_count,
        "heldout_games": list(heldout_games),
        "loo_gate_passed": gate_passed,
        "loo_gate": f"generic_loo_solve_count >= {GENERIC_SOLVE_GATE}",
        "per_game": per_game,
        "attempts": [dict(attempt) for attempt in attempts],
        "offline_reproduced": all(
            not attempt.get("solved_without_own_recipe") or _attempt_solved(attempt)
            for attempt in attempts
        ),
        "missing_verifier_gaps": gaps,
        "preconditions_checked": dict(preconditions_checked),
        "example_corpus_snapshot": {
            "heldout_games": snapshot["heldout_games"],
            "fold_example_games": snapshot["fold_example_games"],
            "primitive_hashes": snapshot["primitive_hashes"],
        },
        "verifier_is_oracle": True,
        "verifier_oracle_note": "execution-grounded reproduction gate; no leaderboard submission",
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "result_path": RESULT_RELATIVE_PATH,
        "leaderboard_submission": False,
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "reproducibility_checksum": _sha256(snapshot | {"random_seed": RANDOM_SEED}),
    }


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _terminal_or_blocked(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("complete:", "success:", "passed:", "shipped:", "blocked_", "blocked:"))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not _terminal_or_blocked(verdict) or (isinstance(verdict, str) and verdict.startswith("partial:")):
        errors.append("honest_verdict must be terminal-prefixed for Exp 4432")
    if type(artifact.get("generic_loo_solve_count")) is not int:
        errors.append("generic_loo_solve_count must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    per_game = artifact.get("per_game")
    if not isinstance(per_game, list):
        errors.append("per_game must be list")
    else:
        solved_count = 0
        for index, row in enumerate(per_game):
            if not isinstance(row, Mapping):
                errors.append(f"per_game[{index}] must be dict")
                continue
            for field in ("game", "solved_without_own_recipe", "routed_to", "residual_delta"):
                if field not in row:
                    errors.append(f"per_game[{index}] missing {field}")
            if type(row.get("solved_without_own_recipe")) is not bool:
                errors.append(f"per_game[{index}].solved_without_own_recipe must be bare bool")
            solved_count += int(row.get("solved_without_own_recipe") is True)
        if isinstance(artifact.get("generic_loo_solve_count"), int) and solved_count != artifact.get(
            "generic_loo_solve_count"
        ):
            errors.append("generic_loo_solve_count must match solved per_game rows")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    attempts = artifact.get("attempts")
    if not isinstance(attempts, list):
        errors.append("attempts must be list")
    else:
        for attempt in attempts:
            if isinstance(attempt, Mapping) and attempt.get("solved_without_own_recipe") is True:
                gate = attempt.get("reproduction_gate")
                if not isinstance(gate, Mapping) or gate.get("reproduced") is not True or _as_int(
                    gate.get("reached_level")
                ) < 1:
                    errors.append("solved attempts must have reproduced gate evidence")
                    break
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
    route_fn: RouteFn = recommend_approach_without_target,
    attempt_fn: AttemptFn = run_adapter_free_attempt,
    preconditions_checked: Mapping[str, Any] | None = None,
    llm_induction_games: set[str] | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-4432: run the held-out generic solve measurement and write JSON."""

    started = now()
    preconditions = dict(preconditions_checked or check_preconditions(root))
    miss = _first_precondition_miss(preconditions)
    if miss:
        artifact = _blocked_artifact(
            reason=miss,
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=now(),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    registry = load_registry(root)
    try:
        heldout_games = select_heldout_games(registry, env_games=environment_games(root))
    except ValueError:
        artifact = _blocked_artifact(
            reason="reproducible_target_count",
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=now(),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    attempts: list[dict[str, Any]] = []
    model_cached = preconditions.get("qwen_gguf_cached") is True
    llm_targets = set(llm_induction_games or ())
    for target in heldout_games:
        if target in llm_targets and not model_cached:
            attempts.append(
                {
                    "game": target,
                    "mode": "blocked_model_not_cached",
                    "route_result": {},
                    "reproduction_gate": {"game": target, "reproduced": False, "reached_level": 0},
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "solved_without_own_recipe": False,
                    "blocked_reason": "blocked_model_not_cached",
                }
            )
            continue
        registry_view = registry_without_game(registry, target)
        route_result = dict(route_fn(target, registry_view, root))
        attempt = dict(attempt_fn(target, root, route_result))
        solved = _attempt_solved(attempt)
        attempt["game"] = target
        attempt["route_result"] = route_result
        attempt["solved_without_own_recipe"] = solved
        attempts.append(attempt)

    artifact = build_artifact(
        root=root,
        registry=registry,
        heldout_games=heldout_games,
        attempts=attempts,
        preconditions_checked=preconditions,
        started_at=started,
        ended_at=now(),
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
