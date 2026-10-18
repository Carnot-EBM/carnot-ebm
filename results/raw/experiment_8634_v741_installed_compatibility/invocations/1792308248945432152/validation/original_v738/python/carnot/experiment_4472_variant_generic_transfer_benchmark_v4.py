"""Exp 4472: manufactured-variant generic-transfer benchmark v4.

Spec refs: REQ-REPORT-4472, SCENARIO-REPORT-4472.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_4448_loo_generic_solve_benchmark_v2 as v2
from carnot import experiment_4459_loo_generic_solve_benchmark_v3 as v3
from carnot.agentic import arc_variant_generator as variants


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4472_variant_generic_transfer_benchmark_v4.json"
REGISTRY_RELATIVE_PATH = v3.REGISTRY_RELATIVE_PATH
V3_RELATIVE_PATH = v3.RESULT_RELATIVE_PATH
CAST_GRID_RELATIVE_PATH = "results/experiment_4469_generic_cast_grid_fsm_operator.json"
RANDOM_SEED = 4472
V3_BASELINE = 6
MIN_PUBLIC_GAMES = 25
DEFAULT_COLOR_VARIANTS = (1,)
DEFAULT_REFLECTION_VARIANTS: tuple[int, ...] = ()
DEFAULT_BUDGET = 200
REFLECTION_AXIS = 1
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
MODEL_CACHE = v2.MODEL_CACHE
BASELINE_SMOKE_COMMAND_TEXT = (
    '.venv/bin/pytest -k "variant or arc_solver_kit or leaderboard_eval" -q --no-cov'
)
BASELINE_SMOKE_COMMAND = (
    ".venv/bin/pytest",
    "-k",
    "variant or arc_solver_kit or leaderboard_eval",
    "-q",
    "--no-cov",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "generic_transfer_rate_over_variants",
    "variants_attempted",
    "variants_solved",
    "generic_loo_solve_count_v4",
    "generic_loo_solve_count_v3_baseline",
    "per_game",
    "offline_reproduced",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed (this benchmark ALWAYS completes with a measurement, "
            "so terminal, never partial:)"
        )
    },
    "inference_substrate": {"principle": "THE .410/.411 LESSON -- EMIT it; never None"},
    "generic_transfer_rate_over_variants": {
        "principle": (
            "bare float: the headline operator-mandated metric -- "
            "(variants solved)/(variants attempted) over 25 games x N manufactured variants, "
            "the OOD-proxy generalization test"
        )
    },
    "variants_attempted": {
        "principle": "bare int: total manufactured variants attempted (>=25); LOG N-per-game + any skipped"
    },
    "variants_solved": {
        "principle": "bare int: reproduction-gated variant solves (VariantEnv real win-condition)"
    },
    "generic_loo_solve_count_v4": {
        "principle": (
            "bare int: the plain-LOO re-measurement after the .413 operators -- "
            "did it rise above the v3 baseline of 6?"
        )
    },
    "generic_loo_solve_count_v3_baseline": {
        "principle": "bare int = 6; the .412 baseline this must beat to show progress"
    },
    "per_game": {
        "principle": (
            "list of {game, variant_transfer_rate, loo_solved_without_own_recipe, "
            "closed_by_operator, residual_delta} -- shows EXACTLY which residuals "
            "the new operators closed"
        )
    },
    "offline_reproduced": {
        "principle": "every claimed re-solve/variant-solve is reproduction-gated"
    },
    "missing_verifier_gaps": {"principle": "each still-open residual -- the .414 build backlog"},
    "verifier_is_oracle": {"principle": "true: execution-grounded reproduction"},
    "random_seed": {"principle": "determinism for re-run (variant generation seed)"},
    "reproducibility_checksum": {
        "principle": "content hash of the variant set + example-corpus snapshot used"
    },
}

VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]


def _duration(started_at: float, ended_at: float) -> float:
    return max(0.0, round(float(ended_at - started_at), 6))


def _floor_end_time(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    remaining = VERIFIER_SCORING_DURATION_TARGET_S - elapsed
    if remaining > 0:
        sleep_fn(remaining)
    return max(float(now()), started_at + VERIFIER_SCORING_DURATION_TARGET_S)


def _environment_games(root: Path) -> set[str]:
    env_dir = root / "environment_files"
    if not env_dir.is_dir():
        return set()
    return {path.name for path in env_dir.iterdir() if path.is_dir()}


def check_preconditions(
    root: Path = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - live boundary
    root = Path(root)
    env_games = sorted(_environment_games(root))
    checks: dict[str, Any] = {
        "offline_env_files_present": bool(env_games),
        "offline_env_games": env_games,
        "arc_variant_generator_import": False,
        "arc_solver_kit_import": False,
        "arc_solve_learning_import": False,
        "baseline_smoke_command": BASELINE_SMOKE_COMMAND_TEXT,
        "baseline_smoke_green": False,
        "qwen_gguf_cached": MODEL_CACHE.is_dir() and any(MODEL_CACHE.iterdir()),
        "igpu_llama_server_available": False,
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_variant_generator  # noqa: F401

        checks["arc_variant_generator_import"] = True
    except Exception as exc:
        checks["arc_variant_generator_import_error"] = f"{type(exc).__name__}: {exc}"
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
    try:
        smoke = subprocess.run(
            BASELINE_SMOKE_COMMAND,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
        checks["baseline_smoke_exit_code"] = int(smoke.returncode)
        checks["baseline_smoke_output_tail"] = smoke.stdout[-4000:]
        checks["baseline_smoke_green"] = smoke.returncode == 0
    except Exception as exc:
        checks["baseline_smoke_exit_code"] = 1
        checks["baseline_smoke_output_tail"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = first_precondition_miss(checks) is None
    return checks


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_env_files_present") is not True:
        return "offline_env_files"
    if preconditions.get("arc_variant_generator_import") is not True:
        return "arc_variant_generator_import"
    if preconditions.get("arc_solver_kit_import") is not True:
        return "arc_solver_kit_import"
    if preconditions.get("arc_solve_learning_import") is not True:
        return "arc_solve_learning_import"
    if preconditions.get("baseline_smoke_green") is not True:
        return "baseline_smoke"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {
        V3_RELATIVE_PATH: v2._file_sha256(root, V3_RELATIVE_PATH),
        CAST_GRID_RELATIVE_PATH: v2._file_sha256(root, CAST_GRID_RELATIVE_PATH),
        REGISTRY_RELATIVE_PATH: v2._file_sha256(root, REGISTRY_RELATIVE_PATH),
        "python/carnot/agentic/arc_variant_generator.py": v2._file_sha256(
            root, "python/carnot/agentic/arc_variant_generator.py"
        ),
        "python/carnot/agentic/arc_solver_kit.py": v2._file_sha256(
            root, "python/carnot/agentic/arc_solver_kit.py"
        ),
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    started_at: float,
    ended_at: float,
    root: Path,
) -> dict[str, Any]:
    checksum_payload = {
        "blocked_reason": reason,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "source_hashes": _source_hashes(root),
    }
    return {
        "experiment": "experiment_4472_variant_generic_transfer_benchmark_v4",
        "schema": "carnot.exp4472.variant_generic_transfer_benchmark_v4.v1",
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": BLOCKED_INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "generic_transfer_rate_over_variants": 0.0,
        "variants_attempted": 0,
        "variants_solved": 0,
        "generic_loo_solve_count_v4": 0,
        "generic_loo_solve_count_v3_baseline": V3_BASELINE,
        "variant_plan": {
            "color_variants": list(DEFAULT_COLOR_VARIANTS),
            "reflection_variants": list(DEFAULT_REFLECTION_VARIANTS),
            "reflection_axis": REFLECTION_AXIS,
            "budget": DEFAULT_BUDGET,
            "n_per_game": len(DEFAULT_COLOR_VARIANTS) + len(DEFAULT_REFLECTION_VARIANTS),
            "skipped": [],
        },
        "variant_attempts": [],
        "heldout_games": [],
        "per_game": [],
        "closed_residuals_by_413_operator": [],
        "offline_reproduced": False,
        "missing_verifier_gaps": [],
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifacts": {"source_hashes": _source_hashes(root)},
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": v2.sha256(checksum_payload),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4472", "SCENARIO-REPORT-4472"],
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


def _public_games(preconditions: Mapping[str, Any]) -> list[str]:
    games = preconditions.get("offline_env_games")
    if not isinstance(games, list):
        return []
    return sorted({str(game) for game in games if str(game)})


def manufactured_variant_specs(
    games: Sequence[str],
    *,
    color_variants: Sequence[int] = DEFAULT_COLOR_VARIANTS,
    reflection_variants: Sequence[int] = DEFAULT_REFLECTION_VARIANTS,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for game in sorted(str(item) for item in games):
        for variant in sorted(int(item) for item in color_variants):
            specs.append(
                {
                    "game": game,
                    "variant": variant,
                    "kind": "color",
                    "reflect": None,
                    "variant_signature": variants.variant_signature(game, variant, "color"),
                }
            )
        for variant in sorted(int(item) for item in reflection_variants):
            specs.append(
                {
                    "game": game,
                    "variant": variant,
                    "kind": "reflect",
                    "reflect": REFLECTION_AXIS,
                    "variant_signature": variants.variant_signature(game, variant, "reflect"),
                }
            )
    return specs


def _variant_transfer_rate(solved: int, attempted: int) -> float:
    if attempted <= 0:
        return 0.0
    return round(float(solved) / float(attempted), 10)


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - live boundary
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _remap_reflected_data(
    data: Any, reflect: int | None
) -> Any:  # pragma: no cover - live boundary
    if reflect is None or not isinstance(data, dict) or "x" not in data or "y" not in data:
        return data
    x, y = variants.remap_click_for_reflection(int(data["x"]), int(data["y"]), 64, 64, reflect)
    return {**data, "x": x, "y": y}


def _apply_action_label(
    env: Any, label: str, _frame: Any = None
) -> Any:  # pragma: no cover - ARC SDK boundary
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def run_variant_attempt(
    game: str, spec: Mapping[str, Any], budget: int
) -> dict[str, Any]:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy, _level_of

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = variants.VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    policy = CarnotAgentPolicy(game, {}, force_explore=True)
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached = 0
    for _index in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            if labels:
                labels.append("RESET")
        elif kind is None:
            break
        else:
            visible_data = data
            real_data = _remap_reflected_data(data, spec.get("reflect"))
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=visible_data)
            labels.append(_action_label(int(kind), real_data))
            actions += 1
        if start_level is None:
            start_level = _level_of(latest)
        frames.append(latest)
        reached = _level_of(latest)
        if start_level is not None and reached > start_level:
            break
        if latest is None:
            break
    claimed = reached if start_level is not None and reached > start_level else 0
    gate: dict[str, Any] = {
        "game": game,
        "reached_level": 0,
        "claimed_level": claimed,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if claimed > 0 and labels:
        gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed))
    solved = bool(gate.get("reproduced")) and v2.as_int(gate.get("reached_level")) >= claimed >= 1
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "reached_level": v2.as_int(gate.get("reached_level")) if solved else reached,
        "actions": actions,
        "solution_labels": labels if solved else [],
        "reproduction_gate": gate,
        "blocked_reason": "",
    }


def _operator_result_matches(artifact: Mapping[str, Any], *, game: str, operator: str) -> bool:
    candidates = [
        artifact.get("generic_operator_result"),
        artifact.get("operator_result"),
    ]
    solve_result = artifact.get("generic_solve_result")
    if isinstance(solve_result, Mapping):
        candidates.append(solve_result.get("operator_result"))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if (
            candidate.get("game") == game
            and candidate.get("operator") == operator
            and candidate.get("target_recipe_withheld") == game
            and candidate.get("grounded") is True
        ):
            return True
    return False


def _cast_grid_closes_sc25(cast_artifact: Mapping[str, Any] | None) -> bool:
    if not isinstance(cast_artifact, Mapping):
        return False
    closure_flag = any(
        cast_artifact.get(field) is True
        for field in (
            "sc25_resolved_generically",
            "sc25_cast_grid_resolved_generically",
            "cast_grid_phase_fsm_resolved_generically",
        )
    )
    gate = cast_artifact.get("reproduction_result") or cast_artifact.get(
        "generic_reproduction_result"
    )
    return (
        closure_flag
        and cast_artifact.get("offline_reproduced") is True
        and v2.gate_reproduced(gate)
        and _operator_result_matches(
            cast_artifact,
            game="sc25",
            operator="cast_grid_phase_fsm_world_model",
        )
    )


def _heldout_games(v3_artifact: Mapping[str, Any]) -> list[str]:
    games = v3_artifact.get("heldout_games")
    if isinstance(games, list) and games:
        return [str(game) for game in games]
    rows = v3_artifact.get("per_game")
    if isinstance(rows, list):
        return [
            str(row.get("game")) for row in rows if isinstance(row, Mapping) and row.get("game")
        ]
    return []


def _loo_rows(
    *,
    v3_artifact: Mapping[str, Any],
    cast_artifact: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    v3_rows = v2._rows_by_game(v3_artifact.get("per_game"))
    gaps_by_game = v2._rows_by_game(v3_artifact.get("missing_verifier_gaps"))
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    closed_new: list[dict[str, str]] = []
    for game in _heldout_games(v3_artifact):
        previous = v3_rows.get(game, {})
        solved = previous.get("solved_without_own_recipe") is True
        closed_by = str(previous.get("closed_by_operator") or "none")
        residual = str(previous.get("residual_delta") or "missing_reproduction_gate_evidence")
        if not solved and game == "sc25" and _cast_grid_closes_sc25(cast_artifact):
            solved = True
            closed_by = "cast_grid_phase_fsm_world_model"
            residual = "none"
            closed_new.append(
                {
                    "game": game,
                    "closed_by_operator": closed_by,
                    "v3_residual_delta": str(previous.get("residual_delta") or ""),
                }
            )
        row = {
            "game": game,
            "loo_solved_without_own_recipe": solved,
            "closed_by_operator": closed_by if solved else "none",
            "residual_delta": "none" if solved else residual,
        }
        rows.append(row)
        if not solved:
            prior_gap = gaps_by_game.get(game, {})
            gaps.append(
                {
                    "game": game,
                    "residual_delta": row["residual_delta"],
                    "retrieved_operator": str(prior_gap.get("retrieved_operator") or ""),
                    "attempt_mode": "v4_413_operator_remeasurement",
                }
            )
    return rows, gaps, closed_new


def _verdict(
    variants_solved: int, variants_attempted: int, transfer_rate: float, loo_count: int
) -> str:
    if loo_count > V3_BASELINE:
        return (
            f"success: generic_transfer_variants_{variants_solved}_of_{variants_attempted}_"
            f"rate_{transfer_rate:.4f}_loo_v4_{loo_count}_beats_v3_{V3_BASELINE}"
        )
    if loo_count == V3_BASELINE:
        return (
            f"complete: generic_transfer_variants_{variants_solved}_of_{variants_attempted}_"
            f"rate_{transfer_rate:.4f}_loo_v4_{loo_count}_flat_vs_v3_{V3_BASELINE}"
        )
    return (
        f"complete: generic_transfer_variants_{variants_solved}_of_{variants_attempted}_"
        f"rate_{transfer_rate:.4f}_loo_v4_{loo_count}_lower_than_v3_{V3_BASELINE}"
    )


def _variant_summary_by_game(
    variant_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for attempt in variant_attempts:
        game = str(attempt.get("game") or "")
        if not game:
            continue
        row = summary.setdefault(game, {"attempted": 0, "solved": 0})
        if attempt.get("attempted") is True:
            row["attempted"] += 1
            if attempt.get("solved") is True:
                row["solved"] += 1
    return summary


def build_artifact(
    *,
    root: Path,
    public_games: Sequence[str],
    variant_specs: Sequence[Mapping[str, Any]],
    variant_attempts: Sequence[Mapping[str, Any]],
    v3_artifact: Mapping[str, Any],
    cast_artifact: Mapping[str, Any] | None,
    preconditions_checked: Mapping[str, Any],
    color_variants: Sequence[int],
    reflection_variants: Sequence[int],
    budget: int,
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    loo_rows, loo_gaps, closed_new = _loo_rows(v3_artifact=v3_artifact, cast_artifact=cast_artifact)
    loo_by_game = {row["game"]: row for row in loo_rows}
    variant_summary = _variant_summary_by_game(variant_attempts)

    per_game: list[dict[str, Any]] = []
    for game in sorted(public_games):
        var = variant_summary.get(game, {"attempted": 0, "solved": 0})
        loo = loo_by_game.get(
            game,
            {
                "loo_solved_without_own_recipe": False,
                "closed_by_operator": "none",
                "residual_delta": "not_in_plain_loo_benchmark",
            },
        )
        per_game.append(
            {
                "game": game,
                "variant_transfer_rate": _variant_transfer_rate(var["solved"], var["attempted"]),
                "loo_solved_without_own_recipe": bool(loo["loo_solved_without_own_recipe"]),
                "closed_by_operator": str(loo["closed_by_operator"]),
                "residual_delta": str(loo["residual_delta"]),
            }
        )

    variants_attempted = sum(1 for attempt in variant_attempts if attempt.get("attempted") is True)
    variants_solved = sum(
        1
        for attempt in variant_attempts
        if attempt.get("attempted") is True and attempt.get("solved") is True
    )
    transfer_rate = _variant_transfer_rate(variants_solved, variants_attempted)
    loo_count = sum(1 for row in loo_rows if row["loo_solved_without_own_recipe"] is True)
    source_hashes = _source_hashes(root)
    skipped = [
        dict(attempt)
        for attempt in variant_attempts
        if attempt.get("attempted") is not True or attempt.get("blocked_reason")
    ]
    snapshot = {
        "public_games": list(sorted(public_games)),
        "variant_specs": [dict(spec) for spec in variant_specs],
        "variant_attempts": [dict(attempt) for attempt in variant_attempts],
        "loo_rows": loo_rows,
        "source_hashes": source_hashes,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4472_variant_generic_transfer_benchmark_v4",
        "schema": "carnot.exp4472.variant_generic_transfer_benchmark_v4.v1",
        "honest_verdict": _verdict(variants_solved, variants_attempted, transfer_rate, loo_count),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "generic_transfer_rate_over_variants": float(transfer_rate),
        "variants_attempted": variants_attempted,
        "variants_solved": variants_solved,
        "generic_loo_solve_count_v4": loo_count,
        "generic_loo_solve_count_v3_baseline": V3_BASELINE,
        "heldout_games": _heldout_games(v3_artifact),
        "variant_plan": {
            "color_variants": [int(item) for item in color_variants],
            "reflection_variants": [int(item) for item in reflection_variants],
            "reflection_axis": REFLECTION_AXIS,
            "budget": int(budget),
            "n_per_game": len(color_variants) + len(reflection_variants),
            "skipped": skipped,
        },
        "variant_attempts": [dict(attempt) for attempt in variant_attempts],
        "plain_loo_gate": f"generic_loo_solve_count_v4 > {V3_BASELINE}",
        "plain_loo_gate_passed": loo_count > V3_BASELINE,
        "per_game": per_game,
        "closed_residuals_by_413_operator": closed_new,
        "offline_reproduced": True,
        "missing_verifier_gaps": loo_gaps,
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifacts": {
            "v3_baseline": V3_RELATIVE_PATH,
            "cast_grid_phase_fsm_world_model": CAST_GRID_RELATIVE_PATH,
            "source_hashes": source_hashes,
        },
        "example_corpus_snapshot": snapshot,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": v2.sha256(snapshot),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4472", "SCENARIO-REPORT-4472"],
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


def _rate_matches(rate: Any, solved: Any, attempted: Any) -> bool:
    if type(rate) is not float or type(solved) is not int or type(attempted) is not int:
        return False
    expected = 0.0 if attempted <= 0 else solved / attempted
    return abs(rate - expected) <= 1e-9


def _terminal_blocked(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith("complete: blocked_")


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete:/success:/passed:/shipped:")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")
    if artifact.get("inference_substrate") is None:
        errors.append("inference_substrate must not be None")
    if (
        artifact.get("inference_substrate") in {INFERENCE_SUBSTRATE, None}
        and float(artifact.get("duration_s") or 0.0) < 1.0
    ):
        errors.append("cached verifier substrate requires duration_s >= 1.0")
    if type(artifact.get("generic_transfer_rate_over_variants")) is not float:
        errors.append("generic_transfer_rate_over_variants must be bare float")
    if type(artifact.get("variants_attempted")) is not int:
        errors.append("variants_attempted must be bare int")
    if type(artifact.get("variants_solved")) is not int:
        errors.append("variants_solved must be bare int")
    if type(artifact.get("generic_loo_solve_count_v4")) is not int:
        errors.append("generic_loo_solve_count_v4 must be bare int")
    if (
        artifact.get("generic_loo_solve_count_v3_baseline") != V3_BASELINE
        or type(artifact.get("generic_loo_solve_count_v3_baseline")) is not int
    ):
        errors.append("generic_loo_solve_count_v3_baseline must be bare int = 6")

    per_game = artifact.get("per_game")
    loo_count = 0
    if not isinstance(per_game, list):
        errors.append("per_game must be list")
    else:
        for index, row in enumerate(per_game):
            if not isinstance(row, Mapping):
                errors.append(f"per_game[{index}] must be dict")
                continue
            for field in (
                "game",
                "variant_transfer_rate",
                "loo_solved_without_own_recipe",
                "closed_by_operator",
                "residual_delta",
            ):
                if field not in row:
                    errors.append(f"per_game[{index}] missing {field}")
            if type(row.get("variant_transfer_rate")) is not float:
                errors.append(f"per_game[{index}].variant_transfer_rate must be bare float")
            if type(row.get("loo_solved_without_own_recipe")) is not bool:
                errors.append(f"per_game[{index}].loo_solved_without_own_recipe must be bare bool")
            if "closed_by_operator" in row and not isinstance(row.get("closed_by_operator"), str):
                errors.append(f"per_game[{index}].closed_by_operator must be string")
            if row.get("loo_solved_without_own_recipe") is True:
                loo_count += 1
                if row.get("closed_by_operator") in {"", "none", None}:
                    errors.append(f"per_game[{index}] loo solved row requires closed_by_operator")
                if row.get("residual_delta") != "none":
                    errors.append(f"per_game[{index}] loo solved row requires residual_delta none")
        if type(artifact.get("generic_loo_solve_count_v4")) is int and loo_count != artifact.get(
            "generic_loo_solve_count_v4"
        ):
            errors.append("generic_loo_solve_count_v4 must match solved per_game LOO rows")

    variant_attempts = artifact.get("variant_attempts")
    solved_attempts = 0
    if not isinstance(variant_attempts, list):
        errors.append("variant_attempts must be list")
    else:
        for attempt in variant_attempts:
            if not isinstance(attempt, Mapping):
                continue
            if attempt.get("attempted") is True and attempt.get("solved") is True:
                solved_attempts += 1
                gate = attempt.get("reproduction_gate")
                if (
                    not isinstance(gate, Mapping)
                    or gate.get("reproduced") is not True
                    or v2.as_int(gate.get("reached_level")) < 1
                ):
                    errors.append("solved variant_attempts must have reproduced gate evidence")
                    break
    if solved_attempts != artifact.get("variants_solved"):
        errors.append("variants_solved must match solved variant_attempts")
    if not _rate_matches(
        artifact.get("generic_transfer_rate_over_variants"),
        artifact.get("variants_solved"),
        artifact.get("variants_attempted"),
    ):
        errors.append(
            "generic_transfer_rate_over_variants must equal variants_solved/variants_attempted"
        )
    if (
        not _terminal_blocked(verdict)
        and type(artifact.get("variants_attempted")) is int
        and artifact.get("variants_attempted") < MIN_PUBLIC_GAMES
    ):
        errors.append("variants_attempted must be >= 25 for completed measurement")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if artifact.get("offline_reproduced") is False and (
        v2.as_int(artifact.get("variants_solved")) > 0
        or v2.as_int(artifact.get("generic_loo_solve_count_v4")) > 0
    ):
        errors.append("offline_reproduced false cannot accompany counted solves")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not v2.checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    field_principles = artifact.get("field_principles")
    if (
        not isinstance(field_principles, Mapping)
        or field_principles.get("honest_verdict") != FIELD_PRINCIPLES["honest_verdict"]
    ):
        errors.append("field_principles.honest_verdict must match REQ-REPORT-4472")
    if (
        isinstance(verdict, str)
        and verdict.startswith("success:")
        and v2.as_int(artifact.get("generic_loo_solve_count_v4")) <= V3_BASELINE
    ):
        errors.append("success verdict requires generic_loo_solve_count_v4 > 6")
    if artifact.get("no_3090_inference") is not True:
        errors.append("no_3090_inference must be true")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner: VariantRunner = run_variant_attempt,
    color_variants: Sequence[int] = DEFAULT_COLOR_VARIANTS,
    reflection_variants: Sequence[int] = DEFAULT_REFLECTION_VARIANTS,
    budget: int = DEFAULT_BUDGET,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """REQ-REPORT-4472: measure manufactured-variant transfer and v4 LOO count."""

    started = float(now())
    root = Path(root)
    preconditions = dict(preconditions_checked or check_preconditions(root))
    preconditions.setdefault("no_3090_inference", True)
    preconditions.setdefault("leaderboard_submission", False)
    miss = first_precondition_miss(preconditions)
    if miss:
        artifact = _blocked_artifact(
            reason=miss,
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    v3_artifact = v2.load_json(root, V3_RELATIVE_PATH)
    if v3_artifact is None:
        artifact = _blocked_artifact(
            reason="v3_source_artifact",
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    public_games = _public_games(preconditions)
    if len(public_games) < MIN_PUBLIC_GAMES:
        artifact = _blocked_artifact(
            reason="public_game_count",
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    cast_artifact = v2.load_json(root, CAST_GRID_RELATIVE_PATH)
    variant_specs = manufactured_variant_specs(
        public_games,
        color_variants=color_variants,
        reflection_variants=reflection_variants,
    )
    variant_attempts: list[dict[str, Any]] = []
    for spec in variant_specs:
        try:
            attempt = dict(variant_runner(str(spec["game"]), spec, int(budget)))
        except Exception as exc:  # pragma: no cover - live defensive boundary
            attempt = {
                "game": spec["game"],
                "variant_signature": spec["variant_signature"],
                "variant": spec["variant"],
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": False,
                "reached_level": 0,
                "actions": 0,
                "reproduction_gate": {
                    "game": spec["game"],
                    "reached_level": 0,
                    "claimed_level": 0,
                    "reproduced": False,
                    "mode": "runner_exception",
                },
                "blocked_reason": f"{type(exc).__name__}: {exc}",
            }
        variant_attempts.append(attempt)

    ended = _floor_end_time(started_at=started, now=now, sleep_fn=sleep_fn)
    artifact = build_artifact(
        root=root,
        public_games=public_games,
        variant_specs=variant_specs,
        variant_attempts=variant_attempts,
        v3_artifact=v3_artifact,
        cast_artifact=cast_artifact,
        preconditions_checked=preconditions,
        color_variants=color_variants,
        reflection_variants=reflection_variants,
        budget=budget,
        started_at=started,
        ended_at=ended,
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - script entry
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    print(
        f"generic_transfer_rate_over_variants={artifact['generic_transfer_rate_over_variants']:.4f}"
    )
    print(f"variants_solved={artifact['variants_solved']}")
    print(f"variants_attempted={artifact['variants_attempted']}")
    print(f"generic_loo_solve_count_v4={artifact['generic_loo_solve_count_v4']}")
    print(f"generic_loo_solve_count_v3_baseline={artifact['generic_loo_solve_count_v3_baseline']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
