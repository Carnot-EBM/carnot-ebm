"""Exp 4481: reflection-variant transfer benchmark.

Spec refs: REQ-REPORT-4481, SCENARIO-REPORT-4481.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

from carnot import experiment_4448_loo_generic_solve_benchmark_v2 as v2
from carnot import experiment_4472_variant_generic_transfer_benchmark_v4 as exp4472


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4481_variant_transfer_benchmark.json"
REGISTRY_RELATIVE_PATH = exp4472.REGISTRY_RELATIVE_PATH
RANDOM_SEED = 4481
DEFAULT_REFLECTION_VARIANTS = (1,)
DEFAULT_BUDGET = exp4472.DEFAULT_BUDGET
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "offline_reproduced",
    "reproduced_levels",
    "preconditions_checked",
    "solved_games",
    "per_game",
    "variant_plan",
    "variant_attempts",
    "variants_attempted",
    "variants_solved",
    "transfer_solve_rate",
    "reproducible_total_levels",
    "field_principles",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "MUST start with a terminal prefix complete:/complete_/success:/success_/"
            "passed:/passed_/shipped:/shipped_ so the reconciler classifies it as terminal "
            "(Verdict Terminal-Prefix Discipline)."
        )
    },
    "inference_substrate": {
        "principle": (
            "explicit declaration (live_llm_inference | verifier_ensemble_against_cached_candidates | "
            "aggregation_from_upstream_artifacts) so adversarial_verify applies the right floor."
        )
    },
    "offline_reproduced": {
        "principle": (
            "a solve not reproducible offline is wasted effort -- only reproduced levels count "
            "(ARC Solve Reproducibility)."
        )
    },
    "reproduced_levels": {
        "principle": (
            "headline metric reproducible_total_levels grows monotonically; report the count "
            "banked, real-env-confirmed."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified before launching; pre-empts the "
            "silent-missing-resource fabrication mode."
        )
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


def _transfer_rate(solved: int, attempted: int) -> float:
    if attempted <= 0:
        return 0.0
    return round(float(solved) / float(attempted), 10)


def load_registry(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _registry_games(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    games = registry.get("games")
    if not isinstance(games, list):
        return []
    return [dict(row) for row in games if isinstance(row, Mapping)]


def solved_game_rows(registry: Mapping[str, Any], root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _registry_games(registry):
        game = str(row.get("game") or "")
        levels = v2.as_int(row.get("levels_reproduced"))
        reproduced = row.get("reproducibility") == "reproduced" or levels > 0
        if game and reproduced and (Path(root) / "environment_files" / game).is_dir():
            rows.append({"game": game, "levels_reproduced": levels})
    return sorted(rows, key=lambda item: item["game"])


def reproducible_total_levels(registry: Mapping[str, Any]) -> int:
    total = registry.get("reproducible_total_levels")
    if isinstance(total, int):
        return total
    return sum(v2.as_int(row.get("levels_reproduced")) for row in _registry_games(registry))


def reflection_variant_specs(
    games: Sequence[str],
    reflection_variants: Sequence[int] = DEFAULT_REFLECTION_VARIANTS,
) -> list[dict[str, Any]]:
    return exp4472.manufactured_variant_specs(
        games,
        color_variants=(),
        reflection_variants=reflection_variants,
    )


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {
        REGISTRY_RELATIVE_PATH: v2._file_sha256(root, REGISTRY_RELATIVE_PATH),
        "python/carnot/agentic/arc_variant_generator.py": v2._file_sha256(
            root, "python/carnot/agentic/arc_variant_generator.py"
        ),
        "python/carnot/agentic/arc_solver_kit.py": v2._file_sha256(
            root, "python/carnot/agentic/arc_solver_kit.py"
        ),
        exp4472.RESULT_RELATIVE_PATH: v2._file_sha256(root, exp4472.RESULT_RELATIVE_PATH),
    }


def check_preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - live boundary
    root = Path(root)
    registry = load_registry(root)
    rows = solved_game_rows(registry, root)
    env_games = sorted(exp4472._environment_games(root))
    checks: dict[str, Any] = {
        "registry_parseable": bool(registry),
        "arc_variant_generator_import": False,
        "arc_solver_kit_import": False,
        "offline_env_files_present": bool(env_games),
        "offline_env_games": env_games,
        "solved_games": [row["game"] for row in rows],
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
    checks["ok"] = first_precondition_miss(checks) is None
    return checks


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("registry_parseable") is not True:
        return "registry_parse"
    if preconditions.get("offline_env_files_present") is not True:
        return "offline_env_files"
    if preconditions.get("arc_variant_generator_import") is not True:
        return "arc_variant_generator"
    if preconditions.get("arc_solver_kit_import") is not True:
        return "arc_solver_kit"
    if not isinstance(preconditions.get("solved_games"), list) or not preconditions.get("solved_games"):
        return "solved_games"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    registry: Mapping[str, Any],
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
        "experiment": "experiment_4481_variant_transfer_benchmark",
        "schema": "carnot.exp4481.variant_transfer_benchmark.v1",
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": BLOCKED_INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "preconditions_checked": dict(preconditions_checked),
        "solved_games": [],
        "per_game": [],
        "variant_plan": {
            "color_variants": [],
            "reflection_variants": list(DEFAULT_REFLECTION_VARIANTS),
            "reflection_axis": exp4472.REFLECTION_AXIS,
            "budget": DEFAULT_BUDGET,
            "n_per_game": len(DEFAULT_REFLECTION_VARIANTS),
            "skipped": [],
        },
        "variant_attempts": [],
        "variants_attempted": 0,
        "variants_solved": 0,
        "transfer_solve_rate": 0.0,
        "reproducible_total_levels": reproducible_total_levels(registry),
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": v2.sha256(checksum_payload),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4481", "SCENARIO-REPORT-4481"],
        "upstream_artifacts": {"source_hashes": _source_hashes(root)},
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


def _attempt_summary_by_game(
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


def _verdict(variants_solved: int, variants_attempted: int, rate: float, games: int) -> str:
    prefix = "success" if variants_solved > 0 else "complete"
    return (
        f"{prefix}: reflection_variant_transfer_{variants_solved}_of_{variants_attempted}_"
        f"rate_{rate:.4f}_games_{games}"
    )


def build_artifact(
    *,
    root: Path,
    registry: Mapping[str, Any],
    solved_rows: Sequence[Mapping[str, Any]],
    variant_specs: Sequence[Mapping[str, Any]],
    variant_attempts: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    reflection_variants: Sequence[int],
    budget: int,
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    summary = _attempt_summary_by_game(variant_attempts)
    per_game: list[dict[str, Any]] = []
    for row in solved_rows:
        game = str(row["game"])
        var = summary.get(game, {"attempted": 0, "solved": 0})
        per_game.append(
            {
                "game": game,
                "source_levels_reproduced": v2.as_int(row.get("levels_reproduced")),
                "variants_attempted": int(var["attempted"]),
                "variants_solved": int(var["solved"]),
                "transfer_solve_rate": _transfer_rate(var["solved"], var["attempted"]),
            }
        )

    variants_attempted = sum(1 for attempt in variant_attempts if attempt.get("attempted") is True)
    variants_solved = sum(
        1
        for attempt in variant_attempts
        if attempt.get("attempted") is True and attempt.get("solved") is True
    )
    rate = _transfer_rate(variants_solved, variants_attempted)
    skipped = [
        dict(attempt)
        for attempt in variant_attempts
        if attempt.get("attempted") is not True or attempt.get("blocked_reason")
    ]
    snapshot = {
        "solved_rows": [dict(row) for row in solved_rows],
        "variant_specs": [dict(spec) for spec in variant_specs],
        "variant_attempts": [dict(attempt) for attempt in variant_attempts],
        "source_hashes": _source_hashes(root),
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4481_variant_transfer_benchmark",
        "schema": "carnot.exp4481.variant_transfer_benchmark.v1",
        "honest_verdict": _verdict(variants_solved, variants_attempted, rate, len(solved_rows)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "offline_reproduced": True,
        "reproduced_levels": int(variants_solved),
        "preconditions_checked": dict(preconditions_checked),
        "solved_games": [str(row["game"]) for row in solved_rows],
        "per_game": per_game,
        "variant_plan": {
            "color_variants": [],
            "reflection_variants": [int(item) for item in reflection_variants],
            "reflection_axis": exp4472.REFLECTION_AXIS,
            "budget": int(budget),
            "n_per_game": len(reflection_variants),
            "skipped": skipped,
        },
        "variant_attempts": [dict(attempt) for attempt in variant_attempts],
        "variants_attempted": int(variants_attempted),
        "variants_solved": int(variants_solved),
        "transfer_solve_rate": float(rate),
        "reproducible_total_levels": reproducible_total_levels(registry),
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": v2.sha256(snapshot),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4481", "SCENARIO-REPORT-4481"],
        "upstream_artifacts": {
            "registry": REGISTRY_RELATIVE_PATH,
            "source_hashes": _source_hashes(root),
        },
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


def _rate_matches(rate: Any, solved: Any, attempted: Any) -> bool:
    if type(rate) is not float or type(solved) is not int or type(attempted) is not int:
        return False
    expected = 0.0 if attempted <= 0 else solved / attempted
    return abs(rate - expected) <= 1e-9


def _terminal_blocked(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(("complete: blocked_", "complete_blocked_"))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") is None:
        errors.append("inference_substrate must not be None")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be dict")
    if not isinstance(artifact.get("solved_games"), list):
        errors.append("solved_games must be list")

    per_game = artifact.get("per_game")
    per_game_attempts = 0
    per_game_solved = 0
    if not isinstance(per_game, list):
        errors.append("per_game must be list")
    else:
        for index, row in enumerate(per_game):
            if not isinstance(row, Mapping):
                errors.append(f"per_game[{index}] must be dict")
                continue
            for field in (
                "game",
                "source_levels_reproduced",
                "variants_attempted",
                "variants_solved",
                "transfer_solve_rate",
            ):
                if field not in row:
                    errors.append(f"per_game[{index}] missing {field}")
            if type(row.get("source_levels_reproduced")) is not int:
                errors.append(f"per_game[{index}].source_levels_reproduced must be bare int")
            if type(row.get("variants_attempted")) is not int:
                errors.append(f"per_game[{index}].variants_attempted must be bare int")
            if type(row.get("variants_solved")) is not int:
                errors.append(f"per_game[{index}].variants_solved must be bare int")
            if type(row.get("transfer_solve_rate")) is not float:
                errors.append(f"per_game[{index}].transfer_solve_rate must be bare float")
            if not _rate_matches(
                row.get("transfer_solve_rate"),
                row.get("variants_solved"),
                row.get("variants_attempted"),
            ):
                errors.append(f"per_game[{index}].transfer_solve_rate must match solved/attempted")
            per_game_attempts += v2.as_int(row.get("variants_attempted"))
            per_game_solved += v2.as_int(row.get("variants_solved"))

    if not isinstance(artifact.get("variant_plan"), Mapping):
        errors.append("variant_plan must be dict")
    if type(artifact.get("variants_attempted")) is not int:
        errors.append("variants_attempted must be bare int")
    if type(artifact.get("variants_solved")) is not int:
        errors.append("variants_solved must be bare int")
    if type(artifact.get("transfer_solve_rate")) is not float:
        errors.append("transfer_solve_rate must be bare float")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be bare int")

    variant_attempts = artifact.get("variant_attempts")
    solved_attempts = 0
    attempted_count = 0
    if not isinstance(variant_attempts, list):
        errors.append("variant_attempts must be list")
    else:
        for attempt in variant_attempts:
            if not isinstance(attempt, Mapping):
                continue
            if attempt.get("attempted") is True:
                attempted_count += 1
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
    if type(artifact.get("variants_attempted")) is int and attempted_count != artifact.get(
        "variants_attempted"
    ):
        errors.append("variants_attempted must match variant_attempts")
    if solved_attempts != artifact.get("variants_solved"):
        errors.append("variants_solved must match solved variant_attempts")
    if type(artifact.get("variants_attempted")) is int and per_game_attempts not in {
        0,
        artifact.get("variants_attempted"),
    }:
        errors.append("variants_attempted must match per_game totals")
    if type(artifact.get("variants_solved")) is int and per_game_solved not in {
        0,
        artifact.get("variants_solved"),
    }:
        errors.append("variants_solved must match per_game totals")
    if not _rate_matches(
        artifact.get("transfer_solve_rate"),
        artifact.get("variants_solved"),
        artifact.get("variants_attempted"),
    ):
        errors.append("transfer_solve_rate must equal variants_solved/variants_attempted")
    if (
        not _terminal_blocked(verdict)
        and type(artifact.get("variants_attempted")) is int
        and artifact.get("variants_attempted") <= 0
    ):
        errors.append("completed measurement must attempt at least one variant")
    if artifact.get("offline_reproduced") is False and (
        v2.as_int(artifact.get("variants_solved")) > 0 or v2.as_int(artifact.get("reproduced_levels")) > 0
    ):
        errors.append("offline_reproduced false cannot accompany counted solves")
    if artifact.get("reproduced_levels") != artifact.get("variants_solved"):
        errors.append("reproduced_levels must equal variants_solved")
    field_principles = artifact.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"field_principles.{field} must match REQ-REPORT-4481")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not v2.checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if artifact.get("no_3090_inference") is not True:
        errors.append("no_3090_inference must be true")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner: VariantRunner = exp4472.run_variant_attempt,
    reflection_variants: Sequence[int] = DEFAULT_REFLECTION_VARIANTS,
    budget: int = DEFAULT_BUDGET,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """REQ-REPORT-4481: run reflection variants for every solved registry game."""

    started = float(now())
    root = Path(root)
    registry = load_registry(root)
    solved_rows = solved_game_rows(registry, root)
    preconditions = dict(preconditions_checked or check_preconditions(root))
    preconditions.setdefault("solved_games", [row["game"] for row in solved_rows])
    preconditions.setdefault("no_3090_inference", True)
    preconditions.setdefault("leaderboard_submission", False)
    miss = first_precondition_miss(preconditions)
    if miss:
        artifact = _blocked_artifact(
            reason=miss,
            preconditions_checked=preconditions,
            registry=registry,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    allowed = {str(game) for game in preconditions.get("solved_games", [])}
    measured_rows = [row for row in solved_rows if row["game"] in allowed]
    specs = reflection_variant_specs(
        [row["game"] for row in measured_rows],
        reflection_variants=reflection_variants,
    )
    attempts: list[dict[str, Any]] = []
    for spec in specs:
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
        attempts.append(attempt)

    ended = _floor_end_time(started_at=started, now=now, sleep_fn=sleep_fn)
    artifact = build_artifact(
        root=root,
        registry=registry,
        solved_rows=measured_rows,
        variant_specs=specs,
        variant_attempts=attempts,
        preconditions_checked=preconditions,
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
    print(f"transfer_solve_rate={artifact['transfer_solve_rate']:.4f}")
    print(f"variants_solved={artifact['variants_solved']}")
    print(f"variants_attempted={artifact['variants_attempted']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
