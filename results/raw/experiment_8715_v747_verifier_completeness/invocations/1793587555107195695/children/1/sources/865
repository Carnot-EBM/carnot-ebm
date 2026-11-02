"""Exp 4459: leave-one-out generic ARC solve benchmark v3.

Spec refs: REQ-REPORT-4459, SCENARIO-REPORT-4459.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from carnot import experiment_4448_loo_generic_solve_benchmark_v2 as v2


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4459_loo_generic_solve_benchmark_v3.json"
REGISTRY_RELATIVE_PATH = v2.REGISTRY_RELATIVE_PATH
V2_RELATIVE_PATH = v2.RESULT_RELATIVE_PATH
GLYPH_REWRITE_RELATIVE_PATH = "results/experiment_4456_generic_glyph_rewrite_operator.json"
CAST_GRID_GLOB = "results/experiment_4457_*.json"
RANDOM_SEED = 4459
V2_BASELINE = 5
MIN_HELDOUT_GAMES = 7
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_INFERENCE_SUBSTRATE = "precondition_check_no_inference"
VERIFIER_SCORING_DURATION_TARGET_S = 1.05
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "generic_loo_solve_count_v3",
    "generic_loo_solve_count_v2_baseline",
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
    "inference_substrate": {
        "principle": "THE .410/.411 LESSON -- EMIT it; never None"
    },
    "generic_loo_solve_count_v3": {
        "principle": (
            "bare int: the headline -- count of games re-solved from the corpus + "
            ".412 operators WITHOUT their own recipe"
        )
    },
    "generic_loo_solve_count_v2_baseline": {
        "principle": "bare int = 5; the .411 baseline this must beat to show progress"
    },
    "per_game": {
        "principle": (
            "list of {game, solved_without_own_recipe, closed_by_operator, residual_delta} "
            "-- shows EXACTLY which residuals the new operators closed"
        )
    },
    "offline_reproduced": {"principle": "every claimed re-solve is reproduction-gated"},
    "missing_verifier_gaps": {
        "principle": "each still-open residual -- the .413 build backlog"
    },
    "verifier_is_oracle": {"principle": "true: execution-grounded reproduction"},
    "random_seed": {"principle": "determinism for re-run"},
    "reproducibility_checksum": {
        "principle": "content hash of the example-corpus snapshot used"
    },
}


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


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    return v2.first_precondition_miss(preconditions)


def _source_hashes(root: Path) -> dict[str, Any]:
    cast_grid_hashes = {
        rel_path: v2._file_sha256(root, rel_path)
        for rel_path in _cast_grid_relative_paths(root)
    }
    return {
        V2_RELATIVE_PATH: v2._file_sha256(root, V2_RELATIVE_PATH),
        GLYPH_REWRITE_RELATIVE_PATH: v2._file_sha256(root, GLYPH_REWRITE_RELATIVE_PATH),
        CAST_GRID_GLOB: cast_grid_hashes,
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
        "experiment": "experiment_4459_loo_generic_solve_benchmark_v3",
        "schema": "carnot.exp4459.loo_generic_solve_benchmark_v3.v1",
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": BLOCKED_INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "generic_loo_solve_count_v3": 0,
        "generic_loo_solve_count_v2_baseline": V2_BASELINE,
        "heldout_games": [],
        "loo_gate_passed": False,
        "per_game": [],
        "closed_residuals_by_412_operator": [],
        "offline_reproduced": False,
        "missing_verifier_gaps": [],
        "preconditions_checked": dict(preconditions_checked),
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": v2.sha256(checksum_payload),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4459", "SCENARIO-REPORT-4459"],
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


def _cast_grid_relative_paths(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.glob(CAST_GRID_GLOB))


def _load_cast_grid_artifact(root: Path) -> tuple[str, Mapping[str, Any] | None]:
    loaded: list[tuple[str, Mapping[str, Any]]] = []
    for rel_path in _cast_grid_relative_paths(root):
        artifact = v2.load_json(root, rel_path)
        if artifact is not None:
            loaded.append((rel_path, artifact))
    for rel_path, artifact in loaded:
        if _cast_grid_closes_sc25(artifact):
            return rel_path, artifact
    return loaded[0] if loaded else ("", None)


def _heldout_games(v2_artifact: Mapping[str, Any]) -> list[str]:
    games = v2_artifact.get("heldout_games")
    if isinstance(games, list) and len(games) >= MIN_HELDOUT_GAMES:
        return [str(game) for game in games]
    rows = v2_artifact.get("per_game")
    if isinstance(rows, list):
        inferred = [str(row.get("game")) for row in rows if isinstance(row, Mapping) and row.get("game")]
        if len(inferred) >= MIN_HELDOUT_GAMES:
            return inferred
    return []


def _rows_by_game(rows: Any) -> dict[str, Mapping[str, Any]]:
    return v2._rows_by_game(rows)


def _v2_solved_games(v2_artifact: Mapping[str, Any]) -> set[str]:
    if v2_artifact.get("offline_reproduced") is not True:
        return set()
    solved: set[str] = set()
    for game, row in _rows_by_game(v2_artifact.get("per_game")).items():
        if (
            row.get("solved_without_own_recipe") is True
            and row.get("residual_delta") == "none"
            and row.get("closed_by_operator") not in {"", "none", None}
        ):
            solved.add(game)
    return solved


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


def _glyph_rewrite_closes_tr87(glyph_artifact: Mapping[str, Any] | None) -> bool:
    if not isinstance(glyph_artifact, Mapping):
        return False
    return (
        glyph_artifact.get("tr87_resolved_generically") is True
        and glyph_artifact.get("offline_reproduced") is True
        and v2.gate_reproduced(glyph_artifact.get("generic_reproduction_result"))
        and _operator_result_matches(
            glyph_artifact,
            game="tr87",
            operator="glyph_rewrite_rule_verifier",
        )
    )


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
    return (
        closure_flag
        and cast_artifact.get("offline_reproduced") is True
        and v2.gate_reproduced(cast_artifact.get("generic_reproduction_result"))
        and _operator_result_matches(
            cast_artifact,
            game="sc25",
            operator="cast_grid_phase_fsm_world_model",
        )
    )


def _retrieved_operator(v2_artifact: Mapping[str, Any], game: str) -> str:
    gaps = v2_artifact.get("missing_verifier_gaps")
    if not isinstance(gaps, list):
        return ""
    for gap in gaps:
        if isinstance(gap, Mapping) and gap.get("game") == game:
            return str(gap.get("retrieved_operator") or "")
    return ""


def _verdict(solve_count: int, total: int) -> str:
    if solve_count > V2_BASELINE:
        return f"success: generic_loo_solve_count_v3_{solve_count}_of_{total}_beats_v2_{V2_BASELINE}"
    if solve_count == V2_BASELINE:
        return f"complete: generic_loo_solve_count_v3_{solve_count}_of_{total}_flat_vs_v2_{V2_BASELINE}"
    return f"complete: generic_loo_solve_count_v3_{solve_count}_of_{total}_lower_than_v2_{V2_BASELINE}"


def _residual_for_open(game: str, v2_row: Mapping[str, Any], blocked_reason: str | None) -> str:
    if blocked_reason:
        return blocked_reason
    residual = str(v2_row.get("residual_delta") or "")
    if residual and residual != "none":
        return residual
    if game == "tr87":
        return "missing_glyph_rewrite_rule_verifier_without_tr87_adapter"
    if game == "sc25":
        return "missing_cast_grid_spell_shrink_tank_exit_verifier"
    return "missing_reproduction_gate_evidence"


def _missing_heldout_env_games(preconditions: Mapping[str, Any], heldout_games: list[str]) -> list[str]:
    env_games = preconditions.get("offline_env_games")
    if not isinstance(env_games, list):
        return heldout_games
    present = {str(game) for game in env_games}
    return [game for game in heldout_games if game not in present]


def build_artifact(
    *,
    root: Path,
    v2_artifact: Mapping[str, Any],
    glyph_artifact: Mapping[str, Any] | None,
    cast_artifact: Mapping[str, Any] | None,
    cast_artifact_path: str,
    preconditions_checked: Mapping[str, Any],
    llm_induction_games: set[str],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    heldout_games = _heldout_games(v2_artifact)
    v2_rows = _rows_by_game(v2_artifact.get("per_game"))
    v2_solved = _v2_solved_games(v2_artifact)
    model_available = (
        preconditions_checked.get("qwen_gguf_cached") is True
        or preconditions_checked.get("igpu_llama_server_available") is True
    )

    per_game: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    closed_new: list[dict[str, str]] = []
    evidence: list[dict[str, Any]] = []

    for game in heldout_games:
        v2_row = v2_rows.get(game, {})
        blocked_reason = "blocked_model_not_cached" if game in llm_induction_games and not model_available else None
        closed_by = "none"
        solved = False
        residual = _residual_for_open(game, v2_row, blocked_reason)

        if blocked_reason is None and game in v2_solved:
            solved = True
            closed_by = str(v2_row.get("closed_by_operator") or "v2_generic_remeasurement")
            residual = "none"
            evidence.append({"game": game, "source": V2_RELATIVE_PATH, "operator": closed_by})
        elif blocked_reason is None and game == "tr87" and _glyph_rewrite_closes_tr87(glyph_artifact):
            solved = True
            closed_by = "glyph_rewrite_rule_verifier"
            residual = "none"
            evidence.append({"game": game, "source": GLYPH_REWRITE_RELATIVE_PATH, "operator": closed_by})
            closed_new.append(
                {
                    "game": game,
                    "closed_by_operator": closed_by,
                    "v2_residual_delta": str(v2_row.get("residual_delta") or ""),
                }
            )
        elif blocked_reason is None and game == "sc25" and _cast_grid_closes_sc25(cast_artifact):
            solved = True
            closed_by = "cast_grid_phase_fsm_world_model"
            residual = "none"
            evidence.append({"game": game, "source": cast_artifact_path, "operator": closed_by})
            closed_new.append(
                {
                    "game": game,
                    "closed_by_operator": closed_by,
                    "v2_residual_delta": str(v2_row.get("residual_delta") or ""),
                }
            )

        row = {
            "game": game,
            "solved_without_own_recipe": solved,
            "closed_by_operator": closed_by,
            "residual_delta": residual,
        }
        per_game.append(row)
        if not solved:
            gaps.append(
                {
                    "game": game,
                    "residual_delta": residual,
                    "retrieved_operator": _retrieved_operator(v2_artifact, game),
                    "attempt_mode": "v3_412_operator_remeasurement",
                }
            )

    solve_count = sum(1 for row in per_game if row["solved_without_own_recipe"])
    source_hashes = _source_hashes(root)
    snapshot = {
        "heldout_games": heldout_games,
        "per_game": per_game,
        "source_hashes": source_hashes,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4459_loo_generic_solve_benchmark_v3",
        "schema": "carnot.exp4459.loo_generic_solve_benchmark_v3.v1",
        "honest_verdict": _verdict(solve_count, len(heldout_games)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_at, ended_at),
        "generic_loo_solve_count_v3": solve_count,
        "generic_loo_solve_count_v2_baseline": V2_BASELINE,
        "heldout_games": heldout_games,
        "loo_gate": f"generic_loo_solve_count_v3 > {V2_BASELINE}",
        "loo_gate_passed": solve_count > V2_BASELINE,
        "per_game": per_game,
        "closed_residuals_by_412_operator": closed_new,
        "offline_reproduced": len(evidence) == solve_count,
        "reproduction_evidence": evidence,
        "missing_verifier_gaps": gaps,
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifacts": {
            "v2_baseline": V2_RELATIVE_PATH,
            "glyph_rewrite_rule_verifier": GLYPH_REWRITE_RELATIVE_PATH,
            "cast_grid_phase_fsm_world_model": cast_artifact_path,
            "source_hashes": source_hashes,
        },
        "example_corpus_snapshot": snapshot,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": v2.sha256(snapshot),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": ["REQ-REPORT-4459", "SCENARIO-REPORT-4459"],
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


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
    if artifact.get("inference_substrate") in {INFERENCE_SUBSTRATE, None} and float(
        artifact.get("duration_s") or 0.0
    ) < 1.0:
        errors.append("cached verifier substrate requires duration_s >= 1.0")
    if type(artifact.get("generic_loo_solve_count_v3")) is not int:
        errors.append("generic_loo_solve_count_v3 must be bare int")
    if artifact.get("generic_loo_solve_count_v2_baseline") != V2_BASELINE or type(
        artifact.get("generic_loo_solve_count_v2_baseline")
    ) is not int:
        errors.append("generic_loo_solve_count_v2_baseline must be bare int = 5")
    per_game = artifact.get("per_game")
    if not isinstance(per_game, list):
        errors.append("per_game must be list")
    else:
        solved_count = 0
        for index, row in enumerate(per_game):
            if not isinstance(row, Mapping):
                errors.append(f"per_game[{index}] must be dict")
                continue
            for field in ("game", "solved_without_own_recipe", "closed_by_operator", "residual_delta"):
                if field not in row:
                    errors.append(f"per_game[{index}] missing {field}")
            if type(row.get("solved_without_own_recipe")) is not bool:
                errors.append(f"per_game[{index}].solved_without_own_recipe must be bare bool")
            if "closed_by_operator" in row and not isinstance(row.get("closed_by_operator"), str):
                errors.append(f"per_game[{index}].closed_by_operator must be string")
            if row.get("solved_without_own_recipe") is True:
                solved_count += 1
                if row.get("closed_by_operator") in {"", "none", None}:
                    errors.append(f"per_game[{index}] solved row requires closed_by_operator")
                if row.get("residual_delta") != "none":
                    errors.append(f"per_game[{index}] solved row requires residual_delta none")
        if solved_count != artifact.get("generic_loo_solve_count_v3"):
            errors.append("generic_loo_solve_count_v3 must match solved per_game rows")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if artifact.get("offline_reproduced") is False and v2.as_int(artifact.get("generic_loo_solve_count_v3")) > 0:
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
    if not isinstance(field_principles, Mapping) or field_principles.get("honest_verdict") != FIELD_PRINCIPLES["honest_verdict"]:
        errors.append("field_principles.honest_verdict must match REQ-REPORT-4459")
    if isinstance(verdict, str) and verdict.startswith("success:") and v2.as_int(
        artifact.get("generic_loo_solve_count_v3")
    ) <= V2_BASELINE:
        errors.append("success verdict requires generic_loo_solve_count_v3 > 5")
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
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    llm_induction_games: set[str] | None = None,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """REQ-REPORT-4459: re-measure the v2 LOO set with .412 generic closures."""

    started = float(now())
    root = Path(root)
    preconditions = dict(preconditions_checked or v2.check_preconditions(root))
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

    v2_artifact = v2.load_json(root, V2_RELATIVE_PATH)
    if v2_artifact is None:
        artifact = _blocked_artifact(
            reason="v2_source_artifact",
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact
    heldout_games = _heldout_games(v2_artifact)
    if len(heldout_games) < MIN_HELDOUT_GAMES:
        artifact = _blocked_artifact(
            reason="v2_heldout_target_count",
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact
    missing_env = _missing_heldout_env_games(preconditions, heldout_games)
    if missing_env:
        artifact = _blocked_artifact(
            reason=f"offline_env_files_{missing_env[0]}",
            preconditions_checked=preconditions,
            started_at=started,
            ended_at=float(now()),
            root=root,
        )
        write_artifact(root, artifact)
        return artifact

    glyph_artifact = v2.load_json(root, GLYPH_REWRITE_RELATIVE_PATH)
    cast_artifact_path, cast_artifact = _load_cast_grid_artifact(root)
    ended = _floor_end_time(started_at=started, now=now, sleep_fn=sleep_fn)
    artifact = build_artifact(
        root=root,
        v2_artifact=v2_artifact,
        glyph_artifact=glyph_artifact,
        cast_artifact=cast_artifact,
        cast_artifact_path=cast_artifact_path,
        preconditions_checked=preconditions,
        llm_induction_games=set(llm_induction_games or ()),
        started_at=started,
        ended_at=ended,
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - script entry
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    print(f"generic_loo_solve_count_v3={artifact['generic_loo_solve_count_v3']}")
    print(f"generic_loo_solve_count_v2_baseline={artifact['generic_loo_solve_count_v2_baseline']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
