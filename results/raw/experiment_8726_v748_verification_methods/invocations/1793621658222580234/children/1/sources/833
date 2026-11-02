"""Exp 4423: generic first-contact breadth over unseen ARC games.

Spec refs: REQ-REPORT-4423, SCENARIO-REPORT-4423.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4423_generic_first_contact_breadth.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SCORECARD_RELATIVE_PATH = "results/arc3_full_pass_scorecard.json"
RANDOM_SEED = 4423
SPEC_REFS = ("REQ-REPORT-4423", "SCENARIO-REPORT-4423")
UPDATED_DATE = "2026-06-19"

REQUIRED_ARTIFACT_FIELDS = (
    "offline_reproduced",
    "reproduced_levels",
    "missing_verifier_gaps",
    "verifier_is_oracle",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "offline_reproduced": (
        "Bare bool: true only when the standing loop result reproduces a level "
        "through the offline ARC environment."
    ),
    "reproduced_levels": "Bare int: reproduced offline levels reached for the attempted target.",
    "missing_verifier_gaps": (
        "List: the precise unselectable-failure backlog when the standing loop "
        "cannot advance a candidate."
    ),
    "verifier_is_oracle": (
        "Bare bool: false for generic non-oracle routing; true only if the loop "
        "explicitly reports an execution-grounded oracle verifier."
    ),
    "honest_verdict": "Terminal-prefixed status: success, complete, or blocked.",
}

CONFIG_RULE_GAMES = ("bp35", "dc22", "g50t", "lf52", "s5i5", "cd82", "wa30")
GLYPH_REWRITE_GAMES = ("tr87",)

RoutingFn = Callable[[str], Mapping[str, Any]]
StandingLoopFn = Callable[[str, Path], Mapping[str, Any]]


@dataclass(frozen=True)
class CandidateGame:
    """One game eligible for first-contact routing."""

    game: str
    reason: str
    signals: tuple[str, ...]
    prior_class: str
    prior_duration_s: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_registry(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {"games": []}
    return loaded if isinstance(loaded, dict) else {"games": []}


def _registered_game_names(registry: Mapping[str, Any]) -> set[str]:
    games = registry.get("games")
    if not isinstance(games, list):
        return set()
    return {
        str(entry.get("game"))
        for entry in games
        if isinstance(entry, Mapping) and entry.get("game")
    }


def _environment_games(root: Path) -> set[str]:
    env_dir = root / "environment_files"
    if not env_dir.exists():
        return set()
    return {path.name for path in env_dir.iterdir() if path.is_dir()}


def select_candidate_games(root: Path = REPO_ROOT) -> list[CandidateGame]:
    """REQ-REPORT-4423: choose unseen or FAIL_EXPLORATION ARC candidates."""

    registry = load_registry(root)
    registered = _registered_game_names(registry)
    env_games = _environment_games(root)
    scorecard = _load_json(root / SCORECARD_RELATIVE_PATH)
    rows = scorecard.get("per_game")
    per_game = rows if isinstance(rows, list) else []
    candidates: dict[str, CandidateGame] = {}

    for row in per_game:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game") or "")
        if not game or game in registered or (env_games and game not in env_games):
            continue
        prior_class = str(row.get("class") or "UNSEEN")
        signals = ["UNSEEN"]
        if prior_class == "FAIL_EXPLORATION":
            signals.append("FAIL_EXPLORATION")
        reason = "unseen_not_in_registry"
        candidates[game] = CandidateGame(
            game=game,
            reason=reason,
            signals=tuple(signals),
            prior_class=prior_class,
            prior_duration_s=float(row.get("dur_s") or 0.0),
        )

    for game in sorted(env_games - registered - set(candidates)):
        candidates[game] = CandidateGame(
            game=game,
            reason="unseen_not_in_registry",
            signals=("UNSEEN",),
            prior_class="UNSEEN",
            prior_duration_s=0.0,
        )

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            0 if candidate.prior_class == "SOLVED" else 1,
            candidate.prior_duration_s,
            candidate.game,
        ),
    )


def routing_options_for(game: str, recommendation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-REPORT-4423: expose closest recipe plus Exp 4421/4422 verifier routes."""

    recommended = recommendation.get("recommended")
    closest = recommended[0] if isinstance(recommended, list) and recommended else {}
    if not isinstance(closest, Mapping):
        closest = {}
    return [
        {
            "id": "closest_solved_recipe",
            "kind": "learned_recipe_route",
            "selected": True,
            "source": "arc_solve_learning.recommend_approach",
            "game": closest.get("game", ""),
            "solver": closest.get("solver", ""),
            "win_condition": closest.get("win_condition", ""),
            "action_model": closest.get("action_model", ""),
            "verifier_is_oracle": False,
        },
        {
            "id": "exp4421_config_rule_unseen",
            "kind": "config_rule_verifier",
            "source_artifact": "results/experiment_4421_config_rule_solve_unseen.json",
            "candidate_games": list(CONFIG_RULE_GAMES),
            "matches_target": game in CONFIG_RULE_GAMES,
            "verifier_is_oracle": True,
        },
        {
            "id": "exp4422_glyph_rewrite_pixels",
            "kind": "glyph_rewrite_pixel_verifier",
            "source_artifact": "results/experiment_4422_glyph_rewrite_perception.json",
            "candidate_games": list(GLYPH_REWRITE_GAMES),
            "matches_target": game in GLYPH_REWRITE_GAMES,
            "verifier_is_oracle": True,
        },
    ]


def _closest_recipe(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    recommended = recommendation.get("recommended")
    if isinstance(recommended, list) and recommended and isinstance(recommended[0], Mapping):
        return dict(recommended[0])
    return {}


def missing_gap_for(
    game: str,
    recommendation: Mapping[str, Any],
    loop_result: Mapping[str, Any],
    routing_options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """SCENARIO-REPORT-4423: describe the exact verifier gap for a no-advance."""

    recipe = _closest_recipe(recommendation)
    config_applicable = any(
        option.get("id") == "exp4421_config_rule_unseen" and option.get("matches_target") is True
        for option in routing_options
    )
    glyph_applicable = any(
        option.get("id") == "exp4422_glyph_rewrite_pixels" and option.get("matches_target") is True
        for option in routing_options
    )
    candidate_design = "derive a selectable verifier from the routed solved-game delta"
    if config_applicable:
        candidate_design = "adapt Exp 4421 config-rule predicate grounding to this game's visible toggles"
    elif glyph_applicable:
        candidate_design = "adapt Exp 4422 from-pixels glyph rewrite grounding to this game"
    return {
        "gap_id": f"GAP-4423-{game.upper()}-UNSELECTABLE-FIRST-CONTACT",
        "status": "open",
        "game": game,
        "failure_mode": (
            str(loop_result.get("status") or loop_result.get("mode") or "standing_loop_no_advance")
        ),
        "missing_discriminator": (
            "selectable verifier that distinguishes the target's winning delta "
            "from the explored non-winning states"
        ),
        "candidate_design": candidate_design,
        "routed_recipe": {
            "game": recipe.get("game", ""),
            "solver": recipe.get("solver", ""),
            "win_condition": recipe.get("win_condition", ""),
            "action_model": recipe.get("action_model", ""),
        },
        "loop_result_summary": {
            "offline_reproduced": bool(loop_result.get("offline_reproduced")),
            "reproduced_levels": int(loop_result.get("reproduced_levels") or 0),
            "mode": loop_result.get("mode", ""),
        },
    }


def record_dead_end(
    root: Path,
    game: str,
    dead_end: Mapping[str, Any],
) -> None:
    """REQ-REPORT-4423: write a dead-end into the ARC registry for skip-ahead reuse."""

    registry = load_registry(root)
    games = registry.setdefault("games", [])
    if not isinstance(games, list):
        games = []
        registry["games"] = games
    entry: dict[str, Any] | None = None
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            entry = row
            break
    if entry is None:
        entry = {
            "game": game,
            "reproducibility": "unsolved",
            "levels_reproduced": 0,
            "solver": f"scripts/arc_loop_solve.py --game {game}",
            "gotchas": [],
            "dead_ends": [],
        }
        games.append(entry)
    dead_ends = entry.setdefault("dead_ends", [])
    if not isinstance(dead_ends, list):
        dead_ends = []
        entry["dead_ends"] = dead_ends
    gap_id = dead_end.get("gap_id")
    if not any(isinstance(row, Mapping) and row.get("gap_id") == gap_id for row in dead_ends):
        dead_ends.append(dict(dead_end))
    registry["updated"] = UPDATED_DATE
    path = root / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")


def check_preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - live boundary
    checks = {
        "offline_env_files_present": (root / "environment_files").is_dir(),
        "arc_solver_kit_import": False,
        "arc_solve_learning_import": False,
        "arc_loop_solve_script": (root / "scripts" / "arc_loop_solve.py").exists(),
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solve_learning, arc_solver_kit

        checks["arc_solver_kit_import"] = callable(getattr(arc_solver_kit, "offline_arcade", None))
        checks["arc_solve_learning_import"] = callable(
            getattr(arc_solve_learning, "recommend_approach", None)
        )
    except Exception as exc:
        checks["import_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = all(
        checks[key]
        for key in (
            "offline_env_files_present",
            "arc_solver_kit_import",
            "arc_solve_learning_import",
            "arc_loop_solve_script",
        )
    )
    return checks


def run_standing_loop_subprocess(game: str, root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    python = root / ".venv" / "bin" / "python"
    executable = str(python if python.exists() else Path(sys.executable))
    command = [executable, str(root / "scripts" / "arc_loop_solve.py"), "--game", game]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    result_path = root / "results" / f"arc_loop_solve_{game}.json"
    payload = _load_json(result_path)
    if payload:
        payload["command"] = " ".join(command)
        payload["returncode"] = completed.returncode
        payload["stdout"] = completed.stdout[-4000:]
        payload["stderr"] = completed.stderr[-4000:]
        return payload
    return {
        "game": game,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "status": "standing_loop_command_failed",
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _default_recommend(game: str) -> Mapping[str, Any]:  # pragma: no cover - import boundary
    from carnot.agentic import arc_solve_learning

    return arc_solve_learning.recommend_approach(game)


def _terminal_prefixed(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(("success:", "complete:", "blocked:"))


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _routing_options_complete(options: Any) -> bool:
    if not isinstance(options, list):
        return False
    ids = {option.get("id") for option in options if isinstance(option, Mapping)}
    return {"exp4421_config_rule_unseen", "exp4422_glyph_rewrite_pixels"}.issubset(ids)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """SCENARIO-REPORT-4423: validate the terminal breadth artifact."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be list")
    if type(artifact.get("verifier_is_oracle")) is not bool:
        errors.append("verifier_is_oracle must be bare bool")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if not _routing_options_complete(artifact.get("routing_options")):
        errors.append("routing_options must include exp4421 and exp4422")
    if type(artifact.get("target_was_new_to_registry")) is not bool:
        errors.append("target_was_new_to_registry must be bare bool")
    if not isinstance(artifact.get("attempts"), list):
        errors.append("attempts must be list")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be dict")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be dict")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles missing exact {field}")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success verdict requires offline_reproduced true")
        if int(artifact.get("reproduced_levels") or 0) < 1:
            errors.append("success verdict requires reproduced_levels>=1")
        if artifact.get("target_was_new_to_registry") is not True:
            errors.append("success verdict requires target_was_new_to_registry true")
    if isinstance(verdict, str) and verdict.startswith("complete:"):
        if not artifact.get("missing_verifier_gaps"):
            errors.append("complete no-new-level verdict requires missing_verifier_gaps")
        if not artifact.get("dead_ends_recorded"):
            errors.append("complete no-new-level verdict requires registry dead-end record")
    return errors


def _build_artifact(
    *,
    root: Path,
    candidates: Sequence[CandidateGame],
    target: CandidateGame | None,
    recommendation: Mapping[str, Any],
    routing_options: Sequence[Mapping[str, Any]],
    loop_result: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    registry_before: Mapping[str, Any],
    missing_gaps: Sequence[Mapping[str, Any]],
    dead_ends_recorded: Sequence[str],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    registered_before = _registered_game_names(registry_before)
    game = target.game if target is not None else ""
    offline_reproduced = bool(loop_result.get("offline_reproduced"))
    reproduced_levels = int(loop_result.get("reproduced_levels") or loop_result.get("reached_level") or 0)
    if not offline_reproduced:
        reproduced_levels = 0
    target_was_new = bool(game and game not in registered_before)
    verifier_is_oracle = bool(loop_result.get("verifier_is_oracle"))
    if offline_reproduced and reproduced_levels >= 1 and target_was_new:
        verdict = f"success: generic_first_contact_{game}_L{reproduced_levels}_offline_reproduced"
    elif game:
        verdict = f"complete: generic_first_contact_{game}_routed_no_new_level_gap_logged"
    else:
        verdict = "blocked: generic_first_contact_no_candidate"
    attempts = []
    if target is not None:
        attempts.append(
            {
                "game": game,
                "candidate": target.to_json(),
                "recommendation": dict(recommendation),
                "routing_options": [dict(option) for option in routing_options],
                "standing_loop_result": dict(loop_result),
                "offline_reproduced": offline_reproduced,
                "reproduced_levels": reproduced_levels,
                "missing_verifier_gaps": [dict(gap) for gap in missing_gaps],
            }
        )
    checksum_payload = {
        "candidates": [candidate.to_json() for candidate in candidates],
        "attempts": attempts,
        "missing_gaps": [dict(gap) for gap in missing_gaps],
        "dead_ends_recorded": list(dead_ends_recorded),
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4423_generic_first_contact_breadth",
        "schema": "carnot.exp4423.generic_first_contact_breadth.v1",
        "target_game": game,
        "candidate_games": [candidate.to_json() for candidate in candidates],
        "attempted_games": [game] if game else [],
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "missing_verifier_gaps": [dict(gap) for gap in missing_gaps],
        "verifier_is_oracle": verifier_is_oracle,
        "honest_verdict": verdict,
        "target_was_new_to_registry": target_was_new,
        "routing_options": [dict(option) for option in routing_options],
        "recommendation": dict(recommendation),
        "standing_loop_result": dict(loop_result),
        "dead_ends_recorded": list(dead_ends_recorded),
        "attempts": attempts,
        "command": f".venv/bin/python scripts/arc_loop_solve.py --game {game}" if game else "",
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "submitted_to_leaderboard": False,
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "reproducibility_checksum": _sha256(checksum_payload),
    }


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
    target_game: str | None = None,
    recommend_fn: RoutingFn = _default_recommend,
    standing_loop_fn: StandingLoopFn = run_standing_loop_subprocess,
    write_registry: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-4423: route one candidate, run the standing loop, write the artifact."""

    started = now()
    preconditions = dict(preconditions_checked or check_preconditions(root))
    registry_before = load_registry(root)
    all_candidates = select_candidate_games(root)
    if target_game:
        candidate = next(
            (item for item in all_candidates if item.game == target_game),
            CandidateGame(
                game=target_game,
                reason="manual_target_not_in_registry",
                signals=("UNSEEN",),
                prior_class="UNSEEN",
                prior_duration_s=0.0,
            ),
        )
    else:
        candidate = all_candidates[0] if all_candidates else None

    recommendation: Mapping[str, Any] = {}
    routing_options: list[dict[str, Any]] = []
    loop_result: Mapping[str, Any] = {
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "status": "no_candidate",
    }
    missing_gaps: list[dict[str, Any]] = []
    dead_ends_recorded: list[str] = []

    if candidate is not None:
        recommendation = recommend_fn(candidate.game)
        routing_options = routing_options_for(candidate.game, recommendation)
        try:
            loop_result = standing_loop_fn(candidate.game, root)
        except Exception as exc:
            loop_result = {
                "game": candidate.game,
                "offline_reproduced": False,
                "reproduced_levels": 0,
                "status": f"standing_loop_exception_{type(exc).__name__}",
                "error": str(exc),
            }
        offline_reproduced = bool(loop_result.get("offline_reproduced"))
        reproduced_levels = int(loop_result.get("reproduced_levels") or loop_result.get("reached_level") or 0)
        target_new = candidate.game not in _registered_game_names(registry_before)
        if not (offline_reproduced and reproduced_levels >= 1 and target_new):
            gap = missing_gap_for(candidate.game, recommendation, loop_result, routing_options)
            missing_gaps.append(gap)
            if write_registry:
                record_dead_end(root, candidate.game, gap)
                dead_ends_recorded.append(candidate.game)

    artifact = _build_artifact(
        root=root,
        candidates=all_candidates,
        target=candidate,
        recommendation=recommendation,
        routing_options=routing_options,
        loop_result=loop_result,
        preconditions_checked=preconditions,
        registry_before=registry_before,
        missing_gaps=missing_gaps,
        dead_ends_recorded=dead_ends_recorded,
        started_at=started,
        ended_at=now(),
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    args = parser.parse_args()
    artifact = run(REPO_ROOT, target_game=args.game)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
