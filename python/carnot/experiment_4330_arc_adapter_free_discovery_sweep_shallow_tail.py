"""Exp 4330: adapter-free ARC shallow-tail discovery sweep.

Spec refs: REQ-PHASE4-077, SCENARIO-PHASE4-077.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail.json"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH
SHALLOW_TAIL_GAMES = ("bp35", "dc22", "g50t", "lf52", "re86", "s5i5", "sb26", "vc33")
EXCLUDED_GAMES = ("ar25", "ka59", "tr87", "ft09", "sc25")
SWEEP_GAMES = SHALLOW_TAIL_GAMES + ("tn36",)
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 13
DEFAULT_MAX_DISCOVERY_EXPANSIONS = 12000
DEFAULT_MAX_DEPTH = 60
RANDOM_SEED = 4330
REQUIREMENTS = ("REQ-PHASE4-077", "SCENARIO-PHASE4-077")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproducible_total_levels",
    "games_advanced",
    "per_game_exploration_actions",
    "tn36_schema_finding",
    "offline_reproduced",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. New reproduced L1 solves, an honest no-advance WITH real exploration "
        "per game (records the dead-ends, informs the next pick), and a tn36 schema-RE result are "
        "ALL COMPLETE -- progress is the metric but a real no-advance is decision-grade."
    ),
    "reproducible_total_levels": (
        "BARE int: the cumulative OFFLINE-REPRODUCED solved-level count (>= the current 13) -- "
        "the north-star accuracy metric (only reproduced levels count)."
    ),
    "games_advanced": (
        "List of games advanced to a new reproduced level this sweep -- the incremental-progress units."
    ),
    "per_game_exploration_actions": (
        "Per swept game: exploration_actions_used (MUST be >0 -- a 0-action game is the "
        "GATE_PASSED_WITHOUT_DATA flag to avoid) + advanced/no-advance + the dead-end class."
    ),
    "tn36_schema_finding": (
        "The reverse-engineered ACTION6 click-payload schema for tn36 (its per-game delta) -- "
        "whether the wrapped-payload explorer could then advance it; logged to the registry."
    ),
    "offline_reproduced": (
        "BARE bool: at least one swept game's advance reproduces offline via arc_solver_kit.reproduce() "
        "(false if no advance -- an honest no-advance sweep)."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- adapter-free graph-explore solves are execution-grounded (the env defines "
        "the win); ARC progress, NOT a moat headline."
    ),
    "preconditions_checked": (
        "Records the offline-env reachability + sweep-driver import; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the sweep.",
    "reproducibility_checksum": (
        "Hash of the sweep trajectories + the reproduce() results; lets a third party re-run."
    ),
}


@dataclass(frozen=True)
class GameSweepResult:
    """One game's adapter-free sweep result, after any reproduction gate."""

    game: str
    solver: str
    status: str
    reached_level: int
    advanced: bool
    exploration_actions_used: int
    dead_end_class: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    reproduction_gate: dict[str, Any] = field(default_factory=dict)
    reproduced_levels: int = 0
    trajectory_path: str = ""
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class CountingEnv:
    """Thin env wrapper that counts real `step()` calls made by graph explore."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.exploration_actions_used = 0

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.reset(*args, **kwargs)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        self.exploration_actions_used += 1
        return self._env.step(action, data=data, reasoning=reasoning)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class Tn36ClickSchemaEnv(CountingEnv):
    """Count actions and normalize tn36 ACTION6 clicks to top-level x/y data."""

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        if _is_action6(action):
            data = normalise_tn36_click_payload(data)
        return super().step(action, data=data, reasoning=reasoning)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _is_action6(action: Any) -> bool:
    if getattr(action, "name", "") == "ACTION6":
        return True
    value = getattr(action, "value", None)
    if value is not None:
        try:
            return int(value) == 6
        except (TypeError, ValueError):
            return False
    try:
        return int(action) == 6
    except (TypeError, ValueError):
        return str(action).endswith("ACTION6")


def normalise_tn36_click_payload(data: Any) -> Any:
    """SCENARIO-PHASE4-077: convert tn36 click payload variants to top-level x/y."""

    if data is None or not isinstance(data, dict):
        return data
    if "x" in data and "y" in data:
        return {"x": int(data["x"]), "y": int(data["y"])}
    for key in ("position", "click", "data"):
        nested = data.get(key)
        if isinstance(nested, dict) and "x" in nested and "y" in nested:
            return {"x": int(nested["x"]), "y": int(nested["y"])}
    return data


def tn36_schema_finding_from_source(source_text: str) -> dict[str, Any]:
    """REQ-PHASE4-077: derive tn36 ACTION6 click schema from the game source."""

    evidence = 'self.action.data["x"], self.action.data["y"]'
    if evidence not in source_text:
        return {
            "game": "tn36",
            "action": "ACTION6",
            "schema": "unknown",
            "payload_schema": {},
            "source_evidence": "",
            "rejects_nested_payloads": False,
            "normalizer": "normalise_tn36_click_payload",
            "wrapped_payload_explorer_advanced": False,
            "exploration_actions_used": 0,
        }
    return {
        "game": "tn36",
        "action": "ACTION6",
        "schema": 'ACTION6 data must be top-level {"x": int, "y": int}',
        "payload_schema": {"x": "int display_x", "y": "int display_y"},
        "source_evidence": evidence,
        "rejects_nested_payloads": True,
        "normalizer": "normalise_tn36_click_payload",
        "wrapped_payload_explorer_advanced": False,
        "exploration_actions_used": 0,
    }


def read_tn36_source(repo: Path) -> str:
    candidates = sorted((repo / "environment_files" / "tn36").glob("*/tn36.py"))
    if not candidates:
        return ""
    return candidates[-1].read_text(encoding="utf-8")


def compute_reproducibility_checksum(
    *,
    per_game_rows: dict[str, Any],
    tn36_schema_finding: dict[str, Any],
    random_seed: int,
) -> str:
    """REQ-PHASE4-077: hash trajectories, reproduce gates, schema finding, and seed."""

    payload = {
        "per_game_rows": per_game_rows,
        "random_seed": int(random_seed),
        "tn36_schema_finding": tn36_schema_finding,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _advanced_games(per_game_rows: dict[str, dict[str, Any]]) -> list[str]:
    return [
        game
        for game in SWEEP_GAMES
        if bool(per_game_rows.get(game, {}).get("advanced"))
        and int(per_game_rows.get(game, {}).get("reproduced_levels", 0) or 0) >= 1
        and bool(per_game_rows.get(game, {}).get("reproduction_gate", {}).get("reproduced"))
    ]


def build_artifact(
    *,
    per_game_results: dict[str, GameSweepResult],
    tn36_schema_finding: dict[str, Any],
    preconditions_checked: dict[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-077: build terminal artifact from sweep and reproduce rows."""

    per_game_rows = {game: result.to_json() for game, result in per_game_results.items()}
    games_advanced = _advanced_games(per_game_rows)
    reproduced_delta = sum(
        int(per_game_rows[game].get("reproduced_levels", 0) or 0) for game in games_advanced
    )
    total = PRIOR_REPRODUCIBLE_TOTAL_LEVELS + reproduced_delta
    if games_advanced:
        verdict = f"success: adapter_free_shallow_tail_{len(games_advanced)}_games_advanced_total{total}"
    else:
        verdict = "complete: adapter_free_shallow_tail_no_advance_real_exploration_total13"
    checksum = compute_reproducibility_checksum(
        per_game_rows=per_game_rows,
        tn36_schema_finding=tn36_schema_finding,
        random_seed=random_seed,
    )
    artifact = {
        "experiment": "experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail",
        "requirements": list(REQUIREMENTS),
        "honest_verdict": verdict,
        "reproducible_total_levels": int(total),
        "prior_reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
        "games_advanced": games_advanced,
        "per_game_exploration_actions": per_game_rows,
        "tn36_schema_finding": tn36_schema_finding,
        "offline_reproduced": bool(games_advanced),
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "swept_games": list(SWEEP_GAMES),
        "shallow_tail_candidates": list(SHALLOW_TAIL_GAMES),
        "excluded_games": list(EXCLUDED_GAMES),
        "discovery_budget": DEFAULT_MAX_DISCOVERY_EXPANSIONS,
        "advance_budget_inflated": False,
        "submitted_to_leaderboard": False,
        "duration_s": round(float(duration_s), 3),
    }
    return artifact


def blocked_artifact(
    *,
    preconditions_checked: dict[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-077: stop honestly when offline ARC resources are unavailable."""

    tn36_finding = tn36_schema_finding_from_source("")
    checksum = compute_reproducibility_checksum(
        per_game_rows={},
        tn36_schema_finding=tn36_finding,
        random_seed=random_seed,
    )
    return {
        "experiment": "experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail",
        "requirements": list(REQUIREMENTS),
        "honest_verdict": "blocked_arc_env_unreachable",
        "reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
        "prior_reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
        "games_advanced": [],
        "per_game_exploration_actions": {},
        "tn36_schema_finding": tn36_finding,
        "offline_reproduced": False,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "swept_games": list(SWEEP_GAMES),
        "shallow_tail_candidates": list(SHALLOW_TAIL_GAMES),
        "excluded_games": list(EXCLUDED_GAMES),
        "discovery_budget": DEFAULT_MAX_DISCOVERY_EXPANSIONS,
        "advance_budget_inflated": False,
        "submitted_to_leaderboard": False,
        "duration_s": round(float(duration_s), 3),
    }


def _terminal_verdict(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(
        ("success:", "complete:", "blocked_arc_env_unreachable")
    )


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-077: validate the terminal Exp 4330 artifact."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing {field_name}")
    if not _terminal_verdict(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be a bare int")
    elif artifact["reproducible_total_levels"] < PRIOR_REPRODUCIBLE_TOTAL_LEVELS:
        errors.append("reproducible_total_levels must be >= 13")
    if not isinstance(artifact.get("games_advanced"), list):
        errors.append("games_advanced must be a list")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(
        char not in "0123456789abcdef" for char in checksum
    ):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles missing")
    else:
        for field_name, principle in REQUIRED_FIELD_PRINCIPLES.items():
            if principles.get(field_name) != principle:
                errors.append(f"principle mismatch for {field_name}")
    if artifact.get("honest_verdict") == "blocked_arc_env_unreachable":
        return errors
    rows = artifact.get("per_game_exploration_actions")
    if not isinstance(rows, dict):
        errors.append("per_game_exploration_actions must be a dict")
        return errors
    if set(rows) != set(SWEEP_GAMES):
        errors.append("per_game_exploration_actions must include exactly swept games")
    for game, row in rows.items():
        if game in EXCLUDED_GAMES:
            errors.append(f"{game} must not be swept")
        if not isinstance(row, dict):
            errors.append(f"{game} row must be a dict")
            continue
        if int(row.get("exploration_actions_used", 0) or 0) <= 0:
            errors.append(f"{game} exploration_actions_used must be >0")
        if type(row.get("advanced")) is not bool:
            errors.append(f"{game}.advanced must be a bare bool")
        if row.get("advanced"):
            if int(row.get("reproduced_levels", 0) or 0) < 1:
                errors.append(f"{game} advanced row must include reproduced_levels>=1")
            gate = row.get("reproduction_gate")
            if not isinstance(gate, dict) or gate.get("reproduced") is not True:
                errors.append(f"{game} advanced row must reproduce offline")
    tn36 = artifact.get("tn36_schema_finding")
    if not isinstance(tn36, dict) or tn36.get("game") != "tn36":
        errors.append("tn36_schema_finding must describe tn36")
    elif tn36.get("schema") == "unknown":
        errors.append("tn36_schema_finding must include the ACTION6 schema")
    return errors


def _script_importable(path: Path, module_name: str) -> bool:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return True


def check_preconditions(repo: Path) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    env_dir = repo / "environment_files"
    env_present = {
        game: (env_dir / game).is_dir() and any((env_dir / game).iterdir())
        for game in SWEEP_GAMES
    }
    preconditions: dict[str, Any] = {
        "sweep_driver_import": False,
        "arc_graph_explore_import": False,
        "arc_solver_kit_import": False,
        "offline_env_reachable": False,
        "environment_files_present": env_present,
        "candidate_games": list(SHALLOW_TAIL_GAMES),
        "excluded_games": list(EXCLUDED_GAMES),
        "discovery_budget": int(os.environ.get("ARC_MAX_EXPANSIONS", DEFAULT_MAX_DISCOVERY_EXPANSIONS)),
        "advance_budget_inflated": False,
        "leaderboard_submission": False,
    }
    try:
        preconditions["sweep_driver_import"] = _script_importable(
            repo / "scripts" / "arc_explore_sweep.py",
            "arc_explore_sweep_preflight",
        )
        from carnot.agentic import arc_solver_kit
        from carnot.agentic import arc_graph_explore

        preconditions["arc_solver_kit_import"] = callable(getattr(arc_solver_kit, "reproduce", None))
        preconditions["arc_graph_explore_import"] = callable(
            getattr(arc_graph_explore, "graph_explore_solve_v2", None)
        )
        preconditions["offline_env_reachable"] = bool(
            all(env_present.values())
            and preconditions["sweep_driver_import"]
            and preconditions["arc_solver_kit_import"]
            and preconditions["arc_graph_explore_import"]
        )
    except Exception as exc:
        preconditions["error"] = f"{type(exc).__name__}: {exc}"
    return preconditions


def _preconditions_ok(preconditions_checked: dict[str, Any]) -> bool:
    return (
        preconditions_checked.get("sweep_driver_import") is True
        and preconditions_checked.get("arc_graph_explore_import") is True
        and preconditions_checked.get("arc_solver_kit_import") is True
        and preconditions_checked.get("offline_env_reachable") is True
        and all((preconditions_checked.get("environment_files_present") or {}).values())
    )


def _write_trajectory(repo: Path, game: str, reached_level: int, trajectory: list[dict[str, Any]]) -> str:
    output = repo / "results" / f"arc_explore_trajectory_{game}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"game": game, "reached_level": int(reached_level), "trajectory": trajectory}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output.relative_to(repo))


def _apply_replay_label(game: str) -> Callable[[Any, str, Any], Any]:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def apply(env: Any, label: str, frame: Any) -> Any:
        step = json.loads(label)
        data = step.get("data")
        if game == "tn36":
            data = normalise_tn36_click_payload(data)
        return env.step(_game_action(GameAction, int(step["action"])), data=data)

    return apply


def sweep_one_game(
    *,
    repo: Path,
    game: str,
    max_expansions: int,
    max_depth: int,
) -> GameSweepResult:  # pragma: no cover - SDK boundary
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels

    arcade = kit.offline_arcade()
    raw_env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    env: CountingEnv = Tn36ClickSchemaEnv(raw_env) if game == "tn36" else CountingEnv(raw_env)
    solver = "graph_explore_solve_v2"
    try:
        trajectory, reached_level = graph_explore_solve_v2(
            env,
            0,
            max_expansions=max_expansions,
            max_depth=max_depth,
            warmup=False,
            mask_hud=True,
        )
        if trajectory is None and game != "tn36":
            warm_trajectory, warm_level = graph_explore_solve_v2(
                env,
                0,
                max_expansions=max_expansions,
                max_depth=max_depth,
                warmup=True,
                mask_hud=True,
            )
            reached_level = max(int(reached_level), int(warm_level))
            if warm_trajectory is not None:
                solver = "graph_explore_solve_v2_warmup"
                trajectory = warm_trajectory
    except Exception as exc:
        return GameSweepResult(
            game=game,
            solver=solver,
            status="error",
            reached_level=0,
            advanced=False,
            exploration_actions_used=env.exploration_actions_used,
            dead_end_class=f"solver_exception_{type(exc).__name__}",
            error=repr(exc),
        )

    if trajectory:
        labels = trajectory_labels(trajectory)
        reproduction_gate = dict(
            kit.reproduce(game, labels, _apply_replay_label(game), claimed_level=int(reached_level))
        )
        reproduced = bool(reproduction_gate.get("reproduced"))
        reproduced_levels = int(reproduction_gate.get("reached_level", 0) or 0) if reproduced else 0
        if reproduced and reproduced_levels >= 1:
            trajectory_path = _write_trajectory(repo, game, int(reached_level), trajectory)
            return GameSweepResult(
                game=game,
                solver=solver,
                status="advanced",
                reached_level=int(reached_level),
                advanced=True,
                exploration_actions_used=env.exploration_actions_used,
                dead_end_class="none",
                trajectory=trajectory,
                reproduction_gate=reproduction_gate,
                reproduced_levels=reproduced_levels,
                trajectory_path=trajectory_path,
            )
        return GameSweepResult(
            game=game,
            solver=solver,
            status="no_advance",
            reached_level=int(reached_level),
            advanced=False,
            exploration_actions_used=env.exploration_actions_used,
            dead_end_class="advance_unreproduced_reproduction_gate_failed",
            trajectory=trajectory,
            reproduction_gate=reproduction_gate,
            reproduced_levels=0,
        )

    dead_end = "tn36_schema_wrapped_no_level_delta_12000_budget" if game == "tn36" else (
        "adapter_free_no_level_delta_12000_budget"
    )
    return GameSweepResult(
        game=game,
        solver=solver,
        status="no_advance",
        reached_level=int(reached_level),
        advanced=False,
        exploration_actions_used=env.exploration_actions_used,
        dead_end_class=dead_end,
        reproduction_gate={"game": game, "reached_level": 0, "claimed_level": 1, "reproduced": False},
    )


def run_sweep(
    *,
    repo: Path,
    max_expansions: int,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict[str, GameSweepResult]:  # pragma: no cover - SDK boundary
    return {
        game: sweep_one_game(
            repo=repo,
            game=game,
            max_expansions=max_expansions,
            max_depth=max_depth,
        )
        for game in SWEEP_GAMES
    }


def _write_artifact(repo: Path, artifact: dict[str, Any]) -> None:
    output = repo / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    repo: Path = REPO,
    write: bool = True,
    sweep_fn: Callable[..., dict[str, GameSweepResult]] | None = None,
    precondition_fn: Callable[[Path], dict[str, Any]] = check_preconditions,
    tn36_source_fn: Callable[[Path], str] = read_tn36_source,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """REQ-PHASE4-077: run the shallow-tail sweep and write the terminal artifact."""

    started = time.time()
    preconditions_checked = precondition_fn(repo)
    if not _preconditions_ok(preconditions_checked):
        artifact = blocked_artifact(
            preconditions_checked=preconditions_checked,
            random_seed=random_seed,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(repo, artifact)
        return artifact

    max_expansions = int(
        preconditions_checked.get("discovery_budget")
        or os.environ.get("ARC_MAX_EXPANSIONS", DEFAULT_MAX_DISCOVERY_EXPANSIONS)
    )
    sweep = sweep_fn or run_sweep
    per_game_results = sweep(
        repo=repo,
        max_expansions=max_expansions,
        max_depth=DEFAULT_MAX_DEPTH,
    )
    tn36_finding = tn36_schema_finding_from_source(tn36_source_fn(repo))
    tn36_result = per_game_results.get("tn36")
    if tn36_result is not None:
        tn36_finding = {
            **tn36_finding,
            "wrapped_payload_explorer_advanced": bool(tn36_result.advanced),
            "exploration_actions_used": int(tn36_result.exploration_actions_used),
            "dead_end_class": tn36_result.dead_end_class,
        }
    artifact = build_artifact(
        per_game_results=per_game_results,
        tn36_schema_finding=tn36_finding,
        preconditions_checked=preconditions_checked,
        random_seed=random_seed,
        duration_s=time.time() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(repo, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    artifact = run(random_seed=args.seed, write=True)
    print(f"-> {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
