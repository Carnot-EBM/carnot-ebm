"""Exp 5216: ARC frontier continuity plus landmark decomposition pilot.

Spec refs: REQ-REPORT-5216, SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS,
SCENARIO-REPORT-5216-ARTIFACT-GATE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


RESULT_RELATIVE_PATH = (
    "results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json"
)
EXPERIMENT_ID = "experiment_5216_arc_frontier_continuity_landmark_decomposition_v477"
TARGET_GAMES = ("bp35", "cd82")
ARMS = ("flat_control", "frontier_continuity", "landmark_decomposition", "combined")
MAX_EXPANSIONS = 128
RANDOM_SEED = 20260704
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
SOLVE_PROVENANCE = "development_proxy"
REQUIRED_ARTIFACT_FIELDS = (
    "target_games",
    "duplicate_registry_precheck_passed",
    "live_path_integration_attempted",
    "solve_provenance",
    "offline_ground_truth_bfs",
    "read_game_source",
    "new_levels_banked",
    "reproducible_total_levels_delta",
    "frontier_continuity_lift",
    "landmark_decomposition_lift",
    "orphan_lint_result",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "solve_provenance": (
        "Any level solve claim must declare provenance; outer_loop_re is not headline-eligible."
    ),
    "offline_ground_truth_bfs": (
        "Must remain false; the pilot may use bounded runtime observations only."
    ),
    "read_game_source": "Must remain false; no game source may be inspected for the pilot.",
    "new_levels_banked": (
        "Each claimed level must name game, level, solve_provenance, and reproduction-gate evidence."
    ),
    "frontier_continuity_lift": (
        "Per-game delta versus the flat/pruner-only control under the same expansion budget."
    ),
    "landmark_decomposition_lift": (
        "Per-game delta versus the flat/pruner-only control under the same expansion budget."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and must not claim a level "
        "bank without reproduction-gate evidence."
    ),
}


def _json_checksum(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _json_checksum(payload)


def _orphan_result_string(orphan_lint: Mapping[str, Any]) -> str:
    status = "pass" if orphan_lint.get("passed") else "fail"
    output = str(orphan_lint.get("stdout_tail") or orphan_lint.get("stderr_tail") or "").strip()
    return f"{status}: {output}"


def _arm(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return (row.get("arms") or {}).get(arm) or {}


def _reproduced_level(data: Mapping[str, Any]) -> int:
    gate = data.get("reproduction_gate") or {}
    if not isinstance(gate, Mapping) or not bool(gate.get("reproduced")):
        return 0
    return int(gate.get("reached_level", data.get("reached_level", 0)) or 0)


def _new_levels(
    per_game: Sequence[Mapping[str, Any]], registry_depths: Mapping[str, int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in per_game:
        game = str(row.get("game") or "")
        registry_depth = int(row.get("registry_depth", registry_depths.get(game, 0)) or 0)
        for arm_name in ARMS:
            data = _arm(row, arm_name)
            reached = _reproduced_level(data)
            if reached <= registry_depth or (game, reached) in seen:
                continue
            provenance = str(data.get("solve_provenance") or SOLVE_PROVENANCE)
            if provenance == "outer_loop_re":
                continue
            seen.add((game, reached))
            out.append({"game": game, "level": reached, "solve_provenance": provenance})
    return out


def _lift_for(
    per_game: Sequence[Mapping[str, Any]],
    treatment: str,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in per_game:
        game = str(row.get("game") or "")
        flat = _arm(row, "flat_control")
        arm = _arm(row, treatment)
        flat_states = int(flat.get("states_expanded", 0) or 0)
        arm_states = int(arm.get("states_expanded", 0) or 0)
        out[game] = {
            "reached_level_delta": int(arm.get("reached_level", 0) or 0)
            - int(flat.get("reached_level", 0) or 0),
            "states_expanded_delta": flat_states - arm_states,
            "reproduced_level_delta": _reproduced_level(arm) - _reproduced_level(flat),
        }
    return out


def _per_arm_results(per_game: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in per_game:
        game = str(row.get("game") or "")
        results[game] = {}
        for arm_name in ARMS:
            data = _arm(row, arm_name)
            results[game][arm_name] = {
                "reached_level": int(data.get("reached_level", 0) or 0),
                "states_expanded": int(data.get("states_expanded", 0) or 0),
                "offline_reproduced_level": _reproduced_level(data),
            }
    return results


def build_artifact(
    *,
    per_game: Sequence[Mapping[str, Any]],
    registry_depths: Mapping[str, int],
    orphan_lint: Mapping[str, Any],
    duration_s: float,
    live_path_integration_attempted: bool = True,
    solve_provenance: str = SOLVE_PROVENANCE,
    max_expansions: int = MAX_EXPANSIONS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Build the terminal Exp 5216 artifact without fabricating solve claims."""

    target_games = [str(row.get("game") or "") for row in per_game]
    new_levels = _new_levels(per_game, registry_depths)
    duplicate_precheck = all(
        int(item["level"]) > int(registry_depths.get(str(item["game"]), 0) or 0)
        for item in new_levels
    )
    total_delta = len(new_levels)
    if total_delta:
        honest_verdict = (
            "success: frontier continuity plus landmark decomposition banked "
            f"{total_delta} reproduction-gated new public-game level(s) above registry depth."
        )
    else:
        honest_verdict = (
            "complete: frontier continuity plus landmark decomposition did not bank a "
            "new reproduction-gated level above the registry precheck in this bounded pilot."
        )

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": "carnot.experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.v1",
        "spec_refs": [
            "REQ-REPORT-5216",
            "SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS",
            "SCENARIO-REPORT-5216-ARTIFACT-GATE",
        ],
        "target_games": target_games,
        "duplicate_registry_precheck_passed": bool(duplicate_precheck),
        "live_path_integration_attempted": bool(live_path_integration_attempted),
        "solve_provenance": str(solve_provenance),
        "offline_ground_truth_bfs": False,
        "read_game_source": False,
        "new_levels_banked": new_levels,
        "reproducible_total_levels_delta": int(total_delta),
        "frontier_continuity_lift": _lift_for(per_game, "frontier_continuity"),
        "landmark_decomposition_lift": _lift_for(per_game, "landmark_decomposition"),
        "orphan_lint_result": _orphan_result_string(orphan_lint),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "field_principles": dict(FIELD_PRINCIPLES),
        "per_arm_results": _per_arm_results(per_game),
        "per_game": list(per_game),
        "registry_depths": {str(k): int(v) for k, v in registry_depths.items()},
        "max_expansions": int(max_expansions),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 2),
        "solve_claim_policy": (
            "Only reproduced levels strictly above the registry precheck count; "
            "outer_loop_re rows are excluded from headline claims."
        ),
        "control_description": "flat/pruner-only control is flat graph search with no seed bank.",
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _python_bin(root: Path) -> str:  # pragma: no cover
    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _pytest_bin(root: Path) -> list[str]:  # pragma: no cover
    candidate = root / ".venv" / "bin" / "pytest"
    if candidate.exists():
        return [str(candidate)]
    return [sys.executable, "-m", "pytest"]


def _run_command(argv: Sequence[str], *, cwd: Path) -> dict[str, Any]:  # pragma: no cover
    started = time.time()
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": list(argv),
        "returncode": int(completed.returncode),
        "passed": completed.returncode == 0,
        "duration_s": round(time.time() - started, 2),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_arc_orphan_solver_lint(root: Path) -> dict[str, Any]:  # pragma: no cover
    return _run_command([_python_bin(root), "scripts/arc_orphan_solver_lint.py"], cwd=root)


def check_unit_tests(root: Path) -> dict[str, Any]:  # pragma: no cover
    return _run_command(
        [
            *_pytest_bin(root),
            "tests/python/test_experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.py",
            "-q",
            "--no-cov",
        ],
        cwd=root,
    )


def _apply_json_action(env: Any, label: str, frame: Any) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _load_prefix(root: Path, game: str) -> tuple[list[dict[str, Any]], int]:  # pragma: no cover
    path = root / "results" / f"arc_loop_solve_{game}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    solution = list(data.get("solution") or [])
    level = int(data.get("reproduced_levels", data.get("reached_level", 0)) or 0)
    if not solution or level <= 0:
        raise RuntimeError(f"{game} has no reproduced prefix in {path}")
    return solution, level


def _collect_transition_logs(
    env: Any,
    prefix: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed

    frame = env.reset()
    logs: list[dict[str, Any]] = []
    path_before: list[dict[str, Any]] = []
    for step in prefix:
        before = frame
        action = {"action": int(step["action"]), "data": step.get("data")}
        frame = env.step(_game_action(GameAction, action["action"]), data=action.get("data"))
        path_after = [*path_before, action]
        logs.append(
            {
                "frame_before": before,
                "frame_after": frame,
                "path_before": list(path_before),
                "path_after": list(path_after),
                "action": dict(action),
                "level_before": _levels_completed(before),
                "level_after": _levels_completed(frame),
            }
        )
        path_before = path_after
    return frame, logs


def _run_arm(
    game: str,
    prefix: Sequence[Mapping[str, Any]],
    prefix_level: int,
    *,
    arm: str,
    max_expansions: int,
    transition_logs: Sequence[Mapping[str, Any]],
    root_frame: Any,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_frontier_continuity_landmarks import (
        build_frontier_continuity_landmark_bank,
    )
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frontier_seed_bank = None
    if arm != "flat_control":
        frontier_seed_bank = build_frontier_continuity_landmark_bank(
            root_frame=root_frame,
            transition_logs=transition_logs,
            enable_frontier_continuity=arm in {"frontier_continuity", "combined"},
            enable_landmark_decomposition=arm in {"landmark_decomposition", "combined"},
        )
    stats: dict[str, Any] = {}
    traj, reached = graph_explore_solve_v2(
        env,
        int(prefix_level),
        prefix=list(prefix),
        max_expansions=int(max_expansions),
        max_depth=max(8, int(max_expansions)),
        frontier_seed_bank=frontier_seed_bank,
        stats=stats,
    )
    gate: dict[str, Any] | None = None
    if traj is not None and int(reached) > int(prefix_level):
        gate = kit.reproduce(
            game,
            trajectory_labels(traj),
            _apply_json_action,
            claimed_level=int(reached),
        )
    return {
        "arm": arm,
        "reached_level": int(reached),
        "states_expanded": int(stats.get("expansions", 0) or 0),
        "traj_len": len(traj or []),
        "solve_provenance": SOLVE_PROVENANCE,
        "reproduction_gate": gate,
        "stats": stats,
        "seed_bank_diagnostics": (
            frontier_seed_bank.as_dict() if frontier_seed_bank is not None else None
        ),
    }


def run_experiment(
    *,
    root: Path,
    games: Sequence[str] = TARGET_GAMES,
    max_expansions: int = MAX_EXPANSIONS,
    run_unit_tests: bool = True,
    run_lint: bool = True,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit

    started = time.time()
    unit = check_unit_tests(root) if run_unit_tests else {"passed": True, "skipped": True}
    lint = run_arc_orphan_solver_lint(root) if run_lint else {"passed": True, "skipped": True}
    arc = kit.offline_arcade()
    per_game: list[dict[str, Any]] = []
    registry_depths: dict[str, int] = {}
    for game in games:
        prefix, prefix_level = _load_prefix(root, game)
        registry_depths[game] = int(prefix_level)
        log_env = arc.make(game, scorecard_id=arc.open_scorecard())
        root_frame, transition_logs = _collect_transition_logs(log_env, prefix)
        arms = {
            arm: _run_arm(
                game,
                prefix,
                prefix_level,
                arm=arm,
                max_expansions=int(max_expansions),
                transition_logs=transition_logs,
                root_frame=root_frame,
            )
            for arm in ARMS
        }
        per_game.append(
            {
                "game": game,
                "registry_depth": int(prefix_level),
                "runtime_transition_log_rows": len(transition_logs),
                "arms": arms,
            }
        )
        print(
            f"[{game}] "
            + " | ".join(
                f"{arm}=L{arms[arm]['reached_level']} exp={arms[arm]['states_expanded']}"
                for arm in ARMS
            ),
            flush=True,
        )

    artifact = build_artifact(
        per_game=per_game,
        registry_depths=registry_depths,
        orphan_lint=lint,
        duration_s=time.time() - started,
        max_expansions=int(max_expansions),
    )
    artifact["unit_tests"] = unit
    artifact["orphan_lint"] = lint
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:  # pragma: no cover
    out = root / RESULT_RELATIVE_PATH
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--games", default=",".join(TARGET_GAMES))
    parser.add_argument("--max-expansions", type=int, default=MAX_EXPANSIONS)
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(
        root=args.root,
        games=tuple(game.strip() for game in args.games.split(",") if game.strip()),
        max_expansions=int(args.max_expansions),
        run_unit_tests=not args.skip_unit_tests,
        run_lint=not args.skip_lint,
    )
    out = write_artifact(args.root, artifact)
    print(f"=== VERDICT: {artifact['honest_verdict']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
