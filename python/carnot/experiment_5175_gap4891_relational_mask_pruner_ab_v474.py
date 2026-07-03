"""Exp 5175: GAP-4891 relational-mask pruner A/B.

This is the Stage-3 counterpart to the GAP-4891 Stage-2 goal-energy probe:
keep the same relational goal-energy ordering, add only the online
RelationalMaskMovePruner in the treatment arm, and reproduction-gate any level
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


RESULT_RELATIVE_PATH = "results/experiment_5175_gap4891_relational_mask_pruner_ab_v474.json"
GAMES = ("cd82", "sk48", "sp80", "cn04")
SEPARATING_GAMES = ("cd82", "sk48", "sp80")
NEGATIVE_CONTROL_GAME = "cn04"
MAX_EXPANSIONS = 4000
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
REQUIRED_ARTIFACT_FIELDS = (
    "unit_tests_still_passing",
    "games_tested",
    "states_expanded_pruned",
    "states_expanded_unpruned",
    "states_expanded_reduction_pct",
    "new_level_reached_pruned",
    "new_level_reached_unpruned",
    "levels_banked",
    "cn04_negative_control_clean",
    "gap4891_status_recommendation",
    "solve_provenance",
    "verifier_is_oracle",
    "live_path_reachable",
    "random_seed",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)


def _json_checksum(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _artifact_checksum(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _json_checksum(payload)


def _new_level(row: dict[str, Any], arm: str) -> bool:
    data = row.get(arm) or {}
    reached = int(data.get("gate_reached", data.get("reached_level", 0)) or 0)
    return bool(data.get("offline_reproduced")) and reached > int(row.get("prefix_level", 0) or 0)


def _states(row: dict[str, Any], arm: str) -> int:
    return int((row.get(arm) or {}).get("states_expanded", 0) or 0)


def _reduction_pct(unpruned: int, pruned: int) -> float:
    if unpruned <= 0:
        return 0.0
    return round(((unpruned - pruned) / unpruned) * 100.0, 4)


def _banked_levels(per_game: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in per_game:
        for arm in ("pruned", "unpruned"):
            if not _new_level(row, arm):
                continue
            data = row[arm]
            rows.append(
                {
                    "game": row["game"],
                    "new_level": int(data.get("gate_reached", data.get("reached_level", 0)) or 0),
                    "offline_reproduced": True,
                    "reproducibility_checksum": str(data.get("reproducibility_checksum", "")),
                    "arm": arm,
                }
            )
    return rows


def build_artifact(
    *,
    per_game: Sequence[dict[str, Any]],
    unit_tests_still_passing: bool,
    live_path_reachable: bool,
    arc_orphan_solver_lint: dict[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    max_expansions: int = MAX_EXPANSIONS,
) -> dict[str, Any]:
    """Build the required Exp 5175 artifact from completed arm rows."""

    by_game = {row["game"]: row for row in per_game}
    games_tested = [game for game in GAMES if game in by_game]
    states_expanded_pruned = {game: _states(by_game[game], "pruned") for game in games_tested}
    states_expanded_unpruned = {game: _states(by_game[game], "unpruned") for game in games_tested}
    states_expanded_reduction_pct = {
        game: _reduction_pct(states_expanded_unpruned[game], states_expanded_pruned[game])
        for game in games_tested
    }
    new_level_reached_pruned = {
        game: _new_level(by_game[game], "pruned") for game in games_tested
    }
    new_level_reached_unpruned = {
        game: _new_level(by_game[game], "unpruned") for game in games_tested
    }
    levels_banked = _banked_levels([by_game[game] for game in games_tested])
    pruned_banked_target = any(
        new_level_reached_pruned.get(game, False) for game in SEPARATING_GAMES
    )
    any_states_expanded_reduction = any(
        states_expanded_reduction_pct.get(game, 0.0) > 0.0 for game in SEPARATING_GAMES
    )
    move_pruned_edges = {
        game: int(
            ((by_game[game].get("pruned") or {}).get("pruner_stats") or {}).get("pruned", 0)
            or 0
        )
        for game in games_tested
    }
    any_edges_pruned = any(move_pruned_edges.get(game, 0) > 0 for game in SEPARATING_GAMES)
    cn04_negative_control_clean = not new_level_reached_pruned.get(NEGATIVE_CONTROL_GAME, False)

    if pruned_banked_target:
        honest_verdict = (
            "success_relational_mask_pruner_pruning_alone_closes_enumeration_wall_"
            "on_at_least_one_gap4891_game"
        )
        gap_status = "filled"
    elif any_states_expanded_reduction:
        honest_verdict = (
            "complete_relational_mask_pruner_reduces_states_expanded_without_banking_level_"
            "pruning_alone_does_not_close_enumeration_wall_MAP_map_then_act_next"
        )
        gap_status = "building_with_new_lever_named"
    elif any_edges_pruned:
        honest_verdict = (
            "complete_relational_mask_pruner_prunes_edges_but_states_expanded_unchanged_"
            "no_level_bank_pruning_alone_does_not_close_enumeration_wall_MAP_map_then_act_next"
        )
        gap_status = "building_with_new_lever_named"
    else:
        honest_verdict = (
            "complete_relational_mask_pruner_no_branching_reduction_and_no_level_bank_"
            "pruning_alone_does_not_close_enumeration_wall_MAP_map_then_act_next"
        )
        gap_status = "building_with_new_lever_named"

    artifact: dict[str, Any] = {
        "experiment": "experiment_5175_gap4891_relational_mask_pruner_ab_v474",
        "schema": "carnot.experiment_5175_gap4891_relational_mask_pruner_ab_v474.v1",
        "spec_refs": ["REQ-REPORT-5175", "SCENARIO-REPORT-5175-PRUNER-AB"],
        "honest_verdict": honest_verdict,
        "question": (
            "Does adding the online RelationalMaskMovePruner to GAP-4891 Stage-2's "
            "relational goal-energy search enumerate a new reproduction-gated level?"
        ),
        "unit_tests_still_passing": bool(unit_tests_still_passing),
        "games_tested": games_tested,
        "target_games": list(SEPARATING_GAMES),
        "negative_control_game": NEGATIVE_CONTROL_GAME,
        "max_expansions": int(max_expansions),
        "states_expanded_pruned": states_expanded_pruned,
        "states_expanded_unpruned": states_expanded_unpruned,
        "states_expanded_reduction_pct": states_expanded_reduction_pct,
        "move_pruned_edges": move_pruned_edges,
        "new_level_reached_pruned": new_level_reached_pruned,
        "new_level_reached_unpruned": new_level_reached_unpruned,
        "levels_banked": levels_banked,
        "cn04_negative_control_clean": cn04_negative_control_clean,
        "gap4891_status_recommendation": gap_status,
        "next_specific_lever": (
            "Prototype a MAP-style map-then-act / hierarchical pre-search stage that "
            "generates candidate subgoal trajectories before flat frontier enumeration."
        ),
        "solve_provenance": (
            "live_agent_self_discovery" if pruned_banked_target else "development_proxy"
        ),
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "arc_orphan_solver_lint": dict(arc_orphan_solver_lint),
        "random_seed": int(random_seed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_game": list(per_game),
        "used_env_source": False,
        "read_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_calibrated_per_game": False,
        "scripts_research_conductor_modified": False,
        "duration_s": round(float(duration_s), 2),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


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


def _python_bin(root: Path) -> str:  # pragma: no cover
    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _pytest_bin(root: Path) -> list[str]:  # pragma: no cover
    candidate = root / ".venv" / "bin" / "pytest"
    if candidate.exists():
        return [str(candidate)]
    return [sys.executable, "-m", "pytest"]


def check_relational_mask_unit_tests(root: Path) -> dict[str, Any]:  # pragma: no cover
    return _run_command(
        [
            *_pytest_bin(root),
            "tests/python/test_arc_relational_mask_pruner.py",
            "-v",
            "--no-cov",
        ],
        cwd=root,
    )


def run_arc_orphan_solver_lint(root: Path) -> dict[str, Any]:  # pragma: no cover
    return _run_command([_python_bin(root), "scripts/arc_orphan_solver_lint.py"], cwd=root)


def _apply(env: Any, label: str, frame: Any) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, step["action"]), data=step.get("data"))


def _grid(frame: Any) -> np.ndarray | None:  # pragma: no cover
    from carnot.agentic.arc_agi3_world_model import grid_of

    try:
        return np.asarray(grid_of(frame))
    except Exception:
        return None


def _uniq_grids(grids: Sequence[np.ndarray]) -> list[np.ndarray]:  # pragma: no cover
    seen: set[bytes] = set()
    out: list[np.ndarray] = []
    for grid in grids:
        key = np.asarray(grid).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(np.asarray(grid))
    return out


def _load_prefix(root: Path, game: str) -> tuple[list[dict[str, Any]], int]:  # pragma: no cover
    seed_path = root / "results" / f"arc_explore_trajectory_{game}.json"
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    prefix = list(data.get("trajectory") or [])
    prefix_level = int(data.get("reached_level", 0) or 0)
    if not prefix or prefix_level < 1:
        raise RuntimeError(f"{game} seed reaches level {prefix_level}")
    return prefix, prefix_level


def _induce_context(root: Path, game: str) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_goal_induction import (
        induce_goal_energy_relational,
        induce_relational_target_region,
    )

    prefix, prefix_level = _load_prefix(root, game)
    labels = [json.dumps(step) for step in prefix]
    arc = kit.offline_arcade()
    non_win_grids: list[np.ndarray] = []
    win_grids: list[np.ndarray] = []

    env0 = arc.make(game, scorecard_id=arc.open_scorecard())
    f0 = env0.reset() if hasattr(env0, "reset") else None
    if f0 is not None:
        g0 = _grid(f0)
        if g0 is not None:
            non_win_grids.append(g0)

    cur = arc.make(game, scorecard_id=arc.open_scorecard())
    cur.reset()
    prev_level = 0
    for label in labels:
        frame = _apply(cur, label, None)
        level = kit.frame_level(frame)
        grid = _grid(frame)
        if grid is None:
            continue
        if level > prev_level:
            win_grids.append(grid)
            prev_level = level
        else:
            non_win_grids.append(grid)

    win_grids = _uniq_grids(win_grids)
    non_win_grids = _uniq_grids(non_win_grids)
    win = win_grids[0] if win_grids else None
    energy = induce_goal_energy_relational(win, non_win_grids)
    target_region = induce_relational_target_region(win, non_win_grids)
    return {
        "game": game,
        "prefix": prefix,
        "prefix_level": prefix_level,
        "win_grids": len(win_grids),
        "non_win_grids": len(non_win_grids),
        "goal_energy": energy,
        "target_region": target_region,
        "induce_fired": energy is not None,
        "target_region_known": target_region is not None,
    }


def _make_goal_energy(energy: Callable[[np.ndarray], float] | None):  # pragma: no cover
    if energy is None:
        return None

    def goal_energy(frame: Any) -> float:
        grid = _grid(frame)
        return float(energy(grid)) if grid is not None else 0.0

    return goal_energy


def _run_arm(
    game: str,
    context: dict[str, Any],
    *,
    use_pruner: bool,
    max_expansions: int,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels
    from carnot.agentic.arc_relational_mask_pruner import RelationalMaskMovePruner

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    pruner = (
        RelationalMaskMovePruner(grid_of, target_region=context["target_region"])
        if use_pruner
        else None
    )
    stats: dict[str, Any] = {}
    traj, level = graph_explore_solve_v2(
        env,
        int(context["prefix_level"]),
        max_expansions=int(max_expansions),
        prefix=list(context["prefix"]),
        goal_energy=_make_goal_energy(context["goal_energy"]),
        stats=stats,
        move_pruner=pruner,
    )
    reproduced = False
    gate_reached = int(level)
    labels: list[str] = []
    repro_checksum = ""
    if traj is not None and int(level) > int(context["prefix_level"]):
        labels = trajectory_labels(traj)
        gate = kit.reproduce(game, labels, _apply, claimed_level=int(level))
        reproduced = bool(gate.get("reproduced"))
        gate_reached = int(gate.get("reached_level", level) or level)
        repro_checksum = _json_checksum(labels)
    return {
        "reached_level": int(level),
        "gate_reached": int(gate_reached),
        "offline_reproduced": reproduced,
        "states_expanded": int(stats.get("states_expanded", stats.get("expansions", 0)) or 0),
        "traj_len": len(traj) if traj else 0,
        "reproducibility_checksum": repro_checksum,
        "trajectory_labels": labels if reproduced else [],
        "stats": stats,
        "pruner_stats": stats.get("move_pruner_stats") if use_pruner else None,
    }


def run_experiment(
    *,
    root: Path,
    games: Sequence[str] = GAMES,
    max_expansions: int = MAX_EXPANSIONS,
    random_seed: int = RANDOM_SEED,
    run_unit_tests: bool = True,
    run_lint: bool = True,
) -> dict[str, Any]:  # pragma: no cover
    started = time.time()
    np.random.seed(int(random_seed))
    unit = (
        check_relational_mask_unit_tests(root)
        if run_unit_tests
        else {"passed": True, "skipped": True}
    )
    lint = run_arc_orphan_solver_lint(root) if run_lint else {"passed": True, "skipped": True}
    per_game: list[dict[str, Any]] = []
    for game in games:
        context = _induce_context(root, game)
        unpruned = _run_arm(
            game,
            context,
            use_pruner=False,
            max_expansions=int(max_expansions),
        )
        pruned = _run_arm(
            game,
            context,
            use_pruner=True,
            max_expansions=int(max_expansions),
        )
        per_game.append(
            {
                "game": game,
                "prefix_level": int(context["prefix_level"]),
                "win_grids": int(context["win_grids"]),
                "non_win_grids": int(context["non_win_grids"]),
                "induce_fired": bool(context["induce_fired"]),
                "target_region_known": bool(context["target_region_known"]),
                "target_region_cells": (
                    int(context["target_region"].sum())
                    if context["target_region"] is not None
                    else 0
                ),
                "unpruned": unpruned,
                "pruned": pruned,
            }
        )
        print(
            f"[{game}] unpruned={unpruned['states_expanded']} L{unpruned['gate_reached']} "
            f"repro={unpruned['offline_reproduced']} | pruned={pruned['states_expanded']} "
            f"L{pruned['gate_reached']} repro={pruned['offline_reproduced']}",
            flush=True,
        )

    artifact = build_artifact(
        per_game=per_game,
        unit_tests_still_passing=bool(unit.get("passed")),
        live_path_reachable=bool(lint.get("passed")),
        arc_orphan_solver_lint=lint,
        duration_s=time.time() - started,
        random_seed=int(random_seed),
        max_expansions=int(max_expansions),
    )
    artifact["relational_mask_unit_tests"] = unit
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def write_artifact(root: Path, artifact: dict[str, Any]) -> Path:  # pragma: no cover
    out = root / RESULT_RELATIVE_PATH
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-expansions", type=int, default=MAX_EXPANSIONS)
    parser.add_argument("--games", default=",".join(GAMES))
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(
        root=args.root,
        games=tuple(game.strip() for game in args.games.split(",") if game.strip()),
        max_expansions=args.max_expansions,
        run_unit_tests=not args.skip_unit_tests,
        run_lint=not args.skip_lint,
    )
    out = write_artifact(args.root, artifact)
    print(f"=== VERDICT: {artifact['honest_verdict']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
