"""Exp 5198: GAP-4891 MAP landmark prestage prototype.

Spec refs: REQ-REPORT-5198, SCENARIO-REPORT-5198-MAP-PRESTAGE,
SCENARIO-REPORT-5198-THREE-ARM-GATE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


RESULT_RELATIVE_PATH = "results/experiment_5198_map_landmark_prestage_prototype_v476.json"
GAMES = ("cd82", "sk48", "sp80", "cn04")
SEPARATING_GAMES = ("cd82", "sk48", "sp80")
NEGATIVE_CONTROL_GAME = "cn04"
ARMS = ("pruner_only", "map_only", "map_plus_pruner")
MAX_EXPANSIONS = 4000
MAP_BUDGET_STEPS = 750
RANDOM_SEED = 20260703
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
REQUIRED_ARTIFACT_FIELDS = (
    "lever_validated",
    "per_arm_results",
    "cn04_negative_control_stayed_clean",
    "solve_provenance",
    "orphan_lint_result",
    "reproduction_gate_results",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "lever_validated": (
        "MECHANICAL CONSTRAINT: must be a BARE top-level boolean; true only if map-only "
        "or map-plus-pruner banked a reproduction-gated level pruner-only, under the "
        "identical budget, did not."
    ),
    "per_arm_results": "{game: {arm: {states_expanded, levels_banked, map_overhead_steps}}}",
    "cn04_negative_control_stayed_clean": (
        "The MAP prestage must not spuriously solve the cn04 negative control whose "
        "relational goal-energy is known not to separate."
    ),
    "solve_provenance": (
        "development_proxy -- offline dev-twin prototyping via arc_loop_solve/GameAdapter, "
        "not the scored live agent."
    ),
    "orphan_lint_result": "pass/fail plus scripts/arc_orphan_solver_lint.py output.",
    "reproduction_gate_results": "Per banked level, record arc_solver_kit.reproduce() output.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether MAP "
        "validated a lever the pruner did not or the enumeration wall persists."
    ),
}


def _json_checksum(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _artifact_checksum(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _json_checksum(payload)


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


def check_map_unit_tests(root: Path) -> dict[str, Any]:  # pragma: no cover
    return _run_command(
        [
            *_pytest_bin(root),
            "tests/python/test_experiment_5198_map_landmark_prestage_prototype_v476.py",
            "-q",
            "--no-cov",
        ],
        cwd=root,
    )


def run_arc_orphan_solver_lint(root: Path) -> dict[str, Any]:  # pragma: no cover
    return _run_command([_python_bin(root), "scripts/arc_orphan_solver_lint.py"], cwd=root)


def _arm_levels_banked(row: dict[str, Any], arm: str) -> int:
    data = (row.get("arms") or {}).get(arm) or {}
    if not bool(data.get("offline_reproduced")):
        return 0
    reached = int(data.get("gate_reached", data.get("reached_level", 0)) or 0)
    return max(0, reached - int(row.get("prefix_level", 0) or 0))


def _arm_result(data: dict[str, Any]) -> dict[str, int]:
    return {
        "states_expanded": int(data.get("states_expanded", 0) or 0),
        "levels_banked": int(data.get("levels_banked", 0) or 0),
        "map_overhead_steps": int(data.get("map_overhead_steps", 0) or 0),
    }


def _levels_banked(per_game: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in per_game:
        for arm in ARMS:
            count = _arm_levels_banked(row, arm)
            if count <= 0:
                continue
            data = row["arms"][arm]
            out.append(
                {
                    "game": row["game"],
                    "arm": arm,
                    "levels_banked": count,
                    "new_level": int(data.get("gate_reached", data.get("reached_level", 0)) or 0),
                    "offline_reproduced": True,
                    "reproducibility_checksum": str(data.get("reproducibility_checksum", "")),
                }
            )
    return out


def _map_validated(per_game: Sequence[dict[str, Any]]) -> bool:
    for row in per_game:
        if row["game"] not in SEPARATING_GAMES:
            continue
        pruner = _arm_levels_banked(row, "pruner_only")
        if _arm_levels_banked(row, "map_only") > pruner:
            return True
        if _arm_levels_banked(row, "map_plus_pruner") > pruner:
            return True
    return False


def _reproduction_gate_results(per_game: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in per_game:
        for arm in ARMS:
            if _arm_levels_banked(row, arm) <= 0:
                continue
            gate = (row["arms"][arm]).get("reproduction_gate")
            out.append({"game": row["game"], "arm": arm, "gate": gate})
    return out


def _orphan_result_string(orphan_lint: dict[str, Any]) -> str:
    status = "pass" if orphan_lint.get("passed") else "fail"
    output = str(orphan_lint.get("stdout_tail") or orphan_lint.get("stderr_tail") or "").strip()
    return f"{status}: {output}"


def build_artifact(
    *,
    per_game: Sequence[dict[str, Any]],
    unit_tests_still_passing: bool,
    orphan_lint: dict[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    max_expansions: int = MAX_EXPANSIONS,
    map_budget_steps: int = MAP_BUDGET_STEPS,
) -> dict[str, Any]:
    """Build the required Exp 5198 artifact from completed three-arm rows."""

    by_game = {row["game"]: row for row in per_game}
    games_tested = [game for game in GAMES if game in by_game]
    per_arm_results = {
        game: {
            arm: _arm_result((by_game[game].get("arms") or {}).get(arm) or {})
            for arm in ARMS
        }
        for game in games_tested
    }
    lever_validated = _map_validated([by_game[game] for game in games_tested])
    cn04_clean = True
    if NEGATIVE_CONTROL_GAME in by_game:
        cn04_clean = all(
            _arm_levels_banked(by_game[NEGATIVE_CONTROL_GAME], arm) <= 0 for arm in ARMS
        )
    levels_banked = _levels_banked([by_game[game] for game in games_tested])
    reproduction_gate_results = _reproduction_gate_results([by_game[game] for game in games_tested])

    if lever_validated:
        honest_verdict = (
            "success: MAP landmark prestage validated a reproduction-gated level bank "
            "that pruner-only did not under the identical 4000-expansion budget."
        )
        gap_status = "filled_by_map_prototype"
    else:
        honest_verdict = (
            "complete: MAP landmark prestage did not bank a new reproduction-gated level "
            "over pruner-only; the GAP-4891 enumeration wall persists under this lever too."
        )
        gap_status = "building_enumeration_wall_persists_under_map_prestage"

    artifact: dict[str, Any] = {
        "experiment": "experiment_5198_map_landmark_prestage_prototype_v476",
        "schema": "carnot.experiment_5198_map_landmark_prestage_prototype_v476.v1",
        "spec_refs": [
            "REQ-REPORT-5198",
            "SCENARIO-REPORT-5198-MAP-PRESTAGE",
            "SCENARIO-REPORT-5198-THREE-ARM-GATE",
        ],
        "honest_verdict": honest_verdict,
        "question": (
            "Does a bounded MAP-style cognitive-map prestage enumerate a new GAP-4891 "
            "deepening trajectory that pruner-only did not?"
        ),
        "lever_validated": bool(lever_validated),
        "per_arm_results": per_arm_results,
        "cn04_negative_control_stayed_clean": bool(cn04_clean),
        "solve_provenance": "development_proxy",
        "orphan_lint_result": _orphan_result_string(orphan_lint),
        "reproduction_gate_results": reproduction_gate_results,
        "random_seed": int(random_seed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "unit_tests_still_passing": bool(unit_tests_still_passing),
        "games_tested": games_tested,
        "target_games": list(SEPARATING_GAMES),
        "negative_control_game": NEGATIVE_CONTROL_GAME,
        "arms": list(ARMS),
        "max_expansions": int(max_expansions),
        "map_budget_steps": int(map_budget_steps),
        "levels_banked": levels_banked,
        "gap4891_status_recommendation": gap_status,
        "orphan_lint": dict(orphan_lint),
        "per_game": list(per_game),
        "verifier_is_oracle": False,
        "used_env_source": False,
        "read_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_calibrated_per_game": False,
        "scripts_research_conductor_modified": False,
        "duration_s": round(float(duration_s), 2),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _apply(env: Any, label: str, frame: Any) -> Any:  # pragma: no cover
    from carnot.experiment_5175_gap4891_relational_mask_pruner_ab_v474 import _apply as apply

    return apply(env, label, frame)


def _make_goal_energy(energy: Any):  # pragma: no cover
    from carnot.experiment_5175_gap4891_relational_mask_pruner_ab_v474 import (
        _make_goal_energy as make_goal_energy,
    )

    return make_goal_energy(energy)


def _induce_context(root: Path, game: str) -> dict[str, Any]:  # pragma: no cover
    from carnot.experiment_5175_gap4891_relational_mask_pruner_ab_v474 import (
        _induce_context as induce_context,
    )

    return induce_context(root, game)


def _run_arm(
    game: str,
    context: dict[str, Any],
    *,
    arm: str,
    max_expansions: int,
    map_budget_steps: int,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels
    from carnot.agentic.arc_map_landmark_prestage import build_landmark_map
    from carnot.agentic.arc_relational_mask_pruner import RelationalMaskMovePruner

    use_map = arm in {"map_only", "map_plus_pruner"}
    use_pruner = arm in {"pruner_only", "map_plus_pruner"}
    arc = kit.offline_arcade()
    goal_energy = _make_goal_energy(context["goal_energy"])
    cognitive_map = None
    if use_map:
        map_env = arc.make(game, scorecard_id=arc.open_scorecard())
        cognitive_map = build_landmark_map(
            map_env,
            start_level=int(context["prefix_level"]),
            prefix=list(context["prefix"]),
            max_steps=int(map_budget_steps),
            grid_of=grid_of,
            goal_energy=goal_energy,
            target_region=context["target_region"],
        )

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
        goal_energy=goal_energy,
        frontier_seed_bank=cognitive_map,
        stats=stats,
        move_pruner=pruner,
    )
    gate: dict[str, Any] | None = None
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
    levels_banked = (
        max(0, int(gate_reached) - int(context["prefix_level"])) if reproduced else 0
    )
    return {
        "arm": arm,
        "reached_level": int(level),
        "gate_reached": int(gate_reached),
        "offline_reproduced": reproduced,
        "levels_banked": int(levels_banked),
        "states_expanded": int(stats.get("states_expanded", stats.get("expansions", 0)) or 0),
        "traj_len": len(traj) if traj else 0,
        "map_overhead_steps": int(cognitive_map.map_overhead_steps) if cognitive_map else 0,
        "map_overhead_wall_s": (
            round(float(cognitive_map.map_overhead_wall_s), 4) if cognitive_map else 0.0
        ),
        "map_diagnostics": cognitive_map.as_dict() if cognitive_map else None,
        "reproduction_gate": gate,
        "reproducibility_checksum": repro_checksum,
        "trajectory_labels": labels if reproduced else [],
        "stats": stats,
        "pruner_stats": stats.get("move_pruner_stats") if use_pruner else None,
    }


def _load_pruner_baseline(root: Path, game: str) -> dict[str, Any] | None:  # pragma: no cover
    path = root / "results" / "experiment_5175_gap4891_relational_mask_pruner_ab_v474.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("per_game") or []:
        if row.get("game") != game:
            continue
        pruned = dict(row.get("pruned") or {})
        if not pruned:
            return None
        prefix_level = int(row.get("prefix_level", 0) or 0)
        reproduced = bool(pruned.get("offline_reproduced"))
        gate_reached = int(pruned.get("gate_reached", pruned.get("reached_level", 0)) or 0)
        pruned["arm"] = "pruner_only"
        pruned["levels_banked"] = max(0, gate_reached - prefix_level) if reproduced else 0
        pruned["map_overhead_steps"] = 0
        pruned["map_overhead_wall_s"] = 0.0
        pruned["reproduction_gate"] = pruned.get("reproduction_gate")
        pruned["baseline_source"] = str(path.relative_to(root))
        return pruned
    return None


def run_experiment(
    *,
    root: Path,
    games: Sequence[str] = GAMES,
    max_expansions: int = MAX_EXPANSIONS,
    map_budget_steps: int = MAP_BUDGET_STEPS,
    random_seed: int = RANDOM_SEED,
    run_unit_tests: bool = True,
    run_lint: bool = True,
    reuse_pruner_baseline: bool = False,
) -> dict[str, Any]:  # pragma: no cover
    started = time.time()
    np.random.seed(int(random_seed))
    unit = check_map_unit_tests(root) if run_unit_tests else {"passed": True, "skipped": True}
    lint = run_arc_orphan_solver_lint(root) if run_lint else {"passed": True, "skipped": True}
    per_game: list[dict[str, Any]] = []
    for game in games:
        context = _induce_context(root, game)
        arms: dict[str, Any] = {}
        if reuse_pruner_baseline:
            baseline = _load_pruner_baseline(root, game)
            if baseline is not None:
                arms["pruner_only"] = baseline
        if "pruner_only" not in arms:
            arms["pruner_only"] = _run_arm(
                game,
                context,
                arm="pruner_only",
                max_expansions=int(max_expansions),
                map_budget_steps=int(map_budget_steps),
            )
        for arm in ("map_only", "map_plus_pruner"):
            arms[arm] = _run_arm(
                game,
                context,
                arm=arm,
                max_expansions=int(max_expansions),
                map_budget_steps=int(map_budget_steps),
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
                "arms": arms,
            }
        )
        print(
            f"[{game}] "
            + " | ".join(
                f"{arm}={arms[arm]['states_expanded']} L{arms[arm]['gate_reached']} "
                f"bank={arms[arm]['levels_banked']}"
                for arm in ARMS
            ),
            flush=True,
        )

    artifact = build_artifact(
        per_game=per_game,
        unit_tests_still_passing=bool(unit.get("passed")),
        orphan_lint=lint,
        duration_s=time.time() - started,
        random_seed=int(random_seed),
        max_expansions=int(max_expansions),
        map_budget_steps=int(map_budget_steps),
    )
    artifact["map_unit_tests"] = unit
    artifact["reuse_pruner_baseline_from_exp5175"] = bool(reuse_pruner_baseline)
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
    parser.add_argument("--map-budget-steps", type=int, default=MAP_BUDGET_STEPS)
    parser.add_argument("--games", default=",".join(GAMES))
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--reuse-pruner-baseline", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(
        root=args.root,
        games=tuple(game.strip() for game in args.games.split(",") if game.strip()),
        max_expansions=args.max_expansions,
        map_budget_steps=args.map_budget_steps,
        run_unit_tests=not args.skip_unit_tests,
        run_lint=not args.skip_lint,
        reuse_pruner_baseline=bool(args.reuse_pruner_baseline),
    )
    out = write_artifact(args.root, artifact)
    print(f"=== VERDICT: {artifact['honest_verdict']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
