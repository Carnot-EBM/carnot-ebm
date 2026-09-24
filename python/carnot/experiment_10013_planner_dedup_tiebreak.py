"""CPU-only Experiment 10013 planner HUD-deduplication/tie-break measurement.

REQ-ARC-WMTE-10013 reuses Experiment 10012's frozen candidates, simulator
rebuilders, and LIVE_SCORED execution semantics. It changes only planner arm
flags and records the resulting search, real-action, and cost evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from carnot import experiment_10012_gate_usefulness as exp12
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of

EXPERIMENT_ID = 10013
REQUIREMENT_ID = "REQ-ARC-WMTE-10013"
SCHEMA = "carnot.experiment_10013_planner_dedup_tiebreak.v1"
RANDOM_SEED = 10013
SOURCE_EXPERIMENT_ID = exp12.EXPERIMENT_ID
LIVE_EXECUTION_ARM = exp12.LIVE_SCORED
INFERENCE_SUBSTRATE = exp12.INFERENCE_SUBSTRATE
SOLVE_PROVENANCE = exp12.SOLVE_PROVENANCE
ARTIFACT_REL = Path("results/experiment_10013_planner_dedup_tiebreak.json")
RAW_REL = Path("results/raw/experiment_10013_planner_dedup_tiebreak")
_PLANNER_ENV_KEYS = (
    "CARNOT_ARC_PLAN_HUD_DEDUP",
    "CARNOT_ARC_PLAN_GOAL_TIEBREAK",
)


@dataclass(frozen=True)
class PlannerArm:
    """REQ-ARC-WMTE-10013 immutable planner-arm definition."""

    name: str
    environment: Mapping[str, str]
    max_nodes: int = exp12.PLAN_MAX_NODES
    max_depth: int = exp12.PLAN_MAX_DEPTH


PLANNER_ARMS = (
    PlannerArm("OFF", {}),
    PlannerArm("HUD_DEDUP", {"CARNOT_ARC_PLAN_HUD_DEDUP": "1"}),
    PlannerArm(
        "HUD_DEDUP+TIEBREAK",
        {
            "CARNOT_ARC_PLAN_HUD_DEDUP": "1",
            "CARNOT_ARC_PLAN_GOAL_TIEBREAK": "novelty",
        },
    ),
)


def progress(message: str) -> None:
    """REQ-ARC-WMTE-10013 emits bounded-run phase and heartbeat progress."""

    print(f"[exp{EXPERIMENT_ID} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def canonical_checksum(value: Mapping[str, Any]) -> str:
    """SCENARIO-ARC-WMTE-10013-ARTIFACT hashes canonical artifact content."""

    return exp12.canonical_checksum(value)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """SCENARIO-ARC-WMTE-10013-ARTIFACT publishes one JSON value atomically."""

    exp12.atomic_json(path, value)


@contextlib.contextmanager
def planner_arm_environment(arm: PlannerArm) -> Iterator[None]:
    """SCENARIO-ARC-WMTE-10013-FLAGS-OFF-IDENTITY isolates arm flags."""

    prior = {key: os.environ.get(key) for key in _PLANNER_ENV_KEYS}
    for key in _PLANNER_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(arm.environment)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_measurement_mask(
    frame: Any,
    cell: int,
    transitions: Sequence[Any],
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """SCENARIO-ARC-WMTE-10013-HUD-DEDUP-SAFETY resolves and guards a mask."""

    frame_mask = agent._compute_hud_mask_from_frame(frame, edge_bar_detector=True)
    logical_mask = e3.logical_hud_mask(frame_mask, cell)
    if logical_mask is None:
        return None, {
            "status": "unresolved",
            "reason": "production_edge_bar_mask_unresolved",
            "swallow_check": None,
        }
    swallow = e3.hud_mask_swallow_check(transitions, logical_mask)
    clean = e3.hud_mask_swallow_clean(swallow)
    return (
        logical_mask if clean else None,
        {
            "status": "applied" if clean else "refused",
            "reason": "clean" if clean else str(swallow.get("reason")),
            "mask_cells": int(np.count_nonzero(logical_mask)),
            "mask_rows": np.flatnonzero(np.asarray(logical_mask).any(axis=1)).tolist(),
            "swallow_check": swallow,
        },
    )


def _harm_cohorts(row: Mapping[str, Any]) -> list[str]:
    if row.get("provenance_cohort") == "stall_window":
        return ["main_candidates"]
    cohorts = ["h2h_replay_counterfactual"]
    if row.get("control_category") == "informative_by_registry_solver":
        cohorts.append("registry_solver_controlled")
    elif row.get("control_category") == "expert_live_planner":
        cohorts.append("h2h_expert_controlled")
    return cohorts


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS reduces frozen guards."""

    materialized = [dict(row) for row in rows]
    arm_names = [arm.name for arm in PLANNER_ARMS]
    primary: dict[str, Any] = {}
    for arm_name in arm_names:
        experts = [
            row
            for row in materialized
            if row.get("arm") == arm_name
            and row.get("provenance_cohort") == "stall_window"
            and row.get("engine_family") == "EXPERT"
        ]
        named_checks = {}
        for game in ("dc22", "wa30", "sb26"):
            match = next((row for row in experts if row.get("game") == game), None)
            named_checks[game] = {
                "level_up": bool(match and match.get("real_level_up")),
                "planner_engine_calls": (
                    int(match.get("planner_engine_calls") or 0) if match is not None else None
                ),
            }
        primary[arm_name] = {
            "expert_control_count": len(experts),
            "expert_win_count": sum(bool(row.get("real_level_up")) for row in experts),
            "planner_engine_calls_used": sum(
                int(row.get("planner_engine_calls") or 0) for row in experts
            ),
            "expert_controls": [
                {
                    "pair_id": row.get("pair_id"),
                    "game": row.get("game"),
                    "level_up": bool(row.get("real_level_up")),
                    "planner_engine_calls": int(row.get("planner_engine_calls") or 0),
                }
                for row in experts
            ],
            "named_checks": named_checks,
        }

    indexed = {
        (str(row.get("pair_id")), str(row.get("provenance_cohort")), str(row.get("arm"))): row
        for row in materialized
    }
    regressions = []
    off_wins = [row for row in materialized if row.get("arm") == "OFF" and row.get("real_level_up")]
    for off_row in off_wins:
        pair_id = str(off_row.get("pair_id"))
        cohort = str(off_row.get("provenance_cohort"))
        for arm_name in arm_names[1:]:
            on_row = indexed.get((pair_id, cohort, arm_name))
            if on_row is None or not on_row.get("real_level_up"):
                regressions.append(
                    {
                        "pair_id": pair_id,
                        "game": off_row.get("game"),
                        "provenance_cohort": cohort,
                        "arm": arm_name,
                        "on_row_missing": on_row is None,
                    }
                )

    harm: dict[str, dict[str, Any]] = {}
    for row in materialized:
        if row.get("is_control"):
            continue
        for cohort in _harm_cohorts(row):
            bucket = harm.setdefault(cohort, {}).setdefault(
                str(row.get("arm")),
                {"pair_count": 0, "useful_count": 0, "wasted_real_actions": 0},
            )
            bucket["pair_count"] += 1
            if row.get("real_level_up"):
                bucket["useful_count"] += 1
            else:
                bucket["wasted_real_actions"] += int(row.get("real_actions_used") or 0)

    cost: dict[str, Any] = {}
    for arm_name in arm_names:
        arm_rows = [row for row in materialized if row.get("arm") == arm_name]
        wall = [float(row.get("planner_wall_s") or 0.0) for row in arm_rows]
        cost[arm_name] = {
            "planner_call_count": len(arm_rows),
            "total_wall_s": round(sum(wall), 6),
            "mean_wall_s_per_call": round(sum(wall) / len(wall), 6) if wall else None,
        }
    return {
        "primary": primary,
        "guard_1_regressions": regressions,
        "guard_1_passed": not regressions,
        "guard_2_harm": harm,
        "guard_3_cost": cost,
    }


def build_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-ARTIFACT builds the terminal measurement."""

    checks = [dict(row) for row in preconditions_checked]
    failed = [row.get("resource") for row in checks if not row.get("available")]
    reductions = reduce_rows(rows)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "requirement_id": REQUIREMENT_ID,
        "honest_verdict": (
            "blocked_precondition_failed" if failed else "complete_planner_dedup_tiebreak_measured"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": checks,
        "failed_preconditions": failed,
        "duration_s": round(float(duration_s), 6),
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "live_execution_semantics": LIVE_EXECUTION_ARM.name,
        "planner_arms": [
            {
                "name": arm.name,
                "environment": dict(arm.environment),
                "max_nodes": arm.max_nodes,
                "max_depth": arm.max_depth,
            }
            for arm in PLANNER_ARMS
        ],
        "per_pair_arm_rows": [dict(row) for row in rows],
        **reductions,
        "limits": [
            "Public offline environments are a development proxy, not hidden-game efficacy.",
            "Cached generated engines are fixed historical draws; no model call was made.",
            "H2H winning-route replay cohorts are appendix evidence and not primary stalls.",
            "No live default, planner budget, candidate order, or goal predicate was changed.",
        ],
    }
    artifact["reproducibility_checksum"] = canonical_checksum(artifact)
    return artifact


def plan_and_execute(
    candidate: exp12.Candidate,
    rebuilt: exp12.RebuiltState,
    *,
    arm: PlannerArm,
    cell: int,
    dedup_mask: Optional[np.ndarray],
    mask_record: Mapping[str, Any],
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS runs LIVE_SCORED."""

    from arcengine import GameAction
    from carnot.agentic.arc_engine_call_guard import EngineCallGuardError, guarded_call

    namespace, load_error = exp12.load_engine_namespace(candidate)
    base: dict[str, Any] = {
        "plan_found": False,
        "plan_length": 0,
        "real_level_up": False,
        "real_actions_used": 0,
        "real_actions_spent_after_first_divergence": 0,
        "matched_steps": 0,
        "matched_steps_before_divergence": 0,
        "first_divergence_index": None,
        "divergence_reason": None,
        "executed_actions": [],
        "execution_termination_reason": "no_plan",
        "planner_engine_calls": 0,
        "planner_wall_s": 0.0,
        "planner_diagnostics": {},
        "mask": dict(mask_record),
    }
    if namespace is None:
        base["planner_error"] = load_error
        return base
    engine = namespace["engine"]
    is_done = namespace["is_level_complete"]

    def binary_goal_energy(grid: np.ndarray) -> float:
        try:
            return 0.0 if bool(is_done(grid)) else exp12.GOAL_GUIDANCE_LAMBDA
        except Exception:
            return exp12.GOAL_GUIDANCE_LAMBDA

    binary_goal_energy.energy_source = "binary"  # type: ignore[attr-defined]
    diagnostics: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        with exp12.Heartbeat(f"exp10013 planning {candidate.pair_id} {arm.name}"):
            plan = e3.plan_in_model(
                engine,
                is_done,
                rebuilt.grid.copy(),
                max_nodes=arm.max_nodes,
                max_depth=arm.max_depth,
                goal_energy=binary_goal_energy,
                diagnostics=diagnostics,
                dedup_mask=dedup_mask,
            )
    except BaseException as exc:
        base["planner_error"] = f"{type(exc).__name__}: {exc}"[:300]
        plan = None
    base["planner_wall_s"] = round(time.perf_counter() - started, 6)
    base["planner_diagnostics"] = diagnostics
    base["planner_engine_calls"] = int(diagnostics.get("nodes_expanded") or 0)
    if not plan:
        return base

    base["plan_found"] = True
    base["plan_length"] = len(plan)
    base["execution_termination_reason"] = "plan_exhausted"
    current = rebuilt.grid.copy()
    for plan_index, step in enumerate(plan):
        action = int(step["action"])
        data = step.get("data")
        predicted: Optional[np.ndarray] = None
        prediction_error: Optional[str] = None
        try:
            predicted = np.asarray(guarded_call(engine, current.copy(), action, data))
        except EngineCallGuardError as exc:
            prediction_error = f"engine_guard:{type(exc).__name__}:{exc}"[:240]
        except BaseException as exc:
            prediction_error = f"engine_raised:{type(exc).__name__}:{exc}"[:240]
        frame = rebuilt.env.step(exp12._game_action(GameAction, action), data=data)
        if frame is None:
            base["first_divergence_index"] = plan_index
            base["divergence_reason"] = "environment_returned_none"
            base["execution_termination_reason"] = "environment_returned_none"
            break
        observed = e3.to_logical(grid_of(frame), cell)
        base["real_actions_used"] += 1
        base["executed_actions"].append(
            {
                "plan_index": plan_index,
                "action": action,
                "data": exp12._jsonable_data(data),
            }
        )
        matched = bool(
            predicted is not None
            and predicted.shape == observed.shape
            and np.array_equal(predicted, observed)
        )
        if matched:
            base["matched_steps"] += 1
            if base["first_divergence_index"] is None:
                base["matched_steps_before_divergence"] += 1
        elif base["first_divergence_index"] is None:
            base["first_divergence_index"] = plan_index
            if prediction_error is not None:
                base["divergence_reason"] = prediction_error
            elif predicted is not None and predicted.shape != observed.shape:
                base["divergence_reason"] = "prediction_shape_mismatch"
            else:
                base["divergence_reason"] = "prediction_cell_mismatch"
        current = observed
        if exp12._levels_completed(frame) > rebuilt.level:
            base["real_level_up"] = True
            base["level_after"] = exp12._levels_completed(frame)
            base["execution_termination_reason"] = "real_level_up_boundary"
            break
    first = base["first_divergence_index"]
    if first is not None:
        base["real_actions_spent_after_first_divergence"] = max(
            0, int(base["real_actions_used"]) - int(first) - 1
        )
    return base


def _pair_arm_row(
    candidate: exp12.Candidate,
    arm: PlannerArm,
    execution: Mapping[str, Any],
    *,
    provenance_cohort: str,
    control_category: str,
    state_rebuild: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pair_id": candidate.pair_id,
        "game": candidate.game,
        "engine_family": candidate.engine_family,
        "variant": candidate.variant,
        "model_family": candidate.model_family,
        "think_mode": candidate.think_mode,
        "token_budget": candidate.token_budget,
        "is_control": candidate.is_control,
        "source_status": candidate.source_status,
        "source_path": candidate.source_path,
        "source_sha256": candidate.source_sha256,
        "provenance_cohort": provenance_cohort,
        "control_category": control_category,
        "arm": arm.name,
        "state_rebuild": dict(state_rebuild),
        **dict(execution),
    }


def _mask_for_arm(
    arm: PlannerArm,
    rebuilt: exp12.RebuiltState,
    cell: int,
    transitions: Sequence[Any],
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    if "CARNOT_ARC_PLAN_HUD_DEDUP" not in arm.environment:
        return None, {"status": "disabled", "reason": "arm_flag_off", "swallow_check": None}
    return resolve_measurement_mask(rebuilt.frame, cell, transitions)


def _state_record(rebuilt: exp12.RebuiltState) -> dict[str, Any]:
    return {
        "recoverable": rebuilt.recoverable,
        "reason": rebuilt.reason,
        "mode": "root_after_reset",
        "reset_sent": True,
        "level_from_frame": rebuilt.level,
        "actions_replayed": rebuilt.actions_replayed,
        "expected_sha256": rebuilt.expected_sha256,
        "observed_sha256": rebuilt.observed_sha256,
    }


def _unrecoverable_execution(reason: Optional[str]) -> dict[str, Any]:
    return {
        "plan_found": False,
        "plan_length": 0,
        "real_level_up": False,
        "real_actions_used": 0,
        "real_actions_spent_after_first_divergence": 0,
        "planner_engine_calls": 0,
        "planner_wall_s": 0.0,
        "planner_diagnostics": {},
        "execution_termination_reason": "state_unrecoverable",
        "excluded_reason": reason or "state_unrecoverable",
    }


def _preconditions(repo_root: Path) -> list[dict[str, Any]]:
    checks = exp12._preconditions(repo_root)
    checks.extend(
        [
            {
                "resource": "python/carnot/experiment_10012_gate_usefulness.py",
                "available": (
                    repo_root / "python/carnot/experiment_10012_gate_usefulness.py"
                ).is_file(),
            },
            {
                "resource": "planner live budget remains 20000",
                "available": exp12.LIVE_SCORED.max_nodes == 20_000,
            },
            {
                "resource": "planner live depth remains 80",
                "available": exp12.LIVE_SCORED.max_depth == 80,
            },
        ]
    )
    return checks


def run_experiment(repo_root: Path, output_path: Path, raw_dir: Path) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS runs all frozen rows."""

    started = time.monotonic()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    checks = _preconditions(repo_root)
    if any(not row["available"] for row in checks):
        artifact = build_artifact(
            [], preconditions_checked=checks, duration_s=time.monotonic() - started
        )
        atomic_json(output_path, artifact)
        progress(f"blocked preconditions: {artifact['failed_preconditions']}")
        return artifact

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = exp12.exp10.EvidencePaths.under(repo_root)
    reports = exp12.exp10.load_control_reports(paths)
    progress("phase main: load Experiment 10012 stall windows and candidates")
    specs = {
        game: exp12.exp10.load_window(game, index, reports[game], paths)
        for index, game in enumerate(exp12.WINDOWS)
    }
    candidates = exp12.load_candidates(repo_root, paths)
    rows: list[dict[str, Any]] = []

    for game in exp12.WINDOWS:
        spec = specs[game]
        actions = exp12.load_episode_actions(repo_root, game)
        alignment = exp12.induction_alignment(repo_root, game, actions)
        progress(f"phase main window {game}: {len(candidates[game])} pairs x 3 arms")
        for candidate in candidates[game]:
            for arm in PLANNER_ARMS:
                progress(f"phase pair {candidate.pair_id} arm {arm.name}: reset and plan")
                rebuilt, evidence = exp12.rebuild_arm_start_state(
                    repo_root,
                    game,
                    spec,
                    actions,
                    alignment,
                    exp12.LIVE_SCORED,
                )
                if evidence.get("mode") != "root_after_reset":
                    rebuilt = exp12.rebuild_reset_state_from_env(
                        env=rebuilt.env,
                        expected_level=rebuilt.level,
                        grid_reader=lambda frame, cell=spec.cell: e3.to_logical(
                            grid_of(frame), cell
                        ),
                        expected_grid=np.asarray(spec.rows[0].grid),
                    )
                    evidence = _state_record(rebuilt)
                with planner_arm_environment(arm):
                    mask, mask_record = _mask_for_arm(arm, rebuilt, spec.cell, spec.rows)
                    execution = (
                        plan_and_execute(
                            candidate,
                            rebuilt,
                            arm=arm,
                            cell=spec.cell,
                            dedup_mask=mask,
                            mask_record=mask_record,
                        )
                        if rebuilt.recoverable
                        else _unrecoverable_execution(rebuilt.reason)
                    )
                row = _pair_arm_row(
                    candidate,
                    arm,
                    execution,
                    provenance_cohort="stall_window",
                    control_category="expert_live_planner",
                    state_rebuild=evidence,
                )
                rows.append(row)
                atomic_json(raw_dir / f"{candidate.pair_id}__{arm.name}.json", row)
                progress(
                    f"phase pair {candidate.pair_id} arm {arm.name}: "
                    f"win={row['real_level_up']} calls={row['planner_engine_calls']}"
                )

    progress("phase appendix: deterministic h2h counterfactual rebuild")
    stored_windows = {
        game: exp12.rebuild_registered_window(repo_root, game, len(exp12.WINDOWS) + index)
        for index, game in enumerate(exp12.ADDED_WINDOWS)
    }
    stored_candidates, qualifications = exp12.load_stored_candidates(
        repo_root, paths, stored_windows
    )
    for game in exp12.ADDED_WINDOWS:
        window = stored_windows[game]
        category = exp12.control_category(has_expert=game in exp12.ADDED_EXPERT_GAMES)
        progress(f"phase appendix window {game}: {len(stored_candidates[game])} pairs x 3 arms")
        for candidate in stored_candidates[game]:
            for arm in PLANNER_ARMS:
                progress(f"phase appendix pair {candidate.pair_id} arm {arm.name}: reset and plan")
                rebuilt, evidence = exp12.rebuild_registered_arm_state(
                    repo_root, game, window, exp12.LIVE_SCORED
                )
                with planner_arm_environment(arm):
                    mask, mask_record = _mask_for_arm(
                        arm, rebuilt, window.spec.cell, window.spec.rows
                    )
                    execution = (
                        plan_and_execute(
                            candidate,
                            rebuilt,
                            arm=arm,
                            cell=window.spec.cell,
                            dedup_mask=mask,
                            mask_record=mask_record,
                        )
                        if rebuilt.recoverable
                        else _unrecoverable_execution(rebuilt.reason)
                    )
                row = _pair_arm_row(
                    candidate,
                    arm,
                    execution,
                    provenance_cohort="h2h_replay_counterfactual",
                    control_category=category,
                    state_rebuild=evidence,
                )
                rows.append(row)
                atomic_json(raw_dir / f"{candidate.pair_id}__{arm.name}.json", row)
                progress(
                    f"phase appendix pair {candidate.pair_id} arm {arm.name}: "
                    f"win={row['real_level_up']} calls={row['planner_engine_calls']}"
                )

    artifact = build_artifact(
        rows,
        preconditions_checked=checks,
        duration_s=time.monotonic() - started,
    )
    artifact["stored_pair_qualification"] = qualifications
    identity_wins = [
        {"pair_id": row["pair_id"], "arm": row["arm"]}
        for row in rows
        if row.get("engine_family") == "IDENTITY" and row.get("real_level_up")
    ]
    artifact["identity_useful_harness_bugs"] = identity_wins
    if identity_wins:
        artifact["honest_verdict"] = "blocked_identity_useful_harness_bug"
    artifact["reproducibility_checksum"] = canonical_checksum(artifact)
    atomic_json(output_path, artifact)
    progress(f"phase artifact: wrote {output_path}")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    """REQ-ARC-WMTE-10013 exposes only repository and artifact locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """SCENARIO-ARC-WMTE-10013-ARTIFACT runs and reports the terminal verdict."""

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / ARTIFACT_REL
    raw_dir = args.raw_dir or repo_root / RAW_REL
    artifact = run_experiment(repo_root, output, raw_dir)
    print(json.dumps({"honest_verdict": artifact["honest_verdict"], "output": str(output)}))
    return 0 if str(artifact["honest_verdict"]).startswith("complete_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
