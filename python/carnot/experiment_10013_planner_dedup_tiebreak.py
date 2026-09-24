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
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from carnot import experiment_10012_gate_usefulness as exp12
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of

EXPERIMENT_ID = 10013
REQUIREMENT_ID = "REQ-ARC-WMTE-10013"
SCHEMA = "carnot.experiment_10013_planner_dedup_tiebreak.v2"
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


class ScoredPlannerPolicyStub(agent.E3AgentPolicy):
    """SCENARIO-ARC-WMTE-10013-AM1-SCORED-PATH supplies wrapper-only state."""

    def __init__(self, frame_mask: Any, *, cell: int, transitions: Sequence[Any]) -> None:
        self.two_sided_goal_contract = None
        self.explorer = SimpleNamespace(hud_mask=frame_mask)
        self.cell = int(cell)
        self.transitions = list(transitions)
        self._episode_transition_start = 0


def _recorded_action(row: Mapping[str, Any]) -> tuple[int, Any]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY parses replay labels."""

    raw = row.get("action")
    data = row.get("data") or None
    if isinstance(raw, str) and raw.strip().startswith("{"):
        payload = json.loads(raw)
        raw = payload.get("action")
        data = payload.get("data") or data
    if isinstance(raw, str) and raw.upper() == "RESET":
        return 0, data
    if isinstance(raw, str) and raw.upper().startswith("ACTION"):
        return int(raw[6:]), data
    return int(raw), data


def replay_live_explorer_mask(
    frames: Sequence[Any], action_rows: Sequence[Mapping[str, Any]]
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY replays production Stage 2."""

    if len(frames) != len(action_rows):
        raise ValueError("live mask replay requires one resulting frame per action row")
    explorer = agent.StepwiseExplorer()
    for frame, row in zip(frames, action_rows):
        action_id, data = _recorded_action(row)
        if action_id == 0:
            explorer.awaiting = None
        else:
            origin = explorer.cur
            node = explorer.graph.get(origin, {}) if origin is not None else {}
            explorer.awaiting = {
                "origin": origin,
                "action": action_id,
                "data": data,
                "grid": explorer._grid_for_hash(origin),
                "level_before": int(explorer.best_level),
                "previous_frame": node.get("frame"),
            }
        explorer._ingest(frame)
    diagnostics = explorer.hud_mask_diagnostics()
    mask = None if explorer.hud_mask is None else np.asarray(explorer.hud_mask, dtype=bool)
    return mask, {
        "status": "resolved" if mask is not None else "unresolved",
        "mask_cells": int(0 if mask is None else np.count_nonzero(mask)),
        "mask_rows": (
            [] if mask is None else np.flatnonzero(np.asarray(mask).any(axis=1)).tolist()
        ),
        "hud_mask_source": diagnostics["hud_mask_source"],
        "stage2": diagnostics["stage2"],
    }


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


def replay_main_window_live_mask(
    repo_root: Path,
    game: str,
    action_rows: Sequence[Mapping[str, Any]],
    induction_action_index: int,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY rebuilds a live stall trace."""

    from arcengine import GameAction

    exp12.kit.ENV_DIR = exp12.resolve_environment_files(repo_root)
    arcade = exp12.kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    replayed_rows: list[Mapping[str, Any]] = []
    for row in action_rows:
        if int(row["action_index"]) >= int(induction_action_index):
            break
        name = str(row["action"])
        data = row.get("data") or None
        frame = env.reset() if name == "RESET" else env.step(GameAction[name], data=data)
        frames.append(frame)
        replayed_rows.append(row)
    mask, record = replay_live_explorer_mask(frames, replayed_rows)
    record["frames_replayed"] = len(frames)
    record["induction_action_index"] = int(induction_action_index)
    return mask, record


def _registered_label_row(adapter: Any, env: Any, label: str) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY resolves stored labels."""

    payload = json.loads(label) if label.strip().startswith("{") else {"action": label}
    raw_action = payload.get("action")
    data = payload.get("data")
    if raw_action is None and {"x", "y"} <= payload.keys():
        raw_action = 6
        data = {"x": payload["x"], "y": payload["y"]}
    resolver = getattr(adapter, "label_to_action_data", None)
    if resolver is not None and isinstance(raw_action, str):
        raw_action, data = resolver(env, raw_action)
    return {"action": int(raw_action), "data": data}


def replay_registered_window_live_mask(
    repo_root: Path, game: str, labels: Sequence[str]
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY rebuilds an appendix trace."""

    from carnot.agentic import arc_game_adapters as adapters

    exp12.kit.ENV_DIR = exp12.resolve_environment_files(repo_root)
    adapter = adapters.get_adapter(game)
    if adapter is None:
        raise ValueError(f"no registered adapter for {game}")
    arcade = exp12.kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames = [env.reset()]
    rows: list[Mapping[str, Any]] = [{"action": "RESET", "data": None}]
    current = frames[0]
    if adapter.warmup_label is not None:
        current = adapter.apply(env, adapter.warmup_label, current)
        frames.append(current)
        rows.append(_registered_label_row(adapter, env, adapter.warmup_label))
    for label in labels:
        current = adapter.apply(env, label, current)
        frames.append(current)
        rows.append(_registered_label_row(adapter, env, label))
    mask, record = replay_live_explorer_mask(frames, rows)
    record["frames_replayed"] = len(frames)
    record["registered_actions_replayed"] = len(labels)
    return mask, record


def _mask_digest(mask: Optional[np.ndarray]) -> Optional[str]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY hashes a mask cell set."""

    if mask is None:
        return None
    array = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return exp12.sha256_bytes(array.tobytes() + json.dumps(list(array.shape)).encode())


def live_mask_window_record(
    *,
    window_id: str,
    cohort: str,
    frame_mask: Optional[np.ndarray],
    replay_record: Mapping[str, Any],
    planner_frame: Any,
    cell: int,
    transitions: Sequence[Any],
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-AM1-LIVE-MASK-REPLAY compares v1 and live masks."""

    live_logical = e3.logical_hud_mask(frame_mask, cell)
    harness_candidate = e3.logical_hud_mask(
        agent._compute_hud_mask_from_frame(planner_frame, edge_bar_detector=True), cell
    )
    harness_mask, harness_record = resolve_measurement_mask(planner_frame, cell, transitions)
    same = (
        live_logical is None
        and harness_mask is None
        or live_logical is not None
        and harness_mask is not None
        and np.array_equal(live_logical, harness_mask)
    )
    stage2 = replay_record.get("stage2")
    return {
        "window_id": window_id,
        "provenance_cohort": cohort,
        "game": window_id.split(":", 1)[-1],
        "live_mask_size": int(0 if live_logical is None else np.count_nonzero(live_logical)),
        "live_mask_digest": _mask_digest(live_logical),
        "live_frame_mask_size": int(replay_record.get("mask_cells") or 0),
        "live_mask_source": replay_record.get("hud_mask_source"),
        "stage2_status": (
            stage2.get("stage2_verdict") if isinstance(stage2, Mapping) else "not_armed"
        ),
        "stage2": stage2,
        "frames_replayed": int(replay_record.get("frames_replayed") or 0),
        "harness_mask_size": int(0 if harness_mask is None else np.count_nonzero(harness_mask)),
        "harness_mask_digest": _mask_digest(harness_mask),
        "harness_candidate_mask_size": int(
            0 if harness_candidate is None else np.count_nonzero(harness_candidate)
        ),
        "harness_mask_status": harness_record["status"],
        "harness_mask_reason": harness_record["reason"],
        "differs_from_10013_harness_mask": not same,
    }


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
        for game in ("dc22", "wa30", "sb26", "ar25"):
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
    harm_with_controls: dict[str, dict[str, Any]] = {}
    for row in materialized:
        for cohort in _harm_cohorts(row):
            inclusive = harm_with_controls.setdefault(cohort, {}).setdefault(
                str(row.get("arm")),
                {
                    "pair_count": 0,
                    "useful_count": 0,
                    "wasted_real_actions": 0,
                    "candidate_pair_count": 0,
                    "candidate_useful_count": 0,
                    "candidate_wasted_real_actions": 0,
                    "control_pair_count": 0,
                    "control_useful_count": 0,
                    "control_wasted_real_actions": 0,
                },
            )
            inclusive["pair_count"] += 1
            kind = "control" if row.get("is_control") else "candidate"
            inclusive[f"{kind}_pair_count"] += 1
            if row.get("real_level_up"):
                inclusive["useful_count"] += 1
                inclusive[f"{kind}_useful_count"] += 1
            else:
                actions = int(row.get("real_actions_used") or 0)
                inclusive["wasted_real_actions"] += actions
                inclusive[f"{kind}_wasted_real_actions"] += actions
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
            "worst_wall_s_per_call": round(max(wall), 6) if wall else None,
            "worst_pair_id": (
                max(arm_rows, key=lambda row: float(row.get("planner_wall_s") or 0.0)).get(
                    "pair_id"
                )
                if arm_rows
                else None
            ),
        }
    off_winner_efficiency = []
    for off_row in off_wins:
        pair_id = str(off_row.get("pair_id"))
        cohort = str(off_row.get("provenance_cohort"))
        off_winner_efficiency.append(
            {
                "pair_id": pair_id,
                "game": off_row.get("game"),
                "provenance_cohort": cohort,
                "arms": {
                    arm_name: {
                        "plan_length": int(
                            (indexed.get((pair_id, cohort, arm_name)) or {}).get("plan_length") or 0
                        ),
                        "real_actions_used": int(
                            (indexed.get((pair_id, cohort, arm_name)) or {}).get(
                                "real_actions_used"
                            )
                            or 0
                        ),
                        "real_level_up": bool(
                            (indexed.get((pair_id, cohort, arm_name)) or {}).get("real_level_up")
                        ),
                    }
                    for arm_name in arm_names
                },
            }
        )
    window_cost: dict[str, dict[str, Any]] = {}
    for row in materialized:
        window_id = f"{row.get('provenance_cohort')}:{row.get('game')}"
        arm_name = str(row.get("arm"))
        bucket = window_cost.setdefault(window_id, {}).setdefault(arm_name, {"calls": []})
        bucket["calls"].append(float(row.get("planner_wall_s") or 0.0))
    for arms in window_cost.values():
        for bucket in arms.values():
            wall = bucket.pop("calls")
            bucket["planner_call_count"] = len(wall)
            bucket["mean_wall_s_per_call"] = round(sum(wall) / len(wall), 6)
            bucket["worst_wall_s_per_call"] = round(max(wall), 6)
    return {
        "primary": primary,
        "guard_1_regressions": regressions,
        "guard_1_passed": not regressions,
        "guard_2_harm": harm,
        "guard_2_harm_including_controls": harm_with_controls,
        "guard_3_cost": cost,
        "off_winner_efficiency": off_winner_efficiency,
        "planner_wall_time_by_window": window_cost,
    }


def build_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    duration_s: float,
    window_live_masks: Sequence[Mapping[str, Any]] = (),
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
        "scored_wrapper_invocation": {
            "wrapper": "E3AgentPolicy._call_plan_in_model",
            "stub": "ScoredPlannerPolicyStub",
            "inherits_real_wrapper": True,
            "wrapper_read_attributes": [
                "two_sided_goal_contract",
                "explorer.hud_mask",
                "cell",
                "transitions",
                "_episode_transition_start",
            ],
            "argument_parity_test": (
                "tests/python/test_experiment_10013_planner_dedup_tiebreak.py::"
                "test_scored_stub_and_real_policy_pass_identical_planner_arguments"
            ),
        },
        "planner_arms": [
            {
                "name": arm.name,
                "environment": dict(arm.environment),
                "max_nodes": arm.max_nodes,
                "max_depth": arm.max_depth,
                "analysis_role": (
                    "secondary_keep_off" if arm.name.endswith("TIEBREAK") else "primary"
                ),
            }
            for arm in PLANNER_ARMS
        ],
        "window_live_masks": [dict(row) for row in window_live_masks],
        "prior_artifact": str(RAW_REL / "v1_harness_mask_artifact.json"),
        "per_pair_arm_rows": [dict(row) for row in rows],
        **reductions,
        "limits": [
            "Public offline environments are a development proxy, not hidden-game efficacy.",
            "Cached generated engines are fixed historical draws; no model call was made.",
            "H2H winning-route replay cohorts are appendix evidence and not primary stalls.",
            "HUD_DEDUP+TIEBREAK is secondary evidence and is not an enablement candidate.",
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
    frame_mask: Optional[np.ndarray],
    transitions: Sequence[Any],
    window_mask_record: Mapping[str, Any],
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10013-AM1-SCORED-PATH runs the scored wrapper."""

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
        "mask": dict(window_mask_record),
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
    policy = ScoredPlannerPolicyStub(frame_mask, cell=cell, transitions=transitions)
    started = time.perf_counter()
    try:
        with exp12.Heartbeat(f"exp10013 planning {candidate.pair_id} {arm.name}"):
            plan = policy._call_plan_in_model(
                e3.plan_in_model,
                engine,
                is_done,
                rebuilt.grid.copy(),
                diagnostics=diagnostics,
                goal_energy_override=binary_goal_energy,
            )
    except BaseException as exc:
        base["planner_error"] = f"{type(exc).__name__}: {exc}"[:300]
        plan = None
    base["planner_wall_s"] = round(time.perf_counter() - started, 6)
    base["planner_diagnostics"] = diagnostics
    base["planner_engine_calls"] = int(diagnostics.get("nodes_expanded") or 0)
    if "CARNOT_ARC_PLAN_HUD_DEDUP" in arm.environment:
        base["mask"]["status"] = diagnostics.get("planner_hud_dedup_mask_status", "unresolved")
        base["mask"]["planner_reason"] = diagnostics.get("planner_hud_dedup_planner_reason")
        base["mask"]["resolution_reason"] = diagnostics.get("planner_hud_dedup_mask_reason")
        base["mask"]["swallow_check"] = diagnostics.get("planner_hud_dedup_swallow")
    else:
        base["mask"]["status"] = "disabled"
        base["mask"]["reason"] = "arm_flag_off"
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
    window_live_masks: list[dict[str, Any]] = []

    for game in exp12.WINDOWS:
        spec = specs[game]
        actions = exp12.load_episode_actions(repo_root, game)
        alignment = exp12.induction_alignment(repo_root, game, actions)
        live_frame_mask, replay_record = replay_main_window_live_mask(
            repo_root,
            game,
            actions,
            int(alignment["enclosing_action_index"]),
        )
        preview, preview_evidence = exp12.rebuild_arm_start_state(
            repo_root,
            game,
            spec,
            actions,
            alignment,
            exp12.LIVE_SCORED,
        )
        if preview_evidence.get("mode") != "root_after_reset":
            preview = exp12.rebuild_reset_state_from_env(
                env=preview.env,
                expected_level=preview.level,
                grid_reader=lambda frame, cell=spec.cell: e3.to_logical(grid_of(frame), cell),
                expected_grid=np.asarray(spec.rows[0].grid),
            )
        window_mask_record = live_mask_window_record(
            window_id=f"experiment_10010:{game}",
            cohort="stall_window",
            frame_mask=live_frame_mask,
            replay_record=replay_record,
            planner_frame=preview.frame,
            cell=spec.cell,
            transitions=spec.rows,
        )
        window_live_masks.append(window_mask_record)
        atomic_json(raw_dir / f"live_mask__stall_window__{game}.json", window_mask_record)
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
                    execution = (
                        plan_and_execute(
                            candidate,
                            rebuilt,
                            arm=arm,
                            cell=spec.cell,
                            frame_mask=live_frame_mask,
                            transitions=spec.rows,
                            window_mask_record=window_mask_record,
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
        live_frame_mask, replay_record = replay_registered_window_live_mask(
            repo_root, game, window.labels
        )
        preview, _ = exp12.rebuild_registered_arm_state(repo_root, game, window, exp12.LIVE_SCORED)
        window_mask_record = live_mask_window_record(
            window_id=f"qwen38_h2h:{game}",
            cohort="h2h_replay_counterfactual",
            frame_mask=live_frame_mask,
            replay_record=replay_record,
            planner_frame=preview.frame,
            cell=window.spec.cell,
            transitions=window.spec.rows,
        )
        window_live_masks.append(window_mask_record)
        atomic_json(
            raw_dir / f"live_mask__h2h_replay_counterfactual__{game}.json",
            window_mask_record,
        )
        progress(f"phase appendix window {game}: {len(stored_candidates[game])} pairs x 3 arms")
        for candidate in stored_candidates[game]:
            for arm in PLANNER_ARMS:
                progress(f"phase appendix pair {candidate.pair_id} arm {arm.name}: reset and plan")
                rebuilt, evidence = exp12.rebuild_registered_arm_state(
                    repo_root, game, window, exp12.LIVE_SCORED
                )
                with planner_arm_environment(arm):
                    execution = (
                        plan_and_execute(
                            candidate,
                            rebuilt,
                            arm=arm,
                            cell=window.spec.cell,
                            frame_mask=live_frame_mask,
                            transitions=window.spec.rows,
                            window_mask_record=window_mask_record,
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
        window_live_masks=window_live_masks,
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
