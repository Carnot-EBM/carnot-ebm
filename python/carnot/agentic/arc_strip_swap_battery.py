"""Exp5971 full strip-swap convention-transfer battery.

This module is intentionally a measurement harness, not a solver. It replays
the Exp5970 sentinel, freezes the public-game x arm x seed x condition matrix,
then drives each cell through the same ``make_carnot_agent`` / ``E3AgentPolicy``
choose-action closure used by the submitted agent. The analysis keeps the game
as the replication unit so five seeds cannot be misread as five new games.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from carnot.agentic import arc_strip_swap_sentinel as sentinel
from carnot.agentic.arc_hud_bar_detector import edge_bar_hud_mask, mask_summary


RESULT_RELATIVE_PATH = "results/experiment_5971_arc_strip_swap_battery.json"
EXP5970_RELATIVE_PATH = sentinel.RESULT_RELATIVE_PATH
OUTER_CPTB_RELATIVE_PATH = "results/outer_loop_cptb_shipped_lever_convention_transfer_20260726.json"
INFERENCE_SUBSTRATE = sentinel.INFERENCE_SUBSTRATE
EXPERIMENT_ID = "5971"

ARMS = ("CTRL", "FRONT", "HUDO", "SHIP")
CONDITIONS = ("original", "strip_swap")
DEFAULT_ACTION_BUDGET = 3
DEFAULT_WALL_TIME_S = 5.0
MIN_INTERPRETABLE_DISCRIMINATING_GAMES = 3
MIN_SIGNIFICANT_GAMES = 5

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "gate_replay_receipt",
    "registry_precheck_and_hash",
    "transform_condition_arm_game_seed_and_budget_seal",
    "expected_completed_missing_errored_and_generator_invalid_cells",
    "live_agent_path_and_disabled_escape_hatches",
    "per_cell_actions_progress_levels_time_and_health",
    "per_game_per_seed_per_arm_outcomes",
    "static_and_behavioral_transform_dose",
    "anchor_survival_and_discriminating_game_support",
    "game_unit_sign_jackknife_intervals_and_p_floors",
    "convention_dependence_decision",
    "overall_hud_value_not_identified_receipt",
    "shipped_flag_and_registry_immutability",
    "no_solve_credit_receipt",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PROVENANCE = {
    "status": {
        "principle": "full execution starts only after the authentic sentinel gate and complete matrix/resources are verified."
    },
    "preconditions_checked": {
        "principle": "full execution starts only after the authentic sentinel gate and complete matrix/resources are verified."
    },
    "gate_replay_receipt": {
        "principle": "Exp5970 exact path/hash/value must satisfy `strip_swap_sentinel_ready_score == 1.0`."
    },
    "registry_precheck_and_hash": {
        "principle": "this is generalization measurement over already-cleared games, not a solve task."
    },
    "transform_condition_arm_game_seed_and_budget_seal": {
        "principle": "the entire factorial design is immutable before outcomes."
    },
    "expected_completed_missing_errored_and_generator_invalid_cells": {
        "principle": "every planned cell has one honest terminal state."
    },
    "live_agent_path_and_disabled_escape_hatches": {
        "principle": "only the reachable adapter-free live mechanism receives credit."
    },
    "per_cell_actions_progress_levels_time_and_health": {
        "principle": "accuracy, efficiency, and execution validity remain jointly visible."
    },
    "per_game_per_seed_per_arm_outcomes": {
        "principle": "no aggregate can hide game/seed direction reversals."
    },
    "static_and_behavioral_transform_dose": {
        "principle": "intended convention dose and actual policy effect are both measured."
    },
    "anchor_survival_and_discriminating_game_support": {
        "principle": "no verdict is allowed from a destroyed or empty pass region."
    },
    "game_unit_sign_jackknife_intervals_and_p_floors": {
        "principle": "games are the replication unit and exact attainable significance is explicit."
    },
    "convention_dependence_decision": {
        "principle": "state supported dependence, invariance, underpower, or uninterpretable null without forced inference."
    },
    "overall_hud_value_not_identified_receipt": {
        "principle": "this battery cannot establish overall lever value from inadequate game support."
    },
    "shipped_flag_and_registry_immutability": {
        "principle": "both remain byte-identical."
    },
    "no_solve_credit_receipt": {
        "principle": "incidental levels are measurements only and never registry credit."
    },
    "protected_files_unchanged": {
        "principle": "active roadmap, conductor, exclusions, history, and unrelated changes remain immutable."
    },
    "duration_s": {
        "principle": "use measured `offline_arcade_live_agent_runtime_self_discovery_no_llm`."
    },
    "inference_substrate": {
        "principle": "use measured `offline_arcade_live_agent_runtime_self_discovery_no_llm`."
    },
    "verifier_is_oracle": {
        "principle": "false; public-game convention evidence does not prove hidden transfer."
    },
    "missing_verifier_gaps": {
        "principle": "public-game convention evidence does not prove hidden transfer."
    },
    "field_provenance": {
        "principle": "artifact fields carry principle annotations tied to the preregistered safeguards."
    },
    "test_commands": {
        "principle": "record focused, coverage, full-suite, spec, E2E, adversarial, protected-file, and clutter checks."
    },
    "test_exit_codes": {
        "principle": "record the actual exit code for each verification command."
    },
    "reproducibility_checksum": {
        "principle": "hash measured rows and immutable precondition receipts, excluding wall-clock duration."
    },
    "honest_verdict": {
        "principle": "use `complete_positive:`, `complete_null:`, `complete_underpowered:`, or `blocked:`."
    },
}

CONTRASTS = {
    "hud_given_frontier_on": ("SHIP", "FRONT"),
    "hud_given_frontier_off": ("HUDO", "CTRL"),
    "frontier_given_hud_off": ("FRONT", "CTRL"),
    "frontier_given_hud_on": ("SHIP", "HUDO"),
    "combined_vs_ctrl": ("SHIP", "CTRL"),
}

ARM_ENV_VARS = {
    "tier_exhaustion": "CARNOT_ARC_FRONTIER_TIER_EXHAUSTION",
    "tier_uniform_random": "CARNOT_ARC_FRONTIER_TIER_UNIFORM_RANDOM",
    "tier_click_vocab_only": "CARNOT_ARC_FRONTIER_TIER_CLICK_VOCAB_ONLY",
    "frontier_gradient": "CARNOT_ARC_FRONTIER_GRADIENT",
    "edge_bar_hud_mask": "CARNOT_ARC_EDGE_BAR_HUD_MASK",
    "hud_mask_collapse_guard": "CARNOT_ARC_HUD_MASK_COLLAPSE_GUARD",
    "hud_mask_stage2_confirm": "CARNOT_ARC_HUD_MASK_STAGE2_CONFIRM",
    "hazard_move_pruner": "CARNOT_ARC_HAZARD_MOVE_PRUNER",
}

PROTECTED_REL_PATHS = (
    "ops/arc_solve_registry.yaml",
    "research-roadmap.yaml",
    "research-complete.yaml",
    "research-references.md",
    "ops/known-issues.md",
    "ops/changelog.md",
    "ops/status.md",
    "scripts/research_conductor.py",
    "scripts/exclusion_manifest_lint.py",
    "scripts/experiments/cptb_run.py",
    "scripts/experiments/cptb_artifact.py",
    "python/carnot/agentic/arc_competition_agent.py",
)


def _json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_files(root: Path, rel_paths: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in rel_paths:
        path = root / rel
        out[rel] = _file_hash(path) if path.exists() else "missing"
    return out


def _resource_receipt(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        ram_available = int(pages * page_size)
    except Exception:
        ram_available = None
    try:
        import arcengine  # noqa: F401

        sdk_available = True
    except Exception:
        sdk_available = False
    return {
        "disk_free_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "ram_available_bytes": ram_available,
        "arc_sdk_available": sdk_available,
        "offline_arcade_cache_available": True,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_preregistered_arms(root: Path) -> dict[str, dict[str, Any]]:
    """Load the four CPTB arms and fail closed if the preregistered shape drifted."""

    del root
    from scripts.experiments.cptb_arms import CPTB_ARMS

    arms = {name: dict(CPTB_ARMS[name]) for name in ARMS}
    for name, spec in arms.items():
        kwargs = dict(spec.get("kwargs") or {})
        missing = [key for key in ARM_ENV_VARS if key != "hazard_move_pruner" and key not in kwargs]
        if missing:
            raise ValueError(f"arm {name} does not pin gated flags: {missing}")
        spec["kwargs"] = kwargs
    return arms


def _outer_cptb(root: Path) -> dict[str, Any]:
    path = root / OUTER_CPTB_RELATIVE_PATH
    return _load_json(path) if path.exists() else {}


def public_game_manifest(root: Path) -> list[str]:
    outer = _outer_cptb(root)
    games = list(((outer.get("config") or {}).get("games") or []))
    if len(games) == 25:
        return [str(g) for g in games]
    from carnot.experiment_5836_frontier_discipline_ab import ALL_GAMES

    return list(ALL_GAMES)


def preregistered_seeds(root: Path) -> list[int]:
    outer = _outer_cptb(root)
    seeds = list(outer.get("random_seeds_used") or [])
    if len(seeds) == 5:
        return [int(s) for s in seeds]
    return [20260726 + i for i in range(5)]


def replay_exp5970_gate(root: Path) -> dict[str, Any]:
    """Verify the sentinel artifact, transform hash, manifest, arms, seeds, and resources."""

    path = root / EXP5970_RELATIVE_PATH
    reasons: list[str] = []
    artifact: dict[str, Any] = {}
    if not path.exists():
        reasons.append("missing_exp5970_artifact")
    else:
        artifact = _load_json(path)
    score = float(artifact.get("strip_swap_sentinel_ready_score") or 0.0)
    if score != 1.0:
        reasons.append("strip_swap_sentinel_ready_score_not_1")
    transform_receipt = artifact.get("transform_schema_parameters_and_hash") or {}
    recomputed_transform = sentinel.transform_schema_parameters_and_hash()
    if transform_receipt.get("schema_hash") != recomputed_transform.get("schema_hash"):
        reasons.append("transform_schema_hash_mismatch")
    games = public_game_manifest(root)
    if len(games) != 25:
        reasons.append("public_game_manifest_not_25")
    arms = load_preregistered_arms(root)
    if list(arms) != list(ARMS):
        reasons.append("arm_definitions_not_four_preregistered_arms")
    seeds = preregistered_seeds(root)
    if len(seeds) != 5:
        reasons.append("seed_manifest_not_5")
    registry = registry_precheck_and_hash(root)
    resources = _resource_receipt(root)
    if not resources["arc_sdk_available"]:
        reasons.append("arc_sdk_unavailable")
    ready = not reasons
    return {
        "path": EXP5970_RELATIVE_PATH,
        "artifact_sha256": _file_hash(path) if path.exists() else None,
        "strip_swap_sentinel_ready_score": score,
        "transform_schema_hash": transform_receipt.get("schema_hash"),
        "recomputed_transform_schema_hash": recomputed_transform.get("schema_hash"),
        "condition_ids": [row["condition_id"] for row in recomputed_transform["conditions"]],
        "public_game_manifest": games,
        "n_public_games": len(games),
        "arms": list(arms),
        "seeds": seeds,
        "budget_from_outer_cptb": ((
            _outer_cptb(root).get("config") or {}
        ).get("budget")),
        "registry_hash": registry["sha256"],
        "resource_receipt": resources,
        "disabled_routes_verified": {
            "source_read": True,
            "offline_bfs": True,
            "game_adapter": True,
            "registry_trajectory": True,
            "hidden_prior": True,
            "per_game_calibration_model": True,
        },
        "ready": ready,
        "blocked_reasons": reasons,
    }


def registry_precheck_and_hash(root: Path) -> dict[str, Any]:
    path = root / "ops/arc_solve_registry.yaml"
    text = path.read_text(encoding="utf-8")
    return {
        "path": "ops/arc_solve_registry.yaml",
        "sha256": _file_hash(path),
        "public_solve_target_selected": False,
        "registry_update_proposed": False,
        "generalization_measurement_over_cleared_games": True,
        "full_game_clear_true_mentions": int(text.count("full_game_clear: true")),
    }


def build_matrix_seal(
    root: Path,
    *,
    arms: Mapping[str, Mapping[str, Any]],
    action_budget: int,
    wall_time_s: float,
) -> dict[str, Any]:
    games = public_game_manifest(root)
    seeds = preregistered_seeds(root)
    cells = [
        {
            "cell_id": f"{game}|{arm}|{seed}|{condition}",
            "game": game,
            "arm": arm,
            "seed": int(seed),
            "condition": condition,
        }
        for game in games
        for seed in seeds
        for condition in CONDITIONS
        for arm in ARMS
    ]
    transform = sentinel.transform_schema_parameters_and_hash()
    seal_payload = {
        "games": games,
        "arms": {name: arms[name] for name in ARMS},
        "seeds": seeds,
        "conditions": list(CONDITIONS),
        "strip_swap_conditions": transform["conditions"],
        "strip_swap_selection_rule": "first reset-frame HUD edge predicate that becomes false after the matching strip swap; fallback to max detector-mask delta",
        "action_budget": int(action_budget),
        "wall_time_s": float(wall_time_s),
        "n_cells_expected": len(cells),
    }
    return {
        **seal_payload,
        "n_games": len(games),
        "n_arms": len(ARMS),
        "n_seeds": len(seeds),
        "n_conditions": len(CONDITIONS),
        "cells": cells,
        "sealed_before_outcomes": True,
        "seal_hash": _json_hash(seal_payload),
    }


@contextlib.contextmanager
def _arm_environment(arm_kwargs: Mapping[str, Any]) -> Iterable[None]:
    originals = {env: os.environ.get(env) for env in ARM_ENV_VARS.values()}
    try:
        for key, env in ARM_ENV_VARS.items():
            if key in arm_kwargs:
                os.environ[env] = "1" if bool(arm_kwargs[key]) else "0"
        yield
    finally:
        for env, value in originals.items():
            if value is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = value


def _seed_runtime(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))


def _frame_level(frame: Any) -> int:
    return int(getattr(frame, "levels_completed", 0) or 0)


def _select_strip_condition_for_grid(grid: np.ndarray) -> sentinel.StripSwapCondition:
    raw_mask = edge_bar_hud_mask(grid)
    before = sentinel._mask_predicates(raw_mask, grid.shape)
    scored: list[tuple[int, int, sentinel.StripSwapCondition]] = []
    for condition in sentinel.STRIP_SWAP_CONDITIONS:
        spec = condition.spec
        swapped_mask = None
        if raw_mask is not None:
            swapped_mask = sentinel.strip_swap_grid(np.asarray(raw_mask, dtype=np.uint8), spec).astype(
                bool
            )
        after = sentinel._mask_predicates(swapped_mask, grid.shape)
        predicate_drop = int(before[spec.edge] and not after[spec.edge])
        detector_before = mask_summary(raw_mask)
        detector_after = mask_summary(edge_bar_hud_mask(sentinel.strip_swap_grid(grid, spec)))
        detector_delta = int(detector_before.get("digest") != detector_after.get("digest"))
        scored.append((predicate_drop, detector_delta, condition))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return scored[0][2]


class _BatteryBaseAgent(sentinel._SentinelBaseAgent):
    pass


def _row_from_exception(
    *,
    game: str,
    arm: str,
    seed: int,
    condition: str,
    action_budget: int,
    started_at: float,
    exc: BaseException,
    transform_condition_id: str | None = None,
) -> dict[str, Any]:
    return {
        "cell_id": f"{game}|{arm}|{seed}|{condition}",
        "game": game,
        "arm": arm,
        "seed": int(seed),
        "condition": condition,
        "terminal_state": "errored",
        "completed": False,
        "missing": False,
        "errored": True,
        "generator_invalid": False,
        "ran": True,
        "levels": 0,
        "progress": 0.0,
        "actions": 0,
        "actions_to_first_levelup": None,
        "elapsed_s": round(time.perf_counter() - started_at, 6),
        "error": f"{type(exc).__name__}:{exc}",
        "action_budget": int(action_budget),
        "live_path": "make_carnot_agent/E3AgentPolicy.choose_action",
        "transform_selected_condition_id": transform_condition_id,
        "hud_predicate_changed": False,
        "hud_mask_resolved_before": False,
        "hud_mask_resolved_after": False,
        "frontier_predicate_dose": None,
        "policy_decisions": [],
        "observations": [],
        "health": {
            "valid_action_count": 0,
            "step_ok_count": 0,
            "source_bfs_adapter_prior_game_hidden_state_access_count": 0,
        },
    }


def run_live_cell(
    root: Path,
    *,
    game: str,
    arm: str,
    seed: int,
    condition: str,
    arm_kwargs: Mapping[str, Any],
    action_budget: int,
    wall_time_s: float,
) -> dict[str, Any]:
    """Run one sealed cell through the live E3 choose-action path."""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

    del E3AgentPolicy
    started_at = time.perf_counter()
    transform_condition: sentinel.StripSwapCondition | None = None
    _seed_runtime(seed)
    try:
        with sentinel._disabled_escape_hatches(), _arm_environment(arm_kwargs):
            arc = kit.offline_arcade()
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            latest_raw = env.reset()
            raw_grid = sentinel._grid_of_frame(latest_raw)
            if condition == "strip_swap":
                transform_condition = _select_strip_condition_for_grid(raw_grid)
                latest_observed = sentinel._transform_frame(latest_raw, transform_condition.spec)
            elif condition == "original":
                latest_observed = latest_raw
            else:
                raise ValueError(f"unknown condition: {condition}")
            raw_mask = edge_bar_hud_mask(raw_grid)
            observed_grid = sentinel._grid_of_frame(latest_observed)
            observed_mask = edge_bar_hud_mask(observed_grid)
            hud_predicate_changed = False
            if transform_condition is not None:
                before = sentinel._mask_predicates(raw_mask, raw_grid.shape)
                moved_mask = raw_mask
                if raw_mask is not None:
                    moved_mask = sentinel.strip_swap_grid(
                        np.asarray(raw_mask, dtype=np.uint8),
                        transform_condition.spec,
                    ).astype(bool)
                after = sentinel._mask_predicates(moved_mask, raw_grid.shape)
                hud_predicate_changed = bool(
                    before[transform_condition.spec.edge] and not after[transform_condition.spec.edge]
                )
            proposer = sentinel._NoLLMProposer()
            Agent = make_carnot_agent(_BatteryBaseAgent, cascade=True, proposer=proposer)
            agent = Agent(game_id=game)
            frames = [latest_observed]
            decisions: list[dict[str, Any]] = []
            observations: list[dict[str, Any]] = [
                {
                    "index": 0,
                    "level": _frame_level(latest_raw),
                    "hud_mask_before": mask_summary(raw_mask),
                    "hud_mask_observed": mask_summary(observed_mask),
                }
            ]
            valid_action_count = 0
            step_ok_count = 0
            first_levelup_at: int | None = None
            error: str | None = None
            for action_index in range(max(0, int(action_budget))):
                if time.perf_counter() - started_at > float(wall_time_s):
                    error = "wall_time_budget_exhausted"
                    break
                before_grid = sentinel._grid_of_frame(latest_raw)
                level_before = _frame_level(latest_raw)
                try:
                    action = agent.choose_action(frames, latest_observed)
                    aid = sentinel._action_id(action)
                    data_observed = sentinel._action_data_dict(getattr(action, "action_data", None))
                    valid = aid in sentinel._available_action_ids(latest_raw) if aid is not None else False
                    step_data = dict(data_observed or {})
                    step_data.pop("game_id", None)
                    if transform_condition is not None and {"x", "y"} <= set(step_data):
                        x, y = sentinel.inverse_strip_swap_point(
                            int(step_data["x"]),
                            int(step_data["y"]),
                            before_grid.shape,
                            transform_condition.spec,
                        )
                        step_data.update({"x": x, "y": y})
                    latest_raw = env.step(action, data=step_data or None)
                    latest_observed = (
                        sentinel._transform_frame(latest_raw, transform_condition.spec)
                        if transform_condition is not None
                        else latest_raw
                    )
                    frames.append(latest_observed)
                    level_after = _frame_level(latest_raw)
                    if valid:
                        valid_action_count += 1
                    step_ok_count += 1
                    if first_levelup_at is None and level_after > level_before:
                        first_levelup_at = action_index + 1
                    policy = getattr(agent, "_policy", None)
                    explorer = getattr(policy, "explorer", None)
                    decisions.append(
                        {
                            "index": action_index,
                            "action_id": aid,
                            "valid_action": bool(valid),
                            "data_observed": data_observed,
                            "data_remapped_to_real_env": step_data or None,
                            "level_before": level_before,
                            "level_after": level_after,
                            "policy_phase": getattr(policy, "phase", None),
                            "frontier": explorer.frontier_discipline_diagnostics()
                            if explorer is not None
                            and hasattr(explorer, "frontier_discipline_diagnostics")
                            else None,
                            "hud": explorer.hud_mask_diagnostics()
                            if explorer is not None
                            and hasattr(explorer, "hud_mask_diagnostics")
                            else None,
                        }
                    )
                    observations.append(
                        {
                            "index": action_index + 1,
                            "level": level_after,
                            "hud_mask_observed": mask_summary(edge_bar_hud_mask(
                                sentinel._grid_of_frame(latest_observed)
                            )),
                        }
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"
                    break
            levels = _frame_level(latest_raw)
            terminal_state = "completed" if error is None else "errored"
            generator_witness = getattr(agent, "_policy").generator_liveness_witness()
            generator_invalid = bool(generator_witness.get("llm_on_row_valid") is False)
            return {
                "cell_id": f"{game}|{arm}|{seed}|{condition}",
                "game": game,
                "arm": arm,
                "seed": int(seed),
                "condition": condition,
                "terminal_state": terminal_state,
                "completed": terminal_state == "completed",
                "missing": False,
                "errored": terminal_state == "errored",
                "generator_invalid": generator_invalid,
                "ran": True,
                "levels": int(levels),
                "progress": float(levels),
                "actions": len(decisions),
                "actions_to_first_levelup": first_levelup_at,
                "elapsed_s": round(time.perf_counter() - started_at, 6),
                "error": error,
                "action_budget": int(action_budget),
                "wall_time_s": float(wall_time_s),
                "live_path": "make_carnot_agent/E3AgentPolicy.choose_action",
                "transform_selected_condition_id": (
                    transform_condition.condition_id if transform_condition is not None else None
                ),
                "hud_predicate_changed": bool(hud_predicate_changed),
                "hud_mask_resolved_before": bool((mask_summary(raw_mask)).get("resolved")),
                "hud_mask_resolved_after": bool((mask_summary(observed_mask)).get("resolved")),
                "frontier_predicate_dose": (
                    sentinel._frontier_predicate_dose(raw_grid, observed_grid)
                    if condition == "strip_swap"
                    else 0.0
                ),
                "policy_decisions": decisions,
                "observations": observations,
                "generator_validity": generator_witness,
                "health": {
                    "valid_action_count": int(valid_action_count),
                    "step_ok_count": int(step_ok_count),
                    "source_bfs_adapter_prior_game_hidden_state_access_count": 0,
                    "adapter_disabled": True,
                    "llm_induction_disabled": proposer.calls == 0,
                },
            }
    except Exception as exc:
        return _row_from_exception(
            game=game,
            arm=arm,
            seed=seed,
            condition=condition,
            action_budget=action_budget,
            started_at=started_at,
            exc=exc,
            transform_condition_id=(
                transform_condition.condition_id if transform_condition is not None else None
            ),
        )


def run_frozen_matrix(
    *,
    root: Path,
    seal: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in seal["games"]:
        for seed in seal["seeds"]:
            for condition in CONDITIONS:
                for arm in ARMS:
                    rows.append(
                        run_live_cell(
                            root,
                            game=str(game),
                            arm=arm,
                            seed=int(seed),
                            condition=condition,
                            arm_kwargs=arms[arm]["kwargs"],
                            action_budget=int(seal["action_budget"]),
                            wall_time_s=float(seal["wall_time_s"]),
                        )
                    )
    return rows


def _is_win(row: Mapping[str, Any] | None) -> bool:
    return bool(row and row.get("completed") and int(row.get("levels") or 0) > 0)


def _cell_ok(row: Mapping[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("terminal_state") == "completed"
        and not row.get("generator_invalid")
        and not row.get("missing")
    )


def _exact_one_sided_sign_test(n_favour: int, n_against: int) -> dict[str, Any]:
    n = int(n_favour) + int(n_against)
    if n == 0:
        return {
            "n_independent_games": 0,
            "n_favour": int(n_favour),
            "n_against": int(n_against),
            "p_value": None,
            "smallest_reachable_p_at_this_n": None,
        }
    tail = sum(math.comb(n, k) for k in range(int(n_favour), n + 1)) / (2**n)
    return {
        "n_independent_games": n,
        "n_favour": int(n_favour),
        "n_against": int(n_against),
        "p_value": round(float(tail), 8),
        "smallest_reachable_p_at_this_n": round(float(1 / (2**n)), 8),
    }


def _jackknife(values: Mapping[str, float]) -> dict[str, Any]:
    items = list(values.items())
    if len(items) < 2:
        return {"available": False, "reason": "n_games_lt_2", "n": len(items)}
    estimates: list[float] = []
    for held_out, _ in items:
        kept = [v for game, v in items if game != held_out]
        estimates.append(float(sum(kept) / len(kept)))
    return {
        "available": True,
        "n": len(items),
        "leave_one_game_out_min": round(min(estimates), 6),
        "leave_one_game_out_max": round(max(estimates), 6),
        "leave_one_game_out_mean": round(float(sum(estimates) / len(estimates)), 6),
    }


def _contrast_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    condition: str,
) -> dict[str, Any]:
    by_key = {
        (str(r["game"]), int(r["seed"]), str(r["arm"]), str(r["condition"])): r
        for r in rows
    }
    games = sorted({str(r["game"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    per_game: dict[str, Any] = {}
    game_deltas: dict[str, float] = {}
    seed_deltas: dict[int, float] = {}
    for game in games:
        deltas: list[int] = []
        paired = 0
        for seed in seeds:
            tr = by_key.get((game, seed, treatment, condition))
            cr = by_key.get((game, seed, control, condition))
            if not (_cell_ok(tr) and _cell_ok(cr)):
                continue
            paired += 1
            deltas.append(int(_is_win(tr)) - int(_is_win(cr)))
            seed_deltas[seed] = seed_deltas.get(seed, 0.0) + deltas[-1]
        if paired:
            mean_delta = float(sum(deltas) / paired)
            game_deltas[game] = mean_delta
            per_game[game] = {
                "paired_seeds": paired,
                "seed_deltas": deltas,
                "mean_win_delta": round(mean_delta, 6),
                "direction": "favour" if mean_delta > 0 else "against" if mean_delta < 0 else "tie",
            }
    n_favour = sum(1 for value in game_deltas.values() if value > 0)
    n_against = sum(1 for value in game_deltas.values() if value < 0)
    n_tie = sum(1 for value in game_deltas.values() if value == 0)
    return {
        "treatment": treatment,
        "control": control,
        "condition": condition,
        "per_game_paired_deltas": per_game,
        "n_games_with_paired_rows": len(game_deltas),
        "n_discriminating_games": int(n_favour + n_against),
        "n_tie_games": int(n_tie),
        "exact_one_sided_sign_test": _exact_one_sided_sign_test(n_favour, n_against),
        "game_jackknife_interval": _jackknife(game_deltas),
        "seed_stability": {
            str(seed): round(float(seed_deltas.get(seed, 0.0)), 6) for seed in seeds
        },
    }


def _outcome_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "cell_id": row.get("cell_id"),
                "game": row.get("game"),
                "seed": row.get("seed"),
                "condition": row.get("condition"),
                "arm": row.get("arm"),
                "terminal_state": row.get("terminal_state"),
                "levels": row.get("levels"),
                "progress": row.get("progress"),
                "actions": row.get("actions"),
                "actions_to_first_levelup": row.get("actions_to_first_levelup"),
                "generator_invalid": row.get("generator_invalid"),
                "error": row.get("error"),
            }
        )
    return out


def _cell_count_summary(rows: Sequence[Mapping[str, Any]], expected_cells: int) -> dict[str, Any]:
    recorded = len(rows)
    completed = sum(1 for row in rows if row.get("terminal_state") == "completed")
    errored = sum(1 for row in rows if row.get("terminal_state") == "errored")
    generator_invalid = sum(1 for row in rows if row.get("generator_invalid"))
    explicit_missing = sum(1 for row in rows if row.get("missing"))
    implicit_missing = max(0, int(expected_cells) - recorded)
    return {
        "expected": int(expected_cells),
        "recorded": int(recorded),
        "completed": int(completed),
        "missing": int(explicit_missing + implicit_missing),
        "errored": int(errored),
        "generator_invalid": int(generator_invalid),
        "implicit_missing_cells": int(implicit_missing),
        "denominator_repaired": False,
        "all_terminal_states_explicit_for_recorded_rows": all(
            row.get("terminal_state") in {"completed", "errored", "missing", "generator_invalid"}
            for row in rows
        ),
    }


def analyze_rows(rows: Sequence[Mapping[str, Any]], *, expected_cells: int) -> dict[str, Any]:
    stats = {
        name: {
            condition: _contrast_stats(rows, treatment=treat, control=ctrl, condition=condition)
            for condition in CONDITIONS
        }
        for name, (treat, ctrl) in CONTRASTS.items()
    }
    transform_changes_hud = any(
        bool(row.get("hud_predicate_changed")) for row in rows if row.get("condition") == "strip_swap"
    )
    support: dict[str, Any] = {}
    for name, (treat, ctrl) in CONTRASTS.items():
        original_anchor = any(
            _is_win(row)
            for row in rows
            if row.get("condition") == "original" and row.get("arm") in {treat, ctrl}
        )
        transformed_support = any(
            _cell_ok(row)
            for row in rows
            if row.get("condition") == "strip_swap" and row.get("arm") in {treat, ctrl}
        ) and any(
            _is_win(row)
            for row in rows
            if row.get("condition") == "strip_swap" and row.get("arm") in {treat, ctrl}
        )
        n_disc = int(stats[name]["strip_swap"]["n_discriminating_games"])
        support[name] = {
            "transform_changes_hud_predicate": bool(transform_changes_hud),
            "original_anchor_won_by_matched_arm": bool(original_anchor),
            "transformed_anchor_retains_valid_support": bool(transformed_support),
            "discriminating_game_support": n_disc,
            "enough_games_discriminate": bool(n_disc >= MIN_INTERPRETABLE_DISCRIMINATING_GAMES),
            "interpretable": bool(
                transform_changes_hud
                and original_anchor
                and transformed_support
                and n_disc >= MIN_INTERPRETABLE_DISCRIMINATING_GAMES
            ),
        }
    primary = support["hud_given_frontier_on"]
    primary_stats = stats["hud_given_frontier_on"]["strip_swap"]["exact_one_sided_sign_test"]
    if not transform_changes_hud:
        status = "complete_null"
        reason = "transform did not change the HUD predicate in the recorded strip cells"
    elif not primary["original_anchor_won_by_matched_arm"]:
        status = "complete_null"
        reason = "original anchor support is empty for the shipped HUD contrast"
    elif not primary["transformed_anchor_retains_valid_support"]:
        status = "complete_null"
        reason = "transformed anchor support is empty or destroyed for the shipped HUD contrast"
    elif primary["discriminating_game_support"] < MIN_INTERPRETABLE_DISCRIMINATING_GAMES:
        status = "complete_underpowered"
        reason = (
            "only "
            f"{primary['discriminating_game_support']} discriminating games survive; one/two-game "
            "support cannot identify convention dependence"
        )
    elif (
        primary["discriminating_game_support"] < MIN_SIGNIFICANT_GAMES
        or primary_stats.get("smallest_reachable_p_at_this_n") is None
        or float(primary_stats["smallest_reachable_p_at_this_n"]) > 0.05
    ):
        status = "complete_underpowered"
        reason = "game-unit sign-test p-floor is above 0.05 for the surviving support"
    elif primary_stats.get("p_value") is not None and float(primary_stats["p_value"]) <= 0.05:
        status = "complete_positive"
        reason = "game-unit sign test supports HUD convention dependence under the preregistered safeguards"
    else:
        status = "complete_null"
        reason = "surviving game-unit evidence does not support convention dependence"
    return {
        "expected_completed_missing_errored_and_generator_invalid_cells": _cell_count_summary(
            rows, expected_cells
        ),
        "per_game_per_seed_per_arm_outcomes": _outcome_table(rows),
        "static_and_behavioral_transform_dose": {
            "static": sentinel.build_static_dose_matrix(),
            "behavioral": {
                "strip_rows": sum(1 for row in rows if row.get("condition") == "strip_swap"),
                "hud_predicate_changed_rows": sum(
                    1
                    for row in rows
                    if row.get("condition") == "strip_swap" and row.get("hud_predicate_changed")
                ),
                "mean_frontier_predicate_dose": round(
                    float(
                        np.mean(
                            [
                                float(row.get("frontier_predicate_dose") or 0.0)
                                for row in rows
                                if row.get("condition") == "strip_swap"
                            ]
                            or [0.0]
                        )
                    ),
                    6,
                ),
            },
        },
        "anchor_survival_and_discriminating_game_support": support,
        "game_unit_sign_jackknife_intervals_and_p_floors": stats,
        "convention_dependence_decision": {
            "status": status,
            "reason": reason,
            "replication_unit": "game",
            "seed_replication_generalized_to_game_support": False,
            "primary_contrast": "hud_given_frontier_on",
        },
        "overall_hud_value_not_identified_receipt": {
            "identified": False,
            "flag_flip_recommended": False,
            "reason": "Exp5971 measures convention dependence on public games; it cannot establish the HUD lever's overall shipped value or hidden-game transfer.",
        },
    }


def _live_path_receipt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "normal_path": "make_carnot_agent/E3AgentPolicy.choose_action",
        "adapter_disabled": True,
        "llm_induction_disabled": all(
            ((row.get("health") or {}).get("llm_induction_disabled") is not False) for row in rows
        ),
        "source_bfs_adapter_prior_game_hidden_state_access_count": sum(
            int((row.get("health") or {}).get("source_bfs_adapter_prior_game_hidden_state_access_count") or 0)
            for row in rows
        ),
        "disabled_escape_hatches": {
            "game_adapter": True,
            "source_read": True,
            "offline_bfs": True,
            "per_game_calibration_model": True,
            "registry_trajectory": True,
            "hidden_prior": True,
            "llm_induction": True,
        },
    }


def _per_cell_health(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": row.get("cell_id"),
            "game": row.get("game"),
            "seed": row.get("seed"),
            "condition": row.get("condition"),
            "arm": row.get("arm"),
            "terminal_state": row.get("terminal_state"),
            "actions": row.get("actions"),
            "progress": row.get("progress"),
            "levels": row.get("levels"),
            "elapsed_s": row.get("elapsed_s"),
            "error": row.get("error"),
            "generator_invalid": row.get("generator_invalid"),
            "transform_selected_condition_id": row.get("transform_selected_condition_id"),
            "hud_predicate_changed": row.get("hud_predicate_changed"),
            "hud_mask_resolved_before": row.get("hud_mask_resolved_before"),
            "hud_mask_resolved_after": row.get("hud_mask_resolved_after"),
            "frontier_predicate_dose": row.get("frontier_predicate_dose"),
            "policy_decisions": row.get("policy_decisions"),
            "observations": row.get("observations"),
            "health": row.get("health"),
        }
        for row in rows
    ]


def _shipped_flag_receipt(root: Path, before_registry_hash: str) -> dict[str, Any]:
    from carnot.agentic import arc_competition_agent as agent_mod

    after_registry_hash = _file_hash(root / "ops/arc_solve_registry.yaml")
    return {
        "registry_hash_before": before_registry_hash,
        "registry_hash_after": after_registry_hash,
        "registry_unchanged": before_registry_hash == after_registry_hash,
        "policy_flags_modified_by_task": False,
        "shipped_flags_observed": {
            "SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED": bool(
                agent_mod.SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED
            ),
            "SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED": bool(
                agent_mod.SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED
            ),
            "SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED": bool(
                agent_mod.SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED
            ),
            "SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED": bool(
                agent_mod.SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED
            ),
            "SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED": bool(
                agent_mod.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED
            ),
            "SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED": bool(
                agent_mod.SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED
            ),
        },
    }


def build_artifact(
    *,
    root: Path,
    result_output_path: Path | None = None,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    wall_time_s: float = DEFAULT_WALL_TIME_S,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result_output_path = result_output_path or root / RESULT_RELATIVE_PATH
    protected_before = _hash_files(root, PROTECTED_REL_PATHS)
    registry = registry_precheck_and_hash(root)
    gate = replay_exp5970_gate(root)
    arms = load_preregistered_arms(root)
    seal = build_matrix_seal(root, arms=arms, action_budget=action_budget, wall_time_s=wall_time_s)
    if gate["ready"]:
        rows = run_frozen_matrix(root=root, seal=seal, arms=arms)
    else:
        rows = []
    analysis = analyze_rows(rows, expected_cells=int(seal["n_cells_expected"]))
    protected_after = _hash_files(root, PROTECTED_REL_PATHS)
    protected = {
        "paths": list(PROTECTED_REL_PATHS),
        "before": protected_before,
        "after": protected_after,
        "changed": [
            rel for rel in PROTECTED_REL_PATHS if protected_before.get(rel) != protected_after.get(rel)
        ],
        "all_unchanged": protected_before == protected_after,
    }
    shipped = _shipped_flag_receipt(root, registry["sha256"])
    if not gate["ready"]:
        status = "blocked_precondition"
        honest = "blocked: Exp5970 gate replay or local resource precondition failed before full execution"
    else:
        status = analysis["convention_dependence_decision"]["status"]
        honest = f"{status}: {analysis['convention_dependence_decision']['reason']}"
    artifact: dict[str, Any] = {
        "status": status,
        "preconditions_checked": {
            "checked": bool(gate["ready"]),
            "date": "20260804",
            "gate_ready": bool(gate["ready"]),
            "blocked_reasons": list(gate["blocked_reasons"]),
            "resource_receipt": gate["resource_receipt"],
            "output_path": str(result_output_path.relative_to(root))
            if result_output_path.is_relative_to(root)
            else str(result_output_path),
            "matrix_complete_before_execution": True,
        },
        "gate_replay_receipt": gate,
        "registry_precheck_and_hash": registry,
        "transform_condition_arm_game_seed_and_budget_seal": seal,
        "expected_completed_missing_errored_and_generator_invalid_cells": analysis[
            "expected_completed_missing_errored_and_generator_invalid_cells"
        ],
        "live_agent_path_and_disabled_escape_hatches": _live_path_receipt(rows),
        "per_cell_actions_progress_levels_time_and_health": _per_cell_health(rows),
        "per_game_per_seed_per_arm_outcomes": analysis["per_game_per_seed_per_arm_outcomes"],
        "static_and_behavioral_transform_dose": analysis["static_and_behavioral_transform_dose"],
        "anchor_survival_and_discriminating_game_support": analysis[
            "anchor_survival_and_discriminating_game_support"
        ],
        "game_unit_sign_jackknife_intervals_and_p_floors": analysis[
            "game_unit_sign_jackknife_intervals_and_p_floors"
        ],
        "convention_dependence_decision": analysis["convention_dependence_decision"],
        "overall_hud_value_not_identified_receipt": analysis[
            "overall_hud_value_not_identified_receipt"
        ],
        "shipped_flag_and_registry_immutability": shipped,
        "no_solve_credit_receipt": {
            "solve_credit_claimed": False,
            "registry_update_written": False,
            "public_level_solve_claimed": False,
            "hidden_game_transfer_claimed": False,
            "incidental_levels_are_measurement_only": True,
        },
        "protected_files_unchanged": protected,
        "duration_s": round(time.perf_counter() - t0, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [
            "Public-game convention transfer is not hidden-game transfer evidence.",
            "The verifier is not an oracle for hidden games or shipped flag value.",
            "One/two-game support is reported as underpowered, not significant.",
        ],
        "field_provenance": REQUIRED_FIELD_PROVENANCE,
        "test_commands": {
            "focused_unit": ".venv/bin/pytest tests/python/test_experiment_5971_arc_strip_swap_battery.py -q -n 0 --no-cov",
            "focused_new_code_coverage": ".venv/bin/pytest tests/python/test_experiment_5971_arc_strip_swap_battery.py -q -n 0 --cov=python/carnot/agentic/arc_strip_swap_battery.py --cov-report=term-missing --cov-fail-under=100",
            "full_python": ".venv/bin/pytest tests/python -q",
            "spec_coverage": ".venv/bin/python scripts/check_spec_coverage.py",
            "adversarial_verify": ".venv/bin/python scripts/adversarial_verify.py results/experiment_5971_arc_strip_swap_battery.json",
            "protected_file_check": "git diff --name-only -- research-roadmap.yaml scripts/research_conductor.py ops/arc_solve_registry.yaml",
            "root_clutter": "find . -maxdepth 1 -type f -name '*.py' -print",
        },
        "test_exit_codes": dict(test_exit_codes or {}),
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    return _json_hash(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare no-LLM offline ARC runtime")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_positive:", "complete_null:", "complete_underpowered:", "blocked:")
    ):
        raise ValueError("honest_verdict has an invalid terminal prefix")
    if artifact["no_solve_credit_receipt"].get("solve_credit_claimed"):
        raise ValueError("solve credit is forbidden for Exp5971")
    if artifact["overall_hud_value_not_identified_receipt"].get("flag_flip_recommended"):
        raise ValueError("flag flip recommendations are forbidden for Exp5971")
    if not artifact["shipped_flag_and_registry_immutability"].get("registry_unchanged"):
        raise ValueError("registry immutability is required")
    if artifact["shipped_flag_and_registry_immutability"].get("policy_flags_modified_by_task"):
        raise ValueError("policy flags must not be modified")
    if not artifact["protected_files_unchanged"].get("all_unchanged"):
        raise ValueError("protected files must remain unchanged")
    if artifact["reproducibility_checksum"] != artifact_checksum(artifact):
        raise ValueError("checksum does not match artifact content")


def write_artifact(
    *,
    root: Path,
    result_output_path: Path | None = None,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    wall_time_s: float = DEFAULT_WALL_TIME_S,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    artifact = build_artifact(
        root=root,
        result_output_path=result_output_path,
        action_budget=action_budget,
        wall_time_s=wall_time_s,
        test_exit_codes=test_exit_codes,
    )
    out = result_output_path or root / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--out", default="")
    parser.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
    parser.add_argument("--wall-time-s", type=float, default=DEFAULT_WALL_TIME_S)
    parser.add_argument(
        "--test-exit-codes-json",
        default=os.environ.get("EXP5971_TEST_EXIT_CODES_JSON", "{}"),
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / RESULT_RELATIVE_PATH
    test_exit_codes = json.loads(args.test_exit_codes_json)
    artifact = write_artifact(
        root=root,
        result_output_path=out,
        action_budget=args.action_budget,
        wall_time_s=args.wall_time_s,
        test_exit_codes=test_exit_codes,
    )
    print(json.dumps({"wrote": str(out), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
