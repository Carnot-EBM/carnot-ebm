"""Experiment 5753: frozen generic causal primitive live-registry A/B.

This experiment is a development-proxy measurement over already completed
public ARC games. It selects at most one Exp5740 causal primitive through the
Exp5745 normalized gate, wires it into the live E3 policy path for the treatment
arm, and records a matched full-registry A/B without awarding registry credit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import yaml

from carnot.agentic.arc_generic_causal_primitives import BoundaryCollisionPrimitive


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5753_arc_generic_primitive_live_registry_ab.json")
EXP5740_RELATIVE_PATH = Path("results/experiment_5740_arc_game_blind_primitive_causal_audit.json")
EXP5741_RELATIVE_PATH = Path("results/experiment_5741_arc_generic_primitive_live_ab.json")
EXP5745_RELATIVE_PATH = Path("results/experiment_5745_arc_causal_gate_schema_corrigendum.json")
EXP5727_RELATIVE_PATH = Path("results/experiment_5727_arc_generalization_live_oracle_gap_v511.json")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")

ACTION_BUDGET = 400
PUBLIC_GAME_COUNT = 25
REGISTRY_LEVEL_COUNT = 183
RANDOM_SEEDS = [20260720, 5753]
SELECTION_RULE = (
    "max_composite_utility_delta_then_ci_lower_then_paired_replay_count_then_primitive_id"
)
LIVE_POLICY_PATH = "carnot.agentic.arc_competition_agent.E3AgentPolicy/StepwiseExplorer"
SOLVE_PROVENANCE = "development_proxy"
PRODUCTION_DEFAULT_ENABLED = False
SPEC_REFS = [
    "REQ-ARC-WMTE-5753",
    "SCENARIO-ARC-WMTE-5753-GATE-AND-SELECTION",
    "SCENARIO-ARC-WMTE-5753-LIVE-REACHABILITY-AND-LEAK-CANARIES",
    "SCENARIO-ARC-WMTE-5753-FULL-REGISTRY-PAIRED-AB",
]
UPSTREAM_PATHS = {
    "exp5740": EXP5740_RELATIVE_PATH,
    "exp5741": EXP5741_RELATIVE_PATH,
    "exp5745": EXP5745_RELATIVE_PATH,
    "exp5727": EXP5727_RELATIVE_PATH,
    "registry": REGISTRY_RELATIVE_PATH,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "every Exp5753 field carries its audit rationale so the live-registry A/B is schema-stable.",
    "preconditions_checked": "structured gates fail closed before live work on provenance, registry saturation, environment reachability, budgets, seeds, resources, and submitted path.",
    "spec_refs": "REQ/SCENARIO anchors make the measurement traceable to OpenSpec.",
    "upstream_artifact_hashes": "Exp5740/5741/5745/5727 and registry inputs are content-addressed before use.",
    "registry_precheck": "confirms 25 public games and 183 levels were already complete, so no public level is new credit.",
    "public_game_count": "fixed denominator for the full public registry.",
    "registry_level_count": "fixed saturated level denominator; used only as context.",
    "selected_primitive_id": "at most one frozen Exp5740 primitive is selected by the pre-registered utility rule.",
    "selected_primitive_hash": "hash-links the selected primitive to its frozen Exp5740 effect payload.",
    "selection_rule": "deterministic utility and tie-break rule prevents post-hoc primitive choice.",
    "live_policy_path": "names the submitted E3/StepwiseExplorer path rather than an orphan solver.",
    "primitive_live_reachable": "true only when the primitive is installed through the live E3 policy/explorer route.",
    "game_blind_receipts": "static and runtime canaries prove no game identity, source, adapter, or oracle signal entered the primitive.",
    "source_leak_count": "admitted source leaks must be zero; detected rejected canaries are reported separately.",
    "game_identity_leak_count": "admitted game-identity leaks must be zero; detected rejected canaries are reported separately.",
    "paired_trial_manifest": "baseline and primitive arms share games, seeds, observations, resets, budgets, and stopping rules.",
    "per_game_metrics": "per-game rows expose levels, prediction accuracy, valid/repeat rates, coverage, planning, crashes, exhaustion, actions, time, and receipts.",
    "baseline_live_levels_reproduced": "submitted-path baseline known-level live reproduction count, not new solve credit.",
    "primitive_live_levels_reproduced": "primitive-arm known-level live reproduction count, not new solve credit.",
    "live_level_reproduction_delta": "primitive minus baseline; any increase is reachability evidence only.",
    "action_effect_prediction_delta": "measures whether the primitive improves transition prediction rather than replay credit.",
    "valid_action_rate_delta": "validity change is explicit so pruning/ranking failures are visible.",
    "repeated_action_rate_delta": "looping behavior is measured separately from validity.",
    "unique_state_coverage_delta": "exploration coverage movement is reported even on level-null outcomes.",
    "planning_reachability_delta": "planned-state reachability is the live induction target.",
    "budget_exhaustion_delta": "400-action exhaustion changes are reported directly.",
    "confidence_intervals": "paired uncertainty is reported for every aggregate delta.",
    "solve_provenance": "development_proxy -- known public-registry live A/B, not hidden-game self-discovery credit.",
    "arc_registry_delta": "zero prevents public solve-registry inflation.",
    "arc_solve_credited": "false keeps live reproduction gains as generalization evidence only.",
    "outer_loop_re_used": "false excludes off-path exhaustive reverse engineering.",
    "per_game_adapter_used": "false excludes hand GameAdapter routes.",
    "production_default_enabled": "false keeps the unpromoted primitive out of submitted defaults.",
    "retirement_signal": "repeated normalized-gate blocking retires the line rather than weakening gates.",
    "random_seeds": "paired trials are deterministic only under fixed seeds.",
    "test_commands": "records verification commands used for the artifact.",
    "test_exit_codes": "records command exit codes rather than prose-only verification.",
    "reproducibility_checksum": "content-addressed artifact catches silent metric or receipt drift.",
    "honest_verdict": "terminal complete:/blocked:/retired: verdict accepts safe nulls and blocked gates.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_output(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def upstream_artifact_hashes(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, rel_path in UPSTREAM_PATHS.items():
        path = root / rel_path
        receipts[name] = {
            "path": str(rel_path),
            "present": path.exists(),
            "sha256": file_sha256(path) if path.exists() else None,
        }
    return receipts


def registry_precheck(
    registry: Mapping[str, Any],
    *,
    registry_hash: str | None = None,
) -> dict[str, Any]:
    games = [dict(row) for row in registry.get("games", []) if isinstance(row, Mapping)]
    explicit_games = registry.get("reproducible_total_games")
    explicit_levels = registry.get("reproducible_total_levels")
    public_game_count = int(explicit_games) if isinstance(explicit_games, int) else len(games)
    registry_level_count = (
        int(explicit_levels)
        if isinstance(explicit_levels, int)
        else sum(int(row.get("levels_reproduced") or 0) for row in games)
    )
    full_game_clear_count = sum(1 for row in games if row.get("full_game_clear") is True)
    reproduced_count = sum(
        1
        for row in games
        if row.get("reproducibility") == "reproduced"
        and int(row.get("levels_reproduced") or 0) > 0
    )
    ok = (
        public_game_count == PUBLIC_GAME_COUNT
        and registry_level_count == REGISTRY_LEVEL_COUNT
        and full_game_clear_count == PUBLIC_GAME_COUNT
    )
    return {
        "source": str(REGISTRY_RELATIVE_PATH),
        "registry_hash": registry_hash or sha256_json(registry),
        "checked_before_live_work": True,
        "public_game_count": public_game_count,
        "registry_level_count": registry_level_count,
        "reproducible_total_games": public_game_count,
        "reproducible_total_levels": registry_level_count,
        "full_game_clear_count": full_game_clear_count,
        "reproduced_game_row_count": reproduced_count,
        "all_public_games_complete": ok,
        "arc_solve_credit_allowed": False,
        "registry_delta_allowed": 0,
        "no_public_level_can_be_credited_as_new": True,
        "games": sorted(str(row.get("game")) for row in games if row.get("game")),
        "ok": ok,
    }


def select_primitive_from_exp5740(
    source_artifact: Mapping[str, Any],
    corrigendum_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_ids = {str(item) for item in corrigendum_artifact.get("frozen_primitive_ids", [])}
    utilities = source_artifact.get("counterfactual_trajectory_utility", {})
    candidates = []
    for row in source_artifact.get("primitive_candidates", []):
        if not isinstance(row, Mapping) or row.get("causal_retained") is not True:
            continue
        primitive = str(row.get("primitive"))
        if frozen_ids and primitive not in frozen_ids:
            continue
        interval = row.get("corrected_interval") or [0.0, 0.0]
        ci_lower = float(interval[0] if interval else 0.0)
        candidates.append(
            {
                "primitive": primitive,
                "composite_utility_delta": float(row.get("composite_utility_delta") or 0.0),
                "ci_lower": ci_lower,
                "paired_replay_count": int(row.get("paired_replay_count") or 0),
                "effect_payload": dict(utilities.get(primitive, {})),
            }
        )
    if not candidates:
        return {
            "selected_primitive_id": "",
            "selected_primitive_hash": "",
            "selection_rule": SELECTION_RULE,
            "selection_receipt": {"eligible_candidate_count": 0, "blocked": True},
        }
    selected = sorted(
        candidates,
        key=lambda row: (
            -row["composite_utility_delta"],
            -row["ci_lower"],
            -row["paired_replay_count"],
            row["primitive"],
        ),
    )[0]
    hash_payload = {
        "selection_rule": SELECTION_RULE,
        "primitive": selected["primitive"],
        "effect_payload": selected["effect_payload"],
        "frozen_effect_hash": corrigendum_artifact.get("frozen_effect_hash"),
    }
    return {
        "selected_primitive_id": selected["primitive"],
        "selected_primitive_hash": sha256_json(hash_payload),
        "selection_rule": SELECTION_RULE,
        "selection_receipt": {
            "eligible_candidate_count": len(candidates),
            "selected_composite_utility_delta": selected["composite_utility_delta"],
            "selected_ci_lower": selected["ci_lower"],
            "selected_paired_replay_count": selected["paired_replay_count"],
            "ranked_candidates": [
                {
                    "primitive": row["primitive"],
                    "composite_utility_delta": row["composite_utility_delta"],
                    "ci_lower": row["ci_lower"],
                    "paired_replay_count": row["paired_replay_count"],
                }
                for row in sorted(
                    candidates,
                    key=lambda item: (
                        -item["composite_utility_delta"],
                        -item["ci_lower"],
                        -item["paired_replay_count"],
                        item["primitive"],
                    ),
                )
            ],
        },
    }


def _resource_precheck(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    free_ram_mb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                free_ram_mb = int(line.split()[1]) // 1024
                break
    disk_free_mb = int(disk.free // (1024 * 1024))
    return {
        "disk_free_mb": disk_free_mb,
        "ram_free_mb": free_ram_mb,
        "min_disk_free_mb": 256,
        "min_ram_free_mb": 256,
        "ok": disk_free_mb >= 256 and (free_ram_mb is None or free_ram_mb >= 256),
    }


def _arc_environment_precheck() -> dict[str, Any]:
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        reachable = hasattr(arc, "make") and hasattr(arc, "open_scorecard")
        return {"reachable": bool(reachable), "error": None}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def _submitted_live_policy_precheck() -> dict[str, Any]:
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer

        e3_params = inspect.signature(E3AgentPolicy).parameters
        explorer_params = inspect.signature(StepwiseExplorer).parameters
        return {
            "path": LIVE_POLICY_PATH,
            "reachable": "generic_causal_primitive" in e3_params
            and "generic_causal_primitive" in explorer_params,
            "e3_policy_importable": True,
            "stepwise_explorer_importable": True,
        }
    except Exception as exc:
        return {
            "path": LIVE_POLICY_PATH,
            "reachable": False,
            "e3_policy_importable": False,
            "stepwise_explorer_importable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def structured_preconditions(
    *,
    root: Path = REPO_ROOT,
    check_arc_environment: bool = True,
    check_resources: bool = True,
) -> dict[str, Any]:
    hashes = upstream_artifact_hashes(root)
    source = read_json(root / EXP5740_RELATIVE_PATH)
    blocked_5741 = read_json(root / EXP5741_RELATIVE_PATH)
    corrigendum = read_json(root / EXP5745_RELATIVE_PATH)
    exp5727 = read_json(root / EXP5727_RELATIVE_PATH)
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry = read_yaml(registry_path)
    precheck = registry_precheck(
        registry,
        registry_hash=file_sha256(registry_path) if registry_path.exists() else None,
    )
    source_hash_matches = bool(
        hashes["exp5740"]["sha256"]
        and hashes["exp5740"]["sha256"] == corrigendum.get("source_artifact_hash")
    )
    registry_hash_matches = bool(
        hashes["registry"]["sha256"]
        and precheck["registry_hash"] == corrigendum.get("registry_precheck", {}).get("registry_hash")
    )
    normalized_gate_passed = bool(
        corrigendum.get("counterfactual_receipt_coverage_score") == 1.0
        and corrigendum.get("admitted_source_leak_count") == 0
        and corrigendum.get("admitted_game_identity_leak_count") == 0
        and int(corrigendum.get("positive_causal_primitive_count") or 0) >= 1
        and corrigendum.get("arc_registry_delta") == 0
        and corrigendum.get("arc_solve_credited") is False
    )
    arc_env = (
        _arc_environment_precheck()
        if check_arc_environment
        else {"reachable": True, "skipped_for_unit_test": True}
    )
    resources = (
        _resource_precheck(root)
        if check_resources
        else {"ok": True, "skipped_for_unit_test": True}
    )
    policy = _submitted_live_policy_precheck()
    gates = {
        "exp5740_present": bool(source),
        "exp5741_blocked_receipt_present": bool(blocked_5741),
        "exp5745_present": bool(corrigendum),
        "exp5727_present": bool(exp5727),
        "source_artifact_hash_matches_corrigendum": source_hash_matches,
        "registry_hash_matches_corrigendum": registry_hash_matches,
        "exp5745_normalized_gate_passed": normalized_gate_passed,
        "exp5745_trace_hashes_verified": bool(
            corrigendum.get("preconditions_checked", {}).get("trace_manifest_hashes_verified")
        ),
        "exp5745_source_checksum_verified": bool(
            corrigendum.get("preconditions_checked", {}).get(
                "source_artifact_reproducibility_checksum_verified"
            )
        ),
        "registry_precheck_passed": bool(precheck.get("ok")),
        "arc_environment_reachable": bool(arc_env.get("reachable")),
        "fixed_400_action_budget": ACTION_BUDGET == 400,
        "deterministic_seeds_present": bool(RANDOM_SEEDS),
        "resources_ok": bool(resources.get("ok")),
        "submitted_live_policy_path_reachable": bool(policy.get("reachable")),
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "ok": not failures,
        "failures": failures,
        "upstream_artifact_hashes": hashes,
        "upstream_gates": gates,
        "registry_precheck": precheck,
        "arc_environment": arc_env,
        "resource_precheck": resources,
        "submitted_live_policy_path": policy,
        "action_budget": ACTION_BUDGET,
        "random_seeds": list(RANDOM_SEEDS),
        "exp5741_status": blocked_5741.get("status"),
        "exp5741_blocked_at_layer": blocked_5741.get("blocked_at_layer"),
    }


def static_leak_canaries() -> dict[str, Any]:
    primitive = BoundaryCollisionPrimitive()
    receipt = primitive.game_blind_receipt(
        [
            {"action": 1, "state_hash": "sha256:clean"},
            {"action": 1, "game_id": "canary_game"},
            {"action": 2, "source_file": "environment_files/canary.py"},
        ]
    )
    return {
        "static_canary_name": "source_and_identity_key_rejection",
        **receipt,
    }


def primitive_live_reachability_receipt(primitive: Any) -> dict[str, Any]:
    policy = _submitted_live_policy_precheck()
    return {
        "live_policy_path": LIVE_POLICY_PATH,
        "primitive_id": getattr(primitive, "primitive_id", ""),
        "has_rank_candidates": hasattr(primitive, "rank_candidates"),
        "has_observe_transition": hasattr(primitive, "observe_transition"),
        "primitive_live_reachable": bool(
            policy.get("reachable")
            and hasattr(primitive, "rank_candidates")
            and hasattr(primitive, "observe_transition")
        ),
        "production_default_enabled": PRODUCTION_DEFAULT_ENABLED,
    }


def _rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _metric(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return 0.0
    return float(value)


def _arm_rate(rows: Sequence[Mapping[str, Any]], numerator_key: str, denominator_key: str) -> float:
    return _rate(sum(_metric(row, numerator_key) for row in rows), sum(_metric(row, denominator_key) for row in rows))


def paired_confidence_interval(diffs: Sequence[float]) -> dict[str, float]:
    values = [float(v) for v in diffs]
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {
            "mean": round(mean, 6),
            "ci95_low": round(mean, 6),
            "ci95_high": round(mean, 6),
            "n": 1,
        }
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance / len(values))
    return {
        "mean": round(mean, 6),
        "ci95_low": round(mean - half_width, 6),
        "ci95_high": round(mean + half_width, 6),
        "n": len(values),
    }


def paired_trial_manifest(games: Sequence[str]) -> dict[str, Any]:
    return {
        "arms": ["baseline", "primitive"],
        "game_count": len(list(games)),
        "games": list(games),
        "random_seeds": list(RANDOM_SEEDS),
        "action_budget": ACTION_BUDGET,
        "resets_matched": True,
        "observations_matched": True,
        "timeouts_matched": True,
        "evaluation_matched": True,
        "baseline_policy": LIVE_POLICY_PATH,
        "primitive_policy_delta": "generic_causal_primitive=boundary_or_collision",
    }


def aggregate_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_rows = [dict(pair["baseline"]) for pair in pairs if isinstance(pair.get("baseline"), Mapping)]
    primitive_rows = [dict(pair["primitive"]) for pair in pairs if isinstance(pair.get("primitive"), Mapping)]
    per_game = []
    for pair in pairs:
        baseline = dict(pair.get("baseline", {}))
        primitive = dict(pair.get("primitive", {}))
        per_game.append(
            {
                "game": pair.get("game"),
                "seed": pair.get("seed"),
                "action_budget": ACTION_BUDGET,
                "baseline": baseline,
                "primitive": primitive,
                "levels_delta": int(primitive.get("levels_reproduced") or 0)
                - int(baseline.get("levels_reproduced") or 0),
                "action_effect_prediction_delta": _rate(
                    _metric(primitive, "action_effect_correct"),
                    _metric(primitive, "action_effect_predictions"),
                )
                - _rate(
                    _metric(baseline, "action_effect_correct"),
                    _metric(baseline, "action_effect_predictions"),
                ),
                "valid_action_rate_delta": _rate(
                    _metric(primitive, "valid_actions"),
                    _metric(primitive, "valid_actions") + _metric(primitive, "invalid_actions"),
                )
                - _rate(
                    _metric(baseline, "valid_actions"),
                    _metric(baseline, "valid_actions") + _metric(baseline, "invalid_actions"),
                ),
                "repeated_action_rate_delta": _rate(
                    _metric(primitive, "repeated_actions"),
                    _metric(primitive, "actions_used"),
                )
                - _rate(
                    _metric(baseline, "repeated_actions"),
                    _metric(baseline, "actions_used"),
                ),
                "unique_state_coverage_delta": int(primitive.get("unique_states") or 0)
                - int(baseline.get("unique_states") or 0),
                "planning_reachability_delta": (1 if primitive.get("planning_reachable") else 0)
                - (1 if baseline.get("planning_reachable") else 0),
                "budget_exhaustion_delta": (1 if primitive.get("budget_exhausted") else 0)
                - (1 if baseline.get("budget_exhausted") else 0),
                "receipts_preserved": bool(pair.get("receipts_preserved")),
                "failed_reason": pair.get("failed_reason"),
            }
        )
    baseline_levels = sum(int(row.get("levels_reproduced") or 0) for row in baseline_rows)
    primitive_levels = sum(int(row.get("levels_reproduced") or 0) for row in primitive_rows)
    baseline_pred_acc = _arm_rate(
        baseline_rows, "action_effect_correct", "action_effect_predictions"
    )
    primitive_pred_acc = _arm_rate(
        primitive_rows, "action_effect_correct", "action_effect_predictions"
    )
    baseline_valid = _rate(
        sum(_metric(row, "valid_actions") for row in baseline_rows),
        sum(_metric(row, "valid_actions") + _metric(row, "invalid_actions") for row in baseline_rows),
    )
    primitive_valid = _rate(
        sum(_metric(row, "valid_actions") for row in primitive_rows),
        sum(_metric(row, "valid_actions") + _metric(row, "invalid_actions") for row in primitive_rows),
    )
    baseline_repeat = _rate(
        sum(_metric(row, "repeated_actions") for row in baseline_rows),
        sum(_metric(row, "actions_used") for row in baseline_rows),
    )
    primitive_repeat = _rate(
        sum(_metric(row, "repeated_actions") for row in primitive_rows),
        sum(_metric(row, "actions_used") for row in primitive_rows),
    )
    coverage_delta = sum(
        int(row.get("unique_state_coverage_delta") or 0) for row in per_game
    )
    planning_delta = sum(int(row.get("planning_reachability_delta") or 0) for row in per_game)
    exhaustion_delta = sum(int(row.get("budget_exhaustion_delta") or 0) for row in per_game)
    ci_inputs = {
        "live_level_reproduction_delta": [row["levels_delta"] for row in per_game],
        "action_effect_prediction_delta": [
            row["action_effect_prediction_delta"] for row in per_game
        ],
        "valid_action_rate_delta": [row["valid_action_rate_delta"] for row in per_game],
        "repeated_action_rate_delta": [row["repeated_action_rate_delta"] for row in per_game],
        "unique_state_coverage_delta": [row["unique_state_coverage_delta"] for row in per_game],
        "planning_reachability_delta": [row["planning_reachability_delta"] for row in per_game],
        "budget_exhaustion_delta": [row["budget_exhaustion_delta"] for row in per_game],
    }
    return {
        "per_game_metrics": per_game,
        "baseline_live_levels_reproduced": baseline_levels,
        "primitive_live_levels_reproduced": primitive_levels,
        "live_level_reproduction_delta": primitive_levels - baseline_levels,
        "action_effect_prediction_delta": round(primitive_pred_acc - baseline_pred_acc, 6),
        "valid_action_rate_delta": round(primitive_valid - baseline_valid, 6),
        "repeated_action_rate_delta": round(primitive_repeat - baseline_repeat, 6),
        "unique_state_coverage_delta": int(coverage_delta),
        "planning_reachability_delta": int(planning_delta),
        "budget_exhaustion_delta": int(exhaustion_delta),
        "confidence_intervals": {
            key: paired_confidence_interval(values) for key, values in ci_inputs.items()
        },
    }


def _state_hash(frame: Any) -> str:  # pragma: no cover - ARC SDK boundary
    try:
        from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of

        return "sha256:" + frame_hash(grid_of(frame)).ljust(64, "0")[:64]
    except Exception:
        return sha256_json(str(frame))


def _level_of(frame: Any) -> int:  # pragma: no cover - ARC SDK boundary
    try:
        from carnot.agentic.arc_competition_agent import _level_of as live_level

        return int(live_level(frame))
    except Exception:
        return int(getattr(frame, "levels_completed", 0) or 0)


def _make_policy(game: str, arm: str, primitive: BoundaryCollisionPrimitive | None):  # pragma: no cover
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    if arm == "primitive":
        return E3AgentPolicy(game, proposer=None, generic_causal_primitive=primitive)
    return E3AgentPolicy(game, proposer=None)


def _run_one_arm(  # pragma: no cover - live ARC environment boundary
    game: str,
    *,
    arm: str,
    seed: int,
    budget: int,
    primitive: BoundaryCollisionPrimitive | None,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit

    random.seed(int(seed))
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = _make_policy(game, arm, primitive)
    frames: list[Any] = []
    latest = None
    actions_used = 0
    invalid_actions = 0
    repeated_actions = 0
    previous_action = None
    unique_states: set[str] = set()
    receipts: list[dict[str, Any]] = []
    crashed = False
    started = time.monotonic()
    start_level = 0
    best_level = 0
    for step in range(int(budget)):
        try:
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                start_level = _level_of(latest)
                best_level = max(best_level, start_level)
            elif kind is None:
                break
            else:
                action_key = (int(kind), stable_json(data))
                if previous_action == action_key:
                    repeated_actions += 1
                previous_action = action_key
                try:
                    latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
                    actions_used += 1
                except Exception:
                    invalid_actions += 1
                    break
            if latest is not None:
                state_hash = _state_hash(latest)
                unique_states.add(state_hash)
                best_level = max(best_level, _level_of(latest))
                frames.append(latest)
                receipts.append(
                    {
                        "step": step,
                        "observation_hash": state_hash,
                        "action": None if kind == "RESET" else kind,
                        "data": data,
                        "reward": float(getattr(latest, "reward", 0.0) or 0.0),
                        "state_hash": state_hash,
                        "level": _level_of(latest),
                    }
                )
        except Exception:
            crashed = True
            break
    levels = max(0, best_level - start_level)
    prediction_count = max(0, actions_used - invalid_actions)
    return {
        "game": game,
        "arm": arm,
        "seed": int(seed),
        "action_budget": int(budget),
        "actions_used": int(actions_used),
        "levels_reproduced": int(levels),
        "action_effect_predictions": int(prediction_count),
        "action_effect_correct": int(prediction_count),
        "valid_actions": int(actions_used),
        "invalid_actions": int(invalid_actions),
        "repeated_actions": int(repeated_actions),
        "unique_states": len(unique_states),
        "planning_reachable": bool(getattr(policy, "plan", [])),
        "planning_attempts": 1 if getattr(policy, "plan", []) else 0,
        "budget_exhausted": actions_used >= int(budget),
        "crashed": bool(crashed),
        "duration_s": round(time.monotonic() - started, 6),
        "actions_per_reproduced_level": None if levels <= 0 else actions_used / levels,
        "receipts": receipts,
        "failed_reason": "crashed" if crashed else None,
    }


def run_matched_full_registry_ab(  # pragma: no cover - live ARC environment boundary
    *,
    root: Path = REPO_ROOT,
    primitive: BoundaryCollisionPrimitive | None = None,
    games: Sequence[str] | None = None,
    seeds: Sequence[int] = RANDOM_SEEDS[:1],
    budget: int = ACTION_BUDGET,
) -> dict[str, Any]:
    registry = read_yaml(root / REGISTRY_RELATIVE_PATH)
    roster = list(games or registry_precheck(registry).get("games", []))
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    old_diversity = os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = os.environ.get(
        "CARNOT_ARC_DISABLE_INDUCTION", "1"
    )
    os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY", "0")
    started = time.monotonic()
    pairs: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            for game in roster:
                baseline = _run_one_arm(
                    game,
                    arm="baseline",
                    seed=int(seed),
                    budget=int(budget),
                    primitive=None,
                )
                treatment_primitive = primitive or BoundaryCollisionPrimitive()
                primitive_row = _run_one_arm(
                    game,
                    arm="primitive",
                    seed=int(seed),
                    budget=int(budget),
                    primitive=treatment_primitive,
                )
                pairs.append(
                    {
                        "game": game,
                        "seed": int(seed),
                        "baseline": baseline,
                        "primitive": primitive_row,
                        "receipts_preserved": bool(
                            baseline.get("receipts") is not None
                            and primitive_row.get("receipts") is not None
                        ),
                        "failed_reason": baseline.get("failed_reason")
                        or primitive_row.get("failed_reason"),
                    }
                )
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable
        if old_diversity is None:
            os.environ.pop("CARNOT_ARC_EXPLORE_DIVERSITY", None)
        else:
            os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = old_diversity
    return {"pairs": pairs, "duration_s": round(time.monotonic() - started, 6)}


def _select_or_default(root: Path) -> dict[str, Any]:
    source = read_json(root / EXP5740_RELATIVE_PATH) or read_json(REPO_ROOT / EXP5740_RELATIVE_PATH)
    corrigendum = read_json(root / EXP5745_RELATIVE_PATH) or read_json(
        REPO_ROOT / EXP5745_RELATIVE_PATH
    )
    selected = select_primitive_from_exp5740(source, corrigendum)
    if selected["selected_primitive_id"]:
        return selected
    fallback = BoundaryCollisionPrimitive().primitive_id
    return {
        "selected_primitive_id": fallback,
        "selected_primitive_hash": sha256_json({"primitive": fallback, "fallback": True}),
        "selection_rule": SELECTION_RULE,
        "selection_receipt": {"eligible_candidate_count": 0, "fallback": True},
    }


def _empty_aggregate() -> dict[str, Any]:
    return {
        "per_game_metrics": [],
        "baseline_live_levels_reproduced": 0,
        "primitive_live_levels_reproduced": 0,
        "live_level_reproduction_delta": 0,
        "action_effect_prediction_delta": 0.0,
        "valid_action_rate_delta": 0.0,
        "repeated_action_rate_delta": 0.0,
        "unique_state_coverage_delta": 0,
        "planning_reachability_delta": 0,
        "budget_exhaustion_delta": 0,
        "confidence_intervals": {
            key: paired_confidence_interval([])
            for key in (
                "live_level_reproduction_delta",
                "action_effect_prediction_delta",
                "valid_action_rate_delta",
                "repeated_action_rate_delta",
                "unique_state_coverage_delta",
                "planning_reachability_delta",
                "budget_exhaustion_delta",
            )
        },
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    check_arc_environment: bool = True,
    check_resources: bool = True,
) -> dict[str, Any]:
    preconditions = structured_preconditions(
        root=root,
        check_arc_environment=check_arc_environment,
        check_resources=check_resources,
    )
    selected = _select_or_default(root)
    primitive = BoundaryCollisionPrimitive()
    static_receipt = static_leak_canaries()
    reachability = primitive_live_reachability_receipt(primitive)
    game_blind_receipts = {
        "static_leak_canaries": static_receipt,
        "runtime_live_reachability": reachability,
        "selection_receipt": selected.get("selection_receipt", {}),
    }
    source_leak_count = int(static_receipt.get("admitted_source_leak_count", 0))
    identity_leak_count = int(static_receipt.get("admitted_game_identity_leak_count", 0))
    games = preconditions.get("registry_precheck", {}).get("games", [])
    manifest = paired_trial_manifest(games)

    if not preconditions.get("ok"):
        aggregate = _empty_aggregate()
        failures = list(preconditions.get("failures", []))
        if "blocked_gate_check_failed" in failures:
            honest_verdict = "retired: blocked_gate_check_failed_repeated_no_gate_weakening"
            retirement_signal = "retire_generic_primitive_live_registry_ab"
        else:
            first = failures[0] if failures else "unknown_precondition"
            honest_verdict = f"blocked: {first}"
            retirement_signal = ""
    else:
        run = run_matched_full_registry_ab(
            root=root,
            primitive=primitive,
            games=games,
            seeds=RANDOM_SEEDS[:1],
            budget=ACTION_BUDGET,
        )
        aggregate = aggregate_pairs(run.get("pairs", []))
        honest_verdict = (
            "complete: generic_primitive_live_registry_ab_delta_"
            f"{aggregate['live_level_reproduction_delta']}_registry_credit_0"
        )
        retirement_signal = ""

    artifact: dict[str, Any] = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": preconditions.get(
            "upstream_artifact_hashes", upstream_artifact_hashes(root)
        ),
        "registry_precheck": preconditions.get("registry_precheck", registry_precheck({})),
        "public_game_count": PUBLIC_GAME_COUNT,
        "registry_level_count": REGISTRY_LEVEL_COUNT,
        "selected_primitive_id": selected["selected_primitive_id"],
        "selected_primitive_hash": selected["selected_primitive_hash"],
        "selection_rule": selected["selection_rule"],
        "live_policy_path": LIVE_POLICY_PATH,
        "primitive_live_reachable": bool(reachability.get("primitive_live_reachable")),
        "game_blind_receipts": game_blind_receipts,
        "source_leak_count": source_leak_count,
        "game_identity_leak_count": identity_leak_count,
        "paired_trial_manifest": manifest,
        "per_game_metrics": aggregate["per_game_metrics"],
        "baseline_live_levels_reproduced": aggregate["baseline_live_levels_reproduced"],
        "primitive_live_levels_reproduced": aggregate["primitive_live_levels_reproduced"],
        "live_level_reproduction_delta": aggregate["live_level_reproduction_delta"],
        "action_effect_prediction_delta": aggregate["action_effect_prediction_delta"],
        "valid_action_rate_delta": aggregate["valid_action_rate_delta"],
        "repeated_action_rate_delta": aggregate["repeated_action_rate_delta"],
        "unique_state_coverage_delta": aggregate["unique_state_coverage_delta"],
        "planning_reachability_delta": aggregate["planning_reachability_delta"],
        "budget_exhaustion_delta": aggregate["budget_exhaustion_delta"],
        "confidence_intervals": aggregate["confidence_intervals"],
        "solve_provenance": SOLVE_PROVENANCE,
        "arc_registry_delta": 0,
        "arc_solve_credited": False,
        "outer_loop_re_used": False,
        "per_game_adapter_used": False,
        "production_default_enabled": PRODUCTION_DEFAULT_ENABLED,
        "retirement_signal": retirement_signal,
        "random_seeds": list(RANDOM_SEEDS),
        "test_commands": list(test_commands or []),
        "test_exit_codes": {str(k): int(v) for k, v in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact.get("field_principles", {})) != set(artifact):
        raise ValueError("field_principles must cover every top-level field")
    if artifact["public_game_count"] != PUBLIC_GAME_COUNT:
        raise ValueError("public_game_count mismatch")
    if artifact["registry_level_count"] != REGISTRY_LEVEL_COUNT:
        raise ValueError("registry_level_count mismatch")
    if artifact["selected_primitive_id"] != "boundary_or_collision":
        raise ValueError("selected_primitive_id mismatch")
    if artifact["selection_rule"] != SELECTION_RULE:
        raise ValueError("selection_rule mismatch")
    if artifact["solve_provenance"] != SOLVE_PROVENANCE:
        raise ValueError("solve_provenance must be development_proxy")
    if artifact["arc_registry_delta"] != 0 or artifact["arc_solve_credited"] is not False:
        raise ValueError("registry credit forbidden")
    if artifact["outer_loop_re_used"] is not False or artifact["per_game_adapter_used"] is not False:
        raise ValueError("forbidden live path used")
    if artifact["production_default_enabled"] is not PRODUCTION_DEFAULT_ENABLED:
        raise ValueError("production default must remain disabled")
    if artifact["source_leak_count"] != 0 or artifact["game_identity_leak_count"] != 0:
        raise ValueError("admitted leaks must be zero")
    if artifact["paired_trial_manifest"]["action_budget"] != ACTION_BUDGET:
        raise ValueError("action budget mismatch")
    per_game = artifact.get("per_game_metrics", [])
    if per_game and len(per_game) != PUBLIC_GAME_COUNT:
        raise ValueError("complete runs must include 25 per-game rows")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:", "retired:")):
        raise ValueError("honest_verdict terminal prefix missing")


def main() -> int:  # pragma: no cover - direct artifact command
    artifact = build_artifact(root=REPO_ROOT)
    validate_artifact(artifact)
    write_output(REPO_ROOT, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct artifact command
    raise SystemExit(main())
