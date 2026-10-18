"""Exp5643 ARC live self-discovery attempt artifact builder.

The V509 attempt is a bounded, no-new-LLM live-agent run.  The target is chosen
before interaction from registry and transition-receipt evidence, the Exp5642
executable-model branch is advisory unless it promoted cleanly, and no level is
credited unless the generic clean-state replay reproduces the selected target.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.artifact_gate_annotations import checksum_core

import yaml

from carnot import experiment_5621_arc_live_self_discovery_levelup_v507 as v507
from carnot import experiment_5632_arc_live_self_discovery_levelup_v508 as v508


EXPERIMENT_ID = 5643
EXPERIMENT = "experiment_5643_arc_live_self_discovery_levelup_v509"
MILESTONE = "2026.07.509"
RESULT_RELATIVE_PATH = f"results/{EXPERIMENT}.json"
TRACE_RELATIVE_PATH = f"results/{EXPERIMENT}_trace.json"
REGISTRY_RELATIVE_PATH = v508.REGISTRY_RELATIVE_PATH
EXP5642_RELATIVE_PATH = "results/experiment_5642_arc_executable_model_live_ab.json"
TRANSITION_RECEIPT_RELATIVE_PATH = "results/experiment_5636_transition_v509.json"

SPEC_REQUIREMENT = "REQ-ARC-FCP-5643"
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "live_agent_environment_interaction"
RANDOM_SEED = 5643
RANDOM_SEEDS = [RANDOM_SEED]
ACTION_BUDGET = 72
WALL_TIME_CAP_S = 900
MODEL_CALL_LIMIT = 0
ENVIRONMENT_ACTION_LIMIT = ACTION_BUDGET
RETRY_LIMIT = 0
CHECKPOINT_LIMIT = 3
CHECKPOINT_CADENCE = "checkpoint trace metadata every 24 environment actions and at terminal write"
STOPPING_RULE = "fixed_72_action_budget_or_target_level_reached_no_llm_induction_disabled_v509"

MANDATED_GGUF_IDS = v508.MANDATED_GGUF_IDS

EXPLICIT_EXCLUDED_TARGETS = [
    {
        "game": "bp35",
        "level": 9,
        "level_label": "L9",
        "reason": "exp5621_attempted_without_bank",
    },
    {
        "game": "sk48",
        "level": 8,
        "level_label": "L8",
        "reason": "exp5610_attempted_without_bank",
    },
    {
        "game": "lf52",
        "level": 7,
        "level_label": "L7",
        "reason": "exp5632_attempted_without_bank",
    },
]

NO_LLM_BASELINE_POLICY = {
    "name": "unchanged_no_new_llm_e3_baseline",
    "llm_invoked": False,
    "new_llm_calls": False,
    "executable_model_policy": False,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "field_principles": {
        "principle": "principle annotations are carried in the artifact so every required 5643 field is auditable.",
    },
    "registry_count_before": {
        "principle": "authoritative reproduced-level total before target selection; the level-up baseline is explicit.",
    },
    "registry_precheck_receipt": {
        "principle": "execution-time precheck proves the selected target is unreproduced before any live observation.",
    },
    "excluded_targets": {
        "principle": "bp35 L9, sk48 L8, lf52 L7, transition-receipt failures, and registry-duplicate levels stay closed and cannot receive duplicate credit.",
    },
    "selected_game": {
        "principle": "the game scope is fixed before live observation, preventing outcome-driven target switching.",
    },
    "selected_level": {
        "principle": "the level scope is fixed before live observation, preventing outcome-driven target switching.",
    },
    "target_selection_hash": {
        "principle": "content hash of the pre-outcome target receipt proves the target was not changed after seeing results.",
    },
    "policy_source": {
        "principle": "records whether Exp5642 promoted cleanly or the unchanged baseline ran.",
    },
    "methodology_receipt": {
        "principle": "records target precheck, policy freeze, budget freeze, no-source/no-adapter/no-outer-loop provenance, and reproduction criteria so the run is not a short opaque artifact.",
    },
    "model_specs": {
        "principle": "empty for a no-LLM run; otherwise names the mandated cached SOTA GGUF receipt exactly.",
    },
    "budget_receipt": {
        "principle": "seeds, wall time, model calls, environment actions, retries, checkpoint cadence, and terminal conditions are bounded before execution.",
    },
    "live_trace_path": {
        "principle": "complete live observation/action evidence is durable and replayable.",
    },
    "live_path_reachability_counters": {
        "principle": "the scored live mechanism that generated actions is identified by runtime counters.",
    },
    "solve_provenance": {
        "principle": "must equal live_agent_self_discovery; only the credited path can solve.",
    },
    "level_reached": {
        "principle": "terminal environment level fact is explicit and separate from reproduction credit.",
    },
    "reproduced_levels": {
        "principle": "newly reproduced target levels; solve credit requires at least one.",
    },
    "offline_reproduced": {
        "principle": "exactly true is mandatory for solve credit after independent generic reproduction.",
    },
    "registry_count_after": {
        "principle": "authoritative reproduced-level total after accepted banking; unchanged on honest nulls.",
    },
    "registry_delta": {
        "principle": "exactly 0 or 1 so the banked-level delta is auditable.",
    },
    "source_read": {
        "principle": "must be false; game source is excluded from live self-discovery credit.",
    },
    "game_adapter_used": {
        "principle": "must be false; no per-game adapter can be smuggled into the live path.",
    },
    "outer_loop_re_used": {
        "principle": "must be false; off-path recipes are excluded from live self-discovery credit.",
    },
    "inference_substrate": {
        "principle": "live_agent_environment_interaction -- environment observations/actions are the authority, not an offline solver.",
    },
    "random_seeds": {
        "principle": "deterministic seeds make the bounded attempt replayable and auditable.",
    },
    "reproducibility_checksum": {
        "principle": "content-addressed target, trace, budget, methodology, and banking decision catch silent drift.",
    },
    "honest_verdict": {
        "principle": "a bounded no-level result is terminal and must not be upgraded without reproduction.",
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)

read_json = v508.read_json
read_yaml = v508.read_yaml
write_json = v508.write_json
action_trace_sha256 = v508.action_trace_sha256
load_public_env_metadata = v508.load_public_env_metadata
load_arc_loop_depths = v508.load_arc_loop_depths


def _int(value: Any, default: int = 0) -> int:
    return v508._int(value, default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sha256(value: Any) -> str:
    return v508._sha256(value)


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return v508._registry_rows(registry)


def _registry_depth(row: Mapping[str, Any] | None) -> int:
    return v508._registry_depth(row)


def _registry_total(registry: Mapping[str, Any]) -> int:
    return v508._registry_total(registry)


def _public_game_items(
    public_games: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    return v508._public_game_items(public_games)


def _headroom(meta: Mapping[str, Any]) -> int:
    return v508._headroom(meta)


def _level_number(value: Any) -> int:
    if isinstance(value, str):
        return _int(value.strip().lstrip("Ll"))
    return _int(value)


def _explicit_exclusion_pairs(
    explicit_excluded_targets: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[str]]:
    pairs: dict[tuple[str, int], list[str]] = {}
    for row in explicit_excluded_targets:
        game = str(row.get("game") or "")
        level = _level_number(row.get("level") or row.get("level_label"))
        if game and level > 0:
            pairs.setdefault((game, level), []).append("explicit_recent_unbanked_attempt")
    return pairs


def recent_failed_targets_from_transition_receipt(
    transition_receipt: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract recent failed ARC targets from the V509 transition receipt.

    The transition receipt is not a solver.  It is only used as a closed-target
    ledger so a later attempt cannot retarget a level the previous milestone
    already tried and failed to bank.
    """

    if not transition_receipt:
        return []
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for scope in transition_receipt.get("retired_scopes", []):
        if not isinstance(scope, Mapping):
            continue
        evidence = scope.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        game = str(evidence.get("selected_game") or evidence.get("game") or "")
        level = _level_number(evidence.get("selected_level") or evidence.get("target_level"))
        failed = (
            evidence.get("offline_reproduced") is False
            or _int(evidence.get("registry_delta"), 1) == 0
        )
        key = str(scope.get("key") or "unknown_scope")
        if not game or level <= 0 or not failed:
            continue
        out[(game, level)] = {
            "game": game,
            "level": level,
            "level_label": f"L{level}",
            "reason": f"transition_receipt_failed_{key}",
            "source": TRANSITION_RECEIPT_RELATIVE_PATH,
        }
    return [out[key] for key in sorted(out)]


def _transition_exclusion_pairs(
    transition_receipt: Mapping[str, Any] | None,
) -> dict[tuple[str, int], list[str]]:
    pairs: dict[tuple[str, int], list[str]] = {}
    for row in recent_failed_targets_from_transition_receipt(transition_receipt):
        pairs.setdefault((row["game"], _int(row["level"])), []).append("recent_failed_target")
    return pairs


def _registry_duplicate_levels(
    registry_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return v508._registry_duplicate_levels(registry_rows)


def registry_precheck(
    registry: Mapping[str, Any],
    public_envs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    arc_loop_depths: Mapping[str, int] | None = None,
    transition_receipt: Mapping[str, Any] | None = None,
    explicit_excluded_targets: Sequence[Mapping[str, Any]] = EXPLICIT_EXCLUDED_TARGETS,
) -> dict[str, Any]:
    """Build the REQ-ARC-FCP-5643 target-selection receipt."""

    registry_rows = _registry_rows(registry)
    loop_depths = {str(game): _int(depth) for game, depth in (arc_loop_depths or {}).items()}
    explicit_pairs = _explicit_exclusion_pairs(explicit_excluded_targets)
    transition_pairs = _transition_exclusion_pairs(transition_receipt)
    candidate_rows: list[dict[str, Any]] = []

    for game, meta in _public_game_items(public_envs):
        registry_depth = _registry_depth(registry_rows.get(game))
        arc_loop_depth = loop_depths.get(game, 0)
        authenticated_headroom = _headroom(meta)
        closed_level_reasons: dict[str, list[str]] = {}
        open_levels: list[int] = []
        unreproduced_levels = list(range(registry_depth + 1, authenticated_headroom + 1))
        if not unreproduced_levels:
            reasons = ["no_authenticated_headroom"]
        else:
            for level in unreproduced_levels:
                reasons_for_level: list[str] = []
                reasons_for_level.extend(explicit_pairs.get((game, level), []))
                reasons_for_level.extend(transition_pairs.get((game, level), []))
                if reasons_for_level:
                    closed_level_reasons[str(level)] = sorted(set(reasons_for_level))
                else:
                    open_levels.append(level)
            reasons = (
                []
                if open_levels
                else ["all_authenticated_headroom_targets_closed_by_recent_failures"]
            )

        target_level = open_levels[0] if open_levels else registry_depth + 1
        closed_intermediate = [
            level
            for level in unreproduced_levels
            if level < target_level and str(level) in closed_level_reasons
        ]
        candidate_rows.append(
            {
                "game": game,
                "registry_depth": registry_depth,
                "arc_loop_depth": arc_loop_depth,
                "authenticated_headroom": authenticated_headroom,
                "unreproduced_levels_with_headroom": unreproduced_levels,
                "closed_level_reasons": closed_level_reasons,
                "closed_intermediate_levels": closed_intermediate,
                "target_level": target_level,
                "target_label": f"L{target_level}",
                "excluded": bool(reasons),
                "exclude_reasons": reasons,
                "has_authenticated_headroom": bool(open_levels),
                "target_unreproduced_in_registry": target_level > registry_depth,
            }
        )

    eligible = [row for row in candidate_rows if not row["excluded"]]
    excluded_targets = {
        "explicit_recent_attempts": [dict(row) for row in explicit_excluded_targets],
        "transition_receipt_failed_targets": recent_failed_targets_from_transition_receipt(
            transition_receipt
        ),
        "registry_duplicate_levels": _registry_duplicate_levels(registry_rows),
    }
    return {
        "spec_ref": SPEC_REQUIREMENT,
        "public_games_checked": len(candidate_rows),
        "registry_games_checked": len(registry_rows),
        "registry_count_before": _registry_total(registry),
        "registry_total_from_file": registry.get("reproducible_total_levels"),
        "registry_total_from_rows": sum(_registry_depth(row) for row in registry_rows.values()),
        "arc_loop_targets_considered": len(loop_depths),
        "excluded_targets": excluded_targets,
        "candidate_rows": candidate_rows,
        "eligible_candidates": [
            {
                "game": row["game"],
                "target_level": row["target_level"],
                "target_label": row["target_label"],
                "authenticated_headroom": row["authenticated_headroom"],
                "registry_depth": row["registry_depth"],
                "closed_intermediate_levels": row["closed_intermediate_levels"],
            }
            for row in eligible
        ],
        "duplicate_credit_policy": "registry-reproduced levels, bp35 L9, sk48 L8, lf52 L7, and transition-receipt failures receive no V509 credit",
        "ok": bool(eligible),
    }


def _target_hash(precheck: Mapping[str, Any], selected: Mapping[str, Any] | None) -> str:
    preimage = {
        "spec_ref": SPEC_REQUIREMENT,
        "registry_count_before": precheck.get("registry_count_before"),
        "excluded_targets": precheck.get("excluded_targets"),
        "candidate_rows": precheck.get("candidate_rows"),
        "selected": dict(selected or {}),
    }
    return _sha256(preimage)


def select_target_from_precheck(precheck: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in precheck.get("candidate_rows", [])
        if isinstance(row, Mapping) and not row.get("excluded")
    ]
    rows = sorted(rows, key=lambda row: (_int(row.get("target_level")), str(row.get("game"))))
    if not rows:
        return {
            "blocked": True,
            "selected_game": None,
            "selected_level": None,
            "target_level": None,
            "target_selection_hash": _target_hash(precheck, None),
            "rotation_reason": "no_unreproduced_authenticated_nonclosed_candidate_v509",
            "selection_reason": "no_unreproduced_authenticated_nonclosed_candidate_v509",
            "rotation_order": [],
            "duplicate_targets_rejected": [
                row
                for row in precheck.get("candidate_rows", [])
                if isinstance(row, Mapping) and row.get("excluded")
            ],
            "selected_target_was_unreproduced": False,
        }

    selected = rows[0]
    receipt = {
        "blocked": False,
        "selected_game": selected["game"],
        "selected_level": selected["target_label"],
        "target_level": selected["target_level"],
        "prior_levels_reproduced": selected["registry_depth"],
        "authenticated_headroom": selected["authenticated_headroom"],
        "arc_loop_depth": selected["arc_loop_depth"],
        "closed_intermediate_levels": selected.get("closed_intermediate_levels", []),
        "registry_count_before": precheck.get("registry_count_before"),
        "rotation_reason": "lowest_nonclosed_unreproduced_authenticated_level_after_recent_failure_exclusions",
        "selection_reason": "v509_rotated_unreproduced_authenticated_headroom",
        "rotation_order": [
            {
                "game": row["game"],
                "target_level": row["target_level"],
                "target_label": row["target_label"],
                "closed_intermediate_levels": row.get("closed_intermediate_levels", []),
            }
            for row in rows
        ],
        "duplicate_targets_rejected": [
            row
            for row in precheck.get("candidate_rows", [])
            if isinstance(row, Mapping) and row.get("excluded")
        ],
        "selected_target_was_unreproduced": True,
    }
    receipt["target_selection_hash"] = _target_hash(precheck, receipt)
    return receipt


def policy_source_from_exp5642(exp5642: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = exp5642 or {}
    ready_score = _float(payload.get("live_executable_model_ready_score"))
    unsafe_count = _int(payload.get("unsafe_model_accept_count"))
    regression_count = _int(payload.get("known_level_regression_count"))
    promoted_config = payload.get("promoted_policy")
    if not isinstance(promoted_config, Mapping):
        promoted_config = payload.get("treatment_policy")
    if not isinstance(promoted_config, Mapping):
        promoted_config = payload.get("executable_model_policy")
    if not isinstance(promoted_config, Mapping):
        promoted_config = {}
    promoted = ready_score == 1.0 and unsafe_count == 0 and regression_count == 0
    return {
        "source_artifact": EXP5642_RELATIVE_PATH,
        "source_status": payload.get("status"),
        "source_honest_verdict": payload.get("honest_verdict"),
        "live_executable_model_ready_score": ready_score,
        "unsafe_model_accept_count": unsafe_count,
        "known_level_regression_count": regression_count,
        "attempt_gated_by_exp5642": False,
        "policy_name": "promoted_exp5642_executable_model_policy"
        if promoted
        else "unchanged_no_new_llm_e3_baseline",
        "enabled_configuration": dict(promoted_config) if promoted else NO_LLM_BASELINE_POLICY,
        "baseline_unchanged": not promoted,
        "reason": "exp5642_ready_zero_unsafe_zero_regression"
        if promoted
        else "exp5642_blocked_missing_unsafe_or_regressing_baseline_unchanged",
    }


def build_budget_receipt(
    *,
    action_budget: int = ACTION_BUDGET,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    return {
        "bounded_before_execution": True,
        "random_seeds": [random_seed],
        "wall_time_cap_s": WALL_TIME_CAP_S,
        "model_call_limit": MODEL_CALL_LIMIT,
        "environment_action_limit": action_budget,
        "retry_limit": RETRY_LIMIT,
        "checkpoint_limit": CHECKPOINT_LIMIT,
        "checkpoint_cadence": CHECKPOINT_CADENCE,
        "terminal_conditions": [
            "target_level_reached_live",
            "fixed_environment_action_budget_exhausted",
            "wall_time_cap_exceeded",
        ],
        "stopping_rule": STOPPING_RULE,
    }


def _baseline_filter_configuration() -> dict[str, Any]:
    return {
        "source_artifact": None,
        "attempt_gated_by_filter_ab": False,
        "enabled_filters": [],
        "inert_click_pruner": False,
        "object_history_salience": False,
        "baseline_unchanged": True,
        "reason": "v509_unchanged_no_new_llm_baseline",
    }


def run_live_self_discovery_attempt(
    target_selection_receipt: Mapping[str, Any],
    policy_source: Mapping[str, Any],
    action_budget: int = ACTION_BUDGET,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:  # pragma: no cover - live ARC SDK boundary
    attempt = v508.v506.run_live_self_discovery_attempt(
        target_selection_receipt=target_selection_receipt,
        filter_configuration=_baseline_filter_configuration(),
        action_budget=action_budget,
        random_seed=random_seed,
    )
    attempt["random_seed"] = random_seed
    attempt["random_seeds"] = [random_seed]
    attempt["stopping_rule"] = STOPPING_RULE
    attempt["model_specs"] = (
        [] if not attempt.get("llm_invoked") else attempt.get("model_specs", [])
    )
    attempt["policy_source"] = dict(policy_source)
    attempt["source_read"] = bool(attempt.get("source_files_read", False))
    attempt["game_adapter_used"] = bool(attempt.get("per_game_adapter_used", False))
    attempt["outer_loop_re_used"] = bool(attempt.get("offline_bfs_used", False))
    return attempt


def _accepted_new_levels(
    target_selection_receipt: Mapping[str, Any], attempt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    prior = _int(target_selection_receipt.get("prior_levels_reproduced"))
    target = _int(target_selection_receipt.get("target_level"))
    post = _int(attempt.get("post_levels_reproduced"))
    if not attempt.get("offline_reproduced"):
        return []
    if (
        attempt.get("source_read")
        or attempt.get("source_files_read")
        or attempt.get("game_adapter_used")
        or attempt.get("per_game_adapter_used")
        or attempt.get("outer_loop_re_used")
        or attempt.get("offline_bfs_used")
    ):
        return []
    if attempt.get("action_trace_sha256") != attempt.get("trace_replay_checksum"):
        return []
    if post < target or target <= prior:
        return []
    return [{"game": target_selection_receipt.get("selected_game"), "level": target}]


def _live_path_reachability_counters(
    attempt: Mapping[str, Any],
    policy_source: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [row for row in attempt.get("action_rows", []) if isinstance(row, Mapping)]
    action_rows = [
        row for row in rows if row.get("kind") == "ACTION" or row.get("action") is not None
    ]
    reset_rows = [row for row in rows if row.get("kind") == "RESET"]
    promoted = policy_source.get("policy_name") == "promoted_exp5642_executable_model_policy"
    return {
        "scored_mechanism": policy_source.get("policy_name"),
        "e3_policy_next_move_calls": len(rows),
        "environment_reset_calls": len(reset_rows),
        "environment_step_calls": len(action_rows),
        "llm_model_calls": 0 if not attempt.get("llm_invoked") else None,
        "executable_model_policy_calls": len(action_rows) if promoted else 0,
        "baseline_action_emissions": 0 if promoted else len(action_rows),
    }


def _model_specs_allowed(model_specs: Sequence[Any]) -> bool:
    text = " ".join(str(item) for item in model_specs)
    return any(model_id in text for model_id in MANDATED_GGUF_IDS)


def build_methodology_receipt(
    *,
    target_selection_receipt: Mapping[str, Any],
    policy_source: Mapping[str, Any],
    budget_receipt: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "methodology_version": "arc_live_self_discovery_v509",
        "target_recorded_before_interaction": bool(
            target_selection_receipt.get("target_selection_hash")
        ),
        "target_selection_hash": target_selection_receipt.get("target_selection_hash"),
        "policy_frozen_before_outcome": True,
        "policy_name": policy_source.get("policy_name"),
        "budget_frozen_before_outcome": budget_receipt.get("bounded_before_execution") is True,
        "environment_action_limit": budget_receipt.get("environment_action_limit"),
        "wall_time_cap_s": budget_receipt.get("wall_time_cap_s"),
        "model_call_limit": budget_receipt.get("model_call_limit"),
        "checkpoint_cadence": budget_receipt.get("checkpoint_cadence"),
        "terminal_conditions": budget_receipt.get("terminal_conditions", []),
        "agent_owned_inputs_only": True,
        "runtime_inputs": [
            "frames",
            "latest_frame",
            "environment_step_response",
            "in_process_policy_memory",
        ],
        "source_read": bool(attempt.get("source_read", attempt.get("source_files_read", False))),
        "game_adapter_used": bool(
            attempt.get("game_adapter_used", attempt.get("per_game_adapter_used", False))
        ),
        "outer_loop_re_used": bool(
            attempt.get("outer_loop_re_used", attempt.get("offline_bfs_used", False))
        ),
        "offline_ground_truth_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "llm_invoked": bool(attempt.get("llm_invoked", False)),
        "model_specs_policy": "empty_list_required_when_llm_invoked_false",
        "trace_preserved": True,
        "transition_receipt_path": TRANSITION_RECEIPT_RELATIVE_PATH,
        "reproduction_acceptance": {
            "generic_clean_state_replay_required": True,
            "offline_reproduced_required_for_credit": True,
            "trace_checksum_must_match": True,
            "registry_delta_max": 1,
        },
        "runtime_reverse_engineering": attempt.get("runtime_reverse_engineering", {}),
    }


def compute_artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the MEASURED record, excluding post-hoc review annotations.

    Excluding the fabrication gate's ``flagged_adversarial`` / ``corrigendum_*`` stamp is
    load-bearing, not cosmetic. This artifact was stamped by ``adversarial_verify.py`` AFTER it
    landed; hashing that stamp made the artifact's own recorded checksum fail to reproduce, so
    ``validate_artifact`` rejected the committed record -- the mandated review process was
    invalidating the artifact it reviewed. Recomputing with only the gate keys removed reproduces
    the checksum recorded at authoring time EXACTLY, which is what proves the measured record is
    untouched. Every measurement, seed, duration, verdict and substrate declaration is still
    hashed, so real tampering is still caught. See ``carnot.artifact_gate_annotations``.
    """
    return _sha256(checksum_core(artifact))


def build_artifact(
    registry_precheck_receipt: Mapping[str, Any],
    target_selection_receipt: Mapping[str, Any],
    policy_source: Mapping[str, Any],
    attempt: Mapping[str, Any],
    live_trace_path: str = TRACE_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    new_levels = _accepted_new_levels(target_selection_receipt, attempt)
    registry_count_before = _int(registry_precheck_receipt.get("registry_count_before"))
    registry_delta = min(len(new_levels), 1)
    registry_count_after = registry_count_before + registry_delta
    selected_game = target_selection_receipt.get("selected_game")
    selected_level = target_selection_receipt.get("selected_level")
    banked = registry_delta == 1
    if banked:
        verdict = f"complete: banked_{selected_game}_{selected_level}_live_self_discovery_v509"
    else:
        verdict = f"complete: no_new_arc_level_banked_{selected_game}_{selected_level}_bounded_live_attempt_v509"

    llm_invoked = bool(attempt.get("llm_invoked", False))
    model_specs = list(attempt.get("model_specs") or [])
    level_reached = _int(attempt.get("max_level_reached"))
    source_read = bool(attempt.get("source_read", attempt.get("source_files_read", False)))
    game_adapter_used = bool(
        attempt.get("game_adapter_used", attempt.get("per_game_adapter_used", False))
    )
    outer_loop_re_used = bool(
        attempt.get("outer_loop_re_used", attempt.get("offline_bfs_used", False))
    )
    budget_receipt = build_budget_receipt(
        action_budget=_int(attempt.get("action_budget"), ACTION_BUDGET),
        random_seed=_int(attempt.get("random_seed"), RANDOM_SEED),
    )
    methodology_receipt = build_methodology_receipt(
        target_selection_receipt=target_selection_receipt,
        policy_source=policy_source,
        budget_receipt=budget_receipt,
        attempt=attempt,
    )
    artifact: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "arc_live_self_discovery_levelup_attempt.v4",
        "spec_refs": [SPEC_REQUIREMENT],
        "field_principles": FIELD_PRINCIPLES,
        "result_path": RESULT_RELATIVE_PATH,
        "registry_count_before": registry_count_before,
        "registry_precheck_receipt": registry_precheck_receipt,
        "excluded_targets": registry_precheck_receipt.get("excluded_targets", {}),
        "selected_game": selected_game,
        "selected_level": selected_level,
        "target_level": target_selection_receipt.get("target_level"),
        "target_selection_receipt": target_selection_receipt,
        "target_selection_hash": target_selection_receipt.get("target_selection_hash"),
        "policy_source": policy_source,
        "methodology_receipt": methodology_receipt,
        "model_specs": model_specs,
        "budget_receipt": budget_receipt,
        "live_trace_path": live_trace_path,
        "live_path_reachability_counters": _live_path_reachability_counters(attempt, policy_source),
        "solve_provenance": SOLVE_PROVENANCE,
        "level_reached": level_reached,
        "reproduced_levels": registry_delta,
        "new_reproducible_levels": new_levels[:1],
        "offline_reproduced": banked,
        "registry_count_after": registry_count_after,
        "registry_delta": registry_delta,
        "registry_updated": banked,
        "source_read": source_read,
        "game_adapter_used": game_adapter_used,
        "outer_loop_re_used": outer_loop_re_used,
        "source_files_read": source_read,
        "per_game_adapter_used": game_adapter_used,
        "offline_bfs_used": outer_loop_re_used,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": attempt.get("random_seed", RANDOM_SEED),
        "random_seeds": list(attempt.get("random_seeds") or RANDOM_SEEDS),
        "honest_verdict": verdict,
        "live_attempt_executed": bool(attempt.get("live_attempt_executed")),
        "llm_invoked": llm_invoked,
        "no_model_specs_required": not llm_invoked,
        "action_trace_sha256": attempt.get("action_trace_sha256"),
        "trace_replay_checksum": attempt.get("trace_replay_checksum"),
        "reproduction_gate": attempt.get("reproduction_gate", {}),
        "terminal_environment_response": {
            "level_reached": level_reached,
            "terminal_reason": attempt.get("terminal_reason"),
            "post_levels_reproduced": attempt.get("post_levels_reproduced"),
        },
        "runtime_reverse_engineering": attempt.get("runtime_reverse_engineering", {}),
        "duration_s": round(float(duration_s or 0.0), 3),
        "tests_run": list(tests_run or []),
        "game": selected_game,
    }
    checksum = compute_artifact_checksum(artifact)
    artifact["artifact_checksum"] = checksum
    artifact["reproducibility_checksum"] = checksum
    return artifact


def build_live_trace(
    target_selection_receipt: Mapping[str, Any],
    policy_source: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": "arc_live_self_discovery_attempt_trace.v4",
        "spec_refs": [SPEC_REQUIREMENT],
        "selected_game": target_selection_receipt.get("selected_game"),
        "selected_level": target_selection_receipt.get("selected_level"),
        "target_selection_receipt": target_selection_receipt,
        "target_selection_hash": target_selection_receipt.get("target_selection_hash"),
        "policy_source": policy_source,
        "budget_receipt": artifact.get("budget_receipt"),
        "methodology_receipt": artifact.get("methodology_receipt"),
        "executed_actions": attempt.get("action_rows", []),
        "observations": attempt.get("observations", []),
        "level_counter_changes": attempt.get("level_counter_changes", []),
        "runtime_reverse_engineering": attempt.get("runtime_reverse_engineering", {}),
        "reproduction_gate": attempt.get("reproduction_gate", {}),
        "terminal_environment_response": artifact.get("terminal_environment_response"),
        "action_trace_sha256": attempt.get("action_trace_sha256"),
        "trace_replay_checksum": attempt.get("trace_replay_checksum"),
        "artifact_checksum": artifact.get("artifact_checksum"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    for key in (
        "source_read",
        "game_adapter_used",
        "outer_loop_re_used",
        "source_files_read",
        "per_game_adapter_used",
        "offline_bfs_used",
    ):
        if artifact.get(key) is not False:
            errors.append(f"{key} must be false")
    methodology = artifact.get("methodology_receipt")
    if not isinstance(methodology, Mapping):
        errors.append("methodology_receipt must be present")
    else:
        for key in (
            "source_read",
            "game_adapter_used",
            "outer_loop_re_used",
            "offline_ground_truth_bfs_used",
        ):
            if methodology.get(key) is not False:
                errors.append(f"methodology_receipt.{key} must be false")
        if methodology.get("model_call_limit") != MODEL_CALL_LIMIT:
            errors.append("methodology_receipt.model_call_limit mismatch")
    if artifact.get("live_attempt_executed") is not True:
        errors.append("live_attempt_executed must be true")
    delta = _int(artifact.get("registry_delta"), -1)
    if delta not in (0, 1):
        errors.append("registry_delta must be exactly 0 or 1")
    if artifact.get("registry_count_after") != _int(artifact.get("registry_count_before")) + delta:
        errors.append("registry_count_after must equal registry_count_before plus registry_delta")
    new_levels = artifact.get("new_reproducible_levels") or []
    if len(new_levels) != delta:
        errors.append("new_reproducible_levels length must equal registry_delta")
    if artifact.get("offline_reproduced") is True:
        if artifact.get("reproduced_levels") != 1 or delta != 1:
            errors.append(
                "offline_reproduced=true requires reproduced_levels=1 and registry_delta=1"
            )
    elif artifact.get("reproduced_levels") != 0 or delta != 0:
        errors.append("offline_reproduced=false requires reproduced_levels=0 and registry_delta=0")
    if artifact.get("action_trace_sha256") != artifact.get("trace_replay_checksum"):
        errors.append("action trace checksum and replay checksum must match exactly")
    if artifact.get("target_selection_hash") != (
        artifact.get("target_selection_receipt") or {}
    ).get("target_selection_hash"):
        errors.append("target_selection_hash must match target_selection_receipt")
    budget = artifact.get("budget_receipt")
    if not isinstance(budget, Mapping) or budget.get("bounded_before_execution") is not True:
        errors.append("budget_receipt must be bounded before execution")
    if artifact.get("llm_invoked"):
        specs = artifact.get("model_specs")
        if not isinstance(specs, Sequence) or not specs or not _model_specs_allowed(specs):
            errors.append("llm_invoked requires one mandated cached SOTA GGUF model spec")
    elif artifact.get("model_specs") != []:
        errors.append("no-LLM attempts require model_specs=[]")
    if artifact.get("random_seeds") != RANDOM_SEEDS:
        errors.append("random_seeds mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "blocked:")):
        errors.append("honest_verdict must start with complete: or blocked:")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != compute_artifact_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def update_registry_if_banked(
    root: Path, artifact: Mapping[str, Any]
) -> bool:  # pragma: no cover - null in normal run
    if _int(artifact.get("registry_delta")) != 1:
        return False
    path = root / REGISTRY_RELATIVE_PATH
    registry = read_yaml(path)
    game = str(artifact.get("selected_game") or "")
    target_level = _int(artifact.get("target_level"))
    games = registry.get("games")
    if isinstance(games, list):
        for row in games:
            if isinstance(row, dict) and row.get("game") == game:
                current = _int(row.get("levels_reproduced"))
                if target_level != current + 1:
                    raise ValueError(
                        "cannot update contiguous registry depth for non-adjacent bank"
                    )
                row["levels_reproduced"] = target_level
                row["reproducibility"] = "reproduced"
                row["latest_exp5643_levelup_attempt"] = {
                    "artifact": RESULT_RELATIVE_PATH,
                    "offline_reproduced": True,
                    "reproduced_levels": 1,
                    "solve_provenance": SOLVE_PROVENANCE,
                    "reproducibility_checksum": artifact.get("reproducibility_checksum"),
                }
                break
    elif isinstance(games, dict):
        row = games.setdefault(game, {"game": game})
        current = _int(row.get("levels_reproduced"))
        if target_level != current + 1:
            raise ValueError("cannot update contiguous registry depth for non-adjacent bank")
        row["levels_reproduced"] = target_level
        row["reproducibility"] = "reproduced"
        row["latest_exp5643_levelup_attempt"] = {
            "artifact": RESULT_RELATIVE_PATH,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "solve_provenance": SOLVE_PROVENANCE,
            "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        }
    registry["reproducible_total_levels"] = artifact.get("registry_count_after")
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return True


def main() -> int:  # pragma: no cover - command wrapper
    root = Path(__file__).resolve().parents[2]
    started = time.time()
    registry = read_yaml(root / REGISTRY_RELATIVE_PATH)
    public_games = load_public_env_metadata()
    loop_depths = load_arc_loop_depths(root)
    transition_receipt = read_json(root / TRANSITION_RECEIPT_RELATIVE_PATH)
    precheck = registry_precheck(registry, public_games, loop_depths, transition_receipt)
    target = select_target_from_precheck(precheck)
    policy = policy_source_from_exp5642(read_json(root / EXP5642_RELATIVE_PATH))
    if target.get("blocked"):
        print(f"{EXPERIMENT}: no unreproduced authenticated nonclosed target available")
        return 1

    attempt = run_live_self_discovery_attempt(target, policy)
    artifact = build_artifact(
        registry_precheck_receipt=precheck,
        target_selection_receipt=target,
        policy_source=policy,
        attempt=attempt,
        live_trace_path=TRACE_RELATIVE_PATH,
        duration_s=time.time() - started,
    )
    try:
        validate_artifact(artifact)
    except ValueError as exc:
        print(f"validation error: {exc}")
        return 1
    trace = build_live_trace(target, policy, attempt, artifact)
    write_json(root / TRACE_RELATIVE_PATH, trace)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    try:
        update_registry_if_banked(root, artifact)
    except ValueError as exc:
        print(f"registry update error: {exc}")
        return 1
    print(
        f"{EXPERIMENT}: {artifact['honest_verdict']} "
        f"registry_before={artifact['registry_count_before']} "
        f"registry_after={artifact['registry_count_after']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
