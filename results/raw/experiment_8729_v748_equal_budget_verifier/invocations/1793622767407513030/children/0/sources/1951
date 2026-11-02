"""Exp5713 ARC live self-discovery level-up attempt builder.

The V510 attempt is deliberately conservative. It binds target selection to the
registry bytes read immediately before runtime, treats Exp5712 as advisory, runs
the no-LLM live E3 baseline unless every promotion gate is satisfied, and only
banks a level after generic clean-state reproduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot import experiment_5610_arc_live_self_discovery_levelup_v506 as live_base


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5713
EXPERIMENT = "experiment_5713_arc_live_self_discovery_levelup_v510"
MILESTONE = "2026.07.510"
RESULT_RELATIVE_PATH = f"results/{EXPERIMENT}.json"
TRACE_RELATIVE_PATH = f"results/{EXPERIMENT}_trace.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
EXP5712_RELATIVE_PATH = "results/experiment_5712_arc_relational_goal_energy_live_ab.json"
SCHEMA = "arc_live_self_discovery_levelup_attempt.v510"
SPEC_REQUIREMENT = "REQ-ARC-WMTE-5713"

SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_own_attempts_no_llm"
AGENT_ENTRYPOINT = "carnot.agentic.arc_competition_agent.E3AgentPolicy"
ACTION_BUDGET = 72
WALL_TIME_CAP_S = 900
MODEL_CALL_LIMIT = 0
RANDOM_SEEDS = [20260715, 5713]
STOPPING_RULE = "fixed_72_action_budget_or_target_level_reached_no_llm_v510"

RECENT_FAILED_TARGETS = (
    {"game": "lf52", "level": 7, "level_label": "L7", "reason": "exp5632_no_bank"},
    {"game": "lf52", "level": 8, "level_label": "L8", "reason": "exp5643_no_bank"},
    {"game": "bp35", "level": 9, "level_label": "L9", "reason": "exp5621_no_bank"},
    {"game": "sk48", "level": 8, "level_label": "L8", "reason": "exp5610_no_bank"},
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "field_principles": {
        "principle": "principle annotations are carried in the artifact so every required 5713 field is auditable.",
    },
    "registry_precheck": {
        "principle": "selection binds to live registry state immediately before interaction.",
    },
    "registry_hash_before": {
        "principle": "the selected target is tied to the exact registry bytes read before the run.",
    },
    "target_selection_candidates": {
        "principle": "eligible and excluded rotations are auditable.",
    },
    "target_exclusions": {
        "principle": "duplicate, known-solve, recent-failure, and current-live-reach exclusions are explicit.",
    },
    "selected_game": {
        "principle": "the selected game is frozen before interaction.",
    },
    "selected_level": {
        "principle": "the selected level is frozen before interaction.",
    },
    "target_frozen_before_interaction": {
        "principle": "target, budget, seeds, and mechanism are frozen before any environment step.",
    },
    "solve_provenance": {
        "principle": "live_agent_self_discovery -- credited path is the agent's own runtime attempt.",
    },
    "agent_entrypoint": {
        "principle": "standard live entrypoint is named so reachability is explicit.",
    },
    "live_path_receipt": {
        "principle": "runtime counters prove the selected mechanism emitted environment actions.",
    },
    "mechanism_selection": {
        "principle": "Exp5712 is advisory and cannot silently replace the baseline.",
    },
    "exp5712_advisory_receipt": {
        "principle": "prototype use is honest and blocked unless ready, safe, and target-local.",
    },
    "model_specs": {
        "principle": "empty list is required when no LLM is used.",
    },
    "llm_used": {
        "principle": "false means no legacy model participated.",
    },
    "environment_action_budget": {
        "principle": "duration is bounded before execution.",
    },
    "environment_actions_used": {
        "principle": "actual environment action count is reported.",
    },
    "wall_time_seconds": {
        "principle": "wall duration is complete and not inferred.",
    },
    "termination_reason": {
        "principle": "the stopping condition is explicit.",
    },
    "trajectory_path": {
        "principle": "lossless trajectory evidence is durable.",
    },
    "trajectory_hash": {
        "principle": "trajectory content is replay-bound.",
    },
    "agent_visible_observation_count": {
        "principle": "visible evidence volume is explicit.",
    },
    "action_count": {
        "principle": "agent-emitted actions are explicit.",
    },
    "level_transition_events": {
        "principle": "level changes are auditable separately from solve credit.",
    },
    "new_level_candidate": {
        "principle": "candidate level-up evidence is not accepted as banked credit without reproduction.",
    },
    "reproduced_levels": {
        "principle": "bankable progress requires generic reproduction.",
    },
    "offline_reproduced": {
        "principle": "true only after independent clean-state generic reproduction.",
    },
    "independent_reproduction_pass": {
        "principle": "solve credit requires independent reproduction, not the original runtime trace alone.",
    },
    "reproduction_receipts": {
        "principle": "reproduction evidence is auditable or empty on a null.",
    },
    "reproduction_seed_count": {
        "principle": "deterministic or preregistered multi-seed reproduction is explicit.",
    },
    "registry_count_before": {
        "principle": "registry delta starts from the prechecked count.",
    },
    "registry_count_after": {
        "principle": "registry count changes only after accepted reproduction.",
    },
    "registry_delta": {
        "principle": "exact banked delta is auditable.",
    },
    "registry_updated": {
        "principle": "false on nulls; true only after generic reproduction.",
    },
    "game_source_read_count": {
        "principle": "zero excludes hidden source inspection.",
    },
    "game_adapter_count": {
        "principle": "zero excludes per-game adapters.",
    },
    "outer_loop_bfs_used": {
        "principle": "false excludes off-path exhaustive solving.",
    },
    "hand_solution_used": {
        "principle": "false excludes manual solve injection.",
    },
    "critical_flags": {
        "principle": "methodology blockers prevent solve credit.",
    },
    "inference_substrate": {
        "principle": "arc_live_agent_own_attempts_no_llm declares the credited substrate.",
    },
    "random_seeds": {
        "principle": "run replay seeds are explicit.",
    },
    "reproducibility_checksum": {
        "principle": "target, trajectory, budget, and banking decision are content-addressed.",
    },
    "honest_verdict": {
        "principle": "terminal complete: or blocked: verdict accepts null outcomes.",
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_hash_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(list(rows))


def _level_number(value: Any) -> int:
    if isinstance(value, str):
        return _int(value.strip().lstrip("Ll"))
    return _int(value)


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return live_base._registry_rows(registry)


def _registry_depth(row: Mapping[str, Any] | None) -> int:
    return live_base._registry_depth(row)


def _registry_total(registry: Mapping[str, Any]) -> int:
    return live_base._registry_total(registry)


def _public_game_items(
    public_games: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    return live_base._public_game_items(public_games)


def _headroom(meta: Mapping[str, Any]) -> int:
    return live_base._headroom(meta)


def _recent_failure_pairs(
    recent_failed_targets: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    pairs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in recent_failed_targets:
        game = str(row.get("game") or "")
        level = _level_number(row.get("level") or row.get("level_label"))
        if game and level > 0:
            pairs.setdefault((game, level), []).append(dict(row))
    return pairs


def _registry_duplicate_levels(
    registry_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for game, row in sorted(registry_rows.items()):
        depth = _registry_depth(row)
        if depth > 0:
            out.append(
                {
                    "game": game,
                    "closed_levels": f"L1-L{depth}" if depth > 1 else "L1",
                    "count": depth,
                    "source": REGISTRY_RELATIVE_PATH,
                }
            )
    return out


def registry_precheck(
    registry: Mapping[str, Any],
    public_envs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    arc_loop_depths: Mapping[str, int] | None = None,
    *,
    registry_hash_before: str | None = None,
    recent_failed_targets: Sequence[Mapping[str, Any]] = RECENT_FAILED_TARGETS,
) -> dict[str, Any]:
    """Build the pre-interaction target roster for REQ-ARC-WMTE-5713."""

    registry_rows = _registry_rows(registry)
    loop_depths = {str(game): _int(depth) for game, depth in (arc_loop_depths or {}).items()}
    recent_pairs = _recent_failure_pairs(recent_failed_targets)
    candidate_rows: list[dict[str, Any]] = []

    for game, meta in _public_game_items(public_envs):
        registry_depth = _registry_depth(registry_rows.get(game))
        current_live_depth = loop_depths.get(game, 0)
        authenticated_headroom = _headroom(meta)
        unreproduced = list(range(registry_depth + 1, authenticated_headroom + 1))
        closed_level_reasons: dict[str, list[str]] = {}
        open_levels: list[int] = []
        for level in unreproduced:
            reasons: list[str] = []
            if (game, level) in recent_pairs:
                reasons.append("explicit_recent_failed_target")
            if level <= current_live_depth:
                reasons.append("current_live_mechanism_already_reaches")
            if reasons:
                closed_level_reasons[str(level)] = sorted(set(reasons))
            else:
                open_levels.append(level)

        if open_levels:
            target_level = open_levels[0]
            exclude_reasons: list[str] = []
        elif unreproduced:
            target_level = unreproduced[0]
            exclude_reasons = ["all_authenticated_headroom_targets_closed"]
        else:
            target_level = registry_depth + 1
            exclude_reasons = ["no_authenticated_headroom"]

        candidate_rows.append(
            {
                "game": game,
                "registry_depth": registry_depth,
                "current_live_depth": current_live_depth,
                "authenticated_headroom": authenticated_headroom,
                "unreproduced_levels_with_headroom": unreproduced,
                "closed_level_reasons": closed_level_reasons,
                "closed_intermediate_levels": [
                    level
                    for level in unreproduced
                    if level < target_level and str(level) in closed_level_reasons
                ],
                "target_level": target_level,
                "target_label": f"L{target_level}",
                "excluded": bool(exclude_reasons),
                "exclude_reasons": exclude_reasons,
                "target_unreproduced_in_registry": target_level > registry_depth,
            }
        )

    eligible = [row for row in candidate_rows if not row["excluded"]]
    return {
        "spec_ref": SPEC_REQUIREMENT,
        "source": REGISTRY_RELATIVE_PATH,
        "checked_immediately_before_selection": True,
        "registry_hash_before": registry_hash_before or _sha256(registry),
        "registry_count_before": _registry_total(registry),
        "registry_total_from_file": registry.get("reproducible_total_levels"),
        "registry_total_from_rows": sum(_registry_depth(row) for row in registry_rows.values()),
        "registry_games_checked": len(registry_rows),
        "public_games_checked": len(candidate_rows),
        "arc_loop_targets_considered": len(loop_depths),
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
        "target_exclusions": {
            "explicit_recent_failed_targets": [dict(row) for row in recent_failed_targets],
            "registry_duplicate_levels": _registry_duplicate_levels(registry_rows),
            "current_live_mechanism_reaches": [
                {"game": game, "reaches_level": depth}
                for game, depth in sorted(loop_depths.items())
                if depth > 0
            ],
        },
        "ranking_rule": "authenticated_headroom_desc_then_target_level_then_game",
        "ok": bool(eligible),
    }


def _target_hash(precheck: Mapping[str, Any], selected: Mapping[str, Any] | None) -> str:
    return _sha256(
        {
            "spec_ref": SPEC_REQUIREMENT,
            "registry_hash_before": precheck.get("registry_hash_before"),
            "candidate_rows": precheck.get("candidate_rows"),
            "target_exclusions": precheck.get("target_exclusions"),
            "selected": dict(selected or {}),
        }
    )


def select_target_from_precheck(precheck: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in precheck.get("candidate_rows", [])
        if isinstance(row, Mapping) and not row.get("excluded")
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            -_int(row.get("authenticated_headroom")),
            _int(row.get("target_level")),
            str(row.get("game")),
        ),
    )
    if not rows:
        receipt = {
            "blocked": True,
            "selected_game": None,
            "selected_level": None,
            "target_level": None,
            "target_frozen_before_interaction": True,
            "selection_reason": "no_eligible_registry_prechecked_target_v510",
            "rotation_order": [],
            "duplicate_targets_rejected": [
                row
                for row in precheck.get("candidate_rows", [])
                if isinstance(row, Mapping) and row.get("excluded")
            ],
        }
        receipt["target_selection_hash"] = _target_hash(precheck, receipt)
        return receipt

    selected = rows[0]
    receipt = {
        "blocked": False,
        "selected_game": selected["game"],
        "selected_level": selected["target_label"],
        "target_level": selected["target_level"],
        "prior_levels_reproduced": selected["registry_depth"],
        "authenticated_headroom": selected["authenticated_headroom"],
        "current_live_depth": selected["current_live_depth"],
        "closed_intermediate_levels": selected.get("closed_intermediate_levels", []),
        "target_frozen_before_interaction": True,
        "registry_count_before": precheck.get("registry_count_before"),
        "registry_hash_before": precheck.get("registry_hash_before"),
        "selection_reason": "highest_authenticated_headroom_after_duplicate_and_recent_failure_exclusions_v510",
        "rotation_order": [
            {
                "game": row["game"],
                "target_level": row["target_level"],
                "target_label": row["target_label"],
                "authenticated_headroom": row["authenticated_headroom"],
            }
            for row in rows
        ],
        "duplicate_targets_rejected": [
            row
            for row in precheck.get("candidate_rows", [])
            if isinstance(row, Mapping) and row.get("excluded")
        ],
        "target_snapshot": {
            "game": selected["game"],
            "level": selected["target_label"],
            "budget": ACTION_BUDGET,
            "random_seeds": list(RANDOM_SEEDS),
        },
    }
    receipt["target_selection_hash"] = _target_hash(precheck, receipt)
    return receipt


def mechanism_selection_from_exp5712(
    exp5712: Mapping[str, Any] | None,
    *,
    target_hypothesis_induced_from_this_run: bool = False,
) -> dict[str, Any]:
    payload = exp5712 or {}
    ready_score = _float(payload.get("relational_live_ab_ready_score"))
    unsafe_count = _int(payload.get("unsafe_route_accept_count"))
    regression_count = _int(payload.get("level_regression_count"))
    clean = ready_score >= 1.0 and unsafe_count == 0 and regression_count == 0
    enabled = bool(clean and target_hypothesis_induced_from_this_run)
    reason = (
        "exp5712_ready_safe_and_target_local"
        if enabled
        else "exp5712_not_enabled_ready_score_or_target_locality_failed_baseline_unchanged"
    )
    advisory = {
        "source_artifact": EXP5712_RELATIVE_PATH,
        "honest_verdict": payload.get("honest_verdict"),
        "relational_live_ab_ready_score": ready_score,
        "unsafe_route_accept_count": unsafe_count,
        "level_regression_count": regression_count,
        "target_hypothesis_induced_from_this_run": bool(target_hypothesis_induced_from_this_run),
        "enabled": enabled,
        "reason": reason,
    }
    mechanism = {
        "policy_name": "exp5712_relational_goal_energy_route"
        if enabled
        else "unchanged_no_new_llm_e3_baseline",
        "agent_entrypoint": AGENT_ENTRYPOINT,
        "enabled_exp5712": enabled,
        "baseline_unchanged": not enabled,
        "llm_required": False,
        "model_call_limit": MODEL_CALL_LIMIT,
        "reason": reason,
    }
    return {
        "mechanism_selection": mechanism,
        "exp5712_advisory_receipt": advisory,
    }


def _baseline_filter_configuration() -> dict[str, Any]:
    return {
        "source_artifact": EXP5712_RELATIVE_PATH,
        "enabled_filters": [],
        "baseline_unchanged": True,
        "reason": "v510_unchanged_no_llm_baseline",
    }


def run_live_self_discovery_attempt(
    target_selection_receipt: Mapping[str, Any],
    mechanism_selection: Mapping[str, Any],
    action_budget: int = ACTION_BUDGET,
    random_seed: int = RANDOM_SEEDS[0],
) -> dict[str, Any]:  # pragma: no cover - ARC SDK/live environment boundary
    if mechanism_selection.get("llm_required"):
        return {
            "live_attempt_executed": False,
            "action_budget": action_budget,
            "random_seed": random_seed,
            "random_seeds": list(RANDOM_SEEDS),
            "action_rows": [],
            "observations": [],
            "level_counter_changes": [],
            "terminal_reason": "blocked_llm_required_but_no_llm_allowed",
            "llm_invoked": False,
            "model_specs": [],
            "source_files_read": False,
            "per_game_adapter_used": False,
            "offline_bfs_used": False,
            "hand_solution_used": False,
            "offline_reproduced": False,
            "reproduction_gate": {"attempted": False, "reproduced": False},
        }
    attempt = live_base.run_live_self_discovery_attempt(
        target_selection_receipt=target_selection_receipt,
        filter_configuration=_baseline_filter_configuration(),
        action_budget=action_budget,
        random_seed=random_seed,
    )
    attempt["random_seed"] = random_seed
    attempt["random_seeds"] = list(RANDOM_SEEDS)
    attempt["stopping_rule"] = STOPPING_RULE
    attempt["model_specs"] = (
        [] if not attempt.get("llm_invoked") else attempt.get("model_specs", [])
    )
    attempt["hand_solution_used"] = False
    return attempt


def _action_rows(attempt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in attempt.get("action_rows", []) if isinstance(row, Mapping)]


def _environment_action_rows(attempt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in _action_rows(attempt)
        if row.get("kind") == "ACTION" or row.get("action") is not None
    ]


def _independent_reproduction_pass(
    target_selection_receipt: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> bool:
    target = _int(target_selection_receipt.get("target_level"))
    post = _int(attempt.get("post_levels_reproduced"))
    gate = attempt.get("reproduction_gate")
    gate_reproduced = isinstance(gate, Mapping) and gate.get("reproduced") is True
    clean_provenance = not any(
        bool(attempt.get(key))
        for key in (
            "source_read",
            "source_files_read",
            "game_adapter_used",
            "per_game_adapter_used",
            "outer_loop_bfs_used",
            "offline_bfs_used",
            "hand_solution_used",
        )
    )
    return bool(
        attempt.get("offline_reproduced")
        and gate_reproduced
        and post >= target
        and target > _int(target_selection_receipt.get("prior_levels_reproduced"))
        and attempt.get("action_trace_sha256") == attempt.get("trace_replay_checksum")
        and clean_provenance
    )


def _critical_flags(attempt: Mapping[str, Any], llm_used: bool) -> list[str]:
    flags: list[str] = []
    if llm_used:
        flags.append("llm_used")
    if bool(attempt.get("source_read") or attempt.get("source_files_read")):
        flags.append("game_source_read")
    if bool(attempt.get("game_adapter_used") or attempt.get("per_game_adapter_used")):
        flags.append("game_adapter_used")
    if bool(attempt.get("outer_loop_bfs_used") or attempt.get("offline_bfs_used")):
        flags.append("outer_loop_bfs_used")
    if bool(attempt.get("hand_solution_used")):
        flags.append("hand_solution_used")
    return flags


def _live_path_receipt(
    attempt: Mapping[str, Any],
    mechanism_selection: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _action_rows(attempt)
    action_rows = _environment_action_rows(attempt)
    reset_rows = [row for row in rows if row.get("kind") == "RESET"]
    return {
        "agent_entrypoint": AGENT_ENTRYPOINT,
        "scored_mechanism": mechanism_selection.get("policy_name"),
        "e3_policy_next_move_calls": len(rows),
        "environment_reset_calls": len(reset_rows),
        "environment_step_calls": len(action_rows),
        "baseline_action_emissions": len(action_rows)
        if mechanism_selection.get("baseline_unchanged")
        else 0,
        "route_activation_count": len(attempt.get("route_activations", []) or []),
        "candidate_energy_receipt_count": len(attempt.get("candidate_energy_receipts", []) or []),
        "llm_model_calls": 0,
    }


def _reproduction_receipts(
    target_selection_receipt: Mapping[str, Any],
    attempt: Mapping[str, Any],
    independent_pass: bool,
) -> list[dict[str, Any]]:
    if not independent_pass:
        return []
    gate = dict(attempt.get("reproduction_gate") or {})
    gate.setdefault("selected_game", target_selection_receipt.get("selected_game"))
    gate.setdefault("selected_level", target_selection_receipt.get("selected_level"))
    gate.setdefault("reproduction_type", "generic_live_path_clean_state_reproduction")
    gate.setdefault("solution_labels", list(attempt.get("solution_labels") or []))
    return [gate]


def compute_artifact_checksum(artifact: Mapping[str, Any]) -> str:
    core = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256(core)


def build_artifact(
    *,
    registry_precheck: Mapping[str, Any],
    target_selection_receipt: Mapping[str, Any],
    mechanism_selection: Mapping[str, Any],
    exp5712_advisory_receipt: Mapping[str, Any],
    attempt: Mapping[str, Any],
    trajectory_path: str = TRACE_RELATIVE_PATH,
    wall_time_seconds: float = 0.0,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    action_rows = _environment_action_rows(attempt)
    observations = list(attempt.get("observations", []) or [])
    transitions = list(attempt.get("level_counter_changes", []) or [])
    trajectory_hash = str(
        attempt.get("trace_replay_checksum")
        or attempt.get("action_trace_sha256")
        or trajectory_hash_from_rows(_action_rows(attempt))
    )
    target_level = _int(target_selection_receipt.get("target_level"))
    level_reached = _int(attempt.get("max_level_reached"))
    candidate = level_reached >= target_level and target_level > 0
    llm_used = bool(attempt.get("llm_invoked", False))
    independent_pass = _independent_reproduction_pass(target_selection_receipt, attempt)
    reproduced_levels = 1 if independent_pass else 0
    registry_count_before = _int(registry_precheck.get("registry_count_before"))
    registry_count_after = registry_count_before + reproduced_levels
    selected_game = target_selection_receipt.get("selected_game")
    selected_level = target_selection_receipt.get("selected_level")
    verdict = (
        f"complete: banked_{selected_game}_{selected_level}_live_self_discovery_v510"
        if independent_pass
        else f"complete: no_new_arc_level_banked_{selected_game}_{selected_level}_bounded_live_attempt_v510"
    )
    critical_flags = _critical_flags(attempt, llm_used)
    artifact: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": SCHEMA,
        "spec_refs": [SPEC_REQUIREMENT],
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "registry_precheck": dict(registry_precheck),
        "registry_hash_before": registry_precheck.get("registry_hash_before"),
        "target_selection_candidates": list(registry_precheck.get("candidate_rows", []) or []),
        "target_exclusions": dict(registry_precheck.get("target_exclusions", {})),
        "selected_game": selected_game,
        "selected_level": selected_level,
        "target_level": target_level,
        "target_selection_receipt": dict(target_selection_receipt),
        "target_selection_hash": target_selection_receipt.get("target_selection_hash"),
        "target_snapshot": dict(target_selection_receipt.get("target_snapshot") or {}),
        "target_frozen_before_interaction": bool(
            target_selection_receipt.get("target_frozen_before_interaction")
        ),
        "solve_provenance": SOLVE_PROVENANCE,
        "agent_entrypoint": AGENT_ENTRYPOINT,
        "live_path_receipt": _live_path_receipt(attempt, mechanism_selection),
        "mechanism_selection": dict(mechanism_selection),
        "exp5712_advisory_receipt": dict(exp5712_advisory_receipt),
        "model_specs": list(attempt.get("model_specs") or []),
        "llm_used": llm_used,
        "environment_action_budget": _int(attempt.get("action_budget"), ACTION_BUDGET),
        "environment_actions_used": len(action_rows),
        "wall_time_seconds": round(float(wall_time_seconds), 3),
        "termination_reason": attempt.get("terminal_reason"),
        "trajectory_path": trajectory_path,
        "trajectory_hash": trajectory_hash,
        "agent_visible_observation_count": len(observations),
        "action_count": len(action_rows),
        "level_transition_events": transitions,
        "new_level_candidate": {
            "candidate": bool(candidate),
            "selected_game": selected_game,
            "selected_level": selected_level,
            "level_reached": level_reached,
            "requires_independent_reproduction": True,
        },
        "reproduced_levels": reproduced_levels,
        "offline_reproduced": bool(independent_pass),
        "independent_reproduction_pass": bool(independent_pass),
        "reproduction_receipts": _reproduction_receipts(
            target_selection_receipt, attempt, independent_pass
        ),
        "reproduction_seed_count": _int(attempt.get("reproduction_seed_count"), 1)
        if independent_pass
        else 0,
        "registry_count_before": registry_count_before,
        "registry_count_after": registry_count_after,
        "registry_delta": reproduced_levels,
        "registry_updated": bool(independent_pass),
        "game_source_read_count": 1
        if bool(attempt.get("source_read") or attempt.get("source_files_read"))
        else 0,
        "game_adapter_count": 1
        if bool(attempt.get("game_adapter_used") or attempt.get("per_game_adapter_used"))
        else 0,
        "outer_loop_bfs_used": bool(
            attempt.get("outer_loop_bfs_used") or attempt.get("offline_bfs_used")
        ),
        "hand_solution_used": bool(attempt.get("hand_solution_used", False)),
        "critical_flags": critical_flags,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(attempt.get("random_seeds") or RANDOM_SEEDS),
        "honest_verdict": verdict,
        "tests_run": list(tests_run or []),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = compute_artifact_checksum(artifact)
    return artifact


def build_trajectory(
    *,
    target_selection_receipt: Mapping[str, Any],
    mechanism_selection: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": "arc_live_self_discovery_attempt_trace.v510",
        "spec_refs": [SPEC_REQUIREMENT],
        "target_snapshot": dict(target_selection_receipt.get("target_snapshot") or {}),
        "target_selection_receipt": dict(target_selection_receipt),
        "mechanism_selection": dict(mechanism_selection),
        "visible_observations": list(attempt.get("observations", []) or []),
        "actions": list(attempt.get("action_rows", []) or []),
        "rewards": list(attempt.get("rewards", []) or []),
        "level_transition_events": list(attempt.get("level_counter_changes", []) or []),
        "candidate_energy_receipts": list(attempt.get("candidate_energy_receipts", []) or []),
        "route_activations": list(attempt.get("route_activations", []) or []),
        "environment_actions": list(
            attempt.get("environment_actions", attempt.get("action_rows", [])) or []
        ),
        "wall_time_seconds": artifact.get("wall_time_seconds"),
        "termination_reason": artifact.get("termination_reason"),
        "reproduction_receipts": artifact.get("reproduction_receipts", []),
        "trajectory_hash": artifact.get("trajectory_hash"),
        "reproducibility_checksum": artifact.get("reproducibility_checksum"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("agent_entrypoint") != AGENT_ENTRYPOINT:
        errors.append("agent_entrypoint mismatch")
    if artifact.get("target_frozen_before_interaction") is not True:
        errors.append("target must be frozen before interaction")
    if artifact.get("llm_used") is not False or artifact.get("model_specs") != []:
        errors.append("no-LLM artifact requires llm_used=false and model_specs=[]")
    if artifact.get("game_source_read_count") != 0:
        errors.append("game_source_read_count must be zero")
    if artifact.get("game_adapter_count") != 0:
        errors.append("game_adapter_count must be zero")
    if artifact.get("outer_loop_bfs_used") is not False:
        errors.append("outer_loop_bfs_used must be false")
    if artifact.get("hand_solution_used") is not False:
        errors.append("hand_solution_used must be false")
    delta = _int(artifact.get("registry_delta"), -1)
    if delta < 0:
        errors.append("registry_delta must be non-negative")
    if artifact.get("registry_count_after") != _int(artifact.get("registry_count_before")) + delta:
        errors.append("registry_count_after must equal registry_count_before plus delta")
    if artifact.get("offline_reproduced") is True:
        if artifact.get("independent_reproduction_pass") is not True:
            errors.append("offline_reproduced requires independent_reproduction_pass")
        if _int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if not artifact.get("reproduction_receipts"):
            errors.append("offline_reproduced requires reproduction receipts")
    else:
        if artifact.get("independent_reproduction_pass") is not False:
            errors.append("null artifact requires independent_reproduction_pass=false")
        if artifact.get("reproduced_levels") != 0:
            errors.append("null artifact requires reproduced_levels=0")
        if artifact.get("registry_updated") is not False or delta != 0:
            errors.append("null artifact cannot update registry")
    if _int(artifact.get("environment_actions_used")) != _int(artifact.get("action_count")):
        errors.append("environment_actions_used and action_count must match")
    if _int(artifact.get("agent_visible_observation_count")) < _int(artifact.get("action_count")):
        errors.append("observation count must cover action count")
    if not str(artifact.get("trajectory_hash") or "").startswith("sha256:"):
        errors.append("trajectory_hash must be sha256")
    if artifact.get("target_selection_hash") != (
        artifact.get("target_selection_receipt") or {}
    ).get("target_selection_hash"):
        errors.append("target_selection_hash mismatch")
    if str(artifact.get("honest_verdict") or "").startswith(("complete:", "blocked:")) is False:
        errors.append("honest_verdict must start with complete: or blocked:")
    checksum = artifact.get("reproducibility_checksum")
    if checksum != compute_artifact_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def update_registry_if_banked(root: Path, artifact: Mapping[str, Any]) -> bool:  # pragma: no cover
    if _int(artifact.get("registry_delta")) <= 0:
        return False
    path = root / REGISTRY_RELATIVE_PATH
    registry = read_yaml(path)
    game = str(artifact.get("selected_game") or "")
    target = _int(artifact.get("target_level"))
    games = registry.get("games")
    if not isinstance(games, list):
        raise ValueError("registry games must be a list for Exp5713 update")
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            current = _int(row.get("levels_reproduced"))
            if target != current + 1:
                raise ValueError("refusing non-contiguous registry update")
            row["levels_reproduced"] = target
            row["reproducibility"] = "reproduced"
            row["latest_exp5713_levelup_attempt"] = {
                "artifact": RESULT_RELATIVE_PATH,
                "offline_reproduced": True,
                "reproduced_levels": artifact.get("reproduced_levels"),
                "solve_provenance": SOLVE_PROVENANCE,
                "reproducibility_checksum": artifact.get("reproducibility_checksum"),
            }
            registry["reproducible_total_levels"] = artifact.get("registry_count_after")
            path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
            return True
    raise ValueError(f"selected game missing from registry: {game}")


def main() -> int:  # pragma: no cover - command wrapper
    started = time.time()
    registry_path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    registry_hash = file_sha256(registry_path)
    registry = read_yaml(registry_path)
    public_envs = live_base.load_public_env_metadata()
    arc_loop_depths = live_base.load_arc_loop_depths(REPO_ROOT)
    precheck = registry_precheck(
        registry,
        public_envs,
        arc_loop_depths,
        registry_hash_before=registry_hash,
    )
    target = select_target_from_precheck(precheck)
    if target.get("blocked"):
        print(f"{EXPERIMENT}: blocked: no eligible target")
        return 1
    selection = mechanism_selection_from_exp5712(read_json(REPO_ROOT / EXP5712_RELATIVE_PATH))
    mechanism = selection["mechanism_selection"]
    advisory = selection["exp5712_advisory_receipt"]
    attempt = run_live_self_discovery_attempt(target, mechanism)
    artifact = build_artifact(
        registry_precheck=precheck,
        target_selection_receipt=target,
        mechanism_selection=mechanism,
        exp5712_advisory_receipt=advisory,
        attempt=attempt,
        trajectory_path=TRACE_RELATIVE_PATH,
        wall_time_seconds=time.time() - started,
    )
    validate_artifact(artifact)
    trajectory = build_trajectory(
        target_selection_receipt=target,
        mechanism_selection=mechanism,
        attempt=attempt,
        artifact=artifact,
    )
    write_json(REPO_ROOT / TRACE_RELATIVE_PATH, trajectory)
    write_json(REPO_ROOT / RESULT_RELATIVE_PATH, artifact)
    update_registry_if_banked(REPO_ROOT, artifact)
    print(
        f"{EXPERIMENT}: {artifact['honest_verdict']} "
        f"registry_before={artifact['registry_count_before']} "
        f"registry_after={artifact['registry_count_after']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
